"""Scientific invariants for frozen task likelihoods and paired forgetting."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.tasks import (choice_predictions, encode_pair, evaluate_task, format_example,
                                hellaswag_preprocess, prepare_choice, transition_summary)


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class UniformLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(256))
        self.observed = []

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        self.observed.append((self.training, torch.is_grad_enabled()))
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))


class BehavioralTests(unittest.TestCase):
    def test_piqa_prompt_and_order_are_literal(self):
        row = format_example("piqa", {"goal": "  Find water?", "sol1": "First.", "sol2": " Second.", "label": 1}, 5)
        self.assertEqual(row["context"], "Question:   Find water?\nAnswer:")
        self.assertEqual(row["choices"], ["First.", " Second."])
        self.assertEqual(row["gold"], 1)
        self.assertEqual(row["item_id"], "piqa:validation:5")

    def test_hellaswag_matches_pinned_official_preprocessor(self):
        path = Path(__file__).resolve().parents[1] / "artifacts/behavioral/sources/harness_lm_eval__tasks__hellaswag__utils.py"
        spec = importlib.util.spec_from_file_location("official_hellaswag_utils", path)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        for text in [" A [title] B  C ", " x [remove me] y ", "one    two", "[x][y] z", "é  Z"]:
            self.assertEqual(hellaswag_preprocess(text), official.preprocess(text))
        row = format_example("hellaswag", {"activity_label": "Walking", "ctx_a": "A man moves.",
            "ctx_b": "HE WALKS", "endings": [" a [x] b ", "next", "third", "last"], "label": "2"}, 0)
        self.assertEqual(row["context"], "Walking: A man moves. He walks")
        self.assertEqual(row["choices"][0], official.preprocess(" a [x] b "))
        self.assertEqual(row["gold"], 2)

    def test_trailing_whitespace_moves_into_scored_continuation(self):
        context, continuation, matches = encode_pair(ByteTokenizer(), "ab  ", " c")
        self.assertEqual(context, list(b"ab"))
        self.assertEqual(continuation, list(b"   c"))
        self.assertTrue(matches)
        context, continuation, _ = encode_pair(ByteTokenizer(), "", " x", prefix_token_id=0)
        self.assertEqual(context, [0])
        self.assertEqual(continuation, list(b" x"))

    def test_joint_encoding_not_independent_answer_encoding(self):
        class MergeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"ab": [1], "ab c": [1, 2], " c": [3, 4]}[text]
        context, continuation, matches = encode_pair(MergeTokenizer(), "ab", " c")
        self.assertEqual(context, [1])
        self.assertEqual(continuation, [2])
        self.assertTrue(matches)

    def test_context_truncation_preserves_full_answer_and_last_label(self):
        item = prepare_choice(ByteTokenizer(), "abcdef", "xy", max_length=4)
        self.assertEqual(item["input_ids"], list(b"ef x"))
        self.assertEqual(item["labels"], list(b"f xy"))
        self.assertEqual(item["continuation_start"], 1)
        self.assertEqual(item["continuation_tokens"], 3)
        self.assertEqual(item["context_tokens_removed"], 4)
        with self.assertRaisesRegex(ValueError, "no silent answer truncation"):
            prepare_choice(ByteTokenizer(), "ab", "abcd", max_length=4)

    def test_official_normalization_is_characters_and_first_index_ties(self):
        result = choice_predictions([-3., -4.], ["éé", "abc"], gold=0)
        self.assertEqual(result["prediction_acc"], 0)
        self.assertEqual(result["prediction_acc_norm"], 1)
        self.assertEqual(result["choice_character_lengths"], [2, 3])
        result = choice_predictions([-2., -2.], ["a", "b"], gold=0)
        self.assertEqual(result["prediction_acc"], 0)
        self.assertEqual(result["prediction_acc_norm"], 0)

    def test_uniform_model_scores_only_answers_not_prompt_or_padding(self):
        model = UniformLM().train()
        rows = [{"task": "piqa", "item_id": "a", "fingerprint": "a", "context": "a", "choices": ["x", "yy"], "gold": 0},
                {"task": "piqa", "item_id": "b", "fingerprint": "b", "context": "a much longer prompt", "choices": ["x", "yy"], "gold": 1}]
        metrics, predictions = evaluate_task(model, ByteTokenizer(), rows, batch_size=3, precision="fp32", max_length=8)
        for row in predictions:
            self.assertAlmostEqual(row["choice_logprobs"][0], -2 * math.log(256), places=5)
            self.assertAlmostEqual(row["choice_logprobs"][1], -3 * math.log(256), places=5)
            self.assertEqual(row["prediction_acc"], 0)
            self.assertEqual(row["prediction_acc_norm"], 1)
        self.assertEqual(metrics["acc"], .5)
        self.assertEqual(metrics["acc_norm"], .5)
        self.assertEqual(metrics["scored_continuation_tokens"], 10)
        self.assertEqual(metrics["choices_with_context_truncation"], 2)
        self.assertTrue(model.training)
        self.assertTrue(all(mode is False and grad is False for mode, grad in model.observed))
        self.assertIsNone(model.logits.grad)

    def test_nonuniform_causal_scores_predict_space_then_answer_token(self):
        class MarkovLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                source, target = torch.arange(256).unsqueeze(1), torch.arange(256).unsqueeze(0)
                self.table = torch.nn.Parameter(((source * 7 + target * 11) % 19).float() / 3)

            def forward(self, input_ids, attention_mask=None, use_cache=False):
                return SimpleNamespace(logits=self.table[input_ids])

        model = MarkovLM()
        rows = [{"task": "piqa", "item_id": "a", "fingerprint": "a", "context": "A", "choices": ["x", "y"], "gold": 0}]
        _, predictions = evaluate_task(model, ByteTokenizer(), rows, batch_size=2, precision="fp32")
        expected = [torch.log_softmax(model.table[ord("A")], -1)[ord(" ")]
                    + torch.log_softmax(model.table[ord(" ")], -1)[ord(choice)] for choice in ["x", "y"]]
        for actual, reference in zip(predictions[0]["choice_logprobs"], expected):
            self.assertAlmostEqual(actual, reference.item(), places=5)

    def test_zero_accuracy_change_can_mask_equal_losses_and_gains(self):
        def rows(values):
            return [{"task": "piqa", "item_id": str(i), "fingerprint": str(i), "gold": 0,
                     "correct_acc": value, "correct_acc_norm": value} for i, value in enumerate(values)]
        initial, adapted = rows([1, 1, 0, 0]), rows([1, 0, 1, 0])
        result = transition_summary(initial, list(reversed(adapted)))
        for metric in result["metrics"].values():
            self.assertEqual(metric["counts"], {"CC": 1, "CW": 1, "WC": 1, "WW": 1})
            self.assertEqual(metric["accuracy_delta"], 0.)
            self.assertEqual(metric["conditional_forgetting_rate"], .5)
            self.assertEqual(metric["conditional_acquisition_rate"], .5)
        with self.assertRaisesRegex(ValueError, "item sets"):
            transition_summary(initial, adapted[:-1])
        adapted[0]["fingerprint"] = "different prompt"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            transition_summary(initial, adapted)


if __name__ == "__main__":
    unittest.main()
