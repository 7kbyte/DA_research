"""Scientific invariants: causal masks, exact KL, features, matched banks."""
import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.evaluate import (aggregate_document_metrics, evaluate_corpora, evaluate_model,
                                    exact_forward_kl, extract_teacher_statistics)
from replay_study.generate import generation_diagnostics
from replay_study.selection import save_bank, select_bank


class TableLM(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = torch.nn.Parameter(torch.as_tensor(table, dtype=torch.float32))
        self.observed_training = []
        self.observed_grad = []

    def forward(self, input_ids, attention_mask=None, use_cache=False, output_hidden_states=False):
        self.observed_training.append(self.training)
        self.observed_grad.append(torch.is_grad_enabled())
        # A hidden vector exposing both token ID and absolute position.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        hidden = torch.stack([input_ids, positions], dim=-1).float()
        return SimpleNamespace(logits=self.table[input_ids], hidden_states=(hidden,))


class EvaluationTests(unittest.TestCase):
    def test_masked_causal_nll_counts_and_mode(self):
        table = torch.tensor([[1., 0., -1.], [0., 2., 1.], [-1., 0., 3.]])
        model = TableLM(table).train()
        data = {"input_ids": torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]]),
                "doc_ids": ["a", "b"]}
        result = evaluate_model(model, data, batch_size=1, include_chunks=True, teacher=model)
        expected = sum(-F.log_softmax(table[source], dim=-1)[target].item()
                       for source, target in ((0, 1), (1, 2), (1, 0), (0, 2)))
        self.assertEqual(result["valid_prediction_tokens"], 4)
        self.assertEqual(result["raw_input_tokens"], 6)
        self.assertAlmostEqual(result["sum_nll"], expected, places=6)
        self.assertAlmostEqual(result["ppl"], math.exp(expected / 4), places=6)
        self.assertEqual(result["sum_kl"], 0.)
        self.assertEqual(result["teacher_forward_seconds"], 0.)
        self.assertTrue(result["teacher_logits_reused_from_student"])
        self.assertGreaterEqual(result["kl_arithmetic_seconds"], 0.)
        self.assertEqual([x["doc_id"] for x in result["chunks"]], ["a", "b"])
        self.assertTrue(model.training)
        self.assertFalse(any(model.observed_training))
        self.assertFalse(any(model.observed_grad))
        self.assertIsNone(model.table.grad)

    def test_exact_forward_kl_matches_reference_across_chunks(self):
        torch.manual_seed(30)
        teacher, student = torch.randn(2, 7, 13), torch.randn(2, 7, 13)
        expected = (teacher.softmax(-1) * (teacher.log_softmax(-1) - student.log_softmax(-1))).sum(-1)
        actual = exact_forward_kl(student, teacher, position_chunk_size=3, vocabulary_chunk_size=4)
        torch.testing.assert_close(actual, expected, atol=5e-7, rtol=2e-6)
        self.assertFalse(torch.allclose(actual, exact_forward_kl(teacher, student)))

    def test_document_metrics_sum_tokens_before_normalizing(self):
        rows = [{"doc_id": "a", "sum_nll": 4., "valid_prediction_tokens": 2, "sum_kl": 1.},
                {"doc_id": "a", "sum_nll": 12., "valid_prediction_tokens": 3, "sum_kl": 2.}]
        result = aggregate_document_metrics(rows)[0]
        self.assertEqual(result["chunk_count"], 2)
        self.assertEqual(result["nll"], 16 / 5)
        self.assertEqual(result["kl"], 3 / 5)

    def test_features_exclude_final_and_masked_contexts(self):
        model = TableLM(torch.zeros(5, 5))
        data = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]])}
        stats = extract_teacher_statistics(model, data)
        torch.testing.assert_close(stats["features"], torch.tensor([[1.5, .5]]))
        self.assertEqual(stats["prediction_token_counts"].tolist(), [2])
        self.assertAlmostEqual(stats["teacher_nll"].item(), math.log(5), places=6)

    def test_separate_corpora_and_partial_flag(self):
        model = TableLM(torch.zeros(3, 3))
        data = {"input_ids": torch.zeros(3, 3, dtype=torch.long)}
        result = evaluate_corpora(model, {"c4": data, "wikitext": data}, batch_size=1, max_batches=2)
        self.assertEqual(set(result), {"c4", "wikitext"})
        self.assertFalse(result["c4"]["complete_corpus"])
        self.assertEqual(result["c4"]["valid_prediction_tokens"], 4)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def pool(n=25):
        return {"input_ids": torch.arange(n * 5, dtype=torch.int32).reshape(n, 5),
                "doc_ids": [f"doc{i}" for i in range(n)]}

    def test_random_and_dedup_null_are_identical_and_budget_exact(self):
        pool = self.pool()
        random_bank, a = select_bank(pool, "random", 11, prediction_token_budget=40)
        dedup_bank, b = select_bank(pool, "dedup_random", 11, bank_size=10)
        torch.testing.assert_close(random_bank["input_ids"], dedup_bank["input_ids"])
        self.assertEqual(a["prediction_tokens"], 40)
        self.assertEqual(b["bank_duplicate_rate"], 0.)
        self.assertEqual(a["selected_indices"], b["selected_indices"])

    def test_dedup_removes_duplicate_content_and_preserves_doc_provenance(self):
        pool = self.pool(6)
        pool["input_ids"][5] = pool["input_ids"][0]
        bank, manifest = select_bank(pool, "dedup_random", 22, bank_size=5)
        self.assertEqual(manifest["pool_unique_chunks"], 5)
        self.assertAlmostEqual(manifest["pool_duplicate_rate"], 1 / 6)
        self.assertEqual(manifest["bank_unique_chunks"], 5)
        self.assertNotIn("doc5", bank["doc_ids"])

    def test_five_quantile_stratification_has_proportional_coverage(self):
        pool = self.pool()
        stats = {"teacher_nll": torch.arange(25).float()}
        bank, manifest = select_bank(pool, "nll_stratified", 33, bank_size=11, statistics=stats)
        self.assertEqual(manifest["stratum_bank_sizes"], [3, 2, 2, 2, 2])
        selected = bank["selection_indices"].numpy()
        self.assertEqual(np.bincount(selected // 5).tolist(), [3, 2, 2, 2, 2])
        self.assertEqual(manifest["prediction_tokens"], 44)

    def test_rejects_nonmatching_token_budgets_and_ragged_validity(self):
        pool = self.pool()
        with self.assertRaises(ValueError):
            select_bank(pool, "random", 11, prediction_token_budget=41)
        pool["attention_mask"] = torch.ones_like(pool["input_ids"])
        pool["attention_mask"][0, -1] = 0
        with self.assertRaises(ValueError):
            select_bank(pool, "random", 11, bank_size=3)

    def test_wrong_pool_statistics_and_heldout_replay_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "pool.pt"
            torch.save(self.pool(), source)
            with self.assertRaisesRegex(ValueError, "different candidate pool hash"):
                select_bank(source, "nll_stratified", 11, bank_size=5,
                            statistics={"teacher_nll": torch.arange(25).float(),
                                        "source_pool_sha256": "wrong_pool"})
            source.with_suffix(".json").write_text(json.dumps({"split": "validation"}))
            with self.assertRaisesRegex(ValueError, "held-out"):
                save_bank(source, Path(directory) / "bank.pt", method="random", seed=11, bank_size=5)

    def test_cluster_exact_budget_unique_even_collapsed_features(self):
        try:
            import sklearn
        except ImportError:
            self.skipTest("scikit-learn installation is pending")
        pool = self.pool()
        bank, manifest = select_bank(pool, "cluster", 11, bank_size=5,
                                     statistics={"features": torch.ones(25, 3)})
        self.assertEqual(len(set(bank["selection_indices"].tolist())), 5)
        self.assertEqual(manifest["prediction_tokens"], 20)
        self.assertEqual(len(manifest["empty_clusters"]), 4)

    def test_diagnostics_describe_without_changing_tokens(self):
        ids = torch.tensor([[0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
        before = ids.clone()
        result = generation_diagnostics(ids, eos_token_id=1)
        self.assertEqual(result["exact_duplicate_chunk_rate"], .5)
        self.assertEqual(result["sampled_eos_tokens"], 10)
        torch.testing.assert_close(ids, before)


if __name__ == "__main__":
    unittest.main()
