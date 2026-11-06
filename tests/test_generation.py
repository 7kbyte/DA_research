"""Real HF sampling tests: EOS continuation, reproducibility, and resume."""
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.generate import GenerationConfig, generate_pool


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 0

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class GenerationTests(unittest.TestCase):
    def test_fixed_length_continues_after_eos_and_exact_resume(self):
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        config = GPTNeoXConfig(vocab_size=2, hidden_size=8, num_hidden_layers=1,
                              num_attention_heads=2, intermediate_size=16, max_position_embeddings=16,
                              bos_token_id=1, eos_token_id=0)
        model = GPTNeoXForCausalLM(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()  # exactly uniform sampling with frequent EOS
        original_generation_config = model.generation_config
        with tempfile.TemporaryDirectory() as directory:
            full_path = str(Path(directory) / "full.pt")
            interrupted_path = str(Path(directory) / "resumed.pt")
            common = dict(model="injected_tiny_model", revision="test-pinned", seed=11,
                          n_chunks=5, sequence_length=8, batch_size=2, device="cpu",
                          precision="fp32", save_every_batches=1)
            complete = generate_pool(GenerationConfig(output=full_path, **common), model, TinyTokenizer())

            def interrupt(event):
                if event["completed_chunks"] == 4:
                    raise RuntimeError("simulated interruption")

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                generate_pool(GenerationConfig(output=interrupted_path, **common), model,
                              TinyTokenizer(), progress=interrupt)
            self.assertTrue(Path(interrupted_path).with_suffix(".partial.pt").exists())
            resumed = generate_pool(GenerationConfig(output=interrupted_path, resume=True, **common),
                                    model, TinyTokenizer())
            full_ids = torch.load(full_path, weights_only=False)["input_ids"]
            resumed_ids = torch.load(interrupted_path, weights_only=False)["input_ids"]
            self.assertEqual(tuple(full_ids.shape), (5, 8))
            # EOS must be followed by a newly sampled non-EOS token. Padding after
            # EOS would fail this even when another row forced the full length.
            self.assertTrue(((full_ids[:, 1:-1] == 0) & (full_ids[:, 2:] == 1)).any())
            torch.testing.assert_close(full_ids, resumed_ids)
            self.assertEqual(complete["prediction_tokens"], 35)
            self.assertEqual(resumed["prediction_tokens"], 35)
            self.assertFalse(Path(interrupted_path).with_suffix(".partial.pt").exists())
            self.assertIs(model.generation_config, original_generation_config)
            self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
