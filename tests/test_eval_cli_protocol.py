"""Exercise evaluator flags with a synthetic CPU LM and no GPU initialization."""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from replay_study.execution import verify_matching_execution


class UniformLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(3))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.value.expand(*input_ids.shape, -1))


class EvaluatorCLIProtocolTests(unittest.TestCase):
    def test_execution_flags_and_provenance_match_frozen_training_defaults(self):
        source = Path(__file__).resolve().parents[1] / "scripts/run_evaluate.py"
        spec = importlib.util.spec_from_file_location("run_evaluate_cli_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
                  torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
                  torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.matmul.allow_tf32,
                  torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        cudnn_sdp_before = torch.backends.cuda.cudnn_sdp_enabled() if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
                data = Path(directory) / "corpus.pt"
                torch.save({"input_ids": torch.tensor([[0, 1, 2]])}, data)
                for backend in ("auto", "flash", "math"):
                    with self.subTest(backend=backend):
                        output = Path(directory) / f"{backend}.json"
                        argv = [str(source), "--model", "injected_uniform", "--revision", "test",
                                "--corpus", f"c4={data}", "--output", str(output), "--device", "cpu",
                                "--teacher-kl", "--attention-backend", backend]
                        with patch.object(sys, "argv", argv), patch.object(module, "load_causal_model", return_value=UniformLM()), contextlib.redirect_stdout(io.StringIO()):
                            module.main()
                        result = json.loads(output.read_text())
                        settings = result["execution_provenance"]
                        self.assertTrue(settings["deterministic_algorithms"])
                        self.assertFalse(settings["deterministic_warn_only"])
                        self.assertEqual(settings["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                        self.assertEqual(settings["attention_backend_requested"], backend)
                        self.assertEqual(settings["flash_sdp_enabled"], backend in ("auto", "flash"))
                        self.assertEqual(settings["mem_efficient_sdp_enabled"], backend == "auto")
                        self.assertEqual(settings["math_sdp_enabled"], backend in ("auto", "math"))
                        self.assertEqual(settings["cudnn_sdp_enabled"], backend == "auto" if cudnn_sdp_before is not None else None)
                        self.assertTrue(settings["allow_tf32_matmul"])
                        self.assertTrue(settings["allow_tf32_cudnn"])
                        self.assertTrue(settings["cudnn_deterministic"])
                        self.assertFalse(settings["cudnn_benchmark"])
                        self.assertEqual(result["corpora"]["c4"]["kl"], 0.)
                        verify_matching_execution(settings, settings)
                        altered = {**settings, "cudnn_sdp_enabled": not settings["cudnn_sdp_enabled"]}
                        with self.assertRaisesRegex(ValueError, "cudnn_sdp_enabled"):
                            verify_matching_execution(settings, altered)
        finally:
            torch.use_deterministic_algorithms(before[0], warn_only=before[1])
            torch.backends.cuda.enable_flash_sdp(before[2])
            torch.backends.cuda.enable_mem_efficient_sdp(before[3])
            torch.backends.cuda.enable_math_sdp(before[4])
            torch.backends.cuda.matmul.allow_tf32 = before[5]
            torch.backends.cudnn.allow_tf32 = before[6]
            torch.backends.cudnn.deterministic = before[7]
            torch.backends.cudnn.benchmark = before[8]
            if cudnn_sdp_before is not None:
                torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp_before)


if __name__ == "__main__":
    unittest.main()
