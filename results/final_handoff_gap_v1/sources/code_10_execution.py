"""Explicit evaluation CLI execution policy, applied only when requested by main."""
import argparse
import os

import torch


def add_execution_arguments(parser):
    parser.add_argument("--deterministic-algorithms", action=argparse.BooleanOptionalAction, default=True,
                        help="Strict deterministic algorithms, warn_only=False (default enabled)")
    parser.add_argument("--attention-backend", choices=("auto", "flash", "math"), default="auto")
    parser.add_argument("--allow-tf32-matmul", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-tf32-cudnn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cudnn-deterministic", action=argparse.BooleanOptionalAction, default=None,
                        help="Default follows strict deterministic-algorithms setting")
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=False)


def configure_execution(args):
    """Call before model/CUDA initialization; flags are process-local."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(args.deterministic_algorithms, warn_only=False)
    torch.backends.cuda.enable_flash_sdp(args.attention_backend in ("auto", "flash"))
    torch.backends.cuda.enable_mem_efficient_sdp(args.attention_backend == "auto")
    torch.backends.cuda.enable_math_sdp(args.attention_backend in ("auto", "math"))
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(args.attention_backend == "auto")
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32_matmul
    torch.backends.cudnn.allow_tf32 = args.allow_tf32_cudnn
    torch.backends.cudnn.deterministic = (args.deterministic_algorithms if args.cudnn_deterministic is None
                                         else args.cudnn_deterministic)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark


def execution_provenance(args, model=None):
    device = torch.device(args.device)
    result = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "precision": args.precision, "attention_implementation": "sdpa",
              "device_type": device.type,
              "effective_precision": "cuda_bfloat16_autocast" if device.type == "cuda" and args.precision == "bf16" else "float32",
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "attention_backend_requested": args.attention_backend,
              "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
              "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
              "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
              "cudnn_sdp_enabled": (torch.backends.cuda.cudnn_sdp_enabled()
                                    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
              "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if model is not None:
        result["model_weight_dtype"] = str(next(model.parameters()).dtype)
    return result


def verify_matching_execution(initial, current):
    fields = ("torch_version", "cuda_version", "precision", "effective_precision", "device_type", "gpu_name", "attention_implementation",
              "attention_backend_requested", "flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled",
              "deterministic_algorithms", "deterministic_warn_only", "allow_tf32_matmul", "allow_tf32_cudnn",
              "cudnn_deterministic", "cudnn_benchmark", "CUBLAS_WORKSPACE_CONFIG")
    mismatches = [field for field in fields if field not in initial or initial[field] != current[field]]
    if mismatches:
        raise ValueError("baseline/adapted execution policy differs: " + ", ".join(mismatches))
