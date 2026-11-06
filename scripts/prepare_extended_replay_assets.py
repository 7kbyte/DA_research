#!/usr/bin/env python3
"""Execute the separately frozen optional-model generation asset protocol."""
import json
import subprocess
import sys
from pathlib import Path


def run(command, log):
    path = Path(log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    protocol_path = Path("configs/extended_generation_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    for model in protocol["models"]:
        name = model["name"]
        if model["memory_smoke_required"]:
            output = Path(model["artifact_root"]) / "smoke/pool_seed909_batch128.pt"
            config = {"model": model["model"], "revision": model["revision"], "output": str(output),
                      "seed": model["memory_smoke_seed"], "n_chunks": 128, "sequence_length": 512,
                      "batch_size": 128, "device": protocol["device"], "dtype": "float32", "precision": "bf16",
                      "role": "optional_model_memory_smoke"}
            config_path = Path(model["config_root"]) / "memory_smoke_seed909.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            run([sys.executable, "scripts/generate_pool.py", "--config", str(config_path)], f"logs/extended_{name}_memory_smoke.jsonl")
            smoke = json.loads(output.with_suffix(".json").read_text())
            if smoke["peak_gpu_memory_bytes"] > model["maximum_smoke_allocated_bytes"]:
                raise RuntimeError("2.8B batch128 exceeds frozen memory safety gate; formal generation not started")
            model["memory_smoke_result"] = {"passed": True, "manifest": str(output.with_suffix(".json")),
                                            "peak_allocated_bytes": smoke["peak_gpu_memory_bytes"],
                                            "tokens_per_second": smoke["sampled_tokens_per_second"]}
            protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        command = [sys.executable, "scripts/prepare_replay_banks.py", "--model", model["model"],
                   "--revision", model["revision"], "--device", protocol["device"],
                   "--generation-batch-size", str(model["generation_batch_size"]),
                   "--artifact-root", model["artifact_root"], "--config-root", model["config_root"],
                   "--summary", model["summary"], "--c4-pool", protocol["c4_pool"]]
        run(command, f"logs/extended_{name}_prepare_replay_banks.jsonl")
        model["generation_complete"] = True
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    protocol["all_assets_complete"] = True
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
