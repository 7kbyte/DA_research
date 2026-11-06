#!/usr/bin/env python3
"""Optional larger final Pythia checkpoints; never starts model training."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import time

from download_assets import download_one

REVISIONS = {
    "pythia-1b": "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2",
    "pythia-2.8b": "2a259cdd96a4beb1cdf467512e3904197345f6a9",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch", type=Path, default=Path("/tmp/grad-research-20261106"))
    p.add_argument("--endpoint", default="https://hf-mirror.com")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    started = time.monotonic()
    plan = [{"repo": "EleutherAI/" + model, "revision": revision, "file": name,
             "destination": str(args.scratch / f"models/{model}-final" / name), "dataset": False}
            for model, revision in REVISIONS.items()
            for name in ["config.json", "tokenizer.json", "tokenizer_config.json",
                         "special_tokens_map.json", "README.md", "model.safetensors"]]
    path = Path("artifacts/assets/extended_model_download_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, asset, args.endpoint) for asset in plan]
        for future in as_completed(futures):
            records.append(future.result())
            path.write_text(json.dumps({"role": "optional_robustness_assets_only", "assets": records,
                "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")


if __name__ == "__main__":
    main()
