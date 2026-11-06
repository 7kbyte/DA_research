#!/usr/bin/env python3
"""Download pinned public research assets without fetching complete corpora.

All repository revisions are immutable commits. Large files live in scratch;
artifacts/assets/download_manifest.json is sufficient to reproduce their origin.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import requests

MODEL_REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
FINEMATH_REVISION = "e92b25a616738fe95dc186b64dfb19f9c8525594"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def asset_plan(scratch: Path) -> list[dict]:
    plan = []
    def add(repo, revision, name, destination, dataset=False):
        plan.append(dict(repo=repo, revision=revision, file=name,
                         destination=str(destination), dataset=dataset))
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "model.safetensors", "README.md"]:
        add("EleutherAI/pythia-410m", MODEL_REVISION, name,
            scratch / "models/pythia-410m-final" / name)
    for shard in [0, 16, 32, 48]:
        add("HuggingFaceTB/finemath", FINEMATH_REVISION,
            f"finemath-4plus/train-{shard:05d}-of-00064.parquet",
            scratch / f"raw/finemath-4plus-train-{shard:05d}.parquet", True)
    for split, total in [("train", "01024"), ("validation", "00008")]:
        add("allenai/c4", C4_REVISION, f"en/c4-{split}.00000-of-{total}.json.gz",
            scratch / f"raw/c4-en-{split}-00000.json.gz", True)
    for split in ["validation", "test"]:
        add("Salesforce/wikitext", WIKITEXT_REVISION,
            f"wikitext-103-raw-v1/{split}-00000-of-00001.parquet",
            scratch / f"raw/wikitext-103-raw-{split}.parquet", True)
    return plan


def download_one(asset: dict, endpoint: str, retries: int = 5) -> dict:
    path = Path(asset["destination"])
    path.parent.mkdir(parents=True, exist_ok=True)
    url = (endpoint.rstrip("/") + "/" + ("datasets/" if asset["dataset"] else "")
           + asset["repo"] + "/resolve/" + asset["revision"] + "/" + asset["file"])
    partial = path.with_name(path.name + ".partial")
    started = time.monotonic()
    if not path.exists():
        for attempt in range(retries):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(15, 120)) as r:
                    r.raise_for_status()
                    mode = "ab" if offset and r.status_code == 206 else "wb"
                    with partial.open(mode) as f:
                        for block in r.iter_content(chunk_size=4 * 1024 * 1024):
                            if block:
                                f.write(block)
                partial.replace(path)
                break
            except Exception as e:
                print(f"retry {attempt + 1}/{retries} {asset['file']}: {type(e).__name__}: {str(e)[:180]}", flush=True)
                if attempt == retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 15))
    record = {**asset, "url": url, "bytes": path.stat().st_size,
              "sha256": sha256_file(path), "seconds": time.monotonic() - started}
    print(json.dumps({k: record[k] for k in ["file", "bytes", "sha256", "seconds"]}), flush=True)
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch", type=Path, default=Path("/tmp/grad-research-20261106"))
    p.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    p.add_argument("--manifest", type=Path, default=Path("artifacts/assets/download_manifest.json"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", choices=["model", "data", "finemath-extra", "all"], default="all")
    args = p.parse_args()
    plan = asset_plan(args.scratch)
    if args.only == "finemath-extra":
        plan = [x for x in plan if "finemath" in x["repo"] and "00000-of" not in x["file"]]
    elif args.only != "all":
        plan = [x for x in plan if x["dataset"] == (args.only == "data")]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if args.manifest.exists():
        records = json.loads(args.manifest.read_text()).get("assets", [])
    records_by_file = {(x["repo"], x["file"]): x for x in records}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, asset, args.endpoint) for asset in plan]
        for future in as_completed(futures):
            record = future.result()
            records_by_file[(record["repo"], record["file"])] = record
            result = {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "endpoint": args.endpoint, "assets": list(records_by_file.values())}
            args.manifest.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
