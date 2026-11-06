#!/usr/bin/env python3
"""Download only pinned public validation splits and prepare frozen task inputs."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.evaluate import write_json
from replay_study.tasks import HARNESS_REVISION, TASK_SOURCES, format_example


def download(url, path, optional=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if not path.exists():
        try:
            response = requests.get(url, timeout=(10, 60 if not optional else 15))
            response.raise_for_status()
            path.write_bytes(response.content)
        except Exception as error:
            if optional:
                return {"url": url, "status": "unavailable", "error": str(error)}
            raise
    return {"url": url, "path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "wall_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/behavioral"))
    args = parser.parse_args()
    root = args.output_root
    started = time.perf_counter()
    license_evidence = {
        "piqa": {"license": "AFL-3.0", "primary_source_url": "https://yonatanbisk.com/piqa/",
                 "evidence": "Author homepage License section explicitly links Academic Free License v3.0.",
                 "mirror_license_metadata": "baber/piqa has no README/license file at pinned revision",
                 "saved_source": download("https://yonatanbisk.com/piqa/", root / "sources/piqa_author_homepage.html", optional=True)},
        "hellaswag": {"license": "MIT", "primary_source_url": "https://huggingface.co/datasets/Rowan/hellaswag/blob/218ec52e09a7e7462a5400043bb9a69a41d06b76/README.md",
                      "evidence": "Author-owned HF dataset card Licensing Information states MIT.",
                      "upstream_license_file_status": "https://raw.githubusercontent.com/rowanz/hellaswag/master/LICENSE returned 404; GitHub repository page returned451 during verification",
                      "saved_source": download(args.endpoint.rstrip("/") + "/datasets/Rowan/hellaswag/resolve/" + TASK_SOURCES["hellaswag"]["revision"] + "/README.md",
                                               root / "sources/hellaswag_dataset_card.md")},
    }
    manifests = {}
    for task, source in TASK_SOURCES.items():
        url = args.endpoint.rstrip("/") + f"/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}"
        raw_path = root / f"raw/{task}_validation.parquet"
        raw = download(url, raw_path)
        documents = pq.read_table(raw_path).to_pylist()
        if len(documents) != source["expected_items"]:
            raise ValueError(f"unexpected {task} validation count: {len(documents)}")
        examples = [format_example(task, document, index) for index, document in enumerate(documents)]
        output = root / f"{task}_validation.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in examples:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        fingerprint = hashlib.sha256("\n".join(row["fingerprint"] for row in examples).encode()).hexdigest()
        manifest = {"task": task, "source": source["repo"], "revision": source["revision"],
                    "split": "validation", "items": len(examples), "raw_asset": raw,
                    "canonical_source_url": f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['file']}",
                    "artifact": str(output), "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "ordered_example_fingerprint": fingerprint,
                    "harness_revision": HARNESS_REVISION, "task_format_version": "official_zero_shot_v1",
                    "target_delimiter": " ", "fewshot": 0, "apply_chat_template": False,
                    "official_metrics": ["acc", "acc_norm"], "official_single_primary_metric": None,
                    "license": license_evidence[task], "evaluation_role": "heldout_behavioral_supplement_not_for_tuning",
                    "label_counts": {str(label): sum(row["gold"] == label for row in examples) for label in range(len(examples[0]["choices"]))}}
        if task == "piqa":
            author_data = download("https://yonatanbisk.com/piqa/data/valid.jsonl", root / "raw/piqa_author_valid.jsonl", optional=True)
            author_labels = download("https://yonatanbisk.com/piqa/data/valid-labels.lst", root / "raw/piqa_author_valid-labels.lst", optional=True)
            equivalence = {"author_data": author_data, "author_labels": author_labels, "verified": False}
            if "path" in author_data and "path" in author_labels:
                official_docs = [json.loads(line) for line in Path(author_data["path"]).read_text().splitlines() if line.strip()]
                official_labels = [int(line) for line in Path(author_labels["path"]).read_text().splitlines() if line.strip()]
                if len(official_docs) != len(documents) or len(official_labels) != len(documents):
                    raise ValueError("author PiQA validation count differs from pinned harness mirror")
                if any(any(doc[key] != official[key] for key in ("goal", "sol1", "sol2")) or int(doc["label"]) != label
                       for doc, official, label in zip(documents, official_docs, official_labels)):
                    raise ValueError("author PiQA documents/labels differ from pinned harness mirror")
                equivalence.update(verified=True, matched_items=len(documents))
            manifest["author_validation_equivalence"] = equivalence
        write_json(output.with_suffix(".json"), manifest)
        manifests[task] = manifest
        print(json.dumps({"task": task, "items": len(examples), "path": str(output), "sha256": manifest["artifact_sha256"]}), flush=True)
    write_json(root / "data_manifest.json", {"harness_revision": HARNESS_REVISION, "tasks": manifests,
                                             "preparation_wall_seconds": time.perf_counter() - started,
                                             "result_inspection": "No model or checkpoint evaluated during preparation"})


if __name__ == "__main__":
    main()
