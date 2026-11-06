#!/usr/bin/env python3
"""Freeze held-out commands from completed active runs; perform no inference."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(registry, output_root, model_tag, checkpoints, expected_seconds):
    rows = read(registry)
    if not rows or any(r.get("active") is not True or r.get("phase") != "formal" for r in rows):
        raise ValueError("Every run must be explicitly active and formal; provisional/pilot runs are excluded")
    queue, records, shared = [], [], None
    output_root = Path(output_root)
    for row in rows:
        cfg, summary = read(row["config_path"]), read(row["summary_path"])
        operational = {"output_dir", "device", "resume_from", "max_steps", "deadline_utc",
                       "deadline_margin_seconds", "save_every_steps", "keep_step_checkpoints"}
        science_matches = all(summary["config"].get(k) == v for k, v in cfg.items() if k not in operational)
        if (summary["status"] != "completed" or not science_matches or
                summary["counters"]["target_prediction_tokens"] != cfg["target_prediction_tokens"]):
            raise ValueError(f"Incomplete run or config mismatch: {row['id']}")
        if not cfg.get("deterministic_algorithms"):
            raise ValueError("Strict deterministic training is required for the active study")
        paths = row.get("test_paths", {})
        if set(paths) != {"target", "c4", "wikitext"}:
            raise ValueError(f"Three explicit held-out corpus paths required: {row['id']}")
        properties = {"model": cfg["model_name_or_path"], "revision": cfg["model_revision"],
                      "precision": cfg["precision"], "backend": cfg.get("attention_backend", "auto"),
                      "allow_tf32": cfg.get("allow_tf32", True), "test_paths": paths,
                      "batch_size": cfg["evaluation_batch_size"]}
        if shared is None:
            shared = properties
        elif shared != properties:
            raise ValueError("One registry must share model, inference policy and held-out sets")
        command = [sys.executable, str(ROOT / "scripts/run_evaluate.py"),
                   "--model", properties["model"], "--revision", properties["revision"],
                   "--device", "cuda:0", "--batch-size", str(properties["batch_size"]),
                   "--precision", properties["precision"], "--attention-backend", properties["backend"],
                   "--allow-tf32-matmul" if properties["allow_tf32"] else "--no-allow-tf32-matmul",
                   "--allow-tf32-cudnn" if properties["allow_tf32"] else "--no-allow-tf32-cudnn",
                   "--cudnn-deterministic", "--no-cudnn-benchmark",
                   "--deterministic-algorithms", "--teacher-kl", "--include-chunks"]
        for name, path in sorted(paths.items()):
            command.extend(["--corpus", f"{name}={path}"])
        if not queue:
            output = output_root / f"initial_{model_tag}.json"
            queue.append({"id": f"evaluate_{model_tag}_initial", "argv": command + ["--output", str(output)],
                          "expected_seconds": expected_seconds, "outputs": [str(output)],
                          "env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}})
        for kind in checkpoints:
            if kind == "final":
                checkpoint = summary["checkpoint_path"]
                target_exposure = summary["counters"]["target_prediction_tokens"]
                replay_exposure = summary["counters"]["replay_prediction_tokens"]
                step = summary["counters"]["step"]
            else:
                best = summary["best_target_validation"]
                checkpoint, step = best["checkpoint_path"], best["step"]
                matches = [x for x in summary["validation_trajectory"] if x["step"] == step]
                if len(matches) != 1:
                    raise ValueError("Best checkpoint must match exactly one validation point")
                target_exposure, replay_exposure = (matches[0]["target_prediction_tokens"],
                                                    matches[0]["replay_prediction_tokens"])
            stat = Path(checkpoint).stat()
            output = output_root / row["id"] / f"{kind}.json"
            queue.append({"id": f"evaluate_{row['id']}_{kind}",
                          "argv": command + ["--checkpoint", checkpoint, "--output", str(output)],
                          "depends_on": [queue[0]["id"]], "expected_seconds": expected_seconds,
                          "outputs": [str(output)], "env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}})
            records.append({"run_id": row["id"], "checkpoint_kind": kind, "checkpoint": checkpoint,
                            "checkpoint_sha256": sha(checkpoint),
                            "checkpoint_bytes": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
                            "target_prediction_tokens_at_checkpoint": target_exposure,
                            "replay_prediction_tokens_at_checkpoint": replay_exposure, "step": step,
                            "summary_path": row["summary_path"], "summary_sha256": sha(row["summary_path"]),
                            "config_sha256": sha(row["config_path"]), "output": str(output)})
    manifest = {"frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "registry": str(registry), "registry_sha256": sha(registry),
                "inference_policy": shared, "checkpoint_selection": "fixed exposure final and target-validation best; no test selection",
                "test_tensor_sha256": {name: sha(path) for name, path in shared["test_paths"].items()},
                "evaluation_script_sha256": sha(ROOT / "scripts/run_evaluate.py"),
                "evaluation_module_sha256": sha(ROOT / "src/replay_study/evaluate.py"),
                "execution_module_sha256": sha(ROOT / "src/replay_study/execution.py"),
                "model_inference_performed_by_generator": False, "records": records}
    return queue, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--registry", type=Path, help="Legacy single active primary registry")
    source.add_argument("--phase-spec", type=Path, help="SHA-bound v2 multi-phase authority; no inference")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--model-tag", help="Required only for the legacy single-registry interface")
    p.add_argument("--checkpoints", nargs="+", choices=("final", "best"), default=["final", "best"])
    p.add_argument("--expected-seconds", type=float, default=180)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    if len(set(args.checkpoints)) != len(args.checkpoints):
        p.error("Duplicate checkpoint kinds")
    if args.queue.resolve() == args.manifest.resolve():
        p.error("queue and manifest must use distinct output paths")
    for path in (args.queue, args.manifest):
        if path.exists():
            raise FileExistsError(f"Use a new frozen output path: {path}")
    if args.phase_spec:
        from evaluation_plan import build as build_phases
        queue, manifest = build_phases(args.phase_spec, args.output_root, args.manifest, "nll", args.expected_seconds)
    else:
        if not args.model_tag:
            p.error("--model-tag is required with --registry")
        queue, manifest = build(args.registry, args.output_root, args.model_tag, args.checkpoints, args.expected_seconds)
    for path, payload in ((args.queue, queue), (args.manifest, manifest)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"jobs": len(queue), "queue": str(args.queue), "model_inference": False}))


if __name__ == "__main__":
    main()
