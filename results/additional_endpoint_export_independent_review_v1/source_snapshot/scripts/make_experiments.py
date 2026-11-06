#!/usr/bin/env python3
"""Generate auditable lambda/formal configs, or execute and summarize one run.

Generation never starts a scheduler. Formal generation is an explicit separate
command, to be invoked only after the root authorizes the formal protocol.
"""
from __future__ import annotations
import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_study.train import TrainConfig, train

GROUPS = {
    "B0": (None, "none"), "B1": ("c4_random", "ce"), "B2": ("c4_random", "kl"),
    "B3": ("generated_random", "ce"), "B4": ("generated_random", "kl"),
    "D": ("generated_dedup_random", "kl"), "M": ("generated_nll_stratified", "kl"),
    "C": ("generated_cluster", "kl"),
}


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def number_label(value: float) -> str:
    return format(value, "g").replace(".", "p").replace("-", "m")


def build_experiments(phase: str, root: Path, scratch: Path, shared_lr: float,
                      shared_lambda: float | None = None, protocol: dict | None = None, namespace: str | None = None):
    """Return configs, registry, queue without filesystem writes or GPU work."""
    if phase == "lambda":
        if protocol is None:
            raise ValueError("Lambda calibration requires the frozen protocol")
        specifications = [(group, protocol["seed"], protocol["replay_ratio"], weight)
                          for weight in protocol["lambda_grid"] for group in protocol["groups"]]
        target_tokens, eval_every = protocol["target_prediction_tokens"], 0
    elif phase == "formal":
        if shared_lambda is None:
            raise ValueError("Formal generation requires an explicitly selected shared lambda")
        specifications = [(group, seed, 0.0 if group == "B0" else .05,
                           0.0 if group == "B0" else shared_lambda)
                          for seed in [11, 22, 33] for group in GROUPS]
        specifications += [(group, seed, ratio, shared_lambda)
                           for group in ["B4", "M"] for ratio in [.01, .1] for seed in [11, 22, 33]]
        target_tokens, eval_every = 10_000_000, 50
    else:
        raise ValueError("phase must be lambda or formal")
    configs, registry, queue = {}, [], []
    namespace = namespace or phase
    for group, seed, ratio, weight in specifications:
        prefix = f"pilot_{namespace}" if phase == "lambda" else namespace
        run_id = f"{prefix}_{group}_s{seed}_b{number_label(ratio)}_w{number_label(weight)}"
        bank, objective = GROUPS[group]
        config_path = root / "configs" / namespace / f"{run_id}.json"
        output_dir = scratch / "runs" / run_id
        summary_copy = root / "results" / ("pilots" if phase == "lambda" else "runs") / f"{run_id}_summary.json"
        cfg = TrainConfig(model_name_or_path=str(scratch / "models/pythia-410m-final"),
            model_revision="9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
            target_path=str(root / "artifacts/data/finemath_train.pt"),
            output_dir=str(output_dir), seed=seed, target_prediction_tokens=target_tokens,
            replay_path=str(root / "artifacts/banks" / f"{bank}_seed{seed}.pt") if bank else None,
            replay_objective=objective, replay_ratio=ratio, replay_weight=weight,
            microbatch_size=16, gradient_accumulation_steps=4, learning_rate=shared_lr,
            precision="bf16", device="cuda:0", eval_every_steps=eval_every, save_every_steps=0,
            deterministic_algorithms=bool((protocol or {}).get("deterministic_algorithms", False)),
            attention_backend=(protocol or {}).get("attention_backend", "auto"),
            evaluate_at_start=False, evaluate_at_end=True, evaluate_teacher_kl=True,
            evaluation_batch_size=16, keep_step_checkpoints=False,
            validation_paths=(protocol.get("validation_paths") if protocol else None) or {
                "target":str(root/"artifacts/data/finemath_validation.pt"),
                "c4":str(root/"artifacts/data/c4_validation.pt"),
                "wikitext":str(root/"artifacts/data/wikitext103_validation.pt")},
            deadline_utc="2026-11-07T03:26:32Z", run_label=run_id)
        cfg.validate()
        configs[str(config_path)] = dataclasses.asdict(cfg)
        record = {"id":run_id, "group":group, "seed":seed, "ratio":ratio, "weight":weight,
                  "learning_rate":shared_lr, "phase":phase, "config_path":str(config_path),
                  "output_dir":str(output_dir), "summary_path":str(summary_copy), "active":True,
                  "test_paths":protocol.get("test_paths",{}) if protocol else {}}
        registry.append(record)
        queue.append({"id":run_id, "argv":[sys.executable,str(root/"scripts/make_experiments.py"),
                    "run","--config",str(config_path),"--summary-copy",str(summary_copy)],
                    "env":{"CUBLAS_WORKSPACE_CONFIG":":4096:8"} if cfg.deterministic_algorithms else {},
                    "expected_seconds":180 if phase=="lambda" else 600,
                    "outputs":[str(output_dir/"summary.json"),str(summary_copy)]})
    return configs, registry, queue


def capture_snapshot(root: Path, scratch: Path) -> dict:
    """Hash scientific code, input tensors/manifests, protocol, and all configs."""
    sections = {
        "code": sorted([*root.glob("src/**/*.py"), *root.glob("scripts/*.py")]),
        "data": sorted(path for directory in ["artifacts/data", "artifacts/evaluation_v3", "artifacts/banks", "artifacts/generated"]
                       for path in (root / directory).rglob("*") if path.is_file()),
        "configs": sorted(root.glob("configs/**/*.json")),
        "protocol": sorted([*root.glob("docs/hyperparameter_selection_protocol*.json"),
                            *root.glob("docs/formal*protocol*.json"), *root.glob("docs/超参数选择协议*.md"),
                            *root.glob("docs/数值复现审查.md")]),
        "execution_evidence": sorted([*root.glob("results/determinism*.json"),
                            *root.glob("results/evaluation_execution_policy_amendment.json"),
                            *root.glob("artifacts/assets/model_provenance.json")]),
        "selection": sorted([*root.glob("results/pilot*selection*.json"), *root.glob("results/pilots/*amendment_reason.json")]),
    }
    result = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "workspace": str(root), "scratch": str(scratch), "hash_algorithm": "sha256"}
    for section, paths in sections.items():
        records = {}
        for path in paths:
            if path.is_symlink():
                resolved = path.resolve()
            else:
                resolved = path
            before = resolved.stat()
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            after = resolved.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Input changed while capturing provenance: {path}")
            records[str(path.relative_to(root))] = {"sha256": digest.hexdigest(), "bytes": after.st_size}
        result[section] = records
    result["aggregate_sha256"] = {section: hashlib.sha256(json.dumps(result[section],sort_keys=True).encode()).hexdigest()
                                   for section in sections}
    return result


def generate(args) -> None:
    protocol = json.loads(args.protocol.read_text())
    lr = args.learning_rate or protocol["shared_learning_rate"]
    namespace = args.namespace or args.phase
    if args.phase == "formal" and protocol.get("data_version") != "v3":
        raise ValueError("Formal generation requires the root-approved v3 evaluation protocol")
    configs, registry, queue = build_experiments(args.phase, ROOT, args.scratch, lr, args.shared_lambda, protocol, namespace)
    if args.phase == "lambda":
        for name,path in protocol["validation_paths"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != protocol["validation_sha256"][name]:
                raise ValueError(f"Frozen validation data changed: {name}")
        for name,path in protocol["bank_paths"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != protocol["bank_tensor_sha256"][name]:
                raise ValueError(f"Frozen replay bank changed: {path}")
    if args.phase == "formal":
        missing = sorted({config["replay_path"] for config in configs.values()
                          if config["replay_path"] and not Path(config["replay_path"]).exists()})
        if missing:
            raise FileNotFoundError(f"Formal bank inputs must be ready before generation: {missing}")
    for path, config in configs.items():
        destination = Path(path)
        if destination.exists() and json.loads(destination.read_text()) != config:
            raise FileExistsError(f"Refusing to overwrite a changed experiment config: {destination}")
        atomic_json(destination, config)
    atomic_json(ROOT/"configs"/f"{namespace}_registry.json", registry)
    atomic_json(ROOT/"configs"/f"{namespace}_queue.json", queue)
    if args.phase == "formal":
        atomic_json(ROOT/"artifacts/provenance"/f"{namespace}_input_snapshot.json", capture_snapshot(ROOT, args.scratch))
    print(json.dumps({"phase":args.phase,"configs":len(configs),
                      "registry":str(ROOT/"configs"/f"{namespace}_registry.json"),
                      "queue":str(ROOT/"configs"/f"{namespace}_queue.json")}))


def evaluation_queue(args) -> None:
    protocol = json.loads(args.protocol.read_text())
    source = json.loads(args.registry.read_text())
    if args.kind == "lr":
        records = [{"id": f"lr_{row['label']}", "checkpoint_path": row["checkpoint_path"],
                    "learning_rate": row["learning_rate"], "source_record": row} for row in source]
        records.insert(0, {"id": "initial", "checkpoint_path": None})
    else:
        records = [{"id":row["id"], "checkpoint_path":str(Path(row["output_dir"])/"checkpoints/latest.pt"),
                    "source_record":row} for row in source]
    queue, evaluation_registry = [], []
    for record in records:
        checkpoint = record["checkpoint_path"]
        if checkpoint and not Path(checkpoint).exists():
            raise FileNotFoundError(checkpoint)
        run_id = record["id"]
        job_id = f"evaluate_{args.namespace}_{run_id}"
        output = ROOT/"results/pilots"/args.namespace/f"{run_id}.json"
        argv = [sys.executable,str(ROOT/"scripts/run_evaluate.py"),"--model",str(args.scratch/"models/pythia-410m-final"),
                "--revision",protocol["model_revision"],"--output",str(output),"--device","cuda:0",
                "--batch-size","16","--precision","bf16","--teacher-kl","--include-chunks"]
        if checkpoint:
            argv += ["--checkpoint",checkpoint]
        for name,path in protocol["validation_paths"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != protocol["validation_sha256"][name]:
                raise ValueError(f"Frozen validation data changed: {name}")
            argv += ["--corpus",f"{name}={path}"]
        queue.append({"id":job_id,"argv":argv,"expected_seconds":120,"outputs":[str(output)]})
        evaluation_registry.append({**record,"job_id":job_id,"evaluation_path":str(output),
                                    "data_version":protocol["data_version"]})
    atomic_json(ROOT/"configs"/f"{args.namespace}_eval_queue.json",queue)
    atomic_json(ROOT/"configs"/f"{args.namespace}_evaluation_registry.json",evaluation_registry)
    print(json.dumps({"namespace":args.namespace,"evaluation_jobs":len(queue)}))


def run(args) -> None:
    cfg = TrainConfig.from_dict(json.loads(args.config.read_text()))
    summary = train(cfg)
    atomic_json(args.summary_copy, summary)
    if summary["status"] != "completed":
        raise RuntimeError(f"Run stopped before its target budget: {summary['status']}")
    print(json.dumps({"id":cfg.run_label,"status":summary["status"],"summary_copy":str(args.summary_copy)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    generate_parser=sub.add_parser("generate")
    generate_parser.add_argument("--phase",choices=["lambda","formal"],required=True)
    generate_parser.add_argument("--scratch",type=Path,default=Path("/tmp/grad-research-20261106"))
    generate_parser.add_argument("--protocol",type=Path,default=ROOT/"docs/hyperparameter_selection_protocol.json")
    generate_parser.add_argument("--learning-rate",type=float)
    generate_parser.add_argument("--shared-lambda",type=float)
    generate_parser.add_argument("--namespace")
    generate_parser.set_defaults(function=generate)
    run_parser=sub.add_parser("run")
    run_parser.add_argument("--config",type=Path,required=True)
    run_parser.add_argument("--summary-copy",type=Path,required=True)
    run_parser.set_defaults(function=run)
    evaluate_parser=sub.add_parser("evaluation-queue")
    evaluate_parser.add_argument("--registry",type=Path,required=True)
    evaluate_parser.add_argument("--protocol",type=Path,required=True)
    evaluate_parser.add_argument("--namespace",required=True)
    evaluate_parser.add_argument("--kind",choices=["lr","lambda"],required=True)
    evaluate_parser.add_argument("--scratch",type=Path,default=Path("/tmp/grad-research-20261106"))
    evaluate_parser.set_defaults(function=evaluation_queue)
    snapshot_parser=sub.add_parser("snapshot")
    snapshot_parser.add_argument("--output",type=Path,required=True)
    snapshot_parser.add_argument("--scratch",type=Path,default=Path("/tmp/grad-research-20261106"))
    snapshot_parser.set_defaults(function=lambda args: atomic_json(args.output,capture_snapshot(ROOT,args.scratch)))
    args=parser.parse_args();args.function(args)


if __name__=="__main__":main()
