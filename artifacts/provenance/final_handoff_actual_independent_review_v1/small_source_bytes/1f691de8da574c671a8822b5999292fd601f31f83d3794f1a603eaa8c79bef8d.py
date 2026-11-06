#!/usr/bin/env python3
"""Dispatch one SHA-bound evaluation job; inherit one GPU, never schedule children."""
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def attestation_path(output):
    path = Path(output)
    return path.with_name(path.stem + "_execution.json")


def verify_outputs(output, manifest_sha256, job_id):
    record = read(attestation_path(output))
    if (record.get("manifest_sha256") != manifest_sha256 or record.get("job_id") != job_id
            or record.get("verified_frozen_inputs") is not True):
        raise ValueError("Baseline/previous output execution attestation differs from frozen job")
    for path, expected in record["output_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Evaluation output changed after execution: {path}")
    return record


def verified_job(path, manifest_sha256, job_id, require_baseline=True):
    if sha(path) != manifest_sha256:
        raise ValueError("Frozen evaluation manifest SHA mismatch")
    manifest = read(path)
    if manifest.get("schema_version") != 2 or manifest.get("mode") not in ("nll", "behavior"):
        raise ValueError("A v2 frozen NLL/behavior manifest is required")
    matches = [job for job in manifest["jobs"] if job["id"] == job_id]
    if len(matches) != 1:
        raise ValueError("Exactly one frozen job ID required")
    job = matches[0]
    for source, expected in {**manifest["common_input_sha256"], **job["input_sha256"]}.items():
        if sha(source) != expected:
            raise ValueError(f"Frozen evaluation input SHA mismatch: {source}")
    if job["kind"] != "initial" and require_baseline:
        group = manifest["baseline_groups"][job["baseline_group"]]
        origin = group.get("initial_origin")
        verify_outputs(group["initial_output"], origin["manifest_sha256"] if origin else manifest_sha256,
                       group["initial_job_id"])
        baseline = read(group["initial_output"])
        policy = group["inference_policy"]
        if baseline.get("checkpoint") is not None or (baseline.get("model"), baseline.get("revision")) != (policy["model"], policy["revision"]):
            raise ValueError("Complete identical initial-model baseline required")
        if manifest["mode"] == "behavior" and baseline.get("complete") is not True:
            raise ValueError("Complete full-split behavioral baseline required")
    return manifest, job


def materialize_aliases(manifest, job, canonical_attestation):
    """Copy only result files; preserve each logical endpoint and zero extra inference."""
    original = read(job["output"])
    canonical = Path(job["output"]).resolve()
    for logical in (r for r in manifest["records"] if r["inference_job_id"] == job["id"]):
        output = Path(logical["output"])
        if output.resolve() == canonical:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        value = copy.deepcopy(original)
        value["checkpoint"] = logical["checkpoint"]
        output_paths = [output]
        if manifest["mode"] == "behavior":
            for name, task in value["tasks"].items():
                source = Path(task["predictions_path"])
                destination = output.with_name(output.stem + f"_{name}.jsonl")
                if destination.exists():
                    if sha(destination) != sha(source):
                        raise ValueError("Existing inference alias predictions differ")
                else:
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copyfile(source, destination)
                task["predictions_path"] = str(destination)
                task["predictions_sha256"] = sha(destination)
                output_paths.append(destination)
        value["inference_reuse"] = {"canonical_output": str(canonical), "canonical_job_id": job["id"],
            "checkpoint_sha256": logical["checkpoint_sha256"], "canonical_checkpoint_sha256": job["checkpoint_sha256"],
            "source_model_equivalence": logical.get("source_model_equivalence"), "logical_record_id": logical["id"],
            "additional_model_forward_calls": 0, "additional_inference_subprocess_wall_seconds": 0,
            "timing_fields_in_metrics_belong_to_canonical_inference": True}
        write(output, value)
        alias_attestation = {**canonical_attestation, "logical_record_id": logical["id"],
            "checkpoint_sha256": logical["checkpoint_sha256"], "weights_identity": logical["weights_identity"],
            "source_model_equivalence": logical.get("source_model_equivalence"),
            "inference_alias": True, "canonical_output": str(canonical),
            "inference_subprocess_wall_seconds": 0,
            "canonical_inference_subprocess_wall_seconds": canonical_attestation["inference_subprocess_wall_seconds"],
            "output_sha256": {str(p): sha(p) for p in output_paths}}
        write(attestation_path(output), alias_attestation)


def validate_result(manifest, job):
    value = read(job["output"])
    group = manifest["baseline_groups"][job["baseline_group"]]
    policy = group["inference_policy"]
    if ((value.get("model"), value.get("revision"), value.get("precision")) !=
            (policy["model"], policy["revision"], policy["precision"]) or
            value.get("checkpoint") != job.get("checkpoint")):
        raise ValueError("Scorer result model/checkpoint/precision differs from frozen command")
    actual = value["execution_provenance"]
    for key in ("torch_version", "cuda_version", "gpu_name", "precision", "attention_implementation", "device_type", "effective_precision", "model_weight_dtype",
                "flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled",
                "deterministic_algorithms", "deterministic_warn_only", "allow_tf32_matmul", "allow_tf32_cudnn",
                "cudnn_deterministic", "cudnn_benchmark", "CUBLAS_WORKSPACE_CONFIG"):
        if actual.get(key) != policy[key]:
            raise ValueError("Scorer execution differs from frozen " + key)
    if actual.get("attention_backend_requested") != policy["attention_backend"]:
        raise ValueError("Scorer attention backend differs from frozen command")
    if manifest["mode"] == "nll":
        if set(value["corpora"]) != set(group["test_paths"]):
            raise ValueError("Scorer held-out corpora differ from frozen command")
        for corpus in value["corpora"].values():
            if corpus.get("complete_corpus") is not True or "kl" not in corpus:
                raise ValueError("Complete held-out NLL and teacher KL required")
    else:
        if (value.get("complete") is not True or value.get("evaluation_role") != "formal" or
                value.get("project_primary_metric") != "acc_norm" or value.get("project_auxiliary_metric") != "acc" or
                set(value["tasks"]) != set(group["tasks"])):
            raise ValueError("Complete full-split official behavioral result required")
        for name, metrics in value["tasks"].items():
            task = group["tasks"][name]
            if (metrics.get("complete_split") is not True or metrics["items"] != task["items"] or
                    metrics["batch_size"] != policy["batch_size"] or metrics["max_length"] != policy["max_length"] or
                    metrics["data_sha256"] != task["sha256"]):
                raise ValueError("Behavior split/data/batch/context differs from frozen command")


def verify_environment(policy, torch_module=None):
    """Fail before scoring when the assigned worker cannot match saved training facts."""
    if torch_module is None:
        import torch as torch_module
    observed = {"torch_version": str(torch_module.__version__), "cuda_version": torch_module.version.cuda,
                "gpu_name": torch_module.cuda.get_device_name("cuda:0")}
    for key, value in observed.items():
        if key not in policy or value != policy[key]:
            raise ValueError("Worker environment differs from recorded training " + key)
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device", choices=("cuda:0",), required=True)
    parser.add_argument("--verify-only", action="store_true", help="CPU source verification without inference")
    args = parser.parse_args()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    begin = time.perf_counter()
    manifest, job = verified_job(args.manifest, args.manifest_sha256, args.job_id,
                                 require_baseline=not args.verify_only)
    verification_seconds = time.perf_counter() - begin
    if args.verify_only:
        print(json.dumps({"verified": True, "job_id": job["id"], "model_inference": False,
                          "baseline_runtime_check_deferred": job["kind"] != "initial"}))
        return
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Frozen execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    # A prior successful attempt may have stopped during cheap alias copying.
    if job["kind"] == "reuse":
        origin = job["inference_origin"]
        original = verify_outputs(origin["output"], origin["manifest_sha256"], origin["job_id"])
        if original.get("weights_identity", "checkpoint:" + str(original.get("checkpoint_sha256"))) != job["weights_identity"]:
            raise ValueError("Reused inference model_state identity differs from frozen endpoint")
        attestation = {**original, "manifest": str(args.manifest), "manifest_sha256": args.manifest_sha256,
            "job_id": job["id"], "inference_origin": origin, "inference_subprocess_wall_seconds": 0,
            "verification_wall_seconds": verification_seconds}
    elif attestation_path(job["output"]).exists():
        attestation = verify_outputs(job["output"], args.manifest_sha256, job["id"])
    else:
        argv = list(job["argv"])
        if argv[argv.index("--device") + 1] != "cuda:0":
            raise ValueError("Frozen single-GPU child must use cuda:0")
        environment = verify_environment(manifest["baseline_groups"][job["baseline_group"]]["inference_policy"])
        inference_begin = time.perf_counter()
        subprocess.run(argv, check=True)
        validate_result(manifest, job)
        paths = [Path(job["output"])]
        if manifest["mode"] == "behavior":
            paths += [Path(task["predictions_path"]) for task in read(job["output"])["tasks"].values()]
        attestation = {"manifest": str(args.manifest), "manifest_sha256": args.manifest_sha256,
            "job_id": job["id"], "baseline_group": job["baseline_group"],
            "checkpoint_sha256": job.get("checkpoint_sha256"), "verified_frozen_inputs": True,
            "weights_identity": job.get("weights_identity"),
            "source_model_equivalence": job.get("source_model_equivalence"),
            "actual_argv": argv, "started_at_utc": started,
            "preflight_environment": environment,
            "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "verification_wall_seconds": verification_seconds,
            "inference_subprocess_wall_seconds": time.perf_counter() - inference_begin,
            "inference_alias": False, "output_sha256": {str(p): sha(p) for p in paths}}
        write(attestation_path(job["output"]), attestation)
    materialize_aliases(manifest, job, attestation)


if __name__ == "__main__":
    main()
