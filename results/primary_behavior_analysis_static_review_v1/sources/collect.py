#!/usr/bin/env python3
"""CPU-only summaries for v2 multi-phase behavior manifests; no inference."""
import argparse
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_study.behavioral_analysis import (CONDITIONAL_SCOPE, grouped_seed_summary, jsonl,
    paired_binary_summary, paired_seed_summary, read, sha, verify_prediction_rows, write_bundle)
from run_frozen_evaluation import validate_result, verify_outputs


STRATA = ("role", "phase_name", "data_family", "model_tag", "baseline_group")


def collect(path):
    path = Path(path)
    manifest, digest = read(path), sha(path)
    if manifest.get("schema_version") != 2 or manifest.get("mode") != "behavior":
        raise ValueError("Explicit v2 behavioral manifest required")
    for source, expected in manifest["common_input_sha256"].items():
        if sha(source) != expected:
            raise ValueError("Frozen behavior plan input changed: " + source)
    policy = manifest["analysis_policy"]
    bootstrap, seed = policy["bootstrap_replicates"], policy["bootstrap_seed"]
    if bootstrap != 5000 or seed != 20261106:
        raise ValueError("Frozen conditional analysis policy changed")
    baseline_predictions, baseline_execution, prepared, evidence = {}, {}, {}, {}
    for key, group in manifest["baseline_groups"].items():
        origin = group.get("initial_origin")
        source_digest = origin["manifest_sha256"] if origin else digest
        output, job_id = group["initial_output"], group["initial_job_id"]
        verify_outputs(output, source_digest, job_id)
        validate_result(manifest, {"output": output, "kind": "initial", "baseline_group": key})
        initial = read(output)
        baseline_execution[key] = initial["execution_provenance"]
        baseline_predictions[key], prepared[key] = {}, {}
        for name, task in group["tasks"].items():
            prepared[key][name] = jsonl(task["artifact"])
            metrics = initial["tasks"][name]
            if sha(metrics["predictions_path"]) != metrics["predictions_sha256"]:
                raise ValueError("Baseline predictions changed")
            baseline_predictions[key][name] = verify_prediction_rows(jsonl(metrics["predictions_path"]), prepared[key][name])
        evidence[job_id] = {"output": output, "sha256": sha(output), "origin_manifest_sha256": source_digest}
    endpoints, strata = [], defaultdict(list)
    from replay_study.execution import verify_matching_execution
    for record in manifest["records"]:
        key, output = record["baseline_group"], record["output"]
        attestation = verify_outputs(output, digest, record["inference_job_id"])
        if attestation.get("checkpoint_sha256") != record["checkpoint_sha256"]:
            raise ValueError("Logical endpoint checkpoint SHA differs from execution attestation")
        if attestation.get("weights_identity") != record["weights_identity"]:
            raise ValueError("Logical endpoint model_state identity differs from execution attestation")
        validate_result(manifest, {"output": output, "checkpoint": record["checkpoint"], "baseline_group": key})
        current = read(output)
        verify_matching_execution(baseline_execution[key], current["execution_provenance"])
        evidence[record["id"]] = {"output": output, "sha256": sha(output), "inference_job_id": record["inference_job_id"],
                                  "inference_alias_of": record["inference_alias_of"]}
        for name, task in manifest["baseline_groups"][key]["tasks"].items():
            metrics = current["tasks"][name]
            prediction_path = metrics["predictions_path"]
            if (sha(prediction_path) != metrics["predictions_sha256"] or
                    attestation["output_sha256"].get(prediction_path) != metrics["predictions_sha256"]):
                raise ValueError("Logical predictions are not bound by summary and execution attestation")
            after = verify_prediction_rows(jsonl(prediction_path), prepared[key][name])
            for metric in ("acc_norm", "acc"):
                stream = (seed + int(hashlib.sha256(f"{record['id']}|{name}|{metric}".encode()).hexdigest()[:8], 16)) % (2 ** 32)
                stats = paired_binary_summary(baseline_predictions[key][name], after, metric, bootstrap, stream)
                if not math.isclose(stats["accuracy"]["estimate"], metrics[metric], rel_tol=0, abs_tol=1e-15):
                    raise ValueError("Stored task accuracy differs from per-item correctness")
                row = {k: record.get(k) for k in (*STRATA, "run_id", "actual_run_id", "phase", "group", "training_seed",
                    "selection_seed", "generation_seed", "ratio", "weight", "learning_rate", "step",
                    "target_prediction_tokens_at_checkpoint", "replay_prediction_tokens_at_checkpoint",
                    "checkpoint_sha256", "weights_identity", "inference_job_id")}
                row.update(checkpoint_kind=record["kind"], task=name, metric=metric, primary_metric=metric == "acc_norm",
                    seed=record["training_seed"], items=stats["items"], **stats["counts"],
                    initial_accuracy=stats["initial_accuracy"], accuracy=stats["accuracy"]["estimate"],
                    accuracy_ci_low=stats["accuracy"]["ci_low"], accuracy_ci_high=stats["accuracy"]["ci_high"],
                    accuracy_delta=stats["accuracy_delta"], delta_ci_low=stats["delta_ci_low"], delta_ci_high=stats["delta_ci_high"],
                    conditional_forgetting_rate=stats["conditional_forgetting"]["estimate"],
                    conditional_acquisition_rate=stats["conditional_acquisition"]["estimate"],
                    bootstrap_seed=stream, uncertainty_scope=CONDITIONAL_SCOPE)
                endpoints.append(row)
                if record.get("selection_seed") is None:
                    strata[tuple(row[k] for k in STRATA)].append({**row, "phase": "formal"})
    aggregates, pairs, pair_aggregates = [], [], []
    for cell, values in strata.items():
        context = dict(zip(STRATA, cell))
        aggregates += [{**context, **row} for row in grouped_seed_summary(values)]
        paired, grouped = paired_seed_summary(values)
        pairs += [{**context, **row} for row in paired]
        pair_aggregates += [{**context, **row} for row in grouped]
    return {"endpoints": endpoints, "seed_aggregates": aggregates, "paired_seed_differences": pairs,
        "paired_seed_aggregates": pair_aggregates, "metadata": {"complete": True, "manifest": str(path),
            "manifest_sha256": digest, "logical_endpoints": len(manifest["records"]),
            "inference_jobs": manifest["workload"]["total_inference_jobs"], "strata": list(STRATA),
            "primary_metric": "acc_norm", "auxiliary_metric": "acc", "bootstrap_replicates": bootstrap,
            "bootstrap_seed": seed, "conditional_uncertainty_scope": CONDITIONAL_SCOPE,
            "question_assumption": "IID/exchangeable questions; same-source activity/context dependence is not corrected",
            "training_seed_report": "individual values and sample SD ddof=1; final/best and aliases are not additional training seeds",
            "full_training_cost": "Read the SHA-bound completed summary once per physical training run; do not scale to best exposure or sum aliases",
            "evidence": evidence, "model_inference_performed_by_analysis": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new analysis output directory")
    write_bundle(collect(args.manifest), args.output)


if __name__ == "__main__":
    main()
