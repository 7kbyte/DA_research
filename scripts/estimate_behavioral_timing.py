#!/usr/bin/env python3
"""Budget from timing-only evidence and frozen work counts; never parse task results."""
import argparse
import datetime as dt
import json
from pathlib import Path

from evaluation_plan import encoded, sha


def estimate(monitor_path, manifest_path, structural_path):
    timing = json.loads(Path(monitor_path).read_text())
    if timing.get("complete_timing_available") is not True or timing.get("scores_or_predictions_parsed") is not False:
        raise ValueError("Complete score-blind initial timing evidence required")
    manifest = json.loads(Path(manifest_path).read_text())
    if sha(manifest_path) != timing["manifest_sha256"]:
        raise ValueError("Timing and primary inference manifest differ")
    structural = json.loads(Path(structural_path).read_text())
    execution = timing["execution"]
    scorer = execution["inference_subprocess_wall_seconds"]
    verify = execution["verification_wall_seconds"]
    deployed = timing["state_snapshot"]["elapsed_seconds"]
    overhead = max(0., deployed - scorer - verify)
    initial = next(j for j in manifest["jobs"] if j["kind"] == "initial")
    adapted = [j for j in manifest["jobs"] if j["kind"] == "adapted"]
    def input_bytes(job):
        inputs = {**manifest["common_input_sha256"], **job["input_sha256"]}
        return sum(Path(source).stat().st_size for source in inputs)
    initial_bytes = input_bytes(initial)
    byte_ratios = [input_bytes(job) / initial_bytes for job in adapted]
    primary_scenarios = []
    for factor in (1., 1.25, 1.5):
        seconds = deployed + len(adapted) * (scorer * factor + overhead) + verify * sum(byte_ratios)
        primary_scenarios.append({"adapted_scorer_factor_assumption": factor,
            "remaining_scorer_job_wall_seconds_proxy": seconds - deployed,
            "all_67_job_wall_seconds_proxy": seconds,
            "ideal_four_uncontended_gpu_wall_seconds_lower_bound": seconds / 4})
    # Only the six proven primary D/B4 aliases are deducted here. Future
    # extended/A reuse remains conditional until its own complete-state proof.
    counts = {"pythia-410m": 109, "pythia-1b": 25, "pythia-2.8b": 16}
    base_n = structural["scope"]["pythia-410m"]["parameter_count"]
    ratios = {model: structural["scope"][model]["parameter_count"] / base_n for model in counts}
    whole_scenarios = []
    for uncertainty in (.5, 1., 2.):
        factors = {model: (1. if model == "pythia-410m" else ratios[model] * uncertainty) for model in counts}
        seconds_by_model = {model: count * deployed * factors[model] for model, count in counts.items()}
        whole_scenarios.append({"unmeasured_large_model_factor_relative_to_parameter_ratio": uncertainty,
            "wall_factors_relative_to_measured_410m_initial": factors,
            "job_wall_seconds_proxy_by_model": seconds_by_model,
            "aggregate_job_wall_seconds_proxy": sum(seconds_by_model.values()),
            "ideal_four_uncontended_gpu_wall_seconds_lower_bound": sum(seconds_by_model.values()) / 4,
            "interpretation": "Assumed sensitivity scenario, not a measured bound or confidence interval; batch8 kernel/launch costs are unmeasured"})
    return {"schema_version": 1, "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timing_source": str(monitor_path), "timing_source_sha256": sha(monitor_path),
        "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
        "structural_source": str(structural_path), "structural_source_sha256": sha(structural_path),
        "measured_410m_initial": {"scorer_subprocess_wall_seconds": scorer, "source_verification_wall_seconds": verify,
            "scheduler_elapsed_seconds": deployed, "unattributed_preflight_polling_overhead_seconds": overhead,
            "source_verification_input_bytes": initial_bytes, "single_observation_not_a_runtime_distribution": True},
        "primary": {"total_inferences_including_completed_initial": 67, "adapted_remaining": 66,
            "checkpoint_input_byte_ratio_range": [min(byte_ratios), max(byte_ratios)], "sensitivity_scenarios": primary_scenarios,
            "verification_proxy_assumption": "SHA time proportional to verified bytes; caching/contention may violate it",
            "adapted_proxy_assumption": "Same forward shapes, extra checkpoint loading/baseline transitions unmeasured; scorer multipliers are sensitivity assumptions"},
        "whole_preregistered_scope": {"maximum_passes_after_six_proven_primary_aliases": sum(counts.values()),
            "model_counts": counts, "parameter_ratios_for_structure_only": ratios, "sensitivity_scenarios": whole_scenarios,
            "further_reuse_not_yet_deducted": "A selected cells may reuse primary, and extended D/B4 may match; require current checkpoint/model_state proof",
            "unmeasured_models": ["pythia-1b", "pythia-2.8b"]},
        "evaluation_window_after_training_seconds": 12392,
        "scope_excludes": ["NLL test", "replay-fit diagnostic", "training", "packaging", "fault retries", "resource contention"],
        "wall_is_not_pure_gpu_time": True, "scores_or_predictions_parsed": False, "new_inference_started": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--monitor", type=Path, default=Path("results/first_behavior_timing_monitor.json"))
    p.add_argument("--manifest", type=Path, default=Path("docs/primary_evaluation_v2_execution_matched_behavior_manifest.json"))
    p.add_argument("--structural", type=Path, default=Path("results/behavioral_budget_preregistered_v2.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new timing budget snapshot output")
    value = estimate(args.monitor, args.manifest, args.structural)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(encoded(value))
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
