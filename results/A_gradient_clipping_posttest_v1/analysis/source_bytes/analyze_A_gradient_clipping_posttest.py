"""Declared post-test A diagnostic using training gradient logs only.

No model/checkpoint/dataset load or score-result read. Every run must be complete.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/A_gradient_clipping_posttest_protocol_v1.json"
PROTOCOL_SHA = "9f29285144a71c38fc804c29f2f3e378198693ae38b5b7e1e38af18579acc986"
METHODS = ("B1", "B2", "B3", "B4", "M", "C")
LAMBDAS = (.01, .05, .1, .5, 1., 5., 10.)
SEEDS = (11, 22, 33)
WINDOWS = ("all", "early", "late")
METRICS = ("threshold_exceedance_fraction", "mean", "median", "p90", "max")
TOKEN_KEYS = tuple(f"{part}_{suffix}" for part in ("target", "replay")
                   for suffix in ("prediction_tokens", "raw_input_tokens", "padded_input_tokens"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def signature(path):
    value = Path(path).stat()
    return {k: getattr(value, "st_" + k) for k in ("dev", "ino", "size", "mtime_ns", "ctime_ns")}


def write_json(path, value):
    with Path(path).open("xb") as handle:
        handle.write(encoded(value)); handle.flush(); os.fsync(handle.fileno())


def stable_source(path, expected=None, *, limit=32 << 20):
    path = Path(path).resolve(strict=True)
    before = signature(path)
    require(before["size"] <= limit, "Unexpected large source; checkpoints/data prohibited")
    raw = path.read_bytes()
    require(before == signature(path), "Source changed during read: " + str(path))
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, "Bound source SHA differs: " + str(path))
    return {"path": str(path), "sha256": digest, "bytes": len(raw), "five_stat": before}, raw


def percentile_linear(values, probability):
    require(bool(values) and 0 <= probability <= 1, "Invalid quantile input")
    ordered = sorted(values)
    at = (len(ordered) - 1) * probability
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def window_member(target_after, window):
    if window == "all":
        return True
    if window == "early":
        return target_after <= 2_000_000
    if window == "late":
        return target_after > 8_000_000
    raise ValueError("Undeclared exposure window")


def describe(values, threshold):
    require(bool(values) and all(math.isfinite(v) and v >= 0 for v in values), "Finite nonnegative complete norms required")
    n = sum(v > threshold for v in values)
    return {"n_steps": len(values), "n_strictly_above_threshold": n,
            "threshold_exceedance_fraction": n / len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p90": percentile_linear(values, .9), "max": max(values)}


def validate_events(events, config, summary, expected_steps=306, final_target=10_000_000, final_replay=500_000):
    """Fail closed on missing/duplicate/reordered steps, partial budgets or resumes."""
    require(bool(events) and events[0].get("event") == "start" and events[-1].get("event") == "finish", "Complete start/finish chain required")
    event_counts = Counter(e.get("event") for e in events)
    require(set(event_counts) <= {"start", "finish", "train_step", "validation", "best_target_validation", "checkpoint"}
            and event_counts["start"] == event_counts["finish"] == 1, "Unknown/error/stop/resume event or repeated lifecycle")
    require(events[0]["config"] == config == summary["config"], "Actual event/summary/config differ")
    require(summary["status"] == events[-1]["status"] == "completed", "Partial run prohibited")
    require(summary["provenance"]["torch_version"] == "2.3.0a0+6ddf5cf85e.nv24.04", "Original installed Torch evidence differs")
    require(events[0]["provenance"] == summary["provenance"], "Actual original provenance differs")
    require(events[-1]["counters"] == summary["counters"], "Finish/summary counters differ")
    counters = {key: 0 for key in TOKEN_KEYS}
    require(events[0]["counters"]["step"] == 0 and all(events[0]["counters"][k] == 0 for k in TOKEN_KEYS), "Nonzero initial committed path")
    train = [event for event in events if event["event"] == "train_step"]
    require(len(train) == expected_steps, "Incomplete/extra train event count")
    per_step = summary["target_tokens_per_step"]
    require(type(per_step) is int and per_step > 0, "Actual target-step capacity missing")
    ratio = Fraction(str(config["replay_ratio"]))
    for index, event in enumerate(train, 1):
        require(event["step"] == event["counters"]["step"] == index, "Train step chain differs")
        wanted = min(per_step, final_target - counters["target_prediction_tokens"])
        require(wanted > 0 and event["step_counts"]["target_prediction_tokens"] == wanted, "Target exposure step differs")
        for key in TOKEN_KEYS:
            value = event["step_counts"][key]
            require(type(value) is int and value >= 0, "Noninteger/negative token delta")
            counters[key] += value
            require(event["counters"][key] == counters[key], "Per-step token delta does not close")
        require(counters["replay_prediction_tokens"] == counters["target_prediction_tokens"] * ratio.numerator // ratio.denominator, "Integer cumulative replay budget differs")
        norm = event["grad_norm"]
        require(type(norm) in (int, float) and math.isfinite(norm) and norm >= 0, "Missing/nonfinite logged preclip norm")
    require(train[-1]["counters"]["step"] == summary["counters"]["step"] == expected_steps, "Final step differs")
    require(counters["target_prediction_tokens"] == config["target_prediction_tokens"] == final_target
            and counters["replay_prediction_tokens"] == final_replay, "Full target/replay exposure required")
    require(all(summary["counters"][key] == counters[key] for key in TOKEN_KEYS), "Final source summary token closure differs")
    return train, dict(event_counts)


def aggregate(individuals):
    result = []
    for method in METHODS:
        for weight in LAMBDAS:
            for window in WINDOWS:
                selected = [r for r in individuals if r["method"] == method and r["lambda"] == weight and r["window"] == window]
                require(len(selected) == 3 and sorted(r["pipeline_seed"] for r in selected) == list(SEEDS), "All three pipeline seeds required per declared cell")
                row = {"method": method, "lambda": weight, "window": window, "n_pipeline_seeds": 3,
                       "n_steps_per_run": selected[0]["n_steps"], "threshold": selected[0]["threshold"]}
                require(all(r["n_steps"] == row["n_steps_per_run"] and r["threshold"] == row["threshold"] for r in selected), "Cell windows/threshold differ")
                for metric in METRICS:
                    row[metric + "_mean"] = statistics.mean(r[metric] for r in selected)
                    row[metric + "_sample_sd"] = statistics.stdev(r[metric] for r in selected)
                result.append(row)
    return result


def physical_accounting(rows):
    physical = {}
    for row in rows:
        key = row["actual_run_id"]
        if key in physical:
            require(physical[key] == row, "Conflicting evidence for aliased actual run")
        else:
            physical[key] = row
    return list(physical.values())


def write_csv(path, rows):
    require(bool(rows), "Empty output prohibited")
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def semantic_evidence(train_raw, clip_raw):
    train, clip = train_raw.decode(), clip_raw.decode()
    ast.parse(train); ast.parse(clip)
    require("grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm if config.max_grad_norm else float(\"inf\"))" in train,
            "Frozen logged norm assignment differs")
    require("grad_norm=float(grad_norm)" in train and train.index("grad_norm = torch.nn.utils.clip_grad_norm_") < train.index("            optimizer.step()"), "Return/logging order differs")
    require("total_norm = torch.linalg.vector_norm" in clip and "clip_coef = max_norm / (total_norm + 1e-6)" in clip
            and "return total_norm" in clip and clip.index("total_norm =") < clip.index("clip_coef =") < clip.index("return total_norm"), "Captured preclip total norm semantics differ")
    return {"logged_value": "total pre-clipping L2 norm returned by clip_grad_norm_", "train_assignment_line": next(i for i, line in enumerate(train.splitlines(), 1) if "grad_norm = torch.nn.utils.clip_grad_norm_" in line),
            "train_logging_line": next(i for i, line in enumerate(train.splitlines(), 1) if "grad_norm=float(grad_norm)" in line),
            "captured_source_returns_total_norm": True, "threshold_comparison": "strict logged norm > configured max_grad_norm",
            "not_exact_floating_point_clip_activation": "clip coefficient uses total_norm+1e-6; no epsilon-boundary activation inference"}


def make_figure(output, individuals, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ("#377eb8", "#ff7f00", "#4daf4a", "#e41a1c", "#984ea3", "#8c564b")))
    fig, axes = plt.subplots(3, 2, figsize=(7, 6.6), sharex=True, constrained_layout=False)
    metrics = ("threshold_exceedance_fraction", "mean")
    labels = {"all": "All 306 updates", "early": "Early: target ≤ 2M", "late": "Late: target > 8M"}
    for i, window in enumerate(WINDOWS):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for method in METHODS:
                selected = sorted([r for r in groups if r["method"] == method and r["window"] == window], key=lambda r: r["lambda"])
                ax.errorbar([r["lambda"] for r in selected], [r[metric + "_mean"] for r in selected],
                            yerr=[r[metric + "_sample_sd"] for r in selected], color=colors[method], marker="o", ms=2.5, lw=.9, elinewidth=.6, capsize=1.5, label=method)
                for seed, marker in zip(SEEDS, (".", "x", "+")):
                    individual = sorted([r for r in individuals if r["method"] == method and r["window"] == window and r["pipeline_seed"] == seed], key=lambda r: r["lambda"])
                    ax.scatter([r["lambda"] for r in individual], [r[metric] for r in individual], s=9, marker=marker, color=colors[method], alpha=.5)
            ax.set_xscale("log"); ax.set_xticks(LAMBDAS, labels=(".01", ".05", ".1", ".5", "1", "5", "10"))
            ax.tick_params(labelsize=7); ax.grid(alpha=.2, linewidth=.5)
            if j == 0:
                ax.set_ylim(-.035, 1.065); ax.set_ylabel(labels[window] + "\nNorm > threshold fraction", fontsize=8)
            else:
                ax.set_yscale("log"); ax.axhline(1, color=".4", ls="--", lw=.6); ax.set_ylabel("Mean preclip norm (log)", fontsize=8)
            if i == 2: ax.set_xlabel("Replay loss weight λ (log)", fontsize=8)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(.52, .955))
    fig.suptitle("Post-test descriptive gradient diagnostic: fixed A grid", fontsize=10, y=.987)
    fig.text(.5, .018, "All three pipeline seeds; mean ± sample SD. Steps/windows and λ cells share random sources.\nThreshold exceedance is not an Adam-update or causal-mechanism measure.", ha="center", fontsize=7)
    fig.subplots_adjust(left=.105, right=.985, top=.9, bottom=.115, hspace=.3, wspace=.4)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / ("gradient_diagnostic." + suffix), dpi=220)
    plt.close(fig)


def analyze(output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(key) == "2" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Explicit empty CVD/two CPU threads required")
    output.mkdir(parents=True, exist_ok=False)
    capture = output / "source_bytes"; capture.mkdir()
    sources = []; source_bytes = 0
    def source(path, expected=None, snapshot_name=None):
        nonlocal source_bytes
        proof, raw = stable_source(path, expected)
        source_bytes += len(raw); sources.append(proof)
        if snapshot_name:
            (capture / snapshot_name).parent.mkdir(parents=True, exist_ok=True)
            with (capture / snapshot_name).open("xb") as handle: handle.write(raw)
        return proof, raw
    start_verify = time.perf_counter()
    protocol_proof, raw = source(PROTOCOL, PROTOCOL_SHA, "protocol.json"); protocol = json.loads(raw)
    producer_proof, _ = source(__file__, snapshot_name="analyze_A_gradient_clipping_posttest.py")
    semantics = protocol["source_semantics"]
    train_proof, train_raw = source(ROOT / semantics["train_source"]["path"], semantics["train_source"]["sha256"], "train.py")
    clip_proof, clip_raw = source(ROOT / semantics["clip_source"]["path"], semantics["clip_source"]["sha256"], "installed_torch_clip_grad_norm_source.py")
    evidence = semantic_evidence(train_raw, clip_raw)
    manifest_proof, raw = source(ROOT / protocol["manifest"]["path"], protocol["manifest"]["sha256"], "scope_manifest.json")
    records = json.loads(raw)["records"]
    expected_grid = {(method, weight, seed) for method in METHODS for weight in LAMBDAS for seed in SEEDS}
    require(len(records) == 126 and {(r["group"], float(r["weight"]), r["training_seed"]) for r in records} == expected_grid, "Full declared 6×7×3 scope required")
    require(sum(r["is_training_alias"] for r in records) == 18, "Exactly eighteen existing-main training references required")
    individuals, steps, run_sources, costs, cached_runs = [], [], [], [], {}
    for record in records:
        require(record["kind"] == "final" and record["data_family"] == "main_v3" and record["model_tag"] == "pythia-410m", "Foreign endpoint/model/data in A grid")
        actual = record["actual_run_id"]
        cfg_path, sum_path = record["config_path"], record["summary_path"]
        if actual not in cached_runs:
            cfg_proof, raw = source(cfg_path, record["input_sha256"][cfg_path], actual + "/config.json"); cfg = json.loads(raw)
            sum_proof, raw = source(sum_path, record["input_sha256"][sum_path], actual + "/summary.json"); summary = json.loads(raw)
            require(cfg["max_grad_norm"] == 1.0, "Unexplained actual clipping threshold difference")
            event_path = Path(cfg["output_dir"]) / "events.jsonl"
            require(event_path.parent.name == actual, "Actual run/event namespace differs")
            event_proof, raw = source(event_path, snapshot_name=actual + "/events.jsonl")
            events = [json.loads(line) for line in raw.splitlines() if line.strip()]
            train, event_counts = validate_events(events, cfg, summary)
            cached_runs[actual] = cfg, summary, train, cfg_proof, sum_proof, event_proof, event_counts
        cfg, summary, train, cfg_proof, sum_proof, event_proof, event_counts = cached_runs[actual]
        require(cfg["seed"] == record["training_seed"] and cfg["replay_weight"] == record["weight"] and cfg["replay_ratio"] == record["ratio"] == .05, "Manifest/config seed/λ/b differ")
        require((cfg["replay_objective"] == "ce") == (record["group"] in ("B1", "B3")), "Manifest/config objective differs")
        identity = {"method": record["group"], "lambda": float(record["weight"]), "pipeline_seed": record["training_seed"],
                    "logical_record_id": record["id"], "logical_run_id": record["run_id"], "actual_run_id": actual,
                    "existing_main_training_reference": record["is_training_alias"], "b": .05, "threshold": cfg["max_grad_norm"],
                    "actual_final_step": summary["counters"]["step"], "actual_final_target_prediction_tokens": summary["counters"]["target_prediction_tokens"],
                    "actual_final_replay_prediction_tokens": summary["counters"]["replay_prediction_tokens"],
                    "config_sha256": cfg_proof["sha256"], "summary_sha256": sum_proof["sha256"], "events_sha256": event_proof["sha256"]}
        run_sources.append({**identity, "actual_event_count": sum(event_counts.values()), "train_event_count": len(train), "event_types": event_counts,
                            "config": cfg_proof, "summary": sum_proof, "events": event_proof})
        for window in WINDOWS:
            norms = [event["grad_norm"] for event in train if window_member(event["counters"]["target_prediction_tokens"], window)]
            individuals.append({**identity, "window": window, **describe(norms, cfg["max_grad_norm"])})
        for event in train:
            steps.append({"logical_record_id": record["id"], "actual_run_id": actual, "method": identity["method"], "lambda": identity["lambda"],
                          "pipeline_seed": identity["pipeline_seed"], "step": event["step"], "target_prediction_tokens_after": event["counters"]["target_prediction_tokens"],
                          "replay_prediction_tokens_after": event["counters"]["replay_prediction_tokens"], "preclip_total_norm": event["grad_norm"], "threshold": cfg["max_grad_norm"],
                          "strictly_above_threshold": event["grad_norm"] > cfg["max_grad_norm"],
                          "in_early_window": window_member(event["counters"]["target_prediction_tokens"], "early"), "in_late_window": window_member(event["counters"]["target_prediction_tokens"], "late")})
        costs.append({"actual_run_id": actual, "existing_main_reference": record["is_training_alias"],
                      **{key: summary["counters"][key] for key in ("setup_seconds", "training_seconds", "validation_seconds", "checkpoint_seconds")}})
    require(len(cached_runs) == 126 and len(steps) == 126 * 306, "Unique original training/event scope differs")
    verification_wall = time.perf_counter() - start_verify
    start_statistics = time.perf_counter(); groups = aggregate(individuals); physical = physical_accounting(costs)
    write_csv(output / "per_run_windows.csv", individuals); write_csv(output / "method_lambda_windows.csv", groups)
    write_csv(output / "all_train_steps.csv", steps); write_csv(output / "existing_physical_training_parent_timers.csv", physical)
    write_json(output / "run_source_manifest.json", {"records": run_sources, "scope_manifest": manifest_proof})
    statistics_wall = time.perf_counter() - start_statistics
    start_plot = time.perf_counter(); make_figure(output, individuals, groups); plot_wall = time.perf_counter() - start_plot
    for item in sources:
        require(signature(item["path"]) == item["five_stat"], "Original completed source changed after analysis")
    outputs = [stable_source(path)[0] for path in output.iterdir() if path.is_file()]
    result = {"schema_version": 1, "status": "complete_declared_posttest_A_gradient_log_diagnostic", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": protocol_proof, "producer": producer_proof, "train_source": train_proof, "captured_installed_clip_source": clip_proof,
              "semantics": evidence, "logical_conditions": 126, "actual_training_runs": len(physical), "existing_main_training_references": 18,
              "previously_completed_supplement_A_training_runs": 108, "new_training_or_inference_runs": 0,
              "train_events": len(steps), "per_run_window_rows": len(individuals), "method_lambda_window_rows": len(groups),
              "source_bytes_read_and_snapshotted": source_bytes, "source_file_count": len(sources),
              "window_step_counts": {w: sorted(set(r["n_steps"] for r in individuals if r["window"] == w)) for w in WINDOWS},
              "total_steps_strictly_above_threshold": sum(row["strictly_above_threshold"] for row in steps),
              "all_run_window_fraction_range": [min(row["threshold_exceedance_fraction"] for row in individuals), max(row["threshold_exceedance_fraction"] for row in individuals)],
              "quantile_convention": "linear interpolation at (n-1)*p; p=.9", "between_run_SD": "sample SD (ddof=1) of three pipeline-seed run summaries",
              "randomness_boundary": "Same three pipeline seeds across lambda, reused main18; steps/windows are correlated descriptions, not independent replications",
              "inference_boundary": "Post-test descriptive covariation with lambda, no Adam update magnitude/direction/causal mechanism or new selection claim",
              "GPU_model_dataset_test_or_behavior_result_files_opened": False,
              "nested_wall_seconds": {"source_verification_parsing_and_snapshot": verification_wall, "statistics_and_tabular_write": statistics_wall, "plot": plot_wall},
              "nested_timers_are_part_of_whole_child_process_do_not_add_again": True,
              "cost_accounting": {"only_this_CPU_analysis_is_new": True, "existing_training_parent_timer_sum_by_unique_actual_run_id": {key: sum(row[key] for row in physical) for key in ("setup_seconds", "training_seconds", "validation_seconds", "checkpoint_seconds")},
                                  "existing_main18_are_not_new_training_cost_or_independent_extra_samples": True,
                                  "training_parent_sums_are_NOT_research_elapsed_and_NOT_pure_GPU_time": True},
              "output_proofs_before_complete": outputs}
    write_json(output / "COMPLETE.json", result)
    print(json.dumps({"status": result["status"], "logical_conditions": 126, "train_events": len(steps), "output": str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    analyze(args.output_dir.resolve())


if __name__ == "__main__":
    main()
