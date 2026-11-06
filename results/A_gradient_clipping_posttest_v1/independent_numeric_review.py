"""Check gradient-only CSV arithmetic with an independent NumPy implementation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"


def rows(path):
    return list(csv.DictReader(path.open()))


complete = json.loads((OUT / "COMPLETE.json").read_text())
for item in complete["output_proofs_before_complete"]:
    path = Path(item["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
steps = rows(OUT / "all_train_steps.csv")
individuals = rows(OUT / "per_run_windows.csv")
groups = rows(OUT / "method_lambda_windows.csv")
assert len(steps) == 126 * 306 and len(individuals) == 126 * 3 and len(groups) == 42 * 3
by_run = {}
for item in steps:
    by_run.setdefault(item["actual_run_id"], []).append(item)
recomputed = {}
max_abs = 0.
comparisons = 0
for row in individuals:
    run_steps = by_run[row["actual_run_id"]]
    assert [int(x["step"]) for x in run_steps] == list(range(1, 307))
    if row["window"] == "all": selected = run_steps
    elif row["window"] == "early": selected = [x for x in run_steps if int(x["target_prediction_tokens_after"]) <= 2_000_000]
    else:
        assert row["window"] == "late"
        selected = [x for x in run_steps if int(x["target_prediction_tokens_after"]) > 8_000_000]
    values = np.array([float(x["preclip_total_norm"]) for x in selected], dtype=np.float64)
    threshold = float(row["threshold"])
    observed = {"n_steps": len(values), "n_strictly_above_threshold": int(np.sum(values > threshold)),
                "threshold_exceedance_fraction": float(np.mean(values > threshold)), "mean": float(np.mean(values)),
                "median": float(np.median(values)), "p90": float(np.quantile(values, .9, method="linear")), "max": float(np.max(values))}
    for key, value in observed.items():
        delta = abs(value - float(row[key])); max_abs = max(max_abs, delta)
        assert delta <= 1e-11, (row["actual_run_id"], row["window"], key, delta)
        comparisons += 1
    key = row["method"], float(row["lambda"]), row["window"], int(row["pipeline_seed"])
    assert key not in recomputed
    recomputed[key] = observed
for row in groups:
    for metric in ("threshold_exceedance_fraction", "mean", "median", "p90", "max"):
        values = np.array([recomputed[(row["method"], float(row["lambda"]), row["window"], seed)][metric] for seed in (11, 22, 33)])
        for suffix, value in (("mean", np.mean(values)), ("sample_sd", np.std(values, ddof=1))):
            delta = abs(float(value) - float(row[metric + "_" + suffix])); max_abs = max(max_abs, delta)
            assert delta <= 1e-11
            comparisons += 1
all_norms = np.array([float(x["preclip_total_norm"]) for x in steps])
assert np.all(all_norms > 1.)
assert len(set(x["actual_run_id"] for x in individuals)) == 126
refs = {x["actual_run_id"] for x in individuals if x["existing_main_training_reference"] == "True"}
assert len(refs) == 18
report = {"schema_version": 1, "status": "passed_independent_gradient_CSV_arithmetic_review", "numpy_version": np.__version__,
          "source": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
          "analysis_COMPLETE": {"path": str(OUT / "COMPLETE.json"), "sha256": hashlib.sha256((OUT / "COMPLETE.json").read_bytes()).hexdigest()},
          "numeric_comparisons": comparisons, "max_absolute_difference": max_abs, "train_steps": len(steps),
          "all_preclip_norms_strictly_above_1": True, "global_min_preclip_norm": float(np.min(all_norms)), "global_max_preclip_norm": float(np.max(all_norms)),
          "all_run_window_fraction": 1.0, "existing_main_training_references": len(refs),
          "no_score_model_dataset_or_checkpoint_reads": True, "independent_method": "NumPy float64 mean/median/linear quantile and ddof1 sample SD from gradient-only per-step CSV"}
with (HERE / "independent_numeric_review.json").open("x") as handle:
    json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
print(json.dumps({key: report[key] for key in ("status", "numeric_comparisons", "max_absolute_difference", "global_min_preclip_norm", "global_max_preclip_norm")}))
