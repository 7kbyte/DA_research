import copy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "posttest_factorial_A", Path(__file__).resolve().parents[1] / "scripts/analyze_posttest_factorial_A.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture(additive=False):
    endpoints, pairs = [], []
    for weight in analysis.WEIGHTS:
        for index, seed in enumerate(analysis.SEEDS):
            effects = {"B1": 0, "B2": 2, "B3": 5,
                       "B4": 7 if additive else 10 + index}
            four = {}
            for group in analysis.GROUPS:
                row = {
                    "role": "supplement_A", "phase_name": "supplement_A",
                    "model_tag": "pythia-410m", "data_family": "main_v3", "checkpoint": "final",
                    "group": group, "weight": str(weight), "training_seed": str(seed),
                    "ratio": ".05", "learning_rate": ".00003", "baseline_group": "shared",
                    "target_prediction_tokens": "10000000", "replay_prediction_tokens": "500000",
                    "full_training_target_prediction_tokens": "10000000",
                    "full_training_replay_prediction_tokens": "500000",
                    "record_id": f"record_{group}_{weight}_{seed}", "run_id": f"run_{group}_{weight}_{seed}",
                    "actual_run_id": f"physical_{group}_{weight}_{seed}", "inference_job_id": f"infer_{group}_{weight}_{seed}",
                    "is_training_alias": "False", "is_inference_alias": "False",
                }
                for metric_index, metric in enumerate(analysis.METRICS):
                    row[metric] = str(weight * 3 + seed + effects[group] * (metric_index + 1))
                endpoints.append(row); four[group] = row
            for comparison in analysis.COMPARISONS:
                left, right = comparison.split("-")
                row = {
                    "comparison": comparison, "weight": str(weight), "training_seed": str(seed),
                    "training_seed_a": str(seed), "training_seed_b": str(seed),
                    "run_a": four[left]["run_id"], "run_b": four[right]["run_id"],
                    "target_tokens_a": "10000000", "target_tokens_b": "10000000",
                    "equal_checkpoint_exposure": "True",
                }
                row.update({m: str(float(four[left][m]) - float(four[right][m])) for m in analysis.METRICS})
                pairs.append(row)
    return endpoints, pairs


def test_additive_source_objective_has_zero_interaction():
    points, stats, checks = analysis.interactions(*fixture(additive=True))
    assert len(points) == 126 and len(stats) == 42
    assert all(abs(r["interaction"]) < 1e-12 for r in points)
    assert sum(r["checked_scalar_equalities"] for r in checks) == 756


def test_nonadditive_contrast_and_sample_sd_use_three_paired_seeds():
    points, stats, checks = analysis.interactions(*fixture())
    cell = [r for r in stats if r["weight"] == .05 and r["metric"] == "g_new"]
    assert cell[0]["mean"] == pytest.approx(4)
    assert cell[0]["std_ddof1"] == pytest.approx(1)
    assert cell[0]["n"] == 3
    assert all(r["max_abs_error_across_four_terms_and_two_identities"] < 1e-12 for r in checks)
    assert points[0]["B1_target_prediction_tokens"] == "10000000"
    assert points[0]["B4_actual_run_id"].startswith("physical_B4")


def test_existing_paired_effect_corruption_fails_independent_identity():
    endpoints, pairs = fixture()
    pairs[0]["g_new"] = str(float(pairs[0]["g_new"]) + .1)
    with pytest.raises(ValueError, match="algebra differs"):
        analysis.interactions(endpoints, pairs)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_lambda"])
def test_complete_exact_lambda_seed_grid_required(mutation):
    endpoints, pairs = fixture()
    if mutation == "missing":
        endpoints.pop()
    elif mutation == "duplicate":
        endpoints.append(copy.deepcopy(endpoints[0]))
    else:
        endpoints[0]["weight"] = ".02"
    with pytest.raises(ValueError, match="grid|Duplicate"):
        analysis.interactions(endpoints, pairs)


@pytest.mark.parametrize("field,value", [
    ("replay_prediction_tokens", "499999"), ("target_prediction_tokens", "9999999"),
    ("checkpoint", "best"), ("baseline_group", "different"), ("ratio", ".1"),
])
def test_exposure_policy_and_endpoint_identity_are_not_interchangeable(field, value):
    endpoints, pairs = fixture(); endpoints[0][field] = value
    with pytest.raises(ValueError):
        analysis.interactions(endpoints, pairs)


@pytest.mark.parametrize("field,value", [("training_seed_b", "22"), ("run_b", "other_run")])
def test_existing_pairs_must_bind_same_actual_source_ids_and_seed(field, value):
    endpoints, pairs = fixture(); pairs[0][field] = value
    with pytest.raises(ValueError, match="identity|seeds"):
        analysis.interactions(endpoints, pairs)


def test_nonfinite_values_rejected():
    endpoints, pairs = fixture(); endpoints[0]["g_new"] = "nan"
    with pytest.raises(ValueError, match="Nonfinite"):
        analysis.interactions(endpoints, pairs)
