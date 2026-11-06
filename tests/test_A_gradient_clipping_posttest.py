import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_A_gradient_clipping_posttest.py"
SPEC = importlib.util.spec_from_file_location("A_gradient_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def tiny_complete_chain():
    config = {"replay_ratio": .05, "target_prediction_tokens": 10_000_000}
    provenance = {"torch_version": "2.3.0a0+6ddf5cf85e.nv24.04"}
    current = {key: 0 for key in diagnostic.TOKEN_KEYS}
    current["step"] = 0
    events = [{"event": "start", "config": config, "provenance": provenance, "counters": copy.deepcopy(current)}]
    for step in range(1, 6):
        deltas = {f"{part}_{suffix}": count + extra
                  for part, count in (("target", 2_000_000), ("replay", 100_000))
                  for suffix, extra in (("prediction_tokens", 0), ("raw_input_tokens", 2), ("padded_input_tokens", 10))}
        for key, value in deltas.items(): current[key] += value
        current["step"] = step
        events.append({"event": "train_step", "step": step, "step_counts": deltas,
                       "counters": copy.deepcopy(current), "grad_norm": float(step - 1)})
    summary = {"config": config, "status": "completed", "provenance": provenance,
               "counters": copy.deepcopy(current), "target_tokens_per_step": 2_000_000}
    events.append({"event": "finish", "status": "completed", "counters": copy.deepcopy(current)})
    return events, config, summary


def test_manual_descriptions_and_linear_quantile():
    result = diagnostic.describe([0., 1., 2., 3.], 1.)
    assert result == {"n_steps": 4, "n_strictly_above_threshold": 2,
                      "threshold_exceedance_fraction": .5, "mean": 1.5,
                      "median": 1.5, "p90": 2.7, "max": 3.}
    assert diagnostic.percentile_linear([1., 2., 3., 4.], .9) == pytest.approx(3.7)
    assert diagnostic.describe([.9999995, 1., 1.0000001], 1.)["threshold_exceedance_fraction"] == 1/3


def test_fixed_window_boundaries_after_update():
    assert diagnostic.window_member(2_000_000, "early")
    assert not diagnostic.window_member(2_000_001, "early")
    assert not diagnostic.window_member(8_000_000, "late")
    assert diagnostic.window_member(8_000_001, "late")
    assert diagnostic.window_member(10_000_000, "all")
    with pytest.raises(ValueError): diagnostic.window_member(1, "posthoc")


def test_complete_step_and_counter_chain():
    events, config, summary = tiny_complete_chain()
    train, counts = diagnostic.validate_events(events, config, summary, expected_steps=5)
    assert len(train) == 5 and counts["train_step"] == 5
    assert sum(diagnostic.window_member(e["counters"]["target_prediction_tokens"], "early") for e in train) == 1
    assert sum(diagnostic.window_member(e["counters"]["target_prediction_tokens"], "late") for e in train) == 1


@pytest.mark.parametrize("failure", ["missing", "duplicate_step", "reordered", "counter", "nan", "resume", "stopped", "partial"])
def test_reject_incomplete_or_inconsistent_chain(failure):
    events, config, summary = tiny_complete_chain()
    if failure == "missing": events.pop(2)
    if failure == "duplicate_step": events[2] = copy.deepcopy(events[1])
    if failure == "reordered": events[2], events[3] = events[3], events[2]
    if failure == "counter": events[2]["counters"]["replay_prediction_tokens"] += 1
    if failure == "nan": events[3]["grad_norm"] = float("nan")
    if failure == "resume": events[0]["counters"]["step"] = 1
    if failure == "stopped": events.insert(3, {"event": "stop"})
    if failure == "partial": summary["status"] = events[-1]["status"] = "deadline"
    with pytest.raises(ValueError): diagnostic.validate_events(events, config, summary, expected_steps=5)


def test_three_seed_mean_and_sample_SD_all_cells():
    rows = []
    for method in diagnostic.METHODS:
        for weight in diagnostic.LAMBDAS:
            for window in diagnostic.WINDOWS:
                for seed, value in zip(diagnostic.SEEDS, (1., 2., 3.)):
                    row = dict(method=method, **{"lambda": weight}, window=window, pipeline_seed=seed, n_steps=5, threshold=1.)
                    row.update({metric: value for metric in diagnostic.METRICS}); rows.append(row)
    groups = diagnostic.aggregate(rows)
    assert len(groups) == 6 * 7 * 3
    assert all(row["mean_mean"] == 2. and row["mean_sample_sd"] == 1. for row in groups)
    with pytest.raises(ValueError): diagnostic.aggregate(rows[:-1])


def test_alias_is_one_physical_training_cost():
    row = {"actual_run_id": "one", "existing_main_reference": True, "training_seconds": 100.}
    assert diagnostic.physical_accounting([row, copy.deepcopy(row)]) == [row]
    with pytest.raises(ValueError): diagnostic.physical_accounting([row, {**row, "training_seconds": 101.}])


def test_source_sha_and_large_input_rejection(tmp_path):
    path = tmp_path / "source.txt"; path.write_bytes(b"abcd")
    with pytest.raises(ValueError): diagnostic.stable_source(path, limit=3)
    with pytest.raises(ValueError): diagnostic.stable_source(path, "0" * 64)
