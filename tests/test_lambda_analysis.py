"""Hand-computable validation selection, alias identity, and completeness guards."""
import copy
import json
from pathlib import Path

import pytest

from replay_study.analysis import sha256, training_validation_execution
from replay_study.lambda_analysis import (PHASE, collect_lambda_analysis,
                                         select_lambdas, write_lambda_analysis)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def three(target, c4=None, wiki=None):
    return {k: {"nll": v, "sum_nll": v * 10, "prediction_tokens": 10,
                "complete_corpus": True, "precision": "bf16", "kl": 0.}
            for k, v in zip(("target", "c4", "wikitext"),
                            (target, target if c4 is None else c4, target if wiki is None else wiki))}


def summary(config, metrics):
    return {"status": "completed", "config": config, "checkpoint_path": config["output_dir"] + "/latest.pt",
            "counters": {"step": 306, "target_prediction_tokens": 10_000_000, "replay_prediction_tokens": 500_000},
            "validation_trajectory": [{"step": 306, "target_prediction_tokens": 10_000_000,
                                       "replay_prediction_tokens": 500_000, "metrics": metrics}],
            "provenance": {"device": "cuda:0", "gpu": "H100", "precision": "bf16",
                "torch_version": "fixture", "cuda_version": "fixture", "attention_implementation": "sdpa",
                "attention_backend_policy": "auto", "sdpa_backends_enabled": {"flash": True, "math": True, "memory_efficient": True, "cudnn": True},
                "deterministic_algorithms": True, "deterministic_warn_only": False,
                "allow_tf32_matmul": True, "allow_tf32_cudnn": True, "cudnn_deterministic": True,
                "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8", "model_weight_dtype": "torch.float32",
                "target": {"sha256_tensors": "frozen-target"}, "replay": {"sha256_tensors": "frozen-bank"},
                "source_sha256": {"train.py": "frozen-train", "objectives.py": "frozen-loss"}}}


def fixture(tmp_path):
    rows, parents = [], []
    for group in ("B4", "M"):
        for seed in (11, 22, 33):
            actual = f"formal_{group}_{seed}"
            config = {"model_revision": "pinned", "model_name_or_path": "same-model", "target_path": "target.pt",
                      "replay_path": f"bank_{group}_{seed}.pt", "seed": seed, "target_prediction_tokens": 10_000_000,
                      "replay_ratio": .05, "replay_weight": .05, "learning_rate": 3e-5,
                      "deterministic_algorithms": True, "precision": "bf16", "device": "cuda:0",
                      "run_label": actual, "output_dir": str(tmp_path / "runs" / actual),
                      "validation_paths": {"target": "val.pt"}, "microbatch_size": 16}
            cp = write(tmp_path / (actual + ".json"), config)
            # Main .05 is always inferior in the predeclared balanced score.
            sp = write(tmp_path / (actual + "_summary.json"), summary(config, three(1., 3., 3.)))
            parent = {"id": actual, "group": group, "seed": seed, "ratio": .05, "weight": .05,
                      "learning_rate": 3e-5, "phase": "formal", "active": True, "config_path": str(cp),
                      "summary_path": str(sp), "output_dir": config["output_dir"]}
            parents.append(parent)
            for weight in (.05, .1):
                condition = f"supp_{group}_{seed}_{weight}"
                child = copy.deepcopy(config)
                if weight == .05:
                    child_cp, child_sp = cp, sp
                else:
                    child.update(replay_weight=weight, run_label=condition, output_dir=str(tmp_path / "runs" / condition))
                    child_cp = write(tmp_path / (condition + ".json"), child)
                    # S=.5 vs .05 cell S=0. Hand calculation below adjusts values in direct scoring tests.
                    child_sp = write(tmp_path / (condition + "_summary.json"), summary(child, three(1.5, 2., 2.)))
                rows.append({"id": condition, "phase": PHASE, "group": group, "seed": seed, "ratio": .05,
                    "weight": weight, "learning_rate": 3e-5, "active": False, "launch_status": "prepared_not_authorized",
                    "source_formal_run_id": actual, "source_config_path": str(cp), "source_config_sha256": sha256(cp),
                    "config_path": str(child_cp), "config_sha256": sha256(child_cp), "summary_path": str(child_sp),
                    "output_dir": child["output_dir"], "execution_kind": "alias" if weight == .05 else "train",
                    "alias_of": actual if weight == .05 else None, "counted_as_new_training": weight != .05})
    registry = write(tmp_path / "registry.json", rows)
    prereg = write(tmp_path / "prereg.json", {"registry_sha256": sha256(registry), "groups": ["B4", "M"],
        "seeds": [11, 22, 33], "lambda_grid": [.05, .1], "target_prediction_tokens_per_run": 10_000_000,
        "replay_ratio": .05, "learning_rate": 3e-5})
    parent = write(tmp_path / "parent_registry.json", parents)
    status = write(tmp_path / "status.json", {"active_registry": str(parent)})
    initial = summary(config, three(2.))
    baseline = write(tmp_path / "baseline.json", {"revision": "pinned", "checkpoint": None,
        "execution_provenance": training_validation_execution(initial), "corpora": three(2.)})
    args = {"preregistered_registry_path": registry, "preregistration_path": prereg,
            "parent_registry_path": parent, "protocol_status_path": status}
    return registry, baseline, args


def test_alias_keeps_actual_identity_does_not_add_training_or_replicates(tmp_path):
    registry, baseline, args = fixture(tmp_path)
    bundle = collect_lambda_analysis(registry, baseline, **args)
    assert bundle["manifest"]["complete"] and len(bundle["validation_endpoints"]) == 12
    assert bundle["manifest"]["main_alias_conditions"] == 6
    assert bundle["manifest"]["new_training_conditions"] == 6
    aliases = [r for r in bundle["validation_endpoints"] if r["execution_kind"] == "alias"]
    assert all(r["actual_run_id"] == r["alias_of"] != r["condition_id"] and r["phase"] == PHASE for r in aliases)
    assert all(r["n"] == 3 for r in bundle["endpoint_statistics"] if r["metric"] == "g_new")
    assert all(r["incremental_training_cost_factor"] == 0 for r in bundle["observed_costs"] if r["execution_kind"] == "alias")
    assert all(r["future_final_evaluation_relative_path"] == r["actual_run_id"] + "/final.json" for r in bundle["run_identities"])
    assert {r["weight"] for r in bundle["lambda_selections"]} == {.1}


def test_predeclared_score_weights_shared_seed_selection_and_exact_ties():
    prereg = {"groups": ["M"], "seeds": [11, 22, 33], "lambda_grid": [.01, .05, .1]}
    endpoints = []
    # .01: strong target adaptation but too much old forgetting -> S=1.
    # .05 and .1: S=-.5, exact tie -> smaller .05; no per-seed picking.
    for weight, gain, c4, wiki in [(.01, 2., 2., 4.), (.05, 1., .25, .75), (.1, .5, 0., 0.)]:
        for seed in prereg["seeds"]:
            endpoints.append({"group": "M", "seed": seed, "weight": weight, "g_new": gain,
                "f_old_c4": c4, "f_old_wikitext": wiki, "condition_id": f"c{seed}{weight}", "actual_run_id": f"a{seed}{weight}"})
    scores, chosen, status = select_lambdas(endpoints, [], prereg)
    assert [r["mean"] for r in scores] == [1., -.5, -.5]
    assert chosen[0]["weight"] == .05 and chosen[0]["n"] == 3
    assert chosen[0]["shared_across_seeds"] and chosen[0]["test_scores_used"] is False
    # A genuinely smaller score must win even if the difference is tiny.
    close = copy.deepcopy(endpoints)
    for row in close:
        if row["weight"] == .1:
            row["g_new"] += 1e-12
    assert select_lambdas(close, [], prereg)[1][0]["weight"] == .1
    # Individual seeds disagree; choose one lambda from the mean, not one per seed.
    crossing = [dict(r) for r in endpoints if r["weight"] != .01]
    values = {.05: [-2., 1., 1.], .1: [-1., 0., 0.]}
    for row in crossing:
        row.update(g_new=-values[row["weight"]][prereg["seeds"].index(row["seed"])], f_old_c4=0., f_old_wikitext=0.)
    assert select_lambdas(crossing, [], {**prereg,"lambda_grid":[.05,.1]})[1][0]["weight"] == .1
    # Missing the unfavorable lambda is still forbidden: don't select a favorable subset.
    _, chosen, status = select_lambdas(endpoints[3:], [], prereg)
    assert chosen == [] and status[0]["selection_ready"] is False


def test_incomplete_grid_requires_partial_and_cannot_select_affected_method(tmp_path):
    registry, baseline, args = fixture(tmp_path)
    rows = json.loads(registry.read_text())
    victim = next(r for r in rows if r["group"] == "M" and r["weight"] == .1 and r["seed"] == 33)
    Path(victim["summary_path"]).unlink()
    bundle = collect_lambda_analysis(registry, baseline, **args)
    assert bundle["manifest"]["complete"] is False
    assert [r["group"] for r in bundle["lambda_selections"]] == ["B4"]
    with pytest.raises(ValueError, match="incomplete"):
        write_lambda_analysis(bundle, tmp_path / "out", plots=False)
    assert not (tmp_path / "out").exists()
    write_lambda_analysis(bundle, tmp_path / "out", allow_partial=True, plots=False)
    result = json.loads((tmp_path / "out/validation_lambda_selection.json").read_text())
    assert result["test_scores_accessed"] is False and result["complete"] is False


def test_no_best_endpoint_or_wrong_execution_or_scientific_override(tmp_path):
    registry, baseline, args = fixture(tmp_path)
    rows = json.loads(registry.read_text())
    victim = next(r for r in rows if r["execution_kind"] == "train")
    summary_path = Path(victim["summary_path"])
    original = json.loads(summary_path.read_text())
    wrong = copy.deepcopy(original)
    wrong["validation_trajectory"][-1]["target_prediction_tokens"] = 5_000_000
    wrong["best_target_validation"] = {"nll": 0., "target_prediction_tokens": 1}
    write(summary_path, wrong)
    with pytest.raises(ValueError, match="final fixed-exposure"):
        collect_lambda_analysis(registry, baseline, **args)
    wrong = copy.deepcopy(original)
    wrong["provenance"]["sdpa_backends_enabled"]["cudnn"] = False
    write(summary_path, wrong)
    with pytest.raises(ValueError, match="execution policy differs"):
        collect_lambda_analysis(registry, baseline, **args)
    write(summary_path, original)
    runtime = copy.deepcopy(rows)
    victim_runtime = next(r for r in runtime if r["id"] == victim["id"])
    child = json.loads(Path(victim["config_path"]).read_text())
    child["microbatch_size"] = 64
    new_path = write(tmp_path / "runtime_config.json", child)
    victim_runtime.update(config_path=str(new_path), config_sha256=sha256(new_path))
    runtime_path = write(tmp_path / "runtime_registry.json", runtime)
    with pytest.raises(ValueError, match="scientific fields"):
        collect_lambda_analysis(runtime_path, baseline, **args)


def test_runtime_deadline_override_is_allowed_but_alias_or_parent_mix_is_not(tmp_path):
    registry, baseline, args = fixture(tmp_path)
    rows = json.loads(registry.read_text())
    runtime = copy.deepcopy(rows)
    victim = next(r for r in runtime if r["execution_kind"] == "train")
    child = json.loads(Path(victim["config_path"]).read_text())
    child["deadline_utc"] = "2026-11-07T00:00:00Z"
    cp = write(tmp_path / "runtime.json", child)
    victim.update(config_path=str(cp), config_sha256=sha256(cp), active=True, operational_override={"deadline_utc": child["deadline_utc"]})
    path = write(tmp_path / "runtime_registry.json", runtime)
    assert collect_lambda_analysis(path, baseline, **args)["manifest"]["complete"]
    victim["phase"] = "formal"
    write(path, runtime)
    with pytest.raises(ValueError, match="explicit supplement"):
        collect_lambda_analysis(path, baseline, **args)
    status = Path(args["protocol_status_path"])
    write(status, {"active_registry": str(tmp_path / "old_provisional.json")})
    with pytest.raises(ValueError, match="active registry"):
        collect_lambda_analysis(registry, baseline, **args)
