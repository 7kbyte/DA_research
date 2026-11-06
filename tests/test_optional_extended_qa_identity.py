"""Tiny CPU fixtures only: reuse the frozen QA verifier for one run/two endpoints."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qa_existing_temp_fixture", ROOT / "tests/test_reproduction_qa_postrun.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
qa = base.qa


@pytest.fixture
def single(tmp_path, monkeypatch):
    prepared = base.prepared.__wrapped__(tmp_path, monkeypatch)
    expected = prepared["expected"]
    expected["runs"] = expected["runs"][:1]
    expected["expected_endpoint_count"] = 2
    expected["planned_gpus"] = [0]
    plan_path = Path(expected["plan"]["path"])
    plan = qa.read(plan_path)
    for field in ("registry", "queue"):
        path = Path(expected[field]["path"])
        rows = qa.read(path)[:1]
        base.put(path, rows)
        expected[field] = qa.proof(path)
        plan[field] = {**qa.proof(path), **({"jobs": rows} if field == "queue" else {})}
    plan["planned_gpus"] = [0]
    plan["future_full_run_comparison"]["expected_runs"] = plan["future_full_run_comparison"]["expected_runs"][:1]
    base.put(plan_path, plan)
    expected["plan"] = qa.proof(plan_path)
    expected["source_proofs"] = [qa.proof(source["path"]) for source in expected["source_proofs"]]
    base.put(prepared["path"], expected)
    return prepared


def test_one_actual_run_and_two_complete_endpoints_pass(single):
    single["complete"]()
    result = base.run(single)
    assert result["status"] == "passed_all_final_and_best_full_model_state_bytes"
    assert len(result["endpoint_proofs"]) == 2
    assert result["checkpoint_read_operations_count"] == 6
    assert {x["kind"] for x in result["endpoint_proofs"]} == {"final", "best"}
    assert {x["actual_exposure"]["step"] for x in result["endpoint_proofs"] if x["kind"] == "final"} == {7}


def test_one_run_without_actual_retraining_cannot_pass(single):
    result = base.run(single)
    assert result["status"].startswith("failed_")
    assert not result["checkpoint_read_operations"]


def test_missing_best_fails_before_new_checkpoint_reads(single):
    single["complete"]()
    best = single["expected"]["runs"][0]["endpoints"][1]
    Path(best["checkpoint"]).unlink()
    result = base.run(single)
    assert result["status"].startswith("failed_")
    assert not result["checkpoint_read_operations"]


def test_one_run_changed_best_full_state_fails(single):
    single["complete"]()
    base.edit_checkpoint(single, "best", lambda x: x["model_state"]["z.last"].view(torch.uint8).__setitem__(-1, 42))
    result = base.run(single)
    assert result["status"].startswith("failed_")
    assert "Full model_state bytes differ" in result["error"]["message"]


def test_existing_state_fingerprint_matches_archived_file_algorithm(tmp_path):
    from replay_study.weight_archive import model_fingerprint
    state = {"z.scalar": torch.tensor(1.25), "a.noncontiguous": torch.arange(12.).reshape(3, 4).t(),
             "e.empty": torch.empty(0)}
    path = tmp_path / "tiny_best.pt"
    torch.save({"model_state": state}, path)
    algorithm = qa.load_algorithm(qa.proof(ROOT / "scripts/audit_formal_runs.py"))
    assert model_fingerprint(state) == algorithm(path)
