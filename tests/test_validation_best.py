"""Best selection is observed validation, at its real earlier exposure."""
import importlib.util
import json
from pathlib import Path

import pytest

from replay_study.analysis import sha256

SPEC = importlib.util.spec_from_file_location("best_validation_reporter", Path(__file__).resolve().parents[1] / "scripts/analyze_validation_best.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def three(nll):
    return {name: {"nll": nll, "prediction_tokens": 10, "complete_corpus": True} for name in ("target", "c4", "wikitext")}


def test_earlier_best_keeps_actual_exposure_and_full_training_charge(tmp_path):
    baseline = tmp_path/"initial.json"
    baseline.write_text(json.dumps({"corpora":three(2.)}))
    path = tmp_path/"summary.json"
    path.write_text(json.dumps({"counters":{"step":10,"target_prediction_tokens":100,"replay_prediction_tokens":5},
        "best_target_validation":{"step":4,"target_prediction_tokens":40,"nll":1.},
        "validation_trajectory":[{"step":4,"target_prediction_tokens":40,"replay_prediction_tokens":2,"metrics":three(1.)},
                                  {"step":10,"target_prediction_tokens":100,"replay_prediction_tokens":5,"metrics":three(1.5)}]}))
    row = {"id":"formal_B4","phase":"formal","group":"B4","seed":11,"ratio":.05,"weight":.05,"summary_path":str(path)}
    bundle = {"registry":[row],"metadata":{"complete":True,"analysis_status":"active","split":"validation",
        "sources_sha256":{str(path):sha256(path),str(baseline):sha256(baseline)}}}
    result = REPORT.best_validation_tables(bundle,baseline)
    endpoint = result["best_validation_endpoints"][0]
    assert endpoint["target_prediction_tokens"]==40 and endpoint["replay_prediction_tokens"]==2
    assert endpoint["full_training_target_prediction_tokens"]==100 and endpoint["full_training_cost_remains_charged"]
    assert endpoint["checkpoint_step"]==4 and endpoint["g_new"]==1.
    assert endpoint["selection_uses_same_validation"] and endpoint["equal_full_budget_endpoint"] is False
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError,match="changed after"):
        REPORT.best_validation_tables(bundle,baseline)
