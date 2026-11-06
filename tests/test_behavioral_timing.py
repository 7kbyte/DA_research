"""Timing tools must never deserialize scored outputs or start inference."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluation_plan import encoded, sha
from inspect_behavior_timing import inspect


def write(path, value):
    path.write_bytes(encoded(value)); return path


def fixture(tmp_path):
    output = tmp_path / "initial.json"
    files = [output, tmp_path / "initial_piqa.jsonl", tmp_path / "initial_hellaswag.jsonl"]
    for path in files:
        path.write_bytes(b"PRIVATE_SCORE_BYTES_intentionally_not_JSON")
    manifest = write(tmp_path / "manifest.json", {"jobs": [{"id": "initial", "kind": "initial", "output": str(output), "baseline_group": "g"}],
        "baseline_groups": {"g": {"tasks": {"piqa": {}, "hellaswag": {}}}}})
    authority = write(tmp_path / "authority.json", {"job_id": "initial", "manifest_sha256": sha(manifest)})
    state = write(tmp_path / "state.json", {"jobs": {"initial": {"status": "completed", "returncode": 0, "elapsed_seconds": 12.,
        "stdout": "PRIVATE_SCORE_BYTES_must_not_escape"}}})
    attest = write(tmp_path / "initial_execution.json", {"job_id": "initial", "manifest_sha256": sha(manifest),
        "verified_frozen_inputs": True, "checkpoint_sha256": None, "verification_wall_seconds": 1.,
        "inference_subprocess_wall_seconds": 10., "output_sha256": {str(p): sha(p) for p in files}})
    return state, manifest, authority, files


def test_only_attestation_hashes_and_allowlisted_scheduler_fields_are_read(tmp_path):
    state, manifest, authority, files = fixture(tmp_path)
    value = inspect(state, manifest, authority)
    assert value["complete_timing_available"] is True
    assert value["scores_or_predictions_parsed"] is False
    assert value["execution"]["inference_subprocess_wall_seconds"] == 10.
    assert b"PRIVATE_SCORE" not in encoded(value)


def test_changed_prediction_bytes_fail_integrity_without_parsing(tmp_path):
    state, manifest, authority, files = fixture(tmp_path)
    files[-1].write_bytes(b"changed")
    with pytest.raises(ValueError, match="attested hashes"):
        inspect(state, manifest, authority)


def test_pending_scheduler_record_does_not_claim_completed_timing(tmp_path):
    state, manifest, authority, files = fixture(tmp_path)
    write(state, {"jobs": {"initial": {"status": "running"}}})
    assert inspect(state, manifest, authority)["complete_timing_available"] is False
