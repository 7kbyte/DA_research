"""Synthetic metadata only; never invoke a real behavior check or collector."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_primary_behavior_analysis_v1.py"
spec = importlib.util.spec_from_file_location("primary_behavior_wrapper", SCRIPT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def tiny_scope():
    jobs = [{"id": "initial", "kind": "initial", "baseline_group": "g", "output": "/fake/initial.json"},
            {"id": "adapted", "kind": "adapted", "baseline_group": "g", "output": "/fake/canonical.json"}]
    base = {"baseline_group": "g", "initial_job_id": "initial", "inference_job_id": "adapted"}
    records = [{**base, "id": "canonical", "output": "/fake/canonical.json"},
               {**base, "id": "alias", "output": "/fake/alias.json", "inference_alias_of": "canonical", "source_model_equivalence": {"proof": "synthetic"}}]
    manifest = {"schema_version": 2, "mode": "behavior", "jobs": jobs, "records": records,
                "baseline_groups": {"g": {"initial_job_id": "initial", "initial_output": "/fake/initial.json", "tasks": {"tiny": {}}}}}
    queue = []
    for job in jobs:
        outputs = m.output_set(job["output"], {"tiny": {}}) | {str(m.attestation_path(job["output"]))}
        if job["kind"] != "initial": outputs |= m.output_set("/fake/alias.json", {"tiny": {}}) | {"/fake/alias_execution.json"}
        queue.append({"id": job["id"], "argv": ["synthetic", job["id"]], "outputs": sorted(outputs)})
    counts = {"physical_jobs": 2, "logical_endpoints": 2, "proved_aliases": 1, "initial_baselines": 1}
    return manifest, queue, counts


def test_full_tiny_baseline_alias_output_closure():
    manifest, queue, counts = tiny_scope()
    assert len(m.validate_scope(manifest, queue, counts)) == 1


@pytest.mark.parametrize("bad", ["scope", "initial", "alias", "outputs", "duplicate"])
def test_wrong_scope_or_incomplete_alias_closure_rejected(bad):
    manifest, queue, counts = tiny_scope()
    if bad == "scope": manifest["mode"] = "nll"
    if bad == "initial": manifest["baseline_groups"]["g"]["initial_job_id"] = "outside"
    if bad == "alias": manifest["records"][1]["source_model_equivalence"] = None
    if bad == "outputs": queue[1]["outputs"].pop()
    if bad == "duplicate": manifest["records"][1]["id"] = "canonical"
    with pytest.raises(ValueError): m.validate_scope(manifest, queue, counts)


def completed_records():
    job = {"id": "synthetic", "argv": ["synthetic", "--frozen"], "outputs": ["/fake/output"], "expected_seconds": 1}
    record = {"status": "completed", "returncode": 0, "missing_outputs": [], "argv": job["argv"], "definition_sha256": m.definition_sha(job)}
    return job, record


@pytest.mark.parametrize("bad", ["failed", "pending", "missing", "stopped", "argv", "definition", "no_initial", "bool_rc"])
def test_state_fail_closed(bad):
    job, record = completed_records(); queue = [copy.deepcopy(job)]; state = {"jobs": {job["id"]: record}}
    if bad == "failed": record["returncode"] = 1
    if bad == "pending": record["status"] = "running"
    if bad == "missing": record["missing_outputs"] = ["missing"]
    if bad == "stopped": record["stop_requested_at"] = "synthetic"
    if bad == "argv": record["argv"] = ["changed"]
    if bad == "definition": queue[0]["expected_seconds"] = 2
    if bad == "no_initial": state["jobs"] = {}
    if bad == "bool_rc": record["returncode"] = False
    with pytest.raises(ValueError): m.state_records([job], queue, state)


def test_complete_state_and_unrelated_jobs():
    job, record = completed_records()
    assert m.state_records([job], [job, {"id": "unrelated"}], {"jobs": {job["id"]: record, "unrelated": {"status": "running"}}}) == {job["id"]: record}


def test_changed_source_rejected_before_use(tmp_path):
    path = tmp_path / "source.py"; path.write_text("old")
    proof, _ = m.small(path); path.write_text("changed")
    with pytest.raises(ValueError): m.small(path, proof["sha256"])


def test_false_authority_fails_without_calling_original_guard(monkeypatch):
    monkeypatch.setattr(m, "load_plan", lambda _: {"manifest": {}, "frozen_queue": {}, "source_sha256": {}, "analysis_output": "synthetic"})
    monkeypatch.setattr(m, "read_proof", lambda _: {"allow_score_analysis": False})
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: pytest.fail("Original guard must not be called"))
    with pytest.raises(ValueError, match="ROOT true authority"): m.authorize({}, {}, {})


def test_journal_is_exclusive_and_not_overwritten(tmp_path):
    path = tmp_path / "synthetic_only_journal.json"
    m.create_journal(path, {"synthetic_fixture": True}); before = path.read_bytes()
    with pytest.raises(FileExistsError): m.create_journal(path, {"synthetic_fixture": "retry"})
    assert path.read_bytes() == before and json.loads(before)["status"] == "semantic_access_enabled"


def fake_operation(monkeypatch, tmp_path, mode="success"):
    attempt = tmp_path / "attempt"; attempt.mkdir()
    journal = tmp_path / "synthetic_only_journal.json"; analysis = tmp_path / "synthetic_analysis"
    plan = {"manifest": {}, "frozen_queue": {}, "source_sha256": {}, "analysis_output": str(analysis)}
    state = {"actual_queue": {"synthetic": True}}
    monkeypatch.setattr(m, "JOURNAL", journal)
    monkeypatch.setattr(m, "authorize", lambda *a: (plan, state))
    monkeypatch.setattr(m, "refresh_state", lambda *a: {"synthetic": True})
    monkeypatch.setattr(m, "load_plan", lambda *a: plan)
    monkeypatch.setattr(m, "read_proof", lambda *a: {})
    sequence = []
    def runner(argv, **kwargs):
        if "--analyze" not in argv:
            sequence.append("check"); assert not journal.exists()
            value = {"ready": mode != "pending", "all_output_integrity_verified": True, "manifest_sha256": m.MANIFEST_SHA,
                     "expected_inference_jobs": 67, "logical_endpoints": 72, "scores_or_predictions_parsed": False, "model_inference_started": False}
            kwargs["stdout"].write(m.encoded(value))
            return SimpleNamespace(returncode=0)
        sequence.append("analyze"); assert journal.exists()
        receipt = json.loads(journal.read_text())
        assert receipt["status"] == "semantic_access_enabled" and receipt["before_original_analyze_invocation"]
        if mode == "analyze_failed": return SimpleNamespace(returncode=1)
        analysis.mkdir(); (analysis / "analysis_manifest.json").write_bytes(m.encoded({"complete": True, "manifest_sha256": m.MANIFEST_SHA, "logical_endpoints": 72}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess, "run", runner)
    return attempt, journal, sequence


def test_check_then_exclusive_journal_then_original_analysis(monkeypatch, tmp_path):
    attempt, journal, sequence = fake_operation(monkeypatch, tmp_path)
    m.execute({}, {}, {"path": "synthetic_only"}, attempt)
    assert sequence == ["check", "analyze"] and journal.exists() and (attempt / "COMPLETE.json").exists()


def test_incomplete_check_never_enables_semantic_access(monkeypatch, tmp_path):
    attempt, journal, sequence = fake_operation(monkeypatch, tmp_path, "pending")
    with pytest.raises(ValueError, match="Original full check"): m.execute({}, {}, {}, attempt)
    assert sequence == ["check"] and not journal.exists()


def test_source_changed_after_check_no_journal_or_analyze(monkeypatch, tmp_path):
    attempt, journal, sequence = fake_operation(monkeypatch, tmp_path)
    def changed(*_): raise ValueError("source changed")
    monkeypatch.setattr(m, "load_plan", changed)
    with pytest.raises(ValueError): m.execute({}, {}, {}, attempt)
    assert sequence == ["check"] and not journal.exists()


def test_analysis_failure_preserves_first_read_attempt(monkeypatch, tmp_path):
    attempt, journal, sequence = fake_operation(monkeypatch, tmp_path, "analyze_failed")
    with pytest.raises(ValueError, match="journal preserved"): m.execute({}, {}, {"path": "synthetic_only"}, attempt)
    assert sequence == ["check", "analyze"] and journal.exists() and not (attempt / "COMPLETE.json").exists()


def test_duplicate_journal_rejected_before_check(monkeypatch, tmp_path):
    attempt, journal, sequence = fake_operation(monkeypatch, tmp_path)
    journal.write_text("synthetic previous attempt")
    with pytest.raises(ValueError, match="never reset"): m.execute({}, {}, {}, attempt)
    assert not sequence


def synthetic_attestations(tmp_path):
    group = {"tasks": {"tiny": {}}, "inference_policy": {"torch_version": "synthetic", "cuda_version": "none", "gpu_name": "none"}}
    initial = {"id": "initial", "kind": "initial", "baseline_group": "g", "output": str(tmp_path / "initial.json"), "argv": ["never_execute"]}
    adapted = {"id": "adapted", "kind": "adapted", "baseline_group": "g", "output": str(tmp_path / "adapted.json"), "argv": ["never_execute", "adapted"],
               "checkpoint_sha256": "a" * 64, "weights_identity": "model_state:" + "f" * 64, "source_model_equivalence": {"synthetic": True}}
    alias = {"id": "alias", "inference_job_id": "adapted", "output": str(tmp_path / "alias.json"), "baseline_group": "g", "checkpoint_sha256": "b" * 64,
             "weights_identity": adapted["weights_identity"], "source_model_equivalence": adapted["source_model_equivalence"]}
    manifest = {"jobs": [initial, adapted], "records": [alias], "baseline_groups": {"g": group}}
    for job, logical in ((initial, None), (adapted, None), (adapted, alias)):
        row = logical or job; output = row["output"]
        hashes = {}
        for path in m.output_set(output, group["tasks"]):
            # Deliberately invalid score JSON: metadata-only function must not parse.
            Path(path).write_text("OPAQUE SYNTHETIC SCORE BYTES; NEVER PARSE")
            hashes[path] = "c" * 64
        att = {"manifest_sha256": "synthetic", "job_id": job["id"], "baseline_group": "g", "verified_frozen_inputs": True,
               "checkpoint_sha256": row.get("checkpoint_sha256"), "weights_identity": row.get("weights_identity"), "actual_argv": job["argv"],
               "source_model_equivalence": row.get("source_model_equivalence"), "inference_alias": logical is not None,
               "preflight_environment": group["inference_policy"], "output_sha256": hashes}
        if logical: att.update(logical_record_id=logical["id"], canonical_output=job["output"], inference_subprocess_wall_seconds=0)
        m.attestation_path(output).write_bytes(m.encoded(att))
    return manifest, adapted, alias


def test_attestation_metadata_never_parses_opaque_score_bytes(tmp_path):
    manifest, _, _ = synthetic_attestations(tmp_path)
    assert len(m.metadata_attestations(manifest, "synthetic")) == 3


@pytest.mark.parametrize("bad", ["checkpoint", "weights", "argv", "outputs", "alias", "environment"])
def test_attestation_bindings_fail_closed(tmp_path, bad):
    manifest, adapted, alias = synthetic_attestations(tmp_path)
    path = m.attestation_path(alias["output"]); att = json.loads(path.read_text())
    if bad == "checkpoint": att["checkpoint_sha256"] = "d" * 64
    if bad == "weights": att["weights_identity"] = "changed"
    if bad == "argv": att["actual_argv"] = ["changed"]
    if bad == "outputs": att["output_sha256"].pop(next(iter(att["output_sha256"])))
    if bad == "alias": att["inference_subprocess_wall_seconds"] = 1
    if bad == "environment": att["preflight_environment"] = {"torch_version": "changed"}
    path.write_bytes(m.encoded(att))
    with pytest.raises(ValueError): m.metadata_attestations(manifest, "synthetic")
