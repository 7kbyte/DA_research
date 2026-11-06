"""No score-byte access before every primary job finishes; timestamps never reset."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
import monitor_primary_nll_readiness as monitor
import analyze_primary_nll as runner


def write(path,value):
    path.write_text(json.dumps(value)+"\n");return path


def fixture(tmp_path):
    source=write(tmp_path/"source.json",{})
    group=dict(id="base",initial_job_id="initial",evaluation_input_sha256={},base_model_assets_sha256={})
    manifest=dict(schema_version=2,mode="nll",phases=[dict(name="primary",role="primary",model_tag="pythia-410m",
        expected_phase="formal",data_family="main_v3",rows=1,protocol=str(source),protocol_sha256=monitor.sha(source),
        registry=str(source),registry_sha256=monitor.sha(source))],baseline_groups={"base":group},jobs=[],records=[])
    for kind in ("initial","final","best"):
        jid=kind;path=tmp_path/(kind+".json")
        path.write_bytes(b"PRIVATE_TEST_SCORE_not_JSON")
        job=dict(id=jid,kind="initial" if kind=="initial" else "adapted",baseline_group="base",output=str(path),argv=[kind])
        if kind!="initial":
            job.update(checkpoint_sha256=kind,weights_identity=kind)
            manifest["records"].append(dict(id=kind,run_id="run1",kind=kind,phase="formal",phase_name="primary",role="primary",
                model_tag="pythia-410m",data_family="main_v3",inference_job_id=jid,output=str(path)))
        manifest["jobs"].append(job)
    mp=write(tmp_path/"manifest.json",manifest);digest=monitor.sha(mp)
    for job in manifest["jobs"]:
        path=Path(job["output"])
        write(path.with_name(path.stem+"_execution.json"),dict(manifest_sha256=digest,job_id=job["id"],verified_frozen_inputs=True,
            baseline_group="base",checkpoint_sha256=job.get("checkpoint_sha256"),weights_identity=job.get("weights_identity"),
            inference_alias=False,actual_argv=job["argv"],output_sha256={str(path):monitor.sha(path)}))
    state=write(tmp_path/"state.json",dict(jobs={j["id"]:dict(status="completed",returncode=0) for j in manifest["jobs"]}))
    return mp,digest,state,manifest


def test_partial_completion_never_probes_reads_or_hashes_score_outputs(tmp_path,monkeypatch):
    mp,digest,state,manifest=fixture(tmp_path)
    s=json.loads(state.read_text());s["jobs"]["best"]={"status":"running"};write(state,s)
    seen=[];original=monitor.sha
    def observed(path):seen.append(str(path));return original(path)
    monkeypatch.setattr(monitor,"sha",observed)
    result=monitor.inspect(mp,digest,state,expected_runs=1,expected_jobs=3)
    assert not result["all_physical_jobs_complete"] and not result["result_files_hashed"]
    assert not {j["output"] for j in manifest["jobs"]}&set(seen)


def test_complete_gate_hashes_but_never_parses_actual_score_bytes(tmp_path,monkeypatch):
    mp,digest,state,manifest=fixture(tmp_path)
    monkeypatch.setattr(monitor,"prepare_plan",lambda *a,**kw:None)
    result=monitor.inspect(mp,digest,state,expected_runs=1,expected_jobs=3)
    assert result["status"]=="passed_complete_primary_integrity" and result["attestation_records"]==3
    assert not result["score_json_semantically_read"]
    Path(manifest["jobs"][1]["output"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError,match="SHA differs"):
        monitor.inspect(mp,digest,state,expected_runs=1,expected_jobs=3)


def test_behavior_or_supplement_cannot_relabel_itself_as_primary(tmp_path):
    _,_,_,m=fixture(tmp_path)
    m["mode"]="behavior"
    with pytest.raises(ValueError):monitor.validate_primary_scope(m,expected_runs=1,expected_jobs=3)
    m["mode"]="nll";m["records"][0]["role"]="supplement_A"
    with pytest.raises(ValueError):monitor.validate_primary_scope(m,expected_runs=1,expected_jobs=3)


def test_first_read_journal_is_immutable_across_final_and_best(tmp_path):
    readiness=write(tmp_path/"ready.json",{"status":"passed"});journal=tmp_path/"first.json"
    first=runner.first_read_receipt("baseline.json",journal,manifest_sha="approved",readiness_receipt=readiness)
    before=journal.read_bytes()
    again=runner.first_read_receipt("best.json",journal,manifest_sha="approved",readiness_receipt=readiness)
    assert first==again and journal.read_bytes()==before
    assert first["first_score_path"]=="baseline.json" and first["before_first_score_json_parse"]
    with pytest.raises(ValueError,match="another scope"):
        runner.first_read_receipt("other.json",journal,manifest_sha="other",readiness_receipt=readiness)
