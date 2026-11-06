"""Replay integrity stays sealed until all fifteen finish; no score JSON loads."""
import hashlib
import importlib.util
import json
from pathlib import Path

source=Path(__file__).resolve().parents[1]/'scripts/inspect_auxiliary_evaluation.py'
spec=importlib.util.spec_from_file_location('aux_monitor',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write(path,value):path.write_text(json.dumps(value)+'\n');return path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_replay_output_hashes_are_deferred_until_all_fifteen_completed(tmp_path,monkeypatch):
    jobs=[];states={};score_paths=[]
    for i in range(15):
        output=tmp_path/f'case{i}.json';output.write_bytes(b'PRIVATE_SCORE_not_JSON');score_paths.append(output)
        jobs.append({'id':str(i),'kind':'initial','output':str(output)})
        states[str(i)]={'status':'completed','returncode':0}
    p=write(tmp_path/'manifest.json',{'jobs':jobs});digest=sha(p)
    for job in jobs:
        output=Path(job['output']);write(output.with_name(output.stem+'_execution.json'),{'manifest_sha256':digest,'job_id':job['id'],
            'verified_frozen_inputs':True,'checkpoint_sha256':None,'output_sha256':{str(output):sha(output)},'verification_wall_seconds':1.,'inference_subprocess_wall_seconds':2.})
    authority=write(tmp_path/'authority.json',{'jobs':[{'phase':'replay_fit_diagnostic_v2','manifest':str(p),'manifest_sha256':digest,'jobs':15}]})
    states['14']={'status':'running'};state=write(tmp_path/'state.json',{'jobs':states});reads=[]
    original=module.sha
    def observed(path):reads.append(Path(path));return original(path)
    monkeypatch.setattr(module,'sha',observed)
    pending=module.inspect(authority,state)
    assert pending['replay_fit']['integrity_check_deferred_until_all_completed'] and not set(score_paths)&set(reads)
    states['14']={'status':'completed','returncode':0};write(state,{'jobs':states});reads.clear()
    completed=module.inspect(authority,state)
    assert completed['replay_fit']['all_15_completed'] and len(set(score_paths)&set(reads))==15
    assert all(r['output_hashes_valid'] for r in completed['replay_fit']['jobs'])
    assert b'PRIVATE_SCORE' not in json.dumps(completed).encode()
