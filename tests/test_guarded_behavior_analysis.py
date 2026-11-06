"""Guard verifies private bytes without parsing them, then requires explicit unseal."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest

source=Path(__file__).resolve().parents[1]/'scripts/guarded_behavior_analysis.py'
spec=importlib.util.spec_from_file_location('behavior_guard',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write(path,value):path.write_text(json.dumps(value)+'\n');return path


def fixture(tmp_path):
    helperpath=Path(__file__).with_name('test_evaluation_cost_inventory.py');s=importlib.util.spec_from_file_location('guard_cost_fixture',helperpath);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
    initial=tmp_path/'initial.json';adapted=tmp_path/'adapted.json';alias=tmp_path/'alias.json'
    jobs=[{'id':'initial','kind':'initial','baseline_group':'g','output':str(initial)},
          {'id':'adapted','kind':'adapted','baseline_group':'g','output':str(adapted),'checkpoint_sha256':'a'*64}]
    records=[{'id':'canonical','output':str(adapted),'inference_job_id':'adapted','checkpoint_sha256':'a'*64},
             {'id':'alias','output':str(alias),'inference_job_id':'adapted','checkpoint_sha256':'b'*64}]
    p=write(tmp_path/'manifest.json',{'schema_version':2,'mode':'behavior','common_input_sha256':{},'jobs':jobs,'records':records,
        'baseline_groups':{'g':{'initial_job_id':'initial','initial_output':str(initial),'tasks':{'piqa':{},'hellaswag':{}}}}});digest=module.sha(p)
    h.attest(initial,digest,'initial',None,1.,10.)
    ap=h.attest(adapted,digest,'adapted','a'*64,2.,20.)
    h.attest(alias,digest,'adapted','b'*64,2.,0.,inference_alias=True,logical_record_id='alias',canonical_output=str(adapted.resolve()),canonical_inference_subprocess_wall_seconds=20.)
    authorization=write(tmp_path/'authorization.json',{'allow_score_analysis':True,'manifest_sha256':digest,'expected_inference_jobs':2,'expected_logical_endpoints':2})
    return p,digest,ap,authorization


def test_full_private_byte_attestations_can_be_checked_without_score_parsing(tmp_path):
    path,digest,_,_=fixture(tmp_path);result=module.check(path,digest,2)
    assert result['ready'] and result['all_output_integrity_verified'] and result['scores_or_predictions_parsed'] is False
    assert b'PRIVATE_SCORE' not in json.dumps(result).encode()
    with pytest.raises(ValueError,match='inference count'):module.check(path,digest,67)


def test_authorized_but_incomplete_phase_never_imports_or_runs_score_collector(tmp_path,monkeypatch):
    path,digest,ap,authorization=fixture(tmp_path);ap.unlink();called=[]
    monkeypatch.setitem(sys.modules,'analyze_behavioral_plan',types.SimpleNamespace(collect=lambda p:called.append(p)))
    with pytest.raises(ValueError,match='must complete before unsealing'):module.analyze(path,digest,authorization,tmp_path/'output')
    assert called==[] and not (tmp_path/'output').exists()


def test_complete_phase_without_matching_root_authorization_stays_sealed(tmp_path):
    path,digest,_,authorization=fixture(tmp_path);write(authorization,{'allow_score_analysis':False,'manifest_sha256':digest})
    with pytest.raises(ValueError,match='authorization'):module.analyze(path,digest,authorization,tmp_path/'output')


def test_changed_alias_prediction_bytes_fail_before_any_analysis(tmp_path):
    path,digest,_,_=fixture(tmp_path);(tmp_path/'alias_piqa.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError,match='output SHA'):module.check(path,digest,2)
