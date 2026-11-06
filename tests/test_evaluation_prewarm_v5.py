import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import prepare_supplemental_evaluations_v5 as module
import stat_bound_digest_cache as cache


def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload));return path


def fixture(tmp_path):
    output=tmp_path/'run';out=output/'checkpoints';out.mkdir(parents=True)
    for name in ['latest.pt','checkpoint_best.pt']:(out/name).write_bytes(b'opaque bytes, not deserialized')
    cfg={'run_label':'run','output_dir':str(output),'target_prediction_tokens':100,'replay_ratio':.05}
    cp=write(tmp_path/'config.json',cfg);summary=write(output/'summary.json',{'status':'completed','config':cfg,
        'counters':{'target_prediction_tokens':100,'replay_prediction_tokens':5},'checkpoint_path':str(out/'latest.pt'),
        'best_target_validation':{'checkpoint_path':str(out/'checkpoint_best.pt')}})
    job={'id':'run','argv':[sys.executable,'run_train.py','--config',str(cp)],'outputs':[str(summary)]}
    state=write(tmp_path/'state.json',{'jobs':{'run':{'status':'completed','returncode':0,
        'definition_sha256':module.digest_job(job),'argv':job['argv'],'missing_outputs':[]}}})
    queue=write(tmp_path/'queue.json',[job]);registry=write(tmp_path/'registry.json',[{'id':'run'}])
    row={'id':'run','config':cache.small_proof(cp),'summary_path':str(summary),'output_dir':str(output),
         'state_path':str(state),'queue_path':str(queue),'job':job,'definition_sha256':module.digest_job(job)}
    cat={'prewarm_runs':[row],'prewarm_sources':[{'registry':cache.small_proof(registry),'state_path':str(state),'queue_path':str(queue)}]}
    h=cache.StatBoundHashes(directory=tmp_path/'cache');gate=module.CompletedRuns(cat,h);h.checkpoint_guard=gate.guard
    return cat,h,gate,out,state,summary


def test_two_endpoints_prewarm_with_completed_job_authority(tmp_path):
    _,h,gate,out,_,_=fixture(tmp_path);cursor,result=module.prewarm(gate,h,0,5)
    assert result['visited']==2 and result['pending']==0
    for name in ['latest.pt','checkpoint_best.pt']:
        receipt=h._receipt(str(out/name))
        assert receipt['source_authority']['returncode']==0
        assert receipt['source_authority']['run_id']=='run'
    old=h.counters['full_read_bytes'];module.prewarm(gate,h,cursor,5)
    assert h.counters['full_read_bytes']==old


def test_pending_scheduler_does_not_hash_present_checkpoint(tmp_path):
    _,h,gate,out,state,_=fixture(tmp_path);value=module.read(state);value['jobs']['run']['status']='running';write(state,value);gate.refresh()
    _,result=module.prewarm(gate,h,0,5)
    assert result['pending']==2 and not any('/checkpoints/' in path for path in h.entries)


@pytest.mark.parametrize('change',['returncode','definition','argv','missing_output','stop'])
def test_failed_or_mismatched_completion_is_rejected(tmp_path,change):
    _,h,gate,out,state,_=fixture(tmp_path);value=module.read(state);record=value['jobs']['run']
    if change=='returncode':record['returncode']=1
    if change=='definition':record['definition_sha256']='0'*64
    if change=='argv':record['argv']=['wrong']
    if change=='missing_output':record['missing_outputs']=['lost']
    if change=='stop':record['stop_requested_at']='deadline'
    write(state,value);gate.refresh()
    with pytest.raises(ValueError):h.digest(out/'latest.pt')


def test_wrong_prediction_budget_or_config_rejected(tmp_path):
    _,h,gate,out,_,summary=fixture(tmp_path);value=module.read(summary);value['counters']['replay_prediction_tokens']=4;write(summary,value)
    with pytest.raises(ValueError,match='exposure mismatch'):h.digest(out/'latest.pt')


def test_current_job_definition_change_rejected_before_prewarm(tmp_path):
    cat,h,gate,_,_,_=fixture(tmp_path);path=Path(cat['prewarm_runs'][0]['queue_path']);value=module.read(path);value[0]['argv']+=['changed'];write(path,value)
    with pytest.raises(ValueError,match='definition changed'):gate.refresh()


def test_missing_completed_checkpoint_not_treated_as_pending_success(tmp_path):
    _,h,gate,out,_,_=fixture(tmp_path);(out/'latest.pt').unlink()
    with pytest.raises(FileNotFoundError):module.prewarm(gate,h,0,5)


def test_arbitrary_checkpoint_outside_frozen_registry_rejected(tmp_path):
    _,h,_,out,_,_=fixture(tmp_path);other=out/'unknown.pt';other.write_bytes(b'unknown')
    with pytest.raises(ValueError,match='explicit frozen run authority'):h.digest(other)


def test_cached_checkpoint_still_requires_same_completed_summary_authority(tmp_path):
    _,h,_,out,_,summary=fixture(tmp_path);h.digest(out/'latest.pt');value=module.read(summary);value['extra']='changed';write(summary,value)
    with pytest.raises(ValueError,match='source authority changed'):h.digest(out/'latest.pt')
