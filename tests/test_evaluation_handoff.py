import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/handoff_evaluation.py'
spec=importlib.util.spec_from_file_location('evaluation_handoff',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload));return path


def fixture(tmp_path):
    scorer=module.ROOT/'scripts/run_evaluate.py'
    initial={'id':'initial','kind':'initial','baseline_group':'same','output':str(tmp_path/'initial.json'),
        'input_sha256':{},'argv':[sys.executable,str(scorer),'--output',str(tmp_path/'initial.json')]}
    adapted={'id':'endpoint','kind':'adapted','baseline_group':'same','output':str(tmp_path/'adapted.json'),
        'input_sha256':{},'checkpoint_sha256':'full_model_file_sha','depends_on':['initial'],
        'argv':[sys.executable,str(scorer),'--output',str(tmp_path/'adapted.json')]}
    manifest={'schema_version':2,'mode':'nll','common_input_sha256':{str(module.WORKER):module.sha256(module.WORKER),
        str(scorer):module.sha256(scorer)},'baseline_groups':{'same':{'initial_job_id':'initial',
        'initial_output':initial['output'],'tasks':{}}},'jobs':[initial,adapted],'records':[]}
    path=write(tmp_path/'manifest.json',manifest);authority={'path':str(path),'sha256':module.sha256(path)}
    def outer(entry):
        job={'id':entry['id'],'argv':[sys.executable,str(module.WORKER),'--manifest',str(path),'--manifest-sha256',authority['sha256'],
            '--job-id',entry['id'],'--device','cuda:0'],'outputs':sorted([entry['output'],str(module.attestation_path(entry['output']))]),
            'expected_seconds':180,'env':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}}
        if entry.get('depends_on'):job['depends_on']=entry['depends_on']
        return job
    jobs=[outer(initial),outer(adapted)]
    queue=write(tmp_path/'endpoints.json',[jobs[1]]);prior=write(tmp_path/'prior.json',[jobs[0]])
    old_queue=write(tmp_path/'old_queue.json',jobs)
    old_state=write(tmp_path/'old_state.json',{'jobs':{'initial':{'status':'completed','returncode':0,'definition_sha256':module.digest_job(jobs[0])},
        'endpoint':{'status':'pending','definition_sha256':module.digest_job(jobs[1])}}})
    source={'queue_path':str(queue),'queue_sha256':module.sha256(queue),'manifests':[authority]}
    prior_source={'queue_path':str(prior),'queue_sha256':module.sha256(prior),'manifests':[authority]}
    plan={'schema_version':1,'old_master_queue':str(old_queue),'old_master_state':str(old_state),
        'evaluation_sources':[source],'prior_sources':[prior_source]}
    plan_path=write(tmp_path/'spec.json',plan)
    # The helper must never deserialize scores, even for a completed baseline.
    Path(initial['output']).write_bytes(b'SCORE FILE DELIBERATELY NOT VALID JSON')
    write(module.attestation_path(initial['output']),{'manifest_sha256':authority['sha256'],'job_id':'initial',
        'verified_frozen_inputs':True,'checkpoint_sha256':None,'actual_argv':initial['argv'],'inference_alias':False,
        'output_sha256':{initial['output']:module.sha256(initial['output'])}})
    args=argparse.Namespace(spec=plan_path,spec_sha256=module.sha256(plan_path),output_dir=tmp_path/'handoff')
    return args,plan,jobs,manifest


def test_old_master_alive_refuses_before_any_successor_queue_is_created(tmp_path):
    args,plan,_,_=fixture(tmp_path);state=module.read(plan['old_master_state']);state['scheduler_pid']=os.getpid()
    write(Path(plan['old_master_state']),state)
    with pytest.raises(RuntimeError,match='Old master is still alive'):module.prepare(args)
    assert not args.output_dir.exists()


def test_explicit_stat_bound_cache_preserves_full_input_validation_and_reports_hits(tmp_path):
    from stat_bound_digest_cache import StatBoundHashes
    args,plan,_,_=fixture(tmp_path)
    h=StatBoundHashes(directory=tmp_path/'cache')
    _,_,inputs=module.evaluation_plan(plan)
    for path,wanted in inputs.items():h.verify(path,wanted)
    plan['digest_cache']=h.snapshot(tmp_path/'snapshot.json')
    write(args.spec,plan);args.spec_sha256=module.sha256(args.spec)
    module.prepare(args)
    report=module.read(args.output_dir/'handoff.json')['input_verification']
    assert report['counters']['cache_hits']==len(inputs)
    assert report['counters']['full_read_bytes']==0
    assert report['cache_hits_are_current_byte_reads'] is False
    assert report['frozen_worker_always_fresh_full_input_hash'] is True


def test_optional_verifier_cannot_silently_ignore_expected_input_sha(tmp_path):
    from stat_bound_digest_cache import StatBoundHashes
    _,plan,_,_=fixture(tmp_path);h=StatBoundHashes(directory=tmp_path/'cache')
    path=tmp_path/'not_the_expected_input';path.write_bytes(b'actual')
    with pytest.raises(ValueError,match='SHA mismatch'):h.verify(path,'0'*64)
    verifier=module.InputVerification(plan)
    with pytest.raises(ValueError,match='input changed'):verifier.verify(path,'0'*64)
    with pytest.raises(ValueError,match='must return the checked canonical path'):
        module.evaluation_plan(plan,lambda path,wanted:None)


def test_surviving_child_process_group_is_not_treated_as_quiescent():
    state={'scheduler_pid':100,'jobs':{'old':{'pid':200,'status':'failed'}}}
    with pytest.raises(RuntimeError,match='children/process groups'):
        module.assert_quiescent(state,{300:{'ppid':1,'pgrp':200,'state':'S'}})


def test_training_argv_is_rejected_even_when_named_like_an_evaluation(tmp_path):
    args,plan,jobs,_=fixture(tmp_path);bad=copy.deepcopy(jobs[1]);bad['argv'][1]=str(module.ROOT/'scripts/run_train.py')
    source=plan['evaluation_sources'][0];write(Path(source['queue_path']),[bad]);source['queue_sha256']=module.sha256(source['queue_path'])
    with pytest.raises(ValueError,match='non-frozen worker argv'):module.evaluation_plan(plan)


def test_complete_baseline_dependency_is_merged_and_inherited_without_rerunning(tmp_path):
    args,plan,_,_=fixture(tmp_path);module.prepare(args)
    queue=module.read(args.output_dir/'queue.json');state=module.read(args.output_dir/'state.json')
    assert [j['id'] for j in queue]==['initial','endpoint']
    assert state['jobs']['initial']['status']=='completed'
    assert state['jobs']['endpoint']['status']=='pending'
    handoff=module.read(args.output_dir/'handoff.json')
    assert handoff['inherited_completed_jobs']==1
    assert handoff['scores_or_predictions_deserialized'] is False
    assert (args.output_dir/'old_state_snapshot.json').read_bytes()==Path(plan['old_master_state']).read_bytes()


def test_completed_output_missing_is_reset_pending_instead_of_falsely_inherited(tmp_path):
    args,_,_,manifest=fixture(tmp_path);Path(manifest['jobs'][0]['output']).unlink();module.prepare(args)
    state=module.read(args.output_dir/'state.json');assert state['jobs']['initial']['status']=='pending'
    evidence=module.read(args.output_dir/'handoff.json')['completion_transfer']['initial']
    assert not evidence['inherited'] and 'Missing completed output' in evidence['reason']


def test_same_id_changed_definition_is_never_adopted_from_old_state(tmp_path):
    args,plan,jobs,_=fixture(tmp_path);old=copy.deepcopy(jobs);old[0]['expected_seconds']=181
    write(Path(plan['old_master_queue']),old)
    with pytest.raises(ValueError,match='same-ID definition differs'):module.prepare(args)


def test_missing_explicit_prior_baseline_is_rejected(tmp_path):
    _,plan,_,_=fixture(tmp_path);plan['prior_sources']=[]
    with pytest.raises(ValueError,match='prior evaluation source required'):module.evaluation_plan(plan)


def test_tail_includes_grace_and_polling_with_at_least_thirty_minutes_remaining():
    settings=module.deadline_settings({'tail_seconds':1800})
    assert settings['hard_deadline_utc']=='2026-11-07T03:26:32Z'
    assert settings['stop_margin_seconds']-settings['terminate_grace_seconds']-2*settings['poll_seconds']>=1800
    with pytest.raises(ValueError,match='At least30 minutes'):module.deadline_settings({'tail_seconds':1799})


def test_prepared_handoff_still_rechecks_old_master_before_start(tmp_path):
    args,plan,_,_=fixture(tmp_path);module.prepare(args)
    record=module.read(args.output_dir/'handoff.json');state=module.read(plan['old_master_state']);state['scheduler_pid']=os.getpid()
    write(Path(plan['old_master_state']),state)
    start=argparse.Namespace(handoff=args.output_dir/'handoff.json',handoff_sha256=module.sha256(args.output_dir/'handoff.json'))
    with pytest.raises(RuntimeError,match='Old master is still alive'):module.start(start)
    assert not (args.output_dir/'launch.json').exists()


def test_unrelated_completed_training_is_never_transferred(tmp_path):
    args,plan,jobs,_=fixture(tmp_path)
    train={'id':'old_training','argv':[sys.executable,'run_train.py']}
    write(Path(plan['old_master_queue']),jobs+[train]);state=module.read(plan['old_master_state'])
    state['jobs']['old_training']={'status':'completed','returncode':0,'definition_sha256':module.digest_job(train)}
    write(Path(plan['old_master_state']),state);module.prepare(args)
    assert 'old_training' not in module.read(args.output_dir/'state.json')['jobs']
    assert all(job['id']!='old_training' for job in module.read(args.output_dir/'queue.json'))


def test_empty_output_hash_attestation_cannot_make_an_incomplete_baseline_complete(tmp_path):
    args,_,_,manifest=fixture(tmp_path);path=module.attestation_path(manifest['jobs'][0]['output'])
    proof=module.read(path);proof['output_sha256']={};write(path,proof);module.prepare(args)
    assert module.read(args.output_dir/'state.json')['jobs']['initial']['status']=='pending'


def test_start_passes_the_old_lock_to_only_a_mocked_new_scheduler(tmp_path,monkeypatch):
    args,_,_,_=fixture(tmp_path);module.prepare(args);observed={}
    real_datetime=module.dt.datetime
    class FixedDateTime(real_datetime):
        @classmethod
        def now(cls,tz=None):return real_datetime(2026,11,6,tzinfo=module.dt.timezone.utc)
    class FakeProcess:
        def __init__(self,command,**kwargs):observed.update(command=command,kwargs=kwargs);self.pid=99999999
    monkeypatch.setattr(module.dt,'datetime',FixedDateTime)
    monkeypatch.setattr(module.subprocess,'Popen',FakeProcess)
    start=argparse.Namespace(handoff=args.output_dir/'handoff.json',handoff_sha256=module.sha256(args.output_dir/'handoff.json'))
    module.start(start)
    assert observed['kwargs']['start_new_session'] is True
    assert len(observed['kwargs']['pass_fds'])==1
    assert observed['command'][observed['command'].index('--gpus')+1]=='0,1,2,3'
    assert observed['command'][observed['command'].index('--deadline')+1]=='2026-11-07T03:26:32Z'
    assert module.read(args.output_dir/'launch.json')['old_master_lock_inherited_by_new_scheduler'] is True


def test_nonfinite_tail_cannot_bypass_the_hard_stop():
    with pytest.raises(ValueError,match='finite final'):module.deadline_settings({'tail_seconds':float('nan')})
