import argparse,copy,importlib.util,json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
master=load('manage_supplemental_queue');selection=load('select_extended_phase')

def job(name):return {'id':name,'argv':['python','scripts/run_train.py','--config',name+'.json']}

def test_priority_append_preserves_existing_jobs_and_is_idempotent():
    old=[job('A'),job('B')];before=copy.deepcopy(old)
    merged=master.merge_jobs(old,[job('LR')],True)
    assert [j['id'] for j in merged]==['LR','A','B'] and old==before
    assert master.merge_jobs(merged,[job('LR')],True)==merged
    changed=job('A');changed['argv']+=['different']
    with pytest.raises(ValueError,match='Immutable'):master.merge_jobs(old,[changed])
    with pytest.raises(ValueError,match='Duplicate'):master.merge_jobs([], [job('A'),job('A')])

def test_gpu_remapping_and_nested_schedulers_are_rejected():
    override=job('bad');override['env']={'CUDA_VISIBLE_DEVICES':'3'}
    with pytest.raises(ValueError,match='inherit'):master.merge_jobs([], [override])
    nested={'id':'nested','argv':['python','scripts/calibrate_extended_experiments.py','--execute']}
    with pytest.raises(ValueError,match='Nested'):master.merge_jobs([], [nested])
    direct={'id':'bench','argv':['python','scripts/benchmark_extended_models.py','--one','config.json']}
    assert master.merge_jobs([], [direct])==[direct]

def test_phase_selector_requires_all_training_and_baseline_jobs_completed():
    queue=[job('initial'),job('LR1'),job('LR2'),job('LR3')]
    state={'jobs':{j['id']:{'status':'completed'} for j in queue}}
    selection.require_complete_jobs(queue,state)
    state['jobs']['initial']['status']='running'
    with pytest.raises(ValueError,match='every job'):selection.require_complete_jobs(queue,state)

def test_supplemental_launch_requires_complete_main_audit(tmp_path,monkeypatch):
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'status':'in_progress','completed_runs':35}))
    monkeypatch.setattr(master.subprocess,'Popen',lambda *a,**kw:pytest.fail('Gate launched training'))
    with pytest.raises(ValueError,match='Main36'):master.start(argparse.Namespace(main_audit=audit))

def test_detached_start_records_activation_and_rejects_held_state_lock(tmp_path,monkeypatch):
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'status':'passed','completed_runs':36}))
    queue=tmp_path/'queue.json';queue.write_text(json.dumps([job('A')]))
    index=tmp_path/'index.json';index.write_text(json.dumps({'master_queue':str(queue),'phases':{'A':{}}}))
    args=argparse.Namespace(main_audit=audit,master_queue=queue,state=tmp_path/'logs/state.json',
        phase_index=index,gpus='0,1,2,3',deadline='2026-11-07T00:00:00Z')
    calls=[]
    def fake_popen(argv,**kwargs):
        calls.append((argv,kwargs));return argparse.Namespace(pid=123456)
    monkeypatch.setattr(master.subprocess,'Popen',fake_popen)
    master.start(args)
    assert calls[0][1]['start_new_session'] and '--exit-when-idle' not in calls[0][0]
    activated=json.loads(index.read_text())
    assert activated['status']=='running' and activated['scheduler_pid']==123456
    assert activated['phases']['A']['active'] and activated['gpus']==['0','1','2','3']
    with args.state.with_suffix('.lock').open('a') as lock:
        master.fcntl.flock(lock,master.fcntl.LOCK_EX|master.fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='state lock'):master.start(args)
    assert len(calls)==1

def test_prepare_separates_phase_registry_and_keeps_A_batch16(tmp_path,monkeypatch):
    monkeypatch.setattr(master,'ROOT',tmp_path)
    cfg=tmp_path/'original.json';original={'microbatch_size':16,'gradient_accumulation_steps':4,
        'deadline_utc':'2026-11-07T03:26:32Z','target_prediction_tokens':10_000_000}
    cfg.write_text(json.dumps(original));registry=tmp_path/'source_registry.json';queue=tmp_path/'source_queue.json'
    rows=[{'id':'B4weight1','execution_kind':'train','config_path':str(cfg),'config_sha256':master.sha(cfg)},
          {'id':'weight05_alias','execution_kind':'alias','alias_of':'formal_B4'}]
    registry.write_text(json.dumps(rows));queue.write_text(json.dumps([{'id':'B4weight1',
        'argv':['python','scripts/make_experiments.py','run','--config',str(cfg)]}]))
    master.prepare(argparse.Namespace(namespace='test_master',scratch=tmp_path/'scratch',
        deadline='2026-11-07T00:00:00Z',A_registry=registry,A_queue=queue))
    generated=json.loads((tmp_path/'configs/test_master_queue.json').read_text())
    assert len(generated)==3 and '--one' in generated[0]['argv'] and '--one' in generated[1]['argv']
    assert all('CUDA_VISIBLE_DEVICES' not in j.get('env',{}) for j in generated)
    rows=json.loads((tmp_path/'configs/test_master_A_registry.json').read_text())
    runtime=json.loads(Path(rows[0]['config_path']).read_text())
    assert runtime['microbatch_size']==16 and runtime['gradient_accumulation_steps']==4
    assert {k for k in original if original[k]!=runtime[k]}=={'deadline_utc'}
    assert (tmp_path/'configs/test_master_benchmark_registry.json').exists()
    assert not (tmp_path/'logs').exists()

def test_prepared_conditional_phase_activation_preserves_source_and_seed_roles(tmp_path):
    config=tmp_path/'prepared_config.json'
    original={'seed':11,'deadline_utc':'2026-11-07T03:26:32Z','replay_ratio':.05,
              'target_prediction_tokens':10_000_000,'run_label':'draw1001'}
    config.write_text(json.dumps(original));registry=tmp_path/'registry.json'
    rows=[{'id':'draw1001','phase':'conditional_bank_variance','active':False,'status':'prepared_only',
           'training_seed':11,'generation_seed':11,'selection_seed':1001,'config_path':str(config)}]
    registry.write_text(json.dumps(rows));source_bytes=registry.read_bytes()
    preparation=tmp_path/'preparation_manifest.json'
    preparation.write_text(json.dumps({'banks':[{'id':'draw1001','config_sha256':master.sha(config)}]}))
    source_queue=tmp_path/'queue_template.json'
    source_queue.write_text(json.dumps({'prepared_only':True,'launch_authorized':False,
        'jobs':[{'id':'draw1001','argv':['python','make_experiments.py','run','--config',str(config)]}]}))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'status':'passed','completed_runs':36}))
    queue=tmp_path/'master_queue.json';queue.write_text(json.dumps([job('existing')]))
    index=tmp_path/'index.json';index.write_text(json.dumps({'phases':{}}))
    args=argparse.Namespace(master_queue=queue,phase_index=index,phase='B_bank_draws',
        phase_queue=source_queue,phase_registry=registry,depends_on=None,priority=False,
        activate_prepared_phase=False,main_audit=audit,runtime_root=tmp_path/'runtime',
        activation_deadline='2026-11-07T00:00:00Z')
    with pytest.raises(ValueError,match='explicit'):master.append(args)
    assert json.loads(queue.read_text())==[job('existing')]
    args.activate_prepared_phase=True;master.append(args);master.append(args)
    assert registry.read_bytes()==source_bytes and json.loads(config.read_text())==original
    merged=json.loads(queue.read_text());assert [j['id'] for j in merged]==['existing','draw1001']
    runtime_path=Path(merged[1]['argv'][-1]);runtime=json.loads(runtime_path.read_text())
    assert {k for k in original if runtime[k]!=original[k]}=={'deadline_utc'}
    executor=json.loads((runtime_path.parent/'executor_registry.json').read_text())[0]
    assert executor['phase']=='conditional_bank_variance' and 'seed' not in executor
    assert executor['training_seed']==11 and executor['selection_seed']==1001 and executor['active']
    assert executor['config_sha256']==master.sha(runtime_path)
    activation=json.loads((runtime_path.parent/'activation.json').read_text())
    assert activation['analysis_registry']==str(registry) and not activation['source_artifacts_modified']
    args.activation_deadline='2026-11-07T01:00:00Z'
    with pytest.raises((ValueError,FileExistsError)):master.append(args)

def test_concurrent_phase_appends_preserve_both_registries_and_all_jobs(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    queue=tmp_path/'master.json';queue.write_text(json.dumps([job('existing')]))
    index=tmp_path/'index.json';index.write_text(json.dumps({'phases':{}}));arguments=[]
    for phase in ['A','B']:
        local=tmp_path/f'{phase}_queue.json';local.write_text(json.dumps([job(phase)]))
        registry=tmp_path/f'{phase}_registry.json';registry.write_text(json.dumps([{'id':phase}]))
        arguments.append(argparse.Namespace(master_queue=queue,phase_index=index,phase=phase,
            phase_queue=local,phase_registry=registry,depends_on=None,priority=False))
    with ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(master.append,arguments))
    assert {j['id'] for j in json.loads(queue.read_text())}=={'existing','A','B'}
    assert set(json.loads(index.read_text())['phases'])=={'A','B'}
