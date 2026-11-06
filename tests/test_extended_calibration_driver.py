import argparse,importlib.util,json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/advance_extended_calibration.py'
spec=importlib.util.spec_from_file_location('calibration_driver',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path

def test_cpu_driver_never_preempts_main_or_starts_an_outer_master(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'cpu_command',lambda *args:pytest.fail('CPU phase must wait for main/master'))
    args=argparse.Namespace(main_audit=tmp_path/'audit.json',master_state=tmp_path/'state.json')
    assert module.advance(args)['status']=='waiting_main36_audit'
    write(args.main_audit,{'status':'passed','completed_runs':36})
    assert module.advance(args)['status']=='waiting_existing_master'
    write(args.master_state,{'stopping':True,'jobs':{}})
    assert module.advance(args)['status']=='outer_master_stopping'

def test_phase_completion_cannot_ignore_the_initial_model_baseline(tmp_path):
    paths={'queue':write(tmp_path/'queue.json',[{'id':'initial'},{'id':'lr1'},{'id':'lr2'},{'id':'lr3'}])}
    state={'jobs':{j['id']:{'status':'completed'} for j in module.read(paths['queue'])}}
    assert module.completion(paths,state)==(True,[])
    state['jobs']['initial']['status']='failed'
    assert module.completion(paths,state)==(False,['initial'])

def test_registered_phase_is_reused_without_rewriting_its_sources(tmp_path,monkeypatch):
    paths={'namespace':'frozen_phase','registry':write(tmp_path/'registry.json',[{'id':'a'}]),
           'queue':write(tmp_path/'queue.json',[{'id':'a'}]),'protocol':tmp_path/'protocol.json'}
    index={'phases':{'frozen_phase':{'registry_sha256':module.digest(paths['registry']),
                                   'queue_source_sha256':module.digest(paths['queue'])}}}
    args=argparse.Namespace(phase_index=write(tmp_path/'index.json',index))
    monkeypatch.setattr(module,'phase_paths',lambda *args:paths)
    monkeypatch.setattr(module,'cpu_command',lambda *args:pytest.fail('Registered phase must remain immutable'))
    assert module.ensure_phase(args,'pythia-1b',16,'lr',tmp_path/'batch.json')==paths
    write(paths['registry'],[{'id':'changed'}])
    with pytest.raises(ValueError,match='artifacts changed'):
        module.ensure_phase(args,'pythia-1b',16,'lr',tmp_path/'batch.json')
