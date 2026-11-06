import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/select_extended_batch.py'
spec=importlib.util.spec_from_file_location('batch_choice',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def report(micro,speed,reserved=40*1024**3):
    return {'status':'succeeded','model':'pythia-1b','microbatch':micro,'accumulation':64//micro,
            'steady_target_prediction_tokens_per_second':speed,
            'peak_gpu_reserved_bytes':reserved,'gpu_total_memory_bytes':80*1024**3}

def test_batch_choice_requires_speed_and_reserved_memory_margin():
    baseline=report(16,100)
    assert module.choose('pythia-1b',baseline,report(32,106))['selected_microbatch']==32
    assert module.choose('pythia-1b',baseline,report(32,104))['selected_microbatch']==16
    assert module.choose('pythia-1b',baseline,report(32,120,73*1024**3))['selected_microbatch']==16
    assert module.choose('pythia-1b',baseline,report(32,105,72*1024**3))['selected_microbatch']==32

def test_failed_candidate_keeps_measured_baseline_and_failure_cost():
    candidate={'status':'failed','model':'pythia-1b','failed_wall_seconds':3.5}
    selected=module.choose('pythia-1b',report(16,100),candidate)
    assert selected['selected_microbatch']==16 and selected['failed_attempts']==1
    assert selected['failed_attempt_seconds']==3.5
    with pytest.raises(ValueError,match='model'):module.choose('pythia-410m',report(16,100),candidate)
