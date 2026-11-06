"""Preparation acceptance: source bytes, timed temporary tests, hand-made toy only."""
from pathlib import Path
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from replay_study.selected_policy_analysis import _aggregate


def encoded(x):return (json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
def proof(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'bytes':Path(p).stat().st_size}
def save(name,x):
    p=OUT/name
    with p.open('xb') as f:f.write(encoded(x))
    return proof(p)


manifest=ROOT/'docs/primary_evaluation_v2_execution_matched_behavior_manifest.json'
# This is frozen metadata, never any inference result or prediction.
metadata=json.loads(manifest.read_text())
frozen=metadata['common_input_sha256']
assert all(proof(p)['sha256']==digest for p,digest in frozen.items())
paths=[ROOT/p for p in ('scripts/review_behavioral_results.py','tests/test_behavioral_independent_review.py',
    'docs/行为结果独立数值复核_准备与用法.md','scripts/analyze_behavioral_plan.py',
    'src/replay_study/behavioral_analysis.py','src/replay_study/selected_policy_analysis.py',
    'scripts/run_primary_behavior_analysis_v1.py','scripts/guarded_behavior_analysis.py',
    'src/replay_study/supplemental_analysis_guard_v2.py','src/replay_study/supplemental_analysis_guard_v3.py',
    'scripts/analyze_supplemental_phase_v3.py','docs/主行为分析静态调用链与首次读取门禁.md')]
sources=[proof(p) for p in paths]+[proof(__file__)]
argv=[sys.executable,'-m','pytest','-q','tests/test_behavioral_independent_review.py']
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
started=time.perf_counter();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
with (OUT/'tests.stdout.log').open('xb') as stdout,(OUT/'tests.stderr.log').open('xb') as stderr:
    child=subprocess.run(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
timer=save('tests_whole_child_timing.json',{'argv':argv,'returncode':child.returncode,'wall_seconds':time.perf_counter()-started,
    'user_cpu_seconds':after.ru_utime-usage.ru_utime,'system_cpu_seconds':after.ru_stime-usage.ru_stime,
    'scope':'Entire pytest child and its waited synthetic review CLI descendant; all temporary synthetic inputs only',
    'nested_synthetic_CLI_timer_not_additive':True})
assert child.returncode==0
assert all(proof(p)['sha256']==digest for p,digest in frozen.items())
assert all(proof(p['path'])==p for p in sources)
# Explicit two-score collision with different means; no real checkpoints/tasks.
toy=[]
for metric,values in [('acc_norm',[.1,.2,.3]),('acc',[-.1,0.,.1])]:
    for seed,value in zip((11,22,33),values):
        toy.append({'comparison':'B4-B3','lambda_a':5.,'lambda_b':.1,'task':'SYNTHETIC_piqa',
            'metric':metric,'training_seed':seed,'accuracy_left_minus_right':value})
old=_aggregate(toy,('accuracy_left_minus_right',),('task','metric'))
resolved=[]
for score in ('acc_norm','acc'):
    values=[r['accuracy_left_minus_right'] for r in toy if r['metric']==score]
    resolved.append({'comparison':'B4-B3','score':score,'quantity':'accuracy_left_minus_right',
        'mean':statistics.mean(values),'std_ddof1':statistics.stdev(values),'n_training_seeds':len(values)})
assert len(old)==len(resolved)==2 and {r['metric'] for r in old}=={'accuracy_left_minus_right'}
assert abs(resolved[0]['mean']-.2)<1e-15 and resolved[1]['mean']==0
collision=save('synthetic_label_collision.json',{'synthetic_only':True,'input_seed_pairs':toy,'frozen_original_aggregate':old,
    'independent_score_resolved_view':resolved,'statistics_or_inference_rules_changed':False})
save('COMPLETE.json',{'complete':True,'prepared_only':True,'real_behavior_scores_or_predictions_read':False,
    'original_check_analyze_or_GPU_run':False,'real_authority_or_first_read_created':False,
    'source_proofs':sources,'frozen_eight_before_after_sha256':frozen,'frozen_manifest':proof(manifest),
    'tests_timer':timer,'tests_stdout':proof(OUT/'tests.stdout.log'),'tests_stderr':proof(OUT/'tests.stderr.log'),
    'synthetic_collision':collision,'checkpoint_payload_bytes_read':0,
    'engineering_and_document_writing_time':'not instrumented; not represented as zero'})
print(json.dumps({'complete':True,'receipt':str(OUT/'COMPLETE.json'),'tests_wall_seconds':json.loads(Path(timer['path']).read_text())['wall_seconds']}))
