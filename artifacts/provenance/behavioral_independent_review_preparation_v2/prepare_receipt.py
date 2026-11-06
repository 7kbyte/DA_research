"""Independent parent timing and immutable acceptance for 11 source-branch tests."""
from pathlib import Path
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
def proof(path):
    p=Path(path)
    return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def save(name,value):
    path=OUT/name
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return proof(path)

old=ROOT/'artifacts/provenance/behavioral_independent_review_preparation_v1/COMPLETE.json'
assert proof(old)['sha256']=='78ec5253461d3aca7a0b50f48613e30b5c71314a73afcfa70e378ef9aa41d42d'
v1=json.loads(old.read_text())
old_proofs=[proof(old),v1['tests_timer'],v1['tests_stdout'],v1['tests_stderr']]
for p in old_proofs:assert proof(p['path'])==p
for name,digest in [
    ('scripts/review_behavioral_results.py','bb261a20ac45fb47c690f57b7461856b5204066eade15178ca92886a9766966b'),
    ('tests/test_behavioral_independent_review.py','203848cc32a60c178275e6d242e7425cdfdf77af2c24661ed5ef669a4554b86b')]:
    assert proof(ROOT/name)['sha256']==digest
sources=[proof(ROOT/p) for p in ['scripts/review_behavioral_results.py','tests/test_behavioral_independent_review.py',
    'tests/test_behavioral_review_source_branches.py','docs/行为独立复核_来源分支补充验收.md']]+[proof(__file__)]
frozen=v1['frozen_eight_before_after_sha256']
assert all(proof(p)['sha256']==h for p,h in frozen.items())
argv=[sys.executable,'-m','pytest','-q','tests/test_behavioral_review_source_branches.py']
before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter()
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
with (OUT/'tests.stdout.log').open('xb') as stdout,(OUT/'tests.stderr.log').open('xb') as stderr:
    child=subprocess.run(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
timer=save('tests_whole_child_timing.json',{'argv':argv,'returncode':child.returncode,'wall_seconds':time.perf_counter()-start,
    'user_cpu_seconds':after.ru_utime-before.ru_utime,'system_cpu_seconds':after.ru_stime-before.ru_stime,
    'scope':'Only the 11 additional temporary synthetic source-branch tests; original 26-test receipt/timer is retained separately',
    'CPU_threads':2,'CUDA_VISIBLE_DEVICES':'','not_a_real_behavior_analysis_timing':True})
assert child.returncode==0
assert all(proof(p['path'])==p for p in sources+old_proofs)
assert all(proof(p)['sha256']==h for p,h in frozen.items())
result=save('COMPLETE.json',{'complete':True,'prepared_only':True,'supersedes_old_validation':False,
    'retained_preparation_COMPLETE':proof(old),'retained_26_test_evidence':old_proofs[1:],
    'new_source_proofs':sources,'additional_tests':11,'tests_timer':timer,
    'tests_stdout':proof(OUT/'tests.stdout.log'),'tests_stderr':proof(OUT/'tests.stderr.log'),
    'source_branches':['v3 operational_repair','C three-seed D/B4 alias','A imported initial + three reuse jobs; full 22-job closure'],
    'rejection_branches':['repair retained manifest mismatch','repair retained plan mismatch','old authority repair SHA',
        'alias/reuse nonzero forward','alias/reuse wrong current manifest','imported baseline wrong prior manifest'],
    'failures_proven_before_semantic_question_parse':True,'reviewer_source_unchanged':True,
    'frozen_eight_before_after_sha256':frozen,'real_behavior_scores_or_predictions_read':False,
    'original_check_analyze_or_GPU_run':False,'real_authority_or_first_read_created':False,'checkpoint_payload_bytes_read':0,
    'engineering_time':'not instrumented; not represented as zero'})
print(json.dumps({'complete':True,'receipt':result,'timer':timer}))
