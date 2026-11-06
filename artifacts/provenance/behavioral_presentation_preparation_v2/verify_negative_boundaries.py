"""Small metadata/CSV refusal probes; no authority or actual score inputs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('present',ROOT/'scripts/present_behavioral_review.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
start=time.perf_counter();before=resource.getrusage(resource.RUSAGE_SELF);checks=[]
def rejected(label,fn):
    try:fn()
    except ValueError as e:checks.append({'case':label,'rejected':True,'reason':str(e)})
    else:raise AssertionError(label+' unexpectedly accepted')
fixture=HERE.with_name('behavioral_presentation_preparation_v1')/'synthetic_inputs/supplement_A/synthetic_fixture.json'
reader=p.SmallInputs();args=SimpleNamespace(synthetic_fixture=fixture)
manifest,tables,family,_,_,_,_=p.load(args,reader)
p.verify_tables(manifest,tables,family)
partial=copy.deepcopy(tables);partial['endpoints'].pop()
rejected('missing logical task/score endpoint',lambda:p.verify_tables(manifest,partial,family))
bad=copy.deepcopy(tables)
for row in bad['selected_policy_score_resolved_statistics']:row['score']='accuracy_left_minus_right'
rejected('A collided score/quantity labels forbidden',lambda:p.verify_tables(manifest,bad,family))
bad_manifest=copy.deepcopy(manifest);bad_manifest['records'][0]['model_tag']='pythia-1b'
rejected('mixed model full phase forbidden',lambda:p.scope(bad_manifest))
rejected('CSV SHA differs before parse',lambda:p.SmallInputs().csv(fixture.parent/'endpoints.csv','0'*64))
rejected('missing explicit production reviewer proof',lambda:p.load(SimpleNamespace(synthetic_fixture=None,review_complete=None,review_complete_sha256=None),p.SmallInputs()))
reader.finish();after=resource.getrusage(resource.RUSAGE_SELF)
with (HERE/'negative_boundaries.json').open('x') as f:json.dump({'complete':True,'synthetic_only':True,'checks':checks,
    'source':{'path':str(Path(__file__)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    'process_scope_wall_seconds':time.perf_counter()-start,'process_scope_user_cpu_seconds':after.ru_utime-before.ru_utime,
    'process_scope_system_cpu_seconds':after.ru_stime-before.ru_stime,
    'timing_scope':'Only in-process toy checks after import, not complete interpreter child wall; engineering total unknown',
    'actual_outputs_or_true_authority_opened_or_created':False},f,indent=2);f.write('\n')
