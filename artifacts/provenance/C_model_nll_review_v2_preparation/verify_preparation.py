"""Explicit small-source and SYNTHETIC receipt QC; no real score/model inputs."""
from pathlib import Path
import ast
import csv
import datetime
import hashlib
import json

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def proof(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}

retained=read(BASE/'retained_source_manifest.json')
for item in retained:
    assert sha(ROOT/item['path'])==item['sha256']==sha(ROOT/item['retained'])
old=ast.parse((ROOT/'scripts/review_C_nll_results.py').read_text());new=ast.parse((ROOT/'scripts/review_C_model_nll_results_v2.py').read_text())
equivalent=[]
for name in ('stats','favorable','documents','delta_vectors','raw_metrics','Resampling','validate_grid'):
    a=next(x for x in old.body if getattr(x,'name',None)==name);b=next(x for x in new.body if getattr(x,'name',None)==name)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False);equivalent.append(name)
core_source=ROOT/'logs/tests_after_behavior_native_restore_v1_full.json';core=read(core_source)['scientific_core_after']
assert len(core)==10
for p,d in core.items():assert sha(ROOT/p)==d

toy=[];stable=BASE/'toy_cli_stable_source'
for model,alias,physical,rows_count,pair_rows in [('pythia-1b',6,43,48,36),('pythia-2.8b',0,31,30,18)]:
    mode='six' if alias else 'none';folder=stable/'toy_cli_inputs'/model/mode/'review';r=read(folder/'independent_review_receipt.json');c=read(folder/'cost_accounting.json')
    assert r['status']=='passed' and r['source']['sha256']==sha(ROOT/'scripts/review_C_model_nll_results_v2.py')
    assert r['endpoint_records']==rows_count and r['paired_records']==pair_rows and r['zero_forward_aliases']==alias and r['physical_inferences']==physical
    assert c['physical_training_runs']==rows_count//2 and c['physical_inferences']==physical and c['physical_initial_inferences']==1 and c['zero_forward_aliases']==alias
    assert len(list(folder.glob('*.csv')))==13 and len(list(folder.glob('*.png')))==4 and len(list(folder.glob('*.pdf')))==4 and len(list(folder.glob('*.svg')))==4
    assert r['IO']['checkpoint_base_archive_payload_reads']==0
    assert all(Path(p).is_relative_to(stable) for p in r['IO']['semantic_json_paths'])
    assert all(Path(p).is_relative_to(stable) or Path(p)==(ROOT/'scripts/review_C_model_nll_results_v2.py') for p in r['IO']['paths'])
    if not alias:
        with (folder/'zero_forward_aliases.csv').open() as f:
            cr=csv.DictReader(f);assert cr.fieldnames==['record_id','run_id','checkpoint','canonical_job_id','inference_subprocess_wall_seconds','source_model_equivalence'] and list(cr)==[]
    toy.append({'model':model,'receipt':proof(folder/'independent_review_receipt.json'),'cost_accounting':proof(folder/'cost_accounting.json'),'logical_endpoints':rows_count,'physical_inferences':physical,'proven_aliases':alias,'paired_records':pair_rows,'max_abs_numerical_error':r['max_absolute_error'],'semantic_inputs_are_synthetic_only':True,'CSV_files':13,'figures':{'png':4,'pdf':4,'svg':4}})

paths=[ROOT/'scripts/review_C_model_nll_results_v2.py',ROOT/'tests/test_C_model_nll_review_v2.py',ROOT/'docs/C1B与C2p8B_NLL完整组独立复核_v2准备.md',BASE/'gates_source.py',BASE/'build_reviewer.py',BASE/'toy_cli_smoke.py',BASE/'verify_preparation.py',BASE/'retained_source_manifest.json',core_source,BASE/'tests_stable_source/whole_child_timing.json',stable/'whole_child_timing.json']
inventory={str(p.relative_to(ROOT)):sha(p) for p in paths};put(BASE/'source_manifest.json',{'schema_version':1,'source_sha256':inventory})
put(BASE/'QC.json',{'status':'passed','original_C410_files_preserved':len(retained),'scientific_core_files_unchanged':10,'core_reference':proof(core_source),'identical_numerical_AST':equivalent,'toy_models':toy,'actual_1B_2p8B_score_semantics_read':False,'actual_guard_check_analyze_invoked':False,'checkpoint_base_archive_payload_reads':0,'GPU_invoked':False})
put(BASE/'COMPLETE.json',{'schema_version':1,'status':'prepared_only','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_manifest':proof(BASE/'source_manifest.json'),'QC':proof(BASE/'QC.json'),'targeted_tests':proof(BASE/'tests_stable_source/whole_child_timing.json'),'toy_cli_whole_parent':proof(stable/'whole_child_timing.json'),'toy_cli_nested':proof(stable/'toy_cli_smoke.json'),'original_C410_files_preserved':len(retained),'scope':'Preparation only; manual toy results are not actual model evaluations and confer no score authority','source_implementation_frozen_after_receipt':True,'engineering_parent_wall_seconds':None,'engineering_parent_timing_status':'unknown; only explicitly recorded test/smoke child operations measured'})
print(json.dumps({'prepared_only':True,'source_manifest':sha(BASE/'source_manifest.json'),'QC':sha(BASE/'QC.json'),'COMPLETE':sha(BASE/'COMPLETE.json'),'reviewer':sha(ROOT/'scripts/review_C_model_nll_results_v2.py')}))
