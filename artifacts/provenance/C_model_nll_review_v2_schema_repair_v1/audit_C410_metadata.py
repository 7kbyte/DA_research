"""Read actual already-open C410 SMALL metadata only; never run reviewer/guard.

C410 is not an enabled model in the new reviewer. This is a field/schema audit,
not a passing actual 1B/2.8B review, check/analyze, or a C410 numerical rerun.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
spec=importlib.util.spec_from_file_location('review_schema',ROOT/'scripts/review_C_model_nll_results_v2.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
FORBIDDEN_SCORES=set();open_denials=[]
def audit_event(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes)):return
    p=Path(args[0]).resolve();mode=args[1]
    # Protect real score JSON, all statistical CSV/bootstrap tables, and binaries.
    if set(p.suffixes)&M.FORBIDDEN_SUFFIXES or str(p) in FORBIDDEN_SCORES:
        open_denials.append(str(p));raise RuntimeError('Payload/score read forbidden in metadata-only audit: '+str(p))
sys.addaudithook(audit_event)
checks=[]
def check(name,value,details=None):
    checks.append({'name':name,'passed':bool(value),'details':details});assert value,name
def load_proof(p):return M.bind(p)
def pproof(p):return {'path':str(Path(p).resolve()),'sha256':M.sha(p)}
def put(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

complete_path=ROOT/'results/supplemental_analysis_v3/extended_pythia_410m_100m_nll/COMPLETE.json'
complete=load_proof(pproof(complete_path))
authority,plan,readiness,journal,repair=[load_proof(complete[k]) for k in ('authority','plan','readiness','first_read_receipt','operational_repair')]
protocol=load_proof(plan['protocol']);ready=load_proof(plan['ready']);manifest=load_proof(plan['manifest'])
for r in manifest['records']:FORBIDDEN_SCORES.add(M.canonical(r['output']))
for j in manifest['jobs']:FORBIDDEN_SCORES.add(M.canonical(j['output']))
for k in ('final','best'):
    for name in ('endpoints.csv','paired_effects.csv','endpoint_statistics.csv','paired_statistics.csv','endpoint_document_bootstrap.json','document_bootstrap.json'):
        FORBIDDEN_SCORES.add(M.canonical(complete_path.parent/k/name))

check('real authority identity is exact lowercase root',authority.get('authorized_by')=='root',{'actual':authority.get('authorized_by')})
check('whole phase root authority booleans',authority.get('allow_score_analysis') is True and authority.get('allow_partial') is False)
for k in ('plan','operational_repair'):
    check('authority binds '+k,M.exact_proof(authority[k],complete[k]))
check('authority manifest and READY',M.exact_proof(authority['manifest'],plan['manifest']) and M.exact_proof(authority['ready'],plan['ready']))
check('readiness plan and repair',M.exact_proof(readiness['plan'],complete['plan']) and M.exact_proof(readiness['operational_repair'],complete['operational_repair']))
check('actual C410 closure metadata shape',readiness['ready'] is True and not readiness['pending_job_ids'] and readiness['closure_job_count']==plan['closure_job_count']==43)
check('source integrity precedes score parse',readiness['score_json_parsed'] is False and readiness['all_source_baseline_alias_output_integrity_verified'] is True)
for k in ('plan','authority','readiness'):
    check('firstread binds '+k,M.exact_proof(journal[k],complete[k]))
check('firstread path and semantic permission declaration',M.canonical(complete['first_read_receipt']['path'])==M.canonical(plan['first_read_receipt_path']) and journal['before_first_score_parse'] is True and journal['complete_phase_source_and_output_SHA_verified'] is True and bool(journal['semantic_access_enabled_at_utc']) and journal['no_test_or_behavior_selection'] is True)
for name,p in [('retained_plan',complete['plan']),('retained_manifest',plan['manifest']),('retained_protocol',plan['protocol'])]:
    check('repair binds '+name,M.exact_proof(repair[name],p))
check('repair does not authorize score or statistical changes',repair['this_is_not_a_ROOT_score_authority'] is True and repair['score_access_authorized'] is False and repair['scientific_or_statistics_changes'] is False)
check('repair preserves journal',M.canonical(repair['actual_first_read_journal_path_preserved'])==M.canonical(plan['first_read_receipt_path']))
check('actual READY manifest proof',ready['ready'] is True and M.exact_proof(ready['manifest'],plan['manifest']))
check('contract identity and endpoint field shapes',complete['parsed_phase']==plan['contract']['phase'] and plan['contract']['logical_endpoints']==48 and plan['contract']['checkpoints']==['final','best'] and complete['endpoints']==[{'endpoint':k,'logical_conditions':24} for k in ('final','best')])
phase=plan['contract']['phase']
check('phase identity fields exist with expected types',phase['name']=='extended_pythia_410m_100m' and phase['model_tag']=='pythia-410m' and phase['data_family']=='extended_v1' and phase['role']=='extended' and phase['rows']==24 and phase['expected_phase']=='formal')
check('one manifest phase shape',len(manifest['phases'])==1 and all(manifest['phases'][0][k]==phase[k] for k in ('name','model_tag','data_family')) and manifest['mode']=='nll')
check('one baseline shape',len(manifest['baseline_groups'])==1)
group=next(iter(manifest['baseline_groups'].values()));identity=plan['contract']['scientific_identity']
check('baseline matched test paths and hashes',group['test_paths']=={k:v['path'] for k,v in identity['test_data'].items()} and group['evaluation_input_sha256']=={v['path']:v['sha256'] for v in identity['test_data'].values()} and group['model_tag']==identity['model_tag']=='pythia-410m' and identity['data_family']=='extended_v1')
M.validate_grid([{'checkpoint':r['kind'],'group':r['group'],'training_seed':r['training_seed']} for r in manifest['records']],M.GROUPS)
check('real 48 records have expected final/best grid',len({r['id'] for r in manifest['records']})==48 and len({r['actual_run_id'] for r in manifest['records']})==24)
for r in manifest['records']:
    assert r['model_tag']=='pythia-410m' and r['phase_name']==phase['name'] and r['data_family']=='extended_v1' and r['phase']=='formal' and r['is_training_alias'] is False and r['run_id']==r['actual_run_id']
    assert r['baseline_group']==group['id'] and r['initial_job_id']==group['initial_job_id']
    assert r['full_training_committed_path']['target_prediction_tokens']==100_000_000
    if r['kind']=='final':assert r['target_prediction_tokens_at_checkpoint']==100_000_000
    for k in ('config_path','summary_path'):assert r['input_sha256'][r[k]]==M.sha(r[k])
    summary=M.read(r['summary_path']);config=M.read(r['config_path']);assert summary['status']=='completed' and summary['config']==config
    for name,value in r['full_training_committed_path']['parent_wall_seconds'].items():assert summary['counters'][name]==value
check('record identity, config/summary small SHA, full-training counter schema',True,{'logical_records':48,'physical_training':24})

# Original C410 used a live path; archived original bytes bind its old SHA.
# Do not read today's live state or silently relabel this historical file.
historic_state=ROOT/'artifacts/provenance/C410_nll_analysis_actual_v3/state_before_authority.json'
state=load_proof(pproof(historic_state));queue_path=ROOT/'artifacts/provenance/C410_early_NLL_actual_v1/enqueue/master_queue_after.json';queue=load_proof(pproof(queue_path))
check('historical state snapshot bytes exactly match original readiness SHA',M.sha(historic_state)==readiness['scheduler_state_sha256'],{'original_analysis_state_path':readiness['scheduler_state'],'historical_byte_copy':str(historic_state),'same_original_bytes':True,'paths_are_different':M.canonical(historic_state)!=M.canonical(readiness['scheduler_state'])})
M.validate_state(plan,queue,state)
check('actual original queue and historical state closure satisfy strict new validator',True,{'closure':43,'integer_rc0':True,'no_stop':True,'actual_wrapper_argv':True,'definition_SHA':True,'metadata_exists_stat_only':True})

jobs={j['id']:j for j in manifest['jobs']};assert jobs[group['initial_job_id']]['kind']=='initial' and jobs[group['initial_job_id']]['output']==group['initial_output']
check('one matched initial actual job',sum(j['kind']=='initial' for j in manifest['jobs'])==1)
attestations={M.canonical(x['attestation']['path']):x for x in readiness['attestations']}
expected_att={M.canonical(M.attestation_path(x['output'])):x['output'] for x in [*manifest['jobs'],*manifest['records']]}
check('complete physical and logical attestation path schema',set(attestations)==set(expected_att) and len(attestations)==len(readiness['attestations']))
for path,e in attestations.items():
    att=load_proof(e['attestation'])
    assert e['manifest_sha256']==plan['manifest']['sha256']==att['manifest_sha256'] and M.canonical(att['manifest'])==M.canonical(plan['manifest']['path'])
    assert att['job_id']==e['job_id'] and att['actual_argv']==jobs[att['job_id']]['argv'] and att['output_sha256']==e['output_sha256'] and set(att['output_sha256'])=={expected_att[path]}
check('all actual attestation argv/proof/hash-map fields match new schema',True,{'attestations':len(attestations),'score_payload_hashes_recomputed':False})
aliases=0
for r in manifest['records']:
    job=jobs[r['inference_job_id']]
    if r['output']!=job['output']:
        M.validate_alias(r,job,manifest,'pythia-410m');att=M.read(M.attestation_path(r['output']))
        assert att['inference_alias'] is True and att['inference_subprocess_wall_seconds']==0 and att['source_model_equivalence']==r['source_model_equivalence'];aliases+=1
check('actual full-model proof and opaque producer source schema',aliases==6 and len(jobs)==1+48-aliases,{'validated_actual_C410_aliases':aliases,'checkpoint_payload_reads':0})
for command in ('check','analyze'):
    tp=ROOT/'artifacts/provenance/C410_nll_analysis_actual_v3'/(command+'_whole_child_timing.json')
    timer=M.validate_timer(pproof(tp),command,complete,plan,readiness['scheduler_state'])
    if command=='analyze':assert M.canonical(M.flag_value(timer['argv'],'--output'))==str(complete_path.parent.resolve())
check('actual check/analyze timers bind plan repair authority output original state argv',True)
for kind in ('final','best'):
    a=M.read(complete_path.parent/kind/'analysis_manifest.json')
    assert a['complete'] is True and a['completed_conditions']==a['expected_conditions']==24 and a['identity']=={'role':'extended','phase_name':phase['name'],'model_tag':'pythia-410m','data_family':'extended_v1','checkpoint':kind} and a['replicate_unit']=='training_seed'
    assert M.exact_proof(a['whole_phase_first_read_receipt'],complete['first_read_receipt']) and M.exact_proof(a['whole_phase_readiness'],complete['readiness'])
    assert a['bootstrap_replicates']==5000 and a['bootstrap_seed']==20261106 and all(isinstance(k,str) and isinstance(v,str) for k,v in a['sources_sha256'].items())
check('actual analysis_manifest schema and firstread/readiness links',True,{'numerical_CSV_or_bootstrap_rows_read':False})

old_path=BASE/'retained/scripts/review_C_model_nll_results_v2.py';new_path=ROOT/'scripts/review_C_model_nll_results_v2.py'
before=old_path.read_text();after=new_path.read_text()
check('production source has exactly one literal substitution',before.replace("authority.get('authorized_by')=='ROOT'","authority.get('authorized_by')=='root'")==after)
old_ast=ast.parse(before);new_ast=ast.parse(after);names=('stats','favorable','documents','delta_vectors','raw_metrics','Resampling','validate_grid')
for name in names:
    assert ast.dump(next(x for x in old_ast.body if getattr(x,'name',None)==name),include_attributes=False)==ast.dump(next(x for x in new_ast.body if getattr(x,'name',None)==name),include_attributes=False)
check('all seven numerical AST nodes unchanged',True,{'nodes':names})
assert not open_denials and not (set(M.IO['paths'])&FORBIDDEN_SCORES)
check('audit performed no score or binary payload reads',True,{'open_denials':open_denials,'scores_hashed':False,'score_semantics_parsed':False,'new_model_metadata_or_scores_read':False})

copies=BASE/'actual_C410_metadata_sources';sources=[]
for path in sorted(M.IO['paths']):
    p=Path(path);assert p.is_relative_to(ROOT)
    dst=copies/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);data=p.read_bytes();dst.write_bytes(data)
    sources.append({'path':path,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'retained_copy':str(dst.relative_to(ROOT))})
put(BASE/'actual_C410_metadata_source_manifest.json',{'schema_version':1,'sources':sources,'score_and_binary_payload_reads':0})
put(BASE/'actual_C410_schema_audit.json',{'status':'passed_metadata_schema_only','checks':checks,'source_inventory':pproof(BASE/'actual_C410_metadata_source_manifest.json'),'real_authority_identity':authority['authorized_by'],'legacy_C410_state_note':"Original check/analyze used the same live path. state_before_authority bytes equal the historical readiness SHA. This audit did not read today's live state, does not relabel the byte-copy path, and does not claim a passing invocation of new reviewer for unsupported C410. Future actual 1B/2.8B check/analyze must originally use the explicitly bound immutable successor state path.",'scope_exclusions':['No numerical rerun or actual reviewer gate invocation','No check/analyze invocation','No score/CSV/bootstrap payload read or hash','No 1B/2.8B score or metadata input','No checkpoint/base/TAR/ZIP read','No GPU']})
print(json.dumps({'status':'passed_metadata_schema_only','checks':len(checks),'small_source_files':len(sources),'score_payload_reads':0,'binary_payload_reads':0}))
