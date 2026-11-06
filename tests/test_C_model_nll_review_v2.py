"""Manual small full-grid fixtures; no real study outputs or checkpoint payloads."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('c_model_review',ROOT/'scripts/review_C_model_nll_results_v2.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)

def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def opaque(path,value):
    path.write_text(value);return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def csvput(path,values):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)

def corpus(delta=0):
    # Two independent documents, unequal fixed token counts, manually set loss.
    chunks=[{'doc_id':'a','chunk_index':0,'valid_prediction_tokens':1,'sum_nll':2.+delta},
            {'doc_id':'b','chunk_index':1,'valid_prediction_tokens':3,'sum_nll':9.+3*delta}]
    return {'complete_corpus':True,'chunks':chunks,'sum_nll':11.+4*delta,
            'valid_prediction_tokens':4,'nll':2.75+delta,'kl':0.,'teacher_forward_seconds':.01}

def metric_values(delta):
    return {m+suffix:sign*delta for m,sign in zip(M.METRICS,(-1,1,1)) for suffix in ('','_equal_document')}

def interval(delta):
    # Both document NLL increments equal delta, so every one of 5000 draws
    # has exactly delta. This expected answer does not call reviewer/bootstrap.
    return {'estimate':delta,'ci_low':delta,'ci_high':delta}

def fixture(tmp_path,model='pythia-1b',alias_mode='none'):
    p=M.PROFILES[model];n=len(p['groups'])*3;phase=p['phase'];cid=phase+'_nll'
    root=tmp_path/model/alias_mode;root.mkdir(parents=True);out=root/'analysis';out.mkdir()
    state_path=root/'actual_executor_state_snapshot.json';queue_path=root/'actual_executor_queue.json'
    manifest_path=root/'manifest.json';protocol=put(root/'protocol.json',{'schema_version':3})
    producer=opaque(root/'producer.py','# real Python bytes, not JSON\npass\n')
    registry=put(root/'registry.json',{'model':model});snapshot=put(root/'snapshot.json',{'model':model});audit=put(root/'audit.json',{'status':'passed'})
    initial=str(root/'initial.json');put(initial,{'corpora':{c:corpus() for c in ('target','c4','wikitext')}})
    matched={c:{'path':str(root/(c+'_test.pt')),'sha256':'f'*64} for c in ('target','c4','wikitext')}
    group={'id':'matched_extended','model_tag':model,'initial_job_id':'initial','initial_output':initial,
           'test_paths':{k:v['path'] for k,v in matched.items()},'evaluation_input_sha256':{v['path']:v['sha256'] for v in matched.values()}}
    jobs=[{'id':'initial','kind':'initial','output':initial,'argv':['python','run_evaluate.py','--output',initial]}]
    records=[];endpoint_rows=[];pair_rows=[];deltas={};run_by_key={};proof_pairs={'final':[],'best':[]}
    for kind in ('final','best'):
        for gi,g in enumerate(p['groups']):
            for si,s in enumerate((11,22,33)):
                run=f'{model}_{g}_s{s}';rid=run+'_'+kind;run_by_key[kind,g,s]=run
                # Full-state aliases are manually equal only where a proof is supplied.
                is_alias=g=='D' and (alias_mode=='six' or alias_mode=='three' and kind=='final')
                effective='B4' if is_alias else g
                value=.01*(p['groups'].index(effective)+1)+.001*si+.002*(kind=='best')
                deltas[kind,g,s]=value
                output=str(root/(rid+'.json'));put(output,{'corpora':{c:corpus(value) for c in ('target','c4','wikitext')}})
                cfg={'id':run,'seed':s,'model':model,'replay_path':None if g=='B0' else '/toy/bank.pt'}
                counters={'setup_seconds':1.,'training_seconds':2.,'validation_seconds':3.,'checkpoint_seconds':4.,
                          'training_teacher_forward_seconds':.1,'validation_teacher_forward_seconds':.2,'validation_kl_arithmetic_seconds':.3}
                cfgp=root/(run+'_config.json');sp=root/(run+'_summary.json');cp=put(cfgp,cfg);su=put(sp,{'status':'completed','config':cfg,'counters':counters})
                target=p['horizon'] if kind=='final' else p['horizon']//2;replay=0 if g=='B0' else target//20
                rec={'id':rid,'run_id':run,'actual_run_id':run,'phase':'formal','phase_name':phase,'role':'extended','data_family':'extended_v1','model_tag':model,'kind':kind,'group':g,'training_seed':s,'is_training_alias':False,
                     'checkpoint':str(root/'checkpoints'/(rid+'.pt')),'checkpoint_sha256':hashlib.sha256(rid.encode()).hexdigest(),'weights_identity':'checkpoint:'+rid,
                     'source_model_equivalence':None,'baseline_group':group['id'],'initial_job_id':'initial','inference_job_id':rid,'output':output,'step':10 if kind=='final' else 5,
                     'target_prediction_tokens_at_checkpoint':target,'replay_prediction_tokens_at_checkpoint':replay,
                     'full_training_committed_path':{'target_prediction_tokens':p['horizon'],'replay_prediction_tokens':0 if g=='B0' else p['horizon']//20,'parent_wall_seconds':{k:counters[k] for k in ('setup_seconds','training_seconds','validation_seconds','checkpoint_seconds')}},
                     'config_path':str(cfgp),'summary_path':str(sp),'input_sha256':{str(cfgp):cp['sha256'],str(sp):su['sha256']}}
                if is_alias:
                    canonical=next(r for r in records if r['kind']==kind and r['group']=='B4' and r['training_seed']==s)
                    rec['inference_job_id']=canonical['inference_job_id'];fp={'sha256_model_state':hashlib.sha256((kind+str(s)).encode()).hexdigest(),'state_elements':2}
                    canonical['weights_identity']=rec['weights_identity']='model_state:'+fp['sha256_model_state']
                    sides=[]
                    for r in (canonical,rec):
                        sides.append({'checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],'model_state_fingerprint':fp,'run_id':r['run_id'],'kind':kind,
                                      'checkpoint_exposure':{k:r[k] for k in ('step','target_prediction_tokens_at_checkpoint','replay_prediction_tokens_at_checkpoint')},
                                      'config_path':r['config_path'],'config_sha256':r['input_sha256'][r['config_path']],'summary_path':r['summary_path'],'summary_sha256':r['input_sha256'][r['summary_path']]})
                    proof_pairs[kind].append({'left':sides[0],'right':sides[1],'bitwise_model_state_equal':True,'endpoint':kind,'seed':s})
                    rec['source_model_equivalence']={'proof':str(root/(kind+'_equivalence.json')),'proof_sha256':'pending','comparison_index':len(proof_pairs[kind])-1,'algorithm':'sha256_sorted_key_dtype_shape_contiguous_uint8_v1','fingerprint':fp,'left_checkpoint_sha256':sides[0]['checkpoint_sha256'],'right_checkpoint_sha256':sides[1]['checkpoint_sha256']}
                else:jobs.append({'id':rid,'kind':'adapted','output':output,'argv':['python','run_evaluate.py','--output',output]})
                records.append(rec)
                endpoint_rows.append({'checkpoint':kind,'group':g,'training_seed':s,'run_id':run,'actual_run_id':run,'record_id':rid,'target_prediction_tokens':target,'replay_prediction_tokens':replay,'full_training_target_prediction_tokens':p['horizon'],'full_training_replay_prediction_tokens':rec['full_training_committed_path']['replay_prediction_tokens'],**metric_values(value)})
    for kind in ('final','best'):
        if proof_pairs[kind]:
            pr=put(root/(kind+'_equivalence.json'),{'schema_version':1,'status':'passed','model':model,'endpoint':kind,'algorithm':'sha256_sorted_key_dtype_shape_contiguous_uint8_v1','algorithm_source':producer,'producer_source':producer,'source_audit':audit,'source_registry':registry,'source_snapshot':snapshot,'comparisons':proof_pairs[kind]})
            for r in records:
                if r['kind']==kind and r['source_model_equivalence']:r['source_model_equivalence']['proof_sha256']=pr['sha256']
        for pair in p['pairs']:
            a,b=pair.split('-')
            for s in (11,22,33):pair_rows.append({'checkpoint':kind,'comparison':pair,'training_seed':s,'run_a':run_by_key[kind,a,s],'run_b':run_by_key[kind,b,s],**metric_values(deltas[kind,a,s]-deltas[kind,b,s])})
    manifest=put(manifest_path,{'mode':'nll','phases':[{'name':phase,'model_tag':model,'data_family':'extended_v1'}],'baseline_groups':{group['id']:group},'records':records,'jobs':jobs})
    closure=[];queue=[];state={'jobs':{}}
    for job in jobs:
        argv=['python',str(root/'run_frozen_evaluation.py'),'--manifest',manifest['path'],'--manifest-sha256',manifest['sha256'],'--job-id',job['id'],'--device','cuda:0']
        definition={'id':job['id'],'argv':argv,'expected_seconds':300,'outputs':[job['output'],M.attestation_path(job['output'])]}
        queue.append(definition);closure.append({'job_id':job['id'],'definition':definition,'manifest':manifest})
        state['jobs'][job['id']]={'status':'completed','returncode':0,'definition_sha256':M.definition_sha(definition),'argv':argv,'elapsed_seconds':.5,'started_at_epoch':1.,'missing_outputs':[]}
    qp=put(queue_path,queue);st=put(state_path,state)
    ar=put(root/'READY.json',{'ready':True,'manifest':manifest,'queue':qp})
    phase_identity={'name':phase,'model_tag':model,'data_family':'extended_v1','role':'extended','rows':n,'expected_phase':'formal'}
    plan=put(root/'plan.json',{'contract_id':cid,'contract':{'phase':phase_identity,'logical_endpoints':2*n,'checkpoints':['final','best'],'scientific_identity':{'model_tag':model,'data_family':'extended_v1','test_data':matched}},'closure_job_count':len(closure),'closure_jobs':closure,'manifest':manifest,'protocol':protocol,'ready':ar,'first_read_receipt_path':str(root/'firstread.json')})
    repair=put(root/'repair.json',{'retained_plan':plan,'retained_manifest':manifest,'retained_protocol':protocol,'this_is_not_a_ROOT_score_authority':True,'score_access_authorized':False,'scientific_or_statistics_changes':False,'actual_first_read_journal_path_preserved':str(root/'firstread.json')})
    authority=put(root/'authority.json',{'contract_id':cid,'authorized_by':'root','allow_score_analysis':True,'allow_partial':False,'plan':plan,'operational_repair':repair,'manifest':manifest,'ready':ar})
    att_evidence=[]
    for output,record in [(j['output'],None) for j in jobs]+[(r['output'],r) for r in records if r['output']!=next(j for j in jobs if j['id']==r['inference_job_id'])['output']]:
        job=next(j for j in jobs if j['id']==(record['inference_job_id'] if record else next(j['id'] for j in jobs if j['output']==output)))
        alias=record is not None;hashes={output:hashlib.sha256(Path(output).read_bytes()).hexdigest()}
        att={'manifest':manifest['path'],'manifest_sha256':manifest['sha256'],'job_id':job['id'],'actual_argv':job['argv'],'output_sha256':hashes,'inference_alias':alias,'inference_subprocess_wall_seconds':0 if alias else .3,'verification_wall_seconds':.1,'source_model_equivalence':record['source_model_equivalence'] if alias else None}
        ap=put(M.attestation_path(output),att);att_evidence.append({'manifest_sha256':manifest['sha256'],'job_id':job['id'],'logical_record_id':record['id'] if alias else None,'attestation':ap,'output_sha256':hashes})
    readiness=put(root/'readiness.json',{'ready':True,'contract_id':cid,'closure_job_count':len(closure),'pending_job_ids':[],'plan':plan,'operational_repair':repair,'score_json_parsed':False,'all_source_baseline_alias_output_integrity_verified':True,'scheduler_state':str(state_path),'scheduler_state_sha256':st['sha256'],'attestations':att_evidence})
    journal=put(root/'firstread.json',{'plan':plan,'authority':authority,'readiness':readiness,'before_first_score_parse':True,'complete_phase_source_and_output_SHA_verified':True,'no_test_or_behavior_selection':True,'semantic_access_enabled_at_utc':'SYNTHETIC_ONLY'})
    complete=put(out/'COMPLETE.json',{'complete':True,'endpoints':[{'endpoint':k,'logical_conditions':n} for k in ('final','best')],'authority':authority,'plan':plan,'readiness':readiness,'first_read_receipt':journal,'operational_repair':repair,'parsed_phase':phase_identity})
    for kind in ('final','best'):
        folder=out/kind;folder.mkdir();er=[r for r in endpoint_rows if r['checkpoint']==kind];pr=[r for r in pair_rows if r['checkpoint']==kind]
        csvput(folder/'endpoints.csv',er);csvput(folder/'paired_effects.csv',pr)
        for name,key,values,categories in [('endpoint_statistics','group',er,p['groups']),('paired_statistics','comparison',pr,p['pairs'])]:
            sr=[]
            for cat in categories:
                for metric in M.SIX:
                    vals=[r[metric] for r in values if r[key]==cat]
                    sr.append({key:cat,'metric':metric,'mean':statistics.mean(vals),'std_ddof1':statistics.stdev(vals)})
            csvput(folder/(name+'.csv'),sr)
        eci=[];pci=[]
        for r in er:
            for c,sign in zip(('target','c4','wikitext'),(-1,1,1)):
                d=deltas[kind,r['group'],r['training_seed']]*sign
                eci.append({'field':'sum_nll','corpus':c,'run_id':r['run_id'],'training_seed':r['training_seed'],'target_prediction_tokens':r['target_prediction_tokens'],'replay_prediction_tokens':r['replay_prediction_tokens'],'token_weighted':interval(d),'equal_document':interval(d)})
        for r in pr:
            a,b=r['comparison'].split('-');d=deltas[kind,a,r['training_seed']]-deltas[kind,b,r['training_seed']]
            for c in ('target','c4','wikitext'):
                pci.append({'field':'sum_nll','corpus':c,**{k:r[k] for k in ('comparison','training_seed','run_a','run_b')},'target_tokens_a':p['horizon'] if kind=='final' else p['horizon']//2,'target_tokens_b':p['horizon'] if kind=='final' else p['horizon']//2,'equal_checkpoint_exposure':True,'token_weighted':interval(d),'equal_document':interval(d)})
        put(folder/'endpoint_document_bootstrap.json',eci);put(folder/'document_bootstrap.json',pci)
        put(folder/'analysis_manifest.json',{'complete':True,'completed_conditions':n,'expected_conditions':n,'identity':{'role':'extended','phase_name':phase,'model_tag':model,'data_family':'extended_v1','checkpoint':kind},'replicate_unit':'training_seed','bootstrap_replicates':5000,'bootstrap_seed':20261106,'sources_sha256':{},'whole_phase_first_read_receipt':journal,'whole_phase_readiness':readiness})
    kwargs={}
    for command in ('check','analyze'):
        argv=['python',str(root/'analyze_supplemental_phase_v3.py'),command,'--plan',plan['path'],'--plan-sha256',plan['sha256'],'--operational-repair',repair['path'],'--operational-repair-sha256',repair['sha256'],'--state',str(state_path),'--output',str(out)]
        if command=='analyze':argv+=['--authority',authority['path'],'--authority-sha256',authority['sha256']]
        value=put(root/(command+'_timing.json'),{'returncode':0,'argv':argv,'wall_seconds':1.,'user_cpu_seconds':.5,'sys_cpu_seconds':.1})
        kwargs[command+'_timing']=Path(value['path']);kwargs[command+'_timing_sha256']=value['sha256']
    return argparse.Namespace(model=model,complete=Path(complete['path']),complete_sha256=complete['sha256'],authority_sha256=authority['sha256'],queue=queue_path,queue_sha256=qp['sha256'],state=state_path,state_sha256=st['sha256'],output=root/'review',**kwargs)

@pytest.fixture(autouse=True)
def reset_io():
    M.BOUND.clear();M.IO.update(read_bytes=0,paths=set(),semantic_json_paths=set())

@pytest.mark.parametrize('model,mode,logical,physical,aliases,pairs',[
    ('pythia-1b','none',48,49,0,6),('pythia-1b','three',48,46,3,6),('pythia-1b','six',48,43,6,6),('pythia-2.8b','none',30,31,0,3)])
def test_full_toy_review(tmp_path,monkeypatch,model,mode,logical,physical,aliases,pairs):
    args=fixture(tmp_path,model,mode);monkeypatch.setattr(M,'plot',lambda *a:None);M.run(args)
    receipt=json.loads((args.output/'independent_review_receipt.json').read_text());cost=json.loads((args.output/'cost_accounting.json').read_text())
    assert receipt['status']=='passed' and receipt['endpoint_records']==logical
    assert receipt['paired_records']==2*3*pairs and receipt['endpoint_conditional_intervals']==logical*6 and receipt['paired_conditional_intervals']==2*3*pairs*6
    assert cost['physical_training_runs']==logical//2 and cost['physical_inferences']==physical and cost['zero_forward_aliases']==aliases
    assert cost['training_committed_parent_sum_seconds']==logical//2*10.
    alias_rows=M.rows(args.output/'zero_forward_aliases.csv');assert len(alias_rows)==aliases
    if not aliases: assert (args.output/'zero_forward_aliases.csv').read_text().splitlines()[0].startswith('record_id,')
    assert not any(Path(x).suffix=='.pt' for x in M.IO['paths'])

@pytest.mark.parametrize('change',['bool_rc','float_rc','stop','stop_false','argv','definition','foreign','missing_output'])
def test_actual_state_strict(tmp_path,change):
    args=fixture(tmp_path,'pythia-2.8b');plan=json.loads(Path(json.loads(args.complete.read_text())['plan']['path']).read_text());queue=json.loads(args.queue.read_text());state=json.loads(args.state.read_text());r=state['jobs']['initial']
    if change=='bool_rc':r['returncode']=False
    elif change=='float_rc':r['returncode']=0.
    elif change=='stop':r['stop_requested_at']=1.
    elif change=='stop_false':r['stop_requested']=False
    elif change=='argv':r['argv']=['python','foreign.py']
    elif change=='definition':r['definition_sha256']='0'*64
    elif change=='foreign':state={'jobs':{}}
    else:Path(queue[0]['outputs'][0]).unlink()
    with pytest.raises(ValueError):M.validate_state(plan,queue,state)

@pytest.mark.parametrize('bad',['model','incomplete','authority','state_sha','state_path'])
def test_gate_rejects_before_score_parse(tmp_path,bad):
    args=fixture(tmp_path,'pythia-2.8b');complete=json.loads(args.complete.read_text())
    if bad=='model':args.model='pythia-1b'
    elif bad=='incomplete':complete['endpoints'][0]['logical_conditions']=14;args.complete_sha256=put(args.complete,complete)['sha256']
    elif bad=='authority':args.authority_sha256='0'*64
    elif bad=='state_sha':args.state_sha256='0'*64
    else:
        other=args.state.with_name('unbound_copy.json');other.write_bytes(args.state.read_bytes());args.state=other
    with pytest.raises(ValueError):M.gate_inputs(args)
    assert not any(Path(x).name=='initial.json' for x in M.IO['semantic_json_paths'])

def test_no_foreign_group_or_contrast():
    g=M.PROFILES['pythia-2.8b']['groups'];endpoints=[{'checkpoint':k,'group':v,'training_seed':s} for k in ('final','best') for v in g for s in M.SEEDS]
    M.validate_grid(endpoints,g);endpoints[0]['group']='D'
    with pytest.raises(ValueError):M.validate_grid(endpoints,g)
    ps=[{'checkpoint':k,'comparison':p,'training_seed':s} for k in ('final','best') for p in M.PROFILES['pythia-2.8b']['pairs'] for s in M.SEEDS];ps[0]['comparison']='C-B4'
    with pytest.raises(ValueError):M.validate_pairs(ps,g,M.PROFILES['pythia-2.8b']['pairs'])

@pytest.mark.parametrize('suffix',['.pt','.safetensors','.tar','.zip','.bin','.npy','.tar.gz'])
def test_binary_payload_block_before_read(tmp_path,suffix):
    p=tmp_path/('missing'+suffix)
    with pytest.raises(ValueError,match='payload'):M.sha(p)

def test_numerical_functions_identical_to_retained_v1():
    old=ast.parse((ROOT/'scripts/review_C_nll_results.py').read_text());new=ast.parse((ROOT/'scripts/review_C_model_nll_results_v2.py').read_text())
    for name in ('stats','favorable','documents','delta_vectors','raw_metrics','Resampling','validate_grid'):
        a=next(x for x in old.body if getattr(x,'name',None)==name);b=next(x for x in new.body if getattr(x,'name',None)==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

def test_nonconstant_bootstrap_matches_direct_manual_draws():
    context=M.Resampling(['a','b'],np.array([1.,3.]));actual=context.interval(['a','b'],np.array([1.,3.]),np.array([2.,8.]))
    ids=np.random.default_rng(20261106).integers(0,2,size=(5000,2));n=np.array([1.,3.]);d=np.array([2.,8.])
    token=d[ids].sum(1)/n[ids].sum(1);equal=(d/n)[ids].mean(1)
    for name,values in [('token_weighted',token),('equal_document',equal)]:
        assert actual[name]['ci_low']==np.quantile(values,.025) and actual[name]['ci_high']==np.quantile(values,.975)

def test_python_producer_is_opaque_and_mutation_rejected(tmp_path):
    args=fixture(tmp_path,'pythia-1b','six');gate=M.gate_inputs(args)
    producer=args.complete.parent.parent/'producer.py'
    assert str(producer.resolve()) in M.IO['paths'] and str(producer.resolve()) not in M.IO['semantic_json_paths']
    producer.write_text('raise RuntimeError("changed")\n')
    r=next(x for x in gate['manifest']['records'] if x['source_model_equivalence']);j=next(x for x in gate['manifest']['jobs'] if x['id']==r['inference_job_id'])
    with pytest.raises(ValueError):M.validate_alias(r,j,gate['manifest'],args.model)

def test_synthetic_plot_smoke(tmp_path):
    args=fixture(tmp_path,'pythia-2.8b');endpoints=[];pairs=[]
    for kind in ('final','best'):
        endpoints+=M.rows(args.complete.parent/kind/'endpoints.csv');pairs+=M.rows(args.complete.parent/kind/'paired_effects.csv')
    target=tmp_path/'plots';target.mkdir();M.plot(endpoints,pairs,target,args.model,M.PROFILES[args.model])
    assert len(list(target.glob('*.png')))==4 and len(list(target.iterdir()))==12

@pytest.mark.parametrize('change',['false_pair','foreign_cp','fingerprint','unknown_algorithm'])
def test_alias_false_or_foreign_proof_rejected(tmp_path,change):
    args=fixture(tmp_path,'pythia-1b','six');gate=M.gate_inputs(args)
    r=next(x for x in gate['manifest']['records'] if x['source_model_equivalence']);job=next(x for x in gate['manifest']['jobs'] if x['id']==r['inference_job_id'])
    eq=r['source_model_equivalence'];p=Path(eq['proof']);value=json.loads(p.read_text());pair=value['comparisons'][eq['comparison_index']]
    if change=='false_pair':pair['bitwise_model_state_equal']=False
    elif change=='foreign_cp':pair['left']['checkpoint']='/foreign/checkpoints/not-read.pt'
    elif change=='fingerprint':pair['right']['model_state_fingerprint']['sha256_model_state']='0'*64
    else:value['algorithm']='unsupported'
    eq['proof_sha256']=put(p,value)['sha256'];M.BOUND.pop(str(p.resolve()),None)
    with pytest.raises(ValueError):M.validate_alias(r,job,gate['manifest'],args.model)

@pytest.mark.parametrize('change',['ci_duplicate','ci_foreign_exposure','csv_foreign_group'])
def test_numerical_input_scope_rejected(tmp_path,monkeypatch,change):
    args=fixture(tmp_path,'pythia-2.8b');monkeypatch.setattr(M,'plot',lambda *x:None)
    if change=='csv_foreign_group':
        path=args.complete.parent/'final/endpoints.csv';values=M.rows(path);values[0]['group'],values[3]['group']=values[3]['group'],values[0]['group'];csvput(path,values)
    else:
        path=args.complete.parent/'final/endpoint_document_bootstrap.json';values=json.loads(path.read_text())
        if change=='ci_duplicate':values[1]=values[0]
        else:values[0]['target_prediction_tokens']-=1
        put(path,values)
    with pytest.raises(ValueError):M.run(args)
    assert not (args.output/'independent_review_receipt.json').exists()

@pytest.mark.parametrize('identity',['ROOT','Root','foreign',None])
def test_authority_identity_matches_actual_lowercase_schema(tmp_path,identity):
    args=fixture(tmp_path,'pythia-2.8b');complete=json.loads(args.complete.read_text())
    path=Path(complete['authority']['path']);authority=json.loads(path.read_text());authority['authorized_by']=identity
    complete['authority']=put(path,authority);args.authority_sha256=complete['authority']['sha256'];args.complete_sha256=put(args.complete,complete)['sha256']
    with pytest.raises(ValueError,match='ROOT score authority'):M.gate_inputs(args)
    assert not any(Path(p).name=='initial.json' for p in M.IO['semantic_json_paths'])

def test_original_check_and_analyze_must_use_same_bound_snapshot(tmp_path):
    args=fixture(tmp_path,'pythia-1b');gate=M.gate_inputs(args)
    timer=json.loads(args.check_timing.read_text());at=timer['argv'].index('--state')+1;timer['argv'][at]='/unbound/other_executor_state.json'
    value=put(args.check_timing,timer);M.BOUND.pop(str(args.check_timing.resolve()),None)
    with pytest.raises(ValueError,match='actual scheduler state differs'):
        M.validate_timer(value,'check',gate['complete'],gate['plan'],args.state)
