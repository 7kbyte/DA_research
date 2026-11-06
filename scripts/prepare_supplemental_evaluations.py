#!/usr/bin/env python3
"""CPU completion watcher: freeze valid A/B/C evaluation plans, never launch them."""
import argparse
import copy
import datetime as dt
from fractions import Fraction
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluation_plan import Hashes,absolute,build,encoded,endpoint,read,sha


def proof(path):return {'path':str(absolute(path)),'sha256':sha(absolute(path))}
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(encoded(value));tmp.replace(path)


def catalog():
    primary_spec=ROOT/'configs/primary_evaluation_v2_execution_matched_phase_spec.json';base=read(primary_spec)['phases'][0]
    nll=proof(ROOT/'docs/primary_evaluation_v2_execution_matched_nll_manifest.json')
    behavior=proof(ROOT/'docs/primary_evaluation_v2_execution_matched_behavior_manifest.json')
    initial=proof(ROOT/'docs/planned_model_behavioral_timing_v1_manifest.json')
    scope=proof(ROOT/'configs/behavioral_scope_execution_freeze_v2.json');nllscope=proof(ROOT/'configs/nll_scope_execution_freeze_v1.json')
    snapshot_path=ROOT/'artifacts/provenance/supplemental_v1_orchestration_snapshot.json';snapshot=read(snapshot_path)
    ap=ROOT/'docs/损失权重敏感性预注册_lambda_sweep_deterministic.json';prereg=read(ap)
    ar=snapshot['initial_phase_registries']['A_lambda'];Hashes().proof(ar)
    original=ROOT/'configs/lambda_sweep_deterministic_registry.json';Hashes().proof({'path':original,'sha256':prereg['registry_sha256']})
    main_data={k:copy.deepcopy(base[k]) for k in ('model','model_tag','training_data','validation_data','test_data','data_family')}
    a={'family':'A','name':'supplement_A','role':'supplement_A','expected_phase':'supplement_A_lambda_sweep','expected_rows':126,
       'registry':ar,'protocol':proof(ap),'source_proof_files':{str(snapshot_path):sha(snapshot_path),str(original):prereg['registry_sha256'],str(primary_spec):sha(primary_spec)},
       'protocol_expect':{'data_version':'v3','target_prediction_tokens_per_run':10000000,'replay_ratio':.05},**main_data}
    activation_path=ROOT/'configs/supplemental_v1/activated/B_conditional_bank_variance/activation.json';activation=read(activation_path)
    br={'path':activation['execution_registry'],'sha256':activation['execution_registry_sha256']};Hashes().proof(br)
    b={'family':'B','name':'supplement_B','role':'supplement_B','expected_phase':'conditional_bank_variance','expected_rows':60,
       'registry':br,'protocol':proof(activation_path),'protocol_expect':{'phase':'conditional_bank_variance','source_artifacts_modified':False},
       'source_proof_files':{str(activation_path):sha(activation_path),str(absolute(activation['preparation_manifest'])):activation['preparation_manifest_sha256'],
                             str(absolute(activation['source_registry'])):activation['source_registry_sha256'],str(primary_spec):sha(primary_spec)},**main_data}
    phases=[a,b];freeze_path=ROOT/'results/extended_formal_freeze.json';freeze=read(freeze_path)
    for source in freeze['phases']:
        sp=absolute(source['snapshot']);snap=read(sp);pp=absolute(source['protocol']);p=read(pp);rows=read(source['active_registry']);cfg=read(rows[0]['config_path'])
        if source['active_registry_sha256']!=snap['active_registry_sha256'] or source['protocol_sha256']!=snap['protocol_sha256']:raise ValueError('C source freeze authorities disagree')
        def frozen_data(path):
            path=absolute(path);return {'path':str(path),'sha256':snap['data'][str(path.relative_to(ROOT))]['sha256']}
        tag=source['model'];name='extended_'+tag.replace('-','_').replace('.','p')+('_25m' if tag=='pythia-2.8b' else '_100m')
        phase={'family':'C','name':name,'role':'extended','expected_phase':'formal','expected_rows':source['runs'],
            'model_tag':tag,'model':{'path':cfg['model_name_or_path'],'revision':p['model_revision']},'data_family':'extended_v1',
            'registry':{'path':source['active_registry'],'sha256':source['active_registry_sha256']},
            'protocol':{'path':str(pp),'sha256':source['protocol_sha256']},
            'protocol_expect':{'model':tag,'data_version':'extended_v1','model_revision':p['model_revision'],
                               'formal_target_prediction_tokens':source['target_prediction_tokens'],'shared_learning_rate':p['shared_learning_rate'],'shared_lambda':p['shared_lambda']},
            'training_data':frozen_data(cfg['target_path']),'validation_data':{k:frozen_data(v) for k,v in cfg['validation_paths'].items()},
            'test_data':{k:frozen_data(v) for k,v in rows[0]['test_paths'].items()},
            'source_proof_files':{str(sp):source['snapshot_sha256'],str(freeze_path):sha(freeze_path)}}
        if tag in ('pythia-410m','pythia-1b'):
            directory=ROOT/'results/extended_phase_equivalence'/snap['configuration_prefix']
            phase['model_equivalence_gate']={'phase':snap['configuration_prefix'],'completion_path':str(directory/'completion.json'),
                'audit_path':str(directory/'audit.json'),'proof_paths':{kind:str(directory/(kind+'_model_equivalence_proofs.json')) for kind in ('final','best')},
                'source_snapshot':{'path':str(sp),'sha256':source['snapshot_sha256']},
                'algorithm_source':proof(ROOT/'scripts/audit_formal_runs.py')}
        phases.append(phase)
    plans=[]
    for phase in phases:
        for mode in ('nll','behavior'):
            if phase['family']=='B' and mode=='behavior':continue
            plans.append({'id':phase['name']+'_'+mode,'mode':mode,'phase':phase,
                          'checkpoints':['final','best'] if phase['family']=='C' and mode=='nll' else ['final'],
                          'prior_plans':[nll] if mode=='nll' else [behavior,initial],
                          'A_selection_path':str(ROOT/'results/supplement_A_validation/validation_lambda_selection.json') if phase['family']=='A' and mode=='behavior' else None})
    return {'schema_version':1,'preparation_revision':4,'prepared_only':True,'launch_authorized':False,'training_completion_claim':False,
            'root_frozen_scope':[scope,nllscope],'plans':plans,'producer_source':proof(__file__),
            'C_equivalence_policy':'410M/1B require independent per-model audit and all six final/best D/B4 comparisons before either plan freezes; only bitwise-equal endpoints alias; 2.8B has no D and no proof gate',
            'primary_final_model_equivalence_proof':proof(ROOT/'results/formal_model_equivalence_proofs.json'),
            'handoff_prior_sources':[
                {'queue_path':str(ROOT/'configs/primary_evaluation_v2_execution_matched_nll_queue.json'),'queue_sha256':sha(ROOT/'configs/primary_evaluation_v2_execution_matched_nll_queue.json'),'manifests':[nll]},
                {'queue_path':str(ROOT/'configs/primary_evaluation_v2_execution_matched_behavior_queue.json'),'queue_sha256':sha(ROOT/'configs/primary_evaluation_v2_execution_matched_behavior_queue.json'),'manifests':[behavior]},
                {'queue_path':str(ROOT/'configs/planned_model_behavioral_timing_v1_queue.json'),'queue_sha256':sha(ROOT/'configs/planned_model_behavioral_timing_v1_queue.json'),'manifests':[initial]}],
            'read_policy':'Training status/counters, frozen identity/provenance and validation-only chosen IDs; never heldout/task score outputs',
            'no_model_initialization_or_scheduler_start':True}


def completion_gate(rows,kinds,hashes):
    pending=[]
    for row in rows:
        cp=absolute(row['config_path']);expected=row.get('config_sha256')
        if not expected:raise ValueError('Missing frozen config proof')
        hashes.proof({'path':cp,'sha256':expected});cfg=read(cp);sp=absolute(row['summary_path'])
        if not sp.exists():pending.append({'id':row['id'],'reason':'missing_summary'});continue
        summary=read(sp);counts=summary.get('counters',{})
        ratio=Fraction(str(cfg['replay_ratio']));target=cfg['target_prediction_tokens']
        if summary.get('status')!='completed' or counts.get('target_prediction_tokens')!=target or counts.get('replay_prediction_tokens')!=target*ratio.numerator//ratio.denominator:
            pending.append({'id':row['id'],'reason':'incomplete_training_or_exposure'});continue
        for kind in kinds:
            try:path,_=endpoint(summary,kind)
            except (ValueError,KeyError):pending.append({'id':row['id'],'reason':'missing_or_invalid_'+kind+'_metadata'});continue
            if not Path(path).is_file() or Path(path).stat().st_size==0:pending.append({'id':row['id'],'reason':'missing_'+kind+'_checkpoint'})
    return {'ready':not pending,'expected_rows':len(rows),'pending_conditions':len({r['id'] for r in pending}),'pending_reasons':pending}


def selected_A_ids(selection,rows):
    candidates=selection.get('all_candidate_scores',[])
    if (selection.get('phase')!='supplement_A_lambda_sweep' or selection.get('complete') is not True or selection.get('completed_conditions')!=126 or selection.get('expected_conditions')!=126 or
            selection.get('selection_uses_test') is not False or selection.get('test_scores_accessed') is not False or
            len(selection.get('selections',[]))!=6 or len(candidates)!=42 or any(c.get('complete') is not True for c in candidates)):
        raise ValueError('Complete validation-only 126-cell/42-candidate/six-method A selection required')
    groups={r['group'] for r in rows};chosen=[];lookup={r['id']:r for r in rows}
    if {s['group'] for s in selection['selections']}!=groups:raise ValueError('A selected methods differ from frozen grid')
    for selected in selection['selections']:
        ids=selected['condition_ids']
        if (selected.get('selection_frozen') is not True or selected.get('shared_across_seeds') is not True or selected.get('test_scores_used') is not False or
                selected.get('selected_by_best_checkpoint') is not False or len(ids)!=3 or len(set(ids))!=3 or any(i not in lookup for i in ids)):
            raise ValueError('A choice requires one frozen shared lambda and three actual conditions')
        actual=[lookup[i] for i in ids]
        if {r['seed'] for r in actual}!={11,22,33} or any(r['group']!=selected['group'] or r['weight']!=selected['weight'] for r in actual):raise ValueError('A choice differs from actual method/lambda/seed identities')
        chosen+=ids
    if len(set(chosen))!=18:raise ValueError('Exactly eighteen distinct A behavior conditions required')
    return chosen


def model_equivalence_gate(phase,rows,hashes):
    """Wait for complete source audits, then bind each endpoint's state proof.

    A producer's successful completion means all pairs were examined. Equality
    is a separate per-pair fact; a different pair remains a distinct inference.
    This routine reads metadata and streams source bytes, never model tensors or
    held-out/task scores. The independent CPU producer performs the tensor hash.
    """
    gate=phase.get('model_equivalence_gate')
    if not gate:return {'ready':True,'required':False,'proofs':[],'source_proof_files':{}}
    completion_path=absolute(gate['completion_path'])
    if not completion_path.exists():return {'ready':False,'status':'waiting_C_model_state_audit','required_completion':str(completion_path)}
    completion=read(completion_path)
    if completion.get('status') in ('in_progress','waiting_training','running'):
        return {'ready':False,'status':'waiting_C_model_state_audit','producer_status':completion['status'],'required_completion':str(completion_path)}
    if completion.get('status')!='passed':raise ValueError('C model-state audit completion failed or lacks explicit passed status')
    sources={}
    def bind(p):
        path,value=hashes.proof(p);sources[path]=value;return path,value
    def same(p,expected,label):
        if str(absolute(p['path']))!=str(absolute(expected['path'])) or p['sha256']!=expected['sha256']:
            raise ValueError('C equivalence source identity mismatch: '+label)
        return bind(p)
    if (completion.get('schema_version')!=1 or completion.get('phase')!=gate['phase'] or
            completion.get('model')!=phase['model_tag'] or type(completion.get('all_pairs_bitwise_equal')) is not bool or
            completion.get('expected_runs')!=phase['expected_rows'] or completion.get('completed_runs')!=phase['expected_rows'] or
            completion.get('comparisons_computed')!=6 or completion.get('test_and_behavior_scores_read') is not False or
            completion.get('gpu_work_launched') is not False):
        raise ValueError('C equivalence completion phase/model/schema mismatch')
    completion_proof=proof(completion_path);bind(completion_proof)
    same(completion['source_registry'],phase['registry'],'registry')
    same(completion['input_snapshot'],gate['source_snapshot'],'snapshot')
    for source in completion['source_inputs']:bind(source)
    audit_path,audit_sha=bind(completion['audit'])
    if audit_path!=str(absolute(gate['audit_path'])):raise ValueError('C equivalence audit path mismatch')
    audit=read(audit_path)
    if (audit.get('status')!='passed' or audit.get('phase')!=gate['phase'] or audit.get('model')!=phase['model_tag'] or
            audit.get('expected_runs')!=phase['expected_rows'] or audit.get('completed_runs')!=phase['expected_rows'] or
            str(absolute(audit['snapshot']))!=str(absolute(gate['source_snapshot']['path'])) or
            audit.get('snapshot_sha256')!=gate['source_snapshot']['sha256'] or audit.get('source_violations')!=[] or
            audit.get('test_and_behavior_scores_read') is not False):
        raise ValueError('C independent audit is incomplete or bound to a different source')
    same(audit['source_registry'],phase['registry'],'audit registry')
    state_gate=audit['source_state_complete_gate']
    if (state_gate.get('ready') is not True or state_gate.get('expected_runs')!=phase['expected_rows'] or
            state_gate.get('completed_runs')!=phase['expected_rows'] or state_gate.get('missing_runs')!=[] or state_gate.get('failed_runs')!=[]):
        raise ValueError('C independent audit lacks a complete successful scheduler cohort gate')
    algorithm='sha256_sorted_key_dtype_shape_contiguous_uint8_v1'
    bind(gate['algorithm_source']);lookup={r['id']:r for r in rows};all_pairs=[];proofs=[]
    if len(lookup)!=len(rows):raise ValueError('Duplicate C registry run ID')
    for kind in ('final','best'):
        record_path,record_sha=bind(completion['proofs'][kind])
        if record_path!=str(absolute(gate['proof_paths'][kind])):raise ValueError('C endpoint proof path mismatch')
        record=read(record_path)
        if (record.get('schema_version')!=1 or record.get('status')!='passed' or record.get('algorithm')!=algorithm or
                record.get('phase')!=gate['phase'] or record.get('model')!=phase['model_tag'] or record.get('endpoint')!=kind or
                record.get('test_scores_accessed') is not False or record.get('checkpoint_files_modified') is not False or
                len(record.get('comparisons',[]))!=3):raise ValueError('C proof requires three complete-state comparisons per endpoint')
        same(record['algorithm_source'],gate['algorithm_source'],'fingerprint algorithm')
        same(record['source_registry'],phase['registry'],'proof registry')
        same(record['source_snapshot'],gate['source_snapshot'],'proof snapshot')
        same(record['source_audit'],{'path':audit_path,'sha256':audit_sha},'proof audit')
        if record.get('producer_source'):bind(record['producer_source'])
        seen=set()
        for comparison in record['comparisons']:
            seed=comparison.get('seed');equal=comparison.get('bitwise_model_state_equal')
            if (comparison.get('kind')!='identical_bank_null_control' or comparison.get('endpoint')!=kind or
                    type(seed) is not int or seed not in (11,22,33) or seed in seen or type(equal) is not bool):
                raise ValueError('C comparison endpoint/seed/kind is missing, repeated or invalid')
            seen.add(seed);expected={r['id'] for r in rows if r.get('seed')==seed and r.get('group') in ('B4','D')}
            sides=[comparison['left'],comparison['right']]
            if len(expected)!=2 or {s['run_id'] for s in sides}!=expected:raise ValueError('C comparison does not cover the actual D/B4 pair')
            fingerprints=[]
            for side in sides:
                row=lookup[side['run_id']];cfg=read(row['config_path']);summary=read(row['summary_path']);cppath,exposure=endpoint(summary,kind)
                selection=summary['best_target_validation'] if kind=='best' else {'rule':'fixed_completed_target_exposure',**exposure}
                if (side.get('kind')!=kind or side.get('model_revision')!=phase['model']['revision'] or
                        str(absolute(side['model_name_or_path']))!=str(absolute(phase['model']['path'])) or
                        cfg.get('model_revision')!=phase['model']['revision'] or str(absolute(cfg['model_name_or_path']))!=str(absolute(phase['model']['path'])) or
                        side.get('checkpoint_exposure')!=exposure or side.get('actual_selection')!=selection or str(absolute(side['checkpoint']))!=cppath):
                    raise ValueError('C endpoint proof differs from actual checkpoint/model/exposure')
                same({'path':side['config_path'],'sha256':side['config_sha256']},{'path':row['config_path'],'sha256':row['config_sha256']},'endpoint config')
                same({'path':side['summary_path'],'sha256':side['summary_sha256']},{'path':row['summary_path'],'sha256':hashes.digest(row['summary_path'])},'endpoint summary')
                bind({'path':side['checkpoint'],'sha256':side['checkpoint_sha256']})
                fingerprint=side['model_state_fingerprint']
                if (not re.fullmatch(r'[0-9a-f]{64}',fingerprint.get('sha256_model_state','')) or
                        type(fingerprint.get('state_elements')) is not int or fingerprint['state_elements']<1):
                    raise ValueError('C proof lacks a complete model-state fingerprint')
                fingerprints.append(fingerprint)
            if equal!=(fingerprints[0]==fingerprints[1]):raise ValueError('C pair equality flag contradicts its model-state fingerprints')
            all_pairs.append({'endpoint':kind,'seed':seed,'bitwise_model_state_equal':equal,
                'left_run_id':sides[0]['run_id'],'right_run_id':sides[1]['run_id'],
                'left_checkpoint_sha256':sides[0]['checkpoint_sha256'],'right_checkpoint_sha256':sides[1]['checkpoint_sha256']})
        if seen!={11,22,33}:raise ValueError('C proof must cover all three registered seeds')
        proofs.append({'path':record_path,'sha256':record_sha})
    all_equal=all(p['bitwise_model_state_equal'] for p in all_pairs)
    if completion['all_pairs_bitwise_equal']!=all_equal:raise ValueError('C completion equality flag contradicts the six explicit comparisons')
    return {'ready':True,'required':True,'completion':completion_proof,'audit':completion['audit'],'proofs':proofs,'source_proof_files':sources,
            'comparisons':all_pairs,'all_pairs_bitwise_equal':all_equal,
            'equal_pairs_by_endpoint':{kind:sum(p['bitwise_model_state_equal'] for p in all_pairs if p['endpoint']==kind) for kind in ('final','best')},
            'unequal_pairs':[p for p in all_pairs if not p['bitwise_model_state_equal']],
            'alias_rule':'Only individually bitwise-equal, actual-checkpoint-SHA-bound endpoint pairs may share inference; all other endpoints remain distinct'}


def prepare_one(cat,plan,folder,hashes):
    phase=copy.deepcopy(plan['phase']);mode=plan['mode'];registry_path,_=hashes.proof(phase['registry']);rows=read(registry_path)
    if len(rows)!=phase['expected_rows']:raise ValueError('Unexpected source registry size')
    gate=completion_gate(rows,plan['checkpoints'],hashes)
    if not gate['ready']:return {'status':'waiting_training',**gate}
    equivalence=model_equivalence_gate(phase,rows,hashes)
    if not equivalence['ready']:return equivalence
    phase.pop('model_equivalence_gate',None)
    phase['source_proof_files'].update(equivalence['source_proof_files'])
    selection_path=Path(plan['A_selection_path']) if plan['A_selection_path'] else None
    if selection_path and not selection_path.exists():return {'status':'waiting_complete_A_validation_selection','ready':False}
    # Original protocol/activation bytes remain authoritative; the adapter only
    # supplies model/heldout identity fields absent from older source formats.
    hashes.proof(phase['protocol'])
    for path,value in phase['source_proof_files'].items():hashes.proof({'path':path,'sha256':value})
    source_registry=copy.deepcopy(phase['registry']);source_protocol=copy.deepcopy(phase['protocol'])
    folder.mkdir(parents=True,exist_ok=True)
    if phase['family'] in ('A','B'):
        heldout={k:str(absolute(v['path'])) for k,v in phase['test_data'].items()}
        for row in rows:
            if row.get('test_paths') and {k:str(absolute(v)) for k,v in row['test_paths'].items()}!=heldout:raise ValueError('Existing heldout identity cannot be overwritten by adapter')
            row['test_paths']=heldout
        adapter_registry=folder/'evaluation_registry.json';write(adapter_registry,rows)
        original=read(source_protocol['path'])
        for key,value in phase['protocol_expect'].items():
            if original.get(key)!=value:raise ValueError('Frozen original protocol identity changed')
        adapter={'adapter_role':'evaluation_identity_only','model_revision':phase['model']['revision'],'source_protocol':source_protocol,
                 'source_registry':source_registry,'family':phase['family'],'training_data_unchanged':True,
                 'validation_paths':{k:str(absolute(v['path'])) for k,v in phase['validation_data'].items()},'test_paths':heldout}
        pp=folder/'evaluation_protocol.json';write(pp,adapter)
        phase['registry']=proof(adapter_registry);phase['protocol']=proof(pp)
        phase['protocol_expect']={'adapter_role':'evaluation_identity_only','family':phase['family'],'training_data_unchanged':True,'model_revision':phase['model']['revision']}
        phase['source_proof_files'].update({str(absolute(source_registry['path'])):source_registry['sha256'],str(absolute(source_protocol['path'])):source_protocol['sha256']})
    if selection_path:
        selection=read(selection_path);ids=selected_A_ids(selection,rows)
        bound={str(absolute(p)):v for p,v in selection['sources_sha256'].items()}
        if bound.get(str(absolute(source_registry['path'])))!=source_registry['sha256']:raise ValueError('A choice bound a different runtime registry')
        for row in rows:
            sp=str(absolute(row['summary_path']))
            if bound.get(sp)!=hashes.digest(sp):raise ValueError('A selected grid summary changed after validation choice')
        normalized={'complete':True,'selection_uses_test':False,'source_selection':proof(selection_path),'source_registry':source_registry,'selected_run_ids':ids}
        normalized_path=folder/'validation_selection_identity.json';write(normalized_path,normalized)
        phase['selection']={'record':proof(normalized_path),'record_expect':{'complete':True,'selection_uses_test':False},'run_ids_field':'selected_run_ids','include_run_ids':ids}
        phase['source_proof_files'][str(absolute(selection_path))]=sha(selection_path)
    phase['science_frozen']=True;phase['checkpoints']=plan['checkpoints']
    scope=read(cat['root_frozen_scope'][0]['path']);taskroot=ROOT/'artifacts/behavioral'
    phase['behavior']={'batch_size':scope['behavior_batch_size_by_model'][phase['model_tag']],'max_length':scope['max_length']}
    spec={'schema_version':2,'science_frozen':True,'phases':[phase],'checkpoints':plan['checkpoints'],'prior_plans':plan['prior_plans'],
          'model_equivalence_proofs':[cat['primary_final_model_equivalence_proof'],*equivalence['proofs']],
          'behavior':{'data_root':str(taskroot),**phase['behavior'],'data_manifest_sha256':sha(taskroot/'data_manifest.json'),'harness_manifest_sha256':sha(taskroot/'sources/harness_manifest.json')},
          'inference_launch_authorized_by_preparation':False}
    phase['source_proof_files'].update({p['path']:p['sha256'] for p in cat['root_frozen_scope']})
    phase['source_proof_files'][str(Path(__file__).resolve())]=sha(__file__)
    spec_path=folder/'phase_spec.json';write(spec_path,spec);manifest_path=folder/'manifest.json';queue_path=folder/'queue.json'
    queue,manifest=build(spec_path,ROOT/'results/future_evaluation_v1'/mode,manifest_path,mode,600 if mode=='behavior' else 180,hashes=hashes)
    write(manifest_path,manifest);write(queue_path,queue)
    evidence={'status':'ready','ready':True,'id':plan['id'],'mode':mode,'phase_spec':proof(spec_path),'queue':proof(queue_path),'manifest':proof(manifest_path),
              'workload':manifest['workload'],'logical_endpoints':len(manifest['records']),
              'source_model_equivalence':{k:v for k,v in equivalence.items() if k not in ('source_proof_files','ready')},
              'model_inference_started':False,'scheduler_started':False}
    write(folder/'READY.json',evidence)
    return evidence


def closure_metadata(draft,hashes):
    """Verify queue authorities/DAG without rehashing all checkpoints every poll."""
    from handoff_evaluation import job_definition
    from schedule_jobs import validate_queue
    authorities={};sources=[];jobs={};bindings={};wanted=[];input_hashes={}
    for role in ('evaluation_sources','prior_sources'):
        for source in draft[role]:
            hashes.proof({'path':source['queue_path'],'sha256':source['queue_sha256']})
            queue=read(source['queue_path'])
            for p in source['manifests']:
                path,value=hashes.proof(p);authorities[(path,value)]=read(path)
            sources.append((role,queue))
    for role,queue in sources:
        for job in queue:
            if job['id'] in jobs and jobs[job['id']]!=job:raise ValueError('Prior job definition changed')
            bindings[job['id']]=job_definition(job,authorities,input_hashes);jobs[job['id']]=job
            if role=='evaluation_sources':wanted.append(job['id'])
    selected=[];seen=set();visiting=set()
    def include(name):
        if name in seen:return
        if name in visiting or name not in jobs:raise ValueError('Cyclic or missing explicit prior dependency')
        visiting.add(name)
        for parent in jobs[name].get('depends_on',[]):include(parent)
        visiting.remove(name);seen.add(name);selected.append(jobs[name])
    for name in wanted:include(name)
    validate_queue(selected)
    def ancestors(name):
        direct=set(jobs[name].get('depends_on',[]));return direct|set().union(*(ancestors(p) for p in direct)) if direct else set()
    for job in selected:
        entry=bindings[job['id']]['entry'];group=bindings[job['id']]['manifest']['baseline_groups'][entry['baseline_group']]
        if entry['kind']!='initial' and (group['initial_job_id'] not in ancestors(job['id']) or bindings[group['initial_job_id']]['entry']['kind']!='initial'):
            raise ValueError('Original initial baseline is absent from complete DAG ancestry')
    return selected


def pass_once(cat,catalog_path,catalog_sha,root,hashes,max_freezes=1):
    hashes.proof({'path':catalog_path,'sha256':catalog_sha});hashes.proof(cat['producer_source'])
    for item in cat['root_frozen_scope']:hashes.proof(item)
    phases=[];new_count=0
    for plan in cat['plans']:
        folder=root/plan['id'];marker=folder/'READY.json'
        try:
            if marker.exists():
                value=read(marker)
                for key in ('queue','manifest','phase_spec'):hashes.proof(value[key])
            elif new_count>=max_freezes:value={'status':'deferred_CPU_freeze_limit','ready':False}
            else:
                value=prepare_one(cat,plan,folder,hashes)
                if value.get('ready'):new_count+=1
        except (ValueError,KeyError,FileNotFoundError) as error:
            value={'status':'invalid_source_or_metadata','ready':False,'error':str(error)}
        phases.append({'id':plan['id'],'mode':plan['mode'],'family':plan['phase']['family'],'model_tag':plan['phase']['model_tag'],**value})
    ready=[p for p in phases if p.get('ready')]
    sources=[{'queue_path':p['queue']['path'],'queue_sha256':p['queue']['sha256'],'manifests':[p['manifest']]} for p in ready]
    draft={'schema_version':1,'example_only':True,'root_finalization_and_launch_authorization_required':True,
           'old_master_queue':str(ROOT/'configs/supplemental_v1_queue.json'),'old_master_state':str(ROOT/'logs/supplemental_job_state.json'),
           'hard_deadline_utc':'2026-11-07T03:26:32Z','tail_seconds':1800,'gpus':'0,1,2,3',
           'evaluation_sources':[ *cat['handoff_prior_sources'][:2],*sources ],'prior_sources':cat['handoff_prior_sources'],
           'pending_plan_ids':[p['id'] for p in phases if not p.get('ready')],
           'scope_rule':'All science and endpoint scopes remain frozen; preparation never launches or analyzes scores'}
    merged=closure_metadata(draft,hashes)
    write(root/'handoff_draft.json',draft)
    index={'schema_version':1,'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'catalog':{'path':str(catalog_path),'sha256':catalog_sha},
           'phases':phases,'ready_phase_count':len(ready),'all_requested_phases_ready':len(ready)==len(phases),
           'handoff_draft':proof(root/'handoff_draft.json'),'handoff_complete_dependency_closure_validated':True,
           'handoff_checkpoint_input_rehash_deferred_to_legal_handoff_and_worker':True,
           'handoff_dependency_closure_jobs':len(merged),'no_scheduler_started':True,'no_model_inference':True,'test_or_behavior_scores_read':False}
    write(root/'ready_index.json',index)
    return index


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--initialize-catalog',type=Path);p.add_argument('--catalog',type=Path);p.add_argument('--catalog-sha256');p.add_argument('--output-root',type=Path,default=ROOT/'artifacts/evaluation_ready_v1');p.add_argument('--watch',action='store_true');p.add_argument('--poll-seconds',type=float,default=30);p.add_argument('--max-freezes-per-pass',type=int,default=1);a=p.parse_args()
    if a.initialize_catalog:
        if a.initialize_catalog.exists():raise FileExistsError('Use a new catalog snapshot')
        c=catalog();write(a.initialize_catalog,c);print(json.dumps({'catalog':str(a.initialize_catalog),'sha256':sha(a.initialize_catalog),'plans':len(c['plans']),'training_completion_claim':False}));return
    if not a.catalog or not a.catalog_sha256:p.error('--catalog and --catalog-sha256 required')
    if a.poll_seconds<5 or a.max_freezes_per_pass<1:p.error('At least five seconds polling and one allowed CPU freeze required')
    a.output_root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.output_root/'watcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    h=Hashes();h.proof({'path':a.catalog,'sha256':a.catalog_sha256});c=read(a.catalog)
    while True:
        result=pass_once(c,a.catalog,a.catalog_sha256,a.output_root,h,a.max_freezes_per_pass)
        print(json.dumps({'ready':result['ready_phase_count'],'total':len(result['phases']),'states':{x['id']:x['status'] for x in result['phases']},'no_inference':True}),flush=True)
        if not a.watch or result['all_requested_phases_ready'] or dt.datetime.now(dt.timezone.utc)>=dt.datetime(2026,11,7,3,26,32,tzinfo=dt.timezone.utc):break
        time.sleep(min(a.poll_seconds,60))


if __name__=='__main__':main()
