#!/usr/bin/env python3
"""Prepare/check supplemental analysis; parsing requires explicit root authority."""
import argparse
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from replay_study import supplemental_analysis_guard_v2 as guard


def initialize_protocol(catalog_proof,preparation_execution=None):
    v=guard.Verifier();catalog=v.proof(catalog_proof)
    preparation_sources=[catalog['producer_source'],*catalog.get('operational_revision',{}).get('source_proofs',[])]
    preparation_ready_root=None
    if catalog.get('preparation_revision',0)>=5:
        if preparation_execution is None:raise ValueError('Explicit frozen v5 preparation execution receipt required')
        receipt=v.proof(preparation_execution)
        if (receipt['catalog']!=catalog_proof or receipt.get('preparation_revision')!=5 or receipt.get('score_files_parsed') is not False or
                receipt.get('gpu_inference_started') is not False):raise ValueError('Preparation execution receipt binds a different catalog or scope')
        preparation_sources.append(preparation_execution);preparation_ready_root=receipt['active_output_root']
    for p in preparation_sources:v.bind(p['path'],p['sha256'])
    files=['scripts/analyze_supplemental_phase_v2.py','src/replay_study/supplemental_analysis_guard_v2.py','src/replay_study/supplemental_analysis_guard.py','src/replay_study/selected_policy_analysis.py','scripts/stat_bound_digest_cache.py',
        'scripts/analyze_nll_v2.py','src/replay_study/nll_v2_analysis.py','src/replay_study/analysis.py',
        'scripts/analyze_behavioral_plan.py','src/replay_study/behavioral_analysis.py']
    source={str(ROOT/p):guard.sha(ROOT/p) for p in files};policy_sources=[guard.proof(ROOT/'results/nll_scope_budget_v1.json')]
    nll_budget=v.proof(policy_sources[0])
    policies={(g['model'],g['data_family']):g['inference_policy'] for g in nll_budget['groups']}
    behaviors={};frozen={}
    for file in ('docs/primary_evaluation_v2_execution_matched_behavior_manifest.json','docs/planned_model_behavioral_timing_v1_manifest.json'):
        p=guard.proof(ROOT/file);policy_sources.append(p);m=v.proof(p);frozen.update(m['scorer_input_sha256'])
        for group in m['baseline_groups'].values():behaviors[group['model_tag']]=group['inference_policy']
    p=guard.proof(ROOT/'docs/primary_evaluation_v2_execution_matched_nll_manifest.json');policy_sources.append(p)
    frozen.update(v.proof(p)['scorer_input_sha256'])
    v1_snapshot=guard.proof(ROOT/'artifacts/provenance/supplemental_aux_analysis_v1/source_bytes_manifest.json');policy_sources.append(v1_snapshot);v.proof(v1_snapshot)
    for path,value in frozen.items():v.bind(path,value)
    source.update(frozen);contracts={}
    for plan in catalog['plans']:
        phase=plan['phase'];rows=18 if phase['family']=='A' and plan['mode']=='behavior' else phase['expected_rows']
        contracts[plan['id']]={'family':phase['family'],'mode':plan['mode'],
            'phase':{'name':phase['name'],'role':phase['role'],'model_tag':phase['model_tag'],'data_family':phase['data_family'],
                     'expected_phase':phase['expected_phase'],'rows':rows},'checkpoints':plan['checkpoints'],'logical_endpoints':rows*len(plan['checkpoints']),
            'inference_policy_expect':behaviors[phase['model_tag']] if plan['mode']=='behavior' else policies[phase['model_tag'],phase['data_family']],
            'source_registry':phase['registry'],
            'scientific_identity':{k:phase[k] for k in ('model','model_tag','training_data','validation_data','test_data','data_family')},
            'source_proof_files':{**phase['source_proof_files'],phase['registry']['path']:phase['registry']['sha256'],phase['protocol']['path']:phase['protocol']['sha256']}}
    a=next(p['phase'] for p in catalog['plans'] if p['phase']['family']=='A')
    return {'schema_version':2,'operational_revision':2,'v1_protocol_retained':guard.proof(ROOT/'docs/supplemental_aux_analysis_protocol_v1.json'),
        'v1_full_byte_snapshot':v1_snapshot,'v1_real_scores_read':False,'operational_change_only':'Checkpoint SHA I/O reuse; selection/statistics/bootstrap/forward scope unchanged',
        'checkpoint_cache_reader_source':guard.proof(ROOT/'scripts/stat_bound_digest_cache.py'),
        'checkpoint_cache_rule':'Optional explicit per-plan SHA-bound v5 snapshot. Only explicit checkpoint/proof-side paths with expected SHA, complete stable-fd receipt and current five-stat may hit. Missing entries fresh fallback. All other files remain fresh. Root authority binds the exact mode/snapshot.',
        'prepared_only':True,'root_future_score_authorization_required':True,'real_supplemental_or_behavior_scores_seen':False,
        'catalog':catalog_proof,'contracts':contracts,'analysis_source_sha256':source,'policy_sources':policy_sources,
        'preparation_sources':preparation_sources,'preparation_execution_receipt':preparation_execution,'preparation_ready_root':preparation_ready_root,
        'first_read_root':str(ROOT/'artifacts/provenance/supplemental_score_first_reads_v1'),
        'prior_sources':[{'queue':{'path':s['queue_path'],'sha256':s['queue_sha256']},'manifests':s['manifests']} for s in catalog['handoff_prior_sources']],
        'A_registry':a['registry'],'A_selection':{'path':str(ROOT/'results/supplement_A_validation/validation_lambda_selection.json'),'sha256':None},
        'A_selection_SHA_rule':'Resolve complete validation-only selection SHA at prepare time; freeze it in the immutable plan; root authority must bind that exact plan',
        'bootstrap_replicates':5000,'bootstrap_seed':20261106,'confidence':.95,
        'policy_pairs':[['M','B4'],['C','B4'],['B2','B1'],['B4','B3'],['B3','B1'],['B4','B2']],
        'policy_interpretation':'Each method plus its own validation-selected lambda; not a fixed-lambda objective effect',
        'print_figure_policy':{'width_inches':7,'height_inches':6.6,'all_individual_training_seed_points':True,'mean_and_sample_SD_ddof':1,'formats':['CSV','PNG','PDF','SVG'],'conditional_bootstrap':'separate tables, not across-training-seed CI'},
        'uncertainty':'Three matched training seeds: individual values and sample SD. Document/question bootstrap is conditional on checkpoints under independent/exchangeable sampling; same-source activity/context correlation may invalidate population CI interpretation.',
        'whole_phase_gate':'All jobs/reuses/aliases and imported canonical baselines complete with frozen definition SHA, rc0 and source/output byte integrity before any score parse; C final and best jointly gated',
        'selection_rule_unchanged':True,'new_GPU_inference':False,'allow_partial_for_final_conclusions':False}


def write_auxiliary(bundle,path):
    from replay_study.analysis import write_csv
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    def flatten(row,prefix=''):
        result={}
        for key,value in row.items():
            name=prefix+key
            if isinstance(value,dict):result.update(flatten(value,name+'_'))
            else:result[name]=json.dumps(value,ensure_ascii=False) if isinstance(value,list) else value
        return result
    for name,values in bundle.items():
        if name=='metadata':continue
        if name.endswith('bootstrap'):(path/(name+'.json')).write_bytes(guard.encoded(values))
        write_csv(path/(name+'.csv'),[flatten(v) for v in values])
    guard.immutable(path/'analysis_manifest.json',bundle['metadata'])


def command_templates(protocol_proof,ready_root,state_path,plan_root,output_root):
    """Runnable argv templates, with deferred SHA placeholders, never launch."""
    v=guard.Verifier();protocol=v.proof(protocol_proof)
    if protocol.get('preparation_ready_root') and Path(ready_root).resolve()!=Path(protocol['preparation_ready_root']).resolve():raise ValueError('Templates must use the frozen active preparation READY root')
    for path,value in protocol['analysis_source_sha256'].items():v.bind(path,value)
    commands=[];script=str(ROOT/'scripts/analyze_supplemental_phase_v2.py')
    for name,c in protocol['contracts'].items():
        ready=Path(ready_root)/name/'READY.json';plan=Path(plan_root)/(name+'.json')
        authority=plan.with_name(plan.stem+'_ROOT_AUTHORITY.json')
        commands.append({'contract_id':name,'family':c['family'],'mode':c['mode'],'phase':c['phase'],
            'optional_prepare_cache_args':['--digest-cache-snapshot','<IMMUTABLE_V5_SNAPSHOT>','--digest-cache-snapshot-sha256','<SNAPSHOT_SHA256>'],
            'cache_mode_rule':'Default fresh. Choose snapshot only after real phase READY; prepare and root authority bind it. No automatic latest-snapshot lookup.',
            'logical_endpoints':c['logical_endpoints'],'jointly_gated_endpoints':c['checkpoints'],
            'ready_path':str(ready.resolve()),'READY_currently_exists':ready.is_file(),'root_authority_template_is_disabled':True,
            'prepare_argv':[sys.executable,script,'prepare','--ready',str(ready.resolve()),'--ready-sha256','<READY_SHA256>',
                '--protocol',protocol_proof['path'],'--protocol-sha256',protocol_proof['sha256'],'--contract-id',name,'--output',str(plan.resolve())],
            'check_argv':[sys.executable,script,'check','--plan',str(plan.resolve()),'--plan-sha256','<PLAN_SHA256>',
                '--state',str(Path(state_path).resolve()),'--output',str((Path(output_root)/(name+'_readiness.json')).resolve())],
            'authorized_analyze_argv':[sys.executable,script,'analyze','--plan',str(plan.resolve()),'--plan-sha256','<PLAN_SHA256>',
                '--authority',str(authority.resolve()),'--authority-sha256','<ROOT_AUTHORITY_SHA256>','--state',str(Path(state_path).resolve()),
                '--output',str((Path(output_root)/name).resolve())],
            'authorization_status':'No authority issued. Root must review complete plan and bind its exact SHA before analysis.',
            'A_selected_view':'Reuse18 exact validation-selected IDs from full126 NLL; no new inference' if c['family']=='A' else None})
    v.finish()
    return {'schema_version':2,'operational_revision':2,'prepared_at_utc':guard.now(),'templates_only':True,'no_inference_or_score_parsing':True,'protocol':protocol_proof,
        'deferred_proofs':'Replace SHA placeholders only with real immutable READY/plan/root-authority bytes; never synthesize a READY for unfinished training',
        'readiness_state_source':'Use the scheduler state containing the complete explicit prior/new dependency closure, including completed baselines and reuse materializations',
        'commands':commands,'extra_GPU_calls_for_selected_A_NLL':0,'allow_partial_for_final_conclusions':False}


def analyze(plan_proof,authority_proof,state_path,output,first_receipt=None,no_plots=False):
    plan,_=guard.authorize(plan_proof,authority_proof);ready=guard.inspect(plan_proof,state_path)
    if not ready['ready']:raise PermissionError('Entire frozen phase closure is incomplete; all score parsing remains sealed')
    journal=Path(plan['first_read_receipt_path'])
    if first_receipt is not None and Path(first_receipt).resolve()!=journal:raise ValueError('First-read receipt path is immutable per phase manifest')
    if journal.exists():raise FileExistsError('Phase first-read receipt already exists; it cannot be reset by choosing another analysis directory')
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new immutable analysis output directory')
    output.mkdir(parents=True);readiness=guard.immutable(output/'whole_phase_readiness.json',ready)
    manifest=guard.read(plan['manifest']['path']);initial=next(iter(manifest['baseline_groups'].values()))['initial_output']
    first=guard.first_read(journal,plan_proof,authority_proof,readiness,initial,plan['checkpoint_verification'])
    contract=plan['contract'];bootstrap=5000;seed=20261106;results=[]
    if contract['mode']=='nll':
        from replay_study import nll_v2_analysis as nll
        from replay_study.selected_policy_analysis import nll_selected,plot_lambda_curves
        phase=contract['phase']
        for kind in contract['checkpoints']:
            prepared=nll.prepare_plan(plan['manifest']['path'],plan['manifest']['sha256'],role=phase['role'],phase_name=phase['name'],
                model_tag=phase['model_tag'],data_family=phase['data_family'],checkpoint=kind)
            bundle=nll.collect_scores(prepared,read_heldout_scores=True,allow_partial=False,n_bootstrap=bootstrap)
            bundle['whole_phase_first_read_receipt']=first;bundle['whole_phase_readiness']=readiness
            nll.write_bundle(bundle,output/kind)
            if contract['family']=='A':
                by_id={r['run_id']:r for r in manifest['records']}
                for row in bundle['endpoints']:
                    record=by_id[row['run_id']];row.update(checkpoint_sha256=record['checkpoint_sha256'],step=record['step'],full_training_committed_path=record['full_training_committed_path'])
                cache={}
                def load(row):
                    record=by_id[row['run_id']]
                    if record['output'] not in cache:
                        p=Path(record['output']);att=guard.read(guard.attestation_path(p))
                        if guard.sha(p)!=att['output_sha256'][str(p)]:raise ValueError('Selected NLL output changed after gate')
                        cache[record['output']]=guard.read(p)['corpora']
                    return cache[record['output']]
                aux=nll_selected(bundle,plan['selection'],load,bootstrap,seed)
                aux['metadata'].update(first_read_receipt=first,whole_phase_readiness=readiness,plan=plan_proof,analysis_source_sha256=guard.read(plan['protocol']['path'])['analysis_source_sha256'])
                write_auxiliary(aux,output/'validation_selected_policy')
                if not no_plots:plot_lambda_curves(bundle['endpoint_statistics'],output/'full_lambda_curves',bundle['endpoints'])
            results.append({'endpoint':kind,'logical_conditions':bundle['completed_conditions']})
    else:
        from analyze_behavioral_plan import collect
        from replay_study.behavioral_analysis import write_bundle
        from replay_study.selected_policy_analysis import behavior_selected
        bundle=collect(plan['manifest']['path']);bundle['metadata'].update(first_read_receipt=first,whole_phase_readiness=readiness,plan=plan_proof)
        write_bundle(bundle,output/'behavior')
        if contract['family']=='A':
            by_id={r['run_id']:r for r in manifest['records']};cache={}
            for row in bundle['endpoints']:row['full_training_committed_path']=by_id[row['run_id']]['full_training_committed_path']
            def load(row):
                from replay_study.behavioral_analysis import jsonl
                record=by_id[row['run_id']];key=(record['output'],row['task'])
                if key not in cache:
                    p=Path(record['output']);att=guard.read(guard.attestation_path(p))
                    if guard.sha(p)!=att['output_sha256'][str(p)]:raise ValueError('Selected behavioral output changed after gate')
                    metrics=guard.read(p)['tasks'][row['task']];pred=metrics['predictions_path']
                    if guard.sha(pred)!=att['output_sha256'][pred] or att['output_sha256'][pred]!=metrics['predictions_sha256']:raise ValueError('Selected predictions changed after gate')
                    cache[key]=jsonl(pred)
                return cache[key]
            aux=behavior_selected(bundle,plan['selection'],load,bootstrap,seed)
            aux['metadata'].update(first_read_receipt=first,whole_phase_readiness=readiness,plan=plan_proof,analysis_source_sha256=guard.read(plan['protocol']['path'])['analysis_source_sha256'])
            write_auxiliary(aux,output/'validation_selected_policy')
        results.append({'endpoint':'final_behavior','logical_conditions':len(manifest['records'])})
    guard.immutable(output/'COMPLETE.json',{'complete':True,'plan':plan_proof,'authority':authority_proof,'readiness':readiness,
        'checkpoint_verification':plan['checkpoint_verification'],'verification_work':ready['verification_work'],
        'first_read_receipt':first,'parsed_phase':contract['phase'],'endpoints':results,'GPU_inference_started':False,'selection_recomputed':False,'frozen_primary_statistics_modified':False})
    return {'complete':True,'contract_id':plan['contract_id'],'output':str(output),'GPU_inference_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    init=sub.add_parser('initialize-protocol');init.add_argument('--catalog',type=Path,required=True);init.add_argument('--catalog-sha256',required=True);init.add_argument('--preparation-execution',type=Path);init.add_argument('--preparation-execution-sha256');init.add_argument('--output',type=Path,required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--ready',type=Path,required=True);prep.add_argument('--ready-sha256',required=True);prep.add_argument('--protocol',type=Path,required=True);prep.add_argument('--protocol-sha256',required=True);prep.add_argument('--contract-id',required=True);prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--digest-cache-snapshot',type=Path);prep.add_argument('--digest-cache-snapshot-sha256')
    commands=sub.add_parser('prepare-commands');commands.add_argument('--protocol',type=Path,required=True);commands.add_argument('--protocol-sha256',required=True);commands.add_argument('--ready-root',type=Path,required=True);commands.add_argument('--state',type=Path,required=True);commands.add_argument('--plan-root',type=Path,required=True);commands.add_argument('--analysis-root',type=Path,required=True);commands.add_argument('--output',type=Path,required=True)
    for action in ('check','analyze'):
        child=sub.add_parser(action);child.add_argument('--plan',type=Path,required=True);child.add_argument('--plan-sha256',required=True);child.add_argument('--state',type=Path,required=True);child.add_argument('--output',type=Path,required=True)
        if action=='analyze':child.add_argument('--authority',type=Path,required=True);child.add_argument('--authority-sha256',required=True);child.add_argument('--first-read-receipt',type=Path);child.add_argument('--no-plots',action='store_true')
    a=p.parse_args()
    if a.action=='initialize-protocol':
        execution={'path':str(a.preparation_execution.resolve()),'sha256':a.preparation_execution_sha256} if a.preparation_execution else None
        result=guard.immutable(a.output,initialize_protocol({'path':str(a.catalog.resolve()),'sha256':a.catalog_sha256},execution))
    elif a.action=='prepare-commands':result=guard.immutable(a.output,command_templates({'path':str(a.protocol.resolve()),'sha256':a.protocol_sha256},a.ready_root,a.state,a.plan_root,a.analysis_root))
    elif a.action=='prepare':
        if bool(a.digest_cache_snapshot)!=bool(a.digest_cache_snapshot_sha256):p.error('Cache snapshot path and SHA must be supplied together')
        cache={'path':str(a.digest_cache_snapshot.resolve()),'sha256':a.digest_cache_snapshot_sha256} if a.digest_cache_snapshot else None
        value=guard.prepare({'path':str(a.ready.resolve()),'sha256':a.ready_sha256},{'path':str(a.protocol.resolve()),'sha256':a.protocol_sha256},a.contract_id,cache);result=guard.immutable(a.output,value)
        template={'authorized_by':'root','allow_score_analysis':False,'plan':result,'protocol':value['protocol'],'ready':value['ready'],'manifest':value['manifest'],
            'checkpoint_verification':value['checkpoint_verification'],
            'contract_id':value['contract_id'],'expected_counts':value['counts'],'allow_partial':False,'training_methods_grid_selection_horizons_frozen':True,'test_or_behavior_used_for_selection':False}
        guard.immutable(a.output.with_name(a.output.stem+'_root_authority_TEMPLATE.json'),template)
    elif a.action=='check':result=guard.immutable(a.output,guard.inspect({'path':str(a.plan.resolve()),'sha256':a.plan_sha256},a.state))
    else:result=analyze({'path':str(a.plan.resolve()),'sha256':a.plan_sha256},{'path':str(a.authority.resolve()),'sha256':a.authority_sha256},a.state,a.output,a.first_read_receipt,a.no_plots)
    print(json.dumps(result))


if __name__=='__main__':main()
