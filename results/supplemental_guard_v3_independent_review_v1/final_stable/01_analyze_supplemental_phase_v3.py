#!/usr/bin/env python3
"""Retained v2 statistics with an explicit, non-authorizing v3 gate repair."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from replay_study import supplemental_analysis_guard_v3 as guard


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


def analyze(plan_proof,authority_proof,state_path,output,repair_proof,first_receipt=None,no_plots=False):
    plan,_=guard.authorize(plan_proof,authority_proof,repair_proof);ready=guard.inspect(plan_proof,state_path,repair_proof)
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
    guard.immutable(output/'COMPLETE.json',{'complete':True,'plan':plan_proof,'authority':authority_proof,'readiness':readiness,'operational_repair':repair_proof,
        'checkpoint_verification':plan['checkpoint_verification'],'verification_work':ready['verification_work'],
        'first_read_receipt':first,'parsed_phase':contract['phase'],'endpoints':results,'GPU_inference_started':False,'selection_recomputed':False,'frozen_primary_statistics_modified':False})
    return {'complete':True,'contract_id':plan['contract_id'],'output':str(output),'GPU_inference_started':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for action in ('check','analyze'):
        child=sub.add_parser(action)
        child.add_argument('--plan',type=Path,required=True);child.add_argument('--plan-sha256',required=True)
        child.add_argument('--state',type=Path,required=True);child.add_argument('--output',type=Path,required=True)
        child.add_argument('--operational-repair',type=Path,required=True);child.add_argument('--operational-repair-sha256',required=True)
        if action=='analyze':
            child.add_argument('--authority',type=Path,required=True);child.add_argument('--authority-sha256',required=True)
            child.add_argument('--first-read-receipt',type=Path);child.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    plan={'path':str(args.plan.resolve()),'sha256':args.plan_sha256}
    repair={'path':str(args.operational_repair.resolve()),'sha256':args.operational_repair_sha256}
    if args.action=='check':result=guard.immutable(args.output,guard.inspect(plan,args.state,repair))
    else:
        authority={'path':str(args.authority.resolve()),'sha256':args.authority_sha256}
        result=analyze(plan,authority,args.state,args.output,repair,args.first_read_receipt,args.no_plots)
    print(json.dumps(result))


if __name__=='__main__':main()
