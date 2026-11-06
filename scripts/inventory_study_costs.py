#!/usr/bin/env python3
"""CPU-only inventory of observed physical runs and recorded preparation costs.

No checkpoint/tensor deserialization, inference, telemetry integration or GPU calls.
Cumulative counters are committed-path accounting, not all physical attempt work.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from replay_study.evaluation_cost_inventory import inventory_evaluations

CATEGORIES={'primary','supplement','pilot','provisional','diagnostic'}
PARENTS=('setup_seconds','training_seconds','validation_seconds','checkpoint_seconds')
CHILDREN=('training_teacher_forward_seconds','validation_teacher_forward_seconds','validation_kl_arithmetic_seconds')
ACCOUNTING='committed-path accounting; excludes rolled-back/discarded work and unfinished operations not reflected in the selected counter snapshot'


class Reader:
    def __init__(self):self.sources={};self.cache={}
    def read(self,path,events=False):
        path=Path(path).resolve();key=str(path)
        if key in self.cache:return self.cache[key]
        if not path.exists():
            self.sources[key]={'exists':False};self.cache[key]=None;return None
        before=path.stat();raw=path.read_bytes();after=path.stat()
        info={'exists':True,'sha256':hashlib.sha256(raw).hexdigest(),'bytes_read':len(raw),
              'mtime_ns':after.st_mtime_ns,'changed_while_reading':before.st_mtime_ns!=after.st_mtime_ns or before.st_size!=after.st_size,
              'snapshot_scope':'exact observed bytes; whole inventory is not an atomic live-study snapshot'}
        if events:
            lines=raw.splitlines();incomplete=bool(raw and not raw.endswith(b'\n'))
            if incomplete:lines=lines[:-1]
            valid=[];invalid=0
            for index,line in enumerate(lines,1):
                if not line.strip():continue
                try:valid.append((index,json.loads(line)))
                except (json.JSONDecodeError,UnicodeDecodeError):invalid+=1
            value={'events':valid,'incomplete_tail_ignored':incomplete,'invalid_complete_lines':invalid}
        else:
            try:value=json.loads(raw)
            except (json.JSONDecodeError,UnicodeDecodeError) as error:
                info['parse_error']=str(error);value=None
        self.sources[key]=info;self.cache[key]=value;return value


def number(value):
    return value if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0 else None


def alias(row):
    return row.get('execution_kind')=='alias' or row.get('counted_as_new_training') is False or bool(row.get('alias_of'))


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as handle:
        if not keys:return
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def pointer_values(value,pointer):
    """Explicit JSON pointer with * expansion; no recursive timer summation."""
    pieces=pointer.strip('/').split('/') if pointer else []
    work=[('',value)]
    for piece in pieces:
        token=piece.replace('~1','/').replace('~0','~');next_work=[]
        for prefix,node in work:
            if token=='*':
                items=enumerate(node) if isinstance(node,list) else node.items() if isinstance(node,dict) else []
                next_work.extend((prefix+'/'+str(key),item) for key,item in items)
            else:
                try:item=node[int(token)] if isinstance(node,list) else node[token]
                except (KeyError,IndexError,ValueError,TypeError):continue
                next_work.append((prefix+'/'+token,item))
        work=next_work
    return work


def coverage_and_postprocessing(spec,reader,runs,preparation,evaluation):
    postprocessing=[];duplicates=[];seen={}
    for item in spec.get('postprocessing_receipts',[]):
        path=str(Path(item['path']).resolve());receipt=reader.read(path)
        def scalar(pointer):
            matches=pointer_values(receipt,pointer) if receipt is not None and pointer else []
            if len(matches)>1:raise ValueError('Postprocessing receipt pointer must identify one scalar')
            return matches[0][1] if matches else None
        status=scalar(item['status_pointer']);cost=number(scalar(item['wall_pointer']))
        complete=status==item.get('expected_status','passed')
        identity=item.get('canonical_id') or scalar(item.get('identity_pointer')) or path
        key=item['component']+':'+str(identity)
        row={'component':item['component'],'canonical_receipt_identity':key,'source':path,
             'source_sha256':reader.sources[path].get('sha256'),'source_wall_pointer':item['wall_pointer'],
             'receipt_status':status,'completed_receipt':complete,'wall_seconds':cost if complete else None,
             'GPU_used':scalar(item.get('gpu_used_pointer')),'timing_kind':'wall_not_GPU_time',
             'included_in_method_cost_totals':False,'parallel_wall_is_study_elapsed':False,
             'scope':item.get('scope'),'unrecorded_remaining_work_is_zero':False}
        if key in seen:
            duplicates.append({'canonical_receipt_identity':key,'kept_source':seen[key]['source'],'duplicate_source':path,
                               'recorded_wall_equal':seen[key]['wall_seconds']==row['wall_seconds']});continue
        seen[key]=row;postprocessing.append(row)
    training=[r['known_parent_component_sum_seconds'] for r in runs if r['known_parent_component_sum_seconds'] is not None]
    prep=[r['cost_seconds'] for r in preparation if r['additive_recorded_cost'] and r['cost_seconds'] is not None]
    evals=[r['known_disjoint_recorded_wall_seconds'] for r in evaluation['evaluation_executions'] if r['known_disjoint_recorded_wall_seconds'] is not None]
    coverage=[]
    for component,status,records,wall,method,limit in [
        ('generation_statistics_selection_and_recorded_diagnostics','recorded_components_partial',len(prep),sum(prep) if prep else None,True,'Explicit source timers only; shared joint stats once, lost/unrecorded attempts excluded'),
        ('physical_training','committed_path_partial',len(training),sum(training) if training else None,True,ACCOUNTING),
        ('evaluation_inference','last_attested_path_partial',len(evals),sum(evals) if evals else None,True,'Explicit canonical inference attestations only; missing/failed/retried work not reconstructed'),
        ('data_download_and_acquisition','not_completely_instrumented',None,None,False,'Network waiting, downloads, decompression and source fetches have no complete structured total'),
        ('environment_installation','not_completely_instrumented',None,None,False,'Dependency/environment setup has no complete structured wall record'),
        ('CPU_dataset_preprocessing','not_completely_instrumented',None,None,False,'Tokenization, overlap audits, data revisions and CPU preparation lack complete structured timing'),
        ('report_plotting_and_packaging','not_completely_instrumented',None,None,False,'Root report, analysis figures, packaging and coordination are not all timed'),
        ('weight_archive_and_verification','partially_measured_receipts' if postprocessing else 'no_explicit_reliable_receipts',len(postprocessing) if postprocessing else None,
         sum(r['wall_seconds'] for r in postprocessing if r['wall_seconds'] is not None) if any(r['wall_seconds'] is not None for r in postprocessing) else None,False,'Only registered receipts below; CPU/IO wall is separate from method/GPU comparison costs'),
        ('shared_storage_publication','not_completely_instrumented_or_in_progress',None,None,False,'Scratch archive acceptance does not establish shared publication completion or full transfer wall'),
        ('rollback_failures_and_discarded_work','lower_bounds_and_gaps_only',None,None,False,'ENOSPC generation 83.313541s is a separate lower bound; remaining losses unknown'),
        ('whole_study_time_and_GPU_activity','timewindow_and_external_sampled_telemetry',None,None,False,'24h authorization/deadline plus sampled telemetry; never sum parallel wall as study elapsed')]:
        coverage.append({'component':component,'coverage_status':status,'structured_record_count':records,'known_recorded_wall_seconds':wall,
                         'included_in_method_comparison_cost_categories':method,'unmeasured_cost_is_zero':False,'complete_physical_cost_claim':False,'limit':limit})
    return {'coverage':coverage,'postprocessing_costs':postprocessing,'postprocessing_deduplication':duplicates,
            'study_timewindow_context':spec.get('study_timewindow_context',{'declared_budget_hours':24,'elapsed_from_component_sum':False})}


def inventory(spec,runs_root,reader=None):
    reader=reader or Reader();runs_root=Path(runs_root).resolve()
    registry_rows=[];by_id=defaultdict(list)
    for source in spec.get('registries',[]):
        category=source['category']
        if category not in CATEGORIES:raise ValueError('Unsupported registry category: '+category)
        rows=reader.read(source['path'])
        if rows is None:continue
        if not isinstance(rows,list):raise ValueError('Registry must be a JSON list: '+source['path'])
        for original in rows:
            row={'category':category,'registry':str(Path(source['path']).resolve()),**original}
            registry_rows.append(row);by_id[row['id']].append(row)
    def physical(row,seen=None):
        seen=set() if seen is None else seen
        if alias(row) and row.get('alias_of'):
            if row['alias_of'] in seen:return None
            targets=by_id.get(row['alias_of'],[])
            resolved={physical(target,seen|{row['alias_of']}) for target in targets}
            resolved.discard(None)
            if len(resolved)==1:return next(iter(resolved))
        return str(Path(row['output_dir']).resolve()) if row.get('output_dir') else None
    references=defaultdict(list);entries=[];counted_physical=set()
    for row in registry_rows:
        path=physical(row)
        record={'id':row['id'],'registry':row['registry'],'declared_category':row['category'],
                'phase':row.get('phase'),'active':row.get('active'),'is_alias':alias(row),
                'alias_of':row.get('alias_of'),'canonical_run_dir':path,'physical_directory_observed':bool(path and Path(path).is_dir()),
                'counted_as_additional_training':bool(not alias(row) and path and Path(path).is_dir() and path not in counted_physical),
                'config_path':row.get('config_path'),'summary_path':row.get('summary_path')}
        entries.append(record)
        if record['counted_as_additional_training']:counted_physical.add(path)
        if path:references[path].append((row,record))
    directories={str(path.resolve()) for path in runs_root.iterdir() if path.is_dir()} if runs_root.exists() else set()
    directories|={path for path in references if Path(path).is_dir()}
    overrides={str(Path(row['path']).resolve()):row for row in spec.get('run_overrides',[])}
    runs=[]
    for key in sorted(directories):
        path=Path(key);refs=references.get(key,[])
        categories={row['category'] for row,_ in refs if not alias(row)}
        category=next(iter(categories)) if len(categories)==1 else 'classification_conflict' if categories else 'unregistered'
        reason='explicit non-alias registry membership' if categories else 'No explicit non-alias registry membership; scientific eligibility is not inferred from a directory name'
        if key in overrides:
            if overrides[key]['category'] not in CATEGORIES:raise ValueError('Unsupported run override category')
            category=overrides[key]['category'];reason=overrides[key]['reason']
        config=reader.read(path/'config.json');summary=reader.read(path/'summary.json')
        summary_source=str((path/'summary.json').resolve()) if summary else None
        # A synced summary is a fallback evidence copy, never another physical run.
        for row,_ in refs:
            if row.get('config_path'):reader.read(row['config_path'])
            copied=reader.read(row['summary_path']) if row.get('summary_path') else None
            if summary is None and copied and str(Path(copied.get('config',{}).get('output_dir','')).resolve())==key:
                summary=copied;summary_source=str(Path(row['summary_path']).resolve())
        config=config or (summary or {}).get('config') or {}
        event_data=reader.read(path/'events.jsonl',events=True)
        events=event_data['events'] if event_data else []
        counter_events=[(line,event) for line,event in events if isinstance(event.get('counters'),dict)]
        counters=(summary or {}).get('counters',{})
        counter_source=summary_source
        selected_event=None
        if counter_events:
            line,event=counter_events[-1]
            event_source=str((path/'events.jsonl').resolve())
            summary_mtime=reader.sources.get(summary_source,{}).get('mtime_ns',-1)
            if not summary or reader.sources[event_source]['mtime_ns']>=summary_mtime:
                counters=event['counters'];counter_source=event_source;selected_event=(line,event)
        status=(summary or {}).get('status','no_summary_in_progress_or_interrupted')
        if selected_event:
            last=selected_event[1]
            if last.get('event')=='finish':status=last.get('status',status)
            elif not summary or counters.get('step')!=(summary or {}).get('counters',{}).get('step'):
                status='event_path_in_progress_or_interrupted'
        all_parents=all(number(counters.get(name)) is not None for name in PARENTS)
        component_sum=sum(number(counters.get(name)) or 0 for name in PARENTS)
        total=number(counters.get('total_accounted_wall_seconds'))
        row={'canonical_run_dir':key,'run_directory_name':path.name,'category':category,'classification_reason':reason,
             'registration_status':'registered' if refs else 'unregistered','status':status,
             'summary_present':summary is not None,'config_present':bool(config),'events_present':bool(event_data),
             'counter_source':counter_source,'counter_event_line':selected_event[0] if selected_event else None,
             'counter_event_kind':selected_event[1].get('event') if selected_event else None,
             'counter_durability':'active_event_path_not_checkpoint_verified' if selected_event and selected_event[1].get('event')!='finish' else 'summary_or_finish_record',
             'step':number(counters.get('step')),'actual_target_prediction_tokens':number(counters.get('target_prediction_tokens')),
             'actual_replay_prediction_tokens':number(counters.get('replay_prediction_tokens')),
             'planned_target_prediction_tokens':config.get('target_prediction_tokens'),'model':config.get('model_name_or_path'),
             'revision':config.get('model_revision'),'seed':config.get('seed'),'run_label':config.get('run_label'),
             **{name:number(counters.get(name)) for name in (*PARENTS,*CHILDREN)},
             'known_parent_component_sum_seconds':component_sum if any(number(counters.get(n)) is not None for n in PARENTS) else None,
             'all_four_parent_components_reported':all_parents,'reported_total_accounted_wall_seconds':total,
             'unclassified_reported_wall_seconds':total-component_sum if total is not None and all_parents else None,
             'teacher_children_are_additive':False,'teacher_timing_complete':(summary or {}).get('timing',{}).get('training_teacher_forward_seconds_complete'),
             'accounting_scope':ACCOUNTING,'live_process_status':'not inspected; missing summary alone does not prove a job is currently running',
             'registry_ids':[entry['id'] for _,entry in refs],'alias_reference_count':sum(alias(r) for r,_ in refs),
             'invalid_event_lines':event_data['invalid_complete_lines'] if event_data else 0,
             'incomplete_event_tail_ignored':event_data['incomplete_tail_ignored'] if event_data else False,
             'source_summary':summary_source,'source_config':str((path/'config.json').resolve()),
             'source_events':str((path/'events.jsonl').resolve()) if event_data else None,
             'summary_timing_metadata':(summary or {}).get('timing',{})}
        archives=sorted(path.glob('events.pre_resume_*.jsonl'))
        row['rollback_archive_paths']=[str(p.resolve()) for p in archives]
        for archive in archives:reader.read(archive,events=True)
        row['archived_rollback_events_added_to_counters']=False
        runs.append(row)
    preparation=[];dedup={};duplicates=[]
    def add(record):
        identity=record.pop('_identity')
        if identity in dedup:
            previous=dedup[identity]
            duplicates.append({'physical_component':identity,'kept_source':previous['source'],'duplicate_source':record['source'],
                               'recorded_costs_equal':previous['cost_seconds']==record['cost_seconds']})
            return
        dedup[identity]=record;preparation.append(record)
    generation_refs=[]
    for item in spec.get('generation_summaries',[]):
        source=str(Path(item['path']).resolve());summary=reader.read(source)
        if not summary:continue
        generation_refs.append({'source':source,'complete':summary.get('complete'),
                                'reported_orchestration_wall_seconds_not_additive':summary.get('orchestration_wall_seconds')})
        pipeline_dirs=sorted({str(Path(pool['generation']['config']['output']).parent.resolve())
                              for pool in summary.get('pools',{}).values()})
        pipeline_key='|'.join(pipeline_dirs) if pipeline_dirs else source
        add({'_identity':pipeline_key+'#model_loading','category':item['category'],'component':'model_loading',
             'cost_seconds':number(summary.get('model_load_wall_seconds')),'source':source,'source_pointer':'/model_load_wall_seconds',
             'timing_kind':'wall','training_exposures':False,'additive_recorded_cost':True})
        for seed,pool in summary.get('pools',{}).items():
            gen=pool['generation'];artifact=str(Path(gen['config']['output']).resolve())
            category='pilot' if pool.get('role')=='pilot' else item['category']
            common={'category':category,'pool_seed':int(seed),'pool_role':pool.get('role'),'canonical_pool':artifact,
                    'source':source,'timing_kind':'wall','training_exposures':False,'additive_recorded_cost':True}
            add({'_identity':artifact+'#generation',**common,'component':'generation',
                 'cost_seconds':number(gen.get('generation_wall_seconds')),'source_pointer':f'/pools/{seed}/generation/generation_wall_seconds',
                 'generated_prediction_tokens':gen.get('prediction_tokens'),'peak_gpu_memory_bytes':gen.get('peak_gpu_memory_bytes'),
                 'source_artifact_sha256_reported':gen.get('artifact_sha256')})
            stats=pool.get('teacher_statistics',{})
            add({'_identity':artifact+'#joint_teacher_stats',**common,'component':'joint_teacher_stats_shared_once',
                 'cost_seconds':number(stats.get('wall_seconds')),'source_pointer':f'/pools/{seed}/teacher_statistics/wall_seconds'})
            for method,bank in pool.get('banks',{}).items():
                add({'_identity':str(Path(bank['artifact']).resolve())+'#selection',**common,'component':'bank_selection_'+method,
                     'cost_seconds':number(bank.get('wall_seconds')),'source_pointer':f'/pools/{seed}/banks/{method}/wall_seconds',
                     'clustering_wall_seconds_nested_diagnostic':number(bank.get('clustering_wall_seconds'))})
    for item in spec.get('reuse_summaries',[]):
        source=str(Path(item['path']).resolve());summary=reader.read(source)
        if not summary:continue
        for index,bank in enumerate(summary.get('banks',[])):
            if 'selection_seconds' not in bank:continue  # exact copied banks incur no new recorded generation/selection
            add({'_identity':str(Path(bank['artifact']).resolve())+'#selection','category':item['category'],
                 'component':'new_c4_selection_for_reused_generation','cost_seconds':number(bank['selection_seconds']),
                 'source':source,'source_pointer':f'/banks/{index}/selection_seconds','timing_kind':'wall',
                 'training_exposures':False,'additive_recorded_cost':True})
    for item in spec.get('extra_costs',[]):
        source=str(Path(item['path']).resolve());value=reader.read(source)
        if value is None:continue
        for pointer in item['pointers']:
            matches=pointer_values(value,pointer)
            for resolved,cost in matches or [(pointer,None)]:
                add({'_identity':source+'#'+resolved,'category':item['category'],'component':item['component'],
                     'cost_seconds':number(cost),'source':source,'source_pointer':resolved,
                     'timing_kind':item.get('timing_kind','wall'),'training_exposures':False,
                     'additive_recorded_cost':item.get('additive_recorded_cost',True),'scope':item.get('scope')})
    bounds=[]
    for item in spec.get('loss_bounds',[]):
        source=str(Path(item['path']).resolve());value=reader.read(source)
        if value is None:continue
        for pointer,cost in pointer_values(value,item['pointer']):
            bounds.append({'source':source,'source_pointer':pointer,'category':item.get('category','diagnostic'),
                           'discarded_work_seconds_lower_bound':number(cost),'exact_discarded_work_seconds':None,
                           'added_to_exact_recorded_totals':False,'reason':item['reason']})
    evaluation=inventory_evaluations(spec,reader)
    totals=[]
    for category in sorted(CATEGORIES|{row['category'] for row in runs}|{row['category'] for row in evaluation['evaluation_category_totals']}):
        selected=[row for row in runs if row['category']==category]
        total={'category':category,'observed_physical_run_directories':len(selected),
               'summary_completed_runs':sum(row['status']=='completed' for row in selected),
               'incomplete_or_interrupted_runs':sum(row['status']!='completed' for row in selected),
               'counting_scope':ACCOUNTING,'parent_timers_are_wall_seconds_not_pure_gpu_time':True}
        for field in ('actual_target_prediction_tokens','actual_replay_prediction_tokens',*PARENTS,*CHILDREN):
            values=[row[field] for row in selected if row[field] is not None]
            total[field+'_known_sum']=sum(values);total[field+'_missing_records']=len(selected)-len(values)
        total['known_parent_component_sum_seconds']=sum(total[k+'_known_sum'] for k in PARENTS)
        total['teacher_child_seconds_added_to_parent_sum']=False
        costs=[row for row in preparation if row['category']==category and row['additive_recorded_cost']]
        total['separate_nontraining_recorded_cost_seconds_known_sum']=sum(row['cost_seconds'] for row in costs if row['cost_seconds'] is not None)
        total['separate_nontraining_cost_records_missing_timer']=sum(row['cost_seconds'] is None for row in costs)
        totals.append(total)
    telemetry=None
    if spec.get('telemetry_summary'):
        path=spec['telemetry_summary'];data=reader.read(path)
        telemetry={'path':str(Path(path).resolve()),'sha256':reader.sources[str(Path(path).resolve())].get('sha256'),
                   'existing_summary':data,'recomputed_by_inventory':False,
                   'scope':'External sampled telemetry proxy; not per-run attribution and not summed with wall counters'}
    for row in totals:
        observed=next((r for r in evaluation['evaluation_category_totals'] if r['category']==row['category']),{})
        row['evaluation_SHA_manifest_verification_seconds_known_sum']=observed.get('SHA_and_manifest_verification_seconds_known_sum',0)
        row['evaluation_scorer_subprocess_seconds_known_sum']=observed.get('scorer_subprocess_seconds_known_sum',0)
        row['evaluation_known_disjoint_recorded_wall_seconds']=observed.get('known_disjoint_recorded_wall_seconds',0)
        row['evaluation_missing_or_failed_unattested_jobs']=observed.get('missing_or_failed_unattested_jobs',0)
        row['known_training_preparation_evaluation_wall_seconds']=row['known_parent_component_sum_seconds']+row['separate_nontraining_recorded_cost_seconds_known_sum']+row['evaluation_known_disjoint_recorded_wall_seconds']
        row['combined_wall_scope']='Known recorded components only; missing/rollback/failed-attempt work is not reconstructed; not elapsed study time or pure GPU time'
    coverage=coverage_and_postprocessing(spec,reader,runs,preparation,evaluation)
    return {**coverage,**evaluation,'physical_runs':runs,'registry_entries':entries,'preparation_costs':preparation,'loss_bounds':bounds,
            'category_totals':totals,'source_snapshots':reader.sources,'preparation_deduplication':duplicates,
            'generation_summary_metadata':generation_refs,'telemetry_reference':telemetry,
            'metadata':{'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'runs_root':str(runs_root),
                'accounting_scope':ACCOUNTING,'canonical_path_deduplication':True,'aliases_are_not_new_training':True,
                'unknown_source_work_not_imputed':True,'teacher_timers_nested_not_additive':True,
                'wall_seconds_are_not_gpu_compute_seconds':True,'rollback_archives_not_summed':True,
                'checkpoint_or_tensor_deserialization':False,'model_or_task_inference':False,
                'evaluation_result_scores_or_internal_timers_parsed':False,
                'evaluation_canonical_key':'manifest SHA256 + inference job ID; explicit reuse resolves original identity',
                'evaluation_accounting':'Last attested path; failed/retried attempts without separate attestations remain unknown',
                'whole_study_cost_completely_instrumented':False,'unmeasured_costs_imputed_as_zero':False,
                'postprocessing_receipt_wall_added_to_method_totals':False,
                'evaluation_cost_module_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'src/replay_study/evaluation_cost_inventory.py').read_bytes()).hexdigest(),
                'inventory_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'status_is_last_recorded_state_not_process_liveness':True,
                'missing_sources':[path for path,row in reader.sources.items() if not row['exists']],
                'non_atomic_live_snapshot':True}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True,help='Explicit registry roles and source JSON pointer specification')
    parser.add_argument('--runs-root',type=Path,default=Path('/tmp/grad-research-20261106/runs'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reader=Reader();spec=reader.read(args.sources)
    if spec is None:parser.error('Missing or invalid source specification')
    result=inventory(spec,args.runs_root,reader)
    result['metadata']['source_spec']=str(args.sources.resolve())
    result['metadata']['source_spec_sha256']=reader.sources[str(args.sources.resolve())]['sha256']
    args.output.mkdir(parents=True,exist_ok=True)
    for key in ('physical_runs','registry_entries','preparation_costs','loss_bounds','category_totals','evaluation_manifest_status','evaluation_executions','evaluation_canonical_inferences','evaluation_references','evaluation_category_totals','coverage','postprocessing_costs'):
        csv_write(args.output/(key+'.csv'),result[key])
    (args.output/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'physical_runs':len(result['physical_runs']),
                      'registry_rows':len(result['registry_entries']),'inference':False}))


if __name__=='__main__':main()
