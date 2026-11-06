"""Read-only accounting invariants; no Torch/CUDA/model imports."""
import importlib.util
import json
from pathlib import Path

import pytest

source=Path(__file__).resolve().parents[1]/'scripts/inventory_study_costs.py'
spec=importlib.util.spec_from_file_location('study_inventory',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n');return path

def test_aliases_symlinks_and_copied_summaries_do_not_make_new_training(tmp_path):
 root=tmp_path/'runs';run=root/'physical';run.mkdir(parents=True)
 (root/'symlink_alias').symlink_to(run,target_is_directory=True)
 cfg={'output_dir':str(run),'target_prediction_tokens':100,'seed':11}
 counts={'target_prediction_tokens':100,'replay_prediction_tokens':5,'step':4,
         'setup_seconds':1,'training_seconds':10,'validation_seconds':2,'checkpoint_seconds':3,
         'training_teacher_forward_seconds':4,'validation_teacher_forward_seconds':1,
         'validation_kl_arithmetic_seconds':.5,'total_accounted_wall_seconds':17}
 summary={'config':cfg,'status':'completed','counters':counts}
 write(run/'config.json',cfg);write(run/'summary.json',summary)
 copied=write(tmp_path/'copy.json',summary)
 primary=write(tmp_path/'primary.json',[{'id':'physical','output_dir':str(run),'summary_path':str(copied)}])
 supplement=write(tmp_path/'supplement.json',[{'id':'alias','alias_of':'physical','execution_kind':'alias',
     'counted_as_new_training':False,'output_dir':str(root/'unused_alias'),'summary_path':str(copied)}])
 result=module.inventory({'registries':[{'path':str(primary),'category':'primary'},
     {'path':str(supplement),'category':'supplement'}]},root)
 assert len(result['physical_runs'])==1
 row=result['physical_runs'][0]
 assert row['category']=='primary' and row['known_parent_component_sum_seconds']==16
 assert row['unclassified_reported_wall_seconds']==1
 assert row['teacher_children_are_additive'] is False
 total=next(r for r in result['category_totals'] if r['category']=='primary')
 assert total['actual_target_prediction_tokens_known_sum']==100
 assert total['known_parent_component_sum_seconds']==16  # not 16+teacher4+validationteacher1+KL.5
 assert sum(r['counted_as_additional_training'] for r in result['registry_entries'])==1
 assert result['registry_entries'][1]['is_alias'] is True


def test_latest_active_path_can_roll_back_and_archived_or_partial_events_are_not_added(tmp_path):
 root=tmp_path/'runs';run=root/'interrupted';run.mkdir(parents=True)
 write(run/'config.json',{'target_prediction_tokens':100})
 events=run/'events.jsonl'
 lines=[{'event':'train_step','counters':{'step':2,'target_prediction_tokens':20,'replay_prediction_tokens':1,'training_seconds':2}},
        {'event':'rollback_after_checkpoint','step':1},
        {'event':'train_step','counters':{'step':1,'target_prediction_tokens':10,'replay_prediction_tokens':0,'training_seconds':1}}]
 events.write_text(''.join(json.dumps(row)+'\n' for row in lines)+'{"event":')
 write(run/'events.pre_resume_1.jsonl',{'event':'finish','counters':{'target_prediction_tokens':999999}})
 result=module.inventory({},root);row=result['physical_runs'][0]
 assert row['registration_status']=='unregistered' and row['category']=='unregistered'
 assert row['actual_target_prediction_tokens']==10  # last active path, not max20, sum30, or archive999999
 assert row['setup_seconds'] is None and row['all_four_parent_components_reported'] is False
 assert row['incomplete_event_tail_ignored'] is True
 assert row['archived_rollback_events_added_to_counters'] is False
 assert row['status']=='event_path_in_progress_or_interrupted'


def test_missing_planned_registry_entry_has_no_observed_physical_cost(tmp_path):
 registry=write(tmp_path/'planned.json',[{'id':'future','output_dir':str(tmp_path/'runs/future')}])
 result=module.inventory({'registries':[{'path':str(registry),'category':'supplement'}]},tmp_path/'runs')
 assert not result['physical_runs']
 assert result['registry_entries'][0]['physical_directory_observed'] is False
 assert result['registry_entries'][0]['counted_as_additional_training'] is False


def test_shared_generation_dedup_and_lower_bound_do_not_double_count(tmp_path):
 pool=tmp_path/'pool.pt';bank=tmp_path/'bank.pt'
 content={'complete':True,'model_load_wall_seconds':1,'pools':{'11':{'role':'formal',
   'generation':{'config':{'output':str(pool)},'generation_wall_seconds':10,'prediction_tokens':511},
   'teacher_statistics':{'wall_seconds':2},'banks':{'cluster':{'artifact':str(bank),'wall_seconds':3,'clustering_wall_seconds':2.5}}}}}
 summary=write(tmp_path/'generation.json',content)
 copied_summary=write(tmp_path/'copied_generation.json',content)
 loss=write(tmp_path/'incident.json',{'lower_bound':83.31354104727507})
 extra=write(tmp_path/'diagnostic.json',{'trials':[{'wall':4},{'wall':5}]})
 spec={'generation_summaries':[{'path':str(summary),'category':'primary'},{'path':str(copied_summary),'category':'primary'}],
       'loss_bounds':[{'path':str(loss),'pointer':'/lower_bound','reason':'discarded ENOSPC work'}],
       'extra_costs':[{'path':str(extra),'category':'diagnostic','component':'profile','pointers':['/trials/*/wall','/missing']} ]}
 result=module.inventory(spec,tmp_path/'runs')
 primary=next(r for r in result['category_totals'] if r['category']=='primary')
 assert primary['separate_nontraining_recorded_cost_seconds_known_sum']==16
 assert len(result['preparation_deduplication'])==4
 assert result['loss_bounds'][0]['discarded_work_seconds_lower_bound']==pytest.approx(83.313541)
 assert result['loss_bounds'][0]['exact_discarded_work_seconds'] is None
 assert result['loss_bounds'][0]['added_to_exact_recorded_totals'] is False
 assert [r['cost_seconds'] for r in result['preparation_costs'] if r['category']=='diagnostic']==[4,5,None]


def test_unmeasured_coverage_stays_unknown_and_archive_receipt_is_separate(tmp_path):
 receipt={'receipt':{'status':'passed','wall_seconds':2053.5508109889925,'gpu_used':False},'receipt_sha256':'a'*64}
 p=write(tmp_path/'acceptance.json',receipt);copy=write(tmp_path/'copied_acceptance.json',receipt)
 item={'path':str(p),'component':'lossless_weight_archive','status_pointer':'/receipt/status','wall_pointer':'/receipt/wall_seconds',
       'gpu_used_pointer':'/receipt/gpu_used','identity_pointer':'/receipt_sha256'}
 result=module.inventory({'postprocessing_receipts':[item,{**item,'path':str(copy)}]},tmp_path/'runs')
 assert len(result['postprocessing_costs'])==1 and len(result['postprocessing_deduplication'])==1
 post=result['postprocessing_costs'][0]
 assert post['wall_seconds']==pytest.approx(2053.5508109889925) and post['GPU_used'] is False
 assert post['included_in_method_cost_totals'] is False
 assert all(r['known_training_preparation_evaluation_wall_seconds']==0 for r in result['category_totals'])
 coverage={r['component']:r for r in result['coverage']}
 for key in ['data_download_and_acquisition','environment_installation','CPU_dataset_preprocessing','report_plotting_and_packaging','shared_storage_publication']:
  assert coverage[key]['known_recorded_wall_seconds'] is None
  assert coverage[key]['unmeasured_cost_is_zero'] is False
 assert result['metadata']['whole_study_cost_completely_instrumented'] is False
