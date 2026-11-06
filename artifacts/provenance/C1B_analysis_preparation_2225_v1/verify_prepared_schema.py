"""Plan/repair/cache metadata validation only; never inspect or analyze scores."""
from pathlib import Path
import json,sys,hashlib,collections,datetime
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from replay_study import supplemental_analysis_guard_v3 as guard
from schedule_jobs import digest_job
from stat_bound_digest_cache import signature

def proof(path):return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
items=json.loads((HERE/'plan_inventory.json').read_bytes())
metadata=json.loads((HERE/'metadata_preflight.json').read_bytes())
results=[]
for item in items:
    plan,v=guard.load_plan(item['plan'],item['non_authorizing_v3_repair'])
    assert plan['counts']==item['counts'] and plan['checkpoint_verification']['cache_snapshot']==metadata['cache_snapshot']
    assert plan['counts']['logical_aliases']==(6 if item['contract_id'].endswith('_nll') else 3) and plan['score_access_authorized'] is False
    assert v.accounting()['counters']['checkpoint_fresh_reads']==0 and v.accounting()['counters']['checkpoint_fresh_bytes']==0
    assert not Path(plan['first_read_receipt_path']).exists()
    false=json.loads(Path(item['v3_FALSE_template']['path']).read_bytes())
    assert false['allow_score_analysis'] is False and false['operational_repair']==item['non_authorizing_v3_repair']
    assert false['actual_successor_immutable_state']['path'] is None and false['actual_successor_immutable_state']['sha256'] is None
    for path,expected in plan['checkpoint_paths'].items():
        m=metadata['checkpoint_expected_and_current_stats'][path]
        assert expected==m['sha256'] and signature(Path(path).stat())==m['current_stat5']
    v.finish()
    results.append({'contract_id':item['contract_id'],'plan':item['plan'],'repair':item['non_authorizing_v3_repair'],'counts':plan['counts'],'schema_validation_only':True,'verification_work':v.accounting()})
statepath=ROOT/'logs/supplemental_job_state.json';raw=statepath.read_bytes();state=json.loads(raw)
queuepath=ROOT/'configs/supplemental_v1_queue.json';queue=json.loads(queuepath.read_bytes());jobs={j['id']:j for j in queue}
manifest=json.loads(Path(items[0]['manifest']['path']).read_bytes());runids={r['run_id'] for r in manifest['records']};assert len(runids)==24
run_records={}
for id in sorted(runids):
    record=state['jobs'][id];job=jobs[id]
    assert record['status']=='completed' and type(record['returncode']) is int and record['returncode']==0
    assert not any(k.startswith('stop_requested') for k in record) and not record.get('missing_outputs')
    assert record['definition_sha256']==digest_job(job) and record['argv']==job['argv']
    run_records[id]=record
value={'status':'passed_metadata_only_prepare_and_v3_schema_validation','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_results':results,
 'original_training_gate_metadata_only':{'state_source_sha256':hashlib.sha256(raw).hexdigest(),'queue':proof(queuepath),'complete_runs':24,'strict_integer_rc0_no_stop_no_missing_exact_definition_actual_argv':True,'records':run_records,'not_the_future_analysis_successor_state':True},
 'payload_read_bytes':{'checkpoint':0,'base':0,'score':0},'check_or_analyze_executed':False,'first_read_journal_created':False,'authority_created':False,'actual_successor_state_binding_pending':True}
with (HERE/'metadata_schema_validation.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'status':value['status'],'plan_counts':[i['counts'] for i in results],'checkpoint_payload_read_bytes':0,'base_payload_read_bytes':0,'score_payload_read_bytes':0,'complete_training_runs':24}))
