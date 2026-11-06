#!/usr/bin/env python3
"""Observe metadata only; no handoff, source-input hashing, scored output reads or starts."""
import collections,datetime,hashlib,json
from pathlib import Path
R=Path('/data/202610/7kb/research').resolve();O=Path(__file__).resolve().parent;proofs={}
def get(path,expected=None,snapshot=None):
 p=Path(path);p=p if p.is_absolute() else R/p;b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 assert expected is None or h==expected, str(p)
 proofs[str(p.resolve())]={'sha256':h,'bytes':len(b)}
 if snapshot:
  with (O/snapshot).open('xb') as f:f.write(b)
 return json.loads(b)
def digest(j):return hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()
def main():
 q=get('configs/supplemental_v1_queue.json',snapshot='master_queue.json');s=get('logs/supplemental_job_state.json',snapshot='master_state.json')
 qm={j['id']:j for j in q};assert len(q)==len(qm)==len(s['jobs'])==607 and set(qm)==set(s['jobs'])
 assert all(s['jobs'][n]['definition_sha256']==digest(j) for n,j in qm.items())
 ix=get('artifacts/evaluation_ready_v5/ready_index.json',snapshot='ready_index.json');d=get(ix['handoff_draft']['path'],ix['handoff_draft']['sha256'],snapshot='handoff_draft.json')
 cp=get('artifacts/evaluation_ready_v5/cache_progress.json',snapshot='cache_progress.json');cache=get(cp['cache_snapshot']['path'],cp['cache_snapshot']['sha256'])
 get(ix['catalog']['path'],ix['catalog']['sha256'])
 union={};wanted=[];counts=[]
 for role in ('evaluation_sources','prior_sources'):
  for source in d[role]:
   queue=get(source['queue_path'],source['queue_sha256']);authorities={}
   for p in source['manifests']:authorities[str(Path(p['path']).resolve())]=(get(p['path'],p['sha256']),p['sha256'])
   for j in queue:
    a=j['argv'];assert len(a)==10 and Path(a[1]).name=='run_frozen_evaluation.py' and a[2]=='--manifest' and a[4]=='--manifest-sha256' and a[6]=='--job-id' and a[7]==j['id'] and a[8:]==['--device','cuda:0']
    m,h=authorities[str(Path(a[3]).resolve())];assert h==a[5];entry=next(e for e in m['jobs'] if e['id']==j['id'])
    assert entry.get('depends_on',[])==j.get('depends_on',[]) and j.get('env')=={'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    assert j['id'] not in union or union[j['id']]==j;union[j['id']]=j
    if j['id'] in qm:assert qm[j['id']]==j
    if role=='evaluation_sources':wanted.append(j['id'])
   counts.append({'role':role,'queue':source['queue_path'],'jobs':len(queue),'exact_same_definition_in_master':sum(j['id'] in qm for j in queue),'allowance_seconds':sorted(set(j['expected_seconds'] for j in queue))})
 selected=set();visiting=set()
 def include(n):
  if n in selected:return
  assert n in union and n not in visiting;visiting.add(n)
  for dep in union[n].get('depends_on',[]):include(dep)
  visiting.remove(n);selected.add(n)
 for n in wanted:include(n)
 assert len(selected)==ix['handoff_dependency_closure_jobs']
 strict={n for n,r in s['jobs'].items() if r['status']=='completed' and type(r.get('returncode')) is int and r['returncode']==0 and not any(k in r for k in ('stop_requested','stop_requested_at')) and not r.get('missing_outputs')}
 phases=[]
 for p in ix['phases']:
  row={k:p.get(k) for k in ('id','ready','status')}
  if p['ready']:
   for k in ('queue','manifest','phase_spec'):get(p[k]['path'],p[k]['sha256'])
   rp=Path(p['manifest']['path']).with_name('READY.json');get(rp);row['actual_READY_file_exists']=True
  phases.append(row)
 pid=s['scheduler_pid'];pidpath=Path('/proc')/str(pid);cmd=pidpath.joinpath('cmdline').read_bytes().decode().split('\0') if pidpath.exists() else []
 groups={r['pid'] for r in s['jobs'].values() if type(r.get('pid')) is int};live=[]
 for p in Path('/proc').glob('[0-9]*/stat'):
  try:
   f=p.read_text().rsplit(')',1)[1].split();n=int(p.parent.name);ppid=int(f[1]);pg=int(f[2])
   if n in groups or pg in groups or ppid==pid:live.append({'pid':n,'ppid':ppid,'pgrp':pg,'state':f[0]})
  except (OSError,ValueError,IndexError):pass
 lock=R/'logs/supplemental_job_state.lock';ls=lock.stat();locks=[x for x in Path('/proc/locks').read_text().splitlines() if f' {pid} ' in x and str(ls.st_ino) in x]
 audit=R/'results/extended_run_audit.json';a=get(audit) if audit.exists() else None
 b=get('results/evaluation_timing_budget_v6.json')
 source_paths=['scripts/handoff_evaluation.py','scripts/schedule_jobs.py','scripts/stat_bound_digest_cache.py','tests/test_evaluation_handoff.py','docs/最终评价交接门槛与来源缺口.md','docs/评价预哈希与v5交接说明.md','docs/评价预算v6_C410完成后更新准备.md']
 for n in source_paths:
  p=R/n;v=p.read_bytes();proofs[str(p)]={'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)}
 assert proofs[str(Path(cache['reader_source']['path']).resolve())]['sha256']==cache['reader_source']['sha256']
 result={'status':'observed_metadata_NOT_finalized_or_authorized','observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'master_jobs':607,'all607_definition_digests_match':True,'master_statuses':dict(collections.Counter(r['status'] for r in s['jobs'].values())),'strict_completed_rc0_no_stop':len(strict),'all607_strict_success':len(strict)==607,'actual_READY_count':sum(bool(p['ready']) for p in phases),'phases':phases,'all63_audit_path':str(audit),'all63_audit_status':a.get('status') if a else 'missing','draft_evaluation_source_count':len(d['evaluation_sources']),'draft_prior_source_count':len(d['prior_sources']),'draft_example_only':d['example_only'],'draft_digest_cache_present':'digest_cache' in d,'source_counts':counts,'raw_source_union_including_unused_priors':len(union),'actual_dependency_closure':len(selected),'selected_jobs_present_in_master':len(selected&set(qm)),'selected_jobs_not_present_in_master':len(selected-set(qm)),'current_structural_inheritance_candidates_NOT_attestation_verified':len(selected&strict),'unused_explicit_prior_job_IDs':sorted(set(union)-selected),'master_only_jobs_outside_current_closure':len(set(qm)-selected),'immutable_cache_snapshot_checked':cp['cache_snapshot'],'cache_progress_timestamp':cp['updated_at_utc'],'cache_entries':len(cache['entries']),'cache_prewarm':cp.get('prewarm'),'cache_selected_input_or_checkpoint_bytes_read':0,'old_master':{'pid':pid,'exists':pidpath.exists(),'argv':cmd,'has_exit_when_idle':'--exit-when-idle' in cmd,'active_jobs':s.get('active_jobs'),'live_historical_child_groups':live,'lock_path':str(lock),'lock_dev_ino':[ls.st_dev,ls.st_ino],'lock_records':locks,'lock_not_acquired_or_written_by_review':True},'budget_v6':{'utc':b['utc'],'accounting':b['accounting'],'scenarios':[{'name':x['name'],'duration_seconds_with_handoff_proxy':x['wall_seconds_with_dispatch_tail_and_handoff_proxy'],'forecast':x['from_current_training_forecast'],'midnight_forecast':x['starts']['from_midnight']} for x in b['sensitivity_scenarios']]},'source_proofs':proofs,'no_scored_output_payload_read_or_hash':True,'no_prepare_check_analyze_start_TRUE_authority_or_live_queue_mutation':True,'tests_not_rerun_existing_coverage_reused':True}
 result['selected_job_IDs']=sorted(selected)
 result['raw_union_definition_sha256']={n:digest(j) for n,j in sorted(union.items())}
 result['master_job_IDs_not_in_selected_closure']=sorted(set(qm)-selected)
 with (O/'review_receipt.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ['master_statuses','strict_completed_rc0_no_stop','actual_READY_count','raw_source_union_including_unused_priors','actual_dependency_closure','selected_jobs_present_in_master','selected_jobs_not_present_in_master','unused_explicit_prior_job_IDs','all63_audit_status']},ensure_ascii=False))
if __name__=='__main__':main()
