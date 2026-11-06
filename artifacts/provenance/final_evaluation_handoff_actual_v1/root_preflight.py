import collections,datetime as dt,hashlib,json,pathlib,sys,subprocess,heapq,math,os
R=pathlib.Path.cwd().resolve(); O=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(R/'scripts'))
from handoff_evaluation import evaluation_plan,process_groups
from schedule_jobs import digest_job
from stat_bound_digest_cache import StatBoundHashes,signature
proofs={}
def read(p,h=None,copy=None):
 p=pathlib.Path(p).resolve();b=p.read_bytes();s=hashlib.sha256(b).hexdigest();assert h is None or s==h,str(p);proofs[str(p)]={'path':str(p),'sha256':s,'bytes':len(b)}
 if copy:(O/copy).write_bytes(b)
 return json.loads(b)
def write(n,x):
 with (O/n).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
q=read('configs/supplemental_v1_queue.json','37f00a8ac4af9d71aabb6b54f425e3fa222f596e25ce6e049015c0a66185e377',copy='old_queue_preflight.json');s=read('logs/supplemental_job_state.json',copy='old_state_preflight.json')
assert len(q)==len(s['jobs'])==607 and s['active_jobs']==[]
for j in q:
 v=s['jobs'][j['id']];assert v['status']=='completed' and type(v['returncode'])is int and v['returncode']==0 and not v.get('missing_outputs') and not any('stop' in k.lower()for k in v),j['id'];assert v['argv']==j['argv'] and v['definition_sha256']==digest_job(j)
a=read('results/extended_run_audit.json');assert a['status']=='passed'and a['completed_runs']==a['expected_runs']==63
assert [(p['completed_runs'],p['expected_runs'])for p in a['phases']]==[(24,24),(24,24),(15,15)]and all(not p['violations']for p in a['phases'])
ix=read('artifacts/evaluation_ready_v5/ready_index.json',copy='ready_index.json');assert len(ix['phases'])==9 and all(p['ready']is True for p in ix['phases'])
for p in ix['phases']:
 ready=read(pathlib.Path(p['manifest']['path']).with_name('READY.json'));assert ready['ready']is True
 for k in ('queue','manifest','phase_spec'):read(p[k]['path'],p[k]['sha256'])
d=read(ix['handoff_draft']['path'],ix['handoff_draft']['sha256'],copy='draft_snapshot.json');assert not d['pending_plan_ids'] and len(d['evaluation_sources'])==11 and len(d['prior_sources'])==3
cp=read('artifacts/evaluation_ready_v5/cache_progress.json',copy='cache_progress.json');assert cp['ready_phase_count']==9 and cp['prewarm']['pending']==0 and cp['prewarm']['visited']==534
c=StatBoundHashes(snapshot=cp['cache_snapshot']['path'],snapshot_sha256=cp['cache_snapshot']['sha256']);miss=[]
def verify(p,h):
 p=pathlib.Path(p).resolve();old=c._receipt(str(p));st=signature(p.stat())
 if old is not None and old['stat_before']==st:
  assert old['sha256']==h;return str(p),h
 if st['size']>10*2**20:miss.append({'path':str(p),'bytes':st['size']});raise AssertionError('Unbudgeted large fresh fallback '+str(p))
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==h;proofs[str(p)]={'path':str(p),'sha256':h,'bytes':len(b)};return str(p),h
jobs,bindings,inputs=evaluation_plan(d,verify)
old={j['id']:j for j in q};pending=[j for j in jobs if j['id']not in old];assert len(jobs)==514 and len(pending)==215
proc=process_groups();pid=s['scheduler_pid'];assert pid==2988506 and pid in proc
argv=pathlib.Path('/proc',str(pid),'cmdline').read_bytes().decode().strip('\0').split('\0');assert 'scripts/schedule_jobs.py' in argv[1]and '--exit-when-idle'not in argv
assert pathlib.Path(argv[argv.index('--queue')+1]).resolve()==R/'configs/supplemental_v1_queue.json'
groups={v['pid']for v in s['jobs'].values()if type(v.get('pid'))is int};assert not [p for p,v in proc.items()if p in groups or v['pgrp']in groups or v['ppid']==pid]
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],check=True,capture_output=True,text=True);assert not gpu.stdout.strip()
b=read('results/evaluation_timing_budget_v6_refresh_2110.json','3a771074ac991b3b4794f4922090c5a6f2c7d72b0c8ab165a143842970a8db29');scenarios=[]
for case in b['sensitivity_scenarios']:
 rates={p['id']:p['per_job_whole_slot_seconds_proxy']for p in case['per_phase']};done=set(old)&{j['id']for j in jobs};remaining=pending.copy();running=[];t=0;rows=[]
 while remaining or running:
  while running and running[0][0]<=t:
   _,name=heapq.heappop(running);done.add(name)
  for j in remaining.copy():
   if len(running)>=4:break
   if not set(j.get('depends_on',[]))<=done:continue
   bind=bindings[j['id']];m=bind['manifest'];e=bind['entry'];tag=m['baseline_groups'][e['baseline_group']]['model_tag'];records=bind['records']
   if m['mode']=='nll':key='C_'+tag.replace('-','_').replace('.','p')+'_nll_'+('initial'if e['kind']=='initial'else records[0]['kind'])
   elif e['kind']=='reuse':key='A_selected_behavior_materialization'
   elif tag!='pythia-410m':key='C_'+tag.replace('-','_').replace('.','p')+'_behavior_final'
   elif any(r.get('role')=='extended'for r in records):key='C_pythia_410m_behavior_final'
   elif 'supplement_A' in bind['manifest_path']:key='A_selected_behavior'
   else:key='main_behavior_adapted'
   dur=rates[key];end=math.ceil((t+dur)/10)*10;heapq.heappush(running,(end,j['id']));remaining.remove(j);rows.append({'id':j['id'],'phase_rate':key,'start_seconds':t,'proxy_seconds':dur,'expected_seconds_unchanged':j['expected_seconds'],'end_poll_rounded_seconds':end})
  if running:t=running[0][0]
  elif remaining:raise AssertionError('deadlocked')
 scenarios.append({'name':case['name'],'actual_FIFO_job_rows':rows,'proxy_wall_seconds_poll_rounded':t,'preflight_allowance_seconds':300,'finish_if_launch_now_plus_preflight':(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=t+300)).isoformat()})
spec=dict(d,example_only=False,root_finalization_and_launch_authorization_required=False,digest_cache=cp['cache_snapshot'],finalized_by='ROOT',finalized_at_utc=dt.datetime.now(dt.timezone.utc).isoformat());spec['note']='Actual complete607/63audit/9READY. Separate successor; frozen scientific queues and deadline settings unchanged.'
write('final_spec.json',spec)
write('ROOT_preflight.json',dict(status='passed_complete_scope_metadata_preflight',utc=dt.datetime.now(dt.timezone.utc).isoformat(),master_jobs=607,formal_training_runs=267,full_C_audit=63,READY=9,selected_jobs=len(jobs),structural_inheritance_candidates=len(jobs)-len(pending),pending_candidates=len(pending),CP_base_payload_bytes_read=0,opaque_score_bytes_read=0,old_scheduler={'pid':pid,'argv':argv,'active':[]},gpu_process_stdout=gpu.stdout,cache=cp['cache_snapshot'],source_proofs=proofs,scenarios=scenarios,actual_inheritance_requires_helper_fresh_output_verification=True))
print(json.dumps({'status':'passed','jobs':len(jobs),'pending':len(pending),'scenarios':[{k:v for k,v in x.items()if k!='actual_FIFO_job_rows'}for x in scenarios],'spec_sha256':hashlib.sha256((O/'final_spec.json').read_bytes()).hexdigest()}))
