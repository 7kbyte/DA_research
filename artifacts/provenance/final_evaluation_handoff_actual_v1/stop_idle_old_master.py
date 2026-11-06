import datetime as dt,fcntl,hashlib,json,os,pathlib,signal,subprocess,sys,time
R=pathlib.Path.cwd().resolve();O=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(R/'scripts'))
from handoff_evaluation import process_groups,assert_quiescent
from schedule_jobs import digest_job
started=time.monotonic();q=json.loads((R/'configs/supplemental_v1_queue.json').read_text());s=json.loads((R/'logs/supplemental_job_state.json').read_text());assert len(q)==len(s['jobs'])==607 and not s['active_jobs']
for j in q:
 v=s['jobs'][j['id']];assert v['status']=='completed'and type(v['returncode'])is int and v['returncode']==0 and not v.get('missing_outputs')and not any('stop'in k.lower()for k in v)and v['argv']==j['argv']and v['definition_sha256']==digest_job(j)
pid=s['scheduler_pid'];assert pid==2988506
argv=pathlib.Path('/proc',str(pid),'cmdline').read_bytes().decode().strip('\0').split('\0');p=json.loads((O/'ROOT_preflight.json').read_text());assert argv==p['old_scheduler']['argv']
proc=process_groups();groups={v['pid']for v in s['jobs'].values()};assert not [n for n,v in proc.items()if n in groups or v['pgrp']in groups or v['ppid']==pid]
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
requested=dt.datetime.now(dt.timezone.utc).isoformat();os.kill(pid,signal.SIGTERM)
for _ in range(150):
 if not pathlib.Path('/proc',str(pid)).exists():break
 time.sleep(.1)
assert not pathlib.Path('/proc',str(pid)).exists(),'Old scheduler not fully gone; do not start'
latest=json.loads((R/'logs/supplemental_job_state.json').read_text());assert_quiescent(latest)
with (R/'logs/supplemental_job_state.lock').open('r')as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
receipt={'status':'old_idle_master_gracefully_terminated_and_quiescent','pid':pid,'argv':argv,'signal':'SIGTERM','requested_at_utc':requested,'completed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'operation_wall_seconds':time.monotonic()-started,'historical_children_and_groups_gone':True,'lock_observed_free_not_lease':True,'GPU_compute_apps_empty':True,'old_queue_sha256':hashlib.sha256((R/'configs/supplemental_v1_queue.json').read_bytes()).hexdigest(),'old_state_after_stop_sha256':hashlib.sha256((R/'logs/supplemental_job_state.json').read_bytes()).hexdigest(),'scientific_jobs_signaled':0}
with(O/'old_master_stop_receipt.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(receipt))
