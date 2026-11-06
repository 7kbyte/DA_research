"""ROOT captures actual complete primary closure; no score or attestation payload reads."""
from pathlib import Path
import json,hashlib,os,sys,datetime as dt
R=Path.cwd().resolve();O=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'scripts'))
from schedule_jobs import digest_job
statepath=R/'artifacts/provenance/final_evaluation_handoff_actual_v1/successor/state.json';queuepath=statepath.with_name('queue.json')
def sig(p):
 s=p.stat();return {'dev':s.st_dev,'ino':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
before=sig(statepath);raw=statepath.read_bytes();after=sig(statepath);assert before==after,'State changed during read; no snapshot created'
s=json.loads(raw);frozen=R/'configs/primary_evaluation_v2_execution_matched_behavior_queue.json';qb=frozen.read_bytes();assert hashlib.sha256(qb).hexdigest()=='393f3df2f0213e2e4e4d2387109b750825c797b909f4b05d16cd3cb739f6aaa9';q=json.loads(qb);actual=json.loads(queuepath.read_text());am={j['id']:j for j in actual};assert len(q)==67 and len(am)==514
selected={}
for j in q:
 assert am[j['id']]==j;v=s['jobs'][j['id']];assert v['status']=='completed'and type(v['returncode'])is int and v['returncode']==0 and v['missing_outputs']==[] and not any('stop'in k.lower()for k in v)and v['argv']==j['argv']and v['definition_sha256']==digest_job(j),j['id'];selected[j['id']]=v
p=O/'actual_successor_state_immutable.json'
with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
r={'status':'captured_actual_complete_primary_closure_metadata_only','utc':dt.datetime.now(dt.timezone.utc).isoformat(),'source':str(statepath),'source_stat_before':before,'source_stat_after':after,'immutable_state':{'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()},'actual_queue':{'path':str(queuepath),'sha256':hashlib.sha256(queuepath.read_bytes()).hexdigest()},'selected_completed_job_count':67,'full_executor_job_count':514,'other_phases_may_be_running':True,'selected_definition_argv_strict_int0_no_any_stop_checked':True,'score_or_attestation_payload_bytes_read':0,'selected_completed_records':selected}
with(O/'ROOT_state_capture_receipt.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(O,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps({k:r[k]for k in ['status','utc','immutable_state','actual_queue','selected_completed_job_count']}))
