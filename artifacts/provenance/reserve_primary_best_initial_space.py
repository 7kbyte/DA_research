from pathlib import Path
import os,json,datetime,time,hashlib
root=Path('/data/202610/7kb/research')
p=Path('/data/202610/7kb/research_delivery/.primary_best_weights_reserve_20261106')
receipt=root/'docs/primary_best_initial_storage_reservation.json'
progress=root/'docs/primary_best_initial_storage_reservation.progress.json'
target=10*2**30
start=time.monotonic();written=0
base=dict(path=str(p),requested_bytes=target,purpose='Initial partial capacity for all36 primary best endpoints (33unique);10GiB is not sufficient to publish full package. Grow only after verifying actual new capacity; both report reserves untouched.',report_reserve_consumed=False,pid=os.getpid(),source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
def record(path,success,error=None):
 d={**base,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'written_bytes':written,'allocated_bytes':p.stat().st_blocks*512 if p.exists() else 0,'file_bytes':p.stat().st_size if p.exists() else 0,'wall_seconds':time.monotonic()-start,'success':success,'error':error}
 t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(path);print(json.dumps(d),flush=True)
try:
 with p.open('xb',buffering=0) as f:
  os.chmod(p,0o600);block=bytes(16<<20)
  while written<target:
   piece=memoryview(block)[:min(len(block),target-written)]
   while piece:
    n=f.write(piece)
    if not n:raise OSError('zero-length reservation write')
    piece=piece[n:];written+=n
   if written%(1<<30)==0:
    os.fsync(f.fileno());record(progress,False)
  os.fsync(f.fileno())
 assert p.stat().st_size==target
 record(receipt,True)
except BaseException as e:
 record(progress,False,repr(e));raise
