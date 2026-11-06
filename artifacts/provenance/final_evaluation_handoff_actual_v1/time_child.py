import datetime as dt,json,os,pathlib,resource,subprocess,sys,time
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);argv=sys.argv[2:];before=resource.getrusage(resource.RUSAGE_CHILDREN);t=time.monotonic();utc=dt.datetime.now(dt.timezone.utc).isoformat()
with(out/'stdout.log').open('x')as stdout,(out/'stderr.log').open('x')as stderr:
 child=subprocess.Popen(argv,stdout=stdout,stderr=stderr);code=child.wait()
after=resource.getrusage(resource.RUSAGE_CHILDREN);r={'argv':argv,'returncode':code,'wall_seconds':time.monotonic()-t,'user_cpu_seconds':after.ru_utime-before.ru_utime,'sys_cpu_seconds':after.ru_stime-before.ru_stime,'child_peak_rss_bytes':after.ru_maxrss*1024,'started_at_utc':utc,'finished_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'timing_scope':'Whole direct helper child spawn through exit. Any detached successor scheduler lifetime is excluded.'}
with(out/'whole_child_timing.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(r));raise SystemExit(code)
