from pathlib import Path
import datetime,hashlib,json,os,resource,subprocess,time
root=Path('/tmp/grad-research-20261106/worktree'); output=Path('/tmp/grad-research-20261106/delivery/research_snapshot_20261106_1415.tar.gz'); base=Path(__file__).parent
assert not output.exists()
source=root/'scripts/package_research.py'; digest=hashlib.sha256(source.read_bytes()).hexdigest(); assert digest=='4a05ffaf3ef6963a68fd7053f172063c54ffde6d173ddb06eba445ca04a729e7'
argv=['python','scripts/package_research.py','--output',str(output),'--runs-root','/tmp/grad-research-20261106/runs','--require-run-records','--fail-on-symlinks','--label','intermediate_1415_A_gradient_primary_behavior_gate_C410_admission_724_CPU_tests_C_training']
launch={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,'packager_sha256':digest,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_workspace':str(root),'consistency':'Per-file stable reads during live C training, not a simultaneous snapshot.'}
(base/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
before=resource.getrusage(resource.RUSAGE_CHILDREN); begin=time.perf_counter()
with (base/'stdout.log').open('xb') as out,(base/'stderr.log').open('xb') as err: p=subprocess.run(argv,cwd=root,env=env,stdout=out,stderr=err)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
timing={'returncode':p.returncode,'whole_child_wall_seconds':time.perf_counter()-begin,'whole_child_user_cpu_seconds':after.ru_utime-before.ru_utime,'whole_child_sys_cpu_seconds':after.ru_stime-before.ru_stime,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Complete package subprocess, including compression and archive member verification. No shared publication yet.','nested_subtimers_do_not_add':True,'launch':launch}
(base/'whole_process_timing.json').write_text(json.dumps(timing,indent=2)+'\n')
raise SystemExit(p.returncode)
