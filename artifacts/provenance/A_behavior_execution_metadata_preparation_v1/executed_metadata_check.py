import json,hashlib,pathlib,stat,time,datetime
P=pathlib.Path;W=P('/tmp/grad-research-20261106/worktree');D=P(__file__).resolve().parent;cache={};start=time.monotonic();cpu=time.process_time()
def need(v,m):
 if not v:raise ValueError(m)
def stamp(p):
 s=P(p).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def raw(p):
 p=P(p).resolve();k=str(p)
 if k not in cache:
  s=stamp(p);need(s['bytes']<32<<20 and p.suffix in ['.py','.json'],'small metadata only');b=p.read_bytes();need(stamp(p)==s,'metadata changed');cache[k]=(b,hashlib.sha256(b).hexdigest())
 return cache[k][0]
def sha(p):raw(p);return cache[str(P(p).resolve())][1]
def read(p):return json.loads(raw(p))
def bind(r):need(sha(r['path'])==r['sha256'],'bound SHA:'+r['path']);return r
planp=W/'artifacts/analysis_prepared_v2/supplement_A_behavior.json';plan=read(planp);need(sha(planp)=='7ab89cc4fc71287d3df8b04098c51db92a4720ae10eff36ed421f5b44d82ed8a','frozen A plan')
falsep=planp.with_name('supplement_A_behavior_root_authority_TEMPLATE.json');false=read(falsep);need(sha(falsep)=='e88c6cd50d38d27cf41a55e8c139a85f3afba356e943fdbd3597a64de367b9ef' and false['allow_score_analysis'] is False and false['allow_partial'] is False,'unchanged false authority')
for key in ['protocol','ready','manifest']:bind(plan[key]);need(false[key]==plan[key],'FALSE source identity')
need(false['plan']==dict(path=str(planp),sha256=sha(planp)) and false['expected_counts']==plan['counts'] and false['checkpoint_verification']==plan['checkpoint_verification'],'FALSE scope exact')
for ref in [plan['checkpoint_verification']['cache_snapshot'],plan['checkpoint_verification']['cache_reader_source'],plan['selection']['selection']]:bind(ref)
protocol=read(plan['protocol']['path']);scripts={}
for rel in ['scripts/analyze_supplemental_phase_v2.py','src/replay_study/supplemental_analysis_guard_v2.py']:
 q=W/rel;expected=plan['preparation_source_sha256'][str(q)];need(sha(q)==expected,'original v2 source frozen');scripts[str(q)]=sha(q)
need(plan['contract_id']=='supplement_A_behavior' and plan['contract']['mode']=='behavior' and plan['counts']==dict(closure_canonical_forward_jobs=19,closure_queue_jobs=22,closure_reuse_materializations=3,imported_baselines=1,imported_canonical_forward_jobs=4,logical_aliases=3,logical_endpoints=18,new_forward_jobs=15,queue_jobs=18,reuse_jobs=3),'exact selected18 closure')
need(len(plan['selection']['selected_ids'])==18 and len(plan['closure_jobs'])==22 and plan['prepared_only'] is True and plan['score_access_authorized'] is False,'frozen preparation only')
statep=W/'artifacts/provenance/final_evaluation_handoff_actual_v1/successor/state.json';state=read(statep);checks=[]
for item in plan['closure_jobs']:
 r=state['jobs'].get(item['job_id'],{});definition=item['definition'];h=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest();ok=r.get('status')=='completed' and type(r.get('returncode')) is int and r['returncode']==0 and not r.get('missing_outputs') and 'stop_requested' not in r and 'stop_requested_at' not in r and r.get('argv')==definition['argv'] and r.get('definition_sha256')==h
 need(ok,'Incomplete/stopped/foreign argv closure:'+item['job_id']);checks.append(dict(job_id=item['job_id'],kind=item['kind'],strict_status_rc0_no_stop_missing_argv_definition_passed=True,definition_sha256=h))
need(not P(plan['first_read_receipt_path']).exists(),'A first-read journal unexpectedly exists')
ops=W/'artifacts/provenance/supplement_A_behavior_analysis_actual_v1';frozenstate=ops/'actual_successor_state_immutable.json';readiness=ops/'check_readiness.json';authority=ops/'ROOT_A_behavior_authority.json';out=W/'results/supplemental_analysis_v2/supplement_A_behavior';need(not out.exists(),'analysis output already exists')
script=str(W/'scripts/analyze_supplemental_phase_v2.py');ph=sha(planp)
commands=dict(prepared_only=True,execution_authorized=False,original_v2_only=True,source_live_state=str(statep),future_operations_namespace=str(ops),immutable_state_path=str(frozenstate),immutable_state_SHA256='PENDING_ROOT_ACTUAL_CAPTURE',check_argv=['/usr/bin/python',script,'check','--plan',str(planp),'--plan-sha256',ph,'--state',str(frozenstate),'--output',str(readiness)],authority_source_template=dict(path=str(falsep),sha256=sha(falsep)),authority_output=str(authority),analyze_argv=['/usr/bin/python',script,'analyze','--plan',str(planp),'--plan-sha256',ph,'--authority',str(authority),'--authority-sha256','PENDING_ROOT_TRUE_AUTHORITY_SHA256','--state',str(frozenstate),'--output',str(out)],first_read_receipt_path=plan['first_read_receipt_path'],checkpoint_verification=plan['checkpoint_verification'],source_SHA_bindings=dict(plan=dict(path=str(planp),sha256=ph),ready=plan['ready'],manifest=plan['manifest'],protocol=plan['protocol']),external_ROOT_requirements=['Capture actual completed phase state only once and bind path/SHA before first check. Other phases may still run.','Readiness must ready=true with counts22 and no pending; scheduler_state/scheduler_state_sha256 must equal captured source.','ROOT independently reviews actual check, then creates new TRUE from exact unchanged FALSE template. Add exact immutable state/check path+SHA as external operational evidence.','Analyze uses identical state path/SHA; never reset or change original first-read journal.','Use unique stdout/stderr/full Popen RUSAGE_CHILDREN timers for check and analyze; nested work not additive.'])
(D/'future_commands_PREPARED_ONLY.json').open('x').write(json.dumps(commands,indent=2,ensure_ascii=False)+'\n')
result=dict(status='passed_metadata_closure_observation_preparation_only_not_guard_check',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=dict(path=str(planp),sha256=ph),FALSE_template=dict(path=str(falsep),sha256=sha(falsep)),original_v2_source_SHA=scripts,counts=plan['counts'],closure_checks=checks,observed_live_state=dict(path=str(statep),sha256_at_observation=sha(statep),current_stat=stamp(statep),copied=False,note='Mutable live-state observation only; this SHA is not the future ROOT immutable-state binding'),first_read_journal=dict(path=plan['first_read_receipt_path'],exists=False),frozen_selection=plan['selection']['selection'],original_check_analyze_or_load_plan_executed=False,guard_source_and_outputs_fresh_verification_not_performed=True,score_output_or_CP_base_TAR_ZIP_payload_reads=False,TRUE_created=False,state_copied=False,read_scope=dict(metadata_files=len(cache),metadata_read_bytes=sum(len(v[0]) for v in cache.values())),region_wall_seconds=time.monotonic()-start,region_process_cpu_seconds=time.process_time()-cpu,manual_setup_and_writing_whole_seconds=None)
(D/'metadata_closure_observation.json').open('x').write(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(dict(status=result['status'],closure_count=len(checks),observed_state_sha256=sha(statep),first_read_exists=False,counts=plan['counts'])))
