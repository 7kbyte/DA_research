"""Metadata-only C1B best firstsample precheck. Never export or authorize."""
from pathlib import Path
import json,hashlib,os,stat,fcntl,datetime,importlib.util,sys,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from schedule_jobs import digest_job
sources={};source_stats={};small_read_bytes=0

def require(ok,message):
    if not ok:raise ValueError(message)
def sig(s):return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def small(path,expected=None,parse=True):
    global small_read_bytes
    p=Path(path).resolve();key=str(p)
    require(p.suffix not in ['.pt','.safetensors','.zip','.tar','.gz'],'payload path forbidden in metadata reader')
    if key not in sources:
        with p.open('rb') as f:
            before=os.fstat(f.fileno());require(stat.S_ISREG(before.st_mode) and before.st_size<32<<20,'bounded regular small source required')
            raw=f.read((32<<20)+1);require(sig(before)==sig(os.fstat(f.fileno()))==sig(p.stat()),'small source changed during read')
        require(len(raw)==before.st_size,'truncated small read');small_read_bytes+=len(raw)
        sources[key]={'path':key,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};source_stats[key]=sig(before);small.cache[key]=raw
    raw=small.cache[key]
    require(expected is None or sources[key]['sha256']==expected,'source SHA changed: '+key)
    return json.loads(raw) if parse else raw
small.cache={}
def proof(path):small(path,parse=False);return dict(sources[str(Path(path).resolve())])
def ref(item,parse=True):return small(item['path'],item['sha256'],parse)
def resource_snapshot():
    folder=Path('/sys/fs/cgroup');current=int((folder/'memory.current').read_text());limit=int((folder/'memory.max').read_text())
    metrics={k:int(v) for k,v in (line.split() for line in (folder/'memory.stat').read_text().splitlines())}
    events={k:int(v) for k,v in (line.split() for line in (folder/'memory.events').read_text().splitlines())}
    dirty,writeback,unevictable=[metrics.get(k,0) for k in ['file_dirty','file_writeback','unevictable']]
    clean=max(0,min(metrics.get('inactive_file',0),metrics.get('file',0)-metrics.get('shmem',0))-dirty-writeback-unevictable)
    available=max(0,limit-current)+clean
    disk=os.statvfs('/tmp/grad-research-20261106/delivery')
    return {'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'memory_current':current,'memory_max':limit,'memory_stat':metrics,'memory_events':events,'clean_reclaimable_bytes':clean,'conservative_available_bytes':available,'conservative_available_GiB':available/(1<<30),'fresh_gate_GiB':24,'scratch_available_bytes':disk.f_bavail*disk.f_frsize,'formula':'max(0,max-current)+max(0,min(inactive_file,file-shmem)-file_dirty-file_writeback-unevictable)','not_host_free_or_peak_guarantee':True}

def main():
    initial_resource=resource_snapshot()
    small(ROOT/'docs/C1B_best首样例_ROOT预检清单.md',parse=False)
    scope_path=Path('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/scope.json')
    scope=small(scope_path,'7adb8f7f0461fd929995dd370ff2a59b2abc5c0d7750a39e745d21d5546d9de3')
    exporter=ROOT/'scripts/export_additional_endpoints.py';small(exporter,'c93e0c1d35fe39b3d71155a126db841b76648ba6b3afd58d475be4eb9c9da853',False)
    wrapper=ROOT/'artifacts/provenance/additional_endpoint_actual_v4/run_one.py';small(wrapper,'030bf3fbc7d4b325adf4e30067a74650c7ff0d5c129204e6bc8e2aa9491c84ec',False)
    for item in scope['code']:ref(item,False)
    phase=scope['phases']['C1B_best'];context=phase['context'];rows=context['rows'];ids={r['id'] for r in rows}
    require(len(rows)==len(ids)==24,'full24 context required')
    original=context['phase'];require(original['model']=='pythia-1b' and original['target_prediction_tokens']==100000000,'fixed source cohort')
    registry=small(original['active_registry'],original['active_registry_sha256']);require({r['id'] for r in registry}==ids,'registry cohort mismatch')
    snapshot=small(ROOT/original['snapshot'],original['snapshot_sha256']);require(snapshot==context['snapshot'] and snapshot['status']=='passed','input snapshot changed')
    protocol=small(original['protocol'],original['protocol_sha256']);require(protocol==context['protocol'],'protocol changed')
    small(original['queue'],original['queue_sha256'])
    for path,record in snapshot['code'].items():small(ROOT/path,record['sha256'],False)
    ready_path=ROOT/'artifacts/evaluation_ready_v5/extended_pythia_1b_100m_nll/READY.json';ready=small(ready_path)
    require(ready['ready'] is True and ready['status']=='ready' and ready['logical_endpoints']==48,'real complete C1B NLL READY required')
    manifest=ref(ready['manifest']);require(len(manifest['records'])==48 and {r['run_id'] for r in manifest['records']}==ids,'full48 identity cohort')
    by_endpoint={(r['run_id'],r['kind']):r for r in manifest['records']};require(len(by_endpoint)==48 and {k for _,k in by_endpoint}=={'final','best'},'endpoint completeness')
    master=small(ROOT/'configs/supplemental_v1_queue.json');lookup={j['id']:j for j in master};state=small(ROOT/'logs/supplemental_job_state.json')
    cfgs={};summaries={};training=[]
    policy=next(iter(manifest['baseline_groups'].values()))['inference_policy'] if 'inference_policy' in next(iter(manifest['baseline_groups'].values())) else next(iter(manifest['baseline_groups'].values()))['policy']
    for row in rows:
        rid=row['id'];job=lookup[rid];record=state['jobs'][rid]
        require(job==context['definitions'][rid] and record['definition_sha256']==digest_job(job),'training definition changed')
        require(record['status']=='completed' and type(record['returncode']) is int and record['returncode']==0 and not record.get('missing_outputs'),'strict completed rc0 required')
        require(not any(k.startswith('stop_requested') for k in record) and record['argv']==job['argv'],'stop/actual argv mismatch')
        b=by_endpoint[rid,'best'];f=by_endpoint[rid,'final']
        cfg=small(row['config_path'],b['input_sha256'][b['config_path']]);summary=small(row['summary_path'],b['input_sha256'][b['summary_path']])
        require(cfg==summary['config'] and summary['status']=='completed' and summary['counters']['target_prediction_tokens']==100000000,'source completion/config mismatch')
        require(cfg['model_revision']==context['protocol']['model_revision'] and cfg['target_prediction_tokens']==100000000,'source model/horizon mismatch')
        require(sources[str(Path(row['config_path']).resolve())]['sha256']==row['config_sha256'],'frozen config differs')
        p=summary['provenance']
        for key,wanted in {'torch_version':policy['torch_version'],'cuda_version':policy['cuda_version'],'gpu':policy['gpu_name'],'precision':'bf16','attention_backend_policy':'auto','attention_implementation':'sdpa','deterministic_algorithms':True,'deterministic_warn_only':False,'allow_tf32_matmul':True,'allow_tf32_cudnn':True,'cudnn_deterministic':True,'cudnn_benchmark':False,'cublas_workspace_config':':4096:8','model_weight_dtype':'torch.float32','teacher_weight_dtype':'torch.float32'}.items():require(p.get(key)==wanted,'strict execution policy mismatch: '+key)
        require(p['sdpa_backends_enabled']=={'flash':True,'math':True,'memory_efficient':True,'cudnn':True},'four SDPA backend mismatch')
        require(cfg['precision']=='bf16' and cfg['deterministic_algorithms'] is True and cfg['attention_backend']=='auto' and cfg['allow_tf32'] is True,'config precision policy mismatch')
        best=summary['best_target_validation'];require(Path(best['checkpoint_path']).resolve()==Path(b['checkpoint']).resolve() and best['step']==b['step'] and best['target_prediction_tokens']==b['target_prediction_tokens_at_checkpoint'],'best source selection path/exposure mismatch')
        points=[v for v in summary['validation_trajectory'] if v['step']==b['step']];require(len(points)==1 and points[0]['target_prediction_tokens']==b['target_prediction_tokens_at_checkpoint'] and points[0]['replay_prediction_tokens']==b['replay_prediction_tokens_at_checkpoint'],'best selected validation counter mismatch')
        require(Path(summary['checkpoint_path']).resolve()==Path(f['checkpoint']).resolve() and summary['counters']['step']==f['step'] and summary['counters']['replay_prediction_tokens']==f['replay_prediction_tokens_at_checkpoint'],'final exposure/path mismatch')
        cfgs[rid]=cfg;summaries[rid]=summary
        training.append({'run_id':rid,'definition_sha256':record['definition_sha256'],'config':proof(row['config_path']),'summary':proof(row['summary_path']),'full_training_counters':summary['counters'],'best_selection_opaque_sha256':canonical(best),'best_step':b['step'],'best_target_prediction_tokens':b['target_prediction_tokens_at_checkpoint'],'best_replay_prediction_tokens':b['replay_prediction_tokens_at_checkpoint']})
    eq=ready['source_model_equivalence'];complete=ref(eq['completion']);audit=ref(eq['audit']);require(complete['status']=='passed' and audit['status']=='passed' and audit['expected_runs']==audit['completed_runs']==24 and not audit['source_violations'],'whole model source audit not passed')
    for key in ['source_registry','input_snapshot','audit']:ref(complete[key])
    proof_pairs=[]
    for kind in ['final','best']:
        payload=ref(complete['proofs'][kind]);require(payload['status']=='passed' and payload['endpoint']==kind and payload['algorithm']=='sha256_sorted_key_dtype_shape_contiguous_uint8_v1' and len(payload['comparisons'])==3,'whole endpoint proof incomplete')
        require(payload['source_audit']==complete['audit'] and payload['source_registry']==complete['source_registry'] and payload['source_snapshot']==complete['input_snapshot'],'proof source binding mismatch')
        for key in ['producer_source','algorithm_source']:ref(payload[key],False)
        for key in ['source_audit','source_registry','source_snapshot']:ref(payload[key])
        require({x['seed'] for x in payload['comparisons']}=={11,22,33},'three seeds proof required')
        for pair in payload['comparisons']:
            require(pair['endpoint']==kind and pair['kind']=='identical_bank_null_control','proof pair identity changed')
            for side in [pair['left'],pair['right']]:
                b=by_endpoint[side['run_id'],kind];exposure={'step':b['step'],'target_prediction_tokens_at_checkpoint':b['target_prediction_tokens_at_checkpoint'],'replay_prediction_tokens_at_checkpoint':b['replay_prediction_tokens_at_checkpoint']}
                require(side['checkpoint_sha256']==b['checkpoint_sha256'] and Path(side['checkpoint']).resolve()==Path(b['checkpoint']).resolve() and side['checkpoint_exposure']==exposure,'proof CP/exposure source mismatch')
                require(side['config_sha256']==proof(b['config_path'])['sha256'] and side['summary_sha256']==proof(b['summary_path'])['sha256'],'proof config/summary changed')
                if kind=='best':require(canonical(side['actual_selection'])==canonical(summaries[side['run_id']]['best_target_validation']),'opaque best selection binding mismatch')
            equal=pair['left']['model_state_fingerprint']==pair['right']['model_state_fingerprint'];require(type(pair['bitwise_model_state_equal']) is bool and pair['bitwise_model_state_equal']==equal,'proof equality mismatch')
            proof_pairs.append({'endpoint':kind,'seed':pair['seed'],'bitwise_model_state_equal':equal,'left_checkpoint_sha256':pair['left']['checkpoint_sha256'],'right_checkpoint_sha256':pair['right']['checkpoint_sha256'],'model_fingerprint_evidence':pair['left']['model_state_fingerprint'] if equal else None})
    require(len(proof_pairs)==6 and all(p['bitwise_model_state_equal'] for p in proof_pairs) and eq['all_pairs_bitwise_equal'] is True,'actual READY aliases differ from six equal endpoint proofs')
    cache_progress=small(ROOT/'artifacts/evaluation_ready_v5/cache_progress.json');cache_ref=cache_progress['cache_snapshot'];ref(cache_ref)
    reader=ROOT/'scripts/stat_bound_digest_cache.py';small(reader,'9cc01b493b46a5bf391a7c649a379d87011fff077ddeea701eaf8a84bdb568e2',False)
    spec=importlib.util.spec_from_file_location('bounded_cache_reader',reader);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);cache=mod.StatBoundHashes(snapshot=cache_ref['path'],snapshot_sha256=cache_ref['sha256'])
    checkpoints=[]
    for b in manifest['records']:
        cp=Path(b['checkpoint']);require(not cp.is_symlink() and cp.resolve()==cp and stat.S_ISREG(cp.lstat().st_mode),'canonical CP regular path required')
        historical=cache._receipt(str(cp));require(historical is not None,'missing historical CP SHA receipt; no fresh payload fallback allowed')
        ref(cache.entries[str(cp)])
        auth=historical['source_authority'];require(auth.get('kind')=='completed_training_checkpoint' and type(auth.get('returncode')) is int and auth['returncode']==0 and not any(k.startswith('stop_requested') for k in auth),'historical CP source completion invalid')
        require(historical['sha256']==b['checkpoint_sha256'] and historical['stat_before']==mod.signature(cp.stat()) and cp.stat().st_size==b['checkpoint_bytes'] and cp.stat().st_mtime_ns==b['checkpoint_mtime_ns'],'current CP5stat/oldSHA identity mismatch')
        checkpoints.append({'run_id':b['run_id'],'kind':b['kind'],'path':str(cp),'expected_sha256':b['checkpoint_sha256'],'current_stat5':sig(cp.stat()),'historical_full_read_receipt':cache.entries[str(cp)],'current_payload_bytes_read':0})
    require(len(checkpoints)==48,'all48 CP identities required')
    native=[]
    for item in phase['base_files']:
        p=Path(item['path']);expected=scope['frozen_files'][str(p)];require(expected['sha256']==item['sha256'] and expected['stat']==sig(p.stat()),'native base historical source/stat changed')
        native.append({**item,'current_stat5':sig(p.stat()),'fresh_payload_SHA':False})
    reserve=Path('/data/202610/7kb/research_delivery/.extra_C_1b_best_weights_reserve_20261106')
    growth=small(ROOT/'artifacts/provenance/extra_endpoint_reservations_v1/C_1b_best_growth_receipt.json','91791c13f04d477d1d7a7d7d47276c3ae240277aae615a07d40345dcefbf24ce');rs=reserve.lstat()
    actual={**sig(rs),'allocated_bytes':rs.st_blocks*512,'uid':rs.st_uid,'gid':rs.st_gid,'mode':stat.S_IMODE(rs.st_mode),'links':rs.st_nlink}
    require(not reserve.is_symlink() and stat.S_ISREG(rs.st_mode),'reserve not canonical regular file')
    require(all(actual[k]==v for k,v in growth['actual_stat'].items()) and actual['device']==106 and actual['inode']==375195308 and actual['bytes']==actual['allocated_bytes']==75<<30 and actual['uid']==actual['gid']==11471 and actual['mode']==0o600 and actual['links']==1,'75GiB exact reserve identity mismatch')
    predecessor=small('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/C410_best/completion.json','1a2ccfa6238ba3636ad086d6ad7fc50c1a5f77b5e0a0a1731c8791455d04c20f');require(predecessor['status']=='passed_complete_scratch_package_not_published' and predecessor['logical_endpoints']==24 and predecessor['unique_weights']==21 and predecessor['alias_count']==3,'predecessor completion differs')
    batch_timer=small(ROOT/'artifacts/provenance/additional_endpoint_actual_v4/C410_best_batch/whole_child_timing.json','c24ba9a50917cf88478bdccd71c79feb2f0541fe4a2572afa155b68c99f74d5a');require(type(batch_timer['returncode']) is int and batch_timer['returncode']==0,'predecessor batch strict rc0')
    launch=small(ROOT/'artifacts/provenance/additional_endpoint_actual_v4/C410_best_batch/child_launch.json','c3a662a1624a589132ef748befa96a9719c397f46437d86d431856d1f35be290')
    publication=small(ROOT/'docs/C410_best_weights_durable_publication.json','139e54cfcbae7763a1b651abd4b54c2646dbc1af99dff26787cce0b28107091a');acceptance=small(ROOT/'artifacts/provenance/C410_best_publication_actual_v1/ROOT_acceptance.json','ce7c7905d5f7a424eb3832bd83080f17b906b5b45fddd685170338ef3623e3a8')
    require(publication['status']=='passed' and acceptance['status']=='ROOT_accepted_completed_durable_C410_best_publication_with_outer_timing_receipt_failure_recorded','predecessor durable publication not accepted')
    require(sig(Path(publication['destination']).stat())=={k:publication['destination_stat'][k] for k in ['device','inode','bytes','mtime_ns','ctime_ns']},'predecessor durable current stat changed')
    progress=small(ROOT/'artifacts/provenance/C410_best_weights_publication_progress.json');require(progress['status']=='passed' and progress['published'] is True and progress['stage']=='complete','predecessor publication incomplete')
    pids=[launch['pid'],launch['wrapper_pid'],progress['pid']];require(all(not Path('/proc',str(pid)).exists() for pid in pids),'predecessor PID still present')
    small(ROOT/'artifacts/provenance/C410_best_publication_actual_v1/outer_timing_receipt_failure.json','8c284e922503b2fd9e747e377db776aa73e146c9ff1f886901e775193626b13f')
    lock_path=Path('/tmp/grad-research-20261106/delivery/.additional_endpoint_slot1.lock');fd=os.open(lock_path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);lock_stat=sig(os.fstat(fd));fcntl.flock(fd,fcntl.LOCK_UN)
    finally:os.close(fd)
    sorted_rows=sorted(rows,key=lambda r:(r['group']=='D',r['seed'],r['group']));sample_id=sorted_rows[0]['id'];require(sample_id=='extended_v2_pythia_1b_m32_formal_B0_s11_w0','fixed first sample sorting differs')
    sample=by_endpoint[sample_id,'best'];require(sample['checkpoint_sha256']=='25b42945e67b92ecf70b7fee99c82a21b3a57cb0646e725b65dcd905b4f469c1' and sample['step']==612 and sample['target_prediction_tokens_at_checkpoint']==20014848 and sample['replay_prediction_tokens_at_checkpoint']==0,'fixed B0s11 best identity changed')
    folder=Path(scope['output'])/'C1B_best';require(not list((folder/'records').glob('*.json')) and not list((folder/'metadata').glob('*.pt')),'existing partial/verified endpoints: no longer pristine firstsample')
    require(not (ROOT/'artifacts/provenance/additional_endpoint_actual_v4/C1B_best_sample').exists(),'future actual sample log namespace already exists')
    for cp in checkpoints:require(sig(Path(cp['path']).stat())==cp['current_stat5'],'CP changed during metadata precheck')
    for path,before in source_stats.items():require(sig(Path(path).stat())==before,'small source changed before precheck completion')
    require(sig(reserve.stat())=={k:actual[k] for k in ['device','inode','bytes','mtime_ns','ctime_ns']},'reserve changed during precheck')
    fresh=resource_snapshot();require(fresh['conservative_available_bytes']>=24<<30,'fresh conservative available below24GiB: defer')
    now=datetime.datetime.now(datetime.timezone.utc);launch_deadline=datetime.datetime.fromisoformat('2026-11-07T02:00:00+00:00');finish_deadline=datetime.datetime.fromisoformat('2026-11-07T02:56:32+00:00')
    require(now<launch_deadline and now+datetime.timedelta(seconds=420+120)<=finish_deadline,'firstsample time admission rejected')
    result={'status':'passed_metadata_precheck_only_NOT_ROOT_authority_spec_or_launch','at_utc':now.isoformat(),'package':'C1B_best','slot':'slot1','max_new_weights':1,'fixed_sample':{k:sample[k] for k in ['run_id','checkpoint','checkpoint_sha256','checkpoint_bytes','checkpoint_mtime_ns','step','target_prediction_tokens_at_checkpoint','replay_prediction_tokens_at_checkpoint']},'source_config':proof(sample['config_path']),'source_summary':proof(sample['summary_path']),'source_code':{'exporter':proof(exporter),'wrapper':proof(wrapper),'scope':proof(scope_path)},'READY':proof(ready_path),'manifest':ready['manifest'],'whole_completion':eq['completion'],'whole_audit':eq['audit'],'actual24training':training,'all48_CP_identity_current5stat':checkpoints,'cache_snapshot':cache_ref,'proof_pairs':proof_pairs,'native_base_historical_identity_and_currentstat_only':native,'reserve':{'path':str(reserve),'stat':actual,'consumed':False,'payload_read':False},'predecessor':{'completion':proof('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/C410_best/completion.json'),'batch_timer':proof(ROOT/'artifacts/provenance/additional_endpoint_actual_v4/C410_best_batch/whole_child_timing.json'),'publication':proof(ROOT/'docs/C410_best_weights_durable_publication.json'),'ROOT_acceptance':proof(ROOT/'artifacts/provenance/C410_best_publication_actual_v1/ROOT_acceptance.json'),'checked_PIDs_absent':pids,'publication_internal_wall_seconds_nested_region_only':publication['wall_seconds'],'publication_whole_wall_CPU_childrc':None,'outer_publication_timing_failure_retained':True,'batch_copy_not_added_as_new_publication_cost':True},'slot_lock_observation':{'path':str(lock_path),'stat':lock_stat,'exclusive_nonblocking_observed_free_then_released':True,'lease':False,'future_exporter_must_acquire_actual_lock':True},'resource_before':initial_resource,'fresh_resource_gate':fresh,'time_admission':{'checked_at_utc':now.isoformat(),'launch_must_be_strictly_before':launch_deadline.isoformat(),'finish_deadline':finish_deadline.isoformat(),'sample_expected_allowance_seconds':420,'margin_seconds':120,'allowance_not_hard_timeout':True,'allowance_finish_utc':(now+datetime.timedelta(seconds=540)).isoformat(),'margin_after_allowance_seconds':(finish_deadline-now).total_seconds()-540},'no_GPU_idle_gate':True,'all_payload_read_bytes':{'CP':0,'base':0,'ZIP':0,'TAR':0,'score':0,'reserve':0},'small_metadata_bytes_read_by_this_reader':small_read_bytes,'cache_reader_counters':cache.counters,'source_proofs':list(sources.values()),'ROOT_authority_spec_launch_created':False,'exporter_original_startup_and_roundtrip_checks_still_required':True,'sample_only_not_batch_or_publication':True}
    return result
if __name__=='__main__':
    try:
        result=main()
        with (HERE/'metadata_precheck.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
        print(json.dumps({'status':result['status'],'fixed_sample':result['fixed_sample'],'resource_available_GiB':result['fresh_resource_gate']['conservative_available_GiB'],'checked_at_utc':result['at_utc'],'small_metadata_bytes_read':result['small_metadata_bytes_read_by_this_reader']}))
    except BaseException as error:
        with (HERE/'FAILED.json').open('x') as f:json.dump({'status':'metadata_precheck_failed_no_authority_or_launch','error':repr(error),'traceback':traceback.format_exc(),'source_proofs':list(sources.values()),'small_metadata_bytes_read':small_read_bytes},f,ensure_ascii=False,indent=2);f.write('\n')
        raise
