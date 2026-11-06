"""Review one completed fixed C1B sample, without loading any model payload."""
from pathlib import Path
import datetime as dt
import hashlib,json,os,stat
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/provenance/additional_endpoint_actual_v4'
DELIVERY=Path('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4')
RID='extended_v2_pythia_1b_m32_formal_B0_s11_w0'
proofs=[]
def need(ok,msg):
    if not ok:raise ValueError(msg)
def read(p,wanted=None):
    p=Path(p);need(p.suffix in {'.json','.py','.md'} or p.name.endswith('.source_metadata.pt'),'Explicit small metadata/source only')
    need(p.stat().st_size<16<<20,'Small evidence bound before open');raw=p.read_bytes()
    h=hashlib.sha256(raw).hexdigest();need(wanted is None or h==wanted,'Evidence SHA mismatch')
    proofs.append(dict(path=str(p),sha256=h,bytes=len(raw)));dest=OUT/'small_evidence'/p.name;dest.parent.mkdir(exist_ok=True)
    if dest.exists() and dest.read_bytes()!=raw:dest=dest.with_name(h+'_'+p.name)
    if not dest.exists():dest.write_bytes(raw)
    return json.loads(raw) if p.suffix=='.json' else raw
def ref(r):return read(r['path'],r['sha256'])
def sig(p):
    s=Path(p).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
spec=read(BASE/'C1B_best_sample_spec.json','18c3b99a6abd2c89b755b6992e97702fc1eed42b7c6d420e80df7a7326c1430c')
auth=ref(spec['authority']);need(spec['authority']['sha256']=='4080f2d1bb5decb7fbf34dfbcd845127a8bc3ddd12f48d19bca1526edd5e5b9f','Fixed sample authority')
launch=read(BASE/'C1B_best_sample/child_launch.json');rootlaunch=read(BASE/'ROOT_C1B_best_sample_launch.json');whole=read(BASE/'C1B_best_sample/whole_child_timing.json')
need(type(whole['returncode']) is int and whole['returncode']==0,'Strict whole success')
need(launch['argv']==spec['argv'] and launch['pid']==3447533 and launch['wrapper_pid']==3447532 and not Path('/proc/3447533').exists() and not Path('/proc/3447532').exists(),'Actual completed child identity')
need(spec['argv'][-6:]==['--slot','slot1','--package','C1B_best','--max-new-weights','1'] and spec['env']['CUDA_VISIBLE_DEVICES']=='','Sample only CPU argv')
pre=ref(auth['metadata_precheck']);need(len(pre['actual24training'])==24,'Full24 precheck')
need(rootlaunch['resource_available_GiB']>=24,'Launch fresh memory evidence')
for p in auth['metadata_SHA_checks']:read(p['path'],p['sha256'])
completion=ref(pre['whole_completion']);audit=ref(pre['whole_audit']);manifest=ref(pre['manifest']);ready=ref(pre['READY'])
need(completion['status']==audit['status']=='passed' and completion['expected_runs']==completion['completed_runs']==audit['expected_runs']==audit['completed_runs']==24,'Actual complete cohort')
best=ref(completion['proofs']['best']);final=ref(completion['proofs']['final'])
for p in [best,final]:need(len(p['comparisons'])==3 and {r['seed'] for r in p['comparisons']}=={11,22,33} and all(r['bitwise_model_state_equal'] is True for r in p['comparisons']),'Actual three endpoint-specific equalities')
bound=next(r for r in manifest['records'] if r['run_id']==RID and r['kind']=='best')
record=read(DELIVERY/'C1B_best/records'/(RID+'__best.json'))
need(record['status']=='verified_additional_endpoint_not_yet_assembled' and record['run_id']==RID and record['group']=='B0' and record['training_seed']==11 and record['kind']=='best' and record['additional_encoding_calls']==1,'Fixed single sample identity')
need(record['actual_exposure']==dict(checkpoint_step=612,target_prediction_tokens=20014848,replay_prediction_tokens=0,full_training_target_prediction_tokens=100000000,full_training_replay_prediction_tokens=0),'Selected and full exposure distinct')
need((bound['step'],bound['target_prediction_tokens_at_checkpoint'],bound['replay_prediction_tokens_at_checkpoint'])==(612,20014848,0),'READY selected exposure')
need(record['source_checkpoint']==bound['checkpoint']==auth['first_sample']['checkpoint'] and record['source_checkpoint_sha256']==bound['checkpoint_sha256']==auth['first_sample']['checkpoint_sha256'],'Original frozen best CP binding')
need(record['source_stat']==record['source_fd_five_stat_before']==record['source_fd_five_stat_after']==sig(record['source_checkpoint']),'Current and original FD five-stat')
need(record['all_nonweight_metadata_roundtrip_verified'] is True,'All nonweight metadata roundtrip')
e,d,v=record['encode'],record['decode'],record['bytewise_verification']
need(e['status']=='encoded' and d['status']=='decoded_and_hash_verified' and v['status']=='passed' and d['tensor_raw_hashes_verified']==v['tensor_count']==196 and v['all_tensor_bytes_equal'] is True,'Own196 tensor complete verification')
need(record['model_fingerprint']==e['model_fingerprint']==d['model_fingerprint']==v['model_fingerprint'],'All four complete fingerprints')
need(record['model_fingerprint']['state_elements']==1011781632,'Own model element count')
need(record['archive_sha256']==e['archive_sha256']==v['archive_sha256'] and record['source_checkpoint_sha256']==e['source_checkpoint_sha256']==v['source_checkpoint_sha256'],'Nested archive/source SHA')
need(sig(record['archive_path'])==record['archive_stat'] and Path(d['restored']).stat().st_size==d['restored_bytes'],'Current derived output stat only')
need(record['exporter_sha256']=='c93e0c1d35fe39b3d71155a126db841b76648ba6b3afd58d475be4eb9c9da853' and record['prepared_scope_sha256']=='7adb8f7f0461fd929995dd370ff2a59b2abc5c0d7750a39e745d21d5546d9de3','Frozen implementation/scope')
metadata=Path(record['metadata_path']);need(metadata.parent==DELIVERY/'C1B_best/metadata' and metadata.stat().st_size<1<<20,'Small metadata only')
opaque=read(metadata,record['metadata_sha256'])
policy=read(BASE/'ROOT_C1B_best_batch_review_policy_before_sample_result.json')
need(policy['batch_authorized'] is False and policy['publication_authorized'] is False and dt.datetime.fromisoformat(policy['utc'])<dt.datetime.fromisoformat(whole['finished_at_utc']),'Pre-result policy only')
need(policy['remaining_unique_after_successful_first_sample']==20 and policy['fresh_conservative_available_minimum_bytes']==24*(1<<30),'Own policy counts/memory')
startup=read(DELIVERY/'operations/slot1_startup_input_audit_1794004204903377510.json');need(startup['status']=='passed_fresh_frozen_input_audit_only' and startup['scope_sha256']==record['prepared_scope_sha256'],'Actual startup input audit')
progress=read(DELIVERY/'slot1_progress.json');need(progress['phase']=='C1B_best' and progress['status']=='resource_sample_limit_reached_requires_ROOT_review' and progress['new_weights_this_invocation']==1,'Stopped after first sample')
events=[]
for which in ['before','after']:
    x=dict(line.split() for line in record['cgroup_memory_'+which]['memory.events'].splitlines());need(int(x['oom'])==int(x['oom_kill'])==0,'No OOM in own endpoint observed interval');events.append(x)
observed=dt.datetime.now(dt.timezone.utc);remaining=20*1.25*record['wall_seconds']+900;projected=observed+dt.timedelta(seconds=remaining);finish=projected+dt.timedelta(seconds=1200+300)
result=dict(schema_version=1,status='accepted_fixed_C1B_first_sample_metadata_only_NOT_batch_authority',observed_at_utc=observed.isoformat(),record_identity=dict(run_id=RID,kind='best',step=612,target_prediction_tokens=20014848,replay_prediction_tokens=0,full_training_target_prediction_tokens=100000000,model_fingerprint=record['model_fingerprint'],tensor_count=196,all_tensor_bytes_equal=True,all_nonweight_metadata_roundtrip_verified=True),complete_source=dict(training_runs=24,final_best_identities=48,actual_final_D_equal_pairs=3,actual_best_D_equal_pairs=3,logical_best=24,unique_best=21),sample_process=dict(wrapper_gone=True,child_gone=True,integer_rc0=True,GPU_idle_not_required=True,launch_available_GiB=rootlaunch['resource_available_GiB']),cost=dict(whole_parent=whole,endpoint_wall_seconds_nested=record['wall_seconds'],startup_input_audit_wall_seconds_nested=startup['wall_seconds'],nested_codec_and_cpu_not_added=True,this_independent_review_whole_cost='unknown'),resource=dict(RSS_process_cumulative_highwater_bytes=whole['child_peak_rss_bytes'],cgroup_events_before_after=events,max_event_change_shared_not_attributed=True,future_batch_needs_new_fresh_24GiB_gate=True),policy_projection=dict(policy_is_not_batch_authority=True,remaining_unique=20,own_endpoint_wall_seconds=record['wall_seconds'],factor=1.25,assembly_startup_unmeasured_allowance_seconds=900,remaining_encode_assembly_seconds=remaining,publication_unmeasured_allowance_seconds=1200,margin_seconds=300,if_started_at_review_time_encode_assembly_finish=projected.isoformat(),if_started_at_review_time_publication_plus_margin_finish=finish.isoformat(),encode_before02=projected<dt.datetime(2026,11,7,2,tzinfo=dt.timezone.utc),all_before02_56_32=finish<=dt.datetime(2026,11,7,2,56,32,tzinfo=dt.timezone.utc)),opaque_small_metadata_bytes_copied=len(opaque),CP_base_ZIP_restored_payload_bytes_read=0,no_batch_publication_authority_or_launch=True,source_proofs=proofs,limits=['Historical successful codec/bytewise receipts were checked; tensor payloads were not independently reread.','Small source_metadata.pt was copied and SHA-checked as opaque bytes, not torch-loaded.','Startup and codec/endpoint timers are nested in whole parent; the 900/1200/300 allowances are not new measured costs.','Projection is a time check only; ROOT needs a fresh resource observation and a separate exact authority.'])
(OUT/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=result['status'],record_identity=result['record_identity'],policy_projection=result['policy_projection']),ensure_ascii=False))
