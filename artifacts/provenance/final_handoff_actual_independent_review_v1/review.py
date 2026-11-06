"""Actual handoff review: only explicit small metadata/source, no scored-output hashes."""
from pathlib import Path
import collections
import datetime as dt
import hashlib
import heapq
import json
import math
import os

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/provenance/final_evaluation_handoff_actual_v1'
OUT=Path(__file__).resolve().parent
proofs={}
def require(ok,msg):
    if not ok:raise ValueError(msg)
def read(p,expected=None):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    require(p.suffix in {'.json','.py'},'Only explicit small metadata/source')
    require(p.stat().st_size<=32<<20,'Small metadata size bound')
    raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
    require(expected is None or h==expected,'SHA mismatch '+str(p))
    proofs[str(p)]=dict(path=str(p),sha256=h,bytes=len(raw))
    dst=OUT/'small_source_bytes'/(h+p.suffix);dst.parent.mkdir(exist_ok=True)
    if not dst.exists():dst.write_bytes(raw)
    return json.loads(raw) if p.suffix=='.json' else raw
def jobsha(j):return hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()
def ref(r):return read(r['path'],r['sha256'])

pre=read(BASE/'ROOT_preflight.json');spec=read(BASE/'final_spec.json');stop=read(BASE/'old_master_stop_receipt.json')
authority=read(BASE/'ROOT_start_authority.json');failure=read(BASE/'ROOT_inline_review_attempt1_failure.json')
handoff=ref(authority['handoff']);launch=read(BASE/'successor/launch.json');queue=read(handoff['queue'],handoff['queue_sha256'])
oldq=read(BASE/'successor/old_queue_snapshot.json',handoff['old_queue_sha256']);olds=read(BASE/'successor/old_state_snapshot.json',handoff['old_state_sha256'])
require(proofs[str(BASE/'final_spec.json')]['sha256']==authority['final_spec_sha256']==handoff['source_spec_sha256'],'Final spec chain')
require(read(BASE/'successor/source_spec.json')==spec,'Exact final spec snapshot')
require(spec['example_only'] is False and not spec['pending_plan_ids'] and len(spec['evaluation_sources'])==11 and len(spec['prior_sources'])==3,'Actual11 sources/3prior required')
require(stop['status']=='old_idle_master_gracefully_terminated_and_quiescent' and stop['pid']==2988506 and stop['scientific_jobs_signaled']==0 and not Path('/proc/2988506').exists(),'Old stop chain')
require(stop['old_queue_sha256']==handoff['old_queue_sha256'] and stop['old_state_after_stop_sha256']==handoff['old_state_sha256'],'Old snapshots differ')
require(len(oldq)==len(olds['jobs'])==607 and olds['active_jobs']==[],'Old complete607')
for job in oldq:
    s=olds['jobs'][job['id']]
    require(s['status']=='completed' and type(s['returncode']) is int and s['returncode']==0 and s['missing_outputs']==[] and not any('stop' in k.lower() for k in s),'Strict old job completion')
    require(s['argv']==job['argv'] and s['definition_sha256']==jobsha(job),'Old exact definition')
require(failure['status']=='failed_before_authority_write' and failure['no_start_authority_written'] is True and failure['whole_process_wall_seconds'] is None,'Overstrict failed review must stay unknown, unauthorized')
require(authority['authorized_by']=='ROOT' and authority['status']=='accepted_for_start' and (authority['total_jobs'],authority['inherited_verified'],authority['pending'])==(514,299,215),'Actual start authority')
require(launch['handoff_sha256']==authority['handoff']['sha256'] and launch['training_jobs_present'] is False and launch['score_files_deserialized'] is False,'Launch binding')
require(launch['old_master_lock_inherited_by_new_scheduler'] is True,'Old lock transfer evidence')
for item in [handoff,launch]:
    v=item['input_verification'];require(v['cache_snapshot']==spec['digest_cache'] and v['counters']==dict(full_reads=38,full_read_bytes=11914634,cache_hits=1063,invalidations=0),'Actual small-fresh/cache counters')
    require(v['cache_hits_are_current_byte_reads'] is False and v['frozen_worker_always_fresh_full_input_hash'] is True,'Cache/fresh distinction')
fresh=authority['fresh_selected_input_metadata_files'];require(len(fresh)==38 and sum(x['bytes'] for x in fresh)==11914634 and authority['fresh_CP_base_input_bytes']==0,'38 small fresh sources')
for x in fresh:
    p=Path(x['path']);require(p.stat().st_size==x['bytes'] and '/checkpoints/' not in str(p) and p.suffix not in {'.pt','.safetensors','.zip','.tar'},'No large payload fallback inventory')
prepare_timer=read(BASE/'prepare_timing/whole_child_timing.json');start_timer=read(BASE/'start_timing/whole_child_timing.json')
for t in [prepare_timer,start_timer]:require(type(t['returncode']) is int and t['returncode']==0,'Helper strict rc0')
require(dt.datetime.fromisoformat(stop['completed_at_utc'])<dt.datetime.fromisoformat(prepare_timer['started_at_utc'])<dt.datetime.fromisoformat(authority['utc'])<dt.datetime.fromisoformat(start_timer['started_at_utc']),'Stop/prepare/authority/start order')

all_jobs={};wanted=[];bindings={}
for role in ['evaluation_sources','prior_sources']:
    for source in spec[role]:
        sq=read(source['queue_path'],source['queue_sha256']);manifests={str(Path(x['path']).resolve()):ref(x) for x in source['manifests']}
        for job in sq:
            a=job['argv'];require(len(a)==10 and Path(a[1]).resolve()==ROOT/'scripts/run_frozen_evaluation.py' and a[2]=='--manifest' and a[4]=='--manifest-sha256' and a[6:]==['--job-id',job['id'],'--device','cuda:0'],'Only frozen evaluation dispatcher')
            mp=str(Path(a[3]).resolve());m=manifests[mp];require(proofs[mp]['sha256']==a[5],'Manifest argv SHA')
            e=next(x for x in m['jobs'] if x['id']==job['id']);require(e.get('depends_on',[])==job.get('depends_on',[]),'Frozen dependencies')
            for path,h in m['common_input_sha256'].items():
                if Path(path).suffix=='.py':read(path,h)
            if job['id'] in all_jobs:require(all_jobs[job['id']]==job,'Same ID definition differs')
            all_jobs[job['id']]=job;bindings[job['id']]=(m,e,[r for r in m['records'] if r['inference_job_id']==job['id']],mp)
            if role=='evaluation_sources':wanted.append(job['id'])
selected=[];seen=set()
def include(name):
    if name in seen:return
    for dep in all_jobs[name].get('depends_on',[]):include(dep)
    seen.add(name);selected.append(all_jobs[name])
for name in wanted:include(name)
require(selected==queue and len(queue)==514,'Exact frozen514 ordered closure')
old={j['id']:j for j in oldq};inherited={k for k,v in handoff['completion_transfer'].items() if v.get('inherited') is True}
pending=[j for j in queue if j['id'] not in inherited]
require(len(inherited)==299 and len(pending)==215 and inherited==set(old).intersection(seen),'Actual verified inheritance partition')
for name in inherited:require(all_jobs[name]==old[name] and handoff['completion_transfer'][name]['definition_sha256']==jobsha(old[name]),'Inherited definition unchanged')

budget_ref=next(v for p,v in pre['source_proofs'].items() if p.endswith('evaluation_timing_budget_v6_refresh_2110.json'))
budget=ref(budget_ref);scenario_reviews=[]
for original in pre['scenarios']:
    case=next(c for c in budget['sensitivity_scenarios'] if c['name']==original['name']);rates={p['id']:p['per_job_whole_slot_seconds_proxy'] for p in case['per_phase']}
    done=set(inherited);remaining=pending.copy();active=[];now=0;rows=[]
    while remaining or active:
        while active and active[0][0]<=now:
            _,name=heapq.heappop(active);done.add(name)
        for job in remaining.copy():
            if len(active)>=4:break
            if not set(job.get('depends_on',[]))<=done:continue
            m,e,records,mp=bindings[job['id']];tag=m['baseline_groups'][e['baseline_group']]['model_tag']
            if m['mode']=='nll':key='C_'+tag.replace('-','_').replace('.','p')+'_nll_'+('initial' if e['kind']=='initial' else records[0]['kind'])
            elif e['kind']=='reuse':key='A_selected_behavior_materialization'
            elif tag!='pythia-410m':key='C_'+tag.replace('-','_').replace('.','p')+'_behavior_final'
            elif any(r.get('role')=='extended' for r in records):key='C_pythia_410m_behavior_final'
            elif '/supplement_A_behavior/' in mp:key='A_selected_behavior'
            else:key='main_behavior_adapted'
            duration=rates[key];end=math.ceil((now+duration)/10)*10
            heapq.heappush(active,(end,job['id']));remaining.remove(job);rows.append(dict(id=job['id'],phase_rate=key,start_seconds=now,proxy_seconds=duration,expected_seconds_unchanged=job['expected_seconds'],end_poll_rounded_seconds=end))
        if active:now=active[0][0]
        elif remaining:raise ValueError('Deadlock')
    require(rows==original['actual_FIFO_job_rows'] and now==original['proxy_wall_seconds_poll_rounded'] and len(rows)==215,'Independent FIFO/rate rows differ')
    start=dt.datetime.fromisoformat(launch['launched_at_utc']);deadline=dt.datetime.fromisoformat(spec['hard_deadline_utc'].replace('Z','+00:00'))
    require(all(start+dt.timedelta(seconds=r['start_seconds']+r['expected_seconds_unchanged']+2210)<deadline for r in rows),'Strict per-job admission estimate fails')
    scenario_reviews.append(dict(name=case['name'],jobs=len(rows),phase_counts=dict(collections.Counter(r['phase_rate'] for r in rows)),FIFO_seconds=now,finish_from_actual_launch_plus300s_allowance=(start+dt.timedelta(seconds=now+300)).isoformat(),rates_used_directly_from_bound_v6=True,extra_rate_multipliers=1,all_job_admission_proxy_passes=True))

state=read(BASE/'successor/state.json');pid=launch['pid'];require(pid==3443910 and state['scheduler_pid']==pid and Path('/proc',str(pid)).exists(),'Live successor identity')
actual=[x.decode() for x in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0') if x]
require(actual==launch['argv'],'Actual successor argv differs')
require(set(state['jobs'])==seen and not state['stopping'],'Successor state cohort/stopping')
for name,s in state['jobs'].items():
    require(s['definition_sha256']==jobsha(all_jobs[name]) and not any('stop' in k.lower() for k in s),'Live definition/stop')
    if s['status'] in ['running','completed']:require(s['argv']==all_jobs[name]['argv'],'Live actual job argv')
    if s['status']=='completed':require(type(s['returncode']) is int and s['returncode']==0 and s['missing_outputs']==[],'Live completed strict rc0')
    require(s['status'] in ['pending','running','completed'],'Live failed/blocked job')
settings=handoff['settings'];require(settings['launch_margin_seconds']==2210 and settings['stop_margin_seconds']==1910 and settings['terminate_grace_seconds']==90 and settings['poll_seconds']==10 and settings['tail_seconds']==1800,'Frozen operational margins')
require(actual[-1]=='--exit-when-idle' and launch['training_jobs_present'] is False,'No science jobs/relaunch')
result=dict(schema_version=1,status='accepted_readonly_actual_handoff_review',observed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),actual=dict(successor_pid=pid,argv=actual,state_snapshot_sha256=proofs[str(BASE/'successor/state.json')]['sha256'],state_counts=dict(collections.Counter(x['status'] for x in state['jobs'].values())),active_jobs=state['active_jobs'],old_master_absent=True,all514_definitions_exactly_frozen=True,inherited299=True,new215=True,new_forward_jobs=212,alias_materialization_jobs=3),input_verification=dict(prepare=handoff['input_verification'],start=launch['input_verification'],fresh_small_paths_statted_not_reread=len(fresh)),FIFO=scenario_reviews,helper_whole_cost=dict(prepare_seconds=prepare_timer['wall_seconds'],start_seconds=start_timer['wall_seconds'],detached_scheduler_lifetime_excluded=True,nested_input_verification_not_added=True,failed_inline_parent_wall_unknown=True),limits=['No score/opaque output payload was read or rehashed by this independent reviewer.','Inherited output SHA/attestation verification is evidence from the frozen actual helper, not rerun here.','FIFO times are proxy scenarios, not actual completion or scoring authority.','Observed live successor state is preserved as an immutable byte snapshot in small_source_bytes; active state itself must not be committed.'],source_proofs=list(proofs.values()),payload_bytes_read=0,score_semantics_parsed=False,no_scheduler_actions_performed=True)
(OUT/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=result['status'],actual=result['actual'],FIFO=[{k:v for k,v in r.items() if k!='phase_counts'} for r in scenario_reviews]),ensure_ascii=False))
