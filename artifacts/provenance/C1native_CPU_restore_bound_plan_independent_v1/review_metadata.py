#!/usr/bin/env python3
"""Fixed 1B restore-plan metadata review; no bind/admission/payload opens."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath

ROOT=Path('/tmp/grad-research-20261106/worktree')
UTIL=ROOT/'artifacts/provenance/C2native_CPU_restore_bound_plan_independent_v1/review_metadata.py'
UTIL_SHA='e064ad995cfec5e888e4a1db4b1db651e4d46424357b008300d595747b29f30b'
assert hashlib.sha256(UTIL.read_bytes()).hexdigest()==UTIL_SHA
spec=importlib.util.spec_from_file_location('unchanged_metadata_utilities',UTIL)
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
raw,load,proof,need=H.raw,H.load,H.proof,H.demand
OUT=Path(__file__).resolve().parent
IN=ROOT/'artifacts/provenance/C_native_final_CPU_restore_actual_v1/1b'
PINS={'bound_plan.json':'820774f9b6bf9a2d54db73460a126cbcbf62e2a5be4cf6bb9d064340b52dd4fe',
      'sources.json':'7451052206dc2dd51671adfc425939730a1564634e7d2fcacc90b98cc0953e5a',
      'COMPLETE.json':'e272bd2c66e67c83a72c2c2aae67a4c540e9ef308a35cb08af580d6e76474680'}

def main():
    raw(UTIL,UTIL_SHA)
    p,s,c=[load(IN/n,PINS[n]) for n in PINS]
    need(p['evidence']==s and c['status']=='bound_metadata_only_no_execution_authority' and c['TRUE_authority_created'] is False,'Not exact preparation plan')
    for f in c['files']:need(len(raw(f['path'],f['sha256']))==f['bytes'],'Sealed metadata size mismatch')
    v={k:proof(q) for k,q in s.items() if k!='metadata_reviewer'}
    raw(s['metadata_reviewer']['path'],s['metadata_reviewer']['sha256'])
    for k in ('runner','checker','wrapper'):raw(c[k]['path'],c[k]['sha256'])
    need(p['runner']==c['runner'] and p['checker']==c['checker'],'Runtime sources mismatch')
    need(p['runner']['sha256']=='bfc748c776f18dff86c378c5c55ca72dbc63995cfc40ddfb96eb919486610db1','Original runner differs')
    bt=proof(c['whole_bind_timing']);need(type(bt['returncode']) is int and bt['returncode']==0,'Bind not strict rc0')
    inv=load(IN/'bind_invocation.json')
    need(inv['argv']==['/usr/bin/python',p['runner']['path'],'bind','--model','pythia-1b','--sources',str(IN/'sources.json'),
          '--sources-sha256',PINS['sources.json'],'--output',str(IN/'bound_plan.json')],'Actual bind argv differs')
    m,i,a,pub=v['manifest'],v['identity'],v['root_acceptance'],v['publication']
    need(m['status']=='complete_phase_all_final_roundtrips_verified' and v['completion']['status']=='passed' and
         v['completion']['fullphase_source_gate_passed'] is True,'Whole phase incomplete')
    need((m['logical_final_endpoints'],m['unique_weights'],m['alias_count'])==(24,21,3),'Not24/21/3')
    need(a['status']=='accepted' and a['scope']=='original_Cfinal_native_package_publication' and
         s['root_acceptance']['sha256']=='58f91602dc3e933ef8b63045c13e11ecf13e07f3226f6558dd09eedd5e515fd3','ROOT acceptance differs')
    for k in ('completion','manifest','metadata_review','metadata_reviewer','publication'):need(a[k]==s[k],'ROOT reference mismatch:'+k)
    need(v['ready']['ready'] is True and v['ready']['manifest']==s['identity'],'READY identity mismatch')
    g=m['fullphase_gate'];need(g['status']=='complete_source_gate_passed','Source gate failed')
    ar=next(q for q in g['sources'] if q['path'].endswith('/extended_v2_pythia_1b_m32_formal/audit.json'))
    audit=proof(ar)
    need(audit['status']=='passed' and audit['expected_runs']==audit['completed_runs']==24 and audit['source_violations']==[],'Source audit failed')
    need(audit['source_state_complete_gate']==dict(ready=True,expected_runs=24,completed_runs=24,missing_runs=[],failed_runs=[]),'Source strict state incomplete')
    er=next(q for q in g['sources'] if q['path'].endswith('/final_model_equivalence_proofs.json'))
    eq=proof(er);pairs=eq['comparisons']
    need(len(pairs)==3 and {q['seed'] for q in pairs}=={11,22,33} and all(q['endpoint']=='final' and q['bitwise_model_state_equal'] is True for q in pairs),'Not three proven final aliases')
    # Python producer is an opaque source pin, never a JSON proof.
    for k in ('producer_source','algorithm_source'):
        if k in eq:raw(eq[k]['path'],eq[k]['sha256'])
    proof(m['freeze']);ps=m['phase_source']
    for k in ('active_registry','snapshot','protocol','queue'):
        q=Path(ps[k]);raw(q if q.is_absolute() else ROOT/q,ps[k+'_sha256'])
    finals={q['run_id']:q for q in i['records'] if q['kind']=='final'}
    rows=m['logical_runs'];byid={q['run_id']:q for q in rows}
    need(g['identity_manifest']['records']==finals and len(rows)==len(finals)==24 and set(byid)==set(finals),'Final24 closure differs')
    need({(q['group'],q['seed']) for q in rows}=={(z,n) for z in ('B0','B1','B2','B3','B4','D','M','C') for n in (11,22,33)},'Wrong full factorial')
    aliases=[];strict=[]
    for q in rows:
        d,r,f=q['definition'],q['completed_scheduler_record'],finals[q['run_id']]
        h=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
        need(q['definition_sha256']==r['definition_sha256']==h and r['status']=='completed' and type(r['returncode']) is int and
             r['returncode']==0 and r['missing_outputs']==[] and not any('stop' in k.lower() for k in r) and r['argv']==d['argv'],'Non-strict original training state')
        need(q['source_checkpoint']==f['checkpoint'] and q['source_checkpoint_sha256']==f['checkpoint_sha256'] and
             q['source_stat']['bytes']==f['checkpoint_bytes'] and q['source_stat']['mtime_ns']==f['checkpoint_mtime_ns'],'CP frozen identity mismatch')
        for pk,hk in [('config_path','config_sha256'),('summary_path','summary_sha256')]:
            raw(f[pk],q[hk]);need(f['input_sha256'][f[pk]]==q[hk],'Current/frozen config-summary mismatch')
        need(q['actual_exposure']['target_prediction_tokens']==100000000 and q['actual_exposure']['replay_prediction_tokens']==(0 if q['group']=='B0' else 5000000),'Wrong final exposure')
        if q['group']=='D':
            b=byid[q['alias_of']];pair=next(z for z in pairs if z['seed']==q['seed'])
            need(b['group']=='B4' and b['seed']==q['seed'] and q['alias_evidence']['pair']==pair and
                 g['decisions'][str(q['seed'])]['comparison']==pair and g['decisions'][str(q['seed'])]['equal'] is True,'D proof mismatch')
            sides={z['run_id']:z for z in (pair['left'],pair['right'])}
            for z in (q,b):
                t=sides[z['run_id']]
                need(t['checkpoint']==z['source_checkpoint'] and t['checkpoint_sha256']==z['source_checkpoint_sha256'] and
                     t['model_state_fingerprint']==z['model_fingerprint'] and t['config_sha256']==z['config_sha256'] and
                     t['summary_sha256']==z['summary_sha256'],'D/B4 full source closure mismatch')
            need(q['model_fingerprint']==b['model_fingerprint'],'D/B4 FP differs')
            aliases.append(dict(run_id=q['run_id'],alias_of=b['run_id'],pair_id=pair['id']))
        else:need('alias_of' not in q,'Invented nonD alias')
        strict.append(dict(run_id=q['run_id'],definition_sha256=h,strict_integer_rc0_no_stop_actual_argv=True))
    q=byid[p['run_id']];fp={'sha256_model_state':'da1fe7db068d1111f973920ac59fb6c0ad70914c288cfdef434db9d7bef42d82','state_elements':1011781632}
    need(p['selected']==q and q['group']=='B4' and q['seed']==11 and p['model']=='pythia-1b','Fixed selected run mismatch')
    need(q['model_fingerprint']==fp and all(q[k]['model_fingerprint']==fp for k in ('encode','decode','bytewise_verification')),'Full FP differs')
    need(p['tensor_count']==q['decode']['tensor_raw_hashes_verified']==q['bytewise_verification']['tensor_count']==196 and
         q['all_nonweight_metadata_roundtrip_verified'] is True and q['bytewise_verification']['all_tensor_bytes_equal'] is True,'Not full196 roundtrip')
    need(q['actual_exposure']==dict(checkpoint_step=3058,target_prediction_tokens=100000000,replay_prediction_tokens=5000000,
         full_training_target_prediction_tokens=100000000,full_training_replay_prediction_tokens=5000000),'Not fixed final100M/5M')
    members=p['members'];native=m['base_dependency']['files']
    need(len(members)==12 and len(native)==6 and m['base_dependency']['kind']=='native_base_config_tokenizer_readme_included','Wrong member/native scope')
    expected={z['member']:dict(sha256=z['sha256'],bytes=z['bytes']) for z in native}
    need({Path(z['member']).name for z in native}=={'model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md'} and
         all(str(PurePosixPath(z['member']).parent)=='base/pythia-1b-final' for z in native),'Wrong native6 model')
    expected['extended_C_final_manifest.json']=dict(sha256=s['manifest']['sha256'],bytes=Path(s['manifest']['path']).stat().st_size)
    expected[q['archive_member']]=dict(sha256=q['archive_sha256'],bytes=q['archive_stat']['bytes'])
    expected[q['metadata_member']]=dict(sha256=q['metadata_sha256'],bytes=q['metadata_stat']['bytes'])
    for name,sha in {'tools/scripts/archive_model_weights.py':'28c5841cae8e7fd87d8cb62e7f756dad40ce15cb558710c31e48dfb45fde668a',
       'tools/src/replay_study/weight_archive.py':'3c2bcf9e38773ae4f69c27f85f342356dbefcb53d817be4321da6e9c4e550e1a',
       'tools/src/replay_study/__init__.py':'b2dce8cc5ea80a67a90b1798232163a7684874487b5ea4102e60851c7017f520'}.items():
        raw(ROOT/name.removeprefix('tools/'),sha);expected[name]=dict(sha256=sha,bytes=None)
    need(expected==members,'Selected12 SHA/size mismatch')
    for n in members:
        x=PurePosixPath(n);need(not x.is_absolute() and '..' not in x.parts and str(x)==n and '\\' not in n,'Unsafe member')
    need(p['native_base']=='base/pythia-1b-final/model.safetensors' and members[p['native_base']]['sha256']==q['encode']['base_sha256']==m['base_dependency']['base_sha256'],'Base SHA differs')
    ts=H.sig(p['archive']);need(ts==p['archive_stat'] and all(ts[k]==pub['destination_stat'][k]==a['durable_archive_current_stat'][k] for k in ts),'Current durable TAR stat differs')
    need(pub['status']=='passed' and pub['source_sha256']==pub['destination_sha256']==v['completion']['tar_sha256']==p['historical_full_TAR_sha256'],'Historical fullTAR SHA mismatch')
    need(all(pub[k] is True for k in ('file_fsync_after_copy','prefix_sha_verified_before_truncate','file_fsync_after_truncate','atomic_destination_commit_noreplace','parent_directory_fsync')),'Durable publication flags differ')
    pt=proof(a['publication_whole_child']);need(type(pt['returncode']) is int and pt['returncode']==0,'Publisher not strict rc0')
    known=sum(z['bytes'] or 0 for z in members.values());rb=fp['state_elements']*4;rp=p['resource_policy']
    need(rb==p['model_raw_bytes']==4047126528 and rp==dict(required_scratch_bytes=known+rb+(2<<30),required_reclaim_aware_memory_bytes=4*rb+(4<<30),
         cpu_threads=2,max_wall_seconds=600,deadline_utc='2026-11-07T02:56:32+00:00',admission_margin_seconds=300),'Resource formula differs')
    need(rp['required_scratch_bytes']==11369697919 and rp['required_reclaim_aware_memory_bytes']==20483473408,'Resource totals differ')
    latest=dt.datetime.fromisoformat(rp['deadline_utc'])-dt.timedelta(seconds=900)
    need(latest.isoformat()=='2026-11-07T02:41:32+00:00' and not Path(p['qa_directory']).exists(),'Admission boundary/absent QA differs')
    receipt=dict(status='accepted_metadata_only_bound_plan_not_execution_authority',at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        bound_plan=c['bound_plan'],sources=c['sources'],source_COMPLETE=dict(path=str(IN/'COMPLETE.json'),sha256=PINS['COMPLETE.json']),
        current_durable_TAR_stat=ts,historical_full_TAR_SHA=p['historical_full_TAR_sha256'],fullphase_logical=24,unique_weights=21,
        actual_proven_aliases=aliases,strict_original_training_records=strict,source_audit=ar,source_equivalence_proof=er,
        fixed_run_id=p['run_id'],actual_exposure=q['actual_exposure'],complete_model_fingerprint=fp,tensor_count=196,native_files=6,selected_members=members,
        resource_policy=rp,strict_start_before_utc=latest.isoformat(),execution_cap_seconds=600,cleanup_seconds_up_to=10,
        teardown_admission_allowance_seconds=12,admission_margin_covers_cleanup=True,actual_resource_admission_executed=False,
        actual_bind_whole_timing=bt,restoration_order='1B first; actual fresh admission and exact ROOT authority still required; C2.8 later',
        large_payload_bytes_read=0,score_output_bytes_read=0,score_semantics_read=False,tests_run=False,bind_or_restore_or_hint_executed=False,
        TRUE_authority_created=False,QA_created=False,metadata_sources=H.reads,unique_metadata_bytes=sum(z['bytes'] for z in H.reads.values()),
        scope='Explicit small metadata/source bytes and durable TAR stat only; summary bytes only hashed, never parsed; no payload traversal',
        source=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),utility_source=dict(path=str(UTIL),sha256=UTIL_SHA))
    H.persist(OUT/'receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],receipt_sha256=hashlib.sha256((OUT/'receipt.json').read_bytes()).hexdigest(),unique_metadata_files=len(H.reads)),indent=2))

if __name__=='__main__':main()
