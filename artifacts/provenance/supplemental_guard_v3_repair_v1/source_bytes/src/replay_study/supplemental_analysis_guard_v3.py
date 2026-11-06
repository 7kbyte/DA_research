"""Explicit independent operational repair; retained v2 bytes and plans stay valid.

Only the producer source JSON/opaque distinction and stricter scheduler/cache
completion evidence change. Statistics, frozen worker/collector inputs, cache
allowlist and the manifest-scoped first-read journal remain retained v2 policy.
"""
from .supplemental_analysis_guard_v2 import *
from . import supplemental_analysis_guard_v2 as retained


ROOT = Path(__file__).resolve().parents[2]
RETAINED_SOURCES = (
    'src/replay_study/supplemental_analysis_guard_v2.py',
    'scripts/analyze_supplemental_phase_v2.py',
)
ACTIVE_SOURCES = (
    'src/replay_study/supplemental_analysis_guard_v3.py',
    'scripts/analyze_supplemental_phase_v3.py',
)


def validate_repair(plan,repair_proof,verifier):
    repair=verifier.proof(repair_proof)
    if (repair.get('schema_version')!=3 or repair.get('operational_revision')!=3 or
            repair.get('kind')!='supplemental_analysis_guard_operational_bugfix' or
            repair.get('retained_protocol')!=plan['protocol'] or
            repair.get('scientific_or_statistics_changes') is not False or
            repair.get('score_access_authorized') is not False or
            repair.get('first_read_rule')!='exact_retained_v2_plan_manifest_namespace'):
        raise ValueError('Explicit non-authorizing operational repair required')
    for key,names in (('retained_source_sha256',RETAINED_SOURCES),('active_source_sha256',ACTIVE_SOURCES)):
        expected={str(ROOT/name) for name in names}
        sources=repair.get(key)
        if not isinstance(sources,dict) or set(sources)!=expected:raise ValueError('Operational repair source inventory differs: '+key)
        for path,value in sources.items():
            if not isinstance(value,str) or not re.fullmatch(r'[0-9a-f]{64}',value):raise ValueError('Operational repair needs explicit source SHA')
            verifier.bind(path,value)
    return repair


def strict_cache_completion_authorities(verifier):
    """Reject bool/float rc0 before retained cache-hit logic can consider it.

    Only eligible explicit checkpoint receipts are inspected, never checkpoint
    payloads. Their bytes are SHA/stat bound by the same retained verifier.
    Generic receipts still follow the original full-read fallback policy.
    """
    if verifier.cache is None:return
    for path in verifier.checkpoint_paths:
        entry=verifier.cache.entries.get(path)
        if entry is None:continue
        verifier.bind(entry['path'],entry['sha256'])
        receipt=verifier.cache._receipt(path)
        authority=receipt.get('source_authority',{})
        if authority.get('kind')=='completed_training_checkpoint':
            if type(authority.get('returncode')) is not int or authority['returncode']!=0:
                raise ValueError('Historical completed checkpoint requires integer rc0')
            if 'stop_requested_at' in authority or 'stop_requested' in authority:
                raise ValueError('Historical checkpoint authority was stop-requested')


def load_plan(plan_proof,repair_proof):
    # Preserve the retained source checks and exact fresh-prepare identity test.
    plan,verifier=retained.load_plan(plan_proof)
    validate_repair(plan,repair_proof,verifier)
    strict_cache_completion_authorities(verifier)
    return plan,verifier


def weights_sources(verifier,evidence):
    """JSON source evidence has schemas; Python producer source is opaque SHA."""
    for name in ('source_audit','source_registry','source_snapshot'):
        if not evidence.get(name):continue
        value=verifier.proof(evidence[name])
        if name=='source_registry':
            if (not isinstance(value,list) or not value or
                    any(not isinstance(row,dict) or not isinstance(row.get('id'),str) or not row['id'] for row in value) or
                    len({row['id'] for row in value})!=len(value)):
                raise ValueError('Complete-state source registry must be a nonempty unique-ID JSON list')
        else:
            if not isinstance(value,dict) or value.get('status')!='passed':raise ValueError('Complete-state JSON evidence must be passed: '+name)
            if name=='source_audit':
                for key in ('expected_runs','completed_runs'):
                    if key in value and (type(value[key]) is not int or value[key]<1):raise ValueError('Complete-state audit requires integer counts')
                if 'expected_runs' in value and value.get('completed_runs')!=value['expected_runs']:raise ValueError('Complete-state source audit is incomplete')
                if value.get('source_violations'):raise ValueError('Complete-state source audit has violations')
    if evidence.get('producer_source'):
        source=evidence['producer_source']
        if not isinstance(source,dict) or not isinstance(source.get('sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',source['sha256']):
            raise ValueError('Opaque producer source requires explicit SHA256')
        verifier.bind(source['path'],source['sha256'])


def authorize(plan_proof,authority_proof,repair_proof):
    # The old ROOT fields remain mandatory. A previous authority cannot silently
    # authorize a different operational implementation.
    plan,authority=retained.authorize(plan_proof,authority_proof)
    verifier=Verifier()
    validate_repair(plan,repair_proof,verifier)
    if authority.get('operational_repair')!=repair_proof:
        raise PermissionError('New ROOT authority must explicitly bind the operational repair')
    verifier.finish()
    return plan,authority


def inspect(plan_proof,state_path,repair_proof):
    """Fail closed on whole scheduler closure before even probing outputs."""
    started=time.perf_counter()
    plan,v=load_plan(plan_proof,repair_proof);raw=Path(state_path).read_bytes();state=json.loads(raw)
    pending=[]
    for item in plan['closure_jobs']:
        s=state['jobs'].get(item['job_id'],{});definition=item['definition']
        # schedule_jobs uses JSON sort_keys=True without pretty-print/newline.
        digest=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
        if s.get('definition_sha256')!=digest:raise ValueError('Scheduler definition differs: '+item['job_id'])
        if s.get('status') not in ('pending','running','completed'):raise ValueError('Scheduler closure has failed/unknown status: '+item['job_id'])
        if 'stop_requested_at' in s or 'stop_requested' in s:raise ValueError('Scheduler closure was stop-requested: '+item['job_id'])
        if s.get('status')=='completed':
            if type(s.get('returncode')) is not int or s['returncode']!=0:raise ValueError('Scheduler completed closure requires integer rc0: '+item['job_id'])
            if s.get('argv')!=definition['argv']:raise ValueError('Scheduler actual wrapper argv differs: '+item['job_id'])
        if s.get('status')!='completed' or s.get('missing_outputs'):pending.append(item['job_id'])
    result={'schema_version':3,'operational_revision':3,'operational_repair':repair_proof,'checked_at_utc':now(),'plan':plan_proof,'contract_id':plan['contract_id'],'counts':plan['counts'],
        'scheduler_state':str(state_path),'scheduler_state_sha256':hashlib.sha256(raw).hexdigest(),'closure_job_count':len(plan['closure_jobs']),
        'ready':not pending,'pending_job_ids':pending,'score_files_hashed':False,'score_json_parsed':False}
    if pending:
        result['verification_work']=v.accounting()
        result['score_blind_verification_wall_seconds']=time.perf_counter()-started
        return result
    # Output byte hashing and all source/checkpoint hashing happen only now.
    manifests={};attestations=[]
    def weights_evidence(checkpoint_sha,identity,evidence):
        if evidence is None:
            if checkpoint_sha is not None and identity!='checkpoint:'+checkpoint_sha:raise ValueError('Unproved model-state inference identity')
            return
        v.bind(evidence['proof'],evidence['proof_sha256']);p=read(evidence['proof']);comparison=p['comparisons'][evidence['comparison_index']]
        fp=evidence['fingerprint'];sides=[comparison['left'],comparison['right']]
        if (p.get('schema_version')!=1 or p.get('status')!='passed' or p.get('algorithm')!='sha256_sorted_key_dtype_shape_contiguous_uint8_v1' or
                not re.fullmatch(r'[0-9a-f]{64}',fp.get('sha256_model_state','')) or type(fp.get('state_elements')) is not int or fp['state_elements']<1 or
                comparison.get('bitwise_model_state_equal') is not True or any(s['model_state_fingerprint']!=fp for s in sides) or
                checkpoint_sha not in {s['checkpoint_sha256'] for s in sides} or identity!='model_state:'+fp['sha256_model_state'] or
                {s['checkpoint_sha256'] for s in sides}!={evidence['left_checkpoint_sha256'],evidence['right_checkpoint_sha256']}):
            raise ValueError('Logical/canonical endpoint lacks a true complete-state equivalence proof')
        source=p['algorithm_source'];sourcepath=Path(source['path'])
        if not sourcepath.is_absolute():sourcepath=Path(__file__).resolve().parents[2]/sourcepath
        v.bind(sourcepath,source['sha256'])
        weights_sources(v,p)
        for side in sides:v.bind(side['checkpoint'],side['checkpoint_sha256'])
    def verify_result(m,digest,job,record=None):
        output=record['output'] if record else job['output'];ap=attestation_path(output);v.bind(ap);att=read(ap)
        expected={'manifest_sha256':digest,'job_id':job['id'],'baseline_group':job['baseline_group'],'verified_frozen_inputs':True,
            'checkpoint_sha256':record['checkpoint_sha256'] if record else job.get('checkpoint_sha256'),
            'weights_identity':record['weights_identity'] if record else job.get('weights_identity')}
        if any(att.get(k)!=value for k,value in expected.items()):raise ValueError('Execution attestation identity differs: '+str(ap))
        alias=record is not None
        if att.get('inference_alias') is not alias:raise ValueError('Canonical/alias attestation differs')
        if alias:
            if (att.get('logical_record_id')!=record['id'] or att.get('inference_subprocess_wall_seconds')!=0 or
                    att.get('canonical_output')!=str(Path(job['output']).resolve()) or att.get('source_model_equivalence')!=record.get('source_model_equivalence')):
                raise ValueError('Logical alias lacks a bound zero-forward canonical proof')
            if job['kind']=='reuse' and att.get('inference_origin')!=job['inference_origin']:raise ValueError('Reuse wrapper origin differs')
        elif att.get('actual_argv')!=job['argv']:raise ValueError('Canonical scorer argv differs')
        paths={str(output)}
        if m['mode']=='behavior':paths.update(str(Path(output).with_name(Path(output).stem+'_'+name+'.jsonl')) for name in m['baseline_groups'][job['baseline_group']]['tasks'])
        if set(att.get('output_sha256',{}))!=paths:raise ValueError('Complete exact summary/task output set required')
        for p,value in att['output_sha256'].items():v.bind(p,value)
        attestations.append({'manifest_sha256':digest,'job_id':job['id'],'logical_record_id':record['id'] if record else None,'attestation':proof(ap),'output_sha256':att['output_sha256']})
    for item in plan['closure_jobs']:
        p=item['manifest']
        if p['sha256'] not in manifests:manifests[p['sha256']]=v.proof(p)
        m=manifests[p['sha256']];job=next(j for j in m['jobs'] if j['id']==item['job_id'])
        for path,value in {**m['common_input_sha256'],**job['input_sha256']}.items():v.bind(path,value)
        weights_evidence(job.get('checkpoint_sha256'),job.get('weights_identity'),job.get('source_model_equivalence'))
        if job['kind']!='reuse':verify_result(m,p['sha256'],job)
        for record in m.get('records',[]):
            if record['inference_job_id']!=job['id']:continue
            for path,value in record['input_sha256'].items():v.bind(path,value)
            weights_evidence(record['checkpoint_sha256'],record['weights_identity'],record.get('source_model_equivalence'))
            if Path(record['output']).resolve()!=Path(job['output']).resolve():verify_result(m,p['sha256'],job,record)
    v.finish();result.update(score_files_hashed=True,attestations=attestations,source_sha256=v.sources,all_source_baseline_alias_output_integrity_verified=True)
    result['verification_work']=v.accounting()
    result.update(score_blind_verification_wall_seconds=time.perf_counter()-started,timing_scope='CPU and I/O wall, including plan/source/output SHA; not GPU time or an additive child of scorer wall')
    return result
