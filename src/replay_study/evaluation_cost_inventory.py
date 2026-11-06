"""Score-blind accounting from explicit v2 plans and execution attestations."""
import hashlib
import math
from pathlib import Path

CATEGORIES={'primary','supplement','pilot','provisional','diagnostic'}


def nonnegative(value):
    return value if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0 else None


def hash_output(path,reader):
    """Optional byte integrity only; scores are never JSON-deserialized."""
    path=Path(path).resolve();key=str(path)
    if not path.exists():reader.sources[key]={'exists':False,'read_purpose':'SHA_only'};return None
    before=path.stat();digest=hashlib.sha256();size=0
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):digest.update(block);size+=len(block)
    after=path.stat();value=digest.hexdigest()
    reader.sources[key]={'exists':True,'sha256':value,'bytes_read':size,'mtime_ns':after.st_mtime_ns,
                         'read_purpose':'SHA_only_no_score_deserialization',
                         'changed_while_reading':before.st_mtime_ns!=after.st_mtime_ns or before.st_size!=after.st_size}
    return value


def inventory_evaluations(spec,reader):
    plans={};jobs={};manifest_status=[];references=[]
    for item in spec.get('evaluation_manifests',[]):
        category=item['category']
        if category not in CATEGORIES:raise ValueError('Unsupported evaluation category')
        path=str(Path(item['path']).resolve());manifest=reader.read(path)
        if manifest is None:
            manifest_status.append({'manifest':path,'category':category,'status':'missing_or_invalid_manifest'});continue
        expected=item.get('sha256')
        if not expected or reader.sources[path]['sha256']!=expected:raise ValueError('Explicit evaluation manifest SHA mismatch')
        if manifest.get('schema_version')!=2 or manifest.get('mode') not in ('nll','behavior'):raise ValueError('A v2 evaluation manifest is required')
        state=reader.read(item['scheduler_state']) if item.get('scheduler_state') else None
        plan=plans.setdefault(expected,{'manifest':manifest,'path':path,'items':[]});plan['items'].append(item)
        manifest_status.append({'manifest':path,'manifest_sha256':expected,'category':category,'status':'verified_manifest','mode':manifest['mode'],'jobs':len(manifest['jobs'])})
        for job in manifest['jobs']:
            key=(expected,job['id']);out=Path(job['output']);override=item.get('execution_attestations',{}).get(job['id'],{})
            # Reuse keeps job.output at the old canonical result. Its own new
            # execution proof is emitted beside a materialized logical alias.
            logical=next((r for r in manifest.get('records',[]) if r['inference_job_id']==job['id'] and Path(r['output']).resolve()!=out.resolve()),None) if job['kind']=='reuse' else None
            if job['kind']=='reuse' and logical is None:raise ValueError('Reuse requires a materialized logical endpoint for its own execution proof')
            if logical:out=Path(logical['output'])
            if isinstance(override,str):override={'path':override}
            ap=str(Path(override.get('path',out.with_name(out.stem+'_execution.json'))).resolve())
            if key in jobs:
                if jobs[key]['job']!=job or jobs[key]['attestation_path']!=ap or jobs[key]['expected_attestation_SHA']!=override.get('sha256'):
                    raise ValueError('Conflicting definitions for one manifest/inference ID')
                jobs[key]['categories'].add(category);jobs[key]['verify_outputs']|=item.get('verify_output_hashes',False);continue
            jobs[key]={'job':job,'manifest':manifest,'manifest_path':path,'categories':{category},'attestation_path':ap,
                       'expected_attestation_SHA':override.get('sha256'),'verify_outputs':item.get('verify_output_hashes',False),
                       'proof_logical':logical,
                       'state':(state or {}).get('jobs',{}).get(job['id'],{}),'scheduler_state':item.get('scheduler_state')}
        for key,group in manifest.get('baseline_groups',{}).items():
            origin=group.get('initial_origin') or {'manifest_sha256':expected,'job_id':group['initial_job_id'],'output':group['initial_output']}
            references.append({'reference_kind':'initial_baseline','manifest_sha256':expected,'baseline_group':key,
                               'canonical_manifest_sha256':origin['manifest_sha256'],'canonical_inference_id':origin['job_id'],
                               'canonical_output':origin['output'],'is_inference_reuse':bool(group.get('initial_origin')),
                               'extra_model_forward_calls':0 if group.get('initial_origin') else None})

    def attestation(manifest_sha,job,path,expected_sha=None,logical=None,verify_outputs=False):
        a=reader.read(path)
        if a is None:return None
        if expected_sha and reader.sources[str(Path(path).resolve())]['sha256']!=expected_sha:raise ValueError('Execution attestation SHA mismatch')
        if a.get('manifest_sha256')!=manifest_sha or a.get('job_id')!=job['id'] or a.get('verified_frozen_inputs') is not True:
            raise ValueError('Attestation does not bind frozen manifest/inference')
        if a.get('checkpoint_sha256')!=(logical.get('checkpoint_sha256') if logical else job.get('checkpoint_sha256')):
            raise ValueError('Attestation checkpoint differs from endpoint')
        output=Path(logical['output'] if logical else job['output']);expected={str(output)};manifest=plans[manifest_sha]['manifest']
        if manifest['mode']=='behavior':
            expected.update(str(output.with_name(output.stem+'_'+name+'.jsonl')) for name in manifest['baseline_groups'][job['baseline_group']]['tasks'])
        if set(a.get('output_sha256',{}))!=expected:raise ValueError('Attested output set differs from manifest')
        if verify_outputs:
            for p,digest in a['output_sha256'].items():
                if hash_output(p,reader)!=digest:raise ValueError('Attested output SHA mismatch')
        return a

    executions=[];canonical={}
    for key,source in jobs.items():
        manifest_sha,jid=key;job=source['job'];state=source['state'];kind=job['kind']
        if kind not in ('initial','adapted','reuse'):raise ValueError('Unsupported inference job kind')
        a=attestation(manifest_sha,job,source['attestation_path'],source['expected_attestation_SHA'],logical=source['proof_logical'],verify_outputs=source['verify_outputs'])
        categories=sorted(source['categories']);category=categories[0] if len(categories)==1 else 'classification_conflict'
        origin=job.get('inference_origin') if kind=='reuse' else None;ck=(origin['manifest_sha256'],origin['job_id']) if origin else key
        if origin and ck==key:raise ValueError('Inference cannot reuse itself')
        if origin and ck in jobs and jobs[ck]['job']['kind']=='reuse':raise ValueError('Reuse chains are not canonical inference identities')
        if a and ((kind!='reuse' and a.get('inference_alias') is True) or (kind=='reuse' and (a.get('inference_alias') is not True or a.get('inference_origin')!=origin or a.get('inference_subprocess_wall_seconds')!=0))):
            raise ValueError('Canonical/reuse attestation has inconsistent alias timing')
        verify=nonnegative(a.get('verification_wall_seconds')) if a else None;scorer=nonnegative(a.get('inference_subprocess_wall_seconds')) if a else None
        elapsed=nonnegative(state.get('elapsed_seconds'));residual=elapsed-verify-scorer if elapsed is not None and verify is not None and scorer is not None else None
        failed=state.get('status') in ('failed','terminated','killed') or state.get('returncode') not in (None,0)
        status=('attested_scorer_scheduler_failed' if failed else 'attested') if a else 'failed_without_attested_timing' if failed else 'missing_attestation_'+str(state.get('status','state_unknown'))
        group=source['manifest'].get('baseline_groups',{}).get(job.get('baseline_group'),{});policy=group.get('inference_policy',{})
        logical=[r for r in source['manifest'].get('records',[]) if r['inference_job_id']==jid]
        row={'manifest':source['manifest_path'],'manifest_sha256':manifest_sha,'inference_job_id':jid,
             'canonical_manifest_sha256':ck[0],'canonical_inference_id':ck[1],'kind':kind,'mode':source['manifest']['mode'],
             'phase':job.get('phase',source['manifest'].get('phase')),'category':category,'category_candidates':categories,'status':status,
             'model_tag':group.get('model_tag',job.get('model_tag')),'model':policy.get('model'),'revision':policy.get('revision'),
             'batch_size':policy.get('batch_size'),'precision':policy.get('precision'),
             'logical_phases':sorted({r.get('phase_name',r.get('phase')) for r in logical if r.get('phase_name',r.get('phase'))}),
             'logical_roles':sorted({r['role'] for r in logical if r.get('role')}),
             'data_families':sorted({r['data_family'] for r in logical if r.get('data_family')}),
             'manifest_work_per_inference':group.get('work_per_inference'),
             'scheduler_status':state.get('status'),'scheduler_returncode':state.get('returncode'),'scheduler_state':source['scheduler_state'],
             'execution_attestation':source['attestation_path'],'attestation_present':bool(a),
             'execution_attestation_sha256':reader.sources[source['attestation_path']].get('sha256'),
             'verification_proof_source':'one materialized logical alias for this reuse wrapper' if kind=='reuse' else 'canonical execution attestation',
             'recorded_additional_model_inference':bool(a and kind!='reuse'),
             'frozen_input_SHA_and_manifest_verification_wall_seconds':verify,'scorer_subprocess_wall_seconds':scorer,
             'missing_recorded_timing_fields':[name for name,value in (('verification_wall_seconds',verify),('inference_subprocess_wall_seconds',scorer)) if value is None],
             'known_disjoint_recorded_wall_seconds':sum(v for v in (verify,scorer) if v is not None) if verify is not None or scorer is not None else None,
             'preflight_environment_wall_seconds':None,'SHA_wall_separately_measured_from_manifest_checks':False,
             'scheduler_elapsed_seconds_nonadditive':elapsed,'scheduler_unclassified_residual_wall_seconds_nonadditive':residual,
             'residual_note':'Uninstrumented environment/preflight, dispatch, alias work and polling; neither pure GPU nor isolated SHA',
             'result_internal_timers_parsed_or_added':False,'output_integrity_verified_by_inventory':bool(a and source['verify_outputs']),
             'timing_scope':'Last attested path only; failed/retried attempts without separate attestations are not reconstructed'}
        executions.append(row)
        if origin:canonical.setdefault(ck,{'canonical_manifest_sha256':ck[0],'canonical_inference_id':ck[1],'status':'external_canonical_not_explicitly_inventoried','category':'unregistered','additional_forward_jobs_observed':None,'scorer_subprocess_wall_seconds':None})
    for row in executions:
        if row['kind']!='reuse':canonical[(row['manifest_sha256'],row['inference_job_id'])]={**row,'additional_forward_jobs_observed':1 if row['attestation_present'] else None}
    for ref in references:
        key=(ref['canonical_manifest_sha256'],ref['canonical_inference_id'])
        canonical.setdefault(key,{'canonical_manifest_sha256':key[0],'canonical_inference_id':key[1],
                                  'status':'external_canonical_not_explicitly_inventoried','category':'unregistered',
                                  'additional_forward_jobs_observed':None,'scorer_subprocess_wall_seconds':None})
    for manifest_sha,plan in plans.items():
        lookup={j['id']:j for j in plan['manifest']['jobs']}
        for logical in plan['manifest'].get('records',[]):
            job=lookup[logical['inference_job_id']];origin=job.get('inference_origin');ck=(origin['manifest_sha256'],origin['job_id']) if origin else (manifest_sha,job['id'])
            alias=Path(logical['output']).resolve()!=Path(job['output']).resolve()
            ref={'reference_kind':'logical_endpoint','manifest_sha256':manifest_sha,'logical_record_id':logical['id'],
                 'run_id':logical.get('run_id'),'actual_run_id':logical.get('actual_run_id'),'canonical_manifest_sha256':ck[0],'canonical_inference_id':ck[1],
                 'checkpoint_sha256':logical.get('checkpoint_sha256'),'source_model_equivalence':logical.get('source_model_equivalence'),
                 'is_inference_reuse':alias or bool(origin),'extra_model_forward_calls':0 if alias or origin else None,
                 'alias_materialization_wall_seconds':None,'alias_materialization_wall_not_instrumented':True}
            if alias:
                out=Path(logical['output']);ap=str(out.with_name(out.stem+'_execution.json').resolve())
                a=attestation(manifest_sha,job,ap,logical=logical,verify_outputs=any(i.get('verify_output_hashes',False) for i in plan['items']))
                if a and (a.get('inference_alias') is not True or a.get('logical_record_id')!=logical['id'] or a.get('canonical_output')!=str(Path(job['output']).resolve()) or a.get('inference_subprocess_wall_seconds')!=0 or a.get('source_model_equivalence')!=logical.get('source_model_equivalence')):
                    raise ValueError('Alias lacks matching zero-forward attestation')
                ref.update(alias_attestation=ap,alias_status='materialized_zero_forward' if a else 'missing_alias_attestation',
                           inherited_verification_seconds_not_added=a.get('verification_wall_seconds') if a else None,
                           inherited_canonical_scorer_seconds_not_added=a.get('canonical_inference_subprocess_wall_seconds') if a else None)
            references.append(ref)
    totals=[]
    for category in sorted({r['category'] for r in executions}):
        rows=[r for r in executions if r['category']==category]
        totals.append({'category':category,'unique_manifest_job_execution_records':len(rows),'attested_new_forward_jobs':sum(r['recorded_additional_model_inference'] for r in rows),
                       'attested_reuse_jobs':sum(r['attestation_present'] and r['kind']=='reuse' for r in rows),
                       'missing_or_failed_unattested_jobs':sum(not r['attestation_present'] for r in rows),
                       'failed_scheduler_jobs':sum(r['scheduler_status'] in ('failed','terminated','killed') or r['scheduler_returncode'] not in (None,0) for r in rows),
                       'SHA_and_manifest_verification_seconds_known_sum':sum(r['frozen_input_SHA_and_manifest_verification_wall_seconds'] or 0 for r in rows),
                       'scorer_subprocess_seconds_known_sum':sum(r['scorer_subprocess_wall_seconds'] or 0 for r in rows),
                       'verification_timer_missing_records':sum(r['frozen_input_SHA_and_manifest_verification_wall_seconds'] is None for r in rows),
                       'scorer_timer_missing_records':sum(r['scorer_subprocess_wall_seconds'] is None for r in rows),
                       'known_disjoint_recorded_wall_seconds':sum(r['known_disjoint_recorded_wall_seconds'] or 0 for r in rows),
                       'result_internal_timers_added':False,'scheduler_elapsed_added_to_components':False,'wall_is_pure_GPU_time':False})
    return {'evaluation_manifest_status':manifest_status,'evaluation_executions':executions,'evaluation_canonical_inferences':list(canonical.values()),
            'evaluation_references':references,'evaluation_category_totals':totals}
