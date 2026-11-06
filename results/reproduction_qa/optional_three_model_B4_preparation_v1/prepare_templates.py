"""One-shot CPU creation of FALSE templates; no actual queue or launch."""
import hashlib, json, resource, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARCH = Path('/tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/research')
OUT = Path(__file__).parent
NS = 'archive0732_qa_Cthree_B4_seed11_v1'
SCRATCH = Path('/tmp/grad-research-20261106/reproduction_qa/optional_Cthree_B4_seed11_v1/scratch')
start = time.perf_counter(); usage = resource.getrusage(resource.RUSAGE_SELF)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def proof(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)

def read(path): return json.loads(Path(path).read_text())

def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

pre = ROOT / 'docs/optional_three_model_B4_archive_QA_ROOT_predeclaration_v1.json'
assert sha(pre) == '23923ef16feda2dd9e4b3c3c0a52528e61a0c70e8e4cb9488dcfc0958bcf583f'
decl = read(pre); manifest = ARCH / 'PACKAGE_MANIFEST.json'
package = {r['path']: r for r in read(manifest)['files']}
extraction = ARCH.parent / 'extraction_receipt.json'
assert sha(manifest) == read(extraction)['package_manifest']['sha256']
tool_receipt_path = OUT / 'assessment_tool_extraction_1415.json'
tool_receipt = read(tool_receipt_path)
verifier = next(p for p in tool_receipt['selected_files'] if p['member'].endswith('verify_reproduction_qa_postrun.py'))
verifier = {k: verifier[k] for k in ['path', 'sha256', 'bytes']}
assert sha(verifier['path']) == verifier['sha256'] == '5417fc0747d94feae828e0e53133edf66745b2fb60885cc591f84f2feceea7e2'
prior1 = ROOT / 'results/reproduction_qa/optional_C1b_original_best_identity_v1/worker/proof.json'
prior2 = ROOT / 'results/reproduction_qa/optional_C2p8b_original_best_identity_v1/worker/proof.json'
bests = {'pythia-1b': (prior1, read(prior1)), 'pythia-2.8b': (prior2, read(prior2))}
assert all(j['status'] == 'passed_original_best_complete_model_state_and_fresh_file_SHA' for p, j in bests.values())
helper_names = ['scripts/reproduce_experiments.py', 'scripts/schedule_jobs.py', 'scripts/audit_formal_runs.py', 'src/replay_study/weight_archive.py', 'src/replay_study/__init__.py', 'artifacts/assets/model_provenance.json', 'artifacts/assets/extended_model_provenance.json', 'artifacts/assets/extended_model_download_manifest.json']
helpers = []
for name in helper_names:
    assert sha(ARCH / name) == package[name]['sha256']
    helpers.append(dict(member=name, **proof(ARCH / name)))
coverage, checked, conditions, runs, jobs = [], {}, [], [], []
sources = [proof(pre), proof(manifest), proof(extraction), proof(tool_receipt_path), verifier]
policy_keys = list(read(ROOT / 'docs/optional_C2p8b_B4_seed11_QA_ROOT_FALSE_TEMPLATE.json')['required_original_execution_provenance'])
for c in decl['conditions']:
    model, old, row = c['model'], c['original_run_id'], c['frozen_registry_row']
    prefix = old.split('_B4_s11_')[0]
    snapname = f'artifacts/provenance/{prefix}_input_snapshot.json'
    snap = read(ARCH / snapname)
    required = {**snap['code'], **snap['data']}
    for name in [snapname, f'configs/{prefix}_active_registry.json', f'configs/{prefix}/{old}.json', f'docs/{prefix}_protocol.json']:
        required[name] = {'sha256': sha(ROOT / name), 'bytes': (ROOT / name).stat().st_size}
    for name, expected in required.items():
        assert name in package and all(package[name][k] == expected[k] for k in ['bytes', 'sha256']), name
        p = ARCH / name
        assert p.is_file() and not p.is_symlink() and p.stat().st_size == expected['bytes']
        if name not in checked:
            checked[name] = proof(p)
            assert checked[name]['sha256'] == expected['sha256'], name
    coverage.append(dict(model=model, frozen_code_count=len(snap['code']), frozen_data_count=len(snap['data']), all_required_payloads_current_fresh_SHA_match=True, required_members=list(required), input_snapshot=proof(ARCH / snapname)))
    dp = Path('/tmp/grad-research-20261106/delivery/extended_C_final_v1') / model / 'records' / (old + '.json')
    draft = read(dp)
    assert draft['status'] == 'verified_draft_not_deliverable' and draft['run_id'] == old and draft['group'] == 'B4' and draft['seed'] == 11
    assert draft['config_sha256'] == row['config_sha256'] and draft['all_nonweight_metadata_roundtrip_verified'] and not draft['gpu_used']
    summary = read(row['summary_path'])
    policy = {k: summary['provenance'][k] for k in policy_keys}
    full = {k: summary['counters'][k] for k in ['step', 'target_prediction_tokens', 'replay_prediction_tokens']}
    new = NS + '__' + old; output = SCRATCH / 'runs' / new
    cfg = ARCH / 'configs/reproductions' / NS / (new + '.json')
    copy = ARCH / 'results/reproductions' / NS / (new + '_summary.json')
    final = dict(kind='final', checkpoint=str(output / 'checkpoints/latest.pt'), counters=full, expected_fingerprint=draft['model_fingerprint'], checkpoint_kind_present=False, checkpoint_kind=None, original_checkpoint=draft['source_checkpoint'], original_checkpoint_sha256=draft['source_checkpoint_sha256'], identity_evidence=proof(dp))
    best = dict(kind='best', checkpoint=str(output / 'checkpoints/checkpoint_best.pt'), counters=None, expected_fingerprint=None, checkpoint_kind_present=True, checkpoint_kind='model_only_best_target_validation', selection=None, original_selection=None, original_checkpoint=str(Path(draft['source_checkpoint']).with_name('checkpoint_best.pt')), original_checkpoint_sha256=None, accepted_metadata=None, status='pending_future_source_bound_full_phase_D_B4_best_proof_no_new_410M_CP_read')
    if model in bests:
        bp, bj = bests[model]
        best.update(counters=bj['actual_best_counters'], expected_fingerprint=bj['model_state_fingerprint'], original_checkpoint_sha256=bj['source_checkpoint_sha256'], original_selection_opaque_sha256=bj['selection']['opaque_sha256'], identity_evidence=proof(bp), status='original_identity_verified_future_exact_metadata_and_durable_dependencies_pending')
        sources.append(proof(bp))
    runs.append(dict(run_id=new, original_run_id=old, model=model, group='B4', config=dict(path=str(cfg), sha256=None), output_dir=str(output), summary_copy=str(copy), internal_summary=str(output / 'summary.json'), full_counters=full, required_provenance=policy, endpoints=[final, best]))
    jobs.append(dict(id=new, argv=['/usr/bin/python', str(ARCH / 'scripts/make_experiments.py'), 'run', '--config', str(cfg), '--summary-copy', str(copy)], env={'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HOME': str(SCRATCH / 'cache/huggingface')}, expected_seconds=c['future_runtime_queue_expected_seconds'], outputs=[str(output / 'summary.json'), str(copy)]))
    conditions.append(dict(model=model, original_run_id=old, new_run_id=new, source_registry=c['frozen_registry'], source_config=dict(path=row['config_path'], sha256=row['config_sha256']), input_snapshot=proof(ROOT / snapname), final_identity_source=proof(dp), original_parent_wall_seconds=c['observed_original_training_parent_wall_seconds'], runtime_expected_seconds=jobs[-1]['expected_seconds'], full_counters=full, base_revision=draft['model_revision'], base_files_required=snap['model_files'], durable_base_package_status='pending_future_ROOT_accepted_actual_C_package_and_offline_base_extraction' if model != 'pythia-410m' else 'primary_durable_base_closure_exists_but_future_isolated_extraction_and_fresh_relocation_pending'))
    sources += [proof(Path(row['config_path'])), proof(Path(row['summary_path'])), proof(dp), proof(ROOT / snapname)]
expected = dict(schema_version=1, status='PENDING_IDENTITY_TEMPLATE_NOT_ACCEPTED_NOT_LAUNCH_AUTHORITY', root_launch_authority_granted=False, pending_fields_must_be_filled_from_real_SHA_bound_evidence=True, namespace=NS, plan=dict(path=None, sha256=None), registry=dict(path=str(ARCH / 'configs/reproductions' / (NS + '_registry.json')), sha256=None), queue=dict(path=str(ARCH / 'configs/reproductions' / (NS + '_runtime_queue.json')), sha256=None), scheduler_state=str(ARCH / 'logs/reproductions' / (NS + '_job_state.json')), planned_gpus=[], archived_workspace=str(ARCH), independent_scratch=str(SCRATCH), source_proofs=sources, archived_fingerprint_algorithm=proof(ARCH / 'scripts/audit_formal_runs.py'), expected_endpoint_count=6, runs=runs, not_compatible_with_verify_until_pending_dependencies_frozen=True, verifier=verifier, planning_CPU_postcheck_allowance_seconds=600, planning_CPU_postcheck_allowance_is_measured=False, selection_values_emitted=False)
ep = ROOT / 'docs/optional_three_model_B4_archive_QA_expected_identity_TEMPLATE.json'
write(ep, expected)
auth = dict(schema_version=1, status='FALSE_TEMPLATE_CPU_PREPARATION_ONLY_NOT_READY', root_authorized=False, GPU_execution_allowed=False, training_started=False, queues_materialized_or_enqueued=False, predeclaration=proof(pre), namespace=NS, archived_workspace=str(ARCH), independent_scratch=str(SCRATCH), conditions=conditions, expected_identity_TEMPLATE=proof(ep), planned_source_subset_registry=str(ARCH / 'configs/reproductions' / (NS + '_original_source_registry.json')), source_subset_rule='Future copy exactly three declared rows from SHA-bound archived registries; preserve original rows and source hashes.', runtime_jobs_TEMPLATE=jobs, generic_helper_queue_expected_seconds=600, runtime_delta_rule='Preserve helper600 queue. Independent runtime queue must differ ONLY in expected_seconds=2700/3600/2340 by exact ID. Bind both SHAs and compare every other key.', scheduler_argv_TEMPLATE=['/usr/bin/python', str(ARCH / 'scripts/schedule_jobs.py'), '--queue', expected['queue']['path'], '--state', expected['scheduler_state'], '--gpus', '<ROOT_ASSIGNED_THREE_DISTINCT_FREE_H100_IDS>', '--deadline', decl['scheduler_deadline_utc'], '--exit-when-idle', '--launch-margin-seconds', '120', '--stop-margin-seconds', '120'], scheduler_environment_TEMPLATE={'PYTHONPATH': str(ARCH / 'src'), 'HF_HOME': str(SCRATCH / 'cache/huggingface'), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}, before_launch_gates=decl['priority'] + ['All earlier scheduler and child process groups exited; all four GPUs free', 'Actual durable bases accepted/offline-restored; fresh core/config/model/data verification completed', 'Fresh namespace and outputs absent; resume_from=None; no live-code fallback', 'ROOT separately binds actual plan/queue/expected endpoints/three GPU IDs and current UTC'], latest_actual_start_utc=decl['three_model_latest_actual_start_utc'], minimum_remaining_wall_seconds=4800, deadline_utc=decl['scheduler_deadline_utc'], launch_margin_seconds=120, stop_margin_seconds=120, CPU_postcheck_planning_allowance_seconds=600, CPU_postcheck_allowance_is_measured=False, single_2p8b_option='Original FALSE template unchanged; preserve its >=3600s gate if three-model gate fails. No auto-authority.', source_constraints=decl['source_constraints'], new_scientific_repeats=0)
ap = ROOT / 'docs/optional_three_model_B4_archive_QA_ROOT_FALSE_TEMPLATE.json'
write(ap, auth)
assert len(runs) == len(jobs) == 3 and {j['expected_seconds'] for j in jobs} == {2700, 3600, 2340}
assert all(not Path(e['checkpoint']).exists() for r in runs for e in r['endpoints'])
assert all(not Path(p).exists() for p in [expected['queue']['path'], expected['registry']['path'], expected['scheduler_state'], str(SCRATCH)])
end = resource.getrusage(resource.RUSAGE_SELF)
receipt = dict(schema_version=1, status='passed_CPU_preparation_only_FALSE_templates_not_execution_ready', created_at_utc=datetime.now(timezone.utc).isoformat(), producer=proof(__file__), predeclaration=proof(pre), archived_manifest=proof(manifest), historical_extraction_proof=proof(extraction), assessment_tool_1415=proof(tool_receipt_path), scope='Fresh SHA of extracted0732 required core/data/config/provenance payloads and small identity proofs; no checkpoint/TAR reads in this builder.', per_model_coverage=coverage, current_archived_payload_fresh_hash_count=len(checked), current_archived_payload_fresh_hash_bytes=sum(x['bytes'] for x in checked.values()), current_archived_payload_fresh_proofs=list(checked.values()), archived_helpers=helpers, missing_from_0732=[dict(member='scripts/verify_reproduction_qa_postrun.py', actual_1415_source=verifier, resolution='Selected original member extracted to assessment-tools namespace, not injected into0732'), dict(member='new three-model templates and C1B original best proof', resolution='Created after1415; must enter later actual snapshot and be frozen before future launch authority')], C1B_actual_best_identity=proof(prior1), C1B_actual_whole_process_timing=proof(ROOT / 'results/reproduction_qa/optional_C1b_original_best_identity_v1/whole_process_timing.json'), C2p8b_existing_best_identity=proof(prior2), templates=[proof(ap), proof(ep)], GPU_execution_started=False, training_queue_created=False, base_archive_extraction_or_relocation_performed=False, C_or_behavior_score_analysis_performed=False, archive_or_original_source_modified=False, selection_NLL_values_emitted=False, preparation_stage_timing=dict(wall_seconds=time.perf_counter()-start, user_cpu_seconds=end.ru_utime-usage.ru_utime, sys_cpu_seconds=end.ru_stime-usage.ru_stime, scope='Builder region after stdlib import through checking/templates; excludes earlier engineering and separate measured C1B identity child/1415 tool extraction'))
rp = ROOT / 'docs/optional_three_model_B4_archive_QA_CPU_preparation_receipt_v1.json'
write(rp, receipt)
print(json.dumps(dict(status=receipt['status'], fresh_hash_count=len(checked), fresh_hash_bytes=receipt['current_archived_payload_fresh_hash_bytes'], timing=receipt['preparation_stage_timing'])))
