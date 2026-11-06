#!/usr/bin/env python3
"""Read-only operational handoff gap receipt; no prepare/start or large input reads."""
import ast
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from handoff_evaluation import deadline_settings


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write(path, blob):
    with path.open('xb') as stream:
        stream.write(blob); stream.flush(); os.fsync(stream.fileno())


def main(out):
    started = time.perf_counter(); cpu = time.process_time()
    if out.exists():
        raise FileExistsError('Use a new immutable receipt namespace')
    blobs, proofs = {}, {}
    def load(name, path, wanted=None):
        path = Path(path); blob = path.read_bytes()
        if wanted is not None and sha(blob) != wanted:
            raise ValueError('Source differs: ' + str(path))
        blobs[name] = blob; proofs[name] = {'path': str(path.resolve()), 'sha256': sha(blob), 'bytes': len(blob)}
        return json.loads(blob) if name.endswith('.json') else blob
    authority = load('v5_execution.json', ROOT / 'docs/supplemental_evaluation_preparation_v5_execution.json')
    for path, wanted in {**authority['sources'], **authority['frozen_scorer_input_sha256']}.items():
        load('code_' + str(len(blobs)) + '_' + Path(path).name, path, wanted)
    cat = load('catalog.json', authority['catalog']['path'], authority['catalog']['sha256'])
    original = load('preparer_v4_source.py', authority['unchanged_v4_producer']['path'], authority['unchanged_v4_producer']['sha256'])
    ready = load('ready_index.json', ROOT / 'artifacts/evaluation_ready_v5/ready_index.json')
    draft = load('handoff_draft.json', ROOT / 'artifacts/evaluation_ready_v5/handoff_draft.json')
    cache_progress = load('cache_progress.json', ROOT / 'artifacts/evaluation_ready_v5/cache_progress.json')
    snapshot = load('cache_snapshot.json', cache_progress['cache_snapshot']['path'], cache_progress['cache_snapshot']['sha256'])
    c_receipt = load('latest_published_C_only_receipt.json', ROOT / 'results/C_pending_order_postpublication_v1/receipt.json',
                     '3ae983665892813f2ce050e1e93dfef8448705198d8be5b4cda59147d8b72f50')
    budget = load('budget_v5.json', ROOT / 'results/evaluation_timing_budget_v5.json',
                  '940a0a0651cc7dd20da16d46321d8d4cac6b4dbbf7dd3e1ba6fbdd787e0b8f7d')
    load('master_cmdline.bin', '/proc/2988506/cmdline')
    load('producer.py', Path(__file__).resolve())
    assert draft['example_only'] is True and 'digest_cache' not in draft
    assert len(draft['pending_plan_ids']) == 6 and len(cat['plans']) == 9
    statuses = {plan['id']: plan for plan in ready['phases']}
    assert sum(bool(plan.get('ready')) for plan in statuses.values()) == 3
    expressions = [node.args[4] for node in ast.walk(ast.parse(original)) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == 'build' and len(node.args) > 4]
    assert len(expressions) == 1 and isinstance(expressions[0], ast.IfExp)
    expression = expressions[0]
    assert ast.unparse(expression.test) == "mode == 'behavior'"
    projected_allowances = {'behavior': ast.literal_eval(expression.body), 'nll': ast.literal_eval(expression.orelse)}
    assert projected_allowances == {'behavior': 600, 'nll': 180}
    source_rows = []
    def actual_source(source_id, source, status):
        queue = load(source_id + '_queue.json', source['queue_path'], source['queue_sha256'])
        manifests = [load(source_id + '_manifest_' + str(index) + '.json', proof['path'], proof['sha256'])
                     for index, proof in enumerate(source['manifests'])]
        entries = {entry['id']: entry for manifest in manifests for entry in manifest['jobs']}
        counts = Counter((entries[job['id']]['kind'], job['expected_seconds']) for job in queue)
        return {'source_id': source_id, 'status': status, 'queue': {'path': source['queue_path'], 'sha256': source['queue_sha256']},
                'manifests': source['manifests'], 'actual_queue_jobs': len(queue),
                'allowance_by_actual_job_kind': [{'kind': kind, 'expected_seconds': seconds, 'actual_jobs': count}
                                                for (kind, seconds), count in sorted(counts.items())],
                'maximum_expected_seconds': max(job['expected_seconds'] for job in queue)}
    for name, source in zip(('main_nll', 'main_behavior'), draft['evaluation_sources'][:2]):
        source_rows.append(actual_source(name, source, 'actual_frozen_primary_source'))
    for plan in cat['plans']:
        status = statuses[plan['id']]
        if status.get('ready'):
            source = {'queue_path': status['queue']['path'], 'queue_sha256': status['queue']['sha256'], 'manifests': [status['manifest']]}
            source_rows.append(actual_source(plan['id'], source, 'actual_READY_frozen_source'))
        else:
            source_rows.append({'source_id': plan['id'], 'status': 'not_READY_frozen_plan_only_no_actual_queue',
                'mode': plan['mode'], 'model_tag': plan['phase']['model_tag'], 'checkpoints': plan['checkpoints'],
                'actual_queue_jobs': None, 'queue': None, 'manifests': None,
                'expected_per_job_seconds_from_frozen_producer': projected_allowances[plan['mode']],
                'applicable_job_kinds_if_materialized': ['initial', 'adapted', 'reuse'],
                'allowance_evidence': {'source': proofs['preparer_v4_source.py'], 'line': expression.lineno,
                                       'expression': ast.unparse(expression)},
                'maximum_expected_seconds': projected_allowances[plan['mode']],
                'future_actual_manifest_queue_and_kind_counts_must_be_checked': True})
    priors = [actual_source('prior_' + str(index), source, 'explicit_frozen_prior_closure_source')
              for index, source in enumerate(draft['prior_sources'])]
    assert len(source_rows) == 11 and len(priors) == 3
    settings = deadline_settings({'hard_deadline_utc': draft['hard_deadline_utc'], 'tail_seconds': 1800})
    assert settings['launch_margin_seconds'] == 2210 and settings['stop_margin_seconds'] == 1910
    deadline = dt.datetime.fromisoformat(settings['hard_deadline_utc'].replace('Z', '+00:00'))
    base_cutoff = deadline - dt.timedelta(seconds=settings['launch_margin_seconds'])
    stop = deadline - dt.timedelta(seconds=settings['stop_margin_seconds'])
    delivery_tail = deadline - dt.timedelta(seconds=1800)
    high = next(case for case in budget['sensitivity_scenarios'] if case['name'] == 'high')
    c_forecast = c_receipt['new_C_only_FIFO_admission_forecast']
    finish_C = dt.datetime.fromisoformat(c_forecast['finish_utc'])
    envelope = finish_C + dt.timedelta(seconds=high['wall_seconds_with_dispatch_tail_and_handoff_proxy'])
    maximum = max(row['maximum_expected_seconds'] for row in source_rows + priors)
    assert maximum == 600
    cutoff_rows = [{'expected_seconds': seconds, 'strict_latest_launch_utc': (base_cutoff - dt.timedelta(seconds=seconds)).isoformat(),
                    'coarse_envelope_end_before_this_launch_cutoff_seconds': ((base_cutoff - dt.timedelta(seconds=seconds)) - envelope).total_seconds()}
                   for seconds in sorted({row['maximum_expected_seconds'] for row in source_rows + priors})]
    # At equality the existing scheduler rejects, independently of an optimistic wall estimate.
    assert all((base_cutoff - dt.timedelta(seconds=row['expected_seconds'])) + dt.timedelta(seconds=row['expected_seconds']) >= base_cutoff
               for row in cutoff_rows)
    result = {'schema_version': 1, 'status': 'read_only_final_handoff_gap_review_not_authority',
        'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'sources': proofs,
        'final_evaluation_source_count': 11, 'actual_frozen_or_READY_sources': 5, 'not_READY_C_plans': 6,
        'eleven_sources': source_rows, 'three_explicit_prior_sources': priors,
        'maximum_actual_allowance_seconds': max(row['maximum_expected_seconds'] for row in source_rows if row['queue']),
        'maximum_actual_or_frozen_projected_allowance_seconds': maximum,
        'handoff_settings_for1800second_tail': settings,
        'actual_stop_threshold_utc': stop.isoformat(), 'delivery_tail_boundary_utc': delivery_tail.isoformat(),
        'strict_launch_rule': 'now + frozen_job.expected_seconds < hard_deadline - launch_margin_seconds',
        'strict_equality_is_rejected': True, 'base_launch_threshold_before_expected_seconds_utc': base_cutoff.isoformat(),
        'per_allowance_strict_launch_cutoffs': cutoff_rows,
        'coarse_conservative_envelope_not_final_FIFO': {
            'published_C_only_runtime_utc': c_forecast['runtime_source_utc'], 'published_C_only_finish_utc': finish_C.isoformat(),
            'full_unchanged_v5_high_remaining_evaluation_seconds': high['wall_seconds_with_dispatch_tail_and_handoff_proxy'],
            'v5_already_contains_its_sensitivity_and_dispatch_tail_and_handoff_proxy': True,
            'additional10percent_or_sensitivity_multiplier_applied': False,
            'envelope_completion_utc': envelope.isoformat(),
            'completion_before_actual_stop_seconds': (stop - envelope).total_seconds(),
            'completion_before_delivery_tail_seconds': (delivery_tail - envelope).total_seconds(),
            'sufficient_coarse_latest_start_bound_assuming_all_starts_no_later_than_envelope_end': True,
            'minimum_margin_to_all_allowance_launch_cutoffs_seconds': min(row['coarse_envelope_end_before_this_launch_cutoff_seconds'] for row in cutoff_rows),
            'final_real_FIFO_done': False, 'new_actual_C_queues_or_equivalence_discounts_assumed': False,
            'unmeasured_phase_audit_READY_or_transfer_delay_not_a_guarantee': True},
        'cache_now': {'snapshot': cache_progress['cache_snapshot'], 'snapshot_entries': len(snapshot['entries']),
            'not_the_final_cache_binding': True, 'future_final_spec_must_bind_latest_complete_immutable_snapshot': True,
            'current_draft_digest_cache_missing': True,
            'must_retain_receipt_JSON_paths_and_checkpoint_canonical_paths_and_stat': True,
            'cache_hit_is_historical_full_read_plus_current_stat_not_current_bytes': True,
            'prepare_and_start_each_create_verifier_and_default_to_full_SHA_without_explicit_binding': True,
            'worker_per_job_fresh_complete_input_SHA_unchanged': True},
        'final_ROOT_checklist': [
            'All9 supplemental plans actual READY with scope/source/full-phase gates; helper does not enforce catalog completeness or pending_plan_ids.',
            'Finalize11 explicit evaluation sources plus3 prior sources, each actual queue/manifest SHA; same-ID definitions must be identical.',
            'Set example_only=false only in a separately finalized immutable spec; keep original draft. Bind digest_cache={path,sha256} to latest immutable snapshot, not mutable index/directory.',
            'Stat/check selected-input receipt coverage before handoff. Missing/changed evidence falls back to fresh full SHA; do not silently budget it as a hit.',
            'Old master must be exited/reaped, all recorded child PIDs/PGs absent and old state.lock exclusively acquired; no overlap.',
            'Prepare in a new namespace. Inherit only rc0 completed identical definitions with all current outputs and canonical/alias attestation SHA; otherwise pending.',
            'Inspect inherited completion counts, full-input/hit counters and actual finalized queue. Check real FIFO/each job allowance against strict launch gates as well as completion and30minute reserve.',
            'ROOT separately authorizes start. Start rechecks source/spec, old queue/state snapshots, all inherited output attestations, and launch.lock; old state lock passes to new4GPU detached scheduler.',
            'Only frozen run_frozen_evaluation.py argv/env accepted. No training jobs; no changed science, scoring policy, horizon or fallback truncation.'],
        'master_still_alive_at_review': Path('/proc/2988506').exists(),
        'handoff_or_prepare_or_start_or_final_evaluation_plan_called': False, 'large_checkpoint_or_score_payload_read': False,
        'live_sources_changed': False, 'new_GPU_scheduler_or_watcher': False,
        'wall_seconds': time.perf_counter() - started, 'process_cpu_seconds': time.process_time() - cpu}
    out.mkdir(); (out / 'sources').mkdir()
    for name, blob in blobs.items():
        write(out / 'sources' / name, blob)
    write(out / 'receipt.json', (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
    with (out / 'source_allowances.csv').open('x', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['source_id', 'status', 'kind', 'expected_seconds', 'actual_jobs'])
        for row in source_rows:
            if row['queue']:
                for kind in row['allowance_by_actual_job_kind']:
                    writer.writerow([row['source_id'], row['status'], kind['kind'], kind['expected_seconds'], kind['actual_jobs']])
            else:
                writer.writerow([row['source_id'], row['status'], 'all_kinds_if_materialized', row['expected_per_job_seconds_from_frozen_producer'], 'not_materialized'])
    print(json.dumps({'receipt': str(out / 'receipt.json'), 'sha256': sha((out / 'receipt.json').read_bytes()),
        'actual_sources': 5, 'not_READY_sources': 6, 'maximum_expected_seconds': maximum,
        'coarse_completion_utc': envelope.isoformat(), 'minimum_strict_launch_cutoff_headroom_minutes': result['coarse_conservative_envelope_not_final_FIFO']['minimum_margin_to_all_allowance_launch_cutoffs_seconds'] / 60}))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: collect.py NEW_IMMUTABLE_OUTPUT_DIRECTORY')
    main(Path(sys.argv[1]).resolve())
