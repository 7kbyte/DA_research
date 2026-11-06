#!/usr/bin/env python3
"""One-shot C410 NLL event-gated timing preview; no executable jobs or authority."""
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from reconcile_main_behavior_runtime import fifo_projection
from schedule_jobs import digest_job

FIXED = {
    'budget_v5.json': ('results/evaluation_timing_budget_v5.json', '940a0a0651cc7dd20da16d46321d8d4cac6b4dbbf7dd3e1ba6fbdd787e0b8f7d'),
    'FIFO_source.py': ('scripts/reconcile_main_behavior_runtime.py', '17d9b7839aef14e8c073124489c455d67512033f6de7658029411e88047a6f17'),
    'C410_registry.json': ('configs/extended_v2_pythia_410m_m64_formal_active_registry.json', 'c7f5c50d6794ba0df0dc3e071198da761f084ad9347ce5d9ce4dddf867ee48bd'),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, raw):
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def produce(output):
    if output.exists():
        raise FileExistsError('One immutable preview namespace only')
    blobs, proofs = {}, {}
    for name, (relative, wanted) in FIXED.items():
        path = ROOT / relative
        raw = path.read_bytes()
        if sha(raw) != wanted:
            raise ValueError('Frozen planning source changed: ' + relative)
        blobs[name] = raw
        proofs[name] = {'path': str(path), 'sha256': wanted}
    for name, relative in {'raw_runtime.json': 'results/runtime_budget_latest.json',
                           'master_state.json': 'logs/supplemental_job_state.json',
                           'master_queue.json': 'configs/supplemental_v1_queue.json',
                           'ready_index.json': 'artifacts/evaluation_ready_v5/ready_index.json'}.items():
        path = ROOT / relative
        blobs[name] = path.read_bytes()
        proofs[name] = {'path': str(path), 'sha256': sha(blobs[name])}
    source = Path(__file__).resolve()
    blobs['producer.py'] = source.read_bytes()
    proofs['producer.py'] = {'path': str(source), 'sha256': sha(blobs['producer.py'])}
    data = {name: json.loads(raw) for name, raw in blobs.items() if name.endswith('.json')}
    raw, state, master, registry, budget = (data[key] for key in
        ['raw_runtime.json', 'master_state.json', 'master_queue.json', 'C410_registry.json', 'budget_v5.json'])
    when = dt.datetime.fromisoformat(raw['updated_at_utc'])
    captured = dt.datetime.now(dt.timezone.utc)
    if not 0 <= (captured - when).total_seconds() < 90:
        raise ValueError('Current observer sample required')
    if (state.get('scheduler_pid') != 2988506 or len(master) != 564
            or raw['master_queue_jobs'] != 564 or raw['failed_or_blocked_jobs']
            or not all(raw['frozen_scientific_core_matches'].values())):
        raise ValueError('Master or frozen source state requires review')
    other = next(row for row in raw['requested_matrix'] if row['phase'] == 'other_queued_jobs')
    if other['job_ids'] or other['remaining_gpu_seconds']:
        raise ValueError('Only the current training-only remainder is priced here')
    c_rows = [row for update in raw['extended_model_runtime_updates'].values() for row in update['jobs']]
    unfinished = {jid for jid, record in state['jobs'].items() if record.get('status') != 'completed'}
    # Observer retains completed C rows at exactly zero remaining duration.
    # They are evidence rows, not unfinished work in the FIFO.
    if (unfinished != {row['id'] for row in c_rows if row['scheduler_status'] != 'completed'}
            or any(row['total_seconds'] != 0 for row in c_rows if row['scheduler_status'] == 'completed')
            or any(state['jobs'][row['id']]['status'] != row['scheduler_status'] for row in c_rows)):
        raise ValueError('Use a consistent observer/state snapshot')
    current = {job['id']: job for job in master}
    if len(current) != len(master) or any(state['jobs'][jid]['definition_sha256'] != digest_job(current[jid]) for jid in unfinished):
        raise ValueError('Frozen running/pending definitions differ')
    ids = {row['id'] for row in registry}
    if len(ids) != 24 or all(state['jobs'][jid]['status'] == 'completed' for jid in ids):
        raise ValueError('This is only a before-completion preview for the complete24 cohort')
    ready = next(phase for phase in data['ready_index.json']['phases'] if phase['id'] == 'extended_pythia_410m_100m_nll')
    if ready.get('ready'):
        raise ValueError('Real READY exists; use its real queue in a new root-reviewed calculation')
    # These descriptive records have no argv or outputs and cannot be scheduled.
    # The extra training dependencies are exclusively an event gate in this model.
    initial = {'id': 'timing_only_C410_extended_NLL_initial', 'depends_on': sorted(ids)}
    phase = [initial] + [{'id': 'timing_only_C410_NLL_' + kind + '_' + row['id'],
                         'depends_on': [initial['id']]} for kind in ('final', 'best') for row in registry]
    nonphase = raw['remaining_slot_hours'] * 3600
    baseline = fifo_projection(master, state, [], {}, raw, nonphase, True)
    if baseline['status'] != 'complete_time_simulation':
        raise ValueError('Current training FIFO is not fully priced')
    baseline_rows = {row['id']: row for row in baseline['future_dispatch_timing_rows']}
    remaining = {row['id']: row['total_seconds'] for row in c_rows}
    ends = {jid: (remaining[jid] * 1.1 if state['jobs'][jid]['status'] == 'running'
                  else baseline_rows[jid]['end_seconds'] if state['jobs'][jid]['status'] == 'pending' else 0.) for jid in ids}
    gate = max(ends.values())
    other_running_at_gate = [row for row in baseline_rows.values()
        if row['start_seconds'] < gate - 1e-7 and row['end_seconds'] > gate + 1e-7]
    stop = dt.datetime.fromisoformat('2026-11-06T23:50:00+00:00')
    tail = dt.datetime.fromisoformat('2026-11-07T02:56:32+00:00')
    cases = []
    for case in budget['sensitivity_scenarios']:
        rates = {row['id'].rsplit('_', 1)[-1]: row['per_job_whole_slot_seconds_proxy'] for row in case['per_phase']
                 if row['id'].startswith('C_pythia_410m_nll_')}
        if set(rates) != {'initial', 'final', 'best'}:
            raise ValueError('Use all three matched C410 NLL rate strata from v5')
        durations = {row['id']: rates['initial' if row is initial else 'final' if '_NLL_final_' in row['id'] else 'best'] for row in phase}
        forecast = fifo_projection(master, state, phase, durations, raw, nonphase, True)
        dispatch = {row['id']: row for row in forecast['future_dispatch_timing_rows']}
        new_initial = dispatch[initial['id']]
        assert new_initial['start_seconds'] >= gate - 1e-7
        for row in phase[1:]:
            assert dispatch[row['id']]['start_seconds'] >= new_initial['end_seconds'] - 1e-7
        for jid, row in baseline_rows.items():
            if row['start_seconds'] < gate - 1e-7:
                assert dispatch[jid] == row
        finish = when + dt.timedelta(seconds=forecast['makespan_seconds_with_10pct'])
        evaluation_done = when + dt.timedelta(seconds=max(dispatch[row['id']]['end_seconds'] for row in phase))
        # No actual completion credit is taken. Keeping the entire current v5
        # tail gives a conservative workflow envelope that intentionally still
        # includes these49 forwards; it is not an additive cost ledger.
        full_v5_envelope = finish + dt.timedelta(seconds=case['wall_seconds_with_dispatch_tail_and_handoff_proxy'])
        cases.append({'name': case['name'], 'v5_rates_already_include_sensitivity_factor': case['whole_wall_sensitivity_factor'],
            'rates_seconds': rates, 'forward_count_upperbound': {'initial': 1, 'final': 24, 'best': 24},
            'phase_slot_seconds_before_single_FIFO_10pct': sum(durations.values()),
            'initial_start_utc_if_zero_extra_READY_latency': (when + dt.timedelta(seconds=new_initial['start_seconds'])).isoformat(),
            'phase_forward_completion_utc_proxy': evaluation_done.isoformat(),
            'master_training_plus_early_NLL_finish_utc_proxy': finish.isoformat(),
            'increment_over_training_only_FIFO_minutes': (forecast['makespan_seconds_with_10pct'] - baseline['makespan_seconds_with_10pct']) / 60,
            'margin_before_2350_minutes': (stop - finish).total_seconds() / 60,
            'meets_15min_reserve_under_timing_proxy': (stop - finish).total_seconds() >= 900,
            'conservative_finish_including_unchanged_full_v5_tail_utc': full_v5_envelope.isoformat(),
            'conservative_margin_before_30min_delivery_tail_minutes': (tail - full_v5_envelope).total_seconds() / 60,
            'current_actual_tail_forward_credit': 0, 'future_actual_complete49_only_NLL_count_example': {'before': 129, 'after': 80},
            'FIFO': forecast})
    result = {'schema_version': 1, 'status': 'one_shot_read_only_preview_not_execution_authority',
        'captured_at_utc': captured.isoformat(), 'runtime_source_utc': when.isoformat(), 'source_proofs': proofs,
        'C410_status_counts': {status: sum(state['jobs'][jid]['status'] == status for jid in ids)
                               for status in ('completed', 'running', 'pending')},
        'C410_full24_training_gate_utc_proxy': (when + dt.timedelta(seconds=gate)).isoformat(),
        'other_training_slots_already_running_at_that_gate': other_running_at_gate,
        'training_only_FIFO': baseline, 'scenarios': cases,
        'full_training_cohort_unchanged': {'all_models_total_runs': 63, 'horizons_changed': False},
        'new_matched_extended_initial_required': True, 'primary_v3_NLL_baseline_reused': False,
        'conceptual_timing_rows_not_executable_jobs': True, 'extra_training_dependencies_are_simulation_only': True,
        'phase_gate_additional_audit_proof_READY_latency_seconds': None,
        'no_actual_remaining_evaluation_budget_credit': True, 'no_unproved_D_or_best_final_alias_discount': True,
        'no_total_compute_saving_claimed': True, 'main66_still_deferred': True, 'optional_2p8b_QA_still_paused': True,
        'checks_executed': ['No simulated initial before all24 training ends', 'Every48 adapted starts only after new initial ends',
            'Training dispatch before the cohort gate is identical to current FIFO', 'Actual currently running children keep their slots',
            'Sensitivity factors are already in rate inputs and not multiplied again; entire FIFO adds10 percent once'],
        'future_ROOT_gate_checklist': [
            'All24 C410 training jobs actually complete/rc0 with immutable definitions, full100M exposures and source audits.',
            'Published actual v5 model-specific READY plus complete audit/final+best state proofs, queue and manifest SHA.',
            'Use the actual matched-extended initial/dependency closure; never alias main_v3 baseline.',
            'Only the actual frozen plan may spend a proved alias; this preview spends none.',
            'Recapture contemporaneous master/state/runtime and reprice actual complete queue with current occupied long-run slots.',
            'Require high scenario >=15min before23:50, retain all63 training/horizons and full30min delivery tail.',
            'Obtain separate ROOT execution authority and append through existing single master only; this file authorizes nothing.',
            'After actual successful whole-phase output/attestation verification, deduct only matching completed work once from final tail.'],
        'limitations': ['Zero extra CPU audit/proof/READY latency is modeled; unmeasured gate delay changes slot placement and must be repriced at real READY.',
            'The unchanged full-v5-tail envelope intentionally gives no future-work credit and includes the same49 again; it is a conservative schedule check, never a total-cost sum.',
            'Main4 archive QA has a separate budget and no execution is authorized here; optional2.8B QA remains paused.',
            'FIFO timing forecasts do not guarantee deadlines or replace actual scheduler admission/source checks.'],
        'GPU_or_queue_or_authority_or_watcher_started': False, 'score_semantics_read': False, 'global_report_modified': False}
    output.mkdir()
    sources = output / 'sources'
    sources.mkdir()
    for name, raw_bytes in blobs.items():
        write(sources / name, raw_bytes)
    write(output / 'preview.json', (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
    with (output / 'scenarios.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scenario', 'phase_done_utc', 'master_finish_utc', 'increment_minutes', 'margin_before2350_minutes', 'full_v5_envelope_finish', 'margin_before_delivery_tail_minutes'])
        for case in cases:
            writer.writerow([case['name'], case['phase_forward_completion_utc_proxy'], case['master_training_plus_early_NLL_finish_utc_proxy'],
                case['increment_over_training_only_FIFO_minutes'], case['margin_before_2350_minutes'],
                case['conservative_finish_including_unchanged_full_v5_tail_utc'], case['conservative_margin_before_30min_delivery_tail_minutes']])
    print(json.dumps({'path': str(output / 'preview.json'), 'sha256': sha((output / 'preview.json').read_bytes()),
        'training_gate_utc_proxy': result['C410_full24_training_gate_utc_proxy'],
        'scenarios': [{key: case[key] for key in ['name', 'increment_over_training_only_FIFO_minutes', 'margin_before_2350_minutes']} for case in cases]}))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: producer.py NEW_PREVIEW_DIRECTORY')
    produce(Path(sys.argv[1]).resolve())
