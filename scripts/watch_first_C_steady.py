#!/usr/bin/env python3
"""Observe existing C runtime samples and hypothetical main-behavior timing only."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time

from stat_bound_digest_cache import atomic, fsync_directory

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('pythia-410m', 'pythia-1b', 'pythia-2.8b')
OBSERVER_SHA = 'aaf4c774b47d00bef8794ac4151f1194ad04c6d6d6dce830ceceb337f331d0f4'
BUDGET_SHA = 'a5154c67177e5d43bc77593c8927ebc06c330989197e8db0dc701a80da381328'


def utc(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def interpret(runtime, budget, now):
    """No score files or event/summary values are read by this observer."""
    age = (now - utc(runtime['updated_at_utc'])).total_seconds()
    unknown = [row for row in runtime['requested_matrix']
               if row['phase'] == 'other_queued_jobs' and row['remaining_runs']]
    healthy = (runtime['status'] == 'monitoring'
               and not runtime['failed_or_blocked_jobs']
               and bool(runtime['frozen_scientific_core_matches'])
               and all(runtime['frozen_scientific_core_matches'].values())
               and runtime['observer_source_sha256'] == OBSERVER_SHA
               and -1 <= age <= 120)
    models = {}
    for model in MODELS:
        update = runtime['extended_model_runtime_updates'][model]
        samples = []
        for candidate in update['reliable_long_run_samples']:
            sample = candidate['sample']
            seconds = sample['sampled_training_seconds']
            tokens = sample['sampled_target_prediction_tokens']
            rate = sample['seconds_per_target_prediction_token']
            if (sample['steps_observed'] < 100 or sample['warmup_steps_excluded'] != 10
                    or tokens <= 0 or not math.isfinite(seconds) or seconds <= 0
                    or not math.isfinite(rate) or rate <= 0
                    or not math.isclose(seconds / tokens, rate, rel_tol=1e-12)):
                continue
            samples.append(candidate)
        models[model] = {
            'qualifying_sample_count': len(samples),
            'samples': samples,
            'remaining_seconds_proxy': update['remaining_seconds'],
            'complete_long_runs': update['complete_long_runs'],
            'validation_seconds_per_point_proxy': update['validation_seconds_per_point'],
            'model_only_best_save_seconds_proxy': update['model_only_best_save_seconds'],
            'latest_checkpoint_seconds_proxy': update['latest_checkpoint_seconds'],
        }
    all_models = all(record['qualifying_sample_count'] for record in models.values())
    base_margin = runtime['margin_before_master_2350_stop_minutes']
    scenarios = []
    for case in budget['sensitivity_scenarios']:
        row = next(row for row in case['per_phase'] if row['id'] == 'main_behavior_adapted')
        if row['remaining_forward_jobs'] != 66 or row['materialization_jobs'] != 0:
            raise ValueError('Budget does not describe the authorized remaining 66 forwards')
        one = row['per_job_whole_slot_seconds_proxy']
        ideal = 66 * one / 4
        # Explicit conservative operational proxy, separate from scientific scoring.
        tail = (math.ceil(66 / 4) - 66 / 4) * one
        increment = ideal * 1.1 + tail
        margin = base_margin - increment / 60
        scenarios.append({
            'name': case['name'],
            'per_job_whole_slot_seconds_proxy': one,
            '66_forward_slot_seconds_proxy': 66 * one,
            'four_slot_increment_with_10pct_and_dispatch_tail_seconds_proxy': increment,
            'existing_training_forecast_already_contains_10pct': True,
            'hypothetical_margin_before_2350_minutes': margin,
            'fits_before_2350_under_timing_proxy': bool(healthy and all_models and not unknown and margin >= 0),
        })
    finish = now + dt.timedelta(hours=runtime['four_gpu_remaining_wall_hours_with_10pct'])
    return {
        'schema_version': 1,
        'utc': now.isoformat(),
        'source_observer_updated_at_utc': runtime['updated_at_utc'],
        'source_observer_age_seconds': age,
        'source_observer_healthy_and_fresh': healthy,
        'status': 'all_three_models_qualified' if healthy and all_models else 'waiting_reliable_samples',
        'all_three_models_have_100_steps_excluding_10_warmup': bool(all_models),
        'minimum_steps_observed': 100,
        'warmup_steps_excluded': 10,
        'models': models,
        'unknown_queued_job_allowances_requiring_reconciliation': unknown,
        'training_finish_utc_with_existing_10pct_proxy': finish.isoformat(),
        'margin_before_master_2350_stop_minutes': base_margin,
        'hypothetical_main_behavior_66_scenarios': scenarios,
        'master_status_counts': runtime['master_status_counts'],
        'scientific_conditions_changed': False,
        'scores_or_predictions_read': False,
        'GPU_work_or_queue_mutation_authorized_by_this_receipt': False,
        'total_compute_reduction_due_to_earlier_evaluation_seconds': 0,
        'limitations': [
            'Early evaluation moves the same work; its time must leave the later evaluation budget once completed.',
            'First B4 samples cover one method per model, not all future methods or contention patterns.',
            'Training steps omit validation/checkpoint time; the source forecast includes those as separate costs.',
            'The observer uses model/method pilot factors outside measured methods and is not a completion guarantee.',
            'Root alone decides whether the unchanged remaining 66 evaluation definitions enter the single master.',
        ],
    }


def write_json(path, value):
    atomic(path, value)


def freeze_receipt(directory, name, report, runtime_raw, budget_path, source_path):
    path = directory / (name + '.json')
    if path.exists():
        return False
    snapshot = directory / (name + '_runtime.json')
    with snapshot.open('xb') as stream:
        stream.write(runtime_raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(snapshot.parent)
    payload = dict(report)
    payload['source_runtime_snapshot'] = {'path': str(snapshot.resolve()), 'sha256': digest(runtime_raw)}
    payload['source_budget'] = {'path': str(budget_path.resolve()), 'sha256': BUDGET_SHA}
    payload['producer'] = {'path': str(source_path.resolve()), 'sha256': digest(source_path.read_bytes())}
    write_json(path, payload)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--poll-seconds', type=float, default=60.)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/C_steady_timing_v1')
    args = p.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        p.error('Poll interval must be between 1 and 60 seconds')
    runtime_path = ROOT / 'results/runtime_budget_latest.json'
    budget_path = ROOT / 'results/evaluation_timing_budget_v4.json'
    budget_raw = budget_path.read_bytes()
    if digest(budget_raw) != BUDGET_SHA:
        raise ValueError('Frozen pure-time evaluation budget changed')
    if digest((ROOT / 'scripts/monitor_runtime_budget.py').read_bytes()) != OBSERVER_SHA:
        raise ValueError('Source runtime observer changed')
    budget = json.loads(budget_raw)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        raw = runtime_path.read_bytes()
        report = interpret(json.loads(raw), budget, dt.datetime.now(dt.timezone.utc))
        report['watcher_pid'] = os.getpid()
        if report['source_observer_healthy_and_fresh']:
            qualified = [model for model, record in report['models'].items() if record['qualifying_sample_count']]
            if qualified:
                freeze_receipt(directory, 'first_model_steady', report, raw, budget_path, Path(__file__))
            for model in qualified:
                freeze_receipt(directory, model.replace('-', '_') + '_first_steady', report, raw, budget_path, Path(__file__))
            if report['all_three_models_have_100_steps_excluding_10_warmup']:
                freeze_receipt(directory, 'all_three_models_steady', report, raw, budget_path, Path(__file__))
        write_json(directory / 'progress.json', report)
        print(json.dumps({'utc': report['utc'], 'status': report['status'],
                          'qualified_models': [model for model, record in report['models'].items() if record['qualifying_sample_count']],
                          'source_healthy': report['source_observer_healthy_and_fresh']}), flush=True)
        if not args.watch or report['status'] == 'all_three_models_qualified':
            break
        if dt.datetime.now(dt.timezone.utc) >= utc('2026-11-07T00:00:00Z'):
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
