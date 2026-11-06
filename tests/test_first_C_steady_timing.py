import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import watch_first_C_steady as module


def fixture():
    now = dt.datetime(2026, 11, 6, 11, tzinfo=dt.timezone.utc)
    sample = {'id': 'run', 'group': 'B4', 'sample': {'steps_observed': 100, 'warmup_steps_excluded': 10,
              'sampled_training_seconds': 90., 'sampled_target_prediction_tokens': 9000,
              'seconds_per_target_prediction_token': .01}}
    update = {'reliable_long_run_samples': [sample], 'remaining_seconds': 100., 'complete_long_runs': 0,
              'validation_seconds_per_point': 10., 'model_only_best_save_seconds': 2., 'latest_checkpoint_seconds': 5.}
    runtime = {'updated_at_utc': now.isoformat(), 'requested_matrix': [], 'status': 'monitoring',
               'failed_or_blocked_jobs': [], 'frozen_scientific_core_matches': {'train': True},
               'observer_source_sha256': module.OBSERVER_SHA,
               'extended_model_runtime_updates': {model: copy.deepcopy(update) for model in module.MODELS},
               'margin_before_master_2350_stop_minutes': 60., 'four_gpu_remaining_wall_hours_with_10pct': 10.,
               'master_status_counts': {'running': 4}}
    budget = {'sensitivity_scenarios': [{'name': 'central', 'per_phase': [{'id': 'main_behavior_adapted',
              'remaining_forward_jobs': 66, 'materialization_jobs': 0, 'per_job_whole_slot_seconds_proxy': 100.}]}]}
    return runtime, budget, now


def test_all_models_and_valid_warmup_required_for_timing_fit():
    runtime, budget, now = fixture()
    result = module.interpret(runtime, budget, now)
    assert result['all_three_models_have_100_steps_excluding_10_warmup']
    assert result['hypothetical_main_behavior_66_scenarios'][0]['fits_before_2350_under_timing_proxy']
    runtime['extended_model_runtime_updates']['pythia-2.8b']['reliable_long_run_samples'][0]['sample']['steps_observed'] = 99
    result = module.interpret(runtime, budget, now)
    assert not result['all_three_models_have_100_steps_excluding_10_warmup']
    assert not result['hypothetical_main_behavior_66_scenarios'][0]['fits_before_2350_under_timing_proxy']


def test_stale_mismatch_or_unknown_job_cost_cannot_produce_positive_fit():
    runtime, budget, now = fixture()
    for mutation in ['stale', 'core', 'source', 'unknown']:
        current = copy.deepcopy(runtime)
        if mutation == 'stale': current['updated_at_utc'] = (now - dt.timedelta(seconds=121)).isoformat()
        if mutation == 'core': current['frozen_scientific_core_matches']['train'] = False
        if mutation == 'source': current['observer_source_sha256'] = 'other'
        if mutation == 'unknown': current['requested_matrix'] = [{'phase': 'other_queued_jobs', 'remaining_runs': 1, 'job_ids': ['new']}]
        result = module.interpret(current, budget, now)
        assert not result['hypothetical_main_behavior_66_scenarios'][0]['fits_before_2350_under_timing_proxy']


def test_extra_time_includes_existing_training_margin_and_new_explicit_tail_once():
    runtime, budget, now = fixture()
    result = module.interpret(runtime, budget, now)
    case = result['hypothetical_main_behavior_66_scenarios'][0]
    assert case['four_slot_increment_with_10pct_and_dispatch_tail_seconds_proxy'] == 66 * 100 / 4 * 1.1 + 50
    assert case['hypothetical_margin_before_2350_minutes'] == 60 - (66 * 100 / 4 * 1.1 + 50) / 60
    assert result['total_compute_reduction_due_to_earlier_evaluation_seconds'] == 0
    assert not result['GPU_work_or_queue_mutation_authorized_by_this_receipt']


def test_bad_rate_or_warmup_does_not_count_as_reliable():
    runtime, budget, now = fixture()
    record = runtime['extended_model_runtime_updates']['pythia-1b']['reliable_long_run_samples'][0]['sample']
    record['seconds_per_target_prediction_token'] = float('nan')
    assert not module.interpret(runtime, budget, now)['models']['pythia-1b']['qualifying_sample_count']
    record['seconds_per_target_prediction_token'] = .01
    record['warmup_steps_excluded'] = 0
    assert not module.interpret(runtime, budget, now)['models']['pythia-1b']['qualifying_sample_count']


def test_immutable_timing_receipt_preserves_exact_observer_bytes(tmp_path):
    runtime, budget, now = fixture()
    report = module.interpret(runtime, budget, now)
    raw = json.dumps(runtime).encode()
    budget_path = tmp_path / 'budget.json'
    budget_path.write_text(json.dumps(budget))
    source_path = tmp_path / 'producer.py'
    source_path.write_text('# CPU timing only\n')
    assert module.freeze_receipt(tmp_path, 'timing', report, raw, budget_path, source_path)
    proof = json.loads((tmp_path / 'timing.json').read_text())
    assert (tmp_path / 'timing_runtime.json').read_bytes() == raw
    assert proof['source_runtime_snapshot']['sha256'] == hashlib.sha256(raw).hexdigest()
    module.write_json(tmp_path / 'progress.json', report)
    assert json.loads((tmp_path / 'progress.json').read_text())['utc'] == now.isoformat()
    assert not module.freeze_receipt(tmp_path, 'timing', {}, b'changed', budget_path, source_path)
    assert (tmp_path / 'timing_runtime.json').read_bytes() == raw
