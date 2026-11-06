#!/usr/bin/env python3
"""Export audited primary36 exposure/input/timer tables; no scoring or GPU work."""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_formal_runs import CORES, events_at, expected_replay, logical_key, sha256
from make_experiments import atomic_json

COUNTS = [f'{kind}_{unit}' for kind in ['target', 'replay'] for unit in
          ['prediction_tokens', 'raw_input_tokens', 'padded_input_tokens']]
COSTS = ['training_seconds', 'training_teacher_forward_seconds', 'validation_seconds',
         'validation_teacher_forward_seconds', 'validation_kl_arithmetic_seconds',
         'checkpoint_seconds', 'setup_seconds']
COVERAGE = ['target_capacity_prediction_tokens', 'target_unique_capacity_prediction_tokens',
            'target_actual_unique_prediction_tokens', 'target_repeated_prediction_token_exposures',
            'replay_bank_capacity_prediction_tokens', 'replay_bank_unique_capacity_prediction_tokens',
            'replay_actual_unique_prediction_tokens', 'replay_repeated_prediction_token_exposures']


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def proof(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256(path)}


def uniform_stream_counts(takes, batch_size, sequence_length=512):
    """Independent scalar reconstruction for full, all-valid, equal-length rows."""
    require(batch_size > 0 and sequence_length >= 2, 'Invalid uniform-row reconstruction settings')
    offset = 0
    result = {'prediction_tokens': 0, 'raw_input_tokens': 0, 'padded_input_tokens': 0,
              'student_forward_calls_derived': 0}
    for requested in takes:
        require(isinstance(requested, int) and requested >= 0, 'Exposure requests must be nonnegative integers')
        remaining, lengths = requested, []
        result['prediction_tokens'] += requested
        while remaining:
            count = min(remaining, sequence_length - 1 - offset)
            lengths.append(offset + count + 1)
            offset = (offset + count) % (sequence_length - 1)
            remaining -= count
            if len(lengths) == batch_size or remaining == 0:
                result['raw_input_tokens'] += sum(lengths)
                result['padded_input_tokens'] += len(lengths) * max(lengths)
                result['student_forward_calls_derived'] += 1
                lengths = []
    return {**result, 'final_offset': offset}


def schedule_counts(config, sequence_length=512):
    step_tokens = config.get('target_tokens_per_step') or (
        config['microbatch_size'] * config['gradient_accumulation_steps'] * (sequence_length - 1))
    target, prior_target, prior_replay = [], 0, 0
    replay = []
    while prior_target < config['target_prediction_tokens']:
        current = min(prior_target + step_tokens, config['target_prediction_tokens'])
        current_replay = expected_replay(current, config['replay_ratio'])
        target.append(current - prior_target)
        replay.append(current_replay - prior_replay)
        prior_target, prior_replay = current, current_replay
    return {kind: uniform_stream_counts(takes, config['microbatch_size'], sequence_length)
            for kind, takes in [('target', target), ('replay', replay)]}


def csv_file(path, rows):
    require(bool(rows), 'Refusing an empty budget CSV')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for values, members in sorted(groups.items()):
        record = {**dict(zip(keys, values)), 'physical_runs': len(members),
                  'groups': '|'.join(sorted({row['group'] for row in members})),
                  'seeds': '|'.join(str(seed) for seed in sorted({row['seed'] for row in members}))}
        for field in [*COUNTS, *COVERAGE, 'replay_ratio_realized', 'lambda', 'lambda_over_b',
                      'target_student_forward_calls_derived', 'replay_student_forward_calls_derived']:
            observed = {row[field] for row in members}
            require(len(observed) == 1, f'Same-budget data counts or coefficients differ: {keys}, {values}, {field}')
            record[field] = next(iter(observed))
        for field in [*COSTS, 'total_accounted_wall_seconds']:
            numbers = [row[field] for row in members]
            record.update({field + '_mean': statistics.mean(numbers), field + '_min': min(numbers),
                           field + '_max': max(numbers),
                           field + '_sample_sd': statistics.stdev(numbers) if len(numbers) > 1 else 0.})
        result.append(record)
    return result


def export(args):
    require(not args.output.exists(), 'Choose a fresh output directory; existing evidence is never overwritten')
    sources = [proof(path) for path in [args.registry, args.audit, args.snapshot, Path(__file__)]]
    registry, audit, snapshot = read(args.registry), read(args.audit), read(args.snapshot)
    require(audit['status'] == 'passed' and audit['completed_runs'] == audit['expected_runs'] == 36
            and audit['model_weights_compared'] and not audit['violations'], 'Complete passed primary36 audit required')
    expected_grid = {(group, seed, 0. if group == 'B0' else .05)
                     for seed in [11, 22, 33] for group in ['B0', 'B1', 'B2', 'B3', 'B4', 'D', 'M', 'C']}
    expected_grid |= {(group, seed, ratio) for group in ['B4', 'M'] for seed in [11, 22, 33] for ratio in [.01, .1]}
    require(len(registry) == 36 and {(row['group'], row['seed'], row['ratio']) for row in registry} == expected_grid,
            'Registry does not contain exactly the frozen36 conditions')
    require(len({row['id'] for row in registry}) == 36, 'Duplicate run ID')
    by_id = {row['id']: row for row in audit['runs']}
    require(set(by_id) == {row['id'] for row in registry}, 'Audit/registry identities disagree')
    for relative in CORES:
        observed = sha256(ROOT / relative)
        require(observed == snapshot['code'][relative]['sha256'] == audit['frozen_scientific_code'][relative]['actual_sha256'],
                f'Frozen scientific code mismatch: {relative}')
        sources.append(proof(ROOT / relative))
    rows, reconstruction = [], {}
    for row in sorted(registry, key=lambda row: (row['group'], row['ratio'], row['seed'])):
        name, audited = row['id'], by_id[row['id']]
        config, summary = read(row['config_path']), read(row['summary_path'])
        sources.extend([proof(row['config_path']), proof(row['summary_path'])])
        require(sha256(row['config_path']) == snapshot['configs'][logical_key(row['config_path'])]['sha256'],
                f'Frozen config SHA mismatch: {name}')
        require(summary['status'] == 'completed' and summary['config'] == config, f'Runtime config/status mismatch: {name}')
        counters, timing = summary['counters'], summary['timing']
        require(config['seed'] == row['seed'] and config['replay_ratio'] == row['ratio'] and
                config['replay_weight'] == row['weight'], f'Registry coefficients differ from config: {name}')
        require(counters['target_prediction_tokens'] == audited['target_tokens'] == config['target_prediction_tokens'] == 10_000_000,
                f'Target budget mismatch: {name}')
        require(counters['replay_prediction_tokens'] == audited['replay_tokens'] ==
                expected_replay(counters['target_prediction_tokens'], row['ratio']) == summary['replay_budget_planned'],
                f'Exact replay prediction budget mismatch: {name}')
        for field in COSTS:
            require(counters[field] == audited['costs'][field] and math.isfinite(counters[field]) and counters[field] >= 0,
                    f'Observed timer differs from completed audit: {name}, {field}')
        require(timing['training_teacher_forward_seconds_complete'] and not timing['nested_timers_are_additive_costs']
                and timing['training_teacher_forward_seconds_nested_in'] == 'training_seconds'
                and timing['validation_teacher_forward_seconds_nested_in'] == 'validation_seconds',
                f'Nested timing policy mismatch: {name}')
        require(counters['training_teacher_forward_seconds'] <= counters['training_seconds'] + .01 and
                counters['validation_teacher_forward_seconds'] <= counters['validation_seconds'] + .01,
                f'Teacher sub-timer exceeds its parent: {name}')
        coverage = summary['coverage_accounting']
        for kind in ['target', 'replay']:
            data = summary['provenance'][kind]
            cov = coverage[kind]
            if data is None:
                require(kind == 'replay' and cov is None and row['group'] == 'B0', 'Unexpected missing dataset')
                continue
            require(data['sequence_length'] == 512 and data['raw_input_tokens'] == data['rows'] * 512
                    and data['prediction_tokens'] == data['rows'] * 511,
                    f'Uniform all-valid512-row reconstruction is inapplicable: {name}, {kind}')
            require(cov['no_duplicate_chunks_verified'] and cov['assumption_no_duplicate_chunks'],
                    f'Unique-exposure assumption is not verified: {name}, {kind}')
            unique = min(counters[kind + '_prediction_tokens'], cov['unique_capacity_prediction_tokens'])
            require(cov['unique_prediction_tokens_exposed'] == summary[kind + '_unique_prediction_tokens_exposed'] == unique
                    and cov['repeated_prediction_token_exposures'] == counters[kind + '_prediction_tokens'] - unique
                    and cov['cumulative_prediction_tokens_exposed'] == counters[kind + '_prediction_tokens'],
                    f'Unique/repeated exposure mismatch: {name}, {kind}')
        expected_capacity = 0 if row['group'] == 'B0' else 100156
        require(summary['replay_bank_capacity_prediction_tokens'] == summary['replay_bank_unique_prediction_tokens'] ==
                audited['bank_capacity'] == expected_capacity, f'Replay K mismatch: {name}')
        events_path = Path(row['output_dir']) / 'events.jsonl'
        sources.append(proof(events_path))
        steps = [event for event in events_at(events_path, False) if event['event'] == 'train_step']
        require(len(steps) == counters['step'] == 306, f'Incomplete step events: {name}')
        for field in COUNTS:
            require(counters[field] == sum(event['step_counts'][field] for event in steps),
                    f'Input/prediction count differs from complete events: {name}, {field}')
        derived = schedule_counts(config)
        for kind in ['target', 'replay']:
            for unit in ['prediction_tokens', 'raw_input_tokens', 'padded_input_tokens']:
                require(counters[kind + '_' + unit] == derived[kind][unit],
                        f'Independent context/padding reconstruction differs: {name}, {kind}, {unit}')
        reconstruction[name] = derived
        target, replay = coverage['target'], coverage['replay']
        ratio = counters['replay_prediction_tokens'] / counters['target_prediction_tokens']
        require(ratio == summary['replay_ratio_realized'] == row['ratio'], f'Realized ratio mismatch: {name}')
        record = {'run_id': name, 'group': row['group'], 'seed': row['seed'],
            'replay_source': 'none' if row['group'] == 'B0' else ('c4' if row['group'] in {'B1', 'B2'} else 'generated'),
            'replay_objective': config['replay_objective'], 'replay_ratio_planned': row['ratio'],
            'replay_ratio_realized': ratio, 'lambda': config['replay_weight'],
            'lambda_over_b': float(Fraction(str(config['replay_weight'])) / Fraction(str(row['ratio']))) if row['ratio'] else None,
            'target_capacity_prediction_tokens': target['capacity_prediction_tokens'],
            'target_unique_capacity_prediction_tokens': target['unique_capacity_prediction_tokens'],
            'target_actual_unique_prediction_tokens': target['unique_prediction_tokens_exposed'],
            'target_repeated_prediction_token_exposures': target['repeated_prediction_token_exposures'],
            'replay_bank_capacity_prediction_tokens': expected_capacity,
            'replay_bank_unique_capacity_prediction_tokens': summary['replay_bank_unique_prediction_tokens'],
            'replay_actual_unique_prediction_tokens': replay['unique_prediction_tokens_exposed'] if replay else 0,
            'replay_repeated_prediction_token_exposures': replay['repeated_prediction_token_exposures'] if replay else 0,
            **{field: counters[field] for field in COUNTS},
            **{kind + '_student_forward_calls_derived': derived[kind]['student_forward_calls_derived'] for kind in ['target', 'replay']},
            'replay_teacher_forward_calls_derived': derived['replay']['student_forward_calls_derived'] if config['replay_objective'] == 'kl' else 0,
            **{field: counters[field] for field in COSTS},
            'elapsed_seconds_excluding_setup': counters['elapsed_seconds'],
            'total_accounted_wall_seconds': counters['total_accounted_wall_seconds'],
            'disjoint_major_timer_sum_seconds': sum(counters[field] for field in ['setup_seconds', 'training_seconds', 'validation_seconds', 'checkpoint_seconds'])}
        record['wall_minus_major_timer_sum_seconds'] = record['total_accounted_wall_seconds'] - record['disjoint_major_timer_sum_seconds']
        # Frozen train.py samples elapsed() separately for these two fields.
        record['total_wall_sampling_gap_seconds'] = counters['total_accounted_wall_seconds'] - (
            counters['setup_seconds'] + counters['elapsed_seconds'])
        require(abs(record['total_wall_sampling_gap_seconds']) < .01,
                f'Adjacent total-wall samples disagree by over10ms: {name}')
        rows.append(record)
    by_ratio = aggregate(rows, ['replay_ratio_planned'])
    grouped = aggregate(rows, ['group', 'replay_ratio_planned'])
    for source in sources:
        require(sha256(source['path']) == source['sha256'], f'Input changed during report: {source["path"]}')
    args.output.mkdir(parents=True)
    for filename, records in [('runs36.csv', rows), ('grouped_summary.csv', grouped), ('ratio_summary.csv', by_ratio)]:
        csv_file(args.output / filename, records)
    checks = {'complete36_frozen_conditions': True, 'all_prediction_budgets_exact': True,
              'all6_input_count_columns_match_complete_step_events': True,
              'all6_input_count_columns_match_independent_uniform_row_reconstruction': True,
              'same_b_all_groups_and_seeds_have_identical_input_counts': True,
              'capacity_actual_unique_and_repeated_exposure_separated': True,
              'unique_exposure_no_duplicate_assumptions_verified': True,
              'teacher_subtimers_not_added_to_parent_timers': True,
              'scientific_core_unchanged': True}
    result = {'schema_version': 1, 'status': 'passed', 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physical_training_runs': 36, 'checks': checks, 'source_inputs': sources,
              'grouped_conditions': len(grouped), 'ratio_groups': len(by_ratio),
              'ratio_count_invariants': [{key: value for key, value in group.items() if key in
                  ['replay_ratio_planned', 'physical_runs', *COUNTS, *COVERAGE,
                   'target_student_forward_calls_derived', 'replay_student_forward_calls_derived']}
                  for group in by_ratio],
              'independent_reconstruction': reconstruction,
              'reconstruction_scope': 'All rows length512, all input positions valid, micro16/accum4, full causal prefixes and persistent partial offsets; scalar arithmetic, not model inference.',
              'timer_policy': 'training_teacher_forward_seconds is inside training_seconds; validation teacher/KL arithmetic are inside validation_seconds; sum only disjoint major categories, never teacher subitems again.',
              'count_units': 'prediction=loss-bearing shifted valid positions; raw=sum attention_mask in student input batches with repeated context; padded=sum input_ids.numel() in student batches. Teacher repeats are not an additional raw/padded counter. None is FLOPs.',
              'lambda_over_b_policy': 'Undefined for B0; descriptive mean-loss normalization coefficient ratio, not an observed gradient contribution or compute budget.',
              'performance_inference': 'Only observed frozen runs; no causal runtime claim, no FLOPs measurement.',
              'training_or_gpu_work_started': False, 'additional_test_or_behavior_scores_read': False}
    result['outputs'] = [proof(args.output / filename) for filename in ['runs36.csv', 'grouped_summary.csv', 'ratio_summary.csv']]
    atomic_json(args.output / 'budget_audit.json', result)
    print(json.dumps({'status': 'passed', 'physical_runs': 36, 'grouped_conditions': len(grouped),
                      'ratio_groups': len(by_ratio), 'output': str(args.output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=ROOT / 'configs/formal_deterministic_registry.json')
    parser.add_argument('--audit', type=Path, default=ROOT / 'results/formal_run_audit.json')
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'artifacts/provenance/formal_deterministic_input_snapshot.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/training_budget_primary36')
    export(parser.parse_args())
