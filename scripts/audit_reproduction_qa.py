#!/usr/bin/env python3
"""Audit an archive-restored four-condition engineering QA plan without training."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import reproduce_experiments as helper


def read(path):
    return json.loads(Path(path).read_text())


def proof(path):
    path = Path(path)
    return {'path': str(path), 'sha256': helper.sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(source_workspace, registry_path, weight_manifest_path):
    """Read only frozen metadata and input bytes; never inspect scored outputs."""
    source_workspace = Path(source_workspace).resolve()
    rows = read(registry_path)
    require(len(rows) == 4, 'Engineering QA must contain exactly four jobs')
    require({row['group'] for row in rows} == {'B1', 'B2', 'B3', 'B4'}, 'QA must cover the complete source/objective 2x2')
    require(len({row['id'] for row in rows}) == 4, 'Duplicate reproduction run ID')
    manifests = {row['reproduction_manifest'] for row in rows}
    require(len(manifests) == 1, 'One explicit relocation manifest is required')
    relocation_path = Path(next(iter(manifests)))
    relocation = read(relocation_path)
    require(helper.sha(registry_path) == relocation['registry_sha256'], 'Relocated registry SHA mismatch')
    workspace, scratch = Path(relocation['new_workspace']), Path(relocation['new_scratch'])
    require(workspace.resolve() == source_workspace, 'This QA must execute entirely within the restored workspace')
    original_registry_path = source_workspace / 'configs/formal_deterministic_registry.json'
    original_registry = {row['id']: row for row in read(original_registry_path)}
    snapshot_path = source_workspace / 'artifacts/provenance/formal_deterministic_input_snapshot.json'
    freeze_path = source_workspace / 'artifacts/provenance/formal_deterministic_repository_freeze.json'
    snapshot, freeze = read(snapshot_path), read(freeze_path)
    catalog, _, configs = helper.hash_catalog([snapshot_path])
    weights = read(weight_manifest_path)
    require(helper.sha(original_registry_path) == weights['registry_sha256'], 'Original registry differs from accepted weight export')
    original_weights = {row['condition_run_id']: row for row in weights['logical_runs']}
    verified = {}
    for path, expected in relocation['verified_inputs'].items():
        helper.verify_file(path, expected, verified)
    core = []
    for relative in sorted(helper.CORE_FILES):
        restored = source_workspace / relative
        original_sha = snapshot['code'][relative]['sha256']
        require(helper.sha(restored) == original_sha == freeze['core_sources'][relative]['current'],
                f'Frozen scientific core SHA mismatch: {relative}')
        core.append({'relative_path': relative, 'restored_path': str(restored), 'sha256': original_sha})
    helper_path = source_workspace / 'scripts/reproduce_experiments.py'
    require(helper.sha(helper_path) == helper.sha(ROOT / 'scripts/reproduce_experiments.py'),
            'Archived helper differs; an explicit reviewed repair record is required')
    queue_path = Path(relocation['queue'])
    require(helper.sha(queue_path) == relocation['queue_sha256'], 'QA queue SHA mismatch')
    queue = read(queue_path)
    require(len(queue) == 4 and {job['id'] for job in queue} == {row['id'] for row in rows},
            'QA queue must contain precisely the four relocated runs')
    job_by_id = {job['id']: job for job in queue}
    inputs, comparisons = {}, []
    coverage = set()
    for row in rows:
        group = row['group']
        expected_id = f'formal_deterministic_{group}_s11_b0p05_w0p05'
        require(row['reproduction_of']['run_id'] == expected_id, 'QA selection differs from predefined engineering coverage')
        original_row = original_registry[expected_id]
        source_config_path = source_workspace / helper.workspace_key(original_row['config_path'])
        source_config, config = read(source_config_path), read(row['config_path'])
        config_sha = helper.sha(source_config_path)
        require(config_sha == configs[helper.workspace_key(original_row['config_path'])], 'Original frozen config SHA mismatch')
        require(config_sha == row['reproduction_of']['config_sha256'] == original_weights[expected_id]['config_sha256'],
                'Original config disagrees with accepted final-weight metadata')
        require(helper.sha(row['config_path']) == row['config_sha256'], 'Relocated config SHA mismatch')
        require(helper.scientific_parameters(source_config) == helper.scientific_parameters(config),
                'Relocation changed scientific parameters')
        require(source_config == original_weights[expected_id]['original_checkpoint_metadata']['config'],
                'Original checkpoint metadata differs from the frozen source config')
        require(config['seed'] == 11 and config['replay_ratio'] == .05 and config['replay_weight'] == .05,
                'QA seed/ratio/weight differs from the predefined 2x2')
        require(config['target_prediction_tokens'] == 10_000_000 and config['resume_from'] is None,
                'QA must restart the full original ten-million-token horizon')
        require(not Path(config['output_dir']).exists(), 'QA output already exists; no dry-run-only claim allowed')
        require(Path(config['output_dir']).is_relative_to(scratch / 'runs'), 'QA output escapes its independent scratch namespace')
        require(config['run_label'] == row['id'] != expected_id, 'QA requires explicit new identities')
        source = 'c4' if group in {'B1', 'B2'} else 'generated'
        objective = 'ce' if group in {'B1', 'B3'} else 'kl'
        require(config['replay_objective'] == objective, 'Source/objective coverage mismatch')
        replay_name = 'c4_random_seed11.pt' if source == 'c4' else 'generated_random_seed11.pt'
        require(Path(config['replay_path']).name == replay_name, 'Replay source bank mismatch')
        coverage.add((source, objective))
        mapped = helper.relocate_value(source_config, workspace, scratch)
        for field in ['model_name_or_path', 'target_path', 'replay_path', 'validation_paths']:
            require(config[field] == mapped[field], f'Input path mapping differs from the explicit relocation: {field}')
        require(row['test_paths'] == helper.relocate_value(original_row['test_paths'], workspace, scratch),
                'Test byte paths differ from the original registry mapping')
        path_pairs = [('target', source_config['target_path'], config['target_path']),
                      ('replay', source_config['replay_path'], config['replay_path'])]
        path_pairs += [('validation:' + name, old, config['validation_paths'][name])
                       for name, old in source_config['validation_paths'].items()]
        path_pairs += [('test_bytes_only:' + name, old, row['test_paths'][name])
                       for name, old in original_row['test_paths'].items()]
        mapped_inputs = []
        for role, old, new in path_pairs:
            relative = helper.workspace_key(old)
            require(relative in catalog, f'Frozen data SHA absent: {relative}')
            helper.verify_file(new, catalog[relative], verified)
            item = {'role': role, 'original_path': old, 'restored_path': new,
                    'sha256': catalog[relative], 'relative_path': relative}
            adjacent_relative = str(Path(relative).with_suffix('.json'))
            if adjacent_relative in catalog:
                adjacent = Path(new).with_suffix('.json')
                helper.verify_file(adjacent, catalog[adjacent_relative], verified)
                item['adjacent_manifest'] = proof(adjacent)
            mapped_inputs.append(item)
            inputs[relative] = item
        job = job_by_id[row['id']]
        require(job['argv'][1:] == [str(workspace / 'scripts/make_experiments.py'), 'run',
                '--config', row['config_path'], '--summary-copy', row['summary_path']],
                'QA job argv does not execute the exact relocated config')
        require(job['env']['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8', 'Strict cuBLAS environment missing')
        expected_weights = original_weights[expected_id]
        provenance = expected_weights['original_checkpoint_metadata']['provenance']
        execution = {key: value for key, value in provenance.items()
                     if key not in {'target', 'replay', 'source_sha256', 'device'}}
        comparisons.append({'group': group, 'source': source, 'objective': objective,
            'training_seed': 11, 'new_scientific_seed': False, 'reproduction_run_id': row['id'],
            'reproduction_of': expected_id, 'source_config': proof(source_config_path),
            'relocated_config': proof(row['config_path']), 'scientific_parameters_identical': True,
            'changed_fields': sorted(key for key in source_config if source_config[key] != config[key]),
            'mapped_inputs': mapped_inputs,
            'expected_final_model_state_fingerprint': expected_weights['model_fingerprint'],
            'original_checkpoint_sha256': expected_weights['source_checkpoint_sha256'],
            'original_weights_archive_member': expected_weights['archive_member'],
            'required_original_execution_provenance': execution,
            'planned_final_checkpoint': str(Path(row['output_dir']) / 'checkpoints/latest.pt'),
            'comparison_rule': 'Full model_state fingerprint; serialized checkpoint SHA may differ because paths/config/optimizer metadata differ.'})
    require(coverage == {(source, objective) for source in ['c4', 'generated'] for objective in ['ce', 'kl']},
            'The QA plan is not the complete engineering 2x2')
    return {'schema_version': 1, 'status': 'passed_scientific_identity_and_paths_dry_run_only',
        'audited_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'runs': comparisons,
        'source_registry': proof(original_registry_path), 'relocated_registry': proof(registry_path),
        'input_snapshot': proof(snapshot_path), 'repository_freeze': proof(freeze_path),
        'accepted_weight_manifest': proof(weight_manifest_path), 'queue': proof(queue_path),
        'relocation_manifest': proof(relocation_path), 'archive_helper': proof(helper_path),
        'scientific_core_files': core, 'verified_inputs': verified,
        'verified_input_count': len(verified), 'unique_dataset_tensor_paths': sorted(inputs),
        'source_objective_coverage': [['c4', 'ce'], ['c4', 'kl'], ['generated', 'ce'], ['generated', 'kl']],
        'training_started': False, 'gpu_work_started': False, 'test_or_behavior_scores_read': False,
        'retraining_bitwise_reproduction_verified': False, 'new_scientific_replicates': 0,
        'producer_source': proof(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-workspace', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--weight-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite an earlier engineering QA audit')
    result = audit(args.source_workspace, args.registry, args.weight_manifest)
    helper.atomic_json(args.output, result)
    print(json.dumps({'status': result['status'], 'runs': len(result['runs']),
                      'verified_inputs': result['verified_input_count'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
