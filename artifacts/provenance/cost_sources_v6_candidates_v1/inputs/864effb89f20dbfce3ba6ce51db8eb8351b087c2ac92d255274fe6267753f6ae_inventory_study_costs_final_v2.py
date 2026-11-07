#!/usr/bin/env python3
"""Cost-only extension of the unchanged v1 collector with exact receipt copies."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import inventory_study_costs_final as v1
from schedule_jobs import digest_job
from stat_bound_digest_cache import atomic, small_proof


class CostReceiptReader(v1.legacy.Reader):
    """Capture only explicitly registered small cost metadata, never model payloads."""
    def __init__(self):
        super().__init__()
        self.capture_paths = set()
        self.exact_cost_bytes = {}

    def register(self, path):
        path = Path(path).resolve()
        if path.suffix != '.json':
            raise ValueError('Cost receipt sources must be explicit small JSON files')
        self.capture_paths.add(str(path))

    def read(self, path, events=False):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.capture_paths or key in self.cache:
            return super().read(path, events)
        if events:
            raise ValueError('Cost receipt snapshots do not capture event streams')
        if not path.exists():
            return super().read(path)
        before = path.stat()
        if before.st_size > 32 << 20:
            raise ValueError('Registered cost metadata unexpectedly exceeds32MiB')
        raw = path.read_bytes()
        after = path.stat()
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if stamp(before) != stamp(after) or len(raw) != after.st_size:
            raise ValueError('Cost metadata changed while reading; retry a new snapshot')
        value = json.loads(raw)
        self.cache[key] = value
        self.exact_cost_bytes[key] = raw
        self.sources[key] = {'exists': True, 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes_read': len(raw), 'mtime_ns': after.st_mtime_ns, 'changed_while_reading': False,
            'read_purpose': 'explicit_cost_metadata_only', 'captured_exact_bytes': True,
            'snapshot_scope': 'exact observed source bytes; study-wide atomicity not claimed'}
        return value


def bound(reader, source, optional=False):
    reader.register(source['path'])
    return v1.read_bound(reader, source, optional)


def op(identity, source, **extra):
    return {'id': identity, 'source': source, 'category': 'lossless_export',
        'status_pointer': '/status', 'accepted_statuses': ['verified_draft_not_deliverable'],
        'wall_pointer': '/wall_seconds', 'process_cpu_pointer': '/process_cpu_seconds',
        'timing_semantics': 'independent_single_draft_operation',
        'add_to_recorded_wall_totals': True, 'scope': 'Actual completed draft only; whole cohort delivery remains gated.', **extra}


def C_operations(source, reader):
    if small_proof(source['exporter_source']['path']) != source['exporter_source']:
        raise ValueError('The explicitly registered C exporter source changed')
    freeze = bound(reader, source['freeze'])
    phases = {phase['model']: phase for phase in freeze['phases']}
    operations, coverage = [], []
    root = Path(source['root']).resolve()
    progress = {'path': str(root / 'progress.json')}
    operations.append(op('C_export_daemon_context', progress, accepted_statuses=[],
        wall_pointer='/elapsed_wall_seconds', timing_semantics='wall_including_waiting',
        add_to_recorded_wall_totals=False,
        scope='Whole daemon invocation cumulative wall/CPU, including prehash, drafts, assembly, lock/poll waits; nonadditive parent.'))
    seen = set()
    for model_source in source['models']:
        model = model_source['model']
        if model in seen:
            raise ValueError('Duplicate model C cost source')
        seen.add(model)
        phase = phases[model]
        if model_source['registry'] != {'path': phase['active_registry'], 'sha256': phase['active_registry_sha256']}:
            raise ValueError('C cost registry differs from frozen formal source')
        registry = bound(reader, model_source['registry'])
        snapshot = bound(reader, {'path': phase['snapshot'], 'sha256': phase['snapshot_sha256']})
        queue = bound(reader, {'path': snapshot['queue_path'], 'sha256': snapshot['queue_sha256']})
        definitions = {job['id']: job for job in queue}
        rows = {row['id']: row for row in registry}
        if len(rows) != phase['runs'] or set(rows) != set(definitions):
            raise ValueError('C cost source does not cover the complete frozen cohort')
        folder = root / model
        records_dir = folder / 'records'
        if {path.name for path in records_dir.glob('*.json')} - {identity + '.json' for identity in rows}:
            raise ValueError('Unregistered JSON in the explicitly bounded C records folder')
        observed = []
        for identity, row in rows.items():
            proof = {'path': str(records_dir / (identity + '.json'))}
            record = bound(reader, proof, optional=True)
            if record is None:
                continue
            completed = record.get('completed_scheduler_record') or {}
            if (record.get('status') != 'verified_draft_not_deliverable' or record.get('run_id') != identity
                    or record.get('model_tag') != model or record.get('phase_name') != phase['phase']
                    or record.get('data_family') != 'extended_v1' or record.get('role') != 'extended'
                    or record.get('input_snapshot_sha256') != phase['snapshot_sha256']
                    or record.get('source_checkpoint') != row['checkpoint_path']
                    or record.get('definition') != definitions[identity]
                    or record.get('definition_sha256') != digest_job(definitions[identity])
                    or completed.get('status') != 'completed' or completed.get('returncode') != 0
                    or completed.get('missing_outputs') or 'stop_requested_at' in completed
                    or completed.get('definition_sha256') != record['definition_sha256']
                    or record.get('all_nonweight_metadata_roundtrip_verified') is not True):
                raise ValueError('C draft cost source lacks the exact completed run binding')
            calls = record.get('additional_encoding_calls')
            if calls == 1:
                if record.get('bytewise_verification', {}).get('all_tensor_bytes_equal') is not True:
                    raise ValueError('C encoded draft lacks full bytewise verification')
            elif calls == 0:
                if not record.get('alias_of') or record['alias_of'] not in rows:
                    raise ValueError('C zero-encoding draft lacks its explicit canonical record')
            else:
                raise ValueError('Invalid C draft encoding-call counter')
            receipt_sha = reader.sources[str(Path(proof['path']).resolve())]['sha256']
            parent = 'C_draft:' + model + ':' + identity
            operations.append(op(parent, proof, parent_id='C_export_daemon_context',
                canonical_id=parent + ':' + receipt_sha,
                scope='One first-created draft operation, deduplicated by registry ID and immutable record SHA; reused references are not new work.'))
            for label, pointer, cpu_pointer in [('encode', '/encode/wall_seconds', '/encode_process_cpu_seconds'),
                    ('decode', '/decode/wall_seconds', '/decode_process_cpu_seconds'),
                    ('verify', '/bytewise_verification/wall_seconds', '/verification_process_cpu_seconds')]:
                if calls == 1:
                    operations.append(op(parent + ':' + label, proof, parent_id=parent,
                        wall_pointer=pointer, process_cpu_pointer=cpu_pointer,
                        timing_semantics='nested_measured_child', add_to_recorded_wall_totals=False,
                        scope='Nested within the same first-created draft; never added again.'))
            observed.append({'run_id': identity, 'record': {'path': proof['path'], 'sha256': receipt_sha},
                             'additional_encoding_calls': calls, 'deliverable_cohort_claimed': False})
        completion_source = {'path': str(folder / 'completion.json')}
        completion = bound(reader, completion_source, optional=True)
        if completion is not None:
            if (completion.get('status') != 'passed' or completion.get('model') != model
                    or completion.get('phase') != phase['phase'] or completion.get('logical_final_endpoints') != len(rows)
                    or completion.get('fullphase_source_gate_passed') is not True or len(observed) != len(rows)
                    or not completion.get('timer_scope', '').startswith('assembly only')):
                raise ValueError('C assembly is not the complete, separately timed model-specific package')
        operations.append(op('C_assembly:' + model, completion_source, parent_id='C_export_daemon_context',
            accepted_statuses=['passed'], timing_semantics='independent_assembly_only',
            scope='Manifest/base closure/TAR build and verification only; prior draft operations are disjoint. Missing completion remains unknown.'))
        publication = model_source.get('publication_receipt')
        if publication:
            operations.append(op('C_publication:' + model, publication, category='storage_publication',
                accepted_statuses=['passed'], timing_semantics='single_measured_stage',
                scope='Explicit future publication receipt only; never inferred from assembly or reserved capacity.'))
        coverage.append({'model': model, 'phase': phase['phase'], 'registry': model_source['registry'],
            'expected_draft_records': len(rows), 'actual_first_created_records_observed': len(observed),
            'draft_records_missing_or_not_created': len(rows) - len(observed), 'records': observed,
            'assembly_complete': completion is not None,
            'publication_source_status': 'explicitly_registered_optional' if publication else 'not_authorized_source_unassigned',
            'publication_completed_assumed': False})
    if seen != set(phases):
        raise ValueError('Register all three frozen C models explicitly')
    return operations, coverage


def collect(spec, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new immutable v2 trial output directory')
    reader = CostReceiptReader()
    expected = spec['v1_collector']
    if small_proof(expected['path']) != expected:
        raise ValueError('The preserved v1 collector source changed')
    effective = copy.deepcopy(bound(reader, spec['v1_source_spec']))
    effective.setdefault('unmeasured_activity_gaps', []).extend(spec.get('unmeasured_additional_gaps', []))
    for item in effective.get('operations', []):
        reader.register(item['source']['path'])
    for source in spec.get('metadata_sources', []):
        bound(reader, source, source.get('optional', False))
    effective['operations'].extend(copy.deepcopy(spec['additional_operations']))
    extra, coverage = C_operations(spec['C_exports'], reader)
    effective['operations'].extend(extra)
    for item in effective['operations']:
        reader.register(item['source']['path'])
    result = v1.collect(effective, output, reader)
    exact_sources = output / 'exact_cost_sources'
    exact_sources.mkdir()
    copies = []
    for original, raw in sorted(reader.exact_cost_bytes.items()):
        digest = hashlib.sha256(raw).hexdigest()
        destination = exact_sources / (digest + '.json')
        if not destination.exists():
            with destination.open('xb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        elif destination.read_bytes() != raw:
            raise ValueError('Receipt SHA collision')
        copies.append({'original_path': original, 'snapshot_path': str(destination.resolve()),
                       'sha256': digest, 'bytes': len(raw)})
    directory = os.open(exact_sources, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    atomic(output / 'effective_v1_operation_source_spec.json', effective)
    atomic(output / 'C_export_record_coverage.json', coverage)
    result['schema_version'] = 2
    result['cost_only_v2_extension'] = {'source_spec_contract': spec['accounting_contract'],
        'preserved_v1_source_spec': spec['v1_source_spec'], 'preserved_v1_collector': expected,
        'C_export_record_coverage': coverage, 'exact_cost_source_copies': copies,
        'planned_operations_without_source': spec.get('planned_operations_without_source', []),
        'B_NLL_additional_manual_inference_cost_rows': 0,
        'old_v1_or_legacy_semantics_modified': False,
        'source_copies_scope': 'Explicit cost receipts/authorities and bounded C registry records. Legacy training/event/scored-output payloads are not copied again.',
        'C_daemon_or_grower_wait_assumed_CPU_work': False,
        'B_whole_CPU_analysis_wall_inferred_from_timestamps': False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started, cpu_started = time.perf_counter(), time.process_time()
    source_spec_raw = args.sources.read_bytes()
    source_spec_sha = hashlib.sha256(source_spec_raw).hexdigest()
    spec = json.loads(source_spec_raw)
    result = collect(spec, args.output)
    if small_proof(args.sources)['sha256'] != source_spec_sha:
        raise ValueError('Source spec changed during collection')
    with (args.output / 'source_spec_v2.json').open('xb') as stream:
        stream.write(source_spec_raw)
        stream.flush()
        os.fsync(stream.fileno())
    for key in ('operational_costs', 'operational_category_totals', 'training_cohort_coverage'):
        v1.legacy.csv_write(args.output / (key + '.csv'), result[key])
    base = result['legacy_method_and_preparation_ledger']
    for key in ('physical_runs', 'evaluation_executions', 'evaluation_canonical_inferences', 'evaluation_category_totals'):
        v1.legacy.csv_write(args.output / (key + '.csv'), base[key])
    result['collector_execution'] = {'source': small_proof(__file__), 'source_spec': small_proof(args.sources),
        'wall_seconds': time.perf_counter() - started, 'process_cpu_seconds': time.process_time() - cpu_started,
        'self_timer_not_recursively_added_to_ledger': True}
    atomic(args.output / 'inventory.json', result)
    receipt = {'schema_version': 1, 'status': 'passed_observed_snapshot_not_whole_study_complete',
        'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'inventory': small_proof(args.output / 'inventory.json'),
        'producer': small_proof(__file__), 'source_spec': small_proof(args.sources),
        'exact_cost_source_copies': result['cost_only_v2_extension']['exact_cost_source_copies'],
        'C_export_record_coverage': [{key: value for key, value in row.items() if key != 'records'}
                                   for row in result['cost_only_v2_extension']['C_export_record_coverage']],
        'whole_study_complete': False, 'score_semantics_read': False, 'GB_model_or_archive_payloads_rehashed': False}
    atomic(args.output / 'collection_receipt.json', receipt)
    print(json.dumps({'inventory': receipt['inventory'], 'receipt': small_proof(args.output / 'collection_receipt.json'),
        'C_observed_drafts': sum(row['actual_first_created_records_observed'] for row in receipt['C_export_record_coverage']),
        'attested_evaluation_forwards': sum(row['recorded_additional_model_inference'] for row in base['evaluation_executions'])}))


if __name__ == '__main__':
    main()
