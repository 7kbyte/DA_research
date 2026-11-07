#!/usr/bin/env python3
"""Static source/manifest review only: never imports or calls guard/check/collect."""
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/primary_behavior_analysis_static_review_v1'


def main():
    if OUT.exists():
        raise FileExistsError('New immutable review namespace required')
    blobs, proofs, trees = {}, {}, {}
    def source(name, path, expected=None):
        path = Path(path); raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError('Pinned source changed: ' + str(path))
        blobs[name] = raw; proofs[name] = {'path': str(path.resolve()), 'sha256': digest, 'bytes': len(raw)}
        if path.suffix == '.py':
            trees[name] = ast.parse(raw)
        return json.loads(raw) if path.suffix == '.json' else raw
    manifest = source('manifest.json', ROOT / 'docs/primary_evaluation_v2_execution_matched_behavior_manifest.json',
        '312d8e150030d489593e7e705699ec1da5f7d505e349ecd15a978eca2c2f7934')
    queue = source('frozen_queue.json', ROOT / 'configs/primary_evaluation_v2_execution_matched_behavior_queue.json',
        '393f3df2f0213e2e4e4d2387109b750825c797b909f4b05d16cd3cb739f6aaa9')
    for name, relative in {
        'guard.py': 'scripts/guarded_behavior_analysis.py', 'inventory.py': 'src/replay_study/evaluation_cost_inventory.py',
        'collect.py': 'scripts/analyze_behavioral_plan.py', 'worker.py': 'scripts/run_frozen_evaluation.py',
        'analysis.py': 'src/replay_study/behavioral_analysis.py',
    }.items():
        source(name, ROOT / relative)
    common = []
    for index, (path, digest) in enumerate(manifest['common_input_sha256'].items()):
        source('common_' + str(index) + '_' + Path(path).name, path, digest)
        common.append({'path': path, 'sha256': digest, 'bytes': Path(path).stat().st_size})
    assert len(common) == 8 and all(Path(row['path']).suffix in ('.py', '.json') for row in common)
    assert len(manifest['jobs']) == len(queue) == 67 and len(manifest['records']) == 72
    jobs = {job['id']: job for job in manifest['jobs']}
    assert len(jobs) == 67 and set(jobs) == {job['id'] for job in queue}
    aliases = [row['id'] for row in manifest['records'] if Path(row['output']).resolve() != Path(jobs[row['inference_job_id']]['output']).resolve()]
    assert len(aliases) == 6
    functions = {}
    for name, tree in trees.items():
        functions[name] = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    def calls(name, function):
        return sorted({node.func.id if isinstance(node.func, ast.Name) else node.func.attr for node in ast.walk(functions[name][function])
                       if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))})
    for name, function in [('guard.py', 'check'), ('guard.py', 'analyze'), ('collect.py', 'collect'), ('inventory.py', 'inventory_evaluations')]:
        assert 'verified_job' not in calls(name, function)
    assert {'inventory_evaluations', 'verify_outputs'} <= set(calls('guard.py', 'check'))
    assert {'check', 'collect', 'write_bundle'} <= set(calls('guard.py', 'analyze'))
    assert {'verify_outputs', 'validate_result'} <= set(calls('collect.py', 'collect'))
    assert 'verified_job' in calls('worker.py', 'main')
    assert 'scheduler_state' not in blobs['guard.py'].decode() and 'journal' not in blobs['guard.py'].decode()
    for function in ('verify_outputs', 'validate_result'):
        assert 'verified_job' not in calls('worker.py', function)
    graph = [
        ('guard.py', 'check', 'manifest + eight small common sources; inventory_evaluations verifies attestation/output SHA and baseline verify_outputs; no scheduler_state supplied'),
        ('inventory.py', 'inventory_evaluations', 'reads manifests/attestations; hash_output streams only expected result/prediction files; checkpoint SHA compared as metadata'),
        ('inventory.py', 'hash_output', 'result bytes streamed for integrity, no score JSON deserialization in this helper'),
        ('guard.py', 'analyze', 'reads matching ROOT authorization, repeats check, then collect and write_bundle; no access journal is emitted'),
        ('collect.py', 'collect', 'after guard: verify_outputs + validate_result, then read score JSON/task/prediction rows and frozen bootstrap/statistics'),
        ('worker.py', 'verify_outputs', 'reads execution attestation and streams its output SHA set; no checkpoint input hashing'),
        ('worker.py', 'validate_result', 'reads result JSON metadata/split/policy and compares checkpoint path string, not checkpoint payload'),
        ('worker.py', 'verified_job', 'GPU dispatcher preflight merges common_input_sha256 + job input_sha256 and fresh hashes all input files, including CP'),
        ('worker.py', 'main', 'actual inference worker calls verified_job; this distinct input-verification path is retained'),
    ]
    result = {'schema_version': 1, 'status': 'static_callgraph_scope_review_only_no_behavior_execution',
        'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'source_proofs': proofs,
        'manifest_common_input_files': common, 'common_input_count': 8,
        'common_input_total_small_file_bytes': sum(row['bytes'] for row in common),
        'manifest_inference_jobs': 67, 'logical_endpoints': 72, 'logical_alias_endpoints': 6,
        'callgraph': [{'source': name, 'function': function, 'line': functions[name][function].lineno,
                       'end_line': functions[name][function].end_lineno, 'scope': scope} for name, function, scope in graph],
        'static_expected_checkpoint_payload_read_calls_on_guard_check_and_analysis_collect_path': 0,
        'actual_primary_behavior_analysis_run_observed': False, 'measured_actual_analysis_checkpoint_bytes': None,
        'measured_actual_primary_analysis_wall_or_CPU_seconds': None,
        'old_233GB_summary_claim_was_not_a_measured_analysis_operation': True,
        'historical_cost_record_or_inventory_changed': False,
        'GPU_worker_fresh_full_input_SHA_path_unchanged': True,
        'guard_limitations': {'scheduler_state_not_supplied': True, 'scheduler_full_completion_not_enforced_by_current_guard': True,
            'first_semantic_access_journal_not_emitted': True, 'original_analyze_rechecks_before_collect': True},
        'future_ROOT_checklist': [
            'Before any score semantics, independently verify all67 frozen job definitions in the actual scheduler state provenance: completed/rc0/no stop_requested/no missing_outputs, including inherited initial baseline.',
            'Independently bind manifest/queue/source SHA, complete initial task output set and all72 logical endpoints including6 alias attestations and exact checkpoint/model-state identities. No CP payload read needed for these stored-identity comparisons.',
            'Only after independent complete-state proof ROOT signs matching analysis authority: allow_score_analysis=true, manifest SHA, expected_inference_jobs67, expected_logical_endpoints72, proof/source hashes.',
            'Run original check only after later explicit authorization; this current review does not run it. Save ready/full-integrity result and bind its SHA.',
            'After successful check, exclusively create a new semantic_access_enabled first-read journal with manifest/queue/state-proof/authority/source/check receipt SHA. Fail if journal exists; preserve failed attempts rather than overwrite. Journal is authorization/attempt evidence, not proof that parsing completed.',
            'Then call original guarded analyze with matching authority in a new output directory; it repeats check and performs original collect/statistics without changes.',
            'Use Bash built-in time around the complete automated check + exclusive journal + original analyze compound command, preserving real/user/sys and exit status. Nested bootstrap/guard work is not added to the parent again.',
            'Record successful completion/integrity and real measured wall/CPU afterward; manual preparation, signing and writing without direct timers remain unknown.'] ,
        'this_review_called_check_collect_analyze_or_verify_outputs': False,
        'behavior_outputs_or_predictions_opened': False, 'checkpoint_payload_opened': False,
        'original_helpers_or_scientific_numbers_or_statistics_modified': False, 'GPU_or_watcher_started': False}
    source('producer.py', Path(__file__).resolve())
    result['source_proofs'] = proofs
    OUT.mkdir(); (OUT / 'sources').mkdir()
    for name, raw in blobs.items():
        with (OUT / 'sources' / name).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    target = OUT / 'receipt.json'
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'receipt': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                      'common_small_file_bytes': result['common_input_total_small_file_bytes'], 'static_expected_CP_read_calls': 0,
                      'actual_analysis_run': False}))


if __name__ == '__main__':
    main()
