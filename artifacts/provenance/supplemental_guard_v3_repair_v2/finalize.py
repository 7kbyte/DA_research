"""Freeze the independent repair and run only synthetic CPU regression tests."""
import ast
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from replay_study import supplemental_analysis_guard_v3 as gate

OUT = Path(__file__).resolve().parent


def node(path, name):
    return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def canonical_ast(value):
    return ast.dump(value, include_attributes=False)


def body_text(path, name):
    n = node(path, name)
    return '\n'.join(Path(path).read_text().splitlines()[n.lineno - 1:n.end_lineno]) + '\n'


def write_text_new(path, text):
    with Path(path).open('x') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    protocol = gate.proof(ROOT / 'docs/supplemental_aux_analysis_protocol_v2.json')
    plan_path = ROOT / 'artifacts/analysis_prepared_v2/extended_pythia_410m_100m_nll.json'
    plan = gate.read(plan_path)  # Prepared metadata only, never an output/score.
    retained_paths = [*gate.RETAINED_SOURCES, 'docs/supplemental_aux_analysis_protocol_v2.json',
                      str(plan_path.relative_to(ROOT)),
                      'artifacts/analysis_prepared_v2/extended_pythia_410m_100m_nll_root_authority_TEMPLATE.json']
    before = {p: gate.sha(ROOT / p) for p in retained_paths}
    repair = {
        'schema_version': 3, 'operational_revision': 3,
        'kind': 'supplemental_analysis_guard_operational_bugfix',
        'created_at_utc': gate.now(),
        'retained_protocol': protocol,
        'retained_plan': gate.proof(plan_path),
        'retained_manifest': plan['manifest'],
        'retained_source_sha256': {str(ROOT / p): gate.sha(ROOT / p) for p in gate.RETAINED_SOURCES},
        'active_source_sha256': {str(ROOT / p): gate.sha(ROOT / p) for p in gate.ACTIVE_SOURCES},
        'scientific_or_statistics_changes': False,
        'score_access_authorized': False,
        'first_read_rule': 'exact_retained_v2_plan_manifest_namespace',
        'reason': 'C complete-state producer_source points to Python source; v2 incorrectly JSON-parsed this opaque SHA-bound file.',
        'changes': [
            'Only producer_source is opaque SHA bound; present source_audit/source_registry/source_snapshot stay JSON parsed and schema checked.',
            'Every scheduler closure completion requires exact integer rc0, absence of stop-request keys, and actual recorded wrapper argv equal to the frozen definition.',
            'Historical completed-checkpoint cache authority also requires exact integer rc0 and absence of stop-request keys.',
            'Retained load_plan/fresh preparation, explicit checkpoint allowlist, frozen cache snapshot, worker/collector/statistics, and first-read path remain unchanged.',
            'Future ROOT authority must bind this operational_repair proof in addition to all retained plan/protocol/manifest/cache fields.'
        ],
        'actual_check_or_analyze_executed_by_repair_preparation': False,
        'GPU_inference_or_training_started': False,
        'real_score_semantics_read': False,
        'actual_first_read_journal_path_preserved': plan['first_read_receipt_path'],
        'actual_plan_at_repair': gate.proof(plan_path),
        'this_is_not_a_ROOT_score_authority': True,
    }
    repair_proof = gate.immutable(ROOT / 'docs/supplemental_aux_analysis_operational_repair_v3_C410.json', repair)

    old = node(ROOT / gate.RETAINED_SOURCES[1], 'analyze')
    actual_new = node(ROOT / gate.ACTIVE_SOURCES[1], 'analyze')
    normalized = copy.deepcopy(actual_new)
    assert normalized.args.args.pop(4).arg == 'repair_proof'
    changes = []
    for item in ast.walk(normalized):
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and isinstance(item.func.value, ast.Name):
            if item.func.value.id == 'guard' and item.func.attr in ('authorize', 'inspect'):
                assert isinstance(item.args[-1], ast.Name) and item.args[-1].id == 'repair_proof'
                item.args.pop()
                changes.append(item.func.attr + '_explicit_repair_argument')
        if isinstance(item, ast.Dict):
            for i in reversed(range(len(item.keys))):
                if isinstance(item.keys[i], ast.Constant) and item.keys[i].value == 'operational_repair':
                    assert isinstance(item.values[i], ast.Name) and item.values[i].id == 'repair_proof'
                    item.keys.pop(i); item.values.pop(i)
                    changes.append('COMPLETE_receipt_operational_repair_metadata')
    assert sorted(changes) == sorted(['authorize_explicit_repair_argument', 'inspect_explicit_repair_argument', 'COMPLETE_receipt_operational_repair_metadata'])
    assert canonical_ast(normalized) == canonical_ast(old)
    old_helper = node(ROOT / gate.RETAINED_SOURCES[1], 'write_auxiliary')
    new_helper = node(ROOT / gate.ACTIVE_SOURCES[1], 'write_auxiliary')
    assert canonical_ast(old_helper) == canonical_ast(new_helper)
    ast_proof = gate.immutable(OUT / 'analyze_AST_equivalence.json', {
        'schema_version': 1, 'status': 'passed', 'operational_repair': repair_proof,
        'old_CLI': gate.proof(ROOT / gate.RETAINED_SOURCES[1]),
        'new_CLI': gate.proof(ROOT / gate.ACTIVE_SOURCES[1]),
        'allowed_differences': ['Function argument repair_proof before the retained optional arguments', *changes],
        'normalized_analyze_AST_equal': True,
        'old_analyze_AST_sha256': hashlib.sha256(canonical_ast(old).encode()).hexdigest(),
        'normalized_new_analyze_AST_sha256': hashlib.sha256(canonical_ast(normalized).encode()).hexdigest(),
        'write_auxiliary_AST_equal': True,
        'bootstrap_replicates_retained': 5000, 'bootstrap_seed_retained': 20261106,
        'all_statistical_calls_imports_constants_loops_and_order_retained': True,
        'real_scores_parsed': False,
    })
    diff = ''.join(difflib.unified_diff(
        body_text(ROOT / gate.RETAINED_SOURCES[1], 'analyze').splitlines(keepends=True),
        body_text(ROOT / gate.ACTIVE_SOURCES[1], 'analyze').splitlines(keepends=True),
        fromfile='v2_retained_analyze', tofile='v3_analyze'))
    write_text_new(OUT / 'analyze_function.diff', diff)

    # These existing suites and the new suite use only temporary synthetic
    # outputs; their deliberately invalid JSON score bytes remain sealed.
    argv = [sys.executable, '-m', 'pytest', '-q',
            'tests/test_supplemental_analysis_guard_v3.py',
            'tests/test_supplemental_analysis_cache_v2.py',
            'tests/test_supplemental_analysis_guard.py']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    start_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with (OUT / 'synthetic_tests_stdout.log').open('xb') as stdout, (OUT / 'synthetic_tests_stderr.log').open('xb') as stderr:
        child = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - started
    end_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    timing = gate.immutable(OUT / 'synthetic_tests_whole_child_timing.json', {
        'argv': argv, 'returncode': child.returncode,
        'whole_child_process_wall_seconds': elapsed,
        'whole_child_user_cpu_seconds': end_usage.ru_utime - start_usage.ru_utime,
        'whole_child_sys_cpu_seconds': end_usage.ru_stime - start_usage.ru_stime,
        'scope': 'Spawn through exit of this CPU synthetic pytest child only; not whole repair engineering/review time.',
        'stdout': gate.proof(OUT / 'synthetic_tests_stdout.log'),
        'stderr': gate.proof(OUT / 'synthetic_tests_stderr.log'),
        'real_phase_check_or_analysis': False,
    })
    if child.returncode != 0:
        raise RuntimeError('Synthetic tests failed; retained files remain untouched')
    after = {p: gate.sha(ROOT / p) for p in retained_paths}
    assert after == before
    # Bind all pre-existing statistical/worker source authorities using only
    # explicitly enumerated source files from the frozen protocol (no data/CP).
    frozen = gate.read(protocol['path'])['analysis_source_sha256']
    source_check = {}
    for path, expected in frozen.items():
        actual = gate.sha(path)
        if actual != expected:
            raise ValueError('Pre-existing frozen source differs: ' + path)
        source_check[path] = actual
    completion = gate.immutable(OUT / 'completion.json', {
        'schema_version': 1, 'status': 'passed', 'created_at_utc': gate.now(),
        'operational_repair': repair_proof, 'analyze_AST_equivalence': ast_proof,
        'analyze_function_diff': gate.proof(OUT / 'analyze_function.diff'),
        'synthetic_tests_whole_child_timing': timing,
        'test_source': gate.proof(ROOT / 'tests/test_supplemental_analysis_guard_v3.py'),
        'initial_builder_source': gate.proof(ROOT / 'artifacts/provenance/supplemental_guard_v3_repair_v1/build.py'), 'finalizer_source': gate.proof(__file__),
        'superseded_initial_receipt': gate.proof(ROOT / 'artifacts/provenance/supplemental_guard_v3_repair_v1/SUPERSEDED.json'),
        'retained_files_unchanged_sha256': after,
        'retained_protocol_analysis_sources_all_match': source_check,
        'retained_plan': gate.proof(plan_path),
        'retained_first_read_receipt_path': plan['first_read_receipt_path'],
        'actual_first_read_receipt_exists': Path(plan['first_read_receipt_path']).exists(),
        'actual_score_authority_issued': False,
        'actual_phase_check_or_analyze_executed': False,
        'actual_scores_or_predictions_parsed': False,
        'GPU_or_new_scheduler_started': False,
        'unknown_costs': ['Engineering/imports/documentation/first unrecorded synthetic pytest invocation/independent review parent time are not included in the measured test-child timer.'],
    })
    print(json.dumps({'completion': completion, 'operational_repair': repair_proof, 'AST': ast_proof}))


if __name__ == '__main__':
    main()
