"""Only synthetic data: complete-state Python producer and sealed invalid JSON."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from replay_study import supplemental_analysis_guard_v2 as retained
from replay_study import supplemental_analysis_guard_v3 as gate


ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gate.encoded(value))
    return path


def repair(tmp_path, plan):
    value = {
        'schema_version': 3, 'operational_revision': 3,
        'kind': 'supplemental_analysis_guard_operational_bugfix',
        'retained_protocol': gate.read(plan)['protocol'],
        'scientific_or_statistics_changes': False,
        'score_access_authorized': False,
        'first_read_rule': 'exact_retained_v2_plan_manifest_namespace',
        'retained_source_sha256': {str(ROOT / p): gate.sha(ROOT / p) for p in gate.RETAINED_SOURCES},
        'active_source_sha256': {str(ROOT / p): gate.sha(ROOT / p) for p in gate.ACTIVE_SOURCES},
    }
    return gate.proof(write(tmp_path / 'repair.json', value))


def prepared(tmp_path, mode='nll', both=False, complete_state=True, with_cache=True):
    cache_fixture = module(ROOT / 'tests/test_supplemental_analysis_cache_v2.py', 'v3_cache_fixture')
    f = module(ROOT / 'tests/test_supplemental_analysis_guard.py', 'v3_whole_fixture')
    ready, protocol, state, mp, qp, manifest = f.fixture(tmp_path, alias=True, mode=mode, both=both)
    p = gate.read(protocol)
    p.update(schema_version=2, operational_revision=2,
             checkpoint_cache_reader_source=gate.proof(ROOT / 'scripts/stat_bound_digest_cache.py'))
    p['analysis_source_sha256'][str(Path(retained.__file__).resolve())] = gate.sha(retained.__file__)
    write(protocol, p)
    producer = tmp_path / 'producer.py'
    if complete_state:
        # Actual proof-shaped fields: source_audit/registry/snapshot are JSON;
        # algorithm/producer are real Python files that are never executed.
        producer.write_text('def produce():\n    raise RuntimeError("Never execute proof producer")\n')
        algorithm = tmp_path / 'algorithm.py'
        algorithm.write_text('def fingerprint():\n    raise RuntimeError("Never execute algorithm")\n')
        audit = write(tmp_path / 'audit.json', {
            'status': 'passed', 'expected_runs': 2, 'completed_runs': 2, 'source_violations': []})
        registry = write(tmp_path / 'proof_registry.json', [{'id': 'B4_s11'}, {'id': 'D_s11'}])
        snapshot = write(tmp_path / 'source_snapshot.json', {'status': 'passed'})
        fp = {'sha256_model_state': 'f' * 64, 'state_elements': 2}
        sides = [{'checkpoint': r['checkpoint'], 'checkpoint_sha256': r['checkpoint_sha256'],
                  'model_state_fingerprint': fp} for r in manifest['records'][:2]]
        proof_file = write(tmp_path / 'equivalence.json', {
            'schema_version': 1, 'status': 'passed',
            'algorithm': 'sha256_sorted_key_dtype_shape_contiguous_uint8_v1',
            'algorithm_source': {**gate.proof(algorithm), 'function': 'fingerprint'},
            'producer_source': gate.proof(producer), 'source_audit': gate.proof(audit),
            'source_registry': gate.proof(registry), 'source_snapshot': gate.proof(snapshot),
            'comparisons': [{'endpoint': 'final', 'seed': 11, 'bitwise_model_state_equal': True,
                             'left': sides[0], 'right': sides[1]}]})
        evidence = {
            'proof': str(proof_file), 'proof_sha256': gate.sha(proof_file),
            'comparison_index': 0, 'fingerprint': fp,
            'left_checkpoint_sha256': sides[0]['checkpoint_sha256'],
            'right_checkpoint_sha256': sides[1]['checkpoint_sha256']}
        for r in manifest['records']:
            r.update(weights_identity='model_state:' + fp['sha256_model_state'], source_model_equivalence=evidence)
        manifest['jobs'][1].update(weights_identity='model_state:' + fp['sha256_model_state'], source_model_equivalence=evidence)
        write(mp, manifest)
    queue = gate.read(qp)
    for definition in queue:
        definition['argv'][definition['argv'].index('--manifest-sha256') + 1] = gate.sha(mp)
    write(qp, queue)
    s = gate.read(state)
    for definition in queue:
        s['jobs'][definition['id']].update(argv=definition['argv'], definition_sha256=f.digest_job(definition))
    write(state, s)
    rr = gate.read(ready)
    rr.update(manifest=gate.proof(mp), queue=gate.proof(qp))
    write(ready, rr)
    for job in manifest['jobs']:
        for output in [job['output'], *[r['output'] for r in manifest['records']
                                       if r['inference_job_id'] == job['id'] and r['output'] != job['output']]]:
            ap = gate.attestation_path(output)
            att = gate.read(ap)
            att['manifest_sha256'] = gate.sha(mp)
            if job['kind'] != 'initial' and complete_state:
                att.update(weights_identity=job['weights_identity'], source_model_equivalence=evidence)
            write(ap, att)
    snapshot = cache_fixture.cache_snapshot(tmp_path, [r['checkpoint'] for r in manifest['records']])[0] if with_cache else None
    plan = write(tmp_path / 'plan_v2.json', retained.prepare(gate.proof(ready), gate.proof(protocol), 'fixture', snapshot))
    return plan, state, manifest, repair(tmp_path, plan), producer


@pytest.mark.parametrize('mode', ['nll', 'behavior'])
def test_real_python_producer_proof_and_full_closure_pass_without_score_parse(tmp_path, monkeypatch, mode):
    plan, state, m, rp, producer = prepared(tmp_path, mode=mode, both=mode == 'nll')
    assert producer.is_file()
    with pytest.raises(json.JSONDecodeError):
        json.loads(producer.read_text())
    private = {j['output'] for j in m['jobs']} | {r['output'] for r in m['records']}
    for output in list(private):
        if mode == 'behavior':
            private.update(str(Path(output).with_name(Path(output).stem + '_' + task + '.jsonl'))
                           for task in m['baseline_groups']['base']['tasks'])
    old_read = retained.read
    def read(path):
        assert str(path) not in private
        assert Path(path) != producer
        return old_read(path)
    monkeypatch.setattr(retained, 'read', read)
    monkeypatch.setattr(gate, 'read', read)
    result = gate.inspect(gate.proof(plan), state, rp)
    assert result['ready'] and result['score_files_hashed'] and not result['score_json_parsed']
    assert result['verification_work']['counters']['checkpoint_cache_hits'] == 2
    assert result['verification_work']['counters']['checkpoint_fresh_bytes'] == 0
    assert gate.read(plan)['first_read_receipt_path'] == retained.load_plan(gate.proof(plan))[0]['first_read_receipt_path']
    assert not Path(gate.read(plan)['first_read_receipt_path']).exists()


def test_v2_exact_failure_and_v3_producer_sha_tamper_rejected(tmp_path):
    plan, state, _, rp, producer = prepared(tmp_path)
    with pytest.raises(json.JSONDecodeError):
        retained.inspect(gate.proof(plan), state)
    assert gate.inspect(gate.proof(plan), state, rp)['ready']
    producer.write_bytes(producer.read_bytes() + b'# changed\n')
    with pytest.raises(ValueError, match='SHA mismatch'):
        gate.inspect(gate.proof(plan), state, rp)


@pytest.mark.parametrize('name,value', [
    ('source_audit', 'not JSON'), ('source_audit', {'status': 'failed'}),
    ('source_audit', {'status': 'passed', 'expected_runs': 2, 'completed_runs': True}),
    ('source_audit', {'status': 'passed', 'expected_runs': 2, 'completed_runs': 1}),
    ('source_audit', {'status': 'passed', 'source_violations': ['changed']}),
    ('source_registry', {'id': 'not-a-list'}), ('source_registry', []),
    ('source_registry', [{'id': 'duplicate'}, {'id': 'duplicate'}]),
    ('source_snapshot', {'status': 'failed'}), ('source_snapshot', []),
])
def test_other_json_sources_still_require_json_and_strict_shapes(tmp_path, name, value):
    source = tmp_path / (name + '.json')
    if value == 'not JSON':
        source.write_text('def not_json(): pass\n')
    else:
        write(source, value)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        gate.weights_sources(gate.Verifier(), {name: gate.proof(source)})


@pytest.mark.parametrize('job_id', ['initial', 'shared'])
@pytest.mark.parametrize('change', ['bool_rc', 'float_rc', 'null_rc', 'string_rc', 'nonzero_rc',
                                   'stop_requested_at', 'stop_requested', 'missing_argv', 'wrong_argv', 'failed'])
def test_strict_scheduler_closure_rejects_before_any_output_hash(tmp_path, monkeypatch, job_id, change):
    plan, state, m, rp, _ = prepared(tmp_path)
    ss = gate.read(state)
    row = ss['jobs'][job_id]
    if change.endswith('_rc'):
        row['returncode'] = {'bool_rc': False, 'float_rc': 0.0, 'null_rc': None, 'string_rc': '0', 'nonzero_rc': 1}[change]
    elif change in ('stop_requested_at', 'stop_requested'):
        row[change] = None  # Presence is enough, including rc0 and null value.
    elif change == 'missing_argv':
        row.pop('argv')
    elif change == 'wrong_argv':
        row['argv'] = ['unfrozen_wrapper']
    else:
        row['status'] = 'failed'
    write(state, ss)
    private = {j['output'] for j in m['jobs']} | {r['output'] for r in m['records']}
    original = retained.sha
    def sha(path):
        assert str(path) not in private
        return original(path)
    monkeypatch.setattr(retained, 'sha', sha)
    with pytest.raises(ValueError):
        gate.inspect(gate.proof(plan), state, rp)
    assert not Path(gate.read(plan)['first_read_receipt_path']).exists()


def test_pending_closure_remains_sealed_and_missing_output_not_assumed_complete(tmp_path):
    plan, state, m, rp, _ = prepared(tmp_path)
    ss = gate.read(state)
    ss['jobs']['shared']['status'] = 'running'
    write(state, ss)
    result = gate.inspect(gate.proof(plan), state, rp)
    assert not result['ready'] and not result['score_files_hashed']
    assert result['verification_work']['counters']['checkpoint_cache_hits'] == 0
    ss['jobs']['shared']['status'] = 'completed'
    write(state, ss)
    Path(m['records'][1]['output']).unlink()
    with pytest.raises(FileNotFoundError):
        gate.inspect(gate.proof(plan), state, rp)


@pytest.mark.parametrize('rc', [False, 0.0, None, '0', 1])
def test_historical_checkpoint_rc_must_be_integer_zero(tmp_path, rc):
    cache_fixture = module(ROOT / 'tests/test_supplemental_analysis_cache_v2.py', 'strict_cache_fixture')
    p = cache_fixture.source(tmp_path)
    h = cache_fixture.reader().StatBoundHashes(directory=tmp_path / 'cache')
    h.digest(p, authority={'kind': 'completed_training_checkpoint', 'returncode': rc})
    snapshot = h.snapshot(tmp_path / 'snapshot.json')
    v = gate.Verifier(cache_fixture.policy(snapshot), {str(p): gate.sha(p)})
    with pytest.raises(ValueError, match='integer rc0'):
        gate.strict_cache_completion_authorities(v)
    assert v.work['checkpoint_fresh_bytes'] == 0


@pytest.mark.parametrize('change', ['missing', 'authorized', 'source_sha', 'protocol', 'journal_rule'])
def test_explicit_non_authorizing_repair_required(tmp_path, change):
    plan, state, _, rp, _ = prepared(tmp_path)
    value = gate.read(rp['path'])
    if change == 'missing':
        value.pop('active_source_sha256')
    elif change == 'authorized':
        value['score_access_authorized'] = True
    elif change == 'source_sha':
        key = next(iter(value['active_source_sha256']))
        value['active_source_sha256'][key] = '0' * 64
    elif change == 'protocol':
        value['retained_protocol']['sha256'] = '0' * 64
    else:
        value['first_read_rule'] = 'new-journal-path'
    changed = gate.proof(write(tmp_path / 'changed_repair.json', value))
    with pytest.raises(ValueError):
        gate.inspect(gate.proof(plan), state, changed)


def authority(tmp_path, plan, rp=None, allowed=True):
    value = gate.read(plan)
    result = {
        'authorized_by': 'root', 'allow_score_analysis': allowed,
        'plan': gate.proof(plan), 'protocol': value['protocol'], 'ready': value['ready'], 'manifest': value['manifest'],
        'contract_id': value['contract_id'], 'expected_counts': value['counts'], 'allow_partial': False,
        'checkpoint_verification': value['checkpoint_verification'],
        'training_methods_grid_selection_horizons_frozen': True, 'test_or_behavior_used_for_selection': False}
    if rp is not None:
        result['operational_repair'] = rp
    return gate.proof(write(tmp_path / 'SYNTHETIC_authority.json', result))


def test_old_authority_cannot_silently_authorize_new_implementation(tmp_path):
    plan, _, _, rp, _ = prepared(tmp_path)
    old = authority(tmp_path, plan)
    with pytest.raises(PermissionError, match='operational repair'):
        gate.authorize(gate.proof(plan), old, rp)
    allowed = authority(tmp_path, plan, rp)
    assert gate.authorize(gate.proof(plan), allowed, rp)[0]['first_read_receipt_path'] == gate.read(plan)['first_read_receipt_path']
    disabled = authority(tmp_path, plan, rp, allowed=False)
    with pytest.raises(PermissionError):
        gate.authorize(gate.proof(plan), disabled, rp)


def test_collector_only_after_same_retained_journal_and_cached_full_gate(tmp_path, monkeypatch):
    plan, state, _, rp, _ = prepared(tmp_path)
    a = authority(tmp_path, plan, rp)
    cli = module(ROOT / 'scripts/analyze_supplemental_phase_v3.py', 'aux_v3_cli_test')
    from replay_study import nll_v2_analysis as nll
    def collect(*args, **kwargs):
        assert Path(gate.read(plan)['first_read_receipt_path']).exists()
        readiness = gate.read(tmp_path / 'analysis/whole_phase_readiness.json')
        assert readiness['operational_repair'] == rp
        assert readiness['verification_work']['counters']['checkpoint_fresh_bytes'] == 0
        assert kwargs['allow_partial'] is False and kwargs['n_bootstrap'] == 5000
        return {'complete': True, 'completed_conditions': 2}
    monkeypatch.setattr(nll, 'collect_scores', collect)
    monkeypatch.setattr(nll, 'write_bundle', lambda *args: None)
    assert cli.analyze(gate.proof(plan), a, state, tmp_path / 'analysis', rp, no_plots=True)['complete']
    with pytest.raises(FileExistsError, match='cannot be reset'):
        cli.analyze(gate.proof(plan), a, state, tmp_path / 'second', rp, no_plots=True)
    assert not (tmp_path / 'second').exists()


def function_ast(path, name):
    return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_analyze_ast_is_identical_except_explicit_repair_arguments_and_receipt():
    old = function_ast(ROOT / 'scripts/analyze_supplemental_phase_v2.py', 'analyze')
    new = copy.deepcopy(function_ast(ROOT / 'scripts/analyze_supplemental_phase_v3.py', 'analyze'))
    assert new.args.args.pop(4).arg == 'repair_proof'
    changes = []
    for node in ast.walk(new):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'guard' and node.func.attr in ('authorize', 'inspect'):
                last = node.args.pop()
                assert isinstance(last, ast.Name) and last.id == 'repair_proof'
                changes.append(node.func.attr)
        if isinstance(node, ast.Dict):
            indices = [i for i, key in enumerate(node.keys) if isinstance(key, ast.Constant) and key.value == 'operational_repair']
            for index in reversed(indices):
                assert isinstance(node.values[index], ast.Name) and node.values[index].id == 'repair_proof'
                node.keys.pop(index)
                node.values.pop(index)
                changes.append('completion_metadata')
    assert sorted(changes) == ['authorize', 'completion_metadata', 'inspect']
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)
    assert ast.dump(function_ast(ROOT / 'scripts/analyze_supplemental_phase_v3.py', 'write_auxiliary'), include_attributes=False) == ast.dump(function_ast(ROOT / 'scripts/analyze_supplemental_phase_v2.py', 'write_auxiliary'), include_attributes=False)


def test_retained_real_bytes_are_not_modified():
    expected = {
        'src/replay_study/supplemental_analysis_guard_v2.py': '64c89e37be3d595c239ad2683d72a85cdef6db776374ef6eeae28e23b97c3aea',
        'scripts/analyze_supplemental_phase_v2.py': 'a2f613babda5f1fd3b520259515b1acda90703ef42d3c7089c32c7e57497af3a',
        'docs/supplemental_aux_analysis_protocol_v2.json': 'eed050602c368e5e4be87172ac4d208b890d9a7f7e9811ce1684ae62acc8f742',
        'artifacts/analysis_prepared_v2/extended_pythia_410m_100m_nll.json': '61f44eba006214462680f656fcb6cc3a90aae550ae506f84f2ec27446f2292fb',
        'artifacts/analysis_prepared_v2/extended_pythia_410m_100m_nll_root_authority_TEMPLATE.json': 'fccd3c1ecbd07d281b1dbd015eb35d84a20b45704cd056c9d46bba542e8e4b77',
    }
    for path, sha in expected.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
