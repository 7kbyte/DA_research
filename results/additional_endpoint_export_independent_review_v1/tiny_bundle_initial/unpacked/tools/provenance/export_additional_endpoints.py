#!/usr/bin/env python3
"""Additional formal model-only delivery, separate outputs and two CPU slots.

Preparation is read-only with respect to checkpoints and all live exporters.
Only explicit --execute may encode; nothing here publishes or grants authority.
"""
from __future__ import annotations
import argparse, copy, datetime as dt, fcntl, gc, hashlib, json, os, re, resource, signal, stat, sys, tarfile, threading, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from replay_study.weight_archive import encode_weights, decode_weights, verify_roundtrip, file_sha256
from replay_study.analysis import _check_summary, training_validation_execution, verify_recorded_execution, selected_checkpoint_exposure
from export_remaining_primary_endpoints import checkpoint_metadata, same_metadata, build_verified_tar, signature, write
from export_A_selected_weights import verify_primary_stat
from export_C_final_weights import load_contexts, complete_phase_gate, run_ready, FrozenFiles, READY_NAMES
from schedule_jobs import digest_job

OUTPUT = Path('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v1')
DEADLINE = dt.datetime(2026, 11, 7, 2, 0, tzinfo=dt.timezone.utc)
MASTER = ROOT / 'configs/supplemental_v1_queue.json'
STATE = ROOT / 'logs/supplemental_job_state.json'
SLOTS = {'slot1': ['A_remaining_final', 'C410_best', 'C1B_best'], 'slot2': ['B_all_final', 'C2p8B_best']}
SLOT_LOCKS = {s: OUTPUT.parent/('.additional_endpoint_'+s+'.lock') for s in SLOTS}
PACKAGES = {
    'A_remaining_final': dict(logical=93, limit_bytes=115 << 30, model='pythia-410m', kind='final', slot='slot1'),
    'B_all_final': dict(logical=60, limit_bytes=75 << 30, model='pythia-410m', kind='final', slot='slot2'),
    'C410_best': dict(logical=24, limit_bytes=30 << 30, model='pythia-410m', kind='best', slot='slot1'),
    'C1B_best': dict(logical=24, limit_bytes=75 << 30, model='pythia-1b', kind='best', slot='slot1'),
    'C2p8B_best': dict(logical=15, limit_bytes=140 << 30, model='pythia-2.8b', kind='best', slot='slot2'),
}
A_REGISTRY = ROOT / 'configs/supplemental_v1_A_registry.json'
B_REGISTRY = ROOT / 'configs/supplemental_v1/activated/B_conditional_bank_variance/executor_registry.json'
IDENTITIES = {'A_remaining_final': ('supplement_A_nll', '88dc927c9b4909cca66901608216290e1e435022e4ff925c00875f5099077a90'),
              'B_all_final': ('supplement_B_nll', 'ffa3304307004d4e013b7414a841c5cdc54df619dd5f0376b276491ab2c515b4')}
RESERVATIONS = {k: '/data/202610/7kb/research_delivery/.extra_'+v+'_weights_reserve_20261106' for k,v in {'A_remaining_final':'A_remaining_final','B_all_final':'B_final','C410_best':'C_410m_best','C1B_best':'C_1b_best','C2p8B_best':'C_2p8b_best'}.items()}

def read(path): return json.loads(Path(path).read_text())
def absolute(path): return Path(path) if Path(path).is_absolute() else ROOT / path
def require(ok, message):
    if not ok: raise ValueError(message)
def proof(path):
    p = absolute(path).resolve()
    return dict(path=str(p), sha256=file_sha256(p))
def counters(obj): return {k: obj[k] for k in ('step', 'target_prediction_tokens', 'replay_prediction_tokens')}
def endpoint_key(row): return row['actual_run_id'], row['checkpoint_sha256']

def remaining_A(registry, identities, primary, selected):
    """Subtract saved identities; method/lambda performance is never consulted."""
    require(len(registry) == len(identities) == 126, 'Complete126 A registry/identities required')
    by_id = {r['run_id']: r for r in identities}
    require(len(by_id) == 126 and set(by_id) == {r['id'] for r in registry}, 'A identity cohort differs')
    saved = {(r['weights_actual_run_id'], r['source_checkpoint_sha256']) for r in primary['logical_runs']}
    saved |= {(r['actual_run_id'], r['source_checkpoint_sha256']) for r in selected['logical_runs']}
    kept, excluded = [], []
    for row in registry:
        bound = by_id[row['id']]
        require(row['phase'] == 'supplement_A_lambda_sweep' and bound['role'] == 'supplement_A' and bound['kind'] == 'final', 'A role/endpoint changed')
        actual = row['alias_of'] if row['execution_kind'] == 'alias' else row['id']
        require(bound['actual_run_id'] == actual, 'A actual run alias differs')
        (excluded if endpoint_key(bound) in saved else kept).append(bound)
    require(len(kept) == 93 and len(excluded) == 33 and len({endpoint_key(r) for r in kept}) == 93, 'Expected A93new/33saved identity partition')
    require(sum(r['execution_kind'] == 'alias' for r in registry if r['id'] in {x['run_id'] for x in excluded}) == 18, 'Expected18 main aliases and15 selected-new exclusions')
    return kept, excluded

def validate_B(registry, identities):
    expected = {(g, k, s) for g in ['B4', 'M', 'C'] for k in [20, 196] for s in range(1001, 1011)}
    require(len(registry) == len(identities) == 60 and {(r['group'], r['bank_chunks'], r['selection_seed']) for r in registry} == expected, 'Complete B60 draw/size/method matrix required')
    by_id = {r['run_id']: r for r in identities}
    require(len(by_id) == 60 and set(by_id) == {r['id'] for r in registry}, 'B identity cohort differs')
    for row in registry:
        b = by_id[row['id']]
        require(row['phase'] == 'conditional_bank_variance' and row['training_seed'] == row['generation_seed'] == 11 and b['role'] == 'supplement_B' and b['kind'] == 'final' and b['actual_run_id'] == row['id'] and b['selection_seed'] == row['selection_seed'] and b['training_seed'] == b['generation_seed'] == 11, 'B draw is not a training seed or aliased run')
    return list(by_id.values())

def published(name, cache, expected_sha=None):
    folder = ROOT / 'artifacts/evaluation_ready_v5' / name
    marker, path = folder / 'READY.json', folder / 'manifest.json'
    if not marker.exists(): return None
    before = signature(marker); raw = marker.read_bytes(); ready = json.loads(raw)
    require(ready.get('ready') is True and ready.get('status') == 'ready' and ready.get('id') == name and ready.get('mode') == 'nll' and absolute(ready['manifest']['path']).resolve() == path.resolve(), 'Exact published READY/id/mode/manifest required')
    digest = file_sha256(path)
    require(digest == ready['manifest']['sha256'] and (expected_sha is None or digest == expected_sha), 'Published identity SHA changed')
    cache.bind(path, digest); cache.bind(marker, hashlib.sha256(raw).hexdigest())
    require(signature(marker) == before, 'READY changed during observation')
    return dict(path=str(path), sha256=digest, ready=proof(marker), records=read(path)['records'])

def durable_manifest(receipt_path, filename, member, tar_sha, member_sha):
    receipt = read(receipt_path); path = Path('/data/202610/7kb/research_delivery') / filename
    require(receipt.get('status') == 'passed' and receipt['destination_sha256'] == tar_sha, 'Historical durable publication differs')
    before = verify_primary_stat(receipt, path)
    with tarfile.open(path, 'r:') as tar: raw = tar.extractfile(member).read()
    require(hashlib.sha256(raw).hexdigest() == member_sha and verify_primary_stat(receipt, path) == before, 'Pinned member/current durable stat differs')
    return json.loads(raw), dict(publication=proof(receipt_path), historical_TAR_sha256=tar_sha, current_stat=before, member=member, member_sha256=member_sha, fresh_entire_TAR_SHA=False)

def source_definition(bound, row, master, state):
    run_id = bound['actual_run_id']; matches = [j for j in master if j['id'] == run_id]
    require(len(matches) == 1, 'Exactly one frozen training definition required')
    definition = matches[0]; record = state.get('jobs', {}).get(run_id, {})
    require(record.get('status') == 'completed' and record.get('returncode') == 0 and not record.get('missing_outputs') and 'stop_requested_at' not in record and record.get('argv') == definition['argv'] and record.get('definition_sha256') == digest_job(definition), 'Actual rc0/no-stop/argv/definition gate failed')
    argv = definition['argv']
    require('--config' in argv and Path(argv[argv.index('--config') + 1]).resolve() == Path(row['config_path']).resolve(), 'Training argv is not source config')
    require(all(absolute(p).is_file() for p in definition.get('outputs', [])), 'Completed original job output missing')
    return definition, record

def validate_bound(bound, row, kind, model, family, cache):
    require(bound['kind'] == kind and bound['model_tag'] == model and bound['data_family'] == family, 'Endpoint/model/data identity changed')
    for key in ['config_path', 'summary_path']:
        require(Path(bound[key]).resolve() == Path(row[key]).resolve(), 'Source path differs')
        cache.bind(row[key], bound['input_sha256'][bound[key]])
    cfg, summary = read(row['config_path']), read(row['summary_path'])
    require(summary['status'] == 'completed' and summary['config'] == cfg and summary['counters']['target_prediction_tokens'] == cfg['target_prediction_tokens'], 'Full trained source/config required')
    cp = Path(summary['checkpoint_path'] if kind == 'final' else summary['best_target_validation']['checkpoint_path'])
    require(cp.resolve() == Path(bound['checkpoint']).resolve() and not cp.is_symlink() and re.fullmatch(r'[A-Za-z0-9_.-]+', bound['run_id']), 'Source checkpoint identity/path differs')
    info = signature(cp)
    require(info['bytes'] == bound['checkpoint_bytes'] and info['mtime_ns'] == bound['checkpoint_mtime_ns'], 'Checkpoint stat differs from frozen identity')
    return cfg, summary

def best_pair_decisions(context, payload, identities):
    phase = context['phase']; by_pair = {(r['seed'], r['group']): r for r in context['rows']}
    require(payload.get('status') == 'passed' and payload.get('phase') == phase['phase'] and payload.get('model') == phase['model'] and payload.get('endpoint') == 'best' and payload.get('algorithm') == 'sha256_sorted_key_dtype_shape_contiguous_uint8_v1' and len(payload.get('comparisons', [])) == 3, 'Exact fullphase best proof required')
    decisions = {}
    for pair in payload['comparisons']:
        seed, equal = pair.get('seed'), pair.get('bitwise_model_state_equal')
        require(seed in [11, 22, 33] and seed not in decisions and type(equal) is bool and pair.get('kind') == 'identical_bank_null_control' and pair.get('endpoint') == 'best', 'Explicit unique best pair decisions required')
        sides = {s['run_id']: s for s in [pair['left'], pair['right']]}
        require(set(sides) == {by_pair[seed, g]['id'] for g in ['B4', 'D']}, 'Cross-seed/model D alias forbidden')
        for group in ['B4', 'D']:
            row = by_pair[seed, group]; side = sides[row['id']]; b = identities[row['id']]
            exposure = side['checkpoint_exposure']
            require(side.get('kind') == 'best' and side.get('model_revision') == context['protocol']['model_revision'] and side['checkpoint_sha256'] == b['checkpoint_sha256'] and Path(side['checkpoint']).resolve() == Path(b['checkpoint']).resolve() and side['config_sha256'] == file_sha256(row['config_path']) and side['summary_sha256'] == file_sha256(row['summary_path']) and all(exposure[k] == b[k] for k in ['target_prediction_tokens_at_checkpoint', 'replay_prediction_tokens_at_checkpoint']), 'Best proof side is not source-bound to this endpoint')
        left, right = sides[by_pair[seed, 'B4']['id']], sides[by_pair[seed, 'D']['id']]
        require(equal == (left['model_state_fingerprint'] == right['model_state_fingerprint']), 'Equality flag contradicts fingerprints')
        decisions[seed] = dict(equal=equal, comparison=pair, b4=left, d=right)
    return decisions

def C_best_gate(context, state, master, cache, folder):
    if not all(run_ready(r, context['definitions'][r['id']], state, master) for r in context['rows']): return None
    model = context['phase']['model']
    identity = published(READY_NAMES[model], cache)
    if identity is None: return None
    best = [b for b in identity['records'] if b['kind'] == 'best']
    require(len(best) == len(context['rows']) and {b['run_id'] for b in best} == {r['id'] for r in context['rows']}, 'Complete model-specific best cohort required')
    identities = {b['run_id']: b for b in best}
    for row in context['rows']:
        b = identities[row['id']]
        require(b['role'] == 'extended' and b['phase'] == 'formal' and b['actual_run_id'] == row['id'] and b['training_seed'] == row['seed'] and b['group'] == row['group'] and b['ratio'] == row['ratio'] and b['weight'] == row['weight'] and b['phase_name'] == READY_NAMES[model].removesuffix('_nll'), 'C best logical role/phase changed')
        cfg, summary = validate_bound(b, row, 'best', model, 'extended_v1', cache)
        require(cfg['model_revision'] == context['protocol']['model_revision'] and cfg['target_prediction_tokens'] == context['phase']['target_prediction_tokens'] and b['full_training_committed_path']['target_prediction_tokens'] == cfg['target_prediction_tokens'], 'C best original model revision/full horizon differs')
    final_gate = complete_phase_gate(context, state, master, cache, folder)
    if final_gate is None: return None
    sources = final_gate['sources'] + [dict(path=identity['path'], sha256=identity['sha256']), identity['ready']]
    decisions = {}
    if model != 'pythia-2.8b':
        completion_path = ROOT / 'results/extended_phase_equivalence' / context['phase']['phase'] / 'completion.json'
        completion = read(completion_path); ref = completion['proofs']['best']
        cache.bind(ref['path'], ref['sha256']); payload = read(ref['path'])
        require(payload['source_audit'] == completion['audit'] and payload['source_registry'] == completion['source_registry'] and payload['source_snapshot'] == completion['input_snapshot'], 'Best proof fullphase source binding differs')
        decisions = best_pair_decisions(context, payload, identities); sources.append(ref)
    return dict(status='complete_model_best_source_gate_passed', identity=identity, records=identities, decisions=decisions, sources=sources, final_source_gate=final_gate)

def frozen_inputs(bound, cfg, catalog, cache):
    by_path = {str(absolute(p).resolve()): h for p, h in bound['input_sha256'].items()}
    by_path.update({str(absolute(p).resolve()): e['sha256'] for p, e in catalog.items()})
    for path in [cfg['target_path'], cfg['replay_path'], *cfg['validation_paths'].values()]:
        key = str(Path(path).resolve())
        require(key in by_path, 'Frozen training/validation/replay file hash absent: ' + key)
        cache.bind(path, by_path[key])

def prepare(output=OUTPUT):
    """No checkpoint tensor load, no encoding, no lock acquisition, no watcher."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit empty CVD required')
    started, cpu = time.monotonic(), time.process_time(); output = Path(output).resolve()
    require(output.is_relative_to(Path('/tmp/grad-research-20261106/delivery')) and not output.exists(), 'Fresh scratch/delivery output namespace required')
    cache = FrozenFiles()
    primary, primary_evidence = durable_manifest(ROOT/'docs/primary_weights_durable_publication.json', 'primary_final_weights_lossless_v1.tar', 'primary_final_weights_manifest.json', '2bb090fbe7df172b31102068715143ab702087ff49f77ef7dfd5db3637c928b1', '4120d177c22200b0bc5ee8390e0c2e2a1f5be1acbc58ef0f6d209bd47f96e8fb')
    selected, selected_evidence = durable_manifest(ROOT/'docs/A_selected_weights_durable_publication.json', 'supplement_A_selected_final_weights_lossless_v1.tar', 'supplement_A_selected_final_manifest.json', '0dfec20b1bf9dde38b0d0d453c821eb088490bde188dc0e19bc498cc37b04631', '98c70968a733f6436ee47be1ddeb0a76162bce1f7fb5cc3271e75fa6bb27c129')
    A = published(IDENTITIES['A_remaining_final'][0], cache, expected_sha=IDENTITIES['A_remaining_final'][1])
    B = published(IDENTITIES['B_all_final'][0], cache, expected_sha=IDENTITIES['B_all_final'][1])
    require(A is not None and B is not None, 'A/B actual full READY identities required')
    registryA, registryB = read(A_REGISTRY), read(B_REGISTRY)
    for registry_path, identity in [(A_REGISTRY,A),(B_REGISTRY,B)]:
        matches={h for p,h in identity['records'][0]['input_sha256'].items() if absolute(p).resolve()==registry_path.resolve()}
        require(len(matches)==1,'Frozen identity does not bind actual runtime registry')
        cache.bind(registry_path, next(iter(matches)))
    rowsA, excluded = remaining_A(registryA, A['records'], primary, selected)
    rowsB = validate_B(registryB, B['records'])
    formal_snapshot = ROOT/'artifacts/provenance/formal_deterministic_input_snapshot.json'
    cache.bind(formal_snapshot, file_sha256(formal_snapshot)); main_catalog = read(formal_snapshot)['data']
    protocol_path = ROOT/'docs/formal_deterministic_protocol.json'; protocol = read(protocol_path)
    cache.bind(protocol_path, read(formal_snapshot)['protocol']['docs/formal_deterministic_protocol.json']['sha256'])
    cache.bind(protocol['initial_validation_path'], protocol['initial_validation_sha256'])
    baseline = read(protocol['initial_validation_path'])['execution_provenance']
    bprep = ROOT/'artifacts/bank_variance_410m_v1/preparation_manifest.json'; cache.bind(bprep, 'a40393f18322acc6ffe060e3ea60e43da458639d65b1adf381516fb70ee3e35d')
    bank_hashes = {r['id']: r['bank_sha256'] for r in read(bprep)['banks']}
    master, state = read(MASTER), read(STATE); phases = {}
    for name, identity_rows, registry in [('A_remaining_final', rowsA, registryA), ('B_all_final', rowsB, registryB)]:
        by_id = {r['id']: r for r in registry}; records = []
        for bound in identity_rows:
            row = by_id[bound['run_id']]
            cfg, summary = validate_bound(bound, row, 'final', 'pythia-410m', 'main_v3', cache)
            require(bound['actual_run_id'] == row['id'], 'New exports cannot reuse an already saved training alias')
            definition, scheduler = source_definition(bound, row, master, state)
            verify_recorded_execution(baseline, training_validation_execution(summary))
            catalog = dict(main_catalog)
            if name == 'B_all_final': catalog[cfg['replay_path']] = {'sha256': bank_hashes[row['id']]}
            frozen_inputs(bound, cfg, catalog, cache)
            require(cfg['model_revision'] == primary['base_revision'] and summary['counters']['target_prediction_tokens'] == 10_000_000 and summary['counters']['replay_prediction_tokens'] == 500_000, 'Main-family source model/horizon differs')
            records.append(dict(bound=bound, row=row, definition=definition, scheduler_record=scheduler, frozen_source_stat=signature(bound['checkpoint'])))
        base_files = [dict(path=x['source_path'], member=x['archive_member'], sha256=x['sha256']) for x in primary['base_files']]
        for f in base_files: cache.bind(f['path'], f['sha256'])
        phases[name] = dict(status='complete_source_metadata_preflight_no_CP_reads', rows=records, baseline_execution=baseline, base_files=base_files, base_path=next(f['path'] for f in base_files if f['path'].endswith('model.safetensors')), model_revision=primary['base_revision'])
    contexts, ccache = load_contexts(); cache.files.update(ccache.files)
    for name in ['C410_best', 'C1B_best', 'C2p8B_best']:
        context = next(c for c in contexts if c['phase']['model'] == PACKAGES[name]['model'])
        phases[name] = dict(status='pending_actual_full_model_READY_audit_and_best_proof', context=context, logical_run_ids=[r['id'] for r in context['rows']], base_files=[dict(path=p, member='base/'+context['phase']['model']+'-final/'+Path(p).name, sha256=v['sha256']) for p,v in context['snapshot']['model_files'].items()])
    code = [Path(__file__), ROOT/'src/replay_study/weight_archive.py', ROOT/'src/replay_study/analysis.py', ROOT/'scripts/archive_model_weights.py', ROOT/'scripts/export_remaining_primary_endpoints.py', ROOT/'scripts/export_C_final_weights.py', ROOT/'scripts/export_A_selected_weights.py', ROOT/'scripts/audit_extended_runs.py', ROOT/'scripts/audit_formal_runs.py', ROOT/'scripts/schedule_jobs.py']
    for p in code: cache.bind(p, file_sha256(p))
    require(file_sha256(ROOT/'src/replay_study/weight_archive.py') == '3c2bcf9e38773ae4f69c27f85f342356dbefcb53d817be4321da6e9c4e550e1a', 'Verified codec changed')
    require(file_sha256(ROOT/'scripts/archive_model_weights.py') == '28c5841cae8e7fd87d8cb62e7f756dad40ce15cb558710c31e48dfb45fde668a', 'Verified archived decoder CLI changed')
    cache.stable(); output.mkdir(parents=True, exist_ok=False)
    scope = dict(schema_version=1, status='prepared_only_no_export_or_ROOT_authority', packages=PACKAGES, slots=SLOTS, output=str(output), phases=phases, excluded_A_saved_identities=[dict(run_id=r['run_id'], actual_run_id=r['actual_run_id'], checkpoint_sha256=r['checkpoint_sha256']) for r in excluded], saved_primary_manifest=primary_evidence, saved_A_selected_manifest=selected_evidence, frozen_files=cache.files, code=[proof(p) for p in code], deadline_utc=DEADLINE.isoformat(), source_checkpoint_tensor_or_fresh_SHA_reads=0, GPU_used=False, shared_storage_written=False, root_authorized=False, reservations_advisory_only=RESERVATIONS, reservation_capacity_not_verified_or_consumed=True, preparation_wall_seconds=time.monotonic()-started, preparation_process_cpu_seconds=time.process_time()-cpu, timing_scope='Read-only whole preflight including native model/data hashes; no original checkpoint payload read', alias_policy='C best only source-bound explicit true D/B4 best pair; false independently encoded', cost_policy='Every run independent parent, nested codec/verification children; assembly independent, daemon waiting is context only')
    scope['worker_lock_paths'] = {k:str(v) for k,v in SLOT_LOCKS.items()}
    write(output/'scope.json', scope)
    return scope

def validate_payload(item, saved, baseline):
    bound, row = item['bound'], item['row']; cfg = read(row['config_path']); summary = read(row['summary_path'])
    _check_summary(summary, {**row, 'seed': row.get('seed', row.get('training_seed'))})
    require(summary['status'] == 'completed' and saved['config'] == cfg == summary['config'] and saved.get('provenance') == summary.get('provenance'), 'Checkpoint/config/summary/provenance differs')
    verify_recorded_execution(baseline, training_validation_execution(summary))
    exposure = selected_checkpoint_exposure(summary, bound['kind'])
    expected = dict(step=bound['step'], target_prediction_tokens=bound['target_prediction_tokens_at_checkpoint'], replay_prediction_tokens=bound['replay_prediction_tokens_at_checkpoint'])
    require(counters(saved['counters']) == expected and exposure['target_prediction_tokens'] == expected['target_prediction_tokens'] and exposure['replay_prediction_tokens'] == expected['replay_prediction_tokens'] and exposure['checkpoint_step'] == expected['step'], 'Actual checkpoint/summary-selected exposure differs')
    require(summary['counters']['target_prediction_tokens'] == cfg['target_prediction_tokens'] and bound['full_training_committed_path']['target_prediction_tokens'] == cfg['target_prediction_tokens'], 'Best is not from fully completed training')
    if bound['kind'] == 'best':
        require(saved.get('checkpoint_kind') == 'model_only_best_target_validation' and 'optimizer_state' not in saved and saved.get('selection') == summary['best_target_validation'], 'Exact validation-selected model-only best metadata required')
    else:
        require(saved.get('validation_trajectory') == summary['validation_trajectory'] and saved.get('best_target_validation') == summary['best_target_validation'], 'Full final trajectory/selection metadata differs')
    require(all(t.dtype == torch.float32 for t in saved['model_state'].values()), 'Only pinned FP32 model state supported')
    return exposure

def validate_resume(old, item):
    b = item['bound']
    require(old['run_id'] == b['run_id'] and old['kind'] == b['kind'] and old['source_checkpoint_sha256'] == b['checkpoint_sha256'] and old['source_stat'] == signature(b['checkpoint']) and old['config_sha256'] == file_sha256(item['row']['config_path']) and old['summary_sha256'] == file_sha256(item['row']['summary_path']) and old['definition_sha256'] == digest_job(item['definition']), 'Earlier verified record source changed')
    require(file_sha256(old['metadata_path']) == old['metadata_sha256'], 'Prior metadata changed')
    if 'archive_path' in old: require(signature(old['archive_path']) == old['archive_stat'], 'Prior verified ZIP stat changed')
    return old

def fd_signature(fd):
    s = os.fstat(fd.fileno())
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)

def cgroup_memory():
    result = {}
    for name in ['memory.current','memory.max','memory.events']:
        try: result[name] = (Path('/sys/fs/cgroup')/name).read_text().strip()
        except OSError: result[name] = 'unknown'
    return result

def export_one(name, item, folder, base_path, base_sha, baseline, alias=None, expected_fingerprint=None):
    b, row = item['bound'], item['row']; endpoint = b['run_id'] + '__' + b['kind']
    record_path = folder/'records'/(endpoint+'.json')
    if record_path.exists():
        old = validate_resume(read(record_path), item)
        require(expected_fingerprint is None or old['model_fingerprint'] == expected_fingerprint, 'Prior record differs from endpoint proof')
        if alias is not None: require(old.get('alias_of') == alias['canonical']['run_id'] and old['model_fingerprint'] == alias['source_side']['model_state_fingerprint'], 'Prior alias no longer matches exact best proof')
        return old
    started, cpu = time.monotonic(), time.process_time(); memory_before = cgroup_memory(); cp = Path(b['checkpoint']); before = signature(cp)
    require(not cp.is_symlink() and before == item['frozen_source_stat'], 'Source checkpoint redirected/changed')
    folder.mkdir(parents=True, exist_ok=True)
    with cp.open('rb') as fd:
        opened = fd_signature(fd); require(opened == before, 'Source path and open FD differ')
        saved = torch.load(cp, map_location='cpu', mmap=True, weights_only=False)
        exposure = validate_payload(item, saved, baseline)
        identity = {k: b.get(k) for k in ['run_id', 'actual_run_id', 'phase', 'phase_name', 'role', 'kind', 'model_tag', 'data_family', 'group', 'training_seed', 'generation_seed', 'selection_seed', 'ratio', 'weight']}
        for key in ['bank_chunks','unique_bank_prediction_tokens']:
            if key in row: identity[key] = row[key]
        sidecar = folder/'metadata'/(endpoint+'.source_metadata.pt'); sidecar.parent.mkdir(parents=True, exist_ok=True)
        require(not sidecar.exists(), 'Preserve unfinished metadata; reviewed fresh namespace needed')
        payload = dict(logical_identity=identity, actual_exposure=exposure, original_checkpoint_metadata=checkpoint_metadata(saved))
        torch.save(payload, sidecar)
        require(same_metadata(payload, torch.load(sidecar, map_location='cpu', weights_only=False)), 'All nonweight/nonoptimizer metadata failed roundtrip')
        del saved, payload; gc.collect()
        result = dict(status='verified_additional_endpoint_not_yet_assembled', package=name, **identity, source_checkpoint=str(cp), source_stat=before, config_sha256=file_sha256(row['config_path']), summary_sha256=file_sha256(row['summary_path']), definition_sha256=digest_job(item['definition']), definition=item['definition'], completed_scheduler_record=item['scheduler_record'], frozen_checkpoint_identity_sha256=b['checkpoint_sha256'], actual_exposure=exposure, metadata_path=str(sidecar), metadata_member='metadata/'+sidecar.name, metadata_sha256=file_sha256(sidecar), all_nonweight_metadata_roundtrip_verified=True)
        if alias is not None:
            source_sha = file_sha256(cp)
            require(source_sha == b['checkpoint_sha256'] == alias['source_side']['checkpoint_sha256'], 'Alias fresh CP SHA differs from frozen proof')
            result.update(source_checkpoint_sha256=source_sha, alias_of=alias['canonical']['run_id'], archive_member=alias['canonical']['archive_member'], archive_sha256=alias['canonical']['archive_sha256'], model_fingerprint=alias['canonical']['model_fingerprint'], alias_evidence=alias['proof'], additional_encoding_calls=0)
            require(result['model_fingerprint'] == alias['source_side']['model_state_fingerprint'], 'Alias proof fingerprint differs')
        else:
            archive = folder/'weights'/(endpoint+'.xor.zip'); restored = folder/'verification'/(endpoint+'.restored.pt')
            encode_cpu = time.process_time(); encoded = encode_weights(cp, base_path, archive, threads=2, expected_model_sha256=None if expected_fingerprint is None else expected_fingerprint['sha256_model_state']); encode_cpu = time.process_time()-encode_cpu
            decode_cpu = time.process_time(); decoded = decode_weights(archive, base_path, restored, threads=2); decode_cpu = time.process_time()-decode_cpu
            verify_cpu = time.process_time(); verified = verify_roundtrip(cp, restored, archive, threads=2); verify_cpu = time.process_time()-verify_cpu
            require(encoded['source_checkpoint_sha256'] == b['checkpoint_sha256'] and encoded['base_sha256'] == base_sha and verified['all_tensor_bytes_equal'], 'Frozen CP/base or complete uint8 verification failed')
            result.update(source_checkpoint_sha256=encoded['source_checkpoint_sha256'], archive_path=str(archive), archive_member='weights/'+archive.name, archive_sha256=encoded['archive_sha256'], archive_stat=signature(archive), model_fingerprint=encoded['model_fingerprint'], encode=encoded, decode=decoded, bytewise_verification=verified, additional_encoding_calls=1, encode_process_cpu_seconds=encode_cpu, decode_process_cpu_seconds=decode_cpu, verification_process_cpu_seconds=verify_cpu)
        after = fd_signature(fd)
        require(signature(cp) == before == after and not cp.is_symlink(), 'Source path/open FD changed during entire operation')
    result.update(source_fd_five_stat_before=opened, source_fd_five_stat_after=after, wall_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu, peak_rss_bytes_process_cumulative=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, cgroup_memory_before=memory_before, cgroup_memory_after=cgroup_memory(), cpu_threads=2, workers=1, GPU_used=False, shared_storage_written=False, original_CP_ZIP_restored_retained=True, timer_scope='Independent endpoint parent; codec/verification child timers nested; alias has its own metadata/freshSHA cost')
    write(record_path, result)
    return result

def safe_members(members):
    names = [n for _, n, _ in members]
    require(len(set(names)) == len(names) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names), 'Unsafe or duplicate TAR member')

def assemble(name, phase, records, folder, scope_path, cache, gate):
    spec = PACKAGES[name]; started, cpu = time.monotonic(), time.process_time()
    require(len(records) == spec['logical'] and len({r['run_id'] for r in records}) == spec['logical'], 'Partial phase cannot become a complete package')
    require({r['run_id'] for r in records} == {x['bound']['run_id'] for x in phase['rows']}, 'Assembly cohort differs')
    members = []
    for item, record in zip(phase['rows'], records):
        validate_resume(record, item)
        require(record['kind'] == spec['kind'] and record['model_tag'] == spec['model'], 'Assembly model/endpoint mixed')
        members.append((Path(record['metadata_path']), record['metadata_member'], record['metadata_sha256']))
        if 'alias_of' not in record: members.append((Path(record['archive_path']), record['archive_member'], record['archive_sha256']))
    for f in phase['base_files']: members.append((Path(f['path']), f['member'], f['sha256']))
    manifest = dict(format='additional_formal_lossless_v1', status='complete_all_logical_endpoints_verified', package=name, model_tag=spec['model'], kind=spec['kind'], logical_endpoints=spec['logical'], unique_weights=sum('alias_of' not in r for r in records), aliases=sum('alias_of' in r for r in records), logical_runs=records, base_dependency=dict(kind='native_base_config_tokenizer_readme_included_no_external_package_needed', files=phase['base_files']), fullphase_source_gate=gate, source_scope=proof(scope_path), optimizer_included=False, source_CP_ZIP_restored_retained=True, publisher_required=True, shared_storage_written=False)
    mp = folder/'additional_weights_manifest.json'; write(mp, manifest)
    note = folder/'RESTORE.md'
    note.write_text('# Model-only lossless weights\n\nEvery independent ZIP was decoded and compared against every original uint8 tensor. Native base/config/tokenizer are included under base/. No network or other package is needed.\n\n`python tools/scripts/archive_model_weights.py --threads 2 decode --archive weights/ENDPOINT.xor.zip --base base/MODEL-final/model.safetensors --output NEW.pt`\n\nD aliases reference exact source-bound B4 best weight members and retain their own metadata sidecars. All original nonweight/nonoptimizer metadata is in metadata/*.source_metadata.pt. Optimizer/Adam resume is excluded; separate ROOT durable publication is required.\n')
    for p,n in [(mp,mp.name),(note,note.name),(scope_path,'provenance/scope.json'),(Path(__file__),'tools/provenance/export_additional_endpoints.py'),(ROOT/'src/replay_study/weight_archive.py','tools/src/replay_study/weight_archive.py'),(ROOT/'src/replay_study/__init__.py','tools/src/replay_study/__init__.py'),(ROOT/'scripts/archive_model_weights.py','tools/scripts/archive_model_weights.py')]: members.append((p,n,file_sha256(p)))
    for i, source in enumerate(gate.get('sources', [])):
        p = absolute(source['path']); members.append((p, f'provenance/gate/{i}_{p.name}', source['sha256']))
    safe_members(members); cache.stable()
    archive = folder/(name+'_weights_lossless_v1.tar')
    receipt = build_verified_tar(archive, members, spec['limit_bytes']); cache.stable()
    receipt.update(status='passed_complete_scratch_package_not_published', tar_path=str(archive), package=name, logical_endpoints=spec['logical'], unique_weights=manifest['unique_weights'], alias_count=manifest['aliases'], manifest_sha256=file_sha256(mp), wall_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu, timer_scope='Independent assembly-only; endpoint codec parents and daemon waiting not added again', shared_storage_written=False, ROOT_publication_required=True, original_CP_ZIP_restored_retained=True)
    write(folder/'completion.json', receipt)
    return receipt

def load_scope(path, digest):
    require(file_sha256(path) == digest, 'Prepared scope SHA differs')
    scope = read(path); require(scope['status'] == 'prepared_only_no_export_or_ROOT_authority' and scope['packages'] == PACKAGES and scope['slots'] == SLOTS, 'Scope/package policy differs')
    cache = FrozenFiles()
    for p, item in scope['frozen_files'].items(): cache.bind(p, item['sha256'])
    return scope, cache

def validate_resource_review(path, digest, scope_sha, slot):
    require(path is not None and digest is not None and file_sha256(path) == digest, 'ROOT resource review path/SHA required for batch')
    r = read(path)
    require(r.get('status') == 'accepted_cpu_resource_samples' and r.get('authorized_by') == 'root' and r.get('scope_sha256') == scope_sha and r.get('exporter_sha256') == file_sha256(__file__) and set(SLOTS[slot]).intersection(r.get('approved_packages', [])), 'Resource review does not bind this slot/scope/code')
    return dict(**proof(path), approved_packages=r['approved_packages'])

def execute(scope_path, scope_sha, slot, max_new_weights=1, review_path=None, review_sha=None, poll_seconds=30):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '2' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']), 'Empty CVD and two CPU threads required')
    require(dt.datetime.now(dt.timezone.utc) < DEADLINE, '02:00 UTC: no new operation may start')
    torch.set_num_threads(2); require(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
    startup_wall, startup_cpu = time.monotonic(), time.process_time()
    scope, cache = load_scope(scope_path, scope_sha); output = Path(scope['output'])
    require(output == OUTPUT or output.is_relative_to(Path('/tmp/grad-research-20261106/delivery')), 'Output must be isolated scratch/delivery')
    write(output/'operations'/(slot+'_startup_input_audit_'+str(time.time_ns())+'.json'),dict(status='passed_fresh_frozen_input_audit_only', scope_sha256=scope_sha, wall_seconds=time.monotonic()-startup_wall, process_cpu_seconds=time.process_time()-startup_cpu, timer_scope='Independent invocation input hashing after imports; excludes later codec/assembly and waiting', no_original_checkpoint_payloads_read=True))
    review = validate_resource_review(review_path, review_sha, scope_sha, slot) if max_new_weights > 1 else None
    stop = threading.Event(); new = 0; done = {}; started = time.monotonic()
    for sig in [signal.SIGTERM, signal.SIGINT]: signal.signal(sig, lambda *_: stop.set())
    def stopping(): return stop.is_set() or dt.datetime.now(dt.timezone.utc) >= DEADLINE
    lock_path = SLOT_LOCKS[slot]
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in SLOTS[slot]:
            if stopping(): break
            phase = copy.deepcopy(scope['phases'][name]); folder = output/name; folder.mkdir(parents=True, exist_ok=True)
            if (folder/'completion.json').exists():
                previous = read(folder/'completion.json')
                require(previous['status'] == 'passed_complete_scratch_package_not_published' and file_sha256(previous['tar_path']) == previous['tar_sha256'], 'Earlier complete package changed')
                done[name] = previous; continue
            gate = dict(status='complete_A_B_source_manifest_bound', sources=[]); decisions = {}
            while 'context' in phase and not stopping():
                context = phase['context']; gate = C_best_gate(context, read(STATE), read(MASTER), cache, folder)
                if gate is not None:
                    phase['rows'] = []
                    for row in sorted(context['rows'], key=lambda r:(r['group']=='D', r['seed'], r['group'])):
                        b = gate['records'][row['id']]; definition, scheduler = source_definition(b, row, read(MASTER), read(STATE))
                        phase['rows'].append(dict(bound=b, row=row, definition=definition, scheduler_record=scheduler, frozen_source_stat=signature(b['checkpoint'])))
                    phase['base_path'] = context['base']; phase['baseline_execution'] = context['baseline']['execution_provenance']; decisions = gate['decisions']; break
                write(output/(slot+'_progress.json'), dict(status='waiting_complete_C_READY_audit_best_proof', phase=name, new_weights_this_invocation=new, daemon_elapsed_wall_context_only=time.monotonic()-started))
                stop.wait(poll_seconds)
            if stopping(): break
            records = []
            for item in phase['rows']:
                if stopping(): break
                b, row = item['bound'], item['row']; previous_path = folder/'records'/(b['run_id']+'__'+b['kind']+'.json')
                alias, expected_fp = None, None
                if row['group'] == 'D':
                    decision = decisions[row['seed']]; expected_fp = decision['d']['model_state_fingerprint']
                    if decision['equal']:
                        canonical = next(r for r in records if r['run_id'] == decision['b4']['run_id'])
                        alias = dict(canonical=canonical, source_side=decision['d'], proof=dict(comparison=decision['comparison'], sources=gate['sources']))
                        require(canonical['model_fingerprint'] == decision['b4']['model_state_fingerprint'], 'B4 verified state differs from pair proof')
                reviewed = review is not None and name in review['approved_packages']
                phase_sample_exists = any('archive_path' in r and 'alias_of' not in r for r in records)
                if not previous_path.exists() and alias is None and (new >= max_new_weights or (not reviewed and phase_sample_exists)):
                    write(output/(slot+'_progress.json'), dict(status='resource_sample_limit_reached_requires_ROOT_review', phase=name, new_weights_this_invocation=new, review=review)); return
                current_definition, current_scheduler = source_definition(b, row, read(MASTER), read(STATE))
                require(current_definition == item['definition'] and current_scheduler == item['scheduler_record'], 'Completed per-job training authority changed')
                cache.stable(); existed = previous_path.exists()
                base_sha = next(f['sha256'] for f in phase['base_files'] if f['path'].endswith('model.safetensors'))
                result = export_one(name, item, folder, phase['base_path'], base_sha, phase['baseline_execution'], alias, expected_fp)
                new += int(not existed and alias is None); records.append(result)
                write(folder/'draft_progress.json', dict(status='verified_endpoints_only_not_complete_package', verified_logical=len(records), expected_logical=PACKAGES[name]['logical'], last_record=proof(previous_path), new_weights_this_invocation=new))
            if not stopping() and len(records) == PACKAGES[name]['logical']:
                done[name] = assemble(name, phase, records, folder, scope_path, cache, gate)
        write(output/(slot+'_progress.json'), dict(status='complete_slot_scratch_packages_not_published' if len(done)==len(SLOTS[slot]) else 'stopped_between_operations', done=done, new_weights_this_invocation=new, daemon_elapsed_wall_context_only=time.monotonic()-started, shared_storage_written=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='mode',required=True)
    pre=sub.add_parser('prepare'); pre.add_argument('--output-dir',type=Path,default=OUTPUT)
    run=sub.add_parser('execute'); run.add_argument('--scope',type=Path,required=True);run.add_argument('--scope-sha256',required=True);run.add_argument('--slot',choices=list(SLOTS),required=True);run.add_argument('--max-new-weights',type=int,default=1);run.add_argument('--resource-review',type=Path);run.add_argument('--resource-review-sha256');run.add_argument('--poll-seconds',type=float,default=30)
    args=parser.parse_args()
    if args.mode=='prepare':
        scope=prepare(args.output_dir); print(json.dumps(dict(status=scope['status'],scope=proof(args.output_dir/'scope.json'), counts={k:v['logical'] for k,v in PACKAGES.items()})))
    else:
        require(args.max_new_weights>=1 and args.poll_seconds>=1,'Positive weight limit and poll required')
        try:
            execute(args.scope,args.scope_sha256,args.slot,args.max_new_weights,args.resource_review,args.resource_review_sha256,args.poll_seconds)
        except Exception as error:
            try:
                output=Path(read(args.scope)['output']).resolve()
                if output.is_relative_to(Path('/tmp/grad-research-20261106/delivery')):
                    write(output/(args.slot+'_error.json'),dict(status='failed_or_blocked_no_complete_phase_claim',error=repr(error),successful_drafts_and_all_originals_retained=True,GPU_used=False,shared_storage_written=False))
            except Exception: pass
            raise

if __name__=='__main__': main()

