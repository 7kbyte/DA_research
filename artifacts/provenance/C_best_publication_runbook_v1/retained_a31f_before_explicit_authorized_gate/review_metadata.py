#!/usr/bin/env python3
"""Prepared metadata/stat-only C-best publication review; never export or publish."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = Path('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4')
SCOPE_SHA = '7adb8f7f0461fd929995dd370ff2a59b2abc5c0d7750a39e745d21d5546d9de3'
EXPORTER_SHA = 'c93e0c1d35fe39b3d71155a126db841b76648ba6b3afd58d475be4eb9c9da853'
PUBLISHER_SHA = '9c194743383a16f98313f767fe2e722c309ec57c1aafc65d9b99d08e5407022e'
NATIVE_REVIEWER = ROOT/'artifacts/provenance/C_native_final_publication_runbook_v2/review_metadata.py'
NATIVE_REVIEWER_SHA = 'c596a56d70b32cedce7bc9d2e00a0923477165aac6c24ee04ed78e8680413627'
CAPACITY_SHA = 'a15321819901d3272a483f724e7613a8afda302ab0fe9a6e05537856a972a62b'
PROFILES = {
    'C410_best': dict(model='pythia-410m', n=24, reserve='C_410m_best', inode=467366543, capacity=30 << 30),
    'C1B_best': dict(model='pythia-1b', n=24, reserve='C_1b_best', inode=375195308, capacity=75 << 30),
    'C2p8B_best': dict(model='pythia-2.8b', n=15, reserve='C_2p8b_best', inode=331039199, capacity=140 << 30),
}
EXTRAS = ['additional_weights_manifest.json', 'RESTORE.md', 'provenance/scope.json',
          'tools/provenance/export_additional_endpoints.py', 'tools/src/replay_study/weight_archive.py',
          'tools/src/replay_study/__init__.py', 'tools/scripts/archive_model_weights.py']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def signature(path):
    p = absolute(path)
    s = p.lstat()
    require(stat.S_ISREG(s.st_mode) and not p.is_symlink(), 'Regular nonsymlink source required')
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


class Metadata:
    """Explicitly bounded small bytes; all model/CP/ZIP/TAR inputs stat only."""
    def __init__(self, folder):
        self.folder = folder.resolve()
        self.evidence = {}
        self.observed = {}

    def opaque(self, path, expected=None, sidecar=False):
        p = absolute(path)
        stamp = signature(p)
        allowed_sidecar = sidecar and p.resolve().parent == self.folder/'metadata' and p.name.endswith('.source_metadata.pt')
        require(not {'.pt', '.zip', '.tar', '.safetensors'}.intersection(p.suffixes) or allowed_sidecar, 'Weight payload reads forbidden')
        require(stamp['bytes'] <= (8 if sidecar else 32) << 20, 'Bounded small metadata only')
        raw = p.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(signature(p) == stamp and (expected is None or digest == expected), 'Small metadata SHA/stat changed: '+str(p))
        self.evidence[str(p.resolve())] = dict(sha256=digest, stat=stamp)
        self.observed[str(p)] = stamp
        return raw

    def read(self, path, expected=None):
        return json.loads(self.opaque(path, expected))

    def bind(self, path, digest):
        self.opaque(path, digest)

    def payload_stat(self, path, expected=None):
        stamp = signature(path)
        require(expected is None or stamp == expected, 'Payload stat differs')
        self.observed[str(absolute(path))] = stamp
        return stamp

    def finish(self):
        for path, stamp in self.observed.items():
            require(signature(path) == stamp, 'Source stat changed during review')


def require_clean_rc(record):
    require(type(record.get('returncode')) is int and record['returncode'] == 0 and
            not any(k in record for k in ('stop_requested', 'stop_requested_at')),
            'Strict integer rc0 and absence of both stop keys required')


def validate_headers(name, receipt, manifest):
    p = PROFILES[name]
    require(receipt.get('status') == 'passed_complete_scratch_package_not_published' and
            manifest.get('status') == 'complete_all_logical_endpoints_verified' and manifest.get('format') == 'additional_formal_lossless_v1', 'Actual complete package required')
    require(receipt.get('package') == manifest.get('package') == name and
            manifest.get('model_tag') == p['model'] and manifest.get('kind') == 'best', 'Wrong package/model/final-versus-best')
    require(receipt.get('logical_endpoints') == manifest.get('logical_endpoints') == p['n'] and
            len(manifest.get('logical_runs', [])) == p['n'], 'Complete exact best cohort required')
    for key in ('unique_weights',):
        require(type(receipt.get(key)) is int and receipt[key] > 0, 'Strict unique count required')
    require(type(receipt.get('alias_count')) is int and receipt['alias_count'] >= 0, 'Strict alias count required')
    require(receipt['unique_weights'] + receipt['alias_count'] == p['n'] and
            manifest.get('unique_weights') == receipt['unique_weights'] and manifest.get('aliases') == receipt['alias_count'], 'Alias/unique inventory differs')
    require(manifest.get('optimizer_included') is False and
            manifest.get('source_scope', {}).get('sha256') == SCOPE_SHA, 'Foreign source scope or optimizer included')
    require(all(receipt.get(k) is True for k in ('file_fsync_completed', 'parent_directory_fsync_completed')), 'TAR and parent fsync required')
    require(re.fullmatch('[0-9a-f]{64}', receipt.get('tar_sha256', '')), 'Explicit TAR SHA required')
    if name == 'C2p8B_best':
        require(receipt['alias_count'] == 0 and receipt['unique_weights'] == 15 and
                all(r.get('group') != 'D' for r in manifest['logical_runs']), '2.8B has no D or alias condition')


def validate_alias_decision(row, canonical, decision, sources):
    """False equality requires independent D encoding; true equality uses exact B4."""
    require(type(decision.get('equal')) is bool, 'Explicit proof equality bool required')
    side = decision['d'] if row['group'] == 'D' else decision['b4']
    require(row['run_id'] == side['run_id'] and row['source_checkpoint_sha256'] == side['checkpoint_sha256'] and
            row['model_fingerprint'] == side['model_state_fingerprint'], 'Proof side FP/source/run differs')
    if row['group'] != 'D' or not decision['equal']:
        require('alias_of' not in row and type(row.get('additional_encoding_calls')) is int and row['additional_encoding_calls'] == 1, 'False proof/non-D requires independent encoding')
        return
    require(canonical is not None and canonical['run_id'] == decision['b4']['run_id'] and canonical['group'] == 'B4' and
            canonical['source_checkpoint_sha256'] == decision['b4']['checkpoint_sha256'] and
            canonical['model_fingerprint'] == decision['b4']['model_state_fingerprint'], 'Canonical B4 is not source/FP-bound')
    require(row.get('alias_of') == canonical['run_id'] and type(row.get('additional_encoding_calls')) is int and row['additional_encoding_calls'] == 0 and
            row['archive_member'] == canonical['archive_member'] and row['archive_sha256'] == canonical['archive_sha256'] and
            row['model_fingerprint'] == canonical['model_fingerprint'], 'Alias does not reference this verified B4 weight')
    require(row.get('alias_evidence') == dict(comparison=decision['comparison'], sources=sources), 'Alias proof/source evidence differs')


def bind_row_sources(meta, row, identity, frozen, snapshot, source_root=ROOT):
    """Bind current small source bytes to READY and the original input snapshot."""
    root = Path(source_root).resolve()
    canonical = lambda path: (Path(path) if Path(path).is_absolute() else root/Path(path)).resolve()
    require(row['run_id'] == identity['run_id'] == identity['actual_run_id'] == frozen['id'], 'Foreign actual run')
    require(canonical(row['source_checkpoint']) == canonical(identity['checkpoint']), 'Foreign source checkpoint')
    for field in ('config_path','summary_path'):
        path = canonical(frozen[field])
        require(canonical(identity[field]) == path, 'Foreign READY source path')
        matches = [v for key,v in identity['input_sha256'].items() if canonical(key) == path]
        require(len(matches) == 1 and row[field.removesuffix('_path')+'_sha256'] == matches[0], 'READY/config-summary SHA differs')
        raw = meta.opaque(path, matches[0])
        if field == 'config_path':
            relative = str(path.relative_to(root))
            expected = snapshot['configs'][relative]
            require(expected['sha256'] == matches[0] and expected['bytes'] == len(raw), 'Frozen snapshot/config binding differs')
    stamp = meta.payload_stat(row['source_checkpoint'], row['source_stat'])
    require(type(identity.get('checkpoint_bytes')) is int and type(identity.get('checkpoint_mtime_ns')) is int and
            stamp['bytes'] == identity['checkpoint_bytes'] and stamp['mtime_ns'] == identity['checkpoint_mtime_ns'], 'Frozen best checkpoint size/mtime differs')


def native_members(files, snapshot, model):
    expected = {str(absolute(path).resolve()): record for path,record in snapshot['model_files'].items()}
    require(len(files) == len(expected) == 6 and {Path(p).name for p in expected} ==
            {'model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md'}, 'Native six-file closure required')
    seen = set()
    for row in files:
        path = str(absolute(row['path']).resolve())
        require(path in expected and path not in seen and row['sha256'] == expected[path]['sha256'] and
                row['member'] == 'base/'+model+'-final/'+Path(path).name, 'Native base source/member/SHA differs')
        seen.add(path)
    return [row['member'] for row in files]


def member_inventory(manifest):
    rows = manifest['logical_runs']
    names = [r['metadata_member'] for r in rows] + [r['archive_member'] for r in rows if 'alias_of' not in r]
    names += [f['member'] for f in manifest['base_dependency']['files']] + EXTRAS
    # The frozen assembler enumerates gate entries, including repeated source paths.
    names += ['provenance/gate/'+str(i)+'_'+Path(r['path']).name for i,r in enumerate(manifest['fullphase_source_gate']['sources'])]
    require(len(names) == len(set(names)) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names), 'Unsafe/duplicate complete member inventory')
    return names


def validate_reserve(name, row, current):
    p = PROFILES[name]
    expected_path = '/data/202610/7kb/research_delivery/.extra_'+p['reserve']+'_weights_reserve_20261106'
    require(row['path'] == expected_path and current == row['actual_stat'] and current['device'] == 106 and
            current['inode'] == p['inode'] and current['uid'] == current['gid'] == 11471 and
            current['mode'] == 0o600 and current['links'] == 1 and current['bytes'] == p['capacity'] and
            current['allocated_bytes'] >= p['capacity'], 'Original extra best reserve identity/capacity differs')


def validate_batch_identity(name, launch, spec, folder, resource_path, resource_sha):
    argv = launch['argv']
    slot = 'slot2' if name == 'C2p8B_best' else 'slot1'
    require(argv == spec['argv'] and spec.get('package') == name and spec.get('slot') == slot and
            absolute(spec['cwd']).resolve() == ROOT and absolute(spec['logs']).resolve() == folder.resolve(), 'Foreign whole-batch identity')
    require(len(argv) >= 3 and absolute(argv[1]).resolve() == ROOT/'scripts/export_additional_endpoints.py' and
            argv[2] == 'execute', 'Whole batch is not the frozen exporter execution')
    pairs = dict(zip(argv[3::2],argv[4::2]))
    require(len(argv[3:]) % 2 == 0 and len(pairs)*2 == len(argv[3:]) and set(pairs) ==
            {'--scope','--scope-sha256','--slot','--package','--max-new-weights','--resource-review','--resource-review-sha256'}, 'Exact nonduplicate batch flags required')
    require(absolute(pairs['--scope']).resolve() == BASE/'scope.json' and pairs['--scope-sha256'] == SCOPE_SHA and
            pairs['--slot'] == slot and pairs['--package'] == name and int(pairs['--max-new-weights']) > 1 and
            absolute(pairs['--resource-review']).resolve() == absolute(resource_path).resolve() and
            pairs['--resource-review-sha256'] == resource_sha, 'Batch argv differs from this package/scope/resource authority')
    require(absolute(spec['authority']['path']).resolve() == absolute(resource_path).resolve() and
            spec['authority']['sha256'] == resource_sha, 'Batch specification authority differs')
    require(spec['env'].get('CUDA_VISIBLE_DEVICES') == '' and
            all(spec['env'].get(k) == '2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')), 'Whole batch CPU policy differs')


def import_pinned(path, digest, name, meta):
    meta.opaque(path, digest)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review(name, resource_path, resource_sha, output):
    started, cpu = time.monotonic(), time.process_time()
    require(not output.exists(), 'Preserve prior result; use a reviewed fresh namespace')
    folder = BASE/name
    require((folder/'completion.json').is_file(), 'Pending actual complete C-best package; no acceptance emitted')
    meta = Metadata(folder)
    meta.opaque(__file__)
    meta.opaque(ROOT/'scripts/publish_reserved_archive.py', PUBLISHER_SHA)
    e = import_pinned(ROOT/'scripts/export_additional_endpoints.py', EXPORTER_SHA, 'additional_best_metadata_frozen', meta)
    native = import_pinned(NATIVE_REVIEWER, NATIVE_REVIEWER_SHA, 'native_roundtrip_metadata_frozen', meta)
    scope = meta.read(BASE/'scope.json', SCOPE_SHA)
    phase = scope['phases'][name]
    context = phase['context']; p = PROFILES[name]; cp = context['phase']
    require(scope['packages'] == e.PACKAGES and scope['slots'] == e.SLOTS and cp['model'] == p['model'], 'Foreign preparation/model policy')
    for source in scope['code']:
        meta.opaque(source['path'], source['sha256'])
    for key, digestkey in [('active_registry', 'active_registry_sha256'), ('snapshot', 'snapshot_sha256'),
                           ('protocol', 'protocol_sha256'), ('queue', 'queue_sha256')]:
        meta.opaque(cp[key], cp[digestkey])
    require(meta.read(cp['active_registry']) == context['rows'] and meta.read(cp['snapshot']) == context['snapshot'] and
            meta.read(cp['protocol']) == context['protocol'], 'Scope context differs from frozen phase bytes')
    # Large training/model bytes were fully bound by v4 preparation, never rehashed here.
    for section in ('code', 'data', 'model_files'):
        for path, record in context['snapshot'][section].items():
            key = str(absolute(path).resolve()); frozen = scope['frozen_files'][key]
            require(frozen['sha256'] == record['sha256'], 'Snapshot/scope input SHA conflict')
            meta.payload_stat(path, frozen['stat'])
    for f in phase['base_files']:
        key = str(Path(f['path']).resolve()); frozen = scope['frozen_files'][key]
        require(frozen['sha256'] == f['sha256'], 'Native base SHA differs from frozen scope')
        meta.payload_stat(f['path'], frozen['stat'])
    native_members(phase['base_files'], context['snapshot'], p['model'])
    receipt = meta.read(folder/'completion.json')
    manifest = meta.read(folder/'additional_weights_manifest.json', receipt['manifest_sha256'])
    validate_headers(name, receipt, manifest)
    gate = manifest['fullphase_source_gate']
    require(gate.get('status') == 'complete_model_best_source_gate_passed', 'Full best-source gate required')
    for ref in gate['sources']:
        meta.opaque(ref['path'], ref['sha256'])
    # This existing pure validator only hashes already-bounded small gate/scope/manifest files and stats CP/TAR.
    e.validate_completion(name, folder, phase, BASE/'scope.json', receipt)
    ready_folder = ROOT/'artifacts/evaluation_ready_v5'/e.READY_NAMES[p['model']]
    meta.opaque(ready_folder/'READY.json'); meta.opaque(ready_folder/'manifest.json')
    identity = e.published(e.READY_NAMES[p['model']], meta)
    require(identity is not None and identity == gate['identity'], 'Current published READY/identity differs from completed package')
    identities = {b['run_id']: b for b in identity['records'] if b['kind'] == 'best'}
    require(identities == gate['records'] and set(identities) == set(phase['logical_run_ids']), 'Foreign/missing best cohort')
    for frozen in context['rows']:
        b = identities[frozen['id']]
        for key in ('config_path','summary_path'):
            matches = [value for path,value in b['input_sha256'].items() if absolute(path).resolve() == absolute(frozen[key]).resolve()]
            require(len(matches) == 1, 'Unique bound small config/summary required')
            meta.opaque(frozen[key], matches[0])
    master = meta.read(ROOT/'configs/supplemental_v1_queue.json'); state = meta.read(ROOT/'logs/supplemental_job_state.json')
    # Mutable master/state are historical observations; only individual training definitions/states are binding.
    for path in (ROOT/'configs/supplemental_v1_queue.json', ROOT/'logs/supplemental_job_state.json'):
        meta.observed.pop(str(path), None)
    decisions = {}
    if name != 'C2p8B_best':
        complete = meta.read(ROOT/'results/extended_phase_equivalence'/cp['phase']/'completion.json')
        require(complete.get('status') == 'passed' and complete.get('phase') == cp['phase'] and complete.get('model') == p['model'] and
                complete.get('expected_runs') == complete.get('completed_runs') == 24 and complete.get('comparisons_computed') == 6, 'Complete matching model phase required')
        require(complete['source_registry']['sha256'] == cp['active_registry_sha256'] and
                complete['input_snapshot']['sha256'] == cp['snapshot_sha256'], 'Foreign phase registry/snapshot')
        audit = meta.read(complete['audit']['path'], complete['audit']['sha256'])
        source_gate = audit.get('source_state_complete_gate', {})
        require(audit.get('status') == 'passed' and audit.get('phase') == cp['phase'] and audit.get('model') == p['model'] and
                not audit.get('source_violations') and source_gate.get('ready') is True and source_gate.get('completed_runs') == 24 and
                not source_gate.get('missing_runs') and not source_gate.get('failed_runs'), 'Source audit not passed')
        proof = meta.read(complete['proofs']['best']['path'], complete['proofs']['best']['sha256'])
        require(proof['source_audit'] == complete['audit'] and proof['source_registry'] == complete['source_registry'] and
                proof['source_snapshot'] == complete['input_snapshot'], 'Best proof source authority differs')
        for ref in (complete['audit'],complete['proofs']['best'],complete['source_registry'],complete['input_snapshot']):
            require(ref in gate['sources'], 'Required source/best proof omitted from package gate')
        decisions = e.best_pair_decisions(context, proof, identities)
        require(json.loads(json.dumps(decisions)) == gate['decisions'], 'Completed alias decisions differ from actual best proof')
    else:
        require(gate.get('decisions') == {}, 'No 2.8B D aliases exist')
        audits = [r for r in gate['sources'] if Path(r['path']).name == 'fullphase_source_audit.json']
        require(len(audits) == 1, 'Actual complete 2.8B source audit required')
        audit = meta.read(audits[0]['path'], audits[0]['sha256'])
        require(audit.get('status') == 'passed' and audit.get('model') == p['model'] and audit.get('phase') == cp['phase'] and
                audit.get('expected_runs') == audit.get('completed_runs') == 15 and not audit.get('violations') and
                audit.get('snapshot_sha256') == cp['snapshot_sha256'], 'Foreign/incomplete 2.8B phase audit')
    require(gate['final_source_gate']['status'] == 'complete_source_gate_passed', 'Original complete source gate required')
    expected_aliases = sum(d['equal'] for d in decisions.values())
    require(receipt['alias_count'] == expected_aliases and receipt['unique_weights'] == p['n']-expected_aliases, 'Actual best equality decisions and alias inventory differ')
    records = {r['run_id']:r for r in manifest['logical_runs']}; rowmap = {r['id']:r for r in context['rows']}
    base_sha = next(f['sha256'] for f in phase['base_files'] if Path(f['path']).name == 'model.safetensors')
    for rid, row in records.items():
        b = identities[rid]; frozen = rowmap[rid]
        require(b['kind'] == row['kind'] == 'best' and b['model_tag'] == row['model_tag'] == p['model'] and
                b['data_family'] == 'extended_v1' and b['role'] == 'extended' and b['phase'] == 'formal' and b['actual_run_id'] == rid and
                b['phase_name'] == e.READY_NAMES[p['model']].removesuffix('_nll') and
                all(b[key] == frozen[source] for key,source in [('training_seed','seed'),('group','group'),('ratio','ratio'),('weight','weight')]), 'Wrong best logical identity')
        original = meta.read(folder/'records'/(rid+'__best.json')); require(original == row, 'Manifest/record splice')
        bind_row_sources(meta, row, b, frozen, context['snapshot'])
        cfg, summary = e.validate_bound(b, frozen, 'best', p['model'], 'extended_v1', meta)
        require(cfg['model_revision'] == context['protocol']['model_revision'] and cfg['target_prediction_tokens'] == cp['target_prediction_tokens'] and
                b['full_training_committed_path']['target_prediction_tokens'] == cfg['target_prediction_tokens'], 'Original model revision/full horizon differs')
        definition, scheduler = e.source_definition(b, frozen, master, state); require_clean_rc(scheduler)
        require(definition == context['definitions'][rid] == row['definition'] and scheduler == row['completed_scheduler_record'], 'Actual argv/frozen completed job differs')
        item = dict(bound=b,row=frozen,definition=definition)
        meta.opaque(row['metadata_path'], row['metadata_sha256'], sidecar=True)
        e.validate_resume(row, item)
        exposure = e.selected_checkpoint_exposure(summary, 'best')
        require(row['actual_exposure'] == exposure and exposure['checkpoint_step'] == b['step'] and
                exposure['target_prediction_tokens'] == b['target_prediction_tokens_at_checkpoint'] and
                exposure['replay_prediction_tokens'] == b['replay_prediction_tokens_at_checkpoint'] and
                summary['counters']['target_prediction_tokens'] == cp['target_prediction_tokens'], 'Actual best/full-training exposure differs')
        e.verify_recorded_execution(context['baseline']['execution_provenance'], e.training_validation_execution(summary))
        require(row.get('status') == 'verified_additional_endpoint_not_yet_assembled' and row.get('package') == name and
                row.get('exporter_sha256') == EXPORTER_SHA and row.get('prepared_scope_sha256') == SCOPE_SHA and
                row.get('all_nonweight_metadata_roundtrip_verified') is True and row.get('GPU_used') is False and
                row.get('shared_storage_written') is False, 'Complete frozen best record required')
        require(row['source_fd_five_stat_before'] == row['source_fd_five_stat_after'] == row['source_stat'], 'Path/FD stability differs')
        meta.payload_stat(row['source_checkpoint'], row['source_stat'])
        if row['group'] in ('D','B4') and decisions:
            decision = decisions[int(row['training_seed'])]
            validate_alias_decision(row, records.get(decision['b4']['run_id']), decision, gate['sources'])
        if 'alias_of' not in row:
            require(type(row.get('additional_encoding_calls')) is int and row['additional_encoding_calls'] == 1, 'Independent encoding receipt missing')
            native.validate_roundtrip_report(row, base_sha)
            meta.payload_stat(row['archive_path'], row['archive_stat'])
        else:
            require(row['group'] == 'D' and decisions, 'Only actual true C best D/B4 proof supports alias')
    dependency = manifest['base_dependency']
    require(dependency == dict(kind='native_base_config_tokenizer_readme_included_no_external_package_needed',files=phase['base_files']), 'Native independent closure differs')
    names = member_inventory(manifest)
    require(len(names) == receipt['verified_tar_members'], 'Exact full TAR member count differs')
    # Read the actual ROOT resource proof supplied later, never manufacture permission.
    resource = meta.read(resource_path, resource_sha)
    for approved in set(e.SLOTS[e.PACKAGES[name]['slot']]).intersection(resource.get('approved_packages',[])):
        ref = resource['package_plans'][approved]['actual_sample_record']
        meta.opaque(ref['path'], ref['sha256'])
    authority = e.validate_resource_review(resource_path, resource_sha, SCOPE_SHA, e.PACKAGES[name]['slot'])
    require(name in authority['approved_packages'], 'ROOT authority did not approve this model package')
    whole_path = ROOT/'artifacts/provenance/additional_endpoint_actual_v4'/(name+'_batch/whole_child_timing.json')
    whole = meta.read(whole_path); require_clean_rc(whole)
    require(whole.get('GPU_used') is False and whole.get('publication_performed') is False, 'CPU-only nonpublication batch required')
    launch = meta.read(whole_path.parent/'child_launch.json')
    specification = meta.read(launch['spec_path'], launch['spec_sha256'])
    validate_batch_identity(name, launch, specification, whole_path.parent, resource_path, resource_sha)
    cap = meta.read(ROOT/'artifacts/provenance/extra_endpoint_reservations_v1/ROOT_full_capacity_acceptance.json', CAPACITY_SHA)
    reserve_row = next(r for r in cap['rows'] if r['name'] == p['reserve'])
    reserve = Path(reserve_row['path']); s = reserve.lstat()
    current = dict(device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,mode=stat.S_IMODE(s.st_mode),bytes=s.st_size,allocated_bytes=s.st_blocks*512,links=s.st_nlink)
    require(stat.S_ISREG(s.st_mode), 'Regular nonsymlink reserve required')
    validate_reserve(name, reserve_row, current)
    key = f'{os.major(s.st_dev):02x}:{os.minor(s.st_dev):02x}:{s.st_ino}'
    require(not any(key in line for line in Path('/proc/locks').read_text().splitlines()), 'Reserve lock held')
    tar = Path(receipt['tar_path']); destination = reserve.parent/tar.name
    require(not os.path.lexists(destination) and destination.parent.stat().st_dev == s.st_dev and
            receipt['tar_bytes'] <= current['allocated_bytes'], 'Destination exists or reserve insufficient')
    meta.payload_stat(tar); meta.finish()
    result = dict(status='passed_metadata_only_C_best_for_ROOT_review_not_publication_authority',package=name,
                  source=str(tar),expected_sha256=receipt['tar_sha256'],expected_bytes=receipt['tar_bytes'],
                  reserved=str(reserve),reserve_original=current,destination=str(destination),
                  logical_endpoints=p['n'],unique_weights=receipt['unique_weights'],alias_count=receipt['alias_count'],
                  metadata_sources=meta.evidence,payload_stats_only=meta.observed,derived_safe_TAR_members=names,
                  actual_best_decisions=gate['decisions'],whole_batch_timing_source=str(whole_path),
                  whole_batch_and_internal_endpoint_assembly_timers_are_nested=True,
                  metadata_sidecars_opened_only_as_bounded_opaque_hash_inputs=True,
                  original_TAR_member_hashes_and_full_TAR_SHA_attested_by_exporter_not_rehashed_here=True,
                  future_publisher_must_verify_fresh_source_and_written_prefix_SHA=True,
                  observed_reserve_lock_free_is_not_an_exclusive_lease=True,
                  review_wall_seconds=time.monotonic()-started,review_process_CPU_seconds=time.process_time()-cpu,
                  no_CP_TAR_ZIP_native_model_or_token_payload_opened=True,no_export_publish_GPU_or_scores=True,
                  ROOT_authorized=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),not_authority=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True,choices=PROFILES)
    parser.add_argument('--resource-review',required=True,type=Path)
    parser.add_argument('--resource-review-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args();review(a.package,a.resource_review,a.resource_review_sha256,a.output)
