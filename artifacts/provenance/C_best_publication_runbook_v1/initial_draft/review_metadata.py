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
        require(p.suffix not in {'.pt', '.zip', '.tar', '.safetensors'} or allowed_sidecar, 'Weight payload reads forbidden')
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
            manifest.get('status') == 'complete_all_logical_endpoints_verified', 'Actual complete package required')
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
        require('alias_of' not in row and row.get('additional_encoding_calls') == 1, 'False proof/non-D requires independent encoding')
        return
    require(canonical is not None and canonical['run_id'] == decision['b4']['run_id'] and canonical['group'] == 'B4' and
            canonical['source_checkpoint_sha256'] == decision['b4']['checkpoint_sha256'] and
            canonical['model_fingerprint'] == decision['b4']['model_state_fingerprint'], 'Canonical B4 is not source/FP-bound')
    require(row.get('alias_of') == canonical['run_id'] and row.get('additional_encoding_calls') == 0 and
            row['archive_member'] == canonical['archive_member'] and row['archive_sha256'] == canonical['archive_sha256'] and
            row['model_fingerprint'] == canonical['model_fingerprint'], 'Alias does not reference this verified B4 weight')
    require(row.get('alias_evidence') == dict(comparison=decision['comparison'], sources=sources), 'Alias proof/source evidence differs')


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
    require(len(phase['base_files']) == 6 and {Path(f['path']).name for f in phase['base_files']} ==
            {'model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md'}, 'All six native files required')
    receipt = meta.read(folder/'completion.json')
    manifest = meta.read(folder/'additional_weights_manifest.json', receipt['manifest_sha256'])
    validate_headers(name, receipt, manifest)
    gate = manifest['fullphase_source_gate']
    require(gate.get('status') == 'complete_model_best_source_gate_passed', 'Full best-source gate required')
    for ref in gate['sources']:
        meta.opaque(ref['path'], ref['sha256'])
    # This existing pure validator only hashes already-bounded small gate/scope/manifest files and stats CP/TAR.
    e.validate_completion(name, folder, phase, BASE/'scope.json', receipt)
    identity = e.published(e.READY_NAMES[p['model']], meta)
    require(identity is not None and identity == gate['identity'], 'Current published READY/identity differs from completed package')
    identities = {b['run_id']: b for b in identity['records'] if b['kind'] == 'best'}
    require(identities == gate['records'] and set(identities) == set(phase['logical_run_ids']), 'Foreign/missing best cohort')
    master = meta.read(ROOT/'configs/supplemental_v1_queue.json'); state = meta.read(ROOT/'logs/supplemental_job_state.json')
    # Mutable master/state are historical observations; only individual training definitions/states are binding.
    for path in (ROOT/'configs/supplemental_v1_queue.json', ROOT/'logs/supplemental_job_state.json'):
        meta.observed.pop(str(path), None)
    decisions = {}
    if name != 'C2p8B_best':
        complete = meta.read(ROOT/'results/extended_phase_equivalence'/cp['phase']/'completion.json')
        require(complete.get('status') == 'passed' and complete.get('phase') == cp['phase'] and complete.get('model') == p['model'] and
                complete.get('expected_runs') == complete.get('completed_runs') == 24, 'Complete matching model phase required')
        require(complete['source_registry']['sha256'] == cp['active_registry_sha256'] and
                complete['input_snapshot']['sha256'] == cp['snapshot_sha256'], 'Foreign phase registry/snapshot')
        audit = meta.read(complete['audit']['path'], complete['audit']['sha256'])
        require(audit.get('status') == 'passed' and not audit.get('source_violations') and
                audit.get('source_state_complete_gate', {}).get('ready') is True, 'Source audit not passed')
        proof = meta.read(complete['proofs']['best']['path'], complete['proofs']['best']['sha256'])
        require(proof['source_audit'] == complete['audit'] and proof['source_registry'] == complete['source_registry'] and
                proof['source_snapshot'] == complete['input_snapshot'], 'Best proof source authority differs')
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
    records = {r['run_id']:r for r in manifest['logical_runs']}; rowmap = {r['id']:r for r in context['rows']}
    base_sha = next(f['sha256'] for f in phase['base_files'] if Path(f['path']).name == 'model.safetensors')
    for rid, row in records.items():
        b = identities[rid]; frozen = rowmap[rid]
        require(b['kind'] == row['kind'] == 'best' and b['model_tag'] == row['model_tag'] == p['model'] and
                b['data_family'] == 'extended_v1' and b['role'] == 'extended' and b['actual_run_id'] == rid, 'Wrong best logical identity')
        original = meta.read(folder/'records'/(rid+'__best.json')); require(original == row, 'Manifest/record splice')
        for key in ('config_path','summary_path'):
            matches = [v for path,v in b['input_sha256'].items() if Path(path).resolve() == Path(frozen[key]).resolve()]
            require(len(matches) == 1 and row[key.removesuffix('_path')+'_sha256'] == matches[0], 'Frozen READY/config-summary SHA differs')
            meta.opaque(frozen[key], matches[0])
        cfg, summary = e.validate_bound(b, frozen, 'best', p['model'], 'extended_v1', meta)
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
            require(row.get('additional_encoding_calls') == 1, 'Independent encoding receipt missing')
            native.validate_roundtrip_report(row, base_sha)
            meta.payload_stat(row['archive_path'], row['archive_stat'])
        else:
            require(row['group'] == 'D' and decisions, 'Only actual true C best D/B4 proof supports alias')
    dependency = manifest['base_dependency']
    require(dependency == dict(kind='native_base_config_tokenizer_readme_included_no_external_package_needed',files=phase['base_files']), 'Native independent closure differs')
    names = [r['metadata_member'] for r in records.values()] + [r['archive_member'] for r in records.values() if 'alias_of' not in r]
    names += [f['member'] for f in phase['base_files']] + EXTRAS
    names += ['provenance/gate/'+str(i)+'_'+Path(r['path']).name for i,r in enumerate(gate['sources'])]
    require(len(names) == len(set(names)) == receipt['verified_tar_members'] and
            all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names), 'Exact safe full TAR member inventory differs')
    # Read the actual ROOT resource proof supplied later, never manufacture permission.
    meta.opaque(resource_path, resource_sha)
    authority = e.validate_resource_review(resource_path, resource_sha, SCOPE_SHA, e.PACKAGES[name]['slot'])
    require(name in authority['approved_packages'], 'ROOT authority did not approve this model package')
    whole_path = ROOT/'artifacts/provenance/additional_endpoint_actual_v4'/(name+'_batch/whole_child_timing.json')
    whole = meta.read(whole_path); require_clean_rc(whole)
    cap = meta.read(ROOT/'artifacts/provenance/extra_endpoint_reservations_v1/ROOT_full_capacity_acceptance.json', CAPACITY_SHA)
    reserve_row = next(r for r in cap['rows'] if r['name'] == p['reserve'])
    reserve = Path(reserve_row['path']); s = reserve.lstat()
    current = dict(device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,mode=stat.S_IMODE(s.st_mode),bytes=s.st_size,allocated_bytes=s.st_blocks*512,links=s.st_nlink)
    require(stat.S_ISREG(s.st_mode) and current == reserve_row['actual_stat'] and
            current['inode'] == p['inode'] and current['bytes'] == p['capacity'], 'Original extra best reserve identity/capacity differs')
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
