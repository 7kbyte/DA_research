#!/usr/bin/env python3
"""Prepare/future-execute one fixed, published native Cfinal CPU recovery example."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import traceback

HERE = Path(__file__).resolve().parent
SCRATCH = Path('/tmp/grad-research-20261106/reproduction_qa')
PROTECTED_END = dt.datetime(2026, 11, 7, 2, 56, 32, tzinfo=dt.timezone.utc)
PROFILES = {
    'pythia-1b': dict(phase='extended_v2_pythia_1b_m32_formal', tag='1b', logical=24, unique=21,
                      target=100_000_000, ready='extended_pythia_1b_100m_nll', max_wall_seconds=600),
    'pythia-2.8b': dict(phase='extended_v2_pythia_2p8b_m8_h25m_formal', tag='2p8b', logical=15, unique=15,
                        target=25_000_000, ready='extended_pythia_2p8b_25m_nll', max_wall_seconds=1200),
}
TOOL_PINS = {
    'tools/scripts/archive_model_weights.py': '28c5841cae8e7fd87d8cb62e7f756dad40ce15cb558710c31e48dfb45fde668a',
    'tools/src/replay_study/weight_archive.py': '3c2bcf9e38773ae4f69c27f85f342356dbefcb53d817be4321da6e9c4e550e1a',
    'tools/src/replay_study/__init__.py': 'b2dce8cc5ea80a67a90b1798232163a7684874487b5ea4102e60851c7017f520',
}
REVIEWER_SHA = 'c596a56d70b32cedce7bc9d2e00a0923477165aac6c24ee04ed78e8680413627'
FREEZE_SHA = '50786c00724008eed6a96cb753b442295152f9dea96d73a2c75f8fcf16033528'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def small(proof):
    path = Path(proof['path'])
    require(path.suffix not in {'.pt', '.zip', '.tar', '.safetensors'} and path.stat().st_size <= 32 << 20,
            'Only explicit small JSON evidence may be opened')
    before = signature(path)
    raw = path.read_bytes()
    require(sha_bytes(raw) == proof['sha256'] and signature(path) == before, 'Small evidence SHA/stat differs')
    return json.loads(raw)


def signature(path):
    path = Path(path)
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and not path.is_symlink(), 'Regular non-symlink file required')
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, uid=s.st_uid, gid=s.st_gid,
                mode=oct(stat.S_IMODE(s.st_mode)), mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_name(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts and str(p) == name and '\\' not in name,
            'Unsafe/noncanonical member name')
    return name


def build_plan(model, proofs, values, archive_stat, manifest_bytes):
    """Small-evidence-only binder; no TAR header or payload reads here."""
    p = PROFILES[model]
    completion, manifest, review, publication, acceptance = (values[k] for k in
        ('completion', 'manifest', 'metadata_review', 'publication', 'root_acceptance'))
    for key in ('model', 'phase'):
        require(completion.get(key) == (model if key == 'model' else p['phase']), 'Wrong completed model/phase')
    require(completion.get('status') == 'passed' and completion.get('fullphase_source_gate_passed') is True,
            'Whole model must actually be complete')
    require(manifest.get('status') == 'complete_phase_all_final_roundtrips_verified' and
            manifest.get('model_tag') == model and manifest.get('phase_name') == p['phase'] and
            manifest.get('kind') == 'fixed_exposure_final' and manifest.get('optimizer_included') is False,
            'Only complete original native Cfinal package accepted')
    require(completion.get('manifest_sha256') == proofs['manifest']['sha256'], 'Completion/manifest binding differs')
    expected = (p['logical'], p['unique'], p['logical'] - p['unique'])
    require(all(tuple(v.get(k) for k in ('logical_final_endpoints', 'unique_weights', 'alias_count')) == expected
                for v in (completion, manifest)), 'Whole logical/unique inventory differs')
    require(all(completion.get(k) is True for k in ('file_fsync_completed', 'parent_directory_fsync_completed')),
            'Completed TAR fsync evidence required')
    require(manifest.get('freeze', {}).get('sha256') == FREEZE_SHA and
            manifest.get('fullphase_gate', {}).get('status') == 'complete_source_gate_passed', 'Frozen fullphase gate required')
    identity = manifest['fullphase_gate']['identity_manifest']
    require(identity.get('published_ready') == proofs['ready'] and identity.get('sha256') == proofs['identity']['sha256'],
            'Actual READY/fullphase identity proof differs')
    require(values['ready'].get('status') == 'ready' and values['ready'].get('ready') is True and
            values['ready'].get('id') == p['ready'] and values['ready'].get('manifest') == proofs['identity'],
            'Actual matched wholemodel READY required')
    finals = {r['run_id']: r for r in values['identity']['records'] if r['kind'] == 'final'}
    require(len(finals) == p['logical'] and identity['records'] == finals, 'All frozen finals required')
    require(review.get('status') == 'passed_metadata_only_for_ROOT_review_not_publication_authority' and
            review.get('model') == model and review.get('phase') == p['phase'] and
            review.get('derived_verified_TAR_member_count') == completion.get('verified_tar_members'),
            'Passed original native publication metadata review required')
    require(proofs['metadata_reviewer']['sha256'] == REVIEWER_SHA, 'Original reviewed metadata validator differs')
    for key in ('completion', 'manifest'):
        evidence = review.get('metadata_sources', {}).get(str(Path(proofs[key]['path']).resolve()), {})
        require(evidence.get('sha256') == proofs[key]['sha256'], 'Publication reviewer did not bind actual ' + key)
    require(acceptance.get('status') == 'accepted' and acceptance.get('model') == model and
            acceptance.get('phase') == p['phase'] and acceptance.get('scope') == 'original_Cfinal_native_package_publication' and
            all(acceptance.get(k) == proofs[k] for k in ('completion', 'manifest', 'metadata_review', 'metadata_reviewer', 'publication')),
            'ROOT publication acceptance must bind all exact evidence; FALSE/pending rejected')
    require(publication.get('status') == 'passed' and publication.get('source_sha256') == publication.get('destination_sha256') ==
            completion.get('tar_sha256') == review.get('expected_sha256') and
            publication.get('bytes') == completion.get('tar_bytes') == review.get('expected_bytes') == archive_stat['bytes'],
            'Actual published complete TAR SHA/size differs')
    for flag in ('file_fsync_after_copy', 'prefix_sha_verified_before_truncate', 'file_fsync_after_truncate',
                 'atomic_destination_commit_noreplace', 'parent_directory_fsync', 'original_inode_preserved'):
        require(publication.get(flag) is True, 'Missing durable publisher condition: ' + flag)
    recorded_stat = publication['destination_stat']
    require(all(archive_stat.get(k) == recorded_stat.get(k) for k in archive_stat), 'Published TAR identity changed')
    require(publication['destination'] == review['destination'], 'Destination differs from accepted publication review')
    expected_tar = '/data/202610/7kb/research_delivery/extended_C_' + p['tag'] + '_final_weights_lossless_v1.tar'
    require(publication['destination'] == expected_tar, 'Must use the actual durable original Cfinal TAR')
    rows = manifest['logical_runs']
    require(len(rows) == p['logical'] and {r['run_id'] for r in rows} == set(finals), 'Full unique frozen run inventory required')
    for item in rows:
        definition = item['definition']; child = item['completed_scheduler_record']
        digest = sha_bytes(json.dumps(definition, sort_keys=True).encode())
        require(item.get('definition_sha256') == digest and child.get('definition_sha256') == digest and
                child.get('status') == 'completed' and type(child.get('returncode')) is int and child['returncode'] == 0 and
                not child.get('missing_outputs') and 'stop_requested_at' not in child and 'stop_requested' not in child and
                child.get('argv') == definition['argv'], 'Every frozen training job needs strict rc0/no-stop/actual argv')
        require(item.get('source_checkpoint_sha256') == finals[item['run_id']]['checkpoint_sha256'], 'Fullphase source CP binding differs')
    run_id = p['phase'] + '_B4_s11_w0p05'
    row = next(r for r in rows if r['run_id'] == run_id)
    require(row.get('group') == 'B4' and type(row.get('seed')) is int and row['seed'] == 11 and 'alias_of' not in row,
            'The preselected B4 seed11 must be independently encoded')
    exposure = row['actual_exposure']
    require(exposure['target_prediction_tokens'] == p['target'] and exposure['replay_prediction_tokens'] == p['target'] // 20 and
            row['source_checkpoint_sha256'] == finals[run_id]['checkpoint_sha256'], 'Fixed final exposure/source identity differs')
    require(row.get('all_nonweight_metadata_roundtrip_verified') is True, 'Complete own metadata roundtrip required')
    fp = row['model_fingerprint']
    require(type(fp.get('state_elements')) is int and fp['state_elements'] > 0 and len(fp.get('sha256_model_state', '')) == 64,
            'Complete model fingerprint required')
    for stage in ('encode', 'decode', 'bytewise_verification'):
        require(row[stage].get('model_fingerprint') == fp, 'Nested roundtrip fingerprint differs')
    require(row['encode'].get('status') == 'encoded' and row['decode'].get('status') == 'decoded_and_hash_verified' and
            row['bytewise_verification'].get('status') == 'passed' and row['bytewise_verification'].get('all_tensor_bytes_equal') is True,
            'Original complete roundtrip evidence required')
    count = row['decode']['tensor_raw_hashes_verified']
    require(type(count) is int and count > 0 and row['bytewise_verification']['tensor_count'] == count, 'Complete tensor inventory differs')
    require(row['encode']['source_checkpoint_sha256'] == row['bytewise_verification']['source_checkpoint_sha256'] == row['source_checkpoint_sha256'] and
            row['encode']['archive_sha256'] == row['bytewise_verification']['archive_sha256'] == row['archive_sha256'],
            'Original CP/ZIP SHA provenance differs')
    dependency = manifest['base_dependency']
    require(dependency.get('kind') == 'native_base_config_tokenizer_readme_included', 'Native single-TAR closure required')
    native = dependency['files']
    require(len(native) == 6 and {Path(r['member']).name for r in native} ==
            {'model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'README.md'},
            'All six native base/config/tokenizer/README members required')
    require(all(str(PurePosixPath(r['member']).parent) == 'base/' + model + '-final' for r in native),
            'Native six-file closure must use the matching package model directory')
    native_base = 'base/' + model + '-final/model.safetensors'
    require(next(r for r in native if r['member'] == native_base)['sha256'] == dependency['base_sha256'] == row['encode']['base_sha256'],
            'Native base SHA differs')
    members = {r['member']: dict(sha256=r['sha256'], bytes=r['bytes']) for r in native}
    members['extended_C_final_manifest.json'] = dict(sha256=proofs['manifest']['sha256'], bytes=manifest_bytes)
    members[row['archive_member']] = dict(sha256=row['archive_sha256'], bytes=row['archive_stat']['bytes'])
    members[row['metadata_member']] = dict(sha256=row['metadata_sha256'], bytes=row['metadata_stat']['bytes'])
    members.update({k: dict(sha256=v, bytes=None) for k, v in TOOL_PINS.items()})
    require(len(members) == 12, 'Selected member collision')
    for name in members:
        safe_name(name)
    raw_bytes = fp['state_elements'] * 4
    return dict(schema_version=1, status='prepared_not_execution_authority', model=model, phase=p['phase'],
                run_id=run_id, archive=publication['destination'], archive_stat=archive_stat,
                historical_full_TAR_sha256=publication['destination_sha256'], whole_TAR_rehash=False,
                evidence=proofs, members=members, native_base=native_base, selected=row,
                tensor_count=count, model_raw_bytes=raw_bytes,
                qa_directory=str(SCRATCH / ('native_Cfinal_' + p['tag'] + '_B4s11_final_v1')),
                resource_policy=dict(required_scratch_bytes=sum(r['bytes'] or 0 for r in members.values()) + raw_bytes + (2 << 30),
                    required_reclaim_aware_memory_bytes=4 * raw_bytes + (4 << 30), cpu_threads=2,
                    max_wall_seconds=p['max_wall_seconds'], deadline_utc=PROTECTED_END.isoformat(), admission_margin_seconds=300),
                boundaries='One preselected numerical recovery example; not new science, original-CP fresh comparison, Adam resume, or retraining')


def resource_gate(policy, qa_parent, now=None, cgroup=Path('/sys/fs/cgroup')):
    now = now or dt.datetime.now(dt.timezone.utc)
    deadline = dt.datetime.fromisoformat(policy['deadline_utc'])
    require(deadline <= PROTECTED_END and (deadline - now).total_seconds() > policy['max_wall_seconds'] + policy['admission_margin_seconds'],
            'Insufficient protected delivery tail; do not start a new recovery')
    disk = shutil.disk_usage(qa_parent)
    require(disk.free >= policy['required_scratch_bytes'], 'Insufficient isolated scratch disk')
    maximum = (cgroup / 'memory.max').read_text().strip()
    require(maximum != 'max', 'Explicit cgroup memory ceiling required for this platform')
    current = int((cgroup / 'memory.current').read_text())
    mem = {k: int(v) for k, v in (r.split() for r in (cgroup / 'memory.stat').read_text().splitlines())}
    # memory.current contains clean file cache. Only bounded inactive clean file is credited here;
    # active file, anon, shmem, unevictable and dirty/writeback bytes receive no reclaim credit.
    clean_inactive = max(0, min(mem.get('inactive_file', 0), mem.get('file', 0) - mem.get('shmem', 0)) -
                         mem.get('file_dirty', 0) - mem.get('file_writeback', 0) - mem.get('unevictable', 0))
    available = max(0, int(maximum) - current) + clean_inactive
    require(available >= policy['required_reclaim_aware_memory_bytes'], 'Insufficient conservative reclaim-aware cgroup headroom')
    events = {k: int(v) for k, v in (r.split() for r in (cgroup / 'memory.events').read_text().splitlines())}
    return dict(disk_free_bytes=disk.free, memory_max_bytes=int(maximum), memory_current_bytes=current,
                anon_bytes=mem.get('anon'), shmem_bytes=mem.get('shmem'), unevictable_bytes=mem.get('unevictable'),
                file_bytes=mem.get('file'), active_file_bytes=mem.get('active_file'), inactive_file_bytes=mem.get('inactive_file'),
                dirty_bytes=mem.get('file_dirty'), writeback_bytes=mem.get('file_writeback'),
                conservative_clean_inactive_credit_bytes=clean_inactive, conservative_available_bytes=available,
                memory_events=events, caveat='Reclaim proxy and admission check, not a reservation or peak-RSS guarantee')


class CountedFile:
    def __init__(self, stream):
        self.stream = stream; self.bytes_read = 0

    def read(self, n=-1):
        raw = self.stream.read(n); self.bytes_read += len(raw); return raw

    def seek(self, *args):
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()


def extract_selected(plan, qa):
    """Exact regular members only; no extractall, CP reads, or whole-TAR hash."""
    source = Path(plan['archive'])
    require(signature(source) == plan['archive_stat'], 'Published TAR changed before extraction')
    selected = plan['members']; checks = []
    with source.open('rb', buffering=0) as fd:
        before = os.fstat(fd.fileno())
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                tuple(plan['archive_stat'][k] for k in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns')), 'Opened TAR identity differs')
        counted = CountedFile(fd)
        with tarfile.open(fileobj=counted, mode='r:') as archive:
            headers = archive.getmembers()
            names = [m.name for m in headers]
            require(len(names) == len(set(names)), 'Duplicate TAR headers rejected')
            byname = {m.name: m for m in headers}
            for name, expected in selected.items():
                safe_name(name)
                member = byname[name]
                require(member.isfile() and (expected['bytes'] is None or member.size == expected['bytes']), 'Unsafe member/type/size')
                output = qa / name; output.parent.mkdir(parents=True, exist_ok=True)
                require(not output.exists() and not output.is_symlink(), 'Selected extraction never overwrites')
                h = hashlib.sha256(); copied = 0
                with archive.extractfile(member) as payload, output.open('xb') as target:
                    for raw in iter(lambda: payload.read(8 << 20), b''):
                        target.write(raw); h.update(raw); copied += len(raw)
                    target.flush(); os.fsync(target.fileno())
                require(copied == member.size and h.hexdigest() == expected['sha256'], 'Fresh extracted-member SHA differs')
                checks.append(dict(member=name, sha256=h.hexdigest(), bytes=copied))
        after = os.fstat(fd.fileno())
        require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) ==
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns), 'Open TAR changed during extraction')
    require(signature(source) == plan['archive_stat'], 'Published TAR identity changed during extraction')
    return dict(files=checks, selected_member_payload_bytes=sum(r['bytes'] for r in checks),
                source_TAR_bytes_returned_to_Python=counted.bytes_read,
                scope='Selected members plus TAR headers; serialized read bytes, not measured physical disk traffic', full_TAR_SHA_recomputed=False)


def execute(plan_proof, authority_proof):
    plan = small(plan_proof); authority = small(authority_proof)
    require(authority.get('authorized') is True and authority.get('plan') == plan_proof and
            authority.get('model') == plan['model'] and authority.get('run_id') == plan['run_id'] and
            authority.get('scope') == 'one_fixed_native_Cfinal_CPU_recovery' and authority.get('sequential_only') is True,
            'Explicit exact single-model ROOT authority required; FALSE templates cannot execute')
    runner = dict(path=str(Path(__file__).resolve()), sha256=sha_bytes(Path(__file__).read_bytes()))
    checker = dict(path=str(HERE / 'check_restored.py'), sha256=sha_bytes((HERE / 'check_restored.py').read_bytes()))
    require(authority.get('runner') == runner and authority.get('checker') == checker and
            plan['runner'] == runner and plan['checker'] == checker, 'Execution source changed')
    require(plan.get('status') == 'prepared_not_execution_authority', 'Prepared bound plan required')
    qa = Path(plan['qa_directory'])
    require(qa == SCRATCH / ('native_Cfinal_' + PROFILES[plan['model']]['tag'] + '_B4s11_final_v1') and
            not qa.exists() and not qa.is_symlink(), 'Fixed fresh isolated directory required')
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with (SCRATCH / '.native_Cfinal_restore.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = resource_gate(plan['resource_policy'], SCRATCH)
        qa.mkdir(); (qa / 'logs').mkdir()
        write_new(qa / 'bound_plan.json', plan)
        shutil.copyfile(HERE / 'check_restored.py', qa / 'check_restored.py')
        env = os.environ.copy()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE'):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONNOUSERSITE='1')
        started = time.monotonic(); su = resource.getrusage(resource.RUSAGE_SELF); cu = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = dict(status='running', model=plan['model'], run_id=plan['run_id'], plan=plan_proof, authority=authority_proof,
                      resource_admission=observed, stages=[], network_calls_requested=False, GPU_used=False,
                      original_checkpoint_or_livebase_opened=False, whole_source_TAR_rehash=False)
        def stage(name, argv):
            before = resource.getrusage(resource.RUSAGE_CHILDREN); start = time.monotonic()
            remaining = plan['resource_policy']['max_wall_seconds'] - (start - started)
            require(remaining > 0, 'Operation wall cap exhausted')
            with (qa / 'logs' / (name + '.stdout.log')).open('xb') as out, (qa / 'logs' / (name + '.stderr.log')).open('xb') as err:
                child = subprocess.Popen(argv, cwd=qa, env=env, stdout=out, stderr=err)
                try:
                    rc = child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(); raise
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            result['stages'].append(dict(name=name, argv=argv, returncode=rc, wall_seconds=time.monotonic() - start,
                child_user_seconds=after.ru_utime - before.ru_utime, child_sys_seconds=after.ru_stime - before.ru_stime,
                timer_relationship='Sequential stage nested in whole runner wall/CPU'))
            require(type(rc) is int and rc == 0, 'Recovery child failed: ' + name)
        try:
            start = time.monotonic(); pc = time.process_time()
            result['extraction'] = extract_selected(plan, qa)
            result['extraction'].update(wall_seconds=time.monotonic() - start, process_cpu_seconds=time.process_time() - pc,
                                         timer_relationship='Parent region nested in whole runner')
            stage('decode', [sys.executable, '-I', 'tools/scripts/archive_model_weights.py', '--threads', '2', 'decode',
                  '--archive', plan['selected']['archive_member'], '--base', plan['native_base'], '--output', 'restored/B4_seed11_final.pt'])
            stage('independent_fingerprint', [sys.executable, '-I', 'check_restored.py'])
            result['independent_check'] = json.loads((qa / 'independent_fingerprint.json').read_text())
            require(result['independent_check']['status'] == 'passed', 'Independent recovered model check failed')
            require(signature(plan['archive']) == plan['archive_stat'], 'Durable TAR changed during restore')
            result['status'] = 'passed'
        except Exception as error:
            result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        finally:
            sa = resource.getrusage(resource.RUSAGE_SELF); ca = resource.getrusage(resource.RUSAGE_CHILDREN)
            result.update(runner_body_wall_seconds=time.monotonic() - started,
                parent_self_user_seconds=sa.ru_utime-su.ru_utime, parent_self_sys_seconds=sa.ru_stime-su.ru_stime,
                children_user_seconds=ca.ru_utime-cu.ru_utime, children_sys_seconds=ca.ru_stime-cu.ru_stime,
                children_peak_RSS_bytes_cumulative=ca.ru_maxrss*1024,
                accounting='Body timer excludes interpreter/import/preflight; external whole-child timer is inclusive. Never sum nested stages with parent.',
                claim='One exact model-only recovery; no new original checkpoint uint8 comparison, optimizer recovery, Adam resume, or training')
            write_new(qa / 'actual_restore_receipt.json', result)
        require(result['status'] == 'passed', 'Failed recovery retained in isolated QA directory')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('bind')
    prepare.add_argument('--model', choices=PROFILES, required=True)
    prepare.add_argument('--sources', required=True, help='Explicit proof map; no directory discovery')
    prepare.add_argument('--sources-sha256', required=True)
    prepare.add_argument('--output', required=True)
    run = commands.add_parser('execute')
    for flag in ('plan', 'plan-sha256', 'authority', 'authority-sha256'):
        run.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    if args.command == 'bind':
        sources = small(dict(path=args.sources, sha256=args.sources_sha256))
        require(set(sources) == {'completion', 'manifest', 'metadata_review', 'metadata_reviewer', 'publication',
                                 'root_acceptance', 'ready', 'identity'}, 'Explicit fixed metadata source inventory required')
        values = {key: small(value) for key, value in sources.items() if key != 'metadata_reviewer'}
        require(sha_bytes(Path(sources['metadata_reviewer']['path']).read_bytes()) == REVIEWER_SHA, 'Reviewed validator source differs')
        manifest_bytes = Path(sources['manifest']['path']).stat().st_size
        binding = build_plan(args.model, sources, values, signature(values['publication']['destination']), manifest_bytes)
        binding['runner'] = dict(path=str(Path(__file__).resolve()), sha256=sha_bytes(Path(__file__).read_bytes()))
        binding['checker'] = dict(path=str(HERE / 'check_restored.py'), sha256=sha_bytes((HERE / 'check_restored.py').read_bytes()))
        write_new(args.output, binding)
    else:
        execute(dict(path=args.plan, sha256=args.plan_sha256), dict(path=args.authority, sha256=args.authority_sha256))


if __name__ == '__main__':
    main()
