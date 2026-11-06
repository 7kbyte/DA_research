#!/usr/bin/env python3
"""Bind independent prepared-tool review to small opaque evidence; no real checker."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BOOK = ROOT / 'artifacts/provenance/C_best_publication_runbook_v1'
evidence = {}

def bind(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    assert path.suffix in ('.py', '.json', '.md', '.log')
    before = path.stat()
    assert stat.S_ISREG(before.st_mode) and before.st_size < 32 << 20
    raw = path.read_bytes()
    after = path.stat()
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    digest = hashlib.sha256(raw).hexdigest()
    assert stamp(before) == stamp(after) and (expected is None or digest == expected)
    ref = dict(path=str(path.resolve()), sha256=digest, bytes=len(raw))
    evidence[str(path.resolve())] = ref
    return raw, ref

_, source = bind(BOOK / 'review_metadata.py', 'a6250f88dd909567b02baa28f5d458becff169e17d5e85a37ec50f8c1e8811eb')
bind(BOOK / 'test_review_metadata.py', 'a9cc4bb2a06d87ee386415f26341577e28ca2513d632d365e8a9b8cc83aee9ea')
bind(BOOK / 'README.md', 'd5175db449cedee2f4e9d72473665e3163326e15c22e272b7dc097f490a30e4e')
raw, _ = bind(BOOK / 'preparation_receipt_v2.json', '510fcf9f7e03bc805235b17a91ab3470ac5051e2d575dd39b94257738349d53b')
revision = json.loads(raw)
for path, record in revision['source_evidence'].items():
    bind(path, record['sha256'])
raw, _ = bind(BOOK / 'preparation_receipt.json', '626a8fb0542ad9eca8d79cbee48901aad874aaf4c62a96960ffe49b5911ab091')
original = json.loads(raw)
for record in original['source_evidence']:
    path = Path(record['path'])
    if path.name not in ('review_metadata.py', 'test_review_metadata.py') or path.parent != BOOK:
        bind(path, record['sha256'])
for path in [BOOK / 'preparation_tests_importlib/stderr.log',
             BOOK / 'retained_a31f_before_explicit_authorized_gate/retention.json']:
    bind(path)
scope_path = Path('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/scope.json')
scope = json.loads(bind(scope_path, '7adb8f7f0461fd929995dd370ff2a59b2abc5c0d7750a39e745d21d5546d9de3')[0])
for name in ('C410_best', 'C1B_best', 'C2p8B_best'):
    phase = scope['phases'][name]['context']['phase']
    for key, sha in [('active_registry','active_registry_sha256'), ('snapshot','snapshot_sha256'),
                     ('protocol','protocol_sha256'), ('queue','queue_sha256')]:
        bind(phase[key], phase[sha])
proofdir = ROOT / 'results/extended_phase_equivalence/extended_v2_pythia_410m_m64_formal'
for name in ('completion.json', 'audit.json', 'best_model_equivalence_proofs.json'):
    bind(proofdir / name)
ready = ROOT / 'artifacts/evaluation_ready_v5/extended_pythia_410m_100m_nll'
for name in ('READY.json', 'manifest.json'):
    bind(ready / name)
batch = ROOT / 'artifacts/provenance/additional_endpoint_actual_v4/C410_best_batch'
launch = json.loads(bind(batch / 'child_launch.json')[0])
bind(launch['spec_path'], launch['spec_sha256'])
probe = json.loads(bind(HERE / 'probe_explicit_authority_v2_results.json')[0])
assert probe['status'] == 'passed_independent_explicit_authority_gate_and_minimal_AST_delta'
assert probe['source']['sha256'] == source['sha256']
bind(HERE / 'retained_a31f_review_metadata.py', 'a31f4062f53e2cfb64e028f822a30dab6e6e500918322b320716e84783f885b5')
for name in ('probe_metadata.py','probe_results.json','probe_stdout.log','probe_stderr.log',
             'whole_probe_timing.json','probe_explicit_authority_v2.py',
             'probe_explicit_authority_v2_stdout.log','probe_explicit_authority_v2_stderr.log',
             'whole_explicit_authority_v2_probe_timing.json'):
    bind(HERE / name)
doc = bind(ROOT / 'docs/Cbest发布前元数据核对器_独立审阅.md')[1]
bind(__file__)
timers = [json.loads((HERE / name).read_text()) for name in
          ('whole_probe_timing.json', 'whole_explicit_authority_v2_probe_timing.json')]
assert all(t['returncode'] == 0 for t in timers)
review = dict(
    schema_version=1, status='passed_prepared_checker_independent_review_after_explicit_authority_fix',
    created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), checker=source, document=doc,
    resolved_finding=dict(original_checker_sha256='a31f4062f53e2cfb64e028f822a30dab6e6e500918322b320716e84783f885b5',
        defect='Inherited resource validator did not require authorized is True; independent in-memory probe reproduced False accepted',
        current_actual_C410_authority_was_true=True,
        repair='New checker requires explicit boolean true before inherited validator; old bytes and receipts retained',
        old_frozen_exporter_and_helpers_changed=False),
    scope=dict(C410_best=24, C1B_best=24, C2p8B_best=15,
               only_best=True, true_alias_requires_both_source_SHA_and_complete_FP=True,
               false_D_requires_independent_encode=True, C2p8B_no_D=True,
               native_model_files_per_package=6, exact_extra_reserve_identity=True,
               strict_integer_rc0_no_stop=True, actual_batch_argv_spec_authority_bound=True),
    validation=dict(initial_independent_probe_checks=6,
                    independent_revision_true_plus_five_rejections=True,
                    minimal_AST_delta_verified=True, author_original_tests_passed=171,
                    author_additional_permission_tests_passed=5,
                    author_suites_not_rerun=True,
                    actual_C410_launch_and_native6_schema_probed_only=True),
    cost=dict(independent_probe_children=timers, manual_engineering_parent_wall_seconds=None,
              manual_engineering_parent_CPU_seconds=None,
              timer_scope='Two independent probe children only; manual review/receipt assembly unknown, do not treat child sums as whole review elapsed'),
    boundaries=dict(actual_checker_called=False, real_package_acceptance=False,
        publication_authority_created=False, actual_export_or_publish_started=False,
        score_files_opened=False, CP_ZIP_TAR_base_token_payload_bytes_read=0,
        original_helpers_or_live_exporter_changed=False,
        future_each_package_requires_complete_batch_and_ROOT_authority=True,
        package_TAR_SHA_and_member_hashes_are_historical_exporter_evidence=True,
        future_publisher_fresh_verification_required=True),
    source_evidence=list(evidence.values()))
with (HERE / 'REVIEW.json').open('x') as f:
    json.dump(review, f, ensure_ascii=False, indent=2); f.write('\n')
raw = (HERE / 'REVIEW.json').read_bytes()
complete = dict(status=review['status'], review=dict(path=str(HERE / 'REVIEW.json'),
    sha256=hashlib.sha256(raw).hexdigest()), checker=source, document=doc,
    ROOT_authorized=False, real_package_acceptance=False)
with (HERE / 'COMPLETE.json').open('x') as f:
    json.dump(complete, f, ensure_ascii=False, indent=2); f.write('\n')
print(json.dumps(dict(complete_sha256=hashlib.sha256((HERE/'COMPLETE.json').read_bytes()).hexdigest(),
                     review_sha256=complete['review']['sha256'], small_evidence_files=len(evidence),
                     evidence_payload_bytes=sum(r['bytes'] for r in evidence.values())), ensure_ascii=False))
