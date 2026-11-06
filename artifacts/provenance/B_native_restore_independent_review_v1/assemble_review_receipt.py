#!/usr/bin/env python3
"""Bind independent metadata-only B native preparation acceptance; no payload opens."""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PREP=ROOT/'artifacts/provenance/B_native_restore_preparation_v1'
NATIVE=ROOT/'artifacts/provenance/C_native_final_restore_preparation_v1'
evidence={}

def bind(path, digest=None):
    path=Path(path)
    assert path.suffix in ('.py','.json','.md','.log') and path.stat().st_size < 32 << 20
    before=path.stat();raw=path.read_bytes();after=path.stat()
    sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    sha=hashlib.sha256(raw).hexdigest()
    assert sig(before)==sig(after) and (digest is None or digest==sha)
    ref=dict(path=str(path.resolve()),sha256=sha,bytes=len(raw))
    evidence[str(path.resolve())]=ref
    return raw,ref

_,runner=bind(PREP/'run_B_restore.py','6d7245e78e225c3879eaf931b85e7be073b4ba9777d7263b7e0e1c39b1630a02')
raw,planref=bind(PREP/'bound_plan_v2.json','61c19ee07d69936da329846fcb8aa8bd67ea8d8fe1ff8cd6b3084c032c3fa3c7')
plan=json.loads(raw)
raw,revision=bind(PREP/'preparation_receipt_v2.json','6403053c3d224aa0e6b8db82aeee3751946dc65a082022e01df53b48093b1a1e')
for name,entry in json.loads(raw)['files'].items():
    bind(PREP/name,entry['sha256'])
for name,digest in plan['reused_helper_pins'].items():
    bind(NATIVE/name,digest)
for ref in plan['evidence'].values():
    bind(ref['path'],ref['sha256'])
bind(PREP/'schema_guard_checks.json')
bind(PREP/'lock_regression_tests/stderr.log')
for name in ('retention.json','run_B_restore.py','README.md','bound_plan.json',
             'ROOT_execution_authority_FALSE_TEMPLATE.json','preparation_receipt.json','delivery_receipt.json'):
    bind(PREP/'retained_before_shared_sequential_lock'/name)
for name in ('probe_bound_identity.py','retained_e3aa_run_B_restore.py','bound_identity_probe.json',
             'bound_identity_stdout.log','bound_identity_stderr.log','whole_bound_identity_probe_timing.json',
             'probe_lock_revision.py','lock_revision_probe.json','lock_revision_stdout.log',
             'lock_revision_stderr.log','whole_lock_revision_probe_timing.json'):
    bind(HERE/name)
for name in ('scripts/archive_model_weights.py','src/replay_study/weight_archive.py',
             'src/replay_study/__init__.py','scripts/export_additional_endpoints.py'):
    bind(ROOT/name)
bind(__file__)
_,doc=bind(ROOT/'docs/B固定首例native持久恢复准备_独立审阅.md')
timers=[json.loads((HERE/name).read_text()) for name in
        ('whole_bound_identity_probe_timing.json','whole_lock_revision_probe_timing.json')]
assert all(t['returncode']==0 for t in timers)
probe=json.loads((HERE/'lock_revision_probe.json').read_text())
assert probe['status']=='passed_independent_B_shared_lock_revision_preparation_only'
assert probe['current_runner_sha256']==runner['sha256'] and probe['current_plan_sha256']==planref['sha256']
# Actual durable payload is never opened here; lstat only.
tar=Path(plan['archive']);s=tar.lstat();assert not tar.is_symlink()
stamp=dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,uid=s.st_uid,gid=s.st_gid,
           mode=oct(s.st_mode&0o7777),mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
assert stamp==plan['archive_stat']
assert all(not Path(plan[k]).exists() and not Path(plan[k]).is_symlink()
           for k in ('qa_directory','actual_evidence_directory'))
review=dict(schema_version=1,status='passed_prepared_B_native_restore_after_shared_lock_fix',
    created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),runner=runner,plan=planref,
    author_revision=revision,document=doc,
    fixed_identity=dict(run_id=plan['run_id'],training_seed=11,generation_seed=11,selection_draw=1001,
        bank_chunks=20,kind='final',steps=306,target_prediction_tokens=10000000,replay_prediction_tokens=500000,
        expected_complete_model_fingerprint=plan['selected']['model_fingerprint'],tensor_count=292),
    closure=dict(selected_members=12,selected_member_bytes=2058150693,actual_manifest_row_exact=True,
        native_base_members=6,archived_tools_only=True,current_archive_stat=stamp,
        historical_full_TAR_SHA=plan['historical_full_TAR_sha256'],historical_SHA_not_recomputed=True,
        future_extraction_requires_every_selected_member_fresh_SHA=True),
    resolved_findings=[dict(original_runner_sha256='e3aa2b08e5361ab7fee92ef7a2f20042df80c1dd57f4d64a58e7285127f37d41',
        issue='Initial thin body bypassed shared C native restore lock',
        repair='Same nonblocking flock acquired before QA/resources/TAR and held across body; toy exception releases',
        original_bytes_retained=True),
        dict(issue='Internal main-entry timer is not a complete Python process wall timer',
             repair='Source and README now explicitly exclude startup/top-level imports/final serialization; future ROOT external Popen parent required')],
    timing=dict(independent_probe_children=timers,manual_review_whole_wall_seconds=None,
        manual_review_whole_CPU_seconds=None,
        future_execution_wait_cap_seconds=600,future_cleanup_budget_seconds=12,margin_seconds=300,
        protected_end_utc='2026-11-07T02:56:32Z',must_admit_before_utc='2026-11-07T02:41:32Z',
        body_decode_checker_and_internal_PG_timers_nested_under_external_parent=True),
    checks=dict(actual_FALSE_authority_refused=True,shared_lock_exception_probe=True,
        old_body_AST_unchanged_within_lock=True,wrong_scope_probes_refused=5,
        pure_900seconds_boundary_refused=True,existing_PG_helper_original_SHA_unchanged=True,
        repeated_existing_large_test_suites=False),
    boundaries=dict(actual_bind_or_execute_or_restore=False,real_QA_created=False,
        true_authority_created=False,real_TAR_headers_or_ZIP_base_CP_token_score_bytes_read=0,
        original_helpers_scientific_core_or_archived_codec_changed=False,
        actual_recovery_pass_claimed=False,Adam_resume_or_all60_coverage_claimed=False),
    source_evidence=list(evidence.values()))
with (HERE/'REVIEW.json').open('x') as f:json.dump(review,f,ensure_ascii=False,indent=2);f.write('\n')
digest=hashlib.sha256((HERE/'REVIEW.json').read_bytes()).hexdigest()
complete=dict(status=review['status'],review=dict(path=str(HERE/'REVIEW.json'),sha256=digest),
    runner=runner,plan=planref,document=doc,actual_recovery_pass_claimed=False,ROOT_authorized=False)
with (HERE/'COMPLETE.json').open('x') as f:json.dump(complete,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(COMPLETE_sha256=hashlib.sha256((HERE/'COMPLETE.json').read_bytes()).hexdigest(),
                     REVIEW_sha256=digest,small_source_evidence_files=len(evidence),
                     small_source_evidence_bytes=sum(r['bytes'] for r in evidence.values()))))
