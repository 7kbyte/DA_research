#!/usr/bin/env python3
"""Bind prepared-tool review to small source/receipts; no actual score inputs."""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
REPAIR=ROOT/'artifacts/provenance/C_model_nll_review_v2_schema_repair_v1'
PREP=ROOT/'artifacts/provenance/C_model_nll_review_v2_preparation'
evidence={}
def bind(path,expected=None):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    assert path.suffix in ('.json','.md','.py','.log','.png')
    assert path.stat().st_size<32<<20
    before=path.stat();raw=path.read_bytes();after=path.stat()
    sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    sha=hashlib.sha256(raw).hexdigest()
    assert sig(before)==sig(after) and (expected is None or expected==sha)
    ref=dict(path=str(path.resolve()),sha256=sha,bytes=len(raw));evidence[ref['path']]=ref
    return raw,ref

_,source=bind(ROOT/'scripts/review_C_model_nll_results_v2.py','7a48fb77a07d5d0d757c7f67f473dd49d4e262ffe497dfbe655184211be3d727')
raw,repair_complete=bind(REPAIR/'COMPLETE.json','a70fe9a78cf1d200c651f8304f79947b047de43cc5cb3f84bd8e0dc6befbf98e')
repair=json.loads(raw)
for key in ('old_reviewer','new_reviewer','retained_manifest','source_manifest','actual_schema_audit',
            'actual_schema_whole_child_timing'):
    ref=repair[key];bind(ref['path'],ref['sha256'])
bind(repair['regression']['whole_child_timing']['path'],repair['regression']['whole_child_timing']['sha256'])
sources=json.loads(bind(REPAIR/'source_manifest.json')[0])
for ref in sources['sources']:bind(ref['path'],ref['sha256'])
audit=json.loads(bind(REPAIR/'actual_C410_schema_audit.json')[0])
assert len(audit['checks'])==36 and all(c['passed'] for c in audit['checks'])
bind(audit['source_inventory']['path'],audit['source_inventory']['sha256'])
for name in ('regression_whole_child_timing.json','regression_stdout.log','regression_stderr.log',
             'actual_schema_audit_whole_child_timing.json','actual_schema_audit_stderr.log',
             'regression_attempt2_stdout.log','regression_attempt2_stderr.log',
             'actual_schema_audit_attempt2_stdout.log','actual_schema_audit_attempt2_stderr.log'):
    bind(REPAIR/name)
raw,prep_complete=bind(PREP/'COMPLETE.json');prepared=json.loads(raw)
for name in ('source_manifest','QC','targeted_tests','toy_cli_whole_parent','toy_cli_nested'):
    ref=prepared[name];bind(ref['path'],ref['sha256'])
bind(ROOT/'scripts/review_C_nll_results.py')
for p in ('src/replay_study/nll_v2_analysis.py','src/replay_study/analysis.py',
          'src/replay_study/supplemental_analysis_guard_v3.py','scripts/analyze_supplemental_phase_v3.py'):
    bind(ROOT/p)
for name in ('probe_numeric_and_sources.py','probe_result.json','whole_probe_timing.json',
             'probe_stdout.log','probe_stderr.log','synthetic_contact_8.png'):
    bind(HERE/name)
probe=json.loads((HERE/'probe_result.json').read_text())
assert probe['status']=='passed_independent_asymmetric_math_and_sources_only'
for ref in probe['all_eight_synthetic_plot_sources']:bind(ref['path'],ref['sha256'])
timer=json.loads((HERE/'whole_probe_timing.json').read_text());assert timer['returncode']==0
bind(__file__)
_,doc=bind(ROOT/'docs/C模型NLL复核v2_独立工程审阅.md')
review=dict(schema_version=1,status='passed_prepared_C_model_nll_reviewer_independent_engineering_review',
    created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),reviewer=source,document=doc,
    author_repair_completion=repair_complete,author_initial_preparation_completion=prep_complete,
    independent_findings='No new blocking defect found; previously identified ROOT/root literal repaired and independently verified minimal',
    scopes={'pythia-1b':dict(training_runs=24,logical_endpoints=48,contrasts=6,full_horizon=100000000,
                            author_tested_alias_counts=[0,3,6]),
            'pythia-2.8b':dict(training_runs=15,logical_endpoints=30,contrasts=3,full_horizon=25000000,
                              author_zero_alias_case_supported=True)},
    verified=dict(seven_numeric_AST_nodes_unchanged=True,exact_production_single_literal_repair=True,
        whole_final_best_gate=True,strict_integer_rc0_no_stop_actual_definition=True,
        bound_successor_queue_and_original_readiness_state=True,original_check_analyze_state_same=True,
        root_authority_and_preserved_first_read_and_operational_repair=True,
        model_extended_test_identity_matched_initial=True,source_bound_complete_model_equivalence=True,
        alias_zero_extra_forward=True,full_training_cost_actual_run_dedup=True,
        conditional_bootstrap_replicates=5000,conditional_bootstrap_seed=20261106,
        asymmetric_manual_interval_cases=4,negative_G_CI_order_swap=True,
        complete_48_30_grids_and_partial_refusal=True,
        CSV_tables_per_model=13,figures_per_model=4,figure_formats=['png','pdf','svg']),
    visual=dict(scope='Eight existing SYNTHETIC 1B/2.8B PNG panels only, contact sheet independently inspected',
        row_shared_scales_and_all_groups_seed_points_readable=True,
        actual_result_figures_inspected=False),
    timing=dict(independent_whole_probe_child=timer,manual_engineering_whole_wall_seconds=None,
        manual_engineering_whole_CPU_seconds=None,
        author_41tests_and_36schema_and_prior_toyCLIs_not_repeated=True,
        nested_timers_are_not_additive=True,failed_author_preparation_operations_retained=True),
    boundaries=dict(actual_new_C_or_behavior_score_bytes_opened=0,
        actual_model_reviewer_run=False,actual_reviewer_gate=False,original_check_analyze=False,
        CP_base_archive_tensor_payload_reads=0,GPU_started=False,true_authority_created=False,
        live_frozen_sources_changed=False,new_statistical_rule_or_result_selection=False),
    source_evidence=list(evidence.values()))
with (HERE/'REVIEW.json').open('x') as f:json.dump(review,f,ensure_ascii=False,indent=2);f.write('\n')
sha=hashlib.sha256((HERE/'REVIEW.json').read_bytes()).hexdigest()
complete=dict(status=review['status'],review=dict(path=str(HERE/'REVIEW.json'),sha256=sha),
    reviewer=source,document=doc,actual_model_result_review_claimed=False,ROOT_authorized=False)
with (HERE/'COMPLETE.json').open('x') as f:json.dump(complete,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(COMPLETE_sha256=hashlib.sha256((HERE/'COMPLETE.json').read_bytes()).hexdigest(),
    REVIEW_sha256=sha,small_source_evidence_files=len(evidence),
    small_metadata_and_synthetic_image_bytes=sum(r['bytes'] for r in evidence.values()))))
