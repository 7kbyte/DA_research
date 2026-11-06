"""Seal independent read-only preparation review; only small source/evidence bytes."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
PREP=ROOT/'artifacts/provenance/C_native_final_restore_preparation_v1'
FIRST=ROOT/'artifacts/provenance/C_native_final_restore_independent_review_v1'
def proof(p):
    p=Path(p)
    assert p.suffix in ('.py','.json','.md','.log') and p.stat().st_size<32<<20
    return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
expected={PREP/'run_whole_child.py':'2a2d09daac8504b099a5cd9682dff722bbad0519922ba39beedf4e96e9498266',
    PREP/'restore_native_final.py':'bfc748c776f18dff86c378c5c55ca72dbc63995cfc40ddfb96eb919486610db1',
    PREP/'check_restored.py':'6b7df26abef4ce0bd09dabbd7878a5a9585c8681d3cb97115f912dd542d3fa8d',
    PREP/'source_manifest_v2.json':'6655937525ecb8ec56e614b0be58578808e6038cc0c931799b7ba9c16cf227f7',
    PREP/'preparation_receipt_v2.json':'5a8d8582cc29b94f4000f572dcc1c70869739b2ebac826168138cea235123b82',
    PREP/'timeout_repair_v1/repair_receipt.json':'39d0d94d552d13d93455e549cf99e548092eddea2891c2f298542f952d09523b'}
assert all(proof(p)['sha256']==h for p,h in expected.items())
manifest=json.loads((PREP/'source_manifest_v2.json').read_text())
files=[]
for path,value in manifest['files'].items():
    observed=proof(ROOT/path);assert observed['sha256']==value['sha256'];files.append(observed)
for path,value in manifest['scientific_core_10'].items():
    assert proof(ROOT/path)['sha256']==value['expected_sha256']==value['actual_sha256']
for path,digest in manifest['retained_existing_helpers'].items():assert proof(ROOT/path)['sha256']==digest
probe=json.loads((FIRST/'timeout_fix_probe.json').read_text())
assert probe['fixed_timeout_probe']['returncode']==124 and probe['fixed_timeout_probe']['independently_observed_descendant_absent']
assert probe['simulated_D_state_unconfirmed_cleanup']['whole_group_absence_confirmed'] is False
paths=[*expected,FIRST/'INITIAL_REVIEW.json',FIRST/'probe_timeout.py',FIRST/'timeout_probe.json',FIRST/'probe_timeout_fix.py',FIRST/'timeout_fix_probe.json',
    ROOT/'docs/Cnative_final恢复准备_独立审阅_v1.md',ROOT/'docs/Cnative_final恢复准备_独立修复验收_v2.md',Path(__file__)]
future=[]
for model,tag in [('pythia-1b','1b'),('pythia-2.8b','2p8b')]:
    for path in [Path('/tmp/grad-research-20261106/delivery/extended_C_final_v1')/model/'completion.json',
        Path('/data/202610/7kb/research_delivery')/('extended_C_'+tag+'_final_weights_lossless_v1.tar'),
        PREP.parent/('C1B_final_publication_actual_v1' if tag=='1b' else 'C2p8B_final_publication_actual_v1')/'ROOT_acceptance.json']:
        future.append({'path':str(path),'exists_at_review':path.exists(),'payload_opened':False})
result={'status':'passed_independent_preparation_review_after_timeout_repair','actual_restore_authorized':False,
    'source_and_evidence_proofs':[proof(p) for p in paths],'author_source_manifest_files_fresh_SHA_verified':files,
    'scientific_10_and_retained_helpers_unchanged':True,
    'schema_scope_review':['original native exporter/completion/manifest/publication reviewer/publisher/ROOTacceptance crossbinding',
        '12 selected members, matching native base and pinned archived tools only','safe selected extraction/current TAR identity/selected byte SHA',
        'independent full FP32 state fingerprint and fresh new serialized SHA; no optimizer or original CP re-read'],
    'resolved_blocker':'Timeout always nonzero and cleanup checks entire PG after direct runner exit; bounded TERM/KILL/reaping, explicit unresolved failure',
    'independent_fixed_toy':proof(FIRST/'timeout_fix_probe.json'),'original_failure_probe_retained':proof(FIRST/'timeout_probe.json'),
    'independent_probe_complete_parent_timing':probe['whole_probe_parent_timing'],
    'nested_wrapper_wall_seconds':probe['fixed_timeout_probe']['whole_child_timing']['whole_child_process_wall_seconds'],
    'admission':probe['admission_boundaries'],'D_state_limit':probe['limits'],
    'future_stat_only_observations':future,
    'future_execution_conditions':['actual complete accepted original native package sources','metadata-only bind exact proofs',
        'separate ROOT execution authority for fixed B4s11 final plus reviewed source hashes','resource/time admission at launch',
        'sequential outer wrappers; wait full PG teardown confirmation before next restore'],
    'original_29_test_suite_not_repeated':True,'actual_bind_or_execute_called':False,'actual_TAR_ZIP_base_CP_payload_bytes_read':0,
    'real_score_outputs_parsed':False,'TRUE_authority_created':False,'GPU_started':False,
    'whole_task_engineering_review_wall_CPU':'not instrumented; not represented as zero; explicit probe timers only'}
with (OUT/'COMPLETE.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(proof(OUT/'COMPLETE.json'))
