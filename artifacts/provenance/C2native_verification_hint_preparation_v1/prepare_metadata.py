#!/usr/bin/env python3
"""Small proof/stat preparation only; does not open derived payload FDs."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/tmp/grad-research-20261106/worktree')
script=HERE/'hint_once.py'
spec=importlib.util.spec_from_file_location('inert_oneoff_metadata_functions',script)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def save(path,o):
    with path.open('x') as f:json.dump(o,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def ref(path):
    b=Path(path).read_bytes();return dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(b).hexdigest())

def main():
    sources=[dict(path=str(ROOT/'scripts/export_additional_endpoints.py'),sha256='c93e0c1d35fe39b3d71155a126db841b76648ba6b3afd58d475be4eb9c9da853'),
      dict(path='/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/scope.json',sha256='7adb8f7f0461fd929995dd370ff2a59b2abc5c0d7750a39e745d21d5546d9de3'),
      dict(path=str(ROOT/'artifacts/provenance/C_native_final_CPU_restore_actual_v1/2p8b/ROOT_preexecution_resource_gate_attempt1.json'),sha256='d55f6f5f32cf58dc280e12b04f81c5e7f976f7762b6f5c55e9eda664f04eb110'),
      dict(path=str(ROOT/'artifacts/provenance/C_native_final_CPU_restore_actual_v1/2p8b/bound_plan.json'),sha256='573c4f4924298408d53ef5fa7b07441549260b89dbae34c5e634dd2baf82a4f2')]
    for p in sources:M.small(p)
    ids=['B1_s22_w0p05','B0_s22_w0','M_s11_w0p05','B4_s11_w0p05']
    copies=HERE/'record_copies';copies.mkdir();items=[]
    for id in ids:
        path=M.BASE/'records'/('extended_v2_pythia_2p8b_m8_h25m_formal_'+id+'__best.json')
        rref=ref(path);b=M.small(rref);r=json.loads(b);p=Path(r['decode']['restored']);st=M.target_stat(p)
        with (copies/path.name).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
        source_stat=M.signature(Path(r['source_checkpoint']).stat())
        M.require(all(source_stat[k]==r['source_stat'][k] for k in r['source_stat']),'Original CP metadata changed')
        items.append(dict(record=rref,record_copy=ref(copies/path.name),path=str(p),before=st,
            historical_restored_file_sha256=r['decode']['restored_file_sha256'],model_fingerprint=r['model_fingerprint'],
            source_CP_path_metadata_only=r['source_checkpoint'],source_CP_sha256=r['source_checkpoint_sha256'],
            source_CP_five_stat_historical=r['source_stat'],source_CP_current_stat_metadata_only=source_stat))
    total=sum(x['before']['bytes'] for x in items)
    M.require(total==44404036408 and total<=48<<30,'Wrong four-file total')
    plan=dict(schema_version=1,status='prepared_not_execution_authority',at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        required_memory_bytes=48698310656,maximum_advice_bytes=48<<30,total_advised_range_bytes=total,
        exporter_sha256=sources[0]['sha256'],prepared_scope_sha256=sources[1]['sha256'],source_proofs=sources,items=items,
        fallback_TAR_selected=False,operation='Per completed derived verification file readonly same-FD fsync then DONTNEED offset0 exact length',
        no_original_CP_base_ZIP_TAR_payload_open=True,resource_threshold_not_lowered=True,
        conditional_skip='If any fresh pre-hint observation is already >=required memory, write SKIP without any fadvise',
        scope_limit='Four fixed completed derived verification outputs; producer active output excluded; no global cache operation')
    validated=M.validate_items(plan);refs=M.references(validated);mem=M.memory()
    save(HERE/'prepared_plan.json',plan)
    planref=ref(HERE/'prepared_plan.json');sourceref=ref(script)
    save(HERE/'ROOT_authority_FALSE_TEMPLATE.json',dict(authorized=False,authorized_by='ROOT',scope=M.SCOPE,plan=planref,source=sourceref,
        note='Preparation only. ROOT must independently review then separately create an exact authority; this template cannot execute.'))
    receipt=dict(status='prepared_fixed_four_completed_derived_outputs_no_hint_executed',at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        plan=planref,source=sourceref,preparation_reference_scan=refs,preparation_memory_observation=mem,
        actual_execution_must_repeat_all_guards=True,required_memory_bytes=48698310656,
        skip_if_already_sufficient=mem['conservative_available_bytes']>=48698310656,
        target_payload_FDs_opened=False,target_payload_bytes_read=0,hint_executed=False,TRUE_authority_created=False,tests_run=False,
        known_limitations=['Reference scan is a point-in-time observation, not a producer lease',
            'Historical record complete SHA and current full stat bind bytes without rereading payload',
            'DONTNEED is advisory; exact reclaimed bytes and future headroom not guaranteed',
            'No ROOT restore authority or scheduler/GPU action is implied'],
        previous_failed_resource_gate=dict(path=sources[2]['path'],sha256=sources[2]['sha256'],observed_bytes=14605336576,required_bytes=48698310656),
        preparation_source=ref(Path(__file__)))
    save(HERE/'preparation_receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],plan=planref,source=sourceref,total_bytes=total,
        current_observed_available_bytes=mem['conservative_available_bytes'],current_preparation_no_FD_maps=True),indent=2))

if __name__=='__main__':main()
