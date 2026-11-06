#!/usr/bin/env python3
"""Export the frozen remaining 48 logical primary endpoints, CPU only.

The tested v1 weight codec is imported unchanged. No score outputs are opened.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_study.analysis import _check_summary, _protocol_status, selected_checkpoint_exposure
from replay_study.weight_archive import decode_weights, encode_weights, file_sha256, verify_roundtrip

SCRATCH = Path("/tmp/grad-research-20261106")
DELIVERY = SCRATCH / "delivery"
ENDPOINTS = DELIVERY / "remaining_primary_endpoints"
SOURCE_MANIFEST = ROOT / "docs/primary_evaluation_v2_execution_matched_nll_manifest.json"
SOURCE_MANIFEST_SHA = "d1502bcaff5cdcd75d984a4b8776f1cb4027b596f474835bfe9ed18875d9fd18"
BASE = SCRATCH / "models/pythia-410m-final"
BASE_SHA = "3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b"
REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
STATE = ROOT / "results/primary_remaining_endpoints_export_state.json"
PLAN = ROOT / "docs/primary_remaining_endpoints_export_plan.json"
PACKAGES = {
    "final": dict(name="budget_final_weights",logical=12,unique=12,limit_bytes=19_882_049_536),
    "best": dict(name="primary_best_weights",logical=36,unique=33,limit_bytes=50 * 2**30),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def proof(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha256(path), bytes=Path(path).stat().st_size)


def signature(path):
    s = Path(path).stat()
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def select_remaining_records(manifest):
    """Selection uses only the pre-frozen factorial identities, never scores."""
    expected = {(g,s,0. if g=="B0" else .05,0. if g=="B0" else .05)
                for g in ("B0","B1","B2","B3","B4","D","M","C") for s in (11,22,33)}
    expected |= {(g,s,b,.05) for g in ("B4","M") for s in (11,22,33) for b in (.01,.1)}
    rows = manifest["records"]
    if len(rows)!=72 or len({r["id"] for r in rows})!=72:
        raise ValueError("complete frozen primary final+best72 manifest required")
    for kind in ("final","best"):
        current = [r for r in rows if r["kind"]==kind]
        if (len(current)!=36 or {(r["group"],r["training_seed"],r["ratio"],r["weight"]) for r in current}!=expected):
            raise ValueError("primary model/seed/b/lambda factorial matrix changed")
    if any(r.get("role")!="primary" or r.get("phase")!="formal" or r.get("phase_name")!="primary" or
           r.get("model_tag")!="pythia-410m" or r.get("data_family")!="main_v3" or
           r.get("selection_seed") is not None or r.get("generation_seed") is not None or
           r.get("is_training_alias") is not False or r["actual_run_id"]!=r["run_id"] for r in rows):
        raise ValueError("remaining delivery cannot mix supplementary/provisional/model/data identities")
    selected = [dict(r) for r in rows if r["kind"]=="best" or
                (r["kind"]=="final" and r["group"] in {"B4","M"} and r["ratio"] in {.01,.1})]
    lookup = {(r["group"],r["training_seed"],r["ratio"],r["weight"],r["kind"]):r for r in selected}
    for r in selected:
        canonical = lookup["B4",r["training_seed"],r["ratio"],r["weight"],"best"] if r["group"]=="D" else r
        r["delivery_endpoint_id"] = r["run_id"] + "__" + r["kind"]
        r["weight_endpoint_id"] = canonical["run_id"] + "__" + canonical["kind"]
        r["weight_source_record_id"] = canonical["id"]
        r["alias_of"] = r["weight_endpoint_id"] if r["group"]=="D" else None
    if len(selected)!=48 or len({r["weight_endpoint_id"] for r in selected})!=45 or sum(bool(r["alias_of"]) for r in selected)!=3:
        raise ValueError("delivery must contain exactly48 logical endpoints,45 unique weights,3 D-best aliases")
    return sorted(selected,key=lambda r:(r["kind"]!="final",r["training_seed"],r["group"],r["ratio"]))


def validate_alias(record, canonical, proof_payload, proof_sha):
    bound = record["source_model_equivalence"]
    if (record["group"]!="D" or record["kind"]!="best" or not bound or
            proof_payload.get("status")!="passed" or bound["proof_sha256"]!=proof_sha or
            bound["algorithm"]!="sha256_sorted_key_dtype_shape_contiguous_uint8_v1"):
        raise ValueError("D-best alias requires the frozen complete weight equivalence proof")
    pair = proof_payload["comparisons"][bound["comparison_index"]]
    sides = {side["run_id"]:side for side in (pair["left"],pair["right"])}
    if (pair.get("seed")!=record["training_seed"] or pair.get("kind")!="identical_bank_null_control" or
            pair.get("bitwise_model_state_equal") is not True or set(sides)!={record["run_id"],canonical["run_id"]}):
        raise ValueError("alias proof does not identify this exact same-seed pair")
    for row in (record,canonical):
        side=sides[row["run_id"]]
        if (side["kind"]!="best" or side["checkpoint_sha256"]!=row["checkpoint_sha256"] or
                side["model_revision"]!=REVISION or side["model_state_fingerprint"]!=bound["fingerprint"] or
                Path(side["checkpoint"]).resolve()!=Path(row["checkpoint"]).resolve()):
            raise ValueError("alias source checkpoint/revision/fingerprint mismatch")
    return dict(proof_sha256=proof_sha,comparison_id=pair["id"],algorithm=bound["algorithm"],
                model_fingerprint=bound["fingerprint"],bitwise_model_state_equal=True,
                left=pair["left"],right=pair["right"])


def checkpoint_metadata(saved):
    """Keep *all* original nonweight/nonoptimizer fields, including RNG/streams."""
    return {k:v for k,v in saved.items() if k not in {"model_state","optimizer_state"}}


def same_metadata(a,b):
    if isinstance(a,torch.Tensor):
        return (isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and
                torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    if isinstance(a,dict):
        return isinstance(b,dict) and set(a)==set(b) and all(same_metadata(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(same_metadata(x,y) for x,y in zip(a,b))
    if isinstance(a,float):
        return isinstance(b,float) and np.float64(a).tobytes()==np.float64(b).tobytes()
    return a==b


def validate_saved_source(record, saved, summary, registry_row):
    _check_summary(summary,registry_row)
    if summary.get("status")!="completed" or summary["counters"]["target_prediction_tokens"]!=10_000_000:
        raise ValueError("source must be the completed frozen main run")
    if saved["config"]!=summary["config"] or saved["config"]["model_revision"]!=REVISION:
        raise ValueError("source checkpoint config differs from the completed pinned summary")
    exposure=selected_checkpoint_exposure(summary,record["kind"])
    for field,key in (("step","checkpoint_step"),("target_prediction_tokens","target_prediction_tokens"),("replay_prediction_tokens","replay_prediction_tokens")):
        expected=exposure[key]
        manifest_key={"step":"step","target_prediction_tokens":"target_prediction_tokens_at_checkpoint","replay_prediction_tokens":"replay_prediction_tokens_at_checkpoint"}[field]
        if saved["counters"][field]!=expected or record[manifest_key]!=expected:
            raise ValueError("checkpoint/manifest/summary actual endpoint exposure mismatch")
    if record["kind"]=="best":
        if saved.get("checkpoint_kind")!="model_only_best_target_validation" or saved.get("selection")!=summary["best_target_validation"]:
            raise ValueError("best checkpoint must retain its exact validation selection record")
    elif (saved.get("validation_trajectory")!=summary["validation_trajectory"] or
          saved.get("best_target_validation")!=summary["best_target_validation"]):
        raise ValueError("final checkpoint lost its validation/selection metadata")
    return exposure


def preflight():
    if file_sha256(SOURCE_MANIFEST)!=SOURCE_MANIFEST_SHA:
        raise ValueError("frozen main NLL identity manifest SHA differs")
    registry_path=ROOT/"configs/formal_deterministic_registry.json"
    audit_path=ROOT/"results/formal_run_audit.json"
    proof_path=ROOT/"results/formal_best_model_equivalence_proofs.json"
    _protocol_status(ROOT/"docs/protocol_status.json",registry_path,"active")
    audit=read(audit_path)
    if audit.get("status")!="passed" or audit.get("completed_runs")!=36 or audit.get("model_weights_compared") is not True:
        raise ValueError("complete strict main36 audit required")
    manifest=read(SOURCE_MANIFEST)
    selected=select_remaining_records(manifest)
    registry={r["id"]:r for r in read(registry_path)}
    equivalence=read(proof_path)
    proof_sha=file_sha256(proof_path)
    for name,path in (("source_audit",audit_path),("source_registry",registry_path)):
        if equivalence[name]["sha256"]!=file_sha256(path):
            raise ValueError("best equivalence proof source audit/registry changed")
    if file_sha256(BASE/"model.safetensors")!=BASE_SHA:
        raise ValueError("pinned410M base changed")
    by_id={r["id"]:r for r in selected}
    rows=[]
    sources=[proof(p) for p in (SOURCE_MANIFEST,registry_path,audit_path,proof_path)]
    for r in selected:
        row=registry[r["run_id"]]
        for key,path in (("config_path",r["config_path"]),("summary_path",r["summary_path"])):
            if Path(row[key]).resolve()!=Path(path).resolve() or file_sha256(path)!=r["input_sha256"][path]:
                raise ValueError("manifest source config/summary no longer frozen")
            sources.append(proof(path))
        if (row.get("active") is not True or row.get("phase")!="formal" or row["group"]!=r["group"] or
                row["seed"]!=r["training_seed"] or row["ratio"]!=r["ratio"] or row["weight"]!=r["weight"]):
            raise ValueError("logical endpoint identity differs from active registry")
        summary=read(r["summary_path"])
        expected_path=summary["checkpoint_path"] if r["kind"]=="final" else summary["best_target_validation"]["checkpoint_path"]
        if Path(expected_path).resolve()!=Path(r["checkpoint"]).resolve():
            raise ValueError("source checkpoint path is not the frozen actual endpoint")
        before=signature(r["checkpoint"])
        saved=torch.load(r["checkpoint"],map_location="cpu",mmap=True,weights_only=False)
        exposure=validate_saved_source(r,saved,summary,row)
        del saved
        if before!=signature(r["checkpoint"]) or before["bytes"]!=r["checkpoint_bytes"]:
            raise ValueError("source checkpoint changed while reading metadata")
        alias=validate_alias(r,by_id[r["weight_source_record_id"]],equivalence,proof_sha) if r["alias_of"] else None
        rows.append({**r,"frozen_source_stat":before,"actual_exposure":exposure,"alias_evidence":alias,
                     "source_config_sha256":file_sha256(r["config_path"]),"source_summary_sha256":file_sha256(r["summary_path"])})
    return rows,sources


def build_verified_tar(tar_path,members,limit_bytes):
    """No overwrite; fsync and independently hash every member before commit."""
    tar_path=Path(tar_path)
    temporary=tar_path.with_name(tar_path.name+".partial")
    if tar_path.exists() or temporary.exists():
        raise FileExistsError("never replace existing complete or partial outer archive")
    expected={name:digest for path,name,digest in members}
    if len(expected)!=len(members) or sum(Path(p).stat().st_size for p,_,_ in members)+(10<<20)>limit_bytes:
        raise ValueError("duplicate member or conservative payload exceeds size ceiling")
    with temporary.open("xb") as handle,tarfile.open(fileobj=handle,mode="w") as package:
        for path,name,digest in members:
            if file_sha256(path)!=digest:
                raise ValueError("outer archive input changed")
            package.add(path,arcname=name,recursive=False)
        package.close();handle.flush();os.fsync(handle.fileno())
    seen=set()
    with tarfile.open(temporary,"r") as package:
        for member in package.getmembers():
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("unexpected/duplicate/non-file TAR member")
            digest=hashlib.sha256()
            with package.extractfile(member) as stream:
                for block in iter(lambda:stream.read(8<<20),b""):digest.update(block)
            if digest.hexdigest()!=expected[member.name]:
                raise ValueError("TAR member SHA mismatch")
            seen.add(member.name)
    if seen!=set(expected) or temporary.stat().st_size>limit_bytes:
        raise ValueError("TAR incomplete or exceeds size ceiling")
    result=dict(tar_sha256=file_sha256(temporary),tar_bytes=temporary.stat().st_size,verified_tar_members=len(seen),file_fsync_completed=True)
    # A non-replacing link also avoids a late destination-overwrite race.
    os.link(temporary,tar_path)
    parent_fd=os.open(tar_path.parent,os.O_DIRECTORY)
    try:os.fsync(parent_fd)
    finally:os.close(parent_fd)
    temporary.unlink()
    parent_fd=os.open(tar_path.parent,os.O_DIRECTORY)
    try:os.fsync(parent_fd)
    finally:os.close(parent_fd)
    result["parent_directory_fsync_completed"]=True
    return result


def assemble_package(rows,records,sources,plan,code_sha,spec,stage,frozen,started,cpu_started,original_archives):
    frozen();stage("preserve_package_logical_metadata_and_aliases",spec["name"])
    manifest_path=DELIVERY/(spec["name"]+"_manifest.json")
    tar_path=DELIVERY/(spec["name"]+"_lossless_v1.tar")
    limit_bytes=spec["limit_bytes"]
    if len(rows)!=spec["logical"] or len(records)!=spec["unique"]:
        raise ValueError("package logical/unique counts differ from frozen split")
    by_weight={r["delivery_endpoint_id"]:r for r in records};logical=[];members=[]
    for r in rows:
        checkpoint=Path(r["checkpoint"])
        if signature(checkpoint)!=r["frozen_source_stat"] or file_sha256(checkpoint)!=r["checkpoint_sha256"]:
            raise ValueError("original source changed before logical metadata publication")
        saved=torch.load(checkpoint,map_location="cpu",mmap=True,weights_only=False)
        validate_saved_source(r,saved,read(r["summary_path"]),{**read(r["config_path"]),
            "id":r["run_id"],"seed":r["training_seed"],"ratio":r["ratio"],"weight":r["weight"],"config_path":r["config_path"]})
        sidecar=ENDPOINTS/"metadata"/(r["delivery_endpoint_id"]+".source_metadata.pt")
        sidecar.parent.mkdir(parents=True,exist_ok=True)
        if sidecar.exists():raise FileExistsError("metadata sidecar already exists")
        original_metadata=checkpoint_metadata(saved)
        torch.save(original_metadata,sidecar)
        if not same_metadata(original_metadata,torch.load(sidecar,map_location="cpu",weights_only=False)):
            raise ValueError("nonweight metadata sidecar roundtrip mismatch")
        del saved
        metadata_member="metadata/"+sidecar.name
        members.append((sidecar,metadata_member,file_sha256(sidecar)))
        actual=by_weight[r["weight_endpoint_id"]]
        if r["alias_evidence"] and actual["model_fingerprint"]!=r["alias_evidence"]["model_fingerprint"]:
            raise ValueError("canonical encoded weights differ from D alias proof")
        logical.append(dict(delivery_endpoint_id=r["delivery_endpoint_id"],source_record_id=r["id"],run_id=r["run_id"],kind=r["kind"],
            group=r["group"],training_seed=r["training_seed"],ratio=r["ratio"],weight=r["weight"],
            weight_endpoint_id=r["weight_endpoint_id"],archive_member=actual["archive_member"],model_fingerprint=actual["model_fingerprint"],
            source_checkpoint_path=str(checkpoint),source_checkpoint_sha256=r["checkpoint_sha256"],source_checkpoint_stat=r["frozen_source_stat"],
            source_config_sha256=r["source_config_sha256"],source_summary_sha256=r["source_summary_sha256"],
            metadata_member=metadata_member,metadata_sha256=file_sha256(sidecar),
            all_nonweight_metadata_roundtrip_verified=True,
            metadata_scope="all original fields except model_state and optimizer_state; model weights separately verified; optimizer excluded",
            actual_exposure=r["actual_exposure"],alias_of=r["alias_of"],alias_evidence=r["alias_evidence"]))
    members += [(Path(r["archive_path"]),r["archive_member"],r["archive_sha256"]) for r in records]
    base=[]
    for name in ("model.safetensors","config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","README.md"):
        p=BASE/name;record=dict(source_path=str(p),archive_member="base/pythia-410m-final/"+name,bytes=p.stat().st_size,sha256=file_sha256(p),
            source_url=f"https://huggingface.co/EleutherAI/pythia-410m/resolve/{REVISION}/{name}")
        base.append(record);members.append((p,record["archive_member"],record["sha256"]))
    for p,name in [(ROOT/"src/replay_study/weight_archive.py","tools/src/replay_study/weight_archive.py"),
                   (ROOT/"src/replay_study/__init__.py","tools/src/replay_study/__init__.py"),
                   (ROOT/"scripts/archive_model_weights.py","tools/scripts/archive_model_weights.py"),
                   (Path(__file__),"tools/provenance/export_remaining_primary_endpoints.py"),
                   (SOURCE_MANIFEST,"provenance/primary_evaluation_v2_execution_matched_nll_manifest.json"),
                   (ROOT/"results/formal_run_audit.json","provenance/formal_run_audit.json"),
                   (ROOT/"results/formal_best_model_equivalence_proofs.json","provenance/formal_best_model_equivalence_proofs.json"),
                   (PLAN,"provenance/primary_remaining_endpoints_export_plan.json")]:
        members.append((p,name,file_sha256(p)))
    manifest=dict(format=spec["name"]+"_lossless_delivery_v1",status="all_unique_weights_roundtrip_verified",logical_endpoints=len(rows),
        unique_weight_archives=len(records),logical_runs=logical,weight_archives=records,aliases=[r for r in logical if r["alias_of"]],
        base_revision=REVISION,base_files=base,source_manifest_sha256=SOURCE_MANIFEST_SHA,sources=sources,frozen_code_sha256=code_sha,
        selection_rule=plan["selection_rule"],gpu_used=False,shared_storage_written=False,cpu_threads=2,workers=1,
        model_only_weight_format=True,optimizer_included=False,original_checkpoints_retained=True,original_primary24_final_tar_unchanged=True,
        archive_model_raw_bytes=sum(r["encode"]["model_raw_bytes"] for r in records),weight_archive_bytes=sum(r["archive_bytes"] for r in records),
        base_assets_bytes=sum(r["bytes"] for r in base),encode_process_cpu_seconds=sum(r["encode_process_cpu_seconds"] for r in records),
        encode_wall_seconds_sum=sum(r["encode"]["wall_seconds"] for r in records),
        decode_process_cpu_seconds=sum(r["decode_process_cpu_seconds"] for r in records),
        verification_process_cpu_seconds=sum(r["verification_process_cpu_seconds"] for r in records),
        wall_seconds_before_tar=time.monotonic()-started,process_cpu_seconds_before_tar=time.process_time()-cpu_started,
        members=[dict(name=name,bytes=Path(p).stat().st_size,sha256=digest) for p,name,digest in members])
    write(manifest_path,manifest)
    readme=DELIVERY/("RESTORE_"+spec["name"].upper()+".md")
    readme.write_text(f"# Lossless {spec['name']}\n\n{len(rows)} logical endpoints map to {len(records)} unique exact FP32 model-state archives. This package is split for storage only; all12 sensitivity finals and all36 frozen best endpoints remain selected. No score-dependent filtering. Earlier primary24-final archive is unchanged.\n\nPinned410M base/config/tokenizer and tested restore tools are included. Find archive_member in {manifest_path.name}, then run from the extracted directory:\n\n```bash\npython tools/scripts/archive_model_weights.py --threads 2 decode --archive weights/ENDPOINT.xor.zip --base base/pythia-410m-final/model.safetensors --output restored.pt\n```\n\nEvery decoded tensor was independently compared with original uint8 bytes, including NaNs and signed zero. The decoder verifies base/tensor/full-model hashes. No optimizer is included. metadata/*.source_metadata.pt preserves every original nonweight/nonoptimizer field, including exact best selection, counters, provenance and any final RNG/stream state. Alias decoding yields canonical B4 metadata; D endpoint metadata is separately retained in its own sidecar. Read trusted-study sidecars with torch.load(map_location='cpu',weights_only=False). These model-only weights are for evaluation, not a complete optimizer-resume checkpoint.\n\nactual_exposure separates checkpoint target/replay from fully paid training10M. Original checkpoints retained; historical scratch paths are provenance only. Scratch size ceiling does not establish a successful shared reservation. Durable publication requires a separate reviewed receipt.\n")
    members += [(manifest_path,manifest_path.name,file_sha256(manifest_path)),(readme,readme.name,file_sha256(readme))]
    for source in sources:
        if file_sha256(source["path"])!=source["sha256"]:raise ValueError("frozen source changed before tar_path")
    frozen();stage("build_fsync_and_independently_verify_outer_tar")
    receipt=build_verified_tar(tar_path,members,limit_bytes)
    receipt.update(status="passed",tar_path=str(tar_path),logical_endpoints=len(rows),verified_unique_weights=len(records),
        alias_count=sum(bool(r["alias_of"]) for r in logical),aliases=[dict(endpoint_id=r["delivery_endpoint_id"],alias_of=r["alias_of"]) for r in logical if r["alias_of"]],
        manifest_sha256=file_sha256(manifest_path),source_manifest_sha256=SOURCE_MANIFEST_SHA,
        all_original_weight_bytes_compared=True,all_original_checkpoints_retained=all(Path(r["checkpoint"]).is_file() and signature(r["checkpoint"])==r["frozen_source_stat"] for r in rows),
        wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu_started,
        encode_process_cpu_seconds=manifest["encode_process_cpu_seconds"],encode_wall_seconds_sum=manifest["encode_wall_seconds_sum"],
        compression_ratio_archive_over_model_raw=manifest["weight_archive_bytes"]/manifest["archive_model_raw_bytes"],
        size_ceiling_bytes=limit_bytes,headroom_to_package_ceiling_bytes=limit_bytes-receipt["tar_bytes"],shared_storage_written=False,gpu_used=False,cpu_threads=2,workers=1,
        original_primary24_final_tar_unchanged=True,awaiting_root_size_and_reservation_review=True)
    if not receipt["all_original_checkpoints_retained"]:raise ValueError("original source retention check failed")
    if any(signature(item["path"])!=item["stat_before"] for item in original_archives):
        raise ValueError("previous primary24 archive stat changed")
    receipt["original_primary24_archives_unchanged_stat_proof"]=original_archives
    write(tar_path.with_suffix(tar_path.suffix+".receipt.json"),receipt)
    return receipt


def run():
    torch.set_num_threads(2)
    started=time.monotonic();cpu_started=time.process_time()
    if STATE.exists() or any((DELIVERY/(spec["name"]+suffix)).exists() for spec in PACKAGES.values() for suffix in ("_lossless_v1.tar","_manifest.json")):
        raise FileExistsError("new remaining-primary namespace required; no earlier attempts overwritten")
    code=[Path(__file__),ROOT/"src/replay_study/weight_archive.py",ROOT/"scripts/archive_model_weights.py"]
    code_sha={str(p):file_sha256(p) for p in code}
    original_archives=[]
    for original in (DELIVERY/"primary_final_weights_lossless_v1.tar",Path("/data/202610/7kb/research_delivery/primary_final_weights_lossless_v1.tar")):
        if original.exists():
            original_archives.append(dict(path=str(original),stat_before=signature(original),
                previously_verified_sha256="2bb090fbe7df172b31102068715143ab702087ff49f77ef7dfd5db3637c928b1"))
    progress=dict(pid=os.getpid(),status="running",cpu_threads=2,workers=1,gpu_used=False,shared_storage_written=False,
        logical_endpoints=48,expected_unique_weights=45,completed_unique_weights=0,completed_packages=[],code_sha256=code_sha,
        source_manifest_sha256=SOURCE_MANIFEST_SHA,source_checkpoints_deleted=False,first_primary24_tar_modified=False)
    def stage(name,record=None):
        progress.update(stage=name,current_endpoint_id=record,updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                        elapsed_wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu_started)
        write(STATE,progress)
        print(json.dumps({k:progress[k] for k in ("stage","current_endpoint_id","completed_unique_weights","elapsed_wall_seconds","updated_at_utc")}),flush=True)
    def frozen():
        if any(file_sha256(p)!=digest for p,digest in code_sha.items()):
            raise ValueError("export/codec code changed during sequential frozen export")
    try:
        stage("preflight_frozen_endpoint_identity_and_actual_exposure")
        rows,sources=preflight()
        for source in sources:
            if file_sha256(source["path"])!=source["sha256"]:raise ValueError("preflight frozen source changed")
        plan=dict(prepared_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status="frozen_export_plan",sources=sources,
            logical_endpoints=[{k:r[k] for k in ("id","run_id","kind","group","training_seed","ratio","weight","checkpoint","checkpoint_sha256",
                "delivery_endpoint_id","weight_endpoint_id","actual_exposure","alias_of","alias_evidence","frozen_source_stat")} for r in rows],
            code_sha256=code_sha,cpu_threads=2,workers=1,gpu_used=False,shared_storage_written=False,
            selection_rule="all36 frozen best + all12 B4/M b.01/.1 final; no score-dependent filtering",
            original_primary24_final_tar_unchanged=True,package_policy="12 sensitivity finals packaged first;36 best33unique next; storage split only; no successful50GiB reservation claimed")
        if PLAN.exists():raise FileExistsError("never overwrite export plan")
        write(PLAN,plan)
        unique=[r for r in rows if not r["alias_of"]]
        records=[]
        for row in unique:
            frozen();name=row["delivery_endpoint_id"];checkpoint=Path(row["checkpoint"])
            if signature(checkpoint)!=row["frozen_source_stat"]:raise ValueError("source checkpoint stat changed")
            archive=ENDPOINTS/"weights"/(name+".xor.zip")
            expected=(row.get("source_model_equivalence") or {}).get("fingerprint",{}).get("sha256_model_state")
            stage("encode_integer_xor_byteplanes",name)
            cpu=time.process_time()
            encoded=encode_weights(checkpoint,BASE/"model.safetensors",archive,threads=2,expected_model_sha256=expected)
            encode_cpu=time.process_time()-cpu
            if encoded["source_checkpoint_sha256"]!=row["checkpoint_sha256"] or encoded["base_sha256"]!=BASE_SHA:
                raise ValueError("encoded source/base differs from frozen manifest")
            restored=ENDPOINTS/"verification"/(name+".restored.pt")
            stage("decode_and_hash_all_tensors",name);cpu=time.process_time()
            decoded=decode_weights(archive,BASE/"model.safetensors",restored,threads=2)
            decode_cpu=time.process_time()-cpu
            frozen();stage("compare_all_original_uint8_bytes",name);cpu=time.process_time()
            verified=verify_roundtrip(checkpoint,restored,archive,threads=2)
            verify_cpu=time.process_time()-cpu
            if not verified["all_tensor_bytes_equal"] or signature(checkpoint)!=row["frozen_source_stat"]:
                raise ValueError("original byte equality/stability check failed")
            record=dict(delivery_endpoint_id=name,source_record_id=row["id"],kind=row["kind"],
                archive_path=str(archive),archive_member="weights/"+archive.name,archive_bytes=archive.stat().st_size,
                archive_sha256=encoded["archive_sha256"],model_fingerprint=encoded["model_fingerprint"],
                encode=encoded,decode=decoded,bytewise_verification=verified,
                encode_process_cpu_seconds=encode_cpu,decode_process_cpu_seconds=decode_cpu,verification_process_cpu_seconds=verify_cpu,
                actual_exposure=row["actual_exposure"],source_checkpoint_retained=True)
            write(archive.with_name(archive.name+".delivery_record.json"),record)
            records.append(record)
            restored.unlink()  # Only this freshly decoded verification duplicate; original source remains.
            progress.update(completed_unique_weights=len(records),verified_archive_bytes=sum(r["archive_bytes"] for r in records),
                            encode_process_cpu_seconds=sum(r["encode_process_cpu_seconds"] for r in records))
            stage("unique_endpoint_verified",name)
            if len(records)==12:
                budget_receipt=assemble_package([r for r in rows if r["kind"]=="final"],
                    records,sources,plan,code_sha,PACKAGES["final"],stage,frozen,started,cpu_started,original_archives)
                progress["completed_packages"].append(budget_receipt)
                stage("budget_final_package_verified_ready_for_root_review")
        frozen()
        best_receipt=assemble_package([r for r in rows if r["kind"]=="best"],
            [r for r in records if r["kind"]=="best"],sources,plan,code_sha,PACKAGES["best"],stage,frozen,started,cpu_started,original_archives)
        progress["completed_packages"].append(best_receipt)
        progress.update(status="passed",all45_original_weight_roundtrips_passed=True)
        stage("complete_two_scratch_packages_ready_for_root_review")
    except BaseException as error:
        progress.update(status="failed",error=repr(error));stage("failed_requires_review")
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    if args.preflight_only:
        rows,sources=preflight()
        print(json.dumps(dict(status="passed",logical_endpoints=len(rows),unique_weights=len({r['weight_endpoint_id'] for r in rows}),
            aliases=sum(bool(r["alias_of"]) for r in rows),source_manifest_sha256=SOURCE_MANIFEST_SHA,source_files=len(sources),gpu_used=False)))
    else:
        run()
