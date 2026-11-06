#!/usr/bin/env python3
"""One-shot guide source calculation: small signed JSON plus archive stat only."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import resource
import shutil
import stat
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HISTORY = ROOT / "artifacts/provenance/delivery_capacity_guide/pre_C410_best_published_v5"
V5_SHA = "bb51825b0bc47ac20306a85228346ac0d8c35ab14f83a50b85a7f210331b28ef"
SOURCES = {}


def small(path, expected=None):
    path = (ROOT / path).resolve()
    if str(path) in SOURCES:
        d, proof = SOURCES[str(path)]
        if expected and proof["sha256"] != expected:
            raise ValueError("conflicting declared source SHA")
        return d, proof
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024 * 1024:
        raise ValueError("bounded small JSON sources only")
    b = path.read_bytes()
    after = path.stat()
    for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
        if getattr(before, key) != getattr(after, key):
            raise ValueError("source changed during read")
    sha = hashlib.sha256(b).hexdigest()
    if expected and sha != expected:
        raise ValueError(f"source SHA mismatch: {path}")
    retained = OUT / "source_snapshots" / f"{sha}.json"
    retained.parent.mkdir(exist_ok=True)
    if retained.exists():
        if retained.read_bytes() != b:
            raise ValueError("snapshot conflict")
    else:
        retained.write_bytes(b)
    proof = {"path": str(path), "sha256": sha, "bytes": len(b), "retained": str(retained)}
    result = (json.loads(b) if path.suffix == ".json" else None, proof)
    SOURCES[str(path)] = result
    return result


def current_stat(path):
    s = Path(path).lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("archive must be a non-symlink regular file")
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
            "allocated_bytes": s.st_blocks * 512, "uid": s.st_uid, "gid": s.st_gid,
            "mode": oct(stat.S_IMODE(s.st_mode)), "mtime_ns": s.st_mtime_ns,
            "ctime_ns": s.st_ctime_ns, "links": s.st_nlink}


def normalize(row, package):
    logical = row.get("condition_run_id", row.get("condition_id", row.get("run_id")))
    actual = row.get("actual_run_id", row.get("condition_run_id", row.get("run_id")))
    kind = row.get("kind", "final")
    if kind in ("validation_selected_lambda_final", "fixed_exposure_final"):
        kind = "final"
    if not logical or not actual or kind not in ("final", "best"):
        raise ValueError("ambiguous run or endpoint identity")
    fp = row["model_fingerprint"]
    if len(fp["sha256_model_state"]) != 64 or type(fp["state_elements"]) is not int:
        raise ValueError("invalid full-state fingerprint metadata")
    physical = row.get("weights_actual_run_id") or row.get("alias_of") or actual
    for suffix in ("__best", "__final", "__fixed_exposure_final"):
        physical = physical.removesuffix(suffix)
    return {"package": package, "logical_id": logical, "actual_training_run": actual,
            "physical_weight_source_training_run": physical, "normalized_kind": kind,
            "model_fingerprint": fp["sha256_model_state"], "state_elements": fp["state_elements"],
            "source_checkpoint_sha256": row["source_checkpoint_sha256"],
            "ZIP_member": row.get("archive_member"), "alias_of": row.get("alias_of"),
            "has_external_durable_reference": bool(row.get("durable_weight_reference"))}


def main():
    start = time.monotonic()
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    if (OUT / "metadata_calculation.json").exists():
        raise ValueError("refuse replacing completed v6 evidence")
    old, old_proof = small("docs/交付文件与恢复容量指南_来源核对_v5.json", V5_SHA)
    HISTORY.mkdir(exist_ok=True)
    retained = []
    for source, name in [(ROOT / "docs/交付文件与恢复容量指南.md", "guide_v5.md"),
                         (ROOT / "docs/交付文件与恢复容量指南_来源核对_v5.json", "source_v5.json"),
                         (ROOT / "README.md", "README_before_C410_best_navigation.md")]:
        if (HISTORY / name).exists():
            if (HISTORY / name).read_bytes() != source.read_bytes():
                raise ValueError("retained prior v5 bytes differ; refuse replacing history")
        else:
            shutil.copyfile(source, HISTORY / name)
        retained.append({"path": str(HISTORY / name), "sha256": hashlib.sha256((HISTORY / name).read_bytes()).hexdigest()})
    if retained[0]["sha256"] != old["guide"]["sha256"]:
        raise ValueError("current guide differs from ROOT-accepted v5")
    pub, pub_proof = small("docs/C410_best_weights_durable_publication.json", "139e54cfcbae7763a1b651abd4b54c2646dbc1af99dff26787cce0b28107091a")
    accepted, accepted_proof = small("artifacts/provenance/C410_best_publication_actual_v1/ROOT_acceptance.json", "ce7c7905d5f7a424eb3832bd83080f17b906b5b45fddd685170338ef3623e3a8")
    adjacent, adjacent_proof = small(pub["destination"] + ".receipt.json", "afcde1091d51b1320ddf8421f28425af22bd85247b196fca1bf72aac9e878f44")
    failure, failure_proof = small("artifacts/provenance/C410_best_publication_actual_v1/outer_timing_receipt_failure.json", "8c284e922503b2fd9e747e377db776aa73e146c9ff1f886901e775193626b13f")
    if pub["status"] != "passed" or pub["source_sha256"] != pub["destination_sha256"]:
        raise ValueError("publication not passed")
    for flag in ("original_inode_preserved", "permissions_and_ownership_preserved", "source_retained", "file_fsync_after_copy", "prefix_sha_verified_before_truncate", "file_fsync_after_truncate", "atomic_destination_commit_noreplace", "parent_directory_fsync"):
        if pub.get(flag) is not True:
            raise ValueError("missing completed durable-publication flag")
    if Path(pub["original_reservation"]).exists():
        raise ValueError("consumed C410best reserve name still present")
    if current_stat(pub["source"]) != {**pub["source_stable_stat"], "links": 1}:
        raise ValueError("source TAR stat changed")
    if accepted["publication"]["sha256"] != pub_proof["sha256"] or accepted["publisher_exact_whole_process_wall_and_CPU_unknown"] is not True:
        raise ValueError("acceptance/timing provenance drift")
    if failure["exact_publisher_spawn_to_exit_wall_seconds_not_persisted"] is not None:
        raise ValueError("must preserve unknown whole publisher timer")

    files = old["files"] + [{"filename": Path(pub["destination"]).name, "path": pub["destination"],
        "bytes": pub["bytes"], "GiB": pub["bytes"] / 2**30, "publication_receipt": pub_proof,
        "ROOT_acceptance": accepted_proof, "durable_adjacent_receipt": adjacent_proof,
        "historical_TAR_sha256": pub["destination_sha256"], "manifest": accepted["manifest"],
        "base_dependency": "native six-file base/config/tokenizer/README and tools included; no external weight package needed",
        "actual_independent_restore_from_this_durable_package_performed": False}]
    rows = []
    for file in files:
        receipt, receipt_proof = small(file["publication_receipt"]["path"], file["publication_receipt"]["sha256"])
        manifest, manifest_proof = small(file["manifest"]["path"], file["manifest"]["sha256"])
        s = current_stat(file["path"])
        for key, value in receipt["destination_stat"].items():
            if s[key] != value:
                raise ValueError(f"current published archive identity changed: {file['filename']} {key}")
        if s["uid"] != 11471 or s["mode"] != "0o600" or s["links"] != 1 or s["bytes"] != file["bytes"]:
            raise ValueError("unexpected owner/mode/link count/size")
        if receipt["destination_sha256"] != file["historical_TAR_sha256"]:
            raise ValueError("historical full TAR SHA differs")
        rs = [normalize(r, file["filename"]) for r in manifest["logical_runs"]]
        rows.extend(rs)
        file.update({"manifest": manifest_proof, "publication_receipt": receipt_proof,
            "current_stat_rechecked": s, "current_stat_matches_publication": True,
            "package_logical_record_count": len(rs), "normalized_actual_run_kind_count_in_package": len({(r['actual_training_run'], r['normalized_kind']) for r in rs}),
            "distinct_model_state_fingerprints_in_package": len({r['model_fingerprint'] for r in rs}),
            "new_weight_ZIP_payload_members": len({r['ZIP_member'] for r in rs if r['ZIP_member']}),
            "full_large_TAR_rehash_performed_now": False})
    endpoint_map = {}
    fp_elements = {}
    for r in rows:
        key = (r["actual_training_run"], r["normalized_kind"])
        if key in endpoint_map and endpoint_map[key] != r["model_fingerprint"]:
            raise ValueError("same actual endpoint has contradictory fingerprint")
        endpoint_map[key] = r["model_fingerprint"]
        fp = r["model_fingerprint"]
        if fp in fp_elements and fp_elements[fp] != r["state_elements"]:
            raise ValueError("same fingerprint has contradictory element count")
        fp_elements[fp] = r["state_elements"]
    snapshot = old["snapshot"]
    snapshot_pub, _ = small(snapshot["publication_receipt"]["path"], snapshot["publication_receipt"]["sha256"])
    small(snapshot["ROOT_acceptance"]["path"], snapshot["ROOT_acceptance"]["sha256"])
    latest, latest_proof = small("docs/durable_snapshot_latest.json", snapshot["latest_external_pointer"]["sha256"])
    snapshot_stat = current_stat(snapshot["path"])
    for key, value in snapshot_pub["destination_stat"].items():
        if snapshot_stat[key] != value:
            raise ValueError("1640 snapshot current stat changed")
    snapshot.update({"current_stat": snapshot_stat, "latest_external_pointer": latest_proof})
    b_root, b_root_proof = small("artifacts/provenance/B_native_restore_actual_parent_v1/ROOT_acceptance.json", "0194effe57624ce7ffac21396c57383cce5cbc067e93451738f755325eb25071")
    b_indie, b_indie_proof = small("artifacts/provenance/B_native_restore_independent_acceptance_v1/COMPLETE.json")
    small(b_indie["document"]["path"], b_indie["document"]["sha256"])
    b_qa = {"ROOT_acceptance": b_root_proof, "independent_COMPLETE": b_indie_proof,
        "full_process_wall_seconds": 36.18444097042084, "one_fixed_draw_only": True,
        "whole_model_fingerprint": b_root["whole_model_fingerprint"], "all_60_redecoded": False}
    for file in files:
        if file["filename"] == "B_all_final_weights_lossless_v1.tar":
            file["actual_independent_restore_from_this_durable_package_performed"] = True
            file["independent_restore_QA"] = b_qa
    weights_bytes = sum(f["bytes"] for f in files)
    totals = {"package_logical_record_count": len(rows), "normalized_actual_run_plus_kind_endpoints": len(endpoint_map),
        "actual_source_training_runs": len({r['actual_training_run'] for r in rows}),
        "physical_weight_source_training_runs": len({r['physical_weight_source_training_run'] for r in rows}),
        "globally_distinct_model_state_fingerprints": len(fp_elements),
        "weight_ZIP_payload_members": sum(f['new_weight_ZIP_payload_members'] for f in files),
        "raw_FP32_tensor_bytes": sum(fp_elements.values()) * 4,
        "weight_packages": len(files), "weight_archive_bytes": weights_bytes,
        "weight_archive_GiB": weights_bytes / 2**30,
        "project_snapshot_files_in_this_table": 1, "actual_distinct_archive_files": len(files) + 1,
        "total_bytes": weights_bytes + snapshot["bytes"], "total_GiB": (weights_bytes + snapshot['bytes']) / 2**30}
    previous = [r for r in rows if r["package"] != "C410_best_weights_lossless_v1.tar"]
    prior, _ = small("results/delivery_capacity_guide_v5/metadata_calculation.json", "722c5a6fe02b772c722772174e817fcfa213d3e2fbbe31913ba1db82fcea00e9")
    def key(r):
        fp = r['model_fingerprint']
        fp = fp['sha256_model_state'] if isinstance(fp, dict) else fp
        return (r['package'], r['logical_id'], r['actual_training_run'], r['normalized_kind'], fp)
    if sorted(map(key, previous)) != sorted(map(key, prior['normalized_rows'])):
        raise ValueError("old seven-package reparse changed identities")
    data = {"schema_version": 1, "status": "passed_small_metadata_stat_only_v6_calculation",
        "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "files": files, "snapshot": snapshot,
        "totals": totals, "normalized_rows": rows, "prior_seven_fresh_reparse_equal_v5": True,
        "count_algorithm": old["count_algorithm"], "capacity": {**old['capacity'],
            "all_distinct_fingerprint_FP32_tensor_bytes": totals['raw_FP32_tensor_bytes'],
            "all_distinct_fingerprint_FP32_tensor_GiB": totals['raw_FP32_tensor_bytes'] / 2**30},
        "B_actual_restore": b_qa, "C410_best_publication_timing_exception": failure_proof,
        "C410_best_publisher_internal_region_seconds": pub['wall_seconds'], "C410_best_publisher_external_whole_seconds": None,
        "prior_v5_source": old_proof, "history": retained, "small_source_proofs": [v[1] for _, v in sorted(SOURCES.items())],
        "large_payload_opened_or_rehashed": False, "scientific_scores_read": False,
        "timing": {"wall_seconds": time.monotonic() - start,
            "self_user_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime - cpu.ru_utime,
            "self_sys_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime - cpu.ru_stime,
            "scope": "This script main-entry small source copy/stat/eight-manifest normalization and preservation region; source writing and serialization below excluded, no children; manual authoring whole unknown",
            "small_source_bytes_read": sum(v[1]['bytes'] for v in SOURCES.values())}}
    path = OUT / "metadata_calculation.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"totals": totals, "metadata_calculation_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
