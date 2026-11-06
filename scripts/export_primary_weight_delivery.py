#!/usr/bin/env python3
"""Sequential CPU export of 24 logical finals / 21 exact weight archives to scratch."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_study.analysis import _check_summary, _protocol_status
from replay_study.weight_archive import (decode_weights, encode_weights, file_sha256,
                                        verify_roundtrip)

SCRATCH = Path("/tmp/grad-research-20261106")
DELIVERY = SCRATCH / "delivery"
WEIGHTS = DELIVERY / "weights"
STATE = ROOT / "results/primary_weight_export_state.json"
MANIFEST = DELIVERY / "primary_final_weights_manifest.json"
BASE = SCRATCH / "models/pythia-410m-final"
BASE_SHA = "3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b"
REVISION = "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f"
RESERVED_BYTES = 26 * 2**30
LEGACY_ID = "formal_deterministic_B4_s11_b0p05_w0p05"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run():
    torch.set_num_threads(2)
    started = time.monotonic()
    registry_path = ROOT / "configs/formal_deterministic_registry.json"
    audit_path = ROOT / "results/formal_run_audit.json"
    _protocol_status(ROOT / "docs/protocol_status.json", registry_path, "active")
    audit = read(audit_path)
    if audit["status"] != "passed" or audit["completed_runs"] != 36 or not audit["model_weights_compared"]:
        raise ValueError("complete strict main36 audit with weight comparisons is required")
    primary = [r for r in read(registry_path) if r["group"] == "B0" or r["ratio"] == .05]
    matrix = {(g, s) for g in ("B0", "B1", "B2", "B3", "B4", "D", "M", "C") for s in (11, 22, 33)}
    if len(primary) != 24 or {(r["group"], r["seed"]) for r in primary} != matrix:
        raise ValueError("exact main24 final-condition matrix required")
    if any(r.get("phase") != "formal" or r.get("active") is not True for r in primary):
        raise ValueError("only active strict formal finals may be delivered")
    if file_sha256(BASE / "model.safetensors") != BASE_SHA:
        raise ValueError("fixed primary base differs")
    nulls = {r["seed"]: r for r in audit["D_B4_null_controls"]}
    for proof in nulls.values():
        if not proof["bitwise_model_state_equal"] or proof["B4_model_fingerprint"] != proof["D_model_fingerprint"]:
            raise ValueError("D alias requires full model-state equivalence evidence")
    source_paths = [Path(__file__), ROOT / "src/replay_study/weight_archive.py", ROOT / "scripts/archive_model_weights.py"]
    code_sha = {str(p): file_sha256(p) for p in source_paths}
    def frozen_code():
        if any(file_sha256(p) != expected for p, expected in code_sha.items()):
            raise ValueError("delivery code changed during the frozen sequential export")
    progress = {"pid": os.getpid(), "status": "running", "cpu_threads": 2, "workers": 1,
                "gpu_used": False, "shared_storage_written": False, "completed_unique_weights": 0,
                "expected_unique_weights": 21, "logical_conditions": 24, "code_sha256": code_sha}
    def stage(name, run_id=None):
        progress.update(stage=name, current_run_id=run_id, updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write(STATE, progress)
        print(json.dumps({k: progress[k] for k in ("stage", "current_run_id", "completed_unique_weights", "updated_at_utc")}), flush=True)
    summaries = {}
    for row in primary:
        summary = read(row["summary_path"])
        _check_summary(summary, row)
        cfg = summary["config"]
        target = summary["counters"]["target_prediction_tokens"]
        replay = summary["counters"]["replay_prediction_tokens"]
        if summary["status"] != "completed" or target != 10_000_000 or replay != (0 if row["group"] == "B0" else 500_000) or cfg["model_revision"] != REVISION:
            raise ValueError("primary source is not its complete pinned final")
        summaries[row["id"]] = summary
    unique = [r for r in primary if r["group"] != "D"]
    unique.sort(key=lambda r: (r["id"] != LEGACY_ID, r["seed"], r["group"]))
    records = []
    try:
        for row in unique:
            frozen_code()
            name = row["id"]
            summary = summaries[name]
            checkpoint = Path(summary["checkpoint_path"])
            archive_path = WEIGHTS / (name + ".xor.zip")
            expected = nulls[row["seed"]]["B4_model_fingerprint"]["sha256_model_state"] if row["group"] == "B4" else None
            stage("encode_or_verify_existing_archive", name)
            reused = archive_path.exists()
            if not reused:
                encode = encode_weights(checkpoint, BASE / "model.safetensors", archive_path,
                                        threads=2, expected_model_sha256=expected)
            else:
                encode = read(archive_path.with_suffix(archive_path.suffix + ".encode.json"))
                with zipfile.ZipFile(archive_path) as z:
                    metadata = json.loads(z.read("metadata.json"))
                if metadata["source_checkpoint"]["sha256"] != file_sha256(checkpoint) or metadata["base"]["sha256"] != BASE_SHA:
                    raise ValueError("existing archive does not match frozen original source/base")
                if expected and metadata["model_fingerprint"]["sha256_model_state"] != expected:
                    raise ValueError("existing archive differs from audited model")
                if file_sha256(archive_path) != encode["archive_sha256"]:
                    raise ValueError("existing archive bytes changed")
            restored = WEIGHTS / "verification" / (name + ".restored.pt")
            stage("decode", name)
            decode = decode_weights(archive_path, BASE / "model.safetensors", restored, threads=2)
            frozen_code()
            stage("compare_all_original_uint8_weight_bytes", name)
            verification = verify_roundtrip(checkpoint, restored, archive_path, threads=2)
            if not verification["all_tensor_bytes_equal"]:
                raise ValueError("real original/restored model byte comparison failed")
            with zipfile.ZipFile(archive_path) as z:
                metadata = json.loads(z.read("metadata.json"))
            source = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
            if "checkpoint_kind" not in metadata["checkpoint_metadata"] and "checkpoint_kind" in source:
                raise ValueError("existing legacy archive lost an original checkpoint kind")
            if "selection" not in metadata["checkpoint_metadata"] and "selection" in source:
                raise ValueError("existing legacy archive lost an original selection record")
            encoder_sha = metadata.get("encoder_source_sha256")
            legacy = encoder_sha is None
            if legacy:
                legacy_source = ROOT / "artifacts/provenance/weight_archive_single_example_v1.py"
                encoder_sha = read(ROOT / "artifacts/assets/lossless_weight_single_example.json")["source_files_sha256"]["src/replay_study/weight_archive.py"]
                if file_sha256(legacy_source) != encoder_sha:
                    raise ValueError("legacy encoder source snapshot not available")
            record = {"actual_run_id": name, "archive_path": str(archive_path), "archive_member": "weights/" + archive_path.name,
                      "encode": encode, "decode": decode, "bytewise_verification": verification,
                      "archive_sha256": file_sha256(archive_path), "archive_bytes": archive_path.stat().st_size,
                      "reused_existing_archive": reused, "encoder_source_sha256": encoder_sha,
                      "decoder_source_sha256": code_sha[str(ROOT / "src/replay_study/weight_archive.py")],
                      "legacy_optional_metadata": legacy,
                      "legacy_compatibility_note": "old v1 omits optional origin-kind fields; original latest had no checkpoint_kind or selection, source SHA and all bits reverified" if legacy else None}
            write(WEIGHTS / (name + ".delivery_record.json"), record)
            records.append(record)
            # This temporary is our own freshly decoded verification copy.
            restored.unlink()
            progress["completed_unique_weights"] = len(records)
            progress["verified_archive_bytes"] = sum(r["archive_bytes"] for r in records)
            stage("weight_verified", name)
        frozen_code()
        stage("assemble_logical_aliases_and_base_manifest")
        by_group_seed = {(r["group"], r["seed"]): r for r in primary}
        actual_records = {r["actual_run_id"]: r for r in records}
        logical = []
        for row in primary:
            actual = by_group_seed["B4", row["seed"]] if row["group"] == "D" else row
            record = actual_records[actual["id"]]
            checkpoint_path = summaries[row["id"]]["checkpoint_path"]
            source = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
            checkpoint_sha = file_sha256(checkpoint_path) if row["group"] == "D" else record["encode"]["source_checkpoint_sha256"]
            item = {"condition_run_id": row["id"], "weights_actual_run_id": actual["id"],
                    "archive_member": record["archive_member"], "model_fingerprint": record["encode"]["model_fingerprint"],
                    "source_checkpoint_sha256": checkpoint_sha, "source_checkpoint_path": checkpoint_path,
                    "source_summary_sha256": file_sha256(row["summary_path"]), "config_sha256": file_sha256(row["config_path"]),
                    "original_checkpoint_metadata": {k: source[k] for k in ("format_version", "checkpoint_kind", "selection", "config", "counters", "provenance") if k in source},
                    "summary_counters": summaries[row["id"]]["counters"],
                    "alias_of": actual["id"] if row["group"] == "D" else None}
            if row["group"] == "D":
                proof = nulls[row["seed"]]
                item["alias_evidence"] = {"audit_sha256": file_sha256(audit_path), "seed": row["seed"],
                    "B4_model_fingerprint": proof["B4_model_fingerprint"], "D_model_fingerprint": proof["D_model_fingerprint"],
                    "bitwise_model_state_equal": True}
            logical.append(item)
        base_records = []
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md"):
            path = BASE / name
            base_records.append({"source_path": str(path), "archive_member": "base/pythia-410m-final/" + name,
                "bytes": path.stat().st_size, "sha256": file_sha256(path),
                "source_url": f"https://huggingface.co/EleutherAI/pythia-410m/resolve/{REVISION}/{name}"})
        receipt_files = [ROOT / "src/replay_study/weight_archive.py", ROOT / "src/replay_study/__init__.py",
                         ROOT / "scripts/archive_model_weights.py", ROOT / "artifacts/provenance/weight_archive_single_example_v1.py"]
        members = [(Path(r["archive_path"]), r["archive_member"], r["archive_sha256"]) for r in records]
        members += [(Path(r["source_path"]), r["archive_member"], r["sha256"]) for r in base_records]
        members += [(receipt_files[0], "tools/src/replay_study/weight_archive.py", file_sha256(receipt_files[0])),
                    (receipt_files[1], "tools/src/replay_study/__init__.py", file_sha256(receipt_files[1])),
                    (receipt_files[2], "tools/scripts/archive_model_weights.py", file_sha256(receipt_files[2])),
                    (receipt_files[3], "tools/provenance/weight_archive_single_example_v1.py", file_sha256(receipt_files[3]))]
        manifest = {"format": "primary_final_lossless_delivery_v1", "status": "all_weights_verified",
                    "prepared_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "cpu_threads": 2, "workers": 1,
                    "gpu_used": False, "shared_storage_written": False, "logical_primary_conditions": 24,
                    "unique_weight_archives": 21, "logical_runs": logical, "weight_archives": records,
                    "base_revision": REVISION, "base_files": base_records, "frozen_code_sha256": code_sha,
                    "formal_audit_sha256": file_sha256(audit_path), "registry_sha256": file_sha256(registry_path),
                    "excluded_weights": "12 b-sensitivity finals; their metrics/configs/logs remain in the ordinary research package",
                    "weight_archive_bytes": sum(r["archive_bytes"] for r in records),
                    "base_assets_bytes": sum(r["bytes"] for r in base_records),
                    "total_export_wall_seconds_before_tar": time.monotonic() - started,
                    "members": [{"name": name, "bytes": path.stat().st_size, "sha256": digest} for path, name, digest in members]}
        write(MANIFEST, manifest)
        readme = DELIVERY / "RESTORE_PRIMARY_WEIGHTS.md"
        readme.write_text("# Exact primary final model weights\n\n24 logical conditions map to 21 model-state archives; three D conditions alias their same-seed B4 by complete audit fingerprints. Read primary_final_weights_manifest.json for each original condition metadata. All trained weight bytes were independently compared to original checkpoints.\n\nThe pinned base and tokenizer/config are included. From this extracted folder, restore with Python, torch, numpy and safetensors installed:\n\n```bash\npython tools/scripts/archive_model_weights.py --threads 2 decode --archive weights/RUN_ID.xor.zip --base base/pythia-410m-final/model.safetensors --output restored_RUN_ID.pt\n```\n\nThe output is model-only evaluation format, with no optimizer. It is bit-exact, including special FP32 values. Decoder checks base, each tensor and full model hashes. Legacy single-example archive uses compatible v1 payload; its original encoder source SHA is separately bound in the manifest. Original absolute paths are provenance, not required restore locations. This package contains final checkpoints only; validation-selected best and b-sensitivity weights are not included. All metrics/configuration/log evidence belongs to the separate research package.\n")
        members += [(MANIFEST, MANIFEST.name, file_sha256(MANIFEST)), (readme, readme.name, file_sha256(readme))]
        if sum(p.stat().st_size for p, _, _ in members) + (10 << 20) > RESERVED_BYTES:
            raise ValueError("verified payload plus conservative TAR overhead exceeds the reserved 26GiB")
        tar_path = DELIVERY / "primary_final_weights_lossless_v1.tar"
        if tar_path.exists():
            raise FileExistsError("refusing to replace an existing complete delivery TAR")
        temporary = tar_path.with_name(tar_path.name + ".partial")
        if temporary.exists():
            raise FileExistsError("unfinished outer TAR already exists")
        stage("build_uncompressed_outer_tar")
        with temporary.open("wb") as output, tarfile.open(fileobj=output, mode="w") as package:
            for path, name, digest in members:
                if file_sha256(path) != digest:
                    raise ValueError("package source changed after verification")
                package.add(path, arcname=name, recursive=False)
            package.close()
            output.flush(); os.fsync(output.fileno())
        if temporary.stat().st_size > RESERVED_BYTES:
            raise ValueError("actual TAR exceeds reserved space")
        stage("verify_every_tar_member_and_full_tar_sha")
        expected = {name: digest for _, name, digest in members}
        seen = set()
        with tarfile.open(temporary, mode="r") as package:
            for member in package.getmembers():
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError("unexpected or duplicate outer TAR member")
                digest = hashlib.sha256()
                with package.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(8 << 20), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected[member.name]:
                    raise ValueError("outer TAR member content SHA differs")
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError("outer TAR omitted a verified member")
        digest = file_sha256(temporary)
        os.replace(temporary, tar_path)
        receipt = {"status": "passed", "tar_path": str(tar_path), "tar_sha256": digest,
                   "tar_bytes": tar_path.stat().st_size, "reserved_bytes": RESERVED_BYTES,
                   "headroom_bytes": RESERVED_BYTES - tar_path.stat().st_size, "logical_conditions": 24,
                   "verified_unique_weights": 21, "verified_tar_members": len(seen),
                   "all_original_weight_bytes_compared": True, "manifest_sha256": file_sha256(MANIFEST),
                   "wall_seconds": time.monotonic() - started, "shared_storage_written": False, "gpu_used": False}
        write(tar_path.with_suffix(tar_path.suffix + ".receipt.json"), receipt)
        progress.update(status="passed", tar_receipt=receipt)
        stage("complete_ready_for_root_publication")
    except BaseException as error:
        progress.update(status="failed", error=repr(error))
        stage("failed")
        raise


if __name__ == "__main__":
    run()
