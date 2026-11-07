"""Lossless FP32 model-only archives relative to a SHA-bound safetensors base.

Only integer XOR and byte permutations act on trained weight bits. Optimizer
state is intentionally outside this model-only delivery format.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from safetensors import safe_open


FORMAT = "replay_study_fp32_xor_byteplanes_zip_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_cpu(threads):
    if not 1 <= threads <= 4:
        raise ValueError("CPU threads must be between 1 and 4")
    if sys.byteorder != "little":
        raise ValueError("v1 archives require a little-endian CPU")
    torch.set_num_threads(threads)


def _raw(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()


def _fingerprint_header(name, tensor):
    # Exactly the convention used by scripts/audit_formal_runs.py.
    return json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()


def model_fingerprint(state):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(state.items()):
        digest.update(_fingerprint_header(name, tensor))
        digest.update(memoryview(_raw(tensor)))
        count += tensor.numel()
    return {"sha256_model_state": digest.hexdigest(), "state_elements": count}


def _plane_shuffle(words):
    return words.view(np.uint8).reshape(-1, 4).T.copy().tobytes()


def _plane_unshuffle(payload, elements):
    if len(payload) != elements * 4:
        raise ValueError("decompressed plane length differs from declared FP32 chunk")
    return np.frombuffer(payload, dtype=np.uint8).reshape(4, elements).T.copy().reshape(-1).view("<u4")


def _write_report(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def encode_weights(checkpoint_path, base_path, archive_path, *, threads=2, chunk_bytes=4 << 20,
                   compression_level=6, expected_model_sha256=None):
    configure_cpu(threads)
    started = time.monotonic()
    checkpoint_path, base_path, archive_path = map(Path, (checkpoint_path, base_path, archive_path))
    if archive_path.exists():
        raise FileExistsError("use a new archive path; original artifacts are never replaced")
    if chunk_bytes < 4 or chunk_bytes % 4 or not 0 <= compression_level <= 9:
        raise ValueError("chunk bytes must be a positive multiple of four; compression level 0..9")
    saved = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    state = saved["model_state"]
    if not state or any(not isinstance(t, torch.Tensor) or t.dtype != torch.float32 for t in state.values()):
        raise ValueError("v1 model-only archive requires nonempty FP32 model_state")
    metadata = {"format": FORMAT, "encoded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_checkpoint": {"path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path),
                                      "bytes": checkpoint_path.stat().st_size},
                "base": {"path": str(base_path), "sha256": file_sha256(base_path),
                         "bytes": base_path.stat().st_size, "cast": "safetensors CPU tensor.to(torch.float32)",
                         "model_revision": saved.get("config", {}).get("model_revision")},
                "torch_version": torch.__version__, "numpy_version": np.__version__,
                "cpu_threads": threads, "chunk_bytes": chunk_bytes, "zlib_level": compression_level,
                "byte_order": "little", "shuffle": "four contiguous byte planes per FP32-XOR chunk, low byte first",
                "checkpoint_metadata": {k: saved[k] for k in ("format_version", "config", "counters", "provenance",
                    "validation_trajectory", "best_target_validation") if k in saved}, "tensors": []}
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(archive_path.name + ".partial")
    if temporary.exists():
        raise FileExistsError("unfinished archive path already exists")
    fingerprint = hashlib.sha256()
    raw_bytes = 0
    elements = 0
    try:
        with safe_open(str(base_path), framework="pt", device="cpu") as base, zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compression_level, allowZip64=True) as archive:
            base_keys = set(base.keys())
            for number, (name, tensor) in enumerate(sorted(state.items())):
                if name not in base_keys:
                    raise ValueError(f"base has no trained model tensor: {name}")
                reference = base.get_tensor(name).to(dtype=torch.float32).contiguous()
                if reference.shape != tensor.shape:
                    raise ValueError(f"base/trained shape differs: {name}")
                source_raw, base_raw = _raw(tensor), _raw(reference)
                source_words, base_words = source_raw.view("<u4"), base_raw.view("<u4")
                tensor_hash = hashlib.sha256()
                base_hash = hashlib.sha256()
                fingerprint.update(_fingerprint_header(name, tensor))
                record = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                          "elements": tensor.numel(), "raw_bytes": source_raw.nbytes, "chunks": []}
                chunk_elements = chunk_bytes // 4
                for index, offset in enumerate(range(0, tensor.numel(), chunk_elements)):
                    end = min(offset + chunk_elements, tensor.numel())
                    raw = source_raw[offset * 4:end * 4]
                    baseline_raw = base_raw[offset * 4:end * 4]
                    tensor_hash.update(memoryview(raw))
                    base_hash.update(memoryview(baseline_raw))
                    fingerprint.update(memoryview(raw))
                    difference = np.bitwise_xor(source_words[offset:end], base_words[offset:end])
                    member = f"tensors/{number:05d}/{index:05d}.planes"
                    archive.writestr(member, _plane_shuffle(difference))
                    record["chunks"].append({"member": member, "offset_elements": offset,
                                              "elements": end - offset})
                record.update(raw_sha256=tensor_hash.hexdigest(), base_fp32_raw_sha256=base_hash.hexdigest())
                metadata["tensors"].append(record)
                raw_bytes += source_raw.nbytes
                elements += tensor.numel()
            metadata["model_fingerprint"] = {"sha256_model_state": fingerprint.hexdigest(), "state_elements": elements}
            if expected_model_sha256 is not None and fingerprint.hexdigest() != expected_model_sha256:
                raise ValueError("trained model fingerprint differs from the audited model")
            metadata["model_raw_bytes"] = raw_bytes
            metadata["encode_seconds_before_zip_close"] = time.monotonic() - started
            archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode())
        os.replace(temporary, archive_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    report = {"status": "encoded", "archive": str(archive_path), "archive_sha256": file_sha256(archive_path),
              "archive_bytes": archive_path.stat().st_size, "model_raw_bytes": raw_bytes,
              "compression_ratio_archive_over_model_raw": archive_path.stat().st_size / raw_bytes,
              "model_fingerprint": metadata["model_fingerprint"], "base_sha256": metadata["base"]["sha256"],
              "source_checkpoint_sha256": metadata["source_checkpoint"]["sha256"],
              "source_checkpoint_bytes_including_optimizer": metadata["source_checkpoint"]["bytes"],
              "wall_seconds": time.monotonic() - started, "cpu_threads": threads,
              "model_only": True, "lossless_weight_bits": True, "gpu_used": False}
    _write_report(archive_path.with_suffix(archive_path.suffix + ".encode.json"), report)
    return report


def decode_weights(archive_path, base_path, output_path, *, threads=2):
    configure_cpu(threads)
    started = time.monotonic()
    archive_path, base_path, output_path = map(Path, (archive_path, base_path, output_path))
    if output_path.exists():
        raise FileExistsError("use a new recovered model-only path")
    state = {}
    with zipfile.ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        if metadata["format"] != FORMAT or metadata.get("byte_order") != "little":
            raise ValueError("unsupported weight archive format")
        if file_sha256(base_path) != metadata["base"]["sha256"]:
            raise ValueError("base SHA256 differs from the archived fixed base")
        if len({r["name"] for r in metadata["tensors"]}) != len(metadata["tensors"]):
            raise ValueError("duplicate archived tensor names")
        with safe_open(str(base_path), framework="pt", device="cpu") as base:
            for record in metadata["tensors"]:
                name = record["name"]
                reference = base.get_tensor(name).to(dtype=torch.float32).contiguous()
                if record["dtype"] != "torch.float32" or list(reference.shape) != record["shape"] or reference.numel() != record["elements"]:
                    raise ValueError("base shape or archived dtype differs")
                baseline_raw = _raw(reference)
                if hashlib.sha256(memoryview(baseline_raw)).hexdigest() != record["base_fp32_raw_sha256"]:
                    raise ValueError("base FP32 conversion bits differ from encoding")
                baseline_words = baseline_raw.view("<u4")
                recovered_words = np.empty(record["elements"], dtype="<u4")
                next_offset = 0
                for chunk in record["chunks"]:
                    offset, count = chunk["offset_elements"], chunk["elements"]
                    if offset != next_offset or count <= 0 or offset + count > record["elements"]:
                        raise ValueError("archive tensor chunks have a gap, overlap or invalid extent")
                    difference = _plane_unshuffle(archive.read(chunk["member"]), count)
                    recovered_words[offset:offset + count] = np.bitwise_xor(difference, baseline_words[offset:offset + count])
                    next_offset += count
                if next_offset != record["elements"]:
                    raise ValueError("archive omits model tensor elements")
                recovered_raw = recovered_words.view(np.uint8)
                if recovered_raw.nbytes != record["raw_bytes"] or hashlib.sha256(memoryview(recovered_raw)).hexdigest() != record["raw_sha256"]:
                    raise ValueError("recovered tensor raw SHA256 differs")
                state[name] = torch.from_numpy(recovered_words.view(np.float32)).reshape(record["shape"])
    fingerprint = model_fingerprint(state)
    if fingerprint != metadata["model_fingerprint"]:
        raise ValueError("recovered full model fingerprint differs")
    saved = {**metadata["checkpoint_metadata"], "format_version": 1,
             "checkpoint_kind": "model_only_final_lossless_xor_restored", "model_state": state,
             "lossless_delivery_provenance": {"format": FORMAT, "archive_sha256": file_sha256(archive_path),
                 "base_sha256": metadata["base"]["sha256"], "source_checkpoint_sha256": metadata["source_checkpoint"]["sha256"],
                 "model_fingerprint": fingerprint}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists():
        raise FileExistsError("unfinished restored output already exists")
    try:
        torch.save(saved, temporary)
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    report = {"status": "decoded_and_hash_verified", "archive": str(archive_path), "restored": str(output_path),
              "model_fingerprint": fingerprint, "tensor_raw_hashes_verified": len(state),
              "restored_file_sha256": file_sha256(output_path), "restored_bytes": output_path.stat().st_size,
              "wall_seconds": time.monotonic() - started, "cpu_threads": threads, "gpu_used": False,
              "model_only_evaluation_format": True}
    _write_report(output_path.with_suffix(output_path.suffix + ".decode.json"), report)
    return report


def verify_roundtrip(checkpoint_path, restored_path, archive_path, *, threads=2):
    configure_cpu(threads)
    started = time.monotonic()
    with zipfile.ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    if file_sha256(checkpoint_path) != metadata["source_checkpoint"]["sha256"]:
        raise ValueError("source checkpoint SHA differs from encoded source")
    source = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    restored = torch.load(restored_path, map_location="cpu", mmap=True, weights_only=False)
    a, b = source["model_state"], restored["model_state"]
    if set(a) != set(b):
        raise ValueError("restored model tensor names differ")
    for name in sorted(a):
        if a[name].dtype != b[name].dtype or a[name].shape != b[name].shape or not np.array_equal(_raw(a[name]), _raw(b[name])):
            raise ValueError(f"weight bytes do not match source: {name}")
    fingerprint = model_fingerprint(b)
    if fingerprint != metadata["model_fingerprint"]:
        raise ValueError("roundtrip model fingerprint differs from archive")
    for field, value in metadata["checkpoint_metadata"].items():
        if field != "format_version" and restored.get(field) != value:
            raise ValueError("restored model-only metadata differs")
    return {"status": "passed", "tensor_count": len(a), "all_tensor_bytes_equal": True,
            "source_checkpoint_sha256": metadata["source_checkpoint"]["sha256"],
            "archive_sha256": file_sha256(archive_path), "model_fingerprint": fingerprint,
            "wall_seconds": time.monotonic() - started, "cpu_threads": threads, "gpu_used": False}
