#!/usr/bin/env python3
"""Audit delivery paths/sizes and archive metadata, without parsing model scores."""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("research_packager", ROOT / "scripts/package_research.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def census(directory):
    count = size = 0
    links = []
    for base, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            path = Path(base) / name
            if path.is_symlink():
                links.append(dict(path=str(path), target=os.readlink(path)))
        for name in files:
            path = Path(base) / name
            if path.is_symlink():
                continue
            try:
                observed = path.stat()
            except FileNotFoundError:
                continue
            count += 1
            size += observed.st_size
    return dict(path=str(directory), files=count, bytes=size, symlinks=links)


def tar_header_bytes(name, size, mode=0o644, mtime=0):
    entry = tarfile.TarInfo("research/" + name)
    entry.size, entry.mode, entry.mtime = size, mode, mtime
    return len(entry.tobuf(format=tarfile.PAX_FORMAT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/tmp/grad-research-20261106/runs"))
    parser.add_argument("--snapshot", type=Path, default=Path("/data/202610/7kb/research_delivery/research_snapshot_20261106_0838.tar.gz"))
    parser.add_argument("--expected-snapshot-sha256", default="96ec2d1cf244af9b25b1dd8d993f7cf659b4e2dfd0f0e0ec4d5d26140882dfde")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/delivery_audit/current")
    args = parser.parse_args()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    actual_sha = sha(args.snapshot)  # Opaque compressed bytes only, not score parsing.
    if actual_sha != args.expected_snapshot_sha256:
        raise ValueError("durable snapshot SHA mismatch")
    with tarfile.open(args.snapshot) as archive:
        members = archive.getmembers()
        by_name = {member.name: member for member in members}
        if len(by_name) != len(members):
            raise ValueError("snapshot has duplicate members")
        manifest_raw = archive.extractfile("research/PACKAGE_MANIFEST.json").read()
        manifest = json.loads(manifest_raw)
        expected_names = {"research/" + row["path"] for row in manifest["files"]}
        if set(by_name) != expected_names | {"research/PACKAGE_MANIFEST.json"}:
            raise ValueError("snapshot manifest/header names disagree")
        for row in manifest["files"]:
            member = by_name["research/" + row["path"]]
            if not member.isfile() or member.size != row["bytes"]:
                raise ValueError("snapshot manifest/header kind or byte size disagrees")
    paths, symlinks, weights = packager.collect_paths(ROOT, args.runs_root, require_run_records=True, fail_on_symlinks=True)
    old = {row["path"]: row for row in manifest["files"]}
    inventory = []
    for path, name in paths:
        observed = path.stat()
        previous = old.get(name)
        state = "new_since_snapshot" if previous is None else (
            "same_size_and_mtime" if previous["bytes"] == observed.st_size and previous["source_mtime_ns"] == observed.st_mtime_ns
            else "size_or_mtime_changed")
        inventory.append(dict(path=name, source=str(path), bytes=observed.st_size, mtime_ns=observed.st_mtime_ns,
                              mode=observed.st_mode & 0o777, change=state))
    current = {row["path"]: row for row in inventory}
    missing = [dict(path=name, bytes=row["bytes"], change="not_selected_now") for name, row in old.items() if name not in current]
    components = []
    for component in sorted({row["path"].split("/")[0] for row in inventory + manifest["files"]}):
        current_rows = [row for row in inventory if row["path"].split("/")[0] == component]
        old_rows = [row for row in manifest["files"] if row["path"].split("/")[0] == component]
        components.append(dict(component=component, current_files=len(current_rows), current_bytes=sum(row["bytes"] for row in current_rows),
            snapshot_files=len(old_rows), snapshot_bytes=sum(row["bytes"] for row in old_rows),
            new_files=sum(row["change"] == "new_since_snapshot" for row in current_rows),
            changed_size_or_mtime=sum(row["change"] == "size_or_mtime_changed" for row in current_rows)))
    excluded = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        name = str(path.relative_to(ROOT))
        if name in current:
            continue
        reason = "cache_directory" if any(part in {"__pycache__", ".pytest_cache", ".venv"} for part in path.relative_to(ROOT).parts) else "runtime_pid_lock_or_tmp"
        excluded.append(dict(path=name, bytes=path.stat().st_size, reason=reason))
    unknown_runs = []
    for path in args.runs_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(args.runs_root)
        name = "run_records/" + str(relative)
        if name not in current and path.suffix not in {".pt", ".safetensors", ".bin"}:
            unknown_runs.append(dict(path=str(path), bytes=path.stat().st_size, reason="outside_run_record_suffix_policy_or_checkpoint_directory"))
    if "requirements-report.lock" not in current:
        raise ValueError("render dependency lock is missing from selected payload")
    # Reproduce the packager's metadata shape with 64-character hash placeholders.
    # No selected payload, including sealed outputs, is opened for this estimate.
    estimated_manifest = dict(created_at_utc=started, label="final_delivery_capacity_preflight" * 3, source_workspace=str(ROOT),
        consistency="Each file stable while read; not a simultaneous process snapshot.",
        files=[dict(path=row["path"], bytes=row["bytes"], sha256="0" * 64, source_mtime_ns=row["mtime_ns"]) for row in inventory],
        omitted_symlinks=symlinks, external_weights_not_included=weights)
    manifest_bytes = len((json.dumps(estimated_manifest, ensure_ascii=False, indent=2) + "\n").encode())
    tar_bytes = sum(tar_header_bytes(row["path"], row["bytes"], row["mode"], row["mtime_ns"] // 1_000_000_000)
                    + math.ceil(row["bytes"] / 512) * 512 for row in inventory)
    tar_bytes += tar_header_bytes("PACKAGE_MANIFEST.json", manifest_bytes) + math.ceil(manifest_bytes / 512) * 512 + 1024
    tar_bytes = math.ceil(tar_bytes / tarfile.RECORDSIZE) * tarfile.RECORDSIZE
    # Deliberately budget 1% above the full uncompressed tar, plus 32MiB for this
    # audit's outputs and concurrently added small metadata. Not a future bound.
    conservative_current_bytes = math.ceil(tar_bytes * 1.01) + (32 << 20)
    reserve = Path("/data/202610/7kb/research_delivery/.final_results_reserve_20261106")
    reserve_stat = reserve.lstat()
    scratch = Path("/tmp/grad-research-20261106")
    external_assets = [census(scratch / name) for name in ("raw", "report_fonts", "report_packages", "report_env", "literature_sources", "models")]
    summary = dict(status="passed_with_explicit_external_assets", started_at_utc=started,
        finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        snapshot=dict(path=str(args.snapshot), bytes=args.snapshot.stat().st_size, sha256=actual_sha,
                      manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(), files=len(old),
                      all_member_names_sizes_and_regular_file_types_match_manifest=True,
                      omitted_symlinks=manifest["omitted_symlinks"]),
        current=dict(selected_files=len(inventory), selected_payload_bytes=sum(row["bytes"] for row in inventory),
                     components=components, changes=dict(collections.Counter(row["change"] for row in inventory)),
                     previously_archived_not_selected_now=missing, symlinks=symlinks,
                     external_weight_files=len(weights), external_weight_bytes=sum(row["bytes"] for row in weights),
                     unclassified_nonweight_runrecord_omissions=unknown_runs),
        capacity=dict(reserve_path=str(reserve), actual_size=reserve_stat.st_size, allocated_bytes=reserve_stat.st_blocks * 512,
                      current_estimated_uncompressed_tar_bytes=tar_bytes, conservative_current_package_budget_bytes=conservative_current_bytes,
                      headroom_to_actual_reserve_bytes=reserve_stat.st_size - conservative_current_bytes,
                      current_fits_without_compression=conservative_current_bytes <= min(reserve_stat.st_size, reserve_stat.st_blocks * 512),
                      future_growth_not_bounded=True, final_preflight_rerun_required=True),
        external_assets_not_automatically_bundled=external_assets,
        fixed_omission="requirements-report.lock was excluded by blanket .lock policy; now explicitly retained",
        science_or_scorer_modified=False, sealed_scores_deserialized_or_interpreted=False,
        payload_opening_policy="Only opaque compressed snapshot SHA and PACKAGE_MANIFEST JSON; current payloads stat-only",
        new_archive_created=False, shared_storage_written=False,
        sources=dict(packager_sha256=sha(ROOT / "scripts/package_research.py"), auditor_sha256=sha(__file__)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "current_delivery_inventory.csv", inventory)
    write_csv(args.output_dir / "coverage_by_component.csv", components)
    write_csv(args.output_dir / "changes_since_0838.csv", [row for row in inventory if row["change"] != "same_size_and_mtime"] + missing)
    write_csv(args.output_dir / "excluded_runtime_cache.csv", excluded)
    (args.output_dir / "delivery_coverage_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"current", "external_assets_not_automatically_bundled"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
