#!/usr/bin/env python3
"""Count the frozen evaluation/training scope without reading result payloads."""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-spec", required=True, type=Path)
    parser.add_argument("--successor-queue", required=True, type=Path)
    parser.add_argument("--registry", action="append", required=True, help="phase=path")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    proofs: dict[str, dict] = {}

    def read(path: str | Path, expected: str | None = None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f"Source SHA mismatch: {path}")
        proof = {"path": str(path), "sha256": digest, "bytes": len(raw)}
        if str(path) in proofs and proofs[str(path)] != proof:
            raise ValueError(f"Source changed during accounting: {path}")
        proofs[str(path)] = proof
        return json.loads(raw)

    spec = read(args.final_spec)
    successor = read(args.successor_queue)
    manifests: dict[str, dict] = {}
    manifest_roles: dict[str, set[str]] = collections.defaultdict(set)
    source_queues: dict[str, dict] = {}
    source_jobs: dict[str, dict] = {}
    job_modes: dict[str, str] = {}
    source_references = []
    for role in ("evaluation_sources", "prior_sources"):
        for source in spec[role]:
            queue = read(source["queue_path"], source["queue_sha256"])
            for item in queue:
                if item["id"] in source_queues and source_queues[item["id"]] != item:
                    raise ValueError(f"Conflicting source job: {item['id']}")
                source_queues[item["id"]] = item
            for proof in source["manifests"]:
                path = str(Path(proof["path"]).resolve())
                manifest = read(path, proof["sha256"])
                manifest_roles[path].add(role)
                source_references.append({"role": role, "manifest": path,
                                          "queue": str(Path(source["queue_path"]).resolve())})
                if path in manifests:
                    if manifests[path] != manifest:
                        raise ValueError(f"Conflicting manifest: {path}")
                    continue
                manifests[path] = manifest
                mode = manifest["mode"]
                if mode not in {"nll", "behavior"}:
                    raise ValueError(f"Unexpected mode: {mode}")
                for job in manifest["jobs"]:
                    if job["kind"] not in {"initial", "adapted", "reuse"}:
                        raise ValueError(f"Unknown job kind: {job['kind']}")
                    if job["id"] in source_jobs and source_jobs[job["id"]] != job:
                        raise ValueError(f"Conflicting manifest job: {job['id']}")
                    source_jobs[job["id"]] = job
                    if job["id"] in job_modes and job_modes[job["id"]] != mode:
                        raise ValueError("Job shared across scoring modes")
                    job_modes[job["id"]] = mode

    successor_ids = [job["id"] for job in successor]
    if len(successor_ids) != len(set(successor_ids)):
        raise ValueError("Duplicate successor job IDs")
    successor_by_id = {job["id"]: job for job in successor}
    if successor_by_id != source_queues or set(successor_by_id) != set(source_jobs):
        raise ValueError("Successor is not the exact source-job union")
    for job in successor:
        argv = job["argv"]
        path = str(Path(argv[argv.index("--manifest") + 1]).resolve())
        digest = argv[argv.index("--manifest-sha256") + 1]
        inner_id = argv[argv.index("--job-id") + 1]
        if path not in manifests or digest != proofs[path]["sha256"] or inner_id != job["id"]:
            raise ValueError(f"Wrapper/source binding mismatch: {job['id']}")
        if not any(j["id"] == inner_id for j in manifests[path]["jobs"]):
            raise ValueError("Wrapper points to the wrong manifest")
        if not set(job.get("depends_on", [])) <= set(successor_by_id):
            raise ValueError("Dependency outside the frozen queue")

    def canonical(job_id: str) -> str:
        job = source_jobs[job_id]
        if job["kind"] != "reuse":
            return job_id
        origin = job["inference_origin"]
        path = str(Path(origin["manifest"]).resolve())
        if path not in manifests or proofs[path]["sha256"] != origin["manifest_sha256"]:
            raise ValueError("Unbound reuse origin")
        origin_id = origin["job_id"]
        if source_jobs[origin_id]["kind"] != "adapted" or job_modes[origin_id] != job_modes[job_id]:
            raise ValueError("Reuse does not point to a same-mode adapted forward")
        if origin_id not in job.get("depends_on", []):
            raise ValueError("Reuse lacks its dependency")
        return origin_id

    records_by_mode: dict[str, list[dict]] = {mode: [] for mode in ("nll", "behavior")}
    endpoints: dict[tuple[str, str], dict] = {}
    phase_rows = []
    seen_pairs: dict[str, set] = {mode: set() for mode in records_by_mode}
    endpoint_alias_rows = []
    for path, manifest in manifests.items():
        mode = manifest["mode"]
        records = manifest["records"]
        record_by_id = {r["id"]: r for r in records}
        pairs = {(r["actual_run_id"], r["kind"]) for r in records}
        equivalence_aliases = training_refs = 0
        for record in records:
            if record["kind"] not in {"final", "best"}:
                raise ValueError("Unexpected endpoint kind")
            if record["inference_job_id"] not in source_jobs or record["initial_job_id"] not in source_jobs:
                raise ValueError("Endpoint has an unclosed adapted/initial dependency")
            if source_jobs[record["initial_job_id"]]["kind"] != "initial":
                raise ValueError("Baseline is not initial")
            alias = record.get("inference_alias_of")
            if isinstance(alias, str):
                other = record_by_id[alias]
                if (not record["source_model_equivalence"] or record["is_training_alias"]
                        or record["inference_job_id"] != other["inference_job_id"]
                        or record["actual_run_id"] == other["actual_run_id"]
                        or record["kind"] != other["kind"]
                        or record["weights_identity"] != other["weights_identity"]):
                    raise ValueError("Invalid declared model-equivalence alias")
                equivalence_aliases += 1
                endpoint_alias_rows.append({"mode": mode, "phase": record["phase_name"],
                    "actual_run_id": record["actual_run_id"], "kind": record["kind"],
                    "canonical_actual_run_id": other["actual_run_id"],
                    "shared_forward_job_id": record["inference_job_id"],
                    "proof": record["source_model_equivalence"]["proof"],
                    "proof_sha256": record["source_model_equivalence"]["proof_sha256"]})
            elif isinstance(alias, dict):
                if not record["is_training_alias"] or source_jobs[record["inference_job_id"]]["kind"] != "reuse":
                    raise ValueError("Invalid cross-phase training/output reference")
                training_refs += 1
            elif alias is not None or record["is_training_alias"]:
                raise ValueError("Unclassified endpoint alias")
            records_by_mode[mode].append(record)
            pair = (record["actual_run_id"], record["kind"])
            entry = endpoints.setdefault(pair, {"actual_run_id": pair[0], "kind": pair[1],
                "model_tag": record["model_tag"], "group": record["group"],
                "full_training_target_prediction_tokens": record["full_training_committed_path"]["target_prediction_tokens"],
                "nll_logical_rows": 0, "behavior_logical_rows": 0,
                "nll_canonical_forward_job_id": "", "behavior_canonical_forward_job_id": ""})
            if any(entry[k] != record[k] for k in ("model_tag", "group")):
                raise ValueError("Endpoint identity conflict")
            target = record["full_training_committed_path"]["target_prediction_tokens"]
            if type(target) is not int or entry["full_training_target_prediction_tokens"] != target:
                raise ValueError("Full training counter conflict")
            entry[f"{mode}_logical_rows"] += 1
            field = f"{mode}_canonical_forward_job_id"
            resolved = canonical(record["inference_job_id"])
            if entry[field] and entry[field] != resolved:
                raise ValueError("One actual endpoint has conflicting canonical inference IDs")
            entry[field] = resolved
        kinds = collections.Counter(job["kind"] for job in manifest["jobs"])
        phase_rows.append({"mode": mode, "phase": ";".join(p["name"] for p in manifest.get("phases", [])) or "planned_initials",
            "logical_endpoint_rows": len(records), "local_unique_actual_run_kind": len(pairs),
            "new_unique_actual_run_kind_in_mode": len(pairs - seen_pairs[mode]),
            "proved_model_equivalence_alias_rows": equivalence_aliases,
            "cross_phase_training_reference_rows": training_refs,
            "adapted_forward_jobs": kinds["adapted"], "initial_forward_jobs": kinds["initial"],
            "reuse_materialization_jobs": kinds["reuse"], "unique_jobs_in_manifest": len(manifest["jobs"]),
            "referenced_initial_jobs": len({r["initial_job_id"] for r in records}),
            "manifest_path": path, "manifest_sha256": proofs[path]["sha256"]})
        seen_pairs[mode].update(pairs)

    all_registry_ids = set()
    training_rows = []
    run_targets = {}
    for record in records_by_mode["nll"]:
        run_id = record["actual_run_id"]
        target = record["full_training_committed_path"]["target_prediction_tokens"]
        if run_id in run_targets and run_targets[run_id] != target:
            raise ValueError("Final/best full-training counter conflict")
        run_targets[run_id] = target
    for registry_spec in args.registry:
        name, path = registry_spec.split("=", 1)
        registry = read(path)
        rows = registry if isinstance(registry, list) else registry["jobs"]
        physical = [r for r in rows if r.get("execution_kind") != "alias"]
        ids = {r["id"] for r in physical}
        if len(ids) != len(physical) or ids & all_registry_ids:
            raise ValueError("Duplicate physical training registry IDs")
        all_registry_ids |= ids
        training_rows.append({"phase": name, "registry_rows": len(rows), "physical_training_runs": len(ids),
            "reused_training_reference_rows": len(rows) - len(ids),
            "physically_trained_D_runs": sum(r["group"] == "D" for r in physical),
            "full_training_target_prediction_tokens": sum(run_targets[run_id] for run_id in ids),
            "registry_path": str(Path(path).resolve()), "registry_sha256": proofs[str(Path(path).resolve())]["sha256"]})
    if all_registry_ids != set(run_targets):
        raise ValueError("NLL actual-run set differs from the physical training registries")

    totals = []
    for mode in ("nll", "behavior", "combined"):
        records = sum(records_by_mode.values(), []) if mode == "combined" else records_by_mode[mode]
        ids = set(source_jobs) if mode == "combined" else {j for j in source_jobs if job_modes[j] == mode}
        counts = collections.Counter(source_jobs[j]["kind"] for j in ids)
        totals.append({"mode": mode, "logical_endpoint_rows": len(records),
            "unique_actual_run_kind": len({(r["actual_run_id"], r["kind"]) for r in records}),
            "proved_model_equivalence_alias_rows": sum(isinstance(r.get("inference_alias_of"), str) for r in records),
            "cross_phase_training_reference_rows": sum(r["is_training_alias"] for r in records),
            "adapted_forward_jobs": counts["adapted"], "initial_forward_jobs": counts["initial"],
            "all_forward_jobs": counts["adapted"] + counts["initial"],
            "reuse_materialization_jobs": counts["reuse"], "unique_jobs": len(ids)})
    baselines = []
    for job_id, job in source_jobs.items():
        if job["kind"] != "initial":
            continue
        refs = [r for records in records_by_mode.values() for r in records if r["initial_job_id"] == job_id]
        baselines.append({"mode": job_modes[job_id], "job_id": job_id, "baseline_group": job["baseline_group"],
            "logical_endpoint_references": len(refs),
            "model_tags": ";".join(sorted({r["model_tag"] for r in refs})),
            "data_families": ";".join(sorted({r["data_family"] for r in refs})),
            "phase_names": ";".join(sorted({r["phase_name"] for r in refs}))})
    job_rows = []
    for item in successor:
        job_id = item["id"]
        job = source_jobs[job_id]
        refs = [r for r in records_by_mode[job_modes[job_id]] if r["inference_job_id"] == job_id]
        endpoint_kinds = sorted({r["kind"] for r in refs})
        if job["kind"] != "initial" and len(endpoint_kinds) != 1:
            raise ValueError("Adapted job has missing or mixed checkpoint kinds")
        job_rows.append({"mode": job_modes[job_id], "job_kind": job["kind"], "job_id": job_id,
            "checkpoint_kind": endpoint_kinds[0] if endpoint_kinds else "initial",
            "requires_model_inference": job["kind"] != "reuse", "canonical_job_id": canonical(job_id),
            "referencing_logical_endpoint_rows": len(refs),
            "phase_names": ";".join(sorted({r["phase_name"] for r in refs}))})

    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)

    def write_csv(name, rows):
        with (out / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("scope_by_mode.csv", totals)
    write_csv("scope_by_phase.csv", phase_rows)
    write_csv("training_scope.csv", training_rows)
    write_csv("initial_baselines.csv", baselines)
    write_csv("successor_job_inventory.csv", job_rows)
    write_csv("actual_run_endpoint_inventory.csv", list(endpoints.values()))
    write_csv("proved_model_equivalence_aliases.csv", endpoint_alias_rows)
    summary = {"schema_version": 1, "status": "frozen_scope_accounting_not_execution_completion",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_reference_counts": {"evaluation_sources": len(spec["evaluation_sources"]),
                                    "prior_sources": len(spec["prior_sources"]), "unique_manifests": len(manifests)},
        "evaluation_totals": totals, "training_scope": {
            "physical_training_runs": len(all_registry_ids),
            "physically_trained_D_runs": sum(r["physically_trained_D_runs"] for r in training_rows),
            "full_training_target_prediction_tokens": sum(run_targets.values()),
            "A_references_are_not_new_training": True,
            "D_runs_remain_in_training_cost_despite_inference_equivalence": True},
        "per_mode_endpoint_kinds": {m: dict(collections.Counter(r["kind"] for r in rs)) for m, rs in records_by_mode.items()},
        "unique_actual_endpoint_kinds": dict(collections.Counter(k[1] for k in endpoints)),
        "job_checkpoint_kinds": {m: dict(collections.Counter(f"{r['job_kind']}:{r['checkpoint_kind']}" for r in job_rows if r["mode"] == m)) for m in records_by_mode},
        "checks": {"successor_unique_ids": True, "source_queue_definitions_equal_by_id": True,
                   "wrapper_manifest_SHA_and_job_bindings": True, "all_reuse_origins_closed": True,
                   "registry_physical_runs_equal_NLL_actual_runs": True,
                   "no_state_or_result_payloads_read": True},
        "scope_limits": ["Counts describe frozen scope; no task completion or publication assertion.",
            "Forward jobs are evaluation work items, not counts of model.forward calls, tasks, corpora, or GPU seconds.",
            "Combined unique actual_run_id/kind is a cross-mode union; per-mode unique counts are not additive.",
            "Proved D/B4 aliases share existing forward jobs and add no queue item; cross-phase reuse items materialize outputs with zero extra model inference.",
            "No unproved weight equality is inferred; adapted forward count is not a count of distinct fingerprints or published ZIPs.",
            "Only final-spec, its explicit manifests/queues, successor queue and formal training registries were opened; source hashes in manifests are retained metadata, not fresh payload verification."],
        "source_references": source_references, "source_proofs": list(proofs.values()),
        "outputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in sorted(out.glob("*.csv"))]}
    with (out / "scope_summary.json").open("x") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"evaluation_totals": totals, "training_scope": summary["training_scope"],
                      "summary": str(out / "scope_summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
