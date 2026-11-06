"""Explicit, SHA-attested v2 NLL analysis; never relabel a supplementary phase.

Planning reads manifests/configuration only. Reading heldout scores requires the
caller to opt in after the study owner unseals results.
"""
from __future__ import annotations

from collections import defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from .analysis import (CORPORA, METRICS, PAIRS, endpoint_metrics, mean_std,
                       paired_document_bootstrap, write_csv)

MODEL_REVISIONS = {
    "pythia-410m": "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
    "pythia-1b": "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2",
    "pythia-2.8b": "2a259cdd96a4beb1cdf467512e3904197345f6a9",
}
ROLE_PHASE = {"primary": "formal", "supplement_A": "supplement_A_lambda_sweep",
              "supplement_B": "conditional_bank_variance", "extended": "formal"}
EXECUTION_KEYS = ("torch_version", "cuda_version", "gpu_name", "precision", "device_type", "effective_precision",
    "attention_implementation", "model_weight_dtype", "flash_sdp_enabled", "mem_efficient_sdp_enabled",
    "math_sdp_enabled", "cudnn_sdp_enabled", "deterministic_algorithms", "deterministic_warn_only",
    "allow_tf32_matmul", "allow_tf32_cudnn", "cudnn_deterministic", "cudnn_benchmark", "CUBLAS_WORKSPACE_CONFIG")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path, expected, sources):
    actual = sha(path)
    if not expected or actual != expected:
        raise ValueError(f"frozen source/result SHA differs: {path}")
    sources[str(Path(path))] = actual


def _attestation_path(path):
    path = Path(path)
    return path.with_name(path.stem + "_execution.json")


def prepare_plan(manifest_path, manifest_sha256, *, role, phase_name, model_tag, data_family, checkpoint):
    """No result paths are opened or probed in this function."""
    sources = {}
    project = Path(__file__).resolve().parents[2]
    for implementation in (Path(__file__), Path(__file__).with_name("analysis.py"), project / "scripts/analyze_nll_v2.py"):
        sources[str(implementation)] = sha(implementation)
    _require_hash(manifest_path, manifest_sha256, sources)
    manifest = read(manifest_path)
    if manifest.get("schema_version") != 2 or manifest.get("mode") != "nll":
        raise ValueError("explicit schema2 NLL manifest required")
    if role not in ROLE_PHASE or checkpoint not in {"final", "best"} or model_tag not in MODEL_REVISIONS:
        raise ValueError("unsupported explicit analysis identity")
    if role in {"supplement_A", "supplement_B"} and checkpoint != "final":
        raise ValueError("supplement A/B analysis permits final endpoints only")
    if role in {"primary", "supplement_A", "supplement_B"} and (model_tag, data_family) != ("pythia-410m", "main_v3"):
        raise ValueError("primary/A/B cannot masquerade as another model/data family")
    if role == "extended" and data_family != "extended_v1":
        raise ValueError("extended model/data analysis must remain separate from main_v3")
    phases = [p for p in manifest["phases"] if p["name"] == phase_name]
    if len(phases) != 1:
        raise ValueError("exactly one explicit phase_name required")
    phase = phases[0]
    if (phase["role"], phase["model_tag"], phase["data_family"]) != (role, model_tag, data_family):
        raise ValueError("phase role/model/data identity differs from requested stratum")
    if phase.get("expected_phase") != ROLE_PHASE[role]:
        raise ValueError("phase scientific identity differs from declared role")
    _require_hash(manifest["phase_spec"], manifest["phase_spec_sha256"], sources)
    _require_hash(phase["registry"], phase["registry_sha256"], sources)
    registry = read(phase["registry"])
    registry_by_id = {r["id"]: r for r in registry}
    if len(registry_by_id) != len(registry):
        raise ValueError("duplicate source registry run IDs")
    rows = [r for r in manifest["records"] if r["phase_name"] == phase_name and r["kind"] == checkpoint]
    if len(rows) != phase["rows"] or {r["run_id"] for r in rows} != set(registry_by_id):
        raise ValueError("frozen endpoint manifest does not cover its complete phase registry")
    if len({r["baseline_group"] for r in rows}) != 1:
        raise ValueError("one analysis stratum requires one exact baseline/data/execution identity")
    jobs = {j["id"]: j for j in manifest["jobs"]}
    if len(jobs) != len(manifest["jobs"]):
        raise ValueError("duplicate inference job IDs")
    seen, checked, physical_science = set(), [], {}
    for row in rows:
        if row["id"] in seen:
            raise ValueError("duplicate logical endpoint record")
        seen.add(row["id"])
        if (row["role"], row["model_tag"], row["data_family"], row["phase"]) != (role, model_tag, data_family, ROLE_PHASE[role]):
            raise ValueError("logical record mixes role/model/data/phase")
        source = registry_by_id[row["run_id"]]
        if (source.get("phase") != ROLE_PHASE[role] or source.get("status") in {"provisional", "invalid", "superseded"} or
                (source.get("execution_kind") != "alias" and source.get("active") is not True)):
            raise ValueError("source registry is provisional, inactive or a different scientific phase")
        for key in ("group", "ratio", "weight", "learning_rate"):
            if row[key] != source[key]:
                raise ValueError("logical scientific condition differs from registry")
        actual = source.get("actual_run_id") or source.get("alias_of") or source["id"]
        if row["actual_run_id"] != actual or row["is_training_alias"] != (source.get("execution_kind") == "alias"):
            raise ValueError("logical/actual training alias identity differs")
        seed = source.get("training_seed", source.get("seed"))
        if row["training_seed"] != seed or row.get("selection_seed") != source.get("selection_seed") or row.get("generation_seed") != source.get("generation_seed"):
            raise ValueError("training/selection/generation seed roles differ")
        science = (row["group"], seed, row["ratio"], row["weight"], row["learning_rate"], source.get("bank_chunks"), row.get("selection_seed"))
        if actual in physical_science and physical_science[actual] != science:
            raise ValueError("one actual training ID cannot represent different scientific conditions")
        physical_science[actual] = science
        if role == "supplement_B" and (row["selection_seed"] is None or seed != 11 or row["generation_seed"] != 11 or not source.get("bank_chunks")):
            raise ValueError("B requires selection draws at fixed training/generation seed11 and explicit K")
        group = manifest["baseline_groups"][row["baseline_group"]]
        policy = group["inference_policy"]
        if (policy.get("deterministic_algorithms") is not True or policy.get("deterministic_warn_only") is not False or
                policy.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
            raise ValueError("strict frozen deterministic execution policy required")
        if group["model_tag"] != model_tag or policy["revision"] != MODEL_REVISIONS[model_tag] or set(group["test_paths"]) != set(CORPORA):
            raise ValueError("baseline model/revision/corpora identity differs")
        if set(group["evaluation_input_sha256"]) != set(group["test_paths"].values()):
            raise ValueError("baseline heldout tensor identities are incomplete")
        job = jobs[row["inference_job_id"]]
        if job["baseline_group"] != row["baseline_group"] or row["id"] not in job["record_ids"] or job.get("weights_identity") != row["weights_identity"]:
            raise ValueError("inference job does not attest this logical endpoint")
        if row["initial_job_id"] != group["initial_job_id"]:
            raise ValueError("logical record points to a different initial baseline")
        equivalence = row.get("source_model_equivalence")
        if equivalence:
            _require_hash(equivalence["proof"], equivalence["proof_sha256"], sources)
            proof = read(equivalence["proof"])
            comparison = proof["comparisons"][equivalence["comparison_index"]]
            fingerprint = equivalence["fingerprint"]
            pair = {comparison[side]["checkpoint_sha256"] for side in ("left", "right")}
            if (proof.get("status") != "passed" or proof.get("algorithm") != "sha256_sorted_key_dtype_shape_contiguous_uint8_v1" or
                    comparison.get("bitwise_model_state_equal") is not True or
                    any(comparison[side]["model_state_fingerprint"] != fingerprint for side in ("left", "right")) or
                    pair != {equivalence["left_checkpoint_sha256"], equivalence["right_checkpoint_sha256"]} or
                    row["checkpoint_sha256"] not in pair or row["weights_identity"] != "model_state:" + fingerprint["sha256_model_state"]):
                raise ValueError("complete checkpoint model-state equivalence proof differs")
        full = row["full_training_committed_path"]
        target, replay = row["target_prediction_tokens_at_checkpoint"], row["replay_prediction_tokens_at_checkpoint"]
        if type(target) is not int or type(replay) is not int or target <= 0 or replay < 0 or target > full["target_prediction_tokens"] or replay > full["replay_prediction_tokens"]:
            raise ValueError("checkpoint exposure exceeds full training exposure")
        if checkpoint == "final" and (target, replay) != (full["target_prediction_tokens"], full["replay_prediction_tokens"]):
            raise ValueError("final endpoint is not the full prescribed exposure")
        checked.append({**row, "bank_chunks": source.get("bank_chunks")})
    return {"manifest_path": str(Path(manifest_path)), "manifest_sha256": manifest_sha256, "manifest": manifest,
            "identity": dict(role=role, phase_name=phase_name, model_tag=model_tag, data_family=data_family, checkpoint=checkpoint),
            "records": checked, "sources_sha256": sources, "score_files_read": False, "expected_conditions": len(rows)}


def _load_score(path, *, manifest_sha, job_id, checkpoint, checkpoint_sha, weights_identity, group, sources,
                before_score_read=None):
    path = Path(path)
    att_path = _attestation_path(path)
    att = read(att_path)
    sources[str(att_path)] = sha(att_path)
    if (att.get("manifest_sha256"), att.get("job_id"), att.get("verified_frozen_inputs")) != (manifest_sha, job_id, True):
        raise ValueError("execution attestation does not match frozen manifest/job")
    if att.get("baseline_group") != group["id"] or att.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("execution attestation baseline/checkpoint identity differs")
    if weights_identity is not None and att.get("weights_identity") != weights_identity:
        raise ValueError("execution attestation model-state identity differs")
    if set(att.get("output_sha256", {})) != {str(path)}:
        raise ValueError("execution attestation must bind this exact NLL result")
    _require_hash(path, att["output_sha256"][str(path)], sources)
    if before_score_read is not None:
        before_score_read(path)
    value = read(path)
    policy = group["inference_policy"]
    if (value.get("model"), value.get("revision"), value.get("precision"), value.get("checkpoint")) != (policy["model"], policy["revision"], policy["precision"], checkpoint):
        raise ValueError("result model/revision/precision/checkpoint differs from frozen input")
    execution = value["execution_provenance"]
    for key in EXECUTION_KEYS:
        if key not in execution or key not in policy or type(execution[key]) is not type(policy[key]) or execution[key] != policy[key]:
            raise ValueError("result execution differs from frozen " + key)
    if execution.get("attention_backend_requested") != policy["attention_backend"]:
        raise ValueError("result attention backend differs")
    if set(value["corpora"]) != set(CORPORA):
        raise ValueError("result mixes heldout corpora")
    for name, corpus in value["corpora"].items():
        work = group["work_per_inference"]["corpora"][name]
        if (corpus.get("complete_corpus") is not True or corpus.get("chunks_evaluated") != work["chunks"] or
                corpus.get("chunks_available") != work["chunks"] or corpus.get("valid_prediction_tokens") != work["prediction_tokens"] or
                corpus.get("raw_input_tokens") != work["raw_input_tokens"] or
                len(corpus.get("chunks", [])) != work["chunks"] or corpus.get("precision") != policy["precision"] or
                corpus.get("model_weight_dtype") != policy["model_weight_dtype"]):
            raise ValueError("complete chunk-level corpus/precision/count binding required")
        if corpus.get("kl_direction") != "teacher||student" or corpus.get("kl_vocabulary") != "exact_full_vocabulary" or corpus.get("kl_arithmetic") != "float32":
            raise ValueError("full-vocabulary forward teacher KL required")
        if not math.isclose(corpus["kl"], corpus["sum_kl"] / work["prediction_tokens"], rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("KL sum/count mismatch")
        if not math.isclose(corpus["sum_kl"], sum(x["sum_kl"] for x in corpus["chunks"]), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("chunk KL disagrees with aggregate")
    endpoint_metrics(value["corpora"], value["corpora"])
    return value, att


def collect_scores(plan, *, read_heldout_scores=False, allow_partial=False, n_bootstrap=5000,
                   before_score_read=None):
    if not read_heldout_scores:
        raise PermissionError("heldout score access is sealed; use prepare_plan until explicit unsealing")
    if n_bootstrap != 0 and n_bootstrap < 2:
        raise ValueError("bootstrap needs at least two replicates, or explicit zero to omit intervals")
    for path, expected_sha in plan["sources_sha256"].items():
        if sha(path) != expected_sha:
            raise ValueError("analysis authority or implementation changed after preparation")
    manifest, identity = plan["manifest"], plan["identity"]
    sources, baseline_cache = dict(plan["sources_sha256"]), {}
    endpoint_rows, availability, loaded, attestations = [], [], {}, {}
    jobs = {j["id"]: j for j in manifest["jobs"]}
    for record in plan["records"]:
        path = Path(record["output"])
        present = path.exists() and _attestation_path(path).exists()
        availability.append({"record_id": record["id"], "run_id": record["run_id"], "result_and_attestation_present": present})
        if not present:
            continue
        group = manifest["baseline_groups"][record["baseline_group"]]
        if group["id"] not in baseline_cache:
            origin = group.get("initial_origin")
            baseline, _ = _load_score(group["initial_output"], manifest_sha=origin["manifest_sha256"] if origin else plan["manifest_sha256"],
                job_id=group["initial_job_id"], checkpoint=None, checkpoint_sha=None, weights_identity=None, group=group, sources=sources,
                before_score_read=before_score_read)
            if any(abs(baseline["corpora"][c]["kl"]) > 1e-8 for c in CORPORA):
                raise ValueError("initial teacher self-KL must be numerically zero")
            baseline_cache[group["id"]] = baseline
        job = jobs[record["inference_job_id"]]
        value, att = _load_score(path, manifest_sha=plan["manifest_sha256"], job_id=job["id"],
            checkpoint=record["checkpoint"], checkpoint_sha=record["checkpoint_sha256"], weights_identity=record["weights_identity"], group=group, sources=sources,
            before_score_read=before_score_read)
        canonical_path = Path(job["output"])
        is_alias = path.resolve() != canonical_path.resolve()
        if is_alias:
            reuse = value.get("inference_reuse", {})
            if (att.get("inference_alias") is not True or att.get("logical_record_id") != record["id"] or
                    att.get("inference_subprocess_wall_seconds") != 0 or reuse.get("additional_model_forward_calls") != 0 or
                    reuse.get("checkpoint_sha256") != record["checkpoint_sha256"] or reuse.get("canonical_job_id") != job["id"] or
                    reuse.get("canonical_checkpoint_sha256") != job["checkpoint_sha256"] or
                    Path(reuse.get("canonical_output", "/missing")).resolve() != canonical_path.resolve()):
                raise ValueError("logical inference alias lacks zero-work canonical proof")
            origin = job.get("inference_origin")
            canonical, _ = _load_score(canonical_path, manifest_sha=origin["manifest_sha256"] if origin else plan["manifest_sha256"],
                job_id=origin["job_id"] if origin else job["id"], checkpoint=job["checkpoint"], checkpoint_sha=job["checkpoint_sha256"],
                weights_identity=job["weights_identity"], group=group, sources=sources, before_score_read=before_score_read)
            if canonical["corpora"] != value["corpora"] or canonical["execution_provenance"] != value["execution_provenance"]:
                raise ValueError("alias metrics differ from their canonical result")
            if record["checkpoint_sha256"] != job["checkpoint_sha256"] and not record.get("source_model_equivalence"):
                raise ValueError("different checkpoint bytes require explicit complete model equivalence")
        metric = endpoint_metrics(value["corpora"], baseline_cache[group["id"]]["corpora"])
        endpoint_rows.append({**identity, "record_id": record["id"], "run_id": record["run_id"], "actual_run_id": record["actual_run_id"],
            "group": record["group"], "ratio": record["ratio"], "weight": record["weight"], "learning_rate": record["learning_rate"],
            "training_seed": record["training_seed"], "selection_seed": record["selection_seed"], "generation_seed": record["generation_seed"],
            "bank_chunks": record["bank_chunks"], "target_prediction_tokens": record["target_prediction_tokens_at_checkpoint"],
            "replay_prediction_tokens": record["replay_prediction_tokens_at_checkpoint"],
            "full_training_target_prediction_tokens": record["full_training_committed_path"]["target_prediction_tokens"],
            "full_training_replay_prediction_tokens": record["full_training_committed_path"]["replay_prediction_tokens"],
            "is_training_alias": record["is_training_alias"], "is_inference_alias": is_alias,
            "baseline_group": group["id"], "inference_job_id": job["id"], **metric})
        loaded[record["id"]], attestations[record["id"]] = value, att
    complete = len(endpoint_rows) == len(plan["records"])
    if not complete and not allow_partial:
        raise ValueError("incomplete frozen phase: final statistical output refused")
    replicate = "selection_seed" if identity["role"] == "supplement_B" else "training_seed"
    def setting(row):
        return (row["group"], row["ratio"], row["weight"], row["learning_rate"], row.get("bank_chunks"))
    expected = defaultdict(set)
    for row in plan["records"]:
        expected[setting(row)].add(row[replicate])
    stats = []
    for key, expected_draws in sorted(expected.items(), key=str):
        rows = [r for r in endpoint_rows if setting(r) == key]
        if len({r[replicate] for r in rows}) != len(rows):
            raise ValueError("logical aliases duplicated a replicate within a scientific condition")
        for metric in METRICS:
            result = mean_std(r[metric] for r in rows if r.get(metric) is not None)
            stats.append(dict(group=key[0], ratio=key[1], weight=key[2], learning_rate=key[3], bank_chunks=key[4],
                metric=metric, **result, expected_n=len(expected_draws), complete=result["n"] == len(expected_draws), replicate_unit=replicate,
                uncertainty_scope="conditional selection-draw variability; one training seed and generation pool" if replicate == "selection_seed" else "trained-seed sample variability"))
    paired, paired_stats, bootstrap, endpoint_bootstrap = [], [], [], []
    if n_bootstrap:
        for row in endpoint_rows:
            for corpus in CORPORA:
                for field in ("sum_nll", "sum_kl"):
                    interval = paired_document_bootstrap(loaded[row["record_id"]]["corpora"][corpus],
                        baseline_cache[row["baseline_group"]]["corpora"][corpus], field=field, n_bootstrap=n_bootstrap)
                    metric = ("g_new" if corpus == "target" else "f_old_" + corpus) if field == "sum_nll" else "kl_" + corpus
                    if metric == "g_new":
                        for weighting in ("token_weighted", "equal_document"):
                            part = interval[weighting]
                            part.update(estimate=-part["estimate"], ci_low=-part["ci_high"], ci_high=-part["ci_low"])
                    interval["sign"] = "initial minus adapted target NLL" if metric == "g_new" else "adapted minus initial"
                    endpoint_bootstrap.append({"run_id": row["run_id"], "actual_run_id": row["actual_run_id"],
                        "record_id": row["record_id"], "ratio": row["ratio"], "weight": row["weight"],
                        "learning_rate": row["learning_rate"], "checkpoint": identity["checkpoint"],
                        "target_prediction_tokens": row["target_prediction_tokens"],
                        "replay_prediction_tokens": row["replay_prediction_tokens"],
                        "full_training_target_prediction_tokens": row["full_training_target_prediction_tokens"],
                        "full_training_replay_prediction_tokens": row["full_training_replay_prediction_tokens"],
                        "metric": metric, "corpus": corpus, replicate: row[replicate], "bank_chunks": row["bank_chunks"], **interval})
    for a, b in PAIRS:
        pairs = defaultdict(list)
        for key in expected:
            if key[0] != a or (b,) + key[1:] not in expected:
                continue
            lookup = {(r["group"], r[replicate]): r for r in endpoint_rows if setting(r)[1:] == key[1:]}
            expected_replicates = expected[key] & expected[(b,) + key[1:]]
            for draw in sorted(expected_replicates):
                ra, rb = lookup.get((a, draw)), lookup.get((b, draw))
                if ra is None or rb is None:
                    continue
                if ra["baseline_group"] != rb["baseline_group"]:
                    raise ValueError("paired comparison mixes baseline/data/execution identities")
                item = {"comparison": a + "-" + b, "replicate_unit": replicate, replicate: draw,
                    "ratio": key[1], "weight": key[2], "learning_rate": key[3], "bank_chunks": key[4],
                    "run_a": ra["run_id"], "run_b": rb["run_id"],
                    "training_seed_a": ra["training_seed"], "training_seed_b": rb["training_seed"],
                    "generation_seed_a": ra["generation_seed"], "generation_seed_b": rb["generation_seed"],
                    "target_tokens_a": ra["target_prediction_tokens"], "target_tokens_b": rb["target_prediction_tokens"],
                    "equal_checkpoint_exposure": ra["target_prediction_tokens"] == rb["target_prediction_tokens"]}
                item.update({m: ra[m] - rb[m] if ra.get(m) is not None and rb.get(m) is not None else None for m in METRICS})
                paired.append(item); pairs[key].append(item)
                if n_bootstrap:
                    for corpus in CORPORA:
                        for field in ("sum_nll", "sum_kl"):
                            interval = paired_document_bootstrap(loaded[ra["record_id"]]["corpora"][corpus], loaded[rb["record_id"]]["corpora"][corpus], field=field, n_bootstrap=n_bootstrap)
                            bootstrap.append({"comparison": a + "-" + b, replicate: draw, "bank_chunks": key[4],
                                "ratio": key[1], "weight": key[2], "learning_rate": key[3],
                                "run_a": ra["run_id"], "run_b": rb["run_id"],
                                "actual_run_id_a": ra["actual_run_id"], "actual_run_id_b": rb["actual_run_id"],
                                "record_id_a": ra["record_id"], "record_id_b": rb["record_id"],
                                "checkpoint": identity["checkpoint"],
                                "target_tokens_a": ra["target_prediction_tokens"], "target_tokens_b": rb["target_prediction_tokens"],
                                "replay_tokens_a": ra["replay_prediction_tokens"], "replay_tokens_b": rb["replay_prediction_tokens"],
                                "full_training_target_tokens_a": ra["full_training_target_prediction_tokens"],
                                "full_training_target_tokens_b": rb["full_training_target_prediction_tokens"],
                                "full_training_replay_tokens_a": ra["full_training_replay_prediction_tokens"],
                                "full_training_replay_tokens_b": rb["full_training_replay_prediction_tokens"],
                                "equal_checkpoint_exposure": ra["target_prediction_tokens"] == rb["target_prediction_tokens"],
                                "corpus": corpus, "raw_sign": "NLL/KL A minus B; G_new difference reverses the NLL sign", **interval})
            for metric in METRICS:
                result = mean_std(p[metric] for p in pairs[key] if p.get(metric) is not None)
                paired_stats.append({"comparison": a + "-" + b, "ratio": key[1], "weight": key[2], "learning_rate": key[3], "bank_chunks": key[4],
                    "metric": metric, **result, "expected_n": len(expected_replicates), "complete": result["n"] == len(expected_replicates), "replicate_unit": replicate})
    candidates = []
    for key in sorted(expected, key=str):
        metric_stats = {s["metric"]: s for s in stats if (s["group"], s["ratio"], s["weight"], s["learning_rate"], s["bank_chunks"]) == key}
        candidates.append({"group": key[0], "ratio": key[1], "weight": key[2], "learning_rate": key[3], "bank_chunks": key[4],
            "complete": all(metric_stats[m]["complete"] for m in ("g_new", "f_old_c4", "f_old_wikitext")),
            "n": metric_stats["g_new"]["n"], "replicate_unit": replicate,
            **{m + "_" + s: metric_stats[m][s] for m in ("g_new", "f_old_c4", "f_old_wikitext") for s in ("mean", "std_ddof1")}})
    ready = [c for c in candidates if c["complete"]]
    for candidate in candidates:
        for name, old in (("c4", ("f_old_c4",)), ("wikitext", ("f_old_wikitext",)), ("joint", ("f_old_c4", "f_old_wikitext"))):
            def dominates(other):
                a = [other["g_new_mean"]] + [-other[m + "_mean"] for m in old]
                b = [candidate["g_new_mean"]] + [-candidate[m + "_mean"] for m in old]
                return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))
            candidate["pareto_" + name] = not any(dominates(other) for other in ready) if candidate["complete"] else None
        candidate["frontier_is_partial"] = not complete
    training_costs, inference_costs = {}, {}
    for row in plan["records"]:
        key = (identity["model_tag"], row["actual_run_id"])
        full_cost = row["full_training_committed_path"]
        parent = full_cost["parent_wall_seconds"]
        total = sum(parent.values()) if all(v is not None for v in parent.values()) else None
        factor = 0 if row["is_training_alias"] else 1
        cost = {"actual_run_id": row["actual_run_id"], "model_tag": identity["model_tag"],
                "full_training_target_prediction_tokens": full_cost["target_prediction_tokens"],
                "full_training_replay_prediction_tokens": full_cost["replay_prediction_tokens"],
                **parent, "committed_parent_sum_seconds": total,
                "incremental_phase_parent_sum_seconds": total * factor if total is not None else None,
                "incremental_phase_training_cost_factor": factor, "teacher_subtimers_additive": False,
                "cost_scope": "committed full training path; may omit failures/uncommitted work; separate project inventory remains authoritative",
                "do_not_sum_across_final_best_tables": True}
        if key in training_costs and training_costs[key] != cost:
            raise ValueError("one actual training ID has inconsistent costs")
        training_costs[key] = cost
        if row["id"] in attestations:
            att = attestations[row["id"]]
            job = jobs[row["inference_job_id"]]
            inference_costs[job["id"]] = {"inference_job_id": job["id"], "kind": job["kind"],
                "incremental_inference_subprocess_wall_seconds": 0 if job["kind"] == "reuse" else att.get("canonical_inference_subprocess_wall_seconds", att["inference_subprocess_wall_seconds"]),
                "teacher_subtimers_additive": False, "records_share_one_physical_inference": True,
                "deduplicate_job_id_across_all_role_and_endpoint_tables": True}
    for path, expected_sha in sources.items():
        if sha(path) != expected_sha:
            raise ValueError("analysis input changed during statistical computation")
    return {"identity": identity, "complete": complete, "expected_conditions": len(plan["records"]), "completed_conditions": len(endpoint_rows),
        "score_files_read": True, "replicate_unit": replicate, "endpoints": endpoint_rows, "endpoint_statistics": stats,
        "all_candidates_frontier": candidates,
        "paired_effects": paired, "paired_statistics": paired_stats, "document_bootstrap": bootstrap, "endpoint_document_bootstrap": endpoint_bootstrap,
        "availability": availability, "actual_training_costs": list(training_costs.values()), "physical_inference_costs": list(inference_costs.values()),
        "sources_sha256": sources, "test_lambda_selection_performed": False,
        "bootstrap_replicates": n_bootstrap, "bootstrap_seed": 20261106, "confidence": .95,
        "conditional_interval_scope": "conditional on trained checkpoints; document-bootstrap does not replace trained-seed or selection-draw SD"}


def write_bundle(bundle, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table_keys = ("endpoints", "endpoint_statistics", "paired_effects", "paired_statistics", "all_candidates_frontier", "availability", "actual_training_costs", "physical_inference_costs")
    for key in table_keys:
        write_csv(output / (key + ".csv"), [{**bundle["identity"], **row} for row in bundle[key]])
    (output / "document_bootstrap.json").write_text(json.dumps(bundle["document_bootstrap"], ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "endpoint_document_bootstrap.json").write_text(json.dumps(bundle["endpoint_document_bootstrap"], ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    meta = {k: v for k, v in bundle.items() if k not in set(table_keys) | {"document_bootstrap", "endpoint_document_bootstrap"}}
    meta.update(generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), analysis_source_sha256=sha(__file__))
    (output / "analysis_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return meta


def plot_frontier(bundle, output, *, figure_width=10):
    """One explicit model/data/role/endpoint per figure, with sample SD bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(figure_width, max(3.3,figure_width*.41)), constrained_layout=True)
    groups = sorted({r["group"] for r in bundle["endpoints"]})
    colors = {g: plt.get_cmap("tab10")(i % 10) for i, g in enumerate(groups)}
    for ax, corpus, label in zip(axes, ("c4", "wikitext"), ("C4", "WikiText-103")):
        for group in groups:
            rows = [r for r in bundle["endpoints"] if r["group"] == group]
            ax.scatter([r["g_new"] for r in rows], [r["f_old_" + corpus] for r in rows], color=colors[group], s=12, alpha=.3)
            candidates = [r for r in bundle["all_candidates_frontier"] if r["group"] == group and r["n"]]
            for i, candidate in enumerate(candidates):
                ax.errorbar(candidate["g_new_mean"], candidate["f_old_" + corpus + "_mean"],
                    xerr=candidate["g_new_std_ddof1"], yerr=candidate["f_old_" + corpus + "_std_ddof1"],
                    fmt="o", ms=4, capsize=2, color=colors[group], label=group if i == 0 else None)
        ax.axhline(0, color="0.7", lw=.7); ax.axvline(0, color="0.7", lw=.7)
        ax.set(xlabel="Adaptation gain G_new (nats/token)", ylabel=f"Retention loss F_old: {label} (nats/token)")
        if figure_width <= 7:
            ax.set_xlabel("Adaptation gain G_new (nats/token)",fontsize=8)
            ax.set_ylabel(f"Retention loss F_old: {label} (nats/token)",fontsize=8)
            ax.tick_params(labelsize=7)
        ax.grid(alpha=.15)
    axes[1].legend(frameon=False, fontsize=8)
    identity = bundle["identity"]
    fig.suptitle(f"{identity['model_tag']} | {identity['role']} | {identity['data_family']} | {identity['checkpoint']}\n"
                 f"{'Complete' if bundle['complete'] else 'PARTIAL'}; points: {bundle['replicate_unit']}; bars: mean +/- sample SD", fontsize=10)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / ("adaptation_retention_frontier." + extension), dpi=240)
    plt.close(fig)
    (output / "图形说明.md").write_text("同一图严格限定一个模型、角色、数据版本及端点；浅色点是逐重复结果，实心点及误差条是各方法/b/λ/K条件的均值与样本SD。SD不是置信区间。B的重复单位是固定训练/生成seed下的selection draw，不能称独立训练seed。图中全部候选均保留，不能据test前沿重新选择λ。具体条件及完整性见CSV与analysis_manifest；partial图不作最终结果。\n")
