"""Analysis restricted to the registered, post-validation in-pool diagnostic.

This module has no heldout/behavioral-result discovery and never calls a model.
Its 15-result scope is checked before any score JSON is opened.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from .analysis import mean_std, write_csv
from .nll_v2_analysis import EXECUTION_KEYS, MODEL_REVISIONS, read, sha

PHASE = "post_validation_exploratory_replay_fit"
CORPORA = ("bank", "complement")
SEEDS = (11, 22, 33)
METRICS = ("bank_nll", "complement_nll", "bank_delta_nll", "complement_delta_nll", "nll_gap",
           "bank_kl", "complement_kl", "bank_delta_kl", "complement_delta_kl", "kl_gap")


def validate_scope(manifest):
    """Reject incomplete/foreign scope before inspecting any actual result."""
    if (manifest.get("schema_version"), manifest.get("mode"), manifest.get("phase"), manifest.get("role")) != (
            2, "nll", PHASE, "exploratory_diagnostic"):
        raise ValueError("only the explicit schema2 exploratory replay-fit scope is permitted")
    if (manifest.get("primary_evaluation") is not False or manifest.get("independent_test") is not False or
            manifest.get("specified_after_main_validation") is not True):
        raise ValueError("in-pool/post-validation interpretation must be explicit")
    jobs = {j["id"]: j for j in manifest["jobs"]}
    groups = manifest["baseline_groups"]
    records = manifest["records"]
    expected = {(seed, method, endpoint) for seed in SEEDS for method in ("B3", "B4") for endpoint in ("final", "best")}
    if (len(jobs) != 15 or len(manifest["jobs"]) != 15 or len(groups) != 3 or len(records) != 12 or
            {(r["training_seed"], r["group"], r["kind"]) for r in records} != expected or
            len({r["id"] for r in records}) != 12):
        raise ValueError("complete 3 initial + 12 adapted diagnostic cases required")
    seen_jobs = set()
    for gid, group in groups.items():
        policy, seed = group["inference_policy"], group["seed"]
        if (seed not in SEEDS or group["id"] != gid or group["model_tag"] != "pythia-410m" or
                policy["revision"] != MODEL_REVISIONS["pythia-410m"] or set(group["test_paths"]) != set(CORPORA) or
                policy.get("deterministic_algorithms") is not True or policy.get("deterministic_warn_only") is not False or
                policy.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
            raise ValueError("model/data/seed/strict execution identity differs")
        initial = jobs[group["initial_job_id"]]
        if (initial["kind"], initial["baseline_group"], initial["output"]) != ("initial", gid, group["initial_output"]):
            raise ValueError("same-seed initial source differs")
        seen_jobs.add(initial["id"])
    if {g["seed"] for g in groups.values()} != set(SEEDS):
        raise ValueError("initial corpus identities must cover all three seeds")
    for record in records:
        group = groups[record["baseline_group"]]
        job = jobs[record["inference_job_id"]]
        if (record["phase"], record["phase_name"], record["role"], record["data_family"], record["model_tag"]) != (
                PHASE, PHASE, "exploratory_diagnostic", "same_pool_bank_complement", "pythia-410m"):
            raise ValueError("foreign logical role/model/data identity")
        if (record["training_seed"] != group["seed"] or record["initial_job_id"] != group["initial_job_id"] or
                record["actual_run_id"] != record["run_id"] or record.get("is_training_alias") is not False or
                record["ratio"] != .05 or record["weight"] != .05 or record["learning_rate"] != 3e-5):
            raise ValueError("scientific condition, source run or same-seed baseline differs")
        if (job["kind"] != "adapted" or job["baseline_group"] != group["id"] or job["record_ids"] != [record["id"]] or
                any(job[k] != record[k] for k in ("checkpoint", "checkpoint_sha256", "weights_identity", "output"))):
            raise ValueError("physical inference source differs from its logical case")
        exposure = (10_000_000, 500_000) if record["kind"] == "final" else (3_270_400, 163_520)
        if (record["target_prediction_tokens_at_checkpoint"], record["replay_prediction_tokens_at_checkpoint"]) != exposure:
            raise ValueError("actual final/best exposure differs from this frozen diagnostic")
        full = record["full_training_committed_path"]
        if (full["target_prediction_tokens"], full["replay_prediction_tokens"]) != (10_000_000, 500_000):
            raise ValueError("complete training cost must be separate from best exposure")
        seen_jobs.add(job["id"])
    if seen_jobs != set(jobs):
        raise ValueError("unregistered/unused inference jobs are forbidden")
    return jobs


def validate_corpus(corpus, doc_ids, *, precision, weight_dtype):
    """Bind ordered raw-data identities, causal counts and full-vocabulary KL."""
    count, tokens = len(doc_ids), len(doc_ids) * 511
    if (corpus.get("complete_corpus") is not True or corpus.get("chunks_evaluated") != count or
            corpus.get("chunks_available") != count or corpus.get("raw_input_tokens") != count * 512 or
            corpus.get("valid_prediction_tokens") != tokens or corpus.get("prediction_tokens") != tokens or
            corpus.get("precision") != precision or corpus.get("model_weight_dtype") != weight_dtype or
            corpus.get("kl_direction") != "teacher||student" or corpus.get("kl_vocabulary") != "exact_full_vocabulary" or
            corpus.get("kl_arithmetic") != "float32"):
        raise ValueError("complete corpus, causal count or KL/precision contract differs")
    chunks = corpus.get("chunks", [])
    if len(chunks) != count:
        raise ValueError("every frozen chunk must be present")
    if [(c.get("chunk_index"), c.get("doc_id"), c.get("valid_prediction_tokens")) for c in chunks] != [
            (i, doc, 511) for i, doc in enumerate(doc_ids)]:
        raise ValueError("ordered chunk identities differ even if counts/doc sets agree")
    for metric in ("nll", "kl"):
        total = math.fsum(float(c["sum_" + metric]) for c in chunks)
        if (not math.isfinite(total) or not math.isfinite(float(corpus[metric])) or
                not math.isclose(total, corpus["sum_" + metric], rel_tol=1e-9, abs_tol=1e-8) or
                not math.isclose(corpus[metric], total / tokens, rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError("chunk sums, aggregate sums and token weighting disagree")
        for chunk in chunks:
            if not math.isclose(chunk[metric], chunk["sum_" + metric] / 511, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("individual chunk mean and causal count disagree")


def contrast(adapted, initial):
    """Positive gap means greater initial-relative fit on bank than complement."""
    result = {}
    for metric in ("nll", "kl"):
        for corpus in CORPORA:
            result[corpus + "_" + metric] = float(adapted[corpus][metric])
            result[corpus + "_initial_" + metric] = float(initial[corpus][metric])
            result[corpus + "_delta_" + metric] = result[corpus + "_" + metric] - result[corpus + "_initial_" + metric]
        result[metric + "_gap"] = result["complement_delta_" + metric] - result["bank_delta_" + metric]
    return result


def summarize(rows):
    if len(rows) != 12 or {(r["seed"], r["group"], r["endpoint"]) for r in rows} != {
            (s, g, e) for s in SEEDS for g in ("B3", "B4") for e in ("final", "best")}:
        raise ValueError("refusing partial or duplicated diagnostic statistics")
    grouped, paired, paired_stats = [], [], []
    for endpoint in ("final", "best"):
        for group in ("B3", "B4"):
            subset = [r for r in rows if (r["group"], r["endpoint"]) == (group, endpoint)]
            for metric in METRICS:
                grouped.append(dict(endpoint=endpoint, group=group, metric=metric, **mean_std([r[metric] for r in subset])))
        for seed in SEEDS:
            left, right = [next(r for r in rows if (r["group"], r["endpoint"], r["seed"]) == (g, endpoint, seed)) for g in ("B3", "B4")]
            paired.append(dict(endpoint=endpoint, seed=seed, contrast="B3-B4", **{k: left[k] - right[k] for k in METRICS}))
        for metric in METRICS:
            paired_stats.append(dict(endpoint=endpoint, contrast="B3-B4", metric=metric,
                                     **mean_std([r[metric] for r in paired if r["endpoint"] == endpoint])))
    return grouped, paired, paired_stats


def collect(manifest_path, expected_sha, timing_path, *, unseal_replay_fit=False):
    if not unseal_replay_fit:
        raise PermissionError("only an explicit replay-fit-only unseal may read these results")
    import torch
    torch.set_num_threads(2)
    sources = {}
    def check(path, expected):
        path = str(path)
        observed = sha(path)
        if observed != expected:
            raise ValueError("frozen SHA differs: " + path)
        sources[path] = observed
    check(manifest_path, expected_sha)
    manifest = read(manifest_path)
    jobs = validate_scope(manifest)
    # Read no primary/behavioral score paths: this authority is a manifest only.
    for path, expected in manifest["common_input_sha256"].items():
        check(path, expected)
    for job in jobs.values():
        for path, expected in job["input_sha256"].items():
            if path != job.get("checkpoint") and path not in sources:
                check(path, expected)
    check(manifest["source_protocol"], manifest["source_protocol_sha256"])
    timing = read(timing_path)["replay_fit"]
    sources[str(timing_path)] = sha(timing_path)
    if timing.get("all_15_completed") is not True or len(timing["jobs"]) != 15:
        raise ValueError("all fifteen completions must be independently attested")
    timed = {r["id"]: r for r in timing["jobs"]}
    if set(timed) != set(jobs):
        raise ValueError("timing scope differs from exact diagnostic jobs")
    data, partition_proofs = {}, []
    for group in manifest["baseline_groups"].values():
        seed = group["seed"]
        pool_path = next(p for p in jobs[group["initial_job_id"]]["input_sha256"] if p.endswith(f"pool_seed{seed}.pt"))
        input_hashes = jobs[group["initial_job_id"]]["input_sha256"]
        for path, expected in input_hashes.items():
            if path not in sources:
                check(path, expected)
        pool = torch.load(pool_path, map_location="cpu", weights_only=False)
        indices = {}
        for corpus, path in group["test_paths"].items():
            check(path, input_hashes[path])
            tensor = torch.load(path, map_location="cpu", weights_only=False)
            idx = tensor["pool_indices"].tolist()
            expected_count = 196 if corpus == "bank" else 1761
            if (len(idx) != expected_count or len(set(idx)) != len(idx) or list(tensor["input_ids"].shape) != [expected_count, 512] or
                    not torch.equal(tensor["input_ids"], pool["input_ids"][idx]) or
                    tensor["doc_ids"] != [pool["doc_ids"][i] for i in idx]):
                raise ValueError("bank/complement raw tokens or original ordered document identities differ")
            data[seed, corpus] = tensor
            indices[corpus] = set(idx)
        if indices["bank"] & indices["complement"] or indices["bank"] | indices["complement"] != set(range(1957)):
            raise ValueError("bank and complement must be disjoint and exactly cover the frozen pool")
        partition_proofs.append(dict(seed=seed, bank_chunks=196, complement_chunks=1761, disjoint=True, cover_original_pool=True,
                                     raw_tokens_and_doc_ids_equal_pool=True))
    loaded, inference_costs = {}, []
    for job in jobs.values():
        completion = timed[job["id"]]
        if completion["state"]["status"] != "completed" or completion["state"]["returncode"] != 0 or completion["output_hashes_valid"] is not True:
            raise ValueError("incomplete/failed inference cannot enter analysis")
        att_path = completion["attestation"]
        check(att_path, completion["attestation_sha256"])
        att = read(att_path)
        group = manifest["baseline_groups"][job["baseline_group"]]
        if (att.get("manifest_sha256"), att.get("job_id"), att.get("baseline_group"), att.get("verified_frozen_inputs"),
                att.get("checkpoint_sha256"), att.get("weights_identity"), att.get("inference_alias")) != (
                expected_sha, job["id"], group["id"], True, job.get("checkpoint_sha256"), job.get("weights_identity"), False):
            raise ValueError("execution attestation source/model/job differs")
        if att.get("actual_argv") != job["argv"] or set(att["output_sha256"]) != {job["output"]}:
            raise ValueError("physical scorer invocation or exact output scope differs")
        check(job["output"], att["output_sha256"][job["output"]])
        result = read(job["output"])
        policy = group["inference_policy"]
        if (result.get("model"), result.get("revision"), result.get("precision"), result.get("checkpoint")) != (
                policy["model"], policy["revision"], policy["precision"], job.get("checkpoint")):
            raise ValueError("result model/revision/precision/checkpoint differs")
        execution = result["execution_provenance"]
        for key in EXECUTION_KEYS:
            if key not in execution or key not in policy or type(execution[key]) is not type(policy[key]) or execution[key] != policy[key]:
                raise ValueError("result execution policy differs: " + key)
        if execution.get("attention_backend_requested") != policy["attention_backend"] or set(result["corpora"]) != set(CORPORA):
            raise ValueError("result backend or corpus scope differs")
        for name in CORPORA:
            validate_corpus(result["corpora"][name], data[group["seed"], name]["doc_ids"],
                            precision=policy["precision"], weight_dtype=policy["model_weight_dtype"])
        loaded[job["id"]] = result
        inference_costs.append(dict(job_id=job["id"], seed=group["seed"], kind=job["kind"],
            subprocess_wall_seconds=att["inference_subprocess_wall_seconds"], verification_seconds=att["verification_wall_seconds"],
            corpus_wall_seconds=sum(c["wall_seconds"] for c in result["corpora"].values()),
            teacher_forward_seconds_nested=sum(c["teacher_forward_seconds"] for c in result["corpora"].values())))
    rows, training_costs = [], {}
    for record in manifest["records"]:
        rows.append(dict(seed=record["training_seed"], group=record["group"], endpoint=record["kind"], run_id=record["run_id"],
            actual_run_id=record["actual_run_id"], target_prediction_tokens=record["target_prediction_tokens_at_checkpoint"],
            replay_prediction_tokens=record["replay_prediction_tokens_at_checkpoint"], full_training_target_tokens=10_000_000,
            full_training_replay_tokens=500_000, bank_prediction_tokens=100156, complement_prediction_tokens=899871,
            **contrast(loaded[record["inference_job_id"]]["corpora"], loaded[record["initial_job_id"]]["corpora"])))
        full = record["full_training_committed_path"]
        if record["actual_run_id"] in training_costs and training_costs[record["actual_run_id"]] != full:
            raise ValueError("best/final full-training cost records disagree")
        training_costs[record["actual_run_id"]] = full
    for path, expected in sources.items():
        if sha(path) != expected:
            raise ValueError("analysis input changed during collection")
    grouped, paired, paired_stats = summarize(rows)
    return dict(status="complete_exploratory_only", role="exploratory_diagnostic", phase=PHASE, rows=rows,
                group_statistics=grouped, paired_rows=paired, paired_statistics=paired_stats, sources_sha256=sources,
                partition_proofs=partition_proofs, inference_costs=inference_costs, unique_training_costs=training_costs,
                checkpoint_source_hash_policy="Executor SHA-verified frozen checkpoints; analyzer rehashes result, attestation, data and scorer files, not 39GB of training checkpoints.",
                score_scope="Only the 15 registered replay-fit outputs; no primary/extended heldout NLL or behavioral scores opened.",
                interpretation=manifest["contrasts"])


def export(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, key in (("per_seed_endpoints", "rows"), ("group_mean_sd", "group_statistics"),
                      ("paired_B3_minus_B4", "paired_rows"), ("paired_mean_sd", "paired_statistics"),
                      ("inference_costs_unique_jobs", "inference_costs")):
        write_csv(output / (name + ".csv"), result[key])
    costs = [dict(actual_run_id=run, target_prediction_tokens=full["target_prediction_tokens"],
                  replay_prediction_tokens=full["replay_prediction_tokens"],
                  incremental_training_seconds_for_this_diagnostic=0, training_already_accounted_in_primary=True,
                  **full["parent_wall_seconds"])
             for run, full in result["unique_training_costs"].items()]
    write_csv(output / "training_costs_unique_runs.csv", costs)
    result["analysis_source_sha256"] = sha(__file__)
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _plot(result, output)


def _plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = {"B3": "#b35a00", "B4": "#007d9c"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    for col, endpoint in enumerate(("final", "best")):
        top, bottom = axes[:, col]
        for gi, group in enumerate(("B3", "B4")):
            rows = sorted((r for r in result["rows"] if (r["group"], r["endpoint"]) == (group, endpoint)), key=lambda r: r["seed"])
            for si, row in enumerate(rows):
                top.plot([gi*3-.2+si*.2, gi*3+1-.2+si*.2], [row["bank_delta_nll"], row["complement_delta_nll"]],
                         "o-", color=colors[group], alpha=.55, linewidth=.8, markersize=4)
            for offset, corpus in enumerate(CORPORA):
                ys = [r[corpus + "_delta_nll"] for r in rows]
                top.errorbar(gi*3+offset, np.mean(ys), yerr=np.std(ys, ddof=1), fmt="s", color="black", capsize=4, markersize=5, zorder=5)
        top.set_xticks([0,1,3,4], ["CE bank", "CE complement", "KL bank", "KL complement"], rotation=15)
        top.set_ylabel("NLL change from initial (nats / token)")
        top.set_title("Final: 10M target tokens" if endpoint == "final" else "Validation-selected best: 3.2704M")
        for si, seed in enumerate(SEEDS):
            rows = [next(r for r in result["rows"] if (r["group"],r["endpoint"],r["seed"]) == (g, endpoint, seed)) for g in ("B3","B4")]
            bottom.plot([-.15+si*.15, 1-.15+si*.15], [r["nll_gap"] for r in rows], "o-", color=f"C{si}", label=f"Seed {seed}", alpha=.75)
        for gi, group in enumerate(("B3","B4")):
            vals = [r["nll_gap"] for r in result["rows"] if (r["group"],r["endpoint"]) == (group,endpoint)]
            bottom.errorbar(gi, np.mean(vals), yerr=np.std(vals,ddof=1), fmt="s", color="black", capsize=5, zorder=5)
        bottom.set_xticks([0,1],["B3: sampled-label CE", "B4: forward KL"])
        bottom.set_ylabel("Gap: delta NLL complement - bank")
        for ax in (top,bottom):
            ax.axhline(0,color="grey",linestyle=":",linewidth=.8); ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",alpha=.2)
    axes[1,1].legend(fontsize=8)
    fig.suptitle("Post-validation exploratory replay fit | same generation pool\nPoints: paired seeds; black squares: mean +/- sample SD (n=3)",fontsize=12)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / ("replay_fit_nll." + suffix), dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, endpoint in zip(axes, ("final", "best")):
        for gi, group in enumerate(("B3", "B4")):
            subset = [r for r in result["rows"] if (r["group"],r["endpoint"]) == (group,endpoint)]
            for row in subset:
                ax.plot([gi*3,gi*3+1],[row["bank_kl"],row["complement_kl"]],"o-",color=colors[group],alpha=.5,markersize=4)
            for offset, corpus in enumerate(CORPORA):
                vals=[r[corpus+"_kl"] for r in subset]
                ax.errorbar(gi*3+offset,np.mean(vals),yerr=np.std(vals,ddof=1),fmt="s",color="black",capsize=4,zorder=5)
        ax.set_xticks([0,1,3,4],["CE bank","CE complement","KL bank","KL complement"],rotation=15)
        ax.set_title("Final: 10M target tokens" if endpoint=="final" else "Validation-selected best: 3.2704M")
        ax.set_ylabel("Forward KL(initial teacher || adapted)")
        ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Exploratory same-pool diagnostic | paired seeds, mean +/- sample SD (n=3)")
    for suffix in ("png","pdf","svg"):
        fig.savefig(output / ("replay_fit_forward_kl."+suffix),dpi=300)
    plt.close(fig)
