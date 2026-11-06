#!/usr/bin/env python3
"""Independent CPU audit of an already ROOT-unsealed, complete behavior phase.

Never imports the collector, scorer or guards; never opens checkpoint payloads.
This is a reviewer, not a mechanism for unsealing an incomplete analysis.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STRATA = ("role", "phase_name", "data_family", "model_tag", "baseline_group")
CELL = (*STRATA, "checkpoint_kind", "group", "ratio", "weight", "learning_rate", "task", "metric")
PAIR_CELL = (*STRATA, "checkpoint_kind", "ratio", "weight", "learning_rate", "task", "metric")
PAIRS = (("M", "B4"), ("C", "B4"), ("B2", "B1"), ("B4", "B3"), ("D", "B4"))
SELECTED_PAIRS = (("M", "B4"), ("C", "B4"), ("B2", "B1"), ("B4", "B3"), ("B3", "B1"), ("B4", "B2"))
QUANTITIES = ("accuracy", "accuracy_delta", "conditional_forgetting_rate", "conditional_acquisition_rate")
SCOPE = ("Conditional on fixed checkpoints and these IID/exchangeable questions. Shared activity/context "
         "dependence (including HellaSwag) is uncorrected; these are not training-replicate or automatic population CIs.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sig(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class Reader:
    """Fresh, stable bytes once per path, with a hard checkpoint/archive prohibition."""
    def __init__(self):
        self.sources, self.raw = {}, {}

    def bytes(self, path, expected=None):
        path = str(Path(path).resolve())
        require(Path(path).suffix in (".json", ".jsonl", ".csv", ".py", ".md"), "Reviewer refuses payload/source type: " + path)
        if path not in self.raw:
            before = sig(Path(path).stat())
            require(before[2] < 256 << 20, "Reviewer refuses unexpectedly large input")
            with open(path, "rb") as handle:
                require(sig(os.fstat(handle.fileno())) == before, "Path/fd identity changed")
                raw = handle.read()
                require(sig(os.fstat(handle.fileno())) == before, "Input changed while reading")
            require(sig(Path(path).stat()) == before, "Input path changed while reading")
            self.raw[path] = raw
            self.sources[path] = {"sha256": sha_bytes(raw), "bytes": len(raw), "five_stat": list(before)}
        require(sig(Path(path).stat()) == tuple(self.sources[path]["five_stat"]), "Cached input changed during review")
        if expected is not None:
            require(self.sources[path]["sha256"] == expected, "SHA differs: " + path)
        return self.raw[path]

    def proof(self, proof):
        require(set(proof) >= {"path", "sha256"}, "Missing path/SHA proof")
        return json.loads(self.bytes(proof["path"], proof["sha256"]))

    def json(self, path):
        return json.loads(self.bytes(path))

    def rows(self, path):
        return [json.loads(line) for line in self.bytes(path).splitlines() if line.strip()]

    def csv(self, path):
        def value(v):
            if v == "": return None
            if v in ("True", "False"): return v == "True"
            try: return float(v)
            except ValueError: return v
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(io.StringIO(self.bytes(path).decode()))]

    def finish(self):
        for path, item in self.sources.items():
            require(sig(Path(path).stat()) == tuple(item["five_stat"]), "Input changed before review completion")


def verify_gate(reader, complete_proof, manifest_proof):
    """Only metadata until every explicit completion/authority/first-read proof passes."""
    complete = reader.proof(complete_proof)
    require(complete.get("complete") is True, "Missing successful COMPLETE")
    plan_proof, authority_proof = complete["plan"], complete["authority"]
    plan, authority = reader.proof(plan_proof), reader.proof(authority_proof)
    require(plan["manifest"] == manifest_proof, "COMPLETE plan binds another manifest")
    require(authority.get("authorized_by") == "root" and authority.get("allow_score_analysis") is True
            and authority.get("allow_partial") is False and authority.get("plan") == plan_proof
            and authority.get("manifest") == manifest_proof, "Exact full-phase ROOT authority required")
    manifest = reader.proof(manifest_proof)
    require(manifest.get("schema_version") == 2 and manifest.get("mode") == "behavior", "Behavior v2 manifest required")
    require(manifest["analysis_policy"]["bootstrap_replicates"] == 5000
            and manifest["analysis_policy"]["bootstrap_seed"] == 20261106, "Frozen bootstrap policy changed")
    require(manifest["records"] and len({r["id"] for r in manifest["records"]}) == len(manifest["records"]), "Incomplete/duplicate logical inventory")
    require(len({j["id"] for j in manifest["jobs"]}) == len(manifest["jobs"]), "Duplicate physical inventory")
    jobs = {j["id"]: j for j in manifest["jobs"]}
    require(all(r["inference_job_id"] in jobs and r.get("selection_seed") is None for r in manifest["records"]), "Unknown physical job or unsupported selection-draw behavior")
    require(len({(r["phase_name"], r["model_tag"], r["data_family"]) for r in manifest["records"]}) == 1, "Do not mix behavior phases/models/data")
    for record in manifest["records"]:
        require(set(manifest["baseline_groups"][record["baseline_group"]]["tasks"]) == {"piqa", "hellaswag"}, "Both complete tasks required")
    if "original_analysis_metadata" in complete:
        profile = "primary"
        require(plan["counts"]["logical_endpoints"] == len(manifest["records"])
                and plan["counts"]["physical_jobs"] == len(jobs)
                and authority.get("expected_logical_endpoints") == len(manifest["records"])
                and authority.get("expected_inference_jobs") == len(jobs), "Primary authorized full counts differ")
        journal = reader.proof(complete["journal"])
        require(journal.get("status") == "semantic_access_enabled" and journal.get("before_original_analyze_invocation") is True, "Primary first-read journal absent")
        require(journal.get("manifest") == manifest_proof and journal.get("state_proof") == complete["state_proof"], "Primary journal source mismatch")
        state = reader.proof(complete["state_proof"])
        require(state.get("ready") is True and state.get("plan") == plan_proof, "Incomplete primary closure")
        definitions = {d["id"]: d for d in plan["frozen_definitions"]}
        require(set(definitions) == set(jobs) == set(state["selected_completed_records"]), "Incomplete primary physical closure")
        for key, result in state["selected_completed_records"].items():
            definition = definitions[key]
            require(result.get("status") == "completed" and type(result.get("returncode")) is int and result["returncode"] == 0
                    and result.get("missing_outputs") == [] and not any("stop_requested" in k for k in result)
                    and result.get("argv") == definition["argv"]
                    and result.get("definition_sha256") == sha_bytes(json.dumps(definition, sort_keys=True).encode()), "Failed/stopped/changed primary job")
        checked = reader.proof(complete["original_check"])
        require(checked.get("ready") is True and checked.get("all_output_integrity_verified") is True
                and checked.get("logical_endpoints") == len(manifest["records"])
                and checked.get("expected_inference_jobs") == len(jobs)
                and checked.get("manifest_sha256") == manifest_proof["sha256"], "Incomplete original check")
        metadata = reader.proof(complete["original_analysis_metadata"])
        bundle_path = Path(complete["original_analysis_metadata"]["path"]).parent
        attestations = state["attestations"]
        source_maps = [plan["source_sha256"]]
    else:
        profile = "supplemental"
        require(plan["contract"]["mode"] == "behavior" and complete.get("parsed_phase") == plan["contract"]["phase"], "Wrong supplemental parsed phase")
        require(plan["contract"]["logical_endpoints"] == len(manifest["records"])
                and plan["counts"]["logical_endpoints"] == len(manifest["records"])
                and plan["counts"]["queue_jobs"] == len(jobs)
                and len({r["run_id"] for r in manifest["records"]}) == plan["contract"]["phase"]["rows"], "Supplemental complete scope count differs")
        require(complete.get("endpoints") == [{"endpoint": "final_behavior", "logical_conditions": len(manifest["records"])}], "Incomplete supplemental logical phase")
        require(all(r["kind"] == "final" for r in manifest["records"]), "Supplemental behavior is final-only")
        readiness = reader.proof(complete["readiness"])
        journal = reader.proof(complete["first_read_receipt"])
        require(readiness.get("ready") is True and readiness.get("pending_job_ids") == []
                and readiness.get("all_source_baseline_alias_output_integrity_verified") is True
                and readiness.get("plan") == plan_proof
                and readiness.get("closure_job_count") == len(plan["closure_jobs"]), "Incomplete supplemental physical/prior/alias closure")
        require(journal.get("before_first_score_parse") is True and journal.get("complete_phase_source_and_output_SHA_verified") is True
                and journal.get("readiness") == complete["readiness"]
                and str(Path(plan["first_read_receipt_path"]).resolve()) == str(Path(complete["first_read_receipt"]["path"]).resolve()), "Supplemental first-read source mismatch")
        require(authority.get("expected_counts") == plan["counts"] and readiness.get("counts") == plan["counts"], "Supplemental expected counts changed")
        protocol = reader.proof(plan["protocol"])
        source_maps = [protocol["analysis_source_sha256"]]
        if "operational_repair" in complete:
            repair_proof = complete["operational_repair"]
            repair = reader.proof(repair_proof)
            require(authority.get("operational_repair") == repair_proof and repair.get("retained_plan") == plan_proof
                    and repair.get("retained_manifest") == manifest_proof, "Repair/authority implementation mismatch")
            source_maps += [repair["active_source_sha256"], repair["retained_source_sha256"]]
        bundle_path = Path(complete_proof["path"]).parent / "behavior"
        metadata = reader.json(bundle_path / "analysis_manifest.json")
        require(metadata.get("first_read_receipt") == complete["first_read_receipt"] and metadata.get("whole_phase_readiness") == complete["readiness"], "Bundle first-read binding differs")
        attestations = readiness["attestations"]
    require(journal.get("plan") == plan_proof and journal.get("authority") == authority_proof, "Journal plan/ROOT authority mismatch")
    require(metadata.get("complete") is True and metadata.get("manifest_sha256") == manifest_proof["sha256"]
            and metadata.get("logical_endpoints") == len(manifest["records"])
            and metadata.get("primary_metric") == "acc_norm" and metadata.get("auxiliary_metric") == "acc", "Incomplete or wrong analysis metadata")
    for mapping in [manifest["common_input_sha256"], *source_maps]:
        for path, digest in mapping.items(): reader.bytes(path, digest)
    # Guard COMPLETE is the authority for original checkpoint/source validation.
    # Only its attested output closure is fresh-read here; no checkpoint rehash.
    approved = {}
    attested_outputs = {}
    for item in attestations:
        att = reader.proof(item["attestation"])
        require(att.get("verified_frozen_inputs") is True and att.get("job_id") == item["job_id"]
                and att.get("output_sha256") == item["output_sha256"], "Changed execution attestation")
        for path, digest in att["output_sha256"].items():
            require(path not in approved or approved[path] == digest, "Conflicting output receipts")
            reader.bytes(path, digest)
            approved[path] = digest
        for path in att["output_sha256"]:
            if Path(path).suffix == ".json": attested_outputs[path] = att
    expected_outputs = {r["output"] for r in manifest["records"]} | {b["initial_output"] for b in manifest["baseline_groups"].values()}
    require(expected_outputs <= set(approved), "Incomplete logical/baseline output closure")
    for record in manifest["records"]:
        att = attested_outputs[record["output"]]
        require(att.get("manifest_sha256") == manifest_proof["sha256"]
                and att.get("job_id") == record["inference_job_id"]
                and att.get("checkpoint_sha256") == record["checkpoint_sha256"]
                and att.get("weights_identity") == record["weights_identity"]
                and att.get("baseline_group") == record["baseline_group"], "Logical endpoint attestation identity changed")
        job = jobs[record["inference_job_id"]]
        if Path(record["output"]).resolve() != Path(job["output"]).resolve():
            require(att.get("inference_alias") is True and att.get("logical_record_id") == record["id"]
                    and att.get("inference_subprocess_wall_seconds") == 0
                    and att.get("canonical_output") == str(Path(job["output"]).resolve())
                    and att.get("source_model_equivalence") == record.get("source_model_equivalence"), "Unproved/nonzero-forward inference alias")
    for key, group in manifest["baseline_groups"].items():
        att = attested_outputs[group["initial_output"]]
        origin = group.get("initial_origin")
        require(att.get("manifest_sha256") == (origin["manifest_sha256"] if origin else manifest_proof["sha256"])
                and att.get("checkpoint_sha256") is None and att.get("job_id") == group["initial_job_id"]
                and att.get("baseline_group") == key, "Wrong actual matched initial")
    return manifest, plan, metadata, bundle_path, approved, profile


def wilson(k, n):
    if not n: return None, None, None
    z = 1.959963984540054
    den = n + z * z
    middle = (k + z * z / 2) / den
    half = z * math.sqrt(k * (n - k) / n + z * z / 4) / den
    return k / n, max(0., middle - half), min(1., middle + half)


def binary(before, after, metric, stream):
    """Independently derive the four-state measure and bootstrap linear contrast."""
    require(before.keys() == after.keys() and before, "Paired question inventory differs")
    counts = Counter()
    for key in sorted(before):
        a, b = before[key], after[key]
        require((a["fingerprint"], a["gold"]) == (b["fingerprint"], b["gold"]), "Paired fingerprint/gold differs")
        counts[(a[metric], b[metric])] += 1
    n = len(before)
    cells = [counts[1, 1], counts[1, 0], counts[0, 1], counts[0, 0]]
    initially, current = cells[0] + cells[1], cells[0] + cells[2]
    estimate, low, high = wilson(current, n)
    # Same frozen stream/category order, independently evaluated as a linear
    # contrast. This checks its finite Monte Carlo output, not independent RNG.
    draws = np.random.default_rng(stream).multinomial(n, np.asarray(cells) / n, size=5000)
    deltas = draws @ np.asarray([0., -1., 1., 0.]) / n
    ci = np.percentile(deltas, (2.5, 97.5), method="linear")
    return dict(items=n, CC=cells[0], CW=cells[1], WC=cells[2], WW=cells[3], initial_accuracy=initially/n,
                accuracy=estimate, accuracy_ci_low=low, accuracy_ci_high=high, accuracy_delta=(current-initially)/n,
                delta_ci_low=float(ci[0]), delta_ci_high=float(ci[1]), conditional_forgetting_rate=wilson(cells[1], initially)[0],
                conditional_acquisition_rate=wilson(cells[2], n-initially)[0], bootstrap_seed=stream)


def questions(reader, task, info, metrics, approved):
    artifact = info["artifact"]
    prepared = reader.rows(artifact)
    reader.bytes(artifact, info["sha256"])
    raw = reader.rows(metrics["predictions_path"])
    path = metrics["predictions_path"]
    require(approved.get(path) == metrics["predictions_sha256"], "Unattested prediction SHA")
    require(metrics.get("complete_split") is True and metrics["items"] == len(prepared) == len(raw)
            and metrics["data_sha256"] == info["sha256"], "Incomplete task split/items/data")
    expected = {(r["task"], r["item_id"]): r for r in prepared}
    require(expected and len(expected) == len(prepared), "Duplicate prepared questions")
    result = {}
    for row in raw:
        key = (row["task"], row["item_id"])
        require(key in expected and key not in result and row["task"] == task, "Duplicate/foreign question")
        source = expected[key]
        require(all(row[k] == source[k] for k in ("fingerprint", "gold", "choices")), "Question source identity differs")
        scores, choices = row["choice_logprobs"], row["choices"]
        lengths = [len(s) for s in choices]
        require(len(scores) == len(choices) > 0 and all(lengths) and lengths == row["choice_character_lengths"]
                and all(math.isfinite(v) for v in scores) and type(row["gold"]) is int and 0 <= row["gold"] < len(scores), "Invalid scores/gold/character lengths")
        value = {"fingerprint": row["fingerprint"], "gold": row["gold"]}
        for metric in ("acc_norm", "acc"):
            vals = [s/n for s,n in zip(scores,lengths)] if metric == "acc_norm" else scores
            prediction = max(enumerate(vals), key=lambda pair: pair[1])[0]
            correct = int(prediction == row["gold"])
            require(row["prediction_"+metric] == prediction and row["correct_"+metric] == correct, "Stored prediction/correctness differs from independent argmax")
            value[metric] = correct
        result[key] = value
    require(result.keys() == expected.keys(), "Missing complete question endpoint")
    return result


class Checks:
    def __init__(self): self.count, self.maximum_error = 0, 0.

    def equal(self, actual, expected, label):
        if isinstance(expected, (float, int)) and not isinstance(expected, bool):
            require(type(actual) in (int, float) and math.isfinite(actual), "Nonfinite/missing numeric: " + label)
            error = abs(actual-expected)
            self.maximum_error = max(self.maximum_error, error)
            require(error <= 3e-14, "Numerical mismatch: " + label)
        else: require(actual == expected, "Identity/value mismatch: " + label)
        self.count += 1

    def table(self, actual, expected, keys, label):
        def index(rows):
            result = {tuple(r.get(k) for k in keys): r for r in rows}
            require(len(result) == len(rows), "Duplicate rows: " + label)
            return result
        aa, ee = index(actual), index(expected)
        require(aa.keys() == ee.keys(), "Incomplete/extra table: " + label)
        for key, row in ee.items():
            for field, value in row.items(): self.equal(aa[key].get(field), value, label+":"+field)


def mean_sd(values):
    values = [v for v in values if v is not None]
    return dict(mean=statistics.mean(values) if values else None,
                std_ddof1=statistics.stdev(values) if len(values)>1 else None, n_training_seeds=len(values))


def seed_tables(rows):
    cells, indexed = defaultdict(list), {}
    for row in rows:
        cell = tuple(row[k] for k in CELL)
        cells[cell].append(row)
        key = tuple(row[k] for k in PAIR_CELL)+(row["seed"],row["group"])
        require(key not in indexed, "Duplicate behavior seed/cell")
        indexed[key] = row
    aggregates = []
    for cell, current in cells.items():
        require(len(current) == 3 and {r["seed"] for r in current} == {11,22,33}, "Incomplete three-seed behavior cell")
        for quantity in QUANTITIES:
            aggregates.append(dict(zip(CELL,cell), quantity=quantity, **mean_sd(r[quantity] for r in current)))
    pairs = []
    for key, left in indexed.items():
        for a,b in PAIRS:
            if left["group"] != a: continue
            right = indexed.get(key[:-1]+(b,))
            if right is None: continue
            pairs.append({**dict(zip(PAIR_CELL,key[:-2])), "seed":left["seed"], "left_group":a,"right_group":b,
                "left_run_id":left["run_id"],"right_run_id":right["run_id"],
                "left_target_prediction_tokens":left["target_prediction_tokens_at_checkpoint"],
                "right_target_prediction_tokens":right["target_prediction_tokens_at_checkpoint"],
                "accuracy_left_minus_right":left["accuracy"]-right["accuracy"],
                "delta_accuracy_left_minus_right":left["accuracy_delta"]-right["accuracy_delta"]})
    grouped = defaultdict(list)
    pair_keys = (*PAIR_CELL,"left_group","right_group")
    for row in pairs: grouped[tuple(row[k] for k in pair_keys)].append(row)
    pair_stats = [dict(zip(pair_keys,cell), quantity="accuracy_left_minus_right", **mean_sd(r["accuracy_left_minus_right"] for r in current)) for cell,current in grouped.items()]
    return aggregates,pairs,pair_stats


def stream(text):
    return (20261106 + int(hashlib.sha256(text.encode()).hexdigest()[:8],16)) % (2**32)


def review(complete_proof, manifest_proof, output):
    reader, checks = Reader(), Checks()
    manifest, plan, metadata, bundle, approved, profile = verify_gate(reader, complete_proof, manifest_proof)
    # Semantic output/prediction parsing starts only after successful gate.
    baseline, loaded, endpoints = {}, {}, []
    for key, group in manifest["baseline_groups"].items():
        initial = reader.json(group["initial_output"])
        require(initial.get("complete") is True and set(initial["tasks"]) == set(group["tasks"]), "Incomplete initial tasks")
        baseline[key] = {}
        for task, info in group["tasks"].items():
            before = questions(reader,task,info,initial["tasks"][task],approved)
            baseline[key][task] = before
            for metric in ("acc_norm","acc"):
                checks.equal(initial["tasks"][task][metric],sum(r[metric] for r in before.values())/len(before),"initial accuracy")
    for record in manifest["records"]:
        result = reader.json(record["output"])
        require(result.get("complete") is True and set(result["tasks"]) == set(manifest["baseline_groups"][record["baseline_group"]]["tasks"]), "Incomplete adapted endpoint")
        for task, info in manifest["baseline_groups"][record["baseline_group"]]["tasks"].items():
            after = questions(reader,task,info,result["tasks"][task],approved)
            loaded[record["run_id"],record["kind"],task] = after
            for metric in ("acc_norm","acc"):
                stats = binary(baseline[record["baseline_group"]][task],after,metric,stream(f"{record['id']}|{task}|{metric}"))
                checks.equal(result["tasks"][task][metric],stats["accuracy"],"adapted accuracy")
                row = {k:record.get(k) for k in (*STRATA,"run_id","actual_run_id","group","ratio","weight","learning_rate","training_seed","step","checkpoint_sha256","weights_identity","inference_job_id","target_prediction_tokens_at_checkpoint","replay_prediction_tokens_at_checkpoint")}
                row.update(checkpoint_kind=record["kind"],seed=record["training_seed"],task=task,metric=metric,primary_metric=metric=="acc_norm",**stats)
                endpoints.append(row)
    keys = ("run_id","checkpoint_kind","task","metric")
    checks.table(reader.csv(bundle/"endpoints.csv"),endpoints,keys,"endpoints")
    aggregates,pairs,pair_stats = seed_tables(endpoints)
    checks.table(reader.csv(bundle/"seed_aggregates.csv"),aggregates,(*CELL,"quantity"),"seed aggregates")
    checks.table(reader.csv(bundle/"paired_seed_differences.csv"),pairs,(*PAIR_CELL,"seed","left_group","right_group"),"fixed-lambda pairs")
    # Original pair aggregates omit STRATA in their internal aggregation, then
    # restore the same single-phase context. Reviewer groups it explicitly.
    checks.table(reader.csv(bundle/"paired_seed_aggregates.csv"),pair_stats,(*PAIR_CELL,"left_group","right_group","quantity"),"paired SD")
    selected = None
    selected_tables = {}
    if profile == "supplemental" and plan["contract"]["family"] == "A":
        selected, selected_tables = review_selected(reader, checks, Path(complete_proof["path"]).parent/"validation_selected_policy", plan["selection"], endpoints, loaded)
    reader.finish()
    output = Path(output)
    require(not output.exists(), "Use a new immutable review directory")
    output.mkdir(parents=True)
    for name,rows in [("endpoints",endpoints),("seed_aggregates",aggregates),("paired_seed_differences",pairs),("paired_seed_aggregates",pair_stats),*selected_tables.items()]:
        with (output/(name+".csv")).open("x",newline="") as handle:
            if rows:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    output_hashes={str(p.resolve()):sha_bytes(p.read_bytes()) for p in sorted(output.glob("*.csv"))}
    value={"complete":True,"reviewer_source":{"path":str(Path(__file__).resolve()),"sha256":sha_bytes(Path(__file__).read_bytes())},
           "original_COMPLETE":complete_proof,"manifest":manifest_proof,"profile":profile,"logical_endpoints":len(manifest["records"]),
           "manifest_jobs":len(manifest["jobs"]),"endpoint_task_metric_rows":len(endpoints),"numeric_and_identity_checks":checks.count,
           "maximum_absolute_numeric_error":checks.maximum_error,"validation_selected_review":selected,
           "conditional_interval_scope":SCOPE,"bootstrap_review":"Exact frozen stream and CC/CW/WC/WW categories; independent counts/linear contrast/Wilson algebra. This is not an independent Monte Carlo sample.",
           "seed_scope":"Sample SD across the three fixed training seeds, separate final/best. Aliases and questions are not extra seeds; A lambda-selection variability is not resampled.",
           "source_files":reader.sources,"review_output_sha256":output_hashes,"fresh_bytes":sum(v["bytes"] for v in reader.sources.values()),"checkpoint_payload_bytes_read":0,"model_forward_calls":0}
    with (output/"COMPLETE.json").open("xb") as handle: handle.write(encoded(value))
    return value


def review_selected(reader,checks,directory,selection,rows,loaded):
    require(set(selection["selected_ids"]) == {r["run_id"] for r in rows} and len(selection["selected_ids"]) == 18, "Incomplete selected18 join")
    reader.proof(selection["selection"]);reader.proof(selection["source_registry"])
    selected_metadata = reader.json(directory/"analysis_manifest.json")
    require(selected_metadata.get("selection") == selection["selection"]
            and selected_metadata.get("source_registry") == selection["source_registry"]
            and selected_metadata.get("bootstrap_replicates") == 5000
            and selected_metadata.get("bootstrap_seed") == 20261106, "Selected policy metadata source/policy mismatch")
    for row in rows:
        identity=selection["selected_identity"][row["run_id"]]
        require(all(row[k] == identity[k] for k in ("group","weight","ratio","training_seed","actual_run_id")), "Changed validation-selected identity")
    checks.table(reader.csv(directory/"selected_endpoints.csv"),rows,("run_id","checkpoint_kind","task","metric"),"selected endpoints")
    lookup={(r["group"],r["seed"],r["task"],r["metric"]):r for r in rows}
    paired, intervals = [],[]
    for left,right in SELECTED_PAIRS:
        for seed in (11,22,33):
            for task in ("hellaswag","piqa"):
                for metric in ("acc_norm","acc"):
                    a,b=lookup[left,seed,task,metric],lookup[right,seed,task,metric]
                    require(all(a[k] == b[k] for k in (*STRATA,"seed","ratio","checkpoint_kind")), "Selected pair mixes source/budget")
                    identity={"comparison":left+"-"+right,"training_seed":seed,"task":task,"metric":metric,
                              "lambda_a":a["weight"],"lambda_b":b["weight"],"run_a":a["run_id"],"run_b":b["run_id"]}
                    diff=a["accuracy"]-b["accuracy"]
                    paired.append({**identity,"accuracy_left_minus_right":diff})
                    stats=binary(loaded[b["run_id"],b["checkpoint_kind"],task],loaded[a["run_id"],a["checkpoint_kind"],task],metric,stream(f"{left}|{right}|{seed}|{task}|{metric}"))
                    checks.equal(stats["accuracy_delta"],diff,"selected question alignment")
                    intervals.append({**identity,"accuracy_left_minus_right":diff,"ci_low":stats["delta_ci_low"],"ci_high":stats["delta_ci_high"],
                        "both_correct":stats["CC"],"right_correct_left_wrong":stats["CW"],"right_wrong_left_correct":stats["WC"],"both_wrong":stats["WW"],
                        "bootstrap_seed":stats["bootstrap_seed"],"bootstrap_replicates":5000})
    pair_keys=("comparison","training_seed","task","metric")
    checks.table(reader.csv(directory/"selected_policy_pairs.csv"),paired,pair_keys,"selected policy pairs")
    checks.table(reader.csv(directory/"selected_question_bootstrap.csv"),intervals,pair_keys,"selected question intervals")
    grouped=defaultdict(list)
    for row in paired: grouped[tuple(row[k] for k in ("comparison","lambda_a","lambda_b","task","metric"))].append(row)
    aggregate=[]
    resolved=[]
    for cell,items in grouped.items():
        stats=mean_sd(r["accuracy_left_minus_right"] for r in items)
        aggregate.append(dict(zip(("comparison","lambda_a","lambda_b","task"),cell[:-1]),metric="accuracy_left_minus_right",
            mean=stats["mean"],std_ddof1=stats["std_ddof1"],n=stats["n_training_seeds"]))
        resolved.append(dict(zip(("comparison","lambda_a","lambda_b","task","score"),cell),quantity="accuracy_left_minus_right",**stats))
    # The frozen selected-policy writer uses `metric` both for the score (acc or
    # acc_norm) and aggregate quantity; its overwrite produces duplicate keys.
    # Preserve score-specific verification by matching complete row multisets.
    actual=reader.csv(directory/"selected_policy_statistics.csv")
    fields=("comparison","lambda_a","lambda_b","task","metric","mean","std_ddof1","n")
    require(len(actual)==len(aggregate),"Missing selected policy aggregates")
    remaining=list(actual)
    for expected in aggregate:
        found=next((i for i,r in enumerate(remaining) if all((r.get(k)==v if not isinstance(v,(float,int)) else isinstance(r.get(k),(float,int)) and abs(r[k]-v)<=3e-14) for k,v in expected.items())),None)
        require(found is not None,"Selected policy aggregate multiset mismatch")
        for k in fields: checks.equal(remaining[found].get(k),expected[k],"selected aggregate")
        remaining.pop(found)
    return ({"rows":len(rows),"pairs":len(paired),"question_intervals":len(intervals),
            "policy":"Each method's own frozen validation-selected lambda; not a fixed-lambda objective contrast.",
            "aggregate_schema_caveat":"Original selected_policy_statistics.csv overwrites score metric with quantity; reviewer verifies its row multiset and reports score-resolved pairs separately. Do not infer score identity from row label alone."},
            {"selected_policy_pairs":paired,"selected_question_intervals":intervals,"selected_policy_score_resolved_statistics":resolved})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete",type=Path,required=True);parser.add_argument("--complete-sha256",required=True)
    parser.add_argument("--manifest",type=Path,required=True);parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--child",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.child:
        result=review({"path":str(args.complete.resolve()),"sha256":args.complete_sha256},{"path":str(args.manifest.resolve()),"sha256":args.manifest_sha256},args.output)
        print(json.dumps({k:result[k] for k in ("complete","endpoint_task_metric_rows","numeric_and_identity_checks","maximum_absolute_numeric_error")}));return
    require(not args.output.exists(),"New output directory required")
    timer=args.output.with_name(args.output.name+"_whole_child_timing.json")
    require(not timer.exists(),"Timing namespace already exists")
    argv=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--child"]
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2"}
    before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.perf_counter()
    child=subprocess.run(argv,cwd=ROOT,env=env)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    timer.parent.mkdir(parents=True,exist_ok=True)
    with timer.open("xb") as handle:handle.write(encoded({"argv":argv,"returncode":child.returncode,"wall_seconds":time.perf_counter()-started,
        "user_cpu_seconds":after.ru_utime-before.ru_utime,"system_cpu_seconds":after.ru_stime-before.ru_stime,
        "scope":"Entire review child including gate, source/output fresh hashes and independent numerical checks; no original check/analyze or inference.",
        "nested_subtimers_not_additive":True,"prelaunch_parent_checks_excluded":True}))
    raise SystemExit(child.returncode)


if __name__=="__main__":main()
