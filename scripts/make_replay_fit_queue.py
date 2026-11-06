#!/usr/bin/env python3
"""CPU-only queue freeze for post-validation exploratory bank/complement fit."""
import argparse
import hashlib
import sys
from pathlib import Path
from evaluation_plan import ROOT, Hashes, absolute, command, encoded, endpoint, identity, outputs, read, training_execution

PHASE = "post_validation_exploratory_replay_fit"


def validate_partitions(data):
    if len(data["partitions"]) != 3 or {p["seed"] for p in data["partitions"]} != {11, 22, 33}:
        raise ValueError("Exactly three frozen seed partitions required")
    for p in data["partitions"]:
        if set(p["corpora"]) != {"bank", "complement"}: raise ValueError("Bank/complement required")
        a, b = [p["corpora"][k]["pool_indices"] for k in ("bank", "complement")]
        if (len(a) != 196 or len(b) != 1761 or len(set(a)) != len(a) or len(set(b)) != len(b)
                or set(a) & set(b) or set(a) | set(b) != set(range(1957))
                or p.get("disjoint_pool_indices") is not True or p.get("exhaustive_pool_cover") is not True
                or p.get("cross_partition_exact_token_duplicates") != 0):
            raise ValueError("Unique disjoint exhaustive partitions with zero cross duplicates required")
        for c in p["corpora"].values():
            if c["chunks"] != len(c["pool_indices"]) or c["prediction_tokens"] != c["chunks"] * 511:
                raise ValueError("Causal prediction-token accounting mismatch")


def build(data_path, data_sha, primary_path, primary_sha, output_root, manifest_path):
    h = Hashes(); data_path, _ = h.proof({"path": data_path, "sha256": data_sha})
    primary_path, _ = h.proof({"path": primary_path, "sha256": primary_sha})
    data, primary = read(data_path), read(primary_path)
    pp = str(Path(data_path).with_name("protocol.json")); h.proof({"path": pp, "sha256": data["protocol_sha256"]}); protocol = read(pp)
    if data.get("phase") != PHASE or data.get("active") is not False or protocol.get("state") != "prepared_only_not_scheduled":
        raise ValueError("Explicit prepared-only exploratory protocol required")
    validate_partitions(data)
    binding = protocol["checkpoint_sha_binding_source"]; oldpath, oldsha = h.proof(binding)
    common = {data_path: data_sha, primary_path: primary_sha, pp: data["protocol_sha256"], oldpath: oldsha,
              str(Path(__file__).resolve()): h.digest(__file__)}
    for path, value in primary["scorer_input_sha256"].items(): h.proof({"path": path, "sha256": value}); common[path] = value
    cases = data["evaluation_cases"]; adapted = [c for c in cases if c["kind"] != "initial"]
    if len(cases) != 15 or len(adapted) != 12 or {(c["seed"], c["group"], c["kind"]) for c in adapted} != {(s,g,k) for s in (11,22,33) for g in ("B3","B4") for k in ("final","best")}:
        raise ValueError("Exactly three initial and twelve B3/B4 final/best cases required")
    sources = {(r["run_id"], r["kind"]): r for r in primary["records"]}; groups, jobs, records = {}, [], []
    output_root, manifest_path = absolute(output_root), absolute(manifest_path)
    for part in data["partitions"]:
        seed = part["seed"]; selected = [c for c in cases if c["seed"] == seed]
        if len([c for c in selected if c["kind"] == "initial"]) != 1 or any(c["corpora"] != part["corpora"] for c in selected):
            raise ValueError("One initial and identical ordered corpus identities per seed required")
        paths, inputs = {}, {}
        for key in ("source_pool", "source_bank"):
            path, value = h.proof({"path": part[key], "sha256": part[key+"_sha256"]}); inputs[path] = value
            meta = str(Path(path).with_suffix(".json")); _, value = h.proof({"path": meta, "sha256": part[key+"_manifest_sha256"]}); inputs[meta] = value
        for name, corpus in part["corpora"].items():
            path, value = h.proof(corpus)
            if Path(path).parent != Path(data_path).parent: raise ValueError("Wrong diagnostic data family")
            paths[name] = path; inputs[path] = value
            meta = Path(path).with_suffix(".json")
            if meta.exists(): inputs[str(meta)] = h.digest(meta)
        first = next(c for c in selected if c["kind"] != "initial"); source = sources[(first["run_id"], first["kind"])]
        source_group = primary["baseline_groups"][source["baseline_group"]]; policy = source_group["inference_policy"]; assets = source_group["base_model_assets_sha256"]
        for case in selected:
            if case["model"] != policy["model"] or case["model_revision"] != policy["revision"]:
                raise ValueError("Case model/revision differs from frozen source")
            if case["kind"] == "initial" and (case.get("checkpoint") is not None or case.get("checkpoint_target_prediction_tokens") != 0 or case.get("checkpoint_replay_prediction_tokens") != 0):
                raise ValueError("Initial case must be unadapted at zero exposure")
        for path, value in assets.items(): h.proof({"path": path, "sha256": value})
        key = identity({"policy": policy, "model_assets": assets, "ordered_corpora": {n:c["sha256"] for n,c in part["corpora"].items()}, "scorer": primary["scorer_input_sha256"]})
        initial_id = f"replay_fit_v1_seed{seed}_initial"; output = str(output_root/f"seed{seed}"/"initial.json")
        groups[key] = {"id": key, "model_tag": source["model_tag"], "inference_policy": policy, "base_model_assets_sha256": assets, "test_paths": paths, "tasks": {}, "seed": seed, "initial_job_id": initial_id, "initial_output": output}
        jobs.append({"id": initial_id, "kind": "initial", "baseline_group": key, "output": output, "input_sha256": {**assets,**inputs}, "record_ids": [], "argv": command(policy,"nll",None,paths)+["--output",output]})
        for case in (c for c in selected if c["kind"] != "initial"):
            source = sources[(case["run_id"],case["kind"])]; cfg, summary = read(source["config_path"]), read(source["summary_path"])
            actual = {"model": str(absolute(cfg["model_name_or_path"])), "revision": cfg["model_revision"], "batch_size": cfg["evaluation_batch_size"], "max_length":512,"device":"cuda:0",**training_execution(cfg,summary)}
            if actual != policy or summary["status"] != "completed" or cfg["seed"] != seed or cfg["replay_ratio"] != .05 or absolute(cfg["replay_path"]) != absolute(part["source_bank"]) or cfg["replay_objective"] != ("ce" if case["group"] == "B3" else "kl"):
                raise ValueError("Completed frozen B3/B4 source and actual execution mismatch")
            cp, exposure = endpoint(summary,case["kind"]); ex = case["exposure"]
            if cp != str(absolute(case["checkpoint"])) or cp != str(absolute(source["checkpoint"])) or case["checkpoint_sha256"] != source["checkpoint_sha256"] or (ex["checkpoint_step"],ex["target_prediction_tokens"],ex["replay_prediction_tokens"]) != (exposure["step"],exposure["target_prediction_tokens_at_checkpoint"],exposure["replay_prediction_tokens_at_checkpoint"]):
                raise ValueError("Current/historical checkpoint identity or actual exposure mismatch")
            if case["source_config_sha256"] != source["input_sha256"][source["config_path"]] or case["source_summary_sha256"] != source["input_sha256"][source["summary_path"]]:
                raise ValueError("Data protocol and frozen primary config/summary proofs differ")
            job_inputs = {**assets,**inputs,cp:case["checkpoint_sha256"],source["config_path"]:case["source_config_sha256"],source["summary_path"]:case["source_summary_sha256"]}
            for path, value in job_inputs.items(): h.proof({"path":path,"sha256":value})
            output = str(output_root/f"seed{seed}"/f"{case['group']}_{case['kind']}.json"); jid = f"replay_fit_v1_{case['group']}_seed{seed}_{case['kind']}"
            record = {**{k:source[k] for k in ("run_id","actual_run_id","model_tag","group","ratio","weight","learning_rate","checkpoint","checkpoint_sha256","checkpoint_bytes","config_path","summary_path","full_training_committed_path","is_training_alias","selection_seed")}, "id":jid,"kind":case["kind"],"phase":PHASE,"phase_name":PHASE,"role":"exploratory_diagnostic","source_training_phase":"formal","data_family":"same_pool_bank_complement","training_seed":seed,"baseline_group":key,"initial_job_id":initial_id,"inference_job_id":jid,"output":output,"weights_identity":"checkpoint:"+case["checkpoint_sha256"],"source_model_equivalence":None,"inference_alias_of":None,**exposure}
            records.append(record); jobs.append({"id":jid,"kind":"adapted","baseline_group":key,"checkpoint":cp,"checkpoint_sha256":case["checkpoint_sha256"],"weights_identity":record["weights_identity"],"output":output,"record_ids":[jid],"input_sha256":job_inputs,"depends_on":[initial_id],"argv":command(policy,"nll",None,paths)+["--checkpoint",cp,"--output",output]})
    manifest = {"schema_version":2,"mode":"nll","phase":PHASE,"role":"exploratory_diagnostic","prepared_only":True,"launch_authorized":False,"primary_evaluation":False,"independent_test":False,"specified_after_main_validation":True,"common_input_sha256":common,"baseline_groups":groups,"jobs":jobs,"records":records,"source_protocol":pp,"source_protocol_sha256":data["protocol_sha256"],"current_execution_authority":primary_path,"historical_checkpoint_SHA_source_not_execution_authority":binding,"contrasts":{k:protocol[k] for k in ("primary_contrast","secondary_metrics","direction","independence_limits","selection_policy")},"workload":{"initial_inferences":3,"adapted_inferences":12,"total_inference_jobs":15,"prediction_tokens_per_case":1000027,"single_forward_prediction_token_equivalent":27000729,"teacher_timing_nested_in_parent_wall":True},"model_inference_performed_by_generator":False}
    digest = hashlib.sha256(encoded(manifest)).hexdigest()
    queue = [{"id":j["id"],"argv":[sys.executable,str(ROOT/"scripts/run_frozen_evaluation.py"),"--manifest",str(manifest_path),"--manifest-sha256",digest,"--job-id",j["id"],"--device","cuda:0"],"env":{"CUBLAS_WORKSPACE_CONFIG":":4096:8"},"expected_seconds":180,"outputs":outputs(j["output"],"nll"),**({"depends_on":j["depends_on"]} if "depends_on" in j else {})} for j in jobs]
    return queue, manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("data-manifest","primary-manifest","output-root","queue","manifest"):p.add_argument("--"+name,type=Path,required=True)
    for name in ("data-sha256","primary-sha256"):p.add_argument("--"+name,required=True)
    a=p.parse_args()
    if a.queue.exists() or a.manifest.exists() or a.queue.resolve()==a.manifest.resolve():raise FileExistsError("New distinct output paths required")
    q,m=build(a.data_manifest,a.data_sha256,a.primary_manifest,a.primary_sha256,a.output_root,a.manifest)
    from schedule_jobs import validate_queue
    validate_queue(q)
    for path,value in ((a.manifest,m),(a.queue,q)):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(value))
    print(f"Prepared {len(q)} exploratory jobs on CPU; inference not started")


if __name__=="__main__":main()
