#!/usr/bin/env python3
"""CPU-only snapshot of root-fixed NLL endpoint scope and timing-only work budget."""
import argparse
import datetime as dt
import json
from pathlib import Path
import statistics
from evaluation_plan import ROOT, Hashes, absolute, encoded, read, training_execution


def make(scope_path):
    h = Hashes()
    primary_path = ROOT / "docs/primary_evaluation_v2_execution_matched_nll_manifest.json"
    primary = read(primary_path)
    sources = {str(primary_path): h.digest(primary_path), str(Path(__file__).resolve()): h.digest(__file__)}
    phases = [
        {"phase": "primary", "model": "pythia-410m", "data_family": "main_v3", "training_cells": 36, "endpoints": ["final", "best"], "logical_endpoints": 72, "proven_forward_aliases": 6, "additional_adapted_inferences": 66},
        {"phase": "supplement_A", "model": "pythia-410m", "data_family": "main_v3", "training_cells": 126, "endpoints": ["final"], "logical_endpoints": 126, "registered_primary_aliases": 18, "additional_adapted_inferences": 108},
        {"phase": "supplement_B", "model": "pythia-410m", "data_family": "main_v3", "training_cells": 60, "endpoints": ["final"], "logical_endpoints": 60, "additional_adapted_inferences": 60},
    ]
    for model, runs in (("pythia-410m", 24), ("pythia-1b", 24), ("pythia-2.8b", 15)):
        phases.append({"phase": "supplement_C", "model": model, "data_family": "extended_v1", "training_cells": runs,
                       "endpoints": ["final", "best"], "logical_endpoints": 2*runs, "additional_adapted_inferences": 2*runs,
                       "formal_completion_claim": False, "future_full_state_equivalence_not_yet_deducted": True})
    for family, file, n in (("A", "configs/supplemental_v1_A_registry.json", 126), ("B", "configs/supplemental_v1/activated/B_conditional_bank_variance/executor_registry.json", 60)):
        path = ROOT/file; rows = read(path)
        if len(rows) != n: raise ValueError("Registered NLL cell count differs from root-fixed scope")
        if family == "A" and sum(r.get("execution_kind") == "alias" and r.get("counted_as_new_training") is False for r in rows) != 18:
            raise ValueError("A must retain exactly eighteen logical primary aliases")
        sources[str(path)] = h.digest(path)
    groups = []
    for model, family, namespace, micro in (("pythia-410m", "main_v3", None, None), ("pythia-410m", "extended_v1", "pythia_410m", 64), ("pythia-1b", "extended_v1", "pythia_1b", 32), ("pythia-2.8b", "extended_v1", "pythia_2p8b", 8)):
        if family == "main_v3":
            base = next(iter(primary["baseline_groups"].values()))
            policy = base["inference_policy"]
            rows = [r for r in primary["records"] if r["group"] == "B4" and r["kind"] == "final"]
            test_paths = base["test_paths"]
        else:
            path = ROOT/f"configs/extended_v2_{namespace}_m{micro}_lr_registry.json"
            rows = read(path); sources[str(path)] = h.digest(path)
            cfg, summary = read(rows[0]["config_path"]), read(rows[0]["summary_path"])
            policy = {"model": str(absolute(cfg["model_name_or_path"])), "revision": cfg["model_revision"],
                      "batch_size": cfg["evaluation_batch_size"], "max_length": 512, "device": "cuda:0", **training_execution(cfg, summary)}
            test_paths = {k: str(ROOT/f"artifacts/data_extended/{name}_test.pt") for k, name in (("target", "finemath"), ("c4", "c4"), ("wikitext", "wikitext103"))}
        timing_samples = []
        for row in rows:
            cfg, summary = read(row["config_path"]), read(row["summary_path"])
            if summary["status"] != "completed" or any(policy[k] != v for k,v in training_execution(cfg,summary).items()) or cfg["evaluation_batch_size"] != policy["batch_size"]:
                raise ValueError("Only completed same-policy validation timing is eligible")
            for source in (row["config_path"], row["summary_path"]): sources[str(absolute(source))] = h.digest(source)
            for point in summary["validation_trajectory"]:
                metrics = point["metrics"]
                if set(metrics) != {"target", "c4", "wikitext"} or any(c.get("complete_corpus") is not True for c in metrics.values()): continue
                tokens = sum(c["valid_prediction_tokens"] for c in metrics.values())
                wall = sum(c["wall_seconds"] for c in metrics.values())
                timing_samples.append({"summary_path": str(absolute(row["summary_path"])), "step": point["step"],
                                       "valid_prediction_tokens": tokens, "corpus_wall_seconds_sum": wall,
                                       "teacher_forward_seconds_nested": sum(c.get("teacher_forward_seconds",0.) for c in metrics.values()),
                                       "seconds_per_prediction_token": wall/tokens})
        if not timing_samples: raise ValueError("No completed validation timing samples")
        corpora = {}
        for key, path in test_paths.items():
            metadata = Path(path).with_suffix(".json"); meta = read(metadata)
            sources[str(metadata)] = h.digest(metadata); sources[str(absolute(path))] = h.digest(path)
            if meta["sequence_length"] != 512 or meta["prediction_tokens"] != meta["chunks"]*511:
                raise ValueError("Fixed-length causal prediction-token accounting required")
            corpora[key] = {"path": str(absolute(path)), "sha256": sources[str(absolute(path))], "chunks": meta["chunks"], "prediction_tokens": meta["prediction_tokens"]}
        pred = sum(c["prediction_tokens"] for c in corpora.values())
        adapted = sum(p["additional_adapted_inferences"] for p in phases if p["model"] == model and p["data_family"] == family)
        rates = [s["seconds_per_prediction_token"] for s in timing_samples]
        median_rate = statistics.median(rates)
        groups.append({"model": model, "data_family": family, "inference_policy": policy, "corpora": corpora,
                       "adapted_inferences_before_further_equivalence_proofs": adapted, "initial_inferences": 1,
                       "prediction_tokens_per_inference": pred, "student_prediction_tokens": (adapted+1)*pred,
                       "teacher_additional_prediction_tokens": adapted*pred,
                       "initial_teacher_logits_reused": True,
                       "timing_proxy": {"source": "completed validation only; no NLL test results read", "samples": timing_samples,
                                        "median_seconds_per_prediction_token": median_rate, "sample_min_max": [min(rates),max(rates)],
                                        "scorer_corpus_loops_seconds_per_endpoint_proxy": median_rate*pred,
                                        "all_group_corpus_loops_seconds_proxy": (adapted+1)*median_rate*pred,
                                        "initial_proxy_uses_adapted_validation_cost": True}})
    logical = sum(p["logical_endpoints"] for p in phases); adapted = sum(p["additional_adapted_inferences"] for p in phases)
    scope = {"schema_version": 1, "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "state": "root_fixed_endpoint_scope_before_first_real_NLL_test", "authority": "Root explicit endpoint scope amendment; no NLL test score observed",
             "prepared_only": True, "inference_launch_authorized_by_this_file": False, "NLL_test_scores_seen": False,
             "prior_evidence_untouched": True, "phases": phases, "logical_endpoints": logical,
             "baseline_rule": "Unique model + heldout tensor SHA + actual inference execution policy; main_v3 and extended_v1 remain separate",
             "planned_initial_baselines": len(groups), "adapted_forward_count_after_only_current_proofs": adapted,
             "total_inference_jobs_before_further_proven_aliases": adapted+len(groups),
             "checkpoint_selection": "Final at actual prescribed exposure; best from target validation only, preserving step/target/replay exposure",
             "physical_training_cost_rule": "Inference aliases never remove actual physical training costs; best exposure never substitutes for complete training cost",
             "C_2p8b_duration": "Root permits complete fifteen-run 25M fallback instead of 50M before first formal, solely from timing; training_engine separate amendment required. Endpoint/seed/group counts unchanged here.",
             "baseline_groups": [{k:v for k,v in g.items() if k != "timing_proxy"} for g in groups], "source_sha256": sources,
             "behavior_scope_unchanged": "configs/behavioral_scope_execution_freeze_v2.json", "new_inference_started": False}
    total = sum(g["timing_proxy"]["all_group_corpus_loops_seconds_proxy"] for g in groups)
    budget = {"schema_version": 1, "scope_path": str(absolute(scope_path)), "source_sha256": sources,
              "logical_endpoints": logical, "unique_adapted_forward_jobs_before_further_proofs": adapted,
              "planned_initial_jobs": len(groups), "all_NLL_inference_jobs_before_further_proofs": adapted+len(groups),
              "groups": groups, "student_prediction_tokens_total": sum(g["student_prediction_tokens"] for g in groups),
              "additional_teacher_prediction_tokens_total": sum(g["teacher_additional_prediction_tokens"] for g in groups),
              "total_model_forward_prediction_token_equivalent": sum(g["student_prediction_tokens"]+g["teacher_additional_prediction_tokens"] for g in groups),
              "summed_validation_rate_corpus_loop_seconds_proxy": total,
              "sensitivity_scenarios": [{"unmeasured_test_vs_validation_rate_factor": f, "aggregate_corpus_loop_seconds_proxy": total*f,
                                          "ideal_four_uncontended_slots_seconds_proxy": total*f/4} for f in (.75,1.,1.5)],
              "proxy_limitations": ["Validation loop wall includes teacher/KL arithmetic; teacher child timings are nested and never added again", "Initial reuses logits; using adapted validation cost for it is a conservative proxy, not a measured bound", "Exclude scorer/model/checkpoint startup, per-job CPU SHA, scheduling and alias materialization; these need separate wall allowances", "Dataset-size scaling assumes identical shapes/policy and does not identify pure GPU time", "Four-slot arithmetic excludes ongoing training contention; scenarios are not confidence intervals or runtime guarantees", "Further complete model_state proofs may reduce forward count; no unproven C duplicates deducted"],
              "combine_with_behavior_budget": "results/behavioral_timing_budget_v1.json; replace large-model timing assumptions only with score-blind measured attestations",
              "replay_fit_separate": {"manifest": "docs/replay_fit_diagnostic_v2_manifest.json", "inferences": 15, "single_forward_prediction_token_equivalent": 27000729},
              "test_or_behavior_scores_parsed": False, "new_inference_started": False}
    return scope, budget


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--scope",type=Path,required=True);p.add_argument("--budget",type=Path,required=True);a=p.parse_args()
    if a.scope.exists() or a.budget.exists() or a.scope.resolve()==a.budget.resolve(): raise FileExistsError("Use new distinct snapshot paths")
    scope,budget=make(a.scope)
    for path,value in ((a.scope,scope),(a.budget,budget)): path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(value))
    print(json.dumps({"logical_endpoints":scope["logical_endpoints"],"NLL_inference_jobs_before_further_proofs":scope["total_inference_jobs_before_further_proven_aliases"],"scope_sha256":Hashes().digest(a.scope),"budget_sha256":Hashes().digest(a.budget)}))


if __name__=="__main__": main()
