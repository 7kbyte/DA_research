#!/usr/bin/env python3
"""Count frozen behavioral candidate tokens on CPU; never load model weights or score."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation_plan import encoded, sha


def describe(data_root, tokenizer_path, revision, max_length=2048, batches=(8, 16)):
    from transformers import AutoTokenizer
    from replay_study.tasks import prepare_choice
    from make_behavioral_queue import task_inputs
    inputs, tasks = task_inputs(data_root)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=revision, local_files_only=True)
    tokenizer_files = [p for p in Path(tokenizer_path).glob("*") if p.name in {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"}]
    if not tokenizer_files:
        raise ValueError("Pinned local tokenizer assets required")
    details = {}
    for name, meta in tasks.items():
        rows = [json.loads(line) for line in Path(meta["artifact"]).open() if line.strip()]
        lengths, continuations, truncated, mismatches = [], 0, 0, 0
        for row in rows:
            for choice in row["choices"]:
                item = prepare_choice(tokenizer, row["context"], choice, max_length)
                lengths.append(len(item["input_ids"]))
                continuations += item["continuation_tokens"]
                truncated += int(item["context_tokens_removed"] > 0)
                mismatches += int(not item["boundary_prefix_matches"])
        lengths.sort(reverse=True)  # exactly the frozen scorer's per-task length sorting
        details[name] = {"questions": len(rows), "choices": len(lengths),
            "scored_continuation_prediction_tokens": continuations,
            "unpadded_forward_input_positions": sum(lengths), "maximum_forward_input_length": max(lengths),
            "choices_with_context_truncation": truncated, "boundary_prefix_mismatch_choices": mismatches,
            "by_batch": {str(batch): {"forward_calls": (len(lengths) + batch - 1) // batch,
                "padded_forward_input_positions": sum(lengths[i] * len(lengths[i:i + batch]) for i in range(0, len(lengths), batch)),
                "padded_sequence_length_squared_sum": sum(lengths[i] ** 2 * len(lengths[i:i + batch]) for i in range(0, len(lengths), batch))}
                for batch in batches}}
    model_scopes = {
        "pythia-410m": {"primary_checkpoint_endpoints": 72, "A_selected_final_maximum": 18,
                         "extended_final": 24, "initial": 1, "proposed_behavior_batch": 16},
        "pythia-1b": {"extended_final": 24, "initial": 1, "proposed_behavior_batch": 16},
        "pythia-2.8b": {"extended_final": 15, "initial": 1, "proposed_behavior_batch": 8}}
    provenance_paths = [ROOT / "artifacts/assets/model_provenance.json", ROOT / "artifacts/assets/extended_model_provenance.json"]
    parameter_counts = {}
    for path in provenance_paths:
        source = json.loads(path.read_text())
        for entry in source.get("models", [source]):
            parameter_counts[entry["repo"].split("/")[-1]] = entry.get("parameter_count", entry.get("trainable_parameter_count"))
    # Work counts do not predict wall time. No model kernel has been measured here.
    for model, scope in model_scopes.items():
        scope["maximum_inference_passes_before_equivalence_reuse"] = sum(scope.get(k, 0) for k in
            ("primary_checkpoint_endpoints", "A_selected_final_maximum", "extended_final", "initial"))
        batch = str(scope["proposed_behavior_batch"])
        one = {"questions": sum(x["questions"] for x in details.values()),
               "choices": sum(x["choices"] for x in details.values()),
               "scored_continuation_prediction_tokens": sum(x["scored_continuation_prediction_tokens"] for x in details.values()),
               "padded_forward_input_positions": sum(x["by_batch"][batch]["padded_forward_input_positions"] for x in details.values()),
               "forward_calls": sum(x["by_batch"][batch]["forward_calls"] for x in details.values())}
        scope["work_per_pass"] = one
        scope["maximum_total_work"] = {k: v * scope["maximum_inference_passes_before_equivalence_reuse"] for k, v in one.items()}
        scope["parameter_count"] = parameter_counts[model]
        scope["structural_parameter_input_positions"] = parameter_counts[model] * scope["maximum_total_work"]["padded_forward_input_positions"]
        config_path = Path(tokenizer_path).parent / f"{model}-final/config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            scope["architecture"] = {k: config[k] for k in
                ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "vocab_size")}
            scope["architecture_config_sha256"] = sha(config_path)
    count = sum(x["maximum_inference_passes_before_equivalence_reuse"] for x in model_scopes.values())
    window = 3 * 3600 + 26 * 60 + 32
    return {"schema_version": 1, "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root": str(data_root), "source_sha256": inputs,
        "tokenizer": str(tokenizer_path), "revision": revision,
        "tokenizer_asset_sha256": {str(p): sha(p) for p in tokenizer_files},
        "model_provenance_sha256": {str(p): sha(p) for p in provenance_paths},
        "counting_module_sha256": sha(ROOT / "src/replay_study/tasks.py"),
        "counting_script_sha256": sha(__file__), "max_length": max_length,
        "tasks": details, "scope": model_scopes, "maximum_inference_passes": count,
        "supplement_B_behavior": False, "extended_best_behavior": False,
        "batch_proposal_requires_root_freeze": True,
        "timing_measured": False, "model_weights_loaded": False, "model_inference_performed": False,
        "time_risk": {"remaining_after_training_deadline_seconds": window,
            "four_gpu_full_window_average_seconds_per_pass_upper_bound": 4 * window / count,
            "note": "Necessary mean allowance if all four GPUs were dedicated to behavior; excludes NLL evaluation, CPU SHA, packaging, contention and failures. This is not a throughput estimate."},
        "cost_formula": "sum_model(N_distinct_passes_model * measured_full_task_seconds_model_batch_policy); use exact padded input positions and architecture for structural comparisons, never convert parameter counts into seconds without measurement",
        "reuse_rule": "Same actual inference inputs/policy plus exact checkpoint bytes or SHA-bound full-model_state equivalence. Preserve all logical endpoints and physical training identities."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "artifacts/behavioral")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new budget snapshot output")
    value = describe(args.data_root.resolve(), args.tokenizer.resolve(), args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded(value))
    print(json.dumps({"maximum_inference_passes": value["maximum_inference_passes"],
                      "output": str(args.output), "model_inference": False}))


if __name__ == "__main__":
    main()
