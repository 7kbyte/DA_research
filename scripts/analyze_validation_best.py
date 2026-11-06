#!/usr/bin/env python3
"""Describe validation-selected checkpoints without calling them equal-budget tests."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_study.analysis import (aggregate_endpoints, collect_analysis, endpoint_metrics,
    paired_seed_differences, read_json, selected_checkpoint_exposure, sha256, write_csv)


def best_validation_tables(bundle, baseline_path):
    meta = bundle["metadata"]
    if not meta["complete"] or meta["analysis_status"] != "active" or meta["split"] != "validation":
        raise ValueError("best validation description requires a complete verified active validation bundle")
    sources = meta["sources_sha256"]
    baseline_path = Path(baseline_path)
    expected_baseline = [value for path, value in sources.items() if Path(path).resolve() == baseline_path.resolve()]
    if not expected_baseline or sha256(baseline_path) not in expected_baseline:
        raise ValueError("baseline changed after final-endpoint verification")
    baseline = read_json(baseline_path)["corpora"]
    endpoints = []
    for entry in bundle["registry"]:
        path = Path(entry["summary_path"])
        if sources.get(str(path)) != sha256(path):
            raise ValueError("summary changed after final-endpoint verification")
        summary = read_json(path)
        exposure = selected_checkpoint_exposure(summary, "best")
        selected = [p for p in summary["validation_trajectory"] if p["step"] == exposure["checkpoint_step"] and
                    p["target_prediction_tokens"] == exposure["target_prediction_tokens"]]
        if len(selected) != 1:
            raise ValueError("best record does not match one actual validation checkpoint")
        endpoints.append({"run_id": entry["id"], "group": entry["group"], "seed": entry["seed"],
                          "ratio": entry["ratio"], "weight": entry["weight"], "split": "validation",
                          "checkpoint": "best_by_target_validation", "selection_uses_same_validation": True,
                          "equal_full_budget_endpoint": exposure["target_prediction_tokens"] == exposure["full_training_target_prediction_tokens"],
                          "full_training_cost_remains_charged": True, **exposure,
                          **endpoint_metrics(selected[0]["metrics"], baseline)})
    pairs, paired_stats = paired_seed_differences(endpoints, bundle["registry"])
    by_id = {r["run_id"]: r for r in endpoints}
    for pair in pairs:
        a, b = by_id[pair["run_a"]], by_id[pair["run_b"]]
        pair.update(target_exposure_a=a["target_prediction_tokens"], target_exposure_b=b["target_prediction_tokens"],
                    replay_exposure_a=a["replay_prediction_tokens"], replay_exposure_b=b["replay_prediction_tokens"],
                    equal_checkpoint_target_exposure=a["target_prediction_tokens"] == b["target_prediction_tokens"],
                    source="target-validation-selected diagnostic; not independent test")
    return {"best_validation_endpoints": endpoints,
            "best_validation_statistics": aggregate_endpoints(endpoints, bundle["registry"]),
            "best_validation_paired": pairs, "best_validation_paired_statistics": paired_stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--protocol-status", default="docs/protocol_status.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle = collect_analysis(args.registry, args.baseline, analysis_status="active",
        protocol_status_path=args.protocol_status, require_execution_provenance=True)
    tables = best_validation_tables(bundle, args.baseline)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    manifest = {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "complete": True, "endpoint_count": len(tables["best_validation_endpoints"]),
        "split": "validation", "checkpoint": "best_by_target_validation",
        "endpoint_policy": "actual saved best checkpoint exposure; full training cost charged separately",
        "not_an_equal_budget_comparison": True, "selection_uses_same_validation": True,
        "test_scores_accessed": False, "model_inference_performed": False,
        "script_sha256": sha256(__file__), "analysis_module_sha256": sha256(ROOT / "src/replay_study/analysis.py"),
        "sources_sha256": bundle["metadata"]["sources_sha256"]}
    (output / "best_validation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "best验证端点说明.md").write_text(
        "本表从已通过完整正式审计的训练摘要中，提取best_target_validation匹配的真实验证轨迹点。"
        "每模型记录实际step/目标和回放曝光，完整训练支出另列，不把best标成10M。"
        "best按目标验证NLL选择，当前结果又在同一验证集描述，具有选择乐观性；不能当独立测试或等曝光主比较。"
        "三seed均值/SD仍以训练seed为单位，配对表明确两模型的实际曝光。补充A的λ选择只用final固定10M端点，不使用此表。\n")
    print(json.dumps({"complete": True, "endpoints": manifest["endpoint_count"], "test_scores_accessed": False}))


if __name__ == "__main__":
    main()
