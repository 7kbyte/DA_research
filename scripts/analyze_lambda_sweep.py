#!/usr/bin/env python3
"""Analyze supplement A final validation only; aliases keep actual run IDs."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_study.lambda_analysis import collect_lambda_analysis, write_lambda_analysis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=ROOT / "configs/lambda_sweep_deterministic_registry.json")
    p.add_argument("--preregistered-registry", type=Path, default=ROOT / "configs/lambda_sweep_deterministic_registry.json")
    p.add_argument("--preregistration", type=Path, default=ROOT / "docs/损失权重敏感性预注册_lambda_sweep_deterministic.json")
    p.add_argument("--parent-registry", type=Path, default=ROOT / "configs/formal_deterministic_registry.json")
    p.add_argument("--protocol-status", type=Path, default=ROOT / "docs/protocol_status.json")
    p.add_argument("--baseline", type=Path, default=ROOT / "results/pilots/lr_deterministic_v3/initial_cudnn_matched.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    bundle = collect_lambda_analysis(args.registry, args.baseline, preregistered_registry_path=args.preregistered_registry,
        preregistration_path=args.preregistration, parent_registry_path=args.parent_registry, protocol_status_path=args.protocol_status)
    result = write_lambda_analysis(bundle, args.output, allow_partial=args.allow_partial, plots=not args.no_plots)
    print(json.dumps({k: result[k] for k in ("phase", "complete", "completed_conditions", "expected_conditions", "test_scores_accessed")}))


if __name__ == "__main__":
    main()
