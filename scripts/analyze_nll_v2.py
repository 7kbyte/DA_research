#!/usr/bin/env python3
"""Prepare one explicit v2 NLL analysis stratum; read scores only after unsealing."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_study.nll_v2_analysis import prepare_plan, collect_scores, write_bundle, plot_frontier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--role", choices=("primary", "supplement_A", "supplement_B", "extended"), required=True)
    parser.add_argument("--phase-name", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--data-family", required=True)
    parser.add_argument("--checkpoint", choices=("final", "best"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--read-heldout-scores", action="store_true", help="Use only after root explicitly authorizes unsealing; default never opens score files")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    plan = prepare_plan(args.manifest, args.manifest_sha256, role=args.role, phase_name=args.phase_name,
        model_tag=args.model_tag, data_family=args.data_family, checkpoint=args.checkpoint)
    if args.read_heldout_scores:
        result = collect_scores(plan, read_heldout_scores=True, allow_partial=args.allow_partial, n_bootstrap=args.bootstrap_replicates)
        meta = write_bundle(result, args.output)
        if not args.no_plots:
            plot_frontier(result, args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        meta = {k: plan[k] for k in ("manifest_path", "manifest_sha256", "identity", "expected_conditions", "sources_sha256", "score_files_read")}
        meta["state"] = "prepared_only_waiting_explicit_unsealing"
        (args.output / "prepared_analysis_plan.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("identity", "expected_conditions", "score_files_read")}))


if __name__ == "__main__":
    main()
