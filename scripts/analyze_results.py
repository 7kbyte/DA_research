#!/usr/bin/env python3
"""Analyze registry-bound formal results without loading any model or GPU."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.analysis import collect_analysis, write_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, help="Explicit active or provisional registry; never glob all result files")
    parser.add_argument("--analysis-status", choices=("active", "provisional-numerical-noise"),
                        default="provisional-numerical-noise", help="Active scientific analysis requires explicit opt-in after protocol freeze")
    parser.add_argument("--protocol-status", default="docs/protocol_status.json", help="Current active-registry authority; archived with the analysis")
    parser.add_argument("--baseline", required=True, help="Explicit initial validation metrics for this protocol/backend")
    parser.add_argument("--output", default="results/analysis/validation")
    parser.add_argument("--generation-summary", default="results/generation_summary.json")
    parser.add_argument("--allow-partial", action="store_true", help="Write explicitly labelled provisional outputs")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--evaluation-root", help="Opt in to frozen test JSON: ROOT/<run_id>/{final,best}.json")
    parser.add_argument("--evaluation-baseline", help="Defaults to ROOT/initial_pythia-410m.json")
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final", help="Never choose a checkpoint using test metrics")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261106)
    args = parser.parse_args()
    if args.evaluation_root and args.analysis_status != "active":
        parser.error("--evaluation-root requires --analysis-status active and a frozen active registry")
    if args.evaluation_root and args.output == "results/analysis/validation":
        parser.error("--evaluation-root requires a separate --output path, so test and validation cannot overwrite one another")
    if args.checkpoint == "best" and not args.evaluation_root:
        parser.error("validation summary analysis uses the fixed-exposure final endpoint; --checkpoint best requires evaluation JSON")
    bundle = collect_analysis(args.registry, args.baseline, evaluation_root=args.evaluation_root,
                              checkpoint=args.checkpoint, evaluation_baseline=args.evaluation_baseline,
                              generation_summary=args.generation_summary,
                              n_bootstrap=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed,
                              analysis_status=args.analysis_status,protocol_status_path=args.protocol_status,
                              require_execution_provenance=args.analysis_status=="active")
    write_analysis(bundle, args.output, allow_partial=args.allow_partial, plots=not args.no_plots)
    print(json.dumps({"output": args.output, **bundle["metadata"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
