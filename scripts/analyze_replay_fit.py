#!/usr/bin/env python3
"""Read only the explicitly unsealed 15-case replay-fit diagnostic."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.replay_fit_analysis import collect, export

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--manifest", type=Path, default=Path("docs/replay_fit_diagnostic_v2_manifest.json"))
p.add_argument("--manifest-sha256", default="391baf218cd0e4e13b902236b3ea0fe799518676cbb2fd2357cd6234e51ac019")
p.add_argument("--timing", type=Path, default=Path("results/auxiliary_evaluation_timing_completed_v1.json"))
p.add_argument("--output", type=Path, default=Path("results/analysis/replay_fit_exploratory_v2"))
p.add_argument("--unseal-replay-fit-only", action="store_true")
args = p.parse_args()
result = collect(args.manifest, args.manifest_sha256, args.timing, unseal_replay_fit=args.unseal_replay_fit_only)
export(result, args.output)
print(result["status"], len(result["rows"]), "adapted endpoint records; no heldout/behavioral results opened")
