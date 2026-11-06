#!/usr/bin/env python3
"""Select an equal-budget replay bank from an immutable shared candidate pool."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replay_study.selection import METHODS, save_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bank-size", type=int)
    parser.add_argument("--prediction-token-budget", type=int)
    parser.add_argument("--statistics")
    args = parser.parse_args()
    manifest = save_bank(args.pool, args.output, method=args.method, seed=args.seed, bank_size=args.bank_size,
                         prediction_token_budget=args.prediction_token_budget, statistics=args.statistics)
    print(f"{args.output}: {manifest['bank_chunks']} chunks, {manifest['prediction_tokens']} prediction tokens")


if __name__ == "__main__":
    main()
