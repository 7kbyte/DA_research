#!/usr/bin/env python3
"""Analyze conditional bank draws without claiming independent training seeds."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from replay_study.bank_variance import analyze_bank_variance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry",required=True)
    p.add_argument("--baseline",required=True,help="Matching strict-execution initial validation JSON")
    p.add_argument("--output",required=True)
    p.add_argument("--allow-partial",action="store_true")
    p.add_argument("--no-plots",action="store_true")
    args=p.parse_args()
    print(json.dumps(analyze_bank_variance(args.registry,args.baseline,args.output,allow_partial=args.allow_partial,plots=not args.no_plots)))


if __name__=="__main__":main()
