#!/usr/bin/env python3
"""Summarize frozen behavioral results on CPU; never load a checkpoint or infer."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from replay_study.behavioral_analysis import collect,write_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--bootstrap-replicates',type=int,default=5000)
    parser.add_argument('--bootstrap-seed',type=int,default=20261106)
    args=parser.parse_args()
    if args.bootstrap_replicates<1:parser.error('Positive bootstrap count required')
    bundle=collect(args.manifest,args.bootstrap_replicates,args.bootstrap_seed)
    write_bundle(bundle,args.output)
    print(json.dumps({'output':args.output,'complete':True,'model_inference':False}))


if __name__=='__main__':main()
