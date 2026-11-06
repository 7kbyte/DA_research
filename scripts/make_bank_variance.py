#!/usr/bin/env python3
"""Prepare 60 conditional selection-draw banks/configs; never launch training."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from replay_study.bank_variance import make_bank_variance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-registry",required=True)
    p.add_argument("--protocol-status",default="docs/protocol_status.json")
    p.add_argument("--output",required=True)
    p.add_argument("--run-root",default="/tmp/grad-research-20261106/runs/conditional_bank_variance")
    p.add_argument("--result-root",default="results/conditional_bank_variance")
    p.add_argument("--cpu-threads",type=int,default=2)
    args=p.parse_args()
    import torch
    from threadpoolctl import threadpool_limits
    if args.cpu_threads<1:p.error("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    with threadpool_limits(limits=args.cpu_threads):
        manifest=make_bank_variance(args.parent_registry,args.protocol_status,args.output,
                                    run_root=args.run_root,result_root=args.result_root)
    print(json.dumps({"prepared_banks":len(manifest["banks"]),"training_started":False,"output":args.output}))


if __name__=="__main__":main()
