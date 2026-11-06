"""Durable manual full-grid CLI examples, SYNTHETIC ONLY; never real study data."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--artifact-dir',type=Path,default=BASE)
STORE=parser.parse_args().artifact_dir.resolve();STORE.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('toy_tests',ROOT/'tests/test_C_model_nll_review_v2.py')
toy=importlib.util.module_from_spec(spec);spec.loader.exec_module(toy)
reports=[]
for model,mode in [('pythia-1b','six'),('pythia-2.8b','none')]:
    args=toy.fixture(STORE/'toy_cli_inputs',model,mode)
    argv=[sys.executable,str(ROOT/'scripts/review_C_model_nll_results_v2.py'),'--model',model]
    for name in ('complete','queue','state','check_timing','analyze_timing'):
        argv+=['--'+name.replace('_','-'),str(getattr(args,name)),'--'+name.replace('_','-')+'-sha256',getattr(args,name+'_sha256')]
    argv+=['--authority-sha256',args.authority_sha256,'--output',str(args.output)]
    log=STORE/(model+'_toy_cli_stdout.log');err=STORE/(model+'_toy_cli_stderr.log')
    begin=time.perf_counter();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with log.open('xb') as stdout,err.open('xb') as stderr:
        result=subprocess.run(argv,stdout=stdout,stderr=stderr,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    reports.append({'model':model,'synthetic_only':True,'argv':argv,'returncode':result.returncode,
                    'nested_cli_whole_child_wall_seconds':time.perf_counter()-begin,
                    'nested_cli_user_seconds':after.ru_utime-before.ru_utime,
                    'nested_cli_system_seconds':after.ru_stime-before.ru_stime,'output':str(args.output)})
    if result.returncode: raise RuntimeError(err.read_text())
(STORE/'toy_cli_smoke.json').write_text(json.dumps({'synthetic_only':True,'nested_cli_timers':reports},indent=2)+'\n')
print(json.dumps({'synthetic_only':True,'models':[r['model'] for r in reports],'rc':[r['returncode'] for r in reports]}))
