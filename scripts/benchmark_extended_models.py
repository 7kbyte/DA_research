#!/usr/bin/env python3
"""Isolated 10-step extended-model throughput probes; never select hyperparameters."""
from __future__ import annotations
import argparse,dataclasses,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from make_experiments import atomic_json
from make_extended_experiments import MODELS,build_extended

def one(config_path,output):
    import torch
    from replay_study.train import TrainConfig,train
    cfg=TrainConfig.from_dict(json.loads(config_path.read_text()));started=time.monotonic()
    try:
        summary=train(cfg)
        rows=[json.loads(line) for line in (Path(cfg.output_dir)/'events.jsonl').read_text().splitlines()]
        steps=[r for r in rows if r['event']=='train_step'];steady=steps[1:]
        seconds=sum(r['step_seconds'] for r in steady)
        tokens=sum(r['step_counts']['target_prediction_tokens'] for r in steady)
        assert len(steps)==10 and summary['status']=='max_steps'
        result={'status':'succeeded','model':Path(cfg.model_name_or_path).name.removesuffix('-final'),
            'microbatch':cfg.microbatch_size,'accumulation':cfg.gradient_accumulation_steps,'steps':10,
            'steady_steps':'2-10','steady_target_tokens':tokens,'steady_seconds':seconds,
            'steady_target_prediction_tokens_per_second':tokens/seconds,'steady_mean_step_seconds':seconds/9,
            'peak_gpu_allocated_bytes':summary['peak_gpu_memory_bytes'],
            'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'checkpoint_bytes':Path(summary['checkpoint_path']).stat().st_size,
            'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
            'counters':summary['counters'],'provenance':summary['provenance'],
            'summary_path':str(Path(cfg.output_dir)/'summary.json'),'config_path':str(config_path),
            'child_wall_seconds':time.monotonic()-started,'timer_note':'Teacher forward is nested in training seconds; checkpoint is separate'}
        atomic_json(output,result)
    except BaseException as error:
        atomic_json(output,{'status':'failed','exception':type(error).__name__,'message':str(error),
            'model':Path(cfg.model_name_or_path).name.removesuffix('-final'),
            'microbatch':cfg.microbatch_size,'accumulation':cfg.gradient_accumulation_steps,
            'is_cuda_oom':isinstance(error,torch.cuda.OutOfMemoryError),'config_path':str(config_path),
            'failed_wall_seconds':time.monotonic()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()})
        traceback.print_exc();raise

def run(gpu,cases=None,report=None):
    report=report or ROOT/'results/benchmarks/extended_throughput.json'
    attempts=[]
    for model,first_micro in cases or [('pythia-1b',16),('pythia-2.8b',8)]:
        candidates=[8,4] if model=='pythia-2.8b' and first_micro==8 else [first_micro]
        for micro in candidates:
            configs,_,_=build_extended(model,'lambda',ROOT,Path('/tmp/grad-research-20261106'),3e-5,.05,micro)
            cfg=next(c for c in configs.values() if c['replay_objective']=='kl' and 'generated_random' in c['replay_path'] and c['replay_weight']==.05)
            run_id=f"benchmark_extended_{model.replace('.','p').replace('-','_')}_m{micro}_s101"
            cfg.update(output_dir=f'/tmp/grad-research-20261106/runs/{run_id}',run_label=run_id,
                max_steps=10,evaluate_at_start=False,evaluate_at_end=False,evaluate_teacher_kl=False,
                eval_every_steps=0,validation_paths={})
            config_path=ROOT/f'configs/benchmarks/{run_id}.json';output=ROOT/f'results/benchmarks/{run_id}.json'
            atomic_json(config_path,cfg);log_path=ROOT/f'logs/{run_id}.log';started=time.monotonic()
            with log_path.open('ab',buffering=0) as stream:
                code=subprocess.call([sys.executable,str(Path(__file__).resolve()),'--one',str(config_path),
                    '--output',str(output)],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,
                    'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            result=json.loads(output.read_text()) if output.exists() else {'status':'failed','is_cuda_oom':False}
            result.update(model=model,microbatch=micro,outer_returncode=code,
                outer_wall_seconds=time.monotonic()-started,log_path=str(log_path))
            attempts.append(result);atomic_json(report,{'attempts':attempts,'gpu':gpu,'formal_configs_changed':False})
            if result['status']=='succeeded':break
            if not result.get('is_cuda_oom'):raise RuntimeError(f'Non-OOM benchmark failure: {log_path}')
    print(json.dumps({'attempts':len(attempts),'output':str(report)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',default='3')
    parser.add_argument('--case',action='append',help='MODEL:MICROBATCH (repeatable); effective batch remains64')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--one',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    cases=[(item.split(':')[0],int(item.split(':')[1])) for item in args.case] if args.case else None
    if cases and any(model not in MODELS or micro<1 or 64%micro for model,micro in cases):parser.error('Invalid model/microbatch case')
    one(args.one,args.output) if args.one else run(args.gpu,cases,args.report)
