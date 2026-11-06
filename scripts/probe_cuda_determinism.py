#!/usr/bin/env python3
"""Independent deterministic model-gradient and three-step repeat probe."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,dataclasses,gc,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from replay_study.train import TrainConfig,TokenStream,load_training_data,train
from replay_study.objectives import causal_cross_entropy
from transformers import AutoModelForCausalLM


def fingerprint(items):
 d=hashlib.sha256()
 for name,t in items:
  t=t.detach().cpu().contiguous()
  d.update(json.dumps([name,str(t.dtype),list(t.shape)]).encode())
  d.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
 return d.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--gpu-label',required=True)
 p.add_argument('--backend',choices=['auto','math'],default='auto')
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 cfg=TrainConfig.from_dict(json.loads((ROOT/'configs/formal/formal_B4_s11_b0p05_w0p05.json').read_text()))
 cfg.deterministic_algorithms=True;cfg.attention_backend=a.backend;cfg.device='cuda:0'
 torch.use_deterministic_algorithms(True,warn_only=False)
 torch.backends.cuda.enable_flash_sdp(a.backend!='math')
 torch.backends.cuda.enable_mem_efficient_sdp(a.backend!='math')
 torch.backends.cuda.enable_math_sdp(True)
 if hasattr(torch.backends.cuda,'enable_cudnn_sdp'):torch.backends.cuda.enable_cudnn_sdp(a.backend!='math')
 report={'gpu':a.gpu_label,'backend':a.backend,'torch_version':torch.__version__,'torch_git_version':torch.version.git_version,
  'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'gradient_repeats':[],'three_step_repeats':[]}
 def save():
  a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
 batch=next(TokenStream(load_training_data(cfg.target_path),cfg.seed).take(16*511,16))
 for repeat in [1,2]:
  torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
  model=AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path,torch_dtype=torch.float32,attn_implementation='sdpa').cuda().train()
  ids=batch['input_ids'].cuda();begin=time.monotonic()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   logits=model(ids,use_cache=False).logits
   loss=causal_cross_entropy(logits,ids).mean
  loss.backward();torch.cuda.synchronize()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'))
  row={'repeat':repeat,'nll':float(loss),'grad_norm':float(norm),
   'gradient_sha256':fingerprint((name,p.grad) for name,p in sorted(model.named_parameters()) if p.grad is not None),
   'seconds':time.monotonic()-begin}
  report['gradient_repeats'].append(row);save();print(json.dumps({'kind':'gradient',**row}),flush=True)
  del model,ids,logits,loss;gc.collect();torch.cuda.empty_cache()
 for repeat in [1,2]:
  run_id=f'diagnostic_deterministic_{a.backend}_gpu{a.gpu_label}_r{repeat}'
  config=dataclasses.replace(cfg,output_dir=f'/tmp/grad-research-20261106/runs/{run_id}',
   max_steps=3,save_every_steps=0,eval_every_steps=0,evaluate_at_start=False,evaluate_at_end=False,
   validation_paths={},run_label=run_id)
  summary=train(config)
  saved=torch.load(summary['checkpoint_path'],map_location='cpu',weights_only=False)
  row={'repeat':repeat,'model_sha256':fingerprint(sorted(saved['model_state'].items())),
   'summary_path':str(Path(config.output_dir)/'summary.json'),'counters':summary['counters']}
  del saved
  report['three_step_repeats'].append(row);save();print(json.dumps({'kind':'three_step',**row}),flush=True)
  gc.collect();torch.cuda.empty_cache()
 report['gradient_bitwise_equal']=len({x['gradient_sha256'] for x in report['gradient_repeats']})==1
 report['three_step_bitwise_equal']=len({x['model_sha256'] for x in report['three_step_repeats']})==1
 report['status']='complete';save()


if __name__=='__main__':main()
