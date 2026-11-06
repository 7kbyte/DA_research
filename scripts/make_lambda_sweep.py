#!/usr/bin/env python3
"""Preregister supplemental lambda sensitivity; never launch training jobs."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
GROUPS=['B4','M','B1','B2','B3','C']
WEIGHTS=[.01,.05,.1,.5,1.,5.,10.]
SEEDS=[11,22,33]
IDENTITY_FIELDS={'replay_weight','output_dir','run_label'}


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temporary,path)


def label(value):return format(value,'g').replace('.','p')


def build_sweep(formal_registry,formal_configs,root,scratch,namespace='lambda_sweep'):
    base={(r['group'],r['seed']):r for r in formal_registry
          if r['group'] in GROUPS and r['ratio']==.05 and r['weight']==.05}
    if set(base)!={(group,seed) for group in GROUPS for seed in SEEDS}:
        raise ValueError('Requires all 18 matching formal baselines')
    new_configs,registry,queue,aliases={},[],[],[]
    for group in GROUPS:
        for weight in WEIGHTS:
            for seed in SEEDS:
                source=base[group,seed]
                original=formal_configs[source['id']]
                if (original['target_prediction_tokens']!=10_000_000 or original['learning_rate']!=3e-5
                        or original['replay_ratio']!=.05 or original['replay_weight']!=.05):
                    raise ValueError('Formal scientific protocol differs from supplement A')
                run_id=f'{namespace}_{group}_s{seed}_b0p05_w{label(weight)}'
                record={'id':run_id,'phase':'supplement_A_lambda_sweep','group':group,'seed':seed,
                        'ratio':.05,'weight':weight,'learning_rate':3e-5,
                        'source_formal_run_id':source['id'],'source_config_path':source['config_path'],
                        'test_paths':source.get('test_paths',{})}
                record.update(active=False,launch_status='prepared_not_authorized')
                if weight==.05:
                    record.update(execution_kind='alias',alias_of=source['id'],
                                  config_path=source['config_path'],output_dir=source['output_dir'],
                                  summary_path=source['summary_path'],counted_as_new_training=False)
                    aliases.append(record)
                else:
                    config=copy.deepcopy(original)
                    output=scratch/'runs'/run_id
                    config.update(replay_weight=weight,output_dir=str(output),run_label=run_id)
                    changes={key for key in original if original[key]!=config[key]}
                    if changes!=IDENTITY_FIELDS:
                        raise AssertionError(f'Unexpected configuration changes: {changes}')
                    path=root/'configs'/namespace/f'{run_id}.json'
                    summary=root/'results'/namespace/f'{run_id}_summary.json'
                    new_configs[str(path)]=config
                    record.update(execution_kind='train',alias_of=None,config_path=str(path),
                                  output_dir=str(output),summary_path=str(summary),counted_as_new_training=True)
                    queue.append({'id':run_id,'argv':[sys.executable,str(root/'scripts/make_experiments.py'),
                        'run','--config',str(path),'--summary-copy',str(summary)],'expected_seconds':600,
                        'env':{'CUBLAS_WORKSPACE_CONFIG':':4096:8'} if config.get('deterministic_algorithms') else {},
                        'outputs':[str(output/'summary.json'),str(summary)]})
                registry.append(record)
    return new_configs,registry,queue,aliases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal-registry',type=Path,default=ROOT/'configs/formal_deterministic_registry.json')
    parser.add_argument('--formal-snapshot',type=Path,default=ROOT/'artifacts/provenance/formal_deterministic_input_snapshot.json')
    parser.add_argument('--namespace',default='lambda_sweep_deterministic')
    parser.add_argument('--scratch',type=Path,default=Path('/tmp/grad-research-20261106'))
    args=parser.parse_args()
    source=json.loads(args.formal_registry.read_text())
    frozen=json.loads(args.formal_snapshot.read_text())
    configs={record['id']:json.loads(Path(record['config_path']).read_text()) for record in source}
    for record in source:
        key=str(Path(record['config_path']).relative_to(ROOT))
        if sha256(record['config_path'])!=frozen['configs'][key]['sha256']:
            raise ValueError(f'Formal config changed: {key}')
    if args.namespace.endswith('deterministic') and not all(c.get('deterministic_algorithms') for c in configs.values()):
        raise ValueError('Strict supplement cannot alias provisional nondeterministic formal runs')
    new_configs,registry,queue,aliases=build_sweep(source,configs,ROOT,args.scratch,args.namespace)
    for path,config in new_configs.items():
        destination=Path(path)
        if destination.exists() and json.loads(destination.read_text())!=config:
            raise FileExistsError(f'Refusing changed supplement config: {path}')
        atomic_json(destination,config)
    for record in registry:
        record['source_config_sha256']=sha256(record['source_config_path'])
        record['config_sha256']=sha256(record['config_path'])
        if record['execution_kind']=='alias':
            record['alias_provenance']={'reason':'Exact lambda=.05 cell already exists in the frozen formal grid',
                'formal_run_id':record['alias_of'],'same_model_training_and_evaluation_config':True,
                'new_optimizer_updates':0,'new_training_cost':0,'formal_cost_counted_once':True}
    atomic_json(ROOT/f'configs/{args.namespace}_registry.json',registry)
    atomic_json(ROOT/f'configs/{args.namespace}_queue.json',queue)
    atomic_json(ROOT/f'configs/{args.namespace}_aliases.json',aliases)
    stamp=dt.datetime.now(dt.timezone.utc).isoformat()
    prereg={'preregistered_at_utc':stamp,'status':'prepared_not_started','source_plan':'docs/补充实验预案.md',
        'source_plan_sha256':sha256(ROOT/'docs/补充实验预案.md'),'groups':GROUPS,'lambda_grid':WEIGHTS,
        'seeds':SEEDS,'cells':126,'aliases_to_formal':18,'new_training_runs':108,
        'target_prediction_tokens_per_run':10_000_000,'replay_ratio':.05,'bank_unique_prediction_tokens':100156,
        'learning_rate':3e-5,'data_version':'v3','all_other_parameters':'inherited exactly from each matching formal config',
        'priority':'Complete B4 and M weights before B1/B2/B3/C if resources become limited',
        'single_point_selection':'Per method, choose lambda minimizing the unweighted mean over seeds 11/22/33 of S=(delta target)+.5*(delta C4+delta Wiki), measured at final 10M target exposure; strict equal-score ties choose smaller lambda; validation only',
        'curves':'Report all prespecified weights and both old distributions; no favorable subset selection',
        'test_policy':'Only frozen validation-selected points or the complete prespecified curve; test results never retune hyperparameters',
        'alias_policy':'Lambda=.05 reuses each matching formal checkpoint/summary with explicit provenance and no duplicate training cost',
        'D_policy':'Excluded from sweep as specified in plan A; main D/B4 numerical control is audited separately',
        'launch_policy':'Do not launch before the complete formal audit and explicit root decision',
        'registry_path':f'configs/{args.namespace}_registry.json','queue_path':f'configs/{args.namespace}_queue.json',
        'aliases_path':f'configs/{args.namespace}_aliases.json','registry_sha256':sha256(ROOT/f'configs/{args.namespace}_registry.json'),
        'queue_sha256':sha256(ROOT/f'configs/{args.namespace}_queue.json'),'generator_sha256':sha256(__file__),
        'formal_snapshot_sha256':sha256(args.formal_snapshot),'new_training_tokens_total':1_080_000_000}
    atomic_json(ROOT/f'docs/损失权重敏感性预注册_{args.namespace}.json',prereg)
    (ROOT/f'docs/损失权重敏感性预注册_{args.namespace}.md').write_text(f'''# 损失权重敏感性补充实验预注册

生成时间：{stamp}。依据主结果及测试损失查看之前制定的 `docs/补充实验预案.md` A；本次仅生成，不启动训练。

B1/B2/B3/B4/M/C × λ={{.01,.05,.1,.5,1,5,10}} × seed={{11,22,33}}，共 126 个网格单元。λ=.05 的 18 单元逐一 alias 到对应 formal run，复用 checkpoint/summary，不增加训练或重复计算成本；其余为 **108 个新训练任务**。alias provenance 见 `configs/{args.namespace}_aliases.json`。

LR=3e-5，10M target prediction tokens，b=.05，K=100156；其余参数逐一继承匹配的正式配置。新配置仅改变 λ、输出目录与运行标识；使用冻结科学模块和 v3 validation/test。资源不足时先完成 B4/M 全部权重。任何单点 λ 按方法分别选择：以最终 10M 暴露时的验证 S=(Δtarget)+.5×(ΔC4+ΔWiki)，先对三 seed 无权平均，再选最低均值 λ，严格平手取更小 λ；报告完整预设曲线及两个旧分布，不选择性只展示有利权重。测试仅评估冻结选择或完整预设曲线，不用于调参。

配置：`configs/{args.namespace}/`；126 单元 registry：`configs/{args.namespace}_registry.json`；108 新任务 queue：`configs/{args.namespace}_queue.json`。启动须等主 36 组审计及 root 决定。此文件不构成启动授权。
''')
    print(json.dumps({'status':'prepared_not_started','cells':len(registry),'new_runs':len(queue),'aliases':len(aliases)}))


if __name__=='__main__':main()
