#!/usr/bin/env python3
"""Freeze supplement A validation-only choices after every physical/alias condition completes."""
import argparse,datetime as dt,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_formal_runs import sha256
from make_experiments import atomic_json


def read(path):return json.loads(Path(path).read_text())


def complete_grid(rows,physical_state,parent_state):
    physical=[row for row in rows if row['execution_kind']=='train'];aliases=[row for row in rows if row['execution_kind']=='alias']
    if len(rows)!=126 or len(physical)!=108 or len(aliases)!=18 or len({r['id'] for r in rows})!=126:
        raise ValueError('Supplement A requires all108 physical and18 parent alias conditions')
    missing=[];failed=[]
    for row in rows:
        state,name=(physical_state,row['id']) if row['execution_kind']=='train' else (parent_state,row['alias_of'])
        record=state['jobs'].get(name,{})
        if record.get('status') in {'failed','blocked'}:failed.append(name)
        if record.get('status')!='completed' or record.get('returncode')!=0:missing.append(name)
    return {'ready':not missing,'missing_actual_runs':missing,'failed_actual_runs':failed}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/supplement_A_validation')
    p.add_argument('--progress',type=Path,default=ROOT/'logs/supplement_A_selection_watch.json')
    p.add_argument('--poll-seconds',type=float,default=60.);args=p.parse_args()
    if not 1<=args.poll_seconds<=60:p.error('Poll at most60 seconds')
    registry=ROOT/'configs/supplemental_v1_A_registry.json';rows=read(registry)
    sources=[registry,ROOT/'configs/lambda_sweep_deterministic_registry.json',
        ROOT/'docs/损失权重敏感性预注册_lambda_sweep_deterministic.json',ROOT/'configs/formal_deterministic_registry.json',
        ROOT/'results/pilots/lr_deterministic_v3/initial_cudnn_matched.json',ROOT/'scripts/analyze_lambda_sweep.py',
        ROOT/'src/replay_study/lambda_analysis.py']
    frozen={str(path):sha256(path) for path in sources}
    while True:
        if any(sha256(path)!=expected for path,expected in frozen.items()):raise ValueError('Frozen A selection authority/source changed')
        gate=complete_grid(rows,read(ROOT/'logs/supplemental_job_state.json'),read(ROOT/'logs/formal_deterministic_job_state.json'))
        record={'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'waiting_complete_grid',
            'pid':os.getpid(),'source_sha256':frozen,'gate':gate,'test_or_behavior_scores_read':False,'gpu_work_started':False}
        if gate['failed_actual_runs']:
            record['status']='failed_job_requires_review';atomic_json(args.progress,record);return 1
        if gate['ready']:
            if (args.output/'validation_lambda_selection.json').exists():raise FileExistsError('Selection output already exists; never overwrite frozen choices')
            command=[sys.executable,str(ROOT/'scripts/analyze_lambda_sweep.py'),'--registry',str(registry),'--output',str(args.output)]
            record.update(status='validating_complete_grid',command=command);atomic_json(args.progress,record)
            subprocess.run(command,cwd=ROOT,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            path=args.output/'validation_lambda_selection.json';selection=read(path)
            if not (selection['complete'] and selection['completed_conditions']==126 and len(selection['all_candidate_scores'])==42 and
                    len(selection['selections'])==6 and all(r['selection_ready'] for r in selection['method_status']) and
                    selection['test_scores_accessed'] is False and selection['selection_uses_test'] is False):
                raise ValueError('Selection must include all42 three-seed candidates and6 complete method choices')
            record.update(status='selection_frozen',updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                selection_path=str(path),selection_sha256=sha256(path),analysis_manifest=str(args.output/'analysis_manifest.json'),
                analysis_manifest_sha256=sha256(args.output/'analysis_manifest.json'),
                choices={r['group']:r['weight'] for r in selection['selections']})
            atomic_json(args.progress,record);print(json.dumps(record));return 0
        atomic_json(args.progress,record)
        if dt.datetime.now(dt.timezone.utc)>=dt.datetime(2026,11,7,0,10,tzinfo=dt.timezone.utc):return 2
        time.sleep(args.poll_seconds)


if __name__=='__main__':raise SystemExit(main())
