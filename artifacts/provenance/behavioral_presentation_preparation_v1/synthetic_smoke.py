"""Self-contained hand-specified counts; all scores are SYNTHETIC examples."""
from collections import defaultdict
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('behavior_presentation',ROOT/'scripts/present_behavioral_review.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def proof(path):return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return proof(path)
def table(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return proof(path)
def stats(values):return {'mean':statistics.mean(values),'std_ddof1':statistics.stdev(values),'n_training_seeds':3}


def fixture(name):
    directory=OUT/'synthetic_inputs'/name
    groups=list(p.GROUP_ORDER)
    if name=='primary':
        model='pythia-410m';kinds=('final','best')
        conditions=[(g,0. if g=='B0' else .05,0. if g=='B0' else .05) for g in groups]+[(g,b,.05) for g in ('B4','M') for b in (.01,.1)]
    elif name=='supplement_A':
        model='pythia-410m';kinds=('final',);groups=['B1','B2','B3','B4','M','C']
        conditions=list(zip(groups,[.05]*6,[.05,1.,.1,5.,5.,1.]))
    else:
        model='pythia-410m' if '410m' in name else 'pythia-1b' if '1b_' in name else 'pythia-2.8b'
        kinds=('final',)
        if '2p8b' in name:groups=['B0','B1','B2','B4','M']
        conditions=[(g,0. if g=='B0' else .05,0. if g=='B0' else .5) for g in groups]
    records=[];endpoints=[]
    for ci,(group,ratio,weight) in enumerate(conditions):
        # Same-seed D equals B4 by construction; not a statement about real data.
        effective_ci=next(i for i,c in enumerate(conditions) if c[0]=='B4' and c[1]==ratio) if group=='D' else ci
        for kind in kinds:
            for seed,offset in zip(p.SEEDS,(-1,0,1)):
                run=f'SYNTHETIC_{name}_{group}_b{ratio}_s{seed}'
                exposure=(100_000_000 if name.startswith('extended_') else 10_000_000) if kind=='final' else 4_000_000
                if '2p8b' in name:exposure=25_000_000
                record={'run_id':run,'kind':kind,'phase_name':name,'model_tag':model,'group':group,'ratio':ratio,'weight':weight,
                    'learning_rate':3e-5,'training_seed':seed,'selection_seed':None,'step':306 if kind=='final' else 122,
                    'target_prediction_tokens_at_checkpoint':exposure,'replay_prediction_tokens_at_checkpoint':int(exposure*ratio)}
                records.append(record)
                for task in p.TASKS:
                    for score in p.SCORES:
                        initial=(70 if score=='acc_norm' else 60) if task=='piqa' else (45 if score=='acc_norm' else 40)
                        after=initial-2+effective_ci+(1+effective_ci%3)*offset+(kind=='best')
                        cw=8+offset;wc=after-initial+cw
                        assert 0<=wc<=100-initial and 0<=cw<=initial
                        endpoint={**record,'checkpoint_kind':kind,'seed':seed,'task':task,'metric':score,'items':100,
                            'initial_accuracy':initial/100,'accuracy':after/100,'accuracy_delta':(after-initial)/100,
                            'CC':initial-cw,'CW':cw,'WC':wc,'WW':100-initial-wc,
                            'conditional_forgetting_rate':cw/initial,'conditional_acquisition_rate':wc/(100-initial),
                            'accuracy_ci_low':max(0,(after-5)/100),'accuracy_ci_high':min(1,(after+5)/100),
                            'delta_ci_low':(after-initial-5)/100,'delta_ci_high':(after-initial+5)/100}
                        endpoints.append(endpoint)
    grouped=defaultdict(list)
    for r in endpoints:grouped[tuple(r[k] for k in ('checkpoint_kind',*p.CELL,'task','metric'))].append(r)
    aggregates=[]
    for cell,rows in grouped.items():
        for quantity in ('accuracy','accuracy_delta','conditional_forgetting_rate','conditional_acquisition_rate'):
            aggregates.append({**dict(zip(('checkpoint_kind',*p.CELL,'task','metric'),cell)),'quantity':quantity,**stats([r[quantity] for r in rows])})
    pairrows=[]
    for r in endpoints:
        for left,right in p.PAIRS:
            if r['group']!=left:continue
            b=next((b for b in endpoints if b['group']==right and all(b[k]==r[k] for k in ('checkpoint_kind','ratio','weight','learning_rate','task','metric','seed'))),None)
            if b:pairrows.append({**{k:r[k] for k in ('checkpoint_kind','ratio','weight','learning_rate','task','metric','seed')},
                'left_group':left,'right_group':right,'accuracy_left_minus_right':r['accuracy']-b['accuracy'],'delta_accuracy_left_minus_right':r['accuracy_delta']-b['accuracy_delta']})
    grouped=defaultdict(list)
    for r in pairrows:grouped[tuple(r[k] for k in ('checkpoint_kind',*p.PAIR_CELL,'task','metric'))].append(r)
    pair_stats=[{**dict(zip(('checkpoint_kind',*p.PAIR_CELL,'task','metric'),cell)),'quantity':'accuracy_left_minus_right',**stats([r['accuracy_left_minus_right'] for r in rows])} for cell,rows in grouped.items()]
    tables=dict(zip(p.BASE_FILES,(endpoints,aggregates,pairrows,pair_stats)))
    if name=='supplement_A':
        selected=[];intervals=[]
        for left,right in p.SELECTED_PAIRS:
            for seed in p.SEEDS:
                for task in p.TASKS:
                    for score in p.SCORES:
                        a=next(r for r in endpoints if r['group']==left and r['seed']==seed and r['task']==task and r['metric']==score)
                        b=next(r for r in endpoints if r['group']==right and r['seed']==seed and r['task']==task and r['metric']==score)
                        row={'comparison':left+'-'+right,'training_seed':seed,'task':task,'metric':score,'lambda_a':a['weight'],'lambda_b':b['weight'],
                             'accuracy_left_minus_right':a['accuracy']-b['accuracy']}
                        selected.append(row);intervals.append({**row,'ci_low':row['accuracy_left_minus_right']-.05,'ci_high':row['accuracy_left_minus_right']+.05})
        grouped=defaultdict(list)
        for r in selected:grouped[tuple(r[k] for k in ('comparison','lambda_a','lambda_b','task','metric'))].append(r)
        resolved=[{**dict(zip(('comparison','lambda_a','lambda_b','task','score'),cell)),'quantity':'accuracy_left_minus_right',**stats([r['accuracy_left_minus_right'] for r in rows])} for cell,rows in grouped.items()]
        tables.update(dict(zip(p.SELECTED_FILES,(selected,intervals,resolved))))
    manifest={'mode':'behavior','synthetic_only':True,'records':records}
    manifest_proof=write(directory/'manifest.json',manifest)
    files={name:table(directory/(name+'.csv'),rows) for name,rows in tables.items()}
    fixture_proof=write(directory/'synthetic_fixture.json',{'synthetic_only':True,'not_actual_evaluation':True,
        'manual_formula':'100 questions; fixed initial correct counts, hand-specified CW/WC and seed offsets. Fixture intervals are illustrative ±5pp, not computed scientific CIs.',
        'manifest':manifest_proof,'files':files})
    p.verify_tables(manifest,tables,'A' if name=='supplement_A' else 'primary' if name=='primary' else 'C')
    # Small meaningful failure checks, not a mirror implementation test suite.
    partial=copy.deepcopy(tables);partial['endpoints'].pop()
    try:p.verify_tables(manifest,partial,'A' if name=='supplement_A' else 'primary' if name=='primary' else 'C')
    except ValueError:pass
    else:raise AssertionError('Incomplete endpoint table must fail')
    return fixture_proof


def main():
    cases=('primary','supplement_A','extended_pythia_410m_100m','extended_pythia_1b_100m','extended_pythia_2p8b_25m')
    results=[]
    for name in cases:
        f=fixture(name);output=OUT/'synthetic_outputs'/name
        argv=[sys.executable,str(ROOT/'scripts/present_behavioral_review.py'),'--synthetic-fixture',f['path'],'--output',str(output)]
        child=subprocess.run(argv,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},check=True)
        results.append({'phase':name,'fixture':f,'output_COMPLETE':proof(output/'COMPLETE.json'),
            'nested_presentation_timer':proof(output.with_name(output.name+'_whole_child_timing.json'))})
    write(OUT/'synthetic_smoke_COMPLETE.json',{'complete':True,'synthetic_only':True,'phases':results,
        'actual_score_or_prediction_files_read':False,'original_check_analyze_reviewer_or_inference_called':False,
        'scope':'All five whole-phase shapes; missing endpoint rejection; figures from hand-defined examples only'})


if __name__=='__main__':main()
