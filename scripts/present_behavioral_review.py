#!/usr/bin/env python3
"""Compact figures from an already ROOT-unsealed independent behavior review.

No scorer/collector imports, question data, checkpoint access, CI calculation or
method selection. Synthetic fixture mode accepts only a self-contained toy.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REVIEWER_SHA='bb261a20ac45fb47c690f57b7461856b5204066eade15178ca92886a9766966b'
BASE_FILES=('endpoints','seed_aggregates','paired_seed_differences','paired_seed_aggregates')
SELECTED_FILES=('selected_policy_pairs','selected_question_intervals','selected_policy_score_resolved_statistics')
GROUP_ORDER=('B0','B1','B2','B3','B4','D','M','C')
TASKS=('piqa','hellaswag');SCORES=('acc_norm','acc');SEEDS=(11,22,33)
PAIRS=(('M','B4'),('C','B4'),('B2','B1'),('B4','B3'),('D','B4'))
SELECTED_PAIRS=(('M','B4'),('C','B4'),('B2','B1'),('B4','B3'),('B3','B1'),('B4','B2'))
CELL=('group','ratio','weight','learning_rate')
PAIR_CELL=('left_group','right_group','ratio','weight','learning_rate')
DEFAULT_FONT=ROOT/'presentations/答辩工作稿_20261106_1730/fonts/NotoSerifCJKsc-Regular.otf'


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()


class SmallInputs:
    def __init__(self):self.sources={};self.content={}
    def bytes(self,path,expected=None):
        path=str(Path(path).resolve());p=Path(path)
        require(p.suffix in ('.json','.csv','.py','.otf','.ttf','.ttc'), 'Small presentation inputs only')
        require(p.stat().st_size<32<<20,'Unexpectedly large presentation input')
        if path not in self.content:
            a=p.stat();raw=p.read_bytes();b=p.stat()
            signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            require(signature(a)==signature(b),'Input changed during read')
            self.content[path]=raw;self.sources[path]={'sha256':sha(raw),'bytes':len(raw),'five_stat':signature(b)}
        if expected is not None:require(self.sources[path]['sha256']==expected,'SHA differs: '+path)
        return self.content[path]
    def proof(self,p):return json.loads(self.bytes(p['path'],p['sha256']))
    def finish(self):
        for path,item in self.sources.items():
            s=Path(path).stat()
            require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==tuple(item['five_stat']),'Presentation input changed before completion')
    def csv(self,path,digest):
        def value(v):
            if v=='':return None
            if v in ('True','False'):return v=='True'
            try:return float(v)
            except ValueError:return v
        return [{k:value(v) for k,v in r.items()} for r in csv.DictReader(io.StringIO(self.bytes(path,digest).decode()))]


def scope(manifest):
    records=manifest['records'];phase={r['phase_name'] for r in records}
    require(manifest.get('mode')=='behavior' and len(phase)==1,'One full behavior phase required')
    phase=next(iter(phase));models={r['model_tag'] for r in records}
    require(len(models)==1,'Do not mix models')
    kinds=Counter(r['kind'] for r in records)
    if phase=='primary':
        family='primary';expected=72;expected_groups=set(GROUP_ORDER)
        require(kinds=={'final':36,'best':36} and models=={'pythia-410m'},'Full primary72 required')
    elif phase=='supplement_A':
        family='A';expected=18;expected_groups={'B1','B2','B3','B4','M','C'}
        require(kinds=={'final':18} and models=={'pythia-410m'},'Full validation-selected A18 required')
    else:
        names={'extended_pythia_410m_100m':('pythia-410m',24), 'extended_pythia_1b_100m':('pythia-1b',24),
               'extended_pythia_2p8b_25m':('pythia-2.8b',15)}
        require(phase in names,'Unknown frozen behavior phase')
        model,expected=names[phase];family='C';expected_groups=set(GROUP_ORDER) if expected==24 else {'B0','B1','B2','B4','M'}
        require(kinds=={'final':expected} and models=={model},'Complete model-specific final behavior required')
    require(len(records)==expected and len({(r['run_id'],r['kind']) for r in records})==expected,'Incomplete/duplicate logical endpoints')
    require({r['group'] for r in records}==expected_groups and all(r.get('selection_seed') is None for r in records),'Missing method or unsupported replicate unit')
    grouped=defaultdict(list)
    for r in records:grouped[(r['kind'],*(r[k] for k in CELL))].append(r)
    require(all(len(v)==3 and {r['training_seed'] for r in v}==set(SEEDS) for v in grouped.values()),'Every cell needs all three fixed training seeds')
    if family=='primary':
        expected_cells={('B0',0.,0.)}|{(g,.05,.05) for g in GROUP_ORDER if g!='B0'}|{(g,b,.05) for g in ('B4','M') for b in (.01,.1)}
        require({(r['group'],r['ratio'],r['weight']) for r in records}==expected_cells,'Primary ratio/weight coverage changed')
    return family,phase,expected


def load(args,reader):
    if args.synthetic_fixture:
        fixture_path=args.synthetic_fixture.resolve();fixture=json.loads(reader.bytes(fixture_path))
        require(fixture.get('synthetic_only') is True and fixture.get('not_actual_evaluation') is True,'Explicit toy fixture required')
        for p in [fixture['manifest'],*fixture['files'].values()]:
            require(Path(p['path']).resolve().is_relative_to(fixture_path.parent),'Synthetic sources must stay within toy directory')
        manifest=reader.proof(fixture['manifest']);family,phase,total=scope(manifest)
        require(manifest.get('synthetic_only') is True and all(r['run_id'].startswith('SYNTHETIC_') for r in manifest['records']),'Toy mode refuses actual endpoint identities')
        tables={name:reader.csv(p['path'],p['sha256']) for name,p in fixture['files'].items()}
        return manifest,tables,family,phase,total,True,{'synthetic_fixture':{'path':str(fixture_path),'sha256':sha(fixture_path.read_bytes())}}
    require(args.review_complete is not None and args.review_complete_sha256,'Explicit reviewer COMPLETE path/SHA required')
    proof={'path':str(args.review_complete.resolve()),'sha256':args.review_complete_sha256}
    review=reader.proof(proof)
    require(review.get('complete') is True and review.get('checkpoint_payload_bytes_read')==0 and review.get('model_forward_calls')==0,'Successful independent reviewer required')
    require(review.get('reviewer_source',{}).get('sha256')==REVIEWER_SHA,'Approved independent reviewer source differs')
    reader.bytes(review['reviewer_source']['path'],REVIEWER_SHA)
    original=reader.proof(review['original_COMPLETE']);authority=reader.proof(original['authority'])
    require(original.get('complete') is True and authority.get('authorized_by')=='root' and authority.get('allow_score_analysis') is True
            and authority.get('allow_partial') is False and authority.get('plan')==original['plan'] and authority.get('manifest')==review['manifest'],'Matching complete ROOT-unsealed analysis required')
    plan=reader.proof(original['plan']);require(plan['manifest']==review['manifest'],'Analysis plan/manifest differs')
    first=reader.proof(original.get('journal') or original['first_read_receipt'])
    require(first.get('authority')==original['authority'] and first.get('plan')==original['plan']
            and (first.get('status')=='semantic_access_enabled' or first.get('before_first_score_parse') is True),'First-read binding missing')
    manifest=reader.proof(review['manifest']);family,phase,total=scope(manifest)
    require(review['logical_endpoints']==total and review['endpoint_task_metric_rows']==total*4,'Incomplete reviewed endpoint inventory')
    if family=='A':require(review.get('validation_selected_review',{}).get('rows')==72,'Selected18×2tasks×2scores review required')
    root=args.review_complete.resolve().parent
    names=BASE_FILES+(SELECTED_FILES if family=='A' else ())
    require(set(review['review_output_sha256'])=={str((root/(name+'.csv')).resolve()) for name in names},'All fixed review tables must be present')
    tables={name:reader.csv(root/(name+'.csv'),review['review_output_sha256'][str((root/(name+'.csv')).resolve())]) for name in names}
    return manifest,tables,family,phase,total,False,{'review_COMPLETE':proof,'original_COMPLETE':review['original_COMPLETE'],'manifest':review['manifest'],'authority':original['authority']}


def index(rows,keys):
    out={tuple(r.get(k) for k in keys):r for r in rows};require(len(out)==len(rows),'Duplicate presentation row');return out


def verify_tables(manifest,tables,family):
    endpoints=tables['endpoints'];by=index(endpoints,('run_id','checkpoint_kind','task','metric'))
    expected={(r['run_id'],r['kind'],task,metric) for r in manifest['records'] for task in TASKS for metric in SCORES}
    require(set(by)==expected,'Missing task/score/logical endpoint')
    for r in manifest['records']:
        for task in TASKS:
            for score in SCORES:
                row=by[r['run_id'],r['kind'],task,score]
                require(all(row[k]==r[k] for k in (*CELL,'training_seed','step','target_prediction_tokens_at_checkpoint','replay_prediction_tokens_at_checkpoint')),'Endpoint identity/exposure differs')
                require(row['seed']==r['training_seed'],'Seed identity differs')
    aggregates=index(tables['seed_aggregates'],('checkpoint_kind',*CELL,'task','metric','quantity'))
    cells={(r['checkpoint_kind'],*(r[k] for k in CELL),r['task'],r['metric']) for r in endpoints}
    quantities=('accuracy','accuracy_delta','conditional_forgetting_rate','conditional_acquisition_rate')
    require(set(aggregates)=={(*c,q) for c in cells for q in quantities},'All seed aggregates required')
    pairrows=tables['paired_seed_differences'];pairkeys=('checkpoint_kind',*PAIR_CELL,'task','metric','seed')
    pairindex=index(pairrows,pairkeys);expected_pairs=set()
    for r in endpoints:
        for left,right in PAIRS:
            if r['group']!=left:continue
            candidate=[b for b in endpoints if b['group']==right and all(b[k]==r[k] for k in ('checkpoint_kind','ratio','weight','learning_rate','task','metric','seed'))]
            if candidate:expected_pairs.add((r['checkpoint_kind'],left,right,r['ratio'],r['weight'],r['learning_rate'],r['task'],r['metric'],r['seed']))
    require(set(pairindex)==expected_pairs,'Frozen same-lambda paired inventory incomplete')
    stats=index(tables['paired_seed_aggregates'],('checkpoint_kind',*PAIR_CELL,'task','metric','quantity'))
    require(set(stats)=={(*k[:-1],'accuracy_left_minus_right') for k in expected_pairs},'Paired aggregate inventory incomplete')
    if family=='A':
        selected=index(tables['selected_policy_pairs'],('comparison','training_seed','task','metric'))
        expected={(a+'-'+b,seed,task,score) for a,b in SELECTED_PAIRS for seed in SEEDS for task in TASKS for score in SCORES}
        require(set(selected)==expected,'All six selected-policy contrasts required')
        resolved=index(tables['selected_policy_score_resolved_statistics'],('comparison','task','score','quantity'))
        require(set(resolved)=={(a+'-'+b,task,score,'accuracy_left_minus_right') for a,b in SELECTED_PAIRS for task in TASKS for score in SCORES},'A score-resolved statistics required; original collision table is forbidden')
        for row in tables['selected_policy_score_resolved_statistics']:
            require(row['score'] in SCORES and row['quantity']=='accuracy_left_minus_right','Score/quantity labels must stay separate')
        require(set(index(tables['selected_question_intervals'],('comparison','training_seed','task','metric')))==expected,'Selected conditional interval inventory incomplete')


def plot(tables,family,phase,synthetic,output,font_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font_path).get_name(),'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path','pdf.fonttype':42})
    kinds=['final','best'] if any(r['checkpoint_kind']=='best' for r in tables['endpoints']) else ['final']
    colors={'final':'#28659A','best':'#C75D20'};markers={11:'o',22:'s',33:'^'}
    def legend(fig):
        handles=[Line2D([],[],color=colors[k],marker='D',label=k+' mean±sample SD') for k in kinds]
        handles += [Line2D([],[],color='#555',marker=markers[s],lw=0,label='seed '+str(s),markersize=3.5) for s in SEEDS]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=len(handles),frameon=False,fontsize=7)
    figures=[]
    def save(fig,name,height):
        for ext in ('png','pdf','svg'):fig.savefig(output/(name+'.'+ext),dpi=180)
        figures.append({'name':name,'width_inches':7,'height_inches':height});plt.close(fig)
    prefix='SYNTHETIC / 合成展示，无真实结果' if synthetic else phase
    conditions=sorted({tuple(r[k] for k in CELL) for r in tables['endpoints']},key=lambda c:(GROUP_ORDER.index(c[0]),c[1],c[2]))
    # Twelve primary conditions share each 3-inch task panel. A compact ratio
    # avoids overlapping labels while retaining every exact frozen cell.
    labels=[f'{c[0]}\n'+(f'λ={c[2]:g}' if family=='A' else f'{c[1]:g}'.removeprefix('0') or '0') for c in conditions]
    endpoints=tables['endpoints'];aggregates=tables['seed_aggregates']
    for score in SCORES:
        fig,axes=plt.subplots(2,2,figsize=(7,6.45));legend(fig)
        for col,task in enumerate(TASKS):
            for row,(quantity,title) in enumerate((('accuracy','Accuracy (%)'),('accuracy_delta','Δ initial (percentage points)'))):
                ax=axes[row,col]
                for ki,kind in enumerate(kinds):
                    offset=(ki-(len(kinds)-1)/2)*.22
                    for ci,condition in enumerate(conditions):
                        points=[r for r in endpoints if r['checkpoint_kind']==kind and r['task']==task and r['metric']==score and tuple(r[k] for k in CELL)==condition]
                        summary=next(r for r in aggregates if r['checkpoint_kind']==kind and r['task']==task and r['metric']==score and r['quantity']==quantity and tuple(r[k] for k in CELL)==condition)
                        ax.errorbar(ci+offset,100*summary['mean'],yerr=100*summary['std_ddof1'] if summary['std_ddof1'] is not None else None,fmt='D',color=colors[kind],markersize=3.2,capsize=2,lw=1,zorder=3)
                        for point in points:
                            jitter=(SEEDS.index(int(point['seed']))-1)*.035
                            ax.plot(ci+offset+jitter,100*point[quantity],marker=markers[int(point['seed'])],color=colors[kind],markersize=2.5,alpha=.7,lw=0)
                values={r['initial_accuracy'] for r in endpoints if r['task']==task and r['metric']==score}
                require(len(values)==1,'One matched initial per task/score required')
                ax.axhline(100*next(iter(values)) if row==0 else 0,color='.45',ls='--',lw=.8)
                ax.set_xticks(range(len(conditions)),labels,fontsize=6.5);ax.set_xlim(-.5,len(conditions)-.5)
                ax.grid(axis='y',alpha=.2);ax.set_ylabel(title);ax.set_title(task if row==0 else 'Matched initial change',fontsize=9)
        # A shared delta scale compares drift magnitude across tasks; absolute
        # accuracies retain task-specific scales and are not cross-task scores.
        lo=min(ax.get_ylim()[0] for ax in axes[1]);hi=max(ax.get_ylim()[1] for ax in axes[1])
        for ax in axes[1]:ax.set_ylim(lo,hi)
        fig.suptitle(prefix+'\n'+score+(' — primary' if score=='acc_norm' else ' — auxiliary'),fontsize=9.5,y=.995)
        fig.text(.5,.018,'Labels: method / '+('validation λ' if family=='A' else 'replay ratio b')+'; dashed = matched initial (top) / zero (bottom).',ha='center',fontsize=6.6)
        fig.text(.5,.003,'All fixed seed points and mean±sample SD; final/best/aliases add no training replicates.',ha='center',fontsize=6.6)
        fig.subplots_adjust(left=.10,right=.985,bottom=.10,top=.865,wspace=.30,hspace=.43)
        save(fig,'endpoints_'+score,6.45)
        pair_plot(plt,Line2D,tables['paired_seed_differences'],tables['paired_seed_aggregates'],False,score,kinds,colors,markers,prefix,output,figures)
        if family=='A':pair_plot(plt,Line2D,tables['selected_policy_pairs'],tables['selected_policy_score_resolved_statistics'],True,score,['final'],colors,markers,prefix,output,figures)
    return figures


def pair_plot(plt,Line2D,rows,statistics,selected,score,kinds,colors,markers,prefix,output,figures):
    if selected:
        keys=('comparison','lambda_a','lambda_b');scorekey='score';quantity='quantity'
        conditions=sorted({tuple(r[k] for k in keys) for r in rows},key=lambda c:[a+'-'+b for a,b in SELECTED_PAIRS].index(c[0]))
        labels=[c[0]+'\nλ '+f'{c[1]:g}/{c[2]:g}' for c in conditions]
    else:
        keys=PAIR_CELL;scorekey='metric';quantity='quantity'
        conditions=sorted({tuple(r[k] for k in keys) for r in rows},key=lambda c:([a+'-'+b for a,b in PAIRS].index(c[0]+'-'+c[1]),c[2]))
        labels=[c[0]+'−'+c[1]+'\nb='+f'{c[2]:g}' for c in conditions]
    if not conditions:return
    fig,axes=plt.subplots(1,2,figsize=(7,3.8),sharey=True)
    for ax,task in zip(axes,TASKS):
        for ki,kind in enumerate(kinds):
            offset=(ki-(len(kinds)-1)/2)*.22
            for ci,condition in enumerate(conditions):
                points=[r for r in rows if r['task']==task and r['metric']==score and tuple(r[k] for k in keys)==condition and (selected or r['checkpoint_kind']==kind)]
                stats=[r for r in statistics if r['task']==task and r[scorekey]==score and tuple(r[k] for k in keys)==condition and r[quantity]=='accuracy_left_minus_right' and (selected or r['checkpoint_kind']==kind)]
                require(len(points)==3 and len(stats)==1,'Missing fixed paired points or score-resolved summary')
                summary=stats[0]
                ax.errorbar(ci+offset,100*summary['mean'],yerr=100*summary['std_ddof1'],fmt='D',color=colors[kind],capsize=2,markersize=3.2,lw=1)
                for point in points:
                    seed=int(point['training_seed'] if selected else point['seed'])
                    ax.plot(ci+offset+(SEEDS.index(seed)-1)*.035,100*point['accuracy_left_minus_right'],marker=markers[seed],color=colors[kind],markersize=2.7,alpha=.7,lw=0)
        ax.axhline(0,color='.45',ls='--',lw=.8);ax.grid(axis='y',alpha=.2);ax.set_title(task,fontsize=9)
        ax.set_xticks(range(len(conditions)),labels,fontsize=6.5);ax.set_xlim(-.5,len(conditions)-.5)
        ax.set_ylabel('Accuracy left − right (pp); higher favors left')
    label='Each policy uses its own validation-selected λ' if selected else 'Frozen same-λ contrasts'
    fig.suptitle(prefix+'\n'+label+' | '+score,fontsize=9,y=.99)
    handles=[Line2D([],[],color=colors[k],marker='D',label=k+' mean±sample SD') for k in kinds]+[Line2D([],[],color='.4',marker=markers[s],lw=0,label='seed '+str(s),markersize=3) for s in SEEDS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.87),ncol=len(handles),frameon=False,fontsize=7)
    fig.text(.5,.015,'Same-seed differences; these are not extra training replicates or significance tests.',ha='center',fontsize=6.7)
    fig.subplots_adjust(left=.10,right=.985,bottom=.19,top=.73,wspace=.25)
    name=('selected_policy_pairs_' if selected else 'fixed_lambda_pairs_')+score
    for ext in ('png','pdf','svg'):fig.savefig(output/(name+'.'+ext),dpi=180)
    figures.append({'name':name,'width_inches':7,'height_inches':3.8});plt.close(fig)


def run(args):
    reader=SmallInputs();manifest,tables,family,phase,total,synthetic,bindings=load(args,reader)
    require(set(tables)==set(BASE_FILES+(SELECTED_FILES if family=='A' else ())), 'All declared tables required')
    verify_tables(manifest,tables,family)
    output=args.output.resolve();require(not output.exists(),'Use a new output namespace');output.mkdir(parents=True)
    reader.bytes(args.font)
    figures=plot(tables,family,phase,synthetic,output,args.font.resolve())
    for name in tables:
        # Preserve numerical rows and full precision, including conditional CIs.
        source=next(p for p in reader.content if Path(p).name==name+'.csv')
        (output/(name+'.csv')).write_bytes(reader.content[source])
    note=('SYNTHETIC: all displayed values are hand-made examples, not actual task results.\n\n' if synthetic else '')
    note+=f'# Behavior presentation: {phase}\n\nAll {total} logical endpoints, both tasks and both scores are retained. acc_norm is primary; acc is auxiliary.\n\n'
    note+='Figures show all three trained seed points and existing mean±sample SD. Same-λ pairs and each-method validation-selected-λ policy pairs are separate. No new CI, score, selection or inference is computed.\n\n'
    note+='All individual identities, actual best/final step and target/replay exposure, transitions and conditional Wilson/bootstrap outputs remain in the copied full-precision CSVs. Best exposure differs from full training cost; aliases/final/best/questions do not increase the training-seed count. Physical training cost must use the original actual_run_id/full_training_committed_path, not plotted exposure.\n\n'
    note+='Question intervals are conditional on fixed checkpoints and IID/exchangeable questions. Shared activities/contexts may be correlated; they are not training-repeat or automatic population CIs.\n\n'
    if family=='A':note+='A policy figures use selected_policy_score_resolved_statistics.csv (separate score/quantity). The original label-collision aggregate is never read. Each method has its own frozen validation λ; this does not isolate a fixed-λ objective effect.\n\n'
    note+='\n'.join('!['+f['name']+']('+f['name']+'.png)' for f in figures)+'\n'
    (output/'REPORT.md').write_text(note)
    reader.finish()
    outputs={str(p):sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    result={'complete':True,'synthetic_only':synthetic,'phase':phase,'family':family,'logical_endpoints':total,'endpoint_rows':len(tables['endpoints']),
        'figures':figures,'bindings':bindings,'source_files':reader.sources,'output_sha256':outputs,
        'source_sha256':sha(Path(__file__).read_bytes()),'new_inference_or_statistics':False,'original_predictions_or_checkpoints_opened':False,
        'scope':'Presentation of every fixed endpoint/score; original reviewed numerical CSV bytes copied unchanged; no new CIs, ranking, filtering or method selection'}
    (output/'COMPLETE.json').write_bytes(encoded(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--review-complete',type=Path);group.add_argument('--synthetic-fixture',type=Path)
    p.add_argument('--review-complete-sha256');p.add_argument('--output',type=Path,required=True);p.add_argument('--font',type=Path,default=DEFAULT_FONT)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if a.child:
        value=run(a);print(json.dumps({k:value[k] for k in ('complete','synthetic_only','logical_endpoints')}));return
    timer=a.output.with_name(a.output.name+'_whole_child_timing.json');require(not timer.exists() and not a.output.exists(),'New output/timer required')
    before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.perf_counter()
    argv=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--child']
    child=subprocess.run(argv,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    after=resource.getrusage(resource.RUSAGE_CHILDREN);timer.parent.mkdir(parents=True,exist_ok=True)
    timer.write_bytes(encoded({'argv':argv,'returncode':child.returncode,'wall_seconds':time.perf_counter()-started,'user_cpu_seconds':after.ru_utime-before.ru_utime,
        'system_cpu_seconds':after.ru_stime-before.ru_stime,'scope':'Complete presentation child with source/CSV proof checks, figures and output hashes; no original check/analyze/reviewer/scorer is invoked','nested_not_additive':True}))
    raise SystemExit(child.returncode)


if __name__=='__main__':main()
