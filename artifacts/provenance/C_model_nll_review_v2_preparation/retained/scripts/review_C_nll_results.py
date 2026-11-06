#!/usr/bin/env python3
"""Independent chunk-based review of one explicitly unsealed, COMPLETE C model.

Does not invoke a collector, model, scorer, or checkpoint reader. Bootstrap
verification reconstructs document multiplicities with a separate implementation.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np

METRICS = ('g_new', 'f_old_c4', 'f_old_wikitext')
SIX = METRICS + tuple(m+'_equal_document' for m in METRICS)
GROUPS = ('B0','B1','B2','B3','B4','D','M','C')
PAIRS = ('M-B4','C-B4','B2-B1','B4-B3','B3-B1','B4-B2')
SEEDS = (11,22,33)

def require(test, message):
    if not test: raise ValueError(message)

def sha(path):
    require('/checkpoints/' not in str(path), 'Checkpoint payload reads forbidden in report reviewer')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path): return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))
def write_json(path, value):
    with Path(path).open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def write_csv(path, values):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
def stats(values): return {'mean':statistics.mean(values),'std_ddof1':statistics.stdev(values),'n':len(values)}
def favorable(metric,x): return (1 if x>0 else -1 if x<0 else 0)*(1 if metric.startswith('g_new') else -1)

def documents(corpus):
    chunks=corpus['chunks'];values=defaultdict(list);counts=defaultdict(int);identities={}
    require(corpus['complete_corpus'] is True and chunks, 'Complete chunk corpus required')
    for c in chunks:
        d=str(c['doc_id']);key=(d,c['chunk_index']);n=c['valid_prediction_tokens']
        require(key not in identities and type(n) is int and n>0,'Duplicate/malformed chunk identity')
        require(math.isfinite(c['sum_nll']),'Finite chunk NLL required')
        identities[key]=n;values[d].append(c['sum_nll']);counts[d]+=n
    require(sum(counts.values())==corpus['valid_prediction_tokens'],'Chunk/token count mismatch')
    total=math.fsum(math.fsum(v) for v in values.values())
    require(abs(total-corpus['sum_nll'])<1e-7,'Chunk NLL sum mismatch')
    return {d:(math.fsum(values[d]),counts[d]) for d in sorted(values)},identities

def delta_vectors(a,b):
    da,ia=documents(a);db,ib=documents(b)
    require(ia==ib and da.keys()==db.keys(),'Paired corpus document/chunk/token identity differs')
    names=list(da);counts=np.asarray([da[d][1] for d in names],dtype=np.float64)
    require(all(da[d][1]==db[d][1] for d in names),'Within-document tokens differ')
    return names,counts,np.asarray([da[d][0]-db[d][0] for d in names],dtype=np.float64)

def raw_metrics(current, initial):
    result={}
    for c,m in zip(('target','c4','wikitext'),METRICS):
        _,n,d=delta_vectors(current[c],initial[c]);sign=-1 if c=='target' else 1
        result[m]=sign*math.fsum(d.tolist())/int(n.sum())
        result[m+'_equal_document']=sign*statistics.mean((d/n).tolist())
    return result

class Resampling:
    """Independent full draw -> integer multiplicities, never collector calls."""
    def __init__(self, names, counts, replicates=5000, seed=20261106):
        self.names=names;self.counts=counts
        draws=np.random.default_rng(seed).integers(0,len(names),size=(replicates,len(names)))
        self.multiplicity=np.stack([np.bincount(r,minlength=len(names)) for r in draws]).astype(np.float64)
        self.token_denominator=self.multiplicity@counts
    def interval(self, names, counts, delta, sign=1):
        require(names==self.names and np.array_equal(counts,self.counts),'Bootstrap paired document identities changed')
        token=self.multiplicity@delta/self.token_denominator
        equal=self.multiplicity@(delta/counts)/len(names)
        out={}
        for weighting,point,draws in [('token_weighted',math.fsum(delta.tolist())/int(counts.sum()),token),
                                      ('equal_document',statistics.mean((delta/counts).tolist()),equal)]:
            lo,hi=np.quantile(draws,[.025,.975])
            out[weighting]={'estimate':sign*point,'ci_low':float(lo if sign==1 else -hi),'ci_high':float(hi if sign==1 else -lo)}
        return out

def validate_grid(endpoints, expected_groups):
    actual=[(r['checkpoint'],r['group'],int(r['training_seed'])) for r in endpoints]
    expected={(k,g,s) for k in ('final','best') for g in expected_groups for s in SEEDS}
    require(len(actual)==len(expected) and len(set(actual))==len(actual) and set(actual)==expected,'Complete model final+best grid required')

def plot(endpoints,pairs,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8.4,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    labels=('G_new (higher is better)','F_C4 (lower is better)','F_Wiki (lower is better)')
    groups=[g for g in GROUPS if any(r['group']==g for r in endpoints)]
    colors={'B0':'#777777','B1':'#CC6677','B2':'#AA4499','B3':'#DDCC77','B4':'#4477AA','D':'#66CCEE','M':'#EE7733','C':'#228833'}
    for suffix,weighting in [('','Token weighted'),('_equal_document','Equal document')]:
        for paired in (False,True):
            fig,axes=plt.subplots(3,2,figsize=(7,6.55),sharey='row')
            categories=[p for p in PAIRS if any(r['comparison']==p for r in pairs)] if paired else groups
            for col,kind in enumerate(('final','best')):
                for row,metric in enumerate(METRICS):
                    ax=axes[row,col];field=metric+suffix
                    for x,label in enumerate(categories):
                        subset=sorted([r for r in (pairs if paired else endpoints) if r['checkpoint']==kind and r['comparison' if paired else 'group']==label],key=lambda r:int(r['training_seed']))
                        ys=[float(r[field]) for r in subset];color=colors[label.split('-')[0]]
                        for dx,y,marker in zip((-.14,0,.14),ys,('o','s','^')):ax.scatter(x+dx,y,marker=marker,s=24,color=color,edgecolors='white',linewidths=.4,zorder=3)
                        st=stats(ys);ax.errorbar(x+.27,st['mean'],yerr=st['std_ddof1'],fmt='D',markersize=3.3,capsize=2,color=color,zorder=4)
                    ax.axhline(0,color='.45',ls='--',lw=.7);ax.grid(axis='y',alpha=.2)
                    ax.set_xticks(range(len(categories)),categories,rotation=35 if paired else 0,fontsize=7.5)
                    if col==0:ax.set_ylabel(labels[row]+'\n(nats)')
                    if row==0:ax.set_title('Final: fixed full exposure' if kind=='final' else 'Best: target-validation selected')
            fig.suptitle(f'C 410M | {weighting} | '+('six frozen paired contrasts' if paired else 'all methods and checkpoints'),fontsize=10.5,y=.994)
            fig.text(.5,.012,'Circle/square/triangle: seeds 11/22/33; diamond +/- sample SD (n=3). Row-shared scales.\nFinal: 100M target; best: actual 35.026M/40.030M/55.041M. b=lambda=.05 except B0.\n'+('Differences = left minus right; G positive / F negative favors left.' if paired else 'D and B4 share verified states; both logical methods retained; initial = dashed zero.'),ha='center',fontsize=7.2)
            fig.tight_layout(rect=(0,.095,1,.97))
            name=('paired' if paired else 'all_endpoints')+('_equal_document' if suffix else '_token')+'_7in'
            for ext in ('png','pdf','svg'):fig.savefig(out/(name+'.'+ext),dpi=220)
            plt.close(fig)

def run(complete_path, expected_complete, expected_authority, output):
    start=time.perf_counter();complete_path=Path(complete_path);out=Path(output)
    require(sha(complete_path)==expected_complete,'COMPLETE SHA differs')
    complete=read(complete_path)
    require(complete['complete'] is True and complete['endpoints']==[{'endpoint':'final','logical_conditions':24},{'endpoint':'best','logical_conditions':24}],'Whole C410 final+best COMPLETE required')
    require(complete['authority']['sha256']==expected_authority,'Explicit ROOT authority SHA differs')
    for name in ('authority','plan','readiness','first_read_receipt','operational_repair'):
        p=complete[name];require(sha(p['path'])==p['sha256'],'Changed completion proof '+name)
    authority=read(complete['authority']['path']);plan=read(complete['plan']['path']);ready=read(complete['readiness']['path'])
    require(authority['allow_score_analysis'] is True and authority['plan']==complete['plan'] and authority['operational_repair']==complete['operational_repair'],'ROOT authority does not bind exact plan/repair')
    require(plan['contract_id']=='extended_pythia_410m_100m_nll' and ready['ready'] is True and ready['closure_job_count']==43 and not ready['pending_job_ids'],'Exact complete C410 closure required')
    require(sha(plan['manifest']['path'])==plan['manifest']['sha256'],'Manifest changed')
    manifest=read(plan['manifest']['path']);require(len(manifest['baseline_groups'])==1,'One matched baseline required')
    group=next(iter(manifest['baseline_groups'].values()));require(group['model_tag']=='pythia-410m' and manifest['phases'][0]['data_family']=='extended_v1','Matched extended model/data required')
    endpoints=[];old_pairs=[];sources={};bundles={};base=complete_path.parent
    for kind in ('final','best'):
        folder=base/kind;analysis=read(folder/'analysis_manifest.json')
        require(analysis['complete'] is True and analysis['completed_conditions']==analysis['expected_conditions']==24 and analysis['identity']['checkpoint']==kind and analysis['replicate_unit']=='training_seed','Incomplete/wrong endpoint analysis')
        require(analysis['bootstrap_replicates']==5000 and analysis['bootstrap_seed']==20261106,'Frozen bootstrap differs')
        for p,d in analysis['sources_sha256'].items():
            if p not in sources:require(sha(p)==d,'Changed analysis source '+p);sources[p]=d
            else:require(sources[p]==d,'Contradictory source proof')
        endpoints.extend(rows(folder/'endpoints.csv'));old_pairs.extend(rows(folder/'paired_effects.csv'));bundles[kind]=analysis
    validate_grid(endpoints,GROUPS);require(len(old_pairs)==36,'All six paired contrasts and seeds required')
    records={r['id']:r for r in manifest['records']};bykey={(r['checkpoint'],r['run_id']):r for r in endpoints};cache={}
    def score(path):
        if path not in cache:cache[path]=read(path)
        return cache[path]
    initial=score(group['initial_output'])['corpora'];contexts={}
    for c in ('target','c4','wikitext'):
        names,n,d=delta_vectors(initial[c],initial[c]);contexts[c]=Resampling(names,n)
    errors=[]
    def same(a,b):
        error=abs(float(a)-float(b));errors.append(error);require(math.isfinite(error) and error<1e-10,f'Independent numerical mismatch: {a} vs {b}')
    rebuilt=[];exposure=[];raw_corpora=[]
    for r in endpoints:
        record=records[r['record_id']];require(record['kind']==r['checkpoint'] and record['run_id']==r['run_id'],'Endpoint manifest identity differs')
        for csvkey,mkey in [('target_prediction_tokens','target_prediction_tokens_at_checkpoint'),('replay_prediction_tokens','replay_prediction_tokens_at_checkpoint')]:require(int(r[csvkey])==record[mkey],'Actual checkpoint exposure differs')
        current=score(record['output'])['corpora'];values=raw_metrics(current,initial)
        for metric,v in values.items():same(v,r[metric])
        rebuilt.append({**r,**values})
        exposure.append({k:r[k] for k in ('checkpoint','group','training_seed','run_id','actual_run_id','target_prediction_tokens','replay_prediction_tokens','full_training_target_prediction_tokens','full_training_replay_prediction_tokens')}|{'step':record['step'],'checkpoint_sha256':record['checkpoint_sha256']})
        for c in ('target','c4','wikitext'):
            corpus=current[c];docs,_=documents(corpus);n=corpus['valid_prediction_tokens'];value=math.fsum(v[0] for v in docs.values())/n;same(value,corpus['nll'])
            raw_corpora.append({'checkpoint':r['checkpoint'],'group':r['group'],'training_seed':r['training_seed'],'run_id':r['run_id'],'corpus':c,'prediction_tokens':n,'documents':len(docs),'nll':value,'ppl':math.exp(value),'kl':corpus['kl'],'equal_document_nll':statistics.mean(v[0]/v[1] for v in docs.values())})
    endpoint_stats=[];paired_stats=[];paired=[];rankings=[]
    for kind in ('final','best'):
        old={(r['group'],r['metric']):r for r in rows(base/kind/'endpoint_statistics.csv')}
        for g in GROUPS:
            for metric in SIX:
                ss=stats([float(r[metric]) for r in rebuilt if r['checkpoint']==kind and r['group']==g]);same(ss['mean'],old[g,metric]['mean']);same(ss['std_ddof1'],old[g,metric]['std_ddof1'])
                endpoint_stats.append({'checkpoint':kind,'group':g,'metric':metric,**ss})
        for metric in SIX:
            ordered=sorted([r for r in endpoint_stats if r['checkpoint']==kind and r['metric']==metric],key=lambda r:r['mean'],reverse=metric.startswith('g_new'))
            rankings.append({'checkpoint':kind,'metric':metric,'better_to_worse_mean':'>'.join(r['group'] for r in ordered),'tied_D_B4_states':True})
    for p in old_pairs:
        a=bykey[p['checkpoint'],p['run_a']];b=bykey[p['checkpoint'],p['run_b']]
        require(a['training_seed']==b['training_seed']==p['training_seed'],'Unmatched training seeds')
        item={**p}
        for metric in SIX:item[metric]=float(a[metric])-float(b[metric]);same(item[metric],p[metric])
        paired.append(item)
    for kind in ('final','best'):
        old={(r['comparison'],r['metric']):r for r in rows(base/kind/'paired_statistics.csv')}
        for comparison in PAIRS:
            for metric in SIX:
                rr=[r for r in paired if r['checkpoint']==kind and r['comparison']==comparison];vals=[r[metric] for r in rr];ss=stats(vals)
                same(ss['mean'],old[comparison,metric]['mean']);same(ss['std_ddof1'],old[comparison,metric]['std_ddof1'])
                paired_stats.append({'checkpoint':kind,'comparison':comparison,'metric':metric,**ss,'favorable_seeds':sum(favorable(metric,v)>0 for v in vals),'unfavorable_seeds':sum(favorable(metric,v)<0 for v in vals),'tied_seeds':sum(v==0 for v in vals),'seed11':vals[0],'seed22':vals[1],'seed33':vals[2]})
    endpoint_ci=[];paired_ci=[]
    for kind in ('final','best'):
        for is_pair,filename in [(False,'endpoint_document_bootstrap.json'),(True,'document_bootstrap.json')]:
            for old in read(base/kind/filename):
                if old['field']!='sum_nll':continue
                c=old['corpus'];metric=METRICS[('target','c4','wikitext').index(c)];sign=-1 if c=='target' else 1
                if is_pair:
                    a=bykey[kind,old['run_a']];b=bykey[kind,old['run_b']];av=score(records[a['record_id']]['output'])['corpora'][c];bv=score(records[b['record_id']]['output'])['corpora'][c]
                    identity={k:old[k] for k in ('comparison','training_seed','run_a','run_b','target_tokens_a','target_tokens_b','equal_checkpoint_exposure')}
                else:
                    a=bykey[kind,old['run_id']];av=score(records[a['record_id']]['output'])['corpora'][c];bv=initial[c]
                    identity={k:old[k] for k in ('run_id','training_seed','target_prediction_tokens','replay_prediction_tokens')};identity['group']=a['group']
                names,n,d=delta_vectors(av,bv);rebuilt_ci=contexts[c].interval(names,n,d,sign)
                for weighting,suffix in [('token_weighted',''),('equal_document','_equal_document')]:
                    original=old[weighting]
                    if is_pair and sign==-1:original={'estimate':-original['estimate'],'ci_low':-original['ci_high'],'ci_high':-original['ci_low']}
                    for key in ('estimate','ci_low','ci_high'):same(rebuilt_ci[weighting][key],original[key])
                    (paired_ci if is_pair else endpoint_ci).append({'checkpoint':kind,**identity,'metric':metric+suffix,**rebuilt_ci[weighting],'documents':len(names),'prediction_tokens':int(n.sum()),'bootstrap_replicates':5000,'bootstrap_seed':20261106,'scope':'conditional on fixed checkpoints and independent/exchangeable documents; not training-seed CI'})
    require(len(endpoint_ci)==288 and len(paired_ci)==216,'Complete six-metric conditional CI coverage required')
    training={};teacher=[];configs={};summaries={}
    for r in manifest['records']:
        key=r['actual_run_id'];full=r['full_training_committed_path'];entry={'actual_run_id':key,'group':r['group'],'training_seed':r['training_seed'],**full['parent_wall_seconds'],'target_prediction_tokens':full['target_prediction_tokens'],'replay_prediction_tokens':full['replay_prediction_tokens']}
        if key in training:require(training[key]==entry,'Training cost differs across endpoints')
        training[key]=entry
        if key not in summaries:
            summaries[key]=read(r['summary_path']);configs[key]=read(r['config_path']);s=summaries[key];require(s['status']=='completed' and s['config']==configs[key],'Incomplete/mismatched training summary')
            for name,v in full['parent_wall_seconds'].items():same(v,s['counters'][name])
            teacher.append({'actual_run_id':key,**{k:v for k,v in s['counters'].items() if k in ('training_teacher_forward_seconds','validation_teacher_forward_seconds','validation_kl_arithmetic_seconds')},'nested_not_additive':True})
    require(len(training)==24,'Count exactly 24 physical training runs')
    state=read(Path(plan['closure_jobs'][0]['definition']['argv'][1]).parents[1]/'logs/supplemental_job_state.json');inferences=[];alias_rows=[]
    for job in manifest['jobs']:
        att=read(Path(job['output']).with_name(Path(job['output']).stem+'_execution.json'));s=state['jobs'][job['id']]
        require(s['status']=='completed' and type(s['returncode']) is int and s['returncode']==0 and 'stop_requested_at' not in s,'Physical inference incomplete/stopped')
        inferences.append({'job_id':job['id'],'kind':job['kind'],'scheduler_whole_slot_seconds':s['elapsed_seconds'],'worker_verification_seconds':att['verification_wall_seconds'],'scorer_subprocess_wall_seconds':att['inference_subprocess_wall_seconds'],'teacher_forward_seconds_nested':sum(v['teacher_forward_seconds'] for v in score(job['output'])['corpora'].values()),'started_at_epoch':s['started_at_epoch'],'scorer_and_verification_nested_in_scheduler':True})
    for r in manifest['records']:
        job=next(j for j in manifest['jobs'] if j['id']==r['inference_job_id'])
        if r['output']!=job['output']:
            att=read(Path(r['output']).with_name(Path(r['output']).stem+'_execution.json'));require(att['inference_subprocess_wall_seconds']==0 and att['inference_alias'] is True,'Alias must add zero forward')
            alias_rows.append({'record_id':r['id'],'run_id':r['run_id'],'checkpoint':r['kind'],'canonical_job_id':job['id'],'inference_subprocess_wall_seconds':0,'source_model_equivalence':json.dumps(r['source_model_equivalence'],sort_keys=True)})
    require(len(inferences)==43 and len(alias_rows)==6,'Physical forward/alias scope differs')
    costs={'physical_training_runs':24,'training_parent_sums':{k:sum(v[k] for v in training.values()) for k in ('setup_seconds','training_seconds','validation_seconds','checkpoint_seconds')},'training_target_prediction_tokens':sum(r['target_prediction_tokens'] for r in training.values()),'training_replay_prediction_tokens':sum(r['replay_prediction_tokens'] for r in training.values()),'physical_inferences':43,'physical_adapted_inferences':42,'physical_initial_inferences':1,'zero_forward_aliases':6,'inference_scheduler_whole_slot_sum_seconds':sum(r['scheduler_whole_slot_seconds'] for r in inferences),'scorer_subprocess_wall_sum_seconds_nested':sum(r['scorer_subprocess_wall_seconds'] for r in inferences),'worker_verification_sum_seconds_nested':sum(r['worker_verification_seconds'] for r in inferences),'teacher_forward_sum_seconds_nested':sum(r['teacher_forward_seconds_nested'] for r in inferences),'whole_slot_is_not_exclusive_calendar_or_pure_GPU_time':True,'training_best_and_final_count_once':True,'shared_bank_preparation_reused_no_new_generation_assumed':sorted(set(c.get('replay_path') or 'B0:none' for c in configs.values()))}
    costs['training_committed_parent_sum_seconds']=sum(costs['training_parent_sums'].values())
    timers=Path('artifacts/provenance/C410_nll_analysis_actual_v3')
    costs['ROOT_check_whole_child']=read(timers/'check_whole_child_timing.json');costs['ROOT_analyze_whole_child']=read(timers/'analyze_whole_child_timing.json')
    require(not out.exists(),'Use new immutable report-review namespace');out.mkdir(parents=True)
    for name,data in [('endpoints_independent',rebuilt),('endpoint_statistics_independent',endpoint_stats),('paired_effects_independent',paired),('paired_statistics_independent',paired_stats),('mean_rankings',rankings),('endpoint_conditional_CIs_independent',endpoint_ci),('paired_conditional_CIs_independent',paired_ci),('actual_checkpoint_exposure',exposure),('raw_corpora',raw_corpora),('unique_training_costs',list(training.values())),('nested_teacher_training_costs',teacher),('physical_inference_costs',inferences),('zero_forward_aliases',alias_rows)]:write_csv(out/(name+'.csv'),data)
    write_json(out/'cost_accounting.json',costs);plot(rebuilt,paired,out)
    write_json(out/'independent_review_receipt.json',{'status':'passed','source':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'complete':{'path':str(complete_path.resolve()),'sha256':expected_complete},'authority':complete['authority'],'independent_numerical_checks':len(errors),'max_absolute_error':max(errors),'source_sha256':sources,'endpoint_records':48,'six_metrics':list(SIX),'paired_records':36,'endpoint_conditional_intervals':288,'paired_conditional_intervals':216,'bootstrap_verification':'Independent document multiplicity matrix, same frozen draws/seed/5000; no collector called and no new uncertainty estimand','inside_run_wall_seconds':time.perf_counter()-start,'no_GPU_or_checkpoint_reads':True,'other_models_or_behavior_read':False})
    print(json.dumps({'passed':True,'checks':len(errors),'max_error':max(errors),'output':str(out)}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--complete',required=True,type=Path);p.add_argument('--complete-sha256',required=True);p.add_argument('--authority-sha256',required=True);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.complete,a.complete_sha256,a.authority_sha256,a.output)
if __name__=='__main__':main()
