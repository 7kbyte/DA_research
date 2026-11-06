#!/usr/bin/env python3
"""Independent review of an explicitly ROOT-unsealed 1B/2.8B COMPLETE C phase.

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

def write_json(path, value):
    with Path(path).open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def write_csv(path, values):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]) if values else ['record_id','run_id','checkpoint','canonical_job_id','inference_subprocess_wall_seconds','source_model_equivalence']);w.writeheader();w.writerows(values)
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

def plot(endpoints,pairs,out,model,profile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8.4,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    labels=('G_new (higher is better)','F_C4 (lower is better)','F_Wiki (lower is better)')
    groups=[g for g in profile['groups'] if any(r['group']==g for r in endpoints)]
    colors={'B0':'#777777','B1':'#CC6677','B2':'#AA4499','B3':'#DDCC77','B4':'#4477AA','D':'#66CCEE','M':'#EE7733','C':'#228833'}
    for suffix,weighting in [('','Token weighted'),('_equal_document','Equal document')]:
        for paired in (False,True):
            fig,axes=plt.subplots(3,2,figsize=(7,6.55),sharey='row')
            categories=[p for p in profile['pairs'] if any(r['comparison']==p for r in pairs)] if paired else groups
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
            fig.suptitle(f'C {model} | {weighting} | '+(str(len(profile['pairs']))+' frozen paired contrasts' if paired else 'all methods and checkpoints'),fontsize=10.5,y=.994)
            best_exposure=sorted({int(r['target_prediction_tokens']) for r in endpoints if r['checkpoint']=='best'})
            best_range=f'{min(best_exposure)/1e6:.3f}--{max(best_exposure)/1e6:.3f}M'
            fig.text(.5,.012,'Circle/square/triangle: seeds 11/22/33; diamond +/- sample SD (n=3). Row-shared scales.\n'+f'Final: {profile["horizon"]/1e6:g}M target; best: actual range {best_range}; exact exposure in CSV.\n'+('Differences = left minus right; G positive / F negative favors left.' if paired else 'Every frozen logical method retained; aliases require complete-state proof; initial = dashed zero.'),ha='center',fontsize=7.2)
            fig.tight_layout(rect=(0,.095,1,.97))
            name=('paired' if paired else 'all_endpoints')+('_equal_document' if suffix else '_token')+'_7in'
            for ext in ('png','pdf','svg'):fig.savefig(out/(name+'.'+ext),dpi=220)
            plt.close(fig)


PROFILES = {
    'pythia-1b': {'phase': 'extended_pythia_1b_100m', 'groups': GROUPS,
                  'pairs': PAIRS, 'horizon': 100_000_000},
    'pythia-2.8b': {'phase': 'extended_pythia_2p8b_25m',
                    'groups': ('B0','B1','B2','B4','M'),
                    'pairs': ('M-B4','B2-B1','B4-B2'), 'horizon': 25_000_000},
}
BOUND = {}
IO = {'read_bytes': 0, 'paths': set(), 'semantic_json_paths': set()}
FORBIDDEN_SUFFIXES = {'.pt','.pth','.bin','.safetensors','.zip','.tar','.gz','.zst','.npz','.npy'}

def safe_bytes(path):
    p=Path(path).resolve()
    require(not (set(p.suffixes)&FORBIDDEN_SUFFIXES) and 'checkpoints' not in p.parts,
            'Checkpoint/base/archive/tensor payload reads forbidden')
    value=p.read_bytes();IO['read_bytes']+=len(value);IO['paths'].add(str(p));return value

def sha(path): return hashlib.sha256(safe_bytes(path)).hexdigest()
def read(path):
    value=safe_bytes(path);IO['semantic_json_paths'].add(str(Path(path).resolve()));return json.loads(value)
def rows(path): return list(csv.DictReader(safe_bytes(path).decode().splitlines()))
def canonical(path): return str(Path(path).resolve())
def proof(path): return {'path':canonical(path),'sha256':sha(path)}
def bind(value, opaque=False):
    require(isinstance(value,dict) and isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str),'Malformed proof')
    actual=sha(value['path']);require(actual==value['sha256'],'Changed bound source '+value['path'])
    path=canonical(value['path']);require(path not in BOUND or BOUND[path]==actual,'Contradictory bound source');BOUND[path]=actual
    return None if opaque else read(path)
def exact_proof(a,b): return a['sha256']==b['sha256'] and canonical(a['path'])==canonical(b['path'])
def definition_sha(job): return hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
def flag_value(argv,flag):
    require(argv.count(flag)==1 and argv.index(flag)+1<len(argv),'Missing/duplicate command flag '+flag)
    return argv[argv.index(flag)+1]
def attestation_path(output):
    p=Path(output);return str(p.with_name(p.stem+'_execution.json'))

def validate_pairs(values,groups,pairs):
    actual=[(r['checkpoint'],r['comparison'],int(r['training_seed'])) for r in values]
    expected={(k,p,s) for k in ('final','best') for p in pairs for s in SEEDS}
    require(len(actual)==len(expected) and len(set(actual))==len(actual) and set(actual)==expected,'Complete frozen contrast grid required')
    for r in values:
        require(r['comparison'].split('-')[0] in groups and r['comparison'].split('-')[1] in groups,'Foreign comparison group')

def validate_state(plan,queue,state):
    require(isinstance(queue,list) and len({j['id'] for j in queue})==len(queue),'Unique actual scheduler queue required')
    actual={j['id']:j for j in queue};closure=plan['closure_jobs']
    require(len({j['job_id'] for j in closure})==len(closure)==plan['closure_job_count'],'Duplicate/incomplete closure')
    closure_ids={j['job_id'] for j in closure}
    for item in closure:
        jid=item['job_id'];definition=item['definition']
        require(jid in actual and actual[jid]==definition and definition['id']==jid,'Foreign/changed actual queue definition '+jid)
        require(set(definition.get('depends_on',[]))<=closure_ids,'Missing baseline dependency closure')
        argv=definition['argv'];require(len(argv)>2 and Path(argv[1]).name=='run_frozen_evaluation.py','Expected frozen evaluation wrapper')
        require(flag_value(argv,'--job-id')==jid and flag_value(argv,'--manifest-sha256')==item['manifest']['sha256'] and canonical(flag_value(argv,'--manifest'))==canonical(item['manifest']['path']),'Actual wrapper manifest/job differs')
        r=state.get('jobs',{}).get(jid,{})
        require(r.get('status')=='completed' and type(r.get('returncode')) is int and r['returncode']==0,'Strict completed integer rc0 required '+jid)
        require(not any(k.startswith('stop_requested') for k in r) and not r.get('missing_outputs'),'Stopped/missing-output closure '+jid)
        require(r.get('definition_sha256')==definition_sha(definition) and r.get('argv')==argv,'Foreign state definition/actual argv '+jid)
        require(isinstance(r.get('elapsed_seconds'),(float,int)) and r['elapsed_seconds']>=0 and isinstance(r.get('started_at_epoch'),(float,int)),'Missing actual whole-slot timing')
        require(all(Path(x).is_file() for x in definition.get('outputs',[])),'Missing declared closure output')
    return actual

def validate_alias(record,job,manifest,model):
    eq=record.get('source_model_equivalence')
    require(isinstance(eq,dict),'Alias requires explicit full-model proof')
    value=bind({'path':eq['proof'],'sha256':eq['proof_sha256']})
    require(value.get('status','passed')=='passed' and value.get('model',model)==model,'Wrong/failed equivalence proof model')
    require(value['algorithm']==eq['algorithm']=='sha256_sorted_key_dtype_shape_contiguous_uint8_v1','Unknown full-state algorithm')
    for key in ('source_audit','source_registry','source_snapshot'):
        if key in value: bind(value[key])
    for key in ('algorithm_source','producer_source'):
        if key in value: bind(value[key],opaque=True)
    index=eq['comparison_index'];require(type(index) is int and 0<=index<len(value['comparisons']),'Bad equivalence index')
    pair=value['comparisons'][index];require(pair.get('bitwise_model_state_equal') is True,'Unproved/nonidentical alias')
    left,right=pair['left'],pair['right'];fp=eq['fingerprint']
    require(left['model_state_fingerprint']==right['model_state_fingerprint']==fp,'Incomplete/mismatched model fingerprint')
    require(eq['left_checkpoint_sha256']==left['checkpoint_sha256'] and eq['right_checkpoint_sha256']==right['checkpoint_sha256'],'Wrong equivalence CP SHA')
    canonical_records=[r for r in manifest['records'] if r['inference_job_id']==job['id'] and r['output']==job['output']]
    require(len(canonical_records)==1,'Alias requires one canonical physical record')
    canonical_record=canonical_records[0]
    for r in (record,canonical_record):
        matches=[s for s in (left,right) if canonical(s['checkpoint'])==canonical(r['checkpoint']) and s['checkpoint_sha256']==r['checkpoint_sha256']]
        require(len(matches)==1 and r['weights_identity']=='model_state:'+fp['sha256_model_state'],'Proof does not cover actual endpoint CP')
        side=matches[0]
        require(side.get('run_id',r['run_id'])==r['run_id'] and side.get('kind',r['kind'])==r['kind'],'Proof selected endpoint differs')
        if 'checkpoint_exposure' in side:
            require(all(side['checkpoint_exposure'][k]==r[k] for k in ('step','target_prediction_tokens_at_checkpoint','replay_prediction_tokens_at_checkpoint')),'Proof exposure differs')
        for k in ('config','summary'):
            if k+'_sha256' in side:
                require(canonical(side[k+'_path'])==canonical(r[k+'_path']) and sha(r[k+'_path'])==side[k+'_sha256'],'Proof source binding differs')
    return eq

def validate_timer(value,command,complete,plan,state_path):
    timer=bind(value);argv=timer.get('argv',[])
    require(type(timer.get('returncode')) is int and timer['returncode']==0 and not timer.get('timed_out',False),'Timer must describe successful complete child')
    require(len(argv)>2 and Path(argv[1]).name=='analyze_supplemental_phase_v3.py' and argv[2]==command,'Wrong actual analysis/check timer argv')
    for flag,p in (('--plan',complete['plan']),('--operational-repair',complete['operational_repair'])):
        require(canonical(flag_value(argv,flag))==canonical(p['path']) and flag_value(argv,flag+'-sha256')==p['sha256'],'Timer source/plan mismatch')
    require(canonical(flag_value(argv,'--state'))==canonical(state_path),'Timer actual scheduler state differs')
    if command=='analyze':
        require(canonical(flag_value(argv,'--authority'))==canonical(complete['authority']['path']) and flag_value(argv,'--authority-sha256')==complete['authority']['sha256'],'Analyze timer authority differs')
    require(math.isfinite(timer['wall_seconds']) and timer['wall_seconds']>=0,'Missing full child timer')
    return timer

def gate_inputs(args):
    profile=PROFILES[args.model];n=len(profile['groups'])*len(SEEDS)
    complete=bind({'path':str(args.complete),'sha256':args.complete_sha256})
    require(complete.get('complete') is True and complete['endpoints']==[{'endpoint':'final','logical_conditions':n},{'endpoint':'best','logical_conditions':n}],'Whole model final+best COMPLETE required')
    require(complete['authority']['sha256']==args.authority_sha256,'Explicit ROOT authority SHA differs')
    authority,plan,ready,journal,repair=[bind(complete[k]) for k in ('authority','plan','readiness','first_read_receipt','operational_repair')]
    cid=profile['phase']+'_nll'
    require(plan['contract_id']==cid and authority['contract_id']==cid and ready['contract_id']==cid,'Wrong model/phase contract')
    require(authority.get('authorized_by')=='ROOT' and authority.get('allow_score_analysis') is True and authority.get('allow_partial') is False,'Exact whole-phase ROOT score authority required')
    for key in ('plan','operational_repair'):
        require(exact_proof(authority[key],complete[key]),'Authority plan/repair mismatch')
    require(exact_proof(authority['manifest'],plan['manifest']) and exact_proof(authority['ready'],plan['ready']),'Authority manifest/READY mismatch')
    require(exact_proof(ready['plan'],complete['plan']) and exact_proof(ready['operational_repair'],complete['operational_repair']),'Readiness source mismatch')
    require(ready.get('ready') is True and not ready['pending_job_ids'] and ready['closure_job_count']==plan['closure_job_count'],'Incomplete whole closure')
    require(ready.get('score_json_parsed') is False and ready.get('all_source_baseline_alias_output_integrity_verified') is True,'Missing pre-semantic integrity gate')
    for key in ('plan','authority','readiness'):
        require(exact_proof(journal[key],complete[key]),'First-read journal binding differs')
    require(canonical(complete['first_read_receipt']['path'])==canonical(plan['first_read_receipt_path']) and journal.get('before_first_score_parse') is True and journal.get('complete_phase_source_and_output_SHA_verified') is True and bool(journal.get('semantic_access_enabled_at_utc')),'Whole-phase first-read journal required')
    require(journal.get('no_test_or_behavior_selection') is True,'Missing no-selection first-read declaration')
    for a,b in (('retained_plan',complete['plan']),('retained_manifest',plan['manifest']),('retained_protocol',plan['protocol'])):
        require(exact_proof(repair[a],b),'Repair retained source differs')
    require(repair.get('this_is_not_a_ROOT_score_authority') is True and repair.get('score_access_authorized') is False and repair.get('scientific_or_statistics_changes') is False,'Repair cannot grant scientific/score changes')
    require(canonical(repair['actual_first_read_journal_path_preserved'])==canonical(plan['first_read_receipt_path']),'Repair moved journal')
    bind(plan['protocol']);actual_ready=bind(plan['ready']);manifest=bind(plan['manifest'])
    require(actual_ready.get('ready') is True and exact_proof(actual_ready['manifest'],plan['manifest']),'Actual READY/manifest differs')
    phase=plan['contract']['phase']
    require(phase['name']==profile['phase'] and phase['model_tag']==args.model and phase['data_family']=='extended_v1' and phase['role']=='extended' and phase['rows']==n and phase['expected_phase']=='formal','Wrong complete model scope')
    require(complete['parsed_phase']==phase and plan['contract']['logical_endpoints']==2*n and plan['contract']['checkpoints']==['final','best'],'Changed parsed phase/endpoints')
    require(len(manifest['phases'])==1 and manifest['phases'][0]['name']==profile['phase'] and manifest['phases'][0]['model_tag']==args.model and manifest['phases'][0]['data_family']=='extended_v1' and manifest['mode']=='nll','Foreign manifest phase/model')
    require(len(manifest['baseline_groups'])==1,'One matched extended baseline required');group=next(iter(manifest['baseline_groups'].values()))
    identity=plan['contract']['scientific_identity']
    require(group['model_tag']==args.model and identity['model_tag']==args.model and identity['data_family']=='extended_v1','Matched model/data baseline required')
    require(group['test_paths']=={k:v['path'] for k,v in identity['test_data'].items()} and group['evaluation_input_sha256']=={v['path']:v['sha256'] for v in identity['test_data'].values()},'Main-v3/foreign baseline cannot replace matched extended initial')
    records=manifest['records'];validate_grid([{'checkpoint':r['kind'],'group':r['group'],'training_seed':r['training_seed']} for r in records],profile['groups'])
    require(len({r['id'] for r in records})==2*n and len({r['actual_run_id'] for r in records})==n,'Duplicate/aliased physical training identity')
    for r in records:
        require(r['model_tag']==args.model and r['phase_name']==profile['phase'] and r['data_family']=='extended_v1' and r['phase']=='formal' and r['is_training_alias'] is False and r['run_id']==r['actual_run_id'],'Foreign/aliased full training record')
        require(r['baseline_group']==group['id'] and r['initial_job_id']==group['initial_job_id'],'Unmatched initial group')
        full=r['full_training_committed_path'];require(full['target_prediction_tokens']==profile['horizon'],'Wrong full scientific horizon')
        if r['kind']=='final': require(r['target_prediction_tokens_at_checkpoint']==profile['horizon'],'Partial final exposure')
        for key in ('config_path','summary_path'):
            require(r['input_sha256'].get(r[key])==sha(r[key]),'Config/summary not bound to frozen manifest')
    require(canonical(args.state)==canonical(ready['scheduler_state']) and args.state_sha256==ready['scheduler_state_sha256'],'Explicit immutable state must be bound actual successor snapshot')
    queue=bind({'path':str(args.queue),'sha256':args.queue_sha256});state=bind({'path':str(args.state),'sha256':args.state_sha256});validate_state(plan,queue,state)
    require(len({j['id'] for j in manifest['jobs']})==len(manifest['jobs']) and all(j['id'] in {x['job_id'] for x in plan['closure_jobs']} for j in manifest['jobs']),'Manifest jobs outside actual closure')
    jobs={j['id']:j for j in manifest['jobs']};require(jobs[group['initial_job_id']]['kind']=='initial' and jobs[group['initial_job_id']]['output']==group['initial_output'],'Matched extended initial job missing')
    require(sum(j['kind']=='initial' for j in manifest['jobs'])==1,'Exactly one matched initial physical job required')
    attestations={canonical(v['attestation']['path']):v for v in ready['attestations']}
    expected_att={canonical(attestation_path(x['output'])):x['output'] for x in [*manifest['jobs'],*records]}
    require(set(attestations)==set(expected_att) and len(attestations)==len(ready['attestations']),'Incomplete/extra whole-phase attestations')
    # Hash score bytes only after explicit whole-phase unseal metadata; parse them later.
    for path,evidence in attestations.items():
        require(evidence['manifest_sha256']==plan['manifest']['sha256'],'Foreign attestation manifest')
        att=bind(evidence['attestation']);require(att['manifest_sha256']==plan['manifest']['sha256'] and canonical(att['manifest'])==canonical(plan['manifest']['path']) and att['job_id']==evidence['job_id'],'Foreign attestation identity')
        require(att['job_id'] in jobs and att['actual_argv']==jobs[att['job_id']]['argv'],'Attestation actual scorer argv differs')
        require(att['output_sha256']==evidence['output_sha256'],'Attested output hash map changed')
        require(set(att['output_sha256'])=={expected_att[path]},'Attestation does not cover its actual score output')
        for p,d in att['output_sha256'].items(): bind({'path':p,'sha256':d},opaque=True)
    alias_count=0
    for r in records:
        require(r['inference_job_id'] in jobs,'Foreign physical inference reference');job=jobs[r['inference_job_id']]
        if r['output']!=job['output']:
            validate_alias(r,job,manifest,args.model);att=read(attestation_path(r['output']))
            require(att.get('inference_alias') is True and att['inference_subprocess_wall_seconds']==0 and att['source_model_equivalence']==r['source_model_equivalence'],'Alias must add zero verified forward');alias_count+=1
    require(len(manifest['jobs'])==1+2*n-alias_count,'Physical work does not close logical endpoints')
    timers={kind:validate_timer({'path':str(getattr(args,kind+'_timing')),'sha256':getattr(args,kind+'_timing_sha256')},kind,complete,plan,args.state) for kind in ('check','analyze')}
    require(canonical(flag_value(timers['analyze']['argv'],'--output'))==canonical(Path(args.complete).parent),'Analyze timer output differs')
    return {'complete':complete,'plan':plan,'ready':ready,'manifest':manifest,'group':group,'state':state,'profile':profile,'aliases':alias_count,'timers':timers}

def run(args):
    start=time.perf_counter();complete_path=Path(args.complete);out=Path(args.output)
    gate=gate_inputs(args);complete=gate['complete'];plan=gate['plan'];ready=gate['ready']
    manifest=gate['manifest'];group=gate['group'];profile=gate['profile'];groups=profile['groups'];pairs=profile['pairs']
    run_count=len(groups)*len(SEEDS);physical_count=len(manifest['jobs']);alias_count=gate['aliases']
    endpoints=[];old_pairs=[];sources={};bundles={};base=complete_path.parent
    for kind in ('final','best'):
        folder=base/kind;analysis=read(folder/'analysis_manifest.json')
        require(analysis['complete'] is True and analysis['completed_conditions']==analysis['expected_conditions']==run_count and analysis['identity']['checkpoint']==kind and analysis['replicate_unit']=='training_seed','Incomplete/wrong endpoint analysis')
        require(analysis['identity']=={'role':'extended','phase_name':profile['phase'],'model_tag':args.model,'data_family':'extended_v1','checkpoint':kind},'Foreign model analysis identity')
        require(exact_proof(analysis['whole_phase_first_read_receipt'],complete['first_read_receipt']) and exact_proof(analysis['whole_phase_readiness'],complete['readiness']),'Analysis output journal/readiness differs')
        require(analysis['bootstrap_replicates']==5000 and analysis['bootstrap_seed']==20261106,'Frozen bootstrap differs')
        for p,d in analysis['sources_sha256'].items():
            if p not in sources:bind({'path':p,'sha256':d},opaque=True);sources[p]=d
            else:require(sources[p]==d,'Contradictory source proof')
        endpoints.extend(rows(folder/'endpoints.csv'));old_pairs.extend(rows(folder/'paired_effects.csv'));bundles[kind]=analysis
    validate_grid(endpoints,groups);validate_pairs(old_pairs,groups,pairs)
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
        record=records[r['record_id']];require(record['kind']==r['checkpoint'] and record['run_id']==r['run_id'] and record['group']==r['group'] and str(record['training_seed'])==r['training_seed'] and record['actual_run_id']==r['actual_run_id'],'Endpoint manifest identity differs')
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
        for g in groups:
            for metric in SIX:
                ss=stats([float(r[metric]) for r in rebuilt if r['checkpoint']==kind and r['group']==g]);same(ss['mean'],old[g,metric]['mean']);same(ss['std_ddof1'],old[g,metric]['std_ddof1'])
                endpoint_stats.append({'checkpoint':kind,'group':g,'metric':metric,**ss})
        for metric in SIX:
            ordered=sorted([r for r in endpoint_stats if r['checkpoint']==kind and r['metric']==metric],key=lambda r:r['mean'],reverse=metric.startswith('g_new'))
            rankings.append({'checkpoint':kind,'metric':metric,'better_to_worse_mean':'>'.join(r['group'] for r in ordered),'rank_is_descriptive_no_new_inference':True})
    for p in old_pairs:
        a=bykey[p['checkpoint'],p['run_a']];b=bykey[p['checkpoint'],p['run_b']];require(p['comparison']==a['group']+'-'+b['group'],'Pair label/run identity differs')
        require(a['training_seed']==b['training_seed']==p['training_seed'],'Unmatched training seeds')
        item={**p}
        for metric in SIX:item[metric]=float(a[metric])-float(b[metric]);same(item[metric],p[metric])
        paired.append(item)
    for kind in ('final','best'):
        old={(r['comparison'],r['metric']):r for r in rows(base/kind/'paired_statistics.csv')}
        for comparison in pairs:
            for metric in SIX:
                rr=sorted([r for r in paired if r['checkpoint']==kind and r['comparison']==comparison],key=lambda r:int(r['training_seed']));vals=[r[metric] for r in rr];ss=stats(vals)
                same(ss['mean'],old[comparison,metric]['mean']);same(ss['std_ddof1'],old[comparison,metric]['std_ddof1'])
                paired_stats.append({'checkpoint':kind,'comparison':comparison,'metric':metric,**ss,'favorable_seeds':sum(favorable(metric,v)>0 for v in vals),'unfavorable_seeds':sum(favorable(metric,v)<0 for v in vals),'tied_seeds':sum(v==0 for v in vals),'seed11':vals[0],'seed22':vals[1],'seed33':vals[2]})
    endpoint_ci=[];paired_ci=[];seen_ci=set()
    for kind in ('final','best'):
        for is_pair,filename in [(False,'endpoint_document_bootstrap.json'),(True,'document_bootstrap.json')]:
            for old in read(base/kind/filename):
                if old['field']!='sum_nll':continue
                c=old['corpus'];ci_key=(kind,is_pair,old.get('run_a',old.get('run_id')),old.get('run_b'),c);require(ci_key not in seen_ci,'Duplicate conditional CI row');seen_ci.add(ci_key);metric=METRICS[('target','c4','wikitext').index(c)];sign=-1 if c=='target' else 1
                if is_pair:
                    a=bykey[kind,old['run_a']];b=bykey[kind,old['run_b']];av=score(records[a['record_id']]['output'])['corpora'][c];bv=score(records[b['record_id']]['output'])['corpora'][c]
                    require(old['comparison'] in pairs and old['comparison']==a['group']+'-'+b['group'] and int(old['training_seed'])==int(a['training_seed'])==int(b['training_seed']),'CI comparison/seed identity differs')
                    require(int(old['target_tokens_a'])==int(a['target_prediction_tokens']) and int(old['target_tokens_b'])==int(b['target_prediction_tokens']) and old['equal_checkpoint_exposure']==(int(a['target_prediction_tokens'])==int(b['target_prediction_tokens'])),'CI actual paired exposure differs')
                    identity={k:old[k] for k in ('comparison','training_seed','run_a','run_b','target_tokens_a','target_tokens_b','equal_checkpoint_exposure')}
                else:
                    a=bykey[kind,old['run_id']];av=score(records[a['record_id']]['output'])['corpora'][c];bv=initial[c]
                    require(all(int(old[k])==int(a[k]) for k in ('training_seed','target_prediction_tokens','replay_prediction_tokens')),'Endpoint CI actual identity/exposure differs')
                    identity={k:old[k] for k in ('run_id','training_seed','target_prediction_tokens','replay_prediction_tokens')};identity['group']=a['group']
                names,n,d=delta_vectors(av,bv);rebuilt_ci=contexts[c].interval(names,n,d,sign)
                for weighting,suffix in [('token_weighted',''),('equal_document','_equal_document')]:
                    original=old[weighting]
                    if is_pair and sign==-1:original={'estimate':-original['estimate'],'ci_low':-original['ci_high'],'ci_high':-original['ci_low']}
                    for key in ('estimate','ci_low','ci_high'):same(rebuilt_ci[weighting][key],original[key])
                    (paired_ci if is_pair else endpoint_ci).append({'checkpoint':kind,**identity,'metric':metric+suffix,**rebuilt_ci[weighting],'documents':len(names),'prediction_tokens':int(n.sum()),'bootstrap_replicates':5000,'bootstrap_seed':20261106,'scope':'conditional on fixed checkpoints and independent/exchangeable documents; not training-seed CI'})
    require(len(endpoint_ci)==2*run_count*6 and len(paired_ci)==2*len(pairs)*len(SEEDS)*6,'Complete six-metric conditional CI coverage required')
    training={};teacher=[];configs={};summaries={}
    for r in manifest['records']:
        key=r['actual_run_id'];full=r['full_training_committed_path'];entry={'actual_run_id':key,'group':r['group'],'training_seed':r['training_seed'],**full['parent_wall_seconds'],'target_prediction_tokens':full['target_prediction_tokens'],'replay_prediction_tokens':full['replay_prediction_tokens']}
        if key in training:require(training[key]==entry,'Training cost differs across endpoints')
        training[key]=entry
        if key not in summaries:
            summaries[key]=read(r['summary_path']);configs[key]=read(r['config_path']);s=summaries[key];require(s['status']=='completed' and s['config']==configs[key],'Incomplete/mismatched training summary')
            for name,v in full['parent_wall_seconds'].items():same(v,s['counters'][name])
            teacher.append({'actual_run_id':key,**{k:v for k,v in s['counters'].items() if k in ('training_teacher_forward_seconds','validation_teacher_forward_seconds','validation_kl_arithmetic_seconds')},'nested_not_additive':True})
    require(len(training)==run_count,'Count every real full training once across final/best')
    state=gate['state'];inferences=[];alias_rows=[]
    for job in manifest['jobs']:
        att=read(Path(job['output']).with_name(Path(job['output']).stem+'_execution.json'));s=state['jobs'][job['id']]
        require(s['status']=='completed' and type(s['returncode']) is int and s['returncode']==0 and 'stop_requested_at' not in s,'Physical inference incomplete/stopped')
        inferences.append({'job_id':job['id'],'kind':job['kind'],'scheduler_whole_slot_seconds':s['elapsed_seconds'],'worker_verification_seconds':att['verification_wall_seconds'],'scorer_subprocess_wall_seconds':att['inference_subprocess_wall_seconds'],'teacher_forward_seconds_nested':sum(v['teacher_forward_seconds'] for v in score(job['output'])['corpora'].values()),'started_at_epoch':s['started_at_epoch'],'scorer_and_verification_nested_in_scheduler':True})
    for r in manifest['records']:
        job=next(j for j in manifest['jobs'] if j['id']==r['inference_job_id'])
        if r['output']!=job['output']:
            att=read(Path(r['output']).with_name(Path(r['output']).stem+'_execution.json'));require(att['inference_subprocess_wall_seconds']==0 and att['inference_alias'] is True,'Alias must add zero forward')
            alias_rows.append({'record_id':r['id'],'run_id':r['run_id'],'checkpoint':r['kind'],'canonical_job_id':job['id'],'inference_subprocess_wall_seconds':0,'source_model_equivalence':json.dumps(r['source_model_equivalence'],sort_keys=True)})
    require(len(inferences)==physical_count and len(alias_rows)==alias_count,'Physical forward/alias scope differs')
    costs={'physical_training_runs':run_count,'training_parent_sums':{k:sum(v[k] for v in training.values()) for k in ('setup_seconds','training_seconds','validation_seconds','checkpoint_seconds')},'training_target_prediction_tokens':sum(r['target_prediction_tokens'] for r in training.values()),'training_replay_prediction_tokens':sum(r['replay_prediction_tokens'] for r in training.values()),'physical_inferences':physical_count,'physical_adapted_inferences':physical_count-1,'physical_initial_inferences':1,'zero_forward_aliases':alias_count,'inference_scheduler_whole_slot_sum_seconds':sum(r['scheduler_whole_slot_seconds'] for r in inferences),'scorer_subprocess_wall_sum_seconds_nested':sum(r['scorer_subprocess_wall_seconds'] for r in inferences),'worker_verification_sum_seconds_nested':sum(r['worker_verification_seconds'] for r in inferences),'teacher_forward_sum_seconds_nested':sum(r['teacher_forward_seconds_nested'] for r in inferences),'whole_slot_is_not_exclusive_calendar_or_pure_GPU_time':True,'training_best_and_final_count_once':True,'shared_bank_preparation_reused_no_new_generation_assumed':sorted(set(c.get('replay_path') or 'B0:none' for c in configs.values()))}
    costs['training_committed_parent_sum_seconds']=sum(costs['training_parent_sums'].values())
    costs['ROOT_check_whole_child']=gate['timers']['check'];costs['ROOT_analyze_whole_child']=gate['timers']['analyze']
    require(not out.exists(),'Use new immutable report-review namespace');out.mkdir(parents=True)
    for name,data in [('endpoints_independent',rebuilt),('endpoint_statistics_independent',endpoint_stats),('paired_effects_independent',paired),('paired_statistics_independent',paired_stats),('mean_rankings',rankings),('endpoint_conditional_CIs_independent',endpoint_ci),('paired_conditional_CIs_independent',paired_ci),('actual_checkpoint_exposure',exposure),('raw_corpora',raw_corpora),('unique_training_costs',list(training.values())),('nested_teacher_training_costs',teacher),('physical_inference_costs',inferences),('zero_forward_aliases',alias_rows)]:write_csv(out/(name+'.csv'),data)
    write_json(out/'cost_accounting.json',costs);plot(rebuilt,paired,out,args.model,profile)
    for p,d in BOUND.items():require(sha(p)==d,'Bound metadata changed during review: '+p)
    write_json(out/'independent_review_receipt.json',{'status':'passed','source':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},'complete':{'path':str(complete_path.resolve()),'sha256':args.complete_sha256},'authority':complete['authority'],'independent_numerical_checks':len(errors),'max_absolute_error':max(errors),'source_sha256':sources,'endpoint_records':2*run_count,'six_metrics':list(SIX),'paired_records':2*len(pairs)*len(SEEDS),'endpoint_conditional_intervals':2*run_count*6,'paired_conditional_intervals':2*len(pairs)*len(SEEDS)*6,'bootstrap_verification':'Independent document multiplicity matrix, same frozen draws/seed/5000; no collector called and no new uncertainty estimand','inside_run_wall_seconds':time.perf_counter()-start,'no_GPU_or_checkpoint_reads':True,'other_models_or_behavior_read':False,'model':args.model,'physical_inferences':physical_count,'zero_forward_aliases':alias_count,'bound_metadata_sha256':dict(BOUND),'actual_scheduler_queue':{'path':canonical(args.queue),'sha256':args.queue_sha256},'actual_scheduler_state':{'path':canonical(args.state),'sha256':args.state_sha256},'IO':{'serialized_bytes_read':IO['read_bytes'],'paths':sorted(IO['paths']),'semantic_json_paths':sorted(IO['semantic_json_paths']),'checkpoint_base_archive_payload_reads':0}})
    print(json.dumps({'passed':True,'checks':len(errors),'max_error':max(errors),'output':str(out)}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=tuple(PROFILES))
    for name in ('complete','queue','state','check-timing','analyze-timing'):
        p.add_argument('--'+name,required=True,type=Path)
        p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--authority-sha256',required=True)
    p.add_argument('--output',required=True,type=Path)
    run(p.parse_args())
if __name__=='__main__':main()
