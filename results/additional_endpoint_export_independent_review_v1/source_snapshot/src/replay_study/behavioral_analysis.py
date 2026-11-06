"""CPU-only, manifest-bound behavioral summaries and conditional uncertainty."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDITIONAL_SCOPE = 'Conditional on the fixed trained checkpoint(s); question sampling does not create training replicates.'
PAIRS = (('M','B4'),('C','B4'),('B2','B1'),('B4','B3'),('D','B4'))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8<<20),b''):value.update(block)
    return value.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def jsonl(path):return [json.loads(line) for line in Path(path).open() if line.strip()]


def align(before,after):
    def index(rows):
        result={(row['task'],row['item_id']):row for row in rows}
        if not result or len(result)!=len(rows):raise ValueError('Nonempty unique task/item identities required')
        return result
    a,b=index(before),index(after)
    if a.keys()!=b.keys():raise ValueError('Paired task/item sets differ')
    keys=sorted(a)
    if any(a[k][f]!=b[k][f] for k in keys for f in ('fingerprint','gold')):
        raise ValueError('Paired task fingerprint/gold differs')
    return [a[k] for k in keys],[b[k] for k in keys]


def wilson(correct,total,z=1.959963984540054):
    if total==0:return {'estimate':None,'ci_low':None,'ci_high':None,'n':0}
    if not 0<=correct<=total:raise ValueError('Invalid binomial counts')
    p=correct/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return {'estimate':p,'ci_low':max(0.,center-half),'ci_high':min(1.,center+half),'n':total}


def paired_binary_summary(before,after,metric,n_bootstrap=5000,seed=20261106):
    """Multinomial transition draws equal iid paired-example bootstrap draws."""
    if metric not in ('acc_norm','acc') or n_bootstrap<1:raise ValueError('Valid metric and positive bootstrap count required')
    before,after=align(before,after)
    counts={name:0 for name in ('CC','CW','WC','WW')}
    for a,b in zip(before,after):
        x,y=a['correct_'+metric],b['correct_'+metric]
        if x not in (0,1) or y not in (0,1):raise ValueError('Correctness must be binary')
        counts[('C' if x else 'W')+('C' if y else 'W')]+=1
    n=len(before);initial=counts['CC']+counts['CW'];current=counts['CC']+counts['WC']
    sampled=np.random.default_rng(seed).multinomial(n,[counts[k]/n for k in counts],size=n_bootstrap)
    deltas=(sampled[:,2]-sampled[:,1])/n
    low,high=np.quantile(deltas,[.025,.975])
    return {'items':n,'counts':counts,'initial_accuracy':initial/n,
            'accuracy':wilson(current,n),'accuracy_delta':(counts['WC']-counts['CW'])/n,
            'delta_ci_low':float(low),'delta_ci_high':float(high),
            'conditional_forgetting':wilson(counts['CW'],initial),
            'conditional_acquisition':wilson(counts['WC'],n-initial),
            'bootstrap_replicates':n_bootstrap,'bootstrap_seed':seed,
            'bootstrap_method':'iid paired-example percentile; exact multinomial transition-count implementation',
            'uncertainty_scope':CONDITIONAL_SCOPE}


def mean_sd(values):
    a=np.asarray([v for v in values if v is not None],dtype=float)
    return {'mean':float(a.mean()) if len(a) else None,
            'std_ddof1':float(a.std(ddof=1)) if len(a)>1 else None,'n_training_seeds':len(a)}


def grouped_seed_summary(rows):
    cells=defaultdict(list)
    keys=('checkpoint_kind','group','ratio','weight','learning_rate','task','metric')
    for row in rows:
        if row['phase']=='formal':cells[tuple(row[k] for k in keys)].append(row)
    result=[]
    for cell,items in sorted(cells.items()):
        if len({r['seed'] for r in items})!=len(items):raise ValueError('Duplicate training seed within behavioral cell')
        for metric in ('accuracy','accuracy_delta','conditional_forgetting_rate','conditional_acquisition_rate'):
            result.append({**dict(zip(keys,cell)),'quantity':metric,**mean_sd(r[metric] for r in items),
                           'training_seeds':','.join(str(r['seed']) for r in sorted(items,key=lambda r:r['seed'])),
                           'scope':'Across independent training seeds; SD uses ddof=1; questions are not training replicates.'})
    return result


def paired_seed_summary(rows):
    keys=('checkpoint_kind','ratio','weight','learning_rate','task','metric','seed')
    indexed={}
    for row in rows:
        if row['phase']!='formal':continue
        key=(row['group'],)+tuple(row[k] for k in keys)
        if key in indexed:raise ValueError('Duplicate behavioral group/seed/cell')
        indexed[key]=row
    paired=[]
    for (group,*cell),left in indexed.items():
        for a,b in PAIRS:
            if group!=a:continue
            right=indexed.get((b,*cell))
            if right is None:continue
            paired.append({'left_group':a,'right_group':b,**dict(zip(keys,cell)),
                'left_run_id':left['run_id'],'right_run_id':right['run_id'],
                'left_target_prediction_tokens':left['target_prediction_tokens_at_checkpoint'],
                'right_target_prediction_tokens':right['target_prediction_tokens_at_checkpoint'],
                'accuracy_left_minus_right':left['accuracy']-right['accuracy'],
                'delta_accuracy_left_minus_right':left['accuracy_delta']-right['accuracy_delta']})
    cells=defaultdict(list)
    aggregate_keys=('left_group','right_group',*keys[:-1])
    for row in paired:cells[tuple(row[k] for k in aggregate_keys)].append(row)
    aggregate=[]
    for cell,items in sorted(cells.items()):
        aggregate.append({**dict(zip(aggregate_keys,cell)),**mean_sd(r['accuracy_left_minus_right'] for r in items),
                          'quantity':'accuracy_left_minus_right','training_seeds':','.join(str(r['seed']) for r in sorted(items,key=lambda r:r['seed'])),
                          'scope':'Match training seeds before computing differences and sample SD; no question pseudo-replicates.'})
    return paired,aggregate


def verify_prediction_rows(rows,prepared):
    _,rows=align(prepared,rows)
    for row in rows:
        scores=row['choice_logprobs'];lengths=row['choice_character_lengths']
        if (len(scores)!=len(lengths) or lengths!=[len(choice) for choice in row['choices']]
                or any(n<=0 for n in lengths) or not all(math.isfinite(x) for x in scores)):
            raise ValueError('Invalid per-choice score/character length')
        for metric,values in (('acc',scores),('acc_norm',[x/n for x,n in zip(scores,lengths)])):
            prediction=max(range(len(values)),key=lambda i:values[i])
            if row['prediction_'+metric]!=prediction or row['correct_'+metric]!=int(prediction==row['gold']):
                raise ValueError('Stored prediction/correctness differs from choice logprobs')
    return rows


def collect(manifest_path,n_bootstrap=5000,seed=20261106):
    from replay_study.execution import verify_matching_execution
    manifest=read(manifest_path);manifest_sha=sha(manifest_path)
    if (n_bootstrap!=manifest['analysis_policy']['bootstrap_replicates']
            or seed!=manifest['analysis_policy']['bootstrap_seed']):
        raise ValueError('Analysis bootstrap settings differ from frozen manifest')
    for path,expected in manifest['common_input_sha256'].items():
        if sha(path)!=expected:raise ValueError('Frozen task/scorer input changed: '+path)
    policy=manifest['inference_policy'];prepared={name:jsonl(row['artifact']) for name,row in manifest['tasks'].items()}
    initial_jobs=[j for j in manifest['jobs'] if j['kind']=='initial']
    if len(initial_jobs)!=1:raise ValueError('Exactly one initial model baseline required')
    initial_predictions={};initial_execution=None;rows=[];evidence={}
    ordered=initial_jobs+[j for j in manifest['jobs'] if j['kind']!='initial']
    for job in ordered:
        output=Path(job['output']);attest_path=output.with_name(output.stem+'_execution.json')
        attest=read(attest_path)
        expected_manifest_sha=job.get('origin_manifest_sha256',manifest_sha)
        if (attest['manifest_sha256']!=expected_manifest_sha or attest['job_id']!=job['id']
                or attest.get('verified_frozen_inputs') is not True
                or attest.get('checkpoint_sha256')!=job.get('checkpoint_sha256')):
            raise ValueError('Behavior execution attestation differs from frozen job')
        for path,expected in attest['output_sha256'].items():
            if sha(path)!=expected:raise ValueError('Behavior output SHA mismatch: '+path)
        if attest['output_sha256'].get(str(output))!=sha(output):raise ValueError('Summary is not bound by execution attestation')
        d=read(output)
        if (d.get('complete') is not True or d.get('evaluation_role')!='formal'
                or d['model']!=policy['model'] or d['revision']!=policy['revision']
                or d.get('checkpoint')!=job.get('checkpoint') or d['precision']!=policy['precision']
                or d['harness_revision']!=manifest['harness_revision']
                or d.get('project_primary_metric')!='acc_norm' or set(d['tasks'])!=set(manifest['tasks'])):
            raise ValueError('Incomplete, smoke, wrong model/checkpoint or task scorer result')
        actual=d['execution_provenance']
        for field in ('deterministic_algorithms','deterministic_warn_only','allow_tf32_matmul','allow_tf32_cudnn',
                      'cudnn_deterministic','cudnn_benchmark','attention_implementation','flash_sdp_enabled',
                      'mem_efficient_sdp_enabled','math_sdp_enabled','cudnn_sdp_enabled','device_type',
                      'model_weight_dtype','effective_precision','CUBLAS_WORKSPACE_CONFIG'):
            if actual[field]!=policy[field]:raise ValueError('Behavior execution differs from frozen '+field)
        if actual['attention_backend_requested']!=policy['attention_backend'] or actual['deterministic_warn_only']:
            raise ValueError('Behavior attention/determinism policy mismatch')
        if initial_execution is None:initial_execution=actual
        else:verify_matching_execution(initial_execution,actual)
        evidence[job['id']]={'output':str(output),'sha256':sha(output),'attestation':str(attest_path),'attestation_sha256':sha(attest_path)}
        for task,task_meta in manifest['tasks'].items():
            metrics=d['tasks'][task]
            if (not metrics.get('complete_split') or metrics['items']!=task_meta['items']
                    or metrics['data_sha256']!=task_meta['sha256'] or metrics['dataset_revision']!=task_meta['revision']
                    or metrics['batch_size']!=policy['batch_size'] or metrics['max_length']!=policy['max_length']):
                raise ValueError('Behavior split/data/batch/context differs from frozen manifest')
            predictions_path=metrics['predictions_path']
            if (sha(predictions_path)!=metrics['predictions_sha256']
                    or attest['output_sha256'].get(predictions_path)!=metrics['predictions_sha256']):
                raise ValueError('Predictions are not bound by summary and execution attestation')
            predictions=verify_prediction_rows(jsonl(predictions_path),prepared[task])
            if job['kind']=='initial':initial_predictions[task]=predictions
            for metric in ('acc_norm','acc'):
                stream_seed=(seed+int(hashlib.sha256(f"{job['id']}|{task}|{metric}".encode()).hexdigest()[:8],16))%(2**32)
                stats=paired_binary_summary(initial_predictions[task],predictions,metric,n_bootstrap,stream_seed)
                if not math.isclose(stats['accuracy']['estimate'],metrics[metric],rel_tol=0,abs_tol=1e-15):
                    raise ValueError('Task summary accuracy differs from per-item correctness')
                row={k:job.get(k) for k in ('run_id','phase','group','seed','ratio','weight','learning_rate',
                                           'step','target_prediction_tokens_at_checkpoint','replay_prediction_tokens_at_checkpoint')}
                row.update(model_tag=manifest['model_tag'],checkpoint_kind=job['kind'],task=task,metric=metric,
                           primary_metric=metric=='acc_norm',items=stats['items'],**stats['counts'],
                           initial_accuracy=stats['initial_accuracy'],accuracy=stats['accuracy']['estimate'],
                           accuracy_ci_low=stats['accuracy']['ci_low'],accuracy_ci_high=stats['accuracy']['ci_high'],
                           accuracy_delta=stats['accuracy_delta'],delta_ci_low=stats['delta_ci_low'],delta_ci_high=stats['delta_ci_high'],
                           conditional_forgetting_rate=stats['conditional_forgetting']['estimate'],
                           conditional_acquisition_rate=stats['conditional_acquisition']['estimate'],
                           bootstrap_seed=stream_seed,uncertainty_scope=CONDITIONAL_SCOPE)
                rows.append(row)
    paired,paired_aggregate=paired_seed_summary(rows)
    return {'endpoints':rows,'seed_aggregates':grouped_seed_summary(rows),'paired_seed_differences':paired,
            'paired_seed_aggregates':paired_aggregate,'metadata':{'complete':True,'manifest':str(manifest_path),
                'manifest_sha256':manifest_sha,'model_tag':manifest['model_tag'],'jobs':len(ordered),
                'primary_metric':'acc_norm','auxiliary_metric':'acc','n_bootstrap':n_bootstrap,'bootstrap_seed':seed,
                'conditional_uncertainty_scope':CONDITIONAL_SCOPE,
                'binomial_interval':'95% Wilson, conditional question-sampling description; task items may be correlated',
                'paired_bootstrap':'95% paired-example percentile, implemented by equivalent multinomial transition counts',
                'seed_variability':'Sample SD ddof=1 across trained seeds; n=1 is undefined; no pooled-question training replication',
                'evidence':evidence,'model_inference_performed_by_analysis':False}}


def write_bundle(bundle,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,rows in bundle.items():
        if name=='metadata':continue
        with (output/(name+'.csv')).open('w',newline='') as handle:
            if rows:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'analysis_manifest.json').write_text(json.dumps(bundle['metadata'],indent=2)+'\n')
