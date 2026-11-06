#!/usr/bin/env python3
"""Frozen C input-only bank geometry; reuse verified 410M primary diagnostics."""
from __future__ import annotations
import csv,datetime as dt,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from analyze_bank_geometry import cdf_distances
from replay_study.weight_archive import file_sha256
from replay_study.selection import chunk_hashes

GROUP_METHOD={'B4':'random','D':'dedup_random','M':'nll_stratified','C':'cluster'}
OUTPUT=ROOT/'results/extended_bank_diagnostics'


def read(path):return json.loads(Path(path).read_text())
def absolute(path):return Path(path) if Path(path).is_absolute() else ROOT/path

def quota_expectation(nll,sizes,allocation):
    """Exact fixed-pool expected mixture; no new draw and no performance claim."""
    nll=np.asarray(nll,dtype=np.float64);sizes=np.asarray(sizes,dtype=int);q=np.asarray(allocation,dtype=int)
    if (len(sizes)!=len(q) or sizes.sum()!=len(nll) or np.any(sizes<=0) or np.any(q<0) or np.any(q>sizes) or q.sum()<=0):
        raise ValueError('invalid finite-pool strata/quota')
    order=np.argsort(nll,kind='stable');strata=np.split(order,np.cumsum(sizes)[:-1]);grid=np.unique(nll)
    weights=q/q.sum();population=sizes/sizes.sum();expected=np.zeros(len(grid));expected_mean=0.
    for ids,w in zip(strata,weights):
        expected+=w*np.searchsorted(np.sort(nll[ids]),grid,side='right')/len(ids)
        expected_mean+=w*nll[ids].mean()
    pool_cdf=np.searchsorted(np.sort(nll),grid,side='right')/len(nll)
    return dict(quota_stratum_sizes=sizes.tolist(),quota_bank_sizes=q.tolist(),
        quota_weight_total_variation=float(np.abs(weights-population).sum()/2),
        quota_expected_cdf_max_distance=float(np.abs(expected-pool_cdf).max()),
        quota_expected_bank_nll_mean=float(expected_mean),quota_expected_bank_nll_bias=float(expected_mean-nll.mean()))


def geometry(nll,features,indices):
    """Same formula as analyze_bank_geometry.analyze; method subset only."""
    selected=nll[indices];norm2=(features**2).sum(axis=1)
    distance=np.maximum(0,norm2[:,None]+norm2[indices][None,:]-2*features@features[indices].T).min(axis=1)
    ks,wasserstein=cdf_distances(nll,selected)
    return dict(teacher_nll_pool_mean=float(nll.mean()),teacher_nll_bank_mean=float(selected.mean()),
        teacher_nll_bank_sd=float(selected.std(ddof=1)),empirical_cdf_max_distance=ks,nll_wasserstein1=wasserstein,
        mean_nearest_normalized_feature_squared_distance=float(distance.mean()),
        p95_nearest_normalized_feature_squared_distance=float(np.quantile(distance,.95)))


def validate_bank(pool,bank,manifest):
    indices=np.asarray(manifest['selected_indices'],dtype=np.int64);tokens=pool['input_ids']
    if indices.ndim!=1 or len(indices)!=manifest['bank_chunks'] or len(np.unique(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=len(tokens)):
        raise ValueError('selected indices do not identify a unique fixed-pool subset')
    if not torch.equal(bank['input_ids'],tokens[indices]) or not torch.equal(bank['selection_indices'],torch.from_numpy(indices)):
        raise ValueError('actual bank tokens/selection indices differ from frozen pool')
    if bank['doc_ids']!=[pool['doc_ids'][int(i)] for i in indices]:raise ValueError('bank doc IDs/order differ from selected pool rows')
    for key in ['attention_mask','loss_mask']:
        if key in pool or key in bank:
            if key not in pool or key not in bank or not torch.equal(bank[key],pool[key][indices]):raise ValueError('validity masks differ')
    return indices


def analyze():
    started,cpu=time.monotonic(),time.process_time();started_utc=dt.datetime.now(dt.timezone.utc).isoformat()
    torch.set_num_threads(2);sources={};cache={};rows=[];indices_payload={};relations=[]
    def bind(path,expected=None):
        path=absolute(path);key=str(path);value=sources.get(key)
        if value is None:value=file_sha256(path);sources[key]=value
        if expected is not None and value!=expected:raise ValueError('frozen input SHA differs: '+key)
        return value
    freeze_path=ROOT/'results/extended_formal_freeze.json';bind(freeze_path);freeze=read(freeze_path)
    if freeze.get('status')!='passed' or freeze.get('formal_runs')!=63:raise ValueError('complete frozen C registry authority required')
    old_path=ROOT/'results/bank_diagnostics/bank_geometry.json';bind(old_path);old=read(old_path)
    old_rows={(r['seed'],r['method']):r for r in old['records']}
    numerical_fields=list(geometry(np.array([0.,1.]),np.eye(2),np.array([0,1])))
    for phase in freeze['phases']:
        registry_path=absolute(phase['active_registry']);bind(registry_path,phase['active_registry_sha256']);registry=read(registry_path)
        snapshot_path=absolute(phase['snapshot']);bind(snapshot_path,phase['snapshot_sha256']);snapshot=read(snapshot_path)
        groups={'B4','M'} if phase['model']=='pythia-2.8b' else set(GROUP_METHOD)
        actual=[r for r in registry if r['group'] in groups]
        if len(actual)!=3*len(groups) or {(r['seed'],r['group']) for r in actual}!={(s,g) for s in [11,22,33] for g in groups}:
            raise ValueError('actual C bank cohort differs from frozen model-specific methods')
        per_model={}
        for row in sorted(actual,key=lambda r:(r['seed'],r['group']=='D',r['group'])):
            config_path=absolute(row['config_path']);relative=str(config_path.relative_to(ROOT));bind(config_path,snapshot['configs'][relative]['sha256']);config=read(config_path)
            bank_path=absolute(config['replay_path']);manifest_path=bank_path.with_suffix('.json');bind(manifest_path);manifest=read(manifest_path)
            relative_bank=str(bank_path.relative_to(ROOT));bank_sha=bind(bank_path,snapshot['data'][relative_bank]['sha256'])
            if bank_sha!=manifest['artifact_sha256'] or manifest['method']!=GROUP_METHOD[row['group']] or manifest['selection_seed']!=row['seed']:
                raise ValueError('actual bank artifact/method/selection seed differs')
            pool_path=absolute(manifest['source_pool']);stats_path=absolute(manifest['teacher_statistics_artifact'])
            bind(pool_path,manifest['source_pool_sha256']);stats_sha=bind(stats_path,manifest['teacher_statistics_sha256'])
            bind(manifest['source_manifest'],manifest['source_manifest_sha256'])
            key=(phase['model'],row['seed'])
            if key not in cache:
                pool=torch.load(pool_path,map_location='cpu',weights_only=False);stats=torch.load(stats_path,map_location='cpu',weights_only=False)
                if (stats['source_pool_sha256']!=manifest['source_pool_sha256'] or stats['revision']!=config['model_revision'] or
                    Path(stats['teacher']).resolve()!=Path(config['model_name_or_path']).resolve()):raise ValueError('teacher statistics model/pool identity differs')
                nll=stats['teacher_nll'].numpy().astype(np.float64);features=stats['features'].numpy().astype(np.float64)
                features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
                counts=stats['prediction_token_counts'].numpy()
                if (len(nll)!=len(pool['input_ids']) or features.shape[0]!=len(nll) or not np.isfinite(nll).all() or
                    not np.isfinite(features).all() or not np.all(counts==pool['input_ids'].shape[1]-1)):
                    raise ValueError('fixed full-chunk statistics/feature shapes invalid')
                cache[key]=(pool,nll,features,counts,stats_sha,chunk_hashes(pool))
            pool,nll,features,counts,cached_sha,pool_hashes=cache[key]
            if cached_sha!=stats_sha:raise ValueError('same model/seed maps to different statistics')
            bank=torch.load(bank_path,map_location='cpu',weights_only=False);indices=validate_bank(pool,bank,manifest)
            hashes=chunk_hashes(bank)
            unique_bank_tokens=int(sum(dict(zip(hashes,counts[indices].tolist())).values()))
            unique_pool_tokens=int(sum(dict(zip(pool_hashes,counts.tolist())).values()))
            if (hashes!=manifest['selected_chunk_hashes'] or hashes!=[pool_hashes[i] for i in indices] or
                len(set(hashes))!=manifest['bank_unique_chunks'] or len(set(pool_hashes))!=manifest['pool_unique_chunks'] or
                unique_bank_tokens!=manifest['unique_prediction_tokens'] or int(counts[indices].sum())!=manifest['prediction_tokens']):
                raise ValueError('canonical token/mask hashes or unique capacity differ')
            reused=phase['model']=='pythia-410m'
            if reused:
                reuse=manifest['extended_reuse_provenance'];original=absolute(reuse['source_artifact']);original_manifest=original.with_suffix('.json')
                bind(original,bank_sha);bind(original_manifest,reuse['source_manifest_sha256'])
                if not reuse['byte_identical_to_primary'] or reuse['additional_generation_or_selection_seconds']!=0:raise ValueError('410M is not proven primary reuse')
                for source,digest in [(stats_path,stats_sha),(original,bank_sha),(original_manifest,sources[str(original_manifest)])]:
                    if old['sources_sha256'].get(str(source.relative_to(ROOT)))!=digest:raise ValueError('old 410M geometry input SHA differs')
                if read(original_manifest)['selected_indices']!=indices.tolist():raise ValueError('410M reused selected indices differ')
                metrics={k:old_rows[row['seed'],manifest['method']][k] for k in numerical_fields}
                if not np.isclose(metrics['teacher_nll_bank_mean'],nll[indices].mean(),atol=1e-12,rtol=0):raise ValueError('reused NLL aggregate differs')
            elif row['group']=='D':
                earlier=per_model[row['seed'],'B4']
                if indices.tolist()==earlier['indices']:metrics=earlier['metrics']
                else:metrics=geometry(nll,features,indices)
            else:metrics=geometry(nll,features,indices)
            selected_id=row['id'];indices_payload[selected_id]=indices.tolist()
            item=dict(model=phase['model'],phase=phase['phase'],run_id=selected_id,seed=row['seed'],group=row['group'],method=manifest['method'],
                pool_chunks=len(nll),bank_chunks=len(indices),pool_prediction_tokens=int(counts.sum()),bank_prediction_tokens=int(counts[indices].sum()),
                pool_unique_chunks=len(set(pool_hashes)),bank_unique_chunks=len(set(hashes)),
                bank_unique_prediction_tokens=unique_bank_tokens,pool_unique_prediction_tokens=unique_pool_tokens,
                sequence_length=pool['input_ids'].shape[1],feature_dimensions=features.shape[1],teacher_nll_pool_sd=float(nll.std(ddof=1)),
                selected_indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),selection_manifest=str(manifest_path.relative_to(ROOT)),
                bank_tensor_sha256=bank_sha,statistics_sha256=stats_sha,pool_sha256=manifest['source_pool_sha256'],
                geometry_reused_from_primary=reused,additional_generation_or_selection=False,**metrics)
            if row['group']=='M':
                strata=np.split(np.argsort(nll,kind='stable'),np.cumsum(manifest['stratum_sizes'])[:-1])
                observed=[int(np.isin(indices,ids).sum()) for ids in strata]
                if observed!=manifest['stratum_bank_sizes']:raise ValueError('actual M indices violate fixed stratum quotas')
                item.update(quota_expectation(nll,manifest['stratum_sizes'],manifest['stratum_bank_sizes']))
                rank=np.empty(len(nll),dtype=np.int64);rank[np.argsort(nll,kind='stable')]=np.arange(len(nll))
                item['selected_stable_nll_rank_sha256']=hashlib.sha256(np.sort(rank[indices]).astype('<i8').tobytes()).hexdigest()
                item['pool_nll_tie_count']=len(nll)-len(np.unique(nll))
            if row['group']=='D':
                earlier=per_model[row['seed'],'B4'];same=indices.tolist()==earlier['indices'] and bank_sha==earlier['bank_sha']
                # Serialization provenance can differ while token tensors and order are identical.
                exact=torch.equal(bank['input_ids'],earlier['tokens']) and indices.tolist()==earlier['indices']
                relations.append(dict(model=phase['model'],seed=row['seed'],B4_run_id=earlier['run_id'],D_run_id=row['id'],
                    same_selection_indices=indices.tolist()==earlier['indices'],token_tensor_bitwise_equal=exact,artifact_file_sha_equal=same,
                    scope='input-bank equality only; no statement about final checkpoint equality'))
            per_model[row['seed'],row['group']]=dict(indices=indices.tolist(),metrics=metrics,bank_sha=bank_sha,tokens=bank['input_ids'],run_id=row['id'])
            rows.append(item)
    bind(__file__);bind(ROOT/'scripts/analyze_bank_geometry.py');bind(ROOT/'src/replay_study/selection.py')
    return dict(status='complete_input_only',records=rows,selected_indices=indices_payload,D_B4_input_relations=relations,sources_sha256=sources,
        started_at_utc=started_utc,completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
        timer_scope='input hashing/loading/validation and analytical geometry only; excludes process startup imports and output/document writing',
        cpu_threads=2,model_inference=False,new_selection=False,validation_test_behavior_scores_read=False,
        reuse='410M geometry reuses existing input-SHA-matched primary diagnostic; no new generation repetition',
        uncertainty='3 generation+selection seeds per model; not fixed-pool conditional bank-draw B60; geometry is model-specific')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError("set CUDA_VISIBLE_DEVICES='' explicitly")
    result=analyze();OUTPUT.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for model,group in sorted({(r['model'],r['group']) for r in result['records']}):
        current=[r for r in result['records'] if r['model']==model and r['group']==group]
        summary=dict(model=model,group=group,n_generation_selection_seeds=len(current),sd_ddof=1)
        for metric in ['teacher_nll_pool_mean','teacher_nll_bank_mean','teacher_nll_bank_sd','empirical_cdf_max_distance',
            'nll_wasserstein1','mean_nearest_normalized_feature_squared_distance','p95_nearest_normalized_feature_squared_distance']:
            values=np.array([r[metric] for r in current]);summary[metric+'_mean']=float(values.mean());summary[metric+'_sd']=float(values.std(ddof=1))
        summaries.append(summary)
    result['descriptive_model_method_summaries']=summaries
    keys=list(dict.fromkeys(k for row in result['records'] for k in row))
    with (OUTPUT/'bank_geometry.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(result['records'])
    with (OUTPUT/'bank_geometry_summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    (OUTPUT/'bank_geometry.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(OUTPUT),rows=len(result['records']),wall_seconds=result['wall_seconds'],cpu_seconds=result['process_cpu_seconds'])))
if __name__=='__main__':main()
