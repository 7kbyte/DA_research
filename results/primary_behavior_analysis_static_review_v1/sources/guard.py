#!/usr/bin/env python3
"""Gate v2 behavioral analysis on full attestation closure and explicit authorization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from replay_study.evaluation_cost_inventory import inventory_evaluations
from run_frozen_evaluation import verify_outputs


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


class Reader:
    def __init__(self):self.sources={};self.cache={}
    def read(self,path):
        p=str(Path(path).resolve())
        if p in self.cache:return self.cache[p]
        if not Path(p).exists():self.sources[p]={'exists':False};self.cache[p]=None;return None
        raw=Path(p).read_bytes();self.sources[p]={'exists':True,'sha256':hashlib.sha256(raw).hexdigest()};self.cache[p]=json.loads(raw)
        return self.cache[p]


def check(path,digest,expected_jobs=None):
    reader=Reader();manifest=reader.read(path)
    if reader.sources[str(Path(path).resolve())]['sha256']!=digest or manifest.get('schema_version')!=2 or manifest.get('mode')!='behavior':raise ValueError('Explicit frozen v2 behavioral manifest required')
    if expected_jobs is not None and len(manifest['jobs'])!=expected_jobs:raise ValueError('Expected complete inference count differs from frozen plan')
    for source,expected in manifest['common_input_sha256'].items():
        if sha(source)!=expected:raise ValueError('Frozen analysis source changed')
    report=inventory_evaluations({'evaluation_manifests':[{'path':str(path),'sha256':digest,'category':'primary','verify_output_hashes':True}]},reader)
    pending=[r['inference_job_id'] for r in report['evaluation_executions'] if not r['attestation_present']]
    pending += [r['logical_record_id'] for r in report['evaluation_references'] if r.get('alias_status')=='missing_alias_attestation']
    for key,group in manifest['baseline_groups'].items():
        origin=group.get('initial_origin');source_digest=origin['manifest_sha256'] if origin else digest
        ap=Path(group['initial_output']).with_name(Path(group['initial_output']).stem+'_execution.json')
        if not ap.exists():pending.append('baseline:'+key);continue
        a=verify_outputs(group['initial_output'],source_digest,group['initial_job_id'])
        if a.get('checkpoint_sha256') is not None:raise ValueError('Unadapted baseline proof required')
        expected={group['initial_output'],*[str(Path(group['initial_output']).with_name(Path(group['initial_output']).stem+'_'+t+'.jsonl')) for t in group['tasks']]}
        if set(a['output_sha256'])!=expected:raise ValueError('Complete baseline task output set required')
    return {'ready':not pending,'manifest':str(path),'manifest_sha256':digest,'expected_inference_jobs':len(manifest['jobs']),
            'logical_endpoints':len(manifest['records']),'pending':pending,'all_output_integrity_verified':not pending,
            'scores_or_predictions_parsed':False,'model_inference_started':False}


def analyze(path,digest,authorization_path,output):
    authorization=json.loads(Path(authorization_path).read_text())
    if authorization.get('allow_score_analysis') is not True or authorization.get('manifest_sha256')!=digest:
        raise ValueError('Explicit matching root authorization required before score parsing')
    state=check(path,digest,authorization['expected_inference_jobs'])
    if not state['ready']:raise ValueError('All original inferences, logical aliases and baselines must complete before unsealing')
    if state['logical_endpoints']!=authorization['expected_logical_endpoints']:raise ValueError('Authorized logical endpoint count differs')
    if Path(output).exists():raise FileExistsError('Use a new analysis output directory')
    # No scoring output is parsed until every guard above succeeds.
    from analyze_behavioral_plan import collect
    from replay_study.behavioral_analysis import write_bundle
    bundle=collect(path);bundle['metadata']['analysis_authorization']={'path':str(authorization_path),'sha256':sha(authorization_path)}
    write_bundle(bundle,output)
    return {'complete':True,'analysis_output':str(output),'model_inference_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--expected-inference-jobs',type=int);p.add_argument('--analyze',action='store_true');p.add_argument('--authorization',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.analyze:
        if not a.authorization or not a.output:p.error('--authorization and --output required to analyze')
        result=analyze(a.manifest,a.manifest_sha256,a.authorization,a.output)
    else:result=check(a.manifest,a.manifest_sha256,a.expected_inference_jobs)
    visible={k:v for k,v in result.items() if k!='pending'}
    if 'pending' in result:visible.update(pending_count=len(result['pending']),pending_examples=result['pending'][:5])
    print(json.dumps(visible))


if __name__=='__main__':main()
