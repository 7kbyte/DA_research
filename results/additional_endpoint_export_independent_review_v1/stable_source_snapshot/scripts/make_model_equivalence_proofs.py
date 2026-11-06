#!/usr/bin/env python3
"""Bind audited full-model equality to immutable checkpoint file identities."""
import argparse,datetime as dt,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_formal_runs import model_fingerprint,sha256
from make_experiments import atomic_json

ALGORITHM='sha256_sorted_key_dtype_shape_contiguous_uint8_v1'

def checkpoint_identity(path,expected=None):
    path=Path(path);before=path.stat();fingerprint=model_fingerprint(path);file_sha=sha256(path);after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise ValueError('Checkpoint changed during CPU proof computation')
    if expected is not None and fingerprint!=expected:raise ValueError('Current full model state differs from completed audit')
    return {'checkpoint':str(path),'checkpoint_sha256':file_sha,'checkpoint_bytes':after.st_size,
            'model_state_fingerprint':fingerprint}

def make(args):
    audit=json.loads(args.audit.read_text());registry=json.loads(args.registry.read_text())
    if audit['status']!='passed' or audit['completed_runs']!=36 or not audit['model_weights_compared']:
        raise ValueError('A complete passed full-weight audit is required')
    lookup={r['id']:r for r in registry};pairs=[];cache={}
    def side(run_id,expected):
        row=lookup[run_id];summary=json.loads(Path(row['summary_path']).read_text())
        path=summary['checkpoint_path'] if args.kind=='final' else summary['best_target_validation']['checkpoint_path']
        if args.kind=='best':expected=None
        if path not in cache:cache[path]=checkpoint_identity(path,expected)
        return {**cache[path],'run_id':run_id,'kind':args.kind,'model_revision':summary['config']['model_revision'],
                'model_name_or_path':summary['config']['model_name_or_path'],
                'config_path':row['config_path'],'config_sha256':sha256(row['config_path'])}
    for control in audit['D_B4_null_controls']:
        left=side(control['B4'],control['B4_model_fingerprint']);right=side(control['D'],control['D_model_fingerprint'])
        equal=left['model_state_fingerprint']==right['model_state_fingerprint']
        if not equal or not control['bitwise_model_state_equal']:raise ValueError('Audited model pair no longer equal')
        pairs.append({'id':f'formal_D_B4_seed{control["seed"]}_{args.kind}','kind':'identical_bank_null_control',
                      'seed':control['seed'],'left':left,'right':right,'bitwise_model_state_equal':True})
    best_final_candidates=[]
    for row in registry:
        summary=json.loads(Path(row['summary_path']).read_text());best=summary.get('best_target_validation')
        if args.kind!='final' or not best or best['step']!=summary['counters']['step']:continue
        best_final_candidates.append(row['id']);final=side(row['id'],None)
        best_side=checkpoint_identity(best['checkpoint_path']);best_side.update(run_id=row['id'],kind='best',
            model_revision=final['model_revision'],model_name_or_path=final['model_name_or_path'],
            config_path=final['config_path'],config_sha256=final['config_sha256'])
        if best_side['model_state_fingerprint']==final['model_state_fingerprint']:
            pairs.append({'id':row['id']+'_best_final','kind':'best_final_full_state',
                          'left':best_side,'right':final,'bitwise_model_state_equal':True})
    result={'schema_version':1,'status':'passed','computed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'computation_device':'CPU only','algorithm':ALGORITHM,
        'algorithm_source':{'path':'scripts/audit_formal_runs.py','sha256':sha256(ROOT/'scripts/audit_formal_runs.py'),
                            'function':'model_fingerprint'},
        'algorithm_detail':'For every sorted state key, SHA256 updates JSON [name,str(dtype),list(shape)] followed by all contiguous uint8 bytes; includes every tensor element and excludes config/optimizer/serialization metadata.',
        'source_audit':{'path':str(args.audit),'sha256':sha256(args.audit)},
        'source_registry':{'path':str(args.registry),'sha256':sha256(args.registry)},
        'producer_source':{'path':str(Path(__file__).resolve()),'sha256':sha256(__file__)},
        'comparisons':pairs,'best_final_candidates':best_final_candidates,
        'other_best_final_pairs':'No equivalence authorized without another full-state CPU proof.',
        'test_scores_accessed':False,'checkpoint_files_modified':False}
    atomic_json(args.output,result);print(json.dumps({'status':'passed','pairs':len(pairs),'output':str(args.output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,default=ROOT/'results/formal_run_audit.json')
    p.add_argument('--registry',type=Path,default=ROOT/'configs/formal_deterministic_registry.json')
    p.add_argument('--kind',choices=['final','best'],default='final');p.add_argument('--output',type=Path)
    args=p.parse_args()
    args.output=args.output or ROOT/('results/formal_model_equivalence_proofs.json' if args.kind=='final' else 'results/formal_best_model_equivalence_proofs.json')
    make(args)
