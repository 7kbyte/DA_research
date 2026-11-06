#!/usr/bin/env python3
"""CPU-only A selected-final export: 15 new weights + 3 durable references.

Separate A identity; no test-score inputs, no mutable master queue binding.
The tested codec and prior metadata/TAR helpers are imported unchanged.
"""
from __future__ import annotations
import argparse, datetime as dt, fcntl, json, os, resource, stat, sys, tarfile, time, zipfile
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from replay_study.analysis import _check_summary, training_validation_execution, verify_recorded_execution
from replay_study.weight_archive import encode_weights, decode_weights, verify_roundtrip, file_sha256
from export_remaining_primary_endpoints import checkpoint_metadata, same_metadata, build_verified_tar, signature, write

SELECTION = ROOT / 'results/supplement_A_validation/validation_lambda_selection.json'
SELECTION_SHA = '9548085fc18278f69fad584b67485bceb5193a18b10c78f516f6b09b4fa4e477'
REGISTRY = ROOT / 'configs/supplemental_v1_A_registry.json'
A_MANIFEST = ROOT / 'artifacts/evaluation_ready_v5/supplement_A_nll/manifest.json'
A_MANIFEST_SHA = '88dc927c9b4909cca66901608216290e1e435022e4ff925c00875f5099077a90'
PRIMARY_TAR = Path('/data/202610/7kb/research_delivery/primary_final_weights_lossless_v1.tar')
PRIMARY_SHA = '2bb090fbe7df172b31102068715143ab702087ff49f77ef7dfd5db3637c928b1'
PRIMARY_MANIFEST_SHA = '4120d177c22200b0bc5ee8390e0c2e2a1f5be1acbc58ef0f6d209bd47f96e8fb'
BASE = Path('/tmp/grad-research-20261106/models/pythia-410m-final/model.safetensors')
BASE_SHA = '3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b'
REVISION = '9879c9b5f8bea9051dcb0e68dff21493d67e9d4f'
KIND = 'validation_selected_lambda_final'
OUTPUT = Path('/tmp/grad-research-20261106/delivery/supplement_A_selected_final_v1')
LIMIT = 20 << 30


def read(path): return json.loads(Path(path).read_text())
def proof(path): return dict(path=str(Path(path).resolve()), sha256=file_sha256(path))


def operational_ledger_diff(old,current):
    allowed={'updated_at_utc','stage','test_scores_seen_scope','supplement_A_nll_analysis_complete',
             'supplement_A_validation_selection','first_supplement_A_nll_read'}
    changed={k for k in set(old)|set(current) if old.get(k)!=current.get(k)}
    if changed-allowed: raise ValueError('non-operational protocol ledger fields changed: '+str(changed-allowed))
    return sorted(changed)


def verify_primary_stat(receipt,path):
    current=path.lstat(); expected=receipt['destination_stat']
    fields=dict(device=current.st_dev,inode=current.st_ino,bytes=current.st_size,mtime_ns=current.st_mtime_ns,
                ctime_ns=current.st_ctime_ns,uid=current.st_uid,gid=current.st_gid,mode=oct(stat.S_IMODE(current.st_mode)))
    if not stat.S_ISREG(current.st_mode) or any(fields[k]!=expected[k] for k in fields):
        raise ValueError('durable primary TAR publication identity/stat changed')
    return fields


def selected_rows(selection, registry):
    if (selection.get('complete') is not True or selection.get('completed_conditions') != 126 or
        selection.get('expected_conditions') != 126 or selection.get('selection_uses_test') is not False or
        selection.get('test_scores_accessed') is not False or len(registry) != 126):
        raise ValueError('complete126 frozen validation-only A selection required')
    by_id = {r['id']: r for r in registry}
    if len(by_id) != 126 or len(selection['selections']) != 6:
        raise ValueError('unique126 registry and six methods required')
    rows = []
    if {c['group'] for c in selection['selections']} != {'B1','B2','B3','B4','M','C'}:
        raise ValueError('all six selected methods required')
    for chosen in selection['selections']:
        if (chosen.get('complete') is not True or chosen.get('selection_frozen') is not True or
            chosen.get('shared_across_seeds') is not True or chosen.get('n') != 3 or
            chosen.get('selected_by_best_checkpoint') is not False or chosen.get('test_scores_used') is not False):
            raise ValueError('final-only shared-lambda selection required')
        current = []
        for condition in chosen['condition_ids']:
            row = dict(by_id[condition]); actual = row['alias_of'] if row['execution_kind']=='alias' else row['id']
            if (row['phase'] != 'supplement_A_lambda_sweep' or row['group'] != chosen['group'] or
                row['weight'] != chosen['weight'] or row['ratio'] != .05):
                raise ValueError('selected condition identity/lambda mismatch')
            external = row['execution_kind']=='alias'
            if external != (row['group']=='B1' and row['weight']==.05):
                raise ValueError('only selected B1 main finals may reuse durable weights')
            current.append({**row, 'condition_id':condition, 'actual_run_id':actual, 'kind':KIND,
                'model_tag':'pythia-410m', 'data_family':'main_v3', 'external_weight_reference':external})
        if (len(current)!=3 or {r['seed'] for r in current}!={11,22,33} or
            set(chosen['actual_run_ids'])!={r['actual_run_id'] for r in current}):
            raise ValueError('three unique selected seeds/actual runs required')
        rows += current
    if len(rows)!=18 or sum(r['external_weight_reference'] for r in rows)!=3:
        raise ValueError('A must have18 logical/15 new/3 references')
    return sorted(rows,key=lambda r:(r['external_weight_reference'],r['group'],r['seed']))


def validate_source(row, summary, saved, baseline, protocol):
    _check_summary(summary,{**row,'id':row['actual_run_id']})
    if summary.get('status')!='completed' or saved['config']!=summary['config']:
        raise ValueError('completed source checkpoint/config required')
    if saved.get('provenance')!=summary.get('provenance'):
        raise ValueError('checkpoint provenance differs from frozen summary')
    config=summary['config']; point=summary['validation_trajectory'][-1]
    if (config['model_revision']!=REVISION or config['seed']!=row['seed'] or
        Path(config['target_path']).resolve()!=(ROOT/'artifacts/data/finemath_train.pt').resolve() or
        config['validation_paths']!=protocol['validation_paths']):
        raise ValueError('source model/data/seed identity mismatch')
    verify_recorded_execution(baseline['execution_provenance'],training_validation_execution(summary))
    for field, expected in [('target_prediction_tokens',10_000_000),('replay_prediction_tokens',500_000)]:
        if any(obj[field]!=expected for obj in [saved['counters'],summary['counters'],point]):
            raise ValueError('selected final must have actual10M/.5M exposure')
    if (saved['counters']['step']!=point['step'] or saved['counters']['step']!=summary['counters']['step'] or
        saved.get('validation_trajectory')!=summary['validation_trajectory'] or
        saved.get('best_target_validation')!=summary['best_target_validation']):
        raise ValueError('source final step/trajectory/selection differs')
    return dict(checkpoint_step=point['step'],target_prediction_tokens=10_000_000,replay_prediction_tokens=500_000,
        full_training_target_prediction_tokens=10_000_000,full_training_replay_prediction_tokens=500_000)


def preflight():
    if file_sha256(SELECTION)!=SELECTION_SHA: raise ValueError('frozen A selection SHA changed')
    selection,registry=read(SELECTION),read(REGISTRY); rows=selected_rows(selection,registry)
    if file_sha256(A_MANIFEST)!=A_MANIFEST_SHA: raise ValueError('frozen A checkpoint identity manifest changed')
    identity=read(A_MANIFEST); identity_by_id={r['run_id']:r for r in identity['records']}
    if len(identity_by_id)!=126: raise ValueError('complete126 source checkpoint identities required')
    sources=[proof(SELECTION),proof(REGISTRY),proof(A_MANIFEST)]
    for path,digest in selection['sources_sha256'].items():
        if Path(path).resolve()==(ROOT/'docs/protocol_status.json').resolve():
            # Root legitimately updates the global scheduling/status ledger.
            # Preserve its selection-time SHA as history; frozen A registry,
            # source configs/summaries and checkpoint manifest remain strict.
            old=ROOT/'artifacts/provenance/A_weight_export/selection_time_protocol_status.json'
            if file_sha256(old)!=digest: raise ValueError('selection-time status byte snapshot changed')
            changed=operational_ledger_diff(read(old),read(path))
            sources.append(dict(path=path,sha256=digest,historical_operational_ledger=True,
                original_byte_snapshot=proof(old),semantic_changed_fields=changed))
            sources.append(proof(old))
            continue
        if file_sha256(path)!=digest: raise ValueError('complete A source changed: '+path)
        sources.append(dict(path=path,sha256=digest))
    protocol=read(ROOT/'docs/formal_deterministic_protocol.json'); baseline=read(protocol['initial_validation_path'])
    for row in registry:
        summary=read(row['summary_path']); actual=row['alias_of'] if row['execution_kind']=='alias' else row['id']
        _check_summary(summary,{**row,'id':actual})
        if summary['status']!='completed' or summary['counters']['target_prediction_tokens']!=10_000_000:
            raise ValueError('all126 sources must be completed')
        verify_recorded_execution(baseline['execution_provenance'],training_validation_execution(summary))
    receipt=read(ROOT/'docs/primary_weights_durable_publication.json')
    if (receipt.get('status')!='passed' or receipt.get('destination_sha256')!=PRIMARY_SHA or
        receipt.get('bytes')!=24_949_391_360 or PRIMARY_TAR.stat().st_size!=24_949_391_360):
        raise ValueError('durable primary TAR identity differs')
    primary_stat=verify_primary_stat(receipt,PRIMARY_TAR)
    # Only its small manifest and ZIP metadata are read, never all TAR payloads.
    with tarfile.open(PRIMARY_TAR) as archive:
        raw=archive.extractfile('primary_final_weights_manifest.json').read(); primary=json.loads(raw)
        if __import__('hashlib').sha256(raw).hexdigest()!=PRIMARY_MANIFEST_SHA:
            raise ValueError('primary manifest differs from previously accepted member SHA')
        by_run={r['condition_run_id']:r for r in primary['logical_runs']}
        b1=by_run['formal_deterministic_B1_s11_b0p05_w0p05']
        with archive.extractfile(b1['archive_member']) as member,zipfile.ZipFile(member) as z:
            metadata=json.loads(z.read('metadata.json'))
    if verify_primary_stat(receipt,PRIMARY_TAR)!=primary_stat: raise ValueError('primary TAR changed while reading small members')
    if metadata['base']['sha256']!=BASE_SHA or file_sha256(BASE)!=BASE_SHA:
        raise ValueError('pinned base identity differs')
    dependency=dict(tar_filename=PRIMARY_TAR.name,tar_sha256=PRIMARY_SHA,tar_bytes=24_949_391_360,
        manifest_member='primary_final_weights_manifest.json',manifest_sha256=__import__('hashlib').sha256(raw).hexdigest(),
        validation='historical full-TAR SHA acceptance + current dev/inode/owner/size/mtime/ctime and pinned small manifest; no new full-TAR hash',
        current_publication_stat=primary_stat,
        base_revision=REVISION,base_files=primary['base_files'],
        base_tensor_hash_source_weight_member=b1['archive_member'],
        base_fp32_tensors=[{k:r[k] for k in ('name','shape','dtype','base_fp32_raw_sha256')} for r in metadata['tensors']])
    snapshot=read(ROOT/'artifacts/provenance/formal_deterministic_input_snapshot.json')['data']
    members={r['name']:r for r in primary['members']}
    for row in rows:
        summary=read(row['summary_path']); checkpoint=Path(summary['checkpoint_path']); before=signature(checkpoint)
        bound=identity_by_id[row['condition_id']]
        if (bound['actual_run_id']!=row['actual_run_id'] or bound['role']!='supplement_A' or bound['kind']!='final' or
            bound['model_tag']!='pythia-410m' or bound['data_family']!='main_v3' or
            Path(bound['checkpoint']).resolve()!=checkpoint.resolve() or bound['checkpoint_bytes']!=before['bytes'] or
            bound['checkpoint_mtime_ns']!=before['mtime_ns'] or bound['weight']!=row['weight'] or bound['training_seed']!=row['seed']):
            raise ValueError('frozen A checkpoint identity/stat mismatch')
        saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        row.update(checkpoint=str(checkpoint),source_stat=before,actual_exposure=validate_source(row,summary,saved,baseline,protocol),
            source_config_sha256=file_sha256(row['config_path']),source_summary_sha256=file_sha256(row['summary_path']),
            expected_checkpoint_sha256=bound['checkpoint_sha256'])
        config=summary['config']
        for path in [config['target_path'],config['replay_path'],*config['validation_paths'].values()]:
            relative=str(Path(path).resolve().relative_to(ROOT))
            if relative not in snapshot or file_sha256(path)!=snapshot[relative]['sha256']:
                raise ValueError('frozen train/replay/validation input changed: '+relative)
        del saved
        if signature(checkpoint)!=before: raise ValueError('checkpoint changed during metadata read')
        if row['external_weight_reference']:
            old=by_run[row['actual_run_id']]
            if old['source_checkpoint_sha256']!=row['expected_checkpoint_sha256']:
                raise ValueError('B1 durable and A frozen source SHA disagree')
            row['durable_weight_reference']=dict(tar_filename=PRIMARY_TAR.name,tar_sha256=PRIMARY_SHA,
                archive_member=old['archive_member'],archive_sha256=members[old['archive_member']]['sha256'],
                source_checkpoint_sha256=old['source_checkpoint_sha256'],model_fingerprint=old['model_fingerprint'])
    sources += [proof(ROOT/'docs/formal_deterministic_protocol.json'),proof(protocol['initial_validation_path']),
                proof(ROOT/'artifacts/provenance/formal_deterministic_input_snapshot.json')]
    return rows,sources,dependency,baseline,protocol


def logical_metadata(row,saved):
    keys=('condition_id','actual_run_id','kind','phase','model_tag','data_family','group','seed','ratio','weight')
    return dict(logical_identity={k:row[k] for k in keys},actual_exposure=row['actual_exposure'],
                original_checkpoint_metadata=checkpoint_metadata(saved))


def export_one(row,output,baseline,protocol):
    name=row['condition_id']+'__'+KIND; checkpoint=Path(row['checkpoint'])
    if signature(checkpoint)!=row['source_stat']: raise ValueError('source stat changed')
    record_path=output/'records'/(name+'.json')
    if record_path.exists():
        record=read(record_path)
        if (record['source_stat']!=row['source_stat'] or record['source_checkpoint_sha256']!=row['expected_checkpoint_sha256'] or
            file_sha256(record['metadata_path'])!=record['metadata_sha256'] or
            file_sha256(checkpoint)!=row['expected_checkpoint_sha256']):
            raise ValueError('verified resume source/metadata changed')
        if not row['external_weight_reference'] and file_sha256(record['archive_path'])!=record['archive_sha256']:
            raise ValueError('verified resume archive changed')
        return {**record,'record_reused_from_prior_verified_invocation':True}
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    validate_source(row,read(row['summary_path']),saved,baseline,protocol)
    sidecar=output/'metadata'/(name+'.source_metadata.pt'); sidecar.parent.mkdir(parents=True,exist_ok=True)
    payload=logical_metadata(row,saved)
    if sidecar.exists(): raise FileExistsError('preserve prior sidecar; use reviewed fresh namespace')
    torch.save(payload,sidecar)
    if not same_metadata(payload,torch.load(sidecar,map_location='cpu',weights_only=False)):
        raise ValueError('full original/logical metadata failed roundtrip')
    del saved,payload
    record=dict(condition_id=row['condition_id'],actual_run_id=row['actual_run_id'],kind=KIND,
        phase=row['phase'],model_tag=row['model_tag'],data_family=row['data_family'],group=row['group'],
        seed=row['seed'],ratio=row['ratio'],weight=row['weight'],
        metadata_path=str(sidecar),metadata_member='metadata/'+sidecar.name,metadata_sha256=file_sha256(sidecar),
        source_stat=row['source_stat'],actual_exposure=row['actual_exposure'])
    started,cpu=time.monotonic(),time.process_time()
    if row['external_weight_reference']:
        reference=row['durable_weight_reference']
        if file_sha256(checkpoint)!=reference['source_checkpoint_sha256']:
            raise ValueError('B1 original source differs from durable source')
        record.update(durable_weight_reference=reference,model_fingerprint=reference['model_fingerprint'],
            source_checkpoint_sha256=reference['source_checkpoint_sha256'],new_weight_forward_or_encoding=False)
    else:
        archive=output/'weights'/(name+'.xor.zip'); restored=output/'verification'/(name+'.restored.pt')
        enc=encode_weights(checkpoint,BASE,archive,threads=2); dec=decode_weights(archive,BASE,restored,threads=2)
        verified=verify_roundtrip(checkpoint,restored,archive,threads=2)
        if not verified['all_tensor_bytes_equal'] or enc['base_sha256']!=BASE_SHA:
            raise ValueError('full original uint8 comparison/base failed')
        if enc['source_checkpoint_sha256']!=row['expected_checkpoint_sha256']:
            raise ValueError('encoded source differs from frozen A checkpoint SHA')
        record.update(archive_path=str(archive),archive_member='weights/'+archive.name,archive_sha256=enc['archive_sha256'],
            model_fingerprint=enc['model_fingerprint'],source_checkpoint_sha256=enc['source_checkpoint_sha256'],
            encode=enc,decode=dec,bytewise_verification=verified,original_and_zip_and_restored_retained=True)
    if signature(checkpoint)!=row['source_stat']: raise ValueError('source changed during export')
    record.update(wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes_process_cumulative=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        all_nonweight_metadata_roundtrip_verified=True,gpu_used=False,shared_storage_written=False)
    write(record_path,record)
    return record


def run(output=OUTPUT,prepare_only=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError("set CUDA_VISIBLE_DEVICES='' explicitly")
    torch.set_num_threads(2); output=Path(output).resolve()
    if not output.is_relative_to(Path('/tmp/grad-research-20261106/delivery')):
        raise ValueError('outputs must remain in study scratch/delivery')
    started,cpu=time.monotonic(),time.process_time()
    rows,sources,dependency,baseline,protocol=preflight(); output.mkdir(parents=True,exist_ok=True)
    code=[proof(Path(__file__)),proof(ROOT/'src/replay_study/weight_archive.py'),proof(ROOT/'scripts/export_remaining_primary_endpoints.py')]
    plan=dict(schema_version=1,kind=KIND,phase='supplement_A_lambda_sweep',logical_endpoints=18,new_weights=15,
        durable_references=3,selection_sha256=SELECTION_SHA,checkpoint_identity_manifest_sha256=A_MANIFEST_SHA,
        rows=rows,sources=sources,code=code,base_dependency=dependency,
        coverage='selected18 A finals; union with main lambda.05 is33/126; excludes other93/A best/B60/C best',
        size_ceiling_bytes=LIMIT,reserved_path='/data/202610/7kb/research_delivery/.supplement_A_selected_final_weights_reserve_20261106',
        gpu_used=False,shared_storage_written=False,cpu_threads=2,workers=1)
    if prepare_only:
        write(output/'preparation_review.json',{**plan,'status':'prepared_only',
            'wall_seconds':time.monotonic()-started,'process_cpu_seconds':time.process_time()-cpu})
        print(json.dumps(dict(status='prepared_only',logical=18,new_weights=15,durable_references=3,output=str(output))))
        return
    plan_path=output/'export_plan.json'
    if plan_path.exists() and read(plan_path)!=plan: raise ValueError('existing source plan changed')
    if not plan_path.exists(): write(plan_path,plan)
    records=[];members=[]
    for row in rows:
        if any(file_sha256(p['path'])!=p['sha256'] for p in code): raise ValueError('adapter/helper/codec changed')
        record=export_one(row,output,baseline,protocol); records.append(record)
        members.append((Path(record['metadata_path']),record['metadata_member'],record['metadata_sha256']))
        if not row['external_weight_reference']:
            members.append((Path(record['archive_path']),record['archive_member'],record['archive_sha256']))
        write(output/'progress.json',dict(status='running',completed_logical=len(records),
            completed_new_weights=sum('encode' in r for r in records),elapsed_wall_seconds=time.monotonic()-started,
            process_cpu_seconds=time.process_time()-cpu,updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    if len(records)!=18 or sum('encode' in r for r in records)!=15: raise ValueError('incomplete A delivery refused')
    for source in sources:
        if not source.get('historical_operational_ledger') and file_sha256(source['path'])!=source['sha256']:
            raise ValueError('frozen complete A source changed')
    manifest={**plan,'status':'all18_logical_verified_15_new_full_roundtrips_passed','logical_runs':records,
        'all_original_checkpoints_and_zips_retained':True,'optimizer_included':False}
    manifest_path=output/'supplement_A_selected_final_manifest.json';write(manifest_path,manifest)
    readme=output/'RESTORE_A_SELECTED.md'
    readme.write_text('# A validation-selected-lambda final weights\n\n18 A logical finals;15 new weights;3 B1 payloads reference the primary TAR. This is selected18, not126.\n\nVerify primary_final_weights_lossless_v1.tar SHA256 '+PRIMARY_SHA+'. Extract base/pythia-410m-final/ and exact B1 weight members from that TAR; extract this increment into the same directory. Base file SHA, FP32 tensor SHA, source TAR/member and B1 fingerprints are in base_dependency and logical_runs. No download required.\n\n```bash\npython tools/scripts/archive_model_weights.py --threads 2 decode --archive weights/SELECTED.xor.zip --base base/pythia-410m-final/model.safetensors --output restored.pt\n```\n\nEach metadata sidecar has its own A logical_identity plus every original nonweight/nonoptimizer checkpoint field. Main B1 decoded source metadata remains main, with A identity recorded separately. No optimizer/full Adam-resume guarantee. Source checkpoint, ZIP and derived verification files remain scratch. Durable status requires separate publisher receipt.\n')
    for path,name in [(manifest_path,manifest_path.name),(readme,readme.name),(plan_path,'provenance/export_plan.json'),
        (Path(__file__),'tools/provenance/export_A_selected_weights.py'),
        (ROOT/'src/replay_study/weight_archive.py','tools/src/replay_study/weight_archive.py'),
        (ROOT/'src/replay_study/__init__.py','tools/src/replay_study/__init__.py'),
        (ROOT/'scripts/archive_model_weights.py','tools/scripts/archive_model_weights.py'),
        (SELECTION,'provenance/validation_lambda_selection.json')]:
        members.append((path,name,file_sha256(path)))
    receipt=build_verified_tar(output/'supplement_A_selected_final_weights_lossless_v1.tar',members,LIMIT)
    receipt.update(status='passed',kind=KIND,logical_endpoints=18,new_unique_weights=15,durable_references=3,
        tar_path=str(output/'supplement_A_selected_final_weights_lossless_v1.tar'),
        wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
        child_record_timers_nested_in_current_total=not any(r.get('record_reused_from_prior_verified_invocation') for r in records),
        record_timer_policy='reused record timers are historical; no new encoding cost; total is this invocation only',
        peak_rss_bytes_process_cumulative=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        original_checkpoints_retained=True,gpu_used=False,shared_storage_written=False,publisher_required=True)
    write(output/'completion.json',receipt);write(output/'progress.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:
        run(args.output_dir,True)
    else:
        # Shared by the later C adapter: one CPU export worker across phases.
        with Path('/tmp/grad-research-20261106/delivery/.supplemental_weight_export.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(args.output_dir,False)
