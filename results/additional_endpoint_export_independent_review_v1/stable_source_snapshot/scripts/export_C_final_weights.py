#!/usr/bin/env python3
"""CPU C-final draft export per rc0 run; deliver only after complete phase gates.

No scoring/inference, no shared writes. Master append/reorder is permitted;
each training definition remains frozen. D waits for explicit final proof.
"""
from __future__ import annotations
import argparse, datetime as dt, fcntl, hashlib, json, math, os, resource, signal, sys, tarfile, threading, time, zipfile
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from replay_study.analysis import _check_summary,training_validation_execution,verify_recorded_execution
from replay_study.weight_archive import encode_weights,decode_weights,verify_roundtrip,file_sha256
from export_remaining_primary_endpoints import checkpoint_metadata,same_metadata,build_verified_tar,signature,write
from export_A_selected_weights import verify_primary_stat,PRIMARY_TAR,PRIMARY_SHA,PRIMARY_MANIFEST_SHA
from audit_extended_runs import audit_steps,audit_validation,audit_phase
from audit_formal_runs import events_at,expected_replay
from schedule_jobs import digest_job

FREEZE=ROOT/'results/extended_formal_freeze.json'
OUTPUT=Path('/tmp/grad-research-20261106/delivery/extended_C_final_v1')
MASTER=ROOT/'configs/supplemental_v1_queue.json'
STATE=ROOT/'logs/supplemental_job_state.json'
LIMITS={'pythia-410m':27<<30,'pythia-1b':70<<30,'pythia-2.8b':140<<30}
READY_NAMES={'pythia-410m':'extended_pythia_410m_100m_nll','pythia-1b':'extended_pythia_1b_100m_nll',
             'pythia-2.8b':'extended_pythia_2p8b_25m_nll'}
RESERVES={'pythia-410m':'.extended_C_410m_final_weights_reserve_20261106',
          'pythia-1b':'.extended_C_1b_final_weights_reserve_20261106',
          'pythia-2.8b':'.extended_C_2p8b_final_weights_reserve_20261106'}

def read(path):return json.loads(Path(path).read_text())
def absolute(path):return Path(path) if Path(path).is_absolute() else ROOT/path
def proof(path):return dict(path=str(absolute(path)),sha256=file_sha256(absolute(path)))


class FrozenFiles:
    """Hash once against authority, then reject any signature change."""
    def __init__(self):self.files={}
    def bind(self,path,digest):
        path=absolute(path);key=str(path);observed=signature(path)
        if key in self.files:
            if self.files[key]!=dict(sha256=digest,stat=observed):raise ValueError('frozen file changed: '+key)
        else:
            if file_sha256(path)!=digest:raise ValueError('source SHA differs: '+key)
            self.files[key]=dict(sha256=digest,stat=observed)
    def stable(self):
        for path,record in self.files.items():
            if signature(path)!=record['stat']:raise ValueError('frozen source changed: '+path)


def load_contexts():
    freeze=read(FREEZE);cache=FrozenFiles();cache.bind(FREEZE,file_sha256(FREEZE));contexts=[]
    if freeze.get('status')!='passed' or freeze.get('formal_runs')!=63:raise ValueError('frozen complete63 C authority required')
    for phase in freeze['phases']:
        snapshot_path=absolute(phase['snapshot']);cache.bind(snapshot_path,phase['snapshot_sha256']);snap=read(snapshot_path)
        cache.bind(phase['active_registry'],phase['active_registry_sha256']);rows=read(phase['active_registry'])
        cache.bind(phase['protocol'],phase['protocol_sha256']);protocol=read(phase['protocol'])
        cache.bind(snap['queue_path'],snap['queue_sha256']);definitions=read(snap['queue_path'])
        if (len(rows)!=phase['runs'] or len({r['id'] for r in rows})!=len(rows) or
            any(r.get('phase')!='formal' or r.get('active') is not True or r.get('model')!=phase['model'] for r in rows) or
            {r['id'] for r in rows}!={j['id'] for j in definitions} or
            snap['model']!=phase['model'] or snap['configuration_prefix']!=phase['phase']):
            raise ValueError('C phase/model/registry/queue mismatch')
        for section in ['code','data','configs','model_files']:
            for path,record in snap[section].items():cache.bind(path,record['sha256'])
        for authority in snap['calibration_audit'].values():
            for name in ['selection','protocol','registry']:cache.bind(authority[name+'_path'],authority[name+'_sha256'])
            calibration={r['id']:r for r in read(authority['registry_path'])}
            for run_id,digest in authority['summaries_sha256'].items():cache.bind(calibration[run_id]['summary_path'],digest)
        lr=snap['calibration_audit']['lr'];cache.bind(lr['selection_path'],lr['selection_sha256'])
        choice=read(lr['selection_path']);cache.bind(choice['initial_validation_path'],choice['initial_validation_sha256'])
        baseline=read(choice['initial_validation_path']);config=read(rows[0]['config_path'])
        if protocol['model_revision']!=config['model_revision'] or protocol['data_version']!='extended_v1':raise ValueError('pinned model/data revision differs')
        model=phase['model'];expected_groups={'B0','B1','B2','B4','M'} if model=='pythia-2.8b' else {'B0','B1','B2','B3','B4','D','M','C'}
        if {(r['seed'],r['group']) for r in rows}!={(s,g) for s in [11,22,33] for g in expected_groups}:
            raise ValueError('complete frozen model-specific factorial required')
        contexts.append(dict(phase=phase,snapshot=snap,snapshot_path=str(snapshot_path),rows=rows,
            definitions={j['id']:j for j in definitions},protocol=protocol,baseline=baseline,
            base=str(Path(config['model_name_or_path'])/'model.safetensors'),base_sha=snap['model_files'][str(Path(config['model_name_or_path'])/'model.safetensors')]['sha256']))
    return contexts,cache


def run_ready(row,definition,state,master):
    current={j['id']:j for j in master};job=state.get('jobs',{}).get(row['id'],{})
    if len(current)!=len(master):raise ValueError('duplicate master job identities')
    if current.get(row['id'])!=definition:raise ValueError('training job definition changed despite permitted master append/reorder')
    if job.get('status')!='completed':return False
    if (job.get('returncode')!=0 or job.get('definition_sha256')!=digest_job(definition) or job.get('missing_outputs') or
        'stop_requested_at' in job or job.get('argv')!=definition['argv']):
        raise ValueError('completed run lacks rc0/frozen definition/actual argv/unstopped outputs')
    if not all(absolute(p).is_file() for p in [row['summary_path'],*definition.get('outputs',[])]):
        raise ValueError('completed run output missing')
    return True


def validate_source(context,row,summary,saved):
    _check_summary(summary,row);config=read(row['config_path']);target=context['phase']['target_prediction_tokens']
    replay=expected_replay(target,row['ratio']);point=summary['validation_trajectory'][-1]
    if (summary.get('status')!='completed' or summary['config']!=config or saved['config']!=config or
            saved.get('provenance')!=summary.get('provenance') or summary.get('provenance',{}).get('teacher_weight_dtype')!='torch.float32' or
            config['model_revision']!=context['protocol']['model_revision'] or
        row['model']!=context['phase']['model'] or config['target_prediction_tokens']!=target):
        raise ValueError('completed C model/config/provenance identity mismatch')
    verify_recorded_execution(context['baseline']['execution_provenance'],training_validation_execution(summary))
    for field,value in [('target_prediction_tokens',target),('replay_prediction_tokens',replay)]:
        if any(x[field]!=value for x in [summary['counters'],saved['counters'],point]):raise ValueError('C final actual exposure differs')
    if (saved['counters']['step']!=summary['counters']['step'] or point['step']!=saved['counters']['step'] or
        saved.get('validation_trajectory')!=summary['validation_trajectory'] or saved.get('best_target_validation')!=summary['best_target_validation']):
        raise ValueError('C final step/selection/trajectory differs')
    problems=audit_validation(summary,config)
    events=events_at(Path(row['output_dir'])/'events.jsonl',False);steps=[e for e in events if e['event']=='train_step']
    accounting=audit_steps(steps,config)
    if (problems or accounting['violations'] or accounting['target_prediction_tokens']!=target or
        accounting['replay_prediction_tokens']!=replay or accounting['steps']!=saved['counters']['step'] or
        any(e['event']=='error' for e in events)):
        raise ValueError('C per-run events/validation audit failed')
    return dict(checkpoint_step=point['step'],target_prediction_tokens=target,replay_prediction_tokens=replay,
        full_training_target_prediction_tokens=target,full_training_replay_prediction_tokens=replay)


def draft_one(context,row,folder,definition,expected_sha=None,expected_fingerprint=None,alias=None,completed_record=None):
    """Preserve one completed source; no complete-phase claim is made here."""
    record_path=folder/'records'/(row['id']+'.json');checkpoint=Path(row['checkpoint_path'])
    if record_path.exists():
        old=read(record_path)
        if (old['source_stat']!=signature(checkpoint) or old['config_sha256']!=file_sha256(row['config_path']) or
            old['summary_sha256']!=file_sha256(row['summary_path']) or old['definition_sha256']!=digest_job(definition) or
            old['events_sha256']!=file_sha256(Path(row['output_dir'])/'events.jsonl') or
            old['input_snapshot_sha256']!=context['phase']['snapshot_sha256']):
            raise ValueError('previous C draft source changed')
        if expected_sha is not None and old['source_checkpoint_sha256']!=expected_sha:raise ValueError('draft differs from final frozen CP SHA')
        for pathkey,statkey in [('metadata_path','metadata_stat'),('archive_path','archive_stat')]:
            if pathkey in old and signature(old[pathkey])!=old[statkey]:raise ValueError('verified C draft payload stat changed')
        return old
    before=signature(checkpoint);started,cpu=time.monotonic(),time.process_time()
    summary=read(row['summary_path']);saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    exposure=validate_source(context,row,summary,saved)
    identity=dict(run_id=row['id'],phase_name=context['phase']['phase'],role='extended',kind='fixed_exposure_final',
        model_tag=context['phase']['model'],data_family='extended_v1',group=row['group'],seed=row['seed'],ratio=row['ratio'],weight=row['weight'])
    payload=dict(logical_identity=identity,actual_exposure=exposure,original_checkpoint_metadata=checkpoint_metadata(saved))
    sidecar=folder/'metadata'/(row['id']+'.source_metadata.pt');sidecar.parent.mkdir(parents=True,exist_ok=True)
    if sidecar.exists():raise FileExistsError('unverified draft sidecar retained; root review needed')
    torch.save(payload,sidecar)
    if not same_metadata(payload,torch.load(sidecar,map_location='cpu',weights_only=False)):raise ValueError('C full metadata roundtrip failed')
    del saved,payload
    record=dict(status='verified_draft_not_deliverable',**identity,source_checkpoint=str(checkpoint),source_stat=before,
        config_sha256=file_sha256(row['config_path']),summary_sha256=file_sha256(row['summary_path']),
        events_sha256=file_sha256(Path(row['output_dir'])/'events.jsonl'),definition=definition,definition_sha256=digest_job(definition),
        completed_scheduler_record=completed_record,
        input_snapshot_sha256=context['phase']['snapshot_sha256'],model_revision=context['protocol']['model_revision'],
        actual_exposure=exposure,metadata_path=str(sidecar),metadata_member='metadata/'+sidecar.name,
        metadata_sha256=file_sha256(sidecar),metadata_stat=signature(sidecar),all_nonweight_metadata_roundtrip_verified=True)
    if alias is not None:
        source_sha=file_sha256(checkpoint)
        if expected_sha is None or source_sha!=expected_sha:raise ValueError('D alias source SHA does not match final proof/manifest')
        record.update(source_checkpoint_sha256=source_sha,alias_of=alias['run_id'],alias_evidence=alias,
            model_fingerprint=alias['model_fingerprint'],additional_encoding_calls=0)
    else:
        archive=folder/'weights'/(row['id']+'.xor.zip');restored=folder/'verification'/(row['id']+'.restored.pt')
        encode_cpu=time.process_time();encoded=encode_weights(checkpoint,context['base'],archive,threads=2,
            expected_model_sha256=None if expected_fingerprint is None else expected_fingerprint['sha256_model_state'])
        encode_cpu=time.process_time()-encode_cpu;decode_cpu=time.process_time();decoded=decode_weights(archive,context['base'],restored,threads=2)
        decode_cpu=time.process_time()-decode_cpu;verify_cpu=time.process_time();verified=verify_roundtrip(checkpoint,restored,archive,threads=2)
        verify_cpu=time.process_time()-verify_cpu
        if (not verified['all_tensor_bytes_equal'] or encoded['base_sha256']!=context['base_sha'] or
            (expected_sha is not None and encoded['source_checkpoint_sha256']!=expected_sha)):
            raise ValueError('C full bytewise/base/frozen checkpoint identity failed')
        record.update(source_checkpoint_sha256=encoded['source_checkpoint_sha256'],archive_path=str(archive),archive_member='weights/'+archive.name,
            archive_sha256=encoded['archive_sha256'],archive_stat=signature(archive),model_fingerprint=encoded['model_fingerprint'],
            encode=encoded,decode=decoded,bytewise_verification=verified,additional_encoding_calls=1,
            encode_process_cpu_seconds=encode_cpu,decode_process_cpu_seconds=decode_cpu,verification_process_cpu_seconds=verify_cpu)
    if signature(checkpoint)!=before:raise ValueError('source checkpoint changed while preserving draft')
    record.update(wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes_process_cumulative=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cpu_threads=2,workers=1,gpu_used=False,shared_storage_written=False,original_checkpoints_and_zips_retained=True,
        source_authority='rc0/frozen training definition/config/inputs; source CP SHA frozen at draft encoding; fullphase gate pending')
    write(record_path,record)
    return record


def final_identity_manifest(context):
    path=ROOT/'artifacts/evaluation_ready_v5'/READY_NAMES[context['phase']['model']]/'manifest.json'
    marker=path.parent/'READY.json'
    if not marker.exists():return None
    before_ready=signature(marker);ready_raw=marker.read_bytes();ready=json.loads(ready_raw)
    ready_sha=hashlib.sha256(ready_raw).hexdigest()
    if (ready.get('ready') is not True or ready.get('status')!='ready' or
        ready.get('id')!=READY_NAMES[context['phase']['model']] or ready.get('mode')!='nll' or
        absolute(ready['manifest']['path']).resolve()!=path.resolve()):
        raise ValueError('C identity requires published READY for exact id/mode/manifest path')
    before_manifest=signature(path);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=ready['manifest']['sha256']:raise ValueError('published READY manifest SHA differs from current bytes')
    manifest=json.loads(raw);rows={r['id']:r for r in context['rows']}
    final=[r for r in manifest['records'] if r['kind']=='final']
    if len(final)!=len(rows) or {r['run_id'] for r in final}!=set(rows):raise ValueError('C final identity manifest incomplete/different cohort')
    for bound in final:
        row=rows[bound['run_id']];config=read(row['config_path'])
        if (bound['actual_run_id']!=row['id'] or bound['role']!='extended' or bound['phase']!='formal' or
            bound['phase_name']!=READY_NAMES[context['phase']['model']].removesuffix('_nll') or bound['group']!=row['group'] or
            Path(bound['config_path']).resolve()!=Path(row['config_path']).resolve() or
            Path(bound['summary_path']).resolve()!=Path(row['summary_path']).resolve() or bound['model_tag']!=context['phase']['model'] or
            bound['data_family']!='extended_v1' or Path(bound['checkpoint']).resolve()!=Path(row['checkpoint_path']).resolve() or
            bound['training_seed']!=row['seed'] or bound['ratio']!=row['ratio'] or bound['weight']!=row['weight'] or
            bound['target_prediction_tokens_at_checkpoint']!=context['phase']['target_prediction_tokens'] or
            bound['replay_prediction_tokens_at_checkpoint']!=expected_replay(config['target_prediction_tokens'],row['ratio']) or
            bound['checkpoint_bytes']!=Path(row['checkpoint_path']).stat().st_size or
            bound['checkpoint_mtime_ns']!=Path(row['checkpoint_path']).stat().st_mtime_ns):
            raise ValueError('C final manifest source/model/data/exposure identity mismatch')
    if signature(marker)!=before_ready or signature(path)!=before_manifest:raise ValueError('published READY/manifest changed while reading identity')
    return dict(path=str(path),sha256=digest,published_ready=dict(path=str(marker),sha256=ready_sha),records={r['run_id']:r for r in final})


def validate_final_pairs(context,record,identities):
    """A passed phase alone never permits D aliasing; bind every pair explicitly."""
    phase=context['phase'];rows={(r['seed'],r['group']):r for r in context['rows']}
    if (record.get('status')!='passed' or record.get('phase')!=phase['phase'] or
        record.get('model')!=phase['model'] or record.get('endpoint')!='final' or
        record.get('algorithm')!='sha256_sorted_key_dtype_shape_contiguous_uint8_v1' or
        len(record.get('comparisons',[]))!=3):raise ValueError('exact C final proof required')
    decisions={}
    for pair in record['comparisons']:
        seed=pair.get('seed');equal=pair.get('bitwise_model_state_equal')
        if (seed not in [11,22,33] or seed in decisions or type(equal) is not bool or
            pair.get('kind')!='identical_bank_null_control' or pair.get('endpoint')!='final'):
            raise ValueError('three unique explicit final pair decisions required')
        sides={s['run_id']:s for s in [pair['left'],pair['right']]}
        wanted={rows[seed,g]['id'] for g in ['B4','D']}
        if set(sides)!=wanted:raise ValueError('cross-seed/phase D alias forbidden')
        for group in ['B4','D']:
            row=rows[seed,group];side=sides[row['id']];identity=identities[row['id']]
            if (side.get('kind')!='final' or side.get('model_revision')!=context['protocol']['model_revision'] or
                side['checkpoint_sha256']!=identity['checkpoint_sha256'] or
                Path(side['checkpoint']).resolve()!=Path(row['checkpoint_path']).resolve() or
                side['config_sha256']!=file_sha256(row['config_path']) or
                side['summary_sha256']!=file_sha256(row['summary_path']) or
                side['checkpoint_exposure']['target_prediction_tokens_at_checkpoint']!=phase['target_prediction_tokens'] or
                side['checkpoint_exposure']['replay_prediction_tokens_at_checkpoint']!=expected_replay(phase['target_prediction_tokens'],row['ratio'])):
                raise ValueError('final equivalence side source identity differs')
        left=sides[rows[seed,'B4']['id']];right=sides[rows[seed,'D']['id']]
        if equal!=(left['model_state_fingerprint']==right['model_state_fingerprint']):
            raise ValueError('explicit pair equality flag contradicts fingerprints')
        decisions[seed]=dict(equal=equal,comparison=pair,b4=left,d=right)
    return decisions


def complete_phase_gate(context,state,master,cache,folder):
    """Only complete, independently audited cohorts can become delivery packages."""
    if not all(run_ready(r,context['definitions'][r['id']],state,master) for r in context['rows']):return None
    identity=final_identity_manifest(context)
    if identity is None:return None
    cache.bind(identity['path'],identity['sha256']);cache.bind(identity['published_ready']['path'],identity['published_ready']['sha256'])
    phase=context['phase'];sources=[proof(identity['path']),identity['published_ready']]
    if phase['model']=='pythia-2.8b':
        result=audit_phase(context['snapshot_path'],state,allow_incomplete=False,compare_weights=False)
        if result.get('status')!='passed' or result.get('completed_runs')!=15 or result.get('violations'):
            raise ValueError('complete 2.8B independent source audit failed')
        audit_path=folder/'fullphase_source_audit.json';write(audit_path,result);sources.append(proof(audit_path));decisions={}
    else:
        directory=ROOT/'results/extended_phase_equivalence'/phase['phase'];completion_path=directory/'completion.json'
        if not completion_path.exists():return None
        completion=read(completion_path)
        if (completion.get('status')!='passed' or completion.get('phase')!=phase['phase'] or
            completion.get('model')!=phase['model'] or completion.get('expected_runs')!=24 or
            completion.get('completed_runs')!=24 or completion.get('comparisons_computed')!=6):
            raise ValueError('full24 C source/proof completion required')
        for bound in [completion['source_registry'],completion['input_snapshot'],completion['audit'],
                      completion['proofs']['final'],*completion['source_inputs']]:
            cache.bind(bound['path'],bound['sha256']);sources.append(bound)
        if (completion['source_registry']['sha256']!=phase['active_registry_sha256'] or
            completion['input_snapshot']['sha256']!=phase['snapshot_sha256']):raise ValueError('different frozen C phase')
        audit=read(completion['audit']['path']);gate=audit.get('source_state_complete_gate',{})
        if (audit.get('status')!='passed' or audit.get('model')!=phase['model'] or audit.get('phase')!=phase['phase'] or
            audit.get('source_violations') or gate.get('ready') is not True or gate.get('completed_runs')!=24 or
            gate.get('missing_runs') or gate.get('failed_runs')):raise ValueError('C full source audit not passed')
        finalproof=read(completion['proofs']['final']['path'])
        if (finalproof['source_audit']!=completion['audit'] or finalproof['source_registry']!=completion['source_registry'] or
            finalproof['source_snapshot']!=completion['input_snapshot']):raise ValueError('proof/audit source binding differs')
        decisions=validate_final_pairs(context,finalproof,identity['records']);sources.append(proof(completion_path))
    return dict(status='complete_source_gate_passed',identity_manifest=identity,decisions=decisions,sources=sources)


def base_closure(context):
    """410M depends on the durable primary closure; new models include native files."""
    model=context['phase']['model'];members=[]
    if model=='pythia-410m':
        receipt=read(ROOT/'docs/primary_weights_durable_publication.json');current=verify_primary_stat(receipt,PRIMARY_TAR)
        if receipt.get('source_sha256',receipt.get('sha256'))!=PRIMARY_SHA:
            # Publisher receipts use verified_prefix_sha256 in current versions.
            if receipt.get('verified_prefix_sha256')!=PRIMARY_SHA:raise ValueError('primary publication SHA differs')
        with tarfile.open(PRIMARY_TAR,'r:') as archive:
            raw=archive.extractfile('primary_final_weights_manifest.json').read()
            import hashlib
            if hashlib.sha256(raw).hexdigest()!=PRIMARY_MANIFEST_SHA:raise ValueError('primary small manifest changed')
            manifest=json.loads(raw);first=next(r for r in manifest['logical_runs'] if r.get('archive_member'))
            with archive.extractfile(first['archive_member']) as member,zipfile.ZipFile(member) as zipped:
                metadata=read_zip_metadata(zipped)
            base_tensor_sha=[{k:r[k] for k in ['name','shape','dtype','base_fp32_raw_sha256']} for r in metadata['tensors']]
        if verify_primary_stat(receipt,PRIMARY_TAR)!=current:raise ValueError('primary dependency changed during small-member read')
        dependency=dict(kind='durable_external_primary_base_closure',tar=str(PRIMARY_TAR),tar_sha256=PRIMARY_SHA,
            primary_manifest_sha256=PRIMARY_MANIFEST_SHA,current_identity_stat=current,base=metadata['base'],base_files=manifest['base_files'],
            base_tensor_metadata=base_tensor_sha,verification='historical full-TAR acceptance + current identity/stat + pinned small manifest; no new full-TAR hash',
            offline_restore='Verify primary TAR SHA, extract base/pythia-410m-final and combine with this package. No download required.')
        if context['base_sha']!=metadata['base']['sha256']:raise ValueError('410M base differs from durable closure')
    else:
        files=[]
        for name,record in context['snapshot']['model_files'].items():
            member='base/'+model+'-final/'+Path(name).name;members.append((Path(name),member,record['sha256']))
            files.append(dict(source=name,member=member,**record))
        dependency=dict(kind='native_base_config_tokenizer_readme_included',base_sha256=context['base_sha'],
                        model_revision=context['protocol']['model_revision'],files=files)
    return dependency,members


def read_zip_metadata(archive):
    # Codec metadata is tiny relative to its tensor payloads; no tensor reads needed.
    return json.loads(archive.read('metadata.json'))


def assemble(context,gate,records,folder,cache):
    rows=context['rows'];ids={r['id'] for r in rows};lookup={r['run_id']:r for r in records};phase=context['phase']
    if gate.get('status')!='complete_source_gate_passed' or set(lookup)!=ids or len(records)!=len(rows):
        raise ValueError('partial C drafts cannot become a delivery package')
    cache.stable();started,cpu=time.monotonic(),time.process_time();members=[]
    identities=gate['identity_manifest']['records']
    for row in rows:
        item=lookup[row['id']]
        if (item['source_checkpoint_sha256']!=identities[row['id']]['checkpoint_sha256'] or
            signature(item['source_checkpoint'])!=item['source_stat'] or item['model_tag']!=phase['model'] or
            item['phase_name']!=phase['phase'] or item['data_family']!='extended_v1'):
            raise ValueError('C draft no longer matches complete final identity')
        scheduler=item.get('completed_scheduler_record') or {}
        if (scheduler.get('status')!='completed' or scheduler.get('returncode')!=0 or
            scheduler.get('definition_sha256')!=item['definition_sha256'] or scheduler.get('missing_outputs') or
            item['definition_sha256']!=digest_job(context['definitions'][row['id']]) or
            'stop_requested_at' in scheduler or scheduler.get('argv')!=context['definitions'][row['id']]['argv'] or
            item['config_sha256']!=file_sha256(row['config_path']) or item['summary_sha256']!=file_sha256(row['summary_path']) or
            item['events_sha256']!=file_sha256(Path(row['output_dir'])/'events.jsonl')):
            raise ValueError('C draft rc0/source metadata evidence incomplete or changed')
        members.append((Path(item['metadata_path']),item['metadata_member'],item['metadata_sha256']))
        if 'alias_of' in item:
            decision=gate['decisions'][item['seed']];canonical=lookup[item['alias_of']]
            if (row['group']!='D' or not decision['equal'] or canonical['group']!='B4' or
                item['model_fingerprint']!=canonical['model_fingerprint'] or
                item['model_fingerprint']!=decision['d']['model_state_fingerprint']):
                raise ValueError('D alias lacks source-bound exact final equality')
        else:
            members.append((Path(item['archive_path']),item['archive_member'],item['archive_sha256']))
    dependency,base_members=base_closure(context);members+=base_members
    manifest=dict(format='extended_C_final_lossless_delivery_v1',status='complete_phase_all_final_roundtrips_verified',
        model_tag=phase['model'],phase_name=phase['phase'],role='extended',data_family='extended_v1',kind='fixed_exposure_final',
        logical_final_endpoints=len(rows),unique_weights=sum('archive_path' in r for r in records),
        alias_count=sum('alias_of' in r for r in records),
        logical_runs=[dict(r,delivery_manifest_status='complete_phase_final_verified') for r in records],base_dependency=dependency,
        fullphase_gate=gate,freeze=proof(FREEZE),phase_source=phase,
        scope='All frozen model-specific C final endpoints; excludes C best, A, B60, and other model/horizon strata',
        original_checkpoints_zips_restored_retained=True,optimizer_included=False,shared_storage_written=False,
        cpu_threads=2,workers=1,gpu_used=False,publisher_required=True)
    manifest_path=folder/'extended_C_final_manifest.json';write(manifest_path,manifest)
    instructions=folder/'RESTORE_C_FINAL.md'
    instructions.write_text('# Extended C final weights\n\nOnly the complete frozen model-specific final cohort is in this package. Every independently encoded model passed bytewise comparison with its source checkpoint. D aliases exist only with explicit source-bound exact final-state equality. Each alias retains its own full metadata sidecar.\n\nFor 410M, verify the primary_final_weights_lossless_v1.tar SHA in base_dependency, then extract its base/pythia-410m-final directory. For 1B/2.8B, native base/config/tokenizer/README are included under base/. No network download is needed.\n\n```bash\npython tools/scripts/archive_model_weights.py --threads 2 decode --archive weights/RUN.xor.zip --base base/MODEL-final/model.safetensors --output restored.pt\n```\n\nFor D aliases decode the referenced B4 weight member and use D own metadata sidecar for logical identity/provenance/exposure. The restored file contains model weights; full original nonweight/nonoptimizer metadata is preserved separately in metadata/*.source_metadata.pt. Optimizer state is excluded; full Adam resume is not claimed. Source checkpoints, ZIPs and verification outputs are retained in scratch. A separate durable publication receipt is required.\n')
    extras=[(manifest_path,manifest_path.name),(instructions,instructions.name),
        (Path(__file__),'tools/provenance/export_C_final_weights.py'),
        (ROOT/'src/replay_study/weight_archive.py','tools/src/replay_study/weight_archive.py'),
        (ROOT/'src/replay_study/__init__.py','tools/src/replay_study/__init__.py'),
        (ROOT/'scripts/archive_model_weights.py','tools/scripts/archive_model_weights.py'),
        (FREEZE,'provenance/extended_formal_freeze.json'),
        (absolute(phase['active_registry']),'provenance/registry.json'),
        (absolute(phase['snapshot']),'provenance/input_snapshot.json'),
        (absolute(phase['protocol']),'provenance/protocol.json')]
    seen=set()
    for source in gate['sources']:
        path=absolute(source['path'])
        if str(path) not in seen:
            extras.append((path,'provenance/gate/'+str(len(seen))+'_'+path.name));seen.add(str(path))
    for path,name in extras:members.append((path,name,file_sha256(path)))
    cache.stable()
    tar_path=folder/('extended_C_'+phase['model'].replace('pythia-','').replace('.','p')+'_final_weights_lossless_v1.tar')
    receipt=build_verified_tar(tar_path,members,LIMITS[phase['model']]);cache.stable()
    receipt.update(status='passed',tar_path=str(tar_path),model=phase['model'],phase=phase['phase'],
        logical_final_endpoints=len(rows),unique_weights=manifest['unique_weights'],alias_count=manifest['alias_count'],
        manifest_sha256=file_sha256(manifest_path),fullphase_source_gate_passed=True,
        wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
        timer_scope='assembly only; draft encode/decode/bytewise receipts are prior child operations, not added twice',
        shared_storage_written=False,publisher_required=True,original_checkpoints_zips_restored_retained=True)
    write(folder/'completion.json',receipt);return receipt


def process_phase(context,state,master,cache,folder,gate=None,should_stop=lambda:False):
    """Draft ready runs immediately; defer D until complete source-bound final proof."""
    folder.mkdir(parents=True,exist_ok=True);records=[]
    rows=sorted(context['rows'],key=lambda r:(r['group']=='D',r['seed'],r['group']))
    for row in rows:
        if should_stop():return None
        if not run_ready(row,context['definitions'][row['id']],state,master):continue
        if row['group']=='D' and gate is None:continue
        bound=gate['identity_manifest']['records'][row['id']] if gate else None
        expected=None if bound is None else bound['checkpoint_sha256'];fingerprint=None;alias=None
        if row['group']=='D':
            decision=gate['decisions'][row['seed']];fingerprint=decision['d']['model_state_fingerprint']
            if decision['equal']:
                canonical=next(r for r in records if r['run_id']==decision['b4']['run_id'])
                if canonical['model_fingerprint']!=decision['b4']['model_state_fingerprint']:
                    raise ValueError('B4 draft differs from independently proven model state')
                alias=dict(run_id=canonical['run_id'],model_fingerprint=canonical['model_fingerprint'],
                           pair=decision['comparison'],proof_sources=gate['sources'])
        cache.stable();item=draft_one(context,row,folder,context['definitions'][row['id']],expected,fingerprint,alias,
                                     completed_record=state['jobs'][row['id']])
        records.append(item)
        write(folder/'draft_progress.json',dict(status='drafts_only_not_deliverable',verified_logical=len(records),
            expected_logical=len(rows),unique_encoded=sum('archive_path' in r for r in records),
            last_run=row['id'],model=context['phase']['model'],updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            first_model_benchmark=next((dict(run_id=r['run_id'],wall_seconds=r['wall_seconds'],process_cpu_seconds=r['process_cpu_seconds'],
                archive_bytes=r['encode']['archive_bytes'],source_model_raw_bytes=r['encode']['model_raw_bytes'],
                compression_ratio=r['encode']['compression_ratio_archive_over_model_raw'],encode_process_cpu_seconds=r['encode_process_cpu_seconds'],
                peak_rss_bytes_process_cumulative=r['peak_rss_bytes_process_cumulative']) for r in records if 'encode' in r),None)))
    if not should_stop() and gate is not None and len(records)==len(rows):return assemble(context,gate,records,folder,cache)
    return None


def run(output=OUTPUT,execute=False,poll_seconds=30,until='2026-11-07T03:00:00Z'):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError("set CUDA_VISIBLE_DEVICES='' explicitly")
    torch.set_num_threads(2);output=Path(output).resolve()
    if not output.is_relative_to(Path('/tmp/grad-research-20261106/delivery')):raise ValueError('scratch/delivery outputs only')
    started,cpu=time.monotonic(),time.process_time();contexts,cache=load_contexts();output.mkdir(parents=True,exist_ok=True)
    for path in [__file__,ROOT/'src/replay_study/weight_archive.py',ROOT/'scripts/export_remaining_primary_endpoints.py',
                 ROOT/'scripts/export_A_selected_weights.py',ROOT/'scripts/audit_extended_runs.py']:
        cache.bind(path,file_sha256(path))
    plan=dict(status='prepared_only',logical_final_endpoints=63,expected_model_counts={c['phase']['model']:len(c['rows']) for c in contexts},
        frozen_files=cache.files,mutable_master_policy='each frozen training definition and rc0 record bound; append/reorder allowed',
        draft_policy='per-run rc0 plus config/inputs/execution/exposure audit; never complete delivery before whole cohort gate',
        alias_policy='D waits for explicit per-pair source-bound final proof; false -> independent encoding',
        size_ceilings=LIMITS,reserved_paths={m:'/data/202610/7kb/research_delivery/'+n for m,n in RESERVES.items()},
        workers=1,cpu_threads=2,gpu_used=False,shared_storage_written=False,
        worker_lock='/tmp/grad-research-20261106/delivery/.supplemental_weight_export.lock',
        wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
    write(output/'preparation_review.json',plan)
    if not execute:print(json.dumps(dict(status='prepared_only',output=str(output),logical=63)));return
    stop=threading.Event();deadline=dt.datetime.fromisoformat(until.replace('Z','+00:00'))
    def should_stop():return stop.is_set() or dt.datetime.now(dt.timezone.utc)>=deadline
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.set())
    lockpath=Path(plan['worker_lock']);done={};gates={}
    with lockpath.open('a+b') as lock:
        while not should_stop():
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                write(output/'progress.json',dict(status='waiting_shared_cpu_worker_lock',gpu_used=False,shared_storage_written=False))
                stop.wait(poll_seconds)
        if should_stop():return
        while not should_stop() and len(done)<len(contexts):
            cache.stable();state=read(STATE);master=read(MASTER)
            for context in contexts:
                name=context['phase']['model'];folder=output/name
                if name in done:continue
                folder.mkdir(parents=True,exist_ok=True)
                if (folder/'completion.json').exists():
                    previous=read(folder/'completion.json')
                    if previous.get('status')!='passed' or file_sha256(previous['tar_path'])!=previous['tar_sha256']:
                        raise ValueError('earlier completed C package no longer valid')
                    done[name]=previous;continue
                if name not in gates:
                    gate=complete_phase_gate(context,state,master,cache,folder)
                    if gate is not None:gates[name]=gate
                completed=process_phase(context,state,master,cache,folder,gates.get(name),should_stop)
                if completed:done[name]=completed
                if should_stop():break
            write(output/'progress.json',dict(status='complete' if len(done)==3 else 'watching_validation_metadata_only',
                completed_models=done,elapsed_wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,
                cpu_threads=2,workers=1,gpu_used=False,shared_storage_written=False,
                updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
            if len(done)<3:stop.wait(poll_seconds)
    if should_stop():write(output/'stopped.json',dict(status='stopped_between_operations',drafts_retained=True,shared_storage_written=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--poll-seconds',type=float,default=30)
    parser.add_argument('--until',default='2026-11-07T03:00:00Z')
    args=parser.parse_args()
    if args.poll_seconds<1:parser.error('poll-seconds must be >=1')
    try:run(args.output_dir,args.execute,args.poll_seconds,args.until)
    except Exception as error:
        if args.output_dir.resolve().is_relative_to(Path('/tmp/grad-research-20261106/delivery')):
            write(args.output_dir/'error.json',dict(status='stopped_requires_root_review',error=repr(error),
                drafts_retained=True,gpu_used=False,shared_storage_written=False))
        raise
