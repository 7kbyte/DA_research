#!/usr/bin/env python3
"""Read-only first-model C export receipt review; no scoring or large-payload hash."""
from __future__ import annotations
import argparse,datetime as dt,json,os,stat,sys,time,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from replay_study.weight_archive import file_sha256
from export_remaining_primary_endpoints import signature,checkpoint_metadata,same_metadata,write
from export_A_selected_weights import verify_primary_stat
from schedule_jobs import digest_job
from audit_formal_runs import expected_replay
MODELS=['pythia-410m','pythia-1b','pythia-2.8b']
DELIVERY=Path('/tmp/grad-research-20261106/delivery/extended_C_final_v1')
A_RECEIPT=ROOT/'docs/A_selected_weights_durable_publication.json'
A_SHA='0dfec20b1bf9dde38b0d0d453c821eb088490bde188dc0e19bc498cc37b04631'
A_BYTES=16_949_309_440
REPORT=ROOT/'docs/C首例权重资源独立复核.json'
TEXT=ROOT/'docs/C首例权重资源实测说明.md'

def read(path):return json.loads(Path(path).read_text())
def bound(path):return dict(path=str(path),sha256=file_sha256(path))
def require(value,message):
    if not value:raise ValueError(message)

def audit_record(path):
    record=read(path);encoded=record['encode'];decoded=record['decode'];verified=record['bytewise_verification']
    require(record['status']=='verified_draft_not_deliverable' and record['role']=='extended' and
        record['data_family']=='extended_v1' and record['kind']=='fixed_exposure_final','C final draft identity required')
    require(record['workers']==1 and record['cpu_threads']==2 and record['gpu_used'] is False and
        record['shared_storage_written'] is False,'CPU-only single-worker receipt required')
    scheduler=record['completed_scheduler_record'];definition=record['definition']
    require(scheduler.get('status')=='completed' and scheduler.get('returncode')==0 and
        'stop_requested_at' not in scheduler and scheduler.get('argv')==definition['argv'] and
        scheduler['definition_sha256']==record['definition_sha256']==digest_job(definition),
        'completed actual training argv/definition/unstopped source binding differs')
    freeze=read(ROOT/'results/extended_formal_freeze.json')
    phase=next(p for p in freeze['phases'] if p['model']==record['model_tag'])
    require(file_sha256(phase['active_registry'])==phase['active_registry_sha256'] and
        file_sha256(phase['snapshot'])==phase['snapshot_sha256'],'frozen model-specific registry/snapshot differs')
    row=next(r for r in read(phase['active_registry']) if r['id']==record['run_id']);snapshot=read(phase['snapshot'])
    config=read(row['config_path']);relative=str(Path(row['config_path']).relative_to(ROOT))
    require(record['phase_name']==phase['phase'] and record['input_snapshot_sha256']==phase['snapshot_sha256'] and
        Path(record['source_checkpoint']).resolve()==Path(row['checkpoint_path']).resolve() and
        file_sha256(row['config_path'])==record['config_sha256']==snapshot['configs'][relative]['sha256'] and
        file_sha256(row['summary_path'])==record['summary_sha256'] and
        file_sha256(Path(row['output_dir'])/'events.jsonl')==record['events_sha256'],
        'first draft source does not bind frozen C model/registry/config/summary/events')
    for pathkey,statkey in [('source_checkpoint','source_stat'),('archive_path','archive_stat'),('metadata_path','metadata_stat')]:
        require(signature(record[pathkey])==record[statkey],pathkey+' current stat differs from accepted record')
    require(file_sha256(record['metadata_path'])==record['metadata_sha256'],'small metadata sidecar SHA differs')
    with zipfile.ZipFile(record['archive_path']) as archive:metadata=json.loads(archive.read('metadata.json'))
    fingerprints=[record['model_fingerprint'],encoded['model_fingerprint'],decoded['model_fingerprint'],
        verified['model_fingerprint'],metadata['model_fingerprint']]
    require(all(f==fingerprints[0] for f in fingerprints),'encode/decode/bytewise/ZIP fingerprints differ')
    require(record['source_checkpoint_sha256']==encoded['source_checkpoint_sha256']==verified['source_checkpoint_sha256']==metadata['source_checkpoint']['sha256'],
        'source checkpoint whole-SHA receipt binding differs')
    require(record['archive_sha256']==encoded['archive_sha256']==verified['archive_sha256'],'archive whole-SHA receipt binding differs')
    require(encoded['base_sha256']==metadata['base']['sha256'],'base SHA receipt binding differs')
    tensors=metadata['tensors'];raw_bytes=sum(t['raw_bytes'] for t in tensors);elements=sum(t['elements'] for t in tensors)
    require(all(t['dtype']=='torch.float32' and t['raw_bytes']==4*t['elements'] and len(t['raw_sha256'])==64 and len(t['base_fp32_raw_sha256'])==64 for t in tensors),
        'all source tensor FP32 raw bytes and base/raw SHA required')
    require(raw_bytes==encoded['model_raw_bytes']==metadata['model_raw_bytes'] and elements==fingerprints[0]['state_elements'],
        'FP32 source byte/element accounting differs')
    require(verified['status']=='passed' and verified['all_tensor_bytes_equal'] is True and
        len(tensors)==verified['tensor_count']==decoded['tensor_raw_hashes_verified'],'full producer tensor-byte verification absent')
    saved=torch.load(record['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    sidecar=torch.load(record['metadata_path'],map_location='cpu',weights_only=False)
    require(same_metadata(checkpoint_metadata(saved),sidecar['original_checkpoint_metadata']),'independent small original metadata comparison failed')
    require(saved['config']==config and record['model_revision']==config['model_revision'] and
        metadata['base']['model_revision']==record['model_revision'] and
        Path(metadata['base']['path']).resolve()==Path(config['model_name_or_path'],'model.safetensors').resolve() and
        metadata['base']['sha256']==snapshot['model_files'][str(Path(config['model_name_or_path'],'model.safetensors'))]['sha256'],
        'source/base/config model identity differs')
    source_state=saved['model_state']
    require(set(source_state)=={t['name'] for t in tensors} and
        all(source_state[t['name']].dtype==torch.float32 and list(source_state[t['name']].shape)==t['shape'] and source_state[t['name']].numel()==t['elements'] for t in tensors),
        'source mmap tensor descriptors differ from ZIP metadata')
    require(saved['counters']['target_prediction_tokens']==record['actual_exposure']['target_prediction_tokens'] and
        saved['counters']['replay_prediction_tokens']==record['actual_exposure']['replay_prediction_tokens'] and
        saved['counters']['target_prediction_tokens']==phase['target_prediction_tokens'] and
        saved['counters']['replay_prediction_tokens']==expected_replay(phase['target_prediction_tokens'],row['ratio']),
        'actual frozen final exposure metadata differs')
    del source_state,saved,sidecar
    timers=dict(encode_wall_seconds=encoded['wall_seconds'],decode_wall_seconds=decoded['wall_seconds'],
        verify_wall_seconds=verified['wall_seconds'],full_roundtrip_record_wall_seconds=record['wall_seconds'],
        encode_cpu_seconds=record['encode_process_cpu_seconds'],decode_cpu_seconds=record['decode_process_cpu_seconds'],
        verify_cpu_seconds=record['verification_process_cpu_seconds'],full_record_cpu_seconds=record['process_cpu_seconds'])
    require(sum(timers[k] for k in ['encode_wall_seconds','decode_wall_seconds','verify_wall_seconds'])<=record['wall_seconds']+1e-6,'child wall times cannot exceed parent')
    require(sum(timers[k] for k in ['encode_cpu_seconds','decode_cpu_seconds','verify_cpu_seconds'])<=record['process_cpu_seconds']+1e-6,'child CPU times cannot exceed parent')
    require(Path(decoded['restored']).stat().st_size==decoded['restored_bytes'],'restored current size differs')
    return dict(status='passed_read_only_record_and_metadata_review',model=record['model_tag'],run_id=record['run_id'],
        source_record=bound(path),source_registry=bound(phase['active_registry']),source_snapshot=bound(phase['snapshot']),
        actual_exposure=record['actual_exposure'],source_checkpoint_sha256=record['source_checkpoint_sha256'],
        archive_sha256=record['archive_sha256'],model_fingerprint=record['model_fingerprint'],tensor_count=len(tensors),
        fp32_model_raw_bytes=raw_bytes,zip_bytes=encoded['archive_bytes'],zip_over_fp32_raw=encoded['archive_bytes']/raw_bytes,
        base_sha256=encoded['base_sha256'],timers=timers,peak_rss_bytes_process_cumulative=record['peak_rss_bytes_process_cumulative'],
        peak_rss_interpretation='entire exporter process high-water mark through this record; not this model alone or an increment',
        producer_reported_all_tensor_bytes_equal=True,independent_metadata_roundtrip=True,
        independent_read_scope='small record and ZIP metadata, source/sidecar nonweight metadata, source mmap tensor descriptors, current file stats; no repeat full CP/ZIP/restored payload hash or byte comparison',
        original_cp_zip_restored_retained=True,source_authority=record['source_authority'])

def audit_A_publication():
    if not A_RECEIPT.exists():return dict(status='pending_publication_receipt')
    receipt=read(A_RECEIPT);destination=Path(receipt['destination']);observed=verify_primary_stat(receipt,destination)
    require(receipt['status']=='passed' and receipt['bytes']==A_BYTES and receipt['source_sha256']==A_SHA and
        receipt['destination_sha256']==A_SHA,'A publisher accepted identity differs')
    require(all(receipt.get(k) is True for k in ['file_fsync_after_copy','prefix_sha_verified_before_truncate',
        'file_fsync_after_truncate','parent_directory_fsync','original_inode_preserved','permissions_and_ownership_preserved']),
        'publisher fsync/inode/owner evidence missing')
    return dict(status='passed_current_identity_stat_review',receipt=bound(A_RECEIPT),destination=str(destination),
        bytes=A_BYTES,sha256=A_SHA,current_stat=observed,
        fresh_full_tar_hash_performed_by_this_reviewer=False,
        verification='publisher full source/copy/readback SHA acceptance + independent current inode/dev/owner/permissions/size/mtime/ctime; no new full TAR hash')

def render(value):
    lines=['# C 首例权重资源实测独立复核','',f"复核状态：{value['status']}。仅工程记录与文件身份审查，不读取 heldout NLL 或行为评分，不改变运行源码、原始记录或参数。",'',
        '| 模型 | 首个完整记录 | FP32 原字节 | ZIP 字节 | ZIP/raw | 往返 wall 秒 | 往返 CPU 秒 | 进程累计 RSS GiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        r=value['models'].get(model)
        if r:lines.append(f"| {model} | {r['run_id']} | {r['fp32_model_raw_bytes']:,} | {r['zip_bytes']:,} | {r['zip_over_fp32_raw']:.4f} | {r['timers']['full_roundtrip_record_wall_seconds']:.2f} | {r['timers']['full_record_cpu_seconds']:.2f} | {r['peak_rss_bytes_process_cumulative']/2**30:.3f} |")
        else:lines.append(f'| {model} | 尚未出现完整记录 | — | — | — | — | — | — |')
    lines+=['','| 模型 | encode wall/CPU 秒 | decode wall/CPU 秒 | 原 CP 全 uint8 验收 wall/CPU 秒 | tensor 数 |',
        '|---|---:|---:|---:|---:|']
    for model,r in value['models'].items():
        t=r['timers'];lines.append(f"| {model} | {t['encode_wall_seconds']:.2f}/{t['encode_cpu_seconds']:.2f} | {t['decode_wall_seconds']:.2f}/{t['decode_cpu_seconds']:.2f} | {t['verify_wall_seconds']:.2f}/{t['verify_cpu_seconds']:.2f} | {r['tensor_count']} |")
    lines+=['','每模型单例不是压缩上界或未来速度保证。ZIP/raw 的分母为全部 FP32 model_state 原始字节，不是包含 optimizer 的完整训练 checkpoint。编码、解码、逐字节验收计时是完整往返记录的嵌套子项，不与父项再次相加。父项另含 source metadata、部分 SHA/文件操作等开销。',
        '', 'CPU 秒是进程使用的处理器时间，wall 秒是实际经过时间；两者不相同。这里的单模型 timer 从源 metadata/编码开始，不包含此前等待 A 的 worker 锁；daemon 自启动 wall 还可能包含锁等待、轮询和无可导出 run 的等待，不能按其差额虚报精确等待时长。顶层 progress 在模型工作期间未刷新也不能据此判断卡住。',
        '', 'RSS 字段是整个 exporter 进程截至该记录的累计高水位，不能解释为该模型独占峰值、增量或同时运行内存总和。worker=1、torch threads=2、GPU=false；单例结果不能泛化到不同系统负载。',
        '', '原生产者对每个模型已执行 encode→decode→原 checkpoint 全 tensor uint8 比较，记录 all_tensor_bytes_equal=true、tensor 数、模型指纹及完整 source/ZIP SHA；本次独立复核核它们的一致关联、FP32 字节数和每 tensor 原始/基底 SHA 元数据，并实际比较原 CP 与 sidecar 的全部非权重/非 optimizer metadata、mmap tensor 描述及当前文件 stat。本次没有重复扫描全部原 CP/ZIP/恢复权重内容，不能称第二次独立全字节验收。原文件均保留。',
        '', '这些记录仍是 verified_draft_not_deliverable。完整 phase、READY 身份、独立审计、逐对 D/B4 final 证明通过以前，不能称完整模型包或持久化交付；D 是否 alias 由完整 final 证明决定。',
        '',f"A 发布复核：{value['A_publication']['status']}。若已完成，使用 publisher 的全源、复制及读回 SHA 验收证据，另独立复核当前 owner/inode/dev/permissions/size/mtime/ctime；本次没有重复 hash 整个 TAR。所有源记录及收据 SHA 详见同名 JSON。",'']
    TEXT.write_text('\n'.join(lines))

def pass_once(previous):
    current=dict(previous);current.setdefault('models',{})
    for model in MODELS:
        if model in current['models']:
            old=current['models'][model];require(file_sha256(old['source_record']['path'])==old['source_record']['sha256'],'already-reviewed original record changed');continue
        files=sorted((DELIVERY/model/'records').glob('*.json'),key=lambda p:p.stat().st_mtime_ns)
        eligible=[p for p in files if 'encode' in read(p)]
        if eligible:
            audited=audit_record(eligible[0]);require(audited['model']==model,'first-record directory/model differs')
            current['models'][model]=audited
    current['A_publication']=audit_A_publication()
    current.update(status='complete' if len(current['models'])==3 and current['A_publication']['status'].startswith('passed') else 'partial_pending_future_records_or_publication',
        observed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source=bound(__file__),no_model_scoring=True,large_payloads_rehashed=False)
    write(REPORT,current);render(current);return current

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');parser.add_argument('--until',default='2026-11-07T02:10:00Z');args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('explicit empty CUDA_VISIBLE_DEVICES required')
    torch.set_num_threads(2);value={};deadline=dt.datetime.fromisoformat(args.until.replace('Z','+00:00'))
    while True:
        try:value=pass_once(value)
        except Exception as error:
            write(REPORT.with_name(REPORT.stem+'_error.json'),dict(status='read_only_review_failed_requires_attention',
                error=repr(error),observed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source=bound(__file__),
                original_records_modified=False,scoring_performed=False))
            raise
        if not args.watch or value['status']=='complete' or dt.datetime.now(dt.timezone.utc)>=deadline:break
        time.sleep(60)
    print(json.dumps(dict(status=value['status'],models=list(value['models']),A=value['A_publication']['status'])))
if __name__=='__main__':main()
