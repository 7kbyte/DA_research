"""Fresh full fingerprint of NEW isolated recovered tensors; no live imports."""
from pathlib import Path
import hashlib
import json
import time
import zipfile
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fresh_fingerprint(state):
    # Same published fingerprint convention, implemented independently of codec.
    digest = hashlib.sha256(); elements = 0; tensor_bytes = 0
    for name, tensor in sorted(state.items()):
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu' and tensor.dtype == torch.float32,
                'CPU FP32 recovered tensors required')
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw)); elements += tensor.numel(); tensor_bytes += raw.nbytes
    return dict(sha256_model_state=digest.hexdigest(), state_elements=elements), tensor_bytes


def main():
    started = time.monotonic(); cpu = time.process_time()
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
    plan = json.loads(Path('bound_plan.json').read_text()); expected = plan['selected']
    restored = Path('restored/B4_seed11_final.pt')
    saved = torch.load(restored, map_location='cpu', mmap=True, weights_only=False)
    fp, raw_bytes = fresh_fingerprint(saved['model_state'])
    require(fp == expected['model_fingerprint'] and raw_bytes == plan['model_raw_bytes'] and
            len(saved['model_state']) == plan['tensor_count'], 'Fresh entire recovered model fingerprint differs')
    require(saved.get('checkpoint_kind') == 'model_only_lossless_xor_restored' and 'optimizer_state' not in saved,
            'Model-only recovery expected; optimizer resume is unavailable')
    provenance = saved['lossless_delivery_provenance']
    with zipfile.ZipFile(expected['archive_member']) as archive:
        zip_metadata = json.loads(archive.read('metadata.json'))
    for key in ('source_checkpoint_sha256', 'model_fingerprint'):
        require(provenance.get(key) == expected[key], 'Source/whole-model provenance differs')
    require(provenance.get('source_checkpoint_kind') == zip_metadata['source_checkpoint_kind'] and
            zip_metadata['source_checkpoint']['sha256'] == expected['source_checkpoint_sha256'] and
            zip_metadata['model_fingerprint'] == fp and
            provenance.get('archive_sha256') == expected['archive_sha256'] and
            provenance.get('base_sha256') == expected['encode']['base_sha256'], 'CP kind/ZIP/base provenance differs')
    exposure = expected['actual_exposure']
    require(all(saved['counters'][key] == exposure[key] for key in ('target_prediction_tokens', 'replay_prediction_tokens')),
            'Recovered final exposure differs')
    report = json.loads(Path(str(restored) + '.decode.json').read_text())
    require(report.get('status') == 'decoded_and_hash_verified' and report.get('tensor_raw_hashes_verified') == plan['tensor_count'] and
            report.get('model_fingerprint') == fp and report.get('gpu_used') is False, 'Decoder all-tensor verification differs')
    sidecar = torch.load(expected['metadata_member'], map_location='cpu', weights_only=False)
    require(sidecar['logical_identity']['run_id'] == plan['run_id'] and sidecar['actual_exposure'] == exposure,
            'Selected own sidecar identity/exposure differs')
    # Fresh output serialized hash is independent of the fingerprint pass.
    h = hashlib.sha256(); serialized_bytes = 0
    with restored.open('rb') as stream:
        for raw in iter(lambda: stream.read(8 << 20), b''):
            h.update(raw); serialized_bytes += len(raw)
    require(h.hexdigest() == report['restored_file_sha256'], 'Fresh recovered serialized file SHA differs from decoder report')
    result = dict(status='passed', model=plan['model'], run_id=plan['run_id'], fingerprint=fp,
        tensor_count=plan['tensor_count'], independent_tensor_payload_bytes_hashed=raw_bytes,
        fresh_new_restored_file_sha256=h.hexdigest(), fresh_new_restored_serialized_bytes_hashed=serialized_bytes,
        source_checkpoint_sha256=provenance['source_checkpoint_sha256'], original_checkpoint_read=False,
        archive_only_native_base=True, original_CP_fresh_uint8_comparison=False, Adam_resume_verified=False,
        GPU_used=False, wall_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
        timer_scope='Checker body after imports; nested in external checker and whole recovery child timers')
    Path('independent_fingerprint.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
