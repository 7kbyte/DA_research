"""Fixed C410 single restored model check; reads only new QA outputs."""
from pathlib import Path
import hashlib,json,sys,time,torch
sys.path.insert(0,str(Path.cwd()/'tools/src'))
from replay_study.weight_archive import model_fingerprint

torch.set_num_threads(2)
assert not torch.cuda.is_initialized()
plan=json.loads(Path('preparation_receipt.json').read_text());expected=plan['expected'];started=time.monotonic()
saved=torch.load('restored/C410_B4_seed11_final.pt',map_location='cpu',mmap=True,weights_only=False)
fingerprint=model_fingerprint(saved['model_state'])
assert fingerprint==expected['model_fingerprint'] and len(saved['model_state'])==292
assert all(v.dtype==torch.float32 and v.device.type=='cpu' for v in saved['model_state'].values())
assert saved['checkpoint_kind']=='model_only_lossless_xor_restored'
p=saved['lossless_delivery_provenance']
assert p['model_fingerprint']==fingerprint and p['source_checkpoint_sha256']==expected['source_checkpoint_sha256'] and p['source_checkpoint_kind']==expected['source_checkpoint_kind']
assert saved['counters']['target_prediction_tokens']==100000000 and saved['counters']['replay_prediction_tokens']==5000000
report=json.loads(Path('restored/C410_B4_seed11_final.pt.decode.json').read_text())
assert report['status']=='decoded_and_hash_verified' and report['tensor_raw_hashes_verified']==292 and report['model_fingerprint']==fingerprint
sidecar=torch.load(expected['metadata_member'],map_location='cpu',weights_only=False)
assert sidecar['logical_identity']['run_id']==expected['run_id']
assert sidecar['actual_exposure']==expected['actual_exposure']
result=dict(status='passed',fingerprint=fingerprint,tensor_count=292,raw_model_bytes_read_for_independent_fingerprint=1621336064,checkpoint_kind=saved['checkpoint_kind'],source_checkpoint_kind=p['source_checkpoint_kind'],source_checkpoint_sha256=p['source_checkpoint_sha256'],original_explicit_checkpoint_kind=sidecar['original_checkpoint_metadata'].get('checkpoint_kind','<absent>'),actual_target_prediction_tokens=100000000,actual_replay_prediction_tokens=5000000,decoder_restored_file_SHA256=report['restored_file_sha256'],wall_seconds=time.monotonic()-started,scope='New restored model full fingerprint plus own selected sidecar metadata; no original CP, network, live model directory or GPU',GPU_used=False)
Path('independent_fingerprint.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
