from pathlib import Path
import json,sys,time,torch
sys.path.insert(0,str(Path.cwd()/'tools/src'))
from replay_study.weight_archive import model_fingerprint

torch.set_num_threads(2)
expected={'main_B4_seed11_final':('63069435cb5ea77931a58d7b00b573bcc4b517fdea283f7b551345997508c28c','unspecified_in_legacy_v1_archive'),'A_B4_lambda5_seed11_final':('ffc55075d0e8c14e77548752bd0de648858f75eeef98b8bffef88b160cb1c374','full_training_checkpoint')}
reports=[]
for name,(digest,kind) in expected.items():
 start=time.monotonic();saved=torch.load('restored/'+name+'.pt',map_location='cpu',mmap=True,weights_only=False)
 fingerprint=model_fingerprint(saved['model_state'])
 assert fingerprint=={'sha256_model_state':digest,'state_elements':405334016}
 assert saved['checkpoint_kind']=='model_only_lossless_xor_restored'
 assert saved['lossless_delivery_provenance']['model_fingerprint']==fingerprint
 assert saved['lossless_delivery_provenance']['source_checkpoint_kind']==kind
 assert saved['counters']['target_prediction_tokens']==10000000 and saved['counters']['replay_prediction_tokens']==500000
 assert len(saved['model_state'])==292 and all(v.dtype==torch.float32 for v in saved['model_state'].values())
 report=json.load(open('restored/'+name+'.pt.decode.json'))
 assert report['status']=='decoded_and_hash_verified' and report['model_fingerprint']==fingerprint and report['tensor_raw_hashes_verified']==292
 reports.append({'model':name,'fingerprint':fingerprint,'tensor_count':292,'source_checkpoint_kind':kind,'checkpoint_kind':saved['checkpoint_kind'],'actual_target_prediction_tokens':10000000,'actual_replay_prediction_tokens':500000,'source_checkpoint_sha256':saved['lossless_delivery_provenance']['source_checkpoint_sha256'],'new_restored_file_sha256_from_decoder':report['restored_file_sha256'],'raw_model_bytes_read_for_independent_fingerprint':405334016*4,'wall_seconds':time.monotonic()-start})
 del saved
json.dump({'status':'passed','scope':'Fresh reload and complete new model-state fingerprint; no original latest or optimizer CP','models':reports},open('independent_fingerprints.json','w'),indent=2)
