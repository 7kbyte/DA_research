"""One authorized CPU usability QA; only archived decoder and new restored files."""
from pathlib import Path
import datetime as dt,hashlib,json,os,resource,stat,subprocess,sys,time,traceback
QA=Path(__file__).resolve().parent
ROOT=Path('/data/202610/7kb/research').resolve()
DELIVERY=Path('/data/202610/7kb/research_delivery')
PRIMARY=DELIVERY/'primary_final_weights_lossless_v1.tar'
A=DELIVERY/'supplement_A_selected_final_weights_lossless_v1.tar'
CODE_SHA='3c2bcf9e38773ae4f69c27f85f342356dbefcb53d817be4321da6e9c4e550e1a'
CLI_SHA='28c5841cae8e7fd87d8cb62e7f756dad40ce15cb558710c31e48dfb45fde668a'
MAIN_ZIP='weights/formal_deterministic_B4_s11_b0p05_w0p05.xor.zip'
A_ZIP='weights/lambda_sweep_deterministic_B4_s11_b0p05_w5__validation_selected_lambda_final.xor.zip'
SIDECAR='metadata/lambda_sweep_deterministic_B4_s11_b0p05_w5__validation_selected_lambda_final.source_metadata.pt'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def sig(path):
 s=Path(path).lstat()
 return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,uid=s.st_uid,gid=s.st_gid,mode=oct(stat.S_IMODE(s.st_mode)),mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()

started=time.monotonic();start_self=resource.getrusage(resource.RUSAGE_SELF);start_children=resource.getrusage(resource.RUSAGE_CHILDREN)
receipt=dict(status='running',started_at_utc=utc(),qa_directory=str(QA),CPU_threads=2,CUDA_VISIBLE_DEVICES='',stages=[],
 parent_timer_scope='Runtime preflight, extraction, two decode calls, literal document metadata example, fresh restored-state fingerprints, and new-artifact inventory; excludes initial directory creation/source preparation',
 source_TARs_preserved=True,original_optimizer_checkpoints_read=False,new_encoding=False,GPU_used=False,whole_TAR_rehash=False,
 prior_read_only_receipt_preserved=dict(path='prior_read_only_receipt_snapshot.json',sha256=sha(QA/'prior_read_only_receipt_snapshot.json')),
 documentation_snapshot=dict(path='documentation_snapshot.md',sha256=sha(QA/'documentation_snapshot.md')))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
env.pop('PYTHONPATH',None)
(QA/'logs').mkdir()

def stage(name,argv,scope):
 before=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.monotonic();when=utc()
 stdout=QA/'logs'/(name+'.stdout.log');stderr=QA/'logs'/(name+'.stderr.log')
 with stdout.open('xb') as out,stderr.open('xb') as err:
  result=subprocess.run(argv,cwd=QA,env=env,stdout=out,stderr=err)
 after=resource.getrusage(resource.RUSAGE_CHILDREN)
 record=dict(stage=name,argv=[str(x) for x in argv],started_at_utc=when,completed_at_utc=utc(),wall_seconds=time.monotonic()-wall,
 child_user_seconds=after.ru_utime-before.ru_utime,child_sys_seconds=after.ru_stime-before.ru_stime,
 returncode=result.returncode,read_scope=scope,stdout=str(stdout.relative_to(QA)),stderr=str(stderr.relative_to(QA)),
 timer_relationship='Independent sequential child stage; child CPU nested under parent inclusive child total, wall nested under parent wall')
 receipt['stages'].append(record);print(json.dumps(dict(stage=name,rc=result.returncode,wall_seconds=record['wall_seconds']),ensure_ascii=False),flush=True)
 if result.returncode:raise RuntimeError('stage failed: '+name)

try:
 authorities=[]
 for tar,proofname in [(PRIMARY,'primary_weights_durable_publication.json'),(A,'A_selected_weights_durable_publication.json')]:
  authority=ROOT/'docs'/proofname;r=json.loads(authority.read_text());observed=sig(tar)
  assert all(observed[k]==r['destination_stat'][k] for k in observed)
  authorities.append(dict(tar=str(tar),publication_receipt=str(authority),publication_receipt_sha256=sha(authority),
   current_stat=observed,historical_full_TAR_sha256=r['destination_sha256'],fresh_whole_TAR_hash=False))
 receipt['source_authorities']=authorities
 common=['primary_final_weights_manifest.json','base/pythia-410m-final/model.safetensors','base/pythia-410m-final/config.json',
  'base/pythia-410m-final/tokenizer.json','base/pythia-410m-final/tokenizer_config.json','base/pythia-410m-final/special_tokens_map.json',
  'base/pythia-410m-final/README.md','tools/src/replay_study/weight_archive.py','tools/src/replay_study/__init__.py','tools/scripts/archive_model_weights.py']
 stage('01_extract_common',['tar','--extract','--file='+str(PRIMARY),'--keep-old-files',*common],
  'Only named primary members copied: pinned base/config/tokenizer/README, archived tools, small manifest; TAR headers skipped, entire TAR not hashed')
 assert sha(QA/'tools/src/replay_study/weight_archive.py')==CODE_SHA and sha(QA/'tools/scripts/archive_model_weights.py')==CLI_SHA
 receipt['archived_tool_sha256']=dict(decoder=CODE_SHA,CLI=CLI_SHA)
 stage('02_extract_main_zip',['tar','--extract','--file='+str(PRIMARY),'--keep-old-files',MAIN_ZIP],
  'Only one named compressed main B4 weight member copied; no original latest/optimizer checkpoint read')
 tool='tools/scripts/archive_model_weights.py';base='base/pythia-410m-final/model.safetensors'
 stage('03_decode_main',[sys.executable,tool,'--threads','2','decode','--archive',MAIN_ZIP,'--base',base,'--output','restored/main_B4_seed11_final.pt'],
  'Archived decoder reads extracted pinned base + selected ZIP; verifies base/FP32 base tensor/raw tensor/full model hashes and writes a new model-only checkpoint/report')
 stage('04_extract_A_members',['tar','--extract','--file='+str(A),'--keep-old-files','supplement_A_selected_final_manifest.json',SIDECAR,A_ZIP],
  'Only named A manifest, own full nonoptimizer metadata sidecar and selected B4 lambda5 ZIP copied; A uses primary fixed base')
 stage('05_decode_A',[sys.executable,tool,'--threads','2','decode','--archive',A_ZIP,'--base',base,'--output','restored/A_B4_lambda5_seed11_final.pt'],
  'Same archived primary decoder and fixed base; selected A ZIP only; no original latest/optimizer checkpoint read')
 stage('06_literal_document_metadata_example',[sys.executable,'metadata_example.py'],
  'Exact Python block from pre-QA user document: only new restored model-only metadata, small main manifest and A own metadata sidecar; mmap CPU')
 (QA/'independent_fingerprint_check.py').write_text('''from pathlib import Path
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
''')
 stage('07_independent_restored_model_fingerprints',[sys.executable,'independent_fingerprint_check.py'],
  'Fresh mmap reload of two NEW restored checkpoints and full sorted tensor-byte model_fingerprint over 2 x1,621,336,064B; no original checkpoints or GPU')
 results=json.loads((QA/'independent_fingerprints.json').read_text());assert results['status']=='passed'
 receipt['independent_model_checks']=results
 (QA/'inventory_new_artifacts.py').write_text('''from pathlib import Path
import hashlib,json
root=Path.cwd();records=[]
for path in sorted(root.rglob('*')):
 if not path.is_file() or path.name in {'artifact_inventory.json','actual_restore_receipt.json'}:continue
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 records.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
json.dump({'status':'passed','files':records,'scope':'All completed new QA files incl selected base/ZIPs/new restored models, scripts, manifests, sidecar and closed logs; excludes inventory/receipt self references; no source TAR or original CP hashed'},open('artifact_inventory.json','w'),indent=2)
''')
 stage('08_new_artifact_SHA_inventory',[sys.executable,'inventory_new_artifacts.py'],
  'Fresh SHA of only new selected/extracted/restored QA artifacts and completed logs; no original TAR or original optimizer checkpoint hashed')
 inventory=json.loads((QA/'artifact_inventory.json').read_text());files={f['path']:f for f in inventory['files']}
 for name in ['main_B4_seed11_final','A_B4_lambda5_seed11_final']:
  report=json.loads((QA/'restored'/(name+'.pt.decode.json')).read_text())
  assert files['restored/'+name+'.pt']['sha256']==report['restored_file_sha256']
 main=json.loads((QA/'primary_final_weights_manifest.json').read_text());main_row=next(r for r in main['logical_runs'] if r['condition_run_id']=='formal_deterministic_B4_s11_b0p05_w0p05');main_weight=next(r for r in main['weight_archives'] if r['actual_run_id']==main_row['weights_actual_run_id'])
 a=json.loads((QA/'supplement_A_selected_final_manifest.json').read_text());a_row=next(r for r in a['logical_runs'] if r['group']=='B4' and r['seed']==11 and r['weight']==5)
 assert files[MAIN_ZIP]['sha256']==main_weight['archive_sha256'] and files[A_ZIP]['sha256']==a_row['archive_sha256']
 assert files[base]['sha256']=='3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b'
 for authority in authorities:assert sig(authority['tar'])==authority['current_stat']
 receipt.update(status='passed',completed_at_utc=utc(),artifact_inventory=dict(path='artifact_inventory.json',sha256=sha(QA/'artifact_inventory.json'),files=len(files)),
  actual_doc_metadata_example_passed=True,all_new_output_SHA_verified=True,
  interpretation='Durable selected-member extraction + archived CPU decoder usability + new restored-model fingerprints; not another uint8 comparison with original latest, not full Adam resume, no GPU retraining QA')
except Exception as error:
 receipt.update(status='failed',completed_at_utc=utc(),error=repr(error),traceback=traceback.format_exc())
finally:
 after_self=resource.getrusage(resource.RUSAGE_SELF);after_children=resource.getrusage(resource.RUSAGE_CHILDREN)
 receipt.update(parent_wall_seconds=time.monotonic()-started,parent_self_user_seconds=after_self.ru_utime-start_self.ru_utime,
  parent_self_sys_seconds=after_self.ru_stime-start_self.ru_stime,children_user_seconds=after_children.ru_utime-start_children.ru_utime,
  children_sys_seconds=after_children.ru_stime-start_children.ru_stime,
  accounting='Parent wall includes all sequential child stages and parent overhead. Child user/sys are sums of disjoint stages; their CPU is nested under inclusive process-tree CPU, not added to a pre-existing inclusive total. Parent self CPU excludes children. Per-model fingerprint wall and decoder internal wall are nested in their child stage wall.')
 write(QA/'actual_restore_receipt.json',receipt)
 print(json.dumps(dict(status=receipt['status'],parent_wall_seconds=receipt['parent_wall_seconds'],children_cpu_seconds=receipt['children_user_seconds']+receipt['children_sys_seconds'],receipt=str(QA/'actual_restore_receipt.json'))),flush=True)
if receipt['status']!='passed':raise SystemExit(1)
