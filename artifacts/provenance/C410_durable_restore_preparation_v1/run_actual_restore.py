"""Prepared fixed two-durable-TAR restore QA. Run only after ROOT authorization."""
import argparse,datetime as dt,hashlib,json,os,resource,shutil,stat,subprocess,sys,tarfile,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
EXPECTED_QA=Path('/tmp/grad-research-20261106/reproduction_qa/C410_external_base_B4s11_final_v1')
PLAN_SHA='22254780c181fcaca9ab27a1e39e2feeda4f9152362312413262bc22f25e97b7'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def sig(path):
 s=Path(path).lstat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,uid=s.st_uid,gid=s.st_gid,mode=oct(stat.S_IMODE(s.st_mode)),mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def main(qa):
 if qa!=EXPECTED_QA or qa.exists() or qa.is_symlink():raise ValueError('Fixed NEW QA directory must not exist')
 start=time.monotonic();su=resource.getrusage(resource.RUSAGE_SELF);cu=resource.getrusage(resource.RUSAGE_CHILDREN)
 assert sha(HERE/'preparation_receipt.json')==PLAN_SHA
 plan=json.loads((HERE/'preparation_receipt.json').read_text());expected=plan['expected'];sources=plan['source_publication_receipts'];tars={}
 for tag,r in sources.items():
  assert sha(r['path'])==r['sha256'];current=json.loads(Path(r['path']).read_text());p=Path(current['destination']);assert sig(p)==r['current_stat'];tars[tag]=p
 for key,header in plan['selected_safe_headers'].items():
  tag,name=key.split(':',1)
  with tarfile.open(tars[tag],'r:') as t:
   m=t.getmember(name);assert m.isfile() and not Path(name).is_absolute() and '..' not in Path(name).parts and m.size==header['bytes']
 qa.mkdir(parents=True,exist_ok=False);(qa/'logs').mkdir()
 for name in ['preparation_receipt.json','check_restored.py']:shutil.copyfile(HERE/name,qa/name)
 shutil.copyfile(ROOT/'artifacts/provenance/durable_weight_restore_examples_v1/inventory_new_artifacts.py',qa/'inventory_new_artifacts.py')
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');env.pop('PYTHONPATH',None)
 receipt=dict(status='running',started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),qa_directory=str(qa),preparation_receipt_SHA256=PLAN_SHA,stages=[],source_publication_authorities=sources,source_TARs_preserved=True,whole_TAR_rehash=False,original_checkpoint_or_live_native_model_read=False,new_encoding=False,GPU_used=False,CPU_threads=2,parent_timer_scope='Preflight, selected extraction, fresh extracted-member SHA, one archived decode, new full model fingerprint and metadata check, new artifact inventory; no original CP and no full source TAR SHA')
 def stage(name,argv,scope):
  before=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.monotonic()
  with (qa/'logs'/(name+'.stdout.log')).open('xb') as out,(qa/'logs'/(name+'.stderr.log')).open('xb') as err:r=subprocess.run(argv,cwd=qa,env=env,stdout=out,stderr=err)
  after=resource.getrusage(resource.RUSAGE_CHILDREN)
  receipt['stages'].append(dict(stage=name,argv=[str(x) for x in argv],returncode=r.returncode,wall_seconds=time.monotonic()-wall,child_user_seconds=after.ru_utime-before.ru_utime,child_sys_seconds=after.ru_stime-before.ru_stime,read_scope=scope,timer_relationship='Independent sequential child stage; nested in whole parent wall and child CPU total'))
  if r.returncode:raise RuntimeError('Failed stage:'+name)
 try:
  stage('01_extract_primary_base',['tar','--extract','--file='+str(tars['primary']),'--keep-old-files','primary_final_weights_manifest.json',*plan['native_base_members']], 'Only named durable primary native base/config/tokenizer/README and small manifest')
  selected=['extended_C_final_manifest.json',expected['archive_member'],expected['metadata_member'],*plan['archived_tools']]
  stage('02_extract_C410_members',['tar','--extract','--file='+str(tars['C410']),'--keep-old-files',*selected], 'Only named durable C410 manifest, B4s11 ZIP/own metadata and archived tools')
  wall=time.monotonic();pc=time.process_time();checks=[]
  manifest_pins={key.split(':',1)[1]:value['sha256'] for key,value in plan['small_fresh_member_reads'].items() if key.endswith('primary_final_weights_manifest.json') or key.endswith('extended_C_final_manifest.json')}
  pins={**plan['native_base_member_expected_SHA256'],**manifest_pins,expected['archive_member']:expected['archive_sha256'],expected['metadata_member']:expected['metadata_sha256'],**{k:v['sha256'] for k,v in plan['archived_tools'].items()}}
  for path,digest in pins.items():assert sha(qa/path)==digest;checks.append(dict(member=path,sha256=digest,bytes=(qa/path).stat().st_size))
  receipt['extracted_member_SHA_checks']=dict(status='passed',files=checks,wall_seconds=time.monotonic()-wall,parent_self_process_CPU_seconds=time.process_time()-pc,timer_relationship='Parent self-only operation nested in parent self CPU/wall; not a child stage')
  stage('03_decode_C410',[sys.executable,'tools/scripts/archive_model_weights.py','--threads','2','decode','--archive',expected['archive_member'],'--base','base/pythia-410m-final/model.safetensors','--output','restored/C410_B4_seed11_final.pt'], 'Archived C410 decoder, extracted selected ZIP and primary base only; all tensor/hash checks and new model-only output')
  stage('04_independent_fingerprint_and_metadata',[sys.executable,'check_restored.py'], 'Fresh full1621336064B new restored tensor fingerprint, decoder report and own extracted sidecar; no original CP')
  check=json.loads((qa/'independent_fingerprint.json').read_text());assert check['status']=='passed';receipt['independent_model_check']=check
  stage('05_inventory_new_artifacts',[sys.executable,'inventory_new_artifacts.py'], 'Fresh SHA only new extracted/restored QA files and closed logs; no source TAR, original CP or live native models')
  inventory=json.loads((qa/'artifact_inventory.json').read_text());files={x['path']:x for x in inventory['files']};report=json.loads((qa/'restored/C410_B4_seed11_final.pt.decode.json').read_text());assert files['restored/C410_B4_seed11_final.pt']['sha256']==report['restored_file_sha256']
  for tag,r in sources.items():assert sig(tars[tag])==r['current_stat']
  receipt.update(status='passed',artifact_inventory_SHA256=sha(qa/'artifact_inventory.json'),scope='One fixed durable C410 B4s11 external-base CPU recovery; not all24 endpoints, not new original-CP uint8 comparison, not Adam resume or GPU retraining')
 except Exception as err:receipt.update(status='failed',error=repr(err),traceback=traceback.format_exc())
 finally:
  sa=resource.getrusage(resource.RUSAGE_SELF);ca=resource.getrusage(resource.RUSAGE_CHILDREN)
  receipt.update(finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),parent_wall_seconds=time.monotonic()-start,parent_self_user_seconds=sa.ru_utime-su.ru_utime,parent_self_sys_seconds=sa.ru_stime-su.ru_stime,children_user_seconds=ca.ru_utime-cu.ru_utime,children_sys_seconds=ca.ru_stime-cu.ru_stime,children_peak_RSS_bytes_cumulative=ca.ru_maxrss*1024,accounting='Parent self CPU excludes children; inclusive CPU equals parent self user/sys plus child user/sys once. Stage/decode/check timers are nested and not added again.')
  (qa/'actual_restore_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
  print(json.dumps(dict(status=receipt['status'],receipt=str(qa/'actual_restore_receipt.json'),parent_wall_seconds=receipt['parent_wall_seconds'])))
 if receipt['status']!='passed':raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qa-dir',type=Path,required=True);a=p.parse_args();main(a.qa_dir)
