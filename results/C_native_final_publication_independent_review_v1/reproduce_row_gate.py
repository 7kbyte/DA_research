"""Isolated original per-row AST gate on tiny fixtures; never full review()."""
import ast,copy,hashlib,importlib.util,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/provenance/C_native_final_publication_runbook_v2/initial_source_bytes/artifacts/provenance/C_native_final_publication_runbook_v1/review_metadata.py'
PIN='95e43930aca1b1f1e544b97863694047b8a1972864180e01ef3849d0a1158b73'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==PIN
sp=importlib.util.spec_from_file_location('original_native_reviewer',SOURCE);r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
fn=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='review')
loop=next(n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='item')
code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),str(SOURCE),'exec')
with tempfile.TemporaryDirectory() as td:
 p=Path(td);correct=p/'frozen.pt';wrong=p/'unrelated.pt';sidecar=p/'own_metadata.pt';archive=p/'own.zip';cfg=p/'cfg.json';summary=p/'summary.json'
 for f,raw in [(correct,b'CORRECT'.ljust(42,b'x')),(wrong,b'bad'),(sidecar,b'own'),(archive,b'zip'),(cfg,b'{"model":"fixed"}'),(summary,b'{"status":"completed"}')]:f.write_bytes(raw)
 h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();rid='synthetic_B4_s11';definition={'id':rid,'argv':['train','--config',str(cfg)]};digest=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest();cp_sha=h(correct)
 item=dict(run_id=rid,definition=definition,definition_sha256=digest,completed_scheduler_record=dict(status='completed',returncode=0,missing_outputs=[],argv=definition['argv'],definition_sha256=digest),source_checkpoint_sha256=cp_sha,model_revision='fixed',input_snapshot_sha256='snapshot',group='B4',seed=11,ratio=.05,weight=.05,actual_exposure=dict(target_prediction_tokens=25000000,replay_prediction_tokens=1250000,checkpoint_step=777),all_nonweight_metadata_roundtrip_verified=True,source_checkpoint=str(correct),source_stat=r.signature(correct),metadata_path=str(sidecar),metadata_stat=r.signature(sidecar),archive_path=str(archive),archive_stat=r.signature(archive),additional_encoding_calls=1,bytewise_verification=dict(all_tensor_bytes_equal=True,model_fingerprint={'sha256_model_state':'frozen_fp'}),config_sha256=h(cfg),summary_sha256=h(summary),model_fingerprint={'sha256_model_state':'frozen_fp'},encode={'model_fingerprint':{'sha256_model_state':'frozen_fp'}},decode={'model_fingerprint':{'sha256_model_state':'frozen_fp'}})
 identity=dict(checkpoint_sha256=cp_sha,step=777,checkpoint=str(correct),checkpoint_bytes=42,checkpoint_mtime_ns=correct.stat().st_mtime_ns,config_path=str(cfg),summary_path=str(summary),input_sha256={str(cfg):h(cfg),str(summary):h(summary)})
 frozen=dict(id=rid,group='B4',seed=11,ratio=.05,weight=.05,config_path=str(cfg),summary_path=str(summary));results=[]
 for name in ['baseline','wrong_config_SHA','wrong_summary_SHA','wrong_CP_path_size','inconsistent_nested_fingerprint']:
  row=copy.deepcopy(item)
  if name=='wrong_config_SHA':row['config_sha256']='b'*64
  if name=='wrong_summary_SHA':row['summary_sha256']='c'*64
  if name=='wrong_CP_path_size':row['source_checkpoint']=str(wrong);row['source_stat']=r.signature(wrong)
  if name=='inconsistent_nested_fingerprint':row['decode']['model_fingerprint']={'sha256_model_state':'wrong'}
  env=dict(r.__dict__,rows=[row],finals={rid:identity},frozen_rows={rid:frozen},definitions={rid:definition},profile={'target':25000000},protocol={'model_revision':'fixed'},phase={'snapshot_sha256':'snapshot'},meta=r.Metadata(),model='pythia-2.8b',byid={rid:row},aliases=[])
  exec(code,env);results.append({'case':name,'exact_original_per_row_loop_accepted':True})
 print(json.dumps({'source_SHA256':PIN,'scope':'Exact original row loop only; no real full review or large payload read','results':results},indent=2))
