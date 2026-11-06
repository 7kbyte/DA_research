"""Targeted small-source probes, never call actual review()/export or payloads."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'artifacts/provenance/C_best_publication_runbook_v1/review_metadata.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)=='a31f4062f53e2cfb64e028f822a30dab6e6e500918322b320716e84783f885b5'
with (HERE/'retained_a31f_review_metadata.py').open('xb') as f:f.write(SOURCE.read_bytes())
spec=importlib.util.spec_from_file_location('independent_C_best',SOURCE);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
checks=[]
def reject(label,fn):
    try:fn()
    except ValueError:checks.append({'case':label,'rejected':True})
    else:raise AssertionError(label)
folder=ROOT/'artifacts/provenance/additional_endpoint_actual_v4/C410_best_batch'
launch=json.loads((folder/'child_launch.json').read_bytes());sp=Path(launch['spec_path'])
assert sha(sp)==launch['spec_sha256'];definition=json.loads(sp.read_bytes())
resource=ROOT/'artifacts/provenance/additional_endpoint_actual_v4/ROOT_C410_best_batch_resource_review.json'
assert sha(resource)=='74f6ac6089e0bfacf49a8f4fed064d09a0eed79660dd635a63d1a494ff84b8f1'
r.validate_batch_identity('C410_best',launch,definition,folder,resource,sha(resource))
checks.append({'case':'Actual C410 launch/spec/74f6 authority path+SHA pure schema validation','accepted':True})
foreign=copy.deepcopy(definition);foreign['authority']['sha256']='0'*64
reject('actual-shape foreign authority refused',lambda:r.validate_batch_identity('C410_best',launch,foreign,folder,resource,sha(resource)))
scope_path=Path('/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/scope.json')
scope=json.loads(scope_path.read_bytes());assert sha(scope_path)==r.SCOPE_SHA
for name,p in r.PROFILES.items():
    context=scope['phases'][name];assert len(r.native_members(context['base_files'],context['context']['snapshot'],p['model']))==6
checks.append({'case':'Actual frozen three-model native6 source/member/SHA schema','accepted':True})
with tempfile.TemporaryDirectory(prefix='C_best_metadata_probe_') as temp:
    t=Path(temp);(t/'metadata').mkdir();(t/'foreign.pt').write_bytes(b'toy opaque')
    symlink=t/'metadata/toy.source_metadata.pt';symlink.symlink_to(t/'foreign.pt')
    reject('sidecar symlink escape refused before read',lambda:r.Metadata(t).opaque(symlink,sidecar=True))
    cp=t/'toy.pt';cp.write_bytes(b'toy checkpoint marker')
    reject('tiny CP denied even when bounded',lambda:r.Metadata(t).opaque(cp))

# Isolate only the frozen pure resource validator. All mappings are hand-built
# in memory; no true ROOT authority file or real sample is created or invoked.
exporter=ROOT/'scripts/export_additional_endpoints.py'
assert sha(exporter)==r.EXPORTER_SHA
tree=ast.parse(exporter.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_resource_review')
authority={'status':'accepted_cpu_resource_samples','authorized_by':'root','authorized':False,'scope_sha256':'S','exporter_sha256':'E',
 'approved_packages':['C410_best'],'protected_finish_deadline_utc':'2026-11-07T02:56:32Z','package_plans':{'C410_best':{'actual_sample_record':{'path':'sample','sha256':'A'},'estimated_seconds_per_unique':120}}}
sample={'package':'C410_best','status':'verified_additional_endpoint_not_yet_assembled','exporter_sha256':'E','prepared_scope_sha256':'S',
 'additional_encoding_calls':1,'all_nonweight_metadata_roundtrip_verified':True,'bytewise_verification':{'all_tensor_bytes_equal':True},'wall_seconds':100}
g={'require':r.require,'file_sha256':lambda path:{'resource':'R','sample':'A','exporter':'E'}[str(path)],'read':lambda path:authority if path=='resource' else sample,'SLOTS':{'slot1':['C410_best']},'__file__':'exporter','proof':lambda path:{'path':path,'sha256':'R'}}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(exporter),'exec'),g)
accepted=g['validate_resource_review']('resource','R','S','slot1')
assert accepted['approved_packages']==['C410_best']
checks.append({'case':'Confirmed inherited validator accepts authorized=False; new checker must independently refuse contradiction','accepted_by_inherited_validator':True,'finding':'requires author correction before final acceptance'})
with (HERE/'probe_results.json').open('x') as f:json.dump({'status':'targeted_probes_complete_with_authority_bool_finding','checks':checks,
 'source_sha256':sha(SOURCE),'large_payload_or_scores_read':False,'actual_full_checker_called':False,'true_authority_created':False},f,indent=2);f.write('\n')
print(json.dumps({'checks':len(checks),'authority_bool_finding_reproduced':True,'payload_bytes_read':0}))
