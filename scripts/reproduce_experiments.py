#!/usr/bin/env python3
"""Relocate frozen experiments with input verification and explicit new run identities."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,re,subprocess,sys,uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD_WORKSPACES=['/data/202610/7kb/research','/tmp/grad-research-20261106/worktree']
OLD_SCRATCH='/tmp/grad-research-20261106'
CORE_FILES={f'src/replay_study/{name}.py' for name in ['train','objectives','evaluate','execution','data','generate','selection']}|{
    'scripts/run_train.py','scripts/run_evaluate.py','scripts/make_experiments.py'}

def atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(temp,path)

def sha(path):
    path=Path(path);before=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError(f'Input changed while hashing: {path}')
    return h.hexdigest()

def workspace_key(path):
    text=str(path)
    for base in [*OLD_WORKSPACES,str(ROOT)]:
        if text==base:return ''
        if text.startswith(base+'/'):return text[len(base)+1:]
    if not Path(text).is_absolute():return text
    raise ValueError(f'Unrecognized saved workspace path: {path}')

def relocate_value(value,workspace,scratch):
    if isinstance(value,dict):return {k:relocate_value(v,workspace,scratch) for k,v in value.items()}
    if isinstance(value,list):return [relocate_value(v,workspace,scratch) for v in value]
    if isinstance(value,str):
        for base in sorted(set([*OLD_WORKSPACES,str(ROOT)]),key=len,reverse=True):
            if value==base or value.startswith(base+'/'):return str(workspace)+value[len(base):]
        if value==OLD_SCRATCH or value.startswith(OLD_SCRATCH+'/'):return str(scratch)+value[len(OLD_SCRATCH):]
    return value

def hash_catalog(paths):
    files,code,configs={},{},{}
    for path in paths:
        obj=json.loads(Path(path).read_text())
        for key,record in obj.get('data',{}).items():files[key]=record['sha256']
        files.update(obj.get('data_sha256',{}))
        for path,record in obj.get('bank_sha256',{}).items():
            files[workspace_key(path)]=record['tensor'];files[str(Path(workspace_key(path)).with_suffix('.json'))]=record['manifest']
        for key,record in obj.get('code',{}).items():code[key]=record['sha256']
        code.update(obj.get('source_code_sha256',{}))
        for key,record in obj.get('configs',{}).items():configs[key]=record['sha256']
    return files,code,configs

def model_catalog():
    models={}
    for path in [ROOT/'artifacts/assets/model_provenance.json',ROOT/'artifacts/assets/extended_model_provenance.json']:
        if not path.exists():continue
        obj=json.loads(path.read_text())
        for record in obj.get('models',[obj]):
            name=Path(record['path']).name
            models[name]={'revision':record['revision'],'files':record.get('source_file_hashes') or
                {'model.safetensors':record['sha256']},'repo':record.get('repo')}
    downloads=ROOT/'artifacts/assets/extended_model_download_manifest.json'
    if downloads.exists():
        for asset in json.loads(downloads.read_text()).get('assets',[]):
            name=Path(asset['destination']).parent.name
            if name not in models:continue
            if asset['revision']!=models[name]['revision']:raise ValueError('Conflicting pinned model revisions')
            filename=asset['file'];expected=asset['sha256']
            if filename in models[name]['files'] and models[name]['files'][filename]!=expected:
                raise ValueError('Conflicting frozen model file hashes')
            models[name]['files'][filename]=expected
    return models

def verify_file(path,expected,checks):
    path=Path(path)
    if str(path) in checks:
        if checks[str(path)]!=expected:raise ValueError(f'Conflicting input hashes: {path}')
        return
    if not path.is_file():raise FileNotFoundError(f'Required input missing: {path}; expected SHA256 {expected}')
    actual=sha(path)
    if actual!=expected:raise ValueError(f'Input hash mismatch: {path}; expected {expected}, got {actual}')
    checks[str(path)]=expected

def scientific_parameters(config):
    ignored={'model_name_or_path','target_path','replay_path','output_dir','validation_paths',
             'resume_from','run_label','deadline_utc','deadline_margin_seconds','device'}
    return {k:v for k,v in config.items() if k not in ignored}

def relocate(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.namespace):raise ValueError('Namespace must be a plain new run-id prefix')
    workspace=args.new_workspace.resolve();scratch=args.new_scratch.resolve()
    source=json.loads(args.registry.read_text())
    selected=[r for r in source if not args.run_id or r['id'] in args.run_id]
    if len({r['id'] for r in selected})!=len(selected):raise ValueError('Duplicate source run IDs')
    if args.run_id and set(args.run_id)!={r['id'] for r in selected}:raise ValueError('Unknown requested run ID')
    aliases=[r for r in selected if r.get('execution_kind')=='alias']
    selected=[r for r in selected if r.get('execution_kind')!='alias']
    if not selected:raise ValueError('No independent training runs selected')
    catalog,code,config_hashes=hash_catalog(args.input_manifest)
    if CORE_FILES-set(code):raise KeyError(f'Missing frozen core hashes: {sorted(CORE_FILES-set(code))}')
    models=model_catalog();checks={};configs=[];records=[];queue=[]
    provenance_path=workspace/f'configs/reproductions/{args.namespace}_provenance.json'
    registry_path=workspace/f'configs/reproductions/{args.namespace}_registry.json'
    queue_path=workspace/f'configs/reproductions/{args.namespace}_queue.json'
    for path in [provenance_path,registry_path,queue_path]:
        if path.exists():raise FileExistsError(f'Choose a fresh namespace; refusing overwrite: {path}')
    for key,expected in code.items():
        # Freeze scientific core; auxiliary analysis/packaging scripts may differ.
        if key in CORE_FILES:verify_file(workspace/key,expected,checks)
    for record in selected:
        source_config=ROOT/workspace_key(record['config_path'])
        expected_config=config_hashes.get(workspace_key(record['config_path']),record.get('config_sha256'))
        if not expected_config:raise KeyError('Original config SHA absent; provide its frozen input snapshot')
        if sha(source_config)!=expected_config:raise ValueError(f'Original frozen config hash mismatch: {source_config}')
        original=json.loads(source_config.read_text());config=relocate_value(copy.deepcopy(original),workspace,scratch)
        if original.get('run_label')!=record['id']:raise ValueError('Source registry ID differs from frozen configuration identity')
        run_id=f"{args.namespace}__{record['id']}"
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id):raise ValueError('Original run ID is not scheduler-safe')
        output=scratch/'runs'/run_id;config_path=workspace/f'configs/reproductions/{args.namespace}/{run_id}.json'
        if output.exists() or config_path.exists():raise FileExistsError(f'Refusing existing reproduction output: {output}')
        config.update(output_dir=str(output),run_label=run_id,resume_from=None,deadline_utc=args.deadline)
        if scientific_parameters(config)!=scientific_parameters(original):raise AssertionError('Relocation changed scientific parameters')
        for source_path,new_path in [(original['target_path'],config['target_path']),
                *([(original['replay_path'],config['replay_path'])] if original.get('replay_path') else []),
                *[(old,config['validation_paths'][name]) for name,old in original.get('validation_paths',{}).items()]]:
            key=workspace_key(source_path)
            if key not in catalog:raise KeyError(f'Input SHA missing from supplied snapshots/protocols: {key}')
            verify_file(new_path,catalog[key],checks)
        test_paths=relocate_value(record.get('test_paths',{}),workspace,scratch)
        for name,old in record.get('test_paths',{}).items():
            key=workspace_key(old)
            if key not in catalog:raise KeyError(f'Test input SHA missing: {key}')
            verify_file(test_paths[name],catalog[key],checks)
        model=Path(original['model_name_or_path']).name
        if model not in models:raise KeyError(f'Model provenance not bundled for {model}')
        expected_model=models[model]
        if original.get('model_revision')!=expected_model['revision']:raise ValueError('Model revision differs from frozen provenance')
        for name,expected in expected_model['files'].items():verify_file(Path(config['model_name_or_path'])/name,expected,checks)
        summary=workspace/f'results/reproductions/{args.namespace}/{run_id}_summary.json'
        configs.append((config_path,config))
        records.append({**record,'id':run_id,'config_path':str(config_path),'output_dir':str(output),
            'summary_path':str(summary),'test_paths':test_paths,'active':True,
            'reproduction_of':{'run_id':record['id'],'config_path':record['config_path'],
                'config_sha256':sha(source_config),'model_revision':original.get('model_revision'),
                'scientific_parameters_unchanged':True,'path_changes_are_new_scientific_settings':False},
            'reproduction_manifest':str(provenance_path)})
        if 'checkpoint_path' in record:records[-1]['checkpoint_path']=str(output/'checkpoints/latest.pt')
        queue.append({'id':run_id,'argv':[sys.executable,str(workspace/'scripts/make_experiments.py'),'run',
            '--config',str(config_path),'--summary-copy',str(summary)],
            'env':{'CUBLAS_WORKSPACE_CONFIG':':4096:8','HF_HOME':str(scratch/'cache/huggingface')},'expected_seconds':600,
            'outputs':[str(output/'summary.json'),str(summary)]})
    # Finish every check before writing any config or creating any run directory.
    for path,config in configs:atomic_json(path,config)
    for record in records:record['config_sha256']=sha(record['config_path'])
    atomic_json(registry_path,records);atomic_json(queue_path,queue)
    atomic_json(provenance_path,{'source_registry':str(args.registry),'source_registry_sha256':sha(args.registry),
        'input_manifests':{str(p):sha(p) for p in args.input_manifest},'verified_inputs':checks,
        'source_workspace':str(ROOT),'new_workspace':str(workspace),'new_scratch':str(scratch),
        'namespace':args.namespace,'operational_changes':['paths','run_label','resume_from=None','deadline_utc'],
        'deadline_utc':args.deadline,'omitted_duplicate_aliases':[{'id':r['id'],'alias_of':r.get('alias_of')} for r in aliases],
        'registry':str(registry_path),'registry_sha256':sha(registry_path),
        'queue':str(queue_path),'queue_sha256':sha(queue_path),'training_started':False})
    print(json.dumps({'status':'verified_prepared','runs':len(records),'registry':str(registry_path),
        'queue':str(queue_path),'provenance':str(provenance_path),'training_started':False}))

def reproduce(args):
    records=json.loads(args.registry.read_text())
    if not records or not all('reproduction_of' in record for record in records):
        raise ValueError('Use a relocated registry created by this helper, not an original result registry')
    manifests={record['reproduction_manifest'] for record in records}
    if len(manifests)!=1:raise ValueError('Reproduction registry must use one verified relocation manifest')
    manifest=json.loads(Path(next(iter(manifests))).read_text())
    if sha(args.registry)!=manifest['registry_sha256']:raise ValueError('Relocated registry changed after verification')
    checks={}
    for path,expected in manifest['verified_inputs'].items():verify_file(path,expected,checks)
    for record in records:
        if sha(record['config_path'])!=record['config_sha256']:raise ValueError('Relocated config changed')
        if Path(record['output_dir']).exists():raise FileExistsError('Choose another fresh namespace; reproduction output already exists')
    queue=Path(manifest['queue'])
    if sha(queue)!=manifest['queue_sha256']:raise ValueError('Relocated queue changed')
    workspace=Path(manifest['new_workspace'])
    state=workspace/f"logs/reproductions/{manifest['namespace']}_job_state.json"
    command=[sys.executable,str(workspace/'scripts/schedule_jobs.py'),'--queue',str(queue),'--state',str(state),
        '--gpus',args.gpus,'--deadline',args.deadline,'--exit-when-idle']
    if not args.execute:
        print(json.dumps({'status':'verified_not_started','argv':command,'note':'--execute explicitly starts new training'}));return
    log=workspace/f"logs/reproductions/{manifest['namespace']}_scheduler.log";log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab',buffering=0) as stream:
        proc=subprocess.Popen(command,cwd=workspace,stdin=subprocess.DEVNULL,stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'status':'started_detached','scheduler_pid':proc.pid,'state':str(state),'log':str(log)}))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('relocate',help='Verify frozen inputs and write a fresh namespace; no training')
    prepare.add_argument('--registry',type=Path,required=True)
    prepare.add_argument('--input-manifest',type=Path,action='append',required=True,
        help='Frozen input snapshot, plus model-specific extended protocol when needed (repeatable)')
    prepare.add_argument('--new-workspace',type=Path,required=True);prepare.add_argument('--new-scratch',type=Path,required=True)
    prepare.add_argument('--namespace',required=True);prepare.add_argument('--run-id',action='append')
    prepare.add_argument('--deadline',help='Optional new operational deadline; original dated deadline is otherwise cleared')
    prepare.set_defaults(function=relocate)
    launch=sub.add_parser('reproduce',help='Reverify relocated files and print or explicitly start detached scheduler')
    launch.add_argument('--registry',type=Path,required=True);launch.add_argument('--gpus',default='0')
    launch.add_argument('--deadline',required=True);launch.add_argument('--execute',action='store_true')
    launch.set_defaults(function=reproduce)
    args=parser.parse_args();args.function(args)

if __name__=='__main__':main()
