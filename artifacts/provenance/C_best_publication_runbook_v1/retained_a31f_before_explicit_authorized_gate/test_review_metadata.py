"""Synthetic metadata only: never review a real pending C-best package."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('C_best_metadata_review', Path(__file__).with_name('review_metadata.py'))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def headers(name):
    p = r.PROFILES[name]
    c = dict(status='passed_complete_scratch_package_not_published', package=name, logical_endpoints=p['n'],
             unique_weights=p['n'], alias_count=0, file_fsync_completed=True,
             parent_directory_fsync_completed=True, tar_sha256='a'*64)
    m = dict(format='additional_formal_lossless_v1', status='complete_all_logical_endpoints_verified', package=name,
             kind='best', model_tag=p['model'], logical_endpoints=p['n'], logical_runs=[{'group':'B0'}]*p['n'],
             unique_weights=p['n'], aliases=0, optimizer_included=False, source_scope={'sha256':r.SCOPE_SHA})
    return c,m


@pytest.mark.parametrize('name',r.PROFILES)
@pytest.mark.parametrize('change',['final','foreign_package','foreign_model','foreign_scope','partial','optimizer','fsync','boolean_count'])
def test_only_complete_this_model_best_scope(name,change):
    c,m=headers(name);r.validate_headers(name,c,m)
    if change=='final': m['kind']='final'
    elif change=='foreign_package': c['package']='A_remaining_final'
    elif change=='foreign_model': m['model_tag']='another-model'
    elif change=='foreign_scope': m['source_scope']['sha256']='b'*64
    elif change=='partial': m['logical_runs'].pop()
    elif change=='optimizer': m['optimizer_included']=True
    elif change=='fsync': c['file_fsync_completed']=False
    else: c['unique_weights']=True
    with pytest.raises(ValueError): r.validate_headers(name,c,m)


@pytest.mark.parametrize('value',[False,True,None,'0',1,-1,0.0])
def test_returncode_strict_integer_zero(value):
    r.require_clean_rc({'returncode':0})
    with pytest.raises(ValueError): r.require_clean_rc({'returncode':value})


@pytest.mark.parametrize('key',['stop_requested','stop_requested_at'])
@pytest.mark.parametrize('value',[None,False,'now'])
def test_stop_key_presence_rejected_even_false_or_null(key,value):
    with pytest.raises(ValueError): r.require_clean_rc({'returncode':0,key:value})


def pair(equal=True):
    fp={'sha256_model_state':'a'*64,'state_elements':11}
    b4=dict(run_id='b4',checkpoint_sha256='b'*64,model_state_fingerprint=fp)
    d=dict(run_id='d',checkpoint_sha256='d'*64,model_state_fingerprint=fp if equal else {'sha256_model_state':'e'*64,'state_elements':11})
    sources=[{'path':'/fake/best_proof.json','sha256':'f'*64}]
    decision=dict(equal=equal,b4=b4,d=d,comparison={'endpoint':'best','opaque':'pair'})
    canonical=dict(run_id='b4',group='B4',source_checkpoint_sha256=b4['checkpoint_sha256'],model_fingerprint=fp,
                   additional_encoding_calls=1,archive_member='weights/b4.zip',archive_sha256='c'*64)
    row=dict(run_id='d',group='D',source_checkpoint_sha256=d['checkpoint_sha256'],model_fingerprint=d['model_state_fingerprint'],
             additional_encoding_calls=0 if equal else 1)
    if equal:
        row.update(alias_of='b4',archive_member=canonical['archive_member'],archive_sha256=canonical['archive_sha256'],
                   alias_evidence={'comparison':decision['comparison'],'sources':sources})
    return row,canonical,decision,sources


def test_false_equality_requires_new_D_encoding_and_cannot_alias():
    row,canonical,decision,sources=pair(False)
    r.validate_alias_decision(row,canonical,decision,sources)
    row['alias_of']='b4'
    with pytest.raises(ValueError): r.validate_alias_decision(row,canonical,decision,sources)


@pytest.mark.parametrize('change',['D_source','D_fingerprint','B4_source','B4_fingerprint','canonical_id','member','ZIP_SHA','source_proof','boolean_calls'])
def test_true_alias_binds_both_sources_fingerprints_and_member(change):
    row,canonical,decision,sources=pair()
    r.validate_alias_decision(row,canonical,decision,sources)
    if change=='D_source': row['source_checkpoint_sha256']='0'*64
    elif change=='D_fingerprint': row['model_fingerprint']={'sha256_model_state':'0'*64,'state_elements':11}
    elif change=='B4_source': canonical['source_checkpoint_sha256']='0'*64
    elif change=='B4_fingerprint': canonical['model_fingerprint']={'sha256_model_state':'0'*64,'state_elements':11}
    elif change=='canonical_id': row['alias_of']='other_seed_b4'
    elif change=='member': row['archive_member']='weights/other.zip'
    elif change=='ZIP_SHA': row['archive_sha256']='0'*64
    elif change=='source_proof': row['alias_evidence']={'comparison':decision['comparison'],'sources':[]}
    else: row['additional_encoding_calls']=False
    with pytest.raises(ValueError): r.validate_alias_decision(row,canonical,decision,sources)


def test_B4_itself_cannot_alias_even_with_true_pair():
    _,canonical,decision,sources=pair()
    r.validate_alias_decision(canonical,None,decision,sources)
    canonical['alias_of']='d'
    with pytest.raises(ValueError): r.validate_alias_decision(canonical,None,decision,sources)


def test_2p8B_never_invents_D_or_alias():
    c,m=headers('C2p8B_best');m['logical_runs'][0]={'group':'D'}
    with pytest.raises(ValueError): r.validate_headers('C2p8B_best',c,m)


@pytest.mark.parametrize('suffix',['.pt','.zip','.tar','.tar.gz','.safetensors'])
def test_payload_even_tiny_is_not_opened(tmp_path,suffix,monkeypatch):
    p=tmp_path/('payload'+suffix);p.write_bytes(b'opaque')
    def forbidden(_): raise AssertionError('payload opened')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    with pytest.raises(ValueError,match='payload'): r.Metadata(tmp_path).opaque(p)


def test_sidecar_only_own_metadata_directory_and_bounded(tmp_path):
    folder=tmp_path/'package';(folder/'metadata').mkdir(parents=True)
    p=folder/'metadata/run.source_metadata.pt';p.write_bytes(b'opaque small sidecar')
    assert r.Metadata(folder).opaque(p,sidecar=True)==b'opaque small sidecar'
    p2=tmp_path/'run.source_metadata.pt';p2.write_bytes(b'foreign')
    with pytest.raises(ValueError): r.Metadata(folder).opaque(p2,sidecar=True)
    p.write_bytes(b'');p.touch()
    with p.open('r+b') as f: f.truncate((8<<20)+1)
    with pytest.raises(ValueError,match='Bounded'): r.Metadata(folder).opaque(p,sidecar=True)


def sources(tmp_path):
    cfg=tmp_path/'configs/run.json';cfg.parent.mkdir();cfg.write_bytes(b'config')
    summary=tmp_path/'summary.json';summary.write_bytes(b'opaque-no-score-parse')
    cp=tmp_path/'best.pt';cp.write_bytes(b'never read original checkpoint')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    frozen=dict(id='run',config_path=str(cfg),summary_path=str(summary))
    identity=dict(run_id='run',actual_run_id='run',checkpoint=str(cp),config_path=str(cfg),summary_path=str(summary),
                  input_sha256={str(cfg):sha(cfg),str(summary):sha(summary)},checkpoint_bytes=cp.stat().st_size,checkpoint_mtime_ns=cp.stat().st_mtime_ns)
    row=dict(run_id='run',source_checkpoint=str(cp),source_stat=r.signature(cp),config_sha256=sha(cfg),summary_sha256=sha(summary))
    snapshot={'configs':{'configs/run.json':{'sha256':sha(cfg),'bytes':cfg.stat().st_size}}}
    return row,identity,frozen,snapshot


@pytest.mark.parametrize('change',['config_SHA','summary_SHA','snapshot_SHA','snapshot_size','READY_path','actual_run','CP_path','CP_size','CP_mtime'])
def test_snapshot_READY_record_current_bytes_crossbound_without_CP_read(tmp_path,change,monkeypatch):
    row,b,f,snapshot=sources(tmp_path)
    old=Path.read_bytes
    def guarded(p):
        assert p.suffix!='.pt'
        return old(p)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    r.bind_row_sources(r.Metadata(tmp_path),row,b,f,snapshot,tmp_path)
    if change=='config_SHA': row['config_sha256']='0'*64
    elif change=='summary_SHA': row['summary_sha256']='0'*64
    elif change=='snapshot_SHA': snapshot['configs']['configs/run.json']['sha256']='0'*64
    elif change=='snapshot_size': snapshot['configs']['configs/run.json']['bytes']+=1
    elif change=='READY_path': b['summary_path']=str(tmp_path/'foreign.json')
    elif change=='actual_run': b['actual_run_id']='foreign'
    elif change=='CP_path': row['source_checkpoint']=str(tmp_path/'foreign.pt')
    elif change=='CP_size': b['checkpoint_bytes']+=1
    else: b['checkpoint_mtime_ns']+=1
    with pytest.raises(ValueError): r.bind_row_sources(r.Metadata(tmp_path),row,b,f,snapshot,tmp_path)


def native(model):
    names=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md']
    files=[dict(path='/fake/'+model+'/'+n,sha256='a'*64,member='base/'+model+'-final/'+n) for n in names]
    snapshot={'model_files':{f['path']:{'sha256':f['sha256'],'bytes':11} for f in files}}
    return files,snapshot


@pytest.mark.parametrize('change',['wrong_model_member','missing','SHA','source','duplicate'])
def test_native_model_base_closure_not_foreign_or_external(change):
    files,snapshot=native('pythia-1b');assert len(r.native_members(files,snapshot,'pythia-1b'))==6
    if change=='wrong_model_member': files[0]['member']='base/pythia-410m-final/model.safetensors'
    elif change=='missing': files.pop()
    elif change=='SHA': files[0]['sha256']='0'*64
    elif change=='source': files[0]['path']='/foreign/model.safetensors'
    else: files[-1]=copy.deepcopy(files[0])
    with pytest.raises(ValueError): r.native_members(files,snapshot,'pythia-1b')


@pytest.mark.parametrize('name',r.PROFILES)
@pytest.mark.parametrize('change',['inode','owner','mode','allocated','path','links'])
def test_original_best_reserve_is_not_original_final_or_foreign(name,change):
    p=r.PROFILES[name]
    current=dict(device=106,inode=p['inode'],uid=11471,gid=11471,mode=0o600,bytes=p['capacity'],allocated_bytes=p['capacity'],links=1)
    row=dict(path='/data/202610/7kb/research_delivery/.extra_'+p['reserve']+'_weights_reserve_20261106',actual_stat=copy.deepcopy(current))
    r.validate_reserve(name,row,current)
    if change=='inode': current['inode']+=1
    elif change=='owner': current['uid']=0
    elif change=='mode': current['mode']=0o644
    elif change=='allocated': current['allocated_bytes']-=512
    elif change=='path': row['path']='/foreign/reserve'
    else: current['links']=2
    with pytest.raises(ValueError): r.validate_reserve(name,row,current)


def test_actual_assembler_retains_repeated_gate_source_members():
    files,_=native('pythia-410m')
    m=dict(logical_runs=[dict(metadata_member='metadata/b4.pt',archive_member='weights/b4.zip'),
                         dict(metadata_member='metadata/d.pt',archive_member='weights/b4.zip',alias_of='b4')],
           base_dependency={'files':files},fullphase_source_gate={'sources':[{'path':'/fake/READY.json'},{'path':'/fake/READY.json'}]})
    names=r.member_inventory(m)
    assert len(names)==2+1+6+len(r.EXTRAS)+2
    assert 'provenance/gate/0_READY.json' in names and 'provenance/gate/1_READY.json' in names
    m['logical_runs'][0]['archive_member']='../bad.zip'
    with pytest.raises(ValueError): r.member_inventory(m)


def test_pending_no_completion_never_emits_acceptance(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'BASE',tmp_path)
    out=tmp_path/'acceptance.json'
    with pytest.raises(ValueError,match='Pending'): r.review('C410_best',tmp_path/'authority.json','0'*64,out)
    assert not out.exists()


@pytest.mark.parametrize('name',r.PROFILES)
@pytest.mark.parametrize('change',['package','scope','resource','argv','env','authority','logs','duplicate_flag'])
def test_whole_batch_cannot_be_foreign_timer_or_argv(tmp_path,monkeypatch,name,change):
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'BASE',tmp_path/'delivery')
    folder=tmp_path/(name+'_batch');resource=tmp_path/'authority.json';sha='a'*64
    slot='slot2' if name=='C2p8B_best' else 'slot1'
    argv=['python',str(tmp_path/'scripts/export_additional_endpoints.py'),'execute','--scope',str(r.BASE/'scope.json'),
          '--scope-sha256',r.SCOPE_SHA,'--slot',slot,'--package',name,'--max-new-weights','20',
          '--resource-review',str(resource),'--resource-review-sha256',sha]
    launch={'argv':argv}
    spec=dict(argv=copy.deepcopy(argv),package=name,slot=slot,cwd=str(tmp_path),logs=str(folder),
              authority={'path':str(resource),'sha256':sha},env={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    r.validate_batch_identity(name,launch,spec,folder,resource,sha)
    if change=='package': spec['package']='A_remaining_final'
    elif change=='scope': spec['argv'][spec['argv'].index('--scope-sha256')+1]='0'*64;launch['argv']=spec['argv'][:]
    elif change=='resource': sha='b'*64
    elif change=='argv': launch['argv']+=['--unexpected','true']
    elif change=='env': spec['env']['CUDA_VISIBLE_DEVICES']='0'
    elif change=='authority': spec['authority']['sha256']='b'*64
    elif change=='logs': spec['logs']=str(tmp_path/'foreign_batch')
    else: spec['argv']+=['--package',name];launch['argv']=spec['argv'][:]
    with pytest.raises(ValueError): r.validate_batch_identity(name,launch,spec,folder,resource,sha)
