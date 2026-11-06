"""Independent, score-blind whole-phase gate for supplemental analysis.

This module never initializes a model or parses evaluation outputs. It binds a
real READY receipt, the full frozen manifest, scheduler definitions, imported
canonical inferences/baselines, logical aliases and their byte attestations.
"""
from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time


def read(path):return json.loads(Path(path).read_text())
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def proof(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def attestation_path(path):p=Path(path);return p.with_name(p.stem+'_execution.json')


class Verifier:
    """Explicit checkpoint-only historical SHA reuse; all other files fresh.

    A snapshot never grants checkpoint eligibility: only SHA-bound manifest
    endpoints and complete-state proof sides grant that eligibility.
    """
    def __init__(self,checkpoint_verification=None,checkpoint_paths=None):
        self.sources={};self.signatures={};self.requested={};self.checkpoint_paths={};self.cache=None;self.events=[]
        self.work={'checkpoint_cache_hits':0,'checkpoint_cached_bytes':0,'checkpoint_fresh_reads':0,'checkpoint_fresh_bytes':0,
            'other_fresh_reads':0,'other_fresh_bytes':0,'within_invocation_reuses':0}
        self.configure(checkpoint_verification or {'mode':'fresh','cache_snapshot':None})
        self.set_checkpoints(checkpoint_paths or {})
    def configure(self,policy):
        if policy['mode'] not in ('fresh','v5_checkpoint_snapshot') or (policy['mode']=='fresh' and policy.get('cache_snapshot') is not None):raise ValueError('Explicit fresh or SHA-bound checkpoint cache mode required')
        self.policy=policy
        if policy['mode']=='v5_checkpoint_snapshot':
            snapshot=policy['cache_snapshot'];source=policy['cache_reader_source']
            if Path(source['path']).resolve()!=Path(__file__).resolve().parents[2]/'scripts/stat_bound_digest_cache.py':raise ValueError('Only the explicitly reviewed v5 cache reader is accepted')
            self.bind(source['path'],source['sha256']);self.bind(snapshot['path'],snapshot['sha256'])
            spec=importlib.util.spec_from_file_location('supplemental_v2_stat_reader',source['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.cache=module.StatBoundHashes(snapshot=snapshot['path'],snapshot_sha256=snapshot['sha256'])
    def set_checkpoints(self,paths):
        for path,expected in paths.items():
            key=str(Path(path).resolve())
            if Path(key).suffix!='.pt' or not re.fullmatch(r'[0-9a-f]{64}',expected):raise ValueError('Explicit SHA-bound checkpoint .pt path required')
            if key in self.checkpoint_paths and self.checkpoint_paths[key]!=expected:raise ValueError('Contradictory checkpoint SHA authorities')
            self.checkpoint_paths[key]=expected
    @staticmethod
    def signature(path):
        s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    def bind(self,path,expected=None):
        p=str(Path(path).resolve());before=self.signature(p)
        self.requested[str(path)]=p
        if p in self.sources:
            if before!=self.signatures[p]:raise ValueError('Source changed during verification: '+p)
            value=self.sources[p]
            self.work['within_invocation_reuses']+=1
        else:
            checkpoint=p in self.checkpoint_paths
            if checkpoint and expected!=self.checkpoint_paths[p]:raise ValueError('Checkpoint verification must use its explicit frozen expected SHA')
            historical=None
            if checkpoint and self.cache is not None:
                entry=self.cache.entries.get(p)
                if entry:self.bind(entry['path'],entry['sha256'])
                previous=self.cache._receipt(p)
                authority=previous.get('source_authority',{}) if previous is not None else {}
                if previous is not None and authority.get('kind')=='completed_training_checkpoint' and authority.get('returncode')==0:
                    if previous['sha256']!=expected:raise ValueError('Historical checkpoint SHA differs from current frozen authority')
                    historical=self.cache.entries[p]
                    old_hits=self.cache.counters['cache_hits'];value=self.cache.digest(p)
                    hit=self.cache.counters['cache_hits']>old_hits
                    self.work['checkpoint_cache_hits' if hit else 'checkpoint_fresh_reads']+=1
                    self.work['checkpoint_cached_bytes' if hit else 'checkpoint_fresh_bytes']+=before[2]
                    self.events.append({'path':p,'sha256':value,'mode':'historical_full_read_receipt_hit' if hit else 'fresh_after_stat_invalidation',
                        'current_bytes_read':0 if hit else before[2],'historical_full_read_receipt':historical,'five_stat':list(before)})
                else:historical=None
            if historical is None:
                value=sha(p)
                self.work['checkpoint_fresh_reads' if checkpoint else 'other_fresh_reads']+=1
                self.work['checkpoint_fresh_bytes' if checkpoint else 'other_fresh_bytes']+=before[2]
                if checkpoint:self.events.append({'path':p,'sha256':value,'mode':'fresh_full_read','current_bytes_read':before[2],'historical_full_read_receipt':None,'five_stat':list(before)})
            if self.signature(p)!=before:raise ValueError('Source changed while hashing: '+p)
            self.sources[p]=value;self.signatures[p]=before
        if expected is not None and value!=expected:raise ValueError('Frozen SHA mismatch: '+p)
        return value
    def proof(self,p):
        if not re.fullmatch(r'[0-9a-f]{64}',p.get('sha256','')):raise ValueError('Explicit SHA256 proof required')
        self.bind(p['path'],p['sha256']);return read(p['path'])
    def finish(self):
        for p,s in self.signatures.items():
            if self.signature(p)!=s:raise ValueError('Source changed before gate completion: '+p)
        if any(str(Path(path).resolve())!=canonical for path,canonical in self.requested.items()):raise ValueError('Source symlink changed before gate completion')
    def accounting(self):
        return {'checkpoint_verification':self.policy,'counters':self.work,'checkpoint_events':self.events,
            'cache_hits_are_current_byte_reads':False,'non_checkpoint_files_always_fresh':True,
            'counter_scope':'Unique paths verified by this explicit guard Verifier; checkpoint counters cover all phase checkpoint checks. Existing collector small-source/output reads are separate.',
            'cache_reader_counters':dict(self.cache.counters) if self.cache else None}


def immutable(path,value):
    """First writer wins, fsync file/directory; never replace a receipt."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('xb') as f:f.write(encoded(value));f.flush();os.fsync(f.fileno())
    try:os.link(temporary,path)
    except FileExistsError:raise FileExistsError('Immutable receipt already exists: '+str(path))
    finally:temporary.unlink()
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return proof(path)


def contract_manifest(manifest,contract):
    phases=manifest.get('phases',[])
    if manifest.get('schema_version')!=2 or manifest.get('mode')!=contract['mode'] or len(phases)!=1:
        raise ValueError('Exactly one frozen supplemental phase manifest required')
    phase=phases[0]
    for key in ('name','role','model_tag','data_family','expected_phase','rows'):
        if phase.get(key)!=contract['phase'][key]:raise ValueError('Phase contract differs: '+key)
    records=manifest['records'];jobs=manifest['jobs'];kinds=contract['checkpoints']
    if len(records)!=contract['logical_endpoints'] or {r['kind'] for r in records}!=set(kinds):raise ValueError('Complete logical endpoint inventory required')
    if len({r['id'] for r in records})!=len(records) or len({j['id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate logical/physical identity')
    if len({r['run_id'] for r in records})!=phase['rows'] or len({(r['run_id'],r['kind']) for r in records})!=len(records):raise ValueError('Every actual phase condition must have every frozen endpoint')
    for r in records:
        for rk,pk in (('phase_name','name'),('role','role'),('model_tag','model_tag'),('data_family','data_family'),('phase','expected_phase')):
            if r[rk]!=phase[pk]:raise ValueError('Logical record mixes scientific phase/model/data')
    job_ids={j['id'] for j in jobs};used={r['inference_job_id'] for r in records}
    used|={g['initial_job_id'] for g in manifest['baseline_groups'].values() if not g.get('initial_origin')}
    if used!=job_ids or any(j['kind'] not in ('initial','adapted','reuse') for j in jobs):raise ValueError('Unexpected or unused physical/reuse job')
    if len(manifest['baseline_groups'])!=1:raise ValueError('One phase requires one model/data/execution baseline identity')
    group=next(iter(manifest['baseline_groups'].values()));policy=group['inference_policy']
    for key,value in contract['inference_policy_expect'].items():
        if policy.get(key)!=value:raise ValueError('Frozen inference policy differs: '+key)
    for job in jobs:
        rr=[r for r in records if r['inference_job_id']==job['id']]
        if {r['id'] for r in rr}!=set(job['record_ids']):raise ValueError('Job/logical-record coverage differs')
        if any(r['weights_identity']!=job.get('weights_identity') or r['baseline_group']!=job['baseline_group'] or
               r['initial_job_id']!=manifest['baseline_groups'][r['baseline_group']]['initial_job_id'] for r in rr):
            raise ValueError('Logical endpoint weights/baseline differ from its canonical job')
    counts={'logical_endpoints':len(records),'queue_jobs':len(jobs),'new_forward_jobs':sum(j['kind']!='reuse' for j in jobs),
        'reuse_jobs':sum(j['kind']=='reuse' for j in jobs),'imported_baselines':sum(bool(g.get('initial_origin')) for g in manifest['baseline_groups'].values()),
        'logical_aliases':sum(Path(r['output']).resolve()!=Path(next(j for j in jobs if j['id']==r['inference_job_id'])['output']).resolve() for r in records)}
    workload=manifest['workload']
    for k,w in (('logical_endpoints','logical_checkpoint_endpoints'),('queue_jobs','total_queue_jobs'),('new_forward_jobs','total_inference_jobs'),('reuse_jobs','reuse_materialization_jobs')):
        if counts[k]!=workload[w]:raise ValueError('Manifest workload differs from complete physical inventory')
    return counts


def source_closure(manifest_proof,queue_proof,priors,verifier):
    """Select only required prior jobs, never unrelated prior phase scores."""
    authorities={};definitions={};owners={}
    for source in [{'queue':queue_proof,'manifests':[manifest_proof]},*priors]:
        queue=verifier.proof(source['queue'])
        for p in source['manifests']:authorities[p['sha256']]=(p,verifier.proof(p))
        for definition in queue:
            if definition['id'] in definitions and definitions[definition['id']]!=definition:raise ValueError('Changed prior job definition')
            argv=definition['argv']
            if len(argv)!=10 or Path(argv[1]).name!='run_frozen_evaluation.py':raise ValueError('Only exact frozen single-GPU workers are accepted')
            args=dict(zip(argv[2::2],argv[3::2]))
            if set(args)!={'--manifest','--manifest-sha256','--job-id','--device'} or args['--device']!='cuda:0' or args['--job-id']!=definition['id']:
                raise ValueError('Worker argv differs from frozen evaluation grammar')
            definitions[definition['id']]=definition;owners[definition['id']]=(str(Path(args['--manifest']).resolve()),args['--manifest-sha256'])
    selected={};visiting=set()
    def include(name):
        if name in selected:return
        if name in visiting or name not in definitions:raise ValueError('Missing/cyclic explicit prior inference closure')
        visiting.add(name);path,digest=owners[name]
        if digest not in authorities or str(Path(authorities[digest][0]['path']).resolve())!=path:raise ValueError('Worker lacks explicit manifest authority')
        p,m=authorities[digest];matches=[j for j in m['jobs'] if j['id']==name]
        if len(matches)!=1:raise ValueError('Prior manifest/job identity missing')
        job=matches[0]
        for parent in definitions[name].get('depends_on',[]):include(parent)
        group=m['baseline_groups'][job['baseline_group']]
        if job['kind']!='initial':include(group['initial_job_id'])
        if job.get('inference_origin'):include(job['inference_origin']['job_id'])
        selected[name]={'manifest':p,'job_id':name,'kind':job['kind'],'output':job['output'],'definition':definitions[name]};visiting.remove(name)
    target=authorities[manifest_proof['sha256']][1]
    for job in target['jobs']:include(job['id'])
    # Every imported reference must resolve to a genuine canonical job with the
    # exact policy/assets/data/scorer identity, not merely a familiar job ID.
    for item in selected.values():
        m=authorities[item['manifest']['sha256']][1];job=next(j for j in m['jobs'] if j['id']==item['job_id']);group=m['baseline_groups'][job['baseline_group']]
        refs=[group['initial_origin']] if group.get('initial_origin') else []
        if job.get('inference_origin'):refs.append(job['inference_origin'])
        for origin in refs:
            source=selected[origin['job_id']]
            if source['manifest']['sha256']!=origin['manifest_sha256'] or str(Path(source['manifest']['path']).resolve())!=str(Path(origin['manifest']).resolve()):raise ValueError('Imported inference manifest differs')
            pm=authorities[source['manifest']['sha256']][1];pj=next(j for j in pm['jobs'] if j['id']==origin['job_id']);pg=pm['baseline_groups'][pj['baseline_group']]
            if pj['kind']=='reuse' or pj['output']!=origin['output'] or m['mode']!=pm['mode'] or job['baseline_group']!=pj['baseline_group']:
                raise ValueError('Imported baseline/inference is not canonical and matched')
            for key in ('inference_policy','base_model_assets_sha256','evaluation_input_sha256'):
                if group[key]!=pg[key]:raise ValueError('Imported baseline has different model/data/policy')
            if m['scorer_input_sha256']!=pm['scorer_input_sha256']:raise ValueError('Imported baseline scorer differs')
            if origin==group.get('initial_origin') and pj['kind']!='initial':raise ValueError('Imported baseline is adapted')
            if origin==job.get('inference_origin') and pj.get('weights_identity')!=job.get('weights_identity'):raise ValueError('Imported inference weights differ')
    return list(selected.values())


def prepare(ready_proof,protocol_proof,contract_id,cache_snapshot=None):
    verifier=Verifier();protocol=verifier.proof(protocol_proof);ready=verifier.proof(ready_proof)
    if (protocol.get('schema_version')!=2 or protocol.get('operational_revision')!=2 or protocol.get('bootstrap_replicates')!=5000 or protocol.get('bootstrap_seed')!=20261106 or
            protocol.get('confidence')!=.95 or protocol.get('allow_partial_for_final_conclusions') is not False):
        raise ValueError('Unsupported frozen supplemental analysis protocol')
    for path,value in protocol['analysis_source_sha256'].items():verifier.bind(path,value)
    if protocol.get('catalog'):verifier.proof(protocol['catalog'])
    for item in protocol.get('policy_sources',[]):verifier.proof(item)
    for item in protocol.get('preparation_sources',[]):verifier.bind(item['path'],item['sha256'])
    if protocol.get('preparation_ready_root') and Path(ready_proof['path']).resolve().parent.parent!=Path(protocol['preparation_ready_root']).resolve():raise ValueError('READY is outside the frozen active preparation root')
    contract=protocol['contracts'][contract_id]
    if ready.get('status')!='ready' or ready.get('ready') is not True or ready.get('id')!=contract_id:raise ValueError('A real matching complete READY receipt is required')
    manifest=verifier.proof(ready['manifest']);counts=contract_manifest(manifest,contract)
    spec=verifier.proof(ready['phase_spec']);verifier.proof(ready['queue'])
    if (manifest['phase_spec_sha256']!=ready['phase_spec']['sha256'] or str(Path(manifest['phase_spec']).resolve())!=str(Path(ready['phase_spec']['path']).resolve()) or
            ready['logical_endpoints']!=counts['logical_endpoints'] or ready['workload']!=manifest['workload']):raise ValueError('READY/spec/manifest inventory binding differs')
    closure=source_closure(ready['manifest'],ready['queue'],protocol['prior_sources'],verifier)
    checkpoint_paths={}
    def checkpoint(path,digest):
        canonical=str(Path(path).resolve())
        if canonical in checkpoint_paths and checkpoint_paths[canonical]!=digest:raise ValueError('Contradictory explicit checkpoint identities')
        checkpoint_paths[canonical]=digest
    closure_manifests={}
    for item in closure:
        digest=item['manifest']['sha256']
        if digest not in closure_manifests:closure_manifests[digest]=verifier.proof(item['manifest'])
        cm=closure_manifests[digest];cj=next(j for j in cm['jobs'] if j['id']==item['job_id'])
        for endpoint in [cj,*[r for r in cm.get('records',[]) if r['inference_job_id']==cj['id']]]:
            if endpoint.get('checkpoint'):checkpoint(endpoint['checkpoint'],endpoint['checkpoint_sha256'])
            evidence=endpoint.get('source_model_equivalence')
            if evidence:
                verifier.bind(evidence['proof'],evidence['proof_sha256']);record=read(evidence['proof'])['comparisons'][evidence['comparison_index']]
                for side in (record['left'],record['right']):checkpoint(side['checkpoint'],side['checkpoint_sha256'])
    verification={'mode':'fresh','cache_snapshot':None}
    for cm in closure_manifests.values():
        if any(str(Path(p).resolve()) in checkpoint_paths for p in cm['common_input_sha256']):raise ValueError('Collector common sources must not contain checkpoint payloads')
    if cache_snapshot:
        source=protocol['checkpoint_cache_reader_source'];verifier.bind(source['path'],source['sha256']);verifier.proof(cache_snapshot)
        verification={'mode':'v5_checkpoint_snapshot','cache_snapshot':cache_snapshot,'cache_reader_source':source}
        Verifier(verification)  # Validate read-only snapshot/reader schema; no CP is eligible or read here.
    counts.update(closure_queue_jobs=len(closure),closure_canonical_forward_jobs=sum(j['kind']!='reuse' for j in closure),
        imported_canonical_forward_jobs=sum(j['kind']!='reuse' and j['manifest']!=ready['manifest'] for j in closure),
        closure_reuse_materializations=sum(j['kind']=='reuse' for j in closure))
    if contract.get('source_registry'):
        source_rows=verifier.proof(contract['source_registry']);lookup={r['id']:r for r in source_rows}
        if len(lookup)!=len(source_rows):raise ValueError('Duplicate original scientific condition')
        for r in manifest['records']:
            if r['run_id'] not in lookup:raise ValueError('Manifest condition is outside the original scientific registry')
            original=lookup[r['run_id']]
            for k in ('group','ratio','weight','learning_rate','selection_seed','generation_seed'):
                if r.get(k)!=original.get(k):raise ValueError('Manifest differs from original scientific condition: '+k)
            if r['training_seed']!=original.get('training_seed',original.get('seed')):raise ValueError('Manifest training seed differs from original registry')
        if len(spec.get('phases',[]))!=1 or spec.get('science_frozen') is not True:raise ValueError('Real complete scientifically frozen phase spec required')
        for k,value in contract['scientific_identity'].items():
            if spec['phases'][0].get(k)!=value:raise ValueError('Phase spec changes frozen model/data identity: '+k)
        for record in manifest['records']:
            inputs={str(Path(p).resolve()):d for p,d in {**manifest['common_input_sha256'],**record['input_sha256']}.items()}
            for p,d in contract['source_proof_files'].items():
                if inputs.get(str(Path(p).resolve()))!=d:raise ValueError('Manifest omits a frozen scientific source proof')
    selection=None
    if contract['family']=='A':
        from .selected_policy_analysis import bind_selection
        selection_proof=protocol['A_selection']
        if selection_proof.get('sha256') is None:selection_proof=proof(selection_proof['path'])
        selection=bind_selection(selection_proof,protocol['A_registry'],verifier)
        ids={r['run_id'] for r in manifest['records']}
        if contract['mode']=='behavior' and ids!=set(selection['selected_ids']):raise ValueError('A behavior manifest differs from the validation-selected18 IDs')
        if contract['mode']=='nll' and ids!=set(selection['grid_ids']):raise ValueError('A NLL manifest must retain the full126-cell grid')
    if contract['mode']=='nll':
        from .nll_v2_analysis import prepare_plan
        phase=contract['phase']
        for kind in contract['checkpoints']:prepare_plan(ready['manifest']['path'],ready['manifest']['sha256'],role=phase['role'],phase_name=phase['name'],model_tag=phase['model_tag'],data_family=phase['data_family'],checkpoint=kind)
    verifier.finish()
    return {'schema_version':2,'operational_revision':2,'prepared_at_utc':now(),'prepared_only':True,'score_access_authorized':False,'contract_id':contract_id,
        'contract':contract,'ready':ready_proof,'protocol':protocol_proof,'manifest':ready['manifest'],'counts':counts,'closure_jobs':closure,
        'closure_job_count':len(closure),'selection':selection,'preparation_source_sha256':verifier.sources,
        'checkpoint_verification':verification,'checkpoint_paths':checkpoint_paths,
        'first_read_receipt_path':str((Path(protocol['first_read_root'])/(contract_id+'_'+ready['manifest']['sha256']+'.json')).resolve()),
        'test_or_behavior_scores_parsed':False,'model_inference_started':False}


def load_plan(plan_proof):
    v=Verifier();plan=v.proof(plan_proof)
    if plan.get('schema_version')!=2 or plan.get('operational_revision')!=2:raise ValueError('Explicit v2 operational plan required')
    for path,value in plan['preparation_source_sha256'].items():v.bind(path,value)
    fresh=prepare(plan['ready'],plan['protocol'],plan['contract_id'],plan['checkpoint_verification'].get('cache_snapshot'))
    for key in ('counts','contract','closure_jobs','selection','first_read_receipt_path','checkpoint_verification','checkpoint_paths'):
        if fresh[key]!=plan[key]:raise ValueError('Prepared analysis identity changed: '+key)
    v.configure(plan['checkpoint_verification']);v.set_checkpoints(plan['checkpoint_paths'])
    return plan,v


def inspect(plan_proof,state_path):
    """Fail closed on whole scheduler closure before even probing outputs."""
    started=time.perf_counter()
    plan,v=load_plan(plan_proof);raw=Path(state_path).read_bytes();state=json.loads(raw)
    pending=[]
    for item in plan['closure_jobs']:
        s=state['jobs'].get(item['job_id'],{});definition=item['definition']
        # schedule_jobs uses JSON sort_keys=True without pretty-print/newline.
        digest=hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()
        if s.get('definition_sha256')!=digest:raise ValueError('Scheduler definition differs: '+item['job_id'])
        if s.get('status')!='completed' or s.get('returncode')!=0 or s.get('missing_outputs'):pending.append(item['job_id'])
    result={'schema_version':2,'operational_revision':2,'checked_at_utc':now(),'plan':plan_proof,'contract_id':plan['contract_id'],'counts':plan['counts'],
        'scheduler_state':str(state_path),'scheduler_state_sha256':hashlib.sha256(raw).hexdigest(),'closure_job_count':len(plan['closure_jobs']),
        'ready':not pending,'pending_job_ids':pending,'score_files_hashed':False,'score_json_parsed':False}
    if pending:
        result['verification_work']=v.accounting()
        result['score_blind_verification_wall_seconds']=time.perf_counter()-started
        return result
    # Output byte hashing and all source/checkpoint hashing happen only now.
    manifests={};attestations=[]
    def weights_evidence(checkpoint_sha,identity,evidence):
        if evidence is None:
            if checkpoint_sha is not None and identity!='checkpoint:'+checkpoint_sha:raise ValueError('Unproved model-state inference identity')
            return
        v.bind(evidence['proof'],evidence['proof_sha256']);p=read(evidence['proof']);comparison=p['comparisons'][evidence['comparison_index']]
        fp=evidence['fingerprint'];sides=[comparison['left'],comparison['right']]
        if (p.get('schema_version')!=1 or p.get('status')!='passed' or p.get('algorithm')!='sha256_sorted_key_dtype_shape_contiguous_uint8_v1' or
                not re.fullmatch(r'[0-9a-f]{64}',fp.get('sha256_model_state','')) or type(fp.get('state_elements')) is not int or fp['state_elements']<1 or
                comparison.get('bitwise_model_state_equal') is not True or any(s['model_state_fingerprint']!=fp for s in sides) or
                checkpoint_sha not in {s['checkpoint_sha256'] for s in sides} or identity!='model_state:'+fp['sha256_model_state'] or
                {s['checkpoint_sha256'] for s in sides}!={evidence['left_checkpoint_sha256'],evidence['right_checkpoint_sha256']}):
            raise ValueError('Logical/canonical endpoint lacks a true complete-state equivalence proof')
        source=p['algorithm_source'];sourcepath=Path(source['path'])
        if not sourcepath.is_absolute():sourcepath=Path(__file__).resolve().parents[2]/sourcepath
        v.bind(sourcepath,source['sha256'])
        for name in ('source_audit','source_registry','source_snapshot','producer_source'):
            if p.get(name):v.proof(p[name])
        for side in sides:v.bind(side['checkpoint'],side['checkpoint_sha256'])
    def verify_result(m,digest,job,record=None):
        output=record['output'] if record else job['output'];ap=attestation_path(output);v.bind(ap);att=read(ap)
        expected={'manifest_sha256':digest,'job_id':job['id'],'baseline_group':job['baseline_group'],'verified_frozen_inputs':True,
            'checkpoint_sha256':record['checkpoint_sha256'] if record else job.get('checkpoint_sha256'),
            'weights_identity':record['weights_identity'] if record else job.get('weights_identity')}
        if any(att.get(k)!=value for k,value in expected.items()):raise ValueError('Execution attestation identity differs: '+str(ap))
        alias=record is not None
        if att.get('inference_alias') is not alias:raise ValueError('Canonical/alias attestation differs')
        if alias:
            if (att.get('logical_record_id')!=record['id'] or att.get('inference_subprocess_wall_seconds')!=0 or
                    att.get('canonical_output')!=str(Path(job['output']).resolve()) or att.get('source_model_equivalence')!=record.get('source_model_equivalence')):
                raise ValueError('Logical alias lacks a bound zero-forward canonical proof')
            if job['kind']=='reuse' and att.get('inference_origin')!=job['inference_origin']:raise ValueError('Reuse wrapper origin differs')
        elif att.get('actual_argv')!=job['argv']:raise ValueError('Canonical scorer argv differs')
        paths={str(output)}
        if m['mode']=='behavior':paths.update(str(Path(output).with_name(Path(output).stem+'_'+name+'.jsonl')) for name in m['baseline_groups'][job['baseline_group']]['tasks'])
        if set(att.get('output_sha256',{}))!=paths:raise ValueError('Complete exact summary/task output set required')
        for p,value in att['output_sha256'].items():v.bind(p,value)
        attestations.append({'manifest_sha256':digest,'job_id':job['id'],'logical_record_id':record['id'] if record else None,'attestation':proof(ap),'output_sha256':att['output_sha256']})
    for item in plan['closure_jobs']:
        p=item['manifest']
        if p['sha256'] not in manifests:manifests[p['sha256']]=v.proof(p)
        m=manifests[p['sha256']];job=next(j for j in m['jobs'] if j['id']==item['job_id'])
        for path,value in {**m['common_input_sha256'],**job['input_sha256']}.items():v.bind(path,value)
        weights_evidence(job.get('checkpoint_sha256'),job.get('weights_identity'),job.get('source_model_equivalence'))
        if job['kind']!='reuse':verify_result(m,p['sha256'],job)
        for record in m.get('records',[]):
            if record['inference_job_id']!=job['id']:continue
            for path,value in record['input_sha256'].items():v.bind(path,value)
            weights_evidence(record['checkpoint_sha256'],record['weights_identity'],record.get('source_model_equivalence'))
            if Path(record['output']).resolve()!=Path(job['output']).resolve():verify_result(m,p['sha256'],job,record)
    v.finish();result.update(score_files_hashed=True,attestations=attestations,source_sha256=v.sources,all_source_baseline_alias_output_integrity_verified=True)
    result['verification_work']=v.accounting()
    result.update(score_blind_verification_wall_seconds=time.perf_counter()-started,timing_scope='CPU and I/O wall, including plan/source/output SHA; not GPU time or an additive child of scorer wall')
    return result


def authorize(plan_proof,authority_proof):
    plan,v=load_plan(plan_proof);a=v.proof(authority_proof)
    expected={'authorized_by':'root','allow_score_analysis':True,'plan':plan_proof,'protocol':plan['protocol'],'ready':plan['ready'],
        'manifest':plan['manifest'],'contract_id':plan['contract_id'],'expected_counts':plan['counts'],'allow_partial':False,
        'checkpoint_verification':plan['checkpoint_verification'],
        'training_methods_grid_selection_horizons_frozen':True,'test_or_behavior_used_for_selection':False}
    if any(a.get(k)!=value for k,value in expected.items()):raise PermissionError('Explicit root authority does not bind this complete phase and analysis')
    return plan,a


def first_read(path,plan_proof,authority_proof,readiness_proof,first_score_path,checkpoint_verification=None):
    value={'schema_version':2,'operational_revision':2,'checkpoint_verification':checkpoint_verification,
        'semantic_access_enabled_at_utc':now(),'plan':plan_proof,'authority':authority_proof,'readiness':readiness_proof,
        'first_planned_score_path':str(first_score_path),'before_first_score_parse':True,'complete_phase_source_and_output_SHA_verified':True,
        'trigger':'Immediately before invoking the authorized existing collector; no score parsing occurred in this guard',
        'no_test_or_behavior_selection':True}
    return immutable(path,value)
