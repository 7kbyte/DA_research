PROFILES = {
    'pythia-1b': {'phase': 'extended_pythia_1b_100m', 'groups': GROUPS,
                  'pairs': PAIRS, 'horizon': 100_000_000},
    'pythia-2.8b': {'phase': 'extended_pythia_2p8b_25m',
                    'groups': ('B0','B1','B2','B4','M'),
                    'pairs': ('M-B4','B2-B1','B4-B2'), 'horizon': 25_000_000},
}
BOUND = {}
IO = {'read_bytes': 0, 'paths': set(), 'semantic_json_paths': set()}
FORBIDDEN_SUFFIXES = {'.pt','.pth','.bin','.safetensors','.zip','.tar','.gz','.zst','.npz','.npy'}

def safe_bytes(path):
    p=Path(path).resolve()
    require(not (set(p.suffixes)&FORBIDDEN_SUFFIXES) and 'checkpoints' not in p.parts,
            'Checkpoint/base/archive/tensor payload reads forbidden')
    value=p.read_bytes();IO['read_bytes']+=len(value);IO['paths'].add(str(p));return value

def sha(path): return hashlib.sha256(safe_bytes(path)).hexdigest()
def read(path):
    value=safe_bytes(path);IO['semantic_json_paths'].add(str(Path(path).resolve()));return json.loads(value)
def rows(path): return list(csv.DictReader(safe_bytes(path).decode().splitlines()))
def canonical(path): return str(Path(path).resolve())
def proof(path): return {'path':canonical(path),'sha256':sha(path)}
def bind(value, opaque=False):
    require(isinstance(value,dict) and isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str),'Malformed proof')
    actual=sha(value['path']);require(actual==value['sha256'],'Changed bound source '+value['path'])
    path=canonical(value['path']);require(path not in BOUND or BOUND[path]==actual,'Contradictory bound source');BOUND[path]=actual
    return None if opaque else read(path)
def exact_proof(a,b): return a['sha256']==b['sha256'] and canonical(a['path'])==canonical(b['path'])
def definition_sha(job): return hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
def flag_value(argv,flag):
    require(argv.count(flag)==1 and argv.index(flag)+1<len(argv),'Missing/duplicate command flag '+flag)
    return argv[argv.index(flag)+1]
def attestation_path(output):
    p=Path(output);return str(p.with_name(p.stem+'_execution.json'))

def validate_pairs(values,groups,pairs):
    actual=[(r['checkpoint'],r['comparison'],int(r['training_seed'])) for r in values]
    expected={(k,p,s) for k in ('final','best') for p in pairs for s in SEEDS}
    require(len(actual)==len(expected) and len(set(actual))==len(actual) and set(actual)==expected,'Complete frozen contrast grid required')
    for r in values:
        require(r['comparison'].split('-')[0] in groups and r['comparison'].split('-')[1] in groups,'Foreign comparison group')

def validate_state(plan,queue,state):
    require(isinstance(queue,list) and len({j['id'] for j in queue})==len(queue),'Unique actual scheduler queue required')
    actual={j['id']:j for j in queue};closure=plan['closure_jobs']
    require(len({j['job_id'] for j in closure})==len(closure)==plan['closure_job_count'],'Duplicate/incomplete closure')
    closure_ids={j['job_id'] for j in closure}
    for item in closure:
        jid=item['job_id'];definition=item['definition']
        require(jid in actual and actual[jid]==definition and definition['id']==jid,'Foreign/changed actual queue definition '+jid)
        require(set(definition.get('depends_on',[]))<=closure_ids,'Missing baseline dependency closure')
        argv=definition['argv'];require(len(argv)>2 and Path(argv[1]).name=='run_frozen_evaluation.py','Expected frozen evaluation wrapper')
        require(flag_value(argv,'--job-id')==jid and flag_value(argv,'--manifest-sha256')==item['manifest']['sha256'] and canonical(flag_value(argv,'--manifest'))==canonical(item['manifest']['path']),'Actual wrapper manifest/job differs')
        r=state.get('jobs',{}).get(jid,{})
        require(r.get('status')=='completed' and type(r.get('returncode')) is int and r['returncode']==0,'Strict completed integer rc0 required '+jid)
        require(not any(k.startswith('stop_requested') for k in r) and not r.get('missing_outputs'),'Stopped/missing-output closure '+jid)
        require(r.get('definition_sha256')==definition_sha(definition) and r.get('argv')==argv,'Foreign state definition/actual argv '+jid)
        require(isinstance(r.get('elapsed_seconds'),(float,int)) and r['elapsed_seconds']>=0 and isinstance(r.get('started_at_epoch'),(float,int)),'Missing actual whole-slot timing')
        require(all(Path(x).is_file() for x in definition.get('outputs',[])),'Missing declared closure output')
    return actual

def validate_alias(record,job,manifest,model):
    eq=record.get('source_model_equivalence')
    require(isinstance(eq,dict),'Alias requires explicit full-model proof')
    value=bind({'path':eq['proof'],'sha256':eq['proof_sha256']})
    require(value.get('status','passed')=='passed' and value.get('model',model)==model,'Wrong/failed equivalence proof model')
    require(value['algorithm']==eq['algorithm']=='sha256_sorted_key_dtype_shape_contiguous_uint8_v1','Unknown full-state algorithm')
    for key in ('source_audit','source_registry','source_snapshot'):
        if key in value: bind(value[key])
    for key in ('algorithm_source','producer_source'):
        if key in value: bind(value[key],opaque=True)
    index=eq['comparison_index'];require(type(index) is int and 0<=index<len(value['comparisons']),'Bad equivalence index')
    pair=value['comparisons'][index];require(pair.get('bitwise_model_state_equal') is True,'Unproved/nonidentical alias')
    left,right=pair['left'],pair['right'];fp=eq['fingerprint']
    require(left['model_state_fingerprint']==right['model_state_fingerprint']==fp,'Incomplete/mismatched model fingerprint')
    require(eq['left_checkpoint_sha256']==left['checkpoint_sha256'] and eq['right_checkpoint_sha256']==right['checkpoint_sha256'],'Wrong equivalence CP SHA')
    canonical_records=[r for r in manifest['records'] if r['inference_job_id']==job['id'] and r['output']==job['output']]
    require(len(canonical_records)==1,'Alias requires one canonical physical record')
    canonical_record=canonical_records[0]
    for r in (record,canonical_record):
        matches=[s for s in (left,right) if canonical(s['checkpoint'])==canonical(r['checkpoint']) and s['checkpoint_sha256']==r['checkpoint_sha256']]
        require(len(matches)==1 and r['weights_identity']=='model_state:'+fp['sha256_model_state'],'Proof does not cover actual endpoint CP')
        side=matches[0]
        require(side.get('run_id',r['run_id'])==r['run_id'] and side.get('kind',r['kind'])==r['kind'],'Proof selected endpoint differs')
        if 'checkpoint_exposure' in side:
            require(all(side['checkpoint_exposure'][k]==r[k] for k in ('step','target_prediction_tokens_at_checkpoint','replay_prediction_tokens_at_checkpoint')),'Proof exposure differs')
        for k in ('config','summary'):
            if k+'_sha256' in side:
                require(canonical(side[k+'_path'])==canonical(r[k+'_path']) and sha(r[k+'_path'])==side[k+'_sha256'],'Proof source binding differs')
    return eq

def validate_timer(value,command,complete,plan,state_path):
    timer=bind(value);argv=timer.get('argv',[])
    require(type(timer.get('returncode')) is int and timer['returncode']==0 and not timer.get('timed_out',False),'Timer must describe successful complete child')
    require(len(argv)>2 and Path(argv[1]).name=='analyze_supplemental_phase_v3.py' and argv[2]==command,'Wrong actual analysis/check timer argv')
    for flag,p in (('--plan',complete['plan']),('--operational-repair',complete['operational_repair'])):
        require(canonical(flag_value(argv,flag))==canonical(p['path']) and flag_value(argv,flag+'-sha256')==p['sha256'],'Timer source/plan mismatch')
    require(canonical(flag_value(argv,'--state'))==canonical(state_path),'Timer actual scheduler state differs')
    if command=='analyze':
        require(canonical(flag_value(argv,'--authority'))==canonical(complete['authority']['path']) and flag_value(argv,'--authority-sha256')==complete['authority']['sha256'],'Analyze timer authority differs')
    require(math.isfinite(timer['wall_seconds']) and timer['wall_seconds']>=0,'Missing full child timer')
    return timer

def gate_inputs(args):
    profile=PROFILES[args.model];n=len(profile['groups'])*len(SEEDS)
    complete=bind({'path':str(args.complete),'sha256':args.complete_sha256})
    require(complete.get('complete') is True and complete['endpoints']==[{'endpoint':'final','logical_conditions':n},{'endpoint':'best','logical_conditions':n}],'Whole model final+best COMPLETE required')
    require(complete['authority']['sha256']==args.authority_sha256,'Explicit ROOT authority SHA differs')
    authority,plan,ready,journal,repair=[bind(complete[k]) for k in ('authority','plan','readiness','first_read_receipt','operational_repair')]
    cid=profile['phase']+'_nll'
    require(plan['contract_id']==cid and authority['contract_id']==cid and ready['contract_id']==cid,'Wrong model/phase contract')
    require(authority.get('authorized_by')=='root' and authority.get('allow_score_analysis') is True and authority.get('allow_partial') is False,'Exact whole-phase ROOT score authority required')
    for key in ('plan','operational_repair'):
        require(exact_proof(authority[key],complete[key]),'Authority plan/repair mismatch')
    require(exact_proof(authority['manifest'],plan['manifest']) and exact_proof(authority['ready'],plan['ready']),'Authority manifest/READY mismatch')
    require(exact_proof(ready['plan'],complete['plan']) and exact_proof(ready['operational_repair'],complete['operational_repair']),'Readiness source mismatch')
    require(ready.get('ready') is True and not ready['pending_job_ids'] and ready['closure_job_count']==plan['closure_job_count'],'Incomplete whole closure')
    require(ready.get('score_json_parsed') is False and ready.get('all_source_baseline_alias_output_integrity_verified') is True,'Missing pre-semantic integrity gate')
    for key in ('plan','authority','readiness'):
        require(exact_proof(journal[key],complete[key]),'First-read journal binding differs')
    require(canonical(complete['first_read_receipt']['path'])==canonical(plan['first_read_receipt_path']) and journal.get('before_first_score_parse') is True and journal.get('complete_phase_source_and_output_SHA_verified') is True and bool(journal.get('semantic_access_enabled_at_utc')),'Whole-phase first-read journal required')
    require(journal.get('no_test_or_behavior_selection') is True,'Missing no-selection first-read declaration')
    for a,b in (('retained_plan',complete['plan']),('retained_manifest',plan['manifest']),('retained_protocol',plan['protocol'])):
        require(exact_proof(repair[a],b),'Repair retained source differs')
    require(repair.get('this_is_not_a_ROOT_score_authority') is True and repair.get('score_access_authorized') is False and repair.get('scientific_or_statistics_changes') is False,'Repair cannot grant scientific/score changes')
    require(canonical(repair['actual_first_read_journal_path_preserved'])==canonical(plan['first_read_receipt_path']),'Repair moved journal')
    bind(plan['protocol']);actual_ready=bind(plan['ready']);manifest=bind(plan['manifest'])
    require(actual_ready.get('ready') is True and exact_proof(actual_ready['manifest'],plan['manifest']),'Actual READY/manifest differs')
    phase=plan['contract']['phase']
    require(phase['name']==profile['phase'] and phase['model_tag']==args.model and phase['data_family']=='extended_v1' and phase['role']=='extended' and phase['rows']==n and phase['expected_phase']=='formal','Wrong complete model scope')
    require(complete['parsed_phase']==phase and plan['contract']['logical_endpoints']==2*n and plan['contract']['checkpoints']==['final','best'],'Changed parsed phase/endpoints')
    require(len(manifest['phases'])==1 and manifest['phases'][0]['name']==profile['phase'] and manifest['phases'][0]['model_tag']==args.model and manifest['phases'][0]['data_family']=='extended_v1' and manifest['mode']=='nll','Foreign manifest phase/model')
    require(len(manifest['baseline_groups'])==1,'One matched extended baseline required');group=next(iter(manifest['baseline_groups'].values()))
    identity=plan['contract']['scientific_identity']
    require(group['model_tag']==args.model and identity['model_tag']==args.model and identity['data_family']=='extended_v1','Matched model/data baseline required')
    require(group['test_paths']=={k:v['path'] for k,v in identity['test_data'].items()} and group['evaluation_input_sha256']=={v['path']:v['sha256'] for v in identity['test_data'].values()},'Main-v3/foreign baseline cannot replace matched extended initial')
    records=manifest['records'];validate_grid([{'checkpoint':r['kind'],'group':r['group'],'training_seed':r['training_seed']} for r in records],profile['groups'])
    require(len({r['id'] for r in records})==2*n and len({r['actual_run_id'] for r in records})==n,'Duplicate/aliased physical training identity')
    for r in records:
        require(r['model_tag']==args.model and r['phase_name']==profile['phase'] and r['data_family']=='extended_v1' and r['phase']=='formal' and r['is_training_alias'] is False and r['run_id']==r['actual_run_id'],'Foreign/aliased full training record')
        require(r['baseline_group']==group['id'] and r['initial_job_id']==group['initial_job_id'],'Unmatched initial group')
        full=r['full_training_committed_path'];require(full['target_prediction_tokens']==profile['horizon'],'Wrong full scientific horizon')
        if r['kind']=='final': require(r['target_prediction_tokens_at_checkpoint']==profile['horizon'],'Partial final exposure')
        for key in ('config_path','summary_path'):
            require(r['input_sha256'].get(r[key])==sha(r[key]),'Config/summary not bound to frozen manifest')
    require(canonical(args.state)==canonical(ready['scheduler_state']) and args.state_sha256==ready['scheduler_state_sha256'],'Explicit immutable state must be bound actual successor snapshot')
    queue=bind({'path':str(args.queue),'sha256':args.queue_sha256});state=bind({'path':str(args.state),'sha256':args.state_sha256});validate_state(plan,queue,state)
    require(len({j['id'] for j in manifest['jobs']})==len(manifest['jobs']) and all(j['id'] in {x['job_id'] for x in plan['closure_jobs']} for j in manifest['jobs']),'Manifest jobs outside actual closure')
    jobs={j['id']:j for j in manifest['jobs']};require(jobs[group['initial_job_id']]['kind']=='initial' and jobs[group['initial_job_id']]['output']==group['initial_output'],'Matched extended initial job missing')
    require(sum(j['kind']=='initial' for j in manifest['jobs'])==1,'Exactly one matched initial physical job required')
    attestations={canonical(v['attestation']['path']):v for v in ready['attestations']}
    expected_att={canonical(attestation_path(x['output'])):x['output'] for x in [*manifest['jobs'],*records]}
    require(set(attestations)==set(expected_att) and len(attestations)==len(ready['attestations']),'Incomplete/extra whole-phase attestations')
    # Hash score bytes only after explicit whole-phase unseal metadata; parse them later.
    for path,evidence in attestations.items():
        require(evidence['manifest_sha256']==plan['manifest']['sha256'],'Foreign attestation manifest')
        att=bind(evidence['attestation']);require(att['manifest_sha256']==plan['manifest']['sha256'] and canonical(att['manifest'])==canonical(plan['manifest']['path']) and att['job_id']==evidence['job_id'],'Foreign attestation identity')
        require(att['job_id'] in jobs and att['actual_argv']==jobs[att['job_id']]['argv'],'Attestation actual scorer argv differs')
        require(att['output_sha256']==evidence['output_sha256'],'Attested output hash map changed')
        require(set(att['output_sha256'])=={expected_att[path]},'Attestation does not cover its actual score output')
        for p,d in att['output_sha256'].items(): bind({'path':p,'sha256':d},opaque=True)
    alias_count=0
    for r in records:
        require(r['inference_job_id'] in jobs,'Foreign physical inference reference');job=jobs[r['inference_job_id']]
        if r['output']!=job['output']:
            validate_alias(r,job,manifest,args.model);att=read(attestation_path(r['output']))
            require(att.get('inference_alias') is True and att['inference_subprocess_wall_seconds']==0 and att['source_model_equivalence']==r['source_model_equivalence'],'Alias must add zero verified forward');alias_count+=1
    require(len(manifest['jobs'])==1+2*n-alias_count,'Physical work does not close logical endpoints')
    timers={kind:validate_timer({'path':str(getattr(args,kind+'_timing')),'sha256':getattr(args,kind+'_timing_sha256')},kind,complete,plan,args.state) for kind in ('check','analyze')}
    require(canonical(flag_value(timers['analyze']['argv'],'--output'))==canonical(Path(args.complete).parent),'Analyze timer output differs')
    return {'complete':complete,'plan':plan,'ready':ready,'manifest':manifest,'group':group,'state':state,'profile':profile,'aliases':alias_count,'timers':timers}
