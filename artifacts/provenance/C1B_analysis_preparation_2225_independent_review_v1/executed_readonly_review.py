import pathlib as pl,json,hashlib,datetime as dt,time,stat
P=pl.Path;W=P('/tmp/grad-research-20261106/worktree');D=P(__file__).resolve().parent;S=W/'artifacts/provenance/C1B_analysis_preparation_2225_v1';seen={};start=time.monotonic()
def need(v,m):
 if not v:raise ValueError(m)
def stamp(p):
 s=P(p).stat();need(stat.S_ISREG(s.st_mode),'regular required');return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def raw(p):
 p=P(p).resolve();k=str(p)
 if k not in seen:
  b=stamp(p);need(b['size']<32<<20 and p.suffix not in ['.pt','.zip','.tar','.safetensors'],'small nonpayload only:'+k);v=p.read_bytes();need(stamp(p)==b and len(v)==b['size'],'metadata stable');seen[k]=(v,hashlib.sha256(v).hexdigest())
 return seen[k][0]
def sha(p):raw(p);return seen[str(P(p).resolve())][1]
def read(p):return json.loads(raw(p))
def ref(r):need(sha(r['path'])==r['sha256'],'bound SHA:'+r['path']);return read(r['path'])
c=read(S/'COMPLETE.json');need(sha(S/'COMPLETE.json')=='b62f9dc271e9155efe80dfd8ade1399ccf1f21fbde07ae90cb7c14168190151d','ROOT COMPLETE SHA')
need(c['status']=='prepared_only_passed_no_score_read_or_authority' and c['check_or_analyze_executed'] is False and c['TRUE_authority_created'] is False and c['first_read_journal_created'] is False,'prepared-only status')
for r in c['namespace_files']:need(sha(r['path'])==r['sha256'],'namespace SHA')
protocol=ref(c['original_protocol']);ref(c['reader_source']);cache=ref(c['cache_snapshot']);raw(c['doc']['path']);need(sha(c['doc']['path'])==c['doc']['sha256'],'doc raw SHA');future=ref(c['future_commands']);pre=read(S/'metadata_preflight.json');need(pre['checkpoint_unique_count']==48 and pre['missing_or_changed_checkpoint_receipts']==[] and pre['checkpoint_payload_current_read_bytes']==pre['base_payload_current_read_bytes']==0,'preflight48/no payload')
need(c['actual_successor_immutable_state_binding']['path'] is None and c['actual_successor_immutable_state_binding']['sha256'] is None and c['actual_successor_immutable_state_binding']['same_actual_immutable_path_and_SHA_required_from_first_check_through_analyze'] is True,'pending immutable successor')
need(future['prepared_only'] is True and future['score_access_authorized'] is False and future['actual_successor_immutable_state_binding']['path'] is None and future['actual_successor_immutable_state_binding']['sha256'] is None and future['actual_successor_immutable_state_binding']['same_source_required_for_both_check_and_analyze'] is True,'future state boundary')
proofs={}
for k,r in c['actual_D_B4_full_state_proof_sources'].items():proofs[k]=ref(r)
for endpoint in ['final','best']:
 p=next(v for k,v in proofs.items() if k.endswith('/'+endpoint+'_model_equivalence_proofs.json'))
 need(p['status']=='passed' and p['endpoint']==endpoint and p['model']=='pythia-1b' and p['algorithm']=='sha256_sorted_key_dtype_shape_contiguous_uint8_v1' and len(p['comparisons'])==3 and {x['seed'] for x in p['comparisons']}=={11,22,33},'endpoint full-state proof')
 need(all(x['bitwise_model_state_equal'] is True and x['left']['model_state_fingerprint']==x['right']['model_state_fingerprint'] for x in p['comparisons']),'three true full-state pairs')
checked=[];cpall=set();kindbytes={'final':0,'best':0};science={}
for contract in c['contracts']:
 plan=ref(contract['plan']);m=ref(contract['manifest']);ready=ref(plan['ready']);mode=plan['contract']['mode'];need(mode in ['nll','behavior'],'known mode');need(plan['prepared_only'] is True and plan['score_access_authorized'] is False and plan['model_inference_started'] is False and plan['test_or_behavior_scores_parsed'] is False,'v2 prepare only')
 need(plan['contract_id']==contract['contract_id']==ready['id'] and ready['status']=='ready' and ready['ready'] is True and ready['mode']==mode and ready['manifest']==contract['manifest']==plan['manifest'],'READY/plan/manifest binding')
 need(plan['protocol']==c['original_protocol'] and plan['checkpoint_verification']['cache_snapshot']==c['cache_snapshot'] and plan['checkpoint_verification']['cache_reader_source']==dict(path=c['reader_source']['path'],sha256=c['reader_source']['sha256']),'protocol/cache reader')
 for p,h in plan['preparation_source_sha256'].items():need(sha(p)==h,'original preparation source:'+p)
 for key in ['v2_FALSE_template','v3_FALSE_template']:
  a=ref(contract[key]);need(a['allow_score_analysis'] is False and a['allow_partial'] is False and a['plan']==contract['plan'] and a['manifest']==contract['manifest'] and a['protocol']==c['original_protocol'],'FALSE source bound')
  if key=='v3_FALSE_template':need(a['operational_repair']==contract['non_authorizing_v3_repair'] and a['actual_successor_immutable_state']['path'] is None and a['actual_successor_immutable_state']['sha256'] is None,'v3 pending state/repair')
 repair=ref(contract['non_authorizing_v3_repair']);need(repair['score_access_authorized'] is False and repair['this_is_not_a_ROOT_score_authority'] is True and repair['scientific_or_statistics_changes'] is False and repair['retained_plan']==contract['plan']==repair['actual_plan_at_repair'] and repair['retained_manifest']==contract['manifest'] and repair['retained_protocol']==c['original_protocol'] and repair['actual_first_read_journal_path_preserved']==plan['first_read_receipt_path'],'repair is no scientific/selection change')
 for p,h in dict(repair['retained_source_sha256'],**repair['active_source_sha256']).items():need(sha(p)==h,'v2/v3 source unchanged')
 family=plan['contract']['phase'];need(family['model_tag']=='pythia-1b' and family['rows']==24 and family['data_family']=='extended_v1','model/data scope')
 records=m['records'];need(len(records)==(48 if mode=='nll' else 24),'complete endpoint count');need({(x['group'],x['training_seed']) for x in records}=={(g,z) for g in ['B0','B1','B2','B3','B4','M','C','D'] for z in [11,22,33]},'24 physical cohort')
 need(all(x['model_tag']=='pythia-1b' and x['data_family']=='extended_v1' and x['is_training_alias'] is False for x in records),'physical trained rows retained')
 need(sum(x['inference_alias_of'] is not None for x in records)==(6 if mode=='nll' else 3),'true inference aliases only')
 for x in records:
  cpall.add(x['checkpoint']);need(x['kind'] in (['final','best'] if mode=='nll' else ['final']),'endpoint kind');need(x['full_training_committed_path']['target_prediction_tokens']==100000000,'fixed full training horizon')
  if x['kind']=='final':need(x['target_prediction_tokens_at_checkpoint']==100000000,'fixed final exposure')
  if x['inference_alias_of'] is not None:
   e=x['source_model_equivalence'];p=read(e['proof']);need(sha(e['proof'])==e['proof_sha256'],'alias proof SHA');pair=p['comparisons'][e['comparison_index']];need(pair['bitwise_model_state_equal'] is True and pair['endpoint']==x['kind'] and pair['seed']==x['training_seed'] and x['group']=='D','alias exact D/seed/endpoint')
   sides={y['run_id']:y for y in [pair['left'],pair['right']]};need(sides[x['run_id']]['checkpoint_sha256']==x['checkpoint_sha256'] and sides[x['run_id']]['model_state_fingerprint']==e['fingerprint'],'alias CP/fullFP bound')
 for p,h in plan['contract']['source_proof_files'].items():need(sha(p)==h,'contract frozen source')
 sp=next(p for p in plan['contract']['source_proof_files'] if p.endswith('_input_snapshot.json'));snapshot=read(sp)
 for p,r in snapshot['code'].items():need(sha(W/p)==r['sha256'],'science code unchanged');science[p]=r['sha256']
 need(plan['counts']==contract['counts'] and len(plan['closure_jobs'])==plan['closure_job_count']==(43 if mode=='nll' else 22),'closure43/22')
 need(contract['counts']['new_forward_jobs']==(43 if mode=='nll' else 21) and contract['counts']['imported_baselines']==(0 if mode=='nll' else 1),'new/import initial closure')
 f=next(x for x in future['phases'] if x['contract_id']==contract['contract_id']);check=f['check_argv_template'];analyze=f['analyze_argv_template'];need(check[check.index('--state')+1]==analyze[analyze.index('--state')+1]=='<ACTUAL_SUCCESSOR_IMMUTABLE_STATE_PATH>','same future state path')
 need(f['first_read_receipt_path']==contract['first_read_receipt_path']==plan['first_read_receipt_path'],'first-read namespace')
 checked.append(dict(contract_id=contract['contract_id'],plan=contract['plan'],READY=plan['ready'],manifest=contract['manifest'],counts=contract['counts'],FALSE_both_preserved=True))
need(len(cpall)==48 and cpall==set(pre['checkpoint_expected_and_current_stats']),'full48 CP allowlist')
for path,x in pre['checkpoint_expected_and_current_stats'].items():
 r=ref(x['historical_full_read_receipt']);need(cache['entries'][path]==x['historical_full_read_receipt'] and r['canonical_path']==str(P(path).resolve()) and r['sha256']==x['sha256'] and r['complete_to_eof'] is True and r['fd_before']==r['fd_after']==r['stat_before']==r['stat_after']==x['current_stat5']==stamp(path),'CP current five-stat historical binding')
 need(type(r['source_authority']['returncode']) is int and r['source_authority']['returncode']==0,'CP source rc0');kindbytes[r['source_authority']['endpoint']]+=x['current_stat5']['size']
need(sum(kindbytes.values())==c['qualified_CP_historical_file_size_bytes_not_current_read']==388533082112,'historical sizes')
for p,r in c['future_source_read_caveat'].items():need(stamp(p)==r['current_stat5'] and r['prepare_payload_read'] is False,'future base stat only')
result=dict(status='passed_independent_metadata_only_preparation_review',created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_COMPLETE=dict(path=str(S/'COMPLETE.json'),sha256=sha(S/'COMPLETE.json')),contracts=checked,qualified_CP_current_stats_checked=48,historical_CP_size_bytes_not_current_IO=kindbytes,actual_D_B4_pairs_per_endpoint=3,physical_training_rows_retained=24,science_code_sha256=science,all_FALSE_templates_remain_false=True,actual_successor_immutable_state_binding=c['actual_successor_immutable_state_binding'],future_base_fresh_bytes_not_authorized=2090701528,read_scope=dict(metadata_files=len(seen),metadata_read_bytes=sum(len(x[0]) for x in seen.values()),score_semantics_CP_base_TAR_ZIP_payload_bytes=0),review_region_wall_seconds=time.monotonic()-start,original_prepare_check_analyze_rerun=False,TRUE_authority_created=False,manual_whole_seconds=None)
(D/'review.json').open('x').write(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(dict(status=result['status'],contracts=[x['contract_id'] for x in checked],CP48=True,read_scope=result['read_scope'])))
