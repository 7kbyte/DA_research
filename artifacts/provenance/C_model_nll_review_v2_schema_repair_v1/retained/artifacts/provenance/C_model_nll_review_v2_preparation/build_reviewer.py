"""Build independent v2 from retained numerical implementation, never edit v1."""
from pathlib import Path

base=Path(__file__).resolve().parent
source=(base/'retained/scripts/review_C_nll_results.py').read_text()
source=source.replace('Independent chunk-based review of one explicitly unsealed, COMPLETE C model.',
                      'Independent review of an explicitly ROOT-unsealed 1B/2.8B COMPLETE C phase.')
cut=source.index('def run(')
prefix=source[:cut]
# Replace the old I/O helpers completely; no shadowed permissive read entry.
prefix=prefix[:prefix.index('def sha(path):')]+prefix[prefix.index('def write_json'):]
prefix=prefix.replace('def plot(endpoints,pairs,out):','def plot(endpoints,pairs,out,model,profile):')
prefix=prefix.replace("groups=[g for g in GROUPS", "groups=[g for g in profile['groups']")
prefix=prefix.replace("[p for p in PAIRS", "[p for p in profile['pairs']")
prefix=prefix.replace("f'C 410M | {weighting} | '+('six frozen paired contrasts' if paired else 'all methods and checkpoints')", "f'C {model} | {weighting} | '+(str(len(profile['pairs']))+' frozen paired contrasts' if paired else 'all methods and checkpoints')")
old="""            fig.text(.5,.012,'Circle/square/triangle: seeds 11/22/33; diamond +/- sample SD (n=3). Row-shared scales.\\nFinal: 100M target; best: actual 35.026M/40.030M/55.041M. b=lambda=.05 except B0.\\n'+('Differences = left minus right; G positive / F negative favors left.' if paired else 'D and B4 share verified states; both logical methods retained; initial = dashed zero.'),ha='center',fontsize=7.2)"""
new="""            best_exposure=sorted({int(r['target_prediction_tokens']) for r in endpoints if r['checkpoint']=='best'})
            best_range=f'{min(best_exposure)/1e6:.3f}--{max(best_exposure)/1e6:.3f}M'
            fig.text(.5,.012,'Circle/square/triangle: seeds 11/22/33; diamond +/- sample SD (n=3). Row-shared scales.\\n'+f'Final: {profile["horizon"]/1e6:g}M target; best: actual range {best_range}; exact exposure in CSV.\\n'+('Differences = left minus right; G positive / F negative favors left.' if paired else 'Every frozen logical method retained; aliases require complete-state proof; initial = dashed zero.'),ha='center',fontsize=7.2)"""
assert old in prefix;prefix=prefix.replace(old,new)
prefix += '\n'+(base/'gates_source.py').read_text()+'\n'
run=source[cut:source.index('def main():')]
start=run.index('    start=time.perf_counter()')
end=run.index('    endpoints=[];')
run=run[:start]+'''    start=time.perf_counter();complete_path=Path(args.complete);out=Path(args.output)
    gate=gate_inputs(args);complete=gate['complete'];plan=gate['plan'];ready=gate['ready']
    manifest=gate['manifest'];group=gate['group'];profile=gate['profile'];groups=profile['groups'];pairs=profile['pairs']
    n=len(groups)*len(SEEDS);physical_count=len(manifest['jobs']);alias_count=gate['aliases']
'''+run[end:]
run=run.replace('def run(complete_path, expected_complete, expected_authority, output):','def run(args):')
run=run.replace("analysis['completed_conditions']==analysis['expected_conditions']==24", "analysis['completed_conditions']==analysis['expected_conditions']==n")
run=run.replace("        require(analysis['bootstrap_replicates']", "        require(analysis['identity']=={'role':'extended','phase_name':profile['phase'],'model_tag':args.model,'data_family':'extended_v1','checkpoint':kind},'Foreign model analysis identity')\n        require(exact_proof(analysis['whole_phase_first_read_receipt'],complete['first_read_receipt']) and exact_proof(analysis['whole_phase_readiness'],complete['readiness']),'Analysis output journal/readiness differs')\n        require(analysis['bootstrap_replicates']")
run=run.replace("validate_grid(endpoints,GROUPS);require(len(old_pairs)==36,'All six paired contrasts and seeds required')", "validate_grid(endpoints,groups);validate_pairs(old_pairs,groups,pairs)")
run=run.replace("if p not in sources:require(sha(p)==d,'Changed analysis source '+p);sources[p]=d", "if p not in sources:bind({'path':p,'sha256':d},opaque=True);sources[p]=d")
run=run.replace('for g in GROUPS:', 'for g in groups:').replace('for comparison in PAIRS:', 'for comparison in pairs:')
run=run.replace(",'tied_D_B4_states':True", ",'rank_is_descriptive_no_new_inference':True")
run=run.replace("a=bykey[p['checkpoint'],p['run_a']];b=bykey[p['checkpoint'],p['run_b']]", "a=bykey[p['checkpoint'],p['run_a']];b=bykey[p['checkpoint'],p['run_b']];require(p['comparison']==a['group']+'-'+b['group'],'Pair label/run identity differs')")
run=run.replace("require(record['kind']==r['checkpoint'] and record['run_id']==r['run_id'],'Endpoint manifest identity differs')", "require(record['kind']==r['checkpoint'] and record['run_id']==r['run_id'] and record['group']==r['group'] and str(record['training_seed'])==r['training_seed'] and record['actual_run_id']==r['actual_run_id'],'Endpoint manifest identity differs')")
run=run.replace("    endpoint_ci=[];paired_ci=[]", "    endpoint_ci=[];paired_ci=[];seen_ci=set()")
run=run.replace("c=old['corpus'];metric=METRICS", "c=old['corpus'];ci_key=(kind,is_pair,old.get('run_a',old.get('run_id')),old.get('run_b'),c);require(ci_key not in seen_ci,'Duplicate conditional CI row');seen_ci.add(ci_key);metric=METRICS")
run=run.replace("identity={k:old[k] for k in ('comparison','training_seed','run_a','run_b','target_tokens_a','target_tokens_b','equal_checkpoint_exposure')}", "require(old['comparison'] in pairs and old['comparison']==a['group']+'-'+b['group'] and int(old['training_seed'])==int(a['training_seed'])==int(b['training_seed']),'CI comparison/seed identity differs')\n                    require(int(old['target_tokens_a'])==int(a['target_prediction_tokens']) and int(old['target_tokens_b'])==int(b['target_prediction_tokens']) and old['equal_checkpoint_exposure']==(int(a['target_prediction_tokens'])==int(b['target_prediction_tokens'])),'CI actual paired exposure differs')\n                    identity={k:old[k] for k in ('comparison','training_seed','run_a','run_b','target_tokens_a','target_tokens_b','equal_checkpoint_exposure')}")
run=run.replace("identity={k:old[k] for k in ('run_id','training_seed','target_prediction_tokens','replay_prediction_tokens')};identity['group']=a['group']", "require(all(int(old[k])==int(a[k]) for k in ('training_seed','target_prediction_tokens','replay_prediction_tokens')),'Endpoint CI actual identity/exposure differs')\n                    identity={k:old[k] for k in ('run_id','training_seed','target_prediction_tokens','replay_prediction_tokens')};identity['group']=a['group']")
run=run.replace("rr=[r for r in paired if r['checkpoint']==kind and r['comparison']==comparison];vals=[r[metric] for r in rr];ss=stats(vals)", "rr=sorted([r for r in paired if r['checkpoint']==kind and r['comparison']==comparison],key=lambda r:int(r['training_seed']));vals=[r[metric] for r in rr];ss=stats(vals)")
run=run.replace("require(len(endpoint_ci)==288 and len(paired_ci)==216", "require(len(endpoint_ci)==2*n*6 and len(paired_ci)==2*len(pairs)*len(SEEDS)*6")
run=run.replace("require(len(training)==24,'Count exactly 24 physical training runs')", "require(len(training)==n,'Count every real full training once across final/best')")
run=run.replace("state=read(Path(plan['closure_jobs'][0]['definition']['argv'][1]).parents[1]/'logs/supplemental_job_state.json');inferences=[];alias_rows=[]", "state=gate['state'];inferences=[];alias_rows=[]")
run=run.replace("require(len(inferences)==43 and len(alias_rows)==6", "require(len(inferences)==physical_count and len(alias_rows)==alias_count")
run=run.replace("'physical_training_runs':24", "'physical_training_runs':n").replace("'physical_inferences':43", "'physical_inferences':physical_count").replace("'physical_adapted_inferences':42", "'physical_adapted_inferences':physical_count-1").replace("'zero_forward_aliases':6", "'zero_forward_aliases':alias_count")
timerstart=run.index("    timers=Path(")
timerend=run.index("    require(not out.exists()",timerstart)
run=run[:timerstart]+"    costs['ROOT_check_whole_child']=gate['timers']['check'];costs['ROOT_analyze_whole_child']=gate['timers']['analyze']\n"+run[timerend:]
run=run.replace("write_json(out/'cost_accounting.json',costs);plot(rebuilt,paired,out)", "write_json(out/'cost_accounting.json',costs);plot(rebuilt,paired,out,args.model,profile)")
run=run.replace("'sha256':expected_complete", "'sha256':args.complete_sha256")
run=run.replace("'endpoint_records':48", "'endpoint_records':2*n").replace("'paired_records':36", "'paired_records':2*len(pairs)*len(SEEDS)").replace("'endpoint_conditional_intervals':288", "'endpoint_conditional_intervals':2*n*6").replace("'paired_conditional_intervals':216", "'paired_conditional_intervals':2*len(pairs)*len(SEEDS)*6")
run=run.replace("'other_models_or_behavior_read':False", "'other_models_or_behavior_read':False,'model':args.model,'physical_inferences':physical_count,'zero_forward_aliases':alias_count,'bound_metadata_sha256':dict(BOUND),'actual_scheduler_queue':{'path':canonical(args.queue),'sha256':args.queue_sha256},'actual_scheduler_state':{'path':canonical(args.state),'sha256':args.state_sha256},'IO':{'serialized_bytes_read':IO['read_bytes'],'paths':sorted(IO['paths']),'semantic_json_paths':sorted(IO['semantic_json_paths']),'checkpoint_base_archive_payload_reads':0}")
run=run.replace("    write_json(out/'independent_review_receipt.json'", "    for p,d in BOUND.items():require(sha(p)==d,'Bound metadata changed during review: '+p)\n    write_json(out/'independent_review_receipt.json'")
# Explicit empty-table schema, preserving all existing nonempty CSV columns.
prefix=prefix.replace("def write_csv(path, values):", "def write_csv(path, values):")
prefix=prefix.replace("w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)", "w=csv.DictWriter(f,fieldnames=list(values[0]) if values else ['record_id','run_id','checkpoint','canonical_job_id','inference_subprocess_wall_seconds','source_model_equivalence']);w.writeheader();w.writerows(values)")
main='''def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=tuple(PROFILES))
    for name in ('complete','queue','state','check-timing','analyze-timing'):
        p.add_argument('--'+name,required=True,type=Path)
        p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--authority-sha256',required=True)
    p.add_argument('--output',required=True,type=Path)
    run(p.parse_args())
if __name__=='__main__':main()
'''
# The retained numerical loops use n for token arrays and token counts.
# Keep scientific grid cardinality in a separate name throughout the new run.
run=run.replace('n=len(groups)*len(SEEDS)','run_count=len(groups)*len(SEEDS)')
run=run.replace("analysis['expected_conditions']==n", "analysis['expected_conditions']==run_count")
run=run.replace('len(training)==n','len(training)==run_count')
run=run.replace('2*n*6','2*run_count*6').replace("'endpoint_records':2*n", "'endpoint_records':2*run_count")
run=run.replace("'physical_training_runs':n", "'physical_training_runs':run_count")
(base.parents[2]/'scripts/review_C_model_nll_results_v2.py').write_text(prefix+run+main)
