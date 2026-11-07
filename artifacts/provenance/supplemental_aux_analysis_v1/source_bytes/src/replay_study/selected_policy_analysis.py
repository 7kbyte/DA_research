"""Auxiliary A comparisons after each method's frozen validation-only lambda.

These are tuned-policy comparisons. They do not replace the primary same-lambda
contrasts, change selection, or create training replicates from evaluation rows.
"""
from collections import defaultdict
import hashlib
from pathlib import Path

from .analysis import METRICS,mean_std,paired_document_bootstrap
from .supplemental_analysis_guard import read

GROUPS=('B1','B2','B3','B4','M','C')
PAIRS=(('M','B4'),('C','B4'),('B2','B1'),('B4','B3'),('B3','B1'),('B4','B2'))
POLICY='Each method uses its own validation-selected shared lambda; method plus selected lambda varies. Not a fixed-lambda objective effect.'


def bind_selection(selection_proof,registry_proof,verifier):
    selection=verifier.proof(selection_proof);rows=verifier.proof(registry_proof)
    if (selection.get('phase')!='supplement_A_lambda_sweep' or selection.get('complete') is not True or
            selection.get('completed_conditions')!=126 or selection.get('expected_conditions')!=126 or
            selection.get('selection_uses_test') is not False or selection.get('test_scores_accessed') is not False or
            len(selection.get('all_candidate_scores',[]))!=42 or any(c.get('complete') is not True for c in selection['all_candidate_scores']) or
            len(selection.get('selections',[]))!=6 or len(rows)!=126):raise ValueError('Complete validation-only126-cell/42-candidate A selection required')
    by_id={r['id']:r for r in rows};sources={str(Path(p).resolve()):v for p,v in selection['sources_sha256'].items()}
    if len(by_id)!=126 or sources.get(str(Path(registry_proof['path']).resolve()))!=registry_proof['sha256']:raise ValueError('A selection bound a different grid registry')
    weights={r['weight'] for r in rows}
    if len(weights)!=7 or {r['group'] for r in rows}!=set(GROUPS):raise ValueError('A requires all seven lambdas and six methods')
    for group in GROUPS:
        actual={(r['weight'],r.get('training_seed',r.get('seed'))) for r in rows if r['group']==group}
        if actual!={(w,s) for w in weights for s in (11,22,33)}:raise ValueError('A method has an incomplete lambda/seed grid')
    candidates=selection['all_candidate_scores']
    if {(c['group'],c['weight']) for c in candidates}!={(g,w) for g in GROUPS for w in weights}:raise ValueError('A candidate scores do not cover the original42 cells')
    for c in candidates:
        ids={r['id'] for r in rows if r['group']==c['group'] and r['weight']==c['weight']}
        if set(c['condition_ids'])!=ids or c.get('n')!=3 or c.get('expected_n')!=3 or c.get('source')!='validation_final_fixed_exposure_only':raise ValueError('A candidate score lacks complete original three-seed validation identities')
    grid_sources={}
    for row in rows:
        if sources.get(str(Path(row['config_path']).resolve()))!=row['config_sha256']:raise ValueError('A selected grid config is not bound by the validation selection')
        for path,expected in ((row['config_path'],row['config_sha256']),(row['summary_path'],sources.get(str(Path(row['summary_path']).resolve())))):
            if not expected:raise ValueError('A choice lacks a frozen grid config/summary SHA')
            verifier.bind(path,expected);grid_sources[str(Path(path).resolve())]=expected
    chosen=[];choices=[]
    if {s['group'] for s in selection['selections']}!=set(GROUPS):raise ValueError('A selection does not cover each method exactly once')
    for item in selection['selections']:
        ids=item['condition_ids']
        if (item.get('selection_frozen') is not True or item.get('shared_across_seeds') is not True or item.get('test_scores_used') is not False or
                item.get('selected_by_best_checkpoint') is not False or len(ids)!=3 or len(set(ids))!=3 or any(i not in by_id for i in ids)):
            raise ValueError('A chosen method requires three actual IDs and one shared frozen lambda')
        selected=[by_id[i] for i in ids]
        if {r.get('training_seed',r.get('seed')) for r in selected}!={11,22,33} or any(r['group']!=item['group'] or r['weight']!=item['weight'] for r in selected):raise ValueError('A choice differs from its method/lambda/seed IDs')
        actual_ids=[r.get('actual_run_id') or r.get('alias_of') or r['id'] for r in selected]
        if item.get('actual_run_ids') is not None and set(item['actual_run_ids'])!=set(actual_ids):raise ValueError('A selected actual training IDs differ')
        choices.append({'group':item['group'],'weight':item['weight'],'condition_ids':ids,'actual_run_ids':actual_ids})
        chosen+=ids
    if len(set(chosen))!=18:raise ValueError('Exactly18 distinct validation-selected IDs required')
    return {'selection':selection_proof,'source_registry':registry_proof,'grid_ids':sorted(by_id),'selected_ids':sorted(chosen),'choices':choices,
        'grid_source_sha256':grid_sources,'selected_identity':{i:{'group':by_id[i]['group'],'weight':by_id[i]['weight'],'ratio':by_id[i]['ratio'],
            'training_seed':by_id[i].get('training_seed',by_id[i].get('seed')),'actual_run_id':by_id[i].get('actual_run_id') or by_id[i].get('alias_of') or i} for i in chosen},
        'selection_recomputed':False,'score_used_to_select':False}


def selected_rows(rows,selection):
    chosen=[];present=set()
    for row in rows:
        if row['run_id'] not in selection['selected_identity']:continue
        expected=selection['selected_identity'][row['run_id']]
        for key in ('group','weight','ratio','training_seed','actual_run_id'):
            if row.get(key)!=expected[key]:raise ValueError('Selected endpoint differs from frozen validation identity: '+key)
        chosen.append({**row,'validation_selection_sha256':selection['selection']['sha256'],'validation_grid_registry_sha256':selection['source_registry']['sha256'],
            'comparison_policy':POLICY,'additional_model_forward_calls':0});present.add(row['run_id'])
    if present!=set(selection['selected_ids']):raise ValueError('Selected18 IDs are not fully represented in the complete score table')
    identities=[(r['run_id'],r.get('checkpoint',r.get('checkpoint_kind')),r.get('task'),r.get('metric')) for r in chosen]
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate selected-policy endpoint identity')
    return chosen


def _pair_identity(a,b):
    for key in ('model_tag','data_family','baseline_group','training_seed','ratio'):
        if a[key]!=b[key]:raise ValueError('Selected-policy pair mixes model/data/baseline/seed/budget')
    if a.get('checkpoint',a.get('checkpoint_kind'))!=b.get('checkpoint',b.get('checkpoint_kind')):raise ValueError('Selected-policy pair mixes endpoints')
    return {'comparison':a['group']+'-'+b['group'],'comparison_policy':POLICY,'training_seed':a['training_seed'],
        'group_a':a['group'],'group_b':b['group'],'lambda_a':a['weight'],'lambda_b':b['weight'],'ratio_a':a['ratio'],'ratio_b':b['ratio'],
        'learning_rate_a':a['learning_rate'],'learning_rate_b':b['learning_rate'],'run_a':a['run_id'],'run_b':b['run_id'],
        'actual_run_id_a':a['actual_run_id'],'actual_run_id_b':b['actual_run_id'],'baseline_group':a['baseline_group'],
        'target_tokens_a':a.get('target_prediction_tokens',a.get('target_prediction_tokens_at_checkpoint')),
        'target_tokens_b':b.get('target_prediction_tokens',b.get('target_prediction_tokens_at_checkpoint')),
        'replay_tokens_a':a.get('replay_prediction_tokens',a.get('replay_prediction_tokens_at_checkpoint')),
        'replay_tokens_b':b.get('replay_prediction_tokens',b.get('replay_prediction_tokens_at_checkpoint')),
        'checkpoint_sha256_a':a.get('checkpoint_sha256'),'checkpoint_sha256_b':b.get('checkpoint_sha256')}


def _aggregate(rows,metrics,extra=()):
    grouped=defaultdict(list)
    keys=('comparison','lambda_a','lambda_b',*extra)
    for row in rows:grouped[tuple(row[k] for k in keys)].append(row)
    out=[]
    for key,current in sorted(grouped.items(),key=str):
        if {r['training_seed'] for r in current}!={11,22,33} or len(current)!=3:raise ValueError('Exactly three matched training seeds required per selected-policy contrast')
        for metric in metrics:out.append({**dict(zip(keys,key)),'metric':metric,**mean_std(r[metric] for r in current if r.get(metric) is not None),
            'training_seeds':'11,22,33','replicate_unit':'training_seed','comparison_policy':POLICY,
            'uncertainty_scope':'Sample SD across these trained checkpoints conditional on their shared validation-selected lambda; not uncertainty of repeating the selection pipeline. Document/question bootstrap is conditional and does not create training replicates'})
    return out


def nll_selected(bundle,selection,load_corpora,n_bootstrap=5000,seed=20261106):
    if bundle['complete'] is not True or len(bundle['endpoints'])!=126:raise ValueError('Full126 NLL endpoints required before the selected18 view')
    if {r['run_id'] for r in bundle['endpoints']}!=set(selection['grid_ids']):raise ValueError('Full NLL score grid differs from validation source IDs')
    rows=selected_rows(bundle['endpoints'],selection);lookup={(r['group'],r['training_seed']):r for r in rows};pairs=[];intervals=[]
    for left,right in PAIRS:
        for training_seed in (11,22,33):
            a,b=lookup[left,training_seed],lookup[right,training_seed];identity=_pair_identity(a,b)
            pairs.append({**identity,**{m:a[m]-b[m] if a.get(m) is not None and b.get(m) is not None else None for m in METRICS}})
            if n_bootstrap:
                ca,cb=load_corpora(a),load_corpora(b)
                for corpus in ('target','c4','wikitext'):
                    for field in ('sum_nll','sum_kl'):
                        ci=paired_document_bootstrap(ca[corpus],cb[corpus],field=field,n_bootstrap=n_bootstrap,seed=seed)
                        intervals.append({**identity,'corpus':corpus,'raw_sign':'left minus right '+field,
                            'gain_sign':'G_new left minus right reverses target NLL sign; other NLL/KL differences retain sign',**ci})
    return {'selected_endpoints':rows,'selected_policy_pairs':pairs,'selected_policy_statistics':_aggregate(pairs,METRICS),'selected_document_bootstrap':intervals,
        'lambda_curve_statistics':bundle['endpoint_statistics'],'metadata':{'selection':selection['selection'],'source_registry':selection['source_registry'],
            'comparison_policy':POLICY,'logical_selected_endpoints':18,'complete_grid_endpoints':126,'additional_model_forward_calls':0,
            'bootstrap_replicates':n_bootstrap,'bootstrap_seed':seed,'selection_recomputed':False}}


def behavior_selected(bundle,selection,load_predictions,n_bootstrap=5000,seed=20261106):
    from .behavioral_analysis import paired_binary_summary
    rows=selected_rows(bundle['endpoints'],selection);lookup={(r['group'],r['training_seed'],r['task'],r['metric']):r for r in rows}
    tasks=sorted({r['task'] for r in rows});pairs=[];intervals=[]
    if tasks!=['hellaswag','piqa'] or len(rows)!=18*len(tasks)*2:raise ValueError('Complete selected18 × both tasks × both-score behavioral inventory required')
    for left,right in PAIRS:
        for training_seed in (11,22,33):
            for task in tasks:
                for metric in ('acc_norm','acc'):
                    a,b=lookup[left,training_seed,task,metric],lookup[right,training_seed,task,metric];identity={**_pair_identity(a,b),'task':task,'metric':metric}
                    diff=a['accuracy']-b['accuracy'];pairs.append({**identity,'accuracy_left_minus_right':diff})
                    stream=(seed+int(hashlib.sha256(f'{left}|{right}|{training_seed}|{task}|{metric}'.encode()).hexdigest()[:8],16))%(2**32)
                    stats=paired_binary_summary(load_predictions(b),load_predictions(a),metric,n_bootstrap,stream)
                    if abs(stats['accuracy_delta']-diff)>1e-12:raise ValueError('Selected-policy endpoint accuracy differs from aligned prediction rows')
                    intervals.append({**identity,'accuracy_left_minus_right':diff,'ci_low':stats['delta_ci_low'],'ci_high':stats['delta_ci_high'],
                        'both_correct':stats['counts']['CC'],'right_correct_left_wrong':stats['counts']['CW'],'right_wrong_left_correct':stats['counts']['WC'],'both_wrong':stats['counts']['WW'],
                        'transition_direction':'right policy -> left policy; these are not initial-teacher forgetting counts',
                        'bootstrap_seed':stream,'bootstrap_replicates':n_bootstrap,'uncertainty_scope':stats['uncertainty_scope'],
                        'question_assumption':'IID/exchangeable questions; same-source activity/context dependence may invalidate population CI interpretation'})
    return {'selected_endpoints':rows,'selected_policy_pairs':pairs,'selected_policy_statistics':_aggregate(pairs,('accuracy_left_minus_right',),('task','metric')),
        'selected_question_bootstrap':intervals,'metadata':{'selection':selection['selection'],'source_registry':selection['source_registry'],
            'comparison_policy':POLICY,'additional_model_forward_calls':0,'bootstrap_replicates':n_bootstrap,'bootstrap_seed':seed}}


def plot_lambda_curves(statistics,output,endpoints):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .analysis import write_csv
    metrics=('nll_target','nll_c4','nll_wikitext','g_new','f_old_c4','f_old_wikitext','kl_target','kl_c4','kl_wikitext')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if len(endpoints)!=126 or len({(r['group'],r['weight'],r['training_seed']) for r in endpoints})!=126:raise ValueError('All126 individual trained checkpoints required for plotted seed points')
    plt.rcParams.update({'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'legend.fontsize':6,'xtick.labelsize':6,'ytick.labelsize':6})
    fig,axes=plt.subplots(3,3,figsize=(7,6.6),constrained_layout=True)
    colors=dict(zip(GROUPS,plt.get_cmap('tab10').colors));markers={11:'o',22:'s',33:'^'}
    for ax,metric in zip(axes.flat,metrics):
        for group in GROUPS:
            rows=sorted((r for r in statistics if r['group']==group and r['metric']==metric),key=lambda r:r['weight'])
            if len(rows)!=7 or any(r['n']!=3 or not r['complete'] for r in rows):raise ValueError('Seven complete lambda cells per method required for curves')
            points=[r for r in endpoints if r['group']==group]
            if {(r['weight'],r['training_seed']) for r in points}!={(r['weight'],s) for r in rows for s in markers}:raise ValueError('Every plotted lambda requires three matched training seeds')
            for s,marker in markers.items():
                p=[r for r in points if r['training_seed']==s]
                ax.scatter([r['weight'] for r in p],[r[metric] for r in p],s=8,marker=marker,color=colors[group],alpha=.45,linewidths=.3)
            ax.errorbar([r['weight'] for r in rows],[r['mean'] for r in rows],yerr=[r['std_ddof1'] for r in rows],fmt='-',lw=.8,capsize=1.5,color=colors[group],label=group)
        ax.set(xscale='log',xlabel='lambda',title=metric);ax.grid(alpha=.2)
    axes.flat[-1].legend(frameon=False,ncol=2)
    fig.suptitle('Supplement A: all126 final checkpoints; mean +/- sample SD (3 training seeds)',fontsize=8)
    axes.flat[0].legend(handles=[Line2D([],[],linestyle='',marker=m,color='gray',markersize=3,label=str(s)) for s,m in markers.items()],frameon=False,title='Seed',ncol=3,fontsize=5,title_fontsize=6)
    for suffix in ('png','pdf','svg'):fig.savefig(output/('full_seven_lambda_curves.'+suffix),dpi=200)
    plt.close(fig)
    write_csv(output/'full_seven_lambda_endpoints.csv',endpoints);write_csv(output/'full_seven_lambda_mean_sample_sd.csv',statistics)
