"""Check published report display values against already accepted small CSVs."""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
LINES = (OUT / '研究报告_5e72749.md').read_text().splitlines()
proofs, checks = [], []


def proof(path):
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def csvread(rel):
    path = ROOT / rel
    pr = proof(path)
    complete_path = path.parent / 'COMPLETE.json'
    complete = json.loads(complete_path.read_text())
    assert complete['complete'] is True
    assert complete['review_output_sha256'][str(path)] == pr['sha256']
    proofs.extend([pr, proof(complete_path)])
    return list(csv.DictReader(path.open()))


def query(rows, **where):
    hits = [r for r in rows if all((float(r[k]) == v if isinstance(v, (float, int)) else r[k] == v) for k, v in where.items())]
    assert len(hits) == 1, (where, len(hits))
    return hits[0]


def check(name, ok, evidence, line=None):
    checks.append({'name': name, 'passed': bool(ok), 'report_line': line, 'evidence': evidence})


def display(row, shown, precision, line, name):
    parts = shown.replace(' ', '').replace('−', '-').split('±')
    expected = [f'{100*float(row[k]):.{precision}f}' for k in ('mean', 'std_ddof1')]
    normalize = lambda x: float(x)
    check(name, list(map(normalize, parts)) == list(map(normalize, expected)),
          {'shown': shown, 'expected_percent_or_pp': expected}, line)


P = 'results/primary_behavior_independent_review_v1/'
A = 'results/supplement_A_behavior_independent_review_v1/'
ps, pp, pe, pd = [csvread(P + n) for n in ('seed_aggregates.csv', 'paired_seed_aggregates.csv', 'endpoints.csv', 'paired_seed_differences.csv')]
ass, ap, ae, aq, ad = [csvread(A + n) for n in ('seed_aggregates.csv', 'selected_policy_score_resolved_statistics.csv', 'endpoints.csv', 'selected_question_intervals.csv', 'selected_policy_pairs.csv')]

# All table cells, including both checkpoints, tasks and scores. No new statistics.
for line in range(212, 224):
    c = [x.strip() for x in LINES[line-1].strip('|').split('|')]
    group, ratio = c[0], float(c[1])
    for task, kind, shown in [('piqa','final',c[2]),('piqa','best',c[3]),('hellaswag','final',c[4]),('hellaswag','best',c[5])]:
        row = query(ps, group=group, ratio=ratio, checkpoint_kind=kind, task=task, metric='acc_norm', quantity='accuracy')
        display(row, shown, 3, line, 'primary_endpoint_table')
for line in range(284, 290):
    c = [x.strip() for x in LINES[line-1].strip('|').split('|')]
    for task, score, shown in [('piqa','acc_norm',c[2]),('piqa','acc',c[3]),('hellaswag','acc_norm',c[4]),('hellaswag','acc',c[5])]:
        row = query(ass, group=c[0], weight=float(c[1]), task=task, metric=score, quantity='accuracy')
        display(row, shown, 4, line, 'A_endpoint_table')
for line in range(297, 303):
    c = [x.strip() for x in LINES[line-1].strip('|').split('|')]
    comparison = c[0].replace('−','-')
    for task, score, shown in [('piqa','acc_norm',c[2]),('piqa','acc',c[3]),('hellaswag','acc_norm',c[4]),('hellaswag','acc',c[5])]:
        row = query(ap, comparison=comparison, task=task, score=score, quantity='accuracy_left_minus_right')
        display(row, shown, 4, line, 'A_score_resolved_pair_table')
        check('A_pair_lambda', [float(x) for x in c[1].split('/')] == [float(row['lambda_a']),float(row['lambda_b'])], row, line)

# Every paired mean±SD quoted in primary prose (pp).
cases = [
 (235,'final','B2','B1','piqa','acc_norm',.05,'−.2358±.1571'),
 (235,'final','B2','B1','hellaswag','acc_norm',.05,'−.8133±.2456'),
 (235,'final','B4','B3','piqa','acc_norm',.05,'+.1451±.5584'),
 (235,'final','B4','B3','hellaswag','acc_norm',.05,'−.2589±.2058'),
 (235,'final','B4','B3','piqa','acc',.05,'−.2720±.3808'),
 (235,'final','B4','B3','hellaswag','acc',.05,'−.0332±.2735'),
 (235,'best','B2','B1','piqa','acc_norm',.05,'+.3446±.2199'),
 (235,'best','B2','B1','piqa','acc',.05,'−.9612±.4876'),
 (237,'final','M','B4','piqa','acc_norm',.05,'+.1269±.4084'),
 (237,'final','M','B4','hellaswag','acc_norm',.05,'−.2622±.1864'),
 (237,'final','C','B4','piqa','acc_norm',.05,'+.2720±.8446'),
 (237,'final','C','B4','hellaswag','acc_norm',.05,'+.3120±.2616'),
 (237,'best','C','B4','piqa','acc_norm',.05,'+.4171±.1749'),
 (237,'best','C','B4','hellaswag','acc_norm',.05,'+.2788±.1340'),
 (237,'final','M','B4','hellaswag','acc_norm',.01,'+.1328±.1685'),
 (237,'final','M','B4','hellaswag','acc_norm',.1,'−.2390±.2939'),
]
for line, kind, left, right, task, score, ratio, shown in cases:
    assert shown in LINES[line-1]
    row = query(pp, checkpoint_kind=kind, left_group=left, right_group=right, task=task, metric=score, ratio=ratio, quantity='accuracy_left_minus_right')
    display(row, shown, 4, line, 'primary_paired_prose')

inside = lambda r: float(r['delta_ci_low']) <= 0 <= float(r['delta_ci_high'])
check('primary_PiQA_all144_endpoint_score_CIs_include_zero', len([r for r in pe if r['task']=='piqa']) == 144 and all(inside(r) for r in pe if r['task']=='piqa'), {'rows':144},229)
hf = [r for r in pe if r['task']=='hellaswag' and r['checkpoint_kind']=='final']
check('primary_HellaSwag_final_all72_positive_delta_and_lowerCI',len(hf)==72 and all(float(r['accuracy_delta'])>0 and float(r['delta_ci_low'])>0 for r in hf),{'rows':len(hf)},229)
best = {(r['group'],float(r['ratio']),int(r['training_seed'])):int(r['target_prediction_tokens_at_checkpoint']) for r in pe if r['checkpoint_kind']=='best'}
exceptions = {k:v for k,v in best.items() if v != 3270400}
check('best_exposure_exceptions_exact', exceptions == {('B2',.05,11):1635200,('B0',0.,22):4905600,('M',.1,22):4905600},str(exceptions),208)
for task, score, expected in [('piqa','acc_norm','66.9206'),('piqa','acc','66.8118'),('hellaswag','acc_norm','40.3605'),('hellaswag','acc','33.8578')]:
    values={float(r['initial_accuracy']) for r in pe+ae if r['task']==task and r['metric']==score}
    check('matched_initial_' + task+'_'+score,len(values)==1 and f'{100*next(iter(values)):.4f}'==expected,{'values':list(values),'display':expected},206)
for kind,task,cw,wc,shown in [('final','hellaswag',345,503,8.5122),('best','hellaswag',361,361,None),('final','piqa',91,85,None)]:
    row=query(pe,group='B4',ratio=.05,training_seed=11,checkpoint_kind=kind,task=task,metric='acc_norm')
    check('B4_s11_transition_'+kind+'_'+task,int(row['CW'])==cw and int(row['WC'])==wc and (shown is None or round(100*float(row['conditional_forgetting_rate']),4)==shown),row,243)
check('primary_final_B2minusB1_all_four_means_negative', all(float(r['mean'])<0 for r in pp if r['checkpoint_kind']=='final' and r['left_group']=='B2' and r['right_group']=='B1' and r['quantity']=='accuracy_left_minus_right'),{},235)
for kind, group, task in [('best','C','piqa'),('best','C','hellaswag')]:
    rr=[r for r in pd if r['checkpoint_kind']==kind and r['left_group']==group and r['right_group']=='B4' and r['task']==task and r['metric']=='acc_norm']
    check('primary_best_CminusB4_positive_'+task,len(rr)==3 and all(float(r['accuracy_left_minus_right'])>0 for r in rr),rr,237)
check('primary_DminusB4_all_point_differences_zero',all(float(r['accuracy_left_minus_right'])==0 for r in pd if r['left_group']=='D'),{},241)
check('A_scope_18_final_3seeds_6methods',len(ae)==72 and {r['checkpoint_kind'] for r in ae}=={'final'} and {int(r['training_seed']) for r in ae}=={11,22,33} and {r['target_prediction_tokens_at_checkpoint'] for r in ae}=={'10000000'}, {},280)
check('A_PiQA_all36_CIs_include_zero',all(inside(r) for r in ae if r['task']=='piqa'),{},291)
for group in ['B1','B3']:
    rr=[r for r in ae if r['group']==group and r['task']=='hellaswag']
    check('A_HellaSwag_'+group+'_all_scores_seeds_positive_and_lowerCI',len(rr)==6 and all(float(r['accuracy_delta'])>0 and float(r['delta_ci_low'])>0 for r in rr),rr,291)
for group, expected in [('B2',-.4514),('B4',-.6074),('M',-.6141),('B1',1.9153),('B3',1.4605),('C',.5079)]:
    row=query(ass,group=group,task='hellaswag',metric='acc_norm',quantity='accuracy_delta')
    rr=[r for r in ae if r['group']==group and r['task']=='hellaswag' and r['metric']=='acc_norm']
    check('A_HellaSwag_delta_mean_'+group,round(100*float(row['mean']),4)==expected and all(math.copysign(1,float(r['accuracy_delta']))==math.copysign(1,expected) for r in rr),row,291)
c33=query(ae,group='C',training_seed=33,task='hellaswag',metric='acc_norm')
check('A_C_s33_HellaSwag_main_CI_crosses_zero',inside(c33),c33,291)
same_lambda = {r['comparison'] for r in ad if float(r['lambda_a'])==float(r['lambda_b'])}
check('A_only_same_lambda_pair_MminusB4',same_lambda=={'M-B4'},str(same_lambda),304)
for comp in ['B2-B1','B4-B3']:
    rr=[r for r in aq if r['comparison']==comp and r['task']=='hellaswag']
    check('A_'+comp+'_HellaSwag_all_scores_negative_upperCI',len(rr)==6 and all(float(r['accuracy_left_minus_right'])<0 and float(r['ci_high'])<0 for r in rr),rr,304)
rr=[r for r in ad if r['comparison']=='C-B4']
check('A_CminusB4_all12_positive',len(rr)==12 and all(float(r['accuracy_left_minus_right'])>0 for r in rr),rr,304)
for task in ['piqa','hellaswag']:
    rr=[r for r in aq if r['comparison']=='M-B4' and r['task']==task and r['metric']=='acc_norm']
    check('A_MminusB4_main_CIs_cross_zero_'+task,len(rr)==3 and all(float(r['ci_low'])<=0<=float(r['ci_high']) for r in rr),rr,304)
rr=[r for r in ad if r['comparison']=='M-B4' and r['metric']=='acc_norm' and r['task']=='piqa']
check('A_MminusB4_PiQA_main_all_positive',all(float(r['accuracy_left_minus_right'])>0 for r in rr),rr,304)
checks_path=OUT/'machine_audit.json'
result={'status':'passed' if all(x['passed'] for x in checks) else 'failed','report_commit':'5e72749',
    'snapshot':proof(OUT/'研究报告_5e72749.md'),'source_proofs':list({x['path']:x for x in proofs}.values()),
    'checks':checks,'checks_count':len(checks),'failed_checks':[x for x in checks if not x['passed']],
    'new_statistics_or_CIs':False,'A_uses_score_resolved_table':True,'sealed_other_phase_scores_read':False,
    'source':proof(Path(__file__))}
with checks_path.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ('status','checks_count','failed_checks')}))
