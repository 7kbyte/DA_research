"""Format complete reviewed C410 behavior; never run scoring or selection."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

W = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
R = W/'results/C410_behavior_independent_review_v1'
P = W/'results/C410_behavior_presented_v1'
N = W/'results/C410_nll_independent_review_v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda n: list(csv.DictReader((R/n).open()))
e, a, q, s = [read(n+'.csv') for n in ['endpoints', 'seed_aggregates', 'paired_seed_differences', 'paired_seed_aggregates']]
nll = list(csv.DictReader((N/'endpoint_statistics_independent.csv').open()))
groups = ['B0', 'B1', 'B2', 'B3', 'B4', 'D', 'M', 'C']
tasks, scores = ['piqa', 'hellaswag'], ['acc_norm', 'acc']
contrasts = [('M', 'B4'), ('C', 'B4'), ('B2', 'B1'), ('B4', 'B3'), ('D', 'B4')]
assert len(e) == 96 and len(q) == 60 and len(s) == 20
assert {r['target_prediction_tokens_at_checkpoint'] for r in e} == {'100000000'}
assert {r['checkpoint_kind'] for r in e} == {'final'}
assert {r['step'] for r in e} == {'3058'}
assert {r['model_tag'] for r in e} == {'pythia-410m'}
assert {r['data_family'] for r in e} == {'extended_v1'}
for f in R.glob('*.csv'):
    assert f.read_bytes() == (P/f.name).read_bytes()
    shutil.copyfile(f, OUT/f.name)

def fmt(x, signed=False):
    return format(100*float(x), '+.4f' if signed else '.4f')

def summary(g, task, score, quantity='accuracy_delta'):
    rows = [r for r in a if r['group']==g and r['task']==task and r['metric']==score and r['quantity']==quantity]
    assert len(rows)==1
    r = rows[0]
    return f"{fmt(r['mean'], quantity!='accuracy')} ± {fmt(r['std_ddof1'])}"

def pair(left, right, task, score):
    rows = [r for r in s if r['left_group']==left and r['right_group']==right and r['task']==task and r['metric']==score]
    assert len(rows)==1
    r = rows[0]
    return f"{fmt(r['mean'], True)} ± {fmt(r['std_ddof1'])}"

def table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |', '| '+' | '.join(['---']*len(header))+' |'] + ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

intro = ('本页覆盖 C410 extended_v1 的全部八组×三个训练 seed、24个 final 逻辑端点，未读取1B/2.8B分数。全部为 step3058、100,000,000目标预测token；除 B0外，b=.05、λ=.05、回放5,000,000预测token；B0的b/λ/回放为0。LR均3×10⁻⁵。'
         'D 与 B4 在各 seed 为已证明相等的模型、逻辑 alias，不增加训练或推断重复。只有 final 行为评价，没有补造 best。\n\n'
         '数值直接读取独立 reviewer 已核表，仅作百分比/百分点转换和四位显示。mean±SD 是三个固定训练 seed 的既有样本SD；端点Δ的5000题目bootstrap区间条件于固定checkpoint，并未重训练或校正 HellaSwag 同 context/activity 相关性，也未调整多重比较。\n\n')

text = '# C410 行为：全部24端点可读表\n\n'+intro
text += 'acc_norm 为主 score，acc 为辅助。CC/CW/WC/WW 按 initial→当前模型定义，CW 是原正确而新错误；条件遗忘率=CW/(CC+CW)，条件获得率=WC/(WC+WW)。百分比与pp单位不可混同。\n\n'
rows=[]
for g in groups:
    for seed in ['11','22','33']:
        r=next(r for r in e if r['group']==g and r['seed']==seed)
        rows.append([g,seed,r['ratio'],r['weight'],r['target_prediction_tokens_at_checkpoint'],r['replay_prediction_tokens_at_checkpoint'],'`'+r['run_id']+'`','`'+r['actual_run_id']+'`'])
text+=table(['方法','seed','b','λ','目标曝光','回放曝光','logical run','actual run'],rows)
for task in tasks:
    for score in scores:
        rows=[r for r in e if r['task']==task and r['metric']==score]
        text+=f'\n## {task} / {score}\n\nmatched initial accuracy {fmt(rows[0]["initial_accuracy"])}%；{rows[0]["items"]}题。\n\n'
        text+=table(['方法','accuracy mean±SD (%)','相对initial Δ mean±SD (pp)'],[[g,summary(g,task,score,'accuracy'),summary(g,task,score)] for g in groups])+'\n'
        rows=sorted(rows,key=lambda r:(groups.index(r['group']),int(r['seed'])))
        text+=table(['方法','seed','accuracy (%)','Δ (pp)','Δ条件95%区间 (pp)','CC','CW','WC','WW','条件遗忘率 (%)','条件获得率 (%)'],[[r['group'],r['seed'],fmt(r['accuracy']),fmt(r['accuracy_delta'],True),f"[{fmt(r['delta_ci_low'],True)}, {fmt(r['delta_ci_high'],True)}]",r['CC'],r['CW'],r['WC'],r['WW'],fmt(r['conditional_forgetting_rate']),fmt(r['conditional_acquisition_rate'])] for r in rows])
(OUT/'全部24端点可读表.md').write_text(text)

text='# C410 行为：全部五个冻结配对\n\n'+intro
text+='方向统一为左减右；M−B4、C−B4、B2−B1、B4−B3、D−B4是原行为方案的全部五对，不新增事后对照。每对先在同seed相减，再读取原三seed mean±样本SD。原行为配对表没有配对题目CI，本页不增算；前页的端点vsinitial区间不能冒充此处配对区间。\n\n'
for task in tasks:
    for score in scores:
        text+=f'## {task} / {score}\n\n'
        rows=[]
        for left,right in contrasts:
            vals=[]
            for seed in ['11','22','33']:
                r=next(r for r in q if r['left_group']==left and r['right_group']==right and r['task']==task and r['metric']==score and r['seed']==seed)
                vals.append(fmt(r['accuracy_left_minus_right'],True))
            rows.append([left+'−'+right,pair(left,right,task,score),*vals])
        text+=table(['对照','mean±SD (pp)','seed11 (pp)','seed22 (pp)','seed33 (pp)'],rows)+'\n'
text+='完整左右run ID、目标实际曝光、b/λ/LR和seed保存在原字节 paired_seed_differences.csv；没有把D alias或不同题目作为额外训练seed。\n'
(OUT/'全部五配对可读表.md').write_text(text)

text='# C410 行为：完整结果与解释边界\n\n'+intro
text+='## 两任务、两score全覆盖\n\n相对matched initial的变化，单位pp，±为三seed样本SD。\n\n'
text+=table(['方法','PiQA acc_norm','PiQA acc','HellaSwag acc_norm','HellaSwag acc'],[[g,summary(g,'piqa','acc_norm'),summary(g,'piqa','acc'),summary(g,'hellaswag','acc_norm'),summary(g,'hellaswag','acc')] for g in groups])
text+=('\n八组在两个任务上呈现不同方向：PiQA 两score的全部seed相对initial均下降，HellaSwag 两score的全部seed均提高，后者所有端点条件题目区间均高于零。PiQA 的 B1 acc_norm 三seed区间均低于零，B3 seed11/22、B4/D seed11、M seed11/22亦低于零，其余acc_norm区间跨零；辅助acc仅B1 seed11/22、B3 seed33、M seed22的区间低于零。不能把均值下降与每seed证据强度混为一谈。\n\n'
       'HellaSwag 的净提高也不意味着没有正确→错误。完整四状态与条件率均保留，题目发生的得失不能只由净accuracy判断。这里不将多端点的区间计数视作独立重复或多重比较校正后的发现。\n\n')
text+='## 五个预定同λ配对\n\n左减右，pp；不同任务/score不合并成新综合指标。\n\n'
text+=table(['对照','PiQA acc_norm','PiQA acc','HellaSwag acc_norm','HellaSwag acc'],[[l+'−'+r,pair(l,r,'piqa','acc_norm'),pair(l,r,'piqa','acc'),pair(l,r,'hellaswag','acc_norm'),pair(l,r,'hellaswag','acc')] for l,r in contrasts])
text+=('\nB2−B1 在 PiQA 两score三个seed全为正；HellaSwag acc_norm均值却为负，seed11略正而seed22/33为负，辅助acc三个seed全负。B4−B3 在 PiQA两score与HellaSwag acc_norm三seed全正，但HellaSwag acc均值接近零、seed22/33为负。这些结果体现具体任务和score差异，不能把objective差异概括为所有行为能力统一提升。\n\n'
       'M−B4 的PiQA主score均值为负、seed方向混合；HellaSwag主score均值接近零且seed方向混合，辅助acc均值略负。C−B4 的PiQA acc_norm三seed都正，但辅助acc seed11为负；HellaSwag两score均值为负且seed方向混合。因此M或C均没有跨任务、跨score、跨seed的稳定优势。D−B4所有点和SD严格为0，是已证明的模型相等控制；它不构成新的训练重复。端点bootstrap的有限Monte Carlo边界若因逻辑namespace不同略异，也不能解释为模型差异。\n\n')
text+='## 与已经完成的C410 NLL结果的限定关联\n\n以下仅复列既有独立NLL表的固定100M final token加权指标，单位nats/预测token；G越高越好、F越低越好。不是新统计或行为选参。\n\n'
nrows=[]
for g in groups:
    vals=[]
    for metric in ['g_new','f_old_c4','f_old_wikitext']:
        r=next(r for r in nll if r['checkpoint']=='final' and r['group']==g and r['metric']==metric)
        vals.append(f"{float(r['mean']):.4f} ± {float(r['std_ddof1']):.4f}")
    nrows.append([g,*vals])
text+=table(['方法','数学G↑','C4 F↓','WikiText F↓'],nrows)
text+=('\nC4/WikiText的NLL保留与选择题accuracy描述不同对象。例如B2相对B1显著较小的旧域NLL退化，在本行为表并未转化为更高的HellaSwag均值；其PiQA两score却更高。M/C的旧域NLL与行为差异也未形成统一排序。这里只说明指标不等价，不把二者关联当作机制或因果证明；数学NLL也不是数学推理能力。\n\n'
       'C410与主实验共享410M基座、训练数据来源及生成池/库，扩展数据包含主训练前缀，评估也不是全新独立总体复制。除了horizon，还改变不同目标位置数量、批量与学习率日程及配套heldout，不能从两阶段差异推出纯训练时长效应，也不能把两组seed拼成六次独立训练重复。C410仍使用预先冻结λ=.05，没有据A选择或当前行为分数更换λ。本次只展示本组final行为，不把NLL的best观察扩展成未评价的best行为。\n\n'
       'NLL全端点、两种文档权重、final/best实际曝光与六个原NLL配对详见 [C410既有完整结果](../../docs/C410_100M_NLL完整结果与正文素材.md)；行为只保留自身冻结的五个配对。\n\n')
text+=('## 审查与费用边界\n\n'
       '原v3成功分析、实际ROOT authority、同一不可变state、operational repair与不可重置first-read journal均由批准reviewer核验；随后独立重算7336项数值/身份，最大绝对误差2.22×10⁻¹⁶。24逻辑端点的两任务两score形成96行；D/B4别名不增加独立模型数。题目区间条件于固定checkpoint及独立/可交换题目假设，HellaSwag共享activity/context相关性未修正，不替代训练seed或总体区间，全部比较未作多重性调整。\n\n'
       'reviewer fresh读取444,908,685B为授权C410行为分数/逐题/小来源，CP/base/TAR/ZIP读取0，无模型forward。展示只复制四份已核小CSV并绘四张PNG/PDF/SVG，没有新推理/区间/选模。两次完整Popen外层分别10.492894s（user10.555002/sys4.338057）和3.714245s（user3.925327/sys1.480114）；adjacent helper timers嵌套不能再加。该wrapper的RUSAGE_CHILDREN maxrss是累计高水位：review时746,762,240B，presentation条目的同值仅为累计上下文，不能认作后者独立峰值。人工读表/写文整体未知。\n\n'
       '完整数值见[24端点全表](全部24端点可读表.md)、[五配对全表](全部五配对可读表.md)，图见[展示报告](../C410_behavior_presented_v1/REPORT.md)，全部来源SHA见同目录COMPLETE。未改原科学输出、当前报告、λ、方法或其他模型分析。\n')
(OUT/'讨论.md').write_text(text)
orig=W/'results/supplemental_analysis_v3/extended_pythia_410m_100m_behavior/COMPLETE.json'
o=json.loads(orig.read_text())
proofs=[orig,R/'COMPLETE.json',P/'COMPLETE.json',Path(o['authority']['path']),Path(o['first_read_receipt']['path']),Path(o['operational_repair']['path']),W/'artifacts/provenance/C_pending_phase_analysis_execution_v1/extended_pythia_410m_100m_behavior/state_capture/actual_successor_state_immutable.json',N/'endpoint_statistics_independent.csv',N/'independent_review_receipt.json',W/'artifacts/provenance/C410_behavior_review_actual_v1/whole_parent_timing.json',W/'artifacts/provenance/C410_behavior_presentation_actual_v1/whole_parent_timing.json']
files=[p for p in OUT.iterdir() if p.is_file()]
result={'complete':True,'phase':'extended_pythia_410m_100m_behavior','model':'pythia-410m','logical_final_endpoints':24,'task_score_rows':96,'seed_pair_rows':60,'aggregate_pair_rows':20,'all_five_frozen_pairs_retained':True,'new_inference_or_statistics':False,'all_four_csv_copies_byte_equal_presenter':True,'final_only_behavior':True,'other_model_scores_accessed':False,'source_proofs':{str(p):sha(p) for p in proofs},'outputs':{str(p):sha(p) for p in files},'manual_overall_wall_seconds':None,'rss_scope':'RUSAGE_CHILDREN maxrss is cumulative across reviewer and presenter children in the same outer wrapper; presenter independent peak unknown.'}
(OUT/'COMPLETE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'complete':True,'receipt_sha256':sha(OUT/'COMPLETE.json'),'discussion_sha256':sha(OUT/'讨论.md')},ensure_ascii=False))
