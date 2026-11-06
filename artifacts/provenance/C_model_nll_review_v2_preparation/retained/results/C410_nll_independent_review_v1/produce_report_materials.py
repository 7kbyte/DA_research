#!/usr/bin/env python3
"""Format only already independently reviewed C410 CSVs; no experiment input reads."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
GROUPS = ('B0', 'B1', 'B2', 'B3', 'B4', 'D', 'M', 'C')
PAIRS = ('M-B4', 'C-B4', 'B2-B1', 'B4-B3', 'B3-B1', 'B4-B2')
METRICS = ('g_new', 'f_old_c4', 'f_old_wikitext')

def read_csv(name):
    with (OUT / name).open(newline='') as f:
        return list(csv.DictReader(f))

def proof(path):
    p = Path(path)
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}

def write_new(path, text):
    with Path(path).open('x') as f:
        f.write(text)

def table(rows, headers):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + ['| ' + ' | '.join(r) + ' |' for r in rows])

def endpoint_table(stats, checkpoint, suffix=''):
    idx = {(r['checkpoint'], r['group'], r['metric']): r for r in stats}
    rows = []
    for g in GROUPS:
        vals = [idx[checkpoint, g, m + suffix] for m in METRICS]
        rows.append([g] + [f"{float(v['mean']):.6f} ± {float(v['std_ddof1']):.6f}" for v in vals])
    return table(rows, ['方法', 'G_new ↑', 'F_C4 ↓', 'F_Wiki ↓'])

def pair_table(stats, checkpoint, suffix=''):
    idx = {(r['checkpoint'], r['comparison'], r['metric']): r for r in stats}
    rows = []
    for p in PAIRS:
        vals = [idx[checkpoint, p, m + suffix] for m in METRICS]
        rows.append([p] + [f"{float(v['mean']):+.6f} ± {float(v['std_ddof1']):.6f} ({v['favorable_seeds']}/3)" for v in vals])
    return table(rows, ['同seed：左−右', 'ΔG ↑；有利seed', 'ΔF_C4 ↓；有利seed', 'ΔF_Wiki ↓；有利seed'])

def main():
    stats = read_csv('endpoint_statistics_independent.csv')
    pairs = read_csv('paired_statistics_independent.csv')
    costs = json.loads((OUT / 'cost_accounting.json').read_text())
    receipt = json.loads((OUT / 'independent_review_receipt.json').read_text())
    assert len(stats) == 96 and len(pairs) == 72 and receipt['status'] == 'passed'
    report = ROOT / 'docs/C410_100M_NLL完整结果与正文素材.md'
    insertion = ROOT / 'docs/C410_NLL主报告插入候选.md'
    body = '''# C410／100M：完整NLL结果、独立核验及正文素材

只读取ROOT已完整解封的C410 NLL。48逻辑端点（24 final＋24 best）共同通过门禁，43物理推理（42适应端点＋1 matched initial），另有6个D/B4等价端点复用、增加0次forward。1B、2.8B及全部行为结果未读。本材料不修改主报告，不按测试结果改变方法、λ、曝光或选择规则。

## 可支持的有限结论

1. 在该固定配置中，两种来源的KL相对CE均减少旧域遗忘，但目标增益需要区分端点。C4来源B2−B1在final与best的三项token指标均为3/3个seed有利。生成来源B4−B3的final ΔG为−0.006843±0.002144，best则为+0.014413±0.002290；两个端点的C4/Wiki遗忘都减少且各3/3个seed方向一致。不能把best的优势写成同预算final的结论。
2. M/C没有跨域、跨端点的一致优势。final的M−B4平均ΔG=+0.001859、ΔF_C4=+0.000793、ΔF_Wiki=−0.007720；best为+0.000236、+0.004266、−0.002412。final C−B4对应+0.001282、+0.008217、−0.012552；best为+0.000102、+0.008096、−0.000809。C4正差表示更差；平均Wiki改善也不代表每个seed均改善。全部个体和方向数保留于后表。
3. 按target-validation选出的24个best，分别相对其自身100M final，在全部六指标上方向均有利。这是预定端点的完整描述，不是根据测试重选训练时长。best只训练到35.026M／40.030M／55.041M，但每个run实际完整训练100M；报告其full-training费用。
4. D与B4的三seed final和best均有绑定双方checkpoint SHA的全model-state等价证明；保持两组逻辑标签和24次实际训练，不把6个推理别名当额外模型、seed或新增forward。它是当前零exact-duplicate池下的机制null，不能泛化为去重永远无用。

## 固定设置、matched initial和曝光

全部方法seed11/22/33；B0为target-only，b=λ=0。B1/B2分别为C4 CE/KL，B3/B4为generated CE/KL；D/M/C分别为去重random、NLL分层、cluster选库，均用generated KL。其余b=λ=.05，LR=3e−5。每库196×512 tokens，100,156个prediction-token位置；final回放5M，约49.922次库容量曝光，不是5M独立样本。

matched initial来自同一410M revision、extended heldout及完整执行policy，baseline_group=`3aedc05b62a7fc71ec81a6d1c94aea9836be91805de091599448b2e9857e437b`。目标/C4/Wiki初始token NLL分别为2.404218149、3.093338067、3.099172284；测试分别979／979／529 chunks、447／571／60 documents、500,269／500,269／270,319预测tokens。**G=initial−adapted目标NLL，越大越好；F=adapted−initial旧域NLL，越小越好，单位nats。** 使用本matched initial，不能用主main_v3 baseline代替。

final：全部24个run均step3058、100,000,000目标tokens；B0回放0，其余21个run回放5,000,000。best：22个run为step1224、40,029,696目标tokens（非B0回放2,001,484）；B1 seed33为step1071、35,025,984目标／1,751,299回放；B3 seed33为step1683、55,040,832目标／2,752,041回放。B0的best回放仍为0。best的seed33部分对比不是相同曝光；逐项身份与完整费用见[48端点曝光表](../results/C410_nll_independent_review_v1/actual_checkpoint_exposure.csv)。

## 全部方法：mean±sample SD，n=3

下列SD均为三pipeline seeds的样本SD（ddof=1），不是文档区间、总体置信区间或三个预训练模型的差异。D/B4在实际精度内完全同state；不是另一个独立重复。

### final，token加权

'''
    body += endpoint_table(stats, 'final')
    body += '\n\n### best，token加权\n\n' + endpoint_table(stats, 'best')
    body += '\n\n### final，等文档\n\n' + endpoint_table(stats, 'final', '_equal_document')
    body += '\n\n### best，等文档\n\n' + endpoint_table(stats, 'best', '_equal_document')
    body += '''

## 六个预定配对：先同seed作差，再求mean±sample SD

括号为左方法相对右方法该指标有利的seed数。G差>0、F差<0有利；不将三个有利方向当显著性检验，不新增p值或多重比较校正。best保留各自validation选点，未强行匹配曝光。

### final，token加权

'''
    body += pair_table(pairs, 'final')
    body += '\n\n### best，token加权\n\n' + pair_table(pairs, 'best')
    body += '\n\n### final，等文档\n\n' + pair_table(pairs, 'final', '_equal_document')
    body += '\n\n### best，等文档\n\n' + pair_table(pairs, 'best', '_equal_document')
    body += '''

## 权重、个体方向和条件文档区间

目标token均值的final顺序为B3>B2>M>C>B4=D>B1>B0；等文档为B3>M>B2>C>B4=D>B1>B0，B2/M互换。best的目标均值两口径均为M>C>B4=D>B2>B0>B3>B1，但顶部差距很小，不能据此另选方法。两个端点的C4均值均是B2最低；best Wiki均值亦是B2最低。此处顺序仅编码均值，不是总体严格排序。原`mean_rankings.csv`用`>`分隔排序列表，必须结合其` tied_D_B4_states=true`读取为B4=D，不能把分隔符当统计严格不等号。

token与等文档按不同目标加权。final M−B4的G方向由token的3/3有利变为等文档2/3；final C−B4的Wiki由3/3变为2/3。best M−B4的G则由1/3变为2/3。保留这种分歧，不能挑较有利口径。等文档是先在每篇文档内平均token NLL，再给每篇文档同权；差异反映长度权重和文档内变化，不自动说明任一估计更接近部署总体。

全部288条端点与216条配对条件区间见CSV。采用冻结5000次、seed20261106的95% percentile paired-document bootstrap；条件于已训练checkpoint及文档独立／可交换重采样假设。文档（尤其Wiki只有60篇）可能存在同源/主题相关，区间不能自动称目标总体或训练随机性的置信区间，也不能替代三seed个体与sample SD。

示例保留全部三seed，展示两类不确定性不能互换：final M−B4的Wiki token差依次为+0.009608 [0.005384,0.013774]、−0.005787 [−0.009851,−0.001676]、−0.026982 [−0.032386,−0.021610]。各自条件区间不跨0，跨seed却有相反方向。best M−B4的G依次为+0.001793 [0.000452,0.003137]、−0.000064 [−0.001337,0.001196]、−0.001021 [−0.002204,0.000233]。这不构成普遍改进或等效证明。

目标域配对原bootstrap记录是NLL左−右；转为ΔG时取负，并把区间[lo,hi]转换为[−hi,−lo]。本次用独立的文档抽样次数矩阵重建504条区间，并非直接复制collector值；所有点估计/区间边界与冻结结果一致，未新建推断目标。

## 费用：24次训练、43次forward、6别名

按actual_run_id跨final/best去重后，24次实际训练合计目标2.4B、回放105M预测tokens。D虽然推理复用B4，三次实际训练仍完整计费。

| 已提交训练路径父项 | 秒 |
| --- | ---: |
'''
    for key, label in [('setup_seconds', 'setup'), ('training_seconds', 'training'), ('validation_seconds', 'validation'), ('checkpoint_seconds', 'checkpoint')]:
        body += f"| {label} | {costs['training_parent_sums'][key]:.6f} |\n"
    body += f"| 合计 | {costs['training_committed_parent_sum_seconds']:.6f}（累计父项13.948380小时，非独占elapsed） |\n"
    body += '''
这里是committed-path parent accounting，不是研究calendar elapsed或纯GPU时间；中断回滚及未结构化工作不伪装已全覆盖。独立训练scheduler whole-slot合计50,393.220313s，与上述父项是两种视角，不能相加。训练teacher-forward894.546607s、验证teacher-forward2067.003189s、验证KL arithmetic1587.912140s均为内嵌诊断项，不再加父项。各方法3次完整训练的费用及mean±SD见[方法费用表](../results/C410_nll_independent_review_v1/training_costs_by_method.csv)。best只恢复早期权重，不把实际100M训练费按40M比例打折。

43次实际推理的scheduler whole-slot合计**3109.684094s**，包含42个适应端点和1个matched initial；6个D逻辑别名增加0次forward。内含scorer subprocess wall1121.408774s、worker verification1665.495799s；teacher-forward186.670214s又内含于scorer。这些不可再相加，也不能把whole-slot称独占GPU计算或四卡calendar elapsed。

ROOT独立check整child wall1.936520s；随后analyze整child wall11.763359s／user11.212102s／sys0.615608s，已包含其内部preflight、核验、统计与写表。本次独立数值审阅＋四图输出整child wall9.171234s／user10.047923s／sys3.944831s，CPU两线程；CPU总和可高于wall。后续纯排版另计父timer，不能混称已包含于推理slot。之前失败check、修复/测试工程、非结构化写作、归档与发布费用另列或unknown，不补0。现有共同生成池/统计准备和扩展数据预处理供方法/模型共享，不把joint准备费重复加到每个方法；详见[最终成本账本说明](最终成本账本收集说明.md)。

## 与主410M／10M的关系及解释边界

本扩展保持410M初始checkpoint与原生成池来源，但目标训练容量增为39,140 chunks／20,000,540预测位置，包含主训练全部4096 chunks；100M约重复五遍该有限容量，不是100M独立token位置。C4候选扩为19,570 chunks／10,000,270预测位置，亦包含主候选2048 chunks。generated banks沿用相应主pool seed；不同协议的C4库来自扩展候选，不能把所有source差异归为目标函数。

matched heldout也与主v3相关：C410目标test的979 chunks有959个与主v3共享，C4的979个全共享；Wiki529个完全相同。扩展heldout针对扩展训练重新审计，不能拿主initial直接代用，也不是新独立测试总体。两阶段三seed共享这些来源，不能合并成六个独立重复。

与主实验相比还同时改变有效优化路径：主microbatch16×gradaccum4，C410为64×1；两者peak LR同为3e−5、warmup比例.03、min-LR比例.1，但total steps由306变3058，warmup/cosine在曝光轴上被拉长。故这些结果是数据容量、曝光、候选库和训练日程共同变化下的相关稳健性扩展，不能解释成纯horizon因果效应或独立复现；也不能从410M外推尚未解封的1B／2.8B。

## 全量可追溯附表与图

- [48端点×六指标＋来源身份](../results/C410_nll_independent_review_v1/endpoints_independent.csv)；[96条mean／sample SD](../results/C410_nll_independent_review_v1/endpoint_statistics_independent.csv)；[144行三域原始NLL／PPL／KL](../results/C410_nll_independent_review_v1/raw_corpora.csv)。
- [36个同seed配对×六指标](../results/C410_nll_independent_review_v1/paired_effects_independent.csv)；[72条配对mean／sample SD／全部seed／方向数](../results/C410_nll_independent_review_v1/paired_statistics_independent.csv)。
- [288条端点条件区间](../results/C410_nll_independent_review_v1/endpoint_conditional_CIs_independent.csv)；[216条配对条件区间](../results/C410_nll_independent_review_v1/paired_conditional_CIs_independent.csv)。
- [24次实际训练父项](../results/C410_nll_independent_review_v1/unique_training_costs.csv)；[43次实际推理父子时长](../results/C410_nll_independent_review_v1/physical_inference_costs.csv)；[6个0-forward别名](../results/C410_nll_independent_review_v1/zero_forward_aliases.csv)；[费用JSON](../results/C410_nll_independent_review_v1/cost_accounting.json)。
- [token全端点7in图](../results/C410_nll_independent_review_v1/all_endpoints_token_7in.png)；[等文档全端点](../results/C410_nll_independent_review_v1/all_endpoints_equal_document_7in.png)；[token六配对](../results/C410_nll_independent_review_v1/paired_token_7in.png)；[等文档六配对](../results/C410_nll_independent_review_v1/paired_equal_document_7in.png)。四图均有同名PDF/SVG，7×6.55in，全部seed点＋mean±sampleSD，final／best左右按metric共享y轴，不隐藏B1大遗忘或不利点。四张PNG均已逐张目检。

独立审阅实现由逐chunk sum_nll／token／doc identity重聚合，均值/SD采用`statistics`，区间采用抽样文档multiplicity矩阵，与冻结collector的计算路径不同。共2592数值检查、最大绝对误差2.498×10⁻¹⁵，10个tiny CPU tests通过；无GPU、无checkpoint载荷读取。来源/授权/完整范围及图与文档SHA见本目录[REPORT_COMPLETE.json](../results/C410_nll_independent_review_v1/REPORT_COMPLETE.json)。
'''
    candidate = '''# C410主报告插入候选（ROOT整合，非直接改报告）

本组使用扩展匹配数据和initial，全部8方法×3seed的final/best共同完整解封。final统一100M目标tokens；B0不回放，其余b=λ=.05并累计5M回放。下表为三seed均值±sample SD，G越大、F越小越好，单位nats；D/B4经双方checkpoint SHA绑定的完整state证明相同，仍分别保留逻辑标签和实际训练费。

'''
    candidate += endpoint_table(stats, 'final')
    candidate += '''

![C410全部方法与两个端点；三seed点和mean±sampleSD，final/best每行共享尺度](../results/C410_nll_independent_review_v1/all_endpoints_token_7in.png)

KL相对CE在两种回放来源中都减少旧域遗忘，但目标增益随端点改变。C4来源B2−B1的final ΔG=+0.009754±0.004606、ΔF_C4=−0.906060±0.031266、ΔF_Wiki=−0.654106±0.049217；best三项方向也均为3/3 seed有利。生成来源B4−B3的final ΔG=−0.006843±0.002144（0/3有利），best变为+0.014413±0.002290（3/3有利）；两端点旧域差均为负且各3/3有利。因此不能把validation选best的目标优势当成固定100M final的结论。

M/C仍未形成跨域、跨端点的一致优势。final的M−B4为ΔG=+0.001859±0.001958、ΔF_C4=+0.000793±0.000220、ΔF_Wiki=−0.007720±0.018371，有利seed数3/0/2；best则为+0.000236±0.001431、+0.004266±0.006515、−0.002412±0.006332，有利数1/1/2。C−B4的final为+0.001282±0.001484、+0.008217±0.001887、−0.012552±0.014334；best为+0.000102±0.001481、+0.008096±0.002828、−0.000809±0.005157。旧域间及seed间的取舍仍存在，不能由均值排名推出更普遍的保留能力。等文档口径下final M−B4的目标有利数由3/3降至2/3，进一步说明长度权重会改变小差异的个体方向。

22个best位于40,029,696目标tokens；B1 seed33为35,025,984，B3 seed33为55,040,832，非B0对应回放分别为2,001,484／1,751,299／2,752,041。best不是相同预算对照，也不减少已经完成的100M训练成本。全部24个best相对各自final在六指标方向均有利，仅描述冻结validation选点政策，未按测试另选时长。逐seed及条件文档区间不能替代n=3训练随机性描述：例如final M−B4的Wiki差三个seed一正两负，虽然各自文档区间不跨0；区间只条件于固定checkpoint与可交换文档假设。

按actual-run去重，24次完整训练committed父项50,214.167753s（setup95.463617、training40,875.164597、validation8,828.295548、checkpoint415.243991）；43次实际forward（42适应＋1初始）whole-slot合计3109.684094s，6别名新增0forward。内含scorer1121.408774s、核验1665.495799s、teacher186.670214s不能再次相加，whole-slot也不是纯GPU时间或独占calendar elapsed。CPU分析整child11.763359s，后续独立核验/四图整child9.171234s，分别发生、不包含在推理slot。

本组是相关稳健性扩展：20,000,540目标训练预测位置包含主训练全部4096 chunks，100M约重复五遍；C4候选扩大且包含主候选，生成池沿用主来源。matched目标test979 chunks中959个与主v3共享，C4的979个全共享、Wiki529个完全相同。主microbatch16×4与本组64×1不同，peak LR虽同为3e−5，306→3058步使warmup/cosine曝光路径改变。因此不能当纯horizon因果效应或独立复现，不能将两阶段三seed合成六次独立重复。

完整best表、token／等文档六配对、48个体、504条条件区间、实际曝光与费用见[完整素材](C410_100M_NLL完整结果与正文素材.md)及[独立收据](../results/C410_nll_independent_review_v1/independent_review_receipt.json)。本文不涉及1B／2.8B或行为结果。
'''
    write_new(report, body)
    write_new(insertion, candidate)
    images = [OUT / (n + '.png') for n in ('all_endpoints_token_7in', 'all_endpoints_equal_document_7in', 'paired_token_7in', 'paired_equal_document_7in')]
    inspection = {
        'status': 'all_four_pngs_visually_inspected',
        'method': 'Each PNG displayed with view_image during this authorized independent review; not inferred only from figure code.',
        'checks': ['all methods or all six frozen pairs', 'all three seeds', 'both endpoints', 'row-shared y scale', 'zero reference and metric direction', 'no clipping or legend obstruction', '7 by 6.55 inch export'],
        'figures': [proof(p) for p in images],
        'PDF_SVG': 'Saved from each same matplotlib Figure instance, not separately inferred data.',
    }
    write_new(OUT / 'figure_inspection.json', json.dumps(inspection, ensure_ascii=False, indent=2) + '\n')
    outputs = [p for p in OUT.iterdir() if p.is_file() and p.name != 'REPORT_COMPLETE.json']
    complete = {
        'status': 'passed', 'scope': 'Only ROOT-unsealed complete C410 NLL; 1B/2.8B/behavior sealed',
        'logical_endpoints': 48, 'six_metrics': 6, 'paired_seed_rows': 36,
        'independent_numeric_checks': receipt['independent_numerical_checks'], 'max_absolute_error': receipt['max_absolute_error'],
        'physical_training_runs': 24, 'physical_forward_jobs': 43, 'zero_forward_alias_endpoints': 6,
        'authority': receipt['authority'], 'complete_input': receipt['complete'],
        'documents': [proof(report), proof(insertion)],
        'outputs': [proof(p) for p in sorted(outputs)],
        'test_source': proof(ROOT / 'tests/test_C_nll_independent_review.py'),
        'review_whole_child_timing': proof(ROOT / 'artifacts/provenance/C410_nll_independent_review_v1/whole_child_timing.json'),
        'timing_note': 'Report formatting producer parent timer stored separately after process exit. Numeric/plot review parent timing already complete; no nested timing added twice.',
        'no_main_report_or_frozen_source_modified': True,
    }
    write_new(OUT / 'REPORT_COMPLETE.json', json.dumps(complete, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': proof(report), 'insertion': proof(insertion), 'completion': proof(OUT / 'REPORT_COMPLETE.json')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
