"""Copy reviewed A results and format existing values; no inference or selection."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

W = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
R = W / 'results/supplement_A_behavior_independent_review_v1'
P = W / 'results/supplement_A_behavior_presented_v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda n: list(csv.DictReader((R / n).open()))
e = read('endpoints.csv')
a = read('seed_aggregates.csv')
s = read('selected_policy_score_resolved_statistics.csv')
q = read('selected_question_intervals.csv')
groups = ['B1', 'B2', 'B3', 'B4', 'M', 'C']
tasks = ['piqa', 'hellaswag']
scores = ['acc_norm', 'acc']
comparisons = ['M-B4', 'C-B4', 'B2-B1', 'B4-B3', 'B3-B1', 'B4-B2']
assert len(e) == 72 and len(q) == 72 and len(s) == 24
assert {r['checkpoint_kind'] for r in e} == {'final'}
assert {r['target_prediction_tokens_at_checkpoint'] for r in e} == {'10000000'}
assert {r['replay_prediction_tokens_at_checkpoint'] for r in e} == {'500000'}
assert {r['step'] for r in e} == {'306'}
for f in R.glob('*.csv'):
    assert f.read_bytes() == (P / f.name).read_bytes()
    shutil.copyfile(f, OUT / f.name)

def fmt(x, signed=False):
    return format(float(x) * 100, '+.4f' if signed else '.4f')

def summary(group, task, score, quantity='accuracy_delta'):
    rows = [r for r in a if r['group'] == group and r['task'] == task and r['metric'] == score and r['quantity'] == quantity]
    assert len(rows) == 1
    r = rows[0]
    return f"{fmt(r['mean'], quantity != 'accuracy')} ± {fmt(r['std_ddof1'])}"

def pair(comparison, task, score):
    rows = [r for r in s if r['comparison'] == comparison and r['task'] == task and r['score'] == score]
    assert len(rows) == 1
    r = rows[0]
    return f"{fmt(r['mean'], True)} ± {fmt(r['std_ddof1'])}"

def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n'

intro = ('全部数值来自独立 reviewer 已复核的 CSV；仅作百分数/百分点单位转换和四位小数显示，未新增区间、检验、推理或选参。'
         '每种方法使用此前冻结的验证选择 λ，final 均为 step306、目标 10,000,000 预测 token、回放 500,000 token、b=.05。'
         'mean±SD 为三个训练 seed 的既有样本 SD（ddof=1）；题目 bootstrap 区间条件于固定 checkpoint，不是训练/重新选择 λ 的不确定性。\n\n')

text = '# A 行为结果：全部 18 个端点\n\n' + intro
text += 'acc_norm 为主 score，acc 为辅助 score；accuracy 是百分比，Δ 和区间是相对 matched initial 的百分点。CW 为 initial 对→本模型错，WC 为 initial 错→本模型对。条件遗忘/获得率也显示为百分比。全部 seed 保留，未按方向删行。\n\n'
identity = []
for g in groups:
    for seed in ['11', '22', '33']:
        r = next(r for r in e if r['group'] == g and r['seed'] == seed)
        identity.append([g, r['weight'], seed, '`' + r['run_id'] + '`', '`' + r['actual_run_id'] + '`'])
text += table(['方法', '验证 λ', 'seed', 'logical run', 'actual run'], identity)
text += '\nB1 的三项复用主实验实际训练/推理来源，不构成新训练 seed；其逻辑身份仍为 supplement_A。\n'
for task in tasks:
    for score in scores:
        text += f'\n## {task} / {score}\n\n'
        text += f"matched initial accuracy：{fmt(next(r['initial_accuracy'] for r in e if r['task'] == task and r['metric'] == score))}%；题目数 {next(r['items'] for r in e if r['task'] == task and r['metric'] == score)}。\n\n"
        text += table(['方法', 'λ', 'accuracy mean±SD (%)', 'Δ mean±SD (pp)'], [[g, next(r['weight'] for r in e if r['group'] == g), summary(g, task, score, 'accuracy'), summary(g, task, score)] for g in groups])
        text += '\n'
        rows = []
        for g in groups:
            for seed in ['11', '22', '33']:
                r = next(r for r in e if r['group'] == g and r['seed'] == seed and r['task'] == task and r['metric'] == score)
                rows.append([g, seed, fmt(r['accuracy']), fmt(r['accuracy_delta'], True), f"[{fmt(r['delta_ci_low'], True)}, {fmt(r['delta_ci_high'], True)}]", r['CC'], r['CW'], r['WC'], r['WW'], fmt(r['conditional_forgetting_rate']), fmt(r['conditional_acquisition_rate'])])
        text += table(['方法', 'seed', 'accuracy (%)', 'Δ (pp)', '5000题目bootstrap Δ 95%区间 (pp)', 'CC', 'CW', 'WC', 'WW', '条件遗忘率 (%)', '条件获得率 (%)'], rows)
(OUT / '全部18端点可读表.md').write_text(text)

text = '# A 行为结果：全部六个验证选 λ 策略对照\n\n' + intro
text += '方向统一为左减右；不同 λ 的策略比较不能解读为固定 λ 的单一 objective、回放来源或选库效应。只有 M−B4 两边 λ 都为5；其原同 λ CSV/图也单独保留。原 selected_policy_statistics.csv 的 score 标签碰撞不改写；本表从 reviewer 的 score-resolved 表及保留 score 的逐 seed 配对/区间表读取。\n\n'
for task in tasks:
    for score in scores:
        text += f'## {task} / {score}\n\n'
        rows = []
        for c in comparisons:
            r = next(r for r in s if r['comparison'] == c and r['task'] == task and r['score'] == score)
            vals = []
            for seed in ['11', '22', '33']:
                x = next(x for x in q if x['comparison'] == c and x['training_seed'] == seed and x['task'] == task and x['metric'] == score)
                vals.append(f"{fmt(x['accuracy_left_minus_right'], True)} [{fmt(x['ci_low'], True)}, {fmt(x['ci_high'], True)}]")
            rows.append([c, r['lambda_a'] + ' / ' + r['lambda_b'], pair(c, task, score), *vals])
        text += table(['对照', '左/右 λ', '三seed mean±SD (pp)', 'seed11: 差 [条件题目CI]', 'seed22: 差 [条件题目CI]', 'seed33: 差 [条件题目CI]'], rows) + '\n'
text += '全部逐 seed run_a/run_b、左右 λ、题目四状态计数、bootstrap seed 与5000次重复保存在原字节 CSV。区间未做多重比较校正，不把个别不跨零区间升级为经过选择/多重性控制的发现。\n'
(OUT / '全部六策略配对可读表.md').write_text(text)

text = '# A 行为结果：解释与边界\n\n'
text += ('本次只分析 ROOT 已完整解封的 supplement_A：六方法各三 seed、18个固定10M final；PiQA1838题、HellaSwag10042题，主 acc_norm 与辅助 acc 分开。'
         '原完整分析、独立 reviewer 和展示均实际成功，未访问其他补充 phase，也没有重新选择 λ。λ 依次为 B1=.05、B2=1、B3=.1、B4=5、M=5、C=1；这些是此前按最终验证 NLL 目标冻结的选择，不是按行为分数挑选。\n\n')
text += '## 完整结果摘要\n\n相对 matched initial 的变化，单位 pp；表中 ± 为三 seed 样本 SD。\n\n'
text += table(['方法', 'λ', 'PiQA acc_norm', 'PiQA acc', 'HellaSwag acc_norm', 'HellaSwag acc'], [[g, next(r['weight'] for r in e if r['group'] == g), summary(g, 'piqa', 'acc_norm'), summary(g, 'piqa', 'acc'), summary(g, 'hellaswag', 'acc_norm'), summary(g, 'hellaswag', 'acc')] for g in groups])
text += ('\nPiQA 中，C 的均值为正，B4 为负；全部18端点的两种 score 相对 initial 的题目区间均跨零。不能据这些小幅均值变化声称明确的 PiQA 保留增益或损害。\n\n'
         'HellaSwag 中，B1、B3 的两种 score 在全部三个 seed 都提高，全部相对 initial 的条件题目区间也高于零。B2、B4、M 的 acc_norm 在三个 seed 都下降；B2 与 B4 各有两个 seed 的区间低于零，M 的三个 seed 都低于零。辅助 acc 的 B2 三个区间均跨零，B4 三个都低于零，M 只有 seed22 低于零。C 的两种 score 均值提高，acc_norm 的 seed11/22 与 acc 的三个 seed 区间高于零，acc_norm seed33 跨零。以上区间均未调整多重比较。\n\n'
         '这显示最终验证 NLL 选择的策略不必保证行为准确率保留；也不能只凭本表把差异归因于 KL、教师强度、选库几何或某一种标签机制。不同方法的 λ 已不同，这些行为任务也没有参与该选择。全部固定同 λ 研究结果和验证目标仍各自保留。\n\n')
text += '## 六个预定策略对照\n\n左减右，单位 pp；不是六种新调参或事后挑选。\n\n'
text += table(['对照', 'PiQA acc_norm', 'PiQA acc', 'HellaSwag acc_norm', 'HellaSwag acc'], [[c, pair(c, 'piqa', 'acc_norm'), pair(c, 'piqa', 'acc'), pair(c, 'hellaswag', 'acc_norm'), pair(c, 'hellaswag', 'acc')] for c in comparisons])
text += ('\nM−B4 是本批唯一相同 λ=5 的方法配对。PiQA acc_norm 三个 seed 都为正，但三个问题区间均跨零；辅助 acc 有一个 seed 为负。HellaSwag acc_norm 均值接近零，三个区间跨零；acc 均值为正但 seed22 为负，且三 seed 的条件区间分别为正/负/正。这不支持 M 在两任务、两 score 上稳定优于 B4。\n\n'
         'C−B4 的两任务、两 score 在三个 seed 都为正；HellaSwag 两种 score 的全部逐 seed 区间均高于零，PiQA 仅 acc_norm seed11 的区间高于零。此处比较 C 的 λ=1 策略与 B4 的 λ=5 策略，不能称同一教师约束强度下的选库净效应。\n\n'
         'B2−B1 与 B4−B3 的 HellaSwag 两 score 在三个 seed 全为负，逐 seed 条件区间也全低于零；这是不同验证 λ 策略的观察，不等于固定 λ 下 KL 普遍差于 CE。B3−B1 的 HellaSwag 两 score 均值也为负，只有 acc_norm seed33 区间低于零；PiQA 有 seed 方向变化。B4−B2 的 HellaSwag acc_norm 均值为负但全部区间跨零，辅助 acc 的 seed11/33 区间低于零；PiQA 主 score 均值为负、辅助 score 为正。两个回放来源对照因此也需保留 score 与 λ 的具体条件，不能概括为来源的普遍排序。\n\n')
text += ('## 不确定性、审查与工程边界\n\n'
         '样本 SD 只描述三个固定训练 seed；题目 bootstrap 条件于训练后的 checkpoint，未重跑训练或验证选择，不替代 seed 不确定性。HellaSwag 同 activity/context 的相关性未校正，不能自动推广为总体置信区间。两个任务、两 score、六配对和多 seed 的区间均未经多重比较校正。准确率净变化也不等于没有遗忘，全部 CW/WC 及条件遗忘/获得率见完整逐 seed 表。\n\n'
         '独立 reviewer 从实际 choice 分数重算 argmax、accuracy、四状态、三 seed SD、固定5000题目 bootstrap，并核对9528项数值/身份；最大绝对误差1.11×10⁻¹⁶。其 fresh读取392,807,220B 为本次授权行为结果/逐题/小来源文件，不是 checkpoint；无模型 forward、无 CP/native base/TAR/ZIP读取。展示器只读取复核后的小表，七份 CSV 与 reviewer 原字节一致，六张图各含 PNG/PDF/SVG。原 CSV 标签碰撞表保持不变，本说明采用独立 score-resolved 视图。\n\n'
         '独立 reviewer 外层完整进程8.667267s（user8.706218/sys3.894478），展示4.851646s（user4.901889/sys2.130553）；其脚本内部 timer 均嵌套，不能再相加。文档人工整体耗时未知。此前执行清单是准备记录；ROOT 实际使用 readiness.json、ROOT_authority.json 和 results/supplement_A_behavior_analysis_actual_v1，与准备中的示例输出路径不同，真实 COMPLETE/authority/不可重置 first-read journal 才是此次来源。\n\n'
         '全部数字的 SHA 与原始/复核/展示/计时来源见同目录 COMPLETE.json。可读全表见 [全部18端点可读表](全部18端点可读表.md)、[全部六策略配对可读表](全部六策略配对可读表.md)，全图见 [展示报告](../supplement_A_behavior_presented_v1/REPORT.md)。\n')
(OUT / '讨论.md').write_text(text)

proofs = [R/'COMPLETE.json', P/'COMPLETE.json', W/'results/supplement_A_behavior_analysis_actual_v1/COMPLETE.json', W/'artifacts/provenance/A_behavior_analysis_actual_v1/ROOT_authority.json', W/'artifacts/provenance/A_behavior_analysis_actual_v1/analyze_timing/whole_child_timing.json', W/'artifacts/provenance/A_behavior_analysis_actual_v1/state_capture/actual_successor_state_immutable.json', W/'artifacts/provenance/A_behavior_independent_review_actual_v1/whole_parent_timing.json', W/'artifacts/provenance/A_behavior_presentation_actual_v1/whole_parent_timing.json']
original = json.loads((W/'results/supplement_A_behavior_analysis_actual_v1/COMPLETE.json').read_text())
proofs.append(Path(original['first_read_receipt']['path']))
allfiles = [p for p in OUT.iterdir() if p.is_file()]
receipt = {'complete': True, 'phase': 'supplement_A_behavior', 'logical_endpoints': 18, 'task_score_rows': 72, 'selected_policy_pairs': 6, 'selected_policy_seed_task_score_rows': 72, 'selected_policy_aggregate_rows': 24, 'selection_changed': False, 'original_outputs_modified': False, 'no_new_inference_CI_or_selection': True, 'source_proofs': {str(p): sha(p) for p in proofs}, 'reviewer_csv_copies_byte_equal_presenter': True, 'output_sha256': {str(p): sha(p) for p in allfiles}, 'manual_overall_wall_seconds': None, 'scope': 'Readable formatting and interpretation of complete authorized A behavior only. All seven reviewed CSVs copied unchanged. Preparation examples are historical, not claims of actual execution paths.'}
(OUT / 'COMPLETE.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'complete': True, 'receipt_sha256': sha(OUT/'COMPLETE.json'), 'discussion_sha256': sha(OUT/'讨论.md'), 'files': len(allfiles)}, ensure_ascii=False))
