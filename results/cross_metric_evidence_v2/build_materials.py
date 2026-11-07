"""Display existing accepted study estimates; no new statistical estimator."""
from pathlib import Path
import csv
import hashlib
import json

W=Path(__file__).resolve().parents[2]
O=Path(__file__).resolve().parent
files={
 'P': 'results/analysis/primary_nll_v2/final/paired_statistics.csv',
 'PE': 'results/analysis/primary_nll_v2/final/endpoint_statistics.csv',
 'PB': 'results/analysis/primary_nll_v2/best/endpoint_statistics.csv',
 'H': 'results/primary_behavior_independent_review_v1/paired_seed_aggregates.csv',
 'HE': 'results/primary_behavior_independent_review_v1/endpoints.csv',
 'A': 'results/supplement_A_report_appendix_v2/selected_policy_token_seed_statistics.csv',
 'AP': 'results/supplemental_analysis_v2/supplement_A_nll/validation_selected_policy/selected_policy_statistics.csv',
 'AH': 'results/supplement_A_behavior_independent_review_v1/seed_aggregates.csv',
 'AHP': 'results/supplement_A_behavior_independent_review_v1/selected_policy_score_resolved_statistics.csv',
 'B': 'results/supplement_B_test_review_v2/six_metric_statistics.csv',
 'BP': 'results/supplement_B_test_review_v2/paired_six_metric_statistics.csv',
 'BC': 'results/analysis/supplement_B_validation_complete_v1/composition_statistics.csv',
 'C': 'results/C410_nll_independent_review_v1/paired_statistics_independent.csv',
 'CH': 'results/C410_behavior_independent_review_v1/paired_seed_aggregates.csv',
 'CHE': 'results/C410_behavior_independent_review_v1/endpoints.csv',
}
tables={k:list(csv.DictReader((W/v).open())) for k,v in files.items()}
citations=[]
def one(source, **selector):
 rows=[r for r in tables[source] if all(r[k]==str(v) for k,v in selector.items())]
 assert len(rows)==1, (source, selector, len(rows))
 r=rows[0];citations.append({'source':source,'selector':selector,'existing_row':r});return r
def estimate(source, scale=1, places=5, **selector):
 r=one(source,**selector)
 return f"{float(r['mean'])*scale:+.{places}f}±{float(r['std_ddof1'])*scale:.{places}f}"
def pn(comp,metric):return estimate('P',comparison=comp,metric=metric,ratio='0.05')
def bh(source,left,right,task,score='acc_norm'):
 return estimate(source,100,4,left_group=left,right_group=right,task=task,metric=score,checkpoint_kind='final',ratio='0.05')
def bpair(k,metric):return estimate('BP',bank_chunks=str(k),comparison='M-B4',metric=metric)
def table(head, rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(row)+' |' for row in rows])+'\n'

rows=[]
rows.append(['同λ的objective对照：主 final B2−B1',f"C4/Wiki ΔF={pn('B2-B1','f_old_c4')}/{pn('B2-B1','f_old_wikitext')}；行为PiQA/HS acc_norm差={bh('H','B2','B1','piqa')}/{bh('H','B2','B1','hellaswag')} pp。",'相同库、b/λ下KL旧域NLL较好，行为均值却较低。支持当前设置的指标分离，不支持KL普遍劣于CE或改善全部知识。','P,H'])
rows.append(['同λ的选库对照：主 final M/C−B4',f"M的ΔG={pn('M-B4','g_new')}、ΔWiki F={pn('M-B4','f_old_wikitext')}；C的ΔG={pn('C-B4','g_new')}、ΔC4 F={pn('C-B4','f_old_c4')}。",'分层有部分有利均值但不是全面优胜；聚类final反例必须保留。不能只报目标或单一旧域。','P'])
rows.append(['选库代理变好', 'M的NLL经验CDF最大差 .0322±.0051，小于随机 .0635±.0277；C的最近代表特征平方距离 .2096±.0030，小于随机 .2413±.0035，但CDF偏移 .1068±.0218 更大。','只是各自代理空间的代表性；不是梯度、独立旧域、语义或行为覆盖率。与上一行并列，不能推断代理指标下降⇒retention收益。','G'])
bc=[]
for k in ['20','196']:
 r=one('BC',group='B4',bank_chunks=k,metric='teacher_nll_mean');m=one('BC',group='M',bank_chunks=k,metric='teacher_nll_mean')
 bc.append(f"K{k}: {float(r['std_ddof1']):.6f}→{float(m['std_ddof1']):.6f}")
sd=[]
for k in ['20','196']:
 r=one('B',group='B4',bank_chunks=k,metric='g_new');m=one('B',group='M',bank_chunks=k,metric='g_new')
 sd.append(f"K{k}: {float(r['std_ddof1']):.6f}→{float(m['std_ddof1']):.6f}")
rows.append(['B：组成波动≠端点波动','bank平均教师NLL的10draw SD（B4→M）为'+ '；'.join(bc)+'；目标G端点SD为'+'；'.join(sd)+'。','组成标量更稳，但本批端点SD未随之下降；不是证明总体方差更大，也没有测得梯度方差。n=10是固定pool11/train11的选库draw。','BC,B'])
rows.append(['B：均值收益依赖K', 'M−B4在K20的 ΔG/ΔC4 F/ΔWiki F='+ '/'.join(bpair(20,m) for m in ['g_new','f_old_c4','f_old_wikitext'])+'；K196为'+'/'.join(bpair(196,m) for m in ['g_new','f_old_c4','f_old_wikitext'])+'。','K196三均值有利而K20相反；保留可能收益与反例，不按test重选K，不称每draw都改善。K改变容量及重复利用，不只改变一个噪声标量。','BP'])
rows.append(['不同端点与估计对象', '主C final数学G：token '+estimate('PE',group='C',ratio='0.05',metric='g_new')+'，等文档 '+estimate('PE',group='C',ratio='0.05',metric='g_new_equal_document')+'；C best token '+estimate('PB',group='C',ratio='0.05',metric='g_new')+'。','近零绝对gain符号可随estimand变化；不能泛称方法排序翻转。best由验证选点、实际曝光不同，不能拿best当固定10M的等预算方法收益。','PE,PB,X'])
rows.append(['C410同λ长程设置', 'B2−B1旧域ΔF='+ '/'.join(estimate('C',checkpoint='final',comparison='B2-B1',metric=m) for m in ['f_old_c4','f_old_wikitext'])+'；行为PiQA/HS acc_norm差='+bh('CH','B2','B1','piqa')+'/'+bh('CH','B2','B1','hellaswag')+' pp。','旧域NLL有利而两行为任务方向不同。C410与主共享来源，且数据量/批量/日程一起变，不是独立seed复制或纯horizon干预。','C,CH'])
cw=one('HE',group='B4',seed='11',checkpoint_kind='final',ratio='0.05',task='hellaswag',metric='acc_norm')
rows.append(['净accuracy≠无遗忘',f"主B4 seed11 HS acc_norm Δ={100*float(cw['accuracy_delta']):+.4f} pp，同时CW={cw['CW']}、WC={cw['WC']}，initial正确题条件遗忘率={100*float(cw['conditional_forgetting_rate']):.4f}%。",'Δaccuracy=(WC−CW)/N；新获得抵消旧损失。不能用净提高说没有遗忘，也不把每题/alias当训练seed。','HE'])

text='# 跨指标证据矩阵与讨论（已验收范围）\n\n'
text+='本研究的可支持结论是：更稳定的库组成、更小的语言模型NLL退化与更高的选择题正确率是不同命题，当前证据没有建立它们的普遍单调链条。下面保留条件收益及反例，不把未识别机制写成因果结论。\n\n供论文与答辩整合；不是新结果分析。仅覆盖已全解封、ROOT验收的主NLL/行为、A NLL及冻结所选A行为、B NLL、C410 NLL/行为；不含C1/C2分数。所有±均为原表sample SD，NLL单位nats/预测token，行为差单位pp；G越大越好，F越小越好。下表使用预定对照呈现反例，不以这些例子代替完整表或重新选择条件。\n\n'
text+=table(['问题/设计','已有数字证据','支持的结论与边界','来源ID'],rows)

text+='\n## A：各自验证选λ的完整策略，不是同λ因子效应\n\n'
text+='六行全部保留。NLL与行为来自同一批固定10M final；λ此前只依最终验证NLL目标选定，未用行为/test重选。NLL列为三seed均值±SD，行为为相对matched initial的acc_norm变化（pp）。acc辅助score与全部seed见[完整A行为表](../supplement_A_behavior_findings_v1/全部18端点可读表.md)。\n\n'
arows=[]
for g in ['B1','B2','B3','B4','M','C']:
 r=one('A',method=g);vals=[f"{float(r[m+'_mean']):+.5f}±{float(r[m+'_sample_sd']):.5f}" for m in ['g_new','f_old_c4','f_old_wikitext']]
 hv=[estimate('AH',100,4,group=g,task=t,metric='acc_norm',quantity='accuracy_delta') for t in ['piqa','hellaswag']]
 arows.append([g,r['lambda'],*vals,*hv])
text+=table(['方法','验证λ','数学G↑','C4 F↓','Wiki F↓','PiQA Δ pp','HS Δ pp'],arows)
text+='\nA的B4(λ5)较B3(λ.1)在数学与两旧域NLL均值有利，但HS主score差为'+estimate('AHP',100,4,comparison='B4-B3',task='hellaswag',score='acc_norm',quantity='accuracy_left_minus_right')+' pp；这比较完整策略，不是固定λ的KL净效应。C−B4（λ1对5）的HS主score差为'+estimate('AHP',100,4,comparison='C-B4',task='hellaswag',score='acc_norm',quantity='accuracy_left_minus_right')+' pp，但其C4 NLL退化差为'+estimate('AP',comparison='C-B4',metric='f_old_c4')+'。同λ=5的M−B4数学G差为'+estimate('AP',comparison='M-B4',metric='g_new')+'，HS主score差为'+estimate('AHP',100,4,comparison='M-B4',task='hellaswag',score='acc_norm',quantity='accuracy_left_minus_right')+' pp；不能概括为M稳定优胜。A的PiQA全部单端点两score对initial的题目区间跨零，HS方向与区间细节不得由这张均值表替代。[A全42候选/同λ与策略对照](../supplement_A_report_appendix_v2/补充A_共有lambda与条件区间附表.md)保持原样；原行为aggregate标签碰撞表不改，本材料只用reviewer的score-resolved表。\n'

text+='\n## 为什么这些量不必共同改善：有界解释\n\n'
text+='**指标对象不同。** NLL评估真实文本标签的平均log loss，教师forward KL约束所选前缀上的完整教师分布；PiQA/HS则按有限选项的argmax及字符长度归一规则判断正确性。概率质量的小幅连续变化未必改变答案排序，也可能改变少量边界题的选择。因此“旧域NLL较少退化”和“行为准确率较高”没有已证明的单调映射。C4不是Pythia原预训练集，数学语料NLL不是数学推理正确率，两个选择题任务也不代表全部知识。\n\n'
text+='**抽样代理不等于优化方向。** teacher-NLL是标量；相同NLL可对应不同参数Jacobian、梯度方向与曲率。理想平滑模型在θ₀附近、冻结前缀测度μ下，教师KL仅局部满足 Rμ(θ₀+δ)≈½δᵀFμδ；选库改变μ及被约束方向，但NLL分层或教师均值特征距离并不保证Fμ更接近旧任务需要的方向。真实AdamW、多步非线性、clip和有限重复库进一步限制该局部解释。B的证据仅说明组成更稳不足以保证本样本终点更稳；未测Fisher谱/梯度方差，不能倒推已经证实机制。[既有推导](../../docs/局部Fisher解释与选库诊断.md)。\n\n'
text+='**CE/KL的期望恒等式不能替代有限bank训练。** 固定学生与前缀、标签严格来自无截断教师条件分布时，E_q[−log p]=H(q)+KL(q‖p)；有限bank只保存一次硬标签，重复训练后θ依赖该bank，选样也改变上下文/标签条件。不能把训练后同bank经验CE当KL，或从上述恒等式推出CE/KL终点相同。现有同池replay-fit是post-validation探索性材料，不是独立test，也不能解释全部旧域/行为差异。\n'

text+='\n## 设计量与不确定性：答辩时必须分开\n\n'
text+='- **K、b、λ、累计量。** K是不同回放预测位置容量，T_R=bT_target是累计处理量，λ是分别求均值后的回放loss权重；每步单回放token相对目标token的系数为λN_target/N_replay，名义比值为λ/b；尾步/整数配额应依实际token计数解释。主K=100,156、b=λ=.05时累计500,000，约4.9922次利用；C410同K但累计5M，约49.9221次。主B4的b=.01/.05/.1固定λ=.05时，C4 F均值 .13830/.14445/.15041，更多曝光未显示统一改善，同时λ/b也改变，不能称只增加独立信息。B K20/196在相同累计量下也改变重复利用；候选pool大小不是实际bank容量。[预算口径](../../docs/扩展回放预算与主实验对照.md)。\n\n'
text+='- **final、best与成本。** 主final均10M，best实际1.6352/3.2704/4.9056M，部分方法的best两侧预算不同；C410也有不同best选点。两个test观测端点不能推出全程单调，full training成本仍计已完成全程。D/B4同模型alias、final/best、A复用主端点均不增加独立训练重复。[主曝光与完整表](../../docs/主NLL测试结果讨论.md)。\n\n'
text+='- **估计对象与重复单位。** token权重与等文档权重回答不同问题；chunk全为512输入，文档留块数分布不同不等于原始文档长度的因果实验。主/A/C410在适用的回放组中随seed改变训练顺序与pool/选库；B0没有回放库。B的十draw固定训练seed和pool，不能拼成n=13。文档bootstrap/题目bootstrap条件于已训练checkpoint，不能替代seed SD或重复选λ流程，HellaSwag同context相关性未修；多端点区间未经多重比较校正。相同数字draw seed也不自动构成严格common random numbers。\n\n'
text+='可支持的研究贡献是：在固定预算与冻结评价下，明确区分并实测了“组成/几何代表性、语言模型保留、选择题得失、条件选库波动”四个层次，保留它们不一致的证据。仍未识别的是哪种参数方向/样本机制导致个别终点变化，以及对新教师、独立数据总体、更多训练seed和更多任务的外推；当前材料不把这些待识别问题写成因果发现，也不据当前结果追加选择规则。\n'

text+='\n## 来源与可追溯性\n\n'
extras={'G':'docs/回放库分布诊断.md','X':'docs/主NLL测试结果讨论.md','BREF':'results/bank_diagnostics/conditional_nll_sampling_reference.json','BREPORT':'docs/补充B_完整60run测试结果解释.md','METHOD':'docs/局部Fisher解释与选库诊断.md','BUDGET':'docs/扩展回放预算与主实验对照.md','PRIMARY_REVIEW':'results/primary_behavior_independent_review_v1/COMPLETE.json','A_REVIEW':'results/supplement_A_behavior_independent_review_v1/COMPLETE.json','C410_REVIEW':'results/C410_behavior_independent_review_v1/COMPLETE.json'}
source_files={**files,**extras}
proofs={k:{'path':v,'sha256':hashlib.sha256((W/v).read_bytes()).hexdigest(),'bytes':(W/v).stat().st_size} for k,v in source_files.items()}
for k,v in proofs.items():text+=f"- `{k}`：[{Path(v['path']).name}](../../{v['path']})，SHA `{v['sha256'][:16]}…`。\n"
text+='\n完整SHA、原表selector与逐字原行在同目录provenance.json/cited_rows.json；只读取既有小结果表/说明，没有重新训练、推理、抽样、回归、相关检验或重新选参。引用的历史说明可能保留当时“其他phase仍封存”字样，此处范围以各实际ROOTaccepted完整分析为准；不改写历史资料。\n'
(O/'跨指标证据矩阵与讨论.md').write_text(text)
(O/'cited_rows.json').write_text(json.dumps(citations,ensure_ascii=False,indent=2)+'\n')
provenance={'complete':True,'scope':['primary_NLL','primary_behavior','A_NLL','A_validation_selected_behavior','B_NLL','C410_NLL','C410_behavior'],'C1_C2_scores_accessed':False,'original_outputs_modified':False,'new_statistics_regression_correlation_bootstrap_or_selection':False,'method':'Exact existing CSV rows formatted at stated units; documentary geometry/budget values cited from existing accepted tables. No fresh estimator.','sources':proofs,'outputs':{str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [O/'跨指标证据矩阵与讨论.md',O/'cited_rows.json',Path(__file__)]},'manual_overall_wall_seconds':None}
(O/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'complete':True,'provenance_sha256':hashlib.sha256((O/'provenance.json').read_bytes()).hexdigest(),'markdown_sha256':hashlib.sha256((O/'跨指标证据矩阵与讨论.md').read_bytes()).hexdigest()},ensure_ascii=False))
