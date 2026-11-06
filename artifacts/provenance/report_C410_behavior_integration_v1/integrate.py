from pathlib import Path
import csv,datetime,difflib,hashlib,json,re,time
ROOT=Path.cwd();OUT=ROOT/'artifacts/provenance/report_C410_behavior_integration_v1';started=time.perf_counter()
def sha(b):return hashlib.sha256(b).hexdigest()
def proof(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p.relative_to(ROOT) if p.is_absolute() else p),'sha256':sha(b),'bytes':len(b)}
def write_new(p,b):
 with p.open('xb') as f:f.write(b)
report=ROOT/'研究报告.md';before=report.read_bytes()
assert sha(before)=='0041826badb90e248643fce8c146fb09c190547ec6096b254e336dd7bc8a62b9'
accept=ROOT/'artifacts/provenance/C410_behavior_review_actual_v1/ROOT_acceptance.json';ab=accept.read_bytes();a=json.loads(ab)
assert sha(ab)=='ae193858aee885c383d80220768725a4904fa8b36b0df6d2f054778eefdca05f'
assert a['logical_endpoints']==24 and a['checks']==7336
source_names=[
 'artifacts/provenance/C410_behavior_review_actual_v1/ROOT_acceptance.json',
 'results/C410_behavior_findings_v1/讨论.md',
 'results/C410_behavior_findings_v1/全部24端点可读表.md',
 'results/C410_behavior_findings_v1/全部五配对可读表.md',
 'results/C410_behavior_independent_review_v1/endpoints.csv',
 'results/C410_behavior_independent_review_v1/seed_aggregates.csv',
 'results/C410_behavior_independent_review_v1/paired_seed_aggregates.csv',
 'results/C410_behavior_independent_review_v1/paired_seed_differences.csv',
 'results/C410_behavior_presented_v1/REPORT.md',
]+['results/C410_behavior_presented_v1/'+n+'.png' for n in ['endpoints_acc_norm','endpoints_acc','fixed_lambda_pairs_acc_norm','fixed_lambda_pairs_acc']]
sources=[proof(ROOT/p) for p in source_names]
for p in sources[1:]:
 accepted=a['fresh_small_and_figure_proofs'][str((ROOT/p['path']).resolve())]
 assert accepted['sha256']==p['sha256'] and accepted['bytes']==p['bytes'],p['path']
rows=list(csv.DictReader((ROOT/source_names[5]).open()));pairs=list(csv.DictReader((ROOT/source_names[6]).open()));ends=list(csv.DictReader((ROOT/source_names[4]).open()))
assert len(rows)==128 and len(pairs)==20 and len(ends)==96
assert {e['checkpoint_kind'] for e in ends}=={'final'}
assert {e['target_prediction_tokens_at_checkpoint'] for e in ends}=={'100000000'}
assert {e['seed'] for e in ends}=={'11','22','33'}
assert {e['group'] for e in ends}=={'B0','B1','B2','B3','B4','D','M','C'}
keys=[('piqa','acc_norm'),('piqa','acc'),('hellaswag','acc_norm'),('hellaswag','acc')]
accuracy_lines=[];rendered=[]
for group in ['B0','B1','B2','B3','B4','D','M','C']:
 cells=[]
 for task,metric in keys:
  z=[r for r in rows if (r['group'],r['task'],r['metric'],r['quantity'])==(group,task,metric,'accuracy')]
  assert len(z)==1
  r=z[0];assert int(r['n_training_seeds'])==3 and r['checkpoint_kind']=='final'
  cell=f"{100*float(r['mean']):.4f}±{100*float(r['std_ddof1']):.4f}"
  cells.append(cell);rendered.append({'table':'accuracy_percent','group':group,'task':task,'score':metric,'source_mean':r['mean'],'source_std_ddof1':r['std_ddof1'],'display':cell})
 accuracy_lines.append('| '+group+' | '+' | '.join(cells)+' |')
pair_lines=[]
for left,right in [('M','B4'),('C','B4'),('B2','B1'),('B4','B3'),('D','B4')]:
 cells=[]
 for task,metric in keys:
  z=[r for r in pairs if (r['left_group'],r['right_group'],r['task'],r['metric'])==(left,right,task,metric)]
  assert len(z)==1
  r=z[0];assert int(r['n_training_seeds'])==3 and r['quantity']=='accuracy_left_minus_right'
  cell=f"{100*float(r['mean']):+.4f}±{100*float(r['std_ddof1']):.4f}"
  cells.append(cell);rendered.append({'table':'paired_difference_pp','comparison':left+'-'+right,'task':task,'score':metric,'source_mean':r['mean'],'source_std_ddof1':r['std_ddof1'],'display':cell})
 pair_lines.append('| '+left+'−'+right+' | '+' | '.join(cells)+' |')
insert='''#### 6.3.1 410M固定100M最终端点的行为评价

C410八组各三个seed的24个final行为端点已完成并通过独立复核。全部为step3058、100M目标预测token；除B0外，b=.05、λ=.05、累计5M回放预测token，学习率均为3×10⁻⁵。下表为准确率百分数的三seed均值±样本SD，主评分与辅助评分分开。任务、评分及匹配初始值与第5.5节相同。本组只评价final行为，没有best行为结果；不能把上文NLL的best优势延伸为best行为结论。

| 方法 | PiQA acc_norm（%） | PiQA acc（%） | HellaSwag acc_norm（%） | HellaSwag acc（%） |
|---|---:|---:|---:|---:|
'''+ '\n'.join(accuracy_lines)+'''

两个任务的方向不同：全部24端点的两评分相对初始均为PiQA下降、HellaSwag提高。HellaSwag各端点的原条件题目区间均高于零；PiQA不能把均值下降等同于各seed区间都低于零。PiQA主评分中，B1三个seed、B3 seed11/22、B4/D seed11、M seed11/22的区间低于零，其余跨零；辅助评分仅B1 seed11/22、B3 seed33、M seed22低于零。这些逐项区间未经多重比较校正，也不能将其数量当作独立重复。HellaSwag净准确率提高仍伴随部分旧正确题变错，完整CW/WC与条件率须一并保留。

以下完整列出行为方案中的五个同λ配对，单位pp，左减右，均值±三seed样本SD；没有为这些配对新增题目区间或检验。

| 对照 | PiQA acc_norm | PiQA acc | HellaSwag acc_norm | HellaSwag acc |
|---|---:|---:|---:|---:|
'''+ '\n'.join(pair_lines)+'''

KL的旧域NLL优势没有全部转为统一的行为优势。B2−B1在PiQA主评分为+1.4509±.7965 pp，三个seed为正；HellaSwag主评分为−.5045±.5200 pp，seed方向混合。B4−B3在PiQA主评分为+.5985±.4249 pp、HellaSwag主评分为+.2788±.1383 pp，三个seed均为正，但HellaSwag辅助acc仅+.0100±.2157 pp，seed22/33为负。它们说明任务和评分的具体差异，不能概括为KL令全部行为能力提高。

M−B4在PiQA主评分均值为负，HellaSwag主评分接近零且seed方向混合；C−B4的PiQA主评分三个seed均为正，但辅助评分seed11为负，HellaSwag两评分均值为负且seed方向混合。M/C没有跨任务、评分和seed的稳定胜者。D−B4的所有点与SD均为零，符合完整模型状态相等控制；D的实际训练成本仍计入，推断复用及相同模型不增加独立seed数。

全部seed端点与原条件区间见[acc_norm个体图](results/C410_behavior_presented_v1/endpoints_acc_norm.png)、[acc个体图](results/C410_behavior_presented_v1/endpoints_acc.png)，全部五配对见[acc_norm图](results/C410_behavior_presented_v1/fixed_lambda_pairs_acc_norm.png)、[acc图](results/C410_behavior_presented_v1/fixed_lambda_pairs_acc.png)。[全部24端点可读表](results/C410_behavior_findings_v1/全部24端点可读表.md)、[全部五配对逐seed表](results/C410_behavior_findings_v1/全部五配对可读表.md)和[原完整精度CSV](results/C410_behavior_independent_review_v1/endpoints.csv)保留全部观察，不依据较有利的任务或评分选择方法。

题目区间仍条件于既定检查点及题目独立或可交换近似，HellaSwag的activity/context依赖未校正，不能替代训练seed不确定性或保证总体、同时覆盖。主实验与C410共享基座和部分数据/生成库，目标覆盖、微批与累积安排、累计曝光、完整学习率日程及配套heldout同时变化；两阶段不能合并成六个独立seed，也不是纯horizon因果对照。C410沿用事前λ=.05，没有根据A的选参或当前行为重选。[完整行为讨论及解释边界](results/C410_behavior_findings_v1/讨论.md)。

'''
s=before.decode('utf-8');original=s
replacements=[
 ('**研究工作稿。预定科学训练、主NLL、主行为及补充A行为评价已完成，最终摘要与结论将在其余预定评价完成后填写。**','**研究工作稿。预定科学训练、主NLL、主行为及补充A/C410行为评价已完成，最终摘要与结论将在其余预定评价完成后填写。**'),
 ('**主实验、补充A/B及长程C的预定科学训练已全部完成，主NLL、主行为与A行为结果已通过完整核验；其余评价继续按冻结方案分阶段执行。**','**主实验、补充A/B及长程C的预定科学训练已全部完成，主NLL、主行为与A/C410行为结果已通过完整核验；其余评价继续按冻结方案分阶段执行。**'),
 ('主行为结果见第5.5节，验证选λ后的A行为结果见第6.1.1节；其余C模型NLL与行为结果仍须各自完整核验后报告。','主行为结果见第5.5节，A与C410行为结果分别见第6.1.1和6.3.1节；其余C模型NLL与行为结果仍须各自完整核验后报告。'),
 ('1B和2.8B训练均已完成，其NLL及C各模型行为结果仍待各自完整核验，本版不报告这些结果。主行为与已复核的A行为分别见第5.5和6.1.1节；A中按验证NLL选择的策略没有形成行为保留保证。局部表示覆盖、输出KL、文本NLL和选择题正确率描述不同对象，后续仅填入真实完成且通过完整性门禁的结果。','1B和2.8B训练均已完成，其NLL与行为结果仍待各自完整核验，本版不报告这些结果。主行为与已复核的A/C410行为分别见第5.5、6.1.1和6.3.1节。A中按验证NLL选择的策略没有形成行为保留保证；C410中旧域NLL较小的退化也没有形成跨行为指标的统一优势。局部表示覆盖、输出KL、文本NLL和选择题正确率描述不同对象，后续仅填入真实完成且通过完整性门禁的结果。'),
 ('主行为与A行为结果分别见第5.5和6.1.1节，其余补充行为评分将在各自整阶段完成并核验后报告。','主行为与A/C410行为结果分别见第5.5、6.1.1和6.3.1节，其余补充行为评分将在各自整阶段完成并核验后报告。')]
for old,new in replacements:
 assert s.count(old)==1,old;s=s.replace(old,new)
anchor='1B和2.8B训练均已完成，其NLL与行为结果仍待各自完整核验，本版不报告这些结果。'
assert s.count(anchor)==1;s=s.replace(anchor,insert+anchor)
def section(t,start,end):return t.split(start,1)[1].split(end,1)[0]
assert section(original,'### 6.1','### 6.2')==section(s,'### 6.1','### 6.2')
assert section(original,'### 5.5','## 6.')==section(s,'### 5.5','## 6.')
assert section(original,'### 6.2','### 6.3')==section(s,'### 6.2','### 6.3')
assert section(original,'## 7.','## 8.')==section(s,'## 7.','## 8.')
assert section(original,'### 6.3','1B和2.8B训练均已完成')==section(s,'### 6.3','#### 6.3.1')
for link in re.findall(r'\]\(([^)]+)\)',insert):assert (ROOT/link).is_file(),link
after=s.encode('utf-8')
write_new(OUT/'before_研究报告.md',before)
write_new(OUT/'C410_behavior_insert.md',insert.encode('utf-8'))
write_new(OUT/'report.diff',''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='before/研究报告.md',tofile='after/研究报告.md')).encode('utf-8'))
assert report.read_bytes()==before;report.write_bytes(after)
receipt={'schema_version':1,'status':'completed_C410_behavior_MD_integration_only','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'report_before':{'sha256':sha(before),'bytes':len(before)},'report_after':{'sha256':sha(after),'bytes':len(after)},'source_acceptance':sources[0],'sources':sources,'formatted_values':rendered,
 'scope':{'logical_final_endpoints':24,'groups':['B0','B1','B2','B3','B4','D','M','C'],'tasks':['piqa','hellaswag'],'scores':['acc_norm','acc'],'training_seeds':[11,22,33],'target_prediction_tokens':100000000,'physical_inference_aliases_do_not_add_seeds':True,'best_behavior_not_evaluated':True},
 'invariants':{'A_NLL_and_behavior_byte_identical':True,'primary_behavior_byte_identical':True,'B_section_byte_identical':True,'prior_C410_NLL_text_byte_identical':True,'delivery_section_byte_identical':True,'no_new_statistics':True,'no_new_inference_or_selection':True,'no_C1_C2_score_semantic_reads':True,'no_PDF_HTML_render_or_mutation':True,'no_checkpoint_base_archive_payload_read':True,'C2_restore_not_claimed_completed':True},
 'measurement':{'script_body_region_wall_seconds_until_receipt':time.perf_counter()-started,'whole_manual_authoring_review_seconds':None,'scope':'accepted small C410 CSV/Markdown/proof/figure SHA and Markdown edit only; means/SD multiplied by100 and formatted, no statistical rerun'},'producer_source':proof(Path(__file__).resolve())}
write_new(OUT/'receipt.json',(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print(json.dumps({'report_after_sha256':sha(after),'receipt':proof(OUT/'receipt.json'),'sources':len(sources),'formatted_values':len(rendered)},ensure_ascii=False))
