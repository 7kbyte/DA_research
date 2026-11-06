from pathlib import Path
import csv, datetime, difflib, hashlib, json, re, time
ROOT=Path.cwd()
OUT=ROOT/'artifacts/provenance/report_A_behavior_integration_v1'
started=time.perf_counter()
def sha(b): return hashlib.sha256(b).hexdigest()
def proof(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p.relative_to(ROOT) if p.is_absolute() else p),'sha256':sha(b),'bytes':len(b)}
def write_new(p,b):
 with p.open('xb') as f:f.write(b)
report=ROOT/'研究报告.md'
before=report.read_bytes()
assert sha(before)=='57ba6b0cd82c9661a19cf88629a799123f524aa5cbac931fca4578b1b679e1db'
accept=ROOT/'artifacts/provenance/A_behavior_independent_review_actual_v1/ROOT_acceptance.json'
a=json.loads(accept.read_bytes())
assert sha(accept.read_bytes())=='7d717008c7b4ec57192278ba6060889204fb81625eadb2d8dad470eb3f202e2d'
assert a['logical_endpoints']==18 and a['checks']==9528
source_names=[
 'artifacts/provenance/A_behavior_independent_review_actual_v1/ROOT_acceptance.json',
 'results/supplement_A_behavior_findings_v1/讨论.md',
 'results/supplement_A_behavior_findings_v1/全部18端点可读表.md',
 'results/supplement_A_behavior_findings_v1/全部六策略配对可读表.md',
 'results/supplement_A_behavior_independent_review_v1/endpoints.csv',
 'results/supplement_A_behavior_independent_review_v1/seed_aggregates.csv',
 'results/supplement_A_behavior_independent_review_v1/selected_policy_score_resolved_statistics.csv',
 'results/supplement_A_behavior_independent_review_v1/selected_question_intervals.csv',
 'results/supplement_A_behavior_presented_v1/REPORT.md',
]+['results/supplement_A_behavior_presented_v1/'+n+'.png' for n in [
 'endpoints_acc_norm','endpoints_acc','selected_policy_pairs_acc_norm','selected_policy_pairs_acc','fixed_lambda_pairs_acc_norm','fixed_lambda_pairs_acc']]
sources=[proof(ROOT/p) for p in source_names]
for p in sources[1:]:
 accepted=a['fresh_small_and_figure_proofs'][str((ROOT/p['path']).resolve())]
 assert accepted['sha256']==p['sha256'] and accepted['bytes']==p['bytes'],p['path']
rows=list(csv.DictReader((ROOT/source_names[5]).open()))
pairs=list(csv.DictReader((ROOT/source_names[6]).open()))
assert len(rows)==96 and len(pairs)==24
keys=[('piqa','acc_norm'),('piqa','acc'),('hellaswag','acc_norm'),('hellaswag','acc')]
weights={'B1':'.05','B2':'1','B3':'.1','B4':'5','M':'5','C':'1'}
accuracy_lines=[];rendered=[]
for group in weights:
 cells=[]
 for task,score in keys:
  selected=[r for r in rows if (r['group'],r['task'],r['metric'],r['quantity'])==(group,task,score,'accuracy')]
  assert len(selected)==1
  r=selected[0]; assert int(r['n_training_seeds'])==3
  cell=f"{100*float(r['mean']):.4f}±{100*float(r['std_ddof1']):.4f}"
  cells.append(cell);rendered.append({'table':'accuracy_percent','group':group,'task':task,'score':score,'source_mean':r['mean'],'source_std_ddof1':r['std_ddof1'],'display':cell})
 accuracy_lines.append('| '+group+' | '+weights[group]+' | '+' | '.join(cells)+' |')
pair_lines=[]
for comparison in ['M-B4','C-B4','B2-B1','B4-B3','B3-B1','B4-B2']:
 cells=[];ls=[]
 for task,score in keys:
  selected=[r for r in pairs if (r['comparison'],r['task'],r['score'])==(comparison,task,score)]
  assert len(selected)==1
  r=selected[0];assert int(r['n_training_seeds'])==3 and r['quantity']=='accuracy_left_minus_right'
  ls.append((float(r['lambda_a']),float(r['lambda_b'])))
  cell=f"{100*float(r['mean']):+.4f}±{100*float(r['std_ddof1']):.4f}"
  cells.append(cell);rendered.append({'table':'paired_difference_pp','comparison':comparison,'task':task,'score':score,'source_mean':r['mean'],'source_std_ddof1':r['std_ddof1'],'display':cell})
 assert len(set(ls))==1
 la,lb=ls[0]
 pair_lines.append('| '+comparison.replace('-','−')+f' | {la:g}/{lb:g} | '+' | '.join(cells)+' |')
insert='''#### 6.1.1 验证选λ策略的行为评价

补充A已完成六方法各三个seed的18个固定10M final行为端点，并通过完整独立复核。任务、评分、初始模型与第5.5节相同；每种方法沿用此前由验证NLL冻结的λ，不按行为结果重选。下表报告准确率百分数的三seed均值±样本SD，`acc_norm`为主评分、`acc`为辅助评分。匹配初始值依次为PiQA 66.9206%/66.8118%、HellaSwag 40.3605%/33.8578%。

| 方法 | 验证所选λ | PiQA acc_norm（%） | PiQA acc（%） | HellaSwag acc_norm（%） | HellaSwag acc（%） |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(accuracy_lines)+'''

验证NLL选择与行为保留并不等价。HellaSwag的acc_norm相对初始均值变化，B2、B4、M分别为−.4514、−.6074、−.6141个百分点（pp），三个seed方向均为负；B1、B3、C分别为+1.9153、+1.4605、+.5079 pp。B1/B3的两评分在三个seed均提高、逐seed条件题目区间均高于零；C的acc_norm seed33区间仍跨零。PiQA全部18端点的两评分相对初始的条件题目区间均跨零，不能将表中的小幅均值变化解释为已确证的保留增益或损害。这句话只针对端点与初始的比较，不代表所有方法配对区间均跨零。

六个预定策略对照全部列于下表，单位为pp，左减右，±仍为三seed样本SD。λ列保留双方实际验证选择，因此比较对象是“方法加验证选参”的政策。

| 对照 | λ左/λ右 | PiQA acc_norm | PiQA acc | HellaSwag acc_norm | HellaSwag acc |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(pair_lines)+'''

M−B4是本批唯一同λ=5的方法配对。其PiQA主评分在三个seed均为正，但条件题目区间均跨零；HellaSwag主评分均值接近零、三个区间跨零，辅助评分还有seed方向变化，仍不支持M在两任务、两评分上统一优于B4。C−B4的两任务、两评分在三个seed均为正，但λ为1对5，不能归因于固定教师约束强度下的选库净效应。B2−B1与B4−B3的HellaSwag两评分在三个seed均为负、条件区间均低于零；由于双方λ不同，也不能解释为固定λ下KL普遍差于CE。来源配对B3−B1、B4−B2同样保留实际λ与评分条件，不据此作跨设置的来源排序。

全部个体端点与条件区间见[acc_norm个体图](results/supplement_A_behavior_presented_v1/endpoints_acc_norm.png)、[acc个体图](results/supplement_A_behavior_presented_v1/endpoints_acc.png)；六策略配对见[acc_norm图](results/supplement_A_behavior_presented_v1/selected_policy_pairs_acc_norm.png)、[acc图](results/supplement_A_behavior_presented_v1/selected_policy_pairs_acc.png)；唯一同λ配对另见[M−B4 acc_norm图](results/supplement_A_behavior_presented_v1/fixed_lambda_pairs_acc_norm.png)、[acc图](results/supplement_A_behavior_presented_v1/fixed_lambda_pairs_acc.png)。[全部18端点可读表](results/supplement_A_behavior_findings_v1/全部18端点可读表.md)及[全部六策略逐seed配对表](results/supplement_A_behavior_findings_v1/全部六策略配对可读表.md)保留准确率变化、CW/WC及条件遗忘/获得率；净准确率变化不代表所有旧正确行为都被保留。

这些结果不倒改上文A的NLL结论或验证选择。三条seed训练路径同时参与了λ选择，样本SD不是三次独立重复调参的波动，也不估计新seed重做整个选择流程的泛化变化。原5,000次题目bootstrap条件于既定检查点；HellaSwag同activity/context的依赖未校正。多方法、任务、评分、seed和配对的逐项95%区间未经多重比较校正，不保证同时覆盖。这里没有新增统计检验、行为调参或新的推理能力主张。[完整行为讨论](results/supplement_A_behavior_findings_v1/讨论.md)、[按评分分别复核的策略统计CSV](results/supplement_A_behavior_independent_review_v1/selected_policy_score_resolved_statistics.csv)。

'''
s=before.decode('utf-8'); original=s
replacements=[
 ('**研究工作稿。预定科学训练、主NLL和主行为评价已完成，最终摘要与结论将在其余预定评价完成后填写。**','**研究工作稿。预定科学训练、主NLL、主行为及补充A行为评价已完成，最终摘要与结论将在其余预定评价完成后填写。**'),
 ('**主实验、补充A/B及长程C的预定科学训练已全部完成，主NLL与主行为结果已通过完整核验；其余评价继续按冻结方案分阶段执行。**','**主实验、补充A/B及长程C的预定科学训练已全部完成，主NLL、主行为与A行为结果已通过完整核验；其余评价继续按冻结方案分阶段执行。**\n\n已开放的A行为结果表明，验证NLL选出的λ不保证选择题准确率保留：HellaSwag主评分在所选B2/B4/M策略中下降、B1/B3/C中提高。该观察不改变NLL选择，也不能把不同λ的政策对照解释为固定λ下的CE/KL因果效应（第6.1.1节）。'),
 ('主行为结果见第5.5节；其余C模型NLL与补充行为结果仍须各自完整核验后报告。','主行为结果见第5.5节，验证选λ后的A行为结果见第6.1.1节；其余C模型NLL与行为结果仍须各自完整核验后报告。'),
 ('1B和2.8B训练均已完成，其NLL及C各模型行为结果仍封存；补充A行为的原分析已完成，待独立复核，本版不报告这些结果。主行为结果见第5.5节。局部表示覆盖、输出KL、文本NLL和选择题正确率描述不同对象；后续仅填入真实完成且通过完整性门禁的结果。','1B和2.8B训练均已完成，其NLL及C各模型行为结果仍待各自完整核验，本版不报告这些结果。主行为与已复核的A行为分别见第5.5和6.1.1节；A中按验证NLL选择的策略没有形成行为保留保证。局部表示覆盖、输出KL、文本NLL和选择题正确率描述不同对象，后续仅填入真实完成且通过完整性门禁的结果。'),
 ('主行为结果见第5.5节，补充行为评分将在各自整阶段完成并核验后报告。','主行为与A行为结果分别见第5.5和6.1.1节，其余补充行为评分将在各自整阶段完成并核验后报告。')]
for old,new in replacements:
 assert s.count(old)==1,old;s=s.replace(old,new)
anchor='### 6.2 固定生成池和训练流下的选库波动'
assert s.count(anchor)==1;s=s.replace(anchor,insert+anchor)
# Original A-NLL contents, main behavior, B, C numerical results and delivery remain exact.
def section(text,start,end):return text.split(start,1)[1].split(end,1)[0]
assert section(original,'### 6.1 完整权重曲线与验证选择的独立测试','### 6.2 固定生成池和训练流下的选库波动')==section(s,'### 6.1 完整权重曲线与验证选择的独立测试','#### 6.1.1 验证选λ策略的行为评价')
assert section(original,'### 5.5','## 6.')==section(s,'### 5.5','## 6.')
assert section(original,'### 6.2','### 6.3')==section(s,'### 6.2','### 6.3')
assert section(original,'## 7.','## 8.')==section(s,'## 7.','## 8.')
after=s.encode('utf-8')
for link in re.findall(r'\]\(([^)]+)\)',insert):assert (ROOT/link).is_file(),link
write_new(OUT/'before_研究报告.md',before)
write_new(OUT/'A_behavior_insert.md',insert.encode('utf-8'))
write_new(OUT/'report.diff',''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='before/研究报告.md',tofile='after/研究报告.md')).encode('utf-8'))
assert report.read_bytes()==before
report.write_bytes(after)
receipt={
 'schema_version':1,'status':'completed_A_behavior_MD_integration_only','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'report_before':{'sha256':sha(before),'bytes':len(before)},'report_after':{'sha256':sha(after),'bytes':len(after)},
 'source_acceptance':sources[0],'sources':sources,'formatted_values':rendered,
 'changes':['status and NLL-versus-behavior interpretation','new 6.1.1: all six selected policies, two tasks, two scores, three-seed mean/sample SD','all six selected-policy paired differences and all six accepted figure links','pending-status text updated for A only'],
 'invariants':{'A_NLL_section_byte_identical':True,'primary_behavior_section_byte_identical':True,'B_section_byte_identical':True,'delivery_section_byte_identical':True,'no_new_statistics':True,'no_behavior_based_reselection':True,'no_C_behavior_or_C1_C2_NLL_semantic_reads':True,'no_PDF_HTML_render_or_mutation':True,'no_checkpoint_base_archive_payload_read':True,'C2_restore_not_claimed_completed':True},
 'original_label_collision':'Original selected_policy_statistics.csv not used; score-resolved accepted table used.',
 'measurement':{'script_body_region_wall_seconds_until_receipt':time.perf_counter()-started,'whole_manual_authoring_review_seconds':None,'scope':'small accepted A CSV/Markdown/proof/figure SHA and Markdown edit; formatting scales existing statistics by 100, no bootstrap/statistical rerun'},
 'producer_source':proof(Path(__file__).resolve())}
write_new(OUT/'receipt.json',(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print(json.dumps({'report_after_sha256':sha(after),'receipt':proof(OUT/'receipt.json'),'sources':len(sources),'formatted_values':len(rendered)},ensure_ascii=False))
