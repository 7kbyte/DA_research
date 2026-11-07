# 补充A：source×objective交互的事后探索性描述

这是**看过主实验及A的测试结果之后**声明的新增探索性分析，不是预注册的验证性结论。声明时间2026-11-06 10:38:07 UTC，协议[posttest_factorial_interaction_A_protocol.json](posttest_factorial_interaction_A_protocol.json) SHA为`18de2ca4e8ff45feebb69e9499ba3bde17e01f58a086ac831dcda183a4dd4187`。本分析只重组已明确授权的A派生CSV，不重新运行主/A统计，不访问B/C或行为分数，不启动训练或模型forward，不改变六个预定配对、validation-selected政策比较或后续λ。

## 定义、配对和方向

在每个固定λ、同一seed内，令B1=C4 CE、B2=C4 KL、B3=generated CE、B4=generated KL，计算：

`I(m) = (m_B4 − m_B3) − (m_B2 − m_B1)`。

它是“generated来源的KL−CE差”减去“C4来源的KL−CE差”，也等于`(B4−B2)−(B3−B1)`。相加响应的该对比为零，但样本中非零点差本身不是特定机制的证明。这里的source/objective是完整训练政策及其有限bank的对照，不是对内部机制做了直接干预观测。

原始指标`G_new = NLL_initial − NLL_adapted`越大越好；`F = NLL_adapted − NLL_initial`越小越好。因此正I(G)表示KL相对CE的目标增益差在generated来源下更大；负I(F)表示KL相对CE的遗忘差在generated来源下更小。这些方向都只针对**差中之差**，不能直接判断任何单个方法的总体表现。单位为nats/token；等文档量是各文档内部先做token平均、再对文档等权。

全部四端点都为final、实际10,000,000 target及500,000 replay prediction tokens，b=.05、LR=3e-5；没有混入best端点或各方法所选的不同λ。保留全部七个共同λ及seed11/22/33。84个被引用端点中，λ=.05的12个B1–B4端点复用已有主实验运行，不新增12次训练或任何科学重复。

## 全部固定λ的描述值

以下各格为同一λ下三个**配对后的交互值**的均值±样本SD（ddof=1，n=3），不是由四个方法SD相加所得，也不是置信区间。这里不新增bootstrap、p值、显著性或等效检验。

Token加权：

|固定λ|I(G_new) ↑|I(F_C4) ↓|I(F_Wiki) ↓|
|---:|---:|---:|---:|
|0.01|-0.009598 ± 0.031585|+0.003599 ± 0.037055|+0.010521 ± 0.022332|
|0.05|-0.002158 ± 0.012707|+0.021665 ± 0.016372|+0.012116 ± 0.010346|
|0.1|-0.014961 ± 0.015182|+0.068633 ± 0.040670|+0.053401 ± 0.030897|
|0.5|+0.003769 ± 0.006817|+0.145297 ± 0.058829|+0.101691 ± 0.045251|
|1|+0.016608 ± 0.004288|+0.186293 ± 0.069762|+0.146238 ± 0.046646|
|5|+0.002356 ± 0.009719|+0.026539 ± 0.046256|+0.044730 ± 0.024659|
|10|-0.002997 ± 0.007619|+0.033126 ± 0.036333|+0.045227 ± 0.033770|

等文档权重：

|固定λ|I(G_new) ↑|I(F_C4) ↓|I(F_Wiki) ↓|
|---:|---:|---:|---:|
|0.01|-0.009974 ± 0.029646|+0.003460 ± 0.035983|+0.014885 ± 0.019925|
|0.05|-0.001522 ± 0.012002|+0.022339 ± 0.016333|+0.011597 ± 0.010509|
|0.1|-0.013616 ± 0.014430|+0.068748 ± 0.040828|+0.048545 ± 0.032247|
|0.5|+0.005039 ± 0.005744|+0.147146 ± 0.060226|+0.095812 ± 0.044992|
|1|+0.019044 ± 0.006034|+0.188874 ± 0.069535|+0.138833 ± 0.046108|
|5|+0.002436 ± 0.011263|+0.025012 ± 0.044880|+0.040335 ± 0.020838|
|10|-0.002725 ± 0.009933|+0.033217 ± 0.037646|+0.039125 ± 0.036000|

两种权重回答不同估计问题，本例各固定λ下的交互均值符号一致，但数值及个体差可不同。例如λ=10、seed33的Wiki交互为token加权+.008940、等文档−.001493，不能把二者当两次独立训练或互相替代。

以既有主设定λ=.05作为算术解释：generated来源的KL−CE均值为G +.001213、C4遗忘−.046606、Wiki遗忘−.062191；C4来源相应为+.003372、−.068271、−.074307。两种来源的遗忘差都为负，故这里正的遗忘交互+.021665/+.012116表示generated来源下的KL−CE遗忘降低幅度较小，**不是KL增加了两个来源的遗忘**。目标交互−.002158的三个seed值为+.012474/−.008532/−.010417，直接保留了方向不一致。

全表中目标交互随λ及seed而变化；两个遗忘交互的七个均值均为正，但并非所有个体值均为正。图中全范围均保留，仅作敏感性描述，不按测试结果选λ、不将某个非零均值升级为机制结论或总体retention结论。

## 依赖关系与解释边界

同一seed下，B1/B2共享C4 bank，B3/B4共享generated bank和pool，且使用共同目标流；七个λ复用这些随机来源与评价文档。因此每个λ的n=3是三个pipeline seed的配对描述，**七个设置不是21个独立训练重复**，六种指标也不是六组独立实验。这里的样本SD不代表重新采集语料、重新选择λ或重复完整研究的置信区间；只有三个seed也不足以证明交互接近零时等效。无新增文档抽样区间，不能借用原selected-policy的条件bootstrap区间作为本交互的不确定性。

这是同λsource×objective交互，与“每个方法各自验证选λ后的政策比较”是不同问题。所选B1/B2/B3/B4的λ不同，不能代入本固定λ2×2再解释为source/objective交互。本次描述不更改已冻结选择，也不影响原分析规则。

## 可追溯产物与复核

- [84个原端点及所有来源字段](../results/posttest_factorial_interaction_A_v1/source_endpoints_84.csv)：保留record/run/actual_run/inference ID、alias身份、λ、seed、六指标及真实曝光。
- [126个逐seed×λ×指标交互](../results/posttest_factorial_interaction_A_v1/interaction_individual_126.csv)：每行含四个原值、两项KL−CE差、交互及四端点来源/曝光。
- [42行均值±样本SD](../results/posttest_factorial_interaction_A_v1/interaction_mean_sample_sd_42.csv)：七λ×六指标。
- [独立代数核对](../results/posttest_factorial_interaction_A_v1/algebra_consistency_126.csv)：先核现有四种同λpaired effects与端点差，再核两种重构。共756个标量等式，容差1e-12；最大绝对差4.163336342344337e-17。
- [7×6.6英寸图 PNG](../results/posttest_factorial_interaction_A_v1/posttest_factorial_interaction_A.png)、[PDF](../results/posttest_factorial_interaction_A_v1/posttest_factorial_interaction_A.pdf)、[SVG](../results/posttest_factorial_interaction_A_v1/posttest_factorial_interaction_A.svg)：全部λ和三个seed，mean±sampleSD，零线与方向标识；线只连接设定，不表示独立重复。
- [执行manifest](../results/posttest_factorial_interaction_A_v1/analysis_manifest.json)：绑定声明、派生输入、独立脚本与全部输出SHA。

输入endpoints.csv SHA为`211b2cae2bf81f2b25902061797762f12dd61083c3aead5f8573e96101d5ef51`；现有paired_effects.csv SHA为`6b91081526b28a8b01e6de9b60ba9a0c752bd019995afb0488079f507d881f6c`。脚本只读取这两份已授权派生表与声明JSON；未读取checkpoint、模型权重或封存结果。

复现本独立计算入口（当前输出已存在时拒绝覆盖，原始执行产物保持不动）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /usr/bin/python scripts/analyze_posttest_factorial_A.py \
  --protocol docs/posttest_factorial_interaction_A_protocol.json
```

14个合成CPU测试通过，覆盖相加响应为零、三个配对seed的sampleSD、两种代数恒等式、缺项/重复/不同λ、不同曝光/模型端点/基线身份、混seed/source ID及非有限值拒绝。没有修改冻结scorer、wrapper、selection或主统计模块。

本次新增CPU算术与绘图wall为1.616210334秒，独立于旧A分析39.654299秒与原GPU推理占槽费用；没有新增GPU工作。该CPU计时不包含本解释文档的人工审阅与写作时间，未计部分不能当0。
