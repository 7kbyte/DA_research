# 补充B：固定pool与训练seed下，完整60-run验证结果

本报告在`logs/auxiliary_validation_analysis_watch.json`的B状态为complete后才读取验证派生输出。完整cohort分析于2026-11-06 11:02:40 UTC完成；60/60、无partial，analysis manifest SHA为`03fd9622e0e56478dbeecf9d800ab8635cf5770255e6093b07b1f8e86bef389d`，全部输出SHA已独立核验。来源为`results/analysis/supplement_B_validation_complete_v1/`，协议SHA `dea9851ec294bc04f10f867380c3f855ba9c5c3395416062d1ab10062fa312bd`。只读验证分析；未读取v5 READY的测试结果、B/C heldout或任何behavior分数，未启动评价、训练或GPU，未改变选择。

## 重复单位和预算

B4=random、M=teacher-NLL五分位分层、C=cluster representative。三方法×两种K（20、196 chunks）×10个selection seeds（1001–1010）共60个物理训练run。**训练shuffle seed始终11，generated pool seed始终11；这不是十个独立训练seed，更不是三seed主实验的扩充。** 每种K内描述的是固定pool、固定训练顺序和既定实现下，bank draw变化造成的条件分布；不能外推跨pool或完整训练随机性。

各run固定final10M target/.5M replay prediction-token曝光，b=.05、原冻结λ=.05、LR=3e-5；没有把A事后所选λ=5搬来改B。K20容量10,220预测token，K196容量100,156；同为.5M累计replay，相对bank容量平均约48.9237与4.9922次曝光。K比较改变了唯一bank容量及重复利用程度，不能叫“相同K”。本次解释不重读训练summary，实际完整曝光与严格数值policy已由完整cohort分析绑定的源summary/config门禁验证。

下表来自既有完整分析CSV，所有误差为10个selection draw的样本SD（ddof=1），不是SEM、置信区间或跨training-seed不确定性。G=initial−adapted目标验证NLL，越大越好；F=adapted−initial通用验证NLL，越小越好，单位nats/token。

## 方法×K的验证端点

|K chunks|方法|G_new ↑|F_C4 ↓|F_Wiki ↓|
|---:|---|---:|---:|---:|
|20|B4|+0.018148 ± 0.011024|+0.169066 ± 0.011285|+0.167778 ± 0.010665|
|20|M|+0.014987 ± 0.012036|+0.170009 ± 0.013761|+0.171774 ± 0.018568|
|20|C|+0.019044 ± 0.013470|+0.172590 ± 0.016862|+0.171103 ± 0.017174|
|196|B4|+0.022062 ± 0.009276|+0.150030 ± 0.009553|+0.145548 ± 0.006782|
|196|M|+0.025830 ± 0.009933|+0.145885 ± 0.009805|+0.140412 ± 0.007806|
|196|C|+0.020971 ± 0.010455|+0.158056 ± 0.010592|+0.148010 ± 0.009799|

在K196，M的均值较B4表现为目标增益高.003767、C4/Wiki遗忘低.004145/.005135；在K20，M的均值方向恰为目标增益低.003160、两域遗忘高.000943/.003996。C在K20的目标均值略高但两域遗忘也更高，K196则目标均值略低且两域遗忘更高。它们是这一个pool和一个training seed下的有限draw描述，不是跨训练seed的一般优势，也不是改变之后实验条件的依据。

## 同selection-seed标签的配对差与方向

每个K内先按同一selection-seed标签做M−B4、C−B4，再在10个差上取均值和sampleSD；不是把两个方法的SD相减，也不将两套方法当20个独立replicates。不同算法即使seed标签相同，也不保证消耗相同随机数或选到同一bank，不能称严格common-random-number对照。40个配对均保留。

|K chunks|同标签配对（左−右）|ΔG_new ↑|ΔF_C4 ↓|ΔF_Wiki ↓|
|---:|---|---:|---:|---:|
|20|M-B4|-0.003160 ± 0.020506|+0.000943 ± 0.023057|+0.003996 ± 0.024576|
|20|C-B4|+0.000896 ± 0.017917|+0.003525 ± 0.020930|+0.003326 ± 0.020785|
|196|M-B4|+0.003767 ± 0.013791|-0.004145 ± 0.012622|-0.005135 ± 0.009073|
|196|C-B4|-0.001091 ± 0.011708|+0.008026 ± 0.011021|+0.002463 ± 0.009982|

以下只是逐draw差的方向计数，不是二项检验、胜率总体置信区间或显著性证据；本例无精确零差。G正、F负记为左方有利。

|K|配对|G方向有利/不利|C4方向有利/不利|Wiki方向有利/不利|
|---:|---|---:|---:|---:|
|20|M-B4|5/5|7/3|5/5|
|20|C-B4|6/4|4/6|4/6|
|196|M-B4|7/3|7/3|8/2|
|196|C-B4|5/5|3/7|4/6|

K196的M−B4在G/C4/Wiki分别7/7/8个标签有利，仍保留3/3/2个反向值；三个差的SD分别.013791/.012622/.009073，不能由均值改善写成“各次都改善”。K20的C4差虽7/10有利，均值却为+.000943：少数较大正差可抵消多数小负差。selection_seed1008的M−B4为G−.048909、C4+.049347、Wiki+.027648，完整保留，不按结果剔除或重抽。均值、样本SD及个体方向各描述不同方面。

## 库组成波动与保留不能等同

teacher-NLL组成的下表同样是10个draw的**bank mean NLL**的均值±sampleSD，不是每个库内部token NLL的SD：

|K|B4库均值teacher NLL|M库均值teacher NLL|C库均值teacher NLL|
|---:|---:|---:|---:|
|20|+3.252748 ± 0.147085|+3.253885 ± 0.075304|+3.355458 ± 0.078715|
|196|+3.293839 ± 0.053865|+3.244875 ± 0.018086|+3.097623 ± 0.029696|

M在这两个K的bank mean teacher-NLL波动均小于B4，但本样本里M的G/C4/Wiki端点SD在两个K均未小于B4。因而“分层使某个组成统计更稳定”不自动推出“使适应/遗忘结果更稳定”，也不证明梯度约束或保留机制。C的teacher-NLL均值和bank特征覆盖也只是组成/几何诊断，不是heldout retention替代指标。

这些验证导出虽保留equal-document字段，但本批该字段为空，本报告只用现有token聚合NLL及其差值，没有补造等文档结果或借A测试分数替代。既有CSV还保留teacher-KL drift，但KL不替代NLL或行为能力结论。本报告不新增bootstrap、p值、等效检验、选库或调参。

## 全个体、费用及可追溯证据

- [60个验证端点](../results/analysis/supplement_B_validation_complete_v1/validation_endpoints.csv)、[36行六指标统计](../results/analysis/supplement_B_validation_complete_v1/endpoint_statistics.csv)。
- [原40个配对](../results/analysis/supplement_B_validation_complete_v1/paired_validation.csv)、[原配对统计](../results/analysis/supplement_B_validation_complete_v1/paired_validation_statistics.csv)。
- [独立可读40配对及来源run ID/方向](../results/supplement_B_validation_interpretation_v1/paired_selection_seed_directions_40.csv)、[12行方向计数](../results/supplement_B_validation_interpretation_v1/direction_counts_12.csv)。
- [7英寸完整验证frontier PDF](../results/analysis/supplement_B_validation_complete_v1/conditional_validation_frontier_7in.pdf)（同目录PNG/SVG），各点是selection draw、误差棒是draw SD。
- [观察到的60run费用](../results/analysis/supplement_B_validation_complete_v1/observed_run_costs.csv)：committed-path父组件之和20,520.290秒，total_accounted之和20,528.041秒；这是逐run墙钟之和，**不是研究elapsed或纯GPU秒**。teacher子计时已含于训练/验证父项，不能再相加。共享pool生成/联合stats准备也不乘以60，独立选库费用另见bank组成表。

解释阶段独立核对240个原配对差及60行现成mean/SD与对应个体值的代数一致性，未运行原训练/统计pipeline；数值误差均在1e-12内。门禁快照、输入/输出SHA及该CPU解释操作的独立wall记录见[review receipt](../results/supplement_B_validation_interpretation_v1/review_receipt.json)。报告只写完整B验证，其他测试/behavior保持封存。
