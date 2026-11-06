# A梯度裁剪：看过A测试后的描述性诊断

本诊断已在完整A测试解封后声明，依据[固定事后协议](A_gradient_clipping_posttest_protocol_v1.json)，不属于原预注册机制检验。它只读取已完成A网格的config、summary和训练events，未打开任何test数据/score输出、补充C结果或behavior结果，也没有新增训练、forward、选库或选参。A内的方法C指cluster选库，不能与扩展模型补充C混淆。

**最直接的结果是所有条件、所有固定窗口的阈值超越率均为100%。** 全126条件×306步，共38,556个原训练步，logged norm都严格大于各自实际配置阈值1。all/early/late每run分别306/61/62步，378个run-window的比例全部1；42个method×λ条件的三seed均值为100%、sample SD为0，三个窗口都如此。early按update后目标token≤2M，late按>8M划分；这些窗口复用all中的步，不能把窗口计数相加当成更多独立观察。

## 字段含义和完整性

原训练源码SHA固定，在`train.py:645`调用`clip_grad_norm_`，在`:664`直接记录返回值。captured installed Torch实现先计算`total_norm`，再in-place缩放梯度，最终返回此前的`total_norm`，所以logged值是**裁剪前总L2范数**。实际缩放系数包含`max_norm/(total_norm+1e-6)`；本分析预先固定的统计是严格`norm > max_grad_norm`的**阈值超越率**，不把它改称epsilon附近精确浮点clip activation。

全部126个实际run均验证单一start、完整step1–306、单一completed finish、summary/config/start一致、每步六类token counter增量和累计闭合、最终10,000,000目标/500,000回放预测token，以及固定installed Torch版本。出现缺步、重复、乱序、resume/stop/error、非有限norm、配置阈值不为1或曝光不完整时脚本拒绝，不支持partial。来源计数为126份config＋126份summary＋126份events＋5份协议/源码/范围材料，合计383份、70,614,680个source字节；均在新目录保存原字节和SHA。

范围中的18项λ=.05引用此前主实验训练，另外108项来自已完成A训练。这里追溯126个unique actual_run_id，18个引用不新增物理训练、不增加独立seed或费用。本任务新增训练/推理数均为0。每个method×λ仍只有11/22/33三个pipeline seed；同seed跨λ共享训练/池/选库随机来源，306个相关更新也不是306个独立实验样本。

## 范数幅度：所有七个λ全部保留

下表为每run先对306步求平均，再对三个pipeline seeds求mean±sample SD（ddof=1）。它描述已形成的训练轨迹，不是把全部steps混成独立重复。完整378行个体窗口表同时保留阈值频率、mean、median、线性插值p90和max；分位数定义为在`(n−1)p`处线性插值，不再临时更换指标或窗口。

| λ | B1 real CE | B2 real KL | B3 generated CE | B4 generated KL | M stratified KL | C cluster KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 2.445 ± 0.074 | 2.809 ± 0.237 | 2.660 ± 0.228 | 3.288 ± 0.063 | 2.674 ± 0.032 | 2.702 ± 0.197 |
| 0.05 | 2.891 ± 0.100 | 2.962 ± 0.369 | 2.952 ± 0.243 | 2.942 ± 0.421 | 2.965 ± 0.150 | 2.582 ± 0.102 |
| 0.1 | 3.372 ± 0.572 | 2.744 ± 0.036 | 3.386 ± 0.097 | 2.769 ± 0.184 | 2.751 ± 0.257 | 2.912 ± 0.168 |
| 0.5 | 9.373 ± 1.760 | 3.215 ± 0.287 | 10.612 ± 1.510 | 3.856 ± 0.576 | 3.576 ± 0.436 | 3.730 ± 0.086 |
| 1 | 17.928 ± 2.780 | 3.953 ± 0.290 | 30.738 ± 3.921 | 4.764 ± 0.443 | 5.417 ± 1.307 | 5.936 ± 0.327 |
| 5 | 128.951 ± 22.688 | 15.087 ± 2.157 | 134.648 ± 14.029 | 21.285 ± 1.317 | 23.019 ± 5.380 | 31.511 ± 5.968 |
| 10 | 222.714 ± 51.458 | 28.432 ± 1.619 | 237.076 ± 43.198 | 52.429 ± 16.043 | 41.311 ± 2.607 | 53.100 ± 6.690 |

较大λ区域通常伴随更大的裁剪前范数，CE两方法B1/B3尤其明显；KL方法也在较大λ时增加。不过这不是全域单调关系，例如B4的三seed全程均值在λ=.01/.05/.1分别为3.288/2.942/2.769。所有七λ和三个窗口均原样保留，没有按这种图形重新选择权重或追加检验。

![固定A网格的共同阈值信息与三窗口裁剪前平均范数](../results/A_gradient_clipping_posttest_v1/presentation_v2/figure/gradient_diagnostic_v2.png)

v2展示图宽7in、高6.6in：顶部共享信息框明确所有126run/38,556step及all/early/late三个窗口全部100%、每run步数306/61/62；下方三个满宽面板依次展示all/early/late裁剪前平均范数。三面板统一y轴范围0.85–500（log），λ也是log轴，六method×七λ×三个seed点及既有三seed mean±sample SD全部保留；虚线为配置阈值1。图内注明post-test descriptive与不能据total norm推Adam更新/因果机制。PDF/SVG配套保存，数字CSV未变。

本次只读取既有CSV及原输出的完整性字节，不读取原logs或重算任何统计。独立v2绘图child完整wall 1.923686s、user 1.969755s、sys 0.487787s，另计于原分析和独立算术复核之后；后续文档编辑/人工看图不在该child计时内。原analysis、三格式v1图、COMPLETE及383份source bytes原样保留（后者仅核五stat、未重读内容）。[旧文档原字节](../results/A_gradient_clipping_posttest_v1/presentation_v2/document_before_v2.md)及[展示修订收据](../results/A_gradient_clipping_posttest_v1/presentation_v2/change_receipt.json)保存本次展示变更。

**解释边界：** 激活频率这一诊断在本固定网格没有区分度，不能据此说“只有大λ才开始裁剪”或用频率解释A曲线。范数随λ共变也不能说明Adam实际参数更新同等放大、某种梯度方向更优或已证明遗忘机制；裁剪、目标混合、参数轨迹与Adam历史共同存在。仅凭total norm不能恢复目标/回放梯度各自大小、夹角、Adam有效步长或反事实无裁剪结果。没有新增p值、bootstrap、等效检验或选参；此前A固定同λ配对和验证选择政策比较保持原定义。

## 计时、复核和复用

完整分析child从spawn前至exit的实测为wall **4.423583s**、user **4.276988s**、sys **0.828320s**。它包含导入、来源核验/原字节快照、完整日志统计、CSV和三格式图写出。内含来源核验/parse/snapshot 1.125926s、统计/表格0.224120s、绘图2.820545s，不能再加到父wall。两线程使CPU user＋sys可超过wall，wall不是GPU时间。

另一次独立NumPy CSV算术复核执行3,906项比较，最大绝对差2.84e-14，完整独立child wall 0.363190s（user 0.336104s，sys 0.136444s），发生在首次分析之后，不能说包含于4.42s。复核极值只作“全部严格>1”的工程检查，不增加科学指标。手工算例/窗口边界/严格不等式/样本SD/alias计费/缺步等14个CPU测试通过。

- [个体run-window表](../results/A_gradient_clipping_posttest_v1/analysis/per_run_windows.csv)、[全部38,556步](../results/A_gradient_clipping_posttest_v1/analysis/all_train_steps.csv)、[42条件×3窗口汇总](../results/A_gradient_clipping_posttest_v1/analysis/method_lambda_windows.csv)。
- [逐run来源与事件数](../results/A_gradient_clipping_posttest_v1/analysis/run_source_manifest.json)、[完成与来源摘要](../results/A_gradient_clipping_posttest_v1/analysis/COMPLETE.json)。
- [完整父计时](../results/A_gradient_clipping_posttest_v1/whole_process_timing.json)、[独立算术复核](../results/A_gradient_clipping_posttest_v1/independent_numeric_review.json)。
- [既有物理训练父计时](../results/A_gradient_clipping_posttest_v1/analysis/existing_physical_training_parent_timers.csv)仅供溯源；不把主18再计新费，不把setup/train/validation/checkpoint父项与teacher子项双加，也不把并行parent timer之和称研究elapsed。

可复查命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 pytest -q tests/test_A_gradient_clipping_posttest.py`。脚本入口为`scripts/analyze_A_gradient_clipping_posttest.py --output-dir NEW_EXCLUSIVE_DIR`，必须新独占目录，并用已记录的父进程计时方式运行；本次输出和来源SHA不覆盖旧主/A统计或研究报告。
