# 补充 A：完整验证网格与选权重解释

本次独立复核只读取验证集与训练日志。126 个逻辑条件、42 个方法×λ 候选及 370 个绑定来源文件均通过 SHA 核对；逐条件从最终验证端点重新计算九项 NLL/KL/增益指标，42 个分数均值、样本标准差及六个选定 λ 与冻结产物一致。108 项为新增训练，λ=.05 的 18 项直接引用主实验实际 run；引用不构成新增训练或独立重复。

主实验共享 λ 与补充 A 回答不同的决策问题：

| 设计 | 主实验共享 λ 校准 | 补充 A 逐方法校准 |
|---|---|---|
| 校准曝光 | 最终 2M target / 100K replay | 最终 10M target / 500K replay |
| 训练 seed | 101 | 11、22、33 |
| λ 网格 | .05、.5、1、5 | .01、.05、.1、.5、1、5、10 |
| 优化对象 | 同一 λ 下 B1–B4 的等权平均分数 | 每种方法内三 seed 的等权平均分数 |
| 用途 | 固定共享设置下比较方法，M/C 同样沿用该 λ | 比较完整网格中各方法的验证选参策略 |

两者均使用 (S=\Delta L_{target}+.5(\Delta L_{C4}+\Delta L_{Wiki})=-G_{new}+.5(F_{C4}+F_{Wiki}))，越小越好；只在浮点分数严格相等时选较小 λ。主校准选择 .05，其 B1–B4 平均 S 为 −.031109。但同一批 2M pilot 内，各方法的网格最小值已分别位于 B1=.5、B2=1、B3=.05、B4=1。因此，共享 .05 从未意味着每种方法各自最优。主正式训练随后统一执行 10M；A 才在该完整 horizon 上逐方法优化。horizon、seed、网格及聚合规则同时变化，不能将差异单独归因为训练变长。

| 方法 | A 选定 λ | 选定 S：均值±SD，n=3 | 同方法 λ=.05 的 S |
|---|---:|---:|---:|
| B1 | .05 | +.172107±.010638 | +.172107 |
| B2 | 1 | −.025193±.003160 | +.094885 |
| B3 | .1 | +.165705±.008835 | +.173462 |
| B4 | 5 | −.045756±.001344 | +.116205 |
| M | 5 | −.041864±.007244 | +.106155 |
| C | 1 | −.041111±.001931 | +.146525 |

在本验证指标与候选范围内，KL 方法在较大权重处得到更好的最终适应—保留折中；这不代表更大 λ 普遍更优，也不代表没有遗忘。选定配置的 C4 与 Wiki 平均 F 仍均为正，负 S 只是加权目标的净改善。b 固定 .05 时，λ/b 表示损失中单个回放预测 token 相对目标 token 的系数；它不等于 CE 与 KL 的梯度强度。

所有 42 个候选和 126 个逐 seed 端点均保留。[原同 λ 配对表](../results/supplement_A_validation/paired_same_lambda.csv)比较固定权重下的来源或目标函数差异；另存的[逐方法选 λ 配对表](../results/supplement_A_validation_review/selected_policy_paired_validation.csv)保留两侧 λ 与实际 run ID，比较的是完整验证选参策略，不能解释为仅改变来源或损失函数的效果。误差条为三个训练 seed 的样本 SD，不包含搜索选择不确定性；在同一验证集选 λ 并报告最小分数存在选择乐观偏差，尚无 A heldout 或行为表现结论。

A 网格及规则于 04:44:34 UTC 预注册，完整选择于 09:16:21 UTC 冻结；后者晚于主 NLL 的首次解封时间，不应写成所有 A 选择均先于主测试。选择程序及本次复核没有读取 A heldout/行为分数，也没有利用已公开的主测试分数改变规则。方法 C 的 A 选择为 1，与补充实验 C 不是同一概念；补充 C 已冻结的 λ=.05 继续保持，不追溯修改主实验。

历史执行证据有一项边界：16 个共享 λ pilot 的旧 provenance 未记录第四个 cuDNN SDP 开关。本复核检查了其余已记录策略，但不声称独立补证该缺失字段。协议保存了 04:41 的初始评估 backend 修订及三个语料修订前后全部 NLL/KL/token 统计完全相同的证据。A 的 126 个正式端点则逐项通过完整执行策略匹配；没有放宽正式检查。

独立复核收据为 [independent_audit.json](../results/supplement_A_validation_review/independent_audit.json)，逐条件复算、全候选及三种打印图均在同目录；图 [PNG](../results/supplement_A_validation_review/all42_lambda_validation_scores_7in.png)、[PDF](../results/supplement_A_validation_review/all42_lambda_validation_scores_7in.pdf)、[SVG](../results/supplement_A_validation_review/all42_lambda_validation_scores_7in.svg) 仅含完整 A 验证结果。冻结选择 SHA256：`9548085fc18278f69fad584b67485bceb5193a18b10c78f516f6b09b4fa4e477`；主共享选择 SHA256：`328dfda238700df8df3d58d1da95046e8b7d19a01173ebdae7aca531315bccda`。
