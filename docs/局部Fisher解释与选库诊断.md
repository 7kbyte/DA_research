# 冻结教师约束的局部解释与回放库诊断

本文可作为报告的方法解释和机制诊断小节。数学部分是带假设的局部推导；数值部分只来自冻结生成池、库组成和准备成本，未使用模型测试损失或行为分数，也不据此重新选择方法。

## 固定前缀分布下的局部二次形式

令 \(c\) 为因果预测上下文，\(\mu(c)\) 为与学生参数无关的固定前缀分布。它可以是教师生成分布，也可以是固定库中有效预测位置的经验分布。设理想实数精度模型满足 \(q_0(v\mid c)=p_{\theta_0}(v\mid c)>0\)，并在 \(\theta_0\) 邻域二阶连续可微；假定导数与期望可以交换。定义冻结教师约束

\[
R_\mu(\theta)=\mathbb E_{c\sim\mu}
  \sum_v q_0(v\mid c)\log\frac{q_0(v\mid c)}{p_\theta(v\mid c)}.
\]

记 \(s_0(v,c)=\nabla_\theta\log p_\theta(v\mid c)|_{\theta_0}\)。概率归一化给出 \(\sum_v q_0s_0=0\)，所以 \(R_\mu(\theta_0)=0\)、\(\nabla R_\mu(\theta_0)=0\)。再对归一化恒等式求导，得到

\[
\nabla^2R_\mu(\theta_0)
=-\mathbb E_{\mu,q_0}\nabla_\theta^2\log p_{\theta_0}(v\mid c)
=\mathbb E_{\mu,q_0}s_0s_0^\top
\equiv F_\mu\succeq0.
\]

因此，对足够小的参数位移 \(\delta\)，

\[
R_\mu(\theta_0+\delta)
=\tfrac12\delta^\top F_\mu\delta+o(\|\delta\|^2).
\]

只有进一步假设三阶导数在相关邻域可控，才可把余项写为 \(O(\|\delta\|^3)\)。这一信息几何解释与 [Martens（JMLR 2020）第5–6节](https://jmlr.org/papers/volume21/17-678/17-678.pdf)的 Fisher/KL 关系一致；这里专门对固定教师到学生的 KL 推导。两个 KL 方向在相同分布处共享二次项，并不意味着离开该点后方向可以互换。

若 \(z_\theta(c)\) 是词表 logits、\(J_0(c)=\partial z_\theta(c)/\partial\theta|_{\theta_0}\)，则同一矩阵还可写成

\[
F_\mu=\mathbb E_{c\sim\mu}
J_0(c)^\top\big[\operatorname{diag}(q_0)-q_0q_0^\top\big]J_0(c).
\]

这里对每个已选前缀仍对完整教师词表分布取期望。它不同于只把库中一次采样标签的 CE 梯度外积相加的“经验 Fisher”，也不同于参数对角重要性近似。本研究没有实际构造或分解这个巨大的矩阵，不把以下解释写成测得的 Fisher 谱。

同一公式还解释有限硬标签的局部噪声。固定前缀且 \(Y\sim q_0(\cdot\mid c)\) 时，初始点的 CE 梯度为 \(g_Y(c)=-s_0(Y,c)\)，故 \(\mathbb E[g_Y\mid c]=0\)、\(\operatorname{Cov}(g_Y\mid c)=F(c)\)；完整教师 KL 在该前缀的梯度则恰为零。对条件标签取期望消除了这一层采样噪声。这一逐前缀恒等式不能直接写成实际整库梯度协方差等于 \(F_B/K\)：自回归上下文相关、整序列选择和重复使用固定标签都需额外处理，NLL分层后保留标签的条件分布也不必仍是原教师。它支持“有限CE与完整KL的噪声不同”，不支持未经训练比较就断言哪种终点更优。

## 选库改变的是哪些方向

对 bank \(B\) 的有效预测位置 \(c_j\)，以实际 loss 的 token 权重 \(w_j\) 定义 \(\mu_B=\sum_jw_j\delta_{c_j}\)、\(\sum_jw_j=1\)。主库各块都有511个有效预测位置，故块均匀和预测 token 均匀相容。更换库会更换 \(F_B=\sum_jw_jF(c_j)\)，包括其强弱方向及其相互耦合；这不是简单地把同一个标量正则项变大或变小。

解释全库平均时可令各预测位置均匀；实际某一步则使用该步取到的前缀及其loss归一化权重。有限流的重复次数、尾部截断和逐步计数不一定让所有位置具有完全相同的累计权重。因此，全库 \(F_B\) 是固定库约束的局部参考，不冒充每一步实际随机曲率或最终累计曲率已经被测得。

若 \(F_Bu_i=\eta_i u_i\)，局部惩罚为 \(\lambda\sum_i\eta_i(u_i^\top\delta)^2/2\)。大 \(\eta_i\) 表示在这些前缀上，沿相应方向移动更容易改变教师预测；小特征值方向约束较弱。有限库可能留下弱约束或零方向，但不能仅凭“有限”就断言其 Fisher 必然低秩。库之外或罕见前缀上的重要方向也可能没有得到充分权重。

进一步把新域损失近似为 \(L_{new}(\theta_0)+g_{new}^\top\delta+\delta^\top H_{new}\delta/2\)。若合成二次矩阵正定，局部模型的极小点形式上满足

\[
\delta_*=-\big(H_{new}+\lambda F_B\big)^{-1}g_{new}.
\]

它说明效果还取决于目标梯度与受约束方向如何对齐。该式不是实际 AdamW 轨迹或最终参数的预测：真实训练包含多步非线性、预条件、裁剪、权重衰减与有限批次，较大位移会离开局部近似有效区间。重复处理同一库能改变优化噪声与暴露，却不会凭空补充库外前缀的约束方向；在当前分别求均值的 loss 中，增加 b 也不等于把平均 KL 权重自动乘上 b。

NLL 五层分组只按一个标量排序，不是按梯度或 Fisher 方向分组。一个直接反例是 \(p_\theta(1\mid c)=\sigma(a_c^\top\theta)\)、\(\theta_0=0\)：所有上下文、任一硬标签的教师 NLL 都等于 \(\log2\)，但 \(F(c)=a_ca_c^\top/4\)。取 \(a_{c_1}=e_1\)、\(a_{c_2}=e_2\)，相同 NLL 对应正交约束方向；缩放 \(a_c\) 还可改变曲率大小。类似地，最后层均值特征的欧氏距离不直接约束 logits 的参数 Jacobian。故保持 NLL 组成或降低特征距离，都不能推出 \(F_B\) 更接近全部旧域约束，更不能推出保留效果的单调改善。

![局部KL几何的解析示意](../results/method_illustrations/local_fisher_geometry.png)

图中只改变两个前缀的权重；轮廓为 \(\delta^\top F_\mu\delta/2\) 等值线，橙色箭头是相同参数位移 \((1,0)\)。三个测度的所有硬标签 NLL 都为 \(\log2\)，但对该位移的二次惩罚分别为0.1125、0.0625、0.0125。**这是二元logistic模型的解析示意，不是Pythia的Fisher估计、数据结果或保留实验。** [可复现脚本](../scripts/plot_local_fisher_geometry.py)与[解析参数记录](../results/method_illustrations/local_fisher_geometry.json)随图提供，另有PDF/SVG导出。

上述结论只在理想平滑模型、固定前缀测度及初始点附近成立。实际混合精度与内核数值路径不是该光滑证明的一部分。教师条件分布也不等于真实旧文本标签分布；在所选前缀上 KL 小，不能保证 C4、WikiText、学生自由生成分布或全部知识能力不退化。保留仍需用冻结独立评估和训练重复验证。

## 冻结库的实测组成与表示空间距离

对三个正式生成 seed 的候选池，各池1957个512-token块，从中各选196块。每块计511个预测 token，因此单池为1,000,027个预测 token，库容量为100,156个唯一预测 token。以下均值和样本 SD 来自三个 pool/选库 seed，描述数据组成，不能当作训练性能误差条；当前主 seed 同时改变池和选择，尚未单独识别条件选库波动。

| 选库规则 | NLL经验CDF最大差 | 平均最近代表特征平方距离 | bank教师NLL均值，nats/token |
| --- | ---: | ---: | ---: |
| 随机 B4 | 0.0635 ± 0.0277 | 0.2413 ± 0.0035 | 3.2432 ± 0.0632 |
| NLL比例分层 M | 0.0322 ± 0.0051 | 0.2387 ± 0.0067 | 3.2585 ± 0.0086 |
| 聚类代表 C | 0.1068 ± 0.0218 | 0.2096 ± 0.0030 | 3.0850 ± 0.0540 |

NLL比例分层的 CDF 距离与 bank 均值的跨 seed 波动较小，符合固定五等频层配额的设计。聚类组在自己的选择空间内降低了平均最近代表距离，但选出的文本平均 NLL 较低，NLL 经验分布相对原池偏移更大。这是两种不同代表性定义的实测差别；尚不构成任何目标适应或保留排序。三个池无整块重复，D 与 B4 indices 完全相同，不能把 D 当作发生了去重干预的库。

特征是冻结教师最后层有效预测上下文的均值，再做 L2 归一；距离是候选点到最近 bank 向量的欧氏平方距离。它不等于语义覆盖率，更不等于独立评估分布覆盖率。CDF最大差和Wasserstein-1只作描述，不据此进行独立样本检验。详见[原始诊断说明](回放库分布诊断.md)、[CSV](../results/bank_diagnostics/bank_geometry.csv)及[含源SHA的JSON](../results/bank_diagnostics/bank_geometry.json)。

## 共享准备成本与独立部署成本

成本诊断固定初始 Pythia410M、seed11 全池、batch16、H100 80GB、FP32权重/BF16 autocast 及当前 strict/auto 后端政策。每种路径先预热一个 batch，再测三次全池；下表为三次 wall 中位数及观测峰值。它们不是三个训练 seed，也不包含生成、模型加载、预热和后续 CPU 选库。

| 教师统计路径 | 全池 wall 中位数（秒） | 包含模型的allocated峰值（GiB） | 必要运算 |
| --- | ---: | ---: | --- |
| M独立所需 NLL-only | 8.6601 | 3.3131 | transformer、LM head及全词表NLL |
| C独立所需 feature-only | 3.3223 | 1.9027 | transformer及上下文池化，绕过LM head |
| 研究共享 joint | 8.6781 | 4.8443 | 同一前向联合保存NLL与特征 |

![教师统计提取成本诊断](../results/method_illustrations/selection_statistics_cost.png)

图左柱为三次全池测量的wall中位数，小黑点为测量重复；图右分别给出包含模型的allocated/reserved峰值。三次计时不是三个训练seed，三条统计路径也不能相加成某个方法的费用。[图形来源与SHA](../results/method_illustrations/selection_statistics_cost.json)单独保存，另有PDF/SVG导出。

feature-only 在这次特定测量中的统计提取 wall 约为 NLL-only 的38.4%。三路径重算的相关 tensor/counts 与原冻结统计逐元素完全相同，最大绝对误差均为0。因此“需要特征和聚类，所以 C 的准备必然比 M 昂贵”不成立。独立部署还必须计入各自 CPU selection：历史 seed11 的 M 选择耗时0.0666秒，C 为2.5923秒，其中聚类2.5153秒已嵌套；这两项来自原资产准备，不是与新统计计时同时测得的端到端 benchmark，不能拼接后冒称已测总耗时。

本研究实际资产复用了每池一次 joint 提取；seed11 当时记录8.1027秒，应按该历史记录记共享费一次。现在的三路径测量是另行开展的工程诊断，其 wall 不能回填成原准备成本。独立运行 M 应计 NLL-only加selection，C应计feature-only加clustering/selection，随机和精确去重随机不需要教师统计；若多个方法共同运行，则按实际共享 joint 计一次，不能再对每组加一次 joint，也不能把嵌套 clustering 重加。独立部署比较仍需另加相应生成及加载等实际支出，当前诊断不宣称已测完整系统成本。

模式顺序固定为 NLL、feature、joint；只有单模型、单池、单设备和三个重复，不能外推到所有模型大小、batch、硬件或冷启动场景。CUDA event记录的是整次stream跨度并含host gaps，不能称kernel active时间。来源为[完整测量](../results/selection_cost_diagnostic_main410m_seed11.json)、[摘要](../results/selection_cost_diagnostic_summary.json)和[计时协议](选择统计独立成本诊断.md)。成本证据也不包含或证明任何 retention 收益。
