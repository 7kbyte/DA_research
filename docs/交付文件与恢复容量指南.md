# 交付文件与恢复容量指南

本版（v2）按已验收的共享盘文件核对，目录为 `/data/202610/7kb/research_delivery/`。GiB = 2³⁰ bytes。表中1415是 **2026-11-06 14:15 中间项目快照**，含工作稿及当时进度，不代表研究最终完成；后续状态以 [最新持久快照指针](durable_snapshot_latest.json) 和 ROOT 更新为准。

| 已发布文件 | bytes / GiB | 用途与逻辑端点 | 内含唯一权重 ZIP / 基座依赖 |
|---|---:|---|---|
| [primary_final_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/primary_final_weights_lossless_v1.tar) | 24,949,391,360 / 23.236 | 主实验 b=.05 加B0，共24项 final | 21；3对D/B4别名，含基座 |
| [budget_final_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/budget_final_weights_lossless_v1.tar) | 14,651,054,080 / 13.645 | B4/M 的 b=.01/.1 共12项 final | 12；含基座 |
| [primary_best_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/primary_best_weights_lossless_v1.tar) | 37,979,074,560 / 35.371 | 主实验全部36项验证选定 best | 33；3对D/B4别名，含基座 |
| [supplement_A_selected_final_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/supplement_A_selected_final_weights_lossless_v1.tar) | 16,949,309,440 / 15.785 | 补充A每方法验证选λ，共18项 final | 15新增；3个B1引用primary，基座也取primary |
| [extended_C_410m_final_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/extended_C_410m_final_weights_lossless_v1.tar) | 25,948,508,160 / 24.166 | 补充C410M的24项100M目标曝光 final | 21；3对D/B4别名；基座取primary_final，本包不独立 |
| [research_snapshot_20261106_1415.tar.gz](/data/202610/7kb/research_delivery/research_snapshot_20261106_1415.tar.gz) | 537,114,665 / 0.500 | 代码、token数据/回放库、配置、报告、指标、运行记录与环境证据 | 0；不含大训练checkpoint或表中权重TAR载荷 |

五个权重包合计 **120,477,337,600 bytes（112.203 GiB）**；加本表唯一一份项目快照，六个不同文件共 **121,014,452,265 bytes（112.703 GiB）**。此口径不含历史快照、预留文件或 scratch 副本，也不按跨包引用再次增加体积。primary/预算/best各自内含相同410M基座，实际TAR体积已经包含这些重复字节。

按五份小manifest逐项核对，共 **114条包内逻辑记录、111个按实际源训练run＋final/best去重的端点、102个不同完整model-state指纹**，对应102个权重ZIP载荷。A的3条主包引用不增加源端点；D与B4的完整状态别名不增加不同模型数，但保留各自真实训练run身份与成本。指纹并集来自已验收的完整权重摘要，本次没有重读权重。不能把记录数、训练run数和不同权重数混称。

**C410M final已持久验收。** 其 [发布收据](C410_final_weights_durable_publication.json) 与 [ROOT验收](../artifacts/assets/C410_final_weight_durable_acceptance.json) 绑定实际TAR；C410包依赖primary_final的native基座、配置和tokenizer，单独一个C410 TAR不能完整离线恢复。其余C模型final、C best以及仍在导出的额外A93/B60包未计入本表已交付项。A selected包仅保存验证选定18项，并非126个λ候选全权重。best的实际曝光以其metadata为准，不把它当成与final相同训练预算。全部权重均是无损 **model-only**；不含optimizer，不能据此续接原Adam训练状态。

容量按“保留的TAR文件”与“新恢复目录”分别规划：单个410M的FP32 tensor共1,621,336,064 bytes（约1.510 GiB），实际序列化文件另有开销；再加共用基座、所选ZIP和少量元数据，建议为单模型另留 **至少4 GiB**。两个固定恢复例的实测新目录文件合计6,421,692,458 bytes（5.981 GiB），建议另留 **7 GiB**。这些是恢复工作空间规划，均不含原TAR占用，也不是RAM或重训练资源保证。若102份不同权重全部保留为解码FP32，单纯tensor共165,376,278,528 bytes（**154.019 GiB**），另计元数据、ZIP、基座与项目文件；按需恢复一项可避免全部展开。

建议使用顺序：

1. **先看报告。** 从项目快照解出 `research/`，打开 `research/reports/研究报告_工作稿.pdf`；HTML需保留相邻相对资源。项目包的 `research/PACKAGE_MANIFEST.json` 列出文件bytes/SHA和未包含的大checkpoint，适合核对完整性；权重TAR无需为了阅读报告而解开。
2. **再恢复所需模型。** 按 [两个按需恢复示例](已交付权重恢复示例.md) 只提取固定base、工具、自己的metadata和一份ZIP。A B4 λ5的新ZIP与C410 final均使用primary_final基座；A B1三项还引用primary中的对应B1 ZIP。已有 [真实CPU恢复演练](已交付权重恢复示例_实际恢复收据.json) 仅对主包与A包完成两次decode、原样metadata示例及新恢复权重完整fingerprint重算；未与原latest再次uint8比较，也不是GPU重训练QA。C410的源端全tensor往返及持久发布已验收，但从这份持久TAR重新独立提取恢复尚未演练。
3. **迁移路径。** 单纯decode使用显式 `--base` 与新输出路径，原metadata中的历史路径保留作来源。重训练时，将项目token数据保留相对目录，固定基座放入新scratch的 `models/pythia-410m-final/`，然后用 `scripts/reproduce_experiments.py relocate` 生成新namespace配置和来源记录；详细参数见 [README重定位说明](../README.md) 与 [环境重建说明](环境重建与移交.md)。
4. **最后重训练。** 先匹配冻结环境与执行策略，验证core/model/data SHA，完成relocate和 `reproduce` 不带 `--execute` 的复核，再用明确的新运行身份启动。恢复出的适配模型用于加载/评估；复现原训练从固定initial base、冻结配置和token输入开始。新平台的数值资格与训练资源另行核验，CPU恢复成功不等于跨平台训练逐位复现。

本版仅核现有持久发布收据、当前owner/inode/stat与小manifest，没有打开TAR/ZIP/checkpoint载荷或读取C/behavior评分。各文件已验收的完整SHA、端点/基座计数和本次读取范围见 [v2来源核对收据](交付文件与恢复容量指南_来源核对_v2.json)。[v1来源收据](交付文件与恢复容量指南_来源核对.json) 和 [旧版指南原字节](../artifacts/provenance/delivery_capacity_guide/pre_C410_v1/guide.md) 保留；本导航更新晚于1415快照，会随后续项目快照纳入。
