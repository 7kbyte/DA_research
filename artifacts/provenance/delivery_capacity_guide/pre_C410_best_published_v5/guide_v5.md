# 交付文件与恢复容量指南

本版（v5）按已验收的共享盘文件核对，目录为 `/data/202610/7kb/research_delivery/`。GiB = 2³⁰ bytes。表中1640是 **2026-11-06 16:40 中间项目快照**，含工作稿及当时进度，不代表研究最终完成；后续状态以 [最新持久快照指针](durable_snapshot_latest.json) 和 ROOT 更新为准。

| 已发布文件 | bytes / GiB | 用途与逻辑端点 | 内含唯一权重 ZIP / 基座依赖 |
|---|---:|---|---|
| [primary_final_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/primary_final_weights_lossless_v1.tar) | 24,949,391,360 / 23.236 | 主实验 b=.05 加B0，共24项 final | 21；3对D/B4别名，含基座 |
| [budget_final_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/budget_final_weights_lossless_v1.tar) | 14,651,054,080 / 13.645 | B4/M 的 b=.01/.1 共12项 final | 12；含基座 |
| [primary_best_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/primary_best_weights_lossless_v1.tar) | 37,979,074,560 / 35.371 | 主实验全部36项验证选定 best | 33；3对D/B4别名，含基座 |
| [supplement_A_selected_final_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/supplement_A_selected_final_weights_lossless_v1.tar) | 16,949,309,440 / 15.785 | 补充A每方法验证选λ，共18项 final | 15新增；3个B1引用primary，基座也取primary |
| [A_remaining_final_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/A_remaining_final_weights_lossless_v1.tar) | 106,627,983,360 / 99.305 | 补充A其余93项 final，按已保存身份排除而非按成绩选择 | 93；native基座、config、tokenizer、README与恢复工具自带 |
| [extended_C_410m_final_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/extended_C_410m_final_weights_lossless_v1.tar) | 25,948,508,160 / 24.166 | 补充C410M的24项100M目标曝光 final | 21；3对D/B4别名；基座取primary_final，本包不独立 |
| [B_all_final_weights_lossless_v1.tar](/data/202610/7kb/research_delivery/B_all_final_weights_lossless_v1.tar) | 69,578,864,640 / 64.800 | 补充B全部60项 final；固定pool与training seed11，10个selection draws | 60；native基座、config、tokenizer、README与恢复工具自带 |
| [research_snapshot_20261106_1640.tar.gz](/data/202610/7kb/research_delivery/research_snapshot_20261106_1640.tar.gz) | 630,111,999 / 0.587 | 代码、token数据/回放库、配置、报告、指标、运行记录与环境证据 | 0；不含大训练checkpoint或表中权重TAR载荷 |

七个权重包合计 **296,684,185,600 bytes（276.309 GiB）**；加本表唯一一份项目快照，八个不同归档文件共 **297,314,297,599 bytes（276.896 GiB）**。此口径不含历史快照、预留文件或 scratch 副本，也不按跨包引用再次增加体积。primary/预算/best/B/A remaining各自内含相同410M基座，实际TAR体积已经包含这些重复字节。

按七份小manifest逐项重新核对，共 **267条包内逻辑记录、264个按实际源训练run＋final/best去重的端点、228个实际训练run、255个不同完整model-state指纹**，对应255个权重ZIP载荷。A的3条主包引用不增加源端点；D与B4的完整状态别名不增加不同模型数，但保留各自真实训练run身份与成本。相同源端点的指纹一致性也已检查。指纹并集来自已验收的完整权重摘要，本次没有重读权重；记录数、训练run数和不同权重数含义不同。

**C410M final已持久验收。** 其 [发布收据](C410_final_weights_durable_publication.json) 与 [ROOT验收](../artifacts/assets/C410_final_weight_durable_acceptance.json) 绑定实际TAR；C410包依赖primary_final的native基座、配置和tokenizer，单独一个C410 TAR不能完整离线恢复。其余C模型final与全部C best未计入本表已交付项。A selected包自身仅保存验证选定18项；全部126个λ候选final的跨包覆盖见下段。best的实际曝光以其metadata为准，不把它当成与final相同训练预算。全部权重均是无损 **model-only**；不含optimizer，不能据此续接原Adam训练状态。

**A完整λ网格的126个final现已跨包覆盖。** 新发布[其余93项包](A_remaining_final_weights_durable_publication.json)、[ROOT验收](../artifacts/provenance/A_remaining_final_publication_actual_v1/ROOT_acceptance.json)与[durable邻接receipt](/data/202610/7kb/research_delivery/A_remaining_final_weights_lossless_v1.tar.receipt.json)绑定106,627,983,360 bytes的实际归档。本包93项均独立编码、native闭包自带；与主包18个A逻辑引用、A selected的15份新增权重合为126项。此范围按固定网格与已保存身份排除，不按测试表现挑选。新A93包尚未实际单例durable恢复，不能把原builder逐权重往返写成新增用户恢复QA。

**B全部final已持久验收。** [发布收据](B_all_final_weights_durable_publication.json)、[ROOT接受](../artifacts/provenance/B_all_final_publication_actual_v1/ROOT_acceptance.json)与[durable邻接receipt](/data/202610/7kb/research_delivery/B_all_final_weights_lossless_v1.tar.receipt.json)绑定69,578,864,640 bytes的实际归档。B固定generation/training seed11，selection_seed1001–1010、K20/196、B4/M/C共60条件；这些是选库条件变异，不是10个独立训练seed。本包恢复闭包自带，尚未另做从durable包解出一项的用户恢复演练；已有builder逐权重完整往返不能写成新增durable恢复QA。

容量按“保留的TAR文件”与“新恢复目录”分别规划：单个410M的FP32 tensor共1,621,336,064 bytes（约1.510 GiB），实际序列化文件另有开销；再加共用基座、所选ZIP和少量元数据，建议为单模型另留 **至少4 GiB**。两个固定恢复例的实测新目录文件合计6,421,692,458 bytes（5.981 GiB），建议另留 **7 GiB**。这些是恢复工作空间规划，均不含原TAR占用，也不是RAM或重训练资源保证。若255份不同权重全部保留为解码FP32，单纯tensor共413,440,696,320 bytes（**385.047 GiB**），另计元数据、ZIP、基座与项目文件；按需恢复一项可避免全部展开。

建议使用顺序：

1. **先看报告。** 从项目快照解出 `research/`，打开 `research/reports/研究报告_工作稿.pdf`；HTML需保留相邻相对资源。项目包的 `research/PACKAGE_MANIFEST.json` 列出文件bytes/SHA和未包含的大checkpoint，适合核对完整性；权重TAR无需为了阅读报告而解开。
2. **再恢复所需模型。** 按 [两个按需恢复示例](已交付权重恢复示例.md) 只提取固定base、工具、自己的metadata和一份ZIP。A B4 λ5的新ZIP与C410 final均使用primary_final基座；A B1三项还引用primary中的对应B1 ZIP。已有 [真实CPU恢复演练](已交付权重恢复示例_实际恢复收据.json) 对主包与A包完成两次decode、原样metadata示例及新恢复权重完整fingerprint重算。C410另于2026-11-06完成固定 **B4 seed11 final** 的[两durable包恢复单例](C410两包权重实际恢复验收.md)：仅从primary/C410 TAR提取所需成员，CPU decode并重新计算完整fingerprint，405,334,016元素/292 tensors一致，外层wall39.208秒。它验证该固定条件的external-base闭包；未与原latest再次uint8比较，不代表24端点全部重新恢复，也不是GPU训练或Adam续训。
3. **迁移路径。** 单纯decode使用显式 `--base` 与新输出路径，原metadata中的历史路径保留作来源。重训练时，将项目token数据保留相对目录，固定基座放入新scratch的 `models/pythia-410m-final/`，然后用 `scripts/reproduce_experiments.py relocate` 生成新namespace配置和来源记录；详细参数见 [README重定位说明](../README.md) 与 [环境重建说明](环境重建与移交.md)。
4. **最后重训练。** 先匹配冻结环境与执行策略，验证core/model/data SHA，完成relocate和 `reproduce` 不带 `--execute` 的复核，再用明确的新运行身份启动。恢复出的适配模型用于加载/评估；复现原训练从固定initial base、冻结配置和token输入开始。新平台的数值资格与训练资源另行核验，CPU恢复成功不等于跨平台训练逐位复现。

本版仅核现有持久发布收据、ROOT验收、当前owner/inode/stat与七份小manifest，没有打开TAR/ZIP/checkpoint载荷或读取C/behavior评分。已验收完整SHA、逐项去重和读取范围见 [v5来源核对收据](交付文件与恢复容量指南_来源核对_v5.json)。旧[v4来源收据](交付文件与恢复容量指南_来源核对_v4.json)及[更新前v4指南原字节](../artifacts/provenance/delivery_capacity_guide/pre_A_remaining_published_v4/guide_v4.md)保留；更早[v3来源收据](交付文件与恢复容量指南_来源核对_v3.json)及[更新前指南原字节](../artifacts/provenance/delivery_capacity_guide/pre_B_published_v3/guide_v3.md)保留；v2及C410实际QA补充证据也保持原字节。

**快照与后发权重的边界：** 1640项目快照已有B科学分析，但早于17:29完成的B权重与18:33完成的A93权重发布收据；两次后发权重由各自包旁邻接receipt及上述ROOT接受补充；不能要求早期快照内含后来生成的发布证明。1640归档内的 `research/docs/durable_snapshot_latest.json` 仍指前一1415里程碑，属于自身SHA无法预先自引用的边界；本包的实际发布由包外[1640发布收据](snapshot_1640_durable_publication.json)和[ROOT验收](../artifacts/provenance/snapshot_1640_publication/ROOT_acceptance.json)证明，当前工作树[最新持久快照指针](durable_snapshot_latest.json)已到1640。本导航v5晚于1640打包，待后续快照收录。表内总量只相加8个实际不同归档文件，不把小型旁侧JSON、历史快照、reserve或scratch副本计入。
