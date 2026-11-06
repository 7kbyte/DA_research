# C 模型 NLL reviewer v2：独立工程审阅

审阅对象为 `scripts/review_C_model_nll_results_v2.py`，SHA `7a48fb77a07d5d0d757c7f67f473dd49d4e262ffe497dfbe655184211be3d727`。本次只看源码、小准备收据及手工合成图，未运行任何真实模型 reviewer、原 check/analyze，未读取真实 1B/2.8B test、behavior 或 checkpoint/base/archive 载荷。结论是**准备工具可用**，不是新模型结果的通过收据。

未发现新的阻塞问题。ROOT 已指出并由作者修复的 `authorized_by` 大小写差异，独立确认确为初版 `96665a63…` 到当前版唯一生产源码字节替换：`'ROOT' → 'root'`。旧源码、旧准备收据、失败测试和 schema audit 操作日志均保留。原 C410 reviewer/结果及冻结 guard、collector、scorer、统计和训练 core 没有修改。

- **模型与端点。** 1B 固定八组×三 seed，final/best 合计 48 逻辑端点、六个预定配对；2.8B 固定 B0/B1/B2/B4/M×三 seed，共 30 端点、三个存在的配对。两端点一起通过完整阶段门禁，随后分开显示。100M/25M final 和 best 真实曝光分开，未硬编码 C410 的 40M best。测试覆盖 1B 的 0/3/6 aliases 和 2.8B 零 alias；物理数从实际 manifest/证明计算，不预记 D 相等。
- **完整来源闭环。** 要求已有 whole-phase COMPLETE、显式小写 root/true whole-phase 权限、原 plan/manifest/READY、非授权 operational repair、固定 first-read journal、全闭包 readiness 与完整 attestation。要求实际 immutable successor state 路径和 SHA 等于原 readiness，check/analyze timer 的 state、plan、repair 必须一致；analyze 另绑定 authority 和输出目录。ROOT 显式提供并绑定实际不可变 queue SHA，逐 job 校验冻结 definition、wrapper argv、严格 integer rc0、无 stop 字段和无 missing output。不得把今天的 live state 或改名历史副本冒充当时快照。
- **matched initial 与 alias。** 只接受该模型 extended family 的一个 initial，test paths/hashes 与 frozen scientific identity 相符。alias 要有双侧真实 CP SHA、完整模型 FP、实际端点/曝光/小 source 绑定，producer Python 只 opaque SHA；别名 attestation 必须零额外 forward。原训练仍按 actual run 各算一次，D/B4 不因推理复用而抹掉训练成本。
- **独立数值复核。** 从 chunk 重新累加文档，重算六个 token/equal-document 指标、每组和配对三 seed mean/sample SD，检查原 CSV 与完整条件 CI。七个基础统计函数/类 AST 与旧 C410 reviewer 完全相同；bootstrap 保持原 5000 次、seed 20261106，用独立 multiplicity matrix 重建相同 draws，不调用 collector。目标 G 的负号同时交换 CI 上下端点。题材相关文档可能不满足独立/可交换假设；该区间条件于固定 checkpoints，不能替代三 seed SD 或总体机制证据。
- **成本与输出。** final/best 的 full-training parent counters 按 actual run 去重；物理 initial/adapted forward 和零 forward aliases 分列，teacher、worker verification、scorer 均标为嵌套，不与 scheduler whole-slot 相加。生成 13 个完整 CSV、成本 JSON、四张共享行尺度的 7×6.55 in 图（各 PNG/PDF/SVG）和独立复核 receipt。实际运行时仍须 ROOT 外层 Popen/resource 记录完整 child 父计时；内部 inside-run wall 不是整个 Python 进程 wall。

独立增加一次小 probe：对三个不同长度文档、非恒定且非对称 NLL 差，手算 index-draw bootstrap，与四种正负号×token/doc 情形对照；核验负号的 CI 端点交换、三点 sample SD、完整 48/30 grid 和缺点拒绝。只读取已有两模型八张 **synthetic** PNG 制作 contact sheet 并逐面板目检，方法/seed、两个 checkpoint、方向标签和共享行尺度可读；没有再跑作者 41 项套件或真实 36 项 schema audit。

本次 probe 完整 child wall **0.763407 s**，user **0.754911 s**、sys **0.229120 s**。作者的 41 项测试、36 项实际 C410 小 schema 检查及两次合成 CLI 只查原收据，不计成本第二次。人工审阅整体父时间未知。失败工程操作保留，不能只记成功的最后一次；现有 C410 schema 审阅也只是已开放模型的字段对接，不是新版工具对 1B/2.8B 的实际通过。

稳定证据入口：[独立审阅目录](../artifacts/provenance/C_model_nll_reviewer_independent_review_v1/)。将来按 [原准备 runbook](C1B与C2p8B_NLL完整组独立复核_v2准备.md) 填入该模型实际 COMPLETE、authority、successor queue/state、check/analyze timers 及 SHA，使用全新输出目录；不能复用 C410 或其他模型的凭据。
