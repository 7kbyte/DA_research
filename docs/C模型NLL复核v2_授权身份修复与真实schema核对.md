# C 模型 NLL reviewer v2：授权身份修复

新版 reviewer 尚未用于真实 1B/2.8B 结果。原 `96665a…` 源码、原测试及准备收据保留在 `artifacts/provenance/C_model_nll_review_v2_schema_repair_v1/retained/`。本次生产源码只有一处常量替换：`authorized_by == 'ROOT'` 改为真实授权协议的 `authorized_by == 'root'`；生成入口同步。其他门禁、统计、循环、5000 次 bootstrap、seed、输出格式均未变化。

toy 原先也使用了错误的大写值，现改为真实小写；补充拒绝 `ROOT` / `Root` / foreign / null 的回归，以及 check/analyze 与绑定快照路径不同的拒绝测试。全部 41 项定向测试通过，完整测试子进程 wall 8.049041 秒（不是全仓库测试）。一次测试夹具 SHA 更新遗漏导致的失败和一次 metadata 审阅脚本语法失败均保留原日志与计时，没有覆盖。

实际兼容审阅只读取已经开放的 C410 小 metadata：COMPLETE、authority、plan、manifest、protocol、repair、first-read journal、READY、readiness、历史状态原字节副本、原 enqueue 不可变队列、check/analyze timer、执行 attestation、完整模型等价证明及其小来源、config/summary、analysis_manifest。36 项核对通过，实际 metadata 审阅完整子进程 wall 0.302662 秒。119 个小来源文件已留存副本和 SHA。没有读取或重新散列 score JSON、数值 CSV/bootstrap 行、checkpoint、基础权重、TAR、ZIP；未读取新 1B/2.8B 分数，也未运行 reviewer 数值主体或任何 check/analyze。

核对涵盖真实的授权主体与布尔门禁、plan/manifest/repair/journal 链、final/best 范围、matched baseline 的 id/test_paths/evaluation_input_sha256、所有 43 个实际 closure 的 definition SHA / wrapper argv / integer rc0 / no-stop、49 份物理和逻辑 attestation 字段、六个完整状态证明、producer Python 的 opaque SHA、真实 timer 的 argv/plan/repair/authority/output 绑定。除已修正的授权主体大小写，未发现其余字段形状与真实 C410 来源不兼容。

历史状态需要单独说明：C410 原 check/analyze 当时均指向同一个 live state 路径。保留的 `state_before_authority.json` 字节 SHA 恰等于 readiness 记录的 `f9b8e410…`，本次用这份历史原字节逐 job 核对；没有读取今天继续更新的 live state，也没有把旧快照的新路径冒充当时原路径。因此本次是**字段/schema 审阅**，不是让仅支持 1B/2.8B 的新版 reviewer 对 C410 获得一次实际通过记录。

未来 ROOT 应先保存实际 successor executor state 原字节不可变快照，再让该模型原 check/analyze 都显式使用这条快照路径。新 reviewer 同时要求 `--state` 路径及 SHA 等于 COMPLETE readiness 的原始绑定，两个 timer 的 `--state` 也必须为同一路径。实际 successor 队列仍须显式提供不可变原字节及 SHA；不会使用或假定旧 master 路径。

未来调用参数仍见 `docs/C1B与C2p8B_NLL完整组独立复核_v2准备.md`。本收据不提供 score authority，不改变旧 C410 代码或冻结科学/guard 源码。准备工程的整体父进程耗时未知；只有明确记录的测试及 metadata 审阅子进程计时可以单独登记成本。
