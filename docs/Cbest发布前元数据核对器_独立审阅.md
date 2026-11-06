# C best 发布前元数据核对器：独立工程审阅

本次只审阅准备中的核对器及小元数据，不是任何 C best 包的实际验收或发布许可。没有调用真实 `review()`、导出器、publisher，也没有打开 checkpoint、ZIP、TAR、native base 或任何评分载荷。完整包和相应 whole batch 收据仍须真实完成后，由 ROOT 决定运行原入口。

初版 `a31f4062…` 有一项授权门禁缺口：它复用的冻结资源验证函数检查 accepted/root 等字段，但不要求 `authorized` 显式为布尔 `true`。独立内存算例复现了 `authorized=false` 可通过该继承函数。当前实际 C410 批许可 `74f6ac60…` 本来就是 `true`，因此这项发现不表示正在运行的批次未经授权。作者保留初版源码、测试、README 和收据原字节后，只在尚未实际使用的新核对器加上 `require_ROOT_batch_permission()`；冻结 exporter/helpers 未改。

修订核对器 SHA 为 `a6250f88dd909567b02baa28f5d458becff169e17d5e85a37ec50f8c1e8811eb`。独立 probe 验证 `true` 通过，而 false、缺失、整数 1、字符串 true、null 均拒绝；AST 核验该门禁确实位于继承资源验证调用之前。移除新增门禁后，`review()` 及其他原函数 AST 与保留初版完全一致。

其余检查未发现新的实质缺口：

- 三模型分别要求完整 **24 / 24 / 15 个 best 逻辑端点**，绑定 v4 scope、冻结 exporter/publisher/native reviewer、原 registry/snapshot/protocol/queue、实际 READY、源 audit 和 best proof。逐行 config/summary 使用小文件 fresh SHA；原 CP 和大载荷只检查记录 SHA 的来源及当前五项 stat。
- 410M/1B 的每个 D/B4 best 决定分别绑定两边 CP SHA、完整模型指纹和真实 best 身份。true 才能引用本 seed 的 B4 ZIP/FP、编码调用为零；false 的 D 必须独立编码，不能以 phase 完成推定等价。2.8B 无 D，要求 15 个独立 ZIP。
- 包内本模型 **native 六文件**、metadata、独立 ZIP、七个固定 manifest/工具成员及全部 gate 证明成员按冻结 assembler 推导，路径安全且唯一，数量须吻合既有完整 TAR 收据；重复 gate 来源保留不同编号。sidecar 只在本包 metadata 目录内按有限大小 opaque hash，不反序列化。
- 实际 source record、严格整数 rc0、两个 stop 键缺席、原 training definition、best 实际曝光、完整训练曝光、冻结数值 policy、完整 encode/decode/原 uint8 与非权重 metadata 往返均有绑定。实际 batch 的 child launch、spec、package、slot、scope、资源许可路径/SHA、argv、CPU 环境也须匹配；A/B 或首例许可不能替代 C 批许可。
- 使用原 **extra best reserve** 的 dev/inode/owner/mode/link/已分配容量身份，不能误用原 C final reserve。锁空闲只是观察，结果明确不形成独占租约；随后仍需 ROOT 单独审阅并由 publisher fresh 核源文件、写入前缀和 fsync。

独立 probe 保留在 [审阅目录](../artifacts/provenance/C_best_publication_independent_review_v1/)。首轮六项 probe whole child 为 **0.107773 s**（user 0.085916 / sys 0.020707 s），修订定向 probe 为 **0.062561 s**（user 0.053824 / sys 0.007983 s）。这是两个独立 CPU child 的实际时间；人工源码审阅父时间未测，不将它们称为完整审阅 elapsed。作者既有 171 项测试及修订后的 5 项新增用例只查收据，未重复全套；新准备收据为 `preparation_receipt_v2.json`，SHA `510fcf9f7e03bc805235b17a91ab3470ac5051e2d575dd39b94257738349d53b`。

工程结论：修订版本可等待真实包、整批计时和各模型独立 ROOT 批许可齐备后使用。当前独立审阅只接受工具的准备状态，不声称 63 个 best 权重已经完成、发布或被本次重新逐字节核验；实际 TAR 成员及整 TAR SHA 仍是原 exporter 的历史证据，未来发布须 fresh 校验。正式稳定来源与本次证据 SHA 以同目录 `COMPLETE.json` 为入口。
