# 补充评价guard v3独立工程验收

结论：最终plan-bound版本通过独立工程审查；这不是C410分数解封授权，也不是一次真实phase完整性check。

- 原失败可在微型真实proof结构复现：producer_source指向非JSON Python源码，v2误作JSON解析。v3只对producer做opaque完整SHA校验，algorithm源码仍完整SHA绑定；audit/registry/snapshot仍要求有效JSON及完整状态结构。Python proof producer不执行。
- 首轮发现repair只绑定protocol与两代源码，缺少精确旧plan/manifest。作者已修复；最终validate_repair在load_plan及authorize同时核对旧plan、manifest、protocol和两代源码。初稿、首轮48tests/stdout与差异保留，不把初稿标成最终验收。
- 最终新旧回归加独立边界测试共104项通过（pytest 2.35s；整个测试child wall 2.708947s，user 2.192399s，sys .518208s）。覆盖producer篡改、JSON来源错误、严格integer rc0（拒绝bool/float等）、stop-key、实际scheduler wrapper argv、缺失输出、完整alias/initial闭包、旧authority不能跨实现、同字节plan换路径不能继承repair、固定journal不可覆盖/改址。
- CP校验继续使用原Verifier与明确48-path allowlist、SHA绑定snapshot和当前5stat；checkpoint换inode触发fresh全读，allowlist外的已缓存.pt仍fresh，stop-requested receipt拒绝。命中与本次fresh bytes分列；分数、attestation、代码/JSON来源不获得CP缓存资格。真实CP未被本审阅读取。
- 独立AST核对：analyze除新增repair参数、两处显式传参和COMPLETE元数据外完全相同；write_auxiliary及嵌套verify_result也完全相同。bootstrap=5000、seed常量20261106、final/best loop、统计/选λ调用与顺序未变。旧protocol绑定18份分析/scorer源码当前SHA全部相符；8个冻结helper及原统计主体未改。
- 真实小metadata另核C410 48逻辑端点、43物理job、6个proved aliases；43个scheduler记录均completed、严格整数rc0、无stop/missing输出标记、definition及记录argv一致。这只核结构，未打开或哈希真实score/attestation，不能替代ROOT将运行的完整check。

最终guard SHA：`6190ded0f8c21c035a1829f0d4f451d3842af9f5f03942dcc1fceb94bdcbc4cb`；CLI：`bd0463e34fe9373ffcb37587f6838f5f5bb1a852b6eafc023bd67923ec97ed0c`；repair：`335a332d7337b3b11cba306722173e56ed7886f40cb9fd1dab3b457f6cae7e2d`。旧plan与manifest分别保持`61f44eba...`、`48f8b223...`；完整来源见独立receipt。

操作仍按[作者v3说明](补充评价guard_v3操作修复说明.md)：ROOT先以exact旧plan+显式repair执行完整check，成功后新TRUE authority须额外绑定该repair，原analyze写同一manifest-scoped exclusive first-read journal后才解析。旧TRUE authority不自动继承；失败不得重置journal。审阅未签任何真实authority、未运行真实check/analyze、未写真实journal、未启动GPU、未读取C测试或behavior分数。

[独立验收receipt](../results/supplemental_guard_v3_independent_review_v1/FINAL_REVIEW.json)保留源码快照、两轮测试stdout/父计时、独立AST核对、真实scheduler结构核对与旧字节来源。计时只描述各测试child，完整工程审阅父级时间未仪表记录，不补造为0；不与推理计时重复相加。
