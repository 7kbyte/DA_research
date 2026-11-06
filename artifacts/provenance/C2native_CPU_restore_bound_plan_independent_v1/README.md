# C2.8 native final 恢复绑定方案独立核对

仅核对准备方案的小元数据与当前持久 TAR stat，结果通过；这不是实际恢复或执行授权。绑定对象固定为 B4 seed11 的 final（765 steps，25,000,000 target / 1,250,000 replay prediction tokens），388 tensors、2,775,208,960 elements，完整 fingerprint 0a819cb2921ba45adfd57b80ddf61f10cbdfeddbc4fb0a3d8846d53a384701b2。

原模型 15 条训练均严格整数 rc=0、无 stop 字段、实际 argv 与冻结定义一致；15 unique、0 aliases，无 D 身份预支。12 个选定 member 含配套 native 6 文件，ZIP/metadata/3 个归档恢复工具的 SHA 均按已发布 manifest 绑定。持久 TAR dev106/inode559646871 当前 stat 与 publication、ROOT acceptance 及 bound plan 一致；本次没有打开 TAR 或任何权重载荷。

准备资源为 scratch 26,574,055,053 B、保守 cgroup memory 48,698,310,656 B（4×raw +4 GiB）、2 CPU threads。1200s 是 execution cap；整组清理可能另用最多约10s，准入按12s teardown allowance 纳入300s余量。必须严格早于 2026-11-07 02:31:32 UTC 启动，02:31:32 整点已不满足严格剩余1500s条件。资源代理不是峰值 RSS 保证；本次没有读取实时资源门禁或作准入判断。

ROOT 必须先完成 1B 恢复，再对 2.8B 作独立 fresh admission 和精确授权。不得把原导出 roundtrip 当作本次新恢复，也不宣称 fresh 原 CP uint8 比对、Adam resume 或 GPU 重训练复现。三个工具 member 的历史 size 为 null，未来实际按提取 member size 和全 SHA 验收；本次未提取。

本次 reviewer whole child 为 0.06288128346204758s，user 0.028375s / sys 0.019918s；手工准备/审阅总耗时未知。原 bind whole 0.05747190862894058s 是独立既有操作，未混入本次 reviewer。没有测试、bind 重跑、解包、恢复、hint、TRUE authority 或评分读取。
