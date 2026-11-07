此文件记录 root 在首次真实 NLL test 之前明确的端点范围，另存 `configs/nll_scope_execution_freeze_v1.json`，不覆盖旧主实验清单或行为范围。它只固定范围，不授权推理；C 的正式完成资格仍须后续 registry/protocol/summary 逐项证明。

|阶段|模型|heldout|逻辑端点|端点|当前新增 adapted forward|
|---|---|---|---:|---|---:|
|主实验|410M|main_v3|72|36 final + 36 best|66|
|A|410M|main_v3|126|全部 final|108|
|B|410M|main_v3|60|全部 final，固定10M|60|
|C|410M|extended_v1|48|24 final + 24 best|48|
|C|1B|extended_v1|48|24 final + 24 best|48|
|C|2.8B|extended_v1|30|15 final + 15 best|30|

合计384个逻辑端点。当前只扣除主实验6个绑定完整权重等价证明的 forward，以及 A 的18个已注册原训练 aliases；其余360次适应后推理尚未扣除未来 C 的可能等价。再加4个 model/heldout/实际执行策略各自匹配的 initial，合计364次 NLL 推理。相同权重只省 forward；物理训练身份和完整成本保留。final/best 相同只在完整 SHA 或 state 等价证明充分时再复用，不能按 step 猜测。

main_v3 每次三个 corpus 共1316847个预测 token，extended_v1 共1270857。student 计划总计473399598、额外 teacher468270180，合计941669778个模型单前向预测 token 等价量。initial 的 teacher logits 复用 student；KL 算术仍保留。batch 为410M/1B16、2.8B8，context512；main_v3 与 extended_v1 不混 baseline。

只读脚本 `scripts/freeze_nll_scope.py` 同时保存 `results/nll_scope_budget_v1.json`。它仅从已经完成且数值策略匹配的 validation 轨迹提取 wall/token、step 和来源 SHA，不读取 NLL test 或行为评分。按验证样本 median wall/token 外推，每端点 corpus 循环代理为 main410M19.51秒、extended410M18.96秒、1B24.04秒、2.8B60.52秒。364项合计8567.97秒（2.38小时）串行循环；理想4槽等分35.70分钟。乘0.75/1/1.5的敏感性情景对应理想4槽26.77/35.70/53.55分钟。

这些是规划代理值，不是实测 test 耗时、置信区间或运行保证。它们排除 scorer 启动、权重/checkpoint 加载、每项 CPU SHA 校验、别名复制与排程；并行训练会影响可用 GPU 和吞吐。initial 使用适应后 validation 的计时代理，未因 teacher logits 复用而单独缩减。teacher/KL 子计时已经包含于父 wall，不再相加。行为预算另见 `results/behavioral_timing_budget_v1.json`；1B/2.8B 行为推理需用 root 授权的 initial-only attestation 替换模型规模外推，replay-fit 的15项也另列。

C 的 best 只按 target-validation 预定规则选择，保留实际 step、target/replay exposure；完整训练成本不会按 best 曝光缩小。root 允许2.8B在首个 formal 前基于纯算时将完整15组50M改为25M；具体修订由 training_engine 单独冻结，本范围不改变组数、seed数、端点数或行为端点。

复刻范围与计时快照应使用新路径，避免覆盖本证据：

```bash
python scripts/freeze_nll_scope.py \
  --scope configs/nll_scope_execution_freeze_NEW.json \
  --budget results/nll_scope_budget_NEW.json
```
