2026-11-06 从持久化的 `research_snapshot_20261106_0732.tar.gz` 解包到独立 scratch，并真实运行归档内原版 `reproduce_experiments.py relocate` 和 `reproduce` dry-run。四项均成功；无需修改归档、复现 helper 或科学核心。目前没有启动 GPU、scheduler 或重训，完整训练的数值复现尚未验证。

四项选择覆盖完整 source × objective 工程路径，固定 seed11、b=.05、λ=.05、LR=3e-5、10M target prediction tokens、micro16×accum4。它们若执行也只算工程数值复现，不增加科学 seed，不依据 test/behavior 优劣选择。

| 原条件 | 回放来源 | 目标 |
| --- | --- | --- |
| B1 | C4 random seed11 | CE |
| B2 | C4 random seed11 | KL |
| B3 | generated random seed11 | CE |
| B4 | generated random seed11 | KL |

原快照完整 SHA256 为 `97fa503d4f286e3d507745190fecd3570fd1279e41f633da4d7716acbfb11335`。实际解包并逐项核对3925个 payload，共847,507,849字节。解包目录为 `/tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/research`；新 scratch 是其同级 `scratch`，新 namespace 为 `archive0732_qa_2x2_seed11_v1`。原有运行输出与378项主队列未改动。

基座仅从已验收、保留在 scratch 的 `primary_final_weights_lossless_v1.tar` 提取6个 base/config/tokenizer 文件和小型权重清单，没有复制21份训练后权重。外层 TAR 的既有全量 SHA 验收通过且当前 inode、大小、mtime 与原验收记录一致；本次没有重读整个24GB TAR计算 SHA。提取后的6个基座文件逐一重新计算 SHA，与原固定 revision 的 provenance 相同。`model.safetensors` SHA 为 `3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b`。

独立审计 `scripts/audit_reproduction_qa.py` 检查了34项输入SHA（包含10个科学核心文件，以及模型、数据和相邻清单）、四份原始配置与已验收原 checkpoint 元数据、四份重定位配置及精确四项 queue 定义。路径、run_label 与操作 deadline 改动单列；科学参数严格相同。模型 revision、数据字节、strict/BF16/FP32参数不因路径变化而改变。CPU 版本检查的 Torch/CUDA 与原四项训练一致，CUDA context 未初始化。test 数据只做文件 SHA，不读取评分。

机器证据均位于 `results/reproduction_qa/archive0732_seed11_2x2_v1/`：

- `extraction_receipt.json`：原归档 SHA、逐文件解包验收、基座来源与准备耗时。
- `relocate_attempt_original.json`：实际 relocate 命令与成功退出。
- `scientific_identity_audit.json`：完整科学配置与输入映射、34项输入SHA（其中含10个core）。
- `reproduce_dry_run.json`：实际 dry-run 命令、成功退出及未启动声明。
- `qa_plan.json`：精确四项 job、registry/queue/config SHA、原 final 权重指纹、软件政策、启动门禁与计时参考。

原四项在当时环境的 scheduler elapsed 为343.48、343.48、350.53、350.35秒，仅作未来四卡并行工程预算。queue 保守给每项600秒，完整权重 CPU 核对与周转另计。新 QA 操作 deadline 为 `2026-11-07T02:56:32Z`，给硬 deadline 留30分钟；既有 helper 的 scheduler launch/stop margin 还会进一步限制开跑时间。

仅在全部预定训练及评价结束、现有 scheduler 和 children 均退出、四GPU空闲且 root 决定剩余时间足够时，才可考虑执行；本准备没有自动启动权限。届时先重新验证输入和软件政策，再运行下面已演练命令；只有 root 后续明确决定执行时才加入 `--execute`。

```bash
python /tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/research/scripts/reproduce_experiments.py reproduce \
  --registry /tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/research/configs/reproductions/archive0732_qa_2x2_seed11_v1_registry.json \
  --gpus 0,1,2,3 --deadline 2026-11-07T02:56:32Z
```

完整重训后应对比每个 model_state 的全部 key、dtype、shape 与 tensor 字节指纹，并分别检查训练/验证轨迹。整个 checkpoint 文件包含新路径、标签、optimizer 和序列化信息，不要求其文件 SHA 与原文件相同。若没有余量执行，交付应如实保留 dry-run 通过和“未验证完整重训”的状态。
