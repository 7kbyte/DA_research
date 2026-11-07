# 归档重训QA：实际执行与逐位验收

执行时间：2026-11-07T01:05:40Z—01:11:31Z（训练），01:13:51Z—01:14:51Z（CPU核验）。硬件：4×NVIDIA H100 80GB。

## 目的与范围

这是研究路线中最后一个未执行的工程项：用**已发布的持久项目归档**里解包出的原始重训helper，在全新namespace中重跑4个固定主实验条件，并在CPU上独立核对重训得到的完整模型状态是否与事前预期逐位相同。它不是新科学条件、不使用任何测试分数、不改变冻结网格。

## 输入与授权

- 被测归档：`2026-11-06 07:32`持久项目快照（SHA `97fa503d4f286e3d507745190fecd3570fd1279e41f633da4d7716acbfb11335`，380,166,539字节）。它不是31页最终归档；两者共享同一冻结科学核心与输入，最终归档多出的项目文件不改变重训输入。
- 重训helper：解包后的原始 `scripts/reproduce_experiments.py`（SHA `ac54da86c4ae278783daf045c40f9c1cd7ed267f9be3a8fb09397f163b28eff1`），调度器 `scripts/schedule_jobs.py`（SHA `c5c264b058ff655315ef896ccb0a89b471637c6612013a51999fc6f48b2d16d9`），均未修改。
- 队列与注册表：`4ec9a68ef6f78bbb…`（4个任务）与 `1c587c00182e3b00…`，SHA与准备计划一致；任务定义、expected_seconds=600、配置均未改。
- 运行前复核：用归档内helper的dry-run重验输入，rc0、1.00秒、状态`verified_not_started`，未启动训练（[artifacts/provenance/archive_qa_actual_v1/prelaunch_reverification.json](artifacts/provenance/archive_qa_actual_v1/prelaunch_reverification.json)，SHA `6071b78e315bebc527cf132840d6c85220ccd9db9dc41010b5e925330dc63ee5`）。
- ROOT授权：[artifacts/provenance/archive_qa_actual_v1/ROOT_operational_amendment_v2.json](artifacts/provenance/archive_qa_actual_v1/ROOT_operational_amendment_v2.json)（SHA `9192938e3b089d326175ed722543d02a5e96b82e7602fb82208834eb9eda7c04`）。相对原计划**只增加** `--launch-margin-seconds 120 --stop-margin-seconds 120`，deadline `2026-11-07T02:56:32Z`，4张GPU，`--exit-when-idle`；四个任务定义与expected_seconds未变。
- 前置条件：全部514个冻结评价已完成（completed 514/514）、无训练或调度进程、4张GPU 0% / 0 MiB。

## 训练执行

调度器PID 3494892，2026-11-07T01:05:40.675Z启动；四个任务分别在GPU 0/1/2/3上运行：

| 条件 | 状态 | rc | 占槽秒 | GPU |
|---|---|---:|---:|---:|
| formal_deterministic_B1_s11_b0p05_w0p05 | completed | 0 | 340.4 | 0 |
| formal_deterministic_B2_s11_b0p05_w0p05 | completed | 0 | 350.4 | 1 |
| formal_deterministic_B3_s11_b0p05_w0p05 | completed | 0 | 350.4 | 2 |
| formal_deterministic_B4_s11_b0p05_w0p05 | completed | 0 | 350.4 | 3 |

全部任务严格整数rc0、无任何stop字段、无缺失输出（[artifacts/provenance/archive_qa_actual_v1/job_state.json](artifacts/provenance/archive_qa_actual_v1/job_state.json)，SHA `2e659e07292a8523efc10b92aaa5aa8fa83bfba5c739a9a2987b73e18b0d0d0c`）。每次训练306步、10,000,000目标预测token、500,000回放预测token，覆盖2,093,056个不同目标位置；生成/真实回放库与冻结配置一致。

## CPU逐位核验

核验工具为先前独立准备并测试的 `scripts/verify_reproduction_qa_postrun.py`（SHA `5417fc0747d94feae828e0e53133edf66745b2fb60885cc591f84f2feceea7e2`），使用归档内的完整状态指纹算法（SHA `b2997465717ddd46b5f80924a022d6a82bafe65f5d348636852fdc10836a2362`），在`CUDA_VISIBLE_DEVICES=''`下运行。结果：

- 状态：**`passed_all_final_and_best_full_model_state_bytes`**；`all_expected_endpoint_weights_bitwise_equal = True`，端点 8 个（4个final + 4个best）。
- 读取：24次检查点操作、序列化读取 77,829,310,128 字节，其中fresh整检查点SHA 25,943,462,104 字节；外层whole wall 60.02秒（user 29.59/sys 30.05），内层op计时嵌套不重复相加。
- 未做的事：不比较训练/验证轨迹、不比较优化器或RNG/流状态、不要求与原检查点序列化字节相等（`serialized_original_checkpoint_equality_required = False`）；新增科学重复 0。
- 收据：[actual_postrun_v2/receipt.json](results/reproduction_qa/archive0732_seed11_2x2_v1/actual_postrun_v2/receipt.json)（SHA `7718682a32a6045e4bdbdbc3e2c352a5362ac5a9e113e5aed471673d2e5039d4`）。

## 可以声称与不能声称

- **可以**：该07:32持久归档在本平台、本容器环境下，可重跑这四个固定条件并得到与事前预期**逐位相同**的final与best完整模型状态（8/8）；这比"归档内容SHA正确"强，是对权重可复现性的实测。
- **不能**：不能外推到全部267次训练或全部488个端点；不能称训练轨迹/优化器状态可复现；不能当作新的科学种子或检验结果；也不能据此声称跨硬件/跨环境可复现（本次是同一平台的同一容器镜像）。
- 复现边界仍受严格确定性设置约束：完整参数更新、固定教师、冻结数据与CUBLAS确定性配置；跨平台或不同CUDA/cuDNN版本需要重新做资格检查。

## 证据位置

- 授权与前置复核：`artifacts/provenance/archive_qa_actual_v1/`
- 调度日志：`artifacts/provenance/archive_qa_actual_v1/scheduler.log`；任务状态：`artifacts/provenance/archive_qa_actual_v1/job_state.json`
- 4个run的config/events与summary副本：`artifacts/provenance/archive_qa_actual_v1/run_records/`、`artifacts/provenance/archive_qa_actual_v1/summaries/`
- 核验收据：`results/reproduction_qa/archive0732_seed11_2x2_v1/actual_postrun_v2/receipt.json`
- 重训checkpoint本体：`/tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/scratch/runs/*/checkpoints/`（临时盘，未纳入项目归档）

## 第二轮（补充）：直接验证**最终交付归档**可重训

第一轮测的是07:32中间归档。为直接验证交付给用户的那份归档，2026-11-07T01:18:47Z—01:25:47Z 又做了一轮：从 `research_snapshot_20261107_final.tar.gz`（SHA `20a30f41…`，30,731载荷文件）解出 `research/`，用**归档自带的** `configs/formal_deterministic_registry.json`、`artifacts/provenance/formal_deterministic_input_snapshot.json` 与脚本执行 `relocate`（rc0，0.94秒），逐项校验冻结核心源码、四个run的config、目标/回放/验证/测试数据以及固定基座文件的SHA。

| 条件 | 状态 | rc | 占槽秒 | GPU |
|---|---|---:|---:|---:|
| formal_deterministic_B1_s11_b0p05_w0p05 | completed | 0 | 340.4 | 0 |
| formal_deterministic_B2_s11_b0p05_w0p05 | completed | 0 | 350.4 | 1 |
| formal_deterministic_B3_s11_b0p05_w0p05 | completed | 0 | 350.4 | 2 |
| formal_deterministic_B4_s11_b0p05_w0p05 | completed | 0 | 350.4 | 3 |

四个任务严格rc0、无停止、无缺失输出，占槽与第一轮相同（340.4/350.4/350.4/350.4秒），这本身也是确定性的重复观察。

随后用归档内的 `model_fingerprint` 函数对新的4个final+4个best检查点计算完整模型状态指纹，与事前预期值比较：
**8/8 端点逐位相同**（每个 405,334,016 元素），状态 `passed_final_archive_retraining_bitwise`，24.82秒。
说明：第二轮是**打包层面的补充直接比较**（同一归档算法、直接调用），而完整的受测 `verify_reproduction_qa_postrun` 流程已在第一轮（0732归档）执行并通过；两轮都只覆盖这4个固定条件。

证据：`artifacts/provenance/final_archive_qa_v1/`（relocate、dry-run、launch、job_state、fingerprint_comparison，SHA `1798fad5a4f65217f3aebaae5450a7c049ba478f45d5b2cacf95207d91c24d4a`）。

## 第三轮（扩展）：最终交付归档上覆盖全部八个主组

为把可复现性从4个条件扩到完整主组，从同一最终交付归档用同名 `relocate` 生成新namespace `final_qa_all8_seed11_v1`（8个run，rc0），在4张H100上**两波**重跑：

| 条件 | 状态 | rc | 占槽秒 | GPU |
|---|---|---:|---:|---:|
| formal_deterministic_B0_s11_b0_w0 | completed | 0 | 320.3 | 0 |
| formal_deterministic_B1_s11_b0p05_w0p05 | completed | 0 | 350.4 | 1 |
| formal_deterministic_B2_s11_b0p05_w0p05 | completed | 0 | 350.4 | 2 |
| formal_deterministic_B3_s11_b0p05_w0p05 | completed | 0 | 350.4 | 3 |
| formal_deterministic_B4_s11_b0p05_w0p05 | completed | 0 | 350.4 | 0 |
| formal_deterministic_D_s11_b0p05_w0p05 | completed | 0 | 350.4 | 1 |
| formal_deterministic_M_s11_b0p05_w0p05 | completed | 0 | 350.4 | 2 |
| formal_deterministic_C_s11_b0p05_w0p05 | completed | 0 | 350.4 | 3 |

全部8个任务严格rc0、无停止、无缺失输出；B0因无回放占槽320.3秒，其余350.4秒。

随后用归档内同一 `model_fingerprint` 函数，把**重训检查点**与**原始训练检查点**（`/tmp/grad-research-20261106/runs/<id>/checkpoints/`）各算一次指纹并直接比较：**16/16 端点逐位相同**（8组×final/best，每个405,334,016元素），状态 `passed_all8_seed11_bitwise`，112.1秒。
这一轮不使用预计算预期文件，两侧都在本次运行中实测；D与B4的指纹相同（`63069435…`/`cb76a09b…`），与已证明的D/B4完整状态别名一致，构成内部一致性检查。

边界不变：不比较训练/验证轨迹与优化器/RNG状态，不要求序列化字节相等，不是新科学重复。证据：`artifacts/provenance/final_archive_qa_v1/relocate_all8.json`、`dry_run_all8.json`、`launch_receipt_all8.json`、`fingerprint_comparison_all8.json`（SHA `335d3401d93670515e240946f8786900b3ddaff40393bd8e9744980730850c62`）。

## 第四轮（补齐）：seed 22/33 的其余十六个主条件

用同一最终交付归档relocate出 `final_qa_all24_seed2233_v1`（16个run，rc0），分4波在4张H100上重跑：

| 条件数 | 状态 | rc | 占槽（秒） |
|---:|---|---:|---|
| 16 | 全部completed | 0 | 320.3—350.4（合计5536.0） |

全部16个任务严格rc0、无停止、无缺失输出。随后用归档指纹函数把重训与原始训练检查点各算一次并直接比较：**32/32端点逐位相同**（16个run × final/best，每个405,334,016元素），状态 `passed_all16_seed2233_bitwise`，224.8秒。D与B4在seed22/33上的指纹亦相同。

**累计覆盖**：针对最终交付归档的三轮重训覆盖全部 **24个主条件（8组×3seed、b=.05家族）的final与best，即48/48端点逐位一致**（另加第一轮07:32归档的8/8端点）。这是本平台、本容器环境下由交付归档直接重训得到的权重可复现性；仍不外推跨平台，不比较轨迹/优化器/RNG，不构成新科学重复。

证据：`artifacts/provenance/final_archive_qa_v1/relocate_all24.json`、`dry_run_all24.json`、`launch_receipt_all24.json`、`fingerprint_comparison_all24.json`（SHA `aef829ad51cd49249717ac4b1f90d99114e21b5bab520b47405a150ecf0732a5`）。
