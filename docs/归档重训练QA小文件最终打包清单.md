# 归档重训练QA：实际完成后的小文件打包清单

本次仅审查路径与源码，未运行QA、后验或测试，未打开模型、checkpoint或归档。以下均为**未来实际执行后**的保留要求，不表示QA已发生。

当前存在遗漏风险：`package_research.py:22–62`只收当前工作树及显式`--runs-root`；通常指定的`/tmp/grad-research-20261106/runs`不包含这两套QA。归档树内的`--summary-copy`也仍写在外部目录。后验器只写指定的`--output`收据，不自动汇集训练日志。已有worktree `prepared_runtime_inputs/`只是准备副本，不能代替未来实际记录。

为缩短路径，以下记：

- `A=/tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/research`
- `S4=/tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/scratch`
- `S3=/tmp/grad-research-20261106/reproduction_qa/optional_Cthree_B4_seed11_v1/scratch`
- `N4=archive0732_qa_2x2_seed11_v1`，四个ID为`N4__formal_deterministic_B{1,2,3,4}_s11_b0p05_w0p05`，其中`N4`须展开为实际namespace。
- `N3=archive0732_qa_Cthree_B4_seed11_v1`，三个实际ID见[三模型expected模板](optional_three_model_B4_archive_QA_expected_identity_TEMPLATE.json)的`runs[].run_id`，不能把模板中的pending状态写成已验收。

## 实际输出与应复制的小文件

|类别|main4实际源路径|optional Cthree实际源路径|
|---|---|---|
|训练记录|`S4/runs/<ID>/config.json`、`events.jsonl`、`summary.json`|`S3/runs/<ID>/`下相同三文件|
|summary副本|`A/results/reproductions/N4/<ID>_summary.json`|`A/results/reproductions/N3/<ID>_summary.json`|
|调度state|`A/logs/reproductions/N4_job_state.json`|`A/logs/reproductions/N3_job_state.json`|
|逐job stdout/stderr合并log|`A/logs/reproductions/jobs/<ID>.log`|同目录下本阶段三个ID的log|
|调度事件|两阶段共享`A/logs/reproductions/scheduler_events.jsonl`；保留原字节的最终快照，不按namespace擅改行|同左，两个阶段结束后可只保留一份并列明共同来源|
|输入配置和运行定义|`A/configs/reproductions/N4/`内四份配置，及同级`N4_{registry,queue,provenance}.json`|`N3/`内三份配置及`N3_{registry,queue,provenance}.json`；另保留实际`N3_runtime_queue.json`、`N3_original_source_registry.json`及两版队列的精确差异证明|

若出现`events.pre_resume_*.jsonl`、失败尝试、重试或异常日志，也需按实际保留，不能只留最后成功副本。训练验证轨迹写入`events.jsonl`和`summary.json`，当前实现没有另一个自动生成的独立validation文件目录。上表由`train.py:345–348, 554, 680–707`、`reproduce_experiments.py:122–164`、`schedule_jobs.py:92–113, 170–175`及两份expected文件核对。

## worktree落点与后验小证据

main4继续使用`results/reproduction_qa/archive0732_seed11_2x2_v1/`；建议在其下新建`actual_run_metadata/`，保持`run_records/<ID>/`、`configs/`、`scheduler/`原相对层次。Cthree未来使用独立`results/reproduction_qa/optional_Cthree_B4_seed11_v1/actual_run_metadata/`。**现在不创建这些实际结果目录。**

除上表外，最终需逐字节保留：实际ROOT授权与argv/env/进程退出核对、实际plan/expected及其SHA、prelaunch输入/科学同一性审计、relocate/dry-run stdout/stderr、scheduler外层stdout/stderr与完整父计时、后验stdout/stderr及完整父计时、ROOT最终验收。外层stdout/stderr和Cthree后验`--output`目前没有固定实际路径，ROOT未来应显式绑定到上述worktree结果目录，不能事后只留下外部路径引用。

main4已有后验runbook固定输出`results/reproduction_qa/archive0732_seed11_2x2_v1/postrun_final_best_ROOT_attempt1.json`，其`endpoint_proofs`包含八端点完整FP、实际曝光、新checkpoint SHA和读取记录；它本身在worktree内，可直接入包。Cthree应以真实补齐且冻结的expected调用同一验收器，将六端点收据也直接写到自己的worktree目录。两者均另保留所引用的**小型**plan/配置/identity proof/metadata来源；必要时复制归档验收器和算法小源码，保留原路径与SHA映射，不改写原始收据中的路径。后验器不比较完整训练轨迹；保留events不等于已经做过逐步一致性比较。

## 最终收口

1. 等该套QA的scheduler、children和后验全部退出，再复制真实生成的小文件；失败和未执行状态如实保留。
2. 对明确列出的源与副本核大小、SHA及复制前后稳定stat，保存简短`source_path → packaged_path`清单；不以软链接代替副本。共享scheduler事件应在两个QA阶段均结束后取得稳定最终副本。
3. 核最终项目包的`PACKAGE_MANIFEST.json`确实包含这些worktree小文件；仅有外部路径引用或准备模板不够。

禁止递归复制整个`S4/S3`或整个外部归档树：`checkpoints/latest.pt`、`checkpoint_best.pt`、base、TAR/ZIP及其他巨大权重继续外置，只在小证据中保存路径、stat、SHA和模型FP。该清单不增加执行权限，也不改变既有QA范围与30分钟交付尾部。
