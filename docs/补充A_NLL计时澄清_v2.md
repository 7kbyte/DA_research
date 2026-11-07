# 补充A计时澄清（不覆盖原交付）

本说明补充《补充A_NLL正文素材.md》，不改变原始计时、统计或first-read文件。

126个A评分/复用作业的scheduler whole-slot合计为4531.54s，其中scorer子进程2988.58s已包含在这些占槽时间内；teacher子计时也已包含在scorer内。这些嵌套项不能重复相加。200.35s的复用materialization占槽没有新增模型forward。

39.65s是所有A推理完成后，独立执行完整来源核验、统计和出图的CPU分析wall，**不包含在4531.54s推理占槽内**。其中1.97s guard核验属于39.65s的内含子项。若账本需要合计异类进程wall工作量，可以把4531.54s占槽和39.65s CPU分别记录或明确相加；但其和不是GPU时间，也不是四卡并行研究elapsed。全研究CPU/GPU并行和重叠还需另行核算。

A评分作业日历窗口为2026-11-06 09:27:45.891673至09:47:19.250959 UTC，1173.36s。CPU分析于09:47:41.076404 UTC启动、09:48:20.732083 UTC完成，二者顺序独立，存在约21.83s watcher等待/启动间隔。该间隔没有包含在39.65s child wall中，不应伪装为模型推理费。

来源为原[完成收据](supplement_A_nll_authorized_analysis_completion.json)、[watcher启动记录](../artifacts/provenance/supplement_A_nll_authorized_watch_v1/ANALYSIS_LAUNCH.json)、[watcher完成记录](../artifacts/provenance/supplement_A_nll_authorized_watch_v1/COMPLETED.json)及原physical_inference_costs.csv。此说明不修改已交付v1材料的字节与SHA。
