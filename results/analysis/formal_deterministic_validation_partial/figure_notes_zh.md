# 图示说明（中文）

当前状态：**PARTIAL，禁止作为完整正式结果引用**；2/36 个端点。数据来源仅为明确指定注册表，未混入 pilot。当前端点为 validation / fixed_exposure_final。

检查点曝光口径：fixed full target-token exposure。`target_prediction_tokens`/`replay_prediction_tokens` 是被评估检查点的实际曝光，`full_training_*` 是完整训练支出；best 不能被当作全部经历了最终目标预算的模型。

- `gain_forgetting_scatter`：分别展示 C4 与 WikiText-103。横轴目标 NLL 降低量越大越好，纵轴旧域 NLL 增量越小越好；小点为独立训练 seed，大点与横纵误差棒为 seed 均值 ± 样本标准差（ddof=1）。n<2 时不绘制 SD，不能解释为 SD=0。主图使用 b=.05，B0 使用 b=0。
- `budget_curves`：B4 与 M 在 b=.01/.05/.10 下的目标收益和两个旧域遗忘量，点与误差棒为实际已完成 seeds 的均值 ± SD；partial 图不能把完成较早的子集与其他方法作最终比较。详见 CSV 的 n/expected_n。
- `validation_trajectories`：细线为 seed，粗线为当前同曝光验证结果均值；不同时间点可能具有不同 n，原始表保留曝光和完成状态，不对缺失 seed 插值。该图始终使用验证数据，与测试端点区分。
- `training_cost_components`：每个正式运行的 setup、训练、验证、checkpoint 父计时的平均分项。教师和 KL 子计时已包含其中，不再次相加。共享生成/联合特征提取/选库开销另见 `shared_preparation_costs.csv`，每 seed 仅记一次。发生恢复时，逻辑 counters 可能不包含失败并回滚的工作，完整成本需外部 wall time。

`paired_statistics.csv` 的差值为同 seed 的 A−B 后再求均值和 SD，不是两个独立组均值相减再当作成对不确定性。文档 bootstrap CI（如启用测试 JSON 接口）条件于已训练 checkpoints，**不能替代训练 seed 的 SD**；每次共同抽取整篇文档及所有块，同时输出 token 加权和等文档平均。

`identical_bank_controls.csv` 单列 D−B4。若两者实际 tensor hash 相同，则去重未改变任何库内容；两次训练的差异是数值一致性诊断，不能解释成去重干预收益，也不能把它当作新的独立数据实验。

`matched_validation_loss.csv` 只描述共同目标验证 loss 范围内的首次下降穿越点，使用分段线性插值并保存两端实际检查点；非单调轨迹明确标记、禁止外推。这些是验证曲线诊断，**不是实际检查点、等预算比较或测试选模结果**，多个插值点也不增加训练重复数。
