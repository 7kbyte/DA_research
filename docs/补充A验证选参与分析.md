# 补充A的独立分析与验证选参

入口为 `scripts/analyze_lambda_sweep.py`，模块为 `src/replay_study/lambda_analysis.py`。本入口只读取验证结果，不执行训练、模型推断或测试/行为评分读取。它保留 `supplement_A_lambda_sweep` 身份，不将补充注册表伪装成主实验的formal。

## 注册表、别名和运行覆盖

原预注册表为 `configs/lambda_sweep_deterministic_registry.json`，其完整文件SHA须与对应预注册JSON一致。矩阵须完整包含六方法、七个λ、三个seed，共126条件；18条件显式alias至主实验λ=.05，108条件为新训练。原配置及父配置的SHA逐一核对，父项必须位于 `docs/protocol_status.json` 当前指向的active formal注册表，旧provisional禁止进入。

可用 `--registry` 指定补充调度器生成的运行注册表。只允许输出、device、恢复、deadline和检查点保存等操作字段差异；microbatch、学习率日程、目标预算、训练数据、模型revision、验证路径和科学配置保持原预注册值。声明的操作覆盖须与实际配置一致。新训练配置相对对应主配置仅可改变λ、输出位置及run label，运行摘要仍逐项对照科学配置。

每个条件同时保留condition_id与actual_run_id。alias的actual ID是原formal ID，summary路径也必须是原summary；不得创建副本后把别名当新训练或新seed。`run_identities.csv` 明确给出未来actual_run_id/final.json路径。未来测试应复用主实验同一真实checkpoint的评估与共享initial_pythia-410m.json，不能按alias条件名重复评估。当前入口尚未开放测试分数读取。

## 完整预算端点与数值政策

只接收status=completed、目标预测token恰为10,000,000、回放预测token恰为500,000，且最后验证点的step和两类曝光均与最终counter一致的端点。不读取best目标验证检查点来替换它。基线必须为同一revision的未适应模型，带完整执行provenance及零self-KL；每个摘要还须匹配严格deterministic/warn_only=false、TF32、四项SDPA开关、模型权重与有效精度、设备类型和GPU型号等实际政策。目标、回放及训练代码的provenance也须与对应正式父实验一致。

定义新域增益G=初始目标NLL−最终目标NLL，旧域遗忘F=最终旧域NLL−初始旧域NLL。表按方法×λ报告三seed的均值、样本SD(ddof=1)与n。方法差先匹配seed、b、λ再求差，包括M−B4、C−B4、B2−B1、B4−B3，以及原RQ1来源对照B3−B1、B4−B2。两类旧域分别展示，不只保留有利的一项。误差棒不解释为置信区间或标准误。

## 共享λ的冻结规则

对每个方法，必须齐备全部七λ×三seed最终端点。逐seed计算

\[
S=\Delta L_{target}+\tfrac12(\Delta L_{C4}+\Delta L_{Wiki})
 =-G+\tfrac12(F_{C4}+F_{Wiki}),
\]

再对三个seed无权平均，选择平均S最小的一个λ，并跨三个seed共享。只有严格相等的已计算数值才按更小λ打破平局，不设置事后近似平局容差。`validation_lambda_selection.json` 保存全部候选逐seed分数、均值/SD/n、真实run ID、选择状态以及全部注册表/配置/摘要/基线SHA；不按seed分别挑λ，不按test或best挑λ。

缺失任何方法中的任意λ/seed时，该方法没有选择结果。默认拒绝输出不完整最终报告；显式 `--allow-partial` 可导出已有端点和缺失状态，仍禁止从不完整方法网格挑λ。如果B4/M先完整而其他方法未完成，可以冻结已完整方法的选择，但整个报告继续标partial，缺失方法明确留空。

适应—保留图连线按预定λ排列，并展示完整三seed候选均值±SD。`all_candidates_frontier.csv` 保存全部候选，包括缺失状态；C4、Wiki和联合三目标的非支配标记仅是描述，不另外用于选λ。partial时前沿仅相对于已完整候选，不能称全网格前沿，也不能隐去被支配点。alias成本是主实验历史成本，其新增补充训练费为零；教师等嵌套分项仍不重复累加。

```bash
python scripts/analyze_lambda_sweep.py \
  --registry configs/lambda_sweep_deterministic_registry.json \
  --output results/analysis/lambda_sweep_deterministic_prepared --allow-partial

# 若使用后续调度注册表，其余参数仍绑定原预注册表及active正式父表。
python scripts/analyze_lambda_sweep.py \
  --registry configs/supplemental_v1_A_registry.json \
  --output results/analysis/supplemental_v1_A --allow-partial
```
