# 损失权重敏感性补充实验预注册

生成时间：2026-11-06T04:44:34.534730+00:00。依据主结果及测试损失查看之前制定的 `docs/补充实验预案.md` A；本次仅生成，不启动训练。

B1/B2/B3/B4/M/C × λ={.01,.05,.1,.5,1,5,10} × seed={11,22,33}，共 126 个网格单元。λ=.05 的 18 单元逐一 alias 到对应 formal run，复用 checkpoint/summary，不增加训练或重复计算成本；其余为 **108 个新训练任务**。alias provenance 见 `configs/lambda_sweep_deterministic_aliases.json`。

LR=3e-5，10M target prediction tokens，b=.05，K=100156；其余参数逐一继承匹配的正式配置。新配置仅改变 λ、输出目录与运行标识；使用冻结科学模块和 v3 validation/test。资源不足时先完成 B4/M 全部权重。任何单点 λ 按方法分别选择：以最终 10M 暴露时的验证 S=(Δtarget)+.5×(ΔC4+ΔWiki)，先对三 seed 无权平均，再选最低均值 λ，严格平手取更小 λ；报告完整预设曲线及两个旧分布，不选择性只展示有利权重。测试仅评估冻结选择或完整预设曲线，不用于调参。

配置：`configs/lambda_sweep_deterministic/`；126 单元 registry：`configs/lambda_sweep_deterministic_registry.json`；108 新任务 queue：`configs/lambda_sweep_deterministic_queue.json`。启动须等主 36 组审计及 root 决定。此文件不构成启动授权。
