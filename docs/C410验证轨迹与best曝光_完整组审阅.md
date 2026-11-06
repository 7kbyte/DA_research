# C410验证轨迹与best曝光：完整组独立审阅

本次仅审阅完整的 C410（100M）validation：8 方法 × seed11/22/33，共24个真实训练、480个验证点。watcher已标整阶段 complete，source audit 为 passed、24/24、无 source violations；当前 scheduler 的24项均 completed/rc0、无 stop request，定义与冻结 phase queue 一致。未读取任何 test NLL、behavior 分数或其余 C 模型 validation。

独立从原训练 summary 重算最小 target validation NLL，并与 events、既有整阶段 validation CSV 逐数值精确比对。best 使用冻结的严格下降规则，完全相等时保留较早点；这里只核验已执行政策，没有重选 λ、LR 或端点。全部字段核对无差异；完整来源 SHA 和计数见 review.json。

## 全24项best曝光

每格均按 seed11 / 22 / 33 排列。token 均为实际参与 loss 的 causal prediction-token 累计曝光，非 raw/padded 输入，也非唯一库容量。下表保留所有方法与 seed。

| 方法 | best step（11/22/33） | best目标曝光（11/22/33） | best回放曝光（11/22/33） |
|---|---|---|---|
| B0 | 1224/1224/1224 | 40029696/40029696/40029696 | 0/0/0 |
| B1 | 1224/1224/1071 | 40029696/40029696/35025984 | 2001484/2001484/1751299 |
| B2 | 1224/1224/1224 | 40029696/40029696/40029696 | 2001484/2001484/2001484 |
| B3 | 1224/1224/1683 | 40029696/40029696/55040832 | 2001484/2001484/2752041 |
| B4 | 1224/1224/1224 | 40029696/40029696/40029696 | 2001484/2001484/2001484 |
| D | 1224/1224/1224 | 40029696/40029696/40029696 | 2001484/2001484/2001484 |
| M | 1224/1224/1224 | 40029696/40029696/40029696 | 2001484/2001484/2001484 |
| C | 1224/1224/1224 | 40029696/40029696/40029696 | 2001484/2001484/2001484 |

best 目标曝光分布：35,025,984 tokens：1项；40,029,696 tokens：22项；55,040,832 tokens：1项。

全部24项 final 为 step3058、目标100,000,000 tokens；B0回放为0，其余各5,000,000。best与final曝光相同的项为0/24；best约在完整目标曝光的35.03%–55.04%，其中22/24为40.03%。24项final的target validation NLL均高于各自已选best，但这只是当前固定验证集和取点政策下的轨迹描述，不能称未观测时刻的连续最优，也不能据此新设训练停止阈值。

验证每153步一次，名义间隔5,003,712目标tokens；最后另在100M评价，共20点/run。best是此网格上的验证选择结果。各方法best端点不自动等预算：B1 seed33为35,025,984，而B3 seed33为55,040,832；做best配对时须保留双方曝光。所有24训练实际都走完100M，不能按best曝光比例打折已经发生的训练费用。D/B4三seed两端点的完整模型状态等价证明仅支持对应推理复用，D仍是独立物理训练与逻辑结果。

## 主410M/10M与扩展410M/100M的解释边界

主训练容量为4,096 chunks / 2,093,056预测位置，扩展为39,140 chunks / 20,000,540预测位置。扩展包含主训练前缀，属于相关稳健性扩展，不能当独立总体重复。主/扩展还分别采用 main_v3 / matched extended_v1 heldout，不能共用NLL initial作跨family基线。

有效batch虽同为64 chunks、context512，主配置为micro16×累积4，扩展为micro64×累积1；浮点累积路径不同。目标曝光从10M改为100M，同时改变训练容量、回放来源池/银行条件、matched heldout以及按各自完整horizon安排的LR日程和验证网格（主每50步=1,635,200目标tokens，扩展每153步=5,003,712）。因此这不是单独改变horizon的因果对照。

银行差异须逐来源区分：410M 的 generated banks 按已存 provenance 逐字节复用主实验冻结库；C4 库从扩展 candidate pool 选择。不能把“扩展”概括为所有回放内容都改变。此处依据 results/extended_pythia-410m_bank_provenance.json 的元数据，未再次读取银行载荷。

扩展在正式训练前按各自seed101/2M validation-only校准冻结共享LR=3e−5、λ=.05（B0的b=λ=0，其余b=.05）。本次24条长程轨迹没有重新选择λ，也没有借用A后续选参。短pilot规则不保证100M长程最优；这限制解释，不构成事后调参授权。

## 可复现证据

- results/C410_validation_review_v1/best_exposure_all24.csv：全部24项含run ID、best/final step、实际目标/回放曝光与target验证NLL。
- best_exposure_by_group.csv：每组三seed的曝光、mean/sample SD及final−best描述。
- validation_trajectories_independent.csv：480点的独立重算与原分析逐值核对。
- review.json：watcher、audit、registry、summary/events/config/CSV来源SHA，完成门禁与精确核对计数。
- whole_process_timing.json：独立审阅子进程完整spawn→exit wall/user/sys；工程编写、先前人工读取和后续复核不在该timer内，未计部分仍unknown。

本说明仅补足验证选择与曝光口径，遵循《C完整结果解释框架_解封前》，不改既定科学条件、统计对照或主报告结论。
