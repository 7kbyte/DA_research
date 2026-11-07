# v2 NLL分析适配与解封边界

实现为 `src/replay_study/nll_v2_analysis.py`，入口为 `scripts/analyze_nll_v2.py`。这是显式v2 collector，不修改冻结scorer，不将补充角色伪装为primary，也不把1B/2.8B改名为410M。当前只运行了7项合成fixture测试及主清单的prepare-only；**未读取任何真实NLL或行为输出，包括已经存在但sealed的initial行为结果**。

每次命令必须显式指定role、phase_name、model_tag、data_family和final/best。清单SHA、phase authority与registry SHA必须匹配；registry拒绝provisional/inactive训练，A合法历史alias例外但保持原actual_run_id。primary/A/B限定410M与main_v3，extended只允许extended_v1并核对各自固定model commit。A、B只允许final；primary与extended的final/best分开。预计完整条件数来自冻结phase registry，缺结果时拒绝最终输出；显式partial保留每项availability及各组expected_n，不从不完整候选挑有利点。

默认prepare-only只读manifest、registry、模型等价证明和源码SHA，不打开或探测任何分数路径。实际解封后才可由root显式传入 `--read-heldout-scores`。每个真实result必须有对应 `_execution.json`：manifest SHA、job ID、verified_frozen_inputs、baseline_group、checkpoint SHA、model-state identity和准确result SHA全部匹配。初始baseline按冻结baseline_group取独立身份路径，允许有明确initial_origin的共享，不再假定flat目录或统一的initial_pythia-410m路径。

模型/revision/precision以及完整执行事实逐项匹配冻结policy，包括TF32、全部四个SDPA开关、GPU/torch/CUDA、严格deterministic、CUBLAS workspace和模型权重dtype。corpora名称、预测token数、完整chunk数及逐chunk doc_id/identity必须一致；KL要求全词表teacher||student，sum/count与逐chunk聚合一致。原始initial自KL只允许记录的数值零范围。

logical inference alias须有零额外forward、零新增进程时间、canonical job/output、双方checkpoint SHA及对应执行证明；还会读取已授权canonical结果，核对corpora和execution完全相等。不同checkpoint文件只可依据完整model_state SHA等价证明复用：证明文件SHA、comparison索引、左右完整指纹、checkpoint SHA及算法均校验。D的权重/推理复用不免除已经真实运行的D训练成本。

输出保留logical runID、actual runID、role/phase/model/data/endpoint、训练/生成/selection seeds及实际曝光。best的target/replay曝光来自冻结record，完整训练成本另外保留，绝不一律标成10M。训练成本按model_tag+actual_run_id去重；A的18个历史主条件alias新增训练费为0。physical inference按job ID去重，alias内复制的计时不相加，teacher子计时也不重加。committed训练费用不冒称包含失败重算的整个项目开销，项目成本inventory仍单独报告。

primary/A/extended按训练seed报告逐seed及mean、样本SD(ddof=1)、n；B按selection_seed分K报告，固定training_seed=generation_seed=11，明确仅为条件选库变异。六项预定同b/λ/LR比较一并保留：M−B4、C−B4、B2−B1、B4−B3、B3−B1、B4−B2。A全λ候选均保留；test分析只描述前沿，**不执行test λ选择**，实际λ选择仍由独立validation-final规则及其SHA记录决定。

文档bootstrap沿用原纯统计函数：每次同一doc同时采两模型，保留全部chunks，分别给预测token加权和等文档平均；默认5,000次、seed20261106、95%区间。逐模型对initial的Gnew/Fold/KL，以及模型间差值都有相应区间。区间只条件于已训练checkpoint，不替代训练seed SD；B也不能靠文档bootstrap变成多个独立训练seed。Gnew为initial−adapted的目标NLL，Fold为adapted−initial旧域NLL，成对效应为前者方法减后者方法，raw paired NLL区间与Gnew差值符号相反，文件明确记录方向。

解封后输出endpoint/paired CSV、全候选前沿CSV、逐文档bootstrap JSON、实际训练/物理推理成本表、SHA分析清单，以及英文标签PNG/PDF/SVG散点图和中文说明。seed/draw点与均值±SD同时显示，partial图明确标记；不混模型、数据版本、角色或端点。

主清单prepare-only命令如下，尚未加读分数开关：

```bash
python scripts/analyze_nll_v2.py \
  --manifest docs/primary_evaluation_v2_execution_matched_nll_manifest.json \
  --manifest-sha256 d1502bcaff5cdcd75d984a4b8776f1cb4027b596f474835bfe9ed18875d9fd18 \
  --role primary --phase-name primary --model-tag pythia-410m \
  --data-family main_v3 --checkpoint final \
  --output results/analysis/nll_v2_prepared/primary_final
```

best使用独立 `--checkpoint best` 与独立输出目录，已同样prepare-only核验36条。A/B/extended须先由评估工程生成并冻结各自真实manifest，再传其真实role、phase_name、model_tag、data_family与SHA；不通过修改现有primary清单身份绕过门禁。完整性/别名/模型/数据/执行/SHA问题均失败即停，不静默复用旧superseded输出。

主 NLL 新增完整批次门禁：`scripts/monitor_primary_nll_readiness.py` 只检查冻结主 67 项 scheduler 状态，全部成功前不探测、哈希或解析任何评分结果；全成功后核验冻结 protocol/registry/scorer/data 身份、67 个物理任务及 6 个零 forward 别名的 attestation 和73份输出字节 SHA，仍不解析分数。`scripts/analyze_primary_nll.py` 默认仅准备；必须提供 root 成功入队确认后，且门禁完整通过，才执行 final/best 各36端点、5000次成对文档bootstrap与六组预定对照。进入第一份真实 test JSON 解析之前，写入不可重置的 `docs/primary_nll_first_test_read.json`，绑定 UTC、manifest SHA与完整性收据。final/best 共享此首次读取时间。所有训练方法、grid、horizon及验证选参规则此前冻结，不因 test 前沿调整。

准备命令与源码 SHA 见 `configs/primary_nll_analysis_execution_prepared.json`；此记录生成时尚未收到成功入队确认，未读取真实主测试分数。主 NLL 唯一67任务成本另外合表，initial只算一次，D/B4别名无额外forward。逐端点成本表不可跨final/best直接累加。行为及其他phase继续封存。
