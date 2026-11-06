# 三模型 B4 seed11 归档QA：条件准备

这是 [ROOT预声明](optional_three_model_B4_archive_QA_ROOT_predeclaration_v1.json) 的CPU准备，**没有GPU权限，没有真实队列，没有新训练**。[FALSE操作模板](optional_three_model_B4_archive_QA_ROOT_FALSE_TEMPLATE.json) 和 [六端点expected模板](optional_three_model_B4_archive_QA_expected_identity_TEMPLATE.json) 含明确pending项，不能直接交给验收器当成已冻结期望。旧单2.8B模板、0732原目录、所有原checkpoint与运行源码均未修改。

| 固定原条件（均B4、seed11、b=λ=.05） | final目标曝光 | 原训练parent秒 | 未来runtime expected_seconds | 原best身份 |
|---|---:|---:|---:|---|
| C410M，m64 | 100M | 2145.995 | 2700 | 等全阶段、源绑定D/B4 best证明；本次未读大CP |
| C1B，m32×2 | 100M | 3058.806 | 3600 | 本次CPU完整证明通过；step612、20,014,848 target、1,000,742 replay |
| C2.8B，m8×8 | 25M | 2125.934 | 2340 | 引用既有CPU完整证明；同样best曝光，不从final推断 |

完整run ID、原config/registry SHA、模型revision、final指纹和未来新路径都在模板内。三份final均只引用正在运行的C exporter已经完成的不可变roundtrip记录；这证明原final身份，**不等于C包已持久发布**。C1B/2.8B基座持久包闭包与提取仍为pending；410M虽已有primary持久基座，未来独立scratch的提取、fresh输入审计也尚未做。

恢复来源是一个明确组合：**0732冻结训练输入/core/helpers + 1415后建CPU验收工具 + 将来实际验收的C基座包**。本次对0732已解出目录的116个不同所需payload重新核SHA，共182,221,436 bytes：三模型各10个core，冻结data条目分别46/46/34，另含原配置、registry、protocol与input snapshot；均和0732 PACKAGE_MANIFEST及冻结输入一致。reproduce、scheduler、完整指纹audit helper与模型来源catalog也齐全。0732没有后建 `verify_reproduction_qa_postrun.py`；本次仅从真实1415包按需取它和PACKAGE_MANIFEST，工具fresh SHA为 `5417fc0747d94feae828e0e53133edf66745b2fb60885cc591f84f2feceea7e2`，放在独立assessment-tools目录。依据已发布收据与当前stat，不重算整包SHA，不改0732目录。

只有所有科学训练、评价及完整分析、优先main4归档QA先完成，旧scheduler与子进程全部退出、四卡空闲，并且距 **2026-11-07 02:56:32 UTC至少4800秒**，ROOT才可考虑新的三卡并行权限；最迟实际启动 **01:36:32 UTC**。否则保留旧单2.8B的≥3600秒方案，旧模板不变。三模型条件在C测试前固定，是否执行只取决于完整输入闭包与时间。CPU postcheck **600秒只是计划，尚无三模型新训练后整体验收实测**；不能拿本次原best身份证明的秒数替代它。

未来执行顺序（现在不执行）：

1. ROOT验收实际C持久包/base成员，向新的 `optional_Cthree_B4_seed11_v1/scratch/models/` 按模型提取；取得410M完整best证明。把本次新证明与最终操作计划纳入真实后续快照，绑定完整来源SHA。
2. 从三个已绑定的0732原registry各取固定那一行，组成新的source-subset JSON，原registry不改。使用归档原 `scripts/reproduce_experiments.py relocate`：一个新namespace，三个 `--run-id`，重复三次 `--input-manifest <原模型freeze>`，指定模板中的新workspace/scratch及 `--deadline 2026-11-07T02:56:32Z`。fresh核core/config/model/token输入；保留生成的原600秒队列、provenance和dry-run输出。不要使用 `reproduce --execute`。
3. 另存runtime队列，逐job仅将 `expected_seconds` 从600改为 **2700/3600/2340**；字节保存两版，精确比较除该字段外的全部job定义并绑定SHA。补齐实际relocation、scientific identity audit、registry、queue、六端点metadata/selection路径映射及expected文件SHA；selection不重选，原值按已有记录精确映射。pending项未补齐不能产生ready/verified结论。
4. ROOT再次绑定实际时间、三张空闲物理H100和新操作权限后，才可使用模板中的归档scheduler命令。统一deadline `02:56:32Z`，launch/stop margin各120秒，隔离归档 `PYTHONPATH`、HF cache与新输出；不得加入原科学master queue。
5. scheduler及子组退出后，使用1415提取的验收器：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python <assessment-tools>/research/scripts/verify_reproduction_qa_postrun.py verify --expected <真实已冻结expected.json> --expected-sha256 <SHA> --output <新的独占receipt.json>`。检查三run/six endpoints完整、rc0/no-stop/exact argv、config/inputs/policy/曝光/selection、新CP inode、fresh整CP SHA与全部model-state指纹。各原训练、未来训练、CPU验收及持久化成本分开记录，子计时不重加。

C1B本次实际证明见 [worker proof](../results/reproduction_qa/optional_C1b_original_best_identity_v1/worker/proof.json)：1,011,781,632个FP32元素、4,047,126,528 tensor bytes，全指纹 `a6ef2de1d4094a2967160742421dcabe63fa8df99aca96d3f282b32f135c109b`；新鲜CP SHA `af699b61e5429e084fa70a42b5d9cd2fb0f0fb399307fd9bb2d43e3688f262a0`，与历史完整读一致。path/open-fd五项stat稳定，config/原argv/rc0/no-stop/counters/执行策略均核对；selection仅输出opaque SHA，不输出NLL。完整子进程spawn至exit **11.109551秒**，user **6.994233**、sys **4.195830**；worker及load/FP、freshSHA子阶段均包含其中。serialized返回读取8,094,408,500 bytes不等于物理磁盘读量。

[CPU准备来源收据](optional_three_model_B4_archive_QA_CPU_preparation_receipt_v1.json) 与 [1415按需工具提取收据](../results/reproduction_qa/optional_three_model_B4_preparation_v1/assessment_tool_extraction_1415.json) 记录范围与独立计时。本准备不能宣称重新训练成功、Adam续接、新容器或跨硬件逐位复现；没有读取C或behavior评分，也没有生成新科学重复。
