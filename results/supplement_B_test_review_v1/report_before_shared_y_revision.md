# 补充B：固定生成池与训练顺序下，60个条件bank-draw端点的完整测试解释

本报告只使用 root 已明确解封的完整 B60 test 及先前已授权的 B validation、bank composition 解析参考；C 和全部行为结果继续封存。真实 READY/manifest、61项完整依赖闭包、输出/attestation、不可重置 first-read 已由原 v2 guard 验收，本次不重跑 collector、bootstrap、训练或模型推理。授权为 `f9de170e447294bec0b3fa87188702c5fa6b00e5100d2ac92affa7d268b6c730`；COMPLETE 为 `19818b92f1906eeae98226d5f3ef60104e29b3972ce2894fbdc632df346b84e7`；first-read 为 `6af4c9ac96050bd7a60578e5461fce140a12a3c252e528cfc95b59e809294578`。原分析与 first-read 均原样保留。

## 能支持的有限结论

在这个固定 pool=11、train=11、λ=.05 的60-run条件实验中，M在K196的三个测试均值方向较random/B4有利，但K20的三个方向均反向；C没有同时改善目标和两个保留域的均值。M的bank均值teacher-NLL波动较小，却没有在本样本中表现为更小的六个NLL终点样本SD。结果支持“单个组成标量更稳，不保证所观察训练终点更稳”的有限边界；不能据此声称所有梯度方差不降、分层永不有效，或选择一个测试后最优K/方法。

所有60个端点的token口径C4、Wiki遗忘均为正，故均值间几千分之一的改善不等于消除遗忘。目标token增益均值仅约−.0027到+.0082；等文档目标增益则约+.0192到+.0290，必须同时报告加权口径，不能只挑有利的一列。

## 条件、重复单位和指标

B4=random，M=teacher-NLL五分位分层，C=cluster representative；3方法×2容量（K20、K196）×10个selection seeds（1001–1010）=60个物理训练run，无训练/推理alias。生成池seed始终11，训练shuffle seed始终11；这里的 n=10 是**固定池和固定训练顺序下的selection-draw条件波动**，不是十个独立训练seed、十个生成池，也不能加入主三seed计算n=13。cluster的随机初始化/选代表也属于该实现条件下的draw变化。

各端点均为final 10,000,000 target + 500,000 replay prediction tokens，b=.05、λ=.05、LR=3e-5，FP32权重+BF16 autocast、strict/auto等执行policy与配套initial一致。没有把A验证选择后的λ5移入B；没有事后选best或修改λ/K。K20容量10,220预测tokens，K196容量100,156；累计replay除容量分别约48.9237和4.9922次，因此K比较同时描述库容量和重复利用程度，不能解释为只改变一个抽样噪声标量。

G_new = initial target NLL − adapted target NLL，越大越好；F_C4/F_Wiki = adapted − initial NLL，越小越好；单位nats/token。等文档口径先在每文档内按有效预测token求均值，再赋每文档相同权重。以下所有“±”为10个selection draws的sample SD（ddof=1），不是SEM、训练seed CI或文档bootstrap区间。完整60个个体和40个同标签配对均保留，不因负增益或极端值剔除。

## 六个条件的测试均值和sample SD

Token加权：

|K|方法|G_new ↑|F_C4 ↓|F_Wiki ↓|
|---:|---|---:|---:|---:|
|20|B4|-0.000243 ± 0.012014|+0.168306 ± 0.010803|+0.171196 ± 0.011375|
|20|M|-0.002738 ± 0.012626|+0.169465 ± 0.013561|+0.175609 ± 0.017856|
|20|C|+0.001383 ± 0.014221|+0.172084 ± 0.017321|+0.174385 ± 0.016630|
|196|B4|+0.004192 ± 0.009500|+0.149419 ± 0.010074|+0.149295 ± 0.008102|
|196|M|+0.008244 ± 0.010760|+0.145004 ± 0.010326|+0.143021 ± 0.008464|
|196|C|+0.003344 ± 0.011057|+0.158017 ± 0.010616|+0.150576 ± 0.010414|

等文档加权：

|K|方法|G_new ↑|F_C4 ↓|F_Wiki ↓|
|---:|---|---:|---:|---:|
|20|B4|+0.021654 ± 0.011040|+0.166849 ± 0.010692|+0.170428 ± 0.011798|
|20|M|+0.019154 ± 0.011617|+0.167935 ± 0.013520|+0.174956 ± 0.018476|
|20|C|+0.022755 ± 0.013224|+0.170744 ± 0.017421|+0.173404 ± 0.016731|
|196|B4|+0.025385 ± 0.008922|+0.147874 ± 0.010028|+0.148959 ± 0.008312|
|196|M|+0.028999 ± 0.009599|+0.143675 ± 0.010223|+0.142520 ± 0.008413|
|196|C|+0.024840 ± 0.010262|+0.156592 ± 0.010587|+0.149348 ± 0.010122|

两种加权下，K20的G均值排名均为 C>B4>M；C4保留为 B4>M>C，Wiki保留为 B4>C>M。K196三个指标均为 M>B4>C（这里“>”表示按相应指标方向较好）。六条件的token/doc均值排名相同，并不意味着数值或每draw方向相同。排名来自全部既定条件的描述，不作为新选择规则。

目标test包含466文档、1024chunks、523,264预测tokens，每文档1–4块：200/96/48/122个文档。单块文档占文档权重42.92%，却只占token权重19.53%；四块文档占文档权重26.18%，占token权重47.66%。以B4/K20为例，同文档、同10draw计算的单块文档平均G为+.067995，四块文档为−.023269；K196对应+.070240/−.017894。更多块的文档在token口径下权重较高，足以使K20/B4的G从等文档+.021654变成token−.000243。这是冻结样本上可直接核对的权重分解，不是更长文档必然更难、原始文档长度效应或因果机制证明；不能更换主token指标。完整24行分解保留两种权重和各自贡献，六个条件均可重构相应G；独立权重解释脚本再次复核24行及12个加权恒等式，最大绝对误差6.37e−16。

C4测试595文档/1024chunks、523,264预测tokens；Wiki测试60文档/529chunks、270,319预测tokens。Wiki每文档块数差异更大，等文档口径的含义仍是每个所定义文档单元等权，不能自动视作独立主题总体。目标G正值个体数（token/doc）为K20：B4 token 3、doc 10；M token 5、doc 9；C token 4、doc 10。K196：B4 token 6、doc 10；M token 9、doc 10；C token 6、doc 10。这里每个计数都以10draw为分母，非“十个训练seed”。

## 同selection-seed标签的配对差

先在同K、同selection-seed标签下求M−B4或C−B4，再对10个差取均值/SD。两个算法使用相同数字seed，不保证消耗相同随机数、选到相同bank或实现严格common random numbers；报告只称同标签配对，不能借此宣称CRN方差缩减。两个方法的SD不能相减来代替配对差的SD。

Token加权配对差：

|K|配对|G_new ↑|F_C4 ↓|F_Wiki ↓|
|---:|---|---:|---:|---:|
|20|M-B4|-0.002495 ± 0.021863|+0.001159 ± 0.022152|+0.004414 ± 0.024368|
|20|C-B4|+0.001626 ± 0.018632|+0.003778 ± 0.020954|+0.003190 ± 0.020573|
|196|M-B4|+0.004052 ± 0.014265|-0.004415 ± 0.013233|-0.006274 ± 0.010198|
|196|C-B4|-0.000848 ± 0.012093|+0.008599 ± 0.011142|+0.001280 ± 0.010417|

等文档配对差：

|K|配对|G_new ↑|F_C4 ↓|F_Wiki ↓|
|---:|---|---:|---:|---:|
|20|M-B4|-0.002500 ± 0.019977|+0.001086 ± 0.022002|+0.004528 ± 0.025458|
|20|C-B4|+0.001101 ± 0.017213|+0.003895 ± 0.020858|+0.002976 ± 0.021409|
|196|M-B4|+0.003614 ± 0.013143|-0.004199 ± 0.013106|-0.006439 ± 0.010549|
|196|C-B4|-0.000545 ± 0.011587|+0.008718 ± 0.011213|+0.000389 ± 0.010580|

K196，M−B4的token均值为ΔG+.004052、ΔF_C4−.004415、ΔF_Wiki−.006274；三个差的sample SD分别.014265/.013233/.010198，个体差并非一致有利。K20均值则为ΔG−.002495、ΔF_C4+.001159、ΔF_Wiki+.004414。C在K20目标增益略高、两域遗忘也更高；K196目标略低、两域遗忘更高，没有支持稳定的joint retention优势。

下表为差的有利/不利方向计数；本例无精确零差，两种加权下所有计数相同。它们不是显著性、独立问题样本数或总体胜率估计：

|K|配对（左−右）|G 有利/不利|C4 有利/不利|Wiki 有利/不利|
|---:|---|---:|---:|---:|
|20|M-B4|5/5|6/4|5/5|
|20|C-B4|6/4|3/7|4/6|
|196|M-B4|7/3|7/3|8/2|
|196|C-B4|5/5|2/8|5/5|

例如K20的M−B4虽有6/10个C4差为有利，均值仍为+.001159：少数较大的反向差可抵消多数小的改善。K196的M虽然有7/7/8个draw在G/C4/Wiki方向有利，仍保留3/3/2个反向个体；不表述成“每次改善”。

## 已冻结的条件文档bootstrap

仅整理原分析已有5000次、seed=20261106、95% percentile paired-document bootstrap，不重新抽样、不新增p值或等效检验。每次重采样将一个文档全部chunks一起抽取；同一对checkpoint使用相同文档索引。Token口径每次重算总NLL差/总token数；等文档口径平均文档内均值。原paired bootstrap保存的是NLL A−B，本次展示G差时同时翻转点估计和区间两端；与原40个配对差一一核对。

这些区间条件于**某一个draw训练得到的两个checkpoint**，不是10draw均值的CI，不替代selection-draw SD，更不描述训练/生成seed总体不确定性。按文档独立/可交换重采样的假设还可能受同源内容/主题相关性限制，不能自动称为目标总体的置信区间；240个逐draw、逐指标区间也不提供多重比较校正。

一个边界清楚的例子：K196、M−B4、draw1001的token ΔG=.018049，条件区间[.016057,.019954]；draw1005为−.024406，条件区间[−.026229,−.022537]。两者各自文档区间都较窄，却有相反的bank-draw方向，说明文档层面的精确性不能消除bank选择层面的差异。该K下10个token ΔG条件区间7个全正、3个全负；C4为7个全有利、3个全不利；Wiki为7/2/1个全有利/全不利/含0。这里只计现成条件区间，不将其重新汇总为一个总体CI。

完整文件包含360行端点六指标区间、240行配对六指标区间和24行区间方向计数；KL原始区间仍在既有原分析目录，KL drift不替代NLL或行为能力。

## 与validation和事前bank组成解析的关系

B validation完整60-run分析在test解封前已经完成。Validation的K196 M−B4均值为G+.003767、C4−.004145、Wiki−.005135；test对应+.004052/−.004415/−.006274。K20 validation为−.003160/+.000943/+.003996，test为−.002495/+.001159/+.004414。两个K、三个指标的均值方向保持一致，但绝对G的差异很大：例如B4/K20的validation G+.018148，test token G−.000243；不能把验证均值复制成test、用测试重选K或推断分布完全一致。Validation导出的等文档字段为空；本次等文档数值只来自真实test的逐文档数据。

既有固定1957chunk总体的teacher-NLL有限总体解析，和本次10draw的观察SD必须分开。下表直接引用已存在的 `conditional_nll_sampling_reference.json` 与已授权composition统计，不重算、不抽新bank：

|K|选库|bank均值NLL解析SD|解析期望相对pool偏差|观察10draw bank均值NLL sampleSD|
|---:|---|---:|---:|---:|
|20|random/B4|.187623|0|.147085|
|20|M|.073860|+.000799|.075304|
|196|random/B4|.057146|0|.053865|
|196|M|.022566|−.005407|.018086|

M的整数配额K20=4/4/4/4/4，K196=40/39/39/39/39；层大小392/392/391/391/391且没有重要性重加权，故存在上述期望偏差。解析SD条件于固定pool和无放回抽样机制，观察SD来自1001–1010这10个draw；两者都不是梯度/训练终点/单库内部SD。C的组成均值SD为K20 .078715、K196 .029696，机制不同，不套用random/M解析公式。

本次test里，M和C的六个端点（token/doc G、C4、Wiki）SD在两个K均高于B4；特别M组成统计更稳仍没有对应的端点SD下降。这只是一组可观察的有限样本边界。没有直接测梯度，也没有对所有梯度或任意投影方差作结论，不能从非线性训练端点反推已经识别的优化机制。K196较K20在三方法均值上遗忘更小、G更大，仍只是既定条件的敏感性描述；本报告没有据test决定之后的K、方法或λ。

## 费用和计时层级

|项目|实测墙钟秒|含义与避免重复计费|
|---|---:|---|
|60个物理训练run setup|201.807|committed-path父项|
|训练父项|11868.985|已含training teacher等子项|
|validation父项|7974.654|已含validation teacher和KL arithmetic|
|checkpoint父项|474.845|与以上父项并列|
|上述训练父项之和|20520.290|不是研究elapsed或纯GPU时间；不再加子计时|
|60个新test整占槽之和|2406.629|逐job scheduler whole-slot；不再加scorer/verify|
|其中scorer subprocess|1689.261|嵌套于整占槽，含模型加载/teacher等|
|其中input verification|341.590|嵌套于整占槽，与scorer均不可再加父项|
|首个test启动至最后退出calendar窗口|922.841|共享master混合运行窗口，非独占四卡耗时|
|60次bank selection wall之和|24.841|独立准备阶段；其中clustering 22.640已含于selection|
|共享既有pool11生成|119.045|引用既有资产，不是B重新生成60次|
|共享既有pool11 joint NLL/features|8.103|一次共用准备，不按三方法或60run重复收费|

Initial baseline已经完成并复用，B新增initial推理为0；61闭包不等于61个新增forward。共同pool/stats也服务主实验，不能把其费用完整重复计入每个使用方法；随机选库本身不需要teacher stats，M/C独立部署成本需另看既有分项成本诊断，而不是给各方法重复摊全部joint费。

B原CPU collector由手动Popen启动，没有外层wall/process-CPU计时器，故**完整CPU分析process wall和CPU耗时均记unknown**。`whole_phase_readiness`的1.310秒只是guard局部操作；first-read和analysis manifest生成时间最多构成排除preflight的calendar区间，不能冒充完整分析wall。该CPU分析发生在推理完整结束后，不含在2406.629秒GPU整占槽中。

本次独立“源SHA+数值复核+两张图”的实际wall为3.416秒、process CPU为4.952秒，独立于原collector和GPU；CPU可因底层多线程而大于wall，不能与并行阶段墙钟相加后称研究elapsed。计时不含后续交互撰文和收据序列化；文档权重表最初构造未单独计父级时间，保留unknown，未伪报为0；后来独立61个输出SHA与24行权重重构复核另有实测wall 0.603秒、CPU 0.601秒，不能追溯冒充最初构造耗时。下载/环境/未结构化CPU写作和整项目归档等不在这些精确费用的覆盖范围内，整研究elapsed/能耗另用timewindow与GPU遥测账本。

## 产物与核验

- [全部60端点CSV](../results/supplement_B_test_review_v1/all_60_endpoints.csv)、[六指标mean/SD](../results/supplement_B_test_review_v1/six_metric_statistics.csv)、[六指标排名](../results/supplement_B_test_review_v1/six_metric_rankings.csv)。
- [40个配对全指标CSV](../results/supplement_B_test_review_v1/all_40_pairs.csv)、[配对六指标mean/SD](../results/supplement_B_test_review_v1/paired_six_metric_statistics.csv)、[240个逐draw方向](../results/supplement_B_test_review_v1/paired_draw_directions.csv)。
- [条件配对文档区间](../results/supplement_B_test_review_v1/conditional_paired_document_CIs.csv)、[端点条件文档区间](../results/supplement_B_test_review_v1/conditional_endpoint_document_CIs.csv)、[validation/test对照](../results/supplement_B_test_review_v1/validation_vs_test.csv)。
- [目标文档权重24行分解](../results/supplement_B_test_review_v1/target_doc_weight_decomposition_24.csv)、[其独立重构脚本](../scripts/describe_supplement_B_document_weights.py)、[权重重构验收](../results/supplement_B_test_review_v1/target_doc_weight_decomposition_verification.json)、[费用分项JSON](../results/supplement_B_test_review_v1/cost_accounting.json)。
- [六条件7×6.5inch PDF](../results/supplement_B_test_review_v1/six_conditions_token_7in.pdf)与[同标签配对draw PDF](../results/supplement_B_test_review_v1/paired_draws_token_7in.pdf)，同目录提供PNG/SVG及配套CSV。点为全部10draw，菱形/误差棒为mean±sampleSD；paired标号01–10对应1001–1010，无CI误作SD。两列按实际数据自动刻度，数值比较以表格/CSV为准。
- [独立审阅脚本](../scripts/review_supplement_B_test.py)、[12项CPU不变量tests](../tests/test_supplement_B_test_review.py)、[审阅收据](../results/supplement_B_test_review_v1/review_receipt.json)。

独立复核共1320项标量一致性，覆盖从原始corpora/chunks重新重构全部60×6指标、40×6配对、均值/sampleSD和原bootstrap点估计；最大绝对误差2.8588e−15。完整输出、原授权/first-read、128个collector绑定来源和新增图表均有SHA追溯；本次没有改运行源、原分析manifest、研究报告.md、README、全局状态、选择规则或其他封存结果。
