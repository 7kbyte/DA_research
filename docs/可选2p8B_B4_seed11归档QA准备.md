# 可选2.8B B4 seed11 QA：仅CPU身份准备

**尚不可执行。** 固定对象为既定`extended_v2_pythia_2p8b_m8_h25m_formal_B4_s11_w0p05`，不按分数挑选：765步、25M目标/1.25M回放预测token，generated KL、b=λ=.05、LR=1e-5、micro8×accum8，base revision `2a259cdd96a4beb1cdf467512e3904197345f6a9`。配置、snapshot、extended protocol、实际严格精度provenance均已绑定。未运行训练/评分，也未读取任何完整原checkpoint或复制归档。

新增仅为[ROOT FALSE操作模板](optional_C2p8b_B4_seed11_QA_ROOT_FALSE_TEMPLATE.json)和[单run/双端点expected身份审阅模板](optional_C2p8b_B4_seed11_QA_expected_identity_TEMPLATE.json)。后者含明确pending字段，**不是可交给verify的已验收expected**，没有伪造plan/manifest/完整C包。现有main4 QA所有计划、配置、queue、验收器、归档helper/core和codec保持原字节。

- **final证据已具备。** 只读已验证exporter draft小JSON及19,584字节metadata，核源config/summary/snapshot SHA及当前checkpoint五stat。final全state指纹为`0a819cb2921ba45adfd57b80ddf61f10cbdfeddbc4fb0a3d8846d53a384701b2`，2,775,208,960个元素；源CP SHA为`1cd4e86e0f99c39b2fbc184584850b3fae357a34f97511678e53849bbd9d07e5`。draft的encode/decode/逐tensor全字节核验通过，但它仍是`verified_draft_not_deliverable`，不冒充持久包验收。
- **best指纹仍缺。** summary的选点身份为step612、target20,014,848、replay1,000,742；未据此读取或重选验证分数。未找到该run best的完整state指纹＋源CP SHA小证据，不能从final推断。源best大小11,101,003,630字节。未来若root另授权CPU大读，最薄方案为一次CPU加载取得真实kind/config/counters/selection，并调用现有、SHA固定的`weight_archive.model_fingerprint(state)`；再fresh顺序读全CP SHA，前后五stat固定，独占保存小证明，无需新archive。约22.20GB名义序列化读取，另触及约11.10GB权重；这是规模估计，尚未执行或计时。
- **持久base闭包仍缺。** 实际C2.8完整持久包尚未发布。未来必须核真实持久验收及离线base/config/tokenizer闭包，不能以当前下载目录或单final draft替代。操作模板列出全部六个pinned base文件SHA；未在本次重读5.68GB base或大TAR。

**复用已有执行与验收。** 可复用现有0732归档恢复树中的原helper、scheduler和十个core；新C namespace及独立scratch与main4分开，当前未创建其config/queue/state/output。归档源C config与原SHA一致，十个core与C snapshot一致。归档入口显式优先导入自身`src`，操作模板也固定`PYTHONPATH`到该归档树，避免live源码混入；未来scheduler只绑定一张实际空闲H100，child仍用`cuda:0`。未来须fresh逐文件审计该run实际输入集合，不把main4的“34项”机械套到新模型。

通用`reproduce_experiments.py relocate`会把queue allowance硬编码600秒，因此只能作未来输入验证/relocation和`reproduce` dry-run。保留其600秒队列原字节；另作仅`expected_seconds: 600→2340`的独立runtime队列及SHA/definition差异证明，直接交原归档scheduler。**不得用generic `reproduce --execute`启动本单项。** 当前只在FALSE模板内列出单项2340秒及`--launch-margin-seconds 120 --stop-margin-seconds 120`，未落地实际queue或入队。

既有`verify_reproduction_qa_postrun.py verify`已支持任意正数run，单run仍要求final+best两个完整端点；新增临时CPU测试5项通过，包含单run成功、无真实重训/缺best拒绝、best最后tensor字节改变拒绝，以及现有state指纹与归档原file算法一致。**无需修改验收器。** 其`prepare-expected`入口依赖main权重manifest结构，不能给它伪造C primary manifest；将来只能在真实best证明、持久闭包、relocation/运行plan全部具备后补齐独立C expected模板，再原样调用verify。验收仍查全state、新CP整文件SHA、五stat、实际曝光与严格policy；不声称比较全部轨迹/optimizer。

**启动与时间边界。** 所有科学训练、NLL/behavior评分及分析结束，priority main4真实QA完整验收后，旧scheduler及全部children/PGroups退出、四卡空闲，root才能考虑单卡可选项。root还须绑定实际来源、完整expected、2340秒job、新GPU和操作authority；本模板永远不自动授予权限。

至少还剩60分钟到`2026-11-07 02:56:32 UTC`，故全部条件满足后的实际启动不得晚于**01:56:32**。scheduler的算术准入界是严格早于02:15:32（2340＋120秒），这不能放松root的60分钟门槛。02:54:32会请求停止running child，90秒后可SIGKILL，轮询和回收不是硬实时保证。若高情景科学流程约02:00才结束，可选项便不满足该门槛，应跳过。

原run whole-slot为2125.934136秒；2340只是新操作allowance。60分钟减2340还剩1260秒，须覆盖输入/base fresh核验、周转、完整CPU后验及持久化。新final约33.30GB、best约11.10GB；沿用原验收器三次独立读，两端点名义约**133.21GB**序列化文件规模，实际随机读/重复头部及wall以真实receipt计。main4的300秒仍是旧规划，不能冒称2.8B后验实测或保证。全QA与CPU证据费用单列工程诊断，增加零个科学重复；保留02:56:32之后完整30分钟交付尾部。

测试入口：`CUDA_VISIBLE_DEVICES='' pytest -q tests/test_optional_extended_qa_identity.py`。本次全部实际操作只涉及小元数据、源码和temporary tiny tensors；缺失依赖保持pending，不新开GPU或分析流程。


## 补录：ROOT专项授权的原best CPU身份证明已完成

本节更新于2026-11-06T13:20:04.683112+00:00；以上“best仍缺、未读完整CP”保留为首次准备时点记录。ROOT随后仅授权固定原best的CPU完整读取，不授权QA训练、relocation、新archive或分数分析。现已补齐该单项依赖，原两份FALSE/expected模板保持原字节且不变为可执行authority。

独占新目录为`results/reproduction_qa/optional_C2p8b_original_best_identity_v1/`。producer先复核原run completed/rc0/no-stop、765步/25,000,000目标/1,250,000回放、实际job definition/config/summary/registry/snapshot及严格数值provenance；随后在空CVD、CPU两线程下加载原best。实际best为step612、target20,014,848、replay1,000,742，kind=`model_only_best_target_validation`。selection只保存opaque SHA及选点身份，未输出或分析验证NLL，未重新选点。

SHA固定的现有`weight_archive.model_fingerprint`遍历完整**388个tensor、2,775,208,960个FP32元素**（11,100,835,840权重字节），所得model-state指纹为`1c142978e5969417f8f1e54226aa7c960fbff231270cb1de8328d05c0a7eb1c6`。另一次fresh顺序整文件读取至EOF取得serialized checkpoint SHA `4d813191a3d598a66142106bd00533bf3e7eb3e828abd3abb82578dbdf1bb49d`，与已验证prewarm receipt一致。两个SHA表示不同身份：前者涵盖按key/dtype/shape排序的完整权重，后者涵盖整个序列化文件；不把它们互相比较，也不称这已验证某次QA重训。path和打开fd的dev/ino/size/mtime/ctime在前后全部一致。

完整子进程从spawn前至退出的实测为**wall 36.585372s / user 17.739065s / sys 18.709136s**，含导入、来源门禁、加载/完整state指纹、fresh整文件SHA与退出。嵌套子项：加载＋完整state指纹24.406903s；fresh整文件SHA 10.219854s，不能再加到整进程wall。实际序列化返回读取**22,201,973,020 bytes（22.201973020GB）**：torch.load 11,100,969,390，fresh SHA 11,101,003,630。两遍完整文件大小的名义估计为22,202,007,260 bytes；loader seek跳过34,240字节非tensor ZIP部分，因此名义估计不是实际读取下界。完整state均已遍历，fresh SHA仍覆盖全部11,101,003,630文件字节。读取计数不是物理磁盘/page-cache I/O，内存中权重摘要触及量也不再计入CP文件读取。

独立小证据复核重新核源码/模板/registry/config/summary及当前stat，验证opaque selection、真实best曝光、整个state receipt与prewarm源文件身份绑定；额外CP payload读取为0。该复核是独立receipt/source绑定检查，未声称再算一遍11GB model指纹；其wall 0.008839s为证明进程结束后的另一次CPU操作。

主要证据及SHA：

- producer：`scripts/prove_optional_C2p8b_best_identity.py`，`28709318c02aebc17e38af0bfd9dac3b8795fdebd9663c2e921d133998206743`。
- 完整证明：`results/reproduction_qa/optional_C2p8b_original_best_identity_v1/worker/proof.json`，`3d8ab51efa4fb3fa427bb0b0241141d9acda40e4d5b9a851689b5e737636cd5e`。
- 完整父进程计时：`results/reproduction_qa/optional_C2p8b_original_best_identity_v1/whole_process_timing.json`，`1b69ca0a7242a8b904d49410c6d204dcb99ac2e552e5325d18eda52f2c8082f2`。
- 独立绑定复核：`results/reproduction_qa/optional_C2p8b_original_best_identity_v1/independent_binding_check.json`，`9b056dea749008d3fdd47a1fc5bfc0241f0fcc7eba96c7eb6e4f89df9b4687fe`。

**仍不可启动QA。** 持久C2.8完整包/base闭包、真实relocated运行plan＋完整expected、后期ROOT显式TRUE操作授权及全部时间/四卡空闲/旧PGroups退出门禁仍pending。原final draft、checkpoint、main4 QA文件和两份FALSE模板均未改；未创建实际可选QA训练namespace或入队。main4的300秒CPU后验仍是规划，以上36.59秒是“原best身份”单项实测，不能替代未来新final+best验收成本。
