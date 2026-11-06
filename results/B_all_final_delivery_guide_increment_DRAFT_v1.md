# B全部final包发布后的指南增量草稿

状态：未发布、不得直接合入当前交付指南。只有ROOT完成真实durable publication、收据通过且当前stat一致，才能记账。本草稿不构成发布或资源许可。

- 范围：`B_all_final_weights_lossless_v1.tar`，条件选库变异补充B的60个final端点。固定training_seed=11和generation_seed=11，selection_seed=1001–1010，K=20/196，B4/M/C共60条件；不能称10个独立训练seed。
- 本包native base、config、tokenizer、README、恢复工具均自带；60份完整encode/decode/原uint8及metadata往返的最终证据必须齐备。model-only，不含optimizer，不续接原Adam。
- 当前已交付指南5权重包+1640快照基线为121,107,449,599 bytes。发布后权重包数增至6，另含一份项目快照；总体积应为121,107,449,599加本包实际TAR bytes，不加reserve、scratch或历史快照。
- 包内逻辑记录由114增加60至174。当前60条完整manifest小记录与旧v2覆盖表join：新增源run+final/best与旧集交集为0，完整FP交集为0；联合端点171、联合FP162。仍需整包验收和ROOT发布后才能写入正式指南，不能仅凭payload数量推断FP数。
- TAR实际bytes/SHA、currentstat、发布receipt、全FP并集、保守恢复工作空间：待真实完成及ROOT发布后填。本轮不得把A93或任何其他待交付包加入已发布清单。

来源：scope v4 SHA7adb8f7f0461fd929995dd370ff2a59b2abc5c0d7750a39e745d21d5546d9de3；冻结exporter c93e0c1d…；当前指南v3及其来源收据。首次草稿生成只读小metadata，未打开TAR/ZIP/CP。

完整scratch包已通过metadata验收（仍未durable）：69,578,864,640 bytes（64.800367 GiB），TAR SHA `b8f7cb88141ad1b6526f05e92fe34abd68a5f5a13b8219b5d4957ff3035d50d1`。若ROOT以后成功发布，6个权重包加1640一份项目快照将为190,686,314,239 bytes（177.590469 GiB）；174条包内记录、171个源端点、162个不同完整FP。仅当实际发布receipt通过且当前stat匹配后，才能把这段移入正式指南。
