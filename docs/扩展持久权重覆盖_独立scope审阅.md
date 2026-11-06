# 扩展持久权重覆盖：独立scope审阅

结论：此次新增范围应为 **216个逻辑端点：A剩余93个final、B全部60个final、C三模型63个best**。这是权重保存覆盖扩充，不增加GPU科学条件、训练重复或评价范围。精确逐项run/config/summary来源和交叉引用见 [scope.json](../results/expanded_weight_scope_review_v1/scope.json)；404个小文件的SHA与核验记录见 [receipt.json](../results/expanded_weight_scope_review_v1/receipt.json)。本审阅没有读取checkpoint、ZIP、TAR或C/behavior分数，也未修改导出器、队列、运行进程或冻结来源。

| 覆盖层 | 已有或原计划 | 本次新增 |
|---|---:|---:|
| 主实验final＋best | 72逻辑端点，66权重载荷，已持久验收 | 0 |
| A全部126 final | 主λ=.05的18项＋已发布selected额外15项＝33项 | 93 |
| B：K20/196×B4/M/C×10 selection draws | 0 | 60 final |
| C：410M/1B/2.8B | 原final计划24/24/15；不在本审阅中宣称已全部持久 | best 24/24/15 |

A缺项按**已发布归档身份的集合差**计算，没有再次看表现或重新选择：B1补18项，B2/B3/B4/M/C各补15项。selected18内只有B1的三项引用主final；其余15项各自对应实际A run，不能引用同方法λ=.05或主best替代。A所有七λ/六方法/三seed完整覆盖，A所选λ的冻结来源仍为`9548085f…fa4e477`，没有改变。

B固定生成池11、训练seed11，K20/196各30个final，10个selection seeds为1001–1010。全部60项均列入，不依据已见测试表现排除任何方法、K或draw。A/B新增153项的config与summary小文件均fresh SHA匹配各自冻结NLL manifest；checkpoint SHA只引用原冻结身份，本次没有重读大文件，实际导出仍须fresh验证。

C scope只取冻结active registry和合同。410M/1B各B0/B1/B2/B3/B4/D/M/C×三seed；2.8B仅B0/B1/B2/B4/M×三seed，没有B3/D/方法C。每项config SHA、run/output namespace及model revision均匹配；summary路径由冻结registry明确绑定。**C summary完全未打开**，所以实际best step/exposure/selection、完成状态与checkpoint SHA由出口完整phase门禁解析、验证，scope中保持明确pending，不把路径存在当作可交付。best是各run轨迹内target-validation选点，与A每方法跨λ选final政策不同。

本次读取时410M/1B的独立等价completion均尚未发布；C best的D/B4别名当前不预扣，best与final可能相同也不推断。实际导出只能用相应模型/seed/端点、双方checkpoint SHA绑定的完整state证明决定weight载荷引用；每个逻辑端点的独立metadata及真实曝光仍保留，评价队列不变。

最终覆盖并集为 **366逻辑端点**＝主72＋A额外108＋B60＋C final63＋C best63。已验收主包和A selected共81份载荷；其余不加C别名时名义载荷数上限为360，若C两模型的final/best合计12对D/B4都证明相等则为348。**348是有条件的载荷计数，不是348个已证明不同的模型**；所有模型指纹是否互异要等实际导出及全state核验，不预排除跨端点相同权重。

最终验收仍需完整source/训练完成门禁、fresh checkpoint SHA与前后五stat、全state指纹与往返逐字节一致、持久base/tokenizer/config/codec依赖闭包和不可覆盖发布收据。model-only权重不含optimizer，不能据此续接原Adam或宣称完整训练状态恢复。
