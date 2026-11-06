#### 6.1.1 验证选λ策略的行为评价

补充A已完成六方法各三个seed的18个固定10M final行为端点，并通过完整独立复核。任务、评分、初始模型与第5.5节相同；每种方法沿用此前由验证NLL冻结的λ，不按行为结果重选。下表报告准确率百分数的三seed均值±样本SD，`acc_norm`为主评分、`acc`为辅助评分。匹配初始值依次为PiQA 66.9206%/66.8118%、HellaSwag 40.3605%/33.8578%。

| 方法 | 验证所选λ | PiQA acc_norm（%） | PiQA acc（%） | HellaSwag acc_norm（%） | HellaSwag acc（%） |
|---|---:|---:|---:|---:|---:|
| B1 | .05 | 67.2651±0.2060 | 67.1745±0.2792 | 42.2758±0.2504 | 35.1656±0.0632 |
| B2 | 1 | 66.8662±0.1962 | 66.4309±0.1439 | 39.9090±0.0720 | 33.6951±0.0402 |
| B3 | .1 | 66.9750±0.3309 | 67.0112±0.3623 | 41.8210±0.1409 | 34.8901±0.1906 |
| B4 | 5 | 66.6123±0.3540 | 66.5579±0.2199 | 39.7530±0.2154 | 33.3234±0.1563 |
| M | 5 | 66.9387±0.2684 | 66.9206±0.3029 | 39.7464±0.1326 | 33.4794±0.2371 |
| C | 1 | 67.4646±0.1885 | 67.4102±0.2879 | 40.8684±0.2251 | 34.4387±0.1478 |

验证NLL选择与行为保留并不等价。HellaSwag的acc_norm相对初始均值变化，B2、B4、M分别为−.4514、−.6074、−.6141个百分点（pp），三个seed方向均为负；B1、B3、C分别为+1.9153、+1.4605、+.5079 pp。B1/B3的两评分在三个seed均提高、逐seed条件题目区间均高于零；C的acc_norm seed33区间仍跨零。PiQA全部18端点的两评分相对初始的条件题目区间均跨零，不能将表中的小幅均值变化解释为已确证的保留增益或损害。这句话只针对端点与初始的比较，不代表所有方法配对区间均跨零。

六个预定策略对照全部列于下表，单位为pp，左减右，±仍为三seed样本SD。λ列保留双方实际验证选择，因此比较对象是“方法加验证选参”的政策。

| 对照 | λ左/λ右 | PiQA acc_norm | PiQA acc | HellaSwag acc_norm | HellaSwag acc |
|---|---:|---:|---:|---:|---:|
| M−B4 | 5/5 | +0.3264±0.1962 | +0.3627±0.4563 | -0.0066±0.2334 | +0.1560±0.3853 |
| C−B4 | 1/5 | +0.8524±0.5284 | +0.8524±0.2199 | +1.1153±0.1828 | +1.1153±0.1630 |
| B2−B1 | 1/0.05 | -0.3990±0.2684 | -0.7436±0.4226 | -2.3667±0.2440 | -1.4705±0.1017 |
| B4−B3 | 5/0.1 | -0.3627±0.6715 | -0.4534±0.5584 | -2.0680±0.0859 | -1.5668±0.1119 |
| B3−B1 | 0.1/0.05 | -0.2902±0.4627 | -0.1632±0.5885 | -0.4548±0.2528 | -0.2755±0.1451 |
| B4−B2 | 5/1 | -0.2539±0.2513 | +0.1269±0.2684 | -0.1560±0.1458 | -0.3718±0.1582 |

M−B4是本批唯一同λ=5的方法配对。其PiQA主评分在三个seed均为正，但条件题目区间均跨零；HellaSwag主评分均值接近零、三个区间跨零，辅助评分还有seed方向变化，仍不支持M在两任务、两评分上统一优于B4。C−B4的两任务、两评分在三个seed均为正，但λ为1对5，不能归因于固定教师约束强度下的选库净效应。B2−B1与B4−B3的HellaSwag两评分在三个seed均为负、条件区间均低于零；由于双方λ不同，也不能解释为固定λ下KL普遍差于CE。来源配对B3−B1、B4−B2同样保留实际λ与评分条件，不据此作跨设置的来源排序。

全部个体端点与条件区间见[acc_norm个体图](results/supplement_A_behavior_presented_v1/endpoints_acc_norm.png)、[acc个体图](results/supplement_A_behavior_presented_v1/endpoints_acc.png)；六策略配对见[acc_norm图](results/supplement_A_behavior_presented_v1/selected_policy_pairs_acc_norm.png)、[acc图](results/supplement_A_behavior_presented_v1/selected_policy_pairs_acc.png)；唯一同λ配对另见[M−B4 acc_norm图](results/supplement_A_behavior_presented_v1/fixed_lambda_pairs_acc_norm.png)、[acc图](results/supplement_A_behavior_presented_v1/fixed_lambda_pairs_acc.png)。[全部18端点可读表](results/supplement_A_behavior_findings_v1/全部18端点可读表.md)及[全部六策略逐seed配对表](results/supplement_A_behavior_findings_v1/全部六策略配对可读表.md)保留准确率变化、CW/WC及条件遗忘/获得率；净准确率变化不代表所有旧正确行为都被保留。

这些结果不倒改上文A的NLL结论或验证选择。三条seed训练路径同时参与了λ选择，样本SD不是三次独立重复调参的波动，也不估计新seed重做整个选择流程的泛化变化。原5,000次题目bootstrap条件于既定检查点；HellaSwag同activity/context的依赖未校正。多方法、任务、评分、seed和配对的逐项95%区间未经多重比较校正，不保证同时覆盖。这里没有新增统计检验、行为调参或新的推理能力主张。[完整行为讨论](results/supplement_A_behavior_findings_v1/讨论.md)、[按评分分别复核的策略统计CSV](results/supplement_A_behavior_independent_review_v1/selected_policy_score_resolved_statistics.csv)。

