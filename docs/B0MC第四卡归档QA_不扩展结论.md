# B0/M/C第四卡归档QA：不扩展

本次仅核查小配置、清单结构及文件存在性：0732提取树中主模型seed11的B0/M/C配置均存在；共同目标输入存在，M的`generated_nll_stratified_seed11.pt`与C的`generated_cluster_seed11.pt`也存在，B0没有回放输入。未读取模型、checkpoint、base、TAR/ZIP或token载荷，未运行QA或测试；没有完成这三项final/best逐端点身份验收。

现有`verify_reproduction_qa_postrun.py:312–313`要求每个run分配不同GPU；因此B0/M/C三任务串行共用GPU3不能原样通过该验收器。另`schedule_jobs.py`目前没有`--log-dir`参数，日志固定写入state父目录下的`jobs/`与`scheduler_events.jsonl`，必须使用独立state父目录才能隔离。ROOT据此决定不增加三份单run编排、不修改验收器，不制作FALSE方案、expected或新执行流程。既有main4与条件Cthree范围保持不变，优先完成原定全范围；本项未执行，不增加科学重复或实际训练费用。
