# 归档复现QA操作预算修订v2建议（尚未授权）

可以使用**原归档scheduler的现成参数**，不改原plan、归档helper、科学源码、四份训练配置或四项job definition。原plan SHA继续为`f9a4077153abf1487fcd28d44f73b05be4403b9d19b45ec89b3cacafd3a98062`，四项expected_seconds仍各600。v2仅建议将scheduler的launch/stop margin同时从1800/600改为120/120秒，deadline仍为2026-11-07 02:56:32 UTC。root的实际执行授权尚不存在；[操作修订模板](reproduction_qa_operational_amendment_v2_TEMPLATE.json)中`root_authorized=false`。

源码核对均针对`/tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/research/scripts/schedule_jobs.py`，SHA `c5c264b058ff655315ef896ccb0a89b471637c6612013a51999fc6f48b2d16d9`：

| 归档源码行 | 实际行为及v2含义 |
|---|---|
| 83—86 | 原CLI接受`--launch-margin-seconds`、`--stop-margin-seconds`；terminate grace默认90秒，poll默认10秒，二者保持原值。 |
| 89—90、172 | scheduler及child的cwd均为归档research；child使用原queue argv、env，源代码不会跳到live工作树。 |
| 94—106 | 同一state有文件锁，拒绝继承running记录；这不能代替检查其他scheduler/state和GPU进程。v2仍要求QA state及四新输出不存在。 |
| 125—126 | scheduler保持job definition SHA一致；改变scheduler margin不改变四项定义SHA。 |
| 159—160 | 准入为`now+expected_seconds < deadline−launch_margin`；120方案的循环时刻界限为**02:44:32之前**。`now`取自循环开头，实际Popen会稍晚，不能把最后毫秒当可用余量。 |
| 128、132—135 | 02:54:32达到stop阈值后，对仍running的child进程组发送SIGTERM；停止新开job。只缩launch而保留原600秒stop会在02:46:32提前终止晚启动训练，因此两项都需覆盖。 |
| 136—137、173 | child以`start_new_session=True`启动，90秒grace后对其PGID发送SIGKILL。若SIGTERM恰在02:54:32发送，名义SIGKILL阈值为02:56:02。poll延迟和CPU/I/O调度会推迟实际动作。 |
| 141—143 | 一旦发送过stop，任务即使随后rc0也标为failed；不能把停止后留下的summary/checkpoint当完整重训成功。 |
| 182—187 | 仅在active清空后退出，`--exit-when-idle`也会在全部job非pending且无active时退出。代码没有“deadline到点无条件杀完并立即退出”的硬实时保证。必须另查scheduler PID、全部children/PGroups退出及四GPU空闲，不能只看state.active_jobs为空。 |

归档train的482—486行捕获SIGTERM/SIGINT，573—575行在下一步边界记录interrupted；677—679行随后仍可能执行末次验证和保存部分checkpoint。模型自己的02:56:32−90秒deadline检查也只在步边界执行。故SIGTERM不是立即成功结束，90秒后仍未退出可被scheduler强杀，partial checkpoint不具备复现资格。验收仍须完整306步、10M/500K实际prediction tokens、rc0、无stop_requested_at及全部模型字节指纹一致。

## 时间边界与完整权重验收

**02:44:32是调度准入上限，不是包含后验的保守最晚开跑保证。** 从最晚界限起算，若四项实际各用满600秒，只剩120秒给完整CPU指纹、checkpoint文件SHA、退出核验和receipt；四份新checkpoint的精确大小及全验收时长尚未实测。本审阅不把它们当0，也不虚报冷缓存吞吐。

四个原始对应run的scheduler elapsed为343.48/343.48/350.53/350.35秒，最慢350.53秒仅是历史参考。四个原checkpoint约19.46GB；复用原`model_fingerprint`加完整checkpoint文件SHA至少需要读取约38.92GB序列化checkpoint字节，并在内存遍历全部model_state。CPU反序列化、验证轨迹和元数据处理另有开销。重训后必须实际绑定新文件SHA和完整405334016元素的key/dtype/shape/全部字节指纹，而不是比较少量tensor或仅看原summary。

建议暂为四项CPU后验预留**300秒操作预算**，明确它是规划余量而非测量结果：600秒训练+300秒后验的纯算术界限为02:41:32，再考虑poll、启动、最终source检查和记录周转，建议实际启动目标不晚于**02:40 UTC**。如果root希望接近02:44启动，须在当时另以可靠完整后验计时和实测余量明确授权，不能凭历史350.53秒保证满600秒场景仍保留完整30分钟交付尾部。

| 假设启动时刻 | 四QA运行假设 | 300秒CPU后验后结束 | 相对02:56:32 QA截止 |
|---|---|---|---|
| 02:24（假设主评价02:22完，再留2分钟检查） | 保守600秒 | 02:39 | 余17分32秒，不含其他未测工作 |
| 02:40 | 保守600秒 | 02:55 | 余1分32秒供周转，仍需root检查实际余量 |
| 02:44 | 历史最慢350.53秒，仅供参考 | 02:54:50.53 | 约余1分41秒；不能保证重训同速 |
| 02:44 | 保守600秒 | 02:59 | 超出QA截止2分28秒，不能声称保留整30分钟最终尾部 |

所有完整QA验收应在02:56:32之前完成，之后至03:26:32保留给最终报告、持久化与交付。若无法同时满足，按现有协议保留“完整重训未验证”即可；不改科学条件，不重置或伪造QA成功。

## root未来可审查命令（本次未执行）

root须先确认全部既定科学训练与评价完成、原master及任何评价scheduler和全部children/PGroups已退出、四GPU compute进程列表为空。不要通过修改科学队列来腾位置。随后复核[独立审阅](归档复现QA独立审阅_20261106.md)中的helper/scheduler/algorithm/包初始化/plan/权重manifest小文件SHA与软件版本；所有相邻JSON也须纳入34项audit。root应将本false模板另存为明确授权receipt，绑定原planSHA、此v2模板SHA与exact argv；归档scheduler自身不解析该授权文件，不能把模板当机器执行门禁。

先执行原helper **dry-run**（不加`--execute`），再运行现有34项audit，输出到新的独立收据文件；任何不匹配均停止：

```bash
cd /tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/research

/usr/bin/python scripts/reproduce_experiments.py reproduce \
  --registry configs/reproductions/archive0732_qa_2x2_seed11_v1_registry.json \
  --gpus 0,1,2,3 --deadline 2026-11-07T02:56:32Z

CUDA_VISIBLE_DEVICES='' /usr/bin/python /data/202610/7kb/research/scripts/audit_reproduction_qa.py \
  --source-workspace "$PWD" \
  --registry configs/reproductions/archive0732_qa_2x2_seed11_v1_registry.json \
  --weight-manifest ../primary_final_weights_manifest.json \
  --output /data/202610/7kb/research/results/reproduction_qa/archive0732_seed11_2x2_v1/prelaunch_reaudit_MARGIN_v2_ROOT.json
```

在root明确授权和全部最终门禁通过后，使用**同一归档scheduler**作为唯一新启动入口。以下exact argv仅在原plan的scheduler argv末尾增加四个margin参数；不再运行helper的`--execute`，避免再起一套scheduler。可由root在独立session中管理该进程及其log，无需新工作流或修改归档脚本：

```bash
/usr/bin/python /tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/research/scripts/schedule_jobs.py \
  --queue /tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/research/configs/reproductions/archive0732_qa_2x2_seed11_v1_queue.json \
  --state /tmp/grad-research-20261106/reproduction_qa/archive0732_seed11_2x2_v1/research/logs/reproductions/archive0732_qa_2x2_seed11_v1_job_state.json \
  --gpus 0,1,2,3 --deadline 2026-11-07T02:56:32Z --exit-when-idle \
  --launch-margin-seconds 120 --stop-margin-seconds 120
```

原四job的CUDA_VISIBLE_DEVICES隔离、内部cuda:0、OMP/MKL8、cuBLAS`:4096:8`、精度与所有科学配置保持不变。root不可仅杀scheduler leader后假设GPU已释放；应等待其收掉children并确认全部相关PGroups退出。完整权重后验可复用上一独立审阅给出的原函数CPU命令，或root另行冻结的独立验收器；本v2不添加新的训练/评分/后验源码。

成本仍按四次真实工程QA训练及其验证/checkpoint、CPU完整验收单独记录为diagnostic；即便与原权重完全相等，也不把这四次物理运行消成零成本alias或增加科学seed。
