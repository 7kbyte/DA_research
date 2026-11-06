# 归档复现QA：final+best独立CPU后验（仅工程准备）

当前没有真实QA重训结果，**未调用真实post-run verify、未产出passed收据**。原qa_plan保持SHA `f9a4077153abf1487fcd28d44f73b05be4403b9d19b45ec89b3cacafd3a98062`，旧audit、归档helper、core、配置与queue均未修改；margin v2授权模板仍为false。训练实际执行仍由root在科学训练/评价全部完成后按预算另行明确授权。

新入口为[scripts/verify_reproduction_qa_postrun.py](../scripts/verify_reproduction_qa_postrun.py)。它只做CPU读取和验收，不启动训练、scheduler、forward或CUDA，要求显式`CUDA_VISIBLE_DEVICES=''`。prepare-expected已运行，仅读取原计划、已验收manifest、JSON及四个约5KB的best元数据文件，未读取真实新/旧完整checkpoint。

## 独立expected与来源

实际采用[postrun_expected_final_best_v2.json](../results/reproduction_qa/archive0732_seed11_2x2_v1/postrun_expected_final_best_v2.json)，SHA：

`93da7a152e4f88968a01a8a569b1d057511474e6ccf1f04a4c336dcaafe891ce`

它是额外预期文件，不改原plan。开发中初版expected v1原字节保留、尚未执行；**v1不作为最终验收入口**，v2额外绑定独立scratch relocation并检查父目录没有被重定向。

final的完整指纹来自原plan，并与其已验收final manifest逐项交叉核对；final step/曝光从已验收original_checkpoint_metadata和summary_counters共同取得，代码没有硬编码306。best来自已完整持久化验收的`primary_best_weights_manifest.json`（SHA `fbd5d91c967d46e7b55d1aca62fd1c4a9a71c6e646ebe697bc0b730199698e4f`）及manifest所绑定的小型source metadata，每个文件SHA fresh核验，绑定真实kind、selection、step/target/replay曝光和全model_state指纹；durable acceptance/receipt也绑定。没有重新解压或全读37.98GB权重TAR，不能把这次准备称为新的整TAR验收。

|原run组（均seed11、b=.05、λ=.05）|final step / target / replay|best step / target / replay|
|---|---|---|
|B1|306 / 10,000,000 / 500,000|100 / 3,270,400 / 163,520|
|B2|306 / 10,000,000 / 500,000|50 / 1,635,200 / 81,760|
|B3|306 / 10,000,000 / 500,000|100 / 3,270,400 / 163,520|
|B4|306 / 10,000,000 / 500,000|100 / 3,270,400 / 163,520|

四个best的`checkpoint_kind`均须为`model_only_best_target_validation`。selection的step、target token、target-validation NLL均按原始精确记录核对，只有checkpoint_path映射至新namespace。final全训练成本与best实际端点曝光分开保留，不能把best写成10M端点。

expected同时绑定32项小型source proof：原plan、registry/queue、final/best manifests与best验收receipt、四份best metadata、原/新配置、scientific identity audit/relocation、10项归档core及归档指纹算法。verify开始和结束均fresh核验这些来源；不采用旧checkpoint哈希缓存。

## 全量门禁与完整权重核验

先对**全部四run和八端点完成结构核验**，任何缺失/partial/错误均在读取新checkpoint之前失败：

- state的job集合恰为新四run，active为空，全部completed、rc0、missing_outputs为空、无stop_requested_at；definition SHA及实际argv与冻结queue逐项相同。
- scheduler PID及全部记录child进程组已退出；四run的GPU分配与冻结四卡集合一致。这不替代启动前“旧科学scheduler/所有children退出、四GPU空闲”的人工门禁，也不声称检测了系统中全部无关GPU进程。
- exact config SHA、新namespace/run_label、resume_from=None、独立scratch/runs路径、internal/copy summary相等及summary.config等于原新config；counter须为完整final实际step/target/replay。
- summary及checkpoint的实际Torch/CUDA/GPU、FP32 teacher/student、BF16 precision、严格deterministic/warn_only=False、四SDPA backend、TF32、cuDNN及cuBLAS全部与原provenance逐项相同，不用当前环境值补缺失证据；实际model_state tensor dtype也检查。
- final/best须在各新output/checkpoints的明确不同路径，有各自新inode；拒绝软链接、与原文件同inode、跨端点共用inode，以及父路径逃出独立scratch。每个新checkpoint的mtime须落在本次实际job起止时间内。
- summary的best selection、两端点payload的kind/实际曝光/selection均须匹配expected；final仍为完整训练checkpoint、best须为model-only。original文件直接复制仍会因旧config/selection/文件SHA而失败，不能拿旧summary或权重文件当新训练成功。

每个端点随后依次执行三次独立读取：

1. `torch.load(..., map_location='cpu')`核对新checkpoint元数据及所有tensor dtype；这是完整反序列化，包含未用于比较的optimizer字节也如实计读取成本。
2. 调用SHA固定的**归档原函数**`model_fingerprint`（算法文件SHA `b2997465717ddd46b5f80924a022d6a82bafe65f5d348636852fdc10836a2362`），遍历排序后的全部key/dtype/shape和每个tensor全部contiguous字节。传入公开支持的计数file-like输入；不monkeypatch、不改算法、不采样tensor。
3. 重新fresh读完整新checkpoint，计算整文件SHA。不得拿原SHA、prepare缓存或model-only指纹冒充这个新整文件SHA。

开始前和三次读取后核对dev/inode/size/mtime_ns/ctime_ns五stat；全部八端点处理完还再次核对所有checkpoint五stat、config/summary、scheduler state和绑定来源，以防较早端点在后续核验中变化。任何失败整体status保持failed；只有八个完整指纹全部匹配才写`passed_all_final_and_best_full_model_state_bytes`。

结论只限于**八个端点的完整model_state字节指纹一致**及其来源/曝光/数值policy合格；没有比较全部训练/验证轨迹、optimizer、RNG或stream state。虽然metadata读取会反序列化optimizer，这不意味着对optimizer做了一致性验证。

## root未来执行命令（当前未执行）

仅在root已实际授权并完成完整QA训练、scheduler/children退出且剩余时间足够后运行。不要在尚无新run时调用来“试出passed”。expected、源码SHA应在root执行receipt中绑定：verifier `5417fc0747d94feae828e0e53133edf66745b2fb60885cc591f84f2feceea7e2`。

```bash
cd /data/202610/7kb/research
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /usr/bin/python scripts/verify_reproduction_qa_postrun.py verify \
  --expected results/reproduction_qa/archive0732_seed11_2x2_v1/postrun_expected_final_best_v2.json \
  --expected-sha256 93da7a152e4f88968a01a8a569b1d057511474e6ccf1f04a4c336dcaafe891ce \
  --output results/reproduction_qa/archive0732_seed11_2x2_v1/postrun_final_best_ROOT_attempt1.json
```

目标receipt以`open('x')`独占预留。成功或失败都不覆盖；失败记录error、已完成部分证明、实际CPU/wall和已读取字节，CLI非零退出。重试必须保留旧收据，明确新attempt路径及原因，不能重置旧结果。真实root命令不会读取封存的B/C/behavior分数。

## 实测成本字段与300秒边界

真实receipt的`wall_seconds`及`process_cpu_seconds`为该CPU验收父计时；`checkpoint_read_operations[]`记录24项读取的wall/CPU与实际serialized_bytes_read，均已含于父计时，不再与父计时相加。

`serialized_checkpoint_read_bytes_total`累计read/readinto/readline实际返回字节，重复读取如实重复累计；`fresh_full_checkpoint_SHA_bytes`仅统计八次完整新文件SHA字节，应等于八文件大小之和。它们不是物理磁盘冷读计数，更不是GPU时间；OS缓存命中仍是这次实际CPU读取，不能说0成本。

按原四final及四best文件大小，八文件合计约25.94GB；本实现三次读取对应约77.83GB的文件规模，具体zip随机读及重复头部字节以真实计数为准。**现在尚无真实QA checkpoint验收时长，300秒仍仅为规划余量，绝不改称已测时间**。新8端点范围和额外读取量应由root纳入最后预算；不能因旧final-only估计而默认足够，不压缩完整30分钟交付尾部。

## 临时CPU测试

[tests/test_reproduction_qa_postrun.py](../tests/test_reproduction_qa_postrun.py)的24项临时小checkpoint测试通过。使用7步小案例证明代码不硬编码306，并覆盖八端点成功、无新run失败、partial/rc/stop/definition/活动进程门禁、final和best最后tensor字节变化、kind/selection/曝光/config/policy缺失或不符、旧文件软硬链接、重复inode、缺best、源码变化、读取期间mtime变化、immutable收据、expected SHA及CPU-only门禁。所有测试数据位于pytest临时目录，不触碰真实运行。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 pytest -q tests/test_reproduction_qa_postrun.py
```
