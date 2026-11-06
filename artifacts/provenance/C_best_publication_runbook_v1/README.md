# C best 无损权重包：发布前小元数据核对

此目录只提供CPU核对入口，不启动导出、占用编码锁或发布文件，也不授予ROOT权限。首版尚未对真实C best完整包运行；缺少完整包或whole batch收据时不能出具通过结果。

| package | 模型／完整best逻辑端点 | 原extra reserve inode／实际容量 | whole batch目录 |
|---|---|---|---|
| `C410_best` | Pythia-410m／24 | 467366543／30 GiB | `additional_endpoint_actual_v4/C410_best_batch` |
| `C1B_best` | Pythia-1b／24 | 375195308／75 GiB | `additional_endpoint_actual_v4/C1B_best_batch` |
| `C2p8B_best` | Pythia-2.8b／15 | 331039199／140 GiB | `additional_endpoint_actual_v4/C2p8B_best_batch` |

三个reserve都在`/data/202610/7kb/research_delivery/`，分别名为`.extra_C_410m_best_weights_reserve_20261106`、`.extra_C_1b_best_weights_reserve_20261106`、`.extra_C_2p8b_best_weights_reserve_20261106`。与原C final的70／140 GiB reserve不是同一inode。核对器要求dev=106、uid/gid=11471、0600、单链接、原始实际分配容量与ROOT全容量收据一致，且目标不存在；观察锁空闲不等于持有独占租约。

所有包在`/tmp/grad-research-20261106/delivery/additional_formal_endpoints_v4/<package>/`，完整文件名为`<package>_weights_lossless_v1.tar`。24端点包的独立ZIP数由实际best三对D/B4证明决定，为21–24；false必须单独编码D。2.8B没有D，必须15份独立ZIP，不能添加未训练条件。

## ROOT实际核对前的门槛

1. 对应`completion.json`与`artifacts/provenance/<whole batch目录>/whole_child_timing.json`均已完成，whole `returncode`严格为整数0，两个stop键均不存在。
2. 全模型READY及源audit已发布，410M／1B有源绑定的三对best完整指纹证明；每对决定必须分别绑定B4和D的原checkpoint SHA及完整模型指纹。不能用phase完成状态推定相等。
3. 该模型已有ROOT单独签发的实际资源批许可。A/B许可和C首例许可均不能替代C批许可。核对器要求显式传入许可路径与SHA，并绑定batch实际argv、child launch及其spec。
4. 独立CPU whole计时、stdout/stderr写入新的review namespace；核对器输出路径必须不存在。不得复用旧通过结果覆盖本次失败。

已收到C410实际批许可：`artifacts/provenance/additional_endpoint_actual_v4/ROOT_C410_best_batch_resource_review.json`，SHA `74f6ac6089e0bfacf49a8f4fed064d09a0eed79660dd635a63d1a494ff84b8f1`。这不表示整包已完成，也不授权此工具现在执行；C1B／2.8B的许可需未来分别提供。

## C410完成后可用的核对命令

ROOT审阅本目录源码与准备收据、确认整包门槛后，可用既有`run_one.py`外层Popen/resource计时方式运行下面的唯一核对命令。不使用本平台不存在的`/usr/bin/time`。

```bash
cd /data/202610/7kb/research
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=src:scripts \
  python artifacts/provenance/C_best_publication_runbook_v1/review_metadata.py \
  --package C410_best \
  --resource-review artifacts/provenance/additional_endpoint_actual_v4/ROOT_C410_best_batch_resource_review.json \
  --resource-review-sha256 74f6ac6089e0bfacf49a8f4fed064d09a0eed79660dd635a63d1a494ff84b8f1 \
  --output results/C410_best_publication_metadata_acceptance_v1.json
```

其他模型分别换package、该模型真实ROOT批许可和全新输出路径，不能只改模型名而沿用C410许可。未来whole batch及spec目录按上表命名；如果实际运行方式不同，先由ROOT审阅新的核对入口，不能跳过检查。

## 覆盖与读取边界

核对器复用冻结additional exporter的`validate_completion`、`published`、`validate_bound`、`source_definition`、`validate_resume`、`best_pair_decisions`及资源许可纯验证函数，并复用已审native-final v2的`validate_roundtrip_report`。不调用`prepare`、`execute`、`load_scope`、`load_contexts`或会重新审计／写文件的`complete_phase_gate`。

核对内容包括v4 scope和源代码SHA、冻结registry/snapshot/protocol/queue、READY身份、逐行config/summary对READY与原snapshot的fresh小文件SHA绑定、actual argv及严格rc0/no-stop、best真实曝光和完整训练曝光、原CP的记录SHA与当前五项stat、独立encode/decode/原uint8往返报告的CP/ZIP/base SHA和完整指纹串联，以及D别名的双源证明和自己的metadata sidecar。best选择值只用于校验已冻结端点，不输出评分或重新选择。合法向master添加评价任务不改变已完成训练定义，核对器不把mutable master整体SHA当长期冻结条件。

每包都必须自带本模型六个base/config/tokenizer/README成员及归档工具，不依赖primary或未来C final发布。依据冻结assembler逐项推导metadata、独立ZIP、六base、七固定工具／manifest成员和全部gate证明成员；重复gate来源按原assembler的不同编号保留。member路径必须安全唯一，数量等于原完整TAR验收记录。

实际只打开有大小上限的小JSON／源码／自己metadata目录内的opaque `.source_metadata.pt` sidecar；后者最多8 MiB，只校验SHA，不反序列化。原checkpoint、ZIP、TAR、native大模型和token tensors只stat，不读载荷、不重新计算大文件SHA。原编码器已完成的全成员／整TAR SHA验收是历史证据，本工具不会把元数据核对称作再次逐字节验证。任何source或身份不符都拒绝且不输出通过文件。

## 通过之后仍由ROOT发布

通过结果只提供`source/expected_sha256/expected_bytes/reserved/destination`及证据清单，`ROOT_authorized=false`。ROOT独立审阅后沿用已验收的`scripts/publish_reserved_archive.py`，显式指定独立`--receipt`与`--progress`，实施fresh全源SHA、原inode写入/fsync、完整有效前缀SHA、truncate与不覆盖原子发布；此处不启动publisher或消费reserve。

成本按独立review whole child计一次，其内部process/wall只作嵌套解释。原首例独立child、后续batch whole child、assembly-only子计时、未来publisher whole child的父子关系各自保留，不将endpoint／assembly子项再加进batch。所有尚不存在的实际收据都保持pending。

合成验证命令：`python -m pytest --import-mode=importlib -q artifacts/provenance/C_best_publication_runbook_v1/test_review_metadata.py artifacts/provenance/C_native_final_publication_runbook_v2/test_review_metadata.py tests/test_additional_endpoint_export.py`。`--import-mode=importlib`用于隔离两个既有同名test模块；旧默认模式的collection失败日志保留在`preparation_tests/`，通过结果及独立计时在`preparation_tests_importlib/`。
