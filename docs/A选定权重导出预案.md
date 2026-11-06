# A选定final薄adapter：准备与审阅

入口 [export_A_selected_weights.py](../scripts/export_A_selected_weights.py) 仅保存已冻结的18个A验证选λ final：15个新权重，3个B1引用已发布主包。kind为 `validation_selected_lambda_final`，各逻辑条件保留A condition ID、actual run、phase/model/data/seed/λ及完整原checkpoint非weight/nonoptimizer metadata。它不改选择，不输出全部126权重；与主λ=.05并集为33/126。B60、C及A best不在此入口范围。

准备检查已完成，无真实encode。完整126/六方法/三seed来源与执行policy通过；选择SHA `9548085fc18278f69fad584b67485bceb5193a18b10c78f516f6b09b4fa4e477`、A身份manifest SHA `88dc927c9b4909cca66901608216290e1e435022e4ff925c00875f5099077a90` 固定。每个实际encode得到的完整CP SHA必须与该manifest匹配，不再预读15×4.8GB做重复哈希。saved provenance、config、final step、10M/.5M曝光及trajectory均与冻结summary显式核对；train/replay/validation文件按主冻结输入snapshot核验。

选择时的 `protocol_status.json` 精确原字节从08:38持久包恢复，SHA为 `fb93296e67a4302ea33f2daaad7adfecd8b48d2cb939a18d1bab001fa0fb443c`，见 [原字节](../artifacts/provenance/A_weight_export/selection_time_protocol_status.json)及[diff](../artifacts/provenance/A_weight_export/protocol_status_diff.patch)。仅六个明确管理字段改变；adapter只对白名单允许语义变化，其余字段变化拒绝。选择JSON未更改。历史管理账本不作为长期mutable全文件科学门禁；registry/config/core/summary/输入/完整CP SHA仍严格。

410M base、config、tokenizer与3B1引用已有主持久TAR `primary_final_weights_lossless_v1.tar`，SHA `2bb090fbe7df172b31102068715143ab702087ff49f77ef7dfd5db3637c928b1`。本次依赖历史整TAR验收及当前publication receipt的dev/inode/owner/权限/size/mtime/ctime，小manifest另匹配已验收SHA `4120d177c22200b0bc5ee8390e0c2e2a1f5be1acbc58ef0f6d209bd47f96e8fb`；没有重新整读25GB TAR。闭包保留准确member、ZIP SHA、base文件SHA和FP32逐tensor SHA，离线提取和恢复说明会进入增量TAR。

CPU临时测试13项，加codec6项，共19项通过；含18/15/3映射、禁用best/test选择、源CP SHA错误拒绝、真实微型NaN/符号零往返、全部原metadata保留、3B1不重新编码、实际曝光拒绝、历史管理diff与持久TAR stat变化拒绝。日志 [A_selected_weight_export_tests.log](../logs/A_selected_weight_export_tests.log)。真实prepare-only来源证据在scratch `delivery/supplement_A_selected_final_v1/preparation_review.json`。

root审阅后可启动（本轮未执行）：

```bash
nohup env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python scripts/export_A_selected_weights.py \
  > logs/A_selected_weights_export.log 2>&1 < /dev/null &
```

实际导出使用独占scratch `.supplemental_weight_export.lock`，后续C adapter沿用同锁，单worker/2CPU线程。输出只在 `/tmp/grad-research-20261106/delivery/supplement_A_selected_final_v1/`；原CP、ZIP和恢复验证副本均保留，逐项encode→decode→原uint8及metadata全比对。首份record记录实测ZIP/raw、CPU、wall和进程累计peakRSS。整体completion计时包括prepare与TAR；record为嵌套子项，若复用已验证旧record，其timer明确为历史成本。

最终TAR大小须不超过20GiB，完整15新增+3引用后才写completion，独立TAR每member SHA复验并fsync。预留路径为 `.supplement_A_selected_final_weights_reserve_20261106`，adapter不创建、扩容或写入它，不自动发布；实际TAR验收后仍由root使用现publisher。恢复为model-only，不含Adam优化器，不能声称原run精确resume。
