# B native持久包：固定首例恢复准备（尚未执行）

固定原始首例`bankvar_B4_tr11_gen11_draw1001_k20_b0p05_w0p05`，训练／生成seed11、selection draw1001、K20、final10M target／0.5M replay。选择沿用旧B首个导出样例，不参考模型分数。

唯一权重来源为已发布`/data/202610/7kb/research_delivery/B_all_final_weights_lossless_v1.tar`（69,578,864,640 B，原publisher整包SHA `b8f7cb88141ad1b6526f05e92fe34abd68a5f5a13b8219b5d4957ff3035d50d1`）。本次准备只读取小manifest／publication／ROOT acceptance／scope及源码，另stat当前TAR，未读TAR header、ZIP、native base、原CP，也没有运行恢复。当前dev106／inode411787337／11471:11471／0600及五项stat与publisher、ROOT验收一致。

新QA目录：`/tmp/grad-research-20261106/reproduction_qa/B_native_B4_tr11_gen11_draw1001_k20_final_v1`。新whole计时目录：`artifacts/provenance/B_native_restore_actual_v1/`；两目录现在均不存在，未来只允许不存在时创建，失败及原文件全部保留。

复用既有安全选成员函数`extract_selected`、cgroup-aware `resource_gate`、独立`check_restored.py`及修复后的`run_group`（含subreaper、整PG TERM/KILL与回收）。原C bind／runner／wrapper／checker均不修改，B薄runner仅处理本包身份、ROOT许可和调用顺序。独立checker直接支持plan中的B身份；其固定输出文件名`restored/B4_seed11_final.pt`中的11指training seed，不是draw1001。

## 12个待提取成员

下表由完整B manifest、v4 frozen scope和已固定归档工具SHA推导。base大小来自原frozen stat，metadata sidecar仅stat取大小。工具SHA来自已验收归档工具pins及冻结assembler的包含规则，不声称本次已打开B TAR验证工具；未来12项均须fresh selected-member SHA通过才继续decode。

| Member | Bytes | Expected SHA-256 |
|---|---:|---|
| `base/pythia-410m-final/model.safetensors` | 911373632 | `3e09001631bb18ec8a6f4522f8e6ab26a53099e69228f9822adc91bfbe27511b` |
| `base/pythia-410m-final/config.json` | 570 | `d4c11e84a59c8af4d88446bba53b718f7aef740daa070ded08fd6a9a3aca4fc6` |
| `base/pythia-410m-final/tokenizer.json` | 2113710 | `c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624` |
| `base/pythia-410m-final/tokenizer_config.json` | 396 | `70e38394e494931c6f773ba41e19460dd4436526b852207367f04341b4066d3f` |
| `base/pythia-410m-final/special_tokens_map.json` | 99 | `6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad` |
| `base/pythia-410m-final/README.md` | 13593 | `d7d60ae94cb0882f19381632532a01d39247fe6f932016c39768b1f67c9f2bf9` |
| `additional_weights_manifest.json` | 561665 | `25ee332b1e42b66b84db4463ca6d3b615a8172d903f3d1309ed80eab71e8462c` |
| `weights/bankvar_B4_tr11_gen11_draw1001_k20_b0p05_w0p05__final.xor.zip` | 1144048546 | `b66a21ebe73bb630b48737c2201d47c96bed8a02f366d09f91cb8adb1fb09b8b` |
| `metadata/bankvar_B4_tr11_gen11_draw1001_k20_b0p05_w0p05__final.source_metadata.pt` | 21068 | `2828e7923cc732eb6258720fa758a1586db4ac29fa61e6ee24879b57f0228b8e` |
| `tools/scripts/archive_model_weights.py` | 2085 | `28c5841cae8e7fd87d8cb62e7f756dad40ce15cb558710c31e48dfb45fde668a` |
| `tools/src/replay_study/weight_archive.py` | 15264 | `3c2bcf9e38773ae4f69c27f85f342356dbefcb53d817be4321da6e9c4e550e1a` |
| `tools/src/replay_study/__init__.py` | 65 | `b2dce8cc5ea80a67a90b1798232163a7684874487b5ea4102e60851c7017f520` |

总选中成员2,058,150,693 B；加1,621,336,064 B恢复模型的最低静态占用约3.68 GB（不含少量日志、序列化开销），准入保守要求8 GiB可用scratch及8 GiB cgroup reclaim-aware内存代理。该代理不等于峰值RSS保证；不能使用host约2TB可用RAM替代容器额度。

预期完整FP：`05724eda110e91eb76c8f6d45b54386b374e486662207a266166a0fde0ffb790`，405,334,016 FP32元素／292 tensors。原CP SHA只作为已发布身份`13dde87cee608d8e677e86fc91e5aaeb734865ee7a2197766dde4e69711bc951`保留，不重新打开原CP。未来checker会重新完整读取新恢复tensor计算FP、检查解码报告／原kind provenance／自己sidecar／实际曝光，并另hash新恢复文件序列化字节。

## ROOT审阅后才可执行

`bound_plan.json`和`ROOT_execution_authority_FALSE_TEMPLATE.json`已生成；FALSE模板不能执行，未制造任何ROOT true。ROOT需另写真实authority并绑定该plan、runner、checker精确SHA。下面命令现在**没有执行**；环境变量值由ROOT未来填入真实许可路径与SHA。

```bash
cd /data/202610/7kb/research
: "${B_RESTORE_AUTHORITY:?Set actual ROOT authority path}"
: "${B_RESTORE_AUTHORITY_SHA:?Set actual ROOT authority SHA}"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -I artifacts/provenance/B_native_restore_preparation_v1/run_B_restore.py execute \
  --plan artifacts/provenance/B_native_restore_preparation_v1/bound_plan.json \
  --plan-sha256 903baac7f57172387bbda7d8f31785d3ac5cd29fe52739b5c856e12d6e74663b \
  --authority "$B_RESTORE_AUTHORITY" --authority-sha256 "$B_RESTORE_AUTHORITY_SHA"
```

只调用`execute`外层入口；`_body`是其内部子进程，不能手动替代whole cap。执行上限600s，另留300s margin；以02:56:32 UTC为保护终点，剩余必须严格大于900s，最晚应在02:41:32之前准入。PG清理最多约12s预留包含在300s中。实际cgroup资源门禁在创建QA和读TAR之前再检查。

父计时从CLI/preflight开始，包含helper检查及spawn到整PG消失；结果在`whole_child_timing.json`。body、提取region、decode与独立FP阶段各有计时，但都嵌套，不再与whole相加。inclusive CPU只取外层self user/sys＋children user/sys一次。原69.58GB TAR不重新整包SHA；沿历史全源／copy／prefix验证和当前stat，实际只读取TAR headers和12个selected members，记录Python返回字节而非物理磁盘流量。

本计划只检验一个固定B model-only恢复闭环，不证明60端点全恢复，不与原CP再次uint8比较，不恢复Adam，不运行GPU、再训练或新科学评分。原TAR／ZIP／CP与未来QA全部保留。
