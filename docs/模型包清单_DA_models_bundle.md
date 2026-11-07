# 模型包清单（DA_models_bundle）

GitHub 仓库：https://github.com/7kbyte/DA_research

持久目录：`/data/202610/7kb/research_delivery/DA_models_bundle`（与外层 TAR 为硬链接，不额外占空间）

共 10 个无损权重 TAR，合计 510,504,622,080 字节（475.444 GiB）。

| 文件 | GiB | SHA-256 | 发布收据 |
|---|---:|---|---|
| primary_final_weights_lossless_v1.tar | 23.236 | `2bb090fbe7df172b31102068715143ab702087ff49f77ef7dfd5db3637c928b1` | `docs/primary_weights_durable_publication.json` |
| primary_best_weights_lossless_v1.tar | 35.371 | `02e104573c0601f92100ad7a32ffb4d0c2db1860250e921a86909657830df0f5` | `docs/primary_best_weights_durable_publication.json` |
| budget_final_weights_lossless_v1.tar | 13.645 | `539c2bc6c51e009464f5d5b16d9c97c4f7c5c5ed7db20b06dbd23b38a6aca3df` | `docs/budget_final_weights_durable_publication.json` |
| supplement_A_selected_final_weights_lossless_v1.tar | 15.785 | `0dfec20b1bf9dde38b0d0d453c821eb088490bde188dc0e19bc498cc37b04631` | `docs/A_selected_weights_durable_publication.json` |
| A_remaining_final_weights_lossless_v1.tar | 99.305 | `9e818d15cb826f0948f7092188e684382fea0d03605deac765e1ac19e5ea08b1` | `docs/A_remaining_final_weights_durable_publication.json` |
| B_all_final_weights_lossless_v1.tar | 64.800 | `b8f7cb88141ad1b6526f05e92fe34abd68a5f5a13b8219b5d4957ff3035d50d1` | `docs/B_all_final_weights_durable_publication.json` |
| extended_C_410m_final_weights_lossless_v1.tar | 24.166 | `121384161d379b518a51d2c3baf38dedab205f41ef7bd11ded2d56ea7b95e1f6` | `docs/C410_final_weights_durable_publication.json` |
| C410_best_weights_lossless_v1.tar | 24.671 | `0e05b8b3d7f8088220b6a1cc9b8daf9d44b30eb758014d8732390361a22424a0` | `docs/C410_best_weights_durable_publication.json` |
| extended_C_1b_final_weights_lossless_v1.tar | 62.350 | `c47360c0a50d2a5a024a4dc417843aad37269567a5f7f2a602e051d127e39ac8` | `docs/C1B_final_weights_durable_publication.json` |
| extended_C_2p8b_final_weights_lossless_v1.tar | 112.115 | `7ca63d1fd4aab0ae491df8577ca2f16483d1e92445e70feaff46a342c837d313` | `docs/C2p8B_final_weights_durable_publication.json` |

说明：TAR 本身已是压缩无损归档，SHA 来自各自验证过的发布收据，本次未重读 510 GB。
C-1B/C-2.8B 的 best 包已在临时盘完成编码与自检但未发布，不在清单内。
