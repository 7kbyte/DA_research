# DA_research 关键模型包（无损权重归档）

生成时间：2026-11-07T02:56:30.111819+00:00
对应代码/报告仓库：https://github.com/7kbyte/DA_research

本目录包含研究交付的 **10 个无损权重 TAR 包**，合计 510,504,622,080 字节（475.444 GiB）。
每个 TAR 本身已是**压缩归档**（uint32 位级无损编码 + 逐 tensor 往返核验），无需再次压缩；再次 gzip 不会减小体积，只会浪费时间与磁盘。

- 这些文件与主目录中的 `*.tar` 是**同一 inode 的硬链接**（不额外占用空间），删除本目录不会影响持久交付。
- SHA-256 来自各自已验证的持久发布收据（见 MANIFEST.json 的 publication_receipt），本次未重新全量读取 510 GB。
- 恢复方法、包内成员与基座依赖见仓库：`docs/交付文件与恢复容量指南.md`、`docs/无损权重交付格式.md`；每个 TAR 旁有 `.receipt.json`（部分早期包除外）。

## 内容与逻辑端点

| 包 | GiB | 内容 |
|---|---:|---|
| primary_final | 23.236 | 主实验24项final（D/B4别名） |
| primary_best | 35.371 | 主实验36项验证best |
| budget_final | 13.645 | b=.01/.10 12项final |
| supplement_A_selected | 15.785 | 补充A验证选λ 18项final |
| A_remaining | 99.305 | 补充A其余93项final |
| B_all | 64.800 | 补充B 60项final |
| extended_C_410m_final | 24.166 | C-410M 24项final |
| C410_best | 24.671 | C-410M 24项best |
| extended_C_1b_final | 62.350 | C-1B 24项final |
| extended_C_2p8b_final | 112.115 | C-2.8B 15项final |

另：C-1B/C-2.8B 的 best 包已在临时盘完成编码与自检但**未发布**，不在本目录。
