# Cora CAM32 节点顺序结果

本文档归档 `Cora` 在 `CACHE_SIZE=32` 下的节点访问顺序对比结果。这里的 `CAM32` 指 Python 全链路前端中的在线缓存容量为 32 个节点；结果来源是 `run_progressive_bfp_fullstack.sh` 产生的全链路日志，不是 `CAM_sim/reports` 下的 512KB C++ 回放报告。

## 1. 实验设定

固定配置如下：

```text
数据集:
    Cora

前端:
    8 个 16 位头
    radius = 2
    score threshold T = 30
    support >= 5   -> 直接复用
    support = 3..4 -> 残差路径
    support < 3    -> 编码器路径

在线 CAM/缓存:
    CACHE_SIZE = 32

后端:
    参考池 = W4BFPA8_B128
    基础路径 = W4BFPA4_B128
    精修路径 = W4BFPA6_B128
    REFINE_RATIO = 0.30
```

对比三种节点访问顺序。默认顺序使用数据集原始节点顺序；哈希顺序使用 `NODE_ORDER_TYPE=hash`，本次配置为 `HASH_NODE_ORDER_HEADS=4`、`HASH_NODE_ORDER_BLOCK_SIZE=0`，分块大小为 0 时脚本会使用 `CACHE_SIZE`，也就是 32；`METIS` 顺序请求使用 `METIS_PARTITION_SIZE=32` 的分区顺序。

重要说明：当前本地 `METIS` 结果没有真正启用分区排序。日志显示缺少 `METIS_DLL`，运行时回退到了默认顺序。因此下面的 `METIS` 行只能说明“请求 METIS 时当前环境会回退到默认顺序”，不能作为有效的 METIS 分区收益结论。

## 2. 三轮主对比

| 节点顺序 | 轮数 | 复用 | 直接 | 残差 | 编码器/P8 | 成本 | 准确率 | FullP8 掉点 | 误差 | Tr | A | 结果来源 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 默认 | 3 | 14.1% | 5.3% | 8.8% | 85.9% | 0.430 | 0.6946 | 0.21% | 0.02589 | 72.0 | 0.122 | `output/progressive_bfp_fullstack/cora_h8_53_T30_cam32_orderdefault_bfpa6_r0.30/logs/cora_runs3.log` |
| 哈希 | 3 | 31.8% | 12.8% | 19.0% | 68.2% | 0.342 | 0.6905 | 1.02% | 0.05699 | 137.7 | 0.227 | `output/progressive_bfp_fullstack/cora_h8_53_T30_cam32_orderhash_hh4_hbauto_bfpa6_r0.30/logs/cora_runs3.log` |
| 请求 METIS | 3 | 14.1% | 5.3% | 8.8% | 85.9% | 0.430 | 0.6946 | 0.21% | 0.02589 | 72.0 | 0.122 | `output/progressive_bfp_fullstack/cora_h8_53_T30_cam32_ordermetis_metis32_bfpa6_r0.30/logs/cora_runs3.log` |

表中 `Tr` 是平均残差训练对数，`A` 是平均残差校正系数。

主结论：

```text
哈希顺序相比默认顺序:
    复用:      14.1% -> 31.8%  (+17.7 pp)
    直接:       5.3% -> 12.8%  (+7.5 pp)
    残差:       8.8% -> 19.0%  (+10.2 pp)
    编码器路径: 85.9% -> 68.2%  (-17.7 pp)

请求 METIS 相比默认顺序:
    本地结果完全一致，因为 METIS 未加载成功，运行时回退到了默认顺序。
```

因此，在 `CAM32` 这个强容量约束下，默认节点顺序只能保留约 `14.1%` 的复用；哈希顺序能把复用率提高到 `31.8%`，已经进入 `30%+` 的可用区间。这个结果说明：小容量 CAM 的主要瓶颈不是哈希命中本身，而是访问顺序造成的局部性损失；哈希顺序能显著恢复局部性。

## 3. 十轮稳定性检查

`hash` 顺序还跑了 10 轮稳定性检查：

| 节点顺序 | 轮数 | 复用 | 直接 | 残差 | 编码器/P8 | 成本 | 准确率 | FullP8 掉点 | 误差 | Tr | A | 结果来源 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 哈希 | 10 | 33.5% | 13.5% | 19.9% | 66.5% | 0.334 | 0.7023 | 1.18% | 0.05802 | 145.5 | 0.223 | `output/progressive_bfp_fullstack/cora_h8_53_T30_cam32_orderhash_hh4_hbauto_bfpa6_r0.30/logs/cora_runs10.log` |

10 轮结果与 3 轮结果一致：`hash` 顺序稳定保持在 `31%` 到 `34%` 复用区间。相比三轮主对比，10 轮的平均复用率进一步升到 `33.5%`，但 `FullP8 Drop` 也从 `1.02%` 升到 `1.18%`。这仍然处于较低掉点区间。

## 4. 复现方法

从 `OneForAll` 根目录运行：

```bash
cd /home/qiumingzhi/Simhash-S/OneForAll
```

默认节点顺序：

```bash
DATASET=cora RUNS=3 THRESHOLD=30 REFINE_RATIO=0.30 \
  CACHE_SIZE=32 NODE_ORDER_TYPE=default FORCE=1 \
  OUT_DIR=output/progressive_bfp_fullstack/cora_h8_53_T30_cam32_orderdefault_bfpa6_r0.30 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

哈希节点顺序，三轮主对比：

```bash
DATASET=cora RUNS=3 THRESHOLD=30 REFINE_RATIO=0.30 \
  CACHE_SIZE=32 NODE_ORDER_TYPE=hash \
  HASH_NODE_ORDER_HEADS=4 HASH_NODE_ORDER_BLOCK_SIZE=0 FORCE=1 \
  OUT_DIR=output/progressive_bfp_fullstack/cora_h8_53_T30_cam32_orderhash_hh4_hbauto_bfpa6_r0.30 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

哈希节点顺序，十轮稳定性检查：

```bash
DATASET=cora RUNS=10 THRESHOLD=30 REFINE_RATIO=0.30 \
  CACHE_SIZE=32 NODE_ORDER_TYPE=hash \
  HASH_NODE_ORDER_HEADS=4 HASH_NODE_ORDER_BLOCK_SIZE=0 FORCE=1 \
  OUT_DIR=output/progressive_bfp_fullstack/cora_h8_53_T30_cam32_orderhash_hh4_hbauto_bfpa6_r0.30 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

请求 METIS 节点顺序：

```bash
DATASET=cora RUNS=3 THRESHOLD=30 REFINE_RATIO=0.30 \
  CACHE_SIZE=32 NODE_ORDER_TYPE=metis METIS_PARTITION_SIZE=32 FORCE=1 \
  OUT_DIR=output/progressive_bfp_fullstack/cora_h8_53_T30_cam32_ordermetis_metis32_bfpa6_r0.30 \
  bash GraphhopSimhash/scripts/run_progressive_bfp_fullstack.sh
```

如果要真正复现 METIS 分区排序，需要先配置 `METIS_DLL`，例如：

```bash
export METIS_DLL=/path/to/libmetis.so
```

如果没有配置成功，日志会出现如下回退信息：

```text
[METIS] Failed to partition graph (...), falling back to default order
```

当前本地 `METIS` 日志就是这种情况，所以它的最终结果与默认顺序相同。

## 5. 结论

`Cora + CAM32` 的关键结果是：哈希节点顺序能在极小 CAM 容量下显著恢复在线复用，三轮平均复用率达到 `31.8%`，十轮平均达到 `33.5%`。默认顺序和当前回退后的 METIS 顺序只有 `14.1%` 复用。

因此，如果论文或设计文档要讨论小容量 CAM 的可行性，应该优先引用 `hash` 顺序结果；`METIS` 需要在配置好 `METIS_DLL` 后重新跑，不能直接引用当前回退结果。
