# `8 x 16bit` 检索下数字 CAM 与模拟 HD-CAM 的多数据集耗时比较

## 1. 这份文档回答什么

这份文档只回答一个很具体的问题：

```text
在当前 `8 x 16bit` 哈希检索结构下，
数字 CAM 和模拟 HD-CAM
在不同数据集上的耗时到底差多少？
```

为了方便直接查结论，这里把原本分散在：

- [HARDWARE_MODEL.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/HARDWARE_MODEL.md)
- [three_trace_three_impls_500mhz.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/reports/three_trace_three_impls_500mhz.md)

里的结果单独整理成一页。

## 2. 比较口径

所有结果统一按下面的口径比较：

- 时钟：`500 MHz`
- 查询结构：`8 x 16bit` hash heads
- 复用规则：`support >= 3`
- 数据集：`cora`、`pubmed`、`arxiv`
- 模拟默认配置：`spice_28nm_16b_timing_proxy`

比较的三种实现分别是：

- `Digital CAM, shared verifier`
- `Digital CAM, per-head verifier`
- `Analog HD-CAM`

说明：

- `cora`、`pubmed` 三种实现都来自直接跑同一份 trace 的结果。
- `arxiv` 的 `Digital CAM, shared verifier` 和 `Analog HD-CAM` 是从同一份 `per-head` 数字跑数逐 query 回推得到。
- 对 `Analog HD-CAM` 而言，`arxiv` 这一行使用的是精确 `reuse/miss` 统计加默认 `1-cycle` 搜索时间公式回填。
- 这个回推方法已经在 `cora` 和 `pubmed` 上与实际跑数对齐验证过。

## 3. 结果总表

| Dataset | Implementation | Reuse | Cycles/query | Latency (ns) | Search cycles/query | Verify cycles/query | Verified rows/query | Energy/query (pJ) | Area proxy (um2) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cora | Digital CAM, shared verifier | 55.80% | 9.447 | 18.894 | 1.000 | 7.005 | 208.651 | 45.243 | 30802.88 |
| cora | Digital CAM, per-head verifier | 55.80% | 4.261 | 8.521 | 1.000 | 1.819 | 208.651 | 45.243 | 35282.88 |
| cora | Analog HD-CAM | 55.80% | 2.442 | 4.884 | 1.000 | 0.000 | 0.000 | 41.236 | 30162.88 |
| pubmed | Digital CAM, shared verifier | 81.41% | 24.366 | 48.731 | 1.000 | 22.180 | 694.308 | 144.677 | 87057.92 |
| pubmed | Digital CAM, per-head verifier | 81.41% | 6.811 | 13.621 | 1.000 | 4.625 | 694.308 | 144.677 | 91537.92 |
| pubmed | Analog HD-CAM | 81.41% | 2.186 | 4.372 | 1.000 | 0.000 | 0.000 | 131.346 | 86417.92 |
| arxiv | Digital CAM, shared verifier | 79.25% | 152.080 | 304.160 | 1.000 | 149.872 | 4780.403 | 1040.007 | 708590.72 |
| arxiv | Digital CAM, per-head verifier | 79.25% | 28.916 | 57.833 | 1.000 | 26.709 | 4780.403 | 1040.007 | 713070.72 |
| arxiv | Analog HD-CAM | 79.25% | 2.207 | 4.415 | 1.000 | 0.000 | 0.000 | 948.224 | 707950.72 |

## 4. 只看耗时时，最重要的结论

如果只看 `Latency (ns)`，那结论可以直接浓缩成下面三句：

- `cora`：最快的是 `Analog HD-CAM`，`4.884 ns` 对 `8.521 ns`。
- `pubmed`：最快的是 `Analog HD-CAM`，`4.372 ns` 对 `13.621 ns`。
- `arxiv`：最快的是 `Analog HD-CAM`，`4.415 ns` 对 `57.833 ns`。

这里要特别注意：

- 上面的“更快”指的是**总查询延迟**
- 不是“扣掉数字 verify 以后，模拟前端本身还更快”

在当前 `500 MHz` 的整周期 C++ 模型里：

- 数字 CAM 前端搜索 = `1 cycle`
- 模拟 HD-CAM 前端搜索 = `1 cycle`

所以把数字版的 `verify_cycles/query` 扣掉以后，两边其实会对齐。
真正让模拟版在总延迟上占优的，是它没有数字 `XOR + popcount + threshold`
这段会随 survivor 数增长的后端校验开销。

换一种更直观的说法：

- `cora` 上，模拟版比最优数字版快约 `1.74x`
- `pubmed` 上，模拟版比最优数字版快约 `3.12x`
- `arxiv` 上，模拟版比最优数字版快约 `13.10x`

## 5. 为什么不同数据集的结论不一样

这里最关键的不是前端 CAM 本身，而是数字方案后面的 `XOR + popcount + threshold` 校验压力。

在当前模型里：

- 数字路线的前端 `CAM` 搜索周期基本固定
- 但 survivor 越多，后端 `verify` 的开销就越大
- 模拟 HD-CAM 直接在前端做阈值判断，因此没有这一步按 survivor 数增长的数字校验成本

这在 `Verified rows/query` 里体现得很明显：

- `cora`：`208.651`
- `pubmed`：`694.308`
- `arxiv`：`4780.403`

也就是说：

- `cora` 的 survivor 不算太多，但在新的 `1-cycle` 搜索时间口径下，模拟前端已经能占优
- `pubmed` survivor 更多，模拟前端优势继续扩大
- `arxiv` survivor 非常多，数字后端校验成为主瓶颈，模拟前端的优势被大幅放大

## 6. 这份比较该怎么用

如果你要回答“在 `8 x 16bit` 检索下，不同数据集里模拟 CAM 和数字 CAM 谁更快”，可以直接引用下面这版简表：

| Dataset | Fastest impl. | Best latency (ns) | Runner-up latency (ns) | 结论 |
|---|---|---:|---:|---|
| cora | Analog HD-CAM | 4.884 | 8.521 | 模拟更快 |
| pubmed | Analog HD-CAM | 4.372 | 13.621 | 模拟更快 |
| arxiv | Analog HD-CAM | 4.415 | 57.833 | 模拟明显更快 |

如果你要回答“模拟 CAM 是否天然总是更快”，答案仍然不能简单说“天然如此”：

- 不是天然总更快
- 它是否占优，仍然取决于数字路线的 survivor 校验负担有多重
- 但在当前默认 `spice_28nm_16b_timing_proxy` 下，三组公开 trace 里模拟版都已经占优

如果只看“前端 search 本身”：

- 在当前 `500 MHz` 整周期模型里，数字和模拟都是 `1 cycle`
- 在 SPICE 的 `ps` 级口径里，模拟前端其实比普通数字/精确 CAM 略慢，约 `1.027x`

## 7. 原始结果位置

如果你后面要继续追原始数据，建议从下面这些文件看起：

- [HARDWARE_MODEL.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/docs/HARDWARE_MODEL.md)
- [three_trace_three_impls_500mhz.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/reports/three_trace_three_impls_500mhz.md)
- [cora_analog_500mhz.json](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/reports/cora_analog_500mhz.json)
- [cora_digital_per_head_verify_500mhz.json](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/reports/cora_digital_per_head_verify_500mhz.json)
- [pubmed_analog_same_trace.json](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/reports/pubmed_analog_same_trace.json)
- [pubmed_digital_per_head_verify_500mhz.json](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/reports/pubmed_digital_per_head_verify_500mhz.json)
- [arxiv_digital_per_head_verify_500mhz.json](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/reports/arxiv_digital_per_head_verify_500mhz.json)
