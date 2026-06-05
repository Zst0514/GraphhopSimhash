# 硬件模型说明

## 共同算法

对每一个查询节点，数字版和模拟版都遵循同一套高层策略：

```text
1. 查询 8 个相互独立的 16-bit hash head。
2. 在前端检索可复用候选。
3. 按 node_id 聚合候选。
4. 如果存在 support_count >= 3 的候选，则执行复用。
5. 否则执行计算，并把该节点插入缓存。
```

缓存更新逻辑与软件控制器一致：

- 命中复用时，不写入新条目。
- 未命中时，把新计算出的节点写入所有 head 对应的表。

候选的决策优先级如下：

```text
support_count 更高
min_hamming_distance 更低
timestamp 更新
node_id 更小
```

## 数字逻辑模型

数字前端采用“普通 CAM 粗筛 + 数字精确校验”的两级结构：

```text
1. 在每个 head 内，对所有 active row 做 chunk 级 exact-match CAM 比较。
2. 只保留满足粗筛必要条件的 row：
   matching_chunks >= num_chunks - radius
3. 对这些 survivor 做 XOR + popcount，只有 dist <= radius 才算命中。
4. 按 node_id 聚合 survivor，并要求 support >= 3。
```

默认参数如下：

```text
clock_mhz = 500
radius = 2
support_threshold = 3
memo_k = 3
candidate_cam_entries = 512
subarray_rows = 512
parallel_subarrays = 1
cam_chunk_bits = 4
cam_search_cycles = 1
verify_lanes = 32
verify_cycles = 1
shared_verifier_lanes = 1
```

在默认配置 `cam_chunk_bits = 4` 下，每个 16-bit head 会被切成 4 个 4-bit chunk。  
当 `radius = 2` 时，粗筛条件要求至少有 2 个 chunk 完全相等，候选才能进入 `XOR + popcount` 阶段。

```text
16-bit -> 4 个 4-bit chunk
required exact chunks = 4 - 2 = 2
```

能耗模型当前是一级近似：

```text
energy =
  cam_compared_rows * hash_bits * cam_compare_energy_fj_per_bit / 1000
  + verified_rows * hash_bits * xor_popcount_energy_fj_per_bit / 1000
  + candidate_inserts * candidate_cam_probe_energy_pj
  + bucket_writes * cam_write_energy_pj
```

这个模型更适合分析下面这种场景：

- CAM 粗筛已经足够有效；
- `XOR + popcount` 看到的 survivor 集合比较小；
- 因而二级精确校验不会成为主导开销。

其中 `shared_verifier_lanes` 控制 `XOR + popcount` 的资源组织方式：

- `shared_verifier_lanes = 1`：8 个 head 共用一组 verifier lanes；
- `shared_verifier_lanes = 0`：8 个 head 各自有本地 verifier lanes，并行验证，但面积更大。

## 模拟 CAM 模型

模拟 CAM 前端采用 RC / 放电阈值模型，而不是理想化的 `dist <= radius` 判定器。  
对每一条 active row，搜索流程如下：

```text
1. 先把 match line 预充到 VDD。
2. 不匹配 bit 为 match line 提供放电导通。
3. 在 t_eval 时刻做评估：
   V_ML = VDD * exp(-G_total * t_eval / C_ML)
4. 将 V_ML 与 V_ref 比较。
5. 若 V_ML >= V_ref，则该 row 被视为阈值命中。
```

默认自动阈值设置下，`V_ref` 放在标称 `d = 2` 与 `d = 3` 的 match-line 电压中点，因此在零噪声条件下，这个模型实现的是 `dist <= 2` 的确定性近似。

默认参数如下：

```text
clock_mhz = 500
radius = 2
support_threshold = 3
memo_k = 3
subarray_rows = 512
parallel_subarrays = 1
cam_search_cycles = 1
candidate_cam_entries = 512
vdd = 0.9
veval = 0.6
meval_threshold_v = 0.35
matchline_base_cap_f = 6.0e-16
matchline_cap_per_bit_f = 2.0e-16
mismatch_conductance_s = 1.5862000976892227e-5
exact_mismatch_conductance_s = 2.245073554097771e-5
match_leak_conductance_s = 2.0e-7
precharge_time_ps = 30.78091366820998
eval_time_ps = 64
sense_time_ps = 20
fixed_vref = 0.6
comparator_vref = -1.0
device_sigma_rel = 0
sense_noise_sigma_v = 0
comparator_noise_sigma_v = 0
```

这里的默认配置已经不再是旧的 `3-cycle` 保守 proxy，而是
`spice_28nm_16b_timing_proxy`。旧口径仍保留在
`analog_cam_cpp/configs/analog_cam_legacy_proxy.json`。

这个模型把两件事分开处理：

- CAM 检索：通过 RC 放电，把 query hash 与 active row 做阈值比较；
- 数字候选聚合：按 node_id 合并，并统计 support。

为了更接近 2025 HD-CAM 硅片论文，当前模型额外支持两类控制量：

- `veval`：用来模拟论文中 `Meval` 管的过驱动电压。降低 `veval` 会减小 mismatch 放电导通，从而放慢 ML 放电速度。
- `fixed_vref`：强制使用固定参考电压，而不是自动取 `d=2` 与 `d=3` 的中点。

这使得模型可以表达更接近论文的工作点，例如：

```text
VDD = 1.2 V
Veval = 0.8 V
Vref = 0.8 V
```

不过，这仍然是 `16-bit / 28nm proxy`，不是对 65nm、64-bit 硅宏的直接参数复刻。

能耗模型同样是代理模型：

```text
energy += active_rows * hash_bits * cam_compare_energy_fj_per_bit / 1000
energy += writes * cam_write_energy_pj
energy += candidate_inserts * candidate_cam_probe_energy_pj
```

搜索延迟由下面的关系推导：

```text
search_time_ps = precharge_time_ps + eval_time_ps + sense_time_ps
search_cycles = ceil(search_time_ps / clock_period_ps)
```

`cam_search_cycles` 仍然保留为一个可配置的周期下限。  

- 如果 `parallel_subarrays = 1`，默认认为多个 subarray 并行评估，因此时延由单个 subarray 的评估时间决定，而能耗仍随 active row 数增长。
- 如果 `parallel_subarrays = 0`，则按串行 subarray 处理，在周期模型里会把多个 subarray 的延迟累加起来。

## 两种模型的可比性

为了保证对比尽量公平，两种引擎都满足以下条件：

- 读取同一份 trace；
- 使用相同的 `support_threshold`；
- 使用相同的 `memo_k` 保留策略；
- 使用相同的 candidate 容量限制；
- 输出相同格式的逐节点决策结果。

两者真正不同的部分，只在于“候选检索前端”：

- 数字版：普通 CAM chunk 粗筛 + `XOR + popcount` 精确校验；
- 模拟版：对 active CAM row 做汉明距离阈值搜索。

当前模型仍然是紧凑的行为级模型，不是 SPICE signoff 级电路模型。  
它目前已经显式捕捉了这些因素：

- match-line 预充；
- mismatch 数量导致的 RC 放电差异；
- 阈值感测；
- 固定器件偏差与动态 sense 噪声。

如果要得到更接近物理实现的结论，应当把这里的 proxy 参数替换成 CAMASim / EvaCAM 或更底层电路模型校准后的数值，并在配置里记录 calibration 来源。

## 论文风格前端速度探针

为了单独分析 “HD-CAM 前端相对普通 CAM 慢多少”，子项目新增了一个前端探针工具：

```bash
./cmake-build-release/analog_cam_cpp/analog_cam_frontend_probe \
  --config analog_cam_cpp/configs/analog_cam_paper_like_28nm16b.json \
  --word_bits 16 \
  --max_dist 5 \
  --out reports/analog_cam_frontend_probe_28nm16b.md
```

它会输出：

- `d = 0..5` 的 `V_ML`
- 每个 `d` 相对固定 `Vref` 的 crossing time
- `Exact CAM` 的 `d=0/1` 判决时间
- `HD-CAM` 的 `d=2/3` 判决时间
- `HD-CAM / Exact-CAM` 的评估时间倍率与完整搜索时间倍率

在论文风格 `16-bit / 28nm` proxy 配置下，探针结果为：

- `Exact CAM`：`d=0/1` 评估时间约 `7.590 ps`
- `HD-CAM`：`d=2/3` 评估时间约 `11.883 ps`
- 纯评估窗口倍率约 `1.5656x`
- 若把共同的 precharge/sense 一起算入，完整搜索时间倍率约 `1.0902x`

这个结果的含义是：

- 如果只看“ML 放电判边界”这一小段，HD-CAM 比普通 CAM 慢得更明显；
- 如果把 precharge 和 sensing 一起算进去，二者的总前端搜索时间差距会被共同开销压缩；
- 因此，HD-CAM 的真实速度损失更应该拆成“纯评估窗口”与“完整搜索路径”两层来看。

## 三个真实 Trace 下的结果对比（500 MHz）

下面的结果统一按 `500 MHz` 口径比较，并且模拟前端已经切换到默认
`spice_28nm_16b_timing_proxy`：

- `cora`、`pubmed`：三种实现都直接跑同一份 trace 得到；
- `arxiv`：`Digital CAM, per-head verifier` 为直接跑数，`Digital CAM, shared verifier` 由同一份逐 query 统计回推；
- `arxiv` 的 `Analog HD-CAM` 延迟按相同 `reuse/miss` 统计和默认 `1-cycle` 搜索时间公式回填，不再使用旧的 `3-cycle` 保守下限。

对齐验证结果如下：

- `cora`: `shared_cycles_delta=0.000000`
- `pubmed`: `shared_cycles_delta=0.000000`

### 汇总表

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

### 结果解读

- `cora`：在默认 `spice_28nm_16b_timing_proxy` 下，模拟版已经快于两种数字组织方式。
- `pubmed`：模拟版明显更快，数字版如果共享 `verifier` 会进一步变慢。
- `arxiv`：模拟版优势非常大；数字版即便采用 `per-head verifier`，二级 `XOR + popcount` 开销仍然很重。
- `shared verifier` 在三个数据集上都是最慢的数字组织方式。
- `per-head verifier` 明显改善延迟，但面积代理更大。

这里的“模拟版更快”说的是**总查询周期**，不是“扣掉 verify 后前端 search 还更快”。
按当前 `500 MHz` 的整周期模型：

- 数字搜索前端：`1 cycle`
- 模拟搜索前端：`1 cycle`

因此把数字版的 `verify_cycles/query` 扣掉以后，两边会对齐。
如果回到 SPICE 的 `ps` 级前端口径，模拟前端实际上仍比普通精确 CAM 略慢，
只是这点差异在 `2 ns` 的整周期模型里不会体现成额外周期。

### 容量 baseline 口径修正

和 `512KB` 容量实验相关的 baseline 口径这里补充说明一下：

- 旧的 `per_hash_fifo` 路径不是“无限大 CAM baseline”；
- 它实际上会对每个 `head`、每个 exact `16-bit hash` 只保留 `memo_k=3` 条记录；
- 因此它更像一个 `per-hash memo heuristic`，而不是一个真正的全局大容量 CAM。

现在容量实验统一改成：

- `global_unbounded`：作为“无限容量存储 baseline”
- `global_lru + total_cam_bytes=512KB`：作为有限容量实现

在这个新口径下：

- `cora`、`pubmed`：`global_unbounded` 和 `512KB` 结果完全一致，说明没有碰到容量墙；
- `arxiv`：`512KB` 相比 `global_unbounded` 的 `reuse_rate` 从 `82.90%` 降到 `80.71%`，也就是 `-2.19 pp`；
- `arxiv` 的功能级 decision drift 为 `21841 / 169343 = 12.90%`。

详细对比见：

- [capacity_lru_512k_summary.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/reports/capacity_lru_512k_summary.md)
- [arxiv_digital_global_unbounded_vs_512k.md](/home/qiumingzhi/Simhash-S/OneForAll/GraphhopSimhash/CAM_sim/reports/arxiv_digital_global_unbounded_vs_512k.md)

### 为什么规模越大，HD-CAM 优势越明显

这组结果体现的不是“模拟 CAM 天然随规模更快”，而是下面这条更具体的规律：

- 数字路线的前端 `CAM` 搜索周期基本固定；
- 但二级 `XOR + popcount` 校验开销会随着 survivor 数快速增长；
- 在固定 `8 x 16-bit`、固定 `HD<=2`、固定 chunk 粗筛配置下，数据集越大，粗筛之后剩余的 survivor 越多；
- `HD-CAM` 在当前默认配置里只占 `1` 个搜索周期，因此搜索延迟基本不随 survivor 数增长。

对应到表中的 `Verified rows/query`：

- `cora`: `208.651`
- `pubmed`: `694.308`
- `arxiv`: `4780.403`

这也是为什么 `arxiv` 上数字版的 `Verify cycles/query` 会明显膨胀，而模拟版延迟仍然保持在接近常数的水平。
