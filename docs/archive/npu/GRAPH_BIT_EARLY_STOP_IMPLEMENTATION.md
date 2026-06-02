# Graph-Bit Bit-Plane Early Stop Implementation

本文档专门解释 Graph-Bit 里 bit-plane early stop 从算法配置到代码实现的完整路径。它回答的是：

```text
一个 miss node 进入 LLaMA encoder 后，
系统如何决定它算到几 bit，
ONNXim 里又如何把这个 stop depth 反映到 activation-side issue、psum update 和最终 trace replay。
```

相关设计总览见：

```text
docs/npu/GRAPH_BIT_NPU_DESIGN.md
docs/archive/npu/GRAPH_BIT_FULLSTACK_REPRODUCTION_GUIDE.md
```

## 0. Current V2 Direction: Tile-Aware Risk Scoring

前面的 hard validation 说明，单纯使用

```text
omitted_low_bits / 255
```

作为 stop bound 过于乐观。真正影响当前 GEMM tile 输出的是：

```text
A_low @ W_tile
```

因此 Graph-Bit 的下一版 stop policy 不再把低位幅度单独当作判断依据，而是把三项合在一起：

```text
node risk:
    degree / propagation risk，表示该节点错误传播到 GNN 后端的风险。

W tile risk:
    当前 W tile 的数值强度，表示低位 activation 乘上这块权重后是否容易被放大。

low-bit budget:
    停在某个 depth 后，剩余 activation low bits 的最大幅度。
```

核心评分：

```text
node_norm(v) = degree_q(v) / 15

w_norm(tile) =
    clamp(W_tile_strength / reference_strength, 0, w_cap)

low_norm(depth) =
    (2^(8 - depth) - 1) / 255

risk_score(v, tile, depth) =
    node_norm(v)^alpha
    * w_norm(tile)^beta
    * low_norm(depth)
```

停止规则：

```text
for depth in 8, 7, 6, 5, 4:
    score = risk_score(v, tile, depth)

    if score <= tau:
        stop at depth
        break
```

这版机制的含义是：

```text
同一个低风险节点：
    在弱 W tile 上可以更早停止。
    在强 W tile / outlier tile 上需要继续执行更多 bit-plane。

同一个 W tile：
    服务高 degree / 高传播风险节点时更保守。
    服务低风险节点时更激进。
```

第一版参数保持轻量：

```text
alpha = 1.0
beta  = 1.0
w_cap = 2.0

tau sweep:
    0.005 / 0.01 / 0.02 / 0.04 / 0.08
```

输出指标：

```text
AvgDepth
P8/P7/P6/P5/P4 distribution
classification drop
actual_delta_ratio distribution
ONNXim activity / replay cost
```

这样得到的 stop depth 不是预设比例，也不是直接指定 P8/P6/P5，而是由：

```text
node risk + W tile strength + remaining low-bit budget
```

共同决定。

### 0.1 W Tile Strength

对 LLaMA Linear 层，GEMM 形式为：

```text
X[M, K] @ W[K, N]
```

实现里通常按 tile 执行：

```text
X_tile[M_t, K_t] @ W_tile[K_t, N_t]
```

当前 profiling 默认：

```text
K_t = 128
N_t = 128
```

也就是一个 W tile 约包含：

```text
128 * 128 = 16384 weights
```

如果权重是 W4，裸权重大约是：

```text
16384 * 4 bit = 65536 bit = 8 KB
```

相比之下，LLaMA-7B 的全模型权重是 GB 级，不可能整体常驻一个小阵列的 RF/SRAM；硬件实际复用的是一块块 W tile。

W tile strength 可以用下面几种统计量：

```text
mean_abs:
    mean(abs(W_tile))

max_abs:
    max(abs(W_tile))

row_l1_p95:
    对 W_tile 每个输出 channel 计算 sum(abs(w_row))，
    再取 p95，表示这块 tile 的保守强度。
```

推荐第一版使用：

```text
W_tile_strength = row_l1_p95 / (layer_mean_abs * K_t)
```

原因是：

```text
1. 它比 mean_abs 更能保护 outlier output channel。
2. 它比 max_abs 没那么极端，不会让所有 tile 都退化成 P8。
3. 它不需要 learned predictor，只是模型权重的静态统计。
```

### 0.2 Why 128x128 Is Not a Fixed Assumption

`128x128` 只是当前 ONNXim / profiling 的默认 tile，用来和 systolic-array block 以及 LLaMA hidden size 对齐：

```text
K = 4096 可以被 128 整除。
N = 4096 / 11008 也可以按 128 分块处理。
W4 tile 大小约 8 KB，适合放入片上 SRAM/RF 做局部复用。
```

但 Graph-Bit 机制不依赖固定 `128x128`。tile size 会影响三件事：

```text
1. W strength 分布
   tile 越大，越容易包含 outlier channel，bound 更保守。

2. control granularity
   tile 越小，stop decision 越细；但 metadata、调度和边界处理开销更高。

3. W-stationary reuse
   tile 越大，单次 W load 更多；如果 bucket 内 token rows 足够多，摊薄收益更明显。
```

因此后续需要做 tile-size sensitivity：

```text
K_t/N_t:
    64x64
    64x128
    128x128
    256x128
    256x256
```

每组都输出：

```text
W strength distribution
AvgDepth
P8/P7/P6/P5/P4 distribution
drop
activity / replay cost
```

如果不同 tile size 下 Pareto 趋势一致，说明机制是 general 的；如果结论只在 `128x128` 成立，就不能把它写成架构主结论，只能作为某个硬件配置下的 profiling 结果。

## 1. 核心定义

Graph-Bit 里的 early stop 不是 learned predictor，也不是 oracle error。它的基本逻辑是：

```text
graph risk -> min_depth + tolerance
runtime bound -> actual stop depth
```

也就是说，Degree / TSER 等图风险分数不直接规定最终一定算 P8/P6/P4，而是规定：

```text
min_depth:
    这个风险等级至少要算到几 bit。

tolerance:
    剩余低位 bit-plane 的理论贡献界限小到什么程度才允许停。
```

然后 runtime bound 从 `min_depth` 开始检查：

```text
for depth in min_depth..8:
    if remaining_low_bit_bound(depth) <= tolerance:
        stop at depth
        break
```

当前默认配置：

```text
high-risk:
    min_depth = 8
    tolerance = 0.00
    -> 基本完整 P8

mid-risk:
    min_depth = 6
    tolerance = 0.02
    -> 通常停在 P6

low-risk:
    min_depth = 4
    tolerance = 0.04
    -> 当前常见会停在 P5
```

其中 P8/P6/P5/P4 在 accuracy validation 里对应已有 embedding pools：

```text
P8 = W4A8
P6 = W4A6
P5 = W4A5
P4 = W4A4
```

在硬件解释里，它们对应 activation bit-plane 执行深度：

```text
A8: b7 b6 b5 b4 b3 b2 b1 b0
P8: b7..b0
P6: b7..b2
P5: b7..b3
P4: b7..b4
```

当前第一版不加入 operator sensitivity LUT。主线使用：

```text
node tolerance:
    degree / propagation risk

runtime bound:
    A_low_bound(depth) * W_tile_abs_bound

op sensitivity:
    统一设为 1
```

因此图风险不直接指定最终 P8/P6/P5/P4，而是给出节点级 tolerance；当前 tile 的 stop depth 由剩余 activation 低位上界和 W tile 强度共同决定。

当前代码里的 W tile 强度先用一个显式标量实现：

```text
A_low_bound(depth) = (2^(8 - depth) - 1) / 255

remaining_low_bit_bound(depth)
    = bound_scale
    * A_low_bound(depth)
    * sqrt(tile_k / 128)
    * w_strength
```

其中 `w_strength` 表示当前 W tile 相对全局平均权重强度的倍数。`w_strength > 1` 时，同样跳过低位 activation 的风险更高；`w_strength < 1` 时，低位可跳过空间更大。第一版用常数扫描验证机制，后续再接真实 `mean(abs(W_tile)) / global_mean(abs(W))`。

除了直接放大 `remaining_low_bit_bound`，当前实现还支持把 W tile 强度并入节点风险，形成一个轻量的加权折中：

```text
node_risk(v) = clamp(graph_risk(v) / risk_max, 0, 1)

w_risk = clamp((w_strength - 1) / (w_reference - 1), 0, 1)

effective_risk(v) =
    (node_weight * node_risk(v) + w_weight * w_risk)
    / (node_weight + w_weight)

tolerance(v) =
    min_tol + (max_tol - min_tol) * (1 - effective_risk(v))^gamma
```

这里有两个互补作用：

```text
1. w_strength 进入 remaining bound:
   W tile 越强，剩余低位的理论影响越大。

2. w_risk 进入 tolerance:
   W tile 越强，节点容忍度越保守。
```

因此最终 stop depth 不是固定比例，也不是只看 degree。它由三项共同决定：

```text
图传播风险: 哪些节点更需要保护。
W tile 强度: 当前 GEMM tile 对低位误差是否敏感。
runtime bound: 当前 depth 后剩余低位是否已经足够小。
```

真实 W tile 统计脚本：

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/profile_llama_w_tile_strength.py \
  --tile_k 128 \
  --tile_n 128 \
  --output_dir output/graphbit_w_tile_strength/llama2_7b_k128_n128
```

脚本按 LLaMA Linear 权重 `[N, K]` 切 tile，并统计：

```text
strength_mean = mean_j(sum_k |W[j,k]|) / (mean_abs(W_layer) * tile_k)
strength_p95  = p95_j(sum_k |W[j,k]|)  / (mean_abs(W_layer) * tile_k)
strength_max  = max_j(sum_k |W[j,k]|)  / (mean_abs(W_layer) * tile_k)
```

推荐先用 `strength_p95` 的全局 p50/p75/p90/p95 作为 `w_strength` sweep 值。

当前 W-bound ablation 分成三类：

```text
No-W-bound:
    w_strength = 1.0
    只看 activation low-bit bound。

Global-W-bound:
    w_strength = global p75 / p90 / p95
    用全模型 W tile profile 给一个统一保守因子。

Module-W-bound:
    w_strength = MAC-weighted module p75 / p90 / p95
    q/k/v/o/gate/up/down 按线性层 MAC 占比合成一个 module-aware 因子。
```

生成并运行这三组 policy：

```bash
DATASETS="cora" RUNS=1 \
bash GraphhopSimhash/scripts/run_graphbit_w_bound_ablation.sh
```

关键中间文件：

```text
scripts/build_graphbit_w_bound_policies.py
output/graphbit_w_bound_ablation/policies.txt
output/graphbit_w_bound_ablation/summary.tsv
output/graphbit_w_bound_ablation/pareto.tsv
```

这组 ablation 用来回答一个具体问题：

```text
在相同 node tolerance 下，
加入 W tile 强度是否能让 stop-depth 更保守地避开高风险低位跳过？
```

如果 W-bound 在相近 AvgDepth 下 drop 更低，或者在相近 drop 下 AvgDepth 更低，说明 W tile 强度项是有效的。当前 Cora 3-run 结果显示，W-bound 会把一部分 P5 决策推回 P6/P7，drop 更稳：

```text
policy      AvgDepth  Drop   ExtraDrop  CostSaveVsFull
now_no_w    5.38      2.56%  0.98%      27.76%
global_p75  5.80      2.42%  0.97%      23.33%
global_p90  6.02      2.29%  0.84%      21.00%
global_p95  6.02      2.29%  0.84%      21.00%
module_p75  5.80      2.40%  0.93%      23.33%
module_p90  6.01      2.27%  0.82%      21.00%
module_p95  6.01      2.40%  0.82%      21.07%
```

这说明 `W_strength` 不是简单让策略更保守，而是在同一套 node tolerance 下改变 stop-depth 分布：No-W 更激进、平均深度更低；Global/Module W-bound 更稳，当前 Cora 上 `module_p90` 是较好的折中点。

进一步把 `W_strength` 映射成 `w_risk` 后，可以验证 node risk 和 W risk 的折中强度。Cora 3-run 结果如下：

```text
policy          node/w  AvgDepth  P6/P5          Drop   ExtraDrop  CostSave
no_w_node       1.0/0.0 5.38      22.0/37.4%    2.55%  1.10%      27.67%
module_p75_node 1.0/0.0 5.80      46.4/12.5%    2.55%  0.97%      23.41%
module_p75_w20  0.8/0.2 6.01      59.4/0.0%     2.30%  0.85%      21.33%
module_p75_w50  0.5/0.5 6.00      59.7/0.0%     2.40%  0.85%      21.26%
module_p90_node 1.0/0.0 6.01      59.1/0.0%     2.27%  0.82%      21.00%
module_p90_w20  0.8/0.2 6.01      59.1/0.0%     2.27%  0.80%      21.00%
module_p90_w50  0.5/0.5 6.01      59.0/0.0%     2.40%  0.82%      21.07%
```

结论：

```text
1. No-W 最激进，AvgDepth 最低，但会把较多 miss nodes 推到 P5，drop 更高。
2. 加入 W-bound 后，P5 基本被拉回 P6/P7，drop 更稳。
3. 20% W risk 已经足够体现 W tile 强度约束；50% W risk 更保守，但没有继续带来精度收益。
4. 当前 Cora 上 module_p90_node / module_p90_w20 是更稳的折中点。
```

PubMed 3-run 结果如下：

```text
policy          node/w  AvgDepth  P6/P5          Drop   ExtraDrop  CostSave
no_w_node       1.0/0.0 5.32      17.0/40.6%    3.27%  1.38%      28.33%
module_p75_node 1.0/0.0 5.58      29.5/27.0%    3.10%  1.25%      25.85%
module_p75_w20  0.8/0.2 6.02      57.6/0.0%     2.94%  1.05%      21.16%
module_p75_w50  0.5/0.5 6.01      58.3/0.0%     2.93%  1.05%      21.16%
module_p90_node 1.0/0.0 6.04      56.4/0.0%     2.85%  1.01%      21.09%
module_p90_w20  0.8/0.2 6.04      56.3/0.0%     2.89%  1.01%      20.82%
module_p90_w50  0.5/0.5 6.04      56.4/0.0%     2.85%  1.01%      21.09%
```

跨 Cora / PubMed 合并看，当前更稳的折中点是 `module_p90_node` 或 `module_p90_w50`：

```text
policy          CoraDrop  PubMedDrop  MaxDrop  MeanCostSave  MeanAvgDepth
module_p90_node 2.27%     2.85%       2.85%    21.05%        6.02
module_p90_w20  2.27%     2.89%       2.89%    20.91%        6.02
module_p90_w50  2.40%     2.85%       2.85%    21.08%        6.02
module_p75_w20  2.30%     2.94%       2.94%    21.24%        6.01
no_w_node       2.55%     3.27%       3.27%    28.00%        5.35
```

解释：

```text
No-W:
    更激进，平均 depth 更低，cost saving 更高；
    但 PubMed 会有大量 P5，drop 超过 3%。

Module/W-bound:
    用 W tile 强度约束 runtime-bound；
    把 PubMed 的 P5 拉回 P6/P7，drop 回到 3% 内。

当前默认建议:
    module_p90_node / module_p90_w20 / module_p90_w50 都是可用折中点。
    如果强调“node risk + W risk 联合控制”的叙事，module_p90_w20 更直接；
    如果强调跨数据集 maxDrop，module_p90_node / module_p90_w50 更稳。
```

对应输出：

```text
output/graphbit_weighted_bound_validation_cora_runs3/summary.tsv
output/graphbit_weighted_bound_validation_cora_runs3/pareto.tsv
output/graphbit_weighted_bound_validation_pubmed_runs3/summary.tsv
output/graphbit_weighted_bound_validation_pubmed_runs3/pareto.tsv
```

## 2. `max_tol` 不是固定常数

`max_tol` 控制低风险 miss node 的最大误差容忍度。它不是常数真理，而是 Graph-Bit 的核心调参旋钮之一。

固定当前前端和 W-bound：

```text
front-end: h8_53_T31
hard direct: support >= 5
residual: support = 3..4
Graph-Bit: support < 3 或 residual gate reject
W policy: module_p90_w20
node/W risk weight: 0.8 / 0.2
```

只扫描 `max_tol`，Cora 3-run 结果如下：

```text
max_tol  P7     P6     P5     AvgDepth  Drop   ExtraDrop  CostSaveVsFull
0.02     46.9%  12.5%  0.0%   6.79      2.01%  0.43%      12.71%
0.03      4.9%  54.7%  0.0%   6.08      2.29%  0.82%      20.33%
0.04      0.6%  59.1%  0.0%   6.01      2.26%  0.79%      21.00%
0.05      0.3%  33.3%  26.1%  5.57      2.45%  1.00%      25.67%
0.06      0.1%   9.5%  50.1%  5.16      2.98%  1.51%      30.00%
0.08      0.1%   1.1%  58.4%  5.02      3.13%  1.55%      31.44%
```

解释：

```text
max_tol=0.02:
    很保守，大量节点停在 P7，drop 低，但 cost saving 小。

max_tol=0.03 / 0.04:
    主要停在 P6，是当前 Cora 上较稳的 P6 平台区间。

max_tol=0.05:
    开始释放 P5，cost saving 提升到 25%+，drop 仍在 2.5% 左右。

max_tol>=0.06:
    大量节点停在 P5，cost saving 更高，但 drop 接近或超过 3%。
```

当前结论：

```text
0.04 不是固定真理，只是较稳的 P6 区间点。
如果主目标是稳定精度，优先 0.03 / 0.04。
如果主目标是更高省算，0.05 是值得保留的激进候选。
0.06 / 0.08 更适合作为上界对照，不建议作为默认主线。
```

复现命令：

```bash
POLICIES=$'tol002:4:0.0:0.02:1.0:15:1.0:1.295612:0.8:0.2:1.473038\ntol003:4:0.0:0.03:1.0:15:1.0:1.295612:0.8:0.2:1.473038\ntol004:4:0.0:0.04:1.0:15:1.0:1.295612:0.8:0.2:1.473038\ntol005:4:0.0:0.05:1.0:15:1.0:1.295612:0.8:0.2:1.473038\ntol006:4:0.0:0.06:1.0:15:1.0:1.295612:0.8:0.2:1.473038\ntol008:4:0.0:0.08:1.0:15:1.0:1.295612:0.8:0.2:1.473038' \
DATASETS="cora" RUNS=3 \
OUT_ROOT="output/graphbit_maxtol_sweep_cora_runs3" \
bash GraphhopSimhash/scripts/run_t31_graphbit_nodewise_bound_sweep.sh
```

输出：

```text
output/graphbit_maxtol_sweep_cora_runs3/summary.tsv
output/graphbit_maxtol_sweep_cora_runs3/pareto.tsv
```

## 3. CLI 参数入口

参数定义在：

```text
GraphhopSimhash/cli.py
```

核心参数如下：

```text
--precision_depth_bound_enable
    打开 Graph-Bit runtime-bound policy。

--precision_depth_bound_priorities degree tser low_unique random
    用哪些 risk proxy 生成 bound policy。

--precision_depth_bound_high_min_depth
--precision_depth_bound_mid_min_depth
--precision_depth_bound_low_min_depth
    high / mid / low 三个风险 bucket 的最低安全深度。

--precision_depth_bound_high_tolerance
--precision_depth_bound_mid_tolerance
--precision_depth_bound_low_tolerance
    high / mid / low 三个风险 bucket 的 early-stop 容忍度。

--precision_depth_bound_scale
    bound 的整体缩放因子。

--precision_depth_bound_tile_k
    bound 估计使用的 K tile 大小，默认 128。

--precision_depth_bound_w_strength
    W tile 强度乘子。当前作为常数输入；后续可替换为 per-tile metadata。

--precision_depth_bound_node_risk_weight
--precision_depth_bound_w_risk_weight
    effective_risk 中 node risk 和 W risk 的权重。

--precision_depth_bound_w_risk_reference
    把 w_strength 映射到 [0,1] 时使用的参考强度。

--precision_depth_trace_export_dir
--precision_depth_trace_export_configs
    导出 per-node trace，供 scheduler replay 使用。
```

脚本入口通常是：

```text
GraphhopSimhash/scripts/run_graphbit_predictor_free_flow.sh
```

该脚本把 residual reuse 前端和 Graph-Bit miss-node 后端接起来。默认使用：

```text
BOUND_ENABLE=1
BOUND_PRIORITIES=(degree tser)
BOUND_HIGH_MIN=8
BOUND_MID_MIN=6
BOUND_LOW_MIN=4
BOUND_HIGH_TOL=0.0
BOUND_MID_TOL=0.02
BOUND_LOW_TOL=0.04
BOUND_TILE_K=128
```

## 4. Runner 里的 stop depth 计算

主要实现在：

```text
GraphhopSimhash/runner.py
```

### 3.1 生成 bound policy

函数：

```text
build_precision_depth_policy_configs(args, bits)
```

如果 `--precision_depth_bound_enable` 打开，会额外加入：

```text
DegBound
TSERBound
UniqBound
RandBound
```

每个 policy 的本质是：

```text
用某个 risk score 排序 miss nodes，
高风险节点进 high bucket，
中等风险节点进 mid bucket，
低风险节点进 low bucket。
```

### 3.2 predictor-free remaining-bit bound

函数：

```text
precision_depth_remaining_bit_bound(depth, ref_bit, args)
```

当前 Python validation 侧使用归一化低位范围作为 bound：

```text
omitted = 2^(ref_bit - depth) - 1
denom   = 2^ref_bit - 1
bound   = scale * omitted / denom * sqrt(tile_k / 128)
```

以 `ref_bit=8, tile_k=128, scale=1` 为例：

```text
depth=8:
    bound = 0

depth=6:
    omitted = 2^(8-6)-1 = 3
    bound = 3 / 255 = 0.01176

depth=5:
    omitted = 7
    bound = 7 / 255 = 0.02745

depth=4:
    omitted = 15
    bound = 15 / 255 = 0.05882
```

所以默认 tolerance 下：

```text
high: min=8, tau=0.00
    depth=8, bound=0

mid: min=6, tau=0.02
    depth=6, bound=0.01176 <= 0.02

low: min=4, tau=0.04
    depth=4, bound=0.05882 > 0.04
    depth=5, bound=0.02745 <= 0.04
```

因此当前常见输出是：

```text
high -> runtime P8
mid  -> runtime P6
low  -> runtime P5
```

这也是为什么你会在 trace 里看到 D5/D6/D8，而不是简单的 D4/D6/D8。

### 3.3 从 min_depth/tolerance 到 runtime depth

函数：

```text
select_runtime_bound_depth(min_depth, tolerance, ref_bit, args)
```

伪代码：

```text
for depth in range(min_depth, ref_bit + 1):
    if remaining_bound(depth) <= tolerance:
        return depth
return ref_bit
```

这里体现了 predictor-free 的核心：没有训练预测器，只根据“剩余低位 bit-plane 的最大可能影响”决定是否继续。

### 3.4 映射到可用 embedding pool

函数：

```text
nearest_available_precision_depth(requested_depth, bits, ref_bit)
```

因为 accuracy validation 只能使用已经生成好的 pools，例如：

```text
ref_bit = 8
bits = [6, 5, 4]
```

如果 runtime bound 要求 `requested_depth=5`，就映射到 W4A5；如果要求 `requested_depth=7`，当前没有 W4A7，就向上映射到 W4A8。

这一步只影响软件 accuracy validation，不代表硬件只能支持这些离散 pools。硬件 bit-serial datapath 可以按 bit-plane 逐位停。

### 3.5 给节点分配 action bit

函数：

```text
select_precision_depth_actions(...)
```

它做三件事：

```text
1. 取 risk priority
   degree     -> propagation_q
   tser       -> sensitivity_q
   low_unique -> low_degree_unique_q
   random     -> random score

2. 只在 eligible nodes 上排序
   在 residual_precision_depth 里，eligible nodes 是 miss nodes。
   direct / residual 节点不进入 LLaMA encoder。

3. 按 high_ratio / mid_ratio 切分
   high bucket -> high min_depth/tolerance -> runtime depth
   mid bucket  -> mid  min_depth/tolerance -> runtime depth
   remaining nodes -> low min_depth/tolerance -> runtime depth
```

输出是每个节点的 `action_bit`：

```text
8 / 6 / 5 / 4
```

然后：

```text
assemble_precision_depth_embeddings(...)
```

用对应 embedding pool 组装最终验证特征。

代码上，`bound_budget` 会先把所有 eligible nodes 初始化为 low bucket：

```text
actions[eligible_idx] = bucket_bits["low"]["pool_bit"]
```

然后再覆盖 high 和 mid：

```text
actions[top high_count] = high pool_bit
actions[next mid_count] = mid pool_bit
```

因此即使 `LOW_RATIO=0.0`，没有被 high/mid 覆盖的 miss nodes 仍然会走 low bucket。这也是当前 quick flow 中 `HIGH_RATIO=0.20, MID_RATIO=0.50, LOW_RATIO=0.0` 仍会得到约 30% low-depth nodes 的原因。

## 5. 与 residual reuse 的结合

Graph-Bit 不是直接作用于所有节点，而是在 reuse/residual 之后只处理 miss nodes。

执行顺序是：

```text
all graph nodes
    |
    |-- hard CAM hit
    |       -> direct reuse
    |
    |-- soft CAM hit accepted by residual gate
    |       -> residual correction
    |
    |-- rejected / miss
            -> Graph-Bit encoder
```

在 `residual_precision_depth` 中：

```text
apply_residual_precision_depth_trace(...)
```

负责把 direct / residual / miss 三类路径合成最终 embedding：

```text
direct:
    使用 anchor embedding

residual:
    使用 anchor + residual adapter correction

miss:
    使用 Graph-Bit action_bit 对应的 P8/P6/P5/P4 pool
```

## 6. Per-node trace 导出

函数：

```text
export_residual_precision_depth_node_trace(...)
```

打开：

```text
--precision_depth_trace_export_dir
--precision_depth_trace_export_configs DegBound
```

后，会导出 JSONL trace：

```text
第一行:
    metadata

后续每行:
    一个 graph node 的 routing / depth 信息
```

每个节点主要字段：

```text
node_id
role:
    direct / residual / miss

is_miss:
    只有 true 的节点进入 Graph-Bit NPU

source_id:
    direct / residual 复用的 anchor

best_dist:
    CAM Hamming distance

route_hits / base_hits / support_hits:
    多 head support 信息

action_bit:
    软件 validation 中实际使用的 bit depth

depth_bucket:
    p8 / p6 / p5 / p4

stop_depth:
    NPU trace replay 使用的 stop depth

degree_q / tser_q / low_unique_q:
    各类 risk score
```

这个 trace 是后续 trace-driven scheduler replay 的输入。

## 7. ONNXim 里的 datapath 实现

ONNXim 侧的核心文件：

```text
GraphhopSimhash/ONNXim/src/SimulationConfig.h
GraphhopSimhash/ONNXim/src/operations/GemmWS.cc
GraphhopSimhash/ONNXim/src/Common.h
GraphhopSimhash/ONNXim/src/SystolicWS.cc
GraphhopSimhash/scripts/onnxim_graphbit_microbench.py
```

### 6.1 SimulationConfig 参数

`SimulationConfig.h` 中新增 Graph-Bit 参数：

```text
graphbit_enable
graphbit_bound_enable
graphbit_full_depth
graphbit_precision_depth
graphbit_min_depth
graphbit_bound_tolerance
graphbit_bound_scale
graphbit_bound_mode
graphbit_issue_gate
graphbit_psum_gate
graphbit_risk_bucket_enable
graphbit_weight_stationary_enable
graphbit_baseline_weight_tile_batch
graphbit_weight_stationary_tile_batch
```

这些参数由：

```text
scripts/onnxim_graphbit_microbench.py
```

写入 ONNXim JSON config。

### 6.2 ONNXim bound estimator

`GemmWS.cc` 中的：

```text
estimate_graphbit_remaining_bound(depth, tile_k, config)
```

支持两类 bound：

```text
range:
    只看低位 bit range 和 tile_k。
    对应 Python validation 里的简化公式。

tile_mean / tile_max:
    使用硬件可见 tile metadata 估计 ||A_low @ W|| 上界。
```

tile-aware bound 的核心形式是：

```text
remaining_bound ~= max_abs(A_low) * sum_abs(W_tile)
normalized_bound = remaining_bound / (partial_norm + remaining_bound)
```

ONNXim 当前不是用真实 runtime activation/weight 值逐元素计算 bound，而是用配置里的 tile 统计参数近似：

```text
graphbit_bound_weight_abs_mean
graphbit_bound_weight_abs_max
graphbit_bound_partial_norm_scale
graphbit_bound_safety_factor
```

这使得 ONNXim component simulation 可以评估 datapath 行为，但不是 full LLaMA runtime value trace。

### 6.3 选择 effective depth

`GemmWS.cc` 中的：

```text
select_graphbit_effective_depth(tile_k, config)
```

逻辑和 Python 侧一致：

```text
config_depth = graphbit_precision_depth
min_depth    = graphbit_min_depth

for depth in min_depth..config_depth:
    if estimate_bound(depth) <= graphbit_bound_tolerance:
        return depth
return config_depth
```

输出写入 instruction：

```text
inst.graphbit_effective_depth
```

### 6.4 effective / issue / psum 三个 depth

一个 instruction 会被标注多个 depth：

```text
effective_depth:
    runtime bound 得到的实际 stop depth。

issue_depth:
    PE array 实际发射的 bit-plane cycles。

psum_depth:
    partial-sum read/update/write 在多少 bit-plane cycle 中发生。
```

对应函数：

```text
graphbit_issue_depth(...)
graphbit_psum_depth(...)
```

#### bit-plane issue

如果开启：

```text
graphbit_issue_gate = true
graphbit_risk_bucket_enable = true
```

则：

```text
issue_depth = effective_depth
```

否则：

```text
issue_depth = 8
```

这表示如果没有 risk bucket，mixed batch 可能被高风险节点拖到 full depth。

#### psum update

如果开启：

```text
graphbit_psum_gate = true
```

则：

```text
psum_depth = issue_depth
```

含义是跳过低位 bit-plane 后，不再进行对应的 partial-sum read/add/write。

### 6.5 weight-stationary HBM 缩放

HBM weight traffic 的缩放来自：

```text
graphbit_weight_hbm_scale(...)
```

公式：

```text
batch_scale = baseline_weight_tile_batch / weight_stationary_tile_batch
weight_hbm_scale = batch_scale * graphbit_weight_memory_scale
```

这一项表示：

```text
同一个 W tile 在片上服务更多同风险 node blocks，
所以 HBM 读入 W tile 的次数减少。
```

它只有在：

```text
graphbit_weight_stationary_enable = true
```

时生效。

更完整的 full workload 里，不直接手动套这个比例，而是由 trace-driven scheduler replay 统计真实 `Wloads / Wscale`。

## 8. ONNXim 统计项

`Common.h` 给每条 instruction 增加 Graph-Bit 字段：

```text
graphbit_effective_depth
graphbit_issue_depth
graphbit_psum_depth
graphbit_remaining_bound
graphbit_weight_hbm_scale
```

`SystolicWS.cc` 在 issue instruction 时统计：

```text
GraphBit Inst
BoundStops
AvgDepth
AvgSavedBitplanes
AvgIssueDepth
AvgPsumDepth
EffectiveDepthHist
IssueDepthHist
RawComputeCycles
EffectiveComputeCycles
```

这些字段会被：

```text
scripts/onnxim_graphbit_microbench.py
```

从 ONNXim log 里解析出来，写入：

```text
summary.tsv
aggregate.json
```

## 9. Trace-driven full-stack replay

脚本：

```text
GraphhopSimhash/scripts/replay_graphbit_trace_scheduler.py
```

输入：

```text
1. GraphhopSimhash 导出的 per-node trace
2. ONNXim component lookup
```

它会重放几种策略：

```text
FullP8-miss:
    所有 miss nodes 都按 D8 执行。

GraphBit-now:
    使用真实 stop_depth，但不扩大 W tile service window。

OriginalOrder-b32 / b64:
    按原始节点顺序组成 batch。
    一个 batch 的执行 depth = batch 内 max stop depth。

RiskBucket-b32 / b64:
    按 stop_depth 分桶后组成 batch。
    每个 bucket 内 depth 一致，保留 D5/D6/D8 分布。
```

trace replay 输出：

```text
Wloads:
    重放过程中实际需要加载多少次 W tile。

Wscale:
    Wloads / FullP8-miss 的 baseline Wloads。

AvgD:
    miss nodes 的平均 stop depth。

Cycles / Traffic / Energy:
    用 ONNXim component lookup 按真实 depth histogram 和 scheduler replay 组合得到。
```

## 10. 一个 Cora h8_54_T40 例子

常用快速命令：

```bash
RUNS=1 \
RUN_ALGO=1 \
RUN_ONNXIM=0 \
DATASET=cora \
THRESHOLD=40 \
HARD_SUPPORT=5 \
SOFT_SUPPORT=4 \
FRONTEND_ID=h8_54_T40 \
BUDGET=boundclean \
HIGH_RATIO=0.20 \
MID_RATIO=0.50 \
LOW_RATIO=0.0 \
OUT_DIR=output/graphbit_trace_replay/cora_h8_54_T40_boundclean_quick \
TRACE_EXPORT=1 \
TRACE_EXPORT_CONFIGS='DegBound' \
BOUND_ENABLE=1 \
BOUND_PRIORITIES='degree' \
bash GraphhopSimhash/scripts/run_graphbit_predictor_free_flow.sh
```

典型 log 会出现：

```text
[GraphBitBound]
priorities=['degree'] | tile_k=128 | scale=1.000 |
high:min=8,tau=0.0000,runtime=P8,pool=P8,bound=0.00000 |
mid:min=6,tau=0.0200,runtime=P6,pool=P6,bound=0.01176 |
low:min=4,tau=0.0400,runtime=P5,pool=P5,bound=0.02745
```

含义：

```text
high bucket:
    Degree 最高的一部分 miss nodes，完整 P8。

mid bucket:
    中等 Degree miss nodes，bound 在 P6 已满足 tolerance。

low bucket:
    低 Degree miss nodes，P4 bound 不够小，继续算到 P5 后停止。
```

然后 trace replay：

```bash
bash GraphhopSimhash/scripts/run_graphbit_trace_replay.sh
```

会生成：

```text
output/graphbit_trace_replay/.../node_traces/*.jsonl
output/graphbit_trace_replay/.../replay/*_trace_replay.txt
output/graphbit_trace_replay/.../replay/*_component_lookup.tsv
```

其中：

```text
*_trace_replay.txt:
    full-stack scheduler replay 主表。

*_component_lookup.tsv:
    ONNXim projection / FFN component cost lookup。
```

## 11. 当前实现边界

已经实现：

```text
1. GraphhopSimhash:
   graph risk -> min_depth/tolerance -> runtime stop depth -> action_bit。

2. residual_precision_depth:
   direct / residual / miss 三路径合成。

3. per-node trace:
   导出每个节点的 role、support、risk score、action_bit、stop_depth。

4. ONNXim GemmWS:
   component-level bit-plane effective depth、fetch depth、issue depth、weight/RF depth、psum depth。

5. trace replay:
   用真实 miss-node trace 重放 risk bucket scheduler，并结合 ONNXim component lookup。
```

仍然是模型化/近似的部分：

```text
1. accuracy validation 用 P8/P6/P5/P4 embedding pools 近似 bit-serial stop depth 的输出。

2. ONNXim bound 的 tile_mean / tile_max 模式使用配置化 tile statistics，
   不是读取真实 LLaMA runtime activation / weight value trace。

3. trace replay 是 workload-level scheduler replay，
   不是 full LLaMA 所有层、所有 tile 的完整 cycle-accurate event simulation。

4. SRAM/RF 能耗目前通过 traffic / depth proxy 表达，
   不是电路级 energy model。
```

因此当前证据链的准确定位是：

```text
GraphhopSimhash accuracy validation
    + ONNXim component-level bit-plane datapath simulation
    + trace-driven workload scheduler replay
```

这已经能回答：

```text
1. 哪些节点进入 encoder？
2. miss nodes 算到几 bit？
3. activation fetch / issue / psum 是否随 stop depth 变化？
4. risk-bucket scheduler 是否在真实节点 trace 上形成更大的 W tile service window？
```

如果后续继续增强，下一步是把 trace replay 下沉到 per-layer / per-tile event trace，让每个真实 miss-node batch 逐层调用 ONNXim component model。
