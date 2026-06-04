# Graph-Aware Dynamic BFP Refinement NPU

本文档整理当前后端 NPU 的主实现方向：**Graph-aware dynamic BFP refinement**。它不是把 BFPA4/BFPA6 当成两个离线固定精度池简单切换，而是在 LLaMA encoder 的 Linear/GEMM 执行过程中，对每个 activation block 动态决定是否从 BFPA4 追加到 BFPA6。

核心机制：

```text
BFPA4 base always compute

for each activation block:
    stress = activation_block_stress(block)
    priority = graph_risk(node) * stress

    if priority >= threshold:
        execute BFPA6 refinement
    else:
        keep BFPA4
```

这条线的关键点是：普通 Transformer accelerator 最多只能看到 activation block 的数值压力；Graph-text / GFM 场景多了节点在图任务中的传播风险。因此，不是所有 high-stress block 都 refine，而是优先 refine **图任务重要节点里的 high-stress block**。

---

## 1. 系统位置

整体数据流如下：

```text
Graph text node
      |
      v
SimHash + LRU/CAM
      |
      +-- direct hit ----------> embedding cache read
      |
      +-- fuzzy hit -----------> residual-gate correction
      |
      +-- miss / reject -------> Graph-aware Dynamic BFP NPU
                                      |
                                      v
                              LLaMA encoder embedding
```

前端 reuse / residual-gate 负责减少进入 encoder 的节点数量；Graph-aware Dynamic BFP NPU 只服务剩余 miss / rejected nodes。

在线路径：

```text
direct reuse:
    no encoder execution

residual reuse:
    anchor embedding + lightweight delta

miss / reject:
    W4 + dynamic BFPA4/BFPA6 encoder path
```

因此后端 NPU 的目标不是替代 SimHash，而是降低剩余 miss nodes 的 LLaMA encoder 成本。

---

## 2. 为什么是 BFPA4 Base + BFPA6 Refinement

### 2.1 BFPA4 作为低成本底座

BFP activation 的表示形式是：

```text
x_i ~= 2^e * m_i
```

其中：

```text
e:
    一个 block 共享的 exponent

m_i:
    每个 activation value 自己的 signed mantissa
```

当前主线使用 rowwise `1 x 128` block：

```text
一个 token row 的连续 128 个 activation values 共享 exponent。
```

BFPA4 的含义是：

```text
shared exponent + 4-bit mantissa
```

它比普通 A4 更稳，因为 shared exponent 保留了 block 的动态范围；它也比 W4A8 更便宜，因为 activation-side mantissa compute 只有 4 bit。

### 2.2 BFPA6 作为 refinement 档

BFPA6 在 BFPA4 基础上多 2 个 mantissa bits：

```text
BFPA4:
    m[3:0]

BFPA6:
    m[5:0]
```

硬件上不需要两套阵列。推荐实现是：

```text
base path:
    execute BFPA4 mantissa compute

refinement path:
    conditionally execute extra 2 mantissa planes
```

也就是：

```text
BFPA6 result = BFPA4 partial sum + extra 2-bit correction partial sum
```

这种设计让阵列默认跑低成本 BFPA4，只在必要 block 上补 BFPA6。

### 2.3 为什么不是直接全 W4A8

W4A8 / BFPA8 可以作为高精度 reference 或保守 baseline，但如果 miss nodes 全部走 W4A8，后端 NPU 没有利用图任务信息。Graph-aware dynamic BFP 的目标是：

```text
低风险、低 stress block:
    保持 BFPA4

高风险、高 stress block:
    追加 BFPA6 refinement
```

这样可以把计算资源集中在对 GNN 下游更敏感的 block 上。

---

## 3. 两类风险信号

### 3.1 Graph Risk

Graph risk 是节点级信号，描述节点 embedding 误差是否容易影响 GNN 分类。

当前第一版实现使用 degree / propagation risk：

```text
degree(v)
    -> log1p(degree)
    -> robust normalize to [0, 1]
    -> graph_risk(v)
```

代码中的实现：

```text
deg = in_degree + out_degree
risk = normalize(log1p(deg))
```

直觉：

```text
high-degree node:
    embedding error 更容易通过 GNN message passing 影响邻居

low-degree node:
    error propagation 范围较小
```

后续也可以把 risk 从 degree 替换为 TSER：

```text
TSER = propagation risk + graph context risk + low-unique risk
```

但当前动态 BFP 实现先使用最轻量、最稳定的 degree risk。

### 3.2 Activation Stress

Activation stress 是 block 级信号，描述 BFP shared exponent 是否会牺牲 block 内小值精度。

BFP 的主要风险来自：

```text
一个 block 内出现 outlier
    -> shared exponent 被 outlier 拉高
    -> 小值 mantissa 右移
    -> 小值有效位减少甚至归零
```

当前实现中，对一个 activation block：

```text
max_abs    = max(abs(block))
median_abs = median(abs(block))

stress = log2(max_abs / median_abs)
stress_norm = clamp(stress / stress_scale, 0, 1)
```

其中：

```text
stress 越大:
    block 内动态范围越不均衡
    BFPA4 越可能损伤小值
```

硬件上，BFP exponent selection 本来就需要统计 block max。stress 可以复用这个路径，再加一个轻量 range / mean / median approximation。

当前软件实现使用 median，是为了验证机制；硬件实现可以使用更便宜的近似：

```text
max / mean
range bucket
zero-pressure counter
log2 bucket
```

---

## 4. Dynamic Refinement Policy

### 4.1 Block-Level Decision

对节点 `v` 的某个 activation block `b`：

```text
priority(v, b) =
    graph_risk(v) * activation_stress_norm(b)
```

然后：

```text
if priority(v, b) >= threshold:
    use BFPA6 for this block
else:
    use BFPA4 for this block
```

这不是节点级粗粒度 routing，而是 encoder 内部 block 级 refinement。

同一个节点内部可以同时出现：

```text
some blocks:
    BFPA4

some blocks:
    BFPA6
```

### 4.2 为什么用乘积

乘积表达的是“二者同时重要”：

```text
high graph risk + low activation stress:
    节点重要，但当前 block 数值上不危险，可以保持 BFPA4

low graph risk + high activation stress:
    block 数值危险，但节点下游传播影响小，不一定 refine

high graph risk + high activation stress:
    优先 refine
```

这比单独按 degree 或单独按 activation stress 更符合 GFM 场景。

### 4.3 Threshold 的含义

`threshold` 是在线 policy register，不是硬件结构变化。

```text
threshold 越低:
    refine blocks 越多
    精度更稳
    compute 更接近 BFPA6

threshold 越高:
    refine blocks 越少
    compute 更接近 BFPA4
    精度风险更高
```

当前 Cora 实验说明，`threshold=0.20` 比 `0.35` 更合理。

---

## 5. NPU Datapath

### 5.1 总体结构

```text
                       graph_risk(v)
                            |
                            v
Miss node token rows -> Activation BFP Loader -> Stress Estimator
                            |                    |
                            |                    v
                            |             priority compare
                            |                    |
                            v                    v
                    BFPA4 base mantissa     refine tag
                            |                    |
                            v                    v
                     W4 x BFPA4 PE array + optional 2-bit refinement
                            |
                            v
                       output partial sums
                            |
                            v
                     LLaMA encoder embedding
```

### 5.2 Activation BFP Loader

对每个 token row 的 activation tile：

```text
1. 切成 1 x 128 blocks
2. 统计 block exponent
3. 统计 block stress
4. 生成 BFPA4 mantissa
5. 若 priority 达标，允许后续读取/执行额外 BFPA6 mantissa bits
```

当前实现中，BFPA4/BFPA6 quantization 在 PyTorch wrapper 中完成；硬件中对应的是 BFP loader + mantissa datapath。

### 5.3 PE Array

每个 Linear/GEMM 本质是：

```text
Y = X @ W
```

其中：

```text
X:
    BFPA4/BFPA6 activation mantissa + shared exponent

W:
    AWQ W4 weight
```

PE 执行：

```text
integer partial sum:
    psum_int = sum(m_i * w_i)

scale restore:
    psum = psum_int * 2^e * weight_scale
```

对于 non-refined block：

```text
execute 4-bit mantissa path
```

对于 refined block：

```text
execute 4-bit base path
execute extra 2-bit correction path
accumulate into the same output tile
```

### 5.4 Weight Path

权重保持 W4：

```text
W:
    fixed 4-bit AWQ weights
```

Dynamic refinement 不改变权重格式，不需要为 high-risk nodes 加载 W6/W8 权重。这点和同时改变 W/A 位宽的混合精度阵列不同。

---

## 6. Hardware Cost

设 refined block 比例为 `r`。因为 BFPA6 比 BFPA4 多 2 个 mantissa bits：

```text
avg_activation_bits = 4 + 2r
```

当前 Cora `threshold=0.20`：

```text
r = 20.79%
avg_activation_bits = 4 + 2 * 0.2079 = 4.416
```

因此：

```text
vs all BFPA4:
    extra activation mantissa compute ~= 4.416 / 4 - 1
                                  ~= 10.4%

vs all BFPA6:
    activation mantissa compute ~= 4.416 / 6
                              ~= 73.6%
    saving ~= 26.4%

vs A8 / BFPA8:
    activation mantissa compute ~= 4.416 / 8
                              ~= 55.2%
```

额外硬件主要包括：

```text
1. stress estimator
   复用 BFP exponent selection 的 max path，
   增加 range / mean / median 近似统计。

2. priority comparator
   graph_risk * stress_norm >= threshold。
   可用低位定点乘法或 LUT 实现。

3. refinement control
   对 selected blocks 发射 extra 2 mantissa-plane cycles。

4. small metadata path
   node_id -> graph_risk lookup。
```

它不需要：

```text
1. 第二套 encoder array
2. W6/W8 weight storage
3. full FP activation path
4. per-node learned predictor
```

---

## 7. Software Prototype

当前实现文件：

```text
GraphhopSimhash/scripts/generate_graph_aware_bfp_dynamic_pool.py
```

主要组件：

```text
GraphAwareBFPController:
    保存 node risk
    计算 activation block stress
    决定 BFPA4 / BFPA6 block mask
    统计 refined_blocks / total_blocks

GraphAwareBFPActivationLinear:
    wrap nn.Linear
    在线 quantize activation
    调用原 Linear

replace_linear_with_graph_bfp:
    将 LLaMA 内部 Linear 替换为 Graph-aware BFP wrapper
```

当前真实执行路径：

```text
1. 加载 LLaMA-7B FP16 checkpoint
2. 应用 official AWQ W4 weight path
3. 替换 Linear 为 Graph-aware BFP activation wrapper
4. 对每个 batch 设置 node_ids
5. encoder forward 内部逐 block 动态选择 BFPA4/BFPA6
6. 保存生成的 embedding pool
7. 用 W4BFPA8_B128 reference 评估 GNN accuracy/drop
```

这一步已经进入 LLaMA forward 内部，不是简单 embedding-level 后处理。

---

## 8. Reproduction Commands

### 8.1 Cora threshold = 0.20

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/generate_graph_aware_bfp_dynamic_pool.py \
  --dataset cora \
  --threshold 0.20 \
  --stress_scale 8.0 \
  --block_size 128 \
  --base_mantissa 4 \
  --refine_mantissa 6 \
  --batch_size 4 \
  --max_length 512 \
  --runs 3 \
  --output_dir output/graphbfp_dynamic_pool \
  --overwrite
```

Output:

```text
output/graphbfp_dynamic_pool/cora/
```

### 8.2 Cora threshold = 0.35

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  GraphhopSimhash/scripts/generate_graph_aware_bfp_dynamic_pool.py \
  --dataset cora \
  --threshold 0.35 \
  --stress_scale 8.0 \
  --block_size 128 \
  --base_mantissa 4 \
  --refine_mantissa 6 \
  --batch_size 4 \
  --max_length 512 \
  --runs 3 \
  --output_dir output/graphbfp_dynamic_pool \
  --overwrite
```

---

## 9. Current Cora Result

Reference:

```text
W4BFPA8_B128
```

3-run result:

| Policy | Refined Blocks | Baseline Acc | Dynamic Acc | Dynamic Drop |
|---|---:|---:|---:|---:|
| threshold = 0.35 | 3.82% | 0.7007 | 0.6928 | 0.79% |
| threshold = 0.20 | 20.79% | 0.7007 | 0.6983 | 0.24% |

Interpretation:

```text
threshold=0.35:
    只 refine 3.82% blocks，过于保守，精度恢复有限。

threshold=0.20:
    refine 20.79% blocks，
    dynamic drop 降到 0.24%，
    平均 activation mantissa bits 约 4.416。
```

这说明 block-level `graph_risk * activation_stress` refinement 是有效的：只追加约 20.8% blocks 的 BFPA6 refinement，就能接近 BFPA8 reference 的下游表现。

---

## 10. Why This Is Graph-Specific

普通 Transformer accelerator 可以做：

```text
activation stress only
```

也就是看到某个 block 数值动态范围大，就 refine。

Graph-aware Dynamic BFP 多了一层任务语义：

```text
activation stress:
    这个 block 在 BFP 数值上是否危险？

graph risk:
    这个节点的误差是否会影响 GNN 下游？
```

因此 refinement 条件从：

```text
stress(block) > threshold
```

变成：

```text
graph_risk(node) * stress(block) > threshold
```

这个设计避免把 BFPA6 refinement 浪费在低图风险节点上，也避免只按 degree 粗粒度地把整个节点提升到 BFPA6。

---

## 11. Relation to Earlier Paths

### 11.1 不同于 node-level BFPA4/BFPA6 routing

早期 pool-level routing 是：

```text
selected node -> BFPA6 embedding pool
other node    -> BFPA4 embedding pool
```

当前 dynamic refinement 是：

```text
same node:
    some activation blocks -> BFPA4
    some activation blocks -> BFPA6
```

粒度更细，也更接近硬件执行。

### 11.2 不同于 cross-row BFP packing

cross-row BFP 尝试让多个 token rows 共享 exponent，但实验显示全层使用不稳。当前方案保留 rowwise `1 x 128`，只在 block 内做 refinement，不强制跨节点共享 exponent。

### 11.3 不同于 optional BFPA6 recovery lane 的静态节点选择

静态 recovery lane 是：

```text
高风险节点全节点 BFPA6
低风险节点全节点 BFPA4
```

当前方案是：

```text
所有节点默认 BFPA4
只有 high-priority activation blocks 追加 BFPA6
```

---

## 12. Pending Validation

当前已完成：

```text
Cora / LLaMA-7B:
    true encoder-side dynamic BFPA4/BFPA6 pool
    threshold=0.35 and threshold=0.20
```

还需要补：

```text
1. Cora threshold sweep:
       threshold = 0.10 / 0.15 / 0.20 / 0.25 / 0.30

2. PubMed validation:
       用 Cora 收敛出的 1-2 个 threshold 跑 PubMed。

3. Stress approximation ablation:
       median-based stress
       mean-based stress
       range-bucket stress
       zero-pressure stress

4. Full-stack integration:
       direct reuse / residual reuse / miss dynamic BFP encoder
       端到端 cost + drop 表。

5. Hardware cost model:
       PE cycles
       mantissa-plane activity
       BFP exponent/stress estimator overhead
       metadata lookup overhead
```

