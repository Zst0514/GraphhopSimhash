# reuse_real_quant 联合实验命令说明

这份文档解释下面这条命令的含义：

```bash
cd /home/qiumingzhi/Simhash-S/OneForAll

python -m GraphhopSimhash \
  --datasets cora \
  --runs 1 \
  --experiment_suite reuse_real_quant \
  --real_quant_policy_suite w4a8_budget \
  --real_quant_model_name ST \
  --real_quant_fp_tag W4A16 \
  --real_quant_int8_tag W4A8 \
  --real_quant_int4_tag W4A4 \
  --real_quant_fp_ratio 0.0 \
  --real_quant_int8_ratio 0.20 \
  --real_quant_error_norm 1.0 \
  --learned_hash_epochs 10 \
  --learned_hash_dim 128 \
  --hamming_only_acceptor \
  --enable_score_gate \
  --main_hash_head_bits 16 16 16 16 16 16 16 16 \
  --route_min_support_hits 3
```

注意：`python -m GraphhopSimhash` 要在 `OneForAll` 目录下执行，因为 `GraphhopSimhash` 是这个目录下面的 Python 包名。包名中的 `hop` 是小写，不是 `GraphHopSimhash`。

## 实验整体在做什么

这条命令跑的是 `hash reuse + 真实 W4A4/W4A8 feature pool` 联合实验。

通俗讲，它把节点分成两类：

```text
hash 命中的节点：直接复用 cache 里的 embedding，不重新计算
hash 没命中的节点：从真实 W4A16/W4A8/W4A4 feature pool 里选一种精度来计算
```

所以这个实验同时回答两个问题：

```text
1. hash reuse 能省掉多少节点的计算？
2. 剩下不能 reuse 的节点，如果用 W4A8/W4A4，会损失多少精度、节省多少成本？
```

它不是单纯量化实验，也不是单纯 hash reuse 实验，而是把两者串在同一条执行链里。

## 执行流程

一次运行大致分为 5 步：

1. 用 `W4A16` feature pool 训练/评估高精度 baseline。
2. 用 learned hash projection 构建多头 hash 检索表。
3. 对每个节点尝试 hash reuse。
4. 如果节点 hash hit，就直接复用缓存 embedding。
5. 如果节点 hash miss，就按照真实量化策略，从 `W4A16/W4A8/W4A4` feature pool 里选 embedding。

最终输出的表会同时包含：

```text
Reuse %
W4A4 %
W4A8 %
FP %
Cost
Acc
Drop %
FinalErr
Reuse n/d
```

其中 `W4A4/W4A8/FP` 百分比都是相对于全图节点数统计的。

## 量化相关参数

### `--experiment_suite reuse_real_quant`

指定跑联合实验。

这里的联合实验含义是：

```text
reuse hit 节点：直接复用，不计 W4A4/W4A8/FP 计算成本
reuse miss 节点：再做真实 feature pool 量化选择
```

如果改成 `real_quant_ablation`，那就是单独的真实量化池实验，不包含 hash reuse。

### `--real_quant_policy_suite w4a8_budget`

指定使用 W4A4/W4A8 固定预算对比策略。

它会自动跑多行配置：

| 配置名 | 通俗含义 |
|---|---|
| `AllFP` | hash miss 节点全部用高精度 `W4A16` |
| `UniformW4A8` | hash miss 节点全部用 `W4A8` |
| `UniformW4A4` | hash miss 节点全部用 `W4A4` |
| `RandomTopK_W4A8` | hash miss 节点里随机挑一部分用 `W4A8`，剩下用 `W4A4` |
| `DegreeTopK_W4A8` | hash miss 节点里度数/传播影响大的节点用 `W4A8`，剩下用 `W4A4` |
| `TSERTopK_W4A8` | hash miss 节点里 TSER 风险高的节点用 `W4A8`，剩下用 `W4A4` |
| `QuantTSERTopK_W4A8` | 用更偏量化风险的 TSER 变体选择 `W4A8` 节点 |
| `DegreeErrorTopK_W4A8` | 用 degree 重要性乘真实 W4A4 误差排序，偏 oracle 上界 |
| `TSERErrorTopK_W4A8` | 用 TSER 重要性乘真实 W4A4 误差排序，偏 oracle 上界 |

`DegreeErrorTopK_W4A8` 和 `TSERErrorTopK_W4A8` 使用了真实 `W4A16` 与 `W4A4` 的 embedding 误差，因此更像上界实验。真实部署时不能提前知道所有节点的真实 W4A4 误差，除非用校准集估计。

### `--real_quant_model_name ST`

指定使用哪一套真实量化 feature pool。

这里的 `ST` 会让程序查找：

```text
cache_data/cora_ST_oracle_W4A16.pt
cache_data/cora_ST_oracle_W4A8.pt
cache_data/cora_ST_oracle_W4A4.pt
```

如果换成 `llama2_7b`，程序会查找：

```text
cache_data/cora_llama2_7b_oracle_W4A16.pt
cache_data/cora_llama2_7b_oracle_W4A8.pt
cache_data/cora_llama2_7b_oracle_W4A4.pt
```

所以这个参数决定“用哪种模型生成出来的真实量化 embedding 池”。

### `--real_quant_fp_tag W4A16`

指定高精度参考池的 tag。

在当前命令中，它对应：

```text
cache_data/cora_ST_oracle_W4A16.pt
```

这里 `W4A16` 被当作 FP/高精度参考。输出表里的 `FP %` 就是使用这个池子的节点比例。

### `--real_quant_int8_tag W4A8`

指定 8-bit activation 路径的 tag。

在当前命令中，它对应：

```text
cache_data/cora_ST_oracle_W4A8.pt
```

通俗理解：

```text
W4A8 = weight 4 bit + activation 8 bit
```

它通常比 `W4A4` 稳，但成本也更高。

### `--real_quant_int4_tag W4A4`

指定 4-bit activation 路径的 tag。

在当前命令中，它对应：

```text
cache_data/cora_ST_oracle_W4A4.pt
```

通俗理解：

```text
W4A4 = weight 4 bit + activation 4 bit
```

它成本最低，但误差通常最大。

### `--real_quant_fp_ratio 0.0`

指定在 TopK / cascade 类策略里，hash miss 节点中保留 FP/W4A16 的比例。

当前设置是：

```text
0.0
```

意思是：

```text
hash miss 节点里不额外保留高精度 FP/W4A16
```

因此 TopK 类策略只在 `W4A8` 和 `W4A4` 之间分配。

如果改成：

```bash
--real_quant_fp_ratio 0.05
```

就表示：

```text
hash miss 节点里最重要的 5% 保持 W4A16/FP
```

### `--real_quant_int8_ratio 0.20`

这是最关键的量化预算参数。

在 `reuse_real_quant` 里，它表示：

```text
hash miss 节点中，有多少比例分配给 W4A8
```

当前设置是：

```text
0.20
```

意思是：

```text
miss 节点里 20% 用 W4A8
miss 节点里 80% 用 W4A4
```

注意，它不是全图 20%，而是 miss 节点中的 20%。

举例，如果输出里：

```text
Reuse = 53.5%
Miss = 46.5%
```

那么 TopK 类策略里：

```text
全图 W4A8 比例 = 46.5% * 20% = 9.3%
全图 W4A4 比例 = 46.5% * 80% = 37.2%
```

所以你会看到类似：

```text
DegreeTopK_W4A8 | W4A4=37.1% | W4A8=9.3%
```

如果你希望全图大约 20% 节点走 W4A8，而 miss 约为 46.5%，应该设置：

```text
0.20 / 0.465 ≈ 0.43
```

也就是：

```bash
--real_quant_int8_ratio 0.43
```

### `--real_quant_error_norm 1.0`

这个参数用于把真实量化误差归一化成 0 到 15 的离散等级。

代码逻辑可以通俗理解成：

```text
cosine_error = 1 - cos(fp_embedding, quant_embedding)
error_q = round(clamp(cosine_error / real_quant_error_norm, 0, 1) * 15)
```

当前设置：

```text
1.0
```

表示：

```text
cosine error = 1.0 时映射到最高错误等级 15
```

如果把它调小，比如：

```bash
--real_quant_error_norm 0.20
```

同样的误差会被放大。例如：

```text
error = 0.10, norm = 1.0  -> error_q 约为 2
error = 0.10, norm = 0.2  -> error_q 约为 8
```

这个参数主要影响 error-aware 策略，例如：

```text
DegreeErrorTopK_W4A8
TSERErrorTopK_W4A8
```

对 `UniformW4A8`、`UniformW4A4` 这种固定策略影响不大。

## Hash reuse 相关参数

### `--learned_hash_epochs 10`

训练 learned hash projection 的 epoch 数。

它影响 hash 表质量。epoch 越多，hash projection 可能更稳定，但运行更慢。

### `--learned_hash_dim 128`

learned hash projection 的中间维度。

这里使用 128 维，和前面实验保持一致。

### `--hamming_only_acceptor`

候选复用时只使用 Hamming 距离和多头支持信息，不再用 cheap-feature cosine 做二次排序接受。

通俗讲：

```text
复用判断更依赖 hash 本身和多头一致性
```

### `--enable_score_gate`

开启 score gate。

score gate 会根据节点风险过滤一部分复用候选，主要保护：

```text
高传播影响节点
低度稀有节点
上下文风险高的节点
```

它通常会降低 reuse 率，但能减少错误复用。

### `--main_hash_head_bits 16 16 16 16 16 16 16 16`

主 hash route 使用 8 个 hash head，每个 head 是 16 bit。

通俗讲：

```text
同一个节点会被 8 张 hash 表同时检索
```

多头 hash 的好处是候选更稳，可以要求多个 head 同时支持同一个候选。

### `--route_min_support_hits 3`

要求同一个复用候选至少被 3 个 hash head 支持。

在当前 8 头设置下，它就是：

```text
8 个 hash head 里，至少 3 个 head 同意这个候选，才允许复用
```

这个参数会降低误复用风险，但也可能减少 reuse。

## 输出表怎么读

典型输出如下：

```text
Config                     | Reuse %   | W4A4 %   | W4A8 %   | FP %     | Cost     | Acc        | Drop %     | FinalErr   | Reuse n/d
AllFP                      | 53.5%     | 0.0%     | 0.0%     | 46.5%    | 0.465    | 0.6030     | 3.38%      | 0.10545    | 1450/2708
UniformW4A8                | 53.5%     | 0.0%     | 46.5%    | 0.0%     | 0.232    | 0.6122     | 2.47%      | 0.12778    | 1450/2708
UniformW4A4                | 53.5%     | 46.5%    | 0.0%     | 0.0%     | 0.116    | 0.5493     | 8.75%      | 0.18632    | 1450/2708
DegreeTopK_W4A8            | 53.5%     | 37.1%    | 9.3%     | 0.0%     | 0.139    | 0.5779     | 5.90%      | 0.17359    | 1450/2708
```

各列含义如下：

| 列名 | 含义 |
|---|---|
| `Reuse %` | 直接复用 cache 的节点比例 |
| `W4A4 %` | 实际走 W4A4 计算的节点比例 |
| `W4A8 %` | 实际走 W4A8 计算的节点比例 |
| `FP %` | 实际走 W4A16/FP 计算的节点比例 |
| `Cost` | 相对计算成本 |
| `Acc` | 当前策略最终准确率 |
| `Drop %` | 相对 `Baseline Acc` 的准确率下降 |
| `FinalErr` | 最终 embedding 与 FP embedding 的平均 cosine error |
| `Reuse n/d` | 复用节点数 / 总节点数 |

## Cost 怎么算

当前实验使用的相对成本约定是：

```text
reuse cache read = 0
W4A4 = 0.25
W4A8 = 0.50
W4A16/FP = 1.00
```

所以：

```text
Cost = FP% * 1.00 + W4A8% * 0.50 + W4A4% * 0.25
```

例如：

```text
AllFP:
Cost = 46.5% * 1.00 = 0.465

UniformW4A8:
Cost = 46.5% * 0.50 = 0.2325

UniformW4A4:
Cost = 46.5% * 0.25 = 0.11625

DegreeTopK_W4A8:
Cost = 37.1% * 0.25 + 9.3% * 0.50
     ≈ 0.139
```

因此，`reuse_real_quant` 的成本含义是：

```text
hash hit 节点不重新计算，只对 hash miss 节点计 W4A4/W4A8/FP 成本
```

## AllFP 行为什么也会掉点

`AllFP` 不是 baseline 本身。

它的含义是：

```text
hash hit 节点复用 cache
hash miss 节点使用 W4A16/FP
```

所以 `AllFP` 仍然包含 hash reuse 带来的近似误差。它不包含 W4A4/W4A8 量化误差，但包含复用误差。

如果 `AllFP` 掉点较大，说明当前 seed 下 hash reuse 本身造成了一些错误复用。

## 和 real_quant_ablation 的区别

`reuse_real_quant`：

```text
先 hash reuse，miss 节点再做真实量化选择
```

`real_quant_ablation`：

```text
不做 hash reuse，直接在全图节点上比较 W4A16/W4A8/W4A4 策略
```

如果只想看 W4A4 本身会让特征掉多少，可以跑：

```bash
python -m GraphhopSimhash \
  --datasets cora \
  --runs 1 \
  --experiment_suite real_quant_ablation \
  --real_quant_policy_suite w4a8_budget \
  --real_quant_model_name ST \
  --real_quant_fp_tag W4A16 \
  --real_quant_int8_tag W4A8 \
  --real_quant_int4_tag W4A4 \
  --real_quant_fp_ratio 0.0 \
  --real_quant_int8_ratio 0.20 \
  --real_quant_error_norm 1.0
```

如果想看“hash reuse 以后，剩下 miss 节点再量化”的端到端效果，就跑本文档这条 `reuse_real_quant` 命令。

## 常见注意事项

1. `GraphhopSimhash` 包名大小写必须正确，不能写成 `GraphHopSimhash`。
2. 每一行反斜杠 `\` 后面不要有多余空格，否则 shell 可能会把下一行当成新命令。
3. `--real_quant_int8_ratio 0.20` 是 miss 节点中的 20%，不是全图节点中的 20%。
4. `DegreeErrorTopK_W4A8` 和 `TSERErrorTopK_W4A8` 使用真实 W4A4 误差，是 oracle 上界，不是严格可部署策略。
5. `--runs 1` 默认只跑 seed 42。不同 seed 的 `AllFP` 掉点可能不同，建议最终表格用 `--runs 3`。

