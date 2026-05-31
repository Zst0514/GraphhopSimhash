# ST 与 Llama 共享检索骨架的残差复用结果

日期：2026-05-30

本文档记录一次针对 `ST full/HQ cache (data.x)` 与 `Llama2-7B W4A16` 的联合调参结果。

目标是：

```text
Cora 和 PubMed
ST 与 Llama2-7B W4A16
ResidualReuse 复用率 >= 40%
精度掉点 < 2%
```

## 结论

当前结果说明：**原来的在线参数确实不完全适合 Llama**。  
尤其是 Llama/Cora 需要更宽的在线 score 阈值，而 Llama/PubMed 需要更严格的 residual accept 阈值。

比较稳妥的做法是：

```text
共享在线检索骨架
+ 按数据源/数据集离线校准 residual gate
```

也就是说，哈希检索、support split、score gate 的主线结构可以统一；但 residual 路径里的 raw accept threshold 和 gate 训练方式不应该强行完全相同。

## 共享在线检索骨架

四组实验共用以下在线检索骨架：

```text
8 个 head
每个 head 16 bit
radius = 2
关闭结构检查
score gate 打开
score 权重 = 3 / 1 / 1
score reuse threshold T = 31

support >= 5   -> hard direct reuse
support = 3..4 -> residual candidate
support < 3    -> compute
```

对应关键参数：

```bash
--hash_heads_per_route 8
--main_hash_head_bits 16 16 16 16 16 16 16 16
--radius 2
--disable_structure_check
--enable_score_gate
--allow_rare_fuzzy
--score_reuse_threshold 31
--score_propagation_weight 3
--score_graph_context_weight 1
--score_low_unique_weight 1
--score_pair_confidence_discount 1
--residual_hard_min_support_hits 5
--residual_soft_min_support_hits 3
```

## 3-run 结果

| Embedding 源 | 数据集 | Baseline Acc | ResidualReuse | Acc | Drop | 关键 gate 设置 |
|---|---|---:|---:|---:|---:|---|
| ST full/HQ cache (data.x) | Cora | 0.7200 | 48.4% | 0.7079 | 1.21% | separate, tau=0.575 |
| ST full/HQ cache (data.x) | PubMed | 0.7587 | 40.4% | 0.7404 | 1.84% | shared, tau=0.65 |
| Llama2-7B W4A16 | Cora | 0.7308 | 40.8% | 0.7132 | 1.76% | classifier-aware separate, tau=0.40 |
| Llama2-7B W4A16 | PubMed | 0.7000 | 40.8% | 0.6819 | 1.81% | shared, tau=0.91 |

这里的 `data.x` 不是 cheap feature，也不是 DistilBERT Layer-1。
在 ST 实验线中，`data.x` 来自 `cache_data/{Cora,Pubmed}/ST/processed/geometric_data_processed.pt`，也就是缓存好的 ST full/HQ 节点特征；`data.x` 只是该 full embedding 在 PyG `Data` 对象里的存放位置。

对应日志形式为：

```text
[ResidualTarget] source=data_x | path=<data.x> | shape=(2708, 384)
[ResidualTarget] source=data_x | path=<data.x> | shape=(19717, 384)
```

Llama2-7B W4A16 则不是从 `data.x` 读取，而是通过 `real_quant_fp` 加载独立的 Llama embedding pool。

## Llama 需要调整的原因

原 ST 配置在 Llama 上不能直接复用，主要原因是 Llama embedding 的“向量近邻”不一定等价于“分类安全复用”。  
在 Llama/Cora 上，原配置会把复用推高，但会放进一批对 GNN 分类有害的 soft hit。

因此新增了分类感知的 accept gate 训练：

```text
离线阶段：
    在 train/val 上模拟 residual reuse
    送入已经训练好的 GNN 分类器
    检查预测是否保持稳定
    检查与 full-compute logits 的 KL 是否足够小
    用该结果监督 accept gate

在线阶段：
    不跑分类器
    不使用标签
    只计算 accept gate 分数并与阈值比较
```

Llama/Cora 使用的新增关键参数：

```bash
--residual_classifier_accept_gate
--residual_classifier_accept_mode both
--residual_classifier_accept_max_kl 0.2
--residual_classifier_accept_after_residual
--residual_classifier_accept_probe_alpha 0.125
```

当前实现里 classifier-aware accept gate 的监督信号来自冻结 GNN：

```text
reference embedding -> frozen GNN -> reference logits
candidate reuse embedding -> frozen GNN -> candidate logits

accept = prediction preserved AND KL(reference || candidate) <= max_kl
```

`--residual_classifier_accept_after_residual` 表示先用一个 probe residual adapter 生成候选修正结果，再用该结果构造 accept gate 的离线监督标签。在线阶段不跑 GNN、不读标签，只使用 residual adapter 的 accept score 和 `--residual_gate_accept_threshold`。

## 参数解释

这组结果不是“所有数字阈值完全相同”。  
更准确地说：

```text
共享的是在线检索骨架：
    8 heads / 16 bit / radius 2 / score T=31 / hard>=5 / soft=3..4

不共享的是 residual gate 校准：
    ST Cora      -> separate gate，较宽松
    ST PubMed    -> shared gate，略保守
    Llama Cora   -> classifier-aware separate gate
    Llama PubMed -> shared gate，高 tau
```

这符合硬件主线设计：在线硬件仍然只需要完成哈希检索、候选聚合、score gate、support split 和一个 learned accept gate 比较；不同 embedding 源的差异由离线 gate 训练和阈值校准吸收。

## 复现实验日志

本轮关键 3-run 日志：

```text
/tmp/st_T31_3run_20260530/cora.log
/tmp/st_pubmed_T31_tau_refine_3run_20260530/tau065.log
/tmp/llama_cora_T31_tau_relax_3run_20260530/tau040.log
/tmp/llama_T31_final_3run_20260530/pubmed.log
```

## 当前 main 分支复现入口

已提供统一脚本：

```bash
bash GraphhopSimhash/scripts/run_t31_shared_frontend_reuse.sh
```

只跑其中一组：

```bash
CASES="llama_cora" RUNS=3 \
bash GraphhopSimhash/scripts/run_t31_shared_frontend_reuse.sh
```

四组 case 对应关系：

| Case | Embedding 源 | 数据集 | gate 设置 |
|---|---|---|---|
| `st_cora` | `data.x` | Cora | `separate`, tau=0.575 |
| `st_pubmed` | `data.x` | PubMed | `shared`, tau=0.65 |
| `llama_cora` | `llama2_7b:W4A16` | Cora | classifier-aware `separate`, tau=0.40 |
| `llama_pubmed` | `llama2_7b:W4A16` | PubMed | `shared`, tau=0.91 |

输出目录：

```text
output/t31_shared_frontend_reuse/logs/
```

## 接入 Graph-Bit 全栈实验

后续 Graph-Bit full-stack 默认使用同一套在线前端：

```text
8 heads x 16 bit
radius = 2
T = 31
hard direct: support >= 5
residual candidate: support = 3..4
compute / Graph-Bit miss: support < 3 或 residual accept reject
```

入口脚本：

```bash
RUNS=10 DATASET=cora \
bash GraphhopSimhash/scripts/run_graphbit_predictor_free_flow.sh
```

其中 Cora 默认启用 classifier-aware accept gate；PubMed 默认使用 shared accept gate 高阈值：

```bash
RUNS=3 DATASET=pubmed \
bash GraphhopSimhash/scripts/run_graphbit_predictor_free_flow.sh
```
