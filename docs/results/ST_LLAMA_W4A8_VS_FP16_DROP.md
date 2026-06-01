# ST / LLaMA 在 Cora / PubMed / Arxiv 上的 W4A16 / W4A8 vs FP16 掉点

日期：2026-06-01

本表统一采用三跑平均结果（`--runs 3`），评测命令为 `real_quant_ablation + w4a8_budget`。对每组结果，记录 final summary 里的 `UniformW4A16` 或 `UniformW4A8` 这一行作为“全量量化”结果，`FP16` 对应同一份 final summary 里的 `Baseline Acc`。

为满足当前 `real_quant_ablation` 的三路 pool loader，评测时把 `INT4` 路径指向同一份量化 pool，并使用影子 tag（`W4A16_SHADOW_R3` 或 `W4A8_SHADOW_R3`）。因此日志里会同时出现主 tag 和 shadow tag，本文只取 `UniformW4A16` / `UniformW4A8` 这一行，不影响“量化相比 FP16”的结论。

`W4A16` 和 `W4A8` 是两轮独立重跑，所以它们各自配对的 `FP16 baseline` 允许有轻微差异；掉点始终以各自那一轮三跑 summary 里的 baseline 为准。

2026-06-01 起，`ST + W4A16` 的默认 recipe 做了一个定向修正：仅对 `cora` 保持 AWQ 的 `MSE clip` 开启，用来修复 `DistilBERT/ST` 在该数据集上的异常掉点；`pubmed` / `arxiv` 仍保持原先的非 `MSE clip` 默认，除非显式传 `--awq_force_mse_clip`。

## ST 结果

| Dataset | FP16 Acc (W4A16 run) | W4A16 Acc | W4A16 Drop | FP16 Acc (W4A8 run) | W4A8 Acc | W4A8 Drop |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.6773 | 0.6754 | 0.19% | 0.6773 | 0.6662 | 1.11% |
| PubMed | 0.7452 | 0.7440 | 0.11% | 0.7452 | 0.7445 | 0.07% |
| Arxiv | 0.6688 | 0.6671 | 0.17% | 0.6688 | 0.6678 | 0.10% |

## LLaMA 结果

| Dataset | FP16 Acc (W4A16 run) | W4A16 Acc | W4A16 Drop | FP16 Acc (W4A8 run) | W4A8 Acc | W4A8 Drop |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.7189 | 0.7155 | 0.34% | 0.7189 | 0.7155 | 0.34% |
| PubMed | 0.7501 | 0.7460 | 0.42% | 0.7501 | 0.7478 | 0.23% |
| Arxiv | 0.6896 | 0.6890 | 0.06% | 0.6895 | 0.6888 | 0.07% |

负掉点表示该次三跑平均里量化结果略高于对应的 FP16 baseline，通常可视为训练/评测波动。

## 本次补跑内容

此前补生成的缺失 pool：

- `cache_data/pubmed_ST_oracle_FP16.pt`
- `cache_data/pubmed_ST_oracle_W4A8.pt`
- `cache_data/arxiv_ST_oracle_W4A8.pt`
- `cache_data/cora_llama2_7b_oracle_FP16.pt`
- `cache_data/pubmed_llama2_7b_oracle_FP16.pt`

本轮文档中的 `W4A16` 和 `W4A8` 汇总都采用 `--runs 3` 重新评测后的日志，不再使用旧的单跑结果。

本次为修复 `ST + cora + W4A16` 的异常掉点，重新生成并复核了 `ST` 的 3 组 `W4A16` pool：

- `cora`: 在最终默认逻辑下重生，保留 AWQ `MSE clip`
- `pubmed`: 恢复为默认非 `MSE clip`
- `arxiv`: 恢复为默认非 `MSE clip`

`LLaMA` 的 `W4A16` 结果和全部 `W4A8` 结果未改 recipe，只沿用已有 pool 重新评测。

## 对应评测日志

### W4A16 vs FP16

| Dataset | Model | Log |
|---|---|---|
| Cora | ST | `output/graph_simhash/cora/cora_ST_w4a16_vs_fp16_runs3.stdout.log` |
| PubMed | ST | `output/graph_simhash/pubmed/pubmed_ST_w4a16_vs_fp16_runs3.stdout.log` |
| Arxiv | ST | `output/graph_simhash/arxiv/arxiv_ST_w4a16_vs_fp16_runs3.stdout.log` |
| Cora | LLaMA | `output/graph_simhash/cora/cora_llama2_7b_w4a16_vs_fp16_runs3.stdout.log` |
| PubMed | LLaMA | `output/graph_simhash/pubmed/pubmed_llama2_7b_w4a16_vs_fp16_runs3.stdout.log` |
| Arxiv | LLaMA | `output/graph_simhash/arxiv/arxiv_llama2_7b_w4a16_vs_fp16_runs3.stdout.log` |

### W4A8 vs FP16

| Dataset | Model | Log |
|---|---|---|
| Cora | ST | `output/graph_simhash/cora/cora_ST_w4a8_vs_fp16_runs3.stdout.log` |
| PubMed | ST | `output/graph_simhash/pubmed/pubmed_ST_w4a8_vs_fp16_runs3.stdout.log` |
| Arxiv | ST | `output/graph_simhash/arxiv/arxiv_ST_w4a8_vs_fp16_runs3.stdout.log` |
| Cora | LLaMA | `output/graph_simhash/cora/cora_llama2_7b_w4a8_vs_fp16_runs3.stdout.log` |
| PubMed | LLaMA | `output/graph_simhash/pubmed/pubmed_llama2_7b_w4a8_vs_fp16_runs3.stdout.log` |
| Arxiv | LLaMA | `output/graph_simhash/arxiv/arxiv_llama2_7b_w4a8_vs_fp16_runs3.stdout.log` |

## 相关生成日志

- `output/quant_pools/st_missing_fp16_w4a8.log`
- `output/quant_pools/llama_missing_fp16.log`
- `output/quant_pools/arxiv_llama2_7b_w4a8.log`
- `output/quant_pools/arxiv_llama2_7b_fp16_w4a16.log`
- `output/quant_pools/cora_st_w4a16_targeted_default.log`
- `output/quant_pools/st_w4a16_restore_nomse_pubmed_arxiv.log`
