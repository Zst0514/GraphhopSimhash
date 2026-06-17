# Frontend Reuse Tradeoff Points

This note records useful reuse/drop operating points from the current
LLaMA2-7B frontend experiments. It is intentionally a short lookup table for
paper writing; the full sweeps remain in the output logs.

## Main TSER + Residual Points

These points use the current graph-aware reuse path with TSER filtering and
residual repair. They are the most useful points for reporting the main
reuse/accuracy tradeoff.

| Dataset | Setting | Reuse | Drop | Why It Is Useful | Source |
| --- | --- | ---: | ---: | --- | --- |
| Cora | `T=28`, ResidualReuse | 32.3% | 0.97% | clean near-1% drop point | `output/llama7b_frontend_reuse_six_datasets/logs/cora_T28_runs3.log` |
| Cora | `T=31`, ResidualReuse | 39.0% | 1.81% | higher-reuse point still within 2% drop | `output/llama7b_frontend_reuse_six_datasets/logs/cora_T31_runs3.log` |
| PubMed | `T=20`, ResidualReuse | 18.6% | 0.92% | clean near-1% drop point | `output/llama7b_frontend_reuse_six_datasets/logs/pubmed_T20_runs3.log` |
| PubMed | `T=24`, ResidualReuse | 33.1% | 1.59% | higher-reuse point still within 2% drop | `output/llama7b_frontend_reuse_six_datasets/logs/pubmed_T24_runs3.log` |

## Useful Ablation Points

These are not the main policy points, but they are useful for explaining why
candidate discovery alone is insufficient and how different filters trade reuse
for accuracy.

| Dataset | Setting | Reuse | Drop | Interpretation | Source |
| --- | --- | ---: | ---: | --- | --- |
| Cora | `T=31`, DirectReuse | 15.8% | 0.77% | very safe but leaves most fuzzy candidates unused | `output/llama7b_frontend_reuse_six_datasets/logs/cora_T31_runs3.log` |
| Cora | `T=31`, SoftDirectReuse | 60.4% | 3.71% | support-only fuzzy reuse is too aggressive | `output/reuse_safety_no_tser_baseline/logs/cora_T31_runs3.log` |
| Cora | `T=31`, no-TSER ResidualReuse | 42.3% | 2.30% | more reuse than main TSER point, but exceeds 2% drop | `output/reuse_safety_no_tser_baseline/logs/cora_T31_runs3.log` |
| PubMed | `T=24`, DirectReuse | 29.9% | 1.49% | strong direct anchors alone already give a usable low-drop point | `output/llama7b_frontend_reuse_six_datasets/logs/pubmed_T24_runs3.log` |
| PubMed | `T=24`, no-TSER ResidualReuse | 26.6% | 1.14% | useful low-drop ablation point without graph-risk filtering | `output/reuse_safety_no_tser_baseline/logs/pubmed_T24_runs3.log` |
| PubMed | `T=24`, SoftDirectReuse | 90.2% | 7.70% | accepting fuzzy hits blindly is clearly unsafe | `output/reuse_safety_no_tser_baseline/logs/pubmed_T24_runs3.log` |

## Current Status

Only Cora and PubMed are complete in the six-dataset sweep at the time of this
note. The remaining datasets should be appended after `llama7b_reuse_six` and
`reuse_safety_no_tser_rest` finish.
