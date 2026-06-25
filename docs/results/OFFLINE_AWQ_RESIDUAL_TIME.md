# Offline AWQ and Residual Adapter Time

This note records the two offline costs that should be reported separately from
the lightweight graph-resident preprocessing table:

1. **AWQ W4 calibration/search**: one-time model compilation for a fixed
   backbone and AWQ configuration.
2. **Residual adapter fitting**: graph/task-policy calibration for the fuzzy
   embedding-reuse path.

The measurements below use LLaMA2-7B on the local RTX 4090 environment.
AWQ uses `n_samples=128`, `seqlen=512`, and `q_group_size=128`.

## 1. AWQ W4 Search Time

The AWQ column reports the official AWQ search progress-bar wall time only.
It does not include downstream embedding-pool generation.

| Task / Dataset | AWQ Search Time | Source |
|---|---:|---|
| CN / CL | 362 s | `output/offline_model_compilation_timing/awq/cora_awq_profile.log` |
| PN / PL | 490 s | `output/offline_model_compilation_timing/awq/pubmed_awq_profile.log` |
| AR | 413 s | `output/_archive_misc_20260528/arxiv_llama_w4a4/arxiv_llama2_7b_W4A4.log` |
| WK | 440 s | `output/extra_llama_bfp_pools/logs/wikics_tape_products_bfpa_8_6_5_4_3.log` |

Additional datasets used in earlier tables:

| Dataset | AWQ Search Time | Source |
|---|---:|---|
| PR / Products subset | 348 s | `output/extra_llama_bfp_pools/logs/wikics_tape_products_bfpa_8_6_5_4_3.log` |
| TA23 | 544 s | `output/final_bfp_validation_runs10/boundary/tape_arxiv23_official_pool_generation.log` |

For CN and PN, the original LLaMA2-7B AWQ logs were not present, so they were
re-measured with `scripts/profile_awq_search_time.py`. The script calls the
same `apply_official_awq_w4()` path but exits before embedding generation.

## 2. Residual Adapter Fitting Time

The residual column is the average of `ResidualFitTiming.total_s` over three
runs. It includes candidate-pair preparation, feature construction, and adapter
fit time, but not LLaMA embedding generation.

| Task / Dataset | Pair Prep. | Adapter Fit | Total |
|---|---:|---:|---:|
| CN / CL | 2.45 s | 3.92 s | 6.82 s |
| PN / PL | 16.87 s | 1.67 s | 18.55 s |
| AR | 550.00 s | 7.22 s | 723.24 s |
| WK | 4.89 s | 1.77 s | 10.71 s |

Additional datasets used in earlier tables:

| Dataset | Pair Prep. | Adapter Fit | Total |
|---|---:|---:|---:|
| PR / Products subset | 195.02 s | 7.54 s | 297.08 s |
| TA23 | 209.67 s | 6.01 s | 231.32 s |

Here `Adapter Fit` is `probe_fit_s + global_fit_s + bucket_fit_s`. The remaining
gap to `Total` is mainly candidate selection, feature-build overhead, logging,
and framework overhead. AR is dominated by pair preparation because it has far
more candidate pairs to assemble.

## Reporting Guidance

For the paper, these two costs should be reported as **offline setup /
calibration** rather than online inference time:

- AWQ is model-resident: for a fixed LLaMA2-7B and AWQ setting, the search result
  can be reused across later inference runs.
- Residual fitting is graph/policy-resident: it should be counted when deploying
  a new graph or changing the reuse policy.
- Neither cost should be mixed into per-query online latency, but both are fair
  to report in an offline-overhead table.
