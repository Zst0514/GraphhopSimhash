# Reuse Safety Ablation

This table isolates the frontend reuse-safety path. The selected threshold per dataset is the highest-reuse `TSER + residual repair` point under a `2.0%` drop budget when available.

## Definitions

- `No reuse`: all nodes use the reference encoder target pool.
- `Direct only`: only high-support SimHash/CAM anchors are reused.
- `Soft direct reuse`: support-only fuzzy reuse without TSER or residual repair; shown as pending if the no-TSER run has not been produced.
- `TSER filtering`: TSER-accepted anchors are directly reused without residual repair.
- `TSER + residual repair`: medium-support TSER candidates are corrected by the residual adapter or rejected to compute.

## Main Table

| Dataset | T | Method | Reuse | Direct | Residual | Compute | Acc | Drop | AvgErr | HitErr | Source |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CR | 31 | No reuse | 0.00% | 0.00% | 0.00% | 100.00% | 0.7007 | 0.00% | 0.00000 | 0.00000 | `tser/cora_T31_runs3.log` |
| CR | 31 | Direct only | 15.80% | 15.80% | 0.00% | 84.20% | 0.6929 | 0.77% | 0.03160 | 0.20039 | `tser/cora_T31_runs3.log` |
| CR | 31 | Soft direct reuse | 60.40% | 60.40% | 0.00% | 39.60% | 0.6636 | 3.71% | 0.14728 | 0.24388 | `no_tser/cora_T31_runs3.log` |
| CR | 31 | TSER filtering | 51.70% | 51.70% | 0.00% | 48.30% | 0.6749 | 2.58% | 0.12053 | 0.23322 | `tser/cora_T31_runs3.log` |
| CR | 31 | TSER + residual repair | 39.00% | 4.95% | 34.05% | 61.00% | 0.6826 | 1.81% | 0.06754 | 0.17273 | `tser/cora_T31_runs3.log` |
| PB | 24 | No reuse | 0.00% | 0.00% | 0.00% | 100.00% | 0.7522 | 0.00% | 0.00000 | 0.00000 | `tser/pubmed_T24_runs3.log` |
| PB | 24 | Direct only | 29.90% | 29.90% | 0.00% | 70.10% | 0.7373 | 1.49% | 0.03812 | 0.12706 | `tser/pubmed_T24_runs3.log` |
| PB | 24 | Soft direct reuse | 90.20% | 90.20% | 0.00% | 9.80% | 0.6751 | 7.70% | 0.12587 | 0.13946 | `no_tser/pubmed_T24_runs3.log` |
| PB | 24 | TSER filtering | 60.90% | 60.90% | 0.00% | 39.10% | 0.7089 | 4.33% | 0.08063 | 0.13225 | `tser/pubmed_T24_runs3.log` |
| PB | 24 | TSER + residual repair | 33.10% | 5.22% | 27.88% | 66.90% | 0.7363 | 1.59% | 0.04162 | 0.12526 | `tser/pubmed_T24_runs3.log` |

## Reading

The expected trend is that candidate discovery alone can expose many anchors but may cause high error when fuzzy hits are accepted blindly. `TSER filtering` lowers unsafe reuse by using graph risk, and `TSER + residual repair` recovers part of the fuzzy bucket while keeping embedding error and accuracy drop controlled.
