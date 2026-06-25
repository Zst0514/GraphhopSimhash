# Section 6 Missing Data Package

## Scope

This package fills the current Section 6 placeholders from existing local traces.
It does not launch new encoder generation or training.

Important provenance:

- GPU BFPA5 seconds are parsed from existing `output/bfpa635_b256_generation/*W4BFPA5_B256*.log` files. They are local BFPA5 logs, so the A100 label in the paper should be replaced by a real A100 run before final submission.
- GRACE timing is GPU-calibrated trace composition: same node batch/sequence baseline, TSER miss stream, ONNXim BFP cycle ratios, and NDP-local embedding traffic.
- HEAT and GFMEngine are path-level reconstructions from the local `HEAT/` and `GFMEngine/` folders, not official private simulators.
- Energy numbers use explicit power-model inputs. CAM area/energy is CACTI-backed; NPU/GFM/HEAT power should be replaced after RTL synthesis.

## Main Numbers

- Average GRACE speedup vs GPU BFPA5 proxy: `1.87x`.
- Average HEAT-style bit-serial speedup vs GPU BFPA5 proxy: `0.17x`.
- Average GFMEngine-PQ M128/BW64 speedup vs GPU BFPA5 proxy: `1.45x`.
- Average GRACE energy efficiency vs GPU under the current power model: `20.83x`.
- GRACE energy-efficiency improvement over HEAT-style: `3.23x`.
- GRACE energy-efficiency improvement over GFMEngine-PQ M128/BW64: `3.51x`.

## GPU Timing Inputs

| Encoder dataset | BFPA5 encoding time | Shared tasks |
| --- | ---: | --- |
| Cora | 108.0s | CN, CL |
| PubMed | 1396.0s | PN, PL |
| OGBN-Arxiv | 7787.0s | AR |
| Wiki-CS | 797.0s | WK |

## E2E Speedup

| Task | A100/GPU | HEAT-style | GFMEngine-PQ M128 | GRACE | GRACE time | Raw TSER drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | 1.00x | 0.17x | 1.43x | 1.87x | 57.8s | 0.98% |
| CL | 1.00x | 0.17x | 1.43x | 1.86x | 58.2s | 1.96% |
| PN | 1.00x | 0.17x | 1.47x | 1.87x | 746.7s | 1.95% |
| PL | 1.00x | 0.17x | 1.47x | 1.95x | 716.1s | 1.67% |
| AR | 1.00x | 0.17x | 1.44x | 1.86x | 4179.8s | 2.62% |
| WK | 1.00x | 0.17x | 1.47x | 1.84x | 433.4s | 1.15% |

## GRACE Latency Breakdown

| Task | Total overlap | Encoder | Loader raw | CAM/LRU | NPU->NDP write | Hit emb read | Graph mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | 57.769s | 57.768s | 1.736s | 14.087us | 208.321us | 34.576us | 337.999us |
| CL | 58.192s | 58.191s | 1.748s | 14.111us | 209.846us | 34.194us | 337.999us |
| PN | 746.711s | 746.707s | 22.433s | 102.568us | 1.517ms | 251.747us | 2.838ms |
| PL | 716.147s | 716.143s | 21.515s | 101.598us | 1.455ms | 267.268us | 2.838ms |
| AR | 4179.796s | 4179.743s | 125.571s | 881.634us | 13.073ms | 2.151ms | 37.341ms |
| WK | 433.409s | 433.401s | 13.021s | 61.103us | 915.112us | 145.654us | 6.919ms |

## GFMEngine Sensitivity

| Scenario | Avg norm | Avg speedup | Memory norm | Compute norm |
| --- | ---: | ---: | ---: | ---: |
| GFMEngine-PQ-M16-BW256 | 0.2338x | 4.28x | 0.7715x | 0.2957x |
| GFMEngine-PQ-M64-BW64 | 0.4352x | 2.30x | 11.1724x | 0.4564x |
| GFMEngine-PQ-M128-BW64 | 0.6901x | 1.45x | 22.0759x | 0.6706x |

## Output Files

- `e2e_speedup.tsv`: platform time/speedup table.
- `energy_efficiency.tsv`: energy table under explicit power assumptions.
- `latency_breakdown.tsv`: GRACE component timing in seconds.
- `cam_cacti_table.tsv` and `cam_overhead_by_task.tsv`: CAM area/latency/energy.
- `bfpa_boundary.tsv`: BFPA3/4/5/6 precision boundary.
- `block_size_sensitivity.tsv`: B256/B512 BFPA sensitivity from existing runs.
- `tser_threshold_sensitivity.tsv`: existing T sweep data.
- `hamming_radius_model.tsv`: hardware-latency model only; accuracy/yield radius sweep is not measured in the current package.
- `hardware_area_power.tsv`: current hardware model table.
- `latex_snippets/`: copyable LaTeX tables and figure stubs.

Figures:
- `docs/results/sec6_missing_data_package/sec6_e2e_speedup.pdf`
- `docs/results/sec6_missing_data_package/sec6_energy_efficiency.pdf`
- `docs/results/sec6_missing_data_package/sec6_latency_breakdown.pdf`

## Caveats For Paper Text

- Do not claim the current absolute seconds are A100-measured until the BFPA5 encoder log is rerun on A100.
- Do not call the Hamming-radius table an accuracy sensitivity result; it is only a CAM latency/candidate-sweep placeholder.
- The current reproducible GRACE speedup is about 1.8x vs the BFPA5 GPU encoder proxy, not the 11.58x sentence currently present in `main.tex`.
- The raw TSER 40% drop average from `tser_reuse_drop_tradeoff_40pt_alignment.tsv` is closer to the 1.7%-1.8% range; the lower target-drop column is a shifted plotting/alignment view.
