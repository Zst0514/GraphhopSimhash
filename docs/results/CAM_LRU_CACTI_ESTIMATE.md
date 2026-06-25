# CACTI-Backed CAM/LRU Overhead Estimate

This note records the preliminary CACTI-backed area, power, and energy estimate
for the SimHash-CAM directory and optional embedding hot buffer.

## Command

```bash
/home/zhangshangtong/.conda/envs/OFA/bin/python \
  scripts/run_cacti_cam_lru_estimate.py \
  --entries 1024 4096 32768 \
  --hot-entries 64
```

Outputs:

- `output/cacti_cam_lru_estimate/CAM_LRU_CACTI_ESTIMATE.md`
- `output/cacti_cam_lru_estimate/cam_lru_cacti_estimate.tsv`
- `output/cacti_cam_lru_estimate/cacti_sram_macros.tsv`
- `output/cacti_cam_lru_estimate/raw/*.txt`

## Modeling Scope

CACTI is used for SRAM macro area, read energy, and leakage. The cloned CACTI
tree crashes on its pure-CAM and fully-associative paths, so the CAM directory
is estimated from an equivalent CACTI SRAM macro using conservative scaling:

- CAM area = 2.0x equivalent SRAM data-array area.
- CAM search energy = 2.0x SRAM bit-read energy across all searched signature bits.
- CAM leakage = 2.0x equivalent SRAM macro leakage.
- Signature = 8 heads x 16 bits = 128 bits per entry.
- Metadata = 64 bits per entry.
- Replacement = tree-PLRU, `entries - 1` bits.
- Hot embedding buffer entry = 4096 x 16-bit = 8192 B.
- CACTI technology = native sample config, 22 nm ITRS-HP.

## Results

| Entries | Directory Area | Directory Energy/Search | Directory Leakage | Hot Entries | Hot Buffer Area | Hot Embedding Read | Total Area |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.0306 mm^2 | 5.88 nJ | 17.69 mW | 64 | 0.5387 mm^2 | 8.07 nJ | 0.5693 mm^2 |
| 4,096 | 0.1339 mm^2 | 61.18 nJ | 61.75 mW | 64 | 0.5387 mm^2 | 8.07 nJ | 0.6726 mm^2 |
| 32,768 | 0.9209 mm^2 | 1777.72 nJ | 449.87 mW | 64 | 0.5387 mm^2 | 8.07 nJ | 1.4596 mm^2 |

## Interpretation

The compact directory is small because it stores signatures and metadata, not
full embeddings. At 4K entries, the directory plus a 64-entry embedding hot
buffer is about 0.673 mm^2, which is below 0.1% of an A100-class die area.

The 32K-entry monolithic CAM has much higher search energy. For the paper, this
supports a banked or active-window CAM organization: keep a compact active
directory close to graph/cache state and avoid broadcasting every query across a
large monolithic CAM.

Dynamic power scales linearly with lookup rate:

```text
P_dynamic ~= E_lookup * lookup_rate
```

For example, the 4K-entry directory costs about 61.18 mW at 1 Mlookup/s and
about 611.8 mW at 10 Mlookup/s, before embedding hot-buffer reads.
