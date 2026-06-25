# Frontend Timing: TSER40 W4BFPA4 vs HEAT-Style Bit-Serial

## Correct Baseline

The local backend/frontend encoder path is fixed `W4BFPA4`, so HEAT-style bit-serial compute must be normalized to `W4 x A4 = 16` bit-plane GEMMs, not to W4A8 or INT8xINT8.

HEAT-style Fig. 6 uses top-degree key vertices at `W8A10` and non-key vertices at `W4A2`.
With `alpha=0.1`, its average compute is `15.2` bit-plane GEMMs per MAC, i.e. `15.2 / 16 = 0.95x` of the local W4BFPA4 compute path.

## Timing Components

- Batch size for streamed weight-load rounding: `64` nodes.
- Weighted total shares: compute `0.7`, weight `0.2`, activation `0.05`, output `0.04`, control `0.01`.
- On-chip cache-read scale for reused embeddings: `0.1` of an encoder output write.

Weight-load is reported in two forms:

- `Weight stream`: split high/low precision passes with batch rounding.
- `Weight lower`: per-node lower bound using average weight bits.

## Aggregate

| Scope | Policy | Reuse | Drop | Compute | Weight Stream | Weight Lower | Activation | Output Write | Cache Read | Weighted Total, Off-Chip Cache | Weighted Total, On-Chip Cache |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AVG_HEAT5 | HEAT-style W8A10/W4A2 | 0.00% | 0.00% | 0.9500x | 1.1160x | 1.1000x | 0.7000x | 1.0000x | 0.0000x | 0.9727x | 0.9727x |
| AVG_HEAT5 | TSER40+W4BFPA4 | 40.26% | 1.44% | 0.5974x | 0.5981x | 0.5974x | 0.5974x | 0.5974x | 0.4026x | 0.6136x | 0.5991x |
| AVG6 | HEAT-style W8A10/W4A2 | 0.00% | 0.00% | 0.9500x | 1.1149x | 1.1000x | 0.7000x | 1.0000x | 0.0000x | 0.9725x | 0.9725x |
| AVG6 | TSER40+W4BFPA4 | 40.03% | 1.39% | 0.5997x | 0.6004x | 0.5997x | 0.5997x | 0.5997x | 0.4003x | 0.6158x | 0.6014x |

## Per Task

| Task | Policy | Reuse | Compute | Weight Stream | Activation | Output Write | Cache Read | Batches | Note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CN | HEAT-style W8A10/W4A2 | 0.00% | 0.9500x | 1.1395x | 0.7000x | 1.0000x | 0.0000x | 44 | HEAT Sec.5.2.1 bit-serial precision split; no TSER reuse |
| CN | TSER40+W4BFPA4 | 39.90% | 0.6010x | 0.6047x | 0.6010x | 0.6010x | 0.3990x | 26 | semantic reuse; miss nodes execute the same fixed W4BFPA4 path |
| CL | HEAT-style W8A10/W4A2 | 0.00% | 0.9500x | 1.1395x | 0.7000x | 1.0000x | 0.0000x | 44 | HEAT Sec.5.2.1 bit-serial precision split; no TSER reuse |
| CL | TSER40+W4BFPA4 | 39.46% | 0.6054x | 0.6047x | 0.6054x | 0.6054x | 0.3946x | 26 | semantic reuse; miss nodes execute the same fixed W4BFPA4 path |
| PN | HEAT-style W8A10/W4A2 | 0.00% | 0.9500x | 1.1003x | 0.7000x | 1.0000x | 0.0000x | 309 | HEAT Sec.5.2.1 bit-serial precision split; no TSER reuse |
| PN | TSER40+W4BFPA4 | 39.90% | 0.6010x | 0.6019x | 0.6010x | 0.6010x | 0.3990x | 186 | semantic reuse; miss nodes execute the same fixed W4BFPA4 path |
| PL | HEAT-style W8A10/W4A2 | 0.00% | 0.9500x | 1.1003x | 0.7000x | 1.0000x | 0.0000x | 309 | HEAT Sec.5.2.1 bit-serial precision split; no TSER reuse |
| PL | TSER40+W4BFPA4 | 42.36% | 0.5764x | 0.5761x | 0.5764x | 0.5764x | 0.4236x | 178 | semantic reuse; miss nodes execute the same fixed W4BFPA4 path |
| AR | HEAT-style W8A10/W4A2 | 0.00% | 0.9500x | 1.1005x | 0.7000x | 1.0000x | 0.0000x | 2647 | HEAT Sec.5.2.1 bit-serial precision split; no TSER reuse |
| AR | TSER40+W4BFPA4 | 39.69% | 0.6031x | 0.6032x | 0.6031x | 0.6031x | 0.3969x | 1596 | semantic reuse; miss nodes execute the same fixed W4BFPA4 path |
| WK | HEAT-style W8A10/W4A2 | 0.00% | 0.9500x | 1.1093x | 0.7000x | 1.0000x | 0.0000x | 184 | HEAT Sec.5.2.1 bit-serial precision split; no TSER reuse |
| WK | TSER40+W4BFPA4 | 38.90% | 0.6110x | 0.6120x | 0.6110x | 0.6110x | 0.3890x | 112 | semantic reuse; miss nodes execute the same fixed W4BFPA4 path |

## Interpretation

- Against fixed `W4BFPA4`, HEAT-style compute is only about `0.95x`, because its high-precision `W8A10` key branch offsets the low `W4A2` branch.
- HEAT-style weight loading is not free: even the per-node lower bound is `1.10x` of W4 weight traffic, and split precision streams are slightly higher after batch rounding.
- TSER40 keeps the same fixed W4BFPA4 miss path, but executes it for only the miss nodes. Its compute, activation load, and lower-bound weight load scale with the miss rate.
- The decisive comparison is therefore not HEAT bit-plane count alone; it is component timing under the same fixed W4BFPA4 baseline.
