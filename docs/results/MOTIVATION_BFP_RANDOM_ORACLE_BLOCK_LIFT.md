# Motivation BFP Random/Oracle Block-Lift Profiling

This profiling is for the Motivation section only.  It does not evaluate the
final Graph x Stress online policy.  The goal is to show that BFPA4 loss is
block-sparse: spending the same BFPA6 budget on the right activation blocks is
much more effective than random lifting.

## Setup

- Dataset/task: Cora node classification (`CN`)
- Model: LLaMA2-7B frontend with AWQ W4 weights
- Base activation format: `BFPA4_B256`
- Refined activation format: `BFPA6_B256`
- Lift budget: 20% of BFP activation blocks
- Reference: `W4BFPA8_B128`

## Policies

| Policy | Meaning |
| --- | --- |
| All BFPA4 | No block is lifted to BFPA6. |
| Random lift | Uniformly random 20% of activation blocks are lifted to BFPA6. |
| Oracle lift | Offline top 20% blocks are lifted by BFPA4-vs-BFPA6 quantization-error reduction. |
| All BFPA6 | Every activation block uses BFPA6. |

## Current Result

| Policy | Lifted Blocks | Accuracy Drop |
| --- | ---: | ---: |
| All BFPA4 | 0% | 1.65% |
| Random lift | 20.00% | 1.07% |
| Oracle lift | 20.00% | 0.66% |
| All BFPA6 | 100% | 0.00% |

## Generated Pools

- `cache_data/cora_llama2_7b_oracle_W4BlockBFPA4to6_B256_random20.pt`
- `cache_data/cora_llama2_7b_oracle_W4BlockBFPA4to6_B256_oracle20.pt`

Metadata:

- `output/motivation_block_lift_profile/generate/cora/W4BlockBFPA4to6_B256_random20_metadata.json`
- `output/motivation_block_lift_profile/generate/cora/W4BlockBFPA4to6_B256_oracle20_metadata.json`

Unified 5-run evaluation summaries:

- `output/motivation_block_lift_profile/cn_random20/summary.md`
- `output/motivation_block_lift_profile/cn_oracle20/summary.md`

## Interpretation

Random lifting improves over all-BFPA4 because it spends extra mantissa work on
some blocks.  Oracle lifting is stronger at the same 20% budget, reducing the
drop from 1.07% to 0.66%.  This supports the Motivation claim that precision
demand is sparse and block-dependent, and motivates an online method that can
approximate this oracle without exposing the final Graph x Stress policy in the
Motivation section.
