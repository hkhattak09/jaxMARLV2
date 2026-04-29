# LoRASA Adapter Diagnostic Summary

**Checkpoint:** `/content/checkpoint_final.pkl`

## Metadata

- **model_type**: mappo_t_lorasa
- **checkpoint_kind**: final
- **step**: 39900000
- **update**: 2500
- **actor_step**: 25000
- **map_name**: protoss_10_vs_10

## Errors / Warnings

_No errors or warnings._

## Overview

- **LoRA blocks discovered:** 10
- **Adapter rows analyzed:** 90
- **Configured ranks seen:** [8]

## Flag Counts

- `effective_rank_lt_half_configured`: 78 / 90
- `factor_imbalance`: 60 / 90
- `ill_conditioned`: 0 / 90
- `large_adapter_ratio`: 0 / 90
- `numerical_rank_rel_1e_3_lt_configured`: 62 / 90
- `tiny_adapter_ratio`: 60 / 90
- `tiny_min_sv`: 90 / 90
- `zero_or_near_zero_adapter`: 60 / 90

## Worst 10 by Effective-Rank / Configured-Rank Ratio

| path | slot | configured_rank | effective_rank | ratio |
|------|------|-----------------|----------------|-------|
| params/action_out | 0 | 8 | 0.0000 | 0.0000 |
| params/action_out | 1 | 8 | 0.0000 | 0.0000 |
| params/action_out | 4 | 8 | 0.0000 | 0.0000 |
| params/action_out | 5 | 8 | 0.0000 | 0.0000 |
| params/action_out | 7 | 8 | 0.0000 | 0.0000 |
| params/action_out | 8 | 8 | 0.0000 | 0.0000 |
| params/base_0 | 0 | 8 | 0.0000 | 0.0000 |
| params/base_0 | 1 | 8 | 0.0000 | 0.0000 |
| params/base_0 | 4 | 8 | 0.0000 | 0.0000 |
| params/base_0 | 5 | 8 | 0.0000 | 0.0000 |

## Worst 10 by Condition Number

| path | slot | condition_number | max_sv | min_nonzero_sv |
|------|------|------------------|--------|----------------|
| params/rnn/gru_cell/input_update | 3 | 1.0895e+03 | 4.5600e+00 | 4.1855e-03 |
| params/rnn/gru_cell/recurrent_update | 6 | 1.0407e+03 | 2.6337e+00 | 2.5307e-03 |
| params/rnn/gru_cell/recurrent_reset | 6 | 9.1727e+02 | 2.5468e+00 | 2.7764e-03 |
| params/rnn/gru_cell/input_reset | 6 | 8.6453e+02 | 2.5869e+00 | 2.9923e-03 |
| params/rnn/gru_cell/input_update | 6 | 6.4877e+02 | 2.6680e+00 | 4.1123e-03 |
| params/rnn/gru_cell/recurrent_reset | 3 | 5.0253e+02 | 4.4573e+00 | 8.8698e-03 |
| params/rnn/gru_cell/recurrent_update | 3 | 4.7179e+02 | 4.5110e+00 | 9.5615e-03 |
| params/rnn/gru_cell/recurrent_candidate | 3 | 4.2542e+02 | 4.2717e+00 | 1.0041e-02 |
| params/rnn/gru_cell/input_update | 2 | 4.2386e+02 | 3.2684e+00 | 7.7110e-03 |
| params/rnn/gru_cell/recurrent_candidate | 6 | 3.6596e+02 | 2.5383e+00 | 6.9361e-03 |

## Smallest Adapter / Backbone Ratios

| path | slot | adapter_fro_norm | backbone_fro_norm | ratio |
|------|------|------------------|-------------------|-------|
| params/action_out | 0 | 0.0000e+00 | 9.3041e+00 | 0.0000e+00 |
| params/action_out | 1 | 0.0000e+00 | 9.3041e+00 | 0.0000e+00 |
| params/action_out | 4 | 0.0000e+00 | 9.3041e+00 | 0.0000e+00 |
| params/action_out | 5 | 0.0000e+00 | 9.3041e+00 | 0.0000e+00 |
| params/action_out | 7 | 0.0000e+00 | 9.3041e+00 | 0.0000e+00 |
| params/action_out | 8 | 0.0000e+00 | 9.3041e+00 | 0.0000e+00 |
| params/base_0 | 0 | 0.0000e+00 | 2.5433e+01 | 0.0000e+00 |
| params/base_0 | 1 | 0.0000e+00 | 2.5433e+01 | 0.0000e+00 |
| params/base_0 | 4 | 0.0000e+00 | 2.5433e+01 | 0.0000e+00 |
| params/base_0 | 5 | 0.0000e+00 | 2.5433e+01 | 0.0000e+00 |

## Largest Adapter / Backbone Ratios

| path | slot | adapter_fro_norm | backbone_fro_norm | ratio |
|------|------|------------------|-------------------|-------|
| params/rnn/gru_cell/recurrent_reset | 3 | 4.8154e+00 | 1.6114e+01 | 2.9884e-01 |
| params/rnn/gru_cell/recurrent_update | 3 | 4.6131e+00 | 1.6259e+01 | 2.8373e-01 |
| params/rnn/gru_cell/recurrent_candidate | 2 | 5.3950e+00 | 1.9321e+01 | 2.7924e-01 |
| params/base_2 | 2 | 4.5208e+00 | 1.6509e+01 | 2.7384e-01 |
| params/rnn/gru_cell/input_update | 3 | 4.6350e+00 | 1.7415e+01 | 2.6616e-01 |
| params/rnn/gru_cell/input_candidate | 3 | 4.0713e+00 | 1.5790e+01 | 2.5784e-01 |
| params/base_1 | 3 | 3.8166e+00 | 1.5891e+01 | 2.4017e-01 |
| params/rnn/gru_cell/input_reset | 2 | 4.1101e+00 | 1.7341e+01 | 2.3702e-01 |
| params/base_2 | 3 | 3.8947e+00 | 1.6509e+01 | 2.3591e-01 |
| params/rnn/gru_cell/input_reset | 3 | 4.0785e+00 | 1.7341e+01 | 2.3519e-01 |

## Per-Block Aggregate Stats

| block | rows | mean_eff_rank | mean_cond_num | mean_adapter_norm | mean_ratio |
|-------|------|---------------|---------------|-------------------|------------|
| params/action_out | 9 | 1.7029 | 6.1161e+01 | 4.0096e-01 | 4.3095e-02 |
| params/base_0 | 9 | 1.7283 | 4.1753e+01 | 1.7301e+00 | 6.8025e-02 |
| params/base_1 | 9 | 1.4122 | 8.1730e+01 | 1.0972e+00 | 6.9042e-02 |
| params/base_2 | 9 | 1.2075 | 1.6653e+02 | 1.1892e+00 | 7.2031e-02 |
| params/rnn/gru_cell/input_candidate | 9 | 1.3729 | 7.9290e+01 | 1.2081e+00 | 7.6509e-02 |
| params/rnn/gru_cell/input_reset | 9 | 1.0072 | 3.7927e+02 | 1.2004e+00 | 6.9224e-02 |
| params/rnn/gru_cell/input_update | 9 | 0.9346 | 7.2070e+02 | 1.2978e+00 | 7.4523e-02 |
| params/rnn/gru_cell/recurrent_candidate | 9 | 0.9162 | 3.3850e+02 | 1.3991e+00 | 7.2416e-02 |
| params/rnn/gru_cell/recurrent_reset | 9 | 1.0310 | 5.0223e+02 | 1.1241e+00 | 6.9763e-02 |
| params/rnn/gru_cell/recurrent_update | 9 | 0.8488 | 5.8282e+02 | 1.1910e+00 | 7.3256e-02 |

## Per-Slot Aggregate Stats

| slot | rows | mean_eff_rank | mean_cond_num | mean_adapter_norm | mean_ratio |
|------|------|---------------|---------------|-------------------|------------|
| 0 | 10 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 1 | 10 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 2 | 10 | 4.1010 | 1.4851e+02 | 3.7645e+00 | 2.1828e-01 |
| 3 | 10 | 3.4371 | 3.2379e+02 | 4.1186e+00 | 2.4104e-01 |
| 4 | 10 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 5 | 10 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 6 | 10 | 3.4074 | 4.1389e+02 | 2.7710e+00 | 1.5978e-01 |
| 7 | 10 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 8 | 10 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |

## Final Recommendation

Many adapters appear rank-collapsed. Consider lower active rank, singular-value flooring, or rank-expanded residual ES before strict fixed-rank Riemannian ES. Many adapters are tiny relative to the backbone. Use norm-scaled sigma and conservative ES step sizes.
