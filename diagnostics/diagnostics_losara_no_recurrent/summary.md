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

- **LoRA blocks discovered:** 7
- **Adapter rows analyzed:** 63
- **Configured ranks seen:** [8]

## Flag Counts

- `effective_rank_lt_half_configured`: 51 / 63
- `factor_imbalance`: 42 / 63
- `ill_conditioned`: 0 / 63
- `large_adapter_ratio`: 0 / 63
- `numerical_rank_rel_1e_3_lt_configured`: 43 / 63
- `tiny_adapter_ratio`: 42 / 63
- `tiny_min_sv`: 63 / 63
- `zero_or_near_zero_adapter`: 42 / 63

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
| params/rnn/gru_cell/input_update | 6 | 1.0598e+03 | 3.7448e+00 | 3.5335e-03 |
| params/rnn/gru_cell/input_reset | 6 | 7.8533e+02 | 3.1251e+00 | 3.9793e-03 |
| params/rnn/gru_cell/input_candidate | 2 | 3.9748e+02 | 4.5388e+00 | 1.1419e-02 |
| params/rnn/gru_cell/input_reset | 2 | 3.2012e+02 | 3.4254e+00 | 1.0701e-02 |
| params/rnn/gru_cell/input_update | 3 | 2.9614e+02 | 3.9883e+00 | 1.3468e-02 |
| params/rnn/gru_cell/input_reset | 3 | 2.4567e+02 | 4.2640e+00 | 1.7357e-02 |
| params/base_2 | 6 | 1.8626e+02 | 2.0217e+00 | 1.0854e-02 |
| params/base_1 | 3 | 1.4545e+02 | 2.6081e+00 | 1.7931e-02 |
| params/rnn/gru_cell/input_candidate | 3 | 1.4114e+02 | 3.5070e+00 | 2.4848e-02 |
| params/action_out | 2 | 1.3962e+02 | 1.5300e+00 | 1.0958e-02 |

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
| params/rnn/gru_cell/input_candidate | 2 | 4.9532e+00 | 1.5790e+01 | 3.1369e-01 |
| params/rnn/gru_cell/input_candidate | 3 | 4.4554e+00 | 1.5790e+01 | 2.8217e-01 |
| params/rnn/gru_cell/input_reset | 3 | 4.8583e+00 | 1.7341e+01 | 2.8016e-01 |
| params/base_0 | 6 | 6.9813e+00 | 2.5433e+01 | 2.7449e-01 |
| params/rnn/gru_cell/input_update | 2 | 4.6322e+00 | 1.7415e+01 | 2.6599e-01 |
| params/rnn/gru_cell/input_update | 3 | 4.3278e+00 | 1.7415e+01 | 2.4851e-01 |
| params/base_0 | 2 | 6.1099e+00 | 2.5433e+01 | 2.4023e-01 |
| params/base_2 | 2 | 3.7906e+00 | 1.6509e+01 | 2.2961e-01 |
| params/rnn/gru_cell/input_reset | 2 | 3.9307e+00 | 1.7341e+01 | 2.2667e-01 |
| params/base_1 | 6 | 3.4835e+00 | 1.5891e+01 | 2.1921e-01 |

## Per-Block Aggregate Stats

| block | rows | mean_eff_rank | mean_cond_num | mean_adapter_norm | mean_ratio |
|-------|------|---------------|---------------|-------------------|------------|
| params/action_out | 9 | 1.6433 | 1.0167e+02 | 5.0454e-01 | 5.4228e-02 |
| params/base_0 | 9 | 1.5674 | 6.4518e+01 | 1.9858e+00 | 7.8080e-02 |
| params/base_1 | 9 | 1.4275 | 1.0658e+02 | 1.0363e+00 | 6.5212e-02 |
| params/base_2 | 9 | 1.4102 | 1.2547e+02 | 1.0142e+00 | 6.1432e-02 |
| params/rnn/gru_cell/input_candidate | 9 | 1.2222 | 2.2219e+02 | 1.3497e+00 | 8.5481e-02 |
| params/rnn/gru_cell/input_reset | 9 | 1.0225 | 4.5037e+02 | 1.3396e+00 | 7.7250e-02 |
| params/rnn/gru_cell/input_update | 9 | 0.9450 | 4.9182e+02 | 1.4138e+00 | 8.1183e-02 |

## Per-Slot Aggregate Stats

| slot | rows | mean_eff_rank | mean_cond_num | mean_adapter_norm | mean_ratio |
|------|------|---------------|---------------|-------------------|------------|
| 0 | 7 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 1 | 7 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 2 | 7 | 4.3338 | 1.6472e+02 | 4.0185e+00 | 2.3678e-01 |
| 3 | 7 | 3.9872 | 1.5728e+02 | 3.6925e+00 | 2.1799e-01 |
| 4 | 7 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 5 | 7 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 6 | 7 | 3.5566 | 3.4770e+02 | 3.4027e+00 | 1.9177e-01 |
| 7 | 7 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| 8 | 7 | 0.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |

## Final Recommendation

Many adapters appear rank-collapsed. Consider lower active rank, singular-value flooring, or rank-expanded residual ES before strict fixed-rank Riemannian ES. Many adapters are tiny relative to the backbone. Use norm-scaled sigma and conservative ES step sizes.
