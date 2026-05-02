# MACA-Role: Role-Conditioned Actor-Critic with Hyperscale ES Fine-Tuning

## Problem Statement

How might we enable expressive, stable role-specific specialization in a parameter-shared cooperative MARL stack, where the shared backbone co-learns a representation useful for all roles, and the resulting architecture is natively compatible with EggRoll's low-rank population evaluation at hyperscale?

## Recommended Direction

Build **MACA-Role**: a MAPPO-T/MACA stack where the shared GRU actor backbone and transformer critic backbone are trained end-to-end alongside role-specific output heads. Rather than injecting role capacity additively inside frozen backbone layers (LoRA), we place it at the output: role-specific actor policy heads and role-specific critic Q/V heads, trained from scratch with the backbone.

## Architecture Overview

### Actor (All Experiments)

```text
obs → Shared base MLP (3×Dense64+ReLU+LN) → Shared GRU → H
H → Role-k head: Dense(64) → ReLU → Dense(32) → ReLU → Dense(action_dim) → π_k
```

Two-layer MLP per role (64→32→action_dim) with ReLU activations. The shared GRU processes all role-conditioned features with a single dynamics model.

**KL diversity penalty:** Cosine decay from 0.001 → 0 over first 30% of steps. Prevents actor head mode collapse.

### Critic

The critic architecture varies by experiment (see below). All variants share:
- **Transformer encoder** (shared across all roles)
- **Attention matrix** Ã (NxN, shared)
- **CorrSet construction** (shared, per-agent)

The NxN attention matrix captures all pairwise agent interactions; since roles are observable per-agent, this matrix is implicitly role-conditioned without additional role-specific attention computation.

## Experimental Sequence

### Exp 1: Post-GRU Actor Heads + Shared Critic (MVP)

**Status:** Design finalized, ready for implementation.

**Actor:** Post-GRU role heads only (64→32→action_dim).

**Critic:** Fully shared — no role-specific components. Uses existing `TransVCritic` exactly as in current MAPPO-T.

**Purpose:** Validate that post-GRU actor heads alone provide sufficient role specialization without critic changes.

### Exp 2: Post-GRU Actor Heads + Role-Specific Critic Heads

**Status:** LOCKED.

**Actor:** Same as Exp 1.

**Critic architecture (LOCKED):**
```text
joint obs → Shared Transformer Encoder → H (n_agents × 64)
H → flatten → z_intermediate = Dense(256) → ReLU → LayerNorm(256)   [SHARED]

Per role k = 0..5:
  z_k = Dense(128)(z_intermediate) → ReLU → LayerNorm(128) → Dense(64) → z_k

  V_k = MLPHead(hidden_dim=64, out_dim=1)(z_k)           [per-role V-head]

  zsa_k = shared_sa_encoder(concat(z_k, action_flat))     [shared linear]
  Q_k = Dense(1)(zsa_k)                                   [per-role linear Q]

  zspi_k = shared_sa_encoder(concat(z_k, policy_flat))    [same sa_encoder]
  EQ_k = Dense(1)(zspi_k)                                 [same Q-head weights]
```

**Marginalization preservation:** Both `sa_encoder` (shared linear) and `q_head_k` (per-role linear) are linear with no intermediate activation. Thus `EQ_k(s, π) = Σ_a π(a) Q_k(s, a)` holds per role.

**GAE targets:**
```text
V_env = mean_k(V_k)
Q_env = mean_k(Q_k)
EQ_env = mean_k(EQ_k)
```
Standard GAE computed on these env-level mixtures.

**Per-agent baseline:**
```text
VQ_i      = Q_{r_i}(shared_sa_encoder(concat(z_{r_i}, mixed_actions_i)))
VQ_COMA_i = Q_{r_i}(shared_sa_encoder(concat(z_{r_i}, coma_actions_i)))
baseline_i = w_self · VQ_COMA_i + w_group · VQ_i + w_joint · EQ_env
```

**Critic loss:**
```text
critic_loss = mean_k [MSE(V_k, V_target) + MSE(Q_k, Q_target) + MSE(EQ_k, EQ_target)]
            + λ_div_critic · diversity_penalty(z_k_means)
```

**Diversity penalty:** Activation-space contrastive loss on per-role mean embeddings:
```text
diversity_penalty = -Σ_{i<j} ||z_i_mean - z_j_mean||²
```
with `λ_div_critic = 1e-4`.

**Why this design:**
- Shared `z_intermediate` extracts common state features (all roles see the same battle)
- Per-role projections `z_k` learn role-specific state abstractions
- Shared `sa_encoder` remains linear to preserve Jensen's inequality / marginalization
- Per-role Q-heads read from role-specific state-action representations
- Mean-pooling for GAE ensures env-level targets don't depend on any single role

**Collapse monitoring:** Track `std({V_k})` and `std({Q_k})` across roles. If relative std < 0.01 after 50% training, increase `λ_div_critic` or add parameter-space diversity.

### Exp 3: Pre-GRU Routes + Post-GRU Actor Heads + Shared Critic

**Status:** LOCKED.

**Actor:** Residual pre-GRU routes + post-GRU heads.
```text
obs ──┬──→ [SHARED base MLP: 3×Dense64+ReLU+LN] ──→ shared_embedding (64-dim)
      │                                                  │
      │                                                  │
      └──→ route_k: Dense(obs_dim→128) → ReLU → Dense(128→64) → ReLU
              [per-role, raw obs input, kernel_init=orthogonal(0.1)]
              │                                         │
              └─────────────────────────────────────────┘
                                                         │
                                                   (+) additive
                                                         │
                                                         ↓
                                                 gru_input (64-dim)
                                                         │
                                                         ↓
                                              [SHARED GRU] → H → head_k → π_k
```

**Route details (LOCKED):**
- Input: **Raw obs** — independent feature extractor, not constrained by shared MLP
- Depth: **2 layers** (obs_dim→128→ReLU→64→ReLU) — sufficient capacity, runs parallel to base MLP
- Combination: **Additive residual** `gru_input = shared_embedding + route_k`
- LayerNorm: **None** on route or sum — shared_embedding already normalized
- Init: `orthogonal(0.1)` — routes start as tiny perturbations, grow via gradients
- Bias: **Yes** — cheap learnable zero-point

**Critic:** Fully shared (same as Exp 1).

**Purpose:** Test whether role-specific observation processing before the GRU improves the shared dynamics model.

### Exp 4: Pre-GRU Routes + Post-GRU Actor Heads + Role-Specific Critic Heads

**Status:** LOCKED.

**Actor:** Same as Exp 3 (pre-GRU residual routes + post-GRU role heads).

**Critic:** Same as Exp 2 (role-specific heads on shared transformer).

**Purpose:** Full role conditioning on both actor (input + output) and critic. Maximum expressive capacity within parameter-sharing constraints.

## Training

End-to-end from scratch. All parameters (backbone + heads) train jointly using per-role MACA advantage. Role IDs derived from `env_state.state.unit_types[:, :num_allies]` (same as existing LoRASA `adapter_ids`).

## EggRoll ES Fine-Tuning

After RL convergence, freeze the shared backbone, shared critic, and all role-specific critic heads. Optimize only the actor role head parameters via EggRoll against win rate. No critic is used in ES — fitness is the metric directly.

EggRoll perturbs head matrices with low-rank noise (rank 1–4); aggregate updates are naturally full-rank and require no SVD or retraction. Architecture designed for compatibility with EggRoll's `do_mm` / `do_Tmm` hooks on Flax `nn.Dense` kernel parameters.

**ES searchable params by experiment:**
- Exp 1–2: Post-GRU actor heads only
- Exp 3–4: Pre-GRU routes + post-GRU actor heads

## Key Assumptions to Validate

- [ ] **Role heads won't collapse to identical policies.** Monitor pairwise KL between role policy distributions on held-out states. If KL < 0.1 after 50% of training, increase diversity penalty or add pre-GRU routes.  
- [ ] **The shared GRU learns a multiplexed representation useful for all roles.** Track per-role policy entropy and effective rank of each head's weight matrix. If one role dominates (near-zero entropy while others remain high), the backbone is biased.  
- [ ] **Two-layer MLP heads (64→32→action_dim) provide sufficient capacity for role separation.** Monitor effective rank of final layer weights. If rank << 32, heads are over-parameterized. If all roles have similar effective rank but low KL, increase hidden dims or add pre-GRU routes.  
- [ ] **EggRoll can effectively search the actor head parameter space.** Compare EggRoll rank-1 vs rank-4 perturbations on frozen actor heads. If rank-1 matches rank-4 in win-rate improvement, the search space is well-conditioned.  
- [x] **Shared attention matrix + role-specific Q/V heads is sufficient.** Confirmed: the transformer encoder learns from signals through all three baseline heads (joint, individual, CorrSet), producing a representation where the NxN attention matrix implicitly encodes role-to-role interaction patterns. No explicit role-specific attention needed.  
- [ ] **Role-specific critic heads won't collapse.** Monitor `std({V_k})` and `std({Q_k})` across roles. If relative std < 0.01, increase `λ_div_critic` or switch to parameter-space diversity.  
- [ ] **Pre-GRU residual routes add value without overwhelming shared backbone.** Monitor gradient magnitudes: `||∇route_k|| / ||∇base_MLP||` should be < 2. If higher, routes are dominating.  

## MVP Scope (Exp 1)

**In scope:**
- Modify `ActorTrans` to support `n_roles` policy heads: Dense(64) → ReLU → Dense(32) → ReLU → Dense(action_dim)
- Add `role_ids` argument to actor forward pass
- Add role routing: agent `i` with role `r_i` uses head `r_i`
- Add KL-diversity penalty between role policies (cosine decay, 0.001 → 0 over 30% of steps)
- Integrate with existing MACA advantage computation using shared critic
- Single SMAX map (`protoss_10_vs_10`) for initial validation
- EggRoll architecture compatibility: Flax `nn.Dense` kernels marked as `MM_PARAM`

**Out of scope:**
- Multiple SMAX maps (validate on one first)
- Role-specific critic heads (Exp 2)
- Pre-GRU role routes (Exp 3–4)
- Role-specific attention matrices or CorrSet construction
- Transformer replacement for GRU in actor
- EggRoll on critic heads (actor-only for ES phase)
- Curriculum or automatic role discovery

## Not Doing (and Why)

- **LoRA-style adapters inside backbone layers** — The effective rank evidence (~4–5) showed role variation is low-dimensional at the output, not inside layers. LoRA also requires SVD/retraction for ES, which fights EggRoll's native design.  
- **Phased training (shared pretrain → add heads)** — Compute budget makes end-to-end training feasible (~40M steps in 4 hours). Phased training creates landscape discontinuities and the "surgery instability" problem.  
- **Role-specific attention / CorrSet** — Confirmed: the shared transformer encoder learns a representation useful for all three baseline heads (joint, individual, CorrSet). The NxN attention matrix encodes pairwise agent interactions where role is a sub-feature of each observation. This is sufficient for role-conditioned credit assignment without explicit role-specific attention mechanisms.  
- **Transformer in actor replacing GRU** — No teacher forcing, O(T²) inference cost, and the GRU is sufficient for the decentralized actor.  
- **ES on critic heads** — ES is zeroth-order black-box optimization. Adding critic heads to the search space increases dimensionality without clear benefit, since the critic isn't needed for fitness evaluation.  
- **Constrained bottleneck heads based on LoRA rank** — LoRA effective rank (~4–5) measures residual adaptation of frozen features, not the capacity needed for full heads trained from scratch. Head sizes should match or slightly exceed the original single head's capacity (64→32→action_dim).  
- **SVD / Riemannian retraction in EggRoll** — EggRoll's aggregate update is already full-rank (`min(N·r, m, n)`). Retracting back to low-rank discards information that EggRoll intentionally generates.  
- **Per-role sa_encoder** — Shared linear `sa_encoder` preserves marginalization and is sufficient. Role-specific Q-heads operate on role-specific `z_k` inputs, giving each role a different view of the state-action space through the same projection.  

## Open Questions

- **Pre-GRU route initialization sensitivity:** Does `orthogonal(0.1)` produce routes that grow to meaningful magnitude, or do they stay near zero? Monitor `||route_k|| / ||shared_embedding||`.
- **Exp 3–4 ordering:** Sequential (Exp 3 then 4) or parallel? Sequential gives cleaner attribution but takes longer.
- **Diversity penalty tuning:** Is `λ_div_critic = 1e-4` sufficient for critic heads, or do we need parameter-space diversity as well?

## Architecture Summary (MACA-Role)

```text
Phase 1–2: RL Training (end-to-end)

  Exp 1 & 3 Actor (shared critic):
    obs → base MLP (3×Dense64+ReLU+LN, shared) → GRU (shared) → H
    H → [head_0, ..., head_5] → π_k
    head_k: Dense(64) → ReLU → Dense(32) → ReLU → Dense(action_dim)
    loss = PPO_clip(π_k, A^MACA) + λ_div·Σ_{i<j} KL(π_i || π_j)

  Exp 2 & 4 Actor (role-specific critic):
    Same as above

  Exp 3 & 4 Actor (+ pre-GRU routes):
    obs → base MLP → shared_embedding (64)
    route_k(obs): Dense(obs_dim→128) → ReLU → Dense(128→64) → ReLU
    gru_input = shared_embedding + route_k(obs)
    gru_input → GRU → H → head_k → π_k

  Exp 1 & 3 Critic (shared):
    joint obs → Transformer Encoder → Ã (NxN, shared) → CorrSet C_i
    zs → shared V-head → V
    zsa → shared Q-head → Q
    A^MACA = Q - b^MACA

  Exp 2 & 4 Critic (role-specific heads):
    joint obs → Transformer Encoder → H → flatten → z_intermediate (256, shared)
    z_k = Dense(128)(z_intermediate) → ReLU → LN → Dense(64) → z_k  [per-role]
    V_k = MLPHead(64→1)(z_k)                                          [per-role]
    zsa_k = shared_sa_encoder(concat(z_k, action_flat))               [shared linear]
    Q_k = Dense(1)(zsa_k)                                             [per-role]
    EQ_k = Dense(1)(shared_sa_encoder(concat(z_k, policy_flat)))      [per-role]
    V_env = mean_k(V_k), Q_env = mean_k(Q_k), EQ_env = mean_k(EQ_k)
    VQ_i = Q_{r_i}(zsa_{r_i}), VQ_COMA_i = Q_{r_i}(coma_{r_i})
    baseline_i = w_self·VQ_COMA_i + w_group·VQ_i + w_joint·EQ_env
    A_i^MACA = EQ_return - baseline_i
    critic_loss = mean_k [MSE(V_k,V_t) + MSE(Q_k,Q_t) + MSE(EQ_k,EQ_t)]
                + λ_div_critic · diversity_penalty(z_k_means)

Phase 3: EggRoll ES
  Freeze:  base MLP, GRU, Transformer, all critic heads, sa_encoder
  Search:  Actor head parameters (pre-GRU routes if present, post-GRU heads)
  Fitness: Win rate (black-box, no critic)
  Update:  M_{t+1} = M_t + (α/N) Σ E_i · f(M_t + σE_i)
           where E_i = (1/√r) A_i B_i^T, r ∈ {1, 2, 4}
```

## Locked Decisions

| Decision | Value | Experiment |
|----------|-------|------------|
| Actor head arch | Dense(64)→ReLU→Dense(32)→ReLU→Dense(action_dim) | All |
| KL diversity schedule | Cosine decay 0.001→0, first 30% steps | All |
| Critic: shared `z_intermediate` | Dense(256)→ReLU→LayerNorm(256) | Exp 2, 4 |
| Critic: per-role projection | Dense(128)→ReLU→LN→Dense(64) | Exp 2, 4 |
| Critic: V-head | MLPHead(hidden_dim=64, out_dim=1) | Exp 2, 4 |
| Critic: Q/EQ-head | Linear on `sa_encoder(concat(z_k, action))` | Exp 2, 4 |
| Critic: `sa_encoder` | Shared, linear, no activation | Exp 2, 4 |
| Critic: GAE pooling | Mean across roles | Exp 2, 4 |
| Critic: diversity penalty | Activation-space L2 on z_k_means, λ=1e-4 | Exp 2, 4 |
| Pre-GRU route | Residual additive: `shared + route_k(obs)` | Exp 3, 4 |
| Pre-GRU route arch | Dense(obs_dim→128)→ReLU→Dense(128→64)→ReLU | Exp 3, 4 |
| ES searchable params | Actor heads (+ routes if present) | All |
| ES frozen params | All backbone + all critic + sa_encoder | All |
