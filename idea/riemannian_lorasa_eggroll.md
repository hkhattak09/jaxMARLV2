# Riemannian LoRASA-EGGROLL Design Memo

This memo records the current plan for adding a third fine-tuning phase after
MAPPO-T/MACA pretraining and LoRASA adapter fine-tuning. It is intended to be
self-contained enough that future work can resume from here without relying on
the original discussion.

## Context

The project currently treats `mappo_t` and LoRASA as the main research path.
Other algorithms are comparison baselines.

The current training stack is:

```text
Phase 1: MAPPO-T/MACA
    Shared recurrent actor
    Transformer critic
    MACA-style multi-agent credit assignment

Phase 2: LoRASA-MAPPO
    Frozen MAPPO-T actor backbone
    Unit-type / role-specific LoRA adapters in the actor only
    Existing MACA critic continues to provide the PPO training signal
```

Empirically, phase 2 appears useful: adding role-specific LoRA adapters and
fine-tuning them gives roughly a 0.10-0.13 win-rate increase over continuing
training without adapters.

The proposed phase 3 is a black-box ES-style optimization stage over the
trained actor adapters:

```text
Phase 3: Riemannian LoRASA-EGGROLL
    Freeze the actor backbone
    Do not train or use the critic for the actor update
    Optimize actor role adapters only
    Use win-rate-heavy black-box fitness
    Use EGGROLL-style hyperscale low-rank population evaluation
    Use fixed-rank manifold update semantics for LoRA adapters
```

## Why ES After LoRASA

PPO/MAPPO is useful for learning from dense shaped rewards and critic-based
credit assignment, but it has drawbacks for a final fine-tuning polish:

```text
More parallel envs increase on-policy batch size.
Larger batches can make PPO updates conservative and slower.
PPO needs log-prob ratios, clipping, advantage estimation, and critic quality.
The final objective we care about is black-box team performance, especially win rate.
```

ES changes the optimization problem:

```text
sample many policy perturbations
evaluate scalar fitness
move toward perturbations with better fitness
```

This is attractive here because SMAX simulation is small and highly parallel,
while policy forward passes with role-routed adapters are likely the dominant
cost. Hyperscale ES can use large populations to better average through SMACv2
stochasticity: spawn locations, unit compositions, and aggressive heuristic
enemy behavior.

The target fitness should be aligned with the final metric:

```text
fitness = win_rate
        + small shaped-return tie-breaker
        - optional timeout penalty
        - optional ally-death penalty
```

Pure win rate may be too binary early in phase 3, so the shaped-return tie
breaker is important for ranking candidates.

## What EGGROLL Gives Us

EGGROLL is useful mainly for systems and scaling:

```text
low-rank perturbations
large batched populations
cheap low-rank forward-pass algebra
antithetic ES-style black-box updates
GPU-friendly population evaluation
```

For an unconstrained dense matrix `W`, vanilla EGGROLL samples low-rank
perturbations:

```text
E_i = (1 / sqrt(q)) U_i V_i
```

and applies an aggregate update:

```text
W_new = W + eta * sum_i weight_i * E_i
```

The aggregate of many low-rank perturbations is generally full-rank. In
vanilla EGGROLL this is a feature: low-rank samples can still produce a rich
full-matrix update over a large population.

For fixed-rank LoRA adapters, that same feature becomes a mismatch.

## Why Vanilla EGGROLL Is Not Enough for LoRA

Each role-specific adapter is represented as:

```text
Delta_{role,layer} = A_{role,layer} B_{role,layer}
```

where:

```text
A: input_dim x rank
B: rank x output_dim
Delta: input_dim x output_dim
rank(Delta) <= rank
```

The actor layer uses:

```text
W_eff(role, layer) = W_backbone(layer) + Delta_{role,layer}
```

If we apply vanilla EGGROLL aggregation directly to `Delta`, the update:

```text
Delta_new = Delta + eta * sum_i weight_i * E_i
```

will generally become higher-rank or full-rank. That breaks the fixed-rank
LoRASA representation unless we compress it afterwards.

Post-hoc compression:

```text
sample ambient perturbations
average ambient updates
compress with SVD back to rank r
```

is a plausible heuristic baseline, but it is not the clean constrained problem.
The deep-research result was clear: if the adapter must remain fixed-rank, the
adapter should be treated as a point on the fixed-rank matrix manifold.

## Fixed-Rank View

For each role/layer adapter, define the fixed-rank manifold:

```text
M_r = { X in R^{m x n}: rank(X) = r }
```

The adapter matrix is:

```text
Delta in M_r
```

with thin SVD:

```text
Delta = U S V^T
```

where:

```text
U: m x r
S: r x r
V: n x r
```

The tangent-space projector for an ambient matrix `Z` is:

```text
P_TDelta(Z) = Z V V^T + U U^T Z - U U^T Z V V^T
```

This matters because tangent vectors are valid local fixed-rank directions.
They can be averaged safely because the tangent space at the current `Delta` is
a linear space.

The rank constraint is then handled by:

```text
sample ambient low-rank noise
project it immediately to the tangent space
evaluate only retracted fixed-rank candidates
average directions in tangent space
retract back to the fixed-rank manifold
```

## Proposed Main Algorithm

Name:

```text
Riemannian LoRASA-EGGROLL
```

For each ES update:

```text
Given current role/layer adapters Delta_j = A_j B_j
where j indexes all adapted actor matrices across roles and layers.
```

1. Compute or maintain SVD coordinates:

```text
Delta_j = U_j S_j V_j^T
```

2. Sample low-rank EGGROLL-style ambient perturbations:

```text
Z_{i,j} = (1 / sqrt(q)) P_{i,j} Q_{i,j}
```

where `i` indexes the population member and `j` indexes the adapter block.

3. Project perturbations to the tangent space:

```text
u_{i,j} = P_TDelta_j(Z_{i,j})
```

4. Form antithetic fixed-rank candidates through a retraction:

```text
Delta_{i,j}^{+} = R_Delta_j(+ sigma * u_{i,j})
Delta_{i,j}^{-} = R_Delta_j(- sigma * u_{i,j})
```

The retraction `R` should initially be the rank-r metric projection via
truncated SVD:

```text
R_Delta(xi) = Pi_r(Delta + xi)
```

5. Evaluate candidate policies on common SMAX seeds:

```text
F_i^+ = fitness(policy with all Delta_{i,j}^{+})
F_i^- = fitness(policy with all Delta_{i,j}^{-})
```

Use deterministic policy evaluation by default. Exploration comes from
parameter perturbations, not action sampling.

6. Convert fitness to stable weights:

```text
Use centered ranks or baseline-adjusted coefficients.
Use common random numbers across candidates.
Optionally subtract per-seed baselines before aggregation.
```

7. Aggregate in tangent space:

```text
u_bar_j = sum_i weight_i * u_{i,j}
```

8. Update each adapter by retraction:

```text
Delta_j_new = R_Delta_j(eta * u_bar_j)
```

9. Store back as balanced LoRA factors:

```text
Delta_j_new = U_new S_new V_new^T
A_j_new = U_new S_new^{1/2}
B_j_new = S_new^{1/2} V_new^T
```

This removes arbitrary factor-scale imbalance and keeps the checkpoint
compatible with the existing LoRASA actor.

## Difference From Vanilla EGGROLL

Vanilla EGGROLL:

```text
Optimize unconstrained matrix W.
Perturb W by low-rank noise E.
Aggregate weighted E in ambient Euclidean space.
The aggregate can become full-rank, and that is allowed.
```

Riemannian LoRASA-EGGROLL:

```text
Optimize fixed-rank adapter Delta = A B.
Use low-rank EGGROLL-style noise only to generate search directions.
Project noise to the tangent space of the current adapter.
Evaluate only retracted fixed-rank candidates.
Aggregate in tangent space.
Retract back to rank r.
```

This does not contradict EGGROLL's theory. It changes the geometry because the
optimization object is different. EGGROLL's original analysis is for
unconstrained matrices; LoRA adapters are constrained fixed-rank matrices.

The clean claim should be:

```text
We adapt EGGROLL's hyperscale low-rank ES machinery to fixed-rank LoRA adapters
using Riemannian zeroth-order optimization.
```

Not:

```text
We run vanilla EGGROLL unchanged on LoRA adapters.
```

## Why Not Direct Factor-Space ES as the Main Method

Directly perturbing factors:

```text
A' = A + sigma U
B' = B + sigma V
```

induces:

```text
A'B' = AB + sigma(UB + AV) + sigma^2 UV
```

The first-order term is a local matrix movement, but the second-order term and
the search distribution depend on the arbitrary factorization.

The same matrix can be represented by many factor pairs:

```text
AB = (A R)(R^{-1} B)
```

for any invertible `R`. Isotropic factor noise is therefore not invariant to
the represented adapter matrix. It can be sensitive to factor balancing and can
waste search effort in directions that change `(A, B)` a lot while barely
changing `Delta`.

Factor-space ES can still be useful as a baseline or fallback, but it is not
the method to build the theoretical story around.

## Variance Reduction Defaults

SMACv2/SMAX is very stochastic, so variance reduction should be part of the
main design, not an afterthought.

Default choices:

```text
Antithetic pairs: evaluate +u and -u.
Common random numbers: every candidate sees the same seed bundle per update.
Centered ranks: convert raw candidate fitness to rank-based utilities.
Per-seed baseline: subtract mean return per seed across population if using a candidate x seed matrix.
Held-out eval seeds: track generalization outside the training seed bundle.
Deterministic action selection: parameter noise supplies exploration in phase 3.
```

Only increase the number of seeds per perturbation after using these variance
reducers.

## Immediate Spectral Diagnostic

Before implementing ES, run a diagnostic on trained phase-2 LoRASA checkpoints.
The goal is to check whether the adapters are healthy rank-r objects and
whether strict fixed-rank manifold optimization is numerically sensible.

For every actor LoRA adapter block:

```text
Delta = A B
```

Compute:

```text
singular_values
effective_rank
numerical_rank at several thresholds
min_nonzero_singular_value
condition_number
frobenius_norm_adapter = ||Delta||_F
frobenius_norm_backbone = ||W_backbone||_F
adapter_to_backbone_norm_ratio = ||Delta||_F / ||W_backbone||_F
lora_a_norm = ||A||_F
lora_b_norm = ||B||_F
factor_balance_ratio = ||A||_F / ||B||_F
```

Recommended effective-rank metric:

```text
p_i = s_i / sum_j s_j
effective_rank = exp(-sum_i p_i log(p_i + eps))
```

Recommended numerical ranks:

```text
rank_tol_abs_1e-8 = count(s_i > 1e-8)
rank_tol_rel_1e-4 = count(s_i / s_max > 1e-4)
rank_tol_rel_1e-3 = count(s_i / s_max > 1e-3)
```

Flag adapters where:

```text
effective_rank << configured_lora_rank
min singular value is near zero
condition number is very large
adapter/backbone norm ratio is extremely tiny or extremely large
factor balance ratio is extreme
```

Interpretation:

```text
Healthy full-rank-ish adapters:
    Riemannian fixed-rank ES is numerically plausible.

Many collapsed singular values:
    Consider lower active rank, singular-value flooring, or rank-adaptive logic.

Adapters much smaller than backbone:
    ES sigma should be small and scale-aware.

Adapters huge relative to backbone:
    Phase 3 may be brittle; use conservative sigma/eta and stronger validation.
```

## Ablations To Keep In Scope

Main comparison:

```text
A. LoRASA-MAPPO continuation
B. Riemannian LoRASA-EGGROLL
```

Important method ablations:

```text
C. Vanilla ambient EGGROLL + SVD compression
   Tests whether the manifold machinery matters.

D. Rank-expanded residual EGGROLL
   Keep existing LoRA adapters and add a new ES residual C D.
   Tests whether the original rank is too tight.

E. Shared-role perturbations vs independent role perturbations
   Tests whether role-specific black-box search is doing real work.

F. Factor-space ES baseline
   Scientifically useful, but not the main theory.
```

Do not spend too much implementation time on all ablations before confirming
that the main method has signal. The ablation list is here to preserve the
experimental story.

## Repo Scan Deferred Until Implementation

Do not inspect or port the official EGGROLL repository yet as part of this
method-design stage. When implementation begins, scan the official code for:

```text
population-axis layout
low-rank perturbation generation
antithetic pairing
RNG discipline
batched forward-pass mechanics
aggregation/update code
memory tricks
benchmarking/profiling structure
```

Use it for systems design and throughput patterns. Do not blindly copy the
ambient update rule, because the LoRASA method needs tangent projection and
rank-r retraction.

## Open Design Questions

These should be resolved after the spectral diagnostic and before full ES
implementation:

```text
1. Are the trained adapters numerically full rank?
2. Should every role/layer use the same active rank?
3. Should sigma be global, per-layer, or scaled by adapter norm?
4. Should perturbations be independent per role/layer or partially shared?
5. How many common seeds are needed per ES update?
6. How often should held-out evaluation run?
7. Do we need singular-value flooring during retraction?
8. Is the small 2r x 2r retraction practical here, or is full SVD acceptable for current layer sizes?
```

## Working Research Claim

The narrow claim to aim for:

```text
MAPPO/MACA learns a competent cooperative shared policy and provides useful
role-specialized LoRA adapters. Riemannian LoRASA-EGGROLL then performs
black-box win-rate optimization on the fixed-rank adapter manifold, preserving
LoRA structure while exploiting hyperscale low-rank ES parallelism.
```

This is intentionally narrower and cleaner than claiming that vanilla EGGROLL
can be applied unchanged to fixed-rank LoRA adapters.
