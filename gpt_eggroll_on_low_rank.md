Yes, this is the right place to be careful. The rank issue is not an implementation detail; it decides what algorithm we are actually claiming to run.

The clean theoretical formulation is:

```text
Backbone theta is frozen.
For each role k and actor layer l:
Delta_{k,l} = A_{k,l} B_{k,l}

Policy:
W_eff,k,l = W_backbone,l + Delta_{k,l}

Objective:
J(Delta) = expected win-rate / return of the whole team
```

We want to optimize `J` while keeping each `Delta_{k,l}` low-rank.

My strongest recommendation is:

```text
Matrix-space EGGROLL perturbations
+ projected/retracted update back onto the LoRA rank manifold
```

Concretely, for every role/layer adapter matrix:

```text
Delta = A B
```

For each ES candidate, sample a low-rank perturbation:

```text
E_i = (1 / sqrt(q)) U_i V_i
```

Evaluate:

```text
Delta_i = Delta + sigma * E_i
```

So during evaluation the candidate policy uses:

```text
W_eff = W_backbone + A B + sigma * E_i
```

Then estimate a matrix-space ES direction:

```text
G = sum_i fitness_i * E_i
```

With antithetic sampling:

```text
G = sum_i (F_i^+ - F_i^-) * E_i
```

Now the rank problem appears: `G` may be full-rank after summing many perturbations. We cannot just add it to `A B` unless we leave fixed-rank LoRASA.

So we do a constrained update:

```text
Delta_temp = Delta + eta * projected(G)
Delta_new  = best rank-r approximation of Delta_temp
```

The rank-r approximation is just truncated SVD:

```text
Delta_temp = U S V^T
A_new = U_r S_r^{1/2}
B_new = S_r^{1/2} V_r^T
```

This is theoretically grounded because it is basically **projected/Riemannian optimization on the low-rank matrix manifold**. EGGROLL estimates a useful ambient matrix direction; the projection/retraction says: “take the part of this improvement that our low-rank adapter class can actually represent.”

That is much cleaner than directly perturbing `A` and `B`.

Why I dislike direct factor-space ES:

```text
A B has scale symmetry: A B = (cA)(B/c)
```

So Gaussian noise on `A` and `B` depends on an arbitrary factorization scale. The same policy can have different ES behavior depending on how the factors are balanced. That is theoretically ugly and can make search ill-conditioned. It is valid black-box optimization, but the search distribution is over a parameterization artifact, not over the actual adapter matrix.

The projected matrix version searches over the real object we care about:

```text
role-specific adapter effect Delta W
```

I would make one critical addition: we should also test a **rank-expansion variant**.

Instead of forcing back to the original LoRA rank every update, we allow phase 3 to add an ES residual:

```text
W_eff = W_backbone + A B + C D
```

where `C D` is a new rank `r_es` residual learned by EGGROLL. This asks a different question:

```text
Is the remaining win-rate headroom inside the existing LoRA rank,
or does ES find improvements that need more adapter rank?
```

That is scientifically useful. If projected rank-r EGGROLL works, great. If only rank-expanded EGGROLL works, then phase 2 may be rank-limited.

So my current preferred plan is:

```text
Main method:
Projected LoRASA-EGGROLL
matrix perturbations -> ES direction -> rank-r SVD retraction

Ablation:
Rank-expanded LoRASA-EGGROLL
keep original adapters, add small ES residual rank

Avoid as main method:
direct ES on A/B factors
because factor scale symmetry muddies the theory
```

For variance, we should absolutely use:

```text
antithetic pairs
common random seeds
seed-wise fitness centering
rank-normalized fitness
held-out eval seeds
```

The exact objective should probably be:

```text
fitness = win_rate + small shaped-return tie-breaker
```

because pure win-rate will tie too often early in ES updates.

If you run deep research, this is the prompt I’d use:

```text
I am developing a phase-3 black-box fine-tuning method for a MARL policy with a frozen backbone and role-specific LoRA actor adapters. The policy has adapter matrices Delta_{role,layer}=A B. I want to use EGGROLL / low-rank evolution strategies to optimize final stochastic environment win-rate, but I need a theoretically grounded way to handle the fact that EGGROLL aggregate updates become full-rank while LoRA adapters must remain rank-r.

Research and compare:
1. Zeroth-order / evolution-strategy optimization over fixed-rank matrix manifolds.
2. Projected or Riemannian gradient methods for low-rank matrix constraints, especially using noisy black-box gradient estimates.
3. ES or zeroth-order optimization directly over LoRA factors versus over the effective low-rank matrix Delta=A B.
4. Whether matrix-space perturbation followed by tangent projection and truncated-SVD retraction is theoretically justified.
5. Alternatives such as rank expansion, low-rank residual banks, periodic SVD compression, or factor-space natural gradients.
6. Variance reduction for ES in noisy RL/MARL objectives: antithetic sampling, common random numbers, centered ranks, seed-wise baselines.
7. Practical recommendations for a JAX implementation where environment simulation is cheap and policy forward passes dominate.

Please produce equations, pros/cons, failure modes, and cite papers directly.
```

My honest position: **Projected matrix-space EGGROLL is the version I’d be comfortable defending theoretically.** Rank-expanded EGGROLL is the important ablation. Direct factor ES is a baseline, not the method I’d build the story around.