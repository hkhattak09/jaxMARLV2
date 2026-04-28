# Deep Dive: LoRASA and MACA — Section-by-Section Guide & Synthesis

> **Note on naming:** The paper you referred to as "LaRosa" is actually **LoRASA** (pronounced similarly) — **Lo**w-**R**ank **A**gent-**S**pecific **A**daptation. This document uses the correct name throughout.

---

## Part I: LoRASA (arXiv:2502.05573)
*Low-Rank Agent-Specific Adaptation (LoRASA) for Multi-Agent Policy Learning*  
*Authors: Beining Zhang, Aditya Kapoor, Mingfei Sun*

### 1. Abstract & Introduction

**Core Problem:** Multi-agent RL (MARL) typically relies on **parameter sharing (PS)** — one neural network serves all agents. This is memory-efficient but stifles specialization. In heterogeneous tasks (e.g., medics vs. marines in StarCraft), agents need distinct behaviors. Simple fixes like appending agent IDs to observations are insufficient.

**Fully distinct policies (NPS)** give each agent its own network. This allows specialization but is sample-inefficient (each network relearns common knowledge independently) and computationally expensive.

**LoRASA's Insight:** Frame MARL as a **multi-task fine-tuning** problem. Each agent is a "task" that fine-tunes a shared backbone. Instead of copying entire networks, LoRASA appends small **low-rank adaptation matrices** to each layer of the shared policy. This induces **parameter-space sparsity** — agent-specific deviations live in a tiny subspace of the full parameter space.

**Key Claim:** LoRASA matches or outperforms NPS while using a fraction of the parameters and compute.

> **Figure 1 (Overview):** Shows the LoRASA framework — a shared backbone with per-agent low-rank residuals (A_i, B_i) injected into every layer.

---

### 2. Methodology

#### 2.1 Preliminaries

- **MARL as POMG:** Partially Observable Markov Games. Each agent i observes o_i, takes action a_i via policy π_i(a_i | o_i; θ_i), and all share a global reward r_t.
- **CTDE:** Centralized Training with Decentralized Execution. During training, global info is available; at execution, each agent acts only on its local observation.
- **Joint policy factorization:** Π(a | o) = ∏_i π_i(a_i | o_i; θ_i)
- **PS vs NPS:** In PS, θ_i = θ_shared for all agents. In NPS, each θ_i is completely separate. **LoRASA sits in the middle.**
- **LoRA (Hu et al. 2021):** Adds a low-rank update δW = AB^T to a frozen weight matrix W. Only A and B are trained. A ∈ R^{d×r}, B ∈ R^{k×r}, with r ≪ min(d,k).

#### 2.2 LoRASA: Low-Rank Adaptation for MARL

**Theoretical Justification (Proposition 2.1):**
> If agent-specific parameter deviations lie within (or near) an r-dimensional affine subspace, then rank-r LoRA can approximate optimal agent-specific policies with bounded least-squares error.

This is grounded in the **Eckart-Young-Mirsky theorem** — the best rank-r approximation minimizes Frobenius norm error. In practice, policy effective dimensionality is much lower than parameter count, so low-rank residuals are sufficient.

**Weight Parameterization in the Actor:**

Consider a recurrent actor (GRU/LSTM) with FC layers. For each layer ℓ with weight matrix θ^ℓ ∈ R^{d_ℓ × k_ℓ}:

```
θ_i^ℓ = θ_shared^ℓ + A_i^ℓ B_i^{ℓT}

where:
  θ_shared^ℓ  := shared backbone (FROZEN in Phase 2)
  A_i^ℓ ∈ R^{d_ℓ × r}
  B_i^ℓ ∈ R^{k_ℓ × r}
  r         := rank (typically 8)
```

**What gets adapted:**
- **Linear transformations** in the recurrent pathway (input-to-hidden, hidden-to-hidden)
- **Final FC layers** that output:
  - **Continuous actions:** mean and log-std of a squashed Gaussian
  - **Discrete actions:** action logits

**What stays frozen:**
- Biases
- Layer normalization parameters
- The shared backbone θ_shared^ℓ

This means the **entire actor network** is modified via low-rank weight residuals, not just the output head.

> **Critical detail:** The forward pass uses the **merged** weight θ_i^ℓ = θ_shared^ℓ + A_i^ℓ B_i^{ℓT}. The residual is on the **weights**, not the activations. At inference, you can pre-merge the adapters into the backbone for each agent (Algorithm 3).

#### 2.3 Training Procedure — THE TWO PHASES

This is the most important section for your questions about freezing and residuals.

**Phase 1: Shared Policy Pretraining**

```
Input: N agents, Environment, MARL algorithm (MAPPO / A2PO)
Output: Pretrained θ_shared

for step = 1 to P_steps:
    Collect joint trajectories from env
    θ_shared ← Algorithm.update_shared(θ_shared, trajectories)
end for
```

- All agents use **exactly the same policy** (pure PS).
- Track cumulative returns and win rates.
- Continue until the policy shows **consistent improvement** and meets a performance threshold.
- **This Phase 1 is NOT optional.** It establishes common coordination knowledge.

**Phase 2: LoRA Fine-Tuning — THIS IS WHEN THE ACTOR IS FROZEN**

```
Input: pretrained θ_shared, rank r, fine-tuning steps F_steps
Output: per-agent LoRA adapters {A_i^ℓ, B_i^ℓ}

Introduce LoRA adapters A_i^ℓ, B_i^ℓ for each agent i, each layer ℓ
FREEZE θ_shared entirely

Initialize:
    A_i^ℓ ← 0          (zero initialization is common in LoRA)
    B_i^ℓ ← random

for step = 1 to F_steps:
    Collect trajectories
    for each agent i:
        Update ONLY (A_i^ℓ, B_i^ℓ) using Algorithm.update_agent_lora()
        θ_shared remains completely untouched
    end for
end for
```

**What happens mathematically:**

```
∀ℓ: θ_i^ℓ = θ_shared^ℓ + A_i^ℓ B_i^{ℓT}

Gradients flow through:
  ∂L/∂A_i^ℓ = (∂L/∂θ_i^ℓ) · B_i^ℓ
  ∂L/∂B_i^ℓ = (∂L/∂θ_i^ℓ)^T · A_i^ℓ

∂L/∂θ_shared^ℓ = 0  (frozen, stop-gradient)
```

**Rank r controls the "distance" from PS to NPS:**
- r = 0 → pure PS (no adaptation)
- Small r (e.g., 4) → minimal specialization, high efficiency
- Moderate r (e.g., 8) → **sweet spot** for most tasks
- Large r (e.g., 16, 64) → approaches NPS, but can overfit or converge slower

#### 2.4 Algorithms

**Algorithm 1 — Phase 1 (Shared Pretraining):** Standard CTDE MARL. No LoRA overhead.

**Algorithm 2 — Phase 2 (LoRA Fine-Tuning):**
- Freeze θ_shared
- Introduce per-agent A_i^ℓ, B_i^ℓ
- Train only adapters with agent-specific trajectories

**Algorithm 3 — Inference (Merge for Speed):**
```
for each agent i:
    for each layer ℓ:
        θ_i^ℓ ← θ_shared^ℓ + A_i^ℓ B_i^{ℓT}   # merge once
    select action using θ_i
```

At inference, after merging, each agent has its own full parameter set θ_i, but the merge is a one-time O(r·d·k) operation per layer.

#### 2.5 Computational & Memory Efficiency

**Pretraining:** Identical to PS. Zero overhead.

**Fine-tuning:** Each agent adds ∑_ℓ r(d_ℓ + k_ℓ) parameters. For typical networks, this is ~1-5% of full network parameters.

**Inference:** After merging, memory footprint is essentially one network per agent, but the "extra" memory is just the merged weights. If you keep A,B separate, it's even smaller.

---

### 3. Experimental Setup & Results

#### 3.1 Environments
- **MAMuJoCo** (continuous): Half Cheetah 2×3, Walker 3×2, Ant 4×2, Humanoid 9|8
- **SMAC** (discrete): 3s5z, 1c3s5z, 3s5z_vs_3s6z, MMM2

SMAC (not SMACv2) is used because agent assignments are consistent — critical for training agent-specific parameters.

#### 3.2 Baselines
- **PS + ID:** Shared policy with agent ID appended to obs
- **NPS:** Fully separate networks per agent
- **SePS:** Selective parameter sharing (cluster agents)
- **MTL:** Multi-task learning, typically only final layer specialized

LoRASA variants: **PS+LoRA** and **SePS+LoRA**.

#### 3.3 Key Results
- **Figure 2 (Performance):** LoRASA frequently outperforms PS and matches/surpasses NPS on many tasks, at far lower parameter cost.
- **Figure 3 (Resource Efficiency):**
  - (1) Memory footprint: LoRASA is barely above PS, far below NPS
  - (2) Scalability: NPS parameters grow linearly with agents; LoRASA grows moderately
  - (3,4) Training/inference speed: LoRASA is much faster than NPS

---

### 3.5 Ablation Studies — CRITICAL FOR YOUR QUESTIONS

> **Figure 4** shows ablations on (A-D) Timing, (E-H) Rank, (I-L) Layer Placement.

#### A. Early vs. Late Fine-Tuning — WHEN TO FREEZE AND ADD RESIDUAL

This directly answers: **"When should the actor be frozen and the residual added?"**

**Findings:**

1. **Ant 4×2 (MAMuJoCo):**
   - **A2PO:** Switching to LoRA at **4×10⁶ steps** works best.
   - **MAPPO:** Switching at **4×10⁶ steps** works best.

2. **MMM2 (SMAC):**
   - **A2PO:** Switching at **2×10⁶ steps** is optimal.
   - **MAPPO:** Switching at **7×10⁶ steps** is optimal. Earlier switches only match PS performance.

**Interpretation:**
- Introduce LoRA when the shared policy exhibits **competent but non-plateaued performance**.
- Too early: The shared backbone hasn't learned useful coordination primitives; LoRA has nothing meaningful to specialize.
- Too late: The shared policy may have overfit to homogeneous behavior; less room for specialization to help.
- **MAPPO on complex tasks (MMM2) needs MORE pretraining** before LoRA helps. This suggests MAPPO forms a less robust foundation than A2PO on that map, so you must wait longer.

**Practical Guideline:**
> **Start LoRA fine-tuning at the checkpoint where shared-policy returns/win-rates show steady improvement but have not yet saturated.** For most SMAC tasks with MAPPO, this is between 200K–7M steps depending on difficulty. For MAMuJoCo, 2M–4M steps is typical.

**Exact Checkpoints from Paper:**

| Scenario | A2PO PS+LoRA Checkpoint | MAPPO PS+LoRA Checkpoint |
|---|---|---|
| Halfcheetah 2×3 | 3.0M | 100K |
| Walker 3×2 | 2.0M | 2.0M |
| Ant 4×2 | 4.0M | 4.0M |
| Humanoid 9\|8 | 3.0M | 1.0M |
| 3s5z | 200K | 200K |
| 1c3s5z | 500K | 500K |
| 3s5z_vs_3s6z | 2.0M | 4.0M |
| MMM2 | 2.0M | **7.0M** |

#### B. Rank r — HOW BIG SHOULD THE RESIDUAL BE?

**Findings:**
- **r = 8** is the sweet spot across most tasks.
- **r = 4** is sometimes insufficient for nuanced behaviors.
- **r = 16 or 64 (full rank)** can lead to **slower convergence or overfitting**.

**Interpretation:**
> Agent diversity lives in a surprisingly small subspace. r=8 captures almost all the benefit. Higher ranks add parameters without proportional gains and harm regularization.

**Practical Guideline:**
> **Start with r = 8.** Only increase to 16 for highly heterogeneous tasks (e.g., Humanoid 9|8, 3s5z_vs_3s6z). Never use full rank — it defeats the purpose.

#### C. Adapter Placement — WHERE TO ADD THE RESIDUAL

**Findings:**
- **Adapting ALL layers** generally performs best.
- **Only final layer:** Strong but not top-tier.
- **Only early layers:** Minimal impact.

**Interpretation:**
- Early layers extract low-level features (positions, unit types) that are largely universal.
- Mid-to-high layers make strategic decisions (targeting, coordination) where agents diverge.
- However, adapting ALL layers is still best because even early layers benefit from slight agent-specific tuning when combined with deeper adaptations.

> **This runs counter to MTL methods that only adapt the last few layers.** LoRASA's full-network low-rank adaptation is uniquely effective.

**Practical Guideline:**
> **Apply LoRA to all linear/FC layers in the actor**, including recurrent connections and the final output layer. Do NOT restrict it to only the policy head.

---

### Appendix Insights

**A.1 Pseudocode:** Confirms the two-phase procedure. Phase 2 explicitly freezes θ_shared.

**A.3 Heterogeneous Nature of Agent Policies:**
- Activation heatmaps show early layers are similar across agents; later layers diverge significantly.
- Wasserstein distances between policy distributions confirm agents of different roles are far apart, while same-role agents are closer but still distinct.

**A.3 Sparsity Analysis:**
- LoRA-adapted parameters have much smaller magnitudes than shared backbone parameters.
- This confirms LoRA acts as a **lightweight, sparse residual** that nudges behavior rather than rewriting it.

**A.4 Hyperparameters:**
- Actor LR during LoRA fine-tuning: **3×10⁻⁴** (same as pretraining for most tasks)
- Clip parameter: 0.2 (standard PPO)
- Epochs: 3-8 depending on task

---

## Part II: MACA (arXiv:2508.06836)
*Multi-level Advantage Credit Assignment for Cooperative Multi-Agent Reinforcement Learning*  
*Authors: Xutong Zhao, Yaqi Xie*

### 1. Abstract & Introduction

**Core Problem:** Credit assignment in MARL — determining each agent's contribution to the shared reward. Existing methods assume a **fixed level** of cooperation:
- **Implicit methods** (QMix, VDN): Decompose joint value function.
- **Explicit methods** (COMA): Counterfactual baselines for individual agents.

**Problem with fixed levels:** Real tasks involve **multiple coexisting levels** of cooperation. Agent A might cooperate with B and C to carry a fridge (3-level), while simultaneously carrying a backpack alone (1-level). The global reward is a sum over all subsets: r(s,a) = ∑_{G⊂N} r_G(s, a_G).

**MACA's Insight:** Formalize credit assignment **level** as the number of agents cooperating to obtain a reward. Propose a **multi-level advantage** that combines counterfactual baselines at different levels.

**Architecture:** Uses a **transformer encoder** to discover which agents are strongly correlated (CorrSet), then builds advantages for individual, joint, and correlated-subset actions.

---

### 2. Related Work

**Implicit Credit Assignment:**
- VDN: linear sum of individual values
- QMix: monotonic mixing network (more expressive but still constrained)
- HAPPO: sequential policy updates based on multi-agent advantage decomposition (scalability issues)

**Explicit Credit Assignment:**
- Difference rewards / COMA: counterfactual baseline marginalizing out the current agent
- Shapley Value: theoretically sound but factorial complexity
- MAPPO/MAA2C: use joint advantage, don't distinguish individual credits

**Attention in MARL:**
- Decision Transformer extensions, hide-and-seek, graph-attention communication
- **MACA is novel** in using attention-captured correlations **for credit assignment**.

---

### 3. Preliminaries

#### 3.1 Dec-POMDP
Standard tuple ⟨N, S, A, P, R, γ, O, Ω⟩. Global reward r is shared.

#### 3.2 Policy Gradient Methods

**Multi-Agent Policy Gradient (MAPG):**

```
g_{θ_i} = ∇_{θ_i} J(θ) = E_{s~d^π, a~π}[ Q(s,a) ∇_{θ_i} log π_i(a_i | s) ]
```

In actor-critic, we use a baseline b to reduce variance. Any **action-independent** baseline preserves unbiasedness.

**CTDE:** Centralized critic learns V(s) or Q(s,a) from global state; decentralized actors execute from local obs.

#### 3.3 Credit Assignment Level

**Formal definition:**
- Let G ⊂ N be a subset of k = |G| agents.
- r_G(s^t, a^t) = r_G(s^t, a_G^t) is the reward obtained only if subset G cooperates.
- Global reward: r(s,a) = Σ_{G⊂N} r_G(s, a_G)
- **k is the credit assignment level.**

A single agent can be involved in multiple levels simultaneously.

---

### 4. Method

#### 4.1 Multi-Level Counterfactual Formulation

**k-Level Counterfactual Baseline:**

For agent i, define a subset G_i containing i with |G_i| = k. The k-level baseline marginalizes out actions of agents in G_i:

```
b^{CF}_i(s,a) = E_{a_{G_i}}[ Q(s,a) ]

where a_{G_i} = {a_j ~ π_j : j ∈ G_i}
```

The resulting advantage:
```
A_i(s,a) = Q(s,a) - b^{CF}_i(s,a)
```

This generalizes:
- **COMA** (k=1): b^{Ind}_i = E_{a_i}[Q(s,a)] — reasons about individual contribution
- **MAPPO** (k=n): b^{Jnt} = V(s) = E_{a~π}[Q(s,a)] — reasons about joint action vs default

**CorrSet Baseline (b^{Cor}_i):**
- Not all k-agent subsets are equally important.
- Use attention weights to find agents **strongly correlated** with agent i.
- C_i = CorrSet(i) = {j : Ã_{i,j} ≥ σ} ∪ {i}
- This is dynamic — changes based on state.

**Multi-Level Advantage (THE MACA ADVANTAGE):**

Combine three baselines with learned state-dependent weights:

```
b^{MACA}_i = ψ^{Jnt}_i · b^{Jnt} + ψ^{Ind}_i · b^{Ind}_i + ψ^{Cor}_i · b^{Cor}_i

A^{MACA}_i = Q(s,a) - b^{MACA}_i
```

where [ψ^{Jnt}, ψ^{Ind}, ψ^{Cor}] ∈ Δ(3) (a probability simplex, i.e., softmaxed weights).

**Key Property:** Each component baseline marginalizes out a_i, so the combined baseline is **action-independent** for agent i. By Lemma A.1, this preserves **unbiasedness** of the MAPG estimate.

**Convergence:** Lemma A.4 proves MACA converges to a local optimum under standard actor-critic assumptions.

#### 4.2 Attention-based Framework

**Figure 1 (MACA Critic Architecture):**
- Input: sequence of agent observations (o_1, ..., o_n)
- Embedding layer + M transformer encoder blocks (self-attention + MLP + residual + layer norm)
- Output: state embedding z_s
- Attention rollout weights Ã from last layer capture inter-agent correlations

**Value Estimation:**
To compute E_{a_{G_i}}[Q(s,a)] efficiently, MACA uses:
```
b^{CF}_i(s,a) = Q_φ(s, π̄_{G_i})
```
where π̄_{G_i} is the marginalized action distribution:
- For j ∈ G_i: use policy distribution π_j
- For j ∉ G_i: one-hot at the taken action

**Important:** In general E[Q(s,a)] ≠ Q(s, E[a]). MACA addresses this by feeding z_s and π̄_{G_i} through a **linear layer** to ensure equality (related to Jensen's inequality).

**CorrSet Construction:**
```
j ∈ C_i  iff  Ã_{i,j} ≥ σ

Always enforce i ∈ C_i
```
σ ∈ [0,1] is a threshold hyperparameter.

**Learning the Weights ψ:**
- The MAPG objective is **not differentiable** w.r.t. ψ (advantages affect policy updates, but through sampling).
- TD updates also don't optimize ψ.
- Solution: **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) optimizes:
  ```
  L(η) = -E_τ[ R(θ^{n+1}) - R(θ^n) ]
  ```
  where η parameterizes ψ via a linear layer + softmax.

**Training Losses:**
```
L_V(φ) = E[ ||V(s^t) - sg(r^t + γV(s^{t+1}))||² ]
L_Q(φ) = E[ ||Q(s^t,a^t) - sg(r^t + γQ(s^{t+1},a^{t+1}))||² ]
```
where sg = stop-gradient.

---

### 5. Experiments

#### 5.1 Setup
- **Baselines:** MAPPO, IPPO, HAPPO, COMA, PPO-Mix (QMix+PPO), PPO-Sum (VDN+PPO)
- **Environments:** SMACv1 (5 tasks), SMACv2 (6 tasks), MPE (3 tasks)
- **Training:** 8M steps (SMACv1), 10M steps (SMACv2), 5 seeds
- **Fair comparison:** All methods built on MAPPO skeleton; only advantage function differs

#### 5.2 Results

**Table 2 (SMACv1 & v2 Win Rates):**
- MACA achieves **superior overall performance**, especially on **SMACv2** (more stochastic, harder).
- On SMACv2, MACA shows **much higher sample efficiency** — improves faster while others plateau.
- Example gains:
  - protoss_5_vs_5: MACA 79.0% vs MAPPO 56.5%
  - terran_10_vs_10: MACA 75.0% vs MAPPO 40.0%
  - zerg_10_vs_10: MACA 62.9% vs MAPPO 39.8%

#### 5.3 Ablations (Table 3, Figure 3)

| Variant | Description | Performance |
|---|---|---|
| MACA | Full method | Best |
| MACA-Jnt | Only joint baseline (≈ MAPPO + transformer) | Worse than MACA |
| MACA-Cor | Only CorrSet baseline | Worse than MACA |
| MACA-Ind | Only individual baseline (≈ COMA + transformer) | Very poor |
| MACA-NoCor | Remove CorrSet | Degraded |
| MACA-NoInd | Remove individual | Degraded |
| MACA-NoJnt | Remove joint | Degraded |
| MACA-Dec | Replace linear layer with transformer decoder | Comparable to MACA |

**Key Takeaway:** Every level matters. No single baseline suffices. The multi-level combination is essential.

---

### Appendix A: Theoretical Results

**Lemma A.1:** Action-independent baselines do not bias the policy gradient.  
**Theorem A.2:** The minimum-variance baseline for MAPG is a weighted conditional expectation.  
**Lemma A.3:** The k-level baseline can be expressed as the minimum-variance baseline minus a covariance term, making it an **optimistic baseline** (justified by Chung et al. 2021).  
**Lemma A.4:** MACA converges to a local optimal policy under standard AC assumptions.

---

## Part III: Connecting the Dots — LoRASA + MACA

### The Core Synthesis

**LoRASA and MACA solve complementary problems in the same CTDE actor-critic pipeline:**

| Component | LoRASA | MACA |
|---|---|---|
| **What it modifies** | The **Actor** (policy network) | The **Critic** (advantage/baseline computation) |
| **Problem it solves** | Agents need heterogeneous policies | Credit assignment is ambiguous |
| **Mechanism** | Low-rank weight residuals per agent | Multi-level counterfactual baselines |
| **When it applies** | After shared pretraining | During every policy gradient update |
| **Key parameter** | Rank r (typically 8) | CorrSet threshold σ, ψ weights |

**They are not competitors. They are synergistic.**

---

### How to Use LoRASA "as a Residual After the Current Actor"

You asked specifically about using LoRASA as a residual **after** the current actor. Let's clarify what this means both architecturally and procedurally.

#### Architectural View: Weight-Space Residual

LoRASA **is** a residual, but it lives in **weight space**, not activation space:

```
Effective weight for agent i at layer ℓ:
    θ_i^ℓ = θ_shared^ℓ + δθ_i^ℓ
    where δθ_i^ℓ = A_i^ℓ B_i^{ℓT}   ← THE RESIDUAL

Forward pass of agent i's actor:
    h^{ℓ+1} = φ( (θ_shared^ℓ + A_i^ℓ B_i^{ℓT}) · h^ℓ )
```

This is **not** an output residual like: `output = f(x) + g(x)`.  
It is a **parameter residual** like: `f_i(x) = f_shared(x; θ_shared + δθ_i)`.

**Why this matters:** By constraining δθ_i to be low-rank, you ensure the agent-specific "delta" is small in terms of degrees of freedom, but can still reshape the network's behavior because it's applied at every layer.

#### Procedural View: Two-Phase Training

**Phase 1 — No Residual (Pure Shared Actor):**
```
Train θ_shared with all agents sharing it
→ This is your "current actor" before specialization
```

**Phase 2 — Add Residual (Freeze Actor, Train Residual):**
```
1. FREEZE θ_shared completely (no gradients, no updates)
2. Initialize A_i^ℓ = 0, B_i^ℓ = random for each agent i, each layer ℓ
3. Effective actor for agent i becomes: θ_shared + A_i B_i^T
4. Continue training, but gradients only update A_i and B_i
```

**This is literally "using LoRASA as a residual after the current actor":**
- The "current actor" = θ_shared (pretrained, frozen)
- The "residual" = A_i B_i^T (trained to specialize each agent)
- The combined actor = θ_shared + residual

#### Why Freeze the Shared Actor?

If you don't freeze θ_shared, you get:
```
θ_i^ℓ = θ_shared^ℓ + A_i^ℓ B_i^{ℓT}
∂L/∂θ_shared = Σ_i ∂L/∂θ_i   ← shared backbone gets pulled in conflicting directions
```

Without freezing:
1. The shared backbone drifts to accommodate all agents simultaneously
2. This destroys the common knowledge that Phase 1 learned
3. The low-rank residual loses its "reference point"
4. You effectively revert to a weird form of PS, not specialization

**Freezing is essential.** It anchors the common behavior while the residual learns the deviation.

---

### When Exactly to Freeze and Add the Residual

Based on the LoRASA ablations, here is a concrete decision framework:

#### Rule 1: Wait for Competent but Non-Plateaued Performance

Do NOT add LoRA at step 0. Do NOT wait until full convergence.

**Optimal window:**
- Shared policy should reliably complete the task basics
- Win rate / return curves should be in the **rising phase**
- Not at the flat plateau

**Concrete heuristics:**
```
If win rate < 20%:     TOO EARLY. Keep pretraining.
If win rate 20-70%:    SWEET SPOT. Introduce LoRA now.
If win rate > 90%:     TOO LATE. Specialization won't help much.
```

#### Rule 2: Algorithm-Dependent Timing

| Algorithm | Task Complexity | Typical Checkpoint |
|---|---|---|
| A2PO | Low-Medium | 2M–4M steps |
| A2PO | High (MMM2) | ~2M steps |
| MAPPO | Low-Medium | 2M–4M steps |
| MAPPO | High (MMM2) | **~7M steps** |

**Why does MAPPO need longer on MMM2?**
- MAPPO updates all agents simultaneously with a joint advantage
- It struggles to learn diverse behaviors early on (see Figure 2P: MAPPO NPS fails on MMM2)
- The shared backbone needs more time to become a robust foundation before residuals can refine it
- A2PO's sequential updates naturally encourage more diversity earlier

#### Rule 3: Task-Dependent Timing

| Task Type | When to Switch |
|---|---|
| Homogeneous, simple coordination | Earlier (shared policy converges fast) |
| Heterogeneous, distinct roles | Later (shared policy needs time to learn basics) |
| Continuous control (MAMuJoCo) | 2M–4M steps typical |
| Discrete combat (SMAC) | 200K–7M steps depending on map |

**Specific values from the paper:**
- 3s5z: **200K steps** (easy map, switch early)
- 1c3s5z: **500K steps**
- Ant 4×2: **4M steps**
- MMM2 (MAPPO): **7M steps** (hardest for MAPPO)

#### Rule 4: Monitor and Adapt

If you don't know the optimal checkpoint:
1. Pretrain for X steps
2. Save checkpoints every 500K–1M steps
3. From each checkpoint, spawn a LoRA fine-tuning run for 2M–4M steps
4. Compare final performance
5. Select the checkpoint with the best final performance after fine-tuning

This is expensive but robust. In practice, the paper shows that a wide range of checkpoints work reasonably well; only extreme early/late choices are harmful.

---

### How to Use LoRASA Effectively: Best Practices

#### 1. Rank Selection
- **Default:** r = 8 for all layers
- **Increase to 16** only for:
  - Very high-dimensional action spaces
  - Extremely heterogeneous agent types (e.g., Humanoid 9|8)
- **Never use r = 64/full rank** unless you explicitly want NPS

#### 2. Layer Placement
- **Apply LoRA to ALL linear/FC layers**, including:
  - Input projections to RNN
  - Hidden-to-hidden RNN transitions
  - Output mean/log-std (continuous) or logits (discrete)
- **Do NOT apply to:** biases, layer norm parameters (simpler, minimal downside)
- **Do NOT restrict to final layer only** — mid-layer adaptation is crucial for role differentiation

#### 3. Initialization
```python
# Standard LoRA initialization
A_i = zeros(d, r)      # Zero init ensures training starts from shared policy
B_i = randn(k, r) * 0.01  # Small random init breaks symmetry
```
With A=0, the initial effective policy is exactly the shared policy. Training gradually "turns on" the residual.

#### 4. Learning Rate
- Keep the same actor LR as pretraining (typically 3×10⁻⁴)
- Do NOT reduce LR for LoRA — the small parameter count already regularizes
- Optionally use slightly higher LR for LoRA (5×10⁻⁴) if adaptation is slow

#### 5. Inference
- **Option A (memory-efficient):** Keep A_i, B_i separate. Compute on-the-fly.
  - Memory: O(r(d+k)) per agent per layer
  - Compute: O(r·d·k) per forward pass (small if r is tiny)
- **Option B (speed-optimized):** Merge once at load time:
  ```python
  θ_i_merged = θ_shared + A_i @ B_i.T
  ```
  - Then use standard forward pass with θ_i_merged
  - Memory becomes O(d·k) per agent (full network), but no runtime overhead

#### 6. What About the Critic?

**The LoRASA paper keeps the critic SHARED and unmodified.**
- Centralized critic continues to use θ_shared
- It sees all agent observations/actions
- This is fine because the critic is centralized — it doesn't need to be lightweight

**However**, if you also want agent-specific value estimation, you could theoretically apply LoRA to individual utility heads (as hinted in the paper's Discussion section). But this is not evaluated.

---

### How MACA Enhances LoRASA Training

Now we connect Paper 2 to Paper 1. If you run LoRASA with a standard MAPPO critic, you are using a **joint-level advantage** (k=n) for all agents during Phase 2. This has a subtle problem:

**Problem:** During LoRA fine-tuning, agents are becoming MORE heterogeneous. But MAPPO's joint advantage `A_i = Q(s,a) - V(s)` treats all agents the same — it only evaluates whether the joint action was good, not which agent's specialization contributed what.

**Result:** The credit assignment signal for updating A_i, B_i is noisy. Agent i's residual gets updated based on team performance, not individual contribution.

**MACA fixes this.** By using MACA's multi-level advantage during Phase 2:

```
A^{MACA}_i = Q(s,a) - [ψ^{Jnt}·b^{Jnt} + ψ^{Ind}·b^{Ind}_i + ψ^{Cor}·b^{Cor}_i]
```

The policy gradient for agent i's LoRA parameters becomes:
```
∇_{A_i} J = E[ A^{MACA}_i · ∇_{A_i} log π_i(a_i | o_i; θ_shared + A_i B_i^T) ]
```

**Benefits:**
1. **Individual baseline (b^{Ind}):** Isolates agent i's own contribution, so its residual updates only when ITS actions (not teammates') improve the outcome.
2. **CorrSet baseline (b^{Cor}):** Identifies which teammates agent i actually coordinates with. If agent i's residual creates a new coordination pattern with agent j, MACA notices via the attention-based CorrSet and credits them appropriately.
3. **Joint baseline (b^{Jnt}):** Preserves the global coordination signal so agents don't become selfish.

**In short:**
> **LoRASA gives agents the CAPACITY to specialize. MACA gives them the CREDIT SIGNAL to learn HOW to specialize.**

---

### Recommended Combined Architecture

```
ACTOR (Decentralized, per-agent):
  Input: o_i (local observation)
  Layer 1: h_1 = ReLU( (W_shared^1 + A_i^1 B_i^1T) · o_i )
  ...
  RNN: h_t = GRU( (W_x + A_i^x B_i^xT) · h_1,
                  (W_h + A_i^h B_i^hT) · h_{t-1} )
  ...
  Output: μ, logσ = (W_out + A_i^out B_i^outT) · h_t
  Action: a_i ~ tanh(Normal(μ, σ))

CRITIC (Centralized, shared):
  Input: [o_1, o_2, ..., o_n] (all observations)
  Embedding + Transformer Encoder → attention weights Ã
  State embedding z_s
  CorrSet C_i = {j : Ã_{i,j} ≥ σ} ∪ {i}
  
  Compute baselines:
    b^{Jnt}    = Q(z_s, π̄)               # all actions marginalized
    b^{Ind}_i  = Q(z_s, π̄_i)             # only agent i marginalized
    b^{Cor}_i  = Q(z_s, π̄_{C_i})         # CorrSet marginalized
  
  MACA baseline: b^{MACA}_i = ψ_i^T · [b^{Jnt}, b^{Ind}_i, b^{Cor}_i]
  Advantage: A^{MACA}_i = Q(s,a) - b^{MACA}_i

TRAINING:
  Phase 1 (0 to T_ckpt):
    Train W_shared with MAPPO (A_i, B_i do not exist)
    Critic uses standard joint advantage (or MACA-Jnt)
  
  Phase 2 (T_ckpt onwards):
    FREEZE W_shared
    Initialize A_i = 0, B_i = random
    Train A_i, B_i using MACA advantage A^{MACA}_i
    Critic continues to train (or freeze if desired)
```

---

### Summary Checklist for Implementation

| Decision | Recommendation |
|---|---|
| **Residual type** | Weight-space residual: θ_i = θ_shared + A_i B_i^T |
| **Rank r** | 8 (default), 16 for complex tasks |
| **Layers to adapt** | All linear/FC + RNN transitions + output head |
| **Freeze shared actor?** | **YES, absolutely.** Freeze at Phase 2 start |
| **When to freeze** | When shared policy shows competent, rising performance (20-70% win rate) |
| **Checkpoint examples** | 200K (easy SMAC), 2-4M (MAMuJoCo), 7M (hard SMAC with MAPPO) |
| **Initialization** | A = 0, B = small random |
| **Learning rate** | 3×10⁻⁴ (same as pretraining) |
| **Critic** | Use MACA multi-level advantage for best credit assignment |
| **Inference** | Pre-merge weights for speed, or keep separate for memory |
| **What NOT to do** | Don't adapt biases/LN; don't use full rank; don't freeze too early/late |

---

## Part IV: Key Figures & Their Meanings

### LoRASA Figures

| Figure | What It Shows |
|---|---|
| **Fig 1** | Framework overview: shared backbone + per-agent low-rank matrices |
| **Fig 2** | Performance curves: LoRASA matches NPS, beats PS across MAMuJoCo & SMAC |
| **Fig 3** | Resource efficiency: memory, scalability, speed comparisons |
| **Fig 4A-D** | **Timing ablation:** when to start LoRA (YOUR QUESTION) |
| **Fig 4E-H** | **Rank ablation:** r=8 is sweet spot |
| **Fig 4I-L** | **Placement ablation:** all layers > final layer only |
| **Fig 5** | SMAC evaluation rewards |
| **Fig 6-7** | Wasserstein distances showing policy heterogeneity |
| **Fig 8** | Sparsity: LoRA parameters are small-magnitude residuals |
| **Fig 9-20** | Layer activation heatmaps: early layers similar, late layers diverge |

### MACA Figures

| Figure | What It Shows |
|---|---|
| **Fig 1** | Critic architecture: transformer encoder → attention → multi-level baselines |
| **Fig 2** | SMACv2 learning curves: MACA dominates on stochastic tasks |
| **Fig 3** | Ablation: every baseline component is necessary |
| **Table 2** | Win rates: MACA SOTA on SMACv2, competitive on SMACv1 |
| **Table 3** | Ablations: removing any level degrades performance |

---

## Closing Thoughts

**LoRASA** provides a principled, efficient mechanism for agent specialization via **low-rank weight residuals**. Its two-phase training — first learn shared coordination, then freeze and specialize — is simple but requires careful timing. The ablations give clear guidance: **start LoRA when performance is rising but not saturated, use rank 8, adapt all layers, and keep the shared backbone frozen.**

**MACA** solves the credit assignment problem that becomes MORE severe as agents specialize. When agents play different roles, joint advantages become noisy. MACA's multi-level formulation — combining joint, individual, and correlation-based baselines — gives each agent a cleaner learning signal.

**Together, they form a powerful combination:** LoRASA creates the architectural capacity for heterogeneous policies; MACA creates the learning signal to train them effectively. If you are implementing this, train MAPPO with MACA's critic for Phase 1, then switch to LoRA-adapted actors with MACA advantages for Phase 2. This is not explicitly tested in either paper, but the theoretical and empirical foundations strongly support it.
