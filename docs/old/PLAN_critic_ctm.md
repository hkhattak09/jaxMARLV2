# Plan: CTM as Coordination Module in Centralized Critic

## Core Idea

Actors use standard GRU (proven baseline). A global CTM in the critic processes
all agents' GRU hidden states, producing a sync matrix that captures pairwise
coordination structure. This sync matrix drives per-agent value decomposition
for better credit assignment.

## Mathematical Formulation

### Setup
- N agents with GRU actors producing hidden states h_1, ..., h_N
- Global CTM processes all h_i, producing:
  - Per-agent activated states: z_1, ..., z_N in R^d
  - Pairwise sync matrix: S in R^{NxN}, where S_ij = cos_sim(sync_i, sync_j)

### Per-Agent Value Decomposition

Standard MAPPO: V(s) — single global value, same advantage for all agents.

Proposed:

```
V_i(s) = f_theta( z_i, sum_j S_ij * z_j )
A_i = G_t - V_i(s_t)
```

Each agent's value baseline is a function of:
1. Its own CTM-processed state z_i
2. A coordination-weighted aggregation of other agents' states

**Properties:**
- S = I (no coordination) -> independent per-agent critics
- S = uniform -> standard centralized critic (all agents weighted equally)
- S = learned -> adaptive credit assignment from data

**Why this reduces variance:** In standard MAPPO with shared reward, A = G - V(s)
is identical for all agents. The policy gradient sum_i A * grad log pi_i(a_i|o_i)
has high variance because each agent is credited/blamed equally regardless of
contribution. Per-agent A_i from coordination-weighted baselines assigns credit
proportional to each agent's coordination role.

---

## Architecture

```
Actors (decentralized, unchanged):
  o_i -> embedding -> GRU -> h_i -> policy head -> pi(a_i|o_i)

Critic (centralized, new):
  All h_i -> Global CTM -> sync matrix S, activated states z_i
  Per agent: V_i = MLP( concat(z_i, sum_j S_ij * z_j) )
```

The global CTM sees all agents simultaneously. Its internal pairwise sync step
naturally computes inter-agent coordination structure. The activated states z_i
are agent-specific representations enriched by the CTM's iterative processing.

### Global CTM Details

The CTM takes as input all N agents' GRU hidden states (stacked). The sync step
computes pairwise similarities across agents (not within-agent as before). This
is the key architectural change: sync now operates at the multi-agent level.

Input: H = [h_1, ..., h_N] in R^{N x d_gru}
CTM backbone projects to: X = [x_1, ..., x_N] in R^{N x d_model}
Sync step: S_ij = cos_sim(sync(x_i), sync(x_j))
Output: activated states z_i enriched by inter-agent sync

### Critic Loss

Standard value loss with per-agent values:

```
L_critic = (1/N) sum_i ( V_i(s_t) - G_t )^2
```

With shared reward, G_t is the same for all agents, but V_i differs per agent
based on coordination structure. The critic learns to predict which agents will
contribute to the shared outcome.

---

## Implementation Steps

### Step 1: Global CTM Module
- New class GlobalCTM that takes stacked agent hidden states
- Reuse CTMCell internals but operate over agent dimension
- Output: per-agent z_i and sync matrix S

### Step 2: Modified CriticRNN
- After GRU, collect all agents' hidden states
- Pass through GlobalCTM
- Per-agent value head: MLP(concat(z_i, S_i @ Z))

### Step 3: Per-Agent Advantages
- Modify GAE computation to use per-agent V_i instead of shared V
- Each agent gets its own advantage A_i

### Step 4: Training Loop Changes
- Store per-agent values in Transition
- Update critic loss to use per-agent targets
- Actor loss uses per-agent advantages

---

## Key Design Decisions To Resolve

1. **How does the global CTM see all agents?** The current CTM operates per-agent.
   Need to either: (a) stack all h_i and run CTM over agent dim, or (b) run CTM
   per-agent then compute cross-agent sync. Option (a) is cleaner.

2. **Sync across agents vs across time?** The current sync computes pairwise
   similarities within one agent's internal state. For the global critic, we want
   pairwise similarities across agents. Need to adapt the sync mechanism.

3. **Gradient flow to actors?** The CTM is in the critic, so its gradients don't
   reach the actors directly. Actors improve via better advantages only. This is
   fine — it's standard CTDE.

4. **CTM iterations in critic?** Can use more iterations since critic is
   training-only. iter=3 might work here without the actor-side cost concerns.

---

## Ablations

1. **MAPPO baseline** — GRU actor + GRU critic, no CTM
2. **MAPPO + Global CTM critic** — GRU actor + CTM-enhanced critic (proposed)
3. **Shared vs per-agent V** — does per-agent decomposition matter, or does
   just having CTM features in a shared critic suffice?
4. **Sync matrix analysis** — visualize S over training to show coordination
   structure emerges (strongest qualitative result)

---

## Expected Results

- On 3m: small or no improvement (coordination is simple, credit assignment is easy)
- On smacv2_10_units: meaningful improvement (10 agents, credit assignment is hard)
- The larger the team, the more credit assignment matters, the more CTM helps

---

## Optional Extensions (if Idea 1 works)

### Idea 2: Sync-Modulated GAE

Use sync matrix to set per-agent lambda in GAE:

```
lambda_i = lambda_base * sigmoid( -beta * sum_{j!=i} S_ij )
```

- High coordination (high sum S_ij) -> lower lambda -> trust critic more (low variance)
- Low coordination -> higher lambda -> rely on MC returns (low bias)
- Recovers standard GAE when beta = 0
- One line change after GAE computation

Motivation: coordinated agents are more predictable by the critic (it models
their joint behavior), so we can afford lower variance at the cost of more bias.
Independent agents are harder to model, so we prefer unbiased MC estimates.

### Idea 3: Coordination-Aware Counterfactual Baseline (Cheap COMA)

COMA marginalizes over all agents' actions for counterfactual baselines — O(|A|^N).
The sync matrix identifies which agents are actually relevant:

```
b_i(s) = E_{a_j : S_ij > tau}[ V(s, a_i, a_j) ]
```

Only marginalize over the k agents most coordinated with agent i (top-k by S_ij).
This reduces cost from O(|A|^N) to O(|A|^k) where k << N.

Harder to implement cleanly. Consider only if Idea 1 shows the sync matrix
captures meaningful structure.
