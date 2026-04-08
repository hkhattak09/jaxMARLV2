# CTM Actor Design Decisions

## What We Are Doing

Replacing the MLP actor in MADDPG with a CTM (Continuous Thought Machine) actor.
The critic remains an MLP — it receives the full global state (all agents' obs + all agents'
actions), a complete Markovian representation, so no temporal memory is needed there.

---

## History of Decisions and What We Learned

### Phase 1 — Initial Implementation (Stateful Rollout, Stateless Update)

The first implementation propagated hidden states across timesteps during rollout
(full temporal memory across 200 steps) but initialised fresh hidden states during
actor updates. The reasoning was:

- Storing hidden states in the replay buffer was deemed impractical due to memory cost
- The gradient bias was assumed to be tolerable (correct direction, biased magnitude)
- `start_activated_trace` as a learned parameter was expected to reduce the rollout/update mismatch

**Training results (4 runs, logs 1-4):**

| Run | Command | ep100 Coverage | ep250 Coverage | Policy loss ep250 |
|---|---|---|---|---|
| MLP baseline | `--use_mlp_actor` | 0.644 | ~0.75 | -1.7 |
| CTM default | default config | 0.061 | 0.061 | -4.77 |
| CTM + tau=0.005 | `--lr_actor 3e-5 --ctm_iterations 2 --tau 0.005` | 0.084 | 0.162 | -2.45 |
| CTM smaller | `--ctm_d_model 128 --ctm_memory_length 8 --lr_actor 3e-5` | 0.069 | stopped ep140 | worse |

**What went wrong — confirmed diagnosis:**

Policy loss dove 3-4× faster than MLP while actual task metrics (coverage, Voronoi
uniformity) stagnated near baseline. This is Q-overestimation caused by the
rollout/update mismatch:

- During rollout the CTM produces actions from a warm, 200-step context-rich hidden state
- During actor update the CTM produces actions from a blank fresh board
- The critic was trained on (obs, action_rollout) pairs and has never seen (obs, action_fresh_board)
- Q-values assigned to fresh-board actions are unreliable — the policy gradient optimises
  critic noise rather than task performance

Reducing model size (log 4) made things worse, not better — ruling out overfitting as
the primary cause. Slowing target network updates (tau=0.005, log 3) reduced the policy
loss dive rate but could not fix the underlying mismatch.

### Phase 2 — Understanding the Literature (R-MADDPG)

We read *R-MADDPG for Partially Observable Environments and Limited Communication*
(Wang et al., 2020) and examined their open-source implementation.

**Key findings from the paper:**

1. Under **fully observable** settings, all variants (plain MADDPG, recurrent actor only,
   recurrent critic only, recurrent actor+critic) perform equivalently well. The recurrent
   critic is only critical for *partial observability*.

2. Our environment is **locally partially observable** — each agent only observes cells
   within `d_sen=0.4` of itself. Shapes span ~2.0×2.2 units with ~500 cells; an agent
   at any position sees at most 9–27% of shape cells. The M=80 cap was not the binding
   constraint — sensor range is. The centralised critic addresses this by aggregating all
   24 agents' local views during training (collective near-complete coverage of the shape).

3. The paper's recurrent actor uses **stored LSTM hidden states** in the replay buffer
   (both `c_t` and `h_t` stored per transition, per agent). During actor update, the
   stored state from rollout is fed back in — not a fresh state. This is what eliminates
   the rollout/update mismatch in their system.

**Why storing CTM hidden states is impractical for us:**

Their LSTM state per agent = 2 × 64 floats = 128 floats per transition.
CTM hidden state per agent = 2 × (d_model × memory_length) = 2 × (256 × 16) = 8,192 floats.

| Config | Storage (buffer_length=20k, 24 agents) |
|---|---|
| R-MADDPG LSTM | ~246 MB — fine |
| CTM (d_model=256, mem=16) | ~15.7 GB — infeasible on T4 |
| CTM (d_model=128, mem=8) | ~3.9 GB — borderline, before other costs |

Even on an H100 with 512 GB host RAM, storage is technically feasible. But a second
problem remains: **staleness**. For a small LSTM, stored states from older policy versions
remain approximately valid — the state space is simple. For CTM, the hidden state is a
rich distributed representation shaped by 200 steps through a specific version of the
network weights. As weights change, stored states become inconsistent with what the
current CTM would produce, potentially worse than a fresh state. The off-policy staleness
problem is more severe for complex high-dimensional hidden states.

**Conclusion from literature review:** The original stateless-update approach is
architecturally broken for our setup. Storing hidden states is the principled fix but
is impractical. This motivates a different approach.

### Phase 3 — Stateless Rollout (Current Direction)

**Core insight:** The mismatch between rollout and update exists because rollout is
stateful and update is stateless. The fix does not have to be making updates stateful —
it can equally be making rollout stateless too.

**What stateless rollout means:**

During rollout, reset hidden states to fresh at every timestep before each forward pass,
exactly as the actor update does. Rollout and update are now structurally identical —
zero mismatch, zero Q-overestimation from this source.

**What the CTM still contributes under stateless rollout:**

The CTM runs `iterations=4` inner loop steps on each observation before producing an
action. This is iterative/recurrent computation *within* a single timestep — the network
refines its representation of the current observation through multiple passes. This is
genuinely different from what an MLP does (single feedforward pass). The CTM can learn
to reason more deeply about each observation even without cross-timestep memory.

**What the CTM loses:**

Cross-timestep memory — the ability to remember where agents were in previous steps,
recognise trajectory patterns, anticipate movements. Given that the environment is fully
observable and the MLP learns without this, cross-timestep memory may not be necessary
for this task. This is an empirical question the stateless rollout experiment will answer.

**Why this is worth testing before bigger hardware:**

If stateless CTM matches or beats MLP, we have a working CTM with zero infrastructure
changes — same buffer, same memory, same training loop (just remove the hidden state
carry-across). If it does not beat MLP, it tells us the CTM's iterative computation
is not adding value for this specific task and no amount of hardware or complexity will
fix that.

---

### Phase 4 — Centralised Critic: Flat MLP Failure and AggregatingCritic Fix

**What was attempted:** Replace the per-agent critic (194-dim MLPNetwork) with a flat
MLPNetwork over the concatenated joint input (4,656-dim = 24 × (192+2)).

**What happened:** Coverage collapsed from 0.664 to 0.412 after 1000 episodes (−38%).
The critic was producing garbage Q-values and the actor followed.

**Root cause — two compounding failures:**

1. **Capacity mismatch:** 4,656-dim input compressed to 256-dim hidden in one layer (18×).
   The critic literally could not represent a meaningful Q-function from this input.

2. **Wrong inductive bias:** A flat MLP over concatenated agent data has no way to exploit
   the permutation equivariance of the Q-function. For a homogeneous cooperative team,
   Q(obs_1, act_1, ..., obs_24, act_24) should be invariant to reordering of agents.
   A flat MLP has to learn this symmetry from data — which requires far more samples and
   a much larger network than the problem warrants.

**Fix — AggregatingCritic (`networks.py`):**

```
for each agent i (shared encoder weights):
    embed_i = MLP(obs_i, act_i)    # 194 → 128 → 64

agg = mean(embed_1, ..., embed_24) # 64-dim — permutation equivariant by construction

Q = MLP_head(agg)                  # 64 → 128 → 1
```

Each encoder sees 194-dim inputs — the same scale as the old per-agent critic. The
aggregation is structurally correct (mean is permutation-invariant). The head sees
a compact 64-dim summary. No inductive bias is wasted; no over-compression occurs.

**Interface:** `forward(X)` accepts `torch.cat([obs_all, act_all], dim=1)` — identical
to the flat MLP call pattern. No changes needed at call sites in `maddpg.py`.

**Environment observability finding (Phase 4):**

Inspection of `assembly.py` and the shapes pkl confirmed that `d_sen=0.4` is the true
observability constraint, not M=80. Each shape has ~500 cells across ~2.0×2.2 units;
from any position an agent sees at most 9–27% of cells. The M=80 cap is only hit near
dense central areas. The prior claim that M=80 makes the shape "almost fully known" was
incorrect — each agent has genuinely local partial observability at all M settings.

---

### Phase 5 — Stateful CTM + Recurrent Critic (Current Direction)

**Why stateless failed:** Stateless mode with `iterations=4` and `memory_length=16` means
the activated_state_trace has 12 identical initial columns + 4 computed columns. NLMs
process mostly padding. Synchronization is computed over this padded trace — it's meaningless.
The CTM is functionally a deeper feedforward network, not leveraging its two core innovations
(temporal NLM dynamics, synchronization-as-representation).

**Evidence from results:**
- MLP + prior reg: Coverage 0.916, Voronoi 0.653 (best)
- CTM seed (d_model=256): Coverage 0.835, Voronoi 0.564
- CTM doesn't beat MLP because it's not actually doing what CTM is designed to do

**Key insight from CTM paper (Appendix G.6):** Their RL setup uses 1-2 internal ticks per
env step, stateful across the episode. After 200 env steps × 1 tick, the activation history
has 200 meaningful entries. Our setup: 4 ticks per step, history reset → 4 entries total.

**Key insight from R-MADDPG (Wang et al., 2020):** Recurrent actor alone doesn't help.
Recurrent critic is what matters for partial observability. Both together is best.

**The plan:**

1. **Stateful CTM actor** — `iterations=1`, hidden states carried across all 200 episode
   steps. CTM builds temporal dynamics naturally. Prior seeding initialises the first step.

2. **Recurrent critic** — LSTM after aggregation in AggregatingCritic:
   ```
   per-agent encoder: (obs_i, act_i) → 128 → 64    [shared, independent]
   mean aggregate → 64-dim team summary              [permutation equivariant]
   LSTM: 64-dim → hidden_dim                         [temporal reasoning]
   head: hidden → Q-value
   ```
   Permutation equivariance preserved (aggregation before recurrence).

3. **Episode-sequence replay buffer** — contiguous episode chunks instead of random
   transitions. R2D2-style burn-in: replay prefix without gradient to reconstruct
   hidden states, then compute gradients on remaining steps.

**What this fixes:**
- Rollout/update mismatch: both are now stateful with burn-in to reconstruct context
- CTM actually functions as designed: meaningful temporal dynamics and synchronization
- Critic gets temporal reasoning: can track team state evolution (R-MADDPG finding)
- Paper story: prior-seeded CTM actor + recurrent equivariant critic for partial obs MARL

---

## Architecture (Current → Next)

### Current (stateless — being replaced)
- **Actor**: CTMActor, `iterations=4`, hidden states fresh every step
- **Critic**: AggregatingCritic, memoryless
- **Buffer**: Random transition sampling

### Next (stateful + recurrent)
- **Actor**: CTMActor, `iterations=1`, stateful across episode steps, prior-seeded init
- **Critic**: AggregatingCritic + LSTM after aggregation
- **Buffer**: Episode-sequence sampling with burn-in

### Unchanged
- Single shared CTM network across all 24 agents (parameter sharing)
- Attention and KV projections removed (heads=0) — flat obs, no sequence
- Action head: Linear(136, 2) + Tanh on synchronization output
- AggregatingCritic's per-agent encoder and aggregation (LSTM added after)

---

## Hyperparameters

| Parameter | Current | Next | Reasoning |
|---|---|---|---|
| d_model | 128 | 128 | Sufficient for 192-dim obs |
| memory_length | 16 | 16 | 16-step sliding window of activations |
| n_synch_out | 16 | 16 | 136-dim synchronisation output (16×17/2) |
| iterations | 4 | **1** | 1 tick per env step; dynamics build across 200 steps |
| synapse_depth | 1 | 1 | 2-block MLP sufficient |
| deep_nlms | False | False | 68× cheaper, sufficient expressiveness |
| lstm_hidden_dim | N/A | TBD | Critic LSTM hidden size |
| sequence_length | N/A | TBD | Sampled sequence length for training |
| burn_in_length | N/A | TBD | Prefix replayed without gradient |
