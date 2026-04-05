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

## Architecture (Current)

- **Single shared CTM network** across all 24 agents (parameter sharing, homogeneous team)
- **Individual hidden states** per agent — one board per agent (under stateful rollout);
  under stateless rollout, these are reinitialised every step
- **Attention and KV projections removed** — flat 192-dim obs, no sequence (heads=0)
- **Action head** — Linear(136, 2) + Tanh
- **Critic** — `AggregatingCritic`: shared encoder per agent (194→128→64), mean aggregation, head MLP (64→128→1)

---

## Stateless Rollout Implementation Plan

**Change required:** In the training loop rollout, instead of carrying hidden states
forward between steps, call `get_initial_hidden_state()` before every `maddpg.step()`.
The per-episode initialisation and done-mask reset logic can be removed entirely.

**Files to change:**
- `train/train_assembly_jax_gpu.py` — remove hidden state carry-forward in rollout loop;
  reinitialise fresh hidden state at every step instead. Same change in `run_eval` and
  `run_final_eval`.

**Files unchanged:**
- `ctm_actor.py`, `ctm_agent.py`, `maddpg.py` — no changes needed
- Buffer, critic, reward structure — unchanged

---

## Hyperparameters

| Parameter | Value | Reasoning |
|---|---|---|
| d_model | 256 | Richer neuron population; GPU makes this cheap |
| memory_length | 16 | Less relevant under stateless rollout, but kept for consistency |
| n_synch_out | 16 | 136-dim synchronisation output (16×17/2) |
| iterations | 4 | 4 inner reasoning passes per observation — the primary value-add under stateless rollout |
| synapse_depth | 1 | Flat obs, 2-block MLP sufficient |
| deep_nlms | False | 68× cheaper, sufficient expressiveness |
| do_layernorm_nlm | True | Training stability |
| memory_hidden_dims | [64] | Unused when deep_nlms=False |

---

## What Does Not Change

- Replay buffer structure and interface
- Reward structure and Reynolds flocking prior
- MADDPG update logic (critic call sites unchanged — AggregatingCritic accepts same input format)
- Environment interface
- CTM actor/agent code
