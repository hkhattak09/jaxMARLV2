# CTM Actor Design Decisions

## What We Are Doing

Replacing the MLP actor in MADDPG with a CTM (Continuous Thought Machine) actor.
The critic remains unchanged as an MLP — the MLP critic has sufficient expressive power
because it already receives the full global state (all agents' obs + all agents' actions),
which is a complete Markovian representation. No temporal memory is needed there.

The actor is where temporal memory adds genuine capability. The MLP actor sees only a
single local observation per timestep with no memory of the past. CTM gives each agent
a sliding window of recent trajectory history, enabling temporally-aware decision making
— anticipating neighbor movements, recognising approach/diverge patterns, course-correcting
based on recent drift. These are things a memoryless MLP simply cannot do regardless of
how wide it is.

---

## Architecture

- **Single shared CTM network** across all 24 agents (parameter sharing, homogeneous team)
- **Individual hidden states** per agent — one board per agent, evolved independently
- **Attention and KV projections removed** — input is a flat 192-dim observation vector,
  no sequence to attend over. The existing `ctm_rl.py` already does this (heads=0).
- **Action head** — linear layer mapping synchronisation output → action_dim=2, followed
  by tanh to constrain to [-1, 1]
- **Critic** — MLP, unchanged

---

## Stateless Actor Updates

### What This Means

During rollout: hidden states are propagated correctly across timesteps. Each agent's board
accumulates genuine trajectory history. Full temporal memory, exactly as CTM was designed.

During actor update: hidden states are initialised fresh from `start_activated_trace` for
every sampled batch. No hidden states stored in the replay buffer. The buffer is unchanged.

### Why Stateless

**Storing hidden states in the buffer does not scale.** With n_rollout_threads up to 7,
buffer_length=20k, and 24 agents:

| n_rollout_threads | Total buffer rows | Hidden state storage (d_model=64, memory_length=8) |
|---|---|---|
| 1 | 480k | ~1.9 GB |
| 3 | 1.44M | ~5.6 GB |
| 5 | 2.4M | ~9.4 GB |
| 7 | 3.36M | ~13.1 GB |

This is on top of existing buffer data, network weights, optimizer state, and JAX environment
memory. Impractical at scale.

Stored hidden states also suffer from **staleness** — the hidden state saved during rollout
was computed by older network weights. By update time the weights have changed, making the
stored state inaccurate anyway.

**Stateless introduces a biased gradient**, not a wrong one. The gradient direction is still
correct (Q-weighted, still points toward higher-value actions). The bias is in magnitude.
MADDPG already tolerates multiple approximation sources (off-policy sampling, TD bootstrapping,
soft target updates) — this is one more bounded noise source.

### Mitigations

**Learnable decay parameters (CTM-specific):** The synchronisation computation uses learnable
exponential decay weights per neuron pair. The rightmost (most recent) slots in the board
are overwritten first during the inner loop and are computed identically in both rollout and
stateless update. The leftmost (oldest) slots are where the mismatch lives. The decay params
will naturally learn to upweight recent reliable slots and downweight stale ones — a
self-correcting mechanism that emerges from training without any engineering.

**Same iterations for rollout and update:** Using different iteration counts introduces a
structural mismatch — the synchronisation is computed over differently-shaped boards in
each case, meaning the gradient optimises a different computation than what runs during
rollout. Using the same iterations keeps the computational structure identical; only the
content of the stale slots differs, which start_activated_trace and decay params handle.

**start_activated_trace is learned:** This parameter receives gradients from every actor
update and learns to approximate a reasonable neutral context — minimising the damage from
using a fresh board during updates.

**Reynolds flocking prior:** The existing regularization term `0.3 × alpha × MSE(π(obs), a_prior)`
anchors policy updates to physically meaningful actions early in training, dampening the
effect of noisy gradients.

---

## Hidden State Management During Rollout

Hidden states live in the training loop, not inside the network.

Shape: `(n_rollout_threads × n_agents, d_model, memory_length)` — two tensors
(state_trace and activated_state_trace).

- Initialised from `start_activated_trace` at training start
- Passed into every actor call, updated version returned
- On `done=True`: that agent's board reset to `start_activated_trace`
- Other agents' boards unaffected (per-env, per-agent masking)
- Live on GPU throughout — the training script keeps networks on GPU for both rollout and
  update (prep_rollouts(device="gpu") and prep_training(device="gpu")). Hidden states
  are initialised as CUDA tensors at training start and never move. Memory cost is negligible
  (~2.7 MB even at 7 parallel envs with d_model=128, memory_length=16).

---

## Target Network

The target actor (used in the critic update to compute target actions for next_obs) also
uses stateless initialization — fresh board from target policy's `start_activated_trace`.
Consistent with the stateless approach and avoids any additional complexity.

---

## Hyperparameters

| Parameter | Value | Reasoning |
|---|---|---|
| d_model | 256 | Increased from initial 128 — GPU rollout makes wider models cheap. Current MLP hidden_dim=180, so 128 would be underpowered. 256 gives richer neuron population and more diverse synchronisation patterns |
| memory_length | 16 | ~8% of episode (200 steps), captures short trajectory patterns |
| n_synch_out | 16 | 136-dim synchronisation output (16×17/2), sufficient for action_dim=2 action head |
| iterations | 4 | Overwrites 25% of board per call, preserves 75% trajectory history. Same value for rollout and update — different values would create structural mismatch in synchronisation computation. Starting point, tune over [3,4,5,6] |
| synapse_depth | 1 | Flat obs vector, 2-block MLP (Linear→GLU→LayerNorm ×2) sufficient |
| deep_nlms | False | Shallow NLMs (single linear over memory window per neuron) are 68× cheaper and sufficient — complex inter-neuron interactions are captured by synchronisation not NLMs |
| do_layernorm_nlm | True | Training stability |
| memory_hidden_dims | [64] | Passed to constructor but unused when deep_nlms=False |

---

## Done Reset Mechanism

Hidden states are reset via a mask applied in the training loop after each `env.step()`.

The done signal from the env is `(1, n_total_agents)`. Reshape to `(n_total_agents, 1, 1)` to
broadcast over the hidden state shape `(n_total_agents, d_model, memory_length)`.

- `state_trace` where done=True → zeros (correct initialisation for state_trace)
- `activated_state_trace` where done=True → `start_activated_trace` expanded for batch

Agents where done=False are untouched. Per-env, per-agent — env 2 finishing does not reset
env 1's agents. Cheap in-place GPU operation on a ~2.7 MB tensor.

---

## Implementation Plan

**New files:**
- `algorithm/utils/ctm_actor.py` — `CTMActor` class (wraps ContinuousThoughtMachineRL + action head)
- `algorithm/utils/ctm_agent.py` — `CTMDDPGAgent` subclass of DDPGAgent

**Modified files:**
- `algorithm/algorithms/maddpg.py` — `init_from_env()` accepts `use_ctm_actor` flag, instantiates CTMDDPGAgent when set; `update()` creates fresh hidden states for actor and target actor forward passes
- `train/train_assembly_jax_gpu.py` — initialize hidden states before episode loop, pass to maddpg.step(), apply done mask after each step
- `cfg/assembly_cfg.py` — CTM switch and hyperparameters (done)

**Unchanged:**
- `DDPGAgent` (MLP version kept intact)
- `MLPNetwork`, `ReplayBufferAgent`, critic, buffer interface

**Notes:**
- `log_pi` returned from CTMDDPGAgent.step() can be None — discarded with `_` in training loop
- `start_activated_trace` is an nn.Parameter, saved/loaded with network weights automatically
- Target actor's `start_activated_trace` gets soft-updated correctly as a normal parameter

---

## What Does Not Change

- Replay buffer structure and interface
- Critic and target critic (MLP, unchanged)
- Reward structure and Reynolds flocking prior
- MADDPG update logic (except actor forward pass now takes hidden state)
- Environment interface
