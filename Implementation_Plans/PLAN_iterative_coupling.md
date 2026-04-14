# Iterative Coupling Refinement for Multi-Agent Credit Assignment

## The Problem

In cooperative MARL with shared rewards (e.g. SMAX), when the team wins or loses,
standard MAPPO gives every agent the same value estimate V(s) and therefore the
same advantage A = G - V(s). Agent 3 who made a brilliant flanking move gets the
same learning signal as agent 7 who did nothing. This is noisy and slows learning,
especially with many agents (10+).

## The Idea

Replace the shared critic value V(s) with per-agent values V_i(s), where each
agent's value reflects its role in the team's coordination structure.

To compute V_i, we add a module inside the critic (training-only, discarded at
test time) that figures out which agents are coupled — whose actions jointly
affect the outcome. It does this through **iterative coupling refinement**:
multiple rounds of processing where each round propagates coupling information
one hop further through the agent graph.

### Why iterations matter

In one pass, you can detect direct coupling: agents 2 and 3 are fighting the same
enemy, clearly linked. But you miss indirect coupling: agent 5 (a healer) connects
agents 2 and 7 — if agent 5 dies, both groups suffer. Iterative refinement
propagates this: round 1 finds direct pairs, round 2 finds one-hop indirect
connections, round 3 captures the full dependency chain.

### What's CTM-inspired about it

Two components from the Continuous Thought Machine carry over with clear
justification:

1. **Learned decay over iteration history.** Instead of just using the final
   coupling matrix C^K, we compute a decay-weighted combination of coupling
   matrices from all iterations C^1, ..., C^K. Learned decay parameters let the
   network decide whether direct coupling (early iterations) or indirect coupling
   (late iterations) matters more, and this can vary per agent-pair. This is
   directly adapted from the CTM's sync decay mechanism.

2. **NLM (Non-Linear Memory) for trajectory compression.** Instead of taking each
   agent's final embedding after K iterations, we feed the full iteration
   trajectory [e_i^0, e_i^1, ..., e_i^K] into an NLM. The NLM compresses each
   agent's deliberation trajectory — encoding not just "what role did this agent
   end up with" but "how did its role evolve through the coupling refinement
   process." The NLM uses SuperLinear layers that operate per-feature across the
   iteration dimension, which is exactly what it was designed for.

### What's NOT carried over from the CTM (and why)

- **Pairwise neuron-product sync:** Designed to measure internal coherence of one
  network. For inter-agent coupling, a learned MLP that directly predicts coupling
  scores from agent embeddings is more appropriate.
- **Synapses module:** Just an MLP with GLU. No structural reason to use it over
  any other update function.
- **Backbone:** Same — generic input projection.

## Architecture

```
ACTORS (decentralised, unchanged from GRU baseline):
  o_i -> embedding -> GRU -> h_i -> policy head -> pi(a_i | o_i)

CRITIC (centralised, training-only):
  Inputs: all agents' observations o_i, actions a_i, done flags

  Step 1 — Per-agent embedding:
    e_i^0 = MLP(o_i, a_i)              -- "what is agent i doing in its situation"

  Step 2 — Iterative coupling refinement (K rounds, shared weights):
    For k = 1..K:
      # Coupling scores between all agent pairs
      C^k_ij = sigmoid(MLP_couple(e_i^{k-1}, e_j^{k-1}))     -- scalar per pair

      # Mask dead agents (C_ij = 0 if j is dead)
      C^k_ij = C^k_ij * alive_mask_j

      # Update each agent's embedding with coupling-weighted context
      context_i^k = sum_j (C^k_ij * e_j^{k-1})    -- for j != i
      e_i^k = MLP_update(e_i^{k-1}, context_i^k)

      # Store iteration trace for NLM
      trace_i[:, k] = e_i^k

  Step 3 — Decay-weighted coupling (CTM-inspired):
    # Learned decay over iterations, per coupling pair
    decay_params: learnable, shape (n_pairs,)
    weights_k = exp(-k * decay_params)
    C_final_ij = sum_k (weights_k * C^k_ij) / sum_k(weights_k)

  Step 4 — NLM trajectory compression (CTM component):
    # Compress each agent's iteration trajectory into final representation
    out_i = NLM(trace_i)               -- trace_i shape: (embed_dim, K)
    # NLM outputs a compressed representation per agent

  Step 5 — Per-agent value head:
    V_i = MLP_value(out_i)

  Step 6 — Per-agent advantages for policy gradient:
    A_i = G_t - V_i(s_t)
```

### Key design choices

- **Actions as input to critic:** We use (o_i, a_i) not just o_i. This means the
  critic computes Q-style values conditioned on what agents actually did. Two agents
  attacking the same target can be identified as coupled even if their observations
  look different.

- **Alive masking:** Dead agents get zero coupling scores. Their value estimates
  still exist but are decoupled from the team. Reuse the existing alive_mask logic
  from AgentConsensus in ctm_jax.py.

- **Shared weights across iterations:** Same MLP_couple and MLP_update parameters
  at every iteration. This is the CTM principle — iterate the same computation to
  refine the answer.

- **Critic-only (CTDE):** None of this runs at test time. Agents execute
  decentralised GRU policies. The benefit flows through better per-agent advantages
  during training.

## What stays the same vs what changes

### Unchanged
- Actor architecture (GRU)
- Environment setup, wrappers
- PPO loss for actor (but uses per-agent advantages)
- Rollout collection

### Changed
- Critic architecture: new IterativeCouplingCritic replaces CriticRNN
- Transition tuple: stores per-agent values instead of shared value
- GAE computation: per-agent instead of shared
- Critic loss: (1/N) sum_i (V_i - G_t)^2

## Files needed

- `smax_ctm/ctm_jax.py` — NLM class, SuperLinear class (reused directly)
- `smax_ctm/train_mappo_gru.py` — base training script to modify (actors,
  env setup, PPO logic, config). This is the starting point, not the CTM
  training script.

Everything else can be cleaned up.

---

## Implementation Stages

Each stage ends with a runnable experiment that tells us something.

### Stage 1: Per-Agent Critic with Static Coupling (no iterations, no CTM parts)

**Goal:** Test whether per-agent value decomposition helps AT ALL, before adding
any complexity.

**Context from first run:** A feedforward (no GRU) critic was tried first. Win rate
rose briefly then collapsed. Entropy did NOT collapse (~0.18, higher than GRU
baseline). The diagnosis: without recurrent state the critic can't track episode
history in partial observability. R-MADDPG confirms this — a recurrent critic is a
prerequisite for stable training in SMAX, not an optional enhancement. The GRU is
not the experimental variable here; per-agent value decomposition is.

**What to do:**

`train_mappo_ic.py` already exists as a starting point. The CriticRNN needs to be
rewritten and the runner state needs to carry a critic hidden state. Specific changes:

1. **Rewrite `CriticRNN`** — the critic processes trajectories with temporal
   dependencies, so it needs a GRU. The structure is:
   ```
   (obs_i, action_i) -> MLP embed -> e_i          # per-agent, per-timestep
   e_i -> ScannedRNN (GRU) -> h_i                 # recurrent over time, per-agent
   context_i = mean(h_j for j != i)               # mean-pool other agents' hidden states
   V_i = MLP(concat(h_i, context_i))              # per-agent value head
   ```
   - The GRU is shared across agents (same parameters, one `ScannedRNN` instance).
   - The critic takes `(obs, actions, dones)` where `dones` drives GRU resets
     (same pattern as `ScannedRNN` in the actor: reset state when done=True).
   - Output shape: `(T, NUM_ACTORS)` — one value per agent per timestep.
   - The mean-other computation must reshape flat `(T, NUM_ACTORS, D)` into
     `(T, NUM_ENVS, NUM_AGENTS, D)` before pooling, then flatten back. Agents are
     stored in env-major order so the reshape is exact.

2. **Add `cr_hstate` to `runner_state`** — the critic needs to carry hidden state
   across rollout steps (during rollout, the critic is called one step at a time
   with T=1, so without this the GRU starts from zeros every step):
   - Initialize: `cr_hstate = ScannedRNN.initialize_carry(NUM_ACTORS, GRU_HIDDEN_DIM)`
   - Add to `runner_state` tuple alongside `ac_hstate`
   - In `_env_step`: pass `cr_hstate` to critic, receive updated `cr_hstate` back
   - The critic's `__call__` must return `(new_hstate, values)` not just `values`

3. **Critic update pass** — during the PPO update epochs, recompute values over the
   full trajectory starting from `initial_cr_hstate` (same pattern as the actor's
   `init_hstate`). Pass `initial_hstate` into the critic's apply call. Store
   `initial_cr_hstate` in `update_state` alongside `initial_hstate`.

4. **Bootstrap value** — when computing `last_val` after rollout, call the critic
   with the current `cr_hstate` (just like the actor uses `ac_hstate`).

5. **GAE and actor loss** — these are already per-agent in the existing
   `train_mappo_ic.py`. No changes needed there.

6. **Minibatching constraint** — already enforced: `NUM_ENVS % NUM_MINIBATCHES == 0`.
   This ensures full agent teams stay together when splitting the flat actor buffer.

**What we learn:** Does per-agent V_i + mean-pool beat shared V(s) from the GRU
baseline? If no on both maps, the whole approach is dead. If equal on 3m and better
on 10m (or better on both), proceed to Stage 2.

**Run:** 3m (expect roughly equal to baseline) and 10m or 10m_vs_11m (expect
improvement if credit assignment matters at larger team size).

### Stage 2: Add Coupling Matrix (still no iterations)

**Goal:** Test whether learned per-agent coupling beats uniform mean-pooling.
This is the first real test of the core idea.

**Skipping Stage 1:** Stage 1 (mean-pool) is structurally too close to shared V(s)
to produce a meaningful signal — with world_state as input, all agent embeddings
are nearly identical (only the one-hot agent ID differs), so mean-pool ≈ shared V.
Stage 2 is where genuine per-agent differentiation first appears.

**What to do — changes to `CriticRNN` only:**

Replace the mean-other pooling block with a coupling matrix:
```
# h: (T, E, A, D) after GRU
h_i = h[:, :, :, :]           # (T, E, A, D)
h_i_exp = h[:, :, :, None, :]  # (T, E, A, 1, D)  — "from" agent
h_j_exp = h[:, :, None, :, :]  # (T, E, 1, A, D)  — "to" agent

pair = concat([broadcast(h_i_exp, (T,E,A,A,D)),
               broadcast(h_j_exp, (T,E,A,A,D))], axis=-1)  # (T, E, A, A, 2D)

C = sigmoid(Dense(1)(relu(Dense(COUPLE_HIDDEN)(pair))))  # (T, E, A, A, 1)
C = C.squeeze(-1) * (1 - eye(A))                         # zero self-coupling

context_i = sum_j(C[:,:,:,j,:] * h[:,:,j,:])            # (T, E, A, D)
```

Dead-agent masking: not needed explicitly. When agent j dies, ScannedRNN resets
h_j to zero → C_ij * h_j = 0 regardless of C_ij. Use sum (not normalized mean)
so dead agents drop out naturally without division by alive count.

Add `COUPLE_HIDDEN_DIM: 64` to config.

**What we learn:** Does learned coupling beat mean-pool? Check two things:
1. Win rate vs GRU baseline — equal or better?
2. Does C_ij show structure (non-uniform)? Log `C.mean()` and `C.std()` per update.
   If std ≈ 0, the network treats all agents equally — coupling adds nothing.

**Run:** 3m first (quick sanity check). Then 10m if 3m looks promising.

### Stage 3: Add Iterations

**Goal:** Test whether iterative refinement of coupling helps — the core hypothesis.

**What to do:**

**1. Switch `CriticRNN` from `@nn.compact` to `setup()` style.**

Inside `@nn.compact`, each call to `nn.Dense(...)` registers new parameters. Calling
the same Dense in a Python loop creates K independent parameter sets — not shared
weights. To share weights across iterations, define the coupling and update layers
as named instance attributes in `setup()` and call them repeatedly in `__call__`.

```python
def setup(self):
    D  = self.config["AGENT_EMBED_DIM"]
    CH = self.config["COUPLE_HIDDEN_DIM"]
    CR = self.config["CRITIC_HIDDEN_DIM"]

    self.embed1    = nn.Dense(D, ...)
    self.embed2    = nn.Dense(D, ...)
    self.rnn       = ScannedRNN()
    self.couple_h  = nn.Dense(CH, ...)   # shared across K iterations
    self.couple_out= nn.Dense(1, ...)    # shared across K iterations
    self.update_h  = nn.Dense(D, ...)    # shared across K iterations
    self.update_out= nn.Dense(D, ...)    # shared across K iterations
    self.value_h1  = nn.Dense(CR, ...)
    self.value_h2  = nn.Dense(CR, ...)
    self.value_out = nn.Dense(1, ...)
```

**2. Add alive masking.**

`rnn_out` for a dead agent is NOT zero. The GRU reset zeros the carry at the start
of the next timestep, but then runs the cell on the agent's (zeroed/invalid)
observation, producing a non-zero output. Without explicit masking:
- Stage 2: one pass of corrupted coupling from dead agents.
- Stage 3: corruption compounds K times as dead embeddings propagate through each
  iteration to every other agent.

Compute the mask from `dones` (already available in the critic):
```python
# dones: (T, num_actors) -> alive_mask: (T, E, A)
alive_mask = 1.0 - dones.reshape(T, num_envs, num_agents)

# Zero dead agents before iterations start
e = rnn_out * alive_mask[..., None]
```

Apply at EVERY iteration:
```python
# Zero coupling weight to dead agents (column mask on j-axis)
C = C * alive_mask[:, :, None, :]         # (T, E, A, A)
# Re-zero dead agents after update so they never inject signal
e = e * alive_mask[..., None]
```

**3. Add update MLP and iterative loop.**

Stage 2 has no update step — it reads from fixed `rnn_out` and appends context to
the value head. Stage 3 needs a real iterative update:

```python
K = self.config["COUPLING_ITERATIONS"]
e = rnn_out.reshape(T, E, A, D)
e = e * alive_mask[..., None]   # zero dead agents

for _ in range(K):
    h_i = jnp.broadcast_to(e[:,:,:,None,:], (T,E,A,A,D))
    h_j = jnp.broadcast_to(e[:,:,None,:,:], (T,E,A,A,D))
    pair = jnp.concatenate([h_i, h_j], axis=-1)          # (T, E, A, A, 2D)

    C = nn.sigmoid(self.couple_out(nn.relu(self.couple_h(pair)))).squeeze(-1)  # (T,E,A,A)
    C = C * alive_mask[:, :, None, :] * (1.0 - jnp.eye(num_agents))

    context = jnp.einsum("teij,tejd->teid", C, e)        # (T, E, A, D)

    update_in = jnp.concatenate([e, context], axis=-1)   # (T, E, A, 2D)
    delta = nn.relu(self.update_out(nn.relu(self.update_h(update_in))))  # (T, E, A, D)
    e = e + delta                                          # residual: ADD, do not replace
    e = e * alive_mask[..., None]                         # re-zero dead agents
```

**Why residual, not replace:** Without the residual (`e = delta` instead of
`e = e + delta`), each iteration replaces the agent's embedding with a blend of
itself and its neighbours. After K=2 rounds this is mild over-smoothing —
agent-specific signal is diluted, the value head receives a neighbourhood mean
rather than a per-agent representation, and V_i prediction degrades. In practice
this showed up as fast early learning (coupling helps when C is noisy) followed
by a plateau and slow convergence after C sharpens during training. The residual
preserves agent identity through all K iterations — the MLP adds coordination
signal rather than overwriting the original.

Value head takes `e` (the final updated embedding, shape (T, E, A, D)) — NOT
`concat(e, context)` as in Stage 2. The context is already baked into `e` through
K iterations, so concatenating it again would double-count.

**4. Config addition:**
```python
"COUPLING_ITERATIONS": 2   # start with K=2, then test K=3
```

**What we learn:** Does K=2 or K=3 beat K=1 (Stage 2)? If yes, iterative coupling
refinement matters — indirect/transitive dependencies are real and captured.
If no, iterations add nothing and we stop here (Stage 2 result is the paper if
it was positive).

**Run:** Primarily the larger map (6h_vs_8z confirmed better in Stage 2). On 3m
with only 3 agents, indirect coupling barely exists so iterations shouldn't help.

### Stage 4: Add CTM Components (Decay + NLM)

**Goal:** Test whether CTM-specific components improve over plain iterations.

**What to do:**
1. Add learned decay over iteration history for the coupling matrix
   (from `compute_synchronisation` in ctm_jax.py — adapt the decay logic)
2. Add NLM trajectory compression: store each agent's embedding at each iteration,
   feed the trajectory (shape: embed_dim x K) into NLM from ctm_jax.py
3. NLM output replaces final embedding as input to value head

**What we learn:** Two ablations:
- Decay only vs no decay: does weighting iteration history help?
- NLM vs just-take-final-embedding: does trajectory compression help?

If neither helps, the CTM-specific parts don't add value and the paper is about
iterative coupling alone (Stage 3). If they help, the paper has a clean
CTM-inspired story.

**Run:** Larger maps. Compare Stage 4 vs Stage 3.

### Stage 5: Ablations and Paper Experiments

**Goal:** Full experiment suite for the paper.

**What to do:**
1. Baselines: MAPPO-GRU (shared V), MAPPO-GRU (per-agent V, mean pool — Stage 1)
2. Ablations: static coupling (Stage 2), iterative coupling (Stage 3),
   iterative + CTM components (Stage 4)
3. Analysis: visualise coupling matrices over training, show coordination structure
   emerging. Compare early vs late training. Show coupling on different scenarios.
4. Scale test: 3m, 5m, 10m — show that benefit grows with team size

**Run:** Multiple seeds, proper statistical comparison.

---

## Sizing and Hyperparameters (starting points)

Based on the existing GRU baseline config:

```
# Actor (unchanged)
GRU_HIDDEN_DIM: 128
FC_DIM_SIZE: 128

# Critic coupling module (new)
AGENT_EMBED_DIM: 64          # per-agent embedding size
COUPLE_HIDDEN_DIM: 64        # MLP inside coupling computation
COUPLING_ITERATIONS: 3       # K rounds of refinement
NLM_MEMORY_LENGTH: 3         # = K, one slot per iteration
NLM_HIDDEN_DIM: 32           # NLM internal hidden dim
NLM_D_MODEL: 64              # = AGENT_EMBED_DIM
DECAY_INIT: 0.0              # learned decay starts uniform
```

These are small. The coupling module should be lightweight relative to the actors
and standard critic — it's learning structure, not memorising trajectories.
