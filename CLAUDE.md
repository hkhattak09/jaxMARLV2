# CLAUDE.md — Project Context

## Project Overview

MADDPG multi-agent RL system for a 24-agent assembly/flocking task. JAX environment (GPU,
vmap parallel envs) + PyTorch networks. DLPack zero-copy GPU bridge. Everything stays on
GPU throughout (rollout, training, and eval). Full documentation in `Docs/`.

Key files:
- `MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py` — main training loop
- `MARL-LLM/marl_llm/algorithm/algorithms/maddpg.py` — MADDPG orchestrator
- `MARL-LLM/marl_llm/algorithm/utils/agents.py` — DDPGAgent (MLP actor+critic; now has `critic_hidden_dim` param)
- `MARL-LLM/marl_llm/algorithm/utils/buffer_agent.py` — ReplayBufferAgent (joint rows — one per timestep)
- `MARL-LLM/marl_llm/algorithm/utils/networks.py` — MLPNetwork
- `MARL-LLM/marl_llm/cfg/assembly_cfg.py` — config (CTM default, pass --use_mlp_actor for MLP)
- `continuous-thought-machines/models/ctm_rl.py` — ContinuousThoughtMachineRL base class
- `Docs/CTM_ACTOR_DESIGN.md` — full design decisions document
- `MARL-LLM/marl_llm/eval/eval_shapes.py` — standalone post-training eval script
- `MARL-LLM/marl_llm/tests/test_ctm_implementation.py` — comprehensive test suite

**CTM actor files:**
- `MARL-LLM/marl_llm/algorithm/utils/ctm_actor.py` — CTMActor class
- `MARL-LLM/marl_llm/algorithm/utils/ctm_agent.py` — CTMDDPGAgent class

---

## CTM Actor — STATELESS ROLLOUT TRANSITION IN PROGRESS

**CTM is the default actor.** To revert to MLP: pass `--use_mlp_actor` to the training command.
All eval paths (training-loop eval, final eval, standalone eval_shapes.py) work with both.

### Why We Are Changing

The original implementation propagated hidden states across timesteps during rollout (stateful)
but initialised fresh hidden states during actor updates (stateless). This caused a
rollout/update mismatch: the critic was trained on actions from warm context-rich hidden states,
but the policy gradient computed actions from a blank board — causing Q-overestimation and
policy loss diving to -6 with near-zero task improvement (coverage stuck at 0.06 vs MLP 0.64
at the same episode count).

Storing hidden states in the replay buffer (the R-MADDPG approach) is the principled fix but
is impractical: CTM hidden state = 2 × (256 × 16) = 8,192 floats per agent per transition,
vs LSTM = 128 floats. At buffer_length=20k, 24 agents: ~15.7 GB — infeasible. Staleness of
complex high-dimensional hidden states under off-policy sampling is also a harder problem than
for small LSTMs.

### What We Are Doing Instead — Stateless Rollout

Reset hidden states to fresh at every timestep during rollout, exactly as actor updates do.
Rollout and update become structurally identical — zero mismatch. The CTM still contributes
via its `iterations=4` inner reasoning passes per observation (iterative computation within a
single timestep). Cross-timestep memory is removed. The environment is fully observable so
the MLP baseline proves cross-timestep memory is not required.

**For stateless rollout only:** `train/train_assembly_jax_gpu.py`
- Remove per-episode hidden state init and carry-forward between steps
- Remove done-mask reset logic
- Call `get_initial_hidden_state()` fresh before every `maddpg.step()` in rollout, run_eval, run_final_eval
- ctm_actor.py, ctm_agent.py unchanged for the stateless rollout change
- Note: maddpg.py, buffer, and agents.py were subsequently changed for the centralised critic (see below)

### What Was Implemented

**`ctm_actor.py` — CTMActor(nn.Module)**
- Wraps `ContinuousThoughtMachineRL` + `nn.Linear(136, 2)` action head + Tanh
- `backbone_type='classic-control-backbone'`, `heads=0`, `n_synch_action=0`
- `get_initial_hidden_state(batch_size, device)`: returns `(zeros, start_activated_trace.expand(batch_size))`
  — gradients flow through `start_activated_trace` during stateless actor updates
- `forward(obs, hidden_states)` → `(actions, new_hidden_states)`
- sys.path manipulation at top of file adds `continuous-thought-machines/` for CTM imports

**`ctm_agent.py` — CTMDDPGAgent(DDPGAgent)**
- Does NOT call `DDPGAgent.__init__` (would create unwanted MLP policy)
- Does a dummy forward pass on both policy and target_policy before `hard_update` to
  materialize `nn.LazyLinear` layers — critical, otherwise both networks would materialize
  independently with different random weights
- Critic and target_critic use MLPNetwork with `critic_hidden_dim` (default 256, separate from actor `hidden_dim`)
- `step(obs, hidden_states, explore=False)` → `(action.t(), None, new_hidden_states)`
  — log_pi is None, discarded with `_` in training loop
- `scale_noise` / `reset_noise` inherited from DDPGAgent (work via `self.exploration`)

**`maddpg.py` — changes (stateless rollout + centralised critic):**
- `__init__`: `use_ctm_actor=False`, `ctm_config=None`, `n_agents=None`, `critic_hidden_dim=None` params;
  stores `self.use_ctm_actor` and `self.n_agents`; passes `critic_hidden_dim` to agent constructors
- `step()`: returns 3-tuple `(actions, log_pis, new_hidden_states)` always;
  `new_hidden_states` is None for MLP. Accepts `hidden_states=None` param.
- `target_policies(agent_i, next_obs_all)`: takes joint next_obs `(batch, n_agents*obs_dim)`;
  reshapes to `(batch*n_agents, obs_dim)`, runs through target policy, returns `(batch, n_agents*2)`.
  For CTM: fresh hidden state at size `batch*n_agents` (stateless).
- `update()` critic: joint `vf_in = cat([obs_all, acs_all])` shape `(batch, n_agents*(obs_dim+2))`
- `update()` actor (Option B): reshapes joint obs to `(batch*n_agents, obs_dim)`, runs ALL agents
  through shared policy, reshapes back to `(batch, n_agents*2)` — gradient through all 24 slots
- `init_from_env()`: computes `n_agents = env.num_agents`;
  `dim_input_critic = n_agents * (obs_dim + action_dim)` (true centralised critic);
  stores `n_agents` and `critic_hidden_dim` in `init_dict` for save/load

**`train_assembly_jax_gpu.py` — changes:**
- Builds `ctm_config` dict from cfg args when `cfg.use_ctm_actor`; passes to `init_from_env`
- Per-episode start: resets hidden states via `get_initial_hidden_state(n_rollout_threads * n_a, cuda)`
- Rollout loop:
  - `maddpg.step()` now 3-tuple unpack everywhere (all call sites: main loop + run_eval + run_final_eval)
  - After step: detaches hidden_states to prevent 200-step computation graph accumulation
  - After env.step(): done mask reset — `dones_gpu.reshape(-1, 1, 1).float()` broadcasts over
    `(N*n_a, d_model, memory_length)`, resets state_trace to zeros and activated_trace to
    `start_activated_trace` for done agents
- `run_eval` / `run_final_eval`: per-episode hidden state init on CUDA + 3-tuple step unpack
  — both already run on GPU (`prep_rollouts(device="gpu")`, `torch.device('cuda')`)

**`cfg/assembly_cfg.py` — changes:**
- CTM is now the default. Replaced `--use_ctm_actor` (store_true) with:
  - `--use_mlp_actor` (store_false, dest="use_ctm_actor") — pass this flag to use MLP
  - `parser.set_defaults(use_ctm_actor=True)`
- All downstream code reading `cfg.use_ctm_actor` is unchanged
- Added `--critic_hidden_dim` (default 256) — controls centralised critic hidden layer width
  (separate from `--hidden_dim` which controls the actor)

**`eval/eval_shapes.py` — changes:**
- Fixed 2-tuple unpack → 3-tuple: `actions, _, eval_hidden = maddpg.step(..., hidden_states=eval_hidden)`
- Added per-episode hidden state init: `get_initial_hidden_state(env.n_a, torch_device)` guarded by `maddpg.use_ctm_actor`
- Moved from CPU to GPU: `prep_rollouts(device=device)` where device='gpu' when CUDA available
- Removed redundant `obs.cpu()` conversion (obs is already a CUDA tensor from JAX adapter)
- `maddpg.use_ctm_actor` is loaded correctly from saved `init_dict` — auto-detects CTM vs MLP

**`eval/eval_shapes.py` — additional changes (new metrics + stateless fix):**
- Fixed stateful CTM bug: was initialising hidden state once per episode and carrying across
  steps — now stateless (fresh every step), matching training behaviour
- Added `mean_neighbor_distance`, `collision_rate` to all eval outputs (periodic, final, summary,
  pickle, comparison table)
- Pass `--topo_nei_max N` when evaluating K≠6 models so env obs_dim matches checkpoint

**`eval/eval_assembly.py`** — deprecated legacy script, not used.

**`tests/test_ctm_implementation.py` — new test suite:**
- Run with: `python tests/test_ctm_implementation.py` or `python -m pytest tests/test_ctm_implementation.py -v`
  from `MARL-LLM/marl_llm/`
- Covers: CTMActor shapes/forward/gradient flow, CTMDDPGAgent init (LazyLinear fix)/step/save-load,
  MADDPG+CTM full update cycle, MADDPG+MLP backwards compatibility, hidden state done-mask reset,
  detach graph-cutting, temporal context, end-to-end rollout+update smoke tests for both actor types

---

## All Design Decisions (Current)

### Architecture
- Single shared CTM network across all 24 agents (parameter sharing, homogeneous team)
- Each agent has its own hidden state — reinitialised fresh every step under stateless rollout
- Attention and KV projections removed (heads=0) — flat 192-dim obs vector, no sequence
- Action head: Linear(136, 2) + Tanh on top of synchronisation output
- Critic: centralised MLP — sees ALL agents' obs + ALL agents' actions during training (true CTDE)
  - Input dim: `n_agents × (obs_dim + 2)` = 4,656 at K=6 M=80
  - Hidden dim: 256 (`--critic_hidden_dim`, separate from actor's `--hidden_dim=180`)
  - Architecture: 4,656 → 256 → 256 → 256 → 1

### Stateless Rollout (NEW — replacing stateful rollout)
- During rollout: hidden states reinitialised fresh before every step (no cross-timestep carry)
- During actor update: hidden states also fresh — rollout and update are now identical
- No hidden states stored in replay buffer — buffer unchanged
- CTM value-add is iterative computation within each timestep (iterations=4 inner passes)
- Cross-timestep memory removed — justified because environment is fully observable

### Centralised Critic (CTDE — implemented)

The previous implementation was **not true MADDPG**: it used parameter sharing (one network)
but the critic saw only one agent's obs+action (194 dims). This is independent DDPG with shared
weights. The centralised critic is now implemented.

**What changed:**
- `buffer_agent.py`: stores **joint rows** — one row per timestep with all 24 agents' data
  concatenated. `total_length = max_steps` (not `max_steps × n_agents`). No `index` slicing.
  - `obs_buffs`: `(max_steps, n_agents × obs_dim)` — e.g. `(20000, 4608)`
  - `ac_buffs`: `(max_steps, n_agents × 2)` — e.g. `(20000, 48)`
  - Rewards: mean across agents → `(max_steps, 1)`. Dones: max → `(max_steps, 1)`.
- `agents.py` / `ctm_agent.py`: added `critic_hidden_dim` param (default: falls back to `hidden_dim`).
  Critic and target_critic constructed with `critic_hidden_dim` instead of `hidden_dim`.
- `maddpg.py`: `dim_input_critic = n_agents × (obs_dim + 2)`. New `target_policies` reshapes
  joint `next_obs_all` → runs all agents through target policy → joint target actions.
  Actor update uses **Option B**: recomputes all 24 agents' actions from joint obs through shared
  policy — gradient through all 24 action slots simultaneously.
- `train_assembly_jax_gpu.py`: `push()` call drops `index` arg; buffer constructed without
  `start_stop_index`; `critic_hidden_dim=cfg.critic_hidden_dim` passed to `init_from_env`.

**Why Option B for actor update:** With a shared policy, recomputing all 24 agents' actions is
internally consistent (no staleness) and gives 24× stronger gradient signal. Standard MADDPG
Option A (substitute only agent i's action) is designed for independent policies; Option B is
correct when parameter sharing means all 24 agents use the same network.

**Why critic_hidden_dim=256:** Input goes from 194 (old) to 4,656 (new). Keeping hidden_dim=180
would create a 26× compression in the first layer — too aggressive. 256 gives an 18× compression,
which is reasonable for a 4-layer MLP processing rich joint state.

### Target Network
- Target actor uses stateless initialisation — consistent with rollout and update
- Target critic receives joint `(next_obs_all, target_acs_all)` — same joint structure as critic

### Hyperparameters
| Parameter | Value | Note |
|---|---|---|
| d_model | 256 | GPU rollout makes this cheap; 128 would be underpowered vs MLP hidden_dim=180 |
| memory_length | 16 | ~8% of episode (200 steps) |
| n_synch_out | 16 | Output size = 16×17/2 = 136-dim |
| iterations | 4 | Overwrites 25% of board per step; tune over [3,4,5,6] empirically |
| synapse_depth | 1 | 2-block MLP (Linear→GLU→LayerNorm ×2) — note: synapse_depth=1 creates TWO LazyLinear blocks |
| deep_nlms | False | 68× cheaper than deep, sufficient expressiveness |
| do_layernorm_nlm | True | Training stability |
| memory_hidden_dims | [64] | Passed to constructor, unused when deep_nlms=False |
| dropout | 0 | Always 0 for RL — not exposed in cfg |

---

## Research Direction — Prior-Seeded Iterative Reasoning

**Goal:** Publish a paper combining CTM, MARL, and physics priors in a principled way.

### What the environment actually is

Each agent observes: own state (4) + K=6 nearest neighbors (24) + 80 nearest target cells (160) = 192 dims.
Agents observe only 6 of 23 teammates — the actor is **genuinely partially observable** at execution.
The centralised critic now sees ALL agents' joint obs+actions during training (true CTDE — implemented).
At M=10 across 24 agents, the union of all agents' local obs covers most of the shape, giving the
critic a near-complete view of the joint state even when each actor sees only 10 cells.

The Reynolds prior is computed every step from the same local obs — it is a physics-based closed-form
action (flocking: cohesion, alignment, separation). Currently used only as a 0.03-weight loss
regularizer in the actor update. This is an underused asset.

### Critical insight — the task was too easy

**The 80 target cells in the observation make the shape almost fully known to every agent.**
At `num_obs_grid_max=80`, each agent sees the positions of up to 80 target cells — essentially
the entire shape. This means each agent can trivially navigate to the nearest uncovered cell
independently, without needing to coordinate with teammates at all. You could hand-code this
behaviour without any learning. Reducing K (neighbor visibility) barely matters because the
hard part of the task — "where do I go?" — is already answered by the shape observation.

This is why CTM and MLP perform similarly across all K values: the architecture doesn't matter
when the task reduces to local reactive navigation on a fully visible map.

**The fix — two axes of partial observability:**
- `topo_nei_max` (K): how many teammates each agent can see (already configurable)
- `num_obs_grid_max` (M): how many target cells each agent can see — **this is the missing axis**

Reducing M (e.g. to 10-15 cells) makes the shape genuinely unknown to each agent. Now agents
must spread across unknown territory using only local teammate information — exactly the regime
where flocking priors and iterative reasoning matter. An MLP with limited obs can only react
to what it sees. A prior-seeded CTM starts from Reynolds flocking (spread out, align, separate)
and refines from there, which is the right inductive bias for covering an unknown shape.

### Core Idea — Prior-Seeded CTM

Instead of regularizing the CTM's output toward the prior, use the prior to **initialize** the CTM's
iterative computation. The CTM then refines from a physics-grounded starting point rather than from zeros.

```
Current:   zeros + start_activated_trace → CTM iterations 1..K → action
Proposed:  obs + prior_action → seed MLP → h_0 → CTM iterations 1..K → action
```

**Why this is structurally different from MLP+prior:**
- MLP with prior regularization: prior shapes the loss, single feedforward pass
- CTM with prior seeding: prior shapes the starting computational state, iterative refinement
  departs from physics as much as learned — the number of iterations controls
  how far from the prior the policy can move
- Under partial observability, when an agent can't see most teammates, the prior provides
  a meaningful starting point that the CTM refines. MLP cannot use the prior this way.

### Ablation Table (revised)

The K-only sweep was the wrong experiment — varying K while keeping M=80 doesn't stress the
task enough because the shape is still fully visible. The meaningful ablation varies M (shape
visibility) and K (teammate visibility) together.

| Model | M (shape cells) | K (neighbors) | Expected |
|---|---|---|---|
| MLP | 80 (full) | 6 | baseline — task is easy, all solve it |
| CTM zero-init | 80 (full) | 6 | ~MLP — confirmed |
| MLP | 10 (limited) | 3 | degrades — shape unknown, coordination hard |
| CTM zero-init | 10 (limited) | 3 | degrades somewhat |
| CTM prior-seeded | 10 (limited) | 3 | degrades least — prior fills the shape-knowledge gap |

**First step before the full ablation:** Run MLP at M=10, K=3 for 500 episodes.
If MLP visibly struggles (coverage drops significantly vs M=80), the regime is real.
If MLP still solves it easily, reduce M further until the task is actually hard.
Only then implement prior-seeded CTM and run the comparison.

### Framing

*Physics-Prior Seeding for Iterative Reasoning in Cooperative MARL* — agents use domain
knowledge not as a behavioral cloning target but as initialization for learned iterative
refinement, granting robustness to partial observability that single-pass architectures
cannot structurally replicate.

### K=3 Results (500 episodes, completed)

CTM and MLP were both run at K=3 for 500 episodes. Final eval summary:

| Metric | CTM | MLP | Notes |
|---|---|---|---|
| Coverage | 0.6857 | 0.6953 | MLP +1.4% — negligible |
| Dist Uniformity | **0.9259** | 0.8800 | CTM +5.2% — consistent |
| Voronoi | 0.8233 | **0.8516** | MLP +3.4% — contradicts uniformity |
| Reward | 0.1660 | 0.1590 | CTM +4.4% — noisy |

**Visual observation (final eval GIFs):** CTM agents actively maintain separation — they push
against each other to avoid clustering. MLP agents cluster in the same corner of the shape
repeatedly across episodes, particularly in tight-geometry shapes (B shape at ep100, A shape
at final eval bottom-right leg). The clustering is deterministic, not random — MLP with K=3
cannot resolve overcrowding from limited neighbor info.

**Why the metrics don't fully capture this:** Coverage doesn't penalise stacking. Voronoi
uniformity averages globally so a local cluster in one corner is diluted. The existing
metrics understate the behavioural difference visible in the GIFs.

**Why new metrics were added:**
- `mean_neighbor_distance`: absolute spacing magnitude — directly measures whether agents
  are spread out, not just whether spacing is uniform. Should be clearly higher for CTM.
- `collision_rate`: fraction of agents physically colliding at episode end. CTM's anti-
  clustering behaviour should show lower collision rate.
- These were added to `AssemblyEnv`, `JaxAssemblyAdapterGPU`, training script eval, and
  `eval_shapes.py`. Run eval_shapes.py on existing K=3 checkpoints to get the new numbers.

**Current conclusion:** Mixed numerics, clear visual difference. The gap is real but the old
metrics couldn't see it cleanly. New metrics should clarify. K=1 still to run.

### What to do next (ordered)

0. ~~**Implement true centralised critic (CTDE)**~~ — **done.** Buffer, maddpg, agents, ctm_agent,
   train script all updated. Critic input: `n_agents × (obs_dim+2)`, `critic_hidden_dim=256`.
1. **Check if `num_obs_grid_max` is a configurable CLI parameter** — it's in AssemblyEnv
   constructor but may not be wired to cfg/assembly_cfg.py yet. Wire it if not.
2. **Run MLP at M=10, K=3 for 500 episodes** — confirm the task becomes genuinely hard.
   If coverage drops substantially vs M=80 baseline, the regime is confirmed.
   This is now scientifically valid: both actor and critic are correctly CTDE.
3. **Run CTM zero-init at M=10, K=3** — establish whether zero-init CTM helps at all.
4. **Implement prior-seeded CTM** only if steps 2-3 confirm a regime where MLP struggles.
5. Full ablation table for paper once prior-seeded CTM shows a clear gap.

**Do not run more K-sweep experiments at M=80 — they will not produce a meaningful result.**

### Intellectual honesty rule

Do not get carried away with ideas. An idea is not a result. Before any implementation:
- State what the experiment would show if the hypothesis is **wrong**
- If the experiment cannot falsify the hypothesis, it is not a useful experiment
- Do not frame incremental or null results as confirmation
- If K-sweep shows no gap between CTM and MLP under partial obs, say so clearly and
  reconsider the direction rather than adding complexity to patch it

---

## Key Constraints

- Do NOT modify DDPGAgent actor logic — subclass only for new actor types; MLP version must remain intact
- `critic_hidden_dim` and `n_agents` are stored in `init_dict` and saved/loaded with checkpoints —
  old checkpoints (without these keys) will fail to load; retrain from scratch for centralised critic
- CTM import path: `from models.ctm_rl import ContinuousThoughtMachineRL`
  (ctm_actor.py adds `continuous-thought-machines/` to sys.path at import time)
- `backbone_type='classic-control-backbone'` for flat vector observations
- Hidden states are NOT saved in checkpoints (transient rollout state);
  `start_activated_trace` IS saved (it's an nn.Parameter in the network weights)
- `maddpg.step()` returns a 3-tuple everywhere — all call sites updated (training loop,
  run_eval, run_final_eval, eval_shapes.py)
- `use_ctm_actor` is stored in `init_dict` and saved/loaded with the model checkpoint —
  loading a CTM model automatically sets `maddpg.use_ctm_actor=True`
- Buffer stores joint rows — one per timestep. `push()` takes no `index` argument.
  Sampled `obs`/`acs` have shape `(batch, n_agents*obs_dim)` and `(batch, n_agents*2)` respectively.
- `MLPNetwork` (networks.py) is unchanged — the critic's wider hidden dim is passed as `hidden_dim`
  to its constructor; no structural change to the network class itself
