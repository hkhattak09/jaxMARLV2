# CLAUDE.md — Project Context

## Project Overview

MADDPG multi-agent RL system for a 24-agent assembly/flocking task. JAX environment (GPU,
vmap parallel envs) + PyTorch networks. DLPack zero-copy GPU bridge. Everything stays on
GPU throughout (rollout and training). Full documentation in `Docs/`.

Key files:
- `MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py` — main training loop
- `MARL-LLM/marl_llm/algorithm/algorithms/maddpg.py` — MADDPG orchestrator
- `MARL-LLM/marl_llm/algorithm/utils/agents.py` — DDPGAgent (MLP actor+critic, unchanged)
- `MARL-LLM/marl_llm/algorithm/utils/networks.py` — MLPNetwork
- `MARL-LLM/marl_llm/cfg/assembly_cfg.py` — config (CTM params already added)
- `continuous-thought-machines/models/ctm_rl.py` — ContinuousThoughtMachineRL base class
- `Docs/CTM_ACTOR_DESIGN.md` — full design decisions document

**CTM actor files (newly created):**
- `MARL-LLM/marl_llm/algorithm/utils/ctm_actor.py` — CTMActor class
- `MARL-LLM/marl_llm/algorithm/utils/ctm_agent.py` — CTMDDPGAgent class

---

## CTM Actor — IMPLEMENTATION COMPLETE

All 5 steps done. To use: add `--use_ctm_actor` to the training command.
Without the flag everything runs the original MLP path unchanged.

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
- Critic and target_critic remain MLPNetwork (hidden_dim from config)
- `step(obs, hidden_states, explore=False)` → `(action.t(), None, new_hidden_states)`
  — log_pi is None, discarded with `_` in training loop
- `scale_noise` / `reset_noise` inherited from DDPGAgent (work via `self.exploration`)

**`maddpg.py` — changes:**
- `__init__`: `use_ctm_actor=False`, `ctm_config=None` params; stores `self.use_ctm_actor`;
  instantiates CTMDDPGAgent when set, DDPGAgent otherwise
- `step()`: now returns 3-tuple `(actions, log_pis, new_hidden_states)` always;
  `new_hidden_states` is None for MLP. Accepts `hidden_states=None` param.
- `target_policies()`: for CTM calls `get_initial_hidden_state` on target_policy (stateless)
- `update()`: for CTM actor, calls `get_initial_hidden_state` for fresh board before policy(obs)
- `init_from_env()`: accepts and passes `use_ctm_actor`, `ctm_config` through to init_dict

**`train_assembly_jax_gpu.py` — changes:**
- Builds `ctm_config` dict from cfg args when `cfg.use_ctm_actor`; passes to `init_from_env`
- Per-episode start: resets hidden states via `get_initial_hidden_state(n_rollout_threads * n_a, cuda)`
- Rollout loop:
  - `maddpg.step()` now 3-tuple unpack everywhere (all call sites: main loop + run_eval + run_final_eval)
  - After step: detaches hidden_states to prevent 200-step computation graph accumulation
  - After env.step(): done mask reset — `dones_gpu.reshape(-1, 1, 1).float()` broadcasts over
    `(N*n_a, d_model, memory_length)`, resets state_trace to zeros and activated_trace to
    `start_activated_trace` for done agents
- run_eval / run_final_eval: per-episode hidden state init + 3-tuple step unpack

---

## All Design Decisions (Final)

### Architecture
- Single shared CTM network across all 24 agents (parameter sharing, homogeneous team)
- Each agent has its own independent hidden state (its own "board")
- Attention and KV projections removed (heads=0) — flat 192-dim obs vector, no sequence
- Action head: Linear(136, 2) + Tanh on top of synchronisation output
- Critic: MLP, unchanged

### Stateless Actor Updates
- During rollout: hidden states propagated correctly across timesteps (full temporal memory)
- During actor update: fresh hidden states from `start_activated_trace` for every batch
- No hidden states stored in replay buffer — buffer unchanged
- Gradient is biased in magnitude but correct in direction (Q-weighted)
- Same `iterations` value for both rollout and update (different values create structural
  mismatch in synchronisation computation between rollout and update)
- Mitigations: learnable decay params upweight reliable recent slots naturally;
  `start_activated_trace` is a learned parameter that optimises to reduce mismatch;
  Reynolds flocking prior stabilises early training

### Hidden State Management
- Shape: `(n_rollout_threads × n_agents, d_model, memory_length)` × 2 tensors
  (state_trace and activated_state_trace)
- Live on GPU throughout — never move (network stays GPU for both rollout and training)
- Initialised at episode start from `get_initial_hidden_state()`
- Done reset in training loop after each env.step():
  - done signal `(1, n_total_agents)` → `.reshape(-1, 1, 1).float()` for broadcasting
  - state_trace where done=True → zeros
  - activated_state_trace where done=True → `start_activated_trace.detach()` expanded
  - Agents where done=False untouched

### Target Network
- Target actor uses its own `start_activated_trace` (gets soft-updated as normal nn.Parameter)
- Stateless initialisation for target actor forward pass in critic update — consistent

### Hyperparameters
| Parameter | Value | Note |
|---|---|---|
| d_model | 256 | GPU rollout makes this cheap; 128 would be underpowered vs MLP hidden_dim=180 |
| memory_length | 16 | ~8% of episode (200 steps) |
| n_synch_out | 16 | Output size = 16×17/2 = 136-dim |
| iterations | 4 | Overwrites 25% of board per step; tune over [3,4,5,6] empirically |
| synapse_depth | 1 | 2-block MLP (Linear→GLU→LayerNorm ×2) |
| deep_nlms | False | 68× cheaper than deep, sufficient expressiveness |
| do_layernorm_nlm | True | Training stability |
| memory_hidden_dims | [64] | Passed to constructor, unused when deep_nlms=False |
| dropout | 0 | Always 0 for RL — not exposed in cfg |

---

## Key Constraints (Still Apply)

- Do NOT modify DDPGAgent — subclass only, MLP version must remain intact
- Do NOT change replay buffer interface
- Do NOT change critic (MLPNetwork)
- CTM import path: `from models.ctm_rl import ContinuousThoughtMachineRL`
  (ctm_actor.py adds `continuous-thought-machines/` to sys.path at import time)
- `backbone_type='classic-control-backbone'` for flat vector observations
- Hidden states are NOT saved in checkpoints (transient rollout state);
  `start_activated_trace` IS saved (it's an nn.Parameter in the network weights)
- `maddpg.step()` now returns a 3-tuple everywhere — all call sites updated
