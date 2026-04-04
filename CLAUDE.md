# CLAUDE.md — Project Context

## Project Overview

MADDPG multi-agent RL system for a 24-agent assembly/flocking task. JAX environment (GPU,
vmap parallel envs) + PyTorch networks. DLPack zero-copy GPU bridge. Everything stays on
GPU throughout (rollout, training, and eval). Full documentation in `Docs/`.

Key files:
- `MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py` — main training loop
- `MARL-LLM/marl_llm/algorithm/algorithms/maddpg.py` — MADDPG orchestrator
- `MARL-LLM/marl_llm/algorithm/utils/agents.py` — DDPGAgent (MLP actor+critic, unchanged)
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

**Only file that needs changing:** `train/train_assembly_jax_gpu.py`
- Remove per-episode hidden state init and carry-forward between steps
- Remove done-mask reset logic
- Call `get_initial_hidden_state()` fresh before every `maddpg.step()` in rollout, run_eval, run_final_eval
- All other files (ctm_actor.py, ctm_agent.py, maddpg.py, buffer, critic) unchanged

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
- `run_eval` / `run_final_eval`: per-episode hidden state init on CUDA + 3-tuple step unpack
  — both already run on GPU (`prep_rollouts(device="gpu")`, `torch.device('cuda')`)

**`cfg/assembly_cfg.py` — changes:**
- CTM is now the default. Replaced `--use_ctm_actor` (store_true) with:
  - `--use_mlp_actor` (store_false, dest="use_ctm_actor") — pass this flag to use MLP
  - `parser.set_defaults(use_ctm_actor=True)`
- All downstream code reading `cfg.use_ctm_actor` is unchanged

**`eval/eval_shapes.py` — changes:**
- Fixed 2-tuple unpack → 3-tuple: `actions, _, eval_hidden = maddpg.step(..., hidden_states=eval_hidden)`
- Added per-episode hidden state init: `get_initial_hidden_state(env.n_a, torch_device)` guarded by `maddpg.use_ctm_actor`
- Moved from CPU to GPU: `prep_rollouts(device=device)` where device='gpu' when CUDA available
- Removed redundant `obs.cpu()` conversion (obs is already a CUDA tensor from JAX adapter)
- `maddpg.use_ctm_actor` is loaded correctly from saved `init_dict` — auto-detects CTM vs MLP

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
- Critic: MLP, unchanged

### Stateless Rollout (NEW — replacing stateful rollout)
- During rollout: hidden states reinitialised fresh before every step (no cross-timestep carry)
- During actor update: hidden states also fresh — rollout and update are now identical
- No hidden states stored in replay buffer — buffer unchanged
- CTM value-add is iterative computation within each timestep (iterations=4 inner passes)
- Cross-timestep memory removed — justified because environment is fully observable

### Target Network
- Target actor uses stateless initialisation — consistent with rollout and update

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

## Key Constraints (Still Apply)

- Do NOT modify DDPGAgent — subclass only, MLP version must remain intact
- Do NOT change replay buffer interface
- Do NOT change critic (MLPNetwork)
- CTM import path: `from models.ctm_rl import ContinuousThoughtMachineRL`
  (ctm_actor.py adds `continuous-thought-machines/` to sys.path at import time)
- `backbone_type='classic-control-backbone'` for flat vector observations
- Hidden states are NOT saved in checkpoints (transient rollout state);
  `start_activated_trace` IS saved (it's an nn.Parameter in the network weights)
- `maddpg.step()` returns a 3-tuple everywhere — all call sites updated (training loop,
  run_eval, run_final_eval, eval_shapes.py)
- `use_ctm_actor` is stored in `init_dict` and saved/loaded with the model checkpoint —
  loading a CTM model automatically sets `maddpg.use_ctm_actor=True`
