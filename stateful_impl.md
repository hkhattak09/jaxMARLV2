# Stateful CTM + Recurrent Critic Implementation Notes

## Overview
Replacing stateless CTM (iterations=4, fresh hidden states every step) with:
1. **Stateful CTM actor** (iterations=1, hidden states carried across 200 episode steps)
2. **Recurrent critic** (LSTM after mean aggregation in AggregatingCritic)
3. **Episode-sequence replay buffer** (stores complete episodes, samples 32-step chunks with 16-step burn-in)

## Agreed Hyperparameters
| Parameter | Value |
|---|---|
| sequence_length | 32 (16 burn-in + 16 training) |
| burn_in_length | 16 (matches CTM memory_length) |
| lstm_hidden_dim | 64 (matches aggregation embed_dim) |
| num_sequences per batch | 16 |
| updates_per_episode | 8 |
| ctm_iterations | 1 |
| GPU target | L4 (24GB), T4 fallback |

## Implementation Order
1. `episode_buffer.py` — episode-sequence storage and sampling
2. `networks.py` — LSTM in AggregatingCritic
3. `cfg/assembly_cfg.py` — new params
4. `maddpg.py` — sequence-based update with burn-in
5. `agents.py` / `ctm_agent.py` — critic hidden state handling
6. `train_assembly_jax_gpu.py` — stateful rollout, episode-based buffer push
7. `eval/eval_shapes.py` — stateful eval

## Files Changed Tracker
- `algorithm/utils/episode_buffer.py` — NEW (Phase 1)
- `algorithm/utils/__init__.py` — MODIFIED: added EpisodeSequenceBuffer, removed deprecated buffer_episode/buffer_expert
- `algorithm/utils/buffer_episode.py` — DELETED (deprecated, unused)
- `algorithm/utils/buffer_expert.py` — DELETED (deprecated, unused)
- `tests/test_episode_buffer.py` — NEW (Phase 1 tests)
- `algorithm/utils/networks.py` — MODIFIED: AggregatingCritic now has LSTM, forward returns (Q, hidden) (Phase 2)
- `tests/test_recurrent_critic.py` — NEW (Phase 2 tests)
- `cfg/assembly_cfg.py` — MODIFIED: ctm_iterations default 4→1, added sequence_length/burn_in_length/lstm_hidden_dim/num_sequences/updates_per_episode (Phase 3)
- `algorithm/algorithms/maddpg.py` — MODIFIED: unpacked (Q, hidden) tuples in update(), added update_sequence() with burn-in (Phase 4)
- `tests/test_sequence_update.py` — NEW (Phase 4 tests)
- `algorithm/utils/agents.py` — MODIFIED: `lstm_hidden_dim` param added, passed to all 4 AggregatingCritic (Phase 5)
- `algorithm/utils/ctm_agent.py` — MODIFIED: same as agents.py (Phase 5)
- `algorithm/algorithms/maddpg.py` — MODIFIED: `lstm_hidden_dim` threaded through __init__ and init_from_env (Phase 5)
- `train/train_assembly_jax_gpu.py` — MODIFIED: passes `cfg.lstm_hidden_dim` to init_from_env (Phase 5)
- `tests/test_agent_lstm_param.py` — NEW (Phase 5 tests)
- `train/train_assembly_jax_gpu.py` — MODIFIED: stateful rollout, episode buffer creation, update_sequence training path (Phase 6)
- `tests/test_stateful_rollout.py` — NEW (Phase 6 tests)
- `train/train_assembly_jax_gpu.py` — MODIFIED: run_eval() and run_final_eval() now stateful (Phase 7)
- `eval/eval_shapes.py` — MODIFIED: _evaluate_single_model() now stateful (Phase 7)
- `tests/test_stateful_eval.py` — NEW (Phase 7 tests)
- `tests/test_gradient_hotpaths.py` — NEW (gradient flow & hot path tests)
- `tests/run_all_tests.py` — NEW (unified test runner, all phases)

---

## Phase 1: Episode-Sequence Replay Buffer — COMPLETE

### What was done
- Created `EpisodeSequenceBuffer` in `algorithm/utils/episode_buffer.py`
- Stores complete episodes in pre-allocated numpy arrays: `(max_episodes, episode_length, feature_dim)`
- `push_episode()` accepts column-major data from training loop bulk transfer, converts to joint rows internally
- `sample()` returns time-first tensors: `(sequence_length, num_sequences, feature_dim)`
- Handles N>1 rollout threads (splits into separate episodes)
- Circular overwrite when buffer is full
- Removed deprecated `buffer_episode.py` and `buffer_expert.py` (unused)
- Updated `__init__.py` exports
- Test file: `tests/test_episode_buffer.py` (12 tests)

### Key Shapes
- Push input: `(T, obs_dim, N*n_a)` column-major (from `torch.stack(obs_list).cpu().numpy()`)
- Internal storage: `(max_episodes, episode_length, n_agents*obs_dim)` joint rows
- Sample output: `(sequence_length, num_sequences, n_agents*obs_dim)` time-first

### Burn-in / training split
NOT handled by the buffer — the consumer (maddpg.py update) will split:
- `[:burn_in_length]` → forward without gradient to reconstruct hidden states
- `[burn_in_length:]` → forward with gradient for actual loss computation

---

## Phase 2: LSTM in AggregatingCritic — COMPLETE

### What was done
- Modified `AggregatingCritic` in `algorithm/utils/networks.py`
- Added LSTM layer after mean aggregation, before head MLP
- New constructor param: `lstm_hidden_dim=64` (default)
- Forward signature changed: `forward(X, hidden=None)` → `(Q, new_hidden)`
- Added `get_initial_hidden(batch_size, device)` method
- LSTM uses `batch_first=True`, processes aggregated embedding as `seq_len=1`
- Test file: `tests/test_recurrent_critic.py` (12 tests)

### Interface (important for downstream phases)
```python
critic = AggregatingCritic(n_agents, obs_dim, act_dim, hidden_dim=180,
                           embed_dim=64, lstm_hidden_dim=64)
# Single timestep:
Q, new_hidden = critic(X, hidden=None)           # X: (batch, n_agents*(obs_dim+act_dim))
Q, new_hidden = critic(X, hidden=(h_0, c_0))     # carry forward
# hidden shapes: (h, c) each (1, batch, 64)
```

### What this breaks (must fix in later phases)
- **maddpg.py (Phase 4)**: All `curr_agent.critic(vf_in)` calls now return `(Q, hidden)` tuple — must unpack
- **agents.py (Phase 5)**: `DDPGAgent.__init__` passes `hidden_dim=180` to AggregatingCritic but not `lstm_hidden_dim` — need to add param
- **ctm_agent.py (Phase 5)**: Same as agents.py

---

## Phase 3: Config Params — COMPLETE

### What was done
File: `MARL-LLM/marl_llm/cfg/assembly_cfg.py`

Added new argparse params (in new "Stateful / Recurrent Training" section after CTM config):
- `--sequence_length` (default 32) — length of sampled sequences for training
- `--burn_in_length` (default 16) — prefix replayed without gradient
- `--lstm_hidden_dim` (default 64) — critic LSTM hidden size
- `--num_sequences` (default 16) — number of sequences per training batch
- `--updates_per_episode` (default 8) — gradient updates per episode

Changed existing defaults:
- `--ctm_iterations` default from 4 → 1 (stateful mode: 1 tick per env step)

### Notes
- The old `--batch_size` (default 512) is still used for random transition sampling (MLP+ReplayBufferAgent). Both coexist.
- No test file for this phase — config is just argparse declarations, tested implicitly by all downstream code.
- Access new params as: `config.sequence_length`, `config.burn_in_length`, `config.lstm_hidden_dim`, `config.num_sequences`, `config.updates_per_episode`

---

## Phase 4: Sequence-based update with burn-in (maddpg.py) — COMPLETE

### What was done
1. **Fixed `update()` for (Q, hidden) tuples**: All critic calls now unpack the tuple.
   - `curr_agent.target_critic(trgt_vf_in)` → `target_Q1, _ = curr_agent.target_critic(trgt_vf_in)`
   - Same for `target_critic2`, `critic`, `critic2`, and critic in actor loss
   - `hidden=None` used implicitly (no temporal context for random transitions)

2. **Added `update_sequence()` method**: Full sequence-based update with R2D2-style burn-in.
   - Input: time-first tensors `(seq_len, num_seq, feat)` from `EpisodeSequenceBuffer.sample()`
   - **Burn-in phase** (`[:burn_in_length]`):
     - All 4 critics (critic, critic2, target_critic, target_critic2) LSTM hidden states reconstructed
     - Main critics burn-in on `(obs, acs)`; target critics on `(next_obs, target_acs)`
     - For CTM actor: actor hidden state also reconstructed during burn-in
     - All done under `torch.no_grad()`, hidden states detached at boundary
   - **Training phase** (`[burn_in_length:]`):
     - TD losses accumulated per-timestep, averaged over `train_len`
     - Critic update: MSE between Q and target value (min of 2 target critics)
     - Actor update (delayed, every 2 iters): recompute agent_i's actions, evaluate Q
     - Prior regularization supported per-timestep for `prior_mode='regularize'`
   - `target_policies()` unchanged — called per-timestep with fresh hidden state (acceptable since target actions are smoothed with noise anyway)

### Interface
```python
vf_loss, pol_loss, reg_loss = maddpg.update_sequence(
    obs_seq, acs_seq, rews_seq, next_obs_seq, dones_seq,
    agent_i=0, prior_seq=prior_seq, alpha=0.5, burn_in_length=16)
```

### What this breaks / requires in later phases
- **agents.py / ctm_agent.py (Phase 5)**: Must pass `lstm_hidden_dim` to AggregatingCritic constructors
- **train_assembly_jax_gpu.py (Phase 6)**: Must call `update_sequence()` instead of `update()` when using episode buffer

---

## Phase 5: Agent Hidden State Handling (agents.py / ctm_agent.py) — COMPLETE

### What was done
1. **`DDPGAgent.__init__`**: Added `lstm_hidden_dim=64` parameter, passed to all 4 `AggregatingCritic` constructors.
2. **`CTMDDPGAgent.__init__`**: Same — added `lstm_hidden_dim=64`, passed to all 4 critics.
3. **`MADDPG.__init__`**: Added `lstm_hidden_dim=64` parameter, threaded to both `DDPGAgent` and `CTMDDPGAgent` constructors.
4. **`MADDPG.init_from_env`**: Added `lstm_hidden_dim=64` parameter, included in `init_dict` for checkpoint save/load.
5. **`train_assembly_jax_gpu.py`**: Added `lstm_hidden_dim=cfg.lstm_hidden_dim` to `init_from_env()` call.
6. Test file: `tests/test_agent_lstm_param.py` (14 tests)

### Backward compatibility
- All new params default to `64`, so existing code/checkpoints without `lstm_hidden_dim` work unchanged.
- `init_from_save` loads `init_dict` from checkpoint. Old checkpoints missing `lstm_hidden_dim` will use `__init__` default of 64.

### Files changed
- `algorithm/utils/agents.py` — MODIFIED: `lstm_hidden_dim` param added to `__init__`, passed to critics
- `algorithm/utils/ctm_agent.py` — MODIFIED: same
- `algorithm/algorithms/maddpg.py` — MODIFIED: `lstm_hidden_dim` threaded through `__init__` and `init_from_env`
- `train/train_assembly_jax_gpu.py` — MODIFIED: passes `cfg.lstm_hidden_dim` to `init_from_env`
- `tests/test_agent_lstm_param.py` — NEW (Phase 5 tests)

---

## Phase 6: Stateful Rollout in Training Loop — COMPLETE

### What was done
File: `MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py`

1. **Import**: Added `EpisodeSequenceBuffer` alongside `ReplayBufferAgent`.

2. **Conditional buffer creation**: CTM mode creates `EpisodeSequenceBuffer` (max_episodes = buffer_length // episode_length). MLP mode keeps `ReplayBufferAgent`. The unused buffer is set to `None`.

3. **Stateful rollout**: Actor hidden states initialized once at episode start (before the step loop), then carried forward across all 200 steps via `new_hidden_states` return from `maddpg.step()`. For seed mode, prior seeds the hidden state only at episode start; CTM dynamics carry forward naturally.

4. **Episode buffer push**: After bulk GPU→CPU transfer, complete episode pushed to `episode_buffer.push_episode()` (CTM) or per-transition to `agent_buffer[0].push()` (MLP).

5. **Sequence-based training**: CTM path calls `update_sequence()` with `cfg.updates_per_episode` iterations, sampling `cfg.num_sequences` sequences from episode buffer each time. MLP path unchanged (10 iterations, random-transition `update()`). Target networks updated after each update round.

### Key design decisions
- **CTM ↔ episode buffer, MLP ↔ replay buffer**: Keyed on `cfg.use_ctm_actor`. The MLP + LSTM critic ablation (from ablation table) can be added later by decoupling the buffer choice from actor type.
- **Prior not re-computed during rollout**: In stateful mode, `env.compute_prior()` is only called at episode start for seed mode. The prior for regularization loss comes from `env.step()` return (stored in `prior_list`), which is computed every step regardless.
- **Hidden state detachment**: Not needed during rollout — hidden states are detached naturally by `torch.no_grad()` during `maddpg.step()` in eval mode.

### Interface (training loop flow)
```
episode start:
    hidden_states = init (seed or zero)
    for t in range(episode_length):
        actions, _, new_hidden = maddpg.step(obs, ..., hidden_states)
        hidden_states = new_hidden   # carry forward
    episode_buffer.push_episode(obs_batch, acs_batch, ...)

training:
    for _ in range(updates_per_episode):
        sample = episode_buffer.sample(num_sequences, to_gpu=True)
        maddpg.update_sequence(sample, ..., burn_in_length)
        maddpg.update_all_targets()
```

### Test file
`tests/test_stateful_rollout.py` — 16 tests covering:
- Episode buffer push with single/multi env, sample shapes, GPU flag
- update_sequence end-to-end with buffer samples, prior, multi-update loop
- MLP path regression (update() still works)
- Hidden state carry-forward and initialization
- Buffer conditional logic and circular overwrite
- Burn-in/training split correctness

---

## Phase 7: Stateful Eval — COMPLETE

### What was done
Made all three eval rollout paths stateful — hidden states initialized once at episode start
and carried forward across all episode steps (matching the stateful training rollout from Phase 6).

**Files modified:**

1. **`train/train_assembly_jax_gpu.py` — `run_eval()`**:
   - Hidden state init moved before the step loop (once per eval episode)
   - `new_eval_hidden` from `maddpg.step()` carried forward each step
   - Seed mode: prior seeds hidden state at episode start only
   - MLP path: `eval_hidden = None` throughout (unchanged)

2. **`train/train_assembly_jax_gpu.py` — `run_final_eval()`**:
   - Same pattern as `run_eval()` — init before loop, carry forward
   - Hidden states reinitialised fresh at each episode start (per shape × per episode)

3. **`eval/eval_shapes.py` — `_evaluate_single_model()`**:
   - Same pattern — init before step loop, carry forward
   - Hidden state init inside `torch.no_grad()` block (already present)

### Key design decisions
- **Eval matches training**: Stateful eval uses the same carry-forward pattern as stateful
  training rollout (Phase 6). This ensures eval measures the same policy behavior that was
  trained — no train/eval mismatch.
- **Hidden states reset per episode**: Each eval episode starts with fresh hidden states
  (zero or prior-seeded). No state leaks between episodes.
- **MLP path unchanged**: When `use_ctm_actor=False`, `eval_hidden=None` and no carry-forward
  happens — identical to pre-Phase 7 behavior.

### Test file
`tests/test_stateful_eval.py` — 12 tests covering:
- Hidden states evolve across eval steps (not reinitialized)
- Hidden states differ at each step
- MLP eval path passes None hidden states
- Hidden states reset between episodes (no cross-episode leak)
- `torch.no_grad()` does not prevent LSTM hidden state updates
- Hidden states under no_grad have no grad_fn
- Eval and training produce identical hidden trajectories for same inputs
- update() and update_sequence() still work (no regression)

---

## Phase 8: TBD (will be filled as implementation proceeds)
