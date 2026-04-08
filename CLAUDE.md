# CLAUDE.md — Project Context
## Error Handling Philosophy: Fail Loud, Never Fake

Prefer a visible failure over a silent fallback.

- Never silently swallow errors to keep things "working."
  Surface the error. Don't substitute placeholder data.
- Fallbacks are acceptable only when disclosed. Show a
  banner, log a warning, annotate the output.
- Design for debuggability, not cosmetic stability.

Priority order:
1. Works correctly with real data
2. Falls back visibly — clearly signals degraded mode
3. Fails with a clear error message
4. Silently degrades to look "fine" — never do this

## Project Overview

MADDPG multi-agent RL system for a 24-agent assembly/flocking task. JAX environment (GPU,
vmap parallel envs) + PyTorch networks. DLPack zero-copy GPU bridge. Everything stays on
GPU throughout (rollout, training, and eval). Full documentation in `Docs/`.

Key files:
- `MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py` — main training loop
- `MARL-LLM/marl_llm/algorithm/algorithms/maddpg.py` — MADDPG orchestrator
- `MARL-LLM/marl_llm/algorithm/utils/agents.py` — DDPGAgent (MLP actor + AggregatingCritic)
- `MARL-LLM/marl_llm/algorithm/utils/buffer_agent.py` — ReplayBufferAgent (joint rows — one per timestep)
- `MARL-LLM/marl_llm/algorithm/utils/networks.py` — MLPNetwork, AggregatingCritic
- `MARL-LLM/marl_llm/cfg/assembly_cfg.py` — config (CTM default, pass --use_mlp_actor for MLP)
- `continuous-thought-machines/models/ctm_rl.py` — ContinuousThoughtMachineRL base class
- `MARL-LLM/marl_llm/eval/eval_shapes.py` — standalone post-training eval script
- `MARL-LLM/marl_llm/tests/test_ctm_implementation.py` — test suite

**CTM actor files:**
- `MARL-LLM/marl_llm/algorithm/utils/ctm_actor.py` — CTMActor class
- `MARL-LLM/marl_llm/algorithm/utils/ctm_agent.py` — CTMDDPGAgent class

---

## Current State — What Is Implemented

### Architecture (as of now)
- **Actor**: CTMActor (default) or MLPNetwork (pass `--use_mlp_actor`)
  - CTMActor wraps `ContinuousThoughtMachineRL` + `nn.Linear(136, 2)` action head + Tanh
  - Single shared network across all 24 agents (parameter sharing)
  - **Currently stateless**: hidden states reinitialised fresh every step (rollout and update identical)
  - `iterations=4` inner passes per observation, `d_model=128`, `memory_length=16`
- **Critic**: `AggregatingCritic` — permutation-equivariant centralised critic (true CTDE)
  - Shared encoder: `(obs_dim + act_dim)` → 128 → 64, applied independently per agent
  - Mean aggregation over 24 agent embeddings → 64-dim summary
  - Head MLP: 64 → 128 → 1
  - **Currently memoryless** — no temporal reasoning in the critic
- **Buffer**: Joint rows — one row per timestep, all 24 agents concatenated. Random transition sampling.
- **Prior**: Reynolds flocking prior used as MSE loss regularizer (`prior_mode='regularize'`)
  or to seed CTM hidden state (`prior_mode='seed'`). Seed mode also implemented but underperforms.

### What works
- MLP + prior regularization: Coverage 0.916, Voronoi 0.653 (best result so far)
- CTM seed (d_model=256): Coverage 0.835, Voronoi 0.564
- MLP no prior: Coverage 0.756, Voronoi 0.512
- All results with current reward function (proximity + crowding + territory balance)

### Known problems
1. **Stateless CTM doesn't leverage its core innovations.** With 4 iterations and `memory_length=16`,
   the activated_state_trace has 12 identical initial columns + 4 computed columns. NLMs process
   mostly padding. Synchronization is meaningless. The CTM is functionally a deeper feedforward
   network, not doing iterative reasoning.
2. **No temporal reasoning in critic.** R-MADDPG (Wang et al., 2020) showed that a recurrent
   critic is what matters for partial observability — recurrent actor alone doesn't help.
   Our critic is memoryless.
3. **Agents don't cover full shape.** Voronoi uniformity plateaus at 0.51-0.65 across architectures.
   Agents cluster in corners (visible in GIFs, especially MLP at K=3).

---

## Next Direction — Stateful CTM Actor + Recurrent Critic

### The plan (AGREED)

**Path A — Stateful CTM actor:** Switch to `iterations=1` per env step. Maintain hidden states
across all 200 episode steps. The CTM builds temporal dynamics naturally — after 16 steps the
memory window is full and synchronization reflects genuine temporal patterns. This matches the
CTM paper's own RL setup (Appendix G.6: 1-2 ticks per step, stateful across episode).

**Recurrent critic — LSTM after aggregation:** Add an LSTM to the critic after the mean
aggregation step. Follows R-MADDPG's key finding that recurrent critic is critical for partial
observability. Preserves permutation equivariance (aggregation before recurrence).

```
Critic architecture (new):
  per-agent encoder: (obs_i, act_i) → 128 → 64    [shared weights, independent]
  mean aggregate → 64-dim team summary              [permutation equivariant]
  LSTM: processes 64-dim team summary over time     [temporal reasoning]
  head: LSTM hidden → Q-value
```

### Why both are needed together
- R-MADDPG showed recurrent actor alone doesn't help. Recurrent actor + recurrent critic does.
- Stateful CTM actor gives genuine temporal dynamics (the innovation), recurrent critic gives
  the training signal that rewards temporal reasoning.
- Paper story: "Physics-prior-seeded CTM actor with recurrent equivariant critic for partially
  observable cooperative MARL."

### Key implementation pieces
1. **Episode-sequence replay buffer** — replace random transition sampling with contiguous
   episode chunks. Both actor and critic need temporal context for updates.
2. **Burn-in** (R2D2-style) — replay a prefix of ~16-32 steps without gradient to reconstruct
   hidden states, then compute gradients on the remaining steps.
3. **LSTM in AggregatingCritic** — after mean aggregation, before head MLP.
4. **CTM actor: iterations=1, stateful** — carry hidden states across episode steps.
   Prior seeding initialises the first step; CTM dynamics carry forward.
5. **Hidden state storage** — store actor + critic hidden states at sequence boundaries
   (sparse, not every transition).

### What changes in which files
- `buffer_agent.py` — episode-sequence storage and sampling (biggest change)
- `networks.py` — LSTM addition to AggregatingCritic
- `ctm_actor.py` — iterations=1 (config change only)
- `ctm_rl.py` — no changes needed
- `maddpg.py` — sequence-based update logic, burn-in, critic hidden state management
- `train_assembly_jax_gpu.py` — stateful hidden state carry-forward (restore from Phase 1,
  but this time update is also stateful so no mismatch)
- `agents.py` / `ctm_agent.py` — critic hidden state init/carry
- `cfg/assembly_cfg.py` — new params (burn_in_length, sequence_length, lstm_hidden_dim)

### Ablation table (for paper)

| Actor | Critic | Expected |
|---|---|---|
| MLP + prior reg | AggregatingCritic (memoryless) | Current baseline (coverage 0.916) |
| MLP + prior reg | AggregatingCritic + LSTM | Improved — critic temporal reasoning |
| CTM stateful + prior seed | AggregatingCritic (memoryless) | Unclear — R-MADDPG says actor alone doesn't help |
| CTM stateful + prior seed | AggregatingCritic + LSTM | Best — both components working |

---

## Environment and Observability

Each agent observes: own state (4) + K nearest neighbors (4×K) + M nearest target cells (2×M).
Default K=6, M=80. At `d_sen=0.3`, each agent sees ~15% of shape cells. Shapes span ~2.0×2.2
units with ~500 cells. The task is genuinely partially observable at the actor level.

The centralised critic sees ALL agents' joint obs+actions during training (true CTDE). With a
recurrent critic, it can track how the team state evolves over time.

---

## Reward & Physics Status

### Implemented (in assembly.py)
- k_ball=2000 + 4 substeps (prevents tunneling)
- is_uniform saturated case → False
- Current reward: proximity + crowding + territory balance (3 continuous components)

### Still TODO from original redesign
- Hardcode r_avoid=0.10 as constructor arg, remove formula
- Rename `is_collision` → `too_close`, fix threshold to `dist < 2 * r_avoid`
- Coverage radius: `r_avoid/2` → `r_avoid` everywhere
- Physical contact penalty: -0.07 per `is_touching` neighbor
- Full r_avoid audit (see `Docs/REWARD_PHYSICS_REDESIGN.md`)

These are independent of the CTM/critic work and can be done in parallel.

---

## Key Constraints

- Do NOT modify DDPGAgent actor logic — subclass only; MLP version must remain intact
- CTM import path: `from models.ctm_rl import ContinuousThoughtMachineRL`
  (ctm_actor.py adds `continuous-thought-machines/` to sys.path at import time)
- `backbone_type='classic-control-backbone'` for flat vector observations
- `use_ctm_actor` and `n_agents` stored in `init_dict` and saved/loaded with checkpoints
- Old checkpoints (pre-AggregatingCritic) will fail to load; retrain from scratch
- `maddpg.step()` returns a 3-tuple everywhere — all call sites updated
- Buffer stores joint rows. `push()` takes no `index` argument.
- Training runs on Colab (T4/A100 GPUs)

### Intellectual honesty rule

Do not get carried away with ideas. An idea is not a result. Before any implementation:
- State what the experiment would show if the hypothesis is **wrong**
- If the experiment cannot falsify the hypothesis, it is not a useful experiment
- Do not frame incremental or null results as confirmation
