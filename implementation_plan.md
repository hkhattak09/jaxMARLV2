# Final Implementation Plan: CTM Prior Seeding on SMAX via MAPPO

## Goal

Validate that **CTM actor with prior integration improves sample efficiency** over a standard GRU actor in multi-agent RL, using SMAX from JaxMARL with MAPPO. Full Flax/JAX pipeline for maximum training speed.

---

## Design Decisions (Finalized)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Framework** | **Flax/JAX** (port CTM) | End-to-end JIT → hours vs days for ablation runs |
| **Algorithm** | **MAPPO** | Standard for cooperative SMAX, centralized critic |
| **Actor** | **GRU first** → then **CTM** | Validate pipeline before experimental variable |
| **Critic** | **GRU** (separate, centralized) | Sees `world_state`, isolates CTM to actor |
| **Actions** | **Discrete** (Categorical) | SMAX uses discrete |
| **Environment** | **SMAX** via JaxMARL | Partial observability, recurrence necessary |
| **Starting map** | **`3m`** then `2s3z` | Simple first, then heterogeneous |

---

## Prior Integration Modes

The research contribution is CTM + prior integration. There are three distinct mechanisms, each building on the previous:

### 1. Pre-training (NEW — before RL)

```
Before any RL training:
  for batch in pre_training_data:
      obs = sample_observations_from_env()
      prior_action = prior(obs)                    # "attack nearest enemy"
      ctm_hidden = ctm.get_initial_hidden()
      predicted_logits = ctm_actor(obs, ctm_hidden) # forward through CTM
      loss = CrossEntropy(predicted_logits, prior_action)
      loss.backward()                               # updates CTM weights
```

**What it does**: Arranges the CTM's synapse weights, NLM weights, and synchronization parameters so that the CTM's *dynamics* already produce prior-like behavior. The latent space is structured before RL begins.

**Why it matters**: Unlike seeding (which only affects the initial hidden state), pre-training shapes the CTM's learned transformations. When RL then starts, the CTM already "knows" how to produce reasonable actions — it just needs to refine them for optimality.

### 2. Seeding (at episode start)

```
At each episode boundary (done=True):
  prior_action = prior(obs)
  ctm_hidden = seed_mlp(obs, prior_action)  # learned mapping to hidden state
  # Instead of default zeros, CTM starts with prior-informed hidden state
```

**What it does**: Initializes CTM hidden state from `(obs, prior_action)` via a learned MLP. Gives a context-dependent starting point each episode.

### 3. Regularization (during RL, decaying)

```
During RL training (first N updates):
  actor_loss = ppo_clipped_loss(...)
  prior_action = prior(obs)
  reg_loss = CrossEntropy(actor_logits, prior_action) * decay_weight
  total_loss = actor_loss + reg_loss
```

**What it does**: Auxiliary loss that pulls actor output toward prior actions. Decays over training so the agent can eventually diverge from the prior when it finds better strategies.

### How they combine

| Mode | Pre-training | Seeding | Regularization |
|------|-------------|---------|----------------|
| `none` | ❌ | ❌ | ❌ |
| `pretrain` | ✅ | ❌ | ❌ |
| `seed` | ❌ | ✅ | ❌ |
| `pretrain+seed` | ✅ | ✅ | ❌ |
| `pretrain+seed+reg` | ✅ | ✅ | ✅ |

---

## CTM Flax Port — What Gets Ported

### From `continuous-thought-machines/models/`:

| PyTorch Component | Flax Equivalent | Notes |
|---|---|---|
| `SuperLinear` (NLMs) | Custom Flax module with `self.param('w1', ...)` + `jnp.einsum` | Core is identical: `einsum('BDM,MHD->BDH', x, w1) + b1` |
| `ContinuousThoughtMachineRL.forward()` | Flax `__call__` with explicit state passing | For-loop over `self.iterations` works in JIT (static count) |
| `ContinuousThoughtMachineRL.synapses` (depth=1) | `nn.Dense → GLU → nn.LayerNorm → nn.Dense → GLU → nn.LayerNorm` | Replace `LazyLinear` with `nn.Dense` (input size = `d_input + d_model`, known at init) |
| `compute_synchronisation` | Pure JAX math | `jnp.triu_indices`, learned decay via `jnp.exp(-decay_params)` |
| FIFO trace management | `jnp.concatenate((trace[:,:,1:], new[...,None]), axis=-1)` | Identical logic |
| `ClassicControlBackbone` | `nn.Dense → GLU → nn.LayerNorm` chain | For vector obs (SMAX uses flat obs vectors) |
| `start_trace`, `start_activated_trace` | `self.param(...)` or `self.variable('state', ...)` | Learned initial states |
| Hidden state reset on done | `jnp.where(done, initial_state, current_state)` | Same as JaxMARL's `ScannedRNN` pattern |

### What does NOT get ported
- Vision backbones (ResNet, ShallowWide, MiniGrid) — not needed for SMAX
- Attention mechanism (`q_proj`, `kv_proj`, `nn.MultiheadAttention`) — RL variant doesn't use it
- `SynapseUNET` — start with depth=1 (simpler synapse), add later if needed
- Certainty computation — RL variant doesn't use it
- Positional embeddings — RL variant doesn't use them

### Port scope: ~200 lines of Flax code

The RL variant (`ctm_rl.py`) is only 192 lines and many of those are comments/setup. The actual forward pass is ~30 lines. The modules needed are:
- `SuperLinear` → ~30 lines Flax
- `ClassicControlBackbone` → ~15 lines Flax  
- `CTM_RL` (synapses + NLMs + synch + forward) → ~100 lines Flax
- `GLU` helper → ~5 lines
- Total: **~150-200 lines of new Flax code**

---

## File Structure

All new code goes in `smax_ctm/`:

```
smax_ctm/
├── __init__.py
├── ctm_flax.py          # CTM ported to Flax (SuperLinear, synapses, NLMs, synch)
├── networks.py          # GRUActor, CTMActor, GRUCritic (all Flax)
├── train_mappo.py       # MAPPO training loop (pure JAX)
├── smax_prior.py        # "Attack nearest enemy" prior computation
├── config.py            # Hydra or dataclass config
├── utils.py             # Helpers (batchify, unbatchify, etc.)
└── reference/           # Copied reference files from old implementation
    ├── ctm_actor.py     # Prior seeding mechanism (from MARL-LLM)
    └── assembly_cfg.py  # Prior mode config pattern (from MARL-LLM)
```

---

## Ablation Table

| Run | Actor | Prior Mode | What it tests |
|-----|-------|-----------|---------------|
| **1** | GRU | none | Baseline — standard MAPPO RNN |
| **2** | CTM | none | CTM vs GRU (architecture alone) |
| **3** | CTM | pretrain | Does pre-training arrange the latent space? |
| **4** | CTM | seed | Does hidden state seeding help? |
| **5** | CTM | pretrain+seed | Pre-training + seeding combined |
| **6** | CTM | pretrain+seed+reg | Full system |

Each run × 3 seeds × 2 maps (`3m`, `2s3z`) = **36 runs**.

With end-to-end JAX JIT, each run takes ~30 min → **36 runs ≈ 18 hours** (can parallelize across GPUs or vectorize seeds).

**Primary metric**: Return vs environment steps (sample efficiency).
**Success criterion**: Any CTM+prior variant reaches 90% of GRU's final performance in ≤50% of the steps.

---

## Phased Plan

All phases will be completed in a single session. Time estimations (Stages) roughly represent incremental logical units of work.

### Phase 1: Foundation (Stages 0-5)

#### Stage 0: Directory Reorganization

Move everything from the old implementation into `old_implementation/`, keeping only what we need at root level.

> [!NOTE]
> `JaxMARL/` has already been cloned. `JaxMARL_old/` is the stripped-down version from before. 
> The cloned `JaxMARL/` contains its own `.git` — we should remove it so it doesn't conflict with the root repo's git.

**Move to `old_implementation/`:**
- `MARL-LLM/` — old MADDPG training code (reference only)
- `JaxMARL_old/` — stripped-down JaxMARL (replaced by full clone)
- `rmaddpg/` — reference MADDPG implementation
- `before_recurrent_critic/` — old checkpoints/code
- `Docs/` — outdated docs for old implementation
- `CLAUDE.md`, `TRAINING_GUIDE.md`, `stateful_impl.md` — old implementation docs
- `ctm_see+reg_lstm.txt`, `ctm_seed_lstm.txt`, `ctm_seed_no_lstm.txt` — old training logs
- `fig/` — old figures

**Keep at root:**
- `continuous-thought-machines/` — reference CTM implementation (read-only, port from this)
- `JaxMARL/` — freshly cloned full JaxMARL library ✅ (already done)
- `requirements.txt` — update for new dependencies
- `.git`, `.gitignore` — version control

**Create at root:**
- `smax_ctm/` — all new code goes here

**Copy as reference into `smax_ctm/reference/`:**
- `MARL-LLM/marl_llm/algorithm/utils/ctm_actor.py` — prior seeding mechanism
- `MARL-LLM/marl_llm/cfg/assembly_cfg.py` — prior mode config pattern

**After reorganization, root should look like:**
```
new_marl_llm_implementation/
├── continuous-thought-machines/  # read-only reference for CTM port
├── JaxMARL/                     # full JaxMARL (SMAX env + MAPPO baselines)
├── smax_ctm/                    # OUR NEW CODE
├── old_implementation/          # everything old, archived
├── requirements.txt
├── .git/
└── .gitignore
```

#### Stage 1: Setup
- Import full JaxMARL, fix any JAX 0.7.2 compatibility issues
- Verify SMAX runs: `env.reset()`, `env.step()`, render a few episodes
- Run JaxMARL's existing `mappo_rnn_smax.py` baseline to get reference numbers

#### Stage 2: Networks + Training Loop
- `GRUActor` (Flax) — obs → Dense → GRU → Dense → logits (with avail_actions mask)
- `GRUCritic` (Flax) — world_state → Dense → GRU → Dense → value
- `train_mappo.py` — MAPPO loop based on JaxMARL's `mappo_rnn_smax.py` structure
  - Separate actor/critic `TrainState`
  - GAE advantage computation
  - PPO clipped loss (actor) + value loss (critic)

#### Stage 3: GRU Baseline Validation
- **Run 1**: GRU actor + GRU critic on `3m`
- Target: >80% win rate within 2M steps
- Compare curve to JaxMARL reference numbers from Stage 1
- ✅ **Gate**: Training loop is correct before proceeding

### Phase 2: CTM Integration (Stages 6-8)


#### Stages 4-5: CTM Flax Port
- Port `SuperLinear` (NLMs) → test with random inputs, verify output shapes
- Port `ClassicControlBackbone` → test with SMAX obs size
- Port `ContinuousThoughtMachineRL` → synapses, NLMs, synch, forward pass
- Unit test: compare Flax CTM output shapes against PyTorch CTM

#### Stage 6: CTM Actor
- `CTMActor` (Flax) — uses ported CTM, adds actor head + avail_actions masking
- Integrate into training loop (swap `actor_type` config flag)
- Verify: forward pass, hidden state shapes, done reset

#### Stage 7: CTM Baseline Run
- **Run 2**: CTM actor (no prior) + GRU critic on `3m`
- Compare to Run 1 — does CTM match/exceed/lag GRU?
- Debug any convergence issues

#### Stage 8: Prior Computation
- `smax_prior.py` — parse SMAX obs, compute "attack nearest enemy" action
- Verify: on random observations, prior output matches expected behavior
- Manual spot-check: visualize a few episodes, confirm prior actions make sense

### Phase 3: Prior Integration + Ablation (Stages 9-11)

#### Stage 9: Implement Prior Modes
- **Pre-training**: supervised loop before RL (sample obs → compute prior → backprop)
- **Seeding**: `seed_mlp` maps `(obs, prior_action)` → CTM hidden state
- **Regularization**: auxiliary cross-entropy loss with cosine decay schedule

#### Stage 10: Run Full Ablation
- Runs 3-6 on `3m` (3 seeds each)
- Monitor training curves, check for instabilities

#### Stage 11: Extended Maps + Analysis
- Run best configs on `2s3z` (heterogeneous agents)
- Plot sample efficiency curves
- Analyze results, write up findings

---

## Verification Plan

### Execution Environment Protocol
- **Local Device**: All code generation and static syntax checking will be done locally. (JAX is not installed locally and will not be installed).
- **Google Colab**: The user will manually run all actual execution, training, and testing workloads on Google Colab, and share the results/logs back. I will provide precise run instructions, code snippets, and expected output for Colab execution.

### Phase 1 Gates
1. SMAX env works with JAX 0.7.2 — reset, step, correct shapes
2. JaxMARL's own `mappo_rnn_smax.py` produces reasonable results (reference)
3. Our GRU baseline (Run 1) matches the reference within 10%

### Phase 2 Gates
1. CTM Flax port produces same-shaped outputs as PyTorch CTM
2. CTM actor integrates cleanly — training runs, losses are finite
3. CTM baseline (Run 2) either converges or fails clearly (not silently wrong)

### Phase 3 Gates
1. Prior computation outputs valid SMAX actions
2. Pre-training reduces actor loss on prior actions before RL starts
3. Sample efficiency curves show clear separation between prior modes

---

## Error Handling Philosophy: Fail Loud, Never Fake

Prefer a visible failure over a silent fallback.

- **Never silently swallow errors** to keep things "working." Surface the error. Don't substitute placeholder data.
- **Fallbacks are acceptable only when disclosed.** Show a banner, log a warning, annotate the output.
- **Design for debuggability**, not cosmetic stability.

Priority order:
1. Works correctly with real data
2. Falls back visibly — clearly signals degraded mode
3. Fails with a clear error message
4. ~~Silently degrades to look "fine"~~ — **never do this**

This applies everywhere: shape mismatches should crash not reshape silently, NaN losses should halt training not get clipped, prior computation failures should raise not return zeros.

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Flax CTM port has subtle bugs | Unit test against PyTorch CTM with same random inputs + weights |
| JAX 0.7.2 breaks JaxMARL | Fix API changes (usually `jax.tree_util.*` → `jax.tree.*`, deprecation warnings) |
| CTM doesn't JIT compile | Static `iterations` count ensures for-loop unrolls. If synapse depth > 1, may need `jax.lax.scan` |
| Pre-training overfits to prior | Limit pre-training steps (e.g., 1000-5000 updates). Monitor entropy — should stay high |
| SMAX prior is trivial | If "attack nearest" is too easy, switch to "attack weakest" or combine with movement heuristics |
