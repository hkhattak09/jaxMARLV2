# MARL-LLM Core Components

**Purpose**: Detailed documentation of the core algorithm components (MADDPG, agents, buffers, networks).

## Overview

The core components implement the MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm with experience replay and optional regularization towards prior actions.

**Key Files**:
- `algorithm/algorithms/maddpg.py` — MADDPG orchestrator
- `algorithm/utils/agents.py` — DDPGAgent (individual agent)
- `algorithm/utils/buffer_agent.py` — ReplayBufferAgent
- `algorithm/utils/networks.py` — MLPNetwork (actor/critic)
- `algorithm/utils/noise.py` — GaussianNoise
- `algorithm/utils/misc.py` — Utility functions (soft_update, etc.)

---

## 1. MADDPG Class

**Location**: `algorithm/algorithms/maddpg.py`

### Purpose
Coordinates multiple DDPG agents for multi-agent learning. In this codebase, typically manages **one homogeneous team** of agents.

### Initialization

```python
MADDPG(agent_init_params, alg_types, epsilon, noise,
       gamma=0.95, tau=0.01, lr_actor=1e-4, lr_critic=1e-3,
       hidden_dim=64, n_agents=None,
       device='cpu', discrete_action=False,
       use_ctm_actor=False, ctm_config=None)
```

**Key Parameters**:
- `agent_init_params`: List of dicts with `dim_input_policy`, `dim_output_policy`, `n_agents` for each agent type
- `alg_types`: List of agent types (e.g., `["agent"]`)
- `n_agents`: Total physical agents (e.g., 24). Used by `AggregatingCritic` and for joint reshape in `update`/`target_policies`.
- `epsilon`: Probability of random action (exploration)
- `noise`: Scale of Gaussian noise for exploration
- `gamma`: Discount factor for rewards
- `tau`: Soft update rate for target networks (θ_target ← τ*θ + (1-τ)*θ_target)
- `hidden_dim`: Hidden layer size for MLP actor networks (default 180). CTM actor ignores this (uses `ctm_config`).
- `use_ctm_actor`: If True, build `CTMDDPGAgent` with `CTMActor` instead of `DDPGAgent` with `MLPNetwork`. **Default True** (pass `--use_mlp_actor` CLI flag to revert).
- `ctm_config`: Dict of CTM hyperparameters (d_model, memory_length, n_synch_out, iterations, synapse_depth, etc.).

**Note on `critic_hidden_dim`**: Previously a constructor parameter for `MLPNetwork`-based critics. **Removed** — the centralised critic is now `AggregatingCritic` which has a fixed internal structure (shared encoder 194→128→64, mean aggregation, head 64→128→1). Its internal dims are not configurable via `MADDPG`.

**Factory Method** (Recommended):
```python
maddpg = MADDPG.init_from_env(env, agent_alg='MADDPG', tau=0.01,
                              lr_actor=1e-4, lr_critic=1e-3,
                              hidden_dim=180,
                              device='gpu', epsilon=0.1, noise=0.9,
                              use_ctm_actor=True, ctm_config={...})
```
- Automatically extracts observation/action dimensions and `n_agents` from `env`
- Passes `n_agents` to `AggregatingCritic` for permutation-equivariant joint Q-value

### Key Methods

#### `step(observations, start_stop_num, explore=False, hidden_states=None)`
**Purpose**: Get actions from all agents for the current observations.

**Inputs**:
- `observations`: Tensor of shape `(obs_dim, n_agents)` — note: features in rows!
- `start_stop_num`: List of slices, e.g., `[slice(0, 30)]` to select agent indices
- `explore`: If True, adds exploration noise; if False, uses deterministic policy
- `hidden_states`: CTM hidden state tuple `(state_trace, activated_trace)` or `None` for MLP. Under stateless rollout, pass fresh hidden state from `get_initial_hidden_state()` every step.

**Outputs** (3-tuple — always, even for MLP):
- `actions`: List of tensors, each shape `(n_agents, action_dim)` — **already transposed**
- `log_pis`: List of log probabilities (None for CTM actor)
- `new_hidden_states`: Updated CTM hidden state tuple, or None for MLP

**Previous return (changed)**: Was a 2-tuple `(actions, log_pis)`. Now always 3-tuple. All call sites updated: training loop, `run_eval`, `run_final_eval`, `eval_shapes.py`.

**Example** (MLP):
```python
obs_gpu = env.reset()  # (192, 24) torch.cuda.FloatTensor
actions, _, _ = maddpg.step(obs_gpu, [slice(0, 24)], explore=True)
# actions[0] shape: (24, 2)
```

**Example** (CTM, stateless):
```python
hidden = maddpg.agents[0].policy.get_initial_hidden_state(n_a, device)
actions, _, _ = maddpg.step(obs_gpu, [slice(0, n_a)], explore=True, hidden_states=hidden)
# hidden states discarded — fresh ones generated next step
```

#### `update(obs, acs, rews, next_obs, dones, agent_i, acs_prior=None, alpha=0.5, parallel=False, logger=None)`
**Purpose**: Update actor and critic networks for a specific agent using sampled batch from replay buffer.

**Process**:
1. **Critic Update** (TD learning):
   - Compute target Q-value: `Q_target = r + γ * Q_target(s', a')`
   - Compute current Q-value: `Q(s, a)`
   - Minimize MSE loss: `L_critic = MSE(Q, Q_target)`

2. **Actor Update** (policy gradient):
   - Compute policy loss: `L_actor = -mean(Q(s, π(s)))`  (maximize Q)
   - If `acs_prior` provided, add regularization: `L_actor += 0.3 * alpha * MSE(π(s), a_prior)`
   - Update policy to maximize Q-values while staying close to prior

**Inputs** (centralised critic — joint tensors from buffer):
- `obs`, `next_obs`: Tensors of shape `(batch_size, n_agents * obs_dim)` — joint observations
- `acs`: Actions taken, shape `(batch_size, n_agents * action_dim)` — joint actions
- `rews`: Mean reward, shape `(batch_size, 1)`
- `dones`: Max done flag, shape `(batch_size, 1)`
- `agent_i`: Index of agent to update (0 for homogeneous team)
- `acs_prior`: Prior/expert actions for regularization, shape `(batch_size, n_agents * action_dim)`
- `alpha`: Regularization coefficient (typically read from `env.alpha`)

**Outputs**:
- `vf_loss`: Critic loss (float)
- `pol_loss`: Actor loss (float)
- `reg_loss`: Regularization loss (float)

**Key Insight**: The regularization term `MSE(policy_action, prior_action)` helps bootstrap learning by encouraging agents to follow Reynolds flocking behavior initially, then gradually reduce `alpha` to allow learned behavior.

#### `update_all_targets()`
**Purpose**: Soft-update target networks for all agents.

**Process**:
```python
θ_target ← τ * θ + (1 - τ) * θ_target
```
- Called once per training iteration after all agent updates
- Stabilizes learning by slowly tracking main networks

#### `prep_training(device='gpu')`
**Purpose**: Set networks to training mode and move to GPU.

**Effects**:
- Enables dropout, batch norm training behavior
- Moves policy, critic, target_policy, target_critic to specified device

#### `prep_rollouts(device='cpu')`
**Purpose**: Set networks to evaluation mode and move to CPU for environment interaction.

**Effects**:
- Disables dropout, uses running stats for batch norm
- Policy networks moved to CPU (only policy needed for action selection)

#### `scale_noise(scale, new_epsilon)` & `reset_noise()`
**Purpose**: Adjust exploration noise level.

**Typical Usage**:
```python
maddpg.scale_noise(0.9, 0.1)  # Set noise scale and epsilon
maddpg.reset_noise()           # Reset noise generator state
```

### Device Management
The class tracks device location for each network type separately:
- `pol_dev`, `critic_dev`, `trgt_pol_dev`, `trgt_critic_dev`
- Only moves networks when device changes (optimization)

---

## 2. DDPGAgent Class

**Location**: `algorithm/utils/agents.py`

### Purpose
Individual DDPG agent with actor-critic architecture. Each agent has:
- **Policy network** (actor): obs → action
- **Target policy network**: slowly-updated copy for stable targets
- **Critic network**: (obs, action) → Q-value
- **Target critic network**: slowly-updated copy

### Initialization

```python
DDPGAgent(dim_input_policy, dim_output_policy, dim_input_critic,
          lr_actor, lr_critic, hidden_dim=64, 
          discrete_action=False, device='cpu', 
          epsilon=0.1, noise=0.1)
```

**Key Parameters**:
- `dim_input_policy`: Observation dimension (e.g., 192)
- `dim_output_policy`: Action dimension (e.g., 2)
- `dim_input_critic`: Obs + action dimension (e.g., 192 + 2 = 194 for single agent; larger for centralized critic)
- `epsilon`: Probability of uniform random action
- `noise`: Std deviation of Gaussian exploration noise

### Network Architecture
Both policy and critic use `MLPNetwork`:
- 4 fully connected layers: `input → hidden → hidden → hidden → output`
- Activation: LeakyReLU
- Policy output: Tanh (continuous actions in [-1, 1])
- Critic output: Linear (Q-values unbounded)

### Key Methods

#### `step(obs, explore=False)`
**Purpose**: Select action for given observation.

**Process**:
1. Forward pass through policy network: `a = π(obs)`
2. If `explore=True`:
   - With probability `epsilon`: sample uniform random action in [-1, 1]
   - Otherwise: add Gaussian noise `N(0, noise²)` and clip to [-1, 1]
3. If `explore=False`: return deterministic action

**Inputs**:
- `obs`: Tensor of shape `(batch_size, obs_dim)` or `(n_agents, obs_dim)`

**Outputs**:
- `action`: Tensor of shape `(action_dim, batch_size)` — **note transpose!**
- `log_pi`: Log probability of action (for regularization)

**Example**:
```python
obs = torch.Tensor(np_obs[:, :30]).T  # (30, 192) - 30 agents
action, log_pi = agent.step(obs, explore=True)
# action shape: (2, 30) - transposed for maddpg.step output format
```

#### `get_params()` & `load_params(params)`
Get/set network parameters for checkpointing.

---

## 2b. CTM Actor (Default)

**CTM is the default actor.** Pass `--use_mlp_actor` to revert to the MLP actor.

**Files**:
- `algorithm/utils/ctm_actor.py` — `CTMActor(nn.Module)` wrapping `ContinuousThoughtMachineRL`
- `algorithm/utils/ctm_agent.py` — `CTMDDPGAgent(DDPGAgent)` subclass

### Why CTM

The MLP actor makes a single feedforward pass: obs → action. The CTM runs `iterations=4` inner loop passes per observation, refining its action iteratively within a single timestep. Under partial observability (limited shape visibility), the CTM can start from a physics prior and iterate toward a learned action — a structural advantage over single-pass MLP.

### Stateless Rollout (critical design decision)

**Problem with stateful rollout**: Propagating hidden states across timesteps during rollout but reinitialising during actor updates caused a rollout/update mismatch. The critic was trained on actions from warm context-rich hidden states; policy gradient computed actions from a blank board → Q-overestimation, policy loss diverged.

**Storing hidden states in buffer** (R-MADDPG approach): infeasible. CTM hidden state = 2 × (256 × 16) = 8,192 floats per agent per transition. At buffer_length=20k, 24 agents: ~15.7 GB.

**Fix — stateless rollout**: Reset hidden states fresh at every timestep, identical to how actor updates work. Zero mismatch. CTM value-add is the iterative computation **within** a single timestep (4 inner passes), not cross-timestep memory.

```python
# Stateless rollout: every step
hidden = maddpg.agents[0].policy.get_initial_hidden_state(n_a, device)
actions, _, _ = maddpg.step(obs_gpu, start_stop_num, explore=True, hidden_states=hidden)
hidden_states = None  # discarded — fresh ones next step
```

### CTMActor Architecture

```python
CTMActor(
    obs_dim: int,         # 192
    action_dim: int,      # 2
    d_model: int = 256,   # Neuron population size
    memory_length: int = 16,
    n_synch_out: int = 16,   # Output size = 16×17/2 = 136-dim
    iterations: int = 4,
    synapse_depth: int = 1,
    ...
)
# Internals: ContinuousThoughtMachineRL + nn.Linear(136, 2) action head + Tanh
# backbone_type='classic-control-backbone', heads=0
```

`get_initial_hidden_state(batch_size, device)` → `(zeros, start_activated_trace.expand(batch_size))`
— gradients flow through `start_activated_trace` (an `nn.Parameter`) during stateless actor updates.

### CTMDDPGAgent

Subclass of `DDPGAgent`. Does **not** call `DDPGAgent.__init__` (would create unwanted MLP policy). Does a dummy forward pass on both `policy` and `target_policy` before `hard_update` to materialise `nn.LazyLinear` layers — critical, otherwise both networks would materialise with different random weights.

---

## 3. ReplayBufferAgent Class

**Location**: `algorithm/utils/buffer_agent.py`

### Purpose
Stores joint experience rows for centralised-critic MADDPG. One row per timestep
concatenates all agents' data. The centralised critic can consume sampled batches
directly without any reshaping.

### Initialization

```python
ReplayBufferAgent(max_steps, num_agents, state_dim, action_dim,
                  start_stop_index=None)  # start_stop_index ignored, kept for API compat
```

**Key Parameters**:
- `max_steps`: Buffer capacity in timesteps (= total rows stored)
- `num_agents`: Total physical agents (e.g., 24) — determines joint row width
- `state_dim`: Per-agent observation dimension
- `action_dim`: Per-agent action dimension
- `start_stop_index`: Ignored (legacy parameter, kept for backward compatibility)

**Memory Allocation** (joint rows):
```python
total_length = max_steps  # one row per timestep (not per-agent)
obs_buffs:       (max_steps, num_agents * state_dim)   # joint obs
ac_buffs:        (max_steps, num_agents * action_dim)  # joint actions
ac_prior_buffs:  (max_steps, num_agents * action_dim)
rew_buffs:       (max_steps, 1)   # mean reward across agents
next_obs_buffs:  (max_steps, num_agents * state_dim)
done_buffs:      (max_steps, 1)   # max done flag across agents
```

### Key Methods

#### `push(observations, actions, rewards, next_observations, dones, index=None, actions_prior_orig=None, log_pi_orig=None)`
**Purpose**: Add one joint-row transition to buffer.

**Inputs** (column-major GPU→CPU tensors from rollout):
- `observations`: Shape `(obs_dim, N*n_agents)` — all agents column-major
- `actions`: Shape `(action_dim, N*n_agents)`
- `rewards`: Shape `(1, N*n_agents)`
- `next_observations`: Shape `(obs_dim, N*n_agents)`
- `dones`: Shape `(1, N*n_agents)`
- `index`: Ignored (removed from call site)
- `actions_prior_orig`: Shape `(action_dim, N*n_agents)` optional

**Process**:
1. Flatten: `obs.T.reshape(N, n_a*obs_dim)` → N joint rows
2. Mean rewards: `rewards.reshape(N, n_a).mean(axis=1)` → `(N, 1)`
3. Max done: `dones.reshape(N, n_a).max(axis=1)` → `(N, 1)`
4. Write N rows to circular buffer (N=1 in standard setup)

**Example**:
```python
# No index argument — pass all agents, buffer handles concatenation
buffer.push(obs, actions, rewards, next_obs, dones,
            actions_prior_orig=prior)
# obs: (192, 24), actions: (2, 24) → stored as 1 row of width 4608/48
```

#### `sample(N, to_gpu=False, is_prior=False, is_log_pi=False)`
**Purpose**: Sample random batch for training.

**Inputs**:
- `N`: Batch size (e.g., 512)
- `to_gpu`: If True, convert to CUDA tensors
- `is_prior`: If True, include prior actions

**Outputs** (joint shapes, K=6 M=80 example):
- `obs`: `(N, n_agents*obs_dim)` = `(512, 4608)`
- `acs`: `(N, n_agents*action_dim)` = `(512, 48)`
- `rews`: `(N, 1)`
- `next_obs`: `(N, n_agents*obs_dim)` = `(512, 4608)`
- `dones`: `(N, 1)`
- `acs_prior`: `(N, n_agents*action_dim)` = `(512, 48)` or None
- `log_pis`: always None

**Example**:
```python
obs, acs, rews, next_obs, dones, acs_prior, _ = buffer.sample(512, to_gpu=True, is_prior=True)
# All joint tensors — ready for centralised critic directly
```

#### `get_average_rewards(N)`
Compute mean reward over last N experiences (for logging).

**UPCOMING — Episode-sequence buffer:**
The buffer will be restructured to store complete episodes and sample contiguous sequences
(instead of random individual transitions). This is required for the stateful CTM actor and
recurrent LSTM critic — both need temporal context during training updates. R2D2-style
burn-in will replay a prefix without gradient to reconstruct hidden states.
See `Docs/CTM_ACTOR_DESIGN.md` Phase 5 for full details.

---

## 4. Network Classes

**Location**: `algorithm/utils/networks.py`

### MLPNetwork

Multi-layer perceptron used for the **actor** (MLP actor path only).

```python
MLPNetwork(input_dim, out_dim, hidden_dim=64,
           nonlin=F.leaky_relu, constrain_out=False,
           discrete_action=False)
```

**Layers**: `input → hidden → hidden → hidden → output [→ Tanh if constrain_out]`

**Usage**:
- **MLP Policy**: `constrain_out=True` → Tanh output in [-1, 1]
- Was previously also used for critic — **replaced** by `AggregatingCritic`

### AggregatingCritic *(centralised critic — replaces flat MLPNetwork critic)*

**Previous critic**: `MLPNetwork` over flat 4,656-dim joint input (24 agents × 194). This caused catastrophic regression — an 18× compression in the first linear layer prevented the critic from learning a useful Q-function. The flat MLP has no way to exploit permutation equivariance of the Q-function over homogeneous agents.

**Current critic** (permutation-equivariant by construction):
```python
AggregatingCritic(n_agents, obs_dim, act_dim)
# Internal structure:
#   Shared encoder: (obs_dim + act_dim) → 128 → 64  (applied independently per agent)
#   Mean aggregation: 24 × 64 → 64-dim team summary
#   Head MLP: 64 → 128 → 1

# Forward pass — same call site as before:
q = critic(torch.cat([obs_all, act_all], dim=1))  # (batch, n_agents*(obs_dim+2)) → (batch, 1)
```

- **Permutation equivariant**: Q(s,a) unchanged if agents reordered
- **24× fewer parameters** than a flat MLP needed to cover the same input space
- Both MLP and CTM actor use this same critic
- `forward(X)` accepts concatenated joint obs+actions — no changes to `maddpg.py` call sites

**UPCOMING — Recurrent critic (LSTM after aggregation):**
An LSTM will be added after the mean aggregation step, before the head MLP. This follows
the R-MADDPG finding that a recurrent critic is critical for partial observability.
The LSTM processes the 64-dim team summary over time, tracking how the team state evolves.
Permutation equivariance is preserved because aggregation happens before recurrence.
See `Docs/CTM_ACTOR_DESIGN.md` Phase 5 for full details.

---

## 5. Exploration Noise

**Location**: `algorithm/utils/noise.py`

### GaussianNoise

Simple Gaussian noise for continuous action exploration:

```python
GaussianNoise(action_dim, scale=0.1)
```

**Methods**:
- `noise(batch_size)`: Sample `(batch_size, action_dim)` noise ~ N(0, scale²)
- `log_prob(noise_sample)`: Compute log probability (for regularization)
- `reset()`: No-op (stateless)

**Usage in DDPGAgent.step**:
```python
action_noise = Tensor(self.exploration.noise(batch_size))
action += action_noise
action = action.clamp(-1, 1)
```

---

## 6. Utility Functions

**Location**: `algorithm/utils/misc.py`

### `soft_update(target, source, tau)`
**Purpose**: Polyak averaging for target network updates.

```python
for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

### `hard_update(target, source)`
**Purpose**: Copy all parameters from source to target (initialization).

### `onehot_from_logits(logits, eps=0.0)`
Convert logits to one-hot (discrete actions).

### `gumbel_softmax(logits, temperature=1.0, hard=False)`
Differentiable sampling for discrete actions.

---

## Component Interactions

### Training Update Flow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Sample batch from ReplayBufferAgent                       │
│    obs, acs, rews, next_obs, dones, acs_prior = buffer.sample│
└────────────────────┬─────────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. MADDPG.update(batch, agent_i, acs_prior, alpha)           │
│    ├─ Critic Update:                                         │
│    │  - Target: r + γ * Q_target(s', π_target(s'))           │
│    │  - Loss: MSE(Q(s,a), target)                            │
│    │  - Backprop through critic                              │
│    └─ Actor Update:                                          │
│       - Loss: -mean(Q(s, π(s))) + α * MSE(π(s), a_prior)     │
│       - Backprop through policy                              │
└────────────────────┬─────────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. MADDPG.update_all_targets()                               │
│    - Soft update: θ_target ← τ*θ + (1-τ)*θ_target            │
└──────────────────────────────────────────────────────────────┘
```

### Rollout Flow

```
MADDPG.prep_rollouts(device='cpu')  # Set eval mode, move to CPU
    ↓
MADDPG.step(obs, explore=True)
    ↓
DDPGAgent.step(obs, explore=True)
    ├─ action = policy(obs)
    ├─ if explore: action += noise OR random
    └─ return action.T  (transpose!)
    ↓
actions sent to environment
    ↓
ReplayBufferAgent.push(obs, actions, rewards, ...)
```

---

## Common Patterns

### Pattern 1: Initialize and Train

```python
# 1. Create MADDPG (centralised critic: critic_hidden_dim=256)
maddpg = MADDPG.init_from_env(env, hidden_dim=180, critic_hidden_dim=256,
                              lr_actor=1e-4, lr_critic=1e-3, device='gpu')
# dim_input_critic = n_agents * (obs_dim + 2) set automatically

# 2. Create buffer (joint rows — no start_stop_index)
buffer = ReplayBufferAgent(buffer_length, env.num_agents,
                           state_dim=env.observation_space.shape[0],
                           action_dim=env.action_space.shape[0])

# 3. Rollout
maddpg.prep_rollouts(device='gpu')
actions, _, _ = maddpg.step(obs_gpu, [slice(0, env.n_a)], explore=True)
next_obs, rewards, dones, _, prior = env.step(actions_gpu.t().detach())
# No index argument — push full joint data
buffer.push(obs_np, actions_np, rewards_np, next_obs_np, dones_np,
            actions_prior_orig=prior_np)

# 4. Train
maddpg.prep_training(device='gpu')
obs, acs, rews, next_obs, dones, acs_prior, _ = buffer.sample(512, to_gpu=True, is_prior=True)
vf_loss, pol_loss, reg_loss = maddpg.update(obs, acs, rews, next_obs, dones,
                                            agent_i=0, acs_prior=acs_prior, alpha=env.alpha)
maddpg.update_all_targets()
```

### Pattern 2: Save/Load Checkpoints

```python
# Save
torch.save({
    'agent_params': [a.get_params() for a in maddpg.agents],
    'episode': ep_i,
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
for a, params in zip(maddpg.agents, checkpoint['agent_params']):
    a.load_params(params)
```

---

## Debugging Tips

1. **Network not updating**: Check `prep_training()` was called and networks are in training mode
2. **NaN losses**: Check learning rates, try gradient clipping, check reward normalization
3. **Slow learning**: Increase `hidden_dim`, check `alpha` regularization is decaying
4. **Memory issues**: Reduce `buffer_length` or `n_rollout_threads`
5. **Shape errors**: Remember `actions.T` transpose in DDPGAgent.step output

---

**Next**: See ENVIRONMENT_INTERFACE.md for environment details, DATA_FLOW.md for tensor shapes.
