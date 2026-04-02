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
       hidden_dim=64, device='cpu', discrete_action=False)
```

**Key Parameters**:
- `agent_init_params`: List of dicts with `num_in_pol`, `num_out_pol`, `num_in_critic` for each agent type
- `alg_types`: List of agent types (e.g., `["agent"]`)
- `epsilon`: Probability of random action (exploration)
- `noise`: Scale of Gaussian noise for exploration
- `gamma`: Discount factor for rewards
- `tau`: Soft update rate for target networks (θ_target ← τ*θ + (1-τ)*θ_target)
- `hidden_dim`: Hidden layer size for actor/critic networks

**Factory Method** (Recommended):
```python
maddpg = MADDPG.init_from_env(env, agent_alg='MADDPG', tau=0.01, 
                              lr_actor=1e-4, lr_critic=1e-3, 
                              hidden_dim=180, device='gpu', 
                              epsilon=0.1, noise=0.9)
```
- Automatically extracts observation/action dimensions from `env`
- Simplifies initialization

### Key Methods

#### `step(observations, start_stop_num, explore=False)`
**Purpose**: Get actions from all agents for the current observations.

**Inputs**:
- `observations`: Tensor of shape `(obs_dim, n_agents)` — note: features in rows!
- `start_stop_num`: List of slices, e.g., `[slice(0, 30)]` to select agent indices
- `explore`: If True, adds exploration noise; if False, uses deterministic policy

**Outputs**:
- `actions`: List of tensors, each shape `(n_agents, action_dim)` — **already transposed by DDPGAgent**
- `log_pis`: List of log probabilities (used for some regularization schemes)

**Example**:
```python
obs = torch.Tensor(np_obs)  # shape: (192, 30) for obs_dim=192, n_a=30
actions, log_pis = maddpg.step(obs, [slice(0, 30)], explore=True)
# actions[0] shape: (30, 2) for 30 agents with 2D continuous actions
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

**Inputs**:
- `obs`, `next_obs`: Tensors of shape `(batch_size, obs_dim)`
- `acs`: Actions taken, shape `(batch_size, action_dim)`
- `rews`: Rewards, shape `(batch_size, 1)`
- `dones`: Episode termination flags, shape `(batch_size, 1)`
- `agent_i`: Index of agent to update (0 for homogeneous team)
- `acs_prior`: Prior/expert actions for regularization, shape `(batch_size, action_dim)`
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

## 3. ReplayBufferAgent Class

**Location**: `algorithm/utils/buffer_agent.py`

### Purpose
Stores experience tuples for off-policy learning. Supports parallel environments.

### Initialization

```python
ReplayBufferAgent(max_steps, num_agents, start_stop_index, 
                  state_dim, action_dim)
```

**Key Parameters**:
- `max_steps`: Buffer capacity in timesteps
- `num_agents`: Total agents (n_rollout_threads × n_a)
- `start_stop_index`: Slice for agent selection (e.g., `slice(0, 30)`)
- `state_dim`: Observation dimension
- `action_dim`: Action dimension

**Memory Allocation**:
```python
total_length = max_steps * num_agents
obs_buffs:       (total_length, state_dim)
ac_buffs:        (total_length, action_dim)
ac_prior_buffs:  (total_length, action_dim)
rew_buffs:       (total_length, 1)
next_obs_buffs:  (total_length, state_dim)
done_buffs:      (total_length, 1)
```

### Key Methods

#### `push(observations, actions, rewards, next_observations, dones, index, actions_prior=None, log_pi=None)`
**Purpose**: Add experience to buffer.

**Inputs**:
- `observations`: Shape `(obs_dim, n_agents)` — features in rows
- `actions`: Shape `(action_dim, n_agents)` — will be transposed to `(n_agents, action_dim)` internally
- `rewards`: Shape `(1, n_agents)`
- `next_observations`: Shape `(obs_dim, n_agents)`
- `dones`: Shape `(1, n_agents)`
- `index`: Slice to select agents (e.g., `slice(0, 30)`)
- `actions_prior`: Prior actions for regularization

**Process**:
1. Transpose to `(n_agents, feature_dim)` via `[:, index].T`
2. Flatten to `(n_agents, feature_dim)` and write sequentially
3. Circular buffer: wraps around when full

**Example**:
```python
buffer.push(obs, actions, rewards, next_obs, dones, 
            slice(0, 30), actions_prior)
# obs: (192, 30), actions: (2, 30), etc.
# Internally stored as 30 separate experiences
```

#### `sample(N, to_gpu=False, norm_rews=True)`
**Purpose**: Sample random batch for training.

**Inputs**:
- `N`: Batch size (e.g., 512)
- `to_gpu`: If True, convert to CUDA tensors
- `norm_rews`: If True, normalize rewards to mean=0, std=1

**Outputs**:
- `obs`: (N, obs_dim)
- `acs`: (N, action_dim)
- `rews`: (N, 1) — normalized if `norm_rews=True`
- `next_obs`: (N, obs_dim)
- `dones`: (N, 1)
- `acs_prior`: (N, action_dim) or None

**Example**:
```python
batch = buffer.sample(512, to_gpu=True)
obs, acs, rews, next_obs, dones, acs_prior = batch
```

#### `get_average_rewards(N)`
Compute mean reward over last N experiences (for logging).

---

## 4. MLPNetwork Class

**Location**: `algorithm/utils/networks.py`

### Purpose
Multi-layer perceptron for policy (actor) or value (critic) function approximation.

### Architecture

```python
MLPNetwork(input_dim, out_dim, hidden_dim=64, 
           nonlin=F.leaky_relu, constrain_out=False, 
           discrete_action=False)
```

**Layers**:
```
Input (input_dim)
    ↓
FC1 → LeakyReLU → hidden_dim
    ↓
FC2 → LeakyReLU → hidden_dim
    ↓
FC3 → LeakyReLU → hidden_dim
    ↓
FC4 → out_dim
    ↓
[Tanh if constrain_out=True] → Output
```

**Usage**:
- **Policy**: `constrain_out=True` → Tanh output in [-1, 1]
- **Critic**: `constrain_out=False` → Linear output (unbounded Q-values)

### Forward Pass

```python
output = network(input_tensor)
```

**Input**: `(batch_size, input_dim)`
**Output**: `(batch_size, out_dim)`

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
# 1. Create MADDPG
maddpg = MADDPG.init_from_env(env, hidden_dim=180, lr_actor=1e-4, 
                              lr_critic=1e-3, device='gpu')

# 2. Create buffer
buffer = ReplayBufferAgent(buffer_length, env.num_agents, 
                           state_dim=env.observation_space.shape[0],
                           action_dim=env.action_space.shape[0],
                           start_stop_index=slice(0, env.num_agents))

# 3. Rollout
maddpg.prep_rollouts(device='cpu')
actions, _ = maddpg.step(torch.Tensor(obs), [slice(0, env.n_a)], explore=True)
next_obs, rewards, dones, _, prior = env.step(actions_np)
buffer.push(obs, actions_np, rewards, next_obs, dones, slice(0, env.n_a), prior)

# 4. Train
maddpg.prep_training(device='gpu')
batch = buffer.sample(512, to_gpu=True)
vf_loss, pol_loss, reg_loss = maddpg.update(*batch, agent_i=0, alpha=env.alpha)
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
