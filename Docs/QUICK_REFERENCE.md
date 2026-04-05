# MARL-LLM Quick Reference

**Purpose**: Fast lookup for function signatures, common operations, and debugging scenarios.

---

## Table of Contents
1. [Key Function Signatures](#1-key-function-signatures)
2. [Common Code Patterns](#2-common-code-patterns)
3. [Configuration Parameters](#3-configuration-parameters)
4. [Debugging Scenarios](#4-debugging-scenarios)
5. [Where to Add New Features](#5-where-to-add-new-features)
6. [File Locations](#6-file-locations)

---

## 1. Key Function Signatures

### Environment

#### AssemblyEnv (JAX)
```python
# Initialization
env = AssemblyEnv(
    results_file: str,           # Path to 'fig/results.pkl'
    n_a: int = 30,               # Number of agents
    topo_nei_max: int = 6,       # Neighbors in observation
    num_obs_grid_max: int = 80,  # Max target cells in obs
    dt: float = 0.1,             # Physics timestep
    vel_max: float = 0.8,        # Max velocity
    k_ball: float = 30.0,        # Agent repulsion
    k_wall: float = 100.0,       # Wall repulsion
    size_a: float = 0.035,       # Agent radius
    max_steps: int = 200,        # Episode length
)

# Reset
obs_dict, state = env.reset(key: PRNGKey)
# Returns: obs_dict: Dict[str, Array(obs_dim)], state: AssemblyState

# Step
obs_dict, new_state, rew_dict, done_dict, prior = env.step_env(
    key: PRNGKey,
    state: AssemblyState,
    actions: Dict[str, Array(2)]
)
# Returns: All dicts, prior: Array(n_a, 2)
```

#### JaxAssemblyAdapterGPUGPU (GPU-optimized)
```python
# Initialization
adapter = JaxAssemblyAdapterGPUGPU(
    jax_env: AssemblyEnv,
    n_envs: int = 1,      # Parallel environments
    seed: int = 0,
    alpha: float = 1.0,   # Regularization coefficient
)

# Reset
obs_gpu = adapter.reset()  # Returns: (obs_dim, n_envs*n_a) torch.cuda.FloatTensor

# Step (MUST .detach() actions for DLPack)
obs, rew, done, info, prior = adapter.step(actions: torch.cuda.FloatTensor)
# Input:  actions (n_envs*n_a, 2) GPU tensor, MUST be .detach()'ed
# Output: obs (obs_dim, n_envs*n_a), rew (1, n_envs*n_a), 
#         done (1, n_envs*n_a), prior (2, n_envs*n_a)
#         All torch.cuda.FloatTensor (except done is BoolTensor)

# Metrics
coverage = adapter.coverage_rate()           # Returns: float in [0, 1]
uniformity = adapter.distribution_uniformity()  # Returns: float in [0, 1]
```

### MADDPG Algorithm

```python
# Initialization (factory method)
maddpg = MADDPG.init_from_env(
    env: JaxAssemblyAdapterGPU,
    agent_alg: str = 'MADDPG',
    adversary_alg: str = None,
    tau: float = 0.01,              # Target network update rate
    lr_actor: float = 1e-4,
    lr_critic: float = 1e-3,
    hidden_dim: int = 180,          # Actor hidden dim
    critic_hidden_dim: int = 256,   # Centralised critic hidden dim (larger — joint input)
    device: str = 'gpu',
    epsilon: float = 0.1,           # Random action probability
    noise: float = 0.9,             # Gaussian noise scale
    name: str = 'assembly',
)
# Sets dim_input_critic = env.num_agents * (obs_dim + action_dim) automatically

# Action selection
actions, log_pis = maddpg.step(
    observations: Tensor,     # (obs_dim, n_agents)
    start_stop_num: List[slice],  # [slice(0, n_agents)]
    explore: bool = False,
)
# Returns: actions: List[Tensor(action_dim, n_agents)], log_pis: List[Tensor(1, n_agents)]

# Update networks (centralised critic — joint tensor inputs)
vf_loss, pol_loss, reg_loss = maddpg.update(
    obs: Tensor,              # (batch_size, n_agents*obs_dim)    — joint observations
    acs: Tensor,              # (batch_size, n_agents*action_dim) — joint actions
    rews: Tensor,             # (batch_size, 1)                   — mean reward
    next_obs: Tensor,         # (batch_size, n_agents*obs_dim)
    dones: Tensor,            # (batch_size, 1)                   — max done flag
    agent_i: int,             # Agent index (0 for homogeneous)
    acs_prior: Tensor = None, # (batch_size, n_agents*action_dim)
    alpha: float = 0.5,       # Regularization weight
    parallel: bool = False,
    logger = None,
)
# Returns: vf_loss: float, pol_loss: float, reg_loss: float

# Target network updates
maddpg.update_all_targets()  # Soft update: θ_target ← τ*θ + (1-τ)*θ_target

# Mode switching
maddpg.prep_training(device='gpu')   # Training mode, move to GPU
maddpg.prep_rollouts(device='cpu')   # Eval mode, move to CPU

# Exploration
maddpg.scale_noise(scale: float, new_epsilon: float)
maddpg.reset_noise()

# Save/Load
maddpg.save(filename: str)
maddpg = MADDPG.load(filename: str)
```

### Replay Buffer

```python
# Initialization (centralised critic — joint rows)
buffer = ReplayBufferAgent(
    max_steps: int,     # Buffer capacity in timesteps (= total rows)
    num_agents: int,    # Physical agents (e.g., 24) — sets joint row width
    state_dim: int,     # Per-agent obs_dim
    action_dim: int,    # Per-agent action_dim (2)
    # start_stop_index: ignored, removed from call sites
)

# Push experience (no index argument)
buffer.push(
    observations: np.ndarray,           # (obs_dim, n_agents) column-major
    actions: np.ndarray,                # (action_dim, n_agents) column-major
    rewards: np.ndarray,                # (1, n_agents)
    next_observations: np.ndarray,      # (obs_dim, n_agents)
    dones: np.ndarray,                  # (1, n_agents)
    # index: removed
    actions_prior_orig: np.ndarray = None,  # (action_dim, n_agents)
)
# Stores 1 joint row: obs flattened to (1, n_agents*obs_dim), rew meaned to (1,1), etc.

# Sample batch
obs, acs, rews, next_obs, dones, acs_prior, _ = buffer.sample(
    N: int,              # Batch size
    to_gpu: bool = False,
    is_prior: bool = False,
)
# Returns joint tensors:
#   obs/next_obs: (N, n_agents*obs_dim)    e.g. (512, 4608) at K=6,M=80
#   acs/acs_prior:(N, n_agents*action_dim) e.g. (512, 48)
#   rews/dones:   (N, 1)

# Query
length = len(buffer)  # Number of experiences stored
avg_reward = buffer.get_average_rewards(N=1000)  # Mean of last N rewards
```

### Networks

```python
# Actor/Critic Network
network = MLPNetwork(
    input_dim: int,
    out_dim: int,
    hidden_dim: int = 64,
    nonlin = F.leaky_relu,
    constrain_out: bool = False,  # True for actor (tanh), False for critic
    discrete_action: bool = False,
)

# Forward pass
output = network(input: Tensor)  # (batch_size, input_dim) → (batch_size, out_dim)
```

---

## 2. Common Code Patterns

### Pattern 1: Complete Training Setup

```python
import torch
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter
from jaxmarl.environments.mpe.assembly import AssemblyEnv
from gym.wrappers import JaxAssemblyAdapterGPU
from algorithm.algorithms import MADDPG
from algorithm.utils import ReplayBufferAgent
from cfg.assembly_cfg import gpsargs as cfg

# 1. Initialize environment
jax_env = AssemblyEnv(results_file='fig/results.pkl', n_a=30)
env = JaxAssemblyAdapterGPU(jax_env, n_envs=1, seed=42, alpha=1.0)

# 2. Initialize MADDPG
maddpg = MADDPG.init_from_env(
    env, agent_alg='MADDPG', tau=0.01, lr_actor=1e-4, 
    lr_critic=1e-3, hidden_dim=180, device='gpu', 
    epsilon=0.1, noise=0.9
)

# 3. Initialize buffer
buffer = ReplayBufferAgent(
    20000, env.num_agents, slice(0, env.num_agents),
    state_dim=192, action_dim=2
)

# 4. Setup logging
logger = SummaryWriter('./logs')

# 5. Training loop
for ep in range(3000):
    # Rollout
    obs = env.reset()
    maddpg.prep_rollouts('cpu')
    for t in range(200):
        actions, _ = maddpg.step(torch.Tensor(obs), [slice(0, 30)], explore=True)
        actions_np = np.column_stack([a.numpy() for a in actions])
        next_obs, rews, dones, _, prior = env.step(actions_np.T)
        buffer.push(obs, actions_np, rews, next_obs, dones, slice(0, 30), prior)
        obs = next_obs
    
    # Train
    maddpg.prep_training('gpu')
    for _ in range(20):
        if len(buffer) >= 512:
            batch = buffer.sample(512, to_gpu=True)
            maddpg.update(*batch, 0, env.alpha, logger=logger)
        maddpg.update_all_targets()
    
    # Save
    if ep % 100 == 0:
        maddpg.save(f'model_ep{ep}.pt')
```

### Pattern 2: Load and Evaluate Trained Model

```python
import torch
import numpy as np
from jaxmarl.environments.mpe.assembly import AssemblyEnv
from gym.wrappers import JaxAssemblyAdapterGPU
from algorithm.algorithms import MADDPG

# 1. Load model
maddpg = MADDPG.load('model_ep3000.pt')

# 2. Create environment
jax_env = AssemblyEnv(results_file='fig/results.pkl', n_a=30)
env = JaxAssemblyAdapterGPU(jax_env, n_envs=1, seed=42)

# 3. Evaluate
maddpg.prep_rollouts('cpu')
obs = env.reset()
total_reward = 0.0

with torch.no_grad():
    for t in range(200):
        actions, _ = maddpg.step(torch.Tensor(obs), [slice(0, 30)], explore=False)
        actions_np = np.column_stack([a.numpy() for a in actions])
        obs, rews, dones, _, _ = env.step(actions_np.T)
        total_reward += rews.mean()

print(f"Episode reward: {total_reward}")
print(f"Coverage: {env.coverage_rate():.3f}")
print(f"Uniformity: {env.distribution_uniformity():.3f}")
```

### Pattern 3: Debugging Shape Mismatches

```python
# Check shapes at each step
print(f"Env reset obs: {obs.shape}")  # Expected: (192, 30)
print(f"Torch obs: {torch_obs.shape}")  # Expected: (192, 30)
print(f"MADDPG actions: {[a.shape for a in actions]}")  # Expected: [(2, 30)]
print(f"Env step input: {actions_np.T.shape}")  # Expected: (30, 2)
print(f"Env step output obs: {next_obs.shape}")  # Expected: (192, 30)
print(f"Env step output prior: {prior.shape}")  # Expected: (2, 30)
print(f"Buffer sample obs: {obs_sample.shape}")  # Expected: (512, 192)
print(f"Buffer sample acs: {acs_sample.shape}")  # Expected: (512, 2)
```

---

## 3. Configuration Parameters

### Environment Config (`cfg/assembly_cfg.py`)

```python
# Environment
n_a = 30                    # Number of agents
agent_strategy = 'input'    # Control: 'input'/'random'/'rule'
results_file = 'fig/results.pkl'  # Target shapes

# Training
seed = 226                  # Random seed
n_rollout_threads = 1       # Parallel environments
n_training_threads = 5      # CPU threads for training
buffer_length = 20000       # Replay buffer capacity
n_episodes = 3000           # Total episodes
episode_length = 200        # Steps per episode
batch_size = 512            # Training batch size

# Networks
hidden_dim = 180            # Hidden layer size
lr_actor = 1e-4             # Actor learning rate
lr_critic = 1e-3            # Critic learning rate

# Exploration
epsilon = 0.1               # Random action probability
noise_scale = 0.9           # Initial Gaussian noise scale
tau = 0.01                  # Target network soft update rate

# Algorithm
agent_alg = 'MADDPG'        # Algorithm type
device = 'gpu'              # 'cpu' or 'gpu'

# Checkpointing
save_interval = 100         # Save every N episodes
eval_interval = 100         # Evaluate every N episodes
eval_episodes = 3           # Episodes per evaluation
gif_dir = './eval_gifs'     # GIF output directory
```

### Memory Calculation

```python
# Buffer memory (centralised critic — joint rows)
# total_length = buffer_length (one joint row per timestep)
buffer_mem_MB = (
    buffer_length *
    n_a *
    (obs_dim * 2 + action_dim * 2) *
    4  # bytes per float32
) / (1024**2)

# K=6, M=80 (n_a=24, obs_dim=192): 20000 * 24 * (192*2 + 2*2) * 4 / 1024^2 ≈ 745 MB
# K=3, M=10 (n_a=24, obs_dim=40):  20000 * 24 * (40*2 + 2*2)  * 4 / 1024^2 ≈ 157 MB
```

---

## 4. Debugging Scenarios

### Scenario 1: Training Not Improving

**Check**:
1. Buffer filling: `print(len(buffer))` — should reach `buffer_length * n_agents`
2. Rewards increasing: check `avg_reward` in logs
3. Exploration: `print(maddpg.noise)` — should decay from 0.9 to 0.5
4. Regularization: check `env.alpha` — should be 0.1 or decaying
5. Losses: `vf_loss` should decrease, `pol_loss` should become more negative

**Fix**:
```python
# Add more logging
print(f"Buffer size: {len(buffer)}, Reward: {avg_reward:.3f}, Noise: {maddpg.noise:.3f}")

# Check if updates happening
if len(buffer) < batch_size:
    print("WARNING: Buffer not full enough for training!")
```

### Scenario 2: NaN Losses

**Check**:
1. Reward normalization: `norm_rews=True` in `buffer.sample()`
2. Gradient explosion: add clipping in `MADDPG.update()`
3. Learning rates too high: reduce `lr_actor`, `lr_critic`

**Fix**:
```python
# Add gradient clipping (in maddpg.py, after loss.backward())
torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)

# Check for NaN
if torch.isnan(pol_loss):
    print("NaN detected in policy loss!")
    # Optionally skip update or reduce LR
```

### Scenario 3: GPU Out of Memory

**Check**:
1. JAX memory: `XLA_PYTHON_CLIENT_MEM_FRACTION` set to 0.15
2. Buffer size: `buffer_length * n_rollout_threads * n_a`
3. Batch size: `batch_size` too large
4. Number of parallel envs: `n_rollout_threads`

**Fix**:
```python
# Reduce buffer
buffer_length = 10000  # Was 20000

# Reduce batch size
batch_size = 256  # Was 512

# Reduce parallel envs
n_rollout_threads = 1  # Was 4

# Check GPU memory usage
import torch
print(f"GPU allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Scenario 4: Actions All Zeros or Constant

**Check**:
1. Network initialization: check policy outputs random values initially
2. Exploration: `explore=True` during training
3. Action clipping: not stuck at boundary

**Fix**:
```python
# Check network output
with torch.no_grad():
    test_obs = torch.randn(10, 192)
    test_action = maddpg.agents[0].policy(test_obs)
    print(f"Policy output mean: {test_action.mean():.3f}, std: {test_action.std():.3f}")
    # Should be non-zero mean and std

# Ensure exploration enabled
actions, _ = maddpg.step(obs, [slice(0, 30)], explore=True)  # ← Must be True!
```

### Scenario 5: Coverage Not Increasing

**Check**:
1. Are agents reaching target region? `print(env._states.p_pos.mean(axis=0))`
2. Target positions valid? Check `grid_center` not all zeros
3. Reward signal strong enough? Check `r_assem` component

**Fix**:
```python
# Check agent positions vs target
state = env._states
print(f"Agent positions: {state.p_pos[:5]}")  # First 5 agents
print(f"Target positions: {state.grid_center.T[:5]}")  # First 5 cells

# Check reward breakdown (add to environment)
# In assembly.py _rewards_fast, add:
# print(f"r_assem: {r_assem.mean():.3f}, penalty_avoid: {penalty_avoid.mean():.3f}")
```

---

## 5. Where to Add New Features

### Add New Reward Component

**File**: `JaxMARL/jaxmarl/environments/mpe/assembly.py`
**Function**: `_rewards_fast(state, cached)`

```python
def _rewards_fast(self, state, cached):
    # ... existing rewards ...
    
    # Add new reward
    new_reward = compute_new_reward(state, cached)
    
    # Combine
    return (
        r_assem 
        + penalty_avoid 
        + penalty_entering 
        + penalty_exploration
        + new_reward  # ← Add here
    )
```

### Add New Observation Component

**File**: `JaxMARL/jaxmarl/environments/mpe/assembly.py`
**Function**: `_get_obs_fast(state, cached)`

```python
def _get_obs_fast(self, state, cached):
    # ... existing obs components ...
    
    # Add new observation
    new_obs = compute_new_obs(state)  # Shape: [n_a, new_dim]
    
    # Update obs_dim in __init__:
    # self.obs_dim = 4*(topo_nei_max+1) + 4 + 2*num_obs_grid_max + new_dim
    
    # Concatenate
    return jnp.concatenate([
        obs_agent_flat, 
        target_rel_pos, 
        target_rel_vel, 
        sensed_flat,
        new_obs,  # ← Add here
    ], axis=-1)
```

### Add New Metric

**File**: `MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py`
**Function**: `run(cfg)` after episode rollout

```python
# Compute new metric
new_metric = compute_new_metric(env)

# Log
logger.add_scalar("agent/new_metric", new_metric, ep_index)

# Print
print(f"New Metric: {new_metric:.3f}")
```

### Change Network Architecture

**File**: `MARL-LLM/marl_llm/algorithm/utils/networks.py`
**Class**: `MLPNetwork`

```python
class MLPNetwork(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64, ...):
        super().__init__()
        # Change: Add more layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)  # ← New layer
        self.fc5 = nn.Linear(hidden_dim, out_dim)     # ← Renamed
        
    def forward(self, X):
        h1 = self.nonlin(self.fc1(X))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.nonlin(self.fc3(h2))
        h4 = self.nonlin(self.fc4(h3))  # ← New
        out = self.out_fn(self.fc5(h4))  # ← Updated
        return out
```

### Add Curriculum Learning

**File**: `MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py`
**Function**: `run(cfg)`

```python
# After initialization, before training loop
difficulty_level = 0

for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):
    # Increase difficulty every 500 episodes
    if ep_index > 0 and ep_index % 500 == 0:
        difficulty_level += 1
        env.set_difficulty(difficulty_level)  # Custom method
        print(f"Increased difficulty to level {difficulty_level}")
    
    # ... rest of training loop ...
```

---

## 6. File Locations

### Core Files (Most Frequently Modified)

| File | Purpose | Modify When... |
|------|---------|----------------|
| `train/train_assembly_jax_gpu.py` | Main training loop | Change training procedure, logging |
| `algorithm/algorithms/maddpg.py` | MADDPG algorithm | Change update rules, losses |
| `JaxMARL/jaxmarl/environments/mpe/assembly.py` | Environment | Change physics, rewards, observations |
| `cfg/assembly_cfg.py` | Configuration | Change hyperparameters |
| `algorithm/utils/networks.py` | Neural networks | Change architecture |

### Supporting Files

| File | Purpose |
|------|---------|
| `algorithm/utils/agents.py` | DDPGAgent (individual agent) |
| `algorithm/utils/buffer_agent.py` | Replay buffer |
| `cus_gym/gym/wrappers/customized_envs/jax_assembly_wrapper.py` | JAX-PyTorch adapter |
| `train/eval_render.py` | Evaluation and GIF rendering |
| `eval/eval_assembly.py` | Standalone evaluation script |

### Documentation Files

| File | Purpose |
|------|---------|
| `Docs/SYSTEM_ARCHITECTURE.md` | High-level overview, navigation |
| `Docs/CORE_COMPONENTS.md` | MADDPG, agents, buffers, networks |
| `Docs/ENVIRONMENT_INTERFACE.md` | Environment mechanics, rewards |
| `Docs/DATA_FLOW.md` | Tensor shapes, transformations |
| `Docs/TRAINING_PIPELINE.md` | Complete training walkthrough |
| `Docs/QUICK_REFERENCE.md` | This file |

---

## Quick Commands

```bash
# Train from scratch (requires GPU)
cd MARL-LLM/marl_llm
python train/train_assembly_jax_gpu.py

# Monitor training
tensorboard --logdir=./models/assembly/

# Evaluate saved model
python eval/eval_assembly.py --model_path=./models/assembly/.../model.pt

# Check GPU usage
nvidia-smi -l 1

# Check DLPack is available
python -c "from jax import dlpack; from torch.utils import dlpack; print('DLPack OK')"
```

---

## Environment Variables

```bash
# Limit JAX GPU memory (set before running Python)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

# Disable JAX preallocation (debugging)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# Increase logging verbosity
export TF_CPP_MIN_LOG_LEVEL=0
```

---

**For more details**: See respective documentation files listed in SYSTEM_ARCHITECTURE.md.
