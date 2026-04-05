# MARL-LLM System Architecture

**Purpose**: Main entry point for agents to understand the codebase structure and navigate to specific components.

## Overview

This is a Multi-Agent Reinforcement Learning (MARL) system using MADDPG (Multi-Agent Deep Deterministic Policy Gradient) to train swarm agents to assemble into target shapes. The system combines:
- **JAX** for fast parallel environment simulation (GPU-accelerated)
- **PyTorch** for neural network training (actor-critic architecture)
- **Assembly swarm environment** where agents learn to form geometric patterns

## System Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
│                (train_assembly_jax_gpu.py)                  │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
    │  ENVIRONMENT  │ │   ALGORITHM  │ │    BUFFER    │
    │               │ │              │ │              │
    │ JAX Assembly  │ │   MADDPG     │ │   Replay     │
    │   (GPU)       │ │  (PyTorch)   │ │   Buffer     │
    └───────────────┘ └──────────────┘ └──────────────┘
            │                 │                 │
            └────────┬────────┴─────────────────┘
                     ▼
            Data Flow (JAX → NumPy → PyTorch)
```

## Repository Structure

```
new_marl_llm_implementation/
├── MARL-LLM/                    # Main training and algorithm code
│   ├── marl_llm/
│   │   ├── train/               # Training scripts
│   │   │   ├── train_assembly_jax_gpu.py  # MAIN TRAINING ENTRY POINT (GPU-optimized)
│   │   │   └── eval_render.py             # Evaluation and GIF rendering
│   │   ├── algorithm/           # RL algorithms
│   │   │   ├── algorithms/
│   │   │   │   └── maddpg.py             # MADDPG implementation
│   │   │   └── utils/
│   │   │       ├── agents.py             # DDPGAgent (individual agent)
│   │   │       ├── buffer_agent.py       # Replay buffer
│   │   │       ├── networks.py           # Actor/critic networks
│   │   │       ├── noise.py              # Exploration noise
│   │   │       └── misc.py               # Utility functions
│   │   ├── cfg/                 # Configuration
│   │   │   └── assembly_cfg.py           # Hyperparameters and image processing
│   │   └── eval/                # Evaluation scripts
│   └── cus_gym/                 # Custom gym wrappers
│       └── gym/wrappers/
│           └── customized_envs/
│               ├── jax_assembly_wrapper.py    # JAX→PyTorch adapter
│               └── jax_assembly_wrapper_gpu.py # GPU variant
├── JaxMARL/                     # JAX environment implementation
│   └── jaxmarl/environments/mpe/
│       └── assembly.py          # AssemblyEnv (core JAX environment)
├── fig/                         # Target shape images and preprocessed data
│   └── results.pkl              # Preprocessed grid coordinates
└── Docs/                        # Documentation (you are here)
```

## Component Hierarchy

### 1. **Environment Layer** (JAX)
- **Location**: `JaxMARL/jaxmarl/environments/mpe/assembly.py`
- **Purpose**: Simulates multi-agent physics and rewards on GPU
- **Key**: Uses `jax.vmap` for parallel environments
- **See**: ENVIRONMENT_INTERFACE.md

### 2. **Adapter Layer** (JAX ↔ PyTorch Bridge)
- **Location**: `MARL-LLM/cus_gym/gym/wrappers/customized_envs/jax_assembly_wrapper_gpu.py`
- **Purpose**: Wraps JAX environment to expose gym-like API for PyTorch with GPU-optimized data flow
- **Key**: Uses DLPack for zero-copy GPU tensor sharing between JAX and PyTorch (no CPU transfers during rollout)
- **See**: ENVIRONMENT_INTERFACE.md, DATA_FLOW.md

### 3. **Algorithm Layer** (PyTorch)
- **Location**: `MARL-LLM/marl_llm/algorithm/`
- **Purpose**: MADDPG actor-critic learning
- **Components**:
  - `MADDPG` class: Orchestrates multiple agents
  - `DDPGAgent` class: Individual agent (actor + critic)
  - `ReplayBufferAgent`: Experience storage
- **See**: CORE_COMPONENTS.md

### 4. **Training Loop** (PyTorch + JAX)
- **Location**: `MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py`
- **Purpose**: Coordinates rollouts, updates, and evaluation with GPU-optimized data flow
- **See**: TRAINING_PIPELINE.md

## Key Concepts

### Multi-Environment Parallelism
- `n_rollout_threads` environments run in parallel via `jax.vmap`
- Each environment has `n_a` agents (default 30)
- Total agents in buffer = `n_rollout_threads × n_a`
- **Memory scaling**: Buffer size = `buffer_length × n_rollout_threads × n_a × obs_dim × 4 bytes`

### Data Flow Pattern (GPU-Optimized)
```
JAX Environment (GPU)  →  DLPack zero-copy  →  PyTorch CUDA Tensors (GPU)  →  MADDPG Networks (GPU)
     AssemblyEnv       →  JaxAssemblyAdapterGPU  →   Stay on GPU during rollout
                                                   ↓
                                          Bulk GPU→CPU transfer at episode end
                                                   ↓
                                            Replay Buffer (CPU storage)
```

### Hybrid Computation Strategy
- **JAX**: Fast parallel environment simulation (~20-100MB GPU memory)
- **PyTorch**: Neural network training (~11+ GB GPU memory)
- **Memory split**: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15` limits JAX to 15% GPU
- **DLPack**: Zero-copy tensor sharing between JAX and PyTorch on same GPU device
- **Performance**: 15-25% faster than CPU version due to no intermediate transfers

## Configuration System

**Location**: `MARL-LLM/marl_llm/cfg/assembly_cfg.py`

Key parameters:
- `n_a`: Number of agents per environment (default: 24)
- `n_rollout_threads`: Parallel environments (default: 1)
- `buffer_length`: Replay buffer capacity in timesteps (default: 20000)
- `episode_length`: Steps per episode (default: 200)
- `hidden_dim`: Actor hidden layer size (default: 180)
- `critic_hidden_dim`: Centralised critic hidden layer size (default: 256 — larger than actor because critic input is `n_agents×(obs_dim+2)`)
- `lr_actor`, `lr_critic`: Learning rates
- `results_file`: Preprocessed target shapes (`fig/results.pkl`)

## Navigation Guide

**I want to...**

| Task | Look Here |
|------|-----------|
| Understand the training loop | `train/train_assembly_jax_gpu.py` + TRAINING_PIPELINE.md |
| Modify the environment physics | `JaxMARL/jaxmarl/environments/mpe/assembly.py` + ENVIRONMENT_INTERFACE.md |
| Change the neural network architecture | `algorithm/utils/networks.py` + CORE_COMPONENTS.md |
| Adjust the learning algorithm | `algorithm/algorithms/maddpg.py` + CORE_COMPONENTS.md |
| Debug tensor shape mismatches | DATA_FLOW.md |
| Add new target shapes | `cfg/assembly_cfg.py` (process images → `results.pkl`) |
| Modify exploration strategy | `algorithm/utils/agents.py` (DDPGAgent.step) |
| Change reward function | `JaxMARL/jaxmarl/environments/mpe/assembly.py` (_rewards_fast) |
| Optimize GPU memory usage | Adjust `XLA_PYTHON_CLIENT_MEM_FRACTION` or reduce `n_rollout_threads` |

## Entry Points

### Training
```bash
python MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py
```
- Requires CUDA-capable GPU
- Uses DLPack for zero-copy JAX↔PyTorch transfers
- Loads config from `assembly_cfg.py`
- Creates environment, MADDPG agent, replay buffer
- Runs GPU-optimized training loop with periodic evaluation

### Evaluation
```bash
python MARL-LLM/marl_llm/eval/eval_assembly.py
```

## Critical Dependencies

1. **GPU Requirements**
   - CUDA-capable GPU required
   - JAX 0.7.2+ with CUDA support
   - PyTorch with CUDA enabled
   - DLPack support (included in modern JAX/PyTorch)
   - Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15` before any JAX import

2. **Tensor shape conventions (all GPU tensors during rollout)**
   - Observations: `(obs_dim, n_envs*n_a)` — **features in rows, agents in columns** — `torch.cuda.FloatTensor`
   - Actions: `(n_envs*n_a, 2)` for step input; `(2, n_envs*n_a)` for buffer storage
   - See DATA_FLOW.md for full details

3. **DLPack zero-copy transfers**
   - JAX GPU arrays → PyTorch CUDA tensors with no intermediate copy
   - Requires `.detach()` on tensors passed to env.step() (DLPack cannot export tensors with gradients)
   - 15-25% faster than CPU-intermediated version

4. **Prior actions (expert behavior)**
   - JAX environment computes Reynolds flocking actions (`robot_policy`)
   - Used for regularization: `pol_loss + α * MSE(policy_action, prior_action)`
   - Helps bootstrap learning with reasonable swarm behavior

## Common Pitfalls

1. **Forgetting .detach() on env.step()**
   - DLPack cannot export tensors with gradients
   - Always use: `env.step(actions_gpu.t().detach())`
   - Error if missing: "RuntimeError: Can't export tensor that requires grad"

2. **Networks on wrong device during rollout**
   - GPU version keeps networks on GPU: `maddpg.prep_rollouts(device='gpu')`
   - NOT `device='cpu'` like older versions
   
3. **Forgetting bulk transfer pattern**
   - Don't transfer to CPU each step — accumulate on GPU
   - Single bulk transfer at episode end for buffer storage

4. **Alpha (regularization) not updating**
   - Must access `env.alpha` (not `env.env.alpha`)
   - See: `train_assembly_jax_gpu.py:425`

## Next Steps

For detailed understanding of specific components:
1. **CORE_COMPONENTS.md** — Deep dive into MADDPG, agents, buffers, networks
2. **ENVIRONMENT_INTERFACE.md** — How the environment works, state, actions, rewards
3. **DATA_FLOW.md** — Tensor shapes and transformations throughout the pipeline
4. **TRAINING_PIPELINE.md** — Step-by-step training process
5. **QUICK_REFERENCE.md** — Function signatures and common operations
