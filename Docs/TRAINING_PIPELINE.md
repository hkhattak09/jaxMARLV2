# MARL-LLM Training Pipeline

**Purpose**: Complete walkthrough of the training process from initialization to checkpointing.

## Overview

The training pipeline orchestrates:
1. Environment and algorithm initialization
2. Episode rollouts (data collection)
3. Network updates (learning)
4. Periodic evaluation and checkpointing

**Main Entry Point**: `MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py::run(cfg)`

**GPU-Optimized**: Uses DLPack for zero-copy JAX↔PyTorch tensor sharing. All data stays on GPU during rollout.

---

## 1. Initialization Phase

### Setup Logging

```python
model_dir = "./models/" / cfg.env_name  # "./models/assembly"
curr_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
run_dir = model_dir / curr_run  # "./models/assembly/2026-04-01-15-30-00"
log_dir = run_dir / "logs"
os.makedirs(log_dir)
logger = SummaryWriter(str(log_dir))  # TensorBoard logger
```

**Output Directory Structure**:
```
./models/assembly/2026-04-01-15-30-00/
├── logs/                    # TensorBoard logs
│   ├── events.out.tfevents.*
│   └── summary.json
├── incremental/             # Periodic checkpoints
│   ├── model_ep400.pt
│   ├── model_ep800.pt
│   └── ...
└── model.pt                 # Final trained model
```

### Set Random Seeds

```python
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

if cfg.device == "gpu":
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
```

**Purpose**: Reproducibility. Same seed → same training trajectory.

### Configure Compute Resources

```python
# Verify GPU is available
if not torch.cuda.is_available():
    raise RuntimeError("GPU-optimized version requires CUDA-enabled PyTorch")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")

if cfg.device == "cpu":
    torch.set_num_threads(cfg.n_training_threads)
elif cfg.device == "gpu":
    # JAX memory limit already set before imports:
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
```

### Build Environment

```python
# 1. Create JAX environment
jax_env = AssemblyEnv(
    results_file=cfg.results_file,  # 'fig/results.pkl'
    n_a=cfg.n_a,                    # Default: 24 agents
    topo_nei_max=cfg.topo_nei_max,  # K nearest neighbours each agent observes (default 6)
    grid_obs_fraction=cfg.grid_obs_fraction,  # Alternative to num_obs_grid_max (None = use 80)
    d_sen=cfg.d_sen,                # Sensing radius (default 0.4, CLI: --d_sen)
    r_avoid=cfg.r_avoid,            # Personal space radius (default 0.10, CLI: --r_avoid)
)

# 2. Wrap with GPU-optimized adapter (uses DLPack)
env = JaxAssemblyAdapterGPU(
    jax_env,
    n_envs=cfg.n_rollout_threads,  # Default: 1 (can be 2, 4, 8 for parallelism)
    seed=cfg.seed,
    alpha=1.0,  # Initial regularization coefficient
)
```

**Note**: `d_sen` and `r_avoid` are now CLI flags (added in physics/reward redesign). Old code that omits them will use the hardcoded defaults (0.4, 0.10). Old checkpoints trained without these flags used the old dynamic `r_avoid=0.29` formula — those checkpoints are incompatible.

**Key**: `JaxAssemblyAdapterGPU` returns PyTorch CUDA tensors via DLPack. All data stays on GPU during rollout.

### Initialize MADDPG

```python
# Build CTM config if using CTM actor (default)
ctm_config = {
    'd_model': cfg.ctm_d_model,               # 256
    'memory_length': cfg.ctm_memory_length,   # 16
    'n_synch_out': cfg.ctm_n_synch_out,       # 16
    'iterations': cfg.ctm_iterations,         # 4
    'synapse_depth': cfg.ctm_synapse_depth,   # 1
    'deep_nlms': cfg.ctm_deep_nlms,           # False
    'do_layernorm_nlm': cfg.ctm_do_layernorm_nlm,  # True
    'memory_hidden_dims': [cfg.ctm_memory_hidden_dims],  # [64]
    'dropout': 0,
} if cfg.use_ctm_actor else None

maddpg = MADDPG.init_from_env(
    env,
    agent_alg=cfg.agent_alg,      # "MADDPG"
    tau=cfg.tau,                  # 0.01 (soft update rate)
    lr_actor=cfg.lr_actor,        # 1e-4
    lr_critic=cfg.lr_critic,      # 1e-3
    hidden_dim=cfg.hidden_dim,    # 180 (MLP actor only; CTM uses ctm_config)
    device=cfg.device,            # "gpu"
    epsilon=cfg.epsilon,          # 0.1 (random action probability)
    noise=cfg.noise_scale,        # 0.9 (Gaussian noise scale)
    use_ctm_actor=cfg.use_ctm_actor,  # True (default); --use_mlp_actor to revert
    ctm_config=ctm_config,
)
```

**Result**: MADDPG object with:
- 1 CTMDDPGAgent (default) or DDPGAgent (--use_mlp_actor)
- Actor: CTMActor (ContinuousThoughtMachineRL + Linear(136,2) + Tanh) or MLPNetwork
- Critic: AggregatingCritic (permutation-equivariant centralised critic, shared across both actor types)
- Target networks (soft-updated copies)
- Optimizers (Adam for actor and critic)

### Create Replay Buffer

```python
agent_buffer = [
    ReplayBufferAgent(
        cfg.buffer_length,      # 20000
        env.num_agents,         # n_rollout_threads × n_a
        state_dim=env.observation_space.shape[0],  # 192
        action_dim=env.action_space.shape[0],      # 2
        start_stop_index=slice(0, env.num_agents),
    )
]
```

**Memory**: ~934 MB for single env (n_rollout_threads=1), ~3.7 GB for 4 parallel envs.

---

## 2. Training Loop Structure

### High-Level Flow

```python
for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):
    # 1. Episode Rollout (collect experience)
    for step in range(cfg.episode_length):  # 200 steps
        actions = select_actions(obs)
        next_obs, rewards, dones, prior = env.step(actions)
        buffer.push(obs, actions, rewards, next_obs, dones, prior)
        obs = next_obs
    
    # 2. Training Phase (update networks)
    for _ in range(20):  # 20 update iterations
        for agent_i in range(n_agent_types):
            sample = buffer.sample(batch_size)
            maddpg.update(sample, agent_i, alpha)
        maddpg.update_all_targets()
    
    # 3. Logging and Evaluation
    if ep_index % eval_interval == 0:
        run_eval(maddpg, env)
    
    if ep_index % save_interval == 0:
        maddpg.save(checkpoint_path)
```

---

## 3. Episode Rollout Phase (Data Collection)

### Preparation

```python
obs_list = []  # Accumulate GPU tensors during rollout
actions_list = []
rewards_list = []
next_obs_list = []
dones_list = []
prior_list = []

obs_gpu = env.reset()  # (192, 30) torch.cuda.FloatTensor or (192, n_envs*30)
start_stop_num = [slice(0, env.n_a)]  # Agent slice for MADDPG

# Set networks to eval mode, KEEP ON GPU
maddpg.prep_rollouts(device="gpu")  # ← GPU, not CPU!
maddpg.scale_noise(maddpg.noise, maddpg.epsilon)  # Set current exploration level
maddpg.reset_noise()  # Reset noise generator
```

**Critical**: Networks stay on GPU during rollout (unlike older CPU version).

### Step Loop (All on GPU)

```python
for et_index in range(cfg.episode_length):  # 200 steps
    # 1. Select actions (networks on GPU, obs already on GPU)
    actions, _ = maddpg.step(
        obs_gpu,  # Already torch.cuda.FloatTensor, no conversion needed!
        start_stop_num, 
        explore=True  # Add exploration noise
    )
    # actions: List[torch.cuda.FloatTensor(2, 30)]
    
    # 2. Stack and prepare for environment
    actions_gpu = torch.column_stack(actions)  # (2, 30) GPU tensor
    
    # 3. Step environment (MUST .detach() for DLPack)
    next_obs_gpu, rewards_gpu, dones_gpu, _, agent_actions_prior_gpu = env.step(
        actions_gpu.t().detach()  # (30, 2) GPU tensor, detached
    )
    # All outputs are torch.cuda tensors: obs (192, 30), rew (1, 30), prior (2, 30)
    
    # 4. Accumulate on GPU (no CPU transfer yet!)
    obs_list.append(obs_gpu)
    actions_list.append(actions_gpu.detach())
    rewards_list.append(rewards_gpu)
    next_obs_list.append(next_obs_gpu)
    dones_list.append(dones_gpu)
    prior_list.append(agent_actions_prior_gpu)
    
    # 5. Update for next step (stay on GPU)
    obs_gpu = next_obs_gpu
```

**Key Difference**: No `.numpy()` conversions, no CPU transfers. Everything accumulates on GPU.

**Timing**: ~5-10ms per step (15-25% faster than CPU version).

### Bulk Transfer and Buffer Storage

```python
# Single bulk GPU→CPU transfer at episode end
t0 = time.time()
obs_batch = torch.stack(obs_list).cpu().numpy()           # (200, 192, 30)
actions_batch = torch.stack(actions_list).cpu().numpy()   # (200, 2, 30)
rewards_batch = torch.stack(rewards_list).cpu().numpy()   # (200, 1, 30)
next_obs_batch = torch.stack(next_obs_list).cpu().numpy() # (200, 192, 30)
dones_batch = torch.stack(dones_list).cpu().numpy()       # (200, 1, 30)
prior_batch = torch.stack(prior_list).cpu().numpy()       # (200, 2, 30)
transfer_time = time.time() - t0

# Push all transitions to buffer (now on CPU)
for t in range(cfg.episode_length):
    agent_buffer[0].push(
        obs_batch[t], actions_batch[t], rewards_batch[t], 
        next_obs_batch[t], dones_batch[t],
        start_stop_num[0], prior_batch[t],
    )
```

**Performance Gain**: Single bulk transfer ~10-20ms vs 200 small transfers ~200-400ms total.

### Episode Statistics

```python
# Compute reward stats on full episode
episode_reward_mean_bar = rewards_batch.mean() * cfg.episode_length
episode_reward_std_bar = rewards_batch.std()
```

**Key Metrics**:
- `episode_reward_mean_bar`: Total cumulative mean reward over episode
- `episode_reward_std_bar`: Reward standard deviation (measures variance across agents/time)

---

## 4. Training Phase (Network Updates)

### Preparation

```python
maddpg.prep_training(device=cfg.device)  # Set to training mode, move to GPU

total_vf_loss = 0.0
total_pol_loss = 0.0
total_reg_loss = 0.0
update_count = 0
```

### Update Loop

```python
for _ in range(20):  # 20 update iterations per episode
    for a_i in range(maddpg.nagents):  # Typically 1 agent type
        # 1. Check buffer has enough data
        if len(agent_buffer[a_i]) >= cfg.batch_size:  # 512
            
            # 2. Sample batch
            sample = agent_buffer[a_i].sample(
                cfg.batch_size,  # 512
                to_gpu=True if cfg.device == "gpu" else False,
                is_prior=True,   # Include prior actions
            )
            obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, acs_prior_sample, _ = sample
            # All shapes: (512, feature_dim)
            
            # 3. Update actor and critic
            vf_loss, pol_loss, reg_loss = maddpg.update(
                obs_sample,      # (512, 192)
                acs_sample,      # (512, 2)
                rews_sample,     # (512, 1) — normalized
                next_obs_sample, # (512, 192)
                dones_sample,    # (512, 1)
                a_i,             # Agent index (0)
                acs_prior_sample,# (512, 2)
                env.alpha,       # Regularization coefficient (0.1)
                logger=logger,   # TensorBoard logger
            )
            
            # 4. Accumulate losses
            total_vf_loss += vf_loss
            total_pol_loss += pol_loss
            total_reg_loss += reg_loss
            update_count += 1
    
    # 5. Soft update target networks (once per iteration)
    maddpg.update_all_targets()
```

**Why 20 iterations?** Balances data efficiency (reuse experience) vs computational cost.

### Post-Update

```python
# Restore to rollout mode for next episode
maddpg.prep_rollouts(device="cpu")

# Decay exploration noise
maddpg.noise = max(0.5, maddpg.noise - cfg.noise_scale / cfg.n_episodes)

# Update regularization coefficient
env.alpha = 0.1  # Fixed in this implementation (could decay)

# Average losses
avg_vf_loss = total_vf_loss / max(update_count, 1)
avg_pol_loss = total_pol_loss / max(update_count, 1)
avg_reg_loss = total_reg_loss / max(update_count, 1)
```

**Exploration Decay Schedule**:
```python
noise_start = 0.9
noise_end = 0.5
decay = noise_scale / n_episodes  # 0.9 / 3000 = 0.0003
# Episode 0: 0.9, Episode 1500: 0.675, Episode 3000: 0.5
```

---

## 5. Logging and Monitoring

### Compute Metrics

```python
# Environment quality metrics (updated names — see metrics redesign)
coverage = env.sensing_coverage()               # Fraction of shape cells sensed by ≥1 agent (was coverage_rate)
uniformity = env.distribution_uniformity()      # Spacing uniformity (unchanged)
voronoi_uniformity = env.voronoi_based_uniformity()  # Voronoi diagram uniformity (unchanged)
neighbor_dist = env.mean_neighbor_distance()    # Mean nearest-neighbour distance

# R-avoid violations (replaces collision_rate + count_collisions)
# Accumulated per step as JAX scalar (no CPU sync during rollout)
# episode_r_avoid_violations = unique pairs with dist < 2*r_avoid, summed over 200 steps / n_envs
episode_r_avoid_violations = float(r_avoid_violation_sum) / cfg.n_rollout_threads

# Spring collisions: unique pairs with dist < 2*size_a = 0.07 (physical body contact)
episode_spring_collisions = float(spring_collision_sum) / cfg.n_rollout_threads

# Reward metrics
avg_reward = episode_reward_mean_bar / cfg.episode_length
reward_uniformity = 1.0 / (1.0 + episode_reward_std_bar / (abs(episode_reward_mean_bar) + 1e-8))
```

**What changed from old metrics:**
- `env.coverage_rate()` → `env.sensing_coverage()` — now uses `d_sen` (not `r_avoid/2`), reaches 1.0
- `collision_sum` / `episode_collisions` → `r_avoid_violation_sum` / `episode_r_avoid_violations` — now pairwise pairs (not agent-steps), threshold `2*r_avoid`
- `coverage_efficiency` removed entirely (was algebraically identical to old `coverage_rate`)

### Console Output (every 10 episodes)

```
====================================================================================================
Episode   100/ 3000 | Agents: 24 | Envs: 1
====================================================================================================
REWARDS (last 10 eps):  Mean:  0.8234 | Std:  0.1456 | Uniformity:  0.850
ENVIRONMENT METRICS (last 10 eps):
  - Sensing Coverage: 0.823(std:0.012) | Dist Uniformity: 0.762(std:0.018) | Voronoi Uniformity: 0.701(std:0.021)
  - R-Avoid Violations (pairs/ep/env): 42.3 | Spring Collisions (pairs/ep/env): 1.2
LOSSES (last 10 eps):   VF:  0.0234 | Policy: -0.8123 | Reg:  0.0145
TIMING (last 10 eps):   Rollout:   5.23 | Policy Exec:   1.24 | Env Step:   3.89 | Training: 12.45
====================================================================================================
```

**R-Avoid Violations interpretation**: unique agent pairs with centre-to-centre dist < 2×r_avoid = 0.20, summed over 200 steps per episode, averaged over N envs. Three mutually-close agents = 3 pairs. With 24 agents the maximum is C(24,2)×200 = 55,200 (all pairs violating every step). Values < 100 indicate good spacing.

**Spring Collisions interpretation**: unique pairs with dist < 2×size_a = 0.07 (physical body contact). Should be near zero once training converges — non-zero indicates agents physically overlapping.

### TensorBoard Logging (every save_interval=100)

```python
logger.add_scalars(
    "agent/data",
    {
        "episode_reward": avg_reward,
        "sensing_coverage": coverage,       # was "coverage"
        "nn_uniformity": uniformity,
        "voronoi_uniformity": voronoi_uniformity,
        "vf_loss": avg_vf_loss,
        "pol_loss": avg_pol_loss,
        "noise": maddpg.noise,
    },
    ep_index,
)
# Also logged: eval scalars under "eval/" and "final_eval/" keys
# All keys use correct names: "sensing_coverage", "r_avoid_violations", "spring_collisions"
```

**Visualization**: Run `tensorboard --logdir=./models/assembly/` to view training curves.

---

## 6. Checkpointing

### Incremental Checkpoints (every 4*save_interval=400 episodes)

```python
if ep_index % (4 * cfg.save_interval) < cfg.n_rollout_threads:
    os.makedirs(run_dir / "incremental", exist_ok=True)
    maddpg.save(run_dir / "incremental" / f"model_ep{ep_index + 1}.pt")
    maddpg.prep_rollouts(device="cpu")  # Restore rollout mode
```

**Checkpoint Contents**:
```python
{
    'init_dict': {
        'nagents': 1,
        'alg_types': ['agent'],
        'gamma': 0.95,
        'tau': 0.01,
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'hidden_dim': 180,
        'agent_init_params': [{'num_in_pol': 192, 'num_out_pol': 2, ...}],
        ...
    },
    'agent_params': [
        {  # Agent 0 parameters
            'policy': OrderedDict(...),
            'critic': OrderedDict(...),
            'target_policy': OrderedDict(...),
            'target_critic': OrderedDict(...),
            'policy_optimizer': {...},
            'critic_optimizer': {...},
        }
    ],
}
```

### Final Checkpoint

```python
# After all episodes
maddpg.prep_training(device=cfg.device)
maddpg.save(run_dir / "model.pt")

logger.export_scalars_to_json(str(log_dir / "summary.json"))
logger.close()
```

---

## 7. Evaluation

### Evaluation Trigger (every eval_interval=100 episodes)

```python
if ep_index > 0 and ep_index % cfg.eval_interval < cfg.n_rollout_threads:
    run_eval(maddpg, env, cfg, ep_index, logger)
```

### Evaluation Procedure

**Location**: `train_assembly_jax_gpu.py::run_eval()`

```python
def run_eval(maddpg, env, cfg, ep_index, logger):
    # 1. Prepare for evaluation — networks stay on GPU
    maddpg.prep_rollouts(device="gpu")  # Eval mode, deterministic policy, GPU
    torch_device = torch.device('cuda')
    start_stop_num = [slice(0, env.n_a)]

    total_reward = 0.0
    total_coverage = 0.0
    total_uniformity = 0.0
    total_r_avoid_violations = 0.0

    # 2. Run multiple evaluation episodes
    with torch.no_grad():
        for ep_i in range(cfg.eval_episodes):  # Default: 3
            obs_gpu = env.reset()
            ep_reward = 0.0
            state_history = [] if (ep_i == cfg.eval_episodes - 1) else None

            # 3. Rollout episode (no exploration)
            # NOTE: Currently stateless (fresh hidden state every step).
            # UPCOMING: stateful rollout — hidden states carried across steps,
            # matching the new stateful training with burn-in.
            for _ in range(cfg.episode_length):  # 200 steps
                eval_hidden = (maddpg.agents[0].policy.get_initial_hidden_state(env.n_a, torch_device)
                               if maddpg.use_ctm_actor else None)
                # step() returns 3-tuple always
                torch_agent_actions, _, _ = maddpg.step(
                    obs_gpu, start_stop_num, explore=False, hidden_states=eval_hidden
                )
                agent_actions_gpu = torch.column_stack(torch_agent_actions)
                obs_gpu, rewards_gpu, _, _, _ = env.step(agent_actions_gpu.t().detach())
                ep_reward += rewards_gpu.mean().item()

                if state_history is not None:
                    state_history.append(env._states)

            # 4. Accumulate metrics (updated names)
            total_reward += ep_reward / cfg.episode_length
            total_coverage += env.sensing_coverage()           # was coverage_rate()
            total_uniformity += env.distribution_uniformity()
            total_r_avoid_violations += env.r_avoid_violation_count()  # was collision_rate()

    # 5. Average and log (updated keys)
    mean_reward = total_reward / cfg.eval_episodes
    mean_coverage = total_coverage / cfg.eval_episodes
    mean_uniformity = total_uniformity / cfg.eval_episodes
    mean_r_avoid_violations = total_r_avoid_violations / cfg.eval_episodes

    print(f"[EVAL] ep {ep_index} | reward: {mean_reward:.4f} | "
          f"sensing_coverage: {mean_coverage:.4f} | r_avoid_violations: {mean_r_avoid_violations:.1f}")
    logger.add_scalars("eval", {
        "reward": mean_reward,
        "sensing_coverage": mean_coverage,       # was "coverage"
        "uniformity": mean_uniformity,
        "r_avoid_violations": mean_r_avoid_violations,  # new (replaced collision_rate)
    }, ep_index)

    # 6. Restore training mode
    maddpg.prep_training(device=cfg.device)
```

**Key Differences from Training Rollout**:
- `explore=False` → deterministic policy
- `torch.no_grad()` → no gradient tracking
- Collect `env._states` for visualization
- GIF saved to `cfg.gif_dir` (default: `./eval_gifs/`)

---

## 8. Complete Training Timeline

### Example: 3000 Episodes, Single Env, 30 Agents

```
Episode 0:
  ├─ Reset env → random agent positions
  ├─ Rollout 200 steps (~1-2 sec)
  │  └─ Buffer: 30 experiences added per step → 6000 total
  ├─ Training: 20 × 1 updates = 20 batches (~10-20 sec)
  │  └─ Buffer now has 6000 experiences
  └─ Log, save, eval (if triggered)

Episode 100:
  ├─ Rollout (~1-2 sec)
  │  └─ Buffer: 6000 + 6000 = 12000 experiences (circular if > 20k)
  ├─ Training: 20 updates (~10-20 sec)
  ├─ Log to TensorBoard ✓
  ├─ Eval (3 episodes, ~6-9 sec) ✓
  └─ Total time: ~18-32 sec

Episode 400:
  ├─ Rollout + Training (~12-22 sec)
  ├─ Log ✓
  ├─ Eval ✓
  └─ Save checkpoint ✓

...

Episode 3000:
  ├─ Rollout + Training
  ├─ Save final checkpoint
  └─ Export logs
```

**Total Training Time**: ~3-6 hours for 3000 episodes (GPU, single env).

---

## 9. Hyperparameter Effects

### Critical Hyperparameters

| Parameter | Default | Effect | Tuning Guidance |
|-----------|---------|--------|-----------------|
| `lr_actor` | 1e-4 | Actor learning rate | Higher → faster policy change (risk instability) |
| `lr_critic` | 1e-3 | Critic learning rate | Typically 10× actor LR |
| `hidden_dim` | 180 | Network capacity | Larger → more expressive, slower training |
| `tau` | 0.01 | Target network update rate | Smaller → more stable, slower target tracking |
| `gamma` | 0.95 | Discount factor | Higher → more far-sighted |
| `epsilon` | 0.1 | Random action probability | Balance exploration vs exploitation |
| `noise_scale` | 0.9 | Gaussian noise std | Decays to 0.5 over training |
| `alpha` | 0.1 | Prior action regularization | Higher → stay closer to Reynolds flocking |
| `batch_size` | 512 | Update batch size | Larger → more stable gradients, more memory |
| `buffer_length` | 20000 | Replay buffer capacity | Larger → more diverse data, more memory |

### Common Issues and Solutions

**Slow learning**:
- Increase `hidden_dim` (180 → 256)
- Increase `lr_actor` (1e-4 → 2e-4)
- Check `alpha` is decaying (not stuck at 1.0)

**Unstable training (loss explodes)**:
- Decrease `lr_actor` (1e-4 → 5e-5)
- Decrease `lr_critic` (1e-3 → 5e-4)
- Enable gradient clipping in MADDPG.update

**Poor exploration**:
- Increase `epsilon` (0.1 → 0.2)
- Slower `noise` decay (increase `n_episodes`)
- Check buffer has diverse data (print coverage stats)

**Memory issues**:
- Reduce `buffer_length` (20000 → 10000)
- Reduce `n_rollout_threads` (4 → 1)
- Reduce `batch_size` (512 → 256)

---

## 10. Extending the Pipeline

### Adding a New Metric

```python
# In run() function, after episode rollout:
new_metric = compute_new_metric(env, obs)

# Add to logging:
logger.add_scalar("agent/new_metric", new_metric, ep_index)
```

### Changing Update Frequency

```python
# Update every K episodes instead of every episode:
if ep_index % K == 0:
    for _ in range(20 * K):  # Do K times more updates
        # ... training code ...
```

### Adding Curriculum Learning

```python
# Gradually increase environment difficulty:
if ep_index % 500 == 0:
    env.increase_difficulty()  # Custom method
```

---

## Quick Start Command

```bash
cd MARL-LLM/marl_llm
python train/train_assembly_jax_gpu.py
```

**Default**: Trains 3000 episodes, saves to `./models/assembly/`, logs to TensorBoard.

**Monitor Progress**:
```bash
tensorboard --logdir=./models/assembly/
# Open browser to http://localhost:6006
```

---

**Next**: See QUICK_REFERENCE.md for function signatures and common operations.
