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
    n_a=cfg.n_a,                    # Default: 30 agents
)

# 2. Wrap with GPU-optimized adapter (uses DLPack)
env = JaxAssemblyAdapterGPU(
    jax_env,
    n_envs=cfg.n_rollout_threads,  # Default: 1 (can be 2, 4, 8 for parallelism)
    seed=cfg.seed,
    alpha=1.0,  # Initial regularization coefficient
)
```

**Key**: `JaxAssemblyAdapterGPU` returns PyTorch CUDA tensors via DLPack. All data stays on GPU during rollout.

### Initialize MADDPG

```python
adversary_alg = None  # Not used in assembly environment
maddpg = MADDPG.init_from_env(
    env,
    agent_alg=cfg.agent_alg,      # "MADDPG"
    adversary_alg=adversary_alg,
    tau=cfg.tau,                  # 0.01 (soft update rate)
    lr_actor=cfg.lr_actor,        # 1e-4
    lr_critic=cfg.lr_critic,      # 1e-3
    hidden_dim=cfg.hidden_dim,    # 180
    device=cfg.device,            # "gpu"
    epsilon=cfg.epsilon,          # 0.1 (random action probability)
    noise=cfg.noise_scale,        # 0.9 (Gaussian noise scale)
    name=cfg.env_name,
)
```

**Result**: MADDPG object with:
- 1 DDPGAgent (homogeneous team)
- Actor/critic networks (4-layer MLP, hidden_dim=180)
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
# Environment quality metrics
coverage = env.coverage_rate()                 # Fraction of target cells occupied
uniformity = env.distribution_uniformity()     # Spacing uniformity
voronoi_uniformity = env.voronoi_based_uniformity()  # Voronoi diagram uniformity

# Collision metric (accumulated during rollout, synced once at episode end)
# episode_collisions = total agent-collision-steps / n_rollout_threads
# Counts agents involved, not collision pairs. A single 2-agent collision
# = 2; three agents all within r_avoid of each other = 3. Minimum is 2.
# Dividing by n_rollout_threads normalises across any number of parallel envs.
episode_collisions = float(collision_sum) / cfg.n_rollout_threads

# Reward metrics
avg_reward = episode_reward_mean_bar / cfg.episode_length
reward_uniformity = 1.0 / (1.0 + episode_reward_std_bar / (abs(episode_reward_mean_bar) + 1e-8))
```

### Console Output (every 10 episodes)

```
====================================================================================================
Episode   100/ 3000 | Agents: 24 | Envs: 1
====================================================================================================
REWARDS (last 10 eps):  Mean:  0.8234 | Std:  0.1456 | Uniformity:  0.850
ENVIRONMENT METRICS (last 10 eps):
  - Coverage: 0.823(std:0.012) | Dist Uniformity: 0.762(std:0.018) | Voronoi Uniformity: 0.701(std:0.021)
  - Collisions (agent-steps/ep/env): 1240.5
LOSSES (last 10 eps):   VF:  0.0234 | Policy: -0.8123 | Reg:  0.0145
TIMING (last 10 eps):   Rollout:   5.23 | Policy Exec:   1.24 | Env Step:   3.89 | Training: 12.45
====================================================================================================
```

**Collision metric interpretation**: `agent-steps/ep/env` = total agent-timesteps in collision, summed over 200 steps, averaged over last 10 episodes and N parallel envs. With 24 agents and 200 steps, theoretical maximum is 24×200 = 4800 (all agents in collision every step). Lower is better. Early training values of 1000-4000 are typical as agents explore randomly; expect this to decrease as policy improves.

### TensorBoard Logging (every save_interval=100)

```python
logger.add_scalars(
    "agent/data",
    {
        "episode_reward": avg_reward,
        "coverage": coverage,
        "nn_uniformity": uniformity,
        "voronoi_uniformity": voronoi_uniformity,
        "vf_loss": avg_vf_loss,
        "pol_loss": avg_pol_loss,
        "noise": maddpg.noise,
    },
    ep_index,
)
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
    # 1. Prepare for evaluation
    maddpg.prep_rollouts(device="cpu")  # Eval mode, deterministic policy
    start_stop_num = [slice(0, env.n_a)]
    
    total_reward = 0.0
    total_coverage = 0.0
    total_uniformity = 0.0
    
    # 2. Run multiple evaluation episodes
    with torch.no_grad():  # Disable gradient tracking
        for ep_i in range(cfg.eval_episodes):  # Default: 3
            obs = env.reset()
            ep_reward = 0.0
            is_last_ep = (ep_i == cfg.eval_episodes - 1)
            state_history = [] if is_last_ep else None
            
            # 3. Rollout episode (no exploration)
            for _ in range(cfg.episode_length):  # 200 steps
                torch_obs = torch.Tensor(obs).requires_grad_(False)
                torch_agent_actions, _ = maddpg.step(
                    torch_obs, 
                    start_stop_num, 
                    explore=False  # ← Deterministic policy!
                )
                agent_actions = np.column_stack(
                    [ac.data.numpy() for ac in torch_agent_actions]
                )
                obs, rewards, _, _, _ = env.step(agent_actions)
                ep_reward += np.mean(rewards)
                
                # 4. Collect states for GIF (last episode only)
                if is_last_ep:
                    state_history.append(env._states)
            
            # 5. Accumulate metrics
            total_reward += ep_reward / cfg.episode_length
            total_coverage += env.coverage_rate()
            total_uniformity += env.distribution_uniformity()
            
            # 6. Render GIF (last episode only)
            if is_last_ep:
                gif_path = Path(cfg.gif_dir) / f"eval_{ep_index}.gif"
                save_eval_gif(state_history, gif_path)
    
    # 7. Average and log
    mean_reward = total_reward / cfg.eval_episodes
    mean_coverage = total_coverage / cfg.eval_episodes
    mean_uniformity = total_uniformity / cfg.eval_episodes
    
    print(
        f"[EVAL] ep {ep_index} | reward: {mean_reward:.4f} | "
        f"coverage: {mean_coverage:.4f} | uniformity: {mean_uniformity:.4f}"
    )
    logger.add_scalars(
        "eval",
        {
            "reward": mean_reward,
            "coverage": mean_coverage,
            "uniformity": mean_uniformity,
        },
        ep_index,
    )
    
    # 8. Restore training mode
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
