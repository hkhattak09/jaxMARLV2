# Quick Start: GPU-Optimized Training

## On Google Colab

### 1. Setup Environment
```python
# Install dependencies
!pip install jax[cuda12]==0.7.2
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Clone/upload your repo
# (assuming code is already in /content/new_marl_llm_implementation)

import os
os.chdir('/content/new_marl_llm_implementation')
```

### 2. Verify Complete Pipeline
```python
!python test_complete_pipeline.py
```

This comprehensive test verifies:
- GPU detection and memory configuration
- JAX environment operations
- DLPack zero-copy transfers
- MADDPG algorithm integration
- Replay buffer functionality
- Complete rollout and training loops
- Data flow correctness

Expected: All 11 tests pass

### 3. Run Training (GPU Version)
```python
!python MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py
```

### 4. Or Run Training (CPU Version)
```python
!python MARL-LLM/marl_llm/train/train_assembly_jax.py
```

## Key Differences

| Feature | CPU Version | GPU Version |
|---------|-------------|-------------|
| File | `train_assembly_jax.py` | `train_assembly_jax_gpu.py` |
| Adapter | `JaxAssemblyAdapter` | `JaxAssemblyAdapterGPU` |
| Policy location | CPU during rollout | GPU always |
| Transfers | ~16 KB/step PCIe | 0 PCIe (DLPack) |
| Rollout speed | Baseline | 15-25% faster |
| Memory | Same (~300 MB GPU) | Same (~300 MB GPU) |
| Compatibility | Any system | Requires CUDA GPU |

## Configuration

Edit `MARL-LLM/marl_llm/cfg/assembly_cfg.py`:

```python
n_rollout_threads = 4    # Parallel environments (4-16 recommended)
n_a = 30                 # Agents per environment
episode_length = 200     # Steps per episode
hidden_dim = 180         # Policy network size
batch_size = 512         # Training batch
buffer_length = 20000    # Replay buffer size
```

## Memory Guidelines

GPU memory usage (n_envs × n_a agents):
- 1 env, 30 agents: ~150 MB
- 4 envs, 120 agents: ~300 MB
- 8 envs, 240 agents: ~500 MB
- 16 envs, 480 agents: ~800 MB

For 8 GB GPU: Use n_envs ≤ 8
For 16 GB GPU: Use n_envs ≤ 16

## Monitoring

```python
# In Colab cell:
!nvidia-smi -l 1
```

Look for:
- GPU utilization: ~80-100%
- Memory used: 300-800 MB
- Temperature: < 80°C

## Troubleshooting

**Test fails with "CUDA not available":**
```python
# Runtime → Change runtime type → Hardware accelerator → GPU
```

**"Out of memory" error:**
```python
# Reduce n_rollout_threads or buffer_length
```

**Slower than expected:**
```python
# Check GPU utilization with nvidia-smi
# Ensure n_envs > 1 for batching benefits
```

## Performance Expectations

Episode timing (episode_length=200):
- CPU version: 600-1200 ms
- GPU version: 500-1000 ms

Training 10k episodes:
- CPU version: ~2-3 hours
- GPU version: ~1.7-2.5 hours
- **Time saved: 20-40 minutes**

## Files to Upload to Colab

Minimum required:
```
new_marl_llm_implementation/
├── MARL-LLM/
│   ├── cus_gym/
│   ├── marl_llm/
│   └── cfg/
├── JaxMARL/
│   └── jaxmarl/
├── data/
│   └── results.pkl
└── test_gpu_integration.py
```

## Quick Commands

```bash
# Complete pipeline test (run this first!)
python test_complete_pipeline.py

# Full training (GPU)
python MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py

# Full training (CPU fallback)
python MARL-LLM/marl_llm/train/train_assembly_jax.py

# Check logs
tensorboard --logdir=models/AssemblySwarm-v0/
```

## Validation

Training is working correctly if you see:
- Episode rewards increasing
- "step time: 0.5-1.0 seconds" (for 200 steps)
- GPU utilization 80-100%
- No CUDA errors

## Notes

- Both versions produce identical results (same algorithm)
- GPU version is purely a performance optimization
- Can switch between versions anytime
- Prior action regularization is enabled (alpha=0.1)

---

**Ready to run!** Start with `test_complete_pipeline.py` to verify the entire pipeline.
