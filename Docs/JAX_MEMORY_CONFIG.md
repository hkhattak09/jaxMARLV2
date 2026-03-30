# JAX Memory Configuration

## Problem
JAX preallocates ~90% of GPU memory by default, even though it only runs lightweight environment simulations (~20-100MB). PyTorch needs most of the GPU memory for neural networks and training batches.

## Solution
Limit JAX to **15% of GPU memory** before importing JAX. On a 14GB T4 GPU:
- **JAX gets**: ~2.1GB (plenty for environments)
- **PyTorch gets**: ~11.9GB (for networks and training)

## Implementation
Added to the top of both training scripts before any imports:

```python
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'
```

## Files Modified
1. `MARL-LLM/marl_llm/train/train_assembly_jax.py` (lines 20-25)
2. `MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py` (lines 21-26)

## Testing
Run the memory test to verify configuration:
```bash
python test_jax_memory_limit.py
```

Expected output:
- JAX limited to 15% of GPU
- PyTorch has ~11.9GB available
- Both can allocate without conflicts

## Why This Works
- **NVIDIA Unified Memory**: T4 GPU has automatic memory management that prevents fragmentation
- **Fixed Preallocation**: JAX gets its 15% upfront, PyTorch gets the rest - no fighting
- **Right-sized**: 2.1GB is more than enough for JAX environments (only need ~100MB)

## Adjusting Memory Fraction
If you change the number of parallel environments:
- **4-8 envs**: 0.10-0.15 (10-15%)
- **16+ envs**: 0.20 (20%)

Edit the environment variable value in the training scripts:
```python
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.20'  # for more envs
```

## Benefits
- **No OOM errors**: Predictable memory allocation
- **Full PyTorch capacity**: 85% of GPU available for networks
- **No fragmentation**: NVIDIA driver manages virtual memory automatically
- **Clean solution**: One line of configuration, no code changes needed
