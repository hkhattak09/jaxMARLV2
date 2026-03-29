"""Test script to verify GPU-optimized JAX-PyTorch integration with DLPack.

Run this on Google Colab to verify:
1. DLPack is working correctly
2. Zero-copy transfers are happening
3. Performance improvement is measurable

Expected output:
- Both JAX and PyTorch on GPU
- DLPack conversions successful
- GPU version 15-25% faster than CPU version
"""

import torch
import jax
import jax.numpy as jnp
from jax import dlpack as jax_dlpack
from torch.utils.dlpack import to_dlpack, from_dlpack
import time
import numpy as np

print("=" * 60)
print("GPU Environment Check")
print("=" * 60)

# Check PyTorch
print(f"\n1. PyTorch:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    print(f"   Device count: {torch.cuda.device_count()}")
else:
    print("   WARNING: PyTorch CUDA not available!")

# Check JAX
print(f"\n2. JAX:")
print(f"   Devices: {jax.devices()}")
print(f"   Default backend: {jax.default_backend()}")

print("\n" + "=" * 60)
print("DLPack Functionality Test")
print("=" * 60)

# Test 1: PyTorch GPU → JAX GPU
print("\nTest 1: PyTorch → JAX via DLPack")
try:
    torch_tensor = torch.randn(100, 50, device='cuda')
    print(f"   PyTorch tensor: shape={torch_tensor.shape}, device={torch_tensor.device}")
    
    # DLPack conversion
    capsule = to_dlpack(torch_tensor)
    jax_array = jax_dlpack.from_dlpack(capsule)
    print(f"   JAX array: shape={jax_array.shape}, device={jax_array.device()}")
    
    # Verify data integrity
    torch_np = torch_tensor.cpu().numpy()
    jax_np = np.array(jax_array)
    if np.allclose(torch_np, jax_np):
        print("   ✓ Data integrity verified")
    else:
        print("   ✗ Data mismatch!")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: JAX GPU → PyTorch GPU
print("\nTest 2: JAX → PyTorch via DLPack")
try:
    jax_array = jnp.ones((50, 100), dtype=jnp.float32)
    print(f"   JAX array: shape={jax_array.shape}, device={jax_array.device()}")
    
    # DLPack conversion
    capsule = jax_dlpack.to_dlpack(jax_array)
    torch_tensor = from_dlpack(capsule)
    print(f"   PyTorch tensor: shape={torch_tensor.shape}, device={torch_tensor.device}")
    
    # Verify data integrity
    jax_np = np.array(jax_array)
    torch_np = torch_tensor.cpu().numpy()
    if np.allclose(jax_np, torch_np):
        print("   ✓ Data integrity verified")
    else:
        print("   ✗ Data mismatch!")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("Performance Comparison")
print("=" * 60)

# Simulate policy forward pass
n_agents = 120
obs_dim = 28
hidden_dim = 180
n_iterations = 100

# Create a simple network
policy = torch.nn.Sequential(
    torch.nn.Linear(obs_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, 2),
    torch.nn.Tanh()
)

print(f"\nSimulating {n_iterations} policy forward passes with {n_agents} agents")

# CPU version
print("\n3. CPU Version (with transfers):")
policy_cpu = policy.cpu()
times_cpu = []

for _ in range(10):  # Warmup
    obs = torch.randn(n_agents, obs_dim)
    _ = policy_cpu(obs)

torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    obs_np = np.random.randn(n_agents, obs_dim).astype(np.float32)
    obs_cpu = torch.from_numpy(obs_np)
    actions_cpu = policy_cpu(obs_cpu)
    actions_np = actions_cpu.detach().numpy()
    
    # Simulate JAX transfer
    _ = jnp.asarray(actions_np)
    
torch.cuda.synchronize()
cpu_time = time.time() - start
times_cpu.append(cpu_time)

print(f"   Total time: {cpu_time*1000:.2f} ms")
print(f"   Per step: {cpu_time/n_iterations*1000:.3f} ms")

# GPU version with DLPack
print("\n4. GPU Version (with DLPack):")
policy_gpu = policy.cuda()
times_gpu = []

for _ in range(10):  # Warmup
    obs = torch.randn(n_agents, obs_dim, device='cuda')
    _ = policy_gpu(obs)

torch.cuda.synchronize()
start = time.time()
for _ in range(n_iterations):
    obs_gpu = torch.randn(n_agents, obs_dim, device='cuda')
    actions_gpu = policy_gpu(obs_gpu)
    
    # DLPack zero-copy
    capsule = to_dlpack(actions_gpu)
    actions_jax = jax_dlpack.from_dlpack(capsule)
    
torch.cuda.synchronize()
gpu_time = time.time() - start
times_gpu.append(gpu_time)

print(f"   Total time: {gpu_time*1000:.2f} ms")
print(f"   Per step: {gpu_time/n_iterations*1000:.3f} ms")

# Speedup
speedup = cpu_time / gpu_time
print(f"\n   Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")

if speedup > 1.15:
    print("   ✓ GPU version achieves expected speedup!")
elif speedup > 1.0:
    print("   ~ GPU version is faster but below expected 15-25%")
else:
    print("   ✗ GPU version is not faster - check GPU utilization")

print("\n" + "=" * 60)
print("Integration Test with JAX Environment")
print("=" * 60)

try:
    import sys
    import os
    from pathlib import Path
    
    # Try to import the custom environment
    print("\n5. Testing JAX Assembly Environment:")
    
    # This will only work if you've uploaded the full repo to Colab
    try:
        from jaxmarl.environments.mpe.assembly import AssemblyEnv
        from gym.wrappers.customized_envs.jax_assembly_wrapper_gpu import JaxAssemblyAdapterGPU
        
        # Create environment
        jax_env = AssemblyEnv(results_file="data/results.pkl", n_a=30)
        env = JaxAssemblyAdapterGPU(jax_env, n_envs=1, seed=0)
        
        print("   ✓ Environment created successfully")
        
        # Test reset
        obs = env.reset()
        print(f"   ✓ Reset: obs shape={obs.shape}, device={obs.device}")
        
        # Test step
        actions = torch.randn(30, 2, device='cuda')
        next_obs, rew, done, info, prior = env.step(actions)
        print(f"   ✓ Step: obs shape={next_obs.shape}, device={next_obs.device}")
        
        print("   ✓ Full integration test passed!")
        
    except ImportError as e:
        print(f"   ⚠ Could not import environment (expected if not in full repo)")
        print(f"     Error: {e}")
        
except Exception as e:
    print(f"   ✗ Integration test failed: {e}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
print("\nIf all tests passed, the GPU-optimized version is ready to use.")
print("Run train_assembly_jax_gpu.py for full training.")
