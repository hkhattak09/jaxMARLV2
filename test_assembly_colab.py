"""
Colab Test Script for Assembly Environment
Run this in Google Colab to test the new JAX assembly environment.
"""

# Install JAX (if needed in Colab)
# !pip install --upgrade jax jaxlib flax chex

import sys
import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("JAX backend:", jax.default_backend())
print()

# Add JaxMARL to path - make sure jaxmarl is importable
import os
import glob as _glob

def _find_jaxmarl_root(search_root):
    """Find the directory that contains jaxmarl/__init__.py"""
    hits = _glob.glob(os.path.join(search_root, '**/jaxmarl/__init__.py'), recursive=True)
    if hits:
        # Return the parent of the jaxmarl package dir
        return os.path.dirname(os.path.dirname(hits[0]))
    return None

repo_root = '/content/jaxMARL'
found = _find_jaxmarl_root(repo_root)
if found:
    sys.path.insert(0, found)
    print(f"Added {found} to path")
else:
    # fallback: add repo root and hope for the best
    sys.path.insert(0, repo_root)
    print(f"Warning: could not find jaxmarl package, added {repo_root} to path")
    print(f"Contents of {repo_root}: {os.listdir(repo_root) if os.path.exists(repo_root) else 'NOT FOUND'}")

print(f"Python path: {sys.path[:2]}")

from jaxmarl.environments.assembly import AssemblyEnv

print("="*60)
print("Testing Assembly Environment")
print("="*60)

# Test 1: Environment creation
print("\n1. Creating environment...")
env = AssemblyEnv(num_agents=30, target_shape="circle", num_grid_cells=30)
print(f"✓ Environment created")
print(f"  - Num agents: {env.num_agents}")
print(f"  - Action space: {env.action_spaces['agent_0']}")
print(f"  - Obs space: {env.observation_spaces['agent_0']}")

# Test 2: Reset
print("\n2. Testing reset...")
key = jax.random.PRNGKey(42)
obs, state = env.reset(key)
print(f"✓ Reset successful")
print(f"  - Obs shape: {obs['agent_0'].shape}")
print(f"  - State positions: {state.p_pos.shape}")
print(f"  - Grid centers: {state.grid_centers.shape}")
print(f"  - Grid cell size: {state.l_cell}")

# Test 3: Single step
print("\n3. Testing single step...")
key, step_key = jax.random.split(key)
actions = {agent: jax.random.uniform(jax.random.PRNGKey(i), (2,), minval=-1, maxval=1) 
           for i, agent in enumerate(env.agents)}
obs, state, rewards, dones, info = env.step_env(step_key, state, actions)
print(f"✓ Step successful")
print(f"  - Rewards (first 5): {[rewards[f'agent_{i}'] for i in range(5)]}")
print(f"  - Episode done: {dones['__all__']}")

# Test 4: JIT compilation
print("\n4. Testing JIT compilation...")
@jax.jit
def run_episode(key):
    obs, state = env.reset(key)
    
    def step_fn(carry, _):
        state, key = carry
        key, action_key = jax.random.split(key)
        actions = {agent: jax.random.uniform(jax.random.PRNGKey(i), (2,), minval=-1, maxval=1) 
                   for i, agent in enumerate(env.agents)}
        obs, state, rewards, dones, info = env.step_env(key, state, actions)
        return (state, key), rewards
    
    (final_state, _), all_rewards = jax.lax.scan(step_fn, (state, key), None, length=10)
    return final_state, all_rewards

# Compile
print("  - Compiling...")
final_state, all_rewards = run_episode(key)
print("✓ JIT compilation successful")
print(f"  - Final state step: {final_state.step}")

# Test 5: Vectorization with vmap
print("\n5. Testing vectorization (parallel episodes)...")
@jax.jit
def run_parallel_episodes(keys):
    return jax.vmap(run_episode)(keys)

keys = jax.random.split(key, 4)
print("  - Running 4 parallel episodes...")
final_states, all_rewards = run_parallel_episodes(keys)
print(f"✓ Vectorization successful")
print(f"  - Final states shape: {final_states.p_pos.shape}")
print(f"  - All rewards shape: {all_rewards['agent_0'].shape}")

# Test 6: Different target shapes
print("\n6. Testing different target shapes...")
for shape in ["circle", "line", "square"]:
    env_test = AssemblyEnv(num_agents=20, target_shape=shape)
    obs, state = env_test.reset(jax.random.PRNGKey(0))
    print(f"✓ Shape '{shape}': {state.grid_centers.shape}")

print("\n" + "="*60)
print("All tests passed! 🎉")
print("="*60)
print("\nNext steps:")
print("1. This environment is ready to use")
print("2. Can be used for parallel data collection with vmap")
print("3. Needs validation against original Gym environment")
