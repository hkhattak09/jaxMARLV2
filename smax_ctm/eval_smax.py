import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import optax
import numpy as np

import sys
import os
# Inject repo root into sys.path so 'jaxmarl' is always found regardless of CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# We import the exact same environment setup as in training
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from jaxmarl.wrappers.baselines import SMAXLogWrapper
# Note: we don't necessarily need the WorldState wrapper for evaluation if we just run random/heuristic,
# but if we run the trained actor, we only need the actor observations anyway.

import os

def _get_avail_actions(env, env_state):
    """Return available actions for either a single state or a batched state."""
    try:
        return env.get_avail_actions(env_state)
    except (TypeError, ValueError):
        return jax.vmap(env.get_avail_actions)(env_state)


def _build_render_actions(raw_state, all_agents, default_action):
    """Build renderer action dict for all agents from raw SMAX state when available."""
    prev_actions = getattr(raw_state, "prev_attack_actions", None)
    if prev_actions is None:
        return {agent: default_action for agent in all_agents}
    return {agent: int(prev_actions[i]) for i, agent in enumerate(all_agents)}


def evaluate_and_render_random_episode(map_name="3m", seed=42, save_name="smax_random_eval.gif"):
    """
    Runs a single episode using random actions and saves it to a GIF.
    In the future, we will swap the random action sampling with the trained Actor network.
    """
    # Ensure visualisations directory exists
    os.makedirs("./visualisations", exist_ok=True)
    save_path = os.path.join("./visualisations", save_name)
    
    print(f"Initializing {map_name} SMAX environment...")
    scenario = map_name_to_scenario(map_name)
    env = HeuristicEnemySMAX(scenario=scenario, attack_mode="closest", walls_cause_death=True, see_enemy_actions=True)
    env = SMAXLogWrapper(env)
    
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    
    # Reset
    obs, state = jax.jit(env.reset)(reset_rng)
    
    # We collect renderer-ready tuples: (key, raw_smax_state, actions).
    # The SMAX renderer expects actions for all agents (allies + enemies).
    stop_action = int(env.num_movement_actions - 1)
    all_render_agents = list(env._env.all_agents)
    init_raw_state = state.env_state.state
    init_actions = _build_render_actions(init_raw_state, all_render_agents, stop_action)
    render_seq = [(None, init_raw_state, init_actions)]
    done = False
    step = 0
    
    print("Simulating episode...")
    while not done:
        rng, action_rng = jax.random.split(rng)
        
        # In a real eval script, this would be: 
        # actions = actor_network.apply(...)
        # For now, we just sample random valid actions to demonstrate rendering:
        avail_actions = _get_avail_actions(env, state.env_state)
        # Random choice among valid actions
        # For a single env, avail_actions is (num_allies, num_actions)
        # We sample an action for each agent
        def sample_valid_action(avail, key):
            # Give tiny probability to unavailable actions so we avoid NaNs
            probs = avail + 1e-8
            probs = probs / probs.sum()
            return jax.random.categorical(key, jnp.log(probs))
            
        keys = jax.random.split(action_rng, env.num_agents)
        if isinstance(avail_actions, dict):
            actions = {
                agent: sample_valid_action(avail_actions[agent], keys[i])
                for i, agent in enumerate(env.agents)
            }
        else:
            actions = {
                agent: sample_valid_action(avail_actions[i], keys[i])
                for i, agent in enumerate(env.agents)
            }
        
        # Step
        rng, step_rng = jax.random.split(rng)
        obs, state, rewards, dones, infos = jax.jit(env.step)(step_rng, state, actions)
        
        raw_state = state.env_state.state
        render_actions = _build_render_actions(raw_state, all_render_agents, stop_action)
        render_seq.append((None, raw_state, render_actions))
        done = dones["__all__"]
        step += 1
        if step > env.max_steps:
            break
            
    print(f"Episode finished in {step} steps. Rendering GIF...")
    
    # Rendering
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Initialize render with first state
    # We pass the underlying env_state since the SMAX wrapper returns LogEnvState
    render_obj = env.init_render(ax, render_seq[0], 0, 0)
    
    def animate(i):
        # Update render
        env.update_render(render_obj, render_seq[i], i, i)
        return ax
        
    anim = FuncAnimation(fig, animate, frames=len(render_seq), interval=100)
    anim.save(save_path, dpi=80, writer='pillow')
    plt.close(fig)
    print(f"Saved recording to {save_path}")

if __name__ == "__main__":
    evaluate_and_render_random_episode()
