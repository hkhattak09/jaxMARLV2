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
    
    # We will collect the states to render them later
    states_seq = [state]
    done = False
    step = 0
    
    print("Simulating episode...")
    while not done:
        rng, action_rng = jax.random.split(rng)
        
        # In a real eval script, this would be: 
        # actions = actor_network.apply(...)
        # For now, we just sample random valid actions to demonstrate rendering:
        avail_actions = jax.vmap(env.get_avail_actions)(state.env_state)
        # Random choice among valid actions
        # For a single env, avail_actions is (num_allies, num_actions)
        # We sample an action for each agent
        def sample_valid_action(avail, key):
            # Give tiny probability to unavailable actions so we avoid NaNs
            probs = avail + 1e-8
            probs = probs / probs.sum()
            return jax.random.categorical(key, jnp.log(probs))
            
        keys = jax.random.split(action_rng, env.num_agents)
        actions_list = [sample_valid_action(avail_actions[i], keys[i]) for i in range(env.num_agents)]
        
        actions = {agent: actions_list[i] for i, agent in enumerate(env.agents)}
        
        # Step
        rng, step_rng = jax.random.split(rng)
        obs, state, rewards, dones, infos = jax.jit(env.step)(step_rng, state, actions)
        
        states_seq.append(state)
        done = dones["__all__"]
        step += 1
        if step > env.max_steps:
            break
            
    print(f"Episode finished in {step} steps. Rendering GIF...")
    
    # Rendering
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Initialize render with first state
    # We pass the underlying env_state since the SMAX wrapper returns LogEnvState
    render_obj = env.init_render(ax, states_seq[0].env_state, 0, 0)
    
    def animate(i):
        # Update render
        env.update_render(render_obj, states_seq[i].env_state, i, i)
        return ax
        
    anim = FuncAnimation(fig, animate, frames=len(states_seq), interval=100)
    anim.save(save_path, dpi=80, writer='pillow')
    plt.close(fig)
    print(f"Saved recording to {save_path}")

if __name__ == "__main__":
    evaluate_and_render_random_episode()
