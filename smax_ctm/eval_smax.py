import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import pickle

import sys
import os
# Inject repo root into sys.path so 'jaxmarl' is always found regardless of CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# We import the exact same environment setup as in training
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from jaxmarl.wrappers.baselines import SMAXLogWrapper

MODEL_DIR = os.path.join(_REPO_ROOT, "model")
MODEL_FILE = "smax_mappo_gru_actor.pkl"

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


def _stack_agent_array(values_by_agent, agents):
    return jnp.stack([values_by_agent[a] for a in agents])


def _load_checkpoint():
    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Train and save the GRU model first."
        )
    with open(ckpt_path, "rb") as f:
        return pickle.load(f), ckpt_path


def _build_actor_for_eval(config, env):
    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    num_agents = env.num_agents

    from train_mappo_gru import ActorRNN, ScannedRNN

    actor_network = ActorRNN(action_dim, config=config)
    hidden = ScannedRNN.initialize_carry(num_agents, config["GRU_HIDDEN_DIM"])
    init_x = (
        jnp.zeros((1, num_agents, obs_dim)),
        jnp.zeros((1, num_agents)),
        jnp.zeros((1, num_agents, action_dim)),
    )
    first_done = jnp.zeros((num_agents,), dtype=bool)

    return actor_network, hidden, init_x, first_done


def evaluate_and_render_trained_episode(map_name=None, seed=42, save_name=None, stochastic=False):
    """
    Runs a single episode with a trained GRU actor and saves it to a GIF.
    """
    checkpoint, ckpt_path = _load_checkpoint()
    config = checkpoint["config"]
    actor_params = checkpoint["actor_params"]

    if map_name is None:
        map_name = config.get("MAP_NAME", "3m")
    if save_name is None:
        save_name = "smax_gru_eval.gif"

    # Ensure visualisations directory exists
    os.makedirs("./visualisations", exist_ok=True)
    save_path = os.path.join("./visualisations", save_name)

    print(f"Loaded GRU checkpoint: {ckpt_path}")
    print(f"Initializing {map_name} SMAX environment...")
    scenario = map_name_to_scenario(map_name)
    env_kwargs = config.get("ENV_KWARGS", {})
    env = HeuristicEnemySMAX(scenario=scenario, **env_kwargs)
    env = SMAXLogWrapper(env)

    actor_network, hidden, init_x, done_batch = _build_actor_for_eval(config, env)
    _ = actor_network.init(jax.random.PRNGKey(0), hidden, init_x)
    
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
    
    print("Simulating episode with trained policy...")
    while not done:
        rng, action_rng = jax.random.split(rng)

        avail_actions = _get_avail_actions(env, state.env_state)
        if isinstance(avail_actions, dict):
            avail_batch = _stack_agent_array(avail_actions, env.agents)
        else:
            avail_batch = avail_actions

        obs_batch = _stack_agent_array(obs, env.agents)
        ac_in = (obs_batch[jnp.newaxis, :], done_batch[jnp.newaxis, :], avail_batch[jnp.newaxis, :])
        hidden, pi = actor_network.apply(actor_params, hidden, ac_in)

        if stochastic:
            action_batch = pi.sample(seed=action_rng).squeeze(0)
        else:
            action_batch = pi.mode().squeeze(0)

        actions = {
            agent: action_batch[i]
            for i, agent in enumerate(env.agents)
        }
        
        # Step
        rng, step_rng = jax.random.split(rng)
        obs, state, rewards, dones, infos = jax.jit(env.step)(step_rng, state, actions)
        done_batch = jnp.array([dones[a] for a in env.agents], dtype=bool)
        
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
    
    # Initialize first frame.
    render_obj = env.init_render(ax, render_seq[0], 0, 0)
    
    def animate(i):
        # Redraw from the axis each frame; SMAX.update_render reuses an artist
        # that can be detached from its axes after clear(), causing None axes.
        nonlocal render_obj
        render_obj = env.init_render(ax, render_seq[i], i, i)
        return [render_obj]
        
    anim = FuncAnimation(fig, animate, frames=len(render_seq), interval=100)
    anim.save(save_path, dpi=80, writer='pillow')
    plt.close(fig)
    print(f"Saved recording to {save_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved SMAX GRU actor and render a GIF.")
    parser.add_argument("--map_name", type=str, default=None, help="SMAX map name. Defaults to map saved in checkpoint config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment/action sampling.")
    parser.add_argument("--save_name", type=str, default=None, help="GIF output name in ./visualisations.")
    parser.add_argument("--stochastic", action="store_true", help="Sample policy actions instead of greedy mode().")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_and_render_trained_episode(
        map_name=args.map_name,
        seed=args.seed,
        save_name=args.save_name,
        stochastic=args.stochastic,
    )
