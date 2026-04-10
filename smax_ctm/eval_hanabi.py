import argparse
import contextlib
import io
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Inject repo root into sys.path so 'jaxmarl' is always found regardless of CWD.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from jaxmarl.environments.hanabi import Hanabi as HanabiEnv


MODEL_DIR = os.path.join(_REPO_ROOT, "model")
MODEL_FILES = {
    "gru": "hanabi_mappo_gru_actor.pkl",
    "ctm": "hanabi_mappo_ctm_actor.pkl",
}


def _stack_agent_array(values_by_agent, agents):
    return jnp.stack([values_by_agent[a] for a in agents])


def _load_checkpoint(model_type, checkpoint_path=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = checkpoint_path or os.path.join(MODEL_DIR, MODEL_FILES[model_type])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            f"Train and save the {model_type.upper()} Hanabi model first."
        )
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)

    ckpt_env = checkpoint.get("env")
    if ckpt_env is not None and ckpt_env != "hanabi":
        raise ValueError(f"Checkpoint env is '{ckpt_env}', expected 'hanabi'.")

    return checkpoint, ckpt_path


def _build_actor_for_eval(model_type, config, env):
    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    num_agents = env.num_agents

    if model_type == "gru":
        from train_mappo_gru_hanabi import ActorRNN, ScannedRNN

        actor_network = ActorRNN(action_dim, config=config)
        hidden = ScannedRNN.initialize_carry(num_agents, config["GRU_HIDDEN_DIM"])
        init_x = (
            jnp.zeros((1, num_agents, obs_dim)),
            jnp.zeros((1, num_agents)),
            jnp.zeros((1, num_agents, action_dim)),
        )
        first_done = jnp.zeros((num_agents,), dtype=bool)
    elif model_type == "ctm":
        from train_mappo_ctm_hanabi import ActorCTM, CTMCell

        actor_network = ActorCTM(action_dim, config=config)
        hidden = CTMCell.initialize_carry(
            num_agents,
            config["CTM_D_MODEL"],
            config["CTM_MEMORY_LENGTH"],
        )
        init_x = (
            jnp.zeros((1, num_agents, obs_dim)),
            jnp.zeros((1, num_agents)),
            jnp.zeros((1, num_agents, action_dim)),
        )
        # Match CTM Hanabi training startup behavior.
        first_done = jnp.ones((num_agents,), dtype=bool)
    else:
        raise ValueError(f"Unsupported model_type={model_type}. Choose 'gru' or 'ctm'.")

    return actor_network, hidden, init_x, first_done


def _capture_render_text(env, state):
    if not hasattr(env, "render"):
        raise RuntimeError("Hanabi environment has no render() method in this repository.")

    buff = io.StringIO()
    with contextlib.redirect_stdout(buff):
        env.render(state)
    return buff.getvalue().strip()


def _make_frame_text(header, board_text):
    return f"{header}\n{'=' * 80}\n{board_text}" if board_text else header


def _save_trace(trace_lines, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(trace_lines))


def _save_text_gif(frames, save_path, fps):
    if not frames:
        raise ValueError("No frames to render.")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    text_artist = ax.text(
        0.01,
        0.99,
        frames[0],
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        wrap=True,
    )

    def animate(i):
        text_artist.set_text(frames[i])
        return [text_artist]

    anim = FuncAnimation(fig, animate, frames=len(frames), interval=int(1000 / fps))
    anim.save(save_path, dpi=100, writer="pillow")
    plt.close(fig)


def evaluate_and_visualize_hanabi(
    model_type,
    checkpoint_path=None,
    seed=42,
    max_steps=512,
    stochastic=False,
    save_trace_name=None,
    save_gif_name=None,
    fps=2,
):
    checkpoint, ckpt_path = _load_checkpoint(model_type, checkpoint_path=checkpoint_path)
    config = checkpoint["config"]
    actor_params = checkpoint["actor_params"]

    env_kwargs = config.get("ENV_KWARGS", {})
    num_agents = config.get("NUM_AGENTS", 2)
    env = HanabiEnv(num_agents=num_agents, **env_kwargs)

    actor_network, hidden, init_x, done_batch = _build_actor_for_eval(model_type, config, env)
    _ = actor_network.init(jax.random.PRNGKey(0), hidden, init_x)

    os.makedirs("./visualisations", exist_ok=True)
    if save_trace_name is None:
        save_trace_name = f"hanabi_{model_type}_eval_trace.txt"
    if save_gif_name is None:
        save_gif_name = f"hanabi_{model_type}_eval.gif"

    trace_path = os.path.join("./visualisations", save_trace_name)
    gif_path = os.path.join("./visualisations", save_gif_name)

    print(f"Loaded {model_type.upper()} checkpoint: {ckpt_path}")
    print(f"Using Hanabi config: num_agents={num_agents}, env_kwargs={env_kwargs}")

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    obs, state = jax.jit(env.reset)(reset_rng)

    trace_lines = []
    frames = []

    init_text = _capture_render_text(env, state)
    init_header = "t=0 | initial state"
    init_frame = _make_frame_text(init_header, init_text)
    trace_lines.append(init_frame)
    frames.append(init_frame)

    done = False
    step = 0
    cum_reward = 0.0

    print("Simulating Hanabi episode with trained policy...")
    while not done:
        if step >= max_steps:
            raise RuntimeError(
                f"Exceeded max_steps={max_steps} without terminal episode. "
                "Increase --max_steps if this is expected."
            )

        rng, action_rng = jax.random.split(rng)

        legal_moves = env.get_legal_moves(state)
        avail_batch = _stack_agent_array(legal_moves, env.agents)
        obs_batch = _stack_agent_array(obs, env.agents)

        ac_in = (
            obs_batch[jnp.newaxis, :],
            done_batch[jnp.newaxis, :],
            avail_batch[jnp.newaxis, :],
        )
        hidden, pi = actor_network.apply(actor_params, hidden, ac_in)

        if stochastic:
            action_batch = pi.sample(seed=action_rng).squeeze(0)
        else:
            action_batch = pi.mode().squeeze(0)

        current_player = int(jnp.argmax(state.cur_player_idx))
        chosen_action = int(action_batch[current_player])
        action_name = env.action_encoding[chosen_action]

        actions = {
            agent: jnp.asarray(action_batch[i], dtype=jnp.int32)
            for i, agent in enumerate(env.agents)
        }

        rng, step_rng = jax.random.split(rng)
        obs, state, rewards, dones, _ = jax.jit(env.step)(step_rng, state, actions)

        step_reward = float(rewards["__all__"])
        cum_reward += step_reward
        done = bool(dones["__all__"])
        done_batch = jnp.array([dones[a] for a in env.agents], dtype=bool)
        step += 1

        board_text = _capture_render_text(env, state)
        header = (
            f"t={step} | player={current_player} | action={chosen_action}:{action_name} "
            f"| reward={step_reward:.2f} | cum_reward={cum_reward:.2f}"
        )
        frame = _make_frame_text(header, board_text)
        trace_lines.append(frame)
        frames.append(frame)

    print(f"Episode complete in {step} steps. Final score={cum_reward:.2f}")

    _save_trace(trace_lines, trace_path)
    print(f"Saved textual trace to {trace_path}")

    _save_text_gif(frames, gif_path, fps=fps)
    print(f"Saved text-rendered GIF to {gif_path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved Hanabi GRU/CTM actor, record step-by-step traces, and render a GIF. "
            "Note: this repository's Hanabi renderer is text-based, so GIF frames are rendered text panels."
        )
    )
    parser.add_argument("--model_type", choices=["gru", "ctm"], required=True, help="Which saved model to evaluate.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional explicit checkpoint .pkl path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment/action sampling.")
    parser.add_argument("--max_steps", type=int, default=512, help="Fail-loud safety cap on episode length.")
    parser.add_argument("--save_trace_name", type=str, default=None, help="Trace output name inside ./visualisations.")
    parser.add_argument("--save_gif_name", type=str, default=None, help="GIF output name inside ./visualisations.")
    parser.add_argument("--fps", type=int, default=2, help="GIF frame rate.")
    parser.add_argument("--stochastic", action="store_true", help="Sample policy actions instead of greedy mode().")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_and_visualize_hanabi(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        max_steps=args.max_steps,
        stochastic=args.stochastic,
        save_trace_name=args.save_trace_name,
        save_gif_name=args.save_gif_name,
        fps=args.fps,
    )
