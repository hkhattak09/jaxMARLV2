"""Test suite for eval GIF rendering pipeline.

Tests the render utilities in MARL-LLM/marl_llm/train/eval_render.py without
requiring a trained model or a full training run.

Tests
-----
1. render_frame produces a correctly shaped and typed RGB array
2. save_eval_gif writes a readable .gif file given real JAX states
3. End-to-end: build a real AssemblyEnv + MADDPG network (init_from_env),
   run a full 200-step episode, save a GIF, read it back and verify
   frame count + dimensions

Run on Colab (GPU or CPU):
    python test_eval_render.py
"""

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'

import sys
from pathlib import Path
import numpy as np

_REPO_ROOT   = Path(__file__).resolve().parent
_JAXMARL_PATH = _REPO_ROOT / "JaxMARL"
_CUS_GYM_PATH = _REPO_ROOT / "MARL-LLM" / "cus_gym"
_MARL_LLM_PATH = _REPO_ROOT / "MARL-LLM" / "marl_llm"

for p in [str(_JAXMARL_PATH), str(_CUS_GYM_PATH), str(_MARL_LLM_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

print("=" * 60)
print("EVAL RENDER PIPELINE TEST")
print("=" * 60)

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


# ─── Test 1: render_frame output shape and dtype ──────────────────────────
def test_render_frame_shape():
    from train.eval_render import render_frame

    n_a    = 10
    n_g    = 20
    l_cell = 0.2

    p_pos_np       = np.random.uniform(-2.0, 2.0, (n_a, 2)).astype(np.float32)
    grid_center_np = np.random.uniform(-2.0, 2.0, (2, n_g)).astype(np.float32)
    valid_mask_np  = np.ones(n_g, dtype=bool)
    valid_mask_np[n_g // 2:] = False  # half invalid

    frame = render_frame(p_pos_np, grid_center_np, valid_mask_np, l_cell, step=42)

    assert frame.ndim == 3,                 f"Expected 3-D array, got shape {frame.shape}"
    assert frame.shape[2] == 3,             f"Expected 3 colour channels, got {frame.shape[2]}"
    assert frame.shape[0] == frame.shape[1], "Expected square frame"
    assert frame.dtype == np.uint8,         f"Expected uint8, got {frame.dtype}"
    # 480×480 at 100 dpi from a 4.8-inch figure
    assert frame.shape[0] == 480,           f"Expected 480px height, got {frame.shape[0]}"
    return frame.shape


def run_test(name, fn):
    try:
        result = fn()
        print(f"{PASS} {name}  →  {result}")
        results.append((name, True))
    except Exception as e:
        print(f"{FAIL} {name}  →  {e}")
        results.append((name, False))


run_test("render_frame: shape and dtype", test_render_frame_shape)


# ─── Test 2: render_frame with empty valid mask (edge case) ───────────────
def test_render_frame_no_valid_cells():
    from train.eval_render import render_frame

    p_pos_np       = np.zeros((5, 2), dtype=np.float32)
    grid_center_np = np.zeros((2, 10), dtype=np.float32)
    valid_mask_np  = np.zeros(10, dtype=bool)  # all invalid

    frame = render_frame(p_pos_np, grid_center_np, valid_mask_np, l_cell=0.1, step=0)
    assert frame.shape == (480, 480, 3)
    return "ok — no cells rendered without crash"

run_test("render_frame: all-invalid grid mask", test_render_frame_no_valid_cells)


# ─── Test 3: save_eval_gif with synthetic JAX states ──────────────────────
def test_save_eval_gif_synthetic():
    """Build minimal chex dataclass-compatible objects to feed save_eval_gif."""
    import jax
    import jax.numpy as jnp
    from train.eval_render import save_eval_gif

    n_a    = 8
    n_g    = 15
    T      = 5  # short episode for speed

    # Build synthetic state objects that match AssemblyState field access
    class FakeState:
        def __init__(self, p_pos, grid_center, valid_mask, l_cell):
            self.p_pos        = p_pos
            self.grid_center  = grid_center
            self.valid_mask   = valid_mask
            self.l_cell       = l_cell

    key = jax.random.PRNGKey(0)
    grid_center_jax = jax.random.uniform(key, (2, n_g), minval=-2.0, maxval=2.0)
    valid_mask_jax  = jnp.ones(n_g, dtype=bool)

    state_history = []
    for t in range(T):
        key, subkey = jax.random.split(key)
        p_pos_jax = jax.random.uniform(subkey, (n_a, 2), minval=-2.0, maxval=2.0)
        state_history.append(FakeState(p_pos_jax, grid_center_jax, valid_mask_jax, 0.15))

    with tempfile.TemporaryDirectory() as tmpdir:
        gif_path = Path(tmpdir) / "test_eval.gif"
        save_eval_gif(state_history, gif_path, fps=10)

        assert gif_path.exists(), "GIF file was not created"
        size_kb = gif_path.stat().st_size / 1024
        return f"created {gif_path.name}, {T} frames, {size_kb:.1f} KB"

run_test("save_eval_gif: synthetic JAX states", test_save_eval_gif_synthetic)


# ─── Test 4: end-to-end with real AssemblyEnv + MADDPG network ───────────
def test_end_to_end_real_env():
    import torch
    import jax
    from jaxmarl.environments.mpe.assembly import AssemblyEnv
    from gym.wrappers import JaxAssemblyAdapter
    from train.eval_render import save_eval_gif
    from algorithm.algorithms import MADDPG
    from cfg.assembly_cfg import gpsargs as cfg
    import imageio

    results_file = str(_REPO_ROOT / "fig" / "results.pkl")
    T = cfg.episode_length  # full 200-step episode

    jax_env = AssemblyEnv(results_file=results_file, n_a=cfg.n_a)
    env     = JaxAssemblyAdapter(jax_env, n_envs=1, seed=cfg.seed, alpha=1.0)

    # Initialise MADDPG with the same cfg used during training
    maddpg = MADDPG.init_from_env(
        env,
        agent_alg=cfg.agent_alg,
        adversary_alg=None,
        tau=cfg.tau,
        lr_actor=cfg.lr_actor,
        lr_critic=cfg.lr_critic,
        hidden_dim=cfg.hidden_dim,
        device=cfg.device,
        epsilon=cfg.epsilon,
        noise=cfg.noise_scale,
        name=cfg.env_name,
    )
    maddpg.prep_rollouts(device="cpu")

    start_stop_num = [slice(0, env.n_a)]
    obs = env.reset()
    state_history = []

    with torch.no_grad():
        for _ in range(T):
            torch_obs = torch.Tensor(obs).requires_grad_(False)
            torch_agent_actions, _ = maddpg.step(torch_obs, start_stop_num, explore=False)
            # DDPGAgent.step returns action.t() → shape (action_dim, n_agents) = (2, 30)
            # env.step expects (n_agents, action_dim) = (30, 2)
            actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])
            obs, _, _, _, _ = env.step(actions.T)
            state_history.append(env._states)

    # Save GIF in the same directory as this test file
    gif_path = _REPO_ROOT / "eval_real.gif"
    save_eval_gif(state_history, gif_path, fps=10)

    assert gif_path.exists(), "GIF file was not created"

    # Read back and verify frame count + dimensions
    reader  = imageio.get_reader(str(gif_path))
    frames  = list(reader)
    reader.close()

    expected_frames = len(state_history[::2])  # frame_skip=2 default
    assert len(frames) == expected_frames, f"Expected {expected_frames} frames, got {len(frames)}"
    h, w, c = frames[0].shape
    assert h == 480 and w == 480, f"Expected 480×480, got {h}×{w}"
    assert c == 3, f"Expected 3 channels, got {c}"

    size_kb = gif_path.stat().st_size / 1024
    return f"{T} frames, {h}×{w}px, {size_kb:.1f} KB — saved to {gif_path}"

run_test("end-to-end: real AssemblyEnv + MADDPG → GIF → readback", test_end_to_end_real_env)


# ─── Summary ──────────────────────────────────────────────────────────────
print()
print("=" * 60)
passed = sum(1 for _, ok in results if ok)
print(f"Results: {passed}/{len(results)} tests passed")
if passed == len(results):
    print("All render tests PASSED — eval GIF pipeline is working.")
else:
    failed = [name for name, ok in results if not ok]
    print("FAILED tests:")
    for name in failed:
        print(f"  - {name}")
print("=" * 60)
