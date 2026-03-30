"""Eval GIF rendering utilities shared by training scripts.

Keeps all matplotlib/imageio logic in one place so both train_assembly_jax.py
and train_assembly_jax_gpu.py can import identical render behaviour.

Transfer strategy
-----------------
States are collected as JAX arrays (on GPU or CPU depending on the script).
After the episode is done, _save_eval_gif performs a single bulk device_get()
to move all timestep data to CPU numpy in one transfer, then renders on CPU.
Grid geometry (grid_center, valid_mask, l_cell) is constant per episode so it
is transferred only once from the first collected state.
"""

import matplotlib
matplotlib.use('Agg')  # headless backend — safe to call even if already set
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import imageio
from pathlib import Path


# AssemblyEnv defaults — used for rendering geometry
_BOUNDARY_HALF = 2.4
_SIZE_A        = 0.035


def render_frame(p_pos_np, grid_center_np, valid_mask_np, l_cell, step):
    """Render one timestep to a 480×480 RGB numpy array.

    Args:
        p_pos_np:      (n_a, 2)   float32 numpy — agent positions.
        grid_center_np:(2, n_g)   float32 numpy — grid cell centres.
        valid_mask_np: (n_g,)     bool numpy    — True = real cell.
        l_cell:        float      — grid cell side-length.
        step:          int        — step index shown in title.

    Returns:
        (480, 480, 3) uint8 numpy RGB array.
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=dpi)

    ax.set_xlim([-_BOUNDARY_HALF, _BOUNDARY_HALF])
    ax.set_ylim([-_BOUNDARY_HALF, _BOUNDARY_HALF])
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')

    # Boundary box
    ax.add_patch(mpatches.Rectangle(
        (-_BOUNDARY_HALF, -_BOUNDARY_HALF),
        2 * _BOUNDARY_HALF, 2 * _BOUNDARY_HALF,
        linewidth=1.5, edgecolor='white', facecolor='none',
    ))

    # Valid grid cells — grid_center_np is [2, n_g], valid_mask_np is [n_g]
    valid_cells = grid_center_np[:, valid_mask_np].T  # [n_valid, 2]
    for cx, cy in valid_cells:
        ax.add_patch(mpatches.Rectangle(
            (cx - l_cell / 2, cy - l_cell / 2), l_cell, l_cell,
            linewidth=0, facecolor='#4caf50', alpha=0.4,
        ))

    # Agents
    for x, y in p_pos_np:
        ax.add_patch(mpatches.Circle((x, y), _SIZE_A, color='#42a5f5', zorder=3))

    ax.set_title(f"Step {step}", color='white', fontsize=9, pad=3)
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    fig.patch.set_facecolor('#1a1a2e')
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3).copy()
    plt.close(fig)
    return frame


def save_eval_gif(state_history_jax, gif_path, fps=12, frame_skip=2):
    """Bulk-transfer episode states from device to CPU, render frames, save GIF.

    Args:
        state_history_jax: list of AssemblyState (JAX arrays), one per step.
        gif_path:          str or Path — where to write the .gif file.
        fps:               int — frames per second for the output GIF.
        frame_skip:        int — render every Nth step (2 = half the frames).
    """
    import jax
    import jax.numpy as jnp

    gif_path = Path(gif_path)

    # ── Single bulk device→CPU transfer ───────────────────────────────────
    state_history_jax = state_history_jax[::frame_skip]
    p_pos_stacked = jnp.stack([s.p_pos for s in state_history_jax], axis=0)
    p_pos_all = jax.device_get(p_pos_stacked)              # [T, n_a, 2] numpy

    s0 = state_history_jax[0]
    grid_center_np = jax.device_get(s0.grid_center)        # [2, n_g_max] numpy
    valid_mask_np  = jax.device_get(s0.valid_mask).astype(bool)  # [n_g_max]
    l_cell         = float(jax.device_get(s0.l_cell))
    # ──────────────────────────────────────────────────────────────────────

    frames = [
        render_frame(p_pos_np, grid_center_np, valid_mask_np, l_cell, t)
        for t, p_pos_np in enumerate(p_pos_all)
    ]

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
    print(f"[EVAL] GIF saved → {gif_path}  ({len(frames)} frames @ {fps}fps)")
