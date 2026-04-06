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
_D_SEN         = 0.4
_R_AVOID       = 0.10


def render_frame(p_pos_np, grid_center_np, valid_mask_np, l_cell, step,
                 size_a=_SIZE_A, d_sen=_D_SEN, r_avoid=_R_AVOID):
    """Render one timestep to a 480×480 RGB numpy array.

    Args:
        p_pos_np:      (n_a, 2)   float32 numpy — agent positions.
        grid_center_np:(2, n_g)   float32 numpy — grid cell centres.
        valid_mask_np: (n_g,)     bool numpy    — True = real cell.
        l_cell:        float      — grid cell side-length.
        step:          int        — step index shown in title.
        size_a:        float      — agent physical radius.
        d_sen:         float      — sensing radius (drawn as thin ring).
        r_avoid:       float      — avoidance radius (drawn as dashed ring).

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

    # Determine which valid cells are sensed by at least one agent (within d_sen)
    # p_pos_np: [n_a, 2], valid_cells: [n_valid, 2]
    diffs = valid_cells[None, :, :] - p_pos_np[:, None, :]   # [n_a, n_valid, 2]
    dists = np.linalg.norm(diffs, axis=-1)                    # [n_a, n_valid]
    sensed_mask = np.any(dists < d_sen, axis=0)               # [n_valid]

    for k, (cx, cy) in enumerate(valid_cells):
        if sensed_mask[k]:
            # Sensed cell — warm orange tint so it stands out from the shape
            ax.add_patch(mpatches.Rectangle(
                (cx - l_cell / 2, cy - l_cell / 2), l_cell, l_cell,
                linewidth=0, facecolor='#ff9800', alpha=0.55,
            ))
        else:
            ax.add_patch(mpatches.Rectangle(
                (cx - l_cell / 2, cy - l_cell / 2), l_cell, l_cell,
                linewidth=0, facecolor='#4caf50', alpha=0.4,
            ))

    # Agents: sensing ring, avoidance ring, body
    for x, y in p_pos_np:
        # Sensing distance — thin solid yellow ring
        ax.add_patch(mpatches.Circle((x, y), d_sen,
                                     linewidth=0.4, edgecolor='#ffeb3b',
                                     facecolor='none', alpha=0.25, zorder=2))
        # Avoidance radius — thin dashed red ring
        ax.add_patch(mpatches.Circle((x, y), r_avoid,
                                     linewidth=0.6, edgecolor='#ef5350',
                                     facecolor='none', alpha=0.4, zorder=2,
                                     linestyle='--'))
        # Agent body
        ax.add_patch(mpatches.Circle((x, y), size_a, color='#42a5f5', zorder=3))

    ax.set_title(f"Step {step}", color='white', fontsize=9, pad=3)
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    fig.patch.set_facecolor('#1a1a2e')
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    frame = buf[:, :, :3].copy()
    plt.close(fig)
    return frame


def save_eval_gif(state_history_jax, gif_path, fps=12, frame_skip=2,
                  size_a=_SIZE_A, d_sen=_D_SEN, r_avoid=_R_AVOID):
    """Bulk-transfer episode states from device to CPU, render frames, save GIF.

    Args:
        state_history_jax: list of AssemblyState (JAX arrays), one per step.
        gif_path:          str or Path — where to write the .gif file.
        fps:               int — frames per second for the output GIF.
        frame_skip:        int — render every Nth step (2 = half the frames).
        size_a:            float — agent physical radius (from env.size_a).
        d_sen:             float — sensing radius (from env.d_sen).
        r_avoid:           float — avoidance radius (from env.r_avoid).
    """
    import jax
    import jax.numpy as jnp

    gif_path = Path(gif_path)

    # ── Single bulk device→CPU transfer ───────────────────────────────────
    state_history_jax = state_history_jax[::frame_skip]
    p_pos_stacked = jnp.stack([s.p_pos for s in state_history_jax], axis=0)
    # With N>1 envs, p_pos_stacked is (T, N, n_a, 2) — render first env only
    if p_pos_stacked.ndim == 4:
        p_pos_stacked = p_pos_stacked[:, 0, :, :]
    p_pos_all = jax.device_get(p_pos_stacked)              # [T, n_a, 2] numpy

    s0 = state_history_jax[0]
    grid_center_np = jax.device_get(s0.grid_center)        # [2, n_g_max] or [N, 2, n_g_max]
    valid_mask_np  = jax.device_get(s0.valid_mask).astype(bool)  # [n_g_max] or [N, n_g_max]
    l_cell_raw     = jax.device_get(s0.l_cell)
    if grid_center_np.ndim == 3:   # batched: take first env
        grid_center_np = grid_center_np[0]
        valid_mask_np  = valid_mask_np[0]
        l_cell         = float(l_cell_raw.flat[0])
    else:
        l_cell = float(l_cell_raw)
    # ──────────────────────────────────────────────────────────────────────

    frames = [
        render_frame(p_pos_np, grid_center_np, valid_mask_np, l_cell, t,
                     size_a=size_a, d_sen=d_sen, r_avoid=r_avoid)
        for t, p_pos_np in enumerate(p_pos_all)
    ]

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
    print(f"[EVAL] GIF saved → {gif_path}  ({len(frames)} frames @ {fps}fps)")
