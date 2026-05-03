"""Compilation smoke test for train_mappo_t_role.py.

Run on Colab with:
    python test_scripts/test_train_role_compile.py
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SMAX_CTM = os.path.join(REPO_ROOT, "smax_ctm")
if SMAX_CTM not in sys.path:
    sys.path.insert(0, SMAX_CTM)

import jax
import jax.numpy as jnp
from train_mappo_t_role import make_train
from mappo_t import get_default_maca_role_config


def test_compile_experiment(exp_id: int):
    """Test that make_train compiles and runs one update for experiment exp_id."""
    cfg = get_default_maca_role_config()
    cfg["ROLE_EXPERIMENT"] = exp_id
    cfg["SEED"] = 0
    cfg["TOTAL_TIMESTEPS"] = 2000  # 1 update: 20 envs * 100 steps
    cfg["NUM_ENVS"] = 2
    cfg["NUM_STEPS"] = 100
    cfg["MAP_NAME"] = "protoss_10_vs_10"
    cfg["N_ROLES"] = 3
    cfg["USE_KL_DIVERSITY"] = True
    # Critic diversity only for role-specific critic experiments
    if exp_id in (2, 4):
        cfg["USE_CRITIC_DIVERSITY"] = True
    cfg["PPO_EPOCH"] = 1
    cfg["CRITIC_EPOCH"] = 1
    cfg["ACTOR_NUM_MINI_BATCH"] = 1
    cfg["CRITIC_NUM_MINI_BATCH"] = 1
    cfg["use_valuenorm"] = False
    cfg["ANNEAL_LR"] = False
    cfg["use_recurrent_policy"] = False
    cfg["use_feature_normalization"] = True
    cfg["hidden_sizes"] = [16, 16]
    cfg["transformer"]["n_embd"] = 16
    cfg["transformer"]["zs_dim"] = 32
    cfg["transformer"]["n_encode_layer"] = 1
    cfg["transformer"]["n_decode_layer"] = 0
    cfg["transformer"]["n_head"] = 2
    cfg["transformer"]["bias"] = True
    cfg["transformer"]["dropout"] = 0.0
    cfg["transformer"]["active_fn"] = "relu"
    cfg["transformer"]["q_value_loss_coef"] = 0.5
    cfg["transformer"]["eq_value_loss_coef"] = 0.5

    rng = jax.random.PRNGKey(0)
    train_fn = make_train(cfg)

    print(f"[Exp {exp_id}] Compiling...")
    t0 = time.time()
    out = jax.block_until_ready(train_fn(rng))
    t1 = time.time()
    print(f"[Exp {exp_id}] Compiled + ran in {t1 - t0:.2f}s")

    metrics = out["metrics"]
    assert "actor_loss" in metrics
    assert "value_loss" in metrics
    print(f"[Exp {exp_id}] actor_loss={float(metrics['actor_loss'][-1]):.4f}, value_loss={float(metrics['value_loss'][-1]):.4f}")
    print(f"[Exp {exp_id}] PASS")


if __name__ == "__main__":
    import time
    for exp in [1, 2, 3, 4]:
        test_compile_experiment(exp)
    print("\nAll 4 experiments compiled successfully!")
