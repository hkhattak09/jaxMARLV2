# Training Guide — Ablation Runs

## Setup (Colab)

```bash
cd /content/jaxMARLV2
TRAIN=MARL-LLM/marl_llm/train/train_assembly_jax_gpu.py
```

---

## Ablation Runs

Four runs that cover the paper's 2×2 ablation table (actor × critic temporal reasoning).

> The LSTM critic only does temporal reasoning when fed sequences (`update_sequence`, CTM path).
> On the MLP path (`update`), it sees single transitions — effectively memoryless.

### Run 1 — Baseline: MLP actor + prior regularization (best result so far)
```bash
python $TRAIN --use_mlp_actor --prior_mode regularize \
  --n_episodes 3000 --seed 226
```
Expected: Coverage ~0.916, Voronoi ~0.653

### Run 2 — MLP actor + LSTM critic (temporal critic, no temporal actor)
```bash
python $TRAIN --use_mlp_actor --prior_mode regularize \
  --n_episodes 3000 --seed 226
```
> Note: MLP path uses `update()` (single transitions), so LSTM critic
> is effectively memoryless here. This run is identical to Run 1 — use
> it only if you add a sequential MLP training path later.

### Run 3 — CTM actor (stateful, iter=1) + prior seed (no recurrent critic benefit)
```bash
python $TRAIN --prior_mode seed \
  --ctm_iterations 1 --lstm_hidden_dim 0 \
  --sequence_length 32 --burn_in_length 16 --num_sequences 16 \
  --n_episodes 3000 --seed 226
```
> Tests R-MADDPG's claim: recurrent actor alone doesn't help.
> If this matches or beats Run 1, that claim is falsified on this task.

### Run 4 — CTM actor (stateful) + prior seed + LSTM critic (full system)
```bash
python $TRAIN --prior_mode seed \
  --ctm_iterations 1 \
  --sequence_length 32 --burn_in_length 16 --num_sequences 16 \
  --lstm_hidden_dim 64 \
  --n_episodes 3000 --seed 226
```
Expected: Best result — both temporal components active.

---

## Key Flags

| Flag | Default | Notes |
|---|---|---|
| `--use_mlp_actor` | off (CTM) | Switch actor to MLP |
| `--prior_mode` | `none` | `none` / `regularize` / `seed` |
| `--ctm_iterations` | `1` | Ticks per env step; 1 = stateful |
| `--sequence_length` | `32` | Replay sequence length for update_sequence |
| `--burn_in_length` | `16` | Burn-in steps (no grad), rest gets grad |
| `--num_sequences` | `16` | Sequences per update call |
| `--lstm_hidden_dim` | `64` | Critic LSTM size; set `0` to disable |
| `--updates_per_episode` | `8` | Gradient updates per episode |
| `--n_episodes` | `3000` | Total training episodes |
| `--seed` | `226` | Random seed |

---

## Run Tests First

```bash
cd /content/jaxMARLV2
python MARL-LLM/marl_llm/tests/run_all_tests.py -s
```
All 165 tests should pass before any training run.

---

## Hypothesis and Falsification

| Run | Hypothesis | Falsified if |
|---|---|---|
| 3 vs 1 | Recurrent actor alone doesn't help | Run 3 > Run 1 by >0.02 Voronoi |
| 4 vs 3 | Recurrent critic adds value on top of recurrent actor | Run 4 ≈ Run 3 |
| 4 vs 1 | Full system beats MLP baseline | Run 4 ≤ Run 1 |
