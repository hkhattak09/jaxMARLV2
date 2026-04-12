# Iterative Neural Consensus (INC) — Implementation Plan

**Paper target:** *Iterative Neural Consensus: Coordinating Multi-Agent Policies Through Shared Internal Dynamics*

**Core mechanism:** Between CTM internal iterations (inside a single env step), each agent's next-iteration synapses are conditioned on its own state **plus a pooled summary of the other agents' sync vectors from the previous iteration**. Consensus happens within a single env step over `CTM_ITERATIONS` internal rounds — zero inter-step communication bandwidth.

**Design decisions already locked:**
- **Centralised pooling** across all teammates (Option A). Decentralised-at-test is a later ablation only if reviewers demand it.
- **Pool the full 528-dim sync vector** (Option X). Matches the Stage 1-3 analysis variable exactly, so motivation and method operate on the same signal.
- **Primary benchmark: Hanabi** (via JaxMARL upstream). Secondary: SMAX 2s3z, 5m_vs_6m.
- **Motivation section of paper** = existing Stage 1-3 sync analysis results.

---

## Stage 0 — Pre-flight checks

**Goal:** confirm assumptions and baseline numbers before touching the architecture.

- [x] Re-run SMAX 3m CTM-MAPPO baseline with current `CTM_ITERATIONS=1` to confirm reproducibility of the ~84% WR number. Save the seed, wall-clock, and checkpoint as the canonical baseline.
- [x] Verify that `CTMCell.__call__` in [ctm_jax.py](smax_ctm/ctm_jax.py) handles the current batch layout: inside `ScannedCTM`, the batch dim is `NUM_ACTORS = num_agents * num_envs`. Confirm the agent-axis ordering by running a small debug print that reshapes `(NUM_ACTORS, ...)` → `(num_envs, num_agents, ...)` and checks per-agent identity.
- [x] Read `jaxmarl/environments/hanabi/` from upstream FLAIROx/JaxMARL. Note its obs dim, action dim, `num_agents`, reward scaling, episode length, and world-state getter. Save a short note in `docs/hanabi_env_notes.md`.
- [x] Identify the upstream MAPPO/IPPO Hanabi baseline script. Note the GRU hidden size, entropy coefficient, and episode return range used there — these are the numbers reviewers will check you against.
- [x] **Axis convention — critical.** Read `batchify` in [train_mappo_ctm.py](smax_ctm/train_mappo_ctm.py) carefully. It stacks per-agent tensors along a new leading axis (`jnp.stack([x[a] for a in agents])` → `(num_agents, num_envs, F)`) and then reshapes to `(num_agents * num_envs, F)`. Consequence: the flat `NUM_ACTORS` axis is **agent-major**, i.e. the first `num_envs` rows are agent 0, the next `num_envs` rows are agent 1, and so on. The correct reshape inside `CTMCell` is therefore:
  ```python
  # (num_actors, synch) -> (num_agents, num_envs, synch)
  synch_per_agent = synch.reshape(num_agents, num_envs, synch_size)
  # pool across axis=0 (the agent axis)
  ```
  **Do NOT write `reshape(num_envs, num_agents, ...)`** — that silently mixes different agents into the same row and will train without erroring. Add an assert in Stage 1 that identifies a deliberately tagged per-agent value survives the round-trip reshape.
- [x] Write a one-shot debug script `smax_ctm/scripts/check_batchify_order.py` that feeds `batchify` a dict `{a0: ones*0, a1: ones*1, a2: ones*2}` and prints the flat result — confirms agent-major layout before we rely on it.

**Stage 0 evidence (2026-04-10):**
- `check_batchify_order.py` output matches expected agent-major ordering and passes.
- `check_ctm_axis_layout.py` output confirms `(num_agents, num_envs, ...)` is correct and `(num_envs, num_agents, ...)` is invalid for INC pooling.
- Notes written to `docs/inc_axis_convention.md` and `docs/hanabi_env_notes.md`.

**Exit criterion:** baseline reproduces and the agent/env axis convention is documented in `docs/inc_axis_convention.md`.

---

## Stage 1 — Refactor CTMCell to expose per-iteration sync

**Goal:** make the iteration loop iterate-by-iterate instead of fused, and emit a per-iteration sync vector that the outer code can pool across agents. No behavioural change yet — this stage should give identical numbers to Stage 0 when `num_consensus_iterations == 0`.

**Files touched:** [ctm_jax.py](smax_ctm/ctm_jax.py), [train_mappo_ctm.py](smax_ctm/train_mappo_ctm.py) (SMAX), [train_mappo_ctm_hanabi.py](smax_ctm/train_mappo_ctm_hanabi.py) (Hanabi). The CTMCell refactor is shared; only the outer-loop hook-in happens per-benchmark script.

**Relationship to `ScannedCTM` / `nn.scan`:** `ScannedCTM` uses `nn.scan` to iterate `CTMCell` over the **time axis** of a rollout/minibatch. The CTM's internal `CTM_ITERATIONS` loop lives **inside** each scan step — it is unrolled at compile time, not scanned. The INC refactor only touches this inner unrolled loop. `nn.scan`'s carry stays exactly as today: `(state_trace, activated_state_trace)`. Consensus is *not* carried across time — it is recomputed fresh every time step, consistent with the "zero inter-step bandwidth" framing.

- [x] In `CTMCell.__call__`, replace the `for _ in range(self.iterations)` body with a helper `_single_iter(state_trace, activated_state_trace, features, consensus_in)` that returns `(new_state_trace, new_activated_state_trace, synch)`. Signature:
  ```python
  def _single_iter(
      self,
      state_trace: jnp.ndarray,           # (B, D, M)
      activated_state_trace: jnp.ndarray,  # (B, D, M)
      features: jnp.ndarray,               # (B, F)
      consensus_in: Optional[jnp.ndarray], # (B, synch_size) or None
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
      # returns (new_state_trace, new_activated_state_trace, synch)
  ```
  Keep the helper as a regular Python method (not `@nn.compact`) and call the existing `Synapses` / `NLMs` / `compute_synchronisation` submodules from it — Flax will register them once at module init time.
- [x] The helper must compute the per-iteration sync vector using the existing `compute_synchronisation` against the current `activated_state_trace`, not just the final one. The `decay_params_out` parameter stays shared across iterations.
- [x] Add a new argument `consensus_in` (shape `(batch, synch_size)` or `None`) to `_single_iter`. When non-None, it is concatenated with `[features, last_activated]` before the synapses: `pre_synapse = concat([features, last_activated, consensus_in])`. Widen the Synapses' first Dense input accordingly. **Important:** `nn.Dense` infers its input dim **only once, at parameter init time**, and then freezes the kernel shape — it does NOT re-widen across calls. So if iteration 0 is called with `consensus_in=None` (input dim `d_input + d_model`) and iteration 1 with a real consensus vector (input dim `d_input + d_model + synch_size`), the second call raises `ScopeParamShapeError`. The fix is to make the Synapses input shape consistent across all iterations: when `INC_ENABLED=True` and `CTM_ITERATIONS > 1`, seed `consensus_in` with `jnp.zeros((batch, synch_size))` on iteration 0 instead of `None`. A zero vector carries no information from teammates, so the "no consensus on iter 0" semantics are preserved while the kernel shape stays stable. For `CTM_ITERATIONS == 1` and for `INC_ENABLED=False`, keep `consensus_in=None` so the Stage 1 no-op equivalence path is bit-identical.
- [x] For Stage 1, pass `consensus_in = None` on every iteration — this reproduces the current code path exactly but through the refactored helper. The final `synch` return value stays the last iteration's.
- [x] Run the Stage 0 baseline command again. The win-rate curve should overlay the original within noise (same seed → bit-identical, ideally).

**Stage 1 evidence (2026-04-11):**
- `CTMCell` refactored to `_single_iter(...)` with per-iteration sync emission and optional `consensus_in` input.
- Training scripts wired with `NUM_CONSENSUS_ITERATIONS` Stage-1 guard (`0` only), preserving no-op behavior.
- SMAX baseline after refactor is very similar to Stage 0 (within expected run noise), so Stage 1 is accepted as complete.

**Exit criterion:** refactor is a no-op; SMAX 3m training curve matches pre-refactor run.

---

## Stage 2 — Centralised cross-agent pooling

**Goal:** introduce the consensus mechanism itself, behind a config flag. When the flag is off, behaviour is identical to Stage 1. When on, between-iteration consensus is active.

**Files touched:** [ctm_jax.py](smax_ctm/ctm_jax.py), [train_mappo_ctm.py](smax_ctm/train_mappo_ctm.py), [train_mappo_ctm_hanabi.py](smax_ctm/train_mappo_ctm_hanabi.py). The `AgentConsensus` module and `CTMCell` changes are shared; both training scripts must set `INC_NUM_AGENTS` from their respective envs and pass it through to `ScannedCTM`.

- [X] Introduce new config keys:
  - `INC_ENABLED`: bool, default `False`.
  - `INC_POOLING`: one of `"mean" | "attention" | "gated"`, default `"mean"`.
  - `INC_NUM_AGENTS`: int, set automatically from env at `make_train` time.
  - `INC_CONSENSUS_DROPOUT`: float, default `0.0` (for later robustness experiments).
- [X] Add an `AgentConsensus` Flax module that takes a tensor of shape `(num_envs, num_agents, synch_size)` and returns a tensor of the same shape where each agent's output is the pooled summary of the **other** agents (leave-one-out pooling, so no self-leakage).
  - Mean: `(sum(axis=agent) - self) / (num_agents - 1)`.
  - Attention: a single-head dot-product attention across the agent axis; queries/keys/values all derived from sync vectors via small Dense layers. Mask out self.
  - Gated: mean of others, passed through `sigmoid(gate) * tanh(value)` parameterised by a Dense layer. Provides a learnable on/off switch per consensus channel.
- [X] In `CTMCell.__call__`, when `INC_ENABLED` is true:
  1. Run `_single_iter` with a zero `consensus_in` on iteration 0 (shape `(B, synch_size)`, all zeros). This carries no teammate information but keeps the Synapses' first Dense input dim identical across all iterations — see the Stage 1 note on `nn.Dense`'s one-shot input-dim inference. Only fall back to `consensus_in = None` when `CTM_ITERATIONS == 1` or `INC_ENABLED=False`, where the Synapses are only ever called with one input width.
  2. Between iterations, reshape the per-iteration `synch` from `(B, synch_size)` → `(num_agents, B // num_agents, synch_size)` (agent-major — see Stage 0 axis note).
  3. Pass through `AgentConsensus` → same shape. Module must be written so it pools along the **agent axis (axis=0)**, not axis=1.
  4. Reshape back to `(B, synch_size)`.
  5. Pass this as `consensus_in` to the next iteration's `_single_iter`.
- [X] Because `CTMCell` currently doesn't know `num_agents`, thread it in as a module field (`num_agents: int = 1`), set from config at construction time in `ScannedCTM`. The "num envs" dimension is inferred at runtime as `B // num_agents`.
- [X] **Batch-shape invariance between rollout and PPO update.** During rollout `B = num_actors = num_envs * num_agents`. During the PPO update, minibatches reshape the flat actor axis to `(num_minibatches, minibatch_size)` where `minibatch_size = num_actors * num_steps_per_env // num_minibatches` — but crucially the minibatch split in [train_mappo_ctm.py](smax_ctm/train_mappo_ctm.py) is done along the **env/agent flat axis while preserving agent-majority**. Verify this: grep for the permutation/reshape in the update step and confirm it does NOT shuffle agents across the flat axis. If it does (e.g. a `jax.random.permutation` over the flat actor axis), INC will train on corrupted consensus groupings. **Fix:** permute over the `num_envs` axis only, keep all agents of a given env together. Add a unit test.
- [X] **Sequence-length axis.** When `ScannedCTM` is applied during an update, inputs have a leading time axis `(T, B, ...)`. The inner `CTMCell` sees one time slice at a time (scan handles the T axis), so `B` is still `num_actors_in_minibatch` and the reshape logic above is unchanged. Double-check by printing shapes during the first update step.
- [X] **Dead-agent masking:** Hanabi and SMAX both kill agents partway through an episode. When an agent is dead its `done` mask is 1 and its sync is essentially frozen/invalid. Pass an `alive_mask` of shape `(num_envs, num_agents)` into `AgentConsensus` and use it to exclude dead agents from pooling. For SMAX derive this from the `dones` input; for Hanabi all agents stay alive so it's identity.
- [X] The final `synch` returned from `CTMCell` remains the last iteration's sync (same shape as before), so the actor head is unchanged.

**Exit criterion:** with `INC_ENABLED=True, INC_POOLING="mean", CTM_ITERATIONS=3`, training runs without shape errors and produces a different (hopefully better) learning curve. With `INC_ENABLED=False` behaviour is bit-identical to Stage 1.
Observations: INC_ENABLED=True, INC_POOLING="mean", CTM_ITERATIONS=3 produced a slightly better learning curve and final win rate in 3m smax. while setting INC_CONSENSUS_DROPOUT>0.0(0.25 used in experiments) in addition to INC_ENABLED=True, INC_POOLING="mean", CTM_ITERATIONS=3 produced a significantly better learning cure and a much higher win rate.

---

## Stage 2.1 — Disambiguating the dropout win ✅ DONE

**Goal:** figure out *why* `INC_CONSENSUS_DROPOUT=0.25` unlocks a much larger gain than plain INC on SMAX 3m (and Hanabi smoke test). The Stage 2 result is that plain INC (`INC_ENABLED=True, mean, iter=3`) is only marginally better than the Stage 1 baseline, but adding 25% dropout on the pooled consensus signal produces a substantially faster learning curve and a higher final win rate. Before committing to Stage 4's hyperparameter sweep and Stage 5's full matrix, we need to know whether the benefit comes from (a) the teammate consensus channel specifically, (b) stochastic regularisation of the CTM iteration loop in general, or (c) both. If it's (b), the paper's framing has to change — INC becomes "iterative refinement with regularised internal dynamics" and pooled teammate sync is a secondary contribution. This stage is cheap (all SMAX 3m) and decides the framing of Stages 4, 5, and 7.

**Files touched:** [ctm_jax.py](smax_ctm/ctm_jax.py) (small additions), [train_mappo_ctm.py](smax_ctm/train_mappo_ctm.py) (one new config key passthrough), new script `smax_ctm/scripts/run_stage2_1_disambig.py`, results written to `analysis_results_inc/stage2_1/`.

**Cells to run** (3 seeds each, same total-timesteps budget as the Stage 2 run, SMAX 3m):

| Cell | `INC_ENABLED` | `INC_CONSENSUS_DROPOUT` | `CTM_ITER_DROPOUT` (new) | `INC_FORCE_ZERO_CONSENSUS` (new) | Status |
|---|---|---|---|---|---|
| A | False | — | 0.0 | — | already logged (Stage 1 baseline) |
| B | True  | 0.0  | 0.0 | False | already logged (plain INC) |
| C | True  | 0.25 | 0.0 | False | already logged (INC + consensus dropout) |
| **D** | **False** | **—** | **0.25** | **—** | **new — stochastic iteration loop, no consensus** |
| **E** | **True**  | **0.25** | **0.0** | **True** | **new — same noise pattern as C but teammate info zeroed** |

Cells A/B/C already have curves from the Stage 2 run — reuse those logs, do not re-run.

- [x] Add config key `CTM_ITER_DROPOUT` (float, default `0.0`) to both [train_mappo_ctm.py](smax_ctm/train_mappo_ctm.py) and [train_mappo_ctm_hanabi.py](smax_ctm/train_mappo_ctm_hanabi.py). Thread it into `CTMCell` as a module field.
- [x] In `CTMCell._single_iter`, when `CTM_ITER_DROPOUT > 0`, apply `nn.Dropout(rate=CTM_ITER_DROPOUT, deterministic=not train)` to `activated_state_trace` **before** it is fed into the next iteration's synapses. Apply it regardless of `INC_ENABLED` — this is the "stochastic iteration loop" control and must be independent of the consensus path. Plumb the `train` flag through `ScannedCTM` the same way Flax's standard dropout usage does; if there is no existing dropout RNG stream, add one (`rngs={"dropout": dropout_rng}` in `apply`).
- [x] Add config key `INC_FORCE_ZERO_CONSENSUS` (bool, default `False`). In `CTMCell.__call__`, after computing the pooled `consensus_in` from `AgentConsensus` and *before* applying `INC_CONSENSUS_DROPOUT`, replace `consensus_in` with `jnp.zeros_like(consensus_in)` when this flag is set. Crucially: the dropout mask is still drawn and applied to the zero tensor, so the RNG consumption and the downstream noise pattern match cell C exactly. This is the cell-E control — same stochasticity pattern as C, same widened Synapses input dim, but zero actual teammate information flowing through.
- [x] Unit test the new flags in [smax_ctm/tests/test_inc.py](smax_ctm/tests/test_inc.py):
  - With `CTM_ITER_DROPOUT=0.0` and `INC_FORCE_ZERO_CONSENSUS=False`, forward pass is bit-identical to pre-change code.
  - With `INC_FORCE_ZERO_CONSENSUS=True`, the input to the iteration-2 Synapses on the consensus slice is all zero (before dropout is applied), verified by hook or by `jax.lax.stop_gradient` + numerical probe.
  - `CTM_ITER_DROPOUT=0.25` at eval time (`deterministic=True`) is a no-op; at train time it produces non-identity outputs on repeat calls with different RNGs.
- [x] Write `smax_ctm/scripts/run_stage2_1_disambig.py` that launches cells D and E, 3 seeds each. Same `TOTAL_TIMESTEPS`, `NUM_ENVS`, `CTM_ITERATIONS=3`, `INC_POOLING="mean"` as the Stage 2 run. Log to the same place as Stage 2 for apples-to-apples plotting.
- [ ] After runs complete, produce two plots into `analysis_results_inc/stage2_1/`:
  - **Learning curves:** all five cells (A, B, C, D, E) on one axis, win rate vs env steps, shaded by seed std. This is the headline plot for the stage.
  - **Final WR bar chart:** mean ± 95% CI at the end of training for each cell.
- [ ] Re-run [analyse_sync.py](smax_ctm/analyse_sync.py) on the cell-B checkpoint (plain INC) and the cell-C checkpoint (INC + dropout), producing the **within-step sync convergence plot** — how much does cross-agent sync disagreement shrink between iteration 0 and iteration K? Do both models converge in sync space, and does the dropout model converge *less* tightly while still performing better? This plot is reused in Stage 4 and becomes Figure 3 of the paper if the story holds.
- [x] Write a short interpretation note at `docs/inc_stage2_1_findings.md`: one paragraph per decision path (below), plus the learning-curve plot embedded.

**What we want to see — decision rules:**

The point of the stage is to pick between three mutually exclusive stories. Cells D and E each sharpen one dimension.

1. **Story "consensus channel is load-bearing":** D is close to A/B (weak), E is close to A/B (weak), and C remains the clear winner. This means the gain requires both (i) real teammate information flowing through the pool *and* (ii) dropout-robustness on that channel. Cleanest possible outcome — write the paper exactly as originally planned, with the dropout result as a prominent sub-finding under the "robust consensus channel" framing.
2. **Story "stochastic iteration loop":** D is close to C (strong), and E is also close to C (strong). This means the benefit is stochasticity inside the CTM iteration unroll, not teammate pooling — the consensus module is load-bearing only insofar as it gives dropout a place to apply. This is a real result but a different paper: INC becomes "regularised iterative refinement" and the "zero inter-step comm bandwidth consensus" framing has to be softened. We'd still run Stage 5, but the headline comparison changes from "INC vs no-INC" to "regularised iter vs plain iter", and Stage 6 ablations shift focus.
3. **Story "mixed":** D is in between A and C (noticeable lift but not matching C), and E is close to D (or close to A). This means both mechanisms contribute. Still a publishable story, framed as "iteration-loop stochasticity plus pooled teammate sync are complementary." Stage 5 stays the same; Stage 7 adds a small decomposition figure.

**Concretely, the numerical thresholds we'll use** (SMAX 3m, 3 seeds, final WR mean):
- "Close to A/B" ≡ within 3 percentage points of the plain-INC mean *and* learning-curve slope visually indistinguishable.
- "Close to C" ≡ within 3 percentage points of the INC+dropout mean *and* learning-curve slope visually indistinguishable.
- "In between" ≡ outside both bands.

If cells D and E together don't cleanly fit any of the three stories (e.g. D is strong but E is also strong, which would be internally inconsistent), treat that as a signal of an implementation bug in the new flags and go back to the unit tests before interpreting.

**Things that would invalidate the stage and force a re-run:**
- RNG stream for the new dropout is not properly split per-iteration (i.e. all iterations get the same mask) — would suppress the effect of `CTM_ITER_DROPOUT` artificially.
- `INC_FORCE_ZERO_CONSENSUS` accidentally also zeros the gradient through the consensus branch entirely, so the widened Synapses kernel receives no training signal on that slice — would make E look artificially weak. Verify with a gradient-flow probe in the unit test.
- Seeds from cells D/E inadvertently overlap with Stage 2's seeds, making the comparison biased. Use three fresh seeds.

**Exit criterion:** `analysis_results_inc/stage2_1/` contains the 5-cell learning-curve plot, the bar chart, and `docs/inc_stage2_1_findings.md` states which of the three stories the data supports, with the corresponding decision for Stage 4's framing written down. The main plan body (Stages 4-7) is updated only *after* this stage completes.

Observations (2026-04-11): Cells D (iter-dropout without INC) and E (INC with pooled teammate vector force-zeroed before dropout) were run on SMAX 3m for 3M timesteps, 3 seeds each (103/104/105), against the already-logged Stage 2 cells A/B/C. Cell D (`CTM_ITER_DROPOUT=0.25`, `INC_ENABLED=False`) *regressed* below even the no-INC CTM baseline (final WR 0.745 vs A's 0.793, −4.8 pp; never held win rate above 0.8), killing the "stochastic iteration loop" story outright. Cell E matched the no-dropout INC baseline B on final WR (0.802 vs 0.811) but stayed well below cell C on the stronger curve-shape metrics — fraction of updates with WR ≥ 0.8 was 14.9% for E vs 41.5% for C (~2.8×), and E's first sustained crossing of 0.8 lagged C by ~24 updates. Taken together this is **Story 1 ("consensus channel is load-bearing")**: the cell C win comes from noised-but-real teammate information flowing through the pool, not from dropout-as-regulariser on the CTM iteration loop, and not from side effects of the dropout call site (RNG consumption, widened Synapses kernel). Stage 4/5/7 framing stays as originally planned — dropout is kept as a prominent sub-finding under the robust consensus channel story. Full numbers and interpretation in [docs/inc_stage2_1_findings.md](docs/inc_stage2_1_findings.md).

---

## Stage 2.5 — Unit tests for INC plumbing

**Goal:** lock in the non-obvious invariants before they regress silently during later stages.

- [X] **Axis round-trip test.** Build a dummy `(num_actors, synch_size)` tensor where row `i` equals `i // num_envs` (so agent 0 rows are all 0, agent 1 rows are all 1, etc.). Reshape via the Stage-1 helper, confirm the agent axis is `[0, 1, ..., num_agents-1]` as expected.
- [X] **Leave-one-out mean test.** With `num_agents=3`, sync = `[[1], [2], [3]]` broadcast over envs, confirm pooled result is `[[2.5], [2.0], [1.5]]`.
- [X] **No-op equivalence.** With `INC_ENABLED=False`, the per-step output of `CTMCell` (including `synch`, action logits) must be bit-identical to the pre-refactor code on the same seed. Use `jnp.allclose(..., atol=0, rtol=0)`.
- [X] **Dead-agent masking.** With `num_agents=3` and agent 1 dead (`alive_mask=[1,0,1]`), the leave-one-out mean for agent 0 should equal agent 2's sync exactly, and vice versa; agent 1's pooled input should be a deterministic zero (or last-alive value — pick one and document it).
- [X] **Gradient flow.** Run one backward pass through an `INC_ENABLED=True` forward, confirm gradients are non-zero on the `AgentConsensus` parameters and on the widened `Synapses` first-layer kernel.
- [X] **Minibatch permutation safety.** Construct a fake update-step minibatch that permutes the flat actor axis, run `CTMCell` over it twice (once with the permutation, once without), and confirm the INC pooling yields the same per-agent result up to the permutation. Fails if the permutation scrambles agent groupings — in which case Stage 2's permutation-fix is required.

**Exit criterion:** all tests pass as a single `pytest` file `smax_ctm/tests/test_inc.py`. This file is checked in before Stage 4 begins.

---

## Stage 3 — Hanabi environment wiring

**Goal:** have CTM-MAPPO and GRU-MAPPO train on Hanabi in JaxMARL. This stage does NOT depend on Stage 2 — it can be done in parallel.

**Files touched:** [jaxmarl/environments/hanabi/](jaxmarl/environments/hanabi/), [jaxmarl/environments/__init__.py](jaxmarl/environments/__init__.py), [jaxmarl/registration.py](jaxmarl/registration.py), [smax_ctm/train_mappo_ctm_hanabi.py](smax_ctm/train_mappo_ctm_hanabi.py), [smax_ctm/train_mappo_gru_hanabi.py](smax_ctm/train_mappo_gru_hanabi.py), [smax_ctm/test&logger/run_hanabi_tests.py](smax_ctm/test&logger/run_hanabi_tests.py).

**Why per-benchmark training scripts?** SMAX and Hanabi have non-trivially different env contracts — `get_avail_actions` vs `get_legal_moves` (returns a per-agent **dict**, not an array, so vmap yields a dict of `(num_envs, num_moves)` arrays and must be batchified directly), different world-state constructions (SMAX state + one-hot agent id vs Hanabi observation concatenation), different episode-return logging (win rate vs score out of 25), and different wrappers (`SMAXLogWrapper` vs generic `LogWrapper`). Forcing one script to branch on env name would obscure both code paths. We keep the two benchmarks in separate files and share only what actually is shared: `ctm_jax.py` (CTMCell, ScannedCTM, AgentConsensus) and eventually a shared INC module.

- [x] Copy `jaxmarl/environments/hanabi/` from upstream FLAIROx/JaxMARL into the local `jaxmarl/environments/` tree. Register it in [jaxmarl/environments/__init__.py](jaxmarl/environments/__init__.py) and [jaxmarl/registration.py](jaxmarl/registration.py).
- [x] Create [smax_ctm/train_mappo_ctm_hanabi.py](smax_ctm/train_mappo_ctm_hanabi.py) as a Hanabi-specific clone of [train_mappo_ctm.py](smax_ctm/train_mappo_ctm.py):
  - SMAX imports replaced: `HanabiEnv` from `jaxmarl.environments.hanabi`, generic `LogWrapper` instead of `SMAXLogWrapper`, no `map_name_to_scenario`.
  - New local `HanabiWorldStateWrapper`: concatenates all agents' obs into a `num_agents * obs_size` world state, tiled per agent. Matches the centralised-critic convention from upstream MAPPO-Hanabi.
  - `avail_actions` is read by vmapping `env._env.get_legal_moves` over envs (returns dict `{a: (num_envs, num_moves)}`), then passed directly to `batchify`. **Do not treat the vmap result as an array** — that was the first bug to get hit and fix during wiring.
  - `step9_report` import removed (SMAX-specific).
  - Episode-return logging reports Hanabi score (max 25 for 2p), no win rate.
  - `last_done` initialised to `jnp.ones(...)` at rollout start to trigger CTM reset-on-done (matches SMAX CTM behaviour — the learned start-trace pathway).
  - Checkpoint saved to `model/hanabi_mappo_ctm_actor{_nosync}.pkl` with `env: "hanabi"` tag for loader disambiguation.
- [x] Create [smax_ctm/train_mappo_gru_hanabi.py](smax_ctm/train_mappo_gru_hanabi.py) as a Hanabi-specific clone of [train_mappo_gru.py](smax_ctm/train_mappo_gru.py) with the same env-wiring adaptations. GRU does not need the force-reset-on-start trick; `last_done` initialised to zeros.
- [x] Basic env contract test: [smax_ctm/test&logger/run_hanabi_tests.py](smax_ctm/test&logger/run_hanabi_tests.py) — reset/step contracts, `get_legal_moves` shape, short rollout stability, 3-player setting, fail-loud on unknown env id.
- [ ] Run GRU-MAPPO on Hanabi 2p. Confirm it reaches published numbers (upstream MAPPO-Hanabi score ~24/25 at convergence).
- [ ] Run CTM-MAPPO on Hanabi 2p with `CTM_ITERATIONS=1, INC_ENABLED=False`. Confirm it learns *something* (non-trivial above random). This is the CTM-without-consensus Hanabi baseline.

**Exit criterion:** both GRU and CTM baselines train on Hanabi 2p end-to-end without errors; GRU matches upstream reported numbers within noise.

---

## Stage 4 — Hyperparameter sanity on the consensus mechanism

**Goal:** find a sensible default before the full experiment matrix. Use SMAX 3m because it's fast, then confirm on Hanabi.

- [ ] Small sweep on SMAX 3m with `INC_ENABLED=True, INC_POOLING="mean"`:
  - `CTM_ITERATIONS ∈ {2, 3, 5}` × 3 seeds.
  - Record final WR, wall-clock per update, sync/obs correlation delta (the Stage 1-3 metric).
- [ ] Pick a lead `CTM_ITERATIONS` value (I expect 3-5). Document rationale in `docs/inc_hparams.md`.
- [ ] Confirm that turning `INC_POOLING` between `mean`, `attention`, `gated` all train stably on SMAX 3m. Don't pick a winner yet — that's the ablation in Stage 6.
- [ ] Re-run the Stage 1-3 sync analysis script ([analyse_sync.py](smax_ctm/analyse_sync.py)) on an INC model and add a new plot: **within-step sync convergence across iterations** (how much does consensus reduce the cross-agent sync disagreement by iteration `k` vs iteration `0`?). This is a key mechanism-verification figure for the paper.

**Exit criterion:** a default `CTM_ITERATIONS` is chosen, all pooling variants train without NaNs, within-step convergence plot exists and shows a non-trivial trajectory.

---

## Stage 5 — Main experiment matrix

**Goal:** produce the numbers the paper reports. Everything here is pre-planned, no fishing.

**Axes:**

| Axis | Values |
|---|---|
| Method | `GRU-MAPPO`, `CTM-MAPPO (iter=1, no INC)`, `CTM-MAPPO (iter=K, no INC)`, `CTM-MAPPO + INC (iter=K)` |
| Benchmark | Hanabi-2p, Hanabi-4p, SMAX 2s3z, SMAX 5m_vs_6m |
| Seeds | 5 per cell |

where `K` is the default chosen in Stage 4.

- [ ] Write a small runner script `smax_ctm/scripts/run_matrix.py` that enumerates the cells and dispatches training jobs with proper logging.
- [ ] Log to Weights & Biases (or local TFEvents if you prefer self-hosted): return, win rate (SMAX) / episode score (Hanabi), entropy, value loss, per-iteration sync convergence metric, wall-clock per update.
- [ ] The **key comparison** for the paper is `CTM-MAPPO (iter=K, no INC)` vs `CTM-MAPPO + INC (iter=K)`. Same compute, only difference is the consensus pooling. If INC doesn't beat this apples-to-apples control the method isn't real.
- [ ] The secondary comparison is `CTM-MAPPO + INC (iter=K)` vs `GRU-MAPPO` and vs `CTM-MAPPO (iter=1)` — needed to show the combined architecture is worth it.
- [ ] Track sample efficiency curves, not just final performance. Hanabi reviewers especially care about this.

**Exit criterion:** a full results table with means + 95% CIs across seeds for every cell, and sample-efficiency plots saved to `analysis_results_inc/`.

---

## Stage 6 — Ablations

**Goal:** answer the obvious reviewer questions before they're asked.

- [ ] **Pooling type.** `mean` vs `attention` vs `gated` on Hanabi-2p and SMAX 2s3z, 3 seeds each.
- [ ] **Iteration count.** `K ∈ {1, 2, 3, 5, 7}` with INC on, holding pooling fixed. Shows the trade-off between consensus depth and compute.
- [ ] **Consensus dropout.** Randomly zero out the pooled consensus input at training time with probability `INC_CONSENSUS_DROPOUT ∈ {0, 0.25, 0.5}`. Tests whether the policy can still act alone when the "channel" is unreliable — robustness story.
- [ ] **Leave-one-out vs full pool (incl. self).** Run one set with self-included pooling; confirm the leave-one-out choice isn't load-bearing.
- [ ] **Decentralised-at-test.** Take a centralised-trained INC model and evaluate it with each agent only pooling over its observable teammates (uses the SMAX sight radius or Hanabi's partial info). Reports performance drop. This is the defence against "this isn't decentralised enough" reviewers.
- [ ] **No-sync ablation on INC model.** Replace `compute_synchronisation` with a linear projection of the flat trace (existing `CTM_USE_SYNC=False` path). Tests whether INC's benefit comes specifically from the sync signal vs any pooled internal state. This is the Stage 5a test that was already on the Stage 1-3 roadmap.

**Exit criterion:** each ablation has a clean table or plot, interpretable in one sentence.

---

## Stage 7 — Analysis & paper figures

**Goal:** turn training logs into the figures and narrative of the paper.

- [ ] **Figure 1:** the pitch. Architecture diagram of CTM + iterative consensus, highlighting the new between-iteration pooling arrow.
- [ ] **Figure 2:** motivation from Stage 1-3 (sync rises during coordination events). Reuse the existing event-conditional bar chart.
- [ ] **Figure 3:** within-step sync convergence under INC — agents start uncorrelated at iteration 0 and align by iteration K. Side-by-side with a no-INC model showing flat-line.
- [ ] **Figure 4:** main results. Sample efficiency curves on Hanabi-2p and SMAX 5m_vs_6m for all four methods. Error bars = 95% CI over seeds.
- [ ] **Figure 5:** ablation grid — pooling type × iteration count heatmap on Hanabi-2p final score.
- [ ] **Figure 6:** interpretability. On a Hanabi episode, plot each agent's sync vector through time, marked with the information-giving move the policy actually takes. Does the consensus pooling visibly precede information moves?
- [ ] **Table 1:** full experiment matrix with means ± CI.
- [ ] Write the method section: problem statement (reuse the Stage 1-3 narrative for motivation), INC mechanism, equations for mean/attention/gated pooling, integration into CTM's iteration loop, complexity analysis (O(num_agents²) per iteration, negligible vs a single synapse forward pass).
- [ ] Write related work: CTM, sync-based coordination in MARL, comm methods (CommNet, TarMAC, QMIX), adaptive-compute (ACT, PonderNet). Frame INC against both "learned comm" (INC has no learned message head, consensus is over the model's own internal variable) and "adaptive compute" (INC uses iterations for *joint* consensus, not per-agent halting).
- [ ] Discussion: limitations (centralised pooling relies on parameter sharing; fully decentralised variants are future work), failure cases (does INC hurt on tasks that don't need coordination? — include a negative control like MPE `simple_tag` or a single-agent task).

**Exit criterion:** draft figures + writing ready for internal review.

---

## Stage 8 — Robustness & negative controls

**Goal:** strengthen against reviewer attacks.

- [ ] **Negative control.** Run INC on a task where agents don't need to coordinate (e.g. MPE `simple` — single agent, or an independent-reward setting). INC should NOT help here. If it does, the story is wrong and we need to rethink.
- [ ] **Compute-matched GRU.** Give GRU-MAPPO the same number of forward-pass FLOPs as INC-CTM (deeper GRU or more hidden units). Show INC still wins. This kills the "you just have more parameters" critique.
- [ ] **Parameter-count-matched.** Same but matched on total params instead of FLOPs.
- [ ] **Unseeded runs.** Final headline numbers on new seeds not used during development, to avoid seed-hacking accusations.
- [ ] **Scaling.** If time permits, try Hanabi 5p to show the mechanism scales with team size.

**Exit criterion:** negative control confirms INC does nothing where it shouldn't; compute-matched GRU still loses; headline numbers reproduce on fresh seeds.

---

---

## Dependencies between stages

```
Stage 0 ──► Stage 1 ──► Stage 2 ──► Stage 2.5 ──► Stage 4 ──► Stage 5 ──► Stage 6 ──► Stage 7 ──► Stage 8
              │                                      ▲
              └─► Stage 3 ──────────────────────────┘
```

Stage 3 (Hanabi wiring) can run in parallel with Stages 1-2 if you split sessions across machines.

---

## Config keys added (quick reference)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `INC_ENABLED` | bool | `False` | Master switch for iterative neural consensus |
| `INC_POOLING` | str | `"mean"` | One of `mean` / `attention` / `gated` |
| `INC_NUM_AGENTS` | int | auto | Set from env at `make_train` time |
| `INC_CONSENSUS_DROPOUT` | float | `0.0` | Train-time dropout on the pooled consensus signal |
| `INC_SELF_INCLUDED` | bool | `False` | Whether pooling includes the agent's own sync |

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| INC doesn't beat `CTM iter=K, no INC` apples-to-apples | Stage 4 gives an early signal before committing to the matrix; fall back to Sync-Alignment Aux Loss (direction #2 in the brainstorm) |
| Hanabi port is trickier than expected | Stage 3 is independent — if it stalls, run the full matrix on SMAX only and add Hanabi later |
| Attention pooling is unstable at high iteration counts | Gradient-clip the consensus branch separately; Stage 6 ablation will catch this |
| Dead-agent masking bug skews SMAX results | Unit test in Stage 2 that compares pooling with a dead agent to a manual expected value |
| Consensus creates implicit off-policy behaviour when `CTM_ITERATIONS` changes between train and eval | Keep `CTM_ITERATIONS` fixed between train and eval; document this as a constraint |

---

## Out of scope for this paper

- Decentralised learned comm heads (we explicitly contrast against this).
- Per-agent adaptive iteration count (that's the fallback "Think Fast, Think Slow" paper).
- Changes to the centralised critic architecture — keep it as GRU for now so the actor is the only independent variable.
- Heterogeneous-agent CTMs (one set of shared weights throughout).
