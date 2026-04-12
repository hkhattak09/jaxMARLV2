# Iterative Neural Consensus (INC) — Research Direction

**Status:** active research direction as of 2026-04-10.
**Paper working title:** *Iterative Neural Consensus: Coordinating Multi-Agent Policies Through Shared Internal Dynamics*
**Primary benchmark:** Hanabi (via JaxMARL).
**Secondary benchmarks:** SMAX 2s3z, SMAX 5m_vs_6m.
**Owners:** Hassan (implementation), with supervision.

This document is written so that a new collaborator — someone who has not followed the prior Stage 1-3 sync-analysis work — can pick it up and understand (a) what we are building, (b) why, (c) how we will measure whether it works, and (d) how we will write it up. Implementation details at the code/file level live in a separate document: [Implementation_Plans/ITERATIVE_NEURAL_CONSENSUS_PLAN.md](../Implementation_Plans/ITERATIVE_NEURAL_CONSENSUS_PLAN.md).

---

## 1. Background: what we are building on

### 1.1 Continuous Thought Machines (CTM)

The Continuous Thought Machine (CTM) is a neural architecture that replaces a single feed-forward pass with an **internal iteration loop**. Within one "external" time step, the network runs `CTM_ITERATIONS` internal rounds. Each round updates a neuron-level *state trace* and *activated-state trace* through learned "synapses" and per-neuron neuron-level models ("NLMs"). After the final iteration, the model reads out a **synchronisation vector** computed from the temporal correlations of activated-state pairs — this is the feature the downstream policy/value head actually uses.

Key point for this project: the sync vector is **the CTM's representation of what it is currently "thinking about."** It is high-dimensional (528 in our configuration), dense, and computed from internal dynamics rather than from the raw observation.

### 1.2 Stage 1-3 sync analysis (the motivation)

In prior work on this codebase we trained a CTM-MAPPO agent on SMAX maps (3m, 2s3z) and analysed what the sync vector encodes. The main findings, in plain terms:

1. **Sync vectors rise during coordination events.** When two teammates are about to jointly focus fire on the same enemy, or when an agent is about to retreat into a teammate's support radius, the magnitude and cross-agent correlation of the sync vectors spike in the few steps leading up to the event.
2. **Sync correlates with observation features only partially.** A linear probe from observation → sync explains well under half of the variance. The sync is *not* just a re-encoding of the obs — it carries genuinely internal state.
3. **Cross-agent sync alignment predicts coordinated actions.** When two agents' sync vectors are close in cosine distance, their subsequent actions are more likely to be mutually consistent (joint focus fire, mutual cover) than chance.

These results told us that **sync is a real coordination signal**. They also surfaced a limitation: under the current architecture, sync is computed **independently per agent**. Any "alignment" that appears is incidental — it emerges from the fact that both agents see overlapping observations, not from any shared computation. The natural next question, and the research direction in this document, is: **what happens if we deliberately let the sync signal flow between agents?**

Full Stage 1-3 results live at [docs/old/stage1to3_results.md](old/stage1to3_results.md).

### 1.3 Why Hanabi

Hanabi is the standard benchmark for *pure coordination under partial information*. Agents see others' cards but not their own, communication is restricted to a small vocabulary of hint actions, and the reward is fully cooperative. Unlike SMAX (which is a mixed reactive/strategic combat benchmark), Hanabi rewards *reasoning about what your teammate is thinking about* — which is exactly what a shared-internal-dynamics mechanism is supposed to give you. If INC cannot help on Hanabi, the method's story is wrong.

SMAX stays in the matrix as a secondary benchmark because (a) our Stage 1-3 analysis pipeline already runs on it, and (b) it is fast enough to use for hyperparameter sanity and ablations.

---

## 2. The research question

> **Does allowing a team of CTM agents to exchange their internal synchronisation vectors between internal iterations — with zero inter-step communication bandwidth — improve coordination on Hanabi and SMAX, compared to a matched-compute CTM that iterates the same number of times without the exchange?**

The key word is **matched compute**. The fair control is not a GRU, and not a CTM with one iteration. The fair control is a CTM with the *same* number of iterations and the *same* parameter count, just without the cross-agent pooling step. Anything else lets a critic say "you just gave it more compute."

We care about three secondary questions:

- Does the consensus mechanism actually use the sync signal we think it does, or does it degenerate into an unrelated channel?
- Does it scale with team size (Hanabi 2p → 4p → 5p)?
- Does it hurt on tasks that *do not* need coordination? (A method that helps everywhere, including single-agent tasks, is suspect — it probably just adds capacity.)

---

## 3. The mechanism, in words

Start with a team of `N` agents sharing CTM weights (parameter-sharing actor, as in standard MAPPO). Normally, each agent runs its CTM independently for `K` internal iterations per environment step, then reads out a sync vector and acts.

INC changes the inner loop. At each iteration `i = 1, 2, ..., K`:

1. **Per-agent step.** Each agent updates its own state trace and activated-state trace from its own features and its previous-iteration state, exactly as today.
2. **Per-agent sync readout.** Each agent computes a fresh sync vector from its current activated-state trace (not just at the final iteration — at every intermediate iteration).
3. **Cross-agent pool.** Collect the sync vectors of all teammates, pool them (mean / attention / gated), produce one **consensus vector per agent**. Leave-one-out: agent `j`'s consensus input does not include agent `j`'s own sync, so there is no trivial self-leakage.
4. **Feed forward.** At iteration `i+1`, the synapses receive not only the agent's own features and prior activated state, but also the consensus vector. The next internal state is conditioned on what the rest of the team was "thinking about" one iteration ago.

After `K` iterations the final sync vector is read out as today and goes to the actor head.

Two framing points that matter when defending against reviewers:

- **Not learned message passing.** There is no learned "message head" that produces a symbol to be decoded by a receiver. The consensus input is the CTM's own internal sync variable, pooled by a fixed (or lightly-parameterised) operation. Nothing new is being *transmitted* — we are exposing the already-computed internal variable across the agent axis inside one env step.
- **Not test-time communication across env steps.** All pooling happens *within* a single env step, between internal CTM iterations. Between env steps, no information flows between agents that wasn't already flowing through the environment itself. Zero bandwidth, in the communication-protocol sense.

---

## 4. Method spec (for reproducibility)

Let `B = num_envs * num_agents` be the flat actor batch. Inputs to `CTMCell` at one env step are features `f ∈ R^{B × F}` plus the CTM carry (state trace, activated-state trace). Let `s ∈ R^{synch_size}` be the per-agent sync vector; `synch_size = 528` in our current config.

At iteration `i`:

```
state_trace_i, activated_i, s_i = CTMCell.single_iter(
    state_trace_{i-1},
    activated_{i-1},
    features=f,
    consensus_in=c_{i-1},     # None at i=0
)
```

The pooling step reshapes `s_i ∈ R^{B × synch_size}` to `(N, B/N, synch_size)` (agent-major — see implementation plan for the axis-order subtlety) and applies a leave-one-out pool `P`:

- **Mean pool:** `c_i[j] = (1/(N-1)) * sum_{k != j} s_i[k]`
- **Attention pool:** single-head dot-product attention across the agent axis, with queries/keys/values from small Dense projections of `s_i`, self-masked so `j` cannot attend to itself.
- **Gated pool:** `c_i[j] = sigmoid(W_g * mean_{k != j} s_i[k]) ⊙ tanh(W_v * mean_{k != j} s_i[k])`.

Dead agents (`alive_mask[j] == 0`) are excluded from the pool at runtime. For Hanabi the mask is always ones; for SMAX it comes from the environment's `dones` state.

The consensus vector `c_i` is reshaped back to `(B, synch_size)` and concatenated with `(f, activated_i)` when computing the synapses' input for iteration `i+1`.

All other CTM details — NLM structure, synapses, memory length, sync decay — are held fixed. INC is a strict addition, not a redesign.

Hyperparameters added: `INC_ENABLED`, `INC_POOLING ∈ {mean, attention, gated}`, `INC_NUM_AGENTS` (auto), `INC_CONSENSUS_DROPOUT` (default 0), `INC_SELF_INCLUDED` (default False).

---

## 5. Relationship to prior work

- **CommNet, TarMAC, IC3Net, and other learned-comm methods.** These learn a dedicated message head at every step and require an explicit communication channel between agents at inference time. INC does not: consensus happens inside the internal iteration loop of one env step, and the "message" is the model's own internal sync variable rather than a learned output head. In fully decentralised deployments INC can still be run with the model's own sync signal masked to visible teammates (Stage 6 ablation).
- **QMIX and value-decomposition methods.** These mix value functions centrally at training time but keep per-agent policies independent at execution. INC is orthogonal — it modifies the *policy internal state*, not the value function.
- **ACT / PonderNet / adaptive computation.** These use iteration loops to give single-agent networks variable compute. INC uses iterations for *cross-agent consensus*, not per-agent halting. A natural follow-up paper ("Think Fast, Think Slow") would combine the two: adaptively decide how many consensus iterations to run. That is explicitly out of scope here.
- **CTM.** INC inherits the CTM's internal-iteration loop and sync readout. It does not change the CTM's single-agent dynamics. The only addition is the between-iteration cross-agent pool.

---

## 6. Experiment plan

### 6.1 Main results matrix

| Method | Hanabi-2p | Hanabi-4p | SMAX 2s3z | SMAX 5m_vs_6m |
|---|---|---|---|---|
| GRU-MAPPO | ✓ | ✓ | ✓ | ✓ |
| CTM-MAPPO, `K=1`, no INC | ✓ | ✓ | ✓ | ✓ |
| CTM-MAPPO, `K=K*`, no INC (compute-matched control) | ✓ | ✓ | ✓ | ✓ |
| **CTM-MAPPO + INC, `K=K*`** | ✓ | ✓ | ✓ | ✓ |

`K*` is chosen in Stage 4 (hyperparameter sanity), expected to be in the 3-5 range. 5 seeds per cell.

The headline claim is that **CTM + INC beats the matched-compute CTM control** by a statistically meaningful margin on Hanabi. If this fails, the method does not justify the paper.

### 6.2 Ablations

- **Pooling type:** mean vs attention vs gated, Hanabi-2p and SMAX 2s3z, 3 seeds.
- **Iteration count:** `K ∈ {1, 2, 3, 5, 7}` with INC on, pooling fixed.
- **Consensus dropout:** `{0, 0.25, 0.5}` — tests robustness to unreliable consensus.
- **Leave-one-out vs self-included pooling** — confirms the LOO choice is not load-bearing.
- **Decentralised-at-test:** train centralised, evaluate with each agent pooling only over teammates inside its observable range. Reports the cost of strict decentralisation at deployment.
- **No-sync ablation:** replace the sync readout with a linear projection of the flat activated-state trace, keeping INC on. Tests whether the benefit comes specifically from the sync signal or from any pooled internal state.

### 6.3 Negative controls and robustness

- **Task where coordination is unnecessary** (single-agent MPE, or an independent-reward multi-agent setting). INC should not help here. If it does, the story is wrong.
- **Compute-matched GRU** (deeper / wider GRU with the same forward-pass FLOPs as INC-CTM). Kills the "you just have more compute" critique.
- **Parameter-matched GRU**, for good measure.
- **Fresh seeds for headline numbers**, not used during development, to avoid seed-hacking.
- **Scaling to Hanabi 5p**, if compute allows.

### 6.4 Baselines we will cite but not necessarily re-run

- SAD (Simplified Action Decoder) — Hanabi SOTA for a long time, we will cite the published numbers rather than re-train unless a reviewer asks.
- R2D2-Hanabi — same.
- Re-running these is expensive and they are not the honest comparison; the honest comparison is against a matched-compute CTM.

---

## 7. Analysis plan — how we will actually look at the results

This section is deliberately written in detail because "look at some plots" is where most papers hide mistakes. For each analysis below: what question it answers, how to compute it, what plot or table we produce, and what counts as a pass.

### 7.1 Learning-curve analysis (main result)

**Question:** does INC outperform the matched-compute CTM control, and by how much, with what sample efficiency?

**How to compute:**
- For each (method, env, seed), log episode return (Hanabi) or win rate (SMAX) every `EVAL_INTERVAL` updates.
- Align curves by environment steps (not updates — methods with different iteration counts differ in wall clock but use the same env steps per update).
- Across 5 seeds, compute mean and 95% bootstrap CI at each eval point.

**Plot:** Figure 4 of the paper — one subplot per environment, x = env steps, y = return, one line per method with shaded CI band.

**Statistical test:** at the final-performance point, rank-sum (Mann-Whitney U) across seeds between `INC` and the matched-compute control. Also compute the probability-of-improvement metric (rliable) across seeds and environments jointly. Pass condition: `p < 0.05` on at least Hanabi-2p AND Hanabi-4p, AND rliable aggregate probability-of-improvement over baseline > 0.6.

### 7.2 Within-step sync convergence (mechanism verification)

**Question:** is the consensus mechanism actually doing what we claim — i.e. are agents' sync vectors becoming more aligned by iteration `K` than at iteration 0?

**How to compute:**
- During eval rollouts, at each env step, log the per-iteration sync vector for every agent: `s_i^{(a)}` for `i = 0, 1, ..., K-1`, `a = 1, ..., N`.
- For each iteration `i`, compute the **mean pairwise cosine similarity** across agents: `C_i = mean_{a != b} cos(s_i^{(a)}, s_i^{(b)})`.
- Aggregate `C_i` over all env steps and seeds.

**Plot:** Figure 3 — line plot with `i` on x-axis, `C_i` on y-axis, one line for INC models, one line for the matched-compute CTM control (which has no consensus step and should show flat `C_i` across iterations, or at least no monotone trend). Shaded region = std across env steps.

**Pass condition:** INC models show a monotone or near-monotone increase in `C_i` with iteration index. No-INC control does not.

**Guard against circularity:** the sync is the thing we're pooling, so of course pooling raises cosine similarity. To avoid the trivial reading, also measure the **post-pool** sync at iteration `i+1` (the sync computed from the *new* activated state that was conditioned on the consensus). Rising `C` in the post-pool sync means the information is actually being used by the next iteration's synapses, not just linearly averaged.

### 7.3 Consensus-use ablation (does the model depend on the consensus signal?)

**Question:** at test time, if we replace the consensus vector with zeros (or with Gaussian noise at the same scale), how much does performance drop?

**How to compute:**
- Take a trained INC model. During eval, at each internal iteration, replace `c_i` with either zeros or `N(0, sigma^2 I)` with `sigma` matched to the empirical std of the true `c_i` over the eval distribution.
- Report eval return under each intervention alongside the unaltered baseline.

**Plot:** Figure 6 (interpretability) — bar chart per environment with three bars: unaltered, zero-consensus, noise-consensus.

**Pass condition:** performance drops meaningfully (>10% relative) under at least one intervention. A policy that is robust to *removing* its own consensus input is not actually using it.

### 7.4 Sync→action influence probe

**Question:** does the consensus signal influence *which action* the agent takes, not just internal state?

**How to compute:**
- At eval time, for each env step, compute the action distribution `pi(· | obs, c)` with the true consensus, and `pi(· | obs, 0)` with the consensus zeroed.
- Measure KL divergence between the two distributions. Average over env steps and agents.
- Bucket by "coordination event" vs "non-coordination step" (coordination events defined from the Stage 1-3 event labels — e.g. joint focus-fire in SMAX, hint-then-play in Hanabi).

**Plot:** bar chart — mean KL on coordination-event steps vs non-coordination steps.

**Pass condition:** KL on coordination-event steps is statistically larger than on non-coordination steps. This is the strongest single piece of evidence that the consensus mechanism is doing what the paper says.

### 7.5 Per-iteration attention pattern analysis (only if `INC_POOLING=attention` wins)

**Question:** if attention pooling is the best variant, does it learn interpretable "who should I listen to" patterns?

**How to compute:**
- Log attention weights `alpha[i, a, b]` (iteration `i`, target agent `a`, source agent `b`).
- Aggregate by role/archetype (e.g. in Hanabi, by hand composition; in SMAX, by unit type).

**Plot:** heatmap of attention weight by (source role, target role, iteration), one panel per environment.

**This is optional** — only produced if attention pooling is the headline variant.

### 7.6 Hanabi-specific behavioural analysis

**Question:** does INC improve Hanabi score by producing more *informative* hints, or just by playing more conservatively?

**How to compute:**
- Log action-type frequencies over eval episodes: `play`, `discard`, `hint_colour`, `hint_rank`.
- Log the **convention metric** from the Hanabi-agent literature: after a hint, what fraction of hinted cards are correctly interpreted (played if playable, kept if not) by the next player action?
- Log average episode score and the distribution over game outcomes (`0`, `1-10`, `11-20`, `21-25`).

**Plots:**
- Stacked bar of action-type frequency per method.
- Histogram of episode scores per method.
- Bar chart of convention-interpretation accuracy per method.

**Pass condition (soft):** INC models should show higher convention-interpretation accuracy than the no-INC control, not just higher mean score. If INC improves score without improving convention accuracy, it is probably improving play through a non-coordination mechanism and the paper's story needs a rewrite.

### 7.7 SMAX-specific behavioural analysis

**Question:** does INC improve SMAX win rate by producing more coordinated focus-fire?

**How to compute:**
- At each env step, for each enemy unit, count how many of our agents targeted it (inferrable from the action + attack-range mask).
- Define a **focus-fire concentration** metric: mean over steps of `max_enemy_targetedness / sum_enemy_targetedness` — 1.0 if all agents hit the same enemy, `1/N` if uniformly spread.
- Compare across methods.

**Plot:** box plot of focus-fire concentration per method per map.

**Pass condition (soft):** INC models focus-fire more tightly than the no-INC control.

### 7.8 Compute and wall-clock accounting

**Question:** what is the actual compute cost of INC relative to the control?

**How to compute:**
- Measure wall-clock per update step for each method, on the same hardware.
- Measure peak GPU memory.
- Count FLOPs analytically from the pooling op (`O(N^2 * synch_size)` per iteration for attention, `O(N * synch_size)` for mean).

**Plot:** a small table in the paper — method × (FLOPs per step, wall clock per step, peak memory). Has to be in the paper to honestly frame the "matched compute" comparison.

### 7.9 Aggregate reliability analysis (rliable)

**Question:** across all environments and seeds, is INC's improvement robust to the specific choice of benchmark and seed subset?

**How to compute:**
- Use the `rliable` library's interquartile-mean, probability-of-improvement, and performance-profile plots across all (env × seed) cells.
- Report IQM and 95% stratified bootstrap CIs per method.

**Plot:** Figure 5 (or Table 1 companion) — IQM bar with CI bars, probability-of-improvement over baseline heatmap, and performance profile.

**Pass condition:** INC's IQM is above the matched-compute control's IQM, CI non-overlapping, and probability-of-improvement over the control is above 0.6.

### 7.10 Negative-result verification

**Question:** on the coordination-free negative control, does INC *not* help (as predicted)?

**How to compute:** run INC on a single-agent or independent-reward task. Report learning curves.

**Pass condition:** INC neither meaningfully outperforms nor meaningfully underperforms the matched-compute control. A method that *always* helps is suspicious. A method that helps on coordination tasks and is neutral elsewhere is the story we want.

---

## 8. Risks to the research direction

| Risk | What it would look like | Mitigation |
|---|---|---|
| INC is indistinguishable from the matched-compute CTM control | Section 7.1 fails; Section 7.2 shows the within-step convergence plot but 7.3 shows the model is robust to zero-consensus — i.e. the mechanism is being ignored | Early signal in Stage 4 (SMAX 3m sanity sweep); fall back to the *sync-alignment auxiliary loss* direction — train the CTM to keep teammates' sync vectors close during coordination events, rather than pooling them |
| INC helps but only because of extra parameters | Compute/param-matched GRU also wins, or a randomly-initialised frozen pooling also helps | Section 7.3 (zero-consensus ablation) and Section 8 compute-matched baselines; if the model is robust to zeroing consensus then the "extra capacity" critique is correct |
| Hanabi port stalls | Stage 3 of the implementation plan does not exit-criterion in a reasonable time | Run the full matrix on SMAX only and add Hanabi as a later update; SMAX-only is not ideal for the story but is a valid fallback |
| Attention pooling is unstable at large `K` | Loss curves diverge; grads explode on the consensus branch | Gradient-clip the consensus branch separately; fall back to mean pooling as the default |
| Results look good on 2 agents but collapse at 4-5 agents | INC helps Hanabi-2p but not Hanabi-4p | Ablate pooling type at scale; may need attention pooling with a learned who-to-listen-to mask for large teams |

---

## 9. Out of scope (written down so we don't drift)

- Learned communication heads. We deliberately contrast against them.
- Per-agent adaptive iteration count (halting / ponder mechanism). Future work.
- Modifying the centralised critic. Critic stays GRU.
- Heterogeneous agent weights. Parameter sharing throughout.
- Offline / imitation learning on Hanabi human replays. Pure self-play only.
- Transfer to real-world robotics or language tasks. Not claimed.

---

## 10. Where to find things

- **Implementation plan** (stage-by-stage, file-level): [Implementation_Plans/ITERATIVE_NEURAL_CONSENSUS_PLAN.md](../Implementation_Plans/ITERATIVE_NEURAL_CONSENSUS_PLAN.md)
- **Prior sync analysis that motivates this direction:** [docs/old/stage1to3_results.md](old/stage1to3_results.md)
- **CTM core implementation:** [smax_ctm/ctm_jax.py](../smax_ctm/ctm_jax.py)
- **MAPPO training loops (per benchmark — see note below):**
  - SMAX CTM: [smax_ctm/train_mappo_ctm.py](../smax_ctm/train_mappo_ctm.py)
  - SMAX GRU: [smax_ctm/train_mappo_gru.py](../smax_ctm/train_mappo_gru.py)
  - Hanabi CTM: [smax_ctm/train_mappo_ctm_hanabi.py](../smax_ctm/train_mappo_ctm_hanabi.py)
  - Hanabi GRU: [smax_ctm/train_mappo_gru_hanabi.py](../smax_ctm/train_mappo_gru_hanabi.py)
- **Sync analysis scripts to be extended for Section 7.2:** [smax_ctm/analyse_sync.py](../smax_ctm/analyse_sync.py) (to be adapted — may need a new `analyse_inc.py` variant that logs per-iteration sync)
- **Hanabi environment (ported from upstream JaxMARL):** [jaxmarl/environments/hanabi/](../jaxmarl/environments/hanabi/)
- **Hanabi env contract tests:** [smax_ctm/test&logger/run_hanabi_tests.py](../smax_ctm/test&logger/run_hanabi_tests.py)

**Note on per-benchmark scripts.** SMAX and Hanabi are wired through two separate training files for each of {CTM, GRU} rather than one script that branches on env name. The benchmarks differ in enough small places — `get_avail_actions` vs dict-returning `get_legal_moves`, SMAX world-state + agent-id one-hot vs Hanabi-obs concatenation, `SMAXLogWrapper` vs generic `LogWrapper`, win-rate vs score-out-of-25 logging, `num_agents` read from `num_allies` vs directly from the env — that a single branching script would hide both code paths rather than simplify them. The CTMCell, ScannedCTM, and (in Stage 2) `AgentConsensus` module live in shared files and are imported by both Hanabi and SMAX scripts, so the actual method is defined once. Anything INC-related that must be implemented needs to be landed in **both** CTM training scripts (Hanabi + SMAX) as part of Stage 2.

---

## 11. A note on honesty

The point of Section 7 is to make it hard to fool ourselves. A method that passes Section 7.1 but fails Sections 7.3 and 7.4 is a method that *happens* to train better under the matched-compute control, not a method that does what we say it does. If that happens, the honest thing is to report it as an observation ("pooling internal state between CTM iterations helps, but we cannot confirm it is the sync signal doing the work") and either (a) rewrite the story or (b) move to the sync-alignment auxiliary loss as a fallback direction. We are not writing a paper called "here is a thing that trains marginally better"; we are writing a paper claiming a specific mechanism. The analysis plan exists so we can tell the two apart.
