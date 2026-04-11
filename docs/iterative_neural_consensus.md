# Sync-Based Cross-Agent Coordination in CTM-MARL — Research Direction

**Status:** active research direction as of 2026-04-11. Updated after a
mid-project pivot; see §4 for the rationale.
**Paper working title:** *Sync-Alignment Losses for Coordinated Multi-Agent
Policies.*
**Primary method:** Sync-Alignment Auxiliary Loss (SAAL) — a training-time loss
that encourages teammates' CTM sync vectors to align on coordination events.
Fully decentralised at execution.
**Comparison method:** Iterative Neural Consensus (INC) — between-iteration
cross-agent pooling of sync vectors inside a single env step. Implemented and
characterised on SMAX (Stages 0–2.5 of the implementation plan); retained in the
paper as a comparison condition and as a cautionary case study on Hanabi.
**Benchmarks:** SMAX 3m, 2s3z, 5m_vs_6m (primary); Hanabi 2p, 4p (secondary —
held back from full training until SAAL lands on SMAX).
**Owners:** Hassan (implementation), with supervision.

This document is written so that a new collaborator — someone who has not
followed the prior Stage 1-3 sync-analysis work — can pick it up and understand
(a) what we are building, (b) why, (c) how we will measure whether it works, and
(d) how we will write it up. Implementation details at the code/file level live
in [Implementation_Plans/ITERATIVE_NEURAL_CONSENSUS_PLAN.md](../Implementation_Plans/ITERATIVE_NEURAL_CONSENSUS_PLAN.md).

---

## 1. Background: what we are building on

### 1.1 Continuous Thought Machines (CTM)

The CTM is a neural architecture that replaces a single feed-forward pass with
an **internal iteration loop**. Within one "external" time step, the network
runs `CTM_ITERATIONS` internal rounds. Each round updates a neuron-level *state
trace* and *activated-state trace* through learned "synapses" and per-neuron
neuron-level models ("NLMs"). After the final iteration, the model reads out a
**synchronisation vector** computed from the temporal correlations of
activated-state pairs — this is the feature the downstream policy/value head
actually uses.

Key point: the sync vector is the CTM's internal representation of *what it is
currently thinking about*. It is high-dimensional (528 in our configuration),
dense, and computed from internal dynamics rather than from the raw observation.
This means the sync is a *richer* coordination signal than the raw obs would
provide.

### 1.2 Stage 1-3 sync analysis (the motivation)

In prior work on this codebase we trained a CTM-MAPPO agent on SMAX maps (3m,
2s3z) and analysed what the sync vector encodes. The main findings:

1. **Sync vectors rise during coordination events.** When two teammates are
   about to jointly focus-fire on the same enemy, or when an agent is about to
   retreat into a teammate's support radius, both the magnitude and the
   cross-agent similarity of the sync vectors spike in the few steps leading up
   to the event.
2. **Sync is not just a re-encoding of obs.** A linear probe from obs → sync
   explains well under half of the sync variance. The sync carries genuinely
   internal state.
3. **Cross-agent sync alignment predicts coordinated actions.** When two agents'
   sync vectors are close in cosine distance, their subsequent actions are more
   likely to be mutually consistent than chance.

Full Stage 1-3 results live at
[docs/old/stage1to3_results.md](old/stage1to3_results.md).

These findings tell us that **sync is a real coordination signal**, and that
cross-agent sync alignment is the specific variable that correlates with good
team behaviour. The research direction in this document is about how to *use*
that signal deliberately, rather than just observing it.

### 1.3 Two ways to exploit sync alignment

Once you know "sync vectors spontaneously align during coord events," there are
two natural architectural interventions:

- **Training-time intervention (SAAL).** Add an auxiliary loss to the actor
  update that rewards high cross-agent sync alignment on coord events and
  punishes it on non-coord steps. Gradients flow through the CTM, pushing the
  model to *learn* to produce aligned sync when coordination is called for. At
  execution, each agent runs a vanilla CTM-MAPPO — zero inter-agent bandwidth,
  fully decentralised.
- **Execution-time intervention (INC).** Between CTM internal iterations, pool
  each agent's sync vector across teammates and feed the pooled "consensus
  vector" into the next iteration's synapses. Agents literally see each other's
  internal state during the forward pass. Fat intervention, highest coordination
  potential, but requires an execution-time channel.

The paper's primary contribution is SAAL; INC is a comparison condition. §4
explains why this ordering is the defensible one.

### 1.4 Why Hanabi is the eventual test venue

Hanabi is the standard benchmark for *pure coordination under partial
information*. Agents see others' cards but not their own, communication is
restricted to a small vocabulary of hint actions, and the reward is fully
cooperative. Unlike SMAX — which is a mixed reactive/strategic combat benchmark
— Hanabi rewards reasoning about what your teammate is thinking about, which is
exactly what sync-based coordination is supposed to give you.

Hanabi is also the venue that *separates* SAAL and INC most clearly. A
training-time alignment loss does not change execution, so it plays Hanabi
legally by construction. INC exchanges internal state between agents at
execution, which in an information-restricted game raises the concern that it
may be smuggling card information across a channel the game rules say should
not exist. Testing both methods on Hanabi is the decisive experiment for the
pivot described in §4.

SMAX stays in the matrix as the primary proving ground because it's fast,
coordination-sensitive, and has no information-restriction rules — any method
that works there is working legitimately.

---

## 2. The research question

> **Can we improve multi-agent coordination in CTM-based policies by training
> teammates' internal sync vectors to align on coordination events, without
> opening an execution-time information channel between agents?**

Primary sub-questions:

- Does SAAL beat a vanilla CTM-MAPPO baseline on SMAX, with the same compute?
- Does SAAL generalise from one SMAX map to others?
- Does SAAL port to Hanabi, where the event detector has to be redesigned for a
  turn-based game?
- How does SAAL compare to INC — same or different kinds of wins?

Secondary sub-questions:

- Does INC post a strong Hanabi self-play number that *collapses* on cross-play,
  identifying it as a private-convention cheat?
- Does SAAL hurt on tasks that do not need coordination? (A method that helps
  everywhere is suspect; a method that helps on coord tasks and is neutral
  elsewhere is the story we want.)

---

## 3. The mechanisms, in words

### 3.1 SAAL — the primary method

Start with a team of `N` agents sharing CTM weights (parameter-sharing actor, as
in standard MAPPO). At each env step each agent runs its CTM forward pass and
produces a sync vector `s_i ∈ R^{synch_size}` (synch_size = 528 in our config).

Define a **coord-event mask** `ff(t, e)` that is True on step `t`, env `e`
whenever teammates are engaged in a coordination event (focus-fire on SMAX,
hint-then-play on Hanabi — see §5 for detector specs). Define the **pair cosine**

```
pair_cos(t, e) = mean_{i < j}  cos(s_i(t, e), s_j(t, e))
```

averaged over ordered agent pairs. The SAAL loss is:

```
L_align = - ALPHA * mean_{(t,e) ∈ ff}   pair_cos(t, e)
          + BETA  * mean_{(t,e) ∉ ff}   pair_cos(t, e)
```

Two coefficients: ALPHA pulls alignment up on coord events, BETA pushes it down
on non-coord steps. The β term matters under parameter sharing — without it, the
degenerate solution is to make sync vectors agent-invariant (no agent-specific
information in sync at all), which kills the policy. §5 goes into the collapse
argument in more detail.

The full actor loss becomes:

```
actor_loss = loss_actor - ENT_COEF * entropy + L_align
```

Gradients flow from `L_align` back through `pair_cos` into the sync vectors,
then through `compute_synchronisation` into the activated state trace, then
through the NLMs and Synapses into the CTM parameters. No new modules, no new
forward-pass ops at execution.

Two framing points that matter when defending SAAL against reviewers:

- **Training-time only.** L_align is computed during the PPO update; at rollout
  and eval each agent runs a vanilla CTM-MAPPO forward pass. There is literally
  no inter-agent communication at execution. This is decentralisable to
  arbitrary deployment scenarios without re-training.
- **Motivated directly from the sync analysis.** Stage 1-3 identified
  cross-agent sync alignment as the variable that correlates with good team
  behaviour. SAAL is the minimum-intervention way to turn that observation into
  a training signal. We are not inventing a new coordination channel — we are
  sharpening a phenomenon the model already exhibits.

### 3.2 INC — the comparison method

INC changes the inner loop of the CTM forward pass. At each internal iteration
`i = 1, …, K`:

1. **Per-agent step.** Each agent updates its own state trace from its own
   features and its previous-iteration state, exactly as today.
2. **Per-agent sync readout.** Each agent computes a fresh sync vector from its
   current activated-state trace.
3. **Cross-agent pool.** Collect the sync vectors of all teammates, pool them
   (mean / attention / gated), produce one **consensus vector per agent**.
   Leave-one-out — agent `j`'s consensus input does not include agent `j`'s own
   sync.
4. **Feed forward.** At iteration `i+1`, the synapses receive the agent's own
   features, prior activated state, and the consensus vector. The next internal
   state is conditioned on what the rest of the team was "thinking about" one
   iteration ago.

After `K` iterations the final sync vector is read out and goes to the actor
head.

INC is "learned communication of internal state during the forward pass." It is
the fatter intervention, with a direct execution-time coupling between agents.
On SMAX 3m it produces a clear win over vanilla CTM-MAPPO (see §6); on Hanabi
it is the source of the cheating concern that drove this pivot.

---

## 4. Why SAAL is primary: the Hanabi cheating concern

INC was the original headline method. It was demoted mid-project for a reason
that should have been caught earlier: on info-restricted benchmarks like Hanabi,
an execution-time channel between agents is not obviously legal.

### 4.1 The concern

Hanabi is constructed around an information asymmetry. Each agent sees others'
cards but not its own, and the only legal way to communicate private information
is to spend a hint token. The entire difficulty of the game — and the reason it
is a standard benchmark for coordination research — is that players have to
reason about what their teammate knows under this constraint.

INC lets agents exchange their internal sync vectors between CTM iterations.
Those sync vectors are computed from each agent's own activated-state trace,
which in turn is computed from its own observation. An agent's observation
includes its teammates' cards. So agent A's sync vector contains features
derived from agent B's cards. When INC pools agent A's sync into agent B's
next-iteration synapses, **it hands agent B information about agent B's own
cards** through a channel that is not a hint token.

Whether or not the model *learns* to exploit this depends on the training
signal, but the point is that INC cannot be defended as a legitimate Hanabi
method without evidence that it is not exploiting it. The natural evidence is
the **cross-play test**: two INC agents trained in self-play will post a strong
score together, but if one is paired with an independently-trained partner at
eval time, any exploitation of a private convention — including conventions
encoded over the sync channel — will collapse. That is the standard diagnostic
from the Hanabi-as-ZSC literature (Jakob Foerster's line of work).

### 4.2 Why SAAL is clean

SAAL has none of this concern because it doesn't touch execution. The loss only
affects gradients during training; at eval time each agent runs a CTM forward
pass that has no access to any other agent's state. Whatever coordination
emerges has to be expressed through the normal game actions (moves in SMAX,
hints/plays in Hanabi). Hanabi rules are respected by construction.

This is why SAAL is the publishable method even without running the Hanabi
discriminating test on INC. The Hanabi INC test is valuable because it produces
a sharp negative (or partial) result that strengthens the paper, but SAAL's
legitimacy does not depend on it.

### 4.3 What Stage 2.1 tells us

Before running Hanabi, we have one data point on the INC channel's role from
Stage 2.1 on SMAX 3m. Cell E of that experiment strips the pooled teammate
information while preserving every other aspect of the INC architecture
(stochasticity, widened Synapses kernel, gradient paths). The result: final win
rate drops only ~2 points (from 0.822 to 0.802) — a modest effect — but the
curve-quality metric (fraction of updates sustained above 0.8 WR) drops from
41.5% to 14.9%, nearly 3×. E first crosses 0.8 about 24 PPO updates later than
C.

The interpretation: the INC channel is load-bearing for *how reliably* and *how
early* the policy stays at high WR, but not for *whether* it reaches the same
peak. Full analysis:
[docs/inc_stage2_1_findings.md](inc_stage2_1_findings.md).

This matters for the pivot in two ways. First, it is SMAX-only evidence —
SMAX has no info-restriction, so a load-bearing channel there is neutral on the
Hanabi cheating question. Second, the modest final-WR gap suggests most of the
benefit should be reachable via a training-time signal; a 2-point gap is
plausibly recoverable by SAAL, even though SAAL cannot replicate everything the
execution-time channel does.

---

## 5. Method spec (for reproducibility)

### 5.1 SAAL loss

Let `B = num_envs * num_agents` be the flat actor batch and `T` be the rollout
length in a PPO minibatch. Inputs to the actor loss fn:

- `synch ∈ R^{T × B × synch_size}` — sync vectors returned by `ActorCTM` at each
  time step.
- `action ∈ Z^{T × B}` — actions taken in the rollout (used for the focus-fire
  detector on SMAX).
- `reward ∈ R^{T × B}` — rewards (used for the lookback detector on Hanabi).

Reshape `synch` to `(T, num_agents, num_envs_per_mb, synch_size)` agent-major
(the Stage 0 axis convention is load-bearing — see
[docs/inc_axis_convention.md](inc_axis_convention.md)). Compute L2-normalised
sync per (t, agent, env) and the pair cosine:

```python
s_norm = synch_am / (jnp.linalg.norm(synch_am, axis=-1, keepdims=True) + 1e-8)
cos_mat = jnp.einsum("taec,tbec->teab", s_norm, s_norm)   # (T, E, A, A)
iu, ju = jnp.triu_indices(num_agents, k=1)
pair_cos = cos_mat[..., iu, ju].mean(axis=-1)             # (T, E)
```

Compute the coord-event mask `ff_mask ∈ {0,1}^{T × E}` (see §5.3 for detectors).
Then the masked pull/push means and the loss itself:

```python
align_pos = jnp.where(ff_mask, pair_cos, 0.0).sum() / (ff_mask.sum() + 1e-8)
align_neg = jnp.where(~ff_mask, pair_cos, 0.0).sum() / ((~ff_mask).sum() + 1e-8)
L_align = -ALPHA * align_pos + BETA * align_neg
```

Add `L_align` to `actor_loss`. Everything else in the PPO update is unchanged.

### 5.2 Why the β term matters under parameter sharing

With parameter sharing, all agents are instances of the same CTM with different
observations. The naive version of the loss — just maximise `pair_cos` — has a
trivial optimum: make the sync vector a constant function of parameters that
ignores the agent-specific observation. This collapses all sync vectors to the
same point, the `pair_cos` target is reached at ~1.0, and the policy loses the
ability to distinguish agent states. BAD.

The β term puts tension into the system. On non-coord-event steps, the loss
actively *punishes* alignment. This forces sync to remain agent-distinct except
at the specific steps where coord events call for it. The two terms together
are a supervised contrastive objective: pull same-event pairs together, push
different-event pairs apart.

An alternative collapse-prevention formulation — subspace alignment, where only
a projection of the sync vector is aligned — is in the back pocket if the β
term turns out to be too coarse. But it is more complex, and the β formulation
is strictly simpler to implement and reason about.

### 5.3 Coord-event detectors

**SMAX focus-fire.** The cheapest online-detectable event. From
`traj_batch.action` shape `(T, B)`, reshape to `(T, num_agents, num_envs_mb)`.
Mark entries where the action index is in `[num_movement_actions,
num_movement_actions + num_enemies)`. For each `(t, env)`, True iff ≥2 agents
target the same enemy index. Logic mirrors the offline detector at
[smax_ctm/analysis/collector.py:73](../smax_ctm/analysis/collector.py#L73).
**No state plumbing needed** — actions are already in `Transition`.

**Grouping and enemy-kill** (deferred to Stage 9 ablation). These need raw SMAX
state fields (positions, `unit_alive`) that are not currently stored in
`Transition`. Plumbing them through rollout is ~50 lines but we hold it back
until we know focus-fire-only is insufficient.

**Hanabi reward lookback** (Stage 7, V1 proposal). On any step where `reward >
0`, mark the previous `k` steps (e.g. `k=2`) as "in coord event". This captures
"something good just happened, the reasoning leading up to it was probably
shared" without needing to parse the Hanabi action space into hint vs play vs
discard. Simpler than a proper hint-then-play detector and gets us a first
signal; Stage 7's ablation list includes "swap the reward detector for a true
hint-then-play detector and compare."

### 5.4 INC (comparison method)

Spec is unchanged from the original plan. Let `s ∈ R^{synch_size}` be the
per-agent sync vector at iteration `i`. The pooling step reshapes `s_i ∈ R^{B ×
synch_size}` to `(num_agents, B/num_agents, synch_size)` agent-major and applies
a leave-one-out pool `P`:

- **Mean pool:** `c_i[j] = (1/(N-1)) * sum_{k ≠ j} s_i[k]`
- **Attention pool:** single-head dot-product attention across agent axis, with
  queries/keys/values from small Dense projections of `s_i`, self-masked so `j`
  cannot attend to itself.
- **Gated pool:** `c_i[j] = sigmoid(W_g * mean_{k ≠ j} s_i[k]) ⊙ tanh(W_v *
  mean_{k ≠ j} s_i[k])`.

The consensus vector `c_i` is concatenated with `[features,
activated_state_{i-1}]` when computing iteration `i+1`'s Synapses input.

Dead agents (`alive_mask[j] == 0`) are excluded from the pool at runtime. For
Hanabi the mask is always ones; for SMAX it comes from the env `dones`.

Config keys: `INC_ENABLED`, `INC_POOLING ∈ {mean, attention, gated}`,
`INC_NUM_AGENTS` (auto), `INC_CONSENSUS_DROPOUT`, `CTM_ITER_DROPOUT`,
`INC_FORCE_ZERO_CONSENSUS`.

---

## 6. Evidence so far (Stages 0–2.5)

**SMAX 3m cells (3 seeds each at cells D/E, 1 seed each at A/B/C; all 3M
timesteps, `CTM_ITERATIONS=3`):**

| Cell | Config | Final WR | Frac ≥ 0.8 | First hit rolling-20 ≥ 0.8 |
|---|---|---|---|---|
| A | CTM baseline | 0.793 | 4.9 % | never |
| B | INC on, dropout 0.0 | 0.811 | 12.0 % | 173 |
| C | **INC on, dropout 0.25** | **0.822** | **41.5 %** | **131** |
| D | iter-dropout only (control) | 0.745 | 1.1 % | never |
| E | INC on, dropout 0.25, force-zero consensus | 0.802 | 14.9 % | 155 |

Full details:
[docs/inc_stage2_1_findings.md](inc_stage2_1_findings.md).

What this tells us:

- **INC does do something real on SMAX.** Cell C is clearly above cell A on both
  final WR and curve-quality metrics.
- **The pooled channel is load-bearing, not an incidental training scaffold.**
  Cell E (same architecture as C, same stochasticity pattern, consensus vector
  zeroed) reverts to the no-INC regime on curve quality. Cell D (dropout on the
  iteration loop without any consensus pooling) actively regresses below the
  no-INC baseline, ruling out "it's just stochastic regularisation".
- **The channel's effect on final WR is modest.** A 2-point gap between C (0.822)
  and E (0.802) suggests most of the sustained-coordination benefit is
  *reachable* by an intervention that does not require the channel at execution.
  That's the hypothesis SAAL is testing.
- **SMAX is silent on the Hanabi cheating question.** SMAX has no
  info-restriction rule, so a load-bearing channel on SMAX is neither evidence
  for nor against Hanabi cheating. The discriminating test is Hanabi zero-consensus
  + cross-play, deferred to Stage 8.

---

## 7. Experiment plan

### 7.1 Execution order

The ordering is deliberately staged to run the cheap, informative thing first
and defer the expensive thing until we know what we are measuring:

1. **SAAL pair-cos logging pass** (Stage 4, SMAX 3m, 1 seed, no loss yet) — cheap,
   measures baseline `pair_cos` so we can pick sensible loss weights.
2. **SAAL SMAX 3m validation** (Stage 5, 1 then 3 seeds) — the minimum
   experiment that tells us whether SAAL has any effect at all.
3. **SAAL SMAX main matrix** (Stage 6, 3 maps × 4 methods × 3 seeds) — the
   coverage experiment. Locks SAAL's SMAX story.
4. **SAAL Hanabi port** (Stage 7) — expensive, deferred until SAAL is
   demonstrated on SMAX. Hanabi-specific event detector added here.
5. **INC Hanabi discriminating test** (Stage 8) — the three-cell H-A/H-C/H-E
   matrix plus cross-play. Produces the cheating-signature result regardless of
   whether the verdict is "INC cheats" or "INC doesn't transfer". Bounded
   compute — H-C and H-E only, reusing Stage 7's H-A baseline.

**Stage 8 can be demoted to "end of paper" if time runs out.** If SAAL on
Hanabi works cleanly, the paper has its spine without any INC Hanabi numbers.
The Hanabi INC cells become "nice to have" rather than "must run."

### 7.2 Main results matrix (after all stages)

| Method | SMAX 3m | SMAX 2s3z | SMAX 5m_vs_6m | Hanabi 2p | Hanabi 4p |
|---|---|---|---|---|---|
| GRU-MAPPO | ✓ | ✓ | ✓ | ✓ | ✓ |
| CTM-MAPPO, `K=1` | ✓ | ✓ | ✓ | ✓ | ✓ |
| CTM-MAPPO, `K=K*` | ✓ | ✓ | ✓ | ✓ | ✓ |
| **CTM-MAPPO + SAAL, `K=K*`** | ✓ | ✓ | ✓ | ✓ | ✓ |
| CTM-MAPPO + INC, `K=K*` | ✓ | ✓ | ✓ | ✓ + zero-consensus + cross-play | ✓ |

`K*` is chosen in the Stage 6 sweep; expected 3–5. 3 seeds per SMAX cell, 3
seeds per Hanabi cell.

The **headline comparison** for the paper is *CTM-MAPPO + SAAL* vs *CTM-MAPPO
`K=K*`*. Same compute, same architecture, only difference is the training-time
alignment loss. If SAAL doesn't beat this control, the paper doesn't work.

The **secondary comparison** is *CTM-MAPPO + SAAL* vs *CTM-MAPPO + INC*. This
tells us how much performance is lost when we go from an execution-time channel
(INC) to a training-time signal (SAAL). On Hanabi this comparison is
asymmetric: SAAL plays legally, INC is under cheating scrutiny — the question
there is whether SAAL can match INC's *self-play* score without needing the
channel.

### 7.3 Ablations (Stage 9)

- **SAAL α/β sensitivity curve** — small grid, SMAX 3m.
- **SAAL event-detector widening** — swap focus-fire-only for {focus-fire,
  grouping, enemy-kill}, SMAX 2s3z.
- **SAAL under no parameter sharing** — confirms the effect is cross-agent
  alignment, not shared-param regularisation.
- **INC pooling type** (mean / attention / gated) — only if INC survives Stage 8.
- **INC iteration count** `K ∈ {1, 2, 3, 5}`.
- **Consensus dropout rate** — follow-up to Stage 2.
- **Decentralised-at-test INC** — train centralised, eval with observable-only
  pools. Measures cost of strict deployment decentralisation.
- **No-sync ablation.** Replace `compute_synchronisation` with a linear
  projection of the flat activated-state trace, keeping SAAL on. Tests whether
  the benefit comes specifically from the sync signal vs any internal
  representation.

### 7.4 Robustness & negative controls (Stage 11)

- **SAAL on a task that does not require coordination** (single-agent MPE or
  independent-reward setting). SAAL should not help. If it does, the story is
  wrong.
- **Compute-matched GRU.** Same FLOPs per step as CTM + SAAL. Defends against
  "you just have more compute."
- **Parameter-count-matched GRU.**
- **Fresh seeds for headline numbers**, not used during development.
- **Hanabi 4p scaling.**

---

## 8. Analysis plan — how we will look at the results

This section is deliberately detailed because "look at some plots" is where
papers hide mistakes. For each analysis: what question, how to compute, plot or
table, and what counts as a pass.

### 8.1 Learning curves (main result)

**Question:** does SAAL outperform the compute-matched CTM control, and by how
much?

**How:** for each (method, env, seed), log return (Hanabi) or WR (SMAX) every
`EVAL_INTERVAL` updates. Align curves by environment steps. Compute mean and 95%
bootstrap CI across 3 seeds.

**Plot:** Figure 4 — one subplot per env, x = env steps, y = return, one line
per method with CI band.

**Statistical test:** rank-sum (Mann-Whitney U) across seeds between SAAL and
the compute-matched control at final performance. Also compute rliable
probability-of-improvement across all seed × env cells jointly. Pass: `p < 0.05`
on SMAX 3m AND 2s3z AND rliable PoI > 0.6.

### 8.2 Alignment mechanism verification

**Question:** is SAAL actually doing what we claim — pushing `pair_cos_ff` up
and `pair_cos_nff` down relative to baseline?

**How:** log `pair_cos_ff` and `pair_cos_nff` throughout training (already
collected in Stages 4/5/6). For each (method, env), plot both over training
steps. Baseline should show the Stage 1-3 spontaneous rise; SAAL should show a
steeper and larger separation.

**Plot:** Figure 3 — two-panel (ff, nff) line plot, baseline vs SAAL, one panel
per env. Shaded region = std across seeds.

**Pass condition:** SAAL produces a larger `pair_cos_ff - pair_cos_nff` gap
than baseline, and the gap appears earlier in training.

### 8.3 SAAL → action influence probe

**Question:** does the alignment loss influence the actions the policy takes,
not just internal state?

**How:** at eval, for each env step compute the action distribution under the
SAAL-trained model and under the baseline. Measure KL divergence. Bucket by
coord vs non-coord step.

**Plot:** bar chart of mean KL on coord-event vs non-coord steps.

**Pass:** KL on coord steps statistically larger than on non-coord steps.

### 8.4 INC Hanabi cheating signature (Stage 8)

**Question:** does INC on Hanabi post a strong self-play score with a collapsing
cross-play gap?

**How:** train 3 seeds of H-C. At eval, pair each seed with itself (self-play)
and with each other seed (cross-play). Report mean ± CI for self-play and
cross-play separately.

**Plot:** Figure 6 — two bars per method (self-play, cross-play), plus the
zero-consensus H-E bar.

**Pass condition (for "INC cheats" verdict):** cross-play mean ≥ 3 points below
self-play mean AND H-E mean within 1 point of H-A baseline.

### 8.5 Compute and wall-clock accounting

**Question:** what does SAAL and INC cost relative to the control?

**How:** wall-clock per PPO update, peak GPU memory, analytical FLOPs. SAAL
adds one `einsum` plus the pair-cos reduction — negligible. INC adds an
`O(N² × synch_size)` pool per iteration.

**Plot:** small table, method × (FLOPs/step, wall clock/step, peak memory).

### 8.6 Aggregate reliability (rliable)

Use `rliable` for IQM + probability-of-improvement + performance profile across
all (env × seed × method) cells. Pass condition: SAAL's IQM strictly above the
CTM control's IQM with non-overlapping CIs and PoI > 0.6.

### 8.7 Negative-result verification

On a task that doesn't need coordination, SAAL should be neutral. If it helps,
the story is wrong. If it clearly hurts, the β term is over-tuned for that
setting.

---

## 9. Relationship to prior work

- **Sync-based coordination in MARL.** Stage 1-3 is the direct predecessor.
  Elsewhere, methods that analyse internal representations for coordination are
  rare; most MARL work operates at the action or value level.
- **Auxiliary losses for MARL coordination.** The closest related line is
  Simplified Action Decoder (SAD) and other Hanabi-specific methods that add
  training-time signals to encourage interpretable conventions. SAAL is similar
  in spirit (train-time signal, clean execution) but operates on the internal
  sync variable rather than on a separate auxiliary head.
- **CommNet, TarMAC, IC3Net, and other learned-comm methods.** These learn a
  dedicated message head and require an explicit communication channel at
  inference. SAAL has neither. INC has an internal-iteration channel but still
  no learned message head — the "message" is the CTM's own internal variable,
  pooled fixed. In info-restricted games like Hanabi, learned-comm methods and
  INC are both under cheating scrutiny; SAAL is not.
- **QMIX and value-decomposition methods.** Mix value functions centrally at
  training, keep per-agent policies independent at execution. SAAL is
  orthogonal — it modifies the *policy internal state* via an aux loss, not the
  value function.
- **Zero-shot coordination (ZSC) / ad-hoc teamwork.** Cross-play evaluation is
  the standard ZSC diagnostic. Our Stage 8 INC cheating test borrows directly
  from this literature.
- **CTM.** SAAL and INC both inherit the CTM's internal-iteration loop and sync
  readout. SAAL does not change the CTM forward pass at all. INC changes only
  the between-iteration inputs.

---

## 10. Risks to the research direction

| Risk | What it would look like | Mitigation |
|---|---|---|
| Baseline `pair_cos` is already saturated | Stage 4 logging pass shows `pair_cos_ff` and `pair_cos_nff` are already near 1.0 | Reformulate SAAL as subspace alignment; see [implementation plan](../Implementation_Plans/ITERATIVE_NEURAL_CONSENSUS_PLAN.md) Stage 4 decision rule |
| SAAL causes agent-invariant sync collapse | Policy entropy collapses mid-training; `pair_cos_nff` saturates to 1.0 | β term keeps push tension; Stage 5 gradient-flow test; retune or switch to subspace variant |
| Focus-fire is too sparse a signal | `ff_frac` logged in Stage 4 is very low (say < 5%) and SAAL has no effect | Widen event detector (grouping, enemy-kill) or advantage-gated fallback — both listed as Stage 5 bailouts |
| SAAL helps SMAX but not Hanabi | Stage 7 fails; alignment signal is SMAX-specific | Hanabi-specific event detector iteration (hint-then-play) before giving up; else paper is SMAX-only on SAAL with Hanabi reserved for the INC negative result |
| INC Hanabi test is ambiguous | H-C ≫ H-A but cross-play holds and H-E ≈ H-C | Report the ambiguity in the paper rather than picking a side; the ambiguity is itself a useful result |
| Hanabi compute budget explodes | Full 3-cell × 3-seed matrix is > 60 GPU-hours | Staged: single-seed H-C first, full matrix only if signal exists |
| INC survives Hanabi cleanly | Cross-play holds, H-E drops a lot, H-C is a genuine win | Paper expands to "SAAL and INC are complementary"; INC regains partial headline status as a legitimate comparison method rather than a cautionary case |

---

## 11. Out of scope (written down so we don't drift)

- Learned communication heads. We deliberately contrast against them.
- Per-agent adaptive iteration count (halting / ponder mechanism). Future work.
- Modifying the centralised critic. Critic stays GRU.
- Heterogeneous agent weights. Parameter sharing throughout.
- Offline / imitation learning on Hanabi human replays. Pure self-play only.
- Transfer to real-world robotics or language tasks. Not claimed.

---

## 12. Where to find things

- **Implementation plan (stage-by-stage, file-level):**
  [Implementation_Plans/ITERATIVE_NEURAL_CONSENSUS_PLAN.md](../Implementation_Plans/ITERATIVE_NEURAL_CONSENSUS_PLAN.md)
- **Prior sync analysis that motivates this direction:**
  [docs/old/stage1to3_results.md](old/stage1to3_results.md)
- **Stage 2.1 findings (INC channel is load-bearing on SMAX):**
  [docs/inc_stage2_1_findings.md](inc_stage2_1_findings.md)
- **Axis convention note:** [docs/inc_axis_convention.md](inc_axis_convention.md)
- **Hanabi env notes:** [docs/hanabi_env_notes.md](hanabi_env_notes.md)
- **CTM core implementation:** [smax_ctm/ctm_jax.py](../smax_ctm/ctm_jax.py)
- **MAPPO training loops:**
  - SMAX CTM: [smax_ctm/train_mappo_ctm.py](../smax_ctm/train_mappo_ctm.py)
  - SMAX GRU: [smax_ctm/train_mappo_gru.py](../smax_ctm/train_mappo_gru.py)
  - Hanabi CTM: [smax_ctm/train_mappo_ctm_hanabi.py](../smax_ctm/train_mappo_ctm_hanabi.py)
  - Hanabi GRU: [smax_ctm/train_mappo_gru_hanabi.py](../smax_ctm/train_mappo_gru_hanabi.py)
- **Sync analysis scripts:** [smax_ctm/analyse_sync.py](../smax_ctm/analyse_sync.py)
- **Hanabi env (ported from upstream JaxMARL):** [jaxmarl/environments/hanabi/](../jaxmarl/environments/hanabi/)
- **Hanabi env contract tests:**
  [smax_ctm/test_and_logger/run_hanabi_tests.py](../smax_ctm/test_and_logger/run_hanabi_tests.py)
- **INC unit tests:** [smax_ctm/tests/test_inc.py](../smax_ctm/tests/test_inc.py)

**Note on per-benchmark scripts.** SMAX and Hanabi are wired through two
separate training files for each of {CTM, GRU} rather than one branching
script. The benchmarks differ in enough small places — `get_avail_actions` vs
dict-returning `get_legal_moves`, SMAX world-state + agent-id one-hot vs
Hanabi-obs concatenation, `SMAXLogWrapper` vs generic `LogWrapper`, win-rate vs
score-out-of-25 logging — that a single branching script would obscure both
code paths. The `CTMCell`, `ScannedCTM`, and `AgentConsensus` modules live in
shared files and are imported by both benchmarks, so the actual method is
defined once. Anything SAAL- or INC-related that must be implemented is landed
in *both* CTM training scripts at the appropriate stage — SMAX in Stage 5,
Hanabi in Stage 7.

---

## 13. A note on honesty

Section 8 exists to make it hard to fool ourselves. A method that passes 8.1
but fails 8.2 is a method that *happens* to train better — not a method that
does what we say. If that happens, the honest thing is to report it
("training-time sync-alignment regularisation improves coordination, but we
cannot confirm the alignment pathway is causal") and either rewrite the story
or drop the claim. We are not writing a paper called "here is a thing that
trains marginally better"; we are writing a paper about a specific mechanism.

The pivot documented in §4 is itself an example of this principle. INC
produced a real SMAX win, but the reviewer question "is this legal on Hanabi?"
is load-bearing for the paper's honesty and could not be answered without
rethinking the primary contribution. SAAL is the pivoted answer, and the INC
Hanabi test is the controlled comparison that keeps the story honest about
*why* the pivot happened.
