# Issues, Root Causes, and Fixes

A running log of non-obvious problems encountered during the iterative coupling
implementation. Ordered chronologically. Each entry explains what was observed,
why it happened, and what was done.

---

## Issue 1: Dead agent masking — GRU reset does NOT zero rnn_out

**Stage:** 2 → 3 transition (caught during Stage 3 plan review)

**Observed:** Stage 2 plan stated "dead agents handled naturally — GRU resets
h_j=0 → C_ij * h_j = 0." This claim was accepted as safe and nearly shipped.

**Why it was wrong:** The GRU reset zeroes the *carry* (hidden state) at the
*start* of the next timestep. After the reset, the GRU cell still runs on the
agent's (now invalid/zeroed) observation and produces a non-zero output. So
`rnn_out` for a dead agent is not zero — it's whatever the GRU produces when
fed a zeroed observation from a fresh hidden state.

**Why it matters at Stage 3:** In Stage 2 (one coupling pass), this injects
a small amount of noise into C. In Stage 3 with K iterations, the dead agent's
non-zero embedding propagates through every iteration, compounding K times into
every other agent's representation.

**Fix:** Explicit alive masking at three points:
1. Before the loop: `e = rnn_out * alive_mask[..., None]` — zero dead embeddings at init
2. Inside the loop, column mask on C: `C = C * alive_mask[:, :, None, :]` — zero coupling weight *to* dead agents
3. After each update: `e = e * alive_mask[..., None]` — re-zero dead agents so they never inject signal in the next iteration

**Lesson:** Never assume a component zeros outputs under reset. Check what the
component actually computes after the reset, not just what the reset does to
internal state. Robustness at every stage, not "probably fine" assumptions.

---

## Issue 2: Flax `@nn.compact` loop creates independent parameters, not shared weights

**Stage:** 2 → 3 transition (caught during Stage 3 plan review)

**Observed:** Stage 2 used `@nn.compact`. The Stage 3 plan initially proposed
calling `nn.Dense(...)` inside a Python `for` loop.

**Why it's wrong:** Inside `@nn.compact`, every call to `nn.Dense(...)` registers
a new set of parameters with a new name (e.g. `Dense_0`, `Dense_1`, ...). Calling
the same `nn.Dense(CH)` K times in a loop creates K *independent* parameter sets —
not shared weights. The iterative coupling hypothesis requires the same MLP to be
applied at every iteration (same parameters, different inputs), which is the
mechanism for weight-sharing and the CTM connection.

**Fix:** Switch `CriticRNN` from `@nn.compact` to `setup()` style. Define coupling
and update layers as named instance attributes (`self.couple_h`, `self.couple_out`,
`self.update_h`, `self.update_out`). Calling `self.couple_h(...)` in a loop
correctly reuses the same parameters at each iteration.

**Lesson:** In Flax, `@nn.compact` is for architectures with a fixed, non-repeated
call sequence. Any iterative computation that must share weights across iterations
requires `setup()`.

---

## Issue 3: Over-smoothing — iterative update MLP dilutes agent-specific signal

**Stage:** Stage 3 training run vs Stage 2 comparison

**Observed:** A distinctive two-phase learning pattern:
- Stage 3 reaches 0.70 WR by step ~500k; Stage 2 is at 0.16 at the same point
- After ~500k, Stage 3 slows dramatically; by step ~1M Stage 2 has overtaken it (0.84 vs 0.72)
- Stage 3 eventually converges to 0.99 but takes ~5M steps vs Stage 2's ~1.5M
- Actor grad norm in Stage 3 at ~1M: 0.088 (weak). Stage 2: 0.140 (strong)
- Stage 3 entropy collapses faster (to ~0.35) vs Stage 2 (~0.47)

**Why it happens:** The update step in Stage 3 was:
```python
e = relu(update_out(relu(update_h(concat(e, context)))))
```
This **replaces** e with a new vector that mixes the agent's own embedding with
the coupling context. After K=2 rounds, each agent's embedding has been blended
with its neighbours' signals twice. Early in training, the coupling matrix C is
noisy so context is weak — e survives mostly intact, providing a regularizing
signal that accelerates early learning. But as training progresses and C sharpens
(6h_vs_8z has real coordination structure), context is dominated by 1-2 strongly
coupled neighbours. The MLP pushes e toward a blend of itself and those neighbours.

The value head receives e^K that no longer cleanly represents "what is agent i
doing" — it represents "what is the team doing near agent i." This is useful for
credit assignment but makes precise V_i prediction harder. The weakened value
signal explains the smaller actor grad norms and the plateau.

This is the standard **over-smoothing** failure mode in graph neural networks:
non-residual message passing with K rounds converges node representations toward
a neighbourhood mean. K=2 is mild but measurable on a 6-agent coordination task.

Stage 2 avoids this by design — rnn_out is never modified, and the value head
receives `concat(rnn_out, context)` giving it direct access to both signals
without any blending.

**Fix:** Residual connection in the update step:
```python
delta = relu(update_out(relu(update_h(concat(e, context)))))
e = e + delta
e = e * alive_mask[..., None]
```
The update MLP learns to **add** coordination signal to the existing embedding
rather than **replace** it. Each agent's own temporal signal is preserved through
all K iterations.

**Lesson:** Non-residual iterative message passing over-smooths representations.
Any time you run K rounds of `e = f(e, context)` with shared weights, use
`e = e + f(e, context)` instead. The residual is not optional for K >= 2.

---

## Issue 4: Residual update MLP corrupts embeddings at initialization

**Stage:** Stage 3 with residual connection

**Observed:** Adding the residual connection fixed the plateau (reaches 0.99)
but made early learning *worse* than both Stage 2 and Stage 3 without residual.
Late-stage behaviour resembled Stage 2 but slower.

**Why it happens:** At initialization, the update MLP computes a random
projection of `concat(e, context)` (256 dims → 128). With residual, this random
vector is added to `e` at each iteration. With K=2 this corrupts `e` twice before
the value head sees it. Stage 2 never modifies `rnn_out` — the value head always
sees clean GRU output. Stage 3 with residual starts by adding random noise to
`rnn_out` twice, making V_i prediction harder in early training and slowing the
initial learning curve below Stage 2.

**Fix:** Near-zero initialization for `update_out` output projection:
```python
self.update_out = nn.Dense(
    agent_embed_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
)
```
With the output layer near-zero at init, `delta ≈ 0` and the network behaves
like Stage 2 (clean embeddings, fast early learning). As the coupling matrix
sharpens and the update MLP has real signal to work with, it learns to grow
the residual. This is the standard technique for residual branches in deep networks.

**Lesson:** When adding a residual branch `e = e + f(e)`, always initialize the
output projection of `f` to near-zero. Otherwise the branch starts as random
noise rather than identity, degrading early training.

---

## Issue 5: Near-zero init creates chicken-and-egg — residual branch never activates

**Stage:** Stage 3 with residual + near-zero update_out init

**Observed:** e_norm pre/post difference was ~0.01–0.02 on vectors of magnitude
~5.5 (0.3% change) throughout 3M steps on smacv2_10_units. The update MLP was
completely dead. Near-zero init fixed early corruption (Issue 4) but the residual
never learned to contribute real signal.

**Why it happens:** Near-zero init → delta ≈ 0 at start → value head learns to
predict V from clean `e` alone → gradient signal to update_out is weak (value
head doesn't need delta) → delta stays near zero. The two failure modes are exact
opposites and both fail:
- Normal init: delta corrupts e early, value head can't learn (Issue 4)
- Near-zero init: value head learns without delta, delta never grows (Issue 5)

**Root cause:** Asking `e` to serve two conflicting purposes — be a clean
agent-specific signal for the value head AND be the evolving coupling
representation. Any modification of e for coupling either corrupts the value
head (normal init) or gets suppressed by it (near-zero init).

**Fix:** Separate the two tracks. `e_orig` is the clean GRU output, never
modified. `e_coup` starts as `e_orig` and evolves through K coupling iterations.
The value head receives `concat(e_orig, e_coup)` explicitly:

```python
e_orig = rnn_out.reshape(...) * alive_mask[..., None]  # never modified
e_coup = e_orig  # separate copy for iterative coupling

for _ in range(K):
    C = coupling_matrix(e_coup)
    context = einsum(C, e_coup)
    delta = relu(update_out(relu(update_h(concat(e_coup, context)))))
    e_coup = e_coup + delta
    e_coup = e_coup * alive_mask[..., None]

value_input = concat(e_orig, e_coup)  # (T, E, A, 2D)
```

With this: update_out uses normal init (value head has e_orig as fallback, no
corruption), gradient to update_out is always strong (e_coup contributes
directly to value head), no chicken-and-egg. value_h1 input is 2D (same
capacity as Stage 2).

**Lesson:** When a module must be both (a) transparent at initialization and
(b) strongly learned later, near-zero init alone is insufficient if the main
path can solve the task without it. The fix is to give the learning target
(value head) explicit access to the module's output so gradient always flows
through it, regardless of the main path.

---

## Issue 6: One-sided residual — relu after update_out forces delta ≥ 0

**Stage:** Stage 3 two-track architecture on smacv2_10_units

**Observed:** Stage 3 with K=3 reaches ~0.37–0.42 WR at 3M steps on smacv2_10_units.
K=2 was worse than K=3, and both are below Stage 2.
e_coup pre/post norms: ~5.6/6.1 — delta is ~8-9% of magnitude (residual is live).
More K iterations helps, but Stage 3 never catches Stage 2.

**Why it happens:** The delta computation ended with:
```python
delta = self.update_h(update_in)
delta = nn.relu(delta)
delta = self.update_out(delta)
delta = nn.relu(delta)   # ← wrong
e_coup = e_coup + delta
```
The final `relu` forces `delta >= 0` element-wise. `e_coup` can only grow
from `e_orig` — it can never subtract signal from any dimension. After K
iterations, `e_coup = e_orig + Σ(positive deltas)`.

Stage 2's context = `C @ rnn_out` can be positive or negative (because
`rnn_out` values span ±). The value head in Stage 2 can both amplify and
suppress dimensions. Stage 3's value head received `concat(e_orig, e_coup)`
where `e_coup ≥ e_orig` element-wise — strictly less expressive.

This also explains why K=3 > K=2: more iterations accumulate more positive
signal, which is directionally correct but still one-sided. More K helps but
never reaches the bidirectional capacity of Stage 2's direct context.

**Fix:** Remove the `relu` after `update_out`. Standard residual branches
have no activation on their output before addition — the activation is
*inside* the branch (after `update_h`), not capping the branch output. The
delta is then unconstrained and can push `e_coup` in any direction.

**Lesson:** Residual branches (`e = e + f(e)`) must not apply an activation
to the output of `f` before addition. The activation on the final projection
is what kills bidirectionality. Only activations on intermediate layers
(inside `f`) are correct.

---
