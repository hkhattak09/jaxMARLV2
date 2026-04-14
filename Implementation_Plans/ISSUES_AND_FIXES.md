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
