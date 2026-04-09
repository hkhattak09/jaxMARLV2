# Research Direction: CTM for Multi-Agent RL

## Current Status (2026-04-09)
- CTM-MAPPO JAX port is working and training on SMAX 3m
- 3-minute training run reaches ~84% win rate, return 1.82
- Both CTM and GRU baselines learn successfully
- CTM config: d_model=128, d_input=64, iterations=1, n_synch_out=32, memory_length=5, deep_nlms=True

---

## Primary Direction: Cross-Agent Neural Synchronisation

### Core Question
Do CTM agents naturally develop correlated internal neural dynamics when they coordinate? If yes, this is a novel finding unique to CTM that no other MARL architecture can produce.

### Why This Direction
1. **Genuine novelty:** CTM's synchronisation mechanism (pairwise temporal correlations between neurons) has no equivalent in GRU/LSTM. Measuring cross-agent correlations of this signal is something fundamentally new.
2. **Neuroscience grounding:** Biological neural synchrony (gamma oscillations, phase-locking between brain regions) is a known mechanism for inter-region coordination. CTM's synchronisation is mathematically analogous.
3. **Discovery paper, not engineering paper:** We're asking "does this phenomenon exist?" not "does our method get +2%?" Discovery papers have higher impact if the finding is real.
4. **Zero implementation risk for initial validation:** The first step is pure analysis on an already-trained model. No architecture changes needed to check if the signal exists.

### Framing: Synchronisation as Implicit Communication
Since agents share CTM weights (parameter sharing), cross-agent sync patterns function as an **implicit communication channel** — agents "speak the same language" because they share neural dynamics. This positions against explicit communication methods (QMIX, CommNet, TarMAC) with zero communication overhead. The paper's core ablation (G.3: NLMs + sync together are essential) provides mechanistic justification — sync isn't just a correlate, it's a functional component the network relies on.

### What the Paper Would Look Like (if analysis succeeds)
**Title direction:** "Coordination Through Neural Synchronisation in Multi-Agent Reinforcement Learning"

- Introduce CTM-MAPPO as a new MARL backbone
- Show competitive baseline performance
- **Key finding:** agents that coordinate show correlated synchronisation patterns — they "think in sync"
- **Mechanistic evidence:** no-sync ablation degrades both performance AND cross-agent coupling (proving sync is necessary, not just correlated)
- **Iterations analysis:** with iterations>1, agents' sync vectors converge within-step during coordination — they reach consensus through iterative internal processing
- Analysis: sync correlates with focus fire, grouping, and precedes coordination events
- Implicit communication framing: sync as zero-overhead coordination signal vs. explicit message-passing methods
- Optional method extension: auxiliary loss encouraging cross-agent sync coherence during training (zero overhead at test time)
- Interpretability: visualize sync matrices to understand what agents are "thinking about" when coordinating

### Validation Step (MUST DO FIRST)
Run the analysis script described in `analysis_plan.md` on the trained 3m model. Specifically:
1. Extract per-agent synchronisation vectors and activated traces during evaluation episodes
2. Compute pairwise cross-agent correlation of sync vectors over time
3. Check if correlation spikes during coordination events (focus fire, kills)
4. Check if won episodes have higher cross-agent sync than lost episodes

**Decision criterion:** If cross-agent sync correlation shows statistically significant correlation with coordination events (p < 0.05) or clear visual structure in the timeseries, proceed with this direction. If it's noise, fall back.

---

## Fallback Direction: Adaptive Computation ("Think Fast, Think Slow")

### Core Idea
Make CTM's `iterations` parameter dynamic per-agent per-timestep. A learned halting head decides how many internal thinking iterations each agent needs based on the situation.

### Why This Is the Fallback (Not Primary)
- It's a **good, safe** paper — clean mechanism, clear experiments, nice narrative
- But it's more "engineering" than "discovery"
- Still publishable at a good venue, just lower ceiling than a discovery paper

### Revised Assessment After Reading Full Paper
This fallback is **stronger than originally thought.** The paper's native loss function L = (L^t1 + L^t2)/2, where t1=argmin(loss) and t2=argmax(certainty), is already a built-in adaptive computation mechanism. With iterations>1, this loss directly identifies which iteration was most useful — no need for the Graves (2016) ACT halting mechanism. This deflects the "you just applied ACT to a new setting" reviewer criticism entirely: we're using CTM's *native* mechanism, not grafting on an external one.

### Implementation Plan (if we need it)
**Option A (preferred): Use CTM's native dual-loss.**
1. Set iterations=3–5
2. Collect per-iteration outputs (policy logits at each iteration)
3. CTM loss picks t1=argmin(policy_loss) and t2=argmax(policy_certainty) across iterations
4. At test time, use the certainty-based output — agents naturally "think longer" when uncertain
5. This is fully differentiable and compatible with `jax.lax.scan`

**Option B (original, less preferred): External halting head.**
1. Add a halting head to CTMCell: small Dense layer that outputs a scalar halt probability after each iteration
2. Use the "remainder" approach (not true early stopping) — all agents run max iterations, but halting probabilities create a weighted combination of per-iteration outputs
3. Add ponder cost to loss: `lambda_ponder * mean(num_iterations_used)` to penalize unnecessary computation
4. This is fully compatible with `jax.lax.scan` — the scan runs max iterations, masking handles the rest

### What the Paper Would Look Like
**Title direction:** "Thinking Fast and Slow in Multi-Agent Systems: Adaptive Computation for Cooperative RL"

- CTM-MAPPO baseline
- Add adaptive halting mechanism
- Show agents learn to think more during complex situations (engagements) and less during simple ones (movement)
- Analysis: plot iterations-used vs tactical complexity, show per-agent iteration heatmaps
- Bonus: heterogeneous maps (2s3z) where different unit types learn different computation budgets
- Bonus: robustness analysis (CTM degrades gracefully under neuron dropout vs GRU)

---

## Ideas Considered But Not Pursued (and why)

### Neuro-Symbolic Grounding (Gemini G1)
Forcing specific neurons to track heuristic values. Rejected because it's essentially feature engineering in disguise — reviewers will say "how is this different from concatenating features to the observation?"

### Thought-Sharing Communication (Gemini G2)
Broadcasting sync matrices between agents. Interesting concept but the sync matrix is 528 floats — 4-8x more expensive than standard MARL communication (64-128 dims). The bandwidth problem undermines the story.

### Dynamic Decay (Gemini G3)
Making temporal decay observation-conditioned. Elegant and clean, but too thin for a standalone paper. Could be a section/ablation within a larger paper.

### Orthogonal Sync for Role Discovery (Gemini G4)
Contrastive loss forcing different unit types to have orthogonal sync patterns. The forced loss undermines the "emergence" claim. Natural emergence would be stronger but can't be guaranteed. Better as analysis than method.

### Theory of Mind / Opponent Modeling (Gemini G5)
Predicting opponent's sync matrix. Scope creep — hard to make work cleanly, and you don't observe opponent internal state at test time in SMAX.

### State Trace Credit Assignment (M3)
Using CTM traces for temporal credit assignment. Too incremental — hard to disentangle "better credit assignment" from "critic just has more information."

---

## Scaling Plan (beyond 3m)

Once we have a direction locked in, the experiment plan for a full paper needs:
1. **3m** — simplest, for validation and debugging (DONE)
2. **2s3z** — heterogeneous (Stalkers + Zealots), tests whether different unit types develop different sync/computation patterns
3. **3s5z** — larger heterogeneous, tests scalability
4. **5m_vs_6m** — asymmetric (harder), tests whether CTM helps in disadvantaged scenarios
5. **SMACv2** (if time permits) — randomized scenarios, the current gold standard for generalization

---

## Key Files
- `smax_ctm/ctm_jax.py` — CTM cell implementation (CTMCell, ScannedCTM, NLM, Synapses, compute_synchronisation)
- `smax_ctm/train_mappo_ctm.py` — Training script (ActorCTM, CriticRNN, make_train)
- `smax_ctm/eval_smax.py` — Evaluation/rendering script
- `model/smax_mappo_ctm_actor.pkl` — Trained checkpoint (contains actor_params + config)

## Implementation Verified Against Paper (2026-04-10)
Checked `ctm_jax.py` against Darlow et al. (arXiv:2505.05522v4) Appendix G.6 (RL-specific CTM):
- ✅ No attention (direct concatenation)
- ✅ 2-layer synapse + GLU + LayerNorm
- ✅ Learned initial traces (`start_trace`, `start_activated_trace`)
- ✅ Dense pairing (last n_synch_out neurons, all triu pairs)
- ✅ Sliding window sync with learnable exponential decay
- ✅ Deep NLMs (2-layer SuperLinear + GLU)
- ✅ iterations=1 (paper uses T=1–5)
- ✅ PPO-based training (we extend to MAPPO with centralised critic — novel)

No implementation changes needed. The paper's RL experiments are single-agent POMDPs only — multi-agent is our novel extension.

## CTM Architecture Summary (for quick reference)
```
Observation -> CTMBackbone (2x Dense+GLU+LN) -> features (d_input=64)
                                                    |
For each iteration (currently 1):                   v
  [features, last_activated_state] -> Synapses (2x Dense+GLU+LN) -> new_state (d_model=128)
  state_trace <- shift window, append new_state     (128, 5)
  state_trace -> NLM (SuperLinear+GLU, deep) -> activated_state  (128,)
  activated_state_trace <- shift window, append     (128, 5)

After iterations:
  activated_state_trace -> compute_synchronisation -> synch vector (528,)
  synch -> ActorHead (3x Dense+ReLU) -> action logits
```

Synchronisation computation:
- Select last n_synch_out=32 neurons from activated_state_trace
- Compute all pairwise products over the time window (32*33/2 = 528 pairs)
- Weight by learnable exponential decay over the memory window
- This measures temporal correlation between neuron pairs
