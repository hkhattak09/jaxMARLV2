# Stage 1-3 Results: Cross-Agent Synchronisation Analysis

## Run Details
- **Date:** 2026-04-09
- **Environment:** Colab (GPU), JAX with CUDA
- **Model:** CTM-MAPPO trained on SMAX 3m (~84% win rate, return 1.82)
- **CTM config:** d_model=128, d_input=64, iterations=1, n_synch_out=32, memory_length=5, deep_nlms=True
- **Episodes collected:** 20 (seed=42, greedy actions)
- **Note:** 272 correlation values were undefined (constant vectors) and stored as NaN

---

## Global Summary Statistics

| Metric | Value |
|---|---|
| Mean cross-agent sync correlation | 0.7089 +/- 0.1896 |
| Mean cross-agent obs correlation | 0.9206 +/- 0.1454 |
| Sync minus obs (delta) | -0.1651 |

Sync correlation is substantially lower than observation correlation. The CTM is **decorrelating** agent representations relative to raw input similarity, not amplifying it. This means the network learns agent-specific internal dynamics despite parameter sharing.

---

## Event-Conditional Analysis (Core Test)

Cross-agent sync correlation during coordination events vs outside them, aggregated across 20 episodes:

| Event Type | During | Outside | Delta | Num Events | t-stat | p (one-sided) | Significant? |
|---|---|---|---|---|---|---|---|
| **Focus fire** | 0.762 | 0.694 | +0.068 | 119 | 2.874 | **0.005** | Yes |
| **Grouping** | 0.786 | 0.570 | +0.216 | 184 | 6.648 | **0.000003** | Yes |
| **Enemy kill** | 0.760 | 0.710 | +0.050 | 40 | 2.092 | **0.025** | Yes |

All three event types pass the p < 0.05 decision criterion. Grouping is the strongest signal (delta = +0.22, p ~ 3e-6).

---

## Timeseries Observations

From the per-episode Sync vs Obs Correlation plots (episodes 0, 1, 2):

1. **Early phase (t=0-6):** Sync correlation starts moderate (0.6-0.85) and climbs to 0.85-0.95. This corresponds to agents approaching and engaging enemies together. Obs correlation stays near 1.0 throughout since agents see similar observations in 3m.

2. **Sharp drop (t=6-8):** Sync correlation drops sharply by 0.3-0.4 in a few timesteps. This coincides with agents dying or the battle breaking apart (enemy kills, agents taking different actions). Obs correlation also drops at this point but less dramatically.

3. **Late phase (t=8+):** Sync correlation stays low (0.4-0.55) while obs correlation partially recovers. With fewer agents alive, the surviving agents' internal states diverge despite similar observations.

**Key pattern:** Sync rises during coordinated engagement and collapses when coordination breaks down. This temporal structure is distinct from observation similarity, which stays high throughout.

---

## Heatmap Observations

The pairwise sync correlation heatmap (episode 0) shows:
- Pair 0-2 and 1-2 maintain higher sync than pair 0-1 in the early phase
- All pairs drop in the late phase
- The heatmap shows clear temporal structure, not random noise

---

## Interpretation

### What the results mean

1. **The signal is real.** Cross-agent sync correlation is significantly higher during coordination events than outside them. This passes the pre-specified decision criterion.

2. **The decorrelation finding is interesting.** Sync < obs means the CTM doesn't just parrot input similarity. It builds agent-specific internal representations that *re-synchronize* during coordination. This is a richer finding than "correlated inputs produce correlated outputs."

3. **Grouping is the dominant signal.** The largest delta (+0.22) is for spatial grouping events. When agents are physically close and presumably acting together, their internal neural dynamics align most strongly. This maps to the neuroscience analogy: brain regions that coordinate show increased phase-locking.

4. **The temporal pattern tells a story.** Sync rises during engagement, peaks at coordination, and collapses when the team fractures. This is consistent with sync serving as an implicit coordination mechanism rather than being an artifact of shared weights + similar inputs.

### Caveats

1. **Parameter sharing confound.** All agents share CTM weights, so similar observations necessarily produce somewhat similar internal states. The obs-correlation control partially addresses this (sync shows different temporal structure than obs), but a **shuffle control** (permuting agent identities across timesteps) would strengthen the argument.

2. **3m is easy.** At 84% win rate with only 3 homogeneous agents, there may not be enough variance in outcomes to test win/loss conditional differences. Harder maps (5m_vs_6m, 2s3z) would provide more signal.

3. **Causal direction unclear.** Sync correlates with coordination, but we don't yet know if sync *causes* coordination or is just a byproduct. The no-sync ablation (Stage 5) is needed to establish causality.

4. **272 undefined correlations.** These come from constant sync vectors (zero variance), likely from dead agents. Not a problem for the analysis but worth noting.

---

## Decision: Proceed with Primary Direction

The results satisfy the decision criterion from `analysis_plan.md`:
> If cross-agent sync correlation shows statistically significant correlation with coordination events (p < 0.05) or clear visual structure in the timeseries, proceed with this direction.

Both conditions are met. **Proceed to Stages 4-6.**

### Immediate next steps

1. **Stage 4 — Time-lagged cross-correlation:** Does sync rise *before* coordination events? A predictive signal (sync leads action) would be the strongest possible finding.

2. **Stage 5a — No-sync ablation:** Train a model without `compute_synchronisation`. If sync correlation drops AND performance degrades, that's mechanistic evidence that sync is functional, not epiphenomenal.

3. **Stage 5b — Iterations=3 model:** Train with more internal thinking steps. If within-step sync convergence is observed, that demonstrates agents reaching consensus through iterative processing.

4. **Stage 6 — Publication figures:** Generate clean versions of the timeseries, heatmaps, and event-conditional bar charts.
