# Cross-Agent Neural Synchronisation Analysis Plan

## Goal
Determine whether CTM agents in a trained MAPPO model develop correlated internal neural dynamics when coordinating. If they do, this is a novel finding that forms the basis of a strong paper. If they don't, we fall back to adaptive computation (see `research_direction.md`).

## What We're Measuring
The CTM produces two key internal states per agent per timestep:
- `activated_state_trace`: shape `(d_model, memory_length)` = `(128, 5)` — the temporal trace of activated neuron states
- `synch`: the synchronisation vector output, shape `(N_SYNCH_OUT*(N_SYNCH_OUT+1)/2,)` = `(528,)` — pairwise temporal correlations between the last 32 neurons

We want to compute correlations of these quantities **across agents** (Agent 0 vs Agent 1 vs Agent 2 in 3m) and see if they relate to coordination events (focus fire, grouping, winning/losing).

---

## Implementation Stages

### Stage 1: Modified CTM Cell That Exposes Internals

**File:** `smax_ctm/ctm_jax.py`

The current `CTMCell.__call__` returns `(new_carry, synch)`. We need a version that also returns the internal state for analysis. **Do NOT modify the existing cell** — create a wrapper or a separate analysis function.

**What to extract per agent per timestep:**
1. `activated_state_trace` — shape `(128, 5)` — the full neuron trace
2. `synch` — shape `(528,)` — the synchronisation vector (already returned)
3. `state_trace` — shape `(128, 5)` — raw (pre-activation) trace

**Approach:** Create a new module `CTMCellWithDiagnostics` that wraps CTMCell and additionally outputs the traces in a dict alongside synch. Or simpler: modify CTMCell to optionally return diagnostics via a flag, or just make the analysis script manually unroll the CTM steps instead of using `nn.scan`.

**Recommended approach:** Since `nn.scan` makes it hard to extract per-step internals, the analysis script should **manually unroll** the CTM cell over timesteps (no scan), collecting traces at each step. This is fine because we're not training, just running inference on a few episodes.

```python
# Pseudocode for manual unroll
carry = CTMCell.initialize_carry(num_agents, d_model, memory_length)
all_traces = []  # list of per-timestep diagnostics

for t in range(episode_length):
    obs_t, done_t, avail_t = ...  # single timestep inputs
    # We need to call CTMCell directly (not ScannedCTM)
    carry, synch = ctm_cell.apply(params, carry, (obs_t, done_t, avail_t))
    state_trace, activated_state_trace = carry
    all_traces.append({
        'synch': synch,                          # (num_agents, 528)
        'activated_trace': activated_state_trace, # (num_agents, 128, 5)
        'state_trace': state_trace,               # (num_agents, 128, 5)
    })
```

### Stage 2: Evaluation Script That Collects Traces

**File to create:** `smax_ctm/analyse_sync.py`

Based on the existing `eval_smax.py` pattern, this script:

1. Loads the trained CTM checkpoint (`model/smax_mappo_ctm_actor.pkl`)
2. Runs N evaluation episodes (e.g., 20-50 episodes, different seeds)
3. At each timestep, extracts the CTM internals for all 3 agents
4. Also records game events: per-agent health, alive/dead status, rewards, actions taken
5. Saves everything to a pickle file for analysis

**Key implementation details:**
- The checkpoint contains `actor_params` and `config` (see `eval_smax.py:_load_checkpoint`)
- The actor is `ActorCTM` which wraps `ScannedCTM`. For analysis we need to bypass `ScannedCTM` and call `CTMCell` directly
- The actor head (Dense layers after synch) is part of `ActorCTM`. We need to extract CTM params separately. Look at the param tree structure: `params['params']['ScannedCTM_0']` should contain CTM weights, and the Dense layers are at the `ActorCTM` level.
- **Critical:** The CTMCell params are nested inside ActorCTM params under the ScannedCTM. We need to figure out the exact param path. Print the param tree first.
- We also need to run the actor head (the Dense layers after synch) to get the actual policy, so we can record what actions the agent takes.

**Approach for param extraction:**
```python
# Load full actor params
checkpoint = pickle.load(open(...))
actor_params = checkpoint['actor_params']

# Print tree structure to find paths
print(jax.tree.map(lambda x: x.shape, actor_params))

# The CTMCell params should be at something like:
# actor_params['params']['ScannedCTM_0']['CTMCell_0']
# The head params should be at:
# actor_params['params']['Dense_0'], ['Dense_1'], ['Dense_2']
```

**Episode collection loop:**
```python
for episode in range(num_episodes):
    rng = PRNGKey(episode)
    obs, state = env.reset(rng)
    carry = CTMCell.initialize_carry(num_agents, d_model, memory_length)
    # Set dones=True initially so carry gets reset to learned start_trace
    done_batch = jnp.ones(num_agents, dtype=bool)
    
    episode_data = []
    for step in range(max_steps):
        # 1. Stack obs across agents
        obs_batch = stack_agent_obs(obs, env.agents)  # (num_agents, obs_dim)
        avail = get_avail_actions(env, state)          # (num_agents, action_dim)
        
        # 2. Run CTMCell manually (not through ScannedCTM)
        carry, synch = ctm_cell.apply(ctm_params, carry, (obs_batch, done_batch, avail))
        state_trace, activated_state_trace = carry
        
        # 3. Run actor head to get policy
        pi = actor_head(head_params, synch, avail)
        action = pi.mode()  # greedy for analysis
        
        # 4. Record everything (including raw obs for parameter-sharing control)
        episode_data.append({
            'synch': np.array(synch),                           # (3, 528)
            'activated_trace': np.array(activated_state_trace), # (3, 128, 5)
            'obs': np.array(obs_batch),                         # (3, obs_dim) — needed for obs-correlation control
            'actions': np.array(action),                        # (3,)
            'health': extract_health(state),                    # (3,) — need to figure out from SMAX state
            'alive': extract_alive(state),                      # (3,)
            'rewards': ...,
        })
        
        # 5. Step environment
        obs, state, rewards, dones, infos = env.step(rng, state, actions)
        done_batch = jnp.array([dones[a] for a in env.agents])
        if dones['__all__']:
            break
```

**SMAX state extraction:** The environment state contains unit health and alive status. Check `HeuristicEnemySMAX` for the state structure — likely `state.env_state.state.unit_health` and `state.env_state.state.unit_alive` or similar. Print the state structure to find the exact paths.

### Stage 3: Cross-Agent Synchronisation Metrics

**Compute these from the collected data:**

#### Metric 0: Observation Correlation (parameter-sharing control)
Since all agents share CTM weights, similar observations will naturally produce similar sync vectors. This baseline measures raw observation similarity so we can check whether sync correlation exceeds what input similarity alone predicts.
```python
# obs_i, obs_j are shape (obs_dim,)
obs_corr_ij_t = pearson_correlation(obs_i_t, obs_j_t)
# Compare against sync_corr_ij_t at same timestep
# Key test: is sync_corr > obs_corr, and does it show different temporal structure?
```

#### Metric 1: Sync Vector Correlation (simplest)
For each timestep, compute pairwise Pearson correlation between Agent i's synch vector and Agent j's synch vector.
```python
# synch_i, synch_j are shape (528,)
corr_ij_t = pearson_correlation(synch_i_t, synch_j_t)
# Result: 3 pairwise correlations per timestep (for 3m): (0,1), (0,2), (1,2)
```

#### Metric 2: Activated Trace Similarity
For each timestep, compute cosine similarity between the flattened activated traces of agent pairs.
```python
# trace_i, trace_j are shape (128, 5) -> flatten to (640,)
cos_sim_ij_t = cosine_similarity(trace_i_t.flatten(), trace_j_t.flatten())
```

#### Metric 3: Neuron-Level Cross-Agent Correlation
For each neuron k (0..127), compute the correlation of its activation time-series between agent pairs over a sliding window.
```python
# For neuron k, agent i has trace activated_trace[i, k, :] = (5,) over memory window
# Correlate this with agent j's trace for the same neuron
# This gives per-neuron cross-agent coupling
```

#### Metric 4: Representational Similarity Analysis (RSA)
At each timestep, compute the Representational Dissimilarity Matrix (RDM) across agents — how different are the agents' internal states from each other? Track how this RDM changes over the episode.

### Stage 4: Correlation With Game Events

**The key analysis:** Do the cross-agent sync metrics from Stage 3 correlate with coordination events?

Define coordination events from the game state:
1. **Focus fire:** 2+ agents attacking the same enemy at the same timestep
2. **Grouping:** agents moving closer together (spatial proximity from state)
3. **Kill event:** an enemy unit dies (all agents contributed)
4. **Winning vs losing phase:** compare sync patterns in won vs lost episodes

For each event type, compute:
- Average cross-agent sync correlation DURING the event vs OUTSIDE the event
- Statistical test (t-test or permutation test) for significance
- Time-lagged cross-correlation: does sync increase BEFORE the coordination event? (This would be the strongest finding — agents synchronize their thinking before acting together)

### Stage 5: Ablation Controls

Two ablation models provide mechanistic evidence that sync is *necessary* for coordination, not just correlated with it.

#### 5a: No-Sync Ablation
Train a variant that bypasses `compute_synchronisation` entirely. Instead of the sync vector, feed the flattened `activated_state_trace` (reshape to `(B, d_model * memory_length)` = `(B, 640)`) through a projection layer to match the original sync output dimension, then into the same actor head.

- Keep all other hyperparameters identical (d_model=128, memory_length=5, iterations=1, same training budget)
- Compare: (a) task performance (win rate, return), (b) cross-agent correlation of the projected trace output — expect lower correlation AND worse coordination
- This isolates the sync mechanism's contribution to cross-agent coupling

#### 5b: Iterations=3 Model
Train with `CTM_ITERATIONS=3` instead of 1. This gives 3 internal "thinking steps" per environment timestep.

- The sync signal accumulates over more iterations, potentially producing richer cross-agent patterns
- **New analysis opportunity:** extract sync vectors at each iteration within a timestep. Measure whether agents' sync vectors *converge* during the thinking process (within-step sync trajectory). This would show agents reaching consensus through iterative internal processing.
- May need to reduce d_model or increase training budget to compensate for the ~3x compute per step

Run the same Stage 3 metrics on both ablation models. The key comparisons:

| Model | Expected sync–coordination correlation | Expected performance |
|---|---|---|
| CTM iter=1 (baseline) | Significant | ~84% win rate |
| No-sync ablation | Weak/absent | Lower (paper ablation: ~50% of baseline) |
| CTM iter=3 | Stronger/richer | ≥ baseline |

### Stage 6: Visualisations

1. **Time-series plot:** Cross-agent sync correlation over episode timesteps, with game events marked (kills, deaths) as vertical lines. One plot per episode.

2. **Heatmap:** Agent-pair sync correlation matrix over time. X-axis = timestep, Y-axis = agent pair, color = correlation strength.

3. **Neuron activation heatmap:** For a single episode, show all 128 neurons x timesteps for each agent side by side. Visually check if neuron activation patterns align across agents during coordination.

4. **Scatter plot:** Cross-agent sync correlation vs. episode outcome (total return or win/loss). Each dot is one episode.

5. **Conditional averages:** Average sync correlation trajectory for WON episodes vs LOST episodes (if there's enough variance in 3m outcomes — might need to use a harder map).

---

## Practical Notes

### Running on Colab
- The analysis script should be self-contained and runnable on Colab
- It needs the trained checkpoint file (`smax_mappo_ctm_actor.pkl`)
- Include print diagnostics at every stage since we can't debug interactively
- Save intermediate results (the collected episode data pickle) so we don't have to re-run collection if analysis code changes

### Expected output structure
```
analysis_results/
  episode_traces.pkl          # Raw collected data from Stage 2
  sync_metrics.pkl            # Computed metrics from Stage 3
  figures/
    sync_timeseries_ep0.png   # Per-episode sync over time
    sync_heatmap_ep0.png      # Neuron activation heatmap
    sync_vs_outcome.png       # Sync correlation vs episode return
    event_conditional.png     # Sync during focus-fire vs not
```

### What "success" looks like
- Cross-agent sync correlation is significantly higher during coordination events (focus fire, grouping) than during independent action — p < 0.05
- Won episodes show higher average cross-agent sync than lost episodes
- Sync correlation increases BEFORE coordination events (predictive, not just reactive)
- Neuron activation heatmaps show visually apparent alignment across agents during key moments

### What "failure" looks like
- Cross-agent sync correlation is essentially random / uncorrelated with events
- No difference between won and lost episodes
- Neuron patterns are agent-specific with no cross-agent structure

### If it fails
See `research_direction.md` for the fallback plan (adaptive computation / "Think Fast, Think Slow").

---

## Implementation Order
1. **First:** Print the checkpoint param tree structure to understand how to extract CTM cell params vs actor head params
2. **Second:** Write the manual CTM unroll + episode collection (Stage 2) — get this running and verify the shapes are correct
3. **Third:** Compute the simplest metric (Metric 1: sync vector correlation) and plot the timeseries for a few episodes — this is the first signal check
4. **Fourth:** If Metric 1 shows anything interesting, compute the other metrics and do the event correlation analysis (Stages 3–4)
5. **Fifth:** Train the two ablation models (no-sync, iterations=3) and run the same metrics on them (Stage 5)
6. **Sixth:** Generate publication-quality figures (Stage 6)

Steps 1-3 should be done in the first script. Only proceed to 4-6 if 3 shows promise.
