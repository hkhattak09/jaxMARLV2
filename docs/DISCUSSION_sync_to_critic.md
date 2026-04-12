# Discussion: Move Sync Vectors from Actor Head to Critic

## Origin of the Idea

### The GRU Advantage
GRUs are effective because they pass their full 512-dim hidden state directly to
the MLP head. No compression, no data loss. The decision-making network sees
everything the recurrent cell learned.

### The CTM Bottleneck
CTM uses neural traces across time and computes synchronization vectors — pairwise
products of selected neurons across the memory window. We observed spikes in
collaborative action in the neural activity, but we can't pass all pairwise traces.
We pick 32 neurons (N_SYNCH_OUT=32), giving 528 dims of pairwise products. This is
a lossy compression of 128-dim internal state. The actor must make decisions through
this bottleneck.

Even after augmenting the actor head with concatenated last_activated state
(sync 528d + activated 128d = 656d), the sync component dominates the input and
the actor still underperforms GRU on harder maps (0.12 vs 0.45 on SMAX 10-unit).

### The Core Insight
What if we split the responsibilities?

- **Actor**: uses the full activated state (128d) like a GRU. The CTM recurrence
  still provides temporal memory, but there's no compression for action selection.
- **Critic**: receives the sync vectors to build coordination-aware value estimates.

The critic can answer: "given what you want to do, are the other agents actually in
a synchronized state to pull it off?" This creates a feedback loop:

1. Agent picks action from full uncompressed state
2. Critic sees sync traces and evaluates: is coordination viable?
3. Low sync + cooperative action → low value → policy gradient discourages it
4. High sync + cooperative action → high value → policy gradient reinforces it

The actor never needs to decode pairwise neuron products. The critic teaches it
*when cooperation is viable* through the value estimate.

## Design Decisions

### Actor Head Input
**Decision**: `last_activated` only (128d), drop sync from actor input.

This matches GRU's approach — full uncompressed state to the decision head. The CTM
recurrence (backbone → synapses → NLM → trace shifting) still provides temporal
memory. The actor just reads the result without lossy sync compression.

### What the Critic Receives
**Decision**: world_state (existing) + sync vectors (new), concatenated.

We considered three approaches:
1. **Simple concat at value head** — GRU processes world_state as before, sync
   gets concatenated to GRU output before the value MLP
2. **Dual-stream** — separate GRU for world_state and separate GRU/MLP for sync,
   then fuse before value head
3. **Replace world_state with sync** — rejected, critic needs enemy positions/health

**We chose option 1 for rapid prototyping.** The sync vector already encodes
temporal information (computed from activated_state_trace sliding window). The value
MLP is perfectly capable of learning "high sync + aggressive positioning = high
value" from the concatenated representation. If this validates the hypothesis, we
can explore dual-stream later.

### Where to Inject Sync in the Critic
**Decision**: after the GRU, before the value MLP.

```
world_state → Dense(FC_DIM) → relu → GRU → h_world (512d)
concat(h_world, synch) → Dense(GRU_HIDDEN) → relu → Dense(1) → value
```

Rationale: the GRU handles temporal modeling of world state (what it's good at).
The sync gets injected as coordination context right where the value decision is
made. Clean separation of concerns.

### SAAL Without Actor Using Sync
SAAL still works. The loss backpropagates through:
`synch → compute_synchronisation → activated_state_trace → NLM → synapses → backbone`

All actor parameters. SAAL still pushes CTM toward synchronized internal dynamics.
The actor head just doesn't read sync anymore, which is arguably *better* — the
head doesn't need to balance action quality with SAAL gradient flow. Fully decoupled.

### Gradient Flow: Critic Cannot Backprop into Actor
Sync vectors stored in the Transition buffer are detached from the actor's
computation graph. The critic learns to *read* sync patterns, not to *shape* them.
SAAL handles shaping. No gradient entanglement between actor and critic — this is
standard MAPPO.

## Potential Issues Identified

### 1. Storing Sync in Trajectory Buffer
Sync must be added to the Transition NamedTuple. During rollout, the actor already
produces synch but it's currently discarded (`_`). We capture and store it.
Shape: (T, num_actors, 528) in the trajectory — same batch structure as obs/world_state.

### 2. Bootstrap Value Needs Sync
After the rollout scan, we compute `last_val` for GAE bootstrapping. The new critic
needs sync for this, but we haven't run the actor on the final observation. Solution:
carry `last_synch` through the rollout scan and use it for bootstrapping. It's one
step stale but the bootstrap is already approximate.

### 3. Sync Dimensionality
528 dims per agent. The critic already handles world_state which is larger. The value
MLP's first Dense layer grows from (512 → GRU_HIDDEN) to (512+528 → GRU_HIDDEN) =
(1040 → GRU_HIDDEN). This is fine — Flax Dense infers input dims.

### 4. Actor Becomes GRU-Like
With only last_activated (128d) as input, the actor's decision pathway resembles GRU.
The CTM is still structurally different (3D trace carry, NLM sliding window, deeper
per-step compute), but the action selection is now apples-to-apples with GRU. This
is the point — match GRU's action quality, add sync to critic as a bonus.

### 5. Is This Just More Dims for the Critic?
Scientific question: does sync carry information world_state doesn't? World_state has
raw positions/health. Sync has learned temporal neural dynamics. The critic would need
very deep layers to extract "are agents coordinated?" from raw state. Sync hands it
directly. This is the hypothesis to test.

## Experimental Progression
1. **Current baseline**: actor=CTM with concat(sync, activated), critic=world_state only
2. **New idea**: actor=CTM activated only, critic=world_state + sync after GRU
3. **Compare against**: GRU baseline (actor=GRU hidden, critic=world_state)

If step 2 closes the gap with GRU (or exceeds it), the hypothesis is validated.
