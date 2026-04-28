# LoRASA + MACA for MAPPO-T: Theory and Implementation Handoff

This document is the implementation plan for adding LoRASA-style unit-type
adapters to the existing MAPPO-T/MACA training code.

The implementation must not modify `smax_ctm/train_mappo_t.py`. Instead, create
a copied trainer and add LoRASA fine-tuning there.

Primary target:

```text
frozen shared MAPPO-T actor
+ one LoRA adapter per allied unit type
+ existing MACA transformer critic and MACA advantage
```

Sources:

- LoRASA: https://arxiv.org/pdf/2502.05573
- MACA: https://arxiv.org/pdf/2508.06836

## 1. Plain-Language Theory

### What LoRASA Gives Us

The normal MAPPO-T actor is shared by all allied agents. This is efficient and
helps coordination, but it can blur together different unit behaviors. In SMAX
SMACv2 race maps, this matters because different unit types have genuinely
different roles:

```text
protoss: stalker, zealot, colossus
terran:  marine, marauder, medivac
zerg:    zergling, hydralisk, baneling
```

LoRASA adds small low-rank residual weights to the actor. The shared actor
backbone learns general combat behavior first. Then the backbone is frozen, and
only the small adapters are trained. This lets each unit type specialize while
keeping almost all of the shared-policy efficiency.

Important: the adapter should be keyed by unit type, not by agent index.

In SMACv2-style stochastic resets, `ally_0` is not a stable role. Its spawn
position and surroundings are random. A fixed `ally_0` adapter would mostly
learn noise. Unit type is stable and meaningful: a medivac should learn healing
behavior, a zealot should learn close-range engagement behavior, a stalker should
learn ranged behavior, and so on.

### What MACA Gives Us

MACA does not change the environment reward. The team reward is still shared.
MACA gives a more granular per-agent advantage signal.

Standard MAPPO gives all agents a broad joint/team advantage. That can be noisy
for specialization: if one unit type made the right move and another made the
wrong move, a single team-level advantage does not clearly separate those
effects.

MACA combines multiple baselines:

```text
individual baseline
correlated-subset baseline
joint baseline
```

This produces an agent-specific advantage, `A_i^MACA`, that is better aligned
with the contribution of agent `i`. That makes it a good training signal for
unit-type adapters:

```text
LoRASA gives each unit type capacity to specialize.
MACA tells that capacity which actions actually helped.
```

### Why Unit-Type Adapters Are the Correct Bridge

For an agent `i`, let `u_i` be its current unit type. The actor uses:

```text
shared backbone + adapter for u_i
```

So in `protoss_10_vs_10`, all stalkers share the stalker adapter, all zealots
share the zealot adapter, and all colossi share the colossus adapter. Since
SMACv2 samples unit types at reset, the adapter routing must come from the
current environment state's allied unit types, not from fixed agent ids.

This is the intended bridge:

```text
stochastic spawn position: handled by the observation and shared backbone
stable unit role/type: handled by the LoRA adapter
credit signal: handled by MACA advantage
```

### When Phase 2 Should Start

LoRASA should not start from an untrained actor. First train the normal shared
MAPPO-T/MACA policy until it shows competent, steadily improving behavior. The
LoRASA paper's ablations emphasize this timing: adding adapters too early gives
them a weak backbone to specialize from, while adding them only after full
plateau can leave less useful specialization headroom.

For this implementation, phase 2 begins from an explicit checkpoint:

```text
pretrained shared actor checkpoint
pretrained MACA critic checkpoint
optional ValueNorm checkpoint
```

The copied LoRASA trainer is a fine-tuning script, not a replacement for the
initial shared-policy pretraining script.

## 2. Mathematical Formulation

### Base Policy

Let the frozen shared actor parameters be `theta`. Let agent `i` observe `o_i`
and have current unit type `u_i`.

The shared MAPPO-T policy before LoRASA is:

```text
pi_theta(a_i | o_i)
```

### Unit-Type LoRASA Policy

For each adapted actor layer `l`, let the frozen shared weight be:

```text
W_l
```

For each unit type `k`, LoRASA adds low-rank matrices:

```text
A_{k,l}
B_{k,l}
```

The effective weight for an agent whose unit type is `k` is:

```text
W_{k,l} = stopgrad(W_l) + A_{k,l} B_{k,l}
```

Use the repository's Dense convention, where a Dense kernel has shape:

```text
input_dim x output_dim
```

Recommended parameter shapes:

```text
A_{k,l}: input_dim x rank
B_{k,l}: rank x output_dim
```

Then the LoRA residual for an activation `x` is:

```text
lora_l(x, k) = (x A_{k,l}) B_{k,l}
```

The layer output is:

```text
y = x stopgrad(W_l) + lora_l(x, k) + stopgrad(b_l)
```

Biases and normalization parameters stay frozen.

### MACA Advantage Used for Adapter Training

The MACA critic computes agent-specific baselines and combines them:

```text
b_i^MACA =
    psi_i^jnt b^jnt
  + psi_i^ind b_i^ind
  + psi_i^cor b_i^cor
```

The agent-specific advantage is:

```text
A_i^MACA = Q(s, a) - b_i^MACA
```

In the current MAPPO-T implementation, this corresponds to the existing
advantage computation using:

```text
eq return
vq_coma baseline
vq baseline
eq baseline
baseline_weights
```

The LoRASA implementation should reuse this MACA advantage logic unchanged.

### Unit-Type Adapter Gradient

Let `phi_k` be all LoRA adapter parameters for unit type `k`.

The adapter gradient is:

```text
grad_{phi_k} J =
  E[
    sum_i 1{u_i = k}
      A_i^MACA
      grad_{phi_k} log pi(a_i | o_i; theta, phi_{u_i})
  ]
```

This is the key mathematical reason the combination is sensible:

- The indicator `1{u_i = k}` routes each sample to the correct unit-type adapter.
- The frozen `theta` preserves shared general combat knowledge.
- `A_i^MACA` gives the unit-type adapter an agent-specific credit signal.
- Agents with the same unit type pool experience even though their spawn
  positions and local observations are stochastic.

### What Is Frozen and What Trains

Frozen during LoRASA phase:

```text
actor shared Dense kernels
actor shared GRU kernels
actor biases
actor LayerNorm parameters
actor feature normalization parameters
```

Trained during LoRASA phase:

```text
actor LoRA A/B adapter matrices only
critic parameters
ValueNorm state, if enabled
```

Not adapted in this implementation:

```text
critic LoRA
bias LoRA
LayerNorm LoRA
per-agent adapters
context-routed adapters
dynamic rank / AdaLoRA
```

Those are out of scope unless explicitly requested later.

## 3. Required Implementation Shape

### Files

Do not modify:

```text
smax_ctm/train_mappo_t.py
```

Create:

```text
smax_ctm/train_mappo_t_lorasa.py
smax_ctm/mappo_t/lorasa_actor.py
```

Optional but recommended tests:

```text
test_scripts/test_lorasa_actor.py
test_scripts/test_mappo_t_lorasa_smoke.py
```

### High-Level Flow

`train_mappo_t_lorasa.py` should be a copy of `train_mappo_t.py` with these
phase-2 changes:

```text
1. Load pretrained actor params.
2. Load pretrained critic params.
3. Initialize LoRASA actor params.
4. Copy pretrained actor weights into the frozen backbone portion.
5. Leave LoRA adapter params freshly initialized.
6. Create actor optimizer that updates only LoRA params.
7. Create critic optimizer as before.
8. During rollout, compute adapter ids from current allied unit types.
9. Pass adapter ids into actor forward calls.
10. Reuse existing MACA advantage computation.
11. Save actor backbone, adapters, critic, config, and ValueNorm state.
```

## 4. Actor Architecture Plan

### New Actor Module

Implement a new actor in:

```text
smax_ctm/mappo_t/lorasa_actor.py
```

Suggested public classes:

```text
LoRADense
LoRAScannedRNN
LoRASAActorTrans
```

`LoRASAActorTrans` should mirror `ActorTrans` from
`smax_ctm/mappo_t/actor.py`, but its forward signature must include adapter ids:

```python
def __call__(self, rnn_states, x, adapter_ids):
    ...
```

where:

```text
x = (obs, resets, available_actions)
adapter_ids shape:
  non-recurrent minibatch: (batch,)
  recurrent sequence:     (time, batch)
  rollout single step:    (1, batch)
```

### LoRADense

`LoRADense` should preserve the normal Dense parameters:

```text
kernel
bias
```

and add:

```text
lora_a: (num_adapter_slots, input_dim, rank)
lora_b: (num_adapter_slots, rank, output_dim)
```

Use raw SMAX unit type ids as adapter ids. For SMACv2 race maps,
`num_adapter_slots` should be `env.unit_type_bits`, usually 9. This means some
adapter slots may be unused in a given map, which is fine and avoids brittle
id-remapping inside JIT.

Initialization:

```text
lora_a = zeros
lora_b = small random normal, e.g. std 0.01
```

With `lora_a = 0`, the initial LoRASA actor must produce exactly the same logits
as the pretrained shared actor, assuming the shared weights were copied
correctly.

Implementation sketch:

```python
base = dense_with_stop_gradient(x, kernel, bias)
a = lora_a[adapter_ids]
b = lora_b[adapter_ids]
low_rank = jnp.einsum("...d,...dr->...r", x, a)
delta = jnp.einsum("...r,...ro->...o", low_rank, b)
y = base + delta
```

The shared `kernel` and `bias` must be wrapped with `jax.lax.stop_gradient`.

### GRU LoRA

This implementation should adapt the recurrent pathway, not only the output
layer. This follows LoRASA's stated actor parameterization and its placement
ablation: full linear placement is preferred over output-only placement.

Adapt:

```text
GRU input-to-hidden kernels
GRU hidden-to-hidden kernels
```

Do not adapt:

```text
GRU biases
rnn_norm parameters
```

Important checkpoint compatibility requirement:

Before adding nonzero LoRA behavior, the custom LoRA-capable GRU must reproduce
the original `nn.GRUCell` actor numerically when all LoRA adapters are zero.

The implementing agent must not guess the old Flax GRU parameter layout. It
must inspect the initialized/checkpointed actor parameter tree and write an
explicit conversion helper if the custom GRU parameter names differ.

Required validation:

```text
original ActorTrans logits
LoRASAActorTrans logits with zero LoRA
max absolute difference <= 1e-5
```

Run this check on:

```text
random obs
random available action masks
both reset=False and reset=True cases
a short recurrent sequence, not only one step
```

If this equivalence fails, do not proceed to training.

`LoRAScannedRNN` should scan over time exactly like the existing `ScannedRNN`,
but its scanned input tuple must include adapter ids:

```python
(embedding, resets, adapter_ids)
```

Use `nn.scan` with time-major adapter ids, so the per-step cell receives:

```text
embedding_t:   (batch, hidden_dim)
resets_t:      (batch,)
adapter_ids_t: (batch,)
```

The recurrent carry remains:

```text
(batch, hidden_dim)
```

### Actor Layers To Adapt

The LoRASA actor should apply LoRA to all actor linear transformations:

```text
base_0 Dense kernel
base_1 Dense kernel
base_2 Dense kernel
GRU input-to-hidden kernels
GRU hidden-to-hidden kernels
action_out Dense kernel
```

Keep these frozen:

```text
feature_norm
base_norm_0
base_norm_1
base_norm_2
rnn_norm
all biases
```

## 5. Adapter Routing

### Where Adapter IDs Come From

Adapter ids must come from the current raw SMAX unit types in the environment
state.

In the copied trainer, after `SMAXLogWrapper`, the batched environment state is
expected to contain raw unit types at:

```python
env_state.env_state.state.unit_types
```

For the allied trainable units:

```python
unit_types_env = env_state.env_state.state.unit_types[:, :env.num_agents]
```

Expected shape:

```text
(NUM_ENVS, num_allies)
```

Convert to the actor's flattened layout with the existing helper:

```python
adapter_ids = env_agent_to_actor(unit_types_env[..., None]).squeeze(-1)
adapter_ids = adapter_ids.astype(jnp.int32)
```

Expected shape:

```text
(NUM_ACTORS,)
```

For actor calls in rollout:

```python
ac_hstate, pi = actor_network.apply(
    actor_train_state.params,
    ac_hstate,
    ac_in,
    adapter_ids[None, :],
)
```

Store `adapter_ids` in the rollout transition so PPO minibatches can evaluate
old actions with the same routing used during collection.

### Transition Change

Add one field to the copied trainer's `Transition`:

```python
adapter_id: jnp.ndarray
```

Store it in actor flattened layout:

```text
(NUM_ACTORS,)
```

### Minibatch Change

Where actor minibatch data is prepared, treat `adapter_id` exactly like
`action`, `log_prob`, and `advantage`.

Non-recurrent:

```python
actor_adapter_id = traj_batch.adapter_id.reshape(actor_sample_count)
```

Recurrent:

```python
actor_adapter_id = _actor_chunks(traj_batch.adapter_id)
```

During shuffled minibatch construction, shuffle it with the same permutation as
observations and actions.

In actor loss:

```python
_, pi = actor_network.apply(
    actor_params,
    ac_init_hstate_mb,
    (mb_obs, mb_done, mb_avail),
    mb_adapter_id,
)
```

For recurrent minibatches, pass adapter ids after `swapaxes(0, 1)`, matching
the observation time-major layout.

### Eval Change

The copied trainer's `_run_eval` must also route by current unit type. In each
eval step:

```python
unit_types_env = eval_env_state.env_state.state.unit_types[:, :env.num_agents]
adapter_ids = unit_types_env.T.reshape(eval_num_envs * env.num_agents)
```

or use the same helper pattern as training.

Then call the LoRASA actor with:

```python
adapter_ids[None, :]
```

## 6. Checkpoint Loading and Saving

### CLI Arguments

Add these arguments to `train_mappo_t_lorasa.py`:

```text
--pretrained_actor_path   required
--pretrained_critic_path  required
--pretrained_valuenorm_path optional
--lorasa_rank default 8
--lorasa_init_scale default 0.01
```

The loader should accept both checkpoint styles:

```text
raw params pickle
dict pickle containing "actor_params" or "critic_params"
```

The updated backbone trainer should be treated as the preferred source. Its
periodic checkpoints contain a combined phase-1 checkpoint:

```text
saved_models/<run>/checkpoint_<step>.pkl
```

with at least:

```python
{
    "model_type": "mappo_t_backbone",
    "format_version": 1,
    "step": step,
    "update": update,
    "config": config,
    "actor_params": actor_params,
    "critic_params": critic_params,
    "value_norm_dict": value_norm_dict,
}
```

It may also include `actor_opt_state`, `critic_opt_state`, `actor_step`, and
`critic_step`.

The final backbone checkpoint is saved in the same timestamped run directory:

```text
saved_models/<run>/checkpoint_final.pkl
```

Do not use the old `saved_model/` directory for new backbone runs, and do not
write split actor/critic/ValueNorm files. A single checkpoint file is less
ambiguous and contains everything phase 2 needs.

All pickle loading must happen on the host side before the jitted training call.
Do not call `pickle.load` inside `make_train`, inside the returned `train`
function, or inside any function that is passed to `jax.jit`.

Recommended structure:

```text
__main__:
  parse CLI
  load actor params with pickle
  load critic params with pickle
  load or initialize ValueNorm dict
  build train_jit
  call train_jit(rng, loaded_actor_params, loaded_critic_params, value_norm_dict)
```

The returned train function should therefore accept loaded params as explicit
arguments, or the copied script should otherwise ensure they are available
before JIT tracing. Prefer explicit train arguments because they avoid embedding
large checkpoint pytrees as closed-over constants.

### ValueNorm Warning

Old MAPPO-T checkpoints that do not contain `value_norm_dict` are not clean
phase-2 starting points. This matters because the critic was trained with
normalized value targets, and the trainer denormalizes critic outputs when
computing returns and MACA advantages.

Implementation requirement:

```text
If --pretrained_valuenorm_path is provided:
    load it and continue exactly.
Else if use_valuenorm is true:
    initialize a fresh ValueNorm dict and print a clear warning.
```

The warning should say that continuation is not exact without the original
ValueNorm state.

Do not silently disable ValueNorm when loading a critic that was trained with
ValueNorm.

The new LoRASA trainer's own checkpoints must save:

```text
actor params, including frozen backbone and adapters
critic params
value_norm_dict
config
```

Recommended checkpoint file:

```text
checkpoint.pkl
```

with keys:

```python
{
    "model_type": "mappo_t_lorasa",
    "config": config,
    "actor_params": actor_params,
    "critic_params": critic_params,
    "value_norm_dict": value_norm_dict,
}
```

It may also save separate actor/critic pickle files for compatibility, but the
combined checkpoint is the safer canonical artifact.

## 7. Optimizer and Freezing

### Actor TrainState

The actor TrainState should contain the full LoRASA actor params:

```text
frozen shared backbone params
trainable LoRA adapter params
```

Use an optimizer mask so only paths containing LoRA adapter params update.

Recommended mask rule:

```text
True for leaves whose path contains "lora_a" or "lora_b"
False for all other actor leaves
```

Also apply `jax.lax.stop_gradient` to shared actor weights inside LoRA modules.
The optimizer mask is still required as a second guard.

### Critic TrainState

The critic should be loaded from `--pretrained_critic_path` and trained normally
with the existing critic optimizer and losses.

Do not add critic LoRA.

Do not freeze the critic. Once adapters change the policy distribution, the
centralized critic must keep tracking the new policy. This is especially
important because the MACA critic consumes `policy_probs_all`.

## 8. Training Loop Changes

Start from the copied `train_mappo_t.py` and make the smallest necessary edits.

### Initialization

Replace:

```python
actor_network = ActorTrans(...)
```

with:

```python
actor_network = LoRASAActorTrans(
    action_dim=action_dim,
    config=config,
    num_adapter_slots=env.unit_type_bits,
    rank=config["LORASA_RANK"],
    init_scale=config["LORASA_INIT_SCALE"],
)
```

Keep:

```python
critic_network = TransVCritic(...)
```

Initialize LoRASA actor params using dummy adapter ids:

```python
dummy_adapter_ids = jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32)
actor_params = actor_network.init(
    actor_rng,
    ac_init_hstate_small,
    ac_init_x,
    dummy_adapter_ids,
)
```

Then overlay/copy pretrained actor params into the shared backbone leaves.

Initialize critic params as in the original script only for shape checking if
needed, then replace with loaded critic params. Do not apply T-Fixup scaling to
loaded critic params.

### Rollout Actor Call

Before actor apply in `_env_step`, compute adapter ids from current unit types.

Then replace:

```python
ac_hstate, pi = actor_network.apply(actor_train_state.params, ac_hstate, ac_in)
```

with:

```python
ac_hstate, pi = actor_network.apply(
    actor_train_state.params,
    ac_hstate,
    ac_in,
    adapter_ids[None, :],
)
```

### Actor Loss Call

Replace:

```python
_, pi = actor_network.apply(actor_params, ac_init_hstate_mb, (mb_obs, mb_done, mb_avail))
```

with:

```python
_, pi = actor_network.apply(
    actor_params,
    ac_init_hstate_mb,
    (mb_obs, mb_done, mb_avail),
    mb_adapter_id,
)
```

No other PPO ratio/loss changes are needed.

### Advantage Logic

Do not rewrite the MACA advantage logic.

Keep the existing computation based on:

```text
eq_targets
vq_coma_value_env
vq_value_env
eq_value_env
baseline_weights
```

The only actor-side change is that the policy being evaluated uses unit-type
LoRA adapters.

## 9. Config Defaults

Add these config fields in the copied trainer after loading defaults:

```python
config["LORASA_RANK"] = 8
config["LORASA_INIT_SCALE"] = 0.01
config["LORASA_NUM_ADAPTER_SLOTS"] = env.unit_type_bits
```

Do not change the original default config file unless needed for imports. Since
the user requested not to touch the original MAPPO-T trainer, keep LoRASA config
overrides local to `train_mappo_t_lorasa.py`.

Rank guidance from LoRASA:

```text
rank 8: default
rank 4: may underfit nuanced roles
rank 16+: more capacity but can converge slower or overfit
```

## 10. Required Tests

### Test 1: Zero-LoRA Equivalence

Create a test that:

```text
1. Initializes original ActorTrans.
2. Initializes LoRASAActorTrans.
3. Copies original actor params into LoRASA shared backbone.
4. Leaves LoRA A matrices at zero.
5. Runs both actors on the same inputs.
6. Asserts logits match within 1e-5.
```

Test both:

```text
single step
short recurrent sequence
reset masks containing both true and false
available action masks
```

### Test 2: Adapter Routing Changes Output

After equivalence is proven, manually set one adapter's LoRA params nonzero and
verify:

```text
same observation + adapter_id A != same observation + adapter_id B
```

This confirms adapter ids are actually used.

### Test 3: Frozen Backbone Does Not Update

Run one actor optimizer update on fake data or a tiny rollout.

Assert:

```text
all non-LoRA actor leaves unchanged
at least one LoRA leaf changed
```

### Test 4: Tiny Training Smoke Test

Run the copied trainer with very small settings, for example:

```text
--num_envs 1
--num_steps 2
--total_timesteps 4
```

Use tiny pretrained test params or a deliberately initialized checkpoint. The
goal is only to verify shape plumbing, adapter routing, checkpoint load, and
checkpoint save.

### Test 5: Checkpoint Round Trip

Save a LoRASA checkpoint and load it again.

Assert:

```text
actor params present
critic params present
value_norm_dict present when use_valuenorm is true
config present
```

## 11. Non-Goals and Guardrails

Do not implement these in the first version:

```text
per-agent adapters
context-routed adapters
critic adapters
LoRA for biases
LoRA for LayerNorm
dynamic rank
adapter merging for inference
new MACA advantage formulas
```

Do not key adapters by:

```text
ally index
spawn position
race name alone
```

Do key adapters by:

```text
current allied unit type id from the raw SMAX state
```

Do not modify:

```text
smax_ctm/train_mappo_t.py
```

Do keep the copied trainer close to the original. The implementation should be
easy to diff against `train_mappo_t.py`.

## 12. Final Expected Behavior

At the start of LoRASA fine-tuning:

```text
LoRASA actor with zero adapters == loaded pretrained shared actor
```

During training:

```text
shared actor backbone remains frozen
unit-type LoRA adapters update
critic updates normally
MACA advantage remains the actor learning signal
```

For a Protoss race map:

```text
stalker samples update the stalker adapter
zealot samples update the zealot adapter
colossus samples update the colossus adapter
```

For a Terran race map:

```text
marine samples update the marine adapter
marauder samples update the marauder adapter
medivac samples update the medivac adapter
```

For a Zerg race map:

```text
zergling samples update the zergling adapter
hydralisk samples update the hydralisk adapter
baneling samples update the baneling adapter
```

This is the intended result:

```text
general combat competence from the frozen shared MAPPO-T actor
+ role-specific specialization from unit-type LoRA adapters
+ cleaner credit assignment from MACA
```
