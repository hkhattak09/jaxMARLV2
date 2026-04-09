import re

with open("smax_ctm/train_mappo_gru.py", "r") as f:
    content = f.read()

# 1. Imports
content = content.replace("import time\n", "import time\nfrom ctm_jax import ScannedCTM, CTMCell\n")

# 2. Add ActorCTM before CriticRNN
actor_ctm_code = """
class ActorCTM(nn.Module):
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        
        ctm_in = (obs, dones, avail_actions)
        hidden, synch = ScannedCTM(self.config)(hidden, ctm_in)
        
        x_head = nn.Dense(self.config["CTM_ACTOR_HEAD_DIM"])(synch)
        x_head = nn.relu(x_head)
        x_head = nn.Dense(self.action_dim)(x_head)
        
        unavail_actions = 1 - avail_actions
        action_logits = x_head - (unavail_actions * 1e10)
        
        pi = distrax.Categorical(logits=action_logits)
        return hidden, pi
"""
content = content.replace("class CriticRNN(nn.Module):", actor_ctm_code + "\nclass CriticRNN(nn.Module):")

# 3. Replace actor network init in train function
old_actor_init = """        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)"""

new_actor_init = """        actor_network = ActorCTM(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        action_dim = env.action_space(env.agents[0]).n
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], obs_dim)),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], action_dim)),
        )
        ac_init_hstate = CTMCell.initialize_carry(config["NUM_ENVS"], config["CTM_D_MODEL"], config["CTM_MEMORY_LENGTH"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)"""

content = content.replace(old_actor_init, new_actor_init)

# 4. Replace environment reset hidden state init
old_env_reset = """        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])"""
new_env_reset = """        ac_init_hstate = CTMCell.initialize_carry(config["NUM_ACTORS"], config["CTM_D_MODEL"], config["CTM_MEMORY_LENGTH"])"""
content = content.replace(old_env_reset, new_env_reset)

# 5. Fix minibatch slicing reshape
old_slicing = """                init_hstates = jax.tree.map(lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates)"""
new_slicing = """                init_hstates = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), init_hstates)"""
content = content.replace(old_slicing, new_slicing)

old_squeeze = """update_state = (train_states, jax.tree.map(lambda x: x.squeeze(), init_hstates), traj_batch, advantages, targets, rng)"""
new_squeeze = """update_state = (train_states, jax.tree.map(lambda x: x[0], init_hstates), traj_batch, advantages, targets, rng)"""
content = content.replace(old_squeeze, new_squeeze)

# Also fix the actor loss fn squeeze
old_actor_loss_squeeze = """_, pi = actor_network.apply(
                            actor_params, init_hstate.squeeze(), (traj_batch.obs, traj_batch.done, traj_batch.avail_actions)
                        )"""
new_actor_loss_squeeze = """_, pi = actor_network.apply(
                            actor_params, jax.tree.map(lambda x: x[0], init_hstate), (traj_batch.obs, traj_batch.done, traj_batch.avail_actions)
                        )"""
content = content.replace(old_actor_loss_squeeze, new_actor_loss_squeeze)

# Also fix critic loss fn squeeze
old_critic_loss_squeeze = """_, value = critic_network.apply(critic_params, init_hstate.squeeze(), (traj_batch.world_state,  traj_batch.done))"""
new_critic_loss_squeeze = """_, value = critic_network.apply(critic_params, jax.tree.map(lambda x: x[0], init_hstate), (traj_batch.world_state,  traj_batch.done))"""
content = content.replace(old_critic_loss_squeeze, new_critic_loss_squeeze)

# 6. Add config
old_config = """        "OBS_WITH_AGENT_ID": True,"""
new_config = """        "OBS_WITH_AGENT_ID": True,
        "CTM_D_MODEL": 128,
        "CTM_D_INPUT": 64,
        "CTM_ITERATIONS": 1,
        "CTM_N_SYNCH_OUT": 16,
        "CTM_MEMORY_LENGTH": 5,
        "CTM_DEEP_NLMS": True,
        "CTM_NLM_HIDDEN_DIM": 2,
        "CTM_NEURON_SELECT": "first-last",
        "CTM_ACTOR_HEAD_DIM": 64,"""
content = content.replace(old_config, new_config)

with open("smax_ctm/train_mappo_ctm.py", "w") as f:
    f.write(content)

