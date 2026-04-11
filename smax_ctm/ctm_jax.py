import jax
import jax.numpy as jnp
import flax.linen as nn
import functools
from typing import Dict, Any, Optional, Tuple

def glu(x, axis=-1):
    a, b = jnp.split(x, 2, axis=axis)
    return a * jax.nn.sigmoid(b)

def symmetric_uniform(scale):
    """Uniform initializer over [-scale, scale], matching PyTorch's .uniform_(-s, s)."""
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)
    return init

class SuperLinear(nn.Module):
    in_dims: int
    out_dims: int
    N: int
    do_norm: bool = False
    
    @nn.compact
    def __call__(self, x):
        if self.do_norm:
            x = nn.LayerNorm()(x)

        w1_init = symmetric_uniform(1.0/jnp.sqrt(self.in_dims + self.out_dims))
        w1 = self.param('w1', w1_init, (self.in_dims, self.out_dims, self.N))
        
        b1_init = nn.initializers.zeros_init()
        b1 = self.param('b1', b1_init, (1, self.N, self.out_dims))
        
        T_init = nn.initializers.constant(1.0)
        T = self.param('T', T_init, (1,))
        
        out = jnp.einsum('BDM,MHD->BDH', x, w1) + b1
        
        # Squeeze happens in NLM
        out = out / T
        return out

class NLM(nn.Module):
    d_model: int
    memory_length: int
    memory_hidden_dims: int
    deep_nlms: bool
    do_layernorm_nlm: bool = False

    @nn.compact
    def __call__(self, state_trace):
        if self.deep_nlms:
            x = SuperLinear(
                in_dims=self.memory_length,
                out_dims=2*self.memory_hidden_dims,
                N=self.d_model,
                do_norm=self.do_layernorm_nlm,
            )(state_trace)
            x = glu(x, axis=-1)
            x = SuperLinear(
                in_dims=self.memory_hidden_dims,
                out_dims=2,
                N=self.d_model,
                do_norm=self.do_layernorm_nlm,
            )(x)
            x = glu(x, axis=-1)
        else:
            x = SuperLinear(
                in_dims=self.memory_length,
                out_dims=2,
                N=self.d_model,
                do_norm=self.do_layernorm_nlm,
            )(state_trace)
            x = glu(x, axis=-1)
            
        x = x.squeeze(-1)
        return x

class CTMBackbone(nn.Module):
    d_input: int
    obs_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.d_input * 2)(obs)
        x = glu(x)
        x = nn.LayerNorm()(x)
        
        x = nn.Dense(self.d_input * 2)(x)
        x = glu(x)
        x = nn.LayerNorm()(x)
        return x

class Synapses(nn.Module):
    d_model: int
    d_input: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.d_model * 2)(x)
        x = glu(x)
        x = nn.LayerNorm()(x)
        
        x = nn.Dense(self.d_model * 2)(x)
        x = glu(x)
        x = nn.LayerNorm()(x)
        return x


class AgentConsensus(nn.Module):
    pooling: str = "mean"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        sync_agent_major: jnp.ndarray,
        alive_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Pool across other agents on axis=0 (agent-major layout).

        Args:
            sync_agent_major: (num_agents, num_envs, synch_size)
            alive_mask: (num_envs, num_agents), True for alive agents.
            deterministic: Dropout mode switch.
        Returns:
            Consensus tensor with same shape as sync_agent_major.
        """
        if sync_agent_major.ndim != 3:
            raise ValueError(
                "AgentConsensus expects rank-3 input (num_agents, num_envs, synch_size), "
                f"got shape {sync_agent_major.shape}."
            )

        num_agents, num_envs, synch_size = sync_agent_major.shape
        if num_agents <= 1:
            return jnp.zeros_like(sync_agent_major)

        if alive_mask is None:
            alive_mask = jnp.ones((num_envs, num_agents), dtype=bool)
        else:
            if alive_mask.shape != (num_envs, num_agents):
                raise ValueError(
                    "alive_mask shape mismatch: "
                    f"got {alive_mask.shape}, expected {(num_envs, num_agents)}."
                )
            alive_mask = alive_mask.astype(bool)

        # Agent-major alive mask: (num_agents, num_envs)
        alive_agent_major = jnp.swapaxes(alive_mask, 0, 1)

        # Pair mask M[i, j, e] is True when source agent j is alive and j != i.
        source_alive = alive_agent_major[None, :, :]  # (1, A, E)
        not_self = ~jnp.eye(num_agents, dtype=bool)[:, :, None]  # (A, A, 1)
        pair_mask = source_alive & not_self  # (A, A, E)
        target_alive = alive_agent_major[:, :, None]  # (A, E, 1)

        def leave_one_out_mean() -> jnp.ndarray:
            sources = sync_agent_major[None, :, :, :]  # (1, A, E, S)
            numer = jnp.sum(sources * pair_mask[:, :, :, None], axis=1)  # (A, E, S)
            denom = jnp.sum(pair_mask, axis=1, keepdims=False)[:, :, None]  # (A, E, 1)
            # Safe-divide: avoid NaN gradients from jnp.where(cond, x/0, 0.0).
            # JAX evaluates both branches in backward, so the "dead" divide must
            # not produce non-finite values even though forward discards them.
            safe_denom = jnp.where(denom > 0, denom, 1.0)
            pooled = jnp.where(denom > 0, numer / safe_denom, 0.0)
            return pooled * target_alive

        if self.pooling == "mean":
            pooled = leave_one_out_mean()
        elif self.pooling == "attention":
            q = nn.Dense(synch_size, name="attn_q")(sync_agent_major)
            k = nn.Dense(synch_size, name="attn_k")(sync_agent_major)
            v = nn.Dense(synch_size, name="attn_v")(sync_agent_major)
            logits = jnp.einsum("aes,bes->abe", q, k) / jnp.sqrt(float(synch_size))
            masked_logits = jnp.where(pair_mask, logits, -1e9)
            weights = jax.nn.softmax(masked_logits, axis=1)
            # Handle all-masked rows robustly (e.g., no alive peers).
            weights = weights * pair_mask
            norm = jnp.sum(weights, axis=1, keepdims=True)
            # Safe-divide: same gradient trap as leave_one_out_mean.
            safe_norm = jnp.where(norm > 0, norm, 1.0)
            weights = jnp.where(norm > 0, weights / safe_norm, 0.0)
            pooled = jnp.einsum("abe,bes->aes", weights, v)
            pooled = pooled * target_alive
        elif self.pooling == "gated":
            mean_others = leave_one_out_mean()
            gate = nn.Dense(synch_size, name="gate_dense")(mean_others)
            value = nn.Dense(synch_size, name="value_dense")(mean_others)
            pooled = jax.nn.sigmoid(gate) * jnp.tanh(value)
            pooled = pooled * target_alive
        else:
            raise ValueError(
                f"Unsupported INC_POOLING='{self.pooling}'. Expected one of: mean, attention, gated."
            )

        if self.dropout_rate > 0.0:
            pooled = nn.Dropout(rate=self.dropout_rate)(pooled, deterministic=deterministic)
        return pooled

def compute_synchronisation(activated_state_trace, decay_params, n_synch_out, memory_length, neuron_select_type='first-last'):
    S = jnp.transpose(activated_state_trace, (0, 2, 1))
    if activated_state_trace.shape[1] < n_synch_out:
        raise ValueError(
            f"n_synch_out ({n_synch_out}) must be <= d_model ({activated_state_trace.shape[1]})"
        )
    if neuron_select_type != 'first-last':
        raise ValueError(
            f"Unsupported neuron_select_type for RL CTM port: {neuron_select_type}. Expected 'first-last'."
        )

    # RL reference uses the last n_synch_out neurons for output synchronisation.
    S_sel = S[:, :, -n_synch_out:]
    
    triu_i, triu_j = jnp.triu_indices(n_synch_out)
    pairwise = S_sel[:, :, triu_i] * S_sel[:, :, triu_j]
    
    indices = jnp.arange(memory_length - 1, -1, -1)
    clamped_params = jnp.clip(decay_params, 0.0, 4.0)
    decay = jnp.exp(-indices[:, None] * clamped_params[None, :])
    
    numerator = jnp.sum(decay[None, :, :] * pairwise, axis=1)
    denominator = jnp.sqrt(jnp.sum(decay, axis=0))[None, :]
    synchronisation = numerator / denominator
    return synchronisation

class CTMCell(nn.Module):
    d_model: int
    d_input: int
    memory_length: int
    n_synch_out: int
    iterations: int
    deep_nlms: bool
    memory_hidden_dims: int
    obs_dim: int
    use_sync: bool = True
    neuron_select_type: str = 'first-last'
    do_layernorm_nlm: bool = False
    num_agents: int = 1
    inc_enabled: bool = False
    inc_pooling: str = "mean"
    inc_consensus_dropout: float = 0.0
    inc_use_alive_mask_from_dones: bool = True
    deterministic: bool = True

    def _single_iter(
        self,
        state_trace: jnp.ndarray,
        activated_state_trace: jnp.ndarray,
        features: jnp.ndarray,
        consensus_in: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run one inner CTM iteration and emit that iteration's synchronisation."""
        last_activated = activated_state_trace[:, :, -1]
        pre_synapse_inputs = [features, last_activated]
        if consensus_in is not None:
            if consensus_in.ndim != 2:
                raise ValueError(
                    f"consensus_in must be rank-2 (batch, synch_size), got shape {consensus_in.shape}."
                )
            if consensus_in.shape[0] != features.shape[0]:
                raise ValueError(
                    "consensus_in batch dimension must match features batch dimension "
                    f"({consensus_in.shape[0]} != {features.shape[0]})."
                )
            expected_synch_size = self.n_synch_out * (self.n_synch_out + 1) // 2
            if consensus_in.shape[1] != expected_synch_size:
                raise ValueError(
                    "consensus_in synch dimension mismatch: "
                    f"got {consensus_in.shape[1]}, expected {expected_synch_size}."
                )
            pre_synapse_inputs.append(consensus_in)
        pre_synapse = jnp.concatenate(pre_synapse_inputs, axis=-1)

        new_state = self.synapses(pre_synapse)
        new_state_trace = jnp.concatenate([state_trace[:, :, 1:], new_state[:, :, None]], axis=-1)

        activated_state = self.nlm(new_state_trace)
        new_activated_state_trace = jnp.concatenate(
            [activated_state_trace[:, :, 1:], activated_state[:, :, None]], axis=-1
        )

        if self.use_sync:
            synch = compute_synchronisation(
                new_activated_state_trace,
                self.decay_params_out,
                self.n_synch_out,
                self.memory_length,
                neuron_select_type=self.neuron_select_type,
            )
        else:
            flat_trace = jnp.reshape(
                new_activated_state_trace,
                (new_activated_state_trace.shape[0], self.d_model * self.memory_length),
            )
            expected_flat_dim = self.d_model * self.memory_length
            if flat_trace.shape[-1] != expected_flat_dim:
                raise ValueError(
                    f"Flattened activated trace has dim {flat_trace.shape[-1]}, expected {expected_flat_dim}."
                )
            synch = self.trace_proj(flat_trace)

        return new_state_trace, new_activated_state_trace, synch

    def setup(self):
        if self.num_agents <= 0:
            raise ValueError(f"num_agents must be >= 1, got {self.num_agents}.")
        if self.inc_consensus_dropout < 0.0 or self.inc_consensus_dropout >= 1.0:
            raise ValueError(
                "INC_CONSENSUS_DROPOUT must be in [0.0, 1.0), "
                f"got {self.inc_consensus_dropout}."
            )
        start_trace_init = symmetric_uniform(1.0 / jnp.sqrt(self.d_model + self.memory_length))
        self.start_trace = self.param('start_trace', start_trace_init, (self.d_model, self.memory_length))
        self.start_activated_trace = self.param(
            'start_activated_trace',
            start_trace_init,
            (self.d_model, self.memory_length),
        )
        self.backbone = CTMBackbone(d_input=self.d_input, obs_dim=self.obs_dim)
        self.synapses = Synapses(d_model=self.d_model, d_input=self.d_input)
        self.nlm = NLM(
            d_model=self.d_model,
            memory_length=self.memory_length,
            memory_hidden_dims=self.memory_hidden_dims,
            deep_nlms=self.deep_nlms,
            do_layernorm_nlm=self.do_layernorm_nlm,
        )
        synch_size = self.n_synch_out * (self.n_synch_out + 1) // 2
        if self.use_sync:
            decay_params_init = nn.initializers.zeros_init()
            self.decay_params_out = self.param('decay_params_out', decay_params_init, (synch_size,))
        else:
            self.decay_params_out = None
        if not self.use_sync:
            self.trace_proj = nn.Dense(synch_size, name="trace_proj")
        if self.inc_enabled:
            self.consensus = AgentConsensus(
                pooling=self.inc_pooling,
                dropout_rate=self.inc_consensus_dropout,
            )

    @staticmethod
    def initialize_carry(batch_size: int, d_model: int, memory_length: int):
        return (jnp.zeros((batch_size, d_model, memory_length)),
                jnp.zeros((batch_size, d_model, memory_length)))

    def __call__(self, carry, x):
        if len(x) != 3:
            raise ValueError(
                "CTMCell expects x=(obs, dones, avail_actions). "
                f"Got tuple length {len(x)}. "
                "The deterministic flag is a module attribute — set it via "
                "CTMCell(..., deterministic=...) or ScannedCTM(..., deterministic=...)."
            )
        obs, dones, avail_actions = x
        state_trace, activated_state_trace = carry

        reset_mask = dones[:, None, None]
        state_trace = jnp.where(reset_mask, self.start_trace[None], state_trace)
        activated_state_trace = jnp.where(reset_mask, self.start_activated_trace[None], activated_state_trace)
        
        features = self.backbone(obs)

        batch_size = obs.shape[0]
        if batch_size % self.num_agents != 0:
            raise ValueError(
                "CTM actor batch size must be divisible by num_agents for INC reshape. "
                f"Got batch_size={batch_size}, num_agents={self.num_agents}."
            )
        num_envs = batch_size // self.num_agents

        if self.iterations <= 0:
            raise ValueError(f"CTM iterations must be >= 1, got {self.iterations}.")
        # When INC is enabled with >1 iterations, the synapses input concatenates
        # a consensus vector. To keep the synapses Dense kernel shape consistent
        # across iterations, seed consensus_in with zeros on the first iteration.
        if self.inc_enabled and self.iterations > 1:
            synch_size = self.n_synch_out * (self.n_synch_out + 1) // 2
            consensus_in = jnp.zeros((batch_size, synch_size), dtype=features.dtype)
        else:
            consensus_in = None
        for iter_idx in range(self.iterations):
            state_trace, activated_state_trace, synch = self._single_iter(
                state_trace,
                activated_state_trace,
                features,
                consensus_in,
            )

            if self.inc_enabled and iter_idx < (self.iterations - 1):
                synch_agent_major = synch.reshape((self.num_agents, num_envs, synch.shape[-1]))
                if self.inc_use_alive_mask_from_dones:
                    done_bool = dones.astype(bool)
                    alive_flat = jnp.logical_not(done_bool)
                    alive_agent_major = alive_flat.reshape((self.num_agents, num_envs))
                    alive_mask = jnp.swapaxes(alive_agent_major, 0, 1)
                else:
                    alive_mask = jnp.ones((num_envs, self.num_agents), dtype=bool)

                consensus_agent_major = self.consensus(
                    synch_agent_major,
                    alive_mask=alive_mask,
                    deterministic=self.deterministic,
                )
                consensus_in = consensus_agent_major.reshape((batch_size, synch.shape[-1]))
            else:
                consensus_in = None
        
        new_carry = (state_trace, activated_state_trace)
        return new_carry, synch

class ScannedCTM(nn.Module):
    config: Dict[str, Any]
    deterministic: bool = True

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False, "dropout": True},
    )
    @nn.compact
    def __call__(self, carry, x):
        obs_dim = x[0].shape[-1]
        return CTMCell(
            d_model=self.config["CTM_D_MODEL"],
            d_input=self.config["CTM_D_INPUT"],
            memory_length=self.config["CTM_MEMORY_LENGTH"],
            n_synch_out=self.config["CTM_N_SYNCH_OUT"],
            iterations=self.config["CTM_ITERATIONS"],
            deep_nlms=self.config["CTM_DEEP_NLMS"],
            memory_hidden_dims=self.config["CTM_NLM_HIDDEN_DIM"],
            obs_dim=obs_dim,
            use_sync=self.config.get("CTM_USE_SYNC", True),
            neuron_select_type=self.config.get("CTM_NEURON_SELECT", "first-last"),
            do_layernorm_nlm=self.config.get("CTM_DO_LAYERNORM_NLM", False),
            num_agents=self.config.get("INC_NUM_AGENTS", 1),
            inc_enabled=self.config.get("INC_ENABLED", False),
            inc_pooling=self.config.get("INC_POOLING", "mean"),
            inc_consensus_dropout=self.config.get("INC_CONSENSUS_DROPOUT", 0.0),
            inc_use_alive_mask_from_dones=self.config.get("INC_USE_ALIVE_MASK_FROM_DONES", True),
            deterministic=self.deterministic,
        )(carry, x)
