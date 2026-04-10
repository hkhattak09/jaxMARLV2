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

    @staticmethod
    def initialize_carry(batch_size: int, d_model: int, memory_length: int):
        return (jnp.zeros((batch_size, d_model, memory_length)),
                jnp.zeros((batch_size, d_model, memory_length)))

    def __call__(self, carry, x):
        obs, dones, avail_actions = x
        state_trace, activated_state_trace = carry

        reset_mask = dones[:, None, None]
        state_trace = jnp.where(reset_mask, self.start_trace[None], state_trace)
        activated_state_trace = jnp.where(reset_mask, self.start_activated_trace[None], activated_state_trace)
        
        features = self.backbone(obs)

        # Stage 1 refactor: keep consensus disabled to preserve Stage 0 behaviour.
        consensus_in = None
        if self.iterations <= 0:
            raise ValueError(f"CTM iterations must be >= 1, got {self.iterations}.")
        for _ in range(self.iterations):
            state_trace, activated_state_trace, synch = self._single_iter(
                state_trace,
                activated_state_trace,
                features,
                consensus_in,
            )
        
        new_carry = (state_trace, activated_state_trace)
        return new_carry, synch

class ScannedCTM(nn.Module):
    config: Dict[str, Any]
    
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
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
        )(carry, x)
