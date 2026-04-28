"""LoRASA actor network for MAPPO-T.

Adds unit-type LoRA adapters to the frozen shared MAPPO-T actor backbone.
Follows the explicit GRU layout from actor.py for checkpoint compatibility.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from .transformer import get_active_func


class FrozenLayerNorm(nn.Module):
    """LayerNorm with checkpoint-compatible frozen scale/bias parameters."""

    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim < 1:
            raise ValueError(f"FrozenLayerNorm input must be at least 1D, got {x.shape}")

        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        var = jnp.maximum(0.0, mean2 - jnp.square(mean))
        y = (x - mean) * jax.lax.rsqrt(var + self.epsilon)

        scale = self.param("scale", constant(1.0), (x.shape[-1],))
        bias = self.param("bias", constant(0.0), (x.shape[-1],))
        return y * jax.lax.stop_gradient(scale) + jax.lax.stop_gradient(bias)


class LoRADense(nn.Module):
    """Dense layer with frozen backbone and trainable LoRA adapters.

    Effective weight for adapter id ``k``:
        W_k = stopgrad(W) + A_k @ B_k

    where ``A_k`` has shape ``(input_dim, rank)`` and ``B_k`` has shape
    ``(rank, output_dim)``.
    """

    features: int
    num_adapter_slots: int
    rank: int
    init_scale: float = 0.01
    use_bias: bool = True
    kernel_init: Any = orthogonal(np.sqrt(2.0))
    bias_init: Any = constant(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray, adapter_ids: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: ``(..., input_dim)`` input tensor.
            adapter_ids: ``(...)`` integer adapter ids indexing into the LoRA
                adapter bank. Must broadcast against the leading dimensions of
                ``x``.

        Returns:
            ``(..., features)`` output tensor.
        """
        # Validate shapes loudly
        if self.num_adapter_slots <= 0:
            raise ValueError(
                f"num_adapter_slots must be positive, got {self.num_adapter_slots}"
            )
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.init_scale <= 0:
            raise ValueError(f"LoRA init_scale must be positive, got {self.init_scale}")
        if x.ndim < 1:
            raise ValueError(f"LoRADense input must be at least 1D, got shape {x.shape}")
        if not jnp.issubdtype(adapter_ids.dtype, jnp.integer):
            raise TypeError(f"adapter_ids must be integer typed, got {adapter_ids.dtype}")
        if adapter_ids.shape != x.shape[:-1]:
            raise ValueError(
                f"adapter_ids shape {adapter_ids.shape} must match x leading dims "
                f"{x.shape[:-1]}"
            )

        input_dim = x.shape[-1]

        # Frozen shared backbone parameters
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (input_dim, self.features),
        )
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
        else:
            bias = None

        # Trainable LoRA adapter parameters
        lora_a = self.param(
            "lora_a",
            constant(0.0),
            (self.num_adapter_slots, input_dim, self.rank),
        )
        lora_b = self.param(
            "lora_b",
            nn.initializers.normal(stddev=self.init_scale),
            (self.num_adapter_slots, self.rank, self.features),
        )

        # Gather adapters for the current batch
        a = lora_a[adapter_ids]  # (..., input_dim, rank)
        b = lora_b[adapter_ids]  # (..., rank, features)

        # Frozen base weights, while preserving gradient flow to earlier adapters.
        base = x @ jax.lax.stop_gradient(kernel)
        if bias is not None:
            base = base + jax.lax.stop_gradient(bias)

        # LoRA residual: (x @ a) @ b
        low_rank = jnp.einsum("...d,...dr->...r", x, a)
        delta = jnp.einsum("...r,...ro->...o", low_rank, b)

        return base + delta


class LoRAExplicitGRUCell(nn.Module):
    """Explicit GRU cell with LoRA adapters on input-to-hidden and hidden-to-hidden kernels.

    Mirrors the parameter layout of ``ExplicitGRUCell`` exactly:
        rnn/gru_cell/input_reset
        rnn/gru_cell/input_update
        rnn/gru_cell/input_candidate
        rnn/gru_cell/recurrent_reset
        rnn/gru_cell/recurrent_update
        rnn/gru_cell/recurrent_candidate

    Biases and normalization parameters remain frozen and are not adapted with LoRA.
    """

    features: int
    num_adapter_slots: int
    rank: int
    init_scale: float = 0.01

    @nn.compact
    def __call__(
        self,
        carry: jnp.ndarray,
        inputs: jnp.ndarray,
        adapter_ids: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single GRU step with LoRA.

        Args:
            carry: ``(batch, features)`` hidden state.
            inputs: ``(batch, input_dim)`` input.
            adapter_ids: ``(batch,)`` integer adapter ids.

        Returns:
            Tuple of (new_carry, new_carry).
        """
        if carry.ndim != 2:
            raise ValueError(f"carry must be 2D, got shape {carry.shape}")
        if inputs.ndim != 2:
            raise ValueError(f"inputs must be 2D, got shape {inputs.shape}")
        if adapter_ids.ndim != 1:
            raise ValueError(f"adapter_ids must be 1D, got shape {adapter_ids.shape}")
        if carry.shape[0] != inputs.shape[0]:
            raise ValueError(
                f"carry batch dim {carry.shape[0]} != inputs batch dim {inputs.shape[0]}"
            )
        if carry.shape[0] != adapter_ids.shape[0]:
            raise ValueError(
                f"carry batch dim {carry.shape[0]} != adapter_ids batch dim {adapter_ids.shape[0]}"
            )

        dense_in = functools.partial(
            LoRADense,
            self.features,
            num_adapter_slots=self.num_adapter_slots,
            rank=self.rank,
            init_scale=self.init_scale,
            use_bias=True,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )
        dense_hidden = functools.partial(
            LoRADense,
            self.features,
            num_adapter_slots=self.num_adapter_slots,
            rank=self.rank,
            init_scale=self.init_scale,
            use_bias=False,
            kernel_init=orthogonal(1.0),
        )

        reset = nn.sigmoid(
            dense_in(name="input_reset")(inputs, adapter_ids)
            + dense_hidden(name="recurrent_reset")(carry, adapter_ids)
        )
        update = nn.sigmoid(
            dense_in(name="input_update")(inputs, adapter_ids)
            + dense_hidden(name="recurrent_update")(carry, adapter_ids)
        )
        candidate = jnp.tanh(
            dense_in(name="input_candidate")(inputs, adapter_ids)
            + reset * dense_hidden(name="recurrent_candidate")(carry, adapter_ids)
        )
        new_carry = (1.0 - update) * candidate + update * carry
        return new_carry, new_carry


class LoRAScannedRNN(nn.Module):
    """GRU layer scanned over the leading time axis with LoRA adapter routing.

    The scanned input tuple is ``(embedding, resets, adapter_ids)`` where
    ``adapter_ids`` is time-major with shape ``(time, batch)``.
    """

    num_adapter_slots: int
    rank: int
    init_scale: float = 0.01

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets, adapter_ids = x

        if rnn_state.ndim != 2:
            raise ValueError(f"carry must be 2D, got shape {rnn_state.shape}")
        if ins.ndim != 2:
            raise ValueError(f"ins must be 2D, got shape {ins.shape}")
        if resets.ndim != 1:
            raise ValueError(f"resets must be 1D, got shape {resets.shape}")
        if adapter_ids.ndim != 1:
            raise ValueError(f"adapter_ids must be 1D, got shape {adapter_ids.shape}")
        if rnn_state.shape[0] != ins.shape[0]:
            raise ValueError(
                f"carry batch dim {rnn_state.shape[0]} != ins batch dim {ins.shape[0]}"
            )
        if rnn_state.shape[0] != resets.shape[0]:
            raise ValueError(
                f"carry batch dim {rnn_state.shape[0]} != resets batch dim {resets.shape[0]}"
            )
        if rnn_state.shape[0] != adapter_ids.shape[0]:
            raise ValueError(
                f"carry batch dim {rnn_state.shape[0]} != adapter_ids batch dim {adapter_ids.shape[0]}"
            )

        rnn_state = jnp.where(
            resets[:, None],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = LoRAExplicitGRUCell(
            features=ins.shape[1],
            num_adapter_slots=self.num_adapter_slots,
            rank=self.rank,
            init_scale=self.init_scale,
            name="gru_cell",
        )(rnn_state, ins, adapter_ids)
        y = FrozenLayerNorm(epsilon=1e-5, name="rnn_norm")(y)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)


class LoRASAActorTrans(nn.Module):
    """Transformer-compatible stochastic actor with unit-type LoRA adapters.

    Mirrors ``ActorTrans`` but accepts ``adapter_ids`` in the forward signature
    and uses LoRA-capable Dense and GRU layers.
    """

    action_dim: int
    config: Dict[str, Any]
    num_adapter_slots: int
    rank: int
    init_scale: float = 0.01

    @nn.compact
    def __call__(
        self,
        rnn_states: jnp.ndarray,
        x: Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]],
        adapter_ids: jnp.ndarray,
    ):
        """Return the next hidden state and action distribution.

        Args:
            rnn_states: ``(batch, hidden)`` actor recurrent state.
            x: tuple ``(obs, resets, available_actions)`` where ``obs`` is
                ``(time, batch, obs_dim)`` and ``resets`` is ``(time, batch)``.
            adapter_ids: integer adapter ids with shape matching the time/batch
                layout:
                    non-recurrent minibatch: ``(batch,)``
                    recurrent sequence:      ``(time, batch)``
                    rollout single step:     ``(1, batch)``
        """
        obs, resets, available_actions = x
        cfg = self.config
        hidden_sizes = cfg["hidden_sizes"]
        activation_name = cfg.get("activation_func", cfg["transformer"]["active_fn"])
        active_fn = get_active_func(activation_name)

        # Validate adapter_ids shape against obs leading dims
        expected_adapter_shape = obs.shape[:-1]
        if adapter_ids.shape != expected_adapter_shape:
            raise ValueError(
                f"adapter_ids shape {adapter_ids.shape} does not match obs leading dims "
                f"{expected_adapter_shape}"
            )

        if cfg["use_feature_normalization"]:
            obs = FrozenLayerNorm(epsilon=1e-5, name="feature_norm")(obs)

        embedding = obs
        for idx, hidden_size in enumerate(hidden_sizes):
            embedding = LoRADense(
                features=hidden_size,
                num_adapter_slots=self.num_adapter_slots,
                rank=self.rank,
                init_scale=self.init_scale,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
                name=f"base_{idx}",
            )(embedding, adapter_ids)
            embedding = active_fn(embedding)
            embedding = FrozenLayerNorm(epsilon=1e-5, name=f"base_norm_{idx}")(
                embedding
            )

        if cfg["use_naive_recurrent_policy"] or cfg["use_recurrent_policy"]:
            rnn_states, embedding = LoRAScannedRNN(
                num_adapter_slots=self.num_adapter_slots,
                rank=self.rank,
                init_scale=self.init_scale,
                name="rnn",
            )(rnn_states, (embedding, resets, adapter_ids))

        logits = LoRADense(
            features=self.action_dim,
            num_adapter_slots=self.num_adapter_slots,
            rank=self.rank,
            init_scale=self.init_scale,
            kernel_init=orthogonal(cfg.get("gain", 0.01)),
            bias_init=constant(0.0),
            name="action_out",
        )(embedding, adapter_ids)

        if available_actions is not None:
            if available_actions.ndim == logits.ndim - 1:
                available_actions = available_actions[None, ...]
            logits = logits - ((1.0 - available_actions) * 1e10)

        return rnn_states, distrax.Categorical(logits=logits)

    def get_actions(
        self,
        rnn_states: jnp.ndarray,
        obs: jnp.ndarray,
        resets: jnp.ndarray,
        available_actions: Optional[jnp.ndarray],
        adapter_ids: jnp.ndarray,
        rng: jnp.ndarray,
        deterministic: bool = False,
    ):
        """MACA-style helper returning actions, log-probs, probs, and state."""
        new_states, pi = self(rnn_states, (obs, resets, available_actions), adapter_ids)
        actions = jnp.argmax(pi.logits, axis=-1) if deterministic else pi.sample(seed=rng)
        action_log_probs = pi.log_prob(actions)
        return actions, action_log_probs, pi.probs, new_states

    def evaluate_actions(
        self,
        rnn_states: jnp.ndarray,
        obs: jnp.ndarray,
        resets: jnp.ndarray,
        actions: jnp.ndarray,
        adapter_ids: jnp.ndarray,
        available_actions: Optional[jnp.ndarray] = None,
        active_masks: Optional[jnp.ndarray] = None,
    ):
        """Evaluate old actions for PPO, matching MACA ACTLayer semantics."""
        _, pi = self(rnn_states, (obs, resets, available_actions), adapter_ids)
        action_log_probs = pi.log_prob(actions)
        entropy = pi.entropy()
        if active_masks is not None:
            entropy = jnp.sum(entropy * active_masks) / (jnp.sum(active_masks) + 1e-8)
        else:
            entropy = jnp.mean(entropy)
        return action_log_probs, entropy, pi
