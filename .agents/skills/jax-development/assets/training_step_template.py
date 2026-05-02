#!/usr/bin/env python3
"""Compiled training-step starter with explicit PRNG plumbing."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


def init_params(key, in_dim: int, out_dim: int):
    k1, k2 = jax.random.split(key)
    w = 0.01 * jax.random.normal(k1, (in_dim, out_dim))
    b = jnp.zeros((out_dim,), dtype=w.dtype)
    return {"w": w, "b": b}, k2


def model(params, x):
    return x @ params["w"] + params["b"]


def loss_fn(params, batch, key):
    x, y = batch
    key, dropout_key = jax.random.split(key)
    logits = model(params, x)
    keep = jax.random.bernoulli(dropout_key, p=0.9, shape=logits.shape)
    logits = jnp.where(keep, logits / 0.9, 0.0)
    loss = jnp.mean((logits - y) ** 2)
    metrics = {"loss": loss}
    return (loss, metrics), key


@functools.partial(jax.jit, donate_argnums=(0,))
def train_step(params, batch, key, lr=1e-2):
    def wrapped_loss(p):
        (loss, metrics), new_key = loss_fn(p, batch, key)
        return loss, (metrics, new_key)

    (loss, (metrics, new_key)), grads = jax.value_and_grad(wrapped_loss, has_aux=True)(params)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    sq_norm = sum((jnp.sum(g * g) for g in jax.tree.leaves(grads)), start=jnp.array(0.0, dtype=loss.dtype))
    metrics = {**metrics, "grad_norm": jnp.sqrt(sq_norm)}
    return new_params, new_key, metrics


def main():
    key = jax.random.key(0)
    params, key = init_params(key, in_dim=8, out_dim=4)
    x = jax.random.normal(key, (16, 8))
    y = jax.random.normal(key, (16, 4))
    params, key, metrics = train_step(params, (x, y), key)
    print(jax.device_get(metrics))


if __name__ == "__main__":
    main()
