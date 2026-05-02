#!/usr/bin/env python3
"""Carry-state loop starter using lax.scan."""

import jax
import jax.numpy as jnp


def step(carry, x_t):
    state = carry
    new_state = 0.95 * state + x_t
    y_t = jnp.tanh(new_state)
    return new_state, y_t


@jax.jit
def run(init_state, xs):
    final_state, ys = jax.lax.scan(step, init_state, xs)
    return final_state, ys


def main():
    xs = jnp.linspace(0.0, 1.0, 10)
    final_state, ys = run(jnp.array(0.0), xs)
    print("final_state:", final_state)
    print("ys:", ys)


if __name__ == "__main__":
    main()
