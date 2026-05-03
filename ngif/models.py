"""Small Flax MLPs used by the NGIF demo."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


def _as_batch(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def _time_features(t: jax.Array, batch_size: int) -> jax.Array:
    t = jnp.asarray(t)
    if t.ndim == 0:
        t = jnp.full((batch_size, 1), t)
    elif t.ndim == 1:
        # A length-one time can condition an entire particle batch.
        t = jnp.full((batch_size, 1), t[0]) if t.size == 1 else t[:, None]
    return jnp.concatenate([t, jnp.sin(2.0 * jnp.pi * t), jnp.cos(2.0 * jnp.pi * t)], axis=-1)


class PeriodicMLP(nn.Module):
    """MLP with raw state inputs and simple time conditioning."""

    width: int = 64
    depth: int = 4
    out_dim: int = 2

    @nn.compact
    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        x = _as_batch(x)
        h_t = _time_features(t, x.shape[0])
        h = jnp.concatenate([x, h_t], axis=-1)

        for _ in range(self.depth):
            h = nn.Dense(self.width)(h)
            h = nn.gelu(h)
        return nn.Dense(self.out_dim)(h)


def velocity_apply(
    model: PeriodicMLP,
    params: dict,
    variant: str,
    x: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """Evaluate either a vector field or the gradient of a scalar potential."""

    if variant == "grad":
        return velocity_from_potential(model, params, x, t)
    return model.apply(params, x, t)


def velocity_from_potential(
    model: PeriodicMLP,
    params: dict,
    x: jax.Array,
    t: jax.Array,
) -> jax.Array:
    """Compute `grad_x s_theta(x, t)` for the gradient baseline."""

    x = _as_batch(x)
    t = jnp.asarray(t)
    if t.ndim == 0:
        t = jnp.full((x.shape[0],), t)
    elif t.ndim == 1 and t.size == 1:
        t = jnp.full((x.shape[0],), t[0])

    def scalar_potential(xi: jax.Array, ti: jax.Array) -> jax.Array:
        return model.apply(params, xi[None, :], jnp.asarray([ti]))[0, 0]

    return jax.vmap(jax.grad(scalar_potential, argnums=0))(x, t)
