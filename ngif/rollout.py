"""Rollout and simple distributional metrics for trained NGIF models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .data import wrap_unit
from .models import PeriodicMLP, velocity_apply


def _wrap_unit_jax(x: jax.Array) -> jax.Array:
    return jnp.mod(x + 1.0, 2.0) - 1.0


def rollout_model(
    model: PeriodicMLP,
    params: dict,
    variant: str,
    x0: np.ndarray,
    t_eval: np.ndarray,
    substeps: int = 4,
) -> np.ndarray:
    """Integrate a learned normalized velocity field on the periodic square."""

    x = jnp.asarray(x0, dtype=jnp.float32)
    t_eval = np.asarray(t_eval, dtype=np.float32)
    out = [np.asarray(x)]

    def field(xi: jax.Array, ti: jax.Array) -> jax.Array:
        t_batch = jnp.full((xi.shape[0],), ti, dtype=xi.dtype)
        return velocity_apply(model, params, variant, xi, t_batch)

    @jax.jit
    def rk4_step(xi: jax.Array, ti: jax.Array, dt: jax.Array) -> jax.Array:
        k1 = field(xi, ti)
        k2 = field(_wrap_unit_jax(xi + 0.5 * dt * k1), ti + 0.5 * dt)
        k3 = field(_wrap_unit_jax(xi + 0.5 * dt * k2), ti + 0.5 * dt)
        k4 = field(_wrap_unit_jax(xi + dt * k3), ti + dt)
        x_next = xi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return _wrap_unit_jax(x_next)

    for k in range(1, len(t_eval)):
        t0 = float(t_eval[k - 1])
        t1 = float(t_eval[k])
        dt = (t1 - t0) / substeps
        ti = t0
        for _ in range(substeps):
            x = rk4_step(x, jnp.asarray(ti, dtype=jnp.float32), jnp.asarray(dt, dtype=jnp.float32))
            ti += dt
        out.append(np.asarray(x))
    return np.stack(out, axis=0).astype(np.float32)


def histogram_tv(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    bins: int = 40,
    domain: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Total variation distance between 2D histograms at each time."""

    lo, hi = domain
    edges = [np.linspace(lo, hi, bins + 1), np.linspace(lo, hi, bins + 1)]
    values = []
    for true_t, pred_t in zip(x_true, x_pred):
        true_t = wrap_unit(true_t)
        pred_t = wrap_unit(pred_t)
        h_true, _, _ = np.histogram2d(true_t[:, 0], true_t[:, 1], bins=edges)
        h_pred, _, _ = np.histogram2d(pred_t[:, 0], pred_t[:, 1], bins=edges)
        p = h_true.ravel() / max(float(h_true.sum()), 1.0)
        q = h_pred.ravel() / max(float(h_pred.sum()), 1.0)
        values.append(0.5 * np.sum(np.abs(p - q)))
    return np.asarray(values, dtype=np.float32)
