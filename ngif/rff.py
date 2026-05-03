"""Random Fourier test functions for weak continuity-equation matching."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import make_smoothing_spline


RFFFeatures = dict[str, np.ndarray]


def sample_frequencies(
    key: jax.Array | int,
    n_frequencies: int,
    dim: int = 2,
    bandwidths: tuple[float, ...] | np.ndarray = (0.08, 0.2, 0.5, 1.0),
) -> jax.Array:
    """Sample Gaussian RFF frequencies from several bandwidths."""

    if isinstance(key, int):
        key = jax.random.PRNGKey(key)

    if n_frequencies <= 0:
        raise ValueError("n_frequencies must be positive")

    bandwidths = np.asarray(bandwidths, dtype=np.float32)
    if len(bandwidths) == 0 or np.any(bandwidths <= 0):
        raise ValueError("bandwidths must be positive")

    n_bands = len(bandwidths)
    base = n_frequencies // n_bands
    extra = n_frequencies % n_bands
    counts = [base + int(i < extra) for i in range(n_bands)]
    keys = jax.random.split(key, n_bands)

    chunks = []
    for subkey, count, sigma in zip(keys, counts, bandwidths):
        if count:
            chunks.append(jax.random.normal(subkey, (count, dim)) / float(sigma))
    return jnp.concatenate(chunks, axis=0)


@jax.jit
def rff_phi(x: jax.Array, omega: jax.Array) -> jax.Array:
    """Return empirical `[E cos(w x), E sin(w x)]` moments."""

    z = x @ omega.T
    return jnp.concatenate([jnp.mean(jnp.cos(z), axis=0), jnp.mean(jnp.sin(z), axis=0)])


@jax.jit
def rff_grad_dot_v(x: jax.Array, v: jax.Array, omega: jax.Array) -> jax.Array:
    """Return `E[grad phi dot v]` for cosine and sine RFF tests."""

    z = x @ omega.T
    omega_dot_v = v @ omega.T
    cos_part = jnp.mean(-jnp.sin(z) * omega_dot_v, axis=0)
    sin_part = jnp.mean(jnp.cos(z) * omega_dot_v, axis=0)
    return jnp.concatenate([cos_part, sin_part])


@jax.jit
def rff_laplace_phi(x: jax.Array, omega: jax.Array) -> jax.Array:
    """Return `E[Delta phi]` for optional Fokker-Planck diffusion terms."""

    z = x @ omega.T
    omega_norm2 = jnp.sum(omega * omega, axis=-1)
    cos_part = jnp.mean(-omega_norm2 * jnp.cos(z), axis=0)
    sin_part = jnp.mean(-omega_norm2 * jnp.sin(z), axis=0)
    return jnp.concatenate([cos_part, sin_part])


def precompute_moments(x: np.ndarray, omega: jax.Array) -> np.ndarray:
    """Compute RFF moments at each snapshot time."""

    x_jax = jnp.asarray(x)
    omega = jnp.asarray(omega)
    moments = [np.asarray(rff_phi(x_jax[k], omega)) for k in range(x_jax.shape[0])]
    return np.stack(moments, axis=0).astype(np.float32)


def spline_time_derivative(
    t: np.ndarray,
    moments: np.ndarray,
    spline_lam: float = 1e-5,
) -> np.ndarray:
    """Estimate time derivatives of moment trajectories."""

    t = np.asarray(t, dtype=np.float64)
    moments = np.asarray(moments, dtype=np.float64)
    if t.ndim != 1 or moments.shape[0] != t.shape[0]:
        raise ValueError("t must be 1D and match the first axis of moments")
    if len(t) < 2:
        raise ValueError("at least two time points are required")

    if len(t) < 5:
        edge_order = 2 if len(t) > 2 else 1
        return np.gradient(moments, t, axis=0, edge_order=edge_order).astype(np.float32)

    try:
        # SciPy can smooth all columns at once on recent versions; keep a column fallback.
        spline = make_smoothing_spline(t, moments, lam=spline_lam)
        deriv = spline.derivative()(t)
    except Exception:
        columns = []
        for j in range(moments.shape[1]):
            spline = make_smoothing_spline(t, moments[:, j], lam=spline_lam)
            columns.append(spline.derivative()(t))
        deriv = np.stack(columns, axis=1)
    return np.asarray(deriv, dtype=np.float32)


def prepare_rff_features(
    x: np.ndarray,
    t: np.ndarray,
    key: jax.Array | int = 0,
    n_frequencies: int = 256,
    bandwidths: tuple[float, ...] | np.ndarray = (0.08, 0.2, 0.5, 1.0),
    spline_lam: float = 1e-5,
) -> RFFFeatures:
    """Sample RFFs and precompute moment derivatives for training."""

    omega = sample_frequencies(key, n_frequencies, dim=x.shape[-1], bandwidths=bandwidths)
    moments = precompute_moments(x, omega)
    d_moments = spline_time_derivative(t, moments, spline_lam=spline_lam)
    return {
        "omega": np.asarray(omega, dtype=np.float32),
        "moments": moments,
        "moment_derivatives": d_moments,
        "bandwidths": np.asarray(bandwidths, dtype=np.float32),
    }
