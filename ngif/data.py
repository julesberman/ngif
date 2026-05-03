"""Tracer-particle data for the minimal NGIF demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = float(2.0 * np.pi)


@dataclass(frozen=True)
class TracerConfig:
    """Configuration for a lightweight periodic tracer-particle problem.

    The velocity field is generated from a time-dependent stream function, so
    it is exactly divergence-free. This mirrors the structural feature used in
    the paper's jax-cfd tracer example without adding a heavy dependency.
    """

    n_particles: int = 512
    n_steps: int = 33
    t_final: float = 3.0
    seed: int = 0
    init_std: float = 1.2
    substeps_per_frame: int = 12
    modes: tuple[tuple[int, int], ...] = (
        (1, 2),
        (2, -1),
        (-2, 3),
        (3, 1),
        (1, -3),
        (3, -2),
    )
    amplitudes: tuple[float, ...] = (0.22, -0.18, 0.15, 0.12, -0.11, 0.08)
    phases: tuple[float, ...] = (0.1, 1.7, 2.4, 3.2, 4.6, 5.3)
    frequencies: tuple[float, ...] = (0.8, -0.6, 0.45, 0.7, -0.35, 0.55)


def wrap_physical(x: np.ndarray) -> np.ndarray:
    """Wrap physical positions into `[0, 2pi)`."""

    return np.mod(x, TWO_PI)


def wrap_unit(x: np.ndarray) -> np.ndarray:
    """Wrap normalized torus positions into `[-1, 1)`."""

    return np.mod(x + 1.0, 2.0) - 1.0


def x_to_unit(x: np.ndarray, domain_size: float = TWO_PI) -> np.ndarray:
    """Map physical torus positions from `[0, domain_size)` to `[-1, 1)`."""

    return wrap_unit(np.asarray(x) / (0.5 * domain_size) - 1.0)


def x_from_unit(x: np.ndarray, domain_size: float = TWO_PI) -> np.ndarray:
    """Map normalized torus positions from `[-1, 1)` back to physical units."""

    return np.mod((wrap_unit(np.asarray(x)) + 1.0) * (0.5 * domain_size), domain_size)


def t_to_unit(t: np.ndarray, t_final: float) -> np.ndarray:
    """Scale snapshot times from `[0, T]` to `[0, 1]`."""

    return np.asarray(t) / t_final


def t_from_unit(t: np.ndarray, t_final: float) -> np.ndarray:
    """Scale normalized times from `[0, 1]` back to `[0, T]`."""

    return np.asarray(t) * t_final


def tracer_velocity(x: np.ndarray, t: float, config: TracerConfig = TracerConfig()) -> np.ndarray:
    """Evaluate the analytic incompressible tracer velocity.

    A stream function `psi(t, x, y)` defines the velocity
    `(d psi / d y, -d psi / d x)`, making `div v = 0` analytically.
    """

    x = np.asarray(x, dtype=np.float64)
    modes = np.asarray(config.modes, dtype=np.float64)
    amp = np.asarray(config.amplitudes, dtype=np.float64)
    phase = np.asarray(config.phases, dtype=np.float64)
    freq = np.asarray(config.frequencies, dtype=np.float64)

    kx = modes[:, 0]
    ky = modes[:, 1]
    theta = x[..., 0, None] * kx + x[..., 1, None] * ky + phase + freq * t
    cos_theta = np.cos(theta)
    vx = np.sum(amp * ky * cos_theta, axis=-1)
    vy = -np.sum(amp * kx * cos_theta, axis=-1)
    return np.stack([vx, vy], axis=-1).astype(np.float32)


def _rk4_step(x: np.ndarray, t: float, dt: float, config: TracerConfig) -> np.ndarray:
    k1 = tracer_velocity(wrap_physical(x), t, config)
    k2 = tracer_velocity(wrap_physical(x + 0.5 * dt * k1), t + 0.5 * dt, config)
    k3 = tracer_velocity(wrap_physical(x + 0.5 * dt * k2), t + 0.5 * dt, config)
    k4 = tracer_velocity(wrap_physical(x + dt * k3), t + dt, config)
    return wrap_physical(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def generate_tracer_data(config: TracerConfig = TracerConfig()) -> dict[str, np.ndarray]:
    """Generate tracer snapshots.

    Returns a dictionary with physical positions `x` of shape
    `(n_steps, n_particles, 2)`, snapshot times `t`, and true velocities `v`.
    The velocities are useful for diagnostics but are not used by NGIF.
    """

    rng = np.random.default_rng(config.seed)
    times = np.linspace(0.0, config.t_final, config.n_steps, dtype=np.float32)
    x = rng.normal(size=(config.n_particles, 2)).astype(np.float32)
    x = wrap_physical(config.init_std * x + np.pi)

    snapshots = np.empty((config.n_steps, config.n_particles, 2), dtype=np.float32)
    velocities = np.empty_like(snapshots)
    snapshots[0] = x
    velocities[0] = tracer_velocity(x, float(times[0]), config)

    t_cur = float(times[0])
    for k in range(1, config.n_steps):
        t_next = float(times[k])
        dt = (t_next - t_cur) / config.substeps_per_frame
        for _ in range(config.substeps_per_frame):
            x = _rk4_step(x, t_cur, dt, config)
            t_cur += dt
        snapshots[k] = x.astype(np.float32)
        velocities[k] = tracer_velocity(x, t_cur, config)

    return {"x": snapshots, "t": times, "v": velocities}


def normalize_snapshots(
    x: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Normalize physical snapshots for NGIF training."""

    t_final = float(np.asarray(t)[-1])
    # Keep normalization metadata plain so callers can pass it to conversion functions.
    info = {"t_final": t_final, "domain_size": TWO_PI}
    return x_to_unit(x).astype(np.float32), t_to_unit(t, t_final).astype(np.float32), info
