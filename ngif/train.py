"""Training utilities for the minimal NGIF tracer demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from .models import PeriodicMLP, velocity_apply
from .rff import rff_grad_dot_v, rff_laplace_phi

VARIANTS = ("div", "curl", "kin", "grad")


@dataclass(frozen=True)
class TrainConfig:
    variant: str = "div"
    steps: int = 1500
    batch_size: int = 128
    learning_rate: float = 5e-4
    gauge_weight: float = 1e-2
    diffusion: float = 0.0
    width: int = 64
    depth: int = 4
    seed: int = 0
    log_every: int = 50
    loss_epsilon: float = 1e-8
    scheduler: bool = True
    verbose: bool = True


def validate_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def build_model(config: TrainConfig) -> PeriodicMLP:
    """Construct the correct MLP for a variant."""

    validate_variant(config.variant)
    out_dim = 1 if config.variant == "grad" else 2
    return PeriodicMLP(
        width=config.width,
        depth=config.depth,
        out_dim=out_dim,
    )


def make_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    """Build Adam with the optional cosine decay used in the notebook."""

    if not config.scheduler:
        return optax.adam(config.learning_rate)

    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=max(config.steps, 1),
    )
    return optax.adam(schedule)


def init_params(model: PeriodicMLP, key: jax.Array) -> dict:
    """Initialize model parameters."""

    return model.init(key, jnp.zeros((1, 2), dtype=jnp.float32), jnp.zeros((1,), dtype=jnp.float32))


def _jacobian_x(
    model: PeriodicMLP,
    params: dict,
    variant: str,
    x: jax.Array,
    t: jax.Array,
) -> jax.Array:
    def single_velocity(xi: jax.Array, ti: jax.Array) -> jax.Array:
        return velocity_apply(model, params, variant, xi[None, :], jnp.asarray([ti]))[0]

    return jax.vmap(jax.jacrev(single_velocity, argnums=0))(x, t)


def gauge_loss(
    model: PeriodicMLP,
    params: dict,
    variant: str,
    x: jax.Array,
    t: jax.Array,
    v: jax.Array,
) -> jax.Array:
    """Regularize the gauge ambiguity selected by a training variant."""

    if variant == "kin":
        return 0.5 * jnp.mean(jnp.sum(v * v, axis=-1))
    if variant not in ("div", "curl"):
        return jnp.zeros((), dtype=v.dtype)

    # Divergence and curl gauges are the only variants that need spatial derivatives.
    jac = _jacobian_x(model, params, variant, x, t)
    if variant == "div":
        value = jnp.trace(jac, axis1=1, axis2=2)
    else:
        value = jac[:, 1, 0] - jac[:, 0, 1]
    return jnp.mean(value * value)


def make_loss_fn(
    model: PeriodicMLP,
    config: TrainConfig,
    omega: jax.Array,
):
    """Create the NGIF weak-form loss with the selected gauge."""

    omega = jnp.asarray(omega, dtype=jnp.float32)

    def loss_fn(params: dict, x_batch: jax.Array, t_batch: jax.Array, lhs: jax.Array):
        v = velocity_apply(model, params, config.variant, x_batch, t_batch)
        rhs = rff_grad_dot_v(x_batch, v, omega)
        if config.diffusion > 0.0:
            rhs = rhs + 0.5 * (config.diffusion**2) * rff_laplace_phi(x_batch, omega)

        residual = lhs - rhs
        denom = jax.lax.stop_gradient(
            jnp.mean(lhs * lhs) + jnp.mean(rhs * rhs) + config.loss_epsilon
        )
        weak_loss = jnp.mean(residual * residual) / denom

        gauge = gauge_loss(model, params, config.variant, x_batch, t_batch, v)
        total = weak_loss + config.gauge_weight * gauge
        return total, {"weak": weak_loss, "gauge": gauge}

    return loss_fn


def train_variant(
    x: np.ndarray,
    t: np.ndarray,
    omega: np.ndarray,
    moment_derivatives: np.ndarray,
    config: TrainConfig,
) -> dict[str, Any]:
    """Train one NGIF variant from normalized snapshots."""

    validate_variant(config.variant)
    if config.log_every <= 0:
        raise ValueError("log_every must be positive")

    x_jax = jnp.asarray(x, dtype=jnp.float32)
    t_jax = jnp.asarray(t, dtype=jnp.float32)
    lhs_jax = jnp.asarray(moment_derivatives, dtype=jnp.float32)
    n_times, n_particles = x_jax.shape[:2]

    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    model = build_model(config)
    params = init_params(model, init_key)
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(params)
    loss_fn = make_loss_fn(model, config, omega)

    @jax.jit
    def step(params, opt_state, key):
        key_t, key_b = jax.random.split(key)
        time_idx = jax.random.randint(key_t, (), 0, n_times)
        batch_idx = jax.random.randint(key_b, (config.batch_size,), 0, n_particles)
        x_batch = x_jax[time_idx, batch_idx]
        t_batch = jnp.full((config.batch_size,), t_jax[time_idx], dtype=jnp.float32)
        lhs = lhs_jax[time_idx]

        (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_batch, t_batch, lhs
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value, aux, time_idx

    history = {"step": [], "loss": [], "weak": [], "gauge": [], "time_idx": []}
    iterator = tqdm(range(config.steps), disable=not config.verbose, desc=f"train {config.variant}")
    for i in iterator:
        key, subkey = jax.random.split(key)
        params, opt_state, loss_value, aux, time_idx = step(params, opt_state, subkey)

        should_log = i % config.log_every == 0 or i == config.steps - 1
        if should_log:
            loss_float = float(loss_value)
            weak_float = float(aux["weak"])
            gauge_float = float(aux["gauge"])
            history["step"].append(i)
            history["loss"].append(loss_float)
            history["weak"].append(weak_float)
            history["gauge"].append(gauge_float)
            history["time_idx"].append(int(time_idx))
            if config.verbose:
                iterator.set_postfix(
                    {
                        "loss": f"{loss_float:.2e}",
                        "weak": f"{weak_float:.2e}",
                        "gauge": f"{gauge_float:.2e}",
                    }
                )

    return {
        "variant": config.variant,
        "model": model,
        "params": params,
        "history": {name: np.asarray(values) for name, values in history.items()},
        "config": asdict(config),
    }


def train_all_variants(
    x: np.ndarray,
    t: np.ndarray,
    omega: np.ndarray,
    moment_derivatives: np.ndarray,
    base_config: TrainConfig,
    variants: tuple[str, ...] = VARIANTS,
    gauge_weights: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Train several variants with a shared dataset and RFF basis."""

    results = {}
    gauge_weights = gauge_weights or {}
    for offset, variant in enumerate(variants):
        config = replace(
            base_config,
            variant=variant,
            seed=base_config.seed + 1009 * offset,
            gauge_weight=gauge_weights.get(variant, base_config.gauge_weight),
        )
        result = train_variant(x, t, omega, moment_derivatives, config)
        results[variant] = result
    return results
