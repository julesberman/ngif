# NGIF Minimal Demo

This is a compact, self-contained implementation of the core method from the
NGIF paper. It intentionally omits Hydra, Slurm, experiment tracking, and the
larger research-code infrastructure.

The demo focuses on one tracer-particle problem. The paper uses `jax-cfd`
forced turbulence; this minimal repo uses a deterministic periodic
incompressible Fourier flow so the example is lightweight and easy to run while
preserving the same learning setup: particles start from a Gaussian cloud in
`[0, 2pi)^2`, move with periodic boundaries, and training only sees independent
time-marginal snapshots.

## Method In One Sentence

NGIF learns a velocity field from snapshots by enforcing the weak continuity
equation

```text
d/dt E[phi(X_t)] = E[grad phi(X_t) dot u_theta(X_t, t)]
```

for many random Fourier test functions, then selects among gauge-equivalent
fields with an explicit regularizer.

Included variants:

- `NGIF-div`: vector field with divergence penalty.
- `NGIF-curl`: vector field with curl penalty.
- `NGIF-kin`: vector field with kinetic-energy penalty.
- `grad`: gradient-parameterized baseline, `u = grad_x s_theta`.

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
jupyter lab
```

If you use `uv`:

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
uv pip install -e . --no-deps
uv run jupyter lab
```

## Notebooks

- `tracer_ngif.ipynb` generates tracer snapshots, trains all four variants,
  rolls them out in memory, and saves `scatter_movie_grid.gif` directly into
  the current directory.

The default notebook settings are deliberately small. Increase `train_steps`,
`n_particles`, and `n_frequencies` for better figures.

## Layout

- `ngif/data.py`: self-contained tracer-particle generator and normalization.
- `ngif/rff.py`: random Fourier test functions and moment derivatives.
- `ngif/models.py`: small Flax MLPs for vector fields and potentials.
- `ngif/train.py`: weak-form NGIF losses, gauges, and optimizer loop.
- `ngif/rollout.py`: learned ODE rollouts and histogram TV distance.
- `ngif/plot.py`: trajectory plotting and `scatter_movie_grid`.
- `tracer_ngif.ipynb`: primary user interface.
