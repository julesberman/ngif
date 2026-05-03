"""Plot helpers for tracer-particle NGIF notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def _periodic_axes(periodic: bool | tuple[bool, bool]) -> tuple[bool, bool]:
    if isinstance(periodic, (bool, np.bool_)):
        return bool(periodic), bool(periodic)

    try:
        x_periodic, y_periodic = periodic
    except (TypeError, ValueError) as exc:
        raise ValueError("periodic must be a bool or a length-2 tuple of bools") from exc
    return bool(x_periodic), bool(y_periodic)


def _periodic_periods(
    pts: np.ndarray,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    periodic_axes: tuple[bool, bool],
) -> tuple[float | None, float | None]:
    periods: list[float | None] = []
    for axis, (is_periodic, lim) in enumerate(zip(periodic_axes, (xlim, ylim), strict=True)):
        if not is_periodic:
            periods.append(None)
            continue

        if lim is not None:
            lo, hi = lim
        else:
            values = pts[..., axis]
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                periods.append(None)
                continue
            lo = float(np.min(finite_values))
            hi = float(np.max(finite_values))

        period = abs(float(hi) - float(lo))
        periods.append(period if np.isfinite(period) and period > 0.0 else None)
    return periods[0], periods[1]


def _break_periodic_trace(
    trace: np.ndarray,
    periods: tuple[float | None, float | None],
) -> tuple[np.ndarray, np.ndarray]:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape[0] < 2 or periods == (None, None):
        return trace[:, 0], trace[:, 1]

    breaks = np.zeros(trace.shape[0] - 1, dtype=bool)
    for axis, period in enumerate(periods):
        if period is None:
            continue
        breaks |= np.abs(np.diff(trace[:, axis])) > 0.5 * period

    if not np.any(breaks):
        return trace[:, 0], trace[:, 1]

    out = np.insert(trace, np.flatnonzero(breaks) + 1, np.nan, axis=0)
    return out[:, 0], out[:, 1]


def _grid_shape(
    n_panels: int,
    grid_height: int | None = None,
    grid_width: int | None = None,
) -> tuple[int, int]:
    if grid_width is None and grid_height is None:
        grid_width = min(3, n_panels)
    if grid_width is None:
        grid_width = int(np.ceil(n_panels / grid_height))
    if grid_height is None:
        grid_height = int(np.ceil(n_panels / grid_width))
    return grid_height, grid_width


def _frame_indices(n_time: int, frames: int | None) -> np.ndarray:
    if frames is None:
        return np.arange(n_time)
    return np.linspace(0, n_time - 1, min(frames, n_time), dtype=int)


def _save_animation(anim: FuncAnimation, save_to: str | Path | None) -> None:
    if save_to is None:
        return

    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    writer = "pillow" if save_to.suffix.lower() == ".gif" else None
    anim.save(save_to, writer=writer)


def plot_trajectory_grid(
    pts: np.ndarray,
    titles: list[str] | tuple[str, ...] | None = None,
    *,
    n_traj: int = 80,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    save_to: str | Path | None = None,
):
    """Static trajectory comparison for arrays shaped `(panels, time, particles, 2)`."""

    pts = np.asarray(pts)
    if pts.ndim != 4 or pts.shape[-1] != 2:
        raise ValueError("pts must have shape (panels, time, particles, 2)")

    n_panels = pts.shape[0]
    n_rows, n_cols = _grid_shape(n_panels)
    if figsize is None:
        figsize = (4.0 * n_cols, 3.8 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    rng = np.random.default_rng(0)
    idx = rng.choice(pts.shape[2], size=min(n_traj, pts.shape[2]), replace=False)
    for i in range(n_panels):
        ax = axes_flat[i]
        panel = pts[i, :, idx]
        for p in range(panel.shape[1]):
            ax.plot(panel[:, p, 0], panel[:, p, 1], color="tab:blue", alpha=0.25, linewidth=0.8)
        ax.scatter(panel[0, :, 0], panel[0, :, 1], s=6, color="black", alpha=0.45)
        ax.scatter(panel[-1, :, 0], panel[-1, :, 1], s=8, color="tab:red", alpha=0.65)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles[i] if titles else f"panel {i}")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=180, bbox_inches="tight")
    return fig, axes


def scatter_movie_grid(
    pts: np.ndarray,
    t: np.ndarray | None = None,
    titles_x: list[str] | tuple[str, ...] | None = None,
    *,
    grid_height: int | None = None,
    grid_width: int | None = None,
    fig_size: tuple[float, float] = (10, 6),
    frames: int | None = 64,
    interval: int = 120,
    n_samples: int | None = 250,
    n_traj: int | None = 60,
    size: float = 8.0,
    alpha: float = 0.75,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    c: str = "tab:blue",
    plot_trajectories: bool = True,
    periodic: bool | tuple[bool, bool] = False,
    trajectory_length: int | None = None,
    trajectory_width: float = 0.8,
    save_to: str | Path | None = None,
    show: bool = True,
) -> FuncAnimation:
    """Animate a grid of particle scatter plots.

    This is a compact, notebook-oriented version of the `scatter_movie_grid`
    utility used in the full GMFM codebase. It supports the subset needed by
    this demo: `pts` with shape `(panels, time, particles, 2)`. When `periodic`
    is true, trajectory lines are broken across wrap jumps so particles do not
    draw screen-spanning lines at periodic boundaries. The period is taken from
    `xlim` and `ylim` when provided, otherwise from the finite data range.
    """

    pts = np.asarray(pts)
    if pts.ndim != 4 or pts.shape[-1] != 2:
        raise ValueError("pts must have shape (panels, time, particles, 2)")

    n_panels, n_time, n_particles, _ = pts.shape
    if t is None:
        t = np.arange(n_time)
    t = np.asarray(t)
    if len(t) != n_time:
        raise ValueError("t must have length matching the time axis")

    frame_idx = _frame_indices(n_time, frames)

    grid_height, grid_width = _grid_shape(n_panels, grid_height, grid_width)

    fig, axes = plt.subplots(grid_height, grid_width, figsize=fig_size, squeeze=False)
    axes_flat = axes.ravel()
    rng = np.random.default_rng(0)
    sample_count = n_particles if n_samples is None else min(n_samples, n_particles)
    sample_idx = [rng.choice(n_particles, size=sample_count, replace=False) for _ in range(n_panels)]
    traj_count = sample_count if n_traj is None else min(n_traj, sample_count)
    traj_local_idx = [rng.choice(sample_count, size=traj_count, replace=False) for _ in range(n_panels)]
    periods = _periodic_periods(pts, xlim, ylim, _periodic_axes(periodic))

    scatters = []
    lines = []
    for i in range(n_panels):
        ax = axes_flat[i]
        first = pts[i, frame_idx[0], sample_idx[i]]
        scatters.append(ax.scatter(first[:, 0], first[:, 1], s=size, alpha=alpha, color=c))
        panel_lines = []
        if plot_trajectories:
            for _ in range(traj_count):
                (line,) = ax.plot([], [], color=c, alpha=0.22, linewidth=trajectory_width)
                panel_lines.append(line)
        lines.append(panel_lines)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles_x[i] if titles_x else f"panel {i}")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    time_text = fig.suptitle("")
    fig.tight_layout()

    def update(frame_number: int):
        k = int(frame_idx[frame_number])
        artists = [time_text]
        time_text.set_text(f"t = {t[k]:.3f}")
        for i in range(n_panels):
            chosen = sample_idx[i]
            current = pts[i, k, chosen]
            scatters[i].set_offsets(current)
            artists.append(scatters[i])
            if plot_trajectories:
                start = 0 if trajectory_length is None else max(0, k - trajectory_length + 1)
                for line, local_j in zip(lines[i], traj_local_idx[i]):
                    particle_j = chosen[local_j]
                    trace = pts[i, start : k + 1, particle_j]
                    line.set_data(*_break_periodic_trace(trace, periods))
                    artists.append(line)
        return artists

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=interval, blit=False)
    _save_animation(anim, save_to)
    if not show:
        plt.close(fig)
    return anim


def _resolve_cmap(cmap: str | mcolors.Colormap) -> str | mcolors.Colormap:
    if not isinstance(cmap, str) or cmap in plt.colormaps():
        return cmap
    if cmap != "mako":
        return cmap

    return mcolors.LinearSegmentedColormap.from_list(
        "mako",
        ["#0b0405", "#1d1b3f", "#2d3f73", "#277aa2", "#2fb7a3", "#def5e5"],
    )


def plot_movie_grid(
    movies: np.ndarray,
    t: np.ndarray | None = None,
    titles_x: list[str] | tuple[str, ...] | None = None,
    *,
    n_movies: int | None = None,
    grid_height: int | None = None,
    grid_width: int | None = None,
    fig_size: tuple[float, float] = (10, 6),
    frames: int | None = 64,
    interval: int = 120,
    cmap: str | mcolors.Colormap = "viridis",
    c_norm: tuple[float, float] | None = None,
    live_cbar: bool = False,
    xticks_on: bool = False,
    yticks_on: bool = False,
    interpolation: str = "nearest",
    origin: str = "lower",
    save_to: str | Path | None = None,
    show: bool = True,
) -> FuncAnimation:
    """Animate a grid of image-valued movies.

    `movies` can have shape `(panels, time, height, width)` or
    `(panels * time, height, width)` when `n_movies` is provided.
    """

    movies = np.asarray(movies)
    if movies.ndim == 3:
        if n_movies is None:
            movies = movies[None]
        elif movies.shape[0] % n_movies != 0:
            raise ValueError("movies.shape[0] must be divisible by n_movies")
        else:
            movies = movies.reshape(n_movies, movies.shape[0] // n_movies, *movies.shape[1:])
    elif movies.ndim == 4 and n_movies is not None and movies.shape[0] != n_movies:
        raise ValueError("n_movies must match movies.shape[0] for 4D inputs")

    if movies.ndim != 4:
        raise ValueError("movies must have shape (panels, time, height, width)")

    n_panels, n_time = movies.shape[:2]
    if t is None:
        t = np.arange(n_time)
    t = np.asarray(t)
    if len(t) != n_time:
        raise ValueError("t must have length matching the time axis")

    frame_idx = _frame_indices(n_time, frames)
    grid_height, grid_width = _grid_shape(n_panels, grid_height, grid_width)

    fig, axes = plt.subplots(grid_height, grid_width, figsize=fig_size, squeeze=False)
    axes_flat = axes.ravel()
    cmap = _resolve_cmap(cmap)
    norm = None if c_norm is None else mcolors.Normalize(vmin=c_norm[0], vmax=c_norm[1])

    images = []
    for i in range(n_panels):
        ax = axes_flat[i]
        im = ax.imshow(
            movies[i, frame_idx[0]],
            cmap=cmap,
            norm=norm,
            interpolation=interpolation,
            origin=origin,
        )
        images.append(im)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles_x[i] if titles_x else f"panel {i}")
        if not xticks_on:
            ax.set_xticks([])
        if not yticks_on:
            ax.set_yticks([])

    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    time_text = fig.suptitle("")
    fig.tight_layout()

    def update(frame_number: int):
        k = int(frame_idx[frame_number])
        time_text.set_text(f"t = {t[k]:.3f}")
        artists = [time_text]
        for i, im in enumerate(images):
            current = movies[i, k]
            im.set_data(current)
            if live_cbar and c_norm is None:
                im.set_clim(float(np.min(current)), float(np.max(current)))
            artists.append(im)
        return artists

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=interval, blit=False)
    _save_animation(anim, save_to)
    if not show:
        plt.close(fig)
    return anim
