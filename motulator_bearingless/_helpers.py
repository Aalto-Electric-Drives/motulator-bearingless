"""Helper functions copied from the private modules of motulator."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from motulator.common.utils import set_style
from numpy.typing import ArrayLike


def wrap(theta: float) -> Any:
    """Limit the angle into the range [-pi, pi)."""
    return np.mod(theta + np.pi, 2 * np.pi) - np.pi


def get_value(u: Any | Callable[[Any], Any] | None, x: Any) -> Any:
    """Return `u(x)` if `u` is callable, otherwise `u`."""
    if u is None:
        raise ValueError("Input value cannot be None")
    return u(x) if callable(u) else u


def empty_array() -> np.ndarray:
    """Return an empty array."""
    return np.array([])


def setup_time_series_plot(latex: bool) -> tuple[float, float]:
    """Set the style and return the figure size (width, height) for time series."""
    set_style(latex)
    width, height = plt.rcParams["figure.figsize"]
    return width, height * (2.2 if latex else 1.8)


def configure_axes(
    axes: list,
    t_lims: tuple[float, float] | None,
    t_ticks: ArrayLike | None,
    y_lims: Sequence[tuple[float, float] | None] | None,
    y_ticks: Sequence[ArrayLike | None] | None,
) -> None:
    """Set the time-axis limits and ticks of all subplots and the y axes per plot."""
    for ax in axes:
        if t_lims is not None:
            ax.set_xlim(t_lims)
        if t_ticks is not None:
            ax.set_xticks(t_ticks)
    if y_lims is not None:
        for ax, ylim in zip(axes, y_lims, strict=False):
            if ylim is not None:
                ax.set_ylim(ylim)
    if y_ticks is not None:
        for ax, ytick in zip(axes, y_ticks, strict=False):
            if ytick is not None:
                ax.set_yticks(ytick)


def save_and_show(save_path: str | Path | None = None, **savefig_kwargs) -> None:
    """Save the figure to `save_path` (if given) and show it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        default_kwargs = {"bbox_inches": "tight"}
        default_kwargs.update(savefig_kwargs)
        plt.savefig(save_path, **default_kwargs)
    plt.show()
