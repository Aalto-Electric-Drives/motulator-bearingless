"""Flux lines and flux density of a solved operating point, read from the .ans file."""

import matplotlib.tri as mtri
import numpy as np


def read_solution(ans: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node coordinates (mm), vector potential (Wb/m), and triangles of an .ans file."""
    with open(ans, errors="ignore") as f:
        text = f.read()
    lines = text[text.find("[Solution]") :].split("\n")
    n = int(lines[1])
    nod = np.array([[float(v) for v in ln.split()[:3]] for ln in lines[2 : 2 + n]])
    n_el = int(lines[2 + n])
    tri = np.array(
        [[int(v) for v in ln.split()[:3]] for ln in lines[3 + n : 3 + n + n_el]]
    )
    return nod[:, :2], nod[:, 2], tri


def flux_density(xy: np.ndarray, a: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Flux density magnitude (T) of the linear elements."""
    x, y = xy[:, 0] * 1e-3, xy[:, 1] * 1e-3
    x1, x2, x3 = x[tri[:, 0]], x[tri[:, 1]], x[tri[:, 2]]
    y1, y2, y3 = y[tri[:, 0]], y[tri[:, 1]], y[tri[:, 2]]
    a1, a2, a3 = a[tri[:, 0]], a[tri[:, 1]], a[tri[:, 2]]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    dax = ((a2 - a1) * (y3 - y1) - (a3 - a1) * (y2 - y1)) / det
    day = ((x2 - x1) * (a3 - a1) - (x3 - x1) * (a2 - a1)) / det
    return np.hypot(dax, day)


def plot_field(ans: str, ax, levels: int = 21, b_max: float = 2.2):
    """Plot the flux density and the flux lines on the given axes."""
    xy, a, tri = read_solution(ans)
    t = mtri.Triangulation(xy[:, 0], xy[:, 1], tri)
    pc = ax.tripcolor(
        t,
        facecolors=flux_density(xy, a, tri),
        cmap="viridis",
        vmin=0,
        vmax=b_max,
        rasterized=True,
    )
    lim = np.abs(a).max()
    ax.tricontour(
        t,
        a,
        levels=np.linspace(-lim, lim, levels),
        colors="w",
        linewidths=0.5,
        linestyles="solid",
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    return pc
