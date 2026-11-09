"""Tests of the dataset tools of the finite-element model."""

import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "fem" / "make_dataset.py"
merge = runpy.run_path(str(SCRIPT))["merge"]


def plan(n: int) -> dict[str, Any]:
    """Plan of `n` points."""
    return {
        "set": np.zeros(n, dtype=int),
        "i_s1": np.arange(n) + 0j,
        "i_s2": np.full(n, 12 + 12j),
        "q_M": np.zeros(n, dtype=complex),
        "theta_M": np.zeros(n),
    }


def result(*indices: int) -> np.ndarray:
    """Result rows as written by `run_maps.py`, one per point index."""
    row = [0.1, 0.02, 0.3, 0.15, 400.0, -20.0, 10.0, 1.5, 1.6, 3.1, 60000]
    return np.array([[k, *row] for k in indices], dtype=float)


def test_single_result_row(tmp_path: Path) -> None:
    """A dataset can be made from a single solved point."""
    np.savez(tmp_path / "plan.npz", **plan(3))
    np.savetxt(tmp_path / "results.txt", result(1))
    out = tmp_path / "dataset.npz"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            tmp_path / "plan.npz",
            tmp_path / "results.txt",
            out,
        ],
        check=True,
        capture_output=True,
    )
    with np.load(out) as data:
        assert data["i_s1"].tolist() == [1 + 0j]
        assert data["f_M"].tolist() == [400 - 20j]


def test_points_are_sorted() -> None:
    """The points are ordered by their indices in the plan."""
    out = merge(plan(3), result(2, 0))
    assert out["i_s1"].tolist() == [0j, 2 + 0j]


@pytest.mark.parametrize(
    ("indices", "message"),
    [((), "No results"), ((0, 3), "outside"), ((1, 1), "Repeated")],
)
def test_invalid_results(indices: tuple[int, ...], message: str) -> None:
    """Missing, unknown, and repeated points are rejected."""
    res = result(*indices) if indices else np.empty((0, 12))
    with pytest.raises(ValueError, match=message):
        merge(plan(3), res)
