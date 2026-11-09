"""
Operating points of the finite-element dataset.

The seven inputs are the two-pole current `i_s1`, the four-pole current `i_s2`, the
rotor displacement `q_M` (all complex, in rotor coordinates), and the rotor angle
`theta_M`. A tensor grid is out of the question in seven dimensions, so the training
points form a Sobol sequence, mapped such that the small two-pole currents and small
displacements of the normal operation are denser. Two other Sobol sequences add training
points in the normal operating range and around no load, where the two-pole inductance
and the force constant change quickly with the q-axis current. A separate, wider Sobol
sequence is held out for testing. The other sets are structured slices for validation
and plots.

Usage: python sample_plan.py plan.npz

"""

import sys

import numpy as np
from scipy.stats import qmc

I_S1_MAX = 15.0  # Maximum two-pole current (A)
I_S2_MAX = 36.0  # Maximum four-pole current (A)
Q_MAX = 400e-6  # Maximum displacement (m), 40 % of the air gap
THETA_PERIOD = np.pi / 3  # The maps are periodic in the rotor angle with this period

# Normal operating range
I_S1_NORMAL = 3.0  # Maximum two-pole current (A)
I_S2_NORMAL = 26.0  # Maximum four-pole current (A), 1.5 times the rated peak current
Q_NORMAL = 100e-6  # Maximum displacement (m)
I_S2Q_NO_LOAD = 4.0  # Maximum four-pole q-axis current (A) of the points around no load

# Sets of the plan
TRAIN, TEST, SLICE_I2, SLICE_F, SLICE_Q, SLICE_THETA, NORMAL, NO_LOAD = range(8)

# Averaging angles of the slices, with no aliasing below the order 48 of the angle
THETA_AVG = np.arange(8) * THETA_PERIOD / 8
I2_OPS = (5, 15, 12 + 12j, 20 + 20j, 5 + 15j)


def sobol_points(
    n_log2: int, seed: int, k_1: float, k_2: float, k_q: float
) -> tuple[np.ndarray, ...]:
    """Sobol points, the magnitudes being the powers `k_1`, `k_2`, `k_q` of them."""
    u = qmc.Sobol(d=7, scramble=True, seed=seed).random_base2(n_log2)
    i_s1 = I_S1_MAX * u[:, 0] ** k_1 * np.exp(2j * np.pi * u[:, 1])
    i_s2 = I_S2_MAX * u[:, 2] ** k_2 * np.exp(2j * np.pi * u[:, 3])
    q_M = Q_MAX * u[:, 4] ** k_q * np.exp(2j * np.pi * u[:, 5])
    theta_M = THETA_PERIOD * u[:, 6]
    return i_s1, i_s2, q_M, theta_M


def normal_points(n_log2: int, seed: int) -> tuple[np.ndarray, ...]:
    """
    Sobol points in the normal operating range.

    The four-pole current is in the first quadrant, since the symmetries of the machine
    map the other quadrants to it.

    """
    u = qmc.Sobol(d=7, scramble=True, seed=seed).random_base2(n_log2)
    i_s1 = I_S1_NORMAL * u[:, 0] ** 1.5 * np.exp(2j * np.pi * u[:, 1])
    i_s2 = I_S2_NORMAL * u[:, 2] ** 0.7 * np.exp(0.5j * np.pi * u[:, 3])
    q_M = Q_NORMAL * u[:, 4] ** 1.5 * np.exp(2j * np.pi * u[:, 5])
    theta_M = THETA_PERIOD * u[:, 6]
    return i_s1, i_s2, q_M, theta_M


def no_load_points(n_log2: int, seed: int) -> tuple[np.ndarray, ...]:
    """Sobol points in the normal range with a small four-pole q-axis current."""
    i_s1, _, q_M, theta_M = normal_points(n_log2, seed)
    u = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(n_log2)
    return i_s1, I_S2_NORMAL * u[:, 0] + 1j * I_S2Q_NO_LOAD * u[:, 1], q_M, theta_M


def _product(i_s1, i_s2, q_M, theta_M) -> tuple[np.ndarray, ...]:
    """All combinations of the given values."""
    grids = np.meshgrid(i_s1, i_s2, q_M, theta_M, indexing="ij")
    return tuple(g.ravel() for g in grids)


def make_plan() -> dict[str, np.ndarray]:
    """Create the plan, the sets for validation first and then the training points."""
    parts = []
    # Held-out test points
    parts.append((TEST, sobol_points(10, 2, 1.5, 1.0, 1.5)))
    # Four-pole flux maps of a centric rotor without the two-pole current
    i_2 = np.arange(0, 37, 3.0)
    i_s2 = (i_2[:, None] + 1j * i_2[None, :]).ravel()
    parts.append((SLICE_I2, _product([0j], i_s2, [0j], THETA_AVG)))
    # Force and two-pole flux as functions of the two-pole current
    i_s1 = np.outer([0.5, 1, 2, 4, 8], np.exp(1j * np.pi / 4 * np.arange(4))).ravel()
    parts.append((SLICE_F, _product(i_s1, I2_OPS, [0j], THETA_AVG)))
    # Unbalanced magnetic pull as a function of the displacement
    q_M = np.outer(
        np.array([100, 200, 300, 400]) * 1e-6, np.exp(1j * np.pi / 4 * np.arange(2))
    ).ravel()
    parts.append((SLICE_Q, _product([0j], I2_OPS, q_M, THETA_AVG)))
    # Dense scans of the rotor angle
    theta_M = np.arange(120) * THETA_PERIOD / 120
    for i_s1_op, i_s2_op, q_M_op in (
        (0j, 12 + 12j, 0j),
        (2 + 0j, 12 + 12j, 0j),
        (0j, 12 + 12j, 100e-6 + 0j),
    ):
        parts.append((SLICE_THETA, _product([i_s1_op], [i_s2_op], [q_M_op], theta_M)))
    # Training points: any leading part of the sequence is usable as a coarser dataset
    parts.append((TRAIN, sobol_points(14, 1, 2.5, 1.2, 2.0)))
    parts.append((NORMAL, normal_points(12, 3)))
    parts.append((NO_LOAD, no_load_points(10, 4)))
    return {
        "set": np.concatenate([np.full(len(p[0]), s) for s, p in parts]),
        "i_s1": np.concatenate([p[0] for _, p in parts]).astype(complex),
        "i_s2": np.concatenate([p[1] for _, p in parts]).astype(complex),
        "q_M": np.concatenate([p[2] for _, p in parts]).astype(complex),
        "theta_M": np.concatenate([p[3] for _, p in parts]).astype(float),
    }


if __name__ == "__main__":
    plan = make_plan()
    for s in range(8):
        print(f"Set {s}: {np.sum(plan['set'] == s)} points")
    print("Total:", plan["set"].size)
    np.savez(sys.argv[1], **plan)
