"""
Collect the results of `run_maps.py` into a dataset and check the energy balance.

Usage: python make_dataset.py plan.npz results.txt dataset.npz

"""

import sys

import numpy as np


def merge(plan: dict[str, np.ndarray], res: np.ndarray) -> dict[str, np.ndarray]:
    """
    Merge the plan and the results, one row per solved point.

    Raises
    ------
    ValueError
        If there are no results, or if a point index is outside the plan or repeated.

    """
    if res.shape[0] == 0:
        raise ValueError("No results")
    idx = res[:, 0].astype(int)
    if np.any(idx < 0) or np.any(idx >= plan["set"].size):
        raise ValueError("Point indices outside the plan")
    if np.unique(idx).size < idx.size:
        raise ValueError("Repeated point indices")
    order = np.argsort(idx)
    idx, res = idx[order], res[order]
    return {
        "set": plan["set"][idx],
        "i_s1": plan["i_s1"][idx],
        "i_s2": plan["i_s2"][idx],
        "q_M": plan["q_M"][idx],
        "theta_M": plan["theta_M"][idx],
        "psi_s1": res[:, 1] + 1j * res[:, 2],
        "psi_s2": res[:, 3] + 1j * res[:, 4],
        "f_M": res[:, 5] + 1j * res[:, 6],
        "tau_M": res[:, 7],
        "W": res[:, 8],
        "W_co": res[:, 9],
    }


def main() -> None:
    """Merge the plan and the results and save the dataset."""
    plan = dict(np.load(sys.argv[1]))
    res = np.loadtxt(sys.argv[2], ndmin=2)
    out = merge(plan, res)
    balance = res[:, 8] + res[:, 9] - res[:, 10]
    print(f"{out['set'].size} of {plan['set'].size} points solved")
    print(f"Largest error of W + W_co = sum(psi*i): {np.abs(balance).max():.2e} J")
    np.savez_compressed(sys.argv[3], **out)


if __name__ == "__main__":
    main()
