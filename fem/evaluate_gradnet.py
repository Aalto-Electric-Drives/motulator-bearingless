"""
Errors of a trained network against the finite-element data.

The errors are those of the current map at the given flux linkages or those of the flux
map at the given currents. The RMS errors are given for the sets of the dataset and for
the points in the normal operating range. The errors of the means over the rotor angle
are given for the slices, which have the same point at several angles.

Usage: python evaluate_gradnet.py dataset.npz network.npz

"""

import sys
from typing import Any

import numpy as np

from motulator_bearingless import gradnet as gn
from motulator_bearingless.utils import torque

SET_NAMES = {
    0: "train",
    1: "test",
    2: "slice i_s2",
    3: "slice f",
    4: "slice q",
    5: "theta",
    6: "train, normal range",
    7: "train, around no load",
}


def rms(x: np.ndarray) -> float:
    """Root mean square."""
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def outputs(map_fcn: Any, d: dict, mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Outputs of the map at the given points and their finite-element values."""
    q = d["q_M"][mask]
    psi_1, psi_2 = d["psi_s1"][mask], d["psi_s2"][mask]
    i_1, i_2 = d["i_s1"][mask], d["i_s2"][mask]
    if isinstance(map_fcn, gn.FluxMap):
        y_1, y_2, f = map_fcn(i_1, i_2, q)
        tau = torque(i_1, i_2, y_1, y_2, q, f)
        ref_1, ref_2 = psi_1, psi_2
    elif isinstance(map_fcn, gn.CurrentMap):
        y_1, y_2, f = map_fcn(psi_1, psi_2, q)
        tau = torque(y_1, y_2, psi_1, psi_2, q, f)
        ref_1, ref_2 = i_1, i_2
    else:
        y_1, y_2, f, tau = map_fcn(psi_1, psi_2, q, np.exp(1j * d["theta_M"][mask]))
        ref_1, ref_2 = i_1, i_2
    return y_1, y_2, f, tau, ref_1, ref_2, d["f_M"][mask], d["tau_M"][mask]


def errors(map_fcn: Any, d: dict, mask: np.ndarray) -> str:
    """Line of the RMS errors of the given points."""
    y_1, y_2, f, tau, ref_1, ref_2, f_ref, tau_ref = outputs(map_fcn, d, mask)
    return (
        f"{int(mask.sum()):6d} {1e3 * rms(y_1 - ref_1):8.2f} "
        f"{1e3 * np.max(np.abs(y_1 - ref_1)):8.1f} {1e3 * rms(y_2 - ref_2):8.2f} "
        f"{1e3 * np.max(np.abs(y_2 - ref_2)):8.1f} {rms(f - f_ref):7.1f} "
        f"{np.max(np.abs(f - f_ref)):7.0f} "
        f"{100 * rms(f - f_ref) / max(rms(f_ref), 1e-9):6.2f} "
        f"{rms(tau - tau_ref):7.3f}"
    )


def mean_errors(map_fcn: Any, d: dict, mask: np.ndarray) -> str:
    """Errors of the means over the rotor angle, the relative ones point by point."""
    keys = np.round(np.stack([d[k][mask] for k in ("i_s1", "i_s2", "q_M")], axis=-1), 9)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()

    def mean(x: np.ndarray) -> np.ndarray:
        n = np.bincount(inv)
        return (np.bincount(inv, x.real) + 1j * np.bincount(inv, x.imag)) / n

    y_1, y_2, f, tau, ref_1, ref_2, f_ref, tau_ref = (
        mean(v) for v in outputs(map_fcn, d, mask)
    )
    e_1, e_2, e_f = y_1 - ref_1, y_2 - ref_2, f - f_ref
    big_1, big_f = np.abs(ref_1) > 2e-3, np.abs(f_ref) > 20
    rel_1 = rms(e_1[big_1] / ref_1[big_1]) if big_1.any() else 0.0
    rel_f = rms(e_f[big_f] / f_ref[big_f]) if big_f.any() else 0.0
    return (
        f"output 1 {1e3 * rms(e_1):6.2f} ({100 * rel_1:4.1f} %), "
        f"output 2 {1e3 * rms(e_2):6.2f}, f {rms(e_f):6.1f} N ({100 * rel_f:4.1f} %), "
        f"tau {rms(tau - tau_ref):6.3f} Nm"
    )


def main() -> None:
    """Print the errors."""
    d = dict(np.load(sys.argv[1]))
    fcn = gn.load_gradnet(sys.argv[2])
    if fcn.map_type == "flux_map":
        map_fcn: Any = gn.FluxMap(fcn)
    else:
        map_fcn = gn.CurrentMapWithHarmonics(fcn) if fcn.k else gn.CurrentMap(fcn)
    unit = "mVs" if fcn.map_type == "flux_map" else "mA"
    print(sys.argv[2], type(map_fcn).__name__, "k", fcn.k, "units", fcn.W.shape)
    print(
        " " * 28 + f"{'n':>2s} {'1 (' + unit + ')':>8s} {'max':>8s} "
        f"{'2 (' + unit + ')':>8s} {'max':>8s} {'f (N)':>7s} {'max':>7s} {'f (%)':>6s} "
        f"{'tau (Nm)':>7s}"
    )
    # The slice of the four-pole current has corners outside the range of the training
    in_range = np.abs(d["i_s2"]) <= 36
    masks = {SET_NAMES[s]: (d["set"] == s) & in_range for s in np.unique(d["set"])}
    mixed = np.isin(d["set"], (0, 1, 6, 7))
    masks["normal range"] = (
        mixed
        & (np.abs(d["i_s1"]) < 3)
        & (np.abs(d["i_s2"]) < 26)
        & (np.abs(d["q_M"]) < 100e-6)
    )
    masks["test, normal range"] = masks["normal range"] & (d["set"] == 1)
    masks["|i_s1| < 5 A, 200 um"] = (
        mixed & (np.abs(d["i_s1"]) < 5) & (np.abs(d["q_M"]) < 200e-6)
    )
    for name, mask in masks.items():
        if mask.sum() > 0:
            print(f"{name:24s}{errors(map_fcn, d, mask)}")
    print("Means over the rotor angle")
    for s in (2, 3, 4, 5):
        mask = (d["set"] == s) & in_range
        if mask.sum() > 0:
            print(f"{SET_NAMES[s]:24s}{mean_errors(map_fcn, d, mask)}")
    mask = (d["set"] == 3) & (np.abs(d["i_s1"]) <= 2)
    if mask.sum() > 0:
        print(f"{'slice f, |i_s1| <= 2 A':24s}{mean_errors(map_fcn, d, mask)}")


if __name__ == "__main__":
    main()
