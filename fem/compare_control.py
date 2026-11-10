"""
Direct force control with different machine models and controller parametrizations.

The rotor runs at the rated speed, driven externally, and it is held at a fixed radial
position. After magnetization, radial force steps are applied, followed by a torque
step, as in the simulations of [#Das2026]_. The errors of the radial force are computed
in the steady states and the rise time from the first force step. The results are
printed as a table and saved to a json file.

Usage: python compare_control.py out.json [case ...]

References
----------
.. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
   bearingless synchronous machines," Proc. ICEM, 2026,
   https://doi.org/10.1109/ICEM68283.2026.11757980

"""

import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any

import numpy as np

from motulator_bearingless import control, model
from motulator_bearingless import gradnet as gn

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "examples" / "trained_models"
PATH_FLUX = MODELS / "bsyrm_fem_flux_map_softmax_d128.npz"
PATH_CURR = MODELS / "bsyrm_fem_curr_map_softmax_d64.npz"
PATH_HARM = MODELS / "bsyrm_fem_curr_map_harm_softmax_d128.npz"
R_S = 0.215
U_DC = 540
W_M = 2 * pi * 1700 / 60  # Rated speed
TAU_RATED = 14.5  # Rated torque (Nm)
I_S2_OP = 12 + 12j  # Operating point of the linear parameters (rated peak current)

# Times of the reference steps and the stop time (s)
T_F_I, T_F_J, T_TAU, T_STOP = 0.15, 0.25, 0.35, 0.5
F_I, F_J = 100.0, 150.0


@dataclass
class Case:
    """Simulation case."""

    plant: str  # "linear", "averaged", or "harmonics"
    ctrl: str  # "linear" or "averaged"
    q_M: complex = 0j  # Static displacement (m)
    k_psi: tuple[float, float, float, float] | None = None  # Errors of the flux maps
    w_M: float = W_M  # Rotor speed (rad/s)
    tau_ref: float = 0.7 * TAU_RATED  # Torque reference after the step (Nm)


class ScaledFluxMap:
    """Flux map with errors, scaled by `k_psi` per component of the flux linkages."""

    def __init__(self, fcn: gn.FluxMap, k_psi: tuple[float, ...]) -> None:
        self.fcn = fcn
        self.k = np.array([*k_psi, 1.0, 1.0])

    def __call__(self, i_s1: Any, i_s2: Any, q_M: Any) -> tuple[Any, Any, Any]:
        """Scaled flux linkages and the force."""
        psi_1, psi_2, f_M = self.fcn(i_s1, i_s2, q_M)
        k = self.k
        return (
            k[0] * np.real(psi_1) + 1j * k[1] * np.imag(psi_1),
            k[2] * np.real(psi_2) + 1j * k[3] * np.imag(psi_2),
            f_M,
        )

    def jacobian(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """Scaled Jacobian matrix."""
        return self.k[:, None] * self.fcn.jacobian(i_s1, i_s2, q_M)


def control_model(name: str, k_psi: tuple[float, ...] | None = None) -> Any:
    """Magnetic model of the controller: the averaged flux map or the linear model."""
    flux_map: Any = gn.FluxMap(gn.load_gradnet(PATH_FLUX))
    if k_psi is not None:
        flux_map = ScaledFluxMap(flux_map, k_psi)
    par = model.SaturatedBearinglessSynchronousMachinePars(R_S, psi_s_fcn=flux_map)
    return par if name == "averaged" else par.linearized(I_S2_OP)


def machine_model(name: str) -> Any:
    """Magnetic model of the machine: a current map or the linear model."""
    if name == "harmonics":
        harm_map = gn.CurrentMapWithHarmonics(gn.load_gradnet(PATH_HARM))
        return model.SpatialSaturatedBearinglessSynchronousMachinePars(R_S, harm_map)
    if name == "averaged":
        curr_map = gn.CurrentMap(gn.load_gradnet(PATH_CURR))
        return model.SaturatedBearinglessSynchronousMachinePars(R_S, i_s_fcn=curr_map)
    return control_model("linear")


def run(case: Case) -> dict:
    """Simulate a case and compute its metrics."""
    machine = model.BearinglessSynchronousMachine(machine_model(case.plant))
    mechanics = model.ExternalRotorSpeed()
    mechanics.set_external_rotor_speed(case.w_M)
    radial = model.ExternalRotorPosition()
    q_M = complex(case.q_M)
    radial.set_external_displacement(q_M)
    converter = model.MultiThreePhaseConverter(u_dc=U_DC)
    sensors = model.BearinglessDriveSensors(w_M=model.Sensor(), theta_M=model.Sensor())
    mdl = model.BearinglessDrive(machine, mechanics, radial, converter, sensors)
    par_c = control_model(case.ctrl, case.k_psi)
    cfg = control.DirectForceControllerCfg(i_s_max=25.5, psi_s_min=0.4)
    ctrl = control.DirectForceControlSystem(control.DirectForceController(par_c, cfg))
    ctrl.set_torque_ref(lambda t: (t > T_TAU) * case.tau_ref)
    ctrl.set_force_ref(lambda t: (t > T_F_I) * F_I + 1j * (t > T_F_J) * F_J)
    t_0 = time.time()
    res = model.Simulation(mdl, ctrl).simulate(t_stop=T_STOP)
    t_m, m = res.mdl.t, res.mdl.machine

    def window(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return x[(t_m > a) & (t_m < b)]

    # Steady states: without and with the torque
    f_ref = F_I + 1j * F_J
    out = {
        "plant": case.plant,
        "ctrl": case.ctrl,
        "q_M_um": [q_M.real * 1e6, q_M.imag * 1e6],
    }
    out["k_psi"] = case.k_psi
    for name, a, b in (
        ("no_load", T_TAU - 0.05, T_TAU),
        ("load", T_STOP - 0.08, T_STOP),
    ):
        f = window(m.f_M_ab, a, b)
        tau = window(m.tau_M, a, b)
        psi = np.abs(window(m.psi_s2_dq, a, b))
        out[f"f_err_{name}"] = float(np.abs(np.mean(f) - f_ref))
        out[f"f_rip_{name}"] = float(np.sqrt(np.mean(np.abs(f - np.mean(f)) ** 2)))
        out[f"tau_{name}"] = float(np.mean(tau))
        out[f"tau_rip_{name}"] = float(np.std(tau))
        out[f"psi_{name}"] = float(np.mean(psi))
    # Rise time of the force from 10 % to 90 % after the first step
    mask = (t_m > T_F_I) & (t_m < T_F_J)
    f_i = m.f_M_ab.real[mask]
    t_w = t_m[mask]
    f_0 = np.real(np.mean(window(m.f_M_ab, T_F_I - 0.01, T_F_I)))
    above_10 = t_w[f_i > f_0 + 0.1 * (F_I - f_0)]
    above_90 = t_w[f_i > f_0 + 0.9 * (F_I - f_0)]
    out["rise_ms"] = (
        float(1e3 * (above_90[0] - above_10[0]))
        if above_90.size and above_10.size
        else None
    )
    i_abs = np.max(np.abs(m.i_c_ab), axis=0)
    out["i_max"] = float(np.max(i_abs))
    out["t_sim"] = time.time() - t_0
    return out


def run_levitation(ctrl_name: str, plant: str, f_q: float, f_o: float) -> dict:
    """
    Levitation control with the given bandwidths (Hz) of the loop and the observer.

    Gravity is applied at 0.1 s, a displacement step at 0.3 s, and torque at 0.45 s.

    """
    m_r = 20.0  # Rotor mass (kg), that of the levitation examples
    machine = model.BearinglessSynchronousMachine(machine_model(plant))
    mechanics = model.ExternalRotorSpeed()
    mechanics.set_external_rotor_speed(W_M)
    radial = model.RadialMechanics(m=m_r)
    radial.set_external_force(lambda t: (t > 0.1) * -9.81j * m_r)
    converter = model.MultiThreePhaseConverter(u_dc=U_DC)
    sensors = model.BearinglessDriveSensors(w_M=model.Sensor(), theta_M=model.Sensor())
    mdl = model.BearinglessDrive(machine, mechanics, radial, converter, sensors)
    cfg = control.DirectForceControllerCfg(i_s_max=25.5, psi_s_min=0.4)
    levitation = control.LevitationController(
        m=m_r, alpha_q=2 * pi * f_q, alpha_o=2 * pi * f_o
    )
    ctrl = control.DirectForceControlSystem(
        control.DirectForceController(control_model(ctrl_name), cfg), None, levitation
    )
    ctrl.set_torque_ref(lambda t: (t > 0.45) * 0.7 * TAU_RATED)
    ctrl.set_displacement_ref(lambda t: (t > 0.3) * 50e-6)
    t_0 = time.time()
    res = model.Simulation(mdl, ctrl).simulate(t_stop=0.6)
    t_m, m = res.mdl.t, res.mdl.machine
    q_end = m.q_M_ab[(t_m > 0.55)]
    q_mid = m.q_M_ab[(t_m > 0.25) & (t_m < 0.3)]
    return {
        "plant": plant,
        "ctrl": ctrl_name,
        "f_q": f_q,
        "q_peak_um": float(1e6 * np.max(np.abs(m.q_M_ab))),
        "q_mid_um": [
            float(1e6 * np.mean(q_mid.real)),
            float(1e6 * np.mean(q_mid.imag)),
        ],
        "q_end_um": [
            float(1e6 * np.mean(q_end.real)),
            float(1e6 * np.mean(q_end.imag)),
        ],
        "q_rip_um": float(1e6 * np.std(np.abs(q_end))),
        "f_end": [
            float(np.mean(m.f_M_ab[t_m > 0.55].real)),
            float(np.mean(m.f_M_ab[t_m > 0.55].imag)),
        ],
        "t_sim": time.time() - t_0,
    }


CASES = {
    "lin_lin": Case("linear", "linear"),
    "lin_avg": Case("linear", "averaged"),
    "avg_lin": Case("averaged", "linear"),
    "avg_avg": Case("averaged", "averaged"),
    "harm_lin": Case("harmonics", "linear"),
    "harm_avg": Case("harmonics", "averaged"),
    "harm_lin_q": Case("harmonics", "linear", 30e-6 - 20e-6j),
    "harm_avg_q": Case("harmonics", "averaged", 30e-6 - 20e-6j),
    "harm_avg_q200": Case("harmonics", "averaged", 200e-6 + 100e-6j),
    "harm_avg_q_err": Case(
        "harmonics", "averaged", 30e-6 - 20e-6j, (0.8, 1.2, 0.9, 1.1)
    ),
    "harm_avg_rated": Case("harmonics", "averaged", tau_ref=TAU_RATED),
    "harm_lin_rated": Case("harmonics", "linear", tau_ref=TAU_RATED),
    "harm_avg_fw": Case("harmonics", "averaged", w_M=2.5 * W_M),
    "harm_lin_fw": Case("harmonics", "linear", w_M=2.5 * W_M),
    "lev_lin": ("linear", "harmonics", 50.0, 150.0),
    "lev_avg": ("averaged", "harmonics", 50.0, 150.0),
    "lev_lin_100": ("linear", "harmonics", 100.0, 300.0),
    "lev_avg_100": ("averaged", "harmonics", 100.0, 300.0),
}


def _work(item: tuple[str, Any]) -> tuple[str, dict]:
    name, case = item
    try:
        if isinstance(case, Case):
            return name, run(case)
        return name, run_levitation(*case)
    except Exception as ex:  # noqa: BLE001
        return name, {"error": repr(ex)}


def main() -> None:
    """Run the cases."""
    out = sys.argv[1]
    names = sys.argv[2:] or list(CASES)
    with mp.Pool(min(4, len(names))) as pool:
        results = dict(pool.imap(_work, [(n, CASES[n]) for n in names]))
    Path(out).write_text(json.dumps(results, indent=1))
    keys = (
        "f_err_no_load",
        "f_rip_no_load",
        "f_err_load",
        "f_rip_load",
        "tau_load",
        "tau_rip_load",
        "psi_load",
        "rise_ms",
        "i_max",
        "t_sim",
    )
    print("case".ljust(16) + "".join(k.rjust(14) for k in keys))
    for name, r in results.items():
        if "error" in r:
            print(name, r["error"])
            continue
        if name.startswith("lev"):
            print(name, r)
            continue
        vals = "".join(
            f"{r[k]:14.3f}" if r[k] is not None else "None".rjust(14) for k in keys
        )
        print(name.ljust(16) + vals)


if __name__ == "__main__":
    main()
