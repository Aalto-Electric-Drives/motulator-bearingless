"""
Closed-loop tests for the six-phase bearingless machine drive.

The tests check the magnetic model against its energy function and the closed-loop
response of the direct and current-based force controllers (force and speed tracking,
torque balance, and flux estimation) and of the levitation controller with loose
tolerances.

"""

import pickle
from dataclasses import replace
from math import pi

import numpy as np
import pytest

from motulator_bearingless import control, model, utils

T_STOP = 0.6
PAR = model.BearinglessSynchronousMachinePars(
    R_s=0.215, L_1=0.03, L_d=0.03, L_q=0.01, m_d=10.0, m_q=0.2
)


def test_magnetic_model_is_conservative() -> None:
    """The force is the gradient of the energy function and the maps are inverses."""
    par = PAR
    i_s1, i_s2, q_M = 0.5 + 0.2j, 12 + 6j, 30e-6 - 20e-6j
    psi_s1, psi_s2 = par.psi_s(i_s1, i_s2, q_M)
    i_s1_b, i_s2_b, f_M, _ = par.magnetic_map(psi_s1, psi_s2, q_M)
    assert i_s1_b == pytest.approx(i_s1)
    assert i_s2_b == pytest.approx(i_s2)
    assert f_M == pytest.approx(par.f_M(i_s1, i_s2, q_M))

    # Magnetic energy W = 1.5*psi_s^T i_s(psi_s, q_M) at constant flux linkages
    def energy(q: complex) -> float:
        i_1, i_2, _, _ = par.magnetic_map(psi_s1, psi_s2, q)
        return 1.5 * ((i_1 * psi_s1.conjugate()).real + (i_2 * psi_s2.conjugate()).real)

    eps = 1e-9
    f_i = -(energy(q_M + eps) - energy(q_M - eps)) / (2 * eps)
    f_j = -(energy(q_M + 1j * eps) - energy(q_M - 1j * eps)) / (2 * eps)
    assert f_M == pytest.approx(f_i + 1j * f_j, rel=1e-5)

    # Current sensitivity to the displacement at constant flux linkages
    G = par.di_dq(i_s1, i_s2, q_M)
    i_1p, i_2p, _, _ = par.magnetic_map(psi_s1, psi_s2, q_M + eps)
    i_1m, i_2m, _, _ = par.magnetic_map(psi_s1, psi_s2, q_M - eps)
    assert G[0, 0] + 1j * G[1, 0] == pytest.approx((i_1p - i_1m) / (2 * eps), rel=1e-5)
    assert G[2, 0] + 1j * G[3, 0] == pytest.approx((i_2p - i_2m) / (2 * eps), rel=1e-5)

    # Force sensitivity to the displacement at constant flux linkages
    K = par.df_dq(i_s1, i_s2, q_M)
    _, _, f_p, _ = par.magnetic_map(psi_s1, psi_s2, q_M + 1j * eps)
    _, _, f_m, _ = par.magnetic_map(psi_s1, psi_s2, q_M - 1j * eps)
    assert K[0, 1] + 1j * K[1, 1] == pytest.approx((f_p - f_m) / (2 * eps), rel=1e-5)


def test_estimates_equal_model_outputs() -> None:
    """With exact flux estimates, the torque and force estimates equal the model."""
    machine = model.BearinglessSynchronousMachine(PAR)
    theta_M, q_M_ab = 0.7, 30e-6 - 20e-6j
    machine.state.psi_s1_dq, machine.state.psi_s2_dq = 0.01 + 0.005j, 0.3 + 0.1j
    machine.state.theta_M = theta_M
    machine.inp.q_M_ab = q_M_ab
    machine.set_outputs(0.0)
    observer = control.FluxObserver(PAR)
    observer.psi_s1, observer.psi_s2 = machine.state.psi_s1_dq, machine.state.psi_s2_dq
    out = machine.out
    fbk = observer.compute_output(
        0j, 0j, out.i_s1_ab, out.i_s2_ab, 0.0, theta_M, q_M_ab
    )

    assert fbk.tau_M == pytest.approx(out.tau_M)
    assert fbk.f_M_ab == pytest.approx(out.f_M_ab)


def test_aux_current_gives_torque_gradient() -> None:
    """The auxiliary current gives the torque gradient with respect to the fluxes."""
    par = PAR
    i_s1, i_s2, q_M = 0.5 + 0.2j, 12 + 6j, 30e-6 - 20e-6j
    psi_s1, psi_s2 = par.psi_s(i_s1, i_s2, q_M)
    psi_s = utils.c2v(psi_s1, psi_s2)

    def torque(psi: np.ndarray) -> float:
        p_1, p_2 = utils.v2c(psi)
        i_1, i_2, f_M, _ = par.magnetic_map(p_1, p_2, q_M)
        return (
            3 * (i_1 * p_1.conjugate()).imag
            + 6 * (i_2 * p_2.conjugate()).imag
            + (q_M * f_M.conjugate()).imag
        )

    # Gradient at constant displacement by central differences
    eps = 1e-7
    grad = np.array(
        [
            (torque(psi_s + eps * v) - torque(psi_s - eps * v)) / (2 * eps)
            for v in np.eye(4)
        ]
    )
    i_a1, i_a2, _, _ = control.aux_vectors(par, i_s1, i_s2, psi_s1, psi_s2, q_M)
    assert 3 * utils.J_M @ utils.c2v(i_a1, i_a2) == pytest.approx(grad, rel=1e-5)


def make_drive(
    radial: model.ExternalRotorPosition | model.RadialMechanics,
    pwm: bool,
    sensored: bool = True,
) -> model.BearinglessDrive:
    """Bearingless drive, with the speed and position sensors if `sensored`."""
    sensors = model.BearinglessDriveSensors(w_M=model.Sensor(), theta_M=model.Sensor())
    return model.BearinglessDrive(
        model.BearinglessSynchronousMachine(PAR),
        model.MechanicalSystem(J=0.15),
        radial,
        model.MultiThreePhaseConverter(
            u_dc=540, modulator=model.CarrierComparison() if pwm else None
        ),
        sensors if sensored else None,
    )


def make_ctrl(
    L_1: float = PAR.L_1,
    sensorless: bool = False,
    levitation_ctrl: control.LevitationController | None = None,
) -> control.DirectForceControlSystem:
    est_par = replace(PAR, L_1=L_1)
    cfg = control.DirectForceControllerCfg(
        i_s_max=25, psi_s_min=0.3, sensorless=sensorless
    )
    return control.DirectForceControlSystem(
        control.DirectForceController(est_par, cfg),
        control.SpeedController(J=0.15, alpha_s=2 * pi * 8),
        levitation_ctrl,
    )


def make_current_based_ctrl(
    flux_vector: bool = False,
    sensorless: bool = False,
    levitation_ctrl: control.LevitationController | None = None,
) -> control.CurrentBasedForceControlSystem:
    # The current of the equivalent machine is twice the four-pole current
    par_s2 = utils.fourpole_pars(PAR)
    vector_ctrl: control.CurrentVectorController | control.FluxVectorController
    if flux_vector:
        cfg = control.FluxVectorControllerCfg(
            i_s_max=50, psi_s_min=0.3, sensorless=sensorless
        )
        vector_ctrl = control.FluxVectorController(par_s2, cfg)
    else:
        cfg = control.CurrentVectorControllerCfg(
            i_s_max=50, psi_s_min=0.3, sensorless=sensorless
        )
        vector_ctrl = control.CurrentVectorController(par_s2, cfg)
    return control.CurrentBasedForceControlSystem(
        vector_ctrl,
        control.CurrentBasedForceController(PAR, i_s1_max=5.0),
        control.SpeedController(J=0.15, alpha_s=2 * pi * 8),
        levitation_ctrl,
    )


def check_force_control(
    mdl: model.BearinglessDrive, ctrl: control.BearinglessControlSystem
) -> None:
    """The force and the speed follow their references and the flux is estimated."""
    w_M_ref, f_M_ref, tau_L = 100.0, 100 + 150j, 5.0
    ctrl.set_speed_ref(lambda t: min(t / 0.2, 1) * w_M_ref)
    ctrl.set_force_ref(
        lambda t: (t > 0.3) * f_M_ref.real + 1j * (t > 0.35) * f_M_ref.imag
    )
    mdl.mechanics.set_external_load_torque(lambda t: (t > 0.4) * tau_L)
    res = model.Simulation(mdl, ctrl, show_progress=False).simulate(t_stop=T_STOP)

    # Averages over the last 50 ms
    end = res.mdl.t > T_STOP - 0.05
    ctrl_end = res.ctrl.t > T_STOP - 0.05
    w_M = np.mean(res.mdl.machine.w_M[end])
    w_M_est = np.mean(res.ctrl.fbk.w_M[ctrl_end])
    tau_M = np.mean(res.mdl.machine.tau_M[end])
    f_M_ab = np.mean(res.mdl.machine.f_M_ab[end])
    f_M_ab_est = np.mean(res.ctrl.fbk.f_M_ab[ctrl_end])
    psi_s2 = np.mean(np.abs(res.mdl.machine.psi_s2_dq[end]))
    psi_s2_est = np.mean(np.abs(res.ctrl.fbk.psi_s2[ctrl_end]))
    psi_s2_ref = np.mean(res.ctrl.ref.psi_s[ctrl_end])

    assert w_M == pytest.approx(w_M_ref, rel=0.02)
    assert w_M_est == pytest.approx(w_M, rel=0.01)
    assert tau_M == pytest.approx(tau_L, abs=0.05 * 14.5)
    assert f_M_ab == pytest.approx(f_M_ref, abs=0.01 * abs(f_M_ref))
    assert f_M_ab_est == pytest.approx(f_M_ab, abs=0.01 * abs(f_M_ref))
    assert psi_s2 == pytest.approx(psi_s2_ref, rel=0.02)
    assert psi_s2_est == pytest.approx(psi_s2, rel=0.02)


# Rotor displacement, switching-cycle PWM, two-pole inductance estimate, and sensorless
CASES = {
    "eccentric": (30e-6 - 20e-6j, False, PAR.L_1, False),
    "pwm": (0j, True, PAR.L_1, False),
    "detuned": (30e-6 - 20e-6j, False, 0.8 * PAR.L_1, False),
    "sensorless": (30e-6 - 20e-6j, False, PAR.L_1, True),
}


@pytest.mark.parametrize("case", CASES)
def test_direct_force_control(case: str) -> None:
    """Direct force control with a rotor held at a fixed position."""
    q_M_ab, pwm, L_1, sensorless = CASES[case]
    radial = model.ExternalRotorPosition()
    radial.set_external_displacement(q_M_ab)
    check_force_control(
        make_drive(radial, pwm, not sensorless), make_ctrl(L_1, sensorless)
    )


# Rotor displacement, flux-vector control of the four-pole plane, and sensorless
CURRENT_BASED_CASES = {
    "eccentric": (30e-6 - 20e-6j, False, False),
    "sensorless": (30e-6 - 20e-6j, True, True),
}


@pytest.mark.parametrize("case", CURRENT_BASED_CASES)
def test_current_based_force_control(case: str) -> None:
    """Current-based force control with a rotor held at a fixed position."""
    q_M_ab, flux_vector, sensorless = CURRENT_BASED_CASES[case]
    radial = model.ExternalRotorPosition()
    radial.set_external_displacement(q_M_ab)
    mdl, ctrl = (
        make_drive(radial, False, not sensorless),
        make_current_based_ctrl(flux_vector, sensorless),
    )
    check_force_control(mdl, ctrl)


def test_discrete_current_based_force_control() -> None:
    """The discrete-time design works and rejects a mismatched sampling period."""
    T_s = 250e-6
    par_s2 = utils.fourpole_pars(PAR)

    def make(T_s_force: float) -> control.CurrentBasedForceControlSystem:
        cfg = control.CurrentVectorControllerCfg(
            i_s_max=50, psi_s_min=0.3, sensorless=False, discrete=True, T_s=T_s
        )
        return control.CurrentBasedForceControlSystem(
            control.CurrentVectorController(par_s2, cfg),
            control.CurrentBasedForceController(PAR, i_s1_max=5.0, T_s=T_s_force),
            control.SpeedController(J=0.15, alpha_s=2 * pi * 8),
        )

    radial = model.ExternalRotorPosition()
    radial.set_external_displacement(30e-6 - 20e-6j)
    check_force_control(make_drive(radial, False), make(T_s))

    ctrl = make(0.5 * T_s)
    ctrl.set_speed_ref(0.0)
    sim = model.Simulation(make_drive(radial, False), ctrl, show_progress=False)
    with pytest.raises(ValueError, match="sampling period"):
        sim.simulate(t_stop=0.01)


def test_models_and_control_systems_can_be_pickled() -> None:
    """Models and control systems can be pickled, e.g., for parallel simulations."""
    levitation_ctrl = control.LevitationController(m=10.0, alpha_q=2 * pi * 50)
    for radial in (model.ExternalRotorPosition(), model.RadialMechanics(m=10.0)):
        pickle.dumps(make_drive(radial, False))
    pickle.dumps(
        (
            make_ctrl(sensorless=True),
            make_ctrl(levitation_ctrl=levitation_ctrl),
            make_current_based_ctrl(sensorless=True),
            make_current_based_ctrl(levitation_ctrl=levitation_ctrl),
        )
    )


@pytest.mark.parametrize("current_based", [False, True])
def test_levitation_control(current_based: bool) -> None:
    """The rotor is levitated against gravity and follows its position reference."""
    m, t_stop = 10.0, 0.3
    radial = model.RadialMechanics(m=m)
    radial.set_external_force(lambda t: (t > 0.05) * -9.81j * m)
    levitation_ctrl = control.LevitationController(m=0.8 * m, alpha_q=2 * pi * 50)
    mdl = make_drive(radial, False)
    if current_based:
        ctrl = make_current_based_ctrl(levitation_ctrl=levitation_ctrl)
    else:
        ctrl = make_ctrl(levitation_ctrl=levitation_ctrl)
    q_M_ab_ref = 50e-6 + 20e-6j

    ctrl.set_speed_ref(lambda t: min(t / 0.2, 1) * 50.0)
    ctrl.set_displacement_ref(lambda t: (t > 0.15) * q_M_ab_ref)
    res = model.Simulation(mdl, ctrl, show_progress=False).simulate(t_stop=t_stop)

    q_M_ab = res.mdl.radial.q_M_ab
    assert np.max(np.abs(q_M_ab)) < 150e-6
    assert q_M_ab[-1] == pytest.approx(q_M_ab_ref, abs=1e-6)
    assert res.ctrl.fbk.f_ext_ab[-1] == pytest.approx(-9.81j * m, abs=0.01 * 9.81 * m)
    assert res.mdl.machine.f_M_ab[-1] == pytest.approx(9.81j * m, abs=0.01 * 9.81 * m)
