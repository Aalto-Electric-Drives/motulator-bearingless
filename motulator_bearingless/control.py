"""
Force control of six-phase bearingless synchronous machines.

Two force control methods are implemented. In direct force control, the radial force
and the torque are controlled directly by means of feedback linearization, based on a
nonlinear electromagnetic model [#Das2026]_. The force control law is designed in
stator coordinates and augmented with a disturbance observer. The stator flux linkages
of the two planes are estimated using a state observer in rotor coordinates. The rotor
angle and the rotor speed are either measured or estimated by the observer. In
current-based force control, the four-pole plane is controlled by a vector controller
of motulator, and the radial force reference is converted to the current reference of
the two-pole plane, which is current-controlled, in the same manner as the common and
differential modes of multi-three-phase machines are controlled in
`motulator_multiphase`. In both methods, the rotor displacement is measured, and it can
be controlled by means of an outer levitation controller.

References
----------
.. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
   bearingless synchronous machines," Proc. ICEM, 2026,
   https://doi.org/10.1109/ICEM68283.2026.11757980

.. [#Hin2018] Hinkkanen, Saarakkala, Awan, Mölsä, Tuovinen, "Observers for sensorless
   synchronous motor drives: Framework for design and analysis," IEEE Trans. Ind. Appl.,
   2018, https://doi.org/10.1109/TIA.2018.2858753

.. [#Sok2019] Sokolov, Gruber, Saarakkala, Hinkkanen, "Modeling of a bearingless
   synchronous reluctance motor with combined windings," Proc. IEEE ECCE, 2019,
   https://doi.org/10.1109/ECCE.2019.8913000

"""

from cmath import exp
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import inf, isclose, pi, sqrt
from typing import Any, Protocol

import numpy as np
from motulator.common.control import (
    PWM,
    ComplexPIController,
    ControlSystem,
    DiscreteComplexController,
    PIController,
    TimeSeries,
)
from motulator.common.utils import abc2complex
from motulator.drive.control.sm import (
    CurrentVectorController,
    CurrentVectorControllerCfg,
    FluxVectorController,
    FluxVectorControllerCfg,
    ObserverOutputs,
    ReferenceGenerator,
    SpeedController,
)

from motulator_bearingless._helpers import get_value, wrap
from motulator_bearingless.model import BearinglessDrive
from motulator_bearingless.utils import (
    J2,
    J_M,
    BearinglessMagneticModel,
    BearinglessSynchronousMachinePars,
    SaturatedBearinglessSynchronousMachinePars,
    c2v,
    fourpole_pars,
    inv_vsd,
    torque,
    v2c,
    vsd,
)

__all__ = [
    "BearinglessControlSystem",
    "BearinglessMagneticModel",
    "BearinglessSynchronousMachinePars",
    "CurrentBasedForceControlSystem",
    "CurrentBasedForceController",
    "CurrentVectorController",
    "CurrentVectorControllerCfg",
    "DirectForceControlSystem",
    "DirectForceController",
    "DirectForceControllerCfg",
    "FluxObserver",
    "FluxVectorController",
    "FluxVectorControllerCfg",
    "ForceDisturbanceObserver",
    "LevitationController",
    "SaturatedBearinglessSynchronousMachinePars",
    "SpeedController",
]

# Delay compensation factor: computational delay plus half of the sampling period
K_COMP = 1.5
# Sensitivity (A/m) of the two-pole current to the displacement, below which the machine
# is considered as not magnetized
G_MIN = 1e-6
# Inverse of the rotation matrix of the planes, J_M = blockdiag(J, 2J)
J_M_INV = np.linalg.inv(J_M)


# %%
@dataclass
class Feedbacks:
    """Feedback signals, space vectors in rotor coordinates."""

    u_dc: float = 0.0
    w_M: float = 0.0  # Measured or estimated rotor speed
    theta_M: float = 0.0  # Measured or estimated rotor angle
    w_c: float = 0.0  # Angular speed of the coordinate system
    eps: float = 0.0  # Position estimation error signal
    q_M: complex = 0j  # Measured rotor displacement
    q_M_ab: complex = 0j  # Measured rotor displacement in stator coordinates
    v_M_ab: complex = 0j  # Lateral rotor velocity estimate in stator coordinates
    f_ext_ab: complex = 0j  # External force estimate in stator coordinates
    i_s1: complex = 0j
    i_s2: complex = 0j
    u_s1: complex = 0j
    u_s2: complex = 0j
    u_s1_zoh: complex = 0j  # Voltages of the ongoing sampling period
    u_s2_zoh: complex = 0j
    psi_s1: complex = 0j  # Flux linkage estimates
    psi_s2: complex = 0j
    e_psi1: complex = 0j  # Flux estimation error signals
    e_psi2: complex = 0j
    i_a2: complex = 0j  # Auxiliary current and flux linkage of the four-pole plane
    psi_a2: complex = 0j
    f_M: complex = 0j  # Radial force estimate
    f_M_ab: complex = 0j  # Radial force estimate in stator coordinates
    d_M_ab: complex = 0j  # Force disturbance estimate (N/s) in stator coordinates
    tau_M: float = 0.0  # Torque estimate


@dataclass
class References:
    """Reference signals."""

    T_s: float = 0.0
    d_abc: Sequence[float] = field(default_factory=lambda: 6 * [0.0])
    tau_M: float = 0.0
    psi_s: float = 0.0  # Flux magnitude reference of the four-pole plane
    f_M_ab: complex = 0j  # Radial force reference in stator coordinates
    q_M_ab: complex = 0j  # Rotor displacement reference in stator coordinates
    u_s1: complex = 0j  # Voltage references in rotor coordinates
    u_s2: complex = 0j
    w_M: float | None = None


@dataclass
class ExternalReferences:
    """External reference signals."""

    w_M: float | Callable[[float], float] | None = None
    tau_M: float | Callable[[float], float] | None = None
    f_M_ab: complex | Callable[[float], complex] = 0j
    q_M_ab: complex | Callable[[float], complex] = 0j


@dataclass
class Measurements:
    """Measured signals."""

    i_c_ab: np.ndarray  # Currents of the two sets in their own coordinates
    u_dc: float
    q_M_ab: complex
    w_M: float | None = None
    theta_M: float | None = None


# %%
def aux_vectors(
    par: BearinglessMagneticModel,
    i_s1: complex,
    i_s2: complex,
    psi_s1: complex,
    psi_s2: complex,
    q_M: complex,
) -> tuple[complex, complex, complex, complex]:
    """
    Auxiliary current and flux linkage vectors.

    The auxiliary current `i_a = -i_s + J_M^-1 [(di_s/dpsi_s) J_M psi_s + (di_s/dq_M)
    J q_M]` defines the torque dynamics, `dtau_M/dt = 3 (J_M i_a)^T dpsi_s/dt`, apart
    from the displacement-induced term [#Das2026]_. The auxiliary flux linkage is
    `psi_a = J_M^-1 L_s J_M i_a`, where `L_s` is the incremental inductance matrix.

    Returns
    -------
    i_a1, i_a2, psi_a1, psi_a2 : complex
        Auxiliary vectors of the two planes.

    """
    i_s = c2v(i_s1, i_s2)
    psi_s = c2v(psi_s1, psi_s2)
    q = c2v(q_M)
    L_s = par.incr_ind_mat(i_s1, i_s2, q_M)
    i_a = -i_s + J_M_INV @ (
        np.linalg.solve(L_s, J_M @ psi_s) + par.di_dq(i_s1, i_s2, q_M) @ (J2 @ q)
    )
    psi_a = J_M_INV @ L_s @ J_M @ i_a
    i_a1, i_a2 = v2c(i_a)
    psi_a1, psi_a2 = v2c(psi_a)
    return complex(i_a1), complex(i_a2), complex(psi_a1), complex(psi_a2)


# %%
class FluxObserver:
    """
    Flux observer in rotor coordinates.

    The stator flux linkages of both planes are estimated based on the voltage model,
    corrected by the flux map of the machine [#Das2026]_. The observer gain of the
    two-pole plane places the estimation-error poles at the distance `w_M` from the
    origin with the damping ratio `zeta`. The gain of the four-pole plane projects the
    error signal onto the auxiliary flux linkage with the damping `b = 2*zeta*|w_M|`.
    The flux estimates are updated using the voltages of the ongoing sampling period.
    The voltage is constant in stator coordinates during the sampling period, so that
    its integral is exact for a constant speed of the coordinates, which avoids the
    coupling from the voltage transients to the position error signal. The resistive
    voltage drop and the correction are held constant in stator coordinates during the
    period.

    In the sensorless mode, the rotor angle and the rotor speed are estimated using the
    component of the four-pole error signal perpendicular to the auxiliary flux linkage,
    as in the public observer for synchronous machines [#Hin2018]_. The projection in
    the four-pole observer gain decouples the flux estimation from the angle estimation.

    Parameters
    ----------
    par : BearinglessMagneticModel
        Machine model parameter estimates.
    zeta : float, optional
        Damping ratio of the estimation-error dynamics, defaults to 0.45.
    sensorless : bool, optional
        If True, the rotor angle and the rotor speed are estimated, defaults to False.
    alpha_o : float, optional
        Speed-estimation pole (rad/s), defaults to 2*pi*100.

    """

    def __init__(
        self,
        par: BearinglessMagneticModel,
        zeta: float = 0.45,
        sensorless: bool = False,
        alpha_o: float = 2 * pi * 100,
    ) -> None:
        self.par = par
        self.zeta = zeta
        self.sensorless = sensorless
        # Critically damped speed and angle estimation
        self.k_theta = 2 * alpha_o
        self.k_w = alpha_o**2
        self.psi_s1: complex = 0j
        self.psi_s2: complex = 0j
        self.theta_M: float = 0.0
        self.w_M: float = 0.0

    def compute_output(
        self,
        u_s1_ab: complex,
        u_s2_ab: complex,
        i_s1_ab: complex,
        i_s2_ab: complex,
        w_M: float | None,
        theta_M: float | None,
        q_M_ab: complex,
    ) -> Feedbacks:
        """
        Compute the feedback signals in rotor coordinates.

        In the sensorless mode, the measured rotor speed `w_M` and the measured rotor
        angle `theta_M` are replaced with their estimates.

        """
        par = self.par
        if self.sensorless:
            w_M, theta_M = self.w_M, self.theta_M
        elif w_M is None or theta_M is None:
            raise ValueError("Rotor speed and angle must be provided in sensored mode")
        fbk = Feedbacks(
            w_M=w_M, theta_M=theta_M, psi_s1=self.psi_s1, psi_s2=self.psi_s2
        )
        exp_mj_theta_M = exp(-1j * theta_M)
        fbk.q_M = exp_mj_theta_M * q_M_ab
        fbk.i_s1 = exp_mj_theta_M * i_s1_ab
        fbk.i_s2 = exp_mj_theta_M**2 * i_s2_ab
        fbk.u_s1 = exp_mj_theta_M * u_s1_ab
        fbk.u_s2 = exp_mj_theta_M**2 * u_s2_ab
        # Flux map and the estimation error signals
        psi_s1_map, psi_s2_map = par.psi_s(fbk.i_s1, fbk.i_s2, fbk.q_M)
        fbk.e_psi1 = complex(psi_s1_map) - fbk.psi_s1
        fbk.e_psi2 = complex(psi_s2_map) - fbk.psi_s2
        # Force map
        fbk.f_M = complex(par.f_M(fbk.i_s1, fbk.i_s2, fbk.q_M))
        fbk.f_M_ab = fbk.f_M / exp_mj_theta_M
        # Torque estimate, including the term of the displaced rotor
        fbk.tau_M = torque(fbk.i_s1, fbk.i_s2, fbk.psi_s1, fbk.psi_s2, fbk.q_M, fbk.f_M)
        # Auxiliary vectors
        _, fbk.i_a2, _, fbk.psi_a2 = aux_vectors(
            par, fbk.i_s1, fbk.i_s2, fbk.psi_s1, fbk.psi_s2, fbk.q_M
        )
        # Position error signal (mechanical rad) from the four-pole plane
        if self.sensorless and fbk.psi_a2 != 0:
            fbk.eps = -0.5 * (fbk.e_psi2 / fbk.psi_a2).imag
        fbk.w_c = w_M + self.k_theta * fbk.eps
        return fbk

    def update(self, T_s: float, fbk: Feedbacks) -> None:
        """Update the flux estimates."""
        par, zeta, w_M = self.par, self.zeta, fbk.w_M
        # Two-pole plane: complex gain k_1 + j*k_2
        k_1 = zeta * abs(w_M)
        k_2 = w_M * (sqrt(1 - zeta**2) - 1)
        corr1 = (k_1 + 1j * k_2) * fbk.e_psi1
        # Four-pole plane: projection onto the auxiliary flux linkage
        b = 2 * zeta * abs(w_M)
        if fbk.psi_a2 != 0:
            corr2 = (
                b
                * fbk.psi_a2
                * (fbk.psi_a2.conjugate() * fbk.e_psi2).real
                / abs(fbk.psi_a2) ** 2
            )
        else:
            corr2 = b * fbk.e_psi2
        # The voltage is constant in stator coordinates during the sampling period, so
        # the flux increments are rotated to the coordinates of the next step
        d_psi_s1 = fbk.u_s1_zoh - par.R_s * fbk.i_s1 + corr1
        d_psi_s2 = fbk.u_s2_zoh - par.R_s * fbk.i_s2 + corr2
        theta = T_s * fbk.w_c
        self.psi_s1 = exp(-1j * theta) * (fbk.psi_s1 + T_s * d_psi_s1)
        self.psi_s2 = exp(-2j * theta) * (fbk.psi_s2 + T_s * d_psi_s2)
        self.theta_M = wrap(self.theta_M + T_s * fbk.w_c)
        self.w_M += T_s * self.k_w * fbk.eps


# %%
class ForceDisturbanceObserver:
    """
    Disturbance observer for the radial force dynamics.

    The force dynamics in stator coordinates are modeled as `df_M_ab/dt = w + d`, where
    `w` is the model-based part and the disturbance `d` is assumed to be constant in
    stator coordinates. The disturbance is estimated as `dd/dt = alpha_i*(df_M_ab/dt -
    w - d)`. The force derivative is avoided by means of the state `z = d -
    alpha_i*f_M_ab`. Combined with the proportional control law, the observer provides
    the integral action, cf. the disturbance-observer structure of the 2DOF PI
    controller. If the realized voltage is used in `w`, the observer is free from
    windup.

    Parameters
    ----------
    alpha_i : float
        Observer bandwidth (rad/s).

    """

    def __init__(self, alpha_i: float) -> None:
        self.alpha_i = alpha_i
        self.z: complex = 0j

    def compute_output(self, f_M_ab: complex) -> complex:
        """Compute the disturbance estimate (N/s) in stator coordinates."""
        return self.z + self.alpha_i * f_M_ab

    def update(self, T_s: float, w: complex, d: complex) -> None:
        """Update the state."""
        self.z -= T_s * self.alpha_i * (w + d)


# %%
@dataclass
class DirectForceControllerCfg:
    """
    Direct force controller configuration.

    Parameters
    ----------
    i_s_max : float
        Maximum current (A) of the four-pole plane.
    psi_s_min : float
        Minimum flux magnitude (Vs) of the four-pole plane, i.e., the bias flux for the
        force production.
    psi_s_max : float, optional
        Maximum flux magnitude (Vs) of the four-pole plane, defaults to `inf`.
    k_u : float, optional
        Voltage utilization factor of the field weakening, defaults to 0.95.
    k_mtpv : float, optional
        Margin of the maximum-torque-per-volt limit, defaults to 0.9.
    alpha_f : float, optional
        Force-control bandwidth (rad/s), defaults to 2*pi*300.
    alpha_i : float, optional
        Integral-action bandwidth (rad/s) of the force control, i.e., the bandwidth of
        the force disturbance observer, defaults to 2*pi*30. The observer is disabled
        if the value is zero.
    alpha_psi : float, optional
        Flux-control bandwidth (rad/s), defaults to 2*pi*200.
    alpha_tau : float, optional
        Torque-control bandwidth (rad/s), defaults to 2*pi*200.
    zeta : float, optional
        Damping ratio of the flux observer, defaults to 0.45.
    sensorless : bool, optional
        If True, the rotor angle and the rotor speed are estimated, defaults to False.
    alpha_o : float, optional
        Speed-estimation pole (rad/s) in the sensorless mode, defaults to 2*pi*100.
    T_s : float, optional
        Sampling period (s), defaults to 125e-6.

    """

    i_s_max: float
    psi_s_min: float
    psi_s_max: float = inf
    k_u: float = 0.95
    k_mtpv: float = 0.9
    alpha_f: float = 2 * pi * 300
    alpha_i: float = 2 * pi * 30
    alpha_psi: float = 2 * pi * 200
    alpha_tau: float = 2 * pi * 200
    zeta: float = 0.45
    sensorless: bool = False
    alpha_o: float = 2 * pi * 100
    T_s: float = 125e-6


class DirectForceController:
    """
    Direct radial force and torque controller.

    The voltage reference is `u_s = R_s i_s + w_M J_M psi_s + e`, where the control
    input `e = [e_1, e_2]` linearizes the force dynamics of the two-pole plane and the
    flux and torque dynamics of the four-pole plane [#Das2026]_. The control laws
    neglect the effect of the four-pole input `e_2` on the force and that of the
    two-pole input `e_1` on the torque, i.e., the planes are assumed to be decoupled in
    the control design, while the machine model includes the couplings. The flux and
    torque references of the four-pole plane are based on the maximum-torque-per-ampere
    and maximum-torque-per-volt loci of the magnetic model, sliced at zero two-pole
    current and at a centric rotor. The two-pole plane loads the converter only
    slightly, so that the voltage limit is assigned to the four-pole plane.

    Parameters
    ----------
    par : BearinglessMagneticModel
        Machine model parameter estimates.
    cfg : DirectForceControllerCfg
        Configuration.

    """

    def __init__(
        self, par: BearinglessMagneticModel, cfg: DirectForceControllerCfg
    ) -> None:
        self.par = par
        self.cfg = cfg
        self.observer = FluxObserver(par, cfg.zeta, cfg.sensorless, cfg.alpha_o)
        self.disturbance_observer = ForceDisturbanceObserver(cfg.alpha_i)
        # The current of the equivalent three-phase machine is twice that of the plane.
        # The current reference is not used, so its tracking bandwidth is zero.
        self.reference_gen = ReferenceGenerator(
            fourpole_pars(par),
            2 * cfg.i_s_max,
            cfg.psi_s_min,
            cfg.psi_s_max,
            cfg.k_u,
            cfg.k_mtpv,
            0.0,
        )
        # Workspace variables for the update method
        self._G_1 = np.zeros((2, 2))
        self._d_q = np.zeros(2)

    def get_feedback(
        self,
        u_s1_ab: complex,
        u_s2_ab: complex,
        i_s1_ab: complex,
        i_s2_ab: complex,
        w_M: float | None,
        theta_M: float | None,
        q_M_ab: complex,
    ) -> Feedbacks:
        """Get the feedback signals."""
        fbk = self.observer.compute_output(
            u_s1_ab, u_s2_ab, i_s1_ab, i_s2_ab, w_M, theta_M, q_M_ab
        )
        fbk.d_M_ab = self.disturbance_observer.compute_output(fbk.f_M_ab)
        return fbk

    def _force_matrices(self, fbk: Feedbacks) -> tuple[np.ndarray, np.ndarray]:
        """
        Force matrix and the known displacement-induced term.

        The force dynamics in rotor coordinates are `df_M/dt = -3*G_1*e_1 - 3*G_2*e_2 +
        (df_M/dq_M)*dq_M/dt`, where `G_h = (di_sh/dq_M)^T` and `df_M/dq_M` are evaluated
        at constant flux linkages. The cross term `-3*G_2*e_2` is neglected. It does not
        vanish even at a centric rotor with linear magnetics, since `G_2` is
        proportional to the two-pole current. The displacement derivative is `dq_M/dt =
        v_M - j*w_M*q_M`, where the lateral velocity `v_M` of the rotor is zero unless
        its estimate is available from the levitation controller.

        """
        par = self.par
        G_1 = par.di_dq(fbk.i_s1, fbk.i_s2, fbk.q_M)[0:2, :].T
        d_q_M = exp(-1j * fbk.theta_M) * fbk.v_M_ab - 1j * fbk.w_M * fbk.q_M
        d_q = par.df_dq(fbk.i_s1, fbk.i_s2, fbk.q_M) @ c2v(d_q_M)
        return G_1, d_q

    def _force_control(
        self, f_M_ab_ref: complex, fbk: Feedbacks, G_1: np.ndarray, d_q: np.ndarray
    ) -> complex:
        """
        Feedback-linearizing force control law of the two-pole plane.

        The control law is designed in stator coordinates, where the force reference
        and the disturbances are constant in the steady state. The force dynamics are
        linearized such that `df_M_ab/dt = alpha_f*(f_M_ab_ref - f_M_ab) + d - d_est`,
        where `d` is the unknown disturbance and `d_est` is its estimate. The desired
        force derivative is transformed into rotor coordinates according to `df_M/dt =
        exp(-j*theta_M)*df_M_ab/dt - j*w_M*f_M`.

        """
        # Singular force matrix if the machine is not magnetized, which appears as zero
        # or as the rounding errors of a magnetic model. For a 2x2 matrix, the ratio of
        # the determinant to the squared Frobenius norm is between 0.5/cond and 1/cond.
        det = G_1[0, 0] * G_1[1, 1] - G_1[0, 1] * G_1[1, 0]
        if np.max(np.abs(G_1)) < G_MIN or abs(det) < 1e-12 * np.sum(G_1**2):
            return 0j
        d_f_M_ab = self.cfg.alpha_f * (f_M_ab_ref - fbk.f_M_ab) - fbk.d_M_ab
        d_f_M = exp(-1j * fbk.theta_M) * d_f_M_ab - 1j * fbk.w_M * fbk.f_M
        e_1 = -np.linalg.solve(G_1, c2v(d_f_M) - d_q) / 3
        return complex(v2c(e_1)[0])

    def _flux_torque_control(
        self, psi_s_ref: float, tau_M_ref: float, fbk: Feedbacks
    ) -> complex:
        """
        Flux-vector control law of the four-pole plane.

        Apart from the displacement-induced term, the torque dynamics are `dtau_M/dt =
        3*(J*i_a1)^T*e_1 + 6*(J*i_a2)^T*e_2`, see `aux_vectors`. The term of the
        two-pole input `e_1` is neglected.

        """
        cfg = self.cfg
        psi_s2, i_a2 = fbk.psi_s2, fbk.i_a2
        c = (i_a2 * psi_s2.conjugate()).real  # Positive in the feasible region
        if c > 0:
            t_psi = abs(psi_s2) * i_a2 / c
            t_tau = 1j * psi_s2 / (6 * c)
        else:
            t_psi, t_tau = 1, 0
        e_psi = psi_s_ref - abs(psi_s2)
        e_tau = tau_M_ref - fbk.tau_M
        return cfg.alpha_psi * e_psi * t_psi + cfg.alpha_tau * e_tau * t_tau

    def compute_output(
        self, tau_M_ref: float, f_M_ab_ref: complex, fbk: Feedbacks
    ) -> References:
        """Compute the references."""
        par = self.par
        ref = References(T_s=self.cfg.T_s, f_M_ab=f_M_ab_ref)
        ref.psi_s, ref.tau_M = self.reference_gen.compute_flux_and_torque_refs(
            tau_M_ref, 2 * fbk.w_M, fbk.u_dc
        )
        self._G_1, self._d_q = self._force_matrices(fbk)
        e_1 = self._force_control(f_M_ab_ref, fbk, self._G_1, self._d_q)
        e_2 = self._flux_torque_control(ref.psi_s, ref.tau_M, fbk)
        ref.u_s1 = par.R_s * fbk.i_s1 + 1j * fbk.w_M * fbk.psi_s1 + e_1
        ref.u_s2 = par.R_s * fbk.i_s2 + 2j * fbk.w_M * fbk.psi_s2 + e_2
        return ref

    def update(self, ref: References, fbk: Feedbacks) -> None:
        """Update the states."""
        self.observer.update(ref.T_s, fbk)
        # Model-based part of the force derivative, using the realized voltage and the
        # force matrices of compute_output
        par, G_1, d_q = self.par, self._G_1, self._d_q
        e_1 = fbk.u_s1 - par.R_s * fbk.i_s1 - 1j * fbk.w_M * fbk.psi_s1
        d_f_M = complex(v2c(-3 * G_1 @ c2v(e_1) + d_q)[0]) + 1j * fbk.w_M * fbk.f_M
        self.disturbance_observer.update(
            ref.T_s, exp(1j * fbk.theta_M) * d_f_M, fbk.d_M_ab
        )


# %%
class LevitationController:
    """
    Rotor displacement controller.

    The radial dynamics of the rotor in stator coordinates are modeled as `m*dv_M_ab/dt
    = f_M_ab + f_ext_ab` and `dq_M_ab/dt = v_M_ab`, where the external force `f_ext_ab`
    (e.g., gravity) is a constant disturbance. The displacement, the lateral velocity,
    and the external force are estimated by a disturbance observer from the measured
    displacement and the radial force estimate. The force reference is obtained from
    the state feedback law::

        f_M_ab_ref = m*(alpha_q**2*(q_M_ab_ref - q_M_ab) - 2*alpha_q*v_M_ab) - f_ext_ab

    which results in the double closed-loop pole at `s = -alpha_q`. The control law has
    no memory: the integral action is provided by the observer. Since the observer is
    driven by the radial force estimate, the dynamics and the limits of the inner force
    control loop do not cause windup.

    Parameters
    ----------
    m : float
        Rotor mass estimate (kg).
    alpha_q : float
        Displacement-control bandwidth (rad/s).
    alpha_o : float, optional
        Triple pole (rad/s) of the observer, defaults to `3*alpha_q`.

    """

    def __init__(self, m: float, alpha_q: float, alpha_o: float | None = None) -> None:
        self.m = m
        self.alpha_q = alpha_q
        self.alpha_o = 3 * alpha_q if alpha_o is None else alpha_o
        self.q_M_ab: complex = 0j
        self.v_M_ab: complex = 0j
        self.f_ext_ab: complex = 0j

    def compute_output(self, q_M_ab_ref: complex) -> complex:
        """Compute the radial force reference (N) in stator coordinates."""
        m, alpha_q = self.m, self.alpha_q
        a_ref = alpha_q**2 * (q_M_ab_ref - self.q_M_ab) - 2 * alpha_q * self.v_M_ab
        return m * a_ref - self.f_ext_ab

    def update(self, T_s: float, q_M_ab: complex, f_M_ab: complex) -> None:
        """Update the state estimates."""
        m, alpha_o = self.m, self.alpha_o
        e = q_M_ab - self.q_M_ab
        d_q_M_ab = self.v_M_ab + 3 * alpha_o * e
        d_v_M_ab = (f_M_ab + self.f_ext_ab) / m + 3 * alpha_o**2 * e
        d_f_ext_ab = m * alpha_o**3 * e
        self.q_M_ab += T_s * d_q_M_ab
        self.v_M_ab += T_s * d_v_M_ab
        self.f_ext_ab += T_s * d_f_ext_ab


# %%
class BaseReferences(Protocol):
    """Reference signals required by the control system base class."""

    T_s: float
    d_abc: Sequence[float]


class BearinglessControlSystem[Ref: BaseReferences, Fbk](
    ControlSystem[BearinglessDrive, Measurements, Ref, Fbk]
):
    """
    Base class of the bearingless control systems.

    This class implements the parts common to the bearingless control systems: the
    measurements, the modulators of the two sets, and the external references. The
    speed and torque references work as in the public `VectorControlSystem`. In the
    force-control mode, the radial force reference is given externally, and in the
    levitation-control mode, the levitation controller gives the force reference based
    on the displacement reference. Subclasses implement `get_feedback`,
    `compute_output`, and `update`. Gate blocking is not supported, since the
    multi-three-phase converter model does not support it.

    Parameters
    ----------
    speed_ctrl : SpeedController | PIController | None
        Speed controller. If None, torque-control mode is used.
    levitation_ctrl : LevitationController | None
        Levitation controller. If None, force-control mode is used.

    """

    def __init__(
        self,
        speed_ctrl: SpeedController | PIController | None,
        levitation_ctrl: LevitationController | None,
    ) -> None:
        super().__init__()
        self.speed_ctrl = speed_ctrl
        self.levitation_ctrl = levitation_ctrl
        self.pwms = [PWM(), PWM()]
        self.ext_ref = ExternalReferences()

    def set_speed_ref(self, ref_fcn: float | Callable[[float], float]) -> None:
        """
        Set the external speed reference for speed-control mode.

        Parameters
        ----------
        ref_fcn : float | Callable[[float], float]
            Speed reference (mechanical rad/s), constant or a function of time.

        """
        if self.speed_ctrl is None:
            raise ValueError("Speed-control mode requires speed_ctrl")
        self.ext_ref.w_M = ref_fcn

    def set_torque_ref(self, ref_fcn: float | Callable[[float], float]) -> None:
        """
        Set the external torque reference for torque-control mode.

        Parameters
        ----------
        ref_fcn : float | Callable[[float], float]
            Torque reference (Nm), constant or a function of time.

        """
        if self.speed_ctrl is not None:
            raise ValueError("The speed controller gives the torque reference")
        self.ext_ref.tau_M = ref_fcn

    def set_force_ref(self, ref_fcn: complex | Callable[[float], complex]) -> None:
        """
        Set the external radial force reference for force-control mode.

        Parameters
        ----------
        ref_fcn : complex | Callable[[float], complex]
            Radial force reference (N) in stator coordinates, constant or a function
            of time. Defaults to zero.

        """
        if self.levitation_ctrl is not None:
            raise ValueError("The levitation controller gives the force reference")
        self.ext_ref.f_M_ab = ref_fcn

    def set_displacement_ref(
        self, ref_fcn: complex | Callable[[float], complex]
    ) -> None:
        """
        Set the external displacement reference for levitation-control mode.

        Parameters
        ----------
        ref_fcn : complex | Callable[[float], complex]
            Rotor displacement reference (m) in stator coordinates, constant or a
            function of time. Defaults to zero, i.e., a centric rotor.

        """
        if self.levitation_ctrl is None:
            raise ValueError("Levitation-control mode requires levitation_ctrl")
        self.ext_ref.q_M_ab = ref_fcn

    def get_measurement(self, mdl: BearinglessDrive) -> Measurements:
        """Sample the sensors, the speed and position sensors if the drive has them."""
        sensors = mdl.sensors
        return Measurements(
            i_c_ab=abc2complex(sensors.i_c.sample()),
            u_dc=sensors.u_dc.sample(),
            q_M_ab=sensors.q_M.sample(),
            w_M=None if sensors.w_M is None else sensors.w_M.sample(),
            theta_M=None if sensors.theta_M is None else wrap(sensors.theta_M.sample()),
        )

    def get_command(self, ref: Ref) -> tuple[float, Sequence[float]]:
        """Extract the sampling period and the duty ratios from the references."""
        return ref.T_s, ref.d_abc

    def compute_standby_output(self, fbk: Fbk) -> Ref:
        """Compute the controller output while the gates are blocked."""
        # The multi-three-phase converter model does not support gate blocking. The
        # message is a separate statement, so that the method is not documented as an
        # abstract method.
        msg = "Bearingless control does not support gate blocking"
        raise NotImplementedError(msg)

    def _realized_voltages(
        self, meas: Measurements, average: bool
    ) -> tuple[complex, complex]:
        """Realized voltages of the two planes in stator coordinates."""
        u_c_ab = [
            pwm.get_realized_voltage(i, meas.u_dc, average=average)
            for pwm, i in zip(self.pwms, meas.i_c_ab, strict=True)
        ]
        return vsd(*u_c_ab)

    def _compute_torque_ref(self, w_M: float) -> tuple[float | None, float]:
        """Speed reference (None in torque-control mode) and torque reference."""
        ext_ref = self.ext_ref
        if self.speed_ctrl is not None:  # Speed-control mode
            if ext_ref.w_M is None:
                raise ValueError("Speed reference must be set")
            w_M_ref = get_value(ext_ref.w_M, self.t)
            return w_M_ref, self.speed_ctrl.compute_output(w_M_ref, w_M)
        if ext_ref.tau_M is None:  # Torque-control mode
            raise ValueError("Torque reference must be set")
        return None, get_value(ext_ref.tau_M, self.t)

    def _compute_force_ref(self) -> tuple[complex, complex]:
        """Displacement reference and radial force reference in stator coordinates."""
        q_M_ab_ref = complex(get_value(self.ext_ref.q_M_ab, self.t))
        if self.levitation_ctrl is not None:  # Levitation-control mode
            return q_M_ab_ref, self.levitation_ctrl.compute_output(q_M_ab_ref)
        return q_M_ab_ref, complex(get_value(self.ext_ref.f_M_ab, self.t))

    def _modulate(
        self, T_s: float, u_s1_ab_ref: complex, u_s2_ab_ref: complex, u_dc: float
    ) -> list[float]:
        """Duty ratios of the two sets from the voltage references of the planes."""
        # The set currents consist of components rotating in opposite directions, so
        # the modulators do not predict them (the angular speed is set to zero)
        return [
            d
            for pwm, u in zip(self.pwms, inv_vsd(u_s1_ab_ref, u_s2_ab_ref), strict=True)
            for d in pwm(T_s, u, u_dc, 0.0)
        ]

    def _update_outer_controllers(
        self, T_s: float, tau_M_ref: float, q_M_ab: complex, f_M_ab: complex
    ) -> None:
        """Update the speed and levitation controllers."""
        if self.speed_ctrl is not None:
            self.speed_ctrl.update(T_s, tau_M_ref)
        if self.levitation_ctrl is not None:
            self.levitation_ctrl.update(T_s, q_M_ab, f_M_ab)


# %%
class DirectForceControlSystem(BearinglessControlSystem[References, Feedbacks]):
    """
    Direct force control system for six-phase bearingless machine drives.

    The reference modes are described in :class:`BearinglessControlSystem`.

    Parameters
    ----------
    force_ctrl : DirectForceController
        Inner force and torque controller.
    speed_ctrl : SpeedController | PIController | None, optional
        Speed controller. If None (default), torque-control mode is used.
    levitation_ctrl : LevitationController | None, optional
        Levitation controller. If None (default), force-control mode is used.

    """

    def __init__(
        self,
        force_ctrl: DirectForceController,
        speed_ctrl: SpeedController | PIController | None = None,
        levitation_ctrl: LevitationController | None = None,
    ) -> None:
        super().__init__(speed_ctrl, levitation_ctrl)
        self.force_ctrl = force_ctrl

    def get_feedback(self, meas: Measurements) -> Feedbacks:
        """Get the feedback signals."""
        u_s1_ab, u_s2_ab = self._realized_voltages(meas, average=True)
        i_s1_ab, i_s2_ab = vsd(*meas.i_c_ab)
        fbk = self.force_ctrl.get_feedback(
            u_s1_ab, u_s2_ab, i_s1_ab, i_s2_ab, meas.w_M, meas.theta_M, meas.q_M_ab
        )
        # Voltages of the ongoing sampling period for the flux observer
        u_s1_ab, u_s2_ab = self._realized_voltages(meas, average=False)
        fbk.u_s1_zoh = exp(-1j * fbk.theta_M) * u_s1_ab
        fbk.u_s2_zoh = exp(-2j * fbk.theta_M) * u_s2_ab
        fbk.u_dc = meas.u_dc
        fbk.q_M_ab = meas.q_M_ab
        if self.levitation_ctrl is not None:
            fbk.v_M_ab = self.levitation_ctrl.v_M_ab
            fbk.f_ext_ab = self.levitation_ctrl.f_ext_ab
        return fbk

    def compute_output(self, fbk: Feedbacks) -> References:
        """Compute the controller output."""
        w_M_ref, tau_M_ref = self._compute_torque_ref(fbk.w_M)
        q_M_ab_ref, f_M_ab_ref = self._compute_force_ref()
        ref = self.force_ctrl.compute_output(tau_M_ref, f_M_ab_ref, fbk)
        ref.q_M_ab, ref.w_M = q_M_ab_ref, w_M_ref
        # Stator coordinates, with the angular delay compensation. In the steady state,
        # the two-pole field producing a force constant in stator coordinates rotates
        # at the same electrical angular speed 2*w_M as the four-pole field. Hence,
        # the angles of both planes are advanced by the same amount.
        theta_comp = 2 * K_COMP * ref.T_s * fbk.w_M
        u_s1_ab = exp(1j * (fbk.theta_M + theta_comp)) * ref.u_s1
        u_s2_ab = exp(1j * (2 * fbk.theta_M + theta_comp)) * ref.u_s2
        ref.d_abc = self._modulate(ref.T_s, u_s1_ab, u_s2_ab, fbk.u_dc)
        return ref

    def update(self, ref: References, fbk: Feedbacks) -> None:
        """Update the controller states."""
        super().update(ref, fbk)
        self.force_ctrl.update(ref, fbk)
        self._update_outer_controllers(ref.T_s, ref.tau_M, fbk.q_M_ab, fbk.f_M_ab)


# %%
class VectorReferences(BaseReferences, Protocol):
    """Reference signals of the vector controllers of motulator."""

    tau_M: float  # Torque reference
    psi_s: float  # Stator flux magnitude reference
    u_s: complex  # Stator voltage reference
    u_s_lim: complex  # Limited voltage reference (set by the control system)
    w_M: float | None  # Speed reference (set by the control system)


class VectorController(Protocol):
    """Interface of the vector controllers of motulator for synchronous machines."""

    sensorless: bool
    k_comp: float  # Angle-advance factor

    def get_feedback(
        self,
        u_s_ab: complex,
        i_s_ab: complex,
        w_M_meas: float | None,
        theta_M_meas: float | None,
    ) -> ObserverOutputs:
        """Get feedback signals from measurements."""
        ...

    def compute_output(
        self, tau_M_ref: float, fbk: ObserverOutputs
    ) -> VectorReferences:
        """Compute control output from feedback signals."""
        ...

    def update(self, ref: Any, fbk: ObserverOutputs) -> None:
        """Update controller states."""
        ...

    def post_process(self, ts: TimeSeries) -> None:
        """Post-process controller outputs."""
        ...


@dataclass
class ForceFeedbacks:
    """Feedback signals of the force control in controller coordinates."""

    i_s1: complex = 0j  # Two-pole current
    i_s2: complex = 0j  # Four-pole current, half of the equivalent machine current
    u_s1: complex = 0j  # Realized two-pole voltage
    f_M_ab: complex = 0j  # Radial force estimate in stator coordinates
    q_M_ab: complex = 0j  # Measured rotor displacement in stator coordinates
    v_M_ab: complex = 0j  # Lateral rotor velocity estimate in stator coordinates
    f_ext_ab: complex = 0j  # External force estimate in stator coordinates


@dataclass
class ForceReferences:
    """Reference signals of the force control in controller coordinates."""

    i_s1: complex = 0j  # Two-pole current reference
    u_s1: complex = 0j  # Two-pole voltage reference
    u_s1_lim: complex = 0j  # Limited voltage reference
    f_M_ab: complex = 0j  # Radial force reference in stator coordinates
    q_M_ab: complex = 0j  # Rotor displacement reference in stator coordinates


class CurrentBasedForceController:
    """
    Current-based radial force controller of the two-pole plane.

    The radial force reference is converted to the current reference of the two-pole
    plane, and the current is controlled by a complex-vector PI controller, in the same
    manner as the differential-mode currents of multi-three-phase machines are
    controlled in `motulator_multiphase`. The two-pole plane is modeled as a
    resistive-inductive circuit with the inductance `L_1`, and its current is controlled
    in the coordinates of the vector controller of the four-pole plane. In these
    coordinates, the force-producing current is constant in the steady state, and the
    radial force in stator coordinates is given by the linear force model [#Sok2019]_::

        f_M_ab = 3*k*conj(i_s1)    k = m_d*i_s2d + 1j*m_q*i_s2q

    where `i_s2` is the current of the four-pole plane in rotor coordinates. The current
    reference is obtained by inverting this model, and the force estimate for the
    levitation control is computed from the measured currents. Magnetic saturation and
    the eccentricity effects on the inductances are not taken into account, cf.
    :class:`DirectForceController`.

    Parameters
    ----------
    par : BearinglessSynchronousMachinePars
        Machine model parameter estimates.
    alpha_c : float, optional
        Current-control bandwidth (rad/s), defaults to 2*pi*200.
    alpha_i : float, optional
        Integral-action bandwidth (rad/s), defaults to `alpha_c`. Not used in the
        direct discrete-time design.
    i_s1_max : float, optional
        Maximum current (A) of the two-pole plane, defaults to `inf`.
    T_s : float, optional
        Sampling period (s). If given, the direct discrete-time design is used instead
        of the continuous-time design. The vector controller must then be a direct
        discrete-time design as well (`k_comp = 0`), since the angle advance of the
        voltage references is common to both planes, and its sampling period must
        equal this value.

    """

    def __init__(
        self,
        par: BearinglessSynchronousMachinePars,
        alpha_c: float = 2 * pi * 200,
        alpha_i: float | None = None,
        i_s1_max: float = inf,
        T_s: float | None = None,
    ) -> None:
        self.par = par
        self.i_s1_max = i_s1_max
        self.T_s = T_s
        self.discrete = T_s is not None
        self.current_ctrl: ComplexPIController | DiscreteComplexController
        if T_s is None:
            alpha_i = alpha_c if alpha_i is None else alpha_i
            k_p, k_i = alpha_c + alpha_i, alpha_c * alpha_i
            self.current_ctrl = ComplexPIController(k_p, k_i, alpha_c)
        else:
            self.current_ctrl = DiscreteComplexController(alpha_c, T_s)

    def _force_constant(self, i_s2: complex) -> complex:
        """Force constant `k` (N/A per 3) of the linear force model."""
        return self.par.m_d * i_s2.real + 1j * self.par.m_q * i_s2.imag

    def get_feedback(
        self,
        i_s1_ab: complex,
        u_s1_ab: complex,
        i_s2: complex,
        theta_c: float,
        q_M_ab: complex,
    ) -> ForceFeedbacks:
        """
        Get the feedback signals.

        Parameters
        ----------
        i_s1_ab : complex
            Two-pole current (A) in stator coordinates.
        u_s1_ab : complex
            Realized two-pole voltage (V) in stator coordinates.
        i_s2 : complex
            Four-pole current (A) in the coordinates of the vector controller.
        theta_c : float
            Angle (rad) of the coordinates of the vector controller.
        q_M_ab : complex
            Measured rotor displacement (m) in stator coordinates.

        """
        exp_mj_theta_c = exp(-1j * theta_c)
        fbk = ForceFeedbacks(
            i_s1=exp_mj_theta_c * i_s1_ab,
            i_s2=i_s2,
            u_s1=exp_mj_theta_c * u_s1_ab,
            q_M_ab=q_M_ab,
        )
        fbk.f_M_ab = 3 * self._force_constant(i_s2) * fbk.i_s1.conjugate()
        return fbk

    def compute_output(
        self, f_M_ab_ref: complex, fbk: ForceFeedbacks
    ) -> ForceReferences:
        """Compute the current and voltage references of the two-pole plane."""
        ref = ForceReferences(f_M_ab=f_M_ab_ref)
        k = self._force_constant(fbk.i_s2)
        # No force can be produced if the machine is not magnetized
        i_s1_ref = (f_M_ab_ref / (3 * k)).conjugate() if k != 0 else 0j
        if abs(i_s1_ref) > self.i_s1_max:
            i_s1_ref *= self.i_s1_max / abs(i_s1_ref)
        ref.i_s1 = i_s1_ref
        L_1 = self.par.L_1
        ref.u_s1 = self.current_ctrl.compute_output(L_1 * i_s1_ref, L_1 * fbk.i_s1)
        return ref

    def update(
        self, T_s: float, ref: ForceReferences, fbk: ForceFeedbacks, w_c: float
    ) -> None:
        """Update the integral state using the realized or limited voltage."""
        if self.T_s is not None and not isclose(T_s, self.T_s):
            raise ValueError(
                "The sampling period must equal the design value of the direct "
                "discrete-time force controller"
            )
        u_s1 = ref.u_s1_lim if self.discrete else fbk.u_s1
        self.current_ctrl.update(T_s, u_s1, w_c)


# %%
class CurrentBasedForceControlSystem(
    BearinglessControlSystem[VectorReferences, ObserverOutputs]
):
    """
    Current-based force control system for six-phase bearingless machine drives.

    The four-pole plane is controlled by a vector controller of motulator for
    synchronous machines, configured with the parameters of the equivalent three-phase
    machine, see :func:`~motulator_bearingless.utils.fourpole_pars`. The feedback
    signals of the vector controller are the voltage of the four-pole plane and twice
    its current, i.e., the sum of the set currents, cf. the common mode of
    multi-three-phase machines in `motulator_multiphase`. The two-pole plane is
    controlled by the current-based force controller in the coordinates of the vector
    controller, cf. the differential mode. Both planes rotate at the same electrical
    angular speed in stator coordinates, so the angle advance `k_comp*T_s*w_c` of the
    voltage references is common to them, as in the public `VectorControlSystem`. The
    reference modes are described in :class:`BearinglessControlSystem`.

    The feedback and reference signals of the vector controller are logged in the
    signal groups `fbk` and `ref`, and those of the force control in the groups
    `force_fbk` and `force_ref`. In post-processing, the latter are merged into the
    former, and the four-pole flux linkage `psi_s2` is added, so that the time series
    have the same layout as those of :class:`DirectForceControlSystem`. The space
    vectors of both planes are in the coordinates of the vector controller.

    Parameters
    ----------
    vector_ctrl : VectorController
        Vector controller of the four-pole plane, configured with the parameters of the
        equivalent three-phase machine.
    force_ctrl : CurrentBasedForceController
        Current-based force controller of the two-pole plane.
    speed_ctrl : SpeedController | PIController | None, optional
        Speed controller. If None (default), torque-control mode is used.
    levitation_ctrl : LevitationController | None, optional
        Levitation controller. If None (default), force-control mode is used.

    """

    def __init__(
        self,
        vector_ctrl: VectorController,
        force_ctrl: CurrentBasedForceController,
        speed_ctrl: SpeedController | PIController | None = None,
        levitation_ctrl: LevitationController | None = None,
    ) -> None:
        super().__init__(speed_ctrl, levitation_ctrl)
        if force_ctrl.discrete and vector_ctrl.k_comp != 0:
            raise ValueError(
                "The direct discrete-time force controller requires a direct "
                "discrete-time vector controller (k_comp = 0)"
            )
        self.vector_ctrl = vector_ctrl
        self.force_ctrl = force_ctrl
        self.k_comp = vector_ctrl.k_comp
        self.force_fbk = ForceFeedbacks()
        self.force_ref = ForceReferences()

    def get_feedback(self, meas: Measurements) -> ObserverOutputs:
        """Get the feedback signals."""
        i_s1_ab, i_s2_ab = vsd(*meas.i_c_ab)
        # The observer uses the voltage of the ongoing sampling period, while the
        # control laws use the voltage averaged over the previous and ongoing periods
        _, u_s2_ab_zoh = self._realized_voltages(meas, average=False)
        u_s1_ab, u_s2_ab = self._realized_voltages(meas, average=True)
        # The equivalent machine of the four-pole plane carries the sum of the set
        # currents. The rotor speed and angle are not used in the sensorless mode.
        if self.vector_ctrl.sensorless:
            w_M, theta_M = None, None
        else:
            w_M, theta_M = meas.w_M, meas.theta_M
        fbk = self.vector_ctrl.get_feedback(u_s2_ab_zoh, 2 * i_s2_ab, w_M, theta_M)
        fbk.u_s = exp(-1j * fbk.theta_c) * u_s2_ab
        fbk.u_dc = meas.u_dc
        self.force_fbk = self.force_ctrl.get_feedback(
            i_s1_ab, u_s1_ab, fbk.i_s / 2, fbk.theta_c, meas.q_M_ab
        )
        if self.levitation_ctrl is not None:
            self.force_fbk.v_M_ab = self.levitation_ctrl.v_M_ab
            self.force_fbk.f_ext_ab = self.levitation_ctrl.f_ext_ab
        return fbk

    def compute_output(self, fbk: ObserverOutputs) -> VectorReferences:
        """Compute the controller output."""
        w_M_ref, tau_M_ref = self._compute_torque_ref(fbk.w_M)
        q_M_ab_ref, f_M_ab_ref = self._compute_force_ref()
        ref = self.vector_ctrl.compute_output(tau_M_ref, fbk)
        ref.w_M = w_M_ref
        force_ref = self.force_ctrl.compute_output(f_M_ab_ref, self.force_fbk)
        force_ref.q_M_ab = q_M_ab_ref
        # Angle-advanced voltage references in stator coordinates
        rot = exp(1j * (fbk.theta_c + self.k_comp * ref.T_s * fbk.w_c))
        ref.d_abc = self._modulate(
            ref.T_s, rot * force_ref.u_s1, rot * ref.u_s, fbk.u_dc
        )
        # Limited voltages in the controller coordinates
        u_s1_lim_ab, u_s2_lim_ab = vsd(*[pwm.limited_voltage for pwm in self.pwms])
        exp_mj_theta_c = exp(-1j * fbk.theta_c)
        ref.u_s_lim = exp_mj_theta_c * u_s2_lim_ab
        force_ref.u_s1_lim = exp_mj_theta_c * u_s1_lim_ab
        self.force_ref = force_ref
        return ref

    def save(self, t: float, **signal_groups: Any) -> None:
        """Extend the base class method by saving the force control signals."""
        super().save(
            t, force_fbk=self.force_fbk, force_ref=self.force_ref, **signal_groups
        )

    def update(self, ref: VectorReferences, fbk: ObserverOutputs) -> None:
        """Update the controller states."""
        super().update(ref, fbk)
        self.vector_ctrl.update(ref, fbk)
        self.force_ctrl.update(ref.T_s, self.force_ref, self.force_fbk, fbk.w_c)
        self._update_outer_controllers(
            ref.T_s, ref.tau_M, self.force_fbk.q_M_ab, self.force_fbk.f_M_ab
        )

    def post_process(self) -> TimeSeries:
        """Extend the post-process method."""
        ts = super().post_process()
        self.vector_ctrl.post_process(ts)
        # Merge the force control signals into the feedback and reference groups
        for group, name in ((ts.fbk, "force_fbk"), (ts.ref, "force_ref")):
            for key, value in vars(getattr(ts, name)).items():
                setattr(group, key, value)
            delattr(ts, name)
        ts.fbk.psi_s2 = ts.fbk.psi_s
        return ts
