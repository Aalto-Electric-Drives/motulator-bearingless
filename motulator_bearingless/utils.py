"""
Utilities for six-phase bearingless machine drives.

The utilities include the machine parameters and the magnetic models [#Das2026]_,
[#Sok2019]_, the vector-space decomposition of the six-phase winding [#Ahm2023]_, and
plotting functions. The base-value classes and the plot style of motulator are
re-exported for convenience.

References
----------
.. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
   bearingless synchronous machines," Proc. ICEM, 2026,
   https://doi.org/10.1109/ICEM68283.2026.11757980

.. [#Sok2019] Sokolov, Gruber, Saarakkala, Hinkkanen, "Modeling of a bearingless
   synchronous reluctance motor with combined windings," Proc. IEEE ECCE, 2019,
   https://doi.org/10.1109/ECCE.2019.8913000

.. [#Ahm2023] Ahmed, Martin, Das, Hinkkanen, "Analytical modeling of a six-phase
   bearingless synchronous reluctance machine using winding function theory," Proc. IEEE
   ECCE, 2023, https://doi.org/10.1109/ECCE53617.2023.10361945

"""

from cmath import exp, pi
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
from motulator.common.model import SimulationResults
from motulator.common.utils import BaseValues, NominalValues, complex2abc, set_style
from motulator.drive.model import (
    SaturatedSynchronousMachinePars,
    SynchronousMachinePars,
)
from motulator.drive.utils import MachineCharacteristics
from motulator_multiphase.utils import to_parallel

from motulator_bearingless._helpers import (
    configure_axes,
    save_and_show,
    setup_time_series_plot,
)

__all__ = [
    "ALPHA",
    "J2",
    "J_M",
    "BaseBearinglessMachinePars",
    "BaseValues",
    "BearinglessMagneticModel",
    "BearinglessSynchronousMachinePars",
    "FluxMapFcn",
    "MachineCharacteristics",
    "NominalValues",
    "SaturatedBearinglessSynchronousMachinePars",
    "SpatialSaturatedBearinglessSynchronousMachinePars",
    "c2v",
    "fourpole_pars",
    "inv_vsd",
    "plot",
    "plot_phase_currents",
    "set_style",
    "sixphase_currents",
    "torque",
    "v2c",
    "vsd",
]

# %%
# Skew-symmetric matrices of the plane rotations, J_M = blockdiag(J, 2J)
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])
J_M = np.block([[J2, np.zeros((2, 2))], [np.zeros((2, 2)), 2 * J2]])


def c2v(*z: Any) -> np.ndarray:
    """Stack complex numbers (or arrays) into a real vector along the last axis."""
    if all(np.ndim(x) == 0 for x in z):
        # Faster for scalars, e.g., in the control systems
        return np.array([c for x in z for c in (np.real(x), np.imag(x))])
    return np.stack([f(x) for x in z for f in (np.real, np.imag)], axis=-1)


def v2c(v: np.ndarray) -> tuple[Any, ...]:
    """Inverse of `c2v`."""
    return tuple(
        v[..., 2 * k] + 1j * v[..., 2 * k + 1] for k in range(v.shape[-1] // 2)
    )


# %%
def torque(i_s1: Any, i_s2: Any, psi_s1: Any, psi_s2: Any, q_M: Any, f_M: Any) -> Any:
    """Electromagnetic torque (Nm) without the term due to the spatial harmonics."""
    return (
        3 * np.imag(i_s1 * np.conj(psi_s1))
        + 6 * np.imag(i_s2 * np.conj(psi_s2))
        + np.imag(q_M * np.conj(f_M))
    )


class BaseBearinglessMachinePars(Protocol):
    """
    Base class for the machine parameters, as used by the machine model.

    The magnetic model is a lossless mapping between the stator flux linkages, the
    stator currents, the rotor displacement, and the radial force, consistent with a
    magnetic energy function `W(psi_s, q_M, theta_M)`. All space vectors are in rotor
    coordinates, and the methods accept scalars as well as arrays of the same shape.

    """

    R_s: float

    def magnetic_map(
        self, psi_s1: Any, psi_s2: Any, q_M: Any, exp_j_theta_M: Any = None
    ) -> tuple[Any, Any, Any, Any]:
        """
        Currents, radial force, and torque from the flux linkages.

        Parameters
        ----------
        psi_s1, psi_s2 : complex | ndarray
            Stator flux linkages (Vs) of the two-pole and four-pole planes.
        q_M : complex | ndarray
            Rotor displacement (m).
        exp_j_theta_M : complex | ndarray, optional
            Complex exponential of the mechanical rotor angle, needed by models with
            spatial harmonics.

        Returns
        -------
        i_s1, i_s2, f_M : complex | ndarray
            Stator currents (A) of the two planes and the radial force (N).
        tau_M : float | ndarray
            Electromagnetic torque (Nm).

        """
        i_s1, i_s2, f_M = self.i_s(psi_s1, psi_s2, q_M, exp_j_theta_M)
        return i_s1, i_s2, f_M, torque(i_s1, i_s2, psi_s1, psi_s2, q_M, f_M)

    def i_s(
        self, psi_s1: Any, psi_s2: Any, q_M: Any, exp_j_theta_M: Any = None
    ) -> tuple[Any, Any, Any]:
        """Currents (A) and radial force (N) from flux linkages (Vs)."""
        ...


class BearinglessMagneticModel(BaseBearinglessMachinePars, Protocol):
    """
    Base class for the machine parameters, as used by the control system.

    In addition to the machine model, the control system needs the flux map `psi_s`, the
    force map `f_M`, the incremental inductance matrix `incr_ind_mat`, and the
    sensitivities `di_dq` and `df_dq` to the displacement, all as functions of the
    currents and the displacement.

    """

    def psi_s(self, i_s1: Any, i_s2: Any, q_M: Any) -> tuple[Any, Any]:
        """Flux linkages (Vs) as functions of currents (A) and displacement (m)."""
        ...

    def f_M(self, i_s1: Any, i_s2: Any, q_M: Any) -> Any:
        """Radial force (N) as a function of currents (A) and displacement (m)."""
        ...

    def incr_ind_mat(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """Incremental inductance matrix (H), shape (..., 4, 4), `d psi_s/d i_s`."""
        ...

    def di_dq(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """Current sensitivity `d i_s/d q_M` at constant flux, shape (..., 4, 2)."""
        ...

    def df_dq(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """Force sensitivity `d f_M/d q_M` at constant flux, shape (..., 2, 2)."""
        ...


@dataclass
class BearinglessSynchronousMachinePars(BearinglessMagneticModel):
    """
    Constant parameters of a six-phase bearingless synchronous machine.

    The four-pole rotor produces the torque with the four-pole field (`h = 2`) and
    the radial force with the two-pole field (`h = 1`). Linear magnetics are
    assumed. The flux linkages are `psi_s = L_s(q_M)*i_s`, where the inductance
    matrix depends linearly on the rotor displacement via the radial force constants::

        L_s = [[L_1*I, L_21.T], [L_21, diag(L_d, L_q)]]
        L_21 = [[m_d*q_i, -m_d*q_j], [m_q*q_j, m_q*q_i]]

    The corresponding radial force is `f_M = 1.5*i_s.T*(dL_s/dq_M)*i_s`, i.e.::

        f_i = 3*(m_d*i_2d*i_1d + m_q*i_2q*i_1q)
        f_j = 3*(m_q*i_2q*i_1d - m_d*i_2d*i_1q)

    Parameters
    ----------
    R_s : float
        Stator resistance (Ω).
    L_1 : float
        Inductance (H) of the two-pole plane, isotropic for the four-pole rotor.
    L_d : float
        d-axis inductance (H) of the four-pole plane.
    L_q : float
        q-axis inductance (H) of the four-pole plane.
    m_d : float
        d-axis radial force constant (H/m).
    m_q : float
        q-axis radial force constant (H/m).

    Notes
    -----
    The negative stiffness (unbalanced magnetic pull proportional to the displacement)
    is a second-order effect of the displacement on the inductances, which is not
    included in this linear model.

    """

    R_s: float
    L_1: float
    L_d: float
    L_q: float
    m_d: float
    m_q: float

    def ind_mat(self, q_M: Any) -> np.ndarray:
        """Inductance matrix (H), shape (..., 4, 4), at the displacement `q_M`."""
        q_i, q_j = np.real(q_M), np.imag(q_M)
        z = np.zeros_like(q_i)
        one = np.ones_like(q_i)
        return np.moveaxis(
            np.array(
                [
                    [self.L_1 * one, z, self.m_d * q_i, self.m_q * q_j],
                    [z, self.L_1 * one, -self.m_d * q_j, self.m_q * q_i],
                    [self.m_d * q_i, -self.m_d * q_j, self.L_d * one, z],
                    [self.m_q * q_j, self.m_q * q_i, z, self.L_q * one],
                ]
            ),
            (0, 1),
            (-2, -1),
        )

    def _dL_i(self, i_s: np.ndarray) -> np.ndarray:
        """Product `(dL_s/dq_i)*i_s`, shape (..., 4)."""
        m_d, m_q = self.m_d, self.m_q
        return np.stack(
            [
                m_d * i_s[..., 2],
                m_q * i_s[..., 3],
                m_d * i_s[..., 0],
                m_q * i_s[..., 1],
            ],
            axis=-1,
        )

    def _dL_j(self, i_s: np.ndarray) -> np.ndarray:
        """Product `(dL_s/dq_j)*i_s`, shape (..., 4)."""
        m_d, m_q = self.m_d, self.m_q
        return np.stack(
            [
                m_q * i_s[..., 3],
                -m_d * i_s[..., 2],
                -m_d * i_s[..., 1],
                m_q * i_s[..., 0],
            ],
            axis=-1,
        )

    def i_s(
        self, psi_s1: Any, psi_s2: Any, q_M: Any, exp_j_theta_M: Any = None
    ) -> tuple[Any, Any, Any]:
        """Currents (A) and radial force (N) from flux linkages (Vs)."""
        psi_s = c2v(psi_s1, psi_s2)
        i_s = np.linalg.solve(self.ind_mat(q_M), psi_s[..., np.newaxis])[..., 0]
        i_s1, i_s2 = v2c(i_s)
        return i_s1, i_s2, self._force(i_s)

    def _force(self, i_s: np.ndarray) -> Any:
        f_i = 1.5 * np.sum(i_s * self._dL_i(i_s), axis=-1)
        f_j = 1.5 * np.sum(i_s * self._dL_j(i_s), axis=-1)
        return f_i + 1j * f_j

    def psi_s(self, i_s1: Any, i_s2: Any, q_M: Any) -> tuple[Any, Any]:
        """Flux linkages (Vs) as functions of currents (A) and displacement (m)."""
        i_s = c2v(i_s1, i_s2)
        psi_s = (self.ind_mat(q_M) @ i_s[..., np.newaxis])[..., 0]
        psi_s1, psi_s2 = v2c(psi_s)
        return psi_s1, psi_s2

    def f_M(self, i_s1: Any, i_s2: Any, q_M: Any) -> Any:
        """Radial force (N) as a function of currents (A) and displacement (m)."""
        return self._force(c2v(i_s1, i_s2))

    def incr_ind_mat(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """Incremental inductance matrix (H), shape (..., 4, 4)."""
        return self.ind_mat(q_M)

    def di_dq(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """Current sensitivity `d i_s/d q_M` at constant flux, shape (..., 4, 2)."""
        i_s = c2v(i_s1, i_s2)
        dL_i_s = np.stack([self._dL_i(i_s), self._dL_j(i_s)], axis=-1)
        return -np.linalg.solve(self.ind_mat(q_M), dL_i_s)

    def df_dq(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """Force sensitivity `d f_M/d q_M` at constant flux, shape (..., 2, 2)."""
        # Since the inductances are linear in the displacement, the force depends on
        # the displacement only via the currents, df/dq = 3*(dL/dq*i_s)^T*(di_s/dq)
        i_s = c2v(i_s1, i_s2)
        dL_i_s = np.stack([self._dL_i(i_s), self._dL_j(i_s)], axis=-1)
        return 3 * np.swapaxes(dL_i_s, -2, -1) @ self.di_dq(i_s1, i_s2, q_M)


class FluxMapFcn(Protocol):
    """
    Flux map with its Jacobian, e.g., :class:`~motulator_bearingless.gradnet.FluxMap`.

    The map gives the flux linkages (Vs) of the two planes and the radial force (N) as
    functions of the currents (A) of the two planes and the displacement (m).

    """

    def __call__(self, i_s1: Any, i_s2: Any, q_M: Any) -> tuple[Any, Any, Any]:
        """Flux linkages (Vs) and radial force (N) from currents (A)."""
        ...

    def jacobian(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """Jacobian matrix of the outputs with respect to the inputs, (..., 6, 6)."""
        ...


@dataclass
class SaturatedBearinglessSynchronousMachinePars(BearinglessMagneticModel):
    """
    Parameters of a saturated bearingless synchronous machine.

    The class corresponds to
    :class:`motulator.drive.model.SaturatedSynchronousMachinePars`. The saturation model
    is specified as a current map. Optionally, to be used only in control systems, a
    flux map can be provided.

    Parameters
    ----------
    R_s : float
        Stator resistance (Ω).
    i_s_fcn : Callable[[Any, Any, Any], tuple[Any, Any, Any]], optional
        Currents (A) and radial force (N) as functions of the flux linkages (Vs) of the
        two planes and the displacement (m), e.g.,
        :class:`~motulator_bearingless.gradnet.CurrentMap`. Needed in the machine model.
    psi_s_fcn : FluxMapFcn, optional
        Flux linkages (Vs) and radial force (N) as functions of the currents (A) of the
        two planes and the displacement (m), with the Jacobian, e.g.,
        :class:`~motulator_bearingless.gradnet.FluxMap`. Needed in the control systems.

    """

    R_s: float
    i_s_fcn: Callable[[Any, Any, Any], tuple[Any, Any, Any]] | None = None
    psi_s_fcn: FluxMapFcn | None = None

    def __post_init__(self) -> None:
        if self.i_s_fcn is None and self.psi_s_fcn is None:
            raise ValueError("Either i_s_fcn or psi_s_fcn must be provided")

    @property
    def _flux_map(self) -> FluxMapFcn:
        if self.psi_s_fcn is None:
            raise ValueError("psi_s_fcn must be provided")
        return self.psi_s_fcn

    def i_s(
        self, psi_s1: Any, psi_s2: Any, q_M: Any, exp_j_theta_M: Any = None
    ) -> tuple[Any, Any, Any]:
        """Currents (A) and radial force (N) from flux linkages (Vs)."""
        if self.i_s_fcn is None:
            raise ValueError("i_s_fcn must be provided")
        return self.i_s_fcn(psi_s1, psi_s2, q_M)

    def psi_s(self, i_s1: Any, i_s2: Any, q_M: Any) -> tuple[Any, Any]:
        """Flux linkages (Vs) as functions of currents (A) and displacement (m)."""
        psi_s1, psi_s2, _ = self._flux_map(i_s1, i_s2, q_M)
        return psi_s1, psi_s2

    def f_M(self, i_s1: Any, i_s2: Any, q_M: Any) -> Any:
        """Radial force (N) as a function of currents (A) and displacement (m)."""
        return self._flux_map(i_s1, i_s2, q_M)[2]

    def incr_ind_mat(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """Incremental inductance matrix (H), shape (..., 4, 4)."""
        return self._flux_map.jacobian(i_s1, i_s2, q_M)[..., :4, :4]

    def di_dq(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """Current sensitivity `d i_s/d q_M` at constant flux, shape (..., 4, 2)."""
        jac = self._flux_map.jacobian(i_s1, i_s2, q_M)
        return -np.linalg.solve(jac[..., :4, :4], jac[..., :4, 4:])

    def df_dq(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """Force sensitivity `d f_M/d q_M` at constant flux, shape (..., 2, 2)."""
        jac = self._flux_map.jacobian(i_s1, i_s2, q_M)
        return jac[..., 4:, 4:] + jac[..., 4:, :4] @ self.di_dq(i_s1, i_s2, q_M)

    def linearized(self, i_s2: complex) -> BearinglessSynchronousMachinePars:
        """
        Constant parameters of the linear model at an operating point.

        The inductances of the four-pole plane are the secant inductances at the given
        current and those of the two-pole plane are the incremental inductances at a
        centric rotor without the two-pole current. The force constants are fitted to
        the sensitivity of the radial force to the two-pole current.

        Parameters
        ----------
        i_s2 : complex
            Current (A) of the four-pole plane in rotor coordinates. Its both components
            must be nonzero.

        Returns
        -------
        BearinglessSynchronousMachinePars
            Parameters of the linear model.

        """
        _, psi_s2 = self.psi_s(0j, i_s2, 0j)
        jac = self._flux_map.jacobian(0j, i_s2, 0j)
        df_di = jac[4:, :2]
        i_d, i_q = i_s2.real, i_s2.imag
        return BearinglessSynchronousMachinePars(
            R_s=self.R_s,
            L_1=0.5 * float(jac[0, 0] + jac[1, 1]),
            L_d=float(psi_s2.real / i_d),
            L_q=float(psi_s2.imag / i_q),
            m_d=float((df_di[0, 0] - df_di[1, 1]) / (6 * i_d)),
            m_q=float((df_di[0, 1] + df_di[1, 0]) / (6 * i_q)),
        )


@dataclass
class SpatialSaturatedBearinglessSynchronousMachinePars(BaseBearinglessMachinePars):
    """
    Parameters of a saturated bearingless synchronous machine with spatial harmonics.

    The class corresponds to
    :class:`motulator.drive.model.SpatialSaturatedSynchronousMachinePars`. This version
    is intended for the machine model, while the control methods do not support spatial
    harmonics.

    Parameters
    ----------
    R_s : float
        Stator resistance (Ω).
    magnetic_map_fcn : Callable[[Any, Any, Any, Any], tuple[Any, Any, Any, Any]]
        Currents (A), radial force (N), and torque (Nm) as functions of the flux
        linkages (Vs) of the two planes, the displacement (m), and the complex
        exponential of the mechanical rotor angle, e.g.,
        :class:`~motulator_bearingless.gradnet.CurrentMapWithHarmonics`.

    """

    R_s: float
    magnetic_map_fcn: Callable[[Any, Any, Any, Any], tuple[Any, Any, Any, Any]]

    def magnetic_map(
        self, psi_s1: Any, psi_s2: Any, q_M: Any, exp_j_theta_M: Any = None
    ) -> tuple[Any, Any, Any, Any]:
        """Currents (A), radial force (N), and torque (Nm) from flux linkages (Vs)."""
        return self.magnetic_map_fcn(psi_s1, psi_s2, q_M, exp_j_theta_M)

    def i_s(
        self, psi_s1: Any, psi_s2: Any, q_M: Any, exp_j_theta_M: Any = None
    ) -> tuple[Any, Any, Any]:
        """Currents (A) and radial force (N) from flux linkages (Vs)."""
        return self.magnetic_map_fcn(psi_s1, psi_s2, q_M, exp_j_theta_M)[:3]


class _FourPoleSlice:
    """Flux map of the four-pole plane at zero two-pole current and a centric rotor."""

    def __init__(self, par: BearinglessMagneticModel) -> None:
        self.par = par

    def __call__(self, i_s_dq: Any) -> Any:
        """Flux linkage (Vs) as a function of the current (A) of a set."""
        zero = np.zeros_like(i_s_dq)
        return self.par.psi_s(zero, i_s_dq, zero)[1]

    def jacobian(self, i_s_dq: Any) -> np.ndarray:
        """Incremental inductance matrix (H)."""
        zero = np.zeros_like(i_s_dq)
        return self.par.incr_ind_mat(zero, i_s_dq, zero)[..., 2:4, 2:4]


def fourpole_pars(
    par: BearinglessMagneticModel,
) -> SynchronousMachinePars | SaturatedSynchronousMachinePars:
    """
    Parameters of the four-pole plane as an equivalent three-phase machine.

    For a centric rotor, the four-pole plane is a three-phase synchronous reluctance
    machine with two pole pairs, whose two three-phase sets are connected in parallel:
    the voltage and the flux linkage of the equivalent machine are those of the plane,
    while its current is twice the current of the plane, i.e., the sum of the set
    currents. Hence, the resistance and the inductances of the plane are halved, cf.
    the common mode of multi-three-phase machines in `motulator_multiphase`. The
    equivalent machine produces the total torque `6*Im(i_s2*conj(psi_s2))`. The
    coupling of the planes via the rotor displacement is omitted.

    Parameters
    ----------
    par : BearinglessMagneticModel
        Magnetic model. A saturated model is sliced at zero two-pole current and at a
        centric rotor.

    Returns
    -------
    SynchronousMachinePars | SaturatedSynchronousMachinePars
        Parameters of the equivalent three-phase machine.

    Examples
    --------
    >>> from motulator_bearingless.utils import (
    ...     BearinglessSynchronousMachinePars,
    ...     fourpole_pars,
    ... )
    >>> par = BearinglessSynchronousMachinePars(
    ...     R_s=0.2, L_1=0.03, L_d=0.03, L_q=0.01, m_d=10.0, m_q=0.2
    ... )
    >>> fourpole_pars(par)
    SynchronousMachinePars(n_p=2, R_s=0.1, L_d=0.015, L_q=0.005, psi_f=0.0, G_c=0.0)

    """
    par_set: SynchronousMachinePars | SaturatedSynchronousMachinePars
    if isinstance(par, BearinglessSynchronousMachinePars):
        par_set = SynchronousMachinePars(
            n_p=2, R_s=par.R_s, L_d=par.L_d, L_q=par.L_q, psi_f=0.0
        )
    else:
        # The currents are computed from the flux map by motulator, if needed
        par_set = SaturatedSynchronousMachinePars(
            n_p=2, R_s=par.R_s, psi_s_dq_fcn=_FourPoleSlice(par)
        )
    return to_parallel(par_set, 2)


# %%
# Complex factor of the vector-space decomposition
ALPHA = exp(1j * pi / 3)


def vsd(x_A_ab: Any, x_B_ab: Any) -> tuple[Any, Any]:
    """
    Vector-space decomposition of the two three-phase sets into the two planes.

    The symmetrical six-phase winding consists of phases 1...6 at 60° intervals. The
    first three-phase set consists of the phases 1, 3, 5 and the second set of the
    phases 2, 4, 6. Each set is described by its standard peak-valued space vector
    `abc2complex([x_1, x_3, x_5])` and `abc2complex([x_2, x_4, x_6])`, where the vector
    of the second set is expressed in its own coordinates (its a-phase axis is at 60°).
    The planes `h = 1` (two-pole) and `h = 2` (four-pole) result from the
    vector-space decomposition with peak-value scaling,
    `x_sh = (1/3)*sum(x_k*alpha**(h*(k - 1)))`, where `alpha = exp(1j*pi/3)`
    [#Ahm2023]_. Zero-sequence components are omitted.

    Parameters
    ----------
    x_A_ab : complex | ndarray
        Space vector of the first three-phase set (phases 1, 3, 5).
    x_B_ab : complex | ndarray
        Space vector of the second three-phase set (phases 2, 4, 6).

    Returns
    -------
    x_s1_ab, x_s2_ab : complex | ndarray
        Space vectors in the two-pole and four-pole planes, in stator coordinates.

    Notes
    -----
    The four-pole plane sees the phases in reversed sequence, which shows up as the
    complex conjugate in the transformation.

    Examples
    --------
    >>> from motulator_bearingless.utils import ALPHA, vsd
    >>> # Equal currents in the opposite phases (i_1 = i_4, i_3 = i_6, i_5 = i_2)
    >>> # produce only the four-pole component
    >>> i_A_ab = 1 + 1j
    >>> i_s1_ab, i_s2_ab = vsd(i_A_ab, ALPHA**2 * i_A_ab)
    >>> round(abs(i_s1_ab), 12)
    0.0
    >>> complex(round(i_s2_ab.real, 12), round(i_s2_ab.imag, 12))
    (1-1j)

    """
    x_s1_ab = 0.5 * (x_A_ab + ALPHA * x_B_ab)
    x_s2_ab = 0.5 * (x_A_ab - ALPHA * x_B_ab).conjugate()  # alpha**(-2) = -alpha
    return x_s1_ab, x_s2_ab


def inv_vsd(x_s1_ab: Any, x_s2_ab: Any) -> tuple[Any, Any]:
    """
    Inverse vector-space decomposition, see :func:`vsd`.

    Parameters
    ----------
    x_s1_ab : complex | ndarray
        Space vector in the two-pole plane, in stator coordinates.
    x_s2_ab : complex | ndarray
        Space vector in the four-pole plane, in stator coordinates.

    Returns
    -------
    x_A_ab, x_B_ab : complex | ndarray
        Space vectors of the two three-phase sets.

    Examples
    --------
    >>> from motulator_bearingless.utils import inv_vsd, vsd
    >>> x_A_ab, x_B_ab = inv_vsd(0.1j, 1 - 0.5j)
    >>> [complex(round(x.real, 12), round(x.imag, 12)) for x in vsd(x_A_ab, x_B_ab)]
    [0.1j, (1-0.5j)]

    """
    x_A_ab = x_s1_ab + x_s2_ab.conjugate()
    x_B_ab = (x_s1_ab - x_s2_ab.conjugate()) / ALPHA
    return x_A_ab, x_B_ab


def sixphase_currents(i_c_ab: Any) -> np.ndarray:
    """
    Phase currents from the space vectors of the two sets.

    Parameters
    ----------
    i_c_ab : ndarray, shape (2,) or (2, N)
        Space vectors of the first (phases 1, 3, 5) and second (phases 2, 4, 6) sets.

    Returns
    -------
    ndarray, shape (6,) or (6, N)
        Phase currents `[i_1, ..., i_6]`.

    """
    # The phases a, b, c of the sets are [[i_1, i_2], [i_3, i_4], [i_5, i_6]]
    i_abc = complex2abc(np.asarray(i_c_ab))
    return np.reshape(i_abc, (6, *i_abc.shape[2:]))


# %%
def _plot_speeds(ax, ctrl, mdl, base: BaseValues) -> None:
    if hasattr(ctrl.ref, "w_M") and np.any(ctrl.ref.w_M):
        ax.plot(
            ctrl.t,
            ctrl.ref.w_M / base.w_M,
            "--",
            ds="steps-post",
            label=r"$\omega_\mathrm{M}^\mathrm{ref}$",
        )
    ax.plot(mdl.t, mdl.machine.w_M / base.w_M, label=r"$\omega_\mathrm{M}$")
    ax.plot(
        ctrl.t,
        ctrl.fbk.w_M / base.w_M,
        ds="steps-post",
        label=r"$\hat{\omega}_\mathrm{M}$",
    )
    ax.legend(loc="lower right")


def _plot_torques(ax, ctrl, mdl, base: BaseValues) -> None:
    if hasattr(ctrl.ref, "tau_M") and np.any(ctrl.ref.tau_M):
        ax.plot(
            ctrl.t,
            ctrl.ref.tau_M / base.tau,
            "--",
            ds="steps-post",
            label=r"$\tau_\mathrm{M}^\mathrm{ref}$",
        )
    ax.plot(mdl.t, mdl.machine.tau_M / base.tau, label=r"$\tau_\mathrm{M}$")
    ax.plot(
        ctrl.t,
        ctrl.fbk.tau_M / base.tau,
        ds="steps-post",
        label=r"$\hat{\tau}_\mathrm{M}$",
    )
    if hasattr(mdl.mechanics, "tau_L_tot"):
        ax.plot(
            mdl.t,
            mdl.mechanics.tau_L_tot / base.tau,
            ":",
            label=r"$\tau_\mathrm{L}^\mathrm{tot}$",
        )
    ax.legend(loc="best", ncols=2)


def _plot_flux_linkages(ax, ctrl, mdl, base: BaseValues) -> None:
    ax.plot(
        ctrl.t,
        ctrl.ref.psi_s / base.psi,
        "--",
        ds="steps-post",
        label=r"$\psi_\mathrm{s2}^\mathrm{ref}$",
    )
    ax.plot(
        mdl.t, np.abs(mdl.machine.psi_s2_dq) / base.psi, label=r"$\psi_\mathrm{s2}$"
    )
    ax.plot(
        ctrl.t,
        np.abs(ctrl.fbk.psi_s2) / base.psi,
        ds="steps-post",
        label=r"$\hat{\psi}_\mathrm{s2}$",
    )
    ax.plot(
        mdl.t,
        np.abs(mdl.machine.psi_s1_dq) / base.psi,
        "-.",
        label=r"$\psi_\mathrm{s1}$",
    )
    ax.legend(loc="best", ncols=2)
    _, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax)


def _plot_forces(ax, ctrl, mdl, base: BaseValues) -> None:
    ax.plot(
        ctrl.t,
        ctrl.ref.f_M_ab.real,
        "--",
        ds="steps-post",
        label=r"$f_\mathrm{M,i}^\mathrm{ref}$",
    )
    ax.plot(
        ctrl.t,
        ctrl.ref.f_M_ab.imag,
        "--",
        ds="steps-post",
        label=r"$f_\mathrm{M,j}^\mathrm{ref}$",
    )
    ax.plot(mdl.t, mdl.machine.f_M_ab.real, label=r"$f_\mathrm{M,i}$")
    ax.plot(mdl.t, mdl.machine.f_M_ab.imag, label=r"$f_\mathrm{M,j}$")
    ax.plot(
        ctrl.t,
        ctrl.fbk.f_M_ab.real,
        ":",
        ds="steps-post",
        label=r"$\hat{f}_\mathrm{M,i}$",
    )
    ax.plot(
        ctrl.t,
        ctrl.fbk.f_M_ab.imag,
        ":",
        ds="steps-post",
        label=r"$\hat{f}_\mathrm{M,j}$",
    )
    ax.legend(loc="best", ncols=3)


def _plot_currents(ax, ctrl, base: BaseValues) -> None:
    ax.plot(
        ctrl.t, ctrl.fbk.i_s2.real / base.i, ds="steps-post", label=r"$i_\mathrm{s2d}$"
    )
    ax.plot(
        ctrl.t, ctrl.fbk.i_s2.imag / base.i, ds="steps-post", label=r"$i_\mathrm{s2q}$"
    )
    ax.plot(
        ctrl.t,
        10 * ctrl.fbk.i_s1.real / base.i,
        ds="steps-post",
        label=r"$10\,i_\mathrm{s1d}$",
    )
    ax.plot(
        ctrl.t,
        10 * ctrl.fbk.i_s1.imag / base.i,
        ds="steps-post",
        label=r"$10\,i_\mathrm{s1q}$",
    )
    ax.legend(loc="best", ncols=2)


def _plot_displacements(ax, ctrl, mdl) -> None:
    if np.any(ctrl.ref.q_M_ab):
        ax.plot(
            ctrl.t,
            1e6 * ctrl.ref.q_M_ab.real,
            "--",
            ds="steps-post",
            label=r"$q_\mathrm{M,i}^\mathrm{ref}$",
        )
        ax.plot(
            ctrl.t,
            1e6 * ctrl.ref.q_M_ab.imag,
            "--",
            ds="steps-post",
            label=r"$q_\mathrm{M,j}^\mathrm{ref}$",
        )
    ax.plot(mdl.t, 1e6 * mdl.machine.q_M_ab.real, label=r"$q_\mathrm{M,i}$")
    ax.plot(mdl.t, 1e6 * mdl.machine.q_M_ab.imag, label=r"$q_\mathrm{M,j}$")
    ax.legend(loc="best", ncols=2)


def plot(
    res: SimulationResults,
    base: BaseValues | None = None,
    subplots: list[str] | None = None,
    t_lims: tuple[float, float] | None = None,
    t_ticks: Any = None,
    y_lims: Sequence[tuple[float, float] | None] | None = None,
    y_ticks: Sequence[Any] | None = None,
    latex: bool = False,
    save_path: str | Path | None = None,
    **savefig_kwargs,
) -> None:
    """
    Plot the simulation results of a bearingless machine drive.

    Parameters
    ----------
    res : SimulationResults
        Simulation results.
    base : BaseValues, optional
        Base values for scaling the electrical waveforms. If not given, the waveforms
        are plotted in SI units. Forces and displacements are always in SI units.
    subplots : list[str], optional
        Subplots to include. Valid names: "speed", "torque", "flux", "force",
        "current", "displacement". Defaults to all but "displacement".
    t_lims : tuple[float, float], optional
        Time axis limits.
    t_ticks : ArrayLike, optional
        Time axis tick locations.
    y_lims : Sequence[tuple[float, float] | None], optional
        y-axis limits for each subplot.
    y_ticks : Sequence[ArrayLike | None], optional
        y-axis tick locations for each subplot.
    latex : bool, optional
        Use LaTeX fonts for the labels, defaults to False.
    save_path : str | Path, optional
        Path to save the figure. If None, the figure is not saved.
    **savefig_kwargs
        Additional keyword arguments passed to `plt.savefig()`.

    """
    width, height = setup_time_series_plot(latex)
    if base is None:
        pu_vals = False
        base = BaseValues.unity()
    else:
        pu_vals = True
    if subplots is None:
        subplots = ["speed", "torque", "flux", "force", "current"]
    if t_lims is None:
        t_lims = (0, res.mdl.t[-1])

    mdl, ctrl = res.mdl, res.ctrl
    height = height * len(subplots) / 5
    fig, axes = plt.subplots(len(subplots), 1, figsize=(width, height), sharex=True)
    axes = np.atleast_1d(axes)

    labels = {
        "speed": "Speed (p.u.)" if pu_vals else "Speed (rad/s)",
        "torque": "Torque (p.u.)" if pu_vals else "Torque (Nm)",
        "flux": "Flux linkage (p.u.)" if pu_vals else "Flux linkage (Vs)",
        "force": "Force (N)",
        "current": "Current (p.u.)" if pu_vals else "Current (A)",
        "displacement": "Displacement (µm)",
    }
    plotters = {
        "speed": lambda ax: _plot_speeds(ax, ctrl, mdl, base),
        "torque": lambda ax: _plot_torques(ax, ctrl, mdl, base),
        "flux": lambda ax: _plot_flux_linkages(ax, ctrl, mdl, base),
        "force": lambda ax: _plot_forces(ax, ctrl, mdl, base),
        "current": lambda ax: _plot_currents(ax, ctrl, base),
        "displacement": lambda ax: _plot_displacements(ax, ctrl, mdl),
    }
    for ax, name in zip(axes, subplots, strict=True):
        if name not in plotters:
            raise ValueError(f"Unknown subplot name: {name}")
        plotters[name](ax)
        ax.set_ylabel(labels[name])

    configure_axes(list(axes), t_lims, t_ticks, y_lims, y_ticks)
    fig.align_ylabels()
    axes[-1].set_xlabel("Time (s)")
    save_and_show(save_path, **savefig_kwargs)


# %%
def plot_phase_currents(
    t: np.ndarray,
    i_abc: np.ndarray,
    t_lims: tuple[float, float] | None = None,
    y_lims: tuple[float, float] | None = None,
    latex: bool = False,
    save_path: str | Path | None = None,
    **savefig_kwargs,
) -> None:
    """
    Plot the six phase currents.

    Parameters
    ----------
    t : ndarray
        Time (s).
    i_abc : ndarray, shape (6, N)
        Phase currents (A), see :func:`sixphase_currents`.
    t_lims : tuple[float, float], optional
        Time axis limits.
    y_lims : tuple[float, float], optional
        Current axis limits.
    latex : bool, optional
        Use LaTeX fonts for the labels, defaults to False.
    save_path : str | Path, optional
        Path to save the figure. If None, the figure is not saved.
    **savefig_kwargs
        Additional keyword arguments passed to `plt.savefig()`.

    """
    set_style(latex)
    fig, ax = plt.subplots()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k in range(6):
        # Phases a, b, c of the sets in the same colors, the second set dashed
        style = {"c": colors[k // 2], "ls": "--" if k % 2 else "-"}
        ax.plot(t, i_abc[k], **style, label=rf"$i_{k + 1}$")
    ax.legend(loc="upper right", ncols=3)
    ax.set_ylabel("Current (A)")
    ax.set_xlabel("Time (s)")
    configure_axes([ax], t_lims, None, [y_lims], None)
    save_and_show(save_path, **savefig_kwargs)
