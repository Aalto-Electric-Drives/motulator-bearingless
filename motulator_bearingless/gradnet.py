"""
Gradient-network magnetic maps of six-phase bearingless synchronous machines.

This module corresponds to :mod:`motulator.drive.gradnet` and it uses the same softmax
gradient networks [#Li2026]_, [#Cha2025]_, which are the gradients of a convex scalar
state function, since `A_0` is positive semidefinite::

    z = A*x + b
    g(x) = A_0*x + b_0 + A.T*softmax(beta*z)

As in *motulator*, the machine model uses the energy-based current map and the
controller the co-energy-based flux map. The inputs are extended with the rotor
displacement. The maps follow from the energy `W(psi_s, q_M, theta_M)` and the co-energy
`W_c(i_s, q_M)`, cf. [#Das2026]_::

    i_s = (1/3)*dW/dpsi_s       f_M = -dW/dq_M
    psi_s = (1/3)*dW_c/di_s     f_M = dW_c/dq_M

The maps differ from those of *motulator* in two ways. First, the output of the network
is averaged over the symmetry group of the bearingless machine, see
:func:`symmetry_group`, which contains the mirror symmetry about the d-axis. Second,
checks of the finite-element model at selected operating points indicate that the
unbalanced magnetic pull is destabilizing also at constant flux linkages, so that the
energy is concave in the displacement. Therefore, the network of the current map
approximates the shifted energy `W + k_q*abs(q_M)**2/2` by a convex function. The
shifted energy is convex where `k_q` exceeds the stiffness of the pull at constant
currents.

References
----------
.. [#Li2026] Li, Foissner, Martin, Piippo, Hinkkanen, "Gradient networks for universal
   magnetic modeling of synchronous machines," 2026, https://arxiv.org/abs/2602.14947

.. [#Cha2025] Chaudhari, Pranav, Moura, "Gradient networks," IEEE Trans. Signal
   Process., 2025, https://doi.org/10.1109/TSP.2024.3496692

.. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
   bearingless synchronous machines," Proc. ICEM, 2026,
   https://doi.org/10.1109/ICEM68283.2026.11757980

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from motulator.drive import gradnet as motulator_gradnet

from motulator_bearingless.utils import c2v, torque, v2c

CHUNK = 4096  # Number of the points of a time series evaluated at once

__all__ = [
    "CurrentMap",
    "CurrentMapWithHarmonics",
    "FluxMap",
    "GradNetFunction",
    "load_gradnet",
    "symmetry_group",
]


# %%
def _rot2(a: float) -> np.ndarray:
    """Rotation matrix of the angle `a`."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _block(*mats: np.ndarray) -> np.ndarray:
    """Block-diagonal matrix of 2x2 matrices."""
    n = 2 * len(mats)
    out = np.zeros((n, n))
    for k, m in enumerate(mats):
        out[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = m
    return out


def symmetry_group(k: tuple[int, ...] = ()) -> np.ndarray:
    """
    Symmetry group of the machine acting on the inputs of the network.

    The rotor-coordinate inputs are `(x_s1, x_s2, q_M, exp(1j*k*theta_M))` as pairs of
    real numbers, where `x_s` is the flux linkage or the current. The state function of
    a four-pole rotor in a symmetrical six-phase stator without zero-sequence currents
    is periodic in the rotor angle with the period of 60 deg and invariant under the
    following generators:

    - reversal of the currents, `(-i_s1, -i_s2, q_M, theta_M)`,
    - mirroring about the d-axis, `(conj(i_s1), conj(i_s2), conj(q_M), -theta_M)`, and
    - turning the rotor by 90 deg and the stator by 60 deg in the opposite direction,
      `(-1j*i_s1, -i_s2, -1j*q_M, theta_M + 30 deg)`.

    Parameters
    ----------
    k : tuple[int, ...], optional
        Orders of the periodic features of the rotor angle, multiples of 6.

    Returns
    -------
    ndarray, shape (16, n, n)
        Orthogonal matrices acting on the input vector of the dimension
        `n = 6 + 2*len(k)`.

    Examples
    --------
    >>> import numpy as np
    >>> from motulator_bearingless.gradnet import symmetry_group
    >>> G = symmetry_group((6, 36))
    >>> G.shape
    (16, 10, 10)
    >>> bool(np.allclose(G @ G.transpose(0, 2, 1), np.eye(10)))
    True

    """
    if any(order % 6 for order in k):
        raise ValueError("The orders of the periodic features must be multiples of 6")
    eye, conj = np.eye(2), np.diag([1.0, -1.0])
    inv = _block(-eye, -eye, eye, *[eye for _ in k])
    mir = _block(conj, conj, conj, *[conj for _ in k])
    rot = _block(
        _rot2(-np.pi / 2),
        -eye,
        _rot2(-np.pi / 2),
        *[_rot2(order * np.pi / 6) for order in k],
    )
    return np.round(
        np.array(
            [
                np.linalg.matrix_power(inv, a)
                @ np.linalg.matrix_power(mir, b)
                @ np.linalg.matrix_power(rot, c)
                for a in range(2)
                for b in range(2)
                for c in range(4)
            ]
        ),
        12,
    )


# %%
@dataclass
class GradNetFunction(motulator_gradnet.GradNetFunction):
    """
    Trained GradNet of a bearingless machine as a numpy function.

    This function extends :class:`motulator.drive.gradnet.GradNetFunction` with the
    quantities needed by the maps of this module, so that they are saved together with
    the network. It also computes the Jacobian of a softmax module as
    `beta*(sum(s_j*a_j*a_j.T) - h*h.T)`, where `s` is the activation and `h = A.T*s`,
    without matrix products of the size of the number of the hidden units.

    Parameters
    ----------
    map_type : str, optional
        Type of the map, `"current_map"` or `"flux_map"`, defaults to the latter.
    k : tuple[int, ...], optional
        Orders of the periodic features of the rotor angle, defaults to no features.
    q_base : float, optional
        Base value of the displacement (m), defaults to 1.
    k_q : float, optional
        Per-unit coefficient of the quadratic term of the energy, zero in the flux maps.

    """

    map_type: str = "flux_map"
    k: tuple[int, ...] = ()
    q_base: float = 1.0
    k_q: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.k = tuple(int(order) for order in np.atleast_1d(self.k))
        n = self.bias.size
        self._A_0 = np.diag(np.pad(np.exp(self.mu_log), (0, n - self.mu_log.size)))
        self._A_outer = (self.W[:, :, :, None] * self.W[:, :, None, :]).reshape(
            *self.W.shape[:2], n**2
        )

    def __call__(
        self, x: np.ndarray, jacobian: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate the function, see the base class."""
        if not jacobian or self.activation != "Softmax":
            return super().__call__(x, jacobian)
        y = x @ self._A_0 + self.bias
        jac = np.broadcast_to(self._A_0, (*x.shape, x.shape[-1]))
        for A, b, beta_log, A_outer in zip(
            self.W, self.b, self.beta_log, self._A_outer, strict=True
        ):
            beta = np.exp(beta_log)
            z = x @ A.T + b
            s = np.exp(beta * (z - z.max(axis=-1, keepdims=True)))
            s /= s.sum(axis=-1, keepdims=True)
            h = s @ A
            y = y + h
            s_aa = (s @ A_outer).reshape(jac.shape)
            jac = jac + beta * (s_aa - h[..., :, None] * h[..., None, :])
        return y, jac


def load_gradnet(model_path: Path | str) -> GradNetFunction:
    """
    Load a trained GradNet from an npz file.

    Parameters
    ----------
    model_path : Path | str
        Path to the saved model file, see
        :meth:`motulator.drive.gradnet.GradNetFunction.save`.

    Returns
    -------
    GradNetFunction
        Trained network as a numpy function.

    """
    with np.load(model_path) as data:
        par = {
            name: value.item() if value.ndim == 0 else value
            for name, value in data.items()
        }
    return GradNetFunction(**par)


# %%
class _SymmetrizedMap:
    """Per-unit map averaged over the symmetry group of the machine."""

    _map_type = "current_map"
    _harmonics = False

    def __init__(self, model: GradNetFunction) -> None:
        if model.map_type != self._map_type or bool(model.k) != self._harmonics:
            raise ValueError(
                f"{type(self).__name__} does not fit the network of the type "
                f"{model.map_type} with the periodic features k = {model.k}"
            )
        self.fcn = model
        self.group = symmetry_group(model.k)
        self.i_base, self.psi_base = model.i_base, model.psi_base
        self.q_base = model.q_base
        self.f_base = 3 * self.i_base * self.psi_base / self.q_base

    def _evaluate(self, *x: Any, jacobian: bool = False) -> tuple[Any, Any]:
        """Output and Jacobian at the per-unit complex inputs."""
        x_v = c2v(*np.broadcast_arrays(*x))
        # Long time series are evaluated in parts, since the group multiplies their size
        if x_v.ndim == 2 and x_v.shape[0] > CHUNK:
            parts = [
                self._symmetrized(x_v[m : m + CHUNK], jacobian)
                for m in range(0, x_v.shape[0], CHUNK)
            ]
            y, jac = zip(*parts, strict=True)
            return np.concatenate(y), np.concatenate(jac) if jacobian else None
        return self._symmetrized(x_v, jacobian)

    def _symmetrized(self, x: np.ndarray, jacobian: bool) -> tuple[Any, Any]:
        """Output and Jacobian averaged over the group, shape (..., n)."""
        G, G_T = self.group, self.group.transpose(0, 2, 1)
        y, jac = self.fcn((G @ x[..., None, :, None])[..., 0], jacobian)
        y_sym = np.mean((G_T @ y[..., None])[..., 0], axis=-2)
        return y_sym, None if jac is None else np.mean(G_T @ jac @ G, axis=-3)


class CurrentMap(_SymmetrizedMap):
    """
    Energy-based current map for the machine model, without spatial harmonics.

    The map corresponds to :class:`motulator.drive.gradnet.CurrentMap`.

    Parameters
    ----------
    model : GradNetFunction
        Trained GradNet for the current map, see :func:`load_gradnet`.

    """

    def __call__(self, psi_s1: Any, psi_s2: Any, q_M: Any) -> tuple[Any, Any, Any]:
        """Currents (A) and radial force (N) from flux linkages (Vs)."""
        q = np.asarray(q_M) / self.q_base
        y, _ = self._evaluate(
            np.asarray(psi_s1) / self.psi_base, np.asarray(psi_s2) / self.psi_base, q
        )
        y_1, y_2, y_q = v2c(y)
        return (
            self.i_base * y_1,
            self.i_base * y_2,
            self.f_base * (self.fcn.k_q * q - y_q),
        )


class CurrentMapWithHarmonics(_SymmetrizedMap):
    """
    Energy-based current map for the machine model, with spatial harmonics.

    The map corresponds to :class:`motulator.drive.gradnet.CurrentMapWithHarmonics`. The
    rotor angle enters as the periodic features `exp(1j*k*theta_M)`, whose orders `k`
    are saved with the network.

    Parameters
    ----------
    model : GradNetFunction
        Trained GradNet for the current map with harmonics, see :func:`load_gradnet`.

    """

    _harmonics = True

    def __call__(
        self, psi_s1: Any, psi_s2: Any, q_M: Any, exp_j_theta_M: Any
    ) -> tuple[Any, Any, Any, Any]:
        """
        Currents (A), radial force (N), and torque (Nm) from flux linkages (Vs).

        Parameters
        ----------
        psi_s1, psi_s2 : complex | ndarray
            Stator flux linkages of the two-pole and four-pole planes.
        q_M : complex | ndarray
            Rotor displacement (m).
        exp_j_theta_M : complex | ndarray
            Complex exponential of the mechanical rotor angle.

        Returns
        -------
        i_s1, i_s2, f_M : complex | ndarray
            Stator currents of the two planes and the radial force.
        tau_M : float | ndarray
            Electromagnetic torque.

        """
        q = np.asarray(q_M) / self.q_base
        k = self.fcn.k
        exp_j_k = [np.asarray(exp_j_theta_M) ** order for order in k]
        y, _ = self._evaluate(
            np.asarray(psi_s1) / self.psi_base,
            np.asarray(psi_s2) / self.psi_base,
            q,
            *exp_j_k,
        )
        y_1, y_2, y_q, *dW = v2c(y)
        i_s1, i_s2 = self.i_base * y_1, self.i_base * y_2
        f_M = self.f_base * (self.fcn.k_q * q - y_q)
        dW_dtheta = sum(
            order * np.imag(np.conj(e) * d)
            for order, e, d in zip(k, exp_j_k, dW, strict=True)
        )
        e_base = self.f_base * self.q_base
        return (
            i_s1,
            i_s2,
            f_M,
            torque(i_s1, i_s2, psi_s1, psi_s2, q_M, f_M) - e_base * dW_dtheta,
        )


class FluxMap(_SymmetrizedMap):
    """
    Co-energy-based flux map for the controller, without spatial harmonics.

    The map corresponds to :class:`motulator.drive.gradnet.FluxMap`. The network is
    trained without the rotor angle on data at different rotor angles, so that it
    approximates the average over the angle.

    Parameters
    ----------
    model : GradNetFunction
        Trained GradNet for the flux map, see :func:`load_gradnet`.

    """

    _map_type = "flux_map"

    def __init__(self, model: GradNetFunction) -> None:
        super().__init__(model)
        in_base = [self.i_base] * 4 + [self.q_base] * 2
        out_base = [self.psi_base] * 4 + [self.f_base] * 2
        self._scale = np.outer(out_base, np.reciprocal(in_base))
        self._last: tuple[tuple, np.ndarray, np.ndarray] | None = None

    def _output(
        self, i_s1: Any, i_s2: Any, q_M: Any, jacobian: bool
    ) -> tuple[Any, Any]:
        """
        Per-unit output and its Jacobian.

        The results of scalar inputs are remembered, since a control system needs both
        of them, and several blocks of the Jacobian, at the same operating point.

        """
        x = (
            np.asarray(i_s1) / self.i_base,
            np.asarray(i_s2) / self.i_base,
            np.asarray(q_M) / self.q_base,
        )
        if np.ndim(i_s1) or np.ndim(i_s2) or np.ndim(q_M):
            return self._evaluate(*x, jacobian=jacobian)
        # Immutable copies, since a zero-dimensional array may be changed in place
        key = (complex(i_s1), complex(i_s2), complex(q_M))
        if self._last is None or self._last[0] != key:
            self._last = (key, *self._evaluate(*x, jacobian=True))
        return self._last[1:]

    def __call__(self, i_s1: Any, i_s2: Any, q_M: Any) -> tuple[Any, Any, Any]:
        """Flux linkages (Vs) and radial force (N) from currents (A)."""
        y, _ = self._output(i_s1, i_s2, q_M, False)
        y_1, y_2, y_f = v2c(y)
        return self.psi_base * y_1, self.psi_base * y_2, self.f_base * y_f

    def jacobian(self, i_s1: Any, i_s2: Any, q_M: Any) -> np.ndarray:
        """
        Jacobian matrix of the map.

        Returns
        -------
        ndarray, shape (..., 6, 6)
            Jacobian matrix of the outputs `(psi_s1, psi_s2, f_M)` with respect to the
            inputs `(i_s1, i_s2, q_M)` as pairs of real numbers in SI units.

        """
        return self._scale * self._output(i_s1, i_s2, q_M, True)[1]
