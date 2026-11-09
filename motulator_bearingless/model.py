"""
Continuous-time models for six-phase bearingless synchronous machine drives.

The symmetrical six-phase winding is described by means of the vector-space
decomposition: the two-pole plane (`h = 1`) and the four-pole plane (`h = 2`) are
represented by peak-valued complex space vectors. The machine model is expressed in
rotor coordinates, where the plane `h` rotates at the electrical angular speed
`h*w_M`. The magnetic model is a gradient of a magnetic energy function, which allows
representing magnetic saturation and rotor eccentricity in a consistent manner
[#Das2026]_. Uppercase `M` denotes mechanical quantities, as in motulator. The rotor
displacement `q_M` and the radial force `f_M` are complex numbers, whose real and
imaginary parts correspond to the i and j axes. The mechanical system models and the
simulation class of motulator and the converter model of motulator_multiphase are
re-exported for convenience.

References
----------
.. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
   bearingless synchronous machines," Proc. ICEM, 2026,
   https://doi.org/10.1109/ICEM68283.2026.11757980

"""

from collections.abc import Callable
from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np
from motulator.common.model import Model, Subsystem, SubsystemTimeSeries
from motulator.drive.model import (
    ZOH,
    CarrierComparison,
    DriveSensors,
    ExternalRotorSpeed,
    MechanicalSystem,
    Sensor,
    Simulation,
    TwoMassMechanicalSystem,
)
from motulator_multiphase.model import MultiThreePhaseConverter

from motulator_bearingless._helpers import empty_array, get_value, wrap
from motulator_bearingless.utils import (
    BaseBearinglessMachinePars,
    BearinglessSynchronousMachinePars,
    SaturatedBearinglessSynchronousMachinePars,
    SpatialSaturatedBearinglessSynchronousMachinePars,
    inv_vsd,
    vsd,
)

__all__ = [
    "ZOH",
    "BearinglessDrive",
    "BearinglessDriveSensors",
    "BearinglessSynchronousMachine",
    "BearinglessSynchronousMachinePars",
    "CarrierComparison",
    "ExternalRotorPosition",
    "ExternalRotorSpeed",
    "MechanicalSystem",
    "MultiThreePhaseConverter",
    "RadialMechanics",
    "SaturatedBearinglessSynchronousMachinePars",
    "Sensor",
    "Simulation",
    "SpatialSaturatedBearinglessSynchronousMachinePars",
    "TwoMassMechanicalSystem",
]


# %%
@dataclass
class MachineInputs:
    """Machine inputs."""

    u_c_ab: np.ndarray  # Voltages of the sets, shape (2,)
    w_M: float = 0.0
    q_M_ab: complex = 0j  # Rotor displacement (m) in stator coordinates


@dataclass
class MachineOutputs:
    """Output variables for interconnection."""

    i_c_ab: np.ndarray  # Currents of the sets, shape (2,)
    i_s1_ab: complex
    i_s2_ab: complex
    tau_M: float
    f_M_ab: complex  # Radial force (N) in stator coordinates


@dataclass
class MachineStates:
    """State variables."""

    psi_s1_dq: complex = 0j
    psi_s2_dq: complex = 0j
    theta_M: complex = 0j  # Mechanical rotor angle (rad), imaginary part always zero


class BearinglessSynchronousMachine(Subsystem):
    """
    Six-phase bearingless synchronous machine model.

    The model is internally represented in rotor coordinates. The interfaces are in
    stator coordinates: the voltages and currents as the space vectors of the two
    three-phase sets, the displacement and the force as complex numbers. The first set
    consists of the phases 1, 3, 5 and the second set of the phases 2, 4, 6. The
    magnetic model is selected via the parameters: linear magnetics with
    :class:`BearinglessSynchronousMachinePars`, or magnetic saturation with
    :class:`SaturatedBearinglessSynchronousMachinePars` or, including spatial harmonics,
    :class:`SpatialSaturatedBearinglessSynchronousMachinePars`.

    Notes
    -----
    A radial force that is constant in stator coordinates results from the interaction
    of the four-pole field (rotating with the rotor) and a two-pole field having the
    same electrical angular frequency `2*w_M` in stator coordinates. Hence, the
    two-pole quantities rotate at `w_M` in rotor coordinates, while the displacement of
    a rotor held at a fixed position rotates at `-w_M`.

    Parameters
    ----------
    par : BaseBearinglessMachinePars
        Machine parameters.

    """

    # The currents depend directly on the rotor displacement
    feedthrough = frozenset({"q_M_ab"})

    def __init__(self, par: BaseBearinglessMachinePars) -> None:
        self.par = par
        self.inp: MachineInputs = MachineInputs(u_c_ab=np.zeros(2, dtype=complex))
        self.state: MachineStates = MachineStates()
        self.out: MachineOutputs = MachineOutputs(*self.compute_outputs(self.state))

    def compute_outputs(
        self, state: Any, q_M_ab: Any = 0j
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Compute output variables."""
        exp_j_theta_M = np.exp(1j * state.theta_M.real)
        q_M_dq = np.conj(exp_j_theta_M) * q_M_ab
        i_s1_dq, i_s2_dq, f_M_dq, tau_M = self.par.magnetic_map(
            state.psi_s1_dq, state.psi_s2_dq, q_M_dq, exp_j_theta_M
        )
        i_s1_ab = exp_j_theta_M * i_s1_dq
        i_s2_ab = exp_j_theta_M**2 * i_s2_dq
        i_c_ab = np.array(inv_vsd(i_s1_ab, i_s2_ab))
        f_M_ab = exp_j_theta_M * f_M_dq
        return i_c_ab, i_s1_ab, i_s2_ab, tau_M, f_M_ab

    def set_states(self, state_list: list[complex], index: int) -> int:
        """Set states, keeping the rotor angle within [-pi, pi)."""
        # The angle is wrapped only in the state of the model, which the next
        # integration interval starts from, so the solver sees no discontinuity
        index = super().set_states(state_list, index)
        self.state.theta_M = wrap(self.state.theta_M.real)
        return index

    def set_outputs(self, t: float) -> None:
        """Set output variables."""
        out = self.out
        (out.i_c_ab, out.i_s1_ab, out.i_s2_ab, out.tau_M, out.f_M_ab) = (
            self.compute_outputs(self.state, self.inp.q_M_ab)
        )

    def rhs(self, t: float) -> list[complex]:
        """Compute state derivatives."""
        state, inp, out, par = self.state, self.inp, self.out, self.par
        exp_j_theta_M = np.exp(1j * state.theta_M.real)
        u_s1_ab, u_s2_ab = vsd(*inp.u_c_ab)
        u_s1_dq = np.conj(exp_j_theta_M) * u_s1_ab
        u_s2_dq = np.conj(exp_j_theta_M) ** 2 * u_s2_ab
        i_s1_dq = np.conj(exp_j_theta_M) * out.i_s1_ab
        i_s2_dq = np.conj(exp_j_theta_M) ** 2 * out.i_s2_ab
        w_M = inp.w_M
        d_psi_s1_dq = u_s1_dq - par.R_s * i_s1_dq - 1j * w_M * state.psi_s1_dq
        d_psi_s2_dq = u_s2_dq - par.R_s * i_s2_dq - 2j * w_M * state.psi_s2_dq
        d_theta_M = w_M
        return [d_psi_s1_dq, d_psi_s2_dq, d_theta_M]

    def create_time_series(self, t: np.ndarray) -> "MachineTimeSeries":
        """Create time series from state list."""
        return MachineTimeSeries(t, self)


@dataclass
class MachineTimeSeries(SubsystemTimeSeries):
    """Continuous time series."""

    t: InitVar[np.ndarray]
    subsystem: InitVar[BearinglessSynchronousMachine]
    # States
    psi_s1_dq: np.ndarray = field(default_factory=empty_array)
    psi_s2_dq: np.ndarray = field(default_factory=empty_array)
    theta_M: np.ndarray = field(default_factory=empty_array)
    # Outputs
    i_c_ab: np.ndarray = field(default_factory=empty_array)
    i_s1_ab: np.ndarray = field(default_factory=empty_array)
    i_s2_ab: np.ndarray = field(default_factory=empty_array)
    tau_M: np.ndarray = field(default_factory=empty_array)
    f_M_ab: np.ndarray = field(default_factory=empty_array)
    # Inputs
    u_c_ab: np.ndarray = field(default_factory=empty_array)
    w_M: np.ndarray = field(default_factory=empty_array)
    q_M_ab: np.ndarray = field(default_factory=empty_array)
    # Derived signals
    exp_j_theta_M: np.ndarray = field(default_factory=empty_array)
    psi_s1_ab: np.ndarray = field(default_factory=empty_array)
    psi_s2_ab: np.ndarray = field(default_factory=empty_array)
    u_s1_ab: np.ndarray = field(default_factory=empty_array)
    u_s2_ab: np.ndarray = field(default_factory=empty_array)
    i_s1_dq: np.ndarray = field(default_factory=empty_array)
    i_s2_dq: np.ndarray = field(default_factory=empty_array)
    f_M_dq: np.ndarray = field(default_factory=empty_array)

    def __post_init__(
        self, t: np.ndarray, subsystem: BearinglessSynchronousMachine
    ) -> None:
        self.psi_s1_dq = np.array(subsystem._history["psi_s1_dq"])
        self.psi_s2_dq = np.array(subsystem._history["psi_s2_dq"])
        theta_M = np.real(np.array(subsystem._history["theta_M"]))
        self.exp_j_theta_M = np.exp(1j * theta_M)
        self.theta_M = np.angle(self.exp_j_theta_M)
        self.psi_s1_ab = self.exp_j_theta_M * self.psi_s1_dq
        self.psi_s2_ab = self.exp_j_theta_M**2 * self.psi_s2_dq

    def set_outputs(
        self, t: np.ndarray, subsystem: BearinglessSynchronousMachine
    ) -> None:
        """Compute the outputs, which depend on the rotor displacement."""
        (self.i_c_ab, self.i_s1_ab, self.i_s2_ab, self.tau_M, self.f_M_ab) = (
            subsystem.compute_outputs(self, self.q_M_ab)
        )
        self.i_s1_dq = np.conj(self.exp_j_theta_M) * self.i_s1_ab
        self.i_s2_dq = np.conj(self.exp_j_theta_M) ** 2 * self.i_s2_ab
        self.f_M_dq = np.conj(self.exp_j_theta_M) * self.f_M_ab

    def compute_derived_signals(
        self, t: np.ndarray, subsystem: BearinglessSynchronousMachine
    ) -> None:
        """Compute the voltages of the planes."""
        self.u_s1_ab, self.u_s2_ab = vsd(*self.u_c_ab)


# %%
@dataclass
class RadialInputs:
    """Input variables."""

    f_M_ab: complex = 0j  # Electromagnetic radial force (N)


@dataclass
class RadialOutputs:
    """Output variables for interconnection."""

    q_M_ab: complex  # Rotor displacement (m)


class ExternalRotorPosition(Subsystem):
    """
    Rotor held at an externally given radial position.

    This corresponds to a rotor supported by active magnetic bearings or by mechanical
    bearings, in which case the radial force of the machine has no effect on the
    position. The displacement is constant or a function of time, defaulting to a
    centric rotor.

    """

    def __init__(self) -> None:
        self.inp: RadialInputs = RadialInputs()
        self.out: RadialOutputs = RadialOutputs(q_M_ab=0j)
        self.state = None
        self.q_M_ab: complex | Callable[[float], complex] = 0j

    def set_external_displacement(
        self, q_M_ab: complex | Callable[[float], complex]
    ) -> None:
        """Set the rotor displacement (m), constant or a function of time."""
        self.q_M_ab = q_M_ab

    def set_outputs(self, t: float) -> None:
        """Set output variables."""
        self.out.q_M_ab = complex(get_value(self.q_M_ab, t))

    def rhs(self, t: float) -> list[complex]:
        """No states."""
        return []

    def create_time_series(self, t: np.ndarray) -> "ExternalRotorPositionTimeSeries":
        """Create time series."""
        return ExternalRotorPositionTimeSeries(t, self)


@dataclass
class ExternalRotorPositionTimeSeries(SubsystemTimeSeries):
    """Continuous time series."""

    t: InitVar[np.ndarray]
    subsystem: InitVar[ExternalRotorPosition]
    q_M_ab: np.ndarray = field(default_factory=empty_array)

    def __post_init__(self, t: np.ndarray, subsystem: ExternalRotorPosition) -> None:
        self.q_M_ab = np.array([complex(get_value(subsystem.q_M_ab, t)) for t in t])


@dataclass
class RadialMechanicsStates:
    """State variables."""

    q_M_ab: complex = 0j
    v_M_ab: complex = 0j


class RadialMechanics(Subsystem):
    """
    Radial rigid-body dynamics of the rotor.

    The rotor is modeled as a point mass, which is acted upon by the electromagnetic
    radial force of the machine, an optional external force (e.g., gravity or
    unbalance), and an optional viscous damping. The tilting and axial motion are not
    modeled.

    Parameters
    ----------
    m : float
        Mass of the rotor (kg).
    c : float, optional
        Viscous damping coefficient (N/(m/s)), defaults to 0.

    """

    def __init__(self, m: float, c: float = 0.0) -> None:
        self.m = m
        self.c = c
        self.inp: RadialInputs = RadialInputs()
        self.state: RadialMechanicsStates = RadialMechanicsStates()
        self.out: RadialOutputs = RadialOutputs(q_M_ab=0j)
        self.f_ext_ab: complex | Callable[[float], complex] = 0j

    def set_external_force(
        self, f_ext_ab: complex | Callable[[float], complex]
    ) -> None:
        """Set the external radial force (N), constant or a function of time."""
        self.f_ext_ab = f_ext_ab

    def set_outputs(self, t: float) -> None:
        """Set output variables."""
        self.out.q_M_ab = self.state.q_M_ab

    def rhs(self, t: float) -> list[complex]:
        """Compute state derivatives."""
        state, inp = self.state, self.inp
        d_q_M_ab = state.v_M_ab
        f_ext_ab = get_value(self.f_ext_ab, t)
        d_v_M_ab = (inp.f_M_ab + f_ext_ab - self.c * state.v_M_ab) / self.m
        return [d_q_M_ab, d_v_M_ab]

    def create_time_series(self, t: np.ndarray) -> "RadialMechanicsTimeSeries":
        """Create time series."""
        return RadialMechanicsTimeSeries(t, self)


@dataclass
class RadialMechanicsTimeSeries(SubsystemTimeSeries):
    """Continuous time series."""

    t: InitVar[np.ndarray]
    subsystem: InitVar[RadialMechanics]
    q_M_ab: np.ndarray = field(default_factory=empty_array)
    v_M_ab: np.ndarray = field(default_factory=empty_array)
    f_ext_ab: np.ndarray = field(default_factory=empty_array)

    def __post_init__(self, t: np.ndarray, subsystem: RadialMechanics) -> None:
        self.q_M_ab = np.array(subsystem._history["q_M_ab"])
        self.v_M_ab = np.array(subsystem._history["v_M_ab"])
        self.f_ext_ab = np.array([complex(get_value(subsystem.f_ext_ab, t)) for t in t])


# %%
@dataclass
class BearinglessDriveSensors(DriveSensors):
    """
    Sensors of a bearingless drive, ideal by default, see `Sensor`.

    Parameters
    ----------
    i_c : Sensor, optional
        Phase-current sensors of the two sets, `[[i_1, i_2], [i_3, i_4], [i_5, i_6]]`.
        Their conversion defaults to `complex2abc`. They cannot have a bandwidth.
    u_dc : Sensor, optional
        DC-bus voltage sensor.
    w_M : Sensor, optional
        Rotor speed sensor (mechanical rad/s), defaults to None, i.e., no sensor.
    theta_M : Sensor, optional
        Rotor position sensor (mechanical rad), defaults to None, i.e., no sensor. The
        sensored control uses the speed and position sensors. The position sensor cannot
        have a bandwidth, since the angle is wrapped.
    q_M : Sensor, optional
        Rotor displacement sensor (m) in stator coordinates.

    """

    q_M: Sensor = field(default_factory=Sensor)


class BearinglessDrive(Model[MultiThreePhaseConverter]):
    """
    Continuous-time system model of a six-phase bearingless machine drive.

    Parameters
    ----------
    machine : BearinglessSynchronousMachine
        Machine model.
    mechanics : MechanicalSystem | TwoMassMechanicalSystem | ExternalRotorSpeed
        Rotational mechanics.
    radial : ExternalRotorPosition | RadialMechanics
        Radial mechanics.
    converter : MultiThreePhaseConverter
        Converter model with two three-phase sets.
    sensors : BearinglessDriveSensors, optional
        Sensors, see `BearinglessDriveSensors`, defaults to ideal phase-current, DC-bus
        voltage, and displacement sensors. Sensored control also needs the speed and
        position sensors.

    """

    def __init__(
        self,
        machine: BearinglessSynchronousMachine,
        mechanics: MechanicalSystem | TwoMassMechanicalSystem | ExternalRotorSpeed,
        radial: ExternalRotorPosition | RadialMechanics,
        converter: MultiThreePhaseConverter,
        sensors: BearinglessDriveSensors | None = None,
    ) -> None:
        if converter.n_sets != 2:
            raise ValueError("The converter must have two three-phase sets.")
        self.machine = machine
        self.mechanics = mechanics
        self.radial = radial
        self.sensors = BearinglessDriveSensors() if sensors is None else sensors
        if self.sensors.i_c.state is not None:
            raise ValueError("The current sensors cannot have a bandwidth")
        subsystems: dict[str, Subsystem] = {
            "converter": converter,
            "radial": radial,
            "machine": machine,
            "mechanics": mechanics,
            "i_c_sensor": self.sensors.i_c,
            "u_dc_sensor": self.sensors.u_dc,
            "q_M_sensor": self.sensors.q_M,
        }
        connections = {
            ("machine", "u_c_ab"): ("converter", "u_c_ab"),
            ("machine", "w_M"): ("mechanics", "w_M"),
            ("machine", "q_M_ab"): ("radial", "q_M_ab"),
            ("converter", "i_c_ab"): ("machine", "i_c_ab"),
            ("mechanics", "tau_M"): ("machine", "tau_M"),
            ("radial", "f_M_ab"): ("machine", "f_M_ab"),
            ("i_c_sensor", "x"): ("machine", "i_c_ab"),
            ("u_dc_sensor", "x"): ("converter", "u_dc"),
            ("q_M_sensor", "x"): ("radial", "q_M_ab"),
        }
        if self.sensors.w_M is not None:
            subsystems["w_M_sensor"] = self.sensors.w_M
            connections[("w_M_sensor", "x")] = ("mechanics", "w_M")
        if self.sensors.theta_M is not None:
            if self.sensors.theta_M.state is not None:
                raise ValueError("The position sensor cannot have a bandwidth")
            subsystems["theta_M_sensor"] = self.sensors.theta_M
            connections[("theta_M_sensor", "x")] = ("mechanics", "theta_M")
        super().__init__(converter, subsystems, connections)
