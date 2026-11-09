"""
2.6-kW bearingless SyRM, comparison of control models in force control
======================================================================

This example compares two parametrizations of the direct force controller [#Das2026]_
of a 2.6-kW six-phase bearingless synchronous reluctance machine (SyRM). The machine
model is a GradNet current map with the spatial harmonics. The controller uses either
constant parameters, computed at the rated peak current, or a GradNet flux map, which
includes saturation and the unbalanced magnetic pull. The rotor runs at the rated speed,
driven externally, and it is held at a fixed radial position. After magnetization, the
radial force references are applied, followed by a torque reference step.

"""

# %%
from math import pi
from pathlib import Path

import numpy as np

import motulator_bearingless.gradnet as gn
from motulator_bearingless import control, model, utils

# %%
# Compute base values based on the nominal values (just for figures).

nom = utils.NominalValues(U=220, I=12, f=50, P=2.6e3, tau=14.5)
base = utils.BaseValues.from_nominal(nom, n_p=2)

# %%
# Load the magnetic models and compute the constant parameters from the flux map.

p = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
R_s = 0.215  # Stator resistance from [#Das2026]_
path = "trained_models/bsyrm_fem_curr_map_harm_softmax_d128.npz"
current_map = gn.CurrentMapWithHarmonics(gn.load_gradnet(p / path))
par = model.SpatialSaturatedBearinglessSynchronousMachinePars(R_s, current_map)

path = "trained_models/bsyrm_fem_flux_map_softmax_d128.npz"
est_flux_map = gn.FluxMap(gn.load_gradnet(p / path))
est_par = control.SaturatedBearinglessSynchronousMachinePars(
    R_s, psi_s_fcn=est_flux_map
)
est_par_lin = est_par.linearized(12 + 12j)
print(est_par_lin)

# %%
# Configure the simulation, a static displacement as in [#Das2026]_.

w_M_rated = 2 * pi * 1700 / 60
q_M = 30e-6 - 20e-6j


def simulate(par_c: control.BearinglessMagneticModel):
    """Simulate the drive with the given control model."""
    machine = model.BearinglessSynchronousMachine(par)
    mechanics = model.ExternalRotorSpeed()
    mechanics.set_external_rotor_speed(w_M_rated)
    radial = model.ExternalRotorPosition()
    radial.set_external_displacement(q_M)
    converter = model.MultiThreePhaseConverter(u_dc=540)
    sensors = model.BearinglessDriveSensors(w_M=model.Sensor(), theta_M=model.Sensor())
    mdl = model.BearinglessDrive(machine, mechanics, radial, converter, sensors)
    cfg = control.DirectForceControllerCfg(i_s_max=25.5, psi_s_min=0.4)
    ctrl = control.DirectForceControlSystem(control.DirectForceController(par_c, cfg))
    ctrl.set_torque_ref(lambda t: (t > 0.35) * 0.7 * nom.tau)
    ctrl.set_force_ref(lambda t: (t > 0.15) * 100 + 1j * (t > 0.25) * 150)
    return model.Simulation(mdl, ctrl).simulate(t_stop=0.5)


res_lin = simulate(est_par_lin)
res_map = simulate(est_par)
for name, res in (("Constant parameters", res_lin), ("Flux map", res_map)):
    mask = res.mdl.t > 0.42
    print(
        f"{name}: mean force {np.mean(res.mdl.machine.f_M_ab[mask]):.1f} N, "
        f"mean torque {np.mean(res.mdl.machine.tau_M[mask]):.2f} Nm"
    )

# %%
# Plot the results of the constant-parameter controller. Its force estimate omits the
# unbalanced magnetic pull, whose force at this displacement is of the same order as
# the force reference. Since the force is not measured, the error remains in the steady
# state.

y_lims = [(-0.1, 1.0), (-500.0, 450.0)]  # Same scales, room for the legends
utils.plot(res_lin, base, ["torque", "force"], y_lims=y_lims)

# %%
# Plot the results of the flux-map controller. The force estimate includes the pull,
# and the ripple comes from the spatial harmonics of the machine model.

utils.plot(res_map, base, ["torque", "force"], y_lims=y_lims)

# %%
# .. rubric:: References
#
# .. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
#    bearingless synchronous machines," Proc. ICEM, 2026,
#    https://doi.org/10.1109/ICEM68283.2026.11757980
