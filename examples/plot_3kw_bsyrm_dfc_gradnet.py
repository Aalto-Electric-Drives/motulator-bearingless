"""
2.6-kW bearingless SyRM, GradNet from FEM data, direct force control
====================================================================

This example simulates direct radial force control of a 2.6-kW six-phase bearingless
synchronous reluctance machine (SyRM) drive, as in [#Das2026]_. The machine model is a
GradNet current map trained on finite-element data, including magnetic saturation, the
rotor displacement, and the spatial harmonics. The control system uses a GradNet flux
map trained on the same data, the spatial harmonics being averaged over the rotor angle.
The rotor speed and the rotor angle are measured, and the rotor is held at a fixed
radial position by external bearings.

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
# Configure the system model using the GradNet current map with spatial harmonics.

p = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
R_s = 0.215  # Stator resistance from [#Das2026]_
path = "trained_models/bsyrm_fem_curr_map_harm_softmax_d128.npz"
current_map = gn.CurrentMapWithHarmonics(gn.load_gradnet(p / path))
par = model.SpatialSaturatedBearinglessSynchronousMachinePars(R_s, current_map)

machine = model.BearinglessSynchronousMachine(par)
w_M_rated = 2 * pi * 1700 / 60
k = nom.tau / w_M_rated**2  # Quadratic load torque, rated torque at the rated speed
mechanics = model.MechanicalSystem(J=0.05, B_L=lambda w_M: k * abs(w_M))
converter = model.MultiThreePhaseConverter(u_dc=540)
radial = model.ExternalRotorPosition()
radial.set_external_displacement(30e-6 - 20e-6j)
sensors = model.BearinglessDriveSensors(w_M=model.Sensor(), theta_M=model.Sensor())
mdl = model.BearinglessDrive(machine, mechanics, radial, converter, sensors)

# %%
# Configure the control system using the GradNet flux map without spatial harmonics.

path = "trained_models/bsyrm_fem_flux_map_softmax_d128.npz"
est_flux_map = gn.FluxMap(gn.load_gradnet(p / path))
est_par = control.SaturatedBearinglessSynchronousMachinePars(
    R_s, psi_s_fcn=est_flux_map
)

cfg = control.DirectForceControllerCfg(i_s_max=1.5 * base.i, psi_s_min=0.4)
force_ctrl = control.DirectForceController(est_par, cfg)
speed_ctrl = control.SpeedController(J=0.05, alpha_s=2 * pi * 4)
ctrl = control.DirectForceControlSystem(force_ctrl, speed_ctrl)

# %%
# Visualize the control loci of the four-pole plane.

mc = utils.MachineCharacteristics(utils.fourpole_pars(est_par))
mc.plot_current_loci([1, 1.5], base)
mc.plot_flux_loci([1, 1.5], base)

# %%
# Set the speed and force references. The machine is magnetized during the first 0.2 s,
# after which the rotor is accelerated to the rated speed of 1700 r/min in 0.6 s against
# the quadratic load torque. The radial force references of 100 N and 150 N, as in
# [#Das2026]_, are applied during the acceleration at t = 0.4 s and t = 0.6 s.

ctrl.set_speed_ref(lambda t: min(max(t - 0.2, 0) / 0.6, 1) * w_M_rated)
ctrl.set_force_ref(lambda t: (t > 0.4) * 100 + 1j * (t > 0.6) * 150)

# %%
# Create the simulation object, simulate, and plot the results.

sim = model.Simulation(mdl, ctrl)
res = sim.simulate(t_stop=1.2)
i_lims = (-1.0, 0.95)  # Room for the legends
utils.plot(res, base, y_lims=[None, None, None, None, i_lims])

# %%
# Plot the force steps. The ripple of the actual force and torque is caused by the
# spatial harmonics, which are not included in the control model.

y_lims = [(-0.05, 1.35), (-30.0, 320.0), i_lims]
utils.plot(
    res, base, ["torque", "force", "current"], t_lims=(0.35, 0.95), y_lims=y_lims
)
mask = res.mdl.t > 1
print(
    "Mean force in the steady state (N):",
    np.round(np.mean(res.mdl.machine.f_M_ab[mask]), 1),
)

# %%
# .. rubric:: References
#
# .. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
#    bearingless synchronous machines," Proc. ICEM, 2026,
#    https://doi.org/10.1109/ICEM68283.2026.11757980
