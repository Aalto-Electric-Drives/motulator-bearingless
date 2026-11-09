"""
2.6-kW bearingless SyRM, GradNet from FEM data, levitation control
==================================================================

This example simulates levitation control of a 2.6-kW six-phase bearingless synchronous
reluctance machine (SyRM) drive. The machine model is a GradNet current map trained on
finite-element data, including magnetic saturation, the spatial harmonics, and the
unbalanced magnetic pull. The force is controlled by the direct force controller
[#Das2026]_, parametrized by a GradNet flux map with the spatial harmonics averaged. Its
radial force estimate includes the unbalanced magnetic pull, since the force map depends
on the measured displacement.

"""

# %%
from math import pi
from pathlib import Path

import motulator_bearingless.gradnet as gn
from motulator_bearingless import control, model, utils

# %%
# Compute base values based on the nominal values (just for figures).

nom = utils.NominalValues(U=220, I=12, f=50, P=2.6e3, tau=14.5)
base = utils.BaseValues.from_nominal(nom, n_p=2)

# %%
# Configure the system model. The rotor mass roughly corresponds to the rotor
# lamination and the shaft inside it.

p = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
R_s = 0.215  # Stator resistance from [#Das2026]_
path = "trained_models/bsyrm_fem_curr_map_harm_softmax_d128.npz"
current_map = gn.CurrentMapWithHarmonics(gn.load_gradnet(p / path))
par = model.SpatialSaturatedBearinglessSynchronousMachinePars(R_s, current_map)

m = 20.0
machine = model.BearinglessSynchronousMachine(par)
mechanics = model.MechanicalSystem(J=0.15)
radial = model.RadialMechanics(m=m)
converter = model.MultiThreePhaseConverter(u_dc=540)
sensors = model.BearinglessDriveSensors(w_M=model.Sensor(), theta_M=model.Sensor())
mdl = model.BearinglessDrive(machine, mechanics, radial, converter, sensors)

# %%
# Configure the control system.

path = "trained_models/bsyrm_fem_flux_map_softmax_d128.npz"
est_flux_map = gn.FluxMap(gn.load_gradnet(p / path))
est_par = control.SaturatedBearinglessSynchronousMachinePars(
    R_s, psi_s_fcn=est_flux_map
)

cfg = control.DirectForceControllerCfg(i_s_max=1.5 * base.i, psi_s_min=0.4)
force_ctrl = control.DirectForceController(est_par, cfg)
speed_ctrl = control.SpeedController(J=0.15, alpha_s=2 * pi * 8)
levitation_ctrl = control.LevitationController(
    m=m, alpha_q=2 * pi * 50, alpha_o=2 * pi * 150
)
ctrl = control.DirectForceControlSystem(force_ctrl, speed_ctrl, levitation_ctrl)

# %%
# Set the references and the external loads. The rotor is accelerated to the rated speed
# of 1700 r/min. The gravity is applied as a step at t = 0.1 s in the negative direction
# of the j axis. A displacement reference step of 50 µm is applied at t = 1.4 s and a
# load torque step at t = 1.6 s.

w_M_rated = 2 * pi * 1700 / 60
ctrl.set_speed_ref(lambda t: min(t / 1.2, 1) * w_M_rated)
ctrl.set_displacement_ref(lambda t: (t > 1.4) * 50e-6)
radial.set_external_force(lambda t: (t > 0.1) * -9.81j * radial.m)
mdl.mechanics.set_external_load_torque(lambda t: (t > 1.6) * 0.7 * nom.tau)

# %%
# Create the simulation object, simulate, and plot the results.

sim = model.Simulation(mdl, ctrl)
res = sim.simulate(t_stop=1.8)
subplots = ["speed", "torque", "force", "displacement", "current"]
f_lims = (-60.0, 500.0)  # Room for the legends
y_lims = [None, (-0.05, 1.75), f_lims, (-75.0, 60.0), (-1.25, 0.9)]
utils.plot(res, base, subplots, y_lims=y_lims)

# %%
# Plot the response to the gravity step and to the displacement reference step. The
# static force in the end of the simulation includes the force of the unbalanced
# magnetic pull at the displacement of 50 µm.

utils.plot(res, base, ["force", "displacement"], t_lims=(0.08, 0.2), y_lims=[f_lims])
utils.plot(res, base, ["force", "displacement"], t_lims=(1.38, 1.5), y_lims=[f_lims])

# %%
# .. rubric:: References
#
# .. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
#    bearingless synchronous machines," Proc. ICEM, 2026,
#    https://doi.org/10.1109/ICEM68283.2026.11757980
