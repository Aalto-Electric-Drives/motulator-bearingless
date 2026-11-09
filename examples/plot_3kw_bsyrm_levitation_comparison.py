"""
2.6-kW bearingless SyRM, comparison of control models in levitation control
===========================================================================

This example compares two parametrizations of the direct force controller [#Das2026]_
in the levitation control of a 2.6-kW six-phase bearingless synchronous reluctance
machine (SyRM). The machine model is a GradNet current map with the spatial harmonics.
The controller uses either constant parameters, computed at the rated peak current, or a
GradNet flux map. The rotor runs at the rated speed, driven externally.

The unbalanced magnetic pull is a negative stiffness in the radial dynamics. The force
estimate of the flux map includes it, while that of the constant-parameter model omits
it, leaving the pull for the disturbance observer of the levitation controller.

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
# Load the magnetic models and compute the stiffness of the unbalanced magnetic pull
# from the force map at constant currents. The rotor mass is that of the levitation
# examples.

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
m = 20.0
for i_s2 in (13.5 + 0j, 12 + 12j):
    k_u = np.real(est_par.f_M(0j, i_s2, 1e-6))  # Force (N) at the displacement of 1 µm
    print(f"Stiffness of the pull at i_s2 = {i_s2} A: {k_u:.1f} N/µm")

# %%
# Configure the simulation. Without the pull in the force estimate, the stability of the
# displacement control is limited. For one axis in continuous time, an ideal inner force
# control loop, and `alpha_o = 3*alpha_q`, the closed-loop poles stay in the left
# half-plane only if the stiffness is below `1.39*m*alpha_q**2`. The limit is 2.75 N/µm
# at the bandwidth of 50 Hz and 11 N/µm at the bandwidth of 100 Hz used here. The
# constant-parameter controller would hence be unstable at 50 Hz, unlike the flux map.

alpha_q = 2 * pi * 100
w_M_rated = 2 * pi * 1700 / 60
t_stop = 0.6


def simulate(par_c: control.BearinglessMagneticModel):
    """Simulate the drive with the given control model."""
    machine = model.BearinglessSynchronousMachine(par)
    mechanics = model.ExternalRotorSpeed()
    mechanics.set_external_rotor_speed(w_M_rated)
    radial = model.RadialMechanics(m=m)
    radial.set_external_force(lambda t: (t > 0.1) * -9.81j * m)
    converter = model.MultiThreePhaseConverter(u_dc=540)
    sensors = model.BearinglessDriveSensors(w_M=model.Sensor(), theta_M=model.Sensor())
    mdl = model.BearinglessDrive(machine, mechanics, radial, converter, sensors)
    cfg = control.DirectForceControllerCfg(i_s_max=25.5, psi_s_min=0.4)
    force_ctrl = control.DirectForceController(par_c, cfg)
    levitation_ctrl = control.LevitationController(m=m, alpha_q=alpha_q)
    ctrl = control.DirectForceControlSystem(force_ctrl, None, levitation_ctrl)
    ctrl.set_torque_ref(lambda t: (t > 0.45) * 0.7 * nom.tau)
    ctrl.set_displacement_ref(lambda t: (t > 0.3) * 50e-6)
    return model.Simulation(mdl, ctrl).simulate(t_stop=t_stop)


res_lin = simulate(est_par_lin)
res_map = simulate(est_par)
for name, res in (("Constant parameters", res_lin), ("Flux map", res_map)):
    q_M_ab = res.mdl.machine.q_M_ab
    print(
        f"{name}: peak displacement {1e6 * np.max(np.abs(q_M_ab)):.1f} µm, "
        f"final displacement {1e6 * np.mean(q_M_ab[res.mdl.t > 0.55]):.1f} µm"
    )

# %%
# Plot the results of the constant-parameter controller. Gravity is applied at
# t = 0.1 s, the displacement reference step at t = 0.3 s, and the torque reference
# step at t = 0.45 s.

y_lims = [(-800.0, 600.0), (-30.0, 70.0)]  # Same scales for both controllers
utils.plot(res_lin, base, ["force", "displacement"], y_lims=y_lims)

# %%
# Plot the results of the flux-map controller.

utils.plot(res_map, base, ["force", "displacement"], y_lims=y_lims)

# %%
# .. rubric:: References
#
# .. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
#    bearingless synchronous machines," Proc. ICEM, 2026,
#    https://doi.org/10.1109/ICEM68283.2026.11757980
