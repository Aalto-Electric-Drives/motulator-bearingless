"""
2.6-kW bearingless SyRM, direct force control
=============================================

This example simulates direct radial force control of a 2.6-kW six-phase bearingless
synchronous reluctance machine (SyRM) drive. The rotor is held at a fixed radial
position by external bearings. The references follow the simulations of [#Das2026]_,
but the rotor angle and the rotor speed are estimated instead of measured. The machine
and the controller use the same constant parameters, i.e., magnetic saturation and the
unbalanced magnetic pull are omitted.

"""

# %%
from math import pi

import numpy as np

from motulator_bearingless import control, model, utils

# %%
# Compute base values based on the nominal values (just for figures).

nom = utils.NominalValues(U=220, I=12, f=50, P=2.6e3, tau=14.5)
base = utils.BaseValues.from_nominal(nom, n_p=2)

# %%
# Configure the system model. The stator resistance and the inertia are from
# [#Das2026]_. The inductances and the radial force constants are scaled from the
# finite-element results of the same machine with separate windings [#Saa2016]_ such
# that the rated torque is reached at about the rated current. The values are
# illustrative.

par = model.BearinglessSynchronousMachinePars(
    R_s=0.215, L_1=0.03, L_d=0.03, L_q=0.01, m_d=10.0, m_q=0.2
)
machine = model.BearinglessSynchronousMachine(par)
mechanics = model.MechanicalSystem(J=0.15)
converter = model.MultiThreePhaseConverter(u_dc=540)

# Rotor held at a fixed position, e.g., by active magnetic bearings. A static
# eccentricity of 30 µm and -20 µm could be set as follows:
radial = model.ExternalRotorPosition()
# radial.set_external_displacement(30e-6 - 20e-6j)

mdl = model.BearinglessDrive(machine, mechanics, radial, converter)

# %%
# Configure the control system. The controller uses the same parameters as the model.

cfg = control.DirectForceControllerCfg(
    i_s_max=1.5 * base.i,
    psi_s_min=0.3,
    alpha_f=2 * pi * 300,
    alpha_i=2 * pi * 30,
    alpha_psi=2 * pi * 200,
    alpha_tau=2 * pi * 200,
    zeta=0.45,
    sensorless=True,
    T_s=125e-6,
)
force_ctrl = control.DirectForceController(par, cfg)
speed_ctrl = control.SpeedController(J=0.15, alpha_s=2 * pi * 8)
ctrl = control.DirectForceControlSystem(force_ctrl, speed_ctrl)

# %%
# Set the speed and force references and the external load torque. The rotor is
# accelerated to the rated speed of 1700 r/min, the radial force references of 100 N
# and 150 N are applied at t = 1.6 s and t = 1.9 s, and a load torque step at t = 2 s.

w_M_rated = 2 * pi * 1700 / 60
ctrl.set_speed_ref(lambda t: min(t / 1.2, 1) * w_M_rated)
ctrl.set_force_ref(lambda t: (t > 1.6) * 100 + 1j * (t > 1.9) * 150)
mdl.mechanics.set_external_load_torque(lambda t: (t > 2) * 0.7 * nom.tau)

# %%
# Create the simulation object, simulate, and plot the results.

sim = model.Simulation(mdl, ctrl)
res = sim.simulate(t_stop=3)
utils.plot(res, base, y_lims=[None, (-0.2, 0.82)])  # Room for the legends

# %%
# Plot the phase currents around the force step.

t_lims = (1.55, 1.65)
mask = (res.mdl.t > t_lims[0]) & (res.mdl.t < t_lims[1])
i_abc = utils.sixphase_currents(res.mdl.machine.i_c_ab[:, mask])
utils.plot_phase_currents(res.mdl.t[mask], i_abc, y_lims=(-11.5, 15.0))
print("Force at the end (N):", np.round(res.mdl.machine.f_M_ab[-1], 1))

# %%
# .. rubric:: References
#
# .. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
#    bearingless synchronous machines," Proc. ICEM, 2026,
#    https://doi.org/10.1109/ICEM68283.2026.11757980
#
# .. [#Saa2016] Saarakkala, Sokolov, Mukherjee, Pippuri, Tammi, Belahcen, Hinkkanen,
#    "Flux-linkage model including cross-saturation for a bearingless synchronous
#    reluctance motor," Proc. ISMB15, 2016.
