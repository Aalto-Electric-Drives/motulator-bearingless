"""
2.6-kW bearingless SyRM, levitation control
===========================================

This example simulates levitation control of a 2.6-kW six-phase bearingless synchronous
reluctance machine (SyRM) drive. The rotor is a point mass, whose radial position is
controlled by means of state feedback with a disturbance observer. The levitation
controller gives the radial force reference to the direct force controller [#Das2026]_.
The machine and the controller use the same constant parameters, i.e., magnetic
saturation and the unbalanced magnetic pull are omitted, while the inductances depend
linearly on the displacement.

"""

# %%
from math import pi

from motulator_bearingless import control, model, utils

# %%
# Compute base values based on the nominal values (just for figures).

nom = utils.NominalValues(U=220, I=12, f=50, P=2.6e3, tau=14.5)
base = utils.BaseValues.from_nominal(nom, n_p=2)

# %%
# Configure the system model. The machine parameters are those of the direct force
# control example. The rotor mass roughly corresponds to the rotor lamination and the
# shaft inside it.

par = model.BearinglessSynchronousMachinePars(
    R_s=0.215, L_1=0.03, L_d=0.03, L_q=0.01, m_d=10.0, m_q=0.2
)
m = 20.0
machine = model.BearinglessSynchronousMachine(par)
mechanics = model.MechanicalSystem(J=0.15)
radial = model.RadialMechanics(m=m)
converter = model.MultiThreePhaseConverter(u_dc=540)
sensors = model.BearinglessDriveSensors(w_M=model.Sensor(), theta_M=model.Sensor())
mdl = model.BearinglessDrive(machine, mechanics, radial, converter, sensors)

# %%
# Configure the control system. The observer of the levitation controller is driven by
# the radial force estimate, so its poles can be placed above the displacement-control
# bandwidth without considering the dynamics of the inner force control loop.

cfg = control.DirectForceControllerCfg(i_s_max=1.5 * base.i, psi_s_min=0.3)
force_ctrl = control.DirectForceController(par, cfg)
speed_ctrl = control.SpeedController(J=0.15, alpha_s=2 * pi * 8)
levitation_ctrl = control.LevitationController(
    m=m, alpha_q=2 * pi * 50, alpha_o=2 * pi * 150
)
ctrl = control.DirectForceControlSystem(force_ctrl, speed_ctrl, levitation_ctrl)

# %%
# Set the references and the external loads. The rotor is accelerated to the rated
# speed of 1700 r/min. The gravity is applied as a step at t = 0.1 s in the negative
# direction of the j axis. A displacement reference step of 50 µm is applied at
# t = 1.4 s and a load torque step at t = 1.6 s.

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
f_lims = (-40.0, 450.0)  # Room for the legends
utils.plot(
    res, base, subplots, y_lims=[None, (-0.05, 1.625), f_lims, None, (-1.0, 0.9)]
)

# %%
# Plot the response to the gravity step and to the displacement reference step.

utils.plot(res, base, ["force", "displacement"], t_lims=(0.08, 0.2), y_lims=[f_lims])
utils.plot(res, base, ["force", "displacement"], t_lims=(1.38, 1.5), y_lims=[f_lims])

# %%
# .. rubric:: References
#
# .. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
#    bearingless synchronous machines," Proc. ICEM, 2026,
#    https://doi.org/10.1109/ICEM68283.2026.11757980
