# Control

This document describes the control methods of the {mod}`motulator_bearingless.control` module. Two force control methods are implemented: the direct force controller {cite}`Das2026` and, for comparison, a current-based force controller. In both methods, the rotor displacement is measured and it can be controlled by the levitation controller.

## Direct Force Control

The {class}`motulator_bearingless.control.DirectForceController` class implements the feedback-linearizing controller of {cite}`Das2026`. The magnetic model of the controller is given by the parameters, either the constant parameters or a flux map without the rotor angle, see {doc}`gradnet`. The voltage reference is

```{math}
---
label: voltage_reference
---
    \ush{1}^\mathrm{ref} = \Rs\ish{1} + \jj\omegaM\psish{1} + \mathbf{e}_1 \qquad
    \ush{2}^\mathrm{ref} = \Rs\ish{2} + 2\jj\omegaM\psish{2} + \mathbf{e}_2
```

where the control inputs $\mathbf{e}_1$ and $\mathbf{e}_2$ linearize the dynamics of the radial force and of the flux linkage and the torque, respectively.

### Radial Force

With the voltage reference {eq}`voltage_reference` and an accurate model, the force dynamics in rotor coordinates are

```{math}
---
label: force_derivative
---
    \frac{\D\fM}{\D t} = -3\mathbf{G}_1\mathbf{e}_1 - 3\mathbf{G}_2\mathbf{e}_2 + \frac{\partial\fM}{\partial\qM}\frac{\D\qM}{\D t}
```

where $\mathbf{G}_h$ is the transpose of the sensitivity of the current of the plane $h$ to the displacement, and the sensitivities are evaluated at constant flux linkages. The control law neglects the cross term $-3\mathbf{G}_2\mathbf{e}_2$, i.e., the planes are assumed to be decoupled in the control design, while the machine model includes the couplings. The cross term does not vanish even at a centric rotor with linear magnetics, since $\mathbf{G}_2$ is proportional to the two-pole current. Hence, a flux or torque transient of the four-pole plane disturbs the force. The control law is designed in stator coordinates, where the force reference and the disturbances are constant in the steady state. The closed-loop dynamics become

```{math}
---
label: force_dynamics
---
    \frac{\D\fM^\mathrm{s}}{\D t} = \alpha_\mathrm{f}(\fM^\mathrm{s,ref} - \fM^\mathrm{s}) + \mathbf{d} - \hat{\mathbf{d}}
```

where $\mathbf{d}$ is the disturbance, which includes the neglected cross term and the effects of the model errors. Its estimate $\hat{\mathbf{d}}$ comes from the disturbance observer {class}`motulator_bearingless.control.ForceDisturbanceObserver`, providing the integral action. Since the force is not measured, the integral action acts on the force estimate, and an error of the force map remains in the steady state.

### Flux Linkage and Torque

The four-pole plane is controlled in the same manner as in flux-vector control. The effect of the two-pole input $\mathbf{e}_1$ on the torque is neglected, in the same manner as the cross term in {eq}`force_derivative`. The flux and torque references are computed from the maximum-torque-per-ampere and maximum-torque-per-volt loci of the magnetic model at zero two-pole current and at a centric rotor, using {class}`motulator.drive.control.sm.ReferenceGenerator` with the equivalent three-phase machine, see {func}`motulator_bearingless.utils.fourpole_pars`. A minimum flux magnitude provides the bias flux for the force production. The two-pole plane loads the converter only slightly, so that the voltage limit is assigned to the four-pole plane.

### Flux Observer

The stator flux linkages are estimated by {class}`motulator_bearingless.control.FluxObserver` in rotor coordinates, based on the voltage model, corrected by the flux map of the machine {cite}`Das2026`. In the sensorless mode, also the rotor angle and the rotor speed are estimated, as in {cite}`Hin2018`.

```{note}
The sensorless mode with a speed controller is sensitive to the errors of the control model at no load. A limit cycle in the torque and the q-axis current appears already with an overestimation of about 10 % in the d-axis inductance, and with the mismatch between the current map and the flux map, while the operation is stable with the measured speed and angle, with a torque reference instead of the speed controller, and under load. The flux-vector controller of *motulator* behaves similarly in the same conditions. Therefore, the examples with the gradient-network maps use the measured speed and angle, as in {cite}`Das2026`.
```

## Levitation Control

The {class}`motulator_bearingless.control.LevitationController` class gives the force reference from a state feedback law based on the estimates of a disturbance observer

```{math}
---
label: levitation
---
    \fM^\mathrm{s,ref} = m\left[\alpha_q^2(\qM^\mathrm{s,ref} - \hat{\qM}^\mathrm{s}) - 2\alpha_q\hat{\vM}^\mathrm{s}\right] - \hat{\mathbf{f}}_\mathrm{ext}^\mathrm{s}
```

which results in the double closed-loop pole at $s = -\alpha_q$. The observer is driven by the radial force estimate, so that the dynamics and the limits of the inner force control loop do not cause windup.

The unbalanced magnetic pull is a negative stiffness in the radial dynamics. A flux map includes it in the force estimate. The constant parameters omit it, leaving the pull for the disturbance observer. Consider one axis in continuous time, with an ideal inner force control loop and a constant stiffness of the pull at constant currents. With $\alpha_\mathrm{o} = 3\alpha_q$, the closed-loop poles then stay in the left half-plane only if the stiffness is below $1.39\,m\alpha_q^2$, which limits the minimum bandwidth of the constant-parameter controller. The limit grows with the ratio $\alpha_\mathrm{o}/\alpha_q$, and the dynamics of the inner loop and the sampling change it.

## Current-Based Force Control

The {class}`motulator_bearingless.control.CurrentBasedForceControlSystem` class implements the classical current-based force control {cite}`Chiba2005`. It corresponds to the control principle of *motulator-multiphase*, where the common mode is controlled by a vector controller of a three-phase machine and the differential modes by current controllers, see {doc}`relation`. The four-pole plane is controlled by any synchronous-machine controller of *motulator*, e.g., {class}`motulator.drive.control.sm.CurrentVectorController` or {class}`motulator.drive.control.sm.FluxVectorController`, configured with the parameters of the equivalent three-phase machine, see {func}`motulator_bearingless.utils.fourpole_pars`. The feedback signals of the vector controller are the voltage of the four-pole plane and twice its current. The two-pole plane is current-controlled by {class}`motulator_bearingless.control.CurrentBasedForceController`, with the radial force reference converted to the current reference.

Both planes rotate at $2\omegaM$ in stator coordinates, so that the natural common coordinates are those of the four-pole plane, $\vartheta_\mathrm{c} = 2\thetaM$, which the vector controller provides also in the sensorless mode. In these coordinates, the two-pole current $\ish{1}^\mathrm{c} = \e^{-\jj\vartheta_\mathrm{c}}\ish{1}^\mathrm{s} = \e^{-\jj\thetaM}\ish{1}$ is constant in the steady state, and the force in stator coordinates follows from the linear force model {eq}`force_complex` as

```{math}
---
label: force_controller
---
    \fM^\mathrm{s} = 3\mathbf{k}\,(\ish{1}^\mathrm{c})^*
```

where $\mathbf{k}$ depends on the four-pole current in rotor coordinates, available from the vector controller as half of the current of the equivalent machine. The current reference of the two-pole plane is obtained by inverting {eq}`force_controller`,

```{math}
    \ish{1}^\mathrm{c,ref} = \left(\frac{\fM^\mathrm{s,ref}}{3\mathbf{k}}\right)^*
```

and no force can be produced if the machine is not magnetized ($\mathbf{k} = 0$). The current is controlled by a complex-vector PI controller in the flux-linkage form, $\boldsymbol{\uppsi}_1 = L_1\ish{1}^\mathrm{c}$, in coordinates rotating at $\omega_\mathrm{c} = 2\omegaM$ {cite}`Awa2019`, which is the controller of the differential modes in *motulator-multiphase*. The force estimate {eq}`force_controller` computed from the measured currents is fed to the levitation controller, as in direct force control. Since both planes rotate at $\omega_\mathrm{c}$ in stator coordinates, the voltage references of both planes are advanced by the same angle $k_\mathrm{comp}T_\mathrm{s}\omega_\mathrm{c}$ before the modulation, as in the vector control systems of *motulator*.

Compared to {class}`motulator_multiphase.control.MultiThreePhaseControlSystem`, three changes are needed: the conjugate in {eq}`vsd_sets` (or, equivalently, the plane-specific rotation $h\thetaM$), the force-to-current conversion in place of the current-sharing references, and the current limit of the two-pole plane. With linear magnetics, the direct force control law reduces to this current-based law {cite}`Das2026`. The methods differ in the transients and in how the model nonlinearities and the displacement enter the control.
