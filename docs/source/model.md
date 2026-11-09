# System Model

This document describes the continuous-time models of the {mod}`motulator_bearingless.model` module. The conventions are those of *motulator*: peak-valued complex space vectors, the superscript s marks the stationary coordinates, and space vectors without a superscript are in rotor coordinates. Uppercase M denotes mechanical quantities.

## Vector-Space Decomposition

The symmetrical six-phase winding consists of the phases $k = 1, \ldots, 6$, whose magnetic axes are at $60^\circ$ intervals. The winding is described by means of the vector-space decomposition {cite}`Ahm2023`

```{math}
---
label: vsd
---
    \mathbf{x}_{\mathrm{s}h}^\mathrm{s} = \frac{1}{3}\sum_{k=1}^{6} x_k \e^{\jj h (k - 1)\pi/3}
```

where the plane $h = 1$ is the two-pole plane and $h = 2$ the four-pole plane. The zero-sequence components are omitted. The first three-phase set consists of the phases 1, 3, 5 and the second set of the phases 2, 4, 6, see {func}`motulator_bearingless.utils.vsd`. The six-phase converter of {class}`motulator_multiphase.model.MultiThreePhaseConverter` feeds the two sets.

## Machine Model

The machine model {class}`motulator_bearingless.model.BearinglessSynchronousMachine` is represented in rotor coordinates, where the space vectors of the planes are $\ish{1} = \e^{-\jj\thetaM}\ish{1}^\mathrm{s}$ and $\ish{2} = \e^{-2\jj\thetaM}\ish{2}^\mathrm{s}$, $\thetaM$ being the mechanical rotor angle. The rotor displacement is $\qM = \e^{-\jj\thetaM}\qM^\mathrm{s}$ and the radial force $\fM = \e^{-\jj\thetaM}\fM^\mathrm{s}$, whose real and imaginary parts in stator coordinates correspond to the i and j axes. The stator voltage equations are

```{math}
---
label: voltage_equations
---
    \frac{\D\psish{1}}{\D t} = \ush{1} - \Rs\ish{1} - \jj\omegaM\psish{1} \qquad
    \frac{\D\psish{2}}{\D t} = \ush{2} - \Rs\ish{2} - 2\jj\omegaM\psish{2}
```

where $\omegaM$ is the mechanical rotor speed. A magnetic model gives the currents, the radial force, and the torque as functions of the flux linkages, the displacement, and the rotor angle. It is lossless, i.e., it follows from the magnetic energy $W(\psish{1}, \psish{2}, \qM, \thetaM)$ {cite}`Das2026`. The electromagnetic torque is

```{math}
---
label: torque
---
    \tauM = 3\,\mathrm{Im}\{\ish{1}\psish{1}^*\} + 6\,\mathrm{Im}\{\ish{2}\psish{2}^*\} + \mathrm{Im}\{\qM\fM^*\} - \frac{\partial W}{\partial\thetaM}
```

where the last term is due to the spatial harmonics.

## Magnetic Models

The magnetic model is selected via the machine parameters, which correspond to those of *motulator*:

- {class}`motulator_bearingless.utils.BearinglessSynchronousMachinePars` holds the constant parameters of the linear model {cite}`Das2026`, {cite}`Sok2019`. The inductances depend linearly on the displacement via the force constants. The unbalanced magnetic pull, which is a second-order effect of the displacement on the inductances, is not included.
- {class}`motulator_bearingless.utils.SaturatedBearinglessSynchronousMachinePars` holds a current map or a flux map, which include magnetic saturation, the cross-saturation, and the unbalanced magnetic pull.
- {class}`motulator_bearingless.utils.SpatialSaturatedBearinglessSynchronousMachinePars` holds a magnetic map, which also includes the spatial harmonics.

The maps can be gradient networks, see {doc}`gradnet`, or any other function with the same signature.

## Radial Dynamics

The rotor is either held at a given radial position by external bearings, see {class}`motulator_bearingless.model.ExternalRotorPosition`, or it levitates, see {class}`motulator_bearingless.model.RadialMechanics`. In the latter case, the rotor is modeled as a point mass $m$ in stator coordinates

```{math}
---
label: radial_dynamics
---
    m\frac{\D\vM^\mathrm{s}}{\D t} = \fM^\mathrm{s} + \mathbf{f}_\mathrm{ext}^\mathrm{s} - c\vM^\mathrm{s} \qquad
    \frac{\D\qM^\mathrm{s}}{\D t} = \vM^\mathrm{s}
```

where $\mathbf{f}_\mathrm{ext}^\mathrm{s}$ is the external force (e.g., gravity) and $c$ is the viscous damping coefficient. The tilting and axial motion are not modeled. The rotational dynamics are modeled by the mechanical system models of *motulator*.
