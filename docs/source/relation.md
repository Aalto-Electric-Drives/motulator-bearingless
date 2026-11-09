# Relation to *motulator-multiphase*

The package *motulator-multiphase* models and controls multi-three-phase machines by means of the decoupled multi-stator (DMS) transformation {cite}`Rub2021`: the three-phase sets are mapped to a common mode (CM), which is a three-phase machine, and to differential modes (DMs), which are resistive-inductive circuits. The package *motulator-bearingless* describes a symmetrical six-phase bearingless synchronous machine by means of the vector-space decomposition (VSD) {cite}`Ahm2023`: the winding is mapped to a two-pole plane and a four-pole plane, which are coupled by the rotor displacement. This document shows how the transformations, the models, and the controls of the two packages are related, and when the control principle of the multi-three-phase drives, i.e., vector control of the main mode and current control of the other modes, can be applied to bearingless machines. The current-based force control described in the section {ref}`current-based-force-control` is implemented in *motulator-bearingless* alongside the direct force control of {cite}`Das2026`.

The notation is that of {doc}`model`. In addition, real vectors and matrices are denoted by bold symbols.

## Transformations

The winding has the phases $1,\ldots,6$ at $60^\circ$ intervals. The first three-phase set consists of the phases $1, 3, 5$ and the second set of the phases $2, 4, 6$. Each set is described by its standard space vector in its own coordinates,

```{math}
    \mathbf{x}_\mathrm{A} = \frac{2}{3}\left(x_1 + \alpha^2 x_3 + \alpha^4 x_5\right) \qquad \mathbf{x}_\mathrm{B} = \frac{2}{3}\left(x_2 + \alpha^2 x_4 + \alpha^4 x_6\right)
```

where $\alpha = \e^{\jj\pi/3}$. The a-phase axis of the second set is at the angle $\theta$ with respect to that of the first set; here $\theta = 60^\circ$ (symmetrical six-phase winding), while the default of *motulator-multiphase* is the asymmetrical arrangement $\theta = 30^\circ$ ({func}`motulator_multiphase.utils.default_set_angles`).

For two sets, the DMS transformation ({func}`motulator_multiphase.utils.dms_matrix`) of the set vectors expressed in common coordinates is

```{math}
    \mathbf{x}_\mathrm{cm}^\mathrm{s} = \frac{1}{2}\left(\mathbf{x}_\mathrm{A} + \e^{\jj\theta}\mathbf{x}_\mathrm{B}\right) \qquad \mathbf{x}_\mathrm{dm}^\mathrm{s} = \frac{1}{2}\left(\mathbf{x}_\mathrm{A} - \e^{\jj\theta}\mathbf{x}_\mathrm{B}\right)
```

The VSD with peak-value scaling maps the six phase quantities to the planes $h = 1, 2$ according to

```{math}
    \mathbf{x}_{\mathrm{s}h}^\mathrm{s} = \frac{1}{3}\sum_{k=1}^{6} x_k\alpha^{h(k-1)}
```

which in terms of the set vectors reads ({func}`motulator_bearingless.utils.vsd`)

```{math}
---
label: vsd_sets
---
    \mathbf{x}_\mathrm{s1}^\mathrm{s} = \frac{1}{2}\left(\mathbf{x}_\mathrm{A} + \alpha\mathbf{x}_\mathrm{B}\right) \qquad \mathbf{x}_\mathrm{s2}^\mathrm{s} = \frac{1}{2}\left(\mathbf{x}_\mathrm{A} - \alpha\mathbf{x}_\mathrm{B}\right)^*
```

Hence, for $\theta = 60^\circ$, the DMS and the VSD are the same transformation up to a complex conjugate,

```{math}
    \mathbf{x}_\mathrm{cm}^\mathrm{s} = \mathbf{x}_\mathrm{s1}^\mathrm{s} \qquad \mathbf{x}_\mathrm{dm}^\mathrm{s} = \left(\mathbf{x}_\mathrm{s2}^\mathrm{s}\right)^*
```

The conjugate results from the fact that the four-pole plane sees the phase sequence of each set reversed, since $2\cdot 120^\circ = 240^\circ = -120^\circ$. Since the conjugation does not commute with the rotation of the coordinates, the identity holds in stator coordinates. Equivalently, if the set vectors are conjugated and $\theta = 120^\circ$ (the set angle seen as an electrical angle in the four-pole plane) is used, the CM equals the four-pole plane and the DM the conjugated two-pole plane. Which plane is called the CM is thus a matter of naming; the physical difference between the packages is in the models attached to the modes. For $\theta = 30^\circ$, the DMS does not correspond to the VSD of {eq}`vsd_sets`, since the asymmetrical winding is a different winding.

## Machine Models

### Multi-Three-Phase Model

In *motulator-multiphase*, the CM is a three-phase machine whose sets are connected in parallel: the voltage and the flux linkage of the equivalent machine are those of the CM, its current is the sum of the set currents, and the resistances and inductances of a set are divided by the number of sets ({func}`motulator_multiphase.utils.to_parallel`). Each DM is a resistive-inductive circuit, $\D\boldsymbol{\uppsi}_\mathrm{dm}^\mathrm{s}/\D t = \mathbf{u}_\mathrm{dm}^\mathrm{s} - R_\mathrm{s}\boldsymbol{\uppsi}_\mathrm{dm}^\mathrm{s}/L_{\ell\mathrm{s}}$, which produces no torque and is not coupled to the CM. The modes are decoupled by assumption.

### Bearingless Machine Model

The four-pole rotor produces the torque with the four-pole plane ($h = 2$) and the radial force with the two-pole plane ($h = 1$). With linear magnetics, the flux linkages in rotor coordinates are $\boldsymbol{\uppsi}_\mathrm{s} = \mathbf{L}_\mathrm{s}(\mathbf{q}_\mathrm{M})\mathbf{i}_\mathrm{s}$, where $\mathbf{i}_\mathrm{s} = [i_\mathrm{s1d}, i_\mathrm{s1q}, i_\mathrm{s2d}, i_\mathrm{s2q}]^\T$, $\mathbf{q}_\mathrm{M} = [q_i, q_j]^\T$, and the inductance matrix is {cite}`Sok2019`, {cite}`Das2026`

```{math}
---
label: inductance_matrix
---
    \mathbf{L}_\mathrm{s} = \begin{bmatrix} L_1\mathbf{I} & \mathbf{L}_{21}^\T \\ \mathbf{L}_{21} & \operatorname{diag}(L_\mathrm{d}, L_\mathrm{q}) \end{bmatrix} \qquad \mathbf{L}_{21} = \begin{bmatrix} m_\mathrm{d} q_i & -m_\mathrm{d} q_j \\ m_\mathrm{q} q_j & m_\mathrm{q} q_i \end{bmatrix}
```

Here $L_1$ is the inductance of the two-pole plane, isotropic for the four-pole rotor, $L_\mathrm{d}$ and $L_\mathrm{q}$ are the inductances of the four-pole plane, and $m_\mathrm{d}$ and $m_\mathrm{q}$ are the radial force constants. The radial force is $\mathbf{f}_\mathrm{M} = \frac{3}{2}\mathbf{i}_\mathrm{s}^\T(\partial\mathbf{L}_\mathrm{s}/\partial\mathbf{q}_\mathrm{M})\mathbf{i}_\mathrm{s}$, which can be written in the complex form

```{math}
---
label: force_complex
---
    \mathbf{f}_\mathrm{M} = 3\mathbf{k}\,\mathbf{i}^*_\mathrm{s1} \qquad \mathbf{k} = m_\mathrm{d}i_\mathrm{s2d} + \jj m_\mathrm{q}i_\mathrm{s2q}
```

The force is bilinear in the currents of the two planes, and it does not depend explicitly on the displacement, since $\mathbf{L}_\mathrm{s}$ is linear in $\mathbf{q}_\mathrm{M}$. The torque is $\tau_\mathrm{M} = 3\operatorname{Im}\{\mathbf{i}_\mathrm{s1}\boldsymbol{\uppsi}^*_\mathrm{s1}\} + 6\operatorname{Im}\{\mathbf{i}_\mathrm{s2}\boldsymbol{\uppsi}^*_\mathrm{s2}\} + \operatorname{Im}\{\mathbf{q}_\mathrm{M}\mathbf{f}^*_\mathrm{M}\}$, and the voltage equations of the planes are

```{math}
    \frac{\D\boldsymbol{\uppsi}_\mathrm{s1}}{\D t} = \mathbf{u}_\mathrm{s1} - R_\mathrm{s}\mathbf{i}_\mathrm{s1} - \jj\omega_\mathrm{M}\boldsymbol{\uppsi}_\mathrm{s1} \qquad \frac{\D\boldsymbol{\uppsi}_\mathrm{s2}}{\D t} = \mathbf{u}_\mathrm{s2} - R_\mathrm{s}\mathbf{i}_\mathrm{s2} - \jj 2\omega_\mathrm{M}\boldsymbol{\uppsi}_\mathrm{s2}
```

A force that is constant in stator coordinates results from a two-pole field rotating at the electrical angular speed $2\omega_\mathrm{M}$ in stator coordinates, i.e., at the same speed as the four-pole field. Hence, the force-producing two-pole quantities rotate at $\omega_\mathrm{M}$ in their own rotor coordinates.

### Relation of the Models

For a centric rotor, $\mathbf{L}_{21} = \mathbf{0}$, and the model is block diagonal: the four-pole plane is a three-phase synchronous reluctance machine with two pole pairs, and the two-pole plane is a resistive-inductive circuit with the inductance $L_1$. In terms of the DMS model with $\theta = 60^\circ$, the four-pole plane is the (conjugated) DM and the two-pole plane the CM. The roles are thus opposite to the multi-three-phase model: the torque is produced in the DM, and the CM is a resistive-inductive circuit. The four-pole plane is exactly the parallel-connected equivalent machine of {func}`motulator_multiphase.utils.to_parallel` with $n_\mathrm{p} = 2$, the current of which is $2\mathbf{i}_\mathrm{s2}$, i.e., the sum of the set currents, and whose torque $6\operatorname{Im}\{\mathbf{i}_\mathrm{s2}\boldsymbol{\uppsi}^*_\mathrm{s2}\}$ is the total torque ({func}`motulator_bearingless.utils.fourpole_pars`).

The force arises solely from the displacement-dependent coupling $\mathbf{L}_{21}$, which the DMS model assumes to be zero. Therefore, the multi-three-phase model cannot represent a bearingless machine, and the bearingless model is a generalization of the DMS model in which the modes are coupled via the displacement. In *motulator-bearingless*, the magnetic model is the gradient of a magnetic energy function, so that the linear model {eq}`inductance_matrix` can be replaced with a saturated one without changing the control laws. The table below summarizes the correspondences.

|                           | *motulator-multiphase*                                                           | *motulator-bearingless*                                                                                                                                                 |
| ------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transformation            | DMS, $\mathbf{x}_\mathrm{cm}^\mathrm{s}$ and $\mathbf{x}_\mathrm{dm}^\mathrm{s}$ | VSD, $\mathbf{x}_\mathrm{s1}^\mathrm{s} = \mathbf{x}_\mathrm{cm}^\mathrm{s}$ and $\mathbf{x}_\mathrm{s2}^\mathrm{s} = \left(\mathbf{x}_\mathrm{dm}^\mathrm{s}\right)^*$ |
| Torque-producing mode     | CM, three-phase machine                                                          | Four-pole plane, synchronous reluctance machine                                                                                                                         |
| Other mode                | DM, $R_\mathrm{s}$ and $L_{\ell\mathrm{s}}$                                      | Two-pole plane, $R_\mathrm{s}$ and $L_1$                                                                                                                                |
| Coupling of the modes     | None                                                                             | $\mathbf{L}_{21}(\mathbf{q}_\mathrm{M})$, produces the force                                                                                                            |
| Rotation of the modes     | Same electrical angular speed                                                    | $2\omega_\mathrm{M}$ in both planes (steady-state force)                                                                                                                |
| Control of the main mode  | Vector controller of *motulator*                                                 | Flux-vector control, or vector controller of *motulator*                                                                                                                |
| Control of the other mode | Current control, references $c_\mathrm{dm}\mathbf{i}_\mathrm{cm}$                | Direct force control, or current control with references from {eq}`force_complex`                                                                                       |

## Force Control

### Direct Force Control

In direct force control {cite}`Das2026`, the force and the torque are controlled by means of feedback linearization, see {doc}`control`. The force control law of the two-pole plane includes the rotation term of the force and the displacement-induced term, while the effect of the four-pole input on the force is neglected, i.e., the planes are assumed to be decoupled in the control design. The flux linkages of both planes are estimated by a state observer in rotor coordinates, which also provides the sensorless operation from the four-pole plane. Any conservative magnetic model, e.g., a gradient network including saturation and eccentricity effects, can be used in the control law. In *motulator-bearingless*, this is {class}`motulator_bearingless.control.DirectForceControlSystem`.

(current-based-force-control)=

### Current-Based Force Control

The control principle of *motulator-multiphase* is to control the CM with a vector controller of a three-phase machine and the DMs with current controllers. Applied to the bearingless machine, this is the classical current-based force control {cite}`Chiba2005`: the four-pole plane is controlled by a vector controller of *motulator*, configured with the parameters of the equivalent machine, and the two-pole plane is current-controlled, with the force reference converted to the current reference.

Both planes rotate at $2\omega_\mathrm{M}$ in stator coordinates, so the natural common coordinates are those of the four-pole plane, $\vartheta_\mathrm{c} = 2\vartheta_\mathrm{M}$, which the vector controller provides (also in the sensorless mode). In these coordinates, the two-pole current $\mathbf{i}_\mathrm{s1}^\mathrm{c} = \e^{-\jj\vartheta_\mathrm{c}}\mathbf{i}_\mathrm{s1}^\mathrm{s} = \e^{-\jj\vartheta_\mathrm{M}}\mathbf{i}_\mathrm{s1}$ is constant in the steady state, and the force in stator coordinates follows from {eq}`force_complex` as

```{math}
---
label: force_controller
---
    \mathbf{f}_\mathrm{M}^\mathrm{s} = 3\mathbf{k}\,(\mathbf{i}_\mathrm{s1}^\mathrm{c})^*
```

where $\mathbf{k}$ depends on the four-pole current in rotor coordinates, available from the vector controller as half of the current of the equivalent machine. The current reference of the two-pole plane is obtained by inverting {eq}`force_controller`,

```{math}
    \mathbf{i}_\mathrm{s1,ref}^\mathrm{c} = \left(\frac{\mathbf{f}_\mathrm{M,ref}^\mathrm{s}}{3\mathbf{k}}\right)^*
```

and no force can be produced if the machine is not magnetized ($\mathbf{k} = 0$). The current is controlled by a complex-vector PI controller in the flux-linkage form, $\boldsymbol{\uppsi}_1 = L_1\mathbf{i}_\mathrm{s1}^\mathrm{c}$, in coordinates rotating at $\omega_\mathrm{c} = 2\omega_\mathrm{M}$ {cite}`Awa2019`, which is the controller of the DMs in *motulator-multiphase*. The force estimate {eq}`force_controller` computed from the measured currents is fed to the levitation controller, as in direct force control. Since both planes rotate at $\omega_\mathrm{c}$ in stator coordinates, the voltage references of both planes are advanced by the same angle $k_\mathrm{comp}T_\mathrm{s}\omega_\mathrm{c}$ before the modulation, as in the vector control systems of *motulator*.

Compared to {class}`motulator_multiphase.control.MultiThreePhaseControlSystem`, three changes are needed: the conjugate in {eq}`vsd_sets` (or, equivalently, the plane-specific rotation $h\vartheta_\mathrm{M}$), the force-to-current conversion in place of the current-sharing references $c_\mathrm{dm}\mathbf{i}_\mathrm{cm}$, and the current limit of the two-pole plane. In *motulator-bearingless*, this is implemented as {class}`motulator_bearingless.control.CurrentBasedForceControlSystem`, in which {class}`motulator_bearingless.control.CurrentBasedForceController` controls the two-pole plane. The four-pole plane is controlled by any synchronous-machine controller of *motulator*, e.g., {class}`motulator.drive.control.sm.CurrentVectorController` or {class}`motulator.drive.control.sm.FluxVectorController`, whose feedback signals are the voltage of the four-pole plane and twice its current. With linear magnetics, the direct force control law reduces to this current-based law {cite}`Das2026`; the methods differ in the transients and in how the model nonlinearities and the displacement enter the control.

## Comparison

Current-based force control (the principle of *motulator-multiphase*):

- Simple and modular: the vector controllers, observers, reference generators (MTPA, MTPV, field weakening), and current limits of *motulator* are used as such. The two-pole plane needs only the force constants $m_\mathrm{d}$ and $m_\mathrm{q}$, or a lookup table $\mathbf{k}(\mathbf{i}_\mathrm{s2}, \mathbf{q}_\mathrm{M})$, and the inductance $L_1$.
- The current loops are well understood, and the current of the two-pole plane is limited directly.
- The force is open loop with respect to the model: an error in $\mathbf{k}$ (force constants, saturation) appears directly as a force error, which only the outer levitation loop corrects.
- A torque transient changes $\mathbf{k}$. The current reference of the two-pole plane is updated algebraically, but the different dynamics of the two current loops produce a force transient.
- The terms induced by the displacement, i.e., the coupling flux $\mathbf{L}_{21}^\T\mathbf{i}_\mathrm{s2}$ in the two-pole plane, the rate of change of the displacement, and the rotation term $-\jj\omega_\mathrm{M}\mathbf{f}_\mathrm{M}$, are not compensated for but left to the current controller. With a static eccentricity and constant currents, the coupling flux is constant in the controller coordinates, $\mathbf{k}\,(\mathbf{q}_\mathrm{M}^\mathrm{s})^*$, so the integral action removes its effect in the steady state. The missing compensation shows in transients and with a dynamic eccentricity, e.g., the whirling caused by unbalance.
- With magnetic saturation, $\mathbf{k}$ becomes a $2\times 2$ matrix function of the four-pole current and the displacement. The same lookup-table data as in direct force control are then needed, and the simplicity advantage shrinks.

Direct force control:

- The force dynamics are linearized with respect to the two-pole input, including the rotation and displacement terms, so that, with an accurate model, the force bandwidth does not depend on the operating point. A transient of the four-pole plane disturbs the force via the neglected cross term {eq}`force_derivative`, but the force does not depend on the dynamics of a separate current loop.
- The magnetic model, a gradient of an energy function, covers saturation, cross-saturation, and eccentricity consistently for both the torque and the force.
- The disturbance observer provides the integral action without windup. The flux-based control of the four-pole plane allows the MTPV operation and field weakening in the same framework and provides the sensorless operation; the reference generation of the package uses the MTPA and MTPV loci with a minimum flux and the voltage limit.
- More complex: the $4\times 4$ incremental inductance matrix, the sensitivities $\partial\mathbf{i}_\mathrm{s}/\partial\mathbf{q}_\mathrm{M}$ and $\partial\mathbf{f}_\mathrm{M}/\partial\mathbf{q}_\mathrm{M}$, and matrix solutions are needed at every sampling instant.
- The force is not measured, so direct force control is open loop with respect to the errors of the force map as well; {cite}`Das2026` shows stable operation with the flux linkages of the control model altered by 10-20 %, but a steady-state force error remains for the levitation controller to correct. This limitation is shared by both methods.
- The current of the two-pole plane is not limited explicitly.
