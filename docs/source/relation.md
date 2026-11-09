# Relation to *motulator-multiphase*

The package *motulator-multiphase* models and controls multi-three-phase machines by means of the decoupled multi-stator (DMS) transformation {cite}`Rub2021`: the three-phase sets are mapped to a common mode (CM), which is a three-phase machine, and to differential modes (DMs), which are resistive-inductive circuits. The package *motulator-bearingless* describes a symmetrical six-phase bearingless synchronous machine by means of the vector-space decomposition (VSD) {cite}`Ahm2023`: the winding is mapped to a two-pole plane and a four-pole plane, which are coupled by the rotor displacement. This document shows how the transformations and the models of the two packages are related. The control principle of the multi-three-phase drives, i.e., vector control of the main mode and current control of the other modes, is applied to the bearingless machine in the current-based force control, see {doc}`control`.

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

|                       | *motulator-multiphase*                                                           | *motulator-bearingless*                                                                                                                                                 |
| --------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transformation        | DMS, $\mathbf{x}_\mathrm{cm}^\mathrm{s}$ and $\mathbf{x}_\mathrm{dm}^\mathrm{s}$ | VSD, $\mathbf{x}_\mathrm{s1}^\mathrm{s} = \mathbf{x}_\mathrm{cm}^\mathrm{s}$ and $\mathbf{x}_\mathrm{s2}^\mathrm{s} = \left(\mathbf{x}_\mathrm{dm}^\mathrm{s}\right)^*$ |
| Torque-producing mode | CM, three-phase machine                                                          | Four-pole plane, synchronous reluctance machine                                                                                                                         |
| Other mode            | DM, $R_\mathrm{s}$ and $L_{\ell\mathrm{s}}$                                      | Two-pole plane, $R_\mathrm{s}$ and $L_1$                                                                                                                                |
| Coupling of the modes | None                                                                             | $\mathbf{L}_{21}(\mathbf{q}_\mathrm{M})$, produces the force                                                                                                            |
| Rotation of the modes | Same electrical angular speed                                                    | $2\omega_\mathrm{M}$ in both planes (steady-state force)                                                                                                                |
