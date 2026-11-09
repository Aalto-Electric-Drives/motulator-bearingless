# Usage

After {doc}`installation`, the package is used in the same manner as *motulator*: a continuous-time system model and a discrete-time control system are connected in a simulation object, see the [usage of *motulator*](https://aalto-electric-drives.github.io/motulator/usage.html). The modules are imported as

```python
from motulator_bearingless import control, model, utils
import motulator_bearingless.gradnet as gn
```

## Quick Start

```{include} ../../README.md
---
start-after: '## Usage'
end-before: '## Scope'
---
```

## Correspondence to *motulator*

The classes correspond to those of *motulator* as follows.

| *motulator*                                                           | *motulator-bearingless*                                                                |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| {class}`motulator.drive.model.SynchronousMachine`                     | {class}`motulator_bearingless.model.BearinglessSynchronousMachine`                     |
| {class}`motulator.drive.model.SynchronousMachinePars`                 | {class}`motulator_bearingless.utils.BearinglessSynchronousMachinePars`                 |
| {class}`motulator.drive.model.SaturatedSynchronousMachinePars`        | {class}`motulator_bearingless.utils.SaturatedBearinglessSynchronousMachinePars`        |
| {class}`motulator.drive.model.SpatialSaturatedSynchronousMachinePars` | {class}`motulator_bearingless.utils.SpatialSaturatedBearinglessSynchronousMachinePars` |
| {func}`motulator.drive.gradnet.load_gradnet`                          | {func}`motulator_bearingless.gradnet.load_gradnet`                                     |
| {class}`motulator.drive.gradnet.CurrentMap`                           | {class}`motulator_bearingless.gradnet.CurrentMap`                                      |
| {class}`motulator.drive.gradnet.CurrentMapWithHarmonics`              | {class}`motulator_bearingless.gradnet.CurrentMapWithHarmonics`                         |
| {class}`motulator.drive.gradnet.FluxMap`                              | {class}`motulator_bearingless.gradnet.FluxMap`                                         |
| {class}`motulator.drive.model.Drive`                                  | {class}`motulator_bearingless.model.BearinglessDrive`                                  |

The mechanical system models, the sensors, the simulation class, and the vector controllers of *motulator* are used as such, and they are re-exported by the modules of this package. The four-pole plane is described as an equivalent three-phase machine by {func}`motulator_bearingless.utils.fourpole_pars`, using {func}`motulator_multiphase.utils.to_parallel`, so that also the tools of *motulator* for that machine are available, e.g., {class}`motulator.drive.utils.MachineCharacteristics` and the control-loci plots.

## Simulation Results

The simulation results are organized as in *motulator*. In addition to the signals of the mechanical system and the converter, the most relevant signals are:

- `res.mdl.machine.i_c_ab` and `res.mdl.converter.u_c_ab` are the currents and voltages of the two three-phase sets in their own coordinates.
- `res.mdl.machine.i_s1_ab` and `res.mdl.machine.i_s2_ab` are the currents of the two-pole and four-pole planes in stator coordinates, and `res.mdl.machine.psi_s1_dq` and `res.mdl.machine.psi_s2_dq` the flux linkages in rotor coordinates.
- `res.mdl.machine.f_M_ab`, `res.mdl.machine.q_M_ab`, and `res.mdl.machine.tau_M` are the radial force, the rotor displacement, and the torque.
- `res.ctrl.ref` and `res.ctrl.fbk` contain the references and the feedback signals of the control system.

The function {func}`motulator_bearingless.utils.plot` plots the main signals.

## Scope

```{include} ../../README.md
---
start-after: '## Scope'
end-before: '## References'
---
```
