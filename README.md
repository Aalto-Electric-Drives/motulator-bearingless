# motulator-bearingless

**Six-phase bearingless machine drives for [motulator](https://github.com/Aalto-Electric-Drives/motulator)**

Documentation: https://aalto-electric-drives.github.io/motulator-bearingless/

This package extends *motulator* with the continuous-time model of a symmetrical six-phase bearingless synchronous machine drive, a direct radial force controller based on feedback linearization, and a levitation controller [[1]](#references). The two three-phase sets of the winding produce both the torque and the radial force. The winding is described by means of the vector-space decomposition, and the magnetic model is a gradient of a magnetic energy function, which represents magnetic saturation and rotor eccentricity in a consistent manner. The six-phase converter model comes from [motulator-multiphase](https://github.com/Aalto-Electric-Drives/motulator-multiphase).

The magnetic models follow *motulator*: constant parameters, or gradient networks [[2]](#references) trained on the [finite-element data](https://aalto-electric-drives.github.io/motulator-bearingless/fem.html) of a 2.6-kW machine. The machine model uses an energy-based current map, with or without the spatial harmonics, and the controller a co-energy-based flux map without the rotor angle.

For comparison, a current-based force controller is included as well: the four-pole plane is controlled by a vector controller of *motulator*, and the force reference is converted to the current reference of the two-pole plane, in the manner of the common and differential modes of *motulator-multiphase*, see the [control methods](https://aalto-electric-drives.github.io/motulator-bearingless/control.html) and the [relation between the packages](https://aalto-electric-drives.github.io/motulator-bearingless/relation.html).

## Installation

Install the package in a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) (*motulator* and *motulator-multiphase* are installed as dependencies). On Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install motulator-bearingless
```

On Windows, the environment is created and activated with `py -m venv .venv` and `.venv\Scripts\activate`.

For development, and for the examples with the gradient networks, which load the networks and the dataset from `examples/`, clone the repository and install the package with `pip install -e ".[dev]"`.

## Usage

A sensorless direct-force-controlled 2.6-kW bearingless SyRM drive, whose rotor is held at a fixed radial position by external bearings:

```python
from math import pi

from motulator_bearingless import control, model, utils

# System model: L_1 is the two-pole inductance, and m_d and m_q are the radial force
# constants of the d and q axes
par = model.BearinglessSynchronousMachinePars(
    R_s=0.215, L_1=0.03, L_d=0.03, L_q=0.01, m_d=10.0, m_q=0.2
)
machine = model.BearinglessSynchronousMachine(par)
w_M_rated = 2 * pi * 1700 / 60
k = 14.5 / w_M_rated**2  # Quadratic load torque, 14.5 Nm (rated) at the rated speed
mechanics = model.MechanicalSystem(J=0.05, B_L=lambda w_M: k * abs(w_M))
radial = model.ExternalRotorPosition()  # Fixed radial position
converter = model.MultiThreePhaseConverter(u_dc=540)  # Two sets by default
mdl = model.BearinglessDrive(machine, mechanics, radial, converter)

# Control system: the direct force controller and the speed controller
cfg = control.DirectForceControllerCfg(i_s_max=25, psi_s_min=0.3, sensorless=True)
force_ctrl = control.DirectForceController(par, cfg)
speed_ctrl = control.SpeedController(J=0.05, alpha_s=2 * pi * 4)
ctrl = control.DirectForceControlSystem(force_ctrl, speed_ctrl)

# Speed and force references
ctrl.set_speed_ref(lambda t: min(max(t - 0.2, 0) / 0.6, 1) * w_M_rated)
ctrl.set_force_ref(lambda t: (t > 0.4) * 100 + 1j * (t > 0.6) * 150)

res = model.Simulation(mdl, ctrl).simulate(t_stop=1.2)
utils.plot(res)
```

The gradient-network maps replace the constant parameters in the same manner as in *motulator*:

```python
import motulator_bearingless.gradnet as gn

models = "examples/trained_models/"
current_map = gn.CurrentMapWithHarmonics(
    gn.load_gradnet(models + "bsyrm_fem_curr_map_harm_softmax_d128.npz")
)
par = model.SpatialSaturatedBearinglessSynchronousMachinePars(0.215, current_map)
machine = model.BearinglessSynchronousMachine(par)

flux_map = gn.FluxMap(gn.load_gradnet(models + "bsyrm_fem_flux_map_softmax_d128.npz"))
est_par = control.SaturatedBearinglessSynchronousMachinePars(0.215, psi_s_fcn=flux_map)
force_ctrl = control.DirectForceController(est_par, cfg)
```

Complete examples, including current-based force control, levitation control, and the magnetic maps, are in the [example gallery](https://aalto-electric-drives.github.io/motulator-bearingless/examples/index.html).

## Scope

- Symmetrical six-phase machines with two identical three-phase sets, without zero-sequence currents.
- The radial motion of the rotor is modeled as a point mass, or the rotor is held at a given radial position. The tilting and axial motion are not modeled.
- The constant machine parameters of the examples are illustrative.
- The trained networks describe a 2.6-kW bearingless synchronous reluctance machine, based on a finite-element model that has not been validated against measurements.
- The control methods do not use the spatial harmonics.

## References

1. Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase bearingless synchronous machines," Proc. ICEM, 2026, https://doi.org/10.1109/ICEM68283.2026.11757980
1. Li, Foissner, Martin, Piippo, Hinkkanen, "Gradient networks for universal magnetic modeling of synchronous machines," 2026, https://arxiv.org/abs/2602.14947

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->

<!-- prettier-ignore-start -->

<!-- markdownlint-disable -->

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nirangkush-das"><img src="https://avatars.githubusercontent.com/u/206182092?v=4?s=50" width="50px;" alt="Nirangkush Das"/><br /><sub><b>Nirangkush Das</b></sub></a><br /><a href="#ideas-nirangkush-das" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-nirangkush-das" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lefterisg4"><img src="https://avatars.githubusercontent.com/u/43557907?v=4?s=50" width="50px;" alt="Eleftherios Kontodinas"/><br /><sub><b>Eleftherios Kontodinas</b></sub></a><br /><a href="#ideas-lefterisg4" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-lefterisg4" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/FloranMartin"><img src="https://avatars.githubusercontent.com/u/98747419?v=4?s=50" width="50px;" alt="Floran Martin"/><br /><sub><b>Floran Martin</b></sub></a><br /><a href="#ideas-FloranMartin" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-FloranMartin" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jacopogertod"><img src="https://avatars.githubusercontent.com/u/214482243?v=4?s=50" width="50px;" alt="jacopogertod"/><br /><sub><b>jacopogertod</b></sub></a><br /><a href="#ideas-jacopogertod" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-jacopogertod" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Exiaolibur"><img src="https://avatars.githubusercontent.com/u/134478666?v=4?s=50" width="50px;" alt="Junyi Li"/><br /><sub><b>Junyi Li</b></sub></a><br /><a href="#ideas-Exiaolibur" title="Ideas, Planning, & Feedback">🤔</a> <a href="#research-Exiaolibur" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://research.aalto.fi/en/persons/marko-hinkkanen"><img src="https://avatars.githubusercontent.com/u/76600872?v=4?s=50" width="50px;" alt="Marko Hinkkanen"/><br /><sub><b>Marko Hinkkanen</b></sub></a><br /><a href="https://github.com/Aalto-Electric-Drives/motulator-bearingless/commits?author=mhinkkan" title="Code">💻</a> <a href="#example-mhinkkan" title="Examples">💡</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->

<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Acknowledgment

This project has been sponsored by the Research Council of Finland through the *Centre of Excellence in High-Speed Electromechanical Energy Conversion Systems* and the Academy Project 355651.
