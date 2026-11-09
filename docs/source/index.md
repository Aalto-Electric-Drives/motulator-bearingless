# *motulator-bearingless:* Six-Phase Bearingless Machine Drives for *motulator*

This package extends [*motulator*](https://aalto-electric-drives.github.io/motulator/) with models and control of symmetrical six-phase bearingless synchronous machines, whose two three-phase sets produce both the torque and the radial force of the rotor. The machine model uses the vector-space decomposition {cite}`Ahm2023`. The magnetic model is either linear or a gradient network {cite}`Li2026`, trained on finite-element data, which represents magnetic saturation, the rotor displacement, and the spatial harmonics. The radial force is controlled by a direct force controller based on feedback linearization {cite}`Das2026` or, for comparison, by a current-based force controller. The rotor displacement can be controlled by an outer levitation controller.

This documentation covers only the extension. The conventions, the simulation environment, and the common models and control methods are those of [*motulator*](https://aalto-electric-drives.github.io/motulator/), and the six-phase converter model comes from [*motulator-multiphase*](https://aalto-electric-drives.github.io/motulator-multiphase/). The names of the classes follow *motulator*, see {doc}`usage`.

```{toctree}
---
titlesonly:
caption: Getting Started
name: getting_started
maxdepth: 1
---
installation
usage
```

```{toctree}
---
titlesonly:
caption: Examples
name: examples
maxdepth: 1
---
examples/index
```

```{toctree}
---
titlesonly:
caption: Theory
name: theory
maxdepth: 1
---
model
gradnet
control
relation
```

```{toctree}
---
titlesonly:
caption: Data
name: data
maxdepth: 1
---
fem
```

```{toctree}
---
titlesonly:
caption: Reference
name: reference
maxdepth: 1
---
API <autoapi/motulator_bearingless/index>
bibliography
motulator <https://aalto-electric-drives.github.io/motulator/>
motulator-multiphase <https://aalto-electric-drives.github.io/motulator-multiphase/>
```

```{rubric} Acknowledgment
```

This project has been sponsored by the Research Council of Finland through the *Centre of Excellence in High-Speed Electromechanical Energy Conversion Systems* and the Academy Project 355651.
