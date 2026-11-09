"""
Six-phase bearingless synchronous machine drives.

This package provides the continuous-time model of a symmetrical six-phase bearingless
synchronous machine drive, a direct radial force controller based on feedback
linearization [#Das2026]_, a current-based force controller for comparison, a
levitation controller, and the related utilities. The magnetic model is either linear
or based on the gradient networks of :mod:`motulator_bearingless.gradnet`, trained on
finite-element data. The six-phase converter model comes from
`motulator_multiphase`, and the mechanical system models, the vector controllers, and
the simulation class from `motulator`. The package is organized like
`motulator.drive`::

    import motulator_bearingless.gradnet as gn
    from motulator_bearingless import control, model, utils

References
----------
.. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase
   bearingless synchronous machines," Proc. ICEM, 2026,
   https://doi.org/10.1109/ICEM68283.2026.11757980

"""
