Examples
========

These examples demonstrate radial force control of six-phase bearingless synchronous machine drives. The two three-phase sets of the symmetrical six-phase winding produce both the torque and the radial force. In direct force control [#Das2026]_, the torque and the force are controlled directly by means of feedback linearization. For comparison, in current-based force control, the four-pole plane is controlled by a vector controller of motulator, and the force reference is converted to the current reference of the two-pole plane. In both cases, the rotor displacement can be controlled by an outer levitation controller. The magnetic model of the machine and of the controller is either linear or a gradient network trained on finite-element data [#Li2026]_, with or without the spatial harmonics. The examples with the gradient networks load the networks and the dataset from the folders ``trained_models`` and ``datasets`` of ``examples/``, so they are run in a clone of the repository.

.. rubric:: References

.. [#Das2026] Das, Martin, Kontodinas, Hinkkanen, "Direct force control of six-phase bearingless synchronous machines," Proc. ICEM, 2026, https://doi.org/10.1109/ICEM68283.2026.11757980

.. [#Li2026] Li, Foissner, Martin, Piippo, Hinkkanen, "Gradient networks for universal magnetic modeling of synchronous machines," 2026, https://arxiv.org/abs/2602.14947
