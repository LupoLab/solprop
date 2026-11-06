Kernel schema 43: under `absorber_scheme="split_step"` a backend's boundary reading also
returns the energy and the guards' and collar's energies of the power each step's split
found (`solprop.monitor.BoundaryEnergies.found`) and the state's energy per walk-off
bucket (`BoundaryEnergies.occupancy`), from the problem's new `split_crossing`; the
native reading grows from six numbers to ten plus the buckets, and the native Adams march
takes the crossing cap per call. Rebuild solprop-native from this tree.
