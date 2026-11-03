Kernel schema 43: under `absorber_scheme="split_step"` a backend's boundary reading also
returns the energy and the guards' and collar's energies of the power each step's split
found (`solprop.monitor.BoundaryEnergies.found`), and the native reading grows from six
numbers to ten. A solprop-native build of an earlier schema is refused at import; rebuild
it from this tree.
