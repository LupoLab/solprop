`Simulation.from_radial` and `prop_radial` are removed: `Simulation.from_free` and
`prop_free` take a `RadialGrid` as well as a `FreeGrid` or `LineGrid`, and choose the
radial or the Cartesian transform from it, with the same keywords and the same
defaults as before for each grid. Replace `from_radial(...)` with `from_free(...)` and
`prop_radial(...)` with `prop_free(...)`; recipes keep the `radial` route.
