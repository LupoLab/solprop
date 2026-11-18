A gas fill no longer takes a pressure it was not given: the same omitted argument meant
0 bar on the modes and 1 bar on the routes and `SpatialMedium`. Wherever a gas is named,
its pressure must now be passed, in bar, or the call is refused with "a gas fill needs
its pressure" naming `pressure`: on the hollow modes, the `ModeCollection` factories,
`SpatialMedium` and the capillary and waveguide routes (for a built-in gas, `"air"` or a
mapping of fractions). Where no gas is named nothing changes: a mode without a gas is
evacuated, `SpatialMedium()` is vacuum, and a `Mixture` carries its own pressure. To
migrate, add the pressure the call used to take by default:
`MarcatiliMode(125e-6, "He", 0.0)` for a mode or mode set (or drop the gas, which is
what 0 bar meant), and `prop_capillary(r, L, "He", 1.0)` or `SpatialMedium("Ar", 1.0)`
for a route or a medium. A recipe's route no longer resolves an omitted `pressure` to
one bar.
