A gas fill no longer takes a pressure it was not given: the same omitted argument meant
0 bar on the modes and 1 bar on the routes and `SpatialMedium`. Wherever a gas is named,
its pressure must now be passed, in bar, or the call is refused with "a gas fill needs
its pressure" naming `pressure`: on `MarcatiliMode`, `VincettiMode`,
`TabulatedHollowMode`, `RectMode` and `SlabMode` (a `ModeError`), on the
`ModeCollection` factories `he1m`, `marcatili`, `vincetti` and `slab`, on
`SpatialMedium` (a `MaterialError`), and on `prop_capillary`, `prop_waveguide`,
`Simulation.from_capillary` and `Simulation.from_waveguide` for a built-in gas, `"air"`
or a mapping of fractions (a `DriverError`). Where no gas is named nothing changes: a
mode without a gas is evacuated, `SpatialMedium()` is vacuum, and a `Mixture` carries its
own pressure. To migrate, add the pressure the call used to take by default:
`MarcatiliMode(125e-6, "He", 0.0)` for a mode or mode set (or drop the gas, which is
what 0 bar meant), and `prop_capillary(r, L, "He", 1.0)` or `SpatialMedium("Ar", 1.0)`
for a route or a medium. A recipe's route no longer resolves an omitted `pressure` to
one bar.
