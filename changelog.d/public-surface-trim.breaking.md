The packages re-export only what a user calls. `solprop.modes` keeps the modes, guides,
collections, materials and the mode protocol, and now also exports the derived
quantities of `solprop.modes.base` (`beta`, `beta1`, `beta2`, `beta3`, `alpha`, `zdw`,
`aeff`, `loss_length`, `db_per_m`, `transmission`); its numerical machinery (the wall
families and limits, connections, quadrature rules, `GasFillMixin`, tolerances) is
imported from its submodule, for example
`from solprop.modes.wall import worst_wall_limits`. `solprop.output` no longer
re-exports `PlaneStore`, `PlaneLayout`, `CheckpointOutput`, `RetainingOutput`,
`TerminalCheckpoint`, `StatedInput`, `serialise`, `resolve_metadata_write`,
`package_versions` or the layout and convention constants; `solprop.driver` no longer
re-exports `metadata_for` or `plane_from_observation` (import the latter from
`solprop.driver.run`), and neither it nor `solprop.driver.run` re-exports `LogReporter`
or `ProgressReporter`: import the reporters from `solprop.log`. `solprop.nonlinear`
exports `OptmatRate`.
