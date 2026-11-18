The `polarisation` keyword of `prop_capillary`, `prop_waveguide`,
`Simulation.from_capillary` and `Simulation.from_waveguide`, and of their recipes, is
removed: it was refused whatever its value, because these routes are scalar. Declare the
polarisation in the modes (a rectangular or slab guide's own partners, each a row of its
own). Passing it now raises `TypeError`, and a recipe holding it is refused as an unknown
keyword.
