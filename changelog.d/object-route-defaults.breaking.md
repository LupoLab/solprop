`prop_free`, `prop_planar`, `Simulation.from_free` and `Simulation.from_planar` take
the defaults the keyword routes take. `SimulationSettings.dz_max` and `dz_init` are
optional: left `None`, a simulation takes half its `grid.length` and a ten-thousandth of
it, the first refined by the start-up rule, as `prop_gnlse`, `prop_capillary` and
`prop_waveguide` do. Settings whose `stats` is `None` now record the route's default
statistics on the free-space and planar routes (`default_spec("radial")` on a radial
grid, `default_spec("free")` otherwise, with a mixture's per-species statistics), as
every modal route already did, rather than none; pass `StatsSpec(names=())` to record
none. The manual no longer says that `rtol`, `atol`, `dz_init` and `dz_max` are
keywords of every `prop_*` function: the free-space and planar ones take them in
their settings.
