`Simulation`'s `grid`, `modes`, `responses`, `initial`, `initial_z`, `noise`,
`noise_resolution`, `profiles`, `configuration`, `spatial_grid`, `medium`, `frame`,
`general`, `guide`, `wall`, `taper_coupling` and `materials` are now read-only
properties, and `profiles` and `configuration` read-only mappings. Assigning one used
to leave the run built from the old value while the record described the new one
(`sim.initial = pulse` ran the old pulse); it now raises `AttributeError`. Derive a run
with a new input, noise or launch position with `with_input(...)`, one with a new fill
with `with_mixture(...)`, and build a new `Simulation` for anything else.
