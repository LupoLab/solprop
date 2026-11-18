One loss vocabulary for every mode. `loss=` is removed, with the `LossSpec` alias, from
the hollow modes (`MarcatiliMode`, `RectMode`, `SlabMode`, `VincettiMode`, the tabulated
hollow modes), the `ModeCollection` factories, `prop_capillary` and
`Simulation.from_capillary`, `capillary_params` and `capillary_design`, and recipes.
`loss_scale` multiplies a hollow mode's own loss (default `1.0`; `0.0` is lossless) and
`alpha` replaces it by an absolute power attenuation in 1/m, a number or a callable
`alpha(omega)`; giving `alpha` beside a `loss_scale` other than one is refused.
`StepIndexMode` and `SpecifiedMode` (and its `from_taylor`, `from_taylor_path` and
`from_table`), whose `loss=` was already an absolute attenuation, take it as `alpha=`.
Migrate `loss=False` to `loss_scale=0.0`, `loss=0.5` to `loss_scale=0.5`, a callable
`loss=f` to `alpha=f`, drop `loss=True`, and write `loss_scale = 0.0` in a recipe for
`loss = false`. `SpecifiedMode.alpha(omega, z)` is now
`SpecifiedMode.attenuation(omega, z)` and `loss_path` is `alpha_path`, because `alpha`
is the declared field.
