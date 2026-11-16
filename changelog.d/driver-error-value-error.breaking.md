`DriverError` is now a `ValueError` rather than a `RuntimeError`, as `GridError`,
`PulseError` and `NoiseError` are: nearly every one is a refused argument, so
`except ValueError` now catches a bad `n_save` as it catches a bad grid. Code that caught
a `DriverError` as a `RuntimeError` should catch `DriverError` (or `SolpropError`).
`Simulation.run` and the `prop_*` functions refuse a `resume` that is not `False`,
`True` or `"interpolated"` with a `DriverError` naming `resume`, rather than a bare
`ValueError`.
