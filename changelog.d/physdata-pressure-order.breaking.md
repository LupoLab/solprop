The gas-property functions take one argument order, `(gas, pressure, temperature)`,
with the pressure in bar named `pressure` and the temperature defaulting to 293.15 K
everywhere. `density`, `ideal_gas_density`, the equations of state's `density`,
`ReferenceState`, `gasflow.mass_flow` and `gasflow.knudsen_number` rename `pressure_bar`
to `pressure` (write `ReferenceState(pressure=1.0, temperature=273.15)`).
`physdata.transport.sound_speed(gas, temperature, pressure_bar=None)` becomes
`sound_speed(gas, pressure=None, temperature=293.15)`: replace
`sound_speed(gas, T)` with `sound_speed(gas, temperature=T)` and
`sound_speed(gas, T, p)` with `sound_speed(gas, p, T)`. `viscosity`,
`heat_capacity_ratio` and `mean_free_path` now default the temperature to 293.15 K. A gas law's
recorded reference state names `pressure`, so a run's configuration hash differs from
one written before.
