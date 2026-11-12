`solprop.driver.simulation` now holds the `Simulation` class alone; the rest moved
into modules by concern. Import `SimulationSettings` and `default_collar` from
`solprop.driver.settings` (or `SimulationSettings` from `solprop.driver`),
`SimulationRunner` and `check_derived_problem` from `solprop.driver.runner`,
`metadata_for` from `solprop.driver.metadata`, `default_field_range` from
`solprop.driver.inputs` and `default_radial_order` from
`solprop.driver.routes.hollow`; the names `solprop.driver` exports are unchanged.
