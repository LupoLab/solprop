`Simulation.run_request()` returns the run request translation built beside the
problem, translating once, as `Simulation.translate()` does for the problem
([#619](https://github.com/LupoLab/solprop/issues/619)). The problem and the request are
what `solprop.backends.registry.select`, `solprop.translate.problem.validate_run`,
`PropagationProblem.validate_launch` and `solprop.driver.run.field_retention` take when
the outer loop is run by hand; the documented examples no longer reach for the private
`Simulation._run_request()`, which is gone.
