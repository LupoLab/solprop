The problem descriptor between translation and the backends (`solprop.translate`,
unstable within 0.x under ADR-058) describes one trajectory and leaves the run's policy
to a separate request
([SEP 47](https://github.com/LupoLab/solprop/blob/main/docs/dev/seps/archive/sep-0047-run-ownership.md),
[SEP 52](https://github.com/LupoLab/solprop/blob/main/docs/dev/seps/archive/sep-0052-backend-shape.md)).
`solprop.translate.problem.KernelProblem` is renamed `PropagationProblem`, with no alias.
Its arrays carry no leading batch axis: the problem's `shape`, `initial.field0` and
`initial.noise0`, the operator's value tables (`LinearOperator.phi_coefficients` is
`(n_intervals, n_deg, n_modes, n_pol, n_omega)`), `StageTerms.phi`,
`Observation.field` and `noise_phase`, the periodic rows of `StepStats`,
`ResponseBudgetError.field`, `RunEstimate.shape` and the fields that
`BoundaryMonitor.power` and `SplitCrossing.occupancy` take lose it, so code that indexed
`[0]` drops the index.
`Initial.batch_params`, `Flags.single_batch` and `assemble(batch_params=...)` are
removed, `assemble` refuses a stacked field, and `plane_from_observation` no longer takes
`member=`. The route-specific fields `transform`, `flux`, `transverse`, `tdse` and
`current_numerator` move into a closed union, `problem.route`, from the new module
`solprop.translate.routes` (a `ModalRoute`, a `GeneralRoute` or a `SpatialRoute`);
`problem.transform` still reads the route's transform, and `tdse_of`, `flux_of` and
`transverse_of` read a route's own leaves. The problem no longer has `stepper` or
`stats_spec`: the tableau, controller, step limits, schedule, save planes and statistics
travel in the immutable `solprop.translate.run_request.RunRequest`, which
`Simulation.run_request()` returns. `assemble` returns a `Translation(problem, request)`
pair (`problem, request = assemble(...)`); `PropagationProblem.validate()` checks the
problem alone, the new `solprop.translate.problem.validate_run(problem, request)` adds
the checks that read the run's policy, and `validate_launch`, `metadata_for`,
`field_retention`, `check_snapshot_request`, `run_summary` and `check_derived_problem`
take the request. The general modal route's and the coupled flow's admission is part of
that validation (`solprop.translate.admission.check_general_route`), so a hand-built
problem whose route, polarisation count, vector Raman terms or TDSE statistics a backend
refused with `BackendError` is now refused with `TranslationError`. The general route's
derivations moved from `solprop.backends.numpy_ref` to `solprop.translate.general`
(`node_kerr_tensor`, `node_kerr_form`, `node_delayed_maps`) and `QUADRATURE_ALARM` to
`solprop.translate.modal`; `CompiledResponses.response_order` is derived from the
compiled leaves and is no longer a constructor argument; `MARCH_SNAPSHOT_AUTO_BYTES`
moved from `solprop.driver.run` to `solprop.translate.run_request`, and `MAX_N_POL` is
`solprop.nonlinear.responses.MAX_N_POL` alone. `Simulation`, `SimulationSettings` and
the `prop_*` functions take the same arguments as before.
