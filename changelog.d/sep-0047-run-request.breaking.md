The run's policy has left the backend seam's problem descriptor
([SEP 47](docs/dev/seps/sep-0047-run-ownership.md) sections 5.2 and 6.2).
`PropagationProblem` no longer has `stepper` or `stats_spec`: the tableau, controller,
step limits, schedule, save planes and statistics travel in the immutable
`solprop.translate.run_request.RunRequest`. `solprop.translate.problem.assemble` returns
a `Translation(problem, request)` pair (`problem, request = assemble(...)`), whose
request has the route's step limits resolved, and `RunRequest.stepper` reads the stepper
settings as one `StepperSettings` record. Backends receive the request where they are
built: `solprop.backends.registry.select(name, problem, request, ...)` and the backend
constructors take it after the problem, check the pair with the new
`solprop.translate.problem.validate_run(problem, request)` before they allocate, and
expose it as `Backend.request`. A run starts through the backend's preparation:
`Backend.prepare(problem, request)` returns a `PreparedExecution` whose `start(...)`
opens the backend's own session, the reference's `HostSession`, the native backend's
`NativeSession` where it admits the run and the transitional session otherwise, and
`solprop.backends.sessions.start_session` only starts it; a preparation refuses a
request that is not one run of the backend's own (`RunRequest.for_run` of it).
`PropagationProblem.validate()` now checks what the problem
promises alone; `validate_run` adds the checks that read the run's policy, with the same
refusals and messages, and `validate_launch(request)` takes the request.
`describe_run` now takes a `StepperSettings`; `RunRequest.for_run(...)` gives one run's
start, retention, monitor, history and snapshot policy, and `RunRequest.with_stepper(...)`
replaces the stepper section. `metadata_for(problem, request, grid, ...)`,
`field_retention(request, ...)`, `check_snapshot_request(request, ...)`,
`run_summary(problem, request, metadata, ...)` and
`check_derived_problem(constructed, derived, constructed_request, derived_request)` take
the request; `run_problem` reads it from the backend. The public `Simulation` and
`SimulationSettings` interfaces are unchanged, configuration hashes and resume identity
are unchanged (a file written before this change resumes), and every result is bitwise
the same.
