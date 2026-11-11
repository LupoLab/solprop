The native backend no longer steps the low-storage march (SEP 24) through its legacy
direct API: `propagate_to` on a native low-storage run raises a `BackendError` that names
the session, which replays the march natively
([SEP 47](docs/dev/seps/sep-0047-run-ownership.md) section 6.8). Start such a run with
`backend.prepare(problem, request).start(start)` or
`solprop.backends.sessions.start_session`, as `Simulation.run` already does. A native
timing probe's low-storage steps are now counted in the backend's `source_evaluations`,
four a step, where they counted none.
