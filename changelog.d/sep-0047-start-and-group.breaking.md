A run now starts from an explicit start rather than from a backend its caller loaded
([SEP 47](docs/dev/seps/archive/sep-0047-run-ownership.md) section 5.2):
`solprop.driver.run.run_problem` takes a required `start=`, a
`solprop.backends.session.Start` (`Start.entrance(problem)` for a fresh run), and loads
it into the backend through the run's session; its `start_plane=` argument is removed,
the start's `plane_index` taking its place. A caller no longer calls
`backend.set_initial` before `run_problem`. `Simulation.run`, `SimulationRunner`, scans
and the benchmark suite start their runs this way, and a resumed run's call ledgers
and Adams history travel in the start, a history the backend cannot install being
warned about and recorded as reset as before.
