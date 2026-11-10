The backend seam's problem descriptor is now one trajectory's description
([SEP 47](docs/dev/seps/sep-0047-run-ownership.md) sections 6.2 and 6.7). The nominal
leading batch axis is gone: `KernelProblem.shape`, `initial.field0` and
`initial.noise0`, `Backend.set_initial`, `Observation.field` and `noise_phase`,
`Backend.stats()`, the periodic rows of `StepStats` and `ResponseBudgetError.field` are
all `(n_modes, n_pol, n_omega)`-shaped with no batch axis, `Initial.batch_params`,
`Flags.single_batch` and `assemble(batch_params=...)` are removed, `assemble` refuses a
stacked field, and `RunEstimate.shape` has three entries. A group of trajectories is a
sequence of member problems, as a batched scan already runs them;
`solprop.driver.batch.split_observation` and `plane_from_observation(member=...)` are
removed. A march snapshot's stored state keeps its leading axis of one on disk, and
existing files stay readable. The public `Simulation` and `SimulationSettings`
interfaces are unchanged, and every result is bitwise the same.
