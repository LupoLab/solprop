A backend (`solprop.backends`, unstable within 0.x under ADR-058) is built for one
problem and request and runs it in a session of its own; 0.2.0's direct stepping
interface and Python step loop are gone
([SEP 47](https://github.com/LupoLab/solprop/blob/main/docs/dev/seps/archive/sep-0047-run-ownership.md)).
`Simulation.run`, `SimulationRunner`, scans and the `prop_*` functions do all of this
themselves, with unchanged interfaces.
`solprop.backends.registry.select(name, problem, request, ...)` and the backend
constructors take the run's request after the problem, check the pair before they
allocate and expose it as `Backend.request`; execution choices such as the node budget
arrive as `solprop.backends.session.ExecutionOptions` (`options=`). A run starts with
`backend.prepare(problem, request).start(start)`, or
`solprop.backends.sessions.start_session`, from a `solprop.backends.session.Start`
(`Start.entrance(problem)` for a fresh run), which also carries a resumed run's call
ledgers and Adams history; `solprop.driver.run.run_problem` takes a required `start=` in
place of `start_plane=`, and a caller no longer calls `set_initial` first. The `Backend`
protocol loses its stepping members (`set_initial`, `propagate_to`, `observe`,
`march_snapshot`, `restore_march` and the rest; `available_device_memory` is the module
function `solprop_native.available_device_memory()`), and `timed_step()` is replaced by
`Backend.timing_probe(start)`, whose `trial(z, dz)` takes one trial step that commits
nothing and whose `field()` reads that trial's end state. Removed with the loop:
`solprop.backends.driver`, `solprop.backends.multistep` (`CallLedger` is now
`solprop.backends.ledger.CallLedger`), `solprop.backends.multistep_general`,
`solprop.backends.low_storage.LowStorageMarch` (the NumPy reference's own marches are in
`solprop.backends.numpy_ref.marches`), `solprop.backends.numpy_ref.admission` and
`solprop.backends.exponential.refuse_formulation` (its one Adams refusal is
`refuse_adams`). `reached` and `replay_tolerance` are now in `solprop.stepper.schedule`,
and the general route's node chunk in `solprop.backends.node_chunk`
(`general_node_chunk`).
