A backend (`solprop.backends`, unstable within 0.x under ADR-058) is built for one
problem and request and runs it in a session of its own; 0.2.0's direct stepping
interface and Python step loop are gone
([SEP 47](https://github.com/LupoLab/solprop/blob/main/docs/dev/seps/archive/sep-0047-run-ownership.md)).
`solprop.backends.registry.select(name, problem, request, ...)` and the backend
constructors take the run's request after the problem, check the pair before they
allocate and expose it as `Backend.request`; execution choices such as the node budget
arrive as `solprop.backends.session.ExecutionOptions` (`options=`). A run starts with
`backend.prepare(problem, request).start(start)`, or
`solprop.backends.sessions.start_session`, from a `solprop.backends.session.Start`
(`Start.entrance(problem)` for a fresh run), which also carries a resumed run's call
ledgers and Adams history; `solprop.driver.run.run_problem` takes a required `start=` in
place of `start_plane=`, and a caller no longer calls `set_initial` first. The `Backend`
protocol no longer has `reads_transverse_edge`, `set_initial`,
`set_progress`, `set_stop`, `attach_monitor`, `propagate_to`, `observe`, `in_flight`,
`stats`, `restore_call_ledgers`, `march_snapshot`, `march_snapshot_nbytes`,
`march_export_refusal`, `restore_march` or `available_device_memory` (the module
function `solprop_native.available_device_memory()` stays), and `timed_step()` is
replaced by `Backend.timing_probe(start)`, whose `trial(z, dz)` takes one trial step that
commits nothing and whose `field()` reads that trial's end state. Removed with the loop:
`solprop.backends.driver` (`StepDriver`, `refuse_general_leaves`),
`solprop.backends.multistep` (`AdamsMarch`; `CallLedger` is now
`solprop.backends.ledger.CallLedger`), `solprop.backends.multistep_general`,
`solprop.backends.low_storage.LowStorageMarch` (the NumPy reference's own marches are in
`solprop.backends.numpy_ref.marches`), `solprop.backends.numpy_ref.admission`, the
reference's `helpers` attribute, `refuse_formulation`, `COUPLED_SECTIONS` and
`EXPONENTIAL_REPRESENTATIONS` of `solprop.backends.exponential` (its one Adams refusal is
`refuse_adams`), `solprop.backends.registry.statistics_refusal`,
`NativeBackend.far_field_statistics`, and `solprop_native`'s `*_FIELDS` tuples.
`reached` and `replay_tolerance` are now in `solprop.stepper.schedule`, and the general
route's node chunk in `solprop.backends.node_chunk` (`general_node_chunk`).
`Simulation.run`, `SimulationRunner`, scans and the `prop_*` functions do all of this
themselves, with unchanged interfaces.
