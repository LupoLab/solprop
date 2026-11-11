The legacy step driver `solprop.backends.driver` (`StepDriver`), the Adams march class
`solprop.backends.multistep.AdamsMarch` and `solprop.backends.low_storage.LowStorageMarch`
are removed ([SEP 47](docs/dev/seps/archive/sep-0047-run-ownership.md) section 6.8): the native
backend runs every run in its native session and inherits no Python step loop, and the
NumPy reference marches with its own `solprop.backends.host` classes. `reached` and
`replay_tolerance` are now in `solprop.stepper.schedule`, and `refuse_general_leaves` in
`solprop.backends.numpy_ref.admission`. A native backend's direct stepping members
(`set_initial`, `propagate_to`, `observe`, `stats`, `attach_monitor` and the rest) are
gone with them: start a run with `backend.prepare(problem, request).start(start)` or
`solprop.backends.sessions.start_session`.
