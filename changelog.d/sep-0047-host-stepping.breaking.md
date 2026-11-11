The `Backend` protocol (`solprop.backends.protocol`) no longer declares the stepping
members a run drives through a session: `set_initial`, `set_progress`, `set_stop`,
`attach_monitor`, `propagate_to`, `observe`, `in_flight`, `stats`,
`restore_call_ledgers`, `march_snapshot`, `march_snapshot_nbytes` and `restore_march` move
to the new `HostStepping` protocol, which the NumPy reference implements and the native
backend does not ([SEP 47](docs/dev/seps/archive/sep-0047-run-ownership.md) section 6.8). Code that
steps a `numpy_ref` backend directly narrows the backend `select` returns with
`solprop.backends.protocol.stepping(backend)`; every backend runs through
`backend.prepare(problem, request).start(start)`.
