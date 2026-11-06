`solprop.backends.protocol.Backend` now declares what the run layer asks of a backend, which
it used to look up by name: `set_stop`, `attach_monitor`, `in_flight`, `march_snapshot`,
`march_snapshot_nbytes`, `march_export_refusal`, `restore_march`, `edge_memory`,
`timed_step`, `closed` and `reads_transverse_edge`. `StepDriver` defines all of them, so a
backend built on it changes nothing.
