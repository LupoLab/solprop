`solprop.backends.protocol.Backend` now declares what the run layer asks of a backend, which
it used to look up by name: `set_stop`, `set_held_plane`, `attach_monitor`, `held_observation`,
`in_flight`, `march_snapshot`, `march_snapshot_nbytes`, `march_export_refusal`, `restore_march`,
`timed_step`, `closed` and `reads_transverse_edge`. `StepDriver` defines all of them, so a
backend built on it changes nothing.
