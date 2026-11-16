The `Batch` scan executor is removed, with the group machinery behind it on both backends
(SEP 52). On the native backend a `Batch` group built a complete backend for every member
and ran one session each, so it cost `size` times the memory of `Local` for no gain in
speed; on `numpy_ref` it fused only the Runge-Kutta stage sums. A script that imports
`Batch` now fails at import: use `Local()` to run the points one after another on one
reused backend, or `Pool()` to run them in parallel worker processes (as many workers as
`Batch` had members hold about what it did, and run their points concurrently); `Queue`
and `Slurm` spread them over controllers and cluster jobs. Every executor still replays a
fixed schedule. Also removed: `solprop.driver.batch`,
`solprop.backends.registry.select_batch`, `solprop.backends.BatchBackend`,
`Backend.max_batch`, `Backend.prepare_group`, the `GroupSession` and `PreparedGroup`
protocols, `check_group`, `start_group`, `solprop.backends.events.GroupBatch`,
`ExportTicket.member`, `StatsSpec.same_tables`, the reference's `HostGroup` and
`NumpyBatchBackend`, `solprop_native.group`, and the benchmark suite's `group_batch` case.
