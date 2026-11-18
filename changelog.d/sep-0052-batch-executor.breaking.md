The `Batch` scan executor is removed, with the batch machinery behind it on both
backends (SEP 52): on the native backend it cost `size` times the memory of `Local` for
no gain in speed. A script that imports `Batch` now fails at import: use `Local()` to
run the points one after another on one reused backend, or `Pool()` to run them in
parallel worker processes; `Queue` and `Slurm` spread them over controllers and cluster
jobs. Every executor still replays a fixed schedule. Also removed:
`solprop.driver.batch`, `solprop.backends.batch`,
`solprop.backends.registry.select_batch`, `solprop.backends.BatchBackend`,
`Backend.max_batch`, `StatsSpec.same_tables`, the reference's `NumpyBatchBackend` and
`solprop_native.batch` (`NativeBatchBackend`).
