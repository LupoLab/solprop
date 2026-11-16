`MemoryOutput` now returns read-only views of its buffers (`run["field"]`, `run["z"]`,
`run["stats/<name>"]`, the step history and statistics, `noise_phase`, `plane_field`): a
write through one raises `ValueError` instead of silently changing the stored run. Take
`.copy()` of an array to change it.
