`SimulationSettings(held_plane=False)` drops the state volume a low-storage run holds at
its last field-free plane, from `memory_ledger()`, the admission and `estimate()`, for a
run that will never resume. An output without `CheckpointOutput` now holds nothing
whatever the setting says, since it could never read the state. A step abandoned in flight
then records `failed` with no terminal checkpoint, and says plainly when the file cannot be
resumed; a stop between steps checkpoints as ever. Outside the configuration hash, with
the backend's record saying `held_plane: false` where nothing was held (#345).
