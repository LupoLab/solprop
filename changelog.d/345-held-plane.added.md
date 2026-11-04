A low-storage run no longer holds a state volume at its last field-free plane unless
`SimulationSettings(held_plane=True)` asks for it, so `memory_ledger()` and `estimate()` are
one volume smaller by default. Without it, a step abandoned in flight records `failed` with
no terminal checkpoint, and says plainly when the file cannot be resumed; a stop between
steps checkpoints as ever. An output without `CheckpointOutput` holds nothing whatever the
setting says, since it could never read the state. Outside the configuration hash, with the
backend's record saying `held_plane: true` where a plane was held for, and a recipe's
`[execution]` table takes it. A backend driven by hand holds nothing either, unless it is
asked with `set_held_plane(True)` (#345, #500).
