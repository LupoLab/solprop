`MaterialSelection` no longer has an `energy_loss` field, and
`solprop.physdata.selection.ENERGY_LOSS_POLICIES` is gone: its one value,
`"peak-weighted"`, was validated and recorded but never read, since each optmat rate
applies its own convention, which its `ionisation` entry in the material manifest still
records. Drop `energy_loss=` from `MaterialSelection(...)` and from a recipe's
`MaterialSelection` table; the run is unchanged. A run's material manifest no longer
carries a `policies.energy_loss` entry.
