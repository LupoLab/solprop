`RunRequest.with_stepper(...)` now takes a fresh request only, and raises
`TranslationError` for one that `RunRequest.for_run(...)` has given a field retention
other than every field, or a start plane beyond the entrance: both are resolved against
the planes the new stepper section replaces, and were copied unchanged before
([#619](https://github.com/LupoLab/solprop/issues/619)). Call `with_stepper` first and
`for_run` after. `Simulation.with_input` already did, and is unchanged.
