A `Simulation` now copies its `configuration=` when it is built, and `with_input` the
caller's `configuration=` keys: dictionaries, lists, tuples and sets are rebuilt and
arrays become read-only copies, so changing them before the run no longer changes the
recorded configuration or its hash. `PiecewiseLinear`, `PressureGradient`,
`FromCallable` (its declared positions) and `SeriesOfPath` keep read-only copies of the
arrays and series they are given, so changing a profile's table after passing it to a
simulation no longer changes the run or its record. `SimulationSettings` keeps
`components` as a tuple copied from the sequence given. The record and hash of an unchanged
input are the same as before
([#471](https://github.com/LupoLab/solprop/issues/471)).
