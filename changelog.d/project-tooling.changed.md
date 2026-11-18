Project and tooling: the benchmark harness gains performance-history rows for 0.2.0
(`benchmarks/history/`, on the same nine hardware sets as 0.1.0), a `sep47` set of short
windows (statistics rows at every step, snapshot controllers, repeated `with_input` runs,
a four-point scan, saved and field-free spatial planes, wide coupled flows), and
`--counters` on `track` and `ab` for one separately reported run per worker with the
native backend's structural counters (returns to Python by cause, native calls,
attempts, planes, statistics rows, edge reads and host copies). Its workers run with the
backend's own OpenMP waiting by default (`--threading backend`; `--threading passive`
keeps the earlier policy), and history rows record the policy; `ab` calibrates each build
at the `standard` level (`--calibrate thorough` for release benchmarks; `track` keeps
`thorough`). The history compares rows only within one native build profile, and
`track --append-history --allow-dirty` admits a native build stamped `+dirty`, marking
its rows dirty, and without the flag refuses it by naming the modified tree
([#138](https://github.com/LupoLab/solprop/issues/138)).
