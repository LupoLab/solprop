Project and tooling: the benchmark harness gains performance-history rows for 0.2.0
(`benchmarks/history/`, on the same nine hardware sets as 0.1.0) and `--counters` on
`track` and `ab`, one separately reported run per worker with the native backend's
structural counters (returns to Python by cause, native calls, attempts, planes,
statistics rows, edge reads and host copies). History rows record the OpenMP wait policy
and are compared only within one native build profile, and `track --append-history`
refuses a native build stamped `+dirty` without `--allow-dirty`
([#138](https://github.com/LupoLab/solprop/issues/138)).
