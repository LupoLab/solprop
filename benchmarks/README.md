# Benchmarks

The standard benchmark suite
([SEP 30](../docs/dev/seps/archive/sep-0030-standard-benchmark-suite.md)); its manual is
`docs/dev/benchmarking.md`. Run it on a quiet machine:

```bash
uv run poe bench                                   # core set, quick mode, native CPU
uv run poe bench run --mode standard --backend numba_cpu
uv run python -m benchmarks compare RUN_A RUN_B
uv run python -m benchmarks ab --case CASE --baseline-python OLD/bin/python  # fast A/B
```

Runs and machine profiles go to `.benchmarks/suite/`, which git ignores; the benchmark rules
and indicative results are in `docs/dev/benchmarks.md`. What has already been tried for CPU
performance is in `docs/dev/cpu-performance.md`. Earlier campaign harnesses, the retired
version-2 suite and their raw data are kept with the project's development records, not
in this repository.
