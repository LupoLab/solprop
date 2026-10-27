# Benchmarks

Release performance tracking
([SEP 38](../docs/dev/seps/archive/sep-0038-performance-tracking.md)); its manual is
`docs/dev/benchmarking.md`. Run it on a quiet machine:

```bash
uv run poe bench                                     # the plan and its estimate
uv run python -m benchmarks track --append-history   # about an hour; records history/
uv run python -m benchmarks history
uv run python -m benchmarks ab --case CASE --baseline-python OLD/bin/python  # fast A/B
```

`history/` holds one CSV per hardware set, a few rows per release. Run bundles go to
`.benchmarks/`, which git ignores; the benchmark rules and indicative results are in
`docs/dev/benchmarks.md`. What has already been tried for CPU performance is in
`docs/dev/cpu-performance.md`. Earlier campaign harnesses, the retired standard suite's
baselines and their raw data are kept with the project's development records, not in
this repository.
