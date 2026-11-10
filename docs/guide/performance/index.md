# Backends and performance

Choose a backend that supports the model, then size the calculation and tune its
execution. Backend choice changes where and how the calculation runs; the
physical and numerical settings still determine what it computes. Installation
begins in the [introduction](../../introduction/installation.md). These chapters
cover the native backend, memory, CPU threads, and GPU execution. The best speed on a
given machine comes from [calibrating solprop to it](calibration.md), once, with
`solprop calibrate`.

To measure a change's performance, or to see roughly what each backend and method
costs, read the developer pages: [Benchmarks](../../dev/benchmarks.md) gives the
measurement rules and indicative results, and
[Running the benchmarks](../../dev/benchmarking.md) is the manual for
`uv run poe bench`.

```{toctree}
:maxdepth: 1

calibration
backends
native
julia
building-native
resources
progress
tuning
single-precision
```
