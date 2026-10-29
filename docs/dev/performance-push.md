# Native performance push: working log

This page is the working log of a performance campaign on the native backend. It covers
the NVIDIA A40 (CUDA), and the CPU on a dual Intel Xeon Gold 6240 and a dual AMD EPYC 7763
at one and sixteen threads. It records what was tried and what each attempt measured.
Adopted changes also have their entries in [benchmarks](benchmarks.md) and
[CPU performance](cpu-performance.md). This page will be folded into those pages, or
removed, before the work is merged.

The baseline is `main` at `740b24d`. Timings follow the
[benchmark rules](benchmarks.md#rules): short `--length` windows, fresh processes,
interleaved arms on the same machine, and medians. Benchmark parameters and stepper
settings are not changed.

The accuracy rule has two levels:

- An execution change on the same step sequence keeps each step within about $10^{-10}$
  of the baseline. Any change that moves results clearly beyond rounding is listed under
  [flagged](#flagged-accuracy-items).
- Over an adaptive run, the step counts may shift slightly. The fields then agree within
  what the stepper's relative tolerance implies.

## Summary

| Track | Scope | Gain adopted so far |
|---|---|---|
| Device | NVIDIA A40: radial filament, Cartesian Kerr, the pinned single upload | — |
| Adams | the x86-64 predict and correct passes | — |
| Ports | ideas from the `cpunative` experiment | — |
| Defaults | build flags, OpenMP policy, thresholds, FFTW planning, fixed costs | — |

## Flagged accuracy items

None yet.

## Log

The campaign has started; entries follow as experiments complete.
