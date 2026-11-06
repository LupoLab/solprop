# The Julia experiment

`backend="julia"` is an experiment on this branch, not a supported backend: it asks how
fast one compact Julia source can run solprop's routes on a CPU and on an NVIDIA GPU,
in binary64 or binary32. Use the [native backend](native.md) for results. The design,
the measurements and what is left are in the branch's `julia/STATUS.md`.

## What it runs

The translated problem is the same one every backend executes ([choosing a
backend](backends.md)); the julia backend steps it behind the same seam as native, and
refuses by name what it does not run.

| | runs | refuses |
|---|---|---|
| routes | resolved and mode-averaged capillaries (constant radius, separable longitudinal profiles), radial, Cartesian and line free space | planar and general modal guides, tapers, coupled flows |
| responses | Kerr (with THG), one field-grid plasma channel, one constant Raman term (modal) | envelope plasma, several channels or Raman terms, varying Raman, dispersive Kerr, TDSE, noise |
| steppers | DOPRI5 with FSAL, the exponential Adams march (not Cartesian), RK4IP (Cartesian) | the interaction Adams formulation, the carrier frame, interpolated planes |
| execution | CPU threads; an NVIDIA GPU (`SOLPROP_JULIA_DEVICE=cuda`); `precision="double"` or `"single"` | vector polarisation, batches |

## Trying it

From a checkout of the branch:

```sh
uv sync --extra julia
uv run --no-sync python julia/bootstrap.py                 # add --cuda for a GPU
SOLPROP_JULIA_THREADS=16 uv run --no-sync python my_run.py   # with backend="julia"
```

The bootstrap downloads Julia 1.13.3 (checksummed) unless `SOLPROP_JULIA_EXE` names one,
and instantiates the committed manifest. A run's first propagation pays nothing for
Julia's compilation: the engine compiles and runs each of its paths once when it is
constructed, which takes seconds. `julia/README.md` lists the environment variables.

## How it is built

One source for every device and precision. Each state-sized operation is a scalar
function launched over the state by a small device layer: on a CPU, Polyester tasks
running contiguous SIMD loops; on a GPU, a KernelAbstractions kernel a sample a thread.
FFTs are AbstractFFTs plans (FFTW or cuFFT) and contractions `mul!` (BLAS or cuBLAS).
On a CPU one pool runs everything, the transforms as blocks of single-threaded plans and
the products as blocks of rows, because the libraries' own threads beside the pool's
made a run slower on four threads than on one. In binary32 the state is held as
`a / s` for SEP 33's power-of-two scale `s`, and the operator stays binary64, so
propagators and the longitudinal root are formed in binary64 and rounded once.

## How fast it is

Indicative only: short windows of the benchmark cases, measured with `python -m
benchmarks ab` against native on the same machine. The branch's ledger
([benchmarks](../../dev/benchmarks.md#julia-backend-experiment)) has the numbers, the
machines and the commits. On CPUs the julia backend is within one to three times
native's time; on an NVIDIA A40 within one to four times native's CUDA build. Most of
what is left is host work per step and per saved plane, which native does in compiled
code: the Adams march and the statistics.
