# The Julia backend experiment

**An experiment, not a supported backend. Not for merge.** It asks how fast a compact,
single-source Julia engine can run solprop's routes on a CPU and on an NVIDIA GPU,
against the native backend. [STATUS.md](STATUS.md) holds the design, the measured
results and what is left; the [manual page](../docs/guide/performance/julia.md) says
how to use it.

`backend="julia"` steps the problem solprop translates, behind the same `StepDriver`
seam as the native backend. The Python front end, the translation, the march's
controller and the output are solprop's own. The engine is the Julia package in
[`src/`](src):

| File | What |
|---|---|
| `SolpropJulia.jl` | the engine type, construction and its warm-up, the bridge's entry points |
| `launch.jl` | the device layer: every pass a scalar body run by `each!`, `each_column!`, `reduce_each` or `rowsum!`; FFT plans and GEMMs |
| `vmath.jl` | branch-free binary64 `exp` and `log` within one ulp, for the CPU's vectorised passes |
| `modal.jl`, `free.jl` | the modal and radial routes, and Cartesian and line free space |
| `responses.jl` | Kerr, one field-plasma channel and one Raman term |
| `dopri.jl`, `rk4.jl`, `adams.jl` | the DOPRI5 trial and commit, SEP 24's RK4IP and SEP 16's exponential Adams |
| `helpers.jl` | the shared code's Chebyshev rows and Adams basis (SEP 29 section 6.6), and an Adams trial's tables |
| `stages.jl` | the longitudinal expansions' stage weights, and the stage scalars from the engine's own tables |
| `observe.jl` | what the driver reads between steps: the state, the boundary monitor's and the transverse edge's sums, the Poynting statistics' power |
| `../ext/SolpropJuliaCUDAExt.jl` | CUDA: the engine's arrays become `CuArray`s |

## Setting up

In a source checkout of this branch, with Julia 1.13 (or let the bootstrap download the
checksummed release):

```sh
uv sync --extra julia
export SOLPROP_JULIA_EXE=$(julia +1.13 -e 'print(joinpath(Sys.BINDIR, "julia"))')
uv run --no-sync python julia/bootstrap.py          # --cuda for the GPU environment too
```

The bootstrap instantiates `julia/Manifest.toml` into `.julia-depot/`
(`SOLPROP_JULIA_DEPOT` names another). `SOLPROP_JULIA_ROOT` names the checkout when
solprop is installed without `-e`. A run then takes `backend="julia"`:

| Variable | Meaning |
|---|---|
| `SOLPROP_JULIA_THREADS` | the engine's threads (default 1); the benchmark harness sets it to the width |
| `SOLPROP_JULIA_DEVICE` | `cpu` (default) or `cuda`, which loads `julia/cuda`: CUDA.jl resolved with the engine |
| `SOLPROP_FFTW_PLAN` | `estimate`, `measure` (default) or `patient` |
| `SOLPROP_JULIA_TIMING` | `1`: the engine's section timers (`julia/profile.py` prints them) |
| `SOLPROP_JULIA_CHECK` | `1`: refuse any kernel body a GPU could not take, on any device (the tests set it) |
| `SOLPROP_JULIA_PLASMA` | `samples`: the GPU's per-sample plasma passes on a CPU too |

A process holds one Julia runtime. Import nothing from `juliacall` before solprop
starts it.

## Measuring

The benchmark harness takes `julia` as an arm, and `julia32` for it in single precision:

```sh
uv run --no-sync python -m benchmarks ab --baseline-backend native --candidate-backend julia \
  --baseline-python .venv/bin/python --case rdw --threads 1,16 --length 0.02 --noise none
```

`python julia/profile.py PLAN.json ROW...` times one plan row's complete run, warmed,
with `SOLPROP_JULIA_TIMING=1` for the sections or `SOLPROP_JULIA_PROFILE=1` for Julia's
sampling profile. `julia/bench/launch.jl` is the launch-strategy measurement the device
layer rests on.

## Testing

```sh
uv run --no-sync pytest tests/backends/test_julia.py -n 4   # SOLPROP_JULIA_DEVICE=cuda on a GPU
```

Seven focus windows against `numpy_ref`: the same steps, and the field to `1e-10` in
binary64, `1e-4` against the reference's binary32 path. Julia runs bounds-checked there.
