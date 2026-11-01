# Choosing a backend

This page says which backends solprop has, which one a run uses, how closely they agree
and how a run's memory is budgeted. The physics of a run is decided before a backend
sees it: solprop first translates a simulation into one numerical problem — the arrays
and tables of its linear operator, its nonlinear responses and its transforms — and
every backend executes that same problem. The backends differ in speed, in where they
run and in the representations they support. Choose a backend that supports the full
configuration; a faster backend does not select a different physical model. Use the
native backend unless you have a reason not to; `numba_cpu` is its slower fallback.

| backend | what it is | use it for |
|---|---|---|
| `numpy_ref` | plain NumPy, the **reference**: every other backend is tested against it | checking a result; the few representations only it runs (`CoherentWall(method="cf4")`, with `integrator=RungeKutta()`); small problems where compiling costs more than running |
| `numba_cpu` | compiled kernels shipped with solprop, with its own thread pools | the fallback where the native backend cannot run: `"auto"` uses it, with a warning, when `solprop-native` is not installed or the processor is below its wheel's baseline. Also problems a GPU build refuses or runs slowly, in a process that has only that build |
| `native` | the [native backend](native.md), the optional solprop-native distribution built from the repository's `native/` directory: OpenMP, FFTW and a vendor BLAS on a CPU, or CUDA/HIP on a GPU | **the one to use**: the default when installed, and the fastest in every recorded comparison with `numba_cpu` |
| `mlx` | the Apple-silicon GPU through Apple's MLX, in **single precision only** ([single precision](single-precision.md)); the `mlx` extra, macOS on Apple silicon; **experimental** in this release | `precision="single"` capillary and fibre runs on a Mac, where it is the `"auto"` choice; it refuses `precision="double"` |

**Tiers.** `numpy_ref` and `native` are the core backends: new features land on both,
and they come first for testing and development (the few features one of them refuses
are named below and in their chapters). `numba_cpu` is Tier 2: mature and tested, but
slower, and a new feature reaches it only where its design says so; until then it refuses
the run by name and names a backend that runs it. `mlx` is experimental
([backends, §10](../../dev/backends.md); [ADR-061](../../dev/decisions.md)).

`backend="auto"`, the default of every entry point, is `"native"` when `solprop_native`
is installed. Without it, `"auto"` chooses `"numba_cpu"` and emits a `SolpropWarning` once
per process, saying so and how to install the native backend, and the run records
`numba_cpu` as its backend. To use `numba_cpu` deliberately and silence the warning, pass
`backend="numba_cpu"`. Under `precision="single"`, `"auto"` is `"mlx"` when MLX and a Metal
GPU are present, and refused otherwise.

With the native backend installed, `"auto"` **never falls back** but in one case: if it
cannot import, fails its compatibility check or refuses the problem, the
error is raised, naming what was refused, and you choose another backend explicitly.
The one exception is a processor below the installed native build's CPU baseline (an
x86-64 processor without AVX2 under a wheel, for example; see
[installation](../../introduction/installation.md)): there `"auto"` chooses `"numba_cpu"`,
with the same warning naming the missing features. `backend="native"` raises instead.

Important unsupported combinations include `CoherentWall(method="cf4")`, which only
`numpy_ref` runs, with the Runge–Kutta march. A GPU build of the native backend also refuses
a problem whose linear operator or nonlinear prefactor varies along z in a way
that could not be expanded in the medium's parameters and had to be tabulated along z
instead: the adiabatic opt-out `taper_coupling=False` of a multimode capillary taper
whose expansion does not resolve (a kinked radius profile, for instance), or a gas
mixture of three or more species on independent density paths. The refusal leads with
what runs it: a CPU build of the native backend, `backend="numba_cpu"` or
`backend="numpy_ref"`. The
[TDSE response](../media/tdse.md) currently requires `numpy_ref`;
select it explicitly rather than relying on automatic backend choice.

`Multistep()` ([the Adams integrator](../numerics/adams.md)) and `RungeKutta()`, each
the default of some interfaces
([the defaults](../numerics/choosing-a-stepper.md#the-defaults)), are available across
the three binary64 backends for their supported configurations; single precision
refuses `Multistep()` on every backend. Beside a coupled flow, Adams requires its
exponential formulation; CF4 requires Runge–Kutta on the reference backend. Check the
response and geometry chapters for further restrictions. The backend used is
logged at INFO and recorded in the output.

```python
from solprop.backends.registry import available, resolve_name

print("installed:", available())
print("auto resolves to:", resolve_name("auto"))
```

**How close they are.** Backend qualification compares short calculations with the
NumPy reference using stated rounding and discretisation budgets. A different
build, transform plan or device can change floating-point summation order, so
cross-backend agreement need not be bitwise. Nor is a repeat on the same build:
solprop promises numerical accuracy, not identical bits, across backends, builds,
machines, thread counts, batches and repeated runs
([SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md)), and a rounding
difference can change an adaptive run's steps. Small differences can also grow in
noise-sensitive propagation: compare the physical observable and numerical
convergence, not only an array digest. Record the backend, build and thread
settings with a result. See [numerical qualification](../../dev/accuracy.md) and
[its backend-equivalence ladder](../../dev/accuracy.md#7-backend-equivalence)
for the scope and measurements of the existing checks.

**Speed and GPUs.** How to install the native backend, build it for a GPU and run on a
device is on [The native backend](native.md); thread counts, OpenMP and BLAS settings,
and what to use for scans are on [Performance tuning](tuning.md).

## Which backend to use

Use the native backend. In the [performance history](../../dev/benchmarking.md), recorded
at 0.1.0 on 2026-11-01 on the three machines below, the native backend's distribution wheel
was faster than `numba_cpu` on every case compared, ten at one thread and twelve on all
cores. The table gives `numba_cpu`'s time over the native backend's; single cases ranged
from 1.2 to 6.7:

| processor | one thread, geometric mean | all cores, geometric mean |
|---|---|---|
| two Intel Xeon Gold 6240 (36 cores) | 2.2 | 2.8 |
| AMD EPYC 7763 (16 cores) | 2.0 | 2.4 |
| Apple M5 Max (18 cores; most of its `numba_cpu` rows are flagged as on a busy machine) | 1.5 | 2.1 |

The measurements behind the advice below are in the
[indicative results](../../dev/benchmarks.md#indicative-results); time your own problem for
ten steps before committing a long campaign to one backend.

- **Which native build.** Build the native backend on the machine that runs it, or install a
  distribution build, the wheel ([build profiles](building-native.md#build-profiles)): the local build was the
  fastest backend on every coherent-wall and general modal case measured, at one, four and
  eighteen threads, and the distribution build ran within about 5 % of it. On Linux,
  prefer MKL (with oneAPI sourced) for TE/TM coherent walls, where it was 1.4–1.8 times
  faster than OpenBLAS; elsewhere the two tied.
- **Independent scans on a many-core CPU.** Many one-thread workers beat one many-thread
  run ([Performance tuning](tuning.md#recipes)). Stay on the native backend for them: it
  also led at one thread in the table above. Those are single-process timings, so for a
  large campaign time one point of the scan with your real number of workers.
- **The general modal route** (`from_waveguide(..., provider=...)`): either compiled
  backend was 1.3 to 7.7 times faster than `numpy_ref` over ten steps at one thread, and a
  local native build gains a further factor at more threads. `numpy_ref` is slowest on the
  analytic cubic, by five to eight times. Budget memory
  ([execution resources](resources.md#memory)) before raising the channel count, the node
  count or the grid.
- **The coupled flow** (a coherent wall, a varying wall, a split block): all three CPU
  backends run the constant-Schur and frozen-residual sections;
  `CoherentWall(method="cf4")` runs on `numpy_ref` alone. A single small wall run gains
  nothing past a few threads. On a varying wall the flow's actions are re-formed every
  adaptive step, in a few per cent of a step on either compiled backend.
- **On a GPU** (a CUDA or HIP build of the native backend): pick it for large grids. Every general-route,
  coupled-flow and moving-guide case measured ran faster on either a data-centre NVIDIA
  GPU or an AMD Instinct GPU than on an 18-core CPU socket (about four and three times on
  a scaled graded-index guide); a small case is limited by kernel-launch latency
  ([The native backend](native.md#running-on-a-gpu)).
- **Batches.** A fixed-schedule batch fuses the accepted updates of its members, while
  the response and transform work stays per member, so it does not amortise a device's
  launch sequence; measure a batch's gain for your own shape. On a CPU, a `Pool` of
  one-thread workers is the better scan pattern.

## Memory and scans

For memory admission, output storage and scan workers, continue to
[execution resources](resources.md). Choose resolution for the physics before
tuning thread counts; reducing modes or nodes to fit memory changes the model
or its numerical accuracy.

## The Apple-GPU backend

`pip install solprop[mlx]` installs [MLX](https://ml-explore.github.io/mlx/) on macOS on
Apple silicon, and with it the `mlx` backend
([SEP 33](../../dev/seps/archive/sep-0033-single-precision-mlx.md)). Apple GPUs have no binary64
arithmetic, so it runs `precision="single"` only, under the contract and with the
accuracy record of [single precision](single-precision.md); it is validated against the
reference backend's single-precision path. It runs the capillary and fibre routes
(mode-averaged and multimode, field and envelope grids, Kerr, Raman, field-grid plasma,
both absorber schemes, Runge–Kutta and the low-storage march, lockstep batches) and
radial and Cartesian free space, including a line, with the safety ramp, and the
envelope-grid plasma. It does not read the transverse edge, so its free-space runs report
`unwatched` and it refuses a requested real-space collar
([transverse boundaries](../free-space/model.md#transverse-boundaries)).

The backend is **experimental** in this release. Its Metal kernels are tested on a
Metal device by hand rather than in continuous integration, and its speed has not been
re-measured on the default grids, whose 2-3-5-smooth sizes MLX's GPU transform runs
four to eight times slower than powers of two: build single-precision grids with
`fft_size="pow2"` ([grid sizes on `mlx`](single-precision.md#grid-sizes-on-mlx)).

MLX computes float32 matrix products in TF32 by default on Apple GPUs, which SEP 25's
arithmetic rules forbid. solprop sets `MLX_ENABLE_TF32=0` when it imports the backend,
but MLX reads the setting once, when it first initialises: in a process that used MLX
before, set it in the shell (`export MLX_ENABLE_TF32=0`) or before any `import mlx`.
The backend measures its matrix products when it is built and refuses to run if they
are TF32.

