# Choosing a backend

This page says which backends solprop has, which one a run uses, how closely they agree
and how a run's memory is budgeted. The physics of a run is decided before a backend
sees it: solprop first translates a simulation into one numerical problem — the arrays
and tables of its linear operator, its nonlinear responses and its transforms — and
every backend executes that same problem. The backends differ in speed, in where they
run and in the representations they support. Choose a backend that supports the full
configuration; a faster backend does not select a different physical model. Use the
native backend, which `"auto"` selects; `numpy_ref` is for the cases below.

| backend | what it is | use it for |
|---|---|---|
| `native` | the [native backend](native.md), the solprop-native distribution built from the repository's `native/` directory and a required dependency of solprop: OpenMP, FFTW and a vendor BLAS on a CPU, or CUDA/HIP on a GPU | **the one to use**: what `"auto"` selects |
| `numpy_ref` | plain NumPy, the numerical **reference**: every other backend is tested against it. It is unthreaded, and roughly 6 to 40 times slower than threaded native | checking a result, and [the features native does not run](#features-that-run-on-the-reference-backend). Also problems a GPU build refuses or runs slowly, in a process that has only that build |

## Which backend a run uses

`backend="auto"`, the default of every entry point, and an explicit name select as follows.
The right-hand columns say what happens on this machine; where native isn't consulted they
repeat the first:

| request | native usable | processor below the floor, or its features unreadable | native fails to load | no native package |
|---|---|---|---|---|
| `auto` | `native` | refuses, naming the missing features or the failed check | the load's own error, unchanged | refuses, saying how to install it |
| `native` by name | `native` | as above | as above | as above |
| TDSE response, `auto` | `numpy_ref`; native isn't consulted | the same | the same | the same |
| `numpy_ref` by name | runs; native isn't consulted | the same | the same | the same |

`"auto"` has no fallback. These cases stay distinct, so an import failure is never reported
as an unsupported platform, and a problem the chosen backend refuses is refused, naming what
was refused and what runs it, for you to choose explicitly. Which platforms and processors
have a native build is on the [installation page](../../introduction/installation.md). The
run records the backend it used, and the `SOLPROP_REQUIRE_DEVICE` setting makes a GPU job
stop rather than run anywhere else ([a job that must run on a
device](native.md#a-job-that-must-run-on-a-device)).

A GPU build of the native backend also refuses
a problem whose linear operator or nonlinear prefactor varies along z in a way
that could not be expanded in the medium's parameters and had to be tabulated along z
instead: the adiabatic opt-out `taper_coupling=False` of a multimode capillary taper
whose expansion does not resolve (a kinked radius profile, for instance), or a gas
mixture of three or more species on independent density paths. The refusal leads with
what runs it: a CPU build of the native backend, in a second environment, or
`backend="numpy_ref"`.

`Multistep()` ([the Adams integrator](../numerics/adams.md)) and `RungeKutta()`, each
the default of some interfaces
([the defaults](../numerics/choosing-a-stepper.md#the-defaults)), are available on
both backends for their supported configurations. Beside a coupled flow, Adams requires its
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

## Features that run on the reference backend

`numpy_ref` is the numerical reference, and it is also the supported way to run these
features, which native does not run yet. Each is a limit of native, and each stays
supported on the supported platforms:

| feature | reached by | tracked by |
|---|---|---|
| the [TDSE response](../media/tdse.md) | `auto` or by name | |
| `CoherentWall(method="cf4")`, and a coupled flow with a CF4 section, with `integrator=RungeKutta()` | by name | [#304](https://github.com/LupoLab/solprop/issues/304) |

The TDSE row's `auto` is that of the entry points; `SimulationSettings` built by hand with
`tdse_execution` [names the backend](../results/interior-launch.md). Running `numpy_ref` by
name on an unsupported platform may keep working, without a support promise. The list is not
a general waiver: a new feature still lands on both `numpy_ref` and `native`
([SEP 0](../../dev/seps/sep-0000-process.md)), and joins the list only as an accepted
exception.

## The removed `numba_cpu` backend

solprop once had a third double-precision backend, `numba_cpu`, and 0.1.0 is the last
release with it. Naming it now raises {py:class}`~solprop.exceptions.BackendError`, naming
`native` and `numpy_ref`, so a script or recipe that selects it needs `backend="native"`, or
no `backend=` at all. Output, snapshots and a scan's execution metadata that record
`numba_cpu` still load, and keep saying so: the record is not rewritten.

The backend is an execution resource, so changing it does not change a run's identity. A
run can continue on `native` or `numpy_ref`, with its script's or recipe's backend changed
deliberately, through the normal [resume](../results/resuming.md) checks; the original
execution record stays and the new one is added. Adams history crosses to the new backend,
except a frozen-residual coupled flow's: a host engine's history continues on `numpy_ref`,
and on native starts afresh and says why. Anything refused is recovered with
`solprop==0.1.0`.

Its settings are gone too: `SOLPROP_NUMBA_THREADS`, `SOLPROP_NUMBA_REGIME`,
`SOLPROP_FFTW_THREADS`, `SOLPROP_TRANSVERSE_FFTW_THREADS`, `SOLPROP_PASS_TIMERS`,
`NUMBA_DISABLE_JIT` and `NUMBA_NUM_THREADS` no longer do anything.

## Which backend to use

Use the native backend. `numpy_ref` is unthreaded and roughly 6 to 40 times slower than
threaded native, so it is for checking a result and for what only it runs; the
[performance history](../../dev/benchmarking.md) records the native backend's own timings.
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
  run ([Performance tuning](tuning.md#recipes)). Stay on the native backend for them. For a
  large campaign time one point of the scan with your real number of workers.
- **The general modal route** (`from_waveguide(..., provider=...)`): the compiled
  backends measured, `native` and the since-retired `numba_cpu`, were 1.3 to 7.7 times faster than `numpy_ref` over ten steps at one thread, and a
  local native build gains a further factor at more threads. `numpy_ref` is slowest on the
  analytic cubic, by five to eight times. Budget memory
  ([execution resources](resources.md#memory)) before raising the channel count, the node
  count or the grid.
- **The coupled flow** (a coherent wall, a varying wall, a split block): both CPU
  backends run the constant-Schur and frozen-residual sections;
  `CoherentWall(method="cf4")` runs on `numpy_ref` alone. A single small wall run gains
  nothing past a few threads. On a varying wall the flow's actions are re-formed every
  adaptive step, in a few per cent of a step on the native backend.
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

For memory, output storage and scan workers, continue to
[execution resources](resources.md). Choose resolution for the physics before
tuning thread counts; reducing modes or nodes to fit memory changes the model
or its numerical accuracy.
