# The native backend: fast CPUs and GPUs

solprop's physics is decided before a backend sees it: every backend executes the same
translated problem, and the NumPy reference backend is the oracle that the others are
tested against (see [Choosing a backend](backends.md)). The **native backend** is the
fast one. It is the package `solprop-native`, built from the `native/` directory of the
solprop repository at solprop's own version: C++ kernels written with
[Kokkos](https://kokkos.org), bound to Python with nanobind, and compiled for the machine
that runs them —

- on a **CPU** with OpenMP threads, FFTW and a vendor BLAS (Accelerate, MKL, OpenBLAS or
  AOCL). A locally built CPU backend is the fastest backend measured on the
  coherent-wall and general modal cases;
- on an **NVIDIA GPU** with CUDA (cuFFT and cuBLAS), or an **AMD GPU** with HIP (rocFFT
  and rocBLAS).

It is the backend to use, and a required dependency: `pip install solprop` installs it, and
`backend="auto"` — the default of every entry point — selects it. `numpy_ref`, which is
unthreaded and roughly 6 to 40 times slower than threaded native
([which backend to use](backends.md#which-backend-to-use)), remains the numerical
reference, and runs [the few features native does not](backends.md#features-that-run-on-the-reference-backend).

## What it runs

The native backend runs:

- field and envelope grids; single-mode, multimode and polarisation-resolved modal runs;
- Kerr and third-harmonic terms, field and envelope plasma, scalar and vector Raman
  responses, gas mixtures with per-species channels, noise and absorbing boundaries;
- the general modal route (guides that supply their own vector **E** and **H**,
  including guides that vary along z), the coupled flow of the coherent capillary wall and
  of tapers, with its `constant_schur` and `frozen_residual` sections;
- radial, Cartesian, line and planar free-space and planar-guide geometries;
- fixed-schedule batches (lockstep scan members);
- **the Chebyshev z table**: a linear operator or nonlinear prefactor that varies along
  z in a way that has to be tabulated, rather than expanded in the medium's parameters.
  Examples are a `SpecifiedMode` whose dispersion moves along z, the adiabatic opt-out
  `taper_coupling=False` of a multimode capillary taper whose expansion does not
  resolve, and a gas mixture of three or more species on independent density paths.
  The CPU build holds the table's coefficients and evaluates it at each step's
  positions in its own kernels ([z tables](../numerics/linear-tables.md)).

The GPU
builds do not run the z table: they refuse it before anything is allocated, giving the
table's size, since it would be held resident on the device; run such a problem on a
CPU build, in a second environment, or with `backend="numpy_ref"` in a process that has
the GPU build installed.

**Adams snapshots and resume history** run on the native backend as on the others. It
keeps the history of the default `Multistep()` integrator resident in its engine and
copies the history's rows to the host only where a snapshot is kept: at each save plane
of an HDF5 output that keeps the resume history, and for each periodic snapshot. A
resume installs them again and continues the history ([resuming](../results/resuming.md)).
A frozen-residual coupled flow's history is kept in a native form (each sample's
generator applied to its state), so a snapshot of it continues only on the native
backend, and a host engine's snapshot of it only on `numpy_ref`; a
mismatch starts the history afresh and says why. Other histories cross between backends.

Separately, the `cf4` section method of a coupled flow (`CoherentWall(method="cf4")`)
runs on `numpy_ref` only, with the Runge–Kutta march; the native backend does not run it.
The planar guide resolved in y is built into the GPU builds but has not been qualified
on a device. solprop has no fallback: an import failure, an incompatible build, a processor
below the build's baseline or a refused problem raises, naming what to do instead
([choosing a backend](backends.md)).

## Installing

The native backend is a required dependency of solprop, at solprop's own version, so there
is no extra:

```sh
pip install solprop
```

Precompiled wheels serve macOS on Apple silicon (osx-arm64), Linux on x86-64 (linux-64),
Linux on aarch64 (linux-aarch64), Windows on x86-64 (win-64) and Windows on Arm
(win-arm64), and solprop runs where they install ([installation](../../introduction/installation.md)). They are the **distribution** build: compiled for a CPU baseline (x86-64-v3
with x86-64-v2 beside it, any Apple silicon Mac, Neoverse N1; [processor requirements](../../introduction/installation.md)) and carrying a BLAS that shares the backend's threads. A
source build is the way past a wheel's limits: an NVIDIA or AMD GPU, or a platform or
processor no wheel serves, where it may work, with no support promise. It can also be the
**local** profile, compiled for the machine that runs it with its own vendor BLAS, which
is a few per cent faster than the wheel on the measured cases. A third profile,
**reference**, is for development checks only and is noticeably slower. The profiles
agree to rounding, not bit for bit. [Building the native backend](building-native.md)
gives the requirements, the commands for Linux and macOS, every build option, and the
CUDA and HIP builds.

An x86-64 wheel holds two builds of the extension, x86-64-v3 and x86-64-v2, and loads the best
one the processor runs. `SOLPROP_NATIVE_CPU_VARIANT=x86-64-v2` asks for the lower one, for
tests and comparisons; a build above what the processor runs is refused by name, as an
unknown name is. `solprop_native.build_info()` reports which build is loaded.

**Check the installation:**

```sh
python -m solprop_native            # the build record, one JSON object
python -c "import solprop_native as sn; sn.check_build()"
```

`check_build()` raises if the build is inconsistent (for example a GPU build that sees
no device) or does not fit the installed solprop: the two packages share a kernel
contract, and `check_build()` names any solprop member the build relies on that is
missing. In a checkout it also refuses an extension built from other sources than
`native/`. Install the two at the same version, from one release or one checkout. The build
record reports the profile (`build_profile`), the BLAS (`blas_selection`,
`blas_library`), FFTW, the compiler flags, the thread counts and, on a GPU build, the
device. Then, in Python:

```python
import solprop
from solprop.backends.registry import available, resolve_name

print("installed backends:", available())
print("backend='auto' means:", resolve_name("auto"))
```

With logging on (`solprop.configure_logging("INFO")`) every run logs
`Using native backend`, and the backend and its settings are recorded in the output file
(`run.metadata.backend`).

**Working on solprop and its native backend.** In a development checkout, a plain sync
builds the native package from `native/` and installs it editable into the repository's
environment beside solprop:

```sh
uv sync
```

It needs a C++20 compiler; a contributor to the Python or the documentation can do without
one ([contributing](../../dev/contributing.md)). The editable install does not rebuild
itself: run `uv sync --reinstall-package solprop-native` after a native change.
`uv run poe check-native` runs the native gate
and `uv run poe check-all` both gates. A build stays valid across solprop changes until
the sources under `native/` change, when `check_build()` refuses it and names that
command, or until `check_build()` names a member the installed solprop lacks. Running solprop's test lanes against the build is described in
the developer notes on [testing](../../dev/testing.md#tiers-and-backends).

## Running on a GPU

GPU support is chosen **when the package is built**, and there are no GPU wheels: for an
NVIDIA GPU (CUDA) or an AMD GPU (HIP), on Linux, build the package from source with the
device enabled and the GPU's architecture named, as
[Building the native backend](building-native.md#gpu-builds) describes. One
installation is either a CPU build or a GPU build; to compare, install the two in
separate environments. `check_build()` refuses a GPU build that cannot see a device.

Once installed, nothing in your scripts changes: `backend="auto"` runs on the device.
The field stays resident on the device between steps and is copied back only at saved
planes. Things to know:

- **One device per process.** Choose it with `CUDA_VISIBLE_DEVICES` or
  `HIP_VISIBLE_DEVICES`. To use several GPUs, run independent problems (scan points,
  noise seeds) in separate processes, each seeing one device.
- **Small problems belong on the CPU.** A single-mode fibre on a few thousand time
  samples is limited by kernel-launch latency, not arithmetic, and usually runs faster
  on the CPU. Install the CPU wheel in a second environment for such
  problems, or pass `backend="numpy_ref"` in a process that has only the GPU build, if the
  problem is small enough for it. The GPU wins on large grids: multimode and
  general modal guides with many nodes, coherent walls with many channels, broadband
  supercontinuum grids, and free-space runs.
- **On-axis plasma statistics.** The four modal `on_axis_*` plasma statistics bring only
  the axis row's spectrum to the host at each requested save plane, not the field, and
  scan it there. Their CPU parity is qualified; they are not yet qualified on a GPU. See
  [the statistics catalogue](../results/statistics-catalogue.md).
- **Reproducibility.** A device answer is not bit for bit the CPU build's, because the
  device sums in its own order; the difference is at the level of rounding and is
  covered by the same derived error bounds that every backend is tested against. A
  repeat on one build is held to the same bounds, not to identical bits
  ([SEP 25](../../dev/seps/archive/sep-0025-rounding-level-reproducibility.md)).
- **Time your own problem first.** Run ten steps on the device and on the CPU before
  committing to either for a long campaign.

### A job that must run on a device

A new release replaces a source build with that release's CPU wheel, with no message,
because solprop requires the native backend at its own version
([upgrading a source build](building-native.md#upgrading-a-source-build)). So a GPU job says
where it must run, and a job that lost its GPU build stops before it starts:

```sh
SOLPROP_REQUIRE_DEVICE=cuda python my_run.py
```

| `SOLPROP_REQUIRE_DEVICE` | meaning |
|---|---|
| unset or empty | no requirement, the default for every CPU user |
| `cuda` | the run must execute on a CUDA build that sees a device |
| `hip` | the run must execute on a HIP build that sees a device |
| `any` | the run must execute on a device build of either kind |
| anything else | raises a `BackendError` naming the accepted values |

It is an environment setting with no keyword, and scan workers inherit it. It is checked at
backend selection, before any state is allocated or a step taken, by reading the build's
`build_info().device_backend` and `device_count`, never a version label. It refuses, with a
`BackendError`, a CPU build, the wrong device backend and a device build that sees no device.
It also refuses `numpy_ref` and `mlx`, by name or through `backend="auto"` for a TDSE or
single-precision run, because such a run would not meet the job's requirement; unset it for
such a run.

## Threads on the CPU build

The CPU build uses the process's thread budget by default: the CPUs in its affinity
mask (which `taskset` or a batch scheduler's allocation narrows), their physical cores,
a container's CPU quota and `SOLPROP_THREADS`, solprop's budget for every CPU backend
([performance tuning](tuning.md)). `SOLPROP_NATIVE_THREADS` narrows it further, and
four further variables size its individual pools; all are read once, when
`solprop_native` is imported, so set them before starting Python:

```sh
SOLPROP_NATIVE_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=passive \
    python my_run.py
```

[Performance tuning](tuning.md) explains every variable, how the defaults are
chosen, the OpenMP and BLAS settings that interact with them, and what to use for one
large run, for a scan of many runs and on a cluster.

## When something goes wrong

- **`ImportError`, or `check_build()` naming a missing member.** The
  installed solprop-native does not fit the installed solprop; install both at the same
  version (`pip install --force-reinstall solprop`, or both from one checkout
  as in [Building the native backend](building-native.md)).
- **`check_build()` saying the extension was built from other sources.** A checkout's
  `native/` changed after the build: run `uv sync --reinstall-package solprop-native`.
- **`backend="auto"` refuses.** The message says why: the native package is missing, the
  processor is below the build's baseline, or its features could not be read
  ([installation](../../introduction/installation.md)). A contributor's lightweight setup has
  no native package, so name `backend="numpy_ref"` there.
- **A job stops at `SOLPROP_REQUIRE_DEVICE`.** The installed build isn't the device build the
  job asked for, or it sees no device. A new release replaces a source build, so
  [rebuild it](building-native.md#upgrading-a-source-build), or check the job's allocation.
- **"refuses this problem".** The message names what was refused and which backend runs
  it; pass that backend explicitly.
- **A `RuntimeError` on macOS about `pyfftw`, or a crash or hang.** Two OpenMP runtimes
  were loaded, or would have been. solprop does not use pyFFTW, but a script or another
  package may. Import `solprop_native` at the top of the script,
  before anything imports `pyfftw` (the package raises the error when it detects the
  wrong order); see [Performance tuning](tuning.md#blas-and-vendor-libraries).
- **Slower than expected.** Check `python -m solprop_native`: a `reference` profile, no
  vendor BLAS (`blas_backend` `none`) or a generic architecture (neither
  `native_arch_enabled` nor a `cpu_baseline`) each cost a large factor. A `cpu_baseline`
  of `x86-64-v2` runs about 20 to 25 % longer; if the processor should run x86-64-v3, a
  virtual machine may be hiding a feature, and the `solprop_native._cpu` logger names it
  at INFO when the package is imported. Then see [Performance tuning](tuning.md).

The native backend's place in this repository and its
precompiled wheels were proposed and carried out by
[SEP 29](../../dev/seps/archive/sep-0029-compiled-backend-distribution.md), and
[SEP 43](../../dev/seps/archive/sep-0043-retire-numba-cpu.md) made it a required dependency.
