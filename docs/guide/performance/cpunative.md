# Experimental CPU backend

`cpunative` is an independent C++ backend for double-precision capillary, radial, and selected Cartesian propagation
on CPUs. It uses OpenMP, FFTW, and small real matrix contractions. It is an experiment
alongside `native`; installing it does not change `backend="auto"`.

Select it explicitly in an ordinary `prop_capillary` call or
`Simulation.from_capillary` construction:

```text
simulation = Simulation.from_capillary(
    radius, length, gas, pressure,
    lambda0=wavelength,
    wavelength_limits=(shortest_wavelength, longest_wavelength),
    trange=time_window,
    fwhm=pulse_duration,
    energy=pulse_energy,
    backend="cpunative",
)
output = simulation.run()
```

Keep the input, seed, integrator, tolerances, observations, and output settings the
same when comparing it with `backend="native"`. The experiment's measurements and
qualification scope are recorded in the [experiment report](../../dev/cpunative-experiment.md).
The [comparison summary](../../dev/cpunative-comparison.md) separates full-run results
from short tuning windows and records where each backend is faster.

## Supported configurations

The initial implementation accepts a scalar electric field on a temporal field grid,
with one batch member, a constant capillary radius, and real static modal maps.
Resolved modes and mode averaging use the same execution path. Linear operators and
nonlinear prefactors may be constant or expanded in the medium's longitudinal
parameters; a pressure gradient need not be constant.

The capillary path is not specialised to the standard CTC case's three modes.
Focused Raman checks cover one through eight modes with appreciable fields in every
mode. An experimental modal-pair convolution reduces repeated Raman transforms when
the number of unordered mode pairs is smaller than the quadrature node count and
fits the existing scratch space. Other dimensions retain the node convolution;
both paths use the same causal response.

The supported responses are real cubic or scalar analytic-signal Kerr, one plasma channel, and one molecular
Raman term with its translated damped components. Plasma uses the translated rate
table and running-integral policy, including point-reflection boundaries, exposure
flooring under the band policy, and all three polarisation/current streams. Raman
uses the full causal convolution support. There is no cheaper substitute model.
Both ordinary temporal absorber schemes are supported.

The default exponential `Multistep()` and DOPRI5 `RungeKutta()` use the existing Python
controllers and landing policies. Stages, state updates, error reductions, FSAL, and
Adams history remain resident in C++. The public Adams orders are supported,
with the carrier frame off and transported compensation for varying operators.
Initial shot noise retains the normal seeded launch semantics.

The standard RDW, CTC, and argon blue-shift recipes fit this subset. Supported field statistics are
energy, mean angular frequency, modal and total peak power, inner and outer temporal
FWHM, and on-axis peak intensity. Profile, population, plasma-integration, and embedded
quadrature statistics retain their ordinary scheduling. Saved fields, selective field
retention, and memory/HDF5 output use the usual driver interfaces.

The unchanged `radial_filament_256` recipe is also admitted: a field grid, real QDHT
maps, analytic-signal Kerr, and band plasma in a constant medium. Its ordinary driver
path supports both integrators. Spatial energy, on-axis spectrum and intensity,
peak intensity, beam radius, radial FWHM, and population statistics use the translated
axial-flux measure, including the complex admittance in local flux. Field-sized
reconstruction and reductions execute in C++; the small radial width calculation
uses the shared host implementation. For square real maps with temporal oversampling,
the source contracts retained spectra before padding and after cropping, reducing
matrix work while preserving the response and transform normalisation.
Radial spatial absorption and varying plane
measures are not yet admitted. Both complete standard trajectories pass independent
convergence envelopes; repeated performance comparisons are recorded in the report.

The Cartesian experiment admits the unchanged `kerr_beam_384` recipe: a scalar
complex envelope, instantaneous Kerr, a constant separable operator, and fixed-step
RK4IP. It uses persistent FFTW transforms and five resident
complex volumes. Its ordinary spatial statistics use the translated axial-flux
measure. Temporal oversampling, spatial absorption, varying plane measures, and
other Cartesian responses or integrators remain unsupported. Small source, four-step,
and statistics comparisons pass. The complete standard field also agrees with an
independent Numba run within the fixed rounding bound.

Unsupported combinations raise a named error before the C++ workspace is allocated.
These include envelope grids outside the Cartesian subset, vector polarisation,
batch execution, varying radius, general modal guides, planar propagation, coupled
operators, TDSE, single precision, GPUs, dense-output interpolation, modified noise,
frequency-weighted error control, alternative Adams formulations, carrier frames,
RK4IP outside the Cartesian subset, dispersive Kerr, spectral responses,
cycle-averaged Raman, and multiple plasma or molecular Raman channels. There is no
automatic fallback.

## Build and install

Build from the repository root. The optional package is `solprop-cpunative`, imported
as `solprop_cpunative`; it does not require Kokkos or `solprop-native`.

```sh
uv sync --extra cpunative
# Install both implementations for comparisons:
uv sync --extra cpunative --extra native
```

The build requires Python 3.14, CMake, a C++23 compiler, OpenMP, and double-precision
FFTW with its threading library. GCC 14 remains the build's minimum version, but the
current compiler/OpenMP experiments target GCC 16 and recent Clang/Intel compilers;
GCC 14 qualification is deferred. A recent LLVM
Clang and matching libomp are used on Apple silicon. FFTW on Apple silicon must have
its NEON codelets enabled. `build_info()` reports the loaded FFTW and OpenMP library
paths, so an unintended installation can be identified.

Choose dependencies explicitly when several installations coexist:

```sh
CXX=clang++ CMAKE_ARGS="-DFFTW_ROOT=$FFTW_NEON_PREFIX" uv sync --extra cpunative
```

On Linux, `SOLPROP_CPUNATIVE_BLAS_ROOT` selects the BLAS installation through CMake;
`FFTW_ROOT` selects FFTW. Accelerate is used when found on macOS. Keep the extension,
FFTW, and an OpenMP BLAS on one compatible OpenMP runtime. Do not suppress duplicate
runtime errors. Existing `native` builds on macOS refuse import after pyFFTW when its
wheel could have loaded a different OpenMP runtime; import `solprop_native` first
when comparing both packages in one process, or use fresh processes.

The default `SOLPROP_CPUNATIVE_PROFILE=LOCAL` CMake profile targets the build machine.
`SOLPROP_CPUNATIVE_TARGET` supplies an explicit compute-node target when building
elsewhere. The `DISTRIBUTION` profile uses conservative architecture baselines; it is
build preparation for this experiment, not a claim of released wheel qualification.
Optimisation uses `-O3` and checked IPO. Blanket fast-math, flush-to-zero, and
unprotected changes to stable increments and defects are excluded. The default
uses ordinary libm; opt-in vector-math experiments are described below.

Experimental CMake math profiles isolate proposed optimisations:
`SOLPROP_CPUNATIVE_MATH_PROFILE=ERRNO_FREE` adds `-fno-math-errno`, `NONTRAPPING`
also adds `-fno-trapping-math`, and `FORTRAN_COMPLEX` further adds
`-fcx-fortran-rules`. `PROTECTED` retains the existing default. These flags are
checked by the compiler and still require numerical qualification; the selected
profile is reported by `build_info()`. For compiler-only comparisons,
`SOLPROP_CPUNATIVE_FFTW_THREADING=PTHREADS` permits one common FFTW dependency
without importing another compiler's OpenMP runtime. Ordinary OpenMP builds use
`OPENMP`. Capillary and radial transforms use serial FFTW plans distributed across OpenMP
workers. Cartesian transforms use the schedule described below.

`SOLPROP_CPUNATIVE_RATE_MATH` selects an experimental math provider for ionisation-rate
interpolation only: `LIBM` (default), `SVML` (Intel compiler), `SLEEF` (with
`SLEEF_ROOT`), or `MKL` (with `MKL_ROOT`). SLEEF uses its `u10` functions, MKL uses
HA mode with subnormal handling retained, and SVML is compiled with a one-ULP
maximum-error request in a separate translation unit. These choices change the
math-library accuracy contract and remain experimental. They do not change the
fraction, scan, or integrator arithmetic. `build_info()` records `rate_math`.

## Threads, memory, and measurement

`SOLPROP_CPUNATIVE_THREADS` requests the kernel width within `SOLPROP_THREADS` and the
process CPU budget. Set thread and affinity variables before importing scientific
runtimes. A one-thread workspace has a serial execution path without OpenMP worker
regions. Parallel buffers use contiguous work partitions and padded reduction scratch.

`SOLPROP_FFTW_PLAN=measure` is the default; `estimate` trades construction effort for
quicker setup. Plans persist for the workspace lifetime. The capillary and radial transform paths
execute serial FFTW plans on independent rows, with matching alignment and separate
worker storage. The Cartesian path supports a decomposition into serial plans or
internally threaded volume plans, and uses serial row plans for temporal absorption.
Construction and planning time should be reported separately from a
warmed propagation run.

The default `SOLPROP_CPUNATIVE_CARTESIAN_FFT=auto` selects serial FFTW transforms
over independent two-dimensional planes, followed by batched transforms along the
remaining axis, distributed by OpenMP, when there are at least as many planes as
workers. It retains internally threaded volume transforms for fewer planes.
`split` and `direct` permit controlled comparisons. Both schedules preserve the
same transform dimensions and normalisation; the split schedule requires no extra
state-sized buffer. Backend metadata records the effective schedule.

The memory ledger counts persistent numerical buffers, immutable tables, and reduction
scratch. FFTW's opaque planner allocations, allocator overhead, and temporary owning
host observations are outside that numerical-buffer count; use peak process memory
as well when comparing implementations.

The optional `SOLPROP_CPUNATIVE_PROFILE=1` environment variable records wall-clock
kernel phases in backend metadata. This runtime profiling switch is separate from the
similarly named CMake build profile. Do not compare an instrumented run with an
uninstrumented baseline. `SOLPROP_CPUNATIVE_CONTRACTION=hand` or `blas` permits a
controlled provider experiment; ordinary execution selects `auto`. Eigen is evaluated
in a standalone experiment rather than included as a runtime dependency.
On Linux x86-64 with the ordinary BLAS provider, `auto` uses handwritten contractions
for one mode at eight or more threads, and BLAS for other shapes. Explicit MKL builds
and other platforms retain their vendor provider. Without BLAS, the handwritten
implementation supplies the contractions. Metadata records the effective choice.

The benchmark A/B command accepts independent backend selections:

```sh
uv run --no-sync python -m benchmarks ab \
  --baseline-python .venv/bin/python --candidate-python .venv/bin/python \
  --baseline-backend native --candidate-backend cpunative \
  --case rdw --case ctc --stepper default,rk \
  --threads 1,4 --rounds 3 --repeats 1 --length 0.0025 \
  --out .benchmarks/cpunative/comparison
```

This command changes the measured propagation window, not the standard case recipe.
Its results are experimental bundles separate from existing native history files.
Use fresh interleaved workers, inspect load flags and effective thread counts, and
compare source evaluations and accepted/rejected steps as well as time. A launch-only
window does not establish performance in the developed plasma regime. Neither removing
Kokkos nor selecting more threads guarantees a speedup.
