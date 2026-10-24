# Experimental Julia CPU backend

`backend="julia"` runs scalar capillary and free-space cases with Julia while retaining
solprop's Python simulation builder, translation, controller, statistics and output
handlers. This is a source-checkout experiment on Apple Silicon and Linux x86_64, with Julia 1.13.2,
PythonCall/JuliaCall, AcceleratedKernels and FFTW. Automatic backend selection remains
unchanged. A reviewed SEP is required before production integration.

The branch is shared as a **draft experiment, not for merge**, without CI
qualification. The [checkout README](../../../julia/README.md) provides a runnable
example and a small Julia/native A/B recipe; the
[handoff status](../../../julia/STATUS.md) summarizes measured results, limitations
and the next work. This implementation remains on schema 34 and needs an explicit
port before use with the separate SEP 34 medium-flux work.

## Installation and example

From a source checkout, prepare a private runtime and package environment:

```sh
uv sync --python 3.14 --extra julia
uv run --no-sync python julia/bootstrap.py
```

The executable, depot and caches are confined to ignored directories in that checkout.
The adapter never selects a system Julia or downloads packages during a simulation.
Installing the extra alone is insufficient; wheels and runtime distribution are
deferred. Existing conflicting JuliaCall environment settings produce an explicit error.

In the prepared environment, the following example propagates only a short window of
a constant capillary. The guide length fixes the absorber rates; the single save plane
ends the calculation after 0.1 mm. This optional-runtime example is shown as text:

```text
from solprop.driver import Simulation
from solprop.stepper import RungeKutta

simulation = Simulation.from_capillary(
    125e-6, 0.01, "Ar", 1.0,
    lambda0=800e-9, wavelength_limits=(300e-9, 2e-6),
    trange=100e-15, fwhm=15e-15, energy=1e-6, modes=4,
    plasma=False, raman=False, integrator=RungeKutta(),
    backend="julia", save_planes=[1e-4],
)
output = simulation.run(progress="none")
output.close()
```

For a mode-averaged field calculation, construct the same simulation with `modes=1`,
then select the existing effective-area representation before running:

```text
from dataclasses import replace
from solprop.stats import default_spec

simulation.settings = replace(
    simulation.settings,
    quadrature=None,
    components=None,
    stats=default_spec("mode_averaged"),
)
assert simulation.translate().transform.kind == "mode_averaged"
output = simulation.run(progress="none")
output.close()
```

The single nonlinear node uses the effective-area conversion already supplied by
translation; the Julia stepping, FFT and Kerr code is shared with resolved modal
propagation. Default energy, peak power, spectral centre and pulse-duration statistics
use the shared host routines. Optional `peak_intensity` still reports the mode's
on-axis intensity, not the effective-area intensity sampled by the response. See
[mode averaging](../waveguides/mode-averaging.md) for that distinction. Mode-averaged
envelopes also support the existing GNLSE Kerr, scalar Raman and self-steepening
coefficients, with complex temporal transforms and the same DOPRI/Adams drivers.
Use the ordinary `Simulation.from_gnlse(..., backend="julia")` constructor; the
[GNLSE guide](../waveguides/gnlse.md) describes its physical inputs.

For a basic axisymmetric Kerr beam, use the existing radial frontend:

```text
from solprop.boundaries import absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import FieldGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import default_spec
from solprop.stepper import PIController, RungeKutta

grid = FieldGrid.from_limits(0.01, 800e-9, (300e-9, 2e-6), 200e-15)
simulation = Simulation.from_radial(
    grid, RadialGrid(1e-3, 64), SpatialMedium("Ar", 1.0),
    [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=True)],
    SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=10e-6),
        waist=100e-6,
    ),
    settings=SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-4], dz_init=1e-5, dz_max=1e-4,
        integrator=RungeKutta(), backend="julia",
        absorber=absorber(grid), stats=default_spec("radial"),
    ),
)
output = simulation.run(progress="none")
output.close()
```

The radial path applies the existing translated Hankel matrices through the shared
synthesis/projection pipeline. Spatial absorption is an additive source independent
of the Kerr prefactor, including when the material response is zero. Its constant
spectral matrix uses the same shared construction as the other CPU backends and
is applied under both temporal-absorber schemes. The default spatial statistics
include energy, on-axis spectrum/intensity, peak intensity, beam radius and radial
FWHM. These use the shared host spatial routines; their time is excluded from the
propagation benchmark. See [radial beams](../free-space/beams.md) and the
[Hankel convention](../free-space/hankel.md).

For Cartesian propagation, replace the radial constructor and grid above with
`Simulation.from_free` and `FreeGrid`, and select `default_spec("free")`:

```text
from solprop.grid import FreeGrid

simulation = Simulation.from_free(
    grid, FreeGrid(500e-6, 32, 500e-6, 32), SpatialMedium("Ar", 1.0),
    [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=True)],
    SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=10e-6),
        waist=100e-6,
    ),
    settings=SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-4], dz_init=1e-5, dz_max=1e-4,
        integrator=RungeKutta(), backend="julia",
        absorber=absorber(grid), stats=default_spec("free"),
    ),
)
output = simulation.run(progress="none")
output.close()
```

This path uses planned transverse complex FFTs, followed by chunked real temporal
transforms and the same scalar Kerr operation. The transverse inverse precedes
the temporal inverse: one transverse spectral column alone is not a real temporal
field. Spatial loss remains independent of the material prefactor. The longitudinal
root, evanescent attenuation cap and nonlinear prefactor use the existing separable
coefficient vectors, including the protected split-square subtraction near grazing.
No dense linear or prefactor table is built. See [Cartesian beams](../free-space/beams.md)
and the [kernel contract](../../dev/kernel-contract.md).

Use the existing `FixedSchedule` to prescribe step sizes. `RungeKutta()` uses the
same Python DOPRI5 controller as the other backends. A Julia call forms each
complete trial and its error norm, then Python accepts or rejects it. Rejected trials
leave the accepted state intact. Accepted steps recycle the FSAL source, except when
split-step absorption changes the field.

## Core benchmark extensions

The source prototype also executes the additional numerical leaves used by the
[SEP 30 core suite](../../dev/benchmarking.md):

- **Field plasma:** one translated scalar ionisation channel, its log-rate spline,
  filtered running integrals and the held-current source. Both the collared current
  and collar-derivative stream are transformed, following the existing
  [plasma integration contract](../../dev/seps/archive/sep-0027-held-plasma-current.md).
  Cartesian and line fields evaluate that same current on physical-space chunks
  and combine it before the transverse forward FFT. Their population statistics
  synthesize the accepted field independently of propagation.
- **Longitudinal gradients:** fixed expansion arrays remain in Julia; Python evaluates
  the existing stage bundles and transfers only their small scalar weights. This
  supports the RDW pressure profile without evaluating material models in Julia.
- **Scalar Raman:** one constant-density component, including its translated damped
  response rows, uses a planned doubled-grid convolution. The kernel spectrum is
  constructed once. This is the response used by the hydrogen CtC benchmark.
- **Exponential Adams:** `Multistep()` uses solprop's shared step/order controller with
  resident Julia source history, exponential moments, factored defects and transported
  compensation for a varying linear operator. Rejection preserves the accepted state
  and history. Snapshot rows use the existing host layout. Dense output and comparison
  variants remain outside this experiment.
- **Cartesian envelopes:** `EnvelopeGrid` without envelope harmonics uses planned
  complex temporal transforms. `LowStorageRungeKutta(scheme="rk4ip")` executes the
  existing four-source-pass prescribed-schedule scheme with eight field volumes,
  rather than allocating DOPRI stage volumes. The frozen 384×384 core beam uses this
  route. This does not add envelope plasma or Raman.

All coefficients, response tables, normalisations and profiles come from schema 34.
There are no new physical models or frontend parameters. See the
[Adams method](../numerics/adams.md) and the existing integrator contracts for their
error control and accepted-step boundary conventions.

The standard harness accepts explicit Julia selection:

```sh
uv run --no-sync python -m benchmarks list --set core --backend julia
uv run --no-sync python -m benchmarks run --set core --backend julia --mode quick
```

Quick mode exercises the standard core adaptive tracks and the fixed RK4IP beam.
Select `--track fixed-rk` for the additional frozen DOPRI schedules. A full standard
collection still needs the usual machine profile and numerical reference receipts;
passing a short development screen does not create those receipts. Adaptive suite
windows include their declared save planes and default statistics. Fixed-track
observation is timed separately. `julia/core_screen.py` is a separate propagation-only
development screen; its adaptive timings exclude observation and should not be mixed
with standard-harness timings.

For a lightweight comparison of prepared builds or math providers, use
`julia/core_compare.py`. It reuses the standard suite's A/B worker: each case is
constructed once per worker, warmed once, then reset outside each timed window.
An arms JSON file lists a `name`, `backend`, optional `python` executable and
environment overrides; Julia arms can also select `fft` as `fftw` or `mkl`.
For example, an arms file can compare the current interpreter's two backends:

```text
[
  {"name": "native", "backend": "native"},
  {"name": "julia", "backend": "julia", "fft": "fftw"}
]
```

```sh
PYTHONPATH=. uv run --no-sync python julia/core_compare.py \
  --arms arms.json --widths 1 4 --rounds 3 --repeats 3 \
  --case rdw:fixed-rk --case ctc:fixed-rk \
  --case radial_filament_256:fixed-rk --output .benchmarks/core-comparison
```

Run on an otherwise idle machine. The output separates setup, the first window,
warm repetitions, source counts and final-field differences. Differences are
reported as evidence, not automatically accepted as an accuracy qualification.
`julia/core_profile.py` profiles warmed windows with Julia's sampler and the
Python controller profiler; it accepts the same case and track names. Signal
sampling is restricted to one Julia thread after a threaded sampling failure.
Use `--python-only` to profile the controller and host statistics at wider budgets.

## Supported scope

The prototype supports resolved and mode-averaged scalar modal field grids at constant
radius, including separable longitudinal material profiles, and radial field grids in
a constant medium. Mode-averaged envelopes support Kerr, constant scalar Raman
and the translated self-steepening factor. Cartesian grids support Kerr field
and envelope propagation, and scalar field plasma with DOPRI5 or prescribed
RK4IP steps.
The same transverse-FFT path supports `LineGrid`, uniform in its missing spatial
axis, using the existing per-unit-width contract and `Simulation.from_free`.
Use the [line-beam example](../free-space/beams.md) with `backend="julia"`;
its pulse energy is per unit width, as described in the
[line geometry](../free-space/model.md#the-line-uniform-in-y).
One batch member, dispersion and loss, scalar Kerr with or without THG, hard spectral
masks and both temporal-absorber schemes are supported. Normal save planes, default
spatial statistics and the capillary's default modal, profile, species and quadrature
statistics use shared host routines. At plasma observations, Julia synthesizes the
accepted field once and computes rates, populations and integration-diagnostic
reductions without reevaluating the nonlinear source. Shared host routines apply
the existing population and diagnostic normalizations. This cost is included when
the benchmark window calls for statistics, and observations preserve source counts
and the accepted state.

It refuses planar and general modal geometries, tapered guides, nonseparable
longitudinal tables, coupled walls, vector polarisation, gas-mixture plasma channels,
varying or multiple Raman components, Cartesian Raman, resolved modal or
radial envelopes, envelope plasma/harmonics, dispersive/regional responses, Cartesian Adams, non-Cartesian
low-storage integration, batching, noise and interpolated planes. A refused feature is
named in `BackendError`; there is no fallback to another stepping backend.

The external state remains complex128 in `(batch, modes, pol, omega)` order. Julia owns
its copies in `(omega, modes)` order, with time/frequency contiguous. The engine uses
the existing translated normalisations and response coefficients; it introduces no
new material model. The [Fourier and amplitude conventions](../../dev/conventions.md),
[IP-DOPRI5 scheme](../../dev/stepper.md), and
[kernel contract](../../dev/kernel-contract.md) apply unchanged, at schema 34.

## Threads, memory and limitations

Set `SOLPROP_JULIA_THREADS` before the first Julia backend is constructed. It defaults
to one. The adapter sizes each temporal FFT plan from its real-side sample count:
fewer than 16384 samples across its columns uses one thread; larger plans use the
smaller of the Julia width and four. Columns are modes for a modal transform and
the bounded node chunk for a Cartesian transform. `SOLPROP_FFTW_THREADS` overrides
this choice. BLAS defaults to one thread for mode averaging or Accelerate, and
otherwise to the smaller of the Julia width and four; `SOLPROP_BLAS_THREADS`
overrides it. CPU pointwise loops
require at least 32768 samples per task, while short error reductions stay serial. Numerical loops, FFTs and matrix products execute in successive
phases, with no Python callbacks inside threaded work. The Julia runtime and library
thread pools are process-wide: use separate processes for different thread policies.
The private loader enables Julia's signal handlers, as required for threaded JuliaCall;
interrupt handling consequently follows JuliaCall's behaviour.

`SOLPROP_JULIA_RAMAN_FFT_THREADS` and `SOLPROP_TRANSVERSE_FFTW_THREADS` select
the doubled Raman and transverse plans independently of the modal FFT width.
Raman applies the same 16384-sample threshold to its doubled time grid and node
chunk. Transverse FFTs default to the modal cap. These are experimental adapter
defaults derived from the recorded CPU measurements, not Julia runtime decisions.
Widths are fixed when plans are constructed;
metadata records each plan's requested width separately. The Raman kernel's
one-column construction transform stays serial. Wider pools do not necessarily
help: use the case's actual mode, time and node dimensions when measuring.

The repeated Xeon 192² and 384² envelope-beam comparisons benefit from eighteen
workers for both temporal and transverse FFTs. Their measured settings are
`SOLPROP_JULIA_THREADS=18`, `SOLPROP_FFTW_THREADS=18`,
`SOLPROP_TRANSVERSE_FFTW_THREADS=18` and `SOLPROP_BLAS_THREADS=1`.
The four-worker temporal default is conservative and is slower on those large
cases; see the [benchmark ledger](../../dev/benchmarks.md) for the paired results.

AppleAccelerate is the default on macOS (`SOLPROP_JULIA_BLAS=accelerate`); use
`SOLPROP_BLAS_THREADS=1` for its explicit serial mode. A larger value enables
library-managed parallelism, not an exact worker count. On Linux x86_64,
`SOLPROP_JULIA_BLAS=mkl` selects MKL. Other prepared platforms default to `openblas`. FFTW remains
the transform provider independently of this BLAS choice. `julia/README.md` describes
the separate MKL FFT experiment and the alternating provider benchmark runner.


`SOLPROP_FFTW_PLAN=estimate|measure|patient` selects Julia's planning effort;
`estimate` remains the experimental default. Measured plans spend additional time
at construction choosing algorithms for the actual array layout and thread count.
Planning is excluded from warm propagation timings and recorded separately.
`julia/compare.py --plan measure` applies the same request to all compared backends.
No FFTW wisdom is imported by this experimental Julia adapter.

The memory ledger counts all Julia-owned configuration and work arrays and the host
power-weight table for modal runs. Radial runs additionally own a real
`n_r × n_r` spatial-loss matrix when the collar is enabled. It excludes the Julia
runtime/JIT, opaque FFTW plan storage,
retained shared-driver statistics rows/on-axis cache and host observation/statistics
temporaries.
Those exclusions also appear in the admission estimate. The DOPRI prototype materialises
seven stage arrays; modal and radial paths also hold forward/inverse stage propagators; it is not yet optimised
for minimum memory. Constant-table propagators are reused only for exactly equal
step sizes and rebuilt when an adaptive trial changes the step. Explicit
`propagator_mode="recompute"` disables reuse between trials. Cartesian propagation
uses the seam's recompute policy and only one reusable forward/inverse pair for the
current stage. Its temporal buffers follow the translated node-chunk limit, including
partial final chunks. DOPRI still keeps seven state-sized stages and two transverse scratch volumes. The
Cartesian RK4IP path omits stage and inverse-flow volumes, but currently retains eight
field volumes plus chunk buffers. Adams additionally keeps source histories, moments
and, for gradients, sample states and rates; its buffers are listed in the ledger.
Cartesian Kerr propagation does not call BLAS: its library-sensitive work is FFTs.
Large Cartesian copies and inverse-transform normalization use the shared parallel
traversal. An exactly real longitudinal root has a direct square-root branch;
lossy inputs keep the complex branch, and the compensated subtraction remains
protected. When both physical and in-stage temporal absorber rates are exactly
zero, the source omits their transforms. Active losses and accepted-step split
absorption retain their existing paths. These changes add no field volume; the
strided spatial layout is retained after alternative layouts measured slower.
Modal and radial sources likewise omit additive absorber transforms when all
their translated rates are exactly zero, under either temporal scheme.
Plasma columns with exactly zero ionisation rates and no initial electrons clear
their outputs and omit the current integration. This is an exact zero case, with
no new small-rate cutoff; reused columns and pre-existing electrons have focused
checks against the reference backend.

The [benchmark ledger](../../dev/benchmarks.md#experimental-julia-cpu-prototype)
separates demonstrated correctness from measured performance. The reproducible runner
is `julia/benchmark.py`; it measures ten-step windows, not full-length propagation.
CPU tests are not evidence of GPU correctness or speed. CUDA, AMDGPU, Metal, Float32,
compilation-cache engineering and production packaging are deferred.

For the relevant library interfaces, see
[JuliaCall threading](https://juliapy.github.io/PythonCall.jl/stable/juliacall/),
[AcceleratedKernels loops](https://juliagpu.github.io/AcceleratedKernels.jl/stable/api/foreachindex/)
and [KernelAbstractions](https://juliagpu.github.io/KernelAbstractions.jl/stable/).
