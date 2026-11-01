# Open questions

These are open, unscheduled questions about solprop as it is, not planned work: a check
not yet made, a choice deferred, or a known limitation accepted for now. Each entry stands
alone: what the question is, where it lives in the code or the documentation, what is
known, and why it is open.

This page is not a list of planned work. Missing functionality and known defects that are
to be fixed are in the [issues](https://github.com/LupoLab/solprop/issues); a major change starts as an
[enhancement proposal](seps/index.md). An entry here moves to an issue if it becomes work
to do, and is removed when it is settled.

Contributions are welcome: an independent check of any entry below — a derivation, a
comparison with a reference, a measurement — settles it.

<!-- Entries are grouped by area; each is a short paragraph with its file or page. -->

## Spatial propagation

**The on-axis spectrum's end bins on the native backend.** The `on_axis_spectrum`
statistic is computed natively on the native backend
(`GeneralWorkspace::spatial_statistics` in `native/src/general.cpp`) and in
Python on the others (`solprop/backends/spatial.py`, through
{func}`~solprop.fourier.spectral_density`). On a field grid the two agree at every
interior bin and differ at the zero-frequency and Nyquist bins: the native kernel does
not double those two in the analytic signal, so that $\delta\omega\sum_k S_k$ is the
axial fluence exactly, while `spectral_density` applies the interior factor at every
bin and, as its docstring states, overcounts those two by a factor of four. Every grid
that `solprop.grid` builds apodises both bins to zero, so no run built through the
public interfaces sees the difference. Open: which definition the statistic should have,
so that the backends also agree on a hand-built grid.

**Transverse components resolved at the entrance only.** With a transverse quadrature and
`components=None`, {func}`~solprop.translate.problem.assemble` decides which Cartesian
field components (x, y or both) the transform carries from each mode's
`principal_component(0.0)`, and refuses an explicitly requested component that is zero
for every mode by the same z = 0 reading (`_components` and `_transform` in
`solprop/translate/problem.py`). The mode protocol's `principal_component(z)` takes a
position, so a mode whose polarisation changed along the guide would gain a component the
transform does not carry. Every mode class today ignores `z`, so no run is affected. Open:
whether the resolution should take the union over the run's length, or the protocol
should declare the component independent of z.

**Thread-count invariance of the transverse FFT pool.** On `numba_cpu` the Cartesian
transverse FFTs run on an FFTW pool of their own, sized by `transverse_fftw_threads=` or
`SOLPROP_TRANSVERSE_FFTW_THREADS` and otherwise by the FFT size rule
(`solprop/backends/numba_cpu/backend.py`); the backend metadata records the size. An FFTW
plan, and so its rounding, can depend on the thread count, as the temporal pool's does.
The tests pin this pool to one thread or check only that the setting is recorded; none
compares a Cartesian run at two transverse thread counts. Open: the rounding bound that
holds between two sizes of this pool, and a test of it (bitwise independence is not
required, [SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)).

**GPU capacity at the largest Cartesian grids.** A Cartesian run's memory is set by its
$n_x n_y$ columns per time sample ([free space](../guide/free-space/model.md), *Memory*). The
backend ledger counts the retained arrays exactly, and `native` also records the device's
free-memory decrease while its FFT plans are built, but cuFFT and rocFFT plan workspace is
opaque and these measurements are not upper bounds. No full-size Cartesian grid (for
example 512 × 512 transverse samples on a few thousand time samples) has been run on a
data-centre GPU, so the largest grid a given device holds is not known. Open: a
measurement of plan usage and peak device memory at the largest intended grid on each
qualified device.

## Translation and materials

**Interior extrema in the scalar table check.** Before a run, `_check_scalar_domain`
(`solprop/translate/problem.py`) checks that each scalar Chebyshev table of a density or
a preionisation fraction stays in its domain, by evaluating every interval at its ends and
at the real roots of its derivative. A root is kept only when `numpy.isreal` holds
exactly, so a stationary point returned with a rounding-size imaginary part is dropped,
and an excursion out of the domain at that point would pass without a refusal. No false
refusal and no missed excursion was seen on the fitted tables checked. Open: whether to
admit roots whose imaginary part is below a tolerance, and which tolerance.

**`with_mixture` and the operator's representation.**
{meth}`Simulation.with_mixture <solprop.driver.simulation.Simulation.with_mixture>`
reuses the existing pressure expansion when the new densities fall within its parameter
ranges. Updated to a constant fill inside those ranges, the run keeps the
`"expansion"` representation of the linear operator, where a fresh build of the same
mixture gives `"constant"`; the two agree to the expansion's tolerance, not bitwise.
Open: whether an update to a constant fill should always rebuild the constant operator,
trading the reuse for bitwise agreement with a fresh build.

**`floor=` beside a constant or profile pressure.**
{meth}`Mixture.fill <solprop.physdata.mixtures.Mixture.fill>` documents `floor` as the
pressure-gradient floor in bar. Beside a scalar pressure or a supplied `ZProfile` it is
accepted and ignored (`_pressure_path` in `solprop/physdata/mixtures.py`). Open: refuse
it by name there, or apply it to a supplied profile.

**A reserved electrons-per-event weight.** The compiled plasma channels
(`PlasmaChannel` in `solprop/translate/responses.py`, `EnvelopePlasmaChannel` in
`solprop/translate/envelope_plasma.py`) carry `electrons_per_event`, and the reference
backend's field-plasma current multiplies by it, but `numba_cpu` and `native` do not carry
it. Validation pins it to 1, so the backends agree. Open: remove the field, or carry it
through the compiled backends with a test, for a model that releases more than one
electron per event.

## Numerical methods

Each of these has a measurable question behind it; none blocks the current solver. The
section numbers refer to the [solver design](stepper.md).

**Error norms for observables far below the pump.** The weak norm of the step controller
(`stepper.md` §3) scores total content, so a weak spectral band — a dispersive wave, a
harmonic — can be under-resolved while the norm is met. The weighted norm is a tool, not
an answer. Open: which weighting controls the error of such an observable, and at what
cost.

**Reject-free efficiency at matched global error.** `reject_free` takes more steps and
buys a smaller error (§3). Open: whether it wins at *matched global* error, and on which
problems.

**Schedule transfer outside the measured ranges.** A calibrated fixed schedule transfers
across the scan cases measured so far (§8). Open: how far it transfers in pressure,
energy and length beyond them, and whether a schedule calibrated on a coarser transverse
grid transfers to the production grid.

**Round-off growth with varying operators.** Open: whether the round-off floor of a long
replayed run grows linearly with the step count when the linear operator varies along z,
and whether dithering the schedule would decorrelate it.

**Save-plane density versus cost.** Every save plane is a step landing (ADR-009). Open:
the cost of dense plane sets, measured against dense output.

**FSAL through a coupled flow.** The first-same-as-last identity and its saving are known
for a coupled flow (§2). Open: whether to enable it, and with which parity statement
between backends.

## Validation

**Systematic differences against Luna's mixture witness.**
`tests/backends/test_luna_mixture_propagation.py` propagates a single gas, the same gas
split into two populations, and a two-channel plasma case over 64 fixed steps from Luna's
own inputs and compares the exit field with Luna.jl's
(`tests/data/luna_mixture_propagation.json`). The relative L2 differences are 5.25e-9
(single and split) and 1.12e-8 (plasma), the same on every run. The test admits them
under a predeclared rounding budget, $64\,n_{to}\,n_{\rm channels}\cdot 7\cdot{\rm
steps}\cdot\varepsilon$ (1.30e-8 and 2.61e-8), which is an allowance rather than a count
of operations, so the margin is 2.3–2.5. Open: what the systematic difference is —
summation order, an input conversion or a small model difference — so that the bound can
be derived rather than budgeted.

**NaN in free-space reference runs under NumPy 2.3 and 2.4.** With every other
dependency at its floor, NumPy 2.3.2, 2.3.5 and 2.4.0 make the `numpy_ref` free-space
runs (line, planar and radial) return NaN fields and fail about 200 tests, among them
`tests/backends/test_line_free.py` and `tests/backends/test_free_nonlinear.py`, with
overflow warnings on the way; NumPy 2.5.0 passes them, independently of the Numba
version. The NumPy floor is held at 2.5 ([dependencies](dependencies.md)) until the
cause is found. Open: which operation's behaviour changed between 2.4 and 2.5, and
whether the code relies on it by accident.
