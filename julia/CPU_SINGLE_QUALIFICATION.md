# CPU binary32 session qualification

`tests/backends/julia/test_cpu_single.py` exercises the internal `JuliaBackend`
through the managed PythonCall bridge. State, source and observation work use
binary32; the controller, coordinates and linear phase use binary64. Physical
exports remain complex128. These tests do not enable public registry selection,
DF32 phase arithmetic or a device, and do not replace those qualifications.

## Short execution matrix

The material witnesses use the existing scalar two-species test mixture, with two
Raman components, independent instantaneous or cycle-averaged plasma channels,
nonzero preionisation and Kerr response. The material constants define synthetic
test functions, not fitted atomic data. The march is one micrometre, with eight
candidate steps. Both field and envelope grids cover averaged, multimode, radial,
Cartesian and line routes. Cartesian witnesses additionally exercise Tsit5,
exponential Adams and fixed low-storage RK4IP beside DOPRI5. Modal witnesses also
carry independent density/linewidth profiles and an uncoupled taper.
Separate eight-micrometre witnesses exercise Kerr-only, Kerr/Raman and plasma
sources.
Physical peak power or intensity and spatial beam widths are compared on every
route at both temporal grid kinds.

Separate adaptive cases use the two accepted relative tolerances, `1e-6` and
`1e-5`, on the averaged field and envelope routes with DOPRI5, Tsit5 and Adams.
Their maximum step is one eighth of the short interval. This is a qualification
window, not a changed physics or controller default. The actual accepted and
rejected counts remain observable; no equality of adaptive histories is required.

The service matrix covers an eight-step uniform split collar with RK, Adams and
low-storage stepping; sixteen dense soliton observation planes with DOPRI5 or
Adams, periodic energy and temporal monitoring; and all three scalar spatial
monitor maps. Tsit5 has no continuous extension and is tested only with landed
observations. Noise is absent throughout. The tests check the actual execution
precision, eight bytes
per retained complex state entry, immutable launch data, finite physical exports,
hard-mask zeros, source-call accounting and exact observation/periodic cadence.

## Accuracy criteria fixed before comparison

The numerical acceptance uses the already accepted SEP 33 survival metrics,
described in [SEP 33](../docs/dev/seps/archive/sep-0033-single-precision-mlx.md),
section 7.3 and the Part II survival record. SEP 25 distinguishes these accuracy
envelopes from matched-history rounding bounds. Ordinary replay, and especially
variable-order Adams, is not assumed to have the same discrete history as the
reference method.

| Quantity | Candidate accuracy envelope | Independent reference settlement |
| --- | --- | --- |
| Entire saved field, relative L2 | `1e-3` | `1e-9` |
| Physical energy in 32 equal-frequency active bands above -40 dB | `1e-2` relative | `1e-4` relative |
| Total physical energy | `1e-4` relative | `1e-8` relative |
| Peak power or intensity; spatial beam radius and FWHM | `1e-3` relative | `1e-8` relative |
| Species ionised fractions and electron densities | `1e-3` relative | `1e-8` relative |

Each material reference uses the identical binary64 translated descriptor and
grid, with independent 16-step and 32-step DOPRI5 marches. The population fixtures
start at an ionised fraction of at least 0.03, above SEP 33's `1e-6` signal floor;
there is no discrepancy-dependent exclusion. All reported source-rate rows must
be finite, but this matrix does not assign a separate rate-accuracy envelope.
Dense soliton observations instead use landed 32-step and 64-step references at
the same sixteen positions. No local `rtol` is used as a global error estimate.
These fixed-step refinements qualify the short paths; they do not reproduce
SEP 33's entire survival campaign or its separate Adams-versus-RK twice-error
admission rule. That rule remains an additional prerequisite for public admission.

The band metric uses physical energy at the final observation coordinate. Modal
bands use the oracle's `stats._poynting_statistics` with the represented
`power_weight_at(problem, z)`, including its frequency and position dependence.
Spatial bands use `backends.spatial.collect_spatial` on frequency-masked fields,
with the actual `FluxPlan.at(z)` and transverse spectral measure. This reducer
already owns the axial admittance and analytic-signal endpoint weights. No new
conversion constants or approximation of the medium are introduced. Units are
joules, or joules per metre on a line. The 32 half-open windows partition the
active band; the last window includes its final bin. The inactive field is zero.

Every candidate and reference checks that the bands sum to the canonical total
energy. For these finite, normal, nonnegative-energy fixtures, with `N` complex
field entries, two evaluations of the positive reduction differ by at most
`2*gamma_(2*N+64)` times the total. Each product path uses fewer than 16 real
operations, at most `2*N` additions cover polarisation, column and chunk
reductions, and 32 more cover the bands. The bound includes both independent
reduction orders and is required to remain below `1e-8`. Actual binary64 session
energy is compared independently with the same bound; binary32 session energy
uses the already declared `1e-4` physical-energy accuracy envelope. These checks
run on both temporal grids and all five routes, including the modal profile and
taper cases.

This corrects the original raw squared-amplitude band metric
([issue 734](https://github.com/LupoLab/solprop/issues/734)). SEP 33 Part I requires
spectral energy; its historical Part II table describes a sum of squared
amplitudes across columns. That historical measurement is not reinterpreted as
physical energy here. The current tests use the accepted Part I definition and
leave prior evidence unchanged.

For the source-free uniform split test, the independent solution is
`a(z) = exp(D*z - alpha*z/2)*a(0)`: a scalar temporal multiplier commutes with the
lossless spectral operator. The field and energy envelopes above are checked at
every delivered plane, including the exact physical energy decay. A quiet monitor
must not omit the required split finish.

Spatial launch-monitor face shares use the existing transverse bridge test's
conditional coarse-transform and positive-reduction model, recomputed with
binary32 unit roundoff `u = 2^-24`:

`8*gamma_(64*(n_columns+n_omega))`, where `gamma_k = k*u/(1-k*u)`.

The test requires this positive bound to remain below `0.01`. Its absolute scale
is the positive total launch energy, including weak faces. The model includes the
qualified finite FFT/twiddle error hypothesis and launch normalization; counting
operations alone is not a proof of a vendor FFT's accuracy. No relative accuracy
for a tiny face or monitor threshold-decision equivalence is inferred.

## Running and interpreting the checks

Publish a managed CPU setup for the exact source, then set `SOLPROP_TEST_JULIA=1`,
`SOLPROP_TEST_JULIA_SETUP_ID` and `SOLPROP_JULIA_DEPOT` as in the Julia runtime lane.
Run only the directly affected module during development:

```sh
uv run --no-sync pytest tests/backends/julia/test_cpu_single.py
```

With the opt-in absent, the existing declared runtime-lane skip reason applies.
An enabled lane must execute the tests; a missing setup identity is an assertion
failure. A native installation is not needed for these Julia-versus-NumPy checks.
No cold-start timings from this module are benchmark results. Full gates and
coverage remain CI responsibilities.

The final qualification record must state the source revision, setup identity,
thread width, executed test count and any failures. Passing this small matrix
establishes those short CPU paths only; it does not claim long-distance survival,
all conditioning regimes, changed noise admission, DF32 or GPU qualification.
