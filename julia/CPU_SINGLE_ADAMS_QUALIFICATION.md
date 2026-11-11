# Short CPU binary32 Adams admission checks

This record fixes the windows, references and criteria before the comparisons in
`tests/backends/julia/test_cpu_single_adams.py`. It applies SEP 33 section 7.3 to
internal CPU binary32 state with binary64 wide arithmetic. It does not enable the
public precision registry or qualify DF32, devices or long-distance survival.

## Windows fixed before comparison

Every window runs from zero to **100 micrometres**, with adaptive stepping,
`dz_init = dz_max = 12.5 micrometres`, no replay, no dense observations and no
noise. The material profiles retain their original full-guide coordinates; they
are not compressed onto the qualification window. These are short entrance
dynamics of the actual configurations, not developed dispersive-wave or
blue-shift spectra.

| Case | Full guide and launch | Temporal grid |
| --- | --- | --- |
| Helium dispersive-wave configuration | 125 µm capillary radius, 1.5 m guide, 0.8 to 0 bar, two resolved HE modes; 275 µJ, 7.5 fs FWHM, 800 nm | Field, 90 nm to 4 µm, 400 fs |
| Argon plasma configuration | 13 µm radius, 0.2 m guide, 1.7 bar, one resolved mode, no modal loss; 3 µJ, 30 fs FWHM, 800 nm | Field, 300 nm to 4 µm, 400 fs |
| Argon envelope plasma configuration | Same argon guide and pulse | Envelope, 300 nm to 4 µm, 400 fs |

These are the physical configurations in SEP 33's survival record, reduced only
in propagated distance. The public capillary constructor supplies material data,
plasma rule, quadrature and boundary defaults, with `plasma=True` explicitly
selecting the declared plasma response (envelope plasma is opt-in). Both methods
and references share one translated scientific descriptor. Changing the method does not rebuild those
data or choose different physical defaults. Absolute tolerance remains the
constructor's value; the compared relative tolerances are `1e-6` and `1e-5`.

## References and metrics

For each case the initial binary64 DOPRI5 reference pair is `rtol=1e-9` and
`1e-10`. Their discrepancy must be no greater than one tenth of each envelope
below, on every scored component. If needed, the predeclared refinement ladder is
`(1e-10, 1e-11)` then `(1e-11, 1e-12)`. If none settles, the test fails without
qualifying either candidate. This is reference refinement, not a change to the
candidate tolerance, window, metric or envelope.

| Metric | Fixed relative envelope |
| --- | --- |
| Whole physical exported field, normwise L2 | `1e-3` |
| Physical energy in each of 32 equal active-frequency bands | `1e-2` |
| Total physical energy | `1e-4` |
| Each reported modal peak-power row | `1e-3` |
| Final ionised fraction | `1e-3` |

Bands use the corrected canonical physical-energy reducer described in
`CPU_SINGLE_QUALIFICATION.md`, evaluated at the actual final coordinate, including
frequency- and position-dependent modal power weights. Only bands with reference
energy at least `1e-4` of the largest are scored, exactly the fixed -40 dB rule.
Band sums must also match total energy under the positive-reduction allowance.
The total-energy and peak-power readouts are the actual session statistics.

The single-gas ionised fraction is `electron_density / density`, using the
statistics' physical gas density at the final coordinate. It is scored only where
the reference exceeds `1e-6`; all excluded values and their reason are recorded.
A structurally absent plasma response is reported as absent. There is no guessed
density, replacement of a zero by a floor, or tolerance fitted to an observed
discrepancy. For another zero-valued reference observable, zero candidate values
have zero error and nonzero values have infinite relative error.

At each tolerance, **every scored Adams error must be at most twice the RK
error**, component by component, and both methods must lie within every fixed
envelope. If an RK error is exactly zero, the corresponding Adams error must also
be exactly zero. Small positive RK errors retain their actual value; there is no
comparison floor. Population and band significance come only from the settled
reference and the fixed rules above.

## Evidence and limits

JUnit properties record every component error, the reference tolerance pair,
population exclusions and each run's actual attempts, acceptances, rejections and
source evaluations. All failed components are collected before the assertion so
one failed metric cannot conceal another. A failed numerical execution remains a
failed test with its work record. No failure is converted to a skip or expected
failure.

Run this module through the managed Julia lane, using the exact-source setup
identity and `SOLPROP_TEST_JULIA=1`. Its default lane uses the existing explicit
opt-in skip reason. Full gates and long survival runs are outside this short
qualification. These three windows alone cannot qualify the complete SEP 33
survival campaign or justify changing the default tolerance floor.

## Recorded CPU result

On 2026-11-11, macOS arm64, Julia 1.13.3/PythonCall 0.9.37 and two Julia threads,
the final typed six-case run produced **two passes and four failures**, with no
skips. It used the published Julia source at `7c534c92` and the same Python
translation at `8d52d715`. Every fixed envelope and the exact twice-RK rule remain
unchanged in the failing test package.

| Case | `rtol` | Result | RK attempts / rejected / source evaluations | Adams attempts / rejected / source evaluations |
| --- | --- | --- | --- | --- |
| Argon field | `1e-6` | All scored criteria pass | 8 / 0 / 56 | 12 / 1 / 13 |
| Argon field | `1e-5` | All scored criteria pass | 8 / 0 / 56 | 10 / 1 / 11 |
| Argon envelope | Both | Cycle-quadrature failure | 0 / 0 / 2 | 0 / 0 / 2 |
| Helium field | Both | Shared translation failure | Not started | Not started |

For the passing argon field rows, the largest componentwise Adams/RK error ratios
were 1.018 and 1.734 respectively (rounded upward for this report). The final
reference fraction was about `6.6409e-4`, above the fixed population threshold.
The complete unrounded errors and selected band indices are in the test's JUnit
properties. Both argon binary64 reference pairs completed in eight capped steps
and agreed exactly. They satisfy the prescribed tolerance-refinement comparison;
because the cap binds, this does not constitute an independent step-size
convergence study.

The helium translation fails while resolving the logarithmic power-weight
endpoint expansion on its unchanged parameter box, before any reference or
candidate session starts ([issue 737](https://github.com/LupoLab/solprop/issues/737)).
The envelope candidates exhaust the unchanged cycle rules, reporting
`PlasmaQuadratureFailure` in channel 1: node 3/sample 336 for RK and node 4/sample
442 for Adams ([issue 738](https://github.com/LupoLab/solprop/issues/738)). The
failure payload does not contain the final two moments or their residual.

An additional control ran the identical argon envelope descriptor and candidate
requests through NumPy's binary32 backend. All four method/tolerance combinations
completed eight accepted attempts with finite exports and preserved input. RK
reported 56 source evaluations; Adams reported its separate charged ledger,
startup 1 and predictor 8, with no rejection or restart. Its ledger is not
relabelled as a physical source counter. The NumPy oracle evaluates envelope
cycle moments and populations in binary64 from the binary32 field before rounding;
this control therefore isolates the failure to Julia's pointwise cycle path,
without yet identifying which error contribution prevents refinement. No physical
quadrature budget was changed: orders 64/128/256/512, `moment_rtol=2e-6` and
`negligible_rate=1e-30` throughout.

The three-case Adams admission is **not qualified** while these failures remain.
The tests keep them as ordinary failures, without expected-failure markers or a
different scientific configuration.
