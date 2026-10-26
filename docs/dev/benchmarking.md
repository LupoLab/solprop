# Running the benchmark suite

Run the four-case core suite with `uv run poe bench`. It measures bounded windows
of full-resolution problems and prints warm time, startup time, peak memory and
check status. Numerical reference preparation is optional for exploratory timing.
The runner is specified by
[SEP 30](seps/archive/sep-0030-standard-benchmark-suite.md). The rules it applies,
the measurement windows and what a performance claim reports are in
[Benchmarks](benchmarks.md); this page is the runner's manual.

```sh
uv run poe bench --dry-run
uv run poe bench --out .benchmarks/suite/runs/first-look
uv run poe bench --case rdw --threads 1
```

The default uses the native CPU backend and requires it to be installed in the
environment. For a small first check without the native backend, use
`uv run poe bench --case rdw --backend numpy_ref --threads 1`.
The 384-square Kerr case is a substantial memory workload: its conservative
admission budget is about 36 GiB. The controller records a refusal if available
memory is insufficient; select individual cases on smaller machines.

`poe bench` accepts the same arguments as `python -m benchmarks`.
Omitting a subcommand means `run`. No profile, baseline environment or reference
preparation is required for a quick run. Missing references are labelled
`not-qualified` and do not prevent timing.

## Select a measurement

Planning imports metadata only; it does not import NumPy, initialise a backend,
translate a simulation or prepare references:

```sh
uv run python -m benchmarks list
uv run python -m benchmarks run --dry-run --set core
uv run python -m benchmarks list --all
```

The table includes conservative memory budgets and reference receipt status.
Planning checks reference metadata compatibility only; the worker verifies array
hashes and numerical results. Missing cost measurements are shown as unknown.

The default selection is `core`, `quick`, `native`, at the available production
width. Declare a machine profile before a standard comparison. The alias must be
suitable for a public report; the width must fit the current allocation:

```sh
uv run python -m benchmarks profile .benchmarks/suite/profiles/workstation.json \
    --alias workstation --production 8
uv run python -m benchmarks run --set core \
    --machine-profile .benchmarks/suite/profiles/workstation.json
```

Use `--case` or `--set` repeatedly to select a deduplicated union. `--all` selects
all sets, including visible pending definitions. A pending definition produces an
explicit exclusion; it never becomes a successful measurement. `--threads` accepts
positive integers and `production`, for example `1,production`. Quick mode defaults
to production; standard mode defaults to both widths. Unsupported tracks and widths
are reported explicitly.

| Core case | Headline measurement | Fixed diagnostic |
|---|---|---|
| Four-mode RDW | 0.02–0.04 m, following a 0–0.02 m lead-in | Ten 2 mm entrance steps |
| Six-mode CtC | 0.00017–0.00034 m, following its lead-in | Ten 17 micrometre entrance steps |
| 384-square Kerr beam | Fixed RK4IP only, one lead-in step and ten 0.5 micrometre steps | The same row |
| 256-node radial filament | 0–0.03 m, four observation planes | Ten frozen RK pilot steps, ending at 0.274524 mm |

RDW, CtC and radial headline rows each use default `Multistep()` and `RungeKutta()`.
The original guides, gas-flow profiles and grids remain intact. The measurement
window does not redefine a pressure-gradient guide's endpoint.
The radial diagnostic uses the first ten accepted steps of the NumPy default-RK
pilot at `rtol=1e-6`; its edges are stored with the registry and included in the
measurement identity. It does not run a pilot during ordinary collection.

The `modal-capillary` and `mode-averaged` sets add nine bounded cases. Both
adaptive methods and a ten-step fixed RK diagnostic are available on NumPy,
Numba and native CPU. Select Numba with `--backend numba_cpu`; its first window
includes JIT compilation or cache loading in startup, before warm timing.
The existing on-disk Numba cache is recorded as uncontrolled, so startup should
not be presented as a reproducible cold compilation measurement.

```sh
uv run poe bench run --set mode-averaged --backend numba_cpu
uv run poe bench run --case argon_blueshift --track fixed-rk --threads 1
```

| Additional case | Adaptive entrance interval | Fixed diagnostic |
|---|---|---|
| One resolved mode RDW | 0–39.1377 mm | Ten frozen pilot steps |
| Reduced one-mode CtC, 2 ps (revision 3) | 0–1.31141 mm | Ten 120 micrometre steps |
| Resolved argon blueshift | 0–2.42717 mm | Ten frozen pilot steps |
| Resolved hydrogen Raman, 6 ps | 0–13.2409 mm | Ten frozen pilot steps |
| Averaged helium RDW | 0–39.3310 mm | Ten frozen pilot steps |
| Averaged argon blueshift | 0–2.43336 mm | Ten frozen pilot steps |
| Averaged hydrogen Raman, 6 ps | 0–13.2409 mm | Ten frozen pilot steps |
| Ideal N=3 GNLSE soliton | 0–90.0316 mm | Ten frozen pilot steps |
| Raman/shock GNLSE | 0–0.945275 mm | Ten frozen pilot steps |

These windows end at the tenth accepted step of independent NumPy RK pilots at
`rtol=1e-6`, `atol=1e-10`. The registry retains the full precision of every edge.
Each adaptive window saves its endpoint with ordinary statistics. The full guides,
including the helium pressure profile over 1.5 m, are preserved. The averaged
cases use the effective-area route; the resolved cases keep transverse quadrature
even with one mode. The two hydrogen cases retain the explicitly named 6 ps grids.
The ideal soliton retains the lesson's full-period guide but propagates only the
short entrance interval. These nine adaptive cases have a reviewed relative field
ceiling of `1e-4`, independent of the solver tolerance. Preparing or selecting
reference receipts remains separate from ordinary timing.

The `free-space` set also adds these windows. The six adaptive additions use
the same ten-step NumPy pilot protocol, with `atol=0`; both default methods and
the frozen fixed-RK diagnostic are available. Kerr remains fixed-RK4IP only.

| Additional case | Measured interval | Fixed diagnostic |
|---|---|---|
| Radial filament, 512 nodes | 0–274.524 micrometres | Ten frozen pilot steps |
| Cartesian filament, 128-square | 0–225.110 micrometres | Ten frozen pilot steps |
| Cartesian filament, 256-square | 0–225.111 micrometres | Ten frozen pilot steps |
| Line, 128 points | 0–10.8716 mm | Ten frozen pilot steps |
| Planar, two modes by 32 points | 0–6.79474 mm | Ten frozen pilot steps |
| Radial absorber, 64 nodes | 0–10.8716 mm | Ten frozen pilot steps |
| Kerr beam, 192-square | 0.5–5.5 micrometres | Ten 0.5 micrometre steps |
| Kerr beam, 768-square | 0.5–3 micrometres | Five 0.5 micrometre steps |

Kerr windows follow one lead-in step. The 768-square case excludes one-thread
selection unless `--diagnostic` is explicit. Memory admission still applies:
large Cartesian cases can be refused before construction on a smaller machine.
Line, planar and radial-absorber cases have reviewed `1e-4` relative field
ceilings. The larger filament additions remain unqualified for accuracy.

The `advanced` set preserves the source guides, coherent wall models and taper
corners. Each case supports both adaptive methods and a fixed RK diagnostic:

| Additional case | Adaptive entrance endpoint | Fixed diagnostic |
|---|---:|---|
| Scalar coherent wall | 200 micrometres | Ten 20 micrometre steps |
| Vector TE/TM coherent wall | 200 micrometres | Ten 20 micrometre steps |
| Taper without wall | 137.401 micrometres | Ten 20 micrometre steps |
| Taper with scalar wall | 137.889 micrometres | Ten 20 micrometre steps |
| General step-index HE11 pair | 135.895 micrometres | Ten frozen pilot steps |
| Linear two-channel GRIN | 19.2601 micrometres | Ten frozen pilot steps |
| Nonlinear four-channel GRIN | 620.057 micrometres | Ten frozen pilot steps |
| Moving solid guide | 21.8692 micrometres | Ten frozen pilot steps |

The linear GRIN case keeps the continuous-wave two-channel launch and its
0.784680 mm self-imaging guide; its temporal grid retains the source's 1 mm
length. Its ten accepted pilot steps include three rejected trials. Nonlinear
GRIN retains input-noise seed 41. The moving guide retains its 5 mm taper corner
in the unexecuted schedule tail. Weak-absorber and grazing-incidence warnings
from the wall tutorials are declared by category and message. General-guide
construction can dominate startup, independently of the short measured interval.
All eight advanced cases have reviewed `1e-4` relative field ceilings, with
reference and policy revisions independent of the workload.

The `backend-exercise` set covers particular execution paths on small problems:

| Exercise | Measured entrance endpoint | Tracks and observations |
|---|---:|---|
| Vector rotational Raman | 0.291590 micrometres | Both adaptive methods; ten fixed pilot steps |
| Envelope plasma | 8 micrometres | Both adaptive methods; four original 2 micrometre fixed steps |
| Helium/argon pressure gradient | 21.8703 mm | Both adaptive methods; ten fixed pilot steps |
| Dispersive Kerr | 15.2189 mm | Both adaptive methods; ten fixed pilot steps |
| Band/trapezoid plasma integration | 135.895 micrometres | Two cases, both adaptive methods; ten fixed pilot steps |
| Batch | 1 mm | Two distinct GNLSE members, ten fixed RK steps |
| Physical noise | 135.895 micrometres | Both adaptive methods; ten fixed pilot steps; seed 13 |
| Dense output | 1.45795 mm | Both adaptive methods; eight interpolated observations |
| Split-step absorber | 135.895 micrometres | Both adaptive methods; ten fixed pilot steps |
| Ordinary statistics | 135.895 micrometres | Both adaptive methods; ten fixed pilot steps |
| Far-field statistics | 863.801 micrometres | Both adaptive methods; eight observations with only the exit field retained; ten fixed pilot steps |

Vector Raman uses the current optmat rotational hydrogen response with vibration
disabled, retaining the circular pump and delayed oblique probe launch. Its
low-density linewidth warning is recorded. The envelope-plasma guide ends after
four source steps, so that shorter interval is retained. The other new windows
use ten accepted default-RK pilot steps. Dense output visits intermediate planes
without forcing the integrator to land there. Far-field observations include
apertures, spectral regions and periodic absorbed power. Fixed diagnostics time
propagation alone and report observation costs separately.

The batch workload uses the actual backend batch implementation, retaining each
member's own statistics, source counts and field check. Reset reuses member
buffers and FFT plans. These exercises can be timed before reference review;
their numerical evidence is collected separately through `qualify`. The eleven
adaptive exercises have reviewed `1e-4` relative field ceilings. The fixed batch
check remains a matched-method diagnostic.

## Timing boundaries and accounting

These boundaries implement the
[measurement windows](benchmarks.md#measurement-windows) of the benchmark policy.
Each sample runs in a fresh process. Its thread environment is set before scientific
imports, with `solprop_native` loaded before pyFFTW. Budget policies clear inherited Kokkos
pool overrides and the FFT-size override. `all-pools-N` explicitly forces the pools;
`as-installed` preserves the environment. This keeps an automatic-policy measurement
separate from a forced-width measurement. The backend and FFT plans are constructed
once. After a first complete window, the worker calls `set_initial`, repeats the
bounded lead-in outside warm timing, and measures the declared window again.

Adaptive timing includes propagation, declared observations, ordinary statistics
and in-memory output. Fixed timing includes propagation; the final observation and
statistics are measured separately. Fresh-process startup is measured by the parent
through receipt of the first completed window. Process startup and warm propagation
answer different questions and are reported separately.

`source_evaluations` counts actual Runge–Kutta source dispatches since the most recent
`set_initial`. Rejected trials contribute work; a reused stage contributes no new
evaluation. NumPy may therefore report more evaluations than a backend that reuses
DOPRI5's last stage. Adams retains its existing response-call ledger, and low-storage
methods retain their existing evaluation counters. These are execution facts, not
changes to the simulation or kernel-contract schemas.

The worker reports lifetime peak RSS after measurement and before reference loading.
Allocation-ledger bytes are reported separately. Unsupported telemetry carries an
unavailable reason; a missing counter is not zero. Admission estimates are conservative
provisional budgets until compatible empirical receipts are qualified. An unknown
estimate requires `--allow-unestimated` and an explicit `--memory-limit` in bytes;
such a collection is exploratory.
Each attempt also reports `controller_cpu_seconds`: CPU time consumed by the parent
while launching, monitoring and reading the worker. It includes process sampling
and receipt handling; it is separate from the child's warm propagation time.

## Collection, recovery and comparison

Quick mode takes one sample per configuration and has no adoption verdict or MAD
estimate. Standard mode starts with five samples per configuration, uses balanced
shuffled rounds, and alternates adjacent baseline/candidate order. A short separate
pilot can select up to ten individually timed repetitions; the same count is frozen
for both arms. No repetition is selected merely because it was fastest.

For a first look, collect the same selection before and after a change, then compare
the saved bundles. Quick comparison prints descriptive percentage changes even
without qualified references or a clean baseline; it does not approve adoption:

```sh
uv run poe bench compare .benchmarks/suite/runs/before .benchmarks/suite/runs/after
```

Use `compare --json` for the detailed machine-readable result, including resource
flags. For a controlled paired comparison, prepare both Python environments first:

```sh
uv run python -m benchmarks run --mode standard --set core \
    --machine-profile .benchmarks/suite/profiles/workstation.json \
    --baseline-python /path/to/baseline/bin/python \
    --candidate-python /path/to/candidate/bin/python \
    --out .benchmarks/suite/runs/comparison
uv run python -m benchmarks compare .benchmarks/suite/runs/comparison
```

The controller admits one measurement worker at a time. A guard miss defers the
whole pair once in standard mode, then records contention if it persists. Contended
samples remain in all-sample tables. Clean paired summaries omit both members if
either is contended. Unknown guard confidence cannot establish a clean comparison.
`--force` skips waiting and explicitly makes the collection exploratory.

Run bundles live below the ignored `.benchmarks/suite/` directory unless `--out`
overrides it. They contain a manifest, immutable attempt shards, raw worker logs,
and derived `samples.csv`, `summary.csv` and `accuracy.csv` tables. A run lock prevents
concurrent writers. Temporary partial files do not count as completed attempts.
Use `run --resume RESULTS` to restore the saved selection and options and retry an
interrupted run; completed sample identities are skipped and previous failures remain
on disk. Worker files left by a canceled collector remain intact; a retry uses a
fresh attempt number. An explicit conflicting option refuses resume.
Changed harness, environment, selection or machine identity refuses resume.

Comparison requires the complete affected selection and compatible workload,
resolved method, measurement and profile identities. Build identities may differ.
Adaptive and fixed tracks, and different widths, have separate aggregates. Standard
adoption applies the aggregate rule of [the benchmark rules](benchmarks.md#rules) and
additionally requires qualified numerical references, sufficient clean paired rounds
and a gain exceeding measured adjacent-round repeatability. More than 20% contention,
insufficient adjacent clean rounds, or missing qualification leaves the result
unresolved. Missing rows are never removed to improve the aggregate. Peak-memory
and startup regressions have separate advisory flags.

Use `--allow-accuracy-failure` when timing should continue despite failed numerical
checks. It preserves each failed check and its measured error, completes timing
rounds, and labels the entire collection exploratory. Successful execution then
exits zero even if an accuracy check failed; crashes, timeouts and memory refusals
still fail. This option does not turn a failed reference check into a pass, and an
override collection cannot establish adoption:

```sh
uv run python -m benchmarks run --case rdw --allow-accuracy-failure \
    --out .benchmarks/suite/runs/rdw-exploratory
```

## Fast A/B comparison while iterating

A standard paired comparison is a final check, not an iteration tool. Each sample is
a fresh process that imports, constructs the case, runs a first window and only then
its warm repetitions; standard mode adds paired pilots, a counter guard before every
pair and five to eleven rounds. Construction dominates: in a standard run of the
advanced set on a 36-core workstation it took about 70 % of the collection time, the
first and timed windows about 9 % each, and guards, process start and validation the
rest. The whole set at two widths took about five hours, though most windows last
milliseconds to seconds.

While developing a performance change, time only the cases it should affect with the
`ab` command, and keep the standard comparison for the settled change:

```sh
uv run python -m benchmarks ab --case step_index_pair --case taper_wall \
    --track adaptive-rk --threads 1 \
    --baseline-python /path/to/baseline/bin/python \
    --candidate-python /path/to/candidate/bin/python
```

Selection (`--case`, `--set`, `--all`, `--track`, `--threads`, `--backend`,
`--machine-profile`) works as for `run`, defaulting like a quick run to the adaptive
tracks at the production width. Cases, measurement windows and lead-ins come from the
registry unchanged. Each round starts one worker per build and thread width. The
worker constructs each selected case once, runs one untimed warm-up window, then
times `--repeats` windows (default 5), each after the usual reset and lead-in. There
are `--rounds` rounds (default 3), and the build that runs first alternates between
rounds. A selection of a few cases therefore costs a few constructions per build
instead of one per sample: four advanced cases at one track compare in a few minutes.

The command prints, per configuration, each build's median window time, the median
over rounds of the paired candidate/baseline ratio of per-round medians (below 1 means
the candidate is faster), the range of those ratios over rounds, whether both builds
took the same accepted and rejected steps, and the relative L2 difference between the
two builds' retained fields from the first round. A geometric mean of the ratios
follows. Passing the same interpreter for both builds gives an A/A control, whose
ratio range shows the noise to expect on that machine.

Results go to `.benchmarks/suite/ab/` unless `--out` names a new directory:
`plan.json`, `results.json` with every timed window and the runtime identity of each
worker, the raw worker logs and the saved fields. The guard takes one one-second
reading before each worker and labels a busy machine, but never waits. There is no
pilot, reference check, memory sampling, resume or adoption verdict; the batch
exercise is not supported. Use a quiet machine, and confirm a promising result with a
standard comparison.

## Numerical qualification

Reference preparation is explicit and runs outside ordinary timing collection:

```sh
uv run python -m benchmarks qualify --case rdw --threads 1 \
    --out .benchmarks/suite/references/rdw-review
uv run python -m benchmarks run --case rdw \
    --references .benchmarks/suite/references/rdw-review
```

Qualification is a one-time preparation step for each reference revision. Routine
timing validates the first successful sample for each arm/configuration after the
timer and memory receipt. Later samples check finite output and the validated shape,
and link to the original validation rather than repeating its error measurements.
A changed reference/policy or a failed sample requires a fresh check. Timing does not
run a reference convergence campaign. The qualification path retains tight RK at relative tolerance `1e-10`, a
`1e-9` refinement witness, a matched-method NumPy result, a candidate result, reset
evidence and array hashes. It evaluates the worst-plane relative field error in the
requested band and preserves the full-band error. Fixed methods use a same-schedule
NumPy reference. Qualification runs also retain fresh-versus-reset fields, work
counts, observed controller histories, statistics and noise state. RDW retains its
weak-bin and 100–150 nm band metrics; CtC retains its weak-bin metric. Modal energy
differences use the ordinary Parseval weights and are diagnostic only.

An empty mode can have an undefined spectral centroid, and a continuous wave
can have an undefined pulse width. Reset evidence records these statistic masks
and compares the finite entries separately. Non-finite fields, histories or
infinities remain errors.

The benchmark's solver tolerance remains `rtol=1e-6`. Its separate accuracy
envelope bounds the measured field and retained-band errors against the reference:

| Adaptive case | Largest pilot error over both integrators | Reviewed ceiling |
|---|---:|---:|
| RDW | 2.045 × 10⁻⁵ (UV band) | 1 × 10⁻⁴ |
| CtC | 2.055 × 10⁻⁶ (weak bins) | 1 × 10⁻⁵ |
| Radial 256 | 1.206 × 10⁻⁶ (worst-plane field) | 1 × 10⁻⁴ |
| Nine added modal/GNLSE cases | 6.013 × 10⁻⁷ (relative field) | 1 × 10⁻⁴ |

These ceilings reuse target levels from the source campaigns with headroom for
ordinary rounding and solver variation; they are not fitted to timing results.
The approved defaults match exact measurement identities. A changed workload must
be reviewed again. Other measurements without a reviewed positive envelope remain
`not-qualified`. The explicit `--accuracy-bound` option supplies a reviewed
replacement ceiling. Adaptive field and band ceilings must remain at least ten
times their respective measured reference refinement floors.
An envelope pass alone does not qualify adoption. After reviewing the retained
reset fields, controller histories, statistics and work counts, use
`qualify --recheck ... --reviewed` to mark that reference eligible for standard
comparison. This checks that all four reference runs succeeded and retained reset
evidence, with reset fields inside the same approved field ceiling. History and
statistics need their case-specific numerical interpretation; this option confirms
that review without imposing identical adaptive trajectories. A changed policy does
not inherit a previous review automatically. Ordinary timing needs neither this
review nor a reference. Clean-pair, warning and resource checks remain part of each
standard comparison.

### Revise an accuracy policy or reference

Workload, reference, accuracy-policy and result-schema revisions are independent.
An improved reference or a revised accuracy limit does not rename an unchanged
benchmark case. Receipts are immutable, carry the source/build evidence, and link
to the receipt they supersede with a revision note.

To reassess retained evidence with a reviewed ceiling, without executing another
simulation:

```sh
uv run python -m benchmarks qualify --case rdw \
    --recheck .benchmarks/suite/references/rdw-v1 --accuracy-bound 2e-4 \
    --revision-note "Reviewed margin after numerical investigation" \
    --out .benchmarks/suite/references/rdw-v2
```

The ceiling in this example illustrates the command; it is not an additional
approved RDW default. Rechecking verifies artifact hashes, reevaluates retained
errors and refinement floors, and writes a new receipt. It cannot fix an incorrect
reference field. After fixing a reference implementation, prepare fresh evidence:

```sh
uv run python -m benchmarks qualify --case rdw \
    --supersedes .benchmarks/suite/references/rdw-v1 \
    --revision-note "Recomputed after correcting the reference implementation" \
    --out .benchmarks/suite/references/rdw-v2
```

Replacement reference revisions advance automatically; a changed ceiling also
advances the policy revision. Explicit `--reference-revision` and `--policy-revision`
values may advance these further. Both comparison arms must be validated against
the same receipt to support adoption. Old bundles remain historical evidence under
their original policy; changing a ceiling does not rewrite their accuracy results.
Revalidation against a replacement needs retained output fields or a new collection.

Warnings are captured with phase, category, message and occurrence count. A case
may declare a narrow expected pattern; undeclared warnings prevent adoption.
The Kerr recipe retains its source's Malitson UV extrapolation warning explicitly.

## Native source coverage

Coverage is an explicit developer diagnostic. It reports how much of the compiled
native CPU backend the selected benchmark cases execute; it is not run by ordinary
timing commands and has no automatic notifications or numerical target.

Prepare a separate Clang CPU build of `native/` with
`SOLPROP_NATIVE_ENABLE_COVERAGE=ON` as described in `native/README.md`. Use LLVM tools from
the same compiler release. Keep the ordinary timing environment separate: timing
commands refuse an instrumented build. Then, from the repository root:

```sh
uv run python -m benchmarks coverage --set backend-exercise \
    --python /path/to/coverage-environment/bin/python \
    --native-root native \
    --llvm-bin /path/to/matching/llvm/bin > coverage.log
```

The command prints a redirectable line/function/branch summary and the local report
path. Defaults are the core set and one thread. All admitted tracks of each case
are included; `--case`, `--set`, `--all`, `--track`, `--threads` and `--diagnostic`
select a narrower or broader diagnostic. Memory admission and timeouts still apply.
Failed, refused and excluded configurations remain visible.

Each configuration runs once, including construction, its declared lead-in and
observations. There are no warmups, calibration loops or timing samples. A separate
second execution collects Kokkos Tools labels. Capability tags describe intended
workloads; observed labels show which named operations executed. Neither implies
that every branch of a capability was exercised.

Reports default to ignored `.benchmarks/suite/coverage/`; `--out` chooses another
local directory. The bundle contains per-case, per-set and union HTML reports,
line/function/branch CSV views, labels, raw profiles and a copy of the exact binary.
They are local development artifacts, never committed automatically. Delete them
when no longer useful. Generated benchmark or coverage results do not belong in
the repository; only definitions, tests, policies and concise implementation notes
are versioned. `export` below writes an explicitly requested sanitized snapshot;
it does not publish or commit one.

Atomic Clang counters support threaded CPU execution. Process/module-specific
profile names avoid collisions; source digests, binary hashes and profile hashes
prevent mixing builds. Dependency code is excluded by source path, including
dependency templates instantiated in a project translation unit. LLVM supplies
the line-count aggregation. Uncovered compiled functions and branch outcomes are
reported as misses. Unsupported configurations and compiled-out device capabilities
are separate. Uninstantiated templates and preprocessor-elided paths have no LLVM
coverage-map denominator; their counts remain unavailable, not zero. See the
[Clang coverage workflow](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html).

Coverage is informational and has no required percentage. A timeout or memory
refusal leaves the selection incomplete and returns a nonzero exit status, while
reports for successful configurations remain available. Inspect `summary.json`
and `configurations.csv` alongside the HTML union to see its actual scope.
Instrumented execution can be much slower than ordinary propagation; its elapsed
time is not a benchmark result. A retry at another thread width is a separate
collection and does not erase the original failure.

## Local tables and public export

Collection writes normalized CSV views for runs, cases, configurations, systems,
CPU allocations, measured processes and observed libraries alongside the existing
sample, summary and accuracy tables. Records link through run, configuration and
attempt identifiers. Missing telemetry carries a reason rather than a measured
zero. Library records distinguish the initial inventory from pools observed after
the measurement, including libraries loaded lazily by a backend.

Linux allocation records include physical cores, sockets, sibling CPUs and NUMA
nodes when the operating system exposes them. Scheduler variables describe the
allocation but do not prove exclusivity. Missing topology or residency information
retains an explicit reason.

Collect thread and binding details separately from a timing campaign:

```sh
uv run poe bench diagnose --threads 1,production \
    --out .benchmarks/suite/thread-diagnostics
```

This runs the small diagnostic case at each width serially, recording per-thread
CPU time and, on Linux, allowed CPUs and resident NUMA pages. It does not change
affinity. An allowed CPU set is not evidence of where a thread actually executed.
These exploratory results cannot establish adoption.

Export a separate public snapshot explicitly:

```sh
uv run poe bench export .benchmarks/suite/runs/my-run \
    --out .benchmarks/public/my-run --alias apple-silicon-development
```

The destination must be new or empty. The export retains timings, failures,
contention, work counts, safe build identities and the selected configurations.
It omits hostnames, original machine aliases, local paths, process identities,
environment values, raw logs and free-form diagnostics. Its manifest states those
omissions. The command writes files; it does not commit or upload them.

The runner and explicit native coverage command are ready for development use.
Clean machine baselines and complete reference checks for the largest spatial cases
remain follow-up work. Planning currently uses conservative memory budgets and
reports unknown runtime costs; automatic reuse of measured memory and cost receipts
is also a follow-up. These limits do not prevent exploratory timing.

A performance baseline belongs to a particular source and build. Prepare a fresh
one after integrating performance changes, using matched workloads and thread
budgets when comparing Numba and native CPU. This does not require repeating
numerical reference preparation unless the reference's workload or numerical
assumptions changed.

Indicative results of earlier measurements are summarised under
[indicative results](benchmarks.md#indicative-results).

See [SEP 30, Part II](seps/archive/sep-0030-standard-benchmark-suite.md#part-ii--implementation-record)
for the implementation decisions, measured evidence and remaining limitations.
