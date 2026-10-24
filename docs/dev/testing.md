# Testing

All code is tested, to 100 % of statements and branches (the coverage lane fails below
it), in lanes, with several kinds of test
per feature (ADR-022). This page fixes the taxonomy, the lanes, the reference problems, the
tolerance policy and the layout. `AGENTS.md` carries the short rules; this is the long form.

## Documentation-only changes

Adding or editing a SEP, its index or roadmap entry, or making other documentation-only
changes does **not** require `uv run poe check`, the full test suite, `uv run poe test-slow`
or `uv run poe cov` before completion, commit or merge. This includes prose, links,
documentation assets, contributor instructions and prose-only docstring edits. Select
checks that cover what changed:

- Run `git diff --check` and `uv run codespell <changed-files>`.
- For SEP content, status, index or roadmap entries, run `uv run pytest tests/test_seps.py`.
- For Sphinx content, links or assets, run `uv run poe docs` (warnings remain errors).
- For docstring prose, run the relevant docstring validation and documentation build.

The exception is based on content, not the file's location or extension. Changes to
runtime code, tests, fixtures or scientific data, dependencies, build configuration,
tooling or executable examples use the validation policy below. A mixed change
also follows that policy. Existing gallery regeneration requirements still apply when a
gallery source changes, even if only its prose changed. Report the checks actually run;
do not launch the full suite merely because a SEP is being committed or merged.

## 1. Kinds of test, and when each is required

| kind | what it asserts | required for |
|---|---|---|
| unit | normal case, edge cases, error paths of every public function | everything |
| invariant | energy and photon number where conserved, Parseval, time reversal, symmetries, positivity, orthonormality of modes | grids, transforms, modes, responses |
| convergence | halve the step or double the grid and the error scales at the expected order; three levels with an order gate that rejects `order > p + 3` | stepper, quadrature, tables, Chebyshev |
| round trip | build → translate → run → process and recover a known input | pulses, output, processing |
| parity | field vs envelope for narrowband pulses; mode-averaged vs one-mode multimode; batch vs sequential; backend vs reference | every nonlinear feature, every backend |
| property-based | `hypothesis` over grid sizes, pulse parameters, pressures | grids, pulses, materials, Chebyshev |
| golden regression | `pytest-regressions` (`num_regression`) with tolerance, reference stored next to the test | materials tables, stored Luna results |
| reference problem | the ladder in [stepper.md §11](stepper.md#11-verification-ladder-for-solprops-solver-tests): P0 linear, P1 N-soliton with phase, P2 analytic z-varying loss and dispersion, P3 cross-scheme Richardson | solver work |
| target simulation | RDW VUV and CtC against stored Luna results and solprop's own converged references (§8) | the named problems M2, M3, M4 |
| doctest | every `Examples` section runs | all docstrings |
| smoke | import time budget; a run of every gallery example in reduced size | package, docs |

## 2. Tiers and lanes

Every test is in exactly one **tier**, chosen by what the test is *for* rather than by what
it costs, and carries at most one tier marker. `tests/test_tiers.py` asserts that over the
whole collected suite: the recording hook in `tests/conftest.py` runs `tryfirst`, before
pytest's own mark plugin deselects by `-m`, so the check covers the slow lane from inside
the default one.

| tier | marker | what belongs in it | budget |
|---|---|---|---|
| **fast** | none | unit, invariant, property and error-path tests; doctests; no propagation longer than a few steps | under a second each |
| **medium** | `medium` | a functional check that needs a real but short run — a few hundred steps — or a parameter sweep that genuinely costs seconds | 60 s for the tier |
| **slow** | `slow` | regressions against a stored witness (M1, M2, M4), convergence ladders and order studies whose numbers are in the convergence records (§5), and the CtC-size build | 4 min for the lane |

Two further markers select **lanes** rather than tiers, and a test may carry one of them
instead of a tier marker:

- `gpu` needs a device and is skipped without one. It selects the native device tests,
  run by hand on a machine with a device; no CI lane reaches them.
- `benchmark` is run by **no lane at all**. It is where a study goes whose numbers are
  recorded in its docstring and on which nothing in the default configuration depends —
  the full-size CtC case, the gigabyte Φ table, the polarisation taper's recurrence
  floor. Each says in its own docstring how to run it by hand; the benchmark suite
  (`poe bench`) is a separate program and does not reach them.

### The lanes

| lane | command | what it selects |
|---|---|---|
| default | `uv run poe test` | fast **and** medium, compiled kernels, random order (`pytest-randomly`), warnings as errors, doctests included, under xdist with every thread pool at one thread per worker. Step 8 of `poe check`. Plain `uv run pytest` runs the same selection serially (`addopts` carries no `-n`), for one file or test and for debugging. |
| medium alone | `uv run poe test-medium` | the medium tier, timed against its budget |
| slow | `uv run poe test-slow` | the slow tier under xdist, one worker per core, `--dist loadgroup`, every thread pool pinned to one thread per worker |
| coverage | `uv run poe cov` | interpreted CPU kernel bodies, then the compiled fast and medium tiers, both under xdist with one thread per worker; 100 % of the package (`coverage-cpu.ini`) |
| gpu | `uv run pytest -m gpu` | needs a card |
| benchmark | by hand, per docstring; `uv run poe bench` for `benchmarks/` | see [benchmarks.md](benchmarks.md) |

Hypothesis properties use the `solprop` profile loaded by `tests/conftest.py`.
Their example count bounds the work; the per-example wall-clock deadline is disabled
because imports, JIT compilation and contention on shared machines can exceed it
without indicating a correctness failure.

Parallel lanes use `--dist loadgroup`. The collection hook groups a test with its
module only when it uses a module-, class- or package-scoped fixture, and with the
tests of the same parameter when that fixture is parametrised, so a shared fixture —
the M2 and M4 witness runs — is built once rather than once per worker. A test that
shares none goes to whichever worker is free: grouping every module would put the
heaviest modules (the coherent capillary, the exponential march) behind one worker and
make them the run's tail. Independent expensive tests that do share a fixture may
still declare an explicit `pytest.mark.xdist_group` to run on separate workers; only
split cases whose expensive fixtures are independent, since splitting a shared witness
can duplicate more work than it saves. The three slow moving-guide studies and two
independent coherent-capillary true-power references use separate groups.
The hook records original node identities before xdist appends group names, so
tier selection and saved numerical witnesses keep their existing identities,
including literal `@` characters in parameter IDs. Convergence-record writes
remain serial-only; normal grouped runs do not regenerate witnesses. The benchmark
suite under `benchmarks/` is outside `testpaths` and is never collected.

A lane's remaining scheduling floor is its longest group. Pinning `SOLPROP_THREADS` (each
worker's thread budget, ADR-052, which every pool of both backends stays within),
`SOLPROP_NATIVE_THREADS`, `VECLIB_MAXIMUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` and `OMP_NUM_THREADS` to one keeps eighteen workers from each opening a
thread pool over the same cores. `NUMBA_NUM_THREADS` is not pinned: it is numba's
ceiling, not its default. A test that compares thread counts lifts the budget for itself
(the `thread_budget_lifted` fixture) and asks for several threads; a native probe that
needs more threads than the runner has CPUs sets `SOLPROP_OVERSUBSCRIBE=1`. Serial and parallel runs select the same tests; inspect
the current `--durations` output before choosing another independent group to split. The
lane stays
reproducible: `pytest-randomly` picks one seed in the controller and the workers share it,
so `-p randomly --randomly-seed=<n>` replays an ordering across workers.

### What a shortened test owes

A test moved to a cheaper tier keeps its claim. It restates its bound from the shortened
run, quotes the measured number in a comment beside the assertion, and — where the
shortened run no longer measures something the full one did — records the full run's
number and how to reproduce it in its docstring, so nothing the suite used to assert is
merely dropped.
Preferring a quantity the test can *compute* over one it quotes is what usually makes a
shortening possible at all: the reference backend and the fixed-step scaffold are compared
against the scaffold's own recurrence error, computed in the test, so a few hundred steps
carry a claim that used to need four thousand.

### Tiers and backends

Medium tests are parametrised over every installed compatible backend and stay medium. The
base checkout has `numpy_ref` and `numba_cpu`; an installed `solprop-native` adds `native`
conditionally. The slow ladders run on the backend selected by `SOLPROP_TEST_BACKEND`,
defaulting to `numba_cpu`; the backend-versus-reference parity test of §1, with the
equivalence tolerance of §3, stands in for duplicate copies already known to agree.

Spatial tests select their backends through the explicit spatial fixtures of
`tests/backends/conftest.py`, which cover NumPy, Numba CPU and an installed native backend.

For native development, `uv sync --extra native` builds `native/` and installs it
editable into the repository's environment before a lane runs; how, and when a build
must be rebuilt, is
[Building the native backend](../guide/performance/building-native.md#a-development-build).
`uv run poe check-native` runs the native package's own gate and `uv run poe check-all`
both gates. What the backend does not support it refuses by name
([backends.md](backends.md)). On macOS import the native package before pyFFTW so the
process loads one OpenMP runtime; `tests/backends/conftest.py` does this deliberately.
The `gpu` lane checks build coherence, one complete device step and same-card
repeats within a rounding bound on an accelerator build; host thread-count tests skip on a
device build by design, and fixed-geometry same-card repeats stand in for them. The slow equivalence and
M2/CtC witnesses are selected by node id so one marker remains one tier. With the build
installed, the lanes run as:

```console
SOLPROP_NATIVE_THREADS=1 OMP_PROC_BIND=false uv run poe check
SOLPROP_NATIVE_THREADS=1 OMP_PROC_BIND=false uv run poe test-slow
SOLPROP_NATIVE_THREADS=1 OMP_PROC_BIND=false uv run poe cov
SOLPROP_TEST_BACKEND=native SOLPROP_NATIVE_THREADS=1 uv run poe test-slow

# On an accelerator build:
SOLPROP_TEST_BACKEND=native uv run pytest -m gpu
```

The two thread settings matter: eighteen xdist workers each initialising an
eighteen-thread Kokkos pool oversubscribe the machine, and OpenMP binding applied to a
process that runs both compiled backends pins numba's and Kokkos's workers to the same
cores.

### Budgets and timing assertions

The medium tier's budget is tight: time it with `poe test-medium` before adding to it. Most
of the default lane's time is the compiled-kernel tests, which compile one kernel per
parametrisation, so parametrise a compiled test only over what the kernel actually
branches on.

A test that asserts a **cost** belongs in the slow tier, because its verdict needs a quiet
machine and repeated fresh processes. The periodic-statistics cost gate
(`tests/driver/test_prop_capillary.py`, driving `tests/statistics_cost_probe.py`) is the
pattern: three fresh processes per configuration, rotating the plain/watched order across
the pairs, each warming its configuration and recording three runs with production
imports and without coverage or the runtime type hook; the parent compares the median of
the process medians against the existing 25 % bound and records the MAD, raw wall and CPU
times, the load, backend metadata and the exact step histories. A child failure fails the
parent. Each measured window is ten fixed Runge–Kutta steps over 2 mm, on the reduced
M2 grid. `poe test-slow` completes its parallel functional tests before running this
cost check serially: competing test workers can otherwise slow one arm more than
the other. The bound has not been relaxed.

## 3. Tolerance policy

A tolerance without a comment justifying it is a defect. Acceptable justifications: a
rounding budget (Higham's `γ_n` bound for a reduction of length `n`, the transform bound
`B_fft`, operation counts per step, with a stated margin); discretisation error of the grid
or step; the accuracy of a reference (a table's stated accuracy, an experimental figure, a
Luna result's own convergence). Independent floating-point computations are compared by
numerical error, never by bits ([SEP 25](seps/archive/sep-0025-rounding-level-reproducibility.md)):
no comparison requires bitwise agreement across backends, builds, machines, thread counts,
batch membership, scan execution or repeated runs. Every comparison belongs to one of five
classes, and `tests/bounds.py` holds the reusable parts of their bounds.

| class | what is compared | criterion |
|---|---|---|
| **P**, structural | lossless storage, copies and views; integer accounting and ledgers; identity metadata and hashes used for configuration identity, cache validation or provenance; hard-mask zeros; explicitly exact no-ops (a feature left off, a row reusing the data its plane stores) | exact (`np.array_equal`, `==`), with a comment saying why the contract is exact |
| **R**, matched discrete computation | two backends, thread counts, a batch member and its solo run, a scan point and its run, a fast path and the general path: the same method, orders, startup and restart history, schedule and inputs | a rounding bound |
| **T**, accuracy | adaptive runs, ordinary replay, resumed against uninterrupted runs: discrete trajectories that may differ | an accuracy envelope |
| **D**, recorded state | a route's stored final state (§5) | `ndarrays_regression` at a bound of class R or T |
| **N**, noise | a seeded realisation | a short-window R comparison, or an ensemble observable with a tolerance set from the ensemble size |

**Rounding bounds** (class R) are fixed *before* the comparison is run, from the
operation and reduction counts, transform lengths and magnitudes, with a stated safety
margin, never fitted to the discrepancy they then accept. The ladder of
[accuracy.md](accuracy.md) section 7 is the pattern (`16 ε` for one step, `128 ε` for
eight), and `tests/bounds.py` provides `gamma`, `higham_gamma`, `fft_bound`, `scan_bound`,
`prefix_bound`, `stencil_bound`, `step_budget` and `propagation_growth`. `u = 2⁻⁵³` and
`ε = 2u` are not interchangeable. A case whose dynamics amplify strongly is compared
over a short window. Backend equivalence uses peak-normalised or normwise deviation
against nontrivial reference fields, so a zeros-returning backend cannot pass:
`assert_within` checks finiteness, refuses an identically zero reference and reports
the observed margin. A result that is mathematically zero but reached through inexact
arithmetic gets a justified absolute tolerance; a zero that is a structural contract
(a hard mask, a disabled term) stays exact.

**Accuracy envelopes** (class T) are set per case, in a comment: the run's error against
a reference at a tighter tolerance, or against an existing stored converged reference,
times a safety factor fixed before the candidate is measured (`AccuracyEnvelope`). Each
run is compared with that reference. `rtol` and `atol` control a local estimate and are
not themselves a bound on the global error. The three-level refinement of section 1 is
how a route's accuracy is established; it is not repeated in every test.

**Replays.** A rounding-level (R) comparison of two propagations runs **both** sides on
the same prescribed schedule with a fixed-order method, matched startup and restart
history and matched save-plane handling. An adaptive run may discover that schedule,
but its own result is not the oracle: an Adams replay chooses its orders afresh and does
not replay rejected trials ([stepper.md](stepper.md)). Ordinary variable-order replay and
resumed-against-uninterrupted comparisons are class T.

**Work counts.** Step, trial and source-call counts of independent adaptive runs are
performance diagnostics, held to loose bands declared from baseline data
(`assert_work_within`), never to equality: a rounding difference may flip one
accept/reject decision. Accounting is exact: a ledger counts the calls actually made,
and a prescribed schedule executes exactly its steps.

**Loosened bounds.** A test whose bound became looser than it was is spot-checked once
with a deliberate perturbation (a flipped sign, a dropped term) to show that it still
fails, and the commit says so.

## 4. Determinism

Randomness is explicit (`np.random.default_rng(seed)`); tests are isolated (`tmp_path`, no
network, no global state). Legacy `np.random.seed` is rejected by ruff. A test whose
result could depend on an adaptive fork is written so that a fork cannot fail it: it
either compares matched discrete computations on a prescribed schedule (class R) or uses
an accuracy envelope (class T). One deliberately near-threshold controller case checks
that a real fork passes. Seeded noise arrays are inputs: arithmetic comparisons use the
same stored launch and noise arrays on both sides, and the generator itself stays
reproducible.

## 5. Reference data

Recorded numerical states are compared at a tolerance, not by digest, so no record is tied
to one platform or one FFTW build and the PyPI wheels suffice. The `REFERENCE` native build
profile is a tidy and sanitizer build and defines no numerical pins. Hashes used for
configuration identity, cache validation, artifact integrity or provenance are exact
(class P). Cross-backend parity, analytic checks and convergence tests are separate from
these records.

- **Recorded final states.** A route that must not move when nothing in it changed (an
  opt-in feature left off, a section with nothing to score, the legacy entry points) is
  checked against its recorded final state with `ndarrays_regression`
  (`tests/state_records.py`, stored beside the test), at a bound derived as class R for
  a fixed-schedule case and as a class T envelope otherwise. That bound is above what a
  fresh environment, another dependency build, another platform or an optmat release
  that moves no value changes, and below every deliberate change. A change that moves a
  route's arithmetic checks old against new within that bound before it records the
  states afresh (`pytest --force-regen`), and says so in its commit message.
  The run record's hash leaves out what an optmat release writes into it (its version,
  the citations, the manifest's model digests), so only solprop's record and its numbers
  pin it.

- **Luna results** for the target simulations are generated once, by hand, by the
  `tests/data/generate_luna_*.py` scripts with the Luna commit recorded, at reduced grid
  sizes for the fast suite and full size for the slow suite, and stored under `tests/data/`
  with a checksum. The suite never runs Julia. They
  are regression witnesses, not truth: where solprop's converged answer differs from Luna's
  because Luna was wrong (linear operator order, static noise, per-step windows), the test
  documents the expected difference.

  The witnesses are **frozen**: none is ever regenerated, and a generator stays only as
  the record of what Luna was given. Every input a comparison depends on — temperature,
  pressure, gas and cladding data, rates, Raman on or off, the grid rules and collar, the
  absorber, noise, the model, the quadrature and the tolerances — is passed explicitly in
  the test, so that a change of a solprop default cannot silently change what is
  compared; where the explicit value is solprop's own rather than Luna's, deliberately,
  to measure the difference, the test says so. A witness is retired, with its
  generator, only when it blocks a change, and that change says so.
- **Analytic problems** (P1, P2) are truth and need no stored data.
- **Converged solprop references** (P3) are produced by the three-level ladder with the order
  gate and stored with their observed order.
- **Convergence records.** Every ladder and order study of
  [accuracy.md](accuracy.md) writes its rungs to `tests/data/convergence/<study>.json`,
  committed like a witness. On every ordinary run the study recomputes its numbers as it
  always did and compares them with the stored ones, so a ladder that moves is a failure
  rather than something a reader notices; the page's figures are drawn from the same
  files. The mechanism is below.

### Convergence records

The mechanism (`tests/convergence_records.py`), how a record is compared or written, the
tolerance each column carries and the commands are in [accuracy.md, "Regenerating the numbers"](accuracy.md#9-regenerating-the-numbers).

## 6. Layout and naming

`tests/` mirrors `src/solprop/` (`tests/modes/test_capillary.py` ↔ `solprop/modes/capillary.py`);
names are `test_<unit>_<behaviour>`. Most test directories are not packages (no
`__init__.py`), so a module's basename must be unique across the whole tree. Fixtures for the standard problems live in the
`conftest.py` of the directory that uses them (`tests/backends/conftest.py`,
`tests/driver/conftest.py`), and shared cases, oracles and parity helpers in modules at the
top of `tests/` (`*_cases.py`, `*_oracles.py`, `*_parity.py`, `convergence_studies.py`);
parametrise instead of copy-pasting; shared tolerances are named constants with their
justification.

Two tests read the documentation rather than the package: `tests/test_guide_snippets.py`
runs the user guide's code, and `tests/test_seps.py` keeps each enhancement proposal's
header, folder, index row and roadmap link consistent with its status
([SEP 0](seps/sep-0000-process.md)).

## 7. Definition of done for a pull request

A documentation-only pull request is done when the relevant checks in
[Documentation-only changes](#documentation-only-changes) pass. Other pull requests
are done when:

- the targeted checks pass, and the full gate and lanes pass or are covered by existing
  results or CI;
- new code has complete coverage: targeted coverage of the changed modules during
  development, and the full coverage lane before a release (`AGENTS.md`, Definition of
  done);
- every new test says which tier it is in and why, and every new tolerance is justified
  (§3);
- a change to a numerical result updates the convergence records it moves, with
  `SOLPROP_WRITE_RECORDS=1` on a clean tree, and says why they moved;
- a hot-path change carries a benchmark entry ([benchmark rules](benchmarks.md#rules)).

A change is qualified on `numpy_ref`, `numba_cpu` and, when installed, the native CPU
build; device builds are exercised separately with `pytest -m gpu`. No test, lane or
timing runs a full-length propagation: qualify on about ten steps or a fibre short enough
to finish in seconds (`AGENTS.md`, qualification scope).

## 8. Named test problems

Tests, records and benchmarks refer to these by name.

| name | problem |
|---|---|
| **P0–P3** | the solver's verification ladder: linear only (P0), N-soliton recurrence with phase (P1), analytic z-varying loss and dispersion (P2), full UPPE against a Richardson reference (P3); see [stepper.md](stepper.md) §11 |
| **M1** | N = 6 soliton recurrence through `prop_gnlse`: GNLSE, envelope grid, mode-averaged, checked against the analytic solution with its phase |
| **M2**, **RDW** | resonant dispersive-wave emission into the VUV in a hollow capillary: 125 µm core radius, 1.5 m, helium from 0.8 bar to vacuum, four HE₁ₘ modes, a 7.5 fs 275 µJ pulse, field grid, Kerr and plasma; full and reduced sizes, against a stored Luna witness and solprop's own converged reference |
| **M3** | the M2 case on `numba_cpu`: the compiled backend against `numpy_ref` (and Luna) at equal achieved error; the whole-run benchmark rows |
| **M4**, **CtC** | the comb-to-continuum supercontinuum: 16 µm core radius, 0.2 m of hydrogen at 40 bar, six HE₁ₘ modes, a 400 fs pulse at 515 nm on a 4 ps field grid, Raman and plasma; full and reduced sizes, against a stored Luna witness |
