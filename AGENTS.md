# AGENTS.md

Working instructions for coding agents (and people) in this repository.

**Read first:** the design lives under `docs/dev/`. Start at
[`docs/dev/index.md`](docs/dev/index.md), then [`architecture.md`](docs/dev/architecture.md)
and [`conventions.md`](docs/dev/conventions.md). This file is the short form: the rules,
each with a pointer to the page that explains it. Those pages win on any detail.
[`CONTRIBUTING.md`](CONTRIBUTING.md) is the short guide for a first contribution, and holds
the standard for issues and pull requests.

**Rules marked ⚙ are enforced automatically** by the ruff, basedpyright, numpydoc,
codespell, reuse and pytest configuration in `pyproject.toml`. The configuration is the
source of truth ([`docs/dev/tooling.md`](docs/dev/tooling.md) explains its intent); if a ⚙
rule and the configuration disagree, the configuration wins and this file is out of
date — say so.

Everything else is a judgement call. When a judgement call concerns physics or numerics
rather than style, ask rather than guess.

---

## Project facts

| | |
|---|---|
| Packages | `solprop` (`src/solprop/`) and the native backend `solprop-native` (`native/`, its own rules in [`native/AGENTS.md`](native/AGENTS.md)): one repository, two distributions, one version |
| Purpose | Ultrafast nonlinear pulse propagation (UPPE and GNLSE; field and envelope; single-mode, multimode and general modal guides; radial, Cartesian, planar and line free space) |
| Python | 3.14 — equal to `requires-python` in `pyproject.toml` |
| Platforms | Linux, macOS and Windows for `numpy_ref` and `numba_cpu`; solprop-native as wheels for osx-arm64, linux-64, linux-aarch64 and win-64, from source on Linux and macOS (GPU builds on Linux); the Queue and Slurm scan executors POSIX-only |
| Type checker | `basedpyright`, pinned exactly in the dev group |
| Backends | In tiers (ADR-061). Tier 1, the core: the NumPy reference (`numpy_ref`, the oracle) and the native backend (`native`, the `solprop-native` distribution built from `native/`: preferred when installed, and the GPU path). Tier 2, mature: Numba CPU (`numba_cpu`, in-tree). Tier 3, experimental: `mlx` (single precision on Apple GPUs) |
| Materials | [optmat](https://github.com/LupoLab/optmat), the curated optical-materials package |
| Design docs | `docs/dev/` — architecture, conventions, stepper, algorithms, backends, kernel contract, decisions, roadmap, open questions, testing, accuracy, benchmarks, benchmarking, dependencies, tooling |
| Planned work | [GitHub issues](https://github.com/LupoLab/solprop/issues): open work goes in an issue, not on the roadmap, which lists only SEPs (ADR-062). [`CONTRIBUTING.md`](CONTRIBUTING.md) says how to write an issue and how to label an issue or a pull request |
| Enhancement proposals | `docs/dev/seps/` — SEPs for major changes; process in SEP 0 (`sep-0000-process.md`), form in `template.md` |

---

## Commands

| Task | Command |
|---|---|
| Sync environment | `uv sync`; `uv sync --extra native` also builds and installs the native backend from `native/` (editable). A plain `uv sync` removes it; `uv sync --inexact` keeps a hand-installed one or Pyroraptor |
| Full gate (see below) | `uv run poe check` |
| Native gate | `uv run poe check-native` (`native/scripts/check.sh`); `uv run poe check-all` runs both gates |
| Lint with autofix | `uv run ruff check --fix .` |
| Format | `uv run ruff format .` |
| Type-check | `uv run basedpyright` |
| Docstring validation | `uv run poe docstrings` |
| Spelling | `uv run codespell` |
| Copyright and licence of every file | `uv run poe reuse` |
| Default tests (fast + medium) | `uv run poe test` (under xdist, one thread per worker); `uv run pytest` runs them serially, for one file or test and for debugging |
| Medium tier | `uv run poe test-medium` |
| Slow lane | `uv run poe test-slow` |
| Coverage lane (100 % required) | `uv run poe cov` (kernel bodies interpreted first, remaining suite compiled) |
| GPU tests | `uv run pytest -m gpu` (needs a GPU build of solprop-native and a device) |
| Benchmarks | `uv run poe bench` plans; `uv run python -m benchmarks track` runs (on an otherwise idle machine) |
| Build docs | `uv run poe docs` (executes no gallery script; refuses stale committed output) |
| Run the galleries | `uv run poe docs-examples [plot_NN_name]` (every stale lesson, tutorial and worked example, or one; commit what it writes) |

Do not narrow a plain `pytest` run with `-m`: it replaces the default marker expression
in `addopts` and selects the slow tier too. Select by path or `-k` instead.

## Definition of done

**Documentation-only changes**, including adding or editing SEPs, do not need the full
gate, the slow lane or the coverage lane. Run `git diff --check` and codespell on the
changed files; `uv run pytest tests/test_seps.py` when proposals, their index or their
roadmap entries change; and `uv run poe docs` when Sphinx content or links change
(`docs/dev/testing.md`, "Documentation-only changes").

**Other changes.** Run the fast checks and directly affected test modules during
development. The full gate runs in CI on the pull request, with the slow and coverage
lanes: do not run it locally before opening the pull request, and do not start a local
full run because a slice, task or SEP is ready to commit, merge or become `Final`. Fix
what CI reports by rerunning the failing modules, not the whole gate.

The full gate, `uv run poe check`, runs in this order (CI runs the same steps):

1. `ruff check --fix .` — lint first: autofix rewrites code.
2. `ruff format .` — format second, so fixes are formatted.
3. `ruff check .` — no `--fix`; catches what autofix could not.
4. `basedpyright` — no new errors.
5. `numpydoc lint` — every public docstring validates.
6. `codespell` — no misspellings.
7. `reuse lint` — every file states its copyright and licence.
8. `poe test` — fast and medium tiers, doctests included, under xdist with one thread
   per worker.
9. `sphinx-build -W -n --keep-going` — docs build clean.

Report what passed, failed or remains deferred. The
coverage lane requires 100 % of statements and branches of the whole package
(`fail_under = 100`) but the experimental `mlx` backend, a recorded omission
(`docs/dev/testing.md`, "The `mlx` omission"); it stays outside `poe check` only
because it takes longer. `# pragma: no cover` needs a reason on the same line.

**CI** (`.github/workflows/ci.yml`) runs on every push to `main` and every pull request:
`poe check` split into parallel jobs (the static checks and the documentation on Linux;
the default lane on macOS and Windows with the native CPU wheel built from this tree; and
on Linux the coverage lane, which is the default lane with coverage and the runtime type
checks); the slow lane on Linux and macOS; and the default lane at the lowest dependency
versions. The long lanes run as shards of whole modules (`SOLPROP_TEST_SHARD`), and only
the coverage lane runs the type checks (ADR-053; `docs/dev/tooling.md`, "CI"). Between
them the jobs run exactly `poe check`'s steps: change both in the same commit.

**Changes touching `native/`** follow [`native/AGENTS.md`](native/AGENTS.md). Their full
gate, `uv run poe check-native` (`uv run poe check-all` for both gates), also runs in CI:
the native gate as the `native-quality` jobs and the distribution wheel's build and smoke
test as `native-wheel`, whose wheel the macOS, Windows and coverage lanes install.

**Slices of a phased SEP.** Develop first and test towards the end. Each slice runs the
fast checks only: `ruff check --fix`, `ruff format`, `ruff check`, `basedpyright` and the
test modules that directly exercise the code it changed (`uv run pytest -n auto
<files>`). The full gate, slow lanes and coverage lane run in CI on the SEP's single
pull request, opened once the SEP is set `Final`. After a CI failure, rerun the failing
modules locally, fix, and push; never the whole lane.

**Qualification scope.** A change is qualified on `numpy_ref`, `numba_cpu` and, when it
is installed, the native **CPU** build; GPU builds are exercised separately with
`pytest -m gpu` on a machine that has a device. **No test, lane or timing runs a
full-length propagation** (a 0.2 m multimode capillary, a metre of fibre): qualify and
benchmark on about ten steps, or on a fibre short enough to finish in seconds. Two
gallery kinds are exempt. An introductory lesson (`docs/introduction/tutorials/`) or a
focused tutorial (`docs/tutorials/`) may run its real problem at full length when that is
what it demonstrates; it runs when its output is refreshed and in CI's docs job. A worked
example (`docs/examples/`) runs a real problem at full size, only by hand with
`poe docs-examples`. The release
tracking set (`python -m benchmarks track`) times complete runs by design
([what is timed](docs/dev/benchmarks.md#benchmark-windows)); it runs by hand, on a quiet
machine, when the history is recorded, never as part of a change's qualification.

Do not declare work finished while any required check fails. If a check genuinely cannot
pass, say so explicitly and explain why — never suppress a check to make the gate green.

New functionality is not done until it is **tested**, **documented in the manual** as
well as in docstrings, and recorded in `CHANGELOG.md`. Any physical constant,
coefficient, citation, reference state or validity range you transcribe is cited to its
primary source in the docstring or the data file; say in your summary what you verified
against that source and what you could not.

---

## Working agreement

- **Agree the plan for large changes.** A major change — a new model, route or numerical
  scheme, a changed physics default or convention, a public interface redesign, a
  substantive seam change — starts from an accepted SEP (below). Any other change that
  touches more than a couple of files or alters a public signature starts from an issue
  or a short agreed plan; a focused fix or documentation update does not need one.
- **Ask before adding a dependency.** Standard library, then NumPy/SciPy, then the
  approved list in [`dependencies.md`](docs/dev/dependencies.md), then ask. A floor is
  the oldest release that works, not the newest; raise one only for a recorded reason.
- **Re-derive, do not transcribe.** When taking an algorithm or a model from another code
  or a paper, re-derive it, choose deliberately, test it against an analytic or
  independent reference and document it. Deliberate differences from Luna.jl, solprop's
  predecessor, are listed in `docs/guide/workflows/luna.md`; add to it.
- **Stay in scope.** No refactoring, renaming or reformatting unrelated to the task.
- **Never hand-edit generated files**: `uv.lock`, `docs/_build/`, the gallery pages
  `docs/introduction/lessons/`, `docs/auto_tutorials/` and `docs/auto_examples/`
  (committed, regenerated by `poe docs-examples`) and `docs/api/generated/`. Regenerate
  them.
- **No `utils` modules.** Organise by domain concept; the package tree is fixed in
  `docs/dev/architecture.md §4`.
- **Report failure honestly.** A check that fails, a test you could not make pass, an
  assumption you had to invent: say it plainly. Silent partial completion is worse than
  an admitted gap.
- **The public tree stands alone.** No local paths, host names, private notes or links to
  them, and no development-register identifiers, in code, tests, docs or the changelog.
  The hardware a benchmark ran on is part of its ledger entry.

---

## Enhancement proposals (SEPs)

The long form is [SEP 0](docs/dev/seps/sep-0000-process.md); the index is
[`docs/dev/seps/index.md`](docs/dev/seps/index.md).

- **When.** New physics models, routes, geometries, mode families or numerical schemes;
  changed defaults or conventions; public interface redesigns; substantive
  kernel-contract changes; new backends; process changes. Not for fixes, documentation,
  tests, refactoring, tooling, or optimisations that leave results unchanged.
- **Writing one.** Copy `template.md` to `sep-NNNN-short-slug.md` with the next free
  number, add it to the index and the roadmap's draft section, list in its *Context* the
  open issues it will fix or that must be resolved first (search them), open its tracking
  issue (SEP 0), and merge it early as `Draft` from a `docs/sep-NNNN-…` branch. A SEP stands alone: it cites solprop's,
  solprop-native's and optmat's own docs and code, the literature (with DOIs) and public
  reports — never another code's source, local paths, machines or private notes. Theory
  is re-derived in SI units and the conventions of `conventions.md`; it compares options
  and recommends one.
- **Lifecycle.** `Draft` → `Accepted` (Part I frozen) → `In progress` → `Final` (moved to
  `docs/dev/seps/archive/` with `git mv`); or `Deferred`, `Rejected`, `Withdrawn`,
  `Superseded`. `tests/test_seps.py` checks that the header, index, folder and roadmap
  agree.
- **Implementing one.** All of a SEP's phases go on one `feature/sep-NNNN-…` branch and
  reach `main` in one pull request, opened when the last commit sets `Final`; there are
  no per-phase pull requests. The branch may start on top of an open pull request's
  branch when pull requests are stacked behind CI. Within each phase: `numpy_ref` first
  with its tests and manual page, then native CPU in `native/`; `numba_cpu` and `mlx`
  only where the SEP says so (the tiers, below). A backend without the feature refuses
  it by name, never falls back silently; `Final` needs parity on `numpy_ref` and native
  CPU, or an exception written into the accepted SEP.
- **Reporting.** Each phase appends its report to Part II of the SEP: delivered paths,
  deviations and why, decisions (an ADR for a lasting one), the verification actually
  run, sources checked and not checked, limitations, numbered open questions and a
  handoff.
- **Finishing.** At `Final`, the manual pages the SEP produced link to it, its open
  questions move to issues, its tracking issue closes, and the `CHANGELOG.md` line cites
  it.

---

## Conventions: units, signs, normalisation, layout

The single source is [`conventions.md`](docs/dev/conventions.md). Every docstring defers
to it; a function that departs from it says so in the first line of its docstring.

| Quantity | Convention |
|---|---|
| Base units | SI exclusively (m, s, J, W, V/m, Pa, K, rad/s, m⁻³); pressures accepted in bar at the user interface only |
| Fourier sign | **Optics convention at the interface**: `Ẽ(ω) = ∫ E(t) e^{+iωt} dt`, plane wave `e^{i(βz − ωt)}`, delay `e^{+iωτ}`. **FFT convention inside the seam**: raw forward DFTs, `e^{i(ωt − βz)}`. The boundary is a conjugation in exactly three named places (`fourier.to_optics`/`to_internal`); HDF5 files are optics-convention (`conventions.md §2`) |
| Linear operator (internal) | `D = −i(β − β₀ − β₁(ω − ω₀)) − α/2`, `Im n_eff > 0` is loss |
| Amplitudes | interface modal amplitude `b = sqrt(Re n_eff) a` in true √W; the internal `a` is vacuum-normalised (`conventions.md §4`) |
| Kernel state layout | `(n_batch, n_modes, n_pol, n_omega)` complex128, ω contiguous; saved fields `(n_z, n_modes, n_pol, n_omega)` |
| Axis names | `n_batch n_modes n_pol n_omega n_omega_fine n_t n_to n_nodes n_z n_intervals n_deg n_windows`, reused verbatim in annotations |

## Guiding principles

- One obvious way to do each thing. Prefer the existing pattern; if you find two ways to
  do the same thing, consolidate them.
- Optimise for the reader. Clear beats clever.
- Small, single-purpose functions with explicit inputs and outputs. Prefer pure
  functions; push side effects (I/O, global state, mutation) to the edges.
- Make the implicit explicit: explicit arguments, explicit types on interfaces, explicit
  errors over silent fallbacks.
- Fail loudly and early with a clear exception; never swallow errors or return a
  sentinel where raising is correct.
- Respect the layers (`architecture.md §3`): physics code never contains performance
  tricks; backend code never contains physics knowledge; the kernel contract
  (`kernel-contract.md`) is the only crossing.

## Environment and dependencies

- Python and dependencies are managed with **uv** (`docs/dev/tooling.md`). Prefix
  commands with `uv run`. Never `pip install` into the environment by hand and never edit
  `uv.lock`; add dependencies with `uv add` / `uv add --dev` (after asking). A change that
  alters resolution commits the updated lockfile (CI runs `uv lock --check`).
- Dependencies come from PyPI; never commit a local-path or private source.
- The package version is derived from git tags. Do not set a version string, and do not
  tag.

## Code style

Formatting and linting belong to `ruff format` and `ruff check` ⚙ (import sorting,
absolute imports, naming, line length, no bare `print`, annotations, NumPy legacy API,
no blanket `# noqa`). What is left to judgement:

- Fix the cause rather than silencing the warning. A suppression is rule-specific and
  carries a reason: `# noqa: E501 - URL must stay on one line`.
- Names state what a thing is or does. Domain abbreviations (`fwhm`, `zdw`, `rdw`,
  `gnlse`, `ppt`) are fine; `tmp2`, `arr_b` are not. Greek letters are spelled out
  (`omega`, `beta1`, `chi3`); wavelengths are `wavelength`, never `lambda`.
- f-strings for interpolation, `pathlib.Path` for paths, and an explicit
  `encoding="utf-8"` on every text read and write (Windows defaults to another encoding).
- Every code file starts with the SPDX header that `uv run reuse annotate --copyright
  "John C. Travers and contributors" --year 2026 --license GPL-3.0-or-later <file>`
  writes ⚙. Documentation, figures and data are covered by `REUSE.toml`.
- Declare the public surface of every public module with `__all__`.

## Types and arrays

- Annotate every public signature and class attribute ⚙; not obvious locals. No new
  basedpyright errors; bump the pinned checker deliberately, in its own commit.
- Prefer `Sequence`/`Mapping` for read-only parameters, `X | None`, built-in generics;
  avoid `Any`, and confine and comment it when unavoidable. Structured data is a frozen
  `@dataclass`, not a loose tuple or dict.
- **Arrays are annotated with jaxtyping on `np.ndarray`** — the one way; no JAX is
  involved — with the axis names of `conventions.md §6`:

  ```python
  def propagate(
      field: Complex[np.ndarray, "n_modes n_pol n_omega"],
      omega: Float[np.ndarray, " n_omega"],
  ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
  ```

- Shapes and dtypes are checked at runtime under beartype in the test session, and by
  explicit validation at public boundaries. The hook constrains how you annotate
  (`ArrayLike | float`, quoted self-returns, no `numpy.typing.NDArray`, ...): see
  `docs/dev/tooling.md`, "jaxtyping + beartype". Run the gate on the merged tree.
- `scipy-stubs` is a dev dependency. numba and pyfftw ship no stubs: use a narrowly
  scoped ignore with a comment at the import.

## Backends, kernels and hot paths

The long form is [`backends.md`](docs/dev/backends.md).

- **Tiers** (ADR-061, `backends.md §10`) set where development goes. Tier 1,
  `numpy_ref` and `native`: every new feature lands on both, and they come first for
  testing and development. Tier 2, `numba_cpu`: mature; keep what it runs correct and
  in parity, and add a feature only where its SEP says so. Tier 3, `mlx`: experimental;
  keep it runnable and correct, add nothing as part of other work. A backend without a
  feature refuses it by name.
- The **NumPy reference backend is the oracle**. Every other backend is validated against
  it (one linear application, one nonlinear evaluation, one step, eight steps, each
  statistic, a full run) with tolerances derived from summation bounds and asserted `> 0`.
- Hot code lives only under `src/solprop/backends/` and `src/solprop/translate/`.
- **numba kernels**: `@njit(cache=True, nogil=True)` with `njit` imported from
  `solprop._numba_cache`, never from numba; explicit signatures for stage kernels. The
  wrapper sets the floating-point licence (below): contraction by default, reassociation
  only where a kernel asks for it, `strict=True` for a protected helper, nothing else.
- **Never disk-cache a parallel closure or a closure that captures a dispatcher**
  (`cache=False`; memoise the factory in-process with `functools.cache`). Why is in
  `backends.md §3.2`; `tests/backends/test_numba_cache_policy.py` enforces it.
- Per-element mathematics is written once in `backends/bodies.py` through
  `define_bodies(decorate)` and composes through the closure, never module globals.
- **No allocation, logging, Python objects or exceptions inside kernels**; every buffer
  is a parameter; failures are status codes checked by the driver.
- **Reproducibility to the error bound** ([SEP 25](docs/dev/seps/archive/sep-0025-rounding-level-reproducibility.md)).
  Independent floating-point computations are compared by numerical error, not bits:
  - No bitwise agreement is required across backends, builds, machines, thread counts,
    batch membership, scan execution or repeated propagation runs.
  - Matched discrete methods and histories use rounding budgets that account for
    operations, magnitudes, conditioning, transforms and propagation sensitivity.
  - Adaptive runs and ordinary replay meet a per-case accuracy envelope set against a
    tighter or stored converged reference; `rtol`/`atol` alone do not bound global
    error. Replay need not reproduce the original adaptive trajectory or agree with it
    at rounding level.
  - Kernels may reorder, vectorise, thread and contract. Stable increments,
    small-argument functions, defect calculations and compensated sums keep their
    protections. Contraction is on by default for compiled kernels; the few protected
    helpers opt out and stay strict. Reassociation is enabled only on the loops that
    need it, per kernel. No umbrella fast-math, `nnan`/`ninf`, `nsz`, approximate
    functions or reciprocals, or flush-to-zero.
  - Preserve exact structural contracts: lossless copies and storage, integer and
    identity metadata, hard-mask zeros and explicitly exact no-op/zero behaviour. A
    mathematical zero reached through inexact arithmetic uses a justified absolute
    tolerance. Adaptive work counts may differ; accounting for actual work is exact.
- **In-place and `out=`** only inside the backends, wrapped by the seam; every buffer is
  listed by `memory_ledger()`, which is tested.
- Device arrays never cross the seam; backends take and return host arrays.
- **`KERNEL_SCHEMA_VERSION` versions the whole seam** (the `KernelProblem` fields, the
  protected `StepDriver` surface and the stage bundles of `translate/stage.py`, frozen in
  `tests/backends/test_driver.py`). Changing any of them bumps the version, rewrites the
  literal in the same commit and adds a changelog line.
- Every hot-path change comes with a benchmark ledger entry
  ([`benchmarks.md`](docs/dev/benchmarks.md)), measured with `python -m benchmarks ab`
  on about ten steps or a short `ab --length` window; report the window measured and
  never extrapolate it to a full run.
- Every kernel is also exercised in the coverage lane (`NUMBA_DISABLE_JIT=1`).

## Errors, warnings and logging

- One exception hierarchy rooted at `SolpropError` (`src/solprop/exceptions.py`): raise a
  specific subclass, never bare `Exception`.
- Never let NaN or Inf propagate silently: validate at the boundary, assert finiteness in
  tests.
- Warnings are errors in the test suite ⚙; assert an expected one with `pytest.warns`.
- `logging.getLogger(__name__)` per module; the package installs a `NullHandler`; no
  handler or level configuration outside `solprop.configure_logging()`. Debug logging is
  welcome in the physics, translation and driver layers, guarded by
  `logger.isEnabledFor(logging.DEBUG)` when formatting costs anything. **Nothing logs in
  compiled kernels or per-stage code**; progress goes through the one `ProgressReporter`
  protocol in `solprop/log.py`.

## Functions and modules

- One function, one responsibility. Keep parameter lists short ⚙; group related
  parameters into a small frozen dataclass.
- Return new values rather than mutating arguments, except for the backends' named work
  buffers.
- Organise modules by domain concept, each with a nameable purpose.

## Docstrings and comments

- Every public module, class and function has a **NumPy-style** docstring ⚙ validated by
  `numpydoc lint` ⚙: an imperative one-line summary, then `Parameters`, `Returns`,
  `Raises`, `Examples`, `References` as applicable. State **units, valid ranges and array
  shapes** (in the standard axis names). `Examples` run as doctests ⚙, so they are
  deterministic. A frozen dataclass documents its fields under `Attributes`.
- Comments explain **why**, never **what**: the physics and mathematics (or the `docs/`
  page with the derivation), the numerics (why this quadrature, cutoff or tolerance;
  where precision is lost), non-obvious sign, unit and normalisation choices, references
  (DOI, equation number), and anything a reader would otherwise "simplify" incorrectly,
  including a measured fact with its ledger entry. Mark follow-ups `# TODO(context): ...`.

## Tests

The long form is [`testing.md`](docs/dev/testing.md).

- **pytest**, in `tests/`, mirroring the package; `test_<unit>_<behaviour>`. Every
  public function has normal, edge and error-path tests; every bug fix brings a
  regression test.
- Deterministic and isolated ⚙ (random order is enforced): fixtures, `parametrize`, no
  network, no writes outside `tmp_path`; randomness through
  `np.random.default_rng(seed)` ⚙.
- Numerical assertions use `assert_allclose`, `pytest.approx` or `tests/bounds.py`'s
  `assert_within` with explicit tolerances, each with a comment saying what sets it: a
  rounding budget fixed before the comparison (matched discrete computations) or an
  accuracy envelope against a tighter or stored reference (adaptive runs, replay,
  resume). Exact equality is for structural contracts only (`testing.md` §3).
- Test invariants and behaviour, not only point values: conservation laws and symmetries,
  convergence orders (three levels), round trips, parity (backend vs reference, field vs
  envelope, batch vs sequential), and `hypothesis` where an invariant holds over a range.
- Golden arrays use `pytest-regressions` (`num_regression`), never a pasted literal;
  stored Luna.jl runs under `tests/data/` are regression witnesses, not truth.
- Tiers by purpose: fast tests unmarked, short functional runs `@pytest.mark.medium`,
  milestones and convergence studies `@pytest.mark.slow`; `@pytest.mark.gpu`,
  `@pytest.mark.benchmark` ⚙. A test that cannot run on a platform skips there and says
  why (`sys.platform == "win32"`).
- Coverage is 100 % (see the definition of done). Write the test alongside the code.

## Documentation

- Sphinx (myst-parser, numpydoc, autosummary, sphinx-gallery); the API reference is
  generated from docstrings.
- **All functionality is described in the user manual under `docs/`** — background, how
  it works (derivation or reference) and an example — not only in docstrings.
- Gallery kinds (`docs/dev/contributing.md`):
  - **Introductory lessons**: sources `docs/introduction/tutorials/plot_*.py`, rendered
    into `docs/introduction/lessons/`. **Focused tutorials**:
    `docs/tutorials/**/plot_*.py`, rendered into `docs/auto_tutorials/`. Both
    deterministic and large enough to show the physics; CI's docs job runs every one
    again on each change (`SOLPROP_RUN_EXAMPLES=tutorials`).
  - **Worked examples**: `docs/examples/**/plot_*.py`, full-size problems of minutes to
    half an hour, rendered into `docs/auto_examples/`.
  - Every gallery's output is committed, and a normal build (Read the Docs' included)
    executes none of them. After editing a script, even its prose, run
    `uv run poe docs-examples [plot_NN_name]` (every stale script, or one) on an idle
    machine and commit what it writes with the source, stating the run time of a worked
    example in the commit message. A normal build fails naming any script whose source
    differs from its committed output.
- The user guide under `docs/guide/` is tested: every `python` block runs in
  `tests/test_guide_snippets.py` (compiled in the default lane, run in the slow lane);
  keep snippets to seconds and fence illustration as `sh` or `text`.
- Keep `README.md` and the introduction current; record notable changes in
  `CHANGELOG.md` (Keep a Changelog) under `Unreleased`.
- The build passes with `-W -n --keep-going`.
- Design documents under `docs/dev/` describe the code as it is: a change that alters
  the design updates them in the same branch. A design problem found while working is
  raised, not silently worked around.

## Git

- **All development happens in a git worktree on a topic branch, never in the main
  checkout**, which stays on a clean `main`. Create
  the branch and the worktree yourself, without asking, named for the topic: `fix/…`,
  `feature/…`, `docs/…`, `chore/…` (`docs/sep-NNNN-…` for a SEP draft,
  `feature/sep-NNNN-…` for a SEP's implementation), for example
  `git worktree add -b fix/scan-lock ../solprop-scan-lock main`. Give the worktree its
  own environment (`uv sync` there).
- Clean, logical commits with descriptive but not verbose messages; several small
  commits rather than one large one. Stage deliberately with `git add <paths>`.
- Changes reach `main` as pull requests with green CI, or by a local merge when the
  maintainers ask for one.
- **Before opening an issue or a pull request,** read [`CONTRIBUTING.md`](CONTRIBUTING.md):
  search the issues, the pull requests and the SEPs, link what is related, and follow its
  sections and labels.
- **Coding agents:** merge, push, tag or release only when asked; never run a
  destructive or history-rewriting command (`git reset --hard`, `git checkout .`,
  `git restore` over uncommitted changes, `git clean`, `git commit --amend`,
  `git rebase`, `git push --force`) — it can destroy work that is not yours.
