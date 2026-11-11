# AGENTS.md

Working instructions for coding agents (and people) in this repository.

**Read first:** the design lives under `docs/dev/`. Start at
[`docs/dev/index.md`](docs/dev/index.md), then [`architecture.md`](docs/dev/architecture.md)
and [`conventions.md`](docs/dev/conventions.md). This file is the short form: the rules,
each with a pointer to the page that explains it. Those pages win on any detail.
[`CONTRIBUTING.md`](CONTRIBUTING.md) is the short guide for a first contribution, and holds
the standard for issues and pull requests.

**Rules marked ⚙ are enforced automatically** by the configuration in `pyproject.toml`
(ruff, basedpyright, numpydoc, codespell, reuse, pytest) and `tools/size_budget.toml`. The
configuration is the source of truth ([`docs/dev/tooling.md`](docs/dev/tooling.md) explains
its intent); if a ⚙ rule and the configuration disagree, the configuration wins and this
file is out of date — say so. Everything else is a judgement call. When one concerns
physics or numerics rather than style, ask rather than guess.

## Project facts

| | |
|---|---|
| Packages | `solprop` (`src/solprop/`), the native backend `solprop-native` (`native/`, its own rules in [`native/AGENTS.md`](native/AGENTS.md)) and the desktop front end `solprop-gui` (`gui/`, its own rules in [`gui/AGENTS.md`](gui/AGENTS.md)): one repository, three distributions, one version |
| Purpose | Ultrafast nonlinear pulse propagation (UPPE and GNLSE; field and envelope; single-mode, multimode and general modal guides; radial, Cartesian, planar and line free space) |
| Python | 3.14 — equal to `requires-python` in `pyproject.toml` |
| Platforms | solprop runs where solprop-native installs: wheels for osx-arm64, linux-64, linux-aarch64, win-64 and win-arm64; from source on Linux and macOS (GPU builds on Linux); the Queue and Slurm scan executors POSIX-only |
| Backends | `native` (required; `auto`; CPU and GPU; implements every feature except the TDSE response) and `numpy_ref` (the numerical reference). Double precision only. No other backend (ADR-077) |
| Type checker | `basedpyright`, pinned exactly in the dev group |
| Materials | [optmat](https://github.com/LupoLab/optmat), the curated optical-materials package |
| Planned work | [GitHub issues](https://github.com/LupoLab/solprop/issues), not the roadmap, which lists only SEPs (ADR-062); [`CONTRIBUTING.md`](CONTRIBUTING.md) says how to write and label one |
| Enhancement proposals | `docs/dev/seps/` — process in SEP 0 (`sep-0000-process.md`), form in `template.md` |

## Commands

| Task | Command |
|---|---|
| Sync environment | `uv sync` (builds native editable; needs a C++20 compiler). `--extra gui --group gui-test` adds the front end. No compiler: the lightweight setup and `poe fetch-wheel` ([`docs/dev/contributing.md`](docs/dev/contributing.md)) |
| Full gate | `uv run poe check` |
| Native gate | `uv run poe check-native`; `uv run poe check-all` runs the three gates |
| Front-end gate | `uv run poe check-gui` (needs `--extra gui --group gui-test`) |
| Lint, format, types | `uv run ruff check --fix .`, `uv run ruff format .`, `uv run basedpyright` |
| Docstrings, spelling, licences | `uv run poe docstrings`, `uv run codespell`, `uv run poe reuse` |
| Size report and ratchets | `uv run poe size [--all] [--json]`; `uv run poe size-check` |
| Default tests (fast + medium) | `uv run poe test` (xdist, one thread per worker); `uv run pytest <path>` runs serially |
| Medium tier, slow lane, coverage (100 %) | `uv run poe test-medium`, `uv run poe test-slow`, `uv run poe cov` |
| GPU tests | `uv run pytest -m gpu` (a GPU build of solprop-native and a device) |
| Benchmarks | `uv run poe bench` plans; `uv run python -m benchmarks track` runs (on an idle machine) |
| Build docs; run the galleries | `uv run poe docs`; `uv run poe docs-examples [plot_NN_name]` |

Do not narrow a plain `pytest` run with `-m`: it replaces the default marker expression
in `addopts` and selects the slow tier too. Select by path or `-k` instead.

## Definition of done

**Documentation-only changes**, SEPs included: `git diff --check` and codespell on the
changed files, `uv run pytest tests/test_seps.py` when proposals, their index or the
roadmap change, and `uv run poe docs` when Sphinx content or links change (`testing.md`).

**Other changes:** develop first, test towards the end. Run the fast checks (ruff check
with `--fix`, ruff format, ruff check, basedpyright) and the modules that exercise what
changed (`uv run pytest -n auto <files>`). The full gate, slow and coverage lanes run in CI
on the pull request, not locally; after a CI failure, rerun the failing modules, never the
whole lane. `poe check` and CI run the same steps ([`tooling.md`](docs/dev/tooling.md)):
change both in the same commit. Coverage is 100 % of statements and branches; a
`# pragma: no cover` gives its reason on its line. `gui/` and `native/` add their own gates.

**Qualification scope:** `numpy_ref` and the native CPU build; GPU builds separately with
`pytest -m gpu` on a device. **No test, lane or timing runs a full-length propagation**
(a 0.2 m capillary, a metre of fibre): use about ten steps or a fibre that finishes in
seconds. Only the galleries run real problems at full length.

Never declare work finished while a required check fails, and never suppress a check to
make the gate green: say what failed and why. New functionality is done when it is
**tested**, **documented in the manual** as well as in docstrings, and has a `changelog.d/`
fragment ([format](changelog.d/README.md)); implementation PRs leave `CHANGELOG.md` to the
rolling release PR. A transcribed constant, coefficient, citation, reference state or
validity range is cited to its primary source; say what you verified and what you could not.

## Working agreement

- **Agree the plan for large changes.** A new model, route or numerical scheme, a changed
  physics default or convention, a public interface redesign or a substantive seam change
  starts from an accepted SEP; another change that touches more than a couple of files or
  alters a public signature, from an issue or a short agreed plan.
- **Ask before adding a dependency:** standard library, then NumPy/SciPy, then the list in
  [`dependencies.md`](docs/dev/dependencies.md), then ask. Dependencies come from PyPI; a
  floor is the oldest release that works, raised only for a recorded reason.
- **Re-derive, do not transcribe.** An algorithm or model from another code or a paper is
  re-derived, chosen deliberately, tested against an independent reference and documented;
  deliberate differences from Luna.jl go in `docs/guide/workflows/luna.md`.
- **Refactor what the task touches;** unrelated cleanup goes in its own commit or pull request.
- **Open an issue for what you find and do not fix** — a bug, a wrong statement, a missing
  test, a slow path — straight away, searching first ([`CONTRIBUTING.md`](CONTRIBUTING.md)).
  A fix says `Closes #N` in a pull request based on `main`, or comments on the issue.
- **Never hand-edit generated files**: `uv.lock`, `docs/_build/`, `docs/introduction/lessons/`,
  `docs/auto_tutorials/`, `docs/auto_examples/`, `docs/api/generated/`. Regenerate them.
- **Report failure honestly:** a failed check, an untested path, an invented assumption.
- **The public tree stands alone.** No local paths, host names, private notes or links to
  them, and no development-register identifiers, in code, tests, docs or the changelog.
  The hardware a benchmark ran on is part of its ledger entry.

## Code

### Principles

- One obvious way to do each thing: prefer the existing pattern, consolidate duplicates.
- Less code: extend an existing path before adding a new one.
- Optimise for the reader; clear beats clever. Small, single-purpose, preferably pure
  functions, with side effects (I/O, global state, mutation) at the edges.
- Make the implicit explicit: explicit arguments, types on interfaces, errors over silent
  fallbacks.
- Validate once at the public boundary; inner layers trust validated objects.
- Respect the layers (`architecture.md §3`): physics code holds no performance tricks,
  backend code no physics; the kernel contract (`kernel-contract.md`) and the protected
  driver surface (frozen in `tests/backends/test_driver.py`) are the crossings. No `utils`
  modules: organise by domain concept, in the package tree of `architecture.md §4`.

### Style

- `ruff format` and `ruff check` ⚙ own formatting and linting. Fix the cause; a suppression
  names its rule and a reason (`# noqa: E501 - URL must stay on one line`) ⚙.
- Names say what a thing is or does. Domain abbreviations (`fwhm`, `zdw`, `gnlse`) are
  fine, `tmp2` is not; Greek letters are spelled out (`omega`, `beta1`); a wavelength is
  `wavelength`, never `lambda`. f-strings, `pathlib.Path`, `encoding="utf-8"` on text I/O.
- Every code file starts with the SPDX header that `uv run reuse annotate --copyright
  "John C. Travers and contributors" --year 2026 --license GPL-3.0-or-later <file>`
  writes ⚙. Every public module declares its surface with `__all__`.

### Types and arrays

- Annotate every public signature and class attribute ⚙; no new basedpyright errors ⚙.
  `Sequence`/`Mapping` for read-only parameters, `X | None`, built-in generics; a confined,
  commented `Any` only where unavoidable. Structured data is a frozen `@dataclass`.
- **Arrays are annotated with jaxtyping on `np.ndarray`** with the axis names of
  `conventions.md §6`: `Complex[np.ndarray, "n_modes n_pol n_omega"]`. beartype checks them
  in the test session, which constrains annotations (`ArrayLike | float`, quoted
  self-returns, no `numpy.typing.NDArray`; `tooling.md`, "jaxtyping + beartype").

### Functions and errors

- One function, one responsibility. Signatures with more than five positional parameters
  are keyword-only ⚙ (PLR0917); group parameters into a dataclass only when the group is a
  domain concept. Return new values rather than mutating arguments, except the backends'
  named buffers.
- One exception hierarchy rooted at `SolpropError` (`exceptions.py`): raise a specific
  subclass, early and loudly; never swallow an error, return a sentinel where raising is
  correct, or let NaN or Inf propagate silently. Warnings are errors in tests ⚙.
- `logging.getLogger(__name__)` per module; only `solprop.configure_logging()` configures
  handlers. Debug logging is welcome in the physics, translation and driver layers, guarded
  by `logger.isEnabledFor` when formatting costs anything. **Nothing logs in kernels or per-stage code**; progress goes through the
  reporter protocols of `solprop/log.py` (`ProgressReporter` and those built on it).

### Docstrings and comments

- **Public API** — a dotted name with no single-underscore component, not in a private
  module and not nested — has a full **NumPy-style** docstring validated by numpydoc ⚙:
  an imperative summary line, then `Parameters`, `Returns`, `Raises`, `Examples`,
  `References` as applicable, with **units, valid ranges and array shapes**. `Examples` run
  as doctests ⚙. A frozen dataclass documents its fields under `Attributes`. A public
  property has a one-line docstring naming the quantity and unit, with
  `# numpydoc ignore=RT01` on the `def` line.
- **Private and nested code** has no `Parameters`, `Returns`, `Yields`, `Raises` or
  `Examples` sections (`poe size-check` caps those left ⚙): one summary line, then only what
  the signature and annotations don't say — a non-SI unit, a valid range, a condition, a
  convention, or why it exists. A function whose name and signature say everything needs
  no docstring; a non-obvious inner function gets a one-line comment.
- Comments explain **why**, never **what**: the physics (or the page with the derivation),
  the numerics (why this quadrature, cutoff or tolerance; where precision is lost), sign,
  unit and normalisation choices, references (DOI, equation), and anything a reader would
  otherwise "simplify" incorrectly. Mark follow-ups `# TODO(context): ...`.

## Conventions: units, signs, normalisation, layout

The single source is [`conventions.md`](docs/dev/conventions.md). Every docstring defers
to it; a function that departs from it says so in the first line of its docstring.

| Quantity | Convention |
|---|---|
| Base units | SI exclusively (m, s, J, W, V/m, Pa, K, rad/s, m⁻³); pressures accepted in bar at the user interface only |
| Fourier sign | **Optics convention at the interface**: `Ẽ(ω) = ∫ E(t) e^{+iωt} dt`, plane wave `e^{i(βz − ωt)}`, delay `e^{+iωτ}`. **FFT convention inside the seam**: raw forward DFTs, `e^{i(ωt − βz)}`. The boundary is a conjugation in exactly three named places (`fourier.to_optics`/`to_internal`); HDF5 files are optics-convention (`conventions.md §2`) |
| Linear operator (internal) | `D = −i(β − β₀ − β₁(ω − ω₀)) − α/2`, `Im n_eff > 0` is loss |
| Amplitudes | interface modal amplitude `b = sqrt(Re n_eff) a` in true √W; the internal `a` is vacuum-normalised (`conventions.md §4`) |
| Kernel state layout | `(n_modes, n_pol, n_omega)` complex128, ω contiguous, one trajectory (a group is a sequence of problems); saved fields `(n_z, n_modes, n_pol, n_omega)` |
| Axis names | `n_batch n_modes n_pol n_omega n_omega_fine n_t n_to n_nodes n_z n_intervals n_deg n_windows`, reused verbatim in annotations |

## Backends, kernels and hot paths

The long form is [`backends.md`](docs/dev/backends.md).

- **Where features land** (ADR-068, ADR-077). A new feature lands on `native` and
  `numpy_ref`, in either order, and `native` may shape the contract. A backend without a
  feature refuses it by name; nothing falls back silently.
- **`numpy_ref` decides the numbers:** `native` is validated against it (one linear
  application, one nonlinear evaluation, one step, eight steps, each statistic, a full run)
  with tolerances derived from summation bounds and asserted `> 0`.
- Hot code lives only under `src/solprop/backends/`, `src/solprop/translate/` and
  `native/` ([`native/AGENTS.md`](native/AGENTS.md)). **No allocation, logging, Python
  objects or exceptions inside kernels**; buffers are parameters, listed by the tested
  `memory_ledger()`, and in-place or `out=` work stays inside the backends; failures are
  status codes. Device arrays never cross the `Backend` protocol.
- **Reproducibility to the error bound** ([SEP 25](docs/dev/seps/archive/sep-0025-rounding-level-reproducibility.md);
  `testing.md` §3): results are compared by numerical error, never bits — matched
  computations within a rounding budget, adaptive runs and replay within an accuracy
  envelope. Kernels may reorder, vectorise, thread and contract; protected helpers stay
  strict; no umbrella fast-math.
- **The seam has no version number** (ADR-068). A change to the `PropagationProblem` or
  `RunRequest` fields or the stage bundles of `translate/stage.py` rewrites the frozen
  tables in `tests/backends/test_driver.py` in the same commit, with a release-note fragment.
- A hot-path change comes with a benchmark ledger entry ([`benchmarks.md`](docs/dev/benchmarks.md))
  from `python -m benchmarks ab` on about ten steps or a short `--length` window; report the
  window, never extrapolate it.

## Tests

The long form is [`testing.md`](docs/dev/testing.md).

- **pytest** in `tests/`, mirroring the package; `test_<unit>_<behaviour>`. Every public
  function has normal, edge and error-path tests; every bug fix a regression test.
- Deterministic and isolated: random order ⚙, no network, no writes outside `tmp_path`,
  randomness through `np.random.default_rng(seed)` ⚙.
- Shared builders go in `tests/builders.py` rather than local copies; guard tests are
  parametrised tables; no measurement or experiment scripts in `tests/`.
- Numerical assertions (`assert_allclose`, `pytest.approx`, `tests/bounds.py`'s
  `assert_within`) carry explicit tolerances, each with a comment on what sets it: a
  rounding budget fixed beforehand, or an accuracy envelope against a tighter or stored
  reference. Exact equality is for structural contracts only (`testing.md` §3).
- Test invariants, not only point values: conservation laws, symmetries, convergence orders
  (three levels), round trips, parity (backend, field/envelope, batch), `hypothesis` over
  ranges. Golden arrays use `num_regression`; stored Luna.jl runs are witnesses, not truth.
- Tiers by purpose: fast unmarked, short functional runs `medium`, milestones and
  convergence studies `slow`; lanes `gpu` and `benchmark` ⚙. A test that cannot run on a
  platform skips there and says why.

## Documentation

- Sphinx (myst-parser, numpydoc, autosummary, sphinx-gallery), building clean with
  `-W -n --keep-going`; the API reference comes from docstrings.
- **All functionality is described in the user manual** — background, how it works and an
  example — not only in docstrings. Design pages under `docs/dev/` describe the code as it
  is and change with it, in the same branch.
- Gallery output is committed and a normal build runs no script: after editing a lesson,
  tutorial or worked example, even its prose, run `uv run poe docs-examples <name>` and
  commit what it writes ([`contributing.md`](docs/dev/contributing.md)).
- Every `python` block of `docs/guide/` runs in `tests/test_guide_snippets.py`: keep them to
  seconds and fence illustration as `sh` or `text`.
- Keep `README.md` and the introduction current. Notable changes get a unique
  `changelog.d/<slug>.<category>.md` fragment
  ([releasing](docs/dev/releasing.md#the-rolling-release-pr)).

## Enhancement proposals (SEPs)

The long form is [SEP 0](docs/dev/seps/sep-0000-process.md); the index is
[`docs/dev/seps/index.md`](docs/dev/seps/index.md).

- **When:** new physics models, routes, geometries, mode families or numerical schemes;
  changed defaults or conventions; public interface redesigns; substantive
  kernel-contract changes; process changes. Not for fixes, documentation, tests,
  refactoring, tooling or optimisations that leave results unchanged.
- **Writing:** from `template.md`, standing alone (SEP 0 §7), with its expected cost in
  code, docstring and test lines; merged early as `Draft`.
- **Implementing:** one `feature/sep-NNNN-…` branch and one pull request, a draft from the
  start of implementation, marked ready when its last commit sets `Final`. Each phase
  appends its report to Part II, with the actual cost. `Final` needs parity on `numpy_ref`
  and native CPU, or an exception written into the accepted SEP.

## Git

- **Develop in a git worktree on a topic branch, never in the main checkout.** Create both
  yourself, named for the topic (`fix/…`, `feature/…`, `docs/…`, `chore/…`;
  `docs/sep-NNNN-…` and `feature/sep-NNNN-…` for SEPs), for example
  `git worktree add -b fix/scan-lock ../solprop-scan-lock main`, and `uv sync` there.
- Small, logical commits with brief messages; stage with `git add <paths>`. Manage
  dependencies with `uv add` (after asking), never `pip install`, and commit the lockfile a
  change alters. The version comes from git tags: set no version string.
- Changes reach `main` as pull requests with green CI, or by a local merge the maintainers
  ask for. **Few pull requests, few pushes:** CI runs its whole matrix on every push, so
  keep one agent's work in one pull request, stack only on another agent's open one when
  your work needs it, and push in batches. Before opening an issue or a pull request, follow
  [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **Coding agents:** merge, push, tag or release only when asked; never run a destructive
  or history-rewriting command (`git reset --hard`, `git checkout .`, `git restore` over
  uncommitted changes, `git clean`, `git commit --amend`, `git rebase`,
  `git push --force`) — it can destroy work that is not yours.
