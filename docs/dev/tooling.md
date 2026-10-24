# Tooling

The configuration in `pyproject.toml` is the source of truth for every rule marked ⚙ in
`AGENTS.md`; this page explains the intent of each block so that the configuration can be
changed deliberately. If this page and `pyproject.toml` disagree, `pyproject.toml` wins and
this page is out of date.

## Environment

- **uv** manages Python (3.14) and dependencies; `uv sync` creates the environment, `uv add`
  / `uv add --dev` change it, `uv lock --check` runs in CI. Never `pip install` by hand; never
  edit `uv.lock`.
- The package version comes from git tags (`hatch-vcs`, falling back to `0.0.0` before the
  first tag); there is no version string to edit.
- The target-specific native backend, `solprop-native`, is the repository's second
  distribution, built from `native/`. The root is a uv workspace with `native/` as its
  member and one `uv.lock`; solprop's `native` extra pins `solprop-native` to solprop's own
  version (`hatch_build.py`), and both take that version from the repository's tags.
  `uv sync --extra native` builds and installs it editable; a plain `uv sync` leaves the
  environment a compiler-free NumPy/Numba one and removes it, as it removes a
  hand-installed Pyroraptor (`uv sync --inexact` and `uv run --no-sync` keep them).
- Dependencies resolve from PyPI. No local-path or private source is committed.
- Supported platforms: Linux, macOS and Windows for `numpy_ref` and `numba_cpu`;
  solprop-native as wheels for osx-arm64, linux-64, linux-aarch64 and win-64, and from
  source on Linux and macOS; the Queue and Slurm scan executors are
  POSIX-only and refuse to start on Windows.

## The gate: `uv run poe check`

A documentation-only change needs only the
[documentation checks](testing.md#documentation-only-changes).

Runs, in order: `ruff check --fix .` → `ruff format .` → `ruff check .` → `basedpyright` →
`numpydoc lint` (on `src/solprop` and the two figure scripts) → `codespell` → `reuse lint` → `poe test`
(fast and medium tiers, doctests included, under xdist) →
`sphinx-build -b html -W -n --keep-going docs docs/_build/html`. CI runs exactly the same
sequence plus `poe test-slow` and the CPU coverage lane; keep the two identical.

Other tasks: `poe test-medium` (the medium tier alone, timed against its budget),
`poe test-slow`, `poe cov` (interpreted bodies, then the compiled suite), `poe bench`
(the standard benchmark suite in `benchmarks/`, on a quiet machine;
[manual](benchmarking.md)), `poe docs`, `poe docs-clean`, `poe figures` (redraw the
committed structure and geometry SVGs) and `poe docs-examples` (execute the worked
examples and refresh their committed output).

The native package has a gate of its own, `uv run poe check-native`
(`native/scripts/check.sh`, bash, so Linux and macOS): its ruff and strict pyright,
clang-format, clang-tidy, the native tests, ASan and UBSan, its pytest suite on an
editable build, and the distribution wheel's build and isolated smoke test. `uv run poe
check-all` runs `check` and then `check-native`; a change that touches `native/` runs it.
Its rules are in `native/AGENTS.md`.

## ruff

`line-length = 88`, `target-version = "py314"`, `src = ["src", "tests"]`, excluding the generated
documentation trees. Rule families:
pycodestyle, pyflakes, isort, pep8-naming, pydocstyle (numpy convention), annotations,
pyupgrade, bugbear, simplify, comprehensions, pathlib, no-print, tidy-imports (absolute only),
return, unused-arguments, PGH (no blanket `noqa`), NumPy (no legacy random), pylint subset
(`max-args = 6`), ruff-specific (stale `noqa`). Ignored: `PLR2004` (magic values in numerics),
`ANN401` (a commented, confined `Any`), `F722` (pyflakes reads jaxtyping shape strings such as
`" n_omega"` as broken forward annotations; jaxtyping's own documentation asks for this
ignore). One pyupgrade hazard is *not* ignored but avoided: `UP037` unquotes a shape string
that looks like a quoted annotation — `"..."` becomes a bare `Ellipsis`, which jaxtyping
rejects — so shape strings are written with names (`"*shape"`, `" n_omega"`), never as
`"..."`, and forward references under `TYPE_CHECKING` are kept out of quoted form.
Per-file: tests drop docstring, annotation, magic-value and unused-argument rules (dummies
implement protocols with unused arguments); the independent oracles and rejected
prototypes under `tests/oracles/` and `tests/prototypes/` are held to the benchmark rules;
`docs/conf.py` drops docstring and annotation rules; gallery scripts and worked examples
drop them too and allow `print` and cell-style imports; benchmarks allow `print`.

## basedpyright

`include = ["src", "tests"]` without the oracles and prototypes (the prototypes, which
kernel tests import, are on `extraPaths`), `typeCheckingMode = "standard"` (strict once the baseline is
clean), `pythonVersion = "3.14"`, `failOnWarnings = true`, `reportMissingTypeStubs = false`
because numba, pyfftw and pyfluids ship no stubs (narrow ignores are still preferred over
broad ones); `scipy-stubs` supplies SciPy's. The checker is pinned to an exact version and
bumped in its own commit.

## pytest

`testpaths = ["tests", "src"]` (so `--doctest-modules` sees the `Examples` sections);
`addopts = "-ra --strict-markers --strict-config --doctest-modules -m 'not slow and not gpu
and not benchmark'"`; random order via `pytest-randomly`; `filterwarnings = ["error"]` with
targeted ignores; tier markers `medium`, `slow` and lane markers `gpu`, `benchmark`;
`xfail_strict = true`. `poe test` and `poe test-slow` use `pytest-xdist`
(`-n auto --dist loadgroup`), grouping tests by the module-scoped fixtures they share
(testing.md), explicit groups for independent heavy cases, and one thread per worker for every pool the suite opens
(`SOLPROP_NATIVE_THREADS`, OpenMP, OpenBLAS, MKL, Accelerate) locally and in CI; plain
`pytest`, whose `addopts` carries no `-n`, runs serially. Tier choice is by purpose, with budgets in
[testing.md](testing.md).

JUnit reports use `junit_family = "legacy"` so the numerical `record_property` fixtures
keep their bounds and measurements when `--junitxml` is requested. The default xunit2
format refuses those properties under the suite's warnings-as-errors policy.

## coverage

One configuration: `coverage-cpu.ini` (`source = solprop`, `branch = True`, nothing
omitted) for `poe cov` — `cov-kernels` with `NUMBA_DISABLE_JIT=1`, then `cov-rest` with
normal JIT, the second appending to the first. Only the compiled modules need the
interpreter to have their bodies counted, and interpreting the whole suite would make
every run in it interpret the kernels, so the first pass runs only the kernel-body suites.
The `[tool.coverage]` tables in `pyproject.toml` are the shared defaults. The lane requires
100 % of statements and branches of the whole package (`fail_under = 100`);
`# pragma: no cover` requires a reason on the same line.

## numpydoc validation

`[tool.numpydoc_validation] checks = ["all", "ES01", "SA01", "EX01", "GL01", "GL02"]` ("all"
minus extended summary, see-also and examples being mandatory, and minus the two quote-placement
rules that conflict with ruff's numpy docstring style), excluding `__init__` and test functions;
run by `numpydoc lint` in the gate and by the pre-commit hook. Sphinx renders docstrings with
the `numpydoc` extension alone (napoleon is not loaded: the two conflict).

## jaxtyping + beartype

`tests/conftest.py` installs the jaxtyping import hook for the whole session, with beartype
configured for the PEP 484 numeric tower
(`beartype.beartype(conf=beartype.BeartypeConf(is_pep484_tower=True))`), before importing
the package, so shape and dtype annotations are checked at runtime during tests only;
production imports are unaffected. numpydoc's `PR02` fires on any dataclass documented with
a `Parameters` section (the hook cannot see the generated `__init__`); document fields under
`Attributes`, or add `# numpydoc ignore=PR02` on the class line.

What the hook means for annotations:

- A plain `float` or an empty tuple is *not* an `ArrayLike` under beartype: annotate such
  parameters `ArrayLike | float` or `Sequence[float] | np.ndarray`.
- An `int` is accepted where a `float` is annotated (the PEP 484 numeric tower above).
- A return annotation that names the enclosing class is quoted (`-> "Grid"`, with
  `# noqa: UP037`): the hook evaluates it while the class body is still running.
- `numpy.typing.NDArray` cannot be used: beartype cannot interpret NumPy's generic alias.
  Use jaxtyping, the one array-annotation style.
- A dataclass whose `__post_init__` validates shape agreement between its fields gives
  them independent axis names, so that the validation stays reachable in production,
  where no hook runs.
- The hook is session-wide, so a branch written before a rule it enforces passes its own
  gate and fails on `main`: run the gate on the merged tree.

## pre-commit

ruff (check with `--fix`, then format), numpydoc validation, codespell, `uv-lock`, and
`reuse lint-file` on the staged files. Check the
`rev` pins against current releases when updating.

## Sphinx

`docs/conf.py`: myst-parser, numpydoc (without napoleon), autosummary (recursive, `generate = True`),
mathjax, intersphinx (numpy, scipy, matplotlib, h5py, numba, python), viewcode, matplotlib
plot directive, sphinx-gallery, pydata-sphinx-theme. sphinx-gallery executes the tutorials
(`docs/tutorials/` → `docs/auto_tutorials/`) on every build; the worked examples
(`docs/examples/` → `docs/auto_examples/`) take minutes each, so their generated pages,
figures and checksums are committed and reused, a normal build refuses a stale one, and
`poe docs-examples` (`SOLPROP_RUN_EXAMPLES`) executes them. Warnings are errors and
missing references are warnings (`-W -n`). `docs/dev/` is included in the manual under
"Development". Intersphinx inventories are fetched from each project's site
first and read from the vendored copies under `docs/_inventories/` when that fails, so the
build works offline and survives a flaky mirror; refresh the copies occasionally.

## Environment variables the tools honour

- `NUMBA_DISABLE_JIT=1` (the first coverage pass) and `NUMBA_NUM_THREADS`.
- The `numba_cpu` pools and plans: `SOLPROP_NUMBA_THREADS`, `SOLPROP_FFTW_THREADS`,
  `SOLPROP_BLAS_THREADS`, `SOLPROP_TRANSVERSE_FFTW_THREADS` (the transverse FFTW pool of a
  Cartesian run), `SOLPROP_NUMBA_REGIME` (the threading regime), `SOLPROP_FFTW_PLAN`
  (`estimate` in CI and tests, `measure` by default) and `SOLPROP_FFTW_WISDOM` (a wisdom
  file).
- `SOLPROP_MEMORY_BUDGET`, the memory admission budget ([backends.md §8](backends.md)).
- In the test suite: `SOLPROP_TEST_BACKEND` (the backend the driver tests select,
  `numba_cpu` by default) and `SOLPROP_WRITE_RECORDS=1` (rewrite the committed convergence
  records, announced at the end of the session).
- In the docs build: `SOLPROP_RUN_EXAMPLES` (set by `poe docs-examples`).

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every pull request, with
`SOLPROP_FFTW_PLAN=estimate`. Its first job, **changes**, compares the commit with its
base: when only `docs/` and Markdown files changed, static, docs and **docs-tests** run
(the test modules that open a documentation file: the SEP structure, the guide's
snippets, the accuracy page, the catalogues, the kernel contract and the figures), and
the test lanes below are skipped. Any other change, or a push whose base is unknown, runs
every job except docs-tests:

- **static** (Linux): `uv lock --check`, ruff (lint and format check), `poe types`,
  `poe docstrings`, `poe spell`, `poe reuse`.
- **docs** (Linux): `poe docs`.
- **test** (macOS arm64): `poe test`, the default lane.
- **coverage** (Linux x86-64): `poe cov` with the native package built from this tree
  (its paths in solprop count towards the 100 %), the default lane with coverage (so
  Linux runs the default lane once), and the report uploaded to Codecov for the badge.
- **slow** (Linux and macOS): `poe test-slow`; on macOS only its functional tests, the
  statistics-cost timing being Linux's.
- **windows**: `uv run poe test`, the default lane, on `numpy_ref` and `numba_cpu`.
- **oldest** (Linux): the package installed with `uv pip install --resolution
  lowest-direct`, so every dependency sits at its floor, then the default lane
  ([dependencies.md](dependencies.md)).
- **native-quality** (Linux): `uv run poe check-native`, the native package's gate
  (`native/scripts/check.sh`): its lint and strict types, clang-format, clang-tidy, the
  native tests, ASan and UBSan, its pytest suite, and the distribution wheel's build and
  isolated smoke test.
- **native-suite** (Linux): this tree's `native/` built as its CPU distribution wheel and
  installed, then the default lane with `SOLPROP_TEST_BACKEND=native`.

`.github/workflows/wheels.yml` builds the native release wheels (Linux x86-64 and
aarch64, macOS arm64, Windows x64) on pull requests that change the wheel recipe
(`native/ci/`, `native/pyproject.toml`, the CMake files, the workflow), weekly, on release
tags and by hand, tests each installed wheel, and publishes on a release tag once enabled
([Releasing and the native wheels](releasing.md)).

`numba_cpu` needs no job of its own: without solprop-native installed, `backend="auto"`
selects it, and the parity tests check it against `numpy_ref` in every job. GPU tests run
by hand on a node with a device (`pytest -m gpu`). The self-hosted
`.github/workflows/cuda.yml`, started by hand, builds the CUDA extension and runs the
native package's pytest suite and Compute Sanitizer on it. Keep the gate's steps identical to
`poe check`; a change to one is a change to the other, in the same commit.
