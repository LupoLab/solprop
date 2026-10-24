# Contributing to solprop

Thank you for your interest in solprop. Bug reports, questions about the physics or the
numerics, documentation fixes and new functionality are all welcome.

## Reporting a problem

Open an issue with the smallest script that shows it, what you expected, what you got, and
the output of

```sh
python -c "import solprop; print(solprop.__version__)"
python -m solprop_native      # only if the native backend is installed
```

For a physics question — a coefficient, a default, a model's validity range — say what
you compared against (a paper, an independent code, an analytic limit) and at what
parameters. Differences from Luna.jl that are deliberate are listed in the manual page
[Noteworthy differences from Luna](docs/guide/workflows/luna.md); check it first.

## Setting up

solprop uses [uv](https://docs.astral.sh/uv/) and Python 3.14:

```sh
git clone https://github.com/LupoLab/solprop.git
cd solprop
uv sync                 # the package, editable, with the development tools
uv run poe check        # the full gate: lint, format, types, docstrings, spelling, tests, docs
```

The material models come from [optmat](https://github.com/LupoLab/optmat), installed as
an ordinary dependency. Development works on Linux, macOS and Windows.

The optional compiled backend, solprop-native, lives in the same repository under
[`native/`](native/README.md) and is a second distribution with the same version. You do
not need it to contribute. To work on it, or to run the suite on it (Linux or macOS, with
a C++20 compiler, CMake and FFTW):

```sh
uv sync --extra native  # also builds and installs native/ (editable); a plain `uv sync` removes it
uv run poe check-all    # solprop's gate, then the native gate (`poe check-native`)
```

A change that touches `native/` runs `poe check-all` and follows the native rules in
[`native/AGENTS.md`](native/AGENTS.md).

## Making a change

1. Work on a branch named for its topic: `fix/…`, `feature/…`,
   `docs/…`. A major change — a new model, route or numerical scheme, a changed physics
   default, a redesigned interface — starts as an enhancement proposal (SEP): see
   [SEP 0](docs/dev/seps/sep-0000-process.md) and the
   [template](docs/dev/seps/template.md). For anything else larger than a focused fix —
   a new keyword, a changed public signature — open an issue first so the approach can be
   agreed.
2. Read [`AGENTS.md`](AGENTS.md): it holds the coding standards (units and sign
   conventions, types, docstrings, tests, documentation) for people and coding agents
   alike. The design is described under [`docs/dev/`](docs/dev/index.md); start with
   `architecture.md` and `conventions.md`.
3. Keep the diff to the task. Add tests with the code, document new functionality in the
   manual as well as in docstrings, and add a line to `CHANGELOG.md` under `Unreleased`.
4. Run fast checks and directly affected tests while you work. The full gate
   (`uv run poe check`) and the slow and coverage lanes run in CI on the pull request, so
   you need not run them locally first. A change to the documentation alone, including a
   new or edited SEP, needs only the
   [documentation checks](docs/dev/testing.md#documentation-only-changes).
5. Open a pull request describing what changed, why, and how it was verified. Say plainly
   what you assumed, skipped or could not make pass. CI runs on every pull request: the
   gate's checks and the default lane on Linux (with coverage) and macOS, the slow lane on
   both, the default lane on Windows, at the lowest supported dependency versions, and
   with the native CPU backend.

## Review checklist

A pull request is ready to merge when:

- The applicable checks pass: the documentation checks for a documentation-only change;
  otherwise the gate and CI, including the native CPU checks when the kernel seam or a
  backend changed.
- The physics is checked against a primary source or an independent derivation, with its
  reference state and validity range; algorithms are checked for conditioning,
  conservation, convergence and field/envelope, geometry and backend parity as relevant.
- Every numerical tolerance in a test is justified in a comment; every new dependency has
  been agreed (`docs/dev/dependencies.md`); the conventions of `docs/dev/conventions.md`
  are followed.
- The manual page, docstrings and changelog line are present; a worked example that was
  edited has been re-run (`uv run poe docs-examples <name>`).
- Backend changes update the memory ledger and the equivalence tests; a hot-path change
  records a benchmark (`docs/dev/benchmarks.md`).
- No full-length propagation runs in a test, benchmark or qualification lane.
  Introductory lessons, focused demonstrations and worked examples execute their
  physical problems through the gallery workflows.
- A pull request that implements a phase of a SEP appends the phase report to the SEP and
  updates its header, index entry and history; the one that completes it archives the
  SEP and links it from the manual pages it produced.

## Licence

By contributing you agree that your contributions are licensed under the project's
licences: code under the [GNU General Public License, version 3 or later](LICENSE)
(`GPL-3.0-or-later`), and documentation and figures under
[CC BY 4.0](LICENSES/CC-BY-4.0.txt).

The repository follows the [REUSE specification](https://reuse.software/): every file
states its copyright and licence. Start a new code file with an SPDX header, which
`uv run reuse annotate --copyright "John C. Travers and contributors" --year 2026
--license GPL-3.0-or-later <file>` writes; documentation, figures and data are covered
by `REUSE.toml`. `uv run poe reuse` (part of `poe check`) checks the whole tree.
