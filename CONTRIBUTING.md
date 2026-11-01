# Contributing to solprop

Thank you for your interest in solprop. Bug reports, questions about the physics or the
numerics, documentation fixes and new functionality are all welcome.

## Check what already exists

Before opening an issue or a pull request, and before starting work on either, look for
what already covers it or touches it:

- the [issues](https://github.com/LupoLab/solprop/issues) and
  [pull requests](https://github.com/LupoLab/solprop/pulls), open and recently closed;
- the [SEPs](docs/dev/seps/index.md), since one may already propose the change or be about
  to alter the code;
- the [roadmap](docs/dev/roadmap.md), until it is gone.

```sh
gh issue list --state all --search "collar"
gh pr list --state all --search "collar"
git grep -il "collar" docs/dev/seps docs/dev/roadmap.md
```

If one already covers your work, add to it rather than opening another. If one only
touches it, such as an open pull request that changes the same files, link it from what
you open, even if you need nothing from it: GitHub shows the reference on both sides, and
the next reader then knows the two are connected. For a bug report, a quick search of the
issues is enough.

## Reporting a problem

Use the [report form](https://github.com/LupoLab/solprop/issues/new?template=report_a_problem.yml),
which asks for the smallest script that shows it, what you expected, what you got, and the
output of

```sh
python -c "import solprop; print(solprop.__version__)"
python -m solprop_native      # only if the native backend is installed
```

For a physics question — a coefficient, a default, a model's validity range — say what
you compared against (a paper, an independent code, an analytic limit) and at what
parameters. Differences from Luna.jl that are deliberate are listed in the manual page
[Noteworthy differences from Luna](docs/guide/workflows/luna.md); check it first.

That is all a report needs. A maintainer fills in the rest.

## Writing an issue

Planned work lives in [GitHub issues](https://github.com/LupoLab/solprop/issues), not on
the roadmap, which takes no new items (ADR-062). So an issue for planned work, written by
a person or an agent, has to let someone pick up the task from its text alone.
[#76](https://github.com/LupoLab/solprop/issues/76) and
[#77](https://github.com/LupoLab/solprop/issues/77) are examples.

[Check what already exists](#check-what-already-exists) first, and link what you find.
Write one issue per coherent piece of work, not one per finding or per file. A major
change starts as a SEP, which has a tracking issue of its own
([SEP 0](docs/dev/seps/sep-0000-process.md),
[template](.github/ISSUE_TEMPLATE/sep_tracking.md)).

Use these sections, which the [planned work template](.github/ISSUE_TEMPLATE/planned_work.md)
lays out, leaving out any that do not apply:

- **What happens**, or what is missing, so that someone new to this code can follow it.
- **Why it matters:** what a user or a developer gets wrong or pays for today.
- **Reproduction.** For a bug, the smallest script that shows it, which you have run,
  with its output. For a performance issue, the numbers, the machine, the build and the
  command.
- **Context:** links to what a reader needs, namely the code, the manual, SEPs, ADRs, pull
  requests and other issues. Link code as a permalink to a commit (press `y` on a file's
  page on GitHub), and say which commit you checked against.
- **To do:** checkboxes that are actions, not restated symptoms. Name what must not change
  or break. Check that a suggested fix would work before writing it down.
- **Done when:** what someone can check to know it is finished, with a tolerance and a
  machine for a measurement. A criterion should cover every box, and one that no pull
  request can show should say what stands in for it.

Be exact about what you know:

- Say what you ran and what you only read ("read, not run"), and label a suspicion as one.
- Give a measurement its machine, width and date, and a count its scope ("273 lines under
  `src/` and `native/src/`").
- Mark a choice that is not the contributor's with **Needs a decision:**, saying who
  decides and what you recommend. A physics judgement, a changed default and a changed
  public signature are the maintainers' ([`AGENTS.md`](AGENTS.md)).
- Issues are public, so leave out local paths, private notes and nicknames for machines.
  Name hardware by its model.

### Labels

Issues are filtered by [label](https://github.com/LupoLab/solprop/labels), so an issue
carries:

- one **type**, or two if it really is both: `bug` (wrong behaviour or a wrong result),
  `performance` (slower or larger than it should be, with no wrong result), `enhancement`
  (a new feature or a request), `documentation`, `refactor` (no change in behaviour) or
  `testing` (tests, test tiers and CI coverage);
- one **priority**: `priority: high` (wrong results, wrong public statements or a user's
  first contact), `priority: medium` (real work that should not wait long) or
  `priority: low` (worth doing, no hurry);
- an **area** where one fits, such as `area: native` or `area: boundaries`;
- `sep` on a SEP's tracking issue, and `good first issue` if it needs no physics judgement
  and no special hardware.

If you cannot set labels, suggest them in the text and a maintainer will.

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

1. [Check what already exists](#check-what-already-exists), so that you neither repeat
   nor collide with work in hand. Look especially for open pull requests that change the
   same files and for SEPs that would change the same behaviour.
2. Work on a branch named for its topic: `fix/…`, `feature/…`,
   `docs/…`. A major change — a new model, route or numerical scheme, a changed physics
   default, a redesigned interface — starts as an enhancement proposal (SEP): see
   [SEP 0](docs/dev/seps/sep-0000-process.md) and the
   [template](docs/dev/seps/template.md). For anything else larger than a focused fix —
   a new keyword, a changed public signature — open an issue first so the approach can be
   agreed.
3. Read [`AGENTS.md`](AGENTS.md): it holds the coding standards (units and sign
   conventions, types, docstrings, tests, documentation) for people and coding agents
   alike. The design is described under [`docs/dev/`](docs/dev/index.md); start with
   `architecture.md` and `conventions.md`.
4. Keep the diff to the task. Add tests with the code, document new functionality in the
   manual as well as in docstrings, and add a line to `CHANGELOG.md` under `Unreleased`.
5. Run fast checks and directly affected tests while you work. The full gate
   (`uv run poe check`) and the slow and coverage lanes run in CI on the pull request, so
   you need not run them locally first. A change to the documentation alone, including a
   new or edited SEP, needs only the
   [documentation checks](docs/dev/testing.md#documentation-only-changes).
6. Open a pull request describing what changed, why, and how it was verified. Say plainly
   what you assumed, skipped or could not make pass. Search again first, since work may
   have appeared while you were busy. Link the issue it closes (`Closes #N`) and any
   issue, pull request or SEP it overlaps with or depends on, and give it a label or two:
   its type and, where one fits, its area, as for an issue. If you cannot set labels, a
   maintainer will. CI runs on every pull request: the gate's checks and the default lane
   on Linux (with coverage) and macOS, the slow lane on both, the default lane on Windows,
   at the lowest supported dependency versions, and with the native CPU backend.

## Review checklist

A pull request is ready to merge when:

- It links the issues, pull requests and SEPs it closes, overlaps with or depends on.
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
