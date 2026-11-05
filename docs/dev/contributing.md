# Contributing to solprop

Contributions can improve the models, numerical methods, interfaces, tests, or
manual. Begin with a reproducible example of the problem or a clear description
of the proposed change. The repository's `AGENTS.md` and `CONTRIBUTING.md` define
the working agreement; the detailed design and conventions are in this guide.

## Set up a development environment

solprop uses uv and Python 3.14. Clone the repository and synchronise its declared
development dependencies:

```sh
git clone https://github.com/LupoLab/solprop.git
cd solprop
uv sync
```

The native backend lives in `native/` in the same repository and is a second distribution,
solprop-native, at the same version and a required dependency. A plain `uv sync` therefore
builds it and installs it editable into the same environment, which needs its C++
toolchain, and `uv run poe check-all` runs solprop's gate and then the native gate; a change
that touches `native/` runs both and follows `native/AGENTS.md`.
[Building the native backend](../guide/performance/building-native.md)
covers its builds; the [architecture](architecture.md) and
[kernel contract](kernel-contract.md) describe the boundary between the two packages.

A checkout has three states, and only the first satisfies the exact requirement a release
carries:

| setup | native | metadata |
|---|---|---|
| `uv sync` | built from this tree | consistent: a workspace member, at the same version |
| the lightweight setup | none | unmet, deliberately |
| `uv run --no-sync poe fetch-wheel` | CI's wheel from this commit or a compatible ancestor | mismatched, deliberately: `uv pip check` reports it |

**The lightweight setup** serves Python and documentation contributors, with no native
package and no compiler:

```sh
uv run --no-project tools/setup_light.py
```

It runs `uv sync --locked --no-install-package solprop-native` and says how to continue. It is
a development workflow, not a way to run solprop in production. It supports the reference
backend and the lint, type, test and documentation checks. uv can't be configured otherwise:
`uv run` has no flag to skip a package, and `uv sync` ignores `UV_NO_SYNC`, so a later plain
`uv run` or `uv sync` builds native. Run the tools with the environment activated, or with
`UV_NO_SYNC=1`. There `backend="auto"` refuses, so scripts name `backend="numpy_ref"`; tests
take `SOLPROP_TEST_BACKEND=numpy_ref`, and native-only tests skip. A native change still needs
a native build and its qualification. Automation that needs no local build, such as Read the
Docs and the lanes on CI's wheel, uses the same two settings.

**`uv run --no-sync poe fetch-wheel`** installs a wheel CI built, for Linux x86-64, macOS arm64,
Windows x64 or Windows on Arm, using `gh` and no compiler. With no compiler, run the lightweight
setup first, because a plain `uv run` would try to build native:

```sh
uv run --no-project tools/setup_light.py
uv run --no-sync poe fetch-wheel
```

It takes the newest artifact from this checkout's
history only if the difference between that artifact's commit and the working tree, including
uncommitted changes, touches neither `native/` nor the solprop modules that `solprop_native`
imports. Anything else is refused, and so is an expired artifact; each refusal names `uv sync`
and the lightweight setup. A later plain `uv sync` or `uv run` builds native in its place, so
run with `uv run --no-sync`. The wheel keeps the commit it was built from, and `check_build`'s
comparison of `KERNEL_SCHEMA_VERSION` remains the guard at run time.

## Agree and implement a change

Work on a topic branch and keep changes focused. New physics, numerical schemes,
public-interface redesigns, and substantive contract changes require an accepted
[enhancement proposal](seps/sep-0000-process.md). Other changes larger than a
focused fix start with an agreed issue or written plan.

Keep the manual, docstrings, tests, and release-note fragments consistent with the
behaviour. Add a unique `changelog.d/<slug>.<category>.md` fragment for a notable change;
leave the shared changelog to the [rolling release PR](releasing.md#the-rolling-release-pr).
Check physical claims against their primary sources and report uncertainties.
Run fast checks and directly affected tests during development. Continuous
integration runs the full gate on every pull request,

```sh
uv run poe check
```

together with the slow and coverage lanes (`poe test-slow`, `poe cov`); a change is
complete when they pass there. A change to the documentation alone needs less; see
[documentation-only changes](testing.md#documentation-only-changes).

## Add a field to the run record

A run record holds every field of every dataclass it serialises, at its default too. Add
the field and nothing else: the configuration hash moves, and a file written before it
reads but is not resumed (ADR-066). Whether the output schema is bumped follows
[the schema rule](../guide/results/output.md#the-schema-version).

## Work on the documentation

Build the manual with `uv run poe docs`. It executes no gallery script: the
introductory lessons, the focused tutorials and the worked examples all have committed
output, which it reuses. After changing a gallery script, including its prose,
regenerate its output with `uv run poe docs-examples <name>` (or with no name, every
stale script) and include the generated files in the change.
Never edit generated gallery or API pages by hand.
