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

The optional native backend lives in `native/` in the same repository and is a second
distribution, solprop-native, at the same version. `uv sync --extra native` builds it and
installs it editable into the same environment, and `uv run poe check-all` runs
solprop's gate and then the native gate; a change that touches `native/` runs both and
follows `native/AGENTS.md`. [Building the native backend](../guide/performance/building-native.md)
covers its builds; the [architecture](architecture.md) and
[kernel contract](kernel-contract.md) describe the boundary between the two packages.

## Agree and implement a change

Work on a topic branch and keep changes focused. New physics, numerical schemes,
public-interface redesigns, and substantive contract changes require an accepted
[enhancement proposal](seps/sep-0000-process.md). Other changes larger than a
focused fix start with an agreed issue or written plan.

Keep the manual, docstrings, tests, and changelog consistent with the behaviour.
Check physical claims against their primary sources and report uncertainties.
Run fast checks and directly affected tests during development, and the full gate
before treating a change as complete:

```sh
uv run poe check
```

Continuous integration runs the slow and coverage lanes (`poe test-slow`,
`poe cov`). A change to the documentation alone needs less; see
[documentation-only changes](testing.md#documentation-only-changes).

## Work on the documentation

Build the manual with `uv run poe docs`. This executes introductory lessons and
focused demonstrations, and reuses
the committed output of the worked examples. After changing a worked-example
source, including its prose, regenerate its output with
`uv run poe docs-examples <name>` and include the generated files in the change.
Never edit generated gallery or API pages by hand.
