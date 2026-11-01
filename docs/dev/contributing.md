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
Run fast checks and directly affected tests during development. Continuous
integration runs the full gate on every pull request,

```sh
uv run poe check
```

together with the slow and coverage lanes (`poe test-slow`, `poe cov`); a change is
complete when they pass there. A change to the documentation alone needs less; see
[documentation-only changes](testing.md#documentation-only-changes).

## Add a field to the run record

A run record holds every field of every dataclass it serialises, at its default too,
and a resume compares records path by path
([what a resume compares](../guide/results/resuming.md#what-a-resume-compares)). A new
recorded field therefore needs a decision, and the frozen records of
`tests/output/test_record_stability.py` fail until it is made:

- if the field's value for an earlier file reproduces what the code did before it
  existed, add a {py:class}`~solprop.output.compatibility.Neutral` entry to
  {py:data}`~solprop.output.compatibility.NEUTRAL_VALUES`, naming the release it lands
  in; it is deleted with that release's frozen records after the next minor release;
- if it changes what runs compute, add no entry: earlier files are then refused, as a
  change of physics should be.

Classify the path in {py:data}`~solprop.output.compatibility.CLASS_TABLE` when it is
not physics: layout, numerics or provenance. A class moved between modules takes an
entry in {py:data}`~solprop.output.compatibility.CLASS_RENAMES`. Whether the output
schema is bumped follows [the schema rule](../guide/results/output.md#the-schema-version),
and a release adds its frozen records ([releasing](releasing.md)).

## Work on the documentation

Build the manual with `uv run poe docs`. It executes no gallery script: the
introductory lessons, the focused tutorials and the worked examples all have committed
output, which it reuses. After changing a gallery script, including its prose,
regenerate its output with `uv run poe docs-examples <name>` (or with no name, every
stale script) and include the generated files in the change.
Never edit generated gallery or API pages by hand.
