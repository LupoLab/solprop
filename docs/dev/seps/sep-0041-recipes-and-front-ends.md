# SEP 41 — Recipes, and the hooks a front end needs

| | |
|---|---|
| **SEP** | 41 |
| **Title** | Recipes, and the hooks a front end needs |
| **Author** | John C. Travers |
| **Drafting agent** | Claude Opus 5.5 (2026-10-31) |
| **Status** | Accepted |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); Opus 5.5 (AI); Codex (AI) |
| **Type** | Change |
| **Created** | 2026-10-31 |
| **Resolution** | Accepted 2026-11-01 by John Travers, after the maintainer's human review, three AI reviews and the responses to them, and decisions Q1–Q12 (Part III) ([LupoLab/solprop#112](https://github.com/LupoLab/solprop/pull/112)). |
| **Requires** | SEP 42 (resuming by what matters to the run; accepted) |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

solprop is driven only from Python today. A complete simulation is a call to one of six
`prop_*` functions, and nothing in solprop can write that call down, read it back, or
explain its parameters to anything other than a person reading the docstrings. This SEP adds
a **recipe**: a versioned TOML description of a simulation that solprop reads, validates,
builds, runs, exports as a plain Python script, records in its output and resumes from.
Around it come the smaller facilities that any front end needs from solprop instead of
copying from it:

- parameter metadata, structured validation and a capability query;
- a supported observer of saved planes, scan progress, a cost estimate and a portable stop;
- launch into chosen modes, beam displacement and tilt;
- design solvers for every guided route, and pinned parameters that use them;
- previews of the modes and the input, and a statement of what a results file can give;
- measured-pulse loaders and geometry drawings.

The first user is a desktop front end being designed separately; scripts and batch systems
gain the same things (`solprop run recipe.toml`). Success is judged by round trips, with the
noise seed fixed: a recipe and the Python call it stands for build the same `Simulation`
(equal `configuration_hash`); a recipe exported to Python and run gives the same run,
execution settings included; and a file written from a recipe resumes from the file alone.

The **Python interface changes are additive**: every name is new and every new keyword
defaults to today's behaviour. No output-schema bump is needed: the recipe is a top-level
group written once, and the launch and beam changes (S5, S6) add recorded fields, which SEP
42 makes additive. This SEP therefore lands after SEP 42.

### 2. Motivation and scope

A front end must:

- show every parameter with its unit, default, range and documentation;
- check a setup as it is typed and put each error next to its field;
- offer only the options a route and species support;
- watch a run as it goes, and stop it;
- save a setup and reopen it, and reopen a results file with the setup that made it;
- say whether a stopped run can be resumed, and resume it;
- say what each results file can show.

Today each of these would mean a hand-written copy of rules that live in solprop's code, and
the copy would drift with every release. The principle of this SEP is that **solprop holds
the physics, the defaults and the rules; a front end holds only presentation**.

The same gaps hurt script users. A results file does not record the call that made it, so a
run can be resumed only by the script that started it (the [resume
key](../../guide/results/resuming.md) checks a rebuilt simulation; it cannot rebuild one). A
batch job is a Python file per configuration. A scan's builder must be a module-level
function for pool executors ([scans](../../guide/workflows/scans.md)), which a declarative
description gives for free.

**In scope** (the labels S1–S18 are used throughout):

| | Change | Section |
|---|---|---|
| S1 | `solprop.recipe`: the TOML format, parameter paths, pins, load, save, build, Python export, `solprop run` | 5.1 |
| S2 | The recipe recorded in the output; resume from a file alone | 5.2 |
| S3 | A supported observer of each saved plane | 5.3 |
| S4 | Progress for scans | 5.4 |
| S5 | Launch into chosen modes on modal routes | 5.5 |
| S6 | Displacement and tilt on free-space beams | 5.6 |
| S7 | Design solvers for every guided route; radius from a zero-dispersion wavelength | 5.7 |
| S8 | A capability query for routes and species | 5.8 |
| S9 | Machine-readable parameter metadata | 5.9 |
| S10 | Structured validation errors and warnings, a check and grid suggestions | 5.10 |
| S11 | A cost estimate before a run | 5.11 |
| S12 | Measured-pulse loaders | 5.12 |
| S13 | Geometry drawings in the library | 5.13 |
| S14 | Route keywords: fill the gaps, document the deliberate differences | 5.14 |
| S15 | The writer-lock sidecar, documented in the output guide | 5.15 |
| S16 | Result capabilities: what a results file can give | 5.16 |
| S17 | A portable stop, and a resumability query | 5.17 |
| S18 | Previews: the modes and the input a recipe would build | 5.18 |

**Out of scope:**

- the front end itself, which is a separate project;
- remote execution and a network service (a recipe and `solprop run` are how remote work is
  done for now, through the existing executors);
- arbitrary Python inside a recipe (custom pulse shapes or responses, callables): the way
  out is the Python export;
- a graphical recipe editor in solprop;
- extending a finished or stopped run onto a changed grid (Q4);
- any change to the physics, the kernel contract or the steppers.

### 3. Context

- **Entry points.** [`prop_gnlse`, `prop_capillary` and
  `prop_waveguide`](../../guide/setup/running.md) take keywords. `prop_radial`, `prop_free`
  and `prop_planar` take a `TimeGrid` first, then objects: `space`, `responses`, `initial`
  and the keyword `settings`, all three with an optional `frame`; `prop_radial` and
  `prop_free` also take a `medium`. Each has a `Simulation.from_*` constructor, so a
  simulation can be built without running it, and `Simulation.run(output, progress=...,
  resume=..., memory_budget=..., tdse_archive=..., snapshots=...)` runs it. The keyword
  surfaces differ: the `prop_*` functions also take `output`, `progress`, `progress_period`,
  `resume`, `snapshots` and, on the capillary, `tdse_archive`, which `from_*` does not;
  `prop_radial`, `prop_free` and `prop_planar` also take `absorber` and `absorber_scheme`,
  which their `from_*` hold in `settings`; `boundary_check` is on every `from_*`;
  `from_waveguide` takes `memory_budget`, which admits a guide's tables at build; and
  `fields`, `field_planes` and `resume_history` are on no keyword surface, only on
  `Simulation.settings`.
- **Names.** The route keywords name the reference wavelength `wavelength` and the pulse
  shape `shape`, as the pulse objects and `solprop.tools` do (since the 0.1.0 release
  preparation, PR #64).
- **Progress.** [`ProgressReporter`](../../guide/performance/progress.md) is a
  `runtime_checkable` protocol with four calls (`started`, `step`, `plane`, `finished`),
  extended by optional protocols (`InterruptibleReporter.interrupted`,
  `SummarisingReporter.summary`) that a reporter may implement; `NullReporter` implements
  both. Membership is checked by name. `plane(z, summary)` receives the plane's cost, not
  its field. A reporter's exception propagates through the run loop, which records the run
  as failed.
- **Output and resume.** The HDF5 output (ADR-058) records the resolved configuration,
  metadata and `configuration_hash`. A file of any other output schema is refused wherever
  it is opened. SEP 42, on which this SEP rests, makes the resume gate a comparison of the
  stored and requested records, adds `resume="force"`, and makes a new recorded field or a
  top-level group additive. When a call gives no noise seed, the rebuild takes the seed
  recorded in the file. The call's arguments are not stored, and the stored configuration
  cannot rebuild a simulation. `RunOutcome.status` is `running`, `complete`,
  `budget_exhausted` or `failed`. SIGTERM and Ctrl-C raise `RunTerminated`, a `SystemExit`,
  at once, and are held only while a commit is being written; the state is written, and
  `resume=True` continues. A dense run whose last plane is interpolated resumes only with
  `resume="interpolated"`.
- **Scans.** `run_scan(scan, build, executor=...)` already reports progress: `"auto"` gives
  an outer bar or one log line per finished point; the inner runs are quiet. A scan's
  builder must pass `point.seed`. `python -m solprop.scan` is the command-line entry point.
- **Design tools.** [`solprop.tools`](../../guide/workflows/design-tools.md) has
  `pressure_zdw`, `pressure_rdw`, `capillary_design`, `soliton_params` and others, built on
  the Marcatili capillary; `pressure_rdw` also takes a `lattice` and builds a
  `VincettiMode`. `pressure_zdw` brackets once over pressure, inside a zero-dispersion
  search that samples $\beta_2$ over wavelength and discards sign changes through a pole;
  `pressure_rdw` samples 129 logarithmic and 129 linear pressures. A mode need not know its
  fill's Kerr response. Modes expose `beta`, `beta2`, `zdw`, `aeff` and loss.
- **Modes.** `prop_waveguide` already selects a launch mode with `channel`, by index or
  display label (labels display as `HE1,1:even`), or computes a launch from an `incident`
  beam and a `coupling`. General modal bases measure physical energy with a flux metric that
  has cross terms ([general modal](../../guide/waveguides/general-modal.md)).
- **Errors.** The builders raise `SolpropError` subclasses (`DriverError`,
  `TranslationError`, `ModeError`); `DriverError` is also a `RuntimeError`, not a
  `ValueError`. Most carry a message only; `ResponseBudgetError` carries `z`, `field` and
  `calls`. Warnings (`MaterialRangeWarning`, `BoundaryWarning`) carry a message.
- **Memory.** `solprop.driver.memory` already estimates a run's persistent memory and admits
  it against a budget (`MemoryBudgetError`).
- **Pulses and beams.** `DataPulse` takes arrays and keeps a supplied unwrapped phase; there
  are no file loaders. `SpatialPulse(temporal, waist, order, propz, jones)` has no
  displacement or tilt; `solprop.coupling.Alignment(displacement, tilt, model_error)`
  applies them as a coupling step, with the tilt a phase screen in the medium's real wave
  number.
- **Results.** The [processing](../../guide/results/processing.md) functions accept spatial
  files but do not return physical powers from them; compression refuses spatial and general
  modal outputs; general modal reconstruction needs the guide and reference tables, of which
  a file may hold only the identities.
- **Figures.** The geometry drawings in the user guide are generated by a docs script and
  styled with CSS custom properties, which only a browser renders.
- **The writer lock.** The `*.writer.lock` sidecar is documented under snapshots in the
  [output guide](../../guide/results/output.md).

**Open issues.** From a keyword search of this SEP's subjects against the open issues,
reading those that matched. It fixes none in full.

- **To resolve first:**
  - [#86](https://github.com/LupoLab/solprop/issues/86): the names to settle in 0.x. Every
    keyword joins recipe `format = 1` when this SEP merges, and a rename then needs a format
    upgrade rule too.
  - [#87](https://github.com/LupoLab/solprop/issues/87): one table for the route defaults.
    S9's resolvers should read that table rather than copy the rules a third time.
- **It changes, or meets the same code as:**
  - [#98](https://github.com/LupoLab/solprop/issues/98): underscore fields in the signature
    of `SimulationSettings`, which S9's sync test checks against.
  - [#5](https://github.com/LupoLab/solprop/issues/5),
    [#6](https://github.com/LupoLab/solprop/issues/6),
    [#7](https://github.com/LupoLab/solprop/issues/7) and
    [#89](https://github.com/LupoLab/solprop/issues/89): the memory estimate that S11
    reports.
  - [#125](https://github.com/LupoLab/solprop/issues/125): `Recipe.scan()` builds each point
    afresh and passes its seed, which avoids this.
  - [#131](https://github.com/LupoLab/solprop/issues/131): reading a scan file's recipe (S2)
    goes through `ScanOutput`'s schema check.
  - [#83](https://github.com/LupoLab/solprop/issues/83) and
    [#95](https://github.com/LupoLab/solprop/issues/95): the boundary warning, which S10
    gives a `parameter`.
  - [#109](https://github.com/LupoLab/solprop/issues/109): a refactor of
    `driver/simulation.py`.

### 4. Theory

No new physics. Four definitions matter:

- **Recipe equivalence.** A recipe $R$ and a Python call $C$ are equivalent when
  `Recipe(R).build()` and the `Simulation` built by $C$ have equal `configuration_hash`,
  with the noise seed fixed in both. The hash already covers everything that decides the
  computed fields (grid, modes, configuration, noise, stepper, materials and launch), so
  this is the right identity and needs no second definition. Equivalence runs one way, from
  table to object: there is no step from an object back to a table (§5.1).
- **Quantities.** Recipe values are numbers in the keyword's documented SI unit, or quantity
  strings (`"125 um"`, `"1.2 bar"`, `"7.5 fs"`, `"2e14 W/cm^2"`) converted to that unit. The
  unit is the one the docstring states today, so pressure stays in bar (conventions.md) and
  a bare `1.2` for `pressure` means 1.2 bar. A string is parsed as a decimal mantissa and a
  unit; the unit's dimension must match the keyword's, so `"125 um"` for `pressure` is
  refused. The conversion is exact in decimal (a power-of-ten prefix shifts the exponent,
  any other factor multiplies exactly, in a local context with enough digits) and is rounded
  once to the nearest binary64 value, so `"800 nm"` gives the literal `800e-9` that a script
  writes. Writing a value back chooses a prefix in which the shortest decimal representation
  of the double is exact, so the round trip is the identity. Keywords whose elements carry
  different units (`phases` is $[\mathrm{rad}, \mathrm{s}, \mathrm{s}^2, \dots]$, `betas` is
  $\mathrm{s}^n/\mathrm{m}$ by order, the `(z, p)` form of `pressure` pairs metres with bar)
  have a unit per element or per form (S9). Temperature accepts K only.
- **Beam tilt.** A tilt $(\theta_x, \theta_y)$ multiplies the launched field by
  $\exp[\mathrm{i}\,k(\omega)(x\sin\theta_x + y\sin\theta_y)]$, $k(\omega)$ the medium's
  real wave number at the launch plane, as `coupling.Alignment` defines it. With $k(\omega)$
  rather than $k(\omega_0)$ every frequency leaves at the same angle: a pure tilt with no
  angular dispersion. Tilt and displacement apply at the launch plane, after any `propz`.
  The screen depends on frequency, so the field is no longer a product of a transverse and a
  temporal part, and the energy is set after it. The transverse wavenumbers
  $k(\omega)\sin\theta$ must lie inside the grid's transverse band at every active
  frequency, the tilted and displaced beam must be negligible at the window's edges, and the
  pulse-front delay of about $x\sin\theta\,n/c$ across the beam must fit in the time window;
  otherwise the launch is refused.
- **Launch normalisation.** On a basis whose modes are flux-orthogonal (a diagonal flux
  metric, as for the Marcatili modes), the energy of a launch is the sum of the modes'
  energies, and fractions of the pulse energy are physical. On a general modal basis the
  flux metric $M$ has cross terms, and the energy of coefficients $a$ is $a^\dagger M a$,
  which depends on the relative phases. There a launch gives relative amplitudes and phases,
  and the coefficients are scaled so that $a^\dagger M a$ is the pulse energy; the per-mode
  shares are coordinate quantities, not an energy split (S5).

### 5. Proposed interface

#### 5.1 S1 — `solprop.recipe`

A recipe is a TOML document with five fixed tables. No other top-level name is allowed, so a
future keyword can never collide with the format.

| Table | Holds |
|---|---|
| `[recipe]` | `format` and `route`, required; `title` and `solprop` (the version that wrote it), informative |
| `[build]` | the keywords of the route's `Simulation.from_*` constructor, under their own names, and nothing else |
| `[execution]` | `fields`, `field_planes`, `resume_history` and `memory_budget`, which are not build keywords. The first three are applied to the built simulation's settings; `memory_budget` goes to `run` |
| `[pins]` | values set from a design target |
| `[scan]` | a scan over paths |

```text
[recipe]
format = 1
route = "capillary"
title = "RDW in a helium capillary"
solprop = "0.2.0"

[build]
radius = "125 um"
length = "1 m"
mode_average = true
gas = "He"
pressure = ["0 bar", "1.2 bar"]   # a gradient, as the keyword accepts
temperature = "293.15 K"
plasma = true
wavelength = "800 nm"
fwhm = "7.5 fs"
energy = "275 uJ"
wavelength_limits = ["100 nm", "1500 nm"]
trange = "200 fs"

[execution]
fields = "planes"
field_planes = [0, -1]
```

A build keyword is a value of `[build]` whether or not it is a table: `[build.plasma]` is
the `plasma` keyword's mapping, and `[[build.pulses]]` its list of pulses. Grouping for
reading is a front end's business, from the metadata's `group` (S9).

- **Refused keys.** `output`, `progress`, `progress_period`, `resume`, `stop`, `snapshots`
  and `tdse_archive` choose files, terminals and resume behaviour, which the caller or the
  command line supplies. A recipe holding one is refused, naming it. `prop_radial`,
  `prop_free` and `prop_planar` unpack `absorber` and `absorber_scheme` onto their settings;
  a recipe uses the `from_*` form, so those go under the `settings` object.
  `from_waveguide`'s own `memory_budget` admits a guide's tables at build and is a build
  keyword; `[execution] memory_budget` admits the run. They share a unit and nothing else.
- **Only what was set is written.** A keyword left out takes solprop's default at build
  time, so a recipe follows solprop's improvements. What ran is recorded in the output
  beside it (S2).
- **`None`.** TOML has no null, and leaving a key out already says `None` wherever that is
  the default. Only the keywords whose default is not `None` need a spelling, `noise` and
  `boundary_check` on the keyword routes, and those write the string `"none"`. The metadata
  lists them, and a test checks that each accepts no `"none"` of its own. `[execution]
  memory_budget = "none"` keeps its existing meaning, no check.
- **Numbers.** A number is a TOML number in the keyword's SI unit or a quantity string
  (section 4). A complex number is `{re = 1.45, im = 0.0}`. An integer too large for TOML's
  64 bits, such as a drawn noise seed, is a decimal string; any integer keyword accepts one.
- **Arrays.** A one-dimensional numeric array (`betas`, `phases`, `save_planes`,
  `StatsSpec.energy_windows`) is a list of up to 10 000 numbers, with the metadata's unit
  per element. A larger one, or a larger-dimensional one, is a data file (below).
- **Objects.** A value that is an object (a pulse, a gas mixture, a coupling, a profile, a
  guide or mode provider, and on the object routes the grids, spaces, media, beams,
  responses and settings) is an inline table or array of tables with a `type` key naming an
  entry in a fixed registry in `solprop.recipe`, and that entry's keywords: `pulses = [{type
  = "GaussianPulse", wavelength = "800 nm", fwhm = "30 fs", energy = "1 mJ"}]`. An entry is
  a **named factory**, a constructor, a class method or a module function, with its keywords
  declared and any **build context** it receives declared too. The context is what format 1
  defines a factory may be given, the route's `grid`. So `absorber = {type = "absorber"}`
  under `settings` builds `boundaries.absorber(grid)`, and `{type =
  "PressureGradient.two_point", ...}` and `{type = "EnvelopeGrid.from_limits", ...}` are
  entries as well. Only registered types load: a recipe never imports a module or evaluates
  code. The registry starts with the documented user-facing types (phase 4 lists them),
  because every keyword of an entry becomes public: adding an entry later is additive, and
  removing one is not. A type outside it is the Python export's job.
- **A recipe is one-way.** A recipe is built into objects, and objects are never turned back
  into tables, because they do not keep what made them: a `PressureGradient.two_point` holds
  positions and pressures, and an `EnvelopeGrid` its arrays and not its limits. So a saved
  recipe always comes from a recipe. `Recipe.from_keywords(route, **keywords)` builds one
  from plain values and refuses an object or a callable by keyword, pointing to the table
  form or the Python export.
- **What a recipe cannot hold.** Callables (`beta`, `loss`, a mode's loss function), and
  types not in the registry, such as `IndexTable` and the `beta` tuple form in format 1. The
  Python export is the way out.
- **Data files.** A data file is referred to by a path relative to the recipe, with a
  SHA-256 recorded beside it and checked on load: `{file = "table.npy", sha256 = "…"}` for
  an array, read with pickling disabled, and `{type = "load_spectrum", file =
  "spectrum.csv", sha256 = "…", energy = "1 uJ"}` for a measured pulse (S12). The path,
  after symbolic links are resolved, must lie inside the recipe's directory unless the
  caller allows otherwise.
- **Parameter paths.** Every value has a canonical path: the keyword, then `.name` for a
  keyword of an object, `[i]` for an element of an array and `["key"]` for a key of a
  mapping, as in `pressure`, `pulses[1].energy` or `launch["HE1,1:even"].share`. In `[pins]`
  and `[scan]` a path is a quoted TOML key where it needs to be. Paths are what `Recipe.set`
  and `Recipe.get` take, what errors and warnings name (S10), what a scan varies and what a
  pin sets. The metadata of a path is that of the keyword it ends on (S9).
- **Pins.** `[pins]` sets a whole value from a design target instead of a literal: `pressure
  = {target = "zdw", wavelength = "800 nm", search = ["0.1 bar", "10 bar"], root =
  "lowest"}`. A pinned path may not also be set literally, and an element of an array cannot
  be pinned.

  | Target | Keys, beside `search` and `root` | Pins | Solved |
  |---|---|---|---|
  | `zdw` | `wavelength` | a parameter of the guide or fill | on the launch mode at the launch position |
  | `rdw` | `rdw`, and optionally `peak_power` with `fwhm`, `shape` and `nonlinear_phase` | the same | the same; the pump is the recipe's `wavelength` |
  | `soliton_order` | `order` | a pulse's `energy` or `peak_power` | on the launch mode at the launch position, with the fill's Kerr coefficient |

  Pins resolve in two fixed stages, after the literals: at most one pin on a parameter of
  the guide or fill, then the pins on pulses, which read it. So no dependency order is
  needed and no cycle can be written. A pressure gradient cannot be pinned, because the
  target needs one mode. An unreachable target is an issue on the pinned path with the
  attainable range. `root` chooses among several roots (`"lowest"`, `"highest"` or an
  index). `Recipe.pin_roots(path)` returns the solver's `DesignRoots`, and
  `Recipe.resolve()` returns the recipe with the pins replaced by their values. A design
  that sets two values at once, such as `capillary_design`, is not a pin: a front end calls
  it and writes its values as literals. The Python export writes the solver call before the
  route's call.
- **Scans.** `[scan]` requires `name`, which sets durable filenames and the derived seeds,
  and may give `base_seed`; the other keys are paths and their values (a list, or `{start,
  stop, num, spacing = "linear" | "log"}`). `Recipe.scan()` returns a `RecipeScan` with the
  `Scan`, a picklable builder that passes `point.seed` and re-solves the pins at each point,
  and the keywords for `run_scan`, which takes `fields` and `field_planes` itself and
  refuses them on a point's settings.
- **Examples.** The recipes of the guide's tutorials and gallery examples ship in the
  package: `Recipe.examples()` lists them and `Recipe.example(name)` loads one. A front end
  starts a new setup from one of them, so a starting point such as Kerr and plasma on in
  free space (where the route's own default is linear) is solprop's named choice, not the
  front end's. Each is tested as the gallery example it mirrors.
- **Format versioning.** `format` is an integer. The loader upgrades older formats and
  refuses newer ones; unknown keys are refused, naming the key and the nearest valid one.
  Format 1 becomes public when this SEP is merged.

The Python surface:

```text
from solprop.recipe import Recipe

recipe = Recipe.load("rdw.toml")          # or Recipe.loads(text), Recipe.from_dict(...)
recipe.set("pressure", "1.5 bar")         # validated against the metadata (S9)
recipe.set("pulses[1].energy", "10 uJ")
problems = recipe.check()                 # list of RecipeIssue (S10), empty if fine
simulation = recipe.build()               # a Simulation, not run, carrying its recipe
output = recipe.run("rdw.h5")             # build, then run with [execution] applied
recipe.to_python()                        # a plain script with the route's call
recipe.save("rdw-1.5bar.toml")
Recipe.from_keywords("capillary", radius=125e-6, ...)
```

`recipe.run` is `build` followed by `run`, with `memory_budget` passed on; `simulation.run`
given a simulation from `build` records the recipe too, and takes the budget as usual. Two
recipes are equal when their values are equal. A recipe keeps the text it was loaded from:
`save` writes that text verbatim while the recipe is unchanged, comments included, and
writes canonical TOML once it has been edited, which keeps no comments and spells each
quantity in the keyword's display unit (S9), in ASCII (`um`; `µm` is read too).

And the command line (a `solprop` console script and `python -m solprop`):

```text
solprop run rdw.toml -o rdw.h5 [--progress log] [--resume]
solprop check rdw.toml                    # validate and build; print issues; exit status
solprop export rdw.toml --python          # print the equivalent script
solprop resume rdw.h5 [--interpolated | --force] [--memory-budget ...]     # S2
solprop scan …                            # the existing python -m solprop.scan, unchanged
```

`to_python()` writes the route's `prop_*` call when every key is on its surface, and
otherwise `Simulation.from_*`, the settings and `run`; keyword arguments in SI, one comment
per keyword stating the recipe's quantity string, and nothing that a reader would not write
by hand. Execution settings are part of what it writes, and the tests compare them, because
the hash does not cover them.

#### 5.2 S2 — the recipe in the output, and resume from a file

A `Simulation` from `Recipe.build` carries its recipe. `with_input`, `with_mixture` and a
change of a setting outside `[execution]` drop it and warn, so a file never holds a recipe
that does not describe its run. Nothing is compared at build, so building costs no
translation. At the first run start the driver writes a top-level group `recipe` in the
output, once, holding:

- `authored`: the recipe's text as loaded, or its canonical TOML if it was edited;
- `resolved`: the recipe to rebuild from, also TOML:
  - the authored keys, with each pin replaced by its solved value and the noise seed that
    was drawn;
  - each keyword the metadata marks **resolvable** (S9), at the value used;
  - `[execution]`, from the settings at run start.
- the recipe format and the SHA-256 of each text;
- the data files the recipe refers to, under generated names, up to 16 MiB in all; a larger
  one is recorded by its path and hash only.

`resolved` is defined by what rebuilds the run, not by "every keyword". A keyword is
resolvable when its value, written back, gives the same record, and a sync test holds each
one to that. `rtol`, `atol`, `dz_max`, `integrator`, `stats`, `temperature`, `plasma`,
`thg`, `raman`, `step_planes`, `save_planes`, `loss` and `model` were measured to qualify on
the keyword routes. These do not, and stay left to the build:

- `dz_init` (an explicit first step turns the start-up rule off, which moves
  `stepper.limits.auto_init`);
- `raman_fraction` (the record says where the fraction came from);
- the default `absorber` and `cladding` (no table form);
- `backend`, which stays as requested, so a run can move between machines;
- the alternatives `grid` and `wavelength_limits`, and `fwhm` and `width`, of which only the
  authored one is written. `save_planes` replaces `n_save`, whose planes it holds.

So a later solprop's default cannot move a resolvable keyword, and for the rest the resume
gate is what notices. The group is provenance outside `configuration_hash`, so it needs no
schema change (SEP 42 §5.8): a reader that does not know it ignores it, and a file without
it is a file from a script. It is never rewritten. A later session that differs, such as
`solprop run edited.toml --resume` with a changed `rtol`, is recorded in that session's
execution record (SEP 42 §5.3), and `solprop resume` afterwards returns to the first
session's numerics, with the warning SEP 42 gives for a numerics change.

`Recipe.from_output(path, resolved=False)` reads the authored recipe back for editing, or
the resolved one to show the setup that ran. Saving such a recipe writes its embedded data
files beside it under generated names; a file that was only referred to stays a dangling
reference, and `check` says so. `solprop resume run.h5` (or `resume_file(path, mode=True)`,
with `mode` as `resume` takes it: `True`, `"interpolated"` or `"force"`) rebuilds the
simulation from the resolved recipe and continues the file through that same `resume=`, so
SEP 42's gate decides, by its comparison and with its record. A file written without a
recipe resumes as it does today, from the script. A scan's file stores the authored recipe
too, in the same group, so a scan reopens with its setup.

**Received files.** Resuming a file runs the setup it holds, and the guide says so. The
limits:

- the recipe passes the same checks as a loaded one (registered types only, no keys that
  choose paths);
- embedded data files are read into memory and written only under generated names;
- `resume_file` ignores the recipe's own `memory_budget` and `node_budget`, and applies the
  caller's, refusing rather than warning by default, because a received file must not choose
  its own limits;
- parsing an HDF5 file is itself an exposure of the HDF5 library, which the guide states.

#### 5.3 S3 — a plane observer

A new optional protocol beside `InterruptibleReporter` and `SummarisingReporter`:

```text
class PlaneObserver(ProgressReporter, Protocol):
    def observe_plane(self, z: float, plane: PlaneView) -> None: ...
```

The specific name avoids reporters that already have an `observe` attribute, and
`NullReporter` does **not** implement it, so a quiet run builds no view. The driver calls
`observe_plane` once per save plane, after the plane is committed to the output and before
`plane()`. That includes the entrance, which has no `plane()` call of its own. A resume does
not deliver earlier planes again; a front end reads those from the file.

`PlaneView` is lazy. It holds the plane's position, the grid, the statistics row and a
description of the basis (S16: whether its quantities are physical flux or coordinate
quantities, and in which units). Its methods compute on request: the frequency- and
time-domain field, by mode where modes exist, and the spectral density. On a host backend
nothing is copied until asked. On a device backend, a plane whose field the output keeps is
already on the host; on a field-free plane (`fields="none"` or `"planes"`, the scan
default), asking for the field fetches one plane from the device, and the driver counts that
transfer in the run's resource summary. The arrays are valid only for the call; an observer
copies what it keeps. The existing rule holds: nothing an observer does may change the
propagation.

**Failure and workload.** An exception from `observe_plane`, other than `RunTerminated` and
`KeyboardInterrupt`, does not fail the run: the driver logs it once with its traceback,
warns, and calls the observer no more in that run; `plane`, `finished` and `interrupted` are
still delivered. The observer runs in the march's thread and solprop bounds nothing about
its time; the driver reports the time spent in observers in the resource summary. The guide
gives the pattern a front end uses: reduce the plane to what it draws, put it in a bounded
queue that replaces an undelivered preview, and return, keeping terminal and error events on
a reliable channel. Transport and disconnection are the front end's; solprop guarantees only
the above.

The boundary monitor needs no hook. It warns once with a `BoundaryWarning` or stops the run
with a `BoundaryError`, and a front end has both, with `parameter` set (S10).

#### 5.4 S4 — scan progress

`run_scan(..., progress=)` accepts a `ScanReporter` as well as its literal choices:
`scan_started(n_points)`, `point_started(index, point)`, `point_finished(index, status,
wall)`, `scan_finished()`, and optionally `point_reporter(index) -> ProgressReporter | None`
for per-point progress.

- The local executor calls `point_reporter` in this process.
- A pool reports `point_started` when it dispatches a point and `point_finished` when it
  collects it. Per-point progress needs a new message on the worker pipe, carrying `z` and
  the `PlaneSummary` of each plane (never a field), at the progress period.
- Batch, queue and Slurm executors report point boundaries only, from the collected files.

#### 5.5 S5 — launch into chosen modes

`prop_capillary` and `prop_waveguide` gain `launch`, a mapping from mode label to a share
and, optionally, a phase in radians: `launch = {"HE1,1": 0.9, "HE1,2": {"share": 0.1,
"phase": 1.57}}`.

- **Labels** resolve as `channel` resolves them, by display label or index; a label that
  leaves a parity open when both polarisations are present is refused, naming the
  candidates. A label not in the mode set is refused, naming the available ones.
- **Shares.** On a flux-orthogonal basis (section 4) the shares are energy fractions,
  normalised to one. On a general modal basis they are relative amplitudes squared, scaled
  with the phases so that the flux-metric energy is the pulse's; the guide and the run's
  metadata say that the split is then a coordinate one.
- **Per pulse.** `launch` is either one mapping, for every pulse, or a list of mappings, one
  per pulse in `pulses`, so a pump and a seed can go into different modes.
- **With the other launch keywords.** `launch` generalises `channel`: giving both is
  refused. `incident` with `coupling` computes the launch from a beam and is exclusive with
  `launch`. `Simulation.with_input` remains the way to launch an array.
- The default, `None`, launches as today.

The launch is recorded in a new `configuration` entry, `modal_launch`, holding each pulse's
labels, shares and phases. `configuration["launch"]` already holds an incident beam's
report, and SEP 42 ignores `configuration.initial` where a pulse is recorded, so without
this entry two recipes with the same pulse and different launches would continue each
other's files. The entry is compared under SEP 42, and is absent from older records through
a neutral entry.

#### 5.6 S6 — beam displacement and tilt

`SpatialPulse` gains `displacement=(d_x, d_y)` in m and `tilt=(theta_x, theta_y)` in rad,
both defaulting to zero, with the names and definitions of `coupling.Alignment` (section 4).
They are recorded as those two fields of `configuration.pulse`, compared under SEP 42, with
a neutral entry of zero for older records. They are applied when the beam is built on a
Cartesian grid. A nonzero value is refused on a radial grid, which it breaks the symmetry
of, and so on `prop_waveguide(incident=...)`, which samples the beam radially, and a nonzero
`y` component on a line grid, which has no `y`. `coupling.Alignment` stays the way to
misalign a beam *into a guide*. The tilt's validity is the band, edge and delay conditions
of section 4, checked at build; `Alignment`'s `model_error` budget, which certifies a
paraxial screen against a guide's modes, does not apply to a free-space launch.

#### 5.7 S7 — design solvers for every guided route

A general solver and thin named wrappers:

```text
tools.solve_parameter(build, search, target, *, z=0.0) -> DesignRoots
    # build(value) -> mode, or (mode, gamma) where the target needs the fill's Kerr
    # target: ZDW(wavelength)
    #       | RDW(pump, rdw, peak_power=None, fwhm=None, shape="gauss", nonlinear_phase=...)
    #       | SolitonOrder(order, pulse)
tools.radius_zdw(gas, pressure, zdw_wavelength, ...)          # new
```

`build(value)` returns a mode for a trial value of the parameter (radius, pressure, core
size, wall thickness), so one solver serves the capillary, antiresonant, rectangular and
slab modes. A mode need not know its fill's Kerr response, so a nonlinear target needs
`build` to return it as well. The solver samples the target over the search range
(logarithmic and linear, as `pressure_rdw` does), discards sign changes through a pole,
refines each bracketed root, and returns **every root** found, with the attainable range
when there is none. For `RDW` and `SolitonOrder` this reuses what `pressure_rdw` does. For
`ZDW` a sampled search over the parameter is new, since `pressure_zdw` brackets once, and it
costs up to 258 mode builds on a non-Marcatili guide. `RDW` reproduces `pressure_rdw` beyond
linear phase matching, which needs `fwhm` and `nonlinear_phase` beside `peak_power`. The
existing capillary functions keep their signatures and gain no `mode` keyword:
`solve_parameter` is the general form.

#### 5.8 S8 — a capability query

```text
solprop.capabilities.route("capillary") -> RouteCapabilities
solprop.capabilities.species("N2") -> SpeciesCapabilities
```

`RouteCapabilities` lists the route's keywords (from S9), the allowed values of each
enumerated keyword, and the **combination rules** of the interface layer (route × keyword ×
species) with their reasons: for example, `tdse` requires the capillary route and a helium
fill; `kerr_form` requires `provider`. Each such rule has an id, and the builder that
enforces it raises with that id and `parameter` set; a lint-style test checks that every
combination refusal, marked as one at the raise, names a listed rule. Refusals that depend
on values (grid coverage, node budgets, material validity) are not rules; `check()` finds
them (S10). `SpeciesCapabilities` lists the ionisation models with their status (curated,
provisional, unavailable), whether a Raman response exists and its status, and the validity
ranges optmat records.

#### 5.9 S9 — parameter metadata

A table beside each interface function and each registered recipe type, one entry per
keyword:

- SI unit (or none), and a unit per element or per form where elements differ (section 4);
- a list of **forms**, because many keywords take several (`pressure` a number, a gradient,
  a table or a profile; `plasma` a boolean, a rate or a per-species mapping), each with its
  kind (number, integer, boolean, choice, quantity array, mapping, object), its units and
  its allowed range or choices;
- whether it needs the `"none"` spelling (§5.1), its recommended display unit, in ASCII
  (Q7), and whether it is **resolvable** (§5.2);
- default as documented (the function's default stays the single source; the table refers to
  it), and, where the default is `None` resolved by a rule (`rtol` by route, `plasma` and
  `thg` by `envelope`, `step_planes` by TDSE, `raman` by species, `pressure` at one bar), a
  resolver that returns the value that will be used with the rule's text;
- group (`geometry`, `medium`, `nonlinear`, `input`, `grid`, `numerics`, `output`), level
  (`basic` or `advanced`), a one-line summary, and the anchor of its manual section.

`solprop.recipe.parameters("capillary")` returns it. A test checks that the table and the
signature name the same keywords and that each default matches, for the `from_*` surfaces,
the `prop_*` surfaces and the execution settings. A second test writes each resolvable
keyword back and checks that the hash holds. A front end's tooltips and field validation
come from this table; its own labels and wording are its business.

#### 5.10 S10 — structured validation

`SolpropError` gains an optional `parameter` attribute (a path, or a tuple of paths for a
conflict) and an optional `rule` (S8), set wherever the interface layer refuses a value. The
warnings that concern a value (`MaterialRangeWarning`, `BoundaryWarning`, the
provisional-Raman warning) carry `parameter` too. No exception changes class.

`Recipe.check()` collects every issue it can at the metadata level (type, unit, dimension,
range, unknown key, refused key), solves the pins, then builds the simulation and translates
it, stopping before the backend is constructed, and reports the first refusal of the builder
or translation with its `parameter`. Translation finds the value-dependent refusals; it
costs about a tenth of a second on the capillary example, and a pin some tenths more on a
Marcatili capillary, so a front end calls it on a pause in typing, not on every keystroke.
`solprop.grid.suggest(pulses, route=...)` proposes `wavelength_limits` and `trange` from the
input pulses and the expected broadening, labelled as a starting point.

#### 5.11 S11 — a cost estimate

`Simulation.estimate()` returns:

- the grid sizes;
- the persistent memory, from `solprop.driver.memory`, admitted against the budget
  **before** anything is allocated (a run that would not be admitted is reported, not
  attempted);
- the output file size for the requested planes;
- the cost of one right-hand-side evaluation on the chosen backend, timed after one untimed
  warm-up call (so compilation and transform planning are excluded and reported separately),
  labelled with the backend and thread count;
- the integrator's calls per step, and, for fixed schedules, the total.

It does not predict the number of adaptive steps, which depends on the physics.

#### 5.12 S12 — measured-pulse loaders

`solprop.pulses.load_spectrum(path, *, energy, axis="frequency", axis_unit=None,
columns=None, phase_unit="rad", phase_convention="optics", wrapped=None, floor=1e-4)` reads
a whitespace- or comma-delimited text file and returns a `DataPulse`. The default layout is
the usual output of a measurement and retrieval, the same three columns Luna.jl's
`DataPulse` documents:

1. frequency in Hz;
2. spectral power density, in arbitrary units, because `energy` sets the scale;
3. spectral phase in radians; a file with only two columns has a flat phase.

`axis="wavelength"` applies the Jacobian between wavelength and frequency densities, and
`axis_unit` defaults to the axis's own SI unit. `columns` picks other columns. A descending
axis is sorted. `phase_convention="fft"` reads the opposite sign: solprop's interface uses
the optics convention (conventions.md §2), so a delay $\tau$ is $+\omega\tau$ here, while a
file written for Luna.jl holds the FFT convention, in which the same delay is $-\omega\tau$.

The phase needs care when it is regridded:

- **It is interpolated as given.** A cubic spline runs through the file's unwrapped phase,
  as `DataPulse.from_spectral_density` does, not through a phase re-derived from the complex
  spectrum. Wrapping an unwrapped phase and unwrapping it again cannot recover a change of
  more than $\pi$ between samples.
- **A wrapped column is recognised.** With `wrapped=None`, a column whose values all lie
  within a range of $2\pi$ is taken as wrapped and unwrapped, and any other is left alone;
  `wrapped` says which where that guess would be wrong. A phase in degrees needs
  `phase_unit="deg"`, or it spans 360 and looks unwrapped.
- **The noise floor does not steer the unwrapping.** A wrapped phase is unwrapped only over
  samples above `floor` times the peak power, and held at the nearest such sample beyond
  them, where the measured phase means nothing. An unwrapped column is taken as it stands.
- **Undersampling is said.** A wrapped column whose steps at significant power cluster near
  $\pm\pi$ cannot be recovered, and is warned on.

Tool-specific export formats are left until someone needs one; the `columns` keyword and
`DataPulse.from_time` already cover most of them. There is no `load_retrieval`.

#### 5.13 S13 — geometry drawings in the library

`solprop.plotting.geometry` draws the cross-section of each guide and free-space grid
(`draw(ax, guide)`, `thumbnail(route, representation, size)`), with an explicit palette
argument instead of CSS custom properties, so the drawings render in Matplotlib, in SVG
viewers and in Qt. The user guide's figure script calls it, so the guide and the library
draw the same geometry.

#### 5.14 S14 — route keywords

The keyword surfaces of the keyword routes differ in places. This SEP **adds** the missing
keywords where the feature applies and documents the deliberate differences; it renames
nothing and changes no default.

| Keyword | On | Proposal |
|---|---|---|
| `interpolate_planes`, `step_planes` | gnlse, capillary | add to waveguide |
| `polarisation_taper` | gnlse | add to capillary and waveguide where a spectral collar exists |
| `noise`, `seed` | all but planar | a documented difference: a planar grid carries no noise draw, and adding one means declaring a photon measure, which is physics |
| `memory_budget` | `from_waveguide` and `Simulation.run` | different gates with one unit, documented: the guide's tables at build, the run at `run` (§5.1) |
| `step_planes` default `0` (gnlse) vs `None` (capillary) | | same behaviour on gnlse (no TDSE); write `None` on both for one signature |
| integrator and `rtol` defaults by route | | deliberate (per-route qualification); exposed through S9's resolvers and documented in one table |
| `atol` defaults (1e-14 gnlse, 1e-10 others) | | deliberate, kept (Q3): `atol` acts only when the field norm falls to about `atol/rtol` (1e-7 √W and 1e-4 √W), far below any real pulse; measured norms are 759 √W for 1 pJ in a capillary and 1.9e3 √W for a 10 W GNLSE pulse, and a capillary run takes the same steps at either value |
| `loss` (capillary) vs `loss_db_per_m` (gnlse) | | different meanings (a switch or model vs a number); keep both, documented |
| `raman` default `False` (gnlse) vs `None` (gas routes) | | deliberate, kept (Q3): the GNLSE has no medium to read a response from, so `None` there would switch fused silica's Raman on for every fibre |

These new keywords map onto settings the run already records.

#### 5.15 S15 — the writer-lock sidecar

Each HDF5 output keeps a `*.writer.lock` file beside it. It persists by design: removing a
lock file after releasing an advisory lock races with a second writer opening it. The
sidecar is documented under snapshots; this SEP moves that text to the output guide's
description of the file, and leaves the behaviour unchanged. Front ends hide it.

#### 5.16 S16 — result capabilities

```text
solprop.capabilities.output(path_or_output) -> OutputCapabilities
```

`OutputCapabilities` says what a results file can give, extending the output capability
machinery that exists (`CheckpointOutput` and the readers' refusals):

- the representation and basis, and for each derived quantity (power, energy, spectral
  density, field) whether it is a **physical flux** or a **coordinate quantity**, with its
  unit;
- which planes keep fields, and which only statistics rows;
- which operations are available and which refused, with the reason: compression on spatial
  and general modal outputs, general modal reconstruction without the guide and reference
  tables (with the identities the file holds), physical powers from a spatial file;
- whether a recipe is recorded (S2) and the file's resumability (S17).

`PlaneView` (S3) carries the same description for the plane it shows; it does not make the
same physical operations available on every basis.

#### 5.17 S17 — a portable stop, and resumability

`Simulation.run` and `run_scan` accept `stop=`: any object with an `is_set()` method
(`threading.Event` qualifies). The flag belongs to that run. The driver polls it before the
run's first step and at every trial step, so it answers within one step or trial, or after
the translation or atomic preparation in progress. When it is set, the run writes its state
as for SIGTERM, tells the reporter through `interrupted`, and **returns the handler**: a
requested stop is not an error. `outcome.message` says that the caller's flag stopped it,
and `resume=True` continues. A front end sets it from any thread on any platform; signals
are not needed (on Windows, `os.kill` with `SIGINT` terminates a process outright). `solprop
run` maps SIGINT, SIGTERM and, on Windows, `SIGBREAK` (sent by `CTRL_BREAK_EVENT`) onto it,
and exits with the usual status.

Across a scan, a stopped point is not committed, and a rerun computes it again:

| Executor | `stop=` |
|---|---|
| local | polled by the point in flight |
| pool | a message on the worker pipe, which that worker's run polls; the pool's abrupt `terminate` remains the last resort |
| batch | between groups |
| queue, Slurm | refused, naming the queue's own tools; points run in other processes or on other machines |

```text
solprop.output.resumability(path, gate=False) -> Resumability
```

`Resumability` answers three questions separately:

- **Can the file continue, and from where?** The `z`, and the `resume=` value it needs:
  `True`, `"interpolated"` or `"force"` (a file with a forced session). Otherwise the
  reason: no work left, no checkpoint written before a failure, another output schema, or a
  writer still alive.
- **Can it be rebuilt from itself?** Whether a recipe is stored, and readable: a newer
  format is named. No recipe does not mean it cannot continue, only that its script must do
  it.
- **Does the gate pass?** Only with `gate=True`, at the cost of a rebuild: SEP 42's
  `RecordComparison` for the stored recipe, so a moved optmat digest shows before a resume
  is tried.

A front end's own run states (queued, stopping, stopped) stay its own; `RunOutcome.status`
is unchanged. A changed grid, a wider time window included, is a new run, not a resume (Q4).

#### 5.18 S18 — previews

Before a run, a front end shows what a recipe will build without building the run:

- `Recipe.modes(z=None)` returns the mode set a guided route would build (labels, `beta`,
  `beta2`, `zdw`, effective areas and loss over the grid's band), at the launch position
  unless `z` is given. Its cost is the mode solve: on a guide route that means the dense
  tables, with the memory admission `from_waveguide` makes.
- `Recipe.preview_input()` returns the input pulse, and the beam on free-space routes, on
  the run's grid, as the launch will set them (S5, S6), with its energy and peak power, and
  without noise, so a preview draws nothing.

### 6. Implementation strategies

- **Layer.** All of this is interface and driver work above translation (architecture.md
  section 3). No backend, kernel or stepper changes; `KERNEL_SCHEMA_VERSION` is unchanged,
  and so is the output schema (section 9).
- **Recipe keys = keywords (recommended) or a designed schema.** A designed schema with its
  own names (`[pulse] shape = "gaussian"`) reads well, but is a second vocabulary to keep in
  step with six functions and their docstrings. Using the build keywords themselves keeps
  one vocabulary, makes `to_python()` direct, and lets the sync test (S9) catch every drift.
  A front end may present any names it likes.
- **One `[build]` table (recommended) or group tables.** Tables such as `[geometry]` and
  `[grid]` read well by hand, but `grid` and `medium` are keywords, `output` is a refused
  key, and a mapping such as `[plasma]` looks like a group. One table has none of these
  problems, and it needs no list of reserved words as keywords are added.
- **Metadata: a table (recommended) or `Annotated` signatures.** `Annotated[float,
  Unit("m"), Range(0, None)]` puts metadata in the signature, but every interface function
  is already long, the type checker and the runtime array-shape checks would see the
  annotations, and the documentation renderer shows them. A table beside each function,
  checked against the signature by a test, costs less and changes no signature.
- **TOML reading and writing.** `tomllib` (standard library) reads, and `tomli-w` writes
  (Q6). It is pure Python with no dependencies of its own, MIT-licensed, and its wheel is
  about 7 kB. It joins the core table of `docs/dev/dependencies.md`. It writes no comments,
  so an edited recipe loses them (§5.1). Tried on this format, it round-trips nested tables,
  arrays of tables and quoted path keys through `tomllib`.
- **Unit parsing.** A small parser in `solprop.recipe` with a fixed table of SI prefixes,
  the units the keywords use and their dimensions, converting with `decimal`; no dependency
  on a units library.
- **Plane observer.** The view wraps what the driver already holds when it calls `plane()`;
  it computes or fetches only on request.
- **Stop.** The flag is polled per trial step, a cost of one attribute call, and it raises
  the same internal path as SIGTERM, which a prototype showed working from another thread
  with the run on a non-main thread: it stopped in 34 ms and the file resumed to completion.
  The run catches it when its own flag set it. It is not held in the process-global signal
  guard.
- **Cost.** Loading and building a recipe add milliseconds. The plane observer costs only
  what the observer asks for. `check()` costs a translation and the pins. The estimate costs
  a warm-up and a few timed calls, only on request.

### 7. Verification and validation

The tolerance classes are those of the [testing policy](../testing.md#3-tolerance-policy).

- **Recipe equivalence** (class P, configuration identity). For the examples of the user
  guide and gallery, a hand-written recipe builds a simulation with the same
  `configuration_hash` as the example's call, with the seed fixed. The manual's list of
  examples that have no recipe (those using `with_input`, callables or an unregistered type)
  is checked too, so it stays true. The keyword routes in phase 1, the object routes and the
  waveguide in phase 4.
- **Python export** (class P). `to_python()` of each of those recipes, executed in a
  subprocess with the run replaced by a build, gives the same hash and the same execution
  settings, which the hash does not cover.
- **Round trip** (class P). Table → text → table returns an equal recipe; each registered
  type goes table → object → hash; a recipe loaded and unchanged saves verbatim; formats are
  upgraded by tests holding a recipe of each older format.
- **Resolved recipes** (class P). Each keyword the metadata marks resolvable, written back,
  keeps the hash. A rebuild from `resolved` has the file's hash, seed included. The §5.2
  list of keywords that are not resolvable is tested as such, so a keyword that becomes
  resolvable is noticed.
- **Quantities** (class P). Each unit and prefix converts to the literal a script writes,
  including the section 5.1 example's values; writing back is the identity; a wrong
  dimension and an ambiguous string are refused. A complex number, a 128-bit seed as a
  decimal string, and a list at and over the array limit.
- **Provenance.** A file written from a recipe holds both texts; a simulation changed after
  its build by a mutator or a setting outside `[execution]` writes no recipe and warns; a
  change of `fields` is written as it was at run start; an unseeded resume through
  `_adopt_recorded_seed` neither warns nor rewrites the group; a file of the current schema
  with and without the group is read by the readers; a scan's file holds its recipe.
- **Resume from a file** (class T). A short capillary run stopped by the `stop` flag after a
  few planes, and another stopped by SIGTERM, resumed with `solprop resume`, agree with an
  uninterrupted run within an accuracy envelope against a tighter reference, set per case in
  a comment. `resume_file` takes each mode. `resumability` is checked on a complete file, a
  stopped file, an interpolated last plane, a failure before the first checkpoint, a forced
  file, a live writer and a newer recipe format, and `gate=True` shows a moved digest.
- **Stop.** The flag stops before the first step, between trials and on a non-main thread;
  two runs in one process stop separately; the run returns and resumes. Each executor as
  §5.17 says.
- **Sync.** Metadata tables match the signatures (names, defaults) on every surface; every
  marked combination refusal names a listed rule; the keywords with the `"none"` spelling
  are those whose default is not `None`, and each accepts no `"none"` of its own.
- **Observer.** The observer's copy of each plane equals the plane read back from the file
  (class P). A fixed-schedule run with an observer that asks for every view agrees with the
  run without one (class R). The entrance is observed, and a resume does not repeat planes.
  A failing observer is detached and the run completes; a stop raised from an observer stops
  the run. Both on `numpy_ref`, `numba_cpu` and the native build, field-free planes
  included.
- **Estimate.** On the three backends, a budget below the persistent memory is reported
  without allocation, and the reported memory agrees with `solprop.driver.memory`.
- **Pins and scans.** A pinned pressure equals `pressure_zdw` for the same target, where it
  has a single root; a scan re-solves its pins per point; two mode-parameter pins, a pinned
  gradient and an unreachable target are refused. `Recipe.scan()`'s keywords carry
  `[execution]` to `run_scan`.
- **Design solvers.** `solve_parameter` with the Marcatili builder reproduces `pressure_rdw`
  (with `peak_power`, `fwhm` and the nonlinear phase) to its tolerance, `pressure_zdw` where
  it has one root, and finds both roots where a resonance gives two; `radius_zdw` inverts
  `zdw`.
- **Loaders.** The test file reads to the stated energy within the quadrature error of its
  sampling, and to the centre wavelength within one sample spacing; a
  wavelength-to-frequency conversion conserves energy to the same bound. A delay of $\tau$
  read in each `phase_convention` puts the pulse at $\pm\tau$. A large chirp in an unwrapped
  column survives, a wrapped column unwraps with a noise floor beside it, degrees are read,
  and an undersampled wrapped column warns.
- **Launch.** On a Marcatili basis the per-mode energies equal the shares; on a general
  modal basis the flux-metric energy equals the pulse energy. Two recipes with the same
  pulse and different launches refuse each other's files under SEP 42's gate.
- **Beams.** Displacement and tilt agree with `coupling.Alignment`'s definition; each
  frequency leaves at one angle; the energy after the screen is the pulse's; each refusal of
  section 4 and §5.6 fires. A record without the fields continues through its neutral entry.
- **Previews and capabilities.** `preview_input` equals the run's noise-free initial field;
  `modes` equals the built mode set. Each refusal `OutputCapabilities` states matches the
  reader's actual refusal, so the two cannot drift.
- **Safety.** A recipe naming an unregistered type, holding a refused key, or reaching a
  path outside its directory (symbolic links resolved) without an explicit allowance is
  refused; an embedded data file is never written at its stored name, and the cap holds; a
  received file's own budgets are ignored.

### 8. Documentation

- A new guide page, *Recipes and the command line* (`guide/setup/recipes.md`): the format,
  quantities, `None`, paths, objects, data files, pins, scans, `solprop run`, `check`,
  `export` and `resume`, with the capillary example above, and the examples that have no
  recipe, with why.
- [Resuming](../../guide/results/resuming.md): resume from a file alone, `resumability`, the
  portable stop, and that resuming a received file runs its setup.
- [Progress](../../guide/performance/progress.md): `PlaneObserver`, its failure policy and
  the coalescing pattern, how the boundary warning reaches a front end, `ScanReporter`.
- [Pulses](../../guide/setup/pulses.md) and [beams](../../guide/free-space/beams.md):
  loaders, mode launch (and its meaning on general bases), displacement and tilt.
- [Design tools](../../guide/workflows/design-tools.md): the general solver, `radius_zdw`
  and pins.
- [Output](../../guide/results/output.md): the recipe group, result capabilities and the
  writer-lock sidecar.
- A table of route defaults (integrators, tolerances) with the reason for each difference.
- Docstrings for every new name; a gallery example that loads a recipe, edits a value and
  runs it; the example recipes listed on the recipes page.

### 9. Compatibility and impact

- **Python interface: additive.** Every name is new, every new keyword defaults to today's
  behaviour, and the new reporter calls are optional protocols with names no reporter is
  likely to have (adding them to `ProgressReporter` itself would break third-party reporters
  and is not proposed). No exception changes class.
- **Output schema: unchanged.** The recipe group is a top-level group written once, and
  outside the hash, so files written before and after phase 2 are read and resumed alike
  (SEP 42 §5.8). S5 and S6 add recorded fields, each with a neutral entry and compared under
  SEP 42. So this SEP lands after SEP 42's phase 1 and bumps nothing.
- **Recipe stability.** Recipe `format = 1` becomes a public format when this SEP merges.
  Renaming a keyword later then needs a format upgrade rule as well as a changelog entry.
  Every keyword of a registered type is part of the format too, so the registry starts
  small. The route vocabulary was aligned (`wavelength`, `shape`) before the first release,
  and S14's `atol` and `raman` rows stay as they are (Q3).
- **Two choices that would be breaking, and are not recommended:** renaming keywords to one
  vocabulary across routes where the meanings differ (`loss`/`loss_db_per_m`), and aligning
  integrator or tolerance defaults across routes (the per-route defaults were qualified
  separately).
- **Validated results:** unchanged; no computed field changes.
- **Dependencies:** `tomli-w`, to write TOML (section 6, Q6).

### 10. Alternatives considered

- **The recipe in the front end only.** Simpler for solprop, but scripts, batch jobs and
  results files would not share it, the front end would copy solprop's defaults and rules,
  and resume from a file would still be impossible.
- **HDF5 as the setup format.** The output already holds the resolved configuration, but it
  is not editable by hand, not diffable, and cannot rebuild a simulation. TOML is for
  setups; HDF5 embeds the TOML.
- **JSON or YAML.** JSON has no comments and is noisy to edit; YAML's implicit typing turns
  values such as `no` or `1e-5` into surprises, and the standard library cannot read it.
  TOML has a standard-library reader and explicit types.
- **Pickled simulations.** Not readable, not stable across versions, and unsafe to load.
- **One SEP per change, or two SEPs (the recipe; the front-end hooks).** The changes share
  one purpose and one test strategy (the recipe round trips), and the hooks name recipe
  paths, so splitting would publish recipe format 1 before the pins, paths and previews it
  must describe. One SEP, on one branch, lets the format freeze once.
- **Turning objects back into tables** (a `Recipe.from_call`). It would need every object to
  keep what made it, or a factory to attach its table, and would be wrong for hand-built
  objects. A recipe always comes from a recipe, or from plain values.
- **A resolved recipe holding every keyword.** Simple to state, but it does not rebuild its
  run: values read back from a default build move the hash or have no table form (§5.2).
- **Comparing hashes to detect a simulation changed after its build.** It costs a
  translation at every build, gives a false alarm on an unseeded resume and misses a changed
  `fields`. Mutators dropping the recipe are cheap and exact.
- **A boundary hook on the observer.** The monitor has no verdict to give: it warns once or
  stops, and a front end has both.
- **A rule language for cross-field validation.** Instant validation of every rule would
  need the value-dependent refusals of translation written a second time. `check()` reuses
  them; only the combination rules are listed (S8).
- **Signals for stopping.** Portable only on POSIX; a flag is portable and testable.

### 11. Phases

The phases are made on one branch and the SEP reaches `main` in one pull request when its
last commit sets `Final` (SEP 0, *Implementation*). The branch starts after SEP 42's phase
1, whose gate S2 and S17 rest on. Phase 4 follows SEP 40, because it registers the
free-space types, whose interface and default SEP 40 changes. A front end in development
builds against the branch; the order puts what it needs first.

1. **Recipe core on `gnlse` and `capillary` (S1, S9, S10, S12).** `solprop.recipe` with
   format 1; quantity parser and TOML writer; paths, `None`, complex numbers, arrays and
   data files; the factory registry for these routes' objects; `load_spectrum` as its first
   data-file entry; metadata tables with forms, resolvers and their sync tests;
   `Recipe.load/save/get/set/check/build/run/to_python/from_keywords/example`; `parameter`
   on errors and warnings; the `solprop` console script with `run`, `check` and `export`;
   the guide page. Verifies recipe equivalence, Python export, quantities and the loader's
   phase rules on every keyword-route example of these routes.
2. **Provenance, resume and stop (S2, S15, S17).** The recipe group in the output, the
   resolvable keywords and their test, `Recipe.from_output`, `solprop resume` and
   `resume_file` with their modes, the `stop` flag per executor, `resumability`; the
   writer-lock text. Verifies resume from a file after a stop and after SIGTERM, and the
   received-file limits.
3. **Watching runs (S3, S4, S11, S16).** `PlaneObserver` and `PlaneView`, `ScanReporter` and
   the pool's progress message, `Simulation.estimate()`, `[scan]` tables and
   `Recipe.scan()`, result capabilities. Verifies observer neutrality and failure on the
   three backends, and scan progress on local and pool executors.
4. **Object routes, waveguide, launches and previews (S1 continued, S5, S6, S18).** Recipes
   for `waveguide`, `radial`, `free` and `planar`, with the documented user-facing types
   registered and listed (guides, mode providers, grids, spaces, media, beams, responses and
   settings, and the factories such as `absorber`); `launch` with its record; beam
   displacement and tilt with their neutral entries; `Recipe.modes()` and `preview_input()`.
   Verifies recipe equivalence on these routes' examples, the launch normalisation and the
   beams.
5. **Design, pins and capability (S7, S1's pins, S8).** `solve_parameter`, `radius_zdw`;
   `[pins]`, `Recipe.resolve` and `pin_roots`; pins carried into what already reads values,
   `Recipe.scan()`, `Recipe.modes()`, `preview_input()` and the resolved recipe;
   `solprop.capabilities.route` and `species` with rule ids.
6. **Drawings and keyword gaps (S13, S14).** `solprop.plotting.geometry` and the guide
   figures moved onto it; the missing keywords added; the route-defaults table.

### 12. Open questions

- **Q1.** Recipe values: accept both quantity strings and SI numbers, and write quantity
  strings (recommended), or SI numbers only? **Decision (John Travers, 2026-11-01):** as
  recommended.
- **Q2.** Should `[recipe] route` be required, or inferred from the keywords present
  (recommended: required; inference is fragile between `capillary` and `waveguide`)?
  **Decision (John Travers, 2026-11-01):** as recommended.
- **Q3.** S14's open rows: align the `atol` defaults, and give `prop_gnlse` the gas-route
  `raman` default semantics? Each is a change of default. Recommended: keep both and
  document them as deliberate; if either is to change, change it in a minor release before
  this SEP merges, listed as a break (ADR-058), so that recipe format 1 is published with
  the final defaults. **Decision (John Travers, 2026-11-01):** as recommended; both stay and
  are documented as deliberate.
- **Q4.** Should `solprop resume` also accept a recipe edited after the run (for example a
  longer `length`), refusing only when the hash differs in more than the run's end?
  Recommended: no in this SEP; extending a finished run is its own question. A changed grid
  is always a new run. **Decision (John Travers, 2026-11-01):** as recommended. Extending a
  finished run is [#141](https://github.com/LupoLab/solprop/issues/141), simplest after this
  SEP's recipes; SEP 42 does not take it on.
- **Q5.** Which pulse-retrieval formats does a retrieval loader support first? **Decision
  (John Travers, 2026-11-01):** the usual three columns of frequency in Hz, spectral power
  density and spectral phase, as Luna.jl's `DataPulse` takes, with care over the phase when
  regridding. §5.12 now specifies this, through `load_spectrum` alone. The phase is in the
  optics convention by default, with a keyword for the other sign.
- **Q6.** Is `tomli-w` acceptable as a dependency instead of a subset writer? **Decision
  (John Travers, 2026-11-01):** yes, take the dependency; it is light, and there is no
  reason to rewrite it.
- **Q7.** Should the metadata include a recommended display unit per keyword (fs for
  durations, µm for radii), or is that left to front ends? Recommended: include it, so that
  recipes written by any front end use the same quantity strings. **Decision (John Travers,
  2026-11-01):** as recommended. Display units are spelled in ASCII in a recipe (`um`); `µm`
  is read too, and a front end shows what it likes.
- **Q8.** If S5 or S6 cannot record its launch through existing records, is one
  output-schema bump in phase 4 acceptable (listed under ADR-058), or should the feature
  wait for the resume-compatibility policy of 1.0? Recommended: the bump, made once and in
  the same minor release as any other schema change. **Decision (John Travers,
  2026-11-01):** avoid the bump through SEP 42, under which an additive record field moves
  neither the schema nor a resume. SEP 42 is now accepted and comes first, so no bump is
  needed and the fallback is gone (section 9).
- **Q9. The shape of format 1.** One `[build]` table with no group tables (recommended,
  §5.1, §6), or the group tables the maintainer first approved, with their names changed so
  that none is a keyword. **Decision (John Travers, 2026-11-01):** as recommended.
- **Q10. A stop that returns.** `run(stop=)` returns the handler when its own flag stops it
  (recommended, §5.17), or raises like SIGTERM, a `SystemExit` that a front end's `except
  Exception` misses and that ends a worker thread silently. **Decision (John Travers,
  2026-11-01):** as recommended.
- **Q11. One way only.** Drop `Recipe.from_call` for `Recipe.from_keywords` with plain
  values, and never turn an object into a table (recommended, §5.1), or have factories
  attach their tables so that more calls can be captured. **Decision (John Travers,
  2026-11-01):** as recommended.
- **Q12. Edited recipes lose their comments.** `save` is verbatim until a recipe is edited
  and canonical after (recommended, §5.1), or preserve comments, which needs a TOML library
  that keeps them, a heavier dependency than Q6 took. **Decision (John Travers,
  2026-11-01):** as recommended.

### 13. References

1. TOML v1.0.0 specification, https://toml.io/en/v1.0.0
2. Python Software Foundation, `tomllib` — Parse TOML files,
   https://docs.python.org/3/library/tomllib.html
3. Python Software Foundation, `os.kill` (Windows behaviour),
   https://docs.python.org/3/library/os.html#os.kill
4. Python Software Foundation, `decimal` — Decimal fixed-point and floating-point
   arithmetic, https://docs.python.org/3/library/decimal.html

## Part II — Implementation record

*One report per phase, appended in the pull request that merges the phase (SEP 0 §8).*

## Part III — History

- 2026-10-31: Drafted, from the requirements of a desktop front end under design.
- 2026-10-31: Opus 5.5 completed the AI review (R1–R20, below). The review read a
  revision 321 commits behind `main`; on `main` a resume takes the noise seed recorded
  in the file when the call gives none, so R3's refused resume no longer occurs (its
  points on equivalence tests and scan seeds stand), and the output schema is 28, not 27.
  The draft was rebased onto `main`. Part I is not yet revised; the response follows the
  maintainer's decisions on R2 and R10.
- 2026-10-31: Codex completed a supplementary AI review (C1–C5, with updates to R3 and
  R19, below), at revision `c4cb347`; Part I unchanged.
- 2026-10-31: Opus 5.5 (drafting agent) answered R1–R20 and C1–C5 and revised Part I
  against `main` at `03807f8` (response below): a defined vocabulary with an
  `[execution]` table, exact decimal quantities, authored and resolved recipes stored
  without a schema change, parameter paths and pins in format 1, a lazy and
  failure-isolated plane observer, result capabilities (S16), a portable stop and
  resumability query (S17), previews (S18), and one branch and one pull request.
- 2026-11-01: Maintainer decisions (John Travers) on Q1, Q2, Q4–Q8 (§12). Part I revised for Q5 and Q6: one `load_spectrum` with the usual three-column
  layout and the phase rules, and `tomli-w` to write TOML.
- 2026-11-01: Maintainer decisions on Q3 (both defaults stay) and the phase sign (optics by
  default). All of Q1–Q8 are decided. Rebased onto `main` after the 0.1.0 release; §3, §5.12,
  §5.14 and §9 brought up to date.
- 2026-11-01: Supplementary AI review of the revised text (Opus 5.5).
- 2026-11-01: Opus 5.5 (drafting agent) answered G1–G25 and revised Part I (response below), with
  the maintainer's leave to reopen earlier decisions: one `[build]` table, factories with a declared
  context, one-way recipes, a resolved recipe defined by what rebuilds the run, a recipe carried by
  the simulation, a stop that returns, and SEP 42 as a requirement. Q9–Q12 ask for confirmation.
- 2026-11-01: Human review (John Travers), and decisions on Q9–Q12 as recommended. Accepted by
  John Travers; Part I frozen.
- 2026-11-01: Tracking issue: [#143](https://github.com/LupoLab/solprop/issues/143). Brought to the standard of ADR-062 before the pull request
  merged: §3 lists the open issues, and Q4 names [#141](https://github.com/LupoLab/solprop/issues/141). §11 notes that phase 4 follows SEP 40, as
  the maintainer ordered.

### AI review, 2026-10-31

Reviewer: Opus 5.5 (AI). Read the whole of Part I at revision `02d09ab` against SEP 0's
acceptance criteria and the code on that revision, with probes in the project
environment (signatures, small builds, one 1 mm GNLSE run on `numpy_ref`, no test
suite). **Recommendation: revise before acceptance.** The purpose and the split of work
between solprop and a front end are sound, and the API changes are additive as claimed.
Three findings are blockers: the default configuration cannot be resumed from a file
(R3), the format's vocabulary is not one set of keywords (R4), and the phase plan
contradicts SEP 0 (R10). Two findings concern the first public release (R1, R2).

One trivial edit was made to Part I: in section 5.1 the inline `GaussianPulse` table
used `lambda0`, which `GaussianPulse` does not accept; it now uses `wavelength`, its
keyword. Nothing else in Part I, the header or the history list was changed.

#### Findings

1. **R1 — major: the schema bump ends resumability of every earlier file (sections 9,
   11, 5.5, 5.6, 5.14).** The schema version is inside the resume key
   (`src/solprop/output/schema.py:2896`), and the guide says so: "only a file written at
   the current output schema … can be resumed" (`docs/guide/results/resuming.md:47`).
   So phase 2's bump from 27 to 28 makes every file written by the first public release
   unresumable by the version that adds S2, by any route. Section 9's "a file written
   before has no recipe and is resumed from its script" is therefore false. The same
   applies to S5, S6 and S14 if they add recorded fields: "a change that adds a field to
   a recorded dataclass moves the configuration hash and bumps the output schema"
   (`docs/guide/results/output.md:336`), and `SpatialPulse` is a recorded dataclass
   (fields `temporal, waist, order, propz, jones`). **Recommendation:** the `recipe`
   dataset is optional and outside the hash, and no reader check refusing a newer
   schema was found, so store it without a bump and test for its presence. Alternatively, reserve it
   before the release is tagged. Require S5, S6 and S14 to record new keywords only
   when they differ from the default, as `configuration["transverse_measure"]` already
   does, so that their phases do not move the hash. Correct section 9 either way.

2. **R2 — major: the release, not recipe format 1, is the deadline for S14's open rows
   (sections 5.14, 9, Q3).** Q3 recommends settling the `atol` defaults (1e-14 on
   `prop_gnlse`, 1e-10 on the gas routes) and the GNLSE `raman` default "before
   recipe format 1 is published". Once solprop is public, changing either is a change
   of default for every script, which section 9 itself rules out. Section 9 calls
   renaming "not recommended". The release also fixes the vocabulary splits that
   recipes inherit at two levels: routes take `lambda0` and `pulseshape`, while pulse
   objects and `solprop.tools` take `wavelength` and `shape` (`GaussianPulse(*,
   wavelength, …)`, `soliton_params(…, wavelength, shape)`). **Recommendation:**
   decide Q3, and whether these splits are ever to be unified, before the release is
   tagged. If they are kept, list them in S14's table as deliberate. Nothing else in
   this SEP needs to land first: every proposed name is new.

3. **R3 — blocker: with default noise, a rebuilt simulation never matches its file
   (sections 4, 5.1, 5.2, 7).** `noise="default"` with `seed=None` draws a fresh seed
   at each build. Two `Simulation.from_capillary` calls with identical arguments
   differ in `noise.seed` and the whole noise spectrum (`RunMetadata.differences`).
   Running `prop_gnlse` for 1 mm and calling it again with `resume=True` is refused:
   "these fields changed: noise.seed, noise.spectrum…". "Only what was set is written"
   therefore makes `solprop resume run.h5` fail for any default run, and the recipe
   equivalence test of section 7 fails unless a seed is pinned. `Recipe.scan()` has the
   same issue: a scan builder must pass `point.seed`, or the point is rejected
   (`docs/guide/workflows/scans.md:255`). **Recommendation:** record the resolved seed
   with the stored recipe, or have resume take it from the file's noise record. If the
   change that reuses a stored seed on resume lands first, cite it; on this revision it
   has not. Define recipe equivalence with the seed fixed, and make `Recipe.scan()`'s
   builder pass `point.seed`.

4. **R4 — blocker: "the route's keywords" are three different sets (sections 5.1, 5.2,
   5.14, 6).** `prop_*` takes `output`, `progress`, `progress_period`, `resume`,
   `snapshots`, `boundary_check` and, on the capillary, `tdse_archive`. None of these is
   a `Simulation.from_*` keyword; `boundary_check` is applied afterwards to
   `Simulation.settings` (`src/solprop/driver/interface.py:387`), and the rest go to
   `Simulation.run`. `from_waveguide` takes `memory_budget`, which no `prop_*` has.
   Some settings a front end needs are on no keyword surface at all: `fields`,
   `field_planes` and `resume_history`. The guide sets them by replacing
   `simulation.settings` (`docs/guide/setup/running.md:445`). `Recipe.build()`
   returns a `Simulation`, and `to_python()` writes a `prop_*` call, so the format must
   say which set it uses. **Recommendation:** define the vocabulary as the `from_*`
   keywords plus a named list of execution settings (`boundary_check`, `memory_budget`,
   `fields`, `field_planes`, `backend`). Refuse `output`, `progress`, `resume` and
   `snapshots` in a recipe, since the command line supplies them. Extend S9's sync
   test to all three surfaces, and add the `memory_budget` asymmetry to S14.

5. **R5 — major: the quantity rule is not exact, and one unit per keyword is not
   enough (sections 4, 5.1, 5.9, 7).** Scaling by a factor table in binary floating
   point changes values. `7.5 * 1e-15` is `7.500000000000001e-15`, `800 * 1e-9` is
   `8.000000000000001e-07` and `275 * 1e-6` is `0.00027499999999999996`, each 1 ulp
   from the literal a script writes. Records serialise floats exactly, so section 1's
   round trips (equal `configuration_hash`) fail for the SEP's own example. Several
   keywords also mix units inside one value: `phases` is `[rad, s, s², s³, …]`
   (`src/solprop/driver/interface.py:531`), `betas` is in sⁿ/m by order, and the
   `(z, p)` form of `pressure` pairs metres with bar (`interface.py:870`). **Recommendation:**
   convert in decimal, by exponent shift of the parsed mantissa, then round once to
   the nearest double; the three examples then give the literal. Check dimensions,
   not only scale, so that `"125 um"` for `pressure` is refused. Let S9 give a unit per
   element or per form for these keywords. Writing `8e-07` back as `"800 nm"` needs the
   same decimal rule.

6. **R6 — major: format 1 must say how to write `None`, and what `from_call` refuses
   (section 5.1).** TOML has no null. Yet `None` is a deliberate non-default value for
   several keywords: `noise=None` turns noise off against the default `"default"`
   (`interface.py:131`), and `absorber=None` and `absorber=False` mean different things.
   Callables are accepted by `beta`, `loss` and `MarcatiliMode(loss=…)`; the format
   cannot hold them. Also, `ZProfile` values built by class methods
   (`PressureGradient.two_point`) are not described by a constructor's keywords.
   **Recommendation:** fix a spelling for `None` in format 1, for example the string
   `"none"` where a keyword accepts `None`, and record it in the metadata. State that
   `from_call` refuses callables and arbitrary arrays, naming the keyword. Test every
   registered type in both directions (table → object → table), not only its
   constructor.

7. **R7 — major: the recorded recipe can disagree with the run it is stored beside
   (section 5.2).** The recipe is outside the hash, so nothing checks it. A simulation
   built from a recipe can still change before it runs: the `settings` setter
   (`src/solprop/driver/simulation.py:1755`), `with_input` (which takes an array) and
   `with_mixture`. The file would then hold a recipe that does not rebuild its run.
   "Only what was set is written" also means a file reopened under a later solprop
   shows that version's defaults, not the run's. **Recommendation:** store with the
   recipe the `configuration_hash` it built. At run start, drop or refuse the recipe
   when the simulation's hash differs, and on resume compare the two. Store a resolved
   recipe (every keyword, the seed included) beside the as-written one, so that a front
   end can show the setup that ran.

8. **R8 — major: the plane observer's device and cost statements do not hold in
   general (sections 5.3, 6).** "On device backends the plane has already been copied
   to the host" is false for field-free planes. Under `fields="none"` or `"planes"`, "a
   field-free plane asks the backend for no field at all … so nothing crosses to the
   host" (`src/solprop/driver/run.py:739`), and `run_scan` defaults to `fields="none"`.
   If `NullReporter` implements `observe`, every `progress="none"` run satisfies
   `ObservingReporter`; the protocols are `runtime_checkable`, so the check is by name
   (`src/solprop/log.py:340`). The driver could then build a `PlaneView` at every plane
   of every quiet run. For the same reason, any existing reporter with an attribute
   named `observe` would start receiving calls; `Backend.observe(*, field=…)` shows
   the name is already in use. The time-domain and per-mode views need transforms that
   someone pays for. **Recommendation:** say what `PlaneView` offers on a field-free
   plane: no field, or a fetch whose transfer is stated. Make `PlaneView` lazy, and do
   not give `NullReporter` the method. Choose a specific name, such as `observe_plane`.
   Test neutrality on `numpy_ref`, `numba_cpu` and the native build, with field-free
   planes included.

9. **R9 — major: S10's premise and S8's completeness test do not match the code
   (sections 3, 5.8, 5.10, 7).** Builders already raise `SolpropError` subclasses:
   across the package there are 249 `raise DriverError`, 574 `TranslationError` and
   409 `ModeError`, and only 17 bare `ValueError`/`TypeError`, none in the interface
   builders. So the reclassification in section 5.10 is nearly empty. The real gap is
   the other way: `DriverError` is a `RuntimeError` (`src/solprop/exceptions.py:106`),
   so `except ValueError` does not catch a refused keyword. "Every refusal is listed"
   cannot be tested as written. `driver/simulation.py` alone has 151 refusals, and
   `translate/` has 533; most depend on values (grid coverage, node budgets, material
   validity), not on keyword combinations. `check()` stops before translation, which
   holds those 533 sites and costs 0.13 s warm against 0.02 s for the build in a probe
   of section 5.1's capillary. **Recommendation:** give rule ids only to the
   combination refusals of the interface layer (route × keyword × species). Enforce
   them with a lint-style test that every `parameter=` refusal names a rule, and leave
   value refusals to `check()`. Let `check()` translate, stopping before backend
   construction. Collect every issue only at the metadata level (type, unit, range,
   unknown key), then report the builder's first refusal.

10. **R10 — blocker: the phase plan contradicts SEP 0 (section 11).** "Each phase
    merges on its own with a green gate" and "a front end can start on phase 1 alone"
    conflict with SEP 0 *Implementation*. Every phase is made on one branch, and the
    SEP reaches `main` in one pull request when its last commit sets `Final`. Under that
    rule the front end gets nothing until all six phases, including drawings and
    retrieval formats, are done. **Recommendation:** either split the proposal, with
    S1, S2, S9, S10 and S15 in this SEP and the front-end hooks (S3–S8, S11–S14) in a
    second, or record a justified exception to SEP 0. Phase 1 also covers
    `prop_waveguide`, whose `modes` and `provider` are objects. That brings mode and
    guide types into phase 1's registry, not phase 4's.

11. **R11 — minor: S5 overlaps keywords that exist and names modes in a form that does
    not exist (section 5.5).** `prop_waveguide` already has `channel` ("by index or
    display label", `interface.py:1376`) and `incident` with `coupling`. Mode labels are
    `ChannelLabel`s that display as `HE1,1:even`, not `"HE11"`, and the parity matters
    when both polarisations are present. No keyword of these routes takes a field
    array; the array path is `Simulation.with_input`. A per-run `launch` cannot express
    a two-colour input with the pump and seed in different modes. **Recommendation:**
    resolve labels as `channel` does, state how `launch` relates to `channel`,
    `incident` and `with_input`, and decide whether `launch` is per pulse.

12. **R12 — minor: S6 introduces physics without theory, and a second name for an
    existing quantity (sections 4, 5.6).** `coupling.Alignment` calls the offset
    `displacement`. Its tilt is a phase screen with the medium's wave number, certified
    against a required `model_error` (`src/solprop/coupling/elements.py:1612`). For a
    broadband beam, a tilt defined with $k(\omega)$ is a pure angle, while one defined
    with $k_0$ adds angular dispersion. Section 4's "no new physics" does not settle
    which. **Recommendation:** define the tilt in section 4, reuse `displacement` (or
    rename both consistently before either is published), and say whether the screen's
    budget applies. See R1 for the record.

13. **R13 — minor: the general design solver needs more than a mode builder (sections
    3, 5.7, 7).** `pressure_rdw` already takes a `lattice` and builds a `VincettiMode`
    (`src/solprop/tools.py:2661`), so section 3's "written for the Marcatili
    capillary" is too narrow. The tools search rather than bracket once: the
    zero-dispersion search samples $\beta_2$ and discards sign changes through a pole,
    because a VUV resonance flips its sign twice, and the modes' own `zdw` misses such
    pairs (`docs/guide/workflows/design-tools.md:98-111`); `pressure_rdw` samples 129
    logarithmic and 129 linear pressures. A generic root finder over a bracket would
    reintroduce the failure the tools avoid, and the parameter-to-target map need not be
    monotonic. `solve_parameter` should reuse these searches and report every root it
    finds. The `("rdw", (pump, rdw))` target omits `pressure_rdw`'s `peak_power` and
    `nonlinear_phase`, so it can reproduce `pressure_rdw` only for linear phase
    matching. A `mode` keyword beside the
    required positional `radius` is awkward. **Recommendation:** carry the nonlinear
    phase in the target, return all roots, and make the `mode` form a separate function
    or make `radius` optional when `mode` is given.

14. **R14 — minor: metadata needs effective defaults and constructor tables (section
    5.9).** Many defaults are `None` resolved by rule: `rtol` (per route), `plasma` and
    `thg` (`not envelope`), `step_planes` (0, or 1 with TDSE), `raman` (by species) and
    `pressure` (one bar). A front end must show the value that will be used, which the
    signature does not give. Object tables (`GaussianPulse`, modes, profiles) need
    units for their own keywords, but S9 covers only interface functions.
    **Recommendation:** add a "resolved by" rule or resolver per keyword, and extend
    S9 to every registered type.

15. **R15 — minor: section 3 is wrong about scan progress (sections 3, 5.4).**
    `run_scan` already has `progress="auto"` with an outer bar or one log line per
    finished point (`src/solprop/driver/_scan_run.py:760`); only the inner runs use
    `progress="none"`. A pool knows when it dispatches a point, but per-point events
    from a worker need new messages on the worker pipe. **Recommendation:** correct
    section 3, state how a `ScanReporter` combines with the literal choices, and say
    what the pool's forwarding sends.

16. **R16 — minor: the cost estimate as described would include start-up and could
    allocate what it is meant to admit (section 5.11).** Timing right-hand sides needs
    a constructed backend, which allocates the persistent memory the estimate reports.
    On a cold process, numba compilation and FFTW planning exceed "about a second".
    Calls per step differ by integrator (one for the exponential multistep, six or
    seven for Runge–Kutta). **Recommendation:** admit memory first, exclude warm-up from
    the timing, and report cost per right-hand side with the integrator's calls per
    step.

17. **R17 — minor: loading other people's files needs stated limits (sections 5.1,
    5.2, 7).** `solprop resume` on a received HDF5 file executes that file's recipe.
    Embedded data files must be read into memory or written under generated names,
    never at their stored names, which could hold `../` paths. The recipe's execution
    keys must not choose output, archive or snapshot paths (R4). Node and memory
    budgets bound a hostile grid. "Outside the recipe's directory" must resolve symbolic
    links. Parsing an untrusted HDF5 file is itself an exposure of the HDF5 library.
    **Recommendation:** add these to section 5.2 and to the safety tests, and say in
    the guide that resuming a received file runs its setup.

18. **R18 — minor: a front end needs a few hooks the SEP does not list (sections 2, 5).**
    Exploring a guide before a run needs the modes a recipe would build, without
    building the run (`Recipe.modes()` or similar). A preview of the input needs the
    pulse and beam on the grid. Warnings should carry `parameter` as errors will
    (`MaterialRangeWarning`, `BoundaryWarning`, the provisional-Raman warning). The
    boundary monitor's verdict should reach a reporter, not only the log and the file.
    Cross-field rules (energy or peak power; pulse inside the window) should be
    machine-evaluable if S8 is to serve instant validation. **Recommendation:** add
    these to S8–S10, or defer each explicitly to a named phase.

19. **R19 — minor: verification tolerances and backends (section 7).** "Matches … to
    rounding" should name the bound and where it is set (the reproducibility page).
    The loaders need a stated tolerance on energy and centre wavelength. SEP 0
    *Completion* requires the observer and the estimate to run on `numpy_ref`,
    `numba_cpu` and the native build, with parity. Round-trip tests need a pinned seed
    (R3) and exact quantity conversion (R5). **Recommendation:** add these to
    section 7.

20. **R20 — minor: other statements about current code (section 3).** `prop_planar`
    takes no `medium`, and all three object routes take `grid` first and `frame`.
    `SpatialPulse`'s signature starts with `temporal`. Not every error carries only a
    message: `ResponseBudgetError` carries `z`, `field` and `calls`. S15's sidecar is
    already documented, under snapshots (`docs/guide/results/output.md:475`), so S15
    moves the text rather than writing it. The guide pages section 8 will edit state
    stale schema numbers: 22 in `resuming.md:47` and 25 in `output.md:308`, against
    `SCHEMA_VERSION = 27`. **Recommendation:** correct section 3. The stale numbers are
    outside this SEP but are worth fixing before the release.

#### Verdict on "every change is additive"

For the Python API the claim holds. Every name is new, and new keywords default to
today's behaviour. Exception reclassification by multiple inheritance keeps existing
`except` clauses. `step_planes=None` on `prop_gnlse` widens an `int` annotation without
changing behaviour. There are three qualifications. Files are not compatible across
the change: the schema bump and any newly recorded fields end resume of earlier files
(R1). `observe` on `NullReporter` changes what the driver does for existing reporters
that already have such an attribute (R8). And the defaults and names S14 leaves open
become fixed when solprop is first published, not when recipe format 1 is (R2).
Nothing must land before the release for API reasons. Q3, the vocabulary splits and the
schema policy for the `recipe` dataset should be decided before it.

**Evidence and limits.** Probes on revision `02d09ab`:

- `inspect.signature` of the six routes, the three `from_*`, `Simulation.run`,
  `run_scan`, `SpatialPulse`, `DataPulse`, `GaussianPulse` and the design tools.
- Repeated `from_capillary` builds compared with `RunMetadata.differences`.
- A 1 mm `prop_gnlse` run on `numpy_ref` and its refused resume. The native backend was
  not used: the shared environment's extension did not match this revision.
- Build and translate timings of the section 5.1 capillary, warm.
- Counts of `raise` sites by class.
- The three float products in Python.

No test suite, propagation qualification or GPU run was performed. Front-end needs were
judged against the requirements stated in Part I, section 2.

### Supplementary AI review, 2026-10-31

Reviewer: Codex (AI). Reviewed Part I, the existing AI review and relevant source and
documentation at revision `c4cb347`, considering the requirements of a desktop front
end. **Recommendation: retain the architecture, but revise the outstanding contracts
before acceptance.** This supplement adds findings or updates earlier ones; it does
not repeat R1–R20 or change Part I, the proposal status or the earlier review.

#### Additional findings

1. **C1 — major: expose result capabilities as well as setup capabilities (sections
   2, 5.3 and 5.8; additional to R18).** A front end needs to know which quantities
   can be reconstructed from a particular file, not only which options a route
   accepts. The [processing guide](../../guide/results/processing.md) warns that
   `get_power` and `get_spectral_density` accept spatial files but do not return
   physical powers or energies there. Compression refuses spatial and general modal
   outputs. General modal reconstruction requires the matching guide and reference
   tables, while the file may contain only their identities. Field retention and
   recorded statistics further limit each plane's available views. **Recommendation:**
   expose these requirements and refusals through a result-capability query, extending
   the existing output capability machinery where appropriate. Distinguish physical
   flux, coordinate quantities and their units. Do not promise that `PlaneView` alone
   makes the same physical operations available on every basis.

2. **C2 — major: define pins and nested parameter paths before format 1 is frozen
   (sections 5.1, 5.7, 5.9 and 5.10; additional to R13 and R14).** A setup that pins
   pressure to a target ZDW needs both the constraint and its resolved value. The
   proposed keyword tables do not yet describe this intent, when to re-solve it, the
   chosen root and search domain, or failure when a target becomes unreachable. A
   scan of a second pulse's energy or a profile knot also needs an address within an
   object, beyond a top-level route keyword. **Recommendation:** define canonical
   nested parameter paths shared by editing, diagnostics, scans and constraints.
   Specify constraint dependencies, cycle refusal, resolution order and whether each
   scan point re-solves its pins. Retain the authored intent beside the resolved run
   definition, and define equivalent headless and Python-export behaviour. If pins
   are deferred, state that limitation explicitly rather than leaving scientific
   setup state to individual front ends.

3. **C3 — major: mode energy fractions are not generally additive (section 5.5;
   additional to R11).** `prop_waveguide` also supports general modal bases whose
   physical energy uses a flux metric with cross terms. A set of coordinate fractions
   normalized to one does not establish the advertised physical energy allocation;
   relative phases can affect the total. The existing
   [general modal processing contract](../../guide/waveguides/general-modal.md)
   distinguishes coordinate quantities from physical flux. **Recommendation:**
   explicitly restrict `launch` fractions to bases where this interpretation holds,
   or define a coordinate-amplitude specification normalized using the actual flux
   metric. Resolve this physical meaning alongside the labels and per-pulse scope
   already raised in R11.

4. **C4 — major: front ends need a portable stop contract (sections 2, 5.2 and 5.3;
   additional to R18).** Sending `os.kill(pid, signal.SIGINT)` does not gracefully
   interrupt a Windows worker: Python documents unconditional termination for values
   other than its special Windows console-control events
   ([`os.kill`](https://docs.python.org/3.14/library/os.html#os.kill)). A front end
   cannot infer resumability from a process exit or from `n_saved` alone. Some
   failures have no checkpoint, and interpolated checkpoints need an explicit policy
   ([resuming](../../guide/results/resuming.md)). **Recommendation:** specify either
   a supported portable cancellation hook respecting commit boundaries or the tested
   platform mechanisms a front end must use. Expose enough recovery information to
   explain whether resume is available. Keep front-end execution states distinct
   from `RunOutcome.status`, which has no `stopped` member. A changed grid, including
   a wider time window, requires a new run rather than ordinary resume.

5. **C5 — major: observer neutrality needs a failure and workload policy (section
   5.3; additional to R8).** Read-only plane views prevent ordinary field mutation,
   but do not prevent a slow observer from delaying the march or a failed transport
   from raising through the run loop. The current `reporter.plane` call is inside
   the driver's exception-recovery boundary. **Recommendation:** define the observer
   exception policy and the handling of a disconnected client. Give front ends a
   documented way to bound observation work and coalesce replaceable display updates
   without blocking propagation on client rendering. Keep reliable terminal/error
   events distinct from replaceable previews. These transport policies can belong to
   the front end, but the solprop hook must state the guarantees it does and does not
   provide. Test observer failures and disconnection as well as successful copies.

#### Updates to earlier findings

- **R3, unseeded resume: the resume blocker is addressed at this revision.**
  `Simulation._adopt_recorded_seed` in `src/solprop/driver/simulation.py` reuses a file's
  seed when the caller supplied none; the behaviour is now documented in
  [resuming](../../guide/results/resuming.md#noise-drawn-without-a-seed). The earlier
  probe remains evidence about the revision it tested. Independent builds used in
  recipe-equivalence tests still need controlled seeds; this does not remove R7's
  requirement for resolved provenance or settle every scan-builder detail.
- **R19, comparison policy: change the comparison class, not just its tolerance.**
  Section 7 asks for resumed versus uninterrupted agreement to rounding and bitwise
  agreement between independent runs with and without an observer. The current
  [testing policy](../testing.md#3-tolerance-policy) puts ordinary resumed versus
  uninterrupted runs in class T, with an accuracy envelope against a tighter or
  stored converged reference. A rounding comparison requires matched discrete
  methods and histories; independent propagation runs do not require bitwise
  agreement. Exact comparison remains appropriate for an observer's copy against
  the same plane read from its file, and for configuration identity metadata.

**Evidence and limits.** Read the designs, the cited manual pages, the driver and
processing implementations, and the existing interruption/output tests. Checked
Windows signal behaviour against Python's documentation. No simulations or test
suite were run as part of this review; the findings about runtime behaviour are
source-level assessments, not new qualification results.

### Response to review — 2026-10-31 (Opus 5.5, drafting agent)

Both reviews were read against `main` at `03807f8`, after the release preparation merged.
Two facts of that revision change the ground the first review stood on: the output schema
is now 1, and a file of any other schema is refused outright (ADR-058); and a resume
takes the seed recorded in the file. Every point is accepted in substance. The principal
changes to Part I are:

- the vocabulary is defined: `from_*` keywords, an `[execution]` table, and refused keys
  (R4);
- quantities convert exactly in decimal, with dimensions checked (R5);
- the recipe is stored as written and as resolved, with the hash it built, and needs no
  schema change (R1, R7);
- parameter paths and pins are part of format 1 (C2);
- the observer is lazy, specifically named, and cannot fail the run (R8, C5);
- three new items: result capabilities (S16), a portable stop with a resumability query
  (S17), and previews (S18) (C1, C4, R18);
- one branch and one pull request (R10).

The maintainer's decisions on the points remain to be recorded beside them.

1. **R1 — accepted, and stronger on the released code.** Since ADR-058 a file of another
   output schema is not read at all, so a bump would cost more than resumability.
   - *Changes:*
     - §5.2: the recipe is an optional group outside the hash, and needs no schema
       change. A reader that does not know the group ignores it. §7 tests a schema-1
       file with and without the group.
     - §5.14: the new keywords map onto settings already recorded.
     - §5.5, §5.6: S5 and S6 record through the input and `Alignment` records where
       they can. Otherwise their record changes are confined to phase 4 as one bump, a
       listed break (§9, new Q8).
     - §1 and §9 are corrected: the API is additive; files may see one break, and it is
       named.
   - *Not taken:* recording new fields only when they differ from the default. The output
     guide's rule that every field is recorded, at its default too, keeps record changes
     deliberate, and an exception for this SEP would weaken it.

2. **R2 — done in part, the rest is Q3.** The vocabulary split is gone. PR #64 renamed
   the route keywords `lambda0` and `pulseshape` to `wavelength` and `shape`, without
   aliases, before the 0.1.0 tag. The recipe example and §3 now use them.
   - *Changes:*
     - Q3 now asks for a decision on the `atol` and GNLSE `raman` defaults before this
       SEP merges, which is when format 1 is published. Under ADR-058 a 0.x minor
       release may change a default if the changelog lists it, so the deadline is that
       merge rather than the release tag.
     - S14 marks both rows as Q3.

3. **R3 — resolved on `main`; its remaining points accepted.**
   `Simulation._adopt_recorded_seed` takes the file's seed when the call gives none, so
   `solprop resume` of a default run is not refused.
   - *Changes:*
     - §4 defines equivalence with the seed fixed, and §7's equivalence tests fix it.
     - §5.2's resolved recipe records the seed.
     - §5.1's `Recipe.scan()` builder passes `point.seed`.

4. **R4 — accepted.**
   - *Changes:*
     - §5.1 defines the vocabulary: the build keywords are those of `Simulation.from_*`.
     - The execution settings `boundary_check`, `memory_budget`, `fields`,
       `field_planes` and `resume_history` go in an `[execution]` table.
     - `output`, `progress`, `progress_period`, `resume`, `snapshots` and
       `tdse_archive` are refused, by name.
     - `to_python()` writes a `prop_*` call when every key is on its surface, and
       otherwise `from_*`, the settings and `run`.
     - S9's sync test covers all three surfaces.
     - S14 lists `memory_budget`.
     - §3 states the three surfaces.

5. **R5 — accepted.** The three products were reproduced.
   - *Changes:* §4 states the decimal rule:
     - parse the mantissa, shift by the prefix, round once to binary64;
     - check the dimension, so `"125 um"` for `pressure` is refused;
     - write back with the prefix in which the shortest representation of the double is
       exact, so the round trip is the identity.

     S9 has a unit per element or per form for `phases`, `betas` and the `(z, p)`
     pressure, and §7 tests the example's values.

6. **R6 — accepted.**
   - *Changes:*
     - §5.1 writes `None` as `"none"`, and the metadata records which keywords accept
       it. A sync test checks that no keyword accepts both `None` and the string `"none"`
       as different values. Without that test the spelling would be ambiguous for
       choices such as `fields = "none"`.
     - §5.1 names what a recipe cannot hold (callables, arrays other than data files),
       and `from_call` refuses those by keyword.
     - Registry entries may be class methods (`PressureGradient.two_point`).
     - §7 tests every registered type in both directions.

7. **R7 — accepted.**
   - *Changes:* §5.2 stores, beside the authored recipe, a resolved one: every keyword,
     the defaults, the pins and the seed, with the hash the recipe built.
     - If the simulation was changed after its build, its hash differs. The driver then
       writes no recipe and warns. This is chosen over refusing the run, because changing
       a built simulation is a legitimate Python workflow.
     - `solprop resume` rebuilds from the resolved recipe, so a later solprop's defaults
       do not move the run, and it checks the stored hash.
     - `Recipe.from_output(path, resolved=...)` gives either recipe.

8. **R8 — accepted.**
   - *Changes:* §5.3 renames the protocol `PlaneObserver` and the call `observe_plane`.
     - `NullReporter` does not implement it, so a quiet run builds no view.
     - `PlaneView` is lazy. On a field-free plane on a device, asking for the field
       fetches one plane, and that transfer is counted in the resource summary.
     - §6 drops the claim that the plane is always on the host.
     - §7 tests neutrality on `numpy_ref`, `numba_cpu` and the native build, field-free
       planes included.

9. **R9 — accepted.** The counts were taken as given; one fact was checked:
   `DriverError(SolpropError, RuntimeError)`.
   - *Changes:*
     - §5.10 drops the reclassification, and no exception changes class.
     - `parameter` (a path) and `rule` are added to `SolpropError` and to the
       value-related warnings.
     - §5.8 gives rule ids only to the interface layer's combination refusals, enforced
       by a lint-style test. Value refusals are left to `check()`.
     - `check()` collects every metadata-level issue, then builds and translates,
       stopping before the backend is constructed. It reports the first refusal.
     - The 0.1 s translation cost is stated, with the advice to call `check()` on a pause
       in typing.
     - §10 records why there is no rule language for cross-field validation.

10. **R10 — accepted: one branch, one pull request.** This follows SEP 0.
    - *Changes:*
      - §11 makes the phases commits on one branch, with `Final` in the last.
      - A front end in development builds against the branch, and the phases are
        ordered so that its needs come first: the recipe, then resume and stop, then
        watching runs.
      - `prop_waveguide` moves to phase 4 with the other object-carrying routes, so
        phase 1's registry holds only the gnlse and capillary objects.
      - §10 records the split into two SEPs as an alternative. It was not taken,
        because the hooks name recipe paths, and format 1 should freeze once, with the
        pins and paths it must describe.

11. **R11 — accepted, with C3.**
    - *Changes:* §5.5:
      - labels resolve as `channel` resolves them, with the parity ambiguity refused;
      - `launch` takes one mapping, or one mapping per pulse, so a pump and a seed can
        go into different modes;
      - `launch` generalises `channel`, so giving both is refused, and it is exclusive
        with `incident`/`coupling`;
      - `with_input` remains the array path.

      The example uses the real display form `HE1,1`.

12. **R12 — accepted.**
    - *Changes:* §4 defines the tilt as `Alignment` does, with $k(\omega)$. That makes
      it a pure angle with no angular dispersion. Its validity is a band and edge
      condition, checked at build.
      - §5.6 renames the offset `displacement` to match `Alignment`.
      - It also states that `Alignment`'s `model_error` budget, which certifies a screen
        against a guide's modes, does not apply to a free-space launch.
      - The record follows R1.

13. **R13 — accepted.**
    - *Changes:* §5.7's `solve_parameter` reuses the sampled searches and the pole
      filter, and returns every root, or the attainable range when there is none.
      - The `rdw` target takes `peak_power`.
      - The capillary functions gain no `mode` keyword: `solve_parameter` is the general
        form.
      - §3 now says that `pressure_rdw` takes a `lattice` and builds a `VincettiMode`.
      - §7 tests the two roots of a resonance.

14. **R14 — accepted.**
    - *Changes:* §5.9 adds a resolver for each default that is `None` resolved by a rule.
      It returns the value and the rule's text.
      - S9 now covers every registered recipe type.
      - Q7 now recommends including the display unit.

15. **R15 — accepted.**
    - *Changes:*
      - §3 now states what `run_scan` already reports.
      - §5.4 says that a `ScanReporter` is accepted beside the literals.
      - It says a pool reports dispatch and collection itself, and that per-point
        progress needs a new pipe message carrying `z` and the `PlaneSummary`, never a
        field.

16. **R16 — accepted.**
    - *Changes:* §5.11 admits memory before anything is allocated. It excludes the warm-up
      from the timing and reports it separately, and it reports cost per right-hand side
      with the integrator's calls per step.

17. **R17 — accepted.**
    - *Changes:* §5.2 adds the received-file limits: the same checks as a loaded recipe,
      embedded files read into memory, never written at their stored names, budgets, and
      the HDF5 exposure.
      - §5.1 resolves symbolic links.
      - §7 and §8 carry the tests and the warning.

18. **R18 — accepted.**
    - *Changes:*
      - S18 (§5.18) adds `Recipe.modes()` and `Recipe.preview_input()`.
      - §5.10 puts `parameter` on warnings.
      - §5.3 adds `observe_boundary` for the monitor's verdict.
      - Cross-field rules are machine-evaluated by `check()`, not by S8 (§10).

19. **R19 — accepted, with the Codex update.**
    - *Changes:* §7 names the class of each comparison:
      - equivalence, export, round trips and quantities are class P;
      - resume against uninterrupted is class T, an accuracy envelope set per case;
      - the observer's copy against its file is class P;
      - with against without an observer is class R, on a fixed schedule.

      The loaders have stated bounds, and the observer and estimate run on the three
      backends.

20. **R20 — accepted.**
    - *Changes:* §3 is rewritten:
      - the object routes take a grid first and `frame`, and `prop_planar` takes no
        `medium`;
      - `SpatialPulse` starts with `temporal`;
      - `ResponseBudgetError` carries data;
      - the sidecar is already documented, so S15 moves the text;
      - the schema numbers are the released ones.

      The guide's stale schema numbers were fixed by the release preparation.

21. **C1 — accepted.**
    - *Changes:* S16 (§5.16) adds `solprop.capabilities.output`. It reports, per derived
      quantity, physical flux or coordinate quantity with units; field retention by
      plane; and the available and refused operations, with reasons. It extends the
      existing output capability machinery.
      - `PlaneView` carries the same description, and §5.16 says it does not make the
        physical operations available on every basis.

22. **C2 — accepted, in format 1.**
    - *Changes:* §5.1 defines canonical paths (`pulses[1].energy`). They are shared by
      `set`/`get`, issues, scans and pins.
      - A `[pins]` table states each pin's target and search range, and how a root is
        chosen.
      - Pins resolve after the literals, in dependency order; cycles are refused, and an
        unreachable target is reported with the attainable range.
      - Each scan point re-solves its pins.
      - The authored pin is kept in the authored recipe and the resolved value in the
        resolved one, and the Python export writes the solver call.
      - Pins are implemented in phase 5 with S7, but they are part of format 1, which is
        published only at the merge.

23. **C3 — accepted.**
    - *Changes:* §4 defines launch normalisation. On a flux-orthogonal basis, shares are
      energy fractions. On a general modal basis, they are relative amplitudes, scaled
      so that the flux-metric energy is the pulse's, and the split is reported as a
      coordinate one.
      - §5.5 applies this, and §7 tests both.

24. **C4 — accepted.**
    - *Changes:* S17 (§5.17) adds a `stop=` flag to `run` and `run_scan`. The flag is
      polled where SIGTERM takes effect, and raises `RunTerminated` through the same path,
      so the state is written.
      - `solprop run` maps signals, including Windows' break event, onto the flag.
      - `resumability(path)` says whether and how a file continues, and why not.
      - `RunOutcome.status` is unchanged, and a front end's states stay its own.
      - A changed grid, including a wider window, is a new run (§5.17, Q4).

25. **C5 — accepted.**
    - *Changes:* §5.3 states the policy. An observer exception is logged once, warned
      about, and the observer detached; the run continues, and terminal calls are still
      delivered. `RunTerminated` and `KeyboardInterrupt` pass through.
      - Observer time is reported in the resource summary.
      - The guide gives the pattern: reduce, a bounded queue that replaces undelivered
        previews, and a reliable channel for terminal events.
      - Transport and disconnection are the front end's.
      - §7 tests a failing observer, and a stop raised from one.

From the review of the front end's design (finding G9, not a point on this SEP): a
front end with a default of its own breaks the principle of §2. §5.1 now ships the guide's
example recipes in the package (`Recipe.example`), so a front end's starting points are
solprop's.

### Supplementary AI review — 2026-11-01 (Opus 5.5): the revised text

Reviewer: Opus 5.5 (AI), reading the revised Part I independently of the drafting session,
against the code at revision `4059dc6`, SEP 42's §4 and §5, and the front-end design's use
of the hooks. **Recommendation: revise §5.1, §5.2 and §5.7 before acceptance.** The split
of work and the choice of recipe keys equal to keywords hold up, and the quantity rule is
right. Two findings are blockers: the resolved recipe as defined does not rebuild its run
(G1), and the round trips rest on an object-to-table step that the classes cannot give
(G2). G3–G14 are significant and G15–G25 are minor. The G numbers are this review's own and
have nothing to do with the finding G9 of the front-end design's review, mentioned above.
Probes are listed at the end; every number below was measured on `numba_cpu` or
`numpy_ref` unless it says that it was read.

#### Findings

1. **G1 — blocker: the resolved recipe as defined does not rebuild its run (§5.2, §5.9).**
   §5.2 stores "every build keyword and execution setting at the value used", and
   `solprop resume` rebuilds from it. Rebuilding with values read back from a default build
   moved the hash in two of about twenty keywords, and others cannot be written at all:
   - `dz_init` (the value in `settings.dz_init`): the hash moves at
     `stepper.limits.auto_init`, because a default build uses the start-up rule and an
     explicit first step switches it off. So every default capillary or GNLSE run would
     restart differently. Under today's gate it would be refused.
   - `raman_fraction` on `from_gnlse(raman=True)`: the default resolves to the silica
     bundle's value (0.2188 here). Writing that recorded number back leaves the Kerr
     coefficient alone but moves the hash at `materials.policies.gnlse_raman.origin`,
     because the record says where the fraction came from. SEP 42's table has no row for
     that path, so it is compared there and the resume is refused under both gates.
   - The default `absorber` is an `Absorber` holding four arrays, built by
     `boundaries.absorber(grid)`. The default `cladding` is an optmat glass object. Neither
     has a table form.
   - `grid`, `wavelength_limits` and `trange` are alternatives, as are `fwhm` and `width`,
     and `n_save` and `save_planes`, so "every keyword" cannot be meant literally.
   - `settings.backend` holds the resolved name (`driver/simulation.py:927`). The hash
     leaves the backend out so that a run can move between machines; a resolved `backend`
     would take that away.

   These did keep the hash: `rtol`, `atol`, `dz_max`, `integrator`, `stats`, `temperature`,
   `plasma`, `thg`, `raman`, `step_planes`, `save_planes`, `loss`, `model`, `pressure=None`
   against `1.0`.

   **Recommendation:** define the resolved recipe positively: the authored keys, the pins'
   solved values, the seed, and a declared list of keywords whose S9 resolver is tested to
   give the same hash when written explicitly. The rest stay "left to the build", and the
   resume gate is what catches a moved default. Say that "a later solprop's defaults do not
   move the run" holds for the listed keywords only. Resolve registered objects recursively,
   since a pulse's or stepper's own defaults can move too. Keep `backend` as requested. Add
   a sync test: each resolver's value, written back, keeps the hash.

2. **G2 — blocker: there is no object-to-table step, and the registry rule cannot express
   the manual's own setups (§1, §5.1, §7).** `Recipe.from_call` and two §7 tests ("every
   registered type goes table → object → table and back unchanged", and equivalence through
   `from_call` "for every example") need a table from a built object.
   - Objects do not keep what made them. `PressureGradient.two_point`, §5.1's own example,
     holds `positions`, `pressures`, `floor` and `_bore` (`profiles.py:1797`), and it comes
     back as a `PressureGradient`, not as the class method. `TubeLattice.from_gap` holds
     `tube_radius` and no gap. `EnvelopeGrid.from_limits`, which every object route needs,
     holds `t`, `omega`, `t_fine`, `omega_fine` and collars, not its limits.
   - The manual's radial setup uses module-level functions, with `absorber=absorber(grid)`
     (`docs/guide/free-space/beams.md:112`) inside the settings. An entry is "a constructor
     or a named class method", so `boundaries.absorber` (`boundaries.py:533`) is not
     admitted, `Absorber`'s constructor wants four arrays, and no table can refer to the
     sibling `grid`. `Mixture.of(*components)` has no keywords to write. `load_spectrum`,
     §5.12's route to a `DataPulse`, is a function as well.
   - "Every example" cannot hold as written. The gallery scripts call `with_input` in at
     least four places, and examples pass `progress="none"` or `output=`, which §5.1
     refuses. `from_call` has to drop those keys, not refuse them.

   **Recommendation:** make an entry a named factory (constructor, class method or function)
   with declared keywords and a declared build context it receives, such as `grid`, and
   define that context in format 1. Let `from_call` invert only the types shown to be
   lossless, and refuse the others by keyword, or have a factory attach the table it was
   built from. Rewrite the §7 tests as table → object → hash and table → text → table, and
   list the examples excluded, with why. Decide this now. Phase 4 would otherwise find it
   after phases 1–3 have been built on a registry that cannot grow this way.

3. **G3 — significant: format 1 has no spelling for values the keywords take (§5.1).**
   I classified all 41 keywords of `from_gnlse` and all 61 of `from_capillary` from their
   annotations. Apart from the callables refused by design and the G2 objects, three gaps
   remain:
   - **Complex numbers.** `cladding` is `GlassIndex | complex | None`, and
     `SpatialPulse.jones` is a tuple of complex numbers, recorded as `((1+0j),)` by default.
     `tomllib` rejects `1.45+0j`, and §5.1 defines no table or string for it.
   - **The seed.** A drawn `noise.seed` is about 128 bits (127 in the draw measured), and
     the resolved recipe must hold it. TOML v1.0.0 specifies 64-bit signed integers and an
     error beyond them. `tomllib` happens to read the larger value, but another TOML reader
     need not, so every unseeded run's stored recipe would fall outside the standard. A
     scan's derived seed is `uint64` (`driver/_scan_model.py`), half of them above $2^{63}$.
   - **Arrays.** "Arrays other than data files" are refused, yet `betas`, `phases`,
     `save_planes`, `StatsSpec.energy_windows` and `FixedSchedule.steps` are typed
     `Sequence | ndarray`, and `betas` has a per-element unit. No layout is defined for the
     files of the `beta` tuple form or an `IndexTable`; S12 covers spectra only.

   **Recommendation:** add a complex spelling (`{re = , im = }`, say) and a decimal-string
   form for seeds to format 1. Say that a one-dimensional numeric array is written as a list
   up to a stated length and as a data file beyond it, and define each file layout, or defer
   those types by name.

4. **G4 — significant: table names collide with keywords, and the path grammar is too small
   (§5.1, §5.9).**
   - `grid` is a keyword of all six routes and `medium` of two, and §5.1's example has
     `[grid]` and `[medium]` as group tables. `[grid]` with `type = ...` beneath it could be
     either. `output` is an S9 group and a refused key. A plain mapping such as
     `[plasma]` with `He = true` looks like a group with an unknown key. The text does not
     say whether unknown group names or top-level keywords are allowed.
   - `.name` and `[i]` cannot address a mapping key that is not an identifier, and `launch`
     keys contain a comma and a colon (`HE1,1:even`, measured on the capillary with
     `modes=3`). In `[pins]` and `[scan]` a path has to be a quoted TOML key, which does
     parse (`"pulses[1].energy" = ...`), but the text does not say so.
   - "A pinned path may not also be set literally" leaves no way to pin `pressure[1]` and
     set `pressure[0]`, because the array needs something in the pinned place.

   **Recommendation:** rename the groups so that none is a keyword, require every top-level
   table to be a group or a reserved table, and refuse anything else. Add a quoted segment
   (`launch["HE1,1:even"].share`) to the grammar, and a placeholder for a pinned element or
   a rule that only whole keywords are pinned.

5. **G5 — significant: the `[execution]` table has four concrete defects (§3, §5.1, §5.14).**
   - `boundary_check` is a keyword of all six `from_*` constructors (measured), and
     `memory_budget` of `from_waveguide`, so both are build keywords and execution settings
     by §5.1's definitions. §3's statement that `from_*` lacks `boundary_check` is wrong.
     `prop_radial`, `prop_free` and `prop_planar` also take `absorber` and `absorber_scheme`,
     which their `from_*` do not and which fit none of §5.1's three lists.
   - The §5.1 example does not build as written. `fields = "planes"` alone is refused:
     "field_planes is required exactly when fields='planes'" (`output/schema.py:3075`).
   - `memory_budget` is "passed to `run`", but §5.1's surface is `recipe.build()` followed by
     `simulation.run("rdw.h5")`. Nothing carries it across, and settings has no such field.
     `from_waveguide`'s budget is another gate: it admits the guide's tables at build and is
     refused without `provider` (`driver/simulation.py:4940`), so S14's "the same budget" is
     true of the unit only.
   - `Recipe.scan()`'s builder cannot apply `[execution] fields`. With `fields` set in a
     builder, the point's status was `failed` (`driver/_scan_run.py:404`); `run_scan` takes
     `fields=` itself, and `Recipe.scan()` returns only the `Scan` and the builder.
     Separately, `Scan` requires a `name`, which also sets durable filenames and the
     derived seeds (`name_seed`), and `base_seed` is optional. `[scan]` has room for neither.

   **Recommendation:** take `boundary_check` as a build keyword. Fix the example with
   `field_planes`. Have the `Simulation` hold the recipe's `memory_budget` for its `run`, or
   add `Recipe.run`. Have `Recipe.scan()` return the `run_scan` keywords too, and reserve
   `name` and `base_seed` in `[scan]`, with `name` required.

6. **G6 — significant: `"none"` collides on a listed execution setting (§5.1).** For
   `memory_budget`, `None` means the environment or the machine's default and the string
   `"none"` means no check (`driver/memory.py:231`, `driver/simulation.py:2900`). So the test
   that "no keyword accepts both" fails on the table's own key, and a loader that maps
   `"none"` to `None` would silently turn a recipe's "no check" into a default budget.
   **Recommendation:** spell `None` only where the default is not `None`, which on
   `from_gnlse` and `from_capillary` is exactly `noise` and `boundary_check` by signature.
   Elsewhere leaving the key out already says `None`. State the test on that set.

7. **G7 — significant: the "changed after its build" rule errs both ways, and the group's
   lifetime is undefined (§5.2, §6).**
   - **False alarm.** `_adopt_recorded_seed` redraws an unseeded realisation on a resume.
     Measured: after it, the hash equals the file's and differs from the build's. So
     `solprop run x.toml --resume` on an unseeded recipe meets "writes no recipe group and
     warns". The same happens to a scan builder that leaves the seed to `_reseed`
     (`driver/_scan_run.py:401`).
   - **Missed change.** Replacing `fields`, `resume_history`, `backend` or `boundary_check`
     leaves the hash the same (measured; `_EXECUTION_SETTINGS`). A file can hold a resolved
     `[execution]` that is not what ran, and a resume through it uses the wrong retention.
   - **Cost.** Comparing needs a hash at build, which is a translation, so §6's "building a
     recipe adds milliseconds" no longer holds. R9 measured 0.13 s on the capillary, and a
     guide route builds its tables there.
   - **Lifetime.** "At run start" does not say what a resume does. SEP 42 §5.8 allows a
     top-level group "written once" and keeps the stored record unrewritten, and under it
     `solprop run edited.toml --resume` may continue with a changed `rtol`. A later
     `solprop resume` then returns to the first session's numerics.

   **Recommendation:** drop the comparison. The simulation carries its recipe, and each
   mutator (`with_input`, `with_mixture`, a settings change outside the execution list) drops
   it and warns. Write the execution section from the settings at run start. Write the group
   once, never rewrite it, and let SEP 42's execution records say how later sessions differed.

8. **G8 — significant: S5's record conflicts with SEP 42, and S6's has no place to go (§1,
   §5.5, §5.6, §6, §9).** S5 records "through the input record that exists". On the capillary
   that record is `configuration.initial` beside `configuration.pulse` (both measured, with
   `initial` shaped `(3, 1, 385)` for `modes=3`). SEP 42 §5.2 makes `configuration.initial`
   **ignored** where `configuration.pulse` is present. So two recipes with the same pulse and
   different `launch` would continue each other's files with nothing said, and the shares
   would be readable from nowhere. In S6, the `Alignment` record belongs to a coupling step,
   and a free-space launch has none, so the fallback is empty; the fields would go in
   `configuration.pulse`. Q8's decision already makes additive fields acceptable.
   **Recommendation:** state plainly that S5 adds a `launch` record (`configuration["launch"]`
   already holds an incident report, `driver/simulation.py:2393`, so it needs its own name),
   that S6 adds fields to `SpatialPulse`, and that each has an SEP 42 neutral entry and a
   compared class. Remove the conditional schema text from §1, §6, §9 and the phase 4 report.

9. **G9 — significant: `solprop resume` and `resumability` do not cover what exists or what
   SEP 42 adds (§5.2, §5.17).**
   - §5.2 continues through `resume=True`. A dense run stopped mid-way was refused there
     (`DriverError`: "ends on a plane interpolated inside a step") and continued under
     `resume="interpolated"` (measured). The command has no such option, though §5.17 and §7
     name the case.
   - SEP 42 adds `resume="force"`, a refusal of a file with a forced session, and "work left"
     judged by position and not status (its §4, §5.5). §5.17's reasons have none of these.
   - The gate needs a requested record, and `resumability(path)` has only the file, so a
     moved optmat digest cannot be seen without a rebuild.
   - "No recipe and so no rebuild from the file alone" is listed as a reason for "not at
     all", yet such a file resumes from its script.
   - A file whose writer is still alive, and a recipe of a newer format, are not listed.

   **Recommendation:** separate three answers: can the file structurally continue, and from
   which `z`; can it be rebuilt from itself; and, on request and at the cost of a rebuild,
   does the gate pass. Give `solprop resume` and `resume_file` a mode. Add the missing reasons.

10. **G10 — significant: the stop's description and its scan form do not match the code
    (§5.17, §6).**
    - §5.17 polls "where SIGTERM takes effect today, between commits". SIGTERM raises at once
      at whatever line is reached and is held only during a commit (`_interrupts.py:10-15`).
      A poll at commit exit runs once per accepted step, so it does not answer during
      rejected trials, `translate`, backend selection or atomic preparation.
    - `_GUARD` is process-global and `committing` has no per-run handle, so a flag polled
      there is not per run (read, not tried with two runs).
    - `run_scan(stop=)`: a `threading.Event` cannot cross to a `Pool` worker, and `Pool._stop`
      ends with `process.terminate()` (`driver/executors/pool.py:430`), the abrupt exit on
      Windows that §5.17 was written to avoid. Batch, Queue and Slurm are not mentioned. The
      point loop catches `Exception` only (`driver/_scan_run.py:627`), so a `RunTerminated`
      leaves it. What happens to the point in flight is not stated.
    - `RunTerminated` needs a signal number and reads "terminated by SIGTERM", code 143
      (`exceptions.py:64`). It is a `SystemExit`, which a front end's `except Exception`
      misses.

    The core does work. A prototype that set a pending `RunTerminated` from another thread,
    with the run on a non-main thread, stopped in 34 ms, and the file resumed to completion.
    **Recommendation:** add polls at the run's start and per trial, and state the latency:
    one step or trial. Make the flag run-scoped. Give the stop reason its own text. Specify
    `stop` per executor: Local polls, Pool sends a pipe message, Queue and Slurm stop between
    points or refuse.

11. **G11 — significant: `solve_parameter` and `[pins]` are not implementable as written
    (§3, §5.1, §5.7, §7).**
    - `("rdw", pump, rdw, peak_power=None)` cannot reproduce `pressure_rdw`:
      `pressure_rdw(..., peak_power=2e9)` raises "the largest constituent needs fwhm"
      (measured; `tools.py:1954`). The target also needs `fwhm`, `shape` and
      `nonlinear_phase`. The §7 test as written would fail.
    - The nonlinear targets need the fill's Kerr coefficient at each trial value, and "a mode
      need not know its fill's Kerr response" (`tools.py:1231`). So `build_mode(value) ->
      mode` is not enough for `soliton_order`, or for `rdw` on a tabulated mode. The role of
      the `parameter` argument is not stated.
    - §3 says the zero-dispersion search samples. `pressure_zdw` brackets once with `brentq`
      and refuses equal signs (`tools.py:2369`); the sampling is over wavelength, inside the
      modes' ZDW search. A sampled ZDW search over a parameter is new work, not reuse, and the
      "equals `pressure_zdw`" test holds only for a single root.
    - The `rdw` and `soliton_order` pin tables are not given, nor which mode a pin solves on,
      nor which $z$ on a pressure gradient. No syntax lets a target read a path. If the
      implicit read set is the mode's inputs, any two pins on mode parameters read each other,
      so a cycle is always refused and the dependency order is never used. That part is a
      judgement: over-built, or a read set is missing.
    - The front-end design asks for `solve_pin(recipe, pin)` with its roots, and lists a pin
      giving pressure and energy from `capillary_design`. §5.1 has no call that returns a
      solved value or the roots, and no two-output pin. Without such a call a front end has
      to write its own `build_mode`, which is the physics the principle of §2 keeps out of it.
    - §5.10 counts pins as "metadata level" and §6 says `check()` costs a translation.
      Measured on the Marcatili capillary on a busy machine: `pressure_zdw` 0.47 s,
      `pressure_rdw` 0.14 s linear and 0.40 s with the nonlinear phase, first-call costs
      included. A non-Marcatili builder pays up to 258 mode builds.

    **Recommendation:** have the builder return a mode and its fill, or take a `gamma(value)`.
    Carry `fwhm`, `shape` and `nonlinear_phase` in the target. Define each pin table, the mode
    and the $z$. Add `Recipe.resolve()` and `Recipe.pin_roots(path)`. Either define read sets
    or drop dependency order from format 1. Count pins in `check()`'s cost.

12. **G12 — significant: "node and memory budgets apply" is not a limit on a received file
    (§5.2, §7).** With `memory_budget=None` the machine's four fifths "only warns"
    (`driver/simulation.py:2903`). A recipe can itself say `memory_budget = "none"`, and
    `node_budget` is a build keyword it can raise. `from_gnlse` has no `node_budget`. So a
    received file chooses its own limits. Also, §5.2 reads embedded files into memory, but
    `Recipe.from_output(...).save(...)` then holds a `file =` with nothing behind it, and
    embedding has no size limit ("usually small").
    **Recommendation:** in `resume_file` and `Recipe.from_output`, ignore the recipe's own
    budgets and apply the caller's, refusing by default. Say what saving such a recipe does
    with its data. Cap what is embedded, and record a hash and a path beyond the cap.

13. **G13 — significant: the phases have dependencies the plan does not state (§11, header).**
    Each phase can otherwise reach a green gate, as far as I can tell from reading it.
    - Phase 2 verifies embedded data files, but the only way to load one, `load_spectrum`,
      arrives in phase 5, and `{type = "DataPulse", file = ...}` is not `DataPulse`'s
      constructor. Move S12 before phase 2, with its entry in the registry.
    - The header says **Requires** —, yet S2 and Q8 rest on SEP 42. §9 and Q4 call it "in
      draft" and it is now Accepted.
    - S12 needs the `DataPulse` fix, which is not on this revision (`pulses.py:2102`,
      `pulses.py:2182`). SEP 42 §9 says it takes the output schema to 2. So §7's "schema-1
      file" is stale, and §9's "no computed field changes" and §6's "unchanged by ... S7–S18"
      need that exception.
    - `Recipe.scan()` (phase 3), `Recipe.modes()` and `preview_input()` (phase 4) all read
      pinned values, and pins come in phase 5. Phase 5 should list those retrofits.

14. **G14 — significant: §7 leaves new behaviour unverified.**
    - S6 has no entry: the band and edge refusals, energy after a tilt, the agreement with
      `Alignment`.
    - S16 and S18 have none. `preview_input` should equal the run's initial field, and each
      S16 refusal should match the reader's actual refusal, or the rules drift.
    - `load_spectrum`'s phase rules and `phase_convention` have none; "Loaders" tests energy
      and centre wavelength only.
    - Equal hashes cannot catch a `to_python()` that drops `fields`, `boundary_check`,
      `memory_budget` or `backend`, none of which is hashed, so §1's "gives the same run" is
      not what is tested.

    **Recommendation:** add these, and compare the execution settings in the export test.

15. **G15 — minor: the entrance plane is not defined for the observer (§5.3, §7).** The
    entrance is saved without a `plane()` call (`driver/run.py:485-505`, against `:580`).
    Measured: at the second `plane()` the file held three rows. So "once per save plane ...
    before `plane()`" and "each plane" disagree there. **Recommendation:** say that the
    entrance is observed. Say also that a resume does not re-deliver earlier planes, which a
    front end reads from the file.

16. **G16 — minor: `observe_boundary` is undefined in its details (§5.3).** The monitor has no
    verdict: it warns once or stops (`monitor.py:706`), and the word appears only for the
    collar's quiet flag (`backends/driver.py:1990`). It runs inside `propagate_to`
    (`backends/driver.py:1974`), so a call there falls outside "after the plane is committed",
    and the failure policy names only `observe_plane`. A `runtime_checkable` protocol checks
    every member, so the optional method needs a protocol of its own. **Recommendation:**
    define the verdict's type, the call site and the failure policy.

17. **G17 — minor: S6 leaves the order and some routes open (§4, §5.6).**
    - `SpatialPulse.build` applies `propz` spectrally (`pulses.py:296`), and whether a tilt
      and offset apply at the waist or the launch plane differs by $\mathrm{propz}\tan\theta$.
    - A $k(\omega)$ screen makes the field non-separable, while `build` multiplies one
      transverse spectrum by the temporal one, so the normalisation has to change.
    - "A beam's displacement and an `Alignment` add" fails on `prop_waveguide(incident=...)`,
      which samples the beam on a radial grid (`driver/simulation.py:9985`), where S6 refuses
      any nonzero value.
    - A `LineGrid` has no $y$.
    - A tilted pulse front is delayed by about $x\sin\theta\,n/c$ across the beam, which
      belongs among the validity conditions for the time window.

18. **G18 — minor: `load_spectrum`'s defaults and heuristic (§5.12).**
    - `axis_unit="Hz"` stays the default with `axis="wavelength"`; it should follow `axis`.
    - `columns=(0, 1, 2)` names a column that "is optional".
    - The "small fraction" is not a keyword, though it changes what a recipe builds.
    - The wrapped test has no override, and a wrapped column in degrees spans 360 and is taken
      as unwrapped radians with nothing said. A wrapped column undersampled past $\pi$ per
      sample cannot be detected, and it deserves a warning: at 1.6π per sample the recovered
      slope was −0.25 of the true one (measured through today's `from_spectral_density`).
    - The hold rule applies to wrapped columns only, so an unwrapped column's noisy floor still
      feeds the spline. That is a judgement.

    The sign rule itself is right (see below).

19. **G19 — minor: S9's `kind` cannot describe the keywords (§5.9).** It is singular and has no
    mapping, while `pressure` has four forms, `plasma` is bool, object or per-species map,
    `loss` and `modes` are unions, and `gas`, `raman` and `launch` are maps. A front end needs
    these. **Recommendation:** add a list of forms, each with its kind and units.

20. **G20 — minor: S14's "`noise`, `seed`: add to planar" is contradicted by the code (§5.14).**
    "A planar grid carries no noise draw" (`docs/guide/media/noise.md:255`), for the reason in
    `PLANAR_NOISE` (`driver/simulation.py:484`). Adding it means declaring a photon measure,
    which §2 rules out. **Recommendation:** list it as a documented difference.

21. **G21 — minor: S8 and S10 contradict each other (§5.8, §5.10, §7).** `parameter` is set
    "wherever the interface layer refuses a value", value refusals "are not rules", and the
    lint requires every refusal that sets `parameter` to name a rule. **Recommendation:** lint
    combination refusals only, by a marker on the raise.

22. **G22 — minor: S18's previews are underspecified (§5.18).** `beta` and the rest depend on
    $z$ on a gradient (`_zdw_near(mode, omega, z)`), and no $z$ is given. On a guide route the
    dense tables are built at construction, so "the mode solve only" is the large cost, and it
    needs the memory admission `from_waveguide` makes. The text also does not say whether
    `preview_input` includes noise, and an unseeded preview would otherwise draw.

23. **G23 — minor: "as written" and the writer (§4, §5.2, §6, §7, Q7).** §5.2's `authored` is
    "canonical ... as written" and §6's is "stored verbatim", and these differ: a canonical
    writer turns `"0.8 um"` into `"800 nm"` and drops comments. §7's "equal recipe" needs a
    meaning. Q7's display unit says µm and the format's strings say `um`. **Recommendation:**
    keep the loaded text verbatim, compare recipes by value, and fix one ASCII spelling for
    writing while reading both. That saving a hand-written recipe loses its comments is a
    judgement for the maintainer.

24. **G24 — minor: a scan's file holds no recipe (§2, §5.2).** The collected file has `common`,
    `executions`, `layouts`, `points`, `scan` and `staged` (measured), and S2 speaks of a run.
    §2's "reopen a results file with the setup that made it" therefore stops short of a
    `Recipe.scan()`. **Recommendation:** store the authored recipe in the scan file, or say that
    it is not stored.

25. **G25 — minor, a judgement: the registry's size is public surface (§5.1, §9).** §9 weighs
    renaming route keywords only, but every keyword of every registered type also becomes part
    of format 1, including `TDSEBasis` (14 keywords), `HaccBasis`, and `Multistep`'s
    `equal_approach` and `max_growth`. **Recommendation:** register the documented user-facing
    types first, and let `from_call` refuse the rest towards the Python export. Adding an entry
    later is additive; removing one is not.

#### Verified and found correct

- §4's decimal rule: the example's values and others convert to the script's literal, where
  naive products fail (`800`, `7.5`, `275`, `100`). 200,000 random doubles written in a chosen
  prefix and read back had no failures. The default decimal context rounds at 28 digits, so an
  implementation should use a local context.
- The §5.1 capillary example builds, apart from `fields` (G5).
- A rebuild with an explicit copy of a drawn seed has the file's hash, and an unseeded rebuild
  adopts it. `wavelength_limits` as a list, `pressure=1` against `1.0`, and `pressure=None`
  against `1.0` do not move the hash.
- An unknown top-level `recipe` group in a written file reads, resumes and survives, which
  agrees with SEP 42 §5.8.
- `RunTerminated` raised from a reporter after a commit leaves a file that `resume=True`
  finishes.
- §5.12: a phase of $+\omega\tau$ overlaps a pulse with `t0=+tau` and not `-tau`, so the sign
  statement is right. Today's `from_spectral_density` does lose a phase step above $\pi$. The
  three-column layout agrees with the other code's documentation; I did not check its sign.
- §4's tilt matches `Alignment`'s definition (read). The capillary's labels display as
  `HE1,1:even`.

#### Not checked

- `tomli-w`: it is not in the shared environment, so its writing of large integers and nested
  tables is untested.
- The native and device backends. `PlaneView`'s neutrality and device fetch were not
  prototyped.
- The guide and provider registry of phase 4, beyond the signatures, and `solve_parameter` on
  anything but the existing Marcatili tools.
- S4's pipe message, S11, S13, Windows, and Queue or Slurm behaviour, which was read only.
- SEP 42 has no code yet, so every statement about it comes from its text.

No test suite and no long run were performed. Capillaries were built and not run; the runs
were GNLSE fibres of at most 30 cm with a handful of saved planes, and a one-point scan.

### Response to the supplementary review — 2026-11-01 (Opus 5.5, drafting agent)

G1 was re-run before this response (`dz_init` moves `stepper.limits.auto_init`, and
`raman_fraction` moves `materials.policies.gnlse_raman.origin`), and the rest of the numbers
quoted are the reviewer's. The revised format was tried with `tomli-w` 1.2.0 and `tomllib`,
including nested tables and quoted path keys, and the §5.1 example's values build on the
capillary. The maintainer gave leave to reopen decisions and approved text for a better and
more consistent result, and Q9–Q12 list what that reopened. All 25 points are taken, with
the departures stated in points 2, 4, 7, 10, 11 and 16.

1. **G1 taken.** The resolved recipe is defined by what rebuilds the run (§5.2), with a
   resolvable flag in the metadata and a sync test. `backend` stays as requested. A keyword
   that cannot qualify stays left to the build, and the resume gate notices a moved default.
2. **G2 taken, more simply than recommended.** Entries are named factories with a declared
   build context, which format 1 limits to the route's `grid`. The inverse is dropped
   altogether: no object becomes a table, so `from_call` gives way to `from_keywords` with
   plain values (Q11). The §7 tests run table → object → hash, with hand-written recipes,
   and the examples without a recipe are listed and checked.
3. **G3 taken.** Complex numbers, decimal-string integers, and lists up to a stated length
   with data files beyond (§5.1). `IndexTable` and the `beta` tuple form are deferred by
   name.
4. **G4 taken, by a different route.** Rather than renaming groups, the format drops them:
   one `[build]` table and four other fixed tables (Q9), so no future keyword can collide.
   This reopens the example the maintainer approved. The path grammar gains `["key"]`, and
   only whole values are pinned.
5. **G5 taken.** `boundary_check` is a build keyword, §3 is corrected, the example has
   `field_planes`, `Recipe.run` carries `memory_budget`, and `Recipe.scan()` returns the
   `run_scan` keywords, with `name` required and `base_seed` reserved. `absorber` on the
   object routes goes under `settings`.
6. **G6 taken.** `"none"` is spelled only where the default is not `None`.
7. **G7 taken, with one addition.** The simulation carries its recipe and mutators drop it;
   the group is written once and never rewritten. Execution settings come from the run
   start. The built hash is no longer stored, since it would only repeat the file's own
   stamp.
8. **G8 taken.** S5 adds `configuration.modal_launch`, S6 adds two fields to the pulse
   record, each compared with a neutral entry, and the conditional schema text is gone. This
   also settles Q8.
9. **G9 taken.** `resumability` answers three questions, `gate=True` costs a rebuild, and
   `solprop resume` and `resume_file` take a mode (§5.1, §5.2, §5.17).
10. **G10 taken, with a further change.** The stop polls before the first step and per
    trial, is per run, and is defined per executor. Beyond the point: a run stopped by its
    own flag returns rather than raising (Q10), because a `SystemExit` ends a worker thread
    silently and misses `except Exception`, and the reviewer's own text-only remedy would
    leave that.
11. **G11 taken, more simply than recommended.** Targets are typed and carry what they need,
    and `build` may return the Kerr coefficient. `Recipe.resolve` and `pin_roots` are added.
    Instead of read sets, pins resolve in two fixed stages with at most one mode-parameter
    pin, so cycles cannot be written. `capillary_design` is not a pin: it sets two values,
    and a front end writes them as literals. §3's account of the search is corrected, and a
    sampled zero-dispersion search is stated as new work.
12. **G12 taken.** `resume_file` applies the caller's budgets and refuses by default;
    embedding has a cap; `check` reports a file left dangling after saving (§5.2).
13. **G13 taken.** S12 moves to phase 1, **Requires** names SEP 42, §7's stale schema text
    goes, and phase 5 lists its retrofits. The `DataPulse` fix lands separately
    ([LupoLab/solprop#70](https://github.com/LupoLab/solprop/pull/70)), and §5.12 only
    relies on it.
14. **G14 taken.** §7 gains tests for S6, S16, S18, the phase rules and the execution
    settings.
15. **G15 taken.** The entrance is observed, and a resume does not repeat planes.
16. **G16 taken by removal.** `observe_boundary` goes: the monitor warns once or stops, and
    a front end has both, so a hook would have no verdict to carry. The front-end design's
    lamp reads those instead.
17. **G17 taken.** Tilt and displacement apply at the launch plane, after `propz`; the
    energy is set after the screen; `incident` and the line grid's `y` are refused; the
    pulse-front delay joins the validity conditions.
18. **G18 taken.** `axis_unit` follows the axis, a two-column file is allowed by default,
    and `wrapped`, `floor` and `phase_unit` are keywords, because they change what a recipe
    builds. Undersampling warns. The hold rule stays for wrapped columns only.
19. **G19 taken.** The metadata has a list of forms.
20. **G20 taken.** Planar noise is a documented difference.
21. **G21 taken.** The lint covers combination refusals, marked at the raise.
22. **G22 taken.** `modes` takes `z`, its cost and admission on a guide route are stated,
    and `preview_input` draws no noise.
23. **G23 taken.** The text is kept verbatim until a recipe is edited, recipes compare by
    value, and writing uses ASCII while both spellings read. Losing comments on an edit is
    Q12.
24. **G24 taken.** A scan's file stores the authored recipe.
25. **G25 taken.** The registry starts with the documented user-facing types, and §9 says
    why.

**Consistency.** The front-end design was revised to match: recipe sketch, pins, the
boundary lamp, and the service table.

**Still unmeasured,** as the review says: the native and device backends, and `PlaneView` on
them; the guide and provider registry of phase 4; `solve_parameter` beyond the Marcatili
tools; Windows; Queue and Slurm. Nobody independent has read this revision.
