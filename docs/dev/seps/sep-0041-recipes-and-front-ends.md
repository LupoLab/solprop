# SEP 41 — Recipes, and the hooks a front end needs

| | |
|---|---|
| **SEP** | 41 |
| **Title** | Recipes, and the hooks a front end needs |
| **Author** | John C. Travers |
| **Drafting agent** | Claude Opus 5.5 (2026-10-31) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | Opus 5.5 (AI), Codex (AI) |
| **Type** | Change |
| **Created** | 2026-10-31 |
| **Resolution** | — |
| **Requires** | — |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

solprop is driven only from Python today. A complete simulation is a call to one of six
`prop_*` functions, and nothing in solprop can write that call down, read it back, or
explain its parameters to anything other than a person reading the docstrings. This SEP
adds a **recipe**: a versioned TOML description of a simulation that solprop reads,
validates, builds, runs, exports as a plain Python script, records in its output and
resumes from. Around it come the smaller facilities that any front end needs from solprop
instead of copying from it:

- parameter metadata, structured validation and a capability query;
- a supported observer of saved planes, scan progress, a cost estimate and a portable stop;
- launch into chosen modes, beam displacement and tilt;
- design solvers for every guided route, and pinned parameters that use them;
- previews of the modes and the input, a statement of what a results file can give;
- measured-pulse loaders and geometry drawings.

The first user is a desktop front end being designed separately; scripts and batch systems
gain the same things (`solprop run recipe.toml`). Success is judged by round trips, with
the noise seed fixed: a recipe and the Python call it stands for build the same
`Simulation` (equal `configuration_hash`); a recipe exported to Python and run gives the
same run; and a file written from a recipe resumes from the file alone.

The **Python interface changes are additive**: every name is new and every new keyword
defaults to today's behaviour. The recipe is stored without an output-schema change. The
one possible break is in files: if the launch or beam changes (S5, S6) cannot be recorded
through records that exist, phase 4 bumps the output schema once, which ADR-058 permits
in a 0.x minor release and the changelog lists (section 9, Q8).

### 2. Motivation and scope

A front end must:

- show every parameter with its unit, default, range and documentation;
- check a setup as it is typed and put each error next to its field;
- offer only the options a route and species support;
- watch a run as it goes, and stop it;
- save a setup and reopen it, and reopen a results file with the setup that made it;
- say whether a stopped run can be resumed, and resume it;
- say what each results file can show.

Today each of these would mean a hand-written copy of rules that live in solprop's code,
and the copy would drift with every release. The principle of this SEP is that **solprop
holds the physics, the defaults and the rules; a front end holds only presentation**.

The same gaps hurt script users. A results file does not record the call that made it, so a
run can be resumed only by the script that started it (the [resume key](../../guide/results/resuming.md)
checks a rebuilt simulation; it cannot rebuild one). A batch job is a Python file per
configuration. A scan's builder must be a module-level function for pool executors
([scans](../../guide/workflows/scans.md)), which a declarative description gives for free.

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
- remote execution and a network service (a recipe and `solprop run` are how remote work
  is done for now, through the existing executors);
- arbitrary Python inside a recipe (custom pulse shapes or responses, callables): the way
  out is the Python export;
- a graphical recipe editor in solprop;
- extending a finished or stopped run onto a changed grid (Q4);
- any change to the physics, the kernel contract or the steppers.

### 3. Context

- **Entry points.** [`prop_gnlse`, `prop_capillary` and `prop_waveguide`](../../guide/setup/running.md)
  take keywords. `prop_radial`, `prop_free` and `prop_planar` take a `TimeGrid` first, then
  objects: `space`, `responses`, `initial` and the keyword `settings`, all three with an
  optional `frame`; `prop_radial` and `prop_free` also take a `medium`. Each has a
  `Simulation.from_*` constructor, so a simulation can be built without running it, and
  `Simulation.run(output, progress=..., resume=..., memory_budget=..., tdse_archive=...,
  snapshots=...)` runs it. The keyword surfaces differ: the `prop_*` functions also take
  `output`, `progress`, `progress_period`, `resume`, `snapshots`, `boundary_check` and, on
  the capillary, `tdse_archive`, which `from_*` does not; `from_waveguide` takes
  `memory_budget`, which no `prop_*` does; and `fields`, `field_planes` and
  `resume_history` are on no keyword surface, only on `Simulation.settings`.
- **Names.** The route keywords name the reference wavelength `wavelength` and the pulse
  shape `shape`, as the pulse objects and `solprop.tools` do (since the 0.1.0 release
  preparation, PR #64).
- **Progress.** [`ProgressReporter`](../../guide/performance/progress.md) is a
  `runtime_checkable` protocol with four calls (`started`, `step`, `plane`, `finished`),
  extended by optional protocols (`InterruptibleReporter.interrupted`,
  `SummarisingReporter.summary`) that a reporter may implement; `NullReporter`
  implements both. Membership is checked by name. `plane(z, summary)` receives the plane's
  cost, not its field. A reporter's exception propagates through the run loop, which
  records the run as failed.
- **Output and resume.** The HDF5 output (output schema 1, the first released layout,
  ADR-058) records the resolved configuration, metadata and `configuration_hash`. A file of
  any other output schema is refused wherever it is opened, and the schema version is
  inside the resume key; a 0.x minor release may change the schema, listing the break.
  Every field of a recorded dataclass is recorded, at its default too, so adding a field
  moves the hash and bumps the schema. `resume=True` continues a file only when a rebuilt
  simulation has the same hash; when the call gives no noise seed, the rebuild takes the
  seed recorded in the file. The call's arguments are not stored, and the stored
  configuration cannot rebuild a simulation. `RunOutcome.status` is `running`,
  `complete`, `budget_exhausted` or `failed`. SIGTERM and Ctrl-C stop a run between its
  commits (`RunTerminated`) with the state written; a dense run whose last plane is
  interpolated resumes only with `resume="interpolated"`.
- **Scans.** `run_scan(scan, build, executor=...)` already reports progress: `"auto"` gives
  an outer bar or one log line per finished point; the inner runs are quiet. A scan's
  builder must pass `point.seed`. `python -m solprop.scan` is the command-line entry point.
- **Design tools.** [`solprop.tools`](../../guide/workflows/design-tools.md) has
  `pressure_zdw`, `pressure_rdw`, `capillary_design`, `soliton_params` and others, built
  on the Marcatili capillary; `pressure_rdw` also takes a `lattice` and builds a
  `VincettiMode`. The searches sample rather than bracket once: the zero-dispersion search
  samples $\beta_2$ and discards sign changes through a pole, and `pressure_rdw` samples
  129 logarithmic and 129 linear pressures. Modes expose `beta`, `beta2`, `zdw`, `aeff` and
  loss.
- **Modes.** `prop_waveguide` already selects a launch mode with `channel`, by index or
  display label (labels display as `HE1,1:even`), or computes a launch from an `incident`
  beam and a `coupling`. General modal bases measure physical energy with a flux metric
  that has cross terms ([general modal](../../guide/waveguides/general-modal.md)).
- **Errors.** The builders raise `SolpropError` subclasses (`DriverError`,
  `TranslationError`, `ModeError`); `DriverError` is also a `RuntimeError`, not a
  `ValueError`. Most carry a message only; `ResponseBudgetError` carries `z`, `field` and
  `calls`. Warnings (`MaterialRangeWarning`, `BoundaryWarning`) carry a message.
- **Memory.** `solprop.driver.memory` already estimates a run's persistent memory and admits
  it against a budget (`MemoryBudgetError`).
- **Pulses and beams.** `DataPulse` takes arrays; there are no file loaders.
  `SpatialPulse(temporal, waist, order, propz, jones)` has no displacement or tilt;
  `solprop.coupling.Alignment(displacement, tilt, model_error)` applies them as a coupling
  step, with the tilt a phase screen in the medium's real wave number.
- **Results.** The [processing](../../guide/results/processing.md) functions accept
  spatial files but do not return physical powers from them; compression refuses spatial
  and general modal outputs; general modal reconstruction needs the guide and reference
  tables, of which a file may hold only the identities.
- **Figures.** The geometry drawings in the user guide are generated by a docs script and
  styled with CSS custom properties, which only a browser renders.
- **The writer lock.** The `*.writer.lock` sidecar is documented under snapshots in the
  [output guide](../../guide/results/output.md).

### 4. Theory

No new physics. Four definitions matter:

- **Recipe equivalence.** A recipe $R$ and a Python call $C$ are equivalent when
  `Recipe(R).build()` and the `Simulation` built by $C$ have equal `configuration_hash`,
  with the noise seed fixed in both. The hash already covers everything that decides the
  computed fields (grid, modes, configuration, noise, stepper, materials and launch), so
  this is the right identity and needs no second definition.
- **Quantities.** Recipe values are numbers in the keyword's documented SI unit, or
  quantity strings (`"125 um"`, `"1.2 bar"`, `"7.5 fs"`, `"2e14 W/cm^2"`) converted to that
  unit. The unit is the one the docstring states today, so pressure stays in bar
  (conventions.md) and a bare `1.2` for `pressure` means 1.2 bar. A string is parsed as a
  decimal mantissa and a unit; the unit's dimension must match the keyword's, so
  `"125 um"` for `pressure` is refused. The conversion is exact in decimal (a power-of-ten
  prefix shifts the exponent, any other factor multiplies exactly) and is rounded once to
  the nearest binary64 value, so `"800 nm"` gives the literal `800e-9` that a script
  writes. Writing a value back chooses a prefix in which the shortest decimal
  representation of the double is exact, so the round trip is the identity. Keywords whose
  elements carry different units (`phases` is $[\mathrm{rad}, \mathrm{s}, \mathrm{s}^2,
  \dots]$, `betas` is $\mathrm{s}^n/\mathrm{m}$ by order, the `(z, p)` form of `pressure`
  pairs metres with bar) have a unit per element or per form (S9). Temperature accepts K
  only.
- **Beam tilt.** A tilt $(\theta_x, \theta_y)$ multiplies the launched field by
  $\exp[\mathrm{i}\,k(\omega)(x\sin\theta_x + y\sin\theta_y)]$, $k(\omega)$ the medium's
  real wave number at the launch plane, as `coupling.Alignment` defines it. With
  $k(\omega)$ rather than $k(\omega_0)$ every frequency leaves at the same angle: a pure
  tilt with no angular dispersion. The transverse wavenumbers $k(\omega)\sin\theta$ must
  lie inside the grid's transverse band at every active frequency, and the tilted and
  displaced beam must be negligible at the window's edges; otherwise the launch is
  refused.
- **Launch normalisation.** On a basis whose modes are flux-orthogonal (a diagonal flux
  metric, as for the Marcatili modes), the energy of a launch is the sum of the
  modes' energies, and fractions of the pulse energy are physical. On a general modal basis
  the flux metric $M$ has cross terms, and the energy of coefficients $a$ is $a^\dagger M a$,
  which depends on the relative phases. There a launch gives relative amplitudes and phases,
  and the coefficients are scaled so that $a^\dagger M a$ is the pulse energy; the
  per-mode shares are coordinate quantities, not an energy split (S5).

### 5. Proposed interface

#### 5.1 S1 — `solprop.recipe`

A recipe is a TOML document. Its vocabulary is defined once:

- **Build keywords** are the keywords of the route's `Simulation.from_*` constructor.
- **Execution settings** are a named list, held in an `[execution]` table:
  `boundary_check`, `memory_budget`, `fields`, `field_planes` and `resume_history`. They
  are applied to the built simulation's settings or passed to `Simulation.run`.
- **Refused keys.** `output`, `progress`, `progress_period`, `resume`, `snapshots` and
  `tdse_archive` choose files, terminals and resume behaviour, which the caller or the
  command line supplies. A recipe holding one is refused, naming it.

Tables other than `[recipe]`, `[execution]`, `[scan]` and `[pins]` group build keywords
for reading; the group of each keyword comes from its metadata (S9). The loader flattens
them, so a keyword may appear in any group table, but at most once.

```text
[recipe]
format = 1
route = "capillary"
title = "RDW in a helium capillary"
solprop = "0.2.0"                 # the version that wrote it; informative

[geometry]
radius = "125 um"
length = "1 m"
mode_average = true

[medium]
gas = "He"
pressure = ["0 bar", "1.2 bar"]   # a gradient, as the keyword accepts
temperature = "293.15 K"

[nonlinear]
plasma = true

[input]
wavelength = "800 nm"
fwhm = "7.5 fs"
energy = "275 uJ"

[grid]
wavelength_limits = ["100 nm", "1500 nm"]
trange = "200 fs"

[execution]
fields = "planes"
```

- **Only what was set is written.** A keyword left out takes solprop's default at build
  time, so a recipe follows solprop's improvements. The resolved values, defaults and
  seed included, are recorded in the output beside it (S2).
- **`None`.** TOML has no null. Where a keyword accepts `None` as a value with its own
  meaning (`noise = None` against the default `"default"`), the recipe writes the string
  `"none"`. The metadata records which keywords accept it, and a test checks that no
  keyword accepts both `None` and the string `"none"` as different values.
- **Objects.** Values that are objects (a pulse list, a `Gas` mixture, a coupling, a
  profile, the guides and mode providers of `prop_waveguide`, and on the object routes the
  grids, spaces, media, beams, responses and settings) are inline tables or arrays of
  tables with a `type` key naming an entry in a fixed registry in `solprop.recipe`, and
  that entry's keywords:
  `pulses = [{type = "GaussianPulse", wavelength = "800 nm", fwhm = "30 fs", energy = "1 mJ"}]`.
  An entry is a constructor or a named class method (`type =
  "PressureGradient.two_point"`). Only registered types load; a recipe never imports a
  module or evaluates code.
- **What a recipe cannot hold.** Callables (`beta`, `loss`, a mode's loss function) and
  arrays other than data files. `Recipe.from_call` refuses them, naming the keyword; the
  Python export is the way out.
- **Data files.** Arrays (a measured spectrum, an index table, a mode table) are referred to
  by a path relative to the recipe, with a SHA-256 recorded beside it and checked on load:
  `{type = "DataPulse", file = "spectrum.csv", sha256 = "…"}`. The path, after symbolic
  links are resolved, must lie inside the recipe's directory unless the caller allows
  otherwise.
- **Parameter paths.** Every value has a canonical path: the keyword, then `.name` for a
  keyword of an object and `[i]` for an element of an array, as in `pressure`,
  `pulses[1].energy` or `pressure[1]`. Paths are what `Recipe.set` and `Recipe.get` take,
  what errors and warnings name (S10), what a scan varies and what a pin sets. The metadata
  of a path is that of the keyword it ends on (S9).
- **Pins.** A `[pins]` table sets a value from a design target instead of a literal:
  `pressure = {target = "zdw", wavelength = "800 nm", search = ["0.1 bar", "10 bar"],
  root = "lowest"}`. Targets are those of the design solvers (S7): `zdw`, `rdw` (with the
  nonlinear phase where `pressure_rdw` has it) and `soliton_order`. A pinned path may not
  also be set literally. Pins are resolved at build, after the literal values, in the
  order of their dependencies (a pin whose target reads another pinned path follows it); a
  cycle is refused, naming its paths. An unreachable target is an issue on the pinned
  path with the attainable range. `root` chooses among several roots (`"lowest"`,
  `"highest"` or an index); the solver reports all of them. The resolved value goes into
  the resolved recipe (S2), and the Python export writes the solver call before the
  route's call.
- **Scans.** An optional `[scan]` table names paths and their values (a list, or
  `{start, stop, num, spacing = "linear" | "log"}`). `Recipe.scan()` returns the `Scan`
  and a picklable builder that passes `point.seed` and re-solves the pins at each point,
  so pool and queue executors work without a module-level function.
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

recipe = Recipe.load("rdw.toml")          # or Recipe.from_dict(...), Recipe.loads(text)
recipe.set("pressure", "1.5 bar")         # validated against the metadata (S9)
recipe.set("pulses[1].energy", "10 uJ")
problems = recipe.check()                 # list of RecipeIssue (S10), empty if fine
simulation = recipe.build()               # a Simulation, not run
output = simulation.run("rdw.h5")         # the recipe is recorded (S2)
recipe.to_python()                        # a plain script with the route's call
recipe.save("rdw-1.5bar.toml")
Recipe.from_call(prop_capillary, radius=125e-6, ...)   # a recipe from keywords
```

And the command line (a `solprop` console script and `python -m solprop`):

```text
solprop run rdw.toml -o rdw.h5 [--progress log] [--resume]
solprop check rdw.toml                    # validate and build; print issues; exit status
solprop export rdw.toml --python          # print the equivalent script
solprop resume rdw.h5                     # S2
solprop scan …                            # the existing python -m solprop.scan, unchanged
```

`to_python()` writes the route's `prop_*` call when every key is on its surface, and
otherwise `Simulation.from_*`, the settings and `run`; keyword arguments in SI, one comment
per keyword stating the recipe's quantity string, and nothing that a reader would not
write by hand.

#### 5.2 S2 — the recipe in the output, and resume from a file

At run start the driver writes, when the simulation was built from a recipe, a group
`recipe` in the output holding:

- `authored`: the recipe's canonical TOML text, as written (only what was set);
- `resolved`: every build keyword and execution setting at the value used, defaults, pins
  and the noise seed included;
- the recipe format, the SHA-256 of each text, and the `configuration_hash` the recipe
  built;
- the data files the recipe refers to, under generated names (they are usually small).

The group is provenance: it is **not** part of `configuration_hash`, so editing a comment
or the title does not block a resume, and it needs **no output-schema change**: the group
is optional, a reader that does not know it ignores it, and a file without it is a file
from a script. If the simulation was changed after it was built (its `settings` replaced,
`with_input`, `with_mixture`), its hash differs from the recipe's; the driver then writes
no recipe group and warns, so a file never holds a recipe that does not rebuild its run.

`Recipe.from_output(path, resolved=False)` reads the authored recipe back (for editing) or
the resolved one (to show the setup that ran). `solprop resume run.h5` (or
`resume_file(path)`) rebuilds the simulation from the **resolved** recipe, so a later
solprop's defaults do not move the run, checks that its hash equals the stored one, and
continues the file through the existing `resume=True` path, which compares hashes as it
does now. A file written without a recipe resumes as it does today, from the script.

**Received files.** Resuming a file runs the setup it holds, and the guide says so. The
limits: the recipe passes the same checks as a loaded one (registered types only, no
execution keys that choose paths); embedded data files are read into memory, never
written at their stored names; node and memory budgets apply; and parsing an HDF5 file is
itself an exposure of the HDF5 library, which the guide states.

#### 5.3 S3 — a plane observer

A new optional protocol beside `InterruptibleReporter` and `SummarisingReporter`:

```text
class PlaneObserver(ProgressReporter, Protocol):
    def observe_plane(self, z: float, plane: PlaneView) -> None: ...
```

The specific name avoids reporters that already have an `observe` attribute, and
`NullReporter` does **not** implement it, so a quiet run builds no view. The driver calls
`observe_plane` once per save plane, after the plane is committed to the output and before
`plane()`.

`PlaneView` is lazy. It holds the plane's position, the grid, the statistics row and a
description of the basis (S16: whether its quantities are physical flux or coordinate
quantities, and in which units). Its methods compute on request: the frequency- and
time-domain field, by mode where modes exist, and the spectral density. On a host backend
nothing is copied until asked. On a device backend, a plane whose field the output keeps
is already on the host; on a field-free plane (`fields="none"` or `"planes"`, the scan
default), asking for the field fetches one plane from the device, and the driver counts
that transfer in the run's resource summary. The arrays are valid only for the call; an
observer copies what it keeps. The existing rule holds: nothing an observer does may
change the propagation.

**Failure and workload.** An exception from `observe_plane`, other than `RunTerminated`
and `KeyboardInterrupt`, does not fail the run: the driver logs it once with its
traceback, warns, and calls the observer no more in that run; `plane`, `finished` and
`interrupted` are still delivered. The observer runs in the march's thread and solprop
bounds nothing about its time; the driver reports the time spent in observers in the
resource summary. The guide gives the pattern a front end uses: reduce the plane to what
it draws, put it in a bounded queue that replaces an undelivered preview, and return,
keeping terminal and error events on a reliable channel. Transport and disconnection are
the front end's; solprop guarantees only the above.

An observer may also implement `observe_boundary(z, verdict)`, called with the boundary
monitor's verdict when it changes, so a front end shows it without reading the log.

#### 5.4 S4 — scan progress

`run_scan(..., progress=)` accepts a `ScanReporter` as well as its literal choices:
`scan_started(n_points)`, `point_started(index, point)`,
`point_finished(index, status, wall)`, `scan_finished()`, and optionally
`point_reporter(index) -> ProgressReporter | None` for per-point progress.

- The local executor calls `point_reporter` in this process.
- A pool reports `point_started` when it dispatches a point and `point_finished` when it
  collects it. Per-point progress needs a new message on the worker pipe, carrying `z`
  and the `PlaneSummary` of each plane (never a field), at the progress period.
- Batch, queue and Slurm executors report point boundaries only, from the collected
  files.

#### 5.5 S5 — launch into chosen modes

`prop_capillary` and `prop_waveguide` gain `launch`, a mapping from mode label to a share
and, optionally, a phase in radians:
`launch = {"HE1,1": 0.9, "HE1,2": {"share": 0.1, "phase": 1.57}}`.

- **Labels** resolve as `channel` resolves them, by display label or index; a label that
  leaves a parity open when both polarisations are present is refused, naming the
  candidates. A label not in the mode set is refused, naming the available ones.
- **Shares.** On a flux-orthogonal basis (section 4) the shares are energy fractions,
  normalised to one. On a general modal basis they are relative amplitudes squared,
  scaled with the phases so that the flux-metric energy is the pulse's; the guide and the
  run's metadata say that the split is then a coordinate one.
- **Per pulse.** `launch` is either one mapping, for every pulse, or a list of mappings,
  one per pulse in `pulses`, so a pump and a seed can go into different modes.
- **With the other launch keywords.** `launch` generalises `channel`: giving both is
  refused. `incident` with `coupling` computes the launch from a beam and is exclusive
  with `launch`. `Simulation.with_input` remains the way to launch an array.
- The default, `None`, launches as today.

The launch is recorded through the input record that exists, not a new field where that
can be done (section 9, Q8).

#### 5.6 S6 — beam displacement and tilt

`SpatialPulse` gains `displacement=(d_x, d_y)` in m and `tilt=(theta_x, theta_y)` in rad,
both defaulting to zero, with the names and definitions of `coupling.Alignment`
(section 4). They are applied when the beam is built on a Cartesian grid; on a radial grid
a nonzero value is refused (it breaks the symmetry). `coupling.Alignment` stays the way to
misalign a beam *into a guide*; a beam's displacement and an `Alignment` add, which the
documentation states. The tilt's validity is the band and edge condition of section 4,
checked at build; `Alignment`'s `model_error` budget, which certifies a paraxial screen
against a guide's modes, does not apply to a free-space launch. Where the existing
`Alignment` record can record the launch, it does; otherwise see section 9 and Q8.

#### 5.7 S7 — design solvers for every guided route

A general solver and thin named wrappers:

```text
tools.solve_parameter(build_mode, parameter, search, target) -> DesignRoots
    # target: ("zdw", wavelength) | ("rdw", pump, rdw, peak_power=None)
    #       | ("soliton_order", N, pulse)
tools.radius_zdw(gas, pressure, zdw_wavelength, ...)          # new
```

`build_mode(value)` returns a mode for a trial value of the parameter (radius, pressure,
core size, wall thickness), so one solver serves the capillary, antiresonant, rectangular
and slab modes. It **reuses the existing searches**: it samples the target over the search
range (logarithmic and linear, as `pressure_rdw` does), discards sign changes through a
pole as the zero-dispersion search does, refines each bracketed root, and returns **every
root** found, with the attainable range when there is none. The `rdw` target carries the
nonlinear phase through `peak_power`, so it reproduces `pressure_rdw` beyond linear phase
matching. The existing capillary functions keep their signatures and gain no `mode`
keyword: `solve_parameter` is the general form.

#### 5.8 S8 — a capability query

```text
solprop.capabilities.route("capillary") -> RouteCapabilities
solprop.capabilities.species("N2") -> SpeciesCapabilities
```

`RouteCapabilities` lists the route's keywords (from S9), the allowed values of each
enumerated keyword, and the **combination rules** of the interface layer (route ×
keyword × species) with their reasons: for example, `tdse` requires the capillary route
and a helium fill; `kerr_form` requires `provider`. Each such rule has an id, and the
builder that enforces it raises with that id and `parameter` set; a lint-style test checks
that every refusal of the interface layer that sets `parameter` names a listed rule.
Refusals that depend on values (grid coverage, node budgets, material validity) are not
rules; `check()` finds them (S10). `SpeciesCapabilities` lists the ionisation models with
their status (curated, provisional, unavailable), whether a Raman response exists and its
status, and the validity ranges optmat records.

#### 5.9 S9 — parameter metadata

A table beside each interface function and each registered recipe type, one entry per
keyword:

- SI unit (or none), and a unit per element or per form where elements differ
  (section 4);
- kind (number, integer, boolean, choice, quantity array, object), allowed range or
  choices, and whether `None` is accepted (written `"none"`);
- default as documented (the function's default stays the single source; the table refers
  to it), and, where the default is `None` resolved by a rule (`rtol` by route, `plasma`
  and `thg` by `envelope`, `step_planes` by TDSE, `raman` by species, `pressure` at one
  bar), a resolver that returns the value that will be used with the rule's text;
- group (`geometry`, `medium`, `nonlinear`, `input`, `grid`, `numerics`, `output`), level
  (`basic` or `advanced`), a one-line summary, and the anchor of its manual section.

`solprop.recipe.parameters("capillary")` returns it. A test checks that the table and the
signature name the same keywords and that each default matches, for the `from_*`
surfaces, the `prop_*` surfaces and the execution settings. A front end's tooltips and
field validation come from this table; its own labels and wording are its business.

#### 5.10 S10 — structured validation

`SolpropError` gains an optional `parameter` attribute (a path, or a tuple of paths for a
conflict) and an optional `rule` (S8), set wherever the interface layer refuses a value.
The warnings that concern a value (`MaterialRangeWarning`, `BoundaryWarning`, the
provisional-Raman warning) carry `parameter` too. No exception changes class.

`Recipe.check()` collects every issue it can at the metadata level (type, unit,
dimension, range, unknown key, refused key, pins), then builds the simulation and
translates it, stopping before the backend is constructed, and reports the first refusal
of the builder or translation with its `parameter`. Translation finds the value-dependent
refusals; it costs about a tenth of a second on the capillary example, so a front end
calls it on a pause in typing, not on every keystroke. `solprop.grid.suggest(pulses,
route=...)` proposes `wavelength_limits` and `trange` from the input pulses and the
expected broadening, labelled as a starting point.

#### 5.11 S11 — a cost estimate

`Simulation.estimate()` returns:

- the grid sizes;
- the persistent memory, from `solprop.driver.memory`, admitted against the budget
  **before** anything is allocated (a run that would not be admitted is reported, not
  attempted);
- the output file size for the requested planes;
- the cost of one right-hand-side evaluation on the chosen backend, timed after one untimed
  warm-up call (so compilation and transform planning are excluded and reported
  separately), labelled with the backend and thread count;
- the integrator's calls per step, and, for fixed schedules, the total.

It does not predict the number of adaptive steps, which depends on the physics.

#### 5.12 S12 — measured-pulse loaders

`solprop.pulses.load_spectrum(path, *, wavelength_unit, columns, phase=None)` reads a
delimited text file of wavelength (or frequency) and intensity, optionally with phase, and
returns a `DataPulse` with the Jacobian applied when converting between wavelength and
frequency densities. `load_retrieval(path, format=...)` reads common pulse-retrieval
outputs (time or frequency amplitude and phase) into a `DataPulse`; the supported formats
are listed in the manual and each has a test file.

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
| `noise`, `seed` | all but planar | add to planar |
| `memory_budget` | `from_waveguide` and `Simulation.run` | a recipe carries it under `[execution]`, passed to `run`; documented as the same budget |
| `step_planes` default `0` (gnlse) vs `None` (capillary) | | same behaviour on gnlse (no TDSE); write `None` on both for one signature |
| integrator and `rtol` defaults by route | | deliberate (per-route qualification); exposed through S9's resolvers and documented in one table |
| `atol` defaults (1e-14 gnlse, 1e-10 others) | | Q3 |
| `loss` (capillary) vs `loss_db_per_m` (gnlse) | | different meanings (a switch or model vs a number); keep both, documented |
| `raman` default `False` (gnlse) vs `None` (gas routes) | | Q3; documented either way |

New keywords record nothing new when left at their defaults: each maps onto a setting the
run already records.

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
- which operations are available and which refused, with the reason: compression on
  spatial and general modal outputs, general modal reconstruction without the guide and
  reference tables (with the identities the file holds), physical powers from a spatial
  file;
- whether a recipe is recorded (S2) and the file's resumability (S17).

`PlaneView` (S3) carries the same description for the plane it shows; it does not make
the same physical operations available on every basis.

#### 5.17 S17 — a portable stop, and resumability

`Simulation.run` and `run_scan` accept `stop=`: any object with an `is_set()` method
(`threading.Event` qualifies). The driver checks it where SIGTERM takes effect today,
between commits, and when it is set raises `RunTerminated` through the same path, so the
state is written and `resume=True` continues the run. A front end sets it from any thread
on any platform; signals are not needed (on Windows, `os.kill` with `SIGINT` terminates a
process outright). `solprop run` maps SIGINT, SIGTERM and, on Windows, `SIGBREAK` (sent by
`CTRL_BREAK_EVENT`) onto it.

```text
solprop.output.resumability(path) -> Resumability
```

`Resumability` says whether the file can be continued and how: `resume=True`,
`resume="interpolated"` (the last plane is interpolated), or not at all, with the reason
(a complete run, no checkpoint written before a failure, another output schema, no
recipe and so no rebuild from the file alone), and the `z` it would continue from. A
front end's own run states (queued, stopping, stopped) stay its own; `RunOutcome.status`
is unchanged. A changed grid, a wider time window included, is a new run, not a resume
(Q4).

#### 5.18 S18 — previews

Before a run, a front end shows what a recipe will build without building the run:

- `Recipe.modes()` returns the mode set a guided route would build (labels, `beta`,
  `beta2`, `zdw`, effective areas and loss over the grid's band), at the cost of the mode
  solve only;
- `Recipe.preview_input()` returns the input pulse, and the beam on free-space routes, on
  the run's grid, as the launch will set them (S5, S6), with its energy and peak power.

### 6. Implementation strategies

- **Layer.** All of this is interface and driver work above translation
  (architecture.md section 3). No backend, kernel or stepper changes;
  `KERNEL_SCHEMA_VERSION` is unchanged. The output schema is unchanged by S1–S4 and
  S7–S18; S5 and S6 change it only if their launches cannot be recorded through the
  existing records (section 9).
- **Recipe keys = keywords (recommended) or a designed schema.** A designed schema with its
  own names (`[pulse] shape = "gaussian"`) reads well, but is a second vocabulary to keep in
  step with six functions and their docstrings. Using the build keywords themselves,
  grouped by metadata, keeps one vocabulary, makes `to_python()` direct, and lets the sync
  test (S9) catch every drift. A front end may present any names it likes.
- **Metadata: a table (recommended) or `Annotated` signatures.** `Annotated[float,
  Unit("m"), Range(0, None)]` puts metadata in the signature, but every interface function
  is already long, the type checker and the runtime array-shape checks would see the
  annotations, and the documentation renderer shows them. A table beside each function,
  checked against the signature by a test, costs less and changes no signature.
- **TOML reading and writing.** `tomllib` (standard library) reads. Writing needs either a
  small dependency (`tomli-w`) or a writer in `solprop.recipe` for the subset recipes use
  (tables, arrays of tables, strings, numbers, booleans, inline tables). The subset writer is
  about a hundred lines and avoids a dependency; recommended.
- **Unit parsing.** A small parser in `solprop.recipe` with a fixed table of SI prefixes,
  the units the keywords use and their dimensions, converting with `decimal`; no
  dependency on a units library.
- **Plane observer.** The view wraps what the driver already holds when it calls
  `plane()`; it computes or fetches only on request.
- **Stop.** The flag is polled where the commit guard already admits SIGTERM, so a stop
  costs one attribute call per commit.
- **Cost.** Loading and building a recipe add milliseconds. The plane observer costs only
  what the observer asks for. `check()` costs a translation. The estimate costs a warm-up
  and a few timed calls, only on request.

### 7. Verification and validation

The tolerance classes are those of the [testing policy](../testing.md#3-tolerance-policy).

- **Recipe equivalence** (class P, configuration identity). For every example in the user
  guide and gallery, a recipe written by `Recipe.from_call` and reloaded builds a
  simulation with the same `configuration_hash` as the original call, with the seed fixed.
  The keyword routes in phase 1, the object routes and the waveguide in phase 4.
- **Python export** (class P). `to_python()` of each of those recipes, executed in a
  subprocess with the run replaced by a build, gives the same hash.
- **Round trip** (class P). `save` then `load` returns an equal recipe; every registered
  type goes table → object → table and back unchanged; formats are upgraded by tests
  holding a recipe of each older format.
- **Quantities** (class P). Each unit and prefix converts to the literal a script writes,
  including the section 5.1 example's values; writing back is the identity; a wrong
  dimension and an ambiguous string are refused.
- **Provenance.** A file written from a recipe holds both texts and the built hash; a
  simulation changed after its build writes no recipe; a schema-1 file with and without
  the recipe group is read by the readers.
- **Resume from a file** (class T). A short capillary run stopped by the `stop` flag after
  a few planes, and another stopped by SIGTERM, resumed with `solprop resume`, agree with
  an uninterrupted run within an accuracy envelope against a tighter reference, set per
  case in a comment. `resumability` is checked on a complete file, a stopped file, an
  interpolated last plane and a failure before the first checkpoint.
- **Sync.** Metadata tables match the signatures (names, defaults) on every surface; every
  interface refusal that sets `parameter` names a listed rule; no keyword accepts both
  `None` and `"none"`.
- **Observer.** The observer's copy of each plane equals the plane read back from the file
  (class P). A fixed-schedule run with an observer that asks for every view agrees with
  the run without one (class R). A failing observer is detached and the run completes; a
  stop raised from an observer stops the run. Both on `numpy_ref`, `numba_cpu` and the
  native build, field-free planes included.
- **Estimate.** On the three backends, a budget below the persistent memory is reported
  without allocation, and the reported memory agrees with `solprop.driver.memory`.
- **Pins and scans.** A pinned pressure equals `pressure_zdw` for the same target; a scan
  re-solves its pins per point; a cycle and an unreachable target are refused.
- **Design solvers.** `solve_parameter` with the Marcatili builder reproduces
  `pressure_zdw` and `pressure_rdw` (with `peak_power`) to their tolerance and finds both
  roots where a resonance gives two; `radius_zdw` inverts `zdw`.
- **Loaders.** Each loader reads its test file to the energy stated in the file within
  the quadrature error of its sampling, and to the centre wavelength within one sample
  spacing; a wavelength-to-frequency conversion conserves energy to the same bound.
- **Launch.** On a Marcatili basis the per-mode energies equal the shares; on a general
  modal basis the flux-metric energy equals the pulse energy.
- **Safety.** A recipe naming an unregistered type, holding a refused key, or reaching a
  path outside its directory (symbolic links resolved) without an explicit allowance is
  refused; an embedded data file is never written at its stored name.

### 8. Documentation

- A new guide page, *Recipes and the command line* (`guide/setup/recipes.md`): the format,
  quantities, `None`, paths, objects, data files, pins, scans, `solprop run`, `check`,
  `export` and `resume`, with the capillary example above.
- [Resuming](../../guide/results/resuming.md): resume from a file alone, `resumability`,
  the portable stop, and that resuming a received file runs its setup.
- [Progress](../../guide/performance/progress.md): `PlaneObserver`, its failure policy and
  the coalescing pattern, `ScanReporter`.
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

- **Python interface: additive.** Every name is new, every new keyword defaults to
  today's behaviour, and the new reporter calls are optional protocols with names no
  reporter is likely to have (adding them to `ProgressReporter` itself would break
  third-party reporters and is not proposed). No exception changes class.
- **Output schema: unchanged by the recipe.** The recipe group is optional and outside the
  hash, so files written before and after phase 2 are read and resumed alike.
- **Output schema and S5, S6.** Every recorded field is recorded at its default, so a new
  field on `SpatialPulse` or a new launch record would move the hash and bump the schema,
  which in a 0.x minor release makes files of the previous release unreadable (ADR-058).
  S5 and S6 are designed to record their launches through records that exist (the input
  and `Alignment` records). If that proves impossible, the record changes are confined to
  phase 4 and made as one schema bump, listed under "Changed (breaking)" in the changelog
  (Q8).
- **Recipe stability.** Recipe `format = 1` becomes a public format when this SEP merges.
  Renaming a keyword later then needs a format upgrade rule as well as a changelog entry.
  The route vocabulary was aligned (`wavelength`, `shape`) before the first release; S14's
  remaining rows are Q3.
- **Two choices that would be breaking, and are not recommended:** renaming keywords to one
  vocabulary across routes where the meanings differ (`loss`/`loss_db_per_m`), and
  aligning integrator or tolerance defaults across routes (the per-route defaults were
  qualified separately).
- **Validated results:** unchanged; no computed field changes.
- **Dependencies:** none new (section 6).

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
- **A rule language for cross-field validation.** Instant validation of every rule would
  need the value-dependent refusals of translation written a second time. `check()`
  reuses them; only the combination rules are listed (S8).
- **Signals for stopping.** Portable only on POSIX; a flag is portable and testable.

### 11. Phases

The phases are made on one branch and the SEP reaches `main` in one pull request when its
last commit sets `Final` (SEP 0, *Implementation*). A front end in development builds
against the branch; the order puts what it needs first.

1. **Recipe core on `gnlse` and `capillary` (S1, S9, S10).** `solprop.recipe` with format
   1; quantity parser and TOML writer; paths and the `None` spelling; the registry for
   these routes' objects; metadata tables with resolvers and their sync tests;
   `Recipe.load/save/get/set/check/build/to_python/from_call/example`; `parameter` on errors and
   warnings; the `solprop` console script with `run`, `check` and `export`; the guide
   page. Verifies recipe equivalence, Python export and quantities on every
   keyword-route example of these routes.
2. **Provenance, resume and stop (S2, S15, S17).** The recipe group in the output,
   `Recipe.from_output`, `solprop resume`, the `stop` flag, `resumability`; the
   writer-lock text. Verifies resume from a file after a stop and after SIGTERM, and the
   received-file limits.
3. **Watching runs (S3, S4, S11, S16).** `PlaneObserver` and `PlaneView`, the boundary
   verdict, `ScanReporter` and the pool's progress message, `Simulation.estimate()`,
   `[scan]` tables and `Recipe.scan()`, result capabilities. Verifies observer neutrality
   and failure on the three backends, and scan progress on local and pool executors.
4. **Object routes, waveguide, launches and previews (S1 continued, S5, S6, S18).**
   Recipes for `waveguide`, `radial`, `free` and `planar` (registered types for guides,
   mode providers, grids, spaces, media, beams, responses and settings); `launch`; beam
   displacement and tilt; `Recipe.modes()` and `preview_input()`. Verifies recipe
   equivalence on these routes' examples and the launch normalisation; reports whether a
   schema bump was needed (Q8).
5. **Design, pins and capability (S7, S1's pins, S8, S12).** `solve_parameter`,
   `radius_zdw`; `[pins]`; `solprop.capabilities.route` and `species` with rule ids; the
   pulse loaders.
6. **Drawings and keyword gaps (S13, S14).** `solprop.plotting.geometry` and the guide
   figures moved onto it; the missing keywords added; the route-defaults table.

### 12. Open questions

- **Q1.** Recipe values: accept both quantity strings and SI numbers, and write quantity
  strings (recommended), or SI numbers only?
- **Q2.** Should `[recipe] route` be required, or inferred from the keywords present
  (recommended: required; inference is fragile between `capillary` and `waveguide`)?
- **Q3.** S14's open rows: align the `atol` defaults, and give `prop_gnlse` the gas-route
  `raman` default semantics? Each is a change of default. Recommended: keep both and
  document them as deliberate; if either is to change, change it in a minor release
  before this SEP merges, listed as a break (ADR-058), so that recipe format 1 is
  published with the final defaults.
- **Q4.** Should `solprop resume` also accept a recipe edited after the run (for example a
  longer `length`), refusing only when the hash differs in more than the run's end?
  Recommended: no in this SEP; extending a finished run is its own question. A changed
  grid is always a new run.
- **Q5.** Which pulse-retrieval formats does `load_retrieval` support first?
- **Q6.** Is `tomli-w` acceptable as a dependency instead of a subset writer?
- **Q7.** Should the metadata include a recommended display unit per keyword (fs for
  durations, µm for radii), or is that left to front ends? Recommended: include it, so
  that recipes written by any front end use the same quantity strings.
- **Q8.** If S5 or S6 cannot record its launch through existing records, is one
  output-schema bump in phase 4 acceptable (listed under ADR-058), or should the feature
  wait for the resume-compatibility policy of 1.0? Recommended: the bump, made once and
  in the same minor release as any other schema change.

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
