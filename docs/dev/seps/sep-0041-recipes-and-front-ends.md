# SEP 41 — Recipes, and the hooks a front end needs

| | |
|---|---|
| **SEP** | 41 |
| **Title** | Recipes, and the hooks a front end needs |
| **Author** | John C. Travers |
| **Drafting agent** | Claude Opus 5.5 (2026-10-31) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | Opus 5.5 (AI) |
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
instead of copying from it: parameter metadata, structured validation errors, a capability
query, a supported observer of saved planes, scan progress, a cost estimate, launch into
chosen modes, beam offset and tilt, design solvers for every guided route, measured-pulse
loaders and geometry drawings. The first user is a desktop front end being designed
separately; scripts and batch systems gain the same things (`solprop run recipe.toml`).
Success is judged by round trips: a recipe and the Python call it stands for build the same
`Simulation` (equal `configuration_hash`), a recipe exported to Python and run gives the
same run, and a file written from a recipe resumes from the file alone.

Every change proposed is **additive**. Nothing here renames a keyword, changes a default or
removes behaviour; section 9 lists the two places where a later choice could be breaking
and recommends against both.

### 2. Motivation and scope

A front end must show every parameter with its unit, default, range and documentation;
check a setup as it is typed and put each error next to its field; offer only the options a
route and species support; watch a run as it goes; save a setup and reopen it; reopen a
results file with the setup that made it; and resume a stopped run. Today each of these
would mean a hand-written copy of rules that live in solprop's code, and the copy would
drift with every release. The principle of this SEP is that **solprop holds the physics,
the defaults and the rules; a front end holds only presentation**.

The same gaps hurt script users. A results file does not record the call that made it, so a
run can be resumed only by the script that started it (the [resume key](../../guide/results/resuming.md)
checks a rebuilt simulation; it cannot rebuild one). A batch job is a Python file per
configuration. A scan's builder must be a module-level function for pool executors
([scans](../../guide/workflows/scans.md)), which a declarative description gives for free.

**In scope** (the labels S1–S15 are used throughout):

| | Change | Section |
|---|---|---|
| S1 | `solprop.recipe`: the TOML format, load, save, build, Python export, `solprop run` | 5.1 |
| S2 | The recipe recorded in the output; resume from a file alone | 5.2 |
| S3 | A supported observer of each saved plane | 5.3 |
| S4 | Progress for scans | 5.4 |
| S5 | Launch into chosen modes on modal routes | 5.5 |
| S6 | Offset and tilt on free-space beams | 5.6 |
| S7 | Design solvers for every guided route; radius from a zero-dispersion wavelength | 5.7 |
| S8 | A capability query | 5.8 |
| S9 | Machine-readable parameter metadata | 5.9 |
| S10 | Structured validation errors, a cheap check and grid suggestions | 5.10 |
| S11 | A cost estimate before a run | 5.11 |
| S12 | Measured-pulse loaders | 5.12 |
| S13 | Geometry drawings in the library | 5.13 |
| S14 | Route keywords: fill the gaps, document the deliberate differences | 5.14 |
| S15 | The writer-lock sidecar, documented | 5.15 |

**Out of scope:** the front end itself, which is a separate project; remote execution and
a network service (a recipe and `solprop run` are how remote work is done for now, through
the existing executors); arbitrary Python inside a recipe (custom pulse shapes or
responses: the way out is the Python export); a graphical recipe editor in solprop; any
change to the physics, the kernel contract or the steppers.

### 3. Context

- **Entry points.** [`prop_gnlse`, `prop_capillary` and `prop_waveguide`](../../guide/setup/running.md)
  take keywords; `prop_radial`, `prop_free` and `prop_planar` take objects (`space`,
  `medium`, `initial`, `responses`, `settings`) with an explicit response list. Each has a
  `Simulation.from_*` constructor, so a simulation can be built without running it, and
  `Simulation.run(output, progress=..., resume=...)` runs it.
- **Progress.** [`ProgressReporter`](../../guide/performance/progress.md) is a runtime
  protocol with four calls (`started`, `step`, `plane`, `finished`), extended by optional
  protocols (`InterruptibleReporter.interrupted`, `SummarisingReporter.summary`) that a
  reporter may implement. `plane(z, summary)` receives the plane's cost, not its field.
- **Output and resume.** The HDF5 output (output schema 27) records the resolved
  configuration, metadata and `configuration_hash`, and readers upgrade older schema
  versions. `resume=True` continues a file only when a rebuilt simulation has the same
  hash; the call's arguments are not stored, and the stored configuration cannot rebuild a
  simulation.
- **Scans.** `run_scan(scan, build, executor=...)` runs points with `progress="none"` and
  reports nothing until it returns; `python -m solprop.scan` is the only command-line entry
  point.
- **Design tools.** [`solprop.tools`](../../guide/workflows/design-tools.md) has
  `pressure_zdw`, `pressure_rdw`, `capillary_design`, `soliton_params` and others, written
  for the Marcatili capillary; modes expose `beta`, `beta2`, `zdw`, `aeff` and loss.
- **Errors.** `DriverError` and the other `SolpropError` classes carry a message only.
- **Memory.** `solprop.driver.memory` already estimates a run's persistent memory and admits
  it against a budget (`MemoryBudgetError`).
- **Pulses.** `DataPulse` takes arrays; there are no file loaders. `SpatialPulse(waist,
  order, propz, jones)` has no offset or tilt; `solprop.coupling.Alignment` applies them as
  a coupling step.
- **Figures.** The geometry drawings in the user guide are generated by a docs script and
  styled with CSS custom properties, which only a browser renders.

### 4. Theory

No new physics. Two definitions matter:

- **Recipe equivalence.** A recipe $R$ and a Python call $C$ are equivalent when
  `Recipe(R).build()` and the `Simulation` built by $C$ have equal `configuration_hash`.
  The hash already covers everything that decides the computed fields (grid, modes,
  configuration, noise, stepper, materials and launch), so this is the right identity and
  needs no second definition.
- **Quantities.** Recipe values are SI numbers in the keyword's documented unit, or
  quantity strings (`"125 um"`, `"1.2 bar"`, `"7.5 fs"`, `"2e14 W/cm^2"`) converted to that
  unit. The keyword's unit is the one its docstring states today, so pressure stays in bar
  (conventions.md) and a bare `1.2` for `pressure` means 1.2 bar. Conversion is exact
  scaling by a factor table; temperature accepts K only.

### 5. Proposed interface

#### 5.1 S1 — `solprop.recipe`

A recipe is a TOML document. Its keys **are the route's keywords**, so there is no second
vocabulary to keep in step with the interface; tables group them for reading, and the group
of each keyword comes from its metadata (S9). The loader flattens the tables, so a keyword
may appear in any table or at the top of `[parameters]`, but at most once.

```text
[recipe]
format = 1
route = "capillary"
title = "RDW in a helium capillary"
solprop = "0.1.0"                 # the version that wrote it; informative

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
lambda0 = "800 nm"
fwhm = "7.5 fs"
energy = "275 uJ"

[grid]
wavelength_limits = ["100 nm", "1500 nm"]
trange = "200 fs"

[output]
n_save = 201
```

- **Only what was set is written.** A keyword left out takes solprop's default at build
  time, so a recipe follows solprop's improvements; the resolved values are recorded in the
  output's metadata, as they are today.
- **Objects.** Values that are objects (a pulse list, a `Gas` mixture, a coupling, a
  profile, and on the object routes `space`, `medium`, `initial`, `responses`, `settings`)
  are inline tables or arrays of tables with a `type` key naming a constructor from a fixed
  registry in `solprop.recipe`, and the constructor's keywords:
  `pulses = [{type = "GaussianPulse", wavelength = "800 nm", fwhm = "30 fs", energy = "1 mJ"}]`.
  Only registered types load; a recipe never imports a module or evaluates code.
- **Data files.** Arrays (a measured spectrum, an index table, a mode table) are referred to
  by a path relative to the recipe, with a SHA-256 recorded beside it and checked on load:
  `{type = "DataPulse", file = "spectrum.csv", sha256 = "…"}`.
- **Scans.** An optional `[scan]` table names keywords and their values (a list, or
  `{start, stop, num, spacing = "linear" | "log"}`); `Recipe.scan()` returns the `Scan` and
  a picklable builder, so pool and queue executors work without a module-level function.
- **Format versioning.** `format` is an integer. The loader upgrades older formats and
  refuses newer ones; unknown keys are refused, naming the key and the nearest valid one.

The Python surface:

```text
from solprop.recipe import Recipe

recipe = Recipe.load("rdw.toml")          # or Recipe.from_dict(...), Recipe.loads(text)
recipe.set("pressure", "1.5 bar")         # validated against the metadata (S9)
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

`to_python()` writes the route's own call with keyword arguments in SI, one comment per
keyword stating the recipe's quantity string, and nothing that a reader would not write by
hand.

#### 5.2 S2 — the recipe in the output, and resume from a file

At run start the driver writes the recipe's canonical TOML text into the output (a string
dataset `recipe` with attributes `recipe_format` and `recipe_sha256`) when the simulation
was built from one; `Recipe.from_call` covers scripts that want the same. The recipe is
provenance: it is **not** part of `configuration_hash`, so editing a comment or the title
does not block a resume. `Recipe.from_output(path)` reads it back; `solprop resume run.h5`
(or `resume_file(path)`) rebuilds the simulation from the stored recipe and continues the
file through the existing `resume=True` path, which compares hashes as it does now. Data
files a recipe refers to are stored beside it in the output (they are usually small), so a
resume does not depend on the original directory. A file written without a recipe resumes
as it does today, from the script.

#### 5.3 S3 — a plane observer

A new optional protocol beside `InterruptibleReporter` and `SummarisingReporter`:

```text
class ObservingReporter(ProgressReporter, Protocol):
    def observe(self, z: float, plane: PlaneView) -> None: ...
```

The driver calls `observe` once per save plane, after the plane is committed to the output
and before `plane()`. `PlaneView` gives read-only access to that plane's field (frequency
and time domain, by mode where modes exist), the grid and the plane's statistics row. The
arrays are views valid only for the call; an observer copies what it keeps. The existing
rule holds: nothing an observer does may change the propagation. `NullReporter` implements
`observe` as a no-op. A front end reduces each plane to what it draws (a spectrum in dB, a
cropped power envelope) inside the run's process and streams only that.

#### 5.4 S4 — scan progress

`run_scan(..., progress=)` also accepts a `ScanReporter`:
`scan_started(n_points)`, `point_started(index, point)`,
`point_finished(index, status, wall)`, `scan_finished()`, and optionally
`point_reporter(index) -> ProgressReporter | None` for per-point progress where the
executor runs points in this process or can forward events (local and pool executors;
batch, queue and Slurm executors report point boundaries only, from the collected file).

#### 5.5 S5 — launch into chosen modes

`prop_capillary` and `prop_waveguide` gain `launch`: a mapping from mode label (as the
mode set names them, for example `"HE11"`, `"HE12"`) to an energy fraction and, optionally,
a phase in radians: `launch = {"HE11": 0.9, "HE12": {"fraction": 0.1, "phase": 1.57}}`.
Fractions are normalised to one and the pulse energy is shared accordingly. The default,
`None`, launches into the lowest mode as today. A label not in the mode set is refused,
naming the available ones. `launch` and an explicit field array are mutually exclusive.

#### 5.6 S6 — beam offset and tilt

`SpatialPulse` gains `offset=(x0, y0)` in m and `tilt=(theta_x, theta_y)` in rad, both
defaulting to zero, applied when the beam is built on a Cartesian grid. On a radial grid a
nonzero offset or tilt is refused (it breaks the symmetry). `coupling.Alignment` stays the
way to misalign a beam *into a guide*; giving both an offset on the beam and an
`Alignment` is allowed and they add, which the documentation states.

#### 5.7 S7 — design solvers for every guided route

A general solver and thin named wrappers:

```text
tools.solve_parameter(build_mode, parameter, bracket, target)
    # target: ("zdw", wavelength) | ("rdw", (pump, rdw)) | ("soliton_order", (N, pulse))
tools.radius_zdw(gas, pressure, zdw_wavelength, ...)          # new
tools.pressure_zdw(..., mode=None) / pressure_rdw(..., mode=None)
```

`build_mode(value)` returns a mode for a trial value of the parameter (radius, pressure,
core size, wall thickness), so the same root finder serves the capillary, antiresonant,
rectangular and slab modes. The existing capillary signatures are kept; a `mode` keyword,
when given, replaces the Marcatili construction. The solver reports when the target is not
reachable inside the bracket, with the attainable range.

#### 5.8 S8 — a capability query

```text
solprop.capabilities.route("capillary") -> RouteCapabilities
solprop.capabilities.species("N2") -> SpeciesCapabilities
```

`RouteCapabilities` lists the route's keywords (from S9), the allowed values of each
enumerated keyword, and the incompatibilities as rules with their reasons (for example,
`tdse` requires the capillary route and a helium fill; `kerr_form` requires `provider`).
`SpeciesCapabilities` lists the ionisation models with their status (curated, provisional,
unavailable), whether a Raman response exists and its status, and the validity ranges
optmat records. The rules are the ones the builders enforce: each refusal in a builder
names the rule it applies, and a test checks that every refusal is listed.

#### 5.9 S9 — parameter metadata

A table beside each interface function, one entry per keyword: SI unit (or none), kind
(number, integer, boolean, choice, quantity array, object), allowed range or choices,
default as documented (the function's default stays the single source; the table refers to
it), group (`geometry`, `medium`, `nonlinear`, `input`, `grid`, `numerics`, `output`),
level (`basic` or `advanced`), a one-line summary, and the anchor of its manual section.
`solprop.recipe.parameters("capillary")` returns it. A test checks that the table and the
signature name the same keywords and that each default matches. A front end's tooltips and
field validation come from this table; its own labels and wording are its business.

#### 5.10 S10 — structured validation

`SolpropError` gains an optional `parameter` attribute (the keyword at fault, or a tuple of
them for a conflict), set wherever a builder refuses a keyword. Errors raised today as
`ValueError` or `TypeError` from the builders become subclasses of both their current class
and `SolpropError`, so existing `except` clauses keep working. `Recipe.check()` and
`Simulation.check(**kw)` build the problem up to, but not including, translation to the
kernel, and return every issue found rather than the first. `solprop.grid.suggest(pulses,
route=...)` proposes `wavelength_limits` and `trange` from the input pulses and the
expected broadening, labelled as a starting point.

#### 5.11 S11 — a cost estimate

`Simulation.estimate()` returns the grid sizes, the persistent memory (from
`solprop.driver.memory`), the output file size for the requested planes, and a cost per
step measured by timing a few right-hand-side evaluations on the chosen backend (taking
about a second, and labelled with the backend and thread count). It does not predict the
number of steps, which depends on the physics; it reports the cost per step and, for fixed
schedules, the total.

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
| `step_planes` default `0` (gnlse) vs `None` (capillary) | | same behaviour on gnlse (no TDSE); write `None` on both for one signature |
| integrator and `rtol` defaults by route | | deliberate (per-route qualification); exposed through S8 and documented in one table |
| `atol` defaults (1e-14 gnlse, 1e-10 others) | | review the scale argument; document it, or align in a separate change |
| `loss` (capillary) vs `loss_db_per_m` (gnlse) | | different meanings (a switch or model vs a number); keep both, documented |
| `raman` default `False` (gnlse) vs `None` (gas routes) | | deliberate (silica response is opt-in; gas responses follow the species); documented |

#### 5.15 S15 — the writer-lock sidecar

Each HDF5 output keeps a `*.writer.lock` file beside it. It persists by design: removing a
lock file after releasing an advisory lock races with a second writer opening it. This SEP
documents the sidecar in the [output guide](../../guide/results/output.md) and leaves the
behaviour unchanged; front ends hide it.

### 6. Implementation strategies

- **Layer.** All of this is interface and driver work above translation
  (architecture.md section 3). No backend, kernel or stepper changes; the kernel contract's
  `SCHEMA_VERSION` (37) is unchanged.
- **Recipe keys = keywords (recommended) or a designed schema.** A designed schema with its
  own names (`[pulse] shape = "gaussian"`) reads well, but is a second vocabulary to keep in
  step with six functions and their docstrings. Using the keywords themselves, grouped by
  metadata, keeps one vocabulary, makes `to_python()` trivial, and lets the sync test (S9)
  catch every drift. A front end may present any names it likes.
- **Metadata: a table (recommended) or `Annotated` signatures.** `Annotated[float,
  Unit("m"), Range(0, None)]` puts metadata in the signature, but every interface function
  is already long, the type checker and the runtime array-shape checks would see the
  annotations, and the documentation renderer shows them. A table beside each function,
  checked against the signature by a test, costs less and changes no signature.
- **TOML reading and writing.** `tomllib` (standard library) reads. Writing needs either a
  small dependency (`tomli-w`) or a writer in `solprop.recipe` for the subset recipes use
  (tables, arrays of tables, strings, numbers, booleans, inline tables). The subset writer is
  about a hundred lines and avoids a dependency; recommended.
- **Unit parsing.** A small parser in `solprop.recipe` with a fixed table of SI prefixes and
  the units the keywords use; no dependency on a units library.
- **Plane observer.** The driver already holds the committed plane when it calls `plane()`;
  `observe` receives read-only views of it. On device backends the plane has already been
  copied to the host for the output, so no extra transfer is needed.
- **Cost.** Loading and building a recipe add milliseconds. The plane observer costs only
  what the observer does. The estimate costs about a second and runs only on request.

### 7. Verification and validation

- **Recipe equivalence.** For every example in the user guide and gallery that uses a
  keyword route, a recipe written by `Recipe.from_call` and reloaded builds a simulation with
  the same `configuration_hash` as the original call. Then the same for the object routes in
  phase 4.
- **Python export.** `to_python()` of each of those recipes, executed in a subprocess with
  the run replaced by a build, gives the same hash.
- **Round trip.** `save` then `load` returns an equal recipe; formats are upgraded by tests
  holding a recipe of each older format.
- **Resume from a file.** A short capillary run stopped by `SIGTERM` after a few planes and
  resumed with `solprop resume` matches an uninterrupted run to rounding (resume is not
  bitwise; [reproducibility](../../guide/numerics/reproducibility.md)).
- **Sync.** Metadata tables match the signatures (names, defaults); every builder refusal is
  listed in the capability rules; every registered recipe type has a constructor test.
- **Observer.** A run with an observer that copies every plane is bitwise the run without
  one, and the copies equal the planes read back from the file.
- **Quantities.** Each unit and prefix converts exactly; ambiguous strings are refused.
- **Design solvers.** `solve_parameter` with the Marcatili builder reproduces
  `pressure_zdw` and `pressure_rdw` to their tolerance; `radius_zdw` inverts `zdw`.
- **Loaders.** Each loader reads its test file to the energy and centre wavelength stated in
  the file, and a wavelength-to-frequency conversion conserves energy.
- **Safety.** A recipe naming an unregistered type, or a path outside the recipe's
  directory without an explicit allowance, is refused.

### 8. Documentation

- A new guide page, *Recipes and the command line* (`guide/setup/recipes.md`): the format,
  quantities, objects, data files, scans, `solprop run`, `check`, `export` and `resume`, with
  the capillary example above.
- [Resuming](../../guide/results/resuming.md): resume from a file alone.
- [Progress](../../guide/performance/progress.md): `ObservingReporter`, `ScanReporter`.
- [Pulses](../../guide/setup/pulses.md) and [beams](../../guide/free-space/beams.md):
  loaders, mode launch, offset and tilt.
- [Design tools](../../guide/workflows/design-tools.md): the general solver and
  `radius_zdw`.
- [Output](../../guide/results/output.md): the recipe dataset and the writer-lock sidecar.
- A table of route defaults (integrators, tolerances) with the reason for each difference.
- Docstrings for every new name; a gallery example that loads a recipe, edits a value and
  runs it.

### 9. Compatibility and impact

- **Breaking changes: none proposed.** Every name is new, every new keyword defaults to
  today's behaviour, and the new reporter calls are optional protocols that existing
  reporters need not implement (adding them to `ProgressReporter` itself would break
  third-party reporters and is not proposed).
- **Errors.** Exceptions that become `SolpropError` subclasses keep their current base
  classes, so `except ValueError` still catches them.
- **Output schema.** Phase 2 bumps the output schema (27 → 28) for the `recipe` dataset.
  Readers upgrade older files as they do now; a file written before has no recipe and is
  resumed from its script.
- **Recipe stability.** Recipe `format = 1` becomes a public format. Renaming a keyword later
  would then need a format upgrade rule as well as a deprecation, which is the main reason to
  settle S14's open rows (section 12, Q3) before the format is published.
- **Two choices that would be breaking, and are not recommended:** renaming keywords to one
  vocabulary across routes (for example `loss`/`loss_db_per_m`), and aligning integrator or
  tolerance defaults across routes (the per-route defaults were qualified separately).
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
- **One SEP per change.** The changes share one purpose and one test strategy (the recipe
  round trips); the small ones are phases here rather than separate proposals.

### 11. Phases

Each phase merges on its own with a green gate.

1. **Recipe core on the keyword routes (S1, S9, S10).** `solprop.recipe` with format 1 for
   `gnlse`, `capillary` and `waveguide`; quantity parser and TOML writer; metadata tables
   and their sync tests; `Recipe.load/save/set/check/build/to_python/from_call`; structured
   errors; the `solprop` console script with `run`, `check` and `export`; the guide page.
   Verifies recipe equivalence and Python export on every keyword-route example.
2. **Provenance and resume (S2, S15).** The recipe in the output (output schema 28),
   `Recipe.from_output`, `solprop resume`; the writer-lock documentation. Verifies resume
   from a file after `SIGTERM`.
3. **Watching runs (S3, S4, S11).** `ObservingReporter` and `PlaneView`, `ScanReporter`,
   `Simulation.estimate()`, `[scan]` tables and `Recipe.scan()`. Verifies observer
   neutrality and scan progress on local and pool executors.
4. **Object routes and launches (S1 continued, S5, S6).** Recipes for `radial`, `free` and
   `planar` (registered types for spaces, media, beams, responses and settings); `launch`;
   beam offset and tilt. Verifies recipe equivalence on the free-space examples.
5. **Design and capability (S7, S8, S12).** `solve_parameter`, `radius_zdw`, the `mode`
   keyword on the pressure solvers; `solprop.capabilities`; the pulse loaders.
6. **Drawings and keyword gaps (S13, S14).** `solprop.plotting.geometry` and the guide
   figures moved onto it; the missing keywords added; the route-defaults table.

A front end can start on phase 1 alone; phase 3 replaces any reading of the output from
inside `plane()`.

### 12. Open questions

- **Q1.** Recipe values: accept both quantity strings and SI numbers, and write quantity
  strings (recommended), or SI numbers only?
- **Q2.** Should `[recipe] route` be required, or inferred from the keywords present
  (recommended: required; inference is fragile between `capillary` and `waveguide`)?
- **Q3.** S14's open rows: align the `atol` defaults, and give `prop_gnlse` the gas-route
  `raman` default semantics? Recommended: document both now, and make any change before
  recipe format 1 is published.
- **Q4.** Should `solprop resume` also accept a recipe edited after the run (for example a
  longer `length`), refusing only when the hash differs in more than the run's end?
  Recommended: no in this SEP; extending a finished run is its own question.
- **Q5.** Which pulse-retrieval formats does `load_retrieval` support first?
- **Q6.** Is `tomli-w` acceptable as a dependency instead of a subset writer?
- **Q7.** Should the metadata include a recommended display unit per keyword (fs for
  durations, µm for radii), or is that left to front ends?

### 13. References

1. TOML v1.0.0 specification, https://toml.io/en/v1.0.0
2. Python Software Foundation, `tomllib` — Parse TOML files,
   https://docs.python.org/3/library/tomllib.html

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
