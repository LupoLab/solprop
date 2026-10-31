# SEP 41 — Recipes, and the hooks a front end needs

| | |
|---|---|
| **SEP** | 41 |
| **Title** | Recipes, and the hooks a front end needs |
| **Author** | John C. Travers |
| **Drafting agent** | Claude Opus 5.5 (2026-10-31) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
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
  `pulses = [{type = "GaussianPulse", lambda0 = "800 nm", fwhm = "30 fs", energy = "1 mJ"}]`.
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
