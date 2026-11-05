# Recipes and the command line

A **recipe** is a simulation written down as a TOML file: the keywords of one route's
call, with each quantity in its unit, and the settings the run executes with. solprop
reads it, checks it, builds and runs it, writes it back, and exports it as the plain
Python script it stands for. The same file runs from a script, from the `solprop`
command, from a batch job or from a front end, and a recipe and the call it stands for
build the same simulation, configuration hash included.

Recipes describe the keyword routes `gnlse`, `capillary` and `waveguide`, and the
free-space routes `radial`, `free` and `planar`, whose keywords are objects
([below](#free-space-routes)). They are specified in
[SEP 41](../../dev/seps/archive/sep-0041-recipes-and-front-ends.md).

## A recipe

```text
[recipe]
format = 1
route = "capillary"
title = "RDW in a helium capillary"

[build]
radius = "125 um"
length = "1 m"
gas = "He"
pressure = ["1.2 bar", "0 bar"]  # entrance to exit: a gradient to vacuum
wavelength = "800 nm"
wavelength_limits = ["100 nm", "1500 nm"]
trange = "200 fs"
fwhm = "7.5 fs"
energy = "275 uJ"
modes = 2
n_save = 51
seed = 20261027

[execution]
fields = "planes"
field_planes = [0, -1]
```

This is the first lesson's run ([resonant dispersive waves](../../introduction/lessons/plot_01_rdw.rst)),
shipped as `Recipe.example("rdw")`. A recipe has five tables and nothing else at the top
level, so a keyword added to solprop later can never collide with the format:

| Table | Holds |
|---|---|
| `[recipe]` | `format` and `route`, required; `title` and `solprop`, informative |
| `[build]` | the keywords of the route's `Simulation.from_*` constructor, under their own names |
| `[execution]` | `fields`, `field_planes` and `resume_history`: settings the run reads when it executes |
| `[pins]` | values solved from a design target (a later release) |
| `[scan]` | a scan over parameter paths (a later release) |

The names in `[build]` are the keywords of
{py:meth}`~solprop.driver.simulation.Simulation.from_capillary`,
{py:meth}`~solprop.driver.simulation.Simulation.from_gnlse`,
{py:meth}`~solprop.driver.simulation.Simulation.from_waveguide`,
{py:meth}`~solprop.driver.simulation.Simulation.from_radial`,
{py:meth}`~solprop.driver.simulation.Simulation.from_free` and
{py:meth}`~solprop.driver.simulation.Simulation.from_planar`, with their meanings and
defaults. A keyword left out takes solprop's default when the recipe is built, so a recipe
follows solprop's improvements; the output records what ran. Keys that choose files,
terminals or resume behaviour (`output`, `progress`, `progress_period`, `resume`,
`snapshots`, `tdse_archive`) are refused in a recipe: the caller or the command line
supplies them. An unknown key is refused with the nearest valid one.

## Values

- **Quantities.** A number is read in the keyword's own unit, which is SI except for
  pressures, which are in bar ([conventions](../../dev/conventions.md)): `pressure = 1.2`
  is 1.2 bar. A string is a decimal number and a unit, `"125 um"`, `"7.5 fs"`,
  `"2e14 W/cm^2"`, converted exactly in decimal and rounded once, so `"800 nm"` is the
  same double as the literal `800e-9`. The unit's dimension must be the keyword's:
  `"125 um"` for a pressure is refused. Write `um` for micro (`µm` is read too); a
  temperature is in K only. Elements of `phases` carry their own units (rad, s, s², ...),
  and so do those of `betas` (sⁿ/m by order).
- **None.** Leaving a key out gives its default; where that default is not `None` but the
  keyword takes `None` (`noise`, `boundary_check`), the string `"none"` says `None`.
- **Complex numbers** are `{re = 1.45, im = 0.0}`; an integer beyond 64 bits, such as a
  drawn noise seed, is a decimal string.
- **Arrays** are lists of at most 10 000 numbers. A longer or a two-dimensional array is a
  data file, `{file = "betas.npy", sha256 = "…"}`, read without pickling and checked
  against its hash. The path is relative to the recipe, and must stay inside its
  directory unless the caller allows otherwise.
- **Objects** are tables with a `type` key naming a registered type, and that type's
  keywords: `pulses = [{type = "SechPulse", wavelength = "835 nm", width = "100 fs",
  peak_power = "9.8 W"}]`. The registered types are the pulses (`GaussianPulse`,
  `SechPulse`, `load_spectrum` for a [measured spectrum](pulses.md#measured-spectra-from-a-file)),
  the pressure profiles (`PressureGradient.two_point`, `.from_stages`, `Constant`, `Linear`,
  `PiecewiseLinear`), the mixtures (`Mixture.fill`, `.partial_pressures`, `.air`), the
  grids (`EnvelopeGrid.from_limits`, `FieldGrid.from_limits`), `StatsSpec`,
  `BoundaryCheck`, `FixedSchedule.uniform`, the integrators (`Multistep`, `RungeKutta`,
  `LowStorageRungeKutta`), `absorber` (built on the route's grid), `TubeLattice`,
  `CoherentWall`, `MaterialSelection`, the supplied modes (`MarcatiliMode`,
  `RectMode`, `SlabMode`, `ModeCollection.slab`), `StepIndexMode` as the GNLSE `fibre`,
  with `glass` for its materials, the graded-index guide (`ModalGuide.on_grid`,
  `GradedIndexProvider`, `PowerLawPermittivity`, `IndexMaterial`, `MaterialKerr`), and the
  free-space types: the grids
  (`RadialGrid`, `FreeGrid`, `LineGrid`, `PlanarGrid`), `SpatialMedium`, the responses
  (`KerrResponse`, with `gamma3` for optmat's coefficient of a gas,
  `EnvelopePlasmaResponse`, `PlasmaResponse`, `OptmatRate`), the beams
  (`SpatialPulse`, `PlanarPulse`), `SimulationSettings`, `PIController`,
  `default_spec` and `SpatialAbsorber`. A recipe never imports a module or evaluates
  code.

**Supplied modes.** On the `waveguide` route, and for the capillary's `modes`, a mode is
a table of its geometry, cladding and loss. Its fill is the route's `gas`, `pressure` and
`temperature`, which every supplied mode is rebound to, so a mode table names none: the
planar guide of [hollow planar waveguides](../waveguides/planar.md), per metre of y, is

```text
[recipe]
format = 1
route = "waveguide"

[build]
modes = { type = "ModeCollection.slab", half_gap = "63.5 um", orders = [1, 3, 5] }
length = "10 cm"
gas = "Ar"
pressure = "1 bar"
wavelength = "800 nm"
wavelength_limits = ["650 nm", "1000 nm"]
trange = "1.2 ps"
fwhm = "100 fs"
energy = 0.5      # J per metre of y, on a slab set
envelope = true
plasma = false
raman = false
```

and a list of tables gives the modes one by one, `modes = [{type = "RectMode",
half_width_x = "50 um", half_width_y = "10 um", polarisation = "y"}]`. A guide that
solves its own fields (`provider`, the [general modal route](../waveguides/general-modal.md))
is written in Python, except a [graded-index fibre](../waveguides/graded-index.md):
`provider = {type = "ModalGuide.on_grid", knots = 21, n_r = 2, provider = {type =
"GradedIndexProvider", ...}}`, with `modes = []` and a `grid` table, since its knots are
taken from that grid. `check()` on such a recipe costs one solve of the guide.

**Parameter paths** name a value inside the recipe: the keyword, then `.name` for a
keyword of an object, `[i]` for an element of a list and `["key"]` for a key of a mapping,
as in `pressure`, `pulses[1].energy` or `betas[2]`. {py:meth}`Recipe.get
<solprop.recipe.core.Recipe.get>` and {py:meth}`Recipe.set <solprop.recipe.core.Recipe.set>` take
them, and so does every error a recipe or a route raises about a value, in its
`parameter` attribute.

## Free-space routes

On `radial`, `free` and `planar` the constructor takes objects, and a recipe writes each
as a table under the keyword's name: `grid` (the time grid), `space` (the transverse
grid), `medium` (not on `planar`, whose slab channels carry the fill), `responses`,
`initial` (the beam) and `settings`. The
[Cartesian tutorial](../../auto_tutorials/free_space/plot_02_cartesian.rst) is
`Recipe.example("cartesian_beam")`:

```text
[recipe]
format = 1
route = "free"

[build]
grid = { type = "EnvelopeGrid.from_limits", length = "20 mm", reference_wavelength = "800 nm", wavelength_limits = ["400 nm", "2 um"], trange = "200 fs", thg = false }
space = { type = "FreeGrid", radius_x = "6 mm", n_x = 32, radius_y = "7.5 mm", n_y = 40 }
medium = { type = "SpatialMedium", material = "Ar", pressure = "4 bar" }
responses = [
    { type = "KerrResponse", chi3 = { type = "gamma3", gas = "Ar", wavelength = "800 nm" }, thg = false },
]
noise = "none"

[build.initial]
type = "SpatialPulse"
temporal = { type = "GaussianPulse", wavelength = "800 nm", fwhm = "30 fs", energy = "1.5 nJ" }
waist = "2 mm"

[build.settings]
type = "SimulationSettings"
controller = { type = "PIController", rtol = 1e-6, atol = 1e-13 }
save_planes = ["20 mm"]
dz_init = "2 mm"
dz_max = "2 mm"
stats = { type = "StatsSpec", names = ["energy"] }
```

A recipe holds the `from_*` form, so the temporal absorber and its scheme are keywords
of the `settings` table, `absorber = { type = "absorber" }` built on the `grid`, and not
of `[build]`, although `prop_radial`, `prop_free` and `prop_planar` take them as
keywords too. The settings table takes the stepping, the planes, the statistics, the
absorber, the backend and the precision; `fields`, `field_planes` and `resume_history` stay
in `[execution]`, and the fields a route's builder sets (the density, the
transverse quadrature) are not the recipe's. On the `planar` route a slab mode names its
fill, `{ type = "SlabMode", half_gap = "63.5 um", gas = "Ar", pressure = "1 bar" }`,
because the fill of its channels is the run's; on the `waveguide` route the same
keywords are refused, since every supplied mode is rebound to the route's fill. A
launch built as an array, a `Frame` and a design pin have no table form on these routes.

## From Python

```python
from solprop.recipe import Recipe

recipe = Recipe.example("rdw")
recipe.set("energy", "250 uJ")  # checked against the keyword's unit and range
print(recipe.check())  # problems, each at its parameter; [] here
script = recipe.to_python("rdw.h5")  # the plain script this recipe stands for
simulation = recipe.build()  # a Simulation, not yet run
```

`Recipe.load` reads a file and `Recipe.loads` text; `recipe.save` writes it. A recipe that
has not been edited saves its text verbatim, comments included; an edited one is written
canonically, each quantity in its keyword's display unit, without comments.
`Recipe.from_keywords("capillary", radius=125e-6, ...)` writes a call made of plain
values as a recipe, and refuses an object or a function, which have no table form: a
recipe is built into objects, never made from them. `recipe.run(output)` builds and runs.

`recipe.check()` decodes every value against the keyword's metadata, then builds the
simulation and translates it, stopping before a backend is built: a refusal that depends
on several values (a grid that does not cover the pulse, two conflicting keywords) is
reported there, with the keywords concerned, and so is every warning solprop gives. It
costs a translation, a fraction of a second on the capillary example.

**What a recipe cannot hold**: a callable (a `beta` function, a loss curve, a custom
absorber's rate), a type that is not registered (a tabulated index or a mode other than
`StepIndexMode` as the GNLSE `fibre`, `CustomPulse`, a TDSE response, a material object as a mode's cladding or
a medium, a guide built from another provider, an `ApertureSet`, a Raman or dispersive Kerr
response on a free-space route), and an array: one given to `Simulation.with_input`, or
a free-space launch built as a spectrum. `to_python()` is the way out: export the recipe
and edit the script.

**A grid to start from.** {py:func}`solprop.grid.suggest` proposes `wavelength_limits`
and `trange` from the input pulses: a band ten input bandwidths wide each side of the
carrier, and at least half the carrier frequency on the `gnlse` route or all of it on the
`capillary` route, and a window thirty input durations long beyond the pulses' delays. It
knows nothing of the medium or the length, so it is a starting point: check the spectrum
at the band's edges and the boundary monitor's verdict.

## Pins

A `[pins]` table sets a value from a design target instead of a literal. The pressure
that puts the zero-dispersion wavelength at 500 nm, and the energy that makes the pulse a
fourth-order soliton there:

```text
[pins]
pressure = {target = "zdw", wavelength = "500 nm", search = ["0.01 bar", "20 bar"]}
energy = {target = "soliton_order", order = 4}
```

| Target | Keys, beside `search` and `root` | Pins |
|---|---|---|
| `zdw` | `wavelength` | the guide or fill: `radius` or a constant `pressure` |
| `rdw` | `rdw`; optionally `peak_power` with `fwhm`, `shape`, `nonlinear_phase` | the same; the pump is the recipe's `wavelength` |
| `soliton_order` | `order` | a pulse's `energy` or `peak_power`, as `energy` or `pulses[0].energy` |

A pin on the guide or fill needs `search`, the range its solver
({py:func}`solprop.tools.solve_parameter`) samples, in the keyword's unit; it is solved
on the launch mode (the first) at the entrance. A soliton-order pin is closed form, on
that mode with the fill's Kerr coefficient, after the guide's pin. So at most one
parameter of the guide or fill is pinned, then the pulses; a pinned value is not also
written in `[build]`, and a pressure gradient, or one element of an array, cannot be
pinned. `root` chooses among several roots: `"lowest"` (the default), `"highest"` or an
index.

`recipe.resolve()` returns the recipe with each pin replaced by its value, and
`recipe.pin_roots("pressure")` the solver's {py:class}`~solprop.tools.DesignRoots`: every
root, or none with the range the target does reach, for a front end to offer. A target
not met in its range is an issue of `check()` on the pinned path. A build solves the pins
(each scan point solves its own), a run records both the authored recipe and the
resolved one, and `to_python()` writes the solver calls
({py:func}`~solprop.recipe.pins.solve_pin`,
{py:func}`~solprop.recipe.pins.solve_pulse`) before the route's. Solving a guide's pin
costs a few hundred evaluations of one mode, about a second for a capillary. A design that
sets two values at once, such as {py:func}`~solprop.tools.capillary_design`, is not a pin:
call it and write its values in `[build]`.

## Previews

Before a run, `recipe.modes()` returns the mode set a guided route builds (the
`capillary` and `waveguide` routes, the slab channels of the `planar` route and the
`gnlse` route's one specified mode, without a fill), at the
launch position or at `z`: each mode's label, its effective index (`neff_minus_one`),
`beta`, `beta1`, `beta2`, `beta3` and loss over the grid's band, its effective area (a
slab mode's effective width), its nonlinear coefficient `gamma` where its fill has Kerr
data (with the fill's `n2` at the grid's reference wavelength), and its
zero-dispersion wavelength in the band. It also gives each mode's phase matching to a
soliton at the pump `wavelength` in the launch mode, the set's first: the
`phase_mismatch` of a dispersive wave in that mode over the band, and the phase-matched
wavelengths `rdw` between the band's short edge and 0.9 of the pump (intermodal for every
mode but the first). With an unchirped sech or Gaussian pulse above the soliton
threshold, in anomalous dispersion, and a nonlinear coefficient (the recipe's `gamma`, or
the fill's Kerr data at the preview's position), the soliton's nonlinear wavenumber is
its largest constituent's, and the set says so (`nonlinear`, with `pump`); otherwise it
is the linear condition. Its cost is the mode
solve. `recipe.preview_input()`
returns the input pulse on the run's grid as the launch sets it, without noise, so a
preview draws nothing: its power in time, its spectrum, its energy and its peak power,
read through the processing functions that read a run's file. On a free-space route it
gives the beam: the power and energy through the launch plane in the medium, and the
fluence there on the transverse grid with its axes (`fluence`, and `axes` as `(r,)`,
`(x,)` or `(y, x)`), measured as a saved plane is
({py:func}`~solprop.spatial.spatial_fluence`). Both solve the pins first.

## Scans

A `[scan]` table varies parameter paths over a grid of points. `name` is required: it
names the scan's files and derives each point's seed; `base_seed` may set the root of
those seeds. Every other key is a path with a list of values, or a range
`{start, stop, num, spacing = "linear" | "log"}`, read in the unit of the keyword the
path ends on:

```text
[scan]
name = "rdw-pressure"
pressure = {start = "0.6 bar", stop = "1.4 bar", num = 9}
energy = ["200 uJ", "275 uJ"]
```

`recipe.scan()` returns the {py:class}`~solprop.driver.scan.Scan`, a builder that sets each
point's values, passes the point's seed and builds (a picklable one, for the `Pool`
executor), and the keywords `run_scan` takes from `[execution]`:

```{code-block} python
scan = recipe.scan()
run_scan(scan.scan, scan.build, output="scan.h5", **scan.keywords)
```

`run_scan` takes `fields` and `field_planes`. The table's other setting, `resume_history`,
goes to each point's run.

## The recipe in the output

A simulation built by `recipe.build()` carries its recipe, and its first run records it
in the output beside the run, as the recipe was written and as it resolved: the recipe
that rebuilds the run, seed included ([the file's layout](../results/output.md#the-recipe-a-run-was-built-from)).
A change the recipe would not describe drops it with a warning, and the run then records
none: `with_input`, `with_mixture`, or a new value of a setting other than `fields`,
`field_planes` and `resume_history`.

## The parameter metadata

{py:func}`solprop.recipe.metadata.parameters` returns, for each keyword of a route, the forms its
value may take with their units and ranges, the unit a front end shows it in, its group
(`geometry`, `medium`, `nonlinear`, `input`, `grid`, `numerics`, `output`), whether a
newcomer sees it, a one-line summary and the manual section that documents it. Where a
default of `None` is resolved by a rule, `resolver(values)` says what it will be and why:

```python
from solprop.recipe import parameters

rtol = parameters("capillary")["rtol"]
print(rtol.resolver({"precision": "single"}))  # the capillary's tolerance in binary32
```

The same call takes a registered type's name for the keywords of its table, so a second
pulse, a pressure gradient, a mode or a free-space grid is described as a route keyword is:
`parameters("GaussianPulse")["fwhm"]` has the unit `s`, the display unit `fs` and the
summary a front end's field shows. A type's table leaves out what the type receives from
the route (the absorber's time grid) and the keywords a recipe may not give.

A test holds the tables to the constructors' signatures and to the registry's units, so
they cannot drift from them.

## What a route and a species admit

{py:func}`solprop.capabilities.route` gives a route's keywords, the allowed values of each
enumerated one, and the **combination rules** its builder enforces: `raman` needs `kerr`,
a TDSE response needs a helium fill, `save_planes` and `n_save` are not given together.
Each rule has a stable id and a reason, and a refusal that enforces one carries the id as
its `rule` beside the keywords as its `parameter`, so a front end can mark the fields in
conflict and say why. A refusal that depends on a value, such as a grid that does not
cover the input or a pressure outside a model's range, is not a rule: `check()` finds it.
{py:func}`solprop.capabilities.species` gives what a built-in gas brings: optmat's
ionisation-rate models with their status, whether it has a Raman response and its
status, and the wavelength and temperature ranges its models hold over.

```python
from solprop import capabilities

capillary = capabilities.route("capillary")
print(capillary.choices["shape"])  # ('gauss', 'sech')
for rule in capillary.rules_for("raman"):
    print(rule.id, rule.keywords, rule.reason)

nitrous_oxide = capabilities.species("N2O")
print(nitrous_oxide.raman, nitrous_oxide.raman_by_default)  # provisional False
```

A test reads the source for every refusal marked with a rule and holds it to
{py:data}`solprop.capabilities.RULES`.

## The command line

```text
solprop run rdw.toml -o rdw.h5 [--progress log] [--resume]
solprop check rdw.toml              # print the issues; exit 1 on an error
solprop export rdw.toml --python    # print the equivalent script
solprop resume rdw.h5               # continue a run from its file alone
solprop scan ...                    # python -m solprop.scan, unchanged
```

`python -m solprop` is the same command. `check` exits 0 when the recipe builds (warnings
included), 1 when it does not, and 2 when the file cannot be read. `run` and `resume` stop
cleanly on Ctrl-C, writing the state, and `solprop resume` continues it
([resuming from the file alone](../results/resuming.md#resuming-from-the-file-alone)).

## The example recipes

`Recipe.examples()` lists the recipes solprop ships, each the run of a gallery example,
and tested to build that example's configuration hash. A front end starts a new setup
from one of them. Every route has one: `hollow_rectangular` on `waveguide`,
`radial_kerr` on `radial`, and `cartesian_beam`, `planar_hybrid` and `planar_compressor`
on the other free-space routes. The solid-core fibres are `step_index_fibre` on `gnlse`
and `graded_index_fibre` on `waveguide`.
Each gallery script on a route a recipe describes that has no recipe is listed, with
its reason, in the test that holds the examples to the gallery, so the list stays true:

- the `waveguide` scripts, which build guides from providers, launch through
  `Simulation.with_input` or clad a guide in a metal, and the tabulated-mode tutorial,
  whose modes are not registered;
- the radial tutorial, the filament and crossing-beam lessons, and the line,
  low-storage, four-wave-mixing and far-field tutorials, which launch a beam built as
  an array;
- the transverse-window and capillary-coupling tutorials, whose beam energy is
  computed from the material functions, which a literal in a recipe would freeze;
- the two dispersive-Kerr scripts, whose pressure is a numerical phase-matching root,
  and the pressure-and-energy scan, built with `with_mixture` and `with_input`.

This page documents the functionality proposed and implemented by
[SEP 41](../../dev/seps/archive/sep-0041-recipes-and-front-ends.md).
