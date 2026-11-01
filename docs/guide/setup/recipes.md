# Recipes and the command line

A **recipe** is a simulation written down as a TOML file: the keywords of one route's
call, with each quantity in its unit, and the settings the run executes with. solprop
reads it, checks it, builds and runs it, writes it back, and exports it as the plain
Python script it stands for. The same file runs from a script, from the `solprop`
command, from a batch job or from a front end, and a recipe and the call it stands for
build the same simulation, configuration hash included.

Recipes describe the `gnlse` and `capillary` routes in this release; the waveguide and
free-space routes follow. They are specified in
[SEP 41](../../dev/seps/sep-0041-recipes-and-front-ends.md).

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
| `[execution]` | `fields`, `field_planes`, `resume_history` and `memory_budget`: settings the run reads when it executes |
| `[pins]` | values solved from a design target (a later release) |
| `[scan]` | a scan over parameter paths (a later release) |

The names in `[build]` are the keywords of
{py:meth}`~solprop.driver.simulation.Simulation.from_capillary` and
{py:meth}`~solprop.driver.simulation.Simulation.from_gnlse`, with their meanings and
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
  keyword takes `None` (`noise`, `boundary_check`, and `memory_budget` under
  `[execution]`), the string `"none"` says `None`.
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
  `CoherentWall` and `MaterialSelection`. A recipe never imports a module or evaluates
  code.

**Parameter paths** name a value inside the recipe: the keyword, then `.name` for a
keyword of an object, `[i]` for an element of a list and `["key"]` for a key of a mapping,
as in `pressure`, `pulses[1].energy` or `betas[2]`. {py:meth}`Recipe.get
<solprop.recipe.core.Recipe.get>` and {py:meth}`Recipe.set <solprop.recipe.core.Recipe.set>` take
them, and so does every error a recipe or a route raises about a value, in its
`parameter` attribute.

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
recipe is built into objects, never made from them. `recipe.run(output)` builds and runs,
passing `[execution]`'s `memory_budget` to the run.

`recipe.check()` decodes every value against the keyword's metadata, then builds the
simulation and translates it, stopping before a backend is built: a refusal that depends
on several values (a grid that does not cover the pulse, two conflicting keywords) is
reported there, with the keywords concerned, and so is every warning solprop gives. It
costs a translation, a fraction of a second on the capillary example.

**What a recipe cannot hold**: a callable (a `beta` function, a loss curve), a type that is
not registered (a mode or a tabulated index as the GNLSE `fibre`, `CustomPulse`, a TDSE
response), and an array given to `Simulation.with_input`. `to_python()` is the way out:
export the recipe and edit the script.

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
`capillary` and `waveguide` routes), at the launch position or at `z`: each mode's label,
its `beta`, `beta2` and loss over the grid's band, its effective area and its
zero-dispersion wavelength in the band. Its cost is the mode solve. `recipe.preview_input()`
returns the input pulse on the run's grid as the launch sets it, without noise, so a
preview draws nothing: its power in time, its spectrum, its energy and its peak power,
read through the processing functions that read a run's file. Both solve the pins first.

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

A test holds the table to the constructors' signatures, so it cannot drift from them.

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
from one of them.
