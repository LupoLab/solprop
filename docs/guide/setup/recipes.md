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
<solprop.recipe.Recipe.get>` and {py:meth}`Recipe.set <solprop.recipe.Recipe.set>` take
them, and so does every error a recipe or a route raises about a value, in its
`parameter` attribute.

## From Python

```python
from solprop.recipe import Recipe

recipe = Recipe.example("rdw")
recipe.set("energy", "250 uJ")       # checked against the keyword's unit and range
print(recipe.check())                # problems, each at its parameter; [] here
script = recipe.to_python("rdw.h5")  # the plain script this recipe stands for
simulation = recipe.build()          # a Simulation, not yet run
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

## The parameter metadata

{py:func}`solprop.recipe.parameters` returns, for each keyword of a route, the forms its
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

## The command line

```text
solprop run rdw.toml -o rdw.h5 [--progress log] [--resume]
solprop check rdw.toml              # print the issues; exit 1 on an error
solprop export rdw.toml --python    # print the equivalent script
solprop scan ...                    # python -m solprop.scan, unchanged
```

`python -m solprop` is the same command. `check` exits 0 when the recipe builds (warnings
included), 1 when it does not, and 2 when the file cannot be read.

## The example recipes

`Recipe.examples()` lists the recipes solprop ships, each the run of a gallery example,
and tested to build that example's configuration hash. A front end starts a new setup
from one of them.
