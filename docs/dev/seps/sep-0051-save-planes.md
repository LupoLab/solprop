# SEP 51 — One rule for the saved planes and the end of a run

| | |
|---|---|
| **SEP** | 51 |
| **Title** | One rule for the saved planes and the end of a run |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-15) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Change |
| **Created** | 2026-11-15 |
| **Resolution** | — |
| **Requires** | — |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Today a run's saved planes are written three different ways, depending on whether it is
started from a keyword interface, from objects or from a recipe, and the end of the run
is the last saved plane, never the length the run was set up for. This SEP gives every
route the same two parameters. `n_save` is a count of planes including the entrance, and
`save_planes` is a list of positions; a run takes one or the other, never both. It also
makes the length the upper limit that decides how far the run goes: the route's `length`
on a keyword route, `grid.length` on an object route. Success means one placement rule in
one helper, the same refusals on every route, recipe, Python export and front-end sheet,
and unchanged numbers for every run that already ends at its length, on `numpy_ref` and
on the native CPU build.

### 2. Motivation and scope

The trigger was [#533](https://github.com/LupoLab/solprop/issues/533). A recipe with
`save_planes = 80` in an object route's `[build.settings]` passes `Recipe.check()` and is
read as one plane at 80 m. The run then marches until something unrelated stops it.
Investigating it showed that no route ties the run's end to its length:

- On a **keyword route**, `length` builds the grid and sets the default steps. The run
  still ends at the last of the explicit `save_planes`, and positions beyond `length` are
  accepted. With `length = 1 mm`, `save_planes=[0.5 mm]` stops at 0.5 mm, and
  `[0.5 mm, 2 mm]` runs to 2 mm on a grid sized for 1 mm.
- On an **object route**, `grid.length` sizes the collar and the window, but the run ends
  at the last save plane, whatever `grid.length` says.

In scope:

- the meaning of `n_save`, `save_planes` and the run's end on every route that runs: the
  keyword interfaces, `Simulation(...)`, `Simulation.from_free`, `Simulation.from_planar`,
  recipes, scans and the front end;
- the refusals and their messages, and the migration;
- how the end reaches the backends.

Not in scope:

- continuing a finished run to a longer length
  ([#141](https://github.com/LupoLab/solprop/issues/141));
- `fields=` and `field_planes=` on the `prop_*` functions
  ([#311](https://github.com/LupoLab/solprop/issues/311));
- any change to stepping, tolerances or physics defaults;
- `FixedSchedule`'s own construction (`FixedSchedule.uniform` and replay), which keeps its
  interface; only how its end relates to the run's length is settled here.

### 3. Context

**Every way save planes are given today:**

| Where | Count | Positions | End of the run | Beyond the length |
|---|---|---|---|---|
| Keyword routes (`prop_gnlse`, `prop_capillary`, `prop_waveguide`, `Simulation.from_gnlse`/`from_capillary`/`from_waveguide`, the general modal route) | `n_save`, counting `z = 0`; default `DEFAULT_N_SAVE = 201` | `save_planes`, m, `z = 0` implicit; exclusive with `n_save` | last plane: `length` for a count, the last position otherwise | accepted |
| Object routes (`Simulation(...)`, `from_free` radial and Cartesian, `from_planar`) | none | `SimulationSettings.save_planes`, required | last plane; `grid.length` is not consulted | accepted |
| Recipes, keyword routes | `n_save` | `save_planes` (resolvable to positions) | as the keyword route | accepted |
| Recipes, object routes | none; `save_planes = 80` passes `check()` and is read as one plane at 80 m (#533) | `[build.settings] save_planes` | as the object route | accepted |
| Front end, "Saved planes" | keyword sheets write `n_save`, counting the entrance | object sheets expand a count that **leaves out** the entrance into a list over `grid.length` (`even_planes` in `gui/src/solprop_gui/classes/problem.py`), and rewrite the list when the length changes | as the route | — |
| TDSE capillary (`tdse=`) | given neither, the run saves the entrance and the end (`n_save = 2`) and every natural step (`step_planes = 1`) | as the keyword route | as the keyword route | accepted |

**Related mechanisms that read the planes:**

- **`step_planes = k`** also saves the end of every `k`-th accepted step, beside the
  requested planes.
- **`field_planes`** gives indices into the saved planes: the entrance is 0 and negative
  indices count from the end. It is refused beside `step_planes`.
- **`interpolate_planes`** observes planes inside steps, except the run's end and the
  hard kinks.
- **`FixedSchedule`** has its own `z_start` and `z_end`, and its save planes must lie in
  `[z_start, z_end]`. A schedule that ends before the last plane fails at run time ("the
  replayed schedule ends at … and cannot reach …"). One that ends after it is accepted,
  and its tail is never run.
- **Interior launches.** `Simulation(initial_z=...)` already requires
  `initial_z < grid.length` and every plane to follow it (`_check_initial_position` in
  `src/solprop/driver/inputs.py`). `Simulation.with_input(..., initial_z=z)` keeps the
  *remaining* planes, those beyond `z`.
- **Resume and the configuration hash.** The configuration records the positions
  (`configuration["save_planes"]`), so a count and the same positions hash alike, and a
  resume continues to the stored last plane.
- **Scans** count `len(settings.save_planes) + 1` planes per point.

**The seam.** `assemble` (`src/solprop/translate/assembly.py`) "covers
`[initial_z, max(save_planes)]`". `StepperSettings.z_end`
(`src/solprop/translate/problem.py`) is `max(save_planes)`. No run end separate from the
planes crosses the seam, in `RunRequest` or in `PropagationProblem`.

**Issues:** this SEP fixes [#533](https://github.com/LupoLab/solprop/issues/533). It
borders on [#141](https://github.com/LupoLab/solprop/issues/141), which continues a
finished run, and [#311](https://github.com/LupoLab/solprop/issues/311), which concerns
`prop_*` and `field_planes`; it changes neither.

### 4. Theory

None. No equation, approximation or default of the physics changes. The only change is
which positions are saved and how far the run goes.

### 5. Proposed interface

**Decided by the maintainer (2026-11-15):**

1. **Length is the limit and the end.** Every run marches from its launch to its length:
   `length` on a keyword route, and `grid.length` on an object route, where the keyword
   route's `length` is the same number, because it builds the grid. Saved planes are
   chosen inside `(launch, length]`.
2. **One pair everywhere.** Every route takes the same two parameters:
   - `n_save`: a count of planes **including `z = 0`**, at least 2, placed evenly over
     `[0, length]`;
   - `save_planes`: positions in m, increasing, in `(0, length]`, with `z = 0` implicit.

   The two are mutually exclusive. Given neither, a run saves `DEFAULT_N_SAVE = 201`
   planes. `SimulationSettings` gains `n_save`, and its `save_planes` becomes optional.
3. **No bare numbers.** A bare integer in `save_planes` is refused with "pass
   `n_save=<int>`", and a lone float with "pass `[<value>]`, or `n_save`". This holds in
   Python and in recipes: `save_planes = 80` in `[build.settings]` gives a `RecipeError`
   suggesting `n_save = 80`. The integer form leaves the `save_planes` metadata, and
   `n_save` joins the object routes' settings metadata.
4. **Recipes, export and front end.** Recipes and the Python export write `n_save` and
   `save_planes` as given. The front end's "Saved planes" means `n_save` on every sheet,
   counting the entrance, and it writes `n_save` rather than expanding positions, so a
   length change no longer rewrites a list.

**Resolution.** Settings do not know the length. `n_save`, or the default, is therefore
resolved into positions where a simulation adopts its settings and the grid is known.
After adoption, `simulation.settings.save_planes` holds the resolved positions and
`n_save` is `None`. Everything downstream (resume, `with_*`, metadata, the hash,
provenance, scans) keeps reading positions, and a count still hashes like its positions.
One private helper holds the placement rule; the keyword routes' `_save_planes` in
`src/solprop/driver/settings.py` becomes it.

**How the seam carries the end (recommended, Q1 option A).** The driver guarantees that
the last resolved save plane *is* the length, so `StepperSettings.z_end =
max(save_planes)` is the run end with no new field. `assemble` and below keep their
meaning, and `RunRequest`, `PropagationProblem` and the frozen tables of
`tests/backends/test_driver.py` are unchanged. A caller of `assemble` directly, below the
driver, still ends at its last plane.

```text
Simulation.from_free(grid, space, medium, responses, beam,
                     settings=SimulationSettings(n_save=81))      # 81 planes over grid.length
SimulationSettings(save_planes=[0.05, 0.1])                       # grid.length = 0.1: two planes and the entrance
SimulationSettings(save_planes=80)        -> DriverError: "... pass n_save=80"
SimulationSettings(save_planes=[0.3])     -> DriverError at adoption: "0.3 m is beyond grid.length = 0.25 m"
```

```text
[build.settings]
type = "SimulationSettings"
n_save = 81                # was a list of 80 positions in the filament recipe
```

### 6. Implementation strategies

- **Layer.** The driver and the recipe layer, plus the front end. The physics and backend
  layers are untouched, and so is the seam under option A of Q1.
- **Where the rules apply.** In one function that the `Simulation` constructor calls for
  the settings it adopts, and that the keyword routes call before they build their
  `Simulation`. Settings that are already resolved, copied by `with_input`,
  `with_mixture`, a reseed or a scan point, pass through unchanged.
- **Backends.** Nothing backend-specific. `numpy_ref` and the native CPU build run the
  same requests as today for every run that ends at its length.
- **Cost estimate** (`poe size`):

  | | Added | Removed |
  |---|---|---|
  | Code | about 150 lines: the resolver and refusals, `n_save` on settings, the recipe metadata and export, the front-end field | about 60 lines: the front end's plane rewriting and list expansion |
  | Docstrings | about 80 lines | — |
  | Tests | about 250 lines: the refusal table, count/positions parity, the end rule per route, recipe and export round trips, the front-end field | — |

  Separately, an estimated 10–30 existing tests and snippets need edits (see Q7). Option
  B of Q1 would add about 150 lines of code and a rewrite of the frozen seam tables.
- **Dependencies:** none.

### 7. Verification and validation

- **Refusals.** A table of every refused form on every route kind, Python and recipe,
  each with its message: a bare integer, a lone float, both parameters given, a count
  below 2, positions not increasing, positions beyond the length, and a schedule that
  does not end at the length.
- **Parity.** `n_save=N` and the positions it resolves to give identical settings, hash
  and output. This is an exact structural check, since both resolve to the same array.
- **Unchanged numbers.** For runs that already end at their length, the request is
  bit-identical, compared on its frozen fields. A short run (about ten steps) per route
  kind on `numpy_ref` and native CPU matches its pre-change output within the rounding
  budget of SEP 25.
- **End rule.** Per route kind, the last saved plane equals the length, the plane count
  is as specified, and `with_input(initial_z=…)` keeps the remaining planes and the end.
- **Recipes and front end.** Round trips of `n_save` through `Recipe.check()`, build and
  export, and the front end's field writing `n_save`.

### 8. Documentation

- **Manual pages:**
  - `guide/setup/running.md`: the one rule, and the end of a run;
  - `guide/setup/recipes.md`;
  - the free-space and planar pages that use the `np.linspace(0, length, N)[1:]` idiom,
    which becomes `n_save` where natural;
  - the front end's page on saved planes.
- **Docstrings:** `SimulationSettings`, `Simulation`, `from_free`, `from_planar`, the
  keyword interfaces and the recipe metadata.
- **Changelog:** a `.changed` fragment with the migration, and a `.fixed` fragment for
  #533.
- **Examples:** the filament and crossing-beam recipes and galleries may switch to
  `n_save`; their output is unchanged.

### 9. Compatibility and impact

This is a breaking change to the public interface. Every break either is refused with a
message naming the fix or is listed in the changelog (Q7). Under option A the output
schema, kernel contract and frozen seam tables are unchanged. Validated results are
unchanged wherever a run already ends at its length. No deliberate difference from Luna.jl
is introduced.

### 10. Alternatives considered

- **Keep positions as the end, and only check against `grid.length`.** This was the first
  fix for #533. The maintainer rejected it because two conventions would remain.
- **Read an integer `save_planes` as a count.** This overloads one name with two meanings,
  and the keyword routes already spell the count `n_save`.
- **Make the run end a separate field of the seam.** This is option B of Q1.

### 11. Phases

1. **Driver and recipes.**
   - Deliverables: the resolver, `n_save` on settings, the refusals, the end rule on
     every route, the recipe metadata and export, the tests and edits, the manual pages
     and the changelog.
   - Survey: run the affected test modules with an instrumented adoption to count every
     run whose end moves, before editing (Q7).
   - Verify: §7 on `numpy_ref` and the native CPU build.
2. **Front end.**
   - Deliverables: "Saved planes" as `n_save` on every sheet, written as `n_save`, with
     the list rewriting removed; the front-end gate.

Both phases are small, and could merge as one pull request if the maintainer prefers.

### 12. Open questions

Each question below gives the options and a recommendation; the maintainer decides.

- **Q1 — Explicit positions that stop short of the length.**
  - A: the end is always saved, appended when it is not listed, the way `z = 0` is
    always saved.
  - B: the run marches on to the length without saving it. This needs a run end
    separate from the planes in `RunRequest` and `assemble`: a seam change with frozen
    tables and a release note.
  - C: refuse positions whose last entry is not the length.

  *Recommended: A.* It is symmetric with the implicit entrance, needs no seam change, and
  computing a stretch that is never saved has no use. `save_planes=[length]` then means
  "only the end", as it does now.

- **Q2 — Positions beyond the length.**
  - A: refuse them on every route, keyword routes included, naming the length:
    "z = 0.3 m is beyond the run's length 0.25 m; raise length (or grid.length)".
  - B: refuse them only on object routes.

  *Recommended: A.* Consistency was the reason for this SEP, and a plane beyond the
  length sits on a grid and collar sized for a shorter run.

- **Q3 — `FixedSchedule`'s own end.**
  - A: `z_start` must equal the launch and `z_end` the length, within a few ulps of the
    length. This is refused at translation, with "the schedule ends at … and the run at
    length …".
  - B: require only `z_end ≥ length`, so a longer schedule's tail is silently unused.

  *Recommended: A.* It turns today's run-time "cannot reach" failure into an early
  refusal, and a schedule that ends past the run would be a second, contradicting
  statement of the length.

- **Q4 — The TDSE default.** Given neither parameter, a TDSE run saves the entrance, the
  end and every natural step.
  - A: keep it, stated as a route default (`n_save = 2`, `step_planes = 1`) under the
    general rule.
  - B: drop it for the general default of 201 planes.

  *Recommended: A.* A TDSE step is expensive and its natural steps are the useful record.
  Under the new rule the default is just a different count.

- **Q5 — `step_planes` and `field_planes`.**
  - `step_planes` needs no change: natural-step planes stay additional, and under Q1 A
    the end is always among the planes.
  - `field_planes` keeps indices into the resolved planes. Under Q1 A, `-1` is always the
    end, which is more robust than today.

  *Recommended:* the one break to document is a script that listed positions short of
  the length: it gains an end plane, so indices counted from the end (`-1`, `-2`, …) now
  point to different planes.

- **Q6 — Resume, interior launches and `with_*`.**
  - The remaining planes after an interior launch at `z` become "the resolved planes
    beyond `z`", which always include the end.
  - A resume continues to the stored length, which equals the stored last plane.
  - `with_*` copies resolved settings unchanged.
  - A file written before this SEP whose last plane is not its `grid.length`:
    - A: refuse to resume it, saying it predates SEP 51, and suggest `with_input` from
      its last plane;
    - B: resume it to its stored last plane under the old rule.

  *Recommended: A.* Such a file can only come from a script that relied on the old end,
  and keeping the old rule alive for resume alone would be a second rule.

- **Q7 — Migration and blast radius.** The breaks, each with its message:
  1. `save_planes=N` as a bare integer: refused, "pass `n_save=N`", in Python and in
     recipes.
  2. `save_planes=x` as a lone float: refused, "pass `[x]`, or `n_save`".
  3. A position beyond the length: refused, naming the length (Q2).
  4. Positions short of the length: the run now continues to the length and saves it
     (Q1 A). This raises no error and is listed in the changelog.
  5. A schedule that does not end at the length: refused (Q3).
  6. The front end's object sheets: "Saved planes" grows by one, the entrance. Saved
     recipes are unaffected.

  *Blast radius, from a static survey of the tree at 2026-11-15:*
  - **Object-route files.** About 117 files build an object-route simulation with save
    planes: 84 in tests, 15 in the manual and galleries, 4 in benchmarks and 3 in the
    package.
  - **Unchanged.** About half write the end as the length symbolically (`[grid.length]`,
    `[LENGTH]`, `[length]`, a `linspace` ending at the length). Most of the rest write a
    literal equal to the grid's literal length.
  - **Flagged.** A pattern scan flags 11 files as possible mismatches. A manual check
    confirmed two:
    - `tests/backends/test_far_field_statistics.py`: planes at 0.1 mm on a 1 mm grid, so
      the run would end later;
    - `tests/driver/test_initial_z_and_checkpoint.py`: a plane at 1 mm on a 0.1 mm grid,
      which would be refused.

    The others depend on fixtures, construct settings without running them, or call
    `assemble` below the driver, which is unaffected.
  - **Recipe examples.** None of the nine object-route example recipes is affected: each
    ends at its `grid.length`. Neither is any gallery script found.
  - **Keyword routes.** Explicit positions short of `length` will be counted in phase 1.

  Phase 1 replaces these estimates with a counted list before any edit. A fix never
  changes a physics setting: it sets the grid's length to the intended end, or uses
  `n_save`.

### 13. References

None beyond solprop's own documentation and issues, cited where used.

## Part II — Implementation record

## Part III — History

- 2026-11-15: Drafted, after the investigation of #533 found that no route ties the
  run's end to its length. Tracking issue: [#846](https://github.com/LupoLab/solprop/issues/846).
