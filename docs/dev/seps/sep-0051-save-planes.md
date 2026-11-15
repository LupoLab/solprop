# SEP 51 — One rule for the saved planes and the end of a run

| | |
|---|---|
| **SEP** | 51 |
| **Title** | One rule for the saved planes and the end of a run |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-15) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
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
- 2026-11-15: GPT-6 completed an AI review of Part I at PR #847, commit
  `4d9af1fc480399c7f8e28814a16f32ca2e257f03`. Findings and questions follow;
  Part I is unchanged.

### AI review — 2026-11-15

Reviewer: **GPT-6 (AI), acting as reviewer only.** Read all of Part I, the PR
discussion, #533 and #846, and the relevant driver, translation, schedule, resume,
recipe and front-end code at the commit above. Also read the overlapping draft
[PR #843](https://github.com/LupoLab/solprop/pull/843), which #846 says this work will
supersede. That relationship should be recorded in the proposal's context.

The shared count/positions interface and the proposed Q1 A, Q2 A and Q4 A choices
are reasonable. The points below need answers before acceptance; this review does
not resolve the maintainer's open questions. Priorities are **High** for conflicts
that can change a run or reject an existing valid one, and **Medium** for missing
contracts or qualification. Small construction and arithmetic probes were run on
this tree; their results are stated below. No propagation was run, and no native
or GPU numerical qualification is claimed.

1. **R1 — High: settle the length when a keyword route receives a grid
   (§§3, 5.1, 6, 7; Q2).** The statement that a keyword route's `length` and
   `grid.length` are the same because the route builds the grid has an existing
   exception: the public `grid=` keyword. `_gnlse_grid` in
   `src/solprop/driver/routes/gnlse.py` and `_capillary_grid` in
   `src/solprop/driver/routes/hollow.py` return a supplied grid without comparing
   its length. A construction probe of `Simulation.from_gnlse` with
   `length=1e-4`, a grid of length `1e-3`, and `n_save=2` retained that grid and
   placed the last plane at `1e-4` m. Running the proposed resolver first against
   the keyword length and then against the adopted grid can therefore append a
   different end, or refuse a run the keyword resolver accepted.

   **Question and recommendation:** must the two lengths agree when `grid=` is
   supplied? Recommend an early refusal naming both values, with an explicit
   rounding policy and migration guidance; do not silently choose an end or
   rebuild the grid. Add this break to Q7 and cover supplied grids in Python and
   recipes, including the general modal path that shares `_capillary_grid`.

2. **R2 — High: use the existing accumulated-rounding bound for a replay's
   end (§§7, 9; Q3).** A fixed allowance of "a few ulps of the length" rejects
   valid schedules whose steps accumulate rounding error. The probe
   `FixedSchedule(np.full(160, 1e-3 / 160))` ended 12 ulps below `1e-3` m;
   1,000 equal steps ended 70 ulps above it. Both are admitted by the existing
   `replay_tolerance` in `src/solprop/stepper/schedule.py`, which accounts for the
   step count. This conflicts with the promise to keep valid runs that already
   reach their length unchanged.

   **Recommendation:** state the endpoint comparison in terms of the existing
   replay bound, or derive and qualify a replacement before acceptance. Keep the
   start comparison distinct: it has no accumulated step sum. Require acceptance
   tests for rounding drift on both sides, `uniform` and `from_history`, and
   refusal tests for a genuinely shorter or longer schedule outside that bound.
   Explain where the check lives without changing the promised semantics of
   direct `assemble` callers.

3. **R3 — High: changing the grid length is not a neutral migration
   (§§4, 7, 9; Q7's final paragraph).** The proposed migration says that a fix
   never changes a physics setting and can simply set the grid's length to the
   intended end. However, `_reference_length` in `src/solprop/boundaries.py`
   defaults to `grid.length / boundary_N`; both spectral and temporal rates
   depend inversely on that length. A probe shortening an otherwise identical
   grid from `1e-3` to `1e-4` m increased its default nonzero rates by a factor of
   ten, to rounding. A keyword route can also derive steps and longitudinal
   profiles from its length. These changes can alter results within the retained
   short window, even though no propagation equation is being redesigned.

   **Recommendation:** distinguish a deliberate correction of a mismatched setup
   from preservation of an existing scientific test. For the latter, preserve
   the original absorber rates and relevant profile/step choices explicitly, or
   retain a deliberately short test below the driver where appropriate. Record
   any intentional numerical change and qualify it. Merely replacing the planes
   by `n_save` extends the run; it does not preserve the original short window.

4. **R4 — Medium: define counts for a directly constructed interior launch
   (§§5.1–5.2, 7; Q6).** The proposal requires planes after the launch, but defines
   `n_save` over `[0, length]`, including zero. Q6 explains filtering for
   `with_input`, not construction with `initial_z > 0`. On a 1 mm grid with
   `initial_z=0.4 mm`, a five-plane lattice from zero contains `0.25 mm`; passing
   that lattice to today's constructor raises the documented initial-position
   error. Filtering it gives four saved planes including the actual entrance;
   redistributing five planes over the remaining interval gives five at different
   positions. Applying the default count also needs a decision.

   **Question and recommendation:** does the count describe the whole zero-based
   lattice or the actual launched run? To retain the stated zero-based rule,
   recommend resolving the lattice and retaining only positions beyond the
   launch, documenting the resulting count and actual entrance. Keep an explicit
   position at or before the launch a named refusal. Test both direct
   construction and `with_input`, with an on-lattice and an off-lattice launch.
   Also state whether today's refusal of fixed schedules for interior launches
   remains: Q3's start-equals-launch condition must not imply new support.

5. **R5 — Medium: cover settings replacement and reuse in the adoption
   contract (§§5–7).** Section 6 names the constructor and keyword builders,
   then says already resolved settings pass through unchanged. The public
   `Simulation.settings` setter also accepts replacement settings before
   translation, and settings can be reused with another grid. A resolver only
   called by the constructor would leave `n_save` unresolved on replacement;
   treating a resolved array as exempt from validation could retain the wrong
   end on another grid. The setter also compares settings to protect a translated
   request and to decide whether its attached recipe is still valid.

   **Recommendation:** define adoption to include the setter and validate the
   resolved positions against the receiving run every time. Resolve before
   comparing settings, so an equivalent count and explicit array remain
   equivalent after translation. Document how to replace a resolved selection
   with a count without accidentally supplying both, and test replacement,
   repeated adoption, reuse at a different length and the existing prohibition
   on changing a translated run's schedule.

6. **R6 — Medium: distinguish the declared end from the last committed plane
   when refusing legacy resume (§§7–9; Q6).** An interrupted valid file normally
   has its last committed plane before `grid.length`. Q6's "file ... whose last
   plane is not its `grid.length`" must mean the last *requested* position in
   `configuration["save_planes"]`, not the last row of `z` or a terminal
   checkpoint. `driver/resume.py` already distinguishes those states. In addition,
   `recipe.resume_file` rebuilds the recipe before the resume gate, and
   `output.hdf5.check_cache` checks the hash before opening for append. Appending
   an endpoint during rebuilding will otherwise produce the generic hash
   mismatch instead of the migration message promised here.

   **Recommendation:** specify the stored metadata used for this decision and
   where each resume entry point makes it. Preserve interrupted and checkpointed
   files whose declared end already equals the length. Test legacy ends both
   below and above the length, ordinary interrupted files, completed files and
   recipe-driven resume. Describe `with_input` recovery as a new run into a new
   output, with its launch and restart limitations, rather than an in-place
   continuation of the old file.

7. **R7 — Medium: separate authored recipes from resolved provenance
   (§§5.4, 6–9).** "Recipes and the Python export write ... as given" is suitable
   for authored text. The resolved recipe has a different promise: it rebuilds
   the same run, independently of omitted defaults. Today
   `recipe/provenance.py:resolved_document` writes resolved keyword-route planes
   and removes top-level `n_save`; it does not resolve nested
   `[build.settings]` fields. Once both object-route selectors can be omitted,
   leaving that table untouched leaves its plane choice dependent on a future
   default. Writing both a count and positions would instead fail exclusivity.

   **Recommendation:** explicitly retain authored syntax and export behavior,
   while the resolved record writes only the canonical positions, including the
   end, at the correct level for every route. Add round trips through
   `Recipe.from_output(..., resolved=True)` and `resume_file`, for a count,
   explicit positions and both selectors omitted; include the TDSE default if
   Q4 A is adopted. Require the rebuilt configuration hash to remain the same
   when the ambient default count changes.

8. **R8 — Medium: define how the GUI preserves existing explicit positions
   (§§5.4, 8, 11; Q7.6).** Changing an object sheet's field to
   `settings.n_save` does not by itself migrate a loaded recipe that still holds
   `settings.save_planes`. Writing the count without removing the list supplies
   both; replacing an arbitrary list by its count silently redistributes its
   observations. A length edit also needs a stated behavior for an explicit list
   after the automatic rewriting is removed. Q7.6's count description is
   ambiguous: the probe `even_planes(80, 0.25)` gives 81 total planes today,
   whereas entering `n_save=80` gives 80. Preserving that old list means displaying
   81 as the total, not reusing its old displayed value of 80.

   **Recommendation:** describe loading and editing both forms, preserving
   explicit positions until the user chooses a count, and removing the other
   selector in the same edit. Test nonuniform and uniform legacy lists,
   count/list switches, length changes, save/reload and undo. State the two count
   cases separately in the migration text.

9. **R9 — Medium: do not promise exact identity when converting existing
   lists to counts (§§7–9).** The structural parity test is valid when explicit
   positions are the resolver's exact array. It does not prove that independently
   written decimal lists have that array. Comparing the filament recipe's 80
   centimetre positions, read with its actual quantity decoder, with
   `np.linspace(0.0, 0.25, 81)[1:]` changed 30 coordinates;
   the maximum difference was `2.7755575615628914e-17` m. Those are small numerical
   differences, but the positions enter `configuration_hash`, so converting the
   example can invalidate resume identity even though its endpoint is unchanged.

   **Recommendation:** retain existing lists where identity is promised, or
   explicitly qualify an example conversion as a migration and test its new
   numerical result separately. Do not round all positions to force matching
   hashes. Regenerate any edited gallery as required by the repository's
   documentation policy, and distinguish unchanged numerical results from
   unchanged requests and hashes.

10. **R10 — Medium: specify endpoint equality before implementing automatic
    appending (§§5–7; Q1–Q2).** The proposal requires the final plane to equal the
    length but does not say how independently calculated positions are compared.
    For example, `0.1 + 0.2` is one representable value above `0.3`, and
    `np.nextafter(0.3, 0.0)` is one below it. A strict upper-bound check rejects
    the first; unconditional appending leaves two almost coincident observations
    for the second. The marches already use a landing tolerance, so this affects
    whether two requested planes mean distinct observations, their indices and
    the recorded hash.

    **Question and recommendation:** choose and document strict comparison or a
    bounded endpoint normalization, including whether near-end duplicates are
    refused or retained. If normalization is chosen, account for its identity
    impact rather than claiming every existing request is unchanged. Add tests
    immediately below, equal to and above the endpoint, and ensure that applying
    the resolver twice cannot append another plane. Complete the refusal table
    with non-finite positions, empty inputs and non-integral or boolean counts,
    so the shared rule has named errors in both Python and recipes.

### Maintainer decisions following AI review — 2026-11-15

John Travers adopted the decisions below in discussion with the AI reviewer.
They are recorded here for the drafting agent to reconcile into Part I; the
reviewer has not edited the proposal. The design choices raised in this review
discussion are settled below. The drafting agent must still revise the proposal
and respond to the review; these decisions do not constitute acceptance of the
SEP.

**Compatibility constraint.** Backwards compatibility is explicitly not a
requirement. Preserving pre-change configuration hashes, converting old recipes
or supporting old output files must not constrain the design. Newly created runs
must still reproduce their recorded configuration, and numerical correctness and
rounding bounds remain requirements. In particular, R9's preservation of old
hashes is no longer an acceptance condition; R3's warning that changing the grid
length changes absorber rates remains relevant when correcting scientific tests.

1. **D1 — Always record the actual entrance and the end as observation planes
   (Q1, Q2, Q5; R6, R8, R10).** The entrance is `initial_z`, and the end is the
   resolved `length`. Both are automatic; append the end when an explicit
   position list omits it. Refuse requested positions outside the propagation
   interval, applying the rounding policy in D5.

   An observation does not require a stored field. Every observation, including
   both endpoints, obeys the existing `fields` and `field_planes` policy:
   `fields="none"` keeps statistics and step history without storing plane
   fields, `fields="planes"` keeps fields only at the selected observation
   indices, and `fields="all"` keeps them at every observation. The mandatory
   endpoints never override field retention. Terminal checkpoints are separate
   and retain their existing restart policy.

2. **D2 — Specify the length once; permit an explicit consistency check
   (R1; §§5–6).** Make `length` optional and keyword-only on the keyword routes,
   with the same resolution rule in recipes:

   | Supplied | Resolution |
   |---|---|
   | `length`, without `grid` | Build the grid with that length. |
   | `grid`, without `length` | Take the run's end from `grid.length`. |
   | Both | Require agreement and refuse a mismatch, naming both values. |
   | Neither | Refuse, explaining that a length or a grid is required. |

   All length-dependent setup uses that one resolved value. Do not silently
   alter a supplied grid or choose between conflicting lengths. The numerical
   allowance for agreement is the endpoint rounding policy in D5.

3. **D3 — Count the observations over the actual launched interval
   (R4; §§5.1–5.2; Q6).** A fresh `n_save=N` request, with `N >= 2`, places
   exactly `N` evenly spaced observations over `[initial_z, length]`, including
   both endpoints. The entrance is observation index 0. `length` remains the
   absolute end coordinate; the distance propagated is `length - initial_z`.

   This supersedes the draft's zero-based count and R4's recommendation to
   filter a newly requested count's zero-based lattice. An existing simulation
   advanced with `with_input(..., initial_z=z)` instead retains its already
   resolved future observations, with the new entrance recorded at `z`; it has
   not requested a new count or a redistribution of the existing positions.

4. **D4 — A replay schedule covers the run's interval (Q3; R2).** On routes
   that support replay, require the schedule's start and end to agree with the
   actual entrance and resolved end. Refuse genuinely shorter or longer
   schedules before execution; neither ignore a longer tail nor modify the
   steps to fit the interval.

   Use the existing `replay_tolerance` for the end comparison, allowing the
   rounding error accumulated by the step sum. The start comparison has no
   accumulated sum and uses the tight endpoint agreement rule. This decision
   does not introduce replay support for interior launches where it is
   currently refused.

5. **D5 — Normalize endpoint rounding, with one canonical end (R1, R10).**
   Normalize a supplied position within the existing landing tolerance of the
   end to that exact end: eight representable floating-point increments at the
   endpoint (`LANDING_ULPS`). Keep one endpoint observation after normalization,
   and refuse positions genuinely beyond the end. Ordinary interior positions
   retain their supplied values.

   Apply the same tight agreement rule when both `length` and `grid` are given;
   after they agree, `grid.length` is the canonical value used throughout setup.
   This settles the outstanding rounding policy in D1 and D2. A replay's end
   retains the separate accumulated-rounding allowance in D4.

6. **D6 — Retain the TDSE observation defaults (Q4–Q5).** When neither
   `n_save` nor `save_planes` is supplied, a TDSE run requests `n_save=2`, its
   entrance and end. Its default `step_planes=1` additionally records every
   accepted natural step. Explicit user settings take precedence.

   Avoid adding intermediate landing positions solely for output by default,
   since they can require extra expensive atomic-response work. The count
   `n_save` describes the evenly spaced requested observations; `step_planes`
   adds observations, so the total recorded count can be larger. All of these
   observations obey the field-retention policy in D1.

7. **D7 — Expose counts and explicit positions as distinct GUI modes (R8).**
   Offer "Evenly spaced" and "Explicit positions" modes. The first edits
   `n_save`; the second edits `save_planes`. Switching modes replaces the other
   selector in one undoable edit, so a document never acquires both selectors
   through a mode switch.

   A length change redistributes count-based observations automatically.
   Explicitly requested coordinates remain unchanged: the automatic end follows
   the new length, and an explicit position beyond a shortened length produces
   a clear validation error. Do not represent an arbitrary explicit list merely
   by its count or silently redistribute its observations.

8. **D8 — Accept an empty position list and explicitly listed endpoints
   (R4, R10).** `save_planes=[]` requests no additional positions, so only the
   automatic entrance and end are requested. An explicit list may include
   `initial_z` and `length`; each endpoint appears once, with the endpoint
   normalization in D5. This supersedes R4's recommendation to refuse an
   explicit position equal to the launch. Positions before the launch remain
   outside the interval and are refused.

   Non-finite values, unsorted positions and repeated interior positions remain
   errors. Omitting both selectors takes the route's default count; an explicitly
   empty list instead requests only the endpoints. If enabled, `step_planes`
   adds its observations as in D6.

**Remaining work from the review.** No further maintainer choice is required by
the findings above. The drafting agent must reconcile Part I with D1–D8 and the
compatibility constraint, and respond to R1–R10 point by point. The following
remain implementation and verification requirements:

- Apply the same resolution and validation when constructing a simulation,
  replacing its settings or adopting settings from another simulation (R5).
- Preserve authored recipe choices while storing resolved positions and defaults
  that reproduce a newly created run, including nested object-route settings
  (R7). No preservation of pre-change hashes is required (R9).
- Distinguish a run's declared end from its last committed plane or checkpoint
  when resuming a newly written interrupted file (R6). No compatibility path for
  old files is required.
- Update scientific tests and examples deliberately, accounting for the effect
  of a changed grid length on absorber rates and other derived settings (R3),
  and qualify the new behavior on the reference and native CPU backends.
