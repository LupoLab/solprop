# SEP 51 — One rule for the saved planes and the end of a run

| | |
|---|---|
| **SEP** | 51 |
| **Title** | One rule for the saved planes and the end of a run |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-15) |
| **Status** | In progress |
| **Review status** | Reviewed |
| **Reviewers** | John Travers (human); GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-11-15 |
| **Resolution** | Accepted 2026-11-15 by John Travers in [PR #847](https://github.com/LupoLab/solprop/pull/847), after the GPT-6 review, decisions D1–D8 and the revision that answered R1–R10. |
| **Requires** | — |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI), for John Travers, on `feature/sep-0051-save-planes` |
| **Manual pages** | `guide/setup/running.md`, `guide/setup/recipes.md`, `guide/free-space/beams.md` |

## Part I — Proposal

### 1. Abstract

Today a run's saved planes are written three different ways, depending on whether it is
started from a keyword interface, from objects or from a recipe, and the end of the run
is the last saved plane, never the length the run was set up for. This SEP gives every
route one length and one pair of plane selectors. The length is stated once, as `length`
or as `grid.length`, and it is both the upper limit and the end: every run marches from
its entrance, `initial_z`, to its length, and always records both. `n_save` is a count of
evenly spaced observations over that interval, both ends included, and `save_planes` is a
list of positions inside it; a run takes one or the other, never both. Success means one
placement rule in one helper, the same refusals on every route, recipe, Python export and
front-end sheet, and, for every run that already ends at its length, the same numbers on
`numpy_ref` and on the native CPU build. Backwards compatibility of configuration hashes,
old recipes and old output files is not a requirement (§9).

### 2. Motivation and scope

The trigger was [#533](https://github.com/LupoLab/solprop/issues/533). A recipe with
`save_planes = 80` in an object route's `[build.settings]` passes `Recipe.check()` and is
read as one plane at 80 m. The run then marches until something unrelated stops it.
Investigating it showed that no route ties the run's end to its length:

- On a **keyword route**, `length` builds the grid and sets the default steps. The run
  still ends at the last of the explicit `save_planes`, and positions beyond `length` are
  accepted. With `length = 1 mm`, `save_planes=[0.5 mm]` stops at 0.5 mm, and
  `[0.5 mm, 2 mm]` runs to 2 mm on a grid sized for 1 mm. A keyword route given both
  `length` and `grid=` keeps the grid without comparing the two.
- On an **object route**, `grid.length` sizes the collar, the window and the default
  absorber rates, but the run ends at the last save plane, whatever `grid.length` says.

In scope:

- the meaning of `length`, `n_save`, `save_planes` and the run's end on every route that
  runs: the keyword interfaces, `Simulation(...)`, `Simulation.from_free`,
  `Simulation.from_planar`, recipes, scans and the front end;
- interior launches, replay schedules, settings replacement, resume and the resolved
  recipe, as far as they read the planes or the end;
- the refusals and their messages, and the migration;
- how the end reaches the backends.

Not in scope:

- continuing a finished run to a longer length
  ([#141](https://github.com/LupoLab/solprop/issues/141));
- `fields=` and `field_planes=` on the `prop_*` functions
  ([#311](https://github.com/LupoLab/solprop/issues/311)), whose retention policy the
  observations here obey unchanged;
- any change to stepping, tolerances or physics defaults;
- `FixedSchedule`'s own construction (`FixedSchedule.uniform` and replay), which keeps its
  interface; only how its start and end relate to the run's interval is settled here;
- replay for interior launches, which stays refused.

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

**The length on keyword routes.** `length` is a required parameter, and `grid=` is an
optional one. `_gnlse_grid` (`src/solprop/driver/routes/gnlse.py`) and `_capillary_grid`
(`src/solprop/driver/routes/hollow.py`, shared by the general modal route) return a
supplied grid without comparing its length with `length`.

**What depends on the length.** Besides the end, `grid.length` sets the default absorber
reference length, `grid.length / boundary_N` (`_reference_length` in
`src/solprop/boundaries.py`), so the default spectral and temporal absorber rates scale
inversely with it unless `boundary_length` is given. Keyword routes also derive default
steps and longitudinal profiles from `length`.

**Related mechanisms that read the planes:**

- **`step_planes = k`** also saves the end of every `k`-th accepted step, beside the
  requested planes.
- **`fields` and `field_planes`** decide which observations keep a stored field:
  `fields="none"` keeps statistics and step history only, `fields="planes"` keeps fields at
  the observation indices in `field_planes` (the entrance is 0 and negative indices count
  from the end), and `fields="all"` keeps them everywhere. `field_planes` is refused
  beside `step_planes`.
- **`interpolate_planes`** observes planes inside steps, except the run's end and the
  hard kinks.
- **`FixedSchedule`** has its own `z_start` and `z_end`, and its save planes must lie in
  `[z_start, z_end]`. A schedule that ends before the last plane fails at run time ("the
  replayed schedule ends at … and cannot reach …"). One that ends after it is accepted,
  and its tail is never run. `replay_tolerance` (`src/solprop/stepper/schedule.py`)
  already bounds how far a replayed edge may sit from a target: `LANDING_ULPS = 8` or the
  step count, whichever is larger, in ulps, because a sum of `n` rounded steps drifts by
  about `n` ulps.
- **Landing.** The marches land on a target within `LANDING_ULPS` ulps of it.
- **Interior launches.** `Simulation(initial_z=...)` already requires
  `initial_z < grid.length` and every plane to follow it, and refuses a fixed schedule
  (`src/solprop/driver/inputs.py`). `Simulation.with_input(..., initial_z=z)` keeps the
  *remaining* planes, those beyond `z`.
- **Settings replacement.** The `Simulation.settings` setter accepts new settings before
  translation, and afterwards only a change of execution resources. It compares settings
  to protect a translated request and to decide whether an attached recipe still
  describes the run.
- **Resume and the configuration hash.** The configuration records the positions
  (`configuration["save_planes"]`), so a count and the same positions hash alike.
  `driver/resume.py` distinguishes the requested planes from the last committed plane and
  a terminal checkpoint. `recipe.resume_file` rebuilds the recipe before the resume gate,
  and `output.hdf5.check_cache` compares the hash before opening a file for append.
- **The resolved recipe.** `recipe/provenance.py:resolved_document` writes a keyword
  route's resolved planes and drops its top-level `n_save`, but leaves a nested
  `[build.settings]` table as written.
- **Scans** count `len(settings.save_planes) + 1` planes per point.

**The seam.** `assemble` (`src/solprop/translate/assembly.py`) "covers
`[initial_z, max(save_planes)]`". `StepperSettings.z_end`
(`src/solprop/translate/problem.py`) is `max(save_planes)`. No run end separate from the
planes crosses the seam, in `RunRequest` or in `PropagationProblem`.

**Issues and pull requests:** this SEP fixes
[#533](https://github.com/LupoLab/solprop/issues/533) and is tracked by
[#846](https://github.com/LupoLab/solprop/issues/846). It supersedes the narrow draft fix
in [PR #843](https://github.com/LupoLab/solprop/pull/843), which checked object-route
positions against `grid.length` and read an integer as a count; that pull request closes
in favour of this SEP's implementation. It borders on
[#141](https://github.com/LupoLab/solprop/issues/141), which continues a finished run,
and [#311](https://github.com/LupoLab/solprop/issues/311), which concerns `prop_*` and
`field_planes`; it changes neither.

### 4. Theory

None. No equation, approximation or default of the physics changes. What changes is
which positions are observed and how far a run goes. The run's length is a setup input
with numerical consequences (the default absorber rates, and a keyword route's default
steps and profiles, §3), so a test or example whose length is *changed* to fit the new
rule changes numerically. §7 and Q7 treat that as a deliberate, qualified change, never as
a neutral migration.

### 5. Proposed interface

The maintainer's decisions D1–D8 (Part III) settle every choice below; the questions
they answer are recorded in §12.

**5.1 The length, stated once (D2).** On every keyword route, `length` becomes optional
and keyword-only, and the same rule holds for a recipe's keyword-route parameters:

| Supplied | Resolution |
|---|---|
| `length`, without `grid` | Build the grid with that length. |
| `grid`, without `length` | Take the length from `grid.length`. |
| Both | Require agreement under §5.4, then use `grid.length`; refuse a mismatch, naming both values. |
| Neither | Refuse: "a length or a grid is required". |

All length-dependent setup (the grid, default steps, profiles, the absorber's default
reference length and the run's end) uses that one value. A supplied grid is never altered
or rebuilt, and no conflict is settled silently. On an object route the length is
`grid.length`.

**5.2 The interval and its endpoints (D1).** Every run marches over
`[initial_z, length]`. `initial_z` is 0 unless an interior launch sets it; `length` is the
absolute end coordinate, so the distance propagated is `length - initial_z`. The
entrance and the end are always observations: the entrance is observation index 0 and
the end is index −1. An observation is a position where the run records its state; it
does not imply a stored field. Every observation, the two endpoints included, obeys
`fields` and `field_planes` unchanged, so `fields="none"` keeps statistics and step
history only, and `fields="planes"` keeps fields only at the selected indices. Terminal
checkpoints are separate and keep their restart policy.

**5.3 The selectors (D3, D6, D8).** Every route takes the same two parameters, which are
mutually exclusive:

- `n_save=N`: an integer `N ≥ 2`. A fresh request places exactly `N` evenly spaced
  observations over `[initial_z, length]`, both endpoints included.
- `save_planes`: a sequence of positions in m, finite and strictly increasing, inside
  `[initial_z, length]`. The entrance and the end are added when absent; either may be
  listed, and each appears once (§5.4). `save_planes=[]` requests only the two
  endpoints.

Given neither, a run takes its route's default count: `DEFAULT_N_SAVE = 201` in general,
and `n_save = 2` with `step_planes = 1` on the TDSE capillary (D6). Explicit settings
take precedence over a route default. `step_planes` adds its natural-step observations to
the requested ones, so the recorded count can exceed `n_save`. `SimulationSettings`
gains `n_save`, and its `save_planes` becomes optional.

**5.4 Endpoint rounding (D5).** A supplied position within `LANDING_ULPS` ulps of the end
(the marches' existing landing tolerance) is the end, and becomes exactly `length`; the
same holds at the entrance. Interior positions keep their supplied values. After
normalization, one observation remains at each endpoint, and a position beyond the end or
before the entrance is refused. When both `length` and `grid` are given, they agree when
they are within the same `LANDING_ULPS` ulps, and `grid.length` is then the canonical
value. For example, `0.1 + 0.2` and `np.nextafter(0.3, 0.0)` are both the end of a run of
length `0.3`, and `0.31` is refused.

**5.5 Refusals.** Every refusal is a `DriverError` in Python and a `RecipeError` in a
recipe, with one message naming the fix:

| Input | Message names |
|---|---|
| `save_planes=80` (a bare integer) | "pass `n_save=80`" |
| `save_planes=0.25` (a lone float) | "pass `[0.25]`, or `n_save`" |
| both `n_save` and `save_planes` | "give one of `n_save` and `save_planes`" |
| `n_save` below 2, non-integral, or boolean | "`n_save` is an integer of at least 2" |
| a non-finite, unsorted or repeated position | the offending position |
| a position before the entrance or beyond the end | the position, the interval and "raise `length` (or `grid.length`)" |
| `length` and `grid.length` that disagree | both values |
| neither `length` nor `grid` on a keyword route | "a length or a grid is required" |
| a replay schedule that does not cover the interval | the schedule's start or end and the run's (§5.7) |

The integer form leaves the `save_planes` metadata, and `n_save` joins the object routes'
settings metadata, so `Recipe.check()` refuses `save_planes = 80` with the same message.

**5.6 Resolution and adoption (D3; R5).** Settings do not know the run's interval, so a
count, an explicit list or the default is resolved where a simulation adopts settings and
the interval is known. *Adoption* is every place settings meet a run: the `Simulation`
constructor, the keyword builders, the `settings` setter, and the `with_*` methods,
scans and reseeds that copy settings into a new run. At every adoption:

1. the settings are validated against the receiving run, whether they hold a count, a
   list or resolved positions;
2. they are resolved into the positions that follow the entrance and end exactly at
   `length`, and `n_save` becomes `None`;
3. comparisons (the setter's protection of a translated request, and of an attached
   recipe) are made on the resolved settings, so a count and its own positions compare
   equal.

Resolution is idempotent: resolving resolved positions against the same run returns them
unchanged, because the end is already present and normalization never adds a second one.
Settings reused on a run with another length are validated afresh, as explicit positions:
a position beyond the new end is refused, and a shorter list gains the new end. To carry
a *count* to another run, pass the authored settings, or
`dataclasses.replace(settings, n_save=N, save_planes=None)`; the message for "both given"
says so. After translation the setter still refuses any change to the planes.

`Simulation.with_input(..., initial_z=z)` does not request a new count. It keeps the
already resolved observations beyond `z`, which always include the end, and records the
new entrance at `z` (D3).

One private helper holds the placement, normalization and refusal rules; the keyword
routes' `_save_planes` in `src/solprop/driver/settings.py` becomes it.

**5.7 Replay schedules (D4).** On routes that support replay, a `FixedSchedule`'s start
must agree with the entrance under §5.4's tight rule, and its end with `length` within
`replay_tolerance`, which allows the step sum's accumulated rounding. A shorter or longer
schedule is refused at adoption, before execution; a longer tail is never ignored and the
steps are never adjusted to fit. Interior launches keep refusing replay.

**5.8 How the seam carries the end.** The driver guarantees that the last resolved
position *is* `length`, so `StepperSettings.z_end = max(save_planes)` is the run's end with
no new field. `assemble` and below keep their meaning, and `RunRequest`,
`PropagationProblem` and the frozen tables of `tests/backends/test_driver.py` are
unchanged. A caller of `assemble` directly, below the driver, still ends at its last
plane, so a deliberately short test may live there.

**5.9 Recipes, export, provenance and resume (R6, R7).**

- *Authored text.* Recipes and the Python export keep what the author wrote: `n_save`,
  `save_planes`, `length` or `grid`, or neither selector.
- *The resolved recipe* (`Recipe.from_output(..., resolved=True)`) rebuilds the same run
  without depending on defaults. It writes only the canonical resolved positions, ending
  at the length, at the level the route reads them (top level on a keyword route,
  `[build.settings]` on an object route), and drops `n_save`. A route default, the TDSE
  `n_save = 2` and `step_planes = 1` included, is written out resolved, so a later change
  of a default does not change the rebuilt configuration hash.
- *Resume* continues a file to its *declared* end, the last requested position in
  `configuration["save_planes"]`, which equals `grid.length` for every file written under
  this SEP. That is distinct from the last committed plane and from a terminal
  checkpoint: an interrupted file, or one that holds a terminal checkpoint, resumes as
  today. No compatibility path is kept for files written before this SEP; they meet the
  existing hash refusal.

**5.10 The front end (D7).** Object and keyword sheets offer two modes for the saved
planes: "Evenly spaced" edits `n_save`, counting both endpoints, and "Explicit positions"
edits `save_planes`. Switching modes replaces the other selector in one undoable edit,
so a document never holds both. A length change redistributes a count automatically.
Explicit positions stay as entered: the automatic end follows the new length, and a
position beyond a shortened length is a validation error shown on the field. A loaded
document opens in the mode of the selector it holds; an arbitrary list is never shown as,
or replaced by, its count.

```text
Simulation.from_free(grid, space, medium, responses, beam,
                     settings=SimulationSettings(n_save=81))     # 81 observations over [0, grid.length]
SimulationSettings(save_planes=[0.05])                           # grid.length = 0.1: entrance, 0.05 m and the end
SimulationSettings(save_planes=[])                               # entrance and end only
Simulation(..., initial_z=0.04, settings=SimulationSettings(n_save=4))
                                                                 # grid.length = 0.1: 0.04, 0.06, 0.08, 0.1 m
SimulationSettings(save_planes=80)       -> DriverError: "... pass n_save=80"
SimulationSettings(save_planes=[0.3])    -> DriverError at adoption: "0.3 m is beyond the end, grid.length = 0.25 m; ..."
prop_gnlse(..., length=1e-3, grid=grid)  -> DriverError if grid.length = 1e-4: names both
```

```text
[build.settings]
type = "SimulationSettings"
n_save = 81                # was a list of 80 positions in the filament recipe
```

### 6. Implementation strategies

- **Layer.** The driver and the recipe layer, plus the front end. The physics and backend
  layers are untouched, and so is the seam (§5.8).
- **Where the rules apply.** One helper places, normalizes and refuses; one adoption
  function calls it for every adoption in §5.6, so no route has its own copy. The keyword
  routes resolve `length` against `grid` (§5.1) before anything that depends on the
  length, then build their `Simulation` through the same adoption.
- **Replay** is checked in the same adoption function, with `replay_tolerance` for the end
  and the tight rule for the start.
- **Resume** reads the declared end from the stored configuration in `driver/resume.py`,
  and the recipe-driven path reaches the same gate.
- **Backends.** Nothing backend-specific. `numpy_ref` and the native CPU build run the
  same requests as today for every run that already ends at its length from `z = 0`.
- **Cost estimate** (`poe size`):

  | | Added | Removed |
  |---|---|---|
  | Code | about 250 lines: the helper and refusals, adoption in the setter and `with_*`, `n_save` on settings, `length`/`grid` resolution, replay checks, the resolved recipe, the front-end modes | about 60 lines: the front end's plane rewriting and list expansion, the routes' private length handling |
  | Docstrings | about 100 lines | — |
  | Tests | about 450 lines: the refusal table, the rounding cases, count/positions parity, the end rule and interior launches per route, adoption and reuse, replay, resume, the resolved recipe, the front-end modes | — |

  Separately, an estimated 10–30 existing tests, snippets and examples need edits (Q7).
- **Dependencies:** none.

### 7. Verification and validation

- **Refusals.** Every row of §5.5's table, on every route kind, in Python and in a
  recipe, with its message.
- **Endpoint rounding.** Positions one representable value below, equal to and above the
  end (`np.nextafter(0.3, 0.0)`, `0.3`, `0.1 + 0.2`) and just beyond `LANDING_ULPS`;
  `length` and `grid.length` agreeing within and outside the bound; and resolving twice,
  which adds no plane.
- **Parity.** `n_save=N` and the positions it resolves to give identical settings, hash
  and output. This is an exact structural check, since both resolve to the same array; it
  says nothing about an independently written decimal list (R9).
- **End rule.** Per route kind, the first observation is the entrance, the last equals the
  length, and the count is as specified. Direct interior launches with `n_save`, on and
  off the lattice of a zero-based count; `with_input(initial_z=…)` keeping the remaining
  observations and the end; `save_planes=[]`; listed endpoints appearing once.
- **Adoption.** Construction, replacement through the setter, repeated adoption, reuse of
  resolved settings at another length, the count/list switch, and the existing refusal to
  change a translated run's planes.
- **Replay.** `FixedSchedule.uniform` and `from_history` schedules whose sums drift below
  and above the length within `replay_tolerance` are accepted; genuinely shorter and
  longer schedules, and a start that is not the entrance, are refused before execution.
- **Resume and provenance.** Interrupted, terminally checkpointed and completed files
  resume to the declared end, directly and through `recipe.resume_file`. Resolved recipes
  round-trip through `Recipe.from_output(..., resolved=True)` for a count, explicit
  positions, neither selector and the TDSE default, and the rebuilt hash does not change
  when a default count is changed.
- **Unchanged numbers.** For runs that already end at their length from `z = 0`, the
  request is bit-identical on its frozen fields, and a short run (about ten steps) per
  route kind on `numpy_ref` and native CPU matches its pre-change output within the
  rounding budget of SEP 25. Identical numbers are not identical hashes: hash
  preservation is not required (§9).
- **Changed numbers.** A test or example whose length or planes change is a deliberate
  change, recorded and re-qualified on `numpy_ref` and native CPU (Q7).
- **Front end.** Both modes, the switch as one undoable edit, length changes in each mode,
  uniform and non-uniform loaded lists, save and reload.

### 8. Documentation

- **Manual pages:**
  - `guide/setup/running.md`: the length stated once, the interval, the two selectors,
    the endpoints and endpoint rounding;
  - `guide/setup/recipes.md`, including the resolved recipe;
  - the free-space and planar pages that use the `np.linspace(0, length, N)[1:]` idiom,
    which becomes `n_save` where natural;
  - the front end's page on saved planes and its two modes.
- **Docstrings:** `SimulationSettings`, `Simulation` and its `settings` setter,
  `with_input`, `from_free`, `from_planar`, the keyword interfaces (`length` and `grid`),
  and the recipe metadata.
- **Changelog:** a `.changed` fragment with the migration (Q7), and a `.fixed` fragment
  for #533.
- **Examples:** the filament and crossing-beam recipes and galleries may switch to
  `n_save`. A converted example is a migration whose positions, and so whose hash, may
  differ in the last bit; an edited gallery is regenerated under the documentation
  policy.

### 9. Compatibility and impact

This is a breaking change to the public interface, and backwards compatibility is not a
requirement (maintainer, Part III): pre-change configuration hashes, old recipes and old
output files do not constrain it. Newly created runs must still reproduce their recorded
configuration, and numerical correctness and rounding bounds still hold. Every break is
either refused with a message naming the fix or listed in the changelog (Q7). The output
schema, the kernel contract and the frozen seam tables are unchanged. Validated results
are unchanged wherever a run already ends at its length; where a test's length or planes
are changed instead, its numbers change deliberately (§4). No deliberate difference from
Luna.jl is introduced.

### 10. Alternatives considered

- **Keep positions as the end, and only check against `grid.length`.** This was the first
  fix for #533 (PR #843). The maintainer rejected it because two conventions would remain.
- **Read an integer `save_planes` as a count.** This overloads one name with two meanings,
  and the keyword routes already spell the count `n_save`.
- **Make the run end a separate field of the seam,** so positions short of the length need
  not save the end. This needs a seam change with frozen tables; recording the end
  always (D1) makes it unnecessary.
- **Refuse positions that do not end at the length.** Rejected for D1: the end is added,
  as the entrance is.
- **Count `n_save` over `[0, length]` and filter for interior launches.** This was the
  draft's rule and R4's recommendation; D3 counts over the launched interval instead, so a
  count always means that many observations.
- **Prefer one of `length` and `grid` when both are given.** Rejected for D2: a conflict
  is refused, never settled silently.
- **A fixed "few ulps" allowance for a replay's end.** It refuses valid schedules whose
  step sums drift (R2); D4 uses `replay_tolerance`.

### 11. Phases

1. **Driver and recipes.**
   - Survey first: run the affected test modules with an instrumented adoption, to count
     every run whose end moves, whose positions are refused, and whose keyword route
     passes both `length` and `grid` (Q7).
   - Deliverables: the helper, adoption, `n_save` on settings, `length`/`grid`
     resolution, the refusals, the end rule on every route, replay, resume, the recipe
     metadata, export and resolved recipe, the tests and edits, the manual pages and the
     changelog.
   - Verify: §7 on `numpy_ref` and the native CPU build.
2. **Front end.**
   - Deliverables: the two modes on every sheet, with the list rewriting removed; the
     front-end gate.

Both phases are small, and may merge as one pull request.

### 12. Open questions

None remain. The draft's questions were settled by the maintainer after the AI review
(D1–D8, Part III):

- **Q1 — positions short of the length:** the end is always recorded, appended when
  absent (D1).
- **Q2 — positions beyond the length:** refused on every route, after endpoint rounding
  (D1, D5).
- **Q3 — a replay schedule's ends:** both must cover the interval; the end within
  `replay_tolerance` (D4).
- **Q4 — the TDSE default:** kept as a route default, `n_save = 2` and `step_planes = 1`
  (D6).
- **Q5 — `step_planes` and `field_planes`:** unchanged; observations obey the retention
  policy, and `-1` is always the end (D1, D6).
- **Q6 — resume, interior launches and `with_*`:** counts are over the launched interval
  (D3); `with_input` keeps the resolved remaining observations (D3); resume reads the
  declared end, and no legacy path is kept (compatibility constraint).
- **Q7 — migration and blast radius:** below.

**Q7 — Migration and blast radius.** The breaks, each with its message:

1. `save_planes=N` as a bare integer: refused, "pass `n_save=N`", in Python and in
   recipes.
2. `save_planes=x` as a lone float: refused, "pass `[x]`, or `n_save`".
3. A position before the entrance or beyond the end: refused, naming the interval.
4. Positions short of the length: the run now continues to the length and records it.
   This raises no error and is listed in the changelog. Indices counted from the end
   (`field_planes=[-1, …]`) then point to different planes.
5. A replay schedule that does not cover the interval: refused.
6. `length` is keyword-only and optional on keyword routes; a positional `length` fails
   with Python's own error, and `length` with a disagreeing `grid` is refused.
7. `n_save` with an interior launch counts over `[initial_z, length]`.
8. The front end's object sheets: "Saved planes" becomes a mode. A saved document holds
   the expanded list, so it opens as "Explicit positions", unchanged. A count entered as
   80 on an object sheet today gave 81 observations with the entrance; the same planes
   are now "Evenly spaced" with `n_save = 81`.

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
- **Keyword routes.** Explicit positions short of `length`, positional `length` and
  `length` beside `grid=` are counted in phase 1.

**How a flagged test is fixed (R3).** Shortening a grid is not neutral: it raises the
default absorber rates in proportion, and on a keyword route it can change the default
steps and profiles. A fix therefore first asks what the test means:

- *A deliberately short window of a longer setup* keeps its length and physics: it keeps
  the original grid and records the end, or pins `boundary_length` (and any step or
  profile choice) to the original values before shortening the grid, or moves below the
  driver to `assemble`, which still ends at its last plane.
- *A mismatched setup* is corrected to the intended length, and its numerical change is
  recorded and re-qualified.

Replacing the planes by `n_save` extends a short run; it never preserves the short window.

### 13. References

None beyond solprop's own documentation and issues, cited where used.

## Part II — Implementation record

### Phase 1 — Driver and recipes

**Scope.** 2026-11-16; Opus 5.5 (AI) for John Travers; branch
`feature/sep-0051-save-planes`, commits from `ebff016c` (the helper and adoption) to the
commit that appends this report. The front end (§5.10, phase 2) is untouched.

**Survey (§11), before any call site was edited.** An instrumented `Simulation.settings`
setter and the two grid helpers recorded, for every run the default test lane builds,
what the new rule would do. The lane was cancelled from outside at 91 % (twice; the
numbers are of the second run); the slow lane and `native/tests` were not run.

| What the rule changes | Count |
|---|---|
| The run's end moves later (last plane before `grid.length`) | 371 adoptions, 189 tests, 52 call sites in 25 files |
| A plane beyond the end, refused | 1 (`tests/output/test_record_arrays.py`, a `length` of 1 cm on a 10 µm grid) |
| A plane before an interior launch, refused | 1 (already a refusal test) |
| A replay schedule not covering the run, refused | 92 adoptions, 32 tests, 16 sites in 10 files |
| A replay start that is not the entrance | 0 |
| A plane normalized onto the end | 0 |
| A keyword call with both `length` and `grid` | 2199 calls at 224 sites; 43 calls at 13 sites disagree |

A static scan of the tree (Python files and the Python blocks of Markdown pages) found 509
keyword-route calls that pass `length` positionally, in 227 hand-written files (150 test
files, 43 gallery scripts, 12 guide pages, 12 native tests, 5 benchmarks, the README, two
package files and one design page), and 185 call sites passing `length` beside `grid=`.
That count is what decided the first deviation below.

**Delivered.**

- `src/solprop/driver/settings.py`: `SimulationSettings.n_save`, an optional
  `save_planes`, and four private helpers: `_selectors` (the refusals that need no
  interval), `_observation_planes` (placement, endpoint normalization within
  `LANDING_ULPS` ulps of each endpoint, the interval refusals; it replaces
  `_save_planes`), `_run_length` (the length/grid table of §5.1) and `_adopted` (the
  entrance check, the replay check of §5.7 with `replay_tolerance`, and resolution).
- `src/solprop/driver/simulation.py`: the `settings` setter adopts every settings object
  through `_adopted` before any comparison, so the constructor, the keyword builders,
  `with_input`, `with_mixture` and anything else that builds a `Simulation` share one
  rule; `length` is optional on `from_gnlse`, `from_capillary` and `from_waveguide` and
  resolved by `_run_length` before anything uses it; the keyword routes pass `n_save` and
  `save_planes` through unresolved. `src/solprop/driver/interface.py`: the same
  signatures on `prop_*`. `src/solprop/driver/routes/general.py`: the general modal
  route's tables span `grid.length`, the run's end.
- `src/solprop/recipe/`: the settings metadata drops the integer form of `save_planes`
  and gains `n_save`; `refuse_plane_count` refuses `save_planes = N` with "write
  `n_save = N`" on every route; `plane_parameter` turns the driver's refusals of the
  planes, the length and replay into a `RecipeError` naming the keyword, in
  `build_object` and `Recipe.build`; a keyword-route recipe with neither `length` nor
  `grid` is refused before the route is called; `resolved_document` writes the resolved
  positions in `[build.settings]` on an object route (and at the top level otherwise)
  and drops `n_save`.
- `benchmarks/builders.py`: `window`, which runs a case to a shorter stop (`ab
  --length`) by rebuilding it on its grid cut at the stop, carrying the whole case's
  collar, steps and profiles; `benchmarks/sessions.py` and the linear GRIN case use it or
  cut their own grid.
- Tests: `tests/driver/test_save_planes.py` (the refusal table, placement and endpoints,
  the rounding cases, count/positions parity of planes and hash, interior launches on and
  off a zero-based lattice, `with_input`, setter replacement and reuse at other lengths,
  the length/grid table on all three keyword routes, replay drift and refusals) and
  `tests/recipe/test_save_planes.py` (refusals as `RecipeError` with their paths on
  object and keyword routes, `n_save` in settings, the resolved recipe for a count, a
  list, a short list and no selector, and its independence of the default count).
  `tests/builders.py` gains `grid_at` and `held_collar`.
- Pages: `docs/guide/setup/running.md` (a section on the length and the saved planes),
  `docs/guide/setup/recipes.md` (the settings' selectors and the resolved recipe),
  `docs/guide/free-space/beams.md` (the `linspace` idiom becomes `n_save`); docstrings of
  `SimulationSettings`, `Simulation`, its setter, `with_input`, `from_free`,
  `from_planar` and the keyword routes. Changelog: `changelog.d/sep-0051-save-planes.changed.md`
  (the migration) and `changelog.d/533-save-planes-count.fixed.md`.

**Existing tests changed (Q7's closing paragraph).** No tolerance was changed.

- *A short window of a longer grid, physics kept* (the grid cut at the run's end with
  `grid_at`, the collar the longer grid gave with `held_collar` or by building it on the
  longer grid, and a length-derived default step pinned where one applied): the
  sixth-order soliton of `tests/driver/test_prop_gnlse.py`; the general modal fixtures of
  `test_general_modal.py`, `test_modal_metrics.py`, `test_modal_resume.py`,
  `test_material_manifest.py`, `test_modal_launch.py` (also `dz_max`) and
  `test_route_keywords.py`; the far-field cases of `test_far_field_statistics.py` and
  `test_native_far_field.py`; two planar-hybrid runs; `tests/record_routes.py`.
- *A `length` that disagreed with `grid`* (the run's own end kept, the grid built there,
  the collar pinned): `tests/backends/adams_source_cases.py` (the Adams source ladder),
  the general entry point of `tests/backends/general_rows.py`, the native noisy-run
  reference of `test_native_session_history.py`, the two capillary receivers of
  `tests/coupling/test_workflows.py`, the wide case of `tests/coupled_flow_cases.py`
  (its collar was already pinned), `test_vector_coupling.py` and the linear GRIN
  benchmark.
- *Construction or refusal only, shortened freely*: three cases of
  `test_simulation.py`, `test_backend_default.py`, `test_record_arrays.py`, one refusal
  of `test_coherent_capillary.py`, `tests/translate/test_routes.py` and two far-field
  geometry refusals.
- *Replay*: the dispersive capillary's one-step row now records its whole replayed
  schedule (the rung takes the first step alone, so the step is unchanged);
  `tests/sessions/test_group.py`'s schedule past the last plane, which existed to test
  the ignored tail, is now a refusal test.
- *Messages and harness*: `tests/recipe/test_pins.py` (the refusal of a recipe with
  neither length nor grid), `tests/test_benchmark_cases.py` and
  `tests/test_benchmark_guides.py` (a shortened case's grid is cut at its stop, with the
  whole case's collar and steps).

**Deviations from the proposal.**

1. *`length` is optional but stays positional-or-keyword* (§5.1, Q7 item 6). Making it
   keyword-only would edit about 509 calls in 227 files and regenerate 43 galleries. This
   is an interim choice, pending with the maintainer; the signature
   change is one commit, so the keyword-only form can follow as one mechanical commit.
   Because Python allows no required parameter after an optional one,
   `from_capillary`'s and `prop_capillary`'s `gas` now defaults to `None` and is refused
   when omitted. A hazard for the keyword-only form: with `length` keyword-only, the next
   positional parameter (`betas` on the GNLSE routes, `gas` on the capillary and
   waveguide routes) takes the freed slot, so `from_capillary(r, 0.2, "Ar")` would bind
   `0.2` to `gas` instead of failing with Python's own error; a guaranteed failure would
   need those parameters keyword-only as well.
2. *A recipe with neither `length` nor `grid`* is refused by the recipe before the route
   is called, as "a length or a grid is required" against `length`, rather than by the
   route; otherwise a recipe with pins reported it against the pinned keyword.
3. *A recipe's integer `save_planes`* says "write `n_save = 80`", the recipe's spelling,
   where Python says "pass `n_save=80`".
4. *Endpoint slack* is `LANDING_ULPS` ulps of the endpoint itself (§5.4, D5), so at an
   entrance of zero it is exact; the replay check is skipped for an interior launch,
   which the existing rule refuses with its own message.

**Decisions.**

- Resolution returns the same settings object when they are already resolved for the
  run, so a repeated adoption is free and a translated run's settings compare by identity.
- The default count is read when the planes are resolved, not when the helper is
  defined, so the resolved-recipe test can change it.
- The benchmark harness rebuilds a shortened case rather than running the whole one, and
  refuses to shorten `grin_nonlinear`, which is translated at construction.

**Verification.** `numpy_ref` and the native CPU build on the Xeon, eight logical CPUs,
one thread per worker. Passing: the 29 modules the survey flagged and every module
edited, the two new modules, `tests/recipe`, the resume and scan modules of
`tests/driver`, `tests/test_guide_snippets.py`, and the doctests of the edited source
modules, `tests/backends/test_parity_catalogue.py` (737 tests with `test_coherent_capillary.py`), and the 15 modules touched only to satisfy the type checker. Repository-wide `ruff check`, `ruff format --check` and `basedpyright` (0 errors) pass; numpydoc, codespell, `reuse` and `tests/test_seps.py` pass on the touched files. Not run locally: the full default lane, the slow lane, the
coverage lane, `native/tests`, the front-end gate and the galleries; CI is the gate.

**Sources checked.** None needed: no constant, coefficient or citation was transcribed.

**Limitations.**

- The survey covers 91 % of the default lane.
- Other setup depends on `grid.length` besides the default absorber, and is not pinned
  when a test's grid is cut: `LossPolicy.limit(grid.length)` clamps a mode's attenuation
  (`src/solprop/linear.py`), the boundary monitor's quiet-window spacing is
  `grid.length / CHECKS`, and the walk-off warning integrates over `grid.length`. None of
  the changed tests moved; §3 does not list them.
- `test_modal_launch.py`'s general runs keep the default first step, which is a fraction
  of the length and refined by the start-up rule; their checks are of the launch.
- A recipe cannot hold a TDSE response (`tdse` has no recipe form), so the TDSE
  capillary's resolved recipe is checked at construction, through `resolved_document`,
  not round-tripped through `Recipe.from_output`. Added after review:
  `tests/driver/test_save_planes.py` resumes an interrupted file, one holding a terminal
  checkpoint and a completed one to the length, directly and through
  `recipe.resume_file`, and replays `from_history` and `uniform` schedules drifting
  below and above the length; `tests/recipe/test_save_planes.py` rebuilds a run from its
  file's resolved recipe with `DEFAULT_N_SAVE` changed, keeping its hash.

**Open questions.**

- Q8: keyword-only `length` (deviation 1), with the slide of `betas`/`gas` into its
  slot. Owner: the maintainer.
- Q9: `driver/scaffold.py`'s fixed-step tool has its own `n_save`, which does not count
  the entrance; it is a separate tool and left unchanged. Owner: the maintainer.
- Q10: whether the loss clamp and the monitor spacing should follow a pinned reference
  length, as the absorber's does. Owner: an issue.

**Cost** (`git diff --numstat` against `094a7fb8`): package code and
docstrings +473 −113 lines (`src/`), benchmarks +94 −12, tests +758 −121 (of which
507 in the two new modules and 29 in `tests/builders.py`), manual pages +72 −2. The
proposal estimated about 250 code lines, 100 docstring lines and 450 test lines, and 10–30
edited tests; about 45 existing test and benchmark files were edited, about a third of
them only to wrap a now-optional `save_planes` for the type checker.

**Handoff.** Phase 2 (§5.10) needs the object sheets to stop expanding a count into a
list over `grid.length`: the settings metadata now offers `n_save` beside `save_planes`
(an array only), and the resolved planes always end at the length.

### Phase 2 — Front end

**Scope.** 2026-11-16; Opus 5.5 (AI) for John Travers; branch
`feature/sep-0051-gui`, on phase 1, commits from `3646b0e5` (the sheets) to the commit
that appends this report.

**Delivered.**

- `gui/src/solprop_gui/classes/problem.py`: `Field.positions` replaces `planes_over` and
  `moves_planes`. A saved-planes row reads the positions a recipe lists, or else its
  count, and never turns a list into its count; its change writes a list as the
  positions and anything else as the count, and unsets the other selector in the same
  change. `even_planes` and the length row's rewriting of the list are removed.
  `sheet_paths` and `other_keywords` count the positions as the row's, so the keyword
  routes' `save_planes` leaves Advanced.
- `gui/src/solprop_gui/classes/catalog.py`: every class's "Saved planes" row is
  `n_save` with `positions="save_planes"`, at the top level on the keyword routes and in
  `settings` on the object routes; the object sheets' length rows no longer move planes.
- `gui/src/solprop_gui/ui/fields.py`: a row may name its forms (`mode_titles`); its
  picker then shows them on a sheet, its text editor shows only what no other editor
  holds, and leaving the text as shown writes nothing, so switching forms never replaces
  a value by itself. `ParameterRow.current_mode` is new.
- `gui/src/solprop_gui/ui/sheet.py`: the row offers "Evenly spaced" (the count's integer
  form) and "Explicit positions" (the positions' array form, as inline TOML). One value
  typed as positions is a list of one. An issue on the positions shows at the row.
- Tests: `gui/tests/test_classes.py` (both forms on every class, at the route's level; a
  uniform and a non-uniform loaded list shown as lists; each form replacing the other,
  and clearing both; a length change moving the length alone; a count on the filament
  built by solprop, ending at the shortened length, and a position beyond it refused at
  `settings.save_planes`). `gui/tests/test_ui_workflow.py` (the keyword sheet: it opens
  on its count; a length change keeps the count; switching forms writes nothing until an
  entry; the entry replaces the count in one undoable edit, undone and redone; a lone
  position; a position beyond a shortened length shown at the row and the Ready line;
  save and reopen on the positions; back to a count; a non-uniform list opening as
  itself; and the filament's object sheet opening its listed planes as explicit
  positions, replaced by a count and restored by one undo). Every step checks that the
  recipe never holds both selectors.
- Pages: `docs/guide/front-end/first-run.md` (the row's two forms),
  `docs/dev/gui/design.md` (W3's rows and the numerics line), and a front-end sentence
  in `changelog.d/sep-0051-save-planes.changed.md`.

**Deviations from the proposal.** None in behaviour. One reading of §5.10: switching
the picker shows the other form's editor, empty, and the first entry there is the one
undoable edit that writes it and removes the other selector. A switch alone writes
nothing, because a count has no positions to show and a list has no count (D7), so
neither can be written for the user.

**Verification.** On the Xeon, through Slurm, one thread per worker: `ruff format`,
`ruff check`, codespell on the touched files, `poe gui-types` (0 errors) and
`poe gui-docstrings` pass; `tests/test_classes.py` and the doctests of
`solprop_gui.classes` and `ui/fields.py` (91 passed); the whole front-end suite without
coverage, 440 passed: 433 in one run, and the seven that start a run passed when rerun
with the Slurm job's environment unset (inside a Slurm job, a run's worker records
itself as a Slurm job, which the window then cannot follow; they fail the same way on
phase 1's commit). `tests/test_seps.py`, `poe changelog-check` and the Sphinx build
(`-W -n`) pass; the build first failed on a cross-reference phase 1 added to
`guide/setup/recipes.md` (`solprop.recipe.RecipeError`, documented as
`solprop.recipe.errors.RecipeError`), corrected in its own commit. Not run locally: `gui-test` with coverage (the 100 % bar on
`classes`), which CI runs.

**Sources checked.** None needed.

**Limitations.** The positions are typed as inline TOML (`["5 cm", "10 cm"]`), as
Advanced edits them; a friendlier list editor is not part of this phase. Advanced's
`settings` editor on the object routes still shows `n_save` and `save_planes` side by
side, so a recipe edited there can hold both, which solprop's check refuses at the row.

**Cost** (`git diff --numstat` against `ccadb278`): front-end code +133 −88
(`classes` +47 −68), tests +206 −20, pages +17 −2 and the changelog +5. The proposal
estimated about 60 lines removed plus the new field and its tests.

## Part III — History

- 2026-11-15: Drafted, after the investigation of #533 found that no route ties the
  run's end to its length. Tracking issue: [#846](https://github.com/LupoLab/solprop/issues/846).
- 2026-11-15: GPT-6 completed an AI review of Part I at PR #847, commit
  `4d9af1fc480399c7f8e28814a16f32ca2e257f03`. Findings and questions follow;
  Part I is unchanged.
- 2026-11-15: John Travers reviewed the proposal and recorded decisions D1–D8. The
  drafting agent revised Part I to match and responded to R1–R10.
- 2026-11-15: Accepted by John Travers in PR #847. Part I is frozen.
- 2026-11-15: Implementation started on `feature/sep-0051-save-planes`.
- 2026-11-16: Phase 1 (driver and recipes) reported in Part II, with an interim
  deviation on keyword-only `length` pending with the maintainer.
- 2026-11-16: Phase 2 (front end) reported in Part II, on `feature/sep-0051-gui`.

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

### Response to review — 2026-11-15

Part I was revised to reconcile D1–D8 and the compatibility constraint, and to answer
R1–R10 point by point. The maintainer's review, recorded as the decisions above, is
added to **Reviewers**, and **Review status** becomes `Reviewed`.

- **R1 (keyword route given a grid):** taken, as D2. §3 records that `_gnlse_grid` and
  `_capillary_grid` (and the general modal route through it) keep a supplied grid
  unchecked. §5.1 makes `length` optional and keyword-only, with the four-row
  resolution table; a disagreement is refused naming both values, under §5.4's rounding
  rule, and a grid is never rebuilt. Python and recipes share the rule (§5.1, §5.5).
  Q7 item 6 lists the break, and phase 1 counts the affected calls.
- **R2 (replay end):** taken, as D4. §5.7 compares the end with `replay_tolerance` and the
  start with the tight rule; §3 cites the existing bound and its drift argument. The
  check is at adoption, so direct `assemble` callers keep their meaning (§5.8). §7 adds
  drift on both sides for `uniform` and `from_history`, and refusals outside the bound.
  §10 records the rejected fixed allowance.
- **R3 (changing a grid's length is not neutral):** taken. §3 states what depends on the
  length, §4 says a changed length is a numerical change, and Q7's closing paragraph
  replaces "a fix never changes a physics setting" with two cases: a deliberately short
  window keeps its physics (keep the grid, pin `boundary_length` and step or profile
  choices, or move below the driver), and a mismatched setup is corrected and
  re-qualified. §7 adds "Changed numbers".
- **R4 (counts with an interior launch):** superseded by D3. §5.3 counts `n_save` over
  `[initial_z, length]`, both endpoints included, with an example in §5.10's listing; the
  filtering recommendation is not taken (§10). Per D8, an explicit position equal to the
  launch is accepted and kept once; positions before it are refused (§§5.3–5.5).
  `with_input` keeps the resolved remaining observations (§5.6). §5.7 and §2 state that
  replay for interior launches stays refused. §7 tests direct construction on and off a
  zero-based lattice, and `with_input`.
- **R5 (settings replacement and reuse):** taken. §5.6 defines adoption to include the
  setter, `with_*`, scans and reseeds; validates against the receiving run every time;
  resolves before comparing; makes resolution idempotent; explains reuse at another
  length and how to pass a count without supplying both. The refusal to change a
  translated run's planes stays. §7 tests each.
- **R6 (declared end on resume):** taken in part. §5.9 resumes to the declared end, the
  last requested position in `configuration["save_planes"]`, distinct from the last
  committed plane and a terminal checkpoint, through both `driver/resume.py` and
  `recipe.resume_file` (§6). The legacy-file refusal and its message are dropped: under
  the compatibility constraint no path for pre-SEP files is kept, and they meet the
  existing hash refusal. `with_input` recovery is therefore not described.
- **R7 (authored recipes versus resolved provenance):** taken. §5.9 keeps authored text and
  export as written, and makes the resolved recipe write only canonical positions at the
  route's level, nested `[build.settings]` included, with defaults (the TDSE default too)
  written out. §7 adds the round trips and the default-change hash check.
- **R8 (front end):** taken, as D7. §5.10 defines the two modes, the switch as one undoable
  edit, length changes in each mode and loading. Q7 item 8 separates the two count cases
  (a legacy object-sheet count of 80 is `n_save = 81`). §7 lists the front-end tests.
- **R9 (converted lists are not identical):** taken, within the compatibility constraint.
  §7 limits structural parity to the resolver's own array and separates unchanged numbers
  from unchanged hashes; §8 treats a converted example as a migration and requires
  regenerated galleries. Hash preservation is no longer required, so lists are not kept
  for identity's sake and nothing is rounded to force a match.
- **R10 (endpoint equality):** taken, as D5 and D8. §5.4 normalizes within `LANDING_ULPS`
  ulps of either endpoint, keeps one endpoint observation and refuses positions genuinely
  outside, with the `0.1 + 0.2` and `np.nextafter` cases. §5.5 completes the refusal table
  (non-finite, unsorted, repeated, non-integral and boolean counts); an empty list is
  accepted as endpoints only (D8). §7 tests below, at, above and just outside the bound,
  and that resolving twice adds no plane.
