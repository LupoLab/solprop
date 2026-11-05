# solprop GUI: design

Status: **the design of the front end being built**, in solprop's repository as the `gui/`
distribution `solprop-gui` (decision D2). First drafted on 2026-10-31 in a separate
repository, revised the same day after the design review of section 18 (response in
section 19), and on 2026-11-01 after the solprop proposals it rests on were accepted. Moved
here on 2026-11-02, when those proposals had been delivered; section 21 records where what
solprop delivered differs from the text, and the text has been corrected where it matters
to the build. Two throwaway prototypes (NiceGUI and Qt, both around `prop_capillary`)
preceded it; they are not kept in the repository. Planned work is in GitHub issues: a
tracking issue for the front end and one issue per milestone of section 16.

**Revised on 2026-11-05 to a workflow per problem class** (D14), and Qt reaffirmed after a fairer
comparison with the web (D1). Section 22 records why. Sections 5 to 11 now describe the front end
being built, not the stepper in draft PR #484, whose code remains the starting point for the
layers without Qt. Sections 18 to 21 are records, and their section numbers still resolve.

**Starting work on the GUI? Read section 0, then section 22, then section 21.** Much of this design depends
on solprop's recipes and front-end hooks, and what solprop delivers takes precedence over
the names used here. The rules for working on the package are `gui/AGENTS.md`.

This document describes what the application is, how it looks and behaves page by page,
how it maps onto solprop, which changes to solprop would make it simpler and better, and
the options for its process architecture. Section 14 lists the open questions; answers
become decisions in section 2.

Contents

0. What solprop provides: SEP 40, 41 and 42
1. Purpose and principles
2. Decisions so far
3. Users and the work they do
4. Core concepts: recipe, run, workspace
5. The window and the route through it
6. Setup: one sheet per problem class
7. Medium, and Design a waveguide
8. Physics
9. Pulse, beam and grid
10. The Run stage
11. Analyse
12. Cross-cutting design: parameters, validation, help, files, scans
13. Architecture and implementation options
14. Open questions
15. Changes proposed to solprop
16. Phasing
17. Lessons from the prototypes
18. Design review, 2026-10-31
19. Response to the design review
20. Changes from the second SEP 41 review
21. What solprop delivered, and what changed here
22. Workflow review and stack decision, 2026-11-04 and 05

---

## 0. What solprop provides: SEP 40, 41 and 42

The GUI holds presentation only, and solprop holds the physics, the defaults and the rules
(section 1). Three solprop Enhancement Proposals provide what that needs, and all three are
`Final`:

| Proposal | What it gives the GUI | Implementation pull request | Tracking issue |
|---|---|---|---|
| **SEP 42**, *Resuming by what matters to the run* | Refusals that name what changed; ADR-066 has since replaced its comparison with an exact hash | [LupoLab/solprop#336](https://github.com/LupoLab/solprop/pull/336) | [LupoLab/solprop#142](https://github.com/LupoLab/solprop/issues/142) |
| **SEP 40**, *Transverse absorbing boundaries for free-space propagation* | The free-space edge: no real-space collar by default (`collar_fraction` 0), a collar on request with its design check | [LupoLab/solprop#71](https://github.com/LupoLab/solprop/pull/71) | [LupoLab/solprop#124](https://github.com/LupoLab/solprop/issues/124) |
| **SEP 41**, *Recipes, and the hooks a front end needs* | The recipe, and changes S1–S18: metadata, validation, observer, stop, previews and the rest | [LupoLab/solprop#376](https://github.com/LupoLab/solprop/pull/376) | [LupoLab/solprop#143](https://github.com/LupoLab/solprop/issues/143) |

Section 21 lists where the delivered interface differs from what this design assumed, and
the solprop changes the GUI still needs, with the milestone that needs each.

**Planned work is in GitHub issues (solprop's ADR-062).** Search solprop's open issues
before reporting something as missing or planning a change to solprop. Finished-run
extension, which this design would like one day, is
[LupoLab/solprop#141](https://github.com/LupoLab/solprop/issues/141).

**Finding out what was actually done.** The three files are under `docs/dev/seps/archive/`:
`sep-0040-transverse-absorbing-boundaries.md`, `sep-0041-recipes-and-front-ends.md` and
`sep-0042-tolerant-resume.md`. For each one:

1. **Read the header.** **Status** says how far it got (Accepted, In progress, Final), **Implementation**
   names the branch and **Manual pages** names the pages that document it.
2. **Read Part II, the implementation record.** Part I is frozen at acceptance, so a change made during
   implementation appears only here: each phase's report lists what was delivered, deviations, decisions,
   limitations and open questions.
3. **Read Part III** for why things are as they are: the reviews, the responses to them and the maintainer's
   decisions.
4. **Check the code and the manual**, which win over everything. Start with `solprop.recipe`,
   `solprop.capabilities`, `solprop.output.resumability` and solprop's `CHANGELOG.md`.

**When they differ, solprop wins.** Every solprop name in this document (`Recipe.pin_roots`, `PlaneView`,
`resumability`) is what the accepted proposals said on 2026-11-01. Where the delivered interface differs,
correct this document rather than working around solprop, and add a row to section 21 saying what changed. If the GUI
seems to need a rule or a default of its own, that is a sign something belongs in solprop.

**Numbering.** `S1`–`S18` are SEP 41's changes, listed in section 15, and the same labels appear in that SEP.
Section 3's user scenarios use the same `S` numbers, so "Serves S1, S3" at the head of a page means scenarios, and
"S17" or "SEP 41 S17" elsewhere means a change.

**Where each is used.**

| Solprop item | Used in sections | SEP phase |
|---|---|---|
| S1 recipe, paths, examples | 4, 5.2, 8.2, 12.5 | SEP 41 phase 1; object routes in phase 4 |
| S1 pins, S7 design solvers | 7.5, 13.3 | SEP 41 phase 5 |
| S2 recipe in the output, resume from a file | 4, 11.1, 13.1 | SEP 41 phase 2 |
| S3 plane observer | 10.2, 13.1 | SEP 41 phase 3 |
| S4 scan progress | 12.7 | SEP 41 phase 3 |
| S5 mode launch, S6 displacement and tilt | 9.2 | SEP 41 phase 4 |
| S8 capabilities, S9 metadata, S10 validation | 8.3, 9.3, 12.1, 12.3 | SEP 41 phases 1 and 5 |
| S11 cost estimate | 10.1 | SEP 41 phase 3 |
| S12 `load_spectrum` | 9.1 | SEP 41 phase 1 |
| S13 geometry drawings, S14 keyword gaps | 6.2, 15 | SEP 41 phase 6 |
| S16 result capabilities | 11.1, 11.3 | SEP 41 phase 3 |
| S17 stop flag, `resumability` | 4, 10.1, 10.2, 13.1 | SEP 41 phase 2 |
| S18 previews | 7.4, 9.2, 13.3 | SEP 41 phase 4 |
| Refusals that name what changed | 4, 13.1 | SEP 42 phase 1, ADR-066 |

**What each milestone of section 16 needs** from the three proposals, all delivered; the
further solprop changes each needs are in 21.2.

| Milestone | Needs |
|---|---|
| M1 Lifecycle | SEP 42 phase 1; SEP 41 phases 1–3 |
| M2 Explorer | SEP 41 phases 4 and 5 (previews and pins) |
| M3 Guided routes | SEP 41 phase 4 |
| M4 Results depth | SEP 41 phase 3 |
| M5 Free space | SEP 41 phase 4 |
| M6 Scans | SEP 41 phase 3; SEP 42 phase 2 |

## 1. Purpose and principles

The GUI should let someone set up, run and understand a solprop simulation without
writing Python, and let an expert do the same faster than a script would, while never
becoming a dead end: everything made in the GUI can leave it as a recipe file or a
Python script, and every result file solprop writes can be opened in it.

Principles, in priority order:

1. **solprop does the physics; the GUI does none.** Every number the GUI shows comes from
   a public solprop call. The GUI holds no dispersion formulas, compatibility rules or
   defaults of its own; where it needs one that solprop doesn't expose, the fix is a
   solprop change (section 15), not a copy in the GUI.
2. **Same machine first.** The local desktop experience must not be compromised by
   remote ambitions. Remote use, in the short term, is the exported recipe or script run
   on the remote machine by hand.
3. **The setup is a document.** A setup is a recipe (TOML) that can be saved, diffed,
   versioned, reopened, run headless and turned into a Python script. Results files carry
   the recipe that made them.
4. **Never block the window.** Anything that can take more than ~50 ms (importing
   solprop, solving for a pressure, building a mode set, running) happens in another
   process. The window stays responsive at all times.
5. **Layered disclosure.** Each sheet shows the few choices that matter to a newcomer, with
   defaults that are solprop's own; "Advanced" folds open in place to reveal the rest.
   No choice is ever hidden from an expert.
6. **Explain everything where it appears.** Every control has a tooltip; every page has a
   help button; every warning says what to do about it.
7. **Honest about cost and validity.** Before a run the GUI shows what it will cost
   (grid size, memory, rough time); during it, the boundary monitor and step statistics;
   after it, solprop's warnings — never silently.

Non-goals (for now): editing arbitrary Python responses or callables in the GUI; a
general plotting package; running on remote machines from the GUI (section 13 keeps the
door open); replacing scripts for production campaigns.

## 2. Decisions so far

| # | Decision | Source |
|---|---|---|
| D1 | Qt for Python (PySide6) with pyqtgraph for interactive plots; matplotlib (solprop's own `solprop.plotting`) for publication figures. NiceGUI is not pursued. | prototype comparison, 2026-10-31; reaffirmed 2026-11-05 after two equal spikes (22.4) |
| D2 | A separate distribution, `solprop-gui` (import package `solprop_gui`), built from `gui/` in solprop's repository as a uv workspace member, as `solprop-native` is from `native/`. solprop's `gui` extra pins it to solprop's version and it pins solprop to its own, so the two are developed, tested and released together and cannot drift. Not a subpackage of `solprop`, whose repository-wide structural checks (the module list, the structure figures, the API reference, the import-time type hook, the 100 % coverage shards) would otherwise take in every Qt module. | 2026-10-31; revised 2026-11-02 (was a separate repository) |
| D3 | Designed for both newcomers and experts through layered pages (principle 5). | 2026-10-31 |
| D4 | Parameter scans are designed in now (data model, results browser) and built after single runs. | 2026-10-31 |
| D5 | A TOML recipe is the saved form of a setup, embedded in every results file; "Export as Python" produces an equivalent plain solprop script. | 2026-10-31 |
| D6 | Same machine first; remote work stays scripted in the short term. | 2026-10-31 |
| D7 | solprop's API and way of working may change to help the GUI (nothing is public yet, so without aliases). | 2026-10-31 |
| D8 | The recipe lives in solprop (`solprop.recipe`, change S1): its format, builders, Python export and `solprop run`; the GUI adds only presentation. Q1 closed. | 2026-10-31 |
| D9 | This design is a living page under `docs/dev/gui/`, not a SEP; the work is a tracking issue and one issue and branch per milestone of section 16. A change to solprop the GUI needs follows solprop's own rules (an issue, or a SEP for a major change) and is merged on its own. | 2026-11-02 |
| D10 | Questions Q3–Q12 of section 14 are decided as recommended there. | 2026-11-02 |
| D11 | The quality bar is split by layer: `model`, `registry`, `service` and `help` meet solprop's standard in full, 100 % coverage included; `ui` and `views` are typed, documented and linted the same way and tested by `pytest-qt` smoke tests and offscreen screenshots, outside the coverage measure. | 2026-11-02 |
| D12 | The GUI is built out and tested locally, milestone on milestone, and iterated on by hand before any of it is merged: the milestone branches stack locally and are not pushed while that goes on, and the local gate (`poe check-gui`) stands in for CI. The solprop changes the GUI needs still go to main as their own pull requests. | 2026-11-02 |
| D13 | The first hands-on review (2026-11-03) reshaped the front end. The start page is a gallery of the geometry classes, each a simple cross-section, grouped and titled; Cartesian and line free space, and rectangular and slab waveguides, are classes of their own, and planar guides are hidden for now. One set of plot options (spectral axis, typed limits, scale, a bandpass for the time domain) sits at the bottom left of Propagate and Results. Results show one run, with a 2 by 2 statistics grid of up to two quantities a panel and an interactive spectrogram (`processing.get_spectrogram`). Compression moves to a Post-process step after Results: a chain of solprop.optics elements, each fixed or searched by solprop's compressor of its kind, one at a time. **Replaced in part by D14:** the gallery, the shared plot options and the steps go; the spectrogram and the compressor chain stay, as pages. | 2026-11-03 |
| D14 | A workflow per problem class replaces the stepper: Home, then Setup, Run and Analyse. A class is a definition held as data (6.1). The first are capillary, radial filament, solid-core fibre and antiresonant fibre; everything else is under Other geometry on the generic form. | 2026-11-04 |
| D15 | Setup is one sheet per class, with live derived numbers and a preview plot, and a target can be typed into a derived number (7.5). Analysis is a rail of pages with one large page shown and one shared z scrubber. | 2026-11-04 |
| D16 | The main user is a physicist who does not script: physics language by default, solprop's terms under Advanced. The usual work is one run at a time; comparison is rare, and scans are a mode of their own. | 2026-11-04 |
| D17 | Design a waveguide is a workflow of its own, needing no pulse, and it never replaces a setup. Tuning a setup is the sheet's job. | 2026-11-04 |
| D18 | The look is calm and light, with white plots, in IBM Plex Sans, carried in the package under the Open Font License. A filament's first screen is its fluence r-z map with its spectrum and spatio-spectral views. The beam is set by waist and focus, with a lens as an option. | 2026-11-04 |

## 3. Users and the work they do

The design is checked against these scenarios; each later section names the ones it
serves.

- **S1 — A student reproduces a textbook effect.** Resonant dispersive-wave emission in a
  helium-filled capillary: pick the capillary, gas, pressure, a 10 fs pulse, run for a
  minute, watch the dispersive wave appear, read off its wavelength and energy.
- **S2 — An experimentalist designs a fibre.** No propagation at all: what pressure puts
  the zero-dispersion wavelength at 450 nm in a 75 µm-core antiresonant fibre, what is
  the loss there, what is the soliton order at 30 µJ? Export the dispersion curve as CSV
  for a paper figure. (*Design a waveguide*, section 7.4.)
- **S3 — An expert sets up a multimode run, then leaves.** Configure a 4-mode capillary
  run with a pressure gradient and Raman in N₂, check the cost estimate, export the
  recipe, run it on the workstation or cluster with a script, later open the results file
  in the GUI.
- **S4 — Anyone revisits an old result.** Open a results file (from the GUI or from a
  script), browse planes, make a spectrogram at the fibre output, compress the pulse,
  export a figure.
- **S5 — Teaching and demos.** A lecturer changes the energy live in front of a class and
  re-runs a short simulation; the display must be legible on a projector.
- **S6 — A free-space filament.** A radially symmetric beam focused into argon:
  waist, focus position, pressure, Kerr plus plasma; watch fluence and on-axis spectrum
  evolve.
- **S7 — A scan (later).** Energy × pressure grid of the S1 setup, then a map of
  dispersive-wave energy over the grid, clicking any point to open its run.

## 4. Core concepts: recipe, run, workspace

- **Recipe.** The complete, declarative description of one simulation: route (which
  solprop entry point), geometry, medium, nonlinear model, input fields, grid, numerics
  and output choices. Values are what the user typed (with units), with pins stated as
  their targets, not resolved solprop objects. A recipe is a TOML file (section 12.5). It
  is the GUI's model: every page edits part of the current recipe, and nothing else holds
  setup state.
- **Authored and resolved.** Following solprop's improvements and reproducing an old run
  are different needs. The *authored* recipe keeps only what the user set, with units and
  pins, so it follows new defaults. The *resolved* recipe that solprop stores beside it in
  every results file freezes what rebuilds the run: what the user set, every solved pin, the
  noise seed, the data files and the defaults solprop can write back and still get the same
  run. "Open setup from this run" shows the resolved values; adopting the current defaults or
  re-solving the pins is a separate, explicit action ("Update to current solprop"), which lists
  what would change. A frozen setup can still be refused at resume if the materials data or the
  physics differ; solprop names the fields that changed (ADR-066).
- **Run.** One execution of a recipe revision: a results file (solprop HDF5) with both
  recipes embedded, a log, and two separate states. The *execution state* is the GUI's:
  queued, starting, running, stopping, stopped, complete, failed, worker lost. The
  *outcome* is solprop's `RunOutcome.status` (running, complete, budget exhausted,
  failed), which has no "stopped". Whether a run can be resumed is never assumed from
  either: it is asked of solprop (`resumability`, SEP 41 S17), which gives the mode
  (`resume=True`, from an interpolated plane, or forced, for a file already forced) or the
  reason it cannot (no checkpoint before a failure, no work left, another output schema, a
  writer still alive), and whether the file can be rebuilt from itself. A changed grid, a
  wider time window included, is always a new run.
- **Workspace.** A folder the user picks (default `~/solprop-runs`), holding one folder
  per run: `<date>-<title>/` with `run.h5`, `recipe.toml`, `run.log` and a thumbnail. The
  GUI keeps an index for fast browsing but the folder is the truth: copying a run folder
  elsewhere loses nothing.
- **Project file.** Just a recipe saved by the user wherever they like ("Save setup
  as…"). Opening a results file offers "Open setup from this run", which loads its
  embedded recipe as a new, unsaved project.

## 5. The window and the route through it

Rewritten on 2026-11-05 (D14, section 22). A newcomer sees one question at a time, and a
problem class decides what is on screen. The sketches in sections 5 to 11 show layout; their
numbers are illustrative.

### 5.1 Layout

```
Home ─┬─ New simulation ──► Setup ──► Run ──► Analyse
      │    (a class)          ◄── edit and rerun ──┘
      ├─ Design a waveguide ──► "Simulate this" ──► Setup
      ├─ Open results ──► Analyse
      └─ Scans (later, a mode of its own, started from a sheet)
```

```
Capillary, helium                  Setup   Run   Analyse         [▶ Run]   ⋯
┌───────────────────────────────┬───────────────────────────────────────────┐
│ the current stage: a sheet,   │                                           │
│ or the rail of analysis pages │        plots, as large as they can be     │
│                               │                                           │
└───────────────────────────────┴───────────────────────────────────────────┘
```

- **Three stages.** Setup, Run and Analyse are the only navigation inside a simulation. Each
  shows one thing; none has tabs of views that do not apply. Finishing a run moves to Analyse
  by itself, and editing a finished run's setup goes back to Setup without losing the results.
- **One primary action.** The button at the top right is what comes next: Run, Stop or Rerun.
  Run, Stop and Resume appear nowhere else.
- **One overflow menu** (⋯) holds Save, Save as, Export recipe, Export Python and Open setup
  from this run. The platform's menu bar repeats these, with the keyboard shortcuts.
- **No permanent console.** The log moves to the Run health page (11.3). A warning appears
  beside what it concerns and only when something is wrong. The recipe and its Python
  equivalent open in a drawer on request.
- **Controls sit beside their plots.** A control that changes a plot is on that plot's page,
  never on another stage or in a shared panel.

The step bar, the left settings panel, the view tabs, the messages panel and the status bar of
the stepper in draft PR #484 all go.

### 5.2 Home

Shown when nothing is open. It asks what you want to do:

- **New simulation**: one tile per problem class (6.1), each opening a sheet that can run at
  once, and *Other geometry* for every other route.
- **Design a waveguide** (7.4), for questions that need no pulse or run.
- **Open results**: a file, or one of the recent runs listed here with its state.
- **Open a setup**, and the recent setups.
- **Scans**, later (12.7).

A class's examples (`Recipe.examples()`) are a short list on its sheet, not a dialog.

### 5.3 Small screens

The design target is a 13-inch laptop (1280 × 800 logical pixels) without scrolling the
main controls; it must remain usable at 1024 × 700 and look good on a 4K monitor and a
projector (S5).

- A sheet is one column of sections beside its plot. On a narrow window the plot moves below.
- An analysis page shows one large view at a time, and the rail can shrink to icons.
- Fonts and plot line widths follow the system scale; a "Presentation" toggle enlarges
  everything for S5.

## 6. Setup: one sheet per problem class

Serves S1, S3, S5, S6. The class fixes the *route*, which solprop entry point builds the
simulation, and so decides what the sheet asks and which analysis pages exist.

### 6.1 Problem classes

A class is a definition, held as data in the layer that imports no Qt (13.5):

| It declares | For example, on a capillary |
|---|---|
| Its sheet: sections and fields, each mapped to a recipe path | Capillary, Gas, Pulse, Physics |
| Its derived numbers: kernel calls | Soliton order, ZDW, RDW, peak intensity |
| Its preview plot | β₂ against wavelength, with the pump, ZDW and RDW marked |
| Its analysis pages, each with the capabilities it requires | Overview, At this plane, Modes when multimode, … |
| The statistics those pages need recorded | Electron density, ionisation fraction |
| Its starting point: a solprop example recipe | `rdw` |

This replaces about 30 branches on the route in the views (22.1): a new class is a new
definition, and no view asks which class it is drawing. The first four classes:

| Class | solprop route | Notes |
|---|---|---|
| Capillary | `prop_capillary` | Mode-averaged, or multimode through the sheet's Modes field, so one tile covers both. A full-transverse rule and a coherent wall are under Advanced. |
| Radial filament | `prop_radial` | Needs a filament example (21.2). |
| Solid-core fibre | `prop_gnlse` | Taylor dispersion, then step-index and tables as recipes allow them (#472, #473). |
| Antiresonant fibre | `prop_capillary(lattice=TubeLattice(...))` | Adds the lattice to the sheet and a loss readout. |

**Other geometry** holds every other route on the generic form, built from solprop's metadata
as in #484:

| Route | solprop | Notes |
|---|---|---|
| Rectangular hollow guide | `prop_waveguide([RectMode…])` | |
| Hollow slab | `prop_waveguide([SlabMode…])` | Values per unit width (J/m, W/m); the GUI labels units accordingly. |
| Slab resolved in y | `prop_planar(PlanarGrid…)` | |
| General modal (vector) | `prop_waveguide((), provider=ModalGuide…)` | Step-index or graded-index vector guides with material regions. |
| Free space, Cartesian | `prop_free(FreeGrid…)` | Full x–y. |
| Free space, line | `prop_free(LineGrid…)` | x only, per metre of y. |

Any of these can become a class when its users need one. A route with no recipe form is shown
disabled with the reason (21.1).

### 6.2 The sheet

One screen, with everything the class needs to run on it:

- **It opens runnable.** Its values come from a solprop example recipe, so the GUI has no
  defaults of its own (principle 1). Errors from solprop's check appear together, beside their
  fields.
- **Derived numbers are live.** They and the preview plot update as you type, off the window's
  process, the latest request winning. The targets are 150 ms for the numbers and 300 ms for
  the curve, which the spike met on the capillary (22.3).
- **A target can be typed into a number** (7.5).
- **One line of help** under the focused field explains it. A longer page opens from the help
  button (12.4).
- **Advanced folds open in place**, section by section. "Find any setting" reaches any keyword
  without browsing. solprop's terms appear there; the basic rows use physics language.
- **A Ready line** states solprop's check, the rough time and the output size. On a radial run
  the size matters, since files can reach gigabytes.

**Cross-section.** Where the shape matters (antiresonant, Other geometry) the plot area can show
the user's actual geometry to scale, from `plotting.geometry.outline()`, updating as they type.
Qt's SVG renderer ignores the guide's CSS palettes, so the kernel sends the shapes as data and
the window paints them (21.1).

### 6.3 Profiles, and changing class

**z-profiles.** Any parameter solprop accepts as a `ZProfile` (radius for a taper, half widths,
pressure, temperature, preionisation) gets a small profile button beside its field: *constant*
(default), *linear*, *piecewise (table of z, value)*, or *from file*. The field then shows a
sparkline of the profile; clicking it opens a profile editor (a table and a plot with draggable
points). The same editor is used everywhere (12.2).

**Changing class.** Setups do not convert. A new class is a new sheet from Home, and the current
setup is kept until it is closed. Other geometry changes route as #484 does: values that still
apply stay, and the ones that will be dropped are listed, with an undo.

### 6.4 Capillary

Serves S1, S3, S5.

```
Capillary                       Setup  Run  Analyse     [▶ Run]  ⋯
┌─────────────────────────┬──────────────────────────────────────┐
│ Capillary               │ What this gives                      │
│  Core radius [125] µm   │  Soliton order  [ 2.5 ]   ZDW  423 nm│
│  Length      [1.0] m    │  Peak power     156 GW    RDW  154 nm│
│  Modes       [ 1 ]      │  Peak intensity 3.7e14 W/cm²         │
│ Gas                     │  Fission length ~0.3 m               │
│  [Helium ▾] [1.0] bar   │ ──────────────────────────────────── │
│ Pulse                   │       β₂ vs wavelength               │
│  Wavelength [800] nm    │    (pump, ZDW and RDW marked;        │
│  Duration   [ 10] fs    │     curve moves as you type)         │
│  Energy     [275] µJ    │                                      │
│ Physics                 │                                      │
│  Kerr ✓  Plasma ✓  Raman│ ✓ Ready · ~40 s · 2 MB               │
│ ▸ Simulation window  ▸ Advanced                                │
└────────────────────────────────────────────────────────────────┘
```

The numbers come from `tools.capillary_params` and `tools.lambda_rdw_all`, and the curve from
`modes.base.beta2` on the sheet's mode, with no recipe built (22.3).

### 6.5 Radial filament

Serves S6. The same frame with different content:

```
Filament                        Setup  Run  Analyse     [▶ Run]  ⋯
┌─────────────────────────┬──────────────────────────────────────┐
│ Medium                  │ What this gives                      │
│  [Argon ▾] [1.0] bar    │  P / P_cr       3.0                  │
│  Propagate  [25] cm     │  Critical power 5.1 GW               │
│ Beam                    │  Rayleigh range 35 cm                │
│  Waist      [300] µm    │  Collapse (Marburger) ~13 cm         │
│  To focus   [  0] cm    │ ──────────────────────────────────── │
│  ▸ From a lens…         │     linear beam radius vs z          │
│ Pulse                   │   (focus and collapse marked)        │
│  Wavelength [800] nm    │                                      │
│  Duration   [ 30] fs    │                                      │
│  Energy     [2.0] mJ    │                                      │
│ Physics                 │                                      │
│  Kerr ✓  Plasma ✓       │ ✓ Ready · ~4 min · 35 MB             │
│ ▸ Resolution  ▸ Advanced                                       │
└────────────────────────────────────────────────────────────────┘
```

The beam is set by waist and focus (`SpatialPulse.propz` carries the distance to focus, with its
sign changed), with lens input as an option. The beam numbers and the lens conversion are physics,
so they come from solprop (`tools.beam_params`, `tools.beam_through_lens`; 21.2).

### 6.6 Solid-core and antiresonant fibre

Each follows the pattern. **Solid-core** has no gas section: fibre parameters (γ and Taylor
coefficients, then a glass and core once #472 is done) with soliton figures and a dispersion
curve. Its Physics section is Kerr and Raman. **Antiresonant** is the capillary sheet with the
lattice (tube count, wall thickness, gap or diameter ratio), a loss readout and the cross-section.

## 7. Medium, and Design a waveguide

Serves S1, S2, S3, S6. Sections 7.1 to 7.3 say what a sheet's medium section can hold; the first
row or two show by default and the rest are under Advanced.

### 7.1 Gas fills

- **Species**: a dropdown from solprop's own list (`[g.value for g in Gas]`: Air, He,
  Ne, Ar, Kr, Xe, N₂, H₂, D₂, O₂, CH₄, SF₆, N₂O), displayed with subscripts; Air expands
  to its components.
- **Mixture**: "Add component" turns the single species into a table of species and
  mole fractions (`gas={species: fraction}`), or of partial pressures
  (`Mixture.partial_pressures`) in advanced mode.
- **Pressure** (bar, as in solprop): constant; *gradient* (`(p_in, p_out)` →
  `PressureGradient.two_point`); *stages* (z and pressure table → `from_stages`); or
  *from gas flow*: chamber pressures and positions → `gasflow.steady_fill`, with
  `flow_report` shown beside it (throughput, mass flow, Reynolds and Knudsen numbers per
  stage, choked ends). Advanced: `pressure_floor`, `vacuum_end`.
- **Temperature** (K, default 293.15): constant or profile.
- **Material model policy** (advanced): `MaterialSelection` — linear, electronic,
  Raman and ionisation policies — with "recommended" defaults. These live with the
  medium because they choose data, not physics; the physics section links here.

### 7.2 Solid media

For solid-core routes: core and cladding glass from `glass_names()` (silica, CaF₂, MgF₂,
BK7, sapphire, BaF₂; the ordinary/extraordinary axis where needed), or a constant complex
index. For GNLSE with Taylor dispersion there is no medium as such: the page becomes a
table of β₂, β₃, … (with units fs²/mm etc.) and γ or n₂, plus loss in dB/m.

### 7.3 Free-space media

`SpatialMedium`: a gas (with pressure, and a pressure profile on the radial route only),
a glass, or a constant index.

### 7.4 Design a waveguide

S2 says this is often the whole job, and it needs no pulse or run: which pressure puts the ZDW at
450 nm in this fibre, and what is the loss there? So it is a workflow of its own, opened from
Home. Tuning a setup belongs to the sheet (6.2).

```
Design a waveguide                          [Simulate this ▸]  ⋯
┌───────────────┬──────────────────────────────────────────────┐
│ Radius [125]µm│ [Curves] [ZDW map] [Phase matching] [Loss]   │
│ Gas [He ▾]    │                                              │
│ Pressure [1]  │     the large plot                           │
│ Sweep:        │                                              │
│  pressure     │                                              │
│  0.2 to 5 bar │ ──────────────────────────────────────────── │
│ ▸ Pulse (opt.)│ At 800 nm: β₂, β₃, loss, A_eff, γ  [CSV]     │
└───────────────┴──────────────────────────────────────────────┘
```

- **It works without a pulse.** Its inputs are all on one panel, and a missing energy blocks
  nothing. In #484 a curve needs a whole valid simulation, because `preview.modes` builds the
  recipe.
- **It never replaces a setup.** *Simulate this* carries the design to a new sheet. In #484 opening
  the explorer replaces your setup with a helium capillary.
- **Clicking a curve or map point sets the design**, so a result can be adopted. In #484 a curve
  or sweep leads nowhere.
- **Two-value designs live here.** `tools.capillary_design` (pressure and energy from a ZDW and a
  soliton order) writes both as literals. This replaces #484's Design dialog.

What #484 built stays, since its plots are good:

- **Quantities**, from `solprop.modes.base` and `tools`: n_eff − 1, β₁ (group index or
  group delay), β₂ (GVD, also as dispersion parameter D), β₃, loss (dB/m, loss length,
  transmission over the length), A_eff, γ, ZDW; also as functions of z for tapers and
  gradients.
- **Axes**: wavelength, frequency, photon energy; linear or log; units selectable.
- **Sweeps**: overlay curves for a list of one parameter (pressure, radius, gas,
  mode order), or show a 2-D map (ZDW vs pressure and radius).
- **Markers**: ZDW, pump wavelength, phase-matched RDW wavelength
  (`tools.lambda_rdw_all`), phase-mismatch curves (`phase_mismatch`).
- **Readouts**: a table at the pump wavelength, and the soliton figures of the optional pulse
  (`soliton_params`: N, L_D, L_NL, L_fiss, critical power).
- **Export**: CSV of every curve, figure as PNG/SVG/PDF, "Copy as Python".

### 7.5 Typing a target into a number

The sheet is two-way: click a derived number, type the value you want, and an input moves to
give it. That input then reads "solved for N = 3", with the target editable where you typed it.
Clearing the target returns the input to a plain value.

This writes solprop's `[pins]` (SEP 41 S1), so the GUI, `solprop run` and the Python export agree,
and a saved setup keeps the intent. It replaces #484's pin switch on the input field. What can
move is solprop's rule, and where more than one input can, the number asks which:

| Type into | What moves | solprop |
|---|---|---|
| ZDW | Constant pressure, or radius | `zdw` pin (`tools.pressure_zdw`, `tools.radius_zdw`) |
| RDW wavelength | Constant pressure, or radius | `rdw` pin (`tools.pressure_rdw`) |
| Soliton order | A pulse's energy or peak power | `soliton_order` pin |

Pins exist on the capillary, with or without a lattice, and are refused on the object routes
(21.1). Other numbers on a sheet are read-only until solprop offers a pin for them, as with
P / P_cr on a filament (21.2).

- the pin holds its target, its search range and which root to take (lowest, highest or
  a chosen one); when solprop finds several roots the number offers them, and when it finds
  none it shows the attainable range as the error;
- pins resolve after the literal values, in two fixed stages: at most one pin on a parameter
  of the guide or fill, then the pins on pulses, which read it, so no cycle can be written. A
  pressure gradient cannot be pinned;
- a scan re-solves the pins at every point;
- the resolved value goes into the resolved recipe of every run.

## 8. Physics

Serves S1, S3, S6.

### 8.1 Layout

A sheet's Physics section shows the class's switches in one line (Kerr, Plasma, Raman). Advanced
opens one row per physical effect, each with an on/off switch, a short description, and its
options in an expandable area:

```
Field representation   (•) Full field (carrier-resolved)   ( ) Envelope        [?]
─────────────────────────────────────────────────────────────────────────────────
☑ Kerr effect            third-harmonic terms: [auto ▾]   model: [recommended ▾]
☑ Photoionisation & plasma  rate: He → PPT (recommended)  integration: [band ▾]
    preionisation: [0]
☐ Raman                  (no Raman-active species in He — not available)
☐ TDSE atomic response   (research; helium, capillary only)                expert
```

- **Field representation** (`envelope=` on the modal routes): full field is solprop's
  default; envelope is faster for narrow-band problems. The choice changes which
  options exist (THG defaults to off and envelope plasma to cycle-averaged), so it sits
  at the top.
- **Kerr**: on/off (`kerr`); third harmonic (`thg`, default "auto" = full field only);
  model (`MaterialSelection.electronic`: recommended, carrier-kerr, symmetric-kerr,
  gas-polynomial — the last two the dispersive Kerr evaluators); `kerr_form` for the
  general modal route.
- **Plasma**: on/off (`plasma`, default on for the full field); rate model per species
  (recommended, or a comparison model: `ppt_ilkov`, `ppt_larochelle`, … for atoms,
  MO-ADK variants for molecules), shown with the status solprop reports
  ("recommended", "comparison", "untested"); preionisation (number or profile, per
  species); integration ("band" or "trapezoid", advanced); `ionisation_field_range`
  (advanced).
- **Raman**: auto (solprop's rule: on for N₂, O₂, H₂, D₂, CH₄, SF₆), off, on, or per
  species with `MolecularOptions` (rotation, vibration, temperature, J range; advanced).
  For GNLSE: on/off and fraction (silica).
- **TDSE** (expert, capillary, helium only): model SAE/haCC, Kerr model, device;
  enabling it locks the options solprop refuses alongside it (plasma, Raman, THG,
  preionisation) and sets the backend it needs.

### 8.2 One set of switches on every route

The modal routes take keyword switches; the free-space routes take an explicit list of
response objects. The GUI shows the same switches everywhere and the recipe stores the
switches; for radial, Cartesian and planar routes the builder calls
`nonlinear.mixtures.responses_for(mixture, kerr=…, plasma=…, raman=…, envelope=…)`.
On free-space routes solprop's own default is linear (no responses), and the GUI keeps
it: it has no defaults of its own (principle 1). New setups start from solprop's
example recipes (`Recipe.example`, SEP 41 S1), so the filament starting point with Kerr
and plasma on is a named solprop example the user chose, shown as such on Home and at the
head of the sheet.

### 8.3 Unavailable options

Options that can't be combined are disabled, not hidden, with a tooltip saying why
("Plasma forbids third-harmonic terms", "Envelope plasma needs the envelope
representation", "No Raman-active species in this fill"). These rules belong to solprop
(principle 1): the GUI asks solprop's capability query (S8) rather than encoding them.
S8 lists only the combination rules (route × keyword × species); rules that depend on
values come from `Recipe.check()` (section 12.3).

## 9. Pulse, beam and grid

Serves S1, S3, S5, S6.

### 9.1 Pulses

A sheet shows one pulse's wavelength, duration and energy. Advanced opens the list of pulses
("Add pulse" for pump–probe or two-colour), each a compact card:

- **Shape**: Gaussian (super-Gaussian order `m` advanced), sech², or *from data*.
- **Wavelength**; **duration** as FWHM (or `width` for sech, advanced); **energy** or
  **peak power** (a switch); **delay** `t0`.
- **Phase**: CEP, GD, GDD, TOD, … as a short table (Taylor coefficients about the
  pulse's own wavelength), with units fs, fs², fs³.
- **From data**: an import dialog for spectra (wavelength or frequency, intensity,
  optional phase) or temporal fields, from CSV/TXT with column picking and a preview,
  building `DataPulse.from_spectral_density` or `DataPulse.from_time`. Common formats
  (two-column spectra, FROG/SPIDER retrievals) are proposed as solprop loaders (S12) so
  scripts get them too.
- **CW** (advanced): `CWPulse` for seeding.

A preview plot beside the list, when Advanced is open, shows |E(t)|² and the spectrum of every pulse and their
sum, **on the simulation's actual grid**, with the window edges drawn, so a pulse that
doesn't fit is visible before the run.

### 9.2 Where the light goes in

- **Modal routes**: today the keyword routes put the pulse in the lowest mode, or in the
  mode `channel=` names. The design has a launch table, per pulse (a pump and a seed may
  go into different modes): a share and a phase per mode, labelled as solprop labels the
  modes (`HE1,1`, with the parity where both polarisations exist). It needs solprop
  change S5; until then it shows "fundamental mode" and explains. On Marcatili-type bases
  the shares are energy fractions. On a general modal basis the modes are not
  flux-orthogonal, so solprop normalises the total energy through the flux metric and the
  table labels the shares "coordinate shares, not an energy split".
- **General modal**: the launch table, or a free-space beam coupled in
  (`incident=SpatialPulse`, `coupling=CouplingModel(...)`); the two exclude each other.
- **Free space**: beam waist (1/e² intensity radius), super-Gaussian order, focus
  position as a distance before or after the waist (`propz`), polarisation (`jones`). A lens
  (focal length and beam on it) is an optional way to give the waist and focus (21.2).
  Cartesian: displacement and tilt (today only through `coupling.Alignment`; S6 puts them
  on the beam, with `Alignment`'s names; the tilt is a pure angle at every frequency).
  Planar: `waist_x`, `offset_x`. A preview shows the beam cross-section on the transverse
  grid (`Recipe.preview_input`, S18).

### 9.3 The grid

A sheet's *Simulation window* section (*Resolution* on a filament), folded by default, because
the example's values and the pulses determine it:

- **Wavelength range** (λmin, λmax) and **time window**, with an *Auto* button that asks
  solprop for a suggestion from the pulses and the route (S10 covers the suggestion).
- **Transverse grid** (free space: radius and points; planar: y extent and points).
- Advanced: `ffac`, `fft_size`, collar fraction, a ready-made grid.
- The resulting grid size, sample spacing and memory appear in the section and update
  immediately, and the output size on the Ready line (6.2).

### 9.4 Noise

`noise` (solprop's default, off, input, modified) and the seed, advanced, with the
default explained in the tooltip.

## 10. The Run stage

Serves S1, S3, S5, S6.

### 10.1 Before the run

There is no pre-flight page. The sheet's Ready line (6.2) states solprop's check, the rough time
and the output size, and Run starts from it.

- **Cost**: the size comes from solprop, and the time from similar past runs in the workspace, or
  it says it is unknown. Timing warm-up steps with `Simulation.estimate()` takes seconds, so it runs
  on request (21.1).
- **Numerics** (Advanced on the sheet): backend (from `backends.registry.available()`), precision,
  tolerance (`rtol`, with the route's default shown), integrator, save planes (`n_save` or a
  spacing), statistics (`StatsSpec`, with the route's default set and the class's required
  statistics ticked), threads (`SOLPROP_THREADS`), boundary check and absorber, snapshots.
- **Output**: run title and workspace folder (defaults filled in).
- **Resume** replaces Run for an existing run of the same recipe that solprop reports resumable
  (`resumability`); otherwise the reason is shown.

### 10.2 During the run

The Run stage is the class's Overview page (11.3) with a progress strip. There is no separate
live view to learn, so what you watch grow is what you then analyse.

```
Capillary, helium              Setup  Run  Analyse              [■ Stop]  ⋯
37 % ▕████████░░░░░░░░░░▏ z 0.37 / 1.00 m   ETA 0:42   step size ▁▂▃▅▅▃▂   boundary ● ok
┌──────────────────────────────────┬──────────────────────────────────────┐
│ Spectrum vs z (dB)               │ Power vs z                           │
│  (fills in plane by plane)       │                                      │
├──────────────────────────────────┼──────────────────────────────────────┤
│ Spectrum at z                    │ Power at z                           │
└──────────────────────────────────┴──────────────────────────────────────┘
```

- **Live maps** fill in at most four times a second whatever the run's speed. Each is a view
  specification (12.8), so its normalisation and reduction are stated, not implied. The z
  scrubber shows any saved plane while the run continues, and "follow" tracks the newest. On a
  filament they are the fluence map and on-axis spectrum (11.3).
- **Boundary monitor** lamp: green until the monitor speaks, amber on `BoundaryWarning`, red
  on `BoundaryError`, which the worker catches like any other warning or error. The suggested
  fix is "Widen the time window and rerun": a wider window is a new grid, so a new run, and
  resuming cannot recover light the old edge has already affected (it was the first failure met
  in the prototype).
- **On completion** the strip goes, the rail appears and you are in Analyse.
- **On failure** an advice card states the fix and keeps the planes reached, which stay open for
  analysis.
- **Stop** sends a stop request on the worker's control pipe; the worker sets solprop's
  stop flag (S17), which takes effect within one step or trial on every platform (on Windows
  a signal would terminate the worker outright). The run writes its state and `run` returns
  normally. The execution state is *stopping* until the worker reports, then *stopped*, and
  Resume follows `resumability`. If the worker hasn't stopped after a grace period the
  GUI offers to kill it; afterwards it asks `resumability` before calling the file
  resumable, since a killed run may have no restart state.
- **Detach** (with the service, 13.3) lets the run continue after the window closes; reopening
  the run re-attaches.

### 10.3 Several runs

The usual work is one run at a time, so the window shows one. Home lists the workspace's runs with
their state, and opening one shows it. Starting a second while one runs is allowed, with a warning
before oversubscribing the machine's thread budget. Campaigns are scans (12.7).

## 11. Analyse

Serves S1, S4, S5, S6; scans (S7) in 12.7. This is the hardest stage, because people want to look
at different things. So a class offers a rail of pages, one shown large, with a good default first.

### 11.1 Opening results

Analyse opens **any** solprop results file of a supported output schema, not only GUI runs: it
needs only solprop's `Output` interface (`open_file`). The class comes from the route in the file's
recipe, or from its data on a scripted run, and an unknown one gets a generic rail. If the file
carries a recipe, its sheet can be restored from it (the resolved values, section 4); if not, Setup
shows the run metadata read-only.

What a file can show depends on what it stores, not on its route. The GUI asks solprop
(`capabilities.output`, SEP 41 S16) and offers only the views whose inputs and physical
meaning are available: which planes kept fields and which only statistics; whether a
quantity is a physical flux or a coordinate quantity, and its unit (per unit width on
slab and line routes); whether general modal reconstruction has its guide and reference
tables; which operations solprop refuses, with the reason, shown on the disabled view.

### 11.2 Layout: a rail of pages and one z scrubber

```
Capillary, helium              Setup  Run  Analyse              [▶ Rerun]  ⋯
┌───────────────┬─────────────────────────────────────────────────────────┐
│ Overview      │ this page's controls: spectral axis, dB floor           │
│ At this plane │                                                         │
│ Spectrogram   │              this page's plots, large                   │
│ Along z       │                                                         │
│ Compress      │                                                         │
│ Run health    │                                                         │
├───────────────┴─────────────────────────────────────────────────────────┤
│ z ──────────●──────────────────────────── 0.42 m (plane 21 of 51)       │
└─────────────────────────────────────────────────────────────────────────┘
```

- **The rail** lists the class's pages. A page whose inputs the file lacks is disabled with
  solprop's reason (11.1).
- **One z scrubber** is shared by every page, so moving between pages keeps your place. Arrow keys
  move through planes and pages.
- **Controls sit on their page.** The band chosen for "At this plane" is the band Compress works
  on; the plane is always the scrubber's.
- **Comparison is rare**, so it is not in the first build; 12.8 keeps it possible.

The Qt docks and saved layouts of the earlier design are dropped, since a page is a fixed set of
views. A class's definition can add pages.

### 11.3 The pages and their views

**Capillary** (solid-core and antiresonant fibre follow it):

| Page | Shows | Its controls |
|---|---|---|
| **Overview** | Spectrum(z) and pulse(z) maps, spectrum and pulse at z | Spectral axis, dB floor |
| **At this plane** | Spectrum with a shaded band and the energy in it; pulse full and band-passed; phase | Band, axis, ranges |
| **Spectrogram** | Spectrogram at z | Gate width, mode |
| **Along z** | Energy, peak intensity, electron density, ionisation fraction | Pick quantities, log |
| **Modes** (multimode only) | Energy per mode vs z, modal spectra | Mode selection |
| **Compress** | Before, after and transform limit, on the current plane and band | Compressor chain |
| **Noise and coherence** (a seeded ensemble only) | g₁₂ | |
| **Run health** | Step size, rejections, boundary, quadrature error, log | |

**Radial filament:**

| Page | Shows | Its controls |
|---|---|---|
| **Overview** | Mirrored fluence r-z map, on-axis spectrum vs z, r-wavelength map at z, fluence(r) at z | Scale, floor |
| **At this plane** | r-t and r-wavelength maps, fluence profile | r range, time range |
| **Along z** | Peak intensity with electron density (Marburger marked), beam radius, energy | Pick quantities, log |
| **On axis** | Pulse and spectrum at r = 0, on-axis spectrogram | Gate |
| **Far field** | Angle-wavelength map showing conical emission | Angle range |
| **Run health** | As for capillary | |

Compress is absent because solprop refuses it on spatial runs (#479). Planes are thinned for
display on large runs. Axes refuse to magnify noise: a near-constant quantity reads as a constant,
and log axes are chosen automatically where the range calls for it.

Where each view comes from:

| View | solprop call | Offered when (from S16) |
|---|---|---|
| Spectrum vs z map (λ, f, ω; dB range) | `processing.fields.get_spectral_density` per plane | fields kept; labelled coordinate on spatial files |
| Power (or field, \|E\|²) vs z map | `get_power`, `get_field` | fields kept; labelled as above |
| Spectrum at plane (lin/log, resolution, bandpass) | `get_spectral_density` | the plane kept its field |
| Power, field, envelope at plane | `get_power`, `get_field(kind=…)` | the plane kept its field |
| Spectral and temporal phase | `get_phase` | the plane kept its field |
| Spectrogram (gate width, ranges) | `plotting.spectrograms.spectrogram` data | the plane kept its field |
| Energy, peak power, FWHM, … vs z | `stats/*` datasets; `processing.time` | the statistic was recorded |
| Step size and rejections vs z | `step_history/*` | always |
| Mode energies vs z, modal spectra | `modes=` selections; `processing.modal` for general modal | modal; general modal with its tables |
| r–t, r–λ, fluence, beam radius, far field | spatial stats and accessors | free space |
| Pulse compression (transform limit; Taylor, material, chirped mirrors, gratings, prisms; CEP) | `processing.compression.*` | not spatial or general modal (solprop refuses) |
| Coherence (g₁₂) over an ensemble | `processing.coherence.g12_of` | ensembles, scans |

A mode sum, a physical intensity and a coordinate intensity are not interchangeable, and
the labels say which is shown. On a modal file `get_power` gives $|b(t)|^2$ summed over the
selected modes, which differs from the broadband Poynting flux of the recorded temporal
statistics; where both appear on one page the help explains the difference.

White plots with plain-language titles (no "display-binned"). Interactive views are pyqtgraph
with the prototype's hover readouts, a crosshair, and linked axes (the spectrum at a plane shares its λ axis with the map). Every view has
**Copy as Python**, giving the solprop processing and plotting calls that reproduce it.

### 11.4 Export

- **Figures**: "Export figure" renders the view with matplotlib through
  `solprop.plotting` (prop_2d, spec_1d, time_1d, spectrogram, stats) in solprop's style,
  as PNG, SVG or PDF at a chosen size and DPI, so exported figures match scripted
  ones. A quick screenshot export of the pyqtgraph view is also there.
- **Data**: a line as CSV; a map as CSV, NPZ or HDF5; the current plane's field as a
  solprop `FieldPlane` (`processing.planes.field_plane`) for coupling into another run.
  Quantitative exports are computed from the file at full resolution through the view's
  specification, never from the reduced arrays on screen.
- **Run**: the run folder is self-contained; "Reveal in Finder/Explorer" and "Copy path".

## 12. Cross-cutting design

### 12.1 Parameters: one registry

Every parameter the GUI shows has one registry entry: recipe key, label, symbol, unit
and display unit, solprop keyword it maps to, default (read from solprop, never
duplicated), range, newcomer/advanced, tooltip text and help anchor. Advanced sections and the
generic form are built from the registry, and a class's sheet names its entries by recipe path,
with a test that every sheet field is a real path of a real recipe; the recipe schema, the tooltips, the "Show code" output and the
tests all come from it. A test checks that every registry entry maps to a real keyword of
the route's solprop function, catching drift when solprop changes.

Because the recipe lives in solprop (D8), the keyword, unit, default, range and doc
anchor live there too (with S9); the GUI's registry adds only presentation: label,
display unit, grouping, newcomer/advanced and the GUI's own tooltip wording.

### 12.2 Quantity input

Fields accept numbers with units and SI prefixes ("125 µm", "0.125 mm", "1.5 bar",
"7.5 fs", "275 uJ", "2e14 W/cm^2") and display in a chosen unit; the tooltip shows the
value in SI. Arithmetic is not needed. Profiles (6.3) use the same fields in their
tables. A small parser of our own is enough; `pint` is an option if it grows.

### 12.3 Validation in three layers

1. **Field**: type, unit, dimension, range — instant, in the widget, from solprop's
   metadata (S9).
2. **Combinations**: options that can't be combined — instant, from solprop's
   combination rules (S8).
3. **solprop**: `Recipe.check()` in the kernel after each pause in typing. It reports
   every metadata-level issue, then builds and translates (about 0.1 s), so value rules
   such as a pulse inside the window or energy *or* peak power come from solprop, not
   from a copy. Each issue names its parameter path (S10), so the GUI puts the message
   next to the field, warnings included. The Ready line comes from this layer, so ✓ means
   solprop accepts the setup.

### 12.4 Tooltips and help

- **Tooltips** on every control: a sentence or two, the symbol and unit, the default and
  where it comes from, the solprop keyword (for the script-minded), and "More…" opening
  the help.
- **Help button** on every page and dialog opens a short help page written
  for the GUI ("What this page sets, what to choose and why"), linking to the solprop
  guide for depth: the class to `guide/setup/model-selection` and
  `spatial-decompositions`, the medium section to `guide/media/materials` and `mixtures`,
  the physics section to `media/nonlinear`, `ionisation`, `raman`, and so on.
- **One line of help** under the focused field gives the sentence that matters while typing. If
  that proves too little for a newcomer compared with the old stepper, a guided mode can be added.
- The guide pages open from the bundled HTML of the installed solprop version when
  present, otherwise from the online documentation.
- A **GUI user guide** (in this repository's docs) walks through S1 and S2 with
  screenshots, generated by an offscreen script so they stay current.

### 12.5 Recipe files (TOML)

The format is solprop's (D8), defined by SEP 41 (S1). Its keys are the route's own
keywords, all in one `[build]` table; execution settings go in `[execution]`; pins and scans
have their own tables. The grouping the GUI shows comes from solprop's metadata, not the file. A
sketch of what the GUI writes:

```toml
[recipe]
format = 1
route = "capillary"
title = "RDW in a helium capillary"
solprop = "0.2.0"                     # version that wrote it

[build]
radius = "125 um"
length = "1 m"
mode_average = true
gas = "He"
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

[pins]
pressure = { target = "zdw", wavelength = "600 nm", search = ["0.1 bar", "10 bar"], root = "lowest" }
```

- Only what the user set is written; defaults stay implicit so a recipe follows
  solprop's improvements. Each run file holds this authored recipe and the resolved one
  (section 4). A hand-written recipe opened and left unedited saves verbatim; an edited one
  is written canonically, without comments.
- Every value has a parameter path (`pressure`, `pulses[1].energy`,
  `launch["HE1,1:even"].share`), used for edits,
  messages, scans and pins alike.
- `"none"` spells `None` only where the default is something else (`noise`,
  `boundary_check`); elsewhere the key is left out. Callables and types outside solprop's
  registry cannot be held (the Python export is the way out, Q9). Objects are tables with a
  `type`, and a recipe is one-way: there is no capturing a Python call that built objects.
- Values are quantity strings (12.2) or plain SI numbers (open question Q2); solprop
  converts them exactly and checks their dimension.
- Data a recipe refers to (a measured spectrum, an n_eff table) is copied next to it and
  referred to by a relative path, and its hash recorded.
- `format` is versioned; the loader upgrades old formats.
- **Python export** writes a plain solprop script with the route's keyword call
  (`prop_capillary(radius=125e-6, …)`), any pin's solver call before it, and a comment
  per non-default choice, so it runs
  wherever solprop is installed (`Recipe.to_python()`). `solprop run recipe.toml` (or
  `python -m solprop run`) is the headless equivalent for scripts and clusters.

### 12.6 Edit, undo, persistence

Every change is an undoable command (Qt's `QUndoStack`) on the recipe; the title bar
shows unsaved changes; autosave keeps a recovery copy. Window layout, last workspace,
units and recent files persist with `QSettings`. The first look is light (D18); a dark theme can
follow the system later.

### 12.7 Scans (designed now, built later)

Scans are a mode of their own (D16), started from a sheet, not a section of the Run stage.

- **Setup**: any numeric field, nested ones included (a second pulse's energy, a
  profile knot), gets a "scan" toggle in its context menu that turns it into a list or
  range (linear, log, explicit) over its parameter path. Pins re-solve at every point. The scan's setup shows
  the variables, the grid shape and the total cost, the retention choice (`fields`:
  none, chosen planes, all) and reductions (energy in a band, spectrum in a band, peak
  power, …, from `processing.scans`).
- **Run**: `Recipe.scan()` gives the `Scan`, a picklable builder (passing each point's seed)
  and the keywords for `run_scan(..., executor=Local() | Pool(workers))`, the retention choice
  among them. Live progress
  needs solprop change S4 (today the scan reports finished points only).
- **Results**: a scan browser shows any reduction as a map over two scan variables (or
  lines over one), with the others as sliders; clicking a point opens that point's run
  in Analyse (when its fields were retained).
- **Data model**: a scan is a recipe plus a `[scan]` table of variables, so the recipe
  format must allow it from format 1.

### 12.8 View specifications

A display product has a scientific meaning, and the GUI states it once. Each view is a
**view specification** shared by the pyqtgraph view, the matplotlib export and "Copy as
Python": the quantity and its unit, the selection (modes, planes, window), the
normalisation and its reference (a spectrum in dB is relative to a stated reference;
in compare mode several runs share one, so energy differences stay visible), any
filtering, and the reduction used for display.

- Display reductions are labelled as such. A strided power trace can miss a narrow
  peak, so strided reductions keep each bin's maximum, and the view says the trace is
  reduced.
- Hover values on a reduced view read the authoritative value from the file on demand
  where the plane is stored, and say "display value" where it is not.
- Quantitative exports and "Copy as Python" are computed from the file, never from
  display arrays.
- Reductions with a scientific claim (energy in a band, a peak, a width) are solprop's
  (`processing`), tested there to preserve what they claim.

## 13. Architecture and implementation options

### 13.1 Constraints from solprop

- `import solprop` itself is lazy (about 30 ms), but building anything is not: importing
  `solprop.recipe` loads the driver, the backends and h5py (about 0.5 s warm, several
  seconds cold), and numba compilation and FFTW planning add more on a first run. None of
  this may happen in the window's process; the light parts it may use are listed in 13.5.
- A run uses all the cores it is given (numba, FFTW, OpenMP, native threads); a run in
  the GUI process would fight the GUI for the GIL and the cores.
- Stopping today is a signal: solprop's handlers turn SIGINT into `KeyboardInterrupt`
  and SIGTERM into `RunTerminated` (a signal arriving while a plane is being committed
  waits for the commit to finish), and the run then writes its state. The handlers are
  installed only in a main thread, and on Windows `os.kill` with SIGINT terminates the
  process outright. So the run needs its own process, and the GUI stops it through
  solprop's portable stop flag (S17) set from the worker's control thread.
- Results files have no SWMR: a writer holds a lock sidecar, and another process can't
  safely read a file being written. Live data must therefore be **streamed** from the
  process that runs the simulation, through solprop's plane observer (S3), which gives
  read-only, lazy access to each committed plane. An observer that raises is detached
  by solprop and the run continues, but the worker still catches its own pipe errors.
- Resume needs a `Simulation` rebuilt in Python with the file's configuration hash (ADR-066):
  solprop rebuilds it from the resolved recipe in the file (S2), so the GUI needs no state of
  its own to resume. `resumability(file, gate=True)` shows what differs before one is tried.

### 13.2 The options

**A. Integrated.** solprop runs in a thread of the GUI process.
Simplest to write, but violates every constraint above: GIL contention freezes the UI,
a crash in native code takes the GUI down with the user's unsaved setup, and stopping
cleanly is awkward. Rejected.

**B. A worker process per run** (what the prototypes do). The GUI spawns a child per run;
the child builds the simulation from the recipe, writes the HDF5 file, and streams
reduced rows over a pipe.
- Good: robust, isolates crashes, clean stop, uses all cores, simple; works today.
- Limits: the run dies with the GUI; the GUI still needs solprop in some process for
  the explorer, validation and previews — a second, long-lived child (a *kernel*) is
  needed anyway.

**C. A local server.** A long-lived background process (*solprop service*) owns runs, the
explorer computations and validation; the GUI is a client over a local socket.
- Good: runs survive the GUI closing and re-attach with full history; several windows
  (or a notebook) can watch the same run; one warm solprop import serves every request;
  the same protocol later works over an SSH tunnel to a workstation, which is the
  remote story.
- Costs: a protocol to design and version; service lifecycle (start on demand, stop when
  idle, find a running one, recover from a crashed one); local security (a socket only
  the user can open, a token); more moving parts to debug.

### 13.3 Recommendation: B now, shaped as C

Build the GUI against a **service interface** — a small set of request and event
messages — and implement it first in-process-tree: a *service* object in the GUI process
that manages a long-lived kernel child (warm solprop for explorer, validation, previews)
and one worker child per run, over multiprocessing pipes. Moving to C later means
running the same service code as a separate process and changing the transport, not the
GUI.

```
 GUI process (Qt, no solprop)                 children (solprop)
 ┌──────────────────────────┐   requests    ┌──────────────────────────────┐
 │ pages, plots, recipe     │ ────────────▶ │ kernel: validate, preview,    │
 │ model, undo              │ ◀──────────── │ explore, solve pins, estimate │
 │        │                 │   replies     └──────────────────────────────┘
 │  service client          │               ┌──────────────────────────────┐
 │  (one interface;         │ ◀──events──── │ worker per run: build from    │
 │   transport: pipes now,  │ ───control──▶ │ recipe, run, write HDF5,      │
 │   socket later)          │               │ stream reduced rows + stats   │
 └──────────────────────────┘               └──────────────────────────────┘
```

Messages (sketch):

| Request (GUI → service) | Reply or events |
|---|---|
| `validate(recipe)` | errors and warnings with parameter paths |
| `summarise(recipe)` | a sheet's numbers, on a route with no lighter call |
| `explore(geometry, medium, quantities, axis, sweep)` | arrays |
| `solve_pin(recipe, pin)` | every root and the chosen value, or the attainable range (`Recipe.pin_roots`) |
| `preview_input(recipe)` | pulse and beam arrays on the grid |
| `estimate(recipe)` | sizes, memory, rough time |
| `start_run(recipe, folder)` | `started`, `progress`, `plane`, `stats`, `warning` (the boundary lamp's among them), `log`, then `finished` / `stopped` / `failed` (with traceback) |
| `stop(run)`, `kill(run)`, `resume(run)` | as above |
| `resumability(file)` | the resume mode, or the reason there is none |
| `subscribe(run, views, from_plane)` | `plane` events from that plane on, reduced for the requested views |
| `list_runs()` | run summaries |

Every request carries an id and the recipe revision it was made for, and every reply
and event carries them back. Kernel requests are *latest wins*: typing in a field
cancels the stale validation or summary request it superseded, and because a numerical
call already running in the kernel may not stop, the client also drops any reply whose
revision is not the current one. `start_run` captures one immutable, validated revision;
editing the setup afterwards never changes a running run.

**Bounded work and reliable endings.** Every queue between the processes is bounded.
Replaceable messages (progress, the latest preview of a view) are coalesced: an
undelivered one is replaced by its successor, so a slow client never blocks the worker
and never sees a backlog. Terminal events (`finished`, `stopped`, `failed`, errors) are
never coalesced or dropped. The worker's observer only reduces the plane and enqueues;
sending happens on another thread, and a broken pipe marks the client disconnected
without touching the run (solprop's observer policy is the second line of defence).

**Reduction in the worker.** Each live view subscribes to a reduced product (the spectral
density on the display's λ grid in dB, power cropped and reduced to the time window with
each bin's maximum kept, per-mode energies, on-axis spectrum, …), computed in the
worker's plane observer from the plane solprop hands it, to the view's specification
(12.8). A plane of the prototype's view is a few kB; a 201-plane run a few MB in total —
cheap over a pipe or a local socket. The worker keeps every reduced row it has made in
memory, so subscribing `from_plane=0` replays history from there, without reading the
file being written; this is what makes re-attaching (and option C) work. A view whose
specification changes during a run is filled from the new rows on, and from the file
once the run ends.

**Detach.** Runs that outlive the window need worker discovery, control ownership and
re-attachment, which only option C gives properly. A sidecar of reduced rows (B+)
preserves display history but solves none of those, so it is not built unless detach is
needed before C (Q4).

**Threads.** The service gives each worker a thread budget (`SOLPROP_THREADS`), shows it,
and warns when concurrent runs would oversubscribe the machine. The kernel runs with one
or two threads.

### 13.4 Remote, later

The recommended route needs no new design: the same service runs on the workstation and
the GUI connects through an SSH tunnel, or (for clusters) the service submits workers
with solprop's `Slurm` executor. Until then, remote work is "Export recipe" or "Export as
Python", run by hand, then "Open results".

### 13.5 Code structure

```
gui/                      the distribution solprop-gui (D2)
  pyproject.toml          its metadata and tool configuration; hatch_build.py pins solprop
  AGENTS.md               the rules for working on it
  src/solprop_gui/
    model/      the editable recipe model over solprop.recipe (undo, dirty state, pins)
    registry/   routes and parameters (presentation over solprop's metadata)
    classes/    the problem classes: sheets, derived numbers, pages (data; added by W1)
    service/    the service interface; kernel and worker processes; transports
    ui/         the window, its stages, sheets and rail, widgets (quantity field, profile editor)
    views/      pyqtgraph views and their matplotlib export counterparts
    help/       page help text and the map to solprop guide pages
    app.py      the solprop-gui command
  tests/
```

Rules, enforced by `gui/tests/test_layers.py`:

- `model`, `registry`, `classes`, `service` and `help` import no Qt.
- The window's process imports from solprop only `solprop.recipe.quantity`, `paths`,
  `document` and `errors` (the quantity parser and writer, parameter paths and the TOML
  document, light once `solprop.recipe` imports lazily, section 21). Metadata,
  capabilities, checks and previews come from the kernel as plain data, which also keeps
  option C open. Only `service` imports the rest of solprop, and only its child processes
  run it.
- Nothing imports a private solprop name.

Tooling follows solprop's (uv, ruff, basedpyright, pytest, numpy-style docstrings,
GPL-3.0-or-later with REUSE headers), and the quality bar is D11's. Tests: model and recipe
tests without Qt; `pytest-qt` for widgets; recipe round trips (recipe → build → solprop
metadata equals the direct keyword call's); exported scripts run for ten steps; offscreen
screenshot tests of each page. CI runs them as the `gui` job on Linux, macOS and Windows
for every change to `gui/` or to solprop itself.

### 13.6 Distribution

First `pip install "solprop[gui]"` (or `pip install solprop-gui`, or `uv tool install
solprop-gui`), giving a `solprop-gui` command; it pulls PySide6 (its Essentials
distribution) and pyqtgraph, and solprop at the same version; add `native` for the native
backend. It is released with solprop from the same tag; its publication to PyPI waits for
a first usable milestone.

The PySide6 wheels carry Qt itself, so Windows and macOS should need nothing more (Windows is
untried by hand). Linux needs a few system libraries the wheels leave out. A desktop
usually has `libegl1`, `libgl1`, `libxkbcommon0`, `libfontconfig1` and `libdbus-1-3`, but often not
`libxcb-cursor0`, without which Qt cannot open an X11 window. A bare install has not been tried. The
wheels also need glibc 2.34 on x86-64 and 2.39 on aarch64, where the window runs. `solprop-gui`
should check at start and print the exact install line in place of Qt's error about the `xcb`
plugin.

Standalone bundles (PyInstaller, Briefcase, or a conda `constructor` installer) come
later; the native backend, numba caches and FFTW make them a separate piece of work,
best done with solprop's release packaging.

## 14. Open questions

All are decided: Q1 by D8, Q2 by SEP 41 as recommended, and Q3–Q12 as recommended on
2026-11-02 (D10).

| # | Question | Decision |
|---|---|---|
| Q1 | ~~Does the recipe live in solprop or only in the GUI?~~ | Decided: solprop (D8). |
| Q2 | Recipe values: quantity strings ("125 um") or plain SI numbers? | Accept both; write quantity strings for readability. |
| Q3 | Process model: B shaped as C (13.3), or straight to C? | B shaped as C. |
| Q4 | Must runs survive closing the GUI in the first release? | Not required; detach comes with option C (13.3), not a B+ sidecar. |
| Q5 | Which routes come first? | Revised by D14 and section 16: capillary, radial filament, solid-core and antiresonant fibre, then the rest on the generic form. |
| Q6 | Help: bundled guide HTML, online, or both? | Both, bundled preferred. |
| Q7 | Name of the application and command. | `solprop-gui` / `solprop-gui`; open to better. |
| Q8 | Minimum screen size. | 1280 × 800, usable at 1024 × 700. |
| Q9 | Should the GUI ever accept Python snippets (a custom pulse shape, a custom response)? | No; "Export as Python" is the way out. |
| Q10 | Compare-runs in the first release? | Revised by D16: no. Comparison is rare, so it comes later, as overlays of lines. |
| Q11 | Expose GPU backends (mlx on Apple silicon, device builds)? | Yes, in Advanced, with the precision rule enforced. |
| Q12 | Accessibility targets (keyboard-only use, screen readers)? | Keyboard-complete from the start; screen readers best effort through Qt. |

## 15. Changes proposed to solprop

D7 allows changing solprop to suit the GUI. These are ordered by value; each would also
help script users.

All of them were proposed together in solprop's SEP 41, *Recipes, and the hooks a front
end needs*, accepted on 2026-11-01 and `Final` on 2026-11-02. This section is kept as it was
proposed; section 21 says where the delivery differs. It rests on SEP 42. The Python
interface changes are additive. The SEP refined them:

- recipe keys are the `Simulation.from_*` keywords, in one `[build]` table, with an
  `[execution]` table, and parameter paths and pins in format 1; the guide's example
  recipes ship in the package (`Recipe.example`);
- the recipe is stored as written and as resolved, without an output-schema change;
- the plane observer is lazy and cannot fail a run, and boundary warnings need no hook of
  their own;
- SEP 41 lands after SEP 42, whose comparison decides every resume;
- S14 adds missing keywords and documents deliberate differences instead of renaming;
- S15 is documentation only.

Three further changes come from the reviews:

| # | Change | Why |
|---|---|---|
| S16 | **Result capabilities**: what a results file can give, physical or coordinate, with units and refusals. | Results views offered by what the file stores (11.1). |
| S17 | **A portable stop flag and a resumability query.** | Stop on every platform; Resume offered only when solprop says it works (4, 10.2). |
| S18 | **Previews**: `Recipe.modes()` and `Recipe.preview_input()`. | The explorer and the input preview without building a run (7.4, 9.1). |

The original table follows.

| # | Change | Why | Size |
|---|---|---|---|
| S1 | **A recipe module**: `solprop.recipe` with a versioned TOML format covering every route; `Recipe.load`, `Recipe.build() -> Simulation`, `Recipe.to_python()`, and `solprop run recipe.toml`. | The GUI becomes thin; scripts, clusters and the GUI share one description; physics defaults and rules stay in solprop. Accepted in principle (D8). | Large; its own SEP. |
| S2 | **Record the recipe in the output at run start**, and **resume from a file alone** (`solprop resume run.h5`). | Results reopen their setup; stopped runs resume from the GUI or the shell without the original script. Today the call's arguments are not recorded and configuration is not reconstructible. | Medium, after S1. |
| S3 | **A plane observer**: a documented reporter extension that receives read access to the just-saved plane (today `plane()` gets z and a summary, and reading the output there works but isn't promised). | Live views stream from a supported hook. | Small. |
| S4 | **Progress for scans**: a reporter for `run_scan` (point started/finished, per-point progress). | Live scan progress; today points run with progress off. | Small. |
| S5 | **Mode launch on modal routes**: a keyword giving energy fraction and phase per mode. | Multimode inputs without building raw arrays. | Small–medium. |
| S6 | **Offset and tilt on free-space beams** (`SpatialPulse`), not only through `coupling.Alignment`. | A beam's position is an input, not a coupling step. | Small. |
| S7 | **Design solvers for any mode**: radius from ZDW; pressure from ZDW or RDW for antiresonant, rectangular and slab modes. | Pinned parameters (7.5) on every guided route. | Medium. |
| S8 | **A capability query**: per route, the allowed options and their incompatibilities; per species, the ionisation models with status and whether Raman exists. | The GUI disables options with solprop's reasons instead of copying its rules. | Medium. |
| S9 | **Machine-readable parameter metadata**: unit, range and doc anchor per keyword (e.g. `Annotated` types or a table beside each interface function). | Tooltips, units and validation without a hand-written copy. | Medium; overlaps S1. |
| S10 | **Structured validation**: `DriverError` (and friends) carry the parameter name; a `check()` that builds and translates, stopping before the backend; grid suggestions from pulses. | Errors next to the right field; Auto grid. | Small–medium. |
| S11 | **A cost estimate**: `Simulation.estimate()` → array sizes, memory, and a step-cost figure. | The pre-flight panel and the scan cost. | Small–medium. |
| S12 | **Measured-pulse loaders**: two-column spectra, FROG/SPIDER retrievals → `DataPulse`. | Inputs from experiments, in the GUI and in scripts. | Small. |
| S13 | **Geometry drawing in the library**: move the guide's cross-section figures into `solprop.plotting.geometry`, with thumbnails and an explicit palette. | Icons and a live to-scale drawing of the user's geometry. | Medium. |
| S14 | **Harmonise route defaults and keywords**: integrators and tolerances differ by route (Multistep at 1e-6 for capillaries, Runge–Kutta at 1e-7 for GNLSE, 1e-5 for waveguides, 1e-4 free space); some keywords exist on one route only (`polarisation_taper`, `interpolate_planes`, `step_planes`). | Where differences are deliberate the GUI shows them as route defaults; where accidental, fix them. | Review first. |
| S15 | **Writer-lock sidecars**: each run leaves a `.writer.lock` file beside its HDF5 file after it finishes (seen in the prototype's `runs/`). | Either remove it when uncontended or document it; the GUI hides it either way. | Small. |

## 16. Phasing

Each milestone is an issue, a branch and a pull request (D9), and each waits for the
solprop changes section 21 names for it.

The workflow redesign (D14) sets what comes next. W1 to W5 were built beside the step window of
#484, which W6 then replaced and deleted. `model`, `registry`, `service` and
`help` carry over, as do the widgets that fit (quantity field, type rows, profile editor, launch
table, scan editor, compressor chain) and the pyqtgraph views.

| Milestone | Content |
|---|---|
| W1 Frame (built locally) | The `classes` layer. The controller moved out of `MainWindow`. Home, the three stages, the sheet frame, the rail and the z scrubber. Built as: `classes/` holds the four classes as data, a test holding every row to its class's example; `ui/document.py` (`SetupDocument`) owns the recipe, undo, checks and files for both windows; `solprop-gui --workflow` opens the new window beside the step window, which Other geometry still opens. Setup edits and checks a real recipe with solprop's numbers; Run and Analyse host the existing views under the new frame, with their sliders replaced by the scrubber. Left to W2: the live sheet (preview plot, typed targets, fast numbers), plain-language rows, and the pages not built, which say so on the rail. |
| W2 Capillary (built locally) | End to end: the live sheet with typed targets, the Run stage and every page of 11.3. Checked by walking S1 and S4 by hand. Built as: `service/sheet.py` gives the numbers (about 120 ms, `tools.capillary_params` on the resolved recipe), the curve and dispersive wave (about 170 ms) and the output's size; a number with a target is a button that writes `[pins]`, and the row it moves shows what it solved. Run and Analyse are one widget (`ui/analyse.py`), the Overview under a progress strip or beside the rail, and the pages are `ui/analysis_pages.py` over a shared `ui/analysis.py`, drawing only while on screen. Not built: a rough time on the Ready line, the step-size sparkline on the strip, and a choice among several roots of a target. A ZDW target takes about 290 ms to solve, over the sheet's 150 ms (P8). |
| W3 Radial filament (built locally) | Its sheet and pages, with the spatial readers and beam numbers of 21.2. Checked by walking S6. Built as: rows that move several paths, because the object routes write a gas or wavelength several times and a run's length as its last saved plane (`Field.also`, `moves_planes`, `planes_over`, #533); switches for Kerr and plasma, whose entries come back from the class's example; a lens as an option on the beam. The live numbers come from `tools.beam_params`. The Overview grows from each plane's fluence and the beam's spectrum, and reads the on-axis spectrum from the file when the run ends. Not built: a far field or spectrogram while a run goes. |
| W4 Fibres (built locally) | Solid-core and antiresonant, as classes. Checked by running both through every page. Built as: the solid-core sheet reads its mode and `tools`, with no targets, since pins cover the capillary only; the antiresonant sheet uses `tools.soliton_params` on its own mode, and only the soliton order is a target, because a fibre has many zeros and resonances. Its zero is read from the curve inside the run's window, where the tool's nearest in frequency can lie far beyond it. Its loss curve and cross-section sit below the dispersion curve. A sheet edit costs about 280 ms on the antiresonant route against the 150 ms of the capillary. Not built: Noise and coherence, which stays disabled as no run is a seeded ensemble until scans exist. |
| W5 Design a waveguide (built locally) | The workflow of 7.4, reusing the explorer's quantities and sweeps. Checked by walking S2 by script. Built as: a stage of the workflow window, opened from Home, that never touches the open setup. A design is always a class's example with the design's values set, so what it shows is what that recipe builds, and *Simulate this* opens it as a new sheet. It covers capillary and antiresonant fibre, with pages Curves (β₂, D, β₃, n_eff − 1, group index), ZDW map (pressure against radius per zero, capillary only), Phase matching and Loss. It has a sweep of pressure, radius or gas, where a click takes a curve's value or a map point's radius and pressure. It solves the pressure for a ZDW, and the energy too for an order. A curve costs about 90 ms, a sweep of four about 250 ms, and a map about 2 s. Not built: a sweep of the mode order, the solid-core fibre, a figure export (CSV only), and Copy as Python. |
| W6 Other geometry (built locally) | Every other route on the generic form. The step window is deleted. Checked by running each example of those routes (hollow rectangular, graded-index, Cartesian and line beams, both planar) from Home through every page. Built as: Home's Other geometry tile lists those examples. A recipe on a route with no class gets a class made from solprop's metadata: its sections are the metadata groups, holding the keywords the recipe sets, with the rest under Advanced. Its numbers are the summary's energy and peak power, and its rail is Overview, At this plane, Along z and Run health. A planar run has no Overview, since its planes are slab amplitudes no map reads yet. A beam's plane is summed across the beam. A general modal guide's energy is its physical energy. `solprop-gui` opens the workflow window, and `--workflow` is accepted and ignored. A service run re-attaches in the new window. Removed with the step window: `ui/main_window`, `ui/start`, `ui/design_tool`, `ui/scan_editor`, `views/explorer`, `views/zdw_chart` and their tests. Not built: scans, which wait for their own mode (12.7); a cross-section on the generic form; per-width labels (J/m) for line and planar beams; a fluence map page for Cartesian and line beams (#455 gives the reader). |
| After W6 (built) | From the first hand review. Opening a class's sheet takes 0.06 to 0.18 s and is ready in 0.3 to 0.5 s, down from 0.7 to 1.9 s and 2 s. The cost was the window's widgets, so an object field now makes a type's form when it is picked, and Advanced rows are made when the section is opened, searched or holds an error. The kernel describes the classes' routes and makes a first check of the slow ones as the window opens. Opening another setup, example or file while a run is going asks first, and a run stopped that way no longer takes the stage when it ends. A "Runs on" line under the Ready line chooses the backend (the recipe's keyword, where the route has one; MLX writes single precision and is shown disabled with the reason where it cannot run) and the thread count (the window's, kept between sessions; the default is solprop's `thread_budget`, which `solprop.backends` exports for this, #540). Home's tiles carry a drawing of each geometry. There are two classes more: a Cartesian beam (`space.type=FreeGrid`), with the beam numbers and a page of the fluence across x and y, and a line beam (`space.type=LineGrid`), with values per metre of y and a map of the fluence across x along z. Not built: a rectangular guide class, a plasma switch on the Cartesian sheet (its example has none to copy), and P/P_cr for a line beam, which has no collapse of its own. |
| Fibres (built) | From the hand review. The solid-core class is the GNLSE fibre, given by its γ and Taylor coefficients alone; its Raman row says silica, the GNLSE's default. A step-index fibre (`fibre.type=StepIndexMode`, from `step_index_fibre`) has its mode solved: its γ from the mode and n₂, its dispersion curve and its cross-section, and rows that say an empty core is fused silica and an empty cladding air. A graded-index fibre (`provider.provider.type=GradedIndexProvider`, from `graded_index_fibre`) is read from its recipe's profile without solving it, for its numerical aperture, V number and the modes a power-law core holds (about V² p / (2 (p + 2))), and draws its index across the core; its Modes page says why it has no energy per mode (its modes are not orthogonal). Not built: per-material Raman models, which belong in solprop. |
| Plots (built) | From the hand review. A Home button starts the stage bar. Every plot zooms with the wheel, on the plot or on one axis, and pans by dragging (on a map with a z line a drag moves the plane, so an axis drag pans it); a double-click, or Reset the zoom, goes back, and a zoomed plot keeps its view while the scrubber or a live run redraws it. The readout on a line plot follows the nearest curve. A right-click opens a plot's own menu: Save this plot (PNG, or SVG drawn through Qt, since pyqtgraph's exporter fails on a legend), Copy this plot as an image, and, on a line plot, Width at half maximum here (−3 dB on a dB axis), which draws the width on the curve with its value, Copy the data as CSV and Clear the widths; on a map with a z line, Go to the peak. The export menu adds This page as an image, and offers the publication figure only where `solprop.plotting` draws one (not on spatial runs); a figure that fails says why. The run strip has the step-size sparkline W2 left out: the accepted step against z on a log scale, from the run's progress reports, with the latest and the range in its tooltip. |

Each class gets one scripted run from Home through Setup, Run and Analyse, checking that each page
loads or says why it cannot. M6, M7 and M8 follow. The rows below record what #484 built; W1 to W6
take over M1 to M5, and their step-page wording is superseded. M6's scan setup moves out of the
Propagate step into its own mode (12.7).

| Milestone | Content |
|---|---|
| M0 (done) | Prototypes: NiceGUI and Qt run-and-watch for `prop_capillary`. |
| Scaffold (done) | The `gui/` distribution in solprop's repository: packaging and the `gui` extra, the layer rules and their tests, the `gui` CI job, this page. |
| M1 Lifecycle (in progress) | On the capillary, mode-averaged, only: main window, step bar, summary panel, service with kernel and worker, and the whole run lifecycle end to end — edit, validate, save, export, run, stop, reopen, resume and a simple compare of two runs — with the Overview results layout. It needs the lazy `solprop.recipe` (21.2). This exercises the architecture more than more forms would. |
| M2 Explorer (in progress, #422) | The waveguide explorer, standalone and in the medium page; pinned pressure; multimode capillary. Built as: a Dispersion view among the views (n_eff − 1, group index, β₂ or D, β₃, loss, A_eff and γ against wavelength, frequency or photon energy; modes on and off; ZDW and pump marked; a sweep of one keyword overlaid; a table at the pump; CSV), fed by a kernel `modes` request over `Recipe.modes` (P4) and brought forward on the Geometry and Medium steps; the start panel's "Waveguide explorer" opens a four-mode helium capillary; pins on pressure, radius, energy and peak power edited in their rows ("pinned": target, search range, root, solved value, every root on request through `Recipe.pin_roots`), a pin and a value of one keyword replacing each other in one undo step; the multimode capillary card; the launch table, labelled by the mode preview; the explorer at a position along the guide (z, for tapers and gradients); the design tool (Tools menu and the explorer's "Design…"), which solves the pressure or radius and the energy (or peak power) from a ZDW or RDW target and a soliton order with solprop's two pin stages on a copy of the setup and writes them as literals in one undo step; and the ZDW chart, the pressure `tools.pressure_zdw` gives each zero-dispersion wavelength against the radius for the setup's fill, the 2-D map of 7.4 drawn as its contours; each mode's phase mismatch to the launch soliton as a quantity and its phase-matched RDW marked (from the mode preview, P5, #445); figures exported as PNG, SVG or PDF, drawn by Matplotlib in the kernel from the view's own description. |
| M3 Guided routes (built locally) | Antiresonant, full-transverse capillary, GNLSE (Taylor, step-index, tables), rectangular, slab. Built as: every keyword that names a registered type is edited in that type's own rows (the kernel's `describe` carries the metadata of every type a route can name, P2 #409), a list of them (pulses, a waveguide's modes) as a column of such editors; cards for the antiresonant fibre, the hollow waveguide of supplied modes and the GNLSE fibre, each starting from a solprop example (`hollow_rectangular`, #402, is new) and recognised by its preset and marked keywords; the GNLSE's Taylor coefficients as one row per order with its units; a slab set's energy and peak power per metre of y (#404); summary numbers with SI prefixes; the explorer on the GNLSE's one specified mode (P5), without the quantities a Taylor fibre does not have. Not built: step-index and tabulated GNLSE fibres (`IndexTable` and mode objects are not registered types), metal claddings (#403). |
| M4 Results depth (in progress, local) | All views of 11.3 under result capabilities, view specifications and export through `solprop.plotting`, richer compare. Built so far: a Plane view (a slider over the field planes, mode selection; the spectral density, power and spectral phase from `solprop.processing`, the phase drawn only where the spectrum is within 30 dB of its peak; the spectrogram drawn by `solprop.plotting.spectrogram`; the best Taylor-phase compression by `compress_taylor` over the power with the transform limit) and a Statistics view (every recorded statistic against z with the catalogue's summary and unit, a column per mode; the step size with rejected steps); figures of `prop_2d`, `spec_1d`, `time_1d`, `spectrogram` and `plot_stats` written by solprop. Still to do: modal energy views, the spatial views (M5), the other compressors, coherence, data export (CSV/NPZ/HDF5, FieldPlane), layouts. |
| M5 Free space (in progress, local) | Radial, Cartesian, line; spatial views. Built so far: the Cartesian free-space and planar-hybrid cards, starting from `cartesian_beam` and `planar_hybrid`, their `[build]` tables edited in the registered types' rows (M3); on a spatial file the Plane view draws the plane's fluence (`node_fluence` on the transverse grid rebuilt from the run's recipe, a map in x and y or a profile in r or x), labels the power and spectrum as the coordinate quantities `capabilities.output` says they are, and disables compression with solprop's reason. Still to do: the radial card (needs a radial example, #405), far field and beam-radius views beyond the recorded statistics, spatial maps along z (r-t, r-lambda), physical spatial power (#406). |
| M6 Scans (built; moved to its own stage after W6) | Scan setup, runs, scan browser. Since W6: a Scan stage beside Setup, Run and Analyse holds the `[scan]` editor, the scan's progress point by point, and its results, a recorded quantity over the first variable with a line for each value of the second (failed points marked, quantities the scan does not move left out). *Scan this…* on a sheet row opens it with that keyword as a variable. A run from the sheet leaves `[scan]` out and runs the setup once; *Run scan* runs the scan; a `scan.h5` opened from Home comes to this stage. After the first hand review: a progress bar per variable, outermost first, with the value it is at, and one for the whole scan; what is also recorded at each point, as solprop reductions the front end declares (energy and peak power within a band, and over one variable the spectrum or the pulse at a distance, held on axes fixed by the recipe so the points line up and drawn as a map); and a point of a profile given as a list (`pressure[0]` of a gradient) offered as a variable, its values read back in their unit. Not built since W6 either: a map over two variables, a third variable's slider, a cost estimate, and opening a point's run. Built first in the step window as: a Scan section on the Propagate step edits the recipe's `[scan]` table (its name; each variable a path, offered from the route's keywords or typed, with a list of values or a range of start, stop, number and spacing; the number of points), as one undo step; Run runs a recipe with a scan through `Recipe.scan` and `run_scan` in the run's worker, a `ScanReporter` streaming each point's start (its values) and end, Stop leaving the points not started; the scan browser draws any result the points recorded (`ScanOutput.as_grid`, each point's value at its last plane summed over its modes) as a line over one variable or a map over two, failed points marked. Every numeric field's context menu has *Scan this…*, which makes it a variable from its value. Not built: nested fields' scan toggle, reductions chosen in the editor, opening a point's run from the browser (fields are not retained by default), executors other than `Local`. In its place, a right click on the results offers *Open this point as a setup*: the recipe the scan's file records, without `[scan]` and with the nearest point's values (the third variable on at its first), opened as a new sheet to run or change. |
| M7 Service out of process (first part, local) | Option C, re-attach, then SSH-tunnel remote. Built as: `service/server.py`, a process of its own that owns one kernel and the run workers and speaks the same requests, replies, runs and events in `Envelope` messages over a Unix socket (a named pipe on Windows) guarded by a random token in a private state file; `service/remote.py`, a window's connection that finds the service or starts it detached; the bridge uses it with `SOLPROP_GUI_SERVICE=1` (opt-in while it is new; option B stays the default), a window that closes leaves its run going, and a new window lists the service's runs at start and re-attaches to the latest one going (its rows replayed by the worker; also Run → Re-attach). The service ends itself after ten idle minutes. Not built: several windows controlling one run (the last attached sees its events alongside the starter), the remote service over SSH with Slurm submission (deferred to #480), a service started at login. |
| M8 Distribution | Standalone installers. Deferred from the first release (#475): it needs a packaging tool chosen first. |

## 17. Lessons from the prototypes

- **Qt over NiceGUI** (D1): the Qt prototype felt like an application, needed no browser
  stack, and its plots were faster; NiceGUI's attraction (remote viewing) is better met
  by option C later. That prototype's plots were not a fair test, and a later pair of spikes
  felt the same. Qt stands for the reasons in 22.4.
- **Streaming reduced rows works.** A child process sends one row per saved plane
  (spectrum in dB, cropped power, energy) and the GUI redraws at most four times a
  second; a fast run never floods the window and a slow one updates as soon as a plane
  lands.
- **pyqtgraph needs its readouts written** (about 70 lines for value boxes on maps and
  dots on curves), after which they are fully under our control.
- **Matplotlib on demand** (`prop_2d` in a separate window) is a good complement for
  publication-style figures and needs no reimplementation.
- **solprop's failures are physics messages.** The first "failure" met was
  `BoundaryError` (light reaching the time-window edge in a long run). The GUI must
  present such errors as advice with a fix ("widen the window and rerun"), keep the
  planes reached, and offer resume only where solprop says it is possible.
- **Stopping by SIGINT** is clean on macOS and Linux: solprop finishes the plane, the
  partial output is saved, and the file stays consistent. It is not portable (Windows
  terminates the process), hence S17.

## 18. Design review, 2026-10-31

Reviewer: Codex (AI). Reviewed this design alongside solprop SEP 41, including its
existing AI review, and the relevant solprop source, manual and tests at revision
`c4cb347`. **Recommendation: keep the overall architecture, but resolve the contracts
below before treating the design as settled.** The separation of physics from
presentation, document-based setup, Python export, standalone explorer and worker per
run are sound. These findings are review comments, not adopted changes to sections
1–17. (The response, with the changes since made, is section 19.)

### 18.1 Findings

1. **G1 — major: result capabilities must depend on the stored data and its physical
   basis (sections 11.1–11.3).** The views table marks `get_spectral_density`,
   `get_power` and compression as available on all routes. Solprop's
   `docs/guide/results/processing.md` explicitly warns that the first two accept
   spatial files but do not return physical energies or powers there; compression
   refuses spatial and general modal outputs. General modal reconstruction also
   requires the matching guide and reference tables, which an arbitrary results file
   need not contain. **Recommendation:** query result capabilities separately from
   setup capabilities, accounting for the basis, retained fields, recorded statistics
   and available references. Open every supported file, but offer only views whose
   inputs and physical interpretation are available. Label the quantity and measure,
   including per-unit-width quantities. Ordinary modal `get_power` reconstructs
   `|b(t)|²`, which is distinct from the broadband Poynting flux used by recorded
   temporal statistics; explain that distinction where it matters. Do not assume a
   mode sum, physical intensity and coordinate intensity are interchangeable.

2. **G2 — major: widening the time window requires a new run, not ordinary resume
   (sections 4, 10.2 and 17).** Changing the window changes the grid and the resume
   identity. The boundary monitor's suggested action should be "Widen the window and
   rerun". Continuing from a saved field on another grid would be a separately defined
   remapping operation with provenance and accuracy requirements; it cannot recover
   light already affected by the old boundary. More generally, a failed run is not
   necessarily resumable: construction may fail before a checkpoint exists, an
   interpolated checkpoint requires an explicit policy, and a killed field-free run
   may have no restart state. **Recommendation:** determine resume availability from
   the file and compatibility checks, with a reason when it is unavailable.

3. **G3 — major: distinguish an authored recipe from an immutable run definition
   (sections 4 and 12.5).** Following new defaults and reconstructing an old run are
   different requirements. Keep the authored recipe, with omissions, units and pins,
   alongside a resolved definition that freezes effective defaults, solved values,
   seed and referenced input data. "Open setup from this run" should initially show
   what ran; adopting current defaults or re-solving constraints is a subsequent
   explicit action. A simulation changed after recipe construction must not retain
   misleading recipe provenance. This agrees with SEP 41 review R7. Freezing
   parameters does not remove the need to check material and environment compatibility.

4. **G4 — major: pins and nested scan targets need a recipe contract (sections 7.5
   and 12.7).** SEP 41's keyword-based format does not yet define the pin's target,
   solver and resolved value promised here. Define when a pin re-solves, the selected
   root and search domain, what happens when a target becomes unreachable, and how
   pins interact with scans and dependent pins. Likewise, "any numeric field" includes
   a second pulse's energy or a profile knot, beyond a scan of top-level route
   keywords. **Recommendation:** define canonical nested parameter paths and pin
   semantics before publishing recipe format 1, so these do not become GUI-only
   scientific setup state. Headless execution and Python export must preserve the
   chosen interpretation.

5. **G5 — major: the service protocol needs bounded work and revision identities
   (sections 13.3 and 17).** Redrawing four times a second does not bound the message
   queue, the cost of reducing every plane, or time blocked writing to a slow client.
   Cancelling an obsolete request does not necessarily interrupt a numerical library
   call already running in the kernel. **Recommendation:** bound queues; coalesce
   replaceable progress and preview updates; reliably deliver completion and errors;
   attach request IDs and recipe revisions to replies; and reject stale replies even
   when cancellation fails. Run captures one immutable validated revision. Specify
   how the writer services historical-plane requests during propagation. Define what
   happens when an observer or transport fails: a disconnected viewer must not
   accidentally fail a simulation through an uncaught pipe exception.

6. **G6 — major: stopping needs a portable contract and a separate execution state
   (sections 4, 10.2 and 13).** The prototype uses
   `os.kill(pid, signal.SIGINT)`. On Windows this is an unconditional termination,
   not the intended graceful interrupt; the special console-control events have
   different requirements ([Python's `os.kill` documentation](https://docs.python.org/3.14/library/os.html#os.kill)).
   **Recommendation:** define a platform-independent stop request, backed by a tested
   platform mechanism or a cooperative solprop hook respecting commit boundaries.
   Keep queued, starting, stopping, disconnected and worker-lost states separate from
   solprop's scientific outcome. `RunOutcome.status` currently has no `stopped` value.
   Force-kill requires recovery inspection before describing a file as resumable;
   `n_saved` alone does not establish that a restart state exists.

7. **G7 — major: a display product needs an explicit scientific meaning (sections
   11.3–11.4 and 13.3).** A spectrum in dB needs a reference; independent per-run
   normalization can conceal energy differences in compare mode. Straight striding
   can miss narrow temporal peaks. Reduced or clipped arrays cannot also promise
   full-resolution hover values and quantitative exports. **Recommendation:** share
   a view specification between pyqtgraph, Matplotlib export and Copy as Python:
   quantity, units, selection, normalization, reference, filtering and reduction.
   Identify display approximations and calculate quantitative exports from the
   authoritative data. Put scientific reductions in solprop, with checks that they
   preserve the features and quantities they claim to represent.

8. **G8 — implementation priority: prove the complete run lifecycle early (section
   16).** The first small capillary milestone should exercise edit, validate, save,
   export, run, stop, reopen, resume and compare. Bring basic recovery and reopening
   forward from M4, before broad route coverage: these operations exercise the
   architecture more thoroughly than additional forms.

9. **G9 — minor: make the free-space nonlinear default an explicit preset (section
   8.2).** A GUI-only Kerr-on default contradicts principle 1. If this is the intended
   starting problem, express it as a named solprop recipe preset and show that choice.

10. **G10 — implementation priority: defer B+ unless detach is needed early (section
    13.3).** A sidecar can preserve selected display history, but does not by itself
    solve worker discovery, control ownership, additional subscriptions or
    reattachment. Retain the service interface, but avoid treating a sidecar as the
    complete lifecycle design for runs that outlive the window.

### 18.2 Additional SEP 41 findings and an update

These points supplement the existing SEP review; the recipe vocabulary, explicit
`None`, resolved defaults, field-free observer behaviour and phase-merging policy
already have findings there.

- **Launch energy fractions need a basis restriction or a metric-aware definition
  (S5).** On a nonorthogonal general modal basis, physical energy includes cross terms
  and coordinate energies are not additive. Normalizing fractions to one is
  insufficient. Restrict the convenience to bases where that interpretation is valid,
  or define amplitudes and normalize through the actual flux metric.
- **Use the project's numerical comparison policy (section 7).** Ordinary resumed
  versus uninterrupted runs require an accuracy envelope, not a rounding-only bound;
  independent observer and non-observer runs do not require bitwise agreement.
  Copying an observed plane and reading that same stored plane remains an exact
  structural comparison.
- **The unseeded-noise resume part of R3 is addressed at the inspected revision.**
  `Simulation._adopt_recorded_seed` now reuses the file's seed when the caller supplied
  none. Independent recipe-equivalence tests still need controlled seeds, and resolved
  provenance remains necessary. Preserve the earlier review as historical evidence,
  with this update alongside it.

**Evidence and limits.** This was a read-only design review against the documents,
relevant source and existing tests; no simulations or test suite were run for the
review. The Windows signal behaviour was checked against Python's documentation.

## 19. Response to the design review

Each finding was checked against solprop at `main` after its 0.1.0 release preparation,
and all ten are taken. The solprop side of G1, G3, G4 and G6 is in the revised SEP 41
(its response answers the same points as C1–C5 and R7); this section says where the
design changed.

| # | Response | Where |
|---|---|---|
| G1 | Taken. Views are offered by what the file stores, through solprop's result capabilities (SEP 41 S16): fields kept per plane, physical flux or coordinate quantity with its unit, available references, refused operations with reasons. The views table now says when each view is offered, and the modal `\|b(t)\|²` versus Poynting-flux distinction is stated. | 11.1, 11.3, 15 |
| G2 | Taken. The boundary fix is "widen the window and rerun"; a changed grid is always a new run; resume is offered only when solprop's `resumability` says so, with the reason otherwise. The design's earlier "widen the time window and resume" was wrong. | 4, 10.1, 10.2, 17 |
| G3 | Taken. Authored and resolved recipes are separate; "Open setup from this run" shows what ran, and adopting current defaults is an explicit action. solprop writes no recipe for a simulation changed after it was built. | 4, 11.1, 12.5 |
| G4 | Taken, in solprop. Parameter paths and `[pins]` (target, search range, root choice, dependency order, cycles refused, re-solved per scan point) are part of recipe format 1 in SEP 41. | 7.5, 12.5, 12.7 |
| G5 | Taken. Request ids and recipe revisions on every message; stale replies dropped even when cancellation fails; a run captures one immutable revision; bounded queues with coalesced replaceable updates and reliable terminal events; history replayed from the worker's own reduced rows; transport failures isolated from the run. | 13.3 |
| G6 | Taken. Stop goes through solprop's portable stop flag (SEP 41 S17); the GUI's execution states are separate from `RunOutcome.status`; a killed run is described as resumable only after `resumability`. | 4, 10.2, 13.1 |
| G7 | Taken. A view specification shared by the live view, the export and "Copy as Python"; references for dB stated and shared in compare mode; max-keeping reductions; quantitative exports and scientific reductions computed from the file by solprop. | 10.2, 11.4, 12.8 |
| G8 | Taken. M1 is now the whole lifecycle on one capillary description; resume and a simple compare move from M4 to M1. | 16 |
| G9 | Taken. The GUI has no free-space default of its own; new setups start from solprop's example recipes (added to SEP 41 S1 as `Recipe.example`), so Kerr and plasma on in the filament example is solprop's named choice. | 8.2 |
| G10 | Taken. B+ is dropped as the detach design; detach comes with option C. | 13.3, Q4 |

The §18.2 notes on SEP 41 (launch normalisation on general bases, the comparison
classes, the R3 update) are answered in the SEP's response as C3, R19 and R3, and the
launch table of 9.2 follows them.

## 20. Changes from the second SEP 41 review

SEP 41's revision after its supplementary review (G1–G25) changed these parts of this design: the
recipe has one `[build]` table (12.5); the resolved recipe holds what rebuilds the run and not every
default (4); a two-value design is a tool and not a pin (7.5); pins resolve in two fixed stages (7.5);
the boundary lamp reads warnings, with no observer hook (10.2); `run` returns when its stop flag stops
it (10.2); and `resumability` also says whether a file can be rebuilt and, on request, whether the gate
passes (4, 13.1).

## 21. What solprop delivered, and what changed here

Checked on 2026-11-02 against SEP 41's implementation record (Part II) and its code as
merged to `main` at `Final`.

### 21.1 Differences from this design

| Here | Delivered | What the GUI does |
|---|---|---|
| A stopped run (4, 10.2) | `Simulation.run(stop=)` returns when its own flag stops it; the outcome's status is `failed`, with "StopRequested: stopped by the caller's stop flag" | The execution state *stopped* comes from the GUI's own flag and the clean return, never from the outcome |
| A live drawing with quadrature nodes (6.2) | S13 narrowed to outlines: `plotting.geometry.outline()` gives the cross-section as shapes in metres, each with its region; `draw()` and `thumbnail()` paint with Matplotlib and an explicit `Palette` (`LIGHT`, `DARK`) | The kernel sends `outline()`'s shapes as data and the window paints them; thumbnails are rendered by the kernel and cached per solprop version; no nodes are drawn |
| Explorer quantities n_eff, β₁, β₂, β₃, γ, D, loss, A_eff (7.4) | `Recipe.modes(z)` gives each mode's label, β, β₂, loss, effective area and zero-dispersion wavelength | The rest comes from solprop (21.2, M2), not from differentiating in the GUI |
| Pins on any guided route (7.5) | Recipe pins on the capillary (with or without a lattice) only: `zdw` and `rdw` set `radius` or a constant `pressure`, `soliton_order` sets a pulse's `energy` or `peak_power`; refused on the object routes; a pin on one of `pulses` is exported as its solved value | Pin switches appear only where solprop's capabilities offer them |
| TDSE row (8.1) | A TDSE response is not registered in recipes | Shown disabled, pointing to "Export as Python" |
| Data pulses from FROG or SPIDER retrievals and temporal fields (9.1) | `load_spectrum` reads three columns, frequency in Hz, power and phase (SEP 41 Q5); a `TimeDataPulse` from a file is not registered | Spectral import only, until a temporal loader is registered |
| The launch table (9.2) | `launch`, a mapping or a list of mappings, labels such as `HE1,1:even`, addressed as `launch["HE1,1"].share` | As designed, with solprop's labels |
| Auto grid, the collar (9.3) | `solprop.grid.suggest`, a stated heuristic; on free space `collar_fraction` defaults to 0 and `spatial_absorber` is a table or `false` (SEP 40) | Auto is labelled as solprop's starting suggestion; the collar is an opt-in under Advanced |
| The cost estimate in the summary panel (10.1) | `Simulation.estimate()` builds the backend and times warm-up trial steps (seconds): memory, output size, the right-hand side's cost and the calls per step, with no total time | Timing runs on request, since it builds the backend and a run that does not fit fails there as the run would (ADR-065); the Ready line shows only the output's size, which needs no backend (21.2); the rough time uses similar runs in the workspace or says it is unknown |
| One registry, its metadata from S9 (12.1) | S9 covers the keywords of the `gnlse`, `capillary` and `waveguide` routes and the execution settings; a registered type (`Factory`) carries units only | Forms for pulse lists, grids, profiles and every free-space table have no summaries, levels or defaults from solprop until 21.2's P2 |
| The geometry cards (6.1) | No example recipe on the waveguide or radial routes; metal claddings, `IndexTable`, mode or table objects as the GNLSE `fibre`, providers and `CustomPulse` are not registered; a slab set's per-width units are not in the metadata | A card with no recipe form is shown disabled, with the reason |
| `get_power` on spatial files (11.3) | Not physical there yet | The view is offered only where `capabilities.output` says so |
| Units (12.5) | `ionisation_field_range` (V/m) and `loss_db_per_m` (dB/m) take plain numbers in their stated units | The quantity field also takes a plain number in a stated unit |

### 21.2 What the GUI still needs from solprop

| | Change | Needed by |
|---|---|---|
| P1 | `solprop.recipe` imports lazily, so its quantity parser and writer, parameter paths and document load without the driver | M1; done ([#408](https://github.com/LupoLab/solprop/issues/408)) |
| P2 | Per-keyword metadata for registered types: summary, level, default, range, manual anchor and display unit, held to the signatures as S9's is | M3, M5 |
| P3 | The output's size without a backend, `Simulation.output_bytes` ([#531](https://github.com/LupoLab/solprop/pull/531)); memory stays with `estimate()`, as the constructed backend's ledger (ADR-065) | W2: the sheet's Ready line, which shows the output size live |
| P4 | `ModeSetPreview` gains n_eff, β₁, β₃ and γ (PR #421) | M2 |
| P5 | `ModePreview` gains each mode's phase mismatch to the launch soliton and its phase-matched wavelengths (#445, after #421) | M2 |
| — | A waveguide example ([#402](https://github.com/LupoLab/solprop/issues/402)); metal claddings in recipes ([#403](https://github.com/LupoLab/solprop/issues/403)); a slab set's per-width units in the metadata ([#404](https://github.com/LupoLab/solprop/issues/404)) | M3 |
| — | A radial example ([#405](https://github.com/LupoLab/solprop/issues/405)) | M5; done, `radial_kerr` |
| — | `get_power` on a spatial output ([#406](https://github.com/LupoLab/solprop/issues/406)) | M4; done |
| — | Registering `TimeDataPulse` from a file and a TDSE response | after M3 |
| P6 | A clear error for a missing keyword ([#454](https://github.com/LupoLab/solprop/issues/454)); today a bare `TypeError` from `build`, `modes` and `preview_input` | W1 |
| P7 | A faster zero search in `capillary_params` ([#528](https://github.com/LupoLab/solprop/pull/528)), and no spurious zeros beside a gas resonance ([#527](https://github.com/LupoLab/solprop/issues/527)) | W2 |
| P8 | A solve for a typed target light enough for a live sheet, without building a whole recipe (22.5) | W2 |
| P9 | Spatial readers: a fluence map on the file's transverse grid ([#455](https://github.com/LupoLab/solprop/issues/455)), and in `processing.beams` the fluence along z, `radial_spectrum` and `on_axis_spectra` | W3; beams built, open as a support PR |
| P10 | A far-field (angle-wavelength) reader: `processing.far_field` | W3; built, as above |
| P11 | An on-axis spectrogram: `processing.on_axis_spectrogram`. `get_spectrogram` also stopped aliasing on a narrow envelope grid | W3; built, as above |
| P12 | Beam numbers: `tools.beam_params` (Rayleigh range, critical power, Marburger) and `tools.beam_through_lens` | W3; built, as above |
| P13 | A filament example with plasma, `Recipe.example("filament")`, with its tutorial | W3; built, as above |
| P14 | Pin targets for free space, such as P / P_cr moving energy; recipe pins cover the capillary only today | after W3 |
| P15 | Numbers and a dispersion curve without a whole recipe on the fibre classes, as `tools` and `MarcatiliMode` allow on the capillary | W4 |

## 22. Workflow review and stack decision, 2026-11-04 and 05

The front end of draft PR #484 worked, but its workflow was poor and its interface clunky. This
section records the review of it, the comparison that settled Qt or web, and what the build must
carry over.

### 22.1 What the review found

The review read the code at 5406ea39 and screenshots of builds dated 2026-11-03. It did not run
the application, so runtime behaviour was inferred from code.

**The root cause: a problem class only picked a starting recipe.** It never selected pages, tabs or
tools, so one interface was bent to fit every class through about 30 branch points, mostly
`basis == "spatial"`, which swap labels or refuse late.

- **Too much at once.** A toolbar of 16 actions (8 of them steps), 8 view tabs, 3 console tabs,
  6 menus with 36 entries and a left panel, none of which ever hide or disable. Plots got about
  45 % of the window on Results, and the console 180 of 740 px by default. Run, Stop and Resume
  each appeared in three places, and there were two progress bars.
- **Classes forced into one frame.** Free space had 3 of 8 tabs and 1 step visible but dead, and
  could open on an error. The radial Overview cut off its third map at 1280 px, its Energy axis
  showed noise (200.00000000 to 200.00000005 µJ), and it had no fluence-vs-z map, the plot a
  filament user wants most. Radial rows were constructor arguments (`Space`, `Medium`,
  `Responses`, `KerrResponse`), not physics.
- **Broken flow.** The waveguide explorer replaced your setup with a helium capillary. Opening
  results from Start appeared to show nothing, because the view changed behind a hidden pane. A
  finished run did not lead to its results. The two main capillary cards opened blank, with at
  least 8 fields across 3 steps before a first run, and errors appeared one at a time. Left-panel
  controls did not follow the tab. Compression worked on "the plane shown on the Results step".
- **The explorer: capable plots, wrong workflow.** See 7.4.

**Worth keeping**, against the complaint:

- The capillary form is not overloaded: 20 basic rows, all in physics language, with 44 under
  Advanced.
- The explorer's nine quantities, markers, table and export are good.
- The layers without Qt are sound: the recipe model, kernel, workers, latest-wins requests and
  "solprop does the physics". These are `model`, `registry`, `service` and `help`, about 8,800
  lines or 42 % of the package. `ui` and `views`, about 11,900 lines, are mostly what the redesign
  replaces.

### 22.2 Two spikes

A clickable mock settled the flow. Then the same real application was built twice on the existing
kernel: the live capillary sheet with a two-way target, the analysis rail on real files and the Run
stage with live maps. One used NiceGUI 4.0 with a custom canvas map and uPlot, arrays sent in
binary. The other used PySide6 and pyqtgraph, with the same font and palette through QSS. The
earlier verdict in D1 had not been a fair test, since that prototype redrew whole `ui.plotly`
figures. The spikes are throwaway, and are not in the repository.

Side by side they looked and felt nearly identical, including by hand on a Linux workstation. So
the look that motivated trying the web does not separate them: QSS gets there.

### 22.3 What was measured

On screen on one Apple M5 Max laptop with a 120 Hz display. The window must be exposed and paints
counted, because macOS does not paint a covered window, and time per update misses paints.

| Interaction | Target | Qt | Web |
|---|---|---|---|
| z scrub with linked slices | ≥ 30 /s | 66 /s | 120 /s |
| dB floor drag | ≥ 30 /s | 49 /s | 120 /s |
| The same on 510 × 512 maps, scrub and floor | ≥ 30 /s | 77 and 48 /s | not measured on screen (60 /s, a frame limit, headless) |
| Sheet edit to numbers | < 150 ms | about 120 ms | about 120 ms |
| Sheet edit to dispersion curve | < 300 ms | about 250 ms | about 250 ms |
| Launch to window | | about 0.7 s | about 1.2 s |
| Launch to first numbers | < 3 s | about 2.1 s | about 2.1 s |

Both met every target. The filament's fluence maps took 0.14 s from a 39 MB file, and one plane 11
ms. A live run did not disturb the sheet in either. OpenGL and removing image smoothing made no
real difference to Qt. The sheet's numbers rely on the faster zero search of #528, which takes
`capillary_params` from about 340 ms to about 125 ms.

These times are from one machine, and Qt on a 4K display was not measured.

### 22.4 The decision: Qt (D1 reaffirmed)

Since the two feel the same, other differences decide:

| | Qt | NiceGUI |
|---|---|---|
| What a user needs | The install, plus about one system package on Linux (13.6) | A browser, or pywebview for a window, which on Linux needs its own GTK/WebKit or Qt backend |
| Languages | Python | Python and JavaScript, about the same amount of code in all |
| Plot tools | pyqtgraph, with the existing readouts | A custom plot element we would maintain |
| Size | 5 packages, about 357 MB | 50 packages, about 78 MB |
| Change rate | Stable; PySide6 needs glibc 2.34 on x86-64 where the window runs | 8 releases in 4 months, 4.0 on 2026-11-04 |
| Existing work | `ui` and `views` widgets, `pytest-qt` gates | Replaced, with browser tests |
| Remote | The existing `service/server.py` and `remote.py` over a tunnel (#480), not yet tried over SSH | Almost free through a tunnel, though a browser cannot give the server a local path |

The maintainer's reason: users should not have to go through a browser or install extra components.
Remote is a gap in convenience, not capability, and it comes later anyway (D6).

The choice would reopen if zero-install access or sharing by URL became central, or if Qt felt slow
on 4K.

### 22.5 What the build must not carry over

The spikes took short cuts, and #484 has faults any design must fix:

- The spike's kernel handler scales energy by `(target / order)²` for a soliton-order target. That
  is physics in the GUI. The build uses solprop's pin, which needs a solve light enough for a live
  sheet (21.2).
- The spike's sheet covers the capillary only, through `tools` and `MarcatiliMode`. Other classes
  need the same from solprop (21.2).
- `handler.fields("fft")[index]` and `_fluence` read every plane to use one, and fail on
  `fields="planes"` files. Use `plane_field`, which matters on large radial files.
- `electron_density` is not a default statistic on radial runs, so the filament class must ask for
  what its pages need.
- Controller logic is trapped in the 1,739-line `MainWindow`, and moves out in W1 (section 16).
