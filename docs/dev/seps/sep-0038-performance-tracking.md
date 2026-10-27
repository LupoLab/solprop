# SEP 38 — Performance tracking: complete runs, about an hour per machine

| | |
|---|---|
| **SEP** | 38 |
| **Title** | Performance tracking: complete runs, about an hour per machine |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-10-27) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Process |
| **Created** | 2026-10-27 |
| **Resolution** | — |
| **Requires** | [SEP 37](sep-0037-noise-and-stepper-defaults.md) (the per-interface defaults the tracked cases run with) |
| **Supersedes** | [SEP 30](archive/sep-0030-standard-benchmark-suite.md) |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI) for John Travers, branch `feature/sep-0038-performance-tracking`, stacked on SEP 37's branch: phase 1 implemented and calibrated 2026-10-27; the first history rows wait for the release |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Replace SEP 30's standard benchmark runner with a small tracking command,
`python -m benchmarks track`. It times **complete propagations** of about a dozen
representative workloads at one thread and at the machine's production width, each with
the library's own default integrator and tolerances for its interface. On one machine it
takes about an hour. Each run appends a few lines per case to a small CSV kept in the
repository, one file per hardware set, so that every release can be compared with the
previous ones on the same hardware. The exploratory `ab` command stays, for comparing two
builds while a change is being made. SEP 30's standard paired mode, its numerical
reference qualification, its native coverage collector and its public export are
removed. Success means the command is run before each release on the maintainer's
machines, and a regression shows in its history.

### 2. Motivation and scope

SEP 30 built a rigorous suite: 41 cases, adaptive and fixed tracks over short entrance
windows, five to eleven fresh-process rounds, frozen references and a sanitized export.
As specified it is too heavy to be used. The core set took 2.5–3.3 h per backend on an
Apple M5 Max and about 6 h on an Intel Xeon Gold 6240; the advanced set at two widths
about 5 h. Yet most measured windows lasted 0.05–6 s: the cost was construction, fresh
processes, guards and rounds. The windows are entrance windows, before compression,
ionisation or the Raman shift develop, so they do not represent a production run's cost
either. The suite has not been run since it merged.

A release needs something smaller: one command, an hour per machine, numbers that mean
"how long does the usual simulation take now", kept where the next release can find
them.

**In scope:**

- the `track` command, its case set and its measurement protocol (§4, §5);
- a hardware identity without host names, and a per-hardware history CSV in the
  repository (§5.3);
- keeping every SEP 30 case runnable by name, outside the default selection;
- `ab` reworked onto the same worker, with a length option for quick iteration;
- removing SEP 30's standard mode, references, coverage and export, and their tests.

**Out of scope:**

- accuracy qualification. SEP 37 measured the defaults' accuracy; tracking times the
  default, it does not certify it. A numerical regression is caught by the test suite;
- GPU tracking (a later SEP, once the GPU builds are released);
- statistical adoption verdicts. The history shows changes; the
  [benchmark rules](../benchmarks.md) still say how a performance claim is judged;
- continuous integration. Shared CI runners are too noisy for timing.

### 3. Context

The [benchmark rules](../benchmarks.md) and the
[runner's manual](../benchmarking.md) describe SEP 30's suite. Its modules are under
`benchmarks/`: the registry (`registry.py`), case builders (`builders.py`,
`exercises.py`, `guides.py`, `detectors.py`, `batching.py`), the controller and worker
(`collection.py`, `worker.py`, `planning.py`), comparison and qualification
(`comparison.py`, `qualification.py`, `references.py`), native coverage
(`native_coverage.py`, `coverage_worker.py`, `coverage_reports.py`), export
(`publication.py`, `views.py`) and machine records (`system.py`, `telemetry.py`). The
`ab` mode (`ab.py`, `ab_worker.py`) was added after SEP 30 to time two builds quickly.

[SEP 37](sep-0037-noise-and-stepper-defaults.md) made input noise the default and chose
an integrator and tolerances for each `prop_*` interface. Its phase 0 timed complete
runs of the cases proposed here on an AMD EPYC 7763 node: these give the cost estimates
below. With noise on by default, a timed case needs a fixed seed to be repeatable.

### 4. Theory

There is no physics here; this section states the measurement.

**What is timed.** One complete propagation, from the launch to the end of the guide,
through the backend's `propagate_to` and the ordinary plane observations and statistics,
as `Simulation.run` performs them. Import, construction, translation and backend
construction are timed separately and reported, not tracked. A run restarts from the
translated initial field (`set_initial`), so its repeats take the same steps.

**Warm-up.** Before the timed repeats, one untimed partial run to 2 % of the length (at
least one step) builds FFT plans and fills caches. A cold first run is not tracked: its
cost is dominated by one-off work the history should not mix with propagation.

**Repeats and statistic.** Three timed repeats per configuration (`--repeats`). The
tracked value is their median; the minimum and maximum are kept beside it. On a quiet
machine the spread of complete runs of tens of seconds is a few per cent, against the
tens of per cent of the millisecond windows SEP 30 measured.

**One process per configuration.** Each case at each width runs in its own worker,
whose thread pools are set in the environment before import. Peak resident memory is
then the configuration's own.

**Aggregate.** For one hardware set and width, the change between two runs is the
geometric mean, over the cases both ran with the same case revision and resolved
integrator, of the ratio of their medians. A changed default integrator is a different
configuration, and is shown as such rather than averaged in.

### 5. Proposed interface

#### 5.1 Commands

```text
python -m benchmarks list    [--all] [--case NAME] [--stepper S] [--threads W]
python -m benchmarks track   [--case NAME] [--all] [--stepper S] [--threads W]
                             [--repeats 3] [--production N] [--instance-type TYPE]
                             [--append-history] [--archive DIR] [--out DIR]
python -m benchmarks ab      --baseline-python PY [--candidate-python PY]
                             [--case NAME] [--stepper S] [--threads W]
                             [--length FRACTION] [--rounds 3] [--repeats 3]
python -m benchmarks history [SLUG] [--width W]
```

- `list` prints the plan and its estimated time; it imports no scientific library.
- `track` with no selection runs the tracking set (§5.2) at `1,production`.
  `--case` selects any registered case, tracked or not.
- `--stepper` is `default` (the interface's own, SEP 37), `multistep`,
  `multistep-interaction` or `rk`, or a comma-separated list of them to A/B steppers in
  one run. An override replaces the integrator only; the tolerances stay the
  interface's. A case on a fixed schedule admits only `default`.
- `--production` gives the production width; otherwise it is the number of physical
  cores in the process's CPU allocation.
- `--append-history` adds the run to `benchmarks/history/<slug>.csv`. It refuses a
  dirty source tree, an incomplete run and a non-native backend.
- `history` prints, per width, one line per recorded run: date, version, commit, the
  number of cases, and the geometric mean relative to the first recorded run.

#### 5.2 The tracking set

The cases represent the main uses of solprop at the sizes they are used. Lengths are
complete guides. Costs are the medians of a complete `track` run (Part II, phase 1) on
16 cores of an AMD EPYC 7763 node, at 1 and 16 threads, with SEP 37's defaults; "—"
marks a width the case does not run at.

| Case | Interface | Problem | Widths | Cost (s) |
|---|---|---|---|---|
| `gnlse_raman_shock` | `prop_gnlse` | Dudley–Genty–Coen supercontinuum, Raman and shock, 0.15 m [1] | 1, P | 84 / 47 |
| `gnlse_soliton` | `prop_gnlse` | ideal N = 3 soliton over one soliton period | 1, P | 13 / 8 |
| `rdw` | `prop_capillary` | He resonant dispersive wave, 4 resolved modes, 1.5 m, 0.8→0 bar gradient | 1, P | 19 / 13 |
| `he_rdw_averaged` | `prop_capillary` | the same, mode-averaged | 1, P | 3 / 3 |
| `argon_blueshift` (rev. 2) | `prop_capillary` | Ar ionisation blueshift, 13 µm, 1.7 bar, 0.2 m, 1 resolved mode | 1, P | 66 / 36 |
| `argon_blueshift_averaged` (rev. 2) | `prop_capillary` | the same, mode-averaged | 1, P | 15 / 15 |
| `argon_blueshift_envelope` (new) | `prop_capillary` | `argon_blueshift` on the envelope grid | 1, P | 3 / 3 |
| `h2_raman_averaged` (new) | `prop_capillary` | H2 soliton self-frequency shift, 10 bar, 0.4 m, mode-averaged | 1, P | 16 / 16 |
| `ctc` (rev. 2) | `prop_capillary` | H2 comb-to-continuum, 3 modes, 40 bar, 0.13 m, 2 ps window | P | — / 291 |
| `radial_filament_256` | `prop_radial` | Ar filament, 256 radial nodes, 3 cm | 1, P | 16 / 6 |
| `grin_nonlinear` (rev. 2) | `prop_waveguide` | four-channel graded-index fibre, 2 cm | P | — / 38 |
| `kerr_beam_384` | `prop_free` | fused-silica Kerr beam, 384² transverse grid, fixed RK4IP | P | — / 134 |
| `kerr_beam_192` | `prop_free` | the same at 192², fixed RK4IP | 1 | 37 / — |

The sum is about 270 s at one thread and 620 s at 16 threads per pass. With three
repeats, the warm-ups and construction, the run took about 50 minutes on that node; the
target is under an hour on any machine in use.

The three most expensive cases run at the production width only: `ctc` (about 600 s at
one thread in SEP 37's phase 0), the 2 cm GRIN window (about 100 s) and the 384² Kerr
beam. The 192² Kerr beam keeps a one-thread free-space row.

Every case keeps its interface's default noise (`"input"` after SEP 37) with a fixed
seed. Every other registered case stays runnable by name; none is in the default
selection.

#### 5.3 Hardware identity and history

The **hardware slug** is the CPU model reduced to lower-case words, followed by the number
of physical cores in the allocation, for example `apple-m5-max-18c` or
`amd-epyc-7763-16c`. It names no host. A cloud instance type, from `--instance-type` or
the DMI product name on Linux, is recorded beside it but is not part of the slug.

Each history row is one (run, case, stepper, width): date, solprop version, commit,
native build identity (version, profile, compiler), slug, CPU model, instance type, OS,
width, case, case revision, requested stepper, resolved integrator and tolerances, noise
model, median, minimum and maximum seconds, repeats, accepted and rejected steps, source
evaluations, peak resident memory, and whether the machine was busy when a worker
started. That is a few kilobytes per release per machine.

The full bundle of a run (plan, worker requests and logs, results, environment) stays in
the output directory, which git ignores. `--archive DIR` copies it to a directory of the
maintainer's choosing.

### 6. Implementation strategies

**Trim and reuse, or rewrite.** The registry, builders, telemetry and `ab` are sound and
are kept. The controller of the standard mode (`collection.py`, 780 lines, with
monitoring, resume and admission) has nothing `track` needs: `track` launches one worker
per configuration and reads its receipt, as `ab` already does. Rewriting the worker once
for both `track` and `ab` is smaller than adapting the standard worker, which is entwined
with references and reset witnesses.

**Library defaults.** The builders stop passing tolerances and integrators. A case is
the interface call a user would write, plus a seed and the backend. Where the interface
has no default controller (free space) or the case builds a `Simulation` directly, the
case declares its own, as now.

**Case revisions.** A case whose definition changes keeps its name and increments its
revision; history rows carry the revision and are compared only within one. The
capillary cases revised here take the definitions SEP 37 phase 0 measured.

**Removed.** `collection.py`, `comparison.py`, `qualification.py`, `references.py`,
`planning.py`, `schedules.py`, `native_coverage.py`, `coverage_worker.py`,
`coverage_reports.py`, `publication.py`, `views.py`, `ab_worker.py`, and the `batch`
exercise (`batching.py`), whose lockstep route is covered by
`tests/backends/test_batch_lockstep.py`. Their tests go with them.

### 7. Verification and validation

- Unit tests, run in the default lane: the registry (every case resolves, the tracking
  set, revisions), the plan and its estimate without scientific imports, the hardware
  slug (formats, no host name), the history writer and summary, the stepper override,
  and `ab`'s pairing.
- An end-to-end test runs `track` and `ab` on the `smoke` case with one repeat.
- Calibration: one complete `track` run on an AMD EPYC 7763 node, recorded in the phase
  report with each case's time, and the lengths adjusted if the total exceeds an hour.
- First history: the Apple M5 Max and the EPYC node, appended at the release commit.

### 8. Documentation

Rewrite [Running the benchmark suite](../benchmarking.md) as the manual for `track`, `ab`
and `history`. In [Benchmarks](../benchmarks.md), replace "Measurement windows" with
the tracking protocol, keep the rules and the indicative results, and point to the
history files. Update `benchmarks/README.md`, the `poe bench` task, `AGENTS.md` and the
performance guide's link.

### 9. Compatibility and impact

No change to solprop itself. The SEP 30 commands `run`, `compare`, `qualify`, `profile`,
`diagnose`, `coverage` and `export` are removed; nothing outside the benchmark package
uses them. SEP 30's archived baselines are not comparable with tracking rows (windows
against complete runs) and are not migrated. SEP 30 becomes `Superseded` when this SEP
is final.

### 10. Alternatives considered

- **Keep SEP 30 with lighter defaults.** Fewer rounds and samples would still time
  entrance windows, which say little about production cost, and keep about 5000 lines
  that nothing uses.
- **Time `Simulation.run` end to end.** Simple, but each repeat would reconstruct the
  backend and its plans, and that construction would vary with the build in ways that
  are not propagation performance.
- **A single width.** One thread isolates per-core regressions; the production width
  shows threading regressions. Both are cheap except on the three cases kept at the
  production width only.
- **Track in CI.** Shared runners vary by tens of per cent between jobs.

### 11. Phases

1. **Implementation:** the registry, builders on library defaults, the worker, `track`,
   `history`, `ab` on the new worker, the removals, tests and documentation.
2. **Calibration and first data:** a complete run on the EPYC node, lengths adjusted,
   then the first history rows on the maintainer's machines.

### 12. Open questions

- Q1. If SEP 37's capillary default becomes `RungeKutta()` at 1e-5, the capillary costs
  fall; should the freed time go to a longer `ctc`, or should `ctc` run at one thread
  too?
- Q3. The 384² Kerr beam costs 134 s per run at 16 threads on the EPYC node (25 GiB),
  a fifth of the run's time for one free-space case. Keep it, or track the 192² beam at
  both widths instead?
- Q2. Should a cloud instance (for example AWS `c8a` or GCP `c4d`) be a standing
  history file, re-rented per release, or measured only when a hardware question arises?

### 13. References

1. J. M. Dudley, G. Genty and S. Coen, "Supercontinuum generation in photonic crystal
   fiber," *Rev. Mod. Phys.* **78**, 1135 (2006). doi:10.1103/RevModPhys.78.1135

## Part II — Implementation record

### Phase 1 report — the tracking command and its calibration

- **Scope:** phases 1 and 2's calibration, 2026-10-27, Opus 5.5 (AI) for John Travers,
  branch `feature/sep-0038-performance-tracking`, stacked on SEP 37's pull request
  (its defaults are provisional until the maintainer approves them).
- **Delivered:** `benchmarks/registry.py` (cases as complete guides, the `tracking`
  set, widths, observation planes and costs), `builders.py` (interface calls on the
  library's defaults with a fixed seed), `worker.py` (one configuration per process,
  timed through `run_problem` after a warm-up), `runner.py`, `track.py`,
  `history.py`, `ab.py` (with `--length`) and `__main__.py` (`list`, `track`, `ab`,
  `history`); tests `tests/test_benchmark_{track,cases,ab,exercises,guides,telemetry}.py`;
  the manual `docs/dev/benchmarking.md` and the rules in `docs/dev/benchmarks.md`.
  Removed as §6 proposed, with their tests.
- **Deviations:** the nonlinear GRIN case couples its launch at construction, which
  translates it, so its save planes and any stepper override go to its constructor
  (`benchmarks/guides.py`); a slow test covers it. The production width counts the
  distinct cores of the allocation from the Linux topology, so an allocation of one
  hardware thread per core counts every core.
- **Calibration:** one complete default run on 16 cores (one NUMA domain) of an AMD EPYC
  7763 cluster node, native backend (LOCAL profile, GCC 16.2, FFTW 3.3.11, OpenBLAS
  0.3.34), three repeats: 22 configurations, about 50 minutes including construction.
  The three repeats of every configuration took the same steps and agreed within 0.7 %,
  except `gnlse_raman_shock` at 16 threads (2.2 %), `ctc` (1.4 %) and
  `radial_filament_256` at 16 threads (26 %, a 6 s run on a node shared with other
  jobs). The costs in §5.2 are this run's medians. The run bundle is kept with the
  project's development records.
- **Not run:** no history row was recorded: the defaults are provisional and the
  branch is not merged, so the first rows wait for the release commit. The Apple M5
  Max was not timed (it was busy).
- **Open questions:** Q1–Q3 (§12).

## Part III — History

- 2026-10-27: Drafted, together with its implementation on the same branch, as the
  maintainer asked; it supersedes SEP 30 at Final.
