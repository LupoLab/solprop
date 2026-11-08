# SEP 46 — Calibrating the native backend to its machine

| | |
|---|---|
| **SEP** | 46 |
| **Title** | Calibrating the native backend to its machine |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-08) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Change |
| **Created** | 2026-11-08 |
| **Resolution** | — |
| **Requires** | — |
| **Supersedes** | SEP 4 (on acceptance) |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

The native backend's CPU execution settings are constants today: about seventeen
size thresholds that decide whether each operation runs serially, on four threads or on
the whole team, a cap on the team of one-column problems, the threads' waiting policy
and the effort FFTW spends choosing plans. They were measured on two or three machines,
and they do not transfer. The same change ran one machine's cases 13 % faster on average
and some of another's 1.4–1.7 times slower. This SEP proposes a **machine calibration**:
a bounded, one-off measurement that fits these settings to the machine. A quick form,
about 20 s, runs by itself on first use and says so; a standard minute or a thorough
few minutes runs on request, and the benchmarks use the thorough one. Its result is kept
in a **per-user store** in the platform's cache directory, beside the FFTW wisdom it
also gathers, and every later run reads it at import and records the settings it used.
A second, optional stage measures one particular simulation and refines the settings
for it. That covers the use case of SEP 4, which this SEP supersedes. A last phase
extends the calibration to GPU builds, whose few settings are fitted when a rented
device is bootstrapped. The calibration is the documented way to get the best speed,
named first in the installation and performance guides. Success means three things. On every machine family measured, a
calibrated default run is never slower than the built-in table beyond the A/B noise.
It is faster on the machines where the table is known to be wrong. And no result
changes beyond rounding.

### 2. Motivation and scope

**The problem.** The native backend's widths and waits are machine properties, set as
constants. The November 2026 CPU work (PR #576) measured them across six machine
families and found each setting's best value to depend on the machine, often strongly:

- **Waiting between parallel regions.** Waking sleeping threads cost 16–72 µs a region;
  a short spin cost 0.6–12 µs. A short spin ran the benchmark windows in 0.79–0.83 of
  their time on an Apple M5 Max (on its six top-tier cores), on a two-socket Intel Xeon
  Gold 6240 and on a shared AMD EPYC 7763 node; in 0.92 on an AMD EPYC 9B45 and 0.94 on a
  Google Axion (Neoverse V2). On the EPYC 9B45 under Windows a process-wide spin was
  mixed (0.96, two cases 1.07–1.12 times slower); the spin scoped to native calls ran
  0.87 at 16 threads, with the same two cases still about 8 % slower.
- **The operation thresholds.** Dividing every threshold by 16 under the spin ran the
  M5 Max's cases in 0.87 of their time. On the Raman GNLSE it was 0.44, and nothing
  was slower. The same division lost on the Xeon Gold at 4, 8, 18 and 36 threads, under
  either OpenMP runtime: the averaged hydrogen and argon capillaries took 1.4–1.7 times
  as long. On the Axion the split was sharper still: the large problems gained up to
  2× and the averaged ones lost 1.2–1.5×. The M5 Max's top-tier cores share an L2
  cache; the Xeon Gold's share only the L3.
- **The team.** On the M5 Max a team spanning its 12 second-tier cores ran most cases
  1.2–1.5 times slower than its 6 top-tier cores (#333). Dynamic scheduling did not
  recover it (1.06). On the Xeon Gold, capping one-column problems at 8 threads beat 4,
  16 and no cap.
- **FFTW plans.** Measured plans for threaded row transforms ran the M5 Max's Raman
  cases in 0.87–0.96 of their time, but added 0.3–4.6 s of planning to each run.
  #332 recorded measured plans as 12 % slower on a 32-core AVX-512 server. Patient
  plans have been seen to run slower than estimated ones, and threaded patient
  planning of one large shape to take tens of minutes. A more exhaustive effort is not
  reliably a faster plan.

So PR #576 had to choose per platform: `__APPLE__` in the source (ADR-073), a Windows
exception to the spin until it was measured (ADR-071), and a table everywhere else
that one machine family chose. Even where a choice wins on average, it loses on some
cases of the same machine. Each of #328, #329 and #81 is a request to measure one of these constants on
one more machine. This SEP stops treating them as constants.

**The use cases.**
1. *Any user on any machine:* the defaults should be right for that machine, without a
   tuning guide.
2. *A user with a costly campaign*, as in SEP 4: tune once for that simulation and
   reuse the result for every run and scan point of the same shape.
3. *A cluster:* calibrate once per node type and share the result with every user.
4. *A rented GPU:* a fresh machine, built from source on every rental, fitted in the
   same bootstrap.
5. *The project:* replace platform switches and per-machine issues with one mechanism
   whose evidence is recorded, and benchmark what users get.

**In scope.**
- Waiting policy and spin duration.
- Default team width, and which cores the team takes.
- The compact team's cap.
- The per-operation thresholds and medium widths.
- FFTW planning: the effort for each shape class, and wisdom for the shapes runs use.
- The store: where it lives, its keys, sharing, invalidation.
- Loading the store at import, and recording which profile a run used.
- A problem-specific refinement.
- The command and API, and the automatic quick calibration on first use.
- The benchmark suite's use of the calibration.
- The run-time settings of GPU builds (phase 6): the node chunk, graph replay, team and
  reduction widths, and the host threads beside the device.
- Documentation that puts the calibration first.
- Superseding SEP 4.

**Out of scope.**
- GPU kernel design: fusing kernels, fewer launches, a device-held controller (#367's
  leads). The calibration chooses among settings; it does not add kernels.
- Choices fixed when the extension is compiled: vector maths, instruction-set
  variants and the OpenMP runtime. These are recorded in the fingerprint, not chosen.
  The GPU team and reduction widths are the exception: phase 6 makes them run-time
  choices.
- Anything that changes a result beyond rounding: steppers, tolerances, grids,
  `fft_size`. The cost model of `fft_size="auto"` (#229) may later be fitted from the
  same store; that is noted in §12, not proposed.
- `numpy_ref`, which runs single-threaded NumPy, and the `mlx` and Julia backends.
- Scheduling across processes beyond what scans already do.

### 3. Context

**Today.**
- `native/include/solprop_native/core.hpp` holds the operation thresholds
  (`stage_thread_min_elements` and sixteen others), the medium width
  (`cpu_medium_threads`), the compact cap (`cpu_compact_threads`) and the
  `stable_kernel_team` rule, with three `__APPLE__` switches. `operation_threads` and
  `CpuExecutionPool::get` (`native/src/core.cpp`) apply them; `ThreadPolicy` carries
  the pool sizes and the FFT start, set once at import.
- The thread budget (ADR-052, ADR-056, ADR-070 in PR #576) bounds every pool.
- The waiting policy is chosen at import by `solprop_native/_wait.py` (ADR-071 in
  PR #576).
- FFTW's effort follows `fftw_seed_effort` (`fft_plan.hpp`): measured for serial rows
  and transverse planes, estimate for threaded rows.
- Wisdom is kept only when `SOLPROP_FFTW_WISDOM` names a file
  (`solprop_native/_wisdom.py`, keyed by the FFTW build and the processor model, with
  atomic replacement and a lock).
- GPU builds fix their response team width (`SOLPROP_NATIVE_DEVICE_TEAM_SIZE`, 256, or
  512 in the CUDA presets), their deterministic-reduction width
  (`SOLPROP_NATIVE_DEVICE_REDUCTION_SIZE`, 512) and graph replay
  (`SOLPROP_NATIVE_DEVICE_GRAPH`) when the extension is compiled. Translation sizes a
  source's node chunk from `node_budget`, whatever the device (#361). The device
  bootstrap scripts build from source on every rental.
- Every choice can be overridden by an environment variable
  ([tuning guide](../../guide/performance/tuning.md)). A run records its pools,
  budget and waiting in its metadata (`execution_sizing`, `thread_budget`,
  `openmp_wait_policy`).

**Prior proposals and records.**
- SEP 4 (Draft, #115) proposes an explicit preflight that searches CPU policies on
  short windows of one simulation, in fresh interpreters, within a budget, and writes
  an execution plan. It predates the thread budget, the compact teams, the native
  marches and PR #576. It asks the user for an importable factory and launches
  production through its own launcher.
- SEP 38 is the performance tracking suite, with its `ab` harness.
- SEP 29 defines the distributed builds.
- `native/benchmarks/size_cliffs.py` measures cost per evaluation across grid sizes.

**Issues.**
- *Answered by this SEP:* #328 (thresholds unmeasured on Windows and aarch64), #329
  (scalar wall under the size policy), #332 (FFTW effort per shape and processor),
  #81 (compact cap and scaling), #115 (SEP 4), #303 (the controls SEP 4 leaves out; this SEP
  takes the CPU ones and scans).
- *Answered in phase 6:* #361 (the node chunk sized whatever the device), and the GPU
  geometries of #303.
- *Changed:* #333 (core kinds), #314 (scan workers against threads a worker: stage 2
  can choose the split), #366 (device runs that take one of two speeds, which the
  calibration's noise test must detect), #136 (an unusable wisdom file stops
  the run, and Apple silicon models share a key, which the fingerprint's core counts
  separate) and #229 (the `fft_size` cost model).
- *Needed first:* none. The proposal builds on PR #576. If its ADR-071 and ADR-073 are
  not accepted, phase 1 measures the waiting policy as one more candidate.

### 4. Theory

**Cost of a threaded operation.** Let an operation hold work $W$ (elements, or
elements × history rows, or rows × samples, as each class counts it), with serial cost
$t_1$ per unit. On a team of $p$ threads with region cost $c(p)$ (fork, join and
barrier) and parallel efficiency $e(p, W) \le 1$,

$$T(W, p) = c(p) + \frac{W\, t_1}{p\, e(p, W)}, \qquad T(W, 1) = W\, t_1 .$$

Threading pays when $T(W,p) < T(W,1)$, that is above the crossover

$$W^*(p) = \frac{c(p)}{t_1\left(1 - \dfrac{1}{p\, e(p, W^*)}\right)} .$$

The model explains the measurements of §2 qualitatively.
- $c(p)$ moves by a factor of 10–30 between sleeping and spinning, so $W^*$ does too.
- $c(p)$ grows with $p$: 0.6 µs at 6 threads against 2.4 µs at 18 under LLVM's
  runtime, and 6–12 µs at 16–36 under GNU's.
- $e(p, W)$ is where the machine enters. For small $W$, a threaded pass moves its data
  between cores. That is cheap when the team shares a cache level holding the data,
  dear when it crosses a mesh or a socket. So $W^*$ can be large, or a medium width
  best, even when $c(p)$ is small.
- On chips with cores of two kinds, statically divided work makes $e$ bounded by the
  slower kind's share. Once the work is that fine, a team spanning both kinds runs at
  the slower kind's pace.

Neither $c(p)$ nor $e(p, W)$ is predictable from a topology description to the
accuracy needed: $e$ depends on the operation's access pattern and the cache sizes.
Both are cheap to measure. The calibration therefore measures $T(W, p)$ directly for
each operation class, over a ladder of $W$ and a few $p$, and sets the crossovers from
those measurements, not from the model. The model is kept to choose the ladder and to
check that measured crossovers are plausible.

**FFT planning.** FFTW's planner times candidate algorithms for one shape at a chosen
effort (estimate, measure, patient, exhaustive), and records the winner as wisdom,
which later planning reuses at no cost [1]. Wisdom changes only which exact algorithm
runs, so it moves results by rounding only. Three properties of the planner shape this
SEP:
- *It times a transform alone.* Each candidate runs repeatedly on warm caches with the
  machine otherwise idle. In a run the transform follows other kernels that have
  evicted its data, and a threaded plan's threads arrive from other regions. A plan
  that wins alone can lose in the step; noise during planning (a busy machine,
  sleeping threads woken at each trial) can also pick a loser. This is the likeliest
  cause of patient plans running slower than estimated ones.
- *Its cost grows steeply with effort and threads.* Patient and exhaustive planning
  try many more decompositions, and with threads also how to divide them; one large
  threaded shape can take tens of minutes.
- *Wisdom is used for any later plan of that shape whose effort is the same or less
  exhaustive.* Wisdom from a patient plan therefore replaces an estimated plan too. A
  bad plan in a shared wisdom file silently degrades every later run of that shape.

So the calibration never assumes a higher effort is better: it plans each shape class
at a short ladder of efforts within a time limit, times the resulting plans *inside*
the calibration problems' steps, at the run's thread count, and keeps the effort, and
the wisdom, only of the plan that wins there.

**Statistics.** Each comparison interleaves its candidates and repeats them. A
candidate replaces the built-in value only when its gain exceeds the spread of
repeated measurements of the built-in value, its own A/A noise. Measurements on a
machine that is visibly busy (load average above the team, or CPU time stolen by a
hypervisor) are refused or marked inconclusive. This is the method of empirical
autotuners such as ATLAS [2], and of PR #576's `ab` harness.

### 5. Proposed interface

**Commands.**

```text
solprop calibrate                    # standard: about a minute
solprop calibrate --quick            # about 20 s: waiting, team and grain at reduced depth
solprop calibrate --thorough         # several minutes: wider ladders, longer FFT planning
solprop calibrate --device           # a GPU build's settings (phase 6), on its device
solprop calibrate --show             # the profile this machine would use, and its evidence
solprop calibrate --recipe run.toml  # refine for one simulation (stage 2), bounded budget
solprop calibrate --export node.json / --import node.json
solprop calibrate --clear            # forget this machine's profile and wisdom
```

**API.** `solprop.calibration.calibrate(level="standard", budget=None) ->
CalibrationReport`, `solprop.calibration.profile() -> MachineProfile | None`, and
`Simulation.calibrate(budget=60.0)` for the stage-2 refinement. A report states each
setting chosen, the built-in value it replaces, the measured gain and noise, and why a
setting was kept.

**When it runs.** `SOLPROP_CALIBRATION` selects:
- `auto` (the default): the first native run on a machine without a matching profile
  runs the quick calibration (about 20 s) before it builds its workspace, stores it,
  and goes on. It logs at warning level what it is doing, how long it will take, where
  the result goes and how to run the standard or thorough level or turn it off, so
  the pause is never silent. Every later run, in any process, uses the stored profile.
  - *Scans* calibrate once in the parent, before the workers start; a worker never
    calibrates.
  - *Concurrent first runs*, such as an array job's tasks: one takes the store's lock
    and calibrates; the others run on the built-in table rather than wait, and log
    that they did.
  - *A busy machine:* if validation is inconclusive the profile is stored as such,
    the built-in table is used, and the quick calibration is retried after a day.
- `use`: use a stored profile if one matches; never measure implicitly.
- `off`: the built-in table, ignoring any profile, for bisection and for comparing
  code paths without calibration's own variation.

A higher level replaces a lower one: a standard or thorough calibration overwrites the
quick profile. A first run in a batch job calibrates on the cores the job was given,
which is the machine the run uses.

Explicit settings keep their precedence over any profile: `SOLPROP_THREADS`, the pool
variables, `OMP_WAIT_POLICY`/`GOMP_SPINCOUNT`/`KMP_BLOCKTIME`, `SOLPROP_FFTW_PLAN`
and `SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES`.

**The store.** One directory per user, by default the platform's standard user cache
directory; `SOLPROP_CACHE_DIR` moves it:
- `$XDG_CACHE_HOME/solprop`, or `$HOME/.cache/solprop`, on Linux;
- `$HOME/Library/Caches/solprop` on macOS;
- `%LOCALAPPDATA%\solprop\Cache` on Windows.

```text
<store>/
  machines/<fingerprint>.json      # the machine profile and its evidence
  devices/<fingerprint>.json       # a GPU build's profile on its device (phase 6)
  fftw/<fingerprint>.wisdom        # the native FFTW's wisdom, gathered by calibration and runs
  problems/<fingerprint>/<signature>.json   # stage-2 refinements
  calibration.log
```

`SOLPROP_CALIBRATION_PATH` lists read-only directories searched after the store, for
example a cluster's profiles for each node type, written once with `--export`.
`SOLPROP_FFTW_WISDOM`, where set, keeps its present meaning and takes precedence for
wisdom.

**Fingerprint.** A digest of what the profile's values depend on:
- the processor model;
- the count of cores of each kind, and the cache groups (which cores share which
  level);
- the operating system;
- the native build: the source digest it already records, the compiler, the OpenMP
  runtime, FFTW, BLAS and the vector maths.

A new native release or a different machine finds no matching profile and uses the
table; nothing stale is ever applied. The thread budget is not in the fingerprint: a
profile records per-width results, and the budget picks among them at import.

**In a run's record.** The metadata gains `calibration`: the profile's level,
fingerprint, digest and creation date, or `"built-in"`, and every setting the run took
from it, each with its built-in value beside it. A benchmark row or a result file then
says which settings produced it, and a setting added later is recorded the same way.
With `SOLPROP_CALIBRATION=off`, two runs are configured identically whatever any store
holds.

### 6. Implementation strategies

**6.1 Two stages.** Stage 1 measures the machine with synthetic problems that solprop
ships. It does not need the user's simulation, so it can run at install or on first
use. Stage 2 times short windows of a given simulation under the few candidates that
stage 1 left open, and stores the best under that simulation's *signature*:
- the route and field family;
- the grid's sample counts and node count;
- the modes and polarisations;
- the responses present, and the stepper.

Later runs with the same signature, every point of a parameter scan included, use it.
This is SEP 4's use case at a fraction of its machinery. Stage 2 needs no factory or
launcher, because it changes only settings that the store already carries. It needs
no new execution-plan format, because the profile is that format.

**6.2 What runs in which process.** The OpenMP runtime reads its waiting policy once,
and Kokkos fixes the team size at start. So each candidate team size and waiting
policy is measured in a fresh interpreter: a handful, about 2 policies × 3 widths in
the standard level. The thresholds and FFT effort apply when a workspace is built, so
they are measured within a process, by building workspaces under candidate settings.
For this, `ThreadPolicy` gains the per-operation thresholds, settable before a
workspace is built through a calibration-only entry point. The constants of
`core.hpp` become the built-in defaults of these fields.

**6.3 Probes.**
- *Region cost:* an empty region and a 1,000-element region at each candidate width
  and waiting policy (`reference_stream_copies`, extended).
- *Operation crossovers:* synthetic workspaces of the calibration problems, a GNLSE,
  a four-mode capillary with plasma, an averaged capillary, a radial grid and a
  one-dimensional free-space line, each at a ladder of grid sizes. The labelled
  operations are timed through the Kokkos Tools hooks every kernel already carries,
  serially, at the medium width and at the team. That gives $T(W, p)$ per operation
  class and so the crossovers, the medium width and the compact cap.
- *FFTW:* the canonical row and plane shapes of those problems and of
  `fft_size="auto"`'s smooth sizes up to a bound, each a *shape class* (size, kind,
  batch, and serial or threaded with its thread count). For each class:
  - **Candidates.** Estimate, then measure; patient only at the thorough level and only
    for serial plans, since the backend threads most transforms itself over serial
    plans of row blocks. Threaded plans stop at measure. Exhaustive is never tried.
  - **Time limits.** Every planner call runs under `fftw_set_timelimit`, which makes
    FFTW return its best plan so far when the limit passes: a few seconds a plan at the
    standard level, tens of seconds at thorough. The FFT stage has its own total budget
    (§6.4); shape classes left when it is spent stay at estimate and are listed in the
    report.
  - **Isolation.** Each candidate effort is planned in a fresh process, starting from
    no wisdom, and its wisdom is exported per shape class. Nothing a losing candidate
    learnt reaches the store.
  - **In-step timing.** The candidates' plans are timed inside the calibration
    problems' steps, interleaved and repeated, on the profile's team and waiting
    policy. A more exhaustive plan is kept only when it beats the less exhaustive one
    by more than the noise; otherwise the class stays at the lower effort, estimate
    included.
  - **Storage.** The store keeps one wisdom string per winning shape class, with its
    effort and evidence; a run imports those for the shapes it plans. A class whose
    winner is estimate carries no wisdom, so no earlier wisdom can override it.
- *Validation:* short windows of the calibration problems, built-in table against
  the candidate profile, interleaved. A setting whose window gain is not established
  reverts to its built-in value. A profile that does not pass validation is stored as
  `inconclusive` and not applied.

**6.4 Budget.** Quick about 20 s, standard about 60 s, both interruptible. Quick
measures the waiting, the team and the coarse grain with a short validation, and
leaves FFTW at today's efforts. Standard adds the full crossover ladder and measured
plans for the canonical shapes, with a few seconds a plan and about 20 s for the FFT
stage. Thorough adds serial patient plans, wider ladders and longer validation
windows, with tens of seconds a plan and a few minutes for the FFT stage, so about
five minutes in all. No level exceeds its budget by more than one plan's time limit;
each stage's time is reported.

**6.5 The built-in table.** Phase 0 moves today's constants, including the
`__APPLE__` switches and ADR-073's divisor, into one table keyed by operating system
and architecture, read through the same profile path. A machine with no profile then
behaves exactly as PR #576 does. The platform switches leave the source.

**6.6 Files and concurrency.** The store follows `_wisdom.py`:
- atomic replacement, a lock on POSIX, and readers that never block;
- an unreadable, corrupt or mismatched file is logged and ignored, never fatal;
- a home directory shared across machines keeps one profile per fingerprint, as the
  wisdom file does today.

**6.7 Layers and backends.** The calibration is a native-backend concern, with a thin
driver command. `numpy_ref` is unaffected. The kernel contract does not change: the
profile changes how native executes a problem, never the problem. No new dependency:
the platform cache directories are a few lines of standard library code.

**6.8 Benchmarks.** The tracking suite (SEP 38) and the `ab` harness measure a user's
default run, so they run calibrated, at the thorough level. Before a series, each
build under test is calibrated thorough on the machine, once per build fingerprint:
an A/B's two builds each get their own profile, as two releases would for a user. Each
row records the `calibration` metadata (§5), so a series that moves can be attributed
to the code or to a recalibration, and further settings can be tracked later the same
way. `SOLPROP_CALIBRATION=off` remains for experiments that must hold the settings
fixed while the code changes.

**6.9 GPU builds (phase 6).** A device run has fewer settings, and today they are fixed
when the extension is compiled or when the problem is translated:
- the node chunk, one kernel launch each, sized from `node_budget` (#361);
- graph replay of an Adams trial (`SOLPROP_NATIVE_DEVICE_GRAPH`);
- the response team width and the deterministic-reduction width;
- the host threads that run beside the device.

Phase 6 makes these run-time fields of the device workspace: Kokkos's team policies
take their width at run time, and the reduction widths become a small
set compiled in and chosen at run time. `solprop calibrate --device` times short
windows of the calibration problems on the device under each candidate, and validates
like the CPU calibration. Its fingerprint is the device's model, memory, driver and
runtime versions, and the build. The node chunk stays an explicit `node_budget` when a
user sets one, and translation still never imports the device runtime: the profile
is read when the device workspace is built.

*Where device profiles live.* A persistent GPU machine (a workstation, a cluster's GPU
nodes) uses the store as the CPU does. A rented GPU is new on every rental and is built
from source anyway, so the bootstrap that builds the extension runs `solprop calibrate
--device --thorough` as its next step, like the build itself, and the profile lives
and dies with the machine. Each run's record carries the settings, so nothing is lost
when the machine goes. Profiles exported from earlier rentals of the same model may be
imported to skip the step, but a fresh calibration is the default: a rental's driver
and runtime change from one image to the next.

### 7. Verification and validation

1. **Never worse than the table.** On each machine family measured in PR #576 (Apple
   M5 Max, two-socket Intel Xeon Gold 6240, AMD EPYC 7763 shared, AMD EPYC 9B45 under Linux
   and Windows, Neoverse V2), calibrated against built-in on the tracking set,
   `--threading backend`:
   - no case slower than its A/A range plus 2 %;
   - the geometric mean at least as good;
   - better by at least 5 % where PR #576 showed the table wrong: the M5 Max without
     ADR-073, the Axion's large problems, Windows' waiting.
2. **It finds what was measured by hand.**
   - M5 Max: six top-tier threads, a short spin, lower thresholds.
   - Xeon Gold: spin, thresholds near the table, compact cap near 8.
   - Axion: lower thresholds for the multi-node operations, not for the one-node ones.
   - Windows: the spin scoped to native calls, which ran 0.87 at 16 threads.
3. **Repeatable.** Two calibrations on an idle machine choose the same waiting and
   team, and thresholds within a factor of two, with the same validation outcome.
4. **Honest under noise.** On a deliberately loaded machine the calibration refuses
   or reports `inconclusive`, and nothing is applied.
5. **Exact.** Fields equal to rounding under every profile. Tests force extreme
   profiles (all serial, all at full width, every waiting policy) on the backend tests
   that compare native with `numpy_ref`.
6. **The store.** Concurrent writers, corrupt files, mismatched fingerprints,
   read-only fallback directories and `SOLPROP_CALIBRATION=off`, each tested. Runs
   record the profile they used.
7. **Stage 2.** On a costly case (a Cartesian beam, a radial filament), the refined
   settings are at least as fast as the machine profile on that case. A scan's points
   reuse them.
8. **Budget.** Each level ends within its stated time on the slowest machine measured.
9. **First use.** A first run with `auto` calibrates once, logs it, and takes no longer
   than the uncalibrated run plus the quick budget. A second run, a scan's workers and
   concurrent first runs do not calibrate again. Tested with a stub calibration, and on
   one machine for real.
10. **GPU builds.** On an NVIDIA and an AMD data-centre GPU, a device-calibrated run is
    no slower than the compiled defaults beyond the noise, and faster where #361's
    whole-node chunk or another width pays. The bootstrap's calibration step stays
    within its stated time. A two-speed device (#366) is reported as inconclusive,
    not fitted.
11. **FFT planning is safe.** On every machine of item 1, no shape class runs slower in
    the step under the profile than at estimate, beyond the noise. A store holding
    wisdom from an earlier, slower patient plan of a class whose winner is estimate
    does not change that class's plan. The FFT stage of the thorough level ends within
    its budget at full thread count on the largest canonical shape.

### 8. Documentation

The calibration is how users get the best speed, so the documentation leads with it.
- **Installation:** the last step of every install route (wheels, source builds,
  clusters, GPU builds) is `solprop calibrate`, with the levels, their times and what
  happens on first use if it is skipped.
- **A new manual page**, *Calibrating solprop to your machine*, first in the
  performance section: what it measures, how long each level takes, where the results
  live, the first-use message, clusters and shared profiles, GPUs and rentals, how to
  read `--show`, and how to turn it off.
- **The performance and tuning guide** opens with the calibration and keeps the
  environment variables as overrides for special cases. Where a calibration replaces
  advice (thread counts per machine, waiting, FFT effort), the guide is shortened to
  point to it; if phases 1–3 succeed, the per-platform tuning sections and their tables
  move to a reference page.
- **The quick start and the README** mention it in one line.
- **GPU guide and bootstrap scripts:** the device calibration as a bootstrap step.
- **Developer documents:** `cpu-performance.md` gives the built-in table's provenance and
  each calibration finding that changes it; `benchmarking.md` describes the thorough
  calibration of each build and the recorded settings.
- An ADR for the store, the automatic first use and the precedence of settings.
- SEP 4 marked superseded.
- The CLI reference.

### 9. Compatibility and impact

- With `SOLPROP_CALIBRATION=off`, or before a first calibration finishes, behaviour is
  PR #576's exactly.
- The first native run on a machine takes about 20 s longer, once, and says why.
- With a profile, the defaults change on that machine. No result changes beyond
  rounding, which the choice of FFT plan and thread count already moves today.
- The store writes to the user's cache directory, which solprop has not done before.
  A read-only or missing cache directory leaves runs on the built-in table, with one
  warning.
- The output metadata gains `calibration`: one schema change, taken with the release
  that ships it (one schema bump per release).
- Benchmarks measure a user's default run (`--threading backend`), calibrated at the
  thorough level (§6.8). The first series after phase 1 starts a new segment in the
  history; each row records its settings.
- No difference from Luna.jl's physics to record. Luna keeps FFTW wisdom per user in
  a cache, as this SEP does, but plans every shape with `FFTW.PATIENT`; this SEP keeps a
  patient plan only where it measures faster in the step, and never plans threaded
  shapes at patient effort.

### 10. Alternatives considered

- **Static tables per platform (today).** They do not transfer: §2.
- **A topology model** (cache groups and core kinds from the operating system or
  hwloc) setting the crossovers analytically. It does not predict $e(p, W)$ well
  enough. It is kept to choose the probes' ladders, and in the fingerprint.
- **Online tuning during a run:** time each operation at two widths during the first
  steps and keep the faster. It needs no store, but it pays in every run, is noisy on
  short ones, and cannot change the waiting policy or the team after start. It is a
  possible later complement for stage 2.
- **SEP 4 as written:** a per-simulation search with its own launcher and plan
  format. It is heavier for the user, does nothing for runs that were not tuned, and
  its plan duplicates what a profile holds. Its use case is stage 2 here.
- **Kokkos Tools' tuning interface** [3]: Kokkos can expose tunable parameters to an
  external tuner. It would put the search outside solprop, and the knowledge in a
  tool the user must install and configure. It could later feed this store.
- **Relying on the OpenMP runtime's own heuristics**, such as LLVM's turning off
  spinning on hybrid chips [4][5]. They decide only the waiting, and PR #576 measured
  them wrong on the M5 Max's top tier.

### 11. Phases

0. **Store and plumbing.**
   - The store and fingerprint; `SOLPROP_CALIBRATION` and `SOLPROP_CALIBRATION_PATH`.
   - `ThreadPolicy` carries the per-operation thresholds, medium width and compact
     cap.
   - The built-in table keyed by operating system and architecture, with the
     `__APPLE__` switches and ADR-073's divisor moved into it.
   - Wisdom kept in the store by default.
   - `solprop calibrate --show/--export/--import/--clear`.
   - Metadata `calibration`.
   - *Verifies:* identical behaviour to PR #576 without a profile; hand-written
     profiles apply and are recorded; the store's tests.
1. **Machine calibration: waiting, team, core kinds, region cost.**
   - `solprop calibrate --quick/standard`, and validation.
   - *Verifies:* §7 items 1–5 for these settings, on at least the M5 Max, an x86 server
     and an aarch64 machine.
2. **Operation grain.**
   - The crossover probes and the threshold, medium-width and compact-cap fields.
   - *Verifies:* §7 items 1–2 including the Axion and Xeon Gold splits; #328, #329 and #81
     closed.
3. **FFTW planning.**
   - The effort ladder per shape class (#332), time limits, isolated planning,
     in-step timing, per-class wisdom; serial patient plans at `--thorough`.
   - *Verifies:* run construction time with a profile no longer than without; no shape
     class slower in the step than at estimate; the FFT stage within its budget on
     the largest canonical shapes at full thread count; window gains where measured
     or patient plans pay.
4. **Stage 2 (SEP 4's use case).**
   - `Simulation.calibrate`, `--recipe`, signatures, scan reuse.
   - *Verifies:* §7 item 7. SEP 4 superseded.
5. **Defaults, benchmarks and documentation.**
   - The automatic quick calibration on first use, with its message; the scan and
     concurrency rules.
   - The tracking suite and `ab` calibrate each build thorough and record the
     settings.
   - The documentation of §8, including the simplified performance guide; the ADR.
   - *Verifies:* §7 item 9; the benchmark history's new segment recorded.
6. **GPU builds.**
   - The device settings of §6.9 become run-time fields; `solprop calibrate --device`;
     device profiles in the store; the bootstrap scripts' calibration step.
   - *Verifies:* §7 item 10; #361 closed with its measurement.

### 12. Open questions

1. **Which level runs on first use?** Decided (maintainer, 2026-11-08): the quick
   level, automatically, with a message. Open: whether a first run should later
   upgrade itself to standard in the background, or leave that to the user.
2. **Where the store lives.** Decided (maintainer, 2026-11-08): the platform's standard
   cache directory, with `SOLPROP_CACHE_DIR` to move it.
3. **Benchmark history.** Decided (maintainer, 2026-11-08): benchmarks run calibrated at
   the thorough level and record the settings used. Open: whether the calibration's own
   variation from build to build is small enough for the history's series; phase 5
   measures it with two thorough calibrations of one build.
4. **Problem signatures.** How coarse a signature can be while its settings stay
   right: should 8,192 and 8,192 + 2 samples share one? Decided in phase 4 from
   measured sensitivity.
5. **The `fft_size="auto"` cost model (#229).** It could be fitted per machine from the
   same probes. Left for a later proposal.
6. **Which device settings pay.** Which of §6.9's settings move a device run by more
   than its noise is unmeasured; phase 6 starts by measuring them on one NVIDIA and one
   AMD device, and drops those that do not.
7. **Why exhaustive plans lose.** Phase 3 records each shape class whose patient or
   measured plan lost in the step, with its alone-and-in-step timings, to tell timing
   in isolation from planning noise and from thread interplay. If serial patient plans
   rarely win, the thorough level drops them.
8. **Heterogeneous cores elsewhere.** Linux and Windows hybrid chips expose core kinds
   (`cpu_capacity`, `EfficiencyClass`); phase 1 reads them, but no such machine has
   been measured.

### 13. References

1. M. Frigo and S. G. Johnson, "The design and implementation of FFTW3," *Proc. IEEE*
   **93**, 216–231 (2005). doi:10.1109/JPROC.2004.840301. With the FFTW 3.3 manual:
   "Planner Flags", "Words of Wisdom" and `fftw_set_timelimit`,
   https://www.fftw.org/fftw3_doc/
2. R. C. Whaley, A. Petitet and J. J. Dongarra, "Automated empirical optimizations of
   software and the ATLAS project," *Parallel Comput.* **27**, 3–35 (2001).
   doi:10.1016/S0167-8191(00)00087-9
3. C. R. Trott *et al.*, "Kokkos 3: Programming model extensions for the exascale era,"
   *IEEE Trans. Parallel Distrib. Syst.* **33**, 805–817 (2022).
   doi:10.1109/TPDS.2021.3097283
4. OpenMP Architecture Review Board, *OpenMP Application Programming Interface,
   Version 5.2* (2021), `OMP_WAIT_POLICY`.
5. LLVM OpenMP runtime documentation, `KMP_BLOCKTIME` and `kmp_set_blocktime`,
   https://openmp.llvm.org/
6. scikit-learn issue 34764, "Impact of active wait policy of OMP implementations in
   scikit-learn" (2026), https://github.com/scikit-learn/scikit-learn/issues/34764

## Part II — Implementation record

## Part III — History

- 2026-11-08: Drafted (Opus 5.5) from the measurements of PR #576. Proposes to
  supersede SEP 4 on acceptance.
- 2026-11-08: Tracking issue [#579](https://github.com/LupoLab/solprop/issues/579).
- 2026-11-08: Revised after the maintainer's answers: automatic quick calibration on
  first use, the platform cache directory, thorough calibration in the benchmarks with
  the settings recorded; GPU builds added as phase 6; documentation led by the
  calibration. Windows' scoped spin measured.
- 2026-11-08: FFT planning revised after the maintainer's report of patient plans
  slower than estimated ones and threaded patient planning taking tens of minutes:
  an effort ladder with time limits, isolated planning, in-step timing and per-class
  wisdom; no threaded patient plans.
