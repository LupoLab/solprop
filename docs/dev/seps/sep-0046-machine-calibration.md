# SEP 46 — Calibrating the native CPU backend to its machine

| | |
|---|---|
| **SEP** | 46 |
| **Title** | Calibrating the native CPU backend to its machine |
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
and some of another's 1.4–1.7 times slower. This SEP proposes a **machine calibration**: a bounded, one-off measurement,
about a minute by default, that fits these settings to the machine. Its result is kept
in a **per-user store** beside the FFTW wisdom it also gathers, and every later run
reads it at import. A second, optional stage measures one particular simulation and
refines the settings for it. That covers the use case of SEP 4, which this SEP
supersedes. Success means three things. On every machine family measured, a
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
  Google Axion (Neoverse V2). On the EPYC 9B45 under Windows a process-wide spin was mixed: two cases
  ran 1.07–1.12 times slower.
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
  #332 recorded measured plans as 12 % slower on a 32-core AVX-512 server.

So PR #576 had to choose per platform: `__APPLE__` in the source (ADR-073), a Windows
exception to the spin (ADR-071), and a table everywhere else that one machine family
chose. Each of #328, #329 and #81 is a request to measure one of these constants on
one more machine. This SEP stops treating them as constants.

**The use cases.**
1. *Any user on any machine:* the defaults should be right for that machine, without a
   tuning guide.
2. *A user with a costly campaign*, as in SEP 4: tune once for that simulation and
   reuse the result for every run and scan point of the same shape.
3. *A cluster:* calibrate once per node type and share the result with every user.
4. *The project:* replace platform switches and per-machine issues with one mechanism
   whose evidence is recorded.

**In scope.**
- Waiting policy and spin duration.
- Default team width, and which cores the team takes.
- The compact team's cap.
- The per-operation thresholds and medium widths.
- FFTW planning: the effort for each shape class, and wisdom for the shapes runs use.
- The store: where it lives, its keys, sharing, invalidation.
- Loading the store at import, and recording which profile a run used.
- A problem-specific refinement.
- The command and API.
- Superseding SEP 4.

**Out of scope.**
- GPU builds, whose block sizes and launch policy are another problem.
- Choices fixed when the extension is compiled: vector maths, instruction-set
  variants and the OpenMP runtime. These are recorded in the fingerprint, not chosen.
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
- *Changed:* #333 (core kinds), #136 (an unusable wisdom file stops
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
effort, and records the winner as wisdom, which later planning reuses at no cost [1].
Wisdom changes only which exact algorithm runs, so it moves results by rounding only.
The effort that pays depends on the shape, the thread count and how long the run is.
The calibration measures the canonical shapes once, for the effort the profile names,
so that ordinary runs pay no planning time.

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
solprop calibrate --thorough         # several minutes: wider ladders, FFTW patient plans
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
- `use` (proposed default): use a stored profile if one matches; never measure
  implicitly. On an interactive run without a profile, log once that
  `solprop calibrate` exists.
- `auto`: on the first import without a matching profile, run `--quick` once (about
  20 s) and store it. Never in a scan worker or a batch job: the import detects no
  terminal, or the scan's environment, and falls back to `use`.
- `off`: the built-in table, ignoring any profile (for bisection and for benchmark
  series that must stay comparable).

Explicit settings keep their precedence over any profile: `SOLPROP_THREADS`, the pool
variables, `OMP_WAIT_POLICY`/`GOMP_SPINCOUNT`/`KMP_BLOCKTIME`, `SOLPROP_FFTW_PLAN`
and `SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES`.

**The store.** One directory per user. `SOLPROP_CACHE_DIR` names it; by default it is
the platform's user cache directory:
- `$XDG_CACHE_HOME/solprop`, or `$HOME/.cache/solprop`, on Linux;
- `$HOME/Library/Caches/solprop` on macOS;
- `%LOCALAPPDATA%\solprop\Cache` on Windows.

```text
<store>/
  machines/<fingerprint>.json      # the machine profile and its evidence
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

**In a run's record.** The metadata gains `calibration`: the profile's fingerprint,
digest and creation date, or `"built-in"`. With `SOLPROP_CALIBRATION=off`, two runs
are configured identically whatever any store holds.

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
  `fft_size="auto"`'s smooth sizes up to a bound, planned at the effort chosen, into
  the store's wisdom. Threaded-row effort is chosen by timing both.
- *Validation:* short windows of the calibration problems, built-in table against
  the candidate profile, interleaved. A setting whose window gain is not established
  reverts to its built-in value. A profile that does not pass validation is stored as
  `inconclusive` and not applied.

**6.4 Budget.** Standard about 60 s and quick about 20 s, interruptible. The report
says what each level measured. Thorough adds the FFTW patient effort and wider
ladders: minutes, which a user who once ran Luna with patient plans for every run will
accept. Each stage's time is bounded and reported.

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
   - Windows: whatever the scoped spin's measurement shows.
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

### 8. Documentation

- A new manual page, *Calibrating solprop to your machine*, under performance: what it
  measures, how long it takes, where it keeps its results, cluster use and how to
  turn it off.
- `tuning.md`: the manual sweep it asks of users becomes `solprop calibrate`; the
  environment variables stay as overrides.
- `cpu-performance.md`: the built-in table's provenance; each calibration finding
  that changes the table.
- An ADR for the store and the precedence of settings.
- SEP 4 marked superseded.
- The CLI reference.

### 9. Compatibility and impact

- Without a profile, behaviour is PR #576's exactly.
- With one, the defaults change on that machine. No result changes beyond rounding,
  which the choice of FFT plan and thread count already moves today.
- The output metadata gains `calibration`: one schema change, taken with the release
  that ships it (one schema bump per release).
- Benchmarks measure a user's default run (`--threading backend`). Whether that
  default includes a stored profile is an open question (§12). Either way the row
  records which profile applied, so series stay comparable.
- No difference from Luna.jl to record: Luna keeps FFTW wisdom per user in a cache
  and plans with `FFTW.PATIENT`, which this SEP generalises.

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
   - Effort per shape class (#332), canonical shapes' wisdom, `--thorough` with
     patient plans.
   - *Verifies:* run construction time with a profile no longer than without; window
     gains where measured plans pay.
4. **Stage 2 (SEP 4's use case).**
   - `Simulation.calibrate`, `--recipe`, signatures, scan reuse.
   - *Verifies:* §7 item 7. SEP 4 superseded.
5. **Defaults and records.**
   - The decision on `use` against `auto`, documentation, ADR, the benchmark
     history's treatment.
   - Windows measured.

### 12. Open questions

1. **`use` or `auto` by default?** Recommended: `use` until phase 2's evidence is in,
   then `auto` for interactive sessions only. A calibration in a batch job, or
   concurrently on many nodes, measures a busy machine.
2. **Where the store lives.** The platform cache directory (proposed) or a fixed
   `$HOME/.solprop`. A cache directory is the convention, and is where cluster
   administrators expect regenerable data. `$HOME/.solprop` is easier to find. The
   proposal accepts either through `SOLPROP_CACHE_DIR`.
3. **Benchmark history.** Should tracked rows run with the machine's profile (a
   user's default once `auto` is on) or with `SOLPROP_CALIBRATION=off`, so that series
   compare code, not calibrations? Recommended: record both for one release, then
   decide.
4. **Problem signatures.** How coarse a signature can be while its settings stay
   right: should 8,192 and 8,192 + 2 samples share one? Decided in phase 4 from
   measured sensitivity.
5. **The `fft_size="auto"` cost model (#229).** It could be fitted per machine from the
   same probes. Left for a later proposal.
6. **Windows.** PR #576 left its waiting passive. A scoped spin's measurement there is
   pending; phase 1 decides it per machine anyway.
7. **Heterogeneous cores elsewhere.** Linux and Windows hybrid chips expose core kinds
   (`cpu_capacity`, `EfficiencyClass`); phase 1 reads them, but no such machine has
   been measured.

### 13. References

1. M. Frigo and S. G. Johnson, "The design and implementation of FFTW3," *Proc. IEEE*
   **93**, 216–231 (2005). doi:10.1109/JPROC.2004.840301
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
