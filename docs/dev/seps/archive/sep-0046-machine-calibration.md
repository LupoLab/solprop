# SEP 46 — Calibrating the native backend to its machine

| | |
|---|---|
| **SEP** | 46 |
| **Title** | Calibrating the native backend to its machine |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-08) |
| **Status** | Final |
| **Review status** | Reviewed |
| **Reviewers** | GPT-6 (AI); John Travers (human) |
| **Type** | Change |
| **Created** | 2026-11-08 |
| **Resolution** | Accepted 2026-11-08 by John Travers, after the AI review, the drafting agent's response and the maintainer's decisions on §12 items 1–7 ([LupoLab/solprop#580](https://github.com/LupoLab/solprop/pull/580)). |
| **Requires** | — |
| **Supersedes** | [SEP 4](sep-0004-execution-autotuning.md) |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI) for John Travers, branch `feature/sep-0046-machine-calibration` ([#582](https://github.com/LupoLab/solprop/pull/582)): started 2026-11-08; phases 0–6 complete and verified (2026-11-09; on a Google Axion, AMD EPYC 7763 and 9B45 under Linux and Windows, Intel Xeon Gold 6240 and Platinum 8581C, an Apple M5 Max and an NVIDIA H200) |
| **Manual pages** | [Calibrating solprop to your machine](../../../guide/performance/calibration.md) |

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
also gathers, and every later run applies it and records the settings it used.
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
  as long, while the large problems gained 3–21 %. On the Axion it lost too (1.06 over
  8 and 16 threads; argon 1.63 at 16, the averaged cases 1.16–1.44), while `line` and the
  Raman hydrogen case gained 6–10 %. The M5 Max's top-tier cores share an L2 cache; the
  Xeon Gold's share only the L3. The best threshold differs between machines and,
  within a machine, between operations: no single divisor captures it.
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
  ([tuning guide](../../../guide/performance/tuning.md)). A run records its pools,
  budget and waiting in its metadata (`execution_sizing`, `thread_budget`,
  `openmp_wait_policy`).

**Prior proposals and records.**
- SEP 4 (Draft, #115) proposes an explicit preflight that searches CPU policies on
  short windows of one simulation, in fresh interpreters, within a budget, and writes
  an execution plan. It predates the thread budget, the compact teams, the native
  marches and PR #576. It asks the user for an importable factory and launches
  production through its own launcher. This SEP replaces it on acceptance:
  - *replaced:* its per-simulation search (by stage 2, §6.1, §6.6), its execution plan
    (by the profile), its selection rules (by §6.5), and its pilot's accuracy contract
    (by §6.6's, which asks for physically realistic accuracy, not replay or
    rounding-level agreement);
  - *not carried over:* the factory and launcher, which stage 2 does not need;
  - *left on their issues:* worker counts (#314), GEMM providers, CPU node chunks and
    propagator policy (#303).
- SEP 38 is the performance tracking suite, with its `ab` harness.
- SEP 29 defines the distributed builds.
- `native/benchmarks/size_cliffs.py` measures cost per evaluation across grid sizes.

**Issues.**
- *Answered by this SEP:* #328 (thresholds unmeasured on Windows and aarch64), #329
  (scalar wall under the size policy), #332 (FFTW effort per shape and processor),
  #81 (compact cap and scaling), #115 (SEP 4).
- *Answered in phase 6, if the measurement pays:* #361 (the node chunk sized whatever
  the device).
- *Partly answered:* #303 (the controls SEP 4 leaves out). This SEP takes the GPU team
  and reduction widths and, under #361's condition, the device chunk. GEMM providers,
  CPU node chunks and propagator policy stay on #303.
- *Not answered:* #314 (scan workers against threads a worker) needs a concurrent scan's
  throughput, which a single process's measurement cannot give; it stays on its issue.
  The calibration measures the workers' allocation the executor chooses, not the split.
- *Changed:* #333 (core kinds), #366 (device runs that take one of two speeds, which the
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
candidate replaces the built-in value only when it wins in every round by more than
the spread of repeated measurements of the built-in value, its own A/A noise; a busy
machine gives an inconclusive result (§6.5). This is the method of empirical autotuners
such as ATLAS [2], and of PR #576's `ab` harness.

### 5. Proposed interface

**Commands.**

```text
solprop calibrate                    # standard: about a minute
solprop calibrate --quick            # about 20 s: waiting, team and grain at reduced depth
solprop calibrate --thorough         # about five minutes: wider ladders, longer FFT planning
solprop calibrate --device           # a GPU build's settings (phase 6), on its device
solprop calibrate --show             # the profile this allocation would use, and its evidence
solprop calibrate --recipe run.toml  # refine for one simulation (stage 2), bounded budget
solprop calibrate --export node.json / --import node.json
solprop calibrate --clear            # forget this machine's profiles and wisdom
```

**API.** `solprop.calibration.calibrate(level="standard", budget=None) ->
CalibrationReport`, `solprop.calibration.profile() -> MachineProfile | None`, and
`Simulation.calibrate(budget=60.0)` for the stage-2 refinement. A report states each
setting chosen, the built-in value it replaces, the measured gain and noise, and why a
setting was kept.

**Two kinds of setting.** What the calibration chooses falls in two kinds, and every
rule below follows from the difference:
- *Process settings* are fixed when the native runtime starts and cannot change in that
  process: the waiting policy (GNU's runtime reads it once; LLVM's process-wide block
  time), the pool width and the cores it is placed on. They are read from the profile
  at native import, before the extension is loaded.
- *Workspace settings* are copied into a workspace when it is built and never read
  again from anywhere global: the per-operation thresholds, the medium width, the
  compact cap, the stable-team rule, LLVM's block time inside native calls, and each
  FFT shape class's effort and wisdom. Two live workspaces may hold different ones.

**When it runs.** `SOLPROP_CALIBRATION` selects:
- `auto` (the default). The first time a native workspace is built in an allocation
  with no matching profile (§6.4), the process prints a notice (§6.10), runs the quick
  calibration in child processes while it waits, about 20 s, and builds its workspace
  with the new workspace settings that hold under the process settings it started with
  (§6.4); the rest stay built-in for this run. The new process settings take effect
  from the next process. Importing the native backend never
  calibrates, so discovery, the GUI and `--version` never pause.
  - *Scans*, local or remote, wait for a proper calibration: the parent runs the
    standard level, about a minute, for the workers' allocation (§6.4) before it starts
    them, unless a standard or thorough row already matches. A scan usually runs for a
    long time, often after waiting for a batch allocation, so a minute up front is
    cheap, and since its workers are new processes they take the process settings too.
    A worker never calibrates.
  - *Concurrent first runs on one host*, such as an array job's tasks, wait for the one
    that holds the calibration claim (§6.8), up to its deadline, then use its profile.
    Running them meanwhile would load the machine it is measuring.
  - *A busy machine:* if validation is inconclusive nothing is published, the run uses
    the built-in table, and `auto` retries after a day.
- `use`: use a matching profile if there is one; never measure implicitly.
- `off`: the built-in table and no automatic wisdom, whatever the store holds; for
  bisection and for holding settings fixed while code changes. `SOLPROP_FFTW_WISDOM`
  set by the user still applies, since the user asked for it.

A higher level replaces a lower one for the same allocation. Explicit settings keep
their precedence over any profile: `SOLPROP_THREADS`, the pool variables,
`OMP_WAIT_POLICY`/`GOMP_SPINCOUNT`/`KMP_BLOCKTIME`, `SOLPROP_FFTW_PLAN` and
`SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES`. A setting measured under conditions an
override changes is not used with it: thresholds fitted under a spinning team, for
example, give way to the built-in table when the user makes the threads sleep (§6.4).

**The store.** One directory per user, by default the platform's standard user cache
directory; `SOLPROP_CACHE_DIR` moves it:
- `$XDG_CACHE_HOME/solprop`, or `$HOME/.cache/solprop`, on Linux;
- `$HOME/Library/Caches/solprop` on macOS;
- `%LOCALAPPDATA%\solprop\Cache` on Windows.

```text
<store>/
  machines/<fingerprint>/
    current                        # names the published generation
    <generation>.json              # profile rows, their evidence and their FFT wisdom
    claim                          # held while one process calibrates (§6.8)
  devices/<fingerprint>/...        # a GPU build's profiles, the same layout (phase 6)
  problems/<fingerprint>/<descriptor>.json   # stage-2 refinements
  calibration.log
```

`SOLPROP_CALIBRATION_PATH` lists read-only directories searched after the store, for
example a cluster's profiles for each node type, written once with `--export`.

**Fingerprint.** A digest of what makes measurements portable between machines:
- the processor model, the count of cores of each kind, and the cache and memory
  topology (which cores share each cache level, NUMA nodes, sockets, SMT siblings);
- the operating system;
- the native build: its source digest, compiler, OpenMP runtime, FFTW, BLAS and vector
  maths, read from a small build record the build writes beside the extension, so
  that nothing loads the extension (and its OpenMP runtime) to compute it;
- the calibration's own version, raised whenever its probes, problems or rules change.

A new release, a changed calibration or a different machine finds no matching profile
and uses the table; nothing stale is ever applied. Within a fingerprint, profile rows
are keyed by *allocation* (§6.4).

**In a run's record.** The metadata gains `calibration`: the profile's level,
fingerprint, generation and creation date and the allocation row used, or
`"built-in"`, and the settings the run *applied*, read back from the runtime and the
workspace, each beside its built-in value. A benchmark row or a result file then says
which settings produced it, and a setting added later is recorded the same way.

### 6. Implementation strategies

**6.1 Two stages.** Stage 1 measures the machine with synthetic problems that solprop
ships, so it never needs to move a user's simulation between processes. Stage 2
refines one simulation, and varies only workspace settings: it builds fresh workspaces
from the simulation's translated problem in the calling process, under each candidate,
and never touches the simulation's own state, output or callbacks. Neither stage needs
SEP 4's factory or launcher. Stage 2's result is stored under the simulation's
*execution descriptor* (§6.6), and reused by later runs and scan points with the same
descriptor.

**6.2 What runs in which process.** Every measurement runs in child processes started
by a controller: one per candidate process setting (about two waiting policies × three
widths at the standard level), each pinned to the allocation being calibrated, with
workspace settings varied inside it. The controller never loads the native extension or
a device runtime. `ThreadPolicy` keeps only process settings; the workspace settings
move into a per-workspace execution record that `operation_threads`,
`CpuExecutionPool::get` and the FFT setup read, filled from the profile, from stage 2 or
from the built-in table when the workspace is built. The constants of `core.hpp` become
that table.

**6.3 Probes.**
- *Region cost:* an empty region and a 1,000-element region at each candidate width
  and waiting policy (`reference_stream_copies`, extended).
- *Operation crossovers:* synthetic workspaces of the calibration problems (a GNLSE, a
  four-mode capillary with plasma, an averaged capillary, a radial grid and a
  one-dimensional free-space line), each at a ladder of at most six grid sizes. The
  labelled operations are timed through the Kokkos Tools hooks every kernel already
  carries, serially, at the medium width and at the team. The crossover is the smallest
  ladder size above which the threaded time is below the serial one at every larger
  size; a ladder with no such point keeps the built-in value. These fits only propose
  candidates; §6.5's validation decides.
- *FFTW:* the canonical row and plane shapes of those problems and of
  `fft_size="auto"`'s smooth sizes up to a bound, each a *shape class* (§6.7). Outside
  the FFT stage, children plan at estimate, so that no other stage spends its time
  planning.
- *Synthetic problems are sized to the allocation:* their memory is bounded by a fifth
  of the memory available to it (the cgroup or job limit where there is one), and a
  ladder stops below that.

**6.4 Allocations, and which rows apply.** The same machine behaves differently in
different allocations: eight cores sharing a cache are not eight cores across two
sockets. A profile therefore holds rows keyed by an *allocation class*:
- the number of cores, and of each kind;
- how many of each cache group, NUMA node and socket they span;
- whether SMT siblings are in use;
- a CPU quota below the core count, where there is one (cgroups, job limits).

A calibration measures the class of the allocation it runs in. At the standard and
thorough levels it also measures the nested classes it can pin children to inside it,
such as one cache group, one NUMA node and one socket, so that one calibration of a
whole node covers most smaller jobs on it. Each row records the waiting policy its
workspace settings were fitted under.

At run time the process computes its own class from its affinity mask and quota:
- an exact match uses that row;
- otherwise the row with the same span structure and the nearest smaller core count is
  used for workspace settings, and the process settings come from the built-in rules;
- otherwise the built-in table, and `auto` calibrates this class on the next first run;
- a row is not used for settings an override invalidates: thresholds fitted under a
  spin give way to the table when the user forces sleeping threads. The same holds for
  a process still running on process settings other than the row's, such as the first
  run that triggered the calibration: only workspace settings validated under its
  actual waiting policy and width apply, and the others stay built-in until the next
  process.

Scan workers have their own partitions, chosen by the executor
(`driver/executors/pool.py`). Before it starts them, the scan's parent computes the
workers' class; unless a standard or thorough row matches, it calibrates that class
at the standard level in a child pinned like one worker, and only then starts them. The other partitions stay idle during that measurement, so shared caches and
memory bandwidth are less loaded than in the scan itself; the row records this. Remote
executors calibrate on their own hosts, in their own stores, at the standard level and
under the same claim rule, before their workers start. GPU hosts are calibrated at
bootstrap (§6.14).
Nothing calibrates in a controller that must not initialise a device.

**6.5 Selection and validation.** One statistical rule throughout, the `ab` harness's:
- *Pairs.* Candidates are timed interleaved with the built-in value, in rounds; a round
  times each once in a fresh order. Quick takes 3 rounds, standard 5, thorough 7. The
  score of a candidate is the median of its per-round ratios to the built-in.
- *Noise.* Each session also times the built-in against itself (A/A). Its noise is the
  largest per-round deviation of that ratio from one, and at least 2 %.
- *Admission.* A candidate is admitted only if its largest per-round ratio is below
  one minus the noise: it wins in every round by more than the noise. Otherwise the
  built-in value stays. Ties keep the built-in. A round that failed or was skipped is
  dropped; fewer than the minimum rounds keeps the built-in.
- *Few candidates.* Each setting is chosen from a ladder of at most four candidates, so
  that the search cannot select noise from a large set.
- *Combined validation.* The admitted settings are then validated together, on fresh
  rounds, without the profiling hooks, through the full code path (realised widths,
  the stable-team rule and all): the calibration problems' windows under the
  candidate profile against the built-in table. The combination is published only if
  no window is slower than one plus the noise and the geometric mean is below one. If
  it fails, the groups are dropped in a fixed order (grain, then FFT, then waiting and
  team) and the remainder revalidated, within the deadline; if nothing passes, nothing
  is published.
- *Load.* Before and during measurement the controller reads the host-wide load and,
  where the platform reports it, time stolen by a hypervisor. If the runnable work not
  belonging to the calibration exceeds the cores outside its allocation, or the A/A
  noise exceeds 10 %, the result is inconclusive. Missing telemetry is treated as
  unknown, and only the A/A noise decides.

**6.6 Stage 2 and the execution descriptor.** A refinement is reused only where the
execution is the same. Its key is the *execution descriptor*: the native problem's
dimensions (time samples, nodes, modes, polarisations, Cartesian or radial shape and
batch), every FFT plan key it creates (§6.7), its responses and their buffers (Raman
history length, plasma columns), its stepper family and propagator, and its node chunk.
Equal descriptors reuse; anything else falls back to the machine profile. The operation
mix can still change along a run, for example as ionisation grows, so a refinement is
validated on windows spread along the run, not only its start. A refinement can only
move workspace settings; it adds to the machine profile and never weakens it.

*The windows* are the simulation's own: built from its translated problem with its
original longitudinal operators and breakpoints, with the same noise seed for every
candidate, output and callbacks off, and nothing shortened to fit (a gradient or taper
is sampled where it lies, by starting windows at chosen positions). Each window runs a
fixed number of accepted steps (20 by default) after a warm-up of five. Candidates
differ only in execution, so they do the same work to within the step controller's
reaction to rounding. A candidate is timed per step taken; if its step count differs
from the built-in's by more than 5 %, the window is repeated and then dropped.

*Accuracy.* The project needs speed within physically realistic accuracy, not
reproducibility to the bit. Every candidate's window must end finite, with the same
step-acceptance structure as the built-in's, and with fields that agree with the
built-in's to within ten times the run's relative tolerance (the step controller's own
error scale). Speed alone never admits a candidate. Nothing here asks for replay or for
rounding-level agreement.

**6.7 FFT planning.** *Choosing the effort*, for each shape class:
- **Candidates.** Estimate, then measure; patient only at the thorough level and only
  for serial plans, since the backend threads most transforms itself over serial plans
  of row blocks. Threaded plans stop at measure. Exhaustive is never tried.
- **Time limits.** Every planner call runs under `fftw_set_timelimit`, which makes FFTW
  return its best plan so far when the limit passes: a few seconds a plan at the
  standard level, tens of seconds at thorough. The limit is approximate; the
  controller's deadline (§6.9) is what bounds the stage. Shape classes left when the
  stage's share is spent stay at estimate and are listed in the report.
- **Isolation.** Each candidate effort is planned in a fresh process, starting from no
  wisdom, and its wisdom is exported per shape class. Nothing a losing candidate learnt
  reaches the store.
- **In-step timing.** The candidates' plans are timed inside the calibration problems'
  steps, interleaved and repeated, on the profile's team and waiting policy, and
  admitted by §6.5's rule: a more exhaustive plan is kept only when it beats the less
  exhaustive one in every round by more than the noise; otherwise the class stays at
  the lower effort, estimate included.

*Applying the winners.* FFTW's wisdom is global to the process and cumulative, and a
plan at any effort uses applicable wisdom (§4). So the store's wisdom is applied in a
*planning transaction* when a workspace is built, under the existing serialisation of
FFTW's planner:
1. Forget all wisdom.
2. Plan every shape class whose winning effort is estimate, with no wisdom present, so
   each gets FFTW's plain estimate plan. A plan, once made, is not changed by wisdom
   imported later.
3. Import exactly the winning wisdom of the remaining classes, from the profile's
   generation, and plan them with `FFTW_WISDOM_ONLY` at their winning effort.
4. A class whose wisdom is missing or no longer reconstructs a plan (a changed stride
   or alignment) falls back to estimate, and the run records it.
5. Forget wisdom again, so the next workspace starts from the same state.

The class key is the exact plan FFTW is asked for: rank, dimensions and strides,
batch dimensions, direction or kind, in- or out-of-place, alignment, and thread count.
Runs no longer merge wisdom back into the store when a profile applies; the store's
wisdom changes only through a calibration. `SOLPROP_FFTW_WISDOM`, when the user sets
it, keeps today's behaviour instead and takes precedence. The calibration itself plans
each candidate effort in its own process starting from no wisdom (§4).

**6.8 Ownership, publication and concurrency.**
- *Claim.* A calibration of one allocation class takes a nonblocking claim: a `claim`
  file created exclusively (`O_CREAT | O_EXCL`, which also holds on Windows), holding
  the host, process id, start time and deadline. Another process that finds it either
  waits (same host, §5) or proceeds without calibrating. A claim whose deadline has
  passed by a minute, or whose process is gone on the same host, is stale and removed.
- *Publication.* A calibration writes one complete generation file, profile rows,
  evidence and wisdom together, then atomically replaces `current` to name it. Readers
  read `current` once, then that generation; they never see a partial profile. A
  failed, interrupted or inconclusive upgrade leaves the previous generation current.
- *Lookup.* The store's current generation, then each read-only directory's, skipping
  inconclusive entries; then the built-in table.
- *`--clear`* removes `current` and the generations, not a live claim.
- *Several nodes sharing a home directory* share one fingerprint's store, and the claim
  makes one of them calibrate per allocation class; the others proceed on the table.

**6.9 Budget and deadline.** Quick takes about 20 s, standard about 60 s, thorough
about five minutes. Quick measures the waiting, the team and the coarse grain with a
short validation, and leaves FFT efforts as today. Standard adds the full crossover
ladder and measured plans for the canonical shapes, about 20 s of FFT stage. Thorough
adds serial patient plans, wider ladders and longer validation, a few minutes of FFT
stage. Each level has one monotonic deadline for all of its work: starting
children, building workspaces, planning, timing, validation and publication. The
controller runs every child under it and terminates and reaps a child that overruns.
FFTW's own time limit only shortens each plan; the controller's deadline bounds the
whole. A share of the budget is reserved for combined validation and publication; a
stage that cannot finish its minimum rounds before its share ends is skipped and
reported. On interruption or deadline nothing partial is published. Cleanup after the
deadline is reported honestly if it overruns.

**6.10 The first-use notice.** solprop's library code never installs a log handler, and
a logger warning is invisible without one. The notice is therefore issued as a Python
warning of its own category, `CalibrationNotice`, which Python shows once on standard
error by default, in terminals and notebooks alike, and which a user silences with the
standard warnings filters or `SOLPROP_CALIBRATION=use`. It is also logged, for those
who configure logging, and reported through a run's progress reporter when one is
active. It is issued before the calibration starts and says what is happening, for how
long, where the result goes and how to choose another level or turn it off.

**6.11 The built-in table.** Phase 0 moves today's constants, including the `__APPLE__`
switches and ADR-073's divisor, into one table keyed by operating system and
architecture, read through the same path as a profile. A machine with no profile then
behaves exactly as PR #576 does, and the platform switches leave the source.

**6.12 Layers and backends.** The calibration is a native-backend concern, with a thin
driver command. `numpy_ref` is unaffected. The kernel contract does not change: the
profile changes how native executes a problem, never the problem. No new dependency:
the platform cache directories are a few lines of standard library code.

**6.13 Benchmarks.** The tracking suite (SEP 38) and the `ab` harness measure a user's
default run, so they run calibrated, at the thorough level. Before a series, each build
under test is calibrated thorough on the machine, once per build fingerprint: an A/B's
two builds each get their own profile, as two releases would for a user. Each row
records the `calibration` metadata (§5), so a series that moves can be attributed to the
code or to a recalibration. `SOLPROP_CALIBRATION=off` remains for experiments that must
hold the settings fixed while the code changes.

**6.14 GPU builds (phase 6).** A device run has fewer settings, and today they are fixed
when the extension is compiled or when the problem is translated:
- the node chunk, one kernel launch each, sized from `node_budget` (#361);
- graph replay of an Adams trial (`SOLPROP_NATIVE_DEVICE_GRAPH`);
- the response team width and the deterministic-reduction width;
- the host threads that run beside the device.

Phase 6 starts by measuring which of these move a device run by more than its noise,
on one NVIDIA and one AMD data-centre GPU, and keeps only those. For each kept setting:
- *Team and reduction widths* become run-time choices among the widths compiled in,
  limited to what the device reports it supports for each kernel.
- *The node chunk* is decided only if #361's re-measurement shows at least 10 %. If it
  does, translation stays device-free and keeps producing the chunk's *rule* for each
  route (granularity, such as whole rows on planar routes; bytes per column including
  response buffers; its own explicit `node_budget`), and the device workspace chooses
  the chunk within that rule. It admits the whole workspace, FFT and graph scratch and
  the concurrent workers' share against the free device memory before enlarging a
  chunk, and the memory ledger records the chosen chunk. The general modal route's
  separate chunk is included only once qualified, and stays as today until then. An
  explicit `node_budget` always wins.
- Every route that takes a calibrated device setting is qualified on CUDA and HIP;
  others keep the compiled defaults.

`solprop calibrate --device` validates like the CPU calibration (§6.5). Its fingerprint
is the device's model, memory, driver and runtime versions, and the build.

*Where device profiles live.* A persistent GPU machine (a workstation, a cluster's GPU
nodes) uses the store as the CPU does. A rented GPU is new on every rental and is built
from source anyway, so the bootstrap that builds the extension runs `solprop calibrate
--device --thorough` as its next step, like the build itself, and the profile lives and
dies with the machine. Each run's record carries the settings, so nothing is lost when
the machine goes.

### 7. Verification and validation

1. **Never worse than the table.** On each machine family measured in PR #576 (Apple
   M5 Max, two-socket Intel Xeon Gold 6240, AMD EPYC 7763 shared, AMD EPYC 9B45 under
   Linux and Windows, Neoverse V2), calibrated against built-in on the tracking set,
   `--threading backend`:
   - no case slower than its A/A range plus 2 %;
   - the geometric mean at least as good;
   - better by at least 5 % where PR #576 showed the table wrong: the M5 Max without
     ADR-073, and Windows' waiting; better on the large problems of the Xeon Gold and
     the Axion without the averaged cases' losses.

   These cases are the claim's scope; the calibration problems used inside the
   calibration are separate from them.
2. **It finds what was measured by hand.**
   - M5 Max: six top-tier threads, a short spin, lower thresholds.
   - Xeon Gold: spin, thresholds near the table, compact cap near 8.
   - Axion: spin, thresholds near the table for the averaged and one-column
     operations, lower only where the large problems gained.
   - Windows: the spin scoped to native calls, which ran 0.87 at 16 threads.
3. **Repeatable enough.** Two calibrations on an idle machine choose the same waiting
   and team, and thresholds within a factor of two, with the same validation outcome.
4. **Honest under noise.** On a deliberately loaded machine the calibration reports
   `inconclusive`, and nothing is published.
5. **Selection logic.** The rules of §6.5 are tested with deterministic fake timings:
   ties, drift, outliers, missing rounds, a candidate winning on average but not in
   every round, and a combined validation that fails and drops groups in order.
6. **Accurate.** Fields within the run's accuracy under every profile. Tests force
   extreme profiles (all serial, all at full width, every waiting policy, every FFT
   effort) on the backend tests that compare native with `numpy_ref`, at those tests'
   tolerances.
7. **Process and workspace settings.** A direct native import, a notebook that has
   already run, and two live workspaces with different refinements each apply the
   settings the record says they applied, read back from the runtime.
8. **Allocations.** Reuse across equal core counts with different placement (one socket
   against two), a small job's profile against a later large job, a scan's worker
   class, and an override that invalidates a row, each takes the row §6.4 says.
9. **Wisdom.** A store holding an old measured plan of a class whose winner is
   estimate gives the plain estimate plan; two workspaces built in either order get
   their own classes' plans; under `off` a cold and a populated store give the same
   plans.
10. **Store and ownership.** Concurrent writers, a claim whose owner died, simultaneous
    readers and a writer, two hosts sharing a store, corrupt and mismatched files,
    read-only directories past an inconclusive local entry, `--clear`, and Windows as
    well as POSIX.
11. **Deadline.** Slow start-up, a stalled plan, cancellation, too little memory and a
    budget too small for one comparison: each ends within the deadline plus reported
    cleanup, and publishes nothing partial.
12. **Stage 2.** On a costly case (a Cartesian beam, a radial filament), the refined
    settings are at least as fast as the machine profile on that case. A scan's points
    with equal descriptors reuse them; descriptors that differ in batch, Cartesian
    shape or response structure do not; a varying guide, a noisy launch and a
    previously run simulation are refined without changing the simulation.
13. **Budget.** Each level ends within its stated time on the slowest machine measured.
14. **First use.** With no logging configured, the notice appears once on standard error
    before the calibration starts, and once, not twice, when logging is configured. A
    first run takes no longer than the uncalibrated run plus the quick budget; a second
    run, a scan's workers and concurrent first runs do not calibrate again.
15. **GPU builds.** On an NVIDIA and an AMD data-centre GPU, a device-calibrated run is
    no slower than the compiled defaults beyond the noise, and faster where a kept
    setting pays. The bootstrap's calibration step stays within its stated time. A
    two-speed device (#366) is reported as inconclusive, not fitted. A scan still
    starts its workers before any device runtime starts.
16. **FFT planning is safe.** On every machine of item 1, no shape class runs slower in
    the step under the profile than at estimate, beyond the noise. The FFT stage of the
    thorough level ends within its budget at full thread count on the largest
    canonical shape.

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
  thorough level (§6.13). The first series after phase 1 starts a new segment in the
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
   - The store, fingerprint, build record beside the extension, claims and
     generations; `SOLPROP_CALIBRATION` and `SOLPROP_CALIBRATION_PATH`.
   - The split into process and workspace settings: a per-workspace execution record
     read by `operation_threads`, `CpuExecutionPool::get` and the FFT setup.
   - The built-in table keyed by operating system and architecture, with the
     `__APPLE__` switches and ADR-073's divisor moved into it.
   - Allocation classes from the affinity mask, topology and quota.
   - The planning transaction for wisdom.
   - `solprop calibrate --show/--export/--import/--clear`; metadata `calibration` with
     the applied settings.
   - *Verifies:* identical behaviour to PR #576 without a profile; hand-written
     profiles apply and are recorded; §7 items 7–10.
1. **Machine calibration: waiting, team, core kinds, region cost.**
   - The controller, its deadline and children; `--quick` and standard; §6.5's
     selection and combined validation.
   - *Verifies:* §7 items 1–5 and 11 for these settings, on at least the M5 Max, an x86
     server and an aarch64 machine.
2. **Operation grain.**
   - The crossover probes and the threshold, medium-width and compact-cap fields.
   - *Verifies:* §7 items 1–2 including the Axion and Xeon Gold splits; #328, #329 and
     #81 closed.
3. **FFTW planning.**
   - The effort ladder per shape class (#332), time limits, isolated planning, in-step
     timing; serial patient plans at `--thorough`.
   - *Verifies:* §7 items 9 and 16; run construction time with a profile no longer than
     without; window gains where measured or patient plans pay.
4. **Stage 2.**
   - `Simulation.calibrate`, `--recipe`, execution descriptors, scan reuse.
   - *Verifies:* §7 item 12.
5. **Defaults, benchmarks and documentation.**
   - The automatic quick calibration on first use, its notice, and the scan and
     concurrency rules.
   - The tracking suite and `ab` calibrate each build thorough and record the
     settings.
   - The documentation of §8, including the simplified performance guide; the ADR.
   - *Verifies:* §7 items 13–14; the benchmark history's new segment recorded.
6. **GPU builds.**
   - Measure which device settings pay; make those run-time choices; `solprop calibrate
     --device`; device profiles; the bootstrap scripts' calibration step.
   - *Verifies:* §7 item 15; #361 closed with its measurement.

SEP 4 is superseded when this SEP is accepted, since it was never implemented; its
index entry and tracking issue change then.

### 12. Open questions

1. **Which level runs on first use?** Decided (maintainer, 2026-11-08): the quick
   level, automatically, with a notice. Open: whether a first run should later upgrade
   itself to standard in the background, or leave that to the user.
2. **Where the store lives.** Decided (maintainer, 2026-11-08): the platform's standard
   cache directory, with `SOLPROP_CACHE_DIR` to move it.
3. **Benchmark history.** Decided (maintainer, 2026-11-08): benchmarks run calibrated at
   the thorough level and record the settings used. Open: whether the calibration's own
   variation from build to build is small enough for the history's series; phase 5
   measures it with two thorough calibrations of one build.
4. **Accuracy rule for candidates.** Decided (maintainer, 2026-11-08): fields within ten
   times the run's relative tolerance of the built-in's, finite, same acceptance
   structure; no replay or rounding-level comparison. The aim is maximum performance
   within realistic physical accuracy.
5. **Allocations without an exact row.** Decided (maintainer, 2026-11-08): use the row
   with the same span structure and the nearest smaller core count for workspace
   settings, rather than the built-in table.
6. **Concurrent first runs on one host.** Decided (maintainer, 2026-11-08): they wait for
   the calibrating process, up to its deadline.
7. **First use without changing the running process.** Decided (maintainer,
   2026-11-08): the first run applies the new workspace settings at once and the
   process settings from the next process, so that importing never calibrates. It
   arises once per machine, allocation class and release. Scans instead wait for a
   standard calibration before starting their workers (§5, §6.4), and GPU hosts are
   calibrated at bootstrap.
8. **The `fft_size="auto"` cost model (#229).** It could be fitted per machine from the
   same probes. Left for a later proposal.
9. **Which device settings pay.** Phase 6 starts by measuring them on one NVIDIA and one
   AMD device and drops those that do not; the device chunk only if #361 shows at least
   10 %.
10. **Why exhaustive plans lose.** Phase 3 records each shape class whose patient or
    measured plan lost in the step, with its alone-and-in-step timings, to tell timing
    in isolation from planning noise and from thread interplay. If serial patient plans
    rarely win, the thorough level drops them.
11. **Heterogeneous cores elsewhere.** Linux and Windows hybrid chips expose core kinds
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

### Phase 0 report — the store, the split and the built-in table

- **Scope:** phase 0, 2026-11-08, Opus 5.5 (AI) for John Travers, on
  `feature/sep-0046-machine-calibration` (LupoLab/solprop#582), commits `47271792` and
  `80f2fe31`, on `main` after #576 and #581.
- **Delivered:**
  - `native/include/solprop_native/core.hpp`, `native/src/core.cpp`,
    `native/src/general.cpp`, `native/src/general_modal.cpp`,
    `native/src/bindings.cpp`: the workspace settings leave the source. A
    `CpuExecution` record (each operation's threshold and full size, the response
    bounds, the medium and compact widths, the compact bound, the stable-team rule) is
    a field of the native problem, copied into every `CpuExecutionPool` of a workspace,
    its general modal source and its batch, which refuses members whose records differ.
    `operation_threads` reads it instead of constants; `ThreadPolicy` keeps only
    process settings. A problem without a valid record is refused. `State.cpu_execution`
    reads it back.
  - `native/src/solprop_native/_execution.py`: the built-in table, keyed by operating
    system, with the measurements behind each value moved from `core.hpp`'s comments;
    the `__APPLE__` switches and ADR-073's divisor are rules of the table, applied for
    the process at import (`BUILTIN_EXECUTION`).
  - `native/src/solprop_native/_store.py`, `_topology.py`, `_allocation.py`,
    `_fingerprint.py`, `_profile.py`: the store (§6.8), topology and allocation classes
    (§6.4), the fingerprint (§5), and the choice of row with its precedence rules;
    standard library only.
  - `native/CMakeLists.txt`: `_build.json` beside the extension.
  - `native/src/solprop_native/__init__.py`, `_wait.py`, `backend.py`: at import, an
    exact row's pool width and wait apply before the runtime starts unless the user set
    them; each workspace takes the record `PROFILE.workspace` resolves (or one given to
    `NativeBackend(cpu_execution=...)`, for stage 2); `metadata()["calibration"]`.
  - `src/solprop/calibration/` and `solprop calibrate --show/--export/--import/--clear`
    (`src/solprop/cli.py`), loading the pure modules by path; `CalibrationError`.
  - Tests: `native/tests/test_execution.py`, `test_store.py`, `test_profile.py`,
    `test_profile_import.py`, `tests/calibration/test_profiles.py`; the C++ tests take
    `native/tests/native/execution.hpp`'s record. Both conftests default
    `SOLPROP_CALIBRATION=off`, so that a developer's store never changes a test.
  - `docs/dev/architecture.md` §4 and figure 1, `docs/guide/setup/recipes.md`, and
    `changelog.d/sep-0046-machine-calibration.added.md`.
- **Deviations from the proposal:**
  - *LLVM's block time inside native calls stays a process setting.* §5 lists it among
    the workspace settings, but the outermost `CpuTeamScope` on a thread sets it, so two
    workspaces nested or batched in one call could not hold different values, and the
    waiting candidates of phase 1 (passive, a spin scoped to native calls, a
    process-wide spin) already choose it.
  - *The planning transaction moves to phase 3,* beside the per-class wisdom it applies.
    Until phase 3 writes wisdom there is none to apply, and a run plans as today.
  - *No schema change.* §9 expected one; the record goes into the backend's free-form
    settings, which ADR-063 does not version.
  - *The build is identified by a digest of the extension file,* with `_build.json` for
    its configuration. The native source digest is generated at build time and is
    readable only from the loaded extension; the file's digest also separates two builds
    of one source. An editable install keeps its built files apart from its sources, so
    every directory of the package is searched.
  - *Claims are per allocation class* (`claim-<class>`), and inconclusive attempts are
    kept in `attempts.json`, so that `auto` can wait a day (phase 5).
  - *Two edges move.* A user's `SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES` equal to the
    default is no longer divided on a spinning Mac, as any other value already was; and
    `BuildInfo`'s threshold fields report the process's table, divided where runs divide
    it, where they reported the undivided constants.
- **Decisions:** the precedence of §5 (the user's settings, then a refinement, then the
  row, then the table), the record's 24 fields as its native and stored form, and the
  allocation class's span structure (kinds, shared L2 groups, L3 groups, NUMA nodes,
  sockets, SMT, a quota) are the store's ADR, written with phase 5.
- **Verification:**
  - The table equals the former constants on every system, frozen per platform
    (`test_execution.py`), including the spinning Mac's divided values.
  - On the Xeon (GCC 16.2, OpenMP, Release): the native C++ tests (`cpu-debug` preset)
    pass; the native Python suite passes (614 tests before one fixture fix, then the
    affected modules again); `tests/test_process.py`, `test_native_forced_threads.py`,
    `test_driver.py`, `tests/calibration`, the recipe tests and the structure figures
    pass.
  - §7 item 7: a direct import applies a hand-written profile's width, wait and
    workspace settings and records them, read back (`test_profile_import.py`); two live
    states hold different records (`test_execution.py`). The notebook case belongs to
    phase 5's first use.
  - §7 item 8: exact and nearest-smaller rows, equal core counts on one and two sockets,
    an override of the thread count or the wait sending workspace settings back to the
    table (`test_profile.py`, `test_profile_import.py`).
  - §7 item 10: concurrent claims admit one, stale claims by deadline and by a dead
    owner, two hosts, corrupt and foreign generations, a failed publication keeping the
    previous generation, read-only directories (`test_store.py`); Windows runs in CI.
  - `solprop calibrate --show` and the native import agree on this machine's fingerprint
    and allocation, and the command never loads the extension.
- **Sources checked:** none new; the table's values and measurements were moved from
  `core.hpp` unchanged.
- **Limitations:**
  - Windows reads its first processor group only (64 logical CPUs) and no core kinds;
    macOS has no affinity, so its classes count cores by kind.
  - The scan workers' settings are still Pool's explicit variables until phase 5 marks
    them as solprop's.
- **Open questions:**
  - Q1. solprop's TDSE consistency cache keeps the Linux-style cache directory on macOS,
    while the store uses the platform's (`Library/Caches` in the home directory). Should
    the two share the platform's directory? (#583)
- **Handoff:** phase 1 builds the controller, its children and §6.5's selection on
  `_profile`, `_store` and `NativeBackend(cpu_execution=...)`; measuring starts there.

### Phase 1 report — the controller; waiting, team and region cost

- **Scope:** phase 1, 2026-11-08, Opus 5.5 (AI) for John Travers, on
  `feature/sep-0046-machine-calibration` (#582), commits `4baf5f94`, `e6398638` and
  `d855e8f9`.
- **Delivered:**
  - `src/solprop/calibration/selection.py`: section 6.5's rule, pure.
  - `src/solprop/calibration/child.py`: one candidate's process settings, pinned before the
    runtime starts, serving JSON lines (prepare, time, sizing, capture, regions, exit).
  - `src/solprop/calibration/problems.py`: the five calibration problems.
  - `src/solprop/calibration/controller.py`: the levels, one monotonic deadline, the
    built-in child timed twice a round for the A/A noise, host load, hypervisor steal and
    busy SMT siblings, the winner's validation by window, publication that keeps other
    classes' and higher levels' rows, nested allocations, and every child reaped.
  - `solprop calibrate [--quick|--thorough] [--budget S]`; `solprop.calibration.calibrate`.
  - `native/src/solprop_native/_wait.py`: `SOLPROP_NATIVE_WAIT_REQUEST`, solprop's own
    request for a wait, below the user's variables and a row.
  - `benchmarks/ab.py`: `--baseline-env` and `--candidate-env`.
- **Deviations from the proposal:**
  - *Windows are trial steps* (`StepDriver.timed_step`), repeated from the initial state:
    the same work for every candidate, without the step controller.
  - *A row is published when the built-in rules win*, with their values, so that `auto`
    does not calibrate the allocation again.
  - *Children warm each window with two untimed steps*: without them the A/A noise on the
    Xeon was 6–8 %, the first window after another child paying for its evicted caches.
- **Verification** (Intel Xeon Gold 6240, GCC 16.2, socket 1 unless stated):
  - `tests/calibration/test_selection.py` (ties, drift, outliers, missing rounds, a win on
    average but not in every round, a failed combination, the drop order) and
    `test_controller.py` on scripted children (every outcome, deadlines, failed children,
    held claims, nested classes, publication rules); one real quick calibration (medium).
  - Standard on 18 cores: 23 s before phases 2–3; passive waiting 1.26–1.32 times slower,
    half the socket 0.97–0.99, the built-in rules kept. Quick: 8–13 s.
  - The whole machine (36 cores, two sockets): A/A noise 10–23 % in three runs, so
    inconclusive, as the Xeon's 36-thread timings are known not to repeat; its socket
    (18 cores with their siblings) got its row as a nested allocation.
- **Limitations:** nested allocations run only while a third of the budget is left, which
  at the standard level is rarely after phases 2 and 3; the thorough level covers them.
- **Handoff:** phase 2 fits the grain in the chosen child.

### Phase 2 report — operation grain

- **Scope:** phase 2, 2026-11-08, commit `eccf9ec7`.
- **Delivered:** `src/solprop/calibration/grain.py` (candidates from the problems'
  `execution_sizing`, the width rule, the best of each setting) and the controller's grain
  stage, timed on the problems each candidate changes and validated together.
- **Deviations from the proposal:**
  - *Candidates come from the work sizes the problems run, not from per-operation
    timings through Kokkos Tools.* A kernel's label does not name its operation class, and
    an operation's threshold matters only where it falls between two work sizes a problem
    runs; so a class's candidates are its one-step moves past the nearest of them (thread
    one more size, serialise one, widen one, narrow one), at most four, with the medium
    width (2, 8) and the compact team (4, 16) beside them. §6.5 still decides each.
  - *Quick fits the medium width and compact team only* (its "coarse grain"); standard
    moves thresholds, at most ten candidates; thorough moves full sizes too.
  - *The Adams trial's passes and a coupled flow keep the table*: no trial-step window
    runs them.
- **Verification:** `tests/calibration/test_grain.py`, the grain cases of
  `test_controller.py`. On the Xeon's socket, standard admitted a medium width of 8 and a
  compact team of 4: windows' geometric mean 0.885 (`line` 0.71, `radial` 0.81,
  capillary 1.02); every stage, FFT and response threshold move lost by 2–126 %, as
  ADR-073 found for this machine. A later run on the shared machine admitted the medium
  width alone; another was inconclusive (A/A 0.11–0.23) while another user's job ran on
  the socket's hyperthreads, which the SMT check now reports.
- **Limitations:** whether these admissions hold on the tracking set is §7 item 1's
  measurement, still to run.

### Phase 3 report — FFTW planning

- **Scope:** phase 3, 2026-11-08, commit `33e9934a`.
- **Delivered:** `FftPlanning` and `FftPlanningScope` (`core.hpp`, `fft_plan.hpp`), the
  seeds' `begin_seed`/`end_seed`, `State.fft_planning`, the controller's FFT stage, the
  row's `planning` and the generation's wisdom, `PLANNING` at import and
  `NativeBackend(fft_planning=...)`.
- **Deviations from the proposal:**
  - *Efforts are chosen per group* (serial rows, threaded rows, transverse planes), the
    policy's own split, not per shape class; the wisdom is still per exact plan key, and
    a run's shape the profile does not hold is seeded at its group's effort.
  - *The planning transaction is per workspace*: it forgets before planning and after,
    imports each plan's own wisdom at its seed and builds it with `FFTW_WISDOM_ONLY`. The
    seeds and the Kokkos-FFT plans are not interleaved in the code (a workspace seeds
    several transforms, then plans them), so an estimate plan can still meet wisdom that
    another plan of the *same* workspace imported, as today's measured seeds already
    allow; nothing from an earlier workspace, a file or another candidate reaches it.
  - *Captures need no fresh process*: each capture forgets before its seed, so it holds
    exactly that plan's wisdom.
- **Verification:** `native/tests/test_planning.py` (§7 item 9: stale wisdom never
  reaches a profiled workspace, two workspaces in either order, unbuildable wisdom falls
  back and is recorded, malformed planning refused); controller tests. On the Xeon's
  socket the 16 transforms of the five problems planned at measure in 10 s; serial rows at
  estimate lost (1.12); threaded rows at measure won once (0.96, validated at 0.972,
  `line` 0.89) and were inside the noise in a later run.
- **Open questions:** Q2. §12 item 10's alone-against-in-step timings are not recorded:
  the capture times planning, not the plan alone.

### Phase 4 report — stage 2

- **Scope:** phase 4, 2026-11-08, commit `9677809b`.
- **Delivered:** `src/solprop/calibration/refine.py`, `Simulation.calibrate`,
  `solprop calibrate --recipe`, `_profile.descriptor`, refinements in the store
  (`problems/<fingerprint>/<descriptor>.json`) and the backend's lookup.
- **Deviations from the proposal:**
  - *Windows are trial steps at the start and a third and two thirds along the run*, from
    the launch field, with the operators there; accuracy compares the trial steps' fields
    with the base's (within ten times rtol), so the "same acceptance structure" test is
    moot. A low-storage march times its own step and has no fields to compare.
  - *The descriptor digests the operations' keys from `execution_sizing`* (class and work
    of every operation: dimensions, chunk, response buffers, transform sizes), the shape,
    grid, integrator, process settings and fingerprint; the plan keys follow from these and
    are not listed separately.
  - *A refinement applies only while the machine record it refined is the process's*, so it
    never outlives the profile beneath it.
- **Verification:** `tests/calibration/test_refine.py` on real native workspaces with
  scripted timings (a stored refinement reused by a later backend, not by another grid,
  not after the profile changed; inaccurate candidates and combinations kept back; noise,
  deadline, no profile); a real refinement of the radial filament ran in 34 s and was
  inconclusive on the shared Xeon (A/A 0.15).

### Phase 5 report — first use, scans, benchmarks and documentation

- **Scope:** phase 5, 2026-11-08, commits `59e15e4d`, `bc66ec18`, `c83cc0ca`, `02062f20`.
- **Delivered:** `src/solprop/calibration/first_use.py`, the hooks in
  `backends/registry.py` and `driver/executors/pool.py`, `CalibrationNotice`,
  `solprop_native.refresh_profile`, the scan workers' wait request and role, the Slurm
  script's request; `benchmarks` `--calibrate`, worker `SOLPROP_CALIBRATION=use`, the
  history's `calibration` column; `docs/guide/performance/calibration.md` and the other
  pages of section 8; ADR-074; the release notes.
- **Deviations from the proposal:**
  - *The notice is a warning and a log record, not a reporter call*: the run's reporter is
    resolved after the backend is chosen. The GUI service already turns warnings into
    events.
  - *The remote install script does not calibrate*: a GUI-managed host is often a
    cluster's login node, which is the wrong machine and shared; compute nodes calibrate
    on first use.
- **Verification:** `tests/calibration/test_first_use.py`; the Slurm snapshots; the
  benchmark tests. A first native run in a fresh script on the Xeon's socket printed the
  notice once, calibrated and ran in 19 s; the second run did not calibrate.
- **Limitations:** §7 items 1–3, 13 and 16 need the A/B campaign on the machines of §7
  item 1 (the Xeon quiet, DMOG's EPYC 7763, Google C4D under Linux and Windows and C4A,
  and the maintainer's M5 Max), which is still to run, as is phase 6.

### Verification report — section 7 items 1–4, and the revisions it forced

- **Scope:** section 7 items 1–4 and 16 on the machines of item 1, 2026-11-08, Opus 5.5
  (AI) for John Travers; the maintainer ran the Apple M5 Max. Commits `27fb0131` to
  `0aaf84f0` of #582. The rows, commands and spend are in the benchmark ledger
  ([Machine calibration against the built-in table](../../benchmarks.md#machine-calibration-against-the-built-in-table-sep-46)).
- **Method:** a thorough calibration on 16 cores (18 on the Xeon, 6 top-tier on the
  M5 Max), then the tracking set calibrated against built-in (`benchmarks ab`, a fifth
  of each case, three rounds), an A/A control and a second calibration. Both arms run
  as default runs, with no thread count of their own (an empty `SOLPROP_THREADS` and
  `SOLPROP_NATIVE_THREADS`). Without that, a worker's explicit width kept the profile's
  team width out, and with it every setting fitted at another width: the first Axion A/B
  measured nothing of its profile (every case 1.00, fields equal to the bit).
- **The first rules failed item 1.**
  - *Team width from small problems.* On a Google Axion a team of 8 won the five
    problems by 16 % and ran the tracking set's large cases up to 1.8 times slower
    (geometric mean 1.094). A round's score was the geometric mean over the problems,
    so a large win on some carried a loss on others, and every problem was far smaller
    than a production run.
  - *Untested wisdom.* On an AMD EPYC 9B45 the FFT stage's fallback, the policy's
    efforts with the captured wisdom, ran `h2_raman_averaged` 1.18 times slower. Its
    4608-point plan, chosen alone in the capture, ran slower in the step than a plan
    measured in place, and the fallback was every candidate's base, so it was never
    compared with the table.
  - *Element-count thresholds.* A full transform width from 98304 elements won 0.56 on
    the line problem's many short rows and ran `grin_nonlinear` 1.08–1.10 slower on
    both EPYCs.
- **Revisions (deviations from sections 6.5, 6.7 and 6.9, all in `75cae69b` to
  `0aaf84f0`):**
  - *Per-problem admission.* A candidate must also be slower on no problem than its
    median ratio's bar, `1 + noise + TOLERANCE`; the combined validation uses the same
    bar. `TOLERANCE` is 5 %, the maintainer's choice (section 7 item 1 asked for 2 %).
  - *Production-sized problems.* Standard and thorough also time `gnlse_large` (16384
    samples), `capillary_large` (four modes on 5120), `radial_large` (384 nodes) and
    `line_large` (4096 columns). Quick tries no narrower team. Standard takes about 90 s
    (78 s on 8 Xeon cores).
  - *FFT planning against the table.* Candidates and their validation are compared with
    the plan policy as a run without a profile plans, and nothing is published unless an
    effort wins. Each capture takes a budget out of the stage's share and keeps what it
    planned when cut short; standard plans at 0.25 s a plan. The grain stage leaves
    planning half the time left.
  - *Transforms at the table.* The grain no longer moves `fft_threshold` or `fft_full`
    (#587: key the transforms' width on rows and row length).
- **Results with the revised rules** (2 % bar):
  - *Google Axion:* medium width 8 and two thresholds in three runs; geometric means
    0.965, 0.964, 0.968 against an A/A of 1.000, no case beyond its A/A range plus 2 %;
    gains to 15 % on the few-mode and Raman cases. The team of 8 was refused on
    `radial_large` (1.17–1.18). The last run's two calibrations agreed (item 3).
  - *AMD EPYC 7763 (exclusive node, one NUMA domain):* the table, as no grain move
    won; A/B 1.003 against an A/A of 0.991.
  - *AMD EPYC 9B45, Windows Server 2025 (the CI wheel, Clang 20, LLVM OpenMP):* the
    table in both calibrations; the A/B, table against table, was not run.
  - *AMD EPYC 9B45, Linux:* inconclusive at 16 cores on two VMs (A/A 35 % and 15 %;
    one Spot VM was preempted mid-run); on a third (5 % bar) conclusive, keeping the
    table (passive 1.25, half the team 1.06); its second calibration kept the wait and
    team and admitted a copy threshold. The VM type's timings spread widely: its
    table-against-table A/B ran 0.60–1.52 per case.
  - *Intel Xeon Gold 6240, one socket:* the table in both conclusive calibrations; the
    machine is shared, and a later session was inconclusive (23 %).
  - *Intel Xeon Platinum 8581C (Emerald Rapids), 12 cores* (5 % bar): the table's
    waiting and team (passive 1.94 times slower), medium width 8 and the Axion's Raman
    and copy thresholds, in both calibrations; A/B 0.989 against an A/A of 0.996,
    `ctc` 0.87, `gnlse_raman_shock` 0.94, no case beyond its noise. The quick level
    (first use) admitted the medium width alone: 0.999, `ctc` 0.90 (section 7 item 13).
  - *Apple M5 Max* (first rules): six top-tier threads kept in both calibrations, a
    Raman threshold of 8193 admitted; A/B 1.002 and 1.008, `h2_raman_averaged` 1.04,
    the stored wisdom the revision removes. With the revised rules (5 % bar), the same
    Raman threshold in both calibrations and no planning; A/B 1.012 against an A/A of
    1.000, no case the profile changes beyond its noise (the argon cases' 1.09–1.15 ran
    identical computations, within their A/A range of 0.89–1.12).
- **Item 2.** Found as measured by hand: spin everywhere, passive refused by 1.10–1.78;
  six top-tier threads on the M5 Max; the Xeon's thresholds near the table; Windows'
  spin, now the table's since #576. Not found: the M5 Max's lower thresholds beyond
  ADR-073's, which the table already applies; the Xeon's compact cap near 8.
- **Item 3.** Same waiting and team in every pair of conclusive calibrations; the same
  grain in the last Axion pair and in every M5 Max pair, while three earlier second calibrations (Axion,
  EPYC 7763, Windows) found the grain inconclusive and kept the table.
- **Item 4.** Every noisy session reported `inconclusive` and published nothing for its
  allocation: the EPYC 9B45 (35 %, 15 %), the EPYC 7763's second (31 %), the Xeon (23 %).
- **Item 16.** Not met by the first rules (above); under the revised rules no machine
  published planning except one EPYC 9B45 calibration (`rows patient`), not yet A/B'd.
- **Open:** the few-mode cases keep part of the gain a narrower team gave them on the
  Axion (`h2_raman_averaged` 0.90 with a team of 8, 1.00 at full width with the grain;
  `argon_blueshift_envelope` 0.80 against 0.91): their operations sit above the full
  sizes, and a grain move must win alone. Searching medium width and full sizes
  together, and a source's transforms as one batch (#584), are the leads. The EPYC 9B45's
  VMs are too noisy to resolve gains of a few per cent.

### Scan-worker report — section 7 item 14 on a cluster node

- **Scope:** a scan's workers, 2026-11-09, Opus 5.5 (AI) for John Travers, on a whole
  node of two AMD EPYC 7763 (128 cores), commit `ad1df42e`; the rows are in the
  benchmark ledger
  ([Scan workers calibrated on an AMD EPYC 7763 node](../../benchmarks.md#scan-workers-calibrated-on-an-amd-epyc-7763-node-sep-46)).
- **Method:** the maintainers' common scan, an RDW tuning scan of a four-mode helium
  capillary over pressure and energy, on `Pool`: 128 single-threaded workers over
  20 × 20 points, 4 workers of 4 threads on one NUMA domain, and 8 workers of 16 threads.
  Each against the table (workers passive) and the table with spinning workers, three
  shuffled rounds, after one warm-up scan that calibrated the workers' allocation.
- **Found:**
  - *Nothing to gain.* The 4- and 16-core rows kept the table (spin, the full team, the
    grain inconclusive, no planning), so a calibrated worker differs only by spinning.
    Spinning and passive workers ran the same within the node's spread; a slower
    calibrated arm at 8 × 16 (1.05–1.17) did not repeat in nine further scans, and the
    parent's first-use check costs 0.4 s.
  - *A defect, fixed.* A one-core allocation's standard calibration ran out of its 90 s
    before three rounds of its problems in series, published nothing and would try again
    a day later. One CPU has no team, waiting or grain to choose: a one-CPU allocation
    is now never measured (`SINGLE_CPU`), so single-threaded scans neither wait nor
    see a notice (a deviation from sections 6.4 and 6.10: one CPU is never calibrated implicitly).
  - *The one-off cost:* about 90 s a worker class, once per machine, allocation class
    and release.
- **#314** (workers against threads a worker) stays open: this measured each split's
  waiting, not the split itself.

### Phase 6 report — GPU builds

- **Scope:** phase 6, 2026-11-08, on an NVIDIA H200 (CUDA 13.1, US$8.46); an AMD
  Instinct MI325X was not available in the session.
- **Measured** (benchmark ledger, "Device settings and the node chunk on an NVIDIA
  H200"): graph replay pays and stays on; team and reduction widths of 128/256 and
  512/1024 do not pay beyond the noise; the node chunk pays 6–25 % on the spatial routes
  (#361's bar is 10 %), and an 8 GiB budget takes nearly all of it. The device tests
  passed on the branch's build.
- **Decision** (the maintainer's): phase 6 keeps no calibrated device setting. There is
  no `solprop calibrate --device`, no device fingerprint or `devices/` store, and no
  bootstrap step: nothing would be fitted. The chunk is a better default, not a
  per-machine measurement, and it raises a device run's default memory use, so it lands
  through #361 as its own PR, qualified on CUDA and HIP. Section 7 item 15 is answered
  by that: no device-calibrated setting, so nothing to verify beyond the measurements.
- **Delivered:** `SOLPROP_BENCH_NODE_BUDGET` in the benchmark builders (one arm of an
  A/B at another node budget), and the measurements on #361.

### Final report — completion

- **Scope:** completion under SEP 0 §6, 2026-11-09, Opus 5.5 (AI) for John Travers.
- **Phases:** 0–6 are implemented and reported above; the verification and scan-worker
  reports cover section 7 items 1–4, 14 and 16, and phase 6 answers item 15. The CI
  gate passes on the branch.
- **Backends:** the calibration fits the native backend's CPU execution and leaves its
  results unchanged beyond rounding (section 7 item 6). `numpy_ref`, `mlx` and Julia
  have no execution settings to fit and are out of scope (section 2).
- **Manual:** [Calibrating solprop to your machine](../../../guide/performance/calibration.md)
  is new and ends with a link here. The installation, tuning, recipes and native build
  pages, the README, ADR-074 and the release notes say what changed.
- **Open questions moved to issues:**
  - §12 item 1, a quick profile upgrading itself to standard: #591.
  - The few-mode cases' narrower-team gain (verification report): #592, beside #584.
  - The grain the Axion and the Emerald Rapids agreed on, as the table's: #593.
  - §12 item 11, hybrid-core Linux and Windows machines: #594.
  - §12 item 10, why plans chosen alone lose in the step: #595.
  - §12 item 8 stays on #229, the transforms' thresholds on #587, the node chunk on
    #361 and the scan split on #314.
  - §12 item 3, answered: the waiting and team agreed in every pair of conclusive
    calibrations and the grain did in the last pairs (item 3). A benchmark row records
    the profile it ran on (`calibration`: its level and generation), so a step in a
    series can be traced to a changed profile; series are not split by it.
- **Issues answered:** #328 (thresholds measured on Windows, aarch64 and macOS, and fitted
  per machine), #332 (FFTW effort per shape and processor: measured, and the policy kept
  where nothing won) and #115 (SEP 4, superseded). #81 and #329 stay open: the
  calibration gained `gnlse_raman_shock` 6–15 % on the untuned machines without
  explaining its scaling, and did not requalify the scalar wall.

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
- 2026-11-08: AI review by GPT-6, recorded below; Part I unchanged.
- 2026-11-08: The Axion's threshold figures corrected to the distribution build with
  OpenBLAS; the earlier ones came from a build without a BLAS library.
- 2026-11-08: Response to the AI review (Opus 5.5), recorded below; Part I revised:
  process and workspace settings, allocation classes, a planning transaction for
  wisdom, one deadline, a stated selection rule, stage 2's windows and accuracy rule,
  execution descriptors, claims and generations, a visible notice, GPU chunk only if
  #361 pays, and SEP 4's parts accounted for. Four decisions left to the maintainer
  (§12 items 4–7).
- 2026-11-08: The maintainer agreed §12 items 4–7. Scans, local and remote, wait for a
  standard calibration of the workers' allocation before starting them.

### AI review — 2026-11-08

**Reviewer:** GPT-6 (AI). Reviewed all of Part I at PR #580 commit `01c198b6`,
against the native import, pool, FFT and workspace code, the scan executors,
SEP 4, SEP 25, and the linked issues. PR #576's proposed runtime changes were
also read at `06b1ce6e`; they are not in this proposal's source tree. Its performance
figures were read, not independently reproduced. The maintainer's decisions in
§12 items 1–3 are taken as given. The comments below concern how to deliver and
verify those decisions; this is not a recommendation to accept the draft yet.
All findings remain open for the drafting agent's response and the maintainer's
decisions, tracked with this proposal in [#579](https://github.com/LupoLab/solprop/issues/579).

1. **R1 — High: place calibration before irreversible runtime initialization
   (§§5, 6.1–6.2, 11).** `solprop_native/__init__.py` chooses the wait environment,
   loads `_core`, and calls `initialize_runtime` at import, including in PR #576.
   `initialize_kokkos` refuses a different pool width after initialization.
   Calibrating before *workspace construction* is therefore too late if native
   discovery, a prior run, or a direct import already initialized the runtime.
   A freshly written profile cannot change that process's waiting policy and
   pool merely by being read again. The same problem affects an interactive
   `Simulation.calibrate()` and two simulations with different refinements.
   **Requested change:** specify the initialization sequence, including how the
   build fingerprint is obtained without starting the runtime, and distinguish
   settings fixed at process start from settings owned by each workspace. Either
   arrange calibration before initialization, or state exactly which settings
   require a new process and how that process is launched. Define transport of a
   translated simulation to candidate interpreters, including unsupported live
   objects; removing the factory does not remove this requirement. Also define
   how the proposed mutable `ThreadPolicy` avoids changing existing workspaces:
   it is currently process-global and `CpuExecutionPool::get` reads it. Verify
   direct native import, an already-used notebook, and two live workspaces, and
   record the settings actually applied rather than the profile merely selected.

2. **R2 — High: key applicability to the allocation and effective policy
   (§§3, 5, 6.1–6.3).** A machine/build fingerprint plus a width does not distinguish
   eight cores sharing a cache from eight spread across sockets, different core
   kinds, SMT siblings, or a fractional CPU quota. These are precisely the costs
   motivating §2. A quick calibration inside a small job cannot supply validated
   rows for every width a later job may use. A threshold measured under spinning
   is also not validated under an explicit passive-wait override, even though the
   override must win. Scan workers receive their own width, affinity and wait
   settings through `driver/executors/pool.py`; a parent's optimum is not
   automatically theirs.
   **Requested change:** define an applicability record for each measured row:
   effective resource limits, placement/core kinds and interacting overrides,
   with a built-in fallback for unmeasured combinations. Separate portable machine
   identity from allocation applicability if desired. Include a calibration/schema
   version: the existing native source digest covers C++ and CMake sources, not
   changes to Python calibration logic. Specify how parent calibration covers the
   eventual worker partitions, including remote executors, without initializing a
   device in the controller. Test equal-width/different-placement reuse, small-job
   then large-job reuse, and an override that invalidates the measured wait/grain
   combination.

3. **R3 — High: isolate FFTW's process-wide wisdom when constructing the actual
   plans (§§4, 5, 6.3, 7.11).** Keeping separate strings on disk does not isolate
   the planner. FFTW wisdom is cumulative and global, and `FFTW_ESTIMATE` consumes
   applicable higher-effort wisdom; the manual documents both behaviours
   ([wisdom](https://www.fftw.org/fftw3_doc/Words-of-Wisdom_002dSaving-Plans.html),
   [planner flags](https://www.fftw.org/fftw3_doc/Planner-Flags.html)). In a small
   FFTW 3.3.11 probe on macOS arm64, complex forward transforms of 1,024, 1,536
   and 4,096 samples each selected a different estimate plan after measured wisdom
   was imported; each then matched the measured plan. Forgetting wisdom restored
   the original estimate plan. This checks plan structure, not speed or the native
   backend. Thus an estimate winner with "no wisdom" does not establish the
   claimed protection from earlier wisdom, including wisdom from another workspace
   in the same interpreter. Existing `_wisdom.store` also merges the file's old
   contents back into the process.
   **Requested change:** specify a serialized planning transaction that establishes
   the intended wisdom state for each plan, keeps losing/unvalidated wisdom out,
   and verifies the final combination of plans. Define exact plan keys, including
   dimensions, direction, strides, placement/alignment requirements and thread
   count; a size/kind/batch label needs an explicit mapping to those keys. State
   what happens when a stored plan cannot be reconstructed, and how runs gathering
   wisdom interact with calibrated winners. Make `off` independent of automatic
   cached wisdom, and explicitly distinguish any user-requested wisdom override.
   Test an old measured plan followed by an estimate winner, multiple workspaces
   in both construction orders, and cold versus populated stores under `off`.

4. **R4 — High: enforce the wall-clock budget outside FFTW (§§5, 6.3–6.4, 7.8–9).**
   `fftw_set_timelimit` is an approximate planning limit: an uninterruptible
   operation may overrun it, and FFTW completes estimate planning even at zero
   ([planner time limits](https://www.fftw.org/fftw3_doc/Planner-Flags.html)). It
   cannot establish "no more than one plan's time limit" for the whole operation.
   Imports, workspace construction, warm-up, validation and child shutdown also
   consume the budget. In particular, quick leaves today's measured FFT efforts
   in place while constructing synthetic workspaces, so its construction can
   contain planning outside the described FFT stage.
   **Requested change:** specify one monotonic deadline covering all work, a
   controller that can stop and reap candidate processes, reserved validation and
   cleanup time, and honest reporting of cleanup overrun. Bound synthetic memory
   as well as time, especially in automatically calibrated batch jobs. Preserve
   the last validated profile on interruption or an inconclusive upgrade, and
   fall back when too little evidence fits. Test slow startup, stalled planning,
   cancellation, insufficient memory and a budget too small for one comparison;
   do not publish a partly validated profile.

5. **R5 — High: retain a specified pilot and numerical validation contract
   (§§6.1–6.3, 7.5, 7.7, 9).** "Short windows" and "equal to rounding" do not say
   what is held fixed. Independently adaptive candidates may take different steps,
   so their timing compares different work and their fields need not agree at
   rounding level. SEP 25 distinguishes matched discrete computations from
   adaptive accuracy envelopes. SEP 4 §§6.2 and 7 already specify bounded pilots,
   identical replay edges, state/noise reset and separate adaptive checks; these
   requirements are not replaced by the profile format.
   **Requested change:** retain or replace those contracts explicitly. Name the
   accepted/attempted-step limits, original longitudinal operators and breakpoints,
   identical launch/noise, warm-up/reset of controller/history/statistics, timed
   observations, and isolation from production output and callbacks. Do not
   shorten a pressure gradient or taper to manufacture a pilot. Specify rounding
   bounds for matched methods, orders, histories and schedules; identical edges
   alone do not make ordinary variable-order replay a rounding comparison. Use
   accuracy envelopes against tighter references for adaptive runs and ordinary
   replay, with finiteness and exact structural checks. Include a
   varying guide, a noisy launch, replay/resume and a previously used simulation.
   Clarify whether every candidate is checked numerically or relies on a separately
   qualified capability set; speed alone must not admit a candidate.

6. **R6 — High: resolve GPU chunk ownership and memory admission (§§6.7, 6.9, 11).**
   Today translation puts the chunk into `Transform.node_chunk`, reduces it for
   plasma/Raman/dispersive response buffers, and imposes whole-row constraints for
   planar geometry. Native `_native_problem` consumes that value; the general
   modal route instead has a separate `general_node_chunk`. The proposed late
   device choice must reconcile those paths with the claim that the kernel
   contract does not change. A device model and its total memory also do not state
   how much memory this particular run or scan worker can use.
   **Requested change:** specify the handoff of an explicit versus automatic node
   budget, the owner of the effective chunk, and when admission and the memory
   ledger see it. Admit the complete workspace, FFT/graph scratch and concurrent
   workers against the available allocation before enlarging a chunk. Enumerate
   legal team/reduction widths against device limits and define any unsupported
   routes. Qualify all admitted routes, including planar and general modal or
   explicitly defer them, on CUDA and HIP. Keep translation device-free and
   include scan startup and reduced-memory reuse tests. This is part of
   [#361](https://github.com/LupoLab/solprop/issues/361)'s existing constraints.

7. **R7 — Medium: make a shape match a candidate for reuse, not a performance
   guarantee (§§2, 6.1, 7.7, 12.4).** The proposed signature omits batch size,
   spatial dimensions, actual FFT batches/chunks, Raman history structure and
   coupled-flow/propagator choices. Equal node counts can describe differently
   shaped Cartesian transforms. Equal response names and sample counts can also
   describe different costs along an ionising or tapering run; changing scan
   parameters need not preserve the measured operation mix. SEP 4 §4.1 explicitly
   disclaims representativeness from shape alone.
   **Requested change:** define the exact execution descriptor and applicability
   checks, then distinguish a reusable hint from a validated refinement. Explain
   how a broad cache hit is confirmed, rejected or labelled unvalidated, and what
   evidence permits reuse across scan points. Tie each speed claim to its measured
   windows. Test collisions in batch size, Cartesian dimensions and response
   structure, not only 8,192 versus 8,194 samples.

8. **R8 — Medium: specify selection and validation well enough to implement
   (§§4, 6.3, 7.1–4).** "Gain exceeds the spread" leaves the estimator, repeat
   count, candidate variability, ties and missing samples unspecified. Searching
   many settings can select noise; changing several settings together does not
   identify which individual setting to revert. Independent operation crossovers
   also interact with `stable_kernel_team`, which can promote a medium-width kernel
   after a full-width one has been selected. The final workload may therefore not
   execute the widths fitted in isolation.
   **Requested change:** state the score, minimum completed pairs, A/A statistic,
   candidate-noise rule, finite size/width ladders, treatment of nonmonotone
   crossovers, and fallback for incomplete evidence. Validate the final combined
   policy on fresh timing repetitions without profiling instrumentation, using
   realised widths and the stable-team rule. Define which cases the no-regression
   claim covers and how the quick level fits that validation in its budget. Treat
   missing load telemetry as unknown, and account for host-wide load on a shared
   node rather than comparing unrelated runnable work with one job's width.
   Restore SEP 4's concrete selection rules or justify replacements; test the
   decisions with deterministic fake timings, including ties, drift and outliers.

9. **R9 — Medium: define calibration ownership separately from atomic file
   replacement (§§5, 6.6, 7.6, 7.9).** The borrowed `_wisdom.py` protocol uses a
   blocking POSIX writer lock around merge-and-replace and no process lock on
   Windows. That is not yet the nonblocking, single-calibrator protocol promised
   for concurrent first runs. With a shared home, the same machine-type fingerprint
   can cover several physical nodes; storage ownership and contention on a node
   are different concerns. Other first runs continuing on built-ins can also make
   the calibration owner's measurements busy.
   **Requested change:** define a portable nonblocking claim, its scope, crash
   cleanup, and atomic publication of a consistent profile/wisdom generation.
   Explain interactions with `--clear`, imports and simultaneous quick/thorough
   runs, including preservation of a valid profile when an upgrade fails. Specify
   lookup past an inconclusive local entry to a valid read-only shared profile.
   Verify Windows as well as POSIX, owner death, simultaneous readers and writers,
   and two nodes sharing a store; confirm that competing production work causes
   rejection rather than a misleading calibration.

10. **R10 — Medium: make the first-use notice visible under normal library use
    (§§1, 5, 7.9).** Both packages install `logging.NullHandler`. In a clean
    interpreter at the reviewed commit, importing `solprop` and issuing
    `logging.getLogger("solprop.calibration").warning(...)` produced no stderr
    output. Raising the log level to warning therefore does not meet the promise
    that a first-use pause is never silent when no handler was configured.
    **Requested change:** specify delivery through the existing progress/notice
    machinery, with the intended terminal, notebook, redirected-output and explicit
    quiet behaviour. Preserve the library's no-global-handler rule. Test visible
    notice before calibration starts with no configured logger, and one notice
    rather than duplicates when logging is configured.

11. **R11 — Medium: reconcile the superseded scope and issue claims
    (§§2–3, 6.1, 11).** §3 says stage 2 can choose the scan worker/thread split,
    but §§5–6 give it a single simulation and §2 excludes additional scheduling.
    SEP 4 itself leaves worker counts to the executor. Issue
    [#314](https://github.com/LupoLab/solprop/issues/314) requires a concurrent scan
    comparison; a single-process speed measurement cannot answer it. Similarly,
    [#303](https://github.com/LupoLab/solprop/issues/303) includes GEMM providers,
    CPU chunks and propagator policy that this draft does not specify. The header
    supersedes SEP 4 on acceptance, while phase 4 lists supersession again.
    **Requested change:** say which existing controls are retained, deliberately
    deferred or replaced, and leave the uncovered work on its existing issues.
    Either define an executor-owned scan-throughput objective, allocation and
    measurement, or remove the claim that this stage chooses the split. Choose
    one supersession point and reconcile the tracking issue and index then.

**Verification and source limits.** The two small probes above were run with
Python 3.14.5 on macOS arm64; the FFTW probe used the independently installed
FFTW 3.3.11 library, not solprop-native. FFTW's planner, wisdom and
[thread-safety documentation](https://www.fftw.org/fftw3_doc/Thread-safety.html)
were checked directly. No physical constants were introduced. No native propagation,
performance qualification, remote-machine or GPU experiment was run, and no
performance figure in Part I was independently verified. The review changes only
the review header fields and Part III; responses and revisions to Part I belong
to the drafting agent and maintainer.

### Response to review — 2026-11-08 (drafting agent, Opus 5.5)

All eleven findings were checked against the tree (the native import, `build_info`,
the wisdom calls in `fft_plan.hpp` and `_wisdom.py`, `solprop.log`) and accepted, R5 in
the form the maintainer's guidance sets. Part I is revised as follows. Four decisions
are left to the maintainer as §12 items 4–7.

- **R1.** Accepted. §5 now separates *process settings* (waiting, pool width, placement),
  read from the profile at native import before the extension loads, from *workspace
  settings* (thresholds, medium width, compact cap, stable-team rule, scoped block time,
  FFT efforts and wisdom), copied into each workspace when it is built. `ThreadPolicy`
  keeps only the former; a per-workspace execution record holds the latter (§6.2), so two
  live workspaces may differ. Every measurement runs in child processes under a
  controller that never loads the extension (§6.2); stage 1 uses shipped problems and
  stage 2 varies only workspace settings in the calling process, so no simulation is
  transported between interpreters (§6.1). Importing never calibrates: a first run
  applies the new workspace settings at once and the process settings from the next
  process (§5; §12 item 7 asks the maintainer to confirm this over calibrating at
  import). The fingerprint is read from a build record beside the extension, so it needs
  no runtime (§5). The run records the settings applied, read back (§5); §7 item 7 tests
  a direct import, a used notebook and two workspaces.
- **R2.** Accepted. Profiles hold rows per *allocation class*: core count and kinds,
  cache groups, NUMA nodes and sockets spanned, SMT, CPU quota (§6.4). The standard and
  thorough levels also measure nested classes inside the allocation. Each row records the
  waiting policy it was fitted under, and an override that changes it sends those
  settings back to the table. Scan workers' class is calibrated by the parent in a child
  pinned like a worker, with its limits recorded; remote executors calibrate on their
  own hosts; no controller initialises a device. The calibration's own version is in the
  fingerprint. The nearest-class rule is §12 item 5, for the maintainer. §7 item 8.
- **R3.** Accepted; the reviewer's probe agrees with FFTW's manual, and the earlier "no
  wisdom" protection was wrong within a process. §6.7 now applies wisdom in a planning
  transaction: forget, plan the estimate winners with no wisdom present, import exactly
  the winners' wisdom and plan those with `FFTW_WISDOM_ONLY`, fall back to estimate when a
  plan no longer reconstructs, forget again. The class key is the exact plan requested,
  strides, alignment and threads included. With a profile, runs no longer merge wisdom
  into the store; `off` uses no automatic wisdom; a user's `SOLPROP_FFTW_WISDOM` keeps
  today's behaviour and takes precedence. §7 item 9.
- **R4.** Accepted. §6.9 gives each level one monotonic deadline for all its work, a
  controller that terminates and reaps children, a reserved share for validation and
  publication, and honest reporting of cleanup; FFTW's time limit only shortens plans.
  Children plan at estimate outside the FFT stage (§6.3). Synthetic problems are bounded
  to a fifth of the allocation's memory. Nothing partial is published, and the previous
  generation stays current (§6.8). §7 item 11.
- **R5.** Accepted in a lighter form, on the maintainer's guidance (2026-11-08) that the
  aim is maximum performance within physically realistic accuracy, with no need for
  replay or rounding-level repeatability. §6.6 specifies the windows: the simulation's
  own operators and breakpoints, windows placed along the run rather than shortened,
  the same noise seed, output and callbacks off, a fixed count of accepted steps after a
  warm-up, timing per step, and a repeat-then-drop rule when step counts differ by more
  than 5 %. Every candidate must end finite, with the same acceptance structure, and
  with fields within ten times the run's relative tolerance of the built-in's; speed
  alone never admits a candidate. §12 item 4 asks the maintainer to confirm the
  tolerance. §3 records that this replaces SEP 4's pilot contract. §7 items 6 and 12.
- **R6.** Accepted. §6.14 now starts phase 6 by measuring which device settings pay, and
  takes the node chunk only if #361's re-measurement shows at least 10 %. If it does,
  translation stays device-free and hands the device workspace the chunk's *rule* per
  route (granularity, bytes per column with response buffers, an explicit budget); the
  workspace admits the whole workspace, scratch and concurrent workers against free
  device memory before enlarging the chunk, and the ledger records it. The general modal
  route's own chunk stays as today until qualified. Team and reduction widths are
  limited to what the device supports per kernel; uncalibrated routes keep compiled
  defaults. §7 item 15.
- **R7.** Accepted. §6.6 replaces the signature with an *execution descriptor*: problem
  dimensions and Cartesian or radial shape and batch, every FFT plan key, responses and
  their buffers, stepper and propagator, node chunk. Only equal descriptors reuse; a
  refinement is validated on windows spread along the run and can only add to the
  machine profile. §7 item 12 tests collisions in batch, Cartesian shape and response
  structure. The former open question on coarse signatures is withdrawn.
- **R8.** Accepted. §6.5 now states the rule: interleaved rounds (3, 5 or 7 by level),
  the median ratio as score, A/A noise as the largest round deviation and at least 2 %,
  admission only when a candidate wins every round by more than the noise, ties and
  incomplete evidence keeping the built-in, at most four candidates a setting, and
  monotone-envelope crossovers (§6.3). The admitted settings are validated together on
  fresh rounds without profiling hooks through the full code path, so realised widths
  and the stable-team rule are what is timed; on failure groups are dropped in a fixed
  order. Load is judged host-wide, missing telemetry is unknown and left to the A/A
  noise. §7 items 1 (scope of the claim) and 5 (fake timings).
- **R9.** Accepted. §6.8 defines an exclusive-create claim file (portable to Windows)
  with host, process, start and deadline; stale claims; publication of one complete
  generation followed by an atomic swap of `current`; lookup past inconclusive entries to
  read-only directories; `--clear` that leaves a live claim alone; and one calibrator per
  allocation class across hosts sharing a home. Concurrent first runs on one host wait
  for the calibrator so as not to load it (§12 item 6, for the maintainer). §7 item 10.
- **R10.** Confirmed: `logging.NullHandler` makes a library logger warning invisible.
  §6.10 issues the notice as a warning of its own category, `CalibrationNotice`, which
  Python shows once by default in terminals and notebooks without any handler, and
  which the standard filters or `SOLPROP_CALIBRATION=use` silence; it is also logged and
  passed to an active progress reporter. §7 item 14.
- **R11.** Accepted. §3 now says what SEP 4 parts are replaced, not carried over, or
  left on their issues. The claim that stage 2 chooses the scan split is withdrawn;
  #314 stays open. #303 is listed as partly answered. Supersession happens once, at
  acceptance (§11); phase 4 no longer repeats it.
- 2026-11-08: Accepted by John Travers (human review and decisions) in #580; SEP 4
  superseded and archived.
- 2026-11-08: Clarified before the acceptance merged, after an automated review (Codex) on
  #580: a first run, still on its original process settings, applies only workspace
  settings validated under them (§5, §6.4).
- 2026-11-08: Implementation started (Opus 5.5) on `feature/sep-0046-machine-calibration`.
- 2026-11-08: Phase 0 complete: the store, the split into process and workspace settings,
  and the built-in table.
- 2026-11-08: Phases 1–5 implemented; their verification on the machines of section 7 item
  1 and phase 6 remain.
- 2026-11-08: Verification on the machines of section 7 item 1. The first rules failed
  item 1; revised (per-problem admission at the noise plus 5 %, production-sized
  problems, planning against the table, transforms at the table). Phase 6 measured on
  an NVIDIA H200; no device setting kept as calibrated, the chunk to #361.
- 2026-11-09: Scan workers measured on an AMD EPYC 7763 node: the table kept, workers'
  waiting immaterial; one-CPU allocations are no longer calibrated.
- 2026-11-09: Final. Open questions moved to #591–#595; archived.
