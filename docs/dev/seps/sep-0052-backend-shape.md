# SEP 52 — One backend shape and a small native seam

| | |
|---|---|
| **SEP** | 52 |
| **Title** | One backend shape and a small native seam |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-15; revised after review, 2026-11-15) |
| **Status** | Draft |
| **Review status** | In review |
| **Reviewers** | GPT-6 (AI) |
| **Type** | Change |
| **Created** | 2026-11-15 |
| **Resolution** | — |
| **Requires** | SEPs 25, 39, 45 and 47 |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Give `numpy_ref` and `native` the same internal structure, and make the C++/Python seam of
`native` as small as that structure allows. SEP 47 gave both backends one external
contract: `prepare` returns a prepared execution, and each `start` opens a `RunSession`
that owns one trajectory. Behind that contract the two backends are built differently.
Native composes a workspace, engines, marches and a session in C++. The reference is an
inheritance stack: `NumpyRefBackend` (3,000 lines) on `HostDriver` (2,600 lines), with
`HostSession` steering it through halt hooks.

This proposal splits both backends into the same five parts:

- the backend's own lowering;
- a **workspace** that holds the state and does every state-sized operation;
- **engines**, the operations a march asks of the workspace;
- **marches**, which own their controller state;
- a **session** that owns the schedule, the boundary monitor and the events.

Both backends read the same run records, built once by shared code, and `HostDriver` is
deleted. Two things that no longer earn their cost are removed: the `Batch` scan executor
with every group path behind it, and the leading axis of one that the trajectory's arrays
still carry for it, in translation, in the backends and on disk.

The native binding shrinks:

| | Today | Proposed |
|---|---|---|
| Bound names | about 220 | a `Workspace` built from one `Problem`, its `start` and `probe`, a `Session`, and a few module functions |
| Configuration | about 400 names in 15 to 20 calls | one `Problem` and one run record |
| Record converters | about 21, hand-written | one generic converter |

Test-only entry points move to a private `_testing` module, which exposes native's engines
through the same Python protocols as the reference's.

Success is judged by four things:

- **Numbers:** the session suites pass with their cases and bounds unchanged.
- **Measured changes:** each change to the reference's numbers is measured and accepted case
  by case.
- **Coverage:** no test or line of coverage is lost except with deleted code.
- **Size:** the measured line counts fall by roughly the estimate of §6.7.

### 2. Motivation and scope

**Why now.** The internal seams were shaped by a history that no longer applies:

- several Python backends that shared `HostDriver`;
- a native backend that subclassed it;
- a Python loop that drove native one step at a time;
- a lockstep batch meant to amortise launch costs on a device.

SEP 43 retired `numba_cpu`. ADR-077 fixed the backends at two. SEP 47 moved every native run
into its session. What remains is a reference built for a family of subclasses that has
one member, a native binding that still carries the per-step interface the session
replaced, and a batch executor that no longer batches. solprop's backends are unstable in
0.x (ADR-058), and `solprop-native` is pinned to solprop's own version. A restructure now
costs no user anything. Once others contribute backend features, the same restructure
costs every open branch.

**The native seam today.** The nanobind module `solprop_native._core` is defined in
`native/src/bindings.cpp` (4,559 lines), `bindings_transverse.cpp` and `helpers.cpp`.
It binds about 220 names:

- **`State`, 96 names.**
  - 31 are on the production path: construction, configuration and a few readbacks.
  - 12 are run metadata and 7 serve the timing probe.
  - 39 are reached only from tests.
  - 7 are reached from nowhere, for example `set_flow_actions`, `table_evaluate`,
    `adams_fused_accept`, `adams_observe` and `adams_exponential_observe`.
- **Module functions, 31.** 8 are on the production path, 3 serve diagnostics or
  calibration, 19 are for tests and 1 is unused.
- **Test-only classes:** `Monitor`, `TransverseEdge`, `ScriptedMarch` and `ScriptedRkMarch`.

Before a session marches, Python makes 15 to 20 configuration calls:

- the `Problem`, 161 leaf fields in 14 groups;
- z-table and flux-table uploads;
- stage tables, the march policy, controller weights, the flow guard and the low-storage
  settings;
- four monitor and edge bindings;
- the session plan.

About 400 distinct names cross. About 21 of the records are dictionaries that the binding
converts one string key at a time, and Python builds most of them by hand beside a
dataclass for the same thing. Some values cross twice:

- the tableau nodes;
- `low_storage_flows`;
- `compensated` and `interpolate`.

Counts cross beside the arrays they describe, and C++ checks them again. About 60 % of
`bindings.cpp` converts or forwards. Real logic sits inside the binding unit itself: the
coupled-flow guard, the march entry points and the lease between state and session. ctest
cannot reach it there. Of the stub `_core.pyi`, 247 lines are generated; the remaining
~700 are hand-written and already miss `ScriptedRkMarch`.

The adapter `solprop_native.backend` still holds the Python engine for the native Adams
history (`_ResidentAdams`). Its per-step methods have no production caller: the reference
Adams march runs only on `numpy_ref`, and the native march is compiled (SEP 39). The
runtime decisions, the dense observation at the planes and the Adams start-up are native.
The only C++-to-Python call is the TDSE host source (SEP 49).

**The reference today.** `HostDriver` holds:

- the Runge–Kutta march, adaptive and replayed;
- landing, halts and yields, step planes and natural stops;
- the dense hold, the periodic statistics buffer and the stage scalars;
- `prepare`, metadata and snapshots;
- every boundary-monitor check, the transverse edge and the noise models.

It also defines the `HostStepping` protocol. `NumpyRefBackend` overrides 18 of its methods
and supplies the physics. The Adams and low-storage marches (`HostAdamsMarch`,
`HostLowStorageMarch`) write the driver's private fields directly: `_z`, `_state`,
`_proposal`, `_schedule_index`, `_accepted_norms` and `_in_flight`. `HostSession` owns the
events and receipts, but schedules nothing itself: it sets a halt check and calls
`propagate_to`. Monitor bookkeeping is interleaved with physics:

- the split collar's sums are kept in `_finish`;
- the quadrature alarm's latch is kept in `_begin_plane` and `_on_accepted_step`;
- the TDSE reservation sits on the backend, not the engine.

A boundary warning is raised synchronously inside the driver on the reference, but
reported as a `WARNING` barrier event by native. The service loop special-cases
`HostSession` to read its driver (`driver/session_loop.py`).

Native's C++ already has the shape this proposal asks of both:

- `session::Resident` is the workspace;
- `rk::Engine` and `march::Engine` are the engines, and native's low-storage step is also
  an `rk::Engine`;
- `adams_march.cpp` and `rk_march.cpp` are the marches;
- `session.cpp` is the session.

The reference has half of it in Python: the `AdamsEngine` and `LowStorageEngine` protocols.

**The batch executor today.** `Batch(size)` advances compatible fixed-schedule scan points
"in lockstep inside one backend". It no longer does what that promises:

- **On native,** a group builds a complete backend, with its own workspace and plans, for
  every member, and runs one session per member, fusing nothing. It costs `size` times the
  memory of `Local`, which reuses one backend across points, for no gain in speed.
- **On the reference,** the lockstep group (`HostGroup`, `NumpyBatchBackend`) fuses only the
  Runge–Kutta stage sums of its members, and the reference's speed is not one of its goals.

Neither the GUI, the recipes nor the examples use it. It needs about 3,000 lines across the
driver and both backends: the group protocols, the group service loop and
`compatible_groups`, the group sessions, the lockstep runner and `select_batch`. It is also
the only reason for the **leading axis of one**, which the trajectory's arrays still carry:

- the operator's value tables in the `PropagationProblem` (`translate/linop.py`,
  `translate/stage.py`);
- the boundary functions that take a four-axis state (`translate/collar_check.py`,
  `boundaries.SplitCrossing`, `monitor.BoundaryMonitor`, #618);
- the reference's passes and history;
- native's load and readback;
- the memory ledgers;
- the stored state of a `MarchSnapshot`.

`n_batch` appears on 165 lines in 20 files.

**In scope:**

- Both backends' internal structure: workspace, engines, marches and session, with the
  shared run records.
- The deletion of `HostDriver` and of the `HostStepping`, `BatchBackend`, `GroupSession` and
  `PreparedGroup` protocols, with `Backend.max_batch` and `Backend.prepare_group`.
- The removal of the `Batch` executor and every group path, on both backends and in the
  driver.
- The removal of the leading axis of one, everywhere it is carried for the batch:
  translation, the boundary functions, both backends and the stored snapshot.
- The native binding: what crosses, how records are converted, the stubs, the test-only
  module and where binding-side logic lives.
- The timing probe's native form.
- Porting or deleting every test that depends on what is removed, under the rules of §7.

**Out of scope:**

- The rest of the public interface: `Simulation`, the other scan executors, recipes and the
  output's other content.
- The kernel contract's numerical content: the route union, the response descriptors and
  every operator's values. Only the unit axis of its value tables changes.
- `RunRequest` and the `RunSession` and event contract of SEP 47.
- #110's split of the integrator from its tolerance, and its per-route pipelines.
- A fused batch through native's stage passes (#356), which would now need its own group
  interface.
- New features and performance work.
- Any change to native's numerics, beyond following the reference where §4 allows it.

### 3. Context

The design pages:

- [backends.md](../backends.md) §1, the seam and the table of marches;
- [architecture.md](../architecture.md) §9;
- [kernel-contract.md](../kernel-contract.md) §3, the value tables; §8, the
  periodic-statistics hooks; and §§11–12, the compiled marches;
- [conventions.md](../conventions.md) §6, the axis names;
- [scans.md](../scans.md) and the user guide's
  [scans](../../guide/workflows/scans.md) page, for `Batch`;
- the native package's `native/README.md`.

**Prior proposals and decisions:**

- **SEP 47:** backend-owned sessions; its §6.7 on groups and snapshots; the explicit test
  interfaces of its §6.8.
- **SEP 39 and SEP 45:** the compiled Adams and Runge–Kutta marches, and their
  decision-trace tests.
- **SEP 25:** the rounding-level comparison rules.
- **SEP 29:** the helpers' ownership, `PLAIN` and `NATIVE`.
- **SEP 46:** calibration's use of the timing probe.
- **SEP 49:** the TDSE host source. Its §10.1 records the deferral of slimming `numpy_ref`
  to a test library, which keeps the reference's complete runs. This SEP keeps them too.
  It removes the reference's group and batch paths, the narrower part §10.1 names, together
  with native's, because the executor they serve goes.
- **ADR-058:** 0.x stability.
- **ADR-068:** the seam has no version number.
- **ADR-077:** two backends, binary64 only.

**Issues:**

- **#110**, the step loops and the kernel seam. This SEP decides two of its four items. The
  step loops become one march per method on each backend, over engine protocols the two
  backends share. The batch axis is removed, with the lockstep batch. The
  integrator–tolerance split and the route union stay with #110, which is narrowed to them.
- **#618**, the batch axis left after SEP 47. Fixed in full.
- **#171**, one diverging point fails every member of its `Batch` group. Closed as moot,
  with the executor.
- **#828**, multi-pass cells cannot use the `Batch` executor. Closed as moot.
- **#344**, the low-storage hook names. Its naming item is fixed: the `_adams_*` hooks
  become the workspace's operations, with names that fit every march.
- **#107**, one host statistics module. Informed: native keeps its statistics in C++
  (SEP 47 §6.4), and the reference's move into its workspace. This SEP recommends closing
  #107 on that decision.
- **#356**, a fused GPU batch. Changed: there is no group interface left to extend, so the
  issue records that a fused batch would bring its own, and its measurement still decides
  whether it is worth one.

The output-schema change that `main` already carries since 0.2.0
(`changelog.d/output-schema-split.breaking.md`) means files and scans written by 0.2.0
cannot be resumed by the next release. The snapshot change of §5.4 therefore removes no
resumability that release still has.

### 4. Theory

No physics changes. This proposal changes how computations are arranged, not what they
compute. Comparisons follow SEP 25: a matched discrete computation is held to a rounding
budget fixed before the comparison, and an adaptive run to its accuracy envelope.

**Rounding and simplification.** The reference may change its numbers where a
simplification needs it. The expected first case is the Runge–Kutta engine keeping its
trial and committing it itself, where today the driver assigns the state the trial returns.
Each such change is recorded in its phase report, and is accepted only on that evidence,
never by regenerating a table without it. The report gives:

- the computation it changes;
- the largest difference found on fixed schedules, against that computation's SEP 25
  budget;
- the frozen tables and decision traces it regenerates;
- whether native follows it.

A difference above rounding needs a stated cause and the maintainer's agreement in the
report.

**Measurements stay where they are.** A move of ownership never moves a measurement to a
different physical state. The split collar's removal and its guard occupancies are the case
in point. They are reductions of the field after the accepted step's first band mask and
**before** the temporal split:

- the removed power is $P\,(1 - e^{-\alpha h})$;
- the occupancies are judged on $P\,e^{-\alpha h/2}$;

where $P$ is the power in each bin before the split, $\alpha$ the collar's rate and $h$ the
step (`BoundaryMonitor.split_sums`; [backends.md](../backends.md) §1). Evaluating them on
the damped, re-masked field changes the removal. Reconstructing the earlier field from it
amplifies mask residue into fictitious loss, which
`tests/test_monitor.py::test_a_deep_collar_reads_no_removal_from_residue` guards against.
The simplification permission above does not extend to such a move.

Native's session transcribes the reference's scheduling decisions, so a change to the
reference's march or session bookkeeping is mirrored in C++ in the same phase. The
decision-trace tests then hold the two together again.

### 5. Proposed interface

`Batch` leaves `solprop.driver.executors`. Its users take `Local` for one process, or
`Pool`, `Queue` or `Slurm` for several, as the scans guide already describes. Nothing else
changes for a user. The other names below are private to the backends or to the binding.

#### 5.1 The shared parts

`solprop.backends` keeps SEP 47's objects:

- `PropagationProblem` and `RunRequest` as inputs;
- a prepared execution;
- `RunSession` and its events.

Behind them, each backend has the same five parts, with names mirrored on both sides:

| Part | Responsibility | `numpy_ref` | `native` |
|---|---|---|---|
| Lowering | Problem and request into the backend's own form | Constructor tables (today's `SourceTables` and friends) | `solprop_native` lowering into `_core.Problem` |
| Workspace | The state and every state-sized operation: load, stage, source, finish, mask, statistics, boundary and edge readings, export. Reused across the points of a scan. Holds no march state | `NumpyWorkspace` | `_core.Workspace` over the C++ state (`session::Resident`) |
| Engines | What a march asks of the workspace (§5.2) | Python classes over `NumpyWorkspace` | `rk::Engine`, `march::Engine` |
| Marches | Each march's controller, landing, history and call ledger, in its own state object | Python marches over `rk_core` and `adams_core` | `rk_march.cpp`, `adams_march.cpp` |
| Session | Planes, step planes, natural stops, yields, dense holds and the dense observation, periodic rows, the snapshot latch, the boundary monitor and transverse edge, the noise reset, events and exports | `HostSession` | `session.cpp` |

**The run records** are Python dataclasses in `solprop.backends`, built once per run by
shared code from the problem, the request and the `Start`. The reference reads them
directly; native converts them with the converter of §5.3. They are:

- the method's policy: `MarchPolicy` for Adams, a new `RungeKuttaPolicy`, and the
  low-storage settings;
- the schedule, when one is fixed;
- the plane plan: planes, step planes, statistics cadence, replay length and capacities;
- the monitor and edge configuration, and a resumed run's continuation records. These move
  from `solprop_native._watch` and `_transverse` into solprop.

The statistics lowering stays per backend: native's is a C++ plan (SEP 47 §6.4), and the
reference evaluates `StatsSpec` directly.

**The `Backend` protocol, completely.** After this SEP it has these members:

| Member | Kind | Consumers |
|---|---|---|
| `name` | attribute | the registry, refusals and metadata (`driver/run.py`, `driver/session_loop.py`) |
| `reads_transverse_edge` | attribute | the refusal of an unread real-space collar before the first step (`session_loop._refuse_unread_collar`) |
| `request` | property | the run request the backend's buffers and plans were prepared for, which a reused backend's runs are checked against (`driver/run.py::run_problem`, `driver/_scan_run.py`) |
| `prepare(problem, request)` | method | the session loop |
| `timing_probe(start)` | method | `Simulation.estimate` and SEP 46 calibration (§5.5) |
| `metadata()`, `memory_ledger()`, `available_device_memory()` | methods | run records, scans and admission |
| `march_export_refusal()`, `edge_memory(space)` | methods | the driver's checkpoint and memory decisions |
| `close()`, `closed` | method, property | ownership |

These leave it: `max_batch`, `prepare_group`, the `HostStepping` and `BatchBackend`
protocols, and the three periodic-statistics storage hooks. The class attribute `helpers`
(SEP 29) stays on each backend.

**Boundary warnings** become `WARNING` barrier events on both backends. The service loop's
`HostSession` special case goes: both sessions report their execution facts through
`execution_record`.

#### 5.2 The engine protocols

The Python protocols mirror the C++ interfaces one-to-one, every argument included.

```text
class RungeKuttaEngine(Protocol):            # rk::Engine; the low-storage step too
    def trial(self, z: float, dz: float) -> float: ...       # the error norm
    def accept(self, dz: float) -> AcceptRecord: ...          # finish and commit
    def extend(self, z: float, dz: float) -> None: ...        # dense stages at a plane
    def reading(self) -> BoundaryReading: ...                 # the committed state's
    @property
    def in_flight(self) -> bool: ...

class AdamsEngine(Protocol):                 # march::Engine
    def start(self, z: float) -> None: ...
    def trial(self, z: float, dz: float, stencil: Stencil) -> NDArray: ...
    def accept(self, dz: float, free: int, stencil: Stencil,
               kept: Sequence[HistoryEntry]) -> AcceptRecord: ...
    def chord_phase(self, z1: float, z2: float) -> float: ...
    def step_carrier(self, order: int) -> NDArray: ...        # the last trial's rotations
    def begin_attempt(self, cost: int) -> None: ...           # the TDSE budget
    def end_attempt(self, outcome: AttemptOutcome) -> None: ...
    def reading(self) -> BoundaryReading: ...
```

**The commit contract:**

- **The Runge–Kutta trial** keeps its stages and end state in the workspace. It commits
  nothing until `accept`, and a rejected trial leaves the committed state as it was.
- **The low-storage step is the same engine, with one documented difference.** Its trial
  overwrites the state, the register `A`. From the start of `trial` until `accept` returns,
  `in_flight` is true. An exception or a non-finite result in that interval leaves no
  committed state at that position, so the session ends the run failed with no checkpoint
  there, as both backends do today. The engine owns the in-flight flag and the finish; the
  march owns the committed position. The stiffness diagnostic is the engine's, a field of
  its `AcceptRecord` (NaN on native, as today). The march reads it from there, and the
  session reads the march's state, never a private field.
- **The Adams `accept`** receives the retained history entries, oldest first, which the
  interaction formulation's transport uses.
- **`step_carrier`** returns the last trial's carrier rotations. The march stores them in
  the accepted step's record, and the session's dense observation applies them to the
  interpolation weights, as `adams_march.cpp` and `dense.cpp` do today.
- **The dense hold, the dense observation and the snapshot rows** are the session's, as they
  already are on native. The engines' `hold`, `restore`, `observe` and snapshot methods
  leave the Python protocol.
- **The TDSE reservation and the trial's outcome** move from the backend onto the engine, as
  `begin_attempt` and `end_attempt`.
- **The split collar's sums.** An accepted step's finish computes them in the workspace,
  from the masked field before the split (§4). It returns them in its `AcceptRecord`, and
  the session accumulates and drains them:
  - the removal adds over every accepted step since the previous check, and the next
    boundary reading drains it;
  - the found occupancies are those of the latest accepted step;
  - a rejected trial, a dense observation and a probe trial contribute nothing, because
    none of them finishes a step;
  - loading a state empties the accumulator, and attaching a monitor, at a launch or a
    resume, starts it from zero, as each backend does today.

  These rules transcribe today's behaviour, and phase 3 pins them with tests before any
  ownership moves.

A march holds its own state and touches no other object's fields.

#### 5.3 The native binding

```text
_core
  build_record() -> dict
  configure_runtime(record)                    # threads, spin, BLAS, as one record
  wisdom_export() -> str | None; wisdom_import(text) -> bool
  available_device_memory() -> int | None
  BackendRefusal, EnvelopeConvergenceError, SessionLifecycleError
  schemas() -> dict                            # every record's fields and array specs

  Problem                                      # one record; includes every set-once
                                               # table, guard and geometry
  Workspace(problem)
    .start(run, host_source=None) -> Session   # run: the shared run records
    .probe(start) -> Probe                     # §5.5
    .diagnostics() -> dict                     # metadata, memory, sizing, planning,
                                               # crossed bytes
    .close(); .closed
  Session
    .advance(work_limit, deadline) -> Batch
    .acknowledge(sequence)
    .export(ticket, recipe, out)
    .noise_phase(ticket, z) -> ndarray
    .request_stop(); .request_snapshot()
    .status() -> Status
    .snapshot(ticket) -> Snapshot | None       # march state, field and history rows
    .records() -> dict                         # monitor, edge, cap, quadrature peak
    .continue_monitor(continuation)
    .fail_outside() -> Batch | None
    .close()

_core._testing                                 # private; no contract (§6.5)
```

**One `Problem`, one run record.** Everything that is fixed for a workspace crosses in
`Problem`:

- the z, flux and stage tables;
- the flow guard's certificate;
- the controller weights;
- the low-storage settings;
- the monitor's weights and occupancy;
- the edge geometry and its lines.

Everything that belongs to one run crosses in the run record:

- the initial field and noise;
- the method policy and schedule;
- the plane plan and statistics plans;
- the monitor configuration;
- a resume's march state and history rows, and its monitor continuation.

Python derives nothing that C++ then derives again. There are three real constraints:

- a workspace is reused across the points of a scan;
- the TDSE host source is a live Python object;
- the CPU execution record's second calibration stage reads a built workspace's sizing.

The first two keep `start` separate from the workspace. The third keeps the rebuild that
exists today. A workspace holds at most one live session or probe; `start` and `probe`
refuse while another holds it, through the lease that exists today.

**Records by field table.** Each C++ record declares its fields once, next to its
definition: one line per field. A field gives its name, its member and, for an array:

- its dtype;
- its rank;
- its extents, written in terms of the record's own counts or another field's extents (for
  example the Raman table's `(n_terms, 2, n_to)`).

A single template converts a Python dataclass, or a mapping, into the record and back. It
handles scalars, enumerations by name, optionals, nested records, lists of records and
arrays. It refuses, naming the field's path:

- a missing or unknown field;
- a wrong dtype;
- a non-contiguous or overlapping array;
- a wrong rank;
- extents that disagree with the relation;
- a size whose product overflows.

All of these are checked before anything is allocated. A count that an array's extents give
is read from the array, not sent beside it. The relations it took part in are kept as
extent relations, so a transposed table with the right element count is still refused, as
`native/tests/test_backend.py` requires today.

**Array ownership:**

- **Input arrays are borrowed** for the conversion only, with the interpreter lock held,
  and copied into the record's own storage. Read-only arrays are admitted, since nothing
  writes them.
- **Arrays returned in `Batch` and `Snapshot` own their memory.** Each holds a capsule over
  storage moved out of the session, so it stays valid after acknowledgment, workspace reuse
  and close, as today's batch arrays do.
- **Fewer copies is an aim, not a criterion.** Removing the full `Problem` copy at
  construction is expected; any further reduction is measured (§7) before it is claimed.

**`_core.schemas()`** exports every table: names, dtypes, ranks and extent relations. One
test holds each Python record to its C++ table, so a renamed field, a changed dtype or a
lost extent relation fails. The `Fields<>` helper that binds `Problem` today is that
converter's first case.

**Generated stubs.** The whole `_core.pyi` is generated by nanobind's `stubgen`, and the
native gate checks it, as it checks `Problem`'s part today. The binding gives each record
argument and result a signature naming its Python dataclass, so the stub stays typed.

**Bindings marshal; logic lives in C++.** The binding unit converts, releases the
interpreter lock and forwards. The lease between workspace and session, the coupled-flow
guard and the march entry points move into the C++ library, where ctest reaches them. The
device refusals stay as they are, the CPU-only TDSE host source among them.

#### 5.4 Arrays without a batch axis

A trajectory's arrays have no leading axis of one anywhere:

- the state, the reference's passes and history, and native's load and readback:
  `(n_modes, n_pol, n_omega)`;
- the operator's value tables in `PropagationProblem` and the stage terms (kernel contract
  §3);
- the boundary functions of #618;
- the memory ledgers;
- the stored state of a `MarchSnapshot`.

Changing the snapshot's layout on disk is an output-schema change. It is made with the next
release's one schema change, and a snapshot written before it is refused by name, not
squeezed. As §3 notes, that release cannot resume 0.2.0's files in any case.
[conventions.md](../conventions.md) §6 drops `n_batch`.

#### 5.5 The timing probe

`Simulation.estimate` times one step for front ends; the service's estimate handler calls
it. SEP 46 calibration, which runs by default on a machine's first native run, times trials
at several positions across a run from the launch state. It repeats each one identically,
and compares the trials' fields between candidate thread settings. A session cannot take a
trial at an arbitrary position without propagating there, so the probe stays, on both
backends, with today's contract:

- `timing_probe(start)` on the backend, and `Workspace.probe(start)` on native, loads the
  start's field, noise, position and resume flag. Its march history, ledgers and monitor
  continuation are not read.
- It is refused while a session holds the workspace.
- A Runge–Kutta trial at `(z, dz)` commits nothing, and `field()` returns an owning copy of
  its end state.
- **The low-storage exception stays.** Its trial advances the discarded registers in place,
  `field()` refuses, and the workspace is closed afterwards.

### 6. Implementation strategies

#### 6.1 Layers and the seam record

The work stays within the backends layer (architecture.md §3), the driver's scan
executors, the native package, and translation's array layout. The seam has no version
number (ADR-068). Every change to a guarded surface updates its guard, with a release-note
fragment, in the phase that makes it:

- **The `PropagationProblem` value tables lose their unit axis** (§5.4). Their frozen
  descriptor tables in `tests/backends/test_driver.py` are rewritten in the commit that
  changes them. No field is added or removed, and no value changes.
- **The `RunRequest` and the stage bundles' scalars are unchanged.**
- **The `Backend`, engine and session protocols change** (§5.1, §5.2). The tests that pin
  their member lists change with them.
- **Native's record tables** are guarded by the schema test of §5.3 from the phase that adds
  them.

#### 6.2 Removing `Batch` and the groups

This is pure deletion, and comes first among the reference's changes because it removes a
user of `HostDriver`'s template methods:

- the executor and `compatible_groups`;
- the group service loop in `driver/_scan_batch.py`;
- `GroupSession`, `PreparedGroup` and `prepare_group`;
- `HostGroup`, `host/group_session.py` and `NumpyBatchBackend`;
- `select_batch` and `BatchBackend`;
- native's `NativeGroupSession`;
- the `group_batch` benchmark.

The scans guide and the API, model-selection, interactive, progress and waveguide pages
that mention `Batch` are updated. A `Removed` changelog fragment names the replacement
executors. The tests that exercise groups are deleted with them. A test that used a group
only to reach a member's behaviour is kept, run singly.

#### 6.3 The reference

**`HostDriver` is dissolved, not refactored in place:**

- The Runge–Kutta march becomes a march over `RungeKuttaEngine`, with its state in a march
  object, beside the Adams and low-storage marches. The low-storage march takes the same
  engine protocol, as on native (§5.2).
- The scheduling, halts, step planes, dense holds and observation, periodic rows, the
  monitor and edge checks and the noise reset move into `HostSession`. That makes the
  session the Python counterpart of `session.cpp`, which native's header already calls a
  transcription of these functions.
- `NumpyRefBackend` keeps lowering, `prepare`, `metadata`, `memory_ledger`, the probe and
  `close`.
- Its physics moves into `NumpyWorkspace`, together with the operations `HostHistory` and
  `HostLowStorage` call today, as named methods. That is roughly 2,000 lines: stage and
  source, finish, mask, statistics and the module tables.

The interleaved bookkeeping is pulled out of the physics, without moving any measurement:

- the split collar's sums are computed in the workspace's finish and accumulated by the
  session (§5.2);
- the quadrature alarm becomes a workspace reading that the session reports, as native's
  `quadrature_peak` already is;
- the TDSE reservation moves onto the engines.

The scripted test drivers become scripted engines under the real marches: today's
`HostScriptedDriver` and the trace harnesses' `_HostHarness`. That is how native's scripted
engines already test its marches. The unit axis leaves the passes, the history and the
translation functions in this phase (§5.4).

#### 6.4 The native seam

The adapter's configuration code collapses into building one `Problem`, then one run record
per run. These all go, with the flags that sequence them (`_configured`, `_rk_staged`,
`_rk_weighted`):

- the lazy `_configure` calls;
- the post-construction uploads (`table_upload`, `adams_stage_tables`);
- the monitor binds (`bind_monitor`, `bind_occupancy`, `bind_edge`, `bind_edge_lines`).

**The probe** becomes native (§5.5). It replaces:

- the adapter's per-step trial path (`_trial_step`, `_prepare_trial`, `_score_trial`);
- `_ResidentLowStorage.step`;
- `_ZTables`' per-position evaluation, and the bindings behind them.

They are removed only once the probe tests below pass on the replacement:

- repeated trials;
- the owning readback;
- the four-evaluation low-storage timing;
- the low-storage refusal of `field()`.

**The Adams engine's adapter** shrinks to its construction and resume. Its per-step methods
are dead and go, with their bindings.

**One `diagnostics()` record** replaces the 12 metadata properties, the memory ledger, the
execution sizing, the planning record and the crossed bytes. It is what
`NativeBackend.metadata` already assembles.

#### 6.5 Tests

The tests that call native directly are kept as tests. What changes is the surface they
call. The binding's test-only names fall into seven groups, each with a destination:

| Group | Today | Destination |
|---|---|---|
| Decision traces | `ScriptedMarch`, `ScriptedRkMarch`, `adams_march_run`, `rk_march_run` | `_testing`, running native's marches over its scripted engines; the Python harness drives both backends' marches over scripted engines |
| Fault injection | `adams_march_fault` (7 session and resume test files) | `_testing.inject_fault(workspace, ...)` |
| Owners | `Monitor`, `TransverseEdge`, `watch_check`, `watch_due`, `watch_split_limit` | `_testing`, unchanged in behaviour |
| Engine-level parity | `trial_step`, `rk_trial`, `stage_to_host`, `projected_to_host`, `weighted_error`, the flow actions, `adams_exponential_trial`, the stage scalars, `phase_sum_to_host` | `_testing.engines(workspace)`: native's engines through the Python protocols of §5.2, plus a read-back of the field and a stage register. A parity case is written once and run against both backends' engines |
| State-level statistics | `statistics`, `spatial_statistics`, `planned_statistics`, `aperture_statistics`, `plane_fluence_trace`, `axis_spectrum`, `monitor_reading`, `edge_reading` | One `_testing.evaluate(workspace, plan, z)` over the session's own statistics plan, or tests through a session's launch observation |
| Helpers | `vector_*`, `basis_rows`, `trial_tables`, `half_width`, the width functions, `strict_witness` | `_testing`; `clenshaw_points` stays in production, where the statistics lowering uses it |
| Runtime and accounting | `crossed_bytes`, wisdom import and forget, `reference_fft3_pairs`, the scoped OpenMP probes | `crossed_bytes` joins `diagnostics()`; the rest go to `_testing` |

`_testing` is built into the same extension and shipped in the wheel. It promises nothing,
and its users are solprop's own tests. The installed-wheel smoke test
(`native/ci/smoke.py`) imports it and runs one of its checks.

On the reference:

- about 39 test files call `propagate_to` directly, and 46 call private physics methods;
- the scripted and trace harnesses subclass `HostDriver`;
- about 31 files exercise groups.

These move to sessions, to workspaces and engines, or to scripted engines, or are deleted
with the groups. A test whose only purpose was the old structure is deleted, and its phase
report says why. The trace corpora keep their decisions; a corpus is regenerated only under
§4's rule.

#### 6.6 Choices of tool

**nanobind stays.**

- It already builds the extension against Python's stable ABI as one `cp314-abi3` wheel per
  platform, tested on 3.14 and 3.15.
- It is the smallest and fastest-compiling C++ binding library that also handles NumPy
  arrays with explicit ownership.
- It releases and reacquires the interpreter lock for the host source.
- Its `stubgen` can generate the whole stub.

The binding's size comes from the interface's shape, not the library.

- **pybind11** would be larger and slower, for nothing here.
- **Cython** fits Kokkos's C++20 templates poorly and adds a language.
- **A plain C interface called through ctypes or cffi** would make the extension independent
  of Python's ABI, and usable from another language. It would cost the typed exceptions,
  array ownership and lock handling that nanobind gives, and move marshalling into Python.
  The interface proposed here, one record in and a session out, would let a C interface be
  added later if another front end ever needs one.

The release build should pin nanobind's version. Today it takes whatever newest release
satisfies `>=2.9` at build time.

**Field tables, not code generation or reflection.** Generating both languages' records
from a schema file would add a build step and a third place for every field. C++26
reflection would remove even the tables, but nvcc, HIP Clang and Apple Clang will not offer
it for years. When they do, the tables can be replaced without touching the Python side.

#### 6.7 Cost

Rough estimates, to be compared with the measured counts at Final:

| Area | Today | Estimate after |
|---|---|---|
| Native binding units (C++) | about 5,000 lines | 1,500 to 2,000, plus about 400 for the converter with its array specifications |
| `solprop_native` (Python) | 14,154 lines | 10,000 to 11,000, `group.py` included |
| `solprop.backends` and the group parts of `solprop.driver` (Python) | about 25,600 lines | 19,000 to 20,500 |
| Hand-written `_core.pyi` | about 700 lines | none |
| Tests | — | churn in about 150 files; the net count is not estimated |

`backends` and the driver lose:

- most of `HostDriver`;
- the protocol surface;
- the duplicated bookkeeping;
- the groups and the batch executor, about 3,000 lines.

They gain the Runge–Kutta march object and a larger session. The axis removal changes many
lines and removes few. No dependency is added.

Native's per-step run time is not expected to change, because the marches are already
native. Preparation may get slightly cheaper with fewer calls and copies, which matters only
for scans of small cases.

### 7. Verification and validation

**A pinned corpus first.** Before phase 3 changes anything, a corpus is recorded from the
current tree on fixed schedules. It covers every march (adaptive and replayed Runge–Kutta,
fused and exponential Adams with and without retained-history transport and carrier dense
output, low-storage) and every route family, with:

- the saved fields;
- the boundary-monitor and edge records, the split collar's removal and occupancies
  included;
- the resume outcomes: a snapshot, a resume and its continuation;
- the probe's trials, readbacks and low-storage refusal.

The restructured reference is compared with it:

- bitwise where the order of operations is unchanged, as a diagnostic on one build and
  thread setting only, never as a cross-build or cross-thread gate (SEP 25);
- otherwise within each computation's SEP 25 budget;
- anything beyond that is handled by §4's rule.

**Numbers:**

- **Session suites.** Both backends' session suites pass with their cases and bounds
  unchanged: lifecycle, resume, snapshot, the boundary monitor and the edge, and every
  route's propagation tests.
- **Matched-discrete parity.** Native against the reference keeps its bounds.
- **Decision traces** hold both marches to the same decisions. A regenerated corpus is
  listed with its cause.
- **Adaptive runs** keep their accuracy envelopes in the existing qualification cases.
- **The engine contracts** are qualified on the cases where they are consequential:
  - interaction Adams with retained history;
  - exponential Adams with a carrier frame and interpolated planes;
  - a low-storage failure during the trial, and during the finish, each ending the run
    failed with no checkpoint at that position;
  - the split collar's removal and occupancies over several accepted steps between checks,
    with rejected trials, dense observations and a resume between them, with the deep-collar
    residue test unchanged.

**Tests are not lost.** Each phase report carries an inventory of every removed binding,
hook, protocol member or group path. It lists the tests that used it, and each test's
destination: ported, or deleted with a reason. The coverage measured by the existing lanes
does not fall except for deleted code: native's C++ line coverage and solprop's Python
coverage of `backends/` and `driver/`. The native C++ tests gain the logic moved out of the
binding unit.

**Records and arrays:**

- The schema test of §5.3 fails when a Python record and its C++ table disagree.
- Each record round-trips through the converter.
- The malformed-layout refusals of today keep passing: transposed tables, a wrong leading
  axis and non-contiguous arrays.
- So do the lifetime tests of returned arrays after acknowledgment, reuse and close.
- The generated stub is checked by the native gate.

**Builds and lanes:**

- CI's lanes run as they do today: the CPU builds and tests on Linux, macOS and Windows; the
  coverage lanes; the native quality stages; and the compilation, with their tests, of the
  CUDA and HIP targets.
- The installed-wheel smoke test imports `_testing`.
- No device run is part of this SEP's qualification. The device builds are qualified by
  compilation only, and the phase report records that limit. A later device campaign runs
  the GPU lane on the new seam.

**Performance:**

- **Native step time** is measured with `python -m benchmarks ab` on about ten steps or a
  short `--length` window, on a few cases covering each march, before and after phase 4. The
  hardware, thread settings and repetitions are recorded. A median slowdown above 3 % on a
  case, beyond its repeat scatter, is investigated and fixed before the phase merges.
- **Native preparation time** is measured on a small-case scan.
- **The probe's time and peak memory** are measured before and after.
- **The reference's test-suite wall time** should not rise noticeably.

**Size:** `poe size` and the binding units' line counts are measured before and after each
phase. The copy counts and peak host storage of a workspace's construction are measured
before any reduction is claimed.

### 8. Documentation

**User manual:**

- The [scans](../../guide/workflows/scans.md) page loses its fixed-schedule batch section.
- The API, model-selection, interactive, progress and waveguide pages that mention `Batch`
  are updated.
- A `Removed` changelog entry names the replacement executors.

**Developer pages:**

- [backends.md](../backends.md) §1, §3 and the table of marches: one shape, two
  implementations, and no groups.
- [architecture.md](../architecture.md) §9.
- [kernel-contract.md](../kernel-contract.md) §3, the value tables' axes; §8; and §§11–12,
  for the hooks and names.
- [conventions.md](../conventions.md) §6, which drops `n_batch`.
- [scans.md](../scans.md) and [benchmarking.md](../benchmarking.md).
- `native/README.md`: the binding's rules, namely records by field table, bindings that only
  marshal, `_testing` that promises nothing, and generated stubs.
- [testing.md](../testing.md): the engine-level parity pattern and the scripted engines.

An ADR records the shared shape, the removal of groups and the native seam's rules.

### 9. Compatibility and impact

- **`Batch` is removed.** A script that imports it fails at import, and the changelog names
  `Local` and `Pool` as replacements. On native, `Local` holds one backend where `Batch`
  held one per member. `Pool` with as many workers holds about what `Batch` did, and runs
  its points concurrently.
- **The output schema** changes once, for the snapshot's layout (§5.4), in the next release's
  one schema change. A snapshot written before is refused by name. Saved fields and records
  are otherwise unchanged.
- **`solprop.backends`** changes its private and protocol surface, which ADR-058 leaves
  unstable.
- **The `PropagationProblem` value tables** lose their unit axis. Their values and every
  other field are unchanged. The frozen tables are rewritten under ADR-068's process.
- **Reference numbers** may move under §4's rule, and each move is recorded.
- **No deliberate difference from Luna.jl** is involved.

### 10. Alternatives considered

- **Shrink the native seam alone.** This is the smaller change, and phase 0 does part of it
  anyway. It leaves two structures for one design. It also leaves the reference unable to
  share the engine-level parity tests, or to serve as a line-by-line source for native's
  session.
- **Keep `Batch` and retire only the reference's fused group.** This was the draft's
  recommendation. It keeps an executor that, on native, costs `size` times `Local`'s memory
  for no speed. It also keeps the group protocols, the group service loop and native's group
  session, and the unit axis they justify.
- **Keep the private unit axis, or keep it on disk only.** It no longer serves anything once
  groups go. Keeping it on disk would preserve no resumability, because the next release
  cannot resume 0.2.0's files in any case (§3).
- **Slim `numpy_ref` to a test library, with no session or step loop** (SEP 49 §10.1). This
  was deferred on 2026-11-12 to keep independent complete runs for development, debugging
  and verification. This SEP keeps them, and gives them the same structure as native's.
- **Let the reference run Python engines under native's session.** This would make the
  structures identical by construction. It breaks SEP 47 §6.6: the reference must not call
  native's controller to establish the agreement it is compared on.
- **Drive native's engines from Python marches.** This is the per-step seam SEP 47 removed.
  It costs a crossing per trial and a second implementation of every march's bookkeeping.
- **Time estimates and calibration through a short session, with no probe.** A session cannot
  take trials at arbitrary positions without propagating there, nor repeat one trial
  identically, both of which calibration needs (§5.5).
- **Keep dictionaries, with a schema test only.** This removes drift but not the ~21
  converters or the hand-built dictionaries.
- **Typed nanobind classes for every record, built by Python.** This is what the 2026-11-11
  cleanup plan proposed. Stubs then type every record for free. But the reference needs the
  same records without native installed, so they would exist twice. The Python dataclass
  stays the one definition, and the field table only describes it on the C++ side.
  `Problem`, which only native reads, stays a bound C++ class.
- **A C interface or another binding library:** §6.6.

### 11. Phases

Everything lands on one branch and in one pull request (SEP 0). Each phase updates the
guards and release notes of what it changes (§6.1).

**Phase 0 — native deletions and the test module.**

- Delete the bindings that nothing reaches, and the dead per-step adapter code (the
  `_ResidentAdams` trials, accepts and observations). The probe's paths stay until phase 4.
- Move the test-only bindings into `_testing` and repoint their tests.
- Add `_testing` to the wheel smoke test.
- Fix the stale docstrings that cite `_ResidentAdams` and `_ResidentLowStorage`.
- Pin nanobind in the release build.
- Verify: the coverage rule, and the inventory of §7.

**Phase 1 — remove `Batch` and the groups** (§6.2), with its documentation and changelog.

- Verify: the coverage rule, the inventory, and the single-run suites unchanged.

**Phase 2 — shared records, engine protocols and the corpus.**

- The run records in `solprop.backends`.
- The `RungeKuttaEngine` protocol.
- `AdamsEngine` aligned with C++.
- Native's adapter builds its dictionaries from the records.
- The corpus of §7 recorded.
- Verify: unchanged results on both backends, bitwise on one build as a diagnostic.

**Phase 3 — the reference restructured, and the axis removed.**

- `NumpyWorkspace`, the engines, the marches and the session's scheduling.
- `HostDriver` deleted.
- Boundary warnings as events.
- The split collar's rules pinned, then moved.
- The unit axis removed from translation, the boundary functions, the reference and native's
  adapter, with the frozen tables rewritten and the snapshot layout changed.
- The tests ported.
- This phase may be split: the axis first, then Runge–Kutta, then Adams and low-storage.
- Verify: the corpus comparison, the session suites, the traces, the engine-contract cases,
  and each numerical change under §4.

**Phase 4 — the native seam.**

- The field-table converter and `schemas()`.
- `Problem` absorbs the set-once data.
- `Workspace`, `start(run)`, `probe(start)` and `Session` as in §5.3.
- The old probe paths removed once the probe tests pass.
- `diagnostics()`.
- The logic moved out of the binding unit.
- The generated stub.
- The engine-level parity tests run against both backends.
- Verify: the session suites, parity, the record tests, the coverage rule, the compilation of
  the device targets, the step-time A/B, the probe's cost and the size counts.

**Phase 5 — documentation and the ADR.**

- The developer pages of §8 not already updated with their phases.
- The ADR.
- #110, #618, #171, #828, #344, #107 and #356 updated or closed.

### 12. Open questions

- **Q1.** Settled by the maintainer on 2026-11-15: `Batch` and every group path are removed
  (§6.2).
- **Q2.** Settled on 2026-11-15: `_testing` is built into the extension and shipped in the
  wheel (§6.5).
- **Q3.** Which of today's binding-level oracle tests to replace with C++ tests or
  session-level tests, not port? This is decided per test in phases 0 and 4. The default is
  to port.
- **Q4.** Settled on 2026-11-15: the probe stays, as `probe(start)` with today's low-storage
  exception (§5.5).
- **Q5.** Which reference simplifications to take that move numbers? Each is decided in
  phase 3's report under §4. The engine holding its trial is the expected first.

### 13. References

1. W. Jakob, "nanobind: tiny and efficient C++/Python bindings," documentation,
   https://nanobind.readthedocs.io (accessed 2026-11-15).
2. M. von Löwis, "PEP 384 — Defining a Stable ABI," Python Enhancement Proposals (2009).
   https://peps.python.org/pep-0384/
3. W. Childers, P. Dimov, D. Katz, B. Revzin, A. Sutton, F. Vali and D. Vandevoorde,
   "Reflection for C++26," ISO/IEC JTC1/SC22/WG21 paper P2996 (2025).

## Part II — Implementation record

## Part III — History

- 2026-11-15: Drafted, from a review of the native binding's use in production and of the
  reference's structure at `main` 07218ed3. Tracking issue:
  [#849](https://github.com/LupoLab/solprop/issues/849).
- 2026-11-15: AI review by GPT-6 at `9a148c340843769474fff390943cdcbaca3e6f2b`,
  recorded below. Only the review metadata and Part III changed; Part I is unchanged.
- 2026-11-15: The maintainer's decisions on the review's questions, and Part I revised in
  response to every point, by Opus 5.5 (drafting agent); both recorded below. The revision
  removes the `Batch` executor and the unit batch axis, keeps the timing probe, and
  completes the protocol, engine, converter and qualification contracts.

### AI review — 2026-11-15

**Reviewer:** GPT-6 (AI). Reviewed all of Part I on
[PR #850](https://github.com/LupoLab/solprop/pull/850), at
[`9a148c34`](https://github.com/LupoLab/solprop/commit/9a148c340843769474fff390943cdcbaca3e6f2b),
against the current backend, session, monitor, native binding and test code, SEP 0,
SEP 47 and ADR-068. Read the tracking issue #849 and the six issues listed in §3.
The two existing automated PR findings are acknowledged in R2 and R3 below.

**Recommendation:** revise before acceptance. The comments below distinguish concrete
conflicts with existing behavior from interface questions that the proposal needs to
settle. No proposal text, acceptance decision or response to this review is supplied here.

**Evidence:** source inspection, plus the existing NumPy-reference tests
`test_a_deep_collar_reads_no_removal_from_residue`,
`test_a_timing_probe_commits_nothing_and_reads_back_its_trial` and
`test_a_low_storage_probe_has_no_trial_field`: three passed, their three native cases
skipped because native was not installed. These checks establish the current reference
behavior only. No native build, device run, performance measurement or proposed
implementation was tested.

1. **R1 — High: preserve the split collar's measurement point (§§4, 5.1–5.2, 6.2).**
   Moving the split sums out of `_finish` into a session reading needs a more precise
   contract than a reading of the committed state. In
   `src/solprop/backends/numpy_ref/backend.py::NumpyRefBackend._finish`, the sums are
   taken after the first hard mask but **before** the temporal split and its final
   mask. `src/solprop/monitor.py::BoundaryMonitor.split_sums` computes the removed
   power as `P * (1 - exp(-alpha * h))` and the guard occupancies from
   `P * exp(-alpha * h / 2)`, with `P` the power before the split. Applying that
   calculation to the damped field changes the removal; reconstructing the earlier
   field from the final masked field can amplify mask residue into fictitious loss.
   The existing deep-collar regression in `tests/test_monitor.py` specifically guards
   the latter failure. Native also gathers the sums during its split operation in
   `native/src/general/trial.cpp`.

   **Requested change:** specify that the workspace captures the required small
   pre-split reductions during each accepted finish, and that the session owns their
   accumulation and reporting. State when these values are drained, including several
   accepted steps between checks, rejected trials, dense observations and resume.
   Preserve the removal and guard-occupancy tests. Moving ownership must not move the
   measurement to a different physical state under the general permission to simplify
   reference arithmetic in §4.

2. **R2 — Medium: independent sessions do not by themselves fix #171
   (§§2–3, 6.2, 7–9; Q1).** Native already has independent member sessions, but
   `native/src/solprop_native/group.py::NativeGroupSession._turn` makes a
   non-boundary terminal failure a group-wide error. The scan consumer
   `src/solprop/driver/_scan_batch.py::_consume` raises `batch.error`. This preserves
   SEP 47 §6.7, which explicitly distinguishes a member's boundary stop from a
   non-finite norm, step floor, trial exception or in-flight failure that stops the
   group. Replacing `HostGroup` with the native arrangement therefore leaves #171's
   failure policy intact. This also conflicts with leaving the event contract out of
   scope and requiring all existing session cases to remain unchanged. The
   [automated review](https://github.com/LupoLab/solprop/pull/850#discussion_r4580456603)
   identified the same conflict.

   **Requested decision:** either leave #171 open and remove the claim that Q1 fixes
   it, or explicitly include a new failure policy for both backends and the scan
   consumer. In the latter case, distinguish recoverable member failures from shared
   executor failures, retain each member's actual commit and checkpoint eligibility,
   add the `['done', 'failed', 'done']` regression requested by #171, and update the
   user guide's batch section. Then §8 cannot promise no manual changes.

3. **R3 — Medium: use the current seam-change process (§§6.1, 9, 11).**
   `KERNEL_SCHEMA_VERSION` no longer exists. ADR-068 and
   [kernel-contract.md](../kernel-contract.md) explicitly removed the counter and
   require the affected frozen surface tests and a release-note fragment to change
   with the seam. The proposed bump cannot be performed as written. The
   [automated review](https://github.com/LupoLab/solprop/pull/850#discussion_r4580456601)
   also reports this.

   **Requested change:** replace all three bump instructions with the current
   process. Distinguish unchanged scientific descriptor fields from changed protocol
   and engine surfaces; preserve the former's frozen cases and update the latter's
   guards in the phase that changes them, rather than postponing the seam record to
   phase 4.

4. **R4 — Medium: retain or replace the live backend facts omitted from the new
   surface (§5.1).** The stated list of what remains on `Backend` omits `request`,
   `name` and `reads_transverse_edge`. These have production consumers outside the
   host driver: `driver/run.py::run_problem` and `driver/_scan_run.py::launch_statistics`
   read `backend.request`, and `driver/session_loop.py::_refuse_unread_collar` reads
   the capability and name. Removing `HostStepping` does not remove these callers.

   **Requested change:** make the proposed protocol inventory complete, including
   its properties and attributes. Retain these facts or name their replacement and
   the caller migration. In particular, keep the requested transverse-collar refusal
   before the first step, and explain how a reused backend exposes the request its
   buffers and plans were prepared for.

5. **R5 — Medium: complete the engine and commit contracts (§§5.1–5.2, 6.2, 6.4).**
   The sketches are described as one-to-one C++ interfaces, but omit consequential
   parts of the current ones. `march::Engine::accept` in
   `native/include/solprop_native/adams_march.hpp` also receives the retained history
   entries. `ResidentEngine::accept` uses their rows, oldest first, to transport the
   interaction-formulation history. `step_carrier(order)` supplies the last trial's
   rotations for dense output. Neither transfer appears in the proposed protocol or
   in the description of what moves to the session.

   Low-storage native execution is already an `rk::Engine`, with `trial`, `accept`,
   `reading` and `in_flight`, rather than just `step`. Its trial overwrites the state;
   an exception or non-finite result before finish leaves a poisoned terminal with no
   current checkpoint. The reference currently records that state in
   `HostLowStorageMarch.replay_step`, on the driver being deleted. The general claim
   that an engine keeps an uncommitted trial is not true for this method.

   **Requested change:** define the retained-history and carrier transfers, whether
   through explicit arguments or a documented derivation from the trial record.
   Decide whether low-storage also uses the Runge–Kutta engine protocol on Python,
   or state its separate commit and failure contract. Name the owner of the in-flight
   flag, the finish, the committed position and the stiffness diagnostic, and the
   session's means of reading their result without private-field access. Include
   interaction Adams with retained history, carrier dense output, and low-storage
   failure during trial and finish in the engine-contract qualification.

6. **R6 — Medium: specify how the new probe loads a run, and preserve the
   low-storage exception (§§5.3, 6.3, 11; Q4).** The proposed `Workspace.probe()`
   has no run or start argument, while the initial field and noise are assigned to
   `start(run)`. Where does a probe on a freshly built workspace obtain its launch,
   resume position, schedule cursor and noise policy without opening a session?
   Today `NativeBackend.timing_probe(start)` explicitly loads these through `_load`.

   Also, the blanket promise of an uncommitted trial and field readback does not
   describe low-storage probing. `TimingProbe` in `backends/protocol.py` explicitly
   permits its discarded registers to advance in place, requires that backend to be
   closed afterwards, and refuses `field()`. The corresponding test in
   `tests/driver/test_estimate.py` passed on the reference during this review.

   **Requested change:** give the probe an explicit load path and an ownership rule
   excluding a live session. Preserve the low-storage exception, or explicitly
   propose and account for the additional storage or reload needed to change it.
   Keep repeated-trial, owning-readback and four-evaluation low-storage timing tests,
   and make removal of the active `_ResidentLowStorage.step` and `_trial_step` probe
   paths depend on their replacement; phase 0's deletions must not remove them early.

7. **R7 — Medium: the generic converter needs an array contract beyond names and
   C++ member types (§§5.3, 6.3–6.6, 7).** A name and member pointer identify a
   `std::vector<double>`, but not the rank and related extents of the array it
   represents. The present `Fields<>` already carries extra rank metadata and keeps
   shapes for validation. For example, `native/tests/test_backend.py` refuses
   transposed Raman tables with the correct element count and fixed-term tables
   whose leading axis is wrong. Removing redundant counts is compatible with those
   checks only if the proposal names the authoritative shapes and the relationships
   still checked before allocation. A schema test comparing names and types alone
   will not detect a loss of those checks.

   The promised reduction from three copies to one also needs an ownership decision.
   The present `exact_vector` copies array data into a record; changing how records
   are described does not by itself remove that copy or the later device upload.
   Returned `Batch` and `Snapshot` arrays must remain valid after acknowledgment,
   workspace reuse and close. Nanobind's
   [array ownership rules](https://nanobind.readthedocs.io/en/latest/ndarray.html#data-ownership)
   require an explicit owner for a view; they do not make session scratch into an
   owning snapshot.

   **Requested change:** specify array dtype, rank, contiguity, read-only admission,
   related-shape validation and size-overflow checks, and say which inputs are borrowed
   during conversion versus copied into persistent storage. Specify owning returned
   arrays and the GIL/lifetime boundary for any borrowed Python memory. Keep the
   malformed-layout and lifetime regressions, add record round-trip checks, and
   measure copy counts and peak host/device storage before making the one-copy claim
   an acceptance criterion. Include how the generated stubs retain useful record
   and array types when the binding accepts generic dataclasses or mappings.

8. **R8 — Medium: clarify whether private array layouts change (§§2–3, 6.1–6.2,
   6.6, 9).** §3 says the single-trajectory batch axis leaves the backends, but the
   concrete scope and phases mainly remove `max_batch` and the fused reference
   group. Those are different changes. `NumpyRefBackend`, its pass functions and
   native's load/readback adapters currently use a private leading axis of one;
   [conventions.md §6](../conventions.md#6-array-layout-and-axis-names) permits that.
   SEP 47 §6.7 separately requires the leading axis in existing `MarchSnapshot`
   payloads to remain readable.

   **Requested decision:** retain the private layouts and narrow the claim, or
   explicitly include their conversion, saying which Python arrays and C++ layouts
   change. In the latter case, account for the pass, transform, test and memory-ledger
   work in the phases and cost estimate, and preserve the snapshot serialization
   boundary with refusal of a non-unit stored axis. Removing a protocol attribute
   alone does not settle this scope.

9. **R9 — Medium: make qualification match the reach of the binding change
   (§§6.4–6.6, 7, 11).** The new records, moved bindings and generated stubs affect
   all native builds, yet §7 does not distinguish CPU, CUDA and HIP qualification or
   say which wheel tests exercise the shipped `_testing` module. Passing a CPU
   session suite does not check device uploads, resident histories or event/export
   lifetimes after their binding changes. The existing device refusals, including
   the CPU-only TDSE host source, must also survive.

   **Requested change:** name the required build and execution lanes, the device
   witnesses or explicitly recorded limits, and the installed-wheel import and
   `_testing` checks. Pin the pre-change comparison corpus before restructuring it,
   including representative monitor records, resume outcomes and probe behavior as
   well as fields. Clarify phase 1's bitwise requirement: it can be a diagnostic for
   an unchanged computation in a controlled configuration, but should not become a
   cross-build or cross-thread numerical gate contrary to SEP 25. Specify that the
   performance A/B uses about ten steps or a short length window, with hardware,
   thread settings, repetitions and a regression decision rule recorded; “a few
   release cases” must not select full-length release tracking runs. Include the
   native probe's time and memory cost, since it is being replaced even though
   ordinary native steps are expected to be unchanged.

### Maintainer decisions — 2026-11-15

Decided by John Travers on questions the review raised, each put to him with its context
and the drafting agent's recommendation.

- **D1 (R2, Q1): remove `Batch` and every group path.** Native's group builds a backend for
  every member and fuses nothing, so the executor costs `size` times `Local`'s memory for no
  speed. The reference's fused stage sums serve only the reference's speed, which is not a
  goal. Nothing in the GUI, the recipes or the examples uses it. #171 and #828 close as moot.
- **D2 (R8): remove the unit batch axis everywhere, on disk too.** This covers translation's
  value tables, the boundary functions of #618, both backends, the memory ledgers and the
  stored snapshot. The snapshot's layout changes with the next release's one output-schema
  change. That release cannot resume 0.2.0's files in any case, because of the schema change
  already on `main`.
- **D3 (R6, Q4): keep the timing probe as `probe(start)`, with today's low-storage
  exception.** SEP 46 calibration, which runs by default on a machine's first native run,
  needs trials at arbitrary positions from the launch state, repeated identically, with
  their fields compared. A session cannot provide that.
- **D4 (R7, §10): run records stay Python dataclasses, described in C++ by field tables.**
  This departs from the 2026-11-11 cleanup plan, which bound the C++ structs as nanobind
  classes, because the reference needs the same records without native installed.
  `Problem` stays a bound C++ class.
- **D5 (R9): device builds are qualified by compilation only.** This SEP makes no device run.
  Its phase report records the limit, and a later device campaign runs the GPU lane.
- **D6 (Q2): `_testing` ships in the wheel,** built into the same extension, so the
  installed-wheel jobs run the native oracle tests against what users install.

### Response to review — 2026-11-15 — Opus 5.5

Every point was checked against the code at `main` 281a9543 before answering. The review is
right on each; R2, R6 and R8 also led to the maintainer's decisions above, which go further
than the review asked.

- **R1 (split collar's measurement point): accepted.** `NumpyRefBackend._finish` takes
  `split_sums` on the masked field before the temporal split. Native gathers the same sums
  during its split, and both zero the accumulator when a monitor is bound and drain it at a
  reading. The changes:
  - §4 now states that a move of ownership never moves a measurement, with the formulae and
    the deep-collar test.
  - §5.2 has the workspace's finish compute the pre-split sums and return them in its
    `AcceptRecord`. It states the session's accumulation and drain rules for several
    accepted steps between checks, rejected trials, dense observations, probe trials, loads
    and resumes, as transcriptions of today's behaviour.
  - §6.3 and §7 pin those rules with tests before ownership moves, and keep the removal,
    occupancy and residue tests.
- **R2 (independent sessions do not fix #171): accepted, and superseded by D1.** The draft's
  claim was wrong: native's group ends the group on a non-boundary member failure, as SEP 47
  §6.7 requires. Rather than fix the policy, the maintainer removed the executor. The
  changes:
  - §§2, 3, 5, 6.2, 8, 9 and 11 now remove `Batch` and every group path.
  - #171 and #828 close as moot, and #356 is told it would need its own group interface.
  - §8 now lists the manual pages that change.
- **R3 (the seam has no version number): accepted.** `KERNEL_SCHEMA_VERSION` was removed by
  ADR-068, and the draft's three bumps are gone. §6.1 now follows ADR-068's process:
  - the value tables' axis change rewrites `test_driver.py`'s frozen tables in the same
    commit;
  - `RunRequest` and the stage bundles' scalars are unchanged;
  - the protocol, engine and record guards change in the phase that changes them, each with a
    release-note fragment, not in a final documentation phase.
- **R4 (live backend facts): accepted.** §5.1 now lists the whole `Backend` protocol with its
  attributes and consumers. It keeps:
  - `name`;
  - `reads_transverse_edge`, for the refusal before the first step;
  - `request`, which a reused backend's runs are checked against;
  - the `helpers` class attribute.
- **R5 (engine and commit contracts): accepted.**
  - §5.2's Adams `accept` takes the retained history entries, oldest first.
  - `step_carrier(order)` is in the protocol, and the carrier rotations' path to the
    session's dense observation is stated.
  - The low-storage step uses the Runge–Kutta engine protocol on both backends, as it already
    does on native. Its destructive trial, the in-flight interval and the failure with no
    checkpoint are documented as one stated difference.
  - The in-flight flag and the finish are the engine's; the committed position is the
    march's; the stiffness diagnostic travels in the `AcceptRecord`.
  - §7 qualifies interaction Adams with retained history, carrier dense output, and
    low-storage failure during the trial and the finish.
- **R6 (the probe's load path): accepted, with D3.** §5.5 gives the probe an explicit
  `probe(start)` load, the same as `timing_probe(start)` today. It is refused while a session
  holds the workspace, and keeps the low-storage exception. §6.4 removes the old probe paths
  only after the repeated-trial, owning-readback, four-evaluation low-storage and refusal
  tests pass on the replacement. Phase 0 no longer deletes them.
- **R7 (the converter's array contract): accepted.** In §5.3 a field table's array entry
  carries its dtype, rank and extent relations. The converter refuses, before any
  allocation and naming the field:
  - a wrong dtype or rank;
  - a non-contiguous or overlapping array;
  - extents that disagree with the relation;
  - a size whose product overflows.

  So the transposed-table refusals survive the removal of redundant counts. Inputs are
  borrowed under the interpreter lock and copied into the record; read-only arrays are
  admitted; returned arrays own their storage through capsules. The one-copy claim is
  withdrawn as a criterion and replaced by a measurement (§7). The schema test now covers
  dtypes, ranks and extent relations, records round-trip, and the stubs name the dataclass
  types through signatures.
- **R8 (private array layouts): accepted, and widened by D2.** The draft's claim and its
  phases disagreed. The axis is now removed everywhere (§5.4), with the frozen tables, the
  passes, the adapters, the ledgers and the snapshot layout changed in phase 3. The
  snapshot's on-disk change is an output-schema change, and an older snapshot is refused by
  name, never squeezed. §6.7 notes that the axis removal touches many lines and removes few.
- **R9 (qualification): accepted, with D5 for devices.** §7 now:
  - names CI's lanes;
  - adds `_testing` to the installed-wheel smoke test;
  - records that the device builds are qualified by compilation only, as a stated limit;
  - pins a corpus (fields, monitor and edge records, resume outcomes and probe behaviour)
    before phase 3;
  - keeps phase 2's bitwise comparison as a single-build diagnostic, not a gate;
  - specifies the step-time A/B as about ten steps or a short window, with hardware, threads,
    repetitions and a 3 % median rule recorded;
  - adds the probe's time and memory.
- **The automated findings on PR #850:** the schema-version bump is answered by R3, the
  failure policy by R2, the pre-split measurement by R1, the probe's load path by R6, and the
  carrier transfer by R5.
