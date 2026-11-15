# SEP 52 — One backend shape and a small native seam

| | |
|---|---|
| **SEP** | 52 |
| **Title** | One backend shape and a small native seam |
| **Author** | John Travers |
| **Drafting agent** | Opus 5.5 (2026-11-15) |
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
`HostSession` steering it through halt hooks. This proposal splits both backends into the
same five parts:

- the backend's own lowering;
- a **workspace** that holds the state and does every state-sized operation;
- **engines**, the operations a march asks of the workspace;
- **marches**, which own their controller state;
- a **session** that owns the schedule, the boundary monitor and the events.

Both backends read the same run records, built once by shared code. `HostDriver` is
deleted. The native binding shrinks from about 220 bound names, about 400 configuration
names crossing in 15 to 20 calls, and about 21 hand-written dictionary converters, to:

- a `Workspace` built from one `Problem`;
- a `start` that takes one run record;
- a `Session`;
- one generic record converter.

Test-only entry points move to a separate private `_testing` module that exposes native's
engines through the same Python protocols as the reference's. Success is judged by four
things:

- **Numbers:** the session suites pass with their cases and bounds unchanged.
- **Simplifications:** each change to the reference's numbers is measured and accepted
  case by case.
- **Coverage:** native's line coverage does not fall except for deleted code.
- **Size:** the measured line counts fall by roughly the estimate of §6.6.

### 2. Motivation and scope

**Why now.** The internal seams were shaped by a history that no longer applies:

- several Python backends that shared `HostDriver`;
- a native backend that subclassed it;
- a Python loop that drove native one step at a time.

SEP 43 retired `numba_cpu`. ADR-077 fixed the backends at two. SEP 47 moved every native run
into its session. What remains is a reference built for a family of subclasses that has
one member, and a native binding that still carries the per-step interface the session
replaced. solprop's backends are unstable in 0.x (ADR-058), and `solprop-native` is pinned
to solprop's own version. A restructure now costs no user anything. Once others contribute
backend features, the same restructure costs every open branch.

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
`HostSession` to read its driver (`driver/session_loop.py`). The lockstep batch
(`HostGroup`, `NumpyBatchBackend`) is built on `HostDriver`'s template methods. It fuses
only the Runge–Kutta stage sums of fixed-schedule members, and native has no counterpart:
its group runs one session per member.

Native's C++ already has the shape this proposal asks of both. `session::Resident` is the
workspace. `rk::Engine` and `march::Engine` are the engines. `adams_march.cpp` and
`rk_march.cpp` are the marches, and `session.cpp` is the session. The reference has half
of it in Python: the `AdamsEngine` and `LowStorageEngine` protocols.

**In scope:**

- Both backends' internal structure: workspace, engines, marches and session, with the
  shared run records.
- The deletion of `HostDriver`, the `HostStepping` and `BatchBackend` protocols, and
  `Backend.max_batch`.
- The native binding: what crosses, how records are converted, the stubs, the test-only
  module and where binding-side logic lives.
- The timing probe's native form.
- The lockstep batch's future on the reference (§12, Q1).
- Porting or deleting every test that depends on what is removed, with the coverage rule
  of §7.

**Out of scope:**

- The public interface: `Simulation`, scans, recipes and output.
- The kernel contract's numerical content: `PropagationProblem`, the route union and the
  response descriptors.
- `RunRequest` and the `RunSession` and event contract of SEP 47.
- #110's split of the integrator from its tolerance, and its per-route pipelines.
- A batch axis through native's stage passes (#356).
- New features and performance work.
- Any change to native's numerics, beyond following the reference where §4 allows it.

### 3. Context

The design pages: [backends.md](../backends.md) §1, the seam and the table of marches;
[architecture.md](../architecture.md) §9; [kernel-contract.md](../kernel-contract.md) §8,
the periodic-statistics hooks, and §§11–12, the compiled marches; and the native package's
`native/README.md`.

**Prior proposals and decisions:**

- **SEP 47:** backend-owned sessions, and the explicit test interfaces of its §6.8.
- **SEP 39 and SEP 45:** the compiled Adams and Runge–Kutta marches, and their
  decision-trace tests.
- **SEP 25:** the rounding-level comparison rules.
- **SEP 29:** the helpers' ownership, `PLAIN` and `NATIVE`.
- **SEP 46:** calibration's use of the timing probe.
- **SEP 49:** the TDSE host source. Its §10.1 records the deferral of slimming `numpy_ref`
  to a test library, which keeps the reference's complete runs. This SEP keeps them too,
  and takes up only the narrower part §10.1 names, the reference's group and batch paths
  (Q1).
- **ADR-058:** 0.x stability.
- **ADR-077:** two backends, binary64 only.

**Issues:**

- **#110**, "Write a SEP for the step loops and the kernel seam". This SEP decides two of
  its four items. The step loops become one march per method on each backend, over engine
  protocols that the two backends share. The single-trajectory batch axis leaves the
  backends, with the lockstep batch decided by Q1. The integrator–tolerance split and the
  route union stay with #110, which is narrowed to them.
- **#618**, the batch axis left after SEP 47. Fixed for `Backend.max_batch` and the
  backends; the functions in `translate/` and `boundaries.py` that it lists stay with that
  issue.
- **#344**, the low-storage hook names. Its naming item is fixed: the `_adams_*` hooks
  become the workspace's operations, with names that fit every march.
- **#107**, one host statistics module. Informed: native keeps its statistics in C++
  (SEP 47 §6.4), and the reference's move into its workspace. This SEP recommends closing
  #107 on that decision.
- **#171**, one diverging point fails every member of a `Batch` group. Fixed under Q1's
  recommendation, because each member runs its own session.
- **#356**, a fused GPU batch axis. Unaffected.

### 4. Theory

No physics changes. This proposal changes how computations are arranged, not what they
compute. Comparisons follow SEP 25: a matched discrete computation is held to a rounding
budget fixed before the comparison, and an adaptive run to its accuracy envelope.

**Rounding and simplification.** The reference may change its numbers where a
simplification needs it. Two examples:

- the Runge–Kutta engine keeps its trial and commits it itself, where the driver now
  assigns the returned state;
- the boundary reading is taken in one place, where it is now spread between `_finish` and
  `_take_step_stats`.

Each such change is recorded in its phase report. A change is accepted only on that
evidence, never by regenerating a table without it. The report gives:

- the computation it changes;
- the largest difference found on fixed schedules, against that computation's SEP 25
  budget;
- the frozen tables and decision traces it regenerates;
- whether native follows it.

A difference above rounding needs a stated cause and the maintainer's agreement in the
report.

Native's session transcribes the reference's scheduling decisions, so a change to the
reference's march or session bookkeeping is mirrored in C++ in the same phase. The
decision-trace tests then hold the two together again.

### 5. Proposed interface

Nothing changes for a user. The names below are private to the backends or to the binding.

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
| Engines | What a march asks of the workspace (§5.2) | Python classes over `NumpyWorkspace` | `rk::Engine`, `march::Engine`, the low-storage step |
| Marches | Each march's controller, landing, history and call ledger, held in its own state object | Python marches over `rk_core` and `adams_core` | `rk_march.cpp`, `adams_march.cpp` |
| Session | Planes, step planes, natural stops, yields, dense holds, periodic rows, the snapshot latch, the boundary monitor and transverse edge, the noise reset, events and exports | `HostSession` | `session.cpp` |

**The run records** are built once per run by shared code in `solprop.backends`, from the
problem, the request and the `Start`. The reference reads them directly; native converts
them with the generic converter of §5.3. They are:

- the method's policy: `MarchPolicy` for Adams, a new `RungeKuttaPolicy` dataclass, and the
  low-storage settings;
- the schedule, when one is fixed;
- the plane plan: planes, step planes, statistics cadence, replay length and capacities;
- the monitor and edge configuration, and a resumed run's continuation records. These move
  from `solprop_native._watch` and `_transverse` into solprop.

The statistics lowering stays per backend: native's is a C++ plan (SEP 47 §6.4), and the
reference evaluates `StatsSpec` directly.

**The `Backend` protocol shrinks.** `HostStepping`, `BatchBackend`, `max_batch` and the
three periodic-statistics storage hooks leave it. What remains is construction, `prepare`
and `prepare_group`, `timing_probe`, `metadata`, `memory_ledger`,
`available_device_memory`, `march_export_refusal`, `edge_memory`, `close` and `closed`.

**Boundary warnings** become `WARNING` barrier events on both backends. The service loop's
`HostSession` special case goes: both sessions report their execution facts through
`execution_record`.

#### 5.2 The engine protocols

The Python protocols mirror the C++ interfaces one-to-one. An engine keeps its trial in
flight and commits it on `accept`.

```text
class RungeKuttaEngine(Protocol):            # rk::Engine
    def trial(self, z: float, dz: float) -> float: ...       # the error norm
    def accept(self, dz: float) -> None: ...                  # finish and commit
    def extend(self, z: float, dz: float) -> None: ...        # dense stages at a plane
    def reading(self) -> BoundaryReading: ...                 # the committed state's
    @property
    def in_flight(self) -> bool: ...

class AdamsEngine(Protocol):                 # march::Engine
    def start(self, z: float) -> None: ...
    def trial(self, z: float, dz: float, stencil: Stencil) -> NDArray: ...
    def accept(self, dz: float, free: int, stencil: Stencil) -> None: ...
    def chord_phase(self, z1: float, z2: float) -> float: ...
    def begin_attempt(self, cost: int) -> None: ...           # the TDSE budget
    def end_attempt(self, outcome: AttemptOutcome) -> None: ...
    def reading(self) -> BoundaryReading: ...

class LowStorageEngine(Protocol):            # the native low-storage step
    def step(self, z: float, dz: float) -> float: ...
```

These align the existing `AdamsEngine` and `LowStorageEngine`. Two members move:

- `hold`, `restore`, `observe` and the snapshot rows move to the session, where native
  already has them;
- the TDSE reservation moves from the backend onto the engine.

A march holds its own state and touches no other object's fields.

#### 5.3 The native binding

```text
_core
  build_record() -> dict
  configure_runtime(record)                    # threads, spin, BLAS, as one record
  wisdom_export() -> str | None; wisdom_import(text) -> bool
  available_device_memory() -> int | None
  BackendRefusal, EnvelopeConvergenceError, SessionLifecycleError
  schemas() -> dict                            # every record's fields and types

  Problem                                      # one record; includes every set-once
                                               # table, guard and geometry
  Workspace(problem)
    .start(run, host_source=None) -> Session   # run: the shared run records
    .probe() -> Probe                          # Probe.trial(z, dz), Probe.field()
    .diagnostics() -> dict                     # metadata, memory, sizing, planning
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

_core._testing                                 # private; no contract (§6.4)
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

Counts that an array's shape gives are read from the array, not sent beside it. Python
derives nothing that C++ then derives again. There are three real constraints:

- a workspace is reused across the points of a scan;
- the TDSE host source is a live Python object;
- the CPU execution record's second calibration stage reads a built workspace's sizing.

The first two keep `start` separate from the workspace. The third keeps the rebuild that
exists today.

**Records by field table.** Each C++ record declares its fields once, next to its
definition: one line per field, giving a name and a member pointer. A single template
converts any record from a Python dataclass, or from a mapping, and back. It handles
scalars, enumerations by name, optionals, nested records, lists of records and exact NumPy
arrays. It refuses a missing or unknown field by its path. `_core.schemas()` exports every
table, and one test holds each Python record to its C++ table, names and types. The
`Fields<>` helper that binds `Problem` today is that converter's first case. Results such as
`Batch`, `Status` and `Snapshot` come back through the same tables.

**Generated stubs.** The whole `_core.pyi` is generated by nanobind's `stubgen`, and the
native gate checks it, as it checks `Problem`'s part today.

**Bindings marshal; logic lives in C++.** The binding unit converts, releases the
interpreter lock and forwards. The lease between workspace and session, the coupled-flow
guard and the march entry points move into the C++ library, where ctest reaches them.

### 6. Implementation strategies

#### 6.1 Layers and the contract

The work stays within the backends layer (architecture.md §3) and the native package. Two
things leave the frozen surface of kernel-contract.md §8: the `HostStepping` methods and
the periodic-statistics storage hooks. So `KERNEL_SCHEMA_VERSION` is bumped once.
`PropagationProblem` and `RunRequest` keep their fields. The frozen descriptor tables in
`tests/backends/test_driver.py` keep their cases. Their harness moves from `HostDriver` to
the workspace and session.

#### 6.2 The reference

**`HostDriver` is dissolved, not refactored in place:**

- The Runge–Kutta march becomes a march over `RungeKuttaEngine`, with its state in a march
  object, beside the Adams and low-storage marches.
- The scheduling, halts, step planes, dense holds, periodic rows, the monitor and edge
  checks and the noise reset move into `HostSession`. That makes the session the Python
  counterpart of `session.cpp`, which native's header already calls a transcription of
  these functions.
- `NumpyRefBackend` keeps lowering, `prepare`, `metadata`, `memory_ledger`, the probe and
  `close`.
- Its physics moves into `NumpyWorkspace` together with the operations `HostHistory` and
  `HostLowStorage` call today, as named methods. That is roughly 2,000 lines: stage and
  source, finish, mask, statistics and the module tables.

The interleaved bookkeeping is pulled out of the physics:

- the split collar's sums leave `_finish` for the session's boundary reading;
- the quadrature alarm becomes a workspace reading the session reports, as native's
  `quadrature_peak` already is;
- the TDSE reservation moves onto the engines.

The scripted test drivers become scripted engines under the real marches: today's
`HostScriptedDriver` and the trace harnesses' `_HostHarness`. That is how native's
scripted engines already test its marches.

**The lockstep batch** (Q1). Recommended: retire the fused reference batch. A reference
group runs one session per member, as native's does. The `Batch` executor keeps its
promise of independent members on one fixed schedule; the reference's stage sums are no
longer fused across members, and the reference's speed is not one of its goals. This
deletes:

- `HostGroup`, `NumpyBatchBackend` and `select_batch`;
- the `BatchBackend` protocol;
- most of `host/group_session.py`.

It fixes #171. The alternative keeps a fused group over workspaces: a group engine whose
trial sums the members' stages. It costs a third march-like loop to keep correct, for a
performance gain that only the reference sees.

#### 6.3 The native seam

The adapter's configuration code collapses into building one `Problem`, then one run record
per run. These all go, with the flags that sequence them (`_configured`, `_rk_staged`,
`_rk_weighted`):

- the lazy `_configure` calls;
- the post-construction uploads (`table_upload`, `adams_stage_tables`);
- the monitor binds (`bind_monitor`, `bind_occupancy`, `bind_edge`, `bind_edge_lines`).

**The timing probe** (`Simulation.estimate` and SEP 46 calibration) becomes a native
`Probe`: one trial from the loaded state, committing nothing, and the field read back. It
replaces:

- the adapter's per-step trial path (`_trial_step`, `_prepare_trial`, `_score_trial`);
- `_ResidentLowStorage.step`;
- `_ZTables`' per-position evaluation, and the bindings behind them.

That is about 120 binding lines and about 430 adapter lines.

**The Adams engine's adapter** shrinks to its construction and resume. Its per-step methods
are dead and go, with their bindings.

**The 12 metadata properties, the memory ledger, the execution sizing and the planning
record** become one `diagnostics()` record. That record is what `NativeBackend.metadata`
already assembles.

#### 6.4 Tests

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

`_testing` is built into the same extension and shipped in the wheel, so the wheel tests
can run it. It promises nothing, and its users are solprop's own tests.

On the reference:

- about 39 test files call `propagate_to` directly, and 46 call private physics methods;
- the scripted and trace harnesses subclass `HostDriver`.

These move to sessions, to workspaces and engines, or to scripted engines. A test whose
only purpose was the old structure is deleted, and its phase report says why. The trace
corpora keep their decisions. A corpus is regenerated only under §4's rule.

#### 6.5 Choices of tool

**nanobind stays.**

- It already builds the extension against Python's stable ABI as one `cp314-abi3` wheel
  per platform, tested on 3.14 and 3.15.
- It is the smallest and fastest-compiling C++ binding library that also handles NumPy
  arrays with explicit ownership.
- It releases and reacquires the interpreter lock for the host source.
- Its `stubgen` can generate the whole stub.

The binding's size comes from the interface's shape, not the library.

- **pybind11** would be larger and slower, for nothing here.
- **Cython** fits Kokkos's C++20 templates poorly and adds a language.
- **A plain C interface called through ctypes or cffi** would make the extension
  independent of Python's ABI, and usable from another language. It would cost the typed
  exceptions, array ownership and lock handling that nanobind gives, and move marshalling
  into Python. The interface proposed here, one record in and a session out, would let a C
  interface be added later if another front end ever needs one.

The release build should pin nanobind's version. Today it takes whatever newest release
satisfies `>=2.9` at build time.

**Field tables, not code generation or reflection.** Generating both languages' records
from a schema file would add a build step and a third place for every field. C++26
reflection would remove even the tables, but nvcc, HIP Clang and Apple Clang will not offer
it for years. When they do, the tables can be replaced without touching the Python side.

#### 6.6 Cost

Rough estimates, to be compared with the measured counts at Final:

| Area | Today | Estimate after |
|---|---|---|
| Native binding units (C++) | about 5,000 lines | 1,500 to 2,000, plus about 300 for the converter |
| `solprop_native` (Python) | 14,154 lines | 11,000 to 12,000 |
| `solprop.backends` (Python) | 24,834 lines | 20,000 to 21,500 |
| Hand-written `_core.pyi` | about 700 lines | none |
| Tests | — | churn in about 120 files; the net count is not estimated |

`backends` loses most of `HostDriver`, the protocol surface, the duplicated bookkeeping and,
under Q1, the lockstep group. It gains the Runge–Kutta march object and a larger session.
No dependency is added. Native's per-step run time is not expected to change, because the
marches are already native. Preparation may get slightly cheaper with fewer calls and
copies, which matters only for scans of small cases. Today each array is copied three
times between NumPy and the Kokkos views; that falls to one.

### 7. Verification and validation

**Numbers:**

- **Session suites.** Both backends' session suites pass with their cases and bounds
  unchanged: lifecycle, resume, snapshot, groups, the boundary monitor and the edge, and
  every route's propagation tests.
- **Matched-discrete parity.** Native against the reference keeps its bounds (SEP 25).
- **Reference before and after.** On a fixed set of fixed-schedule runs covering every
  march and route family, the restructured reference is compared with the reference before
  this SEP:
  - bitwise where the order of operations is unchanged;
  - otherwise within the computation's SEP 25 budget;
  - anything beyond that is handled by §4's rule.
- **Decision traces** hold both marches to the same decisions. A regenerated corpus is
  listed with its cause.
- **Adaptive runs** keep their accuracy envelopes in the existing qualification cases.

**Tests are not lost.** Each phase report carries an inventory of every removed binding,
hook or protocol member: the tests that used it, and each test's destination, ported or
deleted with a reason. The coverage measured by the existing lanes, native's C++ line
coverage and solprop's Python coverage of `backends/`, does not fall except for deleted
code. The native C++ tests gain the logic moved out of the binding unit.

**Records cannot drift.** The schema test of §5.3 fails when a Python record and its C++
table disagree. The generated stub is checked by the native gate.

**Performance:**

- A quick A/B of native step time on a few release cases: no change is expected.
- Native preparation time on a small-case scan.
- The reference's test-suite wall time, which should not rise noticeably.

**Size:** `poe size` and the binding units' line counts before and after each phase.

### 8. Documentation

These developer pages change:

- [backends.md](../backends.md) §1, §3 and the table of marches: one shape, two
  implementations.
- [architecture.md](../architecture.md) §9.
- [kernel-contract.md](../kernel-contract.md) §8 and §§11–12, for the hooks and names.
- `native/README.md`: the binding's rules, namely records by field table, bindings that
  only marshal, `_testing` that promises nothing, and generated stubs.
- [testing.md](../testing.md): the engine-level parity pattern and the scripted engines.

An ADR records the shared shape and the native seam's rules. No user manual page changes.

### 9. Compatibility and impact

- **Users:** nothing changes for users of `Simulation`, scans, recipes or output.
  `solprop.backends` changes its private and protocol surface, which ADR-058 leaves
  unstable.
- **Kernel schema:** `KERNEL_SCHEMA_VERSION` is bumped once. The output schema does not
  change.
- **`Batch` executor:** under Q1, a reference group's members run as independent sessions.
  A diverging member then fails alone (#171). Results otherwise agree with today's
  lockstep members to the bound that already holds between a group member and the same
  member run singly (SEP 25).
- **Reference numbers:** they may move under §4's rule, and each move is recorded.
- **No deliberate difference from Luna.jl is involved.**

### 10. Alternatives considered

- **Shrink the native seam alone.** This is the smaller change, and phase 0 does part of it
  anyway. It leaves two structures for one design. It also leaves the reference unable to
  share the engine-level parity tests, or to serve as a line-by-line source for native's
  session.
- **Slim `numpy_ref` to a test library, with no session or step loop** (SEP 49 §10.1).
  This was deferred on 2026-11-12 to keep independent complete runs for development,
  debugging and verification. This SEP keeps them, and gives them the same structure as
  native's.
- **Let the reference run Python engines under native's session.** This would make the
  structures identical by construction. It breaks SEP 47 §6.6: the reference must not call
  native's controller to establish the agreement it is compared on.
- **Drive native's engines from Python marches.** This is the per-step seam SEP 47 removed.
  It costs a crossing per trial and a second implementation of every march's bookkeeping.
- **Keep dictionaries, with a schema test only.** This removes drift but not the ~21
  converters or the hand-built dictionaries.
- **Typed nanobind classes for every record, built by Python.** Stubs then type every
  record for free. But the reference would need the same records without native installed,
  so they would exist twice. The Python dataclass stays the one definition, and the field
  table only describes it on the C++ side.
- **A C interface or another binding library:** §6.5.

### 11. Phases

Everything lands on one branch and in one pull request (SEP 0).

**Phase 0 — native deletions and the test module.**

- Delete the unused bindings and the dead per-step adapter code.
- Move the test-only bindings into `_testing` and repoint their tests.
- Fix the stale docstrings that cite `_ResidentAdams` and `_ResidentLowStorage`.
- Pin nanobind in the release build.
- Verify: the coverage rule, and the inventory of §7.

**Phase 1 — shared records and engine protocols.**

- The run records in `solprop.backends`.
- The `RungeKuttaEngine` protocol.
- `AdamsEngine` and `LowStorageEngine` aligned with C++.
- Native's adapter builds its dictionaries from the records. No behaviour change.
- Verify: unchanged results, bitwise, on both backends.

**Phase 2 — the reference restructured.**

- `NumpyWorkspace`, the engines, the marches and the session's scheduling.
- `HostDriver` deleted.
- Boundary warnings as events.
- Q1 carried out.
- The tests ported.
- This phase may be split into Runge–Kutta first, then Adams, low-storage and groups.
- Verify: §7's before-and-after comparison, the session suites, the traces, and each
  numerical change under §4.

**Phase 3 — the native seam.**

- The field-table converter and `schemas()`.
- `Problem` absorbs the set-once data.
- `Workspace`, `start(run)` and `Session` as in §5.3.
- The native probe.
- `diagnostics()`.
- The logic moved out of the binding unit.
- The generated stub.
- The engine-level parity tests run against both backends.
- Verify: the session suites, parity, the coverage rule, the A/B and the size counts.

**Phase 4 — documentation and the ADR.** The pages of §8; `KERNEL_SCHEMA_VERSION`; #110,
#618, #344 and #107 updated or closed.

### 12. Open questions

- **Q1.** Retire the fused reference batch (recommended, §6.2), or keep a fused group over
  workspaces?
- **Q2.** Ship `_testing` in the wheel (recommended, so the wheel tests can run it), or
  build it only for development?
- **Q3.** Which of today's binding-level oracle tests to replace with C++ tests or
  session-level tests, not port? This is decided per test in phases 0 and 3. The default is
  to port.
- **Q4.** Keep a separate `Probe` (recommended), or time trials through a one-step
  session? The probe commits nothing, which a session cannot promise without a rollback
  path it does not have.
- **Q5.** Which reference simplifications to take that move numbers? Each is decided in
  phase 2's report under §4. The engine holding its trial and the boundary reading taken in
  one place are the expected first two.

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
