# SEP 47 — Backend-owned execution sessions

| | |
|---|---|
| **SEP** | 47 |
| **Title** | Backend-owned execution sessions |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 (2026-11-08) |
| **Status** | Draft |
| **Review status** | Not reviewed |
| **Reviewers** | — |
| **Type** | Change |
| **Created** | 2026-11-08 |
| **Resolution** | — |
| **Requires** | SEPs 25, 39, 44, 45 and 46 |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | — |
| **Manual pages** | — |

## Part I — Proposal

### 1. Abstract

Make the execution backend the sole owner of a propagation's mutable numerical state
and execution schedule. Python constructs and validates the scientific problem; a
backend prepares its execution and runs a session until an external service or a bounded
yield is needed. The native session owns stepping, controller and history state,
monitoring, observation scheduling, statistics, replay cursors, export preparation and
workspace lifetimes. Python owns material and mode construction, scientific choices,
files, user callbacks and orchestration between simulations. Replace the inherited
`StepDriver` contract with immutable numerical descriptions and an explicit session
protocol; retain an independent NumPy implementation of that protocol. This is a
replacement of ownership, with a deletion plan, rather than another native fast path
under the existing Python driver. Success means a simpler executable contract, preserved
scientific and output behaviour, and measured reductions in avoidable crossings and
field traffic without material performance regressions. A universal speedup is not
assumed.

### 2. Motivation and scope

The current boundary started as a way for Python to conduct a propagation using
replaceable numerical kernels. It has successfully acquired native Adams and
Runge–Kutta marches. Their workspaces, histories and stage tables can remain native, but
the Python adapter still inherits a driver whose original job was to own those things.
State summaries return to Python, which reconstructs accounting and schedules the next
piece of work. Several execution authorities coexist: the outer save-plane loop, the
shared driver, a method-specific Python march, a native march and a native workspace.
Their responsibilities overlap even where the cost of calling between them is small.

The question is therefore larger than the cost of a function call. Can the backend make
one coherent plan for its work, keep that plan and its state, and expose only the events
that the application needs? Doing so should make it easier to optimize large memory-bound
runs, small cheap steps, statistics-heavy runs and fixed-schedule groups without teaching
the Python driver how each native engine works.

The proposed boundary is **scientific description versus numerical execution**. It is
not a restriction to individual kernels, and it is not a transfer of the public simulation
API or material models into C++. Compiled code may own algorithms, scheduling and
representation choices that preserve the specified numerical problem. Scientific
defaults, physical approximations and their error budgets remain explicit inputs.

In scope:

- replace the native adapter's inheritance from `StepDriver` and its mirrored mutable
  march bookkeeping with one native session;
- give built-in integrators, including fixed low-storage replay, one execution lifecycle;
- compile observation, boundary and export work once, then execute it where the state
  lives;
- distinguish the numerical problem from backend-specific packing, workspace planning
  and machine policy;
- give route variants, integrator policies, tolerances and groups explicit contracts;
- preserve existing Python extension cases through an explicit host-owned session;
- remove replaced interfaces and loops, and qualify the replacement across existing
  supported execution modes.

Outside this SEP are new physics, a new integrator, changed tolerances or monitoring
defaults, reduced-precision native propagation, a Julia production backend, a general
expression compiler, distributed propagation and a new output file format. Moving
optmat, mode solvers, recipe parsing, HDF5 or the graphical front end into native is not
proposed. CPU kernel work such as batched forward transforms remains independently
valuable; it need not wait for this restructuring.

### 3. Context

#### 3.1 The baseline and what is already native

This draft was checked against main at
[`e555fcf5`](https://github.com/LupoLab/solprop/commit/e555fcf585b8014f36862422a7e92c7275f5e908).
The [architecture](../architecture.md), [backend design](../backends.md),
[kernel contract](../kernel-contract.md) and [stepper design](../stepper.md) describe
the existing layers. The following implementation details matter to this proposal:

- The [native Python adapter](https://github.com/LupoLab/solprop/blob/e555fcf585b8014f36862422a7e92c7275f5e908/native/src/solprop_native/backend.py)
  constructs persistent stage tables and dispatches native Adams and RK marches. A
  native march releases the GIL and performs many trials per call. It returns scalar
  state and attempt records, which the adapter imports into the host driver and monitor
  accounting. This is already much coarser than a Python call per stage.
- Constant, expansion, separable and general tabulated linear operators are supported
  representations. [Native stage tables](https://github.com/LupoLab/solprop/blob/e555fcf585b8014f36862422a7e92c7275f5e908/native/include/solprop_native/stage_tables.hpp)
  are evaluated natively. Chebyshev packing is not a repeated Python operation on the
  ordinary march, and not every problem uses a state-sized Chebyshev table.
- The [outer driver](https://github.com/LupoLab/solprop/blob/e555fcf585b8014f36862422a7e92c7275f5e908/src/solprop/driver/run.py)
  owns save-plane iteration, output transactions, normalization, snapshots and external
  services. Field-free observations already avoid transferring the field. Native
  statistics and transverse-edge reductions already exist.
- [Low-storage replay](https://github.com/LupoLab/solprop/blob/e555fcf585b8014f36862422a7e92c7275f5e908/src/solprop/backends/low_storage.py)
  still has a Python step loop around resident native registers. The
  [lockstep runner](https://github.com/LupoLab/solprop/blob/e555fcf585b8014f36862422a7e92c7275f5e908/src/solprop/backends/batch.py)
  drives separate member workspaces and reaches into their private state. It is a
  correctness implementation, not general fused native batching.

SEPs [39](archive/sep-0039-native-adams-march.md),
[44](archive/sep-0044-native-edge-read.md) and
[45](archive/sep-0045-native-rk-march.md) establish that native ownership can move
upward without changing the equations. This proposal builds on their algorithms; it
replaces the surrounding ownership contract. Their historical qualification remains
valid and they are not marked superseded.

#### 3.2 What the performance evidence does and does not establish

The following is published evidence **read, not rerun for this draft**. Its commits,
hardware and measurement windows belong to the linked reports. Ratios are not
predictions for a newly restructured engine.

| Evidence | Architectural implication |
|---|---|
| SEP 39's EPYC 7763 measurements at one thread reduced host work per helium trial from 208 to 65 to 1.7 microseconds through packed rows, staged tables and the native march; the corresponding argon sequence was 148, 58 and 1.1 microseconds. | Ownership transfers can matter greatly, but much of this transfer is already complete. |
| SEP 45's initial envelope case reduced crossings from 1,910 to 17 and time outside native calls from 14.5% to 3.3%; its later GNLSE case ran 691 steps in 11 calls. | Removing the remaining call overhead cannot reproduce the whole earlier gain. Remaining time outside a march includes useful observation work. |
| [#576](https://github.com/LupoLab/solprop/pull/576) improved CPU execution through thread waiting, widths, vector math and kernel work. Its M5 comparison changed Julia/native from about 0.79 to 1.05. | A language comparison also measures execution choices inside each backend. |
| [#585](https://github.com/LupoLab/solprop/pull/585) reduced a Xeon Gold 6240 Kerr-beam build from 6.48 to 4.68 s and translation from 5.20 to 2.83 s, while its recorded run stayed about 35.8 s and march about 24.1 s. | Construction, export, statistics and stepping need separate attribution. The run-minus-march difference is not all Python overhead. |
| [#586](https://github.com/LupoLab/solprop/pull/586) improved separable phase evaluation with vector math while preserving the representation. | Compiled tables can enable efficient execution; abandoning them is not itself an optimization. |
| [#582](https://github.com/LupoLab/solprop/pull/582), the draft SEP 46 implementation, makes CPU policy workspace-specific and reports substantial H200 gains for some larger node chunks. | Workspace and chunk decisions need backend information. These results are preliminary implementation evidence, not a claim that every GPU benefits from larger chunks. |

The [Julia experiment, #559](https://github.com/LupoLab/solprop/pull/559), is particularly
instructive because it consumes substantially the same translated numerical description.
It is not a control experiment with the seam removed. Its later A40 report gives a
Julia/native geometric mean of 1.15 in binary64, while the 0.79 binary32 result is a
different precision experiment. Source graph replay reduced host source submission
from about 415 to 4 microseconds, but later whole-trial graph experiments were reverted
after regressions. Direct copying also substantially improved a small field-export case.
The relevant lesson is to own lifetimes, specialization and submission coherently, then
measure. Neither switching language nor capturing a larger graph guarantees a win.
The experiment's evolving ledger and status take precedence over its older PR summary
([reference 8](#13-references)).

#### 3.3 Issue disposition and dependencies

The performance-labelled issues and related design issues were searched on 2026-11-08.
The intended disposition is explicit:

| Issue | Relationship to this SEP |
|---|---|
| [#110](https://github.com/LupoLab/solprop/issues/110), march protocol, batch axis, route union and tolerance split | This proposal supplies all four design decisions. Acceptance can settle its request for a design; implementation still has the gates below. |
| [#346](https://github.com/LupoLab/solprop/issues/346), saved-plane traffic | Direct implementation target: explicit destinations and one export conversion pass, qualified without weakening ownership or file semantics. |
| [#361](https://github.com/LupoLab/solprop/issues/361), backend-owned node chunks; [#349](https://github.com/LupoLab/solprop/issues/349), conservative chunk accounting | Direct targets for backend lowering and workspace planning, coordinated with SEP 46. Do not duplicate work that #582 lands first. |
| [#255](https://github.com/LupoLab/solprop/issues/255), carrier frame with general z tables | Built-in native coverage gap to remove in the session migration. Preserve the existing method; no silent switch to a different carrier representation. |
| [#351](https://github.com/LupoLab/solprop/issues/351), padded Chebyshev degrees | Related optimization enabled by explicit representations; not a prerequisite or an automatic closure. |
| [#356](https://github.com/LupoLab/solprop/issues/356), actual fused batching | The group contract is decided here. A general fused batch implementation and its performance claim remain separate work. |
| [#357](https://github.com/LupoLab/solprop/issues/357), GPU launches and copies; [#560](https://github.com/LupoLab/solprop/issues/560), lessons from Julia | Related evidence. Reprofile current code; this SEP does not promise to remove every device launch or copy. |
| [#584](https://github.com/LupoLab/solprop/issues/584), batched CPU forward transforms; [#587](https://github.com/LupoLab/solprop/issues/587), FFT shape-dependent width | Independent kernel/policy work. Preserve their improvements while moving ownership. |
| [#300](https://github.com/LupoLab/solprop/issues/300), expensive per-step plasma statistics; [#301](https://github.com/LupoLab/solprop/issues/301), reference reuse | Statistics planning must preserve exact requested sampling. Moving scheduling does not make an additional accepted-state calculation free. |
| [#104](https://github.com/LupoLab/solprop/issues/104), reference allocations; [#107](https://github.com/LupoLab/solprop/issues/107), statistics ownership | Related, but the reference need not mimic native's allocation strategy and native reductions must remain resident. |

Two closed issues guard against solving an obsolete problem.
[#362](https://github.com/LupoLab/solprop/issues/362) measured small GPU controller
readback cost in its cases; it does not justify requiring a device-resident controller.
[#89](https://github.com/LupoLab/solprop/issues/89) was closed because
[#506](https://github.com/LupoLab/solprop/pull/506) removed the memory estimator and
admission policy (ADR-065). This SEP does not restore either. Its buffer plan is the
backend's actual allocation plan, used to implement explicit workspace budgets and
report actual storage, not a second host prediction of a backend's total memory.

SEP [46](sep-0046-machine-calibration.md) is an implementation dependency: integrate its
accepted execution-policy design and the final disposition of #582 before qualifying
the new native planner. No kernel optimization listed above must block drafting or
acceptance. Reference-only TDSE support remains governed by SEPs 12, 20, 26 and 43;
this SEP does not admit it on native.

### 4. Theory

#### 4.1 The specified numerical problem stays the same

For the internal spectral amplitude $a(z)$, retain the existing semidiscrete equation

$$
\frac{d a}{dz}=D(z)a+N(z,a),
$$

with $z$ in metres, $D$ in inverse metres and $N$ in amplitude per metre. All signs,
normalizations, Fourier conventions and array meanings are those of
[conventions.md](../conventions.md). In the diagonal commuting case, define a primitive
$\Phi(z)=\int^z D(s)\,ds$. Then

$$
U(z_2,z_1)a=\exp\!\left[\Phi(z_2)-\Phi(z_1)\right]a.
$$

This follows by differentiating the exponential when the operators commute. It is not
a replacement formula for the noncommuting coupled flow; that flow retains its current
qualified integrator and ordering. Loss, masks, collars, source evaluations, dense
output and accepted-step actions retain their specified ordering for every route.

A table is a representation of these specified quantities. For example, a scalar
coefficient path on one interval can be represented as
$f(z)=\sum_{k=0}^{d}c_k T_k(x(z))$, with the existing interval map and fit error. A
backend may evaluate it with Clenshaw's recurrence, pack it differently or fuse its
evaluation with its consumer. It may not silently refit it with a looser error, substitute
a lower rank, change an interval boundary or reinterpret it as a material model.
An exact removal of stored zero coefficients is distinct from discarding small nonzero
coefficients. Approximation budgets remain construction-time scientific inputs.

No new physical constants, empirical coefficients or scientific approximations are
introduced here. The numerical sources are the existing stepper and representation
specifications cited in §13.

#### 4.2 A cost model for the boundary

For one measured window, distinguish

$$
T = T_{\mathrm{construct}}+T_{\mathrm{prepare}}+T_{\mathrm{numerical}}
    +T_{\mathrm{service}}+T_{\mathrm{avoidable}}.
$$

Construction builds the scientific problem; preparation lowers it and creates plans;
numerical work includes requested monitors and statistics; service includes required
exports and file writes. Avoidable work includes duplicate state reconstruction,
redundant field passes and excess submission. Each measured operation is attributed
once. A profiler's time outside native calls is not this last term.

If an attributable fraction $f$ of the baseline can be removed completely, the maximum
speedup from that removal alone is $1/(1-f)$. Thus a run with 4% removable overhead
cannot become twice as fast merely by eliminating that overhead. A structural change
can also enable different kernel packing or buffer reuse; those gains must be measured
separately rather than attributed to function-call removal.

For a complex128 field of $M$ samples, one host read and one host write move at least
$32M$ bytes. An extra full conversion or defensive copy therefore has a bandwidth cost
even if its Python dispatch is negligible. Device transfer, cache reuse and file
buffering add their own costs. Export into a final owner can remove redundant passes;
it cannot eliminate the data the user asked to save.

#### 4.3 Numerical identity and execution identity

Two runs may share their scientific description while differing in execution space,
packing, chunk size, FFT plan and calibrated team width. Keep these identities separate.
Scientific provenance includes the represented operators, response parameters, grid,
initial realization, numerical scheme and its declared policies. Execution metadata
records lowering choices, build, machine policy and actual work. Recording a parameter
does not make it a mandatory equality test on restart: the existing compatibility
policy still decides which scientific or numerical changes a restart permits. Do not
replace that policy with an all-fields hash of either descriptor.

SEP [25](archive/sep-0025-rounding-level-reproducibility.md) governs comparison. Matched
discrete histories use rounding budgets derived from operations, magnitudes,
conditioning and propagation sensitivity. Adaptive trajectories and ordinary replay
use accuracy envelopes against a tighter or converged reference. Execution changes
need not preserve accepted step counts or bits. Integer work accounting, masks, lossless
storage, identities and explicitly exact no-ops remain exact.

### 5. Proposed interface

#### 5.1 User-facing behaviour

Existing `Simulation`, convenience functions, recipes, scan executors, output handlers,
backend names and precision choices remain the entry points. No new required option is
introduced. Existing choices of solver, tolerances, statistics, field retention,
boundaries, snapshots and stop flags keep their meaning. Native still refuses physics
outside its admitted scope.

The session protocol is initially an internal contract shared by solprop and
solprop-native, not a new public user API. Its conceptual use is:

```text
problem = translate(scientific_model)          # immutable numerical meaning
request = describe_run(solver, observations, boundaries, export, limits)
prepared = backend.prepare(problem, request, execution_options)
session = prepared.start(initial_state_or_checkpoint)

while not session.finished:
    events = session.advance(work_limit, yield_deadline)
    service(events)                           # files, Python callbacks, progress
    session.acknowledge(events.receipt)
```

This sketch specifies ownership, not final Python spelling. `work_limit` is a positive
attempt/event bound and `yield_deadline` is a finite host-time service deadline. Neither
changes the physical step size or the requested observation schedule. Deadline expiry
is checked between safe operations; a long kernel or external callback can exceed it.
No real-time latency guarantee is made.

Separate control operations request a stop or a checkpoint at the next eligible
boundary. They preserve existing signal, GUI and snapshot-controller behaviour without
mutating the immutable numerical request. A request that cannot be serviced under the
selected snapshot policy is refused explicitly. Mid-run changes to physics, tolerances
or sampling are not added by this protocol.

#### 5.2 Immutable inputs, opaque preparation and one mutable owner

| Object | Meaning and owner |
|---|---|
| `PropagationProblem` | Validated numerical meaning produced by translation: grid, typed route, operators, source parameters, normalization and representation error records. No mutable controller, workspace or device object. |
| `RunRequest` | Immutable integration, tolerance, observation, boundary, replay and export policies. All defaults are resolved by Python before preparation. |
| Prepared execution | Backend-local lowering and reusable immutable resources. The public simulation may cache it, but cannot inspect or modify its implementation. |
| `RunSession` | One trajectory's state, history, counters, cursors, monitor state, working buffers and pending events. Exactly one mutable owner. |
| Event records | Immutable completed work or explicit service requests, with sequence identifiers and positions. They are not a second live controller. |

Preparation must report unsupported combinations before propagation begins. Its
capability result names the route, method, response or extension that caused refusal.
All sizes and descriptor relationships are validated before their corresponding
allocations. Preparation may allocate plans and workspace; kernels do not.

Prepared resources are reusable only with compatible numerical and execution inputs.
Mutable FFT scratch, controllers, device streams and response workspaces belong to a
session or an explicitly exclusive resource lease. Concurrent sessions cannot silently
share them. `with_input` and scan reuse retain their existing semantics; reuse is not a
new claim of this SEP. Child-process execution creates live backend resources in the
child, preserving existing process and accelerator initialization rules.

Both preparation and sessions have explicit, idempotent close operations and support
scoped ownership. Closing a session releases its pending exports and exclusive
resources without mutating already-delivered arrays or calling user code from native
teardown. A failed or closed session cannot advance. Immutable resources retained across
sessions have an explicit owner; mutating a caller's original array cannot silently
change a prepared problem. The implementation must establish that ownership once at
preparation, rather than defending against aliasing with copies on every advance.

#### 5.3 What an event means

The protocol has a small closed set of event kinds:

- **observation ready:** a requested scheduled or step plane, its statistics and work
  records, and an export ticket only if a field or checkpoint is requested;
- **service required:** an external operation that must complete before advancement,
  such as a Python extension hook supported by a host-owned session;
- **progress/yield:** bounded progress and accounting deltas, with no field transfer;
- **stopped/completed/failed:** a terminal outcome, committed position and explicit
  checkpoint eligibility, plus the relevant diagnostic.

A native session implements built-in numerical observation and boundary policies itself.
It does not call Python to rediscover the next plane, rebuild statistics arguments or
replay accepted-step bookkeeping. Periodic statistics are sampled at their specified
steps even when their rows are delivered later. Attempt records, when required by the
existing output/progress contract, are appended in bulk; a Python reporter may consume
them, but their delivery cannot change the numerical state.

The session keeps separate positions for the committed propagation state and a dense
observation. Interpolating an earlier requested plane does not rewind the controller or
advance a replay cursor twice. Coincident save, step-statistics, boundary and checkpoint
events have one documented ordering inherited from the existing semantics.

Delivery and persistence are distinct. An event receipt identifies a pending batch;
acknowledgment releases its resources and allows advancement across its service barrier.
It does not assert filesystem durability. The Python output transaction retains that
responsibility. A service failure cannot silently acknowledge an event or continue the
run. Session-local sequence numbers prevent accidental duplicate consumption; recovery
after a process crash uses the output's committed records and checkpoint identity,
not an invented exactly-once guarantee for arbitrary callbacks.

Event storage is bounded. A full buffer yields instead of discarding records. The
baseline implementation permits at most one pending field-bearing observation; scalar
rows may be batched within a configured capacity. It does not accumulate an entire run
or eagerly copy every requested plane. A retained dense-output object or extra export
buffer appears in the memory ledger.

Start with synchronous service at these barriers. Overlapping file I/O with propagation
is not required, and no new background writer or callback thread is introduced. Device
work already submitted by a session is synchronized only as its operation dependencies
and host access require. An asynchronous output pipeline would need a separate lifetime
and memory argument; it is not a hidden prerequisite for this design's performance.

#### 5.4 Export and storage ownership

Introduce an internal destination-based export operation. An output adapter requests a
shape, dtype and convention, supplies a validated host destination or accepts a newly
owned result, and the session fills it. Canonical host arrays remain the only field
arrays crossing into ordinary Python outputs. No device arrays enter that interface.

Python specifies the normalization and convention transform once, using the scientific
tables. Native applies the requested scale, conjugation, selection and copy together
where qualified. This moves evaluation, not the definition of true power or Fourier
signs. It must avoid creating a full intermediate internal-convention host field just
to convert it into another full host allocation. Whether conversion occurs before a
device transfer or while filling a host destination is an execution choice measured on
that device.

Existing output handlers may retain every array they receive. The compatibility path
therefore returns detached, owned arrays; the backend never reuses or mutates them.
An optional destination provider may recycle storage only under an explicit ownership
contract and after consumption. A view into a live propagation workspace is never an
ordinary observation. Field-free events allocate and transfer no field. A terminal
checkpoint still includes the state required for restart, independent of regular field
retention.

Lossless storage and round trips remain exact for already-converted values. Fusing a
nontrivial normalization is an independent arithmetic computation and is assessed under
SEP 25; it cannot be justified by silently weakening a lossless-copy test. Initial-state
ownership must likewise avoid redundant retained copies where possible without mutating
user arrays or destroying a reusable simulation's launch field.

#### 5.5 Stop, failure, warning and restart semantics

A stop is observed at a documented safe boundary. Native advances release the GIL;
bounded yields let Python poll arbitrary existing stop flags and signals. A native-safe
stop token may reduce latency but is not required of existing callers. No Python callback
is invoked from a worker or device kernel.

The session owns the complete state of accepted numerical decisions. Boundary checking
and stop decisions are not reconstructed from a second Python monitor. Python formats
and emits the resulting warning or exception. A warning that may become an exception
under the caller's warning filters is a service barrier before further propagation;
its checkpoint and event order preserve the current behaviour.

Checkpoint eligibility is explicit. A rejected trial must preserve the algorithm's
committed state and qualified caches as required by that method. An unexpected failure
during an in-place low-storage step may invalidate the working state; without a retained
copy it cannot claim a checkpoint at the last accepted position. Such a session is
poisoned, reports the in-flight failure and resumes only from an earlier valid persisted
state. This preserves the existing low-storage failure contract without adding an
unconditional backup field.

Restart exports a portable numerical state and, when requested and supported, method
history. It does not serialize pointers, FFT plans, execution-space layouts or thread
pools. Query snapshot size and eligibility before allocating or transferring its
payload, preserving the existing bounded-export policy. Reconstruct observation
cursors from the committed output and checkpoint so a resumed dense or step-plane run
neither loses nor repeats an owed observation. Import checks the existing scientific
compatibility rules. A history whose
format cannot be converted must be explicitly refused or restarted through the existing
documented field-only path, with that choice recorded. It must never silently reset
history. Existing saved files remain readable; output schema changes are not required
by this proposal.

### 6. Implementation strategies

#### 6.1 Recommended structure

Use composition around one session protocol:

```text
Python science and public Simulation
              |
       validated problem + run request
              |
     backend preparation and lowering
              |
     +--------+-------------------+
     |                            |
native RunSession          reference/host RunSession
C++ algorithms             Python algorithms
native kernels             NumPy or explicit kernel executor
     |                            |
     +------ events and exports --+
              |
Python service loop: output transactions, reporters, external hooks
```

The native package owns its complete run-session implementation, including the host CPU
control loop of a GPU run. Device-resident control and whole-trial graph capture are
optional lowering choices, not requirements. The shared contract specifies numerical
meaning and lifecycle; it does not require identical internal data structures or copies
of Python driver objects in C++.

Keep the existing package domains. Put numerical descriptors under `translate/`,
session contracts and the reference executor under `backends/`, and external service
orchestration under `driver/`. Native owns its C++ preparation/session implementation and
a thin Python binding adapter. Split large native implementation units by existing
numerical domains and lifecycle responsibilities when needed; do not replace one giant
Python adapter with an equally entangled C++ class. No new dependency is proposed.

#### 6.2 A semantic description, not a universal compiler

Replace the route-wide optional-field bag with a closed discriminated union of validated
route descriptions. Common grid, operator and source components are shared; each route
variant has the fields its executor actually needs. Admission visits the variant
explicitly. Extending a route changes that typed contract and its tests in both
distributions. It does not require a string-keyed dictionary of optional capabilities.

Keep a finite vocabulary of numerical operator representations: constant diagonal,
existing expansions, separable representations, piecewise polynomial tables and
qualified coupled-flow data. Preserve their mathematical structure and approximation
records through translation. No arbitrary Python callable, material object, serialized
expression tree or general JIT language crosses the ordinary native boundary.

Split translation into two responsibilities:

1. **Scientific translation:** evaluate models, build and validate the chosen numerical
   representation, certify its existing accuracy budget and record its identity.
2. **Backend lowering:** choose packing, retained factors, exact zero-degree elision,
   evaluation fusion, scratch lifetimes, chunk sizes, graph reuse and kernels for that
   representation.

Lowering cannot invent a new physical approximation. If a representation cannot be
executed within the requested budget, it reports the resource problem or uses an
already-qualified equivalent execution strategy. It cannot silently fit a cheaper
different problem. This preserves the useful seam while removing incidental layout
requirements from it.

The canonical imported/exported single-trajectory field has axes
`(n_modes, n_pol, n_omega)`. Internal native storage may add batching, padding or other
qualified layouts; these are private execution choices. The current requirement for
one backend-independent internal field layout is deliberately replaced by exact
boundary layout and numerical-equivalence contracts. Kernel views still have explicit
layouts. Conversion and packing are planned and accounted for, never hidden O(n)
coercions in a binding call.

#### 6.3 Integration policy and tolerance policy

Separate immutable method selection from error scaling and acceptance parameters.
An integration specification identifies Adams, the selected RK method or fixed
low-storage replay and the method's options. An error-control specification supplies
tolerances, norm semantics, controller parameters and step limits. A replay
specification supplies its exact itinerary and existing diagnostic policy. Save planes
and observation sampling belong to the observation request.

These internal descriptors are assembled from existing public objects. This SEP does
not change public constructors or choose new defaults. Method-specific mutable order,
history, error memory, trial state and counters belong only to the session. Coefficients
and validation rules may be shared as immutable data; controller implementations in
NumPy and native remain independently testable.

All built-in combinations currently admitted on native must be supported by its session,
including combinations whose march currently returns to Python for implementation
reasons. General coupled flows and z-table carrier frames are examples. A change of
owner cannot silently change the chosen method, tolerance or carrier approximation.

#### 6.4 Observations as prepared numerical work

Lower statistics and boundary specifications once into an observation plan with explicit
dependencies. At a sample point, share compatible transforms, intensities and reductions
only when their state, position, convention and requested sampling agree. Do not reuse a
rejected-trial or stage field as an accepted-state statistic unless the mathematical
identity is established by the existing method.

Native owns reductions and caches while the state is resident. Python owns names,
units, schema assembly and user presentation. Common specification and validation are
shared; the reference and native numerical implementations remain independent. This
resolves ownership without forcing native statistics through host arrays.

Boundary state, cumulative removed energy, event ordering and actual work counters have
one owner. The Python service loop consumes their records; it does not feed those
records back through a second numerical state machine. Per-step plasma diagnostics
retain their exact requested cadence and may remain expensive. Omitting, approximating
or downsampling them is a different user-visible choice.

#### 6.5 Allocation and machine policy

The backend chooses its node chunk and scratch plan from actual live buffers, available
execution resources, explicit user limits and SEP 46's execution record. Translation
does not freeze a CPU-derived node chunk into the scientific problem. An explicit
workspace budget remains authoritative; automatic selection can use backend knowledge.
Live-range reuse must account for retained dense output, histories, asynchronous
transfers and pending events, not merely the arrays used in one source evaluation.

One allocation description drives the workspace and its named ledger. Record host and
device residence separately, immutable shared tables once per owner, and per-session
storage separately. FFT/library allocations that cannot be prescribed are named as
measured or implementation-dependent storage. This is not a promise of a portable exact
preallocation estimate or admission against total system memory (ADR-065).

Allocation, device setup and FFT planning occur outside the trial kernels. Repeated
advances do not repack the immutable problem or allocate field-sized work buffers.
Per-output storage that the user retains is an external cost; bounded native event and
export storage is not. Every retained internal buffer is visible in `memory_ledger()`.

#### 6.6 The NumPy oracle and Python extensions

Implement the session lifecycle in Python for `numpy_ref`, using readable numerical
descriptions and its independent numerical algorithms. It must not call the native
controller, coefficient evaluator or statistics implementation to establish agreement.
Share semantic descriptors, validation and event types, not the numerical execution
whose correctness is being compared.

Some supported uses intentionally need Python during propagation: custom controller or
norm code, and the current external TDSE binding. Give these an explicit **host-owned
session**. Its numerical operations may use a documented kernel-executor interface where
the selected backend already supports that combination; its one controller and event
state remain on the host. The NumPy executor remains independent. TDSE retains its
current provider lifetime, attempted-call budgets, archive ordering and transaction
semantics and remains reference-only under its existing admission rules.

This is a declared execution mode, reported at preparation and in metadata, not a
surprise fallback midway through a native run. Preserve existing supported public
extension behaviour and refusal/warning policy. Unsupported new callbacks are refused.
Host execution does not justify retaining native mirrors of `StepDriver`: it uses a
small explicit numerical-operations protocol. Ordinary built-in native runs never
construct a host controller shadow.

MLX adapts to the host session and its existing opaque state operations, preserving its
current admitted scope and precision. This change gives it no new physics requirement.
If that adapter cannot preserve an existing supported case, the migration is incomplete;
it cannot silently make the experimental backend unrunnable.

#### 6.7 Groups and the batch axis

A batch is a collection of trajectories, not an extra axis in every single-trajectory
scientific description. Preserve per-member parameters, initial realization, histories,
norms, observations and stop outcomes. The existing shared fixed schedule is an explicit
group execution request. Its maximum error norm remains a diagnostic and cannot adapt
the common schedule.

A group owns its member sessions through their protocol, with no access to private
driver attributes. A native group can execute an entire common schedule window and
return per-member events; a reference group can iterate members in Python. A stopped
member does not force unrelated members to stop where existing scan semantics allow
them to continue. Group output remains separate member outputs under the current schema.

Packed batch kernels are a backend strategy for compatible members, not a guarantee
made by a nominal leading dimension or `max_batch`. This SEP removes the inherited
lockstep/private-member contract and supplies a coarse native group execution path, but
does not require every operation to be fused across members. #356 remains the place to
qualify that further performance work in the production precision. Heterogeneous
adaptive ensembles and new shared-controller semantics are out of scope.

#### 6.8 What must disappear

Completion requires removal, not just bypassing, of the replaced production machinery:

| Existing responsibility | Final owner and removal requirement |
|---|---|
| Native subclass of `StepDriver` and frozen protected-member contract | Native session through explicit descriptors and operations; remove inheritance and private-member reach-through. |
| Per-return `_state`/`_bounds` reconstruction and imported controller shadows | Native session state; expose compact read-only status or explicit restart snapshots only. |
| Python replay of native attempt records to update numerical accounting/monitors | Native append-only accounting and monitor state; Python only delivers records to external consumers. |
| Python low-storage loop around native registers | Native session for built-in execution; reference host implementation remains for the oracle. |
| Native-specific nested save/statistics/step scheduling across adapters | One native event scheduler; the Python driver becomes a service loop. |
| Lockstep access to member private fields and marches | Explicit group/session operations. |
| Translation-owned execution packing and chunk policy | Backend lowering, with numerical representations retained in translation. |

The reference may retain useful host driver code internally while being refactored into
the host session. It must not remain the native contract. Transitional adapters are
permitted during the one implementation branch, but no permanent dual architecture or
user-selectable legacy engine is part of the final design. Standalone kernel tests and
diagnostic probes remain available through explicit test interfaces.

### 7. Verification and validation

#### 7.1 Numerical and lifecycle qualification

Freeze the existing capability matrix and expected refusals before implementation.
Exercise each admitted route and integrator on the reference and native CPU, and
preserve the existing separately qualified GPU and MLX scope. Tests cover:

- analytic linear flow, zero source, dispersion-only and loss limits; existing
  conservation, symmetry and convergence checks for each affected method;
- one operator application, one source evaluation, one step, short matched histories,
  each statistic and short adaptive/replayed runs against the independent NumPy oracle;
- constant, expansion, separable and general z-table representations; density and taper
  variation; coupled flow; carrier history; field and envelope cases;
- endpoint and dense observations, coincident events, step planes, field retention,
  statistics at every accepted step, temporal and transverse boundaries;
- stop before the first step, between trials, at an output barrier and during a long
  advance; rejection, failed output, warning-as-error and an in-flight low-storage
  failure; no false resumable checkpoint;
- restart with and without history, bounded snapshot export, repeated input reuse,
  group member failure, callback exceptions and the existing TDSE transactions;
- reference/host execution of admitted Python extensions and MLX compatibility.

Use the existing independently justified rounding budgets for matched computations,
extending their operation and conditioning terms where arithmetic changes. Derive any
new budgets before inspecting errors. Adaptive and ordinary replay tests use existing
or newly converged per-case accuracy envelopes, not a bare assertion that local `rtol`
bounds global error. Exact assertions remain for identities, event sequence and cursor
accounting, no aliasing, hard zeros and lossless copies. Chunk membership, delivery
frequency and progress reporting must not alter scientific policy.

Protocol tests deliberately use tiny event capacities, slow/failing consumers, retained
output arrays and repeated acknowledgments. Verify bounded storage, refusal of stale
receipts, no double advance, no mutation of delivered fields, correct dense-state
ownership and accurate actual-work totals. Use the same lifecycle conformance suite for
all session implementations, alongside independent numerical tests; do not make the
native implementation the lifecycle oracle.

#### 7.2 Performance experiments

Record a baseline after SEP 46 and any concurrent kernel work used by the implementation.
Measure interleaved baseline/candidate pairs with the same compiler, precision, machine
allocation, calibration, input realization and observation policy. Use about ten steps
or the short `benchmarks ab --length` windows allowed by the benchmarking rules; never
extrapolate them to a full propagation. Separate cold preparation and warm repeated
execution, with the same FFT wisdom policy on both sides.

The matrix must include:

| Case family | Question |
|---|---|
| Cheap single-mode envelope and field steps, with few observations | Is remaining Python orchestration material after current native marches? |
| Adams and RK with per-step statistics and active boundary checks | Does prepared observation work remove duplicate scheduling without changing sampling? |
| Large radial and Cartesian fields, saved versus field-free planes | Are field passes, transfer bytes and retained buffers reduced? |
| Fixed low-storage replay and a compatible fixed-schedule group | Does coarse session ownership remove the remaining Python step loop? |
| General z tables, coupled flow and a varying guide | Does the new contract preserve the general path as well as the common one? |
| Repeated `with_input` runs and short scans | Is preparation reused without stale state or hidden shared mutable resources? |

Measure construction, lowering/planning, march, observation, export and file I/O
separately, as well as total wall time. Count Python/native calls, native/device launches,
host/device bytes, full field passes, per-advance allocations, retained storage and
actual numerical work. Record stop latency as a distribution with the longest measured
kernel. Counts explain a timing; they are not a substitute for it.

Qualify CPU results on at least Apple arm64 and Linux x86-64, at one thread and their
declared production allocation. Exercise CUDA and HIP on real devices before declaring
their migrated paths qualified; host compilation alone is insufficient. Device results
must state whether an optimization affects submission, transfer, synchronization or
kernel work. A device-resident controller or whole-trial graph is adopted only on
separate evidence, not to make the architecture look more native.

#### 7.3 Success and stop criteria

Correctness and the deletion requirements are mandatory. For ordinary built-in native
runs there is no Python numerical controller shadow, no replay of records to rebuild
monitor state, no per-step Python scheduling except an explicitly requested external
service, and no repeated repacking of immutable problem tables. Field-free observations
remain field-free. Saved-plane export has a measured traffic ledger and no redundant
full host conversion intermediate in its qualified common paths.

Performance is assessed per case. Predeclare an investigation threshold of a repeatable
5% end-to-end regression beyond measurement uncertainty on the baseline matrix. A
geometric mean cannot hide a regressed production case. Any retained regression needs
an explicit maintainer decision with its structural benefit and mitigation recorded;
it cannot be dismissed as inevitable overhead. This threshold is a review policy, not
a statistical guarantee or permission to spend 5% everywhere.

There is no mandatory universal speedup. A simpler ownership model is an acceptable
primary benefit if its measured performance is preserved. Claims of speedup require
paired results on the cases that improve. If phase 1 shows that the proposed service
protocol itself demands more copying or duplicated state than it removes, revise the
proposal before expanding the migration. If the implementation merely adds a session
wrapper around the old drivers, the structural objective has not been met.

### 8. Documentation

Rewrite the current ownership descriptions in `architecture.md`, `backends.md`,
`kernel-contract.md`, `kernel-problem.md` and `stepper.md` as the implementation lands.
Replace the protected driver member tables with the descriptor, lifecycle, event,
export and numerical-operation contracts. Update `native/AGENTS.md` and the root rules
for the new boundary and private internal layouts. Record lasting decisions in
`decisions.md`, preserving the history of the old seam.

Update the running, backend, performance, output, snapshot/resume and scan manual pages
to explain execution ownership, capability reporting, stop latency, field retention
and prepared-resource reuse. Document the explicit cost and scope of Python extensions
without exposing internal packet plumbing in normal user flows. Explain that
statistics and saved fields remain requested work, and that a native session need not
put its controller on the GPU. Add a short developer example for implementing a session
and a failing output consumer. A new full-size gallery is unnecessary; existing short
examples exercise the unchanged public API.

Every implementation phase adds its API docstrings, manual changes and benchmark
ledger evidence with its tests. The final release-note fragment names this SEP and
the internal contract break. No runtime feature is advertised merely because this
Draft has merged.

### 9. Compatibility and impact

The public simulation and file interfaces retain their semantics. Existing files remain
readable, saved fields keep their convention and units, and explicit extension cases
keep their supported execution path. Recipe identity and scientific restart checks
must not acquire hardware-dependent packing or calibration data. Execution provenance
can record the new session and lowering choices without changing scientific identity.

The internal backend contract deliberately breaks: `KernelProblem`, the protected
`StepDriver` surface and stage bundles are replaced or reduced to their new roles.
Update the frozen contract tests and native bindings in the same implementation PR.
The distributions remain released together at one pinned version; no independent seam
version is introduced (ADR-068). Editable-source digest checks remain. External code
using internal backend members must migrate; do not preserve those internals at the
cost of permanent duplicate state ownership.

The native C++ code will grow to own a run, while the native Python adapter and shared
driver machinery should shrink substantially. Total line count alone is not success:
review the number of mutable owners, scheduling loops, cross-layer accesses and
representations that must remain synchronized. Maintenance risk moves toward C++
lifecycle code, so native unit tests, sanitizers and narrow explicit ownership are
required. Moving files without removing responsibilities does not meet this proposal.

Scientific behaviour and the existing differences from Luna remain unchanged. Any
new numerical difference discovered during migration needs a separate justified
decision and, where applicable, the existing differences page; performance is not a
reason to introduce it silently.

### 10. Alternatives considered

**Keep the present seam and add fast paths.** Lowest immediate disruption and appropriate
for #584 or #587. It leaves the ownership duplication and increases the number of
adapters as more loops become native. Continue independent kernel work, but do not use
this as the long-term execution architecture.

**Move only the outer `run_problem` loop to C++.** This can reduce per-plane dispatch but
would embed file and callback concerns in native while leaving mirrored controller and
workspace state. The chosen design moves numerical scheduling and export preparation,
and leaves external service transactions in Python.

**Move the full scientific simulator into native.** This could make construction and
execution one language, but duplicates or relocates Python's material, mode and analysis
ecosystem and weakens the independent oracle. Existing evidence does not attribute those
costs to the seam. It would also be a much broader public-platform decision. Keep the
scientific description in Python and optimize particular construction kernels when
their profiles justify it.

**Eliminate compiled tables in favour of native material functions.** This trades a
validated finite representation for backend knowledge of physical models and their
dependencies. Tables often remove runtime work and enable specialization. Preserve the
numerical structures and move their packing/evaluation choices; introduce a new model
representation only on its own numerical and performance evidence.

**Use one native implementation as the reference too.** This reduces duplicated
algorithms but removes an independent numerical check. Share contracts and mathematical
specifications; retain separate NumPy and native implementations.

**Require a fully device-resident run.** This would conflate native ownership with GPU
control and make error handling, callbacks and less common methods harder. #362 and
#559 show why device control and larger graphs need case-specific evidence. A native
host loop already removes Python from the numerical execution path.

**Make a general task graph or compiler IR.** It could express arbitrary fusion and
pipelines, but creates a second programming system before the supported operator
vocabulary needs one. Choose typed finite descriptors and explicit events; backend-local
graphs remain an execution technique. Simulation coupling remains orchestration at
existing physical interfaces, not a new distributed graph engine.

### 11. Phases

All implementation is on one `feature/sep-0047-run-ownership` branch and one draft
implementation PR, under SEP 0. The phases are independently qualified milestones, not
separate production architectures or separate implementation PRs. Each carries tests,
manual updates and a Part II report. Use targeted local checks; the full, slow, coverage
and native gates run in CI as required by the repository rules.

1. **Contract and measured baseline.** Inventory current capabilities, extension cases,
   state owners, event ordering, snapshots and group semantics. Record the §7 matrix
   after the SEP 46 integration point. Specify typed descriptors and executable
   lifecycle/export conformance tests. Prototype the bounded event and destination
   operations on a cheap run and a large field before committing to their layout.
   Verify that the protocol does not require an extra retained field or controller
   shadow. Set the before/after deletion inventory and phase-specific performance
   targets from this evidence.
2. **Reference and host session.** Split immutable method/tolerance/observation inputs,
   introduce route variants and implement the reference lifecycle with its independent
   numerical algorithms. Replace the outer numerical driver with the service loop.
   Preserve Python extensions, TDSE transactions, MLX and old public entry points
   through transitional adapters. Verify event ordering, failure and resume before
   moving native execution. No new native fast path is required in this milestone.
3. **Native preparation and complete built-in session.** Integrate SEP 46 policy,
   backend-owned lowering and allocation records. Adopt the existing native Adams/RK
   engines; move low-storage replay and remaining admitted built-in host march cases
   into the session. Keep controller, history, boundary state and actual work native
   across advances. Verify all built-in routes, replay and checkpoint semantics on
   native CPU against the reference, and instrument crossings and allocation lifetime.
4. **Resident observations, export and groups.** Prepare statistics and monitor work,
   implement bounded event delivery and destination export, remove redundant field
   passes, and implement native fixed-schedule group advancement through explicit
   member ownership. Verify field-free behaviour, retained output safety, dense output,
   failed consumers and independent member outcomes. Record CPU and device traffic and
   timings; do not fold unqualified approximate statistics or batch precision changes
   into this phase.
5. **Removal and release qualification.** Delete the transitional native driver
   inheritance, mirrored records, private lockstep access and replaced contract tables.
   Complete CUDA/HIP qualification on actual devices and MLX compatibility within their
   existing scopes. Record paired performance and structural results, resolve or
   explicitly document every regression, update rules/manuals, and close only issues
   whose own acceptance criteria have been met. Archive this SEP at `Final` only when
   the replacement is the production architecture and no required path depends on the
   old native driver.

### 12. Open questions

The following decisions require maintainer and independent AI review before acceptance.
Recommendations are deliberate, rather than leaving the architecture to implementation.

1. **How far should ownership move?** Recommend the complete numerical session,
   including observations, boundaries, accounting and export preparation, with Python
   retaining science construction and external transactions. Approving only another
   native march wrapper would materially change the proposal.
2. **May internal layouts differ by backend?** Recommend yes, with canonical host
   boundaries and explicit packing costs. This deliberately changes the present native
   rule, and needs agreement before acceptance.
3. **How should Python extension support survive?** Recommend one declared host-owned
   session using explicit numerical operations, with full preservation of currently
   supported cases. Do not force custom Python code into descriptors or quietly drop it.
4. **How much batching belongs here?** Recommend explicit group ownership and a coarse
   native schedule loop, while leaving general fused kernels and adaptive ensembles
   outside this SEP. This resolves the architectural ambiguity without claiming #356's
   unmeasured throughput benefit.
5. **Is structural simplification sufficient?** Recommend yes, provided correctness,
   deletion and non-regression criteria hold. Large universal speedups are not a
   condition of acceptance, and small timings cannot excuse retaining two architectures.

The exact descriptor field names, packet capacities and native class decomposition are
deferred to phase 1 under the invariants above. Whether a particular GPU benefits from
conversion before transfer, a larger graph or a device controller is deferred to phases
3–4 measurements. Neither changes scientific semantics. A need for a new approximation,
output format, public signature or dependency returns to review rather than being
decided implicitly during implementation.

### 13. References

1. solprop [architecture](../architecture.md), [conventions](../conventions.md),
   [kernel contract](../kernel-contract.md), [kernel problem](../kernel-problem.md)
   and [stepper design](../stepper.md). Numerical meanings and current contracts.
2. [SEP 25](archive/sep-0025-rounding-level-reproducibility.md). Rounding-level
   reproducibility and the distinction between matched histories and accuracy envelopes.
3. [SEP 39](archive/sep-0039-native-adams-march.md) and the
   [native Adams benchmark record](../benchmarks.md#the-native-adams-march).
4. [SEP 45](archive/sep-0045-native-rk-march.md). Native RK march, event boundaries,
   qualification and host-overhead measurements.
5. [PR #576](https://github.com/LupoLab/solprop/pull/576),
   [PR #585](https://github.com/LupoLab/solprop/pull/585) and
   [PR #586](https://github.com/LupoLab/solprop/pull/586), with their entries in the
   [benchmark ledger](../benchmarks.md). CPU execution, fixed costs and phase evaluation.
6. [SEP 46](sep-0046-machine-calibration.md) and its draft implementation
   [PR #582](https://github.com/LupoLab/solprop/pull/582). Machine policy and workspace
   choices; implementation and qualification still in progress when this was drafted.
7. [SEP 22](archive/sep-0022-memory-lean-spatial-execution.md),
   [SEP 24](archive/sep-0024-low-storage-interaction-picture-runge-kutta.md) and
   [SEP 44](archive/sep-0044-native-edge-read.md). Field retention, low-storage
   execution and resident edge observations.
8. [PR #559](https://github.com/LupoLab/solprop/pull/559), experimental Julia backend;
   [status at its examined head](https://github.com/LupoLab/solprop/blob/1e609d68c2aa09e2f1b187cbf627a79826f5dd2a/julia/STATUS.md)
   and [benchmark ledger at that head](https://github.com/LupoLab/solprop/blob/1e609d68c2aa09e2f1b187cbf627a79826f5dd2a/docs/dev/benchmarks.md).
   Experimental evidence, not a supported backend or a precision-matched result in
   every reported comparison. Only solprop's own report is used here.
9. [Decisions](../decisions.md), particularly ADR-065 (remove memory admission),
   ADR-068 (one repository and an unversioned internal seam), and
   [SEP 43](archive/sep-0043-retire-numba-cpu.md) for reference-only execution scope.

## Part II — Implementation record

## Part III — History

- 2026-11-08: Drafted following the analysis of current main, PRs #576, #585, #586,
  #582 and #559, and the performance-labelled issues. No implementation or fresh
  performance measurements are claimed. Tracking issue:
  [#588](https://github.com/LupoLab/solprop/issues/588).
