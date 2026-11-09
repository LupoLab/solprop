# SEP 47 — Backend-owned execution sessions

| | |
|---|---|
| **SEP** | 47 |
| **Title** | Backend-owned execution sessions |
| **Author** | John Travers |
| **Drafting agent** | GPT-6 (2026-11-08); Opus 5.5 (acceptance revisions, 2026-11-09) |
| **Status** | In progress |
| **Review status** | Reviewed |
| **Reviewers** | Opus 5.5 (AI); John Travers (human) |
| **Type** | Change |
| **Created** | 2026-11-08 |
| **Resolution** | Accepted 2026-11-09 by John Travers, after two AI reviews, the drafting agent's responses and the maintainer's decisions recorded in Part III ([LupoLab/solprop#589](https://github.com/LupoLab/solprop/pull/589)). |
| **Requires** | SEPs 21, 22, 24, 25, 29, 33, 39, 44, 45 and 46 |
| **Supersedes** | — |
| **Superseded-by** | — |
| **Implementation** | Opus 5.5 (AI) for John Travers. Milestones in delivery order: (1) phases 1–2, branch `feature/sep-0047-contract` (draft PR to follow); started 2026-11-09 |
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
assumed: conceptual simplification is a sufficient primary reason for the change.
Retire the unused experimental MLX backend instead of migrating it, while retaining
NumPy's single-precision reference. Include saved-plane copy reduction (#346) and
backend-owned device chunks (#361) as workstreams of this implementation. The CPU
transform work of #584 is delivered separately by PR #598; this SEP moves its policy
into backend lowering unchanged.

### 2. Motivation and scope

The current boundary started as a way for Python to conduct a propagation using
replaceable numerical kernels. It has successfully acquired native Adams and
Runge–Kutta marches. Their workspaces, histories and stage tables can remain native, but
the Python adapter still inherits a driver whose original job was to own those things.
State summaries return to Python, which reconstructs accounting and schedules the next
piece of work. Several execution authorities coexist: the outer save-plane loop, the
shared driver, a method-specific Python march, a native march and a native workspace.
Their responsibilities overlap even where the cost of calling between them is small.

Concrete cases still pay for this split. Period-1 statistics force a native return
after each accepted step and deliver each reduced row separately. Transverse-edge
checks disable the native watch; an active collar can require a read every step.
Low-storage replay has a Python loop around native registers, and the lockstep runner
forces native Adams members through host replay. Snapshots currently require
`step_planes=1`, causing an observation at every accepted step even when snapshots are
rare. Several built-in coupled-flow and operator cases also use a host march (§6.3).
At a saved plane, the internal host copy, true-power conversion (even at unit weights)
and optics conversion can create three full host arrays. These are the migration's
specific execution and traffic targets; ordinary native Adams/RK trials already run
in coarse calls. Small copy elisions are an early workstream within this SEP and can
land before the complete session; their effect is measured separately from ownership
changes.

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
- retain reference-only execution, including NumPy single precision, through a
  host-owned session, with no host-over-native kernel executor;
- retire MLX before restructuring its inherited driver, with no migration or new
  qualification effort for that backend;
- complete #346 and #361 within the export/lowering work, and move the CPU transform
  policy that PR #598 delivers for #584 into backend lowering unchanged;
- remove replaced interfaces and loops, and qualify the replacement across existing
  supported execution modes.

Public and internal APIs may be broken wherever doing so produces cleaner code or a
cleaner structure. The project is young and has no downstream users; retaining old API
shapes is not a goal. Update solprop's callers, tests and documentation together, keep
the supported functionality working, and list the API changes in the changelog.
Compatibility shims, deprecation periods and migration guides are not required (§9).

Outside this SEP are new physics, a new integrator, changed tolerances or monitoring
defaults, reduced-precision native propagation, a Julia production backend, a general
expression compiler, distributed propagation and a new output file format. Moving
optmat, mode solvers, recipe parsing, HDF5 or the graphical front end into native is not
proposed. Full RK checkpoint continuation is also outside scope (§5.5). A future Julia
backend can implement the new contract, but replacing MLX with Julia is neither a
deliverable nor a prerequisite. The #584 CPU transform work is delivered independently
by PR #598 (§3.3).

### 3. Context

#### 3.1 The baseline and what is already native

The original draft and review used main at
[`e555fcf5`](https://github.com/LupoLab/solprop/commit/e555fcf585b8014f36862422a7e92c7275f5e908).
This response also checked main at
[`eee57c65`](https://github.com/LupoLab/solprop/commit/eee57c656364861fb7c56ceaad7c93ae8228c92f),
where SEP 46 / #582 is merged and Final. Historical timings below retain their
original measurement scope; they are not new measurements of this main.
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
| SEP 45 phase 1's in-march boundary checks **under Adams** reduced envelope-case crossings from 1,910 to 17 and time outside native calls from 14.5% to 3.3%; replayed `monitor.step` still took 2.1% of the run. Its later RK GNLSE case ran 691 steps in 11 calls, with 4.2% outside native calls. | Removing the remaining call overhead cannot reproduce the whole earlier gain. Removing monitor replay has a specific measured target; outside-march time also includes useful work. |
| [#576](https://github.com/LupoLab/solprop/pull/576) improved CPU execution through thread waiting, widths, vector math and kernel work. Its M5 comparison changed Julia/native from about 0.79 to 1.05. | A language comparison also measures execution choices inside each backend. |
| [#585](https://github.com/LupoLab/solprop/pull/585) reduced a Xeon Gold 6240 Kerr-beam build from 6.48 to 4.68 s and translation from 5.20 to 2.83 s, while its recorded run stayed about 35.8 s and march about 24.1 s. | Construction, export, statistics and stepping need separate attribution. The run-minus-march difference is not all Python overhead. |
| [#586](https://github.com/LupoLab/solprop/pull/586) improved separable phase evaluation with vector math while preserving the representation. | Compiled tables can enable efficient execution; abandoning them is not itself an optimization. |
| [#582](https://github.com/LupoLab/solprop/pull/582), now merged as SEP 46, makes CPU policy workspace-specific. Its phase 6 measured 6–25% gains from larger chunks on H200 spatial routes, but retained no calibrated device setting and landed no new device chunk default. | SEP 47 now includes #361 in backend lowering, with explicit budgets and CUDA/HIP qualification. If an independently qualified change lands first, preserve that rule while moving its owner; no separate prerequisite PR is required. |

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

Applying §4.2 to those historical windows gives ceilings, not forecasts: eliminating
all of the Adams case's 3.3% outside-march time would give at most 1.034×; removing
only its 2.1% monitor replay would give 1.021×. Eliminating the RK case's entire 4.2%
outside share would give 1.044×. The #585 Kerr-beam run spent about 2.9 s of 35.9 s in
saves: even eliminating all saving would give only 1.088×, and required transfer and
I/O cannot disappear. These fractions overlap other useful work and must not be added.
The published period-1 statistics profiles in the
[native Adams ledger](../benchmarks.md#the-native-adams-march) report approximately
290 microseconds per trial outside the march on EPYC 7763 and 75 microseconds on an
Apple M5 Max (one and sixteen threads). The ledger attributes that cost to producing
the row, rather than the return itself. The opportunity is therefore resident rows,
shared numerical work and bulk transfer (§6.4), not a 290/75-microsecond saving merely
from removing a return. Required reductions remain work and must be timed separately.

The ledger's `step_planes=1` cases provide a proxy for the per-step observation forced
by today's snapshot requirement: at one thread the M5 host time per trial falls from
39.5 to 15.2 microseconds (Python march to native march), while the native EPYC case
costs about 58 microseconds per step (160 microseconds on the Python march). They
measure the forced observation/natural-stop path, not snapshot persistence or file
copying, and do not make that whole cost removable. Phase 1 must separate avoiding
unrequested per-step observations from servicing actual snapshot requests.

No isolated current measurement supplies a removable-time fraction for collar checks,
low-storage replay, lockstep, snapshot persistence itself or each host fallback.
Inventing a numerical ceiling for each would be misleading. Phase 1 must measure
these separately, including their required work, before ranking performance payoff.
Monitor replay, export traffic and forced-observation proxies have quantified evidence
already; complete ownership and explicit lifecycle contracts also justify work whose
timing gains may be small.

#### 3.3 Issue disposition and dependencies

The performance-labelled issues and related design issues were searched on 2026-11-08.
The intended disposition is explicit:

| Issue | Relationship to this SEP |
|---|---|
| [#110](https://github.com/LupoLab/solprop/issues/110), march protocol, batch axis, route union and tolerance split | This proposal supplies all four design decisions. Acceptance can settle its request for a design; implementation still has the gates below. |
| [#346](https://github.com/LupoLab/solprop/issues/346), saved-plane traffic | Included workstream: trace both file and in-memory outputs, remove simple copies early, and complete destination-based export in phase 3. Measure large-field plane traffic/peak memory and the kept-save cost on Linux; preserve ownership, storage conventions and exact copy semantics. |
| [#361](https://github.com/LupoLab/solprop/issues/361), backend-owned node chunks; [#349](https://github.com/LupoLab/solprop/issues/349), conservative chunk accounting | Included in lowering/workspace planning. SEP 46 measured the opportunity but landed no chunk change. Implement #361 within phase 3, with live-buffer accounting, explicit-budget priority and CUDA/HIP qualification completed in phase 4. No prerequisite chunk PR is needed; preserve an already-qualified rule if concurrent work lands first. #349 remains accountable to its own completion criteria. |
| [#255](https://github.com/LupoLab/solprop/issues/255), carrier frame with general z tables | Built-in native coverage gap to remove in the session migration. Preserve the existing method; no silent switch to a different carrier representation. |
| [#351](https://github.com/LupoLab/solprop/issues/351), padded Chebyshev degrees | Related optimization enabled by explicit representations; not a prerequisite or an automatic closure. |
| [#356](https://github.com/LupoLab/solprop/issues/356), actual fused batching | The group contract is decided here. A general fused batch implementation and its performance claim remain separate work. |
| [#357](https://github.com/LupoLab/solprop/issues/357), GPU launches and copies; [#560](https://github.com/LupoLab/solprop/issues/560), lessons from Julia | Related evidence. Reprofile current code; this SEP does not promise to remove every device launch or copy. |
| [#584](https://github.com/LupoLab/solprop/issues/584), batched CPU forward transforms; [#595](https://github.com/LupoLab/solprop/issues/595), plans chosen in isolation | Delivered independently by [PR #598](https://github.com/LupoLab/solprop/pull/598), with #587 (maintainer's decision at acceptance). If it merges before the phase 1 baseline it is part of baseline A and earns no SEP 47 credit; phase 3 moves its forward-pack and width policy into backend lowering unchanged, as for #361. Any private switch it adds joins the §6.3 forcing-switch row. If it has not merged when phase 1 starts, the maintainer decides its sequencing. |
| [#587](https://github.com/LupoLab/solprop/issues/587), FFT shape-dependent width | Related policy work; preserve or coordinate its improvements without expanding the transform experiment beyond its measured scope. |
| [#300](https://github.com/LupoLab/solprop/issues/300), expensive per-step plasma statistics; [#301](https://github.com/LupoLab/solprop/issues/301), reference reuse | Statistics planning must preserve exact requested sampling. Moving scheduling does not make an additional accepted-state calculation free. |
| [#104](https://github.com/LupoLab/solprop/issues/104), reference allocations; [#107](https://github.com/LupoLab/solprop/issues/107), statistics ownership | Related, but the reference need not mimic native's allocation strategy and native reductions must remain resident. |
| [#596](https://github.com/LupoLab/solprop/issues/596), full RK checkpoint continuation | Deliberately deferred after R9. This SEP records restart choices; complete RK continuation needs separate qualification. |

Two closed issues guard against solving an obsolete problem.
[#362](https://github.com/LupoLab/solprop/issues/362) measured small GPU controller
readback cost in its cases; it does not justify requiring a device-resident controller.
[#89](https://github.com/LupoLab/solprop/issues/89) was closed because
[#506](https://github.com/LupoLab/solprop/pull/506) removed the memory estimator and
admission policy (ADR-065). This SEP does not restore either. Its buffer plan is the
backend's actual allocation plan, used to implement explicit workspace budgets and
report actual storage, not a second host prediction of a backend's total memory.

SEP [46](archive/sep-0046-machine-calibration.md) is now Final; its execution-policy
design and qualification are the baseline for the new native planner. SEPs 21, 22,
24, 29 and 33 supply the boundary, memory, low-storage, helper-ownership and precision
contracts being moved, not replaced numerically. No kernel optimization above blocks
drafting or acceptance. Reference-only TDSE support remains governed by SEPs 12, 20, 26 and 43;
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
need not preserve accepted step counts or bits when numerical trial inputs differ.
Integer work accounting, masks, lossless storage, identities and explicitly exact
no-ops remain exact.

That numerical freedom does **not** apply to bookkeeping given identical inputs.
On one build, the reference and native control implementations must make exactly the
same proposals, accept/reject decisions, landings, cap decisions, monitor verdicts,
cadence and event transitions when fed identical tables, norms and boundary readings.
Exact identity covers the scalar decision logic. Array reductions and transcendental
evaluations that produce a reading, such as a monitor's floor-removal sum over channels
or a transverse judgement's ordered sums, belong to that backend's reading and are
compared under SEP 25; the identical-input tests feed both implementations the same
recorded readings.
Keep the strict scalar arithmetic and operation order of SEPs 39 and 45, and extend
their strict-file/FMA witnesses to the moved control and scheduler code. The current
RK march comparison uses host and native marches over the **same native numerical
kernels**; it does not prove bitwise equality of independent NumPy trial arithmetic.
Replace that production host-over-native path with test-only replay of identical
numerical inputs and exact control traces. Independent NumPy/native field arithmetic
still uses SEP 25's bounds. Both tests are required; an accuracy envelope cannot
excuse a scheduling error, and an exact scripted trace cannot establish field accuracy.

### 5. Proposed interface

#### 5.1 User-facing behaviour

The user-facing layer continues to support simulation construction, recipes, scans,
outputs and backend selection. Their APIs, including public constructors and extension
interfaces, may be redesigned or replaced wherever that makes the code and structure
cleaner. Update all in-repository callers, including the GUI and examples, to the new
API in the same implementation; no old entry point must be retained for compatibility.

Solver, tolerance, statistics, field-retention, boundary, snapshot and stop capabilities
remain working and tested. Their scientific meaning and defaults do not change merely
because their configuration API changes. Native still refuses physics outside its
admitted scope. The manual describes the resulting API directly, and the changelog
lists the API changes; migration instructions and deprecation stages are unnecessary.

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
boundary, using the token and independent snapshot policy in §5.5. They do not mutate
the immutable numerical request. A request that cannot be serviced under the
selected policy is refused explicitly. Mid-run changes to physics, tolerances
or sampling are not added by this protocol.

Yields are exact transport no-ops: they do not round or reload the field, reset a
controller, invalidate a numerical cache or alter the arithmetic plan. Given the same
numerical inputs and external requests, changing attempt bounds, deadlines, event
capacity, acknowledgment timing or observational reporters must preserve the complete
numerical state and control trace. §7.1 distinguishes this exact contract from
independent numerical computations permitted to differ under SEP 25.

#### 5.2 Immutable inputs, opaque preparation and one mutable owner

| Object | Meaning and owner |
|---|---|
| `PropagationProblem` | Validated numerical meaning produced by translation: grid, typed route, operators, source parameters, normalization and representation error records. No mutable controller, workspace or device object. |
| `RunRequest` | Immutable integration, tolerance, observation, boundary, replay and export policies. All defaults are resolved by Python before preparation. |
| Prepared execution | Backend-local lowering and reusable immutable resources. The public simulation may cache it, but cannot inspect or modify its implementation. |
| `RunSession` | One trajectory's field, declared auxiliary physical state, method history, counters, cursors, controller/monitor state, working buffers and pending events. Exactly one mutable owner. |
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
  such as warning delivery, checkpoint persistence or a Python extension hook
  supported by a host-owned session;
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
advance a replay cursor twice. Preserve the following ordering from the existing
driver, march and output paths; it is the specification for the conformance suite.

| Point | Ordered actions and failure boundary |
|---|---|
| Start/resume | Commit any initially owed plane before attaching/restoring the active boundary monitor, as the current run loop does. Restore or explicitly reset method state under §5.5 before the next trial. |
| Adaptive RK proposal | Apply quiet-interval cap, transverse ceiling and split-collar cap in that order, then landing; take the trial and decide acceptance. Record the attempt before committing an accepted field. |
| Adaptive Adams proposal | Check budget/startup, then carrier, quiet, transverse, split-collar and collar-reentry limits at their existing sites; resolve landing/stencil and execute the charged trial. Commit accepted state, history, kink/order and proposal updates as the method specifies, then record the attempt. |
| Accepted step | Finish the field and commit its position before any accepted-state reading. Update method norm/proposal memory and perform method-specific floor checks at their existing sites; a floor failure precedes monitor/statistics. For replay, validate the norm before committing the prescribed endpoint and advancing its cursor, then record the attempt. Rejected trials record their actual work and never sample accepted-state statistics. |
| Committed endpoint | Check or skip the temporal monitor according to cadence; read the transverse edge where due; take the periodic statistics row; then evaluate the natural stop. Attempt indices include rejected trials. A monitor warning is serviced at its precise position in this order. |
| Requested plane | Perform any due plane-monitor checks and prepare statistics/export. Logically commit the output plane, eligible history and execution record under the existing output transaction; service a snapshot request; call the plane observer; call the reporter; acknowledge the barrier. |
| Independent snapshot | At an eligible accepted endpoint, persist a valid checkpoint, next-observation cursor and required history before copying the immutable snapshot. If a plane coincides, its transaction precedes the copy. No synthetic saved plane is required. |
| Final endpoint | Service owed planes, then run the monitor's final check before recording completion. A final warning promoted to an error records failure and an eligible checkpoint, not completion. |

The precise field/history updates within a trial remain those of the chosen method;
this table does not merge RK and Adams into a new numerical algorithm. Dense planes
are served in requested-position order using the retained extension and their own
logical position. They neither replace the committed endpoint nor advance history.
A warning barrier remembers its continuation point: if Python allows the warning,
resume the remaining monitor/statistics actions once; if it raises, stop there.
Failure during output, a callback or checkpoint persistence cannot mark that operation
successful. Preserve the primary failure and report a secondary checkpoint failure
without claiming a valid export. Attempt-record delivery may be batched; numerical
accounting and the order of committed events may not be changed by delivery timing.

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

The output layer defines one explicit conversion recipe, consumed by the reference and
native exporters and inverted by resume. This preserves ADR-017's ownership of the
optics convention. The recipe includes selection, single-precision widening and its
power-of-two state scale where applicable, route-specific amplitude conversion, and
the spectral origin factor `(-1)^k` together with conjugation. Modal routes use the
declared true-power weights; spatial routes retain their field measure and skip that
modal conversion. Native applies the recipe in a fused pass where qualified. This
moves evaluation, not the definition of power or Fourier signs. Avoid a redundant
full host intermediate, such as exporting an unnormalized field merely to normalize
and copy it into another allocation. Whether conversion occurs before a
device transfer or while filling a host destination is an execution choice measured on
that device.

An output handler may retain every ordinary observation array it receives. Such
observations return detached, owned arrays; the backend never reuses or mutates them.
An optional destination provider may recycle storage only under an explicit ownership
contract and after consumption. A view into a live propagation workspace is never an
ordinary observation. Field-free events allocate and transfer no field. A terminal
checkpoint still includes the state required for restart, independent of regular field
retention.

Key an export by committed state generation (or dense observation), selection and
representation. Output and plane observers requesting the same payload share the
detached result, including during failure handling; they do not re-observe it.

Field checkpoints, including the new running checkpoints, keep today's
`TerminalCheckpoint` representation: the accepted field in the output amplitude
convention (true-power modal amplitudes, or the route's spatial field measure), and
the internal FFT convention at the Python record boundary. HDF5 applies the same
optics conversion as a regular saved plane. A field restart reverses that Fourier
conversion and inverts the modal true-power weights where applicable, just as today;
reconstruction of the propagation state is at rounding level, not a lossless inverse
of normalization. This SEP does not introduce a raw-state field-checkpoint dataset.

A landed field checkpoint and a coincident saved plane share one exported field and
its output conversion, including when checkpointing follows an output failure.
They must not trigger a second observation, device transfer or amplitude conversion.
For these field consumers, a dense plane at a different position requires a separate
payload from the endpoint checkpoint. The optional Adams `MarchSnapshot` is a distinct
existing method-history record, including raw state and engine rows; preserve its
representation and retention policy rather than confusing it with the field
checkpoint. Its required payload is accounted for separately. A new raw-state
field-checkpoint format would need its own justified change; it is not implied here.

Lossless storage and round trips remain exact for already-converted values. Fusing a
nontrivial normalization is an independent arithmetic computation and is assessed under
SEP 25; it cannot be justified by silently weakening a lossless-copy test. Initial-state
ownership must likewise avoid redundant retained copies where possible without mutating
user arrays or destroying a reusable simulation's launch field.

#### 5.5 Stop, failure, warning and restart semantics

A thread-safe native atomic token carries stop requests from the GUI/thread control
path. Native checks it before startup and between attempts, including after completing
an attempt's commit and mandatory accounting. Finish the current attempt safely;
do not interrupt an in-place stage. A rejected trial leaves its qualified committed
state, while an accepted trial finishes the endpoint's required monitor/statistics
work before stopping. No Python callback is invoked from a worker or device kernel.

Releasing the GIL lets another Python thread set the token; it does not cause a Python
signal handler to run while the main thread is inside a long native call. Preserve
bounded returns of at most 256 attempts or a 0.1 s service deadline, checked between
attempts, for Python signal handling. Python handlers can set the token when they run;
no new process-wide native signal handler is introduced. A deadline can overrun by
one attempt and mandatory endpoint work; a thread's token has that same safe-boundary
latency without waiting for the batch bound. Checkpoint transfer and external I/O add
to time until the run is fully stopped. These are cooperative limits, not real-time
guarantees ([Python signal semantics](https://docs.python.org/3.14/library/signal.html#execution-of-python-signal-handlers)).

Snapshot requests are independent of saved-step sampling: remove the `step_planes=1`
requirement. A request latched by a control token is serviced at the next accepted
endpoint; a Python-polled request enters at a bounded yield and is eligible from then
on. It does not force a numerical landing or change the accepted itinerary. Before
copying a snapshot file, persist a valid checkpoint and any required method history,
plus the next observation cursor, even when no ordinary plane is due. Running
checkpoints reuse the existing checkpoint record/datasets and output transaction,
with optional restart metadata and a matching running execution record; no new
checkpoint dataset or field representation is added. A running checkpoint includes,
without consuming, the attempt history and periodic statistics rows accumulated since
the last saved plane, so the next saved plane still receives them. A later failure that
cannot checkpoint, such as an abandoned low-storage step, falls back to the latest
running checkpoint rather than to the last saved plane. Do not create a false terminal
outcome or mark an uncommitted field resumable.
Before that checkpoint, drain and commit every earlier owed dense observation crossed
by the accepted step. The baseline does not persist a checkpoint that would require
an unserialized dense extension to reconstruct missing planes on resume.
Request receipts identify fulfillment or an explicit terminal refusal. Coalescing
equivalent outstanding requests must be explicit to the caller. Snapshot timing is
therefore a deliberate operational change; exact wall-clock snapshot timing is not
promised. Periodic statistics and ordinary saved-step policies keep their own cadence.

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

Restart exports portable numerical state and, when requested and supported, method
history. It does not serialize pointers, FFT plans, execution-space layouts or thread
pools. Monitor restoration keeps today's reconstruction from persisted records:
cumulative totals, quiet state and the transverse record are restored, while the check
cadence and transverse cap restart from that restored state as they do now. The resumed
execution record states this; persisting cadence and cap is not claimed. A restored
monitor cannot start as an unrelated fresh monitor. Reconstruct observation and replay cursors from committed
records, preserving every owed dense or step plane and the existing compatibility
rules. Query snapshot size and eligibility before allocating or transferring its
payload. Keep the Adams automatic 256 MiB retention limit and the explicit policies,
including required-history snapshots; a required snapshot must succeed or fail
explicitly, not silently downgrade to a field restart.

Current behaviour is narrower than full continuation: RK resumes reset proposal and
norm memory; low-storage has no method history; Adams history can be omitted by its
size policy with only an INFO log. This SEP deliberately adds a record of retention
(`retained`/`omitted`/`not_applicable`) and restoration
(`restored`/`reset`/`not_applicable`), with reasons such as explicit policy, automatic
size limit, unavailable history or incompatible representation. Record the decision
at persistence and in the resumed execution record. Warn once on resume when expected
Adams history was omitted automatically or cannot be restored; an explicit request
to restart without history is recorded without that warning. Legacy files lacking
this metadata are read as legacy/unknown evidence, never assigned an invented reason.
Add optional metadata under the existing output compatibility rules; existing files
remain readable and no new output format is proposed.

Full RK continuation is deferred to [#596](https://github.com/LupoLab/solprop/issues/596).
Saving controller scalars alone would not account for recycled stages, internal-field
conversion and all continuation state. Existing field restarts retain their accuracy
envelopes; the new record makes their resets explicit, without falsely claiming exact
continuation. These restart-reporting and warning changes are listed in the changelog.

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
native kernels             NumPy operations
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
`(n_modes, n_pol, n_omega)`. Remove nominal `n_batch` from the single-trajectory problem,
state boundary, observations and statistics. A group is a sequence of typed member
descriptions, parameters and initial realizations, replacing `batch_params` and its
parallel arrays (§6.7). Internal native storage may add batching, padding or other
qualified layouts. The existing public `Backend` already makes its state opaque
([backends.md](../backends.md)); the restrictions being removed are the shared
`StepDriver` march's canonical-array coupling and `native/AGENTS.md`'s explicit ban on
backend-dependent field layouts. Kernel views still declare their layouts. Packing
is planned and accounted for, never a hidden O(n) binding coercion.

The extension procedure replacing kernel-contract §10 is explicit:

| Addition | Contract, admission and qualification |
|---|---|
| Route | Add a discriminated route variant and its scientific validation; each backend visits it and either prepares it or refuses it by name before workspace allocation. Qualify the route's transforms, flow, source and export against independent references. |
| Response/operator | Add a typed component within the finite representation vocabulary, its units/shape/validity/error contract and validation. Extend reference evaluation, native lowering and capability tests; never smuggle it through an untyped optional-field bag. |
| Statistic | Add its typed specification, dependencies, units and sampling rules to the observation plan; qualify periodic/plane equivalence and reference/native reductions, including field-free execution. |
| Auxiliary evolving state | Add a typed physical-state component, separately from method history. Its own physics SEP must specify initialization, evolution/stage coupling, rejection rollback, error control, observations, checkpoint/reset and group ownership before admission. The present set can be empty; this SEP does not invent population or pump dynamics for SEPs 7/32. |

Every addition updates the descriptor manifest, capability/refusal tests, memory ledger,
manual and source/error-budget evidence; it lands on NumPy and native CPU unless its
accepted SEP records an exception. MLX is retired under §6.6 and receives no new
interface or feature work. Update affected feature proposals' extension references when the
contract lands. Frozen session operations and descriptor fields/variants replace the
old driver tables and native `_check_seam` manifest. `KERNEL_SCHEMA_VERSION` is already
gone under ADR-068: do not introduce a replacement version counter. Shared release
pinning, structural checks and editable-source digest validation remain.

#### 6.3 Integration policy and tolerance policy

Separate immutable method selection from error scaling and acceptance parameters.
An integration specification identifies Adams, the selected RK method or fixed
low-storage replay and the method's options. An error-control specification supplies
tolerances, norm semantics, controller parameters and step limits. A replay
specification supplies its exact itinerary and existing diagnostic policy. Save planes
and observation sampling belong to the observation request.

Public configuration objects may be redesigned around these descriptors; retaining
existing constructors or aliases is unnecessary. This changes configuration structure,
not scientific defaults. Method-specific mutable order, history, error memory, trial
state and counters belong only to the session. Coefficients and validation rules may be
shared as immutable data; controller implementations in NumPy and native remain
independently testable.

The current fallback inventory has the following final dispositions. Admission is
decided before allocation/propagation, never by falling back halfway through a native
session. Built-in production capabilities are retained; reference-only diagnostic
methods are explicitly separated from that promise.

| Current host-march case | Native CPU disposition and qualifying witness | CUDA/HIP disposition by Final |
|---|---|---|
| RK/Adams with ordinary built-in controller, including supported norm weights/regions | Native session in phase 3. Exercise default and nondefault policies, strict identical-input decision traces, independent field/norm comparisons and refusals of invalid policies. | Port the current admitted device combinations and qualify the session on both CUDA and HIP in phase 4; preserve existing route/method refusals. |
| `PIController` subclasses overriding scale/accept/propose/rejection/start/equivalent-norm methods | Reference host session only; native refuses the override by name. These subclasses currently occur in tests, not a documented native extension API. Preserve reference experiments and test native admission before allocation. | Refuse by name on both CUDA and HIP, as on native CPU; this is the intentional admission change in §9. |
| `ComparisonMultistep` general PEC/PECE, Adams–Bashforth and matched comparison paths | Keep the benchmark-only comparison family on the reference; refuse it on native. Update benchmark callers and admission tests. Production Adams is not removed with these comparison methods. | Refuse by name on both CUDA and HIP, as on native CPU; this is the intentional admission change in §9. |
| Coupled flow off the eigenbasis path or above the current 16-channel march limit; other admitted non-exponential or untransported-compensation host-engine paths | Probe descriptor feasibility in phase 1; port admitted built-in cases in phase 4, preserving the chosen flow and arithmetic protections. Test small/large dimensions, transport variants, matched steps and converged short runs; lack of an existing performance case is not permission to drop a scientific capability. | Port and qualify on both CUDA and HIP in phase 4, explicitly including off-eigenbasis and more-than-16-channel coupled cases. Retain the existing constant-Schur/frozen-residual and exponential-Adams scope beside a coupled flow; no CF4 or new coupled formulation is admitted. |
| Host-formed rows or operator/prefactor representations missing from native stage-table evaluation | Inventory/probe in phase 1; lower common representations in phase 3 and remaining admitted built-in cases in phase 4. Test each representation's values, trial and event lifecycle. Retire private host-row/debug switches as production execution modes; retain diagnostic witnesses through test interfaces. | Port admitted device-ready representations and qualify on both CUDA and HIP in phase 4. General operator/prefactor z tables that are not device-ready remain refused by name, as today. |
| Carrier-frame general z table (#255) | Probe descriptor/history requirements in phase 1; port the native CPU gap in phase 4 without changing the carrier approximation. Test varying tables, history and restart. | All carrier-frame runs remain refused by name on both CUDA and HIP, as today. This CPU port does not admit device carrier frames or general z tables. |
| Native members forced to host replay by lockstep/nominal batching | Native group session in phase 3, with actual member parameters and cursors. Preserve existing coupled-batch refusals; this does not newly admit every coupled flow as a group. | Port the existing fibre/general-guide groups and qualify on both CUDA and HIP in phase 4, including per-member events and partially advanced group failures. Coupled-flow groups remain refused. |
| Low-storage replay's Python loop around resident native registers | Native session in phase 3 on the existing diagonal-flow routes; qualify replay, event bounds, stop and in-flight failure semantics. | Port admitted device-ready diagonal-flow cases to the session and qualify on both CUDA and HIP in phase 4. HIP is already qualified; CUDA is admitted but not yet qualified, so its session needs actual-device evidence before Final. |
| Private host-engine or host-march forcing switches (`SOLPROP_ADAMS_MARCH`, `SOLPROP_RK_ROWS`, `SOLPROP_RK_MARCH`, the `adams_march="host"` keyword, and any added by concurrent work) | Retire from production; update `benchmarks/adams_host.py`; replace their diagnostic purpose with independent reference and identical-input control tests. | Retire on both CUDA and HIP as well; retain explicit test probes, not a production host executor. |

This table is a deliberate native admission change for custom Python controllers and
benchmark-only comparison methods. It does not authorize dropping built-in production
coupled flows or silently selecting a different tolerance, method or carrier model.
Porting includes native ownership of control around device kernels; it does not
require a device-resident controller. No new production device refusal is selected
by this table.

"Qualify on both CUDA and HIP" in this table means: each device capability keeps at
least its current per-vendor status in the backend capability table. What is qualified
on a vendor today is requalified through the session on that vendor; a combination
that is admitted but not yet qualified on a vendor stays admitted and is listed as
such. The only new device qualification this SEP requires is CUDA low-storage replay.
The device witness is the existing `-m gpu` suite plus the session-lifecycle witnesses,
run as one campaign per vendor at the end of phase 4, after the representative device
path of phase 3 (maintainer's decision at acceptance). Temporary internal adapters may preserve existing device paths between
phased merges, but Final requires the listed CUDA/HIP session witnesses on actual
devices. Host compilation is not qualification. If a port cannot meet this gate,
keep the SEP In progress or seek an explicit revised capability decision; do not
silently turn an existing device capability into a refusal.

#### 6.4 Observations as prepared numerical work

Lower statistics and boundary specifications once into an observation plan with explicit
dependencies. At a sample point, share compatible transforms, intensities and reductions
only when their state, position, convention and requested sampling agree. Do not reuse a
rejected-trial or stage field as an accepted-state statistic unless the mathematical
identity is established by the existing method.

Native owns reductions and caches while the state is resident. Python owns names,
units, schema assembly and user presentation. Common specification and validation are
shared; the reference and native numerical implementations remain independent. This
resolves the ownership half of #107 without forcing native statistics through host
arrays; that issue closes only when its implementation criteria are met.

Within each backend, periodic rows and saved planes use one numerical implementation
of each statistic. Reuse an identical committed-state result for coincident requests,
so the same payload is exactly the same; independently recomputed results retain the
specified rounding bound. Sample only after `_finish` and position commit, never from
the adaptive pre-commit accepted-step hook. Monitor actions precede statistics (§5.3).
The statistics row's attempt index includes actual rejected work, not just its row
number times the period. Keep native rows in bounded resident storage and transfer
them in bulk at delivery; a period of one is not a reason to return to Python or copy
a scalar row after every step. Device buffers are included in the ledger.

Today several periodic-capable statistics are finished in Python on the native
backend, and each row forces a return. The native session computes every one of them
resident, with no Python finishing (maintainer's decision at acceptance):

| Statistic family | Current host work on native | Native session |
|---|---|---|
| Species ionisation, species and multi-channel electron density, peak ionisation rate | Native packs per-node arrays; Python reduces them with species densities, effective area and taper evaluated at z | Resident reduction, with its z-dependent inputs evaluated from the representations carried in the descriptor (phase 3) |
| `density`, `pressure`, `core_radius` | Evaluated from Python profile objects at z | Evaluated natively from the same z representations the propagation uses (phase 3) |
| Plasma ratios, the on-axis row and other per-row scalar inputs (plasma density, preionisation, taper, power weight, on-axis synthesis) | Evaluated in Python before each native call | Lowered once at preparation and evaluated natively per row (phase 3) |
| General-modal statistics (functionals of the physical flux `a.H W a`) | Whole field copied to the host and Python `collect` run for each row | Native resident reductions with the basis metric (phase 3) |

Where a statistic is evaluated from a z representation rather than from the Python
profile object, any difference is bounded by that representation's recorded error,
measured and recorded in the phase report; it is not a new approximation of the
propagation, which already uses those representations. Python names the rows and
assembles the schema; it computes no statistic for a native run.

The boundary monitor is controller input, not merely passive observation. Its quiet
interval, transverse ceiling and split-collar cap constrain proposals in the exact
order in §5.3. Its cadence, cumulative removed energy, warnings, stops and reconstructed
restart state have the same owner as the integrator. Replace `_verify_watch_of` and
runtime Python replay with a test harness feeding identical recorded readings through
the independent monitor implementations, comparing caps, verdicts and following
proposals exactly, including resume and warning barriers.

Moving this ownership is a port of its own: today `watch.cpp` only predicts, and
Python owns the launch judgement, the per-step floor removal, the plane rows, finishing,
resuming and the transverse judgement with its records. The native session owns every
decision state and builds every record (plane rows, summary maxima, the finish verdict
and the resume record); Python only formats and emits them (maintainer's decision at
acceptance). The temporal monitor moves in phase 3 and the transverse judgement in
phase 3 under a timebox, reported at its checkpoint as in §11. The Python service loop
consumes records; it does not feed those
records back through a second numerical state machine. Per-step plasma diagnostics
retain their exact requested cadence and may remain expensive. Omitting, approximating
or downsampling them is a different user-visible choice.

#### 6.5 Allocation and machine policy

The backend chooses its node chunk and scratch plan from actual live buffers, available
execution resources, explicit user limits and SEP 46's execution record. Translation
does not freeze a CPU-derived node chunk into the scientific problem. An explicit
workspace budget remains authoritative; automatic selection can use backend knowledge.
SEP 46 retained no calibrated device setting. Complete #361 here: the backend owns
the device chunk rule and its resource policy, with the larger-default candidate
judged on its measured benefit, memory cost and real CUDA/HIP qualification. An
explicit user budget wins. Do not initialize a device runtime in translation before
a scan forks. If concurrent work has already landed a qualified default, move that
rule unchanged; changing it again requires separate evidence and a recorded decision.
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

The **host-owned session** serves NumPy, in its retained double- and single-precision
scopes. Native exposes preparation and a
complete session; it implements no host-over-native trial/finish/mask/statistics
executor. Custom Python controllers and benchmark-only comparison methods remain
reference experiments (§6.3), not a reason to reproduce the old native seam. Built-in
controller variations become validated policy data. TDSE already runs only on the
reference; retain its provider lifetime, attempted-call budgets, archive ordering and
transaction semantics there. Unsupported callbacks are refused during preparation.
Existing extension and private inheritance APIs need no compatibility layer.

Retire the unused Tier 3 MLX backend at the start of phase 2, before restructuring
the driver hierarchy it inherits. Remove its implementation, registration, installation
extra/dependency wiring, GUI choice, backend-specific tests and obsolete documentation
and coverage exclusions. Selecting `backend="mlx"` must fail clearly, without silently
choosing a different backend. This is deletion and documentation work only: build no
MLX session adapter and spend no effort on MLX numerical migration or qualification.
Update affected examples and regenerate their output under the usual gallery rules.

Keep NumPy's single-precision reference, its precision/scaling contracts and useful
qualification tests. Retiring MLX does not retire SEP 33's reference capability or
require a replacement Apple-GPU backend. Julia remains a separately assessed future
backend; the common descriptors and conformance suite should permit it to implement
the session contract without special accommodation in this SEP.

Extract retained NumPy algorithms into private components used by the host session;
replace `StepDriver` and its inheritance hierarchy by Final. Phase 2 builds the host
session independently (maintainer's decision at acceptance): `numpy_ref` stops
inheriting `StepDriver`, and shares only stateless numerical modules (coefficients,
tableaux, exponential and stage evaluation) with the legacy driver. Until the native
session replaces them, `StepDriver` and its march helpers remain solely as the temporary
native bridge, named in each phase report with their removal milestone.

SEP 46 calibration and `Simulation.estimate` time single trials today through private
driver members (`timed_step`, `_trial_step`, `_observed_field`). Prepared execution keeps
an explicit timing probe: one trial from the loaded state that commits nothing, with a
field readback for the plan comparison. It is a supported internal operation for those
callers, not a propagation executor. Readable host numerical
modules may remain, including useful multistep code, but not a second native
integration contract. Do not retain MLX or an old driver hierarchy as a frozen
compatibility branch inside the production package.

Preserve explicit `solprop._helpers.Helpers` ownership from
[backends.md](../backends.md) and `tests/backends/test_helper_ownership.py`: NumPy uses
`PLAIN` for both precisions; native uses its compiled helpers. Remove MLX-specific
helper-selection paths with that backend, retaining the no-native reference environment.
Pass helpers to the extracted private operations, never select them globally through
shared descriptors. Keep the test that forbids the wrong helpers after translation.
These scalar/table helpers and standalone kernel test interfaces do not constitute a
production host-over-native propagation executor.

#### 6.7 Groups and the batch axis

A batch is a collection of trajectories, not an extra axis in every single-trajectory
scientific description. Preserve per-member parameters, initial realization, histories,
norms, observations and stop outcomes. The existing shared fixed schedule is an explicit
group execution request. Its maximum error norm remains a diagnostic and cannot adapt
the common schedule.

A group request is a sequence of independently translated member problems plus the
existing compatibility predicate that groups scan points today; no production caller
builds a multi-member problem. The batched-scan loop becomes the group service loop.
A group owns its member sessions through their protocol, with no access to private
driver attributes. A native group can execute an entire common schedule window and
return per-member events; a reference group can iterate members in Python. Preserve
the current failure distinction: boundary errors/warnings can stop one member while
the others continue; non-finite norms, step floors, trial exceptions and in-flight
failures stop the group. Earlier members may already have advanced that scheduled
step. Report every member's actual committed cursor and checkpoint eligibility; do
not invent group rollback or a common cursor after partial advancement. A common
restart needs valid checkpoints at a common schedule edge. A group-wide user stop
uses the safe-boundary policy; it does not imply new arbitrary per-member stopping.

Group output remains separate member outputs under the current schema. Ordinary saved
fields stay `(n_z, n_modes, n_pol, n_omega)`. Existing `MarchSnapshot` state/history
payloads carry a leading batch axis: retain that disk representation and adapt a
single-member session by adding/removing its validated size-one axis in serialization.
Existing files remain readable; never silently squeeze a non-unit axis. This change
removes the nominal batch dimension from the runtime single-trajectory contract,
not from every historical checkpoint dataset. No new combined group-file format is
introduced. Packed device axes remain private and are judged by measured throughput,
not by an external shape requirement.

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
| Batched-scan outer loop with its member monitor and record plumbing | The group service loop over explicit group/session operations. |
| Python boundary-monitor state, records and transverse judgement for native runs | The native session's monitor (§6.4); Python formats records only. |
| Python finishing of periodic statistics for native runs | Native resident statistics (§6.4). |
| Translation-owned execution packing and chunk policy | Backend lowering, with numerical representations retained in translation. |
| MLX implementation and its inherited driver/packaging/test hooks | Retired in phase 2; retain the independent NumPy single-precision capability, not an MLX adapter. |

The reference may retain useful numerical code internally while the `StepDriver`
hierarchy is removed (§6.6). It must not remain the native contract. Transitional
internal adapters are permitted across the phased merges in §11, but no permanent
dual architecture or user-selectable legacy engine is part of the final design.
Standalone kernel tests and diagnostic probes remain available through explicit test
interfaces.

### 7. Verification and validation

#### 7.1 Numerical and lifecycle qualification

Freeze the existing capability matrix and expected refusals before implementation.
Exercise each admitted route and integrator on the reference and native CPU, and
preserve the native GPU scope and the device-port obligations in §6.3. MLX retirement
is an explicit exception, not a migration to qualify. Tests cover:

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
- reference/host execution of admitted Python extensions and NumPy single precision;
- clear refusal of the retired MLX selection, removal from advertised/GUI backends,
  and continued import and use of the reference without an MLX installation.

Carry over exact identical-input control tests (§4.3), including recorded watch
readings, rejected steps, caps, dense/endpoint coincidences and restart. Add scheduler
and monitor strict-arithmetic witnesses in the built library. In deterministic CPU
cases, compare attempt limits of one, seven and a large bound, tiny/large event
capacities, varying acknowledgment delays, observational reporters and deadlines
triggered at different attempts: fields and numerical/control traces must be
bit-identical, excluding measured wall times. Script the same numerical outcomes to
test this exact lifecycle property on every executor. For kernels with independently
varying reduction order, do not turn this into a new cross-run bitwise reproducibility
promise; their field arithmetic keeps SEP 25's bounds. No tolerance budget may be
spent on rounding, resets or recomputation introduced solely by a yield.

Test the §6.3 admission table case by case, including dimensions above 16, all retained
built-in representation families and named refusals before allocation. Test token
stopping during a long attempt, Python signal delivery at bounded returns, independent
snapshots without saved-step planes, automatic Adams-history omission, explicit
history-off policy and legacy metadata. Check group failures after earlier members
have committed: reported member positions must match reality, including invalid
in-flight members. No checkpoint or success outcome may be fabricated.
Verify that running and terminal checkpoints use the existing field representation
and datasets, with the present rounding-level normalization inverse on resume.
Test non-unit modal power weights, spatial routes, field-free output, old files and
optional Adams history. A landed checkpoint and coincident saved plane must share
one export/transfer; a dense plane at another position uses its own payload. Exercise
each device disposition in §6.3 on CUDA and HIP as specified, including wide and
off-eigenbasis coupled cases, groups and low-storage replay, before Final.

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

Update tests to the new APIs while preserving their physical, numerical, lifecycle
and ownership assertions. Remove tests of deliberately retired API shapes where they
no longer specify supported behaviour; replace coverage of their retained capabilities
through the new interface. Keeping tests green must not mean deleting or weakening
correctness checks to conceal a regression. Solprop, its GUI, examples and supported
workflows must work with the new API before completion.

#### 7.2 Performance experiments

Freeze an initial baseline at phase 1, after SEP 46 and after PR #598 if it has merged,
before this SEP's optimization work. Keep three attributable variants: **A**, that
baseline; **B**, main just before the first native-session merge, which contains the
early copy elisions on the old ownership model; and **C**, the candidate. Compare B to
A, C to B and C to A on matched cases. Use in-tree ablation switches only where they are
cheap; otherwise record a change as inseparable rather than maintaining a backport
branch or attributing all gains to sessions (maintainer's decision at acceptance). A
transform improvement must not hide a session regression. Concurrent changes already
present in A receive no credit as SEP 47 improvements.

Measure interleaved baseline/candidate pairs with the same compiler, precision, machine
allocation, calibration, input realization and observation policy. Use about ten steps
or the short `benchmarks ab --length` windows allowed by the benchmarking rules; never
extrapolate them to a full propagation. Separate cold preparation and warm repeated
execution, with the same FFT wisdom policy on both sides.

The matrix must include:

| Case family | Question |
|---|---|
| Cheap single-mode envelope and field steps, with few observations | Is remaining Python orchestration material after current native marches? |
| Adams and RK with period-1 statistics, and separate temporal/transverse collar-edge cases | Do resident rows and controller-owned boundary checks remove returns and replay without changing cadence or cap decisions? |
| Large radial and Cartesian fields, saved versus field-free planes | Are field passes, transfer bytes and retained buffers reduced? |
| Fixed low-storage replay and a compatible fixed-schedule group | Does coarse session ownership remove the remaining Python step loop? |
| Independent snapshots at every accepted endpoint and sparse snapshot requests, with/without ordinary saved-step planes | Does independent scheduling preserve valid checkpoints while eliminating returns caused solely by enabling snapshots? Every-endpoint snapshots still require real service. |
| General z tables, coupled flow and a varying guide | Does the new contract preserve the general path as well as the common one? |
| Repeated `with_input` runs and short scans | Is preparation reused without stale state or hidden shared mutable resources? |
| CPU nonlinear forward transforms with small and large mode/node counts, at one thread and the production allocation | Does moving PR #598's transform policy into lowering preserve its measured behaviour? |
| Device spatial routes under automatic and explicit workspace budgets (#361) | Does the chunk policy improve CUDA/HIP execution at an acceptable measured memory cost, while respecting explicit limits and safe initialization? |

Measure construction, lowering/planning, march, observation, export and file I/O
separately, as well as total wall time. Count Python/native calls, native/device launches,
host/device bytes, full field passes, per-advance allocations, retained storage and
actual numerical work. Record stop latency as a distribution with the longest measured
kernel. Counts explain a timing; they are not a substitute for it.

Choose short windows that actually contain the planes, rows, collar reads and snapshot
requests being measured, and verify those event counts on both versions before timing.
An observation-free ten-step window cannot qualify observation improvements. Separate
required work from control/export duplication; include equivalent policies as well as
the deliberately changed snapshot scheduling. Independent copy/kernel improvements
landed before the baseline receive no credit as session improvements.

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

There is no mandatory universal speedup. A simpler ownership model is a sufficient
primary benefit; neutral performance can pass. Claims of speedup require paired results
on the cases that improve. Phase 1 tests whether the proposed protocol can deliver
that simpler structure (§7.4), not whether the architecture deserves to exist. If the
protocol itself demands avoidable copying or duplicate mutable ownership, revise that
design before expanding the migration. Required histories, retained outputs and bounded
service buffers are not automatically duplication: their lifetimes and purpose must be
explicit. A session wrapper around the old drivers does not meet the structural goal.

Also require aggregate non-regression beyond measurement uncertainty: report the
geometric mean of paired candidate/baseline total-time ratios over the predeclared
matrix, one per platform at its production allocation (one-thread runs only for the
cheap-step family), with uncertainty from repeated pairs and baseline-versus-baseline noise.
A uniform 3% slowdown must trigger investigation and an explicit maintainer exception
if retained, even though no case crosses 5%. The aggregate cannot excuse a per-case
regression. State the matrix and weighting before seeing results.

Structural counters must show **zero native returns caused solely by periodic
statistics, internal collar checks, transverse-edge reads, low-storage iteration,
scheduled steps of a native fixed-schedule group, or snapshots being enabled with no
request pending**, when the buffer and work bounds permit continued execution.
Bounded service returns, full buffers,
warnings, requested field output and actual snapshot I/O are counted separately;
zero total returns per step is not required. This measures removal of ownership
overhead without hiding required work or relying on noisy sub-percent timings.

#### 7.4 Phase 1 design-readiness checkpoint

The conceptual direction is already agreed. The checkpoint asks whether the concrete
descriptors, event protocol and export contract are ready for wider implementation.
It is not a minimum-speedup hurdle or an automatic cancellation of the refactor.
Prototype a cheap propagation and a large field through the intended Python/native
boundary, using short windows with real observation/export work. The report contains:

| Criterion | Evidence needed before expanding the migration |
|---|---|
| Simpler ownership | One mutable numerical owner in the prototype, no mirrored production controller, and a credible inventory of loops, state and interfaces that the full implementation deletes. The reference remains independently implemented. |
| Sound lifecycle | Executable witnesses for event ordering, bounded yields, retained output, consumer failure and valid restart boundaries. Returning for service must not reset or reconstruct numerical state. |
| Bounded traffic and storage | A buffer/lifetime ledger and measured copies, bytes, allocations and crossings for cheap and large cases. Field-free service stays field-free; coincident consumers share an export. Identify protocol-forced duplication separately from required numerical or retained-output storage. |
| Coverage of difficult cases | Small feasibility probes for wide/off-eigenbasis coupled flows, carrier z-table/history requirements and device event/export lifetimes. They establish that the contract can express these cases; their complete ports and device qualification come later. |
| Known costs | Baseline and prototype timings with uncertainty and separate required-work/control/export attribution, including currently unmeasured crossings in §3.2. Flag material regressions and their likely cause; an unchanged total time is a valid result. |

Record one of three outcomes in Part II: **proceed** with a sound contract; **revise
and recheck** a specific protocol or ownership defect; or **bring a concrete tradeoff
to the maintainer** if it cannot be resolved within the agreed design. Do not request
fresh approval just because a sound prototype has no measurable speedup. Do not press
on with a known unnecessary full-field copy, unsafe lifetime or second numerical
owner because kernel speedups mask it. If extra storage or a regression is genuinely
necessary, explain its scope, cost, alternatives and structural benefit for a decision.
The per-case and aggregate policies in §7.3 remain safeguards, not an automatic veto
on conceptual simplification.

This is readiness evidence from representative prototypes, not a claim of complete
production qualification or a requirement to finish every native port in phase 1.
State remaining risks and their later witnesses. Focused #346 copy elisions can
proceed alongside it; broad session migration follows the recorded readiness outcome.

### 8. Documentation

Rewrite the current ownership descriptions in `architecture.md`, `backends.md`,
`kernel-contract.md`, `kernel-problem.md` and `stepper.md` as the implementation lands.
Replace the protected driver member tables with descriptor, lifecycle, event and
export contracts, and document the extension procedure in §6.2. Update
`native/AGENTS.md` and the root rules for the new boundary and private internal layouts.
Record lasting decisions in `decisions.md`, preserving the history of the old seam.

Update the running, backend, performance, output, snapshot/resume and scan manual pages
to explain execution ownership, capability reporting, stop latency, field retention
and prepared-resource reuse. Document the explicit cost and scope of Python extensions
without exposing internal packet plumbing in normal user flows. Explain that
statistics and saved fields remain requested work, and that a native session need not
put its controller on the GPU. Add a short developer example for implementing a session
and a failing output consumer. A new full-size gallery is unnecessary; update existing
examples to the new API and regenerate any affected gallery output under the usual
documentation rules. Describe the current API directly, without migration guides,
old-to-new API tables or deprecation instructions.

Remove MLX from installation instructions, backend choices, GUI documentation and
current capability claims, and record its retirement in the changelog. Preserve
NumPy single-precision documentation and historical SEP/benchmark records; historical
MLX evidence is not a promise of current availability. No replacement or MLX migration
guide is required.

Every implementation phase adds its API docstrings, manual changes and benchmark
ledger evidence with its tests. The final release-note fragment names this SEP and
lists the public and internal API changes. A simple changelog listing is sufficient
notice of API breaks. No runtime feature is advertised merely because this Draft
has merged.

### 9. Compatibility and impact

**API compatibility is not a constraint on this restructuring.** The project is young
and has no downstream users. Public and internal APIs may be freely broken, and should
be changed when doing so gives cleaner code or structure. There is no requirement for
deprecation periods, compatibility aliases, shims, migration paths or migration
documentation. Public signatures do not need a separate approval merely because they
change within this agreed design.

Update solprop's own callers, GUI, tests, examples, recipes and documentation together.
Keep solprop working, preserve the retained scientific and operational capabilities,
and pass the required checks. The manual and docstrings describe the new API; the
changelog simply lists the changes. Temporary internal adapters may bridge the phased
merges in §11, but must not become permanent compatibility machinery.

API freedom does not change the numerical problem or authorize silently changing
scientific defaults. Existing saved files remain readable and saved fields keep their
convention and units; a new output format remains outside this SEP. Recipe identity and
scientific restart checks must not acquire hardware-dependent packing or calibration
data. Execution provenance can record the new session and lowering choices without
changing scientific identity.

The internal backend contract deliberately breaks: `KernelProblem`, the protected
`StepDriver` surface and stage bundles are replaced or reduced to their new roles.
Update the frozen contract tests and native bindings together in each affected PR.
The distributions remain released together at one pinned version; no independent seam
version is introduced (ADR-068). Editable-source digest checks remain. Remove retired
internal interfaces and their obsolete contract tables instead of preserving duplicate
state ownership for hypothetical callers.

The native C++ code will grow to own a run, while the native Python adapter and shared
driver machinery should shrink substantially. Total line count alone is not success:
review the number of mutable owners, scheduling loops, cross-layer accesses and
representations that must remain synchronized. Maintenance risk moves toward C++
lifecycle code, so native unit tests, sanitizers and narrow explicit ownership are
required. Moving files without removing responsibilities does not meet this proposal.
The independent NumPy control path also remains a second implementation: every moved
scheduler, monitor and controller rule creates a maintenance obligation in both
languages. Exact identical-input conformance tests pay for this independence; merely
sharing native decisions with NumPy would erase the oracle's value.

The intentional operational changes are retirement of MLX, explicit native refusal of
reference-only controller/comparison experiments, recorded restart choices with the
Adams omission warning, and snapshots independent of saved-step sampling. MLX retirement
removes its experimental Apple-GPU execution; NumPy single precision remains available,
and no replacement Apple-GPU backend is promised. Other retained numerical behaviour,
group failure policy and output conventions remain as specified above.

Scientific behaviour and the existing differences from Luna remain unchanged. Any
new numerical difference discovered during migration needs a separate justified
decision and, where applicable, the existing differences page; performance is not a
reason to introduce it silently.

### 10. Alternatives considered

**Keep the present seam and add fast paths.** Lowest immediate disruption; focused
kernel work such as #584 and #587 (delivered by PR #598) can still pay. Alone, it leaves the
ownership duplication and increases the number of adapters as more loops become native.
Keep that work separately measurable without making the present seam the long-term
execution architecture.

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

**Recorded SEP 0 exception, approved by John Travers on 2026-11-09 (R4):** SEP 47
may reach main through a small number of coherent phase PRs, each leaving main usable
with green required CI. Keep one architectural proposal and one tracking issue.
Prefer contracts/reference/service loop and MLX retirement (phases 1–2), common native
execution with resident observations/export/groups (phase 3), difficult ports and
device qualification (phase 4), and final deletion/qualification (phase 5);
adjacent milestones may be combined when that makes a clearer review. Use topic
branches `feature/sep-0047-<milestone>` and draft PRs while each is developed. This is
an exception for SEP 47, not a general change to SEP 0.
It is also indexed in [SEP 0's history](sep-0000-process.md#history). While this SEP
is a Draft, its **Implementation** field stays `—`. At implementation start, list
each successive `feature/sep-0047-<milestone>` branch and linked PR in that field, in
delivery order, adding entries as PRs are opened and retaining merged entries. Part II
records each milestone's commits, current owner and checks; the header must not point
only to the latest branch and lose the earlier implementation PRs.

Each merge has targeted tests, manual updates and a Part II report stating the current
production owner, remaining adapters and their removal milestone. Temporary internal
adapters can preserve main's functionality between phases; there is no user-selectable
legacy architecture and every adapter must disappear by Final. The SEP stays
`In progress` across intermediate merges; only the final qualified removal PR sets
`Final` and archives it. Use targeted local checks; the full, slow, coverage and native
gates run in CI under the repository rules.

Concurrent features need not wait for the entire refactor. Before the descriptor
contract lands, they may use the current seam with a recorded migration owner in both
PRs. Afterwards, new features extend the new typed descriptors (§6.2), and their
implementer updates any still-required internal bridge. Each merge preserves its
admitted reference/native scope and tests. Coordinate conflicting ownership changes
before implementation; do not freeze all accepted physics SEPs or allow untracked
old-seam additions to accumulate. SEP 46 is already Final, so its policy is consumed
as current main rather than treated as an indefinitely moving dependency.

The focused optimizations are workstreams of SEP 47, not prerequisite projects or
extra PRs. They can land in an early coherent milestone when ready, without
waiting for the full native migration. Preserve separate commits and the §7.2
attribution measurements so their value and the session's value remain distinguishable.

| Workstream | Delivery and evidence |
|---|---|
| #346 saved-plane copies | Trace file and in-memory output, unit/non-unit weights and checkpoint sharing in phase 1; take straightforward copy elisions early and complete destination export in phase 3. Qualify retained-array safety, exact copies/storage, rounding-bounded fused conversion, saved-plane time and peak live storage, including the reported Linux large-field case. Reconcile historical estimator assumptions with the current allocation ledger, not renewed memory admission. |
| #361 device chunks | Include resource-aware chunk planning in phase 3's backend lowering. Qualify the default candidate, explicit budgets, memory cost and actual CUDA/HIP timing in phase 4 before adopting a changed production device rule. Preserve any independently qualified rule that has already landed. |
| #584 CPU forward transforms | Delivered independently by PR #598 (§3.3). Phase 3 moves its policy into backend lowering unchanged and checks that its measured behaviour is preserved; no probe or adoption decision is part of this SEP. |

1. **Contract, baseline and design readiness.** Inventory capabilities, extension
   cases, mutable owners, event ordering, snapshots and group semantics. Freeze the
   §7 baseline before focused optimizations. Specify typed descriptors and executable
   lifecycle/export conformance tests. Prototype the intended boundary on a cheap run
   and a large field; measure previously unisolated crossings and required work.
   Perform small feasibility probes for wide/off-eigenbasis coupled flows, #255 and
   device event/export lifetimes now, so they inform the contract before broad porting.
   Record the §7.4 readiness outcome and deletion inventory. Choose the clean public
   configuration API and identify callers/tests to change; no migration plan for old
   APIs is required. Performance neutrality is a valid outcome, not a failed phase.
2. **MLX retirement, reference session and service loop.** Retire MLX first under §6.6;
   spend no effort porting its inherited hierarchy. Retain NumPy double/single precision,
   reference experiments and TDSE transactions. Split method/tolerance/observation
   inputs, introduce route variants and implement the reference lifecycle with
   independent numerical algorithms, so that `numpy_ref` no longer inherits
   `StepDriver` (§6.6). Replace the outer numerical driver with the
   service loop and update interfaces, callers and documentation directly. Verify
   event ordering, failure and resume. Land phases 1–2 early as the usable extension
   contract for feature SEPs; do not wait for expensive native ports. Temporary
   internal bridges keep existing native execution working and are tracked for removal.
3. **Common native session, resident observations, export and groups.** Integrate
   SEP 46 policy, backend-owned lowering and allocation records. Adopt the existing
   native Adams/RK engines and move diagonal-flow low-storage replay into the session.
   Keep controller, history, boundary state and actual work native across advances.
   Prepare statistics/monitor work, bounded delivery and destination export; implement
   coarse native fixed-schedule groups through explicit member ownership. Port the
   boundary monitor's decision state and records and every periodic-capable statistic
   to the native session (§6.4), the transverse judgement under a timebox. Complete
   #346's export work, implement #361's chunk policy and move PR #598's transform
   policy into lowering unchanged. Qualify common native CPU paths against the reference,
   including dense output, retained arrays, failed consumers, partially advanced
   groups, token stopping and independent snapshots. Exercise a representative device
   path early enough to detect contract defects; the complete device matrix is phase 4.
   Record crossings, live buffers, traffic and paired timings. No approximate statistics
   or batch precision change is included.
4. **Remaining native ports and device qualification.** Complete the built-in cases
   in §6.3: wide/off-eigenbasis coupled flows, remaining host-formed representations
   and the CPU carrier-table gap #255. Complete every admitted CUDA/HIP session port
   and actual-device witness, including coupled flows, groups, low-storage replay and
   #361's device-policy qualification. Timebox each substantial port with a stated
   deliverable and effort budget when assigning it. At the checkpoint report achieved
   capability, evidence, the remaining obstacle/cost and alternatives. If it overruns,
   bring a concrete scope or sequencing decision to the maintainer; do not silently
   drop a capability, weaken tests, retain a permanent legacy executor or redesign the
   common architecture solely to hide the overrun. Budgets are decision checkpoints,
   not automatic scope cuts. A demonstrated contract defect still warrants correction.
5. **Removal and release qualification.** Delete transitional native driver
   inheritance, mirrored records, private lockstep access and replaced contract tables.
   Confirm all retained production paths use the new contract and every phase 4
   CPU/device obligation has evidence. Complete paired performance and structural
   reports, obtain explicit decisions for retained regressions, and update rules/manuals.
   Record the disposition of #346 and #361, closing issues only on their delivered
   criteria. Archive at `Final`
   only when the replacement is the production architecture and no required path
   depends on the old native driver. Outstanding scope changes require an explicit
   amendment; reaching a timebox does not itself satisfy Final.

### 12. Open questions

John Travers resolved the seven architectural decisions on 2026-11-09. These are the
agreed design direction, confirmed at acceptance.

1. **Ownership:** the backend owns the complete numerical session, including
   observations, boundaries, accounting and export preparation. Python retains science
   construction and external transactions (§§5–6.1).
2. **Representations:** Python constructs and validates a numerical description that
   retains mathematical structure. The backend owns packing, materialization, evaluation
   and execution choices, without silently changing scientific approximations (§6.2).
3. **Layouts:** internal layouts may differ by backend, with canonical host boundaries,
   explicit packing costs and numerical-equivalence tests (§6.2).
4. **Python extensions, refined after R5/R6 and the scope decision below:** the host
   session serves NumPy in both retained precisions;
   native exposes its complete session, with no host-over-native executor. Keep custom
   Python controllers and benchmark comparisons reference-only; port built-in production
   capabilities including coupled flows and carrier tables. Remove `StepDriver`
   inheritance; extension APIs may change under decision 6 (§§6.3, 6.6).
5. **Groups:** include explicit group ownership and coarse native fixed-schedule
   advancement. General fused kernels and adaptive ensembles remain separate work
   (§6.7).
6. **API freedom:** freely break public and internal APIs wherever that produces cleaner
   code or structure. Keep solprop working and tests passing, update documentation to
   the new API, and list changes in the changelog. No compatibility layers, deprecation
   periods, migration paths or migration documentation are required (§9).
7. **Success:** structural simplification is a sufficient primary benefit, with mandatory
   correctness and deletion requirements. Target performance neutrality or improvement;
   a repeatable per-case regression of 5% beyond measurement uncertainty triggers an
   explicit decision under §7.3, not automatic acceptance or rejection.

John Travers also resolved the review arbitrations on 2026-11-09:

8. **Delivery (R4):** permit coherent phased merges as the SEP-specific exception in
   §11, with main working after each and temporary adapters removed by Final.
9. **Restart (R9):** record retention/restoration/reset decisions and reasons; warn
   when expected Adams history was omitted. Defer full RK continuation to #596 (§5.5).
10. **Control (R12):** use a native stop token checked between attempts, finishing the
    current attempt safely. Retain bounded returns for Python signals. Serve snapshot
    requests independently of `step_planes=1` at eligible accepted endpoints (§5.5).

John Travers further resolved implementation scope and sequencing on 2026-11-09:

11. **MLX:** retire it early, with no adapter or numerical migration/qualification
    effort. Preserve NumPy single precision. Julia replacement is separate future work
    (§6.6); this supersedes the earlier choice to retain an MLX host session.
12. **Included work:** fold #346 and #361 into SEP 47, without separate prerequisite
    PRs, and separate kernel/copy gains from session gains (§7.2). Revised at
    acceptance: #584 is delivered independently by PR #598, whose policy phase 3
    moves into lowering unchanged (§3.3).
13. **Readiness and sequencing:** conceptual simplification is sufficient motivation.
    Phase 1 checks the concrete contract's readiness, not a minimum speedup (§7.4).
    Land the reference/descriptors/service loop early; probe hard cases early but do
    their bulk ports later, with timeboxes that prompt explicit decisions rather than
    silently dropping scope (§11).

John Travers made five further decisions at acceptance on 2026-11-09:

14. **Device scope:** preserve each capability's current per-vendor status; the only
    new device qualification is CUDA low-storage replay (§6.3).
15. **Statistics:** every periodic-capable statistic is computed resident in the native
    session, with no Python finishing (§6.4).
16. **Monitor:** the native session owns every monitor decision state and record;
    Python only formats them. Exact identity covers scalar decision logic on recorded
    readings (§§4.3, 6.4).
17. **Transition:** build the independent reference session in phase 2, sharing only
    stateless numerical modules with the legacy driver, which remains solely as the
    temporary native bridge (§6.6).
18. **Clarifications:** running checkpoints carry pending history and rows (§5.5),
    monitor restoration keeps today's reconstruction (§5.5), calibration keeps an
    explicit timing probe (§6.6), groups are sequences of member problems served by
    the group service loop (§6.7), and attribution uses A/B/C as redefined in §7.2.

No maintainer arbitration requested by this review remains open. Phase 1 determines
the exact public configuration API, descriptor fields, packet capacities and
native class decomposition under these decisions. Whether a particular GPU benefits
from conversion before transfer, a larger graph or a device controller is deferred to
phases 3–4 measurements. These choices do not change scientific semantics. A new
approximation, output format, dependency or departure from the agreed architecture
returns to review; an API break within the agreed design does not by itself require
another decision.

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
6. [SEP 46](archive/sep-0046-machine-calibration.md) and its merged implementation
   [PR #582](https://github.com/LupoLab/solprop/pull/582). Machine policy and workspace
   choices; in progress at original drafting, Final at this review response.
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
10. [SEP 21](archive/sep-0021-absorbing-boundaries.md),
    [SEP 29](archive/sep-0029-compiled-backend-distribution.md) and
    [SEP 33](archive/sep-0033-single-precision-mlx.md). Boundary controller state,
    explicit helper ownership and the independent single-precision/MLX contracts.

## Part II — Implementation record

### Milestone 1 (phases 1–2) — in progress

Branch `feature/sep-0047-contract`, started 2026-11-09 by Opus 5.5 (AI) as manager,
with work packages developed on sub-branches and reviewed before merging. Phase reports
follow as the packages land.

## Part III — History

- 2026-11-08: Drafted following the analysis of current main, PRs #576, #585, #586,
  #582 and #559, and the performance-labelled issues. No implementation or fresh
  performance measurements are claimed. Tracking issue:
  [#588](https://github.com/LupoLab/solprop/issues/588).
- 2026-11-09: John Travers accepted all seven architectural recommendations recorded
  in §12. On API compatibility he clarified that the project is young and has no
  downstream users: APIs may be freely broken wherever this gives cleaner code or
  structure, provided solprop keeps working and its tests pass. Documentation should
  describe the new API, and a simple changelog listing is sufficient; migration paths
  and migration documentation are unnecessary. Revised the interface, extension,
  verification, documentation, compatibility and phase requirements accordingly. This
  records maintainer design decisions, not a completed review of the whole proposal;
  the SEP remains Draft pending the reviews required by SEP 0.
- 2026-11-09: AI review by Opus 5.5 at `d6276194`, recorded below; Part I unchanged.
- 2026-11-09: Responded to R1–R17 and revised Part I after John Travers resolved
  the delivery, native-interface, restart and control arbitrations. Updated against
  main after SEP 46 completed; deferred full RK continuation to #596. The original
  review and the response remain separately readable below.
- 2026-11-09: AI re-review by Opus 5.5 of the revision at `a6a21302`, recorded below;
  Part I unchanged.
- 2026-11-09: Responded to A1–A6 without reopening the maintainer's decisions.
  Clarified checkpoint representation and export reuse, made every native port's
  CUDA/HIP disposition explicit, corrected the SEP 46 and frequent-event evidence,
  completed structural counters and indexed the phased-merge exception in SEP 0.
- 2026-11-09: Recorded the maintainer's MLX retirement, included optimization
  workstreams and revised sequencing. Clarified phase 1 as design readiness with no
  minimum speedup; the full decision record below supersedes earlier scope choices.
- 2026-11-09: Implementation-readiness check by Opus 5.5 against main at `0901bf6f`,
  before implementation. John Travers decided the device scope, statistics, monitor
  and transition questions it raised; Part I revised for those decisions and for the
  clarifications listed in the record below. #584 is now delivered by PR #598.
- 2026-11-09: Accepted by John Travers (human review) in #589.
- 2026-11-09: Implementation started on `feature/sep-0047-contract` (phases 1–2).

### AI review — 2026-11-09 — Opus 5.5

**Reviewer:** Opus 5.5 (AI). I read all of Part I at PR #589 commit `d6276194` and
checked it against main at `e555fcf5`. The code read was the native adapter
(`native/src/solprop_native/backend.py`, `batch.py`, `__init__.py`); the shared driver,
low-storage march and lockstep runner (`src/solprop/backends/driver.py`,
`low_storage.py`, `batch.py`); the run loop, output, snapshot and resume paths
(`driver/run.py`, `output/schema.py`, `output/hdf5.py`, `output/snapshots.py`,
`driver/simulation.py`); the MLX backend; kernel contract §§8–12; backends.md §1;
SEPs 39 and 45 and their reports; and every issue cited in §3.3. I checked the figures
in §3.2 against the benchmark ledger and SEPs 39 and 45, and they match. I did not
rerun them, and I re-measured nothing. File and line references are to `e555fcf5`.

I take the maintainer's seven decisions in §12 as given. The points below are about
delivering and verifying those decisions, not about reopening them. The direction is
sound. The proposal is careful about numerical identity, it avoids reviving memory
admission, and it refuses to count a wrapper as success. Six gaps stand out:

- Part I does not yet name the concrete cases that still cross into Python per step.
- It does not keep the bit-identical native/reference decisions that SEPs 39 and 45
  established.
- It treats the boundary monitor as observation, although it feeds the controller.
- It justifies a host session over native kernels by extension cases that are not
  supported features.
- Its delivery plan, one branch that reaches main only at `Final`, is risky at this
  size.
- It does not say how draft feature SEPs extend the new descriptors.

This review does not recommend acceptance yet. Every point stays open for the drafting
agent's response and the maintainer's decisions on R4–R6, R9 and R12.

1. **R1 — High: name the remaining crossings as the concrete use cases (§§2, 3.2,
   7.2).** SEP 0 asks the motivation to name a concrete use case. Part I argues from
   structure, and its evidence table mostly shows gains already won. After SEP 45,
   `gnlse_soliton` takes 11 native calls for 691 steps, with 4.2 % of the run outside
   them. `argon_blueshift_envelope` spends 3.3 % outside native calls, and 2.1 % of the
   run is the replayed `monitor.step`. What still returns to Python per accepted step,
   or close to it:
   - **Periodic statistics with `period = 1`.** `_bounds` sets the host step to
     `period - accepted % period` (`backend.py:3053-3081`), so every call returns
     after one step. Each row then goes through `StepDriver._record_step_stats` →
     `stats()`, one native reduction and one host copy per row
     (`driver.py:804-821`). `NativeBackend` overrides none of the three step-statistic
     hooks.
   - **A transverse-edge run.** It disables the native watch (`backend.py:3032-3041`),
     and the Adams march stops at each read (`backend.py:317-319`). A collar edge reads
     every step.
   - **Low-storage replay.** One native call and Python booking per step
     (`low_storage.py:562-599`).
   - **Lockstep batches,** on every backend. Native forces its Adams members onto the
     Python replay (`native/.../batch.py:67-108`).
   - **Snapshots.** They require `step_planes = 1` (`run.py:800`), so a run with
     snapshots enabled observes every accepted step.
   - **The host-march fallbacks of R5.**
   - **A saved plane** builds about three full host arrays: the device-to-host copy;
     `power_amplitude`, which copies even when every weight is 1
     (`fourier.py:393-399`); and the optics conversion (`schema.py:4903-4935`).
     `kerr_beam_384` spends 2.9 s in plane saves (#346).

   **Requested change:** name these in §2 as the cases the proposal exists for. Add a
   period-1 statistics run, a collar-edge run and a snapshot-every-step run to the
   §7.2 matrix. Give each a §4.2 bound from the existing measurements. That also ranks
   the phases by payoff and gives §7.3 something to test structurally (R15).

2. **R2 — High: keep bit-identical decisions between native and reference (§§4.3,
   6.3, 7.1).** Kernel contract §§11–12 compile the march decisions strictly, with no
   contraction, so that "on one build the native and reference marches read the same
   tables and norms and take the same steps". `tests/driver/test_native_rk_march.py`
   and the native Adams tests compare bit for bit, and #135's strict-FMA witnesses hold
   the property in the built library. §4.3 says execution changes "need not preserve
   accepted step counts or bits". §7.1 then falls back to SEP 25 budgets and to
   envelopes for adaptive runs. The session moves landings, statistics scheduling,
   monitor prediction, dense observation and event ordering into C++. An off-by-one or
   a reordering there would hide inside an accuracy envelope, and an exact
   step-sequence comparison would catch it.

   **Requested change:** state that the native session's control decisions are strict
   transcriptions of the reference: proposals, acceptance, landings, monitor verdicts,
   statistics cadence and event order. Given the same norms, the native session must
   reproduce the reference's attempt sequence exactly on one build. Matched
   trajectories stay the primary native-against-reference qualification, with
   envelopes only where the trial's own arithmetic differs. Add the new scheduler code
   to the strict-file witnesses. §9 should also count the cost: every piece of
   bookkeeping moved into C++ becomes a second implementation of the Python reference's
   logic, kept in step by these tests.

3. **R3 — High: the boundary monitor is controller input, and a production
   cross-check disappears (§§5.3, 6.4, 6.8).** The monitor caps each proposal: the
   quiet interval, the transverse-edge ceiling and the split collar's crossing cap,
   in that order (kernel contract §12, attempt step 1). Its stop and warning verdicts
   end runs. Today `_verify_watch_of` (`backend.py:2820-2840`) holds the Python
   monitor's booking to the native march's prediction and raises `BackendError` on any
   difference. Part I groups boundaries with "observation, boundary and export work"
   and deletes the Python replay (§6.8 row 3). Deleting it is right, since it is most
   of the residual cost measured by SEP 45. But:
   - (a) Boundary state belongs to the controller state, ordered against the proposal
     as §12 orders it, and it must enter checkpoints (the monitor resumes at
     `run.py:566-574`).
   - (b) Removing the runtime check removes an oracle. Say what replaces it: a
     lifecycle-suite test that drives the same watch policy and recorded readings
     through the native and reference monitors and compares verdicts and caps
     exactly. Do not keep a production shadow.

4. **R4 — High: one branch that reaches main only at `Final` (§11).** Under SEP 0
   the implementation merges when the SEP is `Final`. Phases 2–5 rewrite the hottest
   files: `run.py` (1651 lines), `backends/driver.py` (2995), native `backend.py`
   (4994), `batch.py`, `low_storage.py`, the run paths of `simulation.py`, the frozen
   surface tests and every backend. Meanwhile SEP 46 (#582) is in progress and SEP 17
   is accepted. SEPs 1, 2, 5, 7–11 and 32 will add responses, statistics and state
   through today's `KernelProblem` extension procedure. The branch would live for weeks
   and conflict continuously, and phase reports would qualify a base that keeps moving.

   **Requested change, for the maintainer:** choose one of two options.
   - (a) Split the proposal. SEP 47 keeps the contract, typed descriptors, the
     reference/host session and the service loop that replaces `run_problem` (phases
     1–2). A follow-on SEP takes the native session, resident observations, export,
     groups and deletion (phases 3–5). Each has its own branch and PR.
   - (b) Grant a recorded SEP 0 exception. Each phase merges when main still works and
     no dual architecture is visible to users.

   Either way, state the policy for concurrent SEPs: which land on the old seam and are
   ported by this branch, and which wait for it.

   **Maintainer decision — John Travers, 2026-11-09:** allow phased merges for SEP 47,
   as a narrowly recorded SEP 0 exception. Keep one proposal; each coherent phase PR
   leaves main usable. Temporary internal adapters are allowed and removed by Final.
   The concurrent-feature policy is now in §11.

5. **R5 — High: the scope of "every built-in combination currently admitted on
   native" (§6.3, phase 3).** These configurations run the Python march over native
   kernels today:
   - from the RK march's refusals (`backend.py:3304-3336`): a `PIController` subclass
     that overrides `scale`, `accepts`, `propose`, `rejection_factor`, `start_factor`
     or `equivalent_norm`, and stage rows formed on the host;
   - from the Adams march's refusals (`backend.py:2914-2952`, `517-539`):
     - the `ComparisonMultistep` general paths (PECE, plain Adams–Bashforth, matched
       pairs), which `guide/numerics/adams.md:146-151` documents as "for the
       benchmarks only";
     - the host engine, for coupled flows off the eigenbasis route, wider than
       `ADAMS_MAX_CHANNELS = 16`, non-exponential, batched or with untransported
       compensation;
     - representations the stage tables do not read;
     - a controller with its own norm scale;
     - the carrier frame on a z table (#255).

   Porting all of these to C++ is substantial, and several have no production use.
   Under decision 6, refusing some of them on native by name, while keeping them on
   `numpy_ref`, is the cleaner design.

   **Requested change:** tabulate each fallback in Part I with a disposition (port in
   phase 3, refuse on native and keep on the reference, or retire) and its qualifying
   test. I recommend refusing `ComparisonMultistep` and controller subclasses on
   native. Port #255 and the wide or off-eigenbasis coupled flows only if a production
   case needs them.

   **Maintainer decision — John Travers, 2026-11-09 (R5/R6):** native exposes only its
   complete session. Custom Python controllers and benchmark-only comparison methods
   stay on the reference. Port built-in production capabilities, including coupled
   flows and carrier tables; they are not dropped for lack of an existing use case.
   The disposition and qualification inventory is now in §6.3.

6. **R6 — High: the host-owned session over native kernels (§6.6).** §6.6 says
   custom controller or norm code needs Python during propagation. No such extension
   is documented. `PIController` is a frozen dataclass, `StepperSettings.controller` is
   typed to it, and overriding subclasses appear only in tests
   (`test_native_rk_admission.py:41`, `test_resident_adams.py:402`,
   `test_native_rk_rows.py:192`). TDSE is admitted on `numpy_ref` alone
   (`registry.py:487-488`). MLX has no "opaque state operations" interface. It is
   `MlxBackend(NumpySingleBackend(NumpyRefBackend(StepDriver)))`. It overrides
   `_observed_field`, `_apply_mask` and `_finish`, and it brings its own Adams engine,
   low-storage engine and lockstep runner. A "kernel-executor interface where the
   selected backend already supports that combination" would therefore exist on native
   only to run test-only controller subclasses. It would keep a second native
   integration surface: trial step, finish, mask, statistics and boundary energies,
   which is the substance of today's `DRIVER_SURFACE`.

   **Requested change, for the maintainer:** decide whether native implements the
   host kernel-executor protocol at all. I recommend not. Native would expose only
   preparation and its session. The host session would serve `numpy_ref` and MLX.
   Controller variation would become policy data or be refused on native. Correct
   §6.6's description of the extension cases and of MLX. Say what becomes of
   `StepDriver` itself. §§6.8 and 9 replace "the protected `StepDriver` surface", but
   the reference and MLX keep most of `backends/driver.py` and `multistep.py`, so
   Part I should say whether that hierarchy is the host session, refactored and
   renamed, or is replaced.

   **Maintainer decision — John Travers, 2026-11-09 (R5/R6):** no host-over-native
   kernel executor. NumPy and MLX retain a host session; its implementation replaces
   the inherited `StepDriver` architecture (§6.6).

7. **R7 — Medium: yield bounds are exact no-ops today; keep that (§§5.1, 7.1).**
   Kernel contract §11: "a return hands the state back unrounded, so the bound moves
   no number, and a test holds a bound of one attempt, of seven and of none to one set
   of bits." Progress reporters are held to exact no-ops too (backends.md §1). §7.1's
   wording, that delivery frequency "must not alter scientific policy", is weaker than
   that.

   **Requested change:** state that `work_limit`, `yield_deadline`, event capacity,
   acknowledgment timing and reporters leave the trajectory bit-identical on one build.
   Carry the existing test over to the session API, including a deadline that lands at
   nondeterministic points.

8. **R8 — Medium: the batch axis and group failures (§§6.2, 6.7; #110).** #110 asks
   the SEP to say what becomes of `n_batch` in the state, of `batch_params` and of
   batch runs' saved fields, weighed on measured GPU throughput. §6.2 gives the
   single-trajectory boundary as `(n_modes, n_pol, n_omega)`. But conventions.md §6
   and kernel contract §§1 and 9 fix a leading `n_batch` in the state, in observations
   and in every statistic, and the checkpoint schema stores `n_batch` too
   (`schema.py:2575`).

   **Requested change:** say whether `n_batch` leaves the problem, observations and
   statistics; what `batch_params` becomes; and whether files written with a leading
   `n_batch` stay readable and new ones still carry it.

   Also correct a mismatch in §6.7. It says a stopped member does not stop unrelated
   members, but today only `BoundaryError` and `BoundaryWarning` are per-member. A
   non-finite norm, a floor, a trial exception or an in-flight member stops the whole
   batch (`backends/batch.py:33-61, 144, 183, 193-203`), after earlier members have
   already advanced that step. Say whether the group preserves this or changes it,
   and list a change as a behaviour change.

9. **R9 — Medium: restart as claimed and as implemented (§5.5).** Today:
   - The RK march exports no snapshot. `set_initial` resets the proposal, norms and
     history on resume (`driver.py:1251-1256`).
   - Low-storage keeps no history.
   - With `resume_history=None`, an Adams snapshot over `MARCH_SNAPSHOT_AUTO_BYTES`
     (256 MiB, `run.py:115, 1207-1222`) is dropped with an INFO log. The resume then
     restarts the history silently (`simulation.py:4118-4121`); on refusal it warns.
   - Nothing records the choice in the output.

   §5.5 says history "must never silently reset" and that the choice is "recorded",
   while also "preserving the existing bounded-export policy". Today's behaviour
   breaks the first two.

   **Requested change, for the maintainer:** decide which wins. I recommend recording
   the restart in the resumed run's metadata and warning when the size policy dropped
   the snapshot, listed as a behaviour change. Also decide whether the RK controller's
   few scalars join the checkpoint, which would make RK resumes exact continuations,
   or stay out of scope.

   **Maintainer decision — John Travers, 2026-11-09:** record restart choices and their
   reasons, and warn once on resume when expected Adams history was omitted. Defer
   full RK continuation to separately qualified work (#596); controller scalars alone
   do not establish it (§5.5).

10. **R10 — Medium: extending the new descriptors (§§6.2, 8, 9).** Kernel contract
    §10 is the procedure that the draft feature SEPs assume. Part I replaces
    `KernelProblem` without giving its successor's procedure. SEPs 7 (population and
    pump dynamics) and 32 (Maxwell–Bloch populations) may need state that evolves in
    z beside the field and the method history. The session's state is defined as
    field plus history.

    **Requested change:** specify how a route variant, response, statistic or
    auxiliary state component is added: which descriptor, which validation, which
    capability report, and which reference and native tests. If auxiliary state is out
    of scope, say that such a SEP amends this contract. Also say what replaces
    `KERNEL_SCHEMA_VERSION` and native's `_check_seam`
    (`native/.../__init__.py:930-975`), given ADR-068's unversioned seam.

11. **R11 — Medium: details of export ownership (§5.4).**
    - (a) Part of #346 needs no session: the `power_amplitude` copy at unit weights,
      and fusing the conjugate with the `(-1)^k` pass on the host. Let those land
      independently, like #584, so that §7.3 measures what the session itself removes.
    - (b) Single precision widens with `astype(complex128) * scale`
      (`numpy_ref/single.py:349`, `mlx_gpu/backend.py:597`). The one conversion pass
      must include the precision scale.
    - (c) Spatial routes skip the true-power conversion (`run.py:281-303`), so the
      conversion spec is per route.
    - (d) A plane observer and the failure checkpoint re-observe the state
      (`run.py:832-851, 1274-1282`). The single pending field-bearing observation must
      serve every consumer at a plane without another transfer.
    - (e) The optics conjugation is an output-layer rule (ADR-017). Say where it lives
      after the move: one specification consumed by native and by the reference, with
      resume's inverse.

12. **R12 — Medium: stop latency and snapshots (§§5.1, 5.5).** Today:
    - The stop flag is polled before each native call.
    - Calls are bounded by 256 attempts or 0.1 s (`backend.py:2859-2871`).
    - Signals are acted on only when a call returns, because the GIL is released.
    - The GUI's stop is a `threading.Event` set by a thread that polls for a stop
      file.

    One attempt of a large Cartesian run can exceed 0.1 s. Both options in §5.5 are
    small, so choose now.

    **Requested change, for the maintainer:** I recommend a native atomic flag, set by
    the Python stop path and by the signal handlers. Native checks it between attempts
    and between stages, and the call bound stays as a fallback. State the latency in
    attempts and stages.

    Second, snapshots are serviced after each committed plane, and so they require
    `step_planes = 1`. As a control operation ("checkpoint at the next eligible
    boundary"), they could drop that requirement and the per-step return it forces.
    Say whether they will.

    **Maintainer decision — John Travers, 2026-11-09:** adopt a native stop token
    checked between attempts, finishing the current attempt safely, with bounded
    returns for Python signals. Do not interrupt in-place stages. Snapshot requests
    become independent of `step_planes=1`, serviced at eligible accepted endpoints
    after valid checkpoint/history persistence (§5.5).

13. **R13 — Medium: specify event order in Part I (§5.3).** "One documented ordering
    inherited from the existing semantics" defers the conformance suite's
    specification. The existing order is spread across files:
    - statistics rows after `_finish` and the advance of z, at different points in
      the adaptive and replayed marches (backends.md §1);
    - the monitor reading at that same point, then the natural stop;
    - at a plane: the output commit, then snapshot service, then the plane observer,
      then the reporter (`run.py:606-642`);
    - a warning from `monitor.finish` after the last plane, which replaces "complete"
      with "failed" and writes a checkpoint (`run.py:647`).

    **Requested change:** tabulate this order in §5.3, because §7.1's exact
    event-sequence assertions depend on it.

14. **R14 — Medium: keep the statistics channel's rules (§6.4).** Backends.md §1 has
    two rules that §6.4 should name. First, one implementation of every statistic: a
    row that lands on a plane equals the plane's value to rounding. Second, the call
    site rule. The device storage hooks for resident rows already exist and are
    unused. The native session should keep rows resident and copy them once per
    delivery. This settles the ownership half of #107 and should say so.

15. **R15 — Low: performance criteria (§7.3).** Python's share after SEP 45 is about
    1–4 % on the measured cases, so most expected gains fall below the 5 %
    investigation threshold. A uniform 3 % regression would pass every per-case check.

    **Requested change:** add an aggregate non-regression check, a geometric mean over
    the matrix no slower than baseline beyond its uncertainty, beside the per-case
    rule. Add structural count targets that noise cannot hide, for example native
    returns per accepted step falling to zero for period-1 statistics, collar edges and
    low-storage replay. Choose windows that contain the planes, rows and edge reads
    whose cost is being measured. A ten-step window may contain none.

16. **R16 — Low: helper ownership (§6.6).** The existing `Helpers` record (`PLAIN`
    against `NATIVE`, backends.md §1, `tests/backends/test_helper_ownership.py`) is
    how the oracle avoids running the compiled code it checks. Cite it in §6.6 and keep
    the mechanism and its test. MLX currently takes `NATIVE` helpers. Say what it uses
    under the host session.

17. **R17 — Low: corrections and cross-references.**
    - **Requires** and §3 could add SEPs 21 (boundary monitor), 22 and 24 (memory-lean
      execution, low-storage), 29 (distribution, helper ownership) and 33 (single
      precision and MLX). §13 cites several of them already.
    - In §3.2, the "1,910 to 17" figure comes from SEP 45 phase 1's in-march
      boundary checks under Adams. That 2.1 % of the remaining 3.3 % was the replayed
      `monitor.step` supports §6.8 row 3 directly. Worth saying.
    - In §6.2, "the current requirement for one backend-independent internal field
      layout" is not written anywhere. Backends.md §1 already makes layouts private.
      The real constraint is the shared host march operating on canonical arrays
      through `StepDriver`. Reword it.
    - In §6.6, "custom controller or norm code" and MLX's "existing opaque state
      operations" do not name existing interfaces (R6).

**Questions for the maintainer** (decisions to record beside the points):
R4 (split the SEP or allow per-phase merges), R5 (which host fallbacks to refuse on
native), R6 (whether native implements the host kernel-executor protocol), R9 (history
restart reporting and RK checkpoint continuity) and R12 (stop mechanism; snapshots
without `step_planes = 1`).

### Response to review — 2026-11-09 — GPT-6

Revised Part I against the review and main at `eee57c65`, including the now-Final
SEP 46. The original review is preserved above; separately labelled maintainer
decisions record the arbitrations. This is a drafting response, not a new independent
review or formal acceptance. No runtime change or fresh performance measurement is
claimed. Source checks included the current native admission/engine paths, host RK
and Adams marches, statistics/monitor ordering, group failures, export/resume/snapshot
paths, MLX/helper ownership and strict-control tests.

1. **R1 — Accepted, with limits on available timing evidence.** §§2 and 3.2 now name
   period-1 rows, transverse collars, low-storage, lockstep, snapshots, host fallbacks
   and saved-plane copies. §§3.2/4.2 give historical removable-time ceilings where
   measurements support them: 1.034× for all Adams outside-march time, 1.021× for its
   replayed monitor, 1.044× for all RK outside-march time and 1.088× for all Kerr saves.
   These are generous ceilings including useful work, not predictions. Isolated
   fractions for the other paths do not exist in the examined reports; phase 1 must
   measure them rather than invent a bound. §7.2 now exercises each event-rich case,
   including frequent and sparse independent snapshots.

2. **R2 — Accepted for identical-input control; clarified the numerical premise.**
   §§4.3 and 7.1 require exact proposal, acceptance, landing, cap, cadence and event
   traces on identical numerical inputs, strict arithmetic witnesses, and independent
   NumPy field qualification. `tests/driver/test_native_rk_march.py` explicitly compares
   a host march and native march over the same native norms and staged rows. It does
   not establish bitwise agreement between independent NumPy and native trials.
   Test-only scripted control replay replaces the retired host-over-native production
   path. SEP 25 still governs genuinely different arithmetic. §9 now counts the
   maintenance cost of duplicate control implementations.

3. **R3 — Accepted.** §§5.3, 5.5 and 6.4 make boundary state controller input, preserve
   cap order and restart reconstruction, and replace `_verify_watch_of` with exact
   independent watch-policy tests on identical readings. No production shadow is
   retained. Warning barriers resume at the exact interrupted event phase.

4. **R4 — Resolved by the maintainer; implemented in the proposal.** §11 records the
   SEP-specific phased-merge exception, a suggested four-PR grouping of five phases,
   usable main at every merge, adapter removal obligations and coordination of new
   features before/after the descriptor contract lands. SEP 46 is already Final.

5. **R5 — Resolved by the maintainer; explicit inventory added.** §6.3 lists each
   fallback's port/refuse/retire disposition and qualifying witness. Controller
   overrides and all benchmark comparison methods become reference-only; private
   forcing switches retire. Production coupled flows and representations are ported,
   including wide/off-eigenbasis cases and the CPU carrier-table gap. Unlike the
   review's conditional suggestion, retaining these capabilities is not contingent
   on finding a production timing case. Existing GPU and coupled-group refusals are
   preserved unless separately qualified. The native admission changes are explicit
   in §9 and must be in the changelog.

6. **R6 — Resolved by the maintainer; original rationale corrected.** §6.6 removes
   the host-over-native executor entirely, identifies controller overrides as test
   experiments and TDSE as already reference-only, and accurately describes MLX's
   inherited engines. The host session extracts retained numerical code into private
   components; `StepDriver` and its inheritance hierarchy disappear by Final. Useful
   NumPy algorithms can remain without recreating the native integration surface.

7. **R7 — Accepted as an exact lifecycle contract.** §§5.1 and 7.1 forbid changes to
   numerical state, caches or arithmetic plans caused solely by yields. Deterministic
   CPU witnesses compare one/seven/large bounds, capacities, deadlines, acknowledgment
   delays and reporters bit for bit. Scripted identical outcomes test every executor.
   Independent nondeterministic reductions retain SEP 25's scope; transport itself
   receives no rounding allowance. Actual stop requests are compared under the same
   external itinerary, not mistaken for observational reporting.

8. **R8 — Accepted.** §§6.2 and 6.7 remove nominal runtime `n_batch` and replace
   `batch_params` with typed members. Ordinary saved fields retain their present
   shape, and checkpoint serialization adapts the existing size-one batch axis without
   silently squeezing larger axes. Group failure policy is preserved explicitly:
   boundary outcomes can remain per-member, but numerical/in-flight failures stop
   the group with actual member positions, not invented rollback or synchronization.

9. **R9 — Resolved by the maintainer; reporting is an intentional behaviour change.**
   §5.5 distinguishes current RK resets, history-free low-storage and bounded Adams
   retention from the new retained/omitted/restored/reset records and reasons. It adds
   one warning on resume for unexpectedly omitted Adams history, keeps explicit
   history-off policy quiet, and handles legacy metadata honestly. Full RK continuation
   is tracked separately in #596. The review's suggestion that a few scalars alone
   make continuation exact was not adopted: recycled stages and field conversion also
   require qualification.

10. **R10 — Accepted; version premise corrected.** §6.2 defines the route, response,
    statistic and auxiliary-state extension procedure, including admission, validation,
    reference/native qualification and MLX refusals. New evolving physics still needs
    its own specified rollback/error/restart contract. Frozen descriptor/session
    manifests replace old structural seam checks. `KERNEL_SCHEMA_VERSION` has already
    been removed under ADR-068; no replacement version is introduced. §11 coordinates
    feature SEPs across the transition.

11. **R11 — Accepted (a/b/c/e); (d) qualified by representation and position.** §§2,
    5.4 and 7.2 permit independent copy fixes, include precision scale and spatial-route
    rules, and leave one output-owned conversion recipe with a resume inverse.
    Identical consumers share one detached export. An internal endpoint checkpoint is
    not always the same representation or position as a saved plane: it can require
    a separate payload. The proposal neither promises lossless recovery through rounded
    inverse normalization nor retains an unconditional raw backup to meet an artificial
    one-transfer rule.

12. **R12 — Resolved by the maintainer with safety/latency corrections.** §5.5 chooses
    an atomic thread/GUI token, checked between attempts, and independent snapshot
    requests with persisted checkpoints/history before copying. It declines mid-stage
    interruption of in-place work. Python signal handlers run when the main thread can
    execute Python, not immediately because native released the GIL; bounded returns
    remain necessary. The 256-attempt/0.1 s bounds and possible one-attempt overrun are
    explicit, along with additional checkpoint/I/O latency. The official Python signal
    documentation was checked; no new process signal machinery is assumed.

13. **R13 — Accepted.** §5.3 now tabulates startup, method-specific attempt/commit
    order, monitor-before-statistics, natural stops, planes, independent snapshots and
    final-monitor outcomes. It specifies warning continuation, dense positions, primary
    versus checkpoint failure and acknowledgment barriers. These are executable
    conformance requirements, not a deferral to an unspecified future ordering.

14. **R14 — Accepted.** §6.4 preserves one statistic implementation per backend and
    coincident-result reuse, sampling after finish/position commit, monitor ordering
    and actual attempt indices. Native rows remain resident until bounded bulk
    delivery. This settles #107's ownership design without prematurely closing its
    implementation issue.

15. **R15 — Accepted.** §7.3 adds aggregate geometric-mean non-regression with declared
    weighting and repeated-pair/baseline noise, beside the 5% per-case trigger. A
    uniform 3% slowdown is not automatically acceptable. Structural counters distinguish
    required service returns from those caused solely by statistics, collars or
    low-storage iteration; the latter must disappear. §7.2 requires measured windows
    to contain the events whose costs they claim to test.

16. **R16 — Accepted.** §6.6 retains explicit `Helpers` selection and its ownership
    test: NumPy `PLAIN`, native compiled helpers, MLX `NATIVE` when installed and
    `PLAIN` in the supported no-native environment. Scalar/table helpers do not justify
    a second native propagation interface.

17. **R17 — Accepted with one source correction.** Added SEPs 21/22/24/29/33 to
    Requires, corrected the Adams attribution and monitor fraction, updated SEP 46's
    status/links, and corrected the extension/MLX descriptions. Backends.md already
    makes layouts private, as the review says; however, `native/AGENTS.md` really does
    forbid backend-dependent field layout. §6.2 now names that specific restriction
    and shared-march coupling rather than implying the public backend lacks opacity.

The maintainer arbitrations are resolved; the revised whole proposal remains Draft,
In review, for another review before acceptance. Implementation and performance
qualification are still future work under §§7 and 11.

Documentation validation: `git diff --check`, codespell on changed documentation,
`pytest tests/test_seps.py` (342 passed, three expected process-SEP skips), and
`poe docs` with Sphinx warnings treated as errors passed in the supported lightweight
environment. No native build, parity qualification, new benchmark or full test gate
was run for this documentation-only response.

### AI re-review — 2026-11-09 — Opus 5.5

**Reviewer:** Opus 5.5 (AI). I re-read all of Part I at `a6a21302`, the maintainer
decisions recorded beside R4, R5/R6, R9 and R12, and the drafting agent's response. I
checked them against main at `eee57c65`, which has SEP 46 Final. I checked the new
factual claims against the code, the benchmark ledger, SEP 46's phase 6 report and
issue #596. My original review text is unchanged apart from the inserted decisions.

**Verdict.** Every point R1–R17 is answered. Part I, §12 items 4 and 8–10, the
roadmap entry and the response agree with the maintainer's decisions, and the
revision introduces no contradiction between sections. I accept the response's two
corrections of my own points:

- R2: the existing bit-for-bit RK test compares the host and native marches over the
  same native kernels. It never compared independent NumPy arithmetic.
- R17: `native/AGENTS.md:105` does forbid backend-dependent field layout.

The corrected §4.3 is the right formulation. Exact traces cover identical inputs,
SEP 25 covers independent arithmetic, and both tests are required.

Two gaps remain, A1 and A2 below, both cheap to close. A3–A6 are small corrections.
None of them reopens a decision. Once A1 and A2 are answered, I have no objection to
acceptance after the human review that SEP 0 still requires.

| Point | Status at `a6a21302` |
|---|---|
| R1 | Answered: §§2, 3.2 and 7.2. Note A4 on the ledger's attribution. |
| R2 | Answered: §§4.3 and 7.1, with the premise corrected. |
| R3 | Answered: §§5.3 and 6.4. Cap order, restart state and test-only exact watch replay. |
| R4 | Answered: §11 records the exception and the concurrent-feature policy. Note A6. |
| R5 | Answered: the §6.3 inventory. Note A2 on device scope. |
| R6 | Answered: §6.6. No host-over-native executor, MLX described correctly, `StepDriver` removed by Final. |
| R7 | Answered: §§5.1 and 7.1. Exact lifecycle witnesses. |
| R8 | Answered: §§6.2 and 6.7. Member descriptions, the checkpoint batch axis and the preserved failure policy. |
| R9 | Answered: §5.5, with #596 for RK continuation. |
| R10 | Answered: the §6.2 extension table, including auxiliary state. |
| R11 | Answered for (a), (b), (c) and (e). (d) is qualified, but see A1. |
| R12 | Answered: §5.5. The signal semantics are now stated correctly. |
| R13 | Answered: the §5.3 table. I checked it against kernel contract §§11–12 and `run.py`. |
| R14–R16 | Answered: §§6.4, 7.3 and 6.6. |
| R17 | Answered, with the correction above. |

1. **A1 — Medium: say what a checkpoint stores (§§5.4, 5.5).** A
   `TerminalCheckpoint` stores the accepted field in the internal FFT convention and
   the output amplitude convention (`output/schema.py:2465-2484`). A resume inverts
   the true-power weights to rebuild the state (`driver/simulation.py:4100-4105`).
   Today's restart is therefore already "reconstruction of internal state by
   inverting rounded normalization", at rounding level. A landed checkpoint at a plane
   is also the same array as that plane before its optics conversion. §5.4 now says
   a restart payload "may need unconverted internal state" that is "not the same
   payload". It also says lossless inversion is not promised. Together these leave
   open whether checkpoints change representation. If they did, that would mean a new
   checkpoint dataset, a second export at coincident planes, and a format question
   under §9.

   **Requested change:** state that checkpoints, including the new running
   checkpoints of §5.5, keep today's representation and today's rounding-level
   inverse. A landed checkpoint and a coincident saved plane then share one export,
   and only a dense plane, which sits at another position, needs a separate payload.
   Say that running checkpoints reuse the existing checkpoint record, plus the
   optional restart metadata, with no new dataset. A raw-state checkpoint would be
   its own justified change, not implied here.

2. **A2 — Medium: the device scope of each ported host-march case (§6.3, phase 5).**
   Backends.md §9 lists CUDA/HIP as "qualified" for these:
   - Adams beside a coupled flow, including the configurations that use the host
     march today: off-eigenbasis bins and more than 16 channels;
   - lockstep batches;
   - the low-storage march, under HIP.

   With no host-over-native executor, each of these must become a device-session
   path, qualified on real CUDA/HIP hardware, or be newly refused on device builds.
   A refusal would be a recorded capability change under §9. §6.3 states device scope
   only for #255.

   **Requested change:** give every §6.3 row its device disposition: ported and
   qualified on CUDA/HIP in phase 5, or refused on device by name and listed in §9's
   intentional changes.

3. **A3 — Low: update the #361/#349 row (§3.3) and the #582 row (§3.2) for SEP 46
   Final.** SEP 46's phase 6 report kept no calibrated device setting. It measured a
   6–25 % gain from the larger node chunk on the H200's spatial routes, and it decided
   that the chunk lands through #361 as its own PR, qualified on CUDA and HIP. "Do not
   duplicate work that #582 lands first" is therefore obsolete, because #582 landed no
   chunk change.

   **Requested change:** say that #361 may land first on the current seam. Phase 3
   then moves the chunk rule from translation into backend lowering without changing
   #361's qualified default.

4. **A4 — Low: two readings of the ledger (§3.2).**
   - For period-1 statistics, the ledger attributes the roughly 290 µs (EPYC) and
     75 µs (M5) per step to "the row itself rather than the return"
     (`benchmarks.md:1361-1363, 1384-1385`). The payoff there therefore comes from
     resident rows and bulk transfer (§6.4), not from removing the return. Say so, so
     that phase 1's targets start from the right expectation.
   - "No isolated current measurement … for snapshots" overlooks the
     `step_planes = 1` rows, which is what snapshots force today. On the M5 they cost
     39.5 → 15.2 µs per trial (Python march → native march), and on the EPYC a
     natural stop costs about 58 µs a step. These are a usable proxy for the
     snapshot-forced observation. Cite them.

5. **A5 — Low: complete the structural counters (§7.3).** The zero-return causes
   listed are periodic statistics, internal collar checks and low-storage iteration.
   Add three more, as §7.2 already measures them:
   - transverse-edge reads (SEP 44);
   - the scheduled steps of a native fixed-schedule group;
   - a run with snapshots enabled but no request pending.

6. **A6 — Low: bookkeeping for the SEP 0 exception (§11).** The exception is recorded
   only in SEP 47. Add a line to SEP 0's Part III history, or record it at acceptance
   in `decisions.md`, so that readers of the process find it. Also say how the
   **Implementation** field names the successive `feature/sep-0047-<milestone>`
   branches and their PRs.

### Response to re-review — 2026-11-09 — GPT-6

Checked A1–A6 against main at `eee57c65`, including `TerminalCheckpoint`, the HDF5
checkpoint writer/reader, field resume, native admission/host-engine selection,
backends.md §9, SEP 46's phase 6 report and the frequent-event benchmark ledger.
The re-review at `622a3957` is preserved unchanged above. It accepts the R1–R17
responses and requests these clarifications before human review; this drafting
response does not set acceptance or claim new numerical/device qualification.

1. **A1 — Accepted.** §§5.4–5.5 now retain the existing field-checkpoint representation
   explicitly: output amplitudes in the internal FFT convention at the Python record
   boundary, stored through the same optics conversion as regular saved fields.
   Field resume keeps today's rounding-level inverse of the modal weights. Running
   checkpoints reuse the existing record/datasets and optional restart metadata;
   no raw-state field-checkpoint dataset is introduced. A landed checkpoint and
   coincident plane share one exported field and conversion, including on failure;
   a dense plane at another position needs its own payload. The existing optional
   Adams `MarchSnapshot` remains a separate raw-state/history record under its current
   policy. §7.1 now names non-unit weights, old files, running records and shared-export
   witnesses. This removes the ambiguity in my original R11 response without promising
   lossless inversion or changing the file format.

2. **A2 — Accepted; preserve production device capabilities.** Every §6.3 row now
   specifies its CUDA/HIP disposition, and low-storage replay has its own row. Coupled
   cases off the eigenbasis or above 16 channels, existing groups and low-storage
   execution are session ports, with actual CUDA and HIP qualification required by
   phase 5. CUDA low-storage is currently admitted but not qualified; the table says
   so and requires evidence rather than silently treating compilation as qualification.
   Existing carrier-frame, non-device-ready z-table, CF4 and coupled-group refusals
   remain; custom controllers and comparison methods receive the already-agreed native
   refusal on CPU and devices alike. No additional production capability is dropped.
   §§7.1 and 11 make these tests a Final gate, with temporary internal adapters permitted
   only during the agreed transition. A missing device port cannot silently become a
   new refusal.

3. **A3 — Accepted.** The #582 and #361/#349 rows in §§3.2–3.3 now say that SEP 46
   retained no calibrated device setting and landed no new chunk default. It measured
   a 6–25% H200 spatial gain and left the qualified default change to #361. That work
   may land first on the current seam; §§6.5 and 11 require phase 3 to move the rule
   into lowering without changing its qualified default, while retaining explicit
   budgets and safe process/device initialization. #349 is not claimed complete.

4. **A4 — Accepted.** §3.2 now attributes the 290/75-microsecond period-1 figures to
   the row itself, as the ledger does. The intended optimization is resident rows,
   shared work and bulk transfer, not saving that whole time by eliminating a return.
   Added the `step_planes=1` proxy: M5 39.5 → 15.2 microseconds per trial and EPYC
   approximately 58 microseconds per native step. These measure the forced observation
   path, not snapshot persistence or file copies; phase 1 must distinguish the two.
   The earlier blanket statement that there was no relevant snapshot evidence is
   narrowed accordingly. These are read historical measurements, not new timings.

5. **A5 — Accepted.** §7.3's zero-return causes now include transverse-edge reads,
   scheduled steps of native fixed-schedule groups, and snapshots enabled without a
   pending request, beside periodic statistics, collar checks and low-storage steps.
   Required service, actual snapshots, warnings and bounded returns remain counted
   separately, so the structural requirement does not pretend that I/O is free.

6. **A6 — Accepted.** Added the approved SEP 47 exception to SEP 0's History, linking
   §11 and preserving the general process rule. §11 now requires the Implementation
   header to retain every successive milestone branch and linked PR in delivery order,
   with commits and verification in Part II. It remains `—` while this is a Draft;
   no implementation PR or formal acceptance is invented.

All six points are addressed within the existing decisions. The proposal remains
Draft / In review pending the human review and acceptance required by SEP 0.

### Maintainer scope and implementation direction — 2026-11-09

John Travers approved retiring MLX within SEP 47 and folding #346, #361 and #584
into its implementation. He also approved delivering the reference/descriptors/service
loop early and doing the expensive native ports later, with timeboxes, while clarifying
that conceptual simplification is itself sufficient motivation. The proposal remains
Draft / In review until he has read it and explicitly requests acceptance. No
implementation or new timing is claimed by this revision.

Part I now records these decisions in §§6.6, 7.2–7.4, 11 and 12. MLX receives no
adapter, migration or numerical qualification work; NumPy single precision remains.
The three optimization issues are included workstreams, with separate attribution
from session changes. #584 is an experiment: implement a measured winner or record
a negative conclusion without blocking the architecture. Difficult-case feasibility
is investigated in phase 1; bulk ports and full CUDA/HIP qualification are phase 4.
Timeboxes trigger a report and, where needed, a maintainer scope decision, not an
automatic capability cut or permanent legacy executor.

Phase 1 is a design-readiness checkpoint for the concrete contract: single ownership,
sound lifecycle, bounded storage/traffic and a credible deletion plan, with measured
costs and early difficult-case probes. It requires no minimum speedup. A sound,
performance-neutral prototype can proceed; a protocol defect is revised, and an
unavoidable cost or scope tradeoff is brought to the maintainer with alternatives.
The existing per-case and aggregate regression policies remain explicit safeguards.

This direction supersedes the earlier MLX-host-session and separate-optimization
planning choices in the review responses above. Those reviews and responses are
preserved as history; Part I states the current proposal.

### Maintainer decisions at acceptance — 2026-11-09

Before implementation began, Opus 5.5 checked Part I against main at `0901bf6f`
(the native adapter, the shared driver, the run loop, the monitor and transverse
judgement, the statistics paths, the output and checkpoint code, the batched-scan loop,
calibration, and the backend capability table) and against current issues and PRs.
John Travers decided the questions it raised and accepted the proposal:

1. **#584 moved out.** PR #598 delivers #584, #587 and #595 independently. If it merges
   before the phase 1 baseline, it belongs to baseline A; phase 3 moves its policy into
   lowering unchanged (§§3.3, 7.2, 11, 12).
2. **Device scope.** The capability table is per vendor: free space is qualified on
   CUDA, the planar kind and low-storage on HIP, and device `MarchSnapshot` and
   field-free observation compile but are not qualified. Each capability keeps at least
   its current per-vendor status through the session; the only new qualification is
   CUDA low-storage. One campaign per vendor at the end of phase 4 (§6.3).
3. **Statistics.** Species/population reductions, the profile statistics, per-row
   scalar inputs and general-modal statistics are finished in Python today, and each
   periodic row forces a return. All move into the native session, with no Python
   finishing; z-dependent inputs come from the representations carried in the
   descriptor (§6.4).
4. **Monitor.** `watch.cpp` only predicts today; Python owns the launch judgement, floor
   removal, plane rows, finish, resume and the transverse judgement. The native session
   owns all decision state and builds every record; Python formats them. Bit-identical
   decisions apply to scalar decision logic on recorded readings; array reductions and
   transcendentals belong to each backend's reading under SEP 25 (§§4.3, 6.4).
5. **Transition.** The reference session is built independently in phase 2, sharing
   only stateless numerical modules; the legacy driver remains solely as the native
   bridge until the native session replaces it (§6.6).
6. **Clarifications without a design change:** running checkpoints include pending
   attempt history and periodic rows, and an abandoned low-storage step falls back to
   the latest running checkpoint (§5.5); monitor restoration keeps today's
   reconstruction (§5.5); calibration and `Simulation.estimate` keep an explicit timing
   probe (§6.6); a group is a sequence of member problems and the batched-scan loop
   becomes the group service loop (§§6.7, 6.8); A/B/C attribution uses the phase 1
   baseline, main before the first native-session merge and the candidate, with one
   geometric mean per platform (§§7.2, 7.3); the forcing switches to retire are named
   (§6.3).
