# SEP 48 session interfaces

This is the working interface record for phase 2. The master owns changes to shared
interfaces; implementing agents own only the files in `IMPLEMENTATION.md`.

## Numerical storage and execution

The Python boundary uses the existing C-contiguous `(n_modes, n_pol, n_omega)` state.
Julia owns an equivalent column-major `(n_omega, n_pol * n_modes)` matrix, with
polarisation varying before mode in the column index. Time/nodal buffers similarly
use `A[t, component + n_components * (node - 1)]`, corresponding to Python's
`A[node, component, t]`. Every fill of a buffer passed to full-chunk transforms or
contractions zeros its inactive columns; every scratch cell subsequently read is
overwritten, including recurrence padding and integration endpoints. Conversion
happens once at preparation and explicitly at destination export; no kernel changes
Fourier convention or amplitude normalisation.

The shared numerical execution record is `KernelExecution{B}`, with `backend::B`,
`tasks::Int` and `check_body::Bool`. It is independent of Python and session policy.
The transform agent owns its implementation and these launcher entry points:

- `each!(f, execution, n_rows, n_columns; heavy=false)` calls `f(row, column)`.
- `each_column!(f, execution, n_columns)` calls one sequential column body.
- `oncpu(execution)`, `upload(execution, array)` and
  `dzeros(execution, element_type, dimensions...)` prepare owned storage.

These retain the experiment's useful scalar-kernel shape. CPU execution is synchronous:
every launched task finishes before return. The requested task budget is an upper bound
across scalar loops, FFTs and BLAS scheduling. Kernels capture only concrete arrays/scalars;
no mutable process-wide engine, arbitrary scratch dictionaries, FTZ or blanket fast
math. Work buffers and transform plans are allocated at preparation. Reductions take
explicit scratch owned by their caller. Vendor tuning is deferred to measurements.

The response agent may use these launcher signatures while the execution module is
being implemented. Every other response parameter is explicit, typed numerical data;
Python objects and mutable session state do not enter leaf kernels. Response fields
have component/node counts specified by their prepared record. Scalar and vector
responses must not infer different layouts silently.

The transform package owns temporal FFT padding/normalisation, modal and spatial
linear maps, represented operator evaluation and stable operator increments.
The response package owns Kerr, Raman convolution/recurrence, ionisation-rate tables,
population and current kernels. Its Raman padding includes the small non-THG case
reported in issue #613. Each physical profile is evaluated from its own represented
table: operator, prefactor, Kerr, density, preionisation, Raman and taper weights are
not interchangeable. The stage count comes from the method, never a fixed seven.

## Session and Python boundary

The binary64 exponential Adams leaf owns history, coefficient and trial workspaces.
`adams_trial!` takes source, exponent and rate callables: `source!(out,state,z)`,
`exponent!(out,z0,z1)` and `rate!(out,z,before::Bool)`. Each returns `nothing` or throws
outside a leaf kernel. The master owns order/slot selection, acceptance, masks, collars
and session failure accounting. Source history retains the actual predictor sample;
compensation uses its sampled rate and the derivative of the represented cumulative
operator. All coefficient generation and dense interpolation execute in Julia.
Protected sum-of-squares helpers are shared with Runge–Kutta, but Adams uses its own
old-state norm scaling. CPU binary64 qualification does not qualify DF32 bounds.

Statistics use fixed `StatisticSlot` tuples, an immutable selection/grid, explicit
prepared numerical inputs, and owned workspaces. `statistics!` fills a packed output
without growing storage or returning to Python. The master constructs the inputs from
the actual committed or dense field using shared transforms; no stale RHS buffers are
implicit inputs. Plane-only spatial defaults add `beam_radius`, `fwhm_r` and
`on_axis_spectrum`; other out-of-scope plane-only families refuse by name.
Monitor reductions produce concrete readings; scalar state transitions consume those
readings in reference operation order. Temporal/transverse continuation is explicit.
Fixed notices and plane rows are returned to session-owned buffers; Python formats and
services already-decided events. Neither statistics nor monitors own delivery cursors.

The master owns immutable transfer records, their pure-Python lowering, and the
Julia-owned mutable session. Lowering imports no native backend. The package core
has no Python dependency; the PythonCall extension converts boundary records once.
Preparation caches immutable lowering without retaining its weak-cache key. Each
session owns its mutable state, work, controller, history, monitors, statistics,
cursors, delivery buffers, snapshots and export generations.

Production callers use an execution-owner protocol: name/request, prepare, metadata,
memory ledger, close/closed, timing probe, memory admission and history-export
admission. The existing legacy operation protocol remains for its current adapters;
Julia does not implement placeholder stepping/observation methods. Group construction
uses the owner's `start_group(members, problems, requests, starts, stop=...)` capability.
Julia owns fixed-schedule member marching and
refuses incompatible effective precision, wide arithmetic, setup or layout before
consuming starts. Shared dispatch has no Julia-name branch.

The Julia group coordinator contains independent member sessions, with no fused
field or numerical work buffers. Its core validates fresh ownership and the shared
actual itinerary, edges, planes, entrance and private stop flags needed for scheduling.
Public method/layout/norm-mask and effective execution-option compatibility belongs
to Python preflight before any caller-owned start is consumed. Member constructors
retain their own precision/device admission; the phase 2 constructors admit CPU
binary64 only. Differing immutable physical coefficients remain member-owned.

Every Julia-facing Python call, destruction and exception formatting stays on main.
The existing handle registry retains borrowed buffers and wrapper lifetimes. Python
services closed events and validates destination arrays; it never drives numerical
trials or computes periodic statistics. Failures retain SEP 47's pending/failure
tickets, undelivered data and in-flight facts. Monitor continuation is explicit.

## Shared landed-session composition

`march.jl` owns the internal `_MarchSession` superclass and its common event loop,
acknowledgment, snapshot requests, stop polling, close and failure records. Concrete
integrators provide `_march_trial!(session, began)` and retain the shared fields
`execution`, `field`, `norm`, `schedule`, `source`, `flow`, `stop`, `attempts`, `state`,
`services` and `exports`. A method's trial is entirely Julia-owned. It returns whether
the attempt was accepted, records the actual attempt through `_march_record!`, commits
position with state, and invokes `_session_accepted!` after a successful commit.
`_prepare_march` validates and copies the launch, applies its hard mask and establishes
the common plane/replay cursors. Only the master edits these shared boundaries.

`adams_session.jl` composes the approved Adams policy and kernels for landed
adaptive/fixed sessions, reusing this common event transport. Retain the full
oldest-first `order+1` history for policy and
resume, actual predictor samples, and the original sampled positions even when the
committed endpoint snaps to a landing. Store no unbounded order log inside advancement:
fixed counters and bounded per-attempt order rows can be drained with event chunks.
Master integration owns Python lowering, ledger translation, dense observation hooks,
restart records and the observation-service attachment. No partial method is advertised
as publicly qualified before these integrations and conformance checks pass.

The bridge copies cumulative method counters and accepted orders from each pending
generation without draining it. Rejected attempt rows encode order zero. Python owns
the compact run-length record outside numerical advancement and publishes the record,
array and generation as one immutable value, including on failed-consumer retries.
Acknowledgment cannot clear a chunk before that copy succeeds. The bounded Julia
order buffer and retained Python record array both appear in their memory ledgers.
Budget diagnostics query the exact refused numerical charge and export the committed
field; Python formats the existing shared exception without deciding trial admission.

The next numerical composition is `LowStorageSession(execution, state, z, tableau,
norm, schedule, source, flow; stop, capacity, services, exports, plane_index,
plane_pending)`. It requires a fixed itinerary before allocating. Its in-place trial
marks the field in flight before any source call. Keep attempted source invocations
separate from the existing low-storage method's committed-call and norm counters.

Service numerical hooks are `_session_quiet(services, session)::Bool`,
`_session_cap(services, session, proposal)::Float64` (transverse then split caps, after
the quiet cap), and
`_session_finish_candidate!(services, session, candidate, dz, destination)`.
The last hook sees an active-masked candidate, captures split removal, then applies
split damping; `destination` is the actual planned endpoint, including any landing
snap, and commit applies the final hard mask. A trial records its completed
attempt before reporting a proposal-floor failure, including rejection. This preserves
actual-work accounting even when no further trial is possible.
The existing reference/native Adams omission is tracked in
[#651](https://github.com/LupoLab/solprop/issues/651). Preserve the historical trace
corpus and compare controller decisions and charged calls separately from this
deliberate diagnostic correction.

The frozen modal service composition owns `SessionObservationPolicy` and
`ObservationServices` in `observation_session.jl`. Policy retains immutable observation
preparation, positive periodic cadence, explicit collar-rest eligibility and an optional
owned coarse power-loss rate. Each session owns observation/split workspaces, compact
periodic values with capacity `attempt_capacity + 1`, one packed plane, two temporal
plane rows, one immediate notice and an explicit numerical continuation tag. No service
owns a second receipt or export generation. Accessors `observation_status`,
`observation_rows`, `observation_planes` and `observation_notices` never drain data.

Shared hooks `_session_barrier` and `_session_continue!` return symbols. Barriers are
`:none`, `:warning` or `:boundary_stop`; continuation tags are `:none`, `:launch_rest`,
`:accepted_rest`, `:plane_rest` and `:finish_rest`. A warning interrupts before the
endpoint's periodic statistics or plane preparation. After acknowledgment the numerical
remainder executes once, and the common scheduler services any already reached plane
before another trial. A continued periodic row uses `attempts.count - 1`, including
`-1` when its attempt was delivered by the preceding warning chunk; this is the existing
SEP 47 rebasing convention. The additional capacity covers that one deferred row.

`_session_acknowledge!` resets only published service counts/plane validity after
common generation validation. It preserves continuation, quiet/cap decisions, period
phase and monitor state. `_session_quiet_interval` supplies the resolved quiet cap to
RK; Adams already retains it in its policy. New split-crossing readings check upcoming
fixed steps before further trials and refuse instead of shortening the itinerary.
The exact endpoint's field remains eligible for that diagnostic. The master owns the
common scheduler and generation-checked bridge; the service agent owns only the new
service source and its focused tests. Dense planes remain a separate later integration
and must never rewind the committed monitor.

## Request-specific observation preparation

`SessionPreparation` pairs the cached `ModalPreparation` with immutable
`SessionObservationPolicy`. Statistics selection, cadence and monitor policy belong
to the request, so they never enter the problem-only source cache. Its memory record
counts only observation arrays not already owned by the modal preparation. Every
start constructs its own `ObservationServices`; its ledger excludes both immutable
preparations. An active Python session retains its prepared factory through its
release callback, keeping request preparation alive and accounted until cleanup.

The generation-checked bridge copies packed periodic rows and plane values into
owned Python buffers without draining the engine. Python reshapes the catalogue's
explicit C-order slices and publishes the complete payload atomically. A failed
copy leaves the same generation retryable; acknowledgment occurs only after all
retained execution and observation records have been captured. No statistic or
monitor reduction runs in Python.

## Delivery order

1. Parallel reviewed CPU execution/transforms/operators and response kernels.
2. One complete binary64 route through preparation, Julia stepping and destination
   export, then every admitted route/response and integrator.
3. Full session services, restart/group conformance, and CPU measurements.
4. Binary32 and analysed DF32, then independently qualified device extensions.

The experiment supplies candidate kernels, not a grandfathered correctness claim.
Use independent short reference witnesses, rounding/accuracy budgets, bounds-checked
tests and warmed allocation checks as each executable path is integrated.
