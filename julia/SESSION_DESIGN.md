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

Every Julia-facing Python call, destruction and exception formatting stays on main.
The existing handle registry retains borrowed buffers and wrapper lifetimes. Python
services closed events and validates destination arrays; it never drives numerical
trials or computes periodic statistics. Failures retain SEP 47's pending/failure
tickets, undelivered data and in-flight facts. Monitor continuation is explicit.

## Delivery order

1. Parallel reviewed CPU execution/transforms/operators and response kernels.
2. One complete binary64 route through preparation, Julia stepping and destination
   export, then every admitted route/response and integrator.
3. Full session services, restart/group conformance, and CPU measurements.
4. Binary32 and analysed DF32, then independently qualified device extensions.

The experiment supplies candidate kernels, not a grandfathered correctness claim.
Use independent short reference witnesses, rounding/accuracy budgets, bounds-checked
tests and warmed allocation checks as each executable path is integrated.
