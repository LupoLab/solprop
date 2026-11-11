# The optional Julia backend

Julia support is being implemented under [SEP 48](../../dev/seps/sep-0048-julia-backend.md).
The package and explicit setup commands are available on its implementation branch.
Numerical sessions and device qualification remain in progress; installing a device
environment does not establish that its propagation kernels are qualified.

Select the prepared CPU engine with `SimulationSettings(backend="julia", ...)`.
`backend_options={"setup_id": identity}` selects an explicit published setup; omitting
the identity requires exactly one compatible setup. Availability checks probe the
bridge installation without importing Julia; the first preparation verifies setup.

The current integration supports scalar modal fields and envelopes in binary64
with Kerr, Raman and plasma responses and Runge–Kutta or exponential Adams stepping,
including adaptive and fixed replay schedules. Fixed schedules also support the
three-register low-storage RK4IP method, which records no embedded error estimate.
Constant, tabulated and expanded
operators, prefactors and power weights keep their full represented variation. Each
material response uses its own density, decay and preionisation profiles; taper scaling
follows the translated mode map.
Scalar radial, Cartesian and line propagation is also attached. Radial propagation
retains its translated quadrature and padding factors; Cartesian and line propagation
use spatial FFTs and the represented longitudinal root. A physical transverse absorber
remains a source term even when no material response is present. Spatial exports retain
their spatial measure and do not apply the modal power-normalization flag.
Julia owns the stepping loop, error control, bounded attempt history and event barriers.
Plane and periodic scalar modal statistics are evaluated inside Julia from the current
field. The packed catalogue, frequency windows and independently represented material
profiles belong to each run request. Periodic cadence counts accepted steps and is
unchanged by event yields. A failed copy can retry the same closed chunk; acknowledgment
drains it only after its complete values are owned by Python. Already delivered rows
remain readable after session cleanup.
Temporal monitoring also belongs to Julia: launch checks, accepted-step cadence,
guard decisions and cumulative removal are evaluated there. Python formats the
shared record and warning types from the numerical facts. A warning pauses at a
closed service barrier, with a valid export ticket if a warning filter turns it
into an error. Monitor continuation restores its original launch, cumulative totals
and saved rows; it does not judge a resumed launch again. The first continuation
supplied in `Start` takes precedence over a later output-record fallback. Monitor
metadata remains readable after the session closes.
Binary64 input noise uses the supplied draw without generating another realisation.
Julia adds it once at a fresh problem launch and applies the active-band mask; a
resumed field already contains its draw and never receives it again. The draw remains
available for the monitor's noise floor. Phase exports evaluate the cumulative unitary
phase at the selected observation coordinate, including a dense plane, and preserve
represented absolute offsets. Each start selects its own draw, so reusing preparation
does not reuse an earlier start's replacement noise. Modified noise and every binary32
noise configuration remain refused.
Snapshots, cooperative stops and field exports use the same session contract as other
backends. An acknowledged event invalidates its export ticket; arrays already returned
to Python remain valid. Failed output consumers can recover an undelivered history chunk
once, without repeating numerical work.

Fixed-schedule groups use Julia's member-session coordinator. Members must share the
method and complete tableau, itinerary, landed observation planes, shape, precision,
norm mask, statistics policy and execution choices. Construction validates these
before consuming any entrance field; interrupted construction leaves the original
starts and caller-owned backends available for retry. Julia chooses every member turn;
Python publishes a complete common window and copies each member chunk once. A failed
publication retains its assembled chunks for retry without further numerical work.
Stops and failures preserve each member's actual committed position, history and
available field ticket, and share one primary error. Closing the group closes its
adopted owners. Group resume and interpolated group planes remain refused.

Adams receives its controller and landing constants from the shared resolved method
policy. Its execution record distinguishes logical response-call charges from actual
source invocations, including structurally absent sources, and retains compact accepted
order counts. A spent call budget stops before the next trial and carries the last
accepted field. Portable binary64 Adams snapshots retain that raw state, every history
row, the selected sample positions and the next controller state. Compatible history
continues without an extra startup evaluation; cumulative charges and compact orders
also survive a field-only restart. History identity uses the shared method key, not a
Julia setup, thread count or backend name. Incompatible keys, precision or store layouts
produce the shared explicit field-reset reason before any history is installed.
Short fixed-history continuation between Julia, the NumPy reference and native CPU is
tested in both directions involving Julia. Plain and compensated stores retain every
row exactly. Adaptive Adams can retain its accepted step and evaluate all crossed
observation planes without replacing the committed state or calling the source again.
Statistics and field conversion use the plane's own coordinate. An interior plane has
no portable history from the endpoint ahead of it; it can restart from its exported
field. Snapshot requests wait until the owed planes are delivered, then save the
committed endpoint and its history. Dense output remains incompatible with fixed replay
and split temporal absorption, as in the shared policy.

Adaptive DOPRI5 uses the shared continuous-extension weights for the same dense
observation contract. Other tableaus without a shared extension refuse interpolation.
Endpoint periodic rows accompany their accepted attempt before the first crossed
plane is delivered; later dense planes do not repeat those rows. A plane consumes the
previous stage's quadrature diagnostic, so another plane reached without new source
work reports that diagnostic as unavailable. Both dense methods pass the common
session ordering, snapshot and bounded-yield checks.

Scalar modal runs also support the existing `split_step` temporal absorber. It applies
one coarse-time amplitude factor `exp(-alpha*dz/2)` after each accepted trial, between
the initial and final active-band masks. It is a one-sided Lie split with its existing
order limitation; the default in-stage scheme is unchanged. Split finishing attaches
even when no statistic or boundary monitor is requested, and a quiet monitor verdict
never omits this factor. The split rate does not also enter the nonlinear source.

Low-storage execution records preserve the established `source_evaluations` count
of committed steps and separately report `attempted_source_evaluations` and completed
`norm_evaluations`. If an in-place trial fails, its attempted calls remain visible and
its field is poisoned, so no field-export ticket is issued.

Low-storage propagation accepts `propagator_mode="tabulate"`. A session owns one
factor table for a constant linear operator, or two for a represented varying one.
Constant tables match the actual floating-point interval length exactly; varying
tables match both endpoints. An interval change rebuilds the factors without
interpolation or a step tolerance. `auto` tabulates constant low-storage flows;
other method families continue to recompute, as reported in execution metadata.

Default field-plasma diagnostics (`plasma_undershoot`, `plasma_work_defect`, and
`plasma_boundary_level`) use a fresh population scan of the observed field. Each
ratio divides separate global maxima over all channels and nodes, after combining
chunks. These diagnostics and the three species inventory/peak rows are plane-only.
`species_ionisation_energy` reports final ionised inventory in J/m, including the
initial preionised fraction. It uses the physical transverse weights and represented
area/taper at that plane; it is distinct from newly deposited ionisation work.
Passive species keep their catalogue positions and zero ionisation potential.

The four `on_axis_*` plasma statistics reconstruct the fine-grid axis field
independently at each saved plane. They require a resolved modal plasma route and
the original physical modes. Field simulations retain both real Cartesian field
components. Envelope simulations admit a scalar envelope along a fixed real
polarisation direction: the represented axis-map rows must be exactly proportional,
including zero rows. A general elliptical envelope refuses by name. These rows are
plane-only and do not reuse the last nonlinear stage's populations.

For represented constant unit modal power weights, `physical_energy`, `analytic_flux`
and `spectral_flux` are available. Other weight representations refuse these three
names; use `energy` and `peak_power_total` for weighted modal power. Spectral flux
uses its per-angular-frequency density weights in J/(rad/s), independently of the
Parseval endpoint weights. Physical energy and analytic flux also support periodic
rows; spectral flux is plane-only.

Public CPU selection uses the same preparation and session contract as the other
backends. Unsupported combinations refuse by name before session allocation.
The current session supports in-stage and split temporal absorption. Transverse monitor
attachment now uses the same event barriers as temporal monitoring. It retains original
clean launch columns for noise-floor subtraction, restores per-face cumulative totals,
and uses the first explicitly supplied continuation, including an absent edge record.
An older checkpoint without an edge continuation initializes its baseline from the
resumed field. Python formats the shared warnings and summaries from detached Julia
readings; it does not recompute edge decisions. Vector
polarisation, tensor Raman, general modal and planar routes, coupled flows, TDSE and
dispersive Kerr remain outside SEP 48's admitted scope and refuse by name.
`SimulationSettings.node_budget` bounds the retained numerical array payload that grows
with the source node chunk, population observations and transverse trace chunk together.
The planner uses each actual prepared request, including statistics or monitors added
after construction. Immutable numerical inputs remain shared; each prepared request
keeps its own capacity. A budget smaller than one node refuses before Julia starts.
The default budgets are 64 MiB for modal/radial work and 1 GiB for Cartesian/line work.
The lower-level `ExecutionOptions.node_chunk` overrides the budget for controlled
measurements, capped at the physical node count. These execution choices do not alter
scientific identities.

This is an array-payload budget, not a total process-memory limit. Metadata separately
reports fixed and full-volume workspace arrays, the component inventory, opaque FFT
plans and exclusions. Immutable preparation, state/history/export storage, bounded
transport records, object overhead and temporary planning allocations remain outside
the node budget. The complete retained-memory ledger remains available separately.

The scalar statistics catalogue includes spectral centroids, energies and frequency
windows, peak powers, temporal widths, material profiles and populations, and on-axis
peak intensity when the original physical modes are supplied. The quadrature diagnostic
is NaN at launch and when no distinct embedded projection exists. Short same-field CPU
comparisons use explicit transform and reduction bounds and certify temporal-width
brackets separately; they do not assign a generic relative tolerance to material rates.
Complete route/catalogue qualification remains part of phase 2.
Spatial saved-field observations include physical flux energy, peak and on-axis
intensity, on-axis spectrum and beam widths. Their closure uses a separately translated
flux descriptor, including its own material representation. The on-axis spectrum uses
the shared spectral-density normalization. Width comparisons require stable peak and
threshold-crossing margins; a generic relative tolerance does not certify them.

Binary32 remains unavailable through public backend selection while qualification
continues. Its internal CPU paths now include material responses, observations and
monitors; short tests compare physical energy bands and other observables against
refined binary64 references. Binary32 input noise remains refused.

The reviewed double-float exponential, phase-reduction and phi-function kernels have
conservative function domains and explicit underflow assumptions. Internal landed
RK and low-storage sessions now bind the original operator and actual start, then
certify each actual trial in Julia before source work. Short field-restart and group
checks also pass. The internal landed Adams flow also forms certified complete factors
and binds only compatible portable history, preserving explicit field-reset reasons.
Completed restarts retain raw history without evaluating another source or factor.
The complete restart matrix, dense and spatial execution still need qualification
before public selection. Admission must include actual replay steps,
every represented operator interval and restored history, including the arithmetic
that forms each exponent before the functions receive it.

The engine precompiles representative serial CPU sessions during explicit setup. This
reduces first-use compilation for those paths; new shapes, responses and thread
configurations may still compile on their first use. The workload changes no process
thread policy and stores no mutable simulation state.

CPU preparation shares solprop's host BLAS limits. The default is one BLAS thread;
`SOLPROP_BLAS_THREADS` requests another width within the process budget. Overlapping
construction and Julia preparations retain the narrowest active request, and release
restores the previous width only after the last owner finishes. Metadata records the
actual controllable libraries and their current widths. OpenBLAS is the provider on
Linux and other supported platforms. On macOS with the new LAPACK ABI and documented
calling-thread controls (macOS 15 or later), Julia forwards BLAS/LAPACK to Accelerate
and retains OpenBLAS for symbols Accelerate does not provide. Missing capabilities
leave OpenBLAS selected, with the reason recorded in execution metadata.

Accelerate offers automatic or single-thread mode, so a Julia preparation keeps it in
single-thread mode even when a larger BLAS width was requested. That mode belongs to
Python's main OS thread and affects all Accelerate callers there, including NumPy and
native code using the same framework. Independent BLAS libraries keep their ordinary
count limits. The last preparation restores the exact previous Apple mode; metadata
records the requested width, actual single mode and its calling-thread scope. Julia's
worker threads do not inherit this setting; the current engine's parallel matrix maps
use its own kernels. See [ADR-079](../../dev/decisions.md#adr-079--verify-julias-cpu-blas-provider-and-lease-its-real-controls).

CPU starts plan against their own transform arrays and keep successful FFTW measurements
in the calibration store. Cache identities include the actual loaded FFT libraries and
their inventoried artifacts, Julia and engine versions, CPU topology, target and effective
preferences. Each transform key includes its shape, strides, transform kind, precision,
alignment and actual planner thread count. Reusing a prepared execution refreshes its
offers, including plans measured by earlier starts in the same process.
`SOLPROP_FFTW_PLAN` selects `estimate`, `measure` (the default) or `patient`.
`SOLPROP_CALIBRATION=off` retains new plans only in the process unless an explicit
`SOLPROP_FFTW_WISDOM` path is supplied. That override names a Julia-specific file beside
the requested path. Invalid or unusable cache files produce a warning and leave planning
available. A failed optional file write also leaves successful numerical preparation
usable. `solprop calibrate --clear` removes compatible plans for the current topology
and forgets process offers, without loading Julia or changing already-created plans.
Execution metadata reports measured, applied and reused **distinct transform keys**,
cache rejections and whether a file was supplied or written. Wisdom strings are released
after construction; the session retains only its actual plans and compact metadata.

The internal execution owner also supplies isolated timing probes for cost estimates.
Julia owns their loaded field, stages, source, norms and accepted-step finishing. A probe
creates no session, monitor, history or output event, and a normal embedded trial does
not advance its loaded state. Its field readback is an owning host copy available once
per trial. Low-storage probes update discarded registers and have no readable trial
field. Closing the execution owner closes its probes before releasing preparation.

## Install and inspect

The extra installs the Python bridge, JuliaCall. Julia and its packages are installed
only by an explicit setup command:

```sh
pip install "solprop[julia]"
solprop julia setup
solprop julia status
```

The [Python setup interface](../../api/julia.md) provides the same operations for scripts.

Setup uses Julia 1.13.3 and matching JuliaCall/PythonCall 0.9.37. An existing Julia
executable can be supplied with `--julia`; another patch in the 1.13 series is recorded
as untested. A different minor series requires new manifests and qualification.
Installing solprop without the extra, importing solprop, and selecting another backend
do not initialize Julia.

The default setup prepares the CPU environment. Add a device explicitly:

```sh
solprop julia setup --device cuda
solprop julia setup --device rocm
solprop julia setup --device metal
```

Metal requires macOS on Apple silicon. ROCm requires a discoverable system installation;
missing required ROCm libraries are reported before runtime startup.
The qualification targets are Linux x86-64 and macOS arm64. CPU setup also installs on
Linux aarch64 and Windows x64, which remain unqualified; Windows arm64 is refused.
CUDA on Linux aarch64 refuses until setup identity distinguishes Jetson from SBSA and
records the selected driver platform. These use different artifacts in the pinned
[CUDA runtime selector](https://github.com/JuliaBinaryWrappers/CUDA_Runtime_jll.jl/blob/CUDA_Runtime-v0.25.0%2B3/.pkg/platform_augmentation.jl#L463-L499)
and [driver selector](https://github.com/JuliaBinaryWrappers/CUDA_Driver_jll.jl/blob/CUDA_Driver-v13.5.0%2B0/.pkg/platform_augmentation.jl#L27-L62).

CUDA setup can fetch a specified toolkit without a GPU, for a later compute-node run:

```sh
solprop julia setup --device cuda --cuda-runtime 13.4
```

The default toolkit release is recorded in the bundled CUDA environment. When sharing
a process with a native backend built against a system CUDA toolkit, prepare Julia with
`--cuda-local` so both use that toolkit. This requires the toolkit on the setup host.

## Published identities

Every successful setup prints an immutable identity. Status lists all published
identities, their source and bridge versions, installed devices, CPU target, depot size,
and any reason they cannot be used. Setup keeps an owned source copy, so removing the
Python environment that created it does not remove the engine.

Changed engine sources, preferences or installed device selections produce a different
identity. Selection without an explicit identity requires exactly one compatible setup;
ambiguity reports the available IDs. The session interface selects an
identity through the `setup_id` backend option and records it as execution metadata.

`SimulationSettings.backend_options` carries backend-specific execution choices. It
defaults to an independently owned empty mapping, and preparation takes another owned
copy through `settings.execution_options()`. These choices survive spawned workers and
may be replaced after translation without changing the scientific configuration or
restart identity. The selected backend refuses unsupported keys before constructing a
workspace. The Julia adapter admits `setup_id` and `wide`; the existing native and
NumPy reference backends accept no backend-specific keys.
Recipes that construct `SimulationSettings` carry this mapping under
`build.settings.backend_options`; TOML round trips and Python export preserve it.

The first preparation fixes the process's runtime, depot and installed environment
stack. A different setup, another depot, or a device installed after initialization
requires a fresh process. Every Julia-facing operation must run on Python's main thread
in each process, including preparation, device loading, advancement, export and close.
Calls from Python worker threads refuse before entering Julia. Julia's own parallel
CPU work remains supported; use spawned processes for concurrent simulations. Setup,
status and cancellation requests do not enter the embedded runtime and can be used
from other threads. Do not fork a process after Julia has initialized. An initialization
or extension-loading failure requires a fresh process because the runtime cannot be
safely unloaded.

Close each prepared execution and session explicitly on main. Abandoned Julia handles
are retained until the next main-thread operation or process shutdown, so Python garbage
collection on another thread cannot destroy Julia objects there. Returned public arrays
own Python storage, and engine errors are formatted on main before leaving the adapter.
This restriction avoids a reproduced Julia 1.13.3/PythonCall 0.9.37 garbage-collection
deadlock on both macOS and Linux; [issue #620](https://github.com/LupoLab/solprop/issues/620)
records the evidence and the accepted scope change.

## Offline and cluster use

Setup downloads the selected package and artifact closure, verifies it, and publishes
an inventory only after precompilation succeeds. Runtime startup and a new device
extension check the installed stack before loading. Missing artifacts refuse with a
setup command; runtime execution does not repair them by downloading.

`SOLPROP_JULIA_DEPOT` selects a private depot outside the default calibration store.
For a heterogeneous CPU allocation, choose a compatible compilation target when
preparing it:

```sh
solprop julia setup --cpu-target generic
```

Status reports the CPU name and target used for precompilation. Julia validates native
code caches and recompiles when necessary at load. Sources, manifests, preferences and
artifact contents remain part of the installation's identity. Do not edit them in a
published setup or add Julia artifact overrides, alternate preference files or workspace
configuration. Explicit setup can restore entirely missing shared trees; existing
corrupt trees refuse. Prepare a fresh private depot when the original cannot be repaired
without altering published inputs.

## Cancellation and signals

```{warning}
Once Julia initializes, SIGTERM may terminate the entire process without delivering
pending output or writing a final checkpoint. This also applies to custom or ignored
Python SIGTERM handlers, while the process is idle, and after a Julia session closes.
Fatal shutdown can also stall during garbage collection. A supervisor that needs a
bounded process lifetime must enforce an external deadline and send SIGKILL if the
process has not exited. Neither signal guarantees a final checkpoint.
```

Graceful cancellation uses cooperative stop tokens. The bridge preserves the
application's actual SIGINT handler, including an ignored handler, for Python Ctrl-C
handling. Julia keeps the signal machinery needed by its threaded garbage
collector; disabling that machinery is not a supported workaround.

The maintainer accepted both limitations for Julia 1.13.3. A second SIGTERM ended one
diagnostic stall, but it is not the hard deadline mechanism. Use cooperative stop tokens
for normal cancellation and an external process supervisor for forced termination.
[Issue #610](https://github.com/LupoLab/solprop/issues/610) tracks revisiting it with stable
Julia 1.14 and compatible device packages. An upgrade alone is not evidence that signal
preservation is fixed.

The complete scope, qualification requirements and implementation reports are in
[SEP 48](../../dev/seps/sep-0048-julia-backend.md).
