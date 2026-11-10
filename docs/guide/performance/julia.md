# The optional Julia backend

Julia support is being implemented under [SEP 48](../../dev/seps/sep-0048-julia-backend.md).
The package and explicit setup commands are available on its implementation branch.
Numerical sessions and device qualification remain in progress; installing a device
environment does not establish that its propagation kernels are qualified.

The current internal integration supports scalar modal fields and envelopes in binary64
with Kerr, Raman and plasma responses and Runge–Kutta or exponential Adams stepping,
including adaptive and fixed replay schedules. Fixed schedules also support the
three-register low-storage RK4IP method, which records no embedded error estimate.
Constant, tabulated and expanded
operators, prefactors and power weights keep their full represented variation. Each
material response uses its own density, decay and preionisation profiles; taper scaling
follows the translated mode map.
Julia owns the stepping loop, error control, bounded attempt history and event barriers.
Plane and periodic scalar modal statistics are evaluated inside Julia from the current
field. The packed catalogue, frequency windows and independently represented material
profiles belong to each run request. Periodic cadence counts accepted steps and is
unchanged by event yields. A failed copy can retry the same closed chunk; acknowledgment
drains it only after its complete values are owned by Python. Already delivered rows
remain readable after session cleanup.
Snapshots, cooperative stops and field exports use the same session contract as other
backends. An acknowledged event invalidates its export ticket; arrays already returned
to Python remain valid. Failed output consumers can recover an undelivered history chunk
once, without repeating numerical work.

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
row exactly. Dense checkpoints and the remaining restart matrix still need qualification.

Low-storage execution records preserve the established `source_evaluations` count
of committed steps and separately report `attempted_source_evaluations` and completed
`norm_evaluations`. If an in-place trial fails, its attempted calls remain visible and
its field is poisoned, so no field-export ticket is issued.

This composition is being extended to the remaining accepted routes and session
services before public backend selection is enabled.
Unsupported combinations currently refuse by name at preparation. The standalone
Statistics and temporal/transverse-monitor components have separate tests;
their presence alone does not establish complete Python session support.
The current session supports in-stage temporal absorption. Split absorption, noise,
vector responses, general modal and spatial transforms, and monitor/dense attachment remain
pending. An explicit node chunk bounds the source's physical-node scratch; memory-budget
admission remains pending. These execution choices do not alter scientific identities.

The scalar statistics catalogue includes spectral centroids, energies and frequency
windows, peak powers, temporal widths, material profiles and populations, and on-axis
peak intensity when the original physical modes are supplied. The quadrature diagnostic
is NaN at launch and when no distinct embedded projection exists. Short same-field CPU
comparisons use explicit transform and reduction bounds and certify temporal-width
brackets separately; they do not assign a generic relative tolerance to material rates.
Complete route/catalogue qualification remains part of phase 2.

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
