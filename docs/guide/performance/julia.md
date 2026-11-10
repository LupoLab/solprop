# The optional Julia backend

Julia support is being implemented under [SEP 48](../../dev/seps/sep-0048-julia-backend.md).
The package and explicit setup commands are available on its implementation branch.
Numerical sessions and device qualification remain in progress; installing a device
environment does not establish that its propagation kernels are qualified.

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
ambiguity reports the available IDs. The forthcoming session interface will select an
identity through the `setup_id` backend option and record it as execution metadata.

`SimulationSettings.backend_options` carries backend-specific execution choices. It
defaults to an independently owned empty mapping, and preparation takes another owned
copy through `settings.execution_options()`. These choices survive spawned workers and
may be replaced after translation without changing the scientific configuration or
restart identity. The selected backend refuses unsupported keys before constructing a
workspace. The Julia adapter will admit `setup_id` and `wide`; the existing native and
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
