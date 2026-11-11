# CUDA execution and transform foundation

This package slice implements owned CUDA execution and dense temporal/spatial FFTs.
It does not open CUDA propagation admission. Sources, adaptive norms, numerical
statistics and monitors, history/export transport, and the Python device factory
still require their separate device integration and qualification under SEP 48.
The CPU numerical implementations and public physics defaults are unchanged.

## Execution ownership

`device_execution(Val(:cuda), device_index, tasks; check_body=false)` selects a
zero-based device ordinal and returns the existing `KernelExecution` with an
extension-owned concrete backend. The backend retains the primary CUDA context
and a private stream. `tasks` remains the host CPU budget; GPU launch geometry is
selected by the pinned KernelAbstractions backend.

Each allocation, transfer, kernel and transform enters the retained context/stream
using CUDA.jl's scoped APIs and restores its caller's CUDA.jl task-local state in
`finally`, including on failure. A completed call has synchronized that stream.
No process-global workspace, scalar-indexing policy, math mode, or FFTW planning
policy is changed.
The execution owner and its arrays must remain alive through the synchronous call.
A session-level close/release policy is outside this foundation slice.
The tests establish task-local restoration after CUDA.jl state exists. They do not
establish restoration of an independently installed raw foreign driver context on
first entry. First-entry and native CUDA coexistence qualification remains tracked
in [#733](https://github.com/LupoLab/solprop/issues/733) before public device admission.

The pinned backend is explicitly constructed with `always_inline=true` and
`fastmath=false`. CUDA.jl's default fast-math setting otherwise follows Julia's
command-line mode and can flush Float32 subnormals. The column launcher gives each
column one sequential body; the pointwise launcher visits every row/column pair.
CUDA adapts captured arrays to device arguments. The extension validates capture
storage/context first, including nested tuples and scalar records; a pointer-backed
array is not considered an inert scalar merely because its wrapper is `isbits`.
Raw host, CUDA and LLVM pointer captures refuse because they cannot establish an
owning array or the selected context, including inside otherwise scalar records.
The CPU PtrArray capture adaptation is never used on this path.

Device allocations use `CuArray{T,N,DeviceMemory}` explicitly, preserving element
type and dimensions. This avoids inheriting an ambient unified-memory preference.
The initial storage scope is dense CuArray device memory. Known host array views
can be copied during `upload` preparation, but device views are not silently staged
through the host. Kernel captures and FFT buffers must use the admitted dense form.

`copy_device!` requires exact shape/type agreement and one selected context. An
identical source/destination is an exact numerical no-op. All other byte overlap,
including offset foreign headers, refuses before writes. Address comparisons are
context-aware, use unsigned differences rather than wrapping address endpoints,
and include the actual element width. `copy_host!` is device-to-host only and requires
a dense Julia `Array` destination with the same shape/type. These helpers neither
restore physical field scales nor apply power weights or Fourier conventions.
The broader session alias/export policy remains a separate integration boundary.

## Dense FFT ownership and normalization

`TemporalFFT` accepts dense device matrices of matching binary32 or binary64
precision. Real samples use the leading `n_time ÷ 2 + 1` spectrum; complex samples
retain `n_time` frequencies. The forward cuFFT transform has the internal negative
DFT sign. The backward plan is unnormalized, and the existing inverse API applies
`1/n_time` exactly once. Both out-of-place directions preserve their input.

`SpatialFFT` accepts a dense complex device array and strictly increasing, nonempty
axis indices. Other axes are independent batches. Forward and backward transforms
are raw; the inverse applies the reciprocal product of the transformed dimensions
once. The plan wrappers retain the execution owner and validate context, precision,
shape, placement and overlap before calling cuFFT. Inverse calls require the exact
prepared execution owner, preventing a transform followed by scaling on another
stream or device. Planning preserves the supplied numerical buffers.
Both prepared and replacement buffers must have base pointers aligned to the
complex element width (8 bytes for binary32, 16 bytes for binary64). In particular,
real-transform input/output pointers need complex alignment under NVIDIA's API;
a dense foreign header with only scalar alignment refuses before planning or writes.
Retained real-transform scratch is checked by the same application guard. This is
separate from the pinned wrapper's repair of misaligned external batch starts.

The pinned cuFFT implementation has two relevant storage properties:

- `plan_rfft` retains a complex buffer for its possible inverse interface even when
  forward `mul!` does not use that buffer. `plan_brfft` retains another complex input
  preservation buffer. This slice counts both arrays, by owner identity, through a
  narrowly dispatched memory traversal. It does not traverse arbitrary CUDA handles.
- cuFFT's public `mul!` allocates dense staging for non-dense inputs/outputs, and real
  transforms with misaligned external batch starts can allocate temporary arrays.
  The admitted temporal axis-1 and complex dense spatial layouts avoid those paths.
  Wrapped non-dense arrays are refused during preparation rather than accepted with
  hidden repeated allocations.

Each wrapper also stores `workspace_bytes::Int`, queried once with `cufftGetSize`
after plan construction inside the owned context. The pinned `CuFFTPlan` has no
workspace-size field: `cufftMakePlan` returns a size, but its `handle_ctor` discards
that value and retains cuFFT's default automatic allocation. The post-construction
query supplies the current native work-area requirement; conversion to the ledger's
`Int` is checked. A named `.workspace` ledger entry is counted once per raw-plan
identity, separately from its numerical buffer. Repeated transform execution neither
queries the size nor changes the native work-area configuration. Direct-query and
shared-wrapper/buffer tests cover this accounting correction
([#735](https://github.com/LupoLab/solprop/issues/735)).

The visible plan-buffer payloads and queried active native work areas are included.
CUDA allocator reservations, object overhead, other opaque handle allocations and
driver memory are separate process overhead; no RSS or complete device-memory bound
is claimed. Releasing a Julia plan also does not imply that its native work area is
immediately freed. In pinned cuFFT 6.5.1, `unsafe_free!(plan)` calls
`cufftReleasePlan`, which defers returning the handle to the process-wide
`idle_handles` cache. CUDACore's `HandleCache` destroys idle handles on its
purge/reclaim path. Its default `gc_threshold=32` requests collection when many
handles are active; it is neither an idle-handle limit nor a byte bound. The cache
exposes active/idle counts, not all internal allocation sizes. This foundation does
not mutate that vendor-global cache on session close. Repeated shape preparation,
release and reclamation require separate process-memory measurements tracked in
[#736](https://github.com/LupoLab/solprop/issues/736).
FFTW wisdom and the CPU FFT planning context have no role in these CUDA plans.

## Numerical qualification model

The launch/copy tests use exactly representable integer arithmetic, exact masks and
copy/no-op/ownership assertions. A Float32 minimum-normal times one half checks
that the actual compiled pointwise path preserves a representable subnormal.
This is a necessary compiler witness, not complete DF32 qualification.

The small radix-two FFT witnesses compare the device result against a direct DFT
formed from the represented input with 256-bit arithmetic. Their explicitly stated
model assumes componentwise rounded butterflies/products and twiddle components
accurate to two unit roundoffs. A conservative `gamma(64*n)` sum-of-input-magnitudes
budget covers the tested real splitting, complex products and radix-two stages;
a separate additive minimum-subnormal term accounts for gradual underflow. The
multiaxis witness uses an outward-rounded per-axis recurrence. With a current exact
element-magnitude bound `M`, error bound `E`, axis length `n` and `g=gamma(64*n)`,
one raw transform updates `M` to `n*M` and bounds its error by
`n*(E + g*(M+E)) + 64*n*eta`, where `eta` is the smallest subnormal. This accounts
for propagation of earlier error and the perturbed inputs of the next transform.
The forward and reverse sequences are composed before dividing by the transform
volume, with an additional `eta/2` for the final dyadic scaling. These bounds are
fixed from lengths and input magnitudes before comparing any device discrepancy.
These finite cases test that model; operation counting alone does not prove every
cuFFT plan's twiddle error or an arbitrary transform-size guarantee.

Warm repeated-call checks distinguish device array allocation from Julia host
allocation. `CUDA.@allocated` must report zero repeated device allocation after
specialization; host-side compiler/plan/driver bookkeeping is reported separately.
The tests also preserve input arrays and exercise same-context alternate streams,
foreign offset overlaps, unsupported layouts, wrong precision and failed-call state
restoration. When more than one device is available, additional context-mismatch
checks run; a one-device pass does not claim multi-device qualification.

## Focused device invocation

Use a managed CUDA setup whose pinned packages and artifacts have already been
materialized on a CPU-only host. Run on an allocated GPU node with networking
blocked by the campaign supervisor; `JULIA_PKG_OFFLINE` alone is not a network
sandbox. No package installation occurs in this runner.

```sh
JULIA_PKG_OFFLINE=true JULIA_NUM_THREADS=2 JULIA_LOAD_PATH='@:@stdlib' \
  julia --startup-file=no --check-bounds=yes --project=julia/environments/cuda \
  -e 'using Test; import CUDA; import SolpropJulia; @assert CUDA.functional();
      include("julia/test/cuda_execution.jl");
      include("julia/test/cuda_transforms.jl")'
```

The selected device test files fail when CUDA is unavailable; the CPU test aggregate
must not include them implicitly. Record the engine revision, Julia/package/toolkit/
driver versions, device model, compilation warmup, and host/device allocation facts.
The first implementation record must distinguish actual GPU results from local
syntax, host-boundary and package-analysis checks. No CUDA engine performance or
full-session parity claim follows from this foundation alone.

## Observed qualification and remaining blockers

The first complete device run used an NVIDIA A40 (compute capability 8.6), Julia
1.13.3, CUDA.jl 6.5.1, CUDA runtime 13.4.0 and driver API 13.5.0. With bounds enabled,
two Julia threads and kernel-level network isolation, all 418 assertions in the eight
focused execution/transform test sets passed. This includes the actual native
workspace query, foreign-header alignment refusals, input preservation, gradual
underflow witness and zero repeated device-allocation checks. It does not qualify
multiple devices or complete propagation sessions.

The separate unfiltered device quality run failed and remains a qualification
blocker. Aqua passed all 11 checks. JET analyzed 2,067 core and 51 extension definitions;
the core report contained 76 possible errors, the extension package also reported
errors, and all six concrete device-call checks failed. The concrete traces include
CUDACore's `stream_wait` calling `context!` with a possible `Nothing`, and
`synchronize` calling `mark_completed!` with a possible `Nothing`. Core analysis also
reaches CPU-only Polyester and AcceleratedKernels paths with device arrays. These
reports have not been filtered or suppressed. The foundation remains internal while
vendor inference and shared device-port work are resolved; neither full vendor
quality nor public CUDA session admission is claimed.

## Sources checked

The implementation was checked against the exact installed packages selected by the
committed manifests, using primary package source:

- [CUDA.jl 6.5.1](https://github.com/JuliaGPU/CUDA.jl/tree/v6.5.1): CUDACore
  `src/CUDAKernels.jl`, `src/array.jl`, and `lib/cudadrv/state.jl` define explicit
  compilation flags, typed device storage, context identity and scoped restoration.
- The same pinned distribution's cuFFT `src/fft.jl` defines plan buffers, input
  preservation, dense/strided execution, and external-batch alignment behavior;
  `src/wrappers.jl` selects automatic native work-area allocation and caches released
  handles. CUDACore `src/utils/cache.jl` defines deferred return, the active-handle
  collection threshold and idle-handle purge behavior.
- [NVIDIA's cuFFT API, sections 3.7 and 3.8.1](https://docs.nvidia.com/cuda/cufft/)
  specifies the completed-plan workspace query and default automatic allocation.
- [NVIDIA's cuFFT API, sections 3.9.2 and 3.9.3](https://docs.nvidia.com/cuda/cufft/)
  specifies complex-type alignment for both real-transform pointers.
- [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) supplies the
  indexed kernel and backend launch interface. Numerical loop bodies retain the
  engine's existing scalar operations; no new propagation method is introduced.
