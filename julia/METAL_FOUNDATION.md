# Internal Metal execution and transform foundation

This extension supplies dense device storage, synchronous indexed launches, copies,
and temporal/spatial FFTs for the existing numerical interfaces. It does not admit
public Metal propagation sessions. Complete device source/reduction/history/service
composition and the DF32 arithmetic qualification remain separate prerequisites.

## Task, device and queue ownership

`device_execution(Val(:metal), device_index, tasks; check_body=false)` uses a
zero-based device ordinal and a positive host worker budget. It retains the preparing
Julia `Task`, the selected `MTLDevice` and that task/device's public `global_queue`.
Every operation refuses a different task before allocation or launch. Within the
owning task, it selects the retained device, verifies queue identity, performs the
operation, synchronizes and restores the prior task-local device in `finally`.
Closures and numerical owners remain rooted until synchronous completion, including
the failure cleanup path. No global queue is replaced and no private queue ownership
is promised.

The pinned KernelAbstractions backend launches through Metal's current task/device
queue, and the MPSGraph FFT implementation uses that same queue explicitly. This
provides one ordering domain for pointwise operations, copies and transforms. The
foundation qualifies Metal.jl task-local state, not an unrelated foreign raw-driver
context. The Python bridge must independently demonstrate that its entries retain
the same Julia task before public admission. GPU asynchronous scheduling and graph
optimization are outside this first synchronous implementation.

Only the two extension-owned kernel function types specialize
`KernelAbstractions.compiler_options`, with `fastmath=false` and `always_inline=true`.
Their workgroup and index-domain type parameters preserve the pinned KA `_Size`
invariant, avoiding the initial generic-method ambiguity tracked in
[#742](https://github.com/LupoLab/solprop/issues/742). No global method override changes
unrelated Metal or KA kernels. These flags do not imply gradual underflow on Apple
hardware.

## Dense storage and copies

Numerical arrays use explicit `Metal.PrivateStorage`. Shared host/device arrays and
wrapped device views are currently refused; host arrays and supported host views are
copied into independent private storage. Packed host Boolean arrays are expanded into
ordinary Boolean elements before upload. The element type is preserved, and pointer
fields, binary64, 128-bit integers and zero-size elements are refused. Nested kernel
captures must contain supported scalar values and validated dense private arrays.

Device-to-device copies require identical shape and element type on the prepared
device. Exact object identity is a no-op; every other occupied-byte overlap refuses
before writes. Overlap uses GPU virtual addresses and byte offsets rather than Julia
array-header identity, including independently wrapped headers and unequal element
types. Device-to-host copies accept a matching owned dense Julia `Array`. Device
array spans must fit their retained `MTLBuffer` and have whole-element offsets.

The ledger counts a retained `MTLBuffer` once even if multiple dense headers share
it, using the full buffer length. This deliberately distinguishes numerical array
payload from native objects, allocator reservations, command/compiler state and
framework caches. The public memory ledger has no claim about total process RSS or
all Apple GPU memory.

## Transform semantics and retained memory

Temporal plans transform dimension one of dense Float32 or ComplexF32 matrices.
Real transforms use `n_time÷2+1` complex rows; complex transforms use `n_time` rows.
Both directions preserve their input, and their buffers must be disjoint. The raw
negative-sign forward transform and raw backward transform retain the shared seam
convention. Inverse application multiplies by the prepared `Float32(1/n_time)` once.
No physical units, Fourier sign convention or amplitude scale is changed.
Real temporal lengths must be even: pinned Metal.jl's graph builder does not set
`roundToOddHermitean`, whose Apple default is false. Refuse an odd length before
planning rather than let the graph's even output shape disagree with the caller.

Spatial plans transform a nonempty strictly increasing axis tuple within Julia
dimensions 1 through 4 of a rank-1 through rank-16 ComplexF32 array. The vendor API
supports only the four trailing tensor dimensions, which Metal.jl's reversed layout
maps to Julia's first four. All other dimensions are batches. Application is in place
and requires the same input/output array owner. The raw backward transform receives
one multiplication by the reciprocal transformed volume. Shape, precision, task,
device, offset representation and overlap are validated before calling MPSGraph.
Widening the odd-real and transformed-axis boundaries requires separate qualification
under [#744](https://github.com/LupoLab/solprop/issues/744).

Preparation warms both directions on temporary zeroed arrays of the exact shapes,
preserving the supplied buffers. Those arrays are rooted through queue completion
on success and failure: the scoped warm closure captures both owners until cleanup
synchronization ends. The controlled enqueue-then-error/retry regression covers
[#746](https://github.com/LupoLab/solprop/issues/746). The arrays are not retained by
the plan wrappers.
Pinned `MtlFFTPlan` stores input/output
dimensions and axes, with an initially undefined inverse-plan cache; it stores no
explicit numerical work array or exposed native workspace byte count. Therefore its
numerical-array ledger entry is empty, **not** a zero native-workspace assertion.

Metal.jl caches MPSGraph graphs by device and transform structure in a process-wide
dictionary without a source-level size limit. Its FFT graph has no constant arrays,
but MPSGraph compiler/cache/workspace memory remains opaque. Each execution also
constructs host feed/result dictionaries. The pinned source avoids compiled graph
executables because they retain every passed output buffer. This implementation
does not mutate the vendor-global cache on session close. Repeated shape preparation,
release and collection require separate measurements and a supported retention policy
under [#743](https://github.com/LupoLab/solprop/issues/743). Neither an empty plan-array
ledger nor `Metal.@allocated == 0` establishes complete native memory accounting.

## Arithmetic qualification and the DF32 blocker

The primary Metal Shading Language 4.1 specification permits flushing Float32
subnormal inputs and outputs to zero (§8.1, p.368) and permits round-to-zero as well
as round-to-nearest-even (§8.2, p.368). It defines explicit `fma` as one correctly
rounded exact product-plus-add (p.206), while ordinary contraction is separately
controlled (§1.6.5). Thus `fastmath=false` alone cannot establish the error-free
transformation hypotheses used by the current DF32 implementation.

The initial actual-device primitive probe ran on an Apple M5 Max with 40 GPU cores,
macOS 27.0.1, Julia 1.13.3, Metal.jl 1.12.0, GPUCompiler 2.21.0 and
KernelAbstractions 0.10.5. It tested 136 arithmetic/TwoSum/TwoProd/cancellation results
against independently rounded 256-bit expressions. Twenty checks failed, all involving
subnormal inputs, outputs or residuals. Normal positive/negative odd/even ties,
explicit FMA versus separately rounded multiply/add, and the tested DF32 cancellation
witness passed. Generated LLVM/AIR retained separate `fmul`/`fadd`, explicit
`air.fma.f32` and `air.compile.fast_math_disable`.

Two failures require only normal inputs. First,
`2*floatmin(Float32) - nextfloat(floatmin(Float32))` should produce the largest
subnormal, bit pattern `0x007fffff`, but produced zero. Second, for
`a=b=ldexp(nextfloat(1f0),-50)`, the rounded product is normal (`0x0d800002`), while the
exact FMA product residual is `2^-146` (`0x00000008`); the device residual was zero.
Subnormal-input division also produced infinity when the gradually underflowing
reference remained finite. These are actual failures of the present DF32 hypotheses,
not a request to widen a rounding tolerance or assume a restricted domain.

[#739](https://github.com/LupoLab/solprop/issues/739) preserves a standalone strict
reproducer and tracks the needed numerical design/qualification. Full DF32 support
remains blocked. The base execution and FFT checks below do not waive that failure.

## Focused numerical checks

Launch/copy witnesses use small exactly representable integers, hard masks and exact
ownership/no-op contracts. FFT witnesses compare represented binary32 input against
a direct DFT evaluated with 256-bit arithmetic. Their finite qualification model
assumes rounded butterfly/product operations and twiddle components accurate to two
unit roundoffs. This is a stated model tested on small transforms, not a proof of
arbitrary MPSGraph FFT sizes or undocumented vendor twiddle generation.

For tested axis length `n`, let `g=gamma(64*n)`, exact element-magnitude bound `M`
and preceding element-error bound `E`. One raw transform uses the outward recurrence
`M_next=n*M` and
`E_next=n*(E + g*(M+E)) + 64*n*floatmin(Float32)`.
The count covers the tested radix-two stages, complex products and real splitting;
the last term explicitly permits flushed subnormal arithmetic rather than assuming
gradual underflow. Compose all forward and reverse axes before the final normalization.
The tested volumes are powers of two, so final normal scaling is exact and its
possible flushed result contributes less than one `floatmin(Float32)`. These budgets
are fixed from input magnitudes and shapes before comparing device discrepancies.

Warmed `Metal.@allocated` checks count allocations visible to Metal.jl only. Host
bookkeeping and opaque framework memory are separate; CPU's zero-allocation
advancement requirement is not silently extended to all vendor host calls. Negative
tests preserve owners on wrong-task, wrong-shape/type, unsupported storage,
foreign-offset overlap and malformed offset refusals. One device does not qualify
multiple-device restoration or cross-device copies.

The device test modules are explicit opt-in files; an unavailable Metal device fails
their prerequisite rather than skipping tests. The CPU aggregate must not load them
implicitly. A focused invocation using the pinned prepared Metal environment is:

```sh
JULIA_PKG_OFFLINE=true JULIA_NUM_THREADS=2 JULIA_LOAD_PATH='@:@stdlib' \
  julia --startup-file=no --check-bounds=yes --project=julia/environments/metal \
  -e 'using Test; include("julia/test/metal_execution.jl");
      include("julia/test/metal_transforms.jl")'
```

## Observed foundation results and outstanding quality

On the M5 Max environment recorded above, the final bounds-enabled two-thread device
suite passed 258 assertions in six test sets plus two device prerequisites. The tests
cover private storage, copied ownership, indexed/column launches, task and queue
boundaries, temporal/spatial normalization, independent byte-overlap and offset
checks, native-shape admission and preparation-failure cleanup. Warmed pointwise and
temporal-cycle calls each reported zero Metal.jl device-array allocation. For an
8-by-3 ComplexF32 fixture, the corresponding host allocation measurements were
1,376 bytes for one indexed launch, 7,712 bytes for one forward/normalized-inverse
cycle and 1,088 bytes for one device-to-host copy. These are bounded implementation
facts, not complete GPU memory measurements or a propagation benchmark.

The unfiltered initialized-device quality run still **fails**. Aqua passed all 11
checks. JET analyzed all 2,067 core and 59 extension definitions with zero cached;
it reported 55 possible core errors and 11 extension errors, and all six concrete
leaf call checks failed. Representative vendor traces use nullable Objective-C
device/command-buffer properties in `is_virtual`, `MTLLogState`, `maybe_collect`,
`commit!` and `pointer`. Broad package analysis also reaches unported core array/
backend assumptions and vendor graph/command alternatives. No reports were filtered
or suppressed. These failures remain tracked in
[#745](https://github.com/LupoLab/solprop/issues/745).

The numeric foundation, vendor quality, complete device sessions and DF32 are
separate qualification statements. Only the bounded foundation tests above passed;
public Metal selection remains closed while the other requirements are unresolved.

## Primary sources checked

- [Metal Shading Language 4.1 specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf):
  §§1.6.5, 8.1, 8.2 and the explicit `fma` definition were read directly, including
  the printed rounding/denormal and FMA pages.
- [Metal.jl v1.12.0](https://github.com/JuliaGPU/Metal.jl/tree/v1.12.0):
  `src/state.jl`, `src/MetalKernels.jl`, `src/compiler/execution.jl`, `src/array.jl`,
  `src/memory.jl` and `src/pool.jl` define task-local device/queue selection, compiler
  options, private array spans, ownership and the scope of allocation counters.
- [Metal.jl FFT implementation](https://github.com/JuliaGPU/Metal.jl/blob/v1.12.0/src/fft.jl)
  and [MPSGraph execution](https://github.com/JuliaGPU/Metal.jl/blob/v1.12.0/lib/mpsgraphs/execution.jl)
  define raw normalization, graph keys, queue ordering and cache ownership.
- [Apple's Hermitian-to-real transform](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/hermiteantorealfft(_:axes:descriptor:name:))
  and [odd-output flag](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraphfftdescriptor/roundtooddhermitean)
  specify the trailing-four-dimension restriction and default even output shape.
- [KernelAbstractions v0.10.5](https://github.com/JuliaGPU/KernelAbstractions.jl/tree/v0.10.5):
  `src/backend_launch.jl` and `src/nditeration.jl` define the owned compiler-options
  hook and the workgroup/domain type invariants used by the extension.
