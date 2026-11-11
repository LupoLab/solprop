# CUDA storage and control readback

This internal boundary connects numerical arrays to the existing Julia-owned
control and transport code. It adds no device session, physics route, reduction,
public backend admission or change to the CPU arithmetic graph. CUDA execution and
FFT limitations remain those in [CUDA_FOUNDATION.md](CUDA_FOUNDATION.md).

## Address spaces and alias intervals

`_transfer_storage_span(array)` retains its existing result: either `nothing` for
empty storage or a checked, half-open `(lo, hi)` tuple of machine addresses. Known
host wrappers still use their whole parent span conservatively. The new independent
`_transfer_storage_space(array)` classifies qualified storage:

- Dense Julia Arrays and PythonCall arrays use `_HostStorageSpace()`; supported
  wrappers inherit their parent's space.
- Dense device-memory CuArrays use a token containing the retained `CuContext`,
  including its vendor context identity. Raw equal pointer integers in different
  contexts do not establish overlap.
- Unknown storage refuses. CUDA mapped/unified memory does not inherit the dense
  device-memory contract. A released nonempty array reference or invalid context
  refuses before obtaining a numerical address. Existing empty alias semantics
  remain unchanged.

Only equal address spaces compare `Base.mightalias` or byte intervals. Mutable
gather-index dependencies are still checked in both directions, including across
different numerical-data spaces: a host output can corrupt a host index vector
used by a device-backed view. Known wrapper parents remain conservative; dense
derived CuArray headers carry their own occupied interval. Arbitrary gathered
writable layouts remain refused by the existing affine/injective guard.

The CUDA accessor first validates `array.data[]`, which checks both the particular
reference and shared lifetime, then checks context validity and the complete header
extent against retained bytes. It reads the allocation pointer plus the checked
byte offset without acquiring a vendor stream or marking a pointer as escaped.
The element stride is `Base.elsize(array)`, including padding, rather than an
assumption that `sizeof(eltype(array))` is the array stride. These representations
were checked against [CUDACore 6.5.1 array storage][cuda-array] and
[GPUArrays 12.2.0 reference ownership][gpu-reference].

`_retained_storage(value)` is a narrow opaque-owner extension point, defaulting to
`()`. The generated engine owner walk follows its explicitly returned numerical
buffers; it does not reflect through vendor handles, caches or unrelated metadata.
For `CUDAFFTPlan`, a real-transform preservation array is returned when present;
complex plans return `()`. This brings plan scratch into caller alias checks
without treating the whole vendor runtime as session-owned data.

## Allocation payload and memory ledgers

`_memory_array_owner(array)` returns `(owner_key, payload_bytes)` independently of
alias intervals. The host default follows the same parent chain and counts the
same owning Array as before. The CUDA method returns the live retained `Managed`
allocation and `sizeof(managed)`, whose `DeviceMemory.bytesize` describes the full
explicit payload. Derived headers copy a DataRef but share this Managed owner;
views and reshapes therefore count the allocation once even when they expose only
part of it. This is distinct from their occupied alias intervals.

Pinned CUDACore rounds each nonempty pool allocation up to a multiple of four
bytes so subword atomics cannot touch bytes beyond the allocation. The retained
payload includes this padding; a host mirror and a transfer contain only the
logical elements. For example, a 21-element UInt8 readback retains 24 device bytes
and 21 host bytes. The independent payload assertions use the pinned allocation
rule, rather than assuming twice the logical array size. See the
[CUDACore pool allocator][cuda-pool].

A foreign `unsafe_wrap` header has its own retained memory descriptor and declared
extent. It does not retain the external allocation's original owner, so its ledger
does not invent that owner or its size. The caller must retain the external storage
for the wrapper's full use. Multiple independent foreign wrappers are not presumed
to be the same retained owner just because their pointers overlap.

cuFFT's active native workspace remains counted separately once per raw plan, and
its explicit preservation array follows the allocation-owner ledger. CUDA allocator
reservations, driver/context state and released vendor caches remain process
overhead; this interface neither estimates them nor promises zero process memory
after session close. See issues [735][workspace] and [736][cache].

## Persistent numerical readback and upload

`DeviceReadback(execution, T, dimensions...)` prepares three public fields:

| Field | Meaning |
| --- | --- |
| `device` | Owned zero-filled numerical scratch in the execution memory space. |
| `host` | Persistent dense host Array with exactly the same shape and dtype. |
| `execution` | The prepared execution policy to which both buffers belong. |

CPU execution aliases `device === host`; the usual memory ledger counts that
single owner once. A qualified CUDA execution owns distinct device and host
payloads. Nonnegative dimensions and the padded byte count are checked before
allocation. Element types are fixed-size numbers. Buffers must not be replaced
or resized after preparation.

`readback!(work, execution)` requires the exact prepared policy and returns
`work.host` after synchronous completion. It performs no reduction, conversion,
coordinate selection or controller decision. CPU readback is an exact no-op;
device readback copies into the existing mirror. Future norm/status users should
store bounded per-column or compact numerical summaries in `device`, not silently
download a full evolving state at every trial.

`copy_upload!(destination, source, execution)` copies a dense host Array into an
already allocated matching destination, preserving shape, dtype and units.
The CUDA method validates destination lifetime/context before writes, scopes the
prepared CUDA.jl context/stream, and synchronizes before returning. Both owners
remain rooted while an error path drains queued work; a cleanup failure retains
both errors. The CPU method accepts matching dense Arrays, returns immediately
for the identical array and refuses other overlapping headers before writes.
`copy_device!` still means same-context device-to-device copying; `upload` still
allocates new ownership. No global workspace or vendor policy is modified.

## Qualification scope

`device_storage.jl` and `device_readback.jl` tests cover source-free host and fake
address spaces, symmetric foreign/index dependencies, unsupported/released storage,
opaque numerical owners, payload deduplication, pre-write copy refusals and zero
warmed CPU allocation. The separate `cuda_storage.jl` test leaf requires an actual
CUDA device with scalar indexing disabled in its test process. It covers real
derived/foreign headers, unequal dtypes, contexts, released references, persistent
readbacks, transfer-error completion, and actual export admission against retained
cuFFT scratch. Hardware execution is not implied by host tests.

Actual device results and independent review must be recorded before this leaf is
called qualified. Existing unfiltered CUDA JET failures remain visible under
[741][quality]; no report filter, skip, or public-admission relaxation accompanies
this implementation. The shared storage completion is tracked in [752][storage].

[cuda-array]: https://github.com/JuliaGPU/CUDA.jl/blob/v6.5.1/CUDACore/src/array.jl
[gpu-reference]: https://github.com/JuliaGPU/GPUArrays.jl/blob/v12.2.0/src/host/abstractarray.jl
[workspace]: https://github.com/LupoLab/solprop/issues/735
[cache]: https://github.com/LupoLab/solprop/issues/736
[quality]: https://github.com/LupoLab/solprop/issues/741
[storage]: https://github.com/LupoLab/solprop/issues/752

[cuda-pool]: https://github.com/JuliaGPU/CUDA.jl/blob/v6.5.1/CUDACore/src/resources/backends.jl
