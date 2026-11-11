# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public DeviceReadback, readback!, copy_upload!

"""
    DeviceReadback(execution, element_type, dimensions...)

Prepare numerical scratch and a persistent dense host mirror of the same shape.

`element_type` is fixed-size numerical data and dimensions are nonnegative integers.
The public fields `device`, `host` and `execution` retain their prepared owners.
CPU execution uses one independently allocated zero-filled Array for both buffers;
a qualified device extension allocates device scratch plus a distinct zero-filled
host Array. Callers write only `device` during numerical work and call `readback!`
before inspecting `host`. Values retain their original units and element type.

Preparation allocates; repeated readback reuses all storage. Internal buffers must
not be replaced or resized. The ordinary engine memory ledger counts both buffers,
deduplicating the shared CPU owner. No global workspace or numerical reduction is
introduced. Throw `ArgumentError` for unsupported execution/type/dimensions, and
`OverflowError` if the requested payload cannot fit in a machine integer.
"""
struct DeviceReadback{D, H, E}
    device::D
    host::H
    execution::E

    function DeviceReadback(
            execution::E, ::Type{T}, dimensions::Vararg{Integer, N},
        ) where {E <: KernelExecution, T, N}
        isbitstype(T) && T <: Number && sizeof(T) > 0 ||
            throw(ArgumentError("readback scratch needs fixed-size numerical elements"))
        all(>=(0), dimensions) ||
            throw(ArgumentError("readback dimensions must be nonnegative"))
        shape = map(Int, dimensions)
        count = foldl(Base.checked_mul, shape; init = 1)
        Base.checked_mul(count, Base.elsize(Array{T, N}))
        device, host = _readback_storage(execution, T, shape)
        return new{typeof(device), typeof(host), E}(device, host, execution)
    end
end

function _readback_storage(execution::KernelExecution, ::Type{T}, shape) where {T}
    oncpu(execution) ||
        throw(ArgumentError("this execution backend has no qualified readback storage"))
    device = dzeros(execution, T, shape...)
    return device, device
end

function _readback_copy!(host, device, execution)
    return copy_host!(host, device, execution)
end

"""
    readback!(work::DeviceReadback, execution)

Synchronize prepared numerical scratch into its persistent host mirror and return it.

`execution` must be the exact prepared execution policy. Device storage is validated
before copying; both owners remain retained until normal or exceptional completion.
CPU scratch already aliases its host mirror, so this is an exact no-op. No conversion,
reduction or allocation occurs in the warmed CPU path. Throw `ArgumentError` for an
execution/type/storage mismatch, or `DimensionMismatch` for changed dimensions.
"""
function readback!(work::DeviceReadback, execution::KernelExecution)
    execution === work.execution ||
        throw(ArgumentError("readback execution must match its prepared owner"))
    size(work.device) == size(work.host) ||
        throw(DimensionMismatch("readback buffers no longer have the same shape"))
    eltype(work.device) === eltype(work.host) ||
        throw(ArgumentError("readback buffers must preserve their element type"))
    if work.device === work.host
        oncpu(execution) ||
            throw(ArgumentError("only CPU readback may share its host storage"))
        return work.host
    end
    return _readback_copy!(work.host, work.device, execution)
end

"""
    copy_upload!(destination, source, execution)

Copy a dense host Array into matching already allocated numerical storage.

Shape and element type must agree exactly. A qualified device extension validates
the destination's live owned context, retains both owners through synchronous
completion or error cleanup, and restores the caller's task-local context/stream.
CPU execution accepts dense Arrays: identical storage is an exact no-op, while
other overlapping headers refuse before writes. This operation does not allocate
a replacement destination or change `copy_device!`'s device-to-device contract.
Return `destination`. Throw `ArgumentError` for unsupported storage, precision,
context or overlap, and `DimensionMismatch` for different shapes.
"""
function copy_upload!(destination, source, execution::KernelExecution)
    oncpu(execution) ||
        throw(ArgumentError("this execution backend has no qualified host upload copy"))
    destination isa Array && source isa Array ||
        throw(ArgumentError("CPU upload copies require dense Array storage"))
    size(destination) == size(source) ||
        throw(DimensionMismatch("upload copy shapes must agree"))
    eltype(destination) === eltype(source) ||
        throw(ArgumentError("upload copies preserve the prepared element type"))
    destination === source && return destination
    _transfer_arrays_alias(destination, source) &&
        throw(ArgumentError("upload copies refuse overlapping storage"))
    copyto!(destination, source)
    return destination
end
