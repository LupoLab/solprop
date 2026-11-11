# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct _CUDAStorageSpace
    context::CUDA.CuContext
end

function _cuda_retained_allocation(array::CUDAOwnedArray{T}) where {T}
    isbitstype(T) && sizeof(T) > 0 ||
        throw(ArgumentError("CUDA transfer storage requires fixed-size elements"))
    # Pinned CUDACore 6.5.1: DataRef[] checks both per-reference release and the
    # shared reference count before yielding Managed{DeviceMemory}. Derived headers
    # copy the DataRef but retain the identical Managed allocation object.
    managed = array.data[]
    CUDA.CUDACore.isvalid(managed.mem.ctx) ||
        throw(ArgumentError("CUDA transfer storage belongs to a released context"))
    bytes = sizeof(managed)
    bytes >= 0 && array.offset >= 0 ||
        throw(ArgumentError("CUDA transfer storage has invalid allocation bounds"))
    all(>=(0), size(array)) ||
        throw(ArgumentError("CUDA transfer storage has invalid dimensions"))
    count = foldl(Base.checked_mul, size(array); init = 1)
    extent = Base.checked_mul(count, Base.elsize(array))
    Base.checked_add(array.offset, extent) <= bytes ||
        throw(ArgumentError("CUDA transfer storage extends outside its retained payload"))
    return managed, extent
end

function SolpropJulia._transfer_storage_space(array::CUDAOwnedArray)
    managed, extent = _cuda_retained_allocation(array)
    return _CUDAStorageSpace(managed.mem.ctx)
end

function SolpropJulia._transfer_storage_span(array::CUDAOwnedArray)
    isempty(array) && return nothing
    managed, extent = _cuda_retained_allocation(array)
    # Read the pinned allocation pointer without acquiring stream ownership or
    # marking it escaped. This is a numerical-address comparison, not a submission.
    first = Base.checked_add(UInt(pointer(managed.mem)), UInt(array.offset))
    return first, Base.checked_add(first, UInt(extent))
end

function SolpropJulia._transfer_writable_destination(array::CUDAOwnedArray)
    isempty(array) && return nothing
    _cuda_retained_allocation(array)
    return nothing
end

function SolpropJulia._memory_array_owner(array::CUDAOwnedArray)
    managed, extent = _cuda_retained_allocation(array)
    # DeviceMemory.bytesize is the full retained explicit payload, unlike a
    # derived array's visible sizeof. A foreign wrapper only describes its own
    # supplied extent; it does not retain or invent an external allocation owner.
    return managed, sizeof(managed)
end

function SolpropJulia._memory_array_owner(array::CUDA.CuArray)
    throw(ArgumentError("CUDA memory inventory requires qualified device-memory storage"))
end

function SolpropJulia._retained_storage(plan::CUDAFFTPlan)
    buffer = plan.plan.buffer
    # Real transforms can retain a preservation array; complex plans carry nothing.
    return buffer === nothing ? () : (buffer,)
end

# Unlike a normal lexical preserve inside a submission closure, preserving the
# closure here also roots its captures while the error path drains queued work.
function _with_cuda_storage_transfer(body::F, execution::CUDAExecutionPolicy) where {F}
    return GC.@preserve body execution begin
        _with_cuda_execution(execution) do
            try
                result = body()
                CUDA.synchronize(execution.backend.stream)
                return result
            catch error
                try
                    CUDA.synchronize(execution.backend.stream)
                catch cleanup
                    throw(CompositeException([error, cleanup]))
                end
                rethrow()
            end
        end
    end
end

function SolpropJulia._readback_storage(
        execution::CUDAExecutionPolicy, ::Type{T}, shape,
    ) where {T}
    device = SolpropJulia.dzeros(execution, T, shape...)
    return device, zeros(T, shape)
end

function _cuda_upload_destination(array::CUDAOwnedArray, execution)
    managed, extent = _cuda_retained_allocation(array)
    managed.mem.ctx == execution.backend.context ||
        throw(ArgumentError("CUDA upload storage belongs to another execution context"))
    return nothing
end

function SolpropJulia.copy_upload!(
        destination::CUDAOwnedArray, source::Array, execution::CUDAExecutionPolicy,
    )
    _cuda_upload_destination(destination, execution)
    _cuda_shape(destination, source)
    return _with_cuda_storage_transfer(execution) do
        copyto!(destination, source)
        destination
    end
end

function SolpropJulia.copy_upload!(destination, source, execution::CUDAExecutionPolicy)
    throw(ArgumentError("CUDA upload copies require dense device and host Array storage"))
end

function SolpropJulia._readback_copy!(
        host, device::CUDAOwnedArray, execution::CUDAExecutionPolicy,
    )
    _cuda_upload_destination(device, execution)
    _cuda_shape(host, device)
    return _with_cuda_storage_transfer(execution) do
        SolpropJulia.copy_host!(host, device, execution)
    end
end
