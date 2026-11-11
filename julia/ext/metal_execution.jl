# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct MetalExecution{Q <: Metal.BatchedCommandQueue}
    kernel::Metal.MetalBackend
    device::Metal.MTLDevice
    owner::Task
    queue::Q
end

const MetalExecutionPolicy = SolpropJulia.KernelExecution{<:MetalExecution}
const MetalOwnedArray{T, N} = Metal.MtlArray{T, N, Metal.PrivateStorage}

"""
    SolpropJulia.device_execution(Val(:metal), device_index::Int, tasks::Int;
                                 check_body=false)

Prepare synchronous Metal execution bound to the current Julia task.

`device_index` is zero-based and `tasks` is a positive host worker budget, not a GPU
launch size. Retain the public queue for this task/device so KernelAbstractions and
MPSGraph use the same ordering. Calls refuse another task before work, synchronize the
retained queue and restore the previous task-local device, including on failure.
Dense arrays use explicit PrivateStorage and preserve their element type. Binary64,
raw pointers, shared host/device storage and wrapped device views are not admitted.
Only owned kernel function types receive explicit inlining and disabled fast math;
this does not qualify DF32 or gradual underflow on Metal.

Throw `ArgumentError` for an invalid budget/index or unavailable Metal device.
"""
function SolpropJulia.device_execution(
        ::Val{:metal}, device_index::Int, tasks::Int; check_body::Bool = false,
    )
    tasks > 0 || throw(ArgumentError("the Metal host task budget must be positive"))
    device_index >= 0 || throw(ArgumentError("the Metal device index must be nonnegative"))
    devices = Metal.devices()
    device_index < length(devices) ||
        throw(ArgumentError("the Metal device index is unavailable"))
    Metal.functional() || throw(ArgumentError("Metal is not functional on this host"))
    previous = Metal.device()
    device = devices[device_index + 1]
    try
        Metal.device!(device)
        backend = MetalExecution(
            Metal.MetalBackend(), device, current_task(), Metal.global_queue(device),
        )
        return SolpropJulia.KernelExecution(backend, tasks, check_body)
    finally
        Metal.device!(previous)
    end
end

@inline function _metal_task(execution::MetalExecutionPolicy)
    current_task() === execution.backend.owner ||
        throw(ArgumentError("Metal execution must remain on its preparing Julia task"))
    return nothing
end

function _with_metal_execution(body::F, execution::MetalExecutionPolicy) where {F}
    _metal_task(execution)
    backend = execution.backend
    previous = Metal.device()
    return GC.@preserve body execution begin
        try
            Metal.device!(backend.device)
            Metal.global_queue(backend.device) === backend.queue ||
                throw(ArgumentError("Metal execution's public task queue has changed"))
            result = body()
            Metal.synchronize(backend.queue)
            return result
        catch error
            try
                Metal.synchronize(backend.queue)
            catch cleanup
                throw(CompositeException([error, cleanup]))
            end
            rethrow()
        finally
            Metal.device!(previous)
        end
    end
end

function SolpropJulia.synchronize_execution(execution::MetalExecutionPolicy)
    return _with_metal_execution(execution) do
        nothing
    end
end

@generated function _metal_element_type(::Type{T}) where {T}
    if !isbitstype(T) || iszero(sizeof(T)) ||
            T <: Union{Float64, Int128, UInt128, Ptr, Core.LLVMPtr, Metal.MtlPtr}
        return :(throw(ArgumentError("Metal elements require pointer-free supported bits")))
    end
    checks = [:(_metal_element_type($field)) for field in fieldtypes(T)]
    return quote
        $(checks...)
        nothing
    end
end

function _metal_dimensions(::Type{T}, dimensions::NTuple{N, Integer}) where {T, N}
    _metal_element_type(T)
    all(>=(0), dimensions) ||
        throw(ArgumentError("Metal array dimensions must be nonnegative"))
    shape = map(Int, dimensions)
    count = foldl(Base.checked_mul, shape; init = 1)
    Base.checked_mul(count, sizeof(T))
    return shape
end

function _metal_array(array::MetalOwnedArray{T}, execution::MetalExecutionPolicy) where {T}
    _metal_task(execution)
    Metal.device(array) == execution.backend.device ||
        throw(ArgumentError("Metal storage belongs to a different device"))
    _metal_dimensions(T, size(array))
    bytes, offset = UInt(sizeof(array)), array.offset
    offset >= 0 || throw(ArgumentError("Metal storage has a negative byte offset"))
    iszero(offset % sizeof(T)) ||
        throw(ArgumentError("Metal storage requires whole-element byte offsets"))
    available = UInt(array.data[].buffer.length)
    UInt(offset) <= available && bytes <= available - UInt(offset) ||
        throw(ArgumentError("Metal array dimensions exceed their retained buffer"))
    return nothing
end

function _metal_array(array, execution::MetalExecutionPolicy)
    _metal_task(execution)
    throw(ArgumentError("Metal execution requires dense private-storage MtlArrays"))
end

@inline function _metal_overlap(left::MetalOwnedArray, right::MetalOwnedArray)
    if isempty(left) || isempty(right) || Metal.device(left) != Metal.device(right)
        return false
    end
    first_left = Base.checked_add(
        UInt(Metal.MTL.gpu_address(left.data[].buffer)),
        UInt(left.offset)
    )
    first_right = Base.checked_add(
        UInt(Metal.MTL.gpu_address(right.data[].buffer)),
        UInt(right.offset)
    )
    return first_left <= first_right ? first_right - first_left < UInt(sizeof(left)) :
        first_left - first_right < UInt(sizeof(right))
end

function _metal_shape(destination, source)
    size(destination) == size(source) ||
        throw(DimensionMismatch("Metal copy shapes must agree"))
    eltype(destination) === eltype(source) ||
        throw(ArgumentError("Metal copies preserve the prepared element type"))
    return nothing
end

function SolpropJulia.copy_device!(
        destination::MetalOwnedArray{T, N}, source::MetalOwnedArray{T, N},
        execution::MetalExecutionPolicy,
    ) where {T, N}
    _metal_array(destination, execution)
    _metal_array(source, execution)
    _metal_shape(destination, source)
    destination !== source && _metal_overlap(destination, source) &&
        throw(ArgumentError("Metal copies refuse partial device storage overlap"))
    return _with_metal_execution(execution) do
        destination === source || copyto!(destination, source)
        destination
    end
end

function SolpropJulia.copy_device!(destination, source, execution::MetalExecutionPolicy)
    _metal_array(destination, execution)
    _metal_array(source, execution)
    _metal_shape(destination, source)
    throw(ArgumentError("Metal copies require matching dense device arrays"))
end

function SolpropJulia.copy_host!(
        destination::Array{T, N}, source::MetalOwnedArray{T, N},
        execution::MetalExecutionPolicy,
    ) where {T, N}
    _metal_array(source, execution)
    _metal_shape(destination, source)
    return _with_metal_execution(execution) do
        copyto!(destination, source)
        destination
    end
end

function SolpropJulia.copy_host!(destination, source, execution::MetalExecutionPolicy)
    _metal_task(execution)
    destination isa Array ||
        throw(ArgumentError("Metal host copies require an owned dense Array destination"))
    _metal_array(source, execution)
    _metal_shape(destination, source)
    throw(ArgumentError("Metal host copies require matching dense arrays"))
end

function SolpropJulia.upload(
        execution::MetalExecutionPolicy, input::AbstractArray{T, N},
    ) where {T, N}
    _metal_task(execution)
    shape = _metal_dimensions(T, size(input))
    if input isa Metal.MtlArray
        _metal_array(input, execution)
    elseif !(input isa Union{Array, BitArray})
        SolpropJulia._transfer_storage_span(input)
    end
    host = input isa Array || input isa MetalOwnedArray ? input : Array(input)
    return _with_metal_execution(execution) do
        output = Metal.MtlArray{T, N, Metal.PrivateStorage}(undef, shape)
        copyto!(output, host)
        output
    end
end

function SolpropJulia.dzeros(
        execution::MetalExecutionPolicy, ::Type{T}, dimensions::Vararg{Integer, N},
    ) where {T, N}
    _metal_task(execution)
    shape = _metal_dimensions(T, dimensions)
    return _with_metal_execution(execution) do
        output = Metal.MtlArray{T, N, Metal.PrivateStorage}(undef, shape)
        fill!(output, zero(T))
        output
    end
end

@inline _metal_capture(array::AbstractArray, execution) = _metal_array(array, execution)

@generated function _metal_capture(value::T, execution) where {T}
    isconcretetype(T) || return :(
        throw(ArgumentError("Metal kernels require concrete captured values"))
    )
    T <: Union{Float64, Int128, UInt128, Ptr, Core.LLVMPtr, Metal.MtlPtr} && return :(
        throw(ArgumentError("Metal kernels refuse binary64 or unowned pointer captures"))
    )
    (isbitstype(T) || T <: Union{Function, Tuple, NamedTuple}) || return :(
        throw(ArgumentError("Metal kernels require scalar or dense device-array captures"))
    )
    checks = [
        :(_metal_capture(getfield(value, $index), execution))
            for index in 1:fieldcount(T)
    ]
    return quote
        $(checks...)
        nothing
    end
end

KernelAbstractions.@kernel function _metal_each_kernel(body, rows)
    index = KernelAbstractions.@index(Global, Linear)
    row = mod(index - 1, rows) + 1
    column = div(index - 1, rows) + 1
    @inline body(row, column)
end

KernelAbstractions.@kernel function _metal_column_kernel(body)
    column = KernelAbstractions.@index(Global, Linear)
    @inline body(column)
end

const MetalKernelFunction = Union{
    typeof(_metal_each_kernel(Metal.MetalBackend()).f),
    typeof(_metal_column_kernel(Metal.MetalBackend()).f),
}

function KernelAbstractions.compiler_options(
        kernel::KernelAbstractions.Kernel{Metal.MetalBackend, W, N, F},
    ) where {
        W <: KernelAbstractions.NDIteration._Size,
        N <: KernelAbstractions.NDIteration._Size, F <: MetalKernelFunction,
    }
    return (; fastmath = false, always_inline = true)
end

function SolpropJulia._each!(
        body::F, execution::MetalExecutionPolicy, rows::Int, columns::Int, heavy,
    ) where {F}
    _metal_task(execution)
    count = Base.checked_mul(rows, columns)
    _metal_capture(body, execution)
    return _with_metal_execution(execution) do
        _metal_launch_points!(body, execution, rows, count)
    end
end

function _metal_launch_points!(
        body::F, execution::MetalExecutionPolicy, rows::Int, count::Int,
    ) where {F}
    _metal_each_kernel(execution.backend.kernel)(body, rows; ndrange = count)
    return nothing
end

function SolpropJulia._each_column!(
        body::F, execution::MetalExecutionPolicy, columns::Int,
    ) where {F}
    _metal_task(execution)
    _metal_capture(body, execution)
    return _with_metal_execution(execution) do
        _metal_launch_columns!(body, execution, columns)
    end
end

function _metal_launch_columns!(
        body::F, execution::MetalExecutionPolicy, columns::Int,
    ) where {F}
    _metal_column_kernel(execution.backend.kernel)(body; ndrange = columns)
    return nothing
end

# Count one retained allocation even when several dense foreign headers share it.
# Driver/allocator reservations and MPSGraph caches are separate process overhead.
function SolpropJulia._memory_arrays!(ledger, seen, path, array::MetalOwnedArray)
    owner = array.data[].buffer
    if !haskey(seen, owner)
        owner.length <= typemax(Int) ||
            throw(ArgumentError("Metal retained buffer size exceeds the ledger range"))
        seen[owner] = true
        ledger[Symbol(path)] = Int(owner.length)
    end
    return nothing
end
