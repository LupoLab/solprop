# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct CUDAExecution
    kernel::CUDA.CUDABackend
    device::CUDA.CuDevice
    context::CUDA.CuContext
    stream::CUDA.CuStream
end

const CUDAExecutionPolicy = SolpropJulia.KernelExecution{CUDAExecution}
const CUDAOwnedArray{T, N} = CUDA.CuArray{T, N, CUDA.DeviceMemory}

"""
    SolpropJulia.device_execution(Val(:cuda), device_index::Int, tasks::Int;
                                 check_body=false)

Prepare synchronous CUDA execution on a zero-based device index.

The returned `KernelExecution` retains its own CUDA stream and the selected primary
context. Calls restore CUDA.jl task-local context and stream, including on failure. `tasks`
is the positive host worker budget; it does not choose a GPU launch size. Device
arrays use explicit device memory and retain their supplied element type. CUDA kernel
compilation explicitly enables inlining and disables fast math, including its
flush-to-zero behavior. No global CUDA math or scalar-indexing policy is changed.
Restoration of an independently installed raw foreign driver context is unqualified.

Throw `ArgumentError` for a nonpositive budget, invalid device index, or unavailable
CUDA driver/device. CUDA package errors retain their original cause.
"""
function SolpropJulia.device_execution(
        ::Val{:cuda}, device_index::Int, tasks::Int; check_body::Bool = false,
    )
    tasks > 0 || throw(ArgumentError("the CUDA host task budget must be positive"))
    device_index >= 0 || throw(ArgumentError("the CUDA device index must be nonnegative"))
    CUDA.functional() || throw(ArgumentError("CUDA is not functional on this host"))
    device_index < length(CUDA.devices()) ||
        throw(ArgumentError("the CUDA device index is unavailable"))
    device = CUDA.CuDevice(device_index)
    context = CUDA.context(device)
    stream = CUDA.context!(context) do
        CUDA.CuStream()
    end
    backend = CUDAExecution(
        CUDA.CUDABackend(; always_inline = true, fastmath = false),
        device, context, stream,
    )
    return SolpropJulia.KernelExecution(backend, tasks, check_body)
end

@inline function _with_cuda_execution(body::F, execution::CUDAExecutionPolicy) where {F}
    backend = execution.backend
    return CUDA.context!(backend.context) do
        CUDA.stream!(backend.stream) do
            body()
        end
    end
end

function SolpropJulia.synchronize_execution(execution::CUDAExecutionPolicy)
    _with_cuda_execution(execution) do
        CUDA.synchronize(execution.backend.stream)
    end
    return nothing
end

@inline function _cuda_array(array::CUDAOwnedArray, execution::CUDAExecutionPolicy)
    CUDA.context(array) == execution.backend.context ||
        throw(ArgumentError("CUDA storage belongs to a different execution context"))
    return nothing
end

function _cuda_array(array, execution::CUDAExecutionPolicy)
    throw(ArgumentError("CUDA execution requires owned dense device-memory CuArrays"))
end

@inline function _cuda_overlap(left::CUDAOwnedArray, right::CUDAOwnedArray)
    if isempty(left) || isempty(right) || CUDA.context(left) != CUDA.context(right)
        return false
    end
    first_left, first_right = UInt(pointer(left)), UInt(pointer(right))
    # Subtraction avoids wrapping an address-plus-size endpoint. Different element
    # types and independent foreign headers still compare their occupied bytes.
    return first_left <= first_right ? first_right - first_left < UInt(sizeof(left)) :
        first_left - first_right < UInt(sizeof(right))
end

function _cuda_shape(destination, source)
    size(destination) == size(source) ||
        throw(DimensionMismatch("CUDA copy shapes must agree"))
    eltype(destination) === eltype(source) ||
        throw(ArgumentError("CUDA copies preserve the prepared element type"))
    return nothing
end

function SolpropJulia.copy_device!(
        destination::CUDAOwnedArray{T, N}, source::CUDAOwnedArray{T, N},
        execution::CUDAExecutionPolicy,
    ) where {T, N}
    _cuda_array(destination, execution)
    _cuda_array(source, execution)
    _cuda_shape(destination, source)
    _with_cuda_execution(execution) do
        destination !== source && _cuda_overlap(destination, source) &&
            throw(ArgumentError("CUDA copies refuse partial device storage overlap"))
        destination === source || copyto!(destination, source)
        CUDA.synchronize(execution.backend.stream)
    end
    return destination
end

function SolpropJulia.copy_device!(destination, source, execution::CUDAExecutionPolicy)
    _cuda_array(destination, execution)
    _cuda_array(source, execution)
    _cuda_shape(destination, source)
    throw(ArgumentError("CUDA copies require matching dense device arrays"))
end

function SolpropJulia.copy_host!(
        destination::Array{T, N}, source::CUDAOwnedArray{T, N},
        execution::CUDAExecutionPolicy,
    ) where {T, N}
    _cuda_array(source, execution)
    _cuda_shape(destination, source)
    _with_cuda_execution(execution) do
        copyto!(destination, source)
        CUDA.synchronize(execution.backend.stream)
    end
    return destination
end

function SolpropJulia.copy_host!(destination, source, execution::CUDAExecutionPolicy)
    destination isa Array ||
        throw(ArgumentError("CUDA host copies require an owned dense Array destination"))
    _cuda_array(source, execution)
    _cuda_shape(destination, source)
    throw(ArgumentError("CUDA host copies require matching dense arrays"))
end

function _cuda_dimensions(
        ::Type{T}, dimensions::Tuple{Vararg{Integer, N}},
    ) where {T, N}
    isbitstype(T) || throw(ArgumentError("CUDA arrays require fixed-size elements"))
    all(>=(0), dimensions) ||
        throw(ArgumentError("CUDA array dimensions must be nonnegative"))
    shape = map(Int, dimensions)
    count = foldl(Base.checked_mul, shape; init = 1)
    Base.checked_mul(count, sizeof(T))
    return shape
end

function SolpropJulia.upload(
        execution::CUDAExecutionPolicy, input::AbstractArray{T, N},
    ) where {T, N}
    shape = _cuda_dimensions(T, size(input))
    # Only recognized host storage may be materialized during preparation.
    # In particular a wrapped device view must not silently download to the host.
    if input isa CUDA.CuArray
        _cuda_array(input, execution)
    elseif !(input isa Array)
        SolpropJulia._transfer_storage_span(input)
    end
    host = input isa Array || input isa CUDAOwnedArray ? input : Array(input)
    return _with_cuda_execution(execution) do
        output = CUDA.CuArray{T, N, CUDA.DeviceMemory}(undef, shape)
        copyto!(output, host)
        CUDA.synchronize(execution.backend.stream)
        output
    end
end

function SolpropJulia.dzeros(
        execution::CUDAExecutionPolicy, ::Type{T}, dimensions::Vararg{Integer, N},
    ) where {T, N}
    shape = _cuda_dimensions(T, dimensions)
    return _with_cuda_execution(execution) do
        output = CUDA.CuArray{T, N, CUDA.DeviceMemory}(undef, shape)
        fill!(output, zero(T))
        CUDA.synchronize(execution.backend.stream)
        output
    end
end

@generated function _cuda_capture(value::T, execution) where {T}
    # Pointer-backed arrays can themselves be isbits. Traverse closure/tuple and
    # scalar-record fields before treating a value as an inert scalar capture.
    isconcretetype(T) || return :(
        throw(ArgumentError("CUDA kernels require concrete captured values"))
    )
    T <: Union{Ptr, Core.LLVMPtr, CUDA.CuPtr, CUDA.PtrOrCuPtr, CUDA.CuArrayPtr} &&
        return :(throw(ArgumentError("CUDA kernels refuse unowned raw pointer captures")))
    pointers = Union{Ptr, Core.LLVMPtr, CUDA.CuPtr, CUDA.PtrOrCuPtr, CUDA.CuArrayPtr}
    SolpropJulia._capture_parameter_pointer(T, pointers) && return :(
        throw(ArgumentError("CUDA kernels refuse pointer values in captured type metadata"))
    )
    T <: AbstractArray && return :(_cuda_array(value, execution))
    (isbitstype(T) || T <: Union{Function, Tuple, NamedTuple}) || return :(
        throw(ArgumentError("CUDA kernels require scalar or dense device-array captures"))
    )
    checks = [
        :(_cuda_capture(getfield(value, $index), execution))
            for index in 1:fieldcount(T)
    ]
    return quote
        $(checks...)
        nothing
    end
end

KernelAbstractions.@kernel function _cuda_each_kernel(body, rows)
    index = KernelAbstractions.@index(Global, Linear)
    row = mod(index - 1, rows) + 1
    column = div(index - 1, rows) + 1
    @inline body(row, column)
end

KernelAbstractions.@kernel function _cuda_column_kernel(body)
    column = KernelAbstractions.@index(Global, Linear)
    @inline body(column)
end

function SolpropJulia._each!(
        body::F, execution::CUDAExecutionPolicy, rows::Int, columns::Int, heavy,
    ) where {F}
    count = Base.checked_mul(rows, columns)
    _cuda_capture(body, execution)
    _with_cuda_execution(execution) do
        _cuda_each_kernel(execution.backend.kernel)(body, rows; ndrange = count)
        CUDA.synchronize(execution.backend.stream)
    end
    return nothing
end

function SolpropJulia._each_column!(
        body::F, execution::CUDAExecutionPolicy, columns::Int,
    ) where {F}
    _cuda_capture(body, execution)
    _with_cuda_execution(execution) do
        _cuda_column_kernel(execution.backend.kernel)(body; ndrange = columns)
        CUDA.synchronize(execution.backend.stream)
    end
    return nothing
end
