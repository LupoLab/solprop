# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public KernelExecution, each!, each_column!, oncpu, upload, dzeros
public device_execution, synchronize_execution, copy_host!, copy_device!

"""
    KernelExecution(backend, tasks::Integer, check_body::Bool = false; planning=nothing)

Prepare synchronous numerical execution with a positive CPU worker budget.

# Arguments
- `backend`: The KernelAbstractions backend owning the prepared arrays.
- `tasks`: The maximum number of participating CPU threads, including the caller.
  The available default Julia thread pool supplies an additional upper bound.
- `check_body`: Whether to reject known device-incompatible captured types before launch.
- `planning`: Optional preparation-owned `FFTPlanning` context for CPU transforms.
  `nothing` retains the standalone estimate-only planner. Numerical loop bodies never
  receive this context; it contains no execution buffers or filesystem policy.

# Throws
- `ArgumentError`: If `tasks` is not positive.
"""
struct KernelExecution{B}
    backend::B
    tasks::Int
    check_body::Bool
    planning::Union{Nothing, FFTPlanning}

    function KernelExecution(
            backend::B, tasks::Integer, check_body::Bool = false;
            planning::Union{Nothing, FFTPlanning} = nothing,
        ) where {B}
        tasks > 0 || throw(ArgumentError("the kernel task budget must be positive"))
        return new{B}(backend, Int(tasks), check_body, planning)
    end
end

"""
    oncpu(execution::KernelExecution)

Return whether the execution backend stores and computes arrays on the CPU.
"""
oncpu(execution::KernelExecution) = execution.backend isa KernelAbstractions.CPU

"""
    device_execution(Val(device), device_index::Int, tasks::Int; check_body=false)

Prepare an extension-owned device context and stream with a positive host task budget.

`device` identifies the vendor, and `device_index` is a zero-based device ordinal.
The returned `KernelExecution` retains its context and stream independently of the
caller's active device. Each synchronous operation restores the caller's active state.
This constructor requires a loaded, qualified vendor extension; it does not admit
propagation capabilities. Throw `ArgumentError` for an unavailable extension, invalid
ordinal or task budget. No device allocation occurs in the unavailable fallback.
"""
function device_execution(
        ::Val{device}, device_index::Int, tasks::Int; check_body::Bool = false,
    ) where {device}
    throw(ArgumentError("no qualified execution extension is loaded for $device"))
end

"""
    synchronize_execution(execution::KernelExecution)

Finish pending work in the prepared execution context before returning `nothing`.

CPU work is already synchronous. A device extension synchronizes only its owned
context/stream and restores the caller's active state. Throw `ArgumentError` when
the backend has no qualified synchronization method.
"""
function synchronize_execution(execution::KernelExecution)
    oncpu(execution) ||
        throw(ArgumentError("this execution backend has no qualified synchronization"))
    return nothing
end

"""
    copy_host!(destination, source, execution::KernelExecution)

Copy owned device data into a matching dense host array and return the destination.

Shape and element type must agree exactly; no unit or precision conversion occurs.
The device extension validates source context and destination storage before writes,
retains both owners until synchronous completion, and restores the caller's context.
Throw `ArgumentError` for unsupported storage/context/precision or an unavailable
extension, and `DimensionMismatch` for different shapes.
"""
function copy_host!(destination, source, execution::KernelExecution)
    throw(ArgumentError("this execution backend has no qualified device-to-host copy"))
end

"""
    copy_device!(destination, source, execution::KernelExecution)

Copy between matching dense arrays owned by the same prepared device context.

Shape and element type must agree exactly. Identical arrays are an exact no-op;
other overlap refuses before writes. The extension retains both owners until
synchronous completion, restores the caller's context and returns the destination.
Throw `ArgumentError` for unsupported storage/context/precision, overlap or an
unavailable extension, and `DimensionMismatch` for different shapes.
"""
function copy_device!(destination, source, execution::KernelExecution)
    throw(ArgumentError("this execution backend has no qualified device copy"))
end

"""
    upload(execution::KernelExecution, input::AbstractArray)

Copy an array into independently owned storage on the execution backend.

Shape, element type and values are preserved, without physical unit conversion.
The input must have fixed-size elements; views are copied just like dense arrays.
"""
function upload(execution::KernelExecution, input::AbstractArray{T}) where {T}
    isbitstype(T) || throw(ArgumentError("kernel arrays require fixed-size elements"))
    output = KernelAbstractions.allocate(execution.backend, T, size(input))
    copyto!(output, input)
    KernelAbstractions.synchronize(execution.backend)
    return output
end

"""
    dzeros(execution::KernelExecution, element_type, dimensions...)

Allocate an owned zero-filled array on the execution backend.

Dimensions are nonnegative integers and the element type is fixed-size numerical data.
"""
function dzeros(execution::KernelExecution, ::Type{T}, dimensions::Integer...) where {T}
    isbitstype(T) || throw(ArgumentError("kernel arrays require fixed-size elements"))
    all(>=(0), dimensions) || throw(ArgumentError("array dimensions must be nonnegative"))
    # Allocate through the backend, then fill through its array interface. The pinned
    # CPU backend's zeros helper may launch POCL for larger arrays.
    output = KernelAbstractions.allocate(execution.backend, T, map(Int, dimensions))
    fill!(output, zero(T))
    KernelAbstractions.synchronize(execution.backend)
    return output
end

@inline function _check_kernel_body(::Type{F}) where {F}
    for captured in fieldtypes(F)
        if captured <: Type || captured === Core.Box || captured isa Union ||
                !isconcretetype(captured)
            throw(ArgumentError("kernel body $F captures unsupported type $captured"))
        end
        if captured <: AbstractArray
            if !isbitstype(eltype(captured))
                throw(ArgumentError("kernel body $F captures variable-size array elements"))
            end
        elseif !isbitstype(captured)
            if captured <: Function || captured <: Tuple || captured <: NamedTuple
                _check_kernel_body(captured)
            else
                throw(ArgumentError("kernel body $F captures unsupported owner $captured"))
            end
        end
    end
    return nothing
end

function _has_packed_capture(type, visited)
    (type <: BitArray || type <: StrideArraysCore.BitPtrArray) && return true
    isconcretetype(type) || return false
    type in visited && return false
    push!(visited, type)
    return any(field -> _has_packed_capture(field, visited), fieldtypes(type))
end

@generated function _check_cpu_storage(::Type{F}) where {F}
    if _has_packed_capture(F, Set{Any}())
        return :(
            throw(
                ArgumentError(
                    "packed BitArray captures unsupported; use upload for Array{Bool}",
                )
            )
        )
    end
    return nothing
end

@inline function _partition_bounds(length::Int, part::Int, parts::Int)
    width, extra = divrem(length, parts)
    first_index = (part - 1) * width + min(part - 1, extra) + 1
    return first_index, first_index + width + (part <= extra) - 1
end

struct _CPUCapture end

Adapt.adapt_storage(::_CPUCapture, array::AbstractArray) = StrideArraysCore.PtrArray(array)

# Reflection storage exists only while specializing the wrapper, never in a kernel.
function _capture_roots(type, expression)
    type <: AbstractArray && return Any[expression]
    isbitstype(type) && return Any[]
    type <: Union{Function, Tuple, NamedTuple} ||
        throw(ArgumentError("CPU kernel captures an unsupported managed owner $type"))
    roots = Any[]
    for index in 1:fieldcount(type)
        append!(
            roots,
            _capture_roots(fieldtype(type, index), :(getfield($expression, $index))),
        )
    end
    return roots
end

@generated function _with_cpu_captures(launch::L, body::F, arguments...) where {L, F}
    roots = _capture_roots(F, :body)
    owners = [gensym(:owner) for _ in roots]
    assignments = [:($(owners[index]) = $(roots[index])) for index in eachindex(roots)]
    work = quote
        pointer_body = Adapt.adapt(_CPUCapture(), body)
        isbitstype(typeof(pointer_body)) ||
            throw(ArgumentError("adapted CPU kernel still contains managed captures"))
        launch(pointer_body, arguments...)
    end
    # Root each array (including a view and its parent), not the enclosing closure.
    # Polyester's own PtrArray adaptation cannot see arrays hidden in a closure;
    # rooting that closure instead allocates its wrapper on the warmed parallel path.
    # This lexical region includes adaptation and the complete synchronous launch.
    return quote
        $(Expr(:meta, :inline))
        $(assignments...)
        GC.@preserve $(owners...) begin
            $work
        end
        return nothing
    end
end

"""
    each!(body, execution::KernelExecution, n_rows, n_columns; heavy = false)

Apply an independent scalar body to every `(row, column)` pair, then synchronize.

Rows are contiguous in the package's column-major storage. CPU batches own disjoint
row ranges or complete columns and never exceed the execution's worker budget.
An empty dimension performs no work. Bodies must not allocate, throw, or communicate
between iterations. `heavy` is a scheduling hint reserved for measured tuning.
CPU captures must not contain packed `BitArray` storage, including wrapped views:
prepare ordinary `Array{Bool}` storage with `upload` before using a Boolean mask.

# Throws
- `ArgumentError`: For a negative dimension or a rejected captured type.
- `OverflowError`: If the number of elements does not fit in `Int`.
"""
@inline function each!(
        body::F, execution::KernelExecution, n_rows::Integer, n_columns::Integer;
        heavy::Bool = false,
    ) where {F}
    rows, columns = Int(n_rows), Int(n_columns)
    if rows < 0 || columns < 0
        throw(ArgumentError("kernel dimensions must be nonnegative"))
    end
    Base.checked_mul(rows, columns)
    execution.check_body && _check_kernel_body(F)
    # Packed writes share bytes across logical indices, and PtrArray cannot preserve
    # multidimensional bit strides (#624). Refuse every CPU path, even a serial one.
    oncpu(execution) && _check_cpu_storage(F)
    rows == 0 || columns == 0 || _each!(body, execution, rows, columns, heavy)
    return nothing
end

@inline function _each!(
        body::F, execution::KernelExecution{<:KernelAbstractions.CPU},
        n_rows::Int, n_columns::Int, heavy::Bool,
    ) where {F}
    n_tasks = min(execution.tasks, Threads.nthreads(:default))
    if n_tasks == 1
        for column_index in 1:n_columns
            @simd for row_index in 1:n_rows
                @inline body(row_index, column_index)
            end
        end
    else
        _with_cpu_captures(_parallel_each!, body, n_rows, n_columns, n_tasks)
    end
    return nothing
end

@inline function _parallel_each!(
        body::F, n_rows::Int, n_columns::Int, n_tasks::Int,
    ) where {F}
    if n_columns >= n_tasks
        Polyester.@batch per = thread for task_index in 1:n_tasks
            first_column, last_column = _partition_bounds(n_columns, task_index, n_tasks)
            for column_index in first_column:last_column
                @simd for row_index in 1:n_rows
                    @inline body(row_index, column_index)
                end
            end
        end
    else
        n_parts = min(n_rows, div(n_tasks, n_columns))
        n_tiles = n_parts * n_columns
        Polyester.@batch per = thread for tile_index in 1:n_tiles
            column_index = div(tile_index - 1, n_parts) + 1
            part_index = mod(tile_index - 1, n_parts) + 1
            first_row, last_row = _partition_bounds(n_rows, part_index, n_parts)
            @simd for row_index in first_row:last_row
                @inline body(row_index, column_index)
            end
        end
    end
    return nothing
end

function _each!(body, execution::KernelExecution, n_rows, n_columns, heavy)
    throw(ArgumentError("this execution backend has no qualified scalar launcher"))
end

"""
    each_column!(body, execution::KernelExecution, n_columns)

Run one sequential body per column, synchronizing before returning.

Columns are independent, but a body's operations within one column retain their order.
The CPU worker budget also applies when there are more columns than available workers.
Packed Boolean captures have the same unconditional refusal as `each!`.
"""
@inline function each_column!(
        body::F, execution::KernelExecution, n_columns::Integer,
    ) where {F}
    n_columns >= 0 || throw(ArgumentError("the column count must be nonnegative"))
    execution.check_body && _check_kernel_body(F)
    oncpu(execution) && _check_cpu_storage(F)
    n_columns == 0 || _each_column!(body, execution, Int(n_columns))
    return nothing
end

@inline function _each_column!(
        body::F, execution::KernelExecution{<:KernelAbstractions.CPU}, n_columns::Int,
    ) where {F}
    n_tasks = min(execution.tasks, Threads.nthreads(:default), n_columns)
    if n_tasks == 1
        for column_index in 1:n_columns
            @inline body(column_index)
        end
    else
        _with_cpu_captures(_parallel_columns!, body, n_columns, n_tasks)
    end
    return nothing
end

@inline function _parallel_columns!(body::F, n_columns::Int, n_tasks::Int) where {F}
    Polyester.@batch per = thread for task_index in 1:n_tasks
        first_column, last_column = _partition_bounds(n_columns, task_index, n_tasks)
        for column_index in first_column:last_column
            @inline body(column_index)
        end
    end
    return nothing
end

function _each_column!(body, execution::KernelExecution, n_columns)
    throw(ArgumentError("this execution backend has no qualified column launcher"))
end
