# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct MetalFFTPlan{P <: Metal.MtlFFTPlan, E <: MetalExecutionPolicy}
    plan::P
    execution::E
    inplace::Bool
end

function _metal_fft_buffers(
        plan::MetalFFTPlan{<:Metal.MtlFFTPlan{T, S}}, output, input,
    ) where {T, S}
    _metal_array(output, plan.execution)
    _metal_array(input, plan.execution)
    eltype(output) === T && eltype(input) === S ||
        throw(ArgumentError("Metal FFT arguments must retain their prepared precision"))
    Metal.assert_applicable(plan.plan, input, output)
    if plan.inplace
        output === input ||
            throw(ArgumentError("an in-place Metal FFT requires the same array owner"))
    else
        _metal_overlap(output, input) &&
            throw(ArgumentError("Metal FFT input and output must be disjoint"))
    end
    return nothing
end

function LinearAlgebra.mul!(
        output::MetalOwnedArray{T, N}, plan::MetalFFTPlan{<:Metal.MtlFFTPlan{T, S}},
        input::MetalOwnedArray{S, N},
    ) where {T <: Metal.FFTNumber, S <: Metal.FFTNumber, N}
    _metal_fft_buffers(plan, output, input)
    return _with_metal_execution(plan.execution) do
        LinearAlgebra.mul!(output, plan.plan, input)
        output
    end
end

function LinearAlgebra.mul!(output, plan::MetalFFTPlan, input)
    _metal_fft_buffers(plan, output, input)
    throw(ArgumentError("Metal FFTs require matching dense private-storage arrays"))
end

function _metal_fft_execution(plan::MetalFFTPlan, execution)
    execution === plan.execution ||
        throw(ArgumentError("Metal FFT execution must match its prepared owner"))
    return nothing
end

function _metal_temporal_raw(
        time::MetalOwnedArray{Float32, 2}, spectrum::MetalOwnedArray{ComplexF32, 2},
        count::Int,
    )
    return Metal.plan_rfft(time, (1,)), Metal.plan_brfft(spectrum, count, (1,))
end

function _metal_temporal_raw(
        time::MetalOwnedArray{ComplexF32, 2}, spectrum::MetalOwnedArray{ComplexF32, 2},
        count::Int,
    )
    return Metal.plan_fft(time, (1,)), Metal.plan_bfft(spectrum, (1,))
end

function _metal_warm_plans!(
        execution::MetalExecutionPolicy, forward::Metal.MtlFFTPlan,
        inverse::Metal.MtlFFTPlan, input::MetalOwnedArray, output::MetalOwnedArray,
    )
    # The scoped wrapper preserves this closure, and therefore both temporary
    # owners, through synchronization even if a preceding encode throws.
    return _with_metal_execution(execution) do
        fill!(input, zero(eltype(input)))
        LinearAlgebra.mul!(output, forward, input)
        LinearAlgebra.mul!(input, inverse, output)
        nothing
    end
end

"""
    SolpropJulia.TemporalFFT(execution::MetalExecutionPolicy, time, spectrum)

Prepare synchronous dense binary32 temporal FFTs on the execution's public task queue.

`time` is a private-storage Float32 or ComplexF32 matrix. The disjoint ComplexF32
`spectrum` has `n_time÷2+1` rows for real samples or `n_time` for complex samples,
with the same positive column count. Both application directions preserve their input.
Real lengths must be even: the pinned graph builder does not set the descriptor's
odd Hermitian output flag. This restriction is checked before planning or writes.
The backward MPSGraph plan is raw; inverse application multiplies by `1/n_time` once.
Preparation preserves both caller buffers and warms the exact graph shapes on one
temporary zeroed pair. Those temporary arrays are not retained by these plan wrappers.

MPSGraph's device/shape cache and opaque framework workspace are process overhead,
not a zero-byte workspace claim. There is no FFTW wisdom or private queue replacement.
This foundation does not qualify DF32 or complete Metal propagation sessions.

Throw `ArgumentError` for storage/task/precision/overlap violations, and
`DimensionMismatch` for inconsistent shapes.
"""
function SolpropJulia.TemporalFFT(
        execution::MetalExecutionPolicy, time::MetalOwnedArray{T, 2},
        spectrum::MetalOwnedArray{ComplexF32, 2},
    ) where {T <: Union{Float32, ComplexF32}}
    _metal_array(time, execution)
    _metal_array(spectrum, execution)
    n_time, n_columns = size(time)
    n_time > 0 && n_columns > 0 ||
        throw(ArgumentError("Metal FFT dimensions must be positive"))
    T === Float32 && isodd(n_time) &&
        throw(ArgumentError("Metal real temporal FFTs require an even sample count"))
    n_frequency = T === Float32 ? div(n_time, 2) + 1 : n_time
    size(spectrum) == (n_frequency, n_columns) ||
        throw(DimensionMismatch("Metal temporal FFT spectrum has the wrong shape"))
    _metal_overlap(time, spectrum) &&
        throw(ArgumentError("Metal temporal FFT buffers must be disjoint"))
    return _with_metal_execution(execution) do
        forward, inverse = _metal_temporal_raw(time, spectrum, n_time)
        warm_time = Metal.MtlArray{T, 2, Metal.PrivateStorage}(undef, size(time))
        warm_spectrum = Metal.MtlArray{ComplexF32, 2, Metal.PrivateStorage}(
            undef, size(spectrum),
        )
        _metal_warm_plans!(execution, forward, inverse, warm_time, warm_spectrum)
        SolpropJulia.TemporalFFT(
            MetalFFTPlan(forward, execution, false),
            MetalFFTPlan(inverse, execution, false),
            inv(Float32(n_time)), n_time, n_frequency, n_columns,
        )
    end
end

function SolpropJulia.temporal_inverse!(
        time, plan::SolpropJulia.TemporalFFT{<:MetalFFTPlan}, spectrum,
        execution::SolpropJulia.KernelExecution,
    )
    _metal_fft_execution(plan.inverse, execution)
    LinearAlgebra.mul!(time, plan.inverse, spectrum)
    scale = plan.inverse_scale
    SolpropJulia.each!(execution, size(time)...) do row, column
        time[row, column] *= scale
    end
    return time
end

function _metal_spatial_plans(
        execution::MetalExecutionPolicy, buffer::MetalOwnedArray{ComplexF32, N},
        region::NTuple{R, Int}, count::Int,
    ) where {N, R}
    forward = Metal.plan_fft!(buffer, region)
    inverse = Metal.plan_bfft!(buffer, region)
    warm = Metal.MtlArray{ComplexF32, N, Metal.PrivateStorage}(undef, size(buffer))
    _metal_warm_plans!(execution, forward, inverse, warm, warm)
    return SolpropJulia.SpatialFFT(
        MetalFFTPlan(forward, execution, true), MetalFFTPlan(inverse, execution, true),
        inv(Float32(count)),
    )
end

"""
    SolpropJulia.SpatialFFT(execution::MetalExecutionPolicy, buffer, axes)

Prepare synchronous in-place ComplexF32 transforms on dense private Metal storage.

`axes` is a nonempty strictly increasing tuple within dimensions 1 through 4 of a
rank-1 through rank-16 buffer. Metal's reversed tensor layout maps these to the four
supported trailing MPSGraph dimensions. Other axes are independent batches.
Both raw plans are warmed on one
temporary zeroed array, preserving the supplied buffer. Inverse application applies
the reciprocal transformed volume once. Every call retains task/queue, precision,
shape and exact in-place owner identity. Opaque MPSGraph cache/workspace remains
separately measured process overhead rather than a retained numerical array.

Throw `ArgumentError` for invalid task, storage, precision, dimensions or axes.
"""
function SolpropJulia.SpatialFFT(
        execution::MetalExecutionPolicy, buffer::MetalOwnedArray{ComplexF32, N},
        axes::Tuple{Vararg{Integer}},
    ) where {N}
    _metal_array(buffer, execution)
    1 <= N <= 16 && !isempty(buffer) || throw(
        ArgumentError(
            "Metal FFT buffers require positive dimensions and rank ≤16",
        )
    )
    region = map(Int, axes)
    !isempty(region) && all(axis -> 1 <= axis <= min(N, 4), region) &&
        all(index -> region[index] < region[index + 1], 1:(length(region) - 1)) ||
        throw(ArgumentError("Metal FFT axes must increase strictly within dimensions 1:4"))
    count = 1
    for axis in region
        count = Base.checked_mul(count, size(buffer, axis))
    end
    return _with_metal_execution(execution) do
        _metal_spatial_plans(execution, buffer, region, count)
    end
end

function SolpropJulia.spatial_inverse!(
        buffer, plan::SolpropJulia.SpatialFFT{<:MetalFFTPlan},
        execution::SolpropJulia.KernelExecution,
    )
    _metal_fft_execution(plan.inverse, execution)
    LinearAlgebra.mul!(buffer, plan.inverse, buffer)
    scale = plan.inverse_scale
    SolpropJulia.each!(execution, length(buffer), 1) do index, unused
        buffer[index] *= scale
    end
    return buffer
end

# The pinned raw plan retains dimensions/axes and an uninitialized inverse-plan cache,
# but no numerical array. Never recursively traverse its Objective-C graph/runtime.
# The global MPSGraph cache and opaque native workspace are documented separately.
function SolpropJulia._memory_arrays!(ledger, seen, path, plan::MetalFFTPlan)
    return nothing
end
