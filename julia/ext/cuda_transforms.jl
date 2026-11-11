# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct CUDAFFTPlan{P <: CUDA.cuFFT.CuFFTPlan}
    plan::P
    execution::CUDAExecutionPolicy
    inplace::Bool
    workspace_bytes::Int
end

function CUDAFFTPlan(
        plan::P, execution::CUDAExecutionPolicy, inplace::Bool,
    ) where {P <: CUDA.cuFFT.CuFFTPlan}
    return _with_cuda_execution(execution) do
        plan.ctx == execution.backend.context ||
            throw(ArgumentError("CUDA FFT plan must retain the selected context"))
        # The pinned wrapper discards cufftMakePlan's worksize and uses automatic
        # allocation. NVIDIA section 3.7 provides the completed plan's actual size.
        bytes = Ref{Csize_t}(0)
        CUDA.cuFFT.cufftGetSize(plan.handle, bytes)
        bytes[] <= typemax(Int) ||
            throw(OverflowError("CUDA FFT workspace exceeds the memory ledger's Int range"))
        CUDAFFTPlan{P}(plan, execution, inplace, Int(bytes[]))
    end
end

function _cuda_fft_alignment(array::CUDAOwnedArray, width::Int)
    # NVIDIA cuFFT sections 3.9.2/3.9.3 require complex-type alignment on both
    # real-transform pointers. Dense foreign headers need not have that alignment.
    iszero(UInt(pointer(array)) % UInt(width)) ||
        throw(ArgumentError("CUDA FFT buffers require complex-element pointer alignment"))
    return nothing
end

function _cuda_fft_buffers(plan::CUDAFFTPlan, output, input)
    execution = plan.execution
    _cuda_array(output, execution)
    _cuda_array(input, execution)
    eltype(input) === CUDA.cuFFT.input_type(plan.plan) &&
        eltype(output) === CUDA.cuFFT.output_type(plan.plan) ||
        throw(ArgumentError("CUDA FFT arguments must retain the prepared precision"))
    width = max(sizeof(eltype(input)), sizeof(eltype(output)))
    _cuda_fft_alignment(input, width)
    _cuda_fft_alignment(output, width)
    CUDA.cuFFT.assert_applicable(plan.plan, input, output)
    if plan.inplace
        output === input ||
            throw(ArgumentError("an in-place CUDA FFT requires the same array owner"))
    else
        _cuda_overlap(output, input) &&
            throw(ArgumentError("CUDA FFT input and output must be disjoint"))
    end
    buffer = plan.plan.buffer
    if buffer !== nothing
        _cuda_array(buffer, execution)
        _cuda_fft_alignment(buffer, width)
        (_cuda_overlap(input, buffer) || _cuda_overlap(output, buffer)) &&
            throw(ArgumentError("CUDA FFT arguments must not alias retained plan scratch"))
    end
    return nothing
end

function LinearAlgebra.mul!(output, plan::CUDAFFTPlan, input)
    _with_cuda_execution(plan.execution) do
        _cuda_fft_buffers(plan, output, input)
        LinearAlgebra.mul!(output, plan.plan, input)
        CUDA.synchronize(plan.execution.backend.stream)
    end
    return output
end

function _cuda_fft_execution(plan::CUDAFFTPlan, execution)
    execution === plan.execution ||
        throw(ArgumentError("CUDA FFT execution must match its prepared owner"))
    return nothing
end

"""
    SolpropJulia.TemporalFFT(execution::CUDAExecutionPolicy, time, spectrum)

Prepare synchronous dense CUDA temporal FFTs in binary32 or binary64.

Buffers are disjoint, owned device-memory CuArray matrices from the execution's
context, with base pointers aligned to their complex element width. Real time
samples have `n_time ÷ 2 + 1` complex frequencies; complex samples have `n_time`.
Both directions preserve their inputs. The backward cuFFT plan is raw;
the existing inverse operation applies `1/n_time` exactly once. Retained cuFFT input
preservation buffers and queried native workspaces are enumerated in the memory
ledger. Host FFTW wisdom is not used. Wrapped/strided device views are refused rather
than staged per call.

Throw `ArgumentError` for unsupported precision, ownership or overlapping storage,
and `DimensionMismatch` for inconsistent shapes.
"""
function SolpropJulia.TemporalFFT(
        execution::CUDAExecutionPolicy, time::CUDAOwnedArray{T, 2},
        spectrum::CUDAOwnedArray{Complex{R}, 2},
    ) where {T, R <: Union{Float32, Float64}}
    real_signal = T === R
    real_signal || T === Complex{R} ||
        throw(ArgumentError("CUDA temporal FFT buffers need matching precision"))
    _cuda_array(time, execution)
    _cuda_array(spectrum, execution)
    n_time, n_columns = size(time)
    n_time > 0 && n_columns > 0 ||
        throw(ArgumentError("CUDA FFT dimensions must be positive"))
    n_frequency = real_signal ? div(n_time, 2) + 1 : n_time
    size(spectrum) == (n_frequency, n_columns) ||
        throw(DimensionMismatch("CUDA temporal FFT spectrum has the wrong shape"))
    return _with_cuda_execution(execution) do
        _cuda_fft_alignment(time, sizeof(Complex{R}))
        _cuda_fft_alignment(spectrum, sizeof(Complex{R}))
        _cuda_overlap(time, spectrum) &&
            throw(ArgumentError("CUDA temporal FFT buffers must be disjoint"))
        forward = real_signal ? CUDA.cuFFT.plan_rfft(time, (1,)) :
            CUDA.cuFFT.plan_fft(time, (1,))
        inverse = real_signal ? CUDA.cuFFT.plan_brfft(spectrum, n_time, (1,)) :
            CUDA.cuFFT.plan_bfft(spectrum, (1,))
        for raw in (forward, inverse)
            if raw.buffer !== nothing
                _cuda_array(raw.buffer, execution)
                _cuda_fft_alignment(raw.buffer, sizeof(Complex{R}))
            end
        end
        CUDA.synchronize(execution.backend.stream)
        SolpropJulia.TemporalFFT(
            CUDAFFTPlan(forward, execution, false),
            CUDAFFTPlan(inverse, execution, false),
            inv(R(n_time)), n_time, n_frequency, n_columns,
        )
    end
end

function SolpropJulia.temporal_inverse!(
        time, plan::SolpropJulia.TemporalFFT{<:CUDAFFTPlan}, spectrum,
        execution::SolpropJulia.KernelExecution,
    )
    _cuda_fft_execution(plan.inverse, execution)
    LinearAlgebra.mul!(time, plan.inverse, spectrum)
    scale = plan.inverse_scale
    SolpropJulia.each!(execution, size(time)...) do row, column
        time[row, column] *= scale
    end
    return time
end

function _cuda_spatial_plans(
        execution::CUDAExecutionPolicy, buffer::CUDAOwnedArray{Complex{T}, N},
        region::NTuple{R, Int}, count::Int,
    ) where {T <: Union{Float32, Float64}, N, R}
    _cuda_fft_alignment(buffer, sizeof(Complex{T}))
    forward = CUDA.cuFFT.plan_fft!(buffer, region)
    inverse = CUDA.cuFFT.plan_bfft!(buffer, region)
    CUDA.synchronize(execution.backend.stream)
    return SolpropJulia.SpatialFFT(
        CUDAFFTPlan(forward, execution, true), CUDAFFTPlan(inverse, execution, true),
        inv(T(count)),
    )
end

"""
    SolpropJulia.SpatialFFT(execution::CUDAExecutionPolicy, buffer, axes)

Prepare synchronous in-place complex CUDA transforms on dense owned storage.

`axes` is a nonempty strictly increasing tuple of dimension indices. Other dimensions
are independent batches. Buffer precision is ComplexF32 or ComplexF64, with a base
pointer aligned to that element width. The forward
and backward plans are unnormalized; the existing inverse call applies the reciprocal
transformed volume once. Preparation preserves every buffer value. No per-call view
staging or FFTW wisdom policy is used.
Native workspace bytes are queried at preparation and counted once per raw plan in
the memory ledger; execution does not query or reconfigure that workspace.

Throw `ArgumentError` for unsupported storage, context, precision, axes or empty data.
"""
function SolpropJulia.SpatialFFT(
        execution::CUDAExecutionPolicy, buffer::CUDAOwnedArray{Complex{T}, N},
        axes::Tuple{Vararg{Integer}},
    ) where {T <: Union{Float32, Float64}, N}
    _cuda_array(buffer, execution)
    isempty(buffer) && throw(ArgumentError("CUDA FFT dimensions must be positive"))
    region = map(Int, axes)
    !isempty(region) && all(axis -> 1 <= axis <= N, region) &&
        all(index -> region[index] < region[index + 1], 1:(length(region) - 1)) ||
        throw(ArgumentError("CUDA FFT axes must increase strictly within array dimensions"))
    count = 1
    for axis in region
        count = Base.checked_mul(count, size(buffer, axis))
    end
    return _with_cuda_execution(execution) do
        _cuda_spatial_plans(execution, buffer, region, count)
    end
end

function SolpropJulia.spatial_inverse!(
        buffer, plan::SolpropJulia.SpatialFFT{<:CUDAFFTPlan},
        execution::SolpropJulia.KernelExecution,
    )
    _cuda_fft_execution(plan.inverse, execution)
    LinearAlgebra.mul!(buffer, plan.inverse, buffer)
    scale = plan.inverse_scale
    SolpropJulia.each!(execution, length(buffer), 1) do index, unused
        buffer[index] *= scale
    end
    return buffer
end

# The native work area belongs to the raw plan, which several wrappers may share.
# Its separately owned numerical buffer still uses the normal array deduplication.
function SolpropJulia._memory_arrays!(ledger, seen, path, plan::CUDAFFTPlan)
    if !haskey(seen, plan.plan)
        seen[plan.plan] = true
        ledger[Symbol(string(path, ".workspace"))] = plan.workspace_bytes
    end
    return SolpropJulia._memory_arrays!(
        ledger, seen, string(path, ".buffer"), plan.plan.buffer,
    )
end
