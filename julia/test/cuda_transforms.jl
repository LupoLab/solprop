# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import CUDA
import LinearAlgebra
import SolpropJulia as SP

@test CUDA.functional()
CUDA.allowscalar(false)

function cuda_reference_dft(input, rows)
    return setprecision(256) do
        count = size(input, 1)
        output = zeros(Complex{BigFloat}, rows, size(input, 2))
        for column in axes(input, 2), frequency in 0:(rows - 1), time in 0:(count - 1)
            angle = -2big(pi) * frequency * time / count
            output[frequency + 1, column] +=
                Complex{BigFloat}(input[time + 1, column]) * cis(angle)
        end
        output
    end
end

function cuda_fft_bound(::Type{T}, input) where {T}
    # Qualified FFT model: each represented butterfly/twiddle product has a
    # relative error bounded by the usual componentwise rounding model, with
    # twiddle components accurate to at most two unit roundoffs. This is a
    # hypothesis checked by these independent small radix-two witnesses, not a
    # proof of arbitrary vendor plans. 64n covers real splitting, complex products
    # and all radix-two stages for the tested n<=32. Include gradual underflow.
    count = 64size(input, 1)
    unit = eps(T) / 2
    @assert count * unit < 1
    gamma = count * unit / (1 - count * unit)
    bound = gamma * sum(abs, Complex{BigFloat}.(input)) +
        count * BigFloat(nextfloat(zero(T)))
    @assert bound > 0
    return bound
end

function cuda_fft_roundtrip_bound(::Type{T}, input) where {T}
    count = 64size(input, 1)
    unit = eps(T) / 2
    gamma = count * unit / (1 - count * unit)
    @assert gamma < 1
    # If one transform has absolute error B = gamma*M + eta_budget, the second
    # transform sees the first error too. Its input norm gives (2+gamma)*B,
    # including the gamma^2 cross term. The 64n additive allowance also covers
    # the possible final dyadic-normalization underflow for these n>=8 fixtures.
    return (2 + gamma) * cuda_fft_bound(T, input)
end

function cuda_temporal_cycle!(time, spectrum, plan, execution)
    SP.temporal_forward!(spectrum, plan, time)
    SP.temporal_inverse!(time, plan, spectrum, execution)
    return nothing
end

function cuda_queried_workspace(plan)
    return CUDA.context!(plan.execution.backend.context) do
        bytes = Ref{Csize_t}(0)
        CUDA.cuFFT.cufftGetSize(plan.plan.handle, bytes)
        Int(bytes[])
    end
end

@testset "CUDA dense temporal plans preserve inputs and normalization" begin
    execution = SP.device_execution(Val(:cuda), 0, 2; check_body = true)
    prior_context, prior_stream = CUDA.context(), CUDA.stream()
    for T in (Float32, Float64), real_signal in (true, false), count in (8, 16, 32)
        samples = reshape(T.(sin.(collect(1:(3count)) ./ 5)), count, 3)
        host = real_signal ? samples : complex.(samples, reverse(samples; dims = 1) ./ 4)
        time = SP.upload(execution, host)
        rows = real_signal ? count ÷ 2 + 1 : count
        spectrum = SP.dzeros(execution, Complex{T}, rows, 3)
        plan = SP.TemporalFFT(execution, time, spectrum)
        @test Array(time) == host && all(iszero, Array(spectrum))
        @test plan.inverse_scale == inv(T(count))
        SP.temporal_forward!(spectrum, plan, time)
        reference = cuda_reference_dft(host, rows)
        bound = cuda_fft_bound(T, host)
        @test maximum(abs, Complex{BigFloat}.(Array(spectrum)) .- reference) <= bound
        @test Array(time) == host
        saved = Array(spectrum)
        SP.temporal_inverse!(time, plan, spectrum, execution)
        @test Array(spectrum) == saved
        @test maximum(abs, Complex{BigFloat}.(Array(time)) .- host) <=
            cuda_fft_roundtrip_bound(T, host)
        cuda_temporal_cycle!(time, spectrum, plan, execution)
        @test (CUDA.@allocated cuda_temporal_cycle!(time, spectrum, plan, execution)) == 0
        ledger = Dict{Symbol, Int}()
        SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "fft", plan)
        native_bytes = 0
        for (name, wrapper) in ((:forward, plan.forward), (:inverse, plan.inverse))
            queried = cuda_queried_workspace(wrapper)
            @test wrapper.workspace_bytes == queried
            @test ledger[Symbol("fft.$name.workspace")] == queried
            native_bytes += queried
        end
        @test sum(values(ledger); init = 0) ==
            native_bytes + (real_signal ? 2sizeof(spectrum) : 0)
        fill!(time, zero(eltype(time)))
        cuda_temporal_cycle!(time, spectrum, plan, execution)
        @test all(iszero, Array(time)) && all(iszero, Array(spectrum))
        @test CUDA.context() == prior_context && CUDA.stream() == prior_stream
    end
end

function cuda_spatial_reference(input, region)
    result = Complex{BigFloat}.(input)
    for axis in region
        order = (axis, (i for i in 1:ndims(input) if i != axis)...)
        permuted = permutedims(result, order)
        shape = size(permuted)
        transformed = reshape(
            cuda_reference_dft(reshape(permuted, shape[1], :), shape[1]), shape,
        )
        result = permutedims(transformed, invperm(collect(order)))
    end
    return result
end

function cuda_spatial_cycle!(buffer, plan, execution)
    SP.spatial_forward!(buffer, plan)
    SP.spatial_inverse!(buffer, plan, execution)
    return nothing
end

function cuda_axis_error_bound(::Type{T}, count, magnitude, error) where {T}
    unit = BigFloat(eps(T)) / 2
    gamma = 64count * unit / (1 - 64count * unit)
    @assert 64count * unit < 1
    # Every line has sumabs <= n*M. Exact propagation of earlier componentwise
    # errors is <= n*E. The new transform's represented input has magnitude
    # <= M+E, giving n*(E+gamma*(M+E)) plus the additive underflow allowance.
    next_error = count * (error + gamma * (magnitude + error)) +
        64count * BigFloat(nextfloat(zero(T)))
    return count * magnitude, next_error
end

function cuda_spatial_error_bounds(::Type{T}, input, region) where {T}
    return setprecision(256) do
        setrounding(BigFloat, RoundUp) do
            magnitude = maximum(abs, Complex{BigFloat}.(input))
            error = BigFloat(0)
            for axis in region
                magnitude, error = cuda_axis_error_bound(
                    T, size(input, axis), magnitude, error,
                )
            end
            forward = error
            for axis in reverse(region)
                magnitude, error = cuda_axis_error_bound(
                    T, size(input, axis), magnitude, error,
                )
            end
            # The tested axis lengths are powers of two. The only possible
            # final scaling error is at most half one output subnormal.
            volume = prod(size(input, axis) for axis in region)
            backward = error / volume + BigFloat(nextfloat(zero(T))) / 2
            @assert forward > 0 && backward > 0
            (forward, backward)
        end
    end
end

@testset "CUDA dense spatial batches and preparation refusal" begin
    execution = SP.device_execution(Val(:cuda), 0, 1)
    for T in (Float32, Float64), (shape, region) in (
                ((8, 3), (1,)), ((3, 1, 4), (3,)), ((3, 1, 4, 4), (3, 4)),
            )
        host = reshape(Complex{T}.(sin.(1:prod(shape)) .+ im .* cos.(1:prod(shape))), shape)
        buffer = SP.upload(execution, host)
        plan = SP.SpatialFFT(execution, buffer, region)
        @test Array(buffer) == host
        SP.spatial_forward!(buffer, plan)
        reference = cuda_spatial_reference(host, region)
        # Compose the per-axis model with outward-rounded magnitude/error
        # envelopes, including the previous stage's error in each new operand.
        bound, backward = cuda_spatial_error_bounds(T, host, region)
        @test maximum(abs, Complex{BigFloat}.(Array(buffer)) .- reference) <= bound
        SP.spatial_inverse!(buffer, plan, execution)
        @test maximum(abs, Complex{BigFloat}.(Array(buffer)) .- host) <= backward
        cuda_spatial_cycle!(buffer, plan, execution)
        @test (CUDA.@allocated cuda_spatial_cycle!(buffer, plan, execution)) == 0
        ledger = Dict{Symbol, Int}()
        SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "fft", plan)
        @test ledger[Symbol("fft.forward.workspace")] ==
            cuda_queried_workspace(plan.forward)
        @test ledger[Symbol("fft.inverse.workspace")] ==
            cuda_queried_workspace(plan.inverse)
        @test sum(values(ledger); init = 0) ==
            plan.forward.workspace_bytes + plan.inverse.workspace_bytes
    end
    time = SP.upload(execution, ones(ComplexF32, 8, 2))
    spectrum = SP.dzeros(execution, ComplexF32, 8, 2)
    @test_throws ArgumentError SP.TemporalFFT(execution, time, time)
    @test_throws ArgumentError SP.TemporalFFT(
        execution, zeros(ComplexF32, 8, 2), zeros(ComplexF32, 8, 2),
    )
    @test_throws ArgumentError SP.SpatialFFT(execution, zeros(ComplexF32, 8, 2), (1,))
    @test_throws DimensionMismatch SP.TemporalFFT(
        execution, time, SP.dzeros(execution, ComplexF32, 7, 2),
    )
    @test_throws ArgumentError SP.TemporalFFT(
        execution, time, SP.dzeros(execution, ComplexF64, 8, 2),
    )
    @test_throws ArgumentError SP.TemporalFFT(execution, view(time, 1:2:8, :), spectrum)
    @test_throws ArgumentError SP.SpatialFFT(execution, time, ())
    @test_throws ArgumentError SP.SpatialFFT(execution, time, (2, 1))
    @test_throws ArgumentError SP.SpatialFFT(execution, time, (1, 1))
    @test_throws ArgumentError SP.SpatialFFT(execution, time, (3,))
    @test_throws ArgumentError SP.SpatialFFT(execution, view(time, 1:2:8, :), (1,))
    plan = SP.TemporalFFT(execution, time, spectrum)
    second = SP.device_execution(Val(:cuda), 0, 1)
    before = Array(time)
    @test_throws ArgumentError SP.temporal_inverse!(time, plan, spectrum, second)
    @test Array(time) == before
    @test_throws ArgumentError SP.temporal_forward!(time, plan, time)
    @test Array(time) == before
    wrong = SP.dzeros(execution, ComplexF64, 8, 2)
    @test_throws ArgumentError SP.temporal_forward!(wrong, plan, time)
    @test all(iszero, Array(wrong))
end

@testset "CUDA native FFT workspace and numerical-buffer ownership" begin
    execution = SP.device_execution(Val(:cuda), 0, 1)
    extension = Base.get_extension(SP, :SolpropJuliaCUDAExt)
    for T in (Float32, Float64), real_signal in (true, false), count in (16, 257)
        time = SP.dzeros(execution, real_signal ? T : Complex{T}, count, 2)
        spectrum = SP.dzeros(execution, Complex{T}, real_signal ? count ÷ 2 + 1 : count, 2)
        plan = SP.TemporalFFT(execution, time, spectrum)
        # A second wrapper can refer to the same raw plan. Reuse its prepared
        # metadata without re-querying or changing the vendor plan configuration.
        shared = extension.CUDAFFTPlan(
            plan.forward.plan, execution, false, plan.forward.workspace_bytes,
        )
        ledger = Dict{Symbol, Int}()
        seen = IdDict{Any, Bool}()
        SP._memory_arrays!(ledger, seen, "first", plan.forward)
        expected = plan.forward.workspace_bytes +
            (plan.forward.plan.buffer === nothing ? 0 : sizeof(plan.forward.plan.buffer))
        @test ledger[Symbol("first.workspace")] == cuda_queried_workspace(plan.forward)
        @test sum(values(ledger); init = 0) == expected
        before = copy(ledger)
        SP._memory_arrays!(ledger, seen, "shared", shared)
        SP._memory_arrays!(ledger, seen, "buffer_alias", plan.forward.plan.buffer)
        @test ledger == before
        SP._memory_arrays!(ledger, seen, "inverse", plan.inverse)
        expected += plan.inverse.workspace_bytes +
            (plan.inverse.plan.buffer === nothing ? 0 : sizeof(plan.inverse.plan.buffer))
        @test ledger[Symbol("inverse.workspace")] == cuda_queried_workspace(plan.inverse)
        @test sum(values(ledger); init = 0) == expected
        @test plan.forward.plan !== plan.inverse.plan
        @test all(iszero, Array(time)) && all(iszero, Array(spectrum))
    end
end

@testset "CUDA FFT independent headers cannot bypass overlap admission" begin
    execution = SP.device_execution(Val(:cuda), 0, 1)
    owner = SP.upload(execution, ComplexF32.(1:40))
    CUDA.context!(execution.backend.context) do
        GC.@preserve owner begin
            input = unsafe_wrap(CUDA.CuArray, pointer(owner), (8, 2); own = false)
            output = unsafe_wrap(CUDA.CuArray, pointer(owner, 2), (8, 2); own = false)
            before = Array(owner)
            @test_throws ArgumentError SP.TemporalFFT(execution, input, output)
            @test Array(owner) == before
            plan = SP.TemporalFFT(execution, input, SP.dzeros(execution, ComplexF32, 8, 2))
            @test_throws ArgumentError SP.temporal_forward!(output, plan, input)
            @test Array(owner) == before
        end
    end
    real_time = SP.upload(execution, ones(Float32, 8, 2))
    spectrum = SP.dzeros(execution, ComplexF32, 5, 2)
    plan = SP.TemporalFFT(execution, real_time, spectrum)
    scratch = plan.forward.plan.buffer
    @test_throws ArgumentError SP.temporal_forward!(scratch, plan, real_time)
    @test Array(real_time) == ones(Float32, 8, 2)
end

@testset "CUDA FFT foreign headers retain complex pointer alignment" begin
    execution = SP.device_execution(Val(:cuda), 0, 1)
    for T in (Float32, Float64)
        owner = SP.upload(execution, T.(1:40))
        time = SP.upload(execution, ones(T, 8, 2))
        spectrum = SP.dzeros(execution, Complex{T}, 5, 2)
        plan = SP.TemporalFFT(execution, time, spectrum)
        SP.copy_device!(plan.inverse.plan.buffer, spectrum, execution)
        complex_time = SP.upload(execution, ones(Complex{T}, 8, 2))
        spatial = SP.SpatialFFT(execution, complex_time, (1,))
        CUDA.context!(execution.backend.context) do
            GC.@preserve owner begin
                # All views stay inside owner. Offsetting one real scalar retains
                # Julia scalar alignment but violates cuFFT complex alignment (#732).
                real_bad = unsafe_wrap(CUDA.CuArray, pointer(owner, 2), (8, 2); own = false)
                address = reinterpret(CUDA.CuPtr{Complex{T}}, pointer(owner, 2))
                complex_bad = unsafe_wrap(CUDA.CuArray, address, (5, 2); own = false)
                spatial_bad = unsafe_wrap(CUDA.CuArray, address, (8, 2); own = false)
                before = Array(owner)
                @test_throws ArgumentError SP.TemporalFFT(execution, real_bad, spectrum)
                @test_throws ArgumentError SP.TemporalFFT(execution, time, complex_bad)
                @test_throws ArgumentError SP.SpatialFFT(execution, spatial_bad, (1,))
                @test_throws ArgumentError SP.temporal_forward!(spectrum, plan, real_bad)
                @test_throws ArgumentError SP.temporal_forward!(complex_bad, plan, time)
                @test_throws ArgumentError SP.temporal_inverse!(
                    real_bad, plan, spectrum, execution,
                )
                @test_throws ArgumentError SP.temporal_inverse!(
                    time, plan, complex_bad, execution,
                )
                @test_throws ArgumentError SP.spatial_forward!(spatial_bad, spatial)
                saved_scratch = plan.inverse.plan.buffer
                plan.inverse.plan.buffer = complex_bad
                try
                    @test_throws ArgumentError SP.temporal_inverse!(
                        time, plan, spectrum, execution,
                    )
                finally
                    plan.inverse.plan.buffer = saved_scratch
                end
                @test Array(owner) == before
                @test Array(time) == ones(T, 8, 2)
                @test all(iszero, Array(spectrum))
                @test all(iszero, Array(plan.inverse.plan.buffer))

                # An aligned foreign header remains supported and preserves its
                # backing owner in a successful out-of-place forward transform.
                real_good = unsafe_wrap(
                    CUDA.CuArray, pointer(owner, 3), (8, 2); own = false,
                )
                aligned = SP.TemporalFFT(execution, real_good, spectrum)
                SP.temporal_forward!(spectrum, aligned, real_good)
                reference = cuda_reference_dft(Array(real_good), 5)
                @test maximum(abs, Complex{BigFloat}.(Array(spectrum)) .- reference) <=
                    cuda_fft_bound(T, Array(real_good))
                @test Array(owner) == before
            end
        end
    end
end
