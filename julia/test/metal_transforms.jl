# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import LinearAlgebra
import Metal
import SolpropJulia as SP

@test Metal.functional()
Metal.GPUArrays.allowscalar(false)

function metal_reference_dft(input, rows)
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

function metal_axis_error_bound(count, magnitude, error)
    unit = BigFloat(eps(Float32)) / 2
    gamma = 64count * unit / (1 - 64count * unit)
    @assert 64count * unit < 1
    # Qualified finite FFT model: componentwise rounded butterfly/product operations
    # and twiddle components accurate to two unit roundoffs. 64n covers complex
    # products, real splitting and the radix-two stages for these n<=32 fixtures.
    # This is an explicit hypothesis tested against a 256-bit DFT, not a vendor
    # theorem. Metal permits FTZ: use floatmin, not the gradual-underflow eta.
    # Exact propagated input error is n*E; new input magnitude is at most M+E.
    next_error = count * (error + gamma * (magnitude + error)) +
        64count * BigFloat(floatmin(Float32))
    return count * magnitude, next_error
end

function metal_fft_error_bounds(input, region)
    return setprecision(256) do
        setrounding(BigFloat, RoundUp) do
            magnitude = maximum(abs, Complex{BigFloat}.(input))
            error = BigFloat(0)
            for axis in region
                magnitude, error = metal_axis_error_bound(
                    size(input, axis), magnitude, error,
                )
            end
            forward = error
            for axis in reverse(region)
                magnitude, error = metal_axis_error_bound(
                    size(input, axis), magnitude, error,
                )
            end
            volume = prod(size(input, axis) for axis in region)
            # Every tested volume is a power of two: normal scaling is exact;
            # the possible final flushed result contributes less than floatmin.
            backward = error / volume + BigFloat(floatmin(Float32))
            @assert forward > 0 && backward > 0
            (forward, backward)
        end
    end
end

function metal_temporal_cycle!(time, spectrum, plan, execution)
    SP.temporal_forward!(spectrum, plan, time)
    SP.temporal_inverse!(time, plan, spectrum, execution)
    return nothing
end

function metal_spatial_reference(input, region)
    result = Complex{BigFloat}.(input)
    for axis in region
        order = (axis, (i for i in 1:ndims(input) if i != axis)...)
        permuted = permutedims(result, order)
        shape = size(permuted)
        transformed = reshape(
            metal_reference_dft(reshape(permuted, shape[1], :), shape[1]), shape,
        )
        result = permutedims(transformed, invperm(collect(order)))
    end
    return result
end

function metal_spatial_cycle!(buffer, plan, execution)
    SP.spatial_forward!(buffer, plan)
    SP.spatial_inverse!(buffer, plan, execution)
    return nothing
end

@testset "Metal dense temporal graphs preserve inputs and normalization" begin
    execution = SP.device_execution(Val(:metal), 0, 2; check_body = true)
    prior_device = Metal.device()
    prior_queue = Metal.global_queue(prior_device)
    for real_signal in (true, false), count in (8, 16, 32)
        samples = reshape(Float32.(sin.(collect(1:(3count)) ./ 5)), count, 3)
        host = real_signal ? samples : complex.(samples, reverse(samples; dims = 1) ./ 4)
        time = SP.upload(execution, host)
        rows = real_signal ? count ÷ 2 + 1 : count
        spectrum = SP.dzeros(execution, ComplexF32, rows, 3)
        plan = SP.TemporalFFT(execution, time, spectrum)
        @test Array(time) == host && all(iszero, Array(spectrum))
        @test plan.inverse_scale == inv(Float32(count))
        SP.temporal_forward!(spectrum, plan, time)
        reference = metal_reference_dft(host, rows)
        bound, backward = metal_fft_error_bounds(host, (1,))
        @test maximum(abs, Complex{BigFloat}.(Array(spectrum)) .- reference) <= bound
        @test Array(time) == host
        saved = Array(spectrum)
        SP.temporal_inverse!(time, plan, spectrum, execution)
        @test Array(spectrum) == saved
        @test maximum(abs, Complex{BigFloat}.(Array(time)) .- host) <= backward
        metal_temporal_cycle!(time, spectrum, plan, execution)
        # This counts Metal.jl numerical allocations only; MPSGraph's opaque native
        # workspace/cache and transient host dictionaries are separately reported.
        @test (Metal.@allocated metal_temporal_cycle!(time, spectrum, plan, execution)) == 0
        ledger = Dict{Symbol, Int}()
        SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "fft", plan)
        @test isempty(ledger)
        fill!(time, zero(eltype(time)))
        metal_temporal_cycle!(time, spectrum, plan, execution)
        @test all(iszero, Array(time)) && all(iszero, Array(spectrum))
        @test Metal.device() == prior_device
        @test Metal.global_queue(prior_device) === prior_queue
    end
end

@testset "Metal dense spatial graph batches and preparation refusals" begin
    execution = SP.device_execution(Val(:metal), 0, 1)
    for (shape, region) in (
            ((8, 3), (1,)), ((3, 1, 4), (3,)), ((3, 1, 4, 4), (3, 4)),
        )
        host = reshape(ComplexF32.(sin.(1:prod(shape)) .+ im .* cos.(1:prod(shape))), shape)
        buffer = SP.upload(execution, host)
        plan = SP.SpatialFFT(execution, buffer, region)
        @test Array(buffer) == host
        SP.spatial_forward!(buffer, plan)
        reference = metal_spatial_reference(host, region)
        bound, backward = metal_fft_error_bounds(host, region)
        @test maximum(abs, Complex{BigFloat}.(Array(buffer)) .- reference) <= bound
        SP.spatial_inverse!(buffer, plan, execution)
        @test maximum(abs, Complex{BigFloat}.(Array(buffer)) .- host) <= backward
        metal_spatial_cycle!(buffer, plan, execution)
        @test (Metal.@allocated metal_spatial_cycle!(buffer, plan, execution)) == 0
        ledger = Dict{Symbol, Int}()
        SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "fft", plan)
        @test isempty(ledger)
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
        execution, time, SP.dzeros(execution, ComplexF16, 8, 2),
    )
    @test_throws ArgumentError SP.TemporalFFT(execution, view(time, 1:2:8, :), spectrum)
    for region in ((), (2, 1), (1, 1), (3,))
        @test_throws ArgumentError SP.SpatialFFT(execution, time, region)
    end
    @test_throws ArgumentError SP.SpatialFFT(execution, view(time, 1:2:8, :), (1,))
    @test_throws ArgumentError SP.TemporalFFT(
        execution,
        SP.dzeros(execution, Float32, 0, 2), SP.dzeros(execution, ComplexF32, 1, 2)
    )
    odd = SP.upload(execution, reshape(Float32.(1:18), 9, 2))
    odd_spectrum = SP.upload(execution, fill(ComplexF32(3, 4), 5, 2))
    @test_throws ArgumentError SP.TemporalFFT(execution, odd, odd_spectrum)
    @test Array(odd) == reshape(Float32.(1:18), 9, 2)
    @test Array(odd_spectrum) == fill(ComplexF32(3, 4), 5, 2)
    higher = SP.upload(execution, reshape(ComplexF32.(1:8), 1, 1, 1, 1, 8))
    @test_throws ArgumentError SP.SpatialFFT(execution, higher, (5,))
    @test Array(higher) == reshape(ComplexF32.(1:8), 1, 1, 1, 1, 8)
    plan = SP.TemporalFFT(execution, time, spectrum)
    # Identical immutable policies on the same task/device retain the same public
    # queue and compare identically. Change the explicit policy budget here.
    @test SP.device_execution(Val(:metal), 0, 1) === execution
    second = SP.device_execution(Val(:metal), 0, 2)
    before = Array(time)
    @test_throws ArgumentError SP.temporal_inverse!(time, plan, spectrum, second)
    @test Array(time) == before
    @test_throws ArgumentError SP.temporal_forward!(time, plan, time)
    @test Array(time) == before
    wrong = SP.dzeros(execution, ComplexF16, 8, 2)
    @test_throws ArgumentError SP.temporal_forward!(wrong, plan, time)
    @test all(iszero, Array(wrong))
end

@testset "Metal FFT foreign headers and task changes preserve owners" begin
    execution = SP.device_execution(Val(:metal), 0, 1)
    owner = SP.upload(execution, ComplexF32.(1:40))
    GC.@preserve owner begin
        input = unsafe_wrap(Metal.MtlArray, pointer(owner), (8, 2))
        output = unsafe_wrap(Metal.MtlArray, pointer(owner, 2), (8, 2))
        before = Array(owner)
        @test_throws ArgumentError SP.TemporalFFT(execution, input, output)
        @test Array(owner) == before
        spectrum = SP.dzeros(execution, ComplexF32, 8, 2)
        plan = SP.TemporalFFT(execution, input, spectrum)
        @test_throws ArgumentError SP.temporal_forward!(output, plan, input)
        @test Array(owner) == before
        # A foreign header with a byte-offset inside an element is dense in shape,
        # but cannot satisfy MPSNDArray's element-offset representation.
        misaligned = unsafe_wrap(Metal.MtlArray, pointer(owner) + 1, (8, 2))
        @test_throws ArgumentError SP.TemporalFFT(execution, misaligned, spectrum)
        @test_throws ArgumentError SP.temporal_forward!(spectrum, plan, misaligned)
        @test Array(owner) == before && all(iszero, Array(spectrum))
        spatial = SP.SpatialFFT(execution, input, (1,))
        alias = unsafe_wrap(Metal.MtlArray, pointer(owner), (8, 2))
        @test_throws ArgumentError LinearAlgebra.mul!(alias, spatial.forward, input)
        @test Array(owner) == before
        for body in (
                () -> SP.TemporalFFT(execution, input, spectrum),
                () -> SP.temporal_forward!(spectrum, plan, input),
                () -> SP.temporal_inverse!(input, plan, spectrum, execution),
                () -> SP.SpatialFFT(execution, input, (1,)),
                () -> SP.spatial_forward!(input, spatial),
                () -> SP.spatial_inverse!(input, spatial, execution),
            )
            result = fetch(
                @async try
                    body()
                catch error
                    error
                end
            )
            @test result isa ArgumentError
            @test Array(owner) == before && all(iszero, Array(spectrum))
        end
    end
end

@testset "Metal failed warm work keeps owners and restores queue state" begin
    extension = Base.get_extension(SP, :SolpropJuliaMetalExt)
    execution = SP.device_execution(Val(:metal), 0, 1)
    previous = Metal.device()
    queue = Metal.global_queue(previous)
    for real_signal in (true, false)
        T = real_signal ? Float32 : ComplexF32
        input = SP.upload(execution, fill(T(3), 8, 2))
        output = SP.upload(execution, fill(ComplexF32(5, 7), real_signal ? 5 : 8, 2))
        forward, inverse = extension._metal_temporal_raw(input, output, 8)
        # The first raw transform enqueues successfully. The incompatible inverse
        # then refuses before encoding; cleanup must finish the first transform
        # while the warm closure still roots both arrays. No invalid native graph
        # is executed, and retry exercises the same retained owners/queue.
        bad_input = SP.dzeros(execution, T, 10, 2)
        bad_output = SP.dzeros(execution, ComplexF32, real_signal ? 6 : 10, 2)
        unused, bad_inverse = extension._metal_temporal_raw(bad_input, bad_output, 10)
        @test_throws ArgumentError extension._metal_warm_plans!(
            execution, forward, bad_inverse, input, output,
        )
        @test Metal.device() == previous && Metal.global_queue(previous) === queue
        @test all(iszero, Array(input)) && all(iszero, Array(output))
        @test extension._metal_warm_plans!(execution, forward, inverse, input, output) ===
            nothing
        @test all(iszero, Array(input)) && all(iszero, Array(output))
    end
    input = SP.upload(execution, fill(ComplexF32(3, 4), 8, 2))
    forward, inverse = Metal.plan_fft!(input, (1,)), Metal.plan_bfft!(input, (1,))
    other = SP.dzeros(execution, ComplexF32, 10, 2)
    bad_inverse = Metal.plan_bfft!(other, (1,))
    @test_throws ArgumentError extension._metal_warm_plans!(
        execution, forward, bad_inverse, input, input,
    )
    @test Metal.device() == previous && Metal.global_queue(previous) === queue
    @test all(iszero, Array(input))
    @test extension._metal_warm_plans!(
        execution, forward, inverse, input, input,
    ) === nothing
    @test all(iszero, Array(input))
end
