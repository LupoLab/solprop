# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

function split_precision_dft(input; inverse = false)
    count, columns = size(input)
    sign = inverse ? 1 : -1
    divisor = inverse ? count : 1
    return Complex{BigFloat}[
        sum(
            Complex{BigFloat}(input[index + 1, column]) *
                cispi(BigFloat(sign * 2 * index * row) / count)
                for index in 0:(count - 1)
        ) / divisor for row in 0:(count - 1), column in 1:columns
    ]
end

function split_precision_reference(spectrum, rate, dz, n_time, kind)
    full = if kind == :field
        vcat(spectrum, conj.(spectrum[(isodd(n_time) ? end : end - 1):-1:2, :]))
    else
        spectrum
    end
    samples = split_precision_dft(full; inverse = true)
    kind == :field && (samples = real.(samples))
    factors = exp.((-big(0.5) .* BigFloat.(rate)) .* BigFloat(dz))
    return split_precision_dft(samples .* factors)[1:size(spectrum, 1), :]
end

function split_precision_field(::Type{T}, n_time, columns, kind) where {T}
    samples = [
        kind == :field ? (row - 2column) / 16 :
            complex((row - 2column) / 16, (row + column) / 32)
            for row in 1:n_time, column in 1:columns
    ]
    count = kind == :field ? n_time ÷ 2 + 1 : n_time
    field = Complex{T}.(split_precision_dft(samples)[1:count, :])
    if kind == :field
        field[1, :] .= real.(field[1, :])
        iseven(n_time) && (field[end, :] .= real.(field[end, :]))
    end
    return field
end

function split_precision_allocated(field, work, execution)
    SP.finish_temporal_split!(field, work, 0.25, execution)
    return @allocated SP.finish_temporal_split!(field, work, 0.25, execution)
end

function split_precision_arithmetic(::Type{T}) where {T}
    return T === Float64 ? SP.SourceArithmetic(T) :
        SP.SourceArithmetic(T; state_exponent = 7, field_exponent = -3)
end

function split_precision_budget(::Type{T}, field, count; spatial = 1) where {T}
    unit = BigFloat(eps(T)) / 2
    # These short transforms have radix lengths <=8. Counting 32 real rounded
    # operations per point per transform pair dominates both mixed-radix FFT
    # butterflies and their normalization; spatial pairs add their own count.
    # The conditional twiddle/libm error hypothesis is <=8u. Product and factor
    # rounding are included in the remaining16 operations. Use the input1-norm
    # before cancellation, not the small magnitude of any final spectral bin.
    operations = 32 * (count + spatial) + 16
    return operations * unit / (1 - operations * unit) * sum(abs, field)
end

@testset "Temporal splits retain state precision and physical scale" begin
    for T in (Float64, Float32), kind in (:field, :envelope), n_time in (7, 8),
            tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        arithmetic = split_precision_arithmetic(T)
        rate = collect(0:(n_time - 1)) ./ 8
        work = SP.TemporalSplitWorkspace(execution, kind, rate, 2; arithmetic)
        field = split_precision_field(T, n_time, 2, kind)
        saved = copy(field)
        expected = split_precision_reference(field, rate, 0.25, n_time, kind)
        fill!(rate, 99.0)
        @test work.rate == collect(0:(n_time - 1)) ./ 8
        @test work.arithmetic == arithmetic
        @test eltype(work.samples) === (kind == :field ? T : Complex{T})
        @test eltype(work.spectrum) === Complex{T}
        @test (work.damping === nothing) == (T === Float64)
        SP.finish_temporal_split!(field, work, 0.25, execution)
        budget = split_precision_budget(T, saved, n_time)
        @test budget > 0
        @test maximum(abs.(Complex{BigFloat}.(field) - expected)) <= budget
        if T === Float32
            @test work.damping == Float32.(exp.((-0.5 .* work.rate) .* 0.25))
        end
        @test split_precision_allocated(field, work, execution) == 0

        # The existing recursive owner inventory must see the new narrow buffer;
        # it must not claim an additional wide mutable factor volume.
        ledger, seen = Dict{Symbol, Int}(), IdDict{Any, Bool}()
        SP._memory_arrays!(ledger, seen, "split", work)
        @test ledger[Symbol("split.rate")] == 8n_time
        @test ledger[Symbol("split.samples")] == sizeof(work.samples)
        @test ledger[Symbol("split.spectrum")] == sizeof(work.spectrum)
        @test get(ledger, Symbol("split.damping"), 0) == (T === Float32 ? 4n_time : 0)

        zero_work = SP.TemporalSplitWorkspace(execution, kind, zeros(n_time), 2; arithmetic)
        fill!(zero_work.samples, NaN)
        fill!(zero_work.spectrum, NaN)
        SP.finish_temporal_split!(saved, zero_work, 0.25, execution)
        @test saved == split_precision_field(T, n_time, 2, kind)
        before = copy(field)
        for bad_step in (0.0, -1.0, NaN, Inf)
            @test_throws ArgumentError SP.finish_temporal_split!(
                field, work, bad_step, execution,
            )
            @test isequal(field, before)
        end
        other = T === Float32 ? Float64 : Float32
        @test_throws DimensionMismatch SP.finish_temporal_split!(
            Complex{other}.(field), work, 0.25, execution,
        )
        @test_throws ArgumentError SP.finish_temporal_split!(
            work.spectrum, work, 0.25, execution,
        )
        if kind == :field
            field[1, 1] += im
            invalid = copy(field)
            @test_throws ArgumentError SP.finish_temporal_split!(
                field, work, 0.25, execution,
            )
            @test field == invalid
        end
    end
end

@testset "Binary32 split factors never narrow wide rate or step before multiplication" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    arithmetic = SP.SourceArithmetic(Float32; state_exponent = -8)
    for (rate, dz) in (
            (2.0^500, 2.0^-500), (2.0^-500, 2.0^500),
            (290log(2.0), 1.0), (400log(2.0), 1.0), (floatmax(Float64), 8.0),
        )
        work = SP.TemporalSplitWorkspace(execution, :envelope, fill(rate, 8), 1; arithmetic)
        field = zeros(ComplexF32, 8, 1)
        field[1] = 8
        SP.finish_temporal_split!(field, work, dz, execution)
        expected = Float32(exp((-0.5 * rate) * dz))
        @test all(==(expected), work.damping)
        # A DC-only constant sample and its length-eight transforms use exact
        # powers of two. This structural witness also retains subnormal damping.
        @test field[1] == 8expected
        @test all(iszero, field[2:end])
        @test all(isfinite, field)
    end
    work = SP.TemporalSplitWorkspace(execution, :envelope, ones(8), 1; arithmetic)
    for storage in (work.rate, work.samples, work.spectrum)
        saved = copy(storage)
        GC.@preserve storage begin
            foreign = unsafe_wrap(
                Array, Ptr{ComplexF32}(pointer(storage)), (8, 1); own = false,
            )
            @test_throws ArgumentError SP.finish_temporal_split!(
                foreign, work, 0.25, execution,
            )
        end
        @test storage == saved
    end
end

function split_precision_spatial(input, shape; inverse = false)
    nx, ny = shape
    sign, divisor = inverse ? (1, nx * ny) : (-1, 1)
    return Complex{BigFloat}[
        sum(
            Complex{BigFloat}(input[row, x + nx * y + 1]) *
                cispi(sign * 2 * (BigFloat(x * kx) / nx + BigFloat(y * ky) / ny))
                for x in 0:(nx - 1), y in 0:(ny - 1)
        ) / divisor for row in axes(input, 1), kx in 0:(nx - 1), ky in 0:(ny - 1)
    ] |> array -> reshape(array, size(input))
end

@testset "Binary32 Cartesian and line splitting preserve inverse ordering" begin
    for T in (Float32, Float64), kind in (:field, :envelope), shape in ((4, 1), (2, 2)),
            tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        arithmetic = split_precision_arithmetic(T)
        rate = collect(0:7) ./ 8
        physical = split_precision_field(T, 8, prod(shape), kind)
        field = Complex{T}.(split_precision_spatial(physical, shape))
        if kind == :field && shape == (4, 1)
            @test any(!isreal, field[1, :])
        end
        original = copy(field)
        temporal = SP.TemporalSplitWorkspace(execution, kind, rate, prod(shape); arithmetic)
        band = SP.CartesianBandWork(execution, shape, size(field, 1); arithmetic)
        work = SP._CartesianTemporalSplit(band, temporal)
        expected = split_precision_spatial(field, shape; inverse = true)
        expected = split_precision_reference(expected, rate, 0.25, 8, kind)
        expected = split_precision_spatial(expected, shape)
        SP.finish_temporal_split!(field, work, 0.25, execution)
        budget = split_precision_budget(T, original, 8; spatial = prod(shape))
        @test budget > 0
        @test maximum(abs.(Complex{BigFloat}.(field) - expected)) <= budget
        @test split_precision_allocated(field, work, execution) == 0
        @test_throws ArgumentError SP.finish_temporal_split!(
            band.band, work, 0.25, execution,
        )
        @test_throws ArgumentError SP.finish_temporal_split!(
            temporal.spectrum, work, 0.25, execution,
        )
    end
end
