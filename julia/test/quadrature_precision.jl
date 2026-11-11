# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @inferred, @test, @testset, @test_throws
import SolpropJulia as SP

function quadrature_precision_reference(fine, coarse)
    numerator = denominator = zero(Rational{BigInt})
    for index in eachindex(fine, coarse)
        a, b = fine[index], coarse[index]
        for component in (real, imag)
            first = Rational{BigInt}(component(a))
            second = Rational{BigInt}(component(b))
            denominator += first^2
            numerator += (second - first)^2
        end
    end
    return setprecision(BigFloat, 384) do
        sqrt(BigFloat(numerator) / BigFloat(denominator))
    end
end

function quadrature_precision_bound(fine, coarse)
    reference = quadrature_precision_reference(fine, coarse)
    rows, columns = size(fine)
    unit = BigFloat(eps(Float32)) / 2
    gamma(count) = count * unit / (1 - count * unit)
    # QUADRATURE_PRECISION.md propagates normalized product error through both
    # Sum2s passes, including each column's prior error before the merge. These
    # are bounds for the supplied represented inputs, not source/FFT errors.
    pointwise = gamma(12) + 32length(fine) * 2 * BigFloat(nextfloat(0.0f0))
    row_sum = unit + gamma(rows - 1)^2
    column_sum = unit + gamma(columns - 1)^2
    relative = (1 + pointwise) * (1 + row_sum) * (1 + column_sum) - 1
    @test 0 < relative < 1
    host_unit = BigFloat(eps(Float64)) / 2
    host_gamma = 8host_unit / (1 - 8host_unit)
    bound = (sqrt((1 + relative) / (1 - relative)) * (1 + host_gamma) - 1) * reference
    return reference, bound
end

function quadrature_precision_assert(work, fine, coarse, execution)
    expected, bound = quadrature_precision_bound(fine, coarse)
    actual = SP.quadrature_error!(work, fine, coarse, execution)
    @test bound > 0
    @test isfinite(actual)
    @test abs(BigFloat(actual) - expected) <= bound
    return actual
end

function quadrature_precision_allocated(work, fine, coarse, execution)
    SP.quadrature_error!(work, fine, coarse, execution)
    return @allocated SP.quadrature_error!(work, fine, coarse, execution)
end

@testset "Binary32 quadrature uses narrow owned partials and protected sums" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))),
            (rows, columns) in ((1, 1), (7, 3), (33, 5))
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        arithmetic = SP.SourceArithmetic(Float32; state_exponent = -73, field_exponent = 91)
        work = SP.QuadratureWorkspace(execution, columns; arithmetic)
        @test eltype(work.maxima) === Float32
        @test eltype(work.sums) === Float32
        @test size(work.maxima) == (3, columns)
        @test size(work.sums) == (2, columns)
        ledger, seen = Dict{Symbol, Int}(), IdDict{Any, Bool}()
        SP._memory_arrays!(ledger, seen, "quadrature", work)
        @test sum(values(ledger)) == 20columns
        @test length(ledger) == 2
        fine = ComplexF32[
            complex(sin(0.3row + column), cos(row - 0.2column))
                for row in 1:rows, column in 1:columns
        ]
        coarse = ComplexF32[
            complex(cos(0.7row - column), sin(row + 0.4column))
                for row in 1:rows, column in 1:columns
        ]
        before = copy(fine), copy(coarse)
        quadrature_precision_assert(work, fine, coarse, execution)
        @test fine == before[1] && coarse == before[2]
        @test quadrature_precision_allocated(work, fine, coarse, execution) == 0
        @test SP.quadrature_error!(work, fine, fine, execution) == 0.0
        @test quadrature_precision_allocated(work, fine, fine, execution) == 0
        for factor in (0.0f0, 0.5f0, 2.0f0, -1.0f0)
            scaled = factor .* fine
            actual = quadrature_precision_assert(work, fine, scaled, execution)
            # Exact dyadic scaling gives this analytic ratio. Use the independent
            # operation-derived bound, rather than require backend bit agreement.
            _, bound = quadrature_precision_bound(fine, scaled)
            @test abs(BigFloat(actual) - abs(BigFloat(factor) - 1)) <= bound
        end
        adjacent = complex.(nextfloat.(real.(fine)), prevfloat.(imag.(fine)))
        quadrature_precision_assert(work, fine, adjacent, execution)
        scales = @inferred SP._quadrature_scales!(work.maxima, fine, coarse, execution)
        @test scales isa Tuple{Float32, Float32, Bool}
        sums = @inferred SP._quadrature_sums!(
            work.sums, fine, coarse, scales..., execution,
        )
        @test sums isa Tuple{Float32, Float32}
    end
end

@testset "Binary32 quadrature retains extremes and independent tiny differences" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        work = SP.QuadratureWorkspace(
            execution, 3; arithmetic = SP.SourceArithmetic(Float32),
        )
        for exponent in (-149, -126, -80, 0, 80, 125)
            value = ldexp(1.0f0, exponent)
            fine = fill(complex(value, -value), 9, 3)
            coarse = 2.0f0 .* fine
            quadrature_precision_assert(work, fine, coarse, execution)
        end
        largest = floatmax(Float32)
        fine = ComplexF32[1 largest 2; largest 2 3; 4 3 -largest]
        for input in (fine, reverse(fine; dims = 1), reverse(fine; dims = 2))
            coarse = -input
            @test any(value -> !isfinite(value), coarse - input)
            quadrature_precision_assert(work, input, coarse, execution)
            @test quadrature_precision_allocated(work, input, coarse, execution) == 0
        end
        fine = zeros(ComplexF32, 17, 3)
        fine[1] = 2.0f0^120
        coarse = copy(fine)
        coarse[2:end] .= 2.0f0^-120
        actual = quadrature_precision_assert(work, fine, coarse, execution)
        @test 0 < actual < Float64(nextfloat(0.0f0))
        fine .= nextfloat(0.0f0)
        coarse .= 1
        actual = quadrature_precision_assert(work, fine, coarse, execution)
        @test actual > floatmax(Float32)
        # A large unchanged component must not set the difference scale. Both
        # normal and subnormal next-neighbor differences remain independently seen.
        fine .= 1
        fine[2] = floatmin(Float32)
        coarse = copy(fine)
        coarse[2] = nextfloat(real(fine[2]))
        quadrature_precision_assert(work, fine, coarse, execution)
    end
end

@testset "Long positive sums retain contributions ordinary accumulation loses" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        work = SP.QuadratureWorkspace(
            execution, 2; arithmetic = SP.SourceArithmetic(Float32),
        )
        fine = ones(ComplexF32, 8193, 2)
        coarse = fill(ComplexF32(1 + 2.0f0^-12), size(fine))
        coarse[1, :] .= 2
        quadrature_precision_assert(work, fine, coarse, execution)
        # The exact positive squared-difference sum is a representable dyadic;
        # ordinary Float32 accumulation would round each 2^-24 tail away.
        @test all(==(1.0f0 + 2.0f0^-11), work.sums[2, :])
        @test all(==(8193.0f0), work.sums[1, :])
        @test quadrature_precision_allocated(work, fine, coarse, execution) == 0
    end
end

struct QuadratureCountMatrix <: AbstractMatrix{ComplexF32}
    rows::Int
    columns::Int
end
Base.size(matrix::QuadratureCountMatrix) = (matrix.rows, matrix.columns)
Base.getindex(::QuadratureCountMatrix, ::Int, ::Int) =
    error("an inadmissible reduction count must refuse before reading input")

@testset "Quadrature precision/count/nonfinite admission preserves all scratch" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    arithmetic = SP.SourceArithmetic(Float32)
    @test_throws ArgumentError SP.QuadratureWorkspace(execution, 1 << 24; arithmetic)
    @test SP._float32_sum_count((1 << 24) - 1) === nothing
    work = SP.QuadratureWorkspace(execution, 2; arithmetic)
    fine = ones(ComplexF32, 3, 2)
    fill!(work.maxima, 7)
    fill!(work.sums, 9)
    huge = QuadratureCountMatrix(1 << 24, 2)
    @test_throws ArgumentError SP.quadrature_error!(work, huge, huge, execution)
    @test all(==(7), work.maxima) && all(==(9), work.sums)
    for pair in ((nothing, fine), (fine, nothing), (ComplexF64.(fine), ComplexF64.(fine)))
        @test_throws ArgumentError SP.quadrature_error!(work, pair..., execution)
        @test all(==(7), work.maxima) && all(==(9), work.sums)
    end
    for invalid in (NaN32, Inf32, -Inf32)
        bad = copy(fine)
        bad[2] = complex(invalid, 1.0f0)
        for pair in ((bad, fine), (fine, bad), (zero(fine), bad))
            @test_throws ArgumentError SP.quadrature_error!(work, pair..., execution)
            @test all(==(7), work.maxima) && all(==(9), work.sums)
        end
    end
    fill!(work.maxima, NaN32)
    fill!(work.sums, NaN32)
    @test isnan(SP.quadrature_error!(work, nothing, nothing, execution))
    @test isnan(SP.quadrature_error!(work, zero(fine), fine, execution))
    @test isnan(SP.quadrature_error!(work, zero(fine), zero(fine), execution))
    @test SP.quadrature_error!(work, fine, fine, execution) == 0.0
    quadrature_precision_assert(work, fine, -fine, execution)
    @test all(isfinite, work.maxima) && all(isfinite, work.sums)
    @test quadrature_precision_allocated(work, nothing, nothing, execution) == 0
end

@testset "Quadrature rejects foreign and index-storage aliases before any writes" begin
    for T in (Float32, Float64), tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        work = SP.QuadratureWorkspace(execution, 2; arithmetic = SP.SourceArithmetic(T))
        fine = ones(Complex{T}, 1, 2)
        fill!(work.maxima, 7)
        fill!(work.sums, 9)
        GC.@preserve work begin
            aliased = unsafe_wrap(
                Array, Ptr{Complex{T}}(Ptr{UInt8}(pointer(work.maxima)) + sizeof(T)),
                (1, 2); own = false,
            )
            @test !Base.mightalias(aliased, work.maxima)
            before = copy(work.maxima), copy(work.sums), copy(aliased)
            for pair in ((aliased, fine), (fine, aliased))
                @test_throws ArgumentError SP.quadrature_error!(work, pair..., execution)
                @test work.maxima == before[1] && work.sums == before[2]
                @test aliased == before[3]
            end
            indices = unsafe_wrap(
                Array, Ptr{Int}(Ptr{UInt8}(pointer(work.maxima)) + sizeof(Int)),
                (2,); own = false,
            )
            indices .= (1, 2)
            indexed = view(fine, :, indices)
            before = copy(work.maxima), copy(work.sums), copy(indices)
            @test !Base.mightalias(indexed, work.maxima)
            for pair in ((indexed, fine), (fine, indexed))
                @test_throws ArgumentError SP.quadrature_error!(work, pair..., execution)
                @test work.maxima == before[1] && work.sums == before[2]
                @test indices == before[3]
            end
        end
        storage = fill(T(5), 12)
        GC.@preserve storage begin
            maxima = unsafe_wrap(Array, pointer(storage), (3, 2); own = false)
            sums = unsafe_wrap(
                Array, Ptr{T}(Ptr{UInt8}(pointer(storage)) + sizeof(T)),
                (2, 2); own = false,
            )
            malformed = SP.QuadratureWorkspace(maxima, sums)
            @test !Base.mightalias(maxima, sums)
            @test_throws ArgumentError SP.quadrature_error!(
                malformed, fine, fine, execution,
            )
            @test all(==(5), storage)
        end
        parent = reshape(complex.(T.(1:12), T.(12:-1:1)), 3, 4)
        input = view(parent, :, 1:2:4)
        coarse = -input
        before = copy(parent)
        if T === Float32
            quadrature_precision_assert(work, input, coarse, execution)
        else
            @test abs(SP.quadrature_error!(work, input, coarse, execution) - 2) <= 32eps(T)
        end
        @test parent == before
        @test quadrature_precision_allocated(work, input, coarse, execution) == 0
    end
end
