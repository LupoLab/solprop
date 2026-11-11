# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SJ

@testset "Explicit arithmetic and independent physical scales" begin
    for T in (Float32, Float64)
        arithmetic = SJ.SourceArithmetic(T)
        @test SJ.source_type(arithmetic) === T
        @test SJ.source_state_scale(arithmetic) === 1.0
        @test SJ.source_field_scale(arithmetic) === 1.0
        @test isbitstype(typeof(arithmetic))
    end
    for exponent in (-1074, -126, 0, 80, 1023)
        arithmetic = SJ.SourceArithmetic(Float32; state_exponent = exponent)
        @test SJ.source_state_scale(arithmetic) === ldexp(1.0, exponent)
        @test SJ.source_field_scale(arithmetic) === ldexp(1.0, exponent)
    end
    distinct = SJ.SourceArithmetic(Float32; state_exponent = 5, field_exponent = 12)
    @test SJ.source_state_scale(distinct) === 32.0
    @test SJ.source_field_scale(distinct) === 4096.0
    @test_throws ArgumentError SJ.SourceArithmetic(Float16)
    @test_throws ArgumentError SJ.SourceArithmetic(Int)
    @test_throws ArgumentError SJ.SourceArithmetic(ComplexF32)
    @test_throws ArgumentError SJ.SourceArithmetic(Real)
    @test_throws ArgumentError SJ.SourceArithmetic(Float64; state_exponent = 1)
    @test_throws ArgumentError SJ.SourceArithmetic(Float64; field_exponent = 1)
    @test_throws ArgumentError SJ.SourceArithmetic(Float32; state_exponent = -1075)
    @test_throws ArgumentError SJ.SourceArithmetic(Float32; field_exponent = 1024)
end

function source_narrow_allocated(destination, input)
    SJ.source_narrow!(destination, input, :stage)
    return @allocated SJ.source_narrow!(destination, input, :stage)
end

function source_coefficient_allocated(arithmetic)
    SJ.source_coefficient(arithmetic, :loss, 1.0e25, 2.0e-18; exponent = -20)
    return @allocated SJ.source_coefficient(
        arithmetic, :loss, 1.0e25, 2.0e-18; exponent = -20
    )
end

@testset "Named source narrowing retains scales and oracle range" begin
    execution = SJ.KernelExecution(SJ.KernelAbstractions.CPU(), 1)
    for T in (Float32, Float64), complex_input in (false, true)
        arithmetic = SJ.SourceArithmetic(T)
        real_input = reshape([0.0, -0.0, 1.25, -2.5, 0.125, -0.75], 3, 2)
        input = complex_input ? complex.(real_input, reverse(real_input; dims = 1)) :
            real_input
        original = copy(input)
        output, fact = SJ.source_narrow(execution, input, arithmetic, :map; exponent = 3)
        @test eltype(output) === (complex_input ? Complex{T} : T)
        @test output == T(8) .* input
        @test input == original
        @test !Base.mightalias(output, input)
        @test fact.name === :map
        @test fact.block_exponent == 4
        @test fact.folded_exponent == 3
        @test !fact.normalized
        @test fact.largest == maximum(abs, original) * 8
        normalized, normal_fact = SJ.source_narrow(
            execution, input, arithmetic, :kernel; exponent = 3, normalize = true
        )
        @test 1 <= normal_fact.largest < 2
        @test normalized .* 2.0^normal_fact.block_exponent == output
        @test normal_fact.normalized
        @test normal_fact.folded_exponent == 3
        destination = similar(output)
        @test SJ.source_narrow!(destination, input, :map; exponent = 3) == fact
        @test destination == output
        @test source_narrow_allocated(destination, input) == 0
    end
    arithmetic = SJ.SourceArithmetic(Float32)
    tiny = Float64(floatmin(Float32))
    subnormal = Float64(nextfloat(0.0f0))
    @test_throws ArgumentError SJ.source_narrow(execution, [tiny / 2], arithmetic, :small)
    @test_throws ArgumentError SJ.source_narrow(
        execution, [prevfloat(1.0), subnormal], arithmetic, :small_entry
    )
    admitted, fact = SJ.source_narrow(execution, [1.0, subnormal], arithmetic, :normwise)
    @test admitted == [1.0f0, nextfloat(0.0f0)]
    @test fact.smallest_nonzero == subnormal
    admitted, _ = SJ.source_narrow(execution, [1.0, subnormal / 4], arithmetic, :negligible)
    @test admitted == [1.0f0, 0.0f0]
    normalized, fact = SJ.source_narrow(
        execution, [2.0^-600, -2.0^-601], arithmetic, :raman;
        exponent = -600, normalize = true
    )
    @test normalized == [1.0f0, -0.5f0]
    @test fact.block_exponent == -1200
    @test fact.largest == 1.0
    normalized, fact = SJ.source_narrow(
        execution, [nextfloat(0.0)], arithmetic, :smallest; normalize = true
    )
    @test normalized == [1.0f0]
    @test fact.block_exponent == -1074
    for value in (Inf, NaN, Float64(floatmax(Float32)))
        @test_throws ArgumentError SJ.source_narrow(execution, [value], arithmetic, :bad)
    end
    @test_throws ArgumentError SJ.source_narrow(
        execution, [2.0], arithmetic, :overflow; exponent = typemax(Int)
    )
    zeros, zero_fact = SJ.source_narrow(execution, [-0.0, 0.0], arithmetic, :zeros)
    @test signbit(zeros[1]) && !signbit(zeros[2])
    @test zero_fact.block_exponent == 0
    @test zero_fact.smallest_nonzero == zero_fact.largest == 0
    empty, empty_fact = SJ.source_narrow(execution, Float64[], arithmetic, :empty)
    @test isempty(empty)
    @test empty_fact.smallest_nonzero == 0
    indexed = PermutedDimsArray(reshape(collect(1.0:12.0), 3, 4), (2, 1))
    narrowed, indexed_fact = SJ.source_narrow(execution, indexed, arithmetic, :cartesian)
    @test narrowed == Float32.(indexed)
    @test narrowed isa Matrix{Float32}
    @test indexed_fact.largest == 12.0
end

@testset "Source range admission precedes every destination write" begin
    for bad in (NaN, Inf, Float64(floatmax(Float32)), 1.0e-300)
        destination = fill(7.0f0, 4)
        input = [0.25, 0.125, 0.5, bad]
        @test_throws ArgumentError SJ.source_narrow!(destination, input, :later_invalid)
        @test destination == fill(7.0f0, 4)
    end
    owner = collect(1.0:6.0)
    original = copy(owner)
    @test_throws ArgumentError SJ.source_narrow!(view(owner, 2:6), view(owner, 1:5), :alias)
    @test owner == original
    GC.@preserve owner begin
        foreign = unsafe_wrap(Array, pointer(owner, 2), 5; own = false)
        @test_throws ArgumentError SJ.source_narrow!(foreign, view(owner, 1:5), :foreign)
        @test owner == original
    end
    @test_throws DimensionMismatch SJ.source_narrow!(zeros(Float32, 2), [1.0], :shape)
    @test_throws ArgumentError SJ.source_narrow!(zeros(ComplexF32, 2), [1.0, 2.0], :kind)
    SJ.source_narrow!(owner, owner, :identical; exponent = 1)
    @test owner == 2original
    underflow = fill(7.0, 2)
    @test_throws ArgumentError SJ.source_narrow!(
        underflow, [1.0, nextfloat(0.0)], :binary64; exponent = -1
    )
    @test underflow == fill(7.0, 2)
end

@testset "Host source coefficients combine exponents before one rounding" begin
    for T in (Float32, Float64)
        arithmetic = SJ.SourceArithmetic(T)
        # Each naive initial pair respectively underflows or overflows binary64.
        for factors in ((2.0^-900, 2.0^-900, 2.0^900), (2.0^900, 2.0^900, 2.0^-900))
            shift = factors[1] < 1 ? 900 : -900
            @test SJ.source_coefficient(
                arithmetic, :balanced, factors...; exponent = shift
            ) == one(T)
        end
        @test SJ.source_coefficient(arithmetic, :zero, -0.0, -3.0) === zero(T)
        @test SJ.source_coefficient(arithmetic, :zero, 0.0, -3.0) === -zero(T)
        @test SJ.source_coefficient(arithmetic, :unit) === one(T)
        @test_throws ArgumentError SJ.source_coefficient(arithmetic, :bad, 0.0, NaN)
        @test_throws ArgumentError SJ.source_coefficient(arithmetic, :bad, Inf)
        @test_throws ArgumentError SJ.source_coefficient(
            arithmetic, :overflow, 2.0; exponent = typemax(Int)
        )
        @test_throws ArgumentError SJ.source_coefficient(
            arithmetic, :underflow, 1.0; exponent = -2000
        )
        @test source_coefficient_allocated(arithmetic) == 0
    end
    arithmetic = SJ.SourceArithmetic(Float32)
    @test_throws ArgumentError SJ.source_coefficient(arithmetic, :tiny, 2.0^-127)
    @test SJ.source_coefficient(arithmetic, :normal, 2.0^-126) === floatmin(Float32)
    for density in (1.0e-100, 1.0, 1.0e100), potential in (1.0e-200, 2.1e-18, 1.0e200)
        exponent = -frexp(density)[2] - frexp(potential)[2]
        exact = Rational{BigInt}(density) * Rational{BigInt}(potential) *
            Rational{BigInt}(2)^exponent
        actual = SJ.source_coefficient(arithmetic, :loss, density, potential; exponent)
        # Two binary64 mantissa multiplications plus one binary32 rounding: the
        # normal final result has relative error bounded by gamma_3(binary32).
        budget = 3eps(Float32) / (1 - 3eps(Float32)) * abs(exact)
        @test abs(Rational{BigInt}(actual) - exact) <= budget
    end
end
