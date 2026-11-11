# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

function observation_precision_sum(values)
    sum = SolpropJulia._Float32Sum()
    for value in values
        sum = SolpropJulia._float32_add(sum, value)
    end
    return SolpropJulia._float32_value(sum)
end

function observation_precision_allocated(values)
    observation_precision_sum(values)
    return @allocated observation_precision_sum(values)
end

@testset "Protected observation normalization and compensated reduction" begin
    for values in (
            Float32[1, 2.0^-24, -1],
            Float32[1, 2.0^-24, 2.0^-48, -2.0^-24, -2.0^-48, -1],
            repeat(Float32[1, 2.0^-20, -1, 2.0^-22], 256),
            Float32[floatmin(Float32), nextfloat(0.0f0), -floatmin(Float32)],
        )
        actual = observation_precision_sum(values)
        exact = sum(BigFloat.(values))
        magnitude = sum(abs, BigFloat.(values))
        u = BigFloat(2.0)^-24
        gamma = (length(values) - 1) * u / (1 - (length(values) - 1) * u)
        # Sum2s Prop4.5 includes subnormal additions under gradual underflow.
        bound = u * abs(exact) + gamma^2 * magnitude
        @test bound > 0
        @test abs(BigFloat(actual) - exact) <= bound
        @test observation_precision_allocated(values) == 0
    end
    for value in (0.0f0, nextfloat(0.0f0), floatmin(Float32), 1.0f0, floatmax(Float32))
        values = reshape(ComplexF32[value + value * im], 1, 1)
        exponent = SolpropJulia._observation_exponent(values)
        @test exponent == SolpropJulia._observation_column_exponent(values, 1)
        normalized = SolpropJulia._observation_scaled(values[1], exponent)
        @test isfinite(normalized) && max(abs(real(normalized)), abs(imag(normalized))) <= 1
        @test ldexp(Float64(real(normalized)), exponent) == Float64(value)
        @test isfinite(SolpropJulia._observation_square(values[1], exponent))
    end
    @test SolpropJulia._observation_restore(0.25f0, 1200, 2.0^-300) == 2.0^898
    @test SolpropJulia._observation_restore(0.5f0, -1000, 2.0^1000) == 0.5
    @test SolpropJulia._observation_restore(0.0f0, 2000) == 0.0
    @test_throws ArgumentError SolpropJulia._observation_restore(1.0f0, 1024)
    @test_throws ArgumentError SolpropJulia._observation_restore(Inf32, 0)
    @test_throws ArgumentError SolpropJulia._observation_exponent([NaN32])
    @test_throws ArgumentError SolpropJulia._observation_column_exponent(
        fill(Inf32, 1, 1), 1
    )
end

@testset "Compensated binary32 admission uses the strict theorem domain" begin
    @test SolpropJulia._float32_sum_count(0) === nothing
    @test SolpropJulia._float32_sum_count((1 << 24) - 1) === nothing
    @test_throws ArgumentError SolpropJulia._float32_sum_count(1 << 24)
    @test_throws ArgumentError SolpropJulia._float32_sum_count(-1)
end

@testset "Physical factor restoration avoids unsafe products" begin
    @test SolpropJulia._observation_restore_ratio(
        0.5f0, 1000, (2.0^500,), (2.0^600,)
    ) == 2.0^899
    @test SolpropJulia._observation_restore_ratio(
        0.5f0, -1000, (2.0^-500,), (2.0^-600,)
    ) == 2.0^-901
    @test SolpropJulia._observation_restore_ratio(-0.5f0, 0, (2.0, 4.0), (2.0,)) == -2.0
    @test SolpropJulia._observation_restore_ratio(0.5f0, 2000, (0.0,), (2.0,)) == 0.0
    @test_throws ArgumentError SolpropJulia._observation_restore_ratio(1.0f0, 0, (Inf,), ())
    @test_throws ArgumentError SolpropJulia._observation_restore_ratio(1.0f0, 0, (), (0.0,))
    @test_throws ArgumentError SolpropJulia._observation_restore_ratio(1.0f0, 2000, (), ())
end

function product_precision_allocated(factors, exponent)
    SolpropJulia._observation_product_exponent(factors)
    SolpropJulia._observation_product(factors, exponent)
    return @allocated begin
        SolpropJulia._observation_product_exponent(factors)
        SolpropJulia._observation_product(factors, exponent)
    end
end

@testset "Product exponents follow actual mixed-precision terms" begin
    cases = (
        (nextfloat(0.0f0), 2.0^700),
        (2.0f0^100, 2.0f0^-100, 2.0^-700),
        (-2.0f0^100, 2.0f0^-100, 3.0, 0.125f0),
        (nextfloat(0.0), nextfloat(0.0f0), floatmax(Float64), 2.0f0^100),
        (0.75f0, 0.5, -0.875f0, 0.625, 0.75f0, 0.5, 0.875f0, 0.625),
    )
    for factors in cases
        exponent = SolpropJulia._observation_product_exponent(factors)
        value = SolpropJulia._observation_product(factors, exponent)
        expected = prod(BigFloat.(factors)) * big(2.0)^-exponent
        k = length(factors)
        gamma = (2k - 1) * big(2.0)^-24 / (1 - (2k - 1) * big(2.0)^-24)
        @test gamma > 0
        @test abs(BigFloat(value) - expected) <= gamma * abs(expected)
        @test isfinite(value) && abs(value) >= 2.0f0^-8
        @test product_precision_allocated(factors, exponent) == 0
    end
    @test SolpropJulia._observation_product_exponent((0.0f0, 2.0^1000)) == typemin(Int)
    @test SolpropJulia._observation_product((0.0f0, 2.0^1000), -2000) == 0.0f0
    @test_throws ArgumentError SolpropJulia._observation_product((), 0)
    @test_throws ArgumentError SolpropJulia._observation_product(ntuple(_ -> 1.0f0, 9), 0)
end
