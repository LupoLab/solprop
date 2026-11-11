# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SJ

@testset "Launch scaling uses exact powers without reciprocals" begin
    execution = SJ.KernelExecution(SJ.KernelAbstractions.CPU(), 1)
    for exponent in (-1000, -126, -5, 0, 80, 500, 1000)
        arithmetic = SJ.SourceArithmetic(Float32; state_exponent = exponent)
        input = reshape(
            [
                complex(ldexp(1.1, exponent), ldexp(-0.7, exponent)),
                complex(0.0, -0.0),
            ], 2, 1,
        )
        state = SJ._prepare_scaled_state(execution, input, arithmetic)
        @test eltype(state) === ComplexF32
        @test !Base.mightalias(state, input)
        @test state[1] == ComplexF32(1.1, -0.7)
        @test isequal(state[2], ComplexF32(0.0, -0.0))
        physical = SJ._physical_state_value(state[1], arithmetic)
        # Each independent component has one binary32 rounding; power scaling is
        # exact on these normal-range fixtures. No FFT or propagation is involved.
        for part in (real, imag)
            expected = BigFloat(part(input[1]))
            @test abs(BigFloat(part(physical)) - expected) <= big(2.0)^-24 * abs(expected)
        end
        @test SJ._scaled_absolute_tolerance(ldexp(0.5, exponent), arithmetic) == 0.5
        @test SJ._scaled_absolute_tolerance(0.0, arithmetic) == 0.0
    end
    smallest = nextfloat(0.0)
    arithmetic = SJ.SourceArithmetic(Float32; state_exponent = -1074)
    input = fill(complex(smallest), 1, 1)
    state = SJ._prepare_scaled_state(execution, input, arithmetic)
    @test state == fill(1.0f0 + 0.0f0im, 1, 1)
    @test SJ._physical_state_value(state[1], arithmetic) == input[1]
    @test SJ._scaled_absolute_tolerance(smallest, arithmetic) == 1.0
    @test_throws ArgumentError SJ._scaled_absolute_tolerance(1.0, arithmetic)

    wide = SJ.SourceArithmetic(Float64)
    input = reshape(
        [
            complex(floatmax(Float64), floatmax(Float64)),
            complex(0.0, -0.0),
        ], 2, 1
    )
    state = SJ._prepare_scaled_state(execution, input, wide)
    @test isequal(state, input)
    @test !Base.mightalias(state, input)
    @test isequal(SJ._physical_state_value(state[2], wide), input[2])
    @test SJ._scaled_absolute_tolerance(floatmax(Float64), wide) == floatmax(Float64)

    narrow = SJ.SourceArithmetic(Float32)
    for bad in (Inf, NaN, Float64(floatmax(Float32)) * 2)
        @test_throws ArgumentError SJ._prepare_scaled_state(
            execution, fill(complex(bad), 1, 1), narrow,
        )
    end
    # Gradual state underflow is permitted. It is not the separate table rule
    # which requires a block exponent for a uniformly tiny coefficient buffer.
    tiny = Float64(nextfloat(0.0f0))
    state = SJ._prepare_scaled_state(
        execution, reshape(complex.([tiny, tiny / 4]), 2, 1), narrow,
    )
    @test state[1] == complex(nextfloat(0.0f0))
    @test iszero(state[2])
    for bad in (-1.0, Inf, NaN)
        @test_throws ArgumentError SJ._scaled_absolute_tolerance(bad, narrow)
    end
    @test_throws ArgumentError SJ._prepare_scaled_state(execution, zeros(2, 1), wide)
end

@testset "Launch scaling owns Cartesian-indexed and strided matrix copies" begin
    execution = SJ.KernelExecution(SJ.KernelAbstractions.CPU(), 1)
    values = reshape(ComplexF64.(1:24, -24:-1), 4, 6)
    for input in (PermutedDimsArray(values, (2, 1)), view(values, 1:2:4, 2:2:6))
        original = copy(input)
        for T in (Float32, Float64)
            arithmetic = SJ.SourceArithmetic(T; state_exponent = T === Float32 ? 3 : 0)
            state = SJ._prepare_scaled_state(execution, input, arithmetic)
            expected = Complex{T}.(original ./ (T === Float32 ? 8.0 : 1.0))
            @test isequal(state, expected)
            @test !Base.mightalias(state, input)
            @test isequal(input, original)
        end
    end
end
