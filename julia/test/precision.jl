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
