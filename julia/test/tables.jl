# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function table_cycle!(output, difference, table, positions, execution)
    SolpropJulia.evaluate_table!(output, table, positions, execution; active_positions = 5)
    SolpropJulia.difference_table!(difference, table, 0.25, 1.75, execution)
    return nothing
end

@testset "Owned represented profiles" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    edges = [0.0, 1.0, 2.0]
    coefficients = zeros(2, 3, 2)
    coefficients[1, :, 1] = [3.125, 2.5, 0.375]
    coefficients[1, :, 2] = [11.125, 5.5, 0.375]
    coefficients[2, :, 1] = [0.5, 0.5, 0.0]
    coefficients[2, :, 2] = [1.5, 0.5, 0.0]
    table = SolpropJulia.ChebyshevTable(
        ex, edges, coefficients; lower = -1.0e-10, upper = 2.0 + 1.0e-10,
    )
    @test table.breakpoints == edges
    @test table.coefficients == coefficients
    @test !Base.mightalias(table.breakpoints, edges)
    @test !Base.mightalias(table.coefficients, coefficients)
    coefficients[1] = 0
    @test table.coefficients[1] == 3.125
    positions = [0.0, 0.25, 1.0, 1.75, 2.0, NaN]
    output = fill(NaN, 2, 8)
    difference = zeros(2)
    table_cycle!(output, difference, table, positions, ex)
    for index in 1:5
        position = positions[index]
        expected = [1 + 2position + 3position^2, position]
        # Three Chebyshev coefficients and the independent quadratic evaluation:
        # 32eps times the value scale covers both bounded arithmetic paths.
        @test maximum(abs, output[:, index] - expected) <=
            32eps(Float64) * maximum(abs, expected)
    end
    @test iszero(output[:, 6:8])
    @test maximum(abs, difference - [12.0, 1.5]) <= 64eps(Float64) * 12
    @test @allocated(table_cycle!(output, difference, table, positions, ex)) == 0
    for position in (-5.0e-11, 2.0 + 5.0e-11)
        SolpropJulia.evaluate_table!(difference, table, position, ex)
        expected = [1 + 2position + 3position^2, position]
        @test maximum(abs, difference - expected) <=
            32eps(Float64) * maximum(abs, expected)
    end
    SolpropJulia.difference_table!(difference, table, 1.0, 1.0, ex)
    @test iszero(difference)
    saved = copy(output)
    @test_throws ArgumentError SolpropJulia.evaluate_table!(
        output, table, positions, ex; active_positions = 6,
    )
    @test isequal(output, saved)
    @test_throws ArgumentError SolpropJulia.evaluate_table!(difference, table, Inf, ex)
    @test_throws ArgumentError SolpropJulia.evaluate_table!(difference, table, 3.0, ex)
    @test_throws DimensionMismatch SolpropJulia.evaluate_table!(zeros(3), table, 0.5, ex)
    @test_throws ArgumentError SolpropJulia.ChebyshevTable(ex, Float64[], zeros(2, 3, 0))
    @test_throws ArgumentError SolpropJulia.ChebyshevTable(ex, [0.0, 0.0], zeros(2, 3, 1))
    @test_throws DimensionMismatch SolpropJulia.ChebyshevTable(
        ex, [0.0, 1.0], zeros(2, 3, 2),
    )

    jump = SolpropJulia.ChebyshevTable(ex, edges, reshape([1.0, 2.0], 1, 1, 2))
    one_value = zeros(1)
    for (position, expected) in ((0.0, 1.0), (prevfloat(1.0), 1.0), (1.0, 2.0), (2.0, 2.0))
        SolpropJulia.evaluate_table!(one_value, jump, position, ex)
        @test only(one_value) == expected
    end
    linear = SolpropJulia.ChebyshevTable(ex, [0.0, 1.0], reshape([0.5, 0.5], 1, 2, 1))
    SolpropJulia.evaluate_table!(one_value, linear, 0.25, ex)
    @test abs(only(one_value) - 0.25) <= 2eps(Float64)
    higher = SolpropJulia.ChebyshevTable(
        ex, [-1.0, 1.0], reshape([0.0, 0.0, 0.0, 1.0, 1.0], 1, 5, 1),
    )
    for point in (-1.0, -0.7, -0.25, 0.0, 0.3, 0.75, 1.0)
        SolpropJulia.evaluate_table!(one_value, higher, point, ex)
        expected = 4point^3 - 3point + 8point^4 - 8point^2 + 1
        # Five-term Clenshaw against an independent monomial polynomial, bounded
        # absolutely by 64eps times the sum of monomial term magnitudes.
        scale = 4abs(point)^3 + 3abs(point) + 8abs(point)^4 + 8abs(point)^2 + 1
        @test abs(only(one_value) - expected) <= 64eps(Float64) * scale
    end
end
