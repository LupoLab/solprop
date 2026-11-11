# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

function linear_table_allocations(flow, output, input, origin, destination)
    flow(output, input, origin, destination)
    return @allocated flow(output, input, origin, destination)
end

function linear_table_varying(execution)
    linear, slope = -0.1 + 0.7im, 0.2 - 0.1im
    coefficients = zeros(ComplexF64, 4, 3, 1)
    coefficients[:, 1, 1] .= linear / 2 + 3slope / 16
    coefficients[:, 2, 1] .= linear / 2 + slope / 4
    coefficients[:, 3, 1] .= slope / 16
    table = SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], coefficients)
    return SolpropJulia.RepresentedOperator(execution, table, 2, 2)
end

@testset "Exact flow tables retain bounded session-owned storage" begin
    for tasks in unique((1, Threads.nthreads(:default))), T in (Float32, Float64)
        ex = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), tasks, true)
        input = Complex{T}[1 + 0.3im -0.2 + 0.7im; 0 -0.0im]
        constant = SolpropJulia.DiagonalOperator(ex, fill(-0.1 + 0.7im, 2, 2))
        for operator in (constant, linear_table_varying(ex))
            flow = SolpropJulia.LinearFlow(ex, operator, input; tabulate = true)
            other = SolpropJulia.LinearFlow(ex, operator, input; tabulate = true)
            reference = SolpropJulia.LinearFlow(ex, operator, input)
            expected, output = similar(input), similar(input)
            count = operator === constant ? 1 : 2
            @test length(flow.table.factors) == count
            @test all(eltype(array) === Complex{T} for array in flow.table.factors)
            @test flow.table.factors[1] !== other.table.factors[1]
            other_type = T === Float32 ? ComplexF64 : ComplexF32
            other_input = other_type.(input)
            other_output = copy(other_input)
            @test_throws ArgumentError flow(other_output, other_input, 0.0, 0.125)
            @test other_output == other_input
            @test flow.table.builds == flow.table.hits == 0
            @test_throws ArgumentError reference(
                other_output, other_input, 0.0, 0.125,
            )
            for (origin, destination) in (
                    (0.0, 0.125), (0.0, 0.125), (0.25, 0.375),
                    (0.25, 0.25), (0.375, 0.25), (0.0, 0.25), (0.0, 0.125),
                )
                flow(output, input, origin, destination)
                reference(expected, input, origin, destination)
                # The same wide factor and one complex multiplication give at most
                # eight working-precision roundings on each arm; no interval slack.
                budget = 16eps(T) * maximum(abs, input)
                @test budget > 0
                @test maximum(abs, output - expected) <= budget
                if origin == destination
                    @test all(isequal.(output, input))
                end
                original = copy(input)
                flow(input, input, origin, destination)
                @test maximum(abs, input - expected) <= budget
                copyto!(input, original)
            end
            @test flow.table.hits > 0 && flow.table.builds > 0
            @test linear_table_allocations(flow, output, input, 0.0, 0.125) == 0
            before = copy(output)
            @test_throws ArgumentError flow(output, input, NaN, NaN)
            @test output == before
            @test_throws ArgumentError flow(
                flow.table.factors[1], input, 0.0, 0.125,
            )
            @test_throws ArgumentError flow(
                output, flow.table.factors[1], 0.0, 0.125,
            )
            if operator !== constant
                # A repeated span outside the represented domain cannot hit a
                # cached in-domain factor and bypass position validation.
                @test_throws ArgumentError flow(output, input, 2.0, 2.125)
                @test output == before
            end
        end
    end
end

@testset "Constant table hits compare actual interval length exactly" begin
    ex = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    input = fill(1.0 + 0.2im, 2, 2)
    operator = SolpropJulia.DiagonalOperator(ex, fill(-0.1 + 0.7im, 2, 2))
    flow = SolpropJulia.LinearFlow(ex, operator, input; tabulate = true)
    output = similar(input)
    flow(output, input, 0.0, 0.125)
    @test flow.table.builds == 1
    flow(output, input, 0.25, 0.375)
    @test flow.table.builds == 1 && flow.table.hits == 1
    flow(output, input, 0.0, nextfloat(0.125))
    @test flow.table.builds == 2
    flow(output, input, 0.0, 0.0)
    @test flow.table.builds == 2 && output == input
end

@testset "Failed table formation leaves output and validity uncommitted" begin
    ex = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    input = fill(1.0 + 0im, 2, 2)
    operator = SolpropJulia.DiagonalOperator(ex, fill(1000.0 + 0im, 2, 2))
    flow = SolpropJulia.LinearFlow(ex, operator, input; tabulate = true)
    output = fill(7.0 + 8im, 2, 2)
    @test_throws SolpropJulia.NonfiniteSourceError flow(output, input, 0.0, 1.0)
    @test output == fill(7.0 + 8im, 2, 2)
    @test !any(flow.table.valid) && flow.table.builds == 0
    flow(output, input, 0.0, 0.001)
    @test all(isfinite, output)
    @test flow.table.builds == 1
end
