# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function _weight_table(execution, count)
    coefficients = zeros(Float64, count, 3, 2)
    for interval in 1:2, index in 1:count
        coefficients[index, :, interval] .= (2 + index / 8, 0.125, -0.0625)
    end
    return SolpropJulia.ChebyshevTable(execution, [-1.0, 0.25, 1.0], coefficients)
end

function _weight_reference(table, z)
    interval = z < table.breakpoints[2] ? 1 : 2
    left, right = BigFloat.(table.breakpoints[interval:(interval + 1)])
    coordinate = (2BigFloat(z) - left - right) / (right - left)
    return [
        BigFloat(table.coefficients[index, 1, interval]) +
            BigFloat(table.coefficients[index, 2, interval]) * coordinate +
            BigFloat(table.coefficients[index, 3, interval]) * (2coordinate^2 - 1)
            for index in axes(table.coefficients, 1)
    ]
end

function _weight_allocations(output, weight, z, execution, scratch)
    evaluate! = SolpropJulia.evaluate_weight!
    evaluate!(output, weight, z, execution; work = scratch)
    return @allocated evaluate!(output, weight, z, execution; work = scratch)
end

@testset "Modal power-weight representations and owned data" begin
    for tasks in (1, 2)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        values = reshape(collect(1.0:6.0), 2, 3)
        constant = SolpropJulia.ObservationWeight(execution, values)
        values .= NaN
        output = zeros(2, 3)
        SolpropJulia.evaluate_weight!(output, constant, 0.25, execution; work = Float64[])
        @test output == reshape(collect(1.0:6.0), 2, 3)
        @test _weight_allocations(output, constant, 0.25, execution, Float64[]) == 0
        table = _weight_table(execution, 6)
        direct = SolpropJulia.ObservationWeight(table, (2, 3))
        table.coefficients .= NaN
        for z in (-1.0, -0.1, 0.25, 0.7, 1.0)
            SolpropJulia.evaluate_weight!(output, direct, z, execution; work = Float64[])
            reference = reshape(_weight_reference(direct.data, z), 2, 3)
            # Two degree-2 terms plus the interval map require fewer than 32
            # rounded operations; bound by gamma_32 times this positive scale.
            bound = 32eps(Float64) / (1 - 32eps(Float64)) * maximum(abs, reference)
            @test bound > 0 && maximum(abs, output - reference) <= bound
        end
        @test _weight_allocations(output, direct, 0.25, execution, Float64[]) == 0
        coefficients = _weight_table(execution, 3)
        terms = reshape([1 + sin(index) / 4 for index in 1:18], 2, 3, 3)
        expansion = SolpropJulia.ObservationWeight(execution, terms, coefficients)
        terms .= NaN
        coefficients.breakpoints .= NaN
        scratch = zeros(3)
        for z in (-1.0, -0.1, 0.25, 0.7, 1.0)
            SolpropJulia.evaluate_weight!(output, expansion, z, execution; work = scratch)
            factors = _weight_reference(expansion.weights, z)
            reference = [
                sum(
                    BigFloat(expansion.data[row, column, term]) * factors[term]
                        for term in 1:3
                ) for row in 1:2, column in 1:3
            ]
            # Each polynomial contributes the previous gamma_32 bound; three
            # products and their sum are bounded conservatively by gamma_64.
            bound = 64eps(Float64) / (1 - 64eps(Float64)) * maximum(abs, reference)
            @test bound > 0 && maximum(abs, output - reference) <= bound
        end
        @test _weight_allocations(output, expansion, 0.25, execution, scratch) == 0
    end
end

@testset "Modal power weights refuse invalid or overlapping storage" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for value in (0.0, -1.0, Inf, NaN)
        @test_throws ArgumentError SolpropJulia.ObservationWeight(
            execution, fill(value, 2, 3)
        )
    end
    @test_throws ArgumentError SolpropJulia.ObservationWeight(
        execution, ones(Float32, 2, 3)
    )
    @test_throws ArgumentError SolpropJulia.ObservationWeight(execution, zeros(0, 3))
    table = _weight_table(execution, 6)
    @test_throws DimensionMismatch SolpropJulia.ObservationWeight(table, (2, 4))
    @test_throws DimensionMismatch SolpropJulia.ObservationWeight(
        execution, ones(2, 3, 5), table
    )
    @test_throws ArgumentError SolpropJulia.ObservationWeight(
        execution, ones(Float32, 2, 3, 6), table
    )
    constant = SolpropJulia.ObservationWeight(execution, ones(2, 3))
    retained = copy(constant.data)
    @test_throws ArgumentError SolpropJulia.evaluate_weight!(
        constant.data, constant, 0.0, execution; work = Float64[]
    )
    @test constant.data == retained
    direct = SolpropJulia.ObservationWeight(table, (2, 3))
    alias = reshape(@view(direct.data.coefficients[:, 1, 1]), 2, 3)
    retained = copy(alias)
    @test_throws ArgumentError SolpropJulia.evaluate_weight!(
        alias, direct, 0.0, execution; work = Float64[]
    )
    @test alias == retained
    output = fill(7.0, 2, 3)
    for z in (-1.1, 1.1, Inf, NaN)
        @test_throws ArgumentError SolpropJulia.evaluate_weight!(
            output, direct, z, execution; work = Float64[]
        )
        @test output == fill(7.0, 2, 3)
    end
    @test_throws DimensionMismatch SolpropJulia.evaluate_weight!(
        zeros(1, 3), constant, 0.0, execution; work = Float64[]
    )
    @test_throws ArgumentError SolpropJulia.evaluate_weight!(
        zeros(Float32, 2, 3), constant, 0.0, execution; work = Float64[]
    )
    expansion = SolpropJulia.ObservationWeight(
        execution, ones(2, 3, 3),
        _weight_table(execution, 3)
    )
    @test_throws DimensionMismatch SolpropJulia.evaluate_weight!(
        output, expansion, 0.0, execution; work = zeros(2)
    )
    scratch = @view(output[:, 1])
    @test_throws DimensionMismatch SolpropJulia.evaluate_weight!(
        output, expansion, 0.0, execution; work = scratch
    )
    shared = reshape(zeros(6), 2, 3)
    scratch = @view(vec(shared)[1:3])
    @test_throws ArgumentError SolpropJulia.evaluate_weight!(
        shared, expansion, 0.0, execution; work = scratch
    )
    alias = @view(expansion.weights.coefficients[:, 1, 1])
    retained = copy(alias)
    @test_throws ArgumentError SolpropJulia.evaluate_weight!(
        output, expansion, 0.0, execution; work = alias
    )
    @test alias == retained
    negative = SolpropJulia.ObservationWeight(
        execution, -ones(2, 3, 3),
        _weight_table(execution, 3)
    )
    @test_throws ArgumentError SolpropJulia.evaluate_weight!(
        output, negative, 0.0, execution; work = zeros(3)
    )
    retained = copy(expansion.weights.breakpoints)
    @test_throws ArgumentError SolpropJulia.evaluate_weight!(
        output, expansion, 0.0, execution; work = expansion.weights.breakpoints
    )
    @test expansion.weights.breakpoints == retained
    huge = SolpropJulia.ObservationWeight(
        execution, fill(floatmax(Float64), 2, 3, 3), _weight_table(execution, 3)
    )
    @test_throws ArgumentError SolpropJulia.evaluate_weight!(
        output, huge, 0.0, execution; work = zeros(3)
    )
    zero_table = SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], zeros(6, 1, 1))
    zero_weight = SolpropJulia.ObservationWeight(zero_table, (2, 3))
    @test_throws ArgumentError SolpropJulia.evaluate_weight!(
        output, zero_weight, 0.5, execution; work = Float64[]
    )
end
