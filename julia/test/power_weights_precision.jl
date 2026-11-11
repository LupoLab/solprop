# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

function precision_weight_allocations(output, weight, position, execution, work)
    SolpropJulia.evaluate_weight!(output, weight, position, execution; work)
    return @allocated SolpropJulia.evaluate_weight!(
        output, weight, position, execution; work
    )
end

@testset "Normalized power weights keep the wide representation until final rounding" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    arithmetic = SolpropJulia.SourceArithmetic(Float32)
    for exponent in (-600, 0, 600)
        values = ldexp.(reshape([1.0, 1.25, 1.5, 1.75], 2, 2), exponent)
        original = SolpropJulia.ObservationWeight(execution, values)
        weight = SolpropJulia.NormalizedObservationWeight(execution, original, arithmetic)
        work = SolpropJulia.ObservationWeightWorkspace(weight)
        output = fill(NaN32, 2, 2)
        scale = SolpropJulia.evaluate_weight!(output, weight, 0.0, execution; work)
        @test scale == exponent
        @test work.range.block_exponent == exponent && work.range.normalized
        @test Float64.(output) .* big(2.0)^scale == BigFloat.(values)
        @test eltype(output) === Float32 && isempty(work.host)
        @test !work.identity
        @test precision_weight_allocations(output, weight, 0.0, execution, work) == 0
        @test output !== weight.constant
    end
    unit = SolpropJulia.NormalizedObservationWeight(
        execution,
        SolpropJulia.ObservationWeight(execution, ones(2, 1)), arithmetic
    )
    work = SolpropJulia.ObservationWeightWorkspace(unit)
    output = zeros(Float32, 2, 1)
    @test SolpropJulia.evaluate_weight!(output, unit, 0.0, execution; work) == 0
    @test work.identity && all(isone, output)

    # At x=1 these represented coefficients cancel to256 and512. Narrowing the
    # coefficients before Clenshaw would produce zero and violate SEP33's host rule.
    coefficients = zeros(2, 2, 1)
    coefficients[:, 1, 1] .= 2.0^60
    coefficients[:, 2, 1] .= [-2.0^60 + 256, -2.0^60 + 512]
    table = SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], coefficients)
    direct = SolpropJulia.ObservationWeight(table, (2, 1))
    normalized = SolpropJulia.NormalizedObservationWeight(execution, direct, arithmetic)
    work = SolpropJulia.ObservationWeightWorkspace(normalized)
    for position in (0.0, 0.125, 0.5, 1.0)
        reference = zeros(2, 1)
        SolpropJulia.evaluate_weight!(
            reference, direct, position, execution; work = Float64[]
        )
        scale = SolpropJulia.evaluate_weight!(output, normalized, position, execution; work)
        @test output == Float32.(ldexp.(reference, -scale))
        @test all(
            abs.(BigFloat.(output) .* big(2.0)^scale - BigFloat.(reference)) .<=
                2.0^-24 .* abs.(BigFloat.(reference))
        )
        @test precision_weight_allocations(
            output, normalized, position, execution, work
        ) == 0
    end
    @test output == reshape(Float32[1, 2], 2, 1) ./ 2
    @test work.range.block_exponent == 9
    @test !work.identity

    terms = reshape([2.0^60, 2.0^60, -2.0^60 + 256, -2.0^60 + 512], 2, 1, 2)
    scalar_weights = SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], ones(2, 1, 1))
    expansion = SolpropJulia.ObservationWeight(execution, terms, scalar_weights)
    normalized_expansion = SolpropJulia.NormalizedObservationWeight(
        execution, expansion, arithmetic
    )
    expansion_work = SolpropJulia.ObservationWeightWorkspace(normalized_expansion)
    exponent = SolpropJulia.evaluate_weight!(
        output, normalized_expansion, 0.5, execution;
        work = expansion_work
    )
    @test Float64.(output) .* 2.0^exponent == reshape([256.0, 512.0], 2, 1)
    @test expansion_work.host == ones(2)
    @test precision_weight_allocations(
        output, normalized_expansion, 0.5, execution, expansion_work
    ) == 0
end

@testset "Normalized power-weight refusal preserves output and published facts" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    arithmetic = SolpropJulia.SourceArithmetic(Float32)
    coefficients = reshape([1.0, -2.0], 1, 2, 1)
    table = SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], coefficients)
    original = SolpropJulia.ObservationWeight(table, (1, 1))
    weight = SolpropJulia.NormalizedObservationWeight(execution, original, arithmetic)
    work = SolpropJulia.ObservationWeightWorkspace(weight)
    output = fill(17.0f0, 1, 1)
    SolpropJulia.evaluate_weight!(output, weight, 0.0, execution; work)
    previous, fact = copy(output), work.range
    for position in (1.0, 1.5, NaN)
        @test_throws ArgumentError SolpropJulia.evaluate_weight!(
            output, weight, position, execution; work
        )
        @test output == previous && work.range === fact
    end
    @test_throws DimensionMismatch SolpropJulia.evaluate_weight!(
        zeros(Float32, 2, 1), weight, 0.0, execution; work
    )
    @test_throws ArgumentError SolpropJulia.evaluate_weight!(
        zeros(1, 1), weight, 0.0, execution; work
    )
    @test_throws ArgumentError SolpropJulia.NormalizedObservationWeight(
        execution,
        SolpropJulia.ObservationWeight(execution, reshape([1.0e300, 1.0e-300], 2, 1)),
        arithmetic
    )
    constant = SolpropJulia.NormalizedObservationWeight(
        execution,
        SolpropJulia.ObservationWeight(execution, ones(2, 1)), arithmetic
    )
    constant_work = SolpropJulia.ObservationWeightWorkspace(constant)
    retained = copy(constant.constant)
    @test_throws ArgumentError SolpropJulia.evaluate_weight!(
        constant.constant, constant, 0.0, execution;
        work = constant_work
    )
    @test constant.constant == retained
end
