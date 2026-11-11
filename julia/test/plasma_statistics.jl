# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function plasma_statistics_case(execution; taps = [1.25, -0.125], initial = 0.25)
    samples = [
        0.0 0.5 0.0; 0.0 -0.2 0.0; 1.0 0.8 0.0; -0.7 0.3 0.0;
        0.2 0.0 0.0; 0.0 -0.1 0.0; 0.0 0.2 0.0; 0.0 0.0 0.0;
    ]
    low, high = 1.0e-10, 10.0
    table = SolpropJulia.IonisationRateTable(
        reshape([0.0, 0.0, 2.0, 2log(low)], 4, 1),
        low, high, high^2, log(low), log(high / low),
    )
    rule = SolpropJulia.RunningIntegralRule(taps, 0.125; floor_exposure = true)
    population = SolpropJulia.PopulationWorkspace(
        execution, Float64, SolpropJulia.ResponseLayout(8, 1, 3),
    )
    SolpropJulia.field_populations!(
        samples, table, rule, population, execution;
        preionisation = initial
    )
    work = SolpropJulia.PlasmaDiagnosticWorkspace(execution, Float64, 3)
    return (; samples, table, rule, population, work)
end

function plasma_statistics_reference(samples, population, rule)
    result = zeros(BigFloat, 6)
    reach = min(size(samples, 1), 2(length(rule.taps) - 1))
    for node in axes(samples, 2)
        rate = BigFloat.(population.rate[:, node])
        fraction = BigFloat.(population.fraction[:, node])
        exposure = BigFloat.(population.exposure[:, node])
        increments = diff(exposure)
        integrand = abs.(fraction .* BigFloat.(samples[:, node]))
        work = rate .* (1 .- fraction)
        paid = sum(work[1:(end - 1)] .+ work[2:end]) * BigFloat(rule.dt) / 2
        freed = fraction[end] - fraction[1]
        result = max.(
            result, [
                max(-minimum(increments), 0), max(maximum(increments), 0),
                abs(paid - freed), max(freed, 0),
                reach == 0 ? big"0" : maximum(integrand[1:reach]), maximum(integrand),
            ]
        )
    end
    return result
end

function plasma_statistics_allocated(case, execution)
    SolpropJulia.reset_plasma_diagnostics!(case.work)
    SolpropJulia.plasma_diagnostics!(
        case.work, case.samples, case.population,
        case.rule, execution
    )
    SolpropJulia.plasma_diagnostic_values(case.work)
    return @allocated begin
        SolpropJulia.reset_plasma_diagnostics!(case.work)
        SolpropJulia.plasma_diagnostics!(
            case.work, case.samples, case.population,
            case.rule, execution
        )
        SolpropJulia.plasma_diagnostic_values(case.work)
    end
end

@testset "Plasma diagnostics use raw exposure and global maxima" begin
    for tasks in (1, 2), taps in ([1.0], [1.25, -0.125]), initial in (0.0, 0.25, 1.0)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(),
            tasks, true
        )
        case = plasma_statistics_case(execution; taps, initial)
        reference = plasma_statistics_reference(case.samples, case.population, case.rule)
        SolpropJulia.plasma_diagnostics!(
            case.work, case.samples, case.population,
            case.rule, execution
        )
        # The supplied rounded arrays define this reduction witness. Each paid sum
        # has seven additions and <= three products per interval; subtraction can
        # cancel, so gamma_48 multiplies a positive absolute scale, not the defect.
        gamma = 48eps(Float64) / (1 - 48eps(Float64))
        scale = max(
            1.0, maximum(abs, case.population.rate) * case.rule.dt * 8,
            maximum(abs, case.samples), maximum(abs, case.population.exposure)
        )
        bound = gamma * scale
        @test bound > 0
        @test maximum(abs, BigFloat.(case.work.maxima) - reference) <= bound
        expected = ntuple(3) do index
            denominator = reference[2index]
            denominator > 0 ? reference[2index - 1] / denominator : big"0"
        end
        values = SolpropJulia.plasma_diagnostic_values(case.work)
        for index in 1:3
            denominator = Float64(reference[2index])
            # Include numerator and denominator errors and the final division.
            allowance = denominator > bound ?
                (bound * (1 + abs(Float64(expected[index])))) / (denominator - bound) +
                2eps(Float64) * abs(Float64(expected[index])) : bound
            @test abs(BigFloat(values[index]) - expected[index]) <= allowance
        end
        if length(taps) == 1
            @test iszero(values[3])
        end
        @test plasma_statistics_allocated(case, execution) == 0
        fill!(case.work.parts, NaN)
        SolpropJulia.reset_plasma_diagnostics!(case.work)
        @test all(iszero, case.work.parts) && all(iszero, case.work.maxima)
        SolpropJulia.plasma_diagnostics!(
            case.work, case.samples, case.population,
            case.rule, execution; count = 1
        )
        @test all(iszero, case.work.parts[:, 2:3])
        retained = copy(case.work.maxima)
        SolpropJulia.plasma_diagnostics!(
            case.work, case.samples, case.population,
            case.rule, execution; count = 0
        )
        @test case.work.maxima == retained
    end
end

@testset "Diagnostic failures preserve aggregate maxima and observed arrays" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    case = plasma_statistics_case(execution)
    @test_throws ArgumentError SolpropJulia.PlasmaDiagnosticWorkspace(execution, Float16, 1)
    @test_throws ArgumentError SolpropJulia.PlasmaDiagnosticWorkspace(execution, Float64, 0)
    @test_throws DimensionMismatch SolpropJulia.plasma_diagnostics!(
        case.work, case.samples,
        case.population, case.rule, execution; count = 4
    )
    case.work.maxima .= 0.125
    retained = copy(case.work.maxima)
    case.population.status[3, 2] = UInt8(SolpropJulia.PlasmaNonfiniteInput)
    @test_throws ArgumentError SolpropJulia.plasma_diagnostics!(
        case.work, case.samples,
        case.population, case.rule, execution
    )
    @test case.work.maxima == retained
    case.population.status[3, 2] = UInt8(SolpropJulia.PlasmaSuccess)
    case.population.exposure[3, 2] = Inf
    @test_throws ArgumentError SolpropJulia.plasma_diagnostics!(
        case.work, case.samples,
        case.population, case.rule, execution
    )
    @test case.work.maxima == retained
    case = plasma_statistics_case(execution)
    original = copy(case.population.rate)
    owner = case.population.rate
    GC.@preserve owner begin
        foreign = unsafe_wrap(Array, pointer(owner, 2), (6, 3); own = false)
        alias = SolpropJulia.PlasmaDiagnosticWorkspace(foreign, zeros(6))
        @test_throws ArgumentError SolpropJulia.plasma_diagnostics!(
            alias, case.samples,
            case.population, case.rule, execution
        )
        @test owner == original
    end
    case.work.maxima .= (1.0, nextfloat(0.0), 0.0, 0.0, 0.0, 0.0)
    @test_throws ArgumentError SolpropJulia.plasma_diagnostic_values(case.work)
    case.work.maxima .= 0
    @test SolpropJulia.plasma_diagnostic_values(case.work) == (0.0, 0.0, 0.0)
end
