# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

function source_precision_rate(::Type{T}, exponent::Int = 0) where {T}
    lower, upper, intervals = 0.25, 4.0, 4
    origin, spacing = log(lower), log(upper / lower) / intervals
    coefficients = zeros(T, 4, intervals)
    coefficients[3, :] .= 2
    coefficients[4, :] .= T.(2 .* (origin .+ (0:(intervals - 1)) .* spacing))
    return SP.IonisationRateTable(
        coefficients, ldexp(lower, exponent), ldexp(upper, exponent), upper^2,
        origin + exponent * log(2.0), spacing,
    )
end

function source_precision_cycle(::Type{T}; rtol = 2.0e-6) where {T}
    nodes, weights, ends = T[], T[], Int[]
    for order in (64, 128, 256, 512)
        # Golub--Welsch Jacobi matrix, as in the binary64 plasma witness.
        off = [index / sqrt(4index^2 - 1) for index in 1:(order - 1)]
        eigen = SP.LinearAlgebra.eigen(SP.LinearAlgebra.SymTridiagonal(zeros(order), off))
        append!(nodes, T.(eigen.values))
        append!(weights, T.(2 .* abs2.(eigen.vectors[1, :])))
        push!(ends, length(nodes))
    end
    budget = T(rtol)
    budget = budget > rtol ? prevfloat(budget) : budget
    return SP.CycleQuadrature(
        nodes, weights, Tuple(ends); moment_rtol = budget, negligible_rate = T(1.0e-30)
    )
end

function source_precision_field_allocations(first, second, samples, channel, rule, work, ex)
    SP.field_plasma!(
        first, second, samples, channel, rule, work, ex;
        density = 2.0, preionisation = 0.25, field_exponent = 3
    )
    return @allocated SP.field_plasma!(
        first, second, samples, channel, rule, work, ex;
        density = 2.0, preionisation = 0.25, field_exponent = 3
    )
end

function source_precision_population_allocations(samples, table, rule, work, ex)
    SP.field_populations!(
        samples, table, rule, work, ex;
        preionisation = 0.25, field_exponent = 3
    )
    return @allocated SP.field_populations!(
        samples, table, rule, work, ex;
        preionisation = 0.25, field_exponent = 3
    )
end

function source_precision_envelope_allocations(
        first, second, samples, channel, rule, work, ex,
    )
    SP.envelope_plasma!(
        first, second, samples, channel, rule, work, ex;
        density = 2.0, preionisation = 0.25, dt = 0.125, field_exponent = 3
    )
    return @allocated SP.envelope_plasma!(
        first, second, samples, channel, rule, work, ex;
        density = 2.0, preionisation = 0.25, dt = 0.125, field_exponent = 3
    )
end

@testset "Scaled rate clipping preserves representable threshold sides" begin
    lower, upper = 1 + 2.0^-24, 2 + 2.0^-23
    table = SP.IonisationRateTable(
        zeros(Float32, 4, 1), lower, upper, 2.0, log(lower), log(upper / lower)
    )
    parameters = SP._rate_parameters(table, 0)
    @test parameters[2:6] isa NTuple{5, Float32}
    @test parameters[7] isa SP._PlasmaLogCoordinate
    @test SP._ionisation_rate(1.0f0, parameters) === 0.0f0
    @test SP._ionisation_rate(nextfloat(1.0f0), parameters) === 1.0f0
    @test SP._ionisation_rate(2.0f0, parameters) === 1.0f0
    @test SP._ionisation_rate(nextfloat(2.0f0), parameters) === 2.0f0
end

function source_precision_envelope_population_allocations(samples, table, rule, work, ex)
    SP.envelope_populations!(
        samples, table, rule, work, ex;
        preionisation = 0.25, dt = 0.125, field_exponent = 3
    )
    return @allocated SP.envelope_populations!(
        samples, table, rule, work, ex;
        preionisation = 0.25, dt = 0.125, field_exponent = 3
    )
end

@testset "Binary32 prefix depth retains lost small increments" begin
    for count in (1, 2, 3, 31, 128, 65537)
        for polarity in (1.0f0, -1.0f0)
            input = fill(polarity, count, 1)
            input[1] = 2.0f0^24
            original = copy(input)
            reference = cumsum(Float64.(input); dims = 1)
            SP._plasma_prefix!(input, 1)
            depth = ceil(Int, log2(count))
            gamma = depth * eps(Float32) / (1 - depth * eps(Float32))
            # Each exact integer prefix fits binary64. Each binary32 prefix has
            # at most depth additions, independently of its number of terms.
            scale = cumsum(abs.(Float64.(original)); dims = 1)
            @test all(abs.(Float64.(input) - reference) .<= gamma .* scale)
            @test input[1] === original[1]
            count == 65537 && @test input[end] == Float32(reference[end])
        end
    end
    input = reshape(Float32[1.0e8, 1, -1.0e8, 2, 3, 4, -2], :, 1)
    reference = cumsum(Float64.(input); dims = 1)
    scale = cumsum(abs.(Float64.(input)); dims = 1)
    SP._plasma_prefix!(input, 1)
    @test all(abs.(Float64.(input) - reference) .<= 4eps(Float32) .* scale)
    zeros = fill(-0.0f0, 64, 1)
    SP._plasma_prefix!(zeros, 1)
    @test all(iszero, zeros)
end

@testset "Scaled binary32 field populations and current units" begin
    for tasks in (1, 2)
        ex = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        layout = SP.ResponseLayout(257, 1, 2)
        rule = SP.RunningIntegralRule(Float32[1], 1.0f0 / 256)
        table = source_precision_rate(Float32, 3)
        channel = SP.PlasmaChannel(table, 2.0, 3.0)
        work = SP.FieldPlasmaWorkspace(ex, Float32, layout)
        population = SP.PopulationWorkspace(ex, Float32, layout)
        samples = ones(Float32, layout.n_time, layout.n_nodes)
        samples[:, 2] .= NaN32
        first, second = similar(samples), similar(samples)
        SP.field_plasma!(
            first, second, samples, channel, rule, work, ex;
            density = 2.0, preionisation = 0.25, field_exponent = 3, count = 1
        )
        @test SP.plasma_failure(work.status).code == SP.PlasmaSuccess
        @test all(iszero, first[:, 2]) && all(iszero, second)
        @test all(iszero, work.rate[:, 2])
        exposure = (0:256) ./ 256
        fraction = 0.25 .+ 0.75 .* -expm1.(-exposure)
        survival = 0.75 .* exp.(-exposure)
        free = zeros(257)
        for row in 2:257
            free[row] = free[row - 1] + 4(fraction[row - 1] + fraction[row]) / 512
        end
        expected = free .+ (6 / 64) .* survival
        # The fixture's scaled log-polynomial is quadratic, with bounded log/exp
        # sensitivity on [1/4,4]. Budget 128u covers narrowing, log/exp, eight-level
        # prefixes and current products against this independent analytic exposure.
        gamma = 128eps(Float32) / (1 - 128eps(Float32))
        @test maximum(abs, Float64.(work.rate[:, 1]) .- 1) <= gamma
        @test maximum(abs, Float64.(work.exposure[:, 1]) .- exposure) <= gamma
        @test maximum(abs, Float64.(work.fraction[:, 1]) .- fraction) <= gamma
        @test maximum(abs, Float64.(work.survival[:, 1]) .- survival) <= gamma
        @test maximum(abs, Float64.(first[:, 1]) .- expected) <= gamma * maximum(expected)
        SP.field_populations!(
            samples, table, rule, population, ex;
            preionisation = 0.25, field_exponent = 3, count = 1
        )
        @test SP.plasma_failure(population.status).code == SP.PlasmaSuccess
        @test population.fraction == work.fraction
        @test population.rate == work.rate
        samples[:, 2] .= 1
        @test source_precision_field_allocations(
            first, second, samples, channel,
            rule, work, ex
        ) == 0
        @test source_precision_population_allocations(
            samples, table, rule,
            population, ex
        ) == 0
    end
end

@testset "Stage products avoid raw ionisation-scale underflow" begin
    ex = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    layout = SP.ResponseLayout(2, 1, 1)
    channel = SP.PlasmaChannel(source_precision_rate(Float32, 60), 2.0^-100, 2.0^-100)
    rule = SP.RunningIntegralRule(Float32[1], 0.25f0)
    work = SP.FieldPlasmaWorkspace(ex, Float32, layout)
    samples, first, second = ones(Float32, 2, 1), zeros(Float32, 2, 1), zeros(Float32, 2, 1)
    SP.field_plasma!(
        first, second, samples, channel, rule, work, ex;
        density = 2.0^120, preionisation = 0.0, field_exponent = 60
    )
    @test SP.plasma_failure(work.status).code == SP.PlasmaSuccess
    # Raw Float32(Ip)/2^120 is zero; the combined density*Ip/2^120 is 2^-100.
    @test first[1] > 0
    @test abs(Float64(first[1]) - 2.0^-100) <= 128eps(Float32) * 2.0^-100
    before = copy(first)
    @test_throws ArgumentError SP.field_plasma!(
        first, second, samples, channel, rule, work, ex;
        density = 1.0, preionisation = 0.0, field_exponent = 60
    )
    @test first == before
    @test_throws ArgumentError SP.field_populations!(
        samples, channel.rate, rule,
        SP.PopulationWorkspace(ex, Float32, layout), ex;
        preionisation = 0.0, field_exponent = 1023
    )
end

@testset "Binary32 survival retains small neutral and ionised fractions" begin
    ex = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    layout = SP.ResponseLayout(4, 1, 1)
    work = SP.PopulationWorkspace(ex, Float32, layout)
    table = source_precision_rate(Float32)
    samples = ones(Float32, 4, 1)
    for (spacing, initial) in ((1.0f-8, 0.0), (10.0f0, 0.0), (0.25f0, 1 - 2.0^-30))
        rule = SP.RunningIntegralRule(Float32[1], spacing)
        SP.field_populations!(samples, table, rule, work, ex; preionisation = initial)
        @test SP.plasma_failure(work.status).code == SP.PlasmaSuccess
        reference = (1 - initial) .* exp.(-Float64.(work.exposure))
        @test all(work.survival .> 0)
        @test maximum(abs.(Float64.(work.survival) - reference) ./ reference) <=
            4eps(Float32)
        @test initial != 0 || work.fraction[2] > 0
    end
end

@testset "Binary32 cycle criteria retain the translated default" begin
    ex = SP.KernelExecution(SP.KernelAbstractions.CPU(), 2, true)
    layout = SP.ResponseLayout(8, 1, 1)
    quadrature = source_precision_cycle(Float32)
    @test quadrature.moment_rtol <= 2.0e-6
    table = source_precision_rate(Float32, 3)
    samples = ones(ComplexF32, 8, 1)
    work = SP.EnvelopePlasmaWorkspace(ex, Float32, layout)
    population = SP.PopulationWorkspace(ex, Float32, layout)
    channel = SP.PlasmaChannel(table, 2.0, 3.0)
    first, second = similar(samples), similar(samples)
    SP.envelope_plasma!(
        first, second, samples, channel, quadrature, work, ex;
        density = 2.0, preionisation = 0.25, dt = 0.125, field_exponent = 3
    )
    @test SP.plasma_failure(work.status).code == SP.PlasmaSuccess
    cutoff = acos(0.25)
    cosine = (cutoff + sin(2cutoff) / 2) / pi
    exposure = (0:7) .* (cosine / 8)
    fraction = 0.25 .+ 0.75 .* -expm1.(-exposure)
    survival = 0.75 .* exp.(-exposure)
    # Positive compensated cycle sums, four-node spline Horner, scaled coordinates
    # and a three-level scan. The fixed 256u envelope includes the conditioned
    # binary32 elementary functions for this quadratic, away-from-cutoff fixture.
    bound = 256eps(Float32) / (1 - 256eps(Float32))
    @test maximum(abs, Float64.(work.rate) .- cosine) <= bound
    @test maximum(abs, Float64.(work.fraction[:, 1]) .- fraction) <= bound
    @test maximum(abs, Float64.(real.(first[:, 1])) .- 4fraction) <= 4bound
    expected_loss = (12 / 64) .* cosine .* survival
    @test maximum(abs, Float64.(real.(second[:, 1])) .- expected_loss) <= bound
    SP.envelope_populations!(
        samples, table, quadrature, population, ex;
        preionisation = 0.25, dt = 0.125, field_exponent = 3
    )
    @test SP.plasma_failure(population.status).code == SP.PlasmaSuccess
    @test population.rate == work.rate
    @test population.fraction == work.fraction
    @test source_precision_envelope_allocations(
        first, second, samples, channel,
        quadrature, work, ex
    ) == 0
    @test source_precision_envelope_population_allocations(
        samples, table,
        quadrature, population, ex
    ) == 0
end
