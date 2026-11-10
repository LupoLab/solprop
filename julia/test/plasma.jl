# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function _quadratic_rate_table()
    lower, upper, intervals = 0.25, 4.0, 4
    origin, spacing = log(lower), log(upper / lower) / intervals
    coefficients = zeros(4, intervals)
    coefficients[3, :] .= 2
    coefficients[4, :] .= 2 .* (origin .+ (0:(intervals - 1)) .* spacing)
    return SolpropJulia.IonisationRateTable(
        coefficients, lower, upper, upper^2, origin, spacing
    )
end

function _cycle_test_rules(orders; moment_rtol = 2.0e-6)
    nodes, weights, ends = Float64[], Float64[], Int[]
    for order in orders
        # The orthonormal Legendre recurrence has diagonal zero and off-diagonal
        # k/sqrt((2k-1)(2k+1)); its Jacobi eigenvalues are nodes and twice the
        # squared first eigenvector components are weights. This is the test-only
        # Golub-Welsch construction (1969, DOI: 10.1090/S0025-5718-69-99647-1).
        # Production receives the translated, validated CycleIntegralData rules.
        diagonal = zeros(order)
        off_diagonal = [index / sqrt(4index^2 - 1) for index in 1:(order - 1)]
        eigen = SolpropJulia.LinearAlgebra.eigen(
            SolpropJulia.LinearAlgebra.SymTridiagonal(diagonal, off_diagonal)
        )
        append!(nodes, eigen.values)
        append!(weights, 2 .* abs2.(eigen.vectors[1, :]))
        push!(ends, length(nodes))
    end
    return SolpropJulia.CycleQuadrature(
        nodes, weights, Tuple(ends); moment_rtol, negligible_rate = 1.0e-30
    )
end

function _field_plasma_allocations!(
        gated, edge, samples, channel, rule, workspace, execution,
    )
    SolpropJulia.field_plasma!(
        gated, edge, samples, channel, rule, workspace, execution;
        density = 2.0, preionisation = 0.25,
    )
    return @allocated SolpropJulia.field_plasma!(
        gated, edge, samples, channel, rule, workspace, execution;
        density = 2.0, preionisation = 0.25,
    )
end

function _envelope_plasma_allocations!(
        drive, loss, samples, channel, quadrature, workspace, execution,
    )
    SolpropJulia.envelope_plasma!(
        drive, loss, samples, channel, quadrature, workspace, execution;
        density = 2.0, preionisation = 0.25, dt = 0.125,
    )
    return @allocated SolpropJulia.envelope_plasma!(
        drive, loss, samples, channel, quadrature, workspace, execution;
        density = 2.0, preionisation = 0.25, dt = 0.125,
    )
end

@testset "Field plasma analytic populations, currents and independent channels" begin
    for tasks in (1, 2)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        layout = SolpropJulia.ResponseLayout(8, 1, 2)
        workspace = SolpropJulia.FieldPlasmaWorkspace(execution, Float64, layout)
        channel = SolpropJulia.PlasmaChannel(_quadratic_rate_table(), 2.0, 3.0)
        rule = SolpropJulia.RunningIntegralRule([1.0], 0.125)
        samples, gated, edge = ones(8, 2), zeros(8, 2), zeros(8, 2)
        derivative = fill(0.5, 8)
        SolpropJulia.field_plasma!(
            gated, edge, samples, channel, rule, workspace, execution;
            density = 2.0, preionisation = 0.25, derivative,
        )
        failure = SolpropJulia.plasma_failure(workspace.status)
        @test failure.code == SolpropJulia.PlasmaSuccess
        @test workspace.rate == ones(8, 2)
        @test workspace.exposure[:, 1] == (0:7) .* 0.125
        fraction = [0.25 + 0.75 * -expm1(-BigFloat(index) / 8) for index in 0:7]
        neutral = [0.75 * exp(-BigFloat(index) / 8) for index in 0:7]
        current = zeros(BigFloat, 8)
        for row in 2:8
            current[row] = current[row - 1] + (fraction[row - 1] + fraction[row]) / 8
        end
        reference = Float64.(2 .* (current .+ 3 .* neutral))
        # Eight prefixes, elementary exponentials and ordinary products: gamma_128
        # times the sum of positive current/loss contributions bounds this witness.
        bound = 128eps(Float64) / (1 - 128eps(Float64)) * maximum(reference)
        @test maximum(abs, gated[:, 1] - reference) <= bound
        @test maximum(abs, workspace.fraction[:, 1] - Float64.(fraction)) <= 8eps(Float64)
        @test edge == 0.5 .* gated
        @test workspace.fraction[1, 1] == 0.25
        @test all(0.25 .<= workspace.fraction .<= 1)
        # Each atom pays its ionisation potential; no second density is hidden in
        # the per-atom free current or the independently retained survival.
        loss = gated ./ 2 .- workspace.current
        work = samples .* loss .- 3 .* workspace.rate .* workspace.survival
        @test maximum(abs, work) <= bound
        one_channel = copy(gated)
        SolpropJulia.field_plasma!(
            gated, edge, samples, channel, rule, workspace, execution;
            density = 5.0, preionisation = 0.25, add = true,
        )
        @test maximum(abs, gated - 3.5 .* one_channel) <= 4bound
        @test _field_plasma_allocations!(
            gated, edge, samples, channel, rule, workspace, execution
        ) == 0
    end
end

@testset "Plasma clipping, point reflection and reused scratch" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    layout = SolpropJulia.ResponseLayout(4, 1, 2)
    workspace = SolpropJulia.FieldPlasmaWorkspace(execution, Float64, layout)
    channel = SolpropJulia.PlasmaChannel(_quadratic_rate_table(), 1.0, 1.0)
    rule = SolpropJulia.RunningIntegralRule([1.0], 1.0)
    samples = [0.0 NaN; 1.0 NaN; 0.0 NaN; 0.0 NaN]
    gated, edge = fill(19.0, 4, 2), fill(23.0, 4, 2)
    for array in (
            workspace.rate, workspace.exposure, workspace.fraction, workspace.survival,
            workspace.integrand, workspace.current, workspace.filtered,
        )
        fill!(array, 17)
    end
    for _ in 1:3
        SolpropJulia.field_plasma!(
            gated, edge, samples, channel, rule, workspace, execution;
            density = 1.0, preionisation = 0.0, count = 1,
        )
        # Exact dyadic witness for #622: repeated calls must not read a stale
        # filtered endpoint after the live rate's support.
        @test workspace.exposure[:, 1] == [0.0, 0.5, 1.0, 1.0]
        for array in (
                workspace.rate, workspace.exposure, workspace.fraction, workspace.survival,
                workspace.integrand, workspace.current, workspace.filtered, gated, edge,
            )
            @test all(iszero, array[:, 2])
            @test all(isfinite, array)
        end
        failure = SolpropJulia.plasma_failure(workspace.status)
        @test failure.code == SolpropJulia.PlasmaSuccess
    end
    input = reshape(collect(0.0:5.0), 6, 1)
    filtered, integrated = fill(21.0, 6, 1), fill(22.0, 6, 1)
    SolpropJulia._integrate_plasma_column!(
        integrated, filtered, input, 1, [1.1, -0.05], 0.5
    )
    # Point reflection preserves a linear input; its trapezoid is exact. Account
    # for the binary64 tap coefficients and six rounded prefix increments.
    @test maximum(abs, integrated[:, 1] - (0:5) .^ 2 ./ 4) <= 64eps(Float64) * 25 / 4

    values = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, -8.0]
    parameters = SolpropJulia._rate_parameters(channel.rate)
    rates = [SolpropJulia._ionisation_rate(abs(value), parameters) for value in values]
    @test rates[1:2] == [0.0, 0.0]
    @test rates[7:8] == [16.0, 16.0]
    @test maximum(abs, rates[3:6] - values[3:6] .^ 2) <= 32eps(Float64) * 4
end

@testset "Plasma statuses include finite-channel accumulation overflow" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    layout = SolpropJulia.ResponseLayout(2, 1, 1)
    workspace = SolpropJulia.FieldPlasmaWorkspace(execution, Float64, layout)
    channel = SolpropJulia.PlasmaChannel(_quadratic_rate_table(), 1.0, 1.0)
    rule = SolpropJulia.RunningIntegralRule([1.0], 1.0)
    samples, gated, edge = ones(2, 1), zeros(2, 1), zeros(2, 1)
    samples[2] = NaN
    SolpropJulia.field_plasma!(
        gated, edge, samples, channel, rule, workspace, execution;
        density = 1.0, preionisation = 0.0,
    )
    failure = SolpropJulia.plasma_failure(workspace.status)
    @test failure.code == SolpropJulia.PlasmaNonfiniteInput
    @test (failure.node, failure.sample) == (1, 2)
    @test all(iszero, gated)
    fill!(samples, 1)
    fill!(gated, floatmax(Float64))
    SolpropJulia.field_plasma!(
        gated, edge, samples, channel, rule, workspace, execution;
        density = 1.0e308, preionisation = 1.0, add = true,
    )
    failure = SolpropJulia.plasma_failure(workspace.status)
    @test failure.code == SolpropJulia.PlasmaNonfiniteResult
    @test failure.sample == 2
    @test all(isfinite, gated)
    @test_throws ArgumentError SolpropJulia.FieldPlasmaWorkspace(
        execution, Float64, SolpropJulia.ResponseLayout(2, 2, 1)
    )
    @test_throws ArgumentError SolpropJulia.field_plasma!(
        gated, edge, samples, channel, rule, workspace, execution;
        density = 1.0, preionisation = 1.5,
    )
    @test_throws ArgumentError SolpropJulia.RunningIntegralRule([NaN], 1.0)
    @test_throws ArgumentError SolpropJulia.PlasmaChannel(channel.rate, -1.0, 1.0)
end

@testset "Envelope cycle moments against analytic clipped quadratic rates" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    table = _quadratic_rate_table()
    quadrature = _cycle_test_rules((64, 128, 256, 512))
    parameters = SolpropJulia._rate_parameters(table)
    rules = SolpropJulia._cycle_parameters(quadrature)
    for major in (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 5.0)
        cosine, sine, code = SolpropJulia._scalar_cycle_moments(major, parameters, rules)
        @test code == SolpropJulia.PlasmaSuccess
        if major <= table.field_min
            @test iszero(cosine) && iszero(sine)
        else
            lower = acos(min(1, table.field_max / major))
            upper = acos(table.field_min / major)
            cosine_reference = 2 / pi * (
                table.rate_max * lower + major^2 *
                    ((upper - lower) / 2 + (sin(2upper) - sin(2lower)) / 4)
            )
            sine_reference = 2 / pi * (
                table.rate_max * (tan(lower) - lower) + major^2 *
                    ((upper - lower) / 2 - (sin(2upper) - sin(2lower)) / 4)
            )
            # This is an accuracy check against closed antiderivatives, using the
            # descriptor's fixed 2e-6 moment budget rather than a measured tolerance.
            @test abs(cosine - cosine_reference) <= 2.0e-6 * cosine_reference
            @test abs(sine - sine_reference) <= 2.0e-6 * sine_reference
        end
    end
    layout = SolpropJulia.ResponseLayout(4, 1, 2)
    workspace = SolpropJulia.EnvelopePlasmaWorkspace(execution, Float64, layout)
    channel = SolpropJulia.PlasmaChannel(table, 2.0, 3.0)
    samples = fill(ComplexF64(0.6, 0.8), 4, 2)
    drive, loss = similar(samples), similar(samples)
    SolpropJulia.envelope_plasma!(
        drive, loss, samples, channel, quadrature, workspace, execution;
        density = 2.0, preionisation = 0.25, dt = 0.125,
    )
    @test SolpropJulia.plasma_failure(workspace.status).code == SolpropJulia.PlasmaSuccess
    @test all(workspace.fraction[1, :] .== 0.25)
    @test all(0.25 .<= workspace.fraction .<= 1)
    # Time-averaged work is Re(conj(A)*J_loss)/2 for the complex envelope.
    work = real.(conj.(samples) .* loss) ./ 2
    expected = 3 .* 2 .* workspace.survival .* workspace.rate
    @test maximum(abs, work - expected) <= 32eps(Float64) * maximum(expected)
    @test _envelope_plasma_allocations!(
        drive, loss, samples, channel, quadrature, workspace, execution
    ) == 0
    samples[:, 2] .= ComplexF64(NaN, NaN)
    SolpropJulia.envelope_plasma!(
        drive, loss, samples, channel, quadrature, workspace, execution;
        density = 2.0, preionisation = 0.25, dt = 0.125, count = 1,
    )
    @test all(iszero, drive[:, 2]) && all(iszero, loss[:, 2])
    @test all(iszero, workspace.rate[:, 2]) && all(iszero, workspace.fraction[:, 2])

    inadequate = _cycle_test_rules((2, 4); moment_rtol = 1.0e-30)
    SolpropJulia.envelope_plasma!(
        drive, loss, samples, channel, inadequate, workspace, execution;
        density = 2.0, preionisation = 0.25, dt = 0.125, count = 1,
    )
    failure = SolpropJulia.plasma_failure(workspace.status)
    @test failure.code == SolpropJulia.PlasmaQuadratureFailure
    @test all(isfinite, drive) && all(isfinite, loss)
    @test all(iszero, drive)
    fill!(samples, 1)
    fill!(drive, complex(floatmax(Float64)))
    SolpropJulia.envelope_plasma!(
        drive, loss, samples, channel, quadrature, workspace, execution;
        density = 5.0e307, preionisation = 1.0, dt = 0.125, add = true,
    )
    failure = SolpropJulia.plasma_failure(workspace.status)
    @test failure.code == SolpropJulia.PlasmaNonfiniteResult
    @test all(isfinite, drive)
    @test_throws ArgumentError SolpropJulia.EnvelopePlasmaWorkspace(
        execution, Float64, SolpropJulia.ResponseLayout(2, 2, 1)
    )
end
