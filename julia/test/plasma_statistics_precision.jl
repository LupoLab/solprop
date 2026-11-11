# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

@testset "Binary32 diagnostic parts normalize each independent family" begin
    for tasks in (1, 2), scale in (-140, 0, 120)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        population = SolpropJulia.PopulationWorkspace(
            execution, Float32, SolpropJulia.ResponseLayout(4, 1, 2)
        )
        # Synthetic successful arrays isolate reduction arithmetic and its full
        # admitted finite range; no physical recurrence or material is inferred.
        samples = reshape(ldexp.(Float32[1, -2, 3, -4, 2, -1, 4, -3], scale), 4, 2)
        population.rate .= reshape(ldexp.(Float32[1, 2, 3, 4, 4, 3, 2, 1], scale), 4, 2)
        population.exposure .= reshape(
            ldexp.(Float32[0, -1, 1, 2, 0, 1, 2, 3], scale), 4, 2
        )
        population.fraction .= Float32[0.125 0.25; 0.25 0.5; 0.5 0.625; 0.75 0.875]
        rule = SolpropJulia.RunningIntegralRule(Float32[1.25, -0.125], 0.125f0)
        work = SolpropJulia.PlasmaDiagnosticWorkspace(execution, Float32, 2)
        case = (; samples, population, rule, work)
        reference = plasma_statistics_reference(samples, population, rule)
        SolpropJulia.plasma_diagnostics!(work, samples, population, rule, execution)
        # Normalizations are exact powers except gradual underflow. Each part
        # has <=3 pointwise products, a three-term Sum2s and one subtraction.
        # Use an absolute paid-plus-freed scale for the potentially cancelled
        # defect. Family-specific scales avoid hiding a small diagnostic behind
        # the magnitude of another part.
        u = big(2.0)^-24
        exposure_scale = maximum(abs, BigFloat.(population.exposure))
        field_scale = maximum(abs, BigFloat.(samples))
        paid_scale = 4BigFloat(rule.dt) * maximum(BigFloat.(population.rate)) + 1
        bounds = 64u .* BigFloat[
            exposure_scale, exposure_scale, paid_scale, 1, field_scale, field_scale,
        ]
        @test all(bounds .> 0)
        @test all(abs.(BigFloat.(work.maxima) - reference) .<= bounds)
        @test eltype(work.parts) === Float32
        @test eltype(work.maxima) === Float64
        @test size(work.scales) == (3, 2)
        @test plasma_statistics_allocated(case, execution) == 0
        SolpropJulia.reset_plasma_diagnostics!(work)
        SolpropJulia.plasma_diagnostics!(
            work, samples, population, rule, execution; count = 1
        )
        @test all(iszero, work.parts[:, 2]) && all(iszero, work.scales[:, 2])
    end
end

@testset "Binary32 paid work and tiny boundary products retain separate scales" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    population = SolpropJulia.PopulationWorkspace(
        execution, Float32, SolpropJulia.ResponseLayout(4, 1, 1)
    )
    samples = fill(2.0f0^-140, 4, 1)
    population.rate .= 2.0f0^120
    population.exposure[:, 1] .= Float32[0, -2.0f0^127, 2.0f0^127, 0]
    population.fraction[:, 1] .= ldexp.(Float32[1, 2, 3, 4], -140)
    rule = SolpropJulia.RunningIntegralRule(Float32[1.25, -0.125], 2.0f0^120)
    work = SolpropJulia.PlasmaDiagnosticWorkspace(execution, Float32, 1)
    SolpropJulia.plasma_diagnostics!(work, samples, population, rule, execution)
    reference = plasma_statistics_reference(samples, population, rule)
    @test all(
        abs.(BigFloat.(work.maxima) - reference) .<=
            32big(2.0)^-24 .* abs.(reference)
    )
    @test work.maxima[2] > floatmax(Float32)
    @test work.maxima[3] > floatmax(Float32)
    @test 0 < work.maxima[4] < floatmin(Float32)
    @test 0 < work.maxima[5] <= work.maxima[6] < Float64(nextfloat(0.0f0))
    values = SolpropJulia.plasma_diagnostic_values(work)
    @test isapprox(values[3], 0.5; rtol = 8eps(Float32), atol = 0.0)
    retained = copy(work.maxima)
    population.exposure[2, 1] = -Inf32
    @test_throws ArgumentError SolpropJulia.plasma_diagnostics!(
        work, samples, population, rule, execution
    )
    @test work.maxima == retained
end
