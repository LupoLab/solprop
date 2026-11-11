# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

function derived_spatial_allocated(output, slot, grid, input, work, execution)
    SolpropJulia.spatial_statistics!(output, slot, grid, input, work, execution)
    return @allocated SolpropJulia.spatial_statistics!(
        output, slot, grid, input, work, execution
    )
end

function derived_monitor_allocated(reading, strip, current, execution)
    SolpropJulia.transverse_strip!(reading, (strip,), execution; amplitude_exponent = 1100)
    SolpropJulia.transverse_current!(reading, (current,), 1.0; amplitude_exponent = 1100)
    first = @allocated SolpropJulia.transverse_strip!(
        reading, (strip,), execution; amplitude_exponent = 1100
    )
    second = @allocated SolpropJulia.transverse_current!(
        reading, (current,), 1.0; amplitude_exponent = 1100
    )
    return first, second
end

@testset "Derived observation metadata does not materialize its power" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    samples = fill(complex(ldexp(1.0f0, -100)), 1, 1)
    grid = SolpropJulia.StatisticsGrid([1.0], [0.0], [1.0])
    input = SolpropJulia.SpatialStatisticsInput(;
        analytic = samples, analytic_closure = copy(samples), factor = 1.0,
        source_columns = 1, field_exponent = 100, closure_exponent = 1100
    )
    work = SolpropJulia.StatisticsWorkspace(execution, Float32; n_time = 1, n_columns = 1)
    slot, output = SolpropJulia.StatisticSlot(:peak_intensity, 1, 1), zeros(1)
    SolpropJulia.spatial_statistics!(output, slot, grid, input, work, execution)
    expected = ldexp(1.0, 1000)
    # Exact dyadic factors need no input/transform error. Four scalar products,
    # pair addition and host restoration fit this conservative 16-rounding budget.
    tolerance = 16 * Float64(eps(Float32))
    @test tolerance > 0
    @test isapprox(output[1], expected; rtol = tolerance, atol = 0.0)
    @test derived_spatial_allocated(output, slot, grid, input, work, execution) == 0
    for exponent in (-4 * 1074, 4 * 1074)
        @test SolpropJulia._observation_derived_exponents((exponent,)) === nothing
    end
    for exponent in (typemin(Int), typemax(Int), -4 * 1074 - 1, 4 * 1074 + 1)
        invalid = SolpropJulia.SpatialStatisticsInput(;
            analytic = samples, analytic_closure = copy(samples), factor = 1.0,
            source_columns = 1, field_exponent = 100, closure_exponent = exponent
        )
        before = copy(output)
        @test_throws ArgumentError SolpropJulia.spatial_statistics!(
            output, slot, grid, invalid, work, execution
        )
        @test output == before
    end
    @test_throws ArgumentError SolpropJulia.SourceArithmetic(Float32; field_exponent = 1100)
    @test_throws ArgumentError SolpropJulia._observation_scale_exponents((1100,))
    overflow = SolpropJulia.SpatialStatisticsInput(;
        analytic = samples, analytic_closure = copy(samples), factor = 1.0,
        source_columns = 1, field_exponent = 100, closure_exponent = 1200
    )
    before = copy(output)
    @test_throws ArgumentError SolpropJulia.spatial_statistics!(
        output, slot, grid, overflow, work, execution
    )
    @test output == before

    spectrum = zeros(ComplexF32, 4, 1)
    spectrum[2] = complex(ldexp(1.0f0, -100))
    measure = ldexp(1.0, -1000)
    strip = SolpropJulia.TransverseStripLine(spectrum, 1.0, measure, 2, 2)
    current_samples = reshape(
        ComplexF32[0, ldexp(1.0f0, -100), im * ldexp(1.0f0, -100)], 3, 1
    )
    current = SolpropJulia.TransverseCurrentLine(current_samples, 2, 1, 3, 1.0, measure)
    reading = SolpropJulia.TransverseReading(1, 1)
    SolpropJulia.transverse_strip!(reading, (strip,), execution; amplitude_exponent = 1100)
    SolpropJulia.transverse_current!(reading, (current,), 1.0; amplitude_exponent = 1100)
    @test all(isapprox.(reading.strip, expected / 4; rtol = tolerance, atol = 0.0))
    @test isapprox(reading.current, expected; rtol = tolerance, atol = 0.0)
    @test derived_monitor_allocated(reading, strip, current, execution) == (0, 0)
    weak_samples = reshape(ComplexF32[0, ldexp(1.0f0, 100), im * ldexp(1.0f0, 100)], 3, 1)
    weak_current = SolpropJulia.TransverseCurrentLine(
        weak_samples, 2, 1, 3, 1.0, ldexp(1.0, 1000)
    )
    SolpropJulia.transverse_current!(
        reading, (weak_current,), 1.0; amplitude_exponent = -1100
    )
    @test isapprox(reading.current, ldexp(1.0, -1000); rtol = tolerance, atol = 0.0)
    for exponent in (typemin(Int), typemax(Int), 4 * 1074 + 1)
        before, value = copy(reading.strip), reading.current
        @test_throws ArgumentError SolpropJulia.transverse_strip!(
            reading, (strip,), execution; amplitude_exponent = exponent
        )
        @test_throws ArgumentError SolpropJulia.transverse_current!(
            reading, (current,), 1.0; amplitude_exponent = exponent
        )
        @test reading.strip == before && reading.current == value
    end
end
