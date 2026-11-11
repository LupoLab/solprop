# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

@testset "Binary32 completion rows keep physical scale outside products" begin
    for tasks in (1, 2)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        tiny = nextfloat(0.0f0)
        fractions = Float32[tiny 0.25 0.0; 2tiny 0.5 0.0; 3tiny 0.75 0.0]
        densities = [2.0^700, 2.0^-700, 3.0]
        rates = Float32[1 4 0; 2 3 0; 3 2 0]
        weights, potentials, measure = [1.0e-200, 2.0e-200, 3.0e-200],
            [1.0e-200, 1.0e200, 0.0], 1.0e200
        input = SolpropJulia.PopulationStatisticsInput(
            densities, fractions, rates; potentials, node_weights = weights,
            measure_factor = measure
        )
        work = SolpropJulia.StatisticsWorkspace(
            execution, Float32; n_time = 4, n_columns = 2, n_nodes = 3
        )
        output = zeros(3)
        for (name, reference) in (
                (:species_density, BigFloat.(densities)),
                (:species_peak_ionisation_rate, BigFloat.(vec(maximum(rates; dims = 1)))),
                (
                    :species_ionisation_energy,
                    [
                        BigFloat(densities[species]) * potentials[species] * measure *
                            sum(
                            BigFloat(weights[node]) * fractions[node, species]
                                for node in 1:3
                        )
                            for species in 1:3
                    ],
                ),
                (
                    :species_electron_density,
                    BigFloat.(densities) .* BigFloat.(fractions[3, :]),
                ),
            )
            slot = SolpropJulia.StatisticSlot(name, 1, 3)
            SolpropJulia.population_statistics!(output, slot, input, work, execution)
            # Two normalized inputs, one product and a three-term Sum2s, followed
            # by binary64 mantissa/exponent restoration. The 32u positive-sum
            # budget also encloses the independently normalized species products.
            bound = 32 * 2.0^-24
            @test bound > 0
            @test all(abs.(BigFloat.(output) - reference) .<= bound .* abs.(reference))
            name == :species_electron_density && @test all(>(0), output[1:2])
        end
        axis_input = SolpropJulia.PopulationStatisticsInput(
            densities, fractions[1:1, :], rates[1:1, :]
        )
        SolpropJulia.axis_population_statistics!(
            output, SolpropJulia.StatisticSlot(:on_axis_species_electron_density, 1, 3),
            axis_input, work, execution
        )
        expected = BigFloat.(densities) .* BigFloat.(fractions[1, :])
        @test all(abs.(BigFloat.(output) - expected) .<= 16 * 2.0^-24 .* abs.(expected))
        @test all(>(0), output[1:2])

        spectrum = ComplexF32[2.0f0^120 im * 2.0f0^119; 2.0f0^-120 im * 2.0f0^-119]
        grid = SolpropJulia.StatisticsGrid(
            [1.0, 2.0], [0.0, 1.0], ones(2); spectral_density = [1.0e-10, 1.0e10]
        )
        for exponent in (-200, 200)
            spectral = SolpropJulia.SpectralStatisticsInput(
                spectrum, fill(0.5f0, size(spectrum)); amplitude_exponent = exponent,
                weight_exponent = 1, identity_weight = true
            )
            slot = SolpropJulia.StatisticSlot(:spectral_flux, 1, 2)
            SolpropJulia.spectral_statistics!(output, slot, grid, spectral, work, execution)
            exact = [
                sum(abs2, Complex{BigFloat}.(spectrum[row, :])) *
                    grid.spectral_density[row] * big(2.0)^(2exponent) for row in 1:2
            ]
            @test all(abs.(BigFloat.(output[1:2]) - exact) .<= 16 * 2.0^-24 .* abs.(exact))
            @test all(>(0), output[1:2])
        end
        invalid = SolpropJulia.SpectralStatisticsInput(
            spectrum, ones(Float32, size(spectrum)); amplitude_exponent = typemax(Int)
        )
        @test_throws ArgumentError SolpropJulia.spectral_statistics!(
            output, SolpropJulia.StatisticSlot(:spectral_flux, 1, 2),
            grid, invalid, work, execution
        )
    end
end

function precision_statistics_selection(entries)
    offset = 1
    slots = map(entries) do (name, count)
        slot = SolpropJulia.StatisticSlot(name, offset, count)
        offset += count
        slot
    end
    return SolpropJulia.StatisticsSelection(slots)
end

function precision_statistics_allocations(output, selection, grid, inputs, work, execution)
    SolpropJulia.statistics!(output, selection, grid, inputs, work, execution)
    return @allocated SolpropJulia.statistics!(
        output, selection, grid, inputs, work, execution
    )
end

@testset "Binary32 spectra retain independent column scales" begin
    for tasks in (1, 2), exponent in (-200, 0, 200)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        spectrum = ComplexF32[
            2.0f0^120 2.0f0^-120 0
            (1 + im) * 2.0f0^120 (1 + im) * 2.0f0^-120 0
            2.0f0^121 2.0f0^-119 0
            0 0 0
        ]
        weight = fill(2.0f0, size(spectrum))
        grid = SolpropJulia.StatisticsGrid(
            [1.0e15, 1.5e15, 2.0e15, 2.5e15], [0.0, 1.0], fill(1.0e-17, 4);
            windows = Bool[1 0; 1 0; 0 1; 0 1],
        )
        selection = precision_statistics_selection(
            ((:omega0, 3), (:energy, 3), (:energy_window, 6)),
        )
        work = SolpropJulia.StatisticsWorkspace(
            execution, Float32; n_time = 2, n_columns = 3
        )
        input = SolpropJulia.SpectralStatisticsInput(
            spectrum, weight; amplitude_exponent = exponent
        )
        inputs = SolpropJulia.StatisticsInputs(; spectral = input)
        output = zeros(selection.width)
        SolpropJulia.statistics!(output, selection, grid, inputs, work, execution)
        density = abs2.(Complex{BigFloat}.(spectrum)) .* BigFloat.(weight)
        for column in 1:2
            centroid = sum(density[:, column] .* grid.omega) / sum(density[:, column])
            energy = sum(density[:, column] .* grid.parseval) * big(2.0)^(2exponent)
            # At most three normalized pointwise products, two coordinate/input
            # conversions and Sum2s (n=4) bound numerator/denominator separately.
            # 32u encloses their positive-sum ratio and final binary64 restoration.
            bound = 32 * 2.0^-24
            @test bound > 0
            @test abs(BigFloat(output[column]) - centroid) <= bound * abs(centroid)
            @test abs(BigFloat(output[column + 3]) - energy) <= bound * abs(energy)
            for window in 1:2
                expected = sum(
                    density[row, column] * grid.parseval[row] for row in 1:4
                        if grid.windows[row, window]
                ) * big(2.0)^(2exponent)
                actual = output[6 + (window - 1) * 3 + column]
                @test abs(BigFloat(actual) - expected) <= bound * abs(expected)
            end
        end
        @test isnan(output[3])
        @test output[[6, 9, 12]] == zeros(3)
        @test work.scales.columns[1] != work.scales.columns[2]
        @test precision_statistics_allocations(
            output, selection, grid, inputs, work, execution
        ) == 0
        unit = SolpropJulia.SpectralStatisticsInput(spectrum, ones(Float32, size(weight)))
        SolpropJulia.spectral_statistics!(
            output, selection.slots[1], grid, unit, work, execution
        )
        @test isnan(output[2])
        @test_throws ArgumentError SolpropJulia.spectral_statistics!(
            zeros(Float32, selection.width), selection.slots[2],
            grid, input, work, execution,
        )
    end
end

@testset "Binary32 signed temporal pairs and conditioned widths" begin
    for tasks in (1, 2), amplitude in (-60, 60)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        first = ComplexF32[0 0; 1 0.5; 2 1; 1 0.5; 0 0] .* 2.0f0^amplitude
        second = copy(first)
        second[:, 2] .*= -0.25f0
        window_first = cat(first, first ./ 2.0f0; dims = 3)
        window_second = cat(second, second .* 2.0f0; dims = 3)
        input = SolpropJulia.TemporalStatisticsInput(
            first, second; window_analytic = window_first, window_weighted = window_second,
            axis_analytic = first, axis_weighted = second, intensity_factor = 0.125,
            amplitude_exponent = -amplitude + 20, weight_exponent = 3,
            axis_exponent = -amplitude + 10,
        )
        grid = SolpropJulia.StatisticsGrid(
            [1.0], collect(-2.0:2.0), [1.0]; windows = fill(true, 1, 2)
        )
        selection = precision_statistics_selection(
            (
                (:peak_power, 2), (:peak_power_total, 1), (:peak_power_window, 4),
                (:fwhm_t, 1), (:fwhm_t_outer, 1), (:peak_intensity, 1),
            ),
        )
        work = SolpropJulia.StatisticsWorkspace(
            execution, Float32; n_time = 5, n_columns = 2
        )
        inputs = SolpropJulia.StatisticsInputs(; temporal = input)
        output = zeros(selection.width)
        SolpropJulia.statistics!(output, selection, grid, inputs, work, execution)
        expected = [
            4 * 2.0^43, 0, 3.75 * 2.0^43, 4 * 2.0^43, 0, 4 * 2.0^43, 0,
            4 / 3, 4 / 3, 0.125 * 3.75 * 2.0^23,
        ]
        # Integer/power-of-two pair arithmetic is exact here. Each width crossing
        # has slope magnitude 2.8125 and peak 3.75: two final divisions/subtractions
        # in binary64 suffice; the positive 16u64 bound is an output allowance.
        @test isapprox(output, expected; rtol = 16eps(Float64), atol = 0.0)
        @test minimum(work.power) < 0
        @test eltype(work.power) === eltype(work.total_power) === Float32
        @test precision_statistics_allocations(
            output, selection, grid, inputs, work, execution
        ) == 0
        @test all(isfinite, output)
    end
end

@testset "Binary32 population densities restore species scales" begin
    for tasks in (1, 2), density_scale in (-600, 600)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        densities = ldexp.([1.0, 0.25, 2.0^-300], density_scale)
        fractions = Float32[0.25 1 0.25; 0.5 0.25 1; 0 0 0]
        peaks = Float32[2 3 7; 9 4 8; 0 0 0]
        input = SolpropJulia.PopulationStatisticsInput(
            densities, fractions, peaks; peak_species = 2
        )
        selection = precision_statistics_selection(
            (
                (:species_ionisation, 9), (:species_electron_density, 3),
                (:electron_density, 1), (:peak_ionisation_rate, 1),
            ),
        )
        grid = SolpropJulia.StatisticsGrid([1.0], [0.0], [1.0])
        work = SolpropJulia.StatisticsWorkspace(
            execution, Float32; n_time = 1, n_columns = 1, n_nodes = 3,
        )
        inputs = SolpropJulia.StatisticsInputs(; populations = input)
        output = zeros(selection.width)
        SolpropJulia.statistics!(output, selection, grid, inputs, work, execution)
        expected_species = BigFloat.(densities) .* BigFloat.(fractions[2, :])
        expected_total = sum(expected_species)
        @test output[1:9] == Float64.(vec(fractions))
        # Binary32 products and Sum2s use normalized operands. The dilute species
        # is below the total's rounding scale but its independently restored slot
        # remains nonzero. Four working-precision roundings bound these sums.
        @test all(
            abs.(BigFloat.(output[10:12]) - expected_species) .<=
                4 * 2.0^-24 .* abs.(expected_species)
        )
        @test abs(BigFloat(output[13]) - expected_total) <= 4 * 2.0^-24 * expected_total
        @test output[12] > 0
        @test output[14] == 3.0
        @test eltype(work.node_total) === Float32
        @test precision_statistics_allocations(
            output, selection, grid, inputs, work, execution
        ) == 0
    end
end

function precision_spatial_case(kind, exponent, tasks)
    execution = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), tasks, true,
    )
    x = kind == :radial ? collect(0.25:0.25:2.0) : collect(-2.0:0.25:2.0)
    y = kind == :radial ? Float64[] : kind == :line ? [0.0] : copy(x)
    quadrature = kind == :radial ? x .* 0.25 : Float64[]
    widths = SolpropJulia.SpatialWidthGrid(kind, x, y, quadrature)
    # A compact quadratic intensity has an analytically conditioned half-maximum
    # crossing and no exponentially small tail. Field values are represented once.
    shape = kind == :radial ? max.(0.0, 1 .- x .^ 2 ./ 4) :
        vec([max(0.0, 1 - (a^2 + b^2) / 4) for a in x, b in y])
    samples = reshape(ComplexF32.(sqrt.(shape)), 1, :) .* 2.0f0^exponent
    axis = fill(ComplexF32(2.0f0^exponent), 1, 1)
    input = SolpropJulia.SpatialStatisticsInput(;
        spectrum = samples, closure = copy(samples), axis_spectrum = axis,
        axis_closure = copy(axis), analytic = repeat(samples, 2),
        analytic_closure = repeat(samples, 2), axis_analytic = repeat(axis, 2),
        axis_analytic_closure = repeat(axis, 2), factor = 0.125,
        source_columns = length(shape), widths,
        field_exponent = -exponent + 20, closure_exponent = -exponent - 10,
    )
    grid = SolpropJulia.StatisticsGrid(
        [1.0], [0.0, 1.0], [1.0]; spectral_density = [2.0], d_omega = 0.5,
    )
    selection = precision_statistics_selection(
        (
            (:beam_radius, 2), (:fwhm_r, 1), (:on_axis_spectrum, 1),
            (:peak_intensity, 1), (:on_axis_intensity, 1),
        ),
    )
    work = SolpropJulia.StatisticsWorkspace(
        execution, Float32; n_time = 2, n_columns = length(shape),
        n_nodes = length(shape), n_radii = widths.radius_capacity,
    )
    inputs = SolpropJulia.StatisticsInputs(; spatial = input)
    output = zeros(selection.width)
    SolpropJulia.statistics!(output, selection, grid, inputs, work, execution)
    return (; output, inputs, input, selection, work, grid, execution, shape)
end

@testset "Binary32 spatial widths and physical pair homogeneity" begin
    for kind in (:radial, :line, :cartesian), tasks in (1, 2)
        low = precision_spatial_case(kind, -100, tasks)
        high = precision_spatial_case(kind, 100, tasks)
        @test all(isfinite, low.output)
        @test high.output == low.output
        @test low.output[4] == 256.0
        @test low.output[6] == 128.0
        expected_peak = BigFloat(128) * maximum(
            abs2, Complex{BigFloat}.(low.input.analytic) ./ big(2.0)^-100
        )
        @test abs(BigFloat(low.output[5]) - expected_peak) <= 4 * 2.0^-24 * expected_peak
        # Compare the same discrete observable with binary64 arithmetic on the
        # represented Float32 fields. Positive sums have condition number one;
        # 64u covers products, normalized moments/centroids and interpolation.
        # The quadratic crossing slope is at least1/2 on both selected brackets,
        # giving an additional factor8 in the width error propagation.
        input = low.input
        reference_input = SolpropJulia.SpatialStatisticsInput(;
            spectrum = ComplexF64.(input.spectrum) .* 2.0^100,
            closure = ComplexF64.(input.closure) .* 2.0^100,
            axis_spectrum = ComplexF64.(input.axis_spectrum) .* 2.0^100,
            axis_closure = ComplexF64.(input.axis_closure) .* 2.0^100,
            factor = 1.0, source_columns = input.source_columns, widths = input.widths,
        )
        reference_work = SolpropJulia.StatisticsWorkspace(
            low.execution, Float64; n_time = 2, n_columns = length(low.shape),
            n_nodes = length(low.shape), n_radii = input.widths.radius_capacity,
        )
        reference = zeros(3)
        for slot in low.selection.slots[1:2]
            SolpropJulia.spatial_statistics!(
                reference, slot, low.grid, reference_input,
                reference_work, low.execution
            )
        end
        bound = 512 * 2.0^-24
        @test bound > 0
        @test isapprox(low.output[1:3], reference; rtol = bound, atol = bound)
        @test precision_statistics_allocations(
            low.output, low.selection, low.grid,
            low.inputs, low.work, low.execution
        ) == 0
        @test eltype(low.work.fluence) === eltype(low.work.radial_average) === Float32
    end
end
