# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function _statistics_allocations(output, selection, grid, inputs, workspace, execution)
    SolpropJulia.statistics!(output, selection, grid, inputs, workspace, execution)
    return @allocated SolpropJulia.statistics!(
        output, selection, grid, inputs, workspace, execution
    )
end

function _statistics_selection(entries; periodic = ())
    offset = 1
    slots = map(entries) do (name, count)
        slot = SolpropJulia.StatisticSlot(name, offset, count)
        offset += count
        slot
    end
    return SolpropJulia.StatisticsSelection(slots, periodic)
end

@testset "Statistic catalogue, prepared storage and missing inputs" begin
    slot = SolpropJulia.StatisticSlot(:energy, 1, 2)
    selection = SolpropJulia.StatisticsSelection((slot,), (1,))
    @test selection.width == 2
    @test SolpropJulia.statistics_needs(selection).spectral
    @test !SolpropJulia.statistics_needs(selection).temporal
    @test_throws ArgumentError SolpropJulia.StatisticSlot(:absorbed_power, 1, 1)
    @test_throws ArgumentError SolpropJulia.StatisticSlot(:species_density, 1, 1)
    @test_throws ArgumentError SolpropJulia.StatisticSlot(:energy, 0, 1)
    @test_throws ArgumentError SolpropJulia.StatisticSlot(:energy, 1, 0)
    @test_throws ArgumentError SolpropJulia.StatisticSlot(:energy, big(typemax(Int)) + 1, 1)
    @test_throws ArgumentError SolpropJulia.StatisticsSelection((slot, slot))
    @test_throws ArgumentError SolpropJulia.StatisticsSelection((slot,), (2,))
    @test_throws ArgumentError SolpropJulia.StatisticsSelection((slot,), (1, 1))
    @test_throws ArgumentError SolpropJulia.StatisticsSelection(
        (SolpropJulia.StatisticSlot(:beam_radius, 1, 2),), (1,)
    )
    @test_throws ArgumentError SolpropJulia.StatisticsSelection(
        (SolpropJulia.StatisticSlot(:energy, 1, typemax(Int)),)
    )
    @test_throws ArgumentError SolpropJulia.StatisticsGrid([1.0], [0.0, 0.0], [1.0])
    @test_throws DimensionMismatch SolpropJulia.StatisticsGrid([1.0], [0.0], [1.0, 1.0])
    @test_throws ArgumentError SolpropJulia.StatisticsGrid([1.0], [0.0], [-1.0])
    @test_throws ArgumentError SolpropJulia.StatisticsGrid(
        [1.0], [0.0], [1.0]; windows = trues(1, 1)
    )
    @test_throws ArgumentError SolpropJulia.TemporalStatisticsInput(
        ones(1), ones(1);
        intensity_factor = -1.0
    )
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    grid = SolpropJulia.StatisticsGrid([1.0], [0.0], [1.0])
    workspace = SolpropJulia.StatisticsWorkspace(
        execution, Float64; n_time = 1, n_columns = 2
    )
    @test_throws ArgumentError SolpropJulia.StatisticsWorkspace(
        execution, Float64;
        n_time = 0, n_columns = 2
    )
    @test_throws ArgumentError SolpropJulia.statistics!(
        zeros(2), selection, grid,
        SolpropJulia.StatisticsInputs(), workspace, execution
    )
    @test_throws DimensionMismatch SolpropJulia.statistics!(
        zeros(1), selection, grid,
        SolpropJulia.StatisticsInputs(), workspace, execution
    )
end

@testset "Spectral energy, raw centroid and hard windows" begin
    for tasks in (1, 2)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        omega = [0.0, 2.0, 4.0, 6.0, 8.0]
        spectrum = reshape(complex.(sin.(1:15), cos.(2:16)), 5, 3)
        weights = reshape(0.25 .+ abs.(cos.(1:15)), 5, 3)
        windows = fill(false, 5, 2)
        windows[1:2, 1] .= true
        windows[3:4, 2] .= true
        parseval = [1.0, 4.0, 4.0, 4.0, 1.0] .* (0.125 / 8)
        grid = SolpropJulia.StatisticsGrid(
            omega, collect(0.0:0.125:0.875), parseval;
            windows
        )
        input = SolpropJulia.SpectralStatisticsInput(spectrum, weights)
        selection = _statistics_selection(((:omega0, 3), (:energy, 3), (:energy_window, 6)))
        workspace = SolpropJulia.StatisticsWorkspace(
            execution, Float64;
            n_time = 8, n_columns = 3
        )
        output = zeros(selection.width)
        inputs = SolpropJulia.StatisticsInputs(; spectral = input)
        SolpropJulia.statistics!(output, selection, grid, inputs, workspace, execution)
        density = abs2.(Complex{BigFloat}.(spectrum)) .* BigFloat.(weights)
        centroid = vec(sum(density .* omega; dims = 1) ./ sum(density; dims = 1))
        energy = vec(sum(density .* parseval; dims = 1))
        windowed = [
            sum(
                density[row, column] * parseval[row] for row in 1:5
                    if windows[row, window]
            ) for column in 1:3, window in 1:2
        ]
        reference = Float64.([centroid; energy; vec(windowed)])
        # Five rows, <= six scalar products/sums each, plus centroid division.
        # gamma_64 bounds each positive reduction and denominator perturbation.
        bound = 64eps(Float64) / (1 - 64eps(Float64)) * maximum(abs, reference)
        @test maximum(abs, output - reference) <= bound
        @test _statistics_allocations(
            output, selection, grid, inputs,
            workspace, execution
        ) == 0
        spectrum[:, 2] .= 0
        SolpropJulia.statistics!(output, selection, grid, inputs, workspace, execution)
        @test isnan(output[2])
        @test iszero(output[5])
        # The endpoint factors belong to energy, not the centroid.
        unit = SolpropJulia.SpectralStatisticsInput(ones(ComplexF64, 5, 3), ones(5, 3))
        total_slot = SolpropJulia.StatisticSlot(:physical_energy, 1, 1)
        SolpropJulia.spectral_statistics!(
            output, total_slot, grid, unit, workspace, execution
        )
        @test output[1] == 3sum(parseval)
        @test_throws ArgumentError SolpropJulia.spectral_statistics!(
            output, total_slot,
            grid, input, workspace, execution
        )
        @test_throws ArgumentError SolpropJulia.spectral_statistics!(
            output,
            SolpropJulia.StatisticSlot(:density, 1, 1), grid, unit, workspace, execution
        )
        spectrum[1] = Inf
        @test_throws ArgumentError SolpropJulia.statistics!(
            output, selection, grid,
            inputs, workspace, execution
        )
    end
end

@testset "Signed temporal pairing and analytic scalar widths" begin
    for tasks in (1, 2)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        time = collect(-2.0:0.5:2.0)
        pulse = [0.0, 0.0, 0.25, 0.75, 1.0, 0.75, 0.25, 0.0, 0.0]
        analytic = complex.(hcat(pulse, pulse ./ 2))
        magnetic = complex.(hcat(pulse .* 2, -pulse))
        window_first = cat(analytic, analytic ./ 2; dims = 3)
        window_second = cat(magnetic, magnetic .* 2; dims = 3)
        temporal = SolpropJulia.TemporalStatisticsInput(
            analytic, magnetic;
            window_analytic = window_first, window_weighted = window_second,
            axis_analytic = analytic[:, 1:1], axis_weighted = magnetic[:, 1:1],
            intensity_factor = 0.25
        )
        selection = _statistics_selection(
            (
                (:peak_power, 2), (:peak_power_window, 4),
                (:peak_power_total, 1), (:fwhm_t, 1),
                (:fwhm_t_outer, 1), (:peak_intensity, 1),
            )
        )
        grid = SolpropJulia.StatisticsGrid([1.0], time, [1.0]; windows = fill(true, 1, 2))
        workspace = SolpropJulia.StatisticsWorkspace(
            execution, Float64;
            n_time = 9, n_columns = 2
        )
        output = zeros(selection.width)
        inputs = SolpropJulia.StatisticsInputs(; temporal)
        SolpropJulia.statistics!(output, selection, grid, inputs, workspace, execution)
        @test output[1:7] == [2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.5]
        # Squared samples cross half of the peak between t=.5 and t=1.
        @test output[8] == 1.125
        @test output[9] == output[8]
        @test output[10] == 0.5
        @test minimum(workspace.power) < 0
        @test _statistics_allocations(
            output, selection, grid, inputs,
            workspace, execution
        ) == 0
        alias = SolpropJulia.StatisticSlot(:analytic_flux, 1, 1)
        @test_throws ArgumentError SolpropJulia.temporal_statistics!(
            output, alias, grid,
            temporal, workspace, execution
        )
        missing = SolpropJulia.TemporalStatisticsInput(analytic, nothing)
        @test_throws ArgumentError SolpropJulia.temporal_statistics!(
            output, alias, grid,
            missing, workspace, execution
        )
    end
    time = collect(0.0:8.0)
    trace = [0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.5, 0.0]
    @test SolpropJulia.half_width(time, trace) == 1.0
    # Two line crossings and the subtraction incur at most eight roundings.
    @test abs(SolpropJulia.half_width(time, trace; outer = true) - 23 / 6) <=
        8eps(Float64) * (23 / 6)
    @test isnan(SolpropJulia.half_width(time, zeros(9)))
    @test isnan(SolpropJulia.half_width(time, ones(9)))
    @test isnan(SolpropJulia.half_width(time, -ones(9)))
    @test_throws ArgumentError SolpropJulia.half_width([0.0, 0.0], [0.0, 1.0])
end

@testset "Poynting time pairing is not a square-root weighted envelope" begin
    # Two tones with distinct admittances have cross term (w1+w2), while the
    # incorrect squared sqrt-weight field has 2sqrt(w1*w2). At phase pi their
    # actual signed instantaneous flux is negative, a direct physical witness.
    angle = [0.0, pi]
    electric = reshape(1 .+ 0.5 .* cis.(angle), 2, 1)
    magnetic = reshape(1 .+ 4.5 .* cis.(angle), 2, 1)
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    grid = SolpropJulia.StatisticsGrid([1.0, 2.0], angle, ones(2))
    workspace = SolpropJulia.StatisticsWorkspace(
        execution, Float64; n_time = 2, n_columns = 1
    )
    output = zeros(1)
    input = SolpropJulia.TemporalStatisticsInput(electric, magnetic)
    SolpropJulia.temporal_statistics!(
        output, SolpropJulia.StatisticSlot(:peak_power, 1, 1),
        grid, input, workspace, execution
    )
    # Binary inputs at phase zero are exact; the phase-pi residual contributes
    # quadratically, far below an 8-operation absolute roundoff budget.
    @test output[1] == 8.25
    @test abs(workspace.power[2] + 1.75) <= 8eps(Float64) * 8.25
    @test abs2(1 - 1.5) != workspace.power[2]
end

@testset "Population ordering, peak rule, profiles and periodic slices" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    fractions = [0.5 0.0 0.25; 0.25 0.0 0.5; 0.75 0.0 0.0]
    peaks = [1.0 0.0 4.0; 3.0 0.0 2.0; 2.0 0.0 1.0]
    populations = SolpropJulia.PopulationStatisticsInput([2.0, 3.0, 2.0], fractions, peaks)
    selection = _statistics_selection(
        (
            (:species_ionisation, 9),
            (:species_electron_density, 3), (:electron_density, 1),
            (:peak_ionisation_rate, 1),
            (:density, 1), (:pressure, 1), (:core_radius, 1), (:quadrature_error, 1),
        );
        periodic = (3, 5, 8)
    )
    inputs = SolpropJulia.StatisticsInputs(;
        populations,
        profiles = (7.0, 1.25, 0.003), quadrature_error = 0.125
    )
    grid = SolpropJulia.StatisticsGrid([1.0], [0.0], [1.0])
    workspace = SolpropJulia.StatisticsWorkspace(
        execution, Float64;
        n_time = 1, n_columns = 1, n_nodes = 3
    )
    output = zeros(selection.width)
    SolpropJulia.statistics!(output, selection, grid, inputs, workspace, execution)
    @test output == [vec(fractions); 1.0; 0.0; 0.5; 1.5; 4.0; 7.0; 1.25; 0.003; 0.125]
    @test _statistics_allocations(
        output, selection, grid, inputs, workspace, execution
    ) == 0
    fill!(workspace.node_total, 99.0)
    SolpropJulia.statistics!(
        output, selection, grid, inputs, workspace, execution;
        periodic = true
    )
    @test all(iszero, output[1:12])
    @test output[13] == 1.5 && output[15] == 7.0 && output[18] == 0.125
    @test all(iszero, output[[14, 16, 17]])
    @test SolpropJulia.statistics_needs(selection; periodic = true).populations
    # Legacy single-population peak follows its fraction, even at zero density.
    single = SolpropJulia.PopulationStatisticsInput(
        zeros(3), fractions, peaks; peak_species = 1
    )
    slot = SolpropJulia.StatisticSlot(:peak_ionisation_rate, 1, 1)
    SolpropJulia.population_statistics!(output, slot, single, workspace, execution)
    @test output[1] == 2.0
    fractions[1] = NaN
    @test_throws ArgumentError SolpropJulia.population_statistics!(
        output, slot,
        single, workspace, execution
    )
end

@testset "Unavailable quadrature alarms retain only their diagnostic NaN" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    selection = _statistics_selection(((:quadrature_error, 1),); periodic = (1,))
    grid = SolpropJulia.StatisticsGrid([1.0], [0.0], [1.0])
    workspace = SolpropJulia.StatisticsWorkspace(
        execution, Float64; n_time = 1, n_columns = 1
    )
    output = zeros(1)
    for periodic in (false, true)
        # The launch, absent embedded rule and zero material source are unmeasured,
        # not a zero quadrature error. Issue #653 preserves this catalogue contract.
        SolpropJulia.statistics!(
            output, selection, grid, SolpropJulia.StatisticsInputs(),
            workspace, execution; periodic
        )
        @test isnan(only(output))
        for score in (0.0, 0.125, floatmax(Float64))
            inputs = SolpropJulia.StatisticsInputs(; quadrature_error = score)
            SolpropJulia.statistics!(
                output, selection, grid, inputs, workspace, execution; periodic
            )
            @test only(output) == score
        end
        for invalid in (-1.0, -Inf, Inf)
            inputs = SolpropJulia.StatisticsInputs(; quadrature_error = invalid)
            @test_throws ArgumentError SolpropJulia.statistics!(
                output, selection, grid, inputs, workspace, execution; periodic
            )
        end
    end
    @test _statistics_allocations(
        output, selection, grid, SolpropJulia.StatisticsInputs(), workspace, execution
    ) == 0
    for invalid in (NaN, -Inf, Inf)
        inputs = SolpropJulia.StatisticsInputs(; profiles = (invalid, 0.0, 0.0))
        selected = _statistics_selection(((:density, 1), (:quadrature_error, 1)))
        @test_throws ArgumentError SolpropJulia.statistics!(
            zeros(2), selected, grid, inputs, workspace, execution
        )
        spectral = SolpropJulia.SpectralStatisticsInput(
            fill(complex(invalid), 1, 1), ones(1, 1)
        )
        selected = _statistics_selection(((:energy, 1), (:quadrature_error, 1)))
        @test_throws ArgumentError SolpropJulia.statistics!(
            zeros(2), selected, grid, SolpropJulia.StatisticsInputs(; spectral),
            workspace, execution
        )
    end
end

function _statistics_spatial(widths, fluence, axis; tasks = 2)
    execution = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), tasks, true
    )
    grid = SolpropJulia.StatisticsGrid(
        [1.0], [0.0, 1.0], [1.0];
        spectral_density = [2.0], d_omega = 0.5
    )
    spectrum = reshape(complex.(sqrt.(fluence)), 1, :)
    input = SolpropJulia.SpatialStatisticsInput(;
        spectrum, closure = copy(spectrum),
        axis_spectrum = fill(ComplexF64(sqrt(axis)), 1, 1),
        axis_closure = fill(ComplexF64(sqrt(axis)), 1, 1), factor = 1.0,
        source_columns = length(fluence), widths
    )
    workspace = SolpropJulia.StatisticsWorkspace(
        execution, Float64;
        n_time = 2, n_columns = length(fluence), n_nodes = length(fluence),
        n_radii = widths.radius_capacity
    )
    selection = _statistics_selection(
        ((:beam_radius, 2), (:fwhm_r, 1), (:on_axis_spectrum, 1))
    )
    output = zeros(selection.width)
    inputs = SolpropJulia.StatisticsInputs(; spatial = input)
    SolpropJulia.statistics!(output, selection, grid, inputs, workspace, execution)
    return (; output, selection, grid, input, inputs, workspace, execution)
end

@testset "Spatial radial, Cartesian and line Gaussian observables" begin
    errors = zeros(3, 3)
    for (level, count) in pairs((32, 64, 128))
        radius = collect(range(0.0, 4.0; length = count + 1))[2:end]
        widths = SolpropJulia.SpatialWidthGrid(
            :radial, radius, Float64[],
            2pi .* radius .* (4 / count)
        )
        result = _statistics_spatial(widths, exp.(-2 .* radius .^ 2), 1.0)
        errors[level, :] = abs.(result.output[1:3] - [1.0, 1.0, sqrt(2log(2.0))])
        # Radial rectangle quadrature and linear crossings are second order.
        # A bound 8h² covers the Gaussian's bounded derivatives on this interval.
        @test maximum(errors[level, :]) < 8 * (4 / count)^2
        @test _statistics_allocations(
            result.output, result.selection, result.grid,
            result.inputs, result.workspace, result.execution
        ) == 0
    end
    # The 1/e² radius is exactly a node on all three grids: it is already
    # at roundoff, so only the moment and half-maximum crossing have truncation.
    @test all(errors[:, 2] .<= 64eps(Float64))
    @test all(errors[3, [1, 3]] .< errors[1, [1, 3]] ./ 4)
    for kind in (:cartesian, :line)
        errors = zeros(3)
        for (level, count) in pairs((32, 64, 128))
            x = collect(range(-3.0, 3.0; length = count + 1))
            y = kind == :line ? [0.0] : x
            widths = SolpropJulia.SpatialWidthGrid(kind, x, y, Float64[])
            fluence = [
                exp(-2 * ((horizontal - 0.17)^2 + (vertical + 0.09)^2))
                    for horizontal in x, vertical in y
            ]
            result = _statistics_spatial(widths, vec(fluence), 1.0)
            errors[level] = maximum(abs, result.output[1:3] - [1.0, 1.0, sqrt(2log(2.0))])
            # Bilinear/radial interpolation is second order. The finite-domain
            # Gaussian tail adds <1e-6 to these centred second moments.
            @test errors[level] < 8 * (6 / count)^2 + 1.0e-6
            @test _statistics_allocations(
                result.output, result.selection, result.grid,
                result.inputs, result.workspace, result.execution
            ) == 0
        end
        @test errors[3] < errors[1] / 4
    end
end

@testset "Spatial undefined widths, signed flux and reused scratch" begin
    widths = SolpropJulia.SpatialWidthGrid(:radial, [0.5, 1.0, 1.5], Float64[], ones(3))
    zero_result = _statistics_spatial(widths, zeros(3), 0.0)
    @test all(isnan, zero_result.output[1:3])
    @test iszero(zero_result.output[4])
    hollow = _statistics_spatial(widths, [1.0, 0.5, 0.0], 0.0)
    @test isfinite(hollow.output[1]) && all(isnan, hollow.output[2:3])
    flat = _statistics_spatial(widths, ones(3), 1.0)
    @test all(isnan, flat.output[2:3])
    fill!(flat.workspace.fluence, NaN)
    fill!(flat.workspace.magnitude, NaN)
    SolpropJulia.statistics!(
        flat.output, flat.selection, flat.grid, flat.inputs,
        flat.workspace, flat.execution
    )
    @test all(isfinite, flat.workspace.fluence)
    flat.input.closure[1] = -1.0
    @test_throws ArgumentError SolpropJulia.statistics!(
        flat.output, flat.selection,
        flat.grid, flat.inputs, flat.workspace, flat.execution
    )
    # Intensity keeps a negative signed sample; widths alone require positivity.
    spatial = SolpropJulia.SpatialStatisticsInput(;
        factor = 2.0, source_columns = 1,
        analytic = ComplexF64[1 2; 1 1], analytic_closure = ComplexF64[-1 1; 1 3],
        axis_analytic = reshape(ComplexF64[1, 2], 2, 1),
        axis_analytic_closure = reshape(ComplexF64[-1, 1], 2, 1)
    )
    selection = _statistics_selection(((:peak_intensity, 1), (:on_axis_intensity, 1)))
    SolpropJulia.statistics!(
        zeros(2), selection, flat.grid,
        SolpropJulia.StatisticsInputs(; spatial), flat.workspace, flat.execution
    )
    output = zeros(2)
    SolpropJulia.statistics!(
        output, selection, flat.grid,
        SolpropJulia.StatisticsInputs(; spatial), flat.workspace, flat.execution
    )
    @test output == [6.0, 4.0]
    @test_throws ArgumentError SolpropJulia.SpatialWidthGrid(:unknown, [1, 2], [], [])
    @test_throws ArgumentError SolpropJulia.SpatialWidthGrid(
        :line, [0.0, 1.0, 3.0], [0.0], []
    )
end

@testset "Statistics reject writable aliases before the first write" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    grid = SolpropJulia.StatisticsGrid([1.0, 2.0], [0.0, 1.0], [0.5, 0.5])
    workspace = SolpropJulia.StatisticsWorkspace(
        execution, Float64;
        n_time = 2, n_columns = 2, n_nodes = 2
    )
    weights = [2.0 3.0; 4.0 5.0]
    spectral = SolpropJulia.SpectralStatisticsInput(ones(ComplexF64, 2, 2), weights)
    selection = _statistics_selection(((:energy, 2),))
    inputs = SolpropJulia.StatisticsInputs(; spectral)
    for output in (@view(weights[:, 1]), grid.parseval, workspace.column_energy)
        retained = copy(output)
        @test_throws ArgumentError SolpropJulia.statistics!(
            output, selection, grid,
            inputs, workspace, execution
        )
        @test output == retained
        @test_throws ArgumentError SolpropJulia.spectral_statistics!(
            output,
            selection.slots[1], grid, spectral, workspace, execution
        )
        @test output == retained
    end
    population = SolpropJulia.PopulationStatisticsInput(
        [1.0, 2.0], fill(0.25, 2, 2),
        ones(2, 2)
    )
    population_selection = _statistics_selection(((:species_electron_density, 2),))
    for output in (population.densities, @view(population.fractions[:, 1]))
        retained = copy(output)
        @test_throws ArgumentError SolpropJulia.statistics!(
            output, population_selection,
            grid, SolpropJulia.StatisticsInputs(; populations = population),
            workspace, execution
        )
        @test output == retained
    end
    temporal = SolpropJulia.TemporalStatisticsInput(ones(2, 2), ones(2, 2))
    temporal_selection = _statistics_selection(((:peak_power, 2),))
    @test_throws ArgumentError SolpropJulia.statistics!(
        @view(temporal.analytic[:, 1]),
        temporal_selection, grid, SolpropJulia.StatisticsInputs(; temporal),
        workspace, execution
    )
    # Scratch must not overlap immutable input, nor another writable scratch member.
    aliased = SolpropJulia.StatisticsWorkspace(
        workspace.power, workspace.total_power,
        grid.parseval, workspace.column_moment, workspace.node_total,
        workspace.fluence, workspace.magnitude, workspace.radial_average
    )
    @test_throws ArgumentError SolpropJulia.statistics!(
        zeros(2), selection, grid,
        inputs, aliased, execution
    )
    short = SolpropJulia.StatisticsWorkspace(
        workspace.power, workspace.total_power,
        workspace.column_energy, zeros(1), workspace.node_total,
        workspace.fluence, workspace.magnitude, workspace.radial_average
    )
    @test_throws DimensionMismatch SolpropJulia.statistics!(
        zeros(2), selection, grid,
        inputs, short, execution
    )
    same = SolpropJulia.StatisticsWorkspace(
        workspace.power, workspace.total_power,
        workspace.column_energy, workspace.column_energy, workspace.node_total,
        workspace.fluence, workspace.magnitude, workspace.radial_average
    )
    @test_throws ArgumentError SolpropJulia.statistics!(
        zeros(2), selection, grid,
        inputs, same, execution
    )
end

@testset "Identity centroids preserve the relative empty-column floor" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    grid = SolpropJulia.StatisticsGrid([1.0, 3.0], [0.0, 1.0], ones(2))
    workspace = SolpropJulia.StatisticsWorkspace(
        execution, Float64;
        n_time = 2, n_columns = 3
    )
    spectrum = ComplexF64[1 0 0; 0 1.0e-10 1.0e-5]
    weights = ones(2, 3)
    input = SolpropJulia.SpectralStatisticsInput(spectrum, weights)
    output = zeros(3)
    slot = SolpropJulia.StatisticSlot(:omega0, 1, 3)
    SolpropJulia.spectral_statistics!(output, slot, grid, input, workspace, execution)
    @test output[1] == 1.0 && isnan(output[2]) && output[3] == 3.0
    # Poynting statistics deliberately have no relative floor. The public NumPy
    # dispatch changes at exact unit weight, even for an unlaunched tiny column.
    weights[1] = 1.0 + eps(Float64)
    SolpropJulia.spectral_statistics!(output, slot, grid, input, workspace, execution)
    @test output == [1.0, 3.0, 3.0]
end
