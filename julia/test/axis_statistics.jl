# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The aggregate includes the existing modal observation fixtures first.
function axis_statistics_case(
        execution, kind, matrix; arithmetic = SolpropJulia.SourceArithmetic(Float64),
        moment_rtol = 2.0e-8,
    )
    base = observation_population_case(execution, kind; arithmetic, moment_rtol)
    data, populations = base.data, base.data.populations
    axis = SolpropJulia.ObservationAxisPopulations(
        populations.plasma, populations.grid,
        matrix; species_densities = populations.species_densities,
        channel_species = populations.channel_species, taper = populations.map.taper
    )
    frequencies = length(data.grid.omega)
    density = fill((0.125^2 / (2pi)) * (kind == :field ? 4 : 1), frequencies)
    grid = SolpropJulia.StatisticsGrid(
        data.grid.omega, data.grid.time, data.grid.parseval;
        spectral_density = density
    )
    entries = (
        (:energy, 2), (:on_axis_electron_density, 1), (:on_axis_peak_ionisation_rate, 1),
        (:on_axis_species_electron_density, 3), (:on_axis_species_peak_ionisation_rate, 3),
        (:spectral_flux, frequencies),
    )
    offset = 1
    slots = map(entries) do (name, count)
        slot = SolpropJulia.StatisticSlot(name, offset, count)
        offset += count
        slot
    end
    selection = SolpropJulia.StatisticsSelection(slots, (1,))
    prepared = SolpropJulia.ObservationPreparation(
        execution, grid, selection;
        kind, n_modes = data.n_modes, power_weight = data.power_weight,
        axis_populations = axis, arithmetic,
    )
    return SolpropJulia.ObservationWorkspace(execution, prepared; node_chunk = 1)
end

@testset "Axis population reducers validate the independent row before writes" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    work = SolpropJulia.StatisticsWorkspace(execution, Float64; n_time = 4, n_columns = 2)
    input = SolpropJulia.PopulationStatisticsInput(
        [2.0, 3.0], [0.25 0.5], [4.0 1.0]
    )
    density = SolpropJulia.StatisticSlot(:on_axis_species_electron_density, 1, 2)
    output = fill(-1.0, 2)
    SolpropJulia.axis_population_statistics!(output, density, input, work, execution)
    @test output == [0.5, 1.5]
    @test isempty(work.node_total)
    @test_throws ArgumentError SolpropJulia.axis_population_statistics!(
        output, density, nothing, work, execution
    )
    for invalid in (
            SolpropJulia.PopulationStatisticsInput([2.0, 3.0], ones(2, 2), ones(2, 2)),
            SolpropJulia.PopulationStatisticsInput([2.0, 3.0], ones(1, 2), ones(1, 3)),
        )
        @test_throws DimensionMismatch SolpropJulia.axis_population_statistics!(
            output, density, invalid, work, execution
        )
    end
    for invalid in (
            SolpropJulia.PopulationStatisticsInput([NaN, 3.0], [0.25 0.5], [4.0 1.0]),
            SolpropJulia.PopulationStatisticsInput([2.0, 3.0], [0.25 -0.5], [4.0 1.0]),
            SolpropJulia.PopulationStatisticsInput([2.0, 3.0], [0.25 0.5], [Inf 1.0]),
        )
        @test_throws ArgumentError SolpropJulia.axis_population_statistics!(
            output, density, invalid, work, execution
        )
        @test output == [0.5, 1.5]
    end
    before = copy(input.densities)
    @test_throws ArgumentError SolpropJulia.axis_population_statistics!(
        input.densities, density, input, work, execution
    )
    @test input.densities == before
    @test_throws DimensionMismatch SolpropJulia.axis_population_statistics!(
        output, SolpropJulia.StatisticSlot(:on_axis_electron_density, 1, 2),
        input, work, execution
    )
    overflow = SolpropJulia.PopulationStatisticsInput(
        fill(floatmax(Float64), 2), ones(1, 2), zeros(1, 2)
    )
    @test_throws ArgumentError SolpropJulia.axis_population_statistics!(
        output, SolpropJulia.StatisticSlot(:on_axis_electron_density, 1, 1),
        overflow, work, execution
    )
end

function axis_statistics_reference(work, field, matrix, position)
    data = work.data.axis_populations
    frequencies = data.grid.kind == :field ? 9 : 16
    padded = zeros(ComplexF64, frequencies, size(field, 2))
    padded[data.grid.crop_index .+ 1, :] .= field
    modal = observation_inverse(padded, 16; field = data.grid.kind == :field)
    components = modal * transpose(matrix) * (0.7 + 0.2position)
    magnitude = vec(sum(abs2, components; dims = 2))
    electrons, peaks = zeros(3), zeros(3)
    density = [3.0, 4.0 + position, 2.0]
    for (index, species) in enumerate((3, 1))
        rate = (data.grid.kind == :field ? 0.4index : 0.2index) .* magnitude
        exposure = 0.0625 * (sum(rate[2:15]) + (rate[1] + rate[16]) / 2)
        initial = 0.1index + 0.02position
        fraction = initial + (1 - initial) * -expm1(-exposure)
        electrons[species], peaks[species] = density[species] * fraction, maximum(rate)
    end
    return [sum(electrons), maximum(peaks), electrons..., peaks...]
end

function axis_field_bound(field, matrix, position)
    epsilon = BigFloat(eps(Float64))
    gamma = count -> count * epsilon / (1 - count * epsilon)
    samples, dt = 16, big(0.0625)
    modal = 2 .* vec(sum(abs, Complex{BigFloat}.(field); dims = 1)) ./ samples
    components = (big(0.7) + big(0.2) * position) .* abs.(BigFloat.(matrix)) * modal
    amplitude = sum(components)
    # Qualified FFT/twiddle hypothesis: a length-n inverse has absolute error
    # gamma_(64n)*sum(abs(input))/n. The real Hermitian bound uses twice the stored
    # magnitudes. Thirty-two further operations cover the two-mode contractions,
    # represented taper and reference rounding; Euclidean norm is 1-Lipschitz.
    field_error = 2gamma(64samples + 32) * amplitude
    # On [1e-12,100], the quadratic log spline has |log(E)|<=28. Under the
    # qualified binary64 elementary-function error model (at most four ulps),
    # gamma_512 covers its log/affine/exp path, hypot and square. Bound clipping's
    # discarded [0,1e-12] rate absolutely instead of assuming relative accuracy.
    rate_error = big(0.8) * (2amplitude * field_error + field_error^2) +
        gamma(512) * big(0.8) * (amplitude + field_error)^2 + big(0.8e-24)
    exposure = samples * dt * big(0.8) * amplitude^2
    exposure_error = samples * dt * rate_error +
        gamma(4samples) * (exposure + samples * dt * rate_error)
    # Nonnegative exposure makes |d(initial+(1-initial)*-expm1(-S))/dS|<=1.
    # The two active densities sum to five; passive species are exact zeros.
    fraction_error = exposure_error + gamma(32) * (1 + exposure)
    return Float64(max(rate_error, 5fraction_error + 5gamma(16)))
end

function axis_statistics_allocated(work, field)
    SolpropJulia.observe!(work, field, 0.25)
    return @allocated SolpropJulia.observe!(work, field, 0.25)
end

@testset "Independent fine-axis populations and per-bin spectral flux" begin
    for tasks in (1, 2), kind in (:field, :envelope)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(),
            tasks, true
        )
        matrix = kind == :field ? [1.0 -0.5; 0.25 0.75] : [1.0 -0.5; 0.5 -0.25]
        work = axis_statistics_case(execution, kind, matrix)
        field = observation_field(kind)
        @test work.populations === nothing && work.axis === nothing
        @test size(work.axis_populations.nodes) == (16, 1)
        @test size(work.axis_populations.components, 2) == (kind == :field ? 2 : 1)
        for position in (0.0, 0.25, 0.7)
            SolpropJulia.observe!(work, field, position)
            reference = axis_statistics_reference(work, field, matrix, position)
            # The field bound propagates the transform, contraction, rate and
            # exposure errors above, including the coordinate-free vector magnitude.
            # The envelope witness instead uses the fixed analytic accuracy envelope
            # for the independently converged cycle quadrature (moment_rtol=2e-8),
            # as in the scalar population qualification; it is not a rounding claim.
            tolerance = kind == :field ? axis_field_bound(field, matrix, position) : 4.0e-8
            @test tolerance > 0
            @test isapprox(
                work.values[3:10], reference; rtol = 0.0,
                atol = tolerance * (kind == :field ? 1.0 : max(1.0, maximum(reference)))
            )
            @test iszero(work.values[6]) && iszero(work.values[9])
            exact_spectrum = [
                BigFloat(work.data.grid.spectral_density[row]) *
                    sum(abs2(Complex{BigFloat}(field[row, column])) for column in 1:2)
                    for row in axes(field, 1)
            ]
            bound = 16eps(Float64) * maximum(abs, exact_spectrum)
            @test bound > 0
            @test maximum(abs, BigFloat.(work.values[11:end]) - exact_spectrum) <= bound
        end
        @test axis_statistics_allocated(work, field) == 0
        fill!(work.axis_populations.nodes, eltype(work.axis_populations.nodes)(NaN))
        fill!(work.axis_populations.workspace.fraction, NaN)
        SolpropJulia.observe!(work, field, 0.25; periodic = true)
        @test all(iszero, work.values[3:end])
        @test all(isnan, work.axis_populations.workspace.fraction)
        SolpropJulia.observe!(work, zero(field), 0.25)
        @test all(iszero, work.values[8:10]) && all(iszero, work.values[11:end])
        @test work.values[3] > 0
        @test all(isfinite, work.axis_populations.workspace.fraction)
    end
end

@testset "Scalar envelope axis admission uses exact dyadic proportionality" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    original = observation_population_case(execution, :envelope).data.populations
    constructor = matrix -> SolpropJulia.ObservationAxisPopulations(
        original.plasma,
        original.grid, matrix; species_densities = original.species_densities,
        channel_species = original.channel_species, taper = original.map.taper
    )
    for matrix in (
            [1.0 -0.5; 0.0 0.0], [0.0 0.0; 1.0 -0.5], zeros(2, 2),
            [1.0 -0.5; -2.0 1.0], ldexp.([1.0 -0.5; 0.5 -0.25], -900),
            ldexp.([1.0 -0.5; 0.5 -0.25], 900),
        )
        prepared = constructor(matrix)
        @test size(prepared.synthesis) == (1, 2)
        for column in axes(matrix, 2)
            expected = sqrt(sum(abs2, BigFloat.(matrix[:, column])))
            # Exact proportionality leaves only the division/hypot/product at
            # preparation; this bounded magnitude witness permits eight roundings.
            bound = 8eps(Float64) * expected
            @test abs(abs(BigFloat(prepared.synthesis[1, column])) - expected) <= bound
        end
    end
    adjacent = nextfloat(1.0)
    misleading = [1.0 adjacent; adjacent 1.0 + 2eps(Float64)]
    @test misleading[1, 1] * misleading[2, 2] ==
        misleading[1, 2] * misleading[2, 1]
    @test_throws ArgumentError constructor(misleading)
    @test_throws ArgumentError constructor([1.0 0.0; 0.0 1.0])
    @test_throws ArgumentError constructor(fill(floatmax(Float64), 2, 2))
    @test_throws ArgumentError constructor(fill(NaN, 2, 2))
    @test_throws ArgumentError constructor(ones(3, 2))
    @test_throws ArgumentError constructor(ones(Float32, 2, 2))
    matrix = [1.0 -0.5; 0.5 -0.25]
    owned = constructor(matrix)
    before = copy(owned.synthesis)
    matrix .= NaN
    @test owned.synthesis == before
    for name in (
            :on_axis_electron_density, :on_axis_peak_ionisation_rate,
            :on_axis_species_electron_density, :on_axis_species_peak_ionisation_rate,
            :spectral_flux,
        )
        @test_throws ArgumentError SolpropJulia.StatisticsSelection(
            (SolpropJulia.StatisticSlot(name, 1, 1),), (1,)
        )
    end
end

@testset "Spectral density endpoint factors and unrelated NaN slots stay distinct" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    grid = SolpropJulia.StatisticsGrid(
        [0.0, 1.0, 2.0], collect(0.0:3.0),
        [0.25, 1.0, 0.25]; spectral_density = fill(2 / pi, 3)
    )
    input = SolpropJulia.SpectralStatisticsInput(ComplexF64[1 0; 2 0; 3 0], ones(3, 2))
    work = SolpropJulia.StatisticsWorkspace(execution, Float64; n_time = 4, n_columns = 2)
    selection = SolpropJulia.StatisticsSelection(
        (
            SolpropJulia.StatisticSlot(:omega0, 1, 2),
            SolpropJulia.StatisticSlot(:spectral_flux, 3, 3),
        )
    )
    output = zeros(5)
    SolpropJulia.statistics!(
        output, selection, grid,
        SolpropJulia.StatisticsInputs(; spectral = input), work, execution
    )
    @test isnan(output[2])
    @test isapprox(
        output[3:5], [1.0, 4.0, 9.0] .* (2 / pi);
        rtol = 8eps(Float64), atol = 0.0
    )
    bad = SolpropJulia.SpectralStatisticsInput(input.spectrum, fill(2.0, 3, 2))
    @test_throws ArgumentError SolpropJulia.spectral_statistics!(
        output,
        selection.slots[2], grid, bad, work, execution
    )
    missing = SolpropJulia.StatisticsGrid(grid.omega, grid.time, grid.parseval)
    @test_throws ArgumentError SolpropJulia.spectral_statistics!(
        output,
        selection.slots[2], missing, input, work, execution
    )
end

@testset "Foreign offset observation and axis destinations preserve their owners" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    work = axis_statistics_case(execution, :field, [1.0 -0.5; 0.25 0.75])
    owner = work.axis_populations.spectrum
    GC.@preserve owner begin
        field = unsafe_wrap(
            Array, pointer(owner, 2), size(observation_field(:field)); own = false
        )
        field .= observation_field(:field)
        before = copy(owner)
        @test !Base.mightalias(field, owner)
        @test_throws ArgumentError SolpropJulia.observe!(work, field, 0.25)
        @test owner == before
        @test !work.valid
    end
    field = observation_field(:field)
    @test axis_statistics_allocated(work, field) == 0
    @test work.valid

    storage = [2.0, 3.0, 5.0, 7.0]
    input = SolpropJulia.PopulationStatisticsInput(
        view(storage, 1:3), [0.5 0.25 0.75], zeros(1, 3)
    )
    statistic = SolpropJulia.StatisticSlot(:on_axis_species_electron_density, 1, 3)
    selection = SolpropJulia.StatisticsSelection((statistic,))
    workspace = SolpropJulia.StatisticsWorkspace(
        execution, Float64; n_time = 4, n_columns = 2
    )
    grid = SolpropJulia.StatisticsGrid([0.0, 1.0, 2.0], collect(0.0:3.0), ones(3))
    GC.@preserve storage begin
        output = unsafe_wrap(Array, pointer(storage, 2), (3,); own = false)
        before = copy(storage)
        @test !Base.mightalias(output, input.densities)
        @test_throws ArgumentError SolpropJulia.axis_population_statistics!(
            output, statistic, input, workspace, execution
        )
        @test storage == before
        @test_throws ArgumentError SolpropJulia.statistics!(
            output, selection, grid,
            SolpropJulia.StatisticsInputs(; axis_populations = input), workspace, execution
        )
        @test storage == before
    end
    output = zeros(3)
    SolpropJulia.axis_population_statistics!(output, statistic, input, workspace, execution)
    @test output == [1.0, 0.75, 3.75]
end
