# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import LinearAlgebra
import SolpropJulia

function observation_slots(names, n_modes, n_nodes = 0, n_species = 0; periodic = ())
    slots, offset = SolpropJulia.StatisticSlot[], 1
    for name in names
        count = name in (:energy, :omega0, :peak_power) ? n_modes :
            name in (:energy_window, :peak_power_window) ? 2n_modes :
            name == :species_ionisation ? n_nodes * n_species :
            name == :species_electron_density ? n_species : 1
        push!(slots, SolpropJulia.StatisticSlot(name, offset, count))
        offset += count
    end
    return SolpropJulia.StatisticsSelection(Tuple(slots), periodic)
end

function observation_profile(execution, base, slope)
    coefficients = reshape([base + slope / 2, slope / 2], 1, 2, 1)
    return SolpropJulia.SourceProfile(
        SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], coefficients)
    )
end

function observation_grid(kind)
    n_time, dt = 8, 0.125
    count = kind == :field ? 5 : 8
    omega = collect(0.0:(count - 1))
    time = collect((-4:3) .* dt)
    parseval = kind == :field ? dt / n_time .* [1.0, 4.0, 4.0, 4.0, 1.0] :
        fill(dt / n_time, count)
    windows = fill(false, count, 2)
    windows[1:3, 1] .= true
    windows[3:end, 2] .= true
    return SolpropJulia.StatisticsGrid(omega, time, parseval; windows)
end

function observation_inverse(spectrum, n_time; field = false, analytic = false)
    full = zeros(Complex{BigFloat}, n_time, size(spectrum, 2))
    if field
        full[1, :] .= real.(spectrum[1, :])
        full[n_time ÷ 2 + 1, :] .= real.(spectrum[end, :])
        for row in 2:(n_time ÷ 2)
            full[row, :] .= spectrum[row, :]
            full[n_time - row + 2, :] .= conj.(spectrum[row, :])
        end
        if analytic
            full[2:(n_time ÷ 2), :] .*= 2
            full[(n_time ÷ 2 + 2):end, :] .= 0
        end
    else
        full .= spectrum
    end
    result = [
        sum(
            full[k + 1, column] * cis(2big(pi) * k * t / n_time)
                for k in 0:(n_time - 1)
        ) / n_time
            for t in 0:(n_time - 1), column in axes(spectrum, 2)
    ]
    return field && !analytic ? Float64.(real.(result)) : ComplexF64.(result)
end

function observation_field(kind)
    count = kind == :field ? 5 : 8
    return ComplexF64[
        (0.2row + 0.1column) * cis(0.3row - 0.7column)
            for row in 1:count, column in 1:2
    ]
end

function observation_case(
        execution, kind; weighted = true, names = (
            :omega0, :energy, :energy_window, :peak_power, :peak_power_window,
            :peak_power_total, :peak_intensity, :density, :pressure, :core_radius,
            :quadrature_error,
        ), periodic = (2, 5, 8)
    )
    grid = observation_grid(kind)
    shape = (length(grid.omega), 2)
    entries = weighted ? [
            1.2 + 0.03row + 0.08column
            for row in 1:shape[1], column in 1:2
        ] : ones(shape)
    coefficients = reshape(vcat(vec(entries), vec(0.2entries)), prod(shape), 2, 1)
    weight = weighted ? SolpropJulia.ObservationWeight(
            SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], coefficients), shape
        ) :
        SolpropJulia.ObservationWeight(execution, entries)
    selection = observation_slots(names, 2; periodic)
    prepared = SolpropJulia.ObservationPreparation(
        execution, grid, selection;
        kind, n_modes = 2, power_weight = weight,
        profiles = SolpropJulia.ObservationProfiles(
            density = observation_profile(execution, 2.0, 3.0),
            pressure = observation_profile(execution, 0.3, 0.1),
            core_radius = observation_profile(execution, 1.0, -0.2)
        ),
        axis_synthesis = [0.8 -0.3; 0.4 0.6],
        taper = observation_profile(execution, 1.0, 0.4), intensity_factor = 0.7
    )
    return SolpropJulia.ObservationWorkspace(execution, prepared; node_chunk = 2)
end

function observation_slice(work, name)
    index = findfirst(
        slot -> SolpropJulia._STATISTIC_NAMES[slot.code] == name,
        work.data.selection.slots
    )
    slot = work.data.selection.slots[index]
    return work.values[slot.offset:(slot.offset + slot.count - 1)]
end

function observation_allocated(work, field)
    SolpropJulia.observe!(work, field, 0.25; quadrature_error = 0.01)
    return @allocated SolpropJulia.observe!(work, field, 0.25; quadrature_error = 0.01)
end

@testset "Fresh modal observation pairs and packed periodic slices" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope), weighted in (false, true)
        work = observation_case(execution, kind; weighted)
        field = observation_field(kind)
        before = copy(field)
        for position in (0.25, 0.75, 0.25)
            SolpropJulia.observe!(work, field, position; quadrature_error = 0.01)
            @test work.valid && work.z == position && field == before
            analytic = observation_inverse(
                field, 8; field = kind == :field, analytic = true,
            )
            weighted_time = observation_inverse(
                field .* work.weight, 8;
                field = kind == :field, analytic = true
            )
            # Eight-term BigFloat DFT versus FFT plus synthesis: 128 unit roundoffs.
            tolerance = 128eps(Float64)
            @test isapprox(work.pair.samples, analytic; rtol = tolerance, atol = tolerance)
            @test isapprox(
                work.pair.weighted, weighted_time;
                rtol = tolerance, atol = tolerance
            )
            power = real.(analytic .* conj.(weighted_time))
            expected_peak = vec(maximum(power; dims = 1))
            @test isapprox(
                observation_slice(work, :peak_power), expected_peak;
                rtol = tolerance, atol = tolerance
            )
            @test isapprox(
                only(observation_slice(work, :peak_power_total)),
                maximum(sum(power; dims = 2)); rtol = tolerance, atol = tolerance
            )
            energy = vec(
                sum(
                    abs2.(field) .* work.weight .* work.data.grid.parseval;
                    dims = 1
                )
            )
            @test isapprox(
                observation_slice(work, :energy), energy;
                rtol = tolerance, atol = tolerance
            )
            axis = (1 + 0.4position) .* work.data.axis_synthesis
            first, second = analytic * transpose(axis), weighted_time * transpose(axis)
            intensity = 0.7maximum(sum(real.(first .* conj.(second)); dims = 2))
            @test isapprox(
                only(observation_slice(work, :peak_intensity)), intensity;
                rtol = tolerance, atol = tolerance
            )
            for window in 1:2
                gate = work.data.grid.windows[:, window]
                first = observation_inverse(
                    field .* gate, 8;
                    field = kind == :field, analytic = true
                )
                second = observation_inverse(
                    field .* gate .* work.weight, 8;
                    field = kind == :field, analytic = true
                )
                @test isapprox(
                    work.windows.samples[:, :, window], first;
                    rtol = tolerance, atol = tolerance
                )
                @test isapprox(
                    work.windows.weighted[:, :, window], second;
                    rtol = tolerance, atol = tolerance
                )
            end
            @test only(observation_slice(work, :density)) ≈ 2 + 3position
            @test only(observation_slice(work, :pressure)) ≈ 0.3 + 0.1position
            @test only(observation_slice(work, :core_radius)) ≈ 1 - 0.2position
            @test only(observation_slice(work, :quadrature_error)) == 0.01
            plane = copy(work.values)
            SolpropJulia.observe!(work, field, position; periodic = true)
            for (index, slot) in pairs(work.data.selection.slots)
                indices = slot.offset:(slot.offset + slot.count - 1)
                if index in work.data.selection.periodic
                    @test isapprox(
                        work.values[indices], plane[indices];
                        rtol = tolerance, atol = tolerance
                    )
                else
                    @test all(iszero, work.values[indices])
                end
            end
        end
        @test observation_allocated(work, field) == 0
        # Poisoning old reconstructions cannot affect the next independent observation.
        fill!(work.pair.samples, ComplexF64(NaN, NaN))
        fill!(work.windows.pair.spectrum, ComplexF64(NaN, NaN))
        fill!(work.axis[1], ComplexF64(NaN, NaN))
        SolpropJulia.observe!(work, zero(field), 0.4; quadrature_error = 0.0)
        @test all(iszero, work.pair.samples) && all(iszero, work.windows.samples)
        @test all(iszero, observation_slice(work, :energy))
        @test all(isnan, observation_slice(work, :omega0))
    end
end

@testset "Identity statistics, widths and family-only preparation" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope)
        work = observation_case(
            execution, kind; weighted = false,
            names = (
                :energy, :physical_energy, :peak_power_total, :analytic_flux,
                :fwhm_t, :fwhm_t_outer,
            ), periodic = (2, 4, 5)
        )
        field = observation_field(kind)
        SolpropJulia.observe!(work, field, 0.2)
        @test isapprox(
            only(observation_slice(work, :physical_energy)),
            sum(observation_slice(work, :energy)); rtol = 8eps(Float64), atol = 0.0
        )
        @test isapprox(
            observation_slice(work, :analytic_flux),
            observation_slice(work, :peak_power_total); rtol = 8eps(Float64), atol = 0.0
        )
        @test work.axis === nothing && work.windows === nothing
        @test observation_allocated(work, field) == 0
        windows = observation_case(
            execution, kind;
            names = (:peak_power_window,), periodic = (1,)
        )
        SolpropJulia.observe!(windows, field, 0.2)
        @test windows.pair === nothing && windows.windows !== nothing
        @test all(isfinite, windows.values)
        @test observation_allocated(windows, field) == 0
        refusal = observation_case(
            execution, kind;
            names = (:physical_energy,), periodic = ()
        )
        @test_throws ArgumentError SolpropJulia.observe!(refusal, field, 0.2)
        @test !refusal.valid
    end
end

function observation_cycle(; moment_rtol = 2.0e-8)
    nodes, weights, ends = Float64[], Float64[], Int[]
    for count in (16, 32)
        off_diagonal = [index / sqrt(4index^2 - 1) for index in 1:(count - 1)]
        eigen = LinearAlgebra.eigen(
            LinearAlgebra.SymTridiagonal(zeros(count), off_diagonal)
        )
        append!(nodes, eigen.values)
        append!(weights, 2 .* eigen.vectors[1, :] .^ 2)
        push!(ends, length(nodes))
    end
    return SolpropJulia.CycleQuadrature(
        nodes, weights, Tuple(ends);
        moment_rtol, negligible_rate = 1.0e-30
    )
end

function observation_expansion_witnesses(execution)
    for kind in (:field, :envelope)
        grid = observation_grid(kind)
        field = observation_field(kind)
        terms = ones(length(grid.omega), 2, 2)
        terms[:, :, 2] .= [
            0.2row + 0.1column
                for row in eachindex(grid.omega), column in 1:2
        ]
        coefficients = reshape([1.2, 0.7, 0.2, -0.1], 2, 2, 1)
        weights = SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], coefficients)
        prepared = SolpropJulia.ObservationPreparation(
            execution, grid,
            observation_slots((:energy, :peak_power_total), 2);
            kind, n_modes = 2,
            power_weight = SolpropJulia.ObservationWeight(execution, terms, weights)
        )
        work = SolpropJulia.ObservationWorkspace(execution, prepared; node_chunk = 1)
        for position in (0.1, 0.6)
            SolpropJulia.observe!(work, field, position)
            coordinate = 2position - 1
            expected = (1.2 + 0.2coordinate) .* terms[:, :, 1] .+
                (0.7 - 0.1coordinate) .* terms[:, :, 2]
            # Two represented coefficients, two terms and at most eight spectral sums.
            tolerance = 64eps(Float64)
            @test isapprox(work.weight, expected; rtol = tolerance, atol = 0.0)
            energy = vec(sum(abs2.(field) .* expected .* grid.parseval; dims = 1))
            @test isapprox(
                observation_slice(work, :energy), energy;
                rtol = tolerance, atol = 0.0
            )
        end
        @test length(work.weight_work) == 2
        @test observation_allocated(work, field) == 0
    end
    return
end

@testset "Expansion weights use prepared observation scratch" begin
    for tasks in (1, 2)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        observation_expansion_witnesses(execution)
    end
end

function observation_population_case(
        execution, kind; ionising = true, chunk = 2,
        arithmetic = SolpropJulia.SourceArithmetic(Float64), moment_rtol = 2.0e-8,
    )
    grid = observation_grid(kind)
    n_frequency = length(grid.omega)
    crop = kind == :field ? collect(0:4) : [0, 1, 2, 3, 12, 13, 14, 15]
    active = fill(true, n_frequency)
    active[1] = false
    fine = SolpropJulia.ModalSourceGrid(
        execution; kind, n_time = 16, dt = 0.0625,
        crop_index = crop, omega = grid.omega, active,
        omega_window = ones(n_frequency), source_window = ones(16),
        source_window_derivative = zeros(16), arithmetic
    )
    maps = SolpropJulia.ModalSourceMap(
        execution;
        synthesis = 2 .* [0.8 -0.3; 0.2 0.4; 0.9 0.6], projection = zeros(2, 3),
        taper = observation_profile(execution, 0.7, 0.2), arithmetic
    )
    channels = ntuple(2) do index
        lower, upper, scale = 1.0e-12, 100.0, 0.4index
        rate = SolpropJulia.IonisationRateTable(
            reshape([0.0, 0.0, 2.0, log(scale) + 2log(lower)], 4, 1),
            lower, upper, scale * upper^2, log(lower), log(upper / lower)
        )
        channel = SolpropJulia.PlasmaChannel(rate, 1.0, 0.2)
        SolpropJulia.ModalPlasmaChannel(
            channel,
            observation_profile(execution, index + 1.0, index * 0.1),
            observation_profile(execution, index * 0.1, 0.02);
            quadrature = kind == :field ? nothing : observation_cycle(; moment_rtol),
            arithmetic,
        )
    end
    plasma = !ionising ? nothing : kind == :field ?
        SolpropJulia.ModalPlasma(
            execution, channels,
            SolpropJulia.RunningIntegralRule([1.0], 0.0625; floor_exposure = false);
            arithmetic,
        ) :
        SolpropJulia.ModalPlasma(
            execution, channels;
            drude = -ones(n_frequency), loss = -im .* ones(n_frequency), arithmetic,
        )
    populations = SolpropJulia.ObservationPopulations(
        plasma, fine, maps;
        species_densities = (
            SolpropJulia.SourceProfile(3.0),
            observation_profile(execution, 4.0, 1.0), SolpropJulia.SourceProfile(2.0),
        ),
        channel_species = ionising ? (3, 1) : (), peak_species = 0
    )
    selection = observation_slots(
        (
            :species_ionisation, :species_electron_density,
            :electron_density, :peak_ionisation_rate,
        ), 2, 3, 3; periodic = (1, 3)
    )
    prepared = SolpropJulia.ObservationPreparation(
        execution, grid, selection;
        kind, n_modes = 2, power_weight = SolpropJulia.ObservationWeight(
            execution,
            ones(n_frequency, 2)
        ), populations, arithmetic
    )
    return SolpropJulia.ObservationWorkspace(execution, prepared; node_chunk = chunk)
end

@testset "Fresh independent population scans and complete species order" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope), ionising in (false, true)
        work = observation_population_case(execution, kind; ionising)
        field = observation_field(kind)
        position = 0.3
        SolpropJulia.observe!(work, field, position)
        @test work.pair === nothing && work.axis === nothing
        @test work.populations.densities ≈ [3.0, 4.3, 2.0]
        @test all(iszero, work.populations.fractions[:, 2])
        if ionising
            count = kind == :field ? 9 : 16
            padded = zeros(ComplexF64, count, 2)
            padded[work.data.populations.grid.crop_index .+ 1, :] .= field
            modal = observation_inverse(padded, 16; field = kind == :field)
            nodes = (0.7 + 0.2position) .* modal *
                transpose(work.data.populations.map.synthesis)
            # Quadratic rate has exact scalar cycle average |A|²/2. The envelope
            # quadrature's declared tolerance, log-spline and 16-point FFT set this budget.
            tolerance = kind == :field ? 512eps(Float64) : 4.0e-8
            for (index, species) in enumerate((3, 1))
                rates = (kind == :field ? 0.4index : 0.2index) .* abs2.(nodes)
                exposure = 0.0625 .* vec(
                    sum(rates[2:15, :]; dims = 1) .+
                        0.5 .* (rates[1:1, :] .+ rates[16:16, :])
                )
                initial = 0.1index + 0.02position
                expected = initial .+ (1 - initial) .* -expm1.(-exposure)
                @test isapprox(
                    work.populations.fractions[:, species], expected;
                    rtol = tolerance, atol = tolerance
                )
                @test isapprox(
                    work.populations.peaks[:, species],
                    vec(maximum(rates; dims = 1)); rtol = tolerance, atol = tolerance
                )
            end
            @test all(iszero, work.populations.nodes[:, 2])
            fill!(work.populations.workspace.fraction, NaN)
            fill!(work.populations.nodes, kind == :field ? NaN : ComplexF64(NaN, NaN))
        else
            @test all(iszero, work.populations.fractions)
            @test all(iszero, work.populations.peaks)
        end
        @test observation_allocated(work, field) == 0
        single_chunk = observation_population_case(execution, kind; ionising, chunk = 3)
        SolpropJulia.observe!(single_chunk, field, 0.25)
        @test isapprox(work.values, single_chunk.values; rtol = 128eps(Float64), atol = 0.0)
        SolpropJulia.observe!(work, zero(field), position)
        @test all(iszero, work.populations.peaks)
        if ionising
            @test all(work.populations.fractions[:, 1] .≈ 0.2 + 0.02position)
            @test all(work.populations.fractions[:, 3] .≈ 0.1 + 0.02position)
        end
    end
end

function observation_monitor_case(
        execution, kind; split = false, noise = nothing,
        continuation = nothing
    )
    grid = observation_grid(kind)
    regions = zeros(8, 4)
    regions[1:2, 1] .= 0.2
    regions[7:8, 2] .= 0.3
    regions[:, 3] .= 0.01
    regions[:, 4] .= collect(range(0.1, 0.8; length = 8))
    policy = SolpropJulia.TemporalMonitorPolicy(;
        warn = 0.8, stop = 0.9,
        activity = 1.0, quiet_share = 0.1, quiet_checks = 2, quiet_collar = true,
        cadence = 3, interval = Inf, launch_floor_ratio = 1.0e-6, floor_margin = 1.2,
        split, crossing_share = 0.1, bucket_limits = [0.1, 0.4, Inf]
    )
    buckets = [mod(row + column, 2) + 1 for row in eachindex(grid.omega), column in 1:2]
    weights = SolpropJulia.TemporalMonitorWeights(regions, [0.5, 1.3]; buckets)
    temporal = SolpropJulia.TemporalObservationData(policy, weights)
    prepared = SolpropJulia.ObservationPreparation(
        execution, grid,
        SolpropJulia.StatisticsSelection(()); kind, n_modes = 2,
        power_weight = SolpropJulia.ObservationWeight(
            execution,
            ones(length(grid.omega), 2)
        ), temporal
    )
    return SolpropJulia.ObservationWorkspace(
        execution, prepared; node_chunk = 2,
        noise_spectrum = noise, temporal_continuation = continuation,
        continuation_plane = true
    )
end

function observation_monitor_step(work, field, position, step, notices)
    SolpropJulia.reset_notices!(notices)
    SolpropJulia.capture_split!(work, field, position, step)
    return SolpropJulia.observation_accepted!(work, field, position, step, notices)
end

@testset "Coarse monitor reconstruction and exactly once split capture" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    notices = SolpropJulia.MonitorNotices(8)
    for kind in (:field, :envelope), split in (false, true)
        field = observation_field(kind)
        work = observation_monitor_case(execution, kind; split, noise = 0.1field)
        SolpropJulia.reset_notices!(notices)
        row = SolpropJulia.observation_launch!(work, field, 0.0, notices)
        @test row.emit
        coarse = observation_inverse(field, 8; field = kind == :field)
        power = vec(sum(abs2.(coarse) .* [0.5 1.3]; dims = 2))
        # Direct high-precision inverse versus FFT and the weighted scalar reduction.
        @test isapprox(
            work.monitor.launch, sum(power);
            rtol = 128eps(Float64), atol = 0.0
        )
        @test isapprox(
            work.monitor.floor, 0.01sum(power);
            rtol = 128eps(Float64), atol = 0.0
        )
        @test length(work.monitor_work.reading.occupancy) == 3
        @test work.monitor_work.reading.occupancy[3] == 0.0
        @test_throws ArgumentError SolpropJulia.observation_launch!(
            work, field, 0.0, notices,
        )
        @test_throws ArgumentError SolpropJulia.observation_check!(
            work, field, 0.0, notices,
        )
        before = copy(field)
        removed = 0.0
        for step in 1:3
            position = 0.1step
            if split
                @test_throws ArgumentError SolpropJulia.observation_accepted!(
                    work, field, position, 0.1, notices
                )
                SolpropJulia.capture_split!(work, field, position, 0.1)
                @test_throws ArgumentError SolpropJulia.capture_split!(
                    work, field, position, 0.1
                )
                @test_throws ArgumentError SolpropJulia.observation_accepted!(
                    work, field, position, 0.2, notices
                )
                rates = work.data.temporal.weights.regions[:, 4]
                removed += sum(power .* -expm1.(-rates .* 0.1))
            end
            result = SolpropJulia.observation_accepted!(work, field, position, 0.1, notices)
            @test result.step_cap > 0 && field == before
            @test_throws ArgumentError SolpropJulia.observation_accepted!(
                work, field, position, 0.1, notices
            )
        end
        @test work.monitor.steps == 1 && work.monitor.skipped == 2
        @test work.monitor_work.reading.removed == 0.0
        if split
            @test isapprox(
                work.monitor.raw_removed, removed / sum(power);
                rtol = 128eps(Float64), atol = 0.0
            )
        end
        observation_monitor_step(work, field, 0.4, 0.1, notices)
        counts = (work.monitor.steps, work.monitor.skipped)
        SolpropJulia.observation_check!(work, field, 0.4, notices)
        @test (work.monitor.steps, work.monitor.skipped) == counts
        @test !work.monitor.stale && work.monitor_work.reading.removed == 0.0
        @test_throws ArgumentError SolpropJulia.observation_check!(
            work, field, 0.4, notices; plane = false
        )
        @test SolpropJulia.observation_plane!(work, 0.4).emit
        @test isnan(SolpropJulia.observation_plane!(work, 0.45).collar)
        continuation = SolpropJulia.temporal_continuation(work.monitor)
        restored = observation_monitor_case(
            execution, kind; split,
            noise = 0.1field, continuation
        )
        SolpropJulia.observation_launch!(restored, field, 0.45, notices)
        @test restored.monitor.steps == work.monitor.steps
        @test restored.monitor.quiet_run == work.monitor.quiet_run
        @test restored.monitor.raw_removed == work.monitor.raw_removed
        @test SolpropJulia.observation_finish!(work, notices) === nothing
        observation_monitor_step(work, field, 0.5, 0.1, notices)
        allocated = @allocated observation_monitor_step(work, field, 0.6, 0.1, notices)
        @test allocated == 0
        if split
            SolpropJulia.capture_split!(work, field, 0.7, 0.1)
            @test_throws ArgumentError SolpropJulia.observation_finish!(work, notices)
        end
    end
end

@testset "Observation admission, selected inputs and caller preservation" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    work = observation_case(execution, :field)
    field = observation_field(:field)
    @test_throws DimensionMismatch SolpropJulia.observe!(work, zeros(5, 2), 0.0)
    @test_throws DimensionMismatch SolpropJulia.observe!(work, field[:, 1:1], 0.0)
    @test_throws ArgumentError SolpropJulia.observe!(work, field, NaN)
    @test_throws ArgumentError SolpropJulia.observe!(work, work.spectrum, 0.0)
    @test all(iszero, work.values) && field == observation_field(:field)
    field[1, 1] = Inf
    @test_throws ArgumentError SolpropJulia.observe!(work, field, 0.0)
    @test_throws ArgumentError SolpropJulia.observation_plane!(work, Inf)
    notices = SolpropJulia.MonitorNotices(2)
    field = observation_field(:field)
    @test SolpropJulia.observation_launch!(work, field, 0.0, notices) === nothing
    @test SolpropJulia.capture_split!(work, field, 0.1, 0.1) === nothing
    @test SolpropJulia.observation_accepted!(work, field, 0.1, 0.1, notices) ==
        (quiet = false, step_cap = Inf)
    @test SolpropJulia.observation_check!(work, field, 0.1, notices) ==
        (quiet = false, step_cap = Inf)
    @test SolpropJulia.observation_finish!(work, notices) === nothing
    @test SolpropJulia.observation_plane!(work, 0.1) === nothing
    grid = observation_grid(:field)
    invalid_weight = SolpropJulia.ObservationWeight(
        SolpropJulia.ChebyshevTable(execution, [0.0, 0.1], ones(10, 1, 1)), (5, 2)
    )
    selection = observation_slots((:density,), 2; periodic = (1,))
    prepared = SolpropJulia.ObservationPreparation(
        execution, grid, selection;
        kind = :field, n_modes = 2, power_weight = invalid_weight,
        profiles = SolpropJulia.ObservationProfiles(
            density = SolpropJulia.SourceProfile(2.0),
        ),
    )
    profile_only = SolpropJulia.ObservationWorkspace(execution, prepared; node_chunk = 1)
    SolpropJulia.observe!(profile_only, field, 0.5)
    @test profile_only.values == [2.0] && profile_only.pair === nothing
    @test observation_allocated(profile_only, field) == 0
    for name in (:peak_intensity, :density, :electron_density, :beam_radius)
        selected = observation_slots((name,), 2)
        @test_throws ArgumentError SolpropJulia.ObservationPreparation(
            execution, grid, selected; kind = :field, n_modes = 2,
            power_weight = SolpropJulia.ObservationWeight(execution, ones(5, 2))
        )
    end
    @test_throws ArgumentError SolpropJulia.ObservationWorkspace(
        execution, prepared; node_chunk = 0
    )
    @test_throws ArgumentError SolpropJulia.ObservationProfiles(density = 2.0)
    @test_throws DimensionMismatch SolpropJulia.ObservationPreparation(
        execution, grid,
        SolpropJulia.StatisticsSelection((SolpropJulia.StatisticSlot(:energy, 1, 1),));
        kind = :field, n_modes = 2,
        power_weight = SolpropJulia.ObservationWeight(execution, ones(5, 2))
    )
end

@testset "Population observations are independent of current-only coefficients" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope)
        ordinary = observation_population_case(execution, kind)
        field = observation_field(kind)
        SolpropJulia.observe!(ordinary, field, 0.3)
        previous = ordinary.data.populations
        channels = map(previous.plasma.channels) do original
            rate = original.channel.rate
            current = SolpropJulia.PlasmaChannel(rate, floatmax(Float64), floatmax(Float64))
            SolpropJulia.ModalPlasmaChannel(
                current,
                SolpropJulia.SourceProfile(floatmax(Float64)), original.preionisation;
                quadrature = original.quadrature
            )
        end
        plasma = kind == :field ? SolpropJulia.ModalPlasma(
                execution, channels,
                previous.plasma.rule
            ) : SolpropJulia.ModalPlasma(
                execution, channels;
                drude = previous.plasma.drude, loss = previous.plasma.loss
            )
        populations = SolpropJulia.ObservationPopulations(
            plasma, previous.grid, previous.map;
            species_densities = previous.species_densities,
            channel_species = previous.channel_species, peak_species = previous.peak_species
        )
        prepared = SolpropJulia.ObservationPreparation(
            execution, ordinary.data.grid,
            ordinary.data.selection; kind, n_modes = 2,
            power_weight = ordinary.data.power_weight, populations
        )
        work = SolpropJulia.ObservationWorkspace(execution, prepared; node_chunk = 2)
        SolpropJulia.observe!(work, field, 0.3)
        # Coefficients absent from the equation cannot change its rounding budget.
        @test isapprox(work.values, ordinary.values; rtol = 16eps(Float64), atol = 0.0)
        @test all(isfinite, work.values)
        @test observation_allocated(work, field) == 0
    end
end
