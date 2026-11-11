# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

function spatial_reconstruction_budget(prepared)
    n_time = length(prepared.grid.time)
    columns = prepared.n_modes
    spatial = prepared.map isa SolpropJulia.CartesianObservationMap ? columns : 0
    # Conditional on the qualified FFT/twiddle model, each electric/magnetic
    # path has a <=64n FFT budget per transformed dimension, <=8m real dot
    # operations and one map/coefficient conversion. Two such errors enter the
    # Hermitian product. The final64 operations cover coefficients, products,
    # normalized protected reductions and the host restoration. Absolute envelopes
    # are formed independently before spatial or temporal cancellation.
    count = 2 * (64n_time + 64spatial + 8columns + 32) + 64
    unit = Float64(eps(Float32)) / 2
    return count * unit / (1 - count * unit)
end

@testset "Scalar spatial reconstruction keeps physical scales outside binary32 work" begin
    for kind in (:field, :envelope), route in (:radial, :cartesian, :line)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), 2, true,
        )
        names = (:energy, :peak_intensity, :on_axis_intensity, :on_axis_spectrum)
        reference_data, physical = spatial_observation_case(
            execution, kind, route; names, periodic = (1, 3), varying = route == :radial,
        )
        for (state_scale, field_scale) in ((0, 0), (35, -15), (-35, 20))
            arithmetic = SolpropJulia.SourceArithmetic(
                Float32; state_exponent = state_scale, field_exponent = field_scale,
            )
            prepared, same = spatial_observation_case(
                execution, kind, route; names, periodic = (1, 3),
                varying = route == :radial, arithmetic,
            )
            @test same == physical
            stored = ComplexF32.(
                ldexp.(real.(physical), -state_scale),
                ldexp.(imag.(physical), -state_scale)
            )
            work = SolpropJulia.SpatialObservationWorkspace(
                execution, prepared; node_chunk = 2,
            )
            @test eltype(work.spectrum) === ComplexF32
            @test eltype(work.closure) === ComplexF32
            @test eltype(work.weight) === Float32
            @test eltype(work.statistics.power) === Float32
            @test eltype(work.physical.spectrum) === ComplexF32
            @test eltype(work.physical.pair.samples) === ComplexF32
            @test eltype(work.axis.pair.samples) === ComplexF32
            @test eltype(prepared.on_axis) === ComplexF32
            @test prepared.host_on_axis !== prepared.on_axis
            @test eltype(work.values) === Float64
            if route == :radial
                @test eltype(prepared.map.synthesis) === Float32
                @test prepared.map.host_synthesis !== prepared.map.synthesis
            else
                @test eltype(work.physical.band.band) === ComplexF32
            end
            before = copy(stored)
            for position in (0.2, 0.7, 0.2)
                reference = spatial_observation_reference(
                    reference_data, physical, position,
                )
                expected = vcat(
                    reference.energy, reference.peak,
                    reference.axis_peak, reference.axis_spectrum
                )
                tolerance = spatial_reconstruction_budget(prepared) * reference.bound
                @test tolerance > 0
                SolpropJulia.observe!(work, stored, position)
                @test all(abs.(work.values .- expected) .<= tolerance)
                @test work.valid && work.z == position
                @test stored == before
                @test 1 <= work.weight_range.largest < 2
                @test 1 <= work.closure_range.largest < 2
            end
            @test spatial_observation_allocations(work, stored) == 0
            SolpropJulia.observe!(work, stored, 0.5; periodic = true)
            @test work.values[2] == 0 && all(iszero, work.values[4:end])
            @test_throws ArgumentError SolpropJulia.SpatialObservationWorkspace(
                execution, prepared; node_chunk = 2,
                noise_spectrum = zeros(ComplexF64, size(stored)),
            )
            @test_throws ArgumentError spatial_observation_replace(
                execution, prepared; arithmetic = SolpropJulia.SourceArithmetic(Float32),
                map = reference_data.map,
            )
        end
    end
end

@testset "Finite spatial admittance may carry derived closure exponent above binary64" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    names = (:energy, :peak_intensity, :on_axis_intensity, :on_axis_spectrum)
    arithmetic = SolpropJulia.SourceArithmetic(Float32; state_exponent = 100)
    prepared, physical = spatial_observation_case(
        execution, :envelope, :line; names, periodic = (), arithmetic,
    )
    flux = SolpropJulia.SpatialFluxData(prepared.flux.base; light_speed = ldexp(1.0, 1000))
    prepared = spatial_observation_replace(execution, prepared; flux)
    reference_data, _ = spatial_observation_case(
        execution, :envelope, :line; names, periodic = (),
    )
    reference_data = spatial_observation_replace(execution, reference_data; flux)
    work = SolpropJulia.SpatialObservationWorkspace(execution, prepared; node_chunk = 2)
    stored = ComplexF32.(ldexp.(real.(physical), -100), ldexp.(imag.(physical), -100))
    reference = spatial_observation_reference(reference_data, physical, 0.25)
    expected = vcat(
        reference.energy, reference.peak, reference.axis_peak, reference.axis_spectrum,
    )
    SolpropJulia.observe!(work, stored, 0.25)
    @test prepared.arithmetic.field_exponent + work.closure_exponent > 1023
    @test all(isfinite, work.values)
    @test all(
        abs.(work.values .- expected) .<=
            spatial_reconstruction_budget(prepared) * reference.bound
    )
    @test spatial_observation_allocations(work, stored) == 0
end

function spatial_reconstruction_populations(data, field, position)
    physical = data.map isa SolpropJulia.RadialObservationMap ?
        field * transpose(data.map.host_synthesis) :
        spatial_observation_dft(field, data.map.shape; inverse = true)
    populations = data.populations
    padded = zeros(ComplexF64, data.kind == :field ? 9 : 16, size(physical, 2))
    padded[populations.grid.crop_index .+ 1, :] .= 2physical
    analytic = spatial_observation_analytic(padded, 16, data.kind)
    samples = data.kind == :field ? real.(analytic) : analytic
    fractions, peaks = zeros(size(physical, 2), 3), zeros(size(physical, 2), 3)
    for (index, species) in enumerate((3, 1))
        rate = (data.kind == :field ? 0.4index : 0.2index) .* abs2.(samples)
        exposure = 0.0625 .* vec(
            sum(rate[2:15, :]; dims = 1) .+
                0.5 .* (rate[1:1, :] .+ rate[16:16, :])
        )
        initial = 0.1index + 0.02position
        fractions[:, species] .= initial .+ (1 - initial) .* -expm1.(-exposure)
        peaks[:, species] .= vec(maximum(rate; dims = 1))
    end
    return fractions, peaks
end

function spatial_population_budget(prepared, field)
    n = prepared.n_modes
    synthesis_bound = prepared.map isa SolpropJulia.RadialObservationMap ?
        maximum(sum(abs, prepared.map.host_synthesis; dims = 2)) : 1.0
    amplitude = 4sum(abs, field) * synthesis_bound / 16
    u = Float64(eps(Float32)) / 2
    gamma = count -> count * u / (1 - count * u)
    # Fine transform64*16, transverse transform64n or contraction8n, coefficient
    # and input conversion, then quadratic rate and unit-length trapezoid exposure.
    # The rate function/domain and elementary-function hypothesis match the modal
    # population witnesses; the physical field's exponent enters only log origin.
    field_error = 2gamma(64 * 16 + 72n + 64) * amplitude
    rate_error = 0.8 * (2amplitude * field_error + field_error^2) +
        gamma(512 + 4abs(prepared.arithmetic.field_exponent)) *
        0.8 * (amplitude + field_error)^2 + 0.8e-24
    exposure = 0.8amplitude^2
    fraction_error = rate_error + gamma(64) * (exposure + rate_error) +
        gamma(32) * (1 + exposure)
    return fraction_error, rate_error
end

@testset "Fresh scalar spatial populations share one physical field policy" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope), route in (:radial, :cartesian, :line),
            (es, ef) in ((25, -15), (-25, 20))
        arithmetic = SolpropJulia.SourceArithmetic(
            Float32; state_exponent = es, field_exponent = ef,
        )
        base, physical = spatial_observation_case(
            execution, kind, route; names = (:density,), periodic = (), arithmetic,
        )
        populations = spatial_observation_populations(execution, base; moment_rtol = 2.0e-6)
        nodes = SolpropJulia._spatial_map_shape(base.map)[1]
        selection = spatial_observation_selection(
            (
                :species_ionisation, :species_electron_density,
                :electron_density, :peak_ionisation_rate,
            ),
            length(base.grid.omega), nodes; species = 3,
        )
        data = SolpropJulia.SpatialObservationPreparation(
            execution, base.grid, selection; kind, map = base.map, flux = base.flux,
            spectral_weights = base.spectral_weights, on_axis = base.host_on_axis,
            intensity_factor = base.intensity_factor, populations,
        )
        field = ComplexF32.(physical .* exp2(-es))
        before = copy(field)
        expected_fraction, expected_peak = spatial_reconstruction_populations(
            data, physical, 0.25,
        )
        fraction_bound, rate_bound = spatial_population_budget(data, physical)
        @test fraction_bound > 0 && rate_bound > 0
        for capacity in (1, 2)
            work = SolpropJulia.SpatialObservationWorkspace(
                execution, data; node_chunk = capacity,
            )
            @test eltype(work.populations.nodes) === (kind == :field ? Float32 : ComplexF32)
            @test eltype(work.populations.fractions) === Float32
            for _ in 1:2
                SolpropJulia.observe!(work, field, 0.25)
                @test all(
                    abs.(work.populations.fractions .- expected_fraction) .<=
                        fraction_bound,
                )
                @test all(abs.(work.populations.peaks .- expected_peak) .<= rate_bound)
                @test all(iszero, work.populations.fractions[:, 2])
                @test all(iszero, work.populations.peaks[:, 2])
                @test field == before
            end
            @test spatial_observation_allocations(work, field) == 0
        end
    end
end

function spatial_width_reconstruction(execution, route, arithmetic)
    x = route == :radial ? collect(0.25:0.25:1.0) : collect(-1.5:0.5:1.5)
    y = route == :cartesian ? copy(x) : route == :line ? [0.0] : Float64[]
    shape = route == :radial ? (length(x), 1) : (length(x), length(y))
    nodes, frequency = prod(shape), 8
    grid = SolpropJulia.StatisticsGrid(
        ones(frequency), collect(0.0:7.0), ones(frequency);
        spectral_density = ones(frequency), d_omega = 1.0
    )
    base = SolpropJulia.SeparableSpatialOperator(
        execution; k_squared = ones(ComplexF64, frequency),
        k_squared_low = zeros(ComplexF64, frequency), kperp_sq = zeros(nodes),
        kperp_sq_low = zeros(nodes), frame = zeros(frequency),
        spectral_rate = zeros(frequency), prefactor_numerator = fill(-1.0im, frequency),
        prefactor_clamped = fill(-1.0im, frequency), active = fill(true, frequency),
        evanescent_rate = 1.0, z_span = (0.0, 1.0),
    )
    physical = zeros(ComplexF64, frequency, nodes)
    if route == :radial
        map = SolpropJulia.RadialObservationMap(
            Matrix{Float64}(LinearAlgebra.I, nodes, nodes);
            pad_scale = 1.0, arithmetic
        )
        physical[2, :] .= 1 .- x
        field = physical
        axis = ComplexF64[4 / 3, 0, 0, 0]
    else
        map = SolpropJulia.CartesianObservationMap(shape; arithmetic)
        physical[2, :] .= vec(
            [
                max(1 - abs(a), 0) * max(1 - abs(b), 0) for a in x, b in y
            ]
        )
        field = spatial_observation_dft(physical, shape)
        axis = ComplexF64[
            cis(2pi * (div(shape[1], 2) * a / shape[1] + div(shape[2], 2) * b / shape[2])) /
                nodes for a in 0:(shape[1] - 1), b in 0:(shape[2] - 1)
        ] |> vec
    end
    widths = SolpropJulia.SpatialWidthGrid(route, x, y, ones(nodes))
    selection = SolpropJulia.StatisticsSelection(
        (
            SolpropJulia.StatisticSlot(:beam_radius, 1, 2),
            SolpropJulia.StatisticSlot(:fwhm_r, 3, 1),
        ), (),
    )
    prepared = SolpropJulia.SpatialObservationPreparation(
        execution, grid, selection; kind = :envelope, map,
        flux = SolpropJulia.SpatialFluxData(base; light_speed = 1.0),
        spectral_weights = ones(nodes), on_axis = axis, intensity_factor = 1.0, widths,
    )
    return prepared, field
end

function reconstruction_width_crossing(radius, profile, level)
    first = findfirst(<(level), profile)
    first === nothing && return BigFloat(NaN)
    left = first - 1
    return radius[left] + (profile[left] - level) / (profile[left] - profile[first]) *
        (radius[first] - radius[left])
end

@testset "Reconstructed widths retain their dimensionless scale and conditioning" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for route in (:radial, :cartesian, :line), (es, ef) in ((30, -20), (-30, 20))
        arithmetic = SolpropJulia.SourceArithmetic(
            Float32; state_exponent = es, field_exponent = ef,
        )
        prepared, physical = spatial_width_reconstruction(execution, route, arithmetic)
        widths = prepared.widths
        if route == :radial
            radius = BigFloat.([0; widths.x])
            profile = (1 .- radius) .^ 2
            second = sqrt(
                2sum(BigFloat.(widths.x) .^ 2 .* profile[2:end]) / sum(profile[2:end]),
            )
        else
            radius = BigFloat.(collect(0.0:widths.spacing:1.0))
            # Bilinear interpolation of this separable square exactly factors
            # into the two piecewise-linear one-dimensional sampled profiles.
            ordinate = x -> abs(x) <= big(0.5) ? 1 - big(1.5) * abs(x) :
                big(0.5) * (1 - abs(x))
            profile = route == :line ? ordinate.(radius) : [
                    sum(
                        ordinate(r * c) * ordinate(r * s)
                        for (c, s) in zip(
                            BigFloat.(widths.cosines), BigFloat.(widths.sines),
                        )
                    ) /
                    length(widths.cosines) for r in radius
                ]
            second = sqrt(big(1) / 3)
        end
        expected = [
            second,
            reconstruction_width_crossing(radius, profile, exp(big(-2))),
            2reconstruction_width_crossing(radius, profile, big(0.5)),
        ]
        # Every discrete crossing bracket in this fixed profile has a slope
        # bounded below independently of the comparison. Bound transform/sample
        # perturbation first, then centroid displacement and interpolation. Each
        # threshold moves by <=2*profile_error/slope; the diameter adds a factor2.
        slope = minimum(abs.(diff(profile)) ./ diff(radius))
        @test slope > big(0.1)
        u = Float64(eps(Float32)) / 2
        n = prepared.n_modes
        gamma = count -> count * u / (1 - count * u)
        amplitude_bound = max(1.0, sum(abs, physical) / n)
        amplitude_error = gamma(72n + 64) * amplitude_bound
        delta = 2amplitude_bound * amplitude_error + amplitude_error^2 + gamma(64)
        mass = route == :radial ? sum(profile[2:end]) :
            route == :line ? big(1.5) : big(2.25)
        @test n * delta < mass / 2
        # A zero ring gives a strict radius margin: a perturbed centroid can
        # remove an outer grid radius, but neither bracket below radius1 nor the
        # minimum two-spacing admission can change. The coordinate factor1.5
        # bounds the centroid numerator on this padded Cartesian/line lattice.
        centroid_error = route == :radial ? 0.0 : 3n * delta / Float64(mass)
        if route != :radial
            @test 1.5 - centroid_error > max(1.0, 2widths.spacing)
        end
        profile_error = delta + 6centroid_error + gamma(max(1, length(widths.cosines)))
        tolerance = max(16n * delta / Float64(mass), 4profile_error / Float64(slope))
        @test tolerance > 0
        stored = ComplexF32.(physical .* exp2(-es))
        work = SolpropJulia.SpatialObservationWorkspace(execution, prepared; node_chunk = 2)
        SolpropJulia.observe!(work, stored, 0.25)
        @test all(abs.(BigFloat.(work.values) .- expected) .<= tolerance)
        @test eltype(work.statistics.fluence) === Float32
        @test eltype(work.statistics.scales.nodes) === Int
        @test spatial_observation_allocations(work, stored) == 0
    end
end

@testset "Spatial temporal monitors retain physical canonical-spectrum readings" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope), route in (:radial, :cartesian, :line),
            (es, ef) in ((40, -20), (-40, 30))
        arithmetic = SolpropJulia.SourceArithmetic(
            Float32; state_exponent = es, field_exponent = ef,
        )
        prepared, physical = spatial_observation_case(
            execution, kind, route; names = (), periodic = (), arithmetic,
        )
        reference_data, _ = spatial_observation_case(
            execution, kind, route; names = (), periodic = (),
        )
        work = spatial_observation_monitor(execution, prepared; split = true)
        reference = spatial_observation_monitor(execution, reference_data; split = true)
        stored = ComplexF32.(physical .* exp2(-es))
        before = copy(stored)
        notices = SolpropJulia.MonitorNotices(8)
        SolpropJulia.observation_launch!(work, stored, 0.0, notices)
        SolpropJulia.reset_notices!(notices)
        SolpropJulia.observation_launch!(reference, physical, 0.0, notices)
        first, second = work.monitor_work.reading, reference.monitor_work.reading
        # Only the coarse temporal FFT participates in this monitor proxy. Its
        # conditional64n budget, input/product rounding and protected sums fit
        # gamma(4*64n+16m+64), including the binary64 comparison's smaller error.
        n, m = size(work.monitor_work.samples)
        u = Float64(eps(Float32)) / 2
        count = 4 * 64n + 16m + 64
        bound = count * u / (1 - count * u) * max(1.0, second.total)
        @test bound > 0
        @test abs(first.total - second.total) <= bound
        @test all(abs.(collect(first.regions) .- collect(second.regions)) .<= bound)
        @test all(abs.(first.occupancy .- second.occupancy) .<= bound)
        @test abs(work.monitor.launch - reference.monitor.launch) <= bound
        @test work.physical === nothing && work.axis === nothing
        @test eltype(work.monitor_work.power) === Float32
        @test first.total isa Float64 && eltype(first.occupancy) === Float64
        SolpropJulia.capture_split!(work, stored, 0.1, 0.1)
        SolpropJulia.capture_split!(reference, physical, 0.1, 0.1)
        @test abs(
            work.monitor_work.pending.removed - reference.monitor_work.pending.removed
        ) <= bound
        SolpropJulia.reset_notices!(notices)
        SolpropJulia.observation_accepted!(work, stored, 0.1, 0.1, notices)
        SolpropJulia.reset_notices!(notices)
        SolpropJulia.observation_accepted!(reference, physical, 0.1, 0.1, notices)
        @test abs(work.monitor.raw_removed - reference.monitor.raw_removed) <= bound
        @test stored == before
        spatial_observation_monitor_step(work, stored, 0.2, notices)
        @test (@allocated spatial_observation_monitor_step(work, stored, 0.3, notices)) == 0
        continuation = SolpropJulia.temporal_continuation(work.monitor)
        restored = spatial_observation_monitor(
            execution, prepared; split = true, continuation,
        )
        SolpropJulia.reset_notices!(notices)
        SolpropJulia.observation_launch!(restored, stored, 0.3, notices)
        @test restored.monitor.steps == work.monitor.steps
        @test restored.monitor.raw_removed == work.monitor.raw_removed
        @test restored.monitor.launch == work.monitor.launch
    end
end

function reconstruction_inventory_case(execution, kind, route, arithmetic, capacity)
    if route == :modal
        original = observation_population_case(
            execution, kind; arithmetic, moment_rtol = 2.0e-6,
        )
        base = original.data
        populations = inventory_populations(execution, base.populations, 3; modal = true)
        prepared = SolpropJulia.ObservationPreparation(
            execution, base.grid, inventory_selection(kind);
            kind, n_modes = base.n_modes, power_weight = base.power_weight,
            populations, arithmetic,
        )
        work = SolpropJulia.ObservationWorkspace(execution, prepared; node_chunk = capacity)
        physical = observation_field(kind)
    else
        base, physical = spatial_observation_case(
            execution, kind, route; names = (:density,), periodic = (), arithmetic,
        )
        population_data = spatial_observation_populations(
            execution, base; moment_rtol = 2.0e-6,
        )
        nodes = SolpropJulia._spatial_map_shape(base.map)[1]
        populations = inventory_populations(
            execution, population_data, nodes;
            modal = population_data isa SolpropJulia.ObservationPopulations,
        )
        prepared = SolpropJulia.SpatialObservationPreparation(
            execution, base.grid, inventory_selection(kind); kind, map = base.map,
            flux = base.flux, spectral_weights = base.spectral_weights,
            on_axis = base.host_on_axis, intensity_factor = base.intensity_factor,
            populations,
        )
        work = SolpropJulia.SpatialObservationWorkspace(
            execution, prepared; node_chunk = capacity,
        )
    end
    return work, ComplexF32.(physical .* exp2(-arithmetic.state_exponent))
end

@testset "Complete default plasma rows keep binary32 work and exact retained storage" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    arithmetic = SolpropJulia.SourceArithmetic(
        Float32; state_exponent = 25, field_exponent = -15,
    )
    for kind in (:field, :envelope), route in (:modal, :radial, :cartesian, :line),
            capacity in (1, 2)
        work, field = reconstruction_inventory_case(
            execution, kind, route, arithmetic, capacity,
        )
        for position in (0.25, 0.7, 0.25)
            SolpropJulia.observe!(work, field, position)
            expected = inventory_reference(work, position)
            # This bound isolates the final inventory from its already tested
            # freshly evaluated fractions. Each positive term has <=6 factors,
            # conversion and <=n-node protected additions; no cancellation occurs.
            n = size(work.populations.fractions, 1)
            u = Float64(eps(Float32)) / 2
            count = 16n + 32
            bound = count * u / (1 - count * u) * maximum(abs, expected)
            @test bound > 0
            @test maximum(abs, work.values[1:9] .- expected) <= bound
            @test work.values[5] == 0 && work.values[8] == 0
            @test all(isfinite, work.values)
            if kind == :field
                diagnostics = work.populations.diagnostics
                @test eltype(diagnostics.parts) === Float32
                @test eltype(diagnostics.scales) === Int
                @test size(diagnostics.scales) == (3, capacity)
                @test eltype(diagnostics.maxima) === Float64
                @test all(>=(0), work.values[11:13])
                @test iszero(work.values[13])
            end
            SolpropJulia.observe!(work, field, position; periodic = true)
            @test all(iszero, work.values[1:9]) && all(iszero, work.values[11:end])
            @test isfinite(work.values[10])
        end
        @test inventory_allocated(work, field) == 0
        ledger, seen = Dict{Symbol, Int}(), IdDict{Any, Bool}()
        SolpropJulia._memory_arrays!(ledger, seen, "observation", work)
        scales = work.statistics.scales
        @test haskey(seen, scales.columns) && haskey(seen, scales.nodes)
        expected_bytes = sizeof(scales.columns) + sizeof(scales.nodes)
        if kind == :field
            @test haskey(seen, work.populations.diagnostics.scales)
            expected_bytes += sizeof(work.populations.diagnostics.scales)
        end
        actual_bytes = sum(
            bytes for (name, bytes) in ledger if occursin(".scales", string(name))
        )
        @test actual_bytes == expected_bytes
    end
end

@testset "Spatial coefficient-tail conversion has an absolute weighted budget" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    arithmetic = SolpropJulia.SourceArithmetic(Float32)
    n = 8
    squared = ones(ComplexF64, n)
    squared[3] = 4.0e-100
    operator = SolpropJulia.SeparableSpatialOperator(
        execution; k_squared = squared, k_squared_low = zeros(ComplexF64, n),
        kperp_sq = [0.0], kperp_sq_low = [0.0], frame = zeros(n),
        spectral_rate = zeros(n), prefactor_numerator = zeros(ComplexF64, n),
        prefactor_clamped = zeros(ComplexF64, n), active = fill(true, n),
        evanescent_rate = 1.0, z_span = (0.0, 1.0),
    )
    grid = SolpropJulia.StatisticsGrid(
        collect(0.0:7.0), collect(0.0:7.0), fill(1 / n, n),
    )
    selection = SolpropJulia.StatisticsSelection(
        (
            SolpropJulia.StatisticSlot(:energy, 1, 1),
            SolpropJulia.StatisticSlot(:peak_intensity, 2, 1),
        ), ()
    )
    map = SolpropJulia.RadialObservationMap(ones(1, 1); pad_scale = 1.0, arithmetic)
    prepared = SolpropJulia.SpatialObservationPreparation(
        execution, grid, selection; kind = :envelope, map,
        flux = SolpropJulia.SpatialFluxData(operator; light_speed = 1.0),
        spectral_weights = [1.0], on_axis = ComplexF64[1.0], intensity_factor = 1.0,
    )
    work = SolpropJulia.SpatialObservationWorkspace(execution, prepared; node_chunk = 1)
    field = zeros(ComplexF32, n, 1)
    field[3, 1] = ldexp(1.0f0, 100)
    SolpropJulia.observe!(work, field, 0.25)
    @test work.weight_exponent == work.closure_exponent == 0
    wide = real(sqrt(squared[3])) / grid.omega[3]
    represented = Float32(wide)
    @test wide > 0 && represented == 0
    amplitude = BigFloat(real(field[3, 1]))
    expected = BigFloat(wide) * amplitude^2 .* [inv(big(n)), inv(big(n)^2)]
    @test all(value -> isfinite(value) && value > 0, expected)
    # Global coefficient narrowing allows this subnormal tail to vanish. Its
    # correctly rounded conversion error is <=eta32/2, despite unbounded relative
    # error. Propagate that absolute error through the actual occupied single-bin
    # field and the exact1/n transform factors. The represented-coefficient target
    # is zero; this does not assign a small relative error to the wide target.
    conversion = BigFloat(nextfloat(0.0f0)) / 2
    bound = conversion * amplitude^2 .* [inv(big(n)), inv(big(n)^2)]
    @test all(>(0), bound)
    @test abs(BigFloat(wide) - BigFloat(represented)) <= conversion
    @test all(abs.(BigFloat.(work.values) .- expected) .<= bound)
    @test all(iszero, work.values)
    @test iszero(work.weight[3, 1]) && iszero(work.closure[3, 1])
    @test spatial_observation_allocations(work, field) == 0
end
