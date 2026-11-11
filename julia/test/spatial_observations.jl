# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import LinearAlgebra
import SolpropJulia as SP

function spatial_observation_selection(
        names, frequencies, nodes; species = 0, periodic = (),
    )
    slots, offset = SP.StatisticSlot[], 1
    for name in names
        count = name == :on_axis_spectrum ? frequencies : name == :beam_radius ? 2 :
            name == :species_ionisation ? nodes * species :
            name == :species_electron_density ? species : 1
        push!(slots, SP.StatisticSlot(name, offset, count))
        offset += count
    end
    return SP.StatisticsSelection(Tuple(slots), periodic)
end

function spatial_observation_arrays(value::Tuple)
    return reduce(vcat, (spatial_observation_arrays(item) for item in value); init = Any[])
end
spatial_observation_arrays(value::AbstractArray) = Any[value]

function spatial_observation_monitor(
        execution, prepared; split, noise = nothing,
        continuation = nothing
    )
    regions = zeros(8, 4)
    regions[1:2, 1] .= 0.2
    regions[7:8, 2] .= 0.3
    regions[:, 3] .= 0.01
    regions[:, 4] .= collect(range(0.1, 0.8; length = 8))
    policy = SP.TemporalMonitorPolicy(;
        warn = 0.8, stop = 0.9, activity = 1.0, quiet_share = 0.1,
        quiet_checks = 2, quiet_collar = true, cadence = 3, interval = Inf,
        launch_floor_ratio = 1.0e-6, floor_margin = 1.2, split,
        crossing_share = 0.1, bucket_limits = [0.1, 0.4, Inf],
    )
    columns = prepared.n_modes
    buckets = [
        mod(row + column, 2) + 1 for row in eachindex(prepared.grid.omega),
            column in 1:columns
    ]
    weights = SP.TemporalMonitorWeights(
        regions, collect(range(0.5, 1.3; length = columns)); buckets,
    )
    data = spatial_observation_replace(
        execution, prepared; temporal = SP.TemporalObservationData(policy, weights),
    )
    return SP.SpatialObservationWorkspace(
        execution, data; node_chunk = 2, noise_spectrum = noise,
        temporal_continuation = continuation, continuation_plane = true,
    )
end

function spatial_observation_monitor_step(work, field, position, notices)
    SP.reset_notices!(notices)
    SP.capture_split!(work, field, position, 0.1)
    return SP.observation_accepted!(work, field, position, 0.1, notices)
end

function spatial_observation_replace(execution, prepared; changes...)
    options = (
        kind = prepared.kind, map = prepared.map, flux = prepared.flux,
        spectral_weights = prepared.spectral_weights, on_axis = prepared.host_on_axis,
        intensity_factor = prepared.intensity_factor, widths = prepared.widths,
        profiles = prepared.profiles, populations = prepared.populations,
        temporal = prepared.temporal, arithmetic = prepared.arithmetic,
    )
    return SP.SpatialObservationPreparation(
        execution, prepared.grid, prepared.selection; merge(options, (; changes...))...,
    )
end

function spatial_observation_cycle(; moment_rtol = 2.0e-8)
    nodes, weights, ends = Float64[], Float64[], Int[]
    for count in (16, 32)
        off = [index / sqrt(4index^2 - 1) for index in 1:(count - 1)]
        eigen = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(zeros(count), off))
        append!(nodes, eigen.values)
        append!(weights, 2 .* eigen.vectors[1, :] .^ 2)
        push!(ends, length(nodes))
    end
    return SP.CycleQuadrature(
        nodes, weights, Tuple(ends); moment_rtol, negligible_rate = 1.0e-30,
    )
end

function spatial_observation_populations(
        execution, prepared; ionising = true, moment_rtol = 2.0e-8,
    )
    kind, count = prepared.kind, length(prepared.grid.omega)
    arithmetic = prepared.arithmetic
    crop = kind == :field ? collect(0:4) : [0, 1, 2, 3, 12, 13, 14, 15]
    active = fill(true, count)
    active[1] = false
    fine = SP.ModalSourceGrid(
        execution; kind, n_time = 16, dt = 0.0625, crop_index = crop,
        omega = prepared.grid.omega, active, omega_window = ones(count),
        source_window = ones(16), source_window_derivative = zeros(16), arithmetic,
    )
    channels = ntuple(2) do index
        low, high, scale = 1.0e-12, 100.0, 0.4index
        table = SP.IonisationRateTable(
            reshape([0.0, 0.0, 2.0, log(scale) + 2log(low)], 4, 1),
            low, high, scale * high^2, log(low), log(high / low),
        )
        SP.ModalPlasmaChannel(
            SP.PlasmaChannel(table, 1.0, 0.2), SP.SourceProfile(100.0index),
            spatial_observation_profile(execution, 0.1index, 0.02);
            quadrature = kind == :field ? nothing :
                spatial_observation_cycle(; moment_rtol),
            arithmetic,
        )
    end
    plasma = !ionising ? nothing : kind == :field ?
        SP.ModalPlasma(
            execution, channels,
            SP.RunningIntegralRule([1.0], 0.0625; floor_exposure = false); arithmetic,
        ) : SP.ModalPlasma(
            execution, channels; drude = -Float64.(active), loss = -1.0im .* active,
            arithmetic,
        )
    species_densities = (
        SP.SourceProfile(3.0), spatial_observation_profile(execution, 4.0, 1.0),
        SP.SourceProfile(2.0),
    )
    channel_species = ionising ? (3, 1) : ()
    if prepared.map isa SP.RadialObservationMap
        maps = SP.ModalSourceMap(
            execution; synthesis = 2prepared.map.host_synthesis,
            projection = zeros(reverse(size(prepared.map.host_synthesis))), arithmetic,
        )
        return SP.ObservationPopulations(
            plasma, fine, maps; species_densities, channel_species, peak_species = 0,
        )
    end
    return SP.SpatialObservationPopulations(
        plasma, fine; species_densities, channel_species, peak_species = 0, pad_scale = 2.0,
    )
end

function spatial_observation_dft(input, shape; inverse = false)
    nx, ny = shape
    sign, normalization = inverse ? (1, inv(BigFloat(nx * ny))) : (-1, one(BigFloat))
    return ComplexF64[
        normalization * sum(
            Complex{BigFloat}(input[row, x + nx * y + 1]) *
                cis(sign * 2big(pi) * (BigFloat(ix * x) / nx + BigFloat(iy * y) / ny))
                for x in 0:(nx - 1), y in 0:(ny - 1)
        ) for row in axes(input, 1), ix in 0:(nx - 1), iy in 0:(ny - 1)
    ] |> array -> reshape(array, size(input))
end

function spatial_observation_analytic(input, n_time, kind)
    full = zeros(Complex{BigFloat}, n_time, size(input, 2))
    full[1:size(input, 1), :] .= input
    if kind == :field
        full[2:(size(input, 1) - 1), :] .*= 2
        full[1, :] .= real.(full[1, :])
        full[size(input, 1), :] .= real.(full[size(input, 1), :])
    end
    return ComplexF64[
        sum(
            full[row + 1, column] * cis(2big(pi) * row * time / n_time)
                for row in 0:(n_time - 1)
        ) / n_time for time in 0:(n_time - 1), column in axes(input, 2)
    ]
end

function spatial_observation_profile(execution, constant, slope)
    return SP.SourceProfile(
        SP.ChebyshevTable(
            execution, [0.0, 1.0], reshape([constant + slope / 2, slope / 2], 1, 2, 1),
        ),
    )
end

function spatial_observation_case(
        execution, kind, route; varying = false,
        names = (
            :energy, :peak_intensity, :on_axis_intensity, :on_axis_spectrum,
            :beam_radius, :fwhm_r, :density, :pressure, :core_radius, :quadrature_error,
        ), periodic = (1, 2, 7), arithmetic = SP.SourceArithmetic(Float64),
    )
    n_time, dt = 8, 0.125
    frequencies = kind == :field ? 5 : 8
    shape = route == :line ? (5, 1) : (3, 3)
    columns = route == :radial ? 4 : prod(shape)
    omega = collect(0.0:(frequencies - 1))
    parseval = fill(dt / n_time, frequencies)
    kind == :field && (parseval[2:(end - 1)] .*= 4)
    density = fill(dt^2 / (2pi), frequencies)
    kind == :field && (density .*= 4)
    grid = SP.StatisticsGrid(
        omega, collect((-4:3) .* dt), parseval;
        spectral_density = density, d_omega = 2pi / (dt * n_time),
    )
    high = ComplexF64[100 + row + 0.03im for row in 1:frequencies]
    base = SP.SeparableSpatialOperator(
        execution; k_squared = high, k_squared_low = fill(1.0e-14 + 0.0im, frequencies),
        kperp_sq = collect(range(0.0, 0.6; length = columns)),
        kperp_sq_low = fill(-1.0e-17, columns), frame = zeros(frequencies),
        spectral_rate = zeros(frequencies), prefactor_numerator = fill(-1.0im, frequencies),
        prefactor_clamped = fill(-0.1im, frequencies), active = fill(true, frequencies),
        evanescent_rate = 1.0, z_span = (0.0, 1.0),
    )
    material = varying ? SP.ChebyshevTable(
            execution, [0.0, 0.4, 1.0],
            cat(
                hcat(high, 0.1high, -0.02high), hcat(1.1high, -0.04high, 0.01high);
                dims = 3,
            ),
        ) : nothing
    flux = SP.SpatialFluxData(base; material, light_speed = 2.0)
    if route == :radial
        synthesis = [1.0 0.1 0.0 0.0; 0.1 0.8 0.1 0.0; 0.0 0.1 0.6 0.1; 0.0 0.0 0.1 0.4]
        map = SP.RadialObservationMap(2synthesis; pad_scale = 2.0, arithmetic)
        axis = ComplexF64[1.1, 0.1, 0.01, 0.0]
        widths = SP.SpatialWidthGrid(:radial, [0.1, 0.4, 0.7, 1.0], Float64[], ones(4))
    else
        map = SP.CartesianObservationMap(shape; arithmetic)
        axis = ComplexF64[
            cis(2pi * (div(shape[1], 2) * x / shape[1] + div(shape[2], 2) * y / shape[2])) /
                columns for x in 0:(shape[1] - 1), y in 0:(shape[2] - 1)
        ] |> vec
        x = collect(range(-1.0, 1.0; length = shape[1]))
        y = shape[2] == 1 ? [0.0] : collect(range(-1.0, 1.0; length = shape[2]))
        widths = SP.SpatialWidthGrid(route, x, y, ones(columns))
    end
    physical = ComplexF64[
        (
            kind == :field && row in (1, frequencies) ? 0.0 :
                exp(-0.4(row - 2)^2) * cis(0.3row)
        ) * exp(-0.15(column - 1)^2)
            for row in 1:frequencies, column in 1:columns
    ]
    field = route == :radial ? physical : spatial_observation_dft(physical, shape)
    selection = spatial_observation_selection(names, frequencies, columns; periodic)
    prepared = SP.SpatialObservationPreparation(
        execution, grid, selection; kind, map, flux,
        spectral_weights = ones(columns), on_axis = axis, intensity_factor = 0.7, widths,
        profiles = SP.ObservationProfiles(
            density = spatial_observation_profile(execution, 2.0, 0.3),
            pressure = SP.SourceProfile(0.8),
            core_radius = spatial_observation_profile(execution, 1.0, -0.1),
        ),
    )
    return prepared, field
end

function spatial_observation_slice(work, name)
    index = findfirst(
        slot -> SP._STATISTIC_NAMES[slot.code] == name, work.data.selection.slots,
    )
    slot = work.data.selection.slots[index]
    return work.values[slot.offset:(slot.offset + slot.count - 1)]
end

function spatial_observation_reference(prepared, field, z)
    base, grid = prepared.flux.base, prepared.grid
    high = copy(base.k_squared)
    if prepared.flux.material !== nothing
        table = prepared.flux.material
        interval = min(
            searchsortedlast(table.breakpoints, z), length(table.breakpoints) - 1,
        )
        x = (2BigFloat(z) - table.breakpoints[interval] - table.breakpoints[interval + 1]) /
            (table.breakpoints[interval + 1] - table.breakpoints[interval])
        for row in eachindex(high)
            coefficients = table.coefficients[row, :, interval]
            high[row] = coefficients[1] + coefficients[2] * x +
                coefficients[3] * (2x^2 - 1)
            base.active[row] || (high[row] = 0)
        end
    end
    admittance = zeros(ComplexF64, size(field))
    for column in axes(field, 2), row in axes(field, 1)
        q = (high[row] - base.kperp_sq[column]) +
            (base.k_squared_low[row] - base.kperp_sq_low[column])
        if base.active[row] && grid.omega[row] > 0 && real(q) > 0
            admittance[row, column] = prepared.flux.light_speed *
                sqrt(Complex{BigFloat}(q)) /
                grid.omega[row]
        end
    end
    closure = conj.(admittance) .* field
    if prepared.map isa SP.RadialObservationMap
        electric = field * transpose(prepared.map.synthesis)
        magnetic = closure * transpose(prepared.map.synthesis)
        contribution = abs.(field) * transpose(abs.(prepared.map.synthesis))
        magnetic_contribution = abs.(closure) * transpose(abs.(prepared.map.synthesis))
    else
        electric = spatial_observation_dft(field, prepared.map.shape; inverse = true)
        magnetic = spatial_observation_dft(closure, prepared.map.shape; inverse = true)
        contribution = repeat(
            sum(abs.(field); dims = 2) ./ prepared.n_modes, 1, prepared.n_modes,
        )
        magnetic_contribution = repeat(
            sum(abs.(closure); dims = 2) ./ prepared.n_modes, 1, prepared.n_modes,
        )
    end
    axis = field * prepared.on_axis
    axis_closure = closure * prepared.on_axis
    analytic = spatial_observation_analytic(electric, length(grid.time), prepared.kind)
    analytic_closure = spatial_observation_analytic(
        magnetic, length(grid.time), prepared.kind,
    )
    analytic_axis = spatial_observation_analytic(
        reshape(axis, :, 1), length(grid.time), prepared.kind,
    )
    analytic_axis_closure = spatial_observation_analytic(
        reshape(axis_closure, :, 1), length(grid.time), prepared.kind,
    )
    energy = sum(
        prepared.intensity_factor .* prepared.spectral_weights' .* real.(admittance) .*
            abs2.(field) .* grid.parseval,
    )
    axis_spectrum = prepared.intensity_factor .* grid.spectral_density .*
        real.(axis .* conj.(axis_closure))
    peak = prepared.intensity_factor * maximum(real.(analytic .* conj.(analytic_closure)))
    axis_peak = prepared.intensity_factor *
        maximum(real.(analytic_axis .* conj.(analytic_axis_closure)))
    # Absolute contributions before either spatial/temporal cancellation. The
    # fixtures' roots are separated from grazing; a fixed 4096eps budget covers
    # represented polynomial/root evaluation and all short DFT/products/reductions.
    first_bound = 2sum(contribution; dims = 1) ./ length(grid.time)
    second_bound = 2sum(magnetic_contribution; dims = 1) ./ length(grid.time)
    axis_bound = sum(abs.(field) * abs.(prepared.on_axis))
    magnetic_axis_bound = sum(abs.(closure) * abs.(prepared.on_axis))
    bound = prepared.intensity_factor * max(
        sum(contribution .* magnetic_contribution .* grid.parseval),
        maximum(first_bound .* second_bound),
        4axis_bound * magnetic_axis_bound / length(grid.time)^2,
        maximum(grid.spectral_density) * axis_bound * magnetic_axis_bound,
    )
    return (;
        energy, peak, axis_peak, axis_spectrum, electric, magnetic, analytic,
        analytic_closure, bound,
    )
end

function spatial_observation_allocations(work, field; periodic = false)
    SP.observe!(work, field, 0.25; periodic, quadrature_error = 0.01)
    SP.observe!(work, field, 0.25; periodic, quadrature_error = 0.01)
    return @allocated SP.observe!(work, field, 0.25; periodic, quadrature_error = 0.01)
end

@testset "Spatial observations reconstruct current electric and magnetic fields" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))),
            kind in (:field, :envelope), route in (:radial, :cartesian, :line)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        prepared, field = spatial_observation_case(
            execution, kind, route;
            varying = route == :radial
        )
        work = SP.ObservationWorkspace(execution, prepared; node_chunk = 2)
        saved = copy(field)
        for position in (0.0, 0.25, 0.4, 0.9, 0.25)
            reference = spatial_observation_reference(prepared, field, position)
            SP.observe!(work, field, position; quadrature_error = 0.01)
            bound = 4096eps(Float64) * reference.bound
            @test bound > 0
            @test work.valid && work.z == position && field == saved
            @test abs(
                only(spatial_observation_slice(work, :energy)) - reference.energy,
            ) <= bound
            @test abs(
                only(spatial_observation_slice(work, :peak_intensity)) - reference.peak,
            ) <= bound
            @test abs(
                only(spatial_observation_slice(work, :on_axis_intensity)) -
                    reference.axis_peak
            ) <= bound
            @test maximum(
                abs, spatial_observation_slice(work, :on_axis_spectrum) -
                    reference.axis_spectrum
            ) <= bound
            @test maximum(abs, work.physical.spectrum - reference.electric) <=
                512eps(Float64) * sum(abs, field)
            @test maximum(abs, work.physical.closure - reference.magnetic) <=
                1024eps(Float64) * sum(abs, work.closure)
            @test abs(
                only(spatial_observation_slice(work, :density)) - (2 + 0.3position),
            ) <= 8eps(Float64)
            plane = copy(work.values)
            SP.observe!(work, field, position; periodic = true, quadrature_error = 0.01)
            for (index, slot) in enumerate(prepared.selection.slots)
                rows = slot.offset:(slot.offset + slot.count - 1)
                if index in prepared.selection.periodic
                    @test maximum(abs, work.values[rows] - plane[rows]) <= bound
                else
                    @test iszero(work.values[rows])
                end
            end
        end
        @test spatial_observation_allocations(work, field) == 0
        @test spatial_observation_allocations(work, field; periodic = true) == 0
        fill!(work.physical.pair.samples, ComplexF64(NaN))
        fill!(work.axis.pair.weighted, ComplexF64(NaN))
        SP.observe!(work, zero(field), 0.25)
        @test iszero(spatial_observation_slice(work, :energy))
        @test iszero(spatial_observation_slice(work, :peak_intensity))
        @test all(isnan, spatial_observation_slice(work, :beam_radius))
    end
end

@testset "Spatial populations use fresh physical nodes and independent species" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))),
            kind in (:field, :envelope),
            route in (:radial, :cartesian, :line), ionising in (false, true)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        prepared, field = spatial_observation_case(
            execution, kind, route; names = (:density,), periodic = (),
        )
        populations = spatial_observation_populations(execution, prepared; ionising)
        nodes = SP._spatial_map_shape(prepared.map)[1]
        selection = spatial_observation_selection(
            (
                :species_ionisation, :species_electron_density, :electron_density,
                :peak_ionisation_rate,
            ),
            length(prepared.grid.omega), nodes; species = 3, periodic = (1, 3),
        )
        prepared = SP.SpatialObservationPreparation(
            execution, prepared.grid, selection;
            kind, map = prepared.map, flux = prepared.flux,
            spectral_weights = prepared.spectral_weights, on_axis = prepared.on_axis,
            intensity_factor = prepared.intensity_factor, populations,
        )
        previous = nothing
        for capacity in (1, 2, nodes)
            work = SP.SpatialObservationWorkspace(
                execution, prepared; node_chunk = capacity,
            )
            @test work.physical === nothing && work.axis === nothing &&
                isempty(work.closure)
            position = 0.25
            SP.observe!(work, field, position)
            @test work.populations.densities == [3.0, 4.25, 2.0]
            @test iszero(work.populations.fractions[:, 2])
            if ionising
                physical = prepared.map isa SP.RadialObservationMap ?
                    field * transpose(prepared.map.synthesis) :
                    spatial_observation_dft(field, prepared.map.shape; inverse = true)
                padded = zeros(ComplexF64, kind == :field ? 9 : 16, nodes)
                padded[populations.grid.crop_index .+ 1, :] .= 2physical
                analytic = spatial_observation_analytic(padded, 16, kind)
                samples = kind == :field ? real.(analytic) : analytic
                tolerance = kind == :field ? 1024eps(Float64) : 4.0e-8
                for (index, species) in enumerate((3, 1))
                    rate = (kind == :field ? 0.4index : 0.2index) .* abs2.(samples)
                    exposure = 0.0625 .* vec(
                        sum(rate[2:15, :]; dims = 1) .+
                            0.5 .* (rate[1:1, :] .+ rate[16:16, :]),
                    )
                    initial = 0.1index + 0.02position
                    expected = initial .+ (1 - initial) .* -expm1.(-exposure)
                    @test isapprox(
                        work.populations.fractions[:, species], expected;
                        rtol = tolerance, atol = tolerance
                    )
                    @test isapprox(
                        work.populations.peaks[:, species],
                        vec(maximum(rate; dims = 1)); rtol = tolerance, atol = tolerance
                    )
                end
                remainder = mod(nodes, capacity)
                if remainder != 0
                    @test iszero(work.populations.nodes[:, (remainder + 1):end])
                end
            else
                @test iszero(work.populations.fractions) && iszero(work.populations.peaks)
            end
            @test spatial_observation_allocations(work, field) == 0
            previous === nothing ||
                @test isapprox(work.values, previous; rtol = 256eps(Float64), atol = 0.0)
            previous = copy(work.values)
            SP.observe!(work, zero(field), 0.25)
            @test iszero(work.populations.peaks)
            if ionising
                @test all(abs.(work.populations.fractions[:, 1] .- 0.205) .<= 2eps(Float64))
                @test all(abs.(work.populations.fractions[:, 3] .- 0.105) .<= 2eps(Float64))
            end
        end
    end
end

@testset "Spatial observation ownership, storage and admission" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope), route in (:radial, :cartesian, :line)
        prepared, field = spatial_observation_case(execution, kind, route)
        work = SP.SpatialObservationWorkspace(execution, prepared; node_chunk = 2)
        SP.observe!(work, field, 0.2)
        before = copy(work.values)
        @test_throws DimensionMismatch SP.observe!(work, real.(field), 0.2)
        @test_throws DimensionMismatch SP.observe!(work, field[:, 1:1], 0.2)
        @test_throws ArgumentError SP.observe!(work, field, Inf)
        nonfinite = copy(field)
        nonfinite[1] = NaN
        @test_throws ArgumentError SP.observe!(work, nonfinite, 0.2)
        for retained in (work.spectrum, work.closure, work.physical.spectrum)
            @test_throws ArgumentError SP.observe!(work, retained, 0.2)
        end
        @test isequal(work.values, before) && work.valid
        # Distinct Array headers can overlap even when Base.mightalias says false.
        # Root a larger retained complex buffer while borrowing an offset header.
        owner = work.physical.pair.spectrum
        if length(owner) > length(field)
            GC.@preserve owner begin
                foreign = unsafe_wrap(Array, pointer(owner, 2), size(field); own = false)
                @test !Base.mightalias(foreign, owner)
                @test_throws ArgumentError SP.observe!(work, foreign, 0.2)
                @test isequal(work.values, before)
            end
        end
        parent = zeros(ComplexF64, 2size(field, 1), size(field, 2))
        strided = view(parent, 1:2:size(parent, 1), :)
        strided .= field
        SP.observe!(work, strided, 0.2)
        @test isequal(work.values, before)
        arrays = spatial_observation_arrays(work.buffers)
        @test all(
            array -> any(item -> item === array, arrays), (
                work.spectrum, work.closure, work.weight, work.values, work.material,
                work.physical.spectrum, work.physical.closure,
                work.physical.pair.samples, work.physical.pair.weighted,
                work.axis.spectrum, work.axis.closure, work.axis.pair.samples,
                work.statistics.fluence, work.statistics.radial_average,
            )
        )
        another = SP.SpatialObservationWorkspace(execution, prepared; node_chunk = 2)
        @test all(array -> isempty(array) || !SP._retained_alias(array, another), arrays)
        overflowing = spatial_observation_replace(
            execution, prepared;
            flux = SP.SpatialFluxData(prepared.flux.base; light_speed = floatmax(Float64)),
        )
        failed = SP.SpatialObservationWorkspace(execution, overflowing; node_chunk = 2)
        @test_throws ArgumentError SP.observe!(failed, field, 0.2)
        @test !failed.valid && isnan(failed.z)
        @test_throws ArgumentError SP.SpatialObservationWorkspace(
            execution, prepared; node_chunk = 0,
        )
        @test_throws ArgumentError spatial_observation_replace(
            execution, prepared; spectral_weights = zeros(prepared.n_modes),
        )
        @test_throws ArgumentError spatial_observation_replace(
            execution, prepared; on_axis = ones(prepared.n_modes),
        )
        @test_throws ArgumentError spatial_observation_replace(
            execution, prepared; intensity_factor = NaN,
        )
        @test_throws ArgumentError spatial_observation_replace(
            execution, prepared; kind = :vector,
        )
        for name in (:omega0, :peak_power_total, :physical_energy)
            selection = SP.StatisticsSelection((SP.StatisticSlot(name, 1, 1),))
            @test_throws ArgumentError SP.SpatialObservationPreparation(
                execution, prepared.grid, selection; kind, map = prepared.map,
                flux = prepared.flux, spectral_weights = prepared.spectral_weights,
                on_axis = prepared.on_axis, intensity_factor = 0.7,
            )
        end
    end
    prepared, field = spatial_observation_case(execution, :field, :radial)
    @test_throws ArgumentError SP.RadialObservationMap(ones(2, 2); pad_scale = 0.0)
    @test_throws ArgumentError SP.RadialObservationMap(ones(Float32, 2, 2); pad_scale = 1.0)
    @test_throws ArgumentError SP.CartesianObservationMap((1, 2))
    @test_throws ArgumentError SP.SpatialFluxData(prepared.flux.base; light_speed = -1.0)
    table = SP.ChebyshevTable(execution, [0.0, 1.0], ones(5, 1, 1))
    @test_throws ArgumentError SP.SpatialFluxData(
        prepared.flux.base; material = table,
        light_speed = 1.0
    )
    bad = SP.ChebyshevTable(execution, [0.0, 1.0], fill(ComplexF64(1.0e308), 5, 3, 1))
    data = spatial_observation_replace(
        execution, prepared;
        flux = SP.SpatialFluxData(prepared.flux.base; material = bad, light_speed = 1.0)
    )
    work = SP.SpatialObservationWorkspace(execution, data; node_chunk = 2)
    @test_throws ArgumentError SP.observe!(work, field, 1.0)
    @test !work.valid
end

@testset "Spatial admittance keeps represented low-word branch decisions" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 2, true)
    prepared, field = spatial_observation_case(
        execution, :envelope, :line;
        names = (:energy,), periodic = ()
    )
    frequencies, columns = size(field)
    high = fill(1.0 + 0.0im, frequencies)
    low = ComplexF64[0.0, 2.0^-55, -2.0^-55, 2.0^-80, -2.0^-80, 0.0, 0.0, 0.0]
    active = fill(true, frequencies)
    active[4] = false
    operator = SP.SeparableSpatialOperator(
        execution;
        k_squared = high, k_squared_low = low, kperp_sq = ones(columns),
        kperp_sq_low = zeros(columns), frame = zeros(frequencies),
        spectral_rate = zeros(frequencies), prefactor_numerator = fill(-1.0im, frequencies),
        prefactor_clamped = fill(-0.1im, frequencies), active,
        evanescent_rate = 1.0, z_span = (0.0, 1.0)
    )
    data = spatial_observation_replace(
        execution, prepared;
        flux = SP.SpatialFluxData(operator; light_speed = 2.0)
    )
    work = SP.SpatialObservationWorkspace(execution, data; node_chunk = 2)
    SP.observe!(work, field, 0.25)
    @test all(iszero, work.weight[[1, 3, 4, 5, 6, 7, 8], :])
    expected = Float64(2sqrt(BigFloat(2)^-55)) * data.intensity_factor
    @test all(abs.(work.weight[2, :] .- expected) .<= 4eps(expected))
    @test all(work.weight[2, :] .> 0)
end

@testset "Spatial temporal monitors preserve the canonical-spectrum proxy" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))),
            kind in (:field, :envelope),
            route in (:radial, :cartesian, :line), split in (false, true)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        prepared, field = spatial_observation_case(
            execution, kind, route;
            names = (), periodic = ()
        )
        work = spatial_observation_monitor(execution, prepared; split, noise = 0.1field)
        notices = SP.MonitorNotices(8)
        row = SP.observation_launch!(work, field, 0.0, notices)
        @test row.emit
        analytic = spatial_observation_analytic(field, 8, kind)
        samples = kind == :field ? real.(analytic) : analytic
        columns = work.data.temporal.weights.columns
        power = vec(sum(abs2.(samples) .* columns'; dims = 2))
        @test isapprox(work.monitor.launch, sum(power); rtol = 256eps(Float64), atol = 0.0)
        @test isapprox(
            work.monitor.floor, 0.01sum(power); rtol = 256eps(Float64), atol = 0.0,
        )
        @test work.physical === nothing && work.axis === nothing &&
            work.populations === nothing
        @test_throws ArgumentError SP.observation_launch!(work, field, 0.0, notices)
        saved = copy(field)
        for step in 1:4
            spatial_observation_monitor_step(work, field, 0.1step, notices)
        end
        @test field == saved && work.monitor.committed == 0.4
        counts = (work.monitor.steps, work.monitor.skipped)
        SP.observation_check!(work, field, 0.4, notices)
        @test (work.monitor.steps, work.monitor.skipped) == counts
        @test SP.observation_plane!(work, 0.4).emit
        @test isnan(SP.observation_plane!(work, 0.35).collar)
        continuation = SP.temporal_continuation(work.monitor)
        restored = spatial_observation_monitor(
            execution, prepared; split,
            noise = 0.1field, continuation
        )
        SP.observation_launch!(restored, field, 0.4, notices)
        @test restored.monitor.steps == work.monitor.steps
        @test restored.monitor.raw_removed == work.monitor.raw_removed
        @test !SP.observation_plane!(restored, 0.4).emit
        spatial_observation_monitor_step(work, field, 0.5, notices)
        @test (@allocated spatial_observation_monitor_step(work, field, 0.6, notices)) == 0
        @test SP.observation_finish!(work, notices) === nothing
        @test_throws ArgumentError SP.observation_check!(work, field, 0.4, notices)
        if split
            SP.capture_split!(work, field, 0.7, 0.1)
            @test_throws ArgumentError SP.observation_finish!(work, notices)
        end
    end
end

@testset "Spatial population failures and retained fine storage" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope), route in (:radial, :cartesian, :line)
        prepared, field = spatial_observation_case(
            execution, kind, route; names = (:density,), periodic = (),
        )
        populations = spatial_observation_populations(execution, prepared)
        selection = SP.StatisticsSelection((SP.StatisticSlot(:electron_density, 1, 1),))
        data = SP.SpatialObservationPreparation(
            execution, prepared.grid, selection; kind, map = prepared.map,
            flux = prepared.flux, spectral_weights = prepared.spectral_weights,
            on_axis = prepared.on_axis, intensity_factor = prepared.intensity_factor,
            populations,
        )
        work = SP.SpatialObservationWorkspace(
            execution, data; node_chunk = prepared.n_modes,
        )
        SP.observe!(work, field, 0.25)
        saved = copy(work.values)
        arrays = spatial_observation_arrays(work.buffers)
        @test all(
            array -> any(item -> item === array, arrays), (
                work.populations.padding.crop, work.populations.padding.uncrop,
                work.populations.spectrum, work.populations.nodes,
                work.populations.fractions, work.populations.peaks,
            )
        )
        owner = work.populations.spectrum
        @test length(owner) > length(field)
        GC.@preserve owner begin
            foreign = unsafe_wrap(Array, pointer(owner, 2), size(field); own = false)
            @test_throws ArgumentError SP.observe!(work, foreign, 0.25)
            @test work.values == saved && work.valid
        end
        # Corrupted retained rate data reach the typed scan failure; no values
        # from that partial scan may become a published observation.
        coefficients = populations.plasma.channels[1].channel.rate.coefficients
        coefficients[:, 1] .= [0.0, 0.0, 0.0, NaN]
        @test_throws SP.SourcePlasmaError SP.observe!(work, field, 0.25)
        @test !work.valid && work.z == 0.25
    end
end
