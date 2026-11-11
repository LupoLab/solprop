# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The aggregate includes the existing observation fixture modules before this file.
function inventory_selection(kind; diagnostics = kind == :field)
    names = (
        :species_density, :species_ionisation_energy, :species_peak_ionisation_rate,
        :electron_density,
    )
    diagnostics && (
        names = (
            names..., :plasma_undershoot, :plasma_work_defect,
            :plasma_boundary_level,
        )
    )
    offset = 1
    slots = map(names) do name
        count = startswith(string(name), "species_") ? 3 : 1
        slot = SolpropJulia.StatisticSlot(name, offset, count)
        offset += count
        slot
    end
    return SolpropJulia.StatisticsSelection(slots, (4,))
end

function inventory_populations(execution, populations, nodes; modal)
    measure = SolpropJulia.PopulationMeasure(
        collect(range(0.1, 0.3; length = nodes));
        area = observation_profile(execution, 1.0, 0.2),
        taper = modal ? populations.map.taper : SolpropJulia.SourceProfile(1.0)
    )
    options = (;
        species_densities = populations.species_densities,
        channel_species = populations.channel_species,
        peak_species = populations.peak_species,
        potentials = [0.4, 0.0, 0.2], measure,
    )
    if populations isa SolpropJulia.ObservationPopulations
        return SolpropJulia.ObservationPopulations(
            populations.plasma, populations.grid,
            populations.map; options...
        )
    end
    return SolpropJulia.SpatialObservationPopulations(
        populations.plasma, populations.grid;
        options..., pad_scale = populations.pad_scale
    )
end

function inventory_reference(work, position)
    populations, data = work.populations, work.data.populations
    densities = BigFloat.(populations.densities)
    fractions = BigFloat.(populations.fractions)
    peaks = vec(maximum(populations.peaks; dims = 1))
    factor = BigFloat(SolpropJulia._source_profile(data.measure.area, position)) /
        BigFloat(SolpropJulia._source_profile(data.measure.taper, position))^2
    weights = BigFloat.(data.measure.weights) .* factor
    energy = [
        sum(
            weights[node] * densities[species] * fractions[node, species] *
                BigFloat(data.potentials[species]) for node in axes(fractions, 1)
        ) for species in 1:3
    ]
    return Float64.([densities; energy; peaks])
end

function inventory_allocated(work, field)
    SolpropJulia.observe!(work, field, 0.25)
    return @allocated SolpropJulia.observe!(work, field, 0.25)
end

@testset "Fresh modal default plasma diagnostics and species inventory" begin
    for tasks in (1, 2), kind in (:field, :envelope)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(),
            tasks, true
        )
        original = observation_population_case(execution, kind)
        data = original.data
        populations = inventory_populations(execution, data.populations, 3; modal = true)
        selection = inventory_selection(kind)
        prepared = SolpropJulia.ObservationPreparation(
            execution, data.grid, selection;
            kind, n_modes = data.n_modes, power_weight = data.power_weight, populations
        )
        previous = nothing
        for chunk in (1, 2, 3)
            work = SolpropJulia.ObservationWorkspace(
                execution, prepared; node_chunk = chunk,
            )
            field = observation_field(kind)
            for position in (0.25, 0.7)
                SolpropJulia.observe!(work, field, position)
                reference = inventory_reference(work, position)
                # This witness isolates the inventory reduction of freshly computed
                # populations: <= six products per node plus a three-term sum.
                bound = 32eps(Float64) * maximum(abs, reference)
                @test bound > 0
                @test maximum(abs, work.values[1:9] - reference) <= bound
                @test iszero(work.values[5]) && iszero(work.values[8])
                @test work.populations.densities[2] > 0
                if kind == :field
                    @test work.populations.diagnostics isa
                        SolpropJulia.PlasmaDiagnosticWorkspace
                    @test size(work.populations.diagnostics.parts) == (6, chunk)
                    @test length(work.populations.diagnostics.maxima) == 6
                    @test all(isfinite, work.values[11:13])
                    @test iszero(work.values[13])
                else
                    @test work.populations.diagnostics === nothing
                end
                plane = copy(work.values)
                SolpropJulia.observe!(work, field, position; periodic = true)
                @test all(iszero, work.values[1:9])
                @test work.values[10] == plane[10]
                @test all(iszero, work.values[11:end])
                SolpropJulia.observe!(work, field, position)
                # Independent chunk partitions perform the same scalar rate and
                # population operations; only reductions group differently.
                @test isapprox(work.values, plane; rtol = 32eps(Float64), atol = bound)
            end
            @test inventory_allocated(work, field) == 0
            previous === nothing || @test isapprox(
                work.values, previous;
                rtol = 64eps(Float64), atol = 64eps(Float64)
            )
            previous = copy(work.values)
            if kind == :field
                fill!(work.populations.diagnostics.parts, NaN)
                fill!(work.populations.diagnostics.maxima, NaN)
            end
            fill!(work.populations.workspace.fraction, NaN)
            SolpropJulia.observe!(work, zero(field), 0.25)
            @test all(iszero, work.values[7:9])
            @test all(>=(0), work.values)
            @test work.values[4] > 0 && work.values[6] > 0
        end
    end
end

@testset "Spatial inventory uses physical node measure on every admitted route" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope), route in (:radial, :cartesian, :line)
        original, field = spatial_observation_case(
            execution, kind, route;
            names = (:density,), periodic = ()
        )
        base = spatial_observation_populations(execution, original)
        nodes = SolpropJulia._spatial_map_shape(original.map)[1]
        populations = inventory_populations(
            execution, base, nodes;
            modal = base isa SolpropJulia.ObservationPopulations
        )
        prepared = SolpropJulia.SpatialObservationPreparation(
            execution, original.grid,
            inventory_selection(kind); kind, map = original.map, flux = original.flux,
            spectral_weights = original.spectral_weights, on_axis = original.on_axis,
            intensity_factor = original.intensity_factor, populations
        )
        previous = nothing
        for chunk in (1, 2, nodes)
            work = SolpropJulia.SpatialObservationWorkspace(
                execution, prepared;
                node_chunk = chunk
            )
            SolpropJulia.observe!(work, field, 0.25)
            reference = inventory_reference(work, 0.25)
            # Five products per node and a positive running-node sum, plus the
            # represented scalar area/taper evaluation.
            bound = 16nodes * eps(Float64) * maximum(abs, reference)
            @test bound > 0
            @test maximum(abs, work.values[1:9] - reference) <= bound
            @test iszero(work.values[5]) && iszero(work.values[8])
            @test inventory_allocated(work, field) == 0
            previous === nothing || @test isapprox(
                work.values, previous;
                rtol = 32nodes * eps(Float64), atol = bound
            )
            previous = copy(work.values)
            SolpropJulia.observe!(work, field, 0.25; periodic = true)
            @test all(iszero, work.values[1:9]) && all(iszero, work.values[11:end])
        end
    end
end

@testset "Inventory and diagnostic admission is explicit" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    @test_throws ArgumentError SolpropJulia.PopulationMeasure([-1.0])
    weights = [1.0, 2.0, 3.0]
    measure = SolpropJulia.PopulationMeasure(weights)
    weights[1] = 10.0
    @test measure.weights == [1.0, 2.0, 3.0]
    original = observation_population_case(execution, :field)
    data = original.data
    @test_throws ArgumentError SolpropJulia.ObservationPreparation(
        execution, data.grid,
        inventory_selection(:field); kind = :field, n_modes = data.n_modes,
        power_weight = data.power_weight, populations = data.populations
    )
    original = observation_population_case(execution, :envelope)
    data = original.data
    populations = inventory_populations(execution, data.populations, 3; modal = true)
    @test_throws ArgumentError SolpropJulia.ObservationPreparation(
        execution, data.grid,
        inventory_selection(:envelope; diagnostics = true); kind = :envelope,
        n_modes = data.n_modes, power_weight = data.power_weight, populations
    )
end
