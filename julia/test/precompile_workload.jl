# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

@testset "Optional observation records refuse before nullable dereferences" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    grid = SP.StatisticsGrid(collect(1.0:8), collect(0.0:7.0), ones(8))
    weight = SP.ObservationWeight(execution, ones(8, 1))
    for name in (:species_ionisation, :species_electron_density)
        selection = SP.StatisticsSelection((SP.StatisticSlot(name, 1, 1),))
        message = "selected population statistics need prepared populations"
        @test_throws message SP.ObservationPreparation(
            execution, grid, selection;
            kind = :envelope, n_modes = 1, power_weight = weight,
        )
    end
    selected = SP.StatisticsSelection((SP.StatisticSlot(:peak_intensity, 1, 1),))
    valid = SP.ObservationPreparation(
        execution, grid, selected;
        kind = :envelope, n_modes = 1, power_weight = weight,
        axis_synthesis = ones(1, 1), intensity_factor = 1.0,
    )
    @test SP.ObservationWorkspace(execution, valid; node_chunk = 1).axis !== nothing
    # The positional constructor can describe an inconsistent internal record.
    # Refuse the absent map by name instead of indexing/sizing nothing.
    invalid = SP.ObservationPreparation(
        valid.kind, valid.n_modes, valid.grid, valid.selection, valid.power_weight,
        valid.profiles, nothing, valid.taper, valid.intensity_factor,
        valid.populations, valid.temporal, valid.needs, valid.periodic_needs,
    )
    before = copy(weight.data)
    message = "peak_intensity needs a separate coarse on-axis map"
    @test_throws message SP.ObservationWorkspace(
        execution, invalid; node_chunk = 1,
    )
    @test weight.data == before
end

@testset "An absent temporal rate refuses before touching source scratch" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    samples, scratch = ones(8, 1), fill(7.0, 8, 1)
    output = fill(2.0 + 3.0im, 8, 1)
    source = (
        execution = execution, data = (grid = (temporal_rate = nothing,),),
        modal_time = samples, projection_work = scratch,
    )
    @test_throws "an active temporal source requires a prepared rate" SP._source_temporal!(
        output, source,
    )
    @test samples == ones(8, 1)
    @test scratch == fill(7.0, 8, 1)
    @test output == fill(2.0 + 3.0im, 8, 1)
end

@testset "Statistic tuple boundaries retain exact slot and slice metadata" begin
    first = SP.StatisticSlot(:energy, 1, 1)
    second = SP.StatisticSlot(:peak_power_total, 2, 1)
    @test SP.StatisticsSelection((first, second), (1, 2)).width == 2
    @test SP.StatisticsSelection(()).width == 0
    @test_throws "duplicate statistic in selection" SP.StatisticsSelection(
        (first, SP.StatisticSlot(:energy, 2, 1)),
    )
    destination, source = fill(7.0, 2, 1), [2.0, 3.0]
    SP._observation_copy_slices!(destination, source, ((2, 1, 1), (1, 2, 1)), 1)
    @test destination == reshape([3.0, 2.0], 2, 1)
    before = copy(destination)
    @test_throws MethodError SP._observation_copy_slices!(
        destination, source, ((1, 1, 1), (1, true, 1)), 1,
    )
    @test destination == before
    # Unchecked positional record construction must not make a Boolean slot offset
    # reach an output write. The documented boundary requires StatisticSlot records.
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    invalid = SP.StatisticsSelection(((code = UInt8(20), offset = true, count = 1),), (), 1)
    grid = SP.StatisticsGrid([1.0], [0.0], [1.0])
    workspace = SP.StatisticsWorkspace(execution, Float64; n_time = 1, n_columns = 1)
    output = [7.0]
    @test_throws MethodError SP.statistics!(
        output, invalid, grid, SP.StatisticsInputs(), workspace, execution,
    )
    @test output == [7.0]
end

@testset "Missing analytic field storage has a named refusal" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    data = (
        grid = (kind = :field,),
        kerr = (coefficients = SP.KerrCoefficients(1.0, :analytic; take_real = true),),
    )
    source = (
        data = data, execution = execution, nodes = ones(8, 1),
        polarisation = zeros(8, 1), analytic = nothing,
    )
    message = "analytic field Kerr requires prepared analytic storage"
    @test_throws message SP._source_material!(source, 1.0, 1)
    @test source.nodes == ones(8, 1)
end
