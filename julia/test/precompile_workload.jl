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
        valid.populations, valid.temporal, valid.axis_populations, valid.needs,
        valid.periodic_needs, valid.arithmetic, valid.host_axis_synthesis, valid.ranges,
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
        kerr = SP.ModalKerr(
            SP.KerrCoefficients(1.0, :analytic; take_real = true), SP.SourceProfile(1.0)
        ),
    )
    source = (
        data = data, execution = execution, nodes = ones(8, 1),
        polarisation = zeros(8, 1), analytic = nothing,
    )
    message = "analytic field Kerr requires prepared analytic storage"
    @test_throws message SP._source_material!(source, 1.0, 1)
    @test source.nodes == ones(8, 1)
end

function precompile_process_state()
    return (
        environment = Dict(ENV), directory = pwd(),
        threads = (Threads.nthreads(:default), Threads.nthreads(:interactive)),
        fft_threads = SP.FFTW.get_num_threads(),
        blas_threads = SP.LinearAlgebra.BLAS.get_num_threads(),
        load_path = copy(LOAD_PATH), depot_path = copy(DEPOT_PATH),
    )
end

@noinline function precompile_owner_references()
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    session = SP._precompile_session(execution, Val(:rk), Val(:field))
    refs = (
        WeakRef(session), WeakRef(session.field), WeakRef(session.source),
        WeakRef(session.services), WeakRef(session.flow),
    )
    SP._precompile_drain!(session)
    return refs
end

@testset "Precompile workload runs three bounded real session paths" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    for (method, kind, family) in (
            (:rk, :field, SP.RungeKuttaSession),
            (:adams, :envelope, SP.AdamsSession),
            (:low_storage, :envelope, SP.LowStorageSession),
        )
        session = SP._precompile_session(execution, Val(method), Val(kind))
        @test session isa family
        @test session.source.data.grid.kind == kind
        @test session.source.data.kerr !== nothing
        @test length(session.source.data.raman) == 1
        @test length(session.source.data.plasma.channels) == 1
        @test !session.state.launched
        @test session.source.calls == 0
        if method == :adams
            @test SP._adams_session_free(session, 0, 0) == 1
            session.slots[1] = 1
            @test SP._adams_session_free(session, 1, 0) == 2
            before_slots = copy(session.slots)
            @test_throws MethodError SP._adams_session_free(session, true, 0)
            @test session.slots == before_slots
            session.slots[1] = 0
        end
        receipt = SP._precompile_drain!(session)
        @test isbitstype(typeof(receipt))
        @test receipt.accepted == 4
        @test receipt.calls > 0
        @test receipt.rows == 4
        @test receipt.planes == 3 # launch plus the two requested planes
        @test session.state.finished && session.state.closed
        @test !session.state.pending
        @test session.services.finished
        @test all(isfinite, session.field)
        @test all(iszero, session.field[.!session.norm.active, :])
    end
    @test_throws ArgumentError SP._precompile_material_case(execution, Val(:unknown))
    @test_throws ArgumentError SP._precompile_session(execution, Val(:unknown), Val(:field))
end

@testset "Precompile preparations own independent sources and observation work" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    for kind in (:field, :envelope)
        first = SP._precompile_session(execution, Val(:rk), Val(kind))
        second = SP._precompile_session(execution, Val(:rk), Val(kind))
        @test !SP._retained_alias(first.field, second)
        @test !SP._retained_alias(first.source.data.grid.active, second)
        @test !SP._retained_alias(first.source.data.kerr.density.value.coefficients, second)
        @test !SP._retained_alias(first.services.plane_values, second)
        @test !SP._retained_alias(first.services.rows.values, second)
        before = copy(second.field)
        SP._precompile_drain!(first)
        @test second.field == before
        @test !second.state.launched && !second.services.started
        @test second.source.calls == 0
        SP.close_session!(second)
    end
end

@testset "Precompile workload preserves process policy and retains no global work" begin
    before = precompile_process_state()
    @test SP._precompile_cpu_workload() === nothing
    @test precompile_process_state() == before
    @test SP._precompile_cpu_workload() === nothing
    @test precompile_process_state() == before
    bindings = filter(
        name -> startswith(string(name), "_precompile_"), names(SP; all = true),
    )
    @test !isempty(bindings)
    @test all(name -> getfield(SP, name) isa Function, bindings)
    refs = precompile_owner_references()
    GC.gc()
    @test all(ref -> ref.value === nothing, refs)
    @test Base.get_extension(SP, :SolpropJuliaPythonCallExt) === nothing
    for name in (:SolpropJuliaCUDAExt, :SolpropJuliaAMDGPUExt, :SolpropJuliaMetalExt)
        @test Base.get_extension(SP, name) === nothing
    end
end

@testset "Precompile drain closes a session when its bounded workload refuses" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    session = SP._precompile_session(execution, Val(:rk), Val(:field))
    SP.close_session!(session)
    @test_throws ArgumentError SP._precompile_drain!(session)
    @test session.state.closed && !session.state.pending
end
