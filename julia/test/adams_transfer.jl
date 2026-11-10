# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import SolpropJulia as SP

mutable struct AdamsTransferSource{E}
    execution::E
    calls::Int
end

function (source::AdamsTransferSource)(output, field, position)
    source.calls += 1
    SP.each!(source.execution, size(field)...) do row, column
        output[row, column] = (0.03 + 0.02im) * field[row, column] + 0.1position
    end
    return nothing
end

function adams_transfer_fixture(;
        tasks = 1, compensated = false, z = 0.0, plane_pending = true,
        initial = fill(1.0 + 0.2im, 4, 2),
    )
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    policy = SP.AdamsControlPolicy(;
        order = 4, variable_order = true, safety = 0.8, max_growth = 2.0,
        min_factor = 0.2, rejection_growth = 0.9, hysteresis = 1.2, restart_after = 3,
        collar_stability = 0.7, real_intervals = (2.0, 1.0, 0.5, 0.3, 0.2, 0.15),
        collar_rate = 0.0, carrier = false, carrier_cap = 4.0, max_calls = nothing,
        equal_approach = true, interval_cap = false, dz_min = 1.0e-10, dz_max = 0.05,
        stretch = 1.1, z_end = 0.2, save_planes = [0.1, 0.2],
        hard_kinks = Float64[], breakpoints = [0.0, 0.4], landing_ulps = 8,
        interpolate = false, quiet_interval = nothing,
    )
    rate = -0.2 + 0.3im
    operator = if compensated
        # On [0,.4], these Chebyshev coefficients represent Phi(z)=rate*z^2/2.
        coefficients = zeros(ComplexF64, length(initial), 3, 1)
        coefficients[:, 1, 1] .= 0.03rate
        coefficients[:, 2, 1] .= 0.04rate
        coefficients[:, 3, 1] .= 0.01rate
        table = SP.ChebyshevTable(execution, [0.0, 0.4], coefficients)
        SP.RepresentedOperator(execution, table, size(initial)...)
    else
        SP.DiagonalOperator(execution, fill(rate, size(initial)))
    end
    norm = SP.MarchNorm(execution, fill(true, 4), fill(true, 4), ones(4), 1.0e-3, 0.0)
    steps = fill(0.025, 8)
    schedule = SP.MarchSchedule(;
        planes = policy.save_planes, breakpoints = policy.breakpoints,
        hard_kinks = policy.hard_kinks, stretch = policy.stretch,
        step_planes = 0, replay_steps = steps, replay_edges = [0.0; cumsum(steps)],
    )
    return SP.AdamsSession(
        execution, initial, z, 0.025, policy, norm, schedule,
        AdamsTransferSource(execution, 0), SP.LinearFlow(execution, operator, initial);
        plane_pending,
    )
end

function adams_transfer_buffers(session, count)
    shape, stored = size(session.field), size(session.history.source)
    sentinel = -999.0 + 11.0im
    states = session.history.compensated ? fill(sentinel, stored) : nothing
    rates = session.history.compensated ? fill(sentinel, stored) : nothing
    return SP.AdamsHistoryBuffers(
        fill(sentinel, shape), fill(-99.0, count), fill(0xff, count),
        fill(Int64(-9), count),
        fill(sentinel, stored); states, rates,
    )
end

function adams_transfer_rebuild(
        buffers; state = buffers.state, positions = buffers.positions,
        identities = buffers.identities, slots = buffers.slots, store = buffers.store,
        states = buffers.states, rates = buffers.rates,
    )
    return SP.AdamsHistoryBuffers(state, positions, identities, slots, store; states, rates)
end

function adams_transfer_metadata(
        metadata; z = metadata.z, proposal = metadata.proposal, order = metadata.order,
        rejections = metadata.rejections, last_rejected = metadata.last_rejected,
        collar_was_quiet = metadata.collar_was_quiet,
    )
    return SP.AdamsHistoryMetadata(
        z, proposal, order, rejections, last_rejected, collar_was_quiet,
    )
end

function adams_transfer_record(session)
    return (
        copy(session.field),
        Tuple(
            copy(getfield(session.history, name)) for name in (
                    :source, :states, :rates, :positions, :identities, :occupied,
                )
        ),
        copy(session.positions), copy(session.slots), copy(session.norms),
        Tuple(
            getfield(session.control, name) for name in fieldnames(typeof(session.control))
        ),
        Tuple(getfield(session.state, name) for name in fieldnames(typeof(session.state))),
        Tuple(
            getfield(session.ledger, name) for name in fieldnames(typeof(session.ledger))
        ),
        copy(session.stencil.slots), copy(session.stencil.orders),
        copy(session.stencil.nodes),
        session.stencil.order, session.stencil.count, session.stencil.n_orders,
        session.work.ready, session.work.position, session.work.step, session.work.order,
        session.work.n_orders, session.source.calls,
    )
end

function adams_transfer_saved(; tasks = 1, compensated = false)
    session = adams_transfer_fixture(; tasks, compensated, z = 0.2)
    history = session.history
    for slot in axes(history.source, 3), column in axes(session.field, 2), row in 1:4
        history.source[row, column, slot] = ComplexF64(100slot + 10column + row, -slot)
        if compensated
            history.states[row, column, slot] = ComplexF64(slot, 10column + row)
            history.rates[row, column, slot] = ComplexF64(-column, row / 16)
        end
    end
    # Distinguish unused full-store rows and signed zeros from a selected-row export.
    history.source[1, 1, 4] = ComplexF64(-0.0, 0.0)
    session.field[1, 1] = ComplexF64(0.0, -0.0)
    session.positions[1:3] .= (0.1, 0.15, 0.2)
    session.slots[1:3] .= (5, 2, 6)
    for (index, kind) in enumerate((1, 0, 1))
        slot = session.slots[index]
        history.positions[slot] = session.positions[index]
        history.identities[slot] = kind
        history.occupied[slot] = true
    end
    session.control.held, session.control.order = 3, 2
    session.control.rejections = 2
    session.control.last_rejected = true
    session.control.collar_was_quiet = true
    session.state.proposal = 0.0123
    session.state.pending, session.state.generation = true, 7
    info = SP.adams_history_info(session, 7, 0.2)
    @test info !== nothing
    buffers = adams_transfer_buffers(session, info[2])
    SP.export_adams_history!(buffers, session, 7, 0.2)
    return session, info[1], buffers
end

function adams_transfer_refused(session, metadata, buffers; contains = "snapshot")
    before = adams_transfer_record(session)
    reason = SP.restore_adams_history!(session, metadata, buffers)
    @test reason isa String
    @test occursin(contains, reason)
    @test isequal(adams_transfer_record(session), before)
    return nothing
end

@testset "Adams transfer preserves full stores and portable selected metadata" begin
    for tasks in (1, 2), compensated in (false, true)
        original, metadata, buffers = adams_transfer_saved(; tasks, compensated)
        info = SP.adams_history_info(original, 7, 0.2)
        @test info[2:4] == (3, 6, compensated)
        @test info[5] == sizeof(original.field) + sizeof(original.history.source) *
            (compensated ? 3 : 1)
        @test buffers.positions == [0.1, 0.15, 0.2]
        @test buffers.identities == UInt8[1, 0, 1]
        @test buffers.slots == Int64[4, 1, 5]
        @test isequal(buffers.state, original.field)
        @test isequal(buffers.store, original.history.source)
        if compensated
            @test isequal(buffers.states, original.history.states)
            @test isequal(buffers.rates, original.history.rates)
        end
        fresh = adams_transfer_fixture(;
            tasks, compensated, z = 0.2, plane_pending = false,
            initial = fill(nextfloat(1.0) + 0.2im, 4, 2),
        )
        # Cumulative budget restoration is separate; this leaf must leave it alone.
        fresh.ledger.startup, fresh.ledger.predictor = 9, 27
        ledgers = deepcopy(fresh.ledger)
        @test SP.restore_adams_history!(fresh, metadata, buffers) === nothing
        @test isequal(fresh.field, buffers.state)
        @test isequal(fresh.history.source, buffers.store)
        @test fresh.control.held == 3 && fresh.control.order == 2
        @test fresh.control.rejections == 2 && fresh.control.last_rejected
        @test fresh.control.collar_was_quiet
        @test fresh.state.proposal == 0.0123 && fresh.state.launched
        @test fresh.positions == [0.1, 0.15, 0.2, 0.0, 0.0]
        @test fresh.slots == [5, 2, 6, 0, 0]
        @test findall(fresh.history.occupied) == [2, 5, 6]
        @test fresh.history.identities[fresh.slots[1:3]] == [1, 0, 1]
        @test fresh.history.positions[fresh.slots[1:3]] == buffers.positions
        @test !fresh.work.ready && fresh.stencil.count == fresh.stencil.order == 0
        @test fresh.source.calls == fresh.ledger.source_evaluations == 0
        @test fresh.ledger.startup == ledgers.startup
        @test fresh.ledger.predictor == ledgers.predictor
        if compensated
            @test isequal(fresh.history.states, buffers.states)
            @test isequal(fresh.history.rates, buffers.rates)
        end
        retained = adams_transfer_record(fresh)
        fill!(buffers.store, 77)
        fill!(buffers.state, 88)
        fill!(buffers.positions, 99)
        @test isequal(adams_transfer_record(fresh), retained)
        adams_transfer_refused(fresh, metadata, buffers; contains = "fresh")
    end
end

@testset "Adams restore refuses incompatible payloads before every write" begin
    for compensated in (false, true)
        _, metadata, valid = adams_transfer_saved(; compensated)
        fresh = adams_transfer_fixture(; compensated, z = 0.2, plane_pending = false)
        for changes in (
                (; z = NaN), (; z = 0.1), (; proposal = 0.0), (; proposal = -1.0),
                (; proposal = Inf), (; order = 0), (; order = 5), (; rejections = -1),
            )
            adams_transfer_refused(
                fresh, adams_transfer_metadata(metadata; changes...), valid,
            )
        end
        for changes in (
                (; state = zeros(ComplexF64, 3, 2)), (; state = zeros(Float64, 4, 2)),
                (; positions = Float64[]), (; positions = fill(0.2, 6)),
                (; positions = 0.2), (; positions = Float32[0.1, 0.15, 0.2]),
                (; identities = Int64[1, 0, 1]), (; identities = UInt8[1, 0]),
                (; slots = Int32[4, 1, 5]), (; slots = Int64[4, 1]),
                (; store = zeros(ComplexF64, 4, 2, 5)),
                (; store = zeros(ComplexF32, 4, 2, 6)),
            )
            adams_transfer_refused(
                fresh, metadata, adams_transfer_rebuild(valid; changes...),
            )
        end
        for changed in (:state, :store, :positions, :identities, :slots)
            invalid = deepcopy(valid)
            if changed === :positions
                for positions in ([0.1, NaN, 0.2], [0.15, 0.1, 0.2], [0.1, 0.1, 0.2])
                    copyto!(invalid.positions, positions)
                    adams_transfer_refused(fresh, metadata, invalid)
                end
            elseif changed === :identities
                invalid.identities[2] = 2
                adams_transfer_refused(fresh, metadata, invalid)
            elseif changed === :slots
                for slots in (Int64[-1, 1, 5], Int64[4, 1, 6], Int64[4, 4, 5])
                    copyto!(invalid.slots, slots)
                    adams_transfer_refused(fresh, metadata, invalid)
                end
            else
                array = getfield(invalid, changed)
                # Include an unselected row: full stores have one finite-data policy.
                array[end] = ComplexF64(NaN)
                adams_transfer_refused(fresh, metadata, invalid)
            end
        end
        if compensated
            for name in (:states, :rates)
                for bad in (nothing, zeros(Float64, 4, 2, 6), zeros(ComplexF64, 4, 2, 5))
                    adams_transfer_refused(
                        fresh, metadata,
                        adams_transfer_rebuild(valid; NamedTuple{(name,)}((bad,))...),
                    )
                end
                invalid = deepcopy(valid)
                getfield(invalid, name)[1, 1, 4] = ComplexF64(Inf)
                adams_transfer_refused(fresh, metadata, invalid)
            end
        else
            for name in (:states, :rates)
                adams_transfer_refused(
                    fresh, metadata,
                    adams_transfer_rebuild(
                        valid; NamedTuple{(name,)}((zeros(ComplexF64, 4, 2, 6),))...,
                    ); contains = "plain",
                )
            end
        end
        # Every refusal left a valid retry possible with the original detached data.
        @test SP.restore_adams_history!(fresh, metadata, valid) === nothing
    end
end

@testset "Adams restore keeps actual sample positions and startup next order" begin
    _, metadata, valid = adams_transfer_saved()
    for displacement in (0.0, 0.2 * 0.5e-12, -0.2 * 0.5e-12)
        fresh = adams_transfer_fixture(; z = 0.2, plane_pending = false)
        buffers = deepcopy(valid)
        buffers.positions[end] += displacement
        @test SP.restore_adams_history!(fresh, metadata, buffers) === nothing
        @test fresh.positions[3] == buffers.positions[end]
        @test fresh.history.positions[6] == buffers.positions[end]
        @test fresh.state.z == metadata.z
    end
    invalid = deepcopy(valid)
    invalid.positions[end] += 0.2 * 2.0e-12
    adams_transfer_refused(adams_transfer_fixture(; z = 0.2), metadata, invalid)
    for newest in (0.0, nextfloat(0.0))
        fresh = adams_transfer_fixture(; plane_pending = false)
        buffers = adams_transfer_rebuild(
            valid; positions = [newest], identities = UInt8[0], slots = Int64[0],
        )
        seed = adams_transfer_metadata(metadata; z = 0.0, order = 4)
        if iszero(newest)
            @test SP.restore_adams_history!(fresh, seed, buffers) === nothing
            @test fresh.control.held == 1 && fresh.control.order == 4
        else
            adams_transfer_refused(fresh, seed, buffers)
        end
    end
end

@testset "Adams import and export reject retained and cross-destination aliases" begin
    for compensated in (false, true)
        original, metadata, valid = adams_transfer_saved(; compensated)
        fresh = adams_transfer_fixture(; compensated, z = 0.2)
        for changes in (
                (; state = fresh.field), (; store = fresh.history.source),
                (; positions = view(fresh.stencil.nodes, 1:3)),
                (; slots = view(fresh.slots, 1:3)),
                (; identities = view(reinterpret(UInt8, fresh.history.identities), 1:3)),
            )
            adams_transfer_refused(
                fresh, metadata, adams_transfer_rebuild(valid; changes...);
                contains = "alias",
            )
        end
        if compensated
            for changes in (
                    (; states = fresh.history.states), (; rates = fresh.history.rates),
                )
                adams_transfer_refused(
                    fresh, metadata, adams_transfer_rebuild(valid; changes...);
                    contains = "alias",
                )
            end
        end
        # Immutable policy storage is retained too, not just mutable numerical work.
        pair = adams_transfer_rebuild(
            valid; positions = fresh.policy.save_planes,
            identities = UInt8[0, 1], slots = Int64[0, 1],
        )
        adams_transfer_refused(fresh, metadata, pair; contains = "alias")
        for changes in (
                (; state = original.field), (; store = original.history.source),
                (; positions = view(original.positions, 1:3)),
                (; slots = view(original.slots, 1:3)),
                (; identities = view(reinterpret(UInt8, original.history.identities), 1:3)),
                (; state = view(valid.store, :, :, 1)),
            )
            destinations = adams_transfer_rebuild(valid; changes...)
            before = deepcopy(destinations)
            retained = adams_transfer_record(original)
            @test_throws ArgumentError SP.export_adams_history!(
                destinations, original, 7, 0.2,
            )
            for name in fieldnames(typeof(before))
                @test isequal(getfield(destinations, name), getfield(before, name))
            end
            @test isequal(adams_transfer_record(original), retained)
        end
        # Read-only input overlap is harmless: no destination is in either array.
        shared = adams_transfer_rebuild(valid; state = view(valid.store, :, :, 1))
        @test SP.restore_adams_history!(fresh, metadata, shared) === nothing
        @test isequal(fresh.field, shared.state)
    end
end

@testset "Adams transfer lifecycle guards published tickets and dense observations" begin
    original, metadata, buffers = adams_transfer_saved()
    @test SP.adams_history_info(original, 7, prevfloat(0.2)) === nothing
    @test_throws ArgumentError SP.export_adams_history!(
        buffers, original, 7, prevfloat(0.2),
    )
    @test_throws ArgumentError SP.adams_history_info(original, 6, 0.2)
    @test_throws ArgumentError SP.adams_history_info(original, 7, NaN)
    for field in (:closed, :poisoned)
        setfield!(original.state, field, true)
        @test_throws ArgumentError SP.adams_history_info(original, 7, 0.2)
        setfield!(original.state, field, false)
    end
    original.state.pending = false
    @test_throws ArgumentError SP.adams_history_info(original, 7, 0.2)
    for plane_pending in (true, false)
        fresh = adams_transfer_fixture(; z = 0.2, plane_pending)
        @test SP.restore_adams_history!(fresh, metadata, buffers) === nothing
    end
    for action in (:launch, :acknowledged_launch, :snapshot, :closed)
        fresh = adams_transfer_fixture(; z = 0.2)
        if action in (:launch, :acknowledged_launch)
            report = SP.advance!(fresh)
            @test report.kind == SP.SESSION_LAUNCH
            @test fresh.state.accepted == fresh.control.held == 0
            @test SP.adams_history_info(fresh, report.generation, fresh.state.z) === nothing
            action === :acknowledged_launch && SP.acknowledge!(fresh, report.generation)
        elseif action === :snapshot
            SP.request_snapshot!(fresh)
        else
            SP.close_session!(fresh)
        end
        adams_transfer_refused(fresh, metadata, buffers; contains = "fresh")
    end
end

@testset "Adams export refuses incomplete or corrupt data without partial copies" begin
    for compensated in (false, true)
        session, _, buffers = adams_transfer_saved(; compensated)
        for changes in (
                (; state = zeros(ComplexF32, 4, 2)),
                (; positions = range(0.0, 0.2; length = 3)),
                (; identities = UInt8[0, 1]), (; slots = Int32[4, 1, 5]),
                (; store = zeros(ComplexF64, 4, 2, 5)),
            )
            destinations = adams_transfer_rebuild(buffers; changes...)
            before = deepcopy(destinations)
            @test_throws ArgumentError SP.export_adams_history!(
                destinations, session, 7, 0.2,
            )
            for name in fieldnames(typeof(before))
                @test isequal(getfield(destinations, name), getfield(before, name))
            end
        end
        for defect in (
                :empty, :count, :unoccupied, :kind, :coordinate, :ordering,
                :repeated, :nonfinite_state, :nonfinite_inactive,
            )
            damaged, _, destinations = adams_transfer_saved(; compensated)
            if defect === :empty
                damaged.control.held = 0
                @test SP.adams_history_info(damaged, 7, 0.2) === nothing
            elseif defect === :count
                damaged.control.held = 6
            elseif defect === :unoccupied
                damaged.history.occupied[5] = false
            elseif defect === :kind
                damaged.history.identities[2] = 7
            elseif defect === :coordinate
                damaged.history.positions[2] = nextfloat(0.15)
            elseif defect === :ordering
                damaged.positions[1] = damaged.history.positions[5] = 0.16
            elseif defect === :repeated
                damaged.slots[2] = 5
            elseif defect === :nonfinite_state
                damaged.field[2, 2] = ComplexF64(Inf)
            else
                damaged.history.source[1, 1, 4] = ComplexF64(NaN)
            end
            before = deepcopy(destinations)
            retained = adams_transfer_record(damaged)
            @test_throws ArgumentError SP.export_adams_history!(
                destinations, damaged, 7, 0.2,
            )
            for name in fieldnames(typeof(before))
                @test isequal(getfield(destinations, name), getfield(before, name))
            end
            @test isequal(adams_transfer_record(damaged), retained)
        end
    end
end

@testset "Adams portable buffers admit strided views without retaining them" begin
    original, metadata, valid = adams_transfer_saved(; compensated = true)
    function strided(array)
        large = zeros(eltype(array), (2 .* size(array))...)
        return view(large, (1:2:size(large, dim) for dim in 1:ndims(large))...)
    end
    destinations = SP.AdamsHistoryBuffers(
        strided(valid.state), strided(valid.positions), strided(valid.identities),
        strided(valid.slots), strided(valid.store);
        states = strided(valid.states), rates = strided(valid.rates),
    )
    SP.export_adams_history!(destinations, original, 7, 0.2)
    for name in fieldnames(typeof(valid))
        @test isequal(getfield(destinations, name), getfield(valid, name))
    end
    fresh = adams_transfer_fixture(; compensated = true, z = 0.2)
    @test SP.restore_adams_history!(fresh, metadata, destinations) === nothing
    @test isequal(fresh.field, valid.state)
    @test isequal(fresh.history.source, valid.store)
    @test isequal(fresh.history.states, valid.states)
    @test isequal(fresh.history.rates, valid.rates)
end

function adams_transfer_to_plane(session, plane; work_limit = 256)
    while true
        report = SP.advance!(session; work_limit, deadline = 60.0)
        @test report.kind != SP.SESSION_FAILED
        if report.kind == SP.SESSION_PLANE && report.plane == plane
            return report
        end
        @test report.kind != SP.SESSION_COMPLETED
        SP.acknowledge!(session, report.generation)
    end
    return
end

function adams_transfer_advance_bytes(session)
    return @allocated SP.advance!(session; work_limit = 1, deadline = 60.0)
end

@testset "Adams restored history resumes without source reinitialization" begin
    for tasks in (1, 2), compensated in (false, true)
        original = adams_transfer_fixture(; tasks, compensated)
        launch = SP.advance!(original)
        SP.acknowledge!(original, launch.generation)
        first = SP.advance!(original; work_limit = 1, deadline = 60.0)
        @test original.history.identities[original.slots[1]] == 0
        @test original.history.identities[original.slots[2]] == 1
        SP.acknowledge!(original, first.generation)
        boundary = adams_transfer_to_plane(original, 0; work_limit = 1)
        info = SP.adams_history_info(original, boundary.generation, boundary.z)
        buffers = adams_transfer_buffers(original, info[2])
        SP.export_adams_history!(buffers, original, boundary.generation, boundary.z)
        resumed = adams_transfer_fixture(;
            tasks, compensated, z = boundary.z, plane_pending = false,
            initial = fill(9.0 + 8.0im, size(original.field)),
        )
        @test SP.restore_adams_history!(resumed, info[1], buffers) === nothing
        @test resumed.source.calls == 0
        @test resumed.ledger.startup == resumed.ledger.predictor == 0
        @test isequal(resumed.field, original.field)
        SP.acknowledge!(original, boundary.generation)
        finish = adams_transfer_to_plane(original, 1)
        continued = adams_transfer_to_plane(resumed, 1; work_limit = 1)
        @test finish.z == continued.z == last(original.schedule.replay_edges)
        # Matched four remaining steps use the same retained samples and coefficients.
        # 32768 real operations cover phi/coefficient/contraction work for this tiny
        # order<=4 fixture; exp(2*.2) bounds its linear recurrence sensitivity.
        gamma = (32768eps(Float64)) / (1 - 32768eps(Float64))
        bound = gamma * exp(2 * 0.2) * maximum(abs, original.field)
        @test bound > 0
        @test maximum(abs, resumed.field - original.field) <= bound
        @test resumed.source.calls == resumed.ledger.predictor == 4
        @test resumed.ledger.startup == 0
        @test resumed.control.held == original.control.held
        @test resumed.positions == original.positions
        @test resumed.slots == original.slots
        # Warm the exact restored-session entry before measuring a fresh restored one.
        warm = adams_transfer_fixture(;
            tasks, compensated, z = boundary.z, plane_pending = false,
        )
        @test SP.restore_adams_history!(warm, info[1], buffers) === nothing
        adams_transfer_advance_bytes(warm)
        measured = adams_transfer_fixture(;
            tasks, compensated, z = boundary.z, plane_pending = false,
        )
        @test SP.restore_adams_history!(measured, info[1], buffers) === nothing
        @test adams_transfer_advance_bytes(measured) == 0
        @test measured.source.calls == 1
    end
end
