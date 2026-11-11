# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function monitor_transfer_zero!(output, field, position)
    fill!(output, zero(eltype(output)))
    return nothing
end

function monitor_transfer_case(; monitored = true, services = true, plane_pending = true)
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    field = zeros(ComplexF64, 8, 1)
    field[1] = 8.0
    flow = SolpropJulia.LinearFlow(
        execution, SolpropJulia.DiagonalOperator(execution, zeros(ComplexF64, 8, 1)),
        field,
    )
    norm = SolpropJulia.MarchNorm(
        execution, fill(true, 8), fill(true, 8), ones(8), 1.0e-3, 0.0,
    )
    schedule = SolpropJulia.MarchSchedule(
        planes = [0.2], breakpoints = [0.0, 0.2], hard_kinks = Float64[], stretch = 1.0,
        step_planes = 0,
        replay_steps = [0.1, 0.1], replay_edges = [0.0, 0.1, 0.2],
    )
    tableau = SolpropJulia.EmbeddedTableau(
        [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
    )
    control = SolpropJulia.PIControl(
        safety = 0.8, max_factor = 2.0, min_factor = 0.2, beta1 = 0.3, beta2 = -0.1,
        lag = 0, reject_free = false, estimator_order = 1, start_growth = 100.0,
        dz_min = 1.0e-12, dz_max = 0.2, auto_init = false,
    )
    observer = SolpropJulia._NoSessionServices()
    if services
        grid = SolpropJulia.StatisticsGrid(
            collect(1.0:8.0), collect(0:7) ./ 8, fill(1 / 8, 8),
        )
        selection = SolpropJulia.StatisticsSelection(
            (SolpropJulia.StatisticSlot(:energy, 1, 1),), (1,),
        )
        temporal = if monitored
            regions = zeros(8, 4)
            regions[1, 1] = regions[8, 2] = 1.0
            regions[1, 3] = regions[8, 3] = 1.0
            regions[:, 4] .= 0.2
            policy = SolpropJulia.TemporalMonitorPolicy(
                warn = 1.0, stop = 2.0, activity = 1.0e-4, quiet_share = 1.0e-12,
                quiet_checks = 2, quiet_collar = true, cadence = 8, interval = 0.25,
                launch_floor_ratio = 1.0e-12, floor_margin = 3.0, split = false,
            )
            SolpropJulia.TemporalObservationData(
                policy, SolpropJulia.TemporalMonitorWeights(regions, [1.0]),
            )
        else
            nothing
        end
        preparation = SolpropJulia.ObservationPreparation(
            execution, grid, selection; kind = :envelope, n_modes = 1,
            power_weight = SolpropJulia.ObservationWeight(execution, ones(8, 1)), temporal,
        )
        observer = SolpropJulia.ObservationServices(
            execution, SolpropJulia.SessionObservationPolicy(preparation);
            attempt_capacity = 4, node_chunk = 1,
        )
    end
    return SolpropJulia.RungeKuttaSession(
        execution, field, 0.0, 0.1, tableau, control, norm, schedule,
        monitor_transfer_zero!, flow; services = observer, capacity = 4, plane_pending,
    )
end

function monitor_transfer_old(; has_quiet_run = true, launch = 8.0)
    return SolpropJulia.TemporalContinuation(
        launch, (0.125, 0.25, 0.375), (0.25, 0.375), (0.1, 0.2), 0.5,
        0.5, 0.25, 3, true, 0.625, 7, 5, 2, -0.1, has_quiet_run,
    )
end

function monitor_transfer_payload_equal(left, right)
    for field in fieldnames(SolpropJulia.SessionMonitorPayload)
        one, two = getfield(left, field), getfield(right, field)
        if field === :record
            for name in fieldnames(SolpropJulia.TemporalMonitorRecord)
                a, b = getfield(one, name), getfield(two, name)
                if name === :continuation
                    for member in fieldnames(SolpropJulia.TemporalContinuation)
                        @test isequal(getfield(a, member), getfield(b, member))
                    end
                else
                    @test isequal(a, b)
                end
            end
        else
            @test isequal(one, two)
        end
    end
    return nothing
end

@testset "Temporal transport publishes fresh and pending diagnostic records" begin
    @test isbitstype(SolpropJulia.TemporalMonitorRecord)
    @test Base.datatype_pointerfree(SolpropJulia.SessionMonitorPayload)
    for plane_pending in (false, true)
        session = monitor_transfer_case(; plane_pending)
        initial = SolpropJulia.session_monitor_payload(session, 0)
        @test isnan(initial.record.continuation.launch)
        @test isnan(initial.record.committed)
        @test !initial.record.resuming && !initial.attached && !initial.finished
        @test initial.row_count == 0 && initial.notice === nothing &&
            initial.crossing === nothing
        @test initial.record.removed_signed == initial.record.removed_fraction == 0.0
        @test initial.record.noise_floor == 0.0
        @test !session.state.services_started
        SolpropJulia.session_monitor_rows!(zeros(5, 0), session, 0)
        @test_throws ArgumentError SolpropJulia.session_monitor_payload(session, 1)
    end
    session = monitor_transfer_case()
    launch = SolpropJulia.advance!(session; deadline = 60.0)
    initial = SolpropJulia.session_monitor_payload(session, launch.generation)
    @test isnan(initial.record.continuation.launch) && !initial.attached
    @test_throws ArgumentError SolpropJulia.session_monitor_payload(session, 0)
    @test_throws ArgumentError SolpropJulia.session_monitor_payload(
        session, launch.generation + 1,
    )
    SolpropJulia.acknowledge!(session, launch.generation)
    @test_throws ArgumentError SolpropJulia.session_monitor_payload(
        session, launch.generation,
    )
    @test_throws ArgumentError SolpropJulia.session_monitor_payload(session, 0)
    plane = SolpropJulia.advance!(session; deadline = 60.0)
    payload = SolpropJulia.session_monitor_payload(session, plane.generation)
    @test payload.attached && !payload.finished
    @test payload.row_count == 2
    @test payload.record.committed == session.state.z
    @test payload.record.continuation.has_quiet_run
    monitor_transfer_payload_equal(
        payload, SolpropJulia.session_monitor_payload(session, plane.generation),
    )
    session.state.poisoned = true
    monitor_transfer_payload_equal(
        payload, SolpropJulia.session_monitor_payload(session, plane.generation),
    )
    session.state.poisoned = false
    SolpropJulia.acknowledge!(session, plane.generation)
    terminal = SolpropJulia.advance!(session; deadline = 60.0)
    final = SolpropJulia.session_monitor_payload(session, terminal.generation)
    @test final.finished && final.attached && final.row_count == 0
    SolpropJulia.close_session!(session)
    @test_throws ArgumentError SolpropJulia.session_monitor_payload(
        session, terminal.generation,
    )
end

@testset "Unmonitored generations have an empty numerical transport" begin
    for services in (false, true)
        session = monitor_transfer_case(; monitored = false, services)
        @test SolpropJulia.session_monitor_payload(session, 0) === nothing
        SolpropJulia.session_monitor_rows!(zeros(5, 0), session, 0)
        @test SolpropJulia.restore_session_monitor!(session, nothing, false) === nothing
        launch = SolpropJulia.advance!(session; deadline = 60.0)
        @test SolpropJulia.session_monitor_payload(session, launch.generation) === nothing
        SolpropJulia.acknowledge!(session, launch.generation)
        @test_throws ArgumentError SolpropJulia.session_monitor_payload(
            session, launch.generation,
        )
    end
end

@testset "Monitor copy preserves fixed rows notices and generation on every read" begin
    session = monitor_transfer_case()
    launch = SolpropJulia.advance!(session; deadline = 60.0)
    services = session.services
    services.monitor_rows[1] = SolpropJulia.TemporalPlaneRow(
        true, 0.0, 0.25, 0.125, 0.0, 0.5,
    )
    services.monitor_rows[2] = SolpropJulia.TemporalPlaneRow(true, 0.2, NaN, NaN, NaN, 0.5)
    services.monitor_count = 2
    notice = SolpropJulia.MonitorNotice(
        SolpropJulia.TemporalGuardWarning, NaN, 0.25, 0, 0.0,
    )
    crossing = SolpropJulia.TemporalCrossingRefusal(0.2, 0.01, 0.1)
    services.notices.records[1], services.notices.count = notice, 1
    services.crossing_refusal = crossing
    expected = [0.0 0.2; 0.25 NaN; 0.125 NaN; 0.0 NaN; 0.5 0.5]
    output = fill(-1.0, 5, 2)
    for _ in 1:2
        SolpropJulia.session_monitor_rows!(output, session, launch.generation)
        @test isequal(output, expected)
        payload = SolpropJulia.session_monitor_payload(session, launch.generation)
        @test payload.row_count == 2 && payload.notice === notice &&
            payload.crossing === crossing
        @test session.state.pending && services.monitor_count == 2 &&
            services.notices.count == 1
    end
    sentinel = fill(-1.0, 4, 2)
    @test_throws DimensionMismatch SolpropJulia.session_monitor_rows!(
        sentinel, session, launch.generation,
    )
    @test all(==(-1.0), sentinel)
    before = copy(session.field)
    owned = reshape(view(reinterpret(Float64, vec(session.field)), 2:11), 5, 2)
    @test_throws ArgumentError SolpropJulia.session_monitor_rows!(
        owned, session, launch.generation,
    )
    @test session.field == before
    GC.@preserve session begin
        address = Ptr{Float64}(pointer(session.field)) + sizeof(Float64)
        foreign = unsafe_wrap(Array, address, (5, 2); own = false)
        @test_throws ArgumentError SolpropJulia.session_monitor_rows!(
            foreign, session, launch.generation,
        )
        reversed = unsafe_wrap(Array, address, (2, 5); own = false)
        for wrapped in (transpose(reversed), PermutedDimsArray(reversed, (2, 1)))
            @test_throws ArgumentError SolpropJulia.session_monitor_rows!(
                wrapped, session, launch.generation,
            )
            @test session.field == before
        end
    end
    @test session.field == before
    services.monitor_count = 3
    @test_throws ArgumentError SolpropJulia.session_monitor_payload(
        session, launch.generation,
    )
    services.monitor_count = 2
    SolpropJulia.acknowledge!(session, launch.generation)
    @test services.monitor_count == services.notices.count == 0
    @test services.crossing_refusal === nothing
    @test isequal(output, expected)
end

@testset "Signed removal retains its capped floor and no-light convention" begin
    # All operands are binary fractions. These analytic cases isolate one-launch
    # floor saturation and clipping without introducing a rounding comparison.
    for (launch, floor, raw, taken, signed, noise) in (
            (8.0, 2.0, 0.5, 0.125, 0.375, 0.25),
            (8.0, 2.0, 0.5, 0.5, 0.25, 0.25),
            (8.0, 2.0, 0.125, 0.5, -0.125, 0.25),
            (0.0, 2.0, 0.125, 0.5, 0.125, 0.0),
        )
        session = monitor_transfer_case()
        monitor = session.services.work.monitor
        monitor.launch, monitor.floor, monitor.raw_removed, monitor.floor_removed =
            launch, floor, raw, taken
        record = SolpropJulia.session_monitor_payload(session, 0).record
        @test record.removed_signed == signed
        @test record.removed_fraction == max(signed, 0.0)
        @test record.noise_floor == noise
        @test record.continuation.raw_removed == raw
        @test record.continuation.floor_removed == taken
    end
end

@testset "Restore precedes attachment and keeps the first selected continuation" begin
    for record in (nothing, monitor_transfer_old(), monitor_transfer_old(; launch = NaN))
        session = monitor_transfer_case()
        SolpropJulia.restore_session_monitor!(session, record, true)
        first = SolpropJulia.session_monitor_payload(session, 0)
        @test first.record.resuming && !first.attached && isnan(first.record.committed)
        SolpropJulia.restore_session_monitor!(
            session, monitor_transfer_old(; launch = 16.0), false,
        )
        monitor_transfer_payload_equal(
            first, SolpropJulia.session_monitor_payload(session, 0),
        )
        launch = SolpropJulia.advance!(session; deadline = 60.0)
        SolpropJulia.restore_session_monitor!(session, nothing, false)
        @test session.services.work.monitor.resume_plane
        SolpropJulia.acknowledge!(session, launch.generation)
        plane = SolpropJulia.advance!(session; deadline = 60.0)
        after = SolpropJulia.session_monitor_payload(session, plane.generation)
        @test !after.record.resuming && after.attached && after.record.committed == 0.2
        @test_throws ArgumentError SolpropJulia.restore_session_monitor!(
            session, nothing, false,
        )
        monitor_transfer_payload_equal(
            after, SolpropJulia.session_monitor_payload(session, plane.generation),
        )
    end
    for present in (false, true)
        session = monitor_transfer_case()
        old = monitor_transfer_old(; has_quiet_run = present)
        SolpropJulia.restore_session_monitor!(session, old, false)
        @test session.services.work.monitor.restore_quiet == present
        @test session.services.work.monitor.restored_quiet_run == old.quiet_run
    end
    closed = monitor_transfer_case()
    SolpropJulia.close_session!(closed)
    @test_throws ArgumentError SolpropJulia.restore_session_monitor!(closed, nothing, false)
end

@testset "Resumed-launch facts survive a later failing launch operation" begin
    session = monitor_transfer_case()
    SolpropJulia.restore_session_monitor!(session, nothing, true)
    launch = SolpropJulia.advance!(session; deadline = 60.0)
    @test SolpropJulia.session_monitor_payload(session, launch.generation).record.resuming
    SolpropJulia.acknowledge!(session, launch.generation)
    # A full row buffer fails after temporal_launch! consumed the resume flag, but
    # before ObservationServices.started. The scalar facts must still describe it.
    session.services.monitor_count = 2
    failure = SolpropJulia.advance!(session; deadline = 60.0)
    payload = SolpropJulia.session_monitor_payload(session, failure.generation)
    @test failure.kind === SolpropJulia.SESSION_FAILED
    @test payload.attached && !session.services.started
    @test !payload.record.resuming && payload.record.committed == 0.0
    @test_throws ArgumentError SolpropJulia.restore_session_monitor!(session, nothing, true)
end

function monitor_transfer_allocations(session, output, generation)
    SolpropJulia.session_monitor_payload(session, generation)
    SolpropJulia.session_monitor_rows!(output, session, generation)
    return (
        @allocated(SolpropJulia.session_monitor_payload(session, generation)),
        @allocated(SolpropJulia.session_monitor_rows!(output, session, generation)),
    )
end

@testset "Warmed monitor scalar and copy transports allocate nothing" begin
    session = monitor_transfer_case()
    launch = SolpropJulia.advance!(session; deadline = 60.0)
    SolpropJulia.acknowledge!(session, launch.generation)
    plane = SolpropJulia.advance!(session; deadline = 60.0)
    count = SolpropJulia.session_monitor_payload(session, plane.generation).row_count
    output = zeros(5, count)
    @test monitor_transfer_allocations(session, output, plane.generation) == (0, 0)
end
