# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

mutable struct TransverseSessionSource
    data::Nothing
    calls::Int
end

function (source::TransverseSessionSource)(output, field, position)
    source.calls += 1
    fill!(output, 0)
    return nothing
end

function transverse_session_preparation(
        execution, edge; temporal = false, temporal_warning = false,
    )
    nf, nc = edge.n_frequency, edge.n_columns
    grid = SP.StatisticsGrid(collect(1.0:nf), collect(0:7) ./ 8, fill(1 / 8, nf))
    operator = SP.SeparableSpatialOperator(
        execution; k_squared = fill(100.0 + 0im, nf), k_squared_low = zeros(ComplexF64, nf),
        kperp_sq = zeros(nc), kperp_sq_low = zeros(nc), frame = zeros(nf),
        spectral_rate = zeros(nf), prefactor_numerator = zeros(ComplexF64, nf),
        prefactor_clamped = zeros(ComplexF64, nf), active = fill(true, nf),
        evanescent_rate = 1.0, z_span = (0.0, 1.0),
    )
    monitor = if temporal
        regions = zeros(8, 4)
        regions[1, 1] = regions[8, 2] = 1.0
        regions[1, 3] = regions[8, 3] = 1.0
        policy = SP.TemporalMonitorPolicy(;
            warn = temporal_warning ? 0.05 : 0.9, stop = 2.0, activity = 0.1,
            quiet_share = 0.0, quiet_checks = 2, quiet_collar = false,
            cadence = 8, interval = Inf, launch_floor_ratio = 1.0e-12,
            floor_margin = 3.0, split = false,
        )
        SP.TemporalObservationData(policy, SP.TemporalMonitorWeights(regions, ones(nc)))
    else
        nothing
    end
    return SP.SpatialObservationPreparation(
        execution, grid, SP.StatisticsSelection((SP.StatisticSlot(:energy, 1, 1),), (1,));
        kind = :field, map = edge.map,
        flux = SP.SpatialFluxData(operator; light_speed = 1.0),
        spectral_weights = ones(nc), on_axis = fill(1.0 + 0im, nc),
        intensity_factor = 1.0, temporal = monitor,
    )
end

function transverse_session_edge(execution, prepared; clip = false, coarse = false)
    names = fieldnames(SP.TransversePolicy)
    values = NamedTuple{names}(Tuple(getfield(prepared.policy, name) for name in names))
    policy = SP.TransversePolicy(;
        values..., warn = 0.1, stop = 2.0,
        backstop_start = coarse ? 0.005 : 0.05,
    )
    old = prepared.tables
    clipping = zeros(size(old.clip))
    if clip
        for index in eachindex(old.face_ranks)
            clipping[:, index] .= 0.2 .* old.totals[:, 1]
        end
    end
    tables = SP.TransverseTables(
        old.faces, clipping, old.totals, old.column_weights, old.kappa,
        old.order, old.ramp_rates; face_ranks = old.face_ranks,
        noise_face_share = old.noise_face_share, noise_modes = old.noise_modes,
    )
    return SP.TransversePreparation(
        execution, prepared.map, policy, tables; n_frequency = prepared.n_frequency,
        rows = prepared.rows, lines = prepared.lines, declared = prepared.declared,
    )
end

function transverse_session_case(;
        tasks = 1, route = :line, temporal = false, temporal_warning = false,
        clip = false, coarse = false, capacity = 4,
        collar = true,
        steps = fill(0.01, 4), planes = [sum(steps)], dense = false,
        continuation = nothing, noise = nothing, clean = nothing,
    )
    execution, original, field = transverse_observation_case(route; tasks, collar)
    edge = transverse_session_edge(execution, original; clip, coarse)
    field[1, :] .= real.(field[1, :])
    field[end, :] .= real.(field[end, :])
    temporal_warning && fill!(field, 1.0 + 0im)
    prepared = transverse_session_preparation(execution, edge; temporal, temporal_warning)
    policy = SP.SessionObservationPolicy(prepared; transverse = edge, trace_chunk = 3)
    services = SP.ObservationServices(
        execution, policy; node_chunk = 3, attempt_capacity = capacity,
        noise_spectrum = noise, transverse_clean = clean,
        transverse_continuation = continuation,
    )
    schedule = SP.MarchSchedule(;
        planes, breakpoints = [0.0, last(planes)], hard_kinks = Float64[], stretch = 1.0,
        step_planes = 0, replay_steps = dense ? Float64[] : steps,
        replay_edges = dense ? Float64[] : [0.0; cumsum(steps)],
    )
    control = SP.PIControl(;
        safety = 0.8, max_factor = 2.0, min_factor = 0.2, beta1 = 0.3, beta2 = -0.1,
        lag = 0, reject_free = false, estimator_order = 1, start_growth = 100.0,
        dz_min = 1.0e-12, dz_max = 0.2, auto_init = false,
    )
    tableau = SP.EmbeddedTableau(
        [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
    )
    norm = SP.MarchNorm(
        execution, fill(true, size(field, 1)), fill(true, size(field, 1)),
        ones(size(field, 1)), 1.0e-5, 1.0e-10,
    )
    operator = SP.DiagonalOperator(execution, zeros(ComplexF64, size(field)))
    source = TransverseSessionSource(nothing, 0)
    session = SP.RungeKuttaSession(
        execution, field, 0.0, dense ? 0.2 : first(steps), tableau, control, norm,
        schedule, source, SP.LinearFlow(execution, operator, field);
        services, capacity, dense_weights = dense ? zeros(2) : nothing,
    )
    return session, policy
end

function transverse_session_warning_field!(session)
    work, field = session.services.transverse, session.field
    fill!(field, 1.0 + 0im)
    SP._transverse_form_reading!(work, field; faces = true, full = true)
    wanted = work.state.baseline[1] + 0.25
    field .*= sqrt(wanted * work.state.launch / work.reading.faces[1])
    # Controlled admissible accumulated reading facts: the decision leaf already
    # qualifies these values separately. This test isolates service cursor order.
    work.state.ramp_raw = 0.25
    work.state.spatial_removed = 0.0
    return nothing
end

function transverse_session_collect(session; work_limit = 4, deadline = 60.0)
    events = Tuple{SP.SessionEvent, Float64, Float64}[]
    notices = SP.MonitorNotice[]
    rows = Tuple{Float64, Float64}[]
    states = []
    while !session.state.finished
        report = SP.advance!(session; work_limit, deadline)
        @test report.kind != SP.SESSION_FAILED
        report.kind == SP.SESSION_FAILED && error(session.state.failure_message)
        push!(events, (report.kind, report.z, session.state.z))
        payload = SP.session_monitor_payload(session, report.generation)
        @test Base.datatype_pointerfree(typeof(payload))
        @test isequal(payload, SP.session_monitor_payload(session, report.generation))
        push!(states, payload.transverse)
        payload.notice === nothing || push!(notices, payload.notice)
        values = SP.observation_rows(session.services)
        for index in 1:values.count
            push!(rows, (values.z[index], values.values[1, index]))
        end
        SP.acknowledge!(session, report.generation)
    end
    return (; events, notices, rows, states)
end

function transverse_session_allocation!(session)
    report = SP.advance!(session; work_limit = 1, deadline = 60.0)
    SP.acknowledge!(session, report.generation)
    report = SP.advance!(session; work_limit = 1, deadline = 60.0)
    SP.acknowledge!(session, report.generation)
    return @allocated SP.advance!(session; work_limit = 1, deadline = 60.0)
end

@testset "Transverse services retain one warning and exact launch ordering" begin
    session, _ = transverse_session_case(;
        temporal = true, temporal_warning = true, clip = true, coarse = true,
    )
    initial = SP.session_monitor_payload(session, 0)
    @test initial.record !== nothing && initial.transverse !== nothing
    @test isnan(initial.transverse.continuation.launch) && !initial.attached
    first = SP.advance!(session; deadline = 60.0)
    @test first.kind == SP.SESSION_LAUNCH && !session.services.transverse.started
    SP.acknowledge!(session, first.generation)
    temporal = SP.advance!(session; deadline = 60.0)
    @test temporal.kind == SP.SESSION_WARNING
    @test session.services.notices.records[1].code == SP.TemporalGuardWarning
    @test !session.services.transverse.started && session.source.calls == 0
    SP.acknowledge!(session, temporal.generation)
    for face in 1:2
        report = SP.advance!(session; deadline = 60.0)
        payload = SP.session_monitor_payload(session, report.generation)
        @test report.kind == SP.SESSION_WARNING
        @test payload.notice.code == SP.TransverseClipWarning && payload.notice.face == face
        @test session.services.continuation == SP.ObservationTransverseLaunch
        @test session.services.transverse.state.face == face + 1
        @test session.source.calls == 0 && session.state.accepted == 0
        @test session.services.rows.count == 0
        @test_throws ArgumentError SP.advance!(session)
        SP.acknowledge!(session, report.generation)
    end
    coarse = SP.advance!(session; deadline = 60.0)
    @test coarse.kind == SP.SESSION_WARNING
    notice = SP.session_monitor_payload(session, coarse.generation).notice
    @test notice.code == SP.TransverseCoarseWarning && notice.value == 0.01
    @test notice.auxiliary == 0.005
    @test session.services.transverse.state.stage == 5 && session.source.calls == 0
    SP.acknowledge!(session, coarse.generation)
    collected = transverse_session_collect(session)
    @test last(collected.events)[1] == SP.SESSION_COMPLETED
    @test session.state.accepted == 4 && session.source.calls == 8
    @test session.services.transverse.state.stage == 0
    @test length(collected.rows) == 4
end

@testset "Transverse yield invariance, replay lookahead and bounded storage" begin
    for tasks in (1, 2), route in (:radial, :cartesian, :line)
        whole, _ = transverse_session_case(; tasks, route)
        bounded, _ = transverse_session_case(; tasks, route, capacity = 1)
        initial = copy(whole.field)
        left = transverse_session_collect(whole)
        right = transverse_session_collect(bounded; work_limit = 1, deadline = 0.0)
        @test isempty(left.notices) && isempty(right.notices)
        @test whole.field == bounded.field == initial
        @test whole.source.calls == bounded.source.calls == 8
        @test isequal(last(left.states), last(right.states))
        @test first.(left.rows) == first.(right.rows) == cumsum(fill(0.01, 4))
        # Identical constant states use the same independently prepared transform
        # path; 256 eps times absolute energy covers their short reduction depth.
        budget = 256eps(Float64) * maximum(abs, last.(left.rows))
        @test budget > 0
        @test all(abs.(last.(left.rows) - last.(right.rows)) .<= budget)
        @test length(bounded.services.rows.z) == 2
        @test length(bounded.services.notices.records) == 1
        @test length(bounded.services.monitor_rows) == 2
        allocation, _ = transverse_session_case(; tasks, route, capacity = 1)
        @test transverse_session_allocation!(allocation) == 0
    end
    lookahead, _ = transverse_session_case(; steps = [0.001, 0.08, 0.001], capacity = 1)
    report = SP.advance!(lookahead; deadline = 60.0)
    SP.acknowledge!(lookahead, report.generation)
    report = SP.advance!(lookahead; work_limit = 1, deadline = 60.0)
    @test report.kind == SP.SESSION_WARNING
    @test lookahead.services.notices.records[1].code == SP.TransverseCoarseWarning
    SP.acknowledge!(lookahead, report.generation)
    report = SP.advance!(lookahead; work_limit = 1, deadline = 60.0)
    @test report.kind == SP.SESSION_PROGRESS && lookahead.state.z == 0.001
    @test lookahead.state.proposal == 0.001 && lookahead.state.replay == 2
    @test lookahead.services.transverse.state.reads == 1
    @test lookahead.services.transverse.state.last_z == 0.001
    @test lookahead.services.transverse.committed_z == 0.001
end

@testset "Accepted and plane warnings drain one held reading before statistics" begin
    for at_plane in (false, true)
        session, _ = transverse_session_case(;
            collar = false, capacity = 1, steps = [0.01, 0.01],
        )
        launch = SP.advance!(session; deadline = 60.0)
        SP.acknowledge!(session, launch.generation)
        first = SP.advance!(session; work_limit = 1, deadline = 60.0)
        @test first.kind == SP.SESSION_PROGRESS && session.state.accepted == 1
        SP.acknowledge!(session, first.generation)
        transverse_session_warning_field!(session)
        edge = session.services.transverse
        edge.state.next = at_plane ? Inf : 0.0
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        @test report.kind == SP.SESSION_WARNING
        @test session.services.notices.records[1].code == SP.TransverseFaceWarning
        wanted = at_plane ? SP.ObservationTransversePlane : SP.ObservationTransverseAccepted
        @test session.services.continuation == wanted
        @test edge.state.stage == 6 && edge.state.reads == 1
        @test session.state.accepted == 2 && session.source.calls == 4
        @test !session.services.plane_valid
        @test session.services.rows.count == (at_plane ? 1 : 0)
        before = SP.transverse_continuation(edge.state)
        SP.acknowledge!(session, report.generation)
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        @test report.kind == SP.SESSION_WARNING
        @test session.services.notices.records[1].code == SP.TransverseRampWarning
        @test session.services.continuation == wanted
        @test edge.state.stage == 0 && edge.state.reads == before.reads
        @test edge.state.ramp_raw == before.ramp_raw && edge.state.last_z == 0.02
        @test session.source.calls == 4 && !session.services.plane_valid
        @test session.services.rows.count == 0
        SP.acknowledge!(session, report.generation)
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        @test report.kind == SP.SESSION_PLANE && session.services.plane_valid
        @test session.services.rows.count == (at_plane ? 0 : 1)
        if !at_plane
            @test session.services.rows.step[1] == -1
            @test session.services.rows.z[1] == 0.02
        end
        @test edge.state.reads == 1 && session.source.calls == 4
        SP.acknowledge!(session, report.generation)
        report = SP.advance!(session; deadline = 60.0)
        @test report.kind == SP.SESSION_COMPLETED && session.services.rows.count == 0
    end
end

@testset "Transverse dense planes never rewind committed current" begin
    session, _ = transverse_session_case(;
        dense = true, planes = [0.01, 0.02, 0.03, 0.04], temporal = true,
    )
    initial = copy(session.field)
    rows = 0
    while !session.state.finished
        report = SP.advance!(session; deadline = 60.0)
        @test report.kind != SP.SESSION_FAILED && report.kind != SP.SESSION_WARNING
        if report.kind == SP.SESSION_PLANE
            @test session.state.z == 0.04 && report.z <= session.state.z
            @test session.services.transverse.committed_z == 0.04
            @test session.services.transverse.state.current_last_z == 0.04
            @test session.services.transverse.state.last_z == 0.04
            @test session.services.transverse.state.reads == 1
            @test SP._march_observed_field(session) == initial
            @test session.services.rows.count == (report.z == 0.01 ? 1 : 0)
            rows += session.services.rows.count
        end
        SP.acknowledge!(session, report.generation)
    end
    @test session.state.accepted == 1 && session.source.calls == 2 && rows == 1
end

@testset "Transverse monitor transport and restore preserve detached generations" begin
    old, _ = transverse_session_case(; temporal = true)
    transverse_session_collect(old)
    edge_record = SP.transverse_monitor_record(old.services.transverse).continuation
    temporal_record = SP.temporal_continuation(old.services.work.monitor)
    fresh, _ = transverse_session_case(; temporal = true)
    radial, _ = transverse_session_case(; route = :radial)
    transverse_session_collect(radial)
    wrong = SP.transverse_monitor_record(radial.services.transverse).continuation
    before = SP.session_monitor_payload(fresh, 0)
    @test_throws DimensionMismatch SP.restore_session_monitor!(
        fresh, temporal_record, false, wrong,
    )
    @test isequal(before, SP.session_monitor_payload(fresh, 0))
    @test !fresh.services.transverse_restored
    SP.restore_session_monitor!(fresh, temporal_record, false, edge_record)
    restored = SP.session_monitor_payload(fresh, 0)
    @test restored.record.resuming && restored.transverse.resumed
    @test restored.transverse.continuation.launch == edge_record.launch
    @test restored.transverse.continuation.current == edge_record.current
    SP.restore_session_monitor!(fresh, nothing, true, nothing)
    @test isequal(restored, SP.session_monitor_payload(fresh, 0))
    report = SP.advance!(fresh; deadline = 60.0)
    SP.session_monitor_rows!(zeros(5, 0), fresh, report.generation)
    @test_throws DimensionMismatch SP.session_monitor_rows!(
        fresh.services.rows.values, fresh, report.generation,
    )
    SP.acknowledge!(fresh, report.generation)
    report = SP.advance!(fresh; work_limit = 1, deadline = 60.0)
    @test_throws ArgumentError SP.restore_session_monitor!(
        fresh, temporal_record, false, edge_record,
    )
    @test_throws ArgumentError SP.session_monitor_payload(fresh, report.generation + 1)
    payload = SP.session_monitor_payload(fresh, report.generation)
    @test payload.transverse.resumed && payload.attached
    @test Base.datatype_pointerfree(typeof(payload))
    only_edge, _ = transverse_session_case()
    payload = SP.session_monitor_payload(only_edge, 0)
    @test payload.record === nothing && payload.transverse !== nothing &&
        payload.row_count == 0
    SP.session_monitor_rows!(zeros(5, 0), only_edge, 0)
    @test (@allocated SP.session_monitor_payload(only_edge, 0)) == 0
    SP.restore_session_monitor!(only_edge, nothing, false, nothing)
    restored = SP.session_monitor_payload(only_edge, 0)
    @test restored.resuming && isnan(restored.committed) && !restored.transverse.resumed
    @test only_edge.services.transverse_restored
    SP.restore_session_monitor!(only_edge, nothing, false, edge_record)
    @test isequal(restored, SP.session_monitor_payload(only_edge, 0))
    report = SP.advance!(only_edge; deadline = 60.0)
    @test SP.session_monitor_payload(only_edge, report.generation).resuming
    SP.acknowledge!(only_edge, report.generation)
    report = SP.advance!(only_edge; work_limit = 1, deadline = 60.0)
    attached = SP.session_monitor_payload(only_edge, report.generation)
    @test !attached.resuming && attached.committed == only_edge.state.z

    # A resumed lifecycle with no supplied record has not selected its later
    # output fallback. Explicit restoration, including nothing, selects it once.
    fallback, policy = transverse_session_case(; temporal = true)
    preparation = SP.SessionPreparation(
        SP.SpatialPreparation(
            fallback.execution, nothing, fallback.flow.operator,
            SP.ObservationWeight(fallback.execution, ones(size(fallback.field))), 3,
        ), policy,
    )
    services = SP._launch_observation_services(
        preparation, copy(fallback.field), nothing, fallback.norm.active, 0.0, true,
    )
    fallback = SP.RungeKuttaSession(
        fallback.execution, fallback.field, 0.0, 0.01, fallback.tableau,
        fallback.control, fallback.norm, fallback.schedule, fallback.source,
        fallback.flow; services,
    )
    @test services.resume_pending && !services.transverse_restored
    @test !services.transverse.state.resumed && !services.work.monitor.resuming
    SP.restore_session_monitor!(fallback, temporal_record, false, edge_record)
    resumed = SP.session_monitor_payload(fallback, 0)
    @test services.transverse_restored && resumed.record.resuming &&
        resumed.transverse.resumed
    @test resumed.transverse.continuation.launch == edge_record.launch
    @test resumed.transverse.continuation.current == edge_record.current
    SP.restore_session_monitor!(fallback, nothing, true, nothing)
    @test isequal(resumed, SP.session_monitor_payload(fallback, 0))
end

@testset "Launch noise ordering, optional clean data and complete retained ledger" begin
    session, policy = transverse_session_case()
    execution, field = session.execution, copy(session.field)
    problem = SP.SpatialPreparation(
        execution, nothing, session.flow.operator,
        SP.ObservationWeight(execution, ones(size(field))), 3,
    )
    preparation = SP.SessionPreparation(problem, policy)
    noise = SP.InputNoise(execution, 0.03field, 0.0, session.flow.operator)
    active = Bool[false, true, true, false, true]
    expected = SP.transverse_clean_columns(execution, field, noise.spectrum, active)
    fresh = copy(field)
    services = SP._launch_observation_services(
        preparation, fresh, noise, active, 0.0, false,
    )
    @test services.transverse.clean_columns == expected
    @test fresh[active, :] == (field + noise.spectrum)[active, :]
    @test all(iszero, fresh[.!active, :]) && noise.spectrum == 0.03field
    later = copy(field)
    next_services = SP._launch_observation_services(
        preparation, later, noise, active, 0.2, true,
    )
    @test later[active, :] == field[active, :] && all(iszero, later[.!active, :])
    @test next_services.transverse.clean_columns ==
        SP.transverse_fallback_clean_columns(execution, later, noise.spectrum)
    @test next_services.transverse.clean_columns != expected
    transverse_session_collect(session)
    continuation = SP.transverse_continuation(session.services.transverse.state)
    restored = SP._launch_observation_services(
        preparation, copy(field), noise, active, 0.2, true;
        transverse_continuation = continuation,
    )
    @test restored.transverse.clean_columns === nothing &&
        restored.transverse.state.resumed
    raw = copy(field)
    @test SP._launch_observation_services(problem, raw, noise, active, 0.0, true) isa
        SP._NoSessionServices
    @test raw[active, :] == field[active, :] && all(iszero, raw[.!active, :])
    @test_throws ArgumentError SP.SessionObservationPolicy(
        policy.observation; trace_chunk = 0,
    )
    @test_throws ArgumentError SP.SessionObservationPolicy(
        policy.observation; transverse = 1,
    )
    edge = policy.transverse
    wrong = SP.TransversePreparation(
        execution, edge.map, edge.policy, edge.tables; n_frequency = edge.n_frequency + 1,
        rows = edge.rows, lines = edge.lines, declared = false,
    )
    @test_throws DimensionMismatch SP.SessionObservationPolicy(
        policy.observation; transverse = wrong,
    )
    ledger = SP.session_memory(session)
    @test any(contains("transverse"), string.(keys(ledger)))
    @test !any(contains(".transverse.data"), string.(keys(ledger)))
    prepared_bytes = SP.preparation_memory(preparation)
    @test any(contains("transverse.tables.faces"), string.(keys(prepared_bytes)))
    @test sum(ledger) > sizeof(session.services.transverse.band)
    @test sum(prepared_bytes) >= sum(sizeof, (edge.tables.faces, edge.tables.totals))
end
