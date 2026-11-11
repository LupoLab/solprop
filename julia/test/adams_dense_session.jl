# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

mutable struct DenseSessionSource{E, A}
    execution::E
    forcing::A
    calls::Int
    fail_on::Int
end

function (source::DenseSessionSource)(output, field, position)
    source.calls += 1
    source.calls == source.fail_on && error("controlled dense source failure")
    forcing = source.forcing
    SP.each!(source.execution, size(output)...) do row, column
        output[row, column] = forcing[row, column]
    end
    return nothing
end

function adams_dense_fixture(;
        tasks = 1, capacity = 4, planes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
        step_planes = 0, initial = fill(1.0 + 0.2im, 3, 2),
        forcing = fill(0.2 + 0.1im, size(initial)), rate = 0.0im,
        services = SP._NoSessionServices(), max_step = 0.2, hard_kinks = Float64[],
        source_absent = false, fail_on = 0, active = fill(true, size(initial, 1)),
        varying = false, rtol = 1.0e-5,
    )
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    policy = adams_session_policy(;
        order = 3, variable_order = false, dz_max = max_step,
        equal_approach = false, z_end = last(planes), save_planes = Float64[],
        breakpoints = [0.0, last(planes)], hard_kinks, interpolate = true,
    )
    schedule = SP.MarchSchedule(;
        planes, breakpoints = policy.breakpoints, hard_kinks,
        stretch = policy.stretch, step_planes,
    )
    norm = SP.MarchNorm(
        execution, active, fill(true, size(initial, 1)), ones(size(initial, 1)),
        rtol, 1.0e-8,
    )
    source = DenseSessionSource(execution, copy(forcing), 0, fail_on)
    operator = if varying
        # Phi(z)=rate*z²/2 on [0,end], obtained from x=2z/end-1 and x²=(T₂+1)/2.
        scale = last(planes)^2 * rate / 16
        coefficients = zeros(ComplexF64, length(initial), 3, 1)
        coefficients[:, 1, 1] .= 3scale
        coefficients[:, 2, 1] .= 4scale
        coefficients[:, 3, 1] .= scale
        SP.RepresentedOperator(
            execution, SP.ChebyshevTable(execution, [0.0, last(planes)], coefficients),
            size(initial)...,
        )
    else
        SP.DiagonalOperator(execution, fill(ComplexF64(rate), size(initial)))
    end
    flow = SP.LinearFlow(execution, operator, initial)
    return SP.AdamsSession(
        execution, initial, 0.0, max_step, policy, norm, schedule, source, flow;
        capacity, services, source_absent,
    )
end

function dense_session_collect(session; work_limit = 4, deadline = 60.0, snapshot = false)
    events = Tuple{SP.SessionEvent, Float64, Float64}[]
    fields = Matrix{ComplexF64}[]
    attempts = Tuple{Float64, Float64, Float64, Bool}[]
    while !session.state.finished
        report = SP.advance!(session; work_limit, deadline)
        @test report.kind != SP.SESSION_FAILED
        push!(events, (report.kind, report.z, session.state.z))
        if report.kind == SP.SESSION_PLANE
            push!(fields, copy(SP._march_observed_field(session)))
            if snapshot && report.z == first(session.schedule.planes)
                SP.request_snapshot!(session)
            end
        end
        for index in 1:session.attempts.count
            buffer = session.attempts
            push!(
                attempts, (
                    buffer.z[index], buffer.dz[index], buffer.norm[index],
                    buffer.accepted[index],
                )
            )
        end
        @test SP._march_observed_position(session) == report.z
        SP.acknowledge!(session, report.generation)
        @test SP._march_observed_field(session) === session.field
        @test SP._march_observed_position(session) == session.state.z
    end
    return events, fields, attempts
end

@testset "Adaptive Adams delivers every dense plane without replacing committed state" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        whole = adams_dense_fixture(; tasks)
        bounded = adams_dense_fixture(; tasks, capacity = 1)
        initial, forcing = copy(whole.field), copy(whole.source.forcing)
        events, fields, attempts = dense_session_collect(whole; snapshot = true)
        split_events, split_fields, split_attempts = dense_session_collect(
            bounded; work_limit = 1, deadline = 0.0, snapshot = true,
        )
        @test length(attempts) == length(split_attempts)
        @test map(row -> (row[1], row[2], row[4]), attempts) ==
            map(row -> (row[1], row[2], row[4]), split_attempts)
        # Delivery changes no recurrence/history. Permit rounding-level arithmetic
        # differences using fixed norm/field operation budgets rather than bits.
        for (left, right) in zip(attempts, split_attempts)
            budget = 4096eps(Float64) * max(abs(left[3]), abs(right[3]), 1.0)
            @test budget > 0 && abs(left[3] - right[3]) <= budget
        end
        field_budget = 128eps(Float64) *
            (maximum(abs, initial) + last(whole.schedule.planes) * maximum(abs, forcing))
        @test field_budget > 0
        @test length(fields) == length(split_fields)
        @test all(
            maximum(abs, left - right) <= field_budget
                for (left, right) in zip(fields, split_fields)
        )
        @test maximum(abs, whole.field - bounded.field) <= field_budget
        @test events == split_events
        @test whole.state.accepted == 2
        @test whole.ledger.source_evaluations == whole.source.calls == 3
        @test whole.ledger.dense_observations == 4
        @test [(event[2], event[3]) for event in events if event[1] == SP.SESSION_PLANE] ==
            [(0.05, 0.2), (0.1, 0.2), (0.15, 0.2), (0.2, 0.2), (0.3, 0.4), (0.4, 0.4)]
        snapshot = findfirst(event -> event[1] == SP.SESSION_SNAPSHOT, events)
        endpoint = findfirst(event -> event[1:2] == (SP.SESSION_PLANE, 0.2), events)
        @test snapshot == endpoint + 1
        @test events[snapshot][2:3] == (0.2, 0.2)
        @test fields[4] == whole.observation.work.dense.initial
        @test fields[end] == whole.field
        for (position, field) in zip(whole.schedule.planes, fields)
            expected = ComplexF64.(
                Complex{BigFloat}.(initial) +
                    BigFloat(position) .* Complex{BigFloat}.(forcing)
            )
            # At most two steps and degree-three sums on an exactly linear solution;
            # 512 eps bounds the absolute linear contributions and dense arithmetic.
            budget = 512eps(Float64) .* (abs.(initial) .+ position .* abs.(forcing))
            @test all(budget .> 0)
            @test all(abs.(field .- expected) .<= budget)
        end
    end
end

@testset "Dense planes precede owed step observations and endpoint snapshots" begin
    session = adams_dense_fixture(;
        planes = [0.05, 0.1, 0.15, 0.3, 0.4], step_planes = 1,
    )
    events, _, _ = dense_session_collect(session; snapshot = true)
    step = findfirst(event -> event[1] == SP.SESSION_STEP, events)
    @test events[step - 1][1:2] == (SP.SESSION_PLANE, 0.15)
    @test events[step][2:3] == (0.2, 0.2)
    @test events[step + 1][1:2] == (SP.SESSION_SNAPSHOT, 0.2)
    @test session.ledger.dense_observations == 4
end

@testset "Dense homogeneous damping and hard-kink restart retain accepted windows" begin
    for rate in (-0.2 + 0.3im, -100.0 + 0.2im)
        session = adams_dense_fixture(;
            rate, source_absent = true, hard_kinks = [0.2],
        )
        initial = copy(session.field)
        _, fields, _ = dense_session_collect(session)
        @test session.ledger.restarts == 1
        @test session.source.calls == session.ledger.source_evaluations == 0
        @test session.ledger.dense_observations == 4
        for (position, field) in zip(session.schedule.planes, fields)
            expected = ComplexF64.(
                Complex{BigFloat}.(initial) .*
                    exp(Complex{BigFloat}(rate) * BigFloat(position))
            )
            # One/two flows and the retained phi path; protect strongly damped
            # absolute magnitudes rather than a unit-scale absolute tolerance.
            budget = 2048eps(Float64) .* abs.(expected)
            @test all(budget .> 0)
            @test all(abs.(field .- expected) .<= budget)
        end
    end
end

@testset "Dense observation failures retain the committed endpoint and actual counts" begin
    session = adams_dense_fixture()
    adams_session_ack_launch(session)
    plane = SP.advance!(session; deadline = 60.0)
    @test plane.kind == SP.SESSION_PLANE && plane.z == 0.05
    committed = copy(session.field)
    SP.acknowledge!(session, plane.generation)
    session.observation.work.dense.coefficients[1] = Inf
    failed = SP.advance!(session; deadline = 60.0)
    @test failed.kind == SP.SESSION_FAILED && !failed.poisoned
    @test session.field == committed && failed.z == session.state.z == 0.2
    @test SP._march_observed_field(session) === session.field
    @test session.ledger.dense_observations == 1
    @test session.ledger.source_evaluations == session.source.calls == 2

    source_failure = adams_dense_fixture(; fail_on = 2)
    adams_session_ack_launch(source_failure)
    failed = SP.advance!(source_failure; deadline = 60.0)
    @test failed.kind == SP.SESSION_FAILED && !failed.poisoned
    @test !source_failure.observation.work.ready
    @test source_failure.ledger.dense_observations == 0
    @test source_failure.ledger.source_evaluations == 2
end

function dense_service_fixture(; tasks = 2, monitor = false, warn = 0.05, split = false)
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    script = service_case(
        execution; monitor, warn, split, cadence = 100, interval = 1.0,
    )
    forcing = service_spectrum(2 .* service_samples(:envelope, 1), :envelope)
    return adams_dense_fixture(;
        tasks, initial = script.field, forcing, services = script.services,
    )
end

@testset "Dense planes consume the previous stage diagnostic once" begin
    session = dense_service_fixture()
    SP.acknowledge!(session, SP.advance!(session).generation)
    # This fixture's source has no embedded quadrature. Supply a recognizable
    # previous-stage diagnostic to test the existing per-plane reset contract.
    session.services.quadrature_error = 0.125
    first_plane = SP.advance!(session; deadline = 60.0)
    @test first_plane.kind == SP.SESSION_PLANE && first_plane.z < session.state.z
    @test session.services.quadrature_error == 0.125
    @test !session.services.periodic_pending
    @test session.services.rows.count == 1
    @test session.services.rows.step[1] == 0
    @test session.services.rows.z[1] == session.attempts.z[1]
    calls = session.ledger.predictor + session.ledger.corrected + session.ledger.startup
    SP.acknowledge!(session, first_plane.generation)
    @test isnan(session.services.quadrature_error)
    next_plane = SP.advance!(session; deadline = 60.0)
    @test next_plane.kind == SP.SESSION_PLANE && next_plane.z < session.state.z
    @test session.services.rows.count == 0
    @test isnan(session.services.quadrature_error)
    @test session.ledger.predictor + session.ledger.corrected +
        session.ledger.startup == calls
end

@testset "Dense services retain endpoint chunks and preserve stale monitor checks" begin
    session = dense_service_fixture(; monitor = true)
    services, monitor = session.services, session.services.work.monitor
    events = Tuple{SP.SessionEvent, Float64}[]
    planes = SP.TemporalPlaneRow[]
    periodic = Tuple{Float64, Int}[]
    committed = 0.0
    delivered_attempts = 0
    for iteration in 1:30
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        @test report.kind != SP.SESSION_FAILED
        push!(events, (report.kind, report.z))
        @test session.state.z >= committed
        committed = session.state.z
        if services.started
            @test monitor.committed == committed
        end
        for index in 1:services.monitor_count
            push!(planes, services.monitor_rows[index])
        end
        for index in 1:services.rows.count
            position = services.rows.z[index]
            push!(periodic, (position, delivered_attempts + services.rows.step[index]))
            # Parseval for two disjoint time samples: (1+4z²)*|amplitude|².
            expected = Float64.((1 + 4BigFloat(position)^2) .* BigFloat.([1.0, 0.7]) .^ 2)
            budget = 1024eps(Float64) .* expected
            @test all(budget .> 0)
            @test all(abs.(services.rows.values[2:3, index] - expected) .<= budget)
        end
        delivered_attempts += session.attempts.count
        if report.kind == SP.SESSION_WARNING
            @test report.z == session.state.z
            @test SP._march_observed_field(session) === session.field
        end
        if report.kind == SP.SESSION_PLANE
            expected = Float64.((1 + 4BigFloat(report.z)^2) .* BigFloat.([1.0, 0.7]) .^ 2)
            # Same fixed DFT/map/sum bound as periodic rows, now at observed z.
            budget = 1024eps(Float64) .* expected
            @test all(budget .> 0)
            @test all(abs.(services.plane_values[1:2] - expected) .<= budget)
            if report.z < session.state.z
                @test !services.periodic_pending
                if services.rows.count > 0
                    @test services.rows.step[1] >= 0
                    @test session.attempts.z[services.rows.step[1] + 1] ==
                        services.rows.z[1]
                end
                @test services.plane_z == report.z
            end
        end
        SP.acknowledge!(session, report.generation)
        session.state.finished && break
    end
    @test last(events)[1] == SP.SESSION_COMPLETED
    @test any(event -> first(event) == SP.SESSION_WARNING, events)
    @test [row.z for row in planes] == [0.0; session.schedule.planes]
    # The first stale dense check judges the interpolated field once. Later planes
    # in that same accepted window are unmeasured, just as shared/native monitoring.
    @test isfinite(planes[2].collar)
    @test isnan(planes[3].collar) && isnan(planes[4].collar)
    @test periodic == [(0.2, 0), (0.4, 1)]
    @test monitor.steps + monitor.skipped == session.state.accepted
    @test monitor.committed == session.state.z == 0.4
    @test session.ledger.dense_observations == 4
end

@testset "Dense admission preserves the shared replay and split refusals" begin
    session = adams_dense_fixture()
    schedule = SP.MarchSchedule(;
        planes = session.schedule.planes, breakpoints = session.schedule.breakpoints,
        hard_kinks = Float64[], stretch = session.schedule.stretch, step_planes = 0,
        replay_steps = [0.2, 0.2], replay_edges = [0.0, 0.2, 0.4],
    )
    @test_throws ArgumentError SP.AdamsSession(
        session.execution, session.field, 0.0, 0.2, session.policy,
        session.norm, schedule, session.source, session.flow,
    )
    @test_throws ArgumentError dense_service_fixture(; monitor = true, split = true)
end

@testset "Dense compensated sessions retain actual predictor polynomial samples" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        session = adams_dense_fixture(;
            tasks, varying = true, rate = -0.4 + 0.2im, rtol = 0.1,
        )
        @test session.work.compensated
        interior = 0
        while !session.state.finished
            report = SP.advance!(session; deadline = 60.0)
            @test report.kind != SP.SESSION_FAILED
            if report.kind == SP.SESSION_PLANE && report.z < session.state.z
                interior += 1
                observed = SP._march_observed_field(session)
                expected, budget = setprecision(BigFloat, 256) do
                    adams_observation_reference(session.observation.work.dense, report.z)
                end
                # Independent high-precision Vandermonde and defining phi moments;
                # the shared leaf witness propagates absolute sums and conditioning.
                @test budget > 0
                @test maximum(abs, observed .- expected) <= budget
                @test !Base.mightalias(observed, session.field)
            end
            SP.acknowledge!(session, report.generation)
        end
        @test interior == session.ledger.dense_observations > 0
        @test session.ledger.source_evaluations == session.source.calls
    end
end

function dense_session_drain!(session)
    while !session.state.finished
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        report.kind == SP.SESSION_FAILED && error(session.state.failure_message)
        SP.acknowledge!(session, report.generation)
    end
    return nothing
end

function dense_session_allocations(session)
    return @allocated dense_session_drain!(session)
end

@testset "Warmed dense advancement and warning continuations allocate no storage" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        for varying in (false, true)
            warm = adams_dense_fixture(; tasks, varying, rtol = 0.1, rate = -0.4 + 0.2im)
            measured = adams_dense_fixture(;
                tasks, varying, rtol = 0.1, rate = -0.4 + 0.2im,
            )
            dense_session_allocations(warm)
            @test dense_session_allocations(measured) == 0
            @test measured.ledger.dense_observations > 0
            @test measured.state.accepted <= 10
        end
        dense_session_allocations(dense_service_fixture(; tasks, monitor = true))
        measured = dense_service_fixture(; tasks, monitor = true)
        @test dense_session_allocations(measured) == 0
        @test measured.ledger.dense_observations == 4
    end
end

@testset "Every retained dense array is counted once in the session memory ledger" begin
    landed = adams_modal_session(1)
    policy = adams_session_policy(;
        order = 1, variable_order = false, interpolate = true, save_planes = Float64[],
    )
    schedule = SP.MarchSchedule(;
        planes = [0.05, 0.1, 0.2], breakpoints = policy.breakpoints,
        hard_kinks = policy.hard_kinks, stretch = policy.stretch, step_planes = 0,
    )
    dense = SP.AdamsSession(
        landed.execution, landed.field, 0.0, 0.025, policy, landed.norm, schedule,
        landed.source, landed.flow,
    )
    ledger = SP.session_memory(dense)
    observation = dense.observation.work
    arrays = (
        SP._adams_observation_arrays(observation.dense)...,
        observation.field, observation.failures,
    )
    prefix = "session.observation.work."
    listed = [
        getproperty(ledger, name) for name in keys(ledger)
            if startswith(string(name), prefix)
    ]
    @test length(listed) == length(arrays)
    @test sum(listed) == sum(sizeof, arrays)
    @test length(unique(objectid.(arrays))) == length(arrays)
end
