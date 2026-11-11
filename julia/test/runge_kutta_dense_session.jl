# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

mutable struct RKDenseSource{E, A}
    execution::E
    forcing::A
    coefficient::ComplexF64
    calls::Int
    fail_on::Int
end

function (source::RKDenseSource)(output, field, position)
    source.calls += 1
    source.calls == source.fail_on && error("controlled RK dense source failure")
    forcing, coefficient = source.forcing, source.coefficient
    SolpropJulia.each!(source.execution, size(output)...) do row, column
        output[row, column] = coefficient * field[row, column] + forcing[row, column]
    end
    return nothing
end

function rk_dense_session(;
        tasks = 1, planes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
        capacity = 4, step_planes = 0, initial = fill(1.0 + 0.2im, 2, 2),
        forcing = fill(0.2 + 0.1im, size(initial)), coefficient = 0.0im, rate = 0.0im,
        services = SolpropJulia._NoSessionServices(), hard_kinks = Float64[],
        active = fill(true, size(initial, 1)), fail_on = 0, max_step = 0.2,
        rtol = 1.0e-5, dense = true, replay = false, weights = rk_dense_weights(),
    )
    execution = rk_dense_execution(tasks)
    control = SolpropJulia.PIControl(;
        safety = 0.9, max_factor = 2.0, min_factor = 0.2, beta1 = 0.2, beta2 = -0.04,
        lag = 0, reject_free = false, estimator_order = 4, start_growth = 100.0,
        dz_min = 1.0e-12, dz_max = max_step, auto_init = false,
    )
    norm = SolpropJulia.MarchNorm(
        execution, active, fill(true, size(initial, 1)), ones(size(initial, 1)),
        rtol, 1.0e-10,
    )
    source = RKDenseSource(execution, copy(forcing), ComplexF64(coefficient), 0, fail_on)
    operator = SolpropJulia.DiagonalOperator(
        execution, fill(ComplexF64(rate), size(initial)),
    )
    flow = SolpropJulia.LinearFlow(execution, operator, initial)
    schedule = SolpropJulia.MarchSchedule(;
        planes, breakpoints = [0.0, last(planes)], hard_kinks, stretch = 1.1, step_planes,
        replay_steps = replay ? fill(0.05, 8) : Float64[],
        replay_edges = replay ? collect(0.0:0.05:0.4) : Float64[],
    )
    return SolpropJulia.RungeKuttaSession(
        execution, initial, 0.0, max_step, rk_dense_tableau(), control, norm,
        schedule, source, flow; services, capacity,
        dense_weights = dense ? weights : nothing,
    )
end

function rk_dense_collect(session; work_limit = 256, deadline = 60.0, snapshot = false)
    events = Tuple{SolpropJulia.SessionEvent, Float64, Float64}[]
    fields = Matrix{ComplexF64}[]
    attempts = Tuple{Float64, Float64, Float64, Bool}[]
    while !session.state.finished
        report = SolpropJulia.advance!(session; work_limit, deadline)
        @test report.kind != SolpropJulia.SESSION_FAILED
        push!(events, (report.kind, report.z, session.state.z))
        if report.kind == SolpropJulia.SESSION_PLANE
            push!(fields, copy(SolpropJulia._march_observed_field(session)))
            if snapshot && report.z == first(session.schedule.planes)
                SolpropJulia.request_snapshot!(session)
            end
        end
        for index in 1:session.attempts.count
            b = session.attempts
            push!(attempts, (b.z[index], b.dz[index], b.norm[index], b.accepted[index]))
        end
        @test SolpropJulia._march_observed_position(session) == report.z
        SolpropJulia.acknowledge!(session, report.generation)
        @test SolpropJulia._march_observed_field(session) === session.field
        @test SolpropJulia._march_observed_position(session) == session.state.z
    end
    return events, fields, attempts
end

@testset "RK dense session separates planes from accepted endpoints" begin
    for tasks in (1, 2)
        session = rk_dense_session(; tasks, step_planes = 1, capacity = 2)
        initial = copy(session.field)
        events, fields, attempts = rk_dense_collect(session; snapshot = true)
        planes = [event[2] for event in events if event[1] == SolpropJulia.SESSION_PLANE]
        @test planes == session.schedule.planes
        @test length(attempts) == 2
        @test session.source.calls == 7length(attempts)
        @test session.state.accepted == 2
        for (position, field) in zip(planes, fields)
            expected = initial + position * session.source.forcing
            # Two at-most-seven-term stage/extension sums and linear field
            # additions; 128eps scale exceeds their gamma bound for these inputs.
            @test maximum(abs, field - expected) <= 128eps() * maximum(abs, expected)
        end
        snapshot_index = findfirst(
            event -> event[1] == SolpropJulia.SESSION_SNAPSHOT, events,
        )
        @test snapshot_index !== nothing
        @test events[snapshot_index][2] == 0.2
        @test count(
            event -> event[1] == SolpropJulia.SESSION_PLANE, events[1:snapshot_index],
        ) == 4
        @test last(events)[1] == SolpropJulia.SESSION_COMPLETED
        @test events[2][2] < events[2][3]
        @test session.observation.work.ready
        @test SolpropJulia._march_observed_field(session) === session.field
        @test_throws ArgumentError SolpropJulia.request_snapshot!(session)
    end
end

@testset "RK dense delivery partitions do not change numerical traces" begin
    for tasks in (1, 2)
        full = rk_dense_session(; tasks, coefficient = -0.8 + 1.2im, rtol = 1.0e-7)
        bounded = rk_dense_session(;
            tasks, coefficient = -0.8 + 1.2im, rtol = 1.0e-7, capacity = 1,
        )
        end_only = rk_dense_session(;
            tasks, coefficient = -0.8 + 1.2im, rtol = 1.0e-7, planes = [0.4],
        )
        _, full_fields, full_trace = rk_dense_collect(full)
        _, bounded_fields, bounded_trace = rk_dense_collect(
            bounded; work_limit = 1, deadline = 0.0,
        )
        _, _, end_trace = rk_dense_collect(end_only)
        @test full_trace == bounded_trace == end_trace
        @test full.source.calls == bounded.source.calls == end_only.source.calls
        # These use the identical concrete CPU arithmetic and scalar inputs;
        # scheduling and dense contraction never change a trial input or reduction.
        # Keep a positive output-rounding allowance rather than a bit requirement.
        budget = 2eps() * maximum(abs, full.field)
        @test budget > 0
        @test maximum(abs, bounded.field - full.field) <= budget
        @test maximum(abs, end_only.field - full.field) <= budget
        for (left, right) in zip(full_fields, bounded_fields)
            @test maximum(abs, left - right) <= budget
        end
        @test any(!row[4] for row in full_trace)
    end
end

@testset "RK dense hard kinks, reached tolerance and exact masks" begin
    session = rk_dense_session(; hard_kinks = [0.175], active = Bool[true, false])
    _, fields, trace = rk_dense_collect(session)
    @test any(row -> row[1] == 0.175 && row[4], trace)
    @test all(field -> all(iszero, field[2, :]), fields)
    @test all(iszero, session.field[2, :])
    # RK's reference crossing predicate uses its reached slack, unlike Adams's
    # strict comparison. A requested plane one ulp behind an endpoint is landed.
    near = rk_dense_session(; planes = [prevfloat(0.2), 0.4])
    SolpropJulia.acknowledge!(near, SolpropJulia.advance!(near).generation)
    report = SolpropJulia.advance!(near)
    @test report.kind == SolpropJulia.SESSION_PLANE
    @test report.z == near.state.z == 0.2
    @test !near.observation.selected
    @test SolpropJulia._march_observed_field(near) === near.field
    @test !near.observation.work.ready
    SolpropJulia.acknowledge!(near, report.generation)
    @test_throws ArgumentError rk_dense_session(; replay = true)
    @test_throws ArgumentError rk_dense_session(; weights = [NaN])
    landed = rk_dense_session(; dense = false)
    _, _, landed_trace = rk_dense_collect(landed)
    @test landed.observation === nothing
    @test all(
        any(row -> row[1] == plane && row[4], landed_trace)
            for plane in landed.schedule.planes
    )
end

mutable struct RKDenseBarriers
    barrier::Symbol
    remainder::Symbol
    accepted::Int
    observed::Int
    warned::Bool
    stop::Bool
    endpoints::Int
end

RKDenseBarriers(barrier, remainder, accepted, observed, warned, stop) =
    RKDenseBarriers(barrier, remainder, accepted, observed, warned, stop, 0)

SolpropJulia._session_launch!(::RKDenseBarriers, session) = nothing
SolpropJulia._session_finish!(::RKDenseBarriers, session) = nothing
SolpropJulia._session_quiet(::RKDenseBarriers, session) = false
SolpropJulia._session_quiet_interval(::RKDenseBarriers, session) = nothing
SolpropJulia._session_cap(::RKDenseBarriers, session, proposal::Float64) = proposal
SolpropJulia._session_finish_candidate!(
    ::RKDenseBarriers, session, candidate, dz, destination,
) = nothing

function SolpropJulia._session_accepted!(services::RKDenseBarriers, session, dz)
    services.accepted += 1
    return nothing
end

function SolpropJulia._session_observe!(services::RKDenseBarriers, session, kind, plane)
    if kind === :plane
        services.observed += 1
        if !services.warned
            services.warned = true
            services.barrier = services.stop ? :boundary_stop : :warning
            services.remainder = :plane_rest
        end
    end
    return nothing
end

SolpropJulia._session_barrier(services::RKDenseBarriers, session) = services.barrier

function SolpropJulia._session_endpoint!(services::RKDenseBarriers, session)
    services.endpoints += 1
    return nothing
end

function SolpropJulia._session_continue!(services::RKDenseBarriers, session)
    completed = services.remainder
    services.remainder = :none
    return completed
end

function SolpropJulia._session_acknowledge!(services::RKDenseBarriers, session, generation)
    services.barrier = :none
    return nothing
end

@testset "RK dense warnings and failures export the eligible committed endpoint" begin
    for stop in (false, true)
        services = RKDenseBarriers(:none, :none, 0, 0, false, stop)
        session = rk_dense_session(; services)
        SolpropJulia.acknowledge!(session, SolpropJulia.advance!(session).generation)
        report = SolpropJulia.advance!(session)
        @test report.kind == (
            stop ? SolpropJulia.SESSION_BOUNDARY_STOP :
                SolpropJulia.SESSION_WARNING
        )
        @test report.z == session.state.z == 0.2
        @test SolpropJulia._march_observed_field(session) === session.field
        @test services.accepted == services.observed == 1
        calls = session.source.calls
        if !stop
            SolpropJulia.request_snapshot!(session)
            SolpropJulia.acknowledge!(session, report.generation)
            plane = SolpropJulia.advance!(session)
            @test plane.kind == SolpropJulia.SESSION_PLANE && plane.z == 0.05
            @test session.source.calls == calls
            @test services.observed == 1
            @test SolpropJulia._march_observed_field(session) ===
                session.observation.work.field
            @test_throws ArgumentError SolpropJulia.acknowledge!(session, report.generation)
            SolpropJulia.acknowledge!(session, plane.generation)
            remaining, _, _ = rk_dense_collect(session)
            @test any(event -> event[1] == SolpropJulia.SESSION_SNAPSHOT, remaining)
        else
            @test session.state.finished
            @test !report.poisoned
        end
    end
    failed = rk_dense_session(; fail_on = 10)
    SolpropJulia.acknowledge!(failed, SolpropJulia.advance!(failed).generation)
    while true
        report = SolpropJulia.advance!(failed)
        if report.kind == SolpropJulia.SESSION_FAILED
            @test report.z == 0.2
            @test failed.source.calls == 10
            @test !report.poisoned
            @test SolpropJulia._march_observed_field(failed) === failed.field
            break
        end
        SolpropJulia.acknowledge!(failed, report.generation)
    end
end

function rk_dense_advance_allocations(session)
    report = SolpropJulia.advance!(session; work_limit = 1, deadline = 60.0)
    SolpropJulia.acknowledge!(session, report.generation)
    return @allocated begin
        report = SolpropJulia.advance!(session; work_limit = 1, deadline = 60.0)
        SolpropJulia.acknowledge!(session, report.generation)
    end
end

@testset "RK dense warm advancement remains bounded and allocation-free" begin
    for tasks in (1, 2)
        warm = rk_dense_session(; tasks)
        rk_dense_advance_allocations(warm)
        measured = rk_dense_session(; tasks)
        @test rk_dense_advance_allocations(measured) == 0
        # The next call reads another plane from the same retained step.
        @test rk_dense_advance_allocations(measured) == 0
        @test measured.source.calls == 7
        @test measured.state.z == 0.2
        SolpropJulia.close_session!(measured)
        @test_throws ArgumentError SolpropJulia.advance!(measured)
    end
end

@testset "RK dense retention and evaluation failures preserve accepted state" begin
    # Finite but incompatible synthetic extension weights isolate retention overflow
    # from an otherwise accepted trial; no alternate production method is admitted.
    session = rk_dense_session(; weights = fill(floatmax(Float64), 7))
    initial = copy(session.field)
    SolpropJulia.acknowledge!(session, SolpropJulia.advance!(session).generation)
    failed = SolpropJulia.advance!(session)
    @test failed.kind == SolpropJulia.SESSION_FAILED
    @test failed.z == 0.0 && !failed.poisoned
    @test session.field == initial
    @test session.source.calls == 7
    @test !session.observation.work.ready
    session = rk_dense_session()
    SolpropJulia.acknowledge!(session, SolpropJulia.advance!(session).generation)
    first_plane = SolpropJulia.advance!(session)
    SolpropJulia.acknowledge!(session, first_plane.generation)
    accepted = copy(session.field)
    session.observation.work.coefficients[1, 1, 1] = Inf
    failed = SolpropJulia.advance!(session)
    @test failed.kind == SolpropJulia.SESSION_FAILED
    @test failed.z == 0.2 && !failed.poisoned
    @test session.field == accepted
    @test session.source.calls == 7
    @test SolpropJulia._march_observed_field(session) === session.field
end

@testset "Already-reached RK planes precede a queued snapshot and new trial" begin
    # #679: both requested coordinates lie within the first accepted endpoint's
    # reached slack. Its endpoint services run once, then each plane is delivered
    # under its own index before a snapshot at that same accepted state.
    services = RKDenseBarriers(:none, :none, 0, 0, true, false)
    session = rk_dense_session(; planes = [prevfloat(0.2), 0.2, 0.4], services)
    SolpropJulia.acknowledge!(session, SolpropJulia.advance!(session).generation)
    first_plane = SolpropJulia.advance!(session)
    @test first_plane.kind == SolpropJulia.SESSION_PLANE
    @test first_plane.plane == 0 && first_plane.z == 0.2
    @test services.endpoints == services.accepted == 1
    SolpropJulia.request_snapshot!(session)
    SolpropJulia.acknowledge!(session, first_plane.generation)
    second_plane = SolpropJulia.advance!(session)
    @test second_plane.kind == SolpropJulia.SESSION_PLANE
    @test second_plane.plane == 1 && second_plane.z == 0.2
    @test session.state.z == 0.2 && session.source.calls == 7
    @test services.endpoints == services.accepted == 1
    @test services.observed == 2
    SolpropJulia.acknowledge!(session, second_plane.generation)
    snapshot = SolpropJulia.advance!(session)
    @test snapshot.kind == SolpropJulia.SESSION_SNAPSHOT && snapshot.z == 0.2
    @test session.state.z == 0.2 && session.source.calls == 7
    @test services.endpoints == services.accepted == 1
end
