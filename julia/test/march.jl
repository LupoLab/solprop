# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function march_control(; changes...)
    policy = (
        safety = 0.8, max_factor = 2.0, min_factor = 0.2, beta1 = 0.3, beta2 = -0.1,
        lag = 0, reject_free = false, estimator_order = 1, start_growth = 100.0,
        dz_min = 1.0e-12, dz_max = 0.05, auto_init = false,
    )
    return SolpropJulia.PIControl(; merge(policy, NamedTuple(changes))...)
end

function march_fixture(;
        replay = false, capacity = 256, step_planes = 0, control = march_control(),
        n_steps = 8, z = 0.0, plane_index = nothing, plane_pending = true,
        initial_field = nothing, active = fill(true, 8), prepared_exports = false,
        stop = SolpropJulia.StopFlag(),
    )
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    data = source_data_fixture(
        execution; synthesis = ones(1, 1), projection = ones(1, 1),
        prefactor = fill(-0.5im, 8, 1), omega_window = ones(8),
        source_window = ones(8), density = 1.0,
        kerr = SolpropJulia.KerrCoefficients(4 / 3, :analytic),
    )
    source = SolpropJulia.ModalSource(execution, data; node_chunk = 1)
    initial = zeros(ComplexF64, 8, 1)
    initial[1] = 8.0 + 2.4im
    initial_field === nothing || copyto!(initial, initial_field)
    flow = SolpropJulia.LinearFlow(
        execution, SolpropJulia.DiagonalOperator(execution, fill(-0.2im, 8, 1)), initial,
    )
    tableau = SolpropJulia.EmbeddedTableau(
        [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
    )
    norm = SolpropJulia.MarchNorm(
        execution, active, fill(true, 8), ones(8), 1.0e-4, 0.0,
    )
    schedule = SolpropJulia.MarchSchedule(;
        planes = [0.1, 0.2], breakpoints = [0.0, 0.2], hard_kinks = Float64[],
        stretch = 1.1, step_planes,
        replay_steps = replay ? fill(0.2 / n_steps, n_steps) : Float64[],
        replay_edges = replay ? [0.0; cumsum(fill(0.2 / n_steps, n_steps))] : Float64[],
    )
    exports = prepared_exports ? SolpropJulia._FieldExport(
            execution, SolpropJulia.ObservationWeight(execution, fill(2.0, 8, 1)),
        ) : nothing
    return SolpropJulia.RungeKuttaSession(
        execution, initial, z, 0.05, tableau, control, norm, schedule, source, flow;
        capacity, plane_index, plane_pending, exports, stop,
    )
end

function collect_march(session; work_limit = 256, deadline = 60.0)
    trace = Tuple{Float64, Float64, Float64, Bool}[]
    events = Tuple{SolpropJulia.SessionEvent, Float64}[]
    while !session.state.finished
        report = SolpropJulia.advance!(session; work_limit, deadline)
        push!(events, (report.kind, report.z))
        buffer = session.attempts
        for row in 1:buffer.count
            position, length, norm = buffer.z[row], buffer.dz[row], buffer.norm[row]
            attempt = (position, length, norm, buffer.accepted[row])
            push!(trace, attempt)
        end
        @test buffer.count <= length(buffer.z)
        SolpropJulia.acknowledge!(session, report.generation)
    end
    return trace, events
end

@testset "Julia owns the march independently of delivery boundaries" begin
    for replay in (false, true)
        whole = march_fixture(; replay)
        bounded = march_fixture(; replay, capacity = 2)
        traced, events = collect_march(whole)
        sliced, yielded = collect_march(bounded; work_limit = 1, deadline = 0.0)
        # A transport-only change must preserve the very same discrete computation.
        @test traced == sliced
        @test whole.field == bounded.field
        @test SolpropJulia.control_history(whole.history) ==
            SolpropJulia.control_history(bounded.history)
        @test whole.state.proposal == bounded.state.proposal
        @test whole.state.attempts == length(traced)
        @test whole.state.rejected == count(row -> !last(row), traced)
        @test first(last(events)) == SolpropJulia.SESSION_COMPLETED
        @test last(last(events)) == (replay ? last(whole.schedule.replay_edges) : 0.2)
        @test first(events) == (SolpropJulia.SESSION_LAUNCH, 0.0)
        @test count(row -> first(row) == SolpropJulia.SESSION_PLANE, events) == 2
        @test count(row -> first(row) == SolpropJulia.SESSION_PROGRESS, yielded) > 0
        @test all(isfinite, whole.field)
        @test all(iszero, whole.field[2:end, :])
        if replay
            @test length(traced) == 8
            @test whole.state.rejected == 0
        else
            @test whole.state.rejected > 0
        end
    end
end

@testset "Pending generations and snapshots are exact barriers" begin
    session = march_fixture(; replay = true, capacity = 2)
    @test SolpropJulia.request_snapshot!(session) == (1, false)
    @test SolpropJulia.request_snapshot!(session) == (1, true)
    launch = SolpropJulia.advance!(session)
    @test launch.kind == SolpropJulia.SESSION_LAUNCH
    @test session.source.calls == 0
    @test_throws ArgumentError SolpropJulia.advance!(session)
    @test_throws ArgumentError SolpropJulia.acknowledge!(session, launch.generation + 1)
    SolpropJulia.acknowledge!(session, launch.generation)
    @test_throws ArgumentError SolpropJulia.acknowledge!(session, launch.generation)
    snapshot = SolpropJulia.advance!(session; deadline = 60.0)
    @test snapshot.kind == SolpropJulia.SESSION_SNAPSHOT
    @test snapshot.z == 0.025
    @test session.attempts.count == 1
    retained = copy(session.field)
    norms = SolpropJulia.control_history(session.history)
    SolpropJulia.acknowledge!(session, snapshot.generation)
    @test session.field == retained
    @test SolpropJulia.control_history(session.history) == norms
    @test session.attempts.count == 0
    @test SolpropJulia.request_snapshot!(session) == (2, false)
    again = SolpropJulia.advance!(session)
    @test again.kind == SolpropJulia.SESSION_SNAPSHOT
    @test session.attempts.count == 0
    @test session.field == retained
    SolpropJulia.close_session!(session)
    SolpropJulia.close_session!(session)
    @test_throws ArgumentError SolpropJulia.advance!(session)
    @test_throws ArgumentError SolpropJulia.acknowledge!(session, again.generation)
    @test_throws ArgumentError SolpropJulia.request_snapshot!(session)
end

@testset "Fixed replay retains its steps beyond the adaptive maximum" begin
    session = march_fixture(; replay = true, control = march_control(dz_max = 0.001))
    trace, _ = collect_march(session)
    @test length(trace) == 8
    @test all(row -> row[2] == 0.025, trace)
    @test session.state.proposal == 0.05
end

@testset "Accumulated replay and resumed plane cursors retain rounding slack" begin
    full = march_fixture(; replay = true, n_steps = 160)
    trace, events = collect_march(full)
    @test length(trace) == 160
    @test first(last(events)) == SolpropJulia.SESSION_COMPLETED
    @test full.state.z == last(full.schedule.replay_edges)
    @test all(row -> row[2] == 0.2 / 160, trace)
    resumed = march_fixture(;
        replay = true, n_steps = 160, z = 0.1, plane_index = 1, plane_pending = false,
    )
    trace, events = collect_march(resumed)
    @test length(trace) == 80
    @test all(row -> first(row) != SolpropJulia.SESSION_LAUNCH, events)
    @test count(row -> first(row) == SolpropJulia.SESSION_PLANE, events) == 1
    @test first(last(events)) == SolpropJulia.SESSION_COMPLETED
    @test_throws ArgumentError march_fixture(; z = 0.1, plane_index = 0)
    @test_throws ArgumentError march_fixture(; z = 0.0, plane_index = 1)
end

@testset "Launch masks an owned copy before observation or nonlinear work" begin
    input = fill(3.0 + 4.0im, 8, 1)
    active = fill(true, 8)
    active[2] = false
    session = march_fixture(; initial_field = input, active)
    launch = SolpropJulia.advance!(session)
    @test launch.kind == SolpropJulia.SESSION_LAUNCH
    @test session.field[2, 1] === 0.0 + 0.0im
    @test session.field[1, 1] == input[1, 1]
    @test all(==(3.0 + 4.0im), input)
    @test session.source.calls == 0
end

struct _FailPlaneServices end
SolpropJulia._session_launch!(::_FailPlaneServices, session) = nothing
SolpropJulia._session_accepted!(::_FailPlaneServices, session, dz) = nothing
SolpropJulia._session_finish!(::_FailPlaneServices, session) = nothing
SolpropJulia._session_quiet(::_FailPlaneServices, session) = false
SolpropJulia._session_quiet_interval(::_FailPlaneServices, session) = nothing
SolpropJulia._session_cap(::_FailPlaneServices, session, proposal) = proposal
SolpropJulia._session_barrier(::_FailPlaneServices, session) = :none
SolpropJulia._session_continue!(::_FailPlaneServices, session) = :none
SolpropJulia._session_acknowledge!(::_FailPlaneServices, session, generation) = nothing
SolpropJulia._session_finish_candidate!(
    ::_FailPlaneServices, session, candidate, dz, destination,
) = nothing
function SolpropJulia._session_observe!(::_FailPlaneServices, session, kind, plane)
    kind === :plane && throw(ArgumentError("injected observation failure"))
    return nothing
end

struct _TimedSource{S}
    source::S
end
function (source::_TimedSource)(out, field, z)
    # Make numerical time dominate dispatch overhead, so the outer elapsed interval
    # is an independent upper bound even on a fast host. No wall-time lower threshold.
    until = time_ns() + 2_000_000
    while time_ns() < until
    end
    return source.source(out, field, z)
end

function failed_observation()
    fixture = march_fixture(; replay = true)
    session = SolpropJulia.RungeKuttaSession(
        fixture.execution, fixture.field, 0.0, 0.05, fixture.tableau, fixture.control,
        fixture.norm, fixture.schedule, _TimedSource(fixture.source), fixture.flow;
        services = _FailPlaneServices(),
    )
    launch = SolpropJulia.advance!(session)
    SolpropJulia.acknowledge!(session, launch.generation)
    began = time_ns()
    report = SolpropJulia.advance!(session; deadline = 60.0)
    elapsed = (time_ns() - began) * 1.0e-9
    return session, report, elapsed
end

@testset "Observation failures count completed numerical time once" begin
    failed_observation() # Compile the deliberately failing hook before the witness.
    session, report, elapsed = failed_observation()
    @test report.kind == SolpropJulia.SESSION_FAILED
    @test report.attempts == 4
    @test report.z == 0.1
    @test !report.poisoned
    @test 0.0 < report.wall_time <= elapsed
    @test session.state.failure_message == "ArgumentError: injected observation failure"
end

@testset "Attempt times start at each delivery chunk" begin
    session = march_fixture(; replay = true)
    launch = SolpropJulia.advance!(session)
    SolpropJulia.acknowledge!(session, launch.generation)
    first_chunk = SolpropJulia.advance!(session; work_limit = 1, deadline = 60.0)
    SolpropJulia.acknowledge!(session, first_chunk.generation)
    # A large earlier summary isolates the chunk clock from machine-dependent timing.
    session.state.plane_work = 1.0e20
    second_chunk = SolpropJulia.advance!(session; work_limit = 1, deadline = 60.0)
    @test session.attempts.count == 1
    @test 0.0 <= session.attempts.elapsed[1] < 1.0e20
    @test second_chunk.wall_time >= 1.0e20
end

@testset "Stop and numerical failures retain committed-state facts" begin
    session = march_fixture()
    launch = SolpropJulia.advance!(session)
    SolpropJulia.acknowledge!(session, launch.generation)
    unsafe_store!(session.stop.address, UInt8(1), :release)
    stopped = SolpropJulia.advance!(session)
    @test stopped.kind == SolpropJulia.SESSION_STOPPED
    @test stopped.z == 0.0
    @test stopped.attempts == 0
    @test !stopped.poisoned
    @test session.source.calls == 0
    SolpropJulia.acknowledge!(session, stopped.generation)
    @test_throws ArgumentError SolpropJulia.advance!(session)

    failed = march_fixture(control = march_control(dz_min = 0.04))
    launch = SolpropJulia.advance!(failed)
    SolpropJulia.acknowledge!(failed, launch.generation)
    field = copy(failed.field)
    report = SolpropJulia.advance!(failed; deadline = 60.0)
    @test report.kind == SolpropJulia.SESSION_FAILED
    @test failed.state.failure_code == :StepFloorError
    @test !report.poisoned
    @test report.z == 0.0
    @test report.rejected == 1
    @test failed.field == field
    @test failed.state.proposal == 0.05
    @test failed.history.count == 0
    @test failed.attempts.count == 1
end

function march_allocations(session)
    return @allocated begin
        report = SolpropJulia.advance!(session; work_limit = 1, deadline = 60.0)
        SolpropJulia.acknowledge!(session, report.generation)
        nothing
    end
end

@testset "Warmed Julia advancement uses bounded preallocated storage" begin
    session = march_fixture(; replay = true, capacity = 2)
    launch = SolpropJulia.advance!(session)
    SolpropJulia.acknowledge!(session, launch.generation)
    march_allocations(session)
    @test march_allocations(session) == 0
    @test march_allocations(session) == 0
    @test length(session.attempts.z) == 2
    @test isbitstype(SolpropJulia.SessionReport)
end
