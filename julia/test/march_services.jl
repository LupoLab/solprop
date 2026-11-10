# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# These scripted numerical hooks isolate transport ordering from monitor algebra.
# Real temporal readings and continuation are qualified in observation_session.jl.
Base.@kwdef mutable struct _ScriptedMarchServices
    warn_at::Symbol = :none
    warning_step::Int = 4
    stopping::Bool = false
    barrier::Symbol = :none
    continuation::Symbol = :none
    fired::Bool = false
    launch_calls::Int = 0
    accepted_calls::Int = 0
    observe_calls::Int = 0
    finish_calls::Int = 0
    continued::Int = 0
    acknowledged::Int = 0
    row_count::Int = 0
    row_steps::Vector{Int} = zeros(Int, 9)
    plane_valid::Bool = false
    double_finish_warning::Bool = false
    finished_remainders::Int = 0
    quiet::Bool = false
    interval::Float64 = 0.01
    cap::Float64 = Inf
    candidate_masked::Bool = false
    candidate_endpoint::Float64 = NaN
    fail_candidate::Bool = false
    nonfinite_candidate::Bool = false
end

function _scripted_warning!(service, stage, continuation)
    due = stage !== :accepted || service.accepted_calls == service.warning_step
    if !service.fired && service.warn_at === stage && due
        service.fired = true
        service.barrier = service.stopping ? :boundary_stop : :warning
        service.continuation = continuation
        return true
    end
    return false
end

function _scripted_row!(service, session)
    service.row_count += 1
    service.row_steps[service.row_count] = session.attempts.count - 1
    return nothing
end

function SolpropJulia._session_launch!(service::_ScriptedMarchServices, session)
    service.launch_calls += 1
    _scripted_warning!(service, :launch, :launch_rest)
    return nothing
end

function SolpropJulia._session_accepted!(service::_ScriptedMarchServices, session, dz)
    service.accepted_calls += 1
    _scripted_warning!(service, :accepted, :accepted_rest) && return nothing
    return _scripted_row!(service, session)
end

function SolpropJulia._session_observe!(
        service::_ScriptedMarchServices, session, kind, plane,
    )
    service.observe_calls += 1
    kind === :plane && _scripted_warning!(service, :plane, :plane_rest) && return nothing
    service.plane_valid = true
    return nothing
end

function SolpropJulia._session_finish!(service::_ScriptedMarchServices, session)
    service.finish_calls += 1
    _scripted_warning!(service, :finish, :finish_rest)
    return nothing
end

SolpropJulia._session_barrier(service::_ScriptedMarchServices, session) = service.barrier
SolpropJulia._session_quiet(service::_ScriptedMarchServices, session) = service.quiet
SolpropJulia._session_quiet_interval(service::_ScriptedMarchServices, session) =
    service.interval
SolpropJulia._session_cap(service::_ScriptedMarchServices, session, proposal) =
    min(proposal, service.cap)

function SolpropJulia._session_continue!(service::_ScriptedMarchServices, session)
    continuation = service.continuation
    continuation === :none && return :none
    service.continued += 1
    service.continuation = :none
    if continuation === :accepted_rest
        _scripted_row!(service, session)
    elseif continuation === :plane_rest
        service.plane_valid = true
    elseif continuation === :finish_rest
        service.finished_remainders += 1
        if service.double_finish_warning && service.finished_remainders == 1
            service.continuation = :finish_rest
            service.barrier = :warning
        end
    end
    return continuation
end

function SolpropJulia._session_acknowledge!(
        service::_ScriptedMarchServices, session, generation,
    )
    service.acknowledged += 1
    service.barrier = :none
    service.row_count = 0
    service.plane_valid = false
    return nothing
end

function SolpropJulia._session_finish_candidate!(
        service::_ScriptedMarchServices, session, candidate, dz, destination,
    )
    service.candidate_masked = all(iszero, candidate[.!session.norm.active, :])
    service.candidate_endpoint = destination
    service.fail_candidate && throw(ArgumentError("injected split candidate failure"))
    service.nonfinite_candidate && (candidate[1, 1] = ComplexF64(Inf))
    # A numerical finishing operator can repopulate the inactive band; the final
    # commit still has to enforce its structural zeros.
    for column in axes(candidate, 2), row in axes(candidate, 1)
        session.norm.active[row] || (candidate[row, column] = 17.0 + 3.0im)
    end
    return nothing
end

function scripted_service_session(
        service; replay = true, step_planes = 0, active = fill(true, 8),
    )
    fixture = march_fixture(; replay, step_planes, active)
    return SolpropJulia.RungeKuttaSession(
        fixture.execution, fixture.field, 0.0, 0.05, fixture.tableau, fixture.control,
        fixture.norm, fixture.schedule, fixture.source, fixture.flow; services = service,
    )
end

function acknowledge_launch(session)
    launch = SolpropJulia.advance!(session)
    @test launch.kind == SolpropJulia.SESSION_LAUNCH
    @test session.services.launch_calls == 0
    SolpropJulia.acknowledge!(session, launch.generation)
    return nothing
end

@testset "Accepted warnings resume rows and reached planes before snapshots" begin
    service = _ScriptedMarchServices(; warn_at = :accepted)
    session = scripted_service_session(service)
    acknowledge_launch(session)
    warning = SolpropJulia.advance!(session; deadline = 60.0)
    @test warning.kind == SolpropJulia.SESSION_WARNING
    @test warning.z == 0.1
    @test warning.attempts == session.attempts.count == 4
    @test service.row_count == 3
    @test service.row_steps[1:3] == [0, 1, 2]
    @test session.state.endpoint_pending
    @test session.state.plane == 1
    @test service.observe_calls == 1
    @test_throws ArgumentError SolpropJulia.acknowledge!(session, warning.generation + 1)
    @test service.row_count == 3
    @test service.acknowledged == 1
    @test SolpropJulia.request_snapshot!(session) == (1, false)
    calls, field = session.source.calls, copy(session.field)
    SolpropJulia.acknowledge!(session, warning.generation)
    plane = SolpropJulia.advance!(session; deadline = 60.0)
    @test plane.kind == SolpropJulia.SESSION_PLANE
    @test plane.plane == 0
    @test plane.plane_attempts == 4
    @test session.attempts.count == 0
    @test service.row_count == 1
    @test service.row_steps[1] == -1
    @test service.continued == 1
    @test service.accepted_calls == 4
    @test service.plane_valid
    @test session.state.plane == 2
    @test session.field == field
    @test session.source.calls == calls
    SolpropJulia.acknowledge!(session, plane.generation)
    snapshot = SolpropJulia.advance!(session; deadline = 60.0)
    @test snapshot.kind == SolpropJulia.SESSION_SNAPSHOT
    @test snapshot.z == 0.1
    @test snapshot.plane_attempts == 0
    @test service.row_count == session.attempts.count == 0
    @test session.source.calls == calls
end

@testset "A warning inside plane preparation preserves its cursor and summary" begin
    service = _ScriptedMarchServices(; warn_at = :plane)
    session = scripted_service_session(service)
    acknowledge_launch(session)
    warning = SolpropJulia.advance!(session; deadline = 60.0)
    @test warning.kind == SolpropJulia.SESSION_WARNING
    @test session.state.observation_kind === :plane
    @test session.state.plane == 1
    @test !service.plane_valid
    @test service.row_count == 4
    calls = session.source.calls
    SolpropJulia.acknowledge!(session, warning.generation)
    plane = SolpropJulia.advance!(session; deadline = 60.0)
    @test plane.kind == SolpropJulia.SESSION_PLANE
    @test plane.plane_attempts == 4
    @test service.observe_calls == 2
    @test service.continued == 1
    @test service.plane_valid
    @test session.state.observation_kind === :none
    @test session.state.plane == 2
    @test session.source.calls == calls
    @test session.attempts.count == 0
end

@testset "A resumed warning services an owed step observation" begin
    service = _ScriptedMarchServices(; warn_at = :accepted, warning_step = 1)
    session = scripted_service_session(service; step_planes = 1)
    acknowledge_launch(session)
    warning = SolpropJulia.advance!(session; deadline = 60.0)
    @test warning.kind == SolpropJulia.SESSION_WARNING
    @test SolpropJulia.request_snapshot!(session) == (1, false)
    SolpropJulia.acknowledge!(session, warning.generation)
    step = SolpropJulia.advance!(session; deadline = 60.0)
    @test step.kind == SolpropJulia.SESSION_STEP
    @test step.z == 0.025
    @test session.attempts.count == 0
    @test service.row_steps[1] == -1
    @test session.state.snapshot
end

@testset "Launch notices preserve ordering and boundary stops retain the field" begin
    for stopping in (false, true)
        service = _ScriptedMarchServices(; warn_at = :launch, stopping)
        session = scripted_service_session(service)
        acknowledge_launch(session)
        field = copy(session.field)
        notice = SolpropJulia.advance!(session; deadline = 60.0)
        @test notice.kind == (
            stopping ? SolpropJulia.SESSION_BOUNDARY_STOP :
                SolpropJulia.SESSION_WARNING
        )
        @test notice.z == 0.0
        @test !notice.poisoned
        @test session.source.calls == session.attempts.count == 0
        @test session.field == field
        @test service.launch_calls == 1
        SolpropJulia.acknowledge!(session, notice.generation)
        if stopping
            @test session.state.finished
            @test_throws ArgumentError SolpropJulia.advance!(session)
        else
            next = SolpropJulia.advance!(session; work_limit = 1, deadline = 60.0)
            @test next.kind == SolpropJulia.SESSION_PROGRESS
            @test next.attempts == 1
            @test service.launch_calls == service.continued == 1
        end
    end
end

@testset "Final check and removal warnings never repeat completed finishing" begin
    service = _ScriptedMarchServices(; warn_at = :finish, double_finish_warning = true)
    session = scripted_service_session(service)
    acknowledge_launch(session)
    for expected_plane in (0, 1)
        plane = SolpropJulia.advance!(session; deadline = 60.0)
        @test plane.kind == SolpropJulia.SESSION_PLANE
        @test plane.plane == expected_plane
        SolpropJulia.acknowledge!(session, plane.generation)
    end
    calls = session.source.calls
    for _ in 1:2
        warning = SolpropJulia.advance!(session; deadline = 60.0)
        @test warning.kind == SolpropJulia.SESSION_WARNING
        @test !session.state.finished
        @test session.attempts.count == 0
        @test service.finish_calls == 1
        SolpropJulia.acknowledge!(session, warning.generation)
    end
    completed = SolpropJulia.advance!(session; deadline = 60.0)
    @test completed.kind == SolpropJulia.SESSION_COMPLETED
    @test service.finish_calls == 1
    @test service.finished_remainders == 2
    @test session.source.calls == calls
end

@testset "Candidate finishing sees the mask and its exact chosen endpoint" begin
    active = fill(true, 8)
    active[2] = false
    for failure in (:none, :throw, :nonfinite)
        service = _ScriptedMarchServices(;
            fail_candidate = failure === :throw,
            nonfinite_candidate = failure === :nonfinite,
        )
        session = scripted_service_session(service; active)
        acknowledge_launch(session)
        before = copy(session.field)
        report = SolpropJulia.advance!(session; work_limit = 1, deadline = 60.0)
        @test service.candidate_masked
        @test service.candidate_endpoint == session.schedule.replay_edges[2]
        @test !report.poisoned
        if failure !== :none
            @test report.kind == SolpropJulia.SESSION_FAILED
            @test session.field == before
            @test report.z == 0.0
            @test service.accepted_calls == 0
        else
            @test report.kind == SolpropJulia.SESSION_PROGRESS
            @test report.z == service.candidate_endpoint
            @test session.field[2, 1] === 0.0 + 0.0im
            @test service.accepted_calls == 1
        end
    end
end

@testset "Quiet and crossing caps affect proposals but never shorten replay" begin
    service = _ScriptedMarchServices(; quiet = true, cap = 0.007)
    adaptive = scripted_service_session(service; replay = false)
    @test SolpropJulia._march_landing(adaptive) == (0.007, 0.007)
    service.cap = Inf
    @test SolpropJulia._march_landing(adaptive) == (0.01, 0.01)
    replay = scripted_service_session(service)
    @test SolpropJulia._march_landing(replay) == (0.025, 0.025)
end
