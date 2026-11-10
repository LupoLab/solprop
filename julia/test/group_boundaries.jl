# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

@testset "Fixed replay boundaries yield without changing numerical history" begin
    session = march_fixture(; replay = true)
    initial = copy(session.field)
    launch = SolpropJulia.advance!(session; replay_limit = 1)
    @test launch.kind == SolpropJulia.SESSION_LAUNCH
    SolpropJulia.acknowledge!(session, launch.generation)
    idle = SolpropJulia.advance!(session; replay_limit = 1)
    @test idle.kind == SolpropJulia.SESSION_PROGRESS
    @test session.state.attempts == session.source.calls == 0
    @test session.field == initial
    SolpropJulia.acknowledge!(session, idle.generation)
    progress = SolpropJulia.advance!(
        session; replay_limit = 3, work_limit = 256, deadline = 60.0,
    )
    @test progress.kind == SolpropJulia.SESSION_PROGRESS
    @test session.state.replay == 3
    @test session.state.attempts == session.attempts.count == 2
    @test session.source.calls == 4
    SolpropJulia.acknowledge!(session, progress.generation)
    before = (copy(session.field), session.state.generation, session.source.calls)
    for limit in (0, 2, 10)
        @test_throws ArgumentError SolpropJulia.advance!(session; replay_limit = limit)
        @test session.field == before[1]
        @test (session.state.generation, session.source.calls) == before[2:3]
    end
    adaptive = march_fixture()
    @test_throws ArgumentError SolpropJulia.advance!(adaptive; replay_limit = 1)
    @test !adaptive.state.launched && adaptive.source.calls == 0
end

@testset "Group termination replaces a receipt without draining its chunk" begin
    for kind in (:failed, :stopped, :boundary_stop)
        session = march_fixture(; replay = true)
        launch = SolpropJulia.advance!(session)
        SolpropJulia.acknowledge!(session, launch.generation)
        pending = SolpropJulia.advance!(session; work_limit = 2, deadline = 60.0)
        before = (copy(session.field), copy(session.attempts.z), session.source.calls)
        report = SolpropJulia.terminate_session!(
            session, kind; failure_code = :GroupFailure, failure_message = "peer failed",
        )
        @test report.kind == SolpropJulia._session_event(kind)
        @test report.generation == pending.generation + 1
        @test session.state.finished && session.state.pending
        @test session.field == before[1]
        @test session.attempts.z == before[2]
        @test session.source.calls == before[3]
        @test session.attempts.count == 2
        if kind == :failed
            @test session.state.failure_code == :GroupFailure
            @test session.state.failure_message == "peer failed"
        end
        @test_throws ArgumentError SolpropJulia.acknowledge!(session, pending.generation)
        @test_throws ArgumentError SolpropJulia.terminate_session!(session, kind)
        SolpropJulia.acknowledge!(session, report.generation)
        @test session.attempts.count == 0
    end
    session = march_fixture(; replay = true)
    @test_throws ArgumentError SolpropJulia.terminate_session!(session, :progress)
    @test session.state.generation == 0 && !session.state.pending
    session.state.generation = typemax(Int)
    @test_throws ArgumentError SolpropJulia.terminate_session!(session, :failed)
    @test !session.state.finished && !session.state.pending
end
