# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

@testset "Input phase export follows selected observations and terminal replacement" begin
    for terminal in (:failed, :stopped, :boundary_stop)
        original = adams_dense_fixture(; rate = -0.2 + 3.0im)
        execution = original.execution
        noise = SP.InputNoise(
            execution, fill(0.25im, size(original.field)), 0.0, original.flow.operator,
        )
        weight = SP.ObservationWeight(execution, ones(size(original.field)))
        exports = SP._FieldExport(execution, weight; noise)
        session = SP.AdamsSession(
            execution, original.field, 0.0, 0.2, original.policy, original.norm,
            original.schedule, original.source, original.flow; exports,
        )
        adams_session_ack_launch(session)
        plane = SP.advance!(session; deadline = 60.0)
        @test plane.kind == SP.SESSION_PLANE && plane.z < session.state.z
        output = fill(-99.0, size(session.field))
        SP.copy_session_noise_phase!(output, session, plane.generation)
        # The phase is one multiplication of the same binary64 operands.
        bound = 4eps(Float64) * abs(3plane.z)
        @test bound > 0 && maximum(abs.(output .- 3plane.z)) <= bound
        saved = copy(output)
        @test_throws ArgumentError SP.copy_session_noise_phase!(
            output, session, plane.generation + 1,
        )
        @test isequal(output, saved)
        saved_work = copy(noise.phase)
        @test_throws ArgumentError SP.copy_session_noise_phase!(
            noise.phase, session, plane.generation,
        )
        @test isequal(noise.phase, saved_work)
        report = SP.terminate_session!(session, terminal)
        @test report.z == session.state.z
        SP.copy_session_noise_phase!(output, session, report.generation)
        bound = 4eps(Float64) * abs(3report.z)
        @test bound > 0 && maximum(abs.(output .- 3report.z)) <= bound
        @test_throws ArgumentError SP.copy_session_noise_phase!(
            output, session, plane.generation,
        )
        SP.close_session!(session)
        @test_throws ArgumentError SP.copy_session_noise_phase!(
            output, session, report.generation,
        )
    end
end
