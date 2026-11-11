# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

@testset "Dense field export cannot borrow history from the committed future" begin
    session = adams_dense_fixture()
    crossed = false
    while !session.state.finished
        report = SP.advance!(session; deadline = 60.0)
        @test report.kind != SP.SESSION_FAILED
        if report.kind == SP.SESSION_PLANE && report.z < session.state.z
            crossed = true
            committed = copy(session.field)
            output = similar(session.field)
            SP.export_session!(
                output, session, report.generation, ones(size(output)), false, false,
            )
            # Export is a lossless copy of the already-computed selected field.
            @test isequal(output, SP._march_observed_field(session))
            @test isequal(committed, session.field)
            @test SP.adams_history_info(session, report.generation, report.z) === nothing
            @test SP.adams_history_info(
                session, report.generation, session.state.z
            ) === nothing
            buffers = adams_transfer_buffers(session, session.control.held)
            before = copy(buffers.state)
            @test_throws ArgumentError SP.export_adams_history!(
                buffers, session, report.generation, session.state.z,
            )
            @test isequal(buffers.state, before)
        end
        SP.acknowledge!(session, report.generation)
    end
    @test crossed
end
