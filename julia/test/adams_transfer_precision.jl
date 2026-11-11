# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset
import SolpropJulia as SP

function history_precision_buffers(session, count)
    stored = size(session.history.source)
    T = eltype(session.field)
    return SP.AdamsHistoryBuffers(
        zeros(ComplexF64, size(session.field)), zeros(count), zeros(UInt8, count),
        zeros(Int64, count), zeros(T, stored);
        states = session.history.compensated ? zeros(T, stored) : nothing,
        rates = session.history.compensated ? zeros(ComplexF64, stored) : nothing
    )
end
function history_precision_resume(template, metadata)
    return SP.AdamsSession(
        template.execution, fill(zero(eltype(template.field)), size(template.field)),
        metadata.z, metadata.proposal, template.policy, template.norm, template.schedule,
        template.source, template.flow; arithmetic = template.arithmetic,
        plane_pending = false,
    )
end
function history_precision_to_plane(session)
    for turn in 1:20
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        report.kind === SP.SESSION_PLANE && return report
        report.kind === SP.SESSION_FAILED && error(session.state.failure)
        SP.acknowledge!(session, report.generation)
    end
    error("bounded fixture missed first plane")
end
function history_precision_finish(session)
    for turn in 1:30
        session.state.finished && return
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        @test report.kind !== SP.SESSION_FAILED
        SP.acknowledge!(session, report.generation)
    end
    error("bounded fixture missed end")
end
@testset "Binary32 portable history preserves raw stores and wide rates" begin
    for tasks in (1, 2), compensated in (false, true), exponent in (-80, 3, 80)
        arithmetic = SP.SourceArithmetic(
            Float32; state_exponent = exponent, field_exponent = exponent,
        )
        original, _ = adams_session_precision_fixture(;
            tasks, varying = compensated, order = 4, arithmetic,
        )
        report = history_precision_to_plane(original)
        info = SP.adams_history_info(original, report.generation, report.z)
        @test info !== nothing
        metadata, count, _, flag, bytes = info
        @test flag == compensated
        unused = findfirst(!, original.history.occupied)
        @test unused !== nothing
        original.history.source[1, 1, unused] = ComplexF32(-0.0, 0.0)
        if compensated
            original.history.states[1, 1, unused] = ComplexF32(0.0, -0.0)
            original.history.rates[1, 1, unused] = ComplexF64(nextfloat(1.0), -0.0)
        end
        buffers = history_precision_buffers(original, count)
        SP.export_adams_history!(buffers, original, report.generation, report.z)
        @test isequal(buffers.state, ComplexF64.(original.field))
        @test isequal(buffers.store, original.history.source)
        @test eltype(buffers.store) === ComplexF32
        extra = compensated ? sizeof(buffers.states) + sizeof(buffers.rates) : 0
        @test bytes == sizeof(buffers.state) + sizeof(buffers.store) + extra
        if compensated
            @test isequal(buffers.states, original.history.states)
            @test isequal(buffers.rates, original.history.rates)
            @test real(buffers.rates[1, 1, unused]) == nextfloat(1.0)
        end
        template, _ = adams_session_precision_fixture(;
            tasks, varying = compensated, order = 4, arithmetic,
        )
        resumed = history_precision_resume(template, metadata)
        @test SP.restore_adams_history!(resumed, metadata, buffers) === nothing
        @test isequal(resumed.field, original.field)
        @test isequal(resumed.history.source, original.history.source)
        @test resumed.source.calls == 0
        @test resumed.control.held == count
        @test all(
            !SP._transfer_arrays_alias(arr, resumed.field)
                for arr in (buffers.state, buffers.store)
        )
        if compensated
            @test isequal(resumed.history.states, original.history.states)
            @test isequal(resumed.history.rates, original.history.rates)
        end
        SP.acknowledge!(original, report.generation)
        history_precision_finish(original)
        history_precision_finish(resumed)
        # <=4 remaining short trials, <=1024 rounded scalar operations per path,
        # with bounded linear rate/forcing; 4096u encloses both continued paths.
        budget = 4096 * Float64(eps(Float32)) / 2 * max(1, maximum(abs, original.field))
        @test budget > 0
        @test maximum(abs, original.field - resumed.field) <= budget
        @test original.state.z == resumed.state.z == 1 / 8
        # Restore must reject a finite raw value which is not a widened F32,
        # before changing accepted field/history or evaluating a source.
        badtemplate, _ = adams_session_precision_fixture(;
            tasks, varying = compensated, order = 4, arithmetic,
        )
        refused = history_precision_resume(badtemplate, metadata)
        before_field = copy(refused.field)
        before_store = copy(refused.history.source)
        raw = buffers.state[1, 1]
        buffers.state[1, 1] = complex(nextfloat(real(raw)), imag(raw))
        reason = SP.restore_adams_history!(refused, metadata, buffers)
        @test reason isa String && occursin("losslessly", reason)
        @test isequal(refused.field, before_field)
        @test isequal(refused.history.source, before_store)
        @test refused.control.held == 0 && refused.source.calls == 0
        for session in (original, resumed, refused, template, badtemplate)
            SP.close_session!(session)
        end
    end
end
