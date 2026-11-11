# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SJ

function precision_export_allocated(output, session, generation, weights)
    SJ.export_session!(output, session, generation, weights, true, true)
    return @allocated SJ.export_session!(output, session, generation, weights, true, true)
end

@testset "Physical power conversion refuses atomically after finite restoration" begin
    for T in (Float32, Float64), low in (false, true), optics in (false, true)
        arithmetic = SJ.SourceArithmetic(T; state_exponent = T === Float32 ? 1023 : 0)
        raw = T === Float32 ? ComplexF32(1.5, 0.5) :
            ComplexF64(ldexp(1.5, 1023), ldexp(0.5, 1023))
        session, _ = session_precision_fixture(;
            low, arithmetic, initial = fill(raw, 4, 2),
        )
        report = SJ.advance!(session; work_limit = 1, deadline = 0.0)
        initial = copy(session.field)
        output = fill(7.0 + 8.0im, 4, 2)
        before = copy(output)
        @test all(isfinite, SJ._physical_state_value.(initial, Ref(arithmetic)))
        @test_throws "physical field export exceeds binary64 range" SJ.export_session!(
            output, session, report.generation, fill(4.0, 4, 2), true, optics,
        )
        @test isequal(output, before)
        @test isequal(session.field, initial)
        @test session.state.pending && session.state.generation == report.generation
        SJ.export_session!(
            output, session, report.generation, ones(4, 2), true, optics,
        )
        @test all(isfinite, output)
        @test session.state.pending
        SJ.acknowledge!(session, report.generation)
        SJ.close_session!(session)
    end
end

function physical_probe_cycle!(probe, output)
    SJ.trial_probe!(probe)
    SJ.copy_trial_probe_field!(output, probe; physical = true)
    return nothing
end

function physical_probe_allocated(probe, output)
    physical_probe_cycle!(probe, output)
    return @allocated physical_probe_cycle!(probe, output)
end

@testset "Physical probe readback keeps the prepared state and consumes one trial" begin
    for tasks in (1, 2)
        probe, _ = trial_probe_precision_fixture(; tasks)
        loaded = copy(probe.field)
        SJ.trial_probe!(probe)
        raw = copy(probe.work.candidate)
        output = zeros(ComplexF64, size(raw))
        @test_throws ArgumentError SJ.copy_trial_probe_field!(
            similar(raw), probe; physical = true,
        )
        @test probe.state.ready
        SJ.copy_trial_probe_field!(output, probe; physical = true)
        @test isequal(output, ComplexF64.(raw) .* 8.0)
        @test isequal(probe.field, loaded)
        @test !probe.state.ready && !probe.state.failed
        @test physical_probe_allocated(probe, output) == 0
        @test_throws ArgumentError SJ.copy_trial_probe_field!(
            output, probe; physical = true,
        )
    end
end

@testset "Physical destination export widens before scaling and format conversion" begin
    for T in (Float32, Float64), low in (false, true)
        arithmetic = SJ.SourceArithmetic(T; state_exponent = T === Float32 ? 80 : 0)
        session, _ = session_precision_fixture(; low, arithmetic)
        report = SJ.advance!(session; work_limit = 1, deadline = 0.0)
        raw = copy(session.field)
        output = zeros(ComplexF64, size(raw))
        weights = [1.0 + row / 16 + column / 32 for row in 1:4, column in 1:2]
        scale = SJ.source_state_scale(arithmetic)
        for power in (false, true), optics in (false, true)
            SJ.export_session!(output, session, report.generation, weights, power, optics)
            physical = ComplexF64.(raw) .* scale
            expected = power ? physical .* sqrt.(weights) : physical
            if optics
                expected = conj.(expected) .* reshape([1.0, -1.0, 1.0, -1.0], 4, 1)
            end
            if !power && !optics
                @test isequal(output, physical)
            else
                # Exact widening/scaling precedes at most two real products per
                # format operation and a binary64 square root. Budget each path.
                allowance = 16eps(Float64) * maximum(abs, physical)
                @test allowance > 0
                @test maximum(abs, output .- expected) <= allowance
            end
            @test isequal(session.field, raw)
        end
        @test precision_export_allocated(output, session, report.generation, weights) == 0
        SJ.acknowledge!(session, report.generation)
        @test_throws ArgumentError SJ.export_session!(
            output, session, report.generation, weights, false, false,
        )
    end
    arithmetic = SJ.SourceArithmetic(Float32; state_exponent = 1023)
    session, _ = session_precision_fixture(
        ; arithmetic, initial = fill(4.0f0 + 0.0f0im, 4, 2),
    )
    report = SJ.advance!(session; work_limit = 1, deadline = 0.0)
    output = fill(7.0 + 8.0im, 4, 2)
    before = copy(output)
    @test_throws ArgumentError SJ.export_session!(
        output, session, report.generation, ones(4, 2), false, false,
    )
    @test isequal(output, before)
    @test session.state.pending
end
