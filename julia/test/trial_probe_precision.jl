# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

function trial_probe_precision_fixture(; low = false, tasks = 1, fail_at = 0)
    session, initial = session_precision_fixture(; low, tasks, fail_at)
    probe = SP.TrialProbe(
        session.execution, initial, 0.0, 1 / 64, session.tableau, session.norm,
        session.schedule, session.source, session.flow; session.arithmetic,
    )
    return probe, initial
end

function trial_probe_precision_cycle!(probe, output)
    SP.trial_probe!(probe)
    SP.copy_trial_probe_field!(output, probe)
    return nothing
end

function trial_probe_precision_allocations(probe, output)
    trial_probe_precision_cycle!(probe, output)
    return @allocated trial_probe_precision_cycle!(probe, output)
end

function trial_probe_precision_low_allocations(probe)
    SP.trial_probe!(probe)
    return @allocated SP.trial_probe!(probe)
end

@testset "Binary32 trial probes retain a load and consume one narrow result" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        probe, initial = trial_probe_precision_fixture(; tasks)
        original, loaded = copy(initial), copy(probe.field)
        output = similar(initial)
        @test eltype(probe.work.stages) === ComplexF32
        @test eltype(probe.work.norm_parts) === Float32
        @test probe.arithmetic.state_exponent == 3
        @test !Base.mightalias(initial, probe.field)
        @test_throws ArgumentError SP.copy_trial_probe_field!(output, probe)
        for (z, h) in ((0.0, 1 / 64), (1 / 4, 1 / 32), (1 / 8, 1 / 128))
            @test SP.trial_probe!(probe, z, h) === nothing
            @test probe.state.ready && isfinite(probe.state.error_norm)
            @test SP.copy_trial_probe_field!(output, probe) === nothing
            gain, linear, step = Complex{BigFloat}(-0.25, 0.125),
                Complex{BigFloat}(-0.125, 0.25), BigFloat(h)
            factor = exp(linear * step) * (1 + gain * step + (gain * step)^2 / 2)
            expected = Complex{BigFloat}.(loaded) .* factor
            budget = session_precision_bound(loaded, h, 1)
            @test budget > 0
            @test maximum(abs, Complex{BigFloat}.(output) - expected) <= budget
            @test all(iszero, output[2, :])
            @test probe.field == loaded && initial == original
            @test probe.state.loaded_z == 0.0 && probe.state.default_dz == 1 / 64
            @test_throws ArgumentError SP.copy_trial_probe_field!(output, probe)
        end
        @test probe.state.source_evaluations == probe.source.calls == 6
        @test probe.state.norm_evaluations == probe.state.trials == 3
        @test trial_probe_precision_allocations(probe, output) == 0
        @test trial_probe_precision_allocations(probe, output) == 0
        @test sum(SP.trial_probe_memory(probe)) > sizeof(probe.field)
        SP.close_trial_probe!(probe)
        @test SP.trial_probe_status(probe).closed
        @test_throws ArgumentError SP.trial_probe!(probe)
    end
end

@testset "Invalid binary32 probe range and readback preserve an unread trial" begin
    probe, initial = trial_probe_precision_fixture()
    SP.trial_probe!(probe)
    candidate = copy(probe.work.candidate)
    before = SP.trial_probe_status(probe)
    for step in (1.0e-100, 1.0e100, 0.0, Inf)
        @test_throws ArgumentError SP.trial_probe!(probe, 0.0, step)
        @test SP.trial_probe_status(probe) == before
        @test probe.work.candidate == candidate
    end
    wide = fill(3.0 + 4.0im, size(initial))
    @test_throws ArgumentError SP.copy_trial_probe_field!(wide, probe)
    @test all(==(3.0 + 4.0im), wide)
    @test SP.trial_probe_status(probe) == before
    @test_throws ArgumentError SP.copy_trial_probe_field!(probe.field, probe)
    @test_throws ArgumentError SP.copy_trial_probe_field!(probe.work.candidate, probe)
    @test SP.trial_probe_status(probe) == before && probe.work.candidate == candidate
    output = similar(initial)
    SP.copy_trial_probe_field!(output, probe)
    @test !probe.state.ready && all(isfinite, output)
end

@testset "Binary32 raw low-storage probes retain exact attempt counts" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        probe, initial = trial_probe_precision_fixture(; low = true, tasks)
        loaded = copy(probe.field)
        for repeat in 1:8
            @test SP.trial_probe!(probe) === nothing
            @test !probe.state.ready && !probe.state.in_flight
            @test probe.state.source_evaluations == 4repeat
            @test probe.state.norm_evaluations == probe.state.trials == repeat
            gain, linear, step = Complex{BigFloat}(-0.25, 0.125),
                Complex{BigFloat}(-0.125, 0.25), BigFloat(1 / 64)
            factor = exp(linear * step) * sum((gain * step)^k / factorial(k) for k in 0:4)
            expected = Complex{BigFloat}.(loaded) .* factor^repeat
            budget = session_precision_bound(loaded, repeat / 64, repeat)
            @test budget > 0
            @test maximum(abs, Complex{BigFloat}.(probe.field) - expected) <= budget
            @test_throws ArgumentError SP.copy_trial_probe_field!(similar(initial), probe)
        end
        @test trial_probe_precision_low_allocations(probe) == 0
        @test probe.state.loaded_z == 0.0 && probe.state.trial_z == 0.0
        @test probe.state.default_dz == probe.state.trial_dz == 1 / 64
        before = copy(probe.field)
        facts = SP.trial_probe_status(probe)
        @test_throws ArgumentError SP.trial_probe!(probe, 0.0, 1.0e-100)
        @test probe.field == before && SP.trial_probe_status(probe) == facts
    end
    for low in (false, true), failure in (1, 2)
        probe, _ = trial_probe_precision_fixture(; low, fail_at = failure)
        before = copy(probe.field)
        @test_throws ErrorException SP.trial_probe!(probe)
        @test probe.state.failed && !probe.state.ready
        @test probe.state.in_flight == low
        @test probe.state.source_evaluations == probe.source.calls == failure
        @test probe.state.norm_evaluations == probe.state.trials == 0
        low || @test probe.field == before
        @test_throws ArgumentError SP.trial_probe!(probe)
        SP.close_trial_probe!(probe)
        @test probe.state.closed
    end
end

@testset "Probe precision admission refuses noise and mismatched owned split" begin
    arithmetic = SP.SourceArithmetic(Float32; state_exponent = 3, field_exponent = 1)
    session, initial = session_precision_material_session(arithmetic)
    ex, source, flow = session.execution, session.source, session.flow
    noise = SP.InputNoise(ex, ones(ComplexF64, size(initial)), 0.0, flow.operator)
    split = SP.TemporalSplitWorkspace(ex, :envelope, ones(4), 1)
    changed = SP.SourceArithmetic(Float32; state_exponent = 4, field_exponent = 1)
    wrong_flow = SP.LinearFlow(ex, flow.operator, ComplexF64.(initial))
    for options in (
            (; noise), (; split), (; arithmetic = changed),
            (; arithmetic = SP.SourceArithmetic(Float64)),
        )
        policy = get(options, :arithmetic, arithmetic)
        @test_throws ArgumentError SP.TrialProbe(
            ex, initial, 0.0, 1 / 64, session.tableau, session.norm, session.schedule,
            source, flow; arithmetic = policy,
            noise = get(options, :noise, nothing), split = get(options, :split, nothing),
        )
        @test source.calls == 0 && !source.ready
    end
    @test_throws ArgumentError SP.TrialProbe(
        ex, initial, 0.0, 1 / 64, session.tableau, session.norm, session.schedule,
        source, wrong_flow; arithmetic,
    )
    probe = SP.TrialProbe(
        ex, initial, 0.0, 1 / 64, session.tableau, session.norm, session.schedule,
        source, flow; arithmetic,
    )
    output = similar(initial)
    @test trial_probe_precision_allocations(probe, output) == 0
    @test all(isfinite, output) && source.calls == 4
end
