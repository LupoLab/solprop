# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function test_control(; changes...)
    fields = (
        safety = 0.8, max_factor = 1.1, min_factor = 0.2, beta1 = 3 / 25,
        beta2 = -1 / 25, lag = 1, reject_free = false, estimator_order = 4,
        start_growth = 100.0, dz_min = 1.0e-9, dz_max = 0.1, auto_init = true,
    )
    return SolpropJulia.PIControl(; merge(fields, NamedTuple(changes))...)
end

@testset "PI controller preserves bounded accepted history" begin
    control = test_control()
    history = SolpropJulia.NormHistory(control)
    @test isempty(SolpropJulia.control_history(history))
    @test_throws ArgumentError SolpropJulia.replace_last_norm!(history, 0.5)
    @test SolpropJulia.control_propose(control, 0.001, history) == 0.001
    for index in 1:11
        SolpropJulia.record_norm!(history, Float64(index))
        @test history.count == index
        @test length(history.readings) == control.lag + 2
    end
    @test SolpropJulia.control_history(history) == [9.0, 10.0, 11.0]
    SolpropJulia.replace_last_norm!(history, 0.5)
    @test SolpropJulia.control_history(history) == [9.0, 10.0, 0.5]
    saved = SolpropJulia.control_history(history)
    saved[1] = -7.0
    @test SolpropJulia.control_history(history)[1] == 9.0
    restored = SolpropJulia.NormHistory(control)
    SolpropJulia.restore_control!(restored, SolpropJulia.control_history(history), 11)
    @test restored.count == 11
    @test SolpropJulia.control_history(restored) == SolpropJulia.control_history(history)
    @test SolpropJulia.control_propose(control, 0.001, restored) ==
        SolpropJulia.control_propose(control, 0.001, history)
    SolpropJulia.record_norm!(restored, 0.25)
    @test restored.count == 12
    @test SolpropJulia.control_history(restored) == [10.0, 0.5, 0.25]
    @test_throws ArgumentError SolpropJulia.restore_control!(restored, [1.0], 12)
    @test restored.count == 12
    @test SolpropJulia.control_history(restored) == [10.0, 0.5, 0.25]
    SolpropJulia.restore_control!(restored, Float64[], 0)
    @test restored.count == 0
    @test isempty(SolpropJulia.control_history(restored))
    @test_throws ArgumentError SolpropJulia.control_propose(
        test_control(lag = 0), 0.001, history
    )
end

@testset "PI decisions use the executed norm and retain the original proposal" begin
    control = test_control(estimator_order = 1, safety = 1.0, max_factor = 2.0)
    # The clipped step's error is below tolerance, though the rescaled error is not.
    equivalent, accepted = SolpropJulia.control_decide(control, 0.5, 0.25, 0.5)
    @test equivalent == 2.0
    @test accepted
    @test !last(SolpropJulia.control_decide(control, 2.0, 0.5, 0.5))
    free = test_control(reject_free = true)
    @test last(SolpropJulia.control_decide(free, 2.0, 0.5, 0.5))
    for norm in (Inf, NaN)
        @test !last(SolpropJulia.control_decide(free, norm, 0.5, 0.5))
    end
    @test_throws ArgumentError SolpropJulia.control_decide(control, 0.5, 0.0, 0.5)
    @test SolpropJulia.control_start(control, 0.001, 0.0) == (0.1, 0.0)
    @test SolpropJulia.control_reject(control, 0.001, Inf) == 0.0002
    @test SolpropJulia.control_reject(control, 0.001, 0.0) == 0.001
    # A dyadic norm with the square-root exponent gives an exact factor of one half.
    @test SolpropJulia.control_reject(control, 0.001, 4.0) == 0.0005
    @test_throws SolpropJulia.StepFloorError SolpropJulia.control_reject(
        test_control(dz_min = 0.0005), 0.001, Inf
    )
    @test_throws ArgumentError test_control(safety = 0.0)
    @test_throws ArgumentError test_control(lag = -1)
    @test_throws ArgumentError test_control(beta1 = NaN)
end

function control_allocations(control, history)
    for _ in 1:3
        SolpropJulia.record_norm!(history, 0.5)
        SolpropJulia.control_propose(control, 0.001, history)
    end
    return @allocated begin
        equivalent, accepted = SolpropJulia.control_decide(control, 0.4, 0.01, 0.01)
        accepted || error("expected an accepted control test")
        SolpropJulia.record_norm!(history, equivalent)
        SolpropJulia.replace_last_norm!(history, equivalent)
        SolpropJulia.control_propose(control, 0.001, history)
        SolpropJulia.control_start(control, 0.001, equivalent)
        SolpropJulia.control_reject(control, 0.001, 1.2)
        nothing
    end
end

@testset "Warmed scalar control allocates nothing" begin
    control = test_control()
    history = SolpropJulia.NormHistory(control)
    control_allocations(control, history)
    @test control_allocations(control, history) == 0
end
