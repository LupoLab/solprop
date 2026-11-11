# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

function precision_march_source!(output, input, position)
    coefficient = eltype(input)(0.03, -0.02)
    for index in eachindex(input)
        output[index] = coefficient * input[index]
    end
    return nothing
end

function precision_march_callbacks(execution, initial)
    generator = fill(-0.1 + 0.2im, size(initial))
    operator = SolpropJulia.DiagonalOperator(execution, generator)
    return SolpropJulia.LinearFlow(execution, operator, initial)
end

@testset "Trial-time coefficients refuse before any buffer or callback changes" begin
    ex = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    state = fill(1.0f0 + 0.0f0im, 2, 1)
    original = copy(state)
    tableau = SolpropJulia.EmbeddedTableau([0.0], zeros(1, 1), [1.0], [0.0])
    work = SolpropJulia.RungeKuttaWork(state, tableau)
    fill!(work.candidate, 7)
    calls = Ref(0)
    source! = (output, input, position) -> (calls[] += 1; copyto!(output, input))
    flow = precision_march_callbacks(ex, state)
    for coefficient in (1.0e-100, 1.0e100)
        replacement = SolpropJulia.EmbeddedTableau(
            [0.0], zeros(1, 1), [coefficient], [0.0],
        )
        @test_throws ArgumentError SolpropJulia.rk_trial!(
            work, state, 0.0, 0.01, replacement, source!, flow, ex,
        )
        @test calls[] == 0
        @test state == original && all(==(7), work.candidate)
    end
end

@testset "Masked norm rows do not evaluate unused denominators" begin
    ex = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    state = reshape(ComplexF32[1.0f10, 0], 2, 1)
    tableau = SolpropJulia.EmbeddedTableau([0.0], zeros(1, 1), [1.0], [0.0])
    work = SolpropJulia.RungeKuttaWork(state, tableau)
    copyto!(work.candidate, state)
    fill!(work.defect, 1)
    actual = SolpropJulia.rk_error_norm!(
        work, state, [true, false], [1.0, 1.0e308], 1.0, 0.0, ex,
    )
    expected = inv(Float64(real(state[1])))
    # One binary32 denominator narrowing and division dominate the host merge.
    @test abs(actual - expected) <= 4eps(Float32) * expected
end

function precision_march_rk_allocation(work, state, tableau, flow, execution)
    SolpropJulia.rk_trial!(
        work, state, 0.0, 0.005, tableau, precision_march_source!, flow, execution,
    )
    return @allocated SolpropJulia.rk_trial!(
        work, state, 0.0, 0.005, tableau, precision_march_source!, flow, execution,
    )
end

function precision_march_norm_allocation(work, state, mask, weights, execution)
    SolpropJulia.rk_error_norm!(work, state, mask, weights, 0.001, 0.0, execution)
    return @allocated SolpropJulia.rk_error_norm!(
        work, state, mask, weights, 0.001, 0.0, execution,
    )
end

function precision_march_low_allocation(work, state, method, flow, execution)
    SolpropJulia.low_storage_trial!(
        work, state, 0.0, 0.005, method, precision_march_source!, flow, execution,
    )
    return @allocated SolpropJulia.low_storage_trial!(
        work, state, 0.0, 0.005, method, precision_march_source!, flow, execution,
    )
end

@testset "Binary32 linear propagation rounds the wide factor before state work" begin
    ex = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    state = ComplexF32[1.3 + 0.7im -0.2 + 1.1im; 0 -0.0im]
    phase = ComplexF64[-0.03 + 10000im 1.0e-10 + 2.0e-12im; 0 0]
    output = similar(state)
    SolpropJulia.propagate!(output, state, phase, ex)
    expected = ComplexF32.(exp.(phase)) .* state
    @test maximum(abs, output - expected) <= 2eps(Float32) * maximum(abs, expected)
    @test output[2, 1] === state[2, 1]
    @test output[2, 2] === state[2, 2]
    SolpropJulia.linear_increment!(output, state, phase, ex)
    expected = ComplexF32.(SolpropJulia._complex_expm1.(phase)) .* state
    @test maximum(abs, output - expected) <= 2eps(Float32) * maximum(abs, expected)
    @test all(iszero, output[2, :])
end

@testset "Binary32 RK and low-storage steps retain binary64 coordinates" begin
    tableau = SolpropJulia.EmbeddedTableau(
        [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
    )
    initial = ComplexF32[complex(row / 7, column / 11) for row in 1:9, column in 1:3]
    for tasks in unique((1, Threads.nthreads(:default)))
        ex = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        flow = precision_march_callbacks(ex, initial)
        @test eltype(flow.exponent) === ComplexF64
        for low_storage in (false, true)
            state = copy(initial)
            method = low_storage ? SolpropJulia.RK4IPTableau() : tableau
            work = low_storage ? SolpropJulia.LowStorageWork(ex, state, method) :
                SolpropJulia.RungeKuttaWork(state, method)
            mask = fill(true, size(state, 1))
            for step in 1:8
                if low_storage
                    facts = SolpropJulia.low_storage_trial!(
                        work, state, (step - 1) * 0.005, 0.005, method,
                        precision_march_source!, flow, ex,
                    )
                    @test facts.finite
                    @test facts.state_norm isa Float64
                    @test facts.source_calls == 4
                else
                    SolpropJulia.rk_trial!(
                        work, state, (step - 1) * 0.005, 0.005, method,
                        precision_march_source!, flow, ex,
                    )
                    SolpropJulia.rk_commit!(state, work, mask, ex)
                end
            end
            exact = ComplexF64.(initial) .* exp(0.04 * (-0.07 + 0.18im))
            # Eight trials need fewer than 1024 rounded scalar operations per
            # element including their flows and stage sums. The linear sensitivity
            # is bounded by exp(0.04*(|D|+|N|)); the Heun truncation bound adds
            # L*h^2*|N|^3*exp(L*(|D|+|N|))/6, independently of the measured result.
            growth = exp(0.04 * (abs(-0.1 + 0.2im) + abs(0.03 - 0.02im)))
            gamma = 1024eps(Float32) / (1 - 1024eps(Float32))
            truncation = 0.04 * 0.005^2 * abs(0.03 - 0.02im)^3 * growth / 6
            bound = (gamma * growth + truncation) * maximum(abs, initial)
            @test bound > 0
            @test maximum(abs, ComplexF64.(state) - exact) <= bound
            @test eltype(work.norm_parts) === Float32
            if !low_storage
                @test precision_march_rk_allocation(work, state, method, flow, ex) == 0
            else
                @test precision_march_low_allocation(work, state, method, flow, ex) == 0
                original = copy(state)
                @test_throws ArgumentError SolpropJulia.low_storage_trial!(
                    work, state, 0.0, 1.0e-100, method,
                    precision_march_source!, flow, ex,
                )
                @test state == original
            end
        end
    end
end

@testset "Binary32 norm uses protected sums and wide physical scalar controls" begin
    ex = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    tableau = SolpropJulia.EmbeddedTableau([0.0], zeros(1, 1), [1.0], [0.0])
    rows = 4097
    state = ComplexF32[
        complex(ldexp(Float32(1 + (row % 7) / 8), exponent), 0.0f0)
            for row in 1:rows, exponent in (-140, 0, 120)
    ]
    work = SolpropJulia.RungeKuttaWork(state, tableau)
    work.candidate .= state
    work.defect .= state .* 0.02f0
    mask = [row % 3 != 0 for row in 1:rows]
    weights = [ldexp(1.0, (row % 11) - 5) for row in 1:rows]
    old = sqrt(sum(abs2, ComplexF64.(state[mask, :])))
    expected = sqrt(
        sum(
            abs2, ComplexF64.(work.defect[mask, :]) ./
                (0.001old .* weights[mask])
        )
    )
    actual = SolpropJulia.rk_error_norm!(work, state, mask, weights, 0.001, 0.0, ex)
    # Max normalization, division, squared products and the represented denominator
    # each contribute first-order rounding. Sum2s adds gamma_(2n-1)^2; the small
    # host merge/sqrt uses binary64. Underflowed terms are below 2n*2^-149 before
    # physical restoration and far below this positive fixture's relative budget.
    unit = eps(Float32) / 2
    gamma = (2rows - 1) * unit / (1 - (2rows - 1) * unit)
    budget = (24unit + 4gamma^2 + 32eps()) * expected
    @test budget > 0
    @test abs(actual - expected) <= budget
    @test precision_march_norm_allocation(work, state, mask, weights, ex) == 0
    work.candidate[3, 1] = NaN32
    @test isinf(SolpropJulia.rk_error_norm!(work, state, mask, weights, 0.001, 0.0, ex))
    work.candidate .= state
    work.defect[1, 1] = NaN32
    @test isinf(SolpropJulia.rk_error_norm!(work, state, mask, weights, 0.001, 0.0, ex))
    fill!(state, 0)
    fill!(work.candidate, 0)
    fill!(work.defect, 0)
    @test SolpropJulia.rk_error_norm!(work, state, mask, weights, 0.001, 0.0, ex) === 0.0
    work.defect[1, 1] = 1
    @test isinf(SolpropJulia.rk_error_norm!(work, state, mask, weights, 0.001, 0.0, ex))
    @test_throws ArgumentError SolpropJulia._march_scalar(Float32, 1.0e-100)
    @test_throws ArgumentError SolpropJulia._march_scalar(Float32, 1.0e100)
    @test SolpropJulia._march_scalar(Float32, 0.0) === 0.0f0
end
