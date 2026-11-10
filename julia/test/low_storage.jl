# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

mutable struct LowStorageTestSource
    mixing::Matrix{ComplexF64}
    cubic::Float64
    forcing::ComplexF64
    calls::Int
    fail_at::Int
    positions::Vector{Float64}
end

function (source::LowStorageTestSource)(output, state, position)
    source.calls += 1
    source.positions[mod1(source.calls, 4)] = position
    source.calls == source.fail_at && error("source failed before its numerical work")
    for column in axes(state, 2), row in axes(state, 1)
        total = source.forcing * position^2
        for other in axes(state, 1)
            total += source.mixing[row, other] * state[other, column]
        end
        if !iszero(source.cubic)
            total += im * source.cubic * abs2(state[row, column]) * state[row, column]
        end
        output[row, column] = total
    end
    return nothing
end

mutable struct LowStorageTestFlow
    constant::Matrix{ComplexF64}
    slope::Matrix{ComplexF64}
    calls::Int
    fail_at::Int
    intervals::Matrix{Float64}
end

function (flow::LowStorageTestFlow)(output, state, start, finish)
    flow.calls += 1
    index = mod1(flow.calls, 4)
    flow.intervals[index, 1], flow.intervals[index, 2] = start, finish
    flow.calls == flow.fail_at && error("flow failed before its numerical work")
    for index in eachindex(output)
        phase = (finish - start) *
            (flow.constant[index] + flow.slope[index] * (finish + start) / 2)
        output[index] = iszero(phase) ? state[index] : exp(phase) * state[index]
    end
    return nothing
end

function low_storage_fixture(state)
    rows, columns = size(state)
    mixing = ComplexF64[
        (row == column ? -0.05 + 0.1im : 0.02 - 0.03im)
            for row in 1:rows, column in 1:rows
    ]
    source = LowStorageTestSource(mixing, 0.02, 0.01im, 0, 0, zeros(4))
    constant = ComplexF64[
        -row / 20 + im * column / 3 for row in 1:rows, column in 1:columns
    ]
    slope = ComplexF64[
        row / 100 + im * column / 10 for row in 1:rows, column in 1:columns
    ]
    flow = LowStorageTestFlow(constant, slope, 0, 0, zeros(4, 2))
    return source, flow
end

function low_storage_reference(state, position, step, source, flow)
    # Independent four-row RK4 anchored at the start, with explicit inverse-flow
    # transformed derivatives. It does not use the midpoint three-register folds.
    delta(offset) = offset .* (flow.constant .+ flow.slope .* (2position + offset) / 2)
    rhs(input, z) = source.mixing * input .+
        im * source.cubic .* abs2.(input) .* input .+ source.forcing * z^2
    first = rhs(state, position)
    half_flow = exp.(delta(step / 2))
    full_flow = exp.(delta(step))
    second = rhs(half_flow .* (state .+ step / 2 .* first), position + step / 2) ./
        half_flow
    third = rhs(half_flow .* (state .+ step / 2 .* second), position + step / 2) ./
        half_flow
    fourth = rhs(full_flow .* (state .+ step .* third), position + step) ./ full_flow
    return full_flow .* (state .+ step / 6 .* (first .+ 2second .+ 2third .+ fourth))
end

function low_storage_cycle!(
        work, state, tableau, source::S, flow::F, execution,
    ) where {S, F}
    return SolpropJulia.low_storage_trial!(
        work, state, 0.3, 0.025, tableau, source, flow, execution,
    )
end

@testset "Three-register midpoint RK4IP and exact work facts" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    tableau = SolpropJulia.RK4IPTableau()
    original = ComplexF64[row / 7 + im * column / 11 for row in 1:4, column in 1:3]
    mask = [true, false, true, true]
    work = SolpropJulia.LowStorageWork(ex, original, tableau; norm_mask = mask)
    @test work.norm_mask == mask
    @test !Base.mightalias(work.norm_mask, mask)
    @test sizeof(work.accumulator) + sizeof(work.stage) + sizeof(work.source) ==
        3sizeof(original)
    source, flow = low_storage_fixture(original)
    state = copy(original)
    fill!(work.accumulator, NaN)
    fill!(work.stage, NaN)
    fill!(work.source, NaN)
    expected = low_storage_reference(original, 0.3, 0.025, source, flow)
    facts = low_storage_cycle!(work, state, tableau, source, flow, ex)
    # Equivalent RK4 pictures differ by bounded exponential/product/sum rounding;
    # this small well-conditioned case admits 512eps against the field scale.
    @test maximum(abs, state - expected) <= 512eps(Float64) * maximum(abs, expected)
    @test facts.finite
    @test facts.source_calls == source.calls == 4
    @test facts.norm_calls == 1
    @test facts.error_norm == 0
    @test isnan(facts.stiffness)
    @test source.positions == [0.3, 0.3125, 0.3125, 0.325]
    @test flow.calls == 4
    @test flow.intervals == [0.3 0.3125; 0.3 0.3125; 0.3125 0.325; 0.3125 0.325]
    norm_reference = sqrt(sum(abs2, state[mask, :]))
    @test abs(facts.state_norm - norm_reference) <= 16eps(Float64) * norm_reference
    low_storage_cycle!(work, state, tableau, source, flow, ex)
    @test @allocated(low_storage_cycle!(work, state, tableau, source, flow, ex)) == 0
    for unused in 1:8
        expected = low_storage_reference(expected, 0.3, 0.025, source, flow)
    end
    copyto!(state, original)
    for unused in 1:9
        low_storage_cycle!(work, state, tableau, source, flow, ex)
    end
    @test maximum(abs, state - expected) <= 9 * 512eps(Float64) * maximum(abs, expected)
    @test_throws DimensionMismatch SolpropJulia.LowStorageWork(
        ex, original, tableau; norm_mask = [true],
    )
    @test_throws ArgumentError SolpropJulia.LowStorageWork(
        ex, zeros(ComplexF64, 0, 1), tableau,
    )
    saved = copy(state)
    for step in (0.0, -0.1, Inf, NaN)
        @test_throws ArgumentError SolpropJulia.low_storage_trial!(
            work, state, 0.3, step, tableau, source, flow, ex,
        )
        @test state == saved
    end
    @test_throws ArgumentError low_storage_cycle!(
        work, work.accumulator, tableau, source, flow, ex,
    )
    @test_throws DimensionMismatch low_storage_cycle!(
        work, zeros(ComplexF64, 1, 1), tableau, source, flow, ex,
    )
end

@testset "RK4IP fourth-order convergence" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    tableau = SolpropJulia.RK4IPTableau()
    errors = zeros(3)
    for (level, count) in enumerate((4, 8, 16))
        state = ones(ComplexF64, 1, 1)
        source = LowStorageTestSource(reshape([0.7 - 0.3im], 1, 1), 0, 0, 0, 0, zeros(4))
        flow = LowStorageTestFlow(
            fill(-0.2 + 0.4im, 1, 1), fill(0.1im, 1, 1),
            0, 0, zeros(4, 2)
        )
        work = SolpropJulia.LowStorageWork(ex, state, tableau)
        step = 0.4 / count
        for index in 0:(count - 1)
            SolpropJulia.low_storage_trial!(
                work, state, index * step, step, tableau, source, flow, ex,
            )
        end
        exact = exp((0.5 + 0.1im) * 0.4 + 0.1im * 0.4^2 / 2)
        errors[level] = abs(state[1] - exact)
    end
    # Smooth scalar D(z) and a linear complete source have this closed solution.
    # Fourth-order global error halves by sixteen; finite-step remainders stay
    # within this predeclared 12–20 ratio window at all three levels.
    @test 12 <= errors[1] / errors[2] <= 20
    @test 12 <= errors[2] / errors[3] <= 20
end

@testset "In-place LinearFlow and protected finite-state reporting" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    tableau = SolpropJulia.RK4IPTableau()
    zero_source! = function (output, state, position)
        fill!(output, 0)
        return nothing
    end
    state = ComplexF64[row / 8 + column * im / 9 for row in 1:3, column in 1:2]
    rate = fill(-0.2 + 0.7im, size(state))
    flow = SolpropJulia.LinearFlow(ex, SolpropJulia.DiagonalOperator(ex, rate), state)
    work = SolpropJulia.LowStorageWork(ex, state, tableau)
    initial = copy(state)
    SolpropJulia.low_storage_trial!(
        work, state, 0.0, 0.2, tableau, zero_source!, flow, ex,
    )
    expected = exp.(0.2rate) .* initial
    @test maximum(abs, state - expected) <= 32eps(Float64) * maximum(abs, expected)
    low_storage_cycle!(work, state, tableau, zero_source!, flow, ex)
    @test @allocated(low_storage_cycle!(work, state, tableau, zero_source!, flow, ex)) == 0
    damped = SolpropJulia.LinearFlow(
        ex, SolpropJulia.DiagonalOperator(ex, fill(-1.0e4 + 0im, size(state))), state,
    )
    facts = SolpropJulia.low_storage_trial!(
        work, state, 0.0, 0.2, tableau, zero_source!, damped, ex,
    )
    @test facts.finite
    @test iszero(state)
    identity_flow! = function (output, input, start, finish)
        copyto!(output, input)
        return nothing
    end
    for amplitude in (1.0e-250, 1.0e250)
        fill!(state, complex(amplitude, amplitude))
        facts = low_storage_cycle!(work, state, tableau, zero_source!, identity_flow!, ex)
        @test facts.finite
        expected_norm = amplitude * sqrt(2length(state))
        @test abs(facts.state_norm - expected_norm) <= 16eps(Float64) * expected_norm
    end
    work.norm_mask[3] = false
    state .= 1
    state[3, 1] = NaN
    facts = low_storage_cycle!(work, state, tableau, zero_source!, identity_flow!, ex)
    @test !facts.finite
    @test facts.state_norm == 2
end

@testset "Failed in-place steps have no successful facts or rollback" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    tableau = SolpropJulia.RK4IPTableau()
    original = fill(1.0 + im, 3, 2)
    for failed_call in 1:4
        state = copy(original)
        source, flow = low_storage_fixture(state)
        source.fail_at = failed_call
        work = SolpropJulia.LowStorageWork(ex, state, tableau)
        @test_throws ErrorException low_storage_cycle!(
            work, state, tableau, source, flow, ex,
        )
        @test source.calls == failed_call
        if failed_call == 1
            @test state == original
        else
            @test state != original
            phase = 0.0125 .* (flow.constant .+ flow.slope .* (0.3 + 0.3125) / 2)
            @test maximum(abs, state - exp.(phase) .* original) <=
                32eps(Float64) * maximum(abs, original)
        end
    end
    for failed_call in 1:4
        state = copy(original)
        source, flow = low_storage_fixture(state)
        flow.fail_at = failed_call
        work = SolpropJulia.LowStorageWork(ex, state, tableau)
        @test_throws ErrorException low_storage_cycle!(
            work, state, tableau, source, flow, ex,
        )
        @test flow.calls == failed_call
        @test source.calls == (failed_call <= 2 ? 1 : failed_call)
    end
    bad_source!(output, state, position) = output
    state = copy(original)
    source, flow = low_storage_fixture(state)
    work = SolpropJulia.LowStorageWork(ex, state, tableau)
    @test_throws ArgumentError low_storage_cycle!(
        work, state, tableau, bad_source!, flow, ex,
    )
    @test state == original
    @test flow.calls == 0
end
