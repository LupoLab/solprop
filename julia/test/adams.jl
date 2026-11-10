# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct AdamsPolynomialSource
    degree::Int
    linear::ComplexF64
end

function (source::AdamsPolynomialSource)(output, state, position)
    for index in eachindex(output)
        output[index] = source.linear * state[index] + position^source.degree
    end
    return nothing
end

struct AdamsPolynomialOperator
    constant::ComplexF64
    slope::ComplexF64
end

function (operator::AdamsPolynomialOperator)(output, z0, z1)
    fill!(output, (z1 - z0) * (operator.constant + operator.slope * (z1 + z0) / 2))
    return nothing
end

function (operator::AdamsPolynomialOperator)(output, position, before::Bool)
    fill!(output, operator.constant + operator.slope * position)
    return nothing
end

function adams_test_history(execution, state, maximum_order; compensated = false)
    history = SolpropJulia.AdamsHistory(execution, state, maximum_order; compensated)
    history.positions[1:maximum_order] .= -collect(0:(maximum_order - 1))
    history.occupied[1:maximum_order] .= true
    return history
end

function adams_basis_reference(nodes)
    return setprecision(BigFloat, 256) do
        count = length(nodes)
        vandermonde = [
            BigFloat(nodes[row])^(column - 1) for row in 1:count, column in 1:count
        ]
        transpose(inv(vandermonde))
    end
end

@testset "Julia-owned nonuniform Adams coefficients" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    state = zeros(ComplexF64, 3, 2)
    history = adams_test_history(ex, state, 6)
    stencil = SolpropJulia.AdamsStencil(6)
    coefficients = SolpropJulia.AdamsCoefficients(6)
    for order in 1:6, positions in (
                [0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
                [0.0, -0.7, -1.9, -3.1, -4.8, -6.0],
            )
        history.positions[1:6] .= positions
        SolpropJulia.adams_stencil!(stencil, order, 1:order, (order,))
        SolpropJulia.prepare_adams_coefficients!(coefficients, stencil, history, 0.0, 0.4)
        old = positions[1:order] / 0.4
        for (actual, nodes) in (
                (coefficients.predictor, old), (coefficients.corrector, [1.0; old]),
            )
            expected = adams_basis_reference(nodes)
            count = length(nodes)
            # Direct factor products versus a 256-bit Vandermonde solve; count<=7.
            # 512eps bounds the products/divisions and the conditioning of this set.
            @test maximum(abs, actual[1:count, 1:count] - expected) <=
                512eps(Float64) * maximum(abs, expected)
            @test iszero(actual[(count + 1):end, :])
            @test iszero(actual[:, (count + 1):end])
        end
        # The factored divided difference times the product polynomial equals
        # corrector minus predictor coefficients before integration.
        factored = coefficients.defect_samples[:, 1] *
            transpose(coefficients.defect_polynomial[:, 1])
        difference = copy(coefficients.corrector)
        difference[2:end, :] -= coefficients.predictor[1:(end - 1), :]
        @test maximum(abs, factored - difference) <=
            512eps(Float64) * max(1, maximum(abs, difference))
        @test iszero(coefficients.compensation_defects[:, 2:3])
        @test iszero(coefficients.defect_samples[:, 2:3])
    end
    history.positions[1:6] .= -collect(0:5)
    SolpropJulia.adams_stencil!(stencil, 4, (1, 2, 3, 4, 5), (3, 4, 5))
    SolpropJulia.prepare_adams_coefficients!(coefficients, stencil, history, 0.0, 1.0)
    @test maximum(abs, 24coefficients.compensation_predictor[1:4] - [55, -59, 37, -9]) <=
        64eps(Float64) * 59
    @test maximum(
        abs, 720coefficients.compensation_corrector[1:5] -
            [251, 646, -264, 106, -19]
    ) <= 128eps(Float64) * 646
    SolpropJulia.prepare_adams_coefficients!(coefficients, stencil, history, 0.0, 1.0)
    @test @allocated(
        SolpropJulia.prepare_adams_coefficients!(
            coefficients, stencil, history, 0.0, 1.0,
        )
    ) == 0
    saved_slots, saved_orders = copy(stencil.slots), copy(stencil.orders)
    @test_throws ArgumentError SolpropJulia.adams_stencil!(
        stencil, 4, view(stencil.slots, 1:5), (3, 4, 5),
    )
    @test_throws ArgumentError SolpropJulia.adams_stencil!(
        stencil, 4, (1, 2, 3, 4, 5), view(stencil.orders, 1:3),
    )
    @test stencil.slots == saved_slots
    @test stencil.orders == saved_orders
    history.positions[1:3] .= [1.1 + 0.1, 1.1, 1.0]
    SolpropJulia.adams_stencil!(stencil, 2, (1, 2, 3), (1, 2, 3))
    SolpropJulia.prepare_adams_coefficients!(coefficients, stencil, history, 1.2, 0.1)
    @test stencil.nodes[1] == ((1.1 + 0.1) - 1.2) / 0.1
    @test !iszero(stencil.nodes[1])
    @test history.positions[1] == 1.1 + 0.1
    @test maximum(
        abs, coefficients.predictor[1:2, 1:2] -
            adams_basis_reference(stencil.nodes[1:2])
    ) <= 64eps(Float64)
    @test_throws ArgumentError SolpropJulia.adams_stencil!(stencil, 4, (1, 1, 2, 3), (4,))
    @test_throws ArgumentError SolpropJulia.adams_stencil!(stencil, 4, (1, 2, 3, 4), (3,))
    @test_throws DimensionMismatch SolpropJulia.adams_stencil!(stencil, 4, (1, 2), (4,))
    @test_throws ArgumentError SolpropJulia.prepare_adams_coefficients!(
        coefficients, stencil, history, 0.0, 0.0,
    )
    history.positions[2] = history.positions[1]
    @test_throws ArgumentError SolpropJulia.prepare_adams_coefficients!(
        coefficients, stencil, history, 0.0, 1.0,
    )
    @test_throws ArgumentError SolpropJulia.AdamsStencil(7)
end

function adams_test_trial!(work, state, history, stencil, source, operator, execution)
    SolpropJulia.adams_trial!(
        work, state, 0.0, 0.25, history, stencil, source, operator, operator, execution,
    )
    return nothing
end

@testset "Exponential Adams polynomial trials and stored predictor" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    initial = ComplexF64[row / 8 + im * column / 7 for row in 1:3, column in 1:2]
    operator = AdamsPolynomialOperator(0, 0)
    for order in 1:6
        history = SolpropJulia.AdamsHistory(ex, initial, 6)
        stencil = SolpropJulia.AdamsStencil(6)
        work = SolpropJulia.AdamsWork(ex, initial, 6)
        source = AdamsPolynomialSource(order, 0)
        for slot in 1:order
            SolpropJulia.adams_seed!(
                history, slot, initial, -(slot - 1) / 4,
                source, operator, ex; identity = 10 + slot,
            )
        end
        SolpropJulia.adams_stencil!(stencil, order, 1:order, (order,))
        saved_state, saved_history = copy(initial), copy(history.source)
        @test SolpropJulia.adams_trial!(
            work, initial, 0.0, 0.25, history, stencil, source, operator, operator, ex,
        ) == 1
        expected = initial .+ 0.25^(order + 1) / (order + 1)
        # Corrector interpolates the degree-q manufactured source exactly. The
        # bounded seven-node products/sums get 512eps against the state/increment.
        @test maximum(abs, work.candidate - expected) <= 512eps(Float64)
        @test maximum(abs, work.candidate - work.predictor - work.defects[:, :, 1]) <=
            512eps(Float64)
        @test initial == saved_state
        @test history.source == saved_history
        @test iszero(work.defects[:, :, 2:3])
        adams_test_trial!(work, initial, history, stencil, source, operator, ex)
        @test @allocated(
            adams_test_trial!(
                work, initial, history, stencil, source, operator, ex,
            )
        ) == 0
        SolpropJulia.adams_accept!(history, 8, work, ex; identity = 123)
        @test history.source[:, :, 8] == work.source
        @test history.positions[8] == 0.25
        @test history.identities[8] == 123
    end
    history = SolpropJulia.AdamsHistory(ex, initial, 3; compensated = true)
    stencil = SolpropJulia.AdamsStencil(3)
    work = SolpropJulia.AdamsWork(ex, initial, 3; compensated = true)
    source = AdamsPolynomialSource(0, 0.3im)
    varying = AdamsPolynomialOperator(-0.1im, 0.2im)
    for slot in 1:3
        SolpropJulia.adams_seed!(
            history, slot, initial .* (1 + slot / 8), -(slot - 1) / 4,
            source, varying, ex,
        )
    end
    SolpropJulia.adams_stencil!(stencil, 2, (1, 2, 3), (1, 2, 3))
    adams_test_trial!(work, initial, history, stencil, source, varying, ex)
    @test @allocated(
        adams_test_trial!(
            work, initial, history, stencil, source, varying, ex,
        )
    ) == 0
    SolpropJulia.adams_accept!(history, 5, work, ex; identity = 99)
    @test history.states[:, :, 5] == work.predictor
    @test history.states[:, :, 5] != work.candidate
    @test history.rates[:, :, 5] == work.endpoint_rate
    @test history.source[:, :, 5] == 0.3im .* work.predictor .+ 1
    for sample in 1:3
        chord = work.exponent / 0.25
        expected = exp.((1 - stencil.nodes[sample]) .* work.exponent) .*
            ((history.rates[:, :, sample] - chord) .* history.states[:, :, sample])
        @test maximum(abs, work.carried[:, :, sample] - expected) <=
            32eps(Float64) * maximum(abs, expected)
    end
end

function adams_norm_cycle!(scores, work, state, mask, weights, execution)
    SolpropJulia.adams_error_norms!(
        scores, work, state, mask, weights, 0.1, 0.01, execution,
    )
    return nothing
end

@testset "Adams old-state scaled norms and structural homogeneous flow" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    state = fill(2.0 + 3.0im, 4, 2)
    history = SolpropJulia.AdamsHistory(ex, state, 2)
    stencil = SolpropJulia.AdamsStencil(2)
    work = SolpropJulia.AdamsWork(ex, state, 2)
    operator = AdamsPolynomialOperator(-3 + 4im, 2im)
    forbidden(arguments...) = error("structurally absent callback was invoked")
    @test SolpropJulia.adams_seed!(
        history, 1, state, 0.0, forbidden, forbidden, ex; source_absent = true,
    ) == 0
    SolpropJulia.adams_stencil!(stencil, 1, (1,), (1,))
    @test SolpropJulia.adams_trial!(
        work, state, 0.0, 0.25, history, stencil, forbidden, operator, forbidden, ex;
        source_absent = true,
    ) == 0
    @test work.candidate == exp.((0.25 * (-3 + 4.25im))) .* state
    @test iszero(work.defects)
    @test iszero(work.source)
    scores = zeros(3)
    mask, weights = fill(true, 4), [0.5, 1.0, 2.0, 3.0]
    work.defects[:, :, 1] .= 0.2 + 0.3im
    work.candidate .= 1.0e20
    adams_norm_cycle!(scores, work, state, mask, weights, ex)
    expected = sqrt(
        sum(
            abs2, (0.2 + 0.3im) ./
                (0.1 * sqrt(sum(abs2, state)) .* weights .+ 0.01)
        ) * 2
    )
    @test abs(scores[1] - expected) <= 32eps(Float64) * expected
    @test iszero(scores[2:3])
    @test @allocated(adams_norm_cycle!(scores, work, state, mask, weights, ex)) == 0
    for amplitude in (1.0e-250, 1.0e250)
        state .= amplitude + amplitude * im
        work.candidate .= state
        work.defects[:, :, 1] .= state
        SolpropJulia.adams_error_norms!(scores, work, state, mask, ones(4), 1.0, 0.0, ex)
        @test abs(scores[1] - 1) <= 32eps(Float64)
    end
    state .= 0
    work.defects .= 0
    SolpropJulia.adams_error_norms!(scores, work, state, mask, weights, 0.0, 0.0, ex)
    @test iszero(scores)
    mask[4] = false
    state[4, 1] = NaN
    adams_norm_cycle!(scores, work, state, mask, weights, ex)
    @test scores[1] == Inf
    state .= 1
    @test_throws ErrorException SolpropJulia.adams_trial!(
        work, state, 0.0, 0.25, history, stencil, forbidden, operator, operator, ex,
    )
    @test !work.ready
    @test_throws ArgumentError SolpropJulia.adams_accept!(history, 2, work, ex)
    invalid_return(output, input, position) = output
    @test_throws ArgumentError SolpropJulia.adams_trial!(
        work, state, 0.0, 0.25, history, stencil, invalid_return, operator, operator, ex,
    )
    @test !work.ready
end

function adams_manufactured_trial(order, step, execution)
    state = ones(ComplexF64, 2, 2)
    history = SolpropJulia.AdamsHistory(execution, state, order; compensated = true)
    stencil = SolpropJulia.AdamsStencil(order)
    work = SolpropJulia.AdamsWork(execution, state, order; compensated = true)
    operator = AdamsPolynomialOperator(-0.1 + 0.2im, 0.3 + 0.4im)
    # u'=((a+bz)+lambda)u; the constant forcing of AdamsPolynomialSource is removed
    # here so the independent closed form is exp((a+lambda)z+bz^2/2).
    source! = function (output, input, position)
        for index in eachindex(output)
            output[index] = (0.4 - 0.3im) * input[index]
        end
        return nothing
    end
    for slot in 1:order
        position = -(slot - 1) * step
        sampled = fill(
            exp(
                (operator.constant + 0.4 - 0.3im) * position +
                    operator.slope * position^2 / 2
            ), size(state),
        )
        SolpropJulia.adams_seed!(
            history, slot, sampled, position, source!, operator, execution,
        )
    end
    SolpropJulia.adams_stencil!(stencil, order, 1:order, (order,))
    SolpropJulia.adams_trial!(
        work, state, 0.0, step, history, stencil, source!, operator, operator, execution,
    )
    expected = exp((operator.constant + 0.4 - 0.3im) * step + operator.slope * step^2 / 2)
    return maximum(abs, work.candidate .- expected)
end

@testset "Transported compensation local convergence" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    for order in 1:4
        errors = [adams_manufactured_trial(order, step, ex) for step in (0.2, 0.1, 0.05)]
        # Exact starting history and the manufactured varying-operator solution
        # isolate local error: PEC(q,q+1) has order q+2. These three levels remain
        # above roundoff; a factor-two envelope admits the finite-step remainder.
        expected_ratio = 2.0^(order + 2)
        for level in 1:2
            @test expected_ratio / 2 <= errors[level] / errors[level + 1] <= 2expected_ratio
        end
    end
end

function adams_dense_cycle!(output, dense, work, state, history, stencil, execution)
    SolpropJulia.adams_retain_dense!(
        dense, work, state, work.candidate, history, stencil, execution,
    )
    SolpropJulia.adams_observe!(output, dense, 0.1, execution)
    return nothing
end

@testset "Retained Adams polynomial observations" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    state = fill(1.0 + 2im, 3, 2)
    operator = AdamsPolynomialOperator(0, 0)
    source = AdamsPolynomialSource(3, 0)
    history = SolpropJulia.AdamsHistory(ex, state, 4)
    stencil = SolpropJulia.AdamsStencil(4)
    work = SolpropJulia.AdamsWork(ex, state, 4)
    dense = SolpropJulia.AdamsDense(ex, state, 4)
    output = similar(state)
    for slot in 1:3
        SolpropJulia.adams_seed!(
            history, slot, state, -(slot - 1) / 4, source, operator, ex,
        )
    end
    SolpropJulia.adams_stencil!(stencil, 3, (1, 2, 3), (3,))
    adams_test_trial!(work, state, history, stencil, source, operator, ex)
    adams_dense_cycle!(output, dense, work, state, history, stencil, ex)
    @test @allocated(
        adams_dense_cycle!(
            output, dense, work, state, history, stencil, ex,
        )
    ) == 0
    @test iszero(dense.source[:, :, 5])
    for point in (0.0, 0.025, 0.1, 0.2, 0.25)
        SolpropJulia.adams_observe!(output, dense, point, ex)
        # The retained degree-three source integrates to z^4/4; coefficient
        # contraction and phi moments use a 256eps absolute state-scale budget.
        @test maximum(abs, output - (state .+ point^4 / 4)) <=
            256eps(Float64) * abs(state[1])
    end
    saved = copy(output)
    fill!(history.source, NaN)
    fill!(work.source, NaN)
    fill!(work.candidate, NaN)
    SolpropJulia.adams_observe!(output, dense, 0.25, ex)
    @test output == saved
    SolpropJulia.adams_observe!(output, dense, 0.0, ex)
    @test isequal(output, state)
    @test_throws ArgumentError SolpropJulia.adams_observe!(output, dense, -0.1, ex)
    @test_throws ArgumentError SolpropJulia.adams_observe!(dense.initial, dense, 0.1, ex)

    compensated_history = SolpropJulia.AdamsHistory(ex, state, 2; compensated = true)
    compensated_work = SolpropJulia.AdamsWork(ex, state, 2; compensated = true)
    compensated_dense = SolpropJulia.AdamsDense(ex, state, 2; compensated = true)
    compensated_stencil = SolpropJulia.AdamsStencil(2)
    varying = AdamsPolynomialOperator(-0.1 + 0.2im, 0.3im)
    for slot in 1:2
        SolpropJulia.adams_seed!(
            compensated_history, slot, state, -(slot - 1) / 4, source, varying, ex,
        )
    end
    SolpropJulia.adams_stencil!(compensated_stencil, 2, (1, 2), (2,))
    adams_test_trial!(
        compensated_work, state, compensated_history, compensated_stencil,
        source, varying, ex,
    )
    adams_dense_cycle!(
        output, compensated_dense, compensated_work, state,
        compensated_history, compensated_stencil, ex,
    )
    @test @allocated(
        adams_dense_cycle!(
            output, compensated_dense, compensated_work, state,
            compensated_history, compensated_stencil, ex,
        )
    ) == 0
    # Force the documented strong-damping branch with poisoned interpolation
    # inputs: no inverse endpoint exponential or discarded source is evaluated.
    fill!(compensated_dense.exponent, -2000)
    fill!(compensated_dense.source, NaN)
    fill!(compensated_dense.rates, NaN)
    SolpropJulia.adams_observe!(output, compensated_dense, 0.01, ex)
    @test all(isfinite, output)
    @test output == exp(-80) .* state
end

function adams_big_moment(argument, degree, upper)
    point = Complex{BigFloat}(argument)
    fraction = BigFloat(upper)
    term = fraction^(degree + 1) / (degree + 1)
    total = Complex{BigFloat}(term)
    for index in 1:200
        term *= point * fraction / (degree + index + 1)
        total += term
        abs(term) <= eps(BigFloat) * abs(total) && break
    end
    return total
end

function adams_reference_integral(nodes, samples, exponent, upper)
    basis = adams_basis_reference(nodes)
    count = length(nodes)
    return sum(
        adams_big_moment(exponent, power - 1, upper) *
            sum(basis[sample, power] * samples[sample] for sample in 1:count)
            for power in 1:count
    )
end

function adams_reference_compensation(nodes, samples, exponent, upper)
    basis = adams_basis_reference(nodes)
    count = length(nodes)
    return sum(
        sum(basis[sample, power] * BigFloat(upper)^power / power for power in 1:count) *
            exp((BigFloat(upper) - nodes[sample]) * exponent) * samples[sample]
            for sample in 1:count
    )
end

@testset "Adams trial and dense values against independent high-precision integrals" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    state = reshape(ComplexF64[1 + 0.3im], 1, 1)
    history = SolpropJulia.AdamsHistory(ex, state, 3; compensated = true)
    stencil = SolpropJulia.AdamsStencil(3)
    work = SolpropJulia.AdamsWork(ex, state, 3; compensated = true)
    dense = SolpropJulia.AdamsDense(ex, state, 3; compensated = true)
    source = AdamsPolynomialSource(2, 0.2 - 0.1im)
    operator = AdamsPolynomialOperator(-0.4 + 0.7im, 0.2 + 0.3im)
    positions = [0.0, -0.25, -0.625]
    for slot in 1:3
        sampled = state .* (1 + slot / 11)
        SolpropJulia.adams_seed!(
            history, slot, sampled, positions[slot], source, operator, ex;
            identity = 40 + slot,
        )
    end
    SolpropJulia.adams_stencil!(stencil, 2, (1, 2, 3), (1, 2, 3))
    adams_test_trial!(work, state, history, stencil, source, operator, ex)
    output = similar(state)
    SolpropJulia.adams_retain_dense!(
        dense, work, state, work.candidate, history, stencil, ex,
    )
    setprecision(BigFloat, 256) do
        h = BigFloat(0.25)
        x = Complex{BigFloat}(work.exponent[1])
        initial = Complex{BigFloat}(state[1])
        nodes = BigFloat.(positions) / h
        stored = Complex{BigFloat}.(vec(history.source)[1:3])
        sample_states = Complex{BigFloat}.(vec(history.states)[1:3])
        sample_rates = Complex{BigFloat}.(vec(history.rates)[1:3])
        terms = (sample_rates .- x / h) .* sample_states
        prediction = exp(x) * initial + h * (
            adams_reference_integral(nodes[1:2], stored[1:2], x, 1) +
                adams_reference_compensation(nodes[1:2], terms[1:2], x, 1)
        )
        newest = Complex{BigFloat}(source.linear) * prediction + h^2
        endpoint_rate = Complex{BigFloat}(work.endpoint_rate[1])
        all_nodes = [one(h); nodes]
        all_sources = [newest; stored]
        all_terms = [(endpoint_rate - x / h) * prediction; terms]
        correction = exp(x) * initial + h * (
            adams_reference_integral(all_nodes[1:3], all_sources[1:3], x, 1) +
                adams_reference_compensation(all_nodes[1:3], all_terms[1:3], x, 1)
        )
        # Independent 256-bit Vandermonde basis solves and defining-series
        # integrals; 1024eps covers the seven-node coefficient/phi/contraction path.
        bound = 1024eps(Float64) * max(abs(initial), abs(correction), 1)
        @test abs(work.predictor[1] - prediction) <= bound
        @test abs(work.source[1] - newest) <= bound
        @test abs(work.candidate[1] - correction) <= bound
        for order in 1:3
            corrected_increment = h * (
                adams_reference_integral(
                    all_nodes[1:(order + 1)], all_sources[1:(order + 1)], x, 1,
                ) + adams_reference_compensation(
                    all_nodes[1:(order + 1)], all_terms[1:(order + 1)], x, 1,
                )
            )
            predicted_increment = h * (
                adams_reference_integral(nodes[1:order], stored[1:order], x, 1) +
                    adams_reference_compensation(nodes[1:order], terms[1:order], x, 1)
            )
            @test abs(
                work.defects[1, 1, order] -
                    (corrected_increment - predicted_increment)
            ) <= bound
        end
        for fraction in (0.1, 0.3, 0.7, 0.9)
            SolpropJulia.adams_observe!(output, dense, 0.25fraction, ex)
            expected = exp(fraction * x) * initial + h * (
                adams_reference_integral(all_nodes[1:3], all_sources[1:3], x, fraction) +
                    adams_reference_compensation(
                    all_nodes[1:3], all_terms[1:3], x, fraction,
                )
            )
            @test abs(output[1] - expected) <= bound
        end
    end
    copied_history = SolpropJulia.AdamsHistory(ex, state, 3; compensated = true)
    copied_dense = SolpropJulia.AdamsDense(ex, state, 3; compensated = true)
    SolpropJulia.adams_copy_history!(copied_history, history)
    SolpropJulia.adams_copy_dense!(copied_dense, dense)
    @test @allocated(SolpropJulia.adams_copy_history!(copied_history, history)) == 0
    @test @allocated(SolpropJulia.adams_copy_dense!(copied_dense, dense)) == 0
    for name in (:source, :states, :rates, :positions, :identities, :occupied)
        @test isequal(getproperty(copied_history, name), getproperty(history, name))
        @test !Base.mightalias(
            getproperty(copied_history, name), getproperty(history, name),
        )
    end
    for name in (
            :initial, :endpoint, :exponent, :source, :states, :rates, :coefficients, :nodes,
        )
        @test isequal(getproperty(copied_dense, name), getproperty(dense, name))
        @test !Base.mightalias(getproperty(copied_dense, name), getproperty(dense, name))
    end
    SolpropJulia.adams_observe!(output, dense, 0.125, ex)
    restored_output = similar(output)
    SolpropJulia.adams_observe!(restored_output, copied_dense, 0.125, ex)
    @test isequal(restored_output, output)
    SolpropJulia.adams_accept!(history, 1, work, ex)
    @test_throws ArgumentError SolpropJulia.adams_retain_dense!(
        dense, work, state, work.candidate, history, stencil, ex,
    )
    wrong_history = SolpropJulia.AdamsHistory(ex, state, 2; compensated = true)
    wrong_dense = SolpropJulia.AdamsDense(ex, state, 2; compensated = true)
    @test_throws ArgumentError SolpropJulia.adams_copy_history!(wrong_history, history)
    @test_throws ArgumentError SolpropJulia.adams_copy_dense!(wrong_dense, dense)
end
