# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Average the initial and predicted endpoint slopes: the explicit trapezoidal rule.
# The embedded difference compares that average with the initial Euler slope.
function heun_tableau()
    return SolpropJulia.EmbeddedTableau(
        [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5]
    )
end

function rk_polynomial_callbacks(execution, linear, quadratic)
    flow! = function (output, input, origin, destination)
        factor = exp(linear * (destination - origin))
        SolpropJulia.each!(execution, size(input)...) do row, column
            output[row, column] = factor * input[row, column]
        end
        return nothing
    end
    source! = function (output, input, position)
        SolpropJulia.each!(execution, size(input)...) do row, column
            output[row, column] = quadratic * input[row, column]^2
        end
        return nothing
    end
    return flow!, source!
end

function short_rk_solution(initial, tableau, count, execution)
    state = copy(initial)
    work = SolpropJulia.RungeKuttaWork(state, tableau)
    flow!, source! = rk_polynomial_callbacks(execution, -0.3 + 0.5im, 0.2 + 0.05im)
    mask = fill(true, size(state, 1))
    dz = 0.4 / count
    for step in 1:count
        calls = SolpropJulia.rk_trial!(
            work, state, (step - 1) * dz, dz, tableau, source!, flow!, execution
        )
        @test calls == length(tableau.nodes)
        SolpropJulia.rk_commit!(state, work, mask, execution)
    end
    return state
end

@testset "RK owns translated coefficients and uses every explicit stage" begin
    matrix = [0.0 0.0; 1.0 0.0]
    tableau = SolpropJulia.EmbeddedTableau([0.0, 1.0], matrix, [0.5, 0.5], [-0.5, 0.5])
    matrix[2, 1] = 7.0
    @test tableau.matrix[2][1] == 1.0
    @test isbitstype(typeof(tableau))
    @test_throws ArgumentError SolpropJulia.EmbeddedTableau([], zeros(0, 0), [], [])
    @test_throws ArgumentError SolpropJulia.EmbeddedTableau(
        [0.0], zeros(2, 2), [1.0], [0.0]
    )
    @test_throws ArgumentError SolpropJulia.EmbeddedTableau([0.0], ones(1, 1), [1.0], [0.0])
    @test_throws ArgumentError SolpropJulia.EmbeddedTableau(
        [1.0], zeros(1, 1), [1.0], [0.0]
    )
    @test_throws ArgumentError SolpropJulia.EmbeddedTableau(
        [0.0], zeros(1, 1), [Inf], [0.0]
    )

    # These classical RK4 weights satisfy the fourth-order conditions. Zero error
    # weights here exercise a four-stage flow; no error estimator is claimed for it.
    fourth = SolpropJulia.EmbeddedTableau(
        [0.0, 0.5, 0.5, 1.0],
        [0.0 0.0 0.0 0.0; 0.5 0.0 0.0 0.0; 0.0 0.5 0.0 0.0; 0.0 0.0 1.0 0.0],
        [1 / 6, 1 / 3, 1 / 3, 1 / 6], zeros(4),
    )
    initial = ComplexF64[1 + 0.2im 0.4 - 0.3im; -0.2 + 0.1im 0.7 + 0.1im]
    linear, quadratic, distance = -0.3 + 0.5im, 0.2 + 0.05im, 0.4
    # Separation of y' = linear*y + quadratic*y^2 gives this independent solution.
    exact = linear .* initial .* exp(linear * distance) ./
        (linear .+ quadratic .* initial .* (1 - exp(linear * distance)))
    for tasks in unique((1, Threads.nthreads(:default)))
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        for (method, order) in ((tableau, 2), (fourth, 4))
            errors = [
                maximum(abs, short_rk_solution(initial, method, count, execution) - exact)
                    for count in (4, 8, 16)
            ]
            # A factor-two refinement decreases a convergent order-p global error
            # by 2^p. The 12% interval allows the leading next-order term at h=0.1.
            for ratio in errors[1:2] ./ errors[2:3]
                @test 0.88 * 2^order < ratio < 1.12 * 2^order
            end
        end
    end
end

@testset "RK trials preserve the accepted state until an explicit commit" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    tableau = heun_tableau()
    state = fill(1.0 + 0.5im, 4, 2)
    original = copy(state)
    work = SolpropJulia.RungeKuttaWork(state, tableau)
    flow!, source! = rk_polynomial_callbacks(execution, -0.5im, 0.0im)
    calls = SolpropJulia.rk_trial!(
        work, state, 0.0, 0.2, tableau, source!, flow!, execution
    )
    @test calls == 2
    @test state == original
    # Only one end-point exponential and multiplication form the exact linear flow.
    @test isapprox(work.candidate, original * exp(-0.1im); rtol = 8eps(), atol = 0.0)
    @test all(iszero, work.defect)
    @test_throws ArgumentError SolpropJulia.rk_trial!(
        work, state, 0.0, 0.0, tableau, source!, flow!, execution
    )
    @test_throws ArgumentError SolpropJulia.rk_trial!(
        work, state, Inf, 0.1, tableau, source!, flow!, execution
    )
    @test_throws ArgumentError SolpropJulia.rk_trial!(
        work, state, 1.0, eps() / 4, tableau, source!, flow!, execution
    )

    attempts = Ref(0)
    failing! = function (output, input, position)
        attempts[] += 1
        attempts[] == 2 && throw(DomainError(position, "injected source failure"))
        return source!(output, input, position)
    end
    @test_throws DomainError SolpropJulia.rk_trial!(
        work, state, 0.0, 0.2, tableau, failing!, flow!, execution
    )
    @test attempts[] == 2
    @test state == original
    @test_throws ArgumentError SolpropJulia.rk_trial!(
        work, state, 0.0, 0.2, tableau, (args...) -> :failed, flow!, execution
    )
    @test state == original
    SolpropJulia.rk_trial!(work, state, 0.0, 0.2, tableau, source!, flow!, execution)
    SolpropJulia.rk_commit!(state, work, [true, false, true, false], execution)
    @test all(iszero, state[[2, 4], :])
    @test state[[1, 3], :] == work.candidate[[1, 3], :]
end

@testset "RK follows varying linear flow at nonzero positions" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    phi(z) = 0.2z + 0.3im * z^2
    z0, dz = 0.7, 0.2
    state = fill((1 + 0.1im) * exp(phi(z0)), 3, 2)
    tableau = heun_tableau()
    work = SolpropJulia.RungeKuttaWork(state, tableau)
    flow! = function (output, input, origin, destination)
        factor = exp(phi(destination) - phi(origin))
        SolpropJulia.each!(execution, size(input)...) do row, column
            output[row, column] = factor * input[row, column]
        end
        return nothing
    end
    source! = function (output, input, position)
        fill!(output, 2position * exp(phi(position)))
        return nothing
    end
    SolpropJulia.rk_trial!(work, state, z0, dz, tableau, source!, flow!, execution)
    # In the integrating factor v=exp(-phi(z))*y, v'=2z. Heun integrates that
    # linear source exactly. The budget covers two flows and the endpoint source.
    exact = exp(phi(z0 + dz)) * (1 + 0.1im + (z0 + dz)^2 - z0^2)
    @test all(value -> isapprox(value, exact; rtol = 64eps(), atol = 0.0), work.candidate)
    defect = exp(phi(z0 + dz)) * dz^2
    @test all(value -> isapprox(value, defect; rtol = 64eps(), atol = 0.0), work.defect)
end

function rk_warmed_allocations!(
        work, state, tableau, source!::F, flow!::L, execution,
    ) where {F, L}
    mask, weights = fill(true, size(state, 1)), ones(size(state, 1))
    for _ in 1:3
        SolpropJulia.rk_trial!(work, state, 0.0, 0.01, tableau, source!, flow!, execution)
        SolpropJulia.rk_error_norm!(work, state, mask, weights, 1.0e-4, 1.0e-12, execution)
        SolpropJulia.rk_commit!(state, work, mask, execution)
    end
    return @allocated begin
        SolpropJulia.rk_trial!(work, state, 0.0, 0.01, tableau, source!, flow!, execution)
        SolpropJulia.rk_error_norm!(work, state, mask, weights, 1.0e-4, 1.0e-12, execution)
        SolpropJulia.rk_commit!(state, work, mask, execution)
        nothing
    end
end

@testset "Warmed CPU RK trial, norm and commit allocate nothing" begin
    for tasks in unique((1, Threads.nthreads(:default)))
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        state = fill(1 + 0.2im, 32, 3)
        tableau = heun_tableau()
        work = SolpropJulia.RungeKuttaWork(state, tableau)
        flow!, source! = rk_polynomial_callbacks(execution, -0.3 + 0.5im, 0.2 + 0.05im)
        rk_warmed_allocations!(work, state, tableau, source!, flow!, execution)
        @test rk_warmed_allocations!(work, state, tableau, source!, flow!, execution) == 0
    end
end

@testset "RK weak norm is global, weighted and safe at extreme amplitudes" begin
    tableau = heun_tableau()
    for tasks in unique((1, Threads.nthreads(:default)))
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        state = ComplexF64[3 0; 4 0; 1.0e90 1.0e90]
        work = SolpropJulia.RungeKuttaWork(state, tableau)
        work.candidate .= 2state
        work.defect .= ComplexF64[1 0; 2 0; Inf Inf]
        mask, weights = [true, true, false], [1.0, 0.5, 1.0]
        actual = SolpropJulia.rk_error_norm!(
            work, state, mask, weights, 0.1, 0.5, execution
        )
        # Global norm is 10; bin scales are 1.5 and 1.0, regardless of empty columns.
        @test isapprox(actual, hypot(1 / 1.5, 2.0); rtol = 32eps(), atol = 0.0)
        for magnitude in (1.0e-200, 1.0e200)
            state .= magnitude
            work.candidate .= magnitude
            work.defect .= magnitude
            # Four identical in-band entries: both Euclidean norms scale identically.
            actual = SolpropJulia.rk_error_norm!(
                work, state, mask, ones(3), 1.0, 0.0, execution
            )
            @test isapprox(actual, 1.0; rtol = 32eps(), atol = 0.0)
        end
        for value in (Inf, NaN), row in (1, 3)
            state .= 1.0
            work.candidate .= 1.0
            work.candidate[row, 1] = value
            work.defect .= 0.0
            @test isinf(
                SolpropJulia.rk_error_norm!(
                    work, state, mask, weights, 0.1, 0.0, execution
                )
            )
        end
        state .= 0.0
        work.candidate .= 0.0
        work.defect .= 0.0
        actual = SolpropJulia.rk_error_norm!(
            work, state, mask, weights, 0.1, 0.0, execution
        )
        @test actual == 0.0
        work.defect[1, 1] = 1.0
        actual = SolpropJulia.rk_error_norm!(
            work, state, mask, weights, 0.1, 0.0, execution
        )
        @test isinf(actual)
    end
end
