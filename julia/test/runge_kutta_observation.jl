# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# The existing translated DOPRI5 pair, Dormand and Prince (1980),
# doi:10.1016/0771-050X(80)90013-3. The dense weights are the shared
# stepper.tableau.DOPRI5_DENSE data (Hairer et al., ODE I, II.6 midpoint form).
# Julia does not select these coefficients in production.
function rk_dense_tableau()
    return SolpropJulia.EmbeddedTableau(
        [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0],
        [
            0 0 0 0 0 0 0;
            1 / 5 0 0 0 0 0 0;
            3 / 40 9 / 40 0 0 0 0 0;
            44 / 45 -56 / 15 32 / 9 0 0 0 0;
            19372 / 6561 -25360 / 2187 64448 / 6561 -212 / 729 0 0 0;
            9017 / 3168 -355 / 33 46732 / 5247 49 / 176 -5103 / 18656 0 0;
            35 / 384 0 500 / 1113 125 / 192 -2187 / 6784 11 / 84 0
        ],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        [
            35 / 384 - 5179 / 57600, 0, 500 / 1113 - 7571 / 16695,
            125 / 192 - 393 / 640, -2187 / 6784 + 92097 / 339200,
            11 / 84 - 187 / 2100, -1 / 40,
        ],
    )
end

rk_dense_weights() = [
    -12715105075 / 11282082432, 0.0, 87487479700 / 32700410799,
    -10690763975 / 1880347072, 701980252875 / 199316789632,
    -1453857185 / 822651844, 69997945 / 29380423,
]

rk_dense_execution(tasks) =
    SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), tasks, true)

function rk_dense_fixture(; tasks = 1, rate = 0.0im, z = 0.0, dz = 0.2)
    execution = rk_dense_execution(tasks)
    initial = ComplexF64[1 + 0.2im 0.5 - 0.3im; 0.4 - 0.1im -0.2 + 0.6im]
    operator = SolpropJulia.DiagonalOperator(execution, fill(ComplexF64(rate), 2, 2))
    tableau = rk_dense_tableau()
    work = SolpropJulia.RungeKuttaObservationWork(
        execution, initial, tableau, rk_dense_weights(), operator,
    )
    stages = Array{ComplexF64}(undef, 2, 2, 7)
    for stage in 1:7, column in 1:2, row in 1:2
        stages[row, column, stage] = complex(row + column / 4, row - column / 2) / stage
    end
    endpoint = initial + dz * sum(
        tableau.solution_weights[i] * stages[:, :, i] for i in 1:7
    )
    endpoint .*= exp(rate * dz)
    SolpropJulia.retain_runge_kutta_observation!(
        work, initial, endpoint, stages, z, dz, z + dz, execution,
    )
    return execution, work, initial, endpoint, stages
end

function rk_dense_allocations(work, initial, endpoint, stages, active, execution)
    SolpropJulia.retain_runge_kutta_observation!(
        work, initial, endpoint, stages, 0.0, 0.2, 0.2, execution,
    )
    SolpropJulia.observe_runge_kutta!(work, 0.08, active, execution)
    retained = @allocated SolpropJulia.retain_runge_kutta_observation!(
        work, initial, endpoint, stages, 0.0, 0.2, 0.2, execution,
    )
    observed = @allocated SolpropJulia.observe_runge_kutta!(work, 0.08, active, execution)
    return retained, observed
end

@testset "RK dense coefficients and rounding against independent wide arithmetic" begin
    for tasks in (1, 2)
        execution, work, initial, endpoint, stages = rk_dense_fixture(; tasks)
        active = fill(true, 2)
        before_initial, before_endpoint, before_stages =
            copy(initial), copy(endpoint), copy(stages)
        for theta in (0.125, 0.4, 0.5, 0.875)
            field = SolpropJulia.observe_runge_kutta!(work, 0.2 * theta, active, execution)
            # Each real/imaginary path has <48 rounded operations: two ordered
            # seven-term scale/sums, h scaling, three coefficient differences,
            # and the nested extension. gamma48 times the absolute expression
            # bounds their error without dividing by a cancellation-prone result.
            u = BigFloat(eps()) / 2
            gamma = 48u / (1 - 48u)
            t = BigFloat(0.2 * theta) / BigFloat(0.2)
            rest = 1 - t
            for column in 1:2, row in 1:2
                slopes = Complex{BigFloat}.(stages[row, column, :])
                weights = BigFloat.(work.solution_weights)
                dense = BigFloat.(work.dense_weights)
                h, y = BigFloat(0.2), Complex{BigFloat}(initial[row, column])
                r2 = h * sum(weights .* slopes)
                r3 = h * slopes[1] - r2
                r4 = r2 - h * slopes[end] - r3
                r5 = h * sum(dense .* slopes)
                expected = y + t * (r2 + rest * (r3 + t * (r4 + rest * r5)))
                norm1(value) = abs(real(value)) + abs(imag(value))
                b2 = h * sum(abs.(weights) .* norm1.(slopes))
                b3 = h * norm1(slopes[1]) + b2
                b4 = b2 + h * norm1(slopes[end]) + b3
                b5 = h * sum(abs.(dense) .* norm1.(slopes))
                magnitude = norm1(y) + t * (b2 + rest * (b3 + t * (b4 + rest * b5)))
                budget = gamma * magnitude
                @test budget > 0
                @test norm1(Complex{BigFloat}(field[row, column]) - expected) <= budget
            end
        end
        @test SolpropJulia.observe_runge_kutta!(work, 0.0, active, execution) == initial
        @test SolpropJulia.observe_runge_kutta!(work, 0.2, active, execution) == endpoint
        @test initial == before_initial && endpoint == before_endpoint &&
            stages == before_stages
        @test rk_dense_allocations(
            work, initial, endpoint, stages, active, execution,
        ) == (0, 0)
        bytes = SolpropJulia.runge_kutta_observation_memory(work)
        @test sum(bytes) == 8sizeof(initial) + length(initial) + 2sizeof(Int)
        @test work.operator.rate !== initial && work.coefficients !== stages
    end
end

@testset "RK dense constant derivative and fifth-order local defect" begin
    for tasks in (1, 2)
        execution = rk_dense_execution(tasks)
        tableau, weights = rk_dense_tableau(), rk_dense_weights()
        initial = fill(1.0 + 0.5im, 2, 2)
        active = fill(true, 2)
        flow = SolpropJulia.LinearFlow(
            execution, SolpropJulia.DiagonalOperator(execution, zeros(ComplexF64, 2, 2)),
            initial,
        )
        constant! = (out, field, z) -> (fill!(out, 2.0 + 0.25im); nothing)
        linear! = (out, field, z) -> (out .= (-1 + 3im) .* field; nothing)
        for source in (constant!, linear!)
            errors = Float64[]
            for h in (0.1, 0.05, 0.025)
                trial = SolpropJulia.RungeKuttaWork(initial, tableau)
                work = SolpropJulia.RungeKuttaObservationWork(
                    execution, initial, tableau, weights, flow.operator,
                )
                SolpropJulia.rk_trial!(
                    trial, initial, 0.0, h, tableau, source, flow, execution,
                )
                SolpropJulia.retain_runge_kutta_observation!(
                    work, initial, trial.candidate, trial.stages, 0.0, h, h, execution,
                )
                value = SolpropJulia.observe_runge_kutta!(work, 0.4h, active, execution)
                if source === constant!
                    # Constant slopes cancel r3/r4/r5 mathematically. Sum and
                    # polynomial cancellation are bounded by 64eps times scale.
                    expected = initial .+ 0.4h * (2.0 + 0.25im)
                    @test maximum(abs, value - expected) <= 64eps() * maximum(abs, expected)
                else
                    exact = initial .* exp((-1 + 3im) * 0.4h)
                    push!(errors, maximum(abs, value - exact))
                end
            end
            if source === linear!
                # The established extension has local defect O(h^5); the finite-h
                # interval allows its next Taylor term at |lambda*h| <=0.317.
                for order in log2.(errors[1:2] ./ errors[2:3])
                    @test 4.7 < order < 5.3
                end
            end
        end
    end
end

@testset "RK dense stiff-bin rule, exact mask and represented operators" begin
    rates = (-5.0, prevfloat(-5.0), nextfloat(-5.0), 5.0, nextfloat(5.0))
    for tasks in (1, 2), rate in rates
        execution, work, initial, endpoint, stages = rk_dense_fixture(; tasks, rate)
        active = Bool[true, false]
        observed = copy(SolpropJulia.observe_runge_kutta!(work, 0.08, active, execution))
        @test all(iszero, observed[2, :])
        @test all(work.stiff .== (abs(rate * 0.2) > 1.0))
        if abs(rate * 0.2) > 1.0
            # One complex exponential/multiplication with real exponent; the
            # 8eps budget covers those primitive roundings on normal inputs.
            @test maximum(abs, observed[1, :] - initial[1, :] * exp(rate * 0.08)) <=
                8eps() * maximum(abs, initial[1, :]) * exp(rate * 0.08)
        else
            @test maximum(abs, observed[1, :] - initial[1, :] * exp(rate * 0.08)) > 0.01
        end
        @test SolpropJulia.observe_runge_kutta!(
            work, 0.2, fill(true, 2), execution,
        ) == endpoint
    end
    for tasks in (1, 2), expanded in (false, true)
        execution = rk_dense_execution(tasks)
        initial = fill(1.0 + 0.25im, 2, 2)
        # Phi(z)=offset+rate*z^2 on [0,1]. z=(x+1)/2 gives
        # z²=3/8*T0+1/2*T1+1/8*T2. The offset must cancel in the flow.
        rate, offset = -0.4 + 1.2im, 3.0 - 4.0im
        table_data = zeros(expanded ? Float64 : ComplexF64, expanded ? 1 : 4, 3, 1)
        table_data[:, 1, 1] .= expanded ? 3 / 8 : offset + rate * 3 / 8
        table_data[:, 2, 1] .= expanded ? 1 / 2 : rate / 2
        table_data[:, 3, 1] .= expanded ? 1 / 8 : rate / 8
        table = SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], table_data)
        operator = expanded ? SolpropJulia.ExpansionOperator(
                execution, fill(rate, 2, 2, 1), table,
            ) : SolpropJulia.RepresentedOperator(execution, table, 2, 2)
        work = SolpropJulia.RungeKuttaObservationWork(
            execution, initial, rk_dense_tableau(), rk_dense_weights(), operator,
        )
        stages = zeros(ComplexF64, 2, 2, 7)
        endpoint = initial .* exp(rate * (0.5^2 - 0.25^2))
        SolpropJulia.retain_runge_kutta_observation!(
            work, initial, endpoint, stages, 0.25, 0.25, 0.5, execution,
        )
        observed = SolpropJulia.observe_runge_kutta!(work, 0.375, fill(true, 2), execution)
        expected = initial .* exp(rate * (0.375^2 - 0.25^2))
        # Two degree2 Clenshaw evaluations, endpoint subtraction and complex
        # propagation: 64eps absolute scale covers this well-conditioned witness.
        @test maximum(abs, observed - expected) <= 64eps() * maximum(abs, initial)
        @test iszero(maximum(abs, stages))
        @test length(work.weights) == (expanded ? 1 : 0)
    end
end

@testset "RK dense invalid intervals, alias refusal and poisoned reuse" begin
    execution, work, initial, endpoint, stages = rk_dense_fixture()
    mask = fill(true, 2)
    for position in (-1.0, nextfloat(0.2), NaN, Inf)
        before = copy(work.field)
        @test_throws ArgumentError SolpropJulia.observe_runge_kutta!(
            work, position, mask, execution,
        )
        @test work.field == before
    end
    @test_throws ArgumentError SolpropJulia.observe_runge_kutta!(
        work, 0.1, trues(2), execution,
    )
    @test_throws DimensionMismatch SolpropJulia.observe_runge_kutta!(
        work, 0.1, fill(true, 3), execution,
    )
    @test_throws ArgumentError SolpropJulia.observe_runge_kutta!(
        work, 0.1, @view(work.stiff[:, 1]), execution,
    )
    for (z, h, destination) in ((0.0, 0.0, 0.2), (0.0, 0.2, 0.3), (Inf, 0.2, Inf))
        @test_throws ArgumentError SolpropJulia.retain_runge_kutta_observation!(
            work, initial, endpoint, stages, z, h, destination, execution,
        )
        @test work.ready
    end
    for (start, finish, slopes) in (
            (work.field, endpoint, stages), (initial, work.endpoint, stages),
            (initial, endpoint, work.coefficients),
        )
        @test_throws Union{ArgumentError, DimensionMismatch} (
            SolpropJulia.retain_runge_kutta_observation!(
                work, start, finish, slopes, 0.0, 0.2, 0.2, execution,
            )
        )
    end
    # An independently wrapped interior pointer must not evade ownership checks.
    GC.@preserve work begin
        foreign = unsafe_wrap(
            Array, pointer(work.coefficients) + sizeof(ComplexF64), (2, 2),
        )
        @test_throws ArgumentError SolpropJulia.retain_runge_kutta_observation!(
            work, foreign, endpoint, stages, 0.0, 0.2, 0.2, execution,
        )
    end
    before = copy(initial)
    poisoned = copy(stages)
    poisoned[1] = NaN
    @test_throws ArgumentError SolpropJulia.retain_runge_kutta_observation!(
        work, initial, endpoint, poisoned, 0.0, 0.2, 0.2, execution,
    )
    @test !work.ready && initial == before
    @test_throws ArgumentError SolpropJulia.observe_runge_kutta!(work, 0.1, mask, execution)
    fill!(work.coefficients, NaN)
    fill!(work.endpoint, NaN)
    SolpropJulia.retain_runge_kutta_observation!(
        work, initial, endpoint, stages, 0.0, 0.2, nextfloat(0.2), execution,
    )
    @test work.ready && all(isfinite, work.coefficients)
    @test SolpropJulia.observe_runge_kutta!(
        work, nextfloat(0.2), mask, execution,
    ) == endpoint
    # A positive canonical gap can contain another representable point.
    destination = nextfloat(nextfloat(0.2))
    SolpropJulia.retain_runge_kutta_observation!(
        work, initial, endpoint, stages, 0.0, 0.2, destination, execution,
    )
    @test SolpropJulia.observe_runge_kutta!(
        work, nextfloat(0.2), mask, execution,
    ) == endpoint
    work.coefficients[1, 1, 1] = Inf
    @test_throws SolpropJulia.NonfiniteSourceError SolpropJulia.observe_runge_kutta!(
        work, 0.1, Bool[false, true], execution,
    )
    @test_throws ArgumentError SolpropJulia.RungeKuttaObservationWork(
        execution, ComplexF32.(initial), rk_dense_tableau(), rk_dense_weights(),
        work.operator,
    )
    @test_throws ArgumentError SolpropJulia.RungeKuttaObservationWork(
        execution, initial, rk_dense_tableau(), [NaN], work.operator,
    )
    @test_throws DimensionMismatch SolpropJulia.RungeKuttaObservationWork(
        execution, initial[:, 1:1], rk_dense_tableau(), rk_dense_weights(), work.operator,
    )
end

@testset "RK dense weights recover the published midpoint constraints" begin
    # Dormand and Prince, Runge-Kutta triples (1986), p.1014,
    # doi:10.1016/0898-1221(86)90025-8, gives these b_i*(sigma).
    # Equation5 uses sigma*h, so the effective midpoint weight is sigma*b_i*.
    # Subtract the endpoint Hermite weight b_i/2+delta_i1/8-delta_i7/8;
    # theta²(1-theta)²=1/16 then determines each quartic d_i exactly.
    s = big(1) // 2
    midpoint = [
        (
            157015080s^4 - 13107642775s^3 + 34969693132s^2 -
                32272833064s + 11282082432
        ) / 11282082432,
        zero(s),
        -100s * (15701508s^3 - 914128567s^2 + 2074956840s - 1323431896) /
            32700410799,
        25s * (94209048s^3 - 1518414297s^2 + 2460397220s - 889289856) /
            5641041216,
        -2187s * (52338360s^3 - 451824525s^2 + 687873124s - 259006536) /
            199316789632,
        11s * (106151040s^3 - 661884105s^2 + 946554244s - 361440756) /
            2467955532,
        s * (1 - s) * (8293050s^2 - 82437520s + 44764047) / 29380423,
    ]
    solution = [35 // 384, 0, 500 // 1113, 125 // 192, -2187 // 6784, 11 // 84, 0]
    derived = [
        16 * (
            s * midpoint[i] - solution[i] / 2 - (i == 1 ? s / 4 : 0) +
                (i == 7 ? s / 4 : 0)
        ) for i in 1:7
    ]
    # These are coefficient-data identities, not cross-backend numerical bit tests.
    @test Float64.(derived) == rk_dense_weights()
end
