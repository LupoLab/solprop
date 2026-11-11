# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

# The DOPRI5 fixture helpers come from runge_kutta_observation.jl. Columns are
# flattened with the first transverse coordinate fastest, as in spatial_composition.
function rk_spatial_operator(execution, n_columns)
    columns = 1:n_columns
    return SP.SeparableSpatialOperator(
        execution; k_squared = ComplexF64[9 + 0.25im, 4, 1, 0, 9 + 0.25im],
        k_squared_low = ComplexF64[0, 2.0^-50, 0, 0, 0],
        kperp_sq = [0.0, 4.0, 4.0, 16.0, 36.0, 1.0][columns],
        kperp_sq_low = [0.0, 2.0^-51, 3 * 2.0^-51, 0.0, 0.0, 0.0][columns],
        frame = [1.0, 0.125, 0.0, 0.0, 0.5],
        spectral_rate = [0.25, 0.0, 0.125, 0.5, 4.5],
        prefactor_numerator = fill(-1.0im, 5),
        prefactor_clamped = fill(-0.5im, 5),
        active = [true, true, true, false, true], evanescent_rate = 0.75,
        z_span = (0.0, 1.0), kperp_rate = collect(0:(n_columns - 1)) ./ 16,
    )
end

rk_spatial_norm1(value) = abs(real(value)) + abs(imag(value))
rk_spatial_arithmetic(::Type{T}) where {T} = T === Float64 ? SP.SourceArithmetic(T) :
    SP.SourceArithmetic(T; state_exponent = 3, field_exponent = -2)

function rk_spatial_rate_reference(operator, row, column)
    square = Complex{BigFloat}(operator.k_squared[row]) -
        BigFloat(operator.kperp_sq[column]) +
        Complex{BigFloat}(operator.k_squared_low[row]) -
        BigFloat(operator.kperp_sq_low[column])
    assembled = (operator.k_squared[row] - operator.kperp_sq[column]) +
        (operator.k_squared_low[row] - operator.kperp_sq_low[column])
    @test sign(real(square)) == sign(real(assembled))
    root, represented_root = sqrt(square), sqrt(Complex{BigFloat}(assembled))
    # Exact root conditioning: (sqrt(qhat)-sqrt(q))*(sqrt(qhat)+sqrt(q))=qhat-q.
    # The fixture stays on one principal-root branch. The exact zero is separate.
    denominator = abs(represented_root + root)
    conditioned = iszero(denominator) ? big(0) : abs(assembled - square) / denominator
    unit = BigFloat(eps(Float64)) / 2
    # Conditional qualified libm bound: <=8u relative complex-root error. The
    # operation count alone does not establish the vendor square-root accuracy.
    root_error = conditioned + 8unit * abs(represented_root)
    attenuation = real(square) > 0 ? imag(root) :
        min(abs(imag(root)), BigFloat(operator.evanescent_rate))
    if operator.active[row]
        rate = complex(
            -attenuation - BigFloat(operator.kperp_rate[column]) / 2,
            -(real(root) - BigFloat(operator.frame[row])),
        ) - BigFloat(operator.spectral_rate[row]) / 2
        magnitude = rk_spatial_norm1(root) + abs(BigFloat(operator.frame[row])) +
            BigFloat(operator.kperp_rate[column]) / 2 +
            BigFloat(operator.spectral_rate[row]) / 2
        # min(abs(imag(root)),clamp) is 1-Lipschitz. Eight rounded scalar
        # operations also cover assembly of the two rate components and losses.
        error = 2root_error + 8unit / (1 - 8unit) * magnitude
    else
        rate = complex(-BigFloat(operator.spectral_rate[row]) / 2, big(0))
        error = big(0) # This fixture's halving is exact and normal.
    end
    return rate, error
end

function rk_spatial_dense_reference(work, initial, stages, row, column, theta)
    h, t, rest = BigFloat(work.step), BigFloat(theta), 1 - BigFloat(theta)
    y = Complex{BigFloat}(initial[row, column])
    slopes = Complex{BigFloat}.(stages[row, column, :])
    solution, dense = BigFloat.(work.solution_weights), BigFloat.(work.dense_weights)
    r2 = h * sum(solution .* slopes)
    r3 = h * slopes[1] - r2
    r4 = r2 - h * slopes[end] - r3
    r5 = h * sum(dense .* slopes)
    b2 = h * sum(abs.(solution) .* rk_spatial_norm1.(slopes))
    b3 = h * rk_spatial_norm1(slopes[1]) + b2
    b4 = b2 + h * rk_spatial_norm1(slopes[end]) + b3
    b5 = h * sum(abs.(dense) .* rk_spatial_norm1.(slopes))
    rate, rate_error = rk_spatial_rate_reference(work.operator, row, column)
    stiff = abs(real(rate * h)) > 1
    @test abs(abs(real(rate * h)) - 1) > 1 / 16
    @test work.stiff[row, column] == stiff
    polynomial = stiff ? y : y + t * (r2 + rest * (r3 + t * (r4 + rest * r5)))
    majorant = stiff ? rk_spatial_norm1(y) : rk_spatial_norm1(y) +
        t * (b2 + rest * (b3 + t * (b4 + rest * b5)))
    unit = BigFloat(eps(SP.source_type(work.arithmetic))) / 2
    # Each real path has fewer than128 rounded operations, including all seven
    # coefficient narrowings, ordered stage sums and nested polynomial products.
    # h, theta and 1-theta are exactly represented dyadics in both precisions.
    polynomial_error = stiff ? big(0) : 128unit / (1 - 128unit) * majorant
    distance = t * h
    phase = rate * distance
    wide_unit = BigFloat(eps(Float64)) / 2
    phase_error = distance * rate_error +
        4wide_unit / (1 - 4wide_unit) * distance * rk_spatial_norm1(rate)
    factor = exp(phase)
    # Same qualified <=8u wide exp hypothesis as the existing linear-flow tests;
    # the factor2 permits a component one-norm and its two component roundings.
    factor_error = 2abs(factor) *
        (expm1(phase_error) + 8wide_unit * exp(phase_error))
    work.factor === nothing ||
        (factor_error += unit * (rk_spatial_norm1(factor) + factor_error))
    multiplication_error = 8unit / (1 - 8unit) *
        (majorant + polynomial_error) * (rk_spatial_norm1(factor) + factor_error)
    budget = polynomial_error * rk_spatial_norm1(factor) +
        (rk_spatial_norm1(polynomial) + polynomial_error) * factor_error +
        multiplication_error
    return polynomial * factor, budget
end

function rk_spatial_dense_fixture(::Type{T}, tasks, n_columns) where {T}
    execution = rk_dense_execution(tasks)
    operator = rk_spatial_operator(execution, n_columns)
    arithmetic = rk_spatial_arithmetic(T)
    initial = Complex{T}[
        complex(row / 16, (column - 3) / 32) for row in 1:5, column in 1:n_columns
    ]
    endpoint = initial .* T(0.5)
    stages = Complex{T}[
        complex((row - stage) / 32, (column + stage) / 64)
            for row in 1:5, column in 1:n_columns, stage in 1:7
    ]
    work = SP.RungeKuttaObservationWork(
        execution, initial, rk_dense_tableau(), rk_dense_weights(), operator; arithmetic,
    )
    return execution, work, initial, endpoint, stages
end

function rk_spatial_dense_allocations(work, initial, endpoint, stages, active, execution)
    SP.retain_runge_kutta_observation!(
        work, initial, endpoint, stages, 0.25, 0.5, 0.75, execution,
    )
    SP.observe_runge_kutta!(work, 0.5, active, execution)
    retained = @allocated SP.retain_runge_kutta_observation!(
        work, initial, endpoint, stages, 0.25, 0.5, 0.75, execution,
    )
    observed = @allocated SP.observe_runge_kutta!(work, 0.5, active, execution)
    return retained, observed
end

@testset "Spatial RK dense phases preserve the protected separable assembly" begin
    for T in (Float64, Float32), tasks in unique((1, min(2, Threads.nthreads(:default)))),
            n_columns in (2, 6)
        execution, work, initial, endpoint, stages =
            rk_spatial_dense_fixture(T, tasks, n_columns)
        saved = (copy(initial), copy(endpoint), copy(stages))
        operator = work.operator
        retained = map(copy, SP._spatial_operator_tables(operator))
        active = [true, true, false, true, true]
        SP.retain_runge_kutta_observation!(
            work, initial, endpoint, stages, 0.25, 0.5, 0.75, execution,
        )
        for theta in (0.125, 0.5, 0.875)
            field = SP.observe_runge_kutta!(work, 0.25 + 0.5theta, active, execution)
            for column in 1:n_columns, row in 1:5
                expected, budget = rk_spatial_dense_reference(
                    work, initial, stages, row, column, theta,
                )
                @test budget > 0
                if active[row]
                    @test rk_spatial_norm1(
                        Complex{BigFloat}(field[row, column]) -
                            expected
                    ) <= budget
                else
                    @test iszero(field[row, column])
                end
            end
        end
        @test SP.observe_runge_kutta!(work, 0.25, active, execution) == initial .* active
        @test SP.observe_runge_kutta!(work, 0.75, active, execution) == endpoint .* active
        @test work.operator === operator && isempty(work.weights)
        @test work.arithmetic == rk_spatial_arithmetic(T)
        @test eltype(work.field) === eltype(work.coefficients) === Complex{T}
        @test (work.exponent === nothing) == (T === Float32)
        @test (work.factor === nothing) == (T === Float64)
        @test sum(SP.runge_kutta_observation_memory(work)) ==
            8sizeof(initial) + length(initial) + n_columns * sizeof(Int)
        @test rk_spatial_dense_allocations(
            work, initial, endpoint, stages, active, execution,
        ) == (0, 0)
        @test (initial, endpoint, stages) == saved
        @test SP._spatial_operator_tables(operator) == retained
        # Near grazing, residual words determine opposite branch sides. Never
        # round them away by first forming a binary32 spatial coefficient volume.
        @test SP._spatial_square(SP._spatial_operator_tables(operator), 2, 2) == 2.0^-51
        if n_columns == 6
            @test SP._spatial_square(SP._spatial_operator_tables(operator), 2, 3) ==
                -2.0^-51
        end
    end
end

@testset "Spatial RK dense rejects invalid extents/spans and keeps inputs immutable" begin
    for T in (Float64, Float32)
        execution, work, initial, endpoint, stages = rk_spatial_dense_fixture(T, 1, 2)
        @test_throws DimensionMismatch SP.RungeKuttaObservationWork(
            execution, initial[:, 1:1], rk_dense_tableau(), rk_dense_weights(),
            work.operator; arithmetic = work.arithmetic,
        )
        saved = (copy(initial), copy(endpoint), copy(stages))
        before = (copy(work.coefficients), copy(work.endpoint), copy(work.field))
        for (position, step) in ((-0.25, 0.5), (0.75, 0.5))
            @test_throws ArgumentError SP.retain_runge_kutta_observation!(
                work, initial, endpoint, stages, position, step, position + step, execution,
            )
            @test !work.ready
            @test (work.coefficients, work.endpoint, work.field) == before
            @test (initial, endpoint, stages) == saved
        end
        SP.retain_runge_kutta_observation!(
            work, initial, endpoint, stages, 0.25, 0.5, 0.75, execution,
        )
        active = fill(true, 5)
        for position in (prevfloat(0.25), nextfloat(0.75), NaN)
            @test_throws ArgumentError SP.observe_runge_kutta!(
                work, position, active, execution,
            )
            @test all(iszero, work.field)
        end
        @test SP.observe_runge_kutta!(work, 0.75, active, execution) == endpoint
        @test (initial, endpoint, stages) == saved
    end
end

mutable struct RKSpatialDenseSource
    calls::Int
end

function (source::RKSpatialDenseSource)(output, field, position)
    source.calls += 1
    fill!(output, zero(eltype(output)))
    return nothing
end

function rk_spatial_dense_session(::Type{T}, tasks) where {T}
    execution = rk_dense_execution(tasks)
    operator = rk_spatial_operator(execution, 6)
    initial = fill(Complex{T}(0.25, -0.125), 5, 6)
    flow = SP.LinearFlow(execution, operator, initial)
    norm = SP.MarchNorm(execution, fill(true, 5), fill(true, 5), ones(5), 0.01, 0.0)
    control = SP.PIControl(;
        safety = 0.9, max_factor = 2.0, min_factor = 0.2, beta1 = 0.2, beta2 = -0.04,
        lag = 0, reject_free = false, estimator_order = 4, start_growth = 100.0,
        dz_min = 1.0e-12, dz_max = 0.25, auto_init = false,
    )
    schedule = SP.MarchSchedule(;
        planes = [0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5],
        breakpoints = [0.0, 0.5], hard_kinks = Float64[], stretch = 1.0, step_planes = 0,
        replay_steps = Float64[], replay_edges = Float64[],
    )
    return SP.RungeKuttaSession(
        execution, initial, 0.0, 0.25, rk_dense_tableau(), control, norm, schedule,
        RKSpatialDenseSource(0), flow; capacity = 2, dense_weights = rk_dense_weights(),
        arithmetic = rk_spatial_arithmetic(T),
    )
end

function rk_spatial_advance_allocations(session)
    report = SP.advance!(session; work_limit = 1, deadline = 60.0)
    SP.acknowledge!(session, report.generation)
    return @allocated begin
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        SP.acknowledge!(session, report.generation)
    end
end

@testset "Spatial RK dense sessions deliver planes without extra source evaluations" begin
    for T in (Float64, Float32), tasks in unique((1, min(2, Threads.nthreads(:default))))
        session = rk_spatial_dense_session(T, tasks)
        initial = copy(session.field)
        planes = Float64[]
        interiors = 0
        previous_committed = 0.0
        while !session.state.finished
            report = SP.advance!(session; work_limit = 1, deadline = 60.0)
            @test report.kind != SP.SESSION_FAILED
            @test session.state.z >= previous_committed
            previous_committed = session.state.z
            if report.kind == SP.SESSION_PLANE
                push!(planes, report.z)
                interiors += report.z < session.state.z
                field = SP._march_observed_field(session)
                for column in axes(field, 2), row in axes(field, 1)
                    rate, rate_error = rk_spatial_rate_reference(
                        session.flow.operator,
                        row, column
                    )
                    expected =
                        Complex{BigFloat}(initial[row, column]) * exp(rate * report.z)
                    # At most two committed flow multiplications and one dense
                    # multiplication. gamma128 includes their wide factor error,
                    # state rounding and component products; rate assembly is
                    # separately propagated through exp over the actual distance.
                    unit = BigFloat(eps(T)) / 2
                    budget = abs(initial[row, column]) * exp(abs(rate) * report.z) *
                        (128unit / (1 - 128unit) + expm1(rate_error * report.z))
                    @test budget > 0
                    @test abs(Complex{BigFloat}(field[row, column]) - expected) <= budget
                end
            end
            SP.acknowledge!(session, report.generation)
        end
        @test planes == session.schedule.planes
        @test session.state.accepted == 2 && session.state.rejected == 0
        @test interiors == 4
        @test session.source.calls == 14
        SP.close_session!(session)
        warm = rk_spatial_dense_session(T, tasks)
        rk_spatial_advance_allocations(warm)
        SP.close_session!(warm)
        measured = rk_spatial_dense_session(T, tasks)
        @test rk_spatial_advance_allocations(measured) == 0
        @test rk_spatial_advance_allocations(measured) == 0
        @test measured.source.calls == 7 && measured.state.z == 0.25
        SP.close_session!(measured)
    end
end
