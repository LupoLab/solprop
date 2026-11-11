# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

mutable struct SessionPrecisionSource{T, E}
    execution::E
    rate::Complex{T}
    forcing::Complex{T}
    calls::Int
    fail_at::Int
end

function (source::SessionPrecisionSource)(output, field, position)
    source.calls += 1
    source.calls == source.fail_at && error("controlled precision source failure")
    rate, forcing = source.rate, source.forcing
    SP.each!(source.execution, size(field)...) do row, column
        output[row, column] = rate * field[row, column] + forcing
    end
    return nothing
end

function session_precision_control(; rtol_floor = 1.0e-12)
    return SP.PIControl(;
        safety = 0.8, max_factor = 2.0, min_factor = 0.2,
        beta1 = 0.3, beta2 = -0.1, lag = 0, reject_free = false,
        estimator_order = 1, start_growth = 100.0, dz_min = rtol_floor,
        dz_max = 1 / 16, auto_init = false,
    )
end

function session_precision_fixture(;
        low = false, tasks = 1, replay = true, capacity = 16, fail_at = 0,
        arithmetic = SP.SourceArithmetic(Float32; state_exponent = 3, field_exponent = 1),
        initial = nothing, rtol = 0.01, floor = 1.0e-12, proposal = 1 / 64,
        planes = [1 / 8], dense = false,
    )
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    T = SP.source_type(arithmetic)
    field = initial === nothing ? Complex{T}[
            complex(row / 8, column / 16) for row in 1:4, column in 1:2
        ] : initial
    source = SessionPrecisionSource(
        execution, Complex{T}(-0.25, 0.125), zero(Complex{T}), 0, fail_at,
    )
    flow = SP.LinearFlow(
        execution, SP.DiagonalOperator(execution, fill(-0.125 + 0.25im, size(field))),
        field,
    )
    norm = SP.MarchNorm(
        execution, [true, false, true, true], fill(true, 4), ones(4), rtol, 0.0,
    )
    schedule = SP.MarchSchedule(;
        planes, breakpoints = [0.0, 1 / 4], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0,
        replay_steps = replay ? fill(1 / 64, 8) : Float64[],
        replay_edges = replay ? collect(0:8) ./ 64 : Float64[],
    )
    tableau = dense ? rk_dense_tableau() : SP.EmbeddedTableau(
            [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
        )
    session = low ? SP.LowStorageSession(
            execution, field, 0.0, SP.RK4IPTableau(), norm, schedule, source, flow;
            arithmetic, capacity,
        ) : SP.RungeKuttaSession(
            execution, field, 0.0, proposal, tableau,
            session_precision_control(; rtol_floor = floor), norm, schedule, source, flow;
            arithmetic, capacity, dense_weights = dense ? rk_dense_weights() : nothing,
        )
    return session, field
end

function session_precision_collect(session; work_limit = 256, deadline = 60.0)
    trace = Tuple{Float64, Float64, Float64, Bool}[]
    events = Tuple{SP.SessionEvent, Float64}[]
    while !session.state.finished
        report = SP.advance!(session; work_limit, deadline)
        push!(events, (report.kind, report.z))
        for index in 1:session.attempts.count
            buffer = session.attempts
            push!(
                trace, (
                    buffer.z[index], buffer.dz[index],
                    buffer.norm[index], buffer.accepted[index],
                ),
            )
        end
        SP.acknowledge!(session, report.generation)
    end
    return trace, events
end

function session_precision_one!(session)
    report = SP.advance!(session; work_limit = 1, deadline = 60.0)
    SP.acknowledge!(session, report.generation)
    return nothing
end

function session_precision_allocations(session)
    session_precision_one!(session)
    return @allocated session_precision_one!(session)
end

function session_precision_bound(initial, length, steps)
    # Each two-stage Heun/four-stage RK4 trial contains fewer than 256 real
    # products/sums along a scalar path. 512 units also cover <=8u error per
    # qualified wide complex exponential followed by one binary32 narrowing.
    # The absolute stability-polynomial majorant bounds cancellation and transport.
    unit = Float64(eps(Float32)) / 2
    gamma = 512steps * unit / (1 - 512steps * unit)
    return gamma * maximum(abs, initial) *
        exp(length * (abs(-0.125 + 0.25im) + abs(-0.25 + 0.125im)))
end

@testset "Binary32 sessions preserve scaled state and exact work ownership" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))), low in (false, true)
        session, initial = session_precision_fixture(; low, tasks)
        saved = copy(initial)
        @test session.arithmetic.state_exponent == 3
        @test session.arithmetic.field_exponent == 1
        @test eltype(session.field) === ComplexF32
        @test eltype(session.work.source) === ComplexF32
        @test eltype(session.work.norm_parts) === Float32
        @test !Base.mightalias(session.field, initial)
        @test all(iszero, session.field[2, :])
        trace, events = session_precision_collect(session)
        @test last(events)[1] == SP.SESSION_COMPLETED
        @test session.state.accepted == 8 && session.state.rejected == 0
        @test all(row -> row[4], trace) && length(trace) == 8
        @test initial == saved
        h, gain, linear = big(1) / 64, Complex{BigFloat}(-0.25, 0.125),
            Complex{BigFloat}(-0.125, 0.25)
        order = low ? 4 : 2
        factor = exp(linear * h) * sum((gain * h)^k / factorial(k) for k in 0:order)
        expected = Complex{BigFloat}.(saved) .* factor^8
        expected[2, :] .= 0
        budget = session_precision_bound(saved, 1 / 8, 8)
        @test budget > 0
        @test maximum(abs, Complex{BigFloat}.(session.field) - expected) <= budget
        @test session.source.calls == (low ? 32 : 16)
        if low
            @test session.counts.attempted_source_evaluations == 32
            @test session.counts.committed_source_evaluations == 32
            @test session.counts.norm_evaluations == 8
        end
        warm, _ = session_precision_fixture(; low, tasks)
        session_precision_one!(warm) # launch
        @test session_precision_allocations(warm) == 0
        @test session_precision_allocations(warm) == 0
        SP.close_session!(warm)
    end
end

@testset "Binary32 delivery boundaries, rejection and poisoning retain their policy" begin
    for low in (false, true), replay in (false, true)
        low && !replay && continue
        whole, _ = session_precision_fixture(; low, replay)
        chunked, _ = session_precision_fixture(; low, replay, capacity = 1)
        rows, events = session_precision_collect(whole)
        sliced, yielded = session_precision_collect(chunked; work_limit = 1, deadline = 0.0)
        @test rows == sliced
        @test whole.field == chunked.field
        @test filter(event -> event[1] != SP.SESSION_PROGRESS, yielded) == events
        @test length(chunked.attempts.z) == 1
        @test all(isfinite, whole.field)
    end
    rejected, _ = session_precision_fixture(;
        replay = false, rtol = 1.0e-4, proposal = 1 / 16,
    )
    rows, _ = session_precision_collect(rejected)
    @test rejected.state.rejected > 0
    @test any(row -> !row[4], rows) && rejected.state.accepted > 0
    floor, _ = session_precision_fixture(;
        replay = false, rtol = 1.0e-4, proposal = 1 / 16, floor = 1 / 20,
    )
    session_precision_one!(floor)
    original = copy(floor.field)
    report = SP.advance!(floor; deadline = 60.0)
    @test report.kind == SP.SESSION_FAILED && !report.poisoned
    @test floor.field == original && floor.state.accepted == 0
    @test floor.state.rejected == 1 && floor.attempts.count == 1
    for low in (false, true), failure in (1, 2)
        session, _ = session_precision_fixture(; low, fail_at = failure)
        session_precision_one!(session)
        original = copy(session.field)
        report = SP.advance!(session; deadline = 60.0)
        @test report.kind == SP.SESSION_FAILED
        @test report.poisoned == low
        @test session.source.calls == failure
        @test session.state.accepted == 0
        if low
            @test session.counts.attempted_source_evaluations == failure
            @test session.counts.committed_source_evaluations == 0
        else
            @test session.field == original
        end
    end
end

function session_precision_dense_operator(execution, shape, kind, rate)
    kind == :constant && return SP.DiagonalOperator(execution, fill(rate, shape))
    # Phi(z)=rate*z: explicit binary64 coefficients on [0,1].
    table = SP.ChebyshevTable(
        execution, [0.0, 1.0], reshape(ComplexF64[rate / 2, rate / 2], 1, 2, 1),
    )
    if kind == :represented
        coefficients = repeat(table.coefficients; outer = (prod(shape), 1, 1))
        return SP.RepresentedOperator(
            execution, SP.ChebyshevTable(execution, [0.0, 1.0], coefficients), shape...,
        )
    end
    weights = SP.ChebyshevTable(execution, [0.0, 1.0], reshape([0.5, 0.5], 1, 2, 1))
    return SP.ExpansionOperator(execution, fill(rate, shape..., 1), weights)
end

function session_precision_dense_allocations(work, initial, endpoint, stages, active, ex)
    SP.retain_runge_kutta_observation!(
        work, initial, endpoint, stages, 0.0, 0.125, 0.125, ex,
    )
    SP.observe_runge_kutta!(work, 0.03125, active, ex)
    return (
        @allocated(
            SP.retain_runge_kutta_observation!(
                work, initial, endpoint, stages, 0.0, 0.125, 0.125, ex,
            )
        ),
        @allocated(SP.observe_runge_kutta!(work, 0.03125, active, ex)),
    )
end

@testset "Binary32 dense fields use narrow state and wide represented factors" begin
    initial = ComplexF32[1 + 0.5im 0.5 - 0.25im; 0.25 -0.125im]
    tableau, weights = rk_dense_tableau(), rk_dense_weights()
    stages = ComplexF32[
        complex(row + column / 4, row - column / 2) / stage
            for row in 1:2, column in 1:2, stage in 1:7
    ]
    endpoint = initial .+ 0.125f0 .* stages[:, :, 1]
    arithmetic = SP.SourceArithmetic(Float32; state_exponent = 9)
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        ex = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        for kind in (:constant, :represented, :expansion),
                rate in (-0.25 + 0.5im, -16.0 + 0.5im)
            operator = session_precision_dense_operator(ex, size(initial), kind, rate)
            work = SP.RungeKuttaObservationWork(
                ex, initial, tableau, weights, operator; arithmetic,
            )
            @test work.exponent === nothing && eltype(work.factor) === ComplexF32
            @test eltype(work.coefficients) === eltype(work.endpoint) === ComplexF32
            saved = copy(stages)
            SP.retain_runge_kutta_observation!(
                work, initial, endpoint, stages, 0.0, 0.125, 0.125, ex,
            )
            active = [true, false]
            @test SP.observe_runge_kutta!(work, 0.0, active, ex)[1, :] == initial[1, :]
            @test SP.observe_runge_kutta!(work, 0.125, active, ex)[1, :] == endpoint[1, :]
            @test all(iszero, work.field[2, :])
            for position in (0.015625, 0.046875, prevfloat(0.125))
                SP.observe_runge_kutta!(work, position, active, ex)
                t, h = BigFloat(position) / big(0.125), big(0.125)
                rest = 1 - t
                for column in 1:2
                    slope = Complex{BigFloat}.(stages[1, column, :])
                    y = Complex{BigFloat}(initial[1, column])
                    span = h * sum(BigFloat.(tableau.solution_weights) .* slope)
                    first = h * slope[1] - span
                    last = span - h * slope[end] - first
                    dense = h * sum(BigFloat.(weights) .* slope)
                    expected = abs(real(rate * 0.125)) > 1 ? y :
                        y + t * (span + rest * (first + t * (last + rest * dense)))
                    expected *= exp(Complex{BigFloat}(rate) * BigFloat(position))
                    # Absolute contributions precede every signed stage contraction.
                    # 128u covers <=48 scalar operations, coefficient/step/coordinate
                    # narrowing and the qualified wide phase/factor rounding here.
                    majorant = abs(y) + h * sum(
                        (
                            abs.(BigFloat.(tableau.solution_weights)) .+
                                abs.(BigFloat.(weights)) .+ 4
                        ) .* abs.(slope)
                    )
                    unit = BigFloat(eps(Float32)) / 2
                    budget = 128unit / (1 - 128unit) * majorant * exp(abs(rate) * position)
                    @test budget > 0
                    @test abs(Complex{BigFloat}(work.field[1, column]) - expected) <= budget
                end
            end
            @test stages == saved
            allocations = session_precision_dense_allocations(
                work, initial, endpoint, stages, active, ex,
            )
            @test allocations == (0, 0)
            bytes = SP.runge_kutta_observation_memory(work)
            @test bytes.exponent == 0 && bytes.factor == sizeof(initial)
            expected_bytes = 8sizeof(initial) + length(initial) +
                sizeof(work.weights) + 2sizeof(Int)
            @test sum(bytes) == expected_bytes
            SP.retain_runge_kutta_observation!(
                work, initial, endpoint, stages, 0.0, 0.125,
                nextfloat(nextfloat(0.125)), ex,
            )
            SP.observe_runge_kutta!(work, nextfloat(0.125), active, ex)
            @test work.field[1, :] == endpoint[1, :]
            before = copy(work.coefficients)
            @test_throws ArgumentError SP.retain_runge_kutta_observation!(
                work, initial, endpoint, stages, 0.0, 1.0e-100, 1.0e-100, ex,
            )
            @test work.ready && work.coefficients == before
        end
    end
end

@testset "Binary32 RK dense sessions retain committed and observed coordinates" begin
    session, _ = session_precision_fixture(;
        replay = false, dense = true, planes = [1 / 256, 1 / 128, 1 / 64, 1 / 8],
        proposal = 1 / 16,
    )
    session_precision_one!(session)
    reports = 0
    while !session.state.finished
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        @test report.kind != SP.SESSION_FAILED
        if report.kind == SP.SESSION_PLANE
            reports += 1
            @test SP._march_observed_position(session) <= session.state.z
            @test eltype(SP._march_observed_field(session)) === ComplexF32
        end
        SP.acknowledge!(session, report.generation)
    end
    @test reports == 4
    @test session.state.accepted < 4
end

function session_precision_material(execution, arithmetic)
    grid = SP.ModalSourceGrid(
        execution; kind = :envelope, n_time = 4, dt = 1.0,
        crop_index = collect(0:3), omega = ones(4), active = fill(true, 4),
        omega_window = ones(4), source_window = ones(4),
        source_window_derivative = zeros(4), arithmetic,
    )
    map = SP.ModalSourceMap(
        execution; synthesis = ones(1, 1), projection = ones(1, 1), arithmetic,
    )
    kerr = SP.ModalKerr(
        SP.KerrCoefficients(0.25, :analytic; self_factor = 1.0, conjugate_factor = 0.0),
        SP.SourceProfile(1.0); arithmetic,
    )
    return SP.ModalSourceData(
        grid, map, SP.SourcePrefactor(execution, fill(-1.0im, 4, 1); arithmetic); kerr,
    )
end

function session_precision_material_session(arithmetic; low = false, tasks = 1)
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    T = SP.source_type(arithmetic)
    initial = zeros(Complex{T}, 4, 1)
    initial[1] = T(ldexp(2.0, -arithmetic.state_exponent))
    source = SP.ModalSource(
        execution, session_precision_material(execution, arithmetic); node_chunk = 1,
    )
    flow = SP.LinearFlow(
        execution, SP.DiagonalOperator(execution, zeros(ComplexF64, 4, 1)), initial,
    )
    norm = SP.MarchNorm(execution, fill(true, 4), fill(true, 4), ones(4), 0.01, 0.0)
    schedule = SP.MarchSchedule(;
        planes = [1 / 8], breakpoints = [0.0, 1 / 4], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0, replay_steps = fill(1 / 64, 8),
        replay_edges = collect(0:8) ./ 64,
    )
    session = low ? SP.LowStorageSession(
            execution, initial, 0.0, SP.RK4IPTableau(), norm, schedule, source, flow;
            arithmetic,
        ) : SP.RungeKuttaSession(
            execution, initial, 0.0, 1 / 64,
            SP.EmbeddedTableau([0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5]),
            session_precision_control(), norm, schedule, source, flow; arithmetic,
        )
    return session, initial
end

function session_precision_cubic_reference(low)
    h, value = BigFloat(1 / 64), Complex{BigFloat}(2)
    source = value -> -im * abs2(value) * value / 64
    for _ in 1:8
        first = source(value)
        if low
            second = source(value + (h / 2) * first)
            third = source(value + (h / 2) * second)
            last = source(value + h * third)
            value += (h / 6) * (first + 2second + 2third + last)
        else
            value += (h / 2) * (first + source(value + h * first))
        end
    end
    return value
end

@testset "Binary32 sessions compose the prepared physical Kerr source" begin
    for low in (false, true), tasks in unique((1, min(2, Threads.nthreads(:default))))
        reference = session_precision_cubic_reference(low)
        wide, _ = session_precision_material_session(
            SP.SourceArithmetic(Float64); low, tasks,
        )
        session_precision_collect(wide)
        # One occupied DC bin has E=A/4, so the actual modal cubic is
        # N(A)=-i*|A|²*A/64. On |A|<=3, its real-vector Jacobian norm is <=27/64.
        # Each path contains <2048 finite transform/cubic/stage operations, with
        # the qualified <=8u FFT error model. Eight steps amplify local absolute
        # errors by (1+h*27/64)^8; no cancellation-reduced result sets the scale.
        for (state_exponent, field_exponent) in ((3, 1), (20, 18))
            arithmetic = SP.SourceArithmetic(Float32; state_exponent, field_exponent)
            session, initial = session_precision_material_session(arithmetic; low, tasks)
            saved = copy(initial)
            session_precision_one!(session)
            @test session_precision_allocations(session) == 0
            while !session.state.finished
                session_precision_one!(session)
                physical = ComplexF64.(session.field) .* ldexp(1.0, state_exponent)
                @test maximum(abs, physical) <= 3
            end
            physical = ComplexF64.(session.field) .* ldexp(1.0, state_exponent)
            unit = Float64(eps(Float32)) / 2
            budget = 8 * (2048unit / (1 - 2048unit)) * 3 * (1 + (1 / 64) * 27 / 64)^8
            @test budget > 0
            @test abs(Complex{BigFloat}(physical[1]) - reference) <= budget
            @test maximum(abs, physical - wide.field) <= budget
            @test all(iszero, physical[2:end])
            @test initial == saved
            @test session.source.calls == (low ? 32 : 16)
        end
    end
end

@testset "Session precision admission preserves caller and prepared storage" begin
    arithmetic = SP.SourceArithmetic(Float32; state_exponent = 3, field_exponent = 1)
    session, initial = session_precision_material_session(arithmetic)
    ex, norm, schedule, source, flow = session.execution, session.norm, session.schedule,
        session.source, session.flow
    saved = copy(initial)
    wrong_flow = SP.LinearFlow(ex, flow.operator, ComplexF64.(initial))
    changed = SP.SourceArithmetic(Float32; state_exponent = 4, field_exponent = 1)
    for kind in (:rk, :low)
        construct = (policy, prepared_flow) -> kind == :rk ? SP.RungeKuttaSession(
                ex, initial, 0.0, 1 / 64, session.tableau, session_precision_control(),
                norm, schedule, source, prepared_flow; arithmetic = policy,
            ) : SP.LowStorageSession(
                ex, initial, 0.0, SP.RK4IPTableau(), norm, schedule, source, prepared_flow;
                arithmetic = policy,
            )
        @test_throws ArgumentError construct(SP.SourceArithmetic(Float64), flow)
        @test_throws ArgumentError construct(changed, flow)
        @test_throws ArgumentError construct(arithmetic, wrong_flow)
        @test initial == saved && source.calls == 0 && !source.ready
    end
    for step in (1.0e-100, 1.0e100)
        schedule = SP.MarchSchedule(;
            planes = [step], breakpoints = [0.0, step], hard_kinks = Float64[],
            stretch = 1.0, step_planes = 0,
            replay_steps = [step], replay_edges = [0.0, step],
        )
        low = SP.LowStorageSession(
            ex, initial, 0.0, SP.RK4IPTableau(), norm, schedule, source, flow; arithmetic,
        )
        session_precision_one!(low)
        before = copy(low.field)
        report = SP.advance!(low; deadline = 60.0)
        @test report.kind == SP.SESSION_FAILED && !report.poisoned
        @test low.field == before && low.source.calls == 0
        @test low.counts.attempted_source_evaluations == 0
    end
end

@testset "Binary32 dense preparation and pre-mask factor failures stay explicit" begin
    execution = SP.KernelExecution(
        SP.KernelAbstractions.CPU(), min(2, Threads.nthreads(:default)), true,
    )
    initial = ones(ComplexF32, 2, 2)
    stages = zeros(ComplexF32, 2, 2, 7)
    arithmetic = SP.SourceArithmetic(Float32)
    operator = SP.DiagonalOperator(execution, fill(2000.0 + 0.0im, 2, 2))
    weights = rk_dense_weights()
    @test_throws ArgumentError SP.RungeKuttaObservationWork(
        execution, initial, rk_dense_tableau(), weights, operator,
    )
    tiny = copy(weights)
    tiny[end] = 1.0e-100
    @test_throws ArgumentError SP.RungeKuttaObservationWork(
        execution, initial, rk_dense_tableau(), tiny, operator; arithmetic,
    )
    work = SP.RungeKuttaObservationWork(
        execution, initial, rk_dense_tableau(), weights, operator; arithmetic,
    )
    SP.retain_runge_kutta_observation!(
        work, initial, initial, stages, 0.0, 0.125, 0.125, execution,
    )
    @test all(work.stiff)
    # The full wide phase is finite, but exp(125) cannot be stored in binary32.
    # Inactive rows cannot conceal a failed factor before the exact mask is applied.
    before = copy(work.field)
    @test_throws SP.NonfiniteSourceError SP.observe_runge_kutta!(
        work, 0.0625, fill(false, 2), execution,
    )
    @test work.field == before && work.ready
    @test SP.observe_runge_kutta!(work, 0.125, fill(true, 2), execution) == initial
    # Fault injection proves that an invalid phase is recorded without asking a
    # worker to call complex exp on a nonfinite imaginary argument.
    operator.rate[1, 1] = ComplexF64(0, Inf)
    before = copy(work.field)
    @test_throws SP.NonfiniteSourceError SP.observe_runge_kutta!(
        work, 0.03125, fill(true, 2), execution,
    )
    @test work.field == before
end
