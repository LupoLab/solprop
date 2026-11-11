# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import InteractiveUtils: code_llvm
import SolpropJulia as DD

# The existing translated DOPRI5 pair, Dormand and Prince (1980),
# doi:10.1016/0771-050X(80)90013-3. The dense weights are the shared
# stepper.tableau.DOPRI5_DENSE data (Hairer et al., ODE I, II.6 midpoint form).
# Julia does not select these coefficients in production.
function df32dense_tableau()
    return DD.EmbeddedTableau(
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

df32dense_weights() = [
    -12715105075 / 11282082432, 0.0, 87487479700 / 32700410799,
    -10690763975 / 1880347072, 701980252875 / 199316789632,
    -1453857185 / 822651844, 69997945 / 29380423,
]


function df32dense_fixture(;
        tasks = 1, operator = nothing, state = nothing, origin = 0.0,
        step = 0.25, finish = origin + 4step, replay = false,
    )
    execution = DD.KernelExecution(DD.KernelAbstractions.CPU(), tasks, true)
    initial = state === nothing ? ComplexF32[
            complex(row / 8, column / 16) for row in 1:3, column in 1:2
        ] : state
    tableau = df32dense_tableau()
    arithmetic = DD.SourceArithmetic(Float32; state_exponent = 7, field_exponent = 3)
    schedule = DD.MarchSchedule(;
        planes = [finish], breakpoints = [origin, finish], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0,
        replay_steps = replay ? [finish - origin] : Float64[],
        replay_edges = replay ? [origin, finish] : Float64[],
    )
    spec = DD.WideStartSpec(
        origin, tableau, schedule;
        geometry = DD.WideGeometry(origin, finish; step_max = max(step, finish - origin)),
        arithmetic, method = :rk, order_max = 0, source_absent = false,
    )
    represented = operator === nothing ?
        DD.DiagonalOperator(execution, fill(-0.25 + 0.5im, size(initial))) : operator
    flow = DD.DF32LinearFlow(execution, represented, initial, spec)
    stages = Array{ComplexF32}(undef, size(initial)..., 7)
    for stage in 1:7, column in axes(initial, 2), row in axes(initial, 1)
        stages[row, column, stage] = ComplexF32(
            (-1)^stage * (row + column / 4) / (8stage), (row - column / 2) / (16stage),
        )
    end
    return (; execution, initial, tableau, arithmetic, flow, stages, origin, step)
end

function df32dense_work(fixture)
    return DD.RungeKuttaObservationWork(
        fixture.execution, fixture.initial, fixture.tableau, df32dense_weights(),
        fixture.flow; fixture.arithmetic,
    )
end

function df32dense_retain!(
        work, fixture; endpoint = fixture.initial,
        origin = fixture.origin, step = fixture.step, destination = origin + step
    )
    return DD.retain_runge_kutta_observation!(
        work, fixture.initial, endpoint, fixture.stages,
        origin, step, destination, fixture.execution,
    )
end

function df32dense_allocated(work, fixture, active)
    df32dense_retain!(work, fixture)
    DD.observe_runge_kutta!(
        work, fixture.origin + fixture.step / 4, active, fixture.execution,
    )
    retained = @allocated df32dense_retain!(work, fixture)
    observed = @allocated DD.observe_runge_kutta!(
        work, fixture.origin + fixture.step / 4, active, fixture.execution,
    )
    return retained, observed
end

function df32dense_table(execution, shape; scale = 1.0, offset = 0.0)
    # Phi(z)=offset+scale*z^2 on [0,1], independently expanded in T0,T1,T2.
    coefficients = repeat(
        reshape(ComplexF64[offset + 3scale / 8, scale / 2, scale / 8], 1, 3, 1);
        outer = (prod(shape), 1, 1),
    )
    return DD.RepresentedOperator(
        execution, DD.ChebyshevTable(execution, [0.0, 1.0], coefficients), shape...,
    )
end

function df32dense_phase(operator, row, column, origin, destination)
    if operator isa DD.DiagonalOperator
        return Complex{BigFloat}(operator.rate[row, column]) *
            (BigFloat(destination) - BigFloat(origin))
    end
    table = operator isa DD.RepresentedOperator ?
        operator.cumulative : operator.cumulative_weights
    function value(index, position)
        edges, coefficients = table.breakpoints, table.coefficients
        # Fixtures here have one interval. This explicit Chebyshev polynomial
        # recurrence is independent of production's paired Clenshaw difference.
        x = (2BigFloat(position) - BigFloat(edges[1]) - BigFloat(edges[2])) /
            (BigFloat(edges[2]) - BigFloat(edges[1]))
        previous, current = one(x), x
        result = Complex{BigFloat}(coefficients[index, 1, 1])
        for degree in 1:(size(coefficients, 2) - 1)
            result += Complex{BigFloat}(coefficients[index, degree + 1, 1]) * current
            previous, current = current, 2x * current - previous
        end
        return result
    end
    if operator isa DD.RepresentedOperator
        index = row + size(operator.cumulative.coefficients, 1) ÷ operator.n_columns *
            (column - 1)
        return value(index, destination) - value(index, origin)
    end
    return sum(
        Complex{BigFloat}(operator.terms[row, column, term]) *
            (value(term, destination) - value(term, origin))
            for term in axes(operator.terms, 3)
    )
end

function df32dense_reference(work, fixture, position, row, column)
    norm1(x) = abs(real(x)) + abs(imag(x))
    initial = Complex{BigFloat}(fixture.initial[row, column])
    slopes = Complex{BigFloat}.(fixture.stages[row, column, :])
    h = BigFloat(fixture.step)
    b, d = BigFloat.(work.solution_weights), BigFloat.(work.dense_weights)
    r2 = h * sum(b .* slopes)
    r3 = h * slopes[1] - r2
    r4 = r2 - h * slopes[end] - r3
    r5 = h * sum(d .* slopes)
    theta = (BigFloat(position) - BigFloat(fixture.origin)) / h
    rest = 1 - theta
    operator = fixture.flow.operator
    phase = df32dense_phase(operator, row, column, fixture.origin, position)
    full = df32dense_phase(
        operator, row, column, fixture.origin, fixture.origin + fixture.step,
    )
    stiff = abs(real(full)) > 1
    polynomial = stiff ? initial : initial + theta * (
            r2 + rest * (r3 + theta * (r4 + rest * r5))
        )
    expected = polynomial * exp(phase)
    # A seven-stage coefficient path has <64 binary32 roundings, including
    # coefficient/step conversion and nested evaluation. gamma128 leaves room
    # for both parts of a complex multiply and the <5u certified factor error.
    # Absolute term majorants precede cancellation in every stage contraction.
    u = BigFloat(eps(Float32)) / 2
    gamma = 128u / (1 - 128u)
    # These normal finite fixtures have 0<theta<1. Two host coordinate operations
    # and its complement contribute <8u64; the final RN32 adds <2u.
    et = 8BigFloat(eps(Float64)) / 2 + 2u
    es = et
    t, s = abs(theta) + et, abs(rest) + es
    b2 = h * sum(abs.(b) .* norm1.(slopes))
    b3 = h * norm1(slopes[1]) + b2
    b4 = b2 + h * norm1(slopes[end]) + b3
    b5 = h * sum(abs.(d) .* norm1.(slopes))
    majorant = stiff ? norm1(initial) :
        norm1(initial) + t * (b2 + s * (b3 + t * (b4 + s * b5)))
    coordinate = stiff ? zero(h) :
        et * (b2 + s * b3 + 2t * s * b4 + 2t * s^2 * b5) +
        es * (t * b3 + t^2 * b4 + 2t^2 * s * b5)
    bound = exp(real(phase)) * (gamma * majorant + (1 + 5u) * coordinate) +
        256BigFloat(nextfloat(0.0f0))
    return expected, bound
end

@testset "Certified DF32 RK dense polynomial and represented query phases" begin
    setprecision(BigFloat, 256) do
        for tasks in unique((1, min(2, Threads.nthreads(:default))))
            execution = DD.KernelExecution(DD.KernelAbstractions.CPU(), tasks, true)
            shape = (3, 2)
            constant = DD.DiagonalOperator(
                execution, ComplexF64[
                    -0.25 + 0.5im 0.125 - 1.25im; -8 + 0.25im 6 - 0.75im;
                    0.0im 0.5im
                ],
            )
            represented = df32dense_table(
                execution, shape; scale = -0.75 + 2im, offset = 2.0^30,
            )
            weights = DD.ChebyshevTable(
                execution, [0.0, 1.0],
                reshape([3 / 8, 3 / 8, 1 / 2, 1 / 2, 1 / 8, 1 / 8], 2, 3, 1),
            )
            terms = fill(0.0im, shape..., 2)
            terms[:, :, 1] .= 2.0^14 + 2im
            terms[:, :, 2] .= -2.0^14 - 1.5im
            expansion = DD.ExpansionOperator(execution, terms, weights)
            for operator in (constant, represented, expansion)
                fixture = df32dense_fixture(; tasks, operator)
                work = df32dense_work(fixture)
                saved = (copy(fixture.initial), copy(fixture.stages))
                @test df32dense_retain!(work, fixture) === nothing
                @test work.exponent === nothing && eltype(work.factor) === ComplexF32
                @test work.operator.binding === fixture.flow.binding
                active = fill(true, shape[1])
                for position in (1 / 64, 3 / 32, 7 / 32, prevfloat(0.25))
                    result = DD.observe_runge_kutta!(
                        work, position, active, fixture.execution,
                    )
                    for column in 1:shape[2], row in 1:shape[1]
                        expected, bound = df32dense_reference(
                            work, fixture, position, row, column,
                        )
                        @test bound > 0
                        difference = abs(Complex{BigFloat}(result[row, column]) - expected)
                        @test difference <= bound
                    end
                    operation = work.operator.query
                    @test operation.origin == fixture.origin &&
                        operation.destination == position
                    @test operation.step == fixture.step
                    @test operation.certificate.real_error +
                        operation.certificate.imag_error <= 2.0^-27
                end
                @test fixture.initial == saved[1] && fixture.stages == saved[2]
                @test df32dense_allocated(work, fixture, active) == (0, 0)
                bytes = DD.runge_kutta_observation_memory(work)
                @test bytes.exponent == 0
                @test bytes.factor == sizeof(ComplexF32) * prod(shape)
                @test bytes.coefficients == 5sizeof(ComplexF32) * prod(shape)
            end
        end
    end
end

function df32dense_snapshot(work)
    return (;
        coefficients = copy(work.coefficients), field = copy(work.field),
        endpoint = copy(work.endpoint), factors = copy(work.factor),
        stiff = copy(work.stiff),
        failures = copy(work.failures), position = work.position, step = work.step,
        destination = work.destination, ready = work.ready,
        retained = work.operator.retained, query = work.operator.query,
        generation = work.operator.generation,
    )
end

@testset "DF32 dense certification fails before replacing a ready window" begin
    for direction in (-1.0, 1.0), value in (prevfloat(1.0), 1.0, nextfloat(1.0))
        execution = DD.KernelExecution(DD.KernelAbstractions.CPU(), 1, true)
        operator = DD.DiagonalOperator(execution, fill(complex(direction * value), 3, 2))
        fixture = df32dense_fixture(; operator, step = 0.25)
        work = df32dense_work(fixture)
        df32dense_retain!(work, fixture)
        DD.observe_runge_kutta!(work, 0.125, fill(true, 3), execution)
        saved = df32dense_snapshot(work)
        error = try
            df32dense_retain!(work, fixture; step = 1.0)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("stiff-branch certificate", sprint(showerror, error))
        @test isequal(df32dense_snapshot(work), saved)
    end
    for direction in (-1.0, 1.0), value in
            (Float64(prevfloat(1.0f0)), Float64(nextfloat(1.0f0)))
        execution = DD.KernelExecution(DD.KernelAbstractions.CPU(), 1, true)
        operator = DD.DiagonalOperator(execution, fill(complex(direction * value), 3, 2))
        fixture = df32dense_fixture(; operator, step = 1.0, finish = 1.0)
        work = df32dense_work(fixture)
        @test df32dense_retain!(work, fixture) === nothing
        @test all(==(value > 1), work.stiff)
    end
    execution = DD.KernelExecution(DD.KernelAbstractions.CPU(), 1, true)
    # Phi(z)=80*z*(1-z) has zero endpoint difference and an interior maximum20.
    # Endpoint-only phase admission cannot certify the requested midpoint.
    coefficients = repeat(reshape(ComplexF64[10, 0, -10], 1, 3, 1); outer = (6, 1, 1))
    operator = DD.RepresentedOperator(
        execution, DD.ChebyshevTable(execution, [0.0, 1.0], coefficients), 3, 2,
    )
    fixture = df32dense_fixture(; operator, step = 1.0, finish = 1.0)
    work = df32dense_work(fixture)
    df32dense_retain!(work, fixture)
    DD.observe_runge_kutta!(work, 0.0625, fill(true, 3), execution)
    saved = df32dense_snapshot(work)
    @test_throws ArgumentError DD.observe_runge_kutta!(work, 0.5, fill(true, 3), execution)
    @test isequal(df32dense_snapshot(work), saved)
    @test fixture.flow.plans.generation == 0
    @test DD.observe_runge_kutta!(work, 1.0, fill(true, 3), execution) == work.endpoint
end

@testset "Retained DF32 windows survive trial reuse and preserve exact endpoints" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        fixture = df32dense_fixture(; tasks)
        work = df32dense_work(fixture)
        active = fill(true, 3)
        df32dense_retain!(work, fixture)
        expected = copy(DD.observe_runge_kutta!(work, 0.125, active, fixture.execution))
        generation, retained = work.operator.generation, work.operator.retained
        first = DD.prepare_df32_trial!(
            fixture.flow, fixture.tableau, DD.DF32TrialSite(0.0, 0.25, 0.25, 0),
        )
        DD.prepare_df32_trial!(
            fixture.flow, fixture.tableau, DD.DF32TrialSite(0.25, 0.25, 0.5, 0),
        )
        @test_throws ArgumentError first(
            similar(fixture.initial), fixture.initial, 0.0, 0.25,
        )
        actual = DD.observe_runge_kutta!(work, 0.125, active, fixture.execution)
        for column in 1:2, row in 1:3
            _, bound = df32dense_reference(work, fixture, 0.125, row, column)
            difference = abs(Complex{BigFloat}(actual[row, column]) - expected[row, column])
            @test difference <= 2bound
        end
        @test work.operator.generation == generation && work.operator.retained === retained
        @test fixture.flow.plans.generation == 2
        @test work.operator.binding === fixture.flow.binding

        for landing_direction in (-1, 1)
            zeros = ComplexF32[
                complex(-0.0f0, 0.0f0) complex(0.0f0, -0.0f0);
                complex(-0.0f0, -0.0f0) complex(0.0f0, 0.0f0);
                1.0f0 -1.0f0
            ]
            sample = df32dense_fixture(; tasks, state = zeros, origin = 0.3, step = 0.3)
            dense = df32dense_work(sample)
            endpoint = reverse(zeros; dims = 1)
            raw = sample.origin + sample.step
            destination = landing_direction < 0 ? prevfloat(raw, 2) : nextfloat(raw, 2)
            df32dense_retain!(dense, sample; endpoint, destination)
            @test isequal(
                DD.observe_runge_kutta!(dense, sample.origin, active, sample.execution),
                zeros,
            )
            @test isequal(
                DD.observe_runge_kutta!(dense, destination, active, sample.execution),
                endpoint,
            )
            if landing_direction > 0
                @test isequal(
                    DD.observe_runge_kutta!(
                        dense, nextfloat(raw), active, sample.execution,
                    ),
                    endpoint,
                )
            end
            @test dense.operator.query === nothing
        end
        # Supplied h remains distinct from subtraction of the rounded endpoint.
        sample = df32dense_fixture(; tasks, origin = 0.3, step = 0.1)
        dense = df32dense_work(sample)
        df32dense_retain!(dense, sample)
        operation = dense.operator.retained
        @test operation.step == 0.1
        @test operation.destination == 0.3 + 0.1
        @test operation.destination - operation.origin != operation.step
        observed = DD.observe_runge_kutta!(dense, 0.35, active, sample.execution)
        for column in 1:2, row in 1:3
            reference, bound = df32dense_reference(dense, sample, 0.35, row, column)
            @test abs(Complex{BigFloat}(observed[row, column]) - reference) <= bound
        end
        # A forced landing one ulp above a nonzero anchor uses that actual span;
        # its step is normal binary32, with no invented adaptive lower bound.
        h = 2.0^-51
        tiny = df32dense_fixture(; tasks, origin = 0.5, step = h, finish = 0.5 + 4h)
        dense = df32dense_work(tiny)
        df32dense_retain!(dense, tiny)
        point = tiny.origin + h / 2
        @test all(isfinite, DD.observe_runge_kutta!(dense, point, active, tiny.execution))
        @test dense.operator.query.destination == point && dense.operator.query.step == h
    end
end

@testset "DF32 dense ownership and invalid query preservation" begin
    fixture = df32dense_fixture()
    work = df32dense_work(fixture)
    @test_throws ArgumentError DD._df32_dense_retained(work.operator, 0.0, 0.25)
    active = fill(true, 3)
    df32dense_retain!(work, fixture)
    DD.observe_runge_kutta!(work, 0.125, active, fixture.execution)
    saved = df32dense_snapshot(work)
    for point in (-0.1, 0.5, NaN, Inf)
        @test_throws ArgumentError DD.observe_runge_kutta!(
            work, point, active, fixture.execution,
        )
        @test isequal(df32dense_snapshot(work), saved)
    end
    @test_throws DimensionMismatch DD.observe_runge_kutta!(
        work, 0.125, [true], fixture.execution,
    )
    @test isequal(df32dense_snapshot(work), saved)
    # A valid foreign header lies wholly within a retained coefficient allocation.
    # Base.mightalias cannot identify its owner, but the public retained guard can.
    coefficients = work.coefficients
    GC.@preserve coefficients begin
        foreign = unsafe_wrap(Array, pointer(coefficients, 2), (3, 2); own = false)
        @test !Base.mightalias(foreign, coefficients)
        @test_throws ArgumentError DD.retain_runge_kutta_observation!(
            work, foreign, fixture.initial, fixture.stages,
            0.0, 0.25, 0.25, fixture.execution,
        )
        @test isequal(df32dense_snapshot(work), saved)
        mask = unsafe_wrap(Array, Ptr{Bool}(pointer(coefficients, 2)), (3,); own = false)
        @test_throws ArgumentError DD.observe_runge_kutta!(
            work, 0.125, mask, fixture.execution,
        )
        @test isequal(df32dense_snapshot(work), saved)
    end
    @test_throws ArgumentError DD.RungeKuttaObservationWork(
        fixture.execution, ComplexF64.(fixture.initial), fixture.tableau,
        df32dense_weights(), fixture.flow; fixture.arithmetic,
    )
    @test_throws ArgumentError DD.RungeKuttaObservationWork(
        fixture.execution, fixture.initial, fixture.tableau,
        df32dense_weights(), fixture.flow; arithmetic = DD.SourceArithmetic(Float32),
    )
    replay = df32dense_fixture(; replay = true)
    @test_throws ArgumentError df32dense_work(replay)
    work.operator.generation = typemax(UInt64)
    saved = df32dense_snapshot(work)
    @test_throws ArgumentError df32dense_retain!(work, fixture)
    @test isequal(df32dense_snapshot(work), saved)
end

mutable struct DF32DenseSource{E}
    execution::E
    coefficient::ComplexF32
    calls::Int
    fail_at::Int
end

function (source::DF32DenseSource)(output, field, position)
    source.calls += 1
    source.calls == source.fail_at && error("controlled dense source failure")
    coefficient = source.coefficient
    DD.each!(source.execution, size(field)...) do row, column
        output[row, column] = coefficient * field[row, column]
    end
    return nothing
end

function df32dense_session(;
        tasks = 1, fail_at = 0, rtol = 1.0e-4, phase_rate = -0.125 + 0.25im,
    )
    execution = DD.KernelExecution(DD.KernelAbstractions.CPU(), tasks, true)
    state = fill(ComplexF32(0.5, 0.25), 3, 2)
    arithmetic = DD.SourceArithmetic(Float32; state_exponent = 7, field_exponent = 3)
    tableau = df32dense_tableau()
    schedule = DD.MarchSchedule(;
        planes = collect(1:32) ./ 32, breakpoints = [0.0, 1.0], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0,
    )
    spec = DD.WideStartSpec(
        0.0, tableau, schedule; geometry = DD.WideGeometry(0.0, 1.0; step_max = 0.125),
        arithmetic, method = :rk, order_max = 0, source_absent = false,
    )
    operator = DD.DiagonalOperator(execution, fill(ComplexF64(phase_rate), size(state)))
    flow = DD.DF32LinearFlow(execution, operator, state, spec)
    source = DF32DenseSource(execution, ComplexF32(-0.25, 0.125), 0, fail_at)
    control = DD.PIControl(;
        safety = 0.8, max_factor = 2.0, min_factor = 0.2, beta1 = 0.14, beta2 = -0.08,
        lag = 0, reject_free = false, estimator_order = 5, start_growth = 10.0,
        dz_min = 1.0e-12, dz_max = 0.125, auto_init = false,
    )
    norm = DD.MarchNorm(execution, fill(true, 3), fill(true, 3), ones(3), rtol, 0.0)
    return DD.RungeKuttaSession(
        execution, state, 0.0, 0.125, tableau, control, norm, schedule, source, flow;
        arithmetic, dense_weights = df32dense_weights(),
    )
end

function df32dense_turn!(session; limit = 1)
    report = DD.advance!(session; work_limit = limit, deadline = 60.0)
    DD.acknowledge!(session, report.generation)
    return report.kind
end

function df32dense_turn_allocated(session)
    return @allocated df32dense_turn!(session)
end

function df32dense_collect(session; limit = 1)
    observed = Matrix{ComplexF32}[]
    positions = Float64[]
    committed = Float64[]
    while !session.state.finished
        report = DD.advance!(session; work_limit = limit, deadline = 60.0)
        report.kind == DD.SESSION_FAILED && error(session.state.failure_message)
        if report.kind == DD.SESSION_PLANE
            push!(observed, copy(DD._march_observed_field(session)))
            push!(positions, DD._march_observed_position(session))
            push!(committed, session.state.z)
        end
        DD.acknowledge!(session, report.generation)
    end
    return (; observed, positions, committed)
end

@testset "DF32 adaptive RK dense delivery reuses the shared march" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        session = df32dense_session(; tasks)
        result = df32dense_collect(session)
        @test length(result.positions) == 32 && result.positions == collect(1:32) ./ 32
        @test all(diff(result.committed) .>= 0) && any(result.positions .< result.committed)
        @test session.state.accepted == 8 && session.state.rejected == 0
        @test session.source.calls == 7session.state.attempts
        @test session.state.z == 1.0 && !session.state.poisoned
        # DOPRI's matched scalar stability polynomial is formed independently in
        # MPFR. A path through 7 stages, their flows and dense contraction has
        # <1024 real roundings. gamma_(1024*8+256) includes every <5u flow error;
        # absolute stage recursion bounds propagation without assuming cancellation.
        u = BigFloat(eps(Float32)) / 2
        gamma = (1024 * 8 + 256) * u / (1 - (1024 * 8 + 256) * u)
        gain, rate, h = Complex{BigFloat}(-0.25, 0.125),
            Complex{BigFloat}(-0.125, 0.25), big(1) / 8
        table = session.tableau
        slopes = fill(zero(gain), 7)
        major = fill(BigFloat(0), 7)
        for stage in 1:7
            combination = sum(
                (
                    BigFloat(table.matrix[stage][earlier]) * slopes[earlier]
                        for earlier in 1:(stage - 1)
                ); init = zero(gain),
            )
            slopes[stage] = gain * (1 + h * combination)
            major[stage] = abs(gain) * (
                1 + h * sum(
                    (
                        abs(BigFloat(table.matrix[stage][earlier])) * major[earlier]
                            for earlier in 1:(stage - 1)
                    ); init = BigFloat(0),
                )
            )
        end
        multiplier = exp(rate * h) * (
            1 + h * sum(
                BigFloat(table.solution_weights[j]) * slopes[j] for j in 1:7
            )
        )
        # Four real-flow magnitudes also bound intermediate forward/backward
        # source transport on this unit-interval tableau, before cancellation.
        amplification = exp(4abs(real(rate)) * h) * (
            1 + h * sum(
                abs(BigFloat(table.solution_weights[j])) * major[j] for j in 1:7
            )
        )
        b2 = h * sum(abs(BigFloat(table.solution_weights[j])) * major[j] for j in 1:7)
        b3 = h * major[1] + b2
        b4 = b2 + h * major[end] + b3
        b5 = h * sum(abs(BigFloat(df32dense_weights()[j])) * major[j] for j in 1:7)
        absolute_extension = 1 + b2 + b3 + b4 + b5
        bound = gamma * abs(Complex{BigFloat}(0.5, 0.25)) * amplification^8 *
            absolute_extension * exp(abs(real(rate)) * h)
        for (position, field) in zip(result.positions, result.observed)
            full = floor(Int, position / Float64(h))
            theta = (BigFloat(position) - full * h) / h
            b = sum(BigFloat(table.solution_weights[j]) * slopes[j] for j in 1:7)
            r2, r3 = h * b, h * slopes[1] - h * b
            r4 = r2 - h * slopes[end] - r3
            r5 = h * sum(BigFloat(df32dense_weights()[j]) * slopes[j] for j in 1:7)
            extension = 1 + theta * (
                r2 + (1 - theta) * (r3 + theta * (r4 + (1 - theta) * r5))
            )
            expected = Complex{BigFloat}(0.5, 0.25) * multiplier^full *
                extension * exp(rate * theta * h)
            @test bound > 0
            @test maximum(abs, Complex{BigFloat}.(field) .- expected) <= bound
        end
        chunked = df32dense_session(; tasks)
        other = df32dense_collect(chunked; limit = 256)
        @test other.positions == result.positions && other.committed == result.committed
        for (left, right) in zip(other.observed, result.observed)
            difference = maximum(abs, Complex{BigFloat}.(left) - Complex{BigFloat}.(right))
            @test difference <= 2bound
        end
        warm = df32dense_session(; tasks)
        for unused in 1:7
            df32dense_turn!(warm)
        end
        @test df32dense_turn_allocated(warm) == 0
        @test df32dense_turn_allocated(warm) == 0
        @test df32dense_turn_allocated(warm) == 0
        @test df32dense_turn_allocated(warm) == 0
        failing = df32dense_session(; tasks, fail_at = 1)
        df32dense_turn!(failing)
        field = copy(failing.field)
        @test df32dense_turn!(failing) == DD.SESSION_FAILED
        @test failing.field == field && failing.state.accepted == 0
        @test failing.source.calls == 1 && !failing.state.poisoned
    end
end

@testset "Rejected DF32 trials do not publish a dense window" begin
    session = df32dense_session(; rtol = 1.0e-12)
    df32dense_turn!(session)
    saved = copy(session.field)
    report = DD.advance!(session; work_limit = 1, deadline = 60.0)
    @test report.kind == DD.SESSION_PROGRESS
    @test session.state.attempts == 1 && session.state.rejected == 1
    @test session.state.accepted == 0 && session.field == saved
    @test session.source.calls == 7
    @test !session.observation.work.ready
    @test session.observation.work.operator.retained === nothing
    @test session.observation.work.operator.generation == 0
    DD.acknowledge!(session, report.generation)
end

@testset "A refused dense branch preserves the accepted session field" begin
    session = df32dense_session(; phase_rate = 8.0 + 0.0im)
    df32dense_turn!(session)
    saved = copy(session.field)
    report = DD.advance!(session; work_limit = 1, deadline = 60.0)
    @test report.kind == DD.SESSION_FAILED
    @test occursin("stiff-branch certificate straddles", session.state.failure_message)
    @test session.field == saved && session.state.z == 0.0
    @test !session.state.in_flight && !session.state.poisoned
    # The shared scheduler records the completed trial before candidate finishing
    # and dense retention. Its accepted row is actual work, not an endpoint commit.
    @test session.state.attempts == session.state.accepted == 1
    @test session.state.rejected == 0 && session.source.calls == 7
    @test session.attempts.count == 1 && session.attempts.accepted[1]
    @test !session.observation.work.ready
    @test session.observation.work.operator.retained === nothing
    @test session.observation.work.operator.generation == 0
    DD.acknowledge!(session, report.generation)
end

function df32dense_factor_ir(coefficients, terms, kind, bundle, index)
    argument = DD._df32_dense_argument(coefficients, terms, kind, bundle, index)
    return iszero(argument) ? one(ComplexF32) : first(DD._df32_exp_pair(argument))
end

@testset "DF32 dense retained inventory and owned numerical IR" begin
    fixture = df32dense_fixture()
    work = df32dense_work(fixture)
    df32dense_retain!(work, fixture)
    operation = work.operator.retained
    @test isbitstype(typeof(operation))
    @test sizeof(typeof(operation)) == sizeof(DD._DF32FlowOperation)
    ledger, seen = Dict{Symbol, Int}(), IdDict{Any, Bool}()
    DD._memory_arrays!(ledger, seen, "flow", fixture.flow)
    empty!(ledger)
    DD._memory_arrays!(ledger, seen, "dense", work)
    @test sum(values(ledger)) == sum(values(DD.runge_kutta_observation_memory(work)))
    @test !any(contains("binding"), string.(keys(ledger)))
    prepared = fixture.flow.binding.operator
    io = IOBuffer()
    code_llvm(
        io, df32dense_factor_ir,
        Tuple{
            typeof(prepared.coefficients), typeof(prepared.terms),
            typeof(prepared.kind), typeof(operation.scalars), Int,
        }; raw = true,
    )
    llvm = String(take!(io))
    @test !occursin(r"\b(double|fpext|fptrunc)\b", llvm)
    @test !occursin(r"\b(fast|reassoc|nnan|ninf|nsz|arcp|afn)\b", llvm)
    @test !occursin(r"@(julia_|ijl_)?(sin|cos|exp|log)", llvm)
    @test !occursin("gc_pool_alloc", llvm)
end
