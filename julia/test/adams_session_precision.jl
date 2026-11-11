# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

mutable struct AdamsSessionPrecisionSource{T, E}
    execution::E
    gain::Complex{T}
    offset::Complex{T}
    slope::Complex{T}
    calls::Int
    fail_at::Int
    data::Nothing
end

function (source::AdamsSessionPrecisionSource)(output, field, position)
    source.calls += 1
    source.calls == source.fail_at && error("controlled Adams precision source failure")
    gain = source.gain
    forcing = source.offset + source.slope * typeof(real(source.gain))(position)
    SP.each!(source.execution, size(field)...) do row, column
        output[row, column] = gain * field[row, column] + forcing
    end
    return nothing
end

function adams_session_precision_policy(planes; interpolate = false, changes...)
    fields = (
        order = 1, variable_order = false, safety = 0.8, max_growth = 2.0,
        min_factor = 0.2, rejection_growth = 0.9, hysteresis = 1.2, restart_after = 6,
        collar_stability = 0.7, real_intervals = (2.0, 1.0, 0.5, 0.3, 0.2, 0.15),
        collar_rate = 0.0, carrier = false, carrier_cap = 4.0, max_calls = nothing,
        equal_approach = false, interval_cap = false, dz_min = 1.0e-10,
        dz_max = 1 / 16, stretch = 1.0, z_end = last(planes),
        save_planes = interpolate ? Float64[] : planes, hard_kinks = Float64[],
        breakpoints = [0.0, 1 / 4], landing_ulps = 8, interpolate, quiet_interval = nothing,
    )
    return SP.AdamsControlPolicy(; merge(fields, (; changes...))...)
end

function adams_session_precision_fixture(;
        tasks = 1, replay = true, dense = false, capacity = 16, order = 1,
        arithmetic = SP.SourceArithmetic(Float32; state_exponent = 3, field_exponent = 1),
        initial = nothing, rtol = 0.01, gain = 0.0im, offset = 0.125 + 0.0625im,
        slope = 0.0im, linear = -0.125 + 0.25im, varying = false,
        planes = [1 / 16, 1 / 8], proposal = 1 / 64, fail_at = 0,
        source_absent = false, max_calls = nothing, hard_kinks = Float64[],
    )
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    T = SP.source_type(arithmetic)
    field = initial === nothing ? Complex{T}[
            complex(row / 8, column / 16) for row in 1:4, column in 1:2
        ] : initial
    source = AdamsSessionPrecisionSource(
        execution, Complex{T}(gain), Complex{T}(offset), Complex{T}(slope), 0, fail_at,
        nothing,
    )
    operator = if varying
        # Phi(z)=linear*z^2/2 on [0,1/4], derived from z=(x+1)/8.
        coefficients = zeros(ComplexF64, length(field), 3, 1)
        coefficients[:, 1, 1] .= 3linear / 256
        coefficients[:, 2, 1] .= linear / 64
        coefficients[:, 3, 1] .= linear / 256
        SP.RepresentedOperator(
            execution, SP.ChebyshevTable(execution, [0.0, 1 / 4], coefficients),
            size(field)...,
        )
    else
        SP.DiagonalOperator(execution, fill(ComplexF64(linear), size(field)))
    end
    flow = SP.LinearFlow(execution, operator, field)
    norm = SP.MarchNorm(
        execution, [true, false, true, true], [true, false, true, true],
        ones(4), rtol, 0.0,
    )
    policy = adams_session_precision_policy(
        planes; interpolate = dense, order,
        max_calls, hard_kinks
    )
    schedule = SP.MarchSchedule(;
        planes, policy.breakpoints, policy.hard_kinks, policy.stretch, step_planes = 0,
        replay_steps = replay ? fill(1 / 64, 8) : Float64[],
        replay_edges = replay ? collect(0:8) ./ 64 : Float64[],
    )
    session = SP.AdamsSession(
        execution, field, 0.0, proposal, policy, norm, schedule, source, flow;
        arithmetic, capacity, source_absent,
    )
    return session, field
end

function adams_session_precision_collect(session; work_limit = 256, deadline = 60.0)
    reports = Tuple{SP.SessionEvent, Float64, Int}[]
    rows = Tuple{Float64, Float64, Float64, Bool, Int}[]
    fields = Tuple{Float64, Matrix{eltype(session.field)}}[]
    turns = 0
    while !session.state.finished
        turns += 1
        turns < 500 || error("Adams precision fixture exceeded bounded work")
        report = SP.advance!(session; work_limit, deadline)
        push!(reports, (report.kind, report.z, report.generation))
        for index in 1:session.attempts.count
            buffer = session.attempts
            push!(
                rows, (
                    buffer.z[index], buffer.dz[index], buffer.norm[index],
                    buffer.accepted[index], session.orders[index],
                )
            )
        end
        if report.kind == SP.SESSION_PLANE
            push!(
                fields, (
                    SP._march_observed_position(session),
                    copy(SP._march_observed_field(session)),
                )
            )
        end
        SP.acknowledge!(session, report.generation)
    end
    return (; reports, rows, fields)
end

function adams_session_precision_turn!(session)
    report = SP.advance!(session; work_limit = 1, deadline = 60.0)
    SP.acknowledge!(session, report.generation)
    return nothing
end
function adams_session_precision_allocations(session)
    adams_session_precision_turn!(session)
    return @allocated adams_session_precision_turn!(session)
end

@testset "Binary32 Adams sessions retain arithmetic, raw buffers and exact work" begin
    for tasks in unique((1, Threads.nthreads(:default))), order in (1, 4)
        session, initial = adams_session_precision_fixture(; tasks, order)
        saved = copy(initial)
        result = adams_session_precision_collect(session)
        @test session.arithmetic == SP.SourceArithmetic(
            Float32; state_exponent = 3, field_exponent = 1,
        )
        @test eltype(session.field) === eltype(session.history.source) === ComplexF32
        @test eltype(session.history.states) === ComplexF32
        @test eltype(session.history.rates) === eltype(session.work.exponent) === ComplexF64
        @test session.state.accepted == 8 && session.state.rejected == 0
        @test length(result.rows) == 8 && all(row -> row[4], result.rows)
        @test session.ledger.startup == 1 && session.ledger.predictor == 8
        @test session.ledger.source_evaluations == session.source.calls == 9
        @test SP.adams_charged_calls(session.ledger) == 9
        @test session.ledger.corrected == session.ledger.dense_observations == 0
        @test all(iszero, session.field[2, :])
        @test initial == saved && session.field !== initial
        @test length(unique(last.(result.reports))) == length(result.reports)
        @test all(first(report) != SP.SESSION_FAILED for report in result.reports)
        if order == 1
            setprecision(BigFloat, 256) do
                z, h = BigFloat(1 / 8), BigFloat(1 / 64)
                rate = Complex{BigFloat}(-0.125, 0.25)
                forcing = Complex{BigFloat}(0.125, 0.0625)
                phase = h * rate
                flow = exp(phase)
                phi1 = expm1(phase) / phase
                phi2 = (expm1(phase) - phase) / phase^2
                upper = maximum(abs, Complex{BigFloat}.(saved))
                error = BigFloat(0)
                unit = BigFloat(2)^-24
                for index in 1:8
                    old_bound = exp(abs(rate) * (index - 1) * h) *
                        (upper + (index - 1) * h * abs(forcing))
                    contributions = abs(flow) * old_bound +
                        h * (abs(phi1) + 2abs(phi2)) * abs(forcing)
                    # Complete endpoint/old source weights precede cancellation;
                    # 64 rounded C32 component operations cover this q=1 graph.
                    local_error = (64unit / (1 - 64unit) + BigFloat(2)^-39) * contributions
                    error = abs(flow) * error + local_error
                end
                expected = exp(rate * z) .* Complex{BigFloat}.(saved) .+
                    (expm1(rate * z) / rate) * forcing
                expected[2, :] .= 0
                @test error > 0
                @test maximum(abs, Complex{BigFloat}.(session.field) - expected) <= error
            end
        end
        warm, _ = adams_session_precision_fixture(; tasks, order)
        adams_session_precision_turn!(warm)
        @test adams_session_precision_allocations(warm) == 0
        @test adams_session_precision_allocations(warm) == 0
        SP.close_session!(warm)
    end
end

@testset "Binary32 Adams bounded transport and failure bookkeeping" begin
    for replay in (false, true)
        whole, _ = adams_session_precision_fixture(; replay, order = 4)
        sliced, _ = adams_session_precision_fixture(; replay, order = 4, capacity = 1)
        first, second = adams_session_precision_collect(whole),
            adams_session_precision_collect(sliced; work_limit = 1, deadline = 0.0)
        @test map(row -> (row[1], row[2], row[4], row[5]), first.rows) ==
            map(row -> (row[1], row[2], row[4], row[5]), second.rows)
        @test filter(
            event -> event[1] != SP.SESSION_PROGRESS,
            map(event -> event[1:2], second.reports)
        ) == map(event -> event[1:2], first.reports)
        # Delivery adds no field arithmetic in this controlled same-CPU trace;
        # allow two final component roundings rather than require identical bits.
        budget = 4eps(Float32) * maximum(abs, whole.field)
        @test budget > 0 && maximum(abs, whole.field - sliced.field) <= budget
        @test whole.source.calls == sliced.source.calls
    end
    rejected, _ = adams_session_precision_fixture(;
        replay = false, gain = 1.0 + 0im, offset = 0.0im,
        linear = 0.0im, rtol = 1.0e-3, proposal = 1 / 8,
    )
    rejected_trace = adams_session_precision_collect(rejected)
    @test rejected.state.rejected > 0
    @test any(row -> !row[4], rejected_trace.rows)
    @test rejected.ledger.predictor == length(rejected_trace.rows)
    @test rejected.ledger.source_evaluations == rejected.source.calls ==
        rejected.ledger.startup + rejected.ledger.predictor
    for fail_at in (1, 2)
        failed, _ = adams_session_precision_fixture(; fail_at)
        adams_session_precision_turn!(failed)
        original = copy(failed.field)
        report = SP.advance!(failed; work_limit = 1)
        @test report.kind == SP.SESSION_FAILED && !report.poisoned
        @test failed.source.calls == failed.ledger.source_evaluations == fail_at
        @test failed.ledger.startup == 1 && failed.ledger.predictor == fail_at - 1
        @test failed.field == original && failed.state.z == 0.0
        @test failed.attempts.count == 0
    end
    absent, _ = adams_session_precision_fixture(; source_absent = true, fail_at = 1)
    adams_session_precision_collect(absent)
    @test absent.source.calls == absent.ledger.source_evaluations == 0
    @test absent.ledger.startup == 1 && absent.ledger.predictor == 8
    budgeted, _ = adams_session_precision_fixture(; max_calls = 2)
    adams_session_precision_collect(budgeted; work_limit = 1)
    @test budgeted.state.failure_code == :AdamsBudgetError
    @test budgeted.state.accepted == 1 && budgeted.ledger.source_evaluations == 2
end

@testset "Binary32 adaptive dense and represented compensation keep committed state" begin
    for tasks in unique((1, Threads.nthreads(:default))), varying in (false, true)
        planes = collect(1:8) ./ 64
        session, initial = adams_session_precision_fixture(;
            tasks, replay = false, dense = true, varying, order = 4, planes,
            linear = 0.25 + 0.125im, offset = 0.0im, slope = 0.125 + 0.0625im,
            rtol = 1.0e-4,
        )
        result = adams_session_precision_collect(session; work_limit = 1)
        @test all(report[1] != SP.SESSION_FAILED for report in result.reports)
        @test length(result.fields) == length(planes)
        @test session.observation.work.arithmetic == session.arithmetic
        @test eltype(session.observation.work.field) === ComplexF32
        @test eltype(session.observation.work.dense.rates) === ComplexF64
        @test session.work.compensated == varying
        @test session.ledger.source_evaluations == session.source.calls ==
            session.ledger.startup + session.ledger.predictor
        @test session.ledger.dense_observations > 0
        for (position, field) in result.fields
            @test all(iszero, field[2, :])
            # This is a predeclared short-window accuracy envelope against an
            # independent analytic solution, not a claim that local rtol is global.
            # Varying case: y'=a*z*y+b*z, y=exp(a*z²/2)y0+(b/a)expm1(a*z²/2).
            setprecision(BigFloat, 256) do
                a, b, z = Complex{BigFloat}(0.25, 0.125),
                    Complex{BigFloat}(0.125, 0.0625), BigFloat(position)
                phase = varying ? a * z^2 / 2 : a * z
                increment = varying ? (b / a) * expm1(phase) :
                    b * (expm1(phase) - phase) / a^2
                expected = exp(phase) .* Complex{BigFloat}.(initial) .+ increment
                expected[2, :] .= 0
                envelope = BigFloat(3.0e-5) * maximum(abs, Complex{BigFloat}.(initial))
                @test envelope > 0
                @test maximum(abs, Complex{BigFloat}.(field) - expected) <= envelope
            end
        end
        warm, _ = adams_session_precision_fixture(;
            tasks, replay = false, dense = true, varying, order = 4, planes,
            offset = 0.0im, slope = 0.125 + 0.0625im, linear = 0.25 + 0.125im,
        )
        adams_session_precision_turn!(warm)
        @test adams_session_precision_allocations(warm) == 0
        @test adams_session_precision_allocations(warm) == 0
        @test SP._march_observed_position(session) == session.state.z
        SP.close_session!(warm)
    end
end

function adams_session_precision_material(execution, arithmetic)
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

function adams_session_precision_material_session(arithmetic; tasks = 1)
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    T = SP.source_type(arithmetic)
    initial = zeros(Complex{T}, 4, 1)
    initial[1] = T(ldexp(2.0, -arithmetic.state_exponent))
    source = SP.ModalSource(
        execution, adams_session_precision_material(execution, arithmetic); node_chunk = 1,
    )
    flow = SP.LinearFlow(
        execution, SP.DiagonalOperator(execution, zeros(ComplexF64, 4, 1)), initial,
    )
    norm = SP.MarchNorm(execution, fill(true, 4), fill(true, 4), ones(4), 0.01, 0.0)
    policy = adams_session_precision_policy([1 / 8])
    schedule = SP.MarchSchedule(;
        planes = [1 / 8], breakpoints = policy.breakpoints, hard_kinks = policy.hard_kinks,
        stretch = policy.stretch, step_planes = 0, replay_steps = fill(1 / 64, 8),
        replay_edges = collect(0:8) ./ 64,
    )
    session = SP.AdamsSession(
        execution, initial, 0.0, 1 / 64, policy, norm, schedule, source, flow; arithmetic,
    )
    return session, initial
end

function adams_session_precision_cubic_reference()
    h, value = BigFloat(1 / 64), Complex{BigFloat}(2)
    source = value -> -im * abs2(value) * value / 64
    sampled = source(value)
    for _ in 1:8
        predictor = value + h * sampled
        newest = source(predictor)
        value += (h / 2) * (sampled + newest)
        sampled = newest
    end
    return value
end

@testset "Binary32 Adams sessions compose scaled FFT and physical Kerr arithmetic" begin
    for tasks in unique((1, Threads.nthreads(:default)))
        reference = adams_session_precision_cubic_reference()
        wide, _ = adams_session_precision_material_session(
            SP.SourceArithmetic(Float64); tasks,
        )
        adams_session_precision_collect(wide)
        for (state_exponent, field_exponent) in ((3, 1), (20, 18))
            arithmetic = SP.SourceArithmetic(Float32; state_exponent, field_exponent)
            session, initial = adams_session_precision_material_session(arithmetic; tasks)
            saved = copy(initial)
            # One occupied DC bin has E=A/4 and N(A)=-i*|A|²*A/64. On |A|<=3
            # the source Jacobian norm is <=27/64. In the two-register PEC graph,
            # the weighted max norm max(|delta A|, |delta sampled|/L) grows by at
            # most 1+h*L+(h*L)^2/2 in one step. The initial source error is also
            # counted. Each source/step path has fewer than 2048 rounded scalar
            # operations under the explicitly qualified finite four-point FFT
            # model: its component error is <=8u times the absolute input sum.
            # This is conditional FFT accuracy, not a proof from butterfly counts.
            unit = BigFloat(2)^-24
            growth = 1 + (BigFloat(1) / 64) * (BigFloat(27) / 64) +
                ((BigFloat(1) / 64) * (BigFloat(27) / 64))^2 / 2
            budget = 9 * (2048unit / (1 - 2048unit)) * 3 * growth^8
            @test budget > 0
            adams_session_precision_turn!(session)
            @test adams_session_precision_allocations(session) == 0
            while !session.state.finished
                adams_session_precision_turn!(session)
                physical = ComplexF64.(session.field) .* ldexp(1.0, state_exponent)
                @test maximum(abs, physical) <= 3
                predictor = ComplexF64.(session.work.predictor) .*
                    ldexp(1.0, state_exponent)
                @test maximum(abs, predictor) <= 3
            end
            physical = ComplexF64.(session.field) .* ldexp(1.0, state_exponent)
            @test abs(Complex{BigFloat}(physical[1]) - reference) <= budget
            @test maximum(abs, physical - wide.field) <= budget
            @test all(iszero, physical[2:end])
            @test initial == saved && session.source.calls == 9
            @test session.ledger.source_evaluations == 9
            @test session.field[1] != ComplexF32(physical[1])
        end
    end
end

@testset "Adams session arithmetic admission preserves caller and source storage" begin
    arithmetic = SP.SourceArithmetic(Float32; state_exponent = 3, field_exponent = 1)
    session, initial = adams_session_precision_material_session(arithmetic)
    execution, source, flow = session.execution, session.source, session.flow
    wrong_flow = SP.LinearFlow(execution, flow.operator, ComplexF64.(initial))
    saved = copy(initial)
    variants = (
        (SP.SourceArithmetic(Float64), flow),
        (SP.SourceArithmetic(Float32; state_exponent = 4, field_exponent = 1), flow),
        (SP.SourceArithmetic(Float32; state_exponent = 3, field_exponent = 2), flow),
        (arithmetic, wrong_flow),
    )
    for (policy, prepared_flow) in variants
        @test_throws ArgumentError SP.AdamsSession(
            execution, initial, 0.0, 1 / 64, session.policy, session.norm,
            session.schedule, source, prepared_flow; arithmetic = policy,
        )
        @test initial == saved && source.calls == 0 && !source.ready
    end
    @test_throws ArgumentError SP.AdamsObservationWork(execution, initial, 1)
    @test_throws ArgumentError SP.AdamsObservationWork(
        execution, ComplexF64.(initial), 1; arithmetic,
    )
    @test initial == saved && source.calls == 0
end

@testset "Dense arithmetic refusal preserves an existing observed window" begin
    session, _ = adams_session_precision_fixture(; replay = false, dense = true)
    adams_session_precision_turn!(session)
    report = SP.advance!(session; work_limit = 1)
    @test report.kind != SP.SESSION_FAILED
    work = session.observation.work
    @test work.ready
    SP.observe_adams!(work, work.dense.position, session.norm.active, session.execution)
    field, endpoint, coefficients = copy(work.field), copy(work.dense.endpoint),
        copy(work.dense.coefficients)
    destination = work.destination
    foreign, _ = adams_session_precision_fixture(;
        arithmetic = SP.SourceArithmetic(Float32; state_exponent = 4, field_exponent = 1),
    )
    @test_throws ArgumentError SP.retain_adams_observation!(work, foreign, destination)
    @test work.ready && work.destination == destination
    @test work.field == field && work.dense.endpoint == endpoint
    @test work.dense.coefficients == coefficients
    @test foreign.source.calls == 0 && foreign.state.accepted == 0
    SP.acknowledge!(session, report.generation)
    ledger = SP.session_memory(session)
    arrays = (SP._adams_observation_arrays(work.dense)..., work.field, work.failures)
    listed = [
        getproperty(ledger, name) for name in keys(ledger)
            if startswith(string(name), "session.observation.work.")
    ]
    @test sum(listed) == sum(sizeof, arrays)
    @test length(listed) == length(arrays)
    @test length(unique(objectid.(arrays))) == length(arrays)
    @test !any(occursin("arithmetic", string(name)) for name in keys(ledger))
    @test sizeof(work.field) == 8length(work.field)
    @test sizeof(work.dense.rates) == 16length(work.dense.rates)
end

@testset "Binary32 Adams snapshot receipts do not change source work" begin
    session, _ = adams_session_precision_fixture()
    @test SP.request_snapshot!(session) == (1, false)
    @test SP.request_snapshot!(session) == (1, true)
    adams_session_precision_turn!(session)
    report = SP.advance!(session; work_limit = 1)
    @test report.kind == SP.SESSION_SNAPSHOT && session.state.accepted == 1
    before = copy(session.field)
    calls = session.ledger.source_evaluations
    @test_throws ArgumentError SP.advance!(session)
    @test_throws ArgumentError SP.acknowledge!(session, report.generation + 1)
    @test session.field == before && session.ledger.source_evaluations == calls == 2
    SP.acknowledge!(session, report.generation)
    @test session.attempts.count == 0
    SP.close_session!(session)
    SP.close_session!(session)
    @test_throws ArgumentError SP.advance!(session)
end

@testset "Binary32 dense observations reject nonfinite values before masking" begin
    session, _ = adams_session_precision_fixture(; replay = false, dense = true)
    adams_session_precision_turn!(session)
    report = SP.advance!(session; work_limit = 1)
    work = session.observation.work
    saved, calls = copy(session.field), session.ledger.source_evaluations
    work.dense.endpoint[2, 1] = ComplexF32(Inf)
    @test_throws SP.NonfiniteSourceError SP.observe_adams!(
        work, work.destination, session.norm.active, session.execution,
    )
    @test session.field == saved && session.ledger.source_evaluations == calls
    @test !session.norm.active[2]
    SP.acknowledge!(session, report.generation)
    SP.close_session!(session)
end
