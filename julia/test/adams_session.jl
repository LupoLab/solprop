# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import SolpropJulia as SP

mutable struct AdamsSessionTestSource{E}
    execution::E
    gain::ComplexF64
    intercept::ComplexF64
    slope::ComplexF64
    calls::Int
    fail_on::Int
    nonfinite_on::Int
end

function (source::AdamsSessionTestSource)(output, field, position)
    source.calls += 1
    source.calls == source.fail_on && error("controlled Adams source failure")
    gain = source.gain
    forcing = source.calls == source.nonfinite_on ?
        ComplexF64(Inf) : source.intercept + source.slope * position
    SP.each!(source.execution, size(output)...) do row, column
        output[row, column] = gain * field[row, column] + forcing
    end
    return nothing
end

mutable struct AdamsManufacturedSource{E}
    execution::E
    rate::ComplexF64
    exponent::ComplexF64
    calls::Int
end

function (source::AdamsManufacturedSource)(output, field, position)
    source.calls += 1
    forcing = (source.exponent - source.rate * position) * exp(source.exponent * position)
    SP.each!(source.execution, size(output)...) do row, column
        output[row, column] = forcing
    end
    return nothing
end

function adams_session_policy(; changes...)
    # Small synthetic problems exercise control and lifecycle, not a material model.
    fields = (
        order = 4, variable_order = true, safety = 0.8, max_growth = 2.0,
        min_factor = 0.2, rejection_growth = 0.9, hysteresis = 1.2, restart_after = 3,
        collar_stability = 0.7, real_intervals = (2.0, 1.0, 0.5, 0.3, 0.2, 0.15),
        collar_rate = 0.0, carrier = false, carrier_cap = 4.0, max_calls = nothing,
        equal_approach = true, interval_cap = false, dz_min = 1.0e-10, dz_max = 0.05,
        stretch = 1.1, z_end = 0.2, save_planes = [0.1, 0.2],
        hard_kinks = Float64[], breakpoints = [0.0, 0.4], landing_ulps = 8,
        interpolate = false, quiet_interval = nothing,
    )
    return SP.AdamsControlPolicy(; merge(fields, (; changes...))...)
end

function adams_session_fixture(;
        tasks = 1, capacity = 256, policy = adams_session_policy(), replay = false,
        n_steps = 8, z = 0.0, proposal = 0.05, initial = ones(ComplexF64, 3, 2),
        gain = 0.0 + 0.0im, intercept = 0.2 + 0.1im, slope = -0.3 + 0.2im,
        rate = 0.0 + 0.0im, rtol = 1.0e-3, atol = 0.0, active = nothing,
        services = SP._NoSessionServices(), source_absent = false, fail_on = 0,
        nonfinite_on = 0, varying = false, plane_index = nothing, plane_pending = true,
        step_planes = 0, manufactured = false, fixed_steps = nothing,
    )
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    source = manufactured ?
        AdamsManufacturedSource(execution, ComplexF64(rate), 0.2 - 0.1im, 0) :
        AdamsSessionTestSource(
            execution, ComplexF64(gain), ComplexF64(intercept), ComplexF64(slope),
            0, fail_on, nonfinite_on,
        )
    if varying
        # Phi(z)=rate*z^2/2 on [0,.4]. This is an independently constructed
        # cumulative quadratic; its represented derivative is rate*z.
        coefficients = zeros(ComplexF64, length(initial), 3, 1)
        coefficients[:, 1, 1] .= 0.03rate
        coefficients[:, 2, 1] .= 0.04rate
        coefficients[:, 3, 1] .= 0.01rate
        table = SP.ChebyshevTable(execution, [0.0, 0.4], coefficients)
        operator = SP.RepresentedOperator(execution, table, size(initial)...)
    else
        operator = SP.DiagonalOperator(execution, fill(ComplexF64(rate), size(initial)))
    end
    flow = SP.LinearFlow(execution, operator, initial)
    mask = active === nothing ? fill(true, size(initial, 1)) : active
    norm = SP.MarchNorm(execution, mask, mask, ones(size(initial, 1)), rtol, atol)
    steps = if replay
        fixed_steps === nothing ? fill((policy.z_end - z) / n_steps, n_steps) : fixed_steps
    else
        Float64[]
    end
    edges = replay ? z .+ [0.0; cumsum(steps)] : Float64[]
    schedule = SP.MarchSchedule(;
        planes = policy.save_planes, breakpoints = policy.breakpoints,
        hard_kinks = policy.hard_kinks, stretch = policy.stretch, step_planes,
        replay_steps = steps, replay_edges = edges,
    )
    return SP.AdamsSession(
        execution, initial, z, proposal, policy, norm, schedule, source, flow;
        capacity, services, source_absent, plane_index, plane_pending,
    )
end

function adams_session_collect(session; work_limit = 256, deadline = 60.0)
    rows = Tuple{Float64, Float64, Float64, Bool, Int}[]
    events = Tuple{SP.SessionEvent, Float64}[]
    reports = SP.SessionReport[]
    iterations = 0
    while !session.state.finished
        iterations += 1
        iterations <= 2000 || error("test march exceeded its bounded attempt allowance")
        report = SP.advance!(session; work_limit, deadline)
        push!(reports, report)
        push!(events, (report.kind, report.z))
        buffer = session.attempts
        for index in 1:buffer.count
            push!(
                rows,
                (
                    buffer.z[index], buffer.dz[index], buffer.norm[index],
                    buffer.accepted[index], session.orders[index],
                ),
            )
        end
        @test buffer.count <= length(buffer.z) == length(session.orders)
        SP.acknowledge!(session, report.generation)
    end
    return rows, events, reports
end

function adams_session_ack_launch(session)
    report = SP.advance!(session)
    @test report.kind == SP.SESSION_LAUNCH
    @test SP.adams_charged_calls(session.ledger) == 0
    SP.acknowledge!(session, report.generation)
    return nothing
end

@testset "Adams admission requires one policy and schedule topology" begin
    session = adams_session_fixture()
    original = copy(session.field)
    for changes in (
            (; planes = [0.05, 0.2]), (; hard_kinks = [0.05]),
            (; breakpoints = [0.0, 0.5]), (; stretch = 1.2),
        )
        fields = (
            planes = session.schedule.planes, hard_kinks = session.schedule.hard_kinks,
            breakpoints = session.schedule.breakpoints, stretch = session.schedule.stretch,
            step_planes = 0,
        )
        schedule = SP.MarchSchedule(; merge(fields, changes)...)
        @test_throws ArgumentError SP.AdamsSession(
            session.execution, session.field, 0.0, 0.05, session.policy,
            session.norm, schedule, session.source, session.flow,
        )
        @test session.field == original && session.source.calls == 0
    end
end

@testset "represented Adams compensation converges to a manufactured solution" begin
    errors = Float64[]
    policy = adams_session_policy(order = 4, variable_order = false, save_planes = [0.2])
    for count in (8, 16, 32)
        session = adams_session_fixture(;
            policy, replay = true, n_steps = count, varying = true, manufactured = true,
            rate = -0.6 + 0.4im, initial = ones(ComplexF64, 1, 1),
        )
        _, events, _ = adams_session_collect(session)
        @test last(events)[1] == SP.SESSION_COMPLETED
        @test session.history.compensated && session.work.compensated
        expected = exp((0.2 - 0.1im) * session.state.z)
        discrepancy = abs(session.field[1] - expected)
        push!(errors, discrepancy)
        # The order-one PEC startup limits this sequence to cubic asymptotics.
        # Use a predeclared conservative fixture accuracy envelope, then require
        # that both refinements also demonstrate at least quadratic convergence.
        @test 0 < discrepancy <= 100 * (0.2 / count)^3
        latest = session.slots[session.control.held]
        @test session.history.states[1, 1, latest] == session.work.predictor[1]
        @test session.history.rates[1, 1, latest] == session.work.endpoint_rate[1]
    end
    @test errors[1] > 4errors[2] > 16errors[3]
end

@testset "snapped Adams endpoints keep unsnapped predictor positions" begin
    policy = adams_session_policy(;
        z_end = 1.3, save_planes = [1.3], breakpoints = [1.0, 1.4],
    )
    session = adams_session_fixture(;
        policy, z = 1.0, replay = true, fixed_steps = fill(0.1, 3),
    )
    adams_session_ack_launch(session)
    for _ in 1:2
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        SP.acknowledge!(session, report.generation)
    end
    @test session.state.z == 1.2
    @test session.positions[session.control.held] == 1.1 + 0.1
    @test session.positions[session.control.held] != session.state.z
    report = SP.advance!(session; work_limit = 1, deadline = 60.0)
    @test report.kind == SP.SESSION_PLANE
    @test session.stencil.nodes[1] != 0.0
end

mutable struct AdamsSessionTestServices
    quiet::Bool
    cap::Float64
    factor::Float64
    fail_finish::Bool
    damage_commit::Bool
    accepted::Int
    destination::Float64
end

SP._session_launch!(::AdamsSessionTestServices, session) = nothing
SP._session_observe!(::AdamsSessionTestServices, session, kind, plane) = nothing
SP._session_finish!(::AdamsSessionTestServices, session) = nothing
SP._session_quiet_interval(::AdamsSessionTestServices, session) = nothing
SP._session_barrier(::AdamsSessionTestServices, session) = :none
SP._session_continue!(::AdamsSessionTestServices, session) = :none
SP._session_acknowledge!(::AdamsSessionTestServices, session, generation) = nothing
SP._session_quiet(services::AdamsSessionTestServices, session) = services.quiet
SP._session_cap(services::AdamsSessionTestServices, session, proposal) =
    min(proposal, services.cap)

function SP._session_finish_candidate!(
        services::AdamsSessionTestServices, session, candidate, dz, destination,
    )
    services.destination = destination
    for index in eachindex(candidate)
        candidate[index] *= services.factor
        services.fail_finish && error("controlled candidate finishing failure")
    end
    # Deliberate fault injection exercises poisoning after history commit starts.
    services.damage_commit && resize!(session.positions, 0)
    return nothing
end

function SP._session_accepted!(services::AdamsSessionTestServices, session, dz)
    services.accepted += 1
    services.quiet = false
    return nothing
end

@testset "Adams services cap trials and finish a separate masked candidate" begin
    services = AdamsSessionTestServices(true, 0.02, 0.9, false, false, 0, NaN)
    policy = adams_session_policy(collar_rate = 100.0, quiet_interval = 0.0125)
    session = adams_session_fixture(; policy, services, active = [true, false, true])
    adams_session_ack_launch(session)
    first = SP.advance!(session; work_limit = 1, deadline = 60.0)
    @test first.kind == SP.SESSION_PROGRESS
    @test session.attempts.dz[1] == 0.0125
    @test services.destination == session.state.z
    @test all(iszero, session.field[2, :])
    @test services.accepted == 1 && !services.quiet
    SP.acknowledge!(session, first.generation)
    second = SP.advance!(session; work_limit = 1, deadline = 60.0)
    @test second.kind == SP.SESSION_PROGRESS
    @test session.attempts.dz[1] == policy.collar_stability * policy.real_intervals[1] / 100
    @test services.accepted == 2

    for damage_commit in (false, true)
        failing = AdamsSessionTestServices(
            false, Inf, 0.5, !damage_commit, damage_commit, 0, NaN,
        )
        failed = adams_session_fixture(; services = failing)
        adams_session_ack_launch(failed)
        report = SP.advance!(failed)
        @test report.kind == SP.SESSION_FAILED
        @test report.poisoned == damage_commit
        @test failed.state.z == 0.0 && failed.field == ones(ComplexF64, 3, 2)
        @test failed.ledger.source_evaluations == 2
        @test failed.ledger.accepted_steps == failed.attempts.count == 0
        @test failing.accepted == 0
    end
end

@testset "Adams snapshot stop and admission preserve common lifecycle" begin
    session = adams_session_fixture(; replay = true, capacity = 2)
    @test SP.request_snapshot!(session) == (1, false)
    @test SP.request_snapshot!(session) == (1, true)
    adams_session_ack_launch(session)
    report = SP.advance!(session; deadline = 60.0)
    @test report.kind == SP.SESSION_SNAPSHOT && session.state.accepted == 1
    @test_throws ArgumentError SP.advance!(session)
    @test_throws ArgumentError SP.acknowledge!(session, report.generation + 1)
    SP.acknowledge!(session, report.generation)
    @test session.attempts.count == 0
    GC.@preserve session begin
        ccall(
            Ptr{Cvoid}(SP.stop_setter_address()), Cvoid, (Ptr{UInt8},),
            Ptr{UInt8}(SP.stop_address(session.stop)),
        )
    end
    stopped = SP.advance!(session)
    @test stopped.kind == SP.SESSION_STOPPED
    @test stopped.attempts == 1 && session.ledger.predictor == 1
    SP.close_session!(session)
    SP.close_session!(session)
    @test_throws ArgumentError SP.advance!(session)

    @test_throws ArgumentError adams_session_fixture(capacity = 0)
    interpolated = adams_session_policy(interpolate = true, save_planes = Float64[])
    schedule = SP.MarchSchedule(;
        planes = [0.2], breakpoints = [0.0, 0.4], hard_kinks = Float64[],
        stretch = 1.1, step_planes = 0,
    )
    prepared = adams_session_fixture()
    @test_throws ArgumentError SP.AdamsSession(
        prepared.execution, prepared.field, 0.0, 0.01, interpolated,
        prepared.norm, schedule, prepared.source, prepared.flow,
    )
end

@testset "landed Adams owns bounded adaptive and replay histories" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))), replay in (false, true)
        whole = adams_session_fixture(; tasks, replay)
        sliced = adams_session_fixture(; tasks, replay, capacity = 2)
        rows, events, _ = adams_session_collect(whole)
        pieces, boundaries, _ = adams_session_collect(
            sliced; work_limit = 1, deadline = 0.0,
        )
        @test whole.state.failure_message == ""
        @test last(events)[1] == last(boundaries)[1] == SP.SESSION_COMPLETED
        @test length(rows) == length(pieces)
        @test map(row -> (row[1], row[2], row[4], row[5]), rows) ==
            map(row -> (row[1], row[2], row[4], row[5]), pieces)
        # Delivery boundaries change no scalar operations. A modest operation-count
        # allowance also covers the explicitly permitted compiler contraction.
        @test maximum(abs.(whole.field - sliced.field)) <= 128eps(Float64)
        @test all(
            abs(first[3] - second[3]) <= 128eps(Float64) * max(1.0, abs(first[3]))
                for (first, second) in zip(rows, pieces)
        )
        @test whole.control.held <= whole.policy.order + 1
        @test count(whole.history.occupied) == whole.control.held
        @test whole.state.attempts == length(rows)
        @test whole.ledger.accepted_steps == whole.state.accepted
        @test whole.ledger.rejected_steps == whole.state.rejected
        @test whole.ledger.rejected == whole.ledger.rejected_steps
        @test whole.ledger.predictor == whole.state.attempts
        @test whole.source.calls == whole.ledger.source_evaluations ==
            SP.adams_charged_calls(whole.ledger)
        @test whole.ledger.corrected == whole.ledger.dense_observations == 0
        @test count(event -> event[1] == SP.SESSION_PLANE, events) == 2
        @test any(event -> event[1] == SP.SESSION_PROGRESS, boundaries)
        # A source affine in z integrates to this quadratic at every accepted step,
        # including the first order-one corrector; no ODE convergence error remains.
        z = whole.state.z
        expected = 1 + (0.2 + 0.1im) * z + (-0.3 + 0.2im) * z^2 / 2
        @test maximum(abs.(whole.field .- expected)) <= 4096eps(Float64)
        replay && @test length(rows) == 8 && whole.state.rejected == 0
    end
end

@testset "Adams histories retain predictor samples and snapped source coordinates" begin
    policy = adams_session_policy(;
        order = 1, variable_order = false, z_end = 0.3, save_planes = [0.3],
    )
    session = adams_session_fixture(;
        policy, replay = true, n_steps = 3, initial = ones(ComplexF64, 1, 1),
        gain = 1.0 + 0im, intercept = 0.0 + 0im, slope = 0.0 + 0im,
    )
    adams_session_ack_launch(session)
    predicted, state, sample = 0.0 + 0im, 1.0 + 0im, 1.0 + 0im
    for step in 1:3
        h = session.schedule.replay_steps[step]
        predicted = state + h * sample
        next_sample = predicted
        state += h * (sample + next_sample) / 2
        sample = next_sample
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        @test report.kind in (SP.SESSION_PROGRESS, SP.SESSION_PLANE)
        @test abs(session.field[1] - state) <= 256eps(Float64) * abs(state)
        slot = session.slots[session.control.held]
        @test abs(session.history.source[1, 1, slot] - sample) <=
            256eps(Float64) * abs(sample)
        @test session.positions[session.control.held] ==
            session.work.position + session.work.step
        @test session.history.positions[slot] == session.positions[session.control.held]
        @test session.history.source[1, 1, slot] != session.field[1]
        SP.acknowledge!(session, report.generation)
    end
    @test session.control.held == 2
end

@testset "Adams hard kinks restart while future kinks preserve shortened runs" begin
    for replay in (false, true)
        policy = adams_session_policy(hard_kinks = [0.1, 0.3])
        session = adams_session_fixture(; policy, replay, source_absent = true)
        rows, events, _ = adams_session_collect(session)
        @test last(events)[1] == SP.SESSION_COMPLETED
        @test session.ledger.restarts == 1
        @test session.ledger.startup == 2
        @test session.ledger.predictor == length(rows)
        @test session.ledger.source_evaluations == session.source.calls == 0
        @test session.ledger.rejected_steps == 0
        @test all(row -> iszero(row[3]), rows)
        @test session.field == ones(ComplexF64, 3, 2)
        @test length(session.positions) == policy.order + 1
        @test all(>=(0.1), session.positions[1:session.control.held])
    end
end

@testset "Adams charged budgets precede startup even for absent sources" begin
    for absent in (false, true)
        session = adams_session_fixture(;
            policy = adams_session_policy(max_calls = 1), source_absent = absent,
        )
        adams_session_ack_launch(session)
        report = SP.advance!(session)
        @test report.kind == SP.SESSION_FAILED && !report.poisoned
        @test session.state.failure_code == :AdamsBudgetError
        @test session.ledger.startup == session.ledger.predictor == 0
        @test session.source.calls == 0 && session.state.z == 0.0
        @test session.field == ones(ComplexF64, 3, 2)

        paid = adams_session_fixture(;
            policy = adams_session_policy(max_calls = 2), source_absent = absent,
            replay = true,
        )
        rows, events, _ = adams_session_collect(paid)
        @test last(events)[1] == SP.SESSION_FAILED
        @test length(rows) == paid.ledger.accepted_steps == 1
        @test SP.adams_charged_calls(paid.ledger) == 2
        @test paid.ledger.source_evaluations == (absent ? 0 : 2)
    end
end

@testset "Adams rejection either retains history or pays for a restart" begin
    for threshold in (1, 3)
        policy = adams_session_policy(restart_after = threshold)
        session = adams_session_fixture(; policy, nonfinite_on = 3, capacity = 1)
        adams_session_ack_launch(session)
        first = SP.advance!(session; work_limit = 1, deadline = 60.0)
        @test first.kind == SP.SESSION_PROGRESS
        accepted = copy(session.field)
        position = session.state.z
        stored = copy(session.history.source)
        held = session.control.held
        SP.acknowledge!(session, first.generation)
        rejected = SP.advance!(session; work_limit = 1, deadline = 60.0)
        @test rejected.kind == SP.SESSION_PROGRESS
        @test session.attempts.count == 1 && !session.attempts.accepted[1]
        @test session.field == accepted && session.state.z == position
        @test session.history.source == stored
        @test session.ledger.rejected == session.ledger.rejected_steps == 1
        @test session.control.last_rejected
        @test session.control.held == (threshold == 1 ? 0 : held)
        @test session.ledger.failure_restarts == (threshold == 1 ? 1 : 0)
        SP.acknowledge!(session, rejected.generation)
        resumed = SP.advance!(session; work_limit = 1, deadline = 60.0)
        @test resumed.kind in (SP.SESSION_PROGRESS, SP.SESSION_PLANE)
        @test session.ledger.startup == (threshold == 1 ? 2 : 1)
        @test session.ledger.source_evaluations == SP.adams_charged_calls(session.ledger)
        @test session.ledger.accepted_steps == 2
    end
end

@testset "Adams rejection and accepted floors preserve complete failure facts" begin
    rejected = adams_session_fixture(;
        policy = adams_session_policy(;
            order = 1, variable_order = false, dz_min = 0.2, dz_max = 0.5,
            z_end = 0.5, save_planes = [0.5], breakpoints = [0.0, 0.5],
        ),
        initial = ones(ComplexF64, 1, 1), gain = 1.0 + 0im,
        intercept = 0.0 + 0im, slope = 0.0 + 0im, proposal = 0.25, rtol = 1 / 128,
    )
    adams_session_ack_launch(rejected)
    report = SP.advance!(rejected)
    @test report.kind == SP.SESSION_FAILED && !report.poisoned
    @test rejected.state.failure_code == :StepFloorError
    @test rejected.state.z == 0.0 && rejected.field[1] == 1.0
    @test rejected.ledger.startup == rejected.ledger.predictor ==
        rejected.ledger.rejected == rejected.ledger.rejected_steps == 1
    @test rejected.control.held == 1 && rejected.control.last_rejected
    @test rejected.state.proposal == 0.25
    # Host/native omit this completed row (#651). Julia's shared actual-work
    # record includes it, without changing their controller/ledger/history result.
    @test rejected.attempts.count == 1 && !rejected.attempts.accepted[1]
    @test rejected.attempts.norm[1] == 4.0 && rejected.attempts.dz[1] == 0.25

    accepted = adams_session_fixture(;
        policy = adams_session_policy(;
            order = 1, variable_order = false, dz_min = 0.2, dz_max = 0.5,
            max_growth = 1.5, z_end = 0.125, save_planes = [0.125],
        ),
        initial = ones(ComplexF64, 1, 1), gain = 1.0 + 0im,
        intercept = 0.0 + 0im, slope = 0.0 + 0im, proposal = 0.25, rtol = 1.0,
    )
    adams_session_ack_launch(accepted)
    report = SP.advance!(accepted)
    @test report.kind == SP.SESSION_FAILED && !report.poisoned
    @test accepted.state.failure_code == :StepFloorError
    @test accepted.state.z == 0.125 && accepted.field[1] == 1.1328125
    @test accepted.attempts.count == accepted.ledger.accepted_steps == 1
    @test accepted.attempts.accepted[1] && accepted.control.held == 2
end

@testset "Adams source failures retain the last accepted state" begin
    for fail_on in (1, 2)
        session = adams_session_fixture(; fail_on)
        adams_session_ack_launch(session)
        report = SP.advance!(session)
        @test report.kind == SP.SESSION_FAILED && !report.poisoned
        @test session.ledger.startup == 1
        @test session.ledger.predictor == fail_on - 1
        @test session.ledger.source_evaluations == session.source.calls == fail_on
        @test session.attempts.count == 0
        @test session.field == ones(ComplexF64, 3, 2) && session.state.z == 0.0
        @test session.control.held == fail_on - 1
    end
    replay = adams_session_fixture(; replay = true, nonfinite_on = 2)
    adams_session_ack_launch(replay)
    report = SP.advance!(replay)
    @test report.kind == SP.SESSION_FAILED && !report.poisoned
    @test replay.state.failure_code == :DomainError
    @test replay.ledger.rejected_steps == replay.attempts.count == 1
    @test replay.ledger.source_evaluations == 2 && replay.state.z == 0.0

    # Fault injection after preparation: the predictor charge precedes exponent
    # work, but a failed exponent never invokes the predictor's physical source.
    exponent_failure = adams_session_fixture()
    fill!(exponent_failure.flow.operator.rate, NaN)
    adams_session_ack_launch(exponent_failure)
    failed = SP.advance!(exponent_failure)
    @test failed.kind == SP.SESSION_FAILED && !failed.poisoned
    @test exponent_failure.state.failure_code == :NonfiniteSourceError
    @test SP.adams_charged_calls(exponent_failure.ledger) == 2
    @test exponent_failure.ledger.source_evaluations == exponent_failure.source.calls == 1
    @test exponent_failure.attempts.count == 0 && exponent_failure.state.z == 0.0

    for budget_failure in (false, true)
        restarted = adams_session_fixture(;
            policy = adams_session_policy(;
                hard_kinks = [0.1], max_calls = budget_failure ? 5 : nothing,
            ),
            replay = true, fail_on = budget_failure ? 0 : 6,
        )
        _, events, _ = adams_session_collect(restarted)
        @test last(events)[1] == SP.SESSION_FAILED
        @test restarted.state.z == restarted.state.flight_z == 0.1
        @test restarted.state.flight_dz == 0.0 && !restarted.state.in_flight
        @test !restarted.state.poisoned && restarted.control.held == 0
        @test restarted.ledger.predictor == 4 && restarted.ledger.restarts == 1
        @test restarted.ledger.startup == (budget_failure ? 1 : 2)
        @test restarted.ledger.source_evaluations == (budget_failure ? 5 : 6)
    end
end

@testset "Adams homogeneous represented flow is exact and hard masked" begin
    for varying in (false, true)
        initial = reshape(ComplexF64[1, 2, 3, 4, 5, 6], 3, 2)
        active = [true, false, true]
        session = adams_session_fixture(;
            initial, active, varying, rate = -0.7 + 0.2im,
            source_absent = true, replay = true,
        )
        rows, events, _ = adams_session_collect(session)
        @test last(events)[1] == SP.SESSION_COMPLETED
        @test !session.work.compensated && !session.history.compensated
        @test session.ledger.source_evaluations == 0
        @test all(row -> row[3] == 0.0, rows)
        distance = varying ? session.state.z^2 / 2 : session.state.z
        expected = initial .* exp((-0.7 + 0.2im) * distance)
        expected[.!active, :] .= 0
        # Eight short scalar exponentials and, for Phi, endpoint Clenshaw differences.
        @test maximum(abs.(session.field - expected)) <=
            4096eps(Float64) * maximum(abs, initial)
        @test all(iszero, session.field[.!active, :])
    end
end

function adams_session_one_advance(session)
    report = SP.advance!(session; work_limit = 1, deadline = 60.0)
    SP.acknowledge!(session, report.generation)
    return nothing
end

function adams_session_allocations(session)
    adams_session_ack_launch(session)
    adams_session_one_advance(session)
    adams_session_one_advance(session)
    return @allocated adams_session_one_advance(session)
end

@testset "warmed Adams advancement allocates no storage" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))), varying in (false, true)
        session = adams_session_fixture(;
            tasks, varying, rate = -0.4 + 0.2im, replay = true,
        )
        @test adams_session_allocations(session) == 0
        @test session.work.compensated == varying
        @test session.control.held <= length(session.positions)
    end
end

function adams_modal_session(tasks)
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    n_time = 8
    grid = SP.ModalSourceGrid(
        execution; kind = :envelope, n_time, dt = 1.0,
        crop_index = collect(0:(n_time - 1)), omega = ones(n_time),
        active = fill(true, n_time), omega_window = ones(n_time),
        source_window = ones(n_time), source_window_derivative = zeros(n_time),
    )
    map = SP.ModalSourceMap(execution; synthesis = ones(1, 1), projection = ones(1, 1))
    kerr = SP.ModalKerr(SP.KerrCoefficients(4 / 3, :analytic), SP.SourceProfile(1.0))
    data = SP.ModalSourceData(
        grid, map, SP.SourcePrefactor(execution, fill(-0.5im, n_time, 1)); kerr,
    )
    source = SP.ModalSource(execution, data; node_chunk = 1)
    initial = zeros(ComplexF64, n_time, 1)
    initial[1] = n_time * (1 + 0.3im)
    flow = SP.LinearFlow(
        execution, SP.DiagonalOperator(execution, zeros(ComplexF64, n_time, 1)), initial,
    )
    norm = SP.MarchNorm(
        execution, fill(true, n_time), fill(true, n_time), ones(n_time), 1.0e-4, 0.0,
    )
    policy = adams_session_policy(order = 1, variable_order = false, save_planes = [0.2])
    steps = fill(0.025, 8)
    schedule = SP.MarchSchedule(;
        planes = policy.save_planes, breakpoints = policy.breakpoints,
        hard_kinks = policy.hard_kinks, stretch = policy.stretch, step_planes = 0,
        replay_steps = steps, replay_edges = [0.0; cumsum(steps)],
    )
    return SP.AdamsSession(
        execution, initial, 0.0, 0.025, policy, norm, schedule, source, flow,
    )
end

@testset "Adams composes modal Kerr FFTs with retained PEC samples" begin
    # A DC raw-DFT bin is n_time times its constant sample. The 4/3 Kerr
    # coefficient cancels the scalar analytic factor 3/4, giving a'=-i|a|²a/2.
    # This independent two-register recurrence retains the predictor source.
    state = 1.0 + 0.3im
    sample = -0.5im * abs2(state) * state
    largest = abs(state)
    for _ in 1:8
        predictor = state + 0.025sample
        next_sample = -0.5im * abs2(predictor) * predictor
        state += 0.0125 * (sample + next_sample)
        sample = next_sample
        largest = max(largest, abs(predictor), abs(state))
    end
    @test largest < 1.1
    # On |a|<=1.1 the cubic's real-Jacobian norm is at most 3*1.1²/2<2.
    # gamma_4096 covers eight small DFT/map/cubic/PEC evaluations; exp(2T)
    # carries their absolute contributions through the scalar recurrence.
    gamma = 4096eps(Float64) / (1 - 4096eps(Float64))
    bound = gamma * 8largest * exp(2 * 0.2)
    @test bound > 0
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        session = adams_modal_session(tasks)
        @test adams_session_allocations(session) == 0
        _, events, _ = adams_session_collect(session)
        @test last(events)[1] == SP.SESSION_COMPLETED
        @test abs(session.field[1] - 8state) <= bound
        @test all(iszero, view(session.field, 2:8, :))
        @test session.ledger.startup == 1 && session.ledger.predictor == 8
        @test session.ledger.source_evaluations == session.source.calls == 9
        latest = session.slots[session.control.held]
        @test abs(session.history.source[1, 1, latest] - 8sample) <= bound
    end
end
