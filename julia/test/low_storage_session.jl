# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import SolpropJulia as SP

mutable struct LowStorageSessionTestSource{E}
    execution::E
    gain::ComplexF64
    calls::Int
    fail_on::Int
    nonfinite_on::Int
    nonfinite_row::Int
    invalid_return::Bool
    positions::Vector{Float64}
end

function (source::LowStorageSessionTestSource)(output, field, position)
    source.calls += 1
    source.positions[source.calls] = position
    source.calls == source.fail_on && error("controlled low-storage source failure")
    gain = source.gain
    failed = source.calls == source.nonfinite_on
    failed_row = source.nonfinite_row
    SP.each!(source.execution, size(output)...) do row, column
        output[row, column] = failed && row == failed_row ?
            ComplexF64(Inf) : gain * field[row, column]
    end
    return source.invalid_return ? 1 : nothing
end

function low_storage_session_fixture(;
        tasks = 1, capacity = 256, n_steps = 8, z = 0.0, stop = 0.2,
        initial = ones(ComplexF64, 3, 2), rate = -0.4 + 0.2im, gain = 0.3 - 0.1im,
        varying = false, active = nothing, observed = nothing, planes = [stop],
        hard_kinks = Float64[], step_planes = 0, services = SP._NoSessionServices(),
        fail_on = 0, nonfinite_on = 0, nonfinite_row = 1, invalid_return = false,
        fixed_steps = nothing, plane_index = nothing, plane_pending = true,
    )
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    steps = fixed_steps === nothing ? fill((stop - z) / n_steps, n_steps) : fixed_steps
    edges = z .+ [0.0; cumsum(steps)]
    source = LowStorageSessionTestSource(
        execution, ComplexF64(gain), 0, fail_on, nonfinite_on, nonfinite_row,
        invalid_return, zeros(4length(steps)),
    )
    if varying
        # Phi(z)=rate*z²/2 represented directly on [0,.4], hence D(z)=rate*z.
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
    norm_mask = observed === nothing ? mask : observed
    norm = SP.MarchNorm(execution, mask, norm_mask, ones(size(initial, 1)), 1.0e-4, 0.0)
    schedule = SP.MarchSchedule(;
        planes, breakpoints = [min(0.0, z), max(0.4, stop)], hard_kinks,
        stretch = 1.0, step_planes, replay_steps = steps, replay_edges = edges,
    )
    return SP.LowStorageSession(
        execution, initial, z, SP.RK4IPTableau(), norm, schedule, source, flow;
        capacity, services, plane_index, plane_pending,
    )
end

function low_storage_session_ack_launch(session)
    report = SP.advance!(session)
    @test report.kind == SP.SESSION_LAUNCH
    @test session.counts.attempted_source_evaluations == 0
    SP.acknowledge!(session, report.generation)
    return nothing
end

function low_storage_session_collect(session; work_limit = 256, deadline = 60.0)
    rows = Tuple{Float64, Float64, Float64, Bool}[]
    events = Tuple{SP.SessionEvent, Float64}[]
    reports = SP.SessionReport[]
    iterations = 0
    while !session.state.finished
        iterations += 1
        iterations <= 100 || error("low-storage test exceeded its bounded event allowance")
        report = SP.advance!(session; work_limit, deadline)
        push!(events, (report.kind, report.z))
        push!(reports, report)
        buffer = session.attempts
        @test buffer.count <= length(buffer.z)
        for index in 1:buffer.count
            push!(
                rows, (
                    buffer.z[index], buffer.dz[index], buffer.norm[index],
                    buffer.accepted[index],
                )
            )
        end
        SP.acknowledge!(session, report.generation)
    end
    return rows, events, reports
end

@testset "Low-storage session admits only the fixed recorded itinerary" begin
    session = low_storage_session_fixture()
    original = copy(session.field)
    adaptive = SP.MarchSchedule(;
        planes = [0.2], breakpoints = [0.0, 0.4], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0,
    )
    @test_throws ArgumentError SP.LowStorageSession(
        session.execution, session.field, 0.0, SP.RK4IPTableau(), session.norm,
        adaptive, session.source, session.flow,
    )
    @test_throws ArgumentError SP.LowStorageSession(
        session.execution, session.field, 0.0, SP.RK4IPTableau(), session.norm,
        session.schedule, session.source, session.flow; capacity = 0,
    )
    @test_throws ArgumentError SP.LowStorageSession(
        session.execution, session.field, 0.0, SP.RK4IPTableau(), session.norm,
        session.schedule, session.source, nothing,
    )
    @test_throws DimensionMismatch SP.LowStorageSession(
        session.execution, view(session.field, :, 1:1), 0.0, SP.RK4IPTableau(),
        session.norm, session.schedule, session.source, session.flow,
    )
    @test session.field == original && session.source.calls == 0

    caller = reshape(ComplexF64.(1:6), 3, 2)
    owned = low_storage_session_fixture(initial = caller, active = [true, false, true])
    @test !Base.mightalias(owned.field, caller)
    @test all(iszero, view(owned.field, 2, :))
    caller .= 99
    @test owned.field[1] == 1
    @test !hasproperty(owned, :history) && !hasproperty(owned, :control)
end

@testset "Low-storage fixed sessions match independent scalar RK4 flows" begin
    initial = reshape(ComplexF64[1 + 0.3im, 0.4 - 0.2im, -0.7 + 0.1im], 3, 1)
    rate, gain = -0.4 + 0.2im, 0.3 - 0.1im
    for tasks in unique((1, min(2, Threads.nthreads(:default)))), varying in (false, true)
        for n_steps in (1, 8)
            session = low_storage_session_fixture(; tasks, initial, varying, n_steps)
            rows, events, reports = low_storage_session_collect(session)
            @test last(events)[1] == SP.SESSION_COMPLETED
            @test length(rows) == session.state.accepted == n_steps
            @test session.state.rejected == 0
            @test all(row -> row[3] == 0.0 && row[4], rows)
            @test all(report -> report.met_tolerance && !report.poisoned, reports)
            # A commuting linear source becomes u'=gain*u in the midpoint frame.
            # Its RK4 stability polynomial is independent of the register passes.
            h = 0.2 / n_steps
            x = h * gain
            polynomial = 1 + x + x^2 / 2 + x^3 / 6 + x^4 / 24
            integrated = varying ? rate * 0.2^2 / 2 : rate * 0.2
            expected = initial .* (exp(integrated) * polynomial^n_steps)
            # Fixed gamma covers eight small four-stage complex steps and represented
            # endpoint differences; exp supplies the scalar amplification envelope.
            gamma = 8192eps(Float64) / (1 - 8192eps(Float64))
            bound = gamma * maximum(abs, initial) * exp(0.2 * (abs(rate) + abs(gain)))
            @test bound > 0
            @test maximum(abs.(session.field - expected)) <= bound
            @test session.counts.attempted_source_evaluations == 4n_steps
            @test session.counts.committed_source_evaluations == 4n_steps
            @test session.counts.norm_evaluations == n_steps
            @test session.source.calls == 4n_steps
            @test !session.state.in_flight
        end
    end
end

@testset "Low-storage chunking and deadlines never alter the itinerary" begin
    whole = low_storage_session_fixture(planes = [0.1, 0.2])
    chunked = low_storage_session_fixture(capacity = 1, planes = [0.1, 0.2])
    rows, events, _ = low_storage_session_collect(whole)
    split_rows, split_events, _ = low_storage_session_collect(
        chunked; work_limit = 1, deadline = 0.0,
    )
    @test split_rows == rows
    @test filter(row -> row[1] != SP.SESSION_PROGRESS, split_events) == events
    # Identical numerical work is compared with a fixed rounding allowance, not bits.
    @test maximum(abs.(whole.field - chunked.field)) <= 256eps(Float64)
    @test whole.source.calls == chunked.source.calls == 32
    @test length(chunked.attempts.z) == 1
    @test length(chunked.work.source) == length(whole.work.source)

    snapped = low_storage_session_fixture(;
        z = 1.0, stop = 1.3, fixed_steps = fill(0.1, 3),
        hard_kinks = [1.2], rate = 0.0 + 0im, gain = 0.0 + 0im,
    )
    low_storage_session_ack_launch(snapped)
    for step in 1:3
        origin = snapped.state.z
        report = SP.advance!(snapped; work_limit = 1, deadline = 60.0)
        offset = 4(step - 1)
        @test snapped.source.positions[(offset + 1):(offset + 4)] ==
            [origin, origin + 0.05, origin + 0.05, origin + 0.1]
        @test snapped.state.z == snapped.schedule.replay_edges[step + 1]
        @test snapped.attempts.dz[1] == 0.1
        @test snapped.attempts.norm[1] == 0.0
        SP.acknowledge!(snapped, report.generation)
    end
    @test snapped.source.positions[8] != snapped.schedule.replay_edges[3]
    @test snapped.field == ones(ComplexF64, 3, 2)
end

mutable struct LowStorageSessionTestServices
    failure::Symbol
    factor::Float64
    finished::Int
    accepted::Int
    caps::Int
    same_register::Bool
    destination::Float64
end

function SP._session_launch!(services::LowStorageSessionTestServices, session)
    services.failure == :launch && error("controlled low-storage launch failure")
    return nothing
end

SP._session_observe!(::LowStorageSessionTestServices, session, kind, plane) = nothing
SP._session_finish!(::LowStorageSessionTestServices, session) = nothing
SP._session_quiet(::LowStorageSessionTestServices, session) = true
SP._session_quiet_interval(::LowStorageSessionTestServices, session) = 1.0e-9
SP._session_barrier(::LowStorageSessionTestServices, session) = :none
SP._session_continue!(::LowStorageSessionTestServices, session) = :none
SP._session_acknowledge!(::LowStorageSessionTestServices, session, generation) = nothing

function SP._session_cap(services::LowStorageSessionTestServices, session, proposal)
    services.caps += 1
    return min(proposal, 1.0e-9)
end

function SP._session_finish_candidate!(
        services::LowStorageSessionTestServices, session, candidate, dz, destination,
    )
    services.finished += 1
    services.same_register &= candidate === session.field
    services.destination = destination
    for index in eachindex(candidate)
        candidate[index] *= services.factor
        services.failure == :finish && error("controlled low-storage finish failure")
    end
    services.failure == :nonfinite_finish && (candidate[1] = NaN)
    return nothing
end

function SP._session_accepted!(services::LowStorageSessionTestServices, session, dz)
    services.accepted += 1
    services.failure == :accepted && error("controlled low-storage accepted failure")
    return nothing
end

@testset "Low-storage finishing and services respect the in-flight boundary" begin
    for failure in (:none, :launch, :finish, :nonfinite_finish, :accepted)
        services = LowStorageSessionTestServices(failure, 0.9, 0, 0, 0, true, NaN)
        session = low_storage_session_fixture(;
            services, n_steps = 1, rate = 0.0 + 0im, gain = 0.0 + 0im,
            active = [true, false, true],
        )
        rows, events, _ = low_storage_session_collect(session)
        @test services.caps == 0
        @test services.same_register
        if failure == :none
            @test last(events)[1] == SP.SESSION_COMPLETED
            @test session.field == repeat(ComplexF64[0.9, 0, 0.9], 1, 2)
            @test services.finished == services.accepted == 1
            @test services.destination == session.state.z == 0.2
        else
            @test last(events)[1] == SP.SESSION_FAILED
            in_flight = failure in (:finish, :nonfinite_finish)
            @test session.state.poisoned == session.state.in_flight == in_flight
            committed = failure == :accepted
            @test session.counts.committed_source_evaluations == (committed ? 4 : 0)
            @test session.state.z == (committed ? 0.2 : 0.0)
            @test length(rows) == (committed ? 1 : 0)
            @test session.counts.attempted_source_evaluations ==
                (failure == :launch ? 0 : 4)
        end
    end
end

@testset "Low-storage source failures preserve exact attempted-call facts" begin
    for failed_call in 1:4
        session = low_storage_session_fixture(fail_on = failed_call)
        low_storage_session_ack_launch(session)
        report = SP.advance!(session)
        @test report.kind == SP.SESSION_FAILED && report.poisoned
        @test session.state.in_flight && session.state.flight_z == 0.0
        @test session.state.flight_dz == 0.025 && session.state.z == 0.0
        @test session.counts.attempted_source_evaluations ==
            session.source.calls == failed_call
        @test session.counts.committed_source_evaluations == 0
        @test session.counts.norm_evaluations == 0
        @test session.attempts.count == 0
        failed_call == 1 && @test session.field == ones(ComplexF64, 3, 2)
        detached = fill(7.0 + 0im, size(session.field))
        @test_throws ArgumentError SP.export_session!(
            detached, session, report.generation, ones(size(session.field)), false, false,
        )
        @test all(==(7), detached)
        @test_throws ArgumentError SP.advance!(session)
    end
    invalid = low_storage_session_fixture(invalid_return = true)
    _, events, _ = low_storage_session_collect(invalid)
    @test last(events)[1] == SP.SESSION_FAILED && invalid.state.poisoned
    @test invalid.state.failure_code == :ArgumentError
    @test invalid.counts.attempted_source_evaluations == 1

    nonfinite = low_storage_session_fixture(;
        nonfinite_on = 4, nonfinite_row = 2, observed = [true, false, true],
    )
    _, events, _ = low_storage_session_collect(nonfinite)
    @test last(events)[1] == SP.SESSION_FAILED && nonfinite.state.poisoned
    @test nonfinite.state.failure_code == :DomainError
    @test nonfinite.counts.attempted_source_evaluations == 4
    @test nonfinite.counts.committed_source_evaluations == 0
    @test nonfinite.counts.norm_evaluations == 1
    @test nonfinite.state.z == 0.0 && nonfinite.attempts.count == 0

    after_commit = low_storage_session_fixture(fail_on = 6)
    rows, events, _ = low_storage_session_collect(after_commit)
    @test last(events)[1] == SP.SESSION_FAILED && after_commit.state.poisoned
    @test length(rows) == 1 && only(rows)[4]
    @test after_commit.state.z == after_commit.state.flight_z == 0.025
    @test after_commit.counts.attempted_source_evaluations == 6
    @test after_commit.counts.committed_source_evaluations == 4
    @test after_commit.counts.norm_evaluations == 1
end

@testset "Low-storage pre-trial failures leave accepted storage eligible" begin
    crossing = low_storage_session_fixture(n_steps = 1, planes = [0.1, 0.2])
    low_storage_session_ack_launch(crossing)
    report = SP.advance!(crossing)
    @test report.kind == SP.SESSION_FAILED && !report.poisoned
    @test crossing.state.flight_z == crossing.state.flight_dz == 0.0
    @test crossing.counts.attempted_source_evaluations == 0
    @test crossing.field == ones(ComplexF64, 3, 2)
    detached = similar(crossing.field)
    SP.export_session!(
        detached, crossing, report.generation, ones(size(detached)), false, false,
    )
    @test detached == crossing.field

    already = low_storage_session_fixture(;
        z = 0.1, stop = 0.2, n_steps = 4, plane_pending = false,
    )
    report = SP.advance!(already; work_limit = 1, deadline = 60.0)
    @test report.kind == SP.SESSION_PROGRESS
    @test already.counts.attempted_source_evaluations == 4
    @test already.state.z == already.schedule.replay_edges[2]

    at_end = SP.LowStorageSession(
        crossing.execution, crossing.field, 0.2, SP.RK4IPTableau(), crossing.norm,
        crossing.schedule, crossing.source, crossing.flow;
        plane_index = 2, plane_pending = false,
    )
    completed = SP.advance!(at_end)
    @test completed.kind == SP.SESSION_COMPLETED
    @test at_end.counts.attempted_source_evaluations == 0
    @test at_end.counts.committed_source_evaluations == at_end.attempts.count == 0
end

@testset "Low-storage snapshots and stops occur only at eligible boundaries" begin
    session = low_storage_session_fixture()
    sequence, coalesced = SP.request_snapshot!(session)
    @test sequence == 1 && !coalesced
    @test SP.request_snapshot!(session) == (1, true)
    low_storage_session_ack_launch(session)
    report = SP.advance!(session)
    @test report.kind == SP.SESSION_SNAPSHOT && session.state.accepted == 1
    @test session.counts.committed_source_evaluations == 4
    @test_throws ArgumentError SP.acknowledge!(session, report.generation + 1)
    SP.acknowledge!(session, report.generation)
    @test_throws ArgumentError SP.acknowledge!(session, report.generation)
    flag = session.stop
    setter = SP.stop_setter_address()
    GC.@preserve flag ccall(
        Ptr{Cvoid}(setter), Cvoid, (Ptr{UInt8},), Ptr{UInt8}(SP.stop_address(flag)),
    )
    stopped = SP.advance!(session)
    @test stopped.kind == SP.SESSION_STOPPED && !stopped.poisoned
    @test session.counts.attempted_source_evaluations == 4
    SP.acknowledge!(session, stopped.generation)
    SP.close_session!(session)
    @test_throws ArgumentError SP.advance!(session)
end

function low_storage_session_one_advance(session)
    report = SP.advance!(session; work_limit = 1, deadline = 60.0)
    SP.acknowledge!(session, report.generation)
    return nothing
end

function low_storage_session_allocations(session)
    low_storage_session_ack_launch(session)
    low_storage_session_one_advance(session)
    low_storage_session_one_advance(session)
    return @allocated low_storage_session_one_advance(session)
end

@testset "Low-storage advancement allocates no storage and protects large norms" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))), varying in (false, true)
        session = low_storage_session_fixture(; tasks, varying)
        @test low_storage_session_allocations(session) == 0
        @test session.counts.attempted_source_evaluations == 12
        @test session.counts.committed_source_evaluations == 12
        @test session.counts.norm_evaluations == 3
    end
    for amplitude in (1.0e-250, 1.0e250)
        initial = fill(complex(amplitude, amplitude), 3, 2)
        session = low_storage_session_fixture(;
            initial, gain = 0.0 + 0im, rate = 0.0 + 0im,
        )
        _, events, _ = low_storage_session_collect(session)
        @test last(events)[1] == SP.SESSION_COMPLETED
        @test all(isfinite, session.field)
        @test maximum(abs.(session.field - initial)) <=
            256eps(Float64) * maximum(abs, initial)
        @test session.counts.committed_source_evaluations == 32
    end
end

function low_storage_modal_session(tasks; observed = false)
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    n_time = 8
    omega, time = collect(1.0:n_time), collect(0.0:(n_time - 1))
    grid = SP.ModalSourceGrid(
        execution; kind = :envelope, n_time, dt = 1.0,
        crop_index = collect(0:(n_time - 1)), omega, active = fill(true, n_time),
        omega_window = ones(n_time), source_window = ones(n_time),
        source_window_derivative = zeros(n_time),
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
    services = if observed
        selection = SP.StatisticsSelection(
            (SP.StatisticSlot(:energy, 1, 1), SP.StatisticSlot(:peak_power_total, 2, 1)),
            (1, 2),
        )
        prepared = SP.ObservationPreparation(
            execution, SP.StatisticsGrid(omega, time, fill(1 / n_time, n_time)), selection;
            kind = :envelope, n_modes = 1,
            power_weight = SP.ObservationWeight(execution, ones(n_time, 1)),
        )
        policy = SP.SessionObservationPolicy(prepared; split_rate = fill(0.3, n_time))
        SP.ObservationServices(execution, policy; attempt_capacity = 8, node_chunk = 1)
    else
        SP._NoSessionServices()
    end
    steps = fill(0.025, 8)
    schedule = SP.MarchSchedule(;
        planes = [0.2], breakpoints = [0.0, 0.2], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0,
        replay_steps = steps, replay_edges = [0.0; cumsum(steps)],
    )
    return SP.LowStorageSession(
        execution, initial, 0.0, SP.RK4IPTableau(), norm, schedule, source, flow;
        services, capacity = 8,
    )
end

function low_storage_scalar_reference(observed)
    state = 1.0 + 0.3im
    result = ComplexF64[]
    for _ in 1:8
        h = 0.025
        first = -0.5im * abs2(state) * state
        second_state = state + h * first / 2
        second = -0.5im * abs2(second_state) * second_state
        third_state = state + h * second / 2
        third = -0.5im * abs2(third_state) * third_state
        fourth_state = state + h * third
        fourth = -0.5im * abs2(fourth_state) * fourth_state
        state += h / 6 * (first + 2second + 2third + fourth)
        observed && (state *= exp(-0.3h / 2))
        push!(result, state)
    end
    return result
end

@testset "Low-storage composes modal FFT Kerr and accepted observation services" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))), observed in (false, true)
        reference = low_storage_scalar_reference(observed)
        session = low_storage_modal_session(tasks; observed)
        rows, indices, energies, powers = Float64[], Int[], Float64[], Float64[]
        while !session.state.finished
            report = SP.advance!(session; work_limit = 1, deadline = 60.0)
            @test report.kind != SP.SESSION_FAILED
            if observed
                payload = SP.observation_rows(session.services)
                for index in 1:payload.count
                    push!(rows, payload.z[index])
                    push!(indices, payload.step[index])
                    push!(energies, payload.values[1, index])
                    push!(powers, payload.values[2, index])
                end
            end
            SP.acknowledge!(session, report.generation)
        end
        # The direct RK4 scalar recurrence above contains no FFTs, modal maps or
        # register rewrites. On |a|<=1.1, its cubic Jacobian is bounded by 2.
        # This fixed gamma covers eight tiny transform/source/stage/split passes;
        # exp(2T) propagates absolute rounding contributions through the recurrence.
        gamma = 16384eps(Float64) / (1 - 16384eps(Float64))
        bound = gamma * 8 * 1.1 * exp(2 * 0.2)
        @test bound > 0
        @test abs(session.field[1] - 8last(reference)) <= bound
        @test all(iszero, view(session.field, 2:8, :))
        @test session.source.calls == session.counts.attempted_source_evaluations == 32
        @test session.counts.committed_source_evaluations == 32
        if observed
            @test session.services.accepted_seen == 8
            @test length(rows) == 8 && all(==(0), indices)
            @test rows == session.schedule.replay_edges[2:end]
            # Power differentiates |a|²; energy adds the eight unit-time samples.
            @test maximum(abs.(powers - abs2.(reference))) <= 4bound
            @test maximum(abs.(energies - 8abs2.(reference))) <= 32bound
        end
        warmed = low_storage_modal_session(tasks; observed)
        @test low_storage_session_allocations(warmed) == 0
    end
end
