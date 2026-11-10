# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import SolpropJulia as SP
import LinearAlgebra

mutable struct AdamsObservationSource{E}
    execution::E
    gain::ComplexF64
    calls::Int
end

function (source::AdamsObservationSource)(output, field, position)
    source.calls += 1
    gain = source.gain
    SP.each!(source.execution, size(field)...) do row, column
        output[row, column] = position^3 + gain * field[row, column]
    end
    return nothing
end

function adams_observation_fixture(; tasks = 1, compensated = false, replay = false)
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
    initial = fill(1.0 + 2.0im, 3, 2)
    policy = SP.AdamsControlPolicy(;
        order = 4, variable_order = true, safety = 0.8, max_growth = 2.0,
        min_factor = 0.2, rejection_growth = 0.9, hysteresis = 1.2, restart_after = 3,
        collar_stability = 0.7, real_intervals = (2.0, 1.0, 0.5, 0.3, 0.2, 0.15),
        collar_rate = 0.0, carrier = false, carrier_cap = 4.0, max_calls = nothing,
        equal_approach = true, interval_cap = false, dz_min = 1.0e-10, dz_max = 0.5,
        stretch = 1.1, z_end = 1.5, save_planes = [1.0, 1.5],
        hard_kinks = Float64[], breakpoints = [0.0, 2.0], landing_ulps = 8,
        interpolate = false, quiet_interval = nothing,
    )
    operator = if compensated
        # On [0,2], Phi(z)=a*z+b*z^2/2 has these independent Chebyshev coefficients.
        a, b = -0.2 + 0.3im, 0.15 - 0.2im
        coefficients = zeros(ComplexF64, length(initial), 3, 1)
        coefficients[:, 1, 1] .= a + 0.75b
        coefficients[:, 2, 1] .= a + b
        coefficients[:, 3, 1] .= 0.25b
        SP.RepresentedOperator(
            execution, SP.ChebyshevTable(execution, [0.0, 2.0], coefficients), 3, 2,
        )
    else
        SP.DiagonalOperator(execution, zeros(ComplexF64, 3, 2))
    end
    norm = SP.MarchNorm(execution, fill(true, 3), fill(true, 3), ones(3), 1.0e-3, 0.0)
    steps = replay ? fill(0.25, 6) : Float64[]
    edges = replay ? [0.0; cumsum(steps)] : Float64[]
    schedule = SP.MarchSchedule(;
        planes = policy.save_planes, breakpoints = policy.breakpoints,
        hard_kinks = policy.hard_kinks, stretch = policy.stretch,
        step_planes = 0, replay_steps = steps, replay_edges = edges,
    )
    source = AdamsObservationSource(execution, compensated ? 0.02 - 0.01im : 0.0im, 0)
    session = SP.AdamsSession(
        execution, initial, 0.5, 0.25, policy, norm, schedule,
        source, SP.LinearFlow(execution, operator, initial); plane_pending = false,
    )
    for slot in 1:3
        sampled = initial .* (1 + slot / 11)
        position = 0.5 - (slot - 1) / 4
        SP.adams_seed!(
            session.history, slot, sampled, position, session.sampler,
            session.rate, execution; identity = 1,
        )
        session.positions[4 - slot] = position
        session.slots[4 - slot] = slot
    end
    session.control.held, session.control.order = 3, 3
    SP.adams_stencil!(session.stencil, 3, (1, 2, 3), (3,))
    SP.adams_trial!(
        session.work, session.field, 0.5, 0.25, session.history, session.stencil,
        session.sampler, session.exponent, session.rate, execution,
    )
    observation = SP.AdamsObservationWork(execution, initial, 4; compensated)
    return session, observation
end

function adams_observation_record(session)
    return (
        copy(session.field), copy(session.positions), copy(session.slots),
        Tuple(
            copy(getfield(session.history, name)) for name in (
                    :source, :states, :rates, :positions, :identities, :occupied,
                )
        ),
        Tuple(
            getfield(session.control, name) for name in fieldnames(typeof(session.control))
        ),
        Tuple(getfield(session.state, name) for name in fieldnames(typeof(session.state))),
        Tuple(
            getfield(session.ledger, name) for name in fieldnames(typeof(session.ledger))
        ),
        session.source.calls,
    )
end

function adams_observation_cycle!(observation, session, destination)
    SP.retain_adams_observation!(observation, session, destination)
    SP.observe_adams!(observation, 0.6, session.norm.active, session.execution)
    return nothing
end

@testset "Dense observation retains independently owned polynomial data" begin
    for tasks in (1, 2)
        session, observation = adams_observation_fixture(; tasks)
        before = adams_observation_record(session)
        @test !observation.ready
        @test SP.retain_adams_observation!(observation, session, 0.75) === nothing
        @test observation.ready && observation.destination == 0.75
        @test iszero(observation.field)
        @test isequal(adams_observation_record(session), before)
        @test !Base.mightalias(observation.dense.initial, session.field)
        @test !Base.mightalias(observation.dense.endpoint, session.work.candidate)
        @test !Base.mightalias(observation.dense.source, session.history.source)
        initial = copy(session.field)
        for position in (0.5, 0.525, 0.6, 0.7, 0.75)
            output = SP.observe_adams!(
                observation, position, session.norm.active, session.execution,
            )
            @test output === observation.field
            # The exact degree-three source integrates to (z^4-z0^4)/4. A 512eps
            # state-scale budget covers the fixed four-node coefficient/phi path.
            bound = 512eps(Float64) * maximum(abs, initial)
            @test bound > 0
            @test maximum(abs, output - (initial .+ (position^4 - 0.5^4) / 4)) <= bound
        end
        @test isequal(adams_observation_record(session), before)
        retained = copy(observation.field)
        fill!(session.history.source, NaN)
        fill!(session.work.source, NaN)
        fill!(session.work.candidate, NaN)
        changed = adams_observation_record(session)
        SP.observe_adams!(observation, 0.75, session.norm.active, session.execution)
        @test isequal(observation.field, retained)
        SP.observe_adams!(observation, 0.6, session.norm.active, session.execution)
        @test all(isfinite, observation.field)
        @test isequal(adams_observation_record(session), changed)
    end
end

function adams_observation_moment(exponent, degree, upper)
    # Defining series for integral_0^upper exp((upper-t)*exponent)*t^degree dt.
    term = upper^(degree + 1) / (degree + 1)
    total = Complex{BigFloat}(term)
    for index in 1:200
        term *= exponent * upper / (degree + index + 1)
        total += term
        abs(term) <= eps(BigFloat) * max(abs(total), one(BigFloat)) && break
    end
    return total
end

function adams_observation_reference(dense, position)
    count = dense.order + 1
    step = BigFloat(dense.step)
    fraction = (BigFloat(position) - BigFloat(dense.position)) / step
    nodes = BigFloat.(dense.nodes[1:count])
    vandermonde = [node^power for node in nodes, power in 0:(count - 1)]
    basis = vandermonde \ Matrix{BigFloat}(LinearAlgebra.I, count, count)
    condition = LinearAlgebra.opnorm(vandermonde, Inf) * LinearAlgebra.opnorm(basis, Inf)
    phase = Complex{BigFloat}(dense.exponent[1])
    initial = Complex{BigFloat}(dense.initial[1])
    samples = Complex{BigFloat}[dense.source[1, 1, index] for index in 1:count]
    coefficients = basis * samples
    baseline = exp(fraction * phase) * initial
    source, scale = zero(baseline), abs(baseline) + abs(initial)
    for power in 0:(count - 1)
        moment = adams_observation_moment(phase, power, fraction)
        source += coefficients[power + 1] * moment
        scale += step * abs(moment) * sum(
            abs(basis[power + 1, sample]) * abs(samples[sample]) for sample in 1:count
        )
    end
    compensation = zero(source)
    for sample in 1:count
        weight = sum(
            basis[power, sample] * fraction^power / power for power in 1:count
        )
        absolute_weight = sum(
            abs(basis[power, sample]) * fraction^power / power for power in 1:count
        )
        rate = Complex{BigFloat}(dense.rates[1, 1, sample])
        state = Complex{BigFloat}(dense.states[1, 1, sample])
        transported = exp((fraction - nodes[sample]) * phase)
        compensation += weight * transported * ((rate - phase / step) * state)
        # Preserve cancellation in weights and in the compensated rate subtraction.
        scale += step * absolute_weight * abs(transported) *
            (abs(rate) + abs(phase / step)) * abs(state)
    end
    gamma = BigFloat(8192) * eps(Float64) / (1 - BigFloat(8192) * eps(Float64))
    return baseline + step * (source + compensation), gamma * condition * scale
end

@testset "Compensated dense observations match independent high-precision moments" begin
    for tasks in (1, 2)
        session, observation = adams_observation_fixture(; tasks, compensated = true)
        before = adams_observation_record(session)
        SP.retain_adams_observation!(observation, session, 0.75)
        setprecision(BigFloat, 256) do
            for position in (0.525, 0.6, 0.7)
                expected, bound = adams_observation_reference(observation.dense, position)
                actual = SP.observe_adams!(
                    observation, position, session.norm.active, session.execution,
                )
                # Fixed four-node Vandermonde conditioning and absolute propagated
                # contributions multiply a predeclared 8192-operation rounding budget.
                @test bound > 0
                @test maximum(abs, actual .- expected) <= bound
            end
        end
        @test isequal(adams_observation_record(session), before)
        # The existing protected strong-damping branch ignores discarded source/rate
        # data before forming inverse endpoint transport (ADR-039).
        fill!(observation.dense.exponent, -2000)
        fill!(observation.dense.source, NaN)
        fill!(observation.dense.rates, NaN)
        output = SP.observe_adams!(
            observation, 0.51, session.norm.active, session.execution,
        )
        expected = exp((0.51 - 0.5) / 0.25 * -2000) .* observation.dense.initial
        @test all(isfinite, output)
        @test maximum(abs, output - expected) <= 8eps(Float64) * maximum(abs, expected)
    end
end

@testset "Canonical landing copies both rounding directions without extrapolation" begin
    for tasks in (1, 2), replay in (false, true), direction in (-1, 1)
        session, observation = adams_observation_fixture(; tasks, replay)
        raw_end = session.work.position + session.work.step
        destination = direction > 0 ? nextfloat(raw_end, 2) : prevfloat(raw_end, 2)
        session.field[1, 1] = ComplexF64(0.0, -0.0)
        session.work.candidate[1, 1] = ComplexF64(-0.0, 0.0)
        SP.retain_adams_observation!(observation, session, destination)
        before = adams_observation_record(session)
        @test SP.observe_adams!(
            observation, destination, session.norm.active, session.execution,
        ) === observation.field
        @test isequal(observation.field, session.work.candidate)
        SP.observe_adams!(observation, 0.5, session.norm.active, session.execution)
        @test isequal(observation.field, session.field)
        if direction > 0
            SP.observe_adams!(
                observation, nextfloat(raw_end), session.norm.active, session.execution,
            )
            @test isequal(observation.field, session.work.candidate)
        else
            @test_throws ArgumentError SP.observe_adams!(
                observation, raw_end, session.norm.active, session.execution,
            )
        end
        @test isequal(adams_observation_record(session), before)
        @test_throws ArgumentError SP.retain_adams_observation!(
            observation, session, nextfloat(raw_end, 64),
        )
        @test observation.destination == destination
        @test observation.ready
    end
end

@testset "Dense admission and masking preserve failed observations as failures" begin
    session, observation = adams_observation_fixture()
    @test_throws ArgumentError SP.AdamsObservationWork(session.execution, session.field, 0)
    @test_throws ArgumentError SP.AdamsObservationWork(
        session.execution, zeros(ComplexF64, 0, 2), 4,
    )
    @test_throws ArgumentError SP.observe_adams!(
        observation, 0.6, session.norm.active, session.execution,
    )
    SP.retain_adams_observation!(observation, session, 0.75)
    for point in (NaN, Inf, prevfloat(0.5), nextfloat(0.75))
        before = copy(observation.field)
        @test_throws ArgumentError SP.observe_adams!(
            observation, point, session.norm.active, session.execution,
        )
        @test isequal(observation.field, before)
    end
    @test_throws DimensionMismatch SP.observe_adams!(
        observation, 0.6, fill(true, 2), session.execution,
    )
    @test_throws ArgumentError SP.observe_adams!(
        observation, 0.6, trues(3), session.execution,
    )
    # A reinterpreted mask may alias numerical storage even though its dtype differs.
    mask_alias = view(reinterpret(Bool, vec(observation.field)), 1:3)
    @test_throws ArgumentError SP.observe_adams!(
        observation, 0.6, mask_alias, session.execution,
    )
    observed = observation.field
    observation.field = observation.dense.initial
    retained = copy(observation.dense.initial)
    @test_throws ArgumentError SP.observe_adams!(
        observation, 0.75, session.norm.active, session.execution,
    )
    @test isequal(observation.dense.initial, retained)
    observation.field = observed
    mask = [true, false, true]
    output = SP.observe_adams!(observation, 0.6, mask, session.execution)
    @test iszero(output[2, :])
    @test all(isfinite, output)
    observation.dense.source[2, 1, 1] = ComplexF64(Inf)
    before = adams_observation_record(session)
    @test_throws SP.NonfiniteSourceError SP.observe_adams!(
        observation, 0.6, mask, session.execution,
    )
    @test isequal(adams_observation_record(session), before)
    session.work.candidate[1, 1] = ComplexF64(NaN)
    @test_throws SP.NonfiniteSourceError SP.retain_adams_observation!(
        observation, session, 0.75,
    )
    @test !observation.ready
    @test_throws ArgumentError SP.observe_adams!(
        observation, 0.6, mask, session.execution,
    )
end

@testset "Dense retention precedes history commit and overwrites inactive samples" begin
    for compensated in (false, true)
        session, observation = adams_observation_fixture(; compensated)
        SP.retain_adams_observation!(observation, session, 0.75)
        @test iszero(observation.dense.source[:, :, 5])
        SP.adams_stencil!(session.stencil, 1, (1, 2, 3), (1,))
        SP.adams_trial!(
            session.work, session.field, 0.5, 0.25, session.history, session.stencil,
            session.sampler, session.exponent, session.rate, session.execution,
        )
        SP.retain_adams_observation!(observation, session, 0.75)
        @test iszero(observation.dense.source[:, :, 3:5])
        @test iszero(observation.dense.weights)
        if compensated
            @test iszero(observation.dense.states[:, :, 3:5])
            @test iszero(observation.dense.rates[:, :, 3:5])
        end
        session.state.z = 0.75
        before = adams_observation_record(session)
        @test_throws ArgumentError SP.retain_adams_observation!(observation, session, 0.75)
        @test isequal(adams_observation_record(session), before)
    end
end

@testset "Warmed dense retention and observation allocate no storage" begin
    for tasks in (1, 2), compensated in (false, true)
        session, observation = adams_observation_fixture(; tasks, compensated)
        adams_observation_cycle!(observation, session, 0.75)
        before = adams_observation_record(session)
        @test @allocated(adams_observation_cycle!(observation, session, 0.75)) == 0
        @test isequal(adams_observation_record(session), before)
    end
end
