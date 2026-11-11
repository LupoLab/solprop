# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
using LinearAlgebra: I
import SolpropJulia as DA

dfadams_big(x::DA.DF32) = BigFloat(x.hi) + BigFloat(x.lo)
dfadams_big(x::Complex{DA.DF32}) =
    Complex{BigFloat}(dfadams_big(real(x)), dfadams_big(imag(x)))
dfadams_big(x::DA._DF32AdamsScaled) =
    dfadams_big(DA._da_value(x.value)) * BigFloat(2)^x.exponent

function dfadams_phi(x::Complex{BigFloat}, order)
    iszero(order) && return exp(x)
    if abs(x) > 3
        result = exp(x)
        for k in 1:order
            result = (result - inv(BigFloat(factorial(k - 1)))) / x
        end
        return result
    end
    term = inv(BigFloat(factorial(order))) + zero(x)
    total = term
    for k in 1:160
        term *= x / (order + k)
        total += term
    end
    return total
end

function dfadams_rate_snapshot(record)
    record === nothing && return nothing
    sample = record.sample
    return (;
        saved = copy(record.saved), words = copy(sample.values),
        errors = copy(sample.errors), numerical = copy(record.numerical),
        record.origin_wide, sample.position, sample.before, sample.provenance,
        sample.generation,
    )
end

function dfadams_snapshot(fixture)
    session, flow = fixture.session, fixture.flow
    history = session.history
    provenance = map(dfadams_rate_snapshot, flow.records)
    active_bank = if iszero(flow.plans.active)
        nothing
    else
        bank = flow.plans.banks[flow.plans.active]
        (;
            host = map(copy, DA._da_coefficient_arrays(bank.host)),
            checked = map(copy, DA._da_coefficient_arrays(bank.checked)),
            numerical = map(copy, DA._da_coefficient_arrays(bank.numerical)),
            positions = copy(bank.positions), kinds = copy(bank.kinds),
            generations = copy(bank.generations), weights = copy(bank.weights),
            normalization = copy(bank.normalization), bank.certificate,
            operations = copy(bank.operations[1:(bank.stencil.count + 1)]),
            bundles = copy(bank.bundles[1:(bank.stencil.count + 1)]),
            seed = dfadams_rate_snapshot(bank.seed_rate),
            endpoint = dfadams_rate_snapshot(bank.endpoint_rate),
            bank.step_checked, bank.step_numerical,
        )
    end
    return (;
        field = copy(session.field), source = copy(history.source),
        states = copy(history.states), rates = copy(history.rates),
        positions = copy(history.positions), occupied = copy(history.occupied),
        identities = copy(history.identities),
        work = map(copy, DA._da_work_arrays(session.work)),
        ready = session.work.ready, active = flow.plans.active,
        generation = flow.plans.generation, plan_ready = flow.plans.ready,
        provenance, active_bank, calls = fixture.source.calls,
        ledger = Tuple(
            getfield(session.ledger, name)
                for name in fieldnames(DA.AdamsCallLedger)
        ),
    )
end

function dfadams_unchanged(fixture, saved)
    current = dfadams_snapshot(fixture)
    @test isequal(current, saved)
    return nothing
end

function dfadams_ball(value::ComplexF64; error = 0.0)
    return Complex{DA._WideBall}(
        DA._wide_split(DA._WideHost(real(value), error)),
        DA._wide_split(DA._WideHost(imag(value), error)),
    )
end

function dfadams_basis(nodes)
    count = length(nodes)
    vandermonde = [nodes[row]^(column - 1) for row in 1:count, column in 1:count]
    return transpose(vandermonde \ Matrix{eltype(nodes)}(I, count, count))
end

@testset "DF32 complete factors enclose independent polynomial integrals" begin
    setprecision(BigFloat, 256) do
        for order in 1:6, moved in (false, true)
            origin, step = 0.125, 1 / 128
            positions = [origin - (index - 1) * 3step / 4 for index in 1:order]
            moved && (positions[1] = nextfloat(origin))
            stencil = DA.AdamsStencil(6)
            DA.adams_stencil!(stencil, order, collect(1:order), [order])
            host = DA._DF32AdamsCoefficients(6, DA._wide_host(0.0))
            checked = DA._DF32AdamsCoefficients(6, DA._da_host(DA._wide_host(0.0)))
            numerical = DA._DF32AdamsCoefficients(
                6,
                DA._da_compute(DA._da_host(DA._wide_host(0.0)))
            )
            DA._da_coefficients!(host, stencil, positions, origin, step)
            DA._da_split_coefficients!(
                DA._da_coefficient_arrays(host),
                DA._da_coefficient_arrays(checked), DA._da_coefficient_arrays(numerical)
            )
            nodes = (Rational{BigInt}.(positions) .- Rational{BigInt}(origin)) ./
                Rational{BigInt}(step)
            for (polynomial, coefficients, count) in (
                    (dfadams_basis(nodes), checked.predictor, order),
                    (
                        dfadams_basis(vcat(one(eltype(nodes)), nodes)),
                        checked.corrector, order + 1,
                    ),
                )
                for sample in 1:count, power in 1:count
                    value = coefficients[sample, power]
                    error = ldexp(BigFloat(real(value.value).error), value.exponent)
                    @test error > 0
                    @test abs(real(dfadams_big(value)) - polynomial[sample, power]) <= error
                end
                for argument in (-0.5 + 0.25im, 0.0im, 2pi * im, 3.01im)
                    phase = dfadams_ball(argument)
                    moments = DA._da_moments(DA._da_phi(phase))
                    independent = [
                        factorial(k - 1) *
                            dfadams_phi(Complex{BigFloat}(argument), k) for k in 1:count
                    ]
                    for sample in 1:count
                        factor = DA._da_source_factor(
                            coefficients, moments, 1, 1,
                            sample, count, DA._da_host(DA._wide_host(step))
                        )
                        bound = DA._da_narrow_error(factor)
                        reference = BigFloat(step) * sum(
                            polynomial[sample, k] * independent[k] for k in 1:count
                        )
                        @test bound > 0
                        @test abs(
                            Complex{BigFloat}(
                                DA._da_narrow(
                                    DA._da_compute(factor)
                                )
                            ) - reference
                        ) <= bound
                    end
                end
            end
        end
        # An individual h or rate is outside binary32 range while the complete
        # coefficient is ordinary. Mantissa/exponent products retain both operands.
        for h in (1.0e-300, 1.0e300), sign in (-1.0, 1.0)
            q = dfadams_ball(0.0im)
            raw = ComplexF64(sign / h, -sign / (2h))
            sampled = DA._da_complex_host(raw, 0.0im)
            factor = DA._da_compensation_factor(
                DA._da_host(DA._wide_host(0.5)),
                q, q, sampled, DA._da_host(DA._wide_host(h)), zero(DA._WideBall)
            )
            bound = DA._da_narrow_error(factor)
            expected = BigFloat(0.5) * BigFloat(h) * Complex{BigFloat}(raw)
            @test bound > 0 && abs(
                Complex{BigFloat}(
                    DA._da_narrow(DA._da_compute(factor))
                ) - expected
            ) <= bound
        end
        # Cancellation does not waive an inadmissible transport argument.
        q = dfadams_ball(1.0 + 0im)
        sampled = DA._da_complex_host(1.0 + 0im, 0.0im)
        @test_throws ArgumentError DA._da_compensation_factor(
            DA._da_host(DA._wide_host(0.0)), q, q, sampled,
            DA._da_host(DA._wide_host(1.0)), DA._wide_split(DA._wide_host(17.0))
        )
    end
end


@testset "Complete DF32 factors retain normalized operands and absolute errors" begin
    setprecision(BigFloat, 256) do
        for value in (0.0, 0.5, -0.5, 1.0e300, 1.0e-300, nextfloat(0.0))
            factor = DA._da_host(DA._wide_host(value))
            error = ldexp(BigFloat(real(factor.value).error), factor.exponent)
            @test error >= 0 && abs(real(dfadams_big(factor)) - BigFloat(value)) <= error
        end
        for step in (1.0e-300, 1.0e-200, 1.0e200, 1.0e300)
            weight = DA._da_host(DA._wide_host(inv(step)))
            scalar = DA._da_host(DA._wide_host(step))
            complete = DA._da_mul(scalar, weight)
            expected = BigFloat(step) * BigFloat(inv(step))
            bound = DA._da_narrow_error(complete)
            @test bound > 0
            actual = Complex{BigFloat}(DA._da_narrow(DA._da_compute(complete)))
            @test abs(actual - expected) <= bound
        end
        for rate in (1.0e-300, 1.0e300)
            step = DA._da_host(DA._wide_host(inv(rate)))
            sampled = DA._da_complex_host(ComplexF64(rate), 0.0im)
            zero_phase = dfadams_ball(0.0im)
            coefficient = DA._da_host(DA._wide_host(1.0))
            complete = DA._da_compensation_factor(
                coefficient, zero_phase,
                zero_phase, sampled, step, zero(DA._WideBall)
            )
            expected = BigFloat(rate) * BigFloat(inv(rate))
            bound = DA._da_narrow_error(complete)
            @test bound > 0 && abs(
                Complex{BigFloat}(
                    DA._da_narrow(DA._da_compute(complete))
                ) - expected
            ) <= bound
        end
        for argument in ComplexF64[
                    0, 2.0^-80 * (1 + im), 0.125 + 0.25im, -0.5 + 0.5im,
                    2.99im, 3.0im, nextfloat(3.0) * im, 3.01im,
                    2pi * im, -16 + 0.5im, 16 + 0.5im, 512im,
                ], inherited in (0.0, 2.0^-34)
            ball = dfadams_ball(argument; error = inherited)
            computed = DA._da_phi(ball)
            for order in 0:7, direction in (-1, 0, 1)
                target = Complex{BigFloat}(argument) +
                    direction * BigFloat(inherited) * (1 + im)
                reference = dfadams_phi(target, order)
                actual = computed[order + 1]
                # The implementation exposes componentwise absolute certificates;
                # MPFR's independent recurrence/series error is below 2^-180 here.
                bound = BigFloat(real(actual).error + imag(actual).error) + BigFloat(2)^-180
                @test bound > 0
                @test abs(dfadams_big(DA._da_value(actual)) - reference) <= bound
            end
        end
        @test_throws ArgumentError DA._da_phi(dfadams_ball(17.0 + 0im))
        @test_throws ArgumentError DA._da_phi(dfadams_ball(0.0im; error = 2.0^-26))
    end
end

mutable struct DFAdamsSource{E}
    execution::E
    gain::ComplexF32
    forcing::ComplexF32
    calls::Int
    fail_at::Int
    data::Nothing
end

function (source::DFAdamsSource)(output, field, position)
    source.calls += 1
    source.calls == source.fail_at && error("controlled DF32 Adams source failure")
    gain, forcing = source.gain, source.forcing
    DA.each!(source.execution, size(field)...) do row, column
        output[row, column] = gain * field[row, column] + forcing
    end
    return nothing
end

function dfadams_policy(planes; order = 1, variable_order = false, changes...)
    base = (;
        order, variable_order, safety = 0.8, max_growth = 2.0,
        min_factor = 0.2, rejection_growth = 0.9, hysteresis = 1.2, restart_after = 6,
        collar_stability = 0.7, real_intervals = (2.0, 1.0, 0.5, 0.3, 0.2, 0.15),
        collar_rate = 0.0, carrier = false, carrier_cap = 4.0, max_calls = nothing,
        equal_approach = false, interval_cap = false, dz_min = 1.0e-10,
        dz_max = 1 / 128, stretch = 1.0, z_end = last(planes), save_planes = planes,
        hard_kinks = Float64[], breakpoints = [0.0, 0.25], landing_ulps = 8,
        interpolate = false, quiet_interval = nothing,
    )
    return DA.AdamsControlPolicy(; merge(base, (; changes...))...)
end

function dfadams_fixture(;
        tasks = 1, order = 1, varying = false, source_absent = false,
        replay = true, gain = 0.0im, forcing = 0.125 + 0.0625im, rtol = 0.01,
        initial = nothing, rate = -0.125 + 0.25im, max_calls = nothing,
        hard_kinks = Float64[]
    )
    execution = DA.KernelExecution(DA.KernelAbstractions.CPU(), tasks, true)
    state = initial === nothing ? ComplexF32[
            complex(row / 8, column / 16) for row in 1:4, column in 1:2
        ] : initial
    operator = if varying
        coefficients = zeros(ComplexF64, length(state), 3, 1)
        coefficients[:, 1, 1] .= 3rate / 256
        coefficients[:, 2, 1] .= rate / 64
        coefficients[:, 3, 1] .= rate / 256
        DA.RepresentedOperator(
            execution,
            DA.ChebyshevTable(execution, [0.0, 0.25], coefficients), size(state)...
        )
    else
        DA.DiagonalOperator(execution, fill(ComplexF64(rate), size(state)))
    end
    planes = [1 / 16, 1 / 8]
    policy = dfadams_policy(planes; order, variable_order = !replay, max_calls, hard_kinks)
    schedule = DA.MarchSchedule(;
        planes, policy.breakpoints, policy.hard_kinks,
        policy.stretch, step_planes = 0,
        replay_steps = replay ? fill(1 / 64, 8) : Float64[],
        replay_edges = replay ? collect(0:8) ./ 64 : Float64[]
    )
    arithmetic = DA.SourceArithmetic(Float32)
    spec = DA.WideStartSpec(
        0.0, nothing, schedule;
        geometry = DA.WideGeometry(0.0, 0.25; step_max = 1 / 16), arithmetic,
        method = :adams, order_max = order, source_absent
    )
    flow = DA.DF32AdamsFlow(execution, operator, state, spec, policy)
    source = DFAdamsSource(execution, ComplexF32(gain), ComplexF32(forcing), 0, 0, nothing)
    norm = DA.MarchNorm(execution, fill(true, 4), fill(true, 4), ones(4), rtol, 0.0)
    session = DA.AdamsSession(
        execution, state, 0.0, 1 / 64, policy, norm, schedule,
        source, flow; arithmetic, source_absent
    )
    return (; execution, state, operator, policy, schedule, spec, flow, source, session)
end

function dfadams_drain!(session; work_limit = 256)
    turns = 0
    while !session.state.finished
        turns += 1
        turns <= 512 || error("DF32 Adams short fixture exceeded its bounded window")
        report = DA.advance!(session; work_limit, deadline = 60.0)
        DA.acknowledge!(session, report.generation)
    end
    return nothing
end

function dfadams_turn!(session)
    report = DA.advance!(session; work_limit = 1, deadline = 60.0)
    DA.acknowledge!(session, report.generation)
    return nothing
end

function dfadams_allocations(session)
    dfadams_turn!(session)
    return @allocated dfadams_turn!(session)
end

@testset "Certified DF32 Adams executes actual short plain and compensated sessions" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))), order in 1:6,
            varying in (false, true)
        fixture = dfadams_fixture(; tasks, order, varying)
        dfadams_drain!(fixture.session; work_limit = 1)
        @test fixture.session.state.z == 1 / 8
        @test fixture.source.calls == 9
        @test fixture.session.ledger.accepted_steps == 8
        @test eltype(fixture.session.field) === ComplexF32
        @test eltype(fixture.session.work.exponent) === Complex{DA.DF32}
        @test eltype(fixture.session.history.rates) === ComplexF64
        @test all(isfinite, fixture.session.field)
        @test fixture.session.work.factor_context === fixture.flow
        @test fixture.flow.plans.history === fixture.session.history
    end
end

@testset "DF32 short homogeneous/adaptive paths and warmed storage remain explicit" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))), varying in (false, true)
        warm = dfadams_fixture(; tasks, varying, order = 6)
        dfadams_turn!(warm.session)
        @test dfadams_allocations(warm.session) == 0
        @test dfadams_allocations(warm.session) == 0
        absent = dfadams_fixture(; tasks, varying, source_absent = true, order = 6)
        absent.source.fail_at = 1
        dfadams_drain!(absent.session)
        @test absent.source.calls == absent.session.ledger.source_evaluations == 0
        @test absent.session.ledger.startup == 1 && absent.session.ledger.predictor == 8
        @test all(iszero, absent.session.work.defects)
        setprecision(BigFloat, 256) do
            rate = Complex{BigFloat}(-0.125, 0.25)
            phase = varying ? rate * BigFloat(1 / 8)^2 / 2 : rate * BigFloat(1 / 8)
            reference = exp(phase) .* Complex{BigFloat}.(absent.state)
            # Eight narrowed exp factors and complex state products: 16 scalar
            # rounding units per step, plus the declared exp-pair function bound.
            unit = BigFloat(2)^-24
            gamma = 128unit / (1 - 128unit)
            bound = (gamma + 8BigFloat(2)^-27) *
                exp(abs(phase)) * maximum(abs, Complex{BigFloat}.(absent.state))
            @test bound > 0
            difference = Complex{BigFloat}.(absent.session.field) - reference
            @test maximum(abs, difference) <= bound
        end
        adaptive = dfadams_fixture(;
            tasks, varying, replay = false, order = 3,
            gain = 0.25 + 0.125im, rtol = 1.0e-4
        )
        dfadams_drain!(adaptive.session; work_limit = 1)
        @test adaptive.session.state.z == 1 / 8
        @test adaptive.session.ledger.source_evaluations == adaptive.source.calls ==
            adaptive.session.ledger.startup + adaptive.session.ledger.predictor
        @test adaptive.session.ledger.accepted_steps > 0
        for fail_at in (1, 2, 3)
            failed = dfadams_fixture(; tasks, varying)
            failed.source.fail_at = fail_at
            dfadams_drain!(failed.session)
            @test failed.session.state.failure_code === :ErrorException
            @test failed.source.calls == failed.session.ledger.source_evaluations == fail_at
            @test failed.session.state.accepted == (fail_at <= 2 ? 0 : 1)
        end
    end
    initial = reshape(
        ComplexF32[
            complex(0.0f0, -0.0f0), complex(-0.0f0, -0.0f0),
            complex(1.0f0, -0.0f0), complex(-0.0f0, 1.0f0), 1, 2, 3, 4,
        ], 4, 2
    )
    absent = dfadams_fixture(; initial, rate = 0.0im, source_absent = true)
    dfadams_drain!(absent.session)
    @test reinterpret(UInt32, vec(absent.session.field)) ==
        reinterpret(UInt32, vec(initial))
end

@testset "DF32 Adams tokens and pretrial failures preserve accepted storage" begin
    for varying in (false, true)
        f = dfadams_fixture(; varying, order = 2)
        stencil = DA.AdamsStencil(2)
        DA.adams_stencil!(stencil, 1, [1], [1])
        site = DA.DF32TrialSite(0.0, 1 / 64, 1 / 64, 1)
        trial = DA.prepare_df32_adams_trial!(
            f.flow, f.session.history, stencil, site;
            seed = DA.DF32AdamsSeed(1, 0.0)
        )
        @test f.source.calls == 0 && f.session.ledger.startup == 0
        saved = dfadams_snapshot(f)
        for invalid in (
                DA.DF32TrialSite(0.0, 1 / 128, 1 / 64, 1),
                DA.DF32TrialSite(0.0, 1 / 64, 1 / 64, 2),
                DA.DF32TrialSite(0.0, 1 / 64, 1 / 32, 1),
            )
            @test_throws ArgumentError DA.prepare_df32_adams_trial!(
                f.flow, f.session.history, stencil, invalid; seed = DA.DF32AdamsSeed(1, 0.0)
            )
            dfadams_unchanged(f, saved)
        end
        DA._da_seed!(trial, f.session.field, f.session.sampler, f.execution)
        DA.adams_trial!(
            f.session.work, f.session.field, 0.0, 1 / 64,
            f.session.history, stencil, f.session.sampler, trial, nothing, f.execution
        )
        @test f.source.calls == 2 && f.session.work.ready
        saved = dfadams_snapshot(f)
        @test_throws ArgumentError DA.adams_trial!(
            f.session.work, f.session.field,
            0.0, 1 / 32, f.session.history, stencil, f.session.sampler,
            trial, nothing, f.execution
        )
        dfadams_unchanged(f, saved)
        @test_throws ArgumentError DA.adams_trial!(
            f.session.work, f.session.field,
            0.0, 1 / 64, f.session.history, stencil, f.session.sampler,
            f.session.exponent, nothing, f.execution
        )
        dfadams_unchanged(f, saved)
        if varying
            record = f.flow.records[1]
            generation = record.sample.generation
            record.sample.generation += one(UInt64)
            altered = dfadams_snapshot(f)
            @test_throws ArgumentError DA.adams_trial!(
                f.session.work, f.session.field,
                0.0, 1 / 64, f.session.history, stencil, f.session.sampler,
                trial, nothing, f.execution
            )
            dfadams_unchanged(f, altered)
            record.sample.generation = generation
            raw = f.session.history.rates[1]
            f.session.history.rates[1] = raw + 2.0^-40
            altered = dfadams_snapshot(f)
            @test_throws ArgumentError DA.adams_trial!(
                f.session.work, f.session.field,
                0.0, 1 / 64, f.session.history, stencil, f.session.sampler,
                trial, nothing, f.execution
            )
            dfadams_unchanged(f, altered)
            f.session.history.rates[1] = raw
        end
        owners = (
            f.flow.binding.operator.coefficients,
            f.flow.binding.operator.coefficient_errors,
            f.flow.plans.banks[f.flow.plans.active].numerical.corrector,
            f.session.history.source,
        )
        for owner in owners
            previous = copy(owner)
            GC.@preserve owner begin
                alias = unsafe_wrap(
                    Array, Ptr{ComplexF32}(pointer(owner)) + 8,
                    size(f.session.field); own = false
                )
                @test_throws ArgumentError DA.adams_trial!(
                    f.session.work, alias,
                    0.0, 1 / 64, f.session.history, stencil, f.session.sampler,
                    trial, nothing, f.execution
                )
            end
            @test isequal(owner, previous)
            dfadams_unchanged(f, saved)
        end
        original = f.session.field[1]
        f.session.field[1] = ComplexF32(NaN, 0)
        nonfinite = dfadams_snapshot(f)
        @test_throws ArgumentError DA.adams_trial!(
            f.session.work, f.session.field, 0.0, 1 / 64,
            f.session.history, stencil, f.session.sampler, trial, nothing, f.execution
        )
        dfadams_unchanged(f, nonfinite)
        f.session.field[1] = original
        if varying
            generation = f.flow.records[2].sample.generation
            f.flow.records[2].sample.generation = typemax(UInt64)
            before_commit = dfadams_snapshot(f)
            @test_throws ArgumentError DA.adams_accept!(
                f.session.history, 2, f.session.work, f.execution; identity = 1
            )
            dfadams_unchanged(f, before_commit)
            @test f.flow.records[2].sample.generation == typemax(UInt64)
            f.flow.records[2].sample.generation = generation
        end
        # Acceptance consumes the plan; the same token cannot replay a completed trial.
        DA.adams_accept!(f.session.history, 2, f.session.work, f.execution; identity = 1)
        accepted = dfadams_snapshot(f)
        @test_throws ArgumentError DA.adams_trial!(
            f.session.work, f.session.field,
            0.0, 1 / 64, f.session.history, stencil, f.session.sampler,
            trial, nothing, f.execution
        )
        dfadams_unchanged(f, accepted)
        @test f.session.history.positions[2] == 1 / 64
        @test f.session.history.identities[2] == 1
    end
    fresh = dfadams_fixture(; varying = true)
    fresh.flow.records[1].sample.generation = typemax(UInt64)
    dfadams_drain!(fresh.session)
    @test fresh.source.calls == fresh.session.ledger.startup == 0
    @test fresh.session.state.failure_code === :ArgumentError
    @test fresh.session.field == fresh.state && !any(fresh.session.history.occupied)
    outside = dfadams_fixture(; rate = 1.0e8im)
    dfadams_drain!(outside.session)
    @test outside.source.calls == outside.session.ledger.startup == 0
    @test outside.session.state.failure_code === :ArgumentError
    @test outside.session.field == outside.state && !any(outside.session.history.occupied)
end

@testset "DF32 Adams expansion retains actual and canonical replay coordinates" begin
    execution = DA.KernelExecution(DA.KernelAbstractions.CPU(), 1, true)
    initial = fill(0.5f0 + 0.25f0im, 4, 2)
    # Phi(z)=scale*z² on [0,1], with an exactly represented Chebyshev expansion.
    weights = DA.ChebyshevTable(
        execution, [0.0, 1.0], reshape([3 / 8, 1 / 2, 1 / 8], 1, 3, 1)
    )
    operator = DA.ExpansionOperator(execution, fill(-0.125 + 0.25im, 4, 2, 1), weights)
    policy = dfadams_policy([0.3, 0.4]; order = 2, breakpoints = [0.0, 1.0])
    schedule = DA.MarchSchedule(;
        planes = policy.save_planes, policy.breakpoints, policy.hard_kinks,
        policy.stretch, step_planes = 0, replay_steps = [0.1, 0.1],
        replay_edges = [0.2, 0.3, 0.4]
    )
    arithmetic = DA.SourceArithmetic(Float32)
    spec = DA.WideStartSpec(
        0.2, nothing, schedule; arithmetic, method = :adams, order_max = 2,
        source_absent = false, geometry = DA.WideGeometry(0.0, 1.0; step_max = 0.1)
    )
    flow = DA.DF32AdamsFlow(execution, operator, initial, spec, policy)
    source = DFAdamsSource(execution, 0.0f0im, 0.125f0 + 0.0625f0im, 0, 0, nothing)
    norm = DA.MarchNorm(execution, fill(true, 4), fill(true, 4), ones(4), 0.01, 0.0)
    session = DA.AdamsSession(
        execution, initial, 0.2, 0.1, policy, norm, schedule, source, flow; arithmetic
    )
    @test 0.2 + 0.1 !== 0.3
    @test 0.1 > policy.dz_max
    for turn in 1:4
        session.state.z != 0.2 && break
        dfadams_turn!(session)
    end
    @test session.state.z === 0.3
    @test session.work.position === 0.2 && session.work.step === 0.1
    @test any(==(0.2 + 0.1), session.history.positions[session.history.occupied])
    @test flow.plans.site.destination === 0.3
    @test flow.plans.site.origin + flow.plans.site.step === 0.2 + 0.1
    @test source.calls == 2
    dfadams_drain!(session; work_limit = 1)
    @test session.state.failure_code === :none && session.state.z === 0.4
    @test session.ledger.accepted_steps == 2 && source.calls == 3
    @test session.history.compensated && all(isfinite, session.field)
end

function dfadams_restored(; huge = false, order = 3, discrepancy = 0.0)
    original = dfadams_fixture(;
        varying = true, order,
        rate = huge ? 1.0e8im : -0.125 + 0.25im
    )
    execution, state, operator = original.execution, original.state, original.operator
    start, step = 1 / 8, 1 / 16384
    planes = [start + 4step, start + 8step]
    policy = dfadams_policy(planes; order)
    schedule = DA.MarchSchedule(;
        planes, policy.breakpoints, policy.hard_kinks,
        policy.stretch, step_planes = 0, replay_steps = fill(step, 8),
        replay_edges = start .+ (collect(0:8) .* step)
    )
    positions = start .+ (collect(1:order) .- order) ./ 64
    huge && order > 1 && (positions[1] = start - 0.1)
    slots = collect(0:(order - 1))
    kinds = fill(UInt8(1), order)
    rates = zeros(ComplexF64, size(state)..., order + 2)
    a = huge ? 1.0e8im : -0.125 + 0.25im
    for index in eachindex(positions)
        rates[:, :, slots[index] + 1] .= a * positions[index] + discrepancy
    end
    restored = DA.WideStartHistory(positions, kinds, slots; rates, origin_wide = :binary64)
    spec = DA.WideStartSpec(
        start, nothing, schedule;
        geometry = DA.WideGeometry(0.0, 0.25; step_max = step),
        arithmetic = original.spec.arithmetic, method = :adams, order_max = order,
        source_absent = false,
    )
    flow = DA.DF32AdamsFlow(execution, operator, state, spec, policy; history = restored)
    source = DFAdamsSource(execution, 0.0f0im, ComplexF32(0.125, 0.0625), 0, 0, nothing)
    session = DA.AdamsSession(
        execution, state, start, step, policy, original.session.norm,
        schedule, source, flow; arithmetic = spec.arithmetic, plane_pending = false
    )
    for index in eachindex(positions)
        slot = slots[index] + 1
        session.positions[index], session.slots[index] = positions[index], slot
        session.history.positions[slot] = positions[index]
        session.history.identities[slot] = 1
        session.history.occupied[slot] = true
        session.history.source[:, :, slot] .= source.forcing
        session.history.states[:, :, slot] .= state
    end
    copyto!(session.history.rates, rates)
    session.control.held, session.control.order = order, order
    return (;
        execution, state, operator, policy, schedule, spec, flow, source, session,
        restored,
    )
end

@testset "Restored old history and raw rates bind complete transport certificates" begin
    f = dfadams_restored(; discrepancy = 2.0^-20)
    input = copy(f.restored.rates)
    @test f.flow.binding.origin_wide === :binary64
    for slot in 1:3
        record = f.flow.records[slot]
        @test record.origin_wide === :binary64
        @test record.saved == f.session.history.rates[:, :, slot]
        @test all(real.(record.sample.errors) .>= 2.0^-20)
    end
    dfadams_drain!(f.session; work_limit = 1)
    @test f.session.state.z == last(f.schedule.planes)
    @test f.session.state.failure_code === :none
    @test f.session.ledger.startup == 0 && f.source.calls == 8
    @test f.restored.rates == input
    @test eltype(f.session.history.rates) === ComplexF64
    late = dfadams_restored(; huge = true, order = 2)
    stencil = DA.AdamsStencil(2)
    DA.adams_stencil!(stencil, 1, [2], [1])
    site = DA.DF32TrialSite(
        late.spec.start, first(late.schedule.replay_steps),
        late.schedule.replay_edges[2], 1
    )
    token = DA.prepare_df32_adams_trial!(late.flow, late.session.history, stencil, site)
    DA.adams_stencil!(stencil, 2, [2, 1], [2])
    saved = dfadams_snapshot(late)
    @test_throws ArgumentError DA.prepare_df32_adams_trial!(
        late.flow, late.session.history, stencil, site
    )
    dfadams_unchanged(late, saved)
    @test DA._da_token(token) === late.flow.plans.banks[late.flow.plans.active]
    # Normalized complete factors do not widen the prerequisite stored-rate
    # representation's finite domain. Such a restored rate refuses at preparation.
    rates = copy(f.restored.rates)
    rates[1, 1, 1] = 1.0e300
    unsupported = DA.WideStartHistory(
        f.restored.positions, f.restored.identities, f.restored.slots;
        rates, origin_wide = :binary64
    )
    saved = copy(rates)
    @test_throws ArgumentError DA.DF32AdamsFlow(
        f.execution, f.operator, f.state, f.spec, f.policy; history = unsupported
    )
    @test isequal(rates, saved)
end

@testset "Every Adams order matches an independent complete predictor and corrector" begin
    setprecision(BigFloat, 256) do
        unit = BigFloat(2)^-24
        for order in 1:6
            f = dfadams_restored(; order, discrepancy = 2.0^-30)
            f.source.gain = ComplexF32(-0.125, 0.0625)
            history, work = f.session.history, f.session.work
            stencil = DA.AdamsStencil(order)
            slots = collect(order:-1:1)
            DA.adams_stencil!(stencil, order, slots, [order])
            z, h = f.spec.start, first(f.schedule.replay_steps)
            site = DA.DF32TrialSite(z, h, f.schedule.replay_edges[2], 1)
            token = DA.prepare_df32_adams_trial!(f.flow, history, stencil, site)
            bank = DA._da_token(token)
            certificate = bank.certificate
            positions = history.positions[slots]
            nodes = (Rational{BigInt}.(positions) .- Rational{BigInt}(z)) ./
                Rational{BigInt}(h)
            pred = dfadams_basis(nodes)
            corr = dfadams_basis(vcat(one(eltype(nodes)), nodes))
            a = Complex{BigFloat}(-0.125, 0.25)
            endpoint, step = BigFloat(z + h), BigFloat(h)
            phase = a * (endpoint^2 - BigFloat(z)^2) / 2
            q = phase / step
            moments = [factorial(k - 1) * dfadams_phi(phase, k) for k in 1:(order + 1)]
            source_weight = (p, j) -> step * sum(
                BigFloat(p[j, k]) * moments[k] for k in axes(p, 2)
            )
            integral = (p, j) -> sum(BigFloat(p[j, k]) / k for k in axes(p, 2))
            predicted = similar(Complex{BigFloat}.(f.state))
            corrected = similar(predicted)
            predict_bounds = zeros(BigFloat, size(f.state))
            correct_bounds = similar(predict_bounds)
            # These radii are fixed from input histories and complete certificates
            # before evaluating the numerical trial. gamma covers complex products
            # and sequential state additions; source sensitivity is explicit.
            gamma = 32(order + 2) * unit / (1 - 32(order + 2) * unit)
            for index in eachindex(f.state)
                initial = Complex{BigFloat}(f.state[index])
                baseline = exp(phase) * initial
                linear_error = 16unit * abs(baseline)
                value, contribution, error = baseline, abs(baseline), linear_error
                for sample in 1:order
                    slot = slots[sample]
                    old_source = Complex{BigFloat}(
                        history.source[
                            index +
                                (slot - 1) * length(f.state),
                        ]
                    )
                    old_state = Complex{BigFloat}(
                        history.states[
                            index +
                                (slot - 1) * length(f.state),
                        ]
                    )
                    old_rate = Complex{BigFloat}(
                        history.rates[
                            index +
                                (slot - 1) * length(f.state),
                        ]
                    )
                    pweight = source_weight(pred, sample)
                    cweight = integral(pred, sample) *
                        exp((step - (BigFloat(positions[sample]) - BigFloat(z))) * q) *
                        (step * old_rate - phase)
                    value += pweight * old_source + cweight * old_state
                    contribution += abs(pweight) * abs(old_source) +
                        abs(cweight) * abs(old_state)
                    error += certificate.source_error * abs(old_source) +
                        certificate.compensation_error * abs(old_state)
                end
                predicted[index] = value
                predict_bounds[index] = error + gamma * (contribution + error)
                g = Complex{BigFloat}(f.source.gain)
                forcing = Complex{BigFloat}(f.source.forcing)
                newest = g * value + forcing
                source_error = abs(g) * predict_bounds[index] +
                    (8unit / (1 - 8unit)) *
                    (abs(g) * (abs(value) + predict_bounds[index]) + abs(forcing))
                total, contribution, error = baseline, abs(baseline), linear_error
                endpoint_rate = Complex{BigFloat}(bank.endpoint_rate.saved[index])
                for sample in 1:(order + 1)
                    slot = sample == 1 ? 0 : slots[sample - 1]
                    old_source = sample == 1 ? newest : Complex{BigFloat}(
                            history.source[index + (slot - 1) * length(f.state)]
                        )
                    old_state = sample == 1 ? value : Complex{BigFloat}(
                            history.states[index + (slot - 1) * length(f.state)]
                        )
                    rate = sample == 1 ? endpoint_rate : Complex{BigFloat}(
                            history.rates[index + (slot - 1) * length(f.state)]
                        )
                    transport = sample == 1 ? zero(step) :
                        step - (BigFloat(positions[sample - 1]) - BigFloat(z))
                    pweight = source_weight(corr, sample)
                    cweight = integral(corr, sample) * exp(transport * q) *
                        (step * rate - phase)
                    total += pweight * old_source + cweight * old_state
                    contribution += abs(pweight) * abs(old_source) +
                        abs(cweight) * abs(old_state)
                    error += certificate.source_error * abs(old_source) +
                        certificate.compensation_error * abs(old_state)
                    if sample == 1
                        error += (abs(pweight) + certificate.source_error) * source_error +
                            (abs(cweight) + certificate.compensation_error) *
                            predict_bounds[index]
                    end
                end
                corrected[index] = total
                correct_bounds[index] = error + gamma * (contribution + error)
            end
            @test all(>(0), predict_bounds) && all(>(0), correct_bounds)
            DA.adams_trial!(
                work, f.session.field, z, h, history, stencil,
                f.source, token, nothing, f.execution
            )
            @test all(
                abs.(Complex{BigFloat}.(work.predictor) - predicted) .<=
                    predict_bounds
            )
            @test all(
                abs.(Complex{BigFloat}.(work.candidate) - corrected) .<=
                    correct_bounds
            )
            @test f.source.calls == 1 && f.session.field == f.state
        end
    end
end

function dfadams_material(; tasks = 1)
    execution = DA.KernelExecution(DA.KernelAbstractions.CPU(), tasks, true)
    arithmetic = DA.SourceArithmetic(Float32; state_exponent = 3, field_exponent = 1)
    initial = zeros(ComplexF32, 4, 1)
    initial[1] = 0.25f0
    grid = DA.ModalSourceGrid(
        execution; kind = :envelope, n_time = 4, dt = 1.0,
        crop_index = collect(0:3), omega = ones(4), active = fill(true, 4),
        omega_window = ones(4), source_window = ones(4),
        source_window_derivative = zeros(4), arithmetic
    )
    map = DA.ModalSourceMap(
        execution; synthesis = ones(1, 1), projection = ones(1, 1),
        arithmetic
    )
    kerr = DA.ModalKerr(
        DA.KerrCoefficients(
            0.25, :analytic;
            self_factor = 1.0, conjugate_factor = 0.0
        ), DA.SourceProfile(1.0); arithmetic
    )
    data = DA.ModalSourceData(
        grid, map,
        DA.SourcePrefactor(execution, fill(-1.0im, 4, 1); arithmetic); kerr
    )
    source = DA.ModalSource(execution, data; node_chunk = 1)
    operator = DA.DiagonalOperator(execution, fill(0.25im, 4, 1))
    policy = dfadams_policy([1 / 8])
    schedule = DA.MarchSchedule(;
        planes = policy.save_planes, policy.breakpoints,
        policy.hard_kinks, policy.stretch, step_planes = 0,
        replay_steps = fill(1 / 64, 8), replay_edges = collect(0:8) ./ 64
    )
    spec = DA.WideStartSpec(
        0.0, nothing, schedule; arithmetic, method = :adams,
        order_max = 1, source_absent = false,
        geometry = DA.WideGeometry(0.0, 0.25; step_max = 1 / 64)
    )
    flow = DA.DF32AdamsFlow(execution, operator, initial, spec, policy)
    norm = DA.MarchNorm(execution, fill(true, 4), fill(true, 4), ones(4), 0.01, 0.0)
    session = DA.AdamsSession(
        execution, initial, 0.0, 1 / 64, policy, norm,
        schedule, source, flow; arithmetic
    )
    return session
end

function dfadams_material_reference()
    h, value = BigFloat(1 / 64), Complex{BigFloat}(2)
    source = value -> -im * abs2(value) * value / 64
    sampled = source(value)
    x = Complex{BigFloat}(0, 1 / 256)
    linear, first, second = exp(x), dfadams_phi(x, 1), dfadams_phi(x, 2)
    for unused in 1:8
        predictor = linear * value + h * first * sampled
        newest = source(predictor)
        value = linear * value + h * ((first - second) * sampled + second * newest)
        sampled = newest
    end
    return value
end

@testset "Certified complete DF32 factors compose actual scaled FFT Kerr sources" begin
    setprecision(BigFloat, 256) do
        reference = dfadams_material_reference()
        unit, h, lipschitz = BigFloat(2)^-24, BigFloat(1 / 64), BigFloat(27 / 64)
        growth = 1 + h * lipschitz + (h * lipschitz)^2 / 2
        # The DC-bin source is -i|A|²A/64. On |A|<=3 its Jacobian norm is
        # <=27/64. The two-register PEC max norm grows by at most growth.
        # 2048 scalar roundings per source/trial cover coefficients, complete
        # factors and state sums, under the conditional finite four-point FFT
        # component error <=8u*sum(abs(input)), including its twiddles.
        # This is an explicit FFT hypothesis, not a butterfly-count proof.
        bound = 9 * (2048unit / (1 - 2048unit)) * 3 * growth^8
        @test 0 < bound < abs(reference - 2exp(Complex{BigFloat}(0, 1 / 32))) / 4
        for tasks in unique((1, min(2, Threads.nthreads(:default))))
            session = dfadams_material(; tasks)
            dfadams_turn!(session)
            @test dfadams_allocations(session) == 0
            while !session.state.finished
                dfadams_turn!(session)
                @test maximum(abs, 8 .* ComplexF64.(session.field)) <= 3
                @test maximum(abs, 8 .* ComplexF64.(session.work.predictor)) <= 3
            end
            @test abs(8Complex{BigFloat}(session.field[1]) - reference) <= bound
            @test all(iszero, session.field[2:end])
            @test session.source.calls == session.ledger.source_evaluations == 9
        end
    end
end

@testset "DF32 Adams inventories full provenance and refuses unqualified routes" begin
    f = dfadams_fixture(; varying = true, order = 6)
    arrays = Dict{Symbol, Int}()
    seen = IdDict{Any, Bool}()
    DA._memory_arrays!(arrays, seen, "session", f.session)
    for record in (
            f.flow.records..., f.flow.plans.banks[1].seed_rate,
            f.flow.plans.banks[1].endpoint_rate, f.flow.plans.banks[2].seed_rate,
            f.flow.plans.banks[2].endpoint_rate,
        )
        for array in (
                record.saved, record.sample.values, record.sample.errors,
                record.numerical,
            )
            @test haskey(seen, array)
        end
    end
    expected = (6 + 2 + 4) * length(f.state) *
        (3sizeof(ComplexF64) + sizeof(DA._DF32AdamsScaled{DA.DF32}))
    @test sum(values(DA.session_memory(f.session))) >= expected
    @test sizeof(DA._DF32AdamsScaled{DA.DF32}) == 24
    @test_throws ArgumentError DA.DF32AdamsFlow(
        f.execution, f.operator, ComplexF64.(f.state), f.spec, f.policy
    )
    @test_throws ArgumentError DA.DF32AdamsFlow(
        f.execution, :spatial, f.state, f.spec, f.policy
    )
    @test_throws ArgumentError DA.DF32AdamsFlow(
        f.execution, f.operator, f.state,
        f.spec, f.policy; tabulate = true
    )
    dense_policy = dfadams_policy(
        f.schedule.planes; order = 6, interpolate = true, save_planes = Float64[]
    )
    @test_throws ArgumentError DA.DF32AdamsFlow(
        f.execution, f.operator, f.state,
        f.spec, dense_policy
    )
    dense = DA.AdamsDense(f.execution, f.state, 6; compensated = true)
    @test_throws ArgumentError DA.adams_retain_dense!(
        dense, f.session.work,
        f.session.field, f.session.field, f.session.history, f.session.stencil, f.execution
    )
    @test f.source.calls == 0 && f.session.ledger.startup == 0
    finish = last(f.schedule.planes)
    complete_spec = DA.WideStartSpec(
        finish, nothing, f.schedule;
        arithmetic = f.spec.arithmetic, method = :adams, order_max = 6,
        source_absent = false, geometry = nothing,
    )
    complete = DA.DF32AdamsFlow(f.execution, f.operator, f.state, complete_spec, f.policy)
    @test complete.binding.operator === nothing && complete.plans.banks === nothing
    @test isempty(complete.records)
    @test_throws ArgumentError DA.prepare_df32_adams_trial!(
        complete, f.session.history,
        f.session.stencil, DA.DF32TrialSite(finish, 1 / 64, finish + 1 / 64, 9)
    )
end

@testset "DF32 Adams restarts and rejections preserve real work and seed coordinates" begin
    restarted = dfadams_fixture(; order = 3, varying = true, hard_kinks = [1 / 16])
    dfadams_drain!(restarted.session; work_limit = 1)
    @test restarted.session.state.failure_code === :none
    @test restarted.session.ledger.startup == 2 && restarted.source.calls == 10
    @test restarted.session.ledger.accepted_steps == 8
    @test restarted.session.ledger.restarts == 1
    failed = dfadams_fixture(; order = 3, varying = true, hard_kinks = [1 / 16])
    failed.source.fail_at = 6
    dfadams_drain!(failed.session; work_limit = 1)
    @test failed.session.state.failure_code === :ErrorException
    @test failed.session.state.z == failed.session.state.flight_z == 1 / 16
    @test iszero(failed.session.state.flight_dz)
    @test failed.source.calls == failed.session.ledger.source_evaluations == 6
    @test failed.session.ledger.accepted_steps == 4
    rejected = dfadams_fixture(;
        replay = false, order = 3, varying = true,
        gain = 8.0 + 0im, forcing = 0.0im, rtol = 1.0e-5
    )
    dfadams_drain!(rejected.session; work_limit = 1)
    @test rejected.session.state.failure_code === :none
    @test rejected.session.state.z == 1 / 8
    @test rejected.session.ledger.rejected_steps > 0
    @test rejected.source.calls == rejected.session.ledger.source_evaluations ==
        rejected.session.ledger.startup + rejected.session.ledger.predictor
end

@testset "Certified factored defects retain meaningful high-order contributions" begin
    setprecision(BigFloat, 256) do
        unit = BigFloat(2)^-24
        for order in 1:6
            f = dfadams_restored(; order)
            history, work = f.session.history, f.session.work
            fill!(f.state, 0)
            fill!(f.session.field, 0)
            fill!(history.states, 0)
            f.source.gain = f.source.forcing = zero(ComplexF32)
            for slot in 1:order
                history.source[:, :, slot] .= ComplexF32((-1)^slot * slot, (-1)^slot / 2)
            end
            stencil = DA.AdamsStencil(order)
            slots = collect(order:-1:1)
            DA.adams_stencil!(stencil, order, slots, [order])
            z, h = f.spec.start, first(f.schedule.replay_steps)
            token = DA.prepare_df32_adams_trial!(
                f.flow, history, stencil,
                DA.DF32TrialSite(z, h, f.schedule.replay_edges[2], 1)
            )
            bank = DA._da_token(token)
            certificate = bank.certificate
            nodes = (Rational{BigInt}.(history.positions[slots]) .- Rational{BigInt}(z)) ./
                Rational{BigInt}(h)
            points = vcat(one(eltype(nodes)), nodes)
            pred, corr = dfadams_basis(nodes), dfadams_basis(points)
            phase = Complex{BigFloat}(-0.125, 0.25) *
                (BigFloat(z + h)^2 - BigFloat(z)^2) / 2
            moments = [factorial(k - 1) * dfadams_phi(phase, k) for k in 1:(order + 1)]
            weight = (basis, sample) -> BigFloat(h) * sum(
                BigFloat(basis[sample, k]) * moments[k] for k in axes(basis, 2)
            )
            sources = vcat(
                zero(Complex{BigFloat}),
                Complex{BigFloat}.(history.source[1, 1, slots])
            )
            predicted = sum(weight(pred, j) * sources[j + 1] for j in 1:order)
            pred_scale = sum(abs(weight(pred, j)) * abs(sources[j + 1]) for j in 1:order)
            gamma = 32(order + 2) * unit / (1 - 32(order + 2) * unit)
            pred_error = BigFloat(certificate.source_error) * sum(abs, sources)
            pred_error += gamma * (pred_scale + pred_error)
            normalization = bank.normalization[1]
            ddweights = [
                ldexp(
                    BigFloat(
                        inv(
                            prod(
                                points[j] - points[k] for k in eachindex(points) if k != j
                            )
                        )
                    ), -normalization
                ) for j in eachindex(points)
            ]
            integral = weight(corr, 1) / first(ddweights)
            difference = sum(ddweights .* sources)
            difference_scale = sum(abs.(ddweights) .* abs.(sources))
            difference_error = BigFloat(certificate.weight_error) * sum(abs, sources)
            difference_error += gamma * (difference_scale + difference_error)
            factor_error = BigFloat(certificate.source_error)
            error = abs(integral) * difference_error +
                factor_error * (difference_scale + difference_error)
            error += gamma * (abs(integral) + factor_error) *
                (difference_scale + difference_error)
            endpoint_rate = Complex{BigFloat}(bank.endpoint_rate.saved[1])
            compensation = sum(BigFloat(corr[1, k]) / k for k in axes(corr, 2)) *
                (BigFloat(h) * endpoint_rate - phase)
            compensation_error = BigFloat(certificate.compensation_error)
            error += compensation_error * abs(predicted) +
                (abs(compensation) + compensation_error) * pred_error +
                gamma * (abs(compensation) + compensation_error) *
                (abs(predicted) + pred_error)
            # The reference integrates the exact rational corrector-minus-predictor
            # basis. It does not use the implementation's factored defect arrays.
            reference = sum(weight(corr, j) * sources[j] for j in eachindex(points)) -
                predicted + compensation * predicted
            contribution = abs(integral) * difference_scale +
                abs(compensation) * abs(predicted)
            bound = error + gamma * (contribution + error)
            @test 0 < bound < abs(reference) / 4
            DA.adams_trial!(
                work, f.session.field, z, h, history, stencil,
                f.source, token, nothing, f.execution
            )
            @test all(
                abs.(Complex{BigFloat}.(work.defects[:, :, 1]) .- reference) .<= bound
            )
            @test all(iszero, work.defects[:, :, 2:3])
            @test f.source.calls == 1
        end
    end
end
