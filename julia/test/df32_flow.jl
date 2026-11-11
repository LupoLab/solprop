# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as DF

mutable struct DF32FlowSource{E}
    execution::E
    coefficient::ComplexF32
    calls::Int
end

function (source::DF32FlowSource)(output, field, position)
    source.calls += 1
    coefficient = source.coefficient
    DF.each!(source.execution, size(field)...) do row, column
        output[row, column] = coefficient * field[row, column]
    end
    return nothing
end

function df32flow_fixture(;
        low = false, tasks = 1, replay = true, initial = nothing, operator = nothing,
        start = 0.0, step = 1 / 64, steps = 8, nodes = [0.0, 1.0],
    )
    execution = DF.KernelExecution(DF.KernelAbstractions.CPU(), tasks, true)
    state = initial === nothing ? ComplexF32[
            complex(row / 8, column / 16) for row in 1:4, column in 1:2
        ] : initial
    finish = start + steps * step
    schedule = DF.MarchSchedule(;
        planes = [finish], breakpoints = [start, finish + step], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0,
        replay_steps = replay ? fill(step, steps) : Float64[],
        replay_edges = replay ? [start + index * step for index in 0:steps] : Float64[],
    )
    scheme = low ? DF.RK4IPTableau() : DF.EmbeddedTableau(
            nodes, [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
        )
    arithmetic = DF.SourceArithmetic(Float32; state_exponent = 3, field_exponent = 1)
    geometry = DF.WideGeometry(
        start, finish + step; step_max = step, step_min = replay ? step : 0.0,
    )
    spec = DF.WideStartSpec(
        start, low ? nothing : scheme, schedule;
        geometry, arithmetic, method = low ? :low_storage : :rk,
        order_max = 0, source_absent = false,
    )
    linear = operator === nothing ? DF.DiagonalOperator(
            execution, fill(-0.125 + 0.25im, size(state)),
        ) : operator
    flow = DF.DF32LinearFlow(execution, linear, state, spec)
    source = DF32FlowSource(execution, ComplexF32(-0.25, 0.125), 0)
    return (;
        execution, state, schedule, scheme, arithmetic, spec, flow, source,
        operator = linear,
    )
end

function df32flow_site(fixture; index = 1)
    schedule, spec = fixture.schedule, fixture.spec
    if isempty(schedule.replay_steps)
        return DF.DF32TrialSite(
            spec.start, spec.geometry.step_max,
            spec.start + spec.geometry.step_max, 0
        )
    end
    return DF.DF32TrialSite(
        index == 1 ? spec.start : schedule.replay_edges[index],
        schedule.replay_steps[index],
        schedule.replay_edges[index + 1], index,
    )
end

function df32flow_cycle!(output, fixture)
    trial = DF.prepare_df32_trial!(fixture.flow, fixture.scheme, df32flow_site(fixture))
    origin, destination = DF._df32_flow_interval(fixture.scheme, trial.site, 2)
    trial(output, fixture.state, origin, destination)
    return nothing
end

function df32flow_allocated(output, fixture)
    df32flow_cycle!(output, fixture)
    df32flow_cycle!(output, fixture)
    return @allocated df32flow_cycle!(output, fixture)
end

function df32flow_product_bound(input, phase)
    # The certified assembly plus corrected forward/backward pair contributes <5u
    # relatively. A complex binary32 product has two real products and one sum in
    # each component. gamma_3*sqrt(2) bounds its normwise absolute contribution.
    # 16u bounds 5u+(1+5u)*sqrt(2)*gamma_3, including binary64 bound evaluation.
    unit = Float64(eps(Float32)) / 2
    return 16unit * abs(input) * exp(real(phase)) + 16 * Float64(nextfloat(0.0f0))
end

@testset "DF32 flow banks retain certificates and exact coordinate generations" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        fixture = df32flow_fixture(; tasks)
        flow, state = fixture.flow, fixture.state
        @test flow.operator === fixture.operator
        @test flow.binding.spec === fixture.spec
        @test isbitstype(DF._DF32FlowOperation)
        @test length(flow.plans.banks[1]) == length(fixture.scheme.nodes) + 1
        @test length(flow.failures) == size(state, 2)
        token = DF.prepare_df32_trial!(flow, fixture.scheme, df32flow_site(fixture))
        bank = flow.plans.banks[flow.plans.active]
        for index in eachindex(bank)
            origin, destination = DF._df32_flow_interval(fixture.scheme, token.site, index)
            operation = DF.wide_operation(
                flow.binding.operator, origin, destination; step = token.site.step,
            )
            projection = bank[index]
            @test projection.scalars === operation.scalars
            @test projection.scalar_certificate === operation.scalar_certificate
            @test projection.certificate.real_error === operation.certificate.real_error
            @test projection.certificate.imag_error === operation.certificate.imag_error
            @test projection.certificate.intermediate === operation.certificate.intermediate
            @test projection.certificate.status == 0x00
            @test projection.generation == token.generation
            @test projection.origin === origin && projection.destination === destination
            @test projection.step === token.site.step
        end
        output = similar(state)
        token(output, state, token.site.origin, token.site.origin + token.site.step)
        phase = Complex{BigFloat}(-0.125, 0.25) * BigFloat(token.site.step)
        for index in eachindex(state)
            expected = Complex{BigFloat}(state[index]) * exp(phase)
            bound = df32flow_product_bound(state[index], phase)
            @test bound > 0 && abs(Complex{BigFloat}(output[index]) - expected) <= bound
        end
        token(output, state, token.site.origin + token.site.step, token.site.origin)
        for index in eachindex(state)
            expected = Complex{BigFloat}(state[index]) * exp(-phase)
            bound = df32flow_product_bound(state[index], -phase)
            @test bound > 0 && abs(Complex{BigFloat}(output[index]) - expected) <= bound
        end
        zeros_state = fill(ComplexF32(-0.0f0, 0.0f0), size(state))
        token(output, zeros_state, token.site.origin, token.site.origin)
        @test reinterpret(UInt32, vec(output)) == reinterpret(UInt32, vec(zeros_state))
        next_token = DF.prepare_df32_trial!(flow, fixture.scheme, df32flow_site(fixture))
        saved = copy(output)
        @test_throws ArgumentError token(output, state, 0.0, token.site.step)
        @test output == saved
        @test next_token.generation == token.generation + 1
        @test_throws ArgumentError next_token(output, state, 0.0, 0.5token.site.step)
        @test output == saved
        @test df32flow_allocated(output, fixture) == 0
        @test df32flow_allocated(output, fixture) == 0
    end
end

function df32flow_session(fixture; capacity = 16, proposal = nothing, rtol = 0.01)
    (; execution, state, schedule, scheme, arithmetic, source, flow) = fixture
    norm = DF.MarchNorm(
        execution, fill(true, size(state, 1)), fill(true, size(state, 1)),
        ones(size(state, 1)), rtol, 0.0,
    )
    if scheme isa DF.RK4IPTableau
        return DF.LowStorageSession(
            execution, state, fixture.spec.start, scheme, norm, schedule, source, flow;
            capacity, arithmetic,
        )
    end
    control = DF.PIControl(;
        safety = 0.8, max_factor = 2.0, min_factor = 0.2,
        beta1 = 0.3, beta2 = -0.1, lag = 0, reject_free = false,
        estimator_order = 1, start_growth = 100.0, dz_min = 1.0e-12,
        dz_max = fixture.spec.geometry.step_max / 2, auto_init = false,
    )
    return DF.RungeKuttaSession(
        execution, state, fixture.spec.start,
        proposal === nothing ? fixture.spec.geometry.step_max : proposal,
        scheme, control, norm, schedule, source, flow; capacity, arithmetic,
    )
end

function df32flow_turn!(session)
    report = DF.advance!(session; work_limit = 1, deadline = 60.0)
    DF.acknowledge!(session, report.generation)
    return nothing
end

function df32flow_turn_allocated(session)
    df32flow_turn!(session)
    return @allocated df32flow_turn!(session)
end

@testset "Certified binary32 RK and low-storage sessions propagate actual steps" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))), low in (false, true)
        fixture = df32flow_fixture(; low, tasks)
        session = df32flow_session(fixture)
        while !session.state.finished
            df32flow_turn!(session)
        end
        @test session.state.accepted == 8 && session.state.rejected == 0
        @test fixture.source.calls == (low ? 32 : 16)
        @test eltype(session.field) === ComplexF32
        @test session.flow === fixture.flow
        # Exact stability polynomial for the matched commuting scalar source.
        # Each step has <256 scalar operations on a path (including corrected
        # factors). gamma_(512*steps) covers these and <5u pair errors; the
        # absolute polynomial/exponential majorant bounds propagation sensitivity.
        h = BigFloat(fixture.spec.geometry.step_max)
        gain, linear = Complex{BigFloat}(-0.25, 0.125), Complex{BigFloat}(-0.125, 0.25)
        order = low ? 4 : 2
        factor = exp(linear * h) * sum((gain * h)^k / factorial(k) for k in 0:order)
        expected = Complex{BigFloat}.(fixture.state) .* factor^8
        unit = Float64(eps(Float32)) / 2
        gamma = 4096unit / (1 - 4096unit)
        bound = gamma * maximum(abs, fixture.state) * exp(8h * (abs(linear) + abs(gain)))
        @test bound > 0
        @test maximum(abs, Complex{BigFloat}.(session.field) - expected) <= bound
        warm = df32flow_session(df32flow_fixture(; low, tasks))
        df32flow_turn!(warm)
        @test df32flow_turn_allocated(warm) == 0
        @test df32flow_turn_allocated(warm) == 0
    end
end

function df32flow_polynomial(execution, shape; scale = 1.0, offset = 0.0)
    # Defining Chebyshev polynomial on [0,1.5]: Phi(z)=offset+scale*z^2.
    # z=(3/4)(x+1), so z^2=(27/32)T0+(9/8)T1+(9/32)T2.
    coefficients = repeat(
        reshape(ComplexF64[offset + 27scale / 32, 9scale / 8, 9scale / 32], 1, 3, 1);
        outer = (prod(shape), 1, 1),
    )
    return DF.RepresentedOperator(
        execution, DF.ChebyshevTable(execution, [0.0, 1.5], coefficients), shape...,
    )
end

@testset "Late interval refusal preserves the active plan and accepted state" begin
    execution = DF.KernelExecution(DF.KernelAbstractions.CPU(), 1, true)
    operator = df32flow_polynomial(execution, (4, 2); scale = 40.0)
    for low in (false, true)
        fixture = df32flow_fixture(; low, operator, step = 0.5, steps = 2)
        flow = fixture.flow
        token = DF.prepare_df32_trial!(flow, fixture.scheme, df32flow_site(fixture))
        bank = flow.plans.active
        saved = copy(flow.plans.banks[bank])
        # For RK4IP the second trial's first half has exponent12.5, but the
        # second has17.5; the complete preflight must reject before source entry.
        site = df32flow_site(fixture; index = 2)
        @test_throws ArgumentError DF.prepare_df32_trial!(flow, fixture.scheme, site)
        @test flow.plans.active == bank && flow.plans.generation == token.generation
        @test flow.plans.banks[bank] == saved
        output = similar(fixture.state)
        origin, destination = DF._df32_flow_interval(fixture.scheme, token.site, 1)
        @test token(output, fixture.state, origin, destination) === nothing

        session = df32flow_session(
            df32flow_fixture(;
                low, operator, step = 0.5, steps = 2,
            )
        )
        df32flow_turn!(session) # launch
        df32flow_turn!(session) # first successful step
        field = copy(session.field)
        calls, generation = session.source.calls, session.flow.plans.generation
        report = DF.advance!(session; work_limit = 1, deadline = 60.0)
        @test report.kind == DF.SESSION_FAILED && !report.poisoned
        @test !session.state.in_flight
        @test session.field == field && session.state.accepted == 1
        @test session.source.calls == calls
        @test session.flow.plans.generation == generation
        @test session.attempts.count == 0
        @test session.state.flight_z == 0.5 && session.state.flight_dz == 0.5
    end
end

@testset "Direct trial guards refuse stale plans and retained storage before work" begin
    for low in (false, true)
        fixture = df32flow_fixture(; low)
        (; state, flow, scheme, source, execution) = fixture
        work = low ? DF.LowStorageWork(execution, state, scheme) :
            DF.RungeKuttaWork(state, scheme)
        token = DF.prepare_df32_trial!(flow, scheme, df32flow_site(fixture))
        DF.prepare_df32_trial!(flow, scheme, df32flow_site(fixture))
        trial! = low ? DF.low_storage_trial! : DF.rk_trial!
        for buffer in (
                low ? (work.accumulator, work.stage, work.source) :
                    (
                        work.stages, work.combination, work.lab, work.source,
                        work.transported, work.candidate, work.defect,
                    )
            )
            fill!(buffer, ComplexF32(-3, 7))
        end
        for callback in (flow, token)
            @test_throws ArgumentError trial!(
                work, state, 0.0, 1 / 64, scheme, source, callback, execution,
            )
            @test source.calls == 0
            @test all(==(ComplexF32(-3, 7)), work.source)
        end
        current = DF.prepare_df32_trial!(flow, scheme, df32flow_site(fixture))
        owner = flow.operator.rate
        preserved = copy(owner)
        GC.@preserve owner begin
            alias = unsafe_wrap(
                Array, Ptr{ComplexF32}(pointer(owner, 2)), size(state);
                own = false
            )
            @test_throws ArgumentError trial!(
                work, alias, 0.0, 1 / 64, scheme, source, current, execution,
            )
            @test source.calls == 0 && owner == preserved
            @test all(==(ComplexF32(-3, 7)), work.source)
            aliased_work = low ? DF.LowStorageWork(
                    alias, work.stage, work.source, work.norm_mask, work.norm_parts,
                    work.finite_columns,
                ) : DF.RungeKuttaWork(
                    work.stages, work.combination, work.lab, work.source,
                    work.transported, alias, work.defect, work.norm_parts, work.norm_scale,
                )
            @test_throws ArgumentError trial!(
                aliased_work, state, 0.0, 1 / 64, scheme, source, current, execution,
            )
            @test source.calls == 0 && owner == preserved
            @test all(==(ComplexF32(-3, 7)), work.source)
        end
    end
end

@testset "Flow buffer ownership and valid strided or in-place application" begin
    fixture = df32flow_fixture()
    (; flow, state, scheme) = fixture
    token = DF.prepare_df32_trial!(flow, scheme, df32flow_site(fixture))
    output = fill(ComplexF32(-9, 3), size(state))
    saved = copy(output)
    for (bad, error_type) in (
            (ComplexF64.(state), ArgumentError),
            (zeros(ComplexF32, 1, 1), DimensionMismatch),
        )
        @test_throws error_type token(output, bad, 0.0, 1 / 64)
        @test output == saved
    end
    narrow_owner = fill(ComplexF32(1, 2), length(state) + 1)
    GC.@preserve narrow_owner begin
        input = unsafe_wrap(Array, pointer(narrow_owner), size(state); own = false)
        overlap = unsafe_wrap(Array, pointer(narrow_owner, 2), size(state); own = false)
        previous = copy(narrow_owner)
        @test_throws ArgumentError token(overlap, input, 0.0, 1 / 64)
        @test narrow_owner == previous
    end
    for owner in (
            flow.operator.rate, flow.binding.operator.coefficients,
            flow.binding.operator.coefficient_errors, flow.plans.banks[flow.plans.active],
        )
        previous = copy(owner)
        GC.@preserve owner begin
            alias = unsafe_wrap(
                Array, Ptr{ComplexF32}(pointer(owner)), size(state);
                own = false
            )
            @test_throws ArgumentError token(alias, state, 0.0, 1 / 64)
            @test_throws ArgumentError token(output, alias, 0.0, 1 / 64)
            @test owner == previous && output == saved
        end
    end
    reference = similar(state)
    token(reference, state, 0.0, 1 / 64)
    inplace = copy(state)
    token(inplace, inplace, 0.0, 1 / 64)
    # Same graph and operands; exact equality here checks only layout independence.
    @test inplace == reference
    storage = fill(ComplexF32(7), 2size(state, 1), size(state, 2))
    reversed = @view storage[end:-2:2, :]
    token(reversed, state, 0.0, 1 / 64)
    @test reversed == reference && all(==(ComplexF32(7)), @view storage[1:2:end, :])
    for bad in (NaN32, Inf32)
        invalid = copy(state)
        invalid[1] = bad
        @test_throws DF.NonfiniteSourceError token(output, invalid, 0.0, 1 / 64)
        @test output == saved
    end
end

@testset "Represented and expansion phases retain cancellation and supplied steps" begin
    execution = DF.KernelExecution(DF.KernelAbstractions.CPU(), 1, true)
    for kind in (:represented, :expansion), offset in (0.0, 2.0^40)
        scale = -0.25 + 0.5im
        operator = if kind === :represented
            df32flow_polynomial(execution, (4, 2); scale, offset)
        else
            coefficients = reshape([offset + 27 / 32, 9 / 8, 9 / 32], 1, 3, 1)
            weights = DF.ChebyshevTable(execution, [0.0, 1.5], coefficients)
            DF.ExpansionOperator(execution, fill(scale, 4, 2, 1), weights)
        end
        fixture = df32flow_fixture(; operator)
        output = similar(fixture.state)
        token = DF.prepare_df32_trial!(fixture.flow, fixture.scheme, df32flow_site(fixture))
        token(output, fixture.state, 0.0, 1 / 64)
        phase = Complex{BigFloat}(scale) * (big(1) / 64)^2
        for index in eachindex(output)
            budget = df32flow_product_bound(fixture.state[index], phase)
            expected = exp(phase) * Complex{BigFloat}(fixture.state[index])
            @test budget > 0 && abs(Complex{BigFloat}(output[index]) - expected) <= budget
        end
        @test df32flow_allocated(output, fixture) == 0
    end
    fixture = df32flow_fixture(; start = 0.2, step = 0.1, steps = 8)
    saw_difference = false
    for index in 1:8
        site = df32flow_site(fixture; index)
        token = DF.prepare_df32_trial!(fixture.flow, fixture.scheme, site)
        saw_difference |= site.origin + site.step != site.destination
        @test token.site.step === 0.1
        @test token.site.destination === fixture.schedule.replay_edges[index + 1]
        @test last(fixture.flow.plans.banks[fixture.flow.plans.active]).destination ===
            site.origin + site.step
    end
    @test saw_difference
    previous = fixture.flow.plans.generation
    for bad in (
            DF.DF32TrialSite(0.2, 0.1, 0.3, 0),
            DF.DF32TrialSite(0.2, 0.05, 0.3, 1),
            DF.DF32TrialSite(0.2, 0.1, 0.31, 1),
            DF.DF32TrialSite(0.2, 0.1, 0.3, 9),
        )
        @test_throws ArgumentError DF.prepare_df32_trial!(fixture.flow, fixture.scheme, bad)
        @test fixture.flow.plans.generation == previous
    end
end

@testset "Actual-start attachment refuses unsupported or mismatched preparation" begin
    fixture = df32flow_fixture()
    (; flow, state, spec, scheme, arithmetic, execution, schedule) = fixture
    original = copy(state)
    @test_throws ArgumentError DF.DF32LinearFlow(
        execution, flow.operator, state, spec; tabulate = true,
    )
    @test_throws ArgumentError DF.DF32LinearFlow(
        execution, flow.operator, ComplexF64.(state), spec,
    )
    adams = DF.WideStartSpec(
        0.0, nothing, schedule; geometry = spec.geometry, arithmetic,
        method = :adams, order_max = 6, source_absent = false,
    )
    @test_throws ArgumentError DF.DF32LinearFlow(execution, flow.operator, state, adams)
    @test_throws ArgumentError DF.DF32LinearFlow(execution, :spatial, state, spec)
    other = df32flow_fixture()
    @test_throws ArgumentError DF._flow_start(flow, scheme, 0.0, other.schedule)
    @test_throws ArgumentError DF._flow_start(flow, scheme, 1 / 64, schedule)
    @test_throws ArgumentError DF._flow_start(flow, DF.RK4IPTableau(), 0.0, schedule)
    @test_throws ArgumentError DF._flow_arithmetic(
        flow, state, DF.SourceArithmetic(Float32; state_exponent = 4, field_exponent = 1),
    )
    @test_throws ArgumentError DF._flow_arithmetic(flow, ComplexF64.(state), arithmetic)
    token = DF.prepare_df32_trial!(flow, scheme, df32flow_site(fixture))
    @test_throws ArgumentError DF._flow_start(token, scheme, 0.0, schedule)
    @test state == original
    for arguments in (
            (0.0, 0.0, 0.0, 0), (Inf, 1.0, 1.0, 0),
            (1.0, eps(1.0) / 4, 1.0, 0), (0.0, 0.1, 0.1, -1),
        )
        @test_throws ArgumentError DF.DF32TrialSite(arguments...)
    end
    flow.plans.generation = typemax(UInt64)
    previous = copy(flow.plans.banks[flow.plans.active])
    @test_throws ArgumentError DF.prepare_df32_trial!(flow, scheme, df32flow_site(fixture))
    @test flow.plans.generation == typemax(UInt64)
    @test flow.plans.banks[flow.plans.active] == previous

    completed = DF.WideStartSpec(
        last(schedule.planes), scheme, schedule; geometry = nothing,
        arithmetic, method = :rk, order_max = 0, source_absent = false,
    )
    huge = DF.DiagonalOperator(execution, fill(complex(floatmax(Float64)), size(state)))
    done = DF.DF32LinearFlow(execution, huge, state, completed)
    @test done.binding.operator === nothing
    @test all(isempty, done.plans.banks)
    @test_throws ArgumentError DF.prepare_df32_trial!(done, scheme, df32flow_site(fixture))
end

@testset "Adaptive DF32 RK uses the same actual accepted and rejected trial policy" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        fixture = df32flow_fixture(; tasks, replay = false)
        session = df32flow_session(fixture; rtol = 1.0e-7)
        while !session.state.finished
            df32flow_turn!(session)
        end
        @test session.state.accepted > 0 && session.state.rejected > 0
        @test session.source.calls == 2 * (session.state.accepted + session.state.rejected)
        attempts = session.state.accepted + session.state.rejected
        @test session.flow.plans.generation == attempts
        @test session.state.z === last(session.schedule.planes)
        # Adaptive accuracy is separate from matched-method rounding: compare with
        # the analytic commuting exponential over this short fixture, with a
        # declared 1e-5 relative global envelope (not the local controller rtol).
        exact = Complex{BigFloat}.(fixture.state) .*
            exp(Complex{BigFloat}(-0.375, 0.375) / 8)
        envelope = big(1.0e-5) * maximum(abs, fixture.state)
        @test envelope > 0
        @test maximum(abs, Complex{BigFloat}.(session.field) - exact) <= envelope
    end
end

df32flow_big(value::DF.DF32) = BigFloat(value.hi) + BigFloat(value.lo)
df32flow_big(value::Complex{DF.DF32}) =
    complex(df32flow_big(real(value)), df32flow_big(imag(value)))

@testset "Whole-expression cancellation and exact low-word function domains" begin
    execution = DF.KernelExecution(DF.KernelAbstractions.CPU(), 1, true)
    operator = df32flow_polynomial(execution, (4, 2); scale = 1.0im, offset = 2.0^40)
    fixture = df32flow_fixture(;
        operator, start = 0.5, step = 2.0^-40, steps = 1,
    )
    token = DF.prepare_df32_trial!(fixture.flow, fixture.scheme, df32flow_site(fixture))
    operation = last(fixture.flow.plans.banks[fixture.flow.plans.active])
    prepared = fixture.flow.binding.operator
    rate = DF._wide_graph(
        prepared.coefficients, prepared.terms, nothing, nothing,
        prepared.kind, 1, operation.scalars, Val(:compute),
    )
    phase = DF._wide_result(rate, operation.scalars, 0x00, rate)
    expected = im * (BigFloat(operation.destination)^2 - BigFloat(operation.origin)^2)
    bound = operation.certificate.real_error + operation.certificate.imag_error
    @test 0 < bound < abs(expected)
    @test abs(df32flow_big(phase) - expected) <= bound
    @test !iszero(imag(phase))
    for value in (16.0 + 0im, -16.0 + 0im, (2.0^10) * im, -(2.0^10) * im)
        constant = DF.DiagonalOperator(execution, fill(value, 4, 2))
        valid = df32flow_fixture(; operator = constant, step = 1.0, steps = 1)
        @test DF.prepare_df32_trial!(valid.flow, valid.scheme, df32flow_site(valid)) isa
            DF.DF32TrialFlow
    end
    # Function range is necessary, not sufficient: this complete graph's
    # conservative propagated errors can exceed u/8 inside that rectangle.
    for value in ((2.0^20) * im, -(2.0^20) * im)
        constant = DF.DiagonalOperator(execution, fill(value, 4, 2))
        conservative = df32flow_fixture(; operator = constant, step = 1.0, steps = 1)
        @test_throws ArgumentError DF.prepare_df32_trial!(
            conservative.flow, conservative.scheme, df32flow_site(conservative),
        )
        operation = DF.wide_operation(
            conservative.flow.binding.operator, 0.0, 1.0; step = 1.0,
        )
        @test operation.certificate.status === :assembly_error
        @test conservative.flow.plans.generation == 0
    end
    for value in (
            16.0 + 2.0^-30 + 0im, -16.0 - 2.0^-30 + 0im,
            (2.0^20 + 2.0^-10)im, -(2.0^20 + 2.0^-10)im,
        )
        constant = DF.DiagonalOperator(execution, fill(value, 4, 2))
        invalid = df32flow_fixture(; operator = constant, step = 1.0, steps = 1)
        @test_throws ArgumentError DF.prepare_df32_trial!(
            invalid.flow, invalid.scheme, df32flow_site(invalid),
        )
        @test invalid.flow.plans.generation == 0 && invalid.flow.plans.active == 0
    end
    # Independent endpoint evaluation across a knot cannot discard the large
    # common cumulative offset. Its conservative radius refuses this small jump.
    coefficients = repeat(
        reshape(ComplexF64[2.0^50, 1, 2.0^50, 1], 1, 2, 2);
        outer = (8, 1, 1)
    )
    crossing = DF.RepresentedOperator(
        execution,
        DF.ChebyshevTable(execution, [0.0, 0.5, 1.0], coefficients), 4, 2
    )
    refused = df32flow_fixture(; operator = crossing, start = 0.49, step = 0.02, steps = 1)
    @test_throws ArgumentError DF.prepare_df32_trial!(
        refused.flow, refused.scheme, df32flow_site(refused),
    )
    @test refused.flow.plans.generation == 0 && refused.flow.plans.active == 0
end

function df32flow_material(fixture)
    (; execution, arithmetic) = fixture
    grid = DF.ModalSourceGrid(
        execution; kind = :envelope, n_time = 4, dt = 1.0,
        crop_index = collect(0:3), omega = ones(4), active = fill(true, 4),
        omega_window = ones(4), source_window = ones(4),
        source_window_derivative = zeros(4), arithmetic,
    )
    map = DF.ModalSourceMap(
        execution; synthesis = ones(1, 1), projection = ones(1, 1), arithmetic,
    )
    kerr = DF.ModalKerr(
        DF.KerrCoefficients(0.25, :analytic; self_factor = 1.0, conjugate_factor = 0.0),
        DF.SourceProfile(1.0); arithmetic,
    )
    data = DF.ModalSourceData(
        grid, map, DF.SourcePrefactor(execution, fill(-1.0im, 4, 1); arithmetic); kerr,
    )
    return DF.ModalSource(execution, data; node_chunk = 1)
end

function df32flow_cubic_reference(low)
    value, h = Complex{BigFloat}(0.25), big(1) / 64
    linear = Complex{BigFloat}(-0.125, 0.25)
    nodes = low ? (0, 1 // 2, 1 // 2, 1) : (0, 1)
    weights = low ? (1 // 6, 1 // 3, 1 // 3, 1 // 6) : (1 // 2, 1 // 2)
    for unused in 1:8
        stages = Complex{BigFloat}[]
        for (index, node) in enumerate(nodes)
            previous = index == 1 ? zero(value) : stages[index - 1]
            field = exp(linear * h * node) * (value + h * node * previous)
            push!(stages, exp(-linear * h * node) * (-im * abs2(field) * field))
        end
        value = exp(linear * h) * (value + h * sum(weights .* stages))
    end
    return value
end

@testset "DF32 flow composes with actual scaled Kerr source and FFT plans" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))), low in (false, true)
        initial = zeros(ComplexF32, 4, 1)
        initial[1] = 0.25f0
        fixture = df32flow_fixture(; tasks, low, initial)
        source = df32flow_material(fixture)
        session = df32flow_session(merge(fixture, (; source)))
        df32flow_turn!(session)
        @test df32flow_turn_allocated(session) == 0
        while !session.state.finished
            df32flow_turn!(session)
        end
        @test source.calls == (low ? 32 : 16)
        @test eltype(session.field) === ComplexF32
        @test all(iszero, session.field[2:end])
        # One occupied DC bin gives the raw scaled cubic N(a)=-i|a|²a:
        # physical N(A)=-i|A|²A/64 and A=8a. On |a|<=1 its Jacobian is <=3.
        # Under the qualified four-point FFT <=8u hypothesis, fewer than8192
        # rounded scalar contributions per step cover transforms, cubic responses,
        # stages and <=5u corrected factors. The positive absolute majorant uses
        # the unit-amplitude ball, not a cancellation-reduced final source.
        unit = Float64(eps(Float32)) / 2
        gamma = 8192unit / (1 - 8192unit)
        bound = 8gamma * exp((abs(-0.125 + 0.25im) + 3) / 8)
        @test bound > 0
        reference = df32flow_cubic_reference(low)
        @test abs(Complex{BigFloat}(session.field[1]) - reference) <= bound
        ledger = DF.session_memory(session)
        for index in 1:2
            key = Symbol("session.flow.plans.banks.$index")
            slots = low ? 2 : length(fixture.scheme.nodes) + 1
            @test ledger[key] == slots * sizeof(DF._DF32FlowOperation)
        end
        @test ledger[Symbol("session.flow.failures")] == sizeof(Int)
        @test initial[1] == 0.25f0 && all(iszero, initial[2:end])
    end
end

@testset "Actual interior RK stages are certified even with a valid final phase" begin
    execution = DF.KernelExecution(DF.KernelAbstractions.CPU(), 1, true)
    constant = DF.DiagonalOperator(execution, fill(10.0 + 0im, 4, 2))
    # Phi(z)=128z(1-z) on [0,2] has coefficients -64T0-128T1-64T2.
    # Its two endpoint values vanish while its interior maximum is32.
    coefficients = repeat(
        reshape(ComplexF64[-64, -128, -64], 1, 3, 1);
        outer = (8, 1, 1)
    )
    represented = DF.RepresentedOperator(
        execution, DF.ChebyshevTable(execution, [0.0, 2.0], coefficients), 4, 2,
    )
    for (operator, nodes) in ((constant, [0.0, 2.0]), (represented, [0.0, 0.5]))
        fixture = df32flow_fixture(; operator, nodes, step = 1.0, steps = 1)
        final = DF.wide_operation(fixture.flow.binding.operator, 0.0, 1.0; step = 1.0)
        @test final.certificate.status === :valid
        session = df32flow_session(fixture)
        df32flow_turn!(session)
        original = copy(session.field)
        report = DF.advance!(session; deadline = 60.0)
        @test report.kind == DF.SESSION_FAILED && !report.poisoned
        @test fixture.source.calls == 0
        @test session.field == original && session.state.accepted == 0
        @test fixture.flow.plans.generation == 0
    end
end

@testset "DF32 dense attachment refuses at actual adaptive session preparation" begin
    fixture = df32flow_fixture(; replay = false)
    prepared = df32flow_session(fixture)
    original = copy(fixture.state)
    failure = try
        DF.RungeKuttaSession(
            fixture.execution, fixture.state, fixture.spec.start, 1 / 64,
            fixture.scheme, prepared.control, prepared.norm, fixture.schedule,
            fixture.source, fixture.flow; arithmetic = fixture.arithmetic,
            dense_weights = zeros(length(fixture.scheme.nodes)),
        )
    catch exception
        exception
    end
    @test failure isa ArgumentError
    @test occursin("dense observation requires LinearFlow", sprint(showerror, failure))
    @test fixture.source.calls == 0 && fixture.flow.plans.generation == 0
    @test fixture.state == original
end

@testset "Certified zero phases preserve each signed-zero lane" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = DF.KernelExecution(DF.KernelAbstractions.CPU(), tasks, true)
        state = reshape(
            ComplexF32[
                complex(0.0f0, -0.0f0), complex(-0.0f0, -0.0f0),
                complex(1.0f0, -0.0f0), complex(-0.0f0, 0.0f0),
                complex(0.0f0, 0.0f0), complex(-1.0f0, -0.0f0),
                complex(1.0f0, 0.5f0), complex(-0.5f0, 0.25f0),
            ], 4, 2
        )
        rates = zeros(ComplexF64, size(state))
        rates[7], rates[8] = -0.25 + 0.5im, 0.125 - 0.25im
        operator = DF.DiagonalOperator(execution, rates)
        fixture = df32flow_fixture(; tasks, initial = state, operator)
        token = DF.prepare_df32_trial!(fixture.flow, fixture.scheme, df32flow_site(fixture))
        output = similar(state)
        for (origin, destination, direction) in ((0.0, 1 / 64, 1), (1 / 64, 0.0, -1))
            token(output, state, origin, destination)
            for index in 1:6
                @test reinterpret(UInt32, [real(output[index]), imag(output[index])]) ==
                    reinterpret(UInt32, [real(state[index]), imag(state[index])])
            end
            for index in 7:8
                phase = Complex{BigFloat}(rates[index]) * (direction * big(1) / 64)
                expected = exp(phase) * Complex{BigFloat}(state[index])
                bound = df32flow_product_bound(state[index], phase)
                @test bound > 0 && abs(Complex{BigFloat}(output[index]) - expected) <= bound
            end
            inplace = copy(state)
            token(inplace, inplace, origin, destination)
            @test reinterpret(UInt32, vec(inplace)) == reinterpret(UInt32, vec(output))
        end
        @test df32flow_allocated(output, fixture) == 0
    end
end
