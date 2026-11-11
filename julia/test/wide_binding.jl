# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

widebind_exact(x::Union{Float32, Float64}) = Rational{BigInt}(x)
widebind_exact(x::SolpropJulia.DF32) = widebind_exact(x.hi) + widebind_exact(x.lo)
widebind_exact(x::Complex) = complex(widebind_exact(real(x)), widebind_exact(imag(x)))

function widebind_schedule(; replay = false)
    return SolpropJulia.MarchSchedule(
        planes = [1.0], breakpoints = [0.0, 1.0], hard_kinks = Float64[],
        stretch = 1.2, step_planes = 0,
        replay_steps = replay ? [1.0] : Float64[],
        replay_edges = replay ? [0.0, 1.0] : Float64[],
    )
end

function widebind_spec(;
        start = 0.0, replay = false, method = :rk, order = 0,
        geometry = SolpropJulia.WideGeometry(0.0, 1.0; step_max = 1.0),
        arithmetic = SolpropJulia.SourceArithmetic(Float32; state_exponent = 8),
        source_absent = false,
    )
    tableau = method === :rk ? SolpropJulia.EmbeddedTableau(
            [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
        ) : nothing
    return SolpropJulia.WideStartSpec(
        start, tableau, widebind_schedule(; replay); geometry, arithmetic, method,
        order_max = order, source_absent,
    )
end

function widebind_rate_operator(execution; two_intervals = false)
    coefficients = two_intervals ? reshape(
            ComplexF64[0, 0.25 + 0.5im, 0.25 + 0.5im, 0.75 - 0.25im], 1, 2, 2,
        ) : reshape(ComplexF64[2.0^50, 1.0 + 2.0im], 1, 2, 1)
    edges = two_intervals ? [0.0, 0.5, 1.0] : [0.0, 1.0]
    table = SolpropJulia.ChebyshevTable(execution, edges, coefficients)
    return SolpropJulia.RepresentedOperator(execution, table, 1, 1)
end

function widebind_cycle(output, sample, prepared, execution)
    operation = SolpropJulia.wide_operation(
        prepared, 0.5, 0.625; step = 0.125, kind = :compensation, sample,
    )
    SolpropJulia.wide_compensation!(output, operation, execution)
    return nothing
end

@testset "Actual wide start method and replay bindings" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    operator = SolpropJulia.DiagonalOperator(execution, fill(0.125im, 1, 1))
    for method in (:rk, :low_storage, :adams)
        spec = widebind_spec(; replay = true, method, order = method === :adams ? 6 : 0)
        binding = SolpropJulia.prepare_wide_start(execution, operator, spec)
        @test binding.spec === spec
        @test binding.operator !== nothing
        @test isempty(binding.positions) && isempty(binding.rates)
        @test binding.origin_wide === :unknown
        @test spec.arithmetic === SolpropJulia.SourceArithmetic(Float32; state_exponent = 8)
        @test (spec.tableau !== nothing) == (method === :rk)
    end
    represented = widebind_rate_operator(execution)
    weights = SolpropJulia.ChebyshevTable(
        execution, [0.0, 1.0], reshape([0.5, 0.5], 1, 2, 1),
    )
    expansion = SolpropJulia.ExpansionOperator(
        execution, fill(2.0 + 4im, 1, 1, 1), weights,
    )
    for varying in (represented, expansion)
        spec = widebind_spec()
        bound = SolpropJulia.prepare_wide_start(execution, varying, spec)
        @test bound.operator.shape == (1, 1)
        @test isempty(bound.rates)
        operation = SolpropJulia.wide_operation(bound.operator, 0.0, 0.125; step = 0.125)
        @test operation.certificate.status === :valid
    end
    @test_throws ArgumentError widebind_spec(; method = :low_storage)
    @test_throws ArgumentError widebind_spec(; method = :other)
    @test_throws ArgumentError widebind_spec(; order = 1)
    @test_throws ArgumentError widebind_spec(; method = :adams, order = 0)
    @test_throws ArgumentError widebind_spec(; start = -0.1)
    @test_throws ArgumentError widebind_spec(;
        arithmetic = SolpropJulia.SourceArithmetic(Float64),
    )
    @test_throws ArgumentError widebind_spec(;
        replay = true, geometry = SolpropJulia.WideGeometry(0.0, 1.0; step_max = 0.01),
    )
    @test_throws ArgumentError widebind_spec(;
        geometry = SolpropJulia.WideGeometry(0.0, 1.0; step_min = 0.01),
    )
    @test_throws ArgumentError widebind_spec(; geometry = nothing)
    @test_throws ArgumentError widebind_spec(; start = 1.0)
    completed = widebind_spec(; start = 1.0, replay = true, geometry = nothing)
    done = SolpropJulia.prepare_wide_start(execution, operator, completed)
    @test done.operator === nothing && done.spec.geometry === nothing
    @test isempty(done.rates)
    narrow_table = SolpropJulia.ChebyshevTable(
        execution, [0.0, 1.0], reshape(Float32[0.0, 0.5], 1, 2, 1),
    )
    narrow_edges = SolpropJulia.ChebyshevTable(
        execution, Float32[0.0, 1.0], reshape([0.0, 0.5], 1, 2, 1),
    )
    for narrow in (
            SolpropJulia.DiagonalOperator(execution, fill(ComplexF32(0.125im), 1, 1)),
            SolpropJulia.RepresentedOperator(execution, narrow_table, 1, 1),
            SolpropJulia.RepresentedOperator(execution, narrow_edges, 1, 1),
            SolpropJulia.ExpansionOperator(
                execution, fill(ComplexF32(2im), 1, 1, 1), weights,
            ),
            SolpropJulia.ExpansionOperator(execution, fill(2.0im, 1, 1, 1), narrow_table),
        )
        @test_throws ArgumentError SolpropJulia.prepare_wide_start(
            execution, narrow, completed,
        )
    end
    # A completed start checks the storage contract without splitting values or
    # inventing an operation that would require DF32's numerical range.
    huge_table = SolpropJulia.ChebyshevTable(
        execution, [0.0, 1.0], fill(floatmax(Float64), 1, 1, 1),
    )
    for huge in (
            SolpropJulia.DiagonalOperator(
                execution, fill(complex(floatmax(Float64)), 1, 1),
            ),
            SolpropJulia.RepresentedOperator(execution, huge_table, 1, 1),
            SolpropJulia.ExpansionOperator(
                execution, fill(complex(floatmax(Float64)), 1, 1, 1), huge_table,
            ),
        )
        finished = SolpropJulia.prepare_wide_start(execution, huge, completed)
        @test finished.operator === nothing && isempty(finished.rates)
    end
    @test_throws ArgumentError SolpropJulia.prepare_wide_start(
        execution, :separable, completed,
    )
    spatial = SolpropJulia.SeparableSpatialOperator(
        execution; k_squared = ComplexF64[4], k_squared_low = ComplexF64[0],
        kperp_sq = [0.0], kperp_sq_low = [0.0], frame = [0.0], spectral_rate = [0.0],
        prefactor_numerator = ComplexF64[1im], prefactor_clamped = ComplexF64[0],
        active = [true], evanescent_rate = 1.0, z_span = (0.0, 1.0),
    )
    @test_throws ArgumentError SolpropJulia.prepare_wide_start(
        execution, spatial, widebind_spec(),
    )
    @test spatial.k_squared == [4.0 + 0.0im]

    schedule = widebind_schedule(; replay = true)
    tableau = SolpropJulia.EmbeddedTableau([0.0, 2.0], zeros(2, 2), [1.0, 0.0], [0.0, 0.0])
    @test_throws ArgumentError SolpropJulia.WideStartSpec(
        0.0, tableau, schedule;
        geometry = SolpropJulia.WideGeometry(0.0, 1.0),
        arithmetic = SolpropJulia.SourceArithmetic(Float32), method = :rk,
        order_max = 0, source_absent = false,
    )
    @test_throws ArgumentError SolpropJulia.WideStartSpec(
        0.0, tableau, schedule;
        geometry = SolpropJulia.WideGeometry(0.0, 2.0),
        arithmetic = SolpropJulia.SourceArithmetic(Float32), method = :adams,
        order_max = 3, source_absent = false,
    )
end

@testset "Plain history retains actual old coordinates without sampled rates" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    operator = SolpropJulia.DiagonalOperator(execution, fill(0.0im, 1, 1))
    spec = widebind_spec(; start = 0.5, method = :adams, order = 6)
    positions, kinds, slots = [0.0, 0.5], UInt8[0, 1], Int64[7, 2]
    history = SolpropJulia.WideStartHistory(positions, kinds, slots)
    binding = SolpropJulia.prepare_wide_start(execution, operator, spec; history)
    @test binding.positions == positions && binding.slots == slots
    @test binding.identities == kinds && isempty(binding.rates)
    positions[1], slots[1], kinds[1] = 0.4, 0, 1
    @test binding.positions == [0.0, 0.5]
    @test binding.slots == [7, 2] && binding.identities == [0, 1]
    @test binding.spec.order_max == 6
    narrow = widebind_spec(;
        start = 0.5, method = :adams, order = 6,
        geometry = SolpropJulia.WideGeometry(0.4, 1.0; history_lower = 0.4),
    )
    original = SolpropJulia.WideStartHistory([0.0, 0.5], UInt8[0, 1], Int64[7, 2])
    @test_throws ArgumentError SolpropJulia.prepare_wide_start(
        execution, operator, narrow; history = original,
    )
    @test original.positions == [0.0, 0.5]
    @test_throws ArgumentError SolpropJulia.prepare_wide_start(
        execution, operator, widebind_spec(; start = 0.5); history = original,
    )
    @test_throws ArgumentError SolpropJulia.WideStartHistory(
        [0.0], [0], Int64[0],
    )
    @test_throws ArgumentError SolpropJulia.WideStartHistory(
        [0.0], UInt8[0], Int64[0]; origin_wide = :exact,
    )
    @test_throws ArgumentError SolpropJulia.prepare_wide_start(
        execution, operator, spec;
        history = SolpropJulia.WideStartHistory([0.0, 0.5], UInt8[0, 1], Int64[2, 2]),
    )
    @test_throws ArgumentError SolpropJulia.prepare_wide_start(
        execution, operator, spec;
        history = SolpropJulia.WideStartHistory([0.0, 0.4], UInt8[0, 1], Int64[0, 1]),
    )
    finished = widebind_spec(; start = 1.0, method = :adams, order = 6, geometry = nothing)
    old = SolpropJulia.WideStartHistory([0.0, 1.0], UInt8[0, 1], Int64[0, 1])
    completed = SolpropJulia.prepare_wide_start(
        execution, operator, finished; history = old,
    )
    @test completed.operator === nothing && completed.positions == [0.0, 1.0]
    @test isempty(completed.rates)
end

@testset "Saved rate reconstruction keeps values and derives an outward radius" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        operator = widebind_rate_operator(execution)
        spec = widebind_spec(; start = 0.5, method = :adams, order = 6)
        prepared = SolpropJulia.prepare_wide_operator(
            execution, operator; geometry = spec.geometry,
        )
        for stored in (2.0 + 4.0im, 100.0 + 200im, -0.0 - 0.0im, 2.0^-120 + 1.0e-12im),
                origin in (:unknown, :binary64, :df32)
            values = fill(stored, 1, 1)
            record = SolpropJulia.reconstruct_wide_rate(
                execution, prepared, 0.5, values; origin_wide = origin,
            )
            @test reinterpret(UInt64, vec(record.saved)) == reinterpret(UInt64, vec(values))
            @test record.saved !== values && record.origin_wide === origin
            @test record.sample.position == 0.5
            # Independent derivative: coefficient 1 multiplies x=2z-1.
            exact = 2widebind_exact(1.0 + 2.0im)
            actual = widebind_exact(record.sample.values[1])
            errors = record.sample.errors[1]
            @test abs(real(actual - exact)) <= widebind_exact(real(errors))
            @test abs(imag(actual - exact)) <= widebind_exact(imag(errors))
            @test isfinite(errors) && real(errors) >= 0 && imag(errors) >= 0
            values[1] = 9
            @test record.saved[1] == stored
        end

        rates = zeros(ComplexF64, 1, 1, 8)
        rates[:, :, 1] .= 100 + 200im
        rates[:, :, 3] .= -2 - 4im
        history = SolpropJulia.WideStartHistory(
            [0.0, 0.5], UInt8[0, 1], Int64[0, 2]; rates, origin_wide = :binary64,
        )
        binding = SolpropJulia.prepare_wide_start(execution, operator, spec; history)
        @test length(binding.rates) == 2
        @test binding.rates[1].saved[1] == 100 + 200im
        @test !binding.rates[1].sample.before && binding.rates[2].sample.before
        rates[:] .= 0
        @test binding.rates[1].saved[1] == 100 + 200im
        output = fill(Complex(zero(SolpropJulia.DF32), zero(SolpropJulia.DF32)), 1, 1)
        sample = binding.rates[1].sample
        widebind_cycle(output, sample, binding.operator, execution)
        widebind_cycle(output, sample, binding.operator, execution)
        @test @allocated(widebind_cycle(output, sample, binding.operator, execution)) == 0
        operation = SolpropJulia.wide_operation(
            binding.operator, 0.5, 0.625; step = 0.125, kind = :compensation, sample,
        )
        SolpropJulia.wide_rate!(sample, 0.0, execution)
        saved = copy(output)
        @test_throws ArgumentError SolpropJulia.wide_compensation!(
            output, operation, execution,
        )
        @test output == saved
        generation, before = sample.generation, copy(sample.values)
        @test_throws ArgumentError SolpropJulia.reconstruct_wide_rate(
            execution, binding.operator, 0.0, fill(ComplexF64(NaN), 1, 1),
        )
        @test sample.generation == generation && sample.values == before
        @test_throws ArgumentError SolpropJulia.prepare_wide_start(
            execution, operator, spec;
            history = SolpropJulia.WideStartHistory([0.0, 0.5], UInt8[0, 1], Int64[0, 2]),
        )
        malformed = SolpropJulia.WideStartHistory(
            [0.0, 0.5], UInt8[0, 1], Int64[0, 2]; rates = zeros(ComplexF64, 1, 1, 7),
        )
        @test_throws DimensionMismatch SolpropJulia.prepare_wide_start(
            execution, operator, spec; history = malformed,
        )
    end
end

@testset "Saved-rate discrepancy includes both words and inherited radius" begin
    values = (
        zero(SolpropJulia.DF32),
        SolpropJulia.prepare_df32(1.0),
        SolpropJulia.prepare_df32(1.0f0, 2.0f0^-80),
        SolpropJulia.prepare_df32(-1.0f0, 2.0f0^-25),
        SolpropJulia.prepare_df32(2.0^120),
        SolpropJulia.prepare_df32(2.0^-120),
    )
    for value in values,
            saved in (0.0, -0.0, nextfloat(0.0), 1.0, prevfloat(1.0), -2.0^120, 2.0^-170),
            inherited in (0.0, 2.0^-150, 2.0^-30, 1.0)
        bound = SolpropJulia._wide_saved_radius(saved, value, inherited)
        @test isfinite(bound) && bound >= 0
        # Exact rational interval corners, including low words smaller than a
        # binary64 ulp at the leading word. The derived bound is fixed first.
        for sign in (-1, 1)
            target = widebind_exact(value) + sign * widebind_exact(inherited)
            @test abs(widebind_exact(saved) - target) <= widebind_exact(bound)
        end
    end
end

@testset "Knot sides and unsafe complete-operation counterexamples" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    operator = widebind_rate_operator(execution; two_intervals = true)
    geometry = SolpropJulia.WideGeometry(0.0, 1.0)
    prepared = SolpropJulia.prepare_wide_operator(execution, operator; geometry)
    for (before, expected) in ((true, 1.0 + 2im), (false, 3.0 - 1im))
        saved = fill(0.0im, 1, 1)
        record = SolpropJulia.reconstruct_wide_rate(execution, prepared, 0.5, saved; before)
        @test record.sample.before == before
        @test real(record.sample.errors[1]) >= abs(real(expected))
        @test imag(record.sample.errors[1]) >= abs(imag(expected))
        @test iszero(record.saved[1])
    end

    # The exact represented increment is 2h even though independent binary64
    # endpoint Clenshaw evaluations both round to the same large constant.
    operator = widebind_rate_operator(execution)
    prepared = SolpropJulia.prepare_wide_operator(execution, operator; geometry)
    origin, step = 0.25, 2.0^-10
    narrow = zeros(ComplexF64, 1, 1)
    SolpropJulia.operator_exponent!(narrow, operator, origin, origin + step, execution)
    @test iszero(real(narrow[1]))
    operation = SolpropJulia.wide_operation(prepared, origin, origin + step; step)
    @test operation.certificate.status === :valid
    output = fill(Complex(zero(SolpropJulia.DF32), zero(SolpropJulia.DF32)), 1, 1)
    SolpropJulia.wide_exponent!(output, operation, execution)
    exact = 2widebind_exact(step) * widebind_exact(1.0 + 2im)
    @test abs(real(widebind_exact(output[1]) - exact)) <=
        widebind_exact(operation.certificate.real_error)
    @test abs(real(widebind_exact(narrow[1]) - exact)) > widebind_exact(2.0^-27)

    huge = SolpropJulia.DiagonalOperator(execution, fill((2.0^21) * im, 1, 1))
    prepared = SolpropJulia.prepare_wide_operator(execution, huge; geometry)
    unsafe = SolpropJulia.wide_operation(prepared, 0.0, 1.0; step = 1.0)
    @test unsafe.certificate.status === :function_domain
    saved = copy(output)
    @test_throws ArgumentError SolpropJulia.wide_exponent!(output, unsafe, execution)
    @test output == saved

    offsets = reshape(ComplexF64[nextfloat(1.0e20), nextfloat(1.0e20)], 1, 1, 2)
    table = SolpropJulia.ChebyshevTable(execution, [0.0, 0.5, 1.0], offsets)
    jumped = SolpropJulia.RepresentedOperator(execution, table, 1, 1)
    prepared = SolpropJulia.prepare_wide_operator(execution, jumped; geometry)
    cross = SolpropJulia.wide_operation(
        prepared, prevfloat(0.5), 0.5; step = eps(0.5) / 2,
    )
    @test cross.certificate.status === :assembly_error
    @test_throws ArgumentError SolpropJulia.wide_exponent!(output, cross, execution)
    @test output == saved

    # Current order/step alone misses the transport to an older retained sample.
    coefficients = reshape(ComplexF64[(2.0^21) * im, (2.0^21) * im], 1, 2, 1)
    old_table = SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], coefficients)
    old_operator = SolpropJulia.RepresentedOperator(execution, old_table, 1, 1)
    old_prepared = SolpropJulia.prepare_wide_operator(execution, old_operator; geometry)
    @test 0.01 * 2.0^22 < 2.0^20
    transported = SolpropJulia.wide_operation(
        old_prepared, 0.5, 0.51; step = 0.01, kind = :transport, sample_z = 0.0,
    )
    @test transported.certificate.status === :function_domain
    @test_throws ArgumentError SolpropJulia.wide_transport!(output, transported, execution)
    @test output == saved
end
