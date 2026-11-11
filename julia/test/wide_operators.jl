# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

wideop_exact(x::Union{Float32, Float64}) = Rational{BigInt}(x)
wideop_exact(x::SolpropJulia.DF32) = wideop_exact(x.hi) + wideop_exact(x.lo)
wideop_exact(x::Complex) = complex(wideop_exact(real(x)), wideop_exact(imag(x)))
wideop_bits(x) = reinterpret(UInt32, vec(copy(x)))
wideop_buffer(shape) = fill(
    Complex{SolpropJulia.DF32}(zero(SolpropJulia.DF32), zero(SolpropJulia.DF32)), shape,
)

function wideop_polynomial(table, position; derivative = false, before = false)
    # Independent defining Chebyshev recurrence (not Clenshaw), with exact rational
    # arithmetic. All represented binary64 coefficients and coordinates are exact.
    interval = clamp(
        before ? searchsortedfirst(table.breakpoints, position) - 1 :
            searchsortedlast(table.breakpoints, position),
        1, length(table.breakpoints) - 1,
    )
    a = wideop_exact(table.breakpoints[interval])
    b = wideop_exact(table.breakpoints[interval + 1])
    x = (2wideop_exact(position) - a - b) / (b - a)
    result = fill(complex(zero(x)), size(table.coefficients, 1))
    previous, current = one(x), x
    dprevious, dcurrent = zero(x), one(x)
    for degree in 0:(size(table.coefficients, 2) - 1)
        if degree == 0
            basis, dbasis = one(x), zero(x)
        elseif degree == 1
            basis, dbasis = x, one(x)
        else
            next = 2x * current - previous
            dnext = 2current + 2x * dcurrent - dprevious
            previous, current = current, next
            dprevious, dcurrent = dcurrent, dnext
            basis, dbasis = current, dcurrent
        end
        factor = derivative ? dbasis * 2 / (b - a) : basis
        for row in eachindex(result)
            coefficient = wideop_exact(table.coefficients[row, degree + 1, interval])
            result[row] += coefficient * factor
        end
    end
    return result
end

function wideop_reference(
        operator::SolpropJulia.DiagonalOperator, z0, z1;
        derivative = false, before = false,
    )
    scale = derivative ? one(wideop_exact(z0)) : wideop_exact(z1) - wideop_exact(z0)
    return wideop_exact.(operator.rate) .* scale
end

function wideop_reference(
        operator::SolpropJulia.RepresentedOperator, z0, z1;
        derivative = false, before = false,
    )
    table = operator.cumulative
    result = derivative ? wideop_polynomial(table, z0; derivative, before) :
        wideop_polynomial(table, z1) - wideop_polynomial(table, z0)
    return reshape(result, operator.n_frequency, operator.n_columns)
end

function wideop_reference(
        operator::SolpropJulia.ExpansionOperator, z0, z1;
        derivative = false, before = false,
    )
    table = operator.cumulative_weights
    weights = derivative ? wideop_polynomial(table, z0; derivative, before) :
        wideop_polynomial(table, z1) - wideop_polynomial(table, z0)
    result = fill(complex(zero(wideop_exact(z0))), size(operator.terms)[1:2])
    for term in eachindex(weights), index in eachindex(result)
        fixed = wideop_exact(operator.terms[index + (term - 1) * length(result)])
        result[index] += fixed * weights[term]
    end
    return result
end

function wideop_assert_bound(output, exact, certificate; argument = true)
    @test certificate.status === :valid
    @test certificate.real_error >= 0 && certificate.imag_error >= 0
    for index in eachindex(output)
        difference = wideop_exact(output[index]) - exact[index]
        # The certificate is fixed before evaluating this independent exact target.
        @test abs(real(difference)) <= wideop_exact(certificate.real_error)
        @test abs(imag(difference)) <= wideop_exact(certificate.imag_error)
        for part in (real(output[index]), imag(output[index]))
            @test isfinite(part)
            @test part.hi == part.hi + part.lo
        end
        if argument
            @test isnothing(
                SolpropJulia.check_df32_function_argument(
                    output[index];
                    input_error = certificate.real_error + certificate.imag_error,
                )
            )
        end
    end
    return nothing
end

function wideop_fixture(execution, kind; degree = 5)
    rate = ComplexF64[
        -0.001 - 3im 0.002 + 4im
        -0.003 + 7im 0.004 - 9im
        0.0 + 2im -0.005im
    ]
    kind == 1 && return SolpropJulia.DiagonalOperator(execution, rate)
    edges = [-1.0, 0.5, 2.0]
    count = kind == 2 ? 6 : 3
    coefficients = Array{ComplexF64}(undef, count, degree + 1, 2)
    for interval in 1:2, order in 1:(degree + 1), row in 1:count
        coefficients[row, order, interval] =
            (0.01row * (-1.0)^order + 0.1im * (row + interval)) / (order^3)
    end
    if kind == 2
        table = SolpropJulia.ChebyshevTable(
            execution, edges, coefficients; lower = -1.001, upper = 2.001,
        )
        return SolpropJulia.RepresentedOperator(execution, table, 3, 2)
    end
    table = SolpropJulia.ChebyshevTable(
        execution, edges, real.(coefficients); lower = -1.001, upper = 2.001,
    )
    terms = cat(rate, -0.25 .* rate, 0.125 .* rate; dims = 3)
    return SolpropJulia.ExpansionOperator(execution, terms, table)
end

function wideop_cycle!(output, sample, prepared, execution)
    SolpropJulia.wide_rate!(sample, 0.125, execution)
    exponent = SolpropJulia.wide_operation(prepared, 0.25, 0.375; step = 0.125)
    SolpropJulia.wide_exponent!(output, exponent, execution)
    transport = SolpropJulia.wide_operation(
        prepared, 0.25, 0.375; step = 0.125, kind = :transport, sample,
    )
    SolpropJulia.wide_transport!(output, transport, execution)
    compensation = SolpropJulia.wide_operation(
        prepared, 0.25, 0.375; step = 0.125, kind = :compensation, sample,
    )
    SolpropJulia.wide_compensation!(output, compensation, execution)
    return nothing
end

function wideop_cross_cycle!(output, sample, prepared, execution)
    exponent = SolpropJulia.wide_operation(prepared, 0.25, 0.75; step = 0.5)
    SolpropJulia.wide_exponent!(output, exponent, execution)
    transport = SolpropJulia.wide_operation(
        prepared, 0.25, 0.75; step = 0.5, kind = :transport, sample,
    )
    SolpropJulia.wide_transport!(output, transport, execution)
    compensation = SolpropJulia.wide_operation(
        prepared, 0.25, 0.75; step = 0.5, kind = :compensation, sample,
    )
    SolpropJulia.wide_compensation!(output, compensation, execution)
    return nothing
end

@testset "Wide exact represented expressions and provenance" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        geometry = SolpropJulia.WideGeometry(-1.001, 2.001; step_max = 2.0)
        for kind in 1:3, degree in (0, 1, 2, 5, 9)
            operator = wideop_fixture(execution, kind; degree)
            prepared = SolpropJulia.prepare_wide_operator(execution, operator; geometry)
            output = wideop_buffer(prepared.shape)
            for (z0, z1, step) in (
                    (0.125, 0.375, 0.25), (0.375, 0.125, 0.25),
                    (0.25, nextfloat(0.25), 1.0e-12), (0.25, 0.25, 0.125),
                    (0.25, 0.75, 0.5), (-1.001, -0.75, 0.25), (1.75, 2.001, 0.25),
                    (0.0, 0.8, 0.8), (0.0, 0.25, 0.125),
                )
                operation = SolpropJulia.wide_operation(prepared, z0, z1; step)
                certificate = operation.certificate
                SolpropJulia.wide_exponent!(output, operation, execution)
                exact = wideop_reference(operator, z0, z1)
                wideop_assert_bound(output, exact, certificate)
                if z0 == z1
                    @test all(x -> iszero(real(x)) && iszero(imag(x)), output)
                    @test certificate.real_error == certificate.imag_error == 0
                end
            end
            for position in (0.125, 0.5, nextfloat(0.5)), before in (false, true)
                sample = SolpropJulia.prepare_wide_rate(
                    execution, prepared, position; before,
                )
                exact_rate = wideop_reference(
                    operator, position, position; derivative = true, before,
                )
                for index in eachindex(sample.values)
                    difference = wideop_exact(sample.values[index]) - exact_rate[index]
                    @test abs(real(difference)) <= wideop_exact(real(sample.errors[index]))
                    @test abs(imag(difference)) <= wideop_exact(imag(sample.errors[index]))
                end
            end
            sample = SolpropJulia.prepare_wide_rate(execution, prepared, -0.875)
            for destination in (nothing, 0.4)
                operation = SolpropJulia.wide_operation(
                    prepared, 0.25, 0.5; step = 0.2, kind = :transport, sample, destination,
                )
                certificate = operation.certificate
                SolpropJulia.wide_transport!(output, operation, execution)
                age = wideop_exact(sample.position) - wideop_exact(0.25)
                numerator = destination === nothing ? wideop_exact(0.2) - age :
                    wideop_exact(destination) - wideop_exact(sample.position)
                exact = wideop_reference(operator, 0.25, 0.5) .*
                    (numerator / wideop_exact(0.2))
                wideop_assert_bound(output, exact, certificate)
            end
            operation = SolpropJulia.wide_operation(
                prepared, 0.25, 0.5; step = 0.2, kind = :compensation, sample,
            )
            certificate = operation.certificate
            SolpropJulia.wide_compensation!(output, operation, execution)
            sampled = wideop_reference(
                operator, sample.position, sample.position; derivative = true,
            )
            exact = sampled - wideop_reference(operator, 0.25, 0.5) ./ wideop_exact(0.2)
            wideop_assert_bound(output, exact, certificate; argument = false)
            wideop_cycle!(output, sample, prepared, execution)
            wideop_cycle!(output, sample, prepared, execution)
            @test (@allocated wideop_cycle!(output, sample, prepared, execution)) == 0
            wideop_cross_cycle!(output, sample, prepared, execution)
            wideop_cross_cycle!(output, sample, prepared, execution)
            @test (@allocated wideop_cross_cycle!(output, sample, prepared, execution)) == 0
        end
    end
end

@testset "Correlated cancellation, cross-knot refusal and range boundaries" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    geometry = SolpropJulia.WideGeometry(0.0, 1.0; history_lower = -10.0)
    for offset in (0.0, nextfloat(1.0e20))
        coefficients = reshape(ComplexF64[offset, 2, 3, 4], 1, 4, 1)
        table = SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], coefficients)
        operator = SolpropJulia.RepresentedOperator(execution, table, 1, 1)
        prepared = SolpropJulia.prepare_wide_operator(execution, operator; geometry)
        output = wideop_buffer((1, 1))
        z0, z1 = 0.25, nextfloat(0.25)
        operation = SolpropJulia.wide_operation(prepared, z0, z1; step = z1 - z0)
        SolpropJulia.wide_exponent!(output, operation, execution)
        wideop_assert_bound(
            output, wideop_reference(operator, z0, z1), operation.certificate,
        )
        @test operation.certificate.real_error < 1.0e-25
    end
    coefficients = reshape(ComplexF64[nextfloat(1.0e20), nextfloat(1.0e20)], 1, 1, 2)
    table = SolpropJulia.ChebyshevTable(execution, [0.0, 0.5, 1.0], coefficients)
    prepared = SolpropJulia.prepare_wide_operator(
        execution, SolpropJulia.RepresentedOperator(execution, table, 1, 1); geometry,
    )
    sentinel = Complex(SolpropJulia.prepare_df32(3.0), SolpropJulia.prepare_df32(7.0))
    output = fill(sentinel, 1, 1)
    original = wideop_bits(output)
    operation = SolpropJulia.wide_operation(
        prepared, prevfloat(0.5), 0.5; step = eps(0.5) / 2,
    )
    @test operation.certificate.status === :assembly_error
    @test_throws ArgumentError SolpropJulia.wide_exponent!(output, operation, execution)
    @test wideop_bits(output) == original
    for generator in (17.0 + 0im, 0.0 + (2.0^21) * im)
        operator = SolpropJulia.DiagonalOperator(execution, fill(generator, 1, 1))
        p = SolpropJulia.prepare_wide_operator(execution, operator; geometry)
        op = SolpropJulia.wide_operation(p, 0.0, 1.0; step = 1.0)
        @test op.certificate.status === :function_domain
        @test_throws ArgumentError SolpropJulia.wide_exponent!(output, op, execution)
        @test wideop_bits(output) == original
    end
    # A small final exponent does not legalize a huge divided intermediate.
    operator = SolpropJulia.DiagonalOperator(execution, fill(1.0e30 + 0im, 1, 1))
    huge = SolpropJulia.prepare_wide_operator(execution, operator; geometry)
    op = SolpropJulia.wide_operation(huge, 0.0, 1.0e-30; step = 1.0e-40)
    @test op.certificate.status === :intermediate_range
    @test_throws ArgumentError SolpropJulia.wide_exponent!(output, op, execution)
    @test wideop_bits(output) == original
end

@testset "Wide conversion underflow and exact scalar rounding radii" begin
    for value in (
            0.0, -0.0, 2.0^-149, 2.0^-150, 2.0^-172, 2.0^-400,
            1.0, nextfloat(1.0), Float64(pi), 1.0e20, nextfloat(1.0e20), -1.0e20,
        )
        split = SolpropJulia._wide_split(SolpropJulia._wide_host(value))
        @test split.valid
        difference = wideop_exact(split.value) - wideop_exact(value)
        @test abs(difference) <= wideop_exact(split.error)
        @test split.value.hi == split.value.hi + split.value.lo
    end
    # Normal leading products can still lose a subnormal residual. The additive
    # primitive bound is essential; no residual-floor or FTZ assumption is made.
    scalar = Float64((1.0f0 + 2.0f0^-23) * 2.0f0^-63)
    ball = SolpropJulia._wide_split(SolpropJulia._wide_host(scalar))
    product = ball * ball
    @test product.valid
    difference = wideop_exact(product.value) - wideop_exact(scalar)^2
    @test abs(difference) <= wideop_exact(product.error)
    @test abs(wideop_exact(product.value) - wideop_exact(scalar)^2) > 0
    for a in (-1.0e-30, nextfloat(0.25), 1.0e20), b in (0.125, 3.0, 1.0e-30)
        first, second = SolpropJulia._wide_host(a), SolpropJulia._wide_host(b)
        for (result, exact) in (
                (first + second, wideop_exact(a) + wideop_exact(b)),
                (first - second, wideop_exact(a) - wideop_exact(b)),
                (first * second, wideop_exact(a) * wideop_exact(b)),
                (first / second, wideop_exact(a) / wideop_exact(b)),
            )
            @test abs(wideop_exact(result.value) - exact) <= wideop_exact(result.error)
        end
    end
    @test SolpropJulia._wide_product(nextfloat(0.0), 0.5) > 0
    @test SolpropJulia._wide_quotient(nextfloat(0.0), 2.0) > 0
    for exponent0 in (-149, -126, -80, -63, -30, 0, 30, 80),
            exponent1 in (-149, -126, -80, -63, -30, 0, 30, 80),
            sign in (-1.0, 1.0)
        a = sign * (1 + 2.0^-23) * 2.0^exponent0
        b = (1 + 2.0^-24 + 2.0^-47) * 2.0^exponent1
        first = SolpropJulia._wide_split(SolpropJulia._wide_host(a))
        second = SolpropJulia._wide_split(SolpropJulia._wide_host(b))
        for (result, exact) in (
                (first + second, wideop_exact(a) + wideop_exact(b)),
                (first - second, wideop_exact(a) - wideop_exact(b)),
                (first * second, wideop_exact(a) * wideop_exact(b)),
            )
            if result.valid
                @test abs(wideop_exact(result.value) - exact) <= wideop_exact(result.error)
                @test result.value.hi == result.value.hi + result.value.lo
            else
                @test result.error == Inf
            end
        end
    end
end

@testset "Wide narrow intervals preserve host geometry and scalar provenance" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    lower, upper = 1.0, 1.0 + 2.0^-30
    coefficients = reshape(ComplexF64[nextfloat(1.0e20), 1im, -0.25im, 0.125im], 1, 4, 1)
    table = SolpropJulia.ChebyshevTable(execution, [lower, upper], coefficients)
    operator = SolpropJulia.RepresentedOperator(execution, table, 1, 1)
    geometry = SolpropJulia.WideGeometry(lower, upper)
    prepared = SolpropJulia.prepare_wide_operator(execution, operator; geometry)
    output = wideop_buffer((1, 1))
    z0, z1 = lower + 2.0^-32, nextfloat(lower + 2.0^-32)
    operation = SolpropJulia.wide_operation(prepared, z0, z1; step = 2.0^-40)
    certificate = operation.certificate
    SolpropJulia.wide_exponent!(output, operation, execution)
    wideop_assert_bound(output, wideop_reference(operator, z0, z1), certificate)
    @test operation.z0 == z0 && operation.z1 == z1 && operation.step == 2.0^-40
    @test operation.scalar_certificate.beta.error > 0
    @test operation.scalar_certificate.x0.error > 0
    @test operation.scalars.beta.hi == operation.scalar_certificate.beta.value.hi
    @test operation.scalars.beta.lo == operation.scalar_certificate.beta.value.lo
    # A valid but stricter caller geometry remains an explicit admission boundary.
    strict = SolpropJulia.prepare_wide_operator(
        execution, operator; geometry = SolpropJulia.WideGeometry(
            lower, upper; step_min = 2.0^-35,
        ),
    )
    @test_throws ArgumentError SolpropJulia.wide_operation(strict, z0, z1; step = 2.0^-40)
end

@testset "Wide rate certification fails before refreshing owned history" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    coefficients = zeros(ComplexF64, 1, 13, 1)
    coefficients[1, end, 1] = 1.0e30im
    table = SolpropJulia.ChebyshevTable(
        execution, [-1.0, 1.0], coefficients; lower = -20.0, upper = 20.0,
    )
    prepared = SolpropJulia.prepare_wide_operator(
        execution, SolpropJulia.RepresentedOperator(execution, table, 1, 1);
        geometry = SolpropJulia.WideGeometry(-20.0, 20.0),
    )
    sample = SolpropJulia.prepare_wide_rate(execution, prepared, 0.0)
    saved = (
        wideop_bits(sample.values), copy(sample.errors), sample.generation, sample.position,
    )
    @test_throws ArgumentError SolpropJulia.wide_rate!(sample, 20.0, execution)
    @test wideop_bits(sample.values) == saved[1]
    @test sample.errors == saved[2]
    @test sample.generation == saved[3] && sample.position == saved[4]
    # Endpoints alone do not bound a represented polynomial's interior phase.
    coefficients = reshape(ComplexF64[1.0e7im, 0, -1.0e7im], 1, 3, 1)
    table = SolpropJulia.ChebyshevTable(execution, [-1.0, 1.0], coefficients)
    operator = SolpropJulia.RepresentedOperator(execution, table, 1, 1)
    prepared = SolpropJulia.prepare_wide_operator(
        execution, operator; geometry = SolpropJulia.WideGeometry(-1.0, 1.0),
    )
    endpoint = SolpropJulia.wide_operation(prepared, -1.0, 1.0; step = 2.0)
    interior = SolpropJulia.wide_operation(prepared, -1.0, 0.0; step = 1.0)
    @test endpoint.certificate.status in (:valid, :assembly_error)
    @test interior.certificate.status === :function_domain
    @test wideop_reference(operator, -1.0, 1.0)[1] == 0
    @test imag(wideop_reference(operator, -1.0, 0.0)[1]) == 20_000_000
end

@testset "Certified assembly feeds protected one-step functions" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    for kind in 1:3
        operator = wideop_fixture(execution, kind; degree = 7)
        prepared = SolpropJulia.prepare_wide_operator(
            execution, operator; geometry = SolpropJulia.WideGeometry(-1.001, 2.001),
        )
        operation = SolpropJulia.wide_operation(prepared, 0.125, 0.375; step = 0.25)
        certificate = operation.certificate
        output = wideop_buffer(prepared.shape)
        SolpropJulia.wide_exponent!(output, operation, execution)
        input_error = BigFloat(certificate.real_error) + BigFloat(certificate.imag_error)
        # DF32_BOUNDS (12) bounds the function of the stored argument. Perturbing
        # that argument adds exp(e)-1; multiply, rather than drop, the cross term.
        leaf_error = 8192big(2)^-48 + big(2)^40 * big(2)^-150
        combined = (1 + leaf_error) * exp(input_error) - 1
        @test 0 < combined < big(2)^-24
        pair_bound = 5big(2)^-24
        exact = wideop_reference(operator, 0.125, 0.375)
        for index in eachindex(output)
            target = Complex{BigFloat}(exact[index])
            forward = SolpropJulia._df32_exp(output[index])
            reference = exp(target)
            observed = Complex{BigFloat}(wideop_exact(forward))
            @test abs(observed - reference) <= combined * abs(reference) + big(2)^-200
            pair = SolpropJulia._df32_exp_pair(output[index])
            for (value, sign) in zip(pair, (1, -1))
                reference = exp(sign * target)
                @test abs(Complex{BigFloat}(value) - reference) <=
                    pair_bound * abs(reference) + big(2)^-200
            end
        end
    end
end

@testset "Wide ownership, actual restored rates and transactional refusals" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    geometry = SolpropJulia.WideGeometry(-1.001, 2.001)
    operator = wideop_fixture(execution, 2)
    prepared = SolpropJulia.prepare_wide_operator(execution, operator; geometry)
    saved_coefficients = wideop_bits(prepared.coefficients)
    fill!(operator.cumulative.coefficients, 99.0 + 4im)
    @test wideop_bits(prepared.coefficients) == saved_coefficients
    stored = fill(-10.0 + 100im, prepared.shape)
    prior = fill(1.0e-9 + 2.0e-9im, prepared.shape)
    sample = SolpropJulia.restore_wide_rate(prepared, -0.875, stored; prior_errors = prior)
    @test sample.provenance === :restored
    output = wideop_buffer(prepared.shape)
    foreign_operator = SolpropJulia.DiagonalOperator(execution, fill(1.0im, 1, 1))
    foreign_prepared = SolpropJulia.prepare_wide_operator(
        execution, foreign_operator; geometry,
    )
    foreign_sample = SolpropJulia.prepare_wide_rate(execution, foreign_prepared, 0.0)
    for kind in (:exponent, :transport, :compensation)
        @test_throws ArgumentError SolpropJulia.wide_operation(
            prepared, 0.25, 0.375; step = 0.125, kind, sample = foreign_sample,
        )
        @test_throws ArgumentError SolpropJulia.wide_operation(
            prepared, 0.25, 0.375; step = 0.125, kind, sample, sample_z = 0.0,
        )
    end
    operation = SolpropJulia.wide_operation(
        prepared, 0.25, 0.375; step = 0.125, kind = :compensation, sample,
    )
    SolpropJulia.wide_compensation!(output, operation, execution)
    @test all(x -> real(wideop_exact(x)) < -9, output)
    fill!(stored, 0)
    fill!(prior, 0)
    @test all(x -> imag(wideop_exact(x)) == 100, sample.values)
    @test all(x -> real(x) >= 1.0e-9 && imag(x) >= 2.0e-9, sample.errors)
    original = wideop_bits(output)
    SolpropJulia.wide_rate!(sample, 0.125, execution)
    @test_throws ArgumentError SolpropJulia.wide_compensation!(output, operation, execution)
    @test wideop_bits(output) == original
    saved_values, saved_errors, saved_generation =
        wideop_bits(sample.values), copy(sample.errors), sample.generation
    @test_throws ArgumentError SolpropJulia.wide_rate!(sample, 5.0, execution)
    @test wideop_bits(sample.values) == saved_values
    @test sample.errors == saved_errors && sample.generation == saved_generation
    @test sample.position == 0.125
    valid = SolpropJulia.wide_operation(prepared, 0.25, 0.375; step = 0.125)
    @test_throws ArgumentError SolpropJulia.wide_transport!(output, valid, execution)
    @test_throws ArgumentError SolpropJulia.wide_exponent!(
        view(output, [1, 1, 3], :), valid, execution,
    )
    @test wideop_bits(output) == original
    GC.@preserve prepared begin
        overlap = unsafe_wrap(
            Array, pointer(prepared.coefficients, 2), prepared.shape; own = false,
        )
        @test_throws ArgumentError SolpropJulia.wide_exponent!(overlap, valid, execution)
        @test wideop_bits(prepared.coefficients) == saved_coefficients
    end
    carrier = wideop_buffer((6, 2))
    reversed = view(carrier, 6:-2:2, :)
    SolpropJulia.wide_exponent!(reversed, valid, execution)
    @test any(x -> !iszero(imag(x)), reversed)
    @test_throws UndefKeywordError SolpropJulia.restore_wide_rate(prepared, 0.0, stored)
    @test_throws ArgumentError SolpropJulia.restore_wide_rate(
        prepared, 0.0, stored; prior_errors = fill(-1.0 + 0im, prepared.shape),
    )
    @test_throws ArgumentError SolpropJulia.wide_operation(prepared, 0.0, 1.0; step = 0.0)
    @test_throws ArgumentError SolpropJulia.wide_operation(
        prepared, 0.0, 1.0; step = 0.1, kind = :unknown,
    )
    @test_throws ArgumentError SolpropJulia.wide_operation(
        prepared, 0.0, 1.0; step = 0.1, kind = :compensation,
    )
    @test_throws ArgumentError SolpropJulia.wide_operation(
        prepared, 0.0, 1.0; step = 0.1, sample_z = -20.0,
    )
    # CPU preparation copies valid strided inputs without narrowing their layout.
    carrier = reshape(ComplexF64.(1:24), 6, 4)
    strided = view(carrier, 1:2:5, 1:2:3)
    diagonal = SolpropJulia.DiagonalOperator(strided)
    prepared_stride = SolpropJulia.prepare_wide_operator(execution, diagonal; geometry)
    @test prepared_stride.shape == (3, 2)
    @test wideop_exact.(prepared_stride.coefficients[:, 1, 1]) ==
        vec(wideop_exact.(strided))
    @test_throws ArgumentError SolpropJulia.prepare_wide_operator(
        execution, SolpropJulia.DiagonalOperator(execution, ComplexF32.(strided)); geometry,
    )
    @test_throws ArgumentError SolpropJulia.WideGeometry(0.0, Inf)
    @test_throws ArgumentError SolpropJulia.WideGeometry(1.0, 0.0)
    @test_throws ArgumentError SolpropJulia.WideGeometry(0.0, 1.0; step_min = 2.0)
end
