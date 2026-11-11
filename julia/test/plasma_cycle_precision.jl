# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import InteractiveUtils
import SolpropJulia as SP

function cycle_precision_table(::Type{T}; power = 32) where {T}
    lower, origin, spacing, intervals = 2.0^27, 27log(2.0), 1 / 32, 128
    upper = exp(origin + intervals * spacing)
    ceiling = exp(power * (origin + intervals * spacing - 32log(2.0)) - 3log(2.0))
    coefficients = zeros(T, 4, intervals)
    for index in 1:intervals
        coefficients[3, index] = power
        coefficients[4, index] = T(
            power * (origin + (index - 1) * spacing - 32log(2.0)) - 3log(2.0)
        )
    end
    return SP.IonisationRateTable(coefficients, lower, upper, ceiling, origin, spacing)
end

function cycle_precision_rules(::Type{T}; rtol = 2.0e-6) where {T}
    nodes, weights, ends = T[], T[], Int[]
    for order in (64, 128, 256, 512)
        # Golub--Welsch (1969), doi:10.1090/S0025-5718-69-99647-1:
        # the Legendre Jacobi matrix gives nodes and squared first eigenvectors.
        off = [index / sqrt(4index^2 - 1) for index in 1:(order - 1)]
        eigen = SP.LinearAlgebra.eigen(SP.LinearAlgebra.SymTridiagonal(zeros(order), off))
        append!(nodes, T.(eigen.values))
        append!(weights, T.(2 .* abs2.(eigen.vectors[1, :])))
        push!(ends, length(nodes))
    end
    budget = T(rtol)
    budget = budget > rtol ? prevfloat(budget) : budget
    return SP.CycleQuadrature(
        nodes, weights, Tuple(ends); moment_rtol = budget, negligible_rate = T(1.0e-30)
    )
end

cycle_precision_value(value::SP.DF32) = BigFloat(value.hi) + BigFloat(value.lo)

function cycle_precision_add(left, right, left_error, right_error)
    unit = big(2.0)^-24
    alpha = 3unit^2 / (1 - 4unit)
    result = left + right
    error = left_error + right_error +
        alpha * abs(cycle_precision_value(left) + cycle_precision_value(right))
    return result, error
end

function cycle_precision_product(left, right, left_error, right_error)
    unit, eta = big(2.0)^-24, big(2.0)^-150
    first, second = cycle_precision_value(left), cycle_precision_value(right)
    error = abs(first) * right_error + abs(second) * left_error +
        left_error * right_error + 7unit^2 * abs(first * second) + 5eta
    return left * right, error
end

function cycle_precision_offset_bound(magnitude, table, parameters, exponent, interval)
    unit, eta = big(2.0)^-24, big(2.0)^-150
    reduced = parameters[7]
    mantissa, power = frexp(magnitude)
    if mantissa < sqrt(0.5f0)
        mantissa *= 2.0f0
        power -= 1
    end
    logarithm = log1p(mantissa - 1.0f0)
    exact_logarithm = log(BigFloat(mantissa))
    log_error = 2unit * abs(exact_logarithm) + eta
    # This elementary-function hypothesis is checked, not inferred from disabling
    # fast math. All further allowances follow the reviewed exact word graphs.
    @test abs(BigFloat(logarithm) - exact_logarithm) <= log_error
    anchor = exponent - reduced.exponent
    ln2 = log(big(2.0))
    origin = BigFloat(table.log_field_min) - anchor * ln2
    origin_error = abs(cycle_precision_value(reduced.origin) - origin)
    logarithm_error = abs(cycle_precision_value(reduced.logtwo) - ln2)
    spacing_error = abs(cycle_precision_value(reduced.spacing) - table.d_log_field)
    integer = SP._df32_raw(Float32(power + reduced.exponent), 0.0f0)
    product, error = cycle_precision_product(
        integer, reduced.logtwo, big(0.0), logarithm_error
    )
    coordinate, error = cycle_precision_add(
        SP._df32_raw(logarithm, 0.0f0), product, log_error, error
    )
    coordinate, error = cycle_precision_add(
        coordinate, -reduced.origin, error, origin_error
    )
    index = interval - 1
    high = Float32(index)
    integer = SP._df32_raw(high, Float32(index - Int(high)))
    integer_error = abs(cycle_precision_value(integer) - index)
    knot, knot_error = cycle_precision_product(
        integer, reduced.spacing, integer_error, spacing_error
    )
    offset, error = cycle_precision_add(coordinate, -knot, error, knot_error)
    error += unit * abs(cycle_precision_value(offset)) + eta
    exact = log(BigFloat(magnitude)) + exponent * ln2 - table.log_field_min -
        index * BigFloat(table.d_log_field)
    @test abs(BigFloat(offset.hi) - exact) <= error
    @test isfinite(offset) && offset.hi == offset.hi + offset.lo
    return exact, error
end

function cycle_precision_rate_bound(magnitude, table, parameters, exponent)
    unit, eta = big(2.0)^-24, big(2.0)^-150
    if magnitude < parameters[2]
        return big(0.0), big(0.0)
    elseif magnitude >= parameters[3]
        return BigFloat(parameters[4]), big(0.0)
    end
    interval, offset = SP._plasma_log_offset(
        magnitude, parameters[7], size(table.coefficients, 2)
    )
    exact, coordinate_error = cycle_precision_offset_bound(
        magnitude, table, parameters, exponent, interval
    )
    first, second, third, fourth = BigFloat.(table.coefficients[:, interval])
    radius = max(abs(exact), abs(BigFloat(offset)))
    derivative = 3abs(first) * radius^2 + 2abs(second) * radius + abs(third)
    contribution = abs(first) * radius^3 + abs(second) * radius^2 +
        abs(third) * radius + abs(fourth)
    gamma = 3unit / (1 - 3unit)
    polynomial_error = derivative * coordinate_error + gamma * contribution +
        (1 + unit)^2 * eta * (1 + radius + radius^2)
    polynomial = ((first * exact + second) * exact + third) * exact + fourth
    computed_polynomial = muladd(
        muladd(
            muladd(
                table.coefficients[1, interval], offset,
                table.coefficients[2, interval]
            ), offset, table.coefficients[3, interval]
        ),
        offset, table.coefficients[4, interval]
    )
    @test abs(BigFloat(computed_polynomial) - polynomial) <= polynomial_error
    computed_exp = exp(computed_polynomial)
    exact_exp = exp(BigFloat(computed_polynomial))
    @test abs(BigFloat(computed_exp) - exact_exp) <= 2unit * exact_exp + eta
    unbounded = exp(polynomial)
    reference = min(unbounded, BigFloat(parameters[4]))
    # exp's finite-fixture allowance is independent of the coordinate/Horner
    # bound; the ceiling is 1-Lipschitz. No general vendor exp claim is made.
    error = unbounded * expm1(polynomial_error) +
        2unit * exp(polynomial + polynomial_error) + eta
    value = SP._ionisation_rate(magnitude, parameters)
    @test isfinite(value)
    @test abs(BigFloat(value) - reference) <= error
    return reference, error
end

function cycle_precision_panel_reference(major, table, parameters, rules, first, last)
    unit, eta = big(2.0)^-24, big(2.0)^-150
    # Fix the existing panel's represented geometry and scalar inputs. The MPFR
    # oracle independently evaluates its spline, products and sum; geometry,
    # elementary sincos and rule construction are not requalified by this test.
    upper = acos(clamp(parameters[2] / major, 0.0f0, 1.0f0))
    width, midpoint = upper / Float32(pi), upper / 2
    means = (big(0.0), big(0.0))
    errors = (big(0.0), big(0.0))
    for index in first:last
        theta = muladd(rules[1][index] + 1, midpoint, 0.0f0)
        sine, cosine = sincos(theta)
        magnitude = clamp(major * cosine, parameters[2], parameters[3])
        rate, rate_error = cycle_precision_rate_bound(magnitude, table, parameters, 13)
        weight, ratio = BigFloat(rules[2][index]), BigFloat(sine) / BigFloat(cosine)
        mean, directional = weight * rate, weight * rate * ratio^2
        first_error = weight * rate_error + unit * weight * (rate + rate_error) + eta
        # One division used twice and two products contribute at most gamma_4;
        # preserve the rate error before this positive directional amplification.
        gamma = 4unit / (1 - 4unit)
        second_error = weight * rate_error * ratio^2 +
            gamma * weight * (rate + rate_error) * ratio^2 +
            8eta * (1 + abs(ratio))^2 * (1 + weight * (rate + rate_error))
        means = (means[1] + mean, means[2] + directional)
        errors = (errors[1] + first_error, errors[2] + second_error)
    end
    count = last - first + 1
    gamma = (count - 1) * unit / (1 - (count - 1) * unit)
    # Ogita--Rump--Oishi Prop.4.5 applies to the unchanged Sum2 graph. The last
    # multiplication by the represented width contributes its separate rounding.
    bounds = ntuple(2) do component
        accumulation = (unit + gamma^2) * (means[component] + errors[component])
        inherited = errors[component] + accumulation
        abs(BigFloat(width)) * inherited +
            unit * abs(BigFloat(width)) * (means[component] + inherited) + eta
    end
    return ntuple(component -> BigFloat(width) * means[component], 2), bounds
end

@testset "Reduced plasma coordinates retain local differences" begin
    setprecision(BigFloat, 256) do
        table = cycle_precision_table(Float32)
        for exponent in (-40, 0, 13, 40, 80)
            parameters = SP._rate_parameters(table, exponent)
            for interval in (1, 2, 31, 64, 96, 128), fraction in (1 / 16, 1 / 2, 15 / 16)
                physical = exp(
                    table.log_field_min +
                        (interval - 1 + fraction) * table.d_log_field
                )
                magnitude = Float32(ldexp(physical, -exponent))
                selected, _ = SP._plasma_log_offset(magnitude, parameters[7], 128)
                @test selected == interval
                cycle_precision_rate_bound(magnitude, table, parameters, exponent)
            end
            @test SP._ionisation_rate(prevfloat(parameters[2]), parameters) === 0.0f0
            @test SP._ionisation_rate(parameters[3], parameters) === parameters[4]
            @test SP._ionisation_rate(nextfloat(parameters[3]), parameters) ===
                parameters[4]
        end
        # Both sides of frexp and spline boundaries stay covered by a bound for
        # the actual selected polynomial; no discontinuity is assumed away.
        parameters = SP._rate_parameters(table, 13)
        for field in (2.0f0^17, 2.0f0^18, 2.0f0^19, sqrt(0.5f0) * 2.0f0^19)
            for magnitude in (prevfloat(field), field, nextfloat(field))
                cycle_precision_rate_bound(magnitude, table, parameters, 13)
            end
        end
        for value in (0.0, -0.0, 2.0^-149, -2.0^-149, 1 + 2.0^-52, 700.0)
            pair = SP._plasma_log_pair(value)
            @test isfinite(pair) && pair.hi == pair.hi + pair.lo
            @test abs(cycle_precision_value(pair) - value) <=
                big(2.0)^-48 * abs(BigFloat(value)) + big(2.0)^-150
        end
        @test_throws ArgumentError SP._plasma_log_pair(1.0e100)
        midpoint = Float64(floatmax(Float32)) + 2.0^103
        @test_throws ArgumentError SP._plasma_log_pair(prevfloat(midpoint))
        for value in (1.0, 1.5, 1 + 2.0^-40), shift in (-149, -127, -126, -120, 0, 120)
            original = SP._plasma_log_pair(value)
            scaled = SP._plasma_log_scale(original, shift)
            @test isfinite(scaled) && scaled.hi == scaled.hi + scaled.lo
            @test abs(
                cycle_precision_value(scaled) -
                    ldexp(cycle_precision_value(original), shift)
            ) <= 2big(2.0)^-150
        end
    end
end

@testset "Steep scalar cycles keep the unchanged refinement budget" begin
    table, quadrature = cycle_precision_table(Float32), cycle_precision_rules(Float32)
    parameters, rules = SP._rate_parameters(table, 13), SP._cycle_parameters(quadrature)
    @test rules[3] == (64, 192, 448, 960)
    @test Float64(rules[4]) <= 2.0e-6
    @test Float64(nextfloat(rules[4])) > 2.0e-6
    # This exact represented magnitude also occurs in the reported physical
    # failure. The analytic steep table makes its neighbourhood self-contained.
    centre = reinterpret(UInt32, 466245.6f0)
    old_failures = 0
    for delta in -256:256
        major = reinterpret(Float32, UInt32(Int(centre) + delta))
        value = SP._scalar_cycle_moments(major, parameters, rules)
        @test value[3] == SP.PlasmaSuccess
        @test all(isfinite, value[1:2]) && all(>(0), value[1:2])
        old = SP._scalar_cycle_moments(major, SP._scaled_rate_parameters(table, 13), rules)
        old_failures += old[3] == SP.PlasmaQuadratureFailure
    end
    # Keep the old arithmetic as a diagnostic control, not a fitted count. This
    # catches a reversion even if one elementary implementation moves the failures.
    @test old_failures > 0
    setprecision(BigFloat, 256) do
        for delta in (-256, 0, 256)
            major = reinterpret(Float32, UInt32(Int(centre) + delta))
            first = 1
            for last in rules[3]
                reference, bound = cycle_precision_panel_reference(
                    major, table, parameters, rules, first, last
                )
                value = SP._scalar_cycle_rule(major, parameters, rules, first, last)
                for component in 1:2
                    @test bound[component] > 0
                    @test bound[component] < big(2.0)^-12 * reference[component]
                    @test abs(BigFloat(value[component]) - reference[component]) <=
                        bound[component]
                end
                first = last + 1
            end
        end
    end
    baseline = SP._scalar_cycle_moments(466245.6f0, parameters, rules)
    for exponent in (-40, 0, 13, 40, 80)
        major = ldexp(466245.6f0, 13 - exponent)
        result = SP._scalar_cycle_moments(
            major, SP._rate_parameters(table, exponent), rules
        )
        @test result[3] == SP.PlasmaSuccess
        # Exact powers preserve the represented magnitude and clipping ratios;
        # independent rounding graphs are compared with a four-operation budget.
        @test all(
            abs(Float64(result[index]) - baseline[index]) <=
                4eps(Float32) * abs(Float64(baseline[index])) for index in 1:2
        )
    end
end

function cycle_precision_allocations(samples, table, rules, work, ex)
    SP.envelope_populations!(
        samples, table, rules, work, ex; preionisation = 0.125, dt = 0.25,
        field_exponent = 13, count = 1
    )
    return @allocated SP.envelope_populations!(
        samples, table, rules, work, ex; preionisation = 0.125, dt = 0.25,
        field_exponent = 13, count = 1
    )
end

@testset "Reduced plasma coordinates reach the population kernels" begin
    for tasks in (1, 2)
        ex = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        layout = SP.ResponseLayout(9, 1, 2)
        work = SP.PopulationWorkspace(ex, Float32, layout)
        table, rules = cycle_precision_table(Float32), cycle_precision_rules(Float32)
        samples = fill(ComplexF32(466245.6f0), 9, 2)
        samples[:, 2] .= ComplexF32(NaN32, NaN32)
        original = copy(samples)
        @test cycle_precision_allocations(samples, table, rules, work, ex) == 0
        @test isequal(samples, original)
        @test SP.plasma_failure(work.status).code == SP.PlasmaSuccess
        @test all(iszero, work.rate[:, 2]) && all(iszero, work.fraction[:, 2])
        @test all(iszero, work.filtered)
        expected = 0.125 .+ 0.875 .* -expm1.(-Float64(work.rate[1, 1]) .* (0:8) / 4)
        # Four prefix levels plus survival/fraction products; the rate itself is
        # independently qualified above rather than charged twice in this bound.
        @test maximum(abs, Float64.(work.fraction[:, 1]) - expected) <= 16eps(Float32)
        for array in SP._population_arrays(work)
            fill!(array, 7)
        end
        before = map(copy, SP._population_arrays(work))
        @test_throws ArgumentError SP.envelope_populations!(
            samples, table, rules, work, ex;
            preionisation = 0.125, dt = 0.25, field_exponent = 1000
        )
        @test all(
            isequal(first, second) for (first, second) in
                zip(before, SP._population_arrays(work))
        )
    end
end

@testset "Binary64 rate arithmetic and owned binary32 primitive boundary" begin
    table = cycle_precision_table(Float64)
    for exponent in (0, 13)
        parameters = SP._rate_parameters(table, exponent)
        @test parameters isa Tuple{Matrix{Float64}, Vararg{Float64, 5}}
        for magnitude in (1.0, ldexp(3.5e9, -exponent), ldexp(7.0e9, -exponent))
            coefficients, lower, upper, cap, origin, spacing = parameters
            expected = if magnitude < lower
                0.0
            elseif magnitude >= upper
                cap
            else
                logarithm = log(magnitude)
                interval = clamp(floor(Int, (logarithm - origin) / spacing), 0, 127) + 1
                offset = logarithm - (origin + (interval - 1) * spacing)
                polynomial = muladd(
                    muladd(
                        muladd(
                            coefficients[1, interval], offset,
                            coefficients[2, interval]
                        ), offset, coefficients[3, interval]
                    ),
                    offset, coefficients[4, interval]
                )
                min(exp(polynomial), cap)
            end
            @test SP._ionisation_rate(magnitude, parameters) === expected
        end
    end
    parameters = SP._rate_parameters(cycle_precision_table(Float32), 13)
    @test log1p(0.25f0) isa Float32
    @test exp(0.25f0) isa Float32
    # This inspects our caller graph, stopping at Float32 elementary APIs. Julia
    # 1.13.3 log1p's CPU helpers use Float64 internally (#751); the tests above
    # qualify its returned values, not an all-binary32 transitive instruction set.
    # Vendor replacements require separate emitted-device and numerical evidence.
    ir = sprint() do output
        InteractiveUtils.code_llvm(
            output, SP._ionisation_rate, Tuple{Float32, typeof(parameters)};
            raw = true, debuginfo = :none
        )
    end
    @test !occursin(r"\b(?:fadd|fsub|fmul|fdiv|frem|fcmp)\b[^\n]*\bdouble\b", ir)
    @test !occursin(r"fpext\s+float[^\n]+double", ir)
    @test !occursin(r"\b(?:fast|reassoc|nnan|ninf|nsz|arcp|afn)\b", ir)
end
