# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

df32f_big(x::SolpropJulia.DF32) = BigFloat(x.hi) + BigFloat(x.lo)
df32f_big(x::Complex{SolpropJulia.DF32}) = Complex(df32f_big(real(x)), df32f_big(imag(x)))
df32f_big(x::ComplexF32) = Complex(BigFloat(real(x)), BigFloat(imag(x)))
df32f_argument(x, y) = Complex(SolpropJulia.prepare_df32(x), SolpropJulia.prepare_df32(y))

function df32f_assert_bound(value::Complex{SolpropJulia.DF32}, exact, bound)
    @test bound > 0
    for component in (real(value), imag(value))
        @test isfinite(component)
        @test component.hi == component.hi + component.lo
    end
    # MPFR reference rounding at 256 bits is far below the independent analytical
    # allowances. Keep an explicit absolute reference allowance near exact zeros.
    @test abs(df32f_big(value) - exact) <= bound + big(2.0)^-200 * max(1, abs(exact))
    return nothing
end

function df32f_phi_reference(x::Complex{BigFloat}, highest)
    result = Complex{BigFloat}[]
    if abs(x) <= 4
        # Independent direct defining series: no downward recurrence or doubling.
        for order in 0:highest
            term = Complex(inv(BigFloat(factorial(big(order)))))
            total = term
            for index in 1:256
                term = term * x / (index + order)
                total += term
                abs(term) <= big(2.0)^-240 * max(1, abs(total)) && break
            end
            push!(result, total)
        end
    else
        push!(result, exp(x))
        for order in 1:highest
            push!(result, (result[end] - inv(BigFloat(factorial(big(order - 1))))) / x)
        end
    end
    return result
end

function df32f_product_bound(first_size, first_error, second_size, second_error)
    unit, tiny = big(2.0)^-24, big(2.0)^-150
    mu = 15unit^2
    return (1 + mu) * (
        first_size * second_error + second_size * first_error + first_error * second_error
    ) + mu * first_size * second_size + 15tiny
end

function df32f_sum_bound(first_size, first_error, second_size, second_error)
    alpha = 4big(2.0)^-48
    return (1 + alpha) * (first_error + second_error) + alpha * (first_size + second_size)
end

function df32f_phi_small_bounds(x, highest, squarings)
    unit, tiny = big(2.0)^-24, big(2.0)^-150
    alpha = 4unit^2
    reduced = x / big(2)^squarings
    radius = abs(reduced)
    @test radius <= big(1) / 2
    constants = [inv(BigFloat(factorial(big(index)))) for index in 0:highest]
    errors = zeros(BigFloat, highest + 1)
    sizes = exp(max(real(reduced), 0)) .* constants
    sizes[1] = exp(real(reduced))
    # DF32_BOUNDS.md (14), using S_N(r) <= exp(r)/N!. The fixed allowance
    # covers all 24 terms, not just the terms that happen to survive rounding.
    tail = radius^24 / (
        factorial(big(24 + highest)) * (1 - radius / (25 + highest))
    )
    errors[end] = 128unit^2 * exp(radius) * constants[end] + 1024tiny + tail
    for order in (highest - 1):-1:1
        product = df32f_product_bound(radius, big(0), sizes[order + 2], errors[order + 2])
        errors[order + 1] = (1 + alpha) * (product + 2unit^2 * constants[order + 1]) +
            alpha * sizes[order + 1]
    end
    errors[1] = (64unit^2 + 64tiny) * sizes[1]
    for unused in 1:squarings
        previous_errors, previous_sizes = copy(errors), copy(sizes)
        errors[1] = df32f_product_bound(
            previous_sizes[1], previous_errors[1], previous_sizes[1], previous_errors[1],
        )
        for order in 1:highest
            size = previous_sizes[1] * previous_sizes[order + 1]
            error = df32f_product_bound(
                previous_sizes[1], previous_errors[1],
                previous_sizes[order + 1], previous_errors[order + 1],
            )
            for index in 1:order
                coefficient = constants[order - index + 1]
                product_size = previous_sizes[index + 1] * coefficient
                product_error = df32f_product_bound(
                    previous_sizes[index + 1], previous_errors[index + 1],
                    coefficient, 2unit^2 * coefficient,
                )
                error = df32f_sum_bound(size, error, product_size, product_error)
                size += product_size
            end
            errors[order + 1] = error / big(2)^order + 3tiny
        end
        reduced *= 2
        sizes = exp(max(real(reduced), 0)) .* constants
        sizes[1] = exp(real(reduced))
    end
    return errors .+ 1024tiny
end

function df32f_phi_large_bounds(x, highest)
    unit, tiny = big(2.0)^-24, big(2.0)^-150
    alpha = 4unit^2
    inverse_size = inv(abs(x))
    inverse_error = 32unit^2 / abs(x) + 32(1 + abs(x)) * tiny
    size = exp(real(x))
    errors = BigFloat[(8192unit^2 + big(2)^40 * tiny) * size]
    for order in 1:highest
        constant = inv(BigFloat(factorial(big(order - 1))))
        difference_size = size + constant
        difference_error = (1 + alpha) * (errors[end] + 2unit^2 * constant) +
            alpha * difference_size
        push!(
            errors, df32f_product_bound(
                difference_size, difference_error, inverse_size, inverse_error,
            )
        )
        size = min(
            exp(max(real(x), 0)) / factorial(big(order)), difference_size * inverse_size,
        )
    end
    return errors
end

function df32f_phi_small_scalings(x)
    # Independent reference decision over the computed squared norm. The margin
    # and resulting true-radius requirement are both checked below.
    high = SolpropJulia._df32_complex_abs2(x).hi
    threshold = (1 - big(2)^-22) / 4
    squarings = 0
    while BigFloat(high) > threshold
        threshold *= 4
        squarings += 1
    end
    return squarings
end

@testset "Checked DF32 function argument" begin
    check = SolpropJulia.check_df32_function_argument
    raw = SolpropJulia._df32_raw
    @test check(df32f_argument(16.0, 2.0^20); input_error = 2.0^-27) === nothing
    @test check(df32f_argument(-16.0, -2.0^20)) === nothing
    @test check(df32f_argument(-0.0, -0.0)) === nothing
    for input_error in (-1.0, Inf, NaN, nextfloat(2.0^-27))
        @test_throws ArgumentError check(df32f_argument(1.0, 2.0); input_error)
    end
    for component in (raw(NaN32, 0.0f0), raw(Inf32, 0.0f0), raw(1.0f0, 1.0f0))
        @test_throws ArgumentError check(Complex(component, zero(SolpropJulia.DF32)))
        @test_throws ArgumentError check(Complex(zero(SolpropJulia.DF32), component))
    end
    for (real_limit, imag_limit) in ((16.0f0, 0.0f0), (0.0f0, 1048576.0f0))
        for polarity in (-1.0f0, 1.0f0)
            first = raw(polarity * real_limit, polarity * floatmin(Float32))
            second = raw(polarity * imag_limit, polarity * floatmin(Float32))
            out = iszero(real_limit) ? Complex(zero(first), second) :
                Complex(first, zero(second))
            # The lost low word in a Float64 sum is exactly the admission defect
            # being guarded: an exact-domain check must still refuse this pair.
            component = iszero(real_limit) ? second : first
            @test Float64(component.hi) + Float64(component.lo) == Float64(component.hi)
            @test_throws ArgumentError check(out)
            inward = raw(component.hi, -component.lo)
            inside = iszero(real_limit) ? Complex(zero(inward), inward) :
                Complex(inward, zero(inward))
            @test check(inside) === nothing
        end
    end
    @test_throws ArgumentError check(df32f_argument(nextfloat(16.0), 0.0))
    @test_throws ArgumentError check(df32f_argument(0.0, nextfloat(2.0^20)))
end

@testset "Directed constants and fixed function-bound inequalities" begin
    setprecision(BigFloat, 256) do
        lower = setrounding(BigFloat, RoundDown) do
            2BigFloat(pi)
        end
        upper = setrounding(BigFloat, RoundUp) do
            2BigFloat(pi)
        end
        @test Float32(inv(lower)) == SolpropJulia._DF32_INVERSE_TURN
        @test Float32(inv(upper)) == SolpropJulia._DF32_INVERSE_TURN
        for word in SolpropJulia._DF32_TURN_WORDS
            @test Float32(lower) == word
            @test Float32(upper) == word
            lower = setrounding(BigFloat, RoundDown) do
                lower - BigFloat(word)
            end
            upper = setrounding(BigFloat, RoundUp) do
                upper - BigFloat(word)
            end
        end
        @test 0 < lower <= upper < big(2.0)^-74
        unit, tiny = big(2.0)^-24, big(2.0)^-150
        for (index, coefficient) in enumerate(SolpropJulia._DF32_RECIPROCAL_FACTORIAL)
            exact = inv(BigFloat(factorial(big(index - 1))))
            @test abs(df32f_big(coefficient) - exact) <= 2unit^2 * exact
        end
        # These displayed inequalities are fixed before any function comparison.
        # They check conservative proof constants, not sampled fitted tolerances.
        radius, delta = BigFloat(0.5), 27unit^2
        polynomial = exp(2radius) * (
            2exp((1 + delta) * radius) + 25expm1((1 + delta) * radius) / (1 + delta)
        )
        tail = exp(2radius) * radius^19 / factorial(big(19))
        @test polynomial + tail / unit^2 < 64
        @test 23exp((2 + delta) * radius) < 64
        error = 64unit^2 + 64tiny
        for unused in 1:6
            error = (1 + 15unit^2) * (1 + error)^2 - 1 + big(2)^28 * tiny
        end
        error = (1 + error) * exp(32unit^2 + 192tiny) - 1
        exponential_bound = 8192unit^2 + big(2)^40 * tiny
        @test error < exponential_bound
        @test exp(radius) * radius^19 / factorial(big(19)) < BigFloat("2.586e-23")
        @test radius^24 / (factorial(big(25)) * (1 - radius / 26)) <
            BigFloat("3.919e-33")
        addition = 3unit^2 / (1 - 4unit)
        narrow_error = exponential_bound + unit * (1 + exponential_bound) + sqrt(big(2)) *
            tiny
        @test narrow_error < BigFloat("1.001") * unit
        norm_error = addition * (1 + narrow_error)^2 + 2(1 + addition) * tiny
        deviation = 2narrow_error + narrow_error^2
        correction_error = norm_error / 2 + unit * (deviation + norm_error) / 2 + tiny
        @test correction_error < 4unit^2 + 4tiny
        corrected_error = narrow_error + 3narrow_error^2 / 2 + narrow_error^3 / 2 +
            (1 + narrow_error) * correction_error
        phasor_error = corrected_error + unit * (1 + corrected_error) + sqrt(big(2)) * tiny
        @test phasor_error < 17unit / 8
        before_round_norm = 3deviation^2 / 4 + deviation^3 / 4 +
            (1 + narrow_error)^2 * (
            2(1 + deviation / 2) * correction_error + correction_error^2
        )
        norm_defect = before_round_norm + (2unit + unit^2) * (1 + before_round_norm) +
            2sqrt(big(2)) * tiny * sqrt(1 + before_round_norm) * (1 + unit) + 2tiny^2
        @test norm_defect < 3unit
        magnitude_error = exponential_bound + unit * (1 + exponential_bound) + tiny *
            exp(BigFloat(16))
        @test magnitude_error < 9unit / 8
        pair_error = (1 + phasor_error) * (1 + magnitude_error) * (1 + unit) - 1 +
            sqrt(big(2)) * tiny * exp(BigFloat(16))
        @test pair_error < 9unit / 2
        @test (1 + pair_error) * exp(unit / 8) - 1 < 5unit
    end
    # The scalar graph constants and reduction bound use rational arithmetic;
    # they do not rely on rounding the displayed polynomials in Float64.
    unit = big(1) // big(2)^24
    tiny = big(1) // big(2)^150
    addition = 3unit^2 / (1 - 4unit)
    @test 2unit^2 / (1 - unit) < 3unit^2
    @test 2 + 2unit < 3
    @test (6unit^2 + 8unit^3 + 7unit^4 + 4unit^5 + unit^6) / (1 - unit)^2 < 7unit^2
    @test 4 + 8unit + 4unit^2 + unit^3 < 5
    @test (7unit^2 + 11unit^3 + 10unit^4 + 5unit^5 + unit^6) / (1 - unit) < 8unit^2
    for divisor in 1:32
        first = unit * divisor * (1 + unit) + unit
        second = unit * (divisor + 1) + unit^2 * divisor * (1 + unit) + unit^2
        third = unit * (divisor + 1 + first + second)
        fourth = 1 + unit * (divisor + 1 + first + second) * (1 + unit) / divisor
        @test (1 + first + second + third) / divisor + fourth < 4
    end
    sqrt2_upper = big(99) // 70
    @test sqrt2_upper^2 > 2
    @test sqrt2_upper * (7unit^2 + addition * (1 + 7unit^2)) < 15unit^2
    @test 10sqrt2_upper * (1 + addition) < 15
    @test 7unit^2 + addition * (1 + 7unit^2) < 11unit^2
    @test 10(1 + addition) < 11
    @test (3unit + unit^2) / (1 - unit) * 2^20 / (big(25) // 4) + tiny < big(1) // 32
    first = big(27) // 8 + big(1) // 16 + big(1) // big(2)^29 + big(1) // big(2)^56
    second = big(27) // 8 + big(1) // big(2)^29 + big(1) // big(2)^56
    third = big(27) // 8 + big(1) // big(2)^56
    reduction = big(1) // big(2)^56 + addition * (
        (1 + addition)^2 * first + (1 + addition) * second + third
    )
    @test reduction < 32unit^2
    @test 7unit^2 * (1 + unit^2)^2 + 2unit^2 + unit^4 < 10unit^2
end

@testset "Three-word reduction and selector boundaries" begin
    setprecision(BigFloat, 256) do
        unit = big(2.0)^-24
        angles = SolpropJulia.DF32[]
        for turn in (-166886, -16384, -1, 0, 1, 16384, 166885)
            for offset in (-0.5, 0.0, 0.5)
                point = Float64((BigFloat(turn) + offset) * 2BigFloat(pi))
                for near in (prevfloat(point), point, nextfloat(point))
                    if abs(near) <= 2^20 &&
                            (iszero(near) || abs(near) >= floatmin(Float32))
                        push!(angles, SolpropJulia.prepare_df32(near))
                    end
                end
            end
        end
        append!(
            angles, SolpropJulia.prepare_df32.(
                [
                    -2.0^20, -2.0^-100, -0.0, 0.0, 2.0^-100, 2.0^20,
                ]
            )
        )
        for theta in angles
            exact = df32f_big(theta)
            selected = theta.hi * SolpropJulia._DF32_INVERSE_TURN
            turns = round(Int32, selected, RoundNearest)
            nearest = round(BigInt, exact / (2BigFloat(pi)), RoundNearest)
            @test abs(BigFloat(selected) - exact / (2BigFloat(pi))) < big(1) / 32
            @test abs(turns - nearest) <= 1
            @test abs(turns) < 2^18
            remainder = SolpropJulia._df32_reduce_phase(theta)
            reference = exact - turns * 2BigFloat(pi)
            @test abs(df32f_big(remainder) - reference) <= 32unit^2
            @test abs(df32f_big(remainder)) < big(7) / 2
            @test remainder.hi == remainder.hi + remainder.lo
            if iszero(turns)
                @test isequal(remainder.hi, theta.hi)
                @test isequal(remainder.lo, theta.lo)
            end
        end
    end
end

function df32f_arguments()
    reals = (-16.0, -3.0, -0.5, -2.0^-100, 0.0, 2.0^-100, 0.5, 3.0, 16.0)
    phases = (
        -2.0^20, -1000.25, -2pi, -3.01, -3.0, -0.5, -2.0^-100, -0.0,
        0.0, 2.0^-100, 0.5, 3.0, 3.01, 2pi, 1000.25, 2.0^20,
    )
    return [df32f_argument(first, second) for first in reals for second in phases]
end

@testset "Wide exponential and corrected propagators" begin
    setprecision(BigFloat, 256) do
        unit, tiny = big(2.0)^-24, big(2.0)^-150
        relative = 8192unit^2 + big(2)^40 * tiny
        for argument in df32f_arguments()
            @test SolpropJulia.check_df32_function_argument(argument) === nothing
            exact = df32f_big(argument)
            reference = exp(exact)
            wide = SolpropJulia._df32_exp(argument)
            df32f_assert_bound(wide, reference, relative * abs(reference))
            sine, cosine = SolpropJulia._df32_sincos(imag(argument))
            phasor = Complex(BigFloat(cosine), BigFloat(sine))
            @test abs(phasor - exp(Complex(big(0), imag(exact)))) < 17unit / 8
            @test abs(abs2(phasor) - 1) < 3unit
            forward, backward = SolpropJulia._df32_exp_pair(argument)
            @test abs(df32f_big(forward) - reference) < 9unit / 2 * abs(reference)
            @test abs(df32f_big(backward) - inv(reference)) <
                9unit / 2 * abs(inv(reference))
            # Independent worst allowed assembly perturbation, with opposite
            # component signs, exercises the separate u/8 input-error policy.
            shifted = exact + Complex(unit / 16, -unit / 16)
            @test abs(df32f_big(forward) - exp(shifted)) < 5unit * abs(exp(shifted))
            @test abs(df32f_big(backward) - exp(-shifted)) < 5unit * abs(exp(-shifted))
        end
        for sign in (-0.0f0, 0.0f0)
            sine, cosine = SolpropJulia._df32_sincos(SolpropJulia.prepare_df32(sign))
            @test isequal(sine, sign)
            @test cosine == 1.0f0
            forward, backward = SolpropJulia._df32_exp_pair(df32f_argument(sign, sign))
            @test real(forward) == real(backward) == 1.0f0
            @test isequal(imag(forward), sign)
            @test isequal(imag(backward), -sign)
        end
    end
end

@testset "Phi absolute bounds, zeros and overlapping join" begin
    setprecision(BigFloat, 256) do
        unit, tiny = big(2.0)^-24, big(2.0)^-150
        arguments = df32f_arguments()
        for direction in (1.0 + 0.0im, -1.0 + 0.0im, 0.0 + 1.0im, 0.6 + 0.8im)
            for radius in (2.99, prevfloat(3.0), 3.0, nextfloat(3.0), 3.01)
                point = direction * radius
                push!(arguments, df32f_argument(real(point), imag(point)))
            end
        end
        for argument in arguments
            exact = df32f_big(argument)
            references = df32f_phi_reference(exact, 8)
            for highest in 0:8
                result = SolpropJulia._df32_phi_values(argument, Val(highest))
                if highest == 0
                    bounds = [(8192unit^2 + big(2)^40 * tiny) * abs(references[1])]
                elseif SolpropJulia._df32_complex_abs2(argument).hi > 9.0f0
                    @test abs(exact) >= BigFloat("2.99")
                    bounds = df32f_phi_large_bounds(exact, highest)
                else
                    @test abs(exact) <= BigFloat("3.01")
                    squarings = df32f_phi_small_scalings(argument)
                    @test 0 <= squarings <= 3
                    bounds = df32f_phi_small_bounds(exact, highest, squarings)
                end
                @test length(result) == highest + 1
                for index in eachindex(result)
                    df32f_assert_bound(result[index], references[index], bounds[index])
                end
            end
            if BigFloat("2.99") <= abs(exact) <= BigFloat("3.01")
                small = SolpropJulia._df32_phi_small(argument, Val(8))
                large = SolpropJulia._df32_phi_large(argument, Val(8))
                small_bounds = df32f_phi_small_bounds(
                    exact, 8, df32f_phi_small_scalings(argument),
                )
                large_bounds = df32f_phi_large_bounds(exact, 8)
                for index in 1:9
                    df32f_assert_bound(small[index], references[index], small_bounds[index])
                    df32f_assert_bound(large[index], references[index], large_bounds[index])
                    @test abs(df32f_big(small[index]) - df32f_big(large[index])) <=
                        small_bounds[index] + large_bounds[index]
                end
            end
        end
        for highest in 0:8
            result = SolpropJulia._df32_phi_values(df32f_argument(-0.0, 0.0), Val(highest))
            for index in eachindex(result)
                coefficient = SolpropJulia._DF32_RECIPROCAL_FACTORIAL[index]
                @test isequal(real(result[index]).hi, coefficient.hi)
                @test isequal(real(result[index]).lo, coefficient.lo)
                @test iszero(imag(result[index]))
            end
        end
        # Phi is not periodic. The large branch must retain the original inverse
        # argument even when the exponential has reduced the phase by many turns.
        for turns in (1, 32, 10000)
            requested = Complex(big(0), 2BigFloat(pi) * turns)
            argument = df32f_argument(0.0, Float64(imag(requested)))
            exact = df32f_big(argument)
            result = SolpropJulia._df32_phi_values(argument, Val(8))
            bounds = df32f_phi_large_bounds(exact, 8)
            perturbation = abs(exact - requested)
            @test perturbation < unit / 8
            references = df32f_phi_reference(requested, 8)
            for order in 1:8
                input_bound = perturbation * exp(perturbation) / factorial(big(order + 1))
                df32f_assert_bound(
                    result[order + 1], references[order + 1],
                    bounds[order + 1] + input_bound
                )
            end
            @test abs(df32f_big(result[2])) <= bounds[2] + perturbation / 2
        end
    end
end

@testset "Function arithmetic with gradual underflow" begin
    setprecision(BigFloat, 256) do
        unit, tiny = big(2.0)^-24, big(2.0)^-150
        raw = SolpropJulia._df32_raw
        samples = SolpropJulia.DF32[]
        for exponent in (-149, -148, -130, -126, -110, -80, -63, -20, 0, 10, 20)
            high = ldexp(1.0f0, exponent)
            for sign in (-1.0f0, 1.0f0)
                low = ldexp(1.0f0, exponent - 26)
                push!(samples, raw(sign * high, sign * low))
                push!(samples, raw(sign * high, -sign * low))
            end
        end
        for first in samples, second in samples
            exact = df32f_big(first) * df32f_big(second)
            result = first * second
            @test isfinite(result)
            @test result.hi == result.hi + result.lo
            @test abs(df32f_big(result) - exact) <= 7unit^2 * abs(exact) + 5tiny
            scalar_exact = df32f_big(first) * BigFloat(second.hi)
            scalar_result = first * second.hi
            @test abs(df32f_big(scalar_result) - scalar_exact) <=
                3unit^2 * abs(scalar_exact) + 3tiny
        end
        for value in samples
            for divisor in 1:32
                exact = df32f_big(value) / divisor
                result = value / Float32(divisor)
                @test result.hi == result.hi + result.lo
                @test abs(df32f_big(result) - exact) <= 8unit^2 * abs(exact) + 4tiny
            end
            for high in (8.0f0, nextfloat(8.0f0), 32.0f0, 2.0f0^20, 2.0f0^42)
                divisor = raw(high, -high * 2.0f0^-26)
                exact = df32f_big(value) / df32f_big(divisor)
                result = value / divisor
                absolute_factor = 6abs(df32f_big(value)) *
                    (1 + inv(abs(df32f_big(divisor)))) + 6
                allowance = absolute_factor * tiny
                @test result.hi == result.hi + result.lo
                @test abs(df32f_big(result) - exact) <= 16unit^2 * abs(exact) + allowance
            end
        end
        # A normal represented imaginary input produces a genuinely subnormal
        # scaled component. The absolute allowance, not exact EFT arithmetic,
        # covers the lost terms in the function evaluation.
        argument = df32f_argument(16.0f0, floatmin(Float32))
        scaled = SolpropJulia._df32_complex_scale(argument, 1.0f0 / 64.0f0)
        @test 0 < imag(scaled).hi < floatmin(Float32)
        wide = SolpropJulia._df32_exp(argument)
        reference = exp(df32f_big(argument))
        bound = (8192unit^2 + big(2)^40 * tiny) * abs(reference)
        df32f_assert_bound(wide, reference, bound)
    end
end

function df32f_batch!(reduced, wide, phasors, pairs, phi, arguments, execution)
    SolpropJulia.each!(execution, length(arguments), 1) do row, unused
        argument = arguments[row]
        reduced[row] = SolpropJulia._df32_reduce_phase(imag(argument))
        wide[row] = SolpropJulia._df32_exp(argument)
        phasors[row, 1], phasors[row, 2] = SolpropJulia._df32_sincos(imag(argument))
        pairs[row, 1], pairs[row, 2] = SolpropJulia._df32_exp_pair(argument)
        terms = SolpropJulia._df32_phi_values(argument, Val(8))
        for index in 1:9
            phi[row, index] = terms[index]
        end
    end
    return nothing
end

@testset "Prepared DF32 function launch allocations" begin
    arguments = df32f_arguments()
    reduced = Vector{SolpropJulia.DF32}(undef, length(arguments))
    wide = similar(arguments)
    phasors = Matrix{Float32}(undef, length(arguments), 2)
    pairs = Matrix{ComplexF32}(undef, length(arguments), 2)
    phi = Matrix{Complex{SolpropJulia.DF32}}(undef, length(arguments), 9)
    for tasks in unique((1, Threads.nthreads(:default)))
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        df32f_batch!(reduced, wide, phasors, pairs, phi, arguments, execution)
        df32f_batch!(reduced, wide, phasors, pairs, phi, arguments, execution)
        @test 0 == @allocated df32f_batch!(
            reduced, wide, phasors, pairs, phi, arguments, execution,
        )
        @test all(isfinite, reduced)
        @test all(value -> isfinite(real(value)) && isfinite(imag(value)), wide)
        @test all(isfinite, phasors)
        @test all(isfinite, pairs)
        @test all(value -> isfinite(real(value)) && isfinite(imag(value)), phi)
    end
end
