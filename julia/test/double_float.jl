# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

df32_exact(word::Float32) = Rational{BigInt}(word)
df32_exact(value::SolpropJulia.DF32) = df32_exact(value.hi) + df32_exact(value.lo)

function df32_assert_bound(value, reference, relative_bound)
    @test relative_bound > 0
    @test isfinite(value)
    @test value.hi == value.hi + value.lo
    # All constants are exact rationals fixed by the primary theorem, including
    # cancellation. A zero reference therefore requires an exact real zero.
    @test abs(df32_exact(value) - reference) <= relative_bound * abs(reference)
    return nothing
end

function df32_samples()
    result = SolpropJulia.DF32[]
    # Exact word patterns and integer dyadic low words, with both signs and five
    # binades. All operation intermediates are far from either exponent limit.
    # The strict half-ulp bound excludes ties here; explicit tests cover them.
    patterns = UInt32[
        0x00000000, 0x00000001, 0x00000002, 0x00124925, 0x002aaaab, 0x00400000, 0x007fffff,
    ]
    low_numerators = (-8388607, -4194303, -1, 0, 1, 4194303, 8388607)
    for exponent in (-12, -6, 0, 6, 12)
        for pattern in patterns, polarity in (-1.0f0, 1.0f0)
            high = polarity * ldexp(reinterpret(Float32, 0x3f800000 | pattern), exponent)
            for numerator in low_numerators
                gap = numerator < 0 ? high - prevfloat(high) : nextfloat(high) - high
                low = (Float32(numerator) / 8388608.0f0) * (0.5f0 * gap)
                push!(result, SolpropJulia.prepare_df32(high, low))
            end
        end
    end
    return result
end

function df32_primitive_batch!(output, first, second, factors, execution)
    SolpropJulia.each!(execution, length(first), 1) do row, unused
        x, y, factor = first[row], second[row], factors[row]
        output[row, 1] = SolpropJulia._df32_plus_fp(x, factor)
        output[row, 2] = SolpropJulia._df32_plus_dw(x, y)
        output[row, 3] = SolpropJulia._df32_times_fp(x, factor)
        output[row, 4] = SolpropJulia._df32_times_dw(x, y)
        output[row, 5] = SolpropJulia._df32_div_fp(x, factor)
        output[row, 6] = SolpropJulia._df32_div_dw(x, y)
    end
    return nothing
end

@testset "Checked DF32 representation" begin
    DF = SolpropJulia.DF32
    prepare = SolpropJulia.prepare_df32
    unit = big(1) // big(2)^24
    @test isbitstype(DF)
    @test sizeof(DF) == 2sizeof(Float32)
    @test DF <: Real
    @test !(DF <: AbstractFloat)
    @test_throws MethodError DF(1.0f0, 0.0f0)
    @test_throws MethodError prepare(1)

    words = Float32[
        -0.0, 0.0, -1, 1, prevfloat(1.0f0), nextfloat(1.0f0),
        ldexp(1.0f0, -90), ldexp(-1.0f0, 90), floatmin(Float32), floatmax(Float32),
    ]
    for word in words
        value = prepare(word)
        @test isequal(value.hi, word)
        @test isequal(value.lo, 0.0f0)
        @test isequal(Float32(value), word)
        @test df32_exact(value) == df32_exact(word)
        @test isfinite(value)
    end
    for high in (-0.0f0, 0.0f0), low in (-0.0f0, 0.0f0)
        value = prepare(high, low)
        @test isequal(value.hi, high)
        @test isequal(value.lo, low)
        @test isequal(Float32(value), high)
        @test iszero(value)
        @test isequal((-value).hi, -high)
        @test isequal((-value).lo, -low)
    end
    @test iszero(zero(DF))
    @test iszero(zero(one(DF)))
    @test df32_exact(one(DF)) == 1
    @test df32_exact(one(zero(DF))) == 1
    @test !isfinite(SolpropJulia._df32_raw(1.0f0, NaN32))
    @test !iszero(SolpropJulia._df32_raw(0.0f0, 1.0f0))

    host_values = Float64[
        -0.0, 0.0, -1.0, 1.0, pi, -pi, 1 / 3, -1 / 7,
        ldexp(1.1, -60), ldexp(-1.1, 60),
        Float64(floatmin(Float32)), Float64(floatmax(Float32)),
        1 + 2.0^-24 - 2.0^-49, 1 + 2.0^-24, 1 + 2.0^-24 + 2.0^-49,
    ]
    for point in host_values
        value = prepare(point)
        reference = Rational{BigInt}(point)
        @test value.hi == value.hi + value.lo
        @test abs(df32_exact(value) - reference) <= unit^2 * abs(reference)
        @test isfinite(value)
        if iszero(point)
            @test signbit(value.hi) == signbit(point)
        end
    end
    # Rounding the low word of an ordinary host split can change the even high
    # word. Renormalization preserves the rounded pair's exact sum, not the first
    # split's high bits. The explicit-pair API refuses the noncanonical form.
    tie_point = 1 + 2.0^-24 + 2.0^-49
    tie_high = Float32(tie_point)
    tie_low = Float32(tie_point - Float64(tie_high))
    @test tie_high != tie_high + tie_low
    @test_throws ArgumentError prepare(tie_high, tie_low)
    canonical = prepare(tie_point)
    @test canonical.hi == 1.0f0
    @test df32_exact(canonical) == df32_exact(tie_high) + df32_exact(tie_low)

    for point in (NaN32, Inf32, -Inf32, NaN, Inf, -Inf)
        @test_throws ArgumentError prepare(point)
    end
    for point in (nextfloat(0.0f0), prevfloat(floatmin(Float32)))
        @test_throws ArgumentError prepare(point)
        @test_throws ArgumentError prepare(-point)
        @test_throws ArgumentError prepare(1.0f0, point)
    end
    for point in (
            nextfloat(0.0), Float64(floatmin(Float32)) / 2,
            nextfloat(Float64(floatmin(Float32))), 2.0^-80 + 2.0^-130,
            Float64(floatmax(Float32)) * 2,
        )
        @test_throws ArgumentError prepare(point)
        @test_throws ArgumentError prepare(-point)
    end
    # Just below binary32's overflow midpoint, the first high-word rounding is
    # finite but rounding the residual can reach the midpoint. Refuse that too.
    overflow_midpoint = Float64(floatmax(Float32)) + 2.0^103
    @test isfinite(Float32(prevfloat(overflow_midpoint)))
    @test_throws ArgumentError prepare(prevfloat(overflow_midpoint))
    @test_throws ArgumentError prepare(overflow_midpoint)
    for (high, low) in (
            (0.0f0, 1.0f0), (1.0f0, 1.0f0), (1.0f0, -2.0f0^-24),
            (nextfloat(1.0f0), 2.0f0^-24), (nextfloat(1.0f0), -2.0f0^-24),
            (1.0f0, NaN32), (Inf32, 0.0f0),
        )
        @test_throws ArgumentError prepare(high, low)
    end
    @test df32_exact(prepare(1.0f0, 2.0f0^-24)) == 1 + unit

    # Julia's generic Real promotion must refuse, rather than silently widening
    # a protected operation to Float64. No promotion or conversion rule is added.
    value = prepare(1.25)
    for operation in (+, -, *, /), unsupported in (1.0, 1)
        @test_throws ErrorException operation(value, unsupported)
        @test_throws ErrorException operation(unsupported, value)
    end
    @test_throws MethodError Float64(value)
    @test_throws MethodError eps(DF)
end

@testset "Error-free transforms and their finite-range hypotheses" begin
    samples = df32_samples()
    for index in eachindex(samples)
        first = samples[index].hi
        second = samples[mod1(71index + 19, length(samples))].hi
        high, low = SolpropJulia._df32_two_sum(first, second)
        @test df32_exact(high) + df32_exact(low) == df32_exact(first) + df32_exact(second)
        @test high == high + low
        leading, trailing = abs(first) >= abs(second) ? (first, second) : (second, first)
        fast_high, fast_low = SolpropJulia._df32_fast_two_sum(leading, trailing)
        @test df32_exact(fast_high) + df32_exact(fast_low) ==
            df32_exact(first) + df32_exact(second)
        product_high, product_low = SolpropJulia._df32_two_prod(first, second)
        @test df32_exact(product_high) + df32_exact(product_low) ==
            df32_exact(first) * df32_exact(second)
        @test product_high == product_high + product_low
    end
    for trailing in (2.0f0^-24, -2.0f0^-25)
        high, low = SolpropJulia._df32_two_sum(1.0f0, trailing)
        @test df32_exact(high) + df32_exact(low) == 1 + df32_exact(trailing)
        @test high == 1.0f0
    end
    # Fast2Sum's order condition matters. This reversed pair loses the unit even
    # though TwoSum remains exact; a generic unchecked swap is not permitted.
    high, low = SolpropJulia._df32_fast_two_sum(1.0f0, 2.0f0^25)
    @test df32_exact(high) + df32_exact(low) != 1 + big(2)^25
    high, low = SolpropJulia._df32_two_sum(1.0f0, 2.0f0^25)
    @test df32_exact(high) + df32_exact(low) == 1 + big(2)^25

    # J17 p3's finite-exponent caveat: the rounded product is normal but its
    # residual 2^-172 is below binary32's least subnormal. Normal inputs/prepared
    # representations alone do not establish an exact TwoProd or theorem domain.
    tiny = ldexp(nextfloat(1.0f0), -63)
    @test isfinite(SolpropJulia.prepare_df32(tiny))
    high, low = SolpropJulia._df32_two_prod(tiny, tiny)
    @test high >= floatmin(Float32)
    @test iszero(low)
    @test df32_exact(tiny)^2 - df32_exact(high) == big(1) // big(2)^172
    @test df32_exact(high) + df32_exact(low) != df32_exact(tiny)^2
    high, low = SolpropJulia._df32_two_prod(floatmax(Float32), 2.0f0)
    @test isinf(high)
    @test !isfinite(low)
    # The reciprocal in DWDivDW3 can be subnormal despite a normal divisor.
    # Later whole-expression admission, not preparation, must rule this out.
    @test isfinite(SolpropJulia.prepare_df32(floatmax(Float32)))
    @test 0 < 1.0f0 / floatmax(Float32) < floatmin(Float32)
end

@testset "Audited scalar double-word bounds" begin
    samples = df32_samples()
    unit = big(1) // big(2)^24
    addition_bound = 3unit^2 / (1 - 4unit)
    multiplication_bound = (4unit^2 + unit^3 / 2) / (1 + unit)^2
    division_bound = (big(49) // 5) * unit^2
    for index in eachindex(samples)
        x = samples[index]
        y = samples[mod1(71index + 19, length(samples))]
        factor = samples[mod1(37index + 11, length(samples))].hi
        exact_x, exact_y, exact_factor = df32_exact(x), df32_exact(y), df32_exact(factor)
        df32_assert_bound(x + factor, exact_x + exact_factor, 2unit^2)
        df32_assert_bound(x + y, exact_x + exact_y, addition_bound)
        df32_assert_bound(x * factor, exact_x * exact_factor, 2unit^2)
        df32_assert_bound(x * y, exact_x * exact_y, multiplication_bound)
        df32_assert_bound(x / factor, exact_x / exact_factor, 3unit^2)
        df32_assert_bound(x / y, exact_x / exact_y, division_bound)
        df32_assert_bound(x - y, exact_x - exact_y, addition_bound)
        df32_assert_bound(factor - x, exact_factor - exact_x, 2unit^2)
        df32_assert_bound(factor / y, exact_factor / exact_y, division_bound)
        @test df32_exact(factor + x) == df32_exact(x + factor)
        @test df32_exact(factor * x) == df32_exact(x * factor)
    end

    # Independent 256-bit MPFR division witnesses additionally exercise the final
    # reciprocal correction. Reference error is <2^-255 relative, far below the
    # fixed paper bound; exact rational checks above do not depend on MPFR.
    setprecision(BigFloat, 256) do
        for index in 1:17:length(samples)
            x = samples[index]
            y = samples[mod1(71index + 19, length(samples))]
            reference = (BigFloat(x.hi) + BigFloat(x.lo)) /
                (BigFloat(y.hi) + BigFloat(y.lo))
            result = x / y
            error = abs(BigFloat(result.hi) + BigFloat(result.lo) - reference)
            @test error <= BigFloat(division_bound + big(1) // big(2)^254) * abs(reference)
        end
    end

    # J17 p6: the sloppy sum is zero although the exact sum is u^2. The accurate
    # algorithm preserves this cancellation without a relative-tolerance escape.
    u = 2.0f0^-24
    x = SolpropJulia.prepare_df32(1.0f0 + 8u, -u)
    y = SolpropJulia.prepare_df32(-1.0f0 - 6u, -u + u^2)
    @test df32_exact(x + y) == unit^2
    high, residual = SolpropJulia._df32_two_sum(x.hi, y.hi)
    sloppy_correction = residual + (x.lo + y.lo)
    sloppy_high, sloppy_low = SolpropJulia._df32_fast_two_sum(high, sloppy_correction)
    @test iszero(sloppy_high) && iszero(sloppy_low)

    # M22 Property 2.1 is asymptotically sharp at 3u^2. It explicitly rules out
    # borrowing DWPlusFP's smaller 2u^2 bound for the double-word sum.
    x = SolpropJulia.prepare_df32(1.0f0, u - u^2)
    y = SolpropJulia.prepare_df32(-0.5f0 + u / 2, -u^2 / 2 + u^3)
    exact_sum = df32_exact(x) + df32_exact(y)
    sum_error = abs(df32_exact(x + y) - exact_sum)
    @test sum_error > 2unit^2 * abs(exact_sum)
    df32_assert_bound(x + y, exact_sum, addition_bound)

    # J17 p15 supplies this binary32 witness for DWTimesDW2, whose error exceeds
    # 4u^2. DWTimesDW3's retained low-by-low term and two-FMA order meet the
    # improved M22 bound on the same normalized operands.
    x = SolpropJulia.prepare_df32(8404039.0f0, -ldexp(8284843.0f0, -24))
    y = SolpropJulia.prepare_df32(8409182.0f0, -ldexp(4193899.0f0, -23))
    exact_product = df32_exact(x) * df32_exact(y)
    df32_assert_bound(x * y, exact_product, multiplication_bound)
    high, residual = SolpropJulia._df32_two_prod(x.hi, y.hi)
    first_cross = x.hi * y.lo
    second_cross = fma(x.lo, y.hi, first_cross)
    older_high, older_low = SolpropJulia._df32_fast_two_sum(high, residual + second_cross)
    @test abs(df32_exact(older_high) + df32_exact(older_low) - exact_product) >
        4unit^2 * abs(exact_product)

    # Structural zeros and identities are exact in the represented real value.
    # Only preparation, narrowing and unary negation promise particular zero bits.
    for x in samples[1:13:end]
        @test iszero(x - x)
        @test df32_exact(x + zero(x)) == df32_exact(x)
        @test df32_exact(x * one(x)) == df32_exact(x)
        @test df32_exact(x * 1.0f0) == df32_exact(x)
        @test df32_exact(x / 1.0f0) == df32_exact(x)
        @test iszero(x * zero(x))
        @test iszero(zero(x) / x)
    end
end

@testset "Allocation-free DF32 kernel execution" begin
    first = df32_samples()
    second = [first[mod1(71index + 19, length(first))] for index in eachindex(first)]
    factors = [value.hi for value in second]
    output = fill(zero(SolpropJulia.DF32), length(first), 6)
    for tasks in unique((1, Threads.nthreads(:default)))
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        df32_primitive_batch!(output, first, second, factors, execution)
        allocated = @allocated df32_primitive_batch!(
            output, first, second, factors, execution,
        )
        @test allocated == 0
        for index in eachindex(first)
            expected = (
                first[index] + factors[index], first[index] + second[index],
                first[index] * factors[index], first[index] * second[index],
                first[index] / factors[index], first[index] / second[index],
            )
            for column in 1:6
                @test df32_exact(output[index, column]) == df32_exact(expected[column])
            end
        end
    end
end
