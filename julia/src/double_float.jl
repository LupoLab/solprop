# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public DF32, prepare_df32

"""
    DF32 <: Real

Store a normalized finite unevaluated sum of two binary32 words.

The immutable `hi::Float32` and `lo::Float32` retain the input's units. Normalization
means `hi == RN(hi + lo)` for the exact real sum, with round-to-nearest ties-to-even.
Use [`prepare_df32`](@ref) at host preparation; no unchecked two-argument constructor
is provided. This is a restricted scalar building block, not an `AbstractFloat`
implementation or admission of double-float propagation.

Arithmetic implements the six algorithms cited below with binary32 operations and
explicit FMA only. Its conditional bounds require normalized finite operands, correct
RN-even arithmetic/FMA/division, no overflow, and no loss of an intermediate to
underflow. In particular, a TwoProd residual must remain representable. A nonzero
divisor is required. Whole-expression range admission must establish these conditions;
successful preparation alone does not establish closure of subsequent arithmetic.

Only DF32/DF32 and DF32/Float32 addition, subtraction, multiplication and division
are provided. Mixed integer/binary64 promotion, complex arithmetic, `eps` and
transcendental functions are outside this interface. `Float32(x)` returns the already
correctly rounded high word, preserving its zero sign. Unary negation changes both
word signs; arithmetic zeros otherwise follow the explicit operations' RN behavior.

# References
- Joldes, Muller and Popescu (2017), Algorithms 4, 6, 9, 12, 15 and 18,
  https://doi.org/10.1145/3121432.
- Muller and Rideau (2022), Definition 1 and Theorem 2.8,
  https://doi.org/10.1145/3484514.
"""
struct DF32 <: Real
    hi::Float32
    lo::Float32

    # Only proved internal sequences may bypass preparation. A distinct third
    # argument suppresses Julia's automatically generated unchecked pair constructor.
    DF32(hi::Float32, lo::Float32, ::Val{:raw}) = new(hi, lo)
end

@inline _df32_raw(hi::Float32, lo::Float32) = DF32(hi, lo, Val(:raw))

# J17 Algorithms 1--3. Every displayed operation is separately RN-rounded except
# the explicit FMA. Do not contract/reassociate these error-free transformations.
# Fast2Sum requires the leading operand's exponent to be at least the second's
# (or a zero operand); |a| >= |b| is sufficient, but not necessary.
@inline function _df32_fast_two_sum(a::Float32, b::Float32)
    high = a + b
    recovered = high - a
    low = b - recovered
    return high, low
end

@inline function _df32_two_sum(a::Float32, b::Float32)
    high = a + b
    a_recovered = high - b
    b_recovered = high - a_recovered
    a_error = a - a_recovered
    b_error = b - b_recovered
    low = a_error + b_error
    return high, low
end

@inline function _df32_two_prod(a::Float32, b::Float32)
    high = a * b
    low = fma(a, b, -high)
    return high, low
end

function _df32_check_word(value::Float32)
    isfinite(value) || throw(ArgumentError("DF32 preparation requires finite words"))
    if !iszero(value) && abs(value) < floatmin(Float32)
        throw(ArgumentError("DF32 preparation refuses nonzero subnormal words"))
    end
    return nothing
end

"""
    prepare_df32(value::Union{Float32,Float64})
    prepare_df32(hi::Float32, lo::Float32)

Prepare checked finite DF32 storage on the host.

Preserve the scalar's units. A binary64 input is split into a rounded high word and
rounded residual, then canonically renormalized without changing their real sum.
The split's absolute error is at most `2^-48 * abs(value)` under the admitted normal
range. A binary32 input is retained exactly with a positive-zero low word. Scalar
signed zero preserves its high-word sign. An explicit pair is validated without
changing either word, including their zero signs.

Each nonzero input/word and the binary64 split residual must be in binary32's normal
range; residuals that would round to zero are refused too. This deliberately excludes
subnormal representations. These checks establish storage and normalization only:
the later expression must independently establish every arithmetic intermediate's
finite-exponent hypotheses. No hardware rounding or device admission is performed.

# Returns
A finite normalized `DF32`, without generic numeric promotion.

# Throws
- `ArgumentError`: For nonfinite input, overflow, nonzero input/residual underflow,
  or an explicitly supplied pair whose high word does not round its exact sum.
"""
function prepare_df32(hi::Float32, lo::Float32)
    _df32_check_word(hi)
    _df32_check_word(lo)
    hi == hi + lo || throw(ArgumentError("DF32 preparation requires a normalized pair"))
    return _df32_raw(hi, lo)
end

function prepare_df32(value::Float32)
    return prepare_df32(value, 0.0f0)
end

function prepare_df32(value::Float64)
    isfinite(value) || throw(ArgumentError("DF32 preparation requires a finite scalar"))
    high = Float32(value)
    _df32_check_word(high)
    if !iszero(value) && abs(value) < Float64(floatmin(Float32))
        throw(ArgumentError("DF32 preparation refuses nonzero input underflow"))
    end
    iszero(value) && return _df32_raw(high, 0.0f0)
    # Sterbenz makes this host subtraction exact: high and value differ only by
    # one binary32 rounding. Reject a tiny residual before it can disappear.
    residual = value - Float64(high)
    if !iszero(residual) && abs(residual) < Float64(floatmin(Float32))
        throw(ArgumentError("DF32 preparation refuses nonzero residual underflow"))
    end
    low = Float32(residual)
    # Rounding the residual can put the represented sum exactly on a midpoint.
    # Fast2Sum selects the even high word; |high| >= |low| holds for this split.
    normalized_high, normalized_low = _df32_fast_two_sum(high, low)
    return prepare_df32(normalized_high, normalized_low)
end

# In the following source bounds p=24 and u=2^-24. Each bound is conditional on
# the type's documented intermediate-range assumptions, not just finite operands.
# J17 Algorithm 4 / Theorem 2.2: |relative error| <= 2u^2, p >= 3.
@inline function _df32_plus_fp(x::DF32, y::Float32)
    high, residual = _df32_two_sum(x.hi, y)
    low = x.lo + residual
    result_high, result_low = _df32_fast_two_sum(high, low)
    return _df32_raw(result_high, result_low)
end

# J17 Algorithm 6 / Theorem 3.1: <= 3u^2/(1-4u), p >= 3. TwoSum on
# BOTH word pairs preserves cancellation; the sloppy high-only form does not.
@inline function _df32_plus_dw(x::DF32, y::DF32)
    high, high_residual = _df32_two_sum(x.hi, y.hi)
    low, low_residual = _df32_two_sum(x.lo, y.lo)
    correction = high_residual + low
    middle_high, middle_low = _df32_fast_two_sum(high, correction)
    remainder = low_residual + middle_low
    result_high, result_low = _df32_fast_two_sum(middle_high, remainder)
    return _df32_raw(result_high, result_low)
end

# J17 Algorithm 9 / Theorem 4.3: <= 2u^2, p >= 3. The cross term and
# high-product residual are combined in ONE correctly rounded FMA.
@inline function _df32_times_fp(x::DF32, y::Float32)
    high, residual = _df32_two_prod(x.hi, y)
    low = fma(x.lo, y, residual)
    result_high, result_low = _df32_fast_two_sum(high, low)
    return _df32_raw(result_high, result_low)
end

# J17 Algorithm 12, with M22 Theorem 2.8's corrected proof and sharper bound:
# <= (4u^2 + u^3/2)/(1+u)^2 < 4u^2, p >= 5. The low-by-low product and
# this exact FMA order are part of the algorithm, not optional refinements.
@inline function _df32_times_dw(x::DF32, y::DF32)
    high, residual = _df32_two_prod(x.hi, y.hi)
    low_product = x.lo * y.lo
    first_cross = fma(x.hi, y.lo, low_product)
    second_cross = fma(x.lo, y.hi, first_cross)
    low = residual + second_cross
    result_high, result_low = _df32_fast_two_sum(high, low)
    return _df32_raw(result_high, result_low)
end

# J17 Algorithm 15 / Theorem 6.2: <= 3u^2, p >= 4, y != 0. The
# subtraction order is DWDivFP3; changing it selects a different theorem.
@inline function _df32_div_fp(x::DF32, y::Float32)
    quotient = x.hi / y
    product_high, product_low = _df32_two_prod(quotient, y)
    high_error = x.hi - product_high
    product_error = high_error - product_low
    residual = product_error + x.lo
    correction = residual / y
    result_high, result_low = _df32_fast_two_sum(quotient, correction)
    return _df32_raw(result_high, result_low)
end

# J17 Algorithm 18 / Theorem 7.2: <= 9.8u^2, p >= 13 (the authors' Coq
# DWDDW3_correct uses p >= 14; p=24 satisfies both). The final printed name
# DWTimesDW2 is a typo: the cited Algorithm 12/proof and formal supplement use
# DWTimesDW3. Fast2Sum's exponent condition here is proved in that supplement;
# it is NOT necessary to replace it with a magnitude comparison or TwoSum.
@inline function _df32_div_dw(x::DF32, y::DF32)
    reciprocal = 1.0f0 / y.hi
    high_error = fma(-y.hi, reciprocal, 1.0f0)
    low_error = -(y.lo * reciprocal)
    error_high, error_low = _df32_fast_two_sum(high_error, low_error)
    correction = _df32_times_fp(_df32_raw(error_high, error_low), reciprocal)
    inverse = _df32_plus_fp(correction, reciprocal)
    return _df32_times_dw(x, inverse)
end

@inline Base.:+(x::DF32, y::DF32) = _df32_plus_dw(x, y)
@inline Base.:+(x::DF32, y::Float32) = _df32_plus_fp(x, y)
@inline Base.:+(x::Float32, y::DF32) = _df32_plus_fp(y, x)
@inline Base.:-(x::DF32) = _df32_raw(-x.hi, -x.lo)
@inline Base.:-(x::DF32, y::DF32) = _df32_plus_dw(x, -y)
@inline Base.:-(x::DF32, y::Float32) = _df32_plus_fp(x, -y)
@inline Base.:-(x::Float32, y::DF32) = _df32_plus_fp(-y, x)
@inline Base.:*(x::DF32, y::DF32) = _df32_times_dw(x, y)
@inline Base.:*(x::DF32, y::Float32) = _df32_times_fp(x, y)
@inline Base.:*(x::Float32, y::DF32) = _df32_times_fp(y, x)
@inline Base.:/(x::DF32, y::DF32) = _df32_div_dw(x, y)
@inline Base.:/(x::DF32, y::Float32) = _df32_div_fp(x, y)
@inline Base.:/(x::Float32, y::DF32) = _df32_div_dw(_df32_raw(x, 0.0f0), y)

@inline Base.zero(::Type{DF32}) = _df32_raw(0.0f0, 0.0f0)
@inline Base.zero(::DF32) = zero(DF32)
@inline Base.one(::Type{DF32}) = _df32_raw(1.0f0, 0.0f0)
@inline Base.one(::DF32) = one(DF32)
@inline Base.Float32(x::DF32) = x.hi
@inline Base.isfinite(x::DF32) = isfinite(x.hi) && isfinite(x.lo)
@inline Base.iszero(x::DF32) = iszero(x.hi) && iszero(x.lo)
