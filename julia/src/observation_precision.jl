# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Sum2s, Ogita--Rump--Oishi (2005), Algorithm 4.1, uses the protected TwoSum
# already shared by DF32. Its correction is an ordinary working-precision sum;
# the final result obeys Proposition 4.5 for n*u<1 and finite intermediates.
# DOI: 10.1137/030601818. No reassociation or contraction is permitted here.
struct _Float32Sum
    total::Float32
    correction::Float32
end
_Float32Sum() = _Float32Sum(0.0f0, 0.0f0)

@inline function _float32_add(sum::_Float32Sum, value::Float32)
    total, error = _df32_two_sum(sum.total, value)
    return _Float32Sum(total, sum.correction + error)
end
@inline _float32_value(sum::_Float32Sum) = sum.total + sum.correction

function _float32_sum_count(count::Int)
    0 <= count < (1 << 24) ||
        throw(ArgumentError("binary32 compensated reduction needs fewer than 2^24 terms"))
    return nothing
end

function _observation_scale_exponents(exponents::Tuple)
    all(exponent -> -1074 <= exponent <= 1023, exponents) ||
        throw(ArgumentError("observation scales require finite nonzero binary64 powers"))
    return nothing
end

# A reconstructed closure uses a base field scale plus a wide coefficient.
# Strip work may also combine two wide factors and one stored component scale.
# Each constituent exponent has magnitude <=1074; the derived integer need not
# itself describe a materializable Float64 power. The complete restoration graph
# combines fewer than 32 such blocks, well within even a signed 32-bit Int.
# OBSERVATION_PRECISION.md gives the per-path bound and its finite-input hypotheses.
function _observation_derived_exponents(exponents::Tuple)
    limit = 4 * 1074
    all(exponent -> -limit <= exponent <= limit, exponents) ||
        throw(
        ArgumentError(
            "derived observation exponent exceeds four finite scale blocks",
        )
    )
    return nothing
end

# Component maxima avoid overflowing abs(complex) at finite component extremes.
# The normalized components lie in [-1,1], including original subnormal inputs.
function _observation_exponent(values::AbstractArray)
    largest = zero(real(zero(eltype(values))))
    for value in values
        isfinite(value) || throw(ArgumentError("observation normalization is nonfinite"))
        largest = max(largest, abs(real(value)), abs(imag(value)))
    end
    return iszero(largest) ? 0 : last(frexp(largest))
end

function _observation_column_exponent(values::AbstractArray, column, window = 1)
    largest = zero(real(zero(eltype(values))))
    for row in axes(values, 1)
        value = values[row, column, window]
        isfinite(value) || throw(ArgumentError("observation normalization is nonfinite"))
        largest = max(largest, abs(real(value)), abs(imag(value)))
    end
    return iszero(largest) ? 0 : last(frexp(largest))
end

@inline _observation_scaled(value::Float32, exponent::Int) = ldexp(value, -exponent)
@inline _observation_scaled(value::Float64, exponent::Int) =
    Float32(ldexp(value, -exponent))
@inline _observation_scaled(value::Integer, exponent::Int) =
    _observation_scaled(Float64(value), exponent)
@inline _observation_scaled(value::Real, exponent::Int) =
    throw(ArgumentError("observation normalization requires binary32 or binary64 values"))
@inline _observation_scaled(value::Complex, exponent::Int) = complex(
    _observation_scaled(real(value), exponent),
    _observation_scaled(imag(value), exponent),
)

@inline function _observation_pair(first, second, first_exponent, second_exponent)
    a = _observation_scaled(first, first_exponent)
    b = _observation_scaled(second, second_exponent)
    return real(a) * real(b) + imag(a) * imag(b)
end

@inline function _observation_square(value, exponent)
    scaled = _observation_scaled(value, exponent)
    return real(scaled) * real(scaled) + imag(scaled) * imag(scaled)
end

# Choose scales from actual products, not independently from their operands:
# a large coefficient can make a tiny sample the dominant physical term.
# At most eight mantissas in [1/2,1) are multiplied, so every intermediate is
# normal. Only the final alignment can underflow. Zero uses a scale sentinel
# that cannot determine the exponent of a nonzero reduction.
@inline function _observation_product_exponent(values::Tuple)
    1 <= length(values) <= 8 ||
        throw(ArgumentError("observation products require one to eight factors"))
    any(iszero, values) && return typemin(Int)
    return _observation_exponent_sum(values)
end

@inline _observation_exponent_sum(::Tuple{}) = 0
@inline _observation_exponent_sum(values::Tuple) =
    last(frexp(first(values))) + _observation_exponent_sum(Base.tail(values))

@inline _observation_product_scale(exponent::Int) =
    exponent == typemin(Int) ? 0 : exponent

@inline _observation_align(value::Float32, exponent::Int) = ldexp(value, exponent)
@inline _observation_host_align(value::Float32, exponent::Int) =
    ldexp(Float64(value), exponent)

@inline _observation_shifted_exponent(exponent::Int, shift::Int) =
    exponent == typemin(Int) ? exponent : exponent + shift

@inline function _observation_product(values::Tuple, exponent::Int)
    1 <= length(values) <= 8 ||
        throw(ArgumentError("observation products require one to eight factors"))
    any(iszero, values) && return 0.0f0
    product, power = _observation_product_parts(values, 1.0f0, 0)
    return ldexp(product, power - exponent)
end

# Tuple recursion specializes each mixed-precision host coefficient separately;
# iterating a heterogeneous tuple can box frexp's mixed result on this Julia pin.
@inline _observation_product_parts(::Tuple{}, product::Float32, exponent::Int) =
    (product, exponent)
@inline function _observation_product_parts(values::Tuple, product::Float32, exponent::Int)
    mantissa, shift = frexp(first(values))
    return _observation_product_parts(
        Base.tail(values), product * Float32(mantissa), exponent + shift
    )
end

@inline function _observation_dot_exponent(first, second, factors::Tuple = ())
    return max(
        _observation_product_exponent((real(first), real(second), factors...)),
        _observation_product_exponent((imag(first), imag(second), factors...)),
    )
end

@inline function _observation_dot(first, second, exponent, factors::Tuple = ())
    return _observation_product((real(first), real(second), factors...), exponent) +
        _observation_product((imag(first), imag(second), factors...), exponent)
end

function _observation_dot_exponent(first::AbstractArray, second::AbstractArray)
    exponent = typemin(Int)
    for index in eachindex(first, second)
        exponent = max(exponent, _observation_dot_exponent(first[index], second[index]))
    end
    return _observation_product_scale(exponent)
end

function _observation_dot_column_exponent(first, second, column, window = 1)
    exponent = typemin(Int)
    for row in axes(first, 1)
        exponent = max(
            exponent,
            _observation_dot_exponent(
                first[row, column, window], second[
                    row,
                    column, window,
                ]
            ),
        )
    end
    return _observation_product_scale(exponent)
end

@inline function _observation_abs_parts(value)
    largest = max(abs(real(value)), abs(imag(value)))
    iszero(largest) && return (0.0f0, 0)
    exponent = last(frexp(largest))
    return abs(_observation_scaled(value, exponent)), exponent
end

@inline function _observation_abs_exponent(first, second, factors::Tuple)
    a, a_exponent = _observation_abs_parts(first)
    b, b_exponent = _observation_abs_parts(second)
    scale = _observation_product_exponent((a, b, factors...))
    return scale == typemin(Int) ? scale : scale + a_exponent + b_exponent
end

@inline function _observation_abs_product(first, second, factors::Tuple, exponent)
    a, a_exponent = _observation_abs_parts(first)
    b, b_exponent = _observation_abs_parts(second)
    return _observation_product((a, b, factors...), exponent - a_exponent - b_exponent)
end

function _observation_restore(value::Real, exponent::Int, factor::Float64 = 1.0)
    isfinite(value) && isfinite(factor) ||
        throw(ArgumentError("observation physical result must be finite"))
    iszero(value) && return Float64(value)
    iszero(factor) && return 0.0
    fraction, value_exponent = frexp(Float64(value))
    multiplier, factor_exponent = frexp(factor)
    result = ldexp(fraction * multiplier, exponent + value_exponent + factor_exponent)
    isfinite(result) || throw(ArgumentError("observation physical result overflowed"))
    return result
end

function _observation_restore_ratio(
        value::Real, exponent::Int, factors::Tuple, divisors::Tuple,
    )
    isfinite(value) && all(isfinite, factors) &&
        all(factor -> isfinite(factor) && !iszero(factor), divisors) ||
        throw(ArgumentError("observation physical factors must be finite"))
    iszero(value) && return Float64(value)
    any(iszero, factors) && return 0.0
    fraction, value_scale = frexp(Float64(value))
    scale = exponent + value_scale
    for factor in factors
        mantissa, power = frexp(Float64(factor))
        fraction *= mantissa
        scale += power
    end
    for divisor in divisors
        mantissa, power = frexp(Float64(divisor))
        fraction /= mantissa
        scale -= power
    end
    result = ldexp(fraction, scale)
    isfinite(result) || throw(ArgumentError("observation physical result overflowed"))
    return result
end

mutable struct _ObservationScales{V}
    columns::V
    nodes::V
    spectral::Int
    temporal::Int
    windows::Int
    spatial::Int
    axis::Int
end
_ObservationScales(columns::Int, nodes::Int = 0) =
    _ObservationScales(zeros(Int, columns), zeros(Int, nodes), 0, 0, 0, 0, 0)
