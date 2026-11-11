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
    spectral::Int
    temporal::Int
    windows::Int
    spatial::Int
    axis::Int
end
_ObservationScales(columns::Int) =
    _ObservationScales(zeros(Int, columns), 0, 0, 0, 0, 0)
