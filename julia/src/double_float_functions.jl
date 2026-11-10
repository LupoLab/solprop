# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public check_df32_function_argument

# Successive RN-even MPFR splits of mathematical 2pi, not of rounded Float64 2pi.
# DF32_BOUNDS.md sections 2--5 derive the domain, allowances and polynomial orders.
# The functions require gradual underflow and RN-even division/FMA; neither the
# host checker nor a CPU test establishes that arithmetic model for a device.
const _DF32_TURN_WORDS = (
    reinterpret(Float32, 0x40c90fdb), reinterpret(Float32, 0xb43bbd2e),
    reinterpret(Float32, 0xa7f72ced),
)
const _DF32_INVERSE_TURN = reinterpret(Float32, 0x3e22f983)
const _DF32_RECIPROCAL_FACTORIAL = ntuple(
    index -> prepare_df32(1.0 / factorial(index - 1)), Val(9),
)

# Comparing the rounded Float64 sum is insufficient at a domain endpoint: a
# positive low word can be much smaller than a binary64 ulp of the high word.
@inline function _df32_abs_leq(value::DF32, limit::Float32)
    magnitude = signbit(value.hi) ? -value : value
    return magnitude.hi < limit || (magnitude.hi == limit && magnitude.lo <= 0.0f0)
end

"""
    check_df32_function_argument(z::Complex{DF32}; input_error::Float64=0.0)

Check the stored argument and supplied error for the initial DF32 function domain.

The dimensionless normalized finite word pairs must satisfy `abs(real(z)) <= 16`
and `abs(imag(z)) <= 2^20` as exact real sums. `input_error` is a finite nonnegative
upper bound on the real-plus-imaginary absolute difference from the requested
argument, including assembly and representation; it must be at most `2^-27`.
The caller supplies this independently derived bound. This function neither
discovers it nor verifies an arithmetic model, a device or a future trajectory.

The protected function kernels use binary32 RN-even arithmetic, correct FMA and
division, and gradual underflow. `DF32_BOUNDS.md` derives their finite-domain bounds
from explicit scalar operation graphs, Taylor series and convolution identities.
Successful checking is not runtime admission. No Base transcendental overload is
provided; use `prepare_df32` for checked host construction of each component.

# Returns
`nothing` when the representation, stored domain and supplied bound are valid.

# Throws
- `ArgumentError`: For nonfinite or nonnormalized words, an out-of-domain exact
  component, or a negative, nonfinite or excessive input-error bound.
"""
function check_df32_function_argument(z::Complex{DF32}; input_error::Float64 = 0.0)
    for component in (real(z), imag(z))
        isfinite(component) && component.hi == component.hi + component.lo ||
            throw(ArgumentError("DF32 functions require finite normalized word pairs"))
    end
    _df32_abs_leq(real(z), 16.0f0) ||
        throw(ArgumentError("DF32 function real range exceeds 16"))
    _df32_abs_leq(imag(z), 1048576.0f0) ||
        throw(ArgumentError("DF32 function phase range exceeds 2^20"))
    isfinite(input_error) && 0.0 <= input_error <= 2.0^-27 ||
        throw(ArgumentError("DF32 function input-error budget exceeds 2^-27"))
    return nothing
end

@inline function _df32_complex_add(x::Complex{DF32}, y::Complex{DF32})
    return Complex{DF32}(real(x) + real(y), imag(x) + imag(y))
end

@inline function _df32_complex_times(x::Complex{DF32}, y::Complex{DF32})
    ac, bd = real(x) * real(y), imag(x) * imag(y)
    ad, bc = real(x) * imag(y), imag(x) * real(y)
    return Complex{DF32}(ac - bd, ad + bc)
end

@inline function _df32_complex_times_real(x::Complex{DF32}, y::DF32)
    return Complex{DF32}(real(x) * y, imag(x) * y)
end

@inline function _df32_complex_div_integer(x::Complex{DF32}, divisor::Float32)
    return Complex{DF32}(real(x) / divisor, imag(x) / divisor)
end

@inline function _df32_scale_words(x::DF32, scale::Float32)
    high, low = _df32_fast_two_sum(x.hi * scale, x.lo * scale)
    return _df32_raw(high, low)
end

@inline function _df32_complex_scale(x::Complex{DF32}, scale::Float32)
    return Complex{DF32}(
        _df32_scale_words(real(x), scale), _df32_scale_words(imag(x), scale),
    )
end

@inline function _df32_complex_abs2(x::Complex{DF32})
    return real(x) * real(x) + imag(x) * imag(x)
end

# Only the large phi branch uses this inverse: 2.99 <= |x| <= 2^21. Its
# squared-modulus high word is a positive divisor in [8, 2^42].
@inline function _df32_complex_inverse(x::Complex{DF32})
    squared = _df32_complex_abs2(x)
    return Complex{DF32}(real(x) / squared, -(imag(x) / squared))
end

@inline function _df32_reduce_phase(theta::DF32)
    # The admitted phase gives finite |turns| < 2^18. Separate nearest-even
    # rounding from the proved-safe conversion, avoiding an exception slow path.
    selected = round(theta.hi * _DF32_INVERSE_TURN, RoundNearest)
    turns = unsafe_trunc(Int32, selected)
    iszero(turns) && return theta
    exact_integer = Float32(turns)
    remainder = theta
    for word in _DF32_TURN_WORDS
        high, low = _df32_two_prod(exact_integer, word)
        remainder = remainder - _df32_raw(high, low)
    end
    return remainder
end

# |z| <= 1/2. Fixed degree/order and explicit multiplication before division are
# part of the bound; a vendor exponential or reassociation is not interchangeable.
@inline function _df32_exp_small(z::Complex{DF32})
    result = Complex{DF32}(one(DF32), zero(DF32))
    for divisor in 18:-1:1
        product = _df32_complex_times(result, z)
        term = _df32_complex_div_integer(product, Float32(divisor))
        result = Complex{DF32}(real(term) + 1.0f0, imag(term))
    end
    return result
end

@inline function _df32_exp(z::Complex{DF32})
    reduced = Complex{DF32}(real(z), _df32_reduce_phase(imag(z)))
    threshold = 0.25f0
    scale = 1.0f0
    squarings = 0
    while !_df32_abs_leq(real(reduced), threshold) ||
            !_df32_abs_leq(imag(reduced), threshold)
        threshold *= 2.0f0
        scale *= 0.5f0
        squarings += 1
    end
    result = _df32_exp_small(_df32_complex_scale(reduced, scale))
    for unused in 1:squarings
        result = _df32_complex_times(result, result)
    end
    return result
end

@inline function _df32_sincos(theta::DF32)
    # Preserve the structural signed-zero sine rather than losing its sign in
    # additions of Taylor terms. The exact-zero cosine is positive one.
    iszero(theta) && return theta.hi, 1.0f0
    wide = _df32_exp(Complex{DF32}(zero(DF32), theta))
    cosine, sine = real(wide).hi, imag(wide).hi
    cosine_high, cosine_low = _df32_two_prod(cosine, cosine)
    sine_high, sine_low = _df32_two_prod(sine, sine)
    squared = _df32_raw(cosine_high, cosine_low) + _df32_raw(sine_high, sine_low)
    correction = -0.5f0 * ((squared.hi - 1.0f0) + squared.lo)
    return fma(sine, correction, sine), fma(cosine, correction, cosine)
end

@inline function _df32_exp_pair(z::Complex{DF32})
    sine, cosine = _df32_sincos(imag(z))
    positive = real(_df32_exp(Complex{DF32}(real(z), zero(DF32)))).hi
    negative = real(_df32_exp(Complex{DF32}(-real(z), zero(DF32)))).hi
    return (
        ComplexF32(positive * cosine, positive * sine),
        ComplexF32(negative * cosine, -(negative * sine)),
    )
end

@inline function _df32_phi_constants(::Val{N}) where {N}
    return ntuple(
        index -> Complex{DF32}(_DF32_RECIPROCAL_FACTORIAL[index], zero(DF32)), Val(N + 1),
    )
end

# N is a prepared integer from one through eight. Both branch helpers remain
# separate so the overlapping join can be tested against two independent bounds.
@inline function _df32_phi_small(z::Complex{DF32}, order::Val{N}) where {N}
    squared = _df32_complex_abs2(z)
    threshold = (1.0f0 - 4.0f0 * 2.0f0^-24) * 0.25f0
    scale = 1.0f0
    squarings = 0
    while squared.hi > threshold
        threshold *= 4.0f0
        scale *= 0.5f0
        squarings += 1
    end
    reduced = _df32_complex_scale(z, scale)
    result = _df32_phi_constants(order)
    term = result[N + 1]
    total = term
    for index in 1:23
        product = _df32_complex_times(term, reduced)
        term = _df32_complex_div_integer(product, Float32(index + N))
        total = _df32_complex_add(total, term)
    end
    result = Base.setindex(result, total, N + 1)
    for index in (N - 1):-1:1
        product = _df32_complex_times(reduced, result[index + 2])
        value = Complex{DF32}(
            real(product) + _DF32_RECIPROCAL_FACTORIAL[index + 1], imag(product),
        )
        result = Base.setindex(result, value, index + 1)
    end
    result = Base.setindex(result, _df32_exp_small(reduced), 1)
    for unused in 1:squarings
        old = result
        result = Base.setindex(result, _df32_complex_times(old[1], old[1]), 1)
        half = 1.0f0
        for index in 1:N
            half *= 0.5f0
            value = _df32_complex_times(old[1], old[index + 1])
            for term_index in 1:index
                product = _df32_complex_times_real(
                    old[term_index + 1], _DF32_RECIPROCAL_FACTORIAL[index - term_index + 1],
                )
                value = _df32_complex_add(value, product)
            end
            result = Base.setindex(result, _df32_complex_scale(value, half), index + 1)
        end
    end
    return result
end

@inline function _df32_phi_large(z::Complex{DF32}, order::Val{N}) where {N}
    result = _df32_phi_constants(order)
    inverse = _df32_complex_inverse(z)
    result = Base.setindex(result, _df32_exp(z), 1)
    for index in 1:N
        previous = result[index]
        difference = Complex{DF32}(
            real(previous) - _DF32_RECIPROCAL_FACTORIAL[index], imag(previous),
        )
        result = Base.setindex(result, _df32_complex_times(difference, inverse), index + 1)
    end
    return result
end

@inline function _df32_phi_values(z::Complex{DF32}, order::Val{N}) where {N}
    iszero(real(z)) && iszero(imag(z)) && return _df32_phi_constants(order)
    N == 0 && return (_df32_exp(z),)
    if _df32_complex_abs2(z).hi > 9.0f0
        return _df32_phi_large(z, order)
    end
    return _df32_phi_small(z, order)
end
