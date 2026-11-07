# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Double-float: a value carried as the unevaluated sum of two binary32s, about 48
# significand bits. A binary32 run on a device without fast binary64 (an Apple GPU
# has none, an A40 runs it at 1/64 rate) forms the operator's exponents, the
# propagators and the Adams flow in it, and rounds each result once to binary32: the
# precision SEP 33 asks of them (4.1 item 3). The state itself stays binary32.
#
# The primitives are the mlx backend's (src/solprop/backends/mlx_gpu/arithmetic.py),
# whose accuracy SEP 33 measured: Knuth's two-sum, the fused two-product, and complex
# arithmetic on pairs. Every operation here is plain Julia over Float32, so the same
# code runs in a CPU loop, a CUDA kernel or a Metal kernel. Nothing here may be
# reassociated or contracted; Julia does neither without `@fastmath` or `muladd`.

"""A double-float: `hi + lo` with `|lo| <= ulp(hi) / 2`."""
struct DF32 <: AbstractFloat
    hi::Float32
    lo::Float32
end

@inline function two_sum(a::Float32, b::Float32)
    s = a + b
    bb = s - a
    s, (a - (s - bb)) + (b - bb)
end
@inline function quick_two_sum(a::Float32, b::Float32)
    s = a + b
    s, b - (s - a)
end
@inline function two_prod(a::Float32, b::Float32)
    p = a * b
    p, fma(a, b, -p)
end

DF32(x::Float32) = DF32(x, 0.0f0)
DF32(x::DF32) = x
# On the host only: a binary64 value split exactly enough (the pair holds 48 bits).
function DF32(x::Float64)
    hi = Float32(x)
    DF32(hi, Float32(x - Float64(hi)))
end
# Integers exactly, without binary64: the remainder of the rounded high part.
function DF32(x::Integer)
    hi = Float32(x)
    DF32(hi, Float32(x - unsafe_trunc(typeof(x), hi)))
end
Base.Float32(x::DF32) = x.hi + x.lo
Base.Float64(x::DF32) = Float64(x.hi) + Float64(x.lo)
Base.promote_rule(::Type{DF32}, ::Type{Float32}) = DF32
Base.promote_rule(::Type{DF32}, ::Type{<:Integer}) = DF32

Base.zero(::Type{DF32}) = DF32(0.0f0, 0.0f0)
Base.one(::Type{DF32}) = DF32(1.0f0, 0.0f0)
Base.zero(::DF32) = zero(DF32)
Base.one(::DF32) = one(DF32)
# The pair's unit roundoff, about 2^-48 (Float32's 24 bits twice).
Base.eps(::Type{DF32}) = DF32(Float32(2.0^-48), 0.0f0)
Base.isnan(x::DF32) = isnan(x.hi)
Base.isinf(x::DF32) = isinf(x.hi)
Base.isfinite(x::DF32) = isfinite(x.hi)
Base.iszero(x::DF32) = iszero(x.hi)
Base.signbit(x::DF32) = signbit(x.hi)

@inline function Base.:+(x::DF32, y::DF32)
    s, e = two_sum(x.hi, y.hi)
    DF32(quick_two_sum(s, (e + x.lo) + y.lo)...)
end
@inline Base.:-(x::DF32) = DF32(-x.hi, -x.lo)
@inline Base.:-(x::DF32, y::DF32) = x + (-y)
@inline function Base.:*(x::DF32, y::DF32)
    p, e = two_prod(x.hi, y.hi)
    DF32(quick_two_sum(p, fma(x.hi, y.lo, fma(x.lo, y.hi, e)))...)
end
# A binary32 factor exactly: the product of the high part and the low part's term.
@inline function Base.:*(x::DF32, y::Float32)
    p, e = two_prod(x.hi, y)
    DF32(quick_two_sum(p, fma(x.lo, y, e))...)
end
@inline Base.:*(y::Float32, x::DF32) = x * y
@inline function Base.:/(a::DF32, b::Float32)
    q1 = a.hi / b
    p, e = two_prod(q1, b)
    r = ((a.hi - p) - e) + a.lo
    DF32(quick_two_sum(q1, r / b)...)
end
@inline function Base.:/(a::DF32, b::DF32)
    q1 = a.hi / b.hi
    r = a - b * q1
    q2 = r.hi / b.hi
    r = r - b * q2
    q3 = r.hi / b.hi
    DF32(quick_two_sum(q1, q2)...) + DF32(q3)
end
@inline Base.inv(x::DF32) = one(DF32) / x
@inline Base.abs(x::DF32) = signbit(x.hi) ? -x : x
@inline Base.:<(x::DF32, y::DF32) = x.hi < y.hi || (x.hi == y.hi && x.lo < y.lo)
@inline Base.:<=(x::DF32, y::DF32) = x.hi < y.hi || (x.hi == y.hi && x.lo <= y.lo)
@inline Base.:(==)(x::DF32, y::DF32) = x.hi == y.hi && x.lo == y.lo

# Complex double-floats: Base's generic `+`, `-`, `*` and `abs2` serve; the division,
# the reciprocal and the exponential are written out, since Base's forms scale through
# `abs`, `floatmin` and binary64 constants.
const CDF32 = Complex{DF32}
@inline function Base.:/(a::CDF32, b::CDF32)
    d = real(b) * real(b) + imag(b) * imag(b)
    Complex((real(a) * real(b) + imag(a) * imag(b)) / d,
            (imag(a) * real(b) - real(a) * imag(b)) / d)
end
@inline Base.inv(b::CDF32) = Complex(one(DF32), zero(DF32)) / b
@inline Base.:/(a::CDF32, b::DF32) = Complex(real(a) / b, imag(a) / b)

const TWO_PI_DF = DF32(2pi)
const INV_TWO_PI_32 = Float32(1 / 2pi)

"""A phase less whole turns of a double-float 2 pi, as a double-float: about 2^-48 of
the phase for one of up to 1e7 rad."""
@inline function reduce_phase(phase::DF32)
    turns = round(phase.hi * INV_TWO_PI_32)
    phase + TWO_PI_DF * (-turns)
end

"""`exp(y)` for `|y| <= 1/2`: Horner's form of 18 Taylor terms (a tail below 2^-60)."""
@inline function exp_small(y::CDF32)
    e = Complex(one(DF32), zero(DF32))
    for m in 18:-1:1
        p = e * y
        e = Complex(real(p) / Float32(m) + one(DF32), imag(p) / Float32(m))
    end
    e
end

"""Scale by `2^-s`, exactly: `s` halvings of both parts."""
@inline function halve(x::CDF32, s)
    for _ in 1:s
        x = Complex(real(x) * 0.5f0, imag(x) * 0.5f0)
    end
    x
end

"""`exp(x)` in double-float: the phase reduced by whole turns of 2 pi, the argument
halved to at most 1/2, the series, and squared back."""
@inline function Base.exp(x::CDF32)
    r = Complex(real(x), reduce_phase(imag(x)))
    size = sqrt(real(r).hi * real(r).hi + imag(r).hi * imag(r).hi)
    s = 0
    while size > 0.5f0
        size *= 0.5f0
        s += 1
    end
    e = exp_small(halve(r, s))
    for _ in 1:s
        e = e * e
    end
    e
end

"""`exp(+-z)` rounded once to binary32 each, for a propagator pair.

With `W = Float64` it is Base's binary64 exponential, as the engine always formed it.
In double-float it is the mlx backend's pair (arithmetic.py, `_PAIR`): the phase
reduced by turns of 2 pi, binary32 cosine and sine of the reduced angle, renormalised
by one Newton step on the double-float squared modulus (binary32 cosine and sine
leave it a few u off one, with a bias a long march accumulates in the energy), and
the magnitude `exp(hi) (1 + lo)`.
"""
@inline wide_exp_pair(z::ComplexF64) = (exp(z), exp(-z))
@inline function wide_exp_pair(z::CDF32)
    re = real(z)
    phase = reduce_phase(imag(z))
    angle = phase.hi + phase.lo
    c, s = cos(angle), sin(angle)
    norm = DF32(two_prod(c, c)...) + DF32(two_prod(s, s)...)
    delta = -0.5f0 * ((norm.hi - 1.0f0) + norm.lo)
    cc, ss = fma(c, delta, c), fma(s, delta, s)
    forward = exp(re.hi) * (1.0f0 + re.lo)
    back = exp(-re.hi) * (1.0f0 - re.lo)
    (Complex(forward * cc, forward * ss), Complex(back * cc, -(back * ss)))
end
@inline wide_exp(z::ComplexF64) = exp(z)
@inline wide_exp(z::CDF32) = first(wide_exp_pair(z))

"""The wide type of a binary32 run: `"binary64"` (the default off Metal) or `"df32"`."""
function wide_type(T, choice)
    T === Float64 && return Float64
    choice == "binary64" && return Float64
    choice == "df32" && return DF32
    throw(ArgumentError("SOLPROP_JULIA_WIDE must be binary64 or df32"))
end

# The engine's wide type is held as a zero of it (`c.wide`): a type in a named tuple is
# a `DataType` to inference, a value is concrete.
wide_eltype(c) = typeof(c.wide)
"""A host binary64 scalar in the engine's wide type: itself, or split into a pair."""
wide_value(c, x::Float64) = oftype(c.wide, x)

"""A host binary64 array in the engine's wide type, for `upload`."""
widen_host(::Type{Float64}, a::AbstractArray{<:Union{Float64,ComplexF64}}) = a
widen_host(::Type{DF32}, a::AbstractArray{Float64}) = DF32.(a)
widen_host(::Type{DF32}, a::AbstractArray{ComplexF64}) =
    [Complex(DF32(real(z)), DF32(imag(z))) for z in a]

# The double-float propagator pair of a binary64 exponent, widened back (tests).
exp_pair_snapshot_df(z) =
    map(ComplexF64, wide_exp_pair(only(widen_host(DF32, [ComplexF64(z)]))))
