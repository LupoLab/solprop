# Branch-free binary64 `exp` and `log` for loops a CPU vectorises (native's vmath.hpp
# role, ADR-059): each within one ulp of the correctly rounded value on the domain
# below, by Cody-Waite reduction and Taylor polynomials evaluated with explicit
# fused multiply-adds (a library function's own error bound, SEP 25; no fast-math).
# Integers are formed by the 1.5 * 2^52 shift, so AVX2 (no 64-bit conversions)
# vectorises them as well as AVX-512. Other precisions take Base's functions.

const SHIFT = 6.755399441055744e15  # 1.5 * 2^52: x + SHIFT holds round(x) in its low bits
const INV_LN2 = 1.4426950408889634
const LN2_HI = 6.93147180369123816490e-01  # ln 2, its low 21 bits zero: n LN2_HI is exact
const LN2_LO = 1.90821492927058770002e-10
const ONE_BITS = reinterpret(Int64, 1.0)
const SQRT_HALF_BITS = Int64(0x3fe6a09e) << 32  # just below sqrt(1/2)'s bits
const MANTISSA = Int64(0x000fffffffffffff)

"""Horner's rule with fused steps, `p r + c` for each coefficient in turn, unrolled."""
@inline horner(r, p, c::Tuple) = horner(r, fma(p, r, first(c)), Base.tail(c))
@inline horner(r, p, ::Tuple{}) = p

"""2^q for an integer-valued q in [-1022, 1023], from its bits."""
@inline pow2(q::Int64) = reinterpret(Float64, (q + 1023) << 52)

"""`exp(x)`, branch-free; any binary64, NaN to NaN, overflow to Inf, underflow to 0."""
@inline function vexp(x::Float64)
    x = clamp(x, -1000.0, 1000.0)
    shifted = fma(x, INV_LN2, SHIFT)
    q = reinterpret(Int64, shifted) - reinterpret(Int64, SHIFT)
    n = shifted - SHIFT
    r = fma(n, -LN2_HI, x)
    r = fma(n, -LN2_LO, r)
    # exp(r) - 1 - r on |r| <= ln2 / 2 by Taylor's series to r^13 (remainder < 2^-60).
    p = horner(r, 1 / 6227020800, (1 / 479001600, 1 / 39916800, 1 / 3628800, 1 / 362880,
                                     1 / 40320, 1 / 5040, 1 / 720, 1 / 120, 1 / 24, 1 / 6,
                                     1 / 2))
    u = fma(r * r, p, r) + 1.0
    # 2^q as two factors, so that q from -1443 to 1443 neither wraps nor loses range.
    half = q >> 1
    u * pow2(half) * pow2(q - half)
end
vexp(x) = exp(x)

"""`log(x)` for a positive normal binary64 (NaN to NaN), branch-free."""
@inline function vlog(x::Float64)
    bits = reinterpret(Int64, x)
    # x = m 2^e with m in [sqrt(1/2), sqrt(2)): shift the mantissa's top half down.
    adjusted = bits + (ONE_BITS - SQRT_HALF_BITS)
    e = (adjusted >> 52) - 1023
    m = reinterpret(Float64, (adjusted & MANTISSA) + SQRT_HALF_BITS)
    f = m - 1.0
    s = f / (2.0 + f)
    z = s * s
    # log(1 + f) = 2 atanh(s) = 2s + s R(z), R(z) = sum_k 2 z^k / (2k + 1), to z^11.
    R = z * horner(z, 2 / 23, (2 / 21, 2 / 19, 2 / 17, 2 / 15, 2 / 13, 2 / 11, 2 / 9,
                                2 / 7, 2 / 5, 2 / 3))
    half = 0.5 * f * f
    k = reinterpret(Float64, e + reinterpret(Int64, SHIFT)) - SHIFT
    value = fma(k, LN2_HI, f - (half - fma(s, half + R, k * LN2_LO)))
    ifelse(x == x, value, x)
end
vlog(x) = log(x)
