# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Branch-free exp and log in binary64, which a SIMD loop vectorises: Base's look up
# tables and branch, so a loop of them runs one lane at a time. Native calls SLEEF's
# 1-ulp vector functions for the same reason (native/include/solprop_native/vmath.hpp).
# Both here are within about one ulp (julia/test/runtests.jl checks it); binary32 and
# complex arguments keep Base's.

const LN2_HI = 6.93147180369123816490e-01
const LN2_LO = 1.90821492927058770002e-10

"""`exp(x)` by Cody-Waite reduction to |r| <= ln2/2 and its Taylor series to r^13."""
@inline function vexp(x::Float64)
    n = round(x * 1.4426950408889634)
    r = muladd(-n, LN2_LO, muladd(-n, LN2_HI, x))
    # r^14/14! < 5e-18 at |r| = ln2/2: below half an ulp of the sum.
    p = evalpoly(r, (1.0, 1.0, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720, 1 / 5040, 1 / 40320,
                     1 / 362880, 1 / 3628800, 1 / 39916800, 1 / 479001600, 1 / 6227020800))
    # 2^n in two exact factors, so that a subnormal result keeps its bits.
    k = unsafe_trunc(Int64, clamp(n, -2000.0, 2000.0))
    half = k >> 1
    scale1 = reinterpret(Float64, (clamp(half, -1022, 1023) + 1023) << 52)
    scale2 = reinterpret(Float64, (clamp(k - half, -1022, 1023) + 1023) << 52)
    value = p * scale1 * scale2
    value = ifelse(x > 709.782712893384, Inf, value)
    value = ifelse(x < -745.1332191019412, 0.0, value)
    ifelse(isnan(x), x, value)
end
@inline vexp(x) = exp(x)

"""`log(x)` for positive normal `x` (fdlibm's e_log.c, without its branches)."""
@inline function vlog(x::Float64)
    bits = reinterpret(UInt64, x)
    e = Int64((bits >> 52) & 0x7ff) - 1023
    m = reinterpret(Float64, (bits & 0x000fffffffffffff) | 0x3ff0000000000000)
    high = m > 1.4142135623730951
    m = ifelse(high, 0.5 * m, m)
    k = Float64(e + high)
    f = m - 1.0
    hfsq = 0.5 * f * f
    s = f / (2.0 + f)
    z = s * s
    w = z * z
    t1 = w * evalpoly(w, (3.999999999940941908e-01, 2.222219843214978396e-01, 1.531383769920937332e-01))
    t2 = z * evalpoly(w, (6.666666666666735130e-01, 2.857142874366239149e-01,
                          1.818357216161805012e-01, 1.479819860511658591e-01))
    R = t2 + t1
    k * LN2_HI - ((hfsq - (s * (hfsq + R) + k * LN2_LO)) - f)
end
@inline vlog(x) = log(x)
