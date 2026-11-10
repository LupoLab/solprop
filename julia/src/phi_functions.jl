# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public phi_values

const _PHI_RECIPROCAL_FACTORIAL = (
    1.0, 1.0, 0.5, 1 / 6, 1 / 24, 1 / 120, 1 / 720, 1 / 5040, 1 / 40320,
)
const _PHI_INVERSE_DIVISOR = ntuple(index -> 1.0 / index, 31)

# These protected recurrences deliberately use separate products and sums. The
# production exponential Adams method (SEP 16) uses join 3, at most 24 Taylor terms,
# exact binary halving and the doubling identity below; no reassociation is allowed.
@inline function _phi_values(argument::ComplexF64, ::Val{N}) where {N}
    reciprocal = _PHI_RECIPROCAL_FACTORIAL
    result = ntuple(index -> ComplexF64(reciprocal[index]), Val(N + 1))
    iszero(argument) && return result
    N == 0 && return (exp(argument),)
    magnitude = abs(argument)
    if magnitude > 3.0
        inverse = inv(argument)
        result = Base.setindex(result, exp(argument), 1)
        for order in 1:N
            value = (result[order] - reciprocal[order]) * inverse
            result = Base.setindex(result, value, order + 1)
        end
        return result
    end
    scale = 1.0
    squarings = 0
    while magnitude > 0.5
        scale *= 0.5
        magnitude *= 0.5
        squarings += 1
    end
    reduced = scale * argument
    term = ComplexF64(reciprocal[N + 1])
    total = term
    stop = (0.25eps(Float64))^2
    for index in 1:23
        term = (term * reduced) * _PHI_INVERSE_DIVISOR[index + N]
        total += term
        # The remaining ratio is at most (1/2)/(index+N+1). A quarter-ulp term
        # therefore leaves a tail well below half an ulp of the accumulated sum.
        abs2(term) <= stop * abs2(total) && break
    end
    result = Base.setindex(result, total, N + 1)
    for order in (N - 1):-1:1
        result = Base.setindex(
            result, reciprocal[order + 1] + reduced * result[order + 2], order + 1,
        )
    end
    result = Base.setindex(result, exp(reduced), 1)
    for unused in 1:squarings
        old = result
        result = Base.setindex(result, old[1] * old[1], 1)
        half = 1.0
        for order in 1:N
            half *= 0.5
            value = old[1] * old[order + 1]
            for index in 1:order
                value += reciprocal[order - index + 1] * old[index + 1]
            end
            result = Base.setindex(result, half * value, order + 1)
        end
    end
    return result
end

"""
    phi_values(argument::ComplexF64, ::Val{N})

Return the protected binary64 tuple `phi_0(argument), ..., phi_N(argument)`.

The dimensionless finite argument has real part at most 700, and `N` is an integer
from zero through eight. Here `phi_k(x) = sum(x^m/(m+k)!, m=0:Inf)`. At zero the
result is exactly the rounded reciprocal-factorial tuple. For `abs(x) <= 3`, evaluate
the highest order by a scaled Taylor series, recur downward, then double; elsewhere
recur upward from `exp(x)`. This is the production exponential Adams algorithm in
SEP 16 and the protected small-argument policy of SEP 25.

The doubling identity is
`2^k phi_k(2x) = exp(x)phi_k(x) + sum(phi_j(x)/(k-j)!, j=1:k)`.
It follows by splitting the defining exponential convolution at its midpoint.
Do not wrap this helper in fast-math or contract its cancellation-sensitive updates.

# Throws
- `ArgumentError`: For an unsupported order, nonfinite argument or excessive gain.
"""
@inline function phi_values(argument::ComplexF64, order::Val{N}) where {N}
    N isa Integer && 0 <= N <= 8 ||
        throw(ArgumentError("phi order must be an integer from zero through eight"))
    isfinite(argument) && real(argument) <= 700 ||
        throw(ArgumentError("phi argument must be finite with real part at most 700"))
    return _phi_values(argument, order)
end
