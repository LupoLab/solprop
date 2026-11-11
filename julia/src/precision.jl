# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public SourceArithmetic, source_type, source_state_scale, source_field_scale

"""
    SourceArithmetic(T; state_exponent=0, field_exponent=state_exponent)

Declare source arithmetic and exact physical power-of-two scales.

`T` is `Float64` or `Float32`. The stored state represents physical modal amplitude
divided by `2^state_exponent`; nodal samples represent physical electric amplitude
divided by `2^field_exponent`. With map exponents folded at upload, the exponents
coincide. Keep the physical scale separate from cubic and plasma-loss coefficients.
Both powers must be finite nonzero binary64 values. Binary64 uses zero exponents,
preserving its existing equations. This record alone does not admit numerical inputs
or prove that a particular binary32 coefficient/intermediate is in range.

Throw `ArgumentError` for an unsupported type, scale or scaled binary64 request.
"""
struct SourceArithmetic{T <: AbstractFloat}
    state_exponent::Int
    field_exponent::Int

    function SourceArithmetic(
            ::Type{T}; state_exponent::Int = 0, field_exponent::Int = state_exponent,
        ) where {T <: AbstractFloat}
        T in (Float32, Float64) ||
            throw(ArgumentError("source arithmetic requires binary32 or binary64"))
        all(exponent -> -1074 <= exponent <= 1023, (state_exponent, field_exponent)) ||
            throw(ArgumentError("source scales must be finite nonzero binary64 powers"))
        T === Float64 && !(iszero(state_exponent) && iszero(field_exponent)) &&
            throw(ArgumentError("binary64 source arithmetic requires unit scales"))
        return new{T}(state_exponent, field_exponent)
    end
end

function SourceArithmetic(
        ::Type; state_exponent::Int = 0, field_exponent::Int = state_exponent,
    )
    throw(ArgumentError("source arithmetic requires binary32 or binary64"))
end

"""
    source_type(arithmetic::SourceArithmetic)

Return the real arithmetic type of source arrays and pointwise kernels.
"""
source_type(::SourceArithmetic{T}) where {T} = T

"""
    source_state_scale(arithmetic::SourceArithmetic)

Return the exact binary64 power that converts stored state amplitudes to physical units.
"""
source_state_scale(arithmetic::SourceArithmetic) = ldexp(1.0, arithmetic.state_exponent)

"""
    source_field_scale(arithmetic::SourceArithmetic)

Return the exact binary64 power that converts stored nodal samples to physical units.
"""
source_field_scale(arithmetic::SourceArithmetic) = ldexp(1.0, arithmetic.field_exponent)
