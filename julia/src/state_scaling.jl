# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# A launch or replacement field is in physical internal units. Scale its components
# separately: forming the reciprocal scale can overflow for a subnormal launch.
# Underflow during the final binary32 conversion is gradual, as in SEP 33's state
# conversion; table-specific normal-range admission does not apply to state entries.
@inline function _scaled_state_value(value, arithmetic::SourceArithmetic{Float32})
    exponent = -arithmetic.state_exponent
    return ComplexF32(
        ldexp(Float64(real(value)), exponent),
        ldexp(Float64(imag(value)), exponent),
    )
end
@inline _scaled_state_value(value, ::SourceArithmetic{Float64}) = ComplexF64(value)

function _prepare_scaled_state(execution, input, arithmetic::SourceArithmetic{T}) where {T}
    input isa AbstractMatrix{ComplexF64} ||
        throw(ArgumentError("session launch requires a physical binary64 complex matrix"))
    all(isfinite, input) || throw(ArgumentError("session launch must be finite"))
    all(value -> isfinite(_scaled_state_value(value, arithmetic)), input) ||
        throw(ArgumentError("session launch exceeds its scaled state precision"))
    owned = Matrix{Complex{T}}(undef, size(input))
    for index in eachindex(owned, input)
        owned[index] = _scaled_state_value(input[index], arithmetic)
    end
    return upload(execution, owned)
end

@inline function _physical_state_value(value, arithmetic::SourceArithmetic{Float32})
    exponent = arithmetic.state_exponent
    return ComplexF64(
        ldexp(Float64(real(value)), exponent),
        ldexp(Float64(imag(value)), exponent),
    )
end
@inline _physical_state_value(value, ::SourceArithmetic{Float64}) = ComplexF64(value)

function _scaled_absolute_tolerance(value::Float64, arithmetic::SourceArithmetic)
    isfinite(value) && value >= 0 ||
        throw(ArgumentError("the physical absolute tolerance must be finite and nonnegative"))
    scaled = ldexp(value, -arithmetic.state_exponent)
    isfinite(scaled) ||
        throw(ArgumentError("the scaled absolute tolerance exceeds binary64 range"))
    return scaled
end
