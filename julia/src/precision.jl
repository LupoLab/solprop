# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public SourceArithmetic, source_type, source_state_scale, source_field_scale
public SourceRange, source_narrow, source_narrow!, source_coefficient

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

"""
    SourceRange

Describe one named, admitted source buffer conversion.

`name` identifies the numerical buffer. `block_exponent` is the largest-magnitude
binary exponent of the represented input after the requested `folded_exponent`.
It is zero for an all-zero input. If `normalized` is true, the returned buffer must
be multiplied by `2^block_exponent` to recover those units; otherwise that power has
already been folded. `largest` and `smallest_nonzero` are the binary64 magnitudes
before rounding of the buffer actually stored (normalized when requested). The
smallest nonzero magnitude is zero for an all-zero buffer.

This immutable fact records preparation admission, not a bound on later arithmetic.
The direct binary32 rule matches SEP 33: below unit maximum, every nonzero entry
must be normal; at or above unit maximum, normwise negligible entries may underflow.
Normalized buffers therefore require their exponent to remain explicit in the pass.
"""
struct SourceRange
    name::Symbol
    block_exponent::Int
    folded_exponent::Int
    normalized::Bool
    largest::Float64
    smallest_nonzero::Float64
end

@inline function _source_scaled(value::Real, exponent::Int)
    return ldexp(Float64(value)::Float64, exponent)
end
@inline function _source_scaled(value::Complex, exponent::Int)
    return complex(
        ldexp(Float64(real(value))::Float64, exponent),
        ldexp(Float64(imag(value))::Float64, exponent)
    )
end

@inline _source_storage_type(::Type{T}, ::Type{<:Real}) where {T} = T
@inline _source_storage_type(::Type{T}, ::Type{<:Complex}) where {T} = Complex{T}
@inline _source_real_type(::Type{T}) where {T <: AbstractFloat} = T
@inline _source_real_type(::Type{Complex{T}}) where {T <: AbstractFloat} = T

function _source_exponent_sum(left::Int, right::Int, name::Symbol)
    value, overflow = Base.add_with_overflow(left, right)
    overflow && throw(ArgumentError("$name source scale exponent overflows Int"))
    return value
end

function _source_range(
        input, ::Type{T}, name::Symbol, power::Int, normalized::Bool,
    ) where {T}
    T in (Float32, Float64) ||
        throw(ArgumentError("$name source buffer requires binary32 or binary64 storage"))
    largest_input = 0.0
    for value in input
        magnitude = abs(_source_scaled(value, 0))
        isfinite(magnitude) || throw(ArgumentError("$name source buffer must be finite"))
        largest_input = max(largest_input, magnitude)
    end
    original = iszero(largest_input) ? 0 : frexp(largest_input)[2] - 1
    block = iszero(largest_input) ? 0 : _source_exponent_sum(original, power, name)
    shift = normalized ? -original : power
    largest, smallest = 0.0, Inf
    vanished = false
    for value in input
        magnitude = abs(_source_scaled(value, shift))
        isfinite(magnitude) ||
            throw(ArgumentError("$name scaled source buffer exceeds binary64 range"))
        largest = max(largest, magnitude)
        if iszero(magnitude)
            vanished |= !iszero(value)
        else
            smallest = min(smallest, magnitude)
        end
    end
    if T === Float32
        largest < Float64(floatmax(Float32)) ||
            throw(ArgumentError("$name source buffer reaches binary32 overflow range"))
        largest < 1 && (vanished || smallest < Float64(floatmin(Float32))) &&
            throw(ArgumentError("$name source buffer needs an explicit block exponent"))
    elseif vanished
        throw(ArgumentError("$name source buffer has binary64 scaling underflow"))
    end
    fact = SourceRange(
        name, block, power, normalized, largest, isinf(smallest) ? 0.0 : smallest
    )
    return fact, shift
end

function _source_host_narrow(
        values::AbstractArray{S, N}, arithmetic::SourceArithmetic{T},
        name::Symbol; exponent::Int = 0, normalize::Bool = false,
    ) where {S <: Union{Real, Complex}, N, T}
    fact, shift = _source_range(values, T, name, exponent, normalize)
    all(dimension -> axes(values, dimension) == Base.OneTo(size(values, dimension)), 1:N) ||
        throw(ArgumentError("$name source buffer must use one-based axes"))
    storage = _source_storage_type(T, S)
    prepared = Array{storage, N}(undef, size(values))
    for index in CartesianIndices(prepared)
        prepared[index] = _source_scaled(values[index], shift)
    end
    return prepared, fact
end

"""
    source_narrow(execution, values, arithmetic, name; exponent=0, normalize=false)

Own and convert a host numerical buffer with a named range and explicit scale fact.

`values` is a real or complex host array. `arithmetic` selects binary32 or binary64;
shape and real/complex kind are preserved. Apply the exact integer power `exponent`
before conversion. When `normalize=true`, keep the largest magnitude in `[1,2)`
instead, and return its required power in `SourceRange.block_exponent`. Normalization
never forms a potentially overflowing intermediate scale. Copy the admitted buffer
to `execution` with `upload` and return `(owned_array, range_fact)`.

The admission rule is the NumPy single-precision oracle's direct-fold rule, not an
elementwise relative-error promise. No nonzero small buffer is silently discarded.
Throw `ArgumentError` naming the buffer for nonfinite data, exponent overflow or
unsupported range. Inputs are unchanged, including on refusal.
"""
function source_narrow(
        execution, values::AbstractArray, arithmetic::SourceArithmetic, name::Symbol;
        exponent::Int = 0, normalize::Bool = false,
    )
    prepared, fact = _source_host_narrow(values, arithmetic, name; exponent, normalize)
    return upload(execution, prepared), fact
end

# Existing binary64 constructors already validate their finite inputs. A no-op
# ownership copy is not a narrowing and must not impose the binary32 magnitude
# admission (a finite complex pair can have an overflowing binary64 magnitude).
function _source_prepare_host(
        values::AbstractArray{S, N}, arithmetic::SourceArithmetic{T}, name::Symbol;
        exponent::Int = 0,
    ) where {S <: Union{Real, Complex}, N, T}
    T === Float32 &&
        return _source_host_narrow(values, arithmetic, name; exponent)
    iszero(exponent) ||
        throw(ArgumentError("binary64 source preparation requires unit scale"))
    all(dimension -> axes(values, dimension) == Base.OneTo(size(values, dimension)), 1:N) ||
        throw(ArgumentError("$name source buffer must use one-based axes"))
    owned = Array{S, N}(undef, size(values))
    for index in CartesianIndices(owned)
        owned[index] = values[index]
    end
    return owned, nothing
end

function _source_prepare(
        execution, values::AbstractArray, arithmetic::SourceArithmetic, name::Symbol;
        exponent::Int = 0,
    )
    owned, fact = _source_prepare_host(values, arithmetic, name; exponent)
    return upload(execution, owned), fact
end

"""
    source_narrow!(destination, values, name; exponent=0, normalize=false)

Convert host values into existing host storage after complete range admission.

Both arrays are strided, one-based and have the same shape and real/complex kind;
`destination` uses binary32 or binary64. `values` may be the exact destination,
otherwise storage must be disjoint. Powers and the returned `SourceRange` have the
same meaning as `source_narrow`. No destination element is written before every
range and alias check passes. A device pass copies the admitted host buffer through
its own prepared transfer boundary; this routine performs no device scalar indexing.

Return the range fact. Warm calls allocate nothing. Throw `DimensionMismatch` for
shape mismatch and `ArgumentError` for unsupported precision, aliases or range.
"""
function source_narrow!(
        destination::StridedArray{D, N}, values::StridedArray{S, N}, name::Symbol;
        exponent::Int = 0, normalize::Bool = false,
    ) where {
        D <: Union{AbstractFloat, Complex{<:AbstractFloat}}, S <: Union{Real, Complex}, N,
    }
    axes(destination) == axes(values) ||
        throw(DimensionMismatch("$name source conversion axes differ"))
    (D <: Real) == (S <: Real) ||
        throw(ArgumentError("$name source conversion must preserve real/complex kind"))
    destination === values || !_transfer_arrays_alias(destination, values) ||
        throw(ArgumentError("$name source conversion has overlapping input and output"))
    fact, shift = _source_range(values, _source_real_type(D), name, exponent, normalize)
    for index in eachindex(destination, values)
        destination[index] = _source_scaled(values[index], shift)
    end
    return fact
end

"""
    source_coefficient(arithmetic, name, factors...; exponent=0)

Combine finite real host coefficients and an integer power before one response rounding.

Accumulate binary64 mantissas and integer exponents, renormalizing after each factor.
This avoids an intermediate raw material/density/scale product overflowing or erasing
a nonzero coefficient. The final value uses `source_type(arithmetic)`; binary32
requires a normal value or an exact structural zero, as the SEP 33 oracle does.
All inputs are validated even when a factor is zero. A zero retains the represented
product's sign. The routine has no allocation on a successful warmed path.

Throw `ArgumentError` naming the coefficient for nonfinite factors, exponent overflow
or final underflow/overflow. This host operation does not license intermediate range
assumptions inside pointwise response kernels.
"""
function source_coefficient(
        ::SourceArithmetic{T}, name::Symbol, factors::Vararg{Real, N}; exponent::Int = 0,
    ) where {T, N}
    mantissa, power, exact_zero = 1.0, exponent, false
    for factor in factors
        value = Float64(factor)::Float64
        isfinite(value) || throw(ArgumentError("$name source coefficient must be finite"))
        if iszero(value)
            exact_zero = true
            negative = signbit(mantissa) ⊻ signbit(value)
            mantissa = copysign(abs(mantissa), negative ? -1.0 : 1.0)
            continue
        end
        fraction, shift = frexp(value)
        mantissa *= fraction
        power = _source_exponent_sum(power, shift, name)
        mantissa, shift = frexp(mantissa)
        power = _source_exponent_sum(power, shift, name)
    end
    exact_zero && return copysign(zero(T), mantissa)
    value = ldexp(mantissa, power)
    rounded = T(value)
    isfinite(rounded) && !iszero(rounded) ||
        throw(ArgumentError("$name source coefficient exceeds $T range"))
    T === Float32 && abs(rounded) < floatmin(Float32) &&
        throw(ArgumentError("$name source coefficient is below normal binary32 range"))
    return rounded
end
