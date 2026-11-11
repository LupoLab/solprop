# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# State-sized arithmetic stays binary32. Only the final per-column merge and
# translated scalar tolerance construction use host binary64 (SEP 33).
struct _Float32NormScale{A, I}
    mantissa::A
    exponent::I
end

function _march_norm_scale(state::AbstractMatrix{ComplexF32})
    _float32_sum_count(Base.checked_mul(2, size(state, 1)))
    return _Float32NormScale(
        similar(state, Float32, size(state, 1)), similar(state, Int, size(state, 1)),
    )
end
_march_norm_scale(state::AbstractMatrix{ComplexF64}) = nothing

@inline _march_scalar(::Type{Float64}, value::Float64) = value
function _march_scalar(::Type{Float32}, value::Float64)
    narrowed = Float32(value)
    isfinite(narrowed) && (iszero(value) || abs(narrowed) >= floatmin(Float32)) ||
        throw(ArgumentError("binary32 march coefficient is outside the normal range"))
    return narrowed
end

function _float32_state_parts!(parts, state, mask, offset, execution)
    rows, columns = size(state)
    each_column!(execution, columns) do column
        largest = 0.0f0
        finite = true
        for row in 1:rows
            value = state[row, column]
            finite &= isfinite(value)
            if mask[row]
                largest = max(largest, abs(real(value)), abs(imag(value)))
            end
        end
        total = _Float32Sum()
        if finite && !iszero(largest)
            for row in 1:rows
                if mask[row]
                    value = state[row, column]
                    first, second = real(value) / largest, imag(value) / largest
                    total = _float32_add(total, first * first)
                    total = _float32_add(total, second * second)
                end
            end
        end
        parts[offset, column] = finite ? largest : Inf32
        parts[offset + 1, column] = _float32_value(total)
    end
    return nothing
end

function _float32_norm_value(parts, offset)
    scale, total = 0.0, 0.0
    for column in axes(parts, 2)
        scale, total = _rk_merge_sumsq(
            scale, total, Float64(parts[offset, column]),
            Float64(parts[offset + 1, column]),
        )
    end
    return scale * sqrt(total)
end

function _float32_norm_scale!(scale, weights, mask, relative, atol)
    for row in eachindex(weights)
        mask[row] || continue
        denominator = relative * weights[row] + atol
        isfinite(denominator) && denominator >= 0 || return false
        mantissa, exponent = frexp(denominator)
        scale.mantissa[row], scale.exponent[row] = Float32(mantissa), exponent
    end
    return true
end

@inline function _float32_norm_ratio(value, mantissa, exponent)
    iszero(mantissa) && return iszero(value) ? 0.0f0 : Inf32
    return ldexp(value, -exponent) / mantissa
end

function _float32_defect_parts!(parts, defect, mask, scale, execution)
    rows, columns = size(defect)
    mantissas, exponents = scale.mantissa, scale.exponent
    each_column!(execution, columns) do column
        largest = 0.0f0
        for row in 1:rows
            if mask[row]
                value = defect[row, column]
                first = _float32_norm_ratio(real(value), mantissas[row], exponents[row])
                second = _float32_norm_ratio(imag(value), mantissas[row], exponents[row])
                largest = max(largest, abs(first), abs(second))
            end
        end
        total = _Float32Sum()
        if isfinite(largest) && !iszero(largest)
            for row in 1:rows
                if mask[row]
                    value = defect[row, column]
                    first = _float32_norm_ratio(
                        real(value), mantissas[row], exponents[row],
                    ) / largest
                    second = _float32_norm_ratio(
                        imag(value), mantissas[row], exponents[row],
                    ) / largest
                    total = _float32_add(total, first * first)
                    total = _float32_add(total, second * second)
                end
            end
        end
        parts[1, column] = isfinite(largest) ? largest : Inf32
        parts[2, column] = _float32_value(total)
    end
    return nothing
end

function _float32_error_norm!(work, state, mask, weights, rtol, atol, execution)
    parts = work.norm_parts
    _float32_state_parts!(parts, state, mask, 1, execution)
    _float32_state_parts!(parts, work.candidate, mask, 3, execution)
    magnitude = max(_float32_norm_value(parts, 1), _float32_norm_value(parts, 3))
    isfinite(magnitude) || return Inf
    _float32_norm_scale!(work.norm_scale, weights, mask, rtol * magnitude, atol) || return Inf
    _float32_defect_parts!(parts, work.defect, mask, work.norm_scale, execution)
    return _float32_norm_value(parts, 1)
end
