# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public QuadratureWorkspace, quadrature_error!

"""
    QuadratureWorkspace(execution, n_columns::Int;
                        arithmetic=SourceArithmetic(Float64))

Allocate bounded CPU scratch for an embedded-source quadrature diagnostic.

`execution` is a CPU `KernelExecution`; `n_columns` is the positive modal column count.
The independently owned `maxima` and `sums` matrices have shapes `(3,n_columns)` and
`(2,n_columns)` in `source_type(arithmetic)`. They retain per-column norm scales and
normalized squared sums; the common physical source scale cancels from the ratio. No
field, source, accepted-step count, warning or observation history is retained.

Throw `ArgumentError` for unsupported execution or a nonpositive column count.
"""
struct QuadratureWorkspace{
        T <: Union{Float32, Float64}, M <: AbstractMatrix{T}, S <: AbstractMatrix{T},
    }
    maxima::M
    sums::S
end

function QuadratureWorkspace(
        execution::KernelExecution, n_columns::Int;
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    oncpu(execution) || throw(ArgumentError("quadrature reduction currently requires CPU"))
    n_columns > 0 || throw(ArgumentError("quadrature reduction needs positive columns"))
    T = source_type(arithmetic)::Union{Type{Float32}, Type{Float64}}
    if T === Float32
        _float32_sum_count(n_columns)
        return QuadratureWorkspace(
            dzeros(execution, Float32, 3, n_columns),
            dzeros(execution, Float32, 2, n_columns),
        )
    end
    return QuadratureWorkspace(
        dzeros(execution, Float64, 3, n_columns),
        dzeros(execution, Float64, 2, n_columns),
    )
end

function _quadrature_workspace(work, execution)
    oncpu(execution) || throw(ArgumentError("quadrature reduction currently requires CPU"))
    columns = size(work.maxima, 2)
    size(work.maxima) == (3, columns) && size(work.sums) == (2, columns) && columns > 0 ||
        throw(DimensionMismatch("quadrature workspace dimensions disagree"))
    Base.require_one_based_indexing(work.maxima, work.sums)
    _transfer_writable_destination(work.maxima)
    _transfer_writable_destination(work.sums)
    _transfer_arrays_alias(work.maxima, work.sums) &&
        throw(ArgumentError("quadrature workspace buffers must be disjoint"))
    return columns
end

@inline function _quadrature_difference_max(scale, doubled, coarse, fine)
    half = oftype(scale, 0.5)
    difference = coarse - fine
    if isfinite(difference)
        magnitude = abs(difference)
        doubled && (magnitude *= half)
    else
        # Only a finite-input subtraction overflow reaches this branch. Halving its
        # large operands is exact; its difference is representable. Keep the extra
        # factor of two in an exponent instead of overflowing that difference again.
        magnitude = abs(half * coarse - half * fine)
        if !doubled
            scale *= half
            doubled = true
        end
    end
    return max(scale, magnitude), doubled
end

function _quadrature_scales!(maxima, fine, coarse, execution)
    neutral = zero(eltype(maxima))
    half, unit = oftype(neutral, 0.5), one(neutral)
    each_column!(execution, size(fine, 2)) do column
        column_fine_scale = column_difference_scale = neutral
        column_doubled = false
        for row in axes(fine, 1)
            first, second = fine[row, column], coarse[row, column]
            column_fine_scale = max(column_fine_scale, abs(real(first)), abs(imag(first)))
            column_difference_scale, column_doubled = _quadrature_difference_max(
                column_difference_scale, column_doubled, real(second), real(first)
            )
            column_difference_scale, column_doubled = _quadrature_difference_max(
                column_difference_scale, column_doubled, imag(second), imag(first)
            )
        end
        maxima[1, column], maxima[2, column] = column_fine_scale, column_difference_scale
        maxima[3, column] = column_doubled ? unit : neutral
    end
    fine_scale = difference_scale = neutral
    doubled = false
    for column in axes(maxima, 2)
        fine_scale = max(fine_scale, maxima[1, column])
        local_scale, local_doubled = maxima[2, column], maxima[3, column] == unit
        if local_doubled && !doubled
            difference_scale *= half
            doubled = true
        elseif doubled && !local_doubled
            local_scale *= half
        end
        difference_scale = max(difference_scale, local_scale)
    end
    return fine_scale, difference_scale, doubled
end

@inline function _quadrature_scaled_difference(coarse, fine, scale, doubled)
    half = oftype(scale, 0.5)
    difference = coarse - fine
    if doubled
        if isfinite(difference)
            return half * (difference / scale)
        end
        return (half * coarse - half * fine) / scale
    end
    return difference / scale
end

function _quadrature_sums!(
        sums, fine, coarse, fine_scale, difference_scale, doubled, execution,
    )
    each_column!(execution, size(fine, 2)) do column
        column_fine_sum = column_difference_sum = 0.0
        for row in axes(fine, 1)
            first, second = fine[row, column], coarse[row, column]
            first_real, first_imaginary = real(first) / fine_scale, imag(first) / fine_scale
            difference_real = _quadrature_scaled_difference(
                real(second), real(first), difference_scale, doubled
            )
            difference_imaginary = _quadrature_scaled_difference(
                imag(second), imag(first), difference_scale, doubled
            )
            column_fine_sum += first_real * first_real + first_imaginary * first_imaginary
            column_difference_sum += difference_real * difference_real +
                difference_imaginary * difference_imaginary
        end
        sums[1, column], sums[2, column] = column_fine_sum, column_difference_sum
    end
    fine_sum = difference_sum = 0.0
    for column in axes(sums, 2)
        fine_sum += sums[1, column]
        difference_sum += sums[2, column]
    end
    return fine_sum, difference_sum
end

function _quadrature_sums!(
        sums::AbstractMatrix{Float32}, fine, coarse, fine_scale, difference_scale,
        doubled, execution,
    )
    each_column!(execution, size(fine, 2)) do column
        column_fine_sum = column_difference_sum = _Float32Sum()
        for row in axes(fine, 1)
            first, second = fine[row, column], coarse[row, column]
            first_real, first_imaginary = real(first) / fine_scale, imag(first) / fine_scale
            difference_real = _quadrature_scaled_difference(
                real(second), real(first), difference_scale, doubled,
            )
            difference_imaginary = _quadrature_scaled_difference(
                imag(second), imag(first), difference_scale, doubled,
            )
            column_fine_sum = _float32_add(
                column_fine_sum,
                first_real * first_real + first_imaginary * first_imaginary,
            )
            column_difference_sum = _float32_add(
                column_difference_sum,
                difference_real * difference_real +
                    difference_imaginary * difference_imaginary,
            )
        end
        sums[1, column] = _float32_value(column_fine_sum)
        sums[2, column] = _float32_value(column_difference_sum)
    end
    fine_sum = difference_sum = _Float32Sum()
    for column in axes(sums, 2)
        fine_sum = _float32_add(fine_sum, sums[1, column])
        difference_sum = _float32_add(difference_sum, sums[2, column])
    end
    return _float32_value(fine_sum), _float32_value(difference_sum)
end

"""
    quadrature_error!(work, fine, coarse, execution) -> Float64

Compare an accepted source's embedded coarse and fine modal projections.

`work` is a `QuadratureWorkspace`. `fine` and `coarse` are finite, one-based CPU
`AbstractMatrix{Complex{T}}` inputs of equal nonempty shape `(n_omega,n_columns)`,
where `T` matches the workspace's binary32 or binary64 precision.
They already contain the identical crop, frequency window and physical source factors,
including plasma streams, before the common temporal absorber. The dimensionless
result is `norm(coarse-fine)/norm(fine)`, as defined by
`src/solprop/backends/numpy_ref/backend.py::_update_alarm` and
`native/src/general.cpp::GeneralWorkspace::score_quadrature`.

Return unavailable NaN for two `nothing` inputs or an exactly zero fine source.
Equal nonzero inputs return exact zero. Source inputs are preserved and may alias one
another, but must be disjoint from all work buffers. Warmed calls allocate nothing.
The caller scores only accepted trials, validates the source's readiness, and owns
the last score, plane reset and warning delivery; this function advances no source.

Independent maximum-component scales keep squared sums bounded. Overflowing scalar
differences use a half-difference and a retained binary exponent. Combining the scales
with `frexp`/`ldexp` avoids an overflowing intermediate ratio. Gradual underflow is
retained, including normal final rounding to a subnormal result or zero. This is a
resolution diagnostic, not a certified bound on the continuum quadrature error.
Binary32 uses protected Sum2s for each row pass and its column-partial merge; both
pass lengths must be smaller than `2^24`. Its pointwise work and retained scratch
remain binary32. Only the final four scale/sum scalars widen to binary64. The
rounding derivation, including product and prior partial errors, is in
`QUADRATURE_PRECISION.md`. Binary64 retains its original reduction order.

Throw `ArgumentError` for nonfinite source entries, partial availability, mismatched
data, aliases, unsupported execution or a quotient that overflows binary64. Throw
`DimensionMismatch` for incompatible dimensions. Admission precedes scratch writes.
"""
function quadrature_error!(
        work::QuadratureWorkspace{T}, fine, coarse, execution,
    )::Float64 where {T}
    columns = _quadrature_workspace(work, execution)
    fine === nothing && coarse === nothing && return NaN
    fine isa AbstractMatrix{Complex{T}} && coarse isa AbstractMatrix{Complex{T}} ||
        throw(ArgumentError("quadrature source precision must match its workspace"))
    size(fine) == size(coarse) && size(fine, 1) > 0 && size(fine, 2) == columns ||
        throw(DimensionMismatch("quadrature source dimensions disagree"))
    Base.require_one_based_indexing(fine, coarse)
    if T === Float32
        _float32_sum_count(size(fine, 1))
        _float32_sum_count(columns)
    end
    (
        _transfer_arrays_alias(fine, work.maxima) ||
            _transfer_arrays_alias(fine, work.sums) ||
            _transfer_arrays_alias(coarse, work.maxima) ||
            _transfer_arrays_alias(coarse, work.sums)
    ) && throw(ArgumentError("quadrature sources must be disjoint from scratch"))
    all(isfinite, fine) && all(isfinite, coarse) ||
        throw(ArgumentError("quadrature source entries must be finite"))
    fine_scale, difference_scale, doubled = _quadrature_scales!(
        work.maxima, fine, coarse, execution
    )
    iszero(fine_scale) && return NaN
    iszero(difference_scale) && return 0.0
    fine_sum, difference_sum = _quadrature_sums!(
        work.sums, fine, coarse, fine_scale, difference_scale, doubled, execution
    )
    fine_fraction, fine_exponent = frexp(Float64(fine_scale))
    difference_fraction, difference_exponent = frexp(Float64(difference_scale))
    ratio = ldexp(
        (difference_fraction / fine_fraction) *
            sqrt(Float64(difference_sum) / Float64(fine_sum)),
        difference_exponent - fine_exponent + Int(doubled)
    )
    isfinite(ratio) || throw(ArgumentError("quadrature error overflows binary64"))
    return ratio
end
