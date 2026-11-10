# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public QuadratureWorkspace, quadrature_error!

"""
    QuadratureWorkspace(execution, n_columns::Int)

Allocate bounded CPU binary64 scratch for an embedded-source quadrature diagnostic.

`execution` is a CPU `KernelExecution`; `n_columns` is the positive modal column count.
The independently owned `maxima` and `sums` matrices have shapes `(3,n_columns)` and
`(2,n_columns)`. They retain per-column norm scales and normalized squared sums; no
field, source, accepted-step count, warning or observation history is retained.

Throw `ArgumentError` for unsupported execution or a nonpositive column count.
"""
struct QuadratureWorkspace{M <: AbstractMatrix{Float64}, S <: AbstractMatrix{Float64}}
    maxima::M
    sums::S
end

function QuadratureWorkspace(execution::KernelExecution, n_columns::Int)
    oncpu(execution) || throw(ArgumentError("quadrature reduction currently requires CPU"))
    n_columns > 0 || throw(ArgumentError("quadrature reduction needs positive columns"))
    return QuadratureWorkspace(
        dzeros(execution, Float64, 3, n_columns),
        dzeros(execution, Float64, 2, n_columns)
    )
end

function _quadrature_workspace(work, execution)
    oncpu(execution) || throw(ArgumentError("quadrature reduction currently requires CPU"))
    columns = size(work.maxima, 2)
    size(work.maxima) == (3, columns) && size(work.sums) == (2, columns) && columns > 0 ||
        throw(DimensionMismatch("quadrature workspace dimensions disagree"))
    Base.require_one_based_indexing(work.maxima, work.sums)
    Base.mightalias(work.maxima, work.sums) &&
        throw(ArgumentError("quadrature workspace buffers must be disjoint"))
    return columns
end

@inline function _quadrature_difference_max(scale, doubled, coarse, fine)
    difference = coarse - fine
    if isfinite(difference)
        magnitude = abs(difference)
        doubled && (magnitude *= 0.5)
    else
        # Only a finite-input subtraction overflow reaches this branch. Halving its
        # large operands is exact; its difference is representable. Keep the extra
        # factor of two in an exponent instead of overflowing that difference again.
        magnitude = abs(0.5 * coarse - 0.5 * fine)
        if !doubled
            scale *= 0.5
            doubled = true
        end
    end
    return max(scale, magnitude), doubled
end

function _quadrature_scales!(maxima, fine, coarse, execution)
    each_column!(execution, size(fine, 2)) do column
        column_fine_scale = column_difference_scale = 0.0
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
        maxima[3, column] = column_doubled ? 1.0 : 0.0
    end
    fine_scale = difference_scale = 0.0
    doubled = false
    for column in axes(maxima, 2)
        fine_scale = max(fine_scale, maxima[1, column])
        local_scale, local_doubled = maxima[2, column], maxima[3, column] == 1.0
        if local_doubled && !doubled
            difference_scale *= 0.5
            doubled = true
        elseif doubled && !local_doubled
            local_scale *= 0.5
        end
        difference_scale = max(difference_scale, local_scale)
    end
    return fine_scale, difference_scale, doubled
end

@inline function _quadrature_scaled_difference(coarse, fine, scale, doubled)
    difference = coarse - fine
    if doubled
        if isfinite(difference)
            return 0.5 * (difference / scale)
        end
        return (0.5 * coarse - 0.5 * fine) / scale
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

"""
    quadrature_error!(work, fine, coarse, execution) -> Float64

Compare an accepted source's embedded coarse and fine modal projections.

`work` is a `QuadratureWorkspace`. `fine` and `coarse` are finite, one-based CPU
`AbstractMatrix{ComplexF64}` inputs of equal nonempty shape `(n_omega,n_columns)`.
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

Throw `ArgumentError` for nonfinite source entries, partial availability, nonbinary64
data, aliases, unsupported execution or a quotient that overflows binary64. Throw
`DimensionMismatch` for incompatible dimensions. Admission precedes scratch writes.
"""
function quadrature_error!(work::QuadratureWorkspace, fine, coarse, execution)::Float64
    columns = _quadrature_workspace(work, execution)
    fine === nothing && coarse === nothing && return NaN
    fine isa AbstractMatrix{ComplexF64} && coarse isa AbstractMatrix{ComplexF64} ||
        throw(ArgumentError("quadrature sources require two complex binary64 matrices"))
    size(fine) == size(coarse) && size(fine, 1) > 0 && size(fine, 2) == columns ||
        throw(DimensionMismatch("quadrature source dimensions disagree"))
    Base.require_one_based_indexing(fine, coarse)
    (
        Base.mightalias(fine, work.maxima) || Base.mightalias(fine, work.sums) ||
            Base.mightalias(coarse, work.maxima) || Base.mightalias(coarse, work.sums)
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
    fine_fraction, fine_exponent = frexp(fine_scale)
    difference_fraction, difference_exponent = frexp(difference_scale)
    ratio = ldexp(
        (difference_fraction / fine_fraction) * sqrt(difference_sum / fine_sum),
        difference_exponent - fine_exponent + Int(doubled)
    )
    isfinite(ratio) || throw(ArgumentError("quadrature error overflows binary64"))
    return ratio
end
