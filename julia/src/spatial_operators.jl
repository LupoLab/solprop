# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public SeparableSpatialOperator, spatial_prefactor!

"""
    SeparableSpatialOperator(execution; k_squared, k_squared_low, kperp_sq,
        kperp_sq_low, frame, spectral_rate, prefactor_numerator, prefactor_clamped,
        active, evanescent_rate, z_span, kperp_rate=nothing, n_pol=1, kappa_sq=nothing)

Own constant scalar Cartesian/line coefficients without a spectral coefficient volume.

All arrays are finite binary64 vectors copied at preparation. The complex material
square and residual are in m⁻² with one entry per frequency. The real nonnegative
transverse square and its signed residual have one entry per transverse column.
`frame` and the nonnegative power rates `spectral_rate`/`kperp_rate` are in m⁻¹;
missing `kperp_rate` is zero. The prefactor numerator and nonpropagating clamp are
purely imaginary translated values, already including temporal crop scaling. Boolean
`active` has the frequency extent. `evanescent_rate` is a finite positive field rate
in m⁻¹, and `z_span` is the finite increasing admitted interval in metres.

The protected assembly follows `translate.spatial.longitudinal_root` and SEP 22:
`q = (k_squared-kperp_sq) + (k_squared_low-kperp_sq_low)`, then the principal square
root. The propagating test is exactly `real(q)>0`, with no added grazing threshold.
Its attenuation is `imag(sqrt(q))`; otherwise it is bounded by `evanescent_rate`.
The frame and transverse loss apply on the active band, while spectral loss applies
everywhere. Every resulting rate and prefactor is checked for finiteness before the
record is published. This does not certify a DF32 assembly graph or a device backend.

Throw `ArgumentError` or `DimensionMismatch` for unsupported execution, nonconstant
coefficients, invalid data or nonfinite assembled values. Vector polarisation and
planar `kappa_sq` are refused by name. Preparation retains no caller-owned array.
"""
struct SeparableSpatialOperator
    k_squared::Vector{ComplexF64}
    k_squared_low::Vector{ComplexF64}
    kperp_sq::Vector{Float64}
    kperp_sq_low::Vector{Float64}
    frame::Vector{Float64}
    spectral_rate::Vector{Float64}
    prefactor_numerator::Vector{ComplexF64}
    prefactor_clamped::Vector{ComplexF64}
    active::Vector{Bool}
    evanescent_rate::Float64
    z_span::NTuple{2, Float64}
    kperp_rate::Vector{Float64}

    function SeparableSpatialOperator(
            execution::KernelExecution;
            k_squared, k_squared_low, kperp_sq, kperp_sq_low, frame, spectral_rate,
            prefactor_numerator, prefactor_clamped, active,
            evanescent_rate::Float64, z_span::NTuple{2, Float64},
            kperp_rate = nothing, n_pol::Int = 1, kappa_sq = nothing,
        )
        oncpu(execution) ||
            throw(ArgumentError("separable spatial operators currently require CPU"))
        n_pol == 1 ||
            throw(ArgumentError("Julia spatial sources refuse vector polarisation"))
        kappa_sq === nothing ||
            throw(ArgumentError("Julia spatial operators refuse the planar route"))
        # Assert the owned storage invariant before `new`, rather than invoking
        # its implicit conversion path for an abstract caller-provided array.
        k_squared = _spatial_constant_copy(k_squared, ComplexF64)::Vector{ComplexF64}
        k_squared_low =
            _spatial_constant_copy(k_squared_low, ComplexF64)::Vector{ComplexF64}
        prefactor_numerator =
            _spatial_constant_copy(prefactor_numerator, ComplexF64)::Vector{ComplexF64}
        prefactor_clamped =
            _spatial_constant_copy(prefactor_clamped, ComplexF64)::Vector{ComplexF64}
        kperp_sq = _spatial_constant_copy(kperp_sq, Float64)::Vector{Float64}
        kperp_sq_low = _spatial_constant_copy(kperp_sq_low, Float64)::Vector{Float64}
        frame = _spatial_constant_copy(frame, Float64)::Vector{Float64}
        spectral_rate = _spatial_constant_copy(spectral_rate, Float64)::Vector{Float64}
        active isa AbstractVector{Bool} && !isempty(active) &&
            axes(active, 1) == Base.OneTo(length(active)) ||
            throw(ArgumentError("the spatial mask needs a nonempty one-based Bool vector"))
        n_frequency, n_columns = length(k_squared), length(kperp_sq)
        all(
            array -> length(array) == n_frequency, (
                k_squared_low, frame, spectral_rate, prefactor_numerator, prefactor_clamped,
                active,
            )
        ) && length(kperp_sq_low) == n_columns ||
            throw(DimensionMismatch("separable spatial coefficient extents disagree"))
        all(>=(0), kperp_sq) && all(>=(0), spectral_rate) ||
            throw(ArgumentError("spatial squares and power rates must be nonnegative"))
        all(iszero, real.(prefactor_numerator)) && all(iszero, real.(prefactor_clamped)) ||
            throw(ArgumentError("spatial prefactors must be purely imaginary"))
        isfinite(evanescent_rate) && evanescent_rate > 0 &&
            all(isfinite, z_span) && first(z_span) < last(z_span) ||
            throw(ArgumentError("invalid spatial evanescent clamp or admitted interval"))
        transverse_rate = kperp_rate === nothing ? zeros(n_columns) :
            _spatial_constant_copy(kperp_rate, Float64)::Vector{Float64}
        length(transverse_rate) == n_columns ||
            throw(DimensionMismatch("the transverse rate must have one value per column"))
        all(>=(0), transverse_rate) ||
            throw(ArgumentError("the transverse power rate must be nonnegative"))
        owned = new(
            k_squared, k_squared_low, kperp_sq, kperp_sq_low,
            frame, spectral_rate, prefactor_numerator,
            prefactor_clamped, Array(active), evanescent_rate, z_span, transverse_rate,
        )
        tables = _spatial_operator_tables(owned)
        for column in 1:n_columns, row in 1:n_frequency
            square = _spatial_square(tables, row, column)
            root = sqrt(square)
            unmasked = real(square) > 0 ? tables.numerator[row] / real(root) :
                tables.clamped[row]
            rate, prefactor = _spatial_element(tables, row, column)
            isfinite(square) && isfinite(root) && isfinite(unmasked) &&
                isfinite(rate) && isfinite(prefactor) ||
                throw(ArgumentError("assembled spatial rate or prefactor is nonfinite"))
        end
        return owned
    end
end

function _spatial_constant_copy(array::StridedVector{T}, ::Type{T}) where {T}
    !isempty(array) && all(isfinite, array) ||
        throw(
        ArgumentError(
            "Cartesian material coefficients require finite constant binary64 vectors"
        )
    )
    axes(array, 1) == Base.OneTo(length(array)) ||
        throw(ArgumentError("spatial coefficient vectors require one-based axes"))
    output = Vector{T}(undef, length(array))
    copyto!(output, array)
    return output
end

function _spatial_constant_copy(array, ::Type{T}) where {T}
    throw(
        ArgumentError(
            "Cartesian material coefficients require finite constant binary64 vectors"
        )
    )
end

# Tuple/array captures use the launcher's rooted Adapt path; do not capture an
# ordinary managed operator record in a Polyester body.
@inline function _spatial_operator_tables(operator)
    return (
        k_squared = operator.k_squared, k_squared_low = operator.k_squared_low,
        kperp_sq = operator.kperp_sq, kperp_sq_low = operator.kperp_sq_low,
        frame = operator.frame, spectral_rate = operator.spectral_rate,
        numerator = operator.prefactor_numerator, clamped = operator.prefactor_clamped,
        active = operator.active, evanescent_rate = operator.evanescent_rate,
        kperp_rate = operator.kperp_rate,
    )
end

@inline function _spatial_square(tables, row, column)
    # Keep each subtraction and the residual correction in this represented order.
    return (tables.k_squared[row] - tables.kperp_sq[column]) +
        (tables.k_squared_low[row] - tables.kperp_sq_low[column])
end

@inline function _spatial_element(tables, row, column)
    square = _spatial_square(tables, row, column)
    root = sqrt(square)
    propagating = real(square) > 0
    attenuation = propagating ? imag(root) :
        min(abs(imag(root)), tables.evanescent_rate)
    rate = complex(
        -attenuation - 0.5 * tables.kperp_rate[column],
        -(real(root) - tables.frame[row]),
    )
    prefactor = propagating ? tables.numerator[row] / real(root) : tables.clamped[row]
    # The represented root/branch above is common to active and inactive bins.
    # An inactive coefficient is structurally zero before adding spectral loss.
    if !tables.active[row]
        rate = zero(rate)
        prefactor = zero(prefactor)
    end
    return rate - 0.5 * tables.spectral_rate[row], prefactor
end

@inline _spatial_operator_shape(operator::SeparableSpatialOperator) =
    (length(operator.k_squared), length(operator.kperp_sq))

@inline function _spatial_operator_alias(array, operator::SeparableSpatialOperator)
    return _spatial_operator_alias(
        array, (
            operator.k_squared, operator.k_squared_low, operator.kperp_sq,
            operator.kperp_sq_low, operator.frame, operator.spectral_rate,
            operator.prefactor_numerator, operator.prefactor_clamped, operator.active,
            operator.kperp_rate,
        )
    )
end

@inline _spatial_operator_alias(array, ::Tuple{}) = false
@inline function _spatial_operator_alias(array, inputs::Tuple{T, Vararg}) where {T}
    return _transfer_arrays_alias(array, first(inputs)) ||
        _spatial_operator_alias(array, Base.tail(inputs))
end

function _spatial_operator_output(output, operator)
    output isa StridedMatrix{ComplexF64} ||
        throw(ArgumentError("spatial coefficient output requires strided complex binary64"))
    size(output) == _spatial_operator_shape(operator) ||
        throw(DimensionMismatch("spatial coefficient output has the wrong shape"))
    _spatial_operator_alias(output, operator) &&
        throw(ArgumentError("a spatial output must not overwrite its coefficient inputs"))
    return nothing
end

function _spatial_operator_position(operator, position)
    isfinite(position) && first(operator.z_span) <= position <= last(operator.z_span) ||
        throw(ArgumentError("the position lies outside the spatial operator span"))
    return nothing
end

function operator_exponent!(
        output, operator::SeparableSpatialOperator, z0::Real, z1::Real,
        execution::KernelExecution; work = nothing,
    )
    _spatial_operator_output(output, operator)
    _spatial_operator_position(operator, z0)
    _spatial_operator_position(operator, z1)
    # Match host binary64 coordinates without subtracting in a bounded input type.
    distance = Float64(z1) - Float64(z0)
    isfinite(distance) || throw(ArgumentError("the spatial interval must be finite"))
    tables = _spatial_operator_tables(operator)
    each!(execution, size(output)...) do row, column
        output[row, column] = iszero(distance) ? zero(eltype(output)) :
            first(_spatial_element(tables, row, column)) * distance
    end
    return output
end

function operator_rate!(
        output, operator::SeparableSpatialOperator, position::Real,
        execution::KernelExecution; before::Bool = false, work = nothing,
    )
    _spatial_operator_output(output, operator)
    _spatial_operator_position(operator, position)
    tables = _spatial_operator_tables(operator)
    each!(execution, size(output)...) do row, column
        output[row, column] = first(_spatial_element(tables, row, column))
    end
    return output
end

"""
    spatial_prefactor!(output, operator::SeparableSpatialOperator, execution)

Write the translated spatial prefactor, retaining crop scaling and exact inactive zeros.

`output` is a strided complex binary64 matrix `(n_frequency,n_columns)` disjoint from
all coefficient arrays. The denominator is the real principal longitudinal root only
where the corrected square is propagating; all other columns use the supplied clamp.
Return `output`. Warmed CPU calls allocate nothing; invalid shape/storage/overlap
refuses before mutation.
"""
function spatial_prefactor!(output, operator::SeparableSpatialOperator, execution)
    _spatial_operator_output(output, operator)
    tables = _spatial_operator_tables(operator)
    each!(execution, size(output)...) do row, column
        output[row, column] = last(_spatial_element(tables, row, column))
    end
    return output
end
