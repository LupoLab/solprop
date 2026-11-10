# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public ObservationWeight, evaluate_weight!

"""
    ObservationWeight(execution, values::Matrix{Float64})
    ObservationWeight(table::ChebyshevTable, shape::Tuple{Int,Int})
    ObservationWeight(execution, terms::Array{Float64,3}, weights::ChebyshevTable)

Own a constant, direct-value table or expansion of modal power weights.

Values have shape `(n_omega,n_modes)` and are dimensionless. Expansion terms have
one additional last axis; their real weight table has one row per term. Tables
describe direct values, never cumulative propagation operators. Arrays are copied;
every evaluated power weight must be positive finite. Invalid data or shapes raise
`ArgumentError` or `DimensionMismatch` before observation workspace allocation.
"""
struct ObservationWeight{D, W}
    data::D
    weights::W
    shape::Tuple{Int, Int}
end

function ObservationWeight(execution, values::AbstractMatrix{Float64})
    !isempty(values) && all(value -> isfinite(value) && value > 0, values) ||
        throw(ArgumentError("modal power weights must be positive finite"))
    return ObservationWeight(upload(execution, values), nothing, size(values))
end

function ObservationWeight(execution, values::AbstractMatrix)
    throw(ArgumentError("constant modal power weights require real binary64 storage"))
end

function ObservationWeight(table::ChebyshevTable, shape::Tuple{Int, Int})
    eltype(table.coefficients) === Float64 &&
        eltype(table.breakpoints) === Float64 && all(>(0), shape) ||
        throw(ArgumentError("power-weight tables require real binary64 values"))
    prod(shape) == size(table.coefficients, 1) ||
        throw(DimensionMismatch("power-weight table does not match its modal shape"))
    return ObservationWeight(_copy_power_weight_table(table), nothing, shape)
end

function ObservationWeight(
        execution, terms::AbstractArray{Float64, 3}, weights::ChebyshevTable,
    )
    !isempty(terms) && all(isfinite, terms) &&
        eltype(weights.coefficients) === Float64 &&
        eltype(weights.breakpoints) === Float64 ||
        throw(ArgumentError("power-weight expansions require finite binary64 data"))
    size(terms, 3) == size(weights.coefficients, 1) ||
        throw(DimensionMismatch("power-weight expansion terms and weights disagree"))
    return ObservationWeight(
        upload(execution, terms), _copy_power_weight_table(weights), size(terms)[1:2]
    )
end

function ObservationWeight(execution, terms::AbstractArray, weights::ChebyshevTable)
    throw(ArgumentError("power-weight expansion terms require a real binary64 3-D array"))
end

function _copy_power_weight_table(table::ChebyshevTable)
    return ChebyshevTable(
        copy(table.breakpoints), copy(table.coefficients), table.lower, table.upper
    )
end

_power_weight_alias(buffer, ::Nothing) = false
_power_weight_alias(buffer, values::AbstractArray) = Base.mightalias(buffer, values)
_power_weight_alias(buffer, table::ChebyshevTable) =
    Base.mightalias(buffer, table.coefficients) ||
    Base.mightalias(buffer, table.breakpoints)

"""
    evaluate_weight!(output, weight, z, execution; work)

Evaluate real modal power weights at the actual observation or export position.

`output` is a caller-owned binary64 matrix of the prepared `(n_omega,n_modes)` shape.
`weight` is an `ObservationWeight`; `z` is finite in m and lies inside every supplied
table's admitted span. `work` is a caller-owned binary64 vector with one value per
expansion term, or zero length for a constant or direct table. All writable storage
must be disjoint from retained input arrays and from each other. Invalid shape,
precision or aliasing is refused before the first write. A nonpositive or nonfinite
evaluated result raises `ArgumentError` instead of being used in an amplitude export.

Return `nothing`. The direct table uses the shared Chebyshev evaluator; an expansion
first evaluates its scalar weights, then forms each explicit term sum. Warmed CPU
calls allocate nothing. This leaf contains no source-call accounting or physics model.
"""
function evaluate_weight!(output, weight::ObservationWeight, z, execution; work)
    output isa AbstractMatrix{Float64} && work isa AbstractVector{Float64} ||
        throw(ArgumentError("modal power-weight output and work require binary64 storage"))
    size(output) == weight.shape ||
        throw(DimensionMismatch("power-weight output does not match its modal shape"))
    n_terms = weight.weights === nothing ? 0 : size(weight.weights.coefficients, 1)
    length(work) == n_terms ||
        throw(DimensionMismatch("power-weight expansion scratch has the wrong length"))
    isfinite(z) || throw(ArgumentError("power-weight position must be finite"))
    (
        Base.mightalias(output, work) || _power_weight_alias(output, weight.data) ||
            _power_weight_alias(output, weight.weights) ||
            _power_weight_alias(work, weight.data) ||
            _power_weight_alias(work, weight.weights)
    ) &&
        throw(ArgumentError("power-weight writable storage aliases prepared input"))
    if weight.weights !== nothing
        evaluate_table!(work, weight.weights, z, execution)
        terms = weight.data
        each!(execution, size(output)...) do row, column
            total = 0.0
            for term in axes(terms, 3)
                total += work[term] * terms[row, column, term]
            end
            output[row, column] = total
        end
    elseif weight.data isa ChebyshevTable
        evaluate_table!(output, weight.data, z, execution)
    else
        copyto!(output, weight.data)
    end
    all(value -> isfinite(value) && value > 0, output) ||
        throw(ArgumentError("evaluated modal power weights must be positive finite"))
    return nothing
end
