# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public ObservationWeight, evaluate_weight!

"""
    ObservationWeight(execution, values::Matrix{Float64})
    ObservationWeight(table::ChebyshevTable, shape::Tuple{Int,Int})
    ObservationWeight(execution, table::ChebyshevTable, shape::Tuple{Int,Int})
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
        execution::KernelExecution, table::ChebyshevTable, shape::Tuple{Int, Int},
    )
    eltype(table.coefficients) === Float64 &&
        eltype(table.breakpoints) === Float64 && all(>(0), shape) ||
        throw(ArgumentError("power-weight tables require real binary64 values"))
    prod(shape) == size(table.coefficients, 1) ||
        throw(DimensionMismatch("power-weight table does not match its modal shape"))
    return ObservationWeight(_copy_power_weight_table(execution, table), nothing, shape)
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
        upload(execution, terms), _copy_power_weight_table(execution, weights),
        size(terms)[1:2],
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

function _copy_power_weight_table(execution, table::ChebyshevTable)
    oncpu(execution) && return _copy_power_weight_table(table)
    return ChebyshevTable(
        upload(execution, table.breakpoints), upload(execution, table.coefficients),
        table.lower, table.upper,
    )
end

_power_weight_alias(buffer, ::Nothing) = false
_power_weight_alias(buffer, values::AbstractArray) = Base.mightalias(buffer, values)
_power_weight_alias(buffer, table::ChebyshevTable) =
    Base.mightalias(buffer, table.coefficients) ||
    Base.mightalias(buffer, table.breakpoints)

"""
    evaluate_weight!(output, weight, z, execution; work, status=nothing)

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
Device callers supply an owned `DeviceReadback` of Int shape `(2,n_modes)` in `status`.
Only those per-column validity/identity facts are read back; numerical values and
expansion scratch remain on the selected device.
"""
function evaluate_weight!(
        output, weight::ObservationWeight, z, execution; work, status = nothing,
    )
    output isa AbstractMatrix{Float64} && work isa AbstractVector{Float64} ||
        throw(ArgumentError("modal power-weight output and work require binary64 storage"))
    size(output) == weight.shape ||
        throw(DimensionMismatch("power-weight output does not match its modal shape"))
    n_terms = weight.weights === nothing ? 0 : size(weight.weights.coefficients, 1)
    length(work) == n_terms ||
        throw(DimensionMismatch("power-weight expansion scratch has the wrong length"))
    isfinite(z) || throw(ArgumentError("power-weight position must be finite"))
    if !oncpu(execution)
        _weight_summary_admit(status, output, execution)
        _transfer_writable_destination(output)
        _transfer_writable_destination(work)
        (
            _transfer_arrays_alias(output, work) || _retained_alias(output, weight) ||
                _retained_alias(work, weight) || _retained_alias(work, status) ||
                _retained_alias(status.device, weight) ||
                _retained_alias(status.host, weight)
        ) &&
            throw(ArgumentError("power-weight writable storage aliases prepared input"))
    end
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
        if oncpu(execution)
            copyto!(output, weight.data)
        else
            copy_device!(output, weight.data, execution)
        end
    end
    if oncpu(execution)
        all(value -> isfinite(value) && value > 0, output) ||
            throw(ArgumentError("evaluated modal power weights must be positive finite"))
    else
        _weight_summary!(status, output, execution)
    end
    return nothing
end

function _weight_summary_admit(status, values, execution)
    status isa DeviceReadback && status.execution === execution ||
        throw(ArgumentError("device power weights require their owned status readback"))
    eltype(status.device) === Int && eltype(status.host) === Int &&
        size(status.device) == size(status.host) == (2, size(values, 2)) ||
        throw(DimensionMismatch("power-weight status needs two Int rows per column"))
    _retained_alias(values, status) &&
        throw(ArgumentError("power-weight values must not alias status storage"))
    return nothing
end

function _weight_summary!(status, values, execution)
    _weight_summary_admit(status, values, execution)
    parts = status.device
    rows, columns = size(values)
    each_column!(execution, columns) do column
        invalid, nonunit = 0, 0
        for row in 1:rows
            value = values[row, column]
            if !(isfinite(value) && value > 0) && invalid == 0
                invalid = row
            end
            nonunit |= !isone(value)
        end
        parts[1, column], parts[2, column] = invalid, nonunit
    end
    host = readback!(status, execution)
    identity = true
    for column in 1:columns
        iszero(host[1, column]) ||
            throw(ArgumentError("evaluated modal power weights must be positive finite"))
        identity &= iszero(host[2, column])
    end
    return identity
end

public NormalizedObservationWeight, ObservationWeightWorkspace

"""
    NormalizedObservationWeight(execution, weight, arithmetic::SourceArithmetic{Float32})

Own a host-wide power-weight representation with normalized binary32 output.

`weight` is an `ObservationWeight`. Constant values are narrowed once with an explicit
power-of-two exponent. A varying direct table or expansion retains its immutable
binary64 representation: every element is evaluated in the original wide scalar
order, normalized, then rounded once. This is SEP 33's host coefficient boundary,
not binary32 Clenshaw evaluation. No state-sized mutable binary64 buffer is allocated.
The initial implementation requires CPU execution; it makes no device-wide claim.
Nonpositive, nonfinite or unrepresentable normalized weights raise `ArgumentError`.
"""
struct NormalizedObservationWeight{W, A, R}
    original::W
    constant::A
    range::R
    shape::Tuple{Int, Int}
    identity::Bool
end

function NormalizedObservationWeight(
        execution, weight::ObservationWeight, arithmetic::SourceArithmetic{Float32},
    )
    oncpu(execution) ||
        throw(ArgumentError("normalized power weights require host-wide CPU evaluation"))
    if weight.weights === nothing && weight.data isa Matrix{Float64}
        data, fact = source_narrow(
            execution, weight.data, arithmetic, :observation_power_weight; normalize = true,
        )
        all(>(0.0f0), data) ||
            throw(ArgumentError("normalized power weights underflow binary32 positivity"))
        return NormalizedObservationWeight(
            nothing, data, fact, weight.shape, all(isone, weight.data)
        )
    end
    original = if weight.weights === nothing
        ObservationWeight(weight.data, weight.shape)
    else
        ObservationWeight(execution, weight.data, weight.weights)
    end
    return NormalizedObservationWeight(original, nothing, nothing, weight.shape, false)
end

"""
    ObservationWeightWorkspace(weight::NormalizedObservationWeight)

Allocate only the small host expansion-coefficient vector and scalar range facts.

`host` has one binary64 value per represented expansion term, or zero length for a
constant/direct table. `range` and `identity` describe the last successful evaluation.
The caller owns a separate binary32 matrix for normalized output. The physical value
is that matrix multiplied by `2^range.block_exponent`; this exponent must remain
explicit through subsequent pointwise products and final host restoration.
"""
mutable struct ObservationWeightWorkspace{V}
    host::V
    range::SourceRange
    identity::Bool
end
function ObservationWeightWorkspace(weight::NormalizedObservationWeight)
    original = weight.original
    count = original === nothing || original.weights === nothing ? 0 :
        size(original.weights.coefficients, 1)
    return ObservationWeightWorkspace(
        zeros(count), SourceRange(:observation_power_weight, 0, 0, true, 0.0, 0.0), false,
    )
end

function _normalized_weight_value(weight, index, z, coefficients)
    if weight.weights === nothing
        table = weight.data
        return _table_value(table.coefficients, table.breakpoints, index, z)
    end
    terms = weight.data
    row = mod(index - 1, weight.shape[1]) + 1
    column = div(index - 1, weight.shape[1]) + 1
    total = 0.0
    for term in axes(terms, 3)
        total += coefficients[term] * terms[row, column, term]
    end
    return total
end

"""
    evaluate_weight!(output, weight::NormalizedObservationWeight, z, execution; work)

Evaluate wide host coefficients and publish one normalized binary32 weight matrix.

`output` is a disjoint writable binary32 matrix with `weight.shape`. `work` is an
`ObservationWeightWorkspace`. Before the first output write, a scalar scan verifies
positive finite wide values and normalized binary32 representability. A second scan
rounds each wide value once into output. No state-sized binary64 scratch is retained.
Return the explicit block exponent and update `work.range`/`work.identity` only after
success. Exact identity classification uses the wide represented values, not rounded
values. Invalid position, storage, overlap or range raises before output mutation.
Warmed CPU calls allocate nothing. The original binary64 overload is unchanged.
"""
function evaluate_weight!(
        output, weight::NormalizedObservationWeight, z, execution;
        work::ObservationWeightWorkspace,
    )
    oncpu(execution) && output isa AbstractMatrix{Float32} ||
        throw(ArgumentError("normalized power weights require CPU binary32 output"))
    size(output) == weight.shape ||
        throw(DimensionMismatch("power-weight output does not match its modal shape"))
    original = weight.original
    count = original === nothing || original.weights === nothing ? 0 :
        size(original.weights.coefficients, 1)
    work.host isa StridedVector{Float64} ||
        throw(ArgumentError("power-weight host coefficients require binary64 storage"))
    length(work.host) == count ||
        throw(DimensionMismatch("power-weight expansion scratch has the wrong length"))
    isfinite(z) || throw(ArgumentError("power-weight position must be finite"))
    _transfer_writable_destination(output)
    (
        _retained_alias(output, weight) || _retained_alias(work.host, weight) ||
            _transfer_arrays_alias(output, work.host)
    ) && throw(ArgumentError("power-weight writable storage aliases prepared input"))
    if original === nothing
        copyto!(output, weight.constant)
        work.range, work.identity = weight.range, weight.identity
        return weight.range.block_exponent
    end
    if original.weights === nothing
        _check_table_position(original.data, z)
    else
        evaluate_table!(work.host, original.weights, z, execution)
    end
    largest, smallest, identity = 0.0, Inf, true
    for index in 1:length(output)
        value = _normalized_weight_value(original, index, z, work.host)
        isfinite(value) && value > 0 ||
            throw(ArgumentError("evaluated modal power weights must be positive finite"))
        largest, smallest = max(largest, value), min(smallest, value)
        identity &= isone(value)
    end
    exponent = last(frexp(largest)) - 1
    normalized_smallest = ldexp(smallest, -exponent)
    Float32(normalized_smallest) > 0 ||
        throw(ArgumentError("normalized power weights underflow binary32 positivity"))
    fact = SourceRange(
        :observation_power_weight, exponent, 0, true,
        ldexp(largest, -exponent), normalized_smallest,
    )
    for index in 1:length(output)
        value = _normalized_weight_value(original, index, z, work.host)
        output[index] = Float32(ldexp(value, -exponent))
    end
    work.range, work.identity = fact, identity
    return exponent
end
