# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public ChebyshevTable, evaluate_table!, difference_table!, derivative_table!

"""
    ChebyshevTable(execution, breakpoints, coefficients;
                   lower = first(breakpoints), upper = last(breakpoints))

Copy one translated piecewise Chebyshev profile into owned numerical storage.

`breakpoints` is a strictly increasing finite vector in metres. Coefficients have
shape `(n_values, n_coefficients, n_intervals)`, lowest degree first. This is the
column-major counterpart of the translated packed `(interval, degree, value)` array;
values keep their supplied units and ordering. Each physical profile owns a distinct
table. No stage count or physical interpretation is encoded here.

`lower` and `upper` explicitly carry the translated admissible span, including any
edge slack. Defaults admit the breakpoint span exactly. Interior edges belong to the
interval starting there; the final edge belongs to the last interval. Points in the
admitted slack extrapolate the endpoint interval, matching the translated contract.

# Throws
- `ArgumentError`: For empty, nonfinite or unsupported values or invalid span bounds.
- `DimensionMismatch`: When the interval count disagrees with the breakpoints.
"""
struct ChebyshevTable{E, A, T}
    breakpoints::E
    coefficients::A
    lower::T
    upper::T
end

function ChebyshevTable(
        execution::KernelExecution, breakpoints::AbstractVector{T},
        coefficients::AbstractArray{C, 3};
        lower::Union{Nothing, Real} = nothing, upper::Union{Nothing, Real} = nothing,
    ) where {
        T <: Union{Float32, Float64},
        C <: Union{Float32, Float64, ComplexF32, ComplexF64},
    }
    length(breakpoints) >= 2 ||
        throw(ArgumentError("a represented profile needs at least two breakpoints"))
    all(isfinite, breakpoints) && all(isfinite, coefficients) ||
        throw(ArgumentError("represented profile data must be finite"))
    all(>(zero(T)), diff(breakpoints)) ||
        throw(ArgumentError("profile breakpoints must increase strictly"))
    size(coefficients, 1) > 0 && size(coefficients, 2) > 0 ||
        throw(ArgumentError("a represented profile must have values and coefficients"))
    size(coefficients, 3) == length(breakpoints) - 1 ||
        throw(DimensionMismatch("profile coefficients and intervals disagree"))
    low = lower === nothing ? first(breakpoints) : T(lower)
    high = upper === nothing ? last(breakpoints) : T(upper)
    isfinite(low) && isfinite(high) && low <= first(breakpoints) &&
        last(breakpoints) <= high ||
        throw(ArgumentError("the admitted profile span must contain its breakpoints"))
    return ChebyshevTable(
        upload(execution, breakpoints), upload(execution, coefficients), low, high,
    )
end

function ChebyshevTable(execution::KernelExecution, breakpoints, coefficients; kwargs...)
    throw(ArgumentError("represented profiles require binary32 or binary64 numerical data"))
end

@inline function _check_table_position(table::ChebyshevTable, position::Real)
    isfinite(position) && table.lower <= position <= table.upper ||
        throw(ArgumentError("position lies outside the admitted represented profile span"))
    return nothing
end

@inline function _table_interval(edges, position, before::Bool = false)
    low, high = 1, length(edges) + 1
    while low < high
        middle = low + div(high - low, 2)
        if middle <= length(edges) &&
                (before ? edges[middle] < position : edges[middle] <= position)
            low = middle + 1
        else
            high = middle
        end
    end
    return clamp(low - 1, 1, length(edges) - 1)
end

@inline function _table_derivative(coefficients, edges, value_index, position, before)
    interval = _table_interval(edges, position, before)
    lower, upper = edges[interval], edges[interval + 1]
    coordinate = (2 * position - lower - upper) / (upper - lower)
    count = size(coefficients, 2)
    count == 1 && return zero(eltype(coefficients))
    count == 2 && return coefficients[value_index, 2, interval] * (2 / (upper - lower))
    previous = coefficients[value_index, count - 1, interval]
    current = coefficients[value_index, count, interval]
    dprevious = zero(previous)
    dcurrent = zero(current)
    twice_coordinate = 2 * coordinate
    # Differentiate the same represented Clenshaw polynomial, not a separately
    # sampled physical rate. The exact knot coordinate stays unchanged on either side.
    for index in (count - 2):-1:1
        saved, dsaved = previous, dprevious
        previous = coefficients[value_index, index, interval] - current
        dprevious = -dcurrent
        dcurrent = dsaved + dcurrent * twice_coordinate + 2 * current
        current = saved + current * twice_coordinate
    end
    return (dprevious + dcurrent * coordinate + current) * (2 / (upper - lower))
end

"""
    derivative_table!(output, table::ChebyshevTable, position, execution; before=false)

Differentiate the represented polynomial into a caller-owned array of `n_values`.

`position` is in metres within the admitted span; derivatives have profile units per
metre. At an interior breakpoint `before=true` selects the interval ending there,
otherwise the interval starting there. Neither policy perturbs the evaluation point.
The endpoint intervals also cover admitted slack. This differentiates the same
cumulative representation used by `difference_table!`, as required by SEP 16's
transported compensation. Return `output` after synchronous completion.
"""
function derivative_table!(
        output, table::ChebyshevTable, position::Real, execution::KernelExecution;
        before::Bool = false,
    )
    length(output) == size(table.coefficients, 1) ||
        throw(DimensionMismatch("profile derivative has the wrong value count"))
    Base.mightalias(output, table.coefficients) &&
        throw(ArgumentError("profile outputs must be disjoint from their coefficients"))
    _check_table_position(table, position)
    coefficients, edges = table.coefficients, table.breakpoints
    each!(execution, length(output), 1) do value_index, unused
        output[value_index] =
            _table_derivative(coefficients, edges, value_index, position, before)
    end
    return output
end

@inline function _table_value(coefficients, edges, value_index, position)
    interval = _table_interval(edges, position)
    lower, upper = edges[interval], edges[interval + 1]
    coordinate = (2 * position - lower - upper) / (upper - lower)
    n_coefficients = size(coefficients, 2)
    if n_coefficients == 1
        return coefficients[value_index, 1, interval]
    elseif n_coefficients == 2
        return coefficients[value_index, 1, interval] +
            coordinate * coefficients[value_index, 2, interval]
    end
    # The two-register Clenshaw recurrence is the one used by the translated
    # PiecewiseChebyshev/NumPy chebval contract, without reordering its updates.
    previous = coefficients[value_index, n_coefficients - 1, interval]
    current = coefficients[value_index, n_coefficients, interval]
    twice_coordinate = 2 * coordinate
    for index in (n_coefficients - 2):-1:1
        saved = previous
        previous = coefficients[value_index, index, interval] - current
        current = saved + current * twice_coordinate
    end
    return previous + current * coordinate
end

"""
    evaluate_table!(output, table::ChebyshevTable, position::Real, execution)
    evaluate_table!(output, table::ChebyshevTable, positions::AbstractVector, execution;
                    active_positions = length(positions))

Evaluate a represented profile into preallocated storage using Clenshaw recurrence.

A scalar position, in metres, fills any array with `n_values` elements, in its linear
column-major order. A position vector fills a matrix `(n_values, n_position_capacity)`;
only `active_positions` are read and every remaining output column is zeroed. The
whole admitted span is checked before launch. Return `output`.
"""
function evaluate_table!(
        output, table::ChebyshevTable, position::Real, execution::KernelExecution,
    )
    length(output) == size(table.coefficients, 1) ||
        throw(DimensionMismatch("profile output has the wrong value count"))
    Base.mightalias(output, table.coefficients) &&
        throw(ArgumentError("profile outputs must be disjoint from their coefficients"))
    _check_table_position(table, position)
    coefficients, edges = table.coefficients, table.breakpoints
    each!(execution, length(output), 1) do value_index, unused
        output[value_index] = _table_value(coefficients, edges, value_index, position)
    end
    return output
end

function evaluate_table!(
        output::AbstractMatrix, table::ChebyshevTable, positions::AbstractVector,
        execution::KernelExecution; active_positions::Integer = length(positions),
    )
    size(output, 1) == size(table.coefficients, 1) ||
        throw(DimensionMismatch("profile output has the wrong value count"))
    0 <= active_positions <= min(length(positions), size(output, 2)) ||
        throw(DimensionMismatch("active profile positions exceed prepared storage"))
    if Base.mightalias(output, table.coefficients) || Base.mightalias(output, positions)
        throw(ArgumentError("profile output must not overlap coefficients or positions"))
    end
    for index in 1:active_positions
        _check_table_position(table, positions[index])
    end
    coefficients, edges = table.coefficients, table.breakpoints
    neutral = zero(eltype(output))
    each!(execution, size(output)...) do value_index, position_index
        if position_index <= active_positions
            output[value_index, position_index] =
                _table_value(coefficients, edges, value_index, positions[position_index])
        else
            output[value_index, position_index] = neutral
        end
    end
    return output
end

"""
    difference_table!(output, table::ChebyshevTable, z0, z1, execution)

Evaluate the represented endpoint difference `table(z1)-table(z0)` into caller storage.

`output` has `n_values` elements, in linear column-major order. Positions are metres
within the admitted span. This follows the cumulative-Phi oracle's two Clenshaw
evaluations and subtraction: close endpoints can lose relative accuracy against the
cumulative value scale. It is not a new local quadrature. Constant diagonal operators
use their separate exact-distance route. Equal endpoints write exact zeros.
"""
function difference_table!(
        output, table::ChebyshevTable, z0::Real, z1::Real, execution::KernelExecution,
    )
    length(output) == size(table.coefficients, 1) ||
        throw(DimensionMismatch("profile difference has the wrong value count"))
    Base.mightalias(output, table.coefficients) &&
        throw(ArgumentError("profile outputs must be disjoint from their coefficients"))
    _check_table_position(table, z0)
    _check_table_position(table, z1)
    coefficients, edges = table.coefficients, table.breakpoints
    neutral = zero(eltype(output))
    each!(execution, length(output), 1) do value_index, unused
        output[value_index] = z0 == z1 ? neutral :
            _table_value(coefficients, edges, value_index, z1) -
            _table_value(coefficients, edges, value_index, z0)
    end
    return output
end
