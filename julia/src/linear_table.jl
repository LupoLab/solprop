# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

mutable struct _LinearFlowTable{A}
    factors::A
    origins::Vector{Float64}
    destinations::Vector{Float64}
    valid::Vector{Bool}
    constant::Bool
    replaced::Int
    builds::Int
    hits::Int
end

_linear_constant(::DiagonalOperator) = true
_linear_constant(::SeparableSpatialOperator) = true
_linear_constant(::Union{RepresentedOperator, ExpansionOperator}) = false

function _linear_flow_table(operator, state)
    constant = _linear_constant(operator)
    count = constant ? 1 : 2
    return _LinearFlowTable(
        ntuple(unused -> similar(state), count), zeros(count), zeros(count),
        fill(false, count), constant, 0, 0, 0,
    )
end

_linear_position(::DiagonalOperator, position) = nothing
_linear_position(operator::RepresentedOperator, position) =
    _check_table_position(operator.cumulative, position)
_linear_position(operator::ExpansionOperator, position) =
    _check_table_position(operator.cumulative_weights, position)
_linear_position(operator::SeparableSpatialOperator, position) =
    _spatial_operator_position(operator, position)

function _linear_table_interval(operator, origin, destination)
    isfinite(origin) && isfinite(destination) && isfinite(destination - origin) ||
        throw(ArgumentError("linear flow table endpoints and distance must be finite"))
    # A cache hit must not bypass the represented or spatial admitted domain.
    _linear_position(operator, origin)
    _linear_position(operator, destination)
    return nothing
end

function _linear_table_slot(table, origin, destination)
    start = table.constant ? 0.0 : origin
    finish = table.constant ? destination - origin : destination
    for slot in eachindex(table.factors)
        if table.valid[slot] && table.origins[slot] == start &&
                table.destinations[slot] == finish
            table.hits += 1
            return slot, true
        end
    end
    slot = mod1(table.replaced + 1, length(table.factors))
    table.valid[slot] = false
    table.origins[slot], table.destinations[slot] = start, finish
    table.replaced = slot
    return slot, false
end

function _tabulated_flow!(flow, output, input, origin, destination)
    _linear_table_interval(flow.operator, origin, destination)
    if origin == destination
        copyto!(output, input)
        return nothing
    end
    table = flow.table
    slot, valid = _linear_table_slot(table, origin, destination)
    factor = table.factors[slot]
    unit = one(eltype(factor))
    if !valid
        operator_exponent!(
            flow.exponent, flow.operator, origin, destination, flow.execution;
            work = flow.work,
        )
        _source_finite!(flow.failures, flow.exponent, flow.execution, :exponent)
        exponent = flow.exponent
        each!(flow.execution, size(factor)...) do row, column
            phase = exponent[row, column]
            factor[row, column] = iszero(phase) ? unit : exp(phase)
        end
        _source_finite!(flow.failures, factor, flow.execution, :propagator)
        table.valid[slot] = true
        table.builds += 1
    end
    each!(flow.execution, size(output)...) do row, column
        multiplier = factor[row, column]
        output[row, column] = multiplier == unit ? input[row, column] :
            multiplier * input[row, column]
    end
    return nothing
end
