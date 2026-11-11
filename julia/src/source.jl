# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public NonfiniteSourceError, LinearFlow

"""
    NonfiniteSourceError(stage, row, column)

Report a nonfinite numerical source value at one owned matrix location.

`stage` identifies the operation; `row` and `column` are one-based indices. Source
failures occur outside leaf kernels. A separate trial leaves the accepted state intact;
an in-place integrator must poison a state it already overwrote before this failure.
"""
struct NonfiniteSourceError <: Exception
    stage::Symbol
    row::Int
    column::Int
end

function Base.showerror(io::IO, error::NonfiniteSourceError)
    return print(
        io, "nonfinite ", error.stage, " at row ", error.row, ", column ", error.column
    )
end

function _source_failure_rows!(failures, array, execution)
    rows, columns = size(array)
    length(failures) >= columns ||
        throw(DimensionMismatch("source check buffer is too small"))
    each_column!(execution, columns) do column
        failed_row = 0
        for row in 1:rows
            if !isfinite(array[row, column])
                failed_row = row
                break
            end
        end
        failures[column] = failed_row
    end
    return nothing
end

function _source_check_rows(failures, columns, stage)
    for column in 1:columns
        if failures[column] != 0
            throw(NonfiniteSourceError(stage, failures[column], column))
        end
    end
    return nothing
end

function _source_finite!(failures, array, execution, stage)
    _source_failure_rows!(failures, array, execution)
    return _source_check_rows(failures, size(array, 2), stage)
end

function _source_finite!(failures::DeviceReadback, array, execution, stage)
    failures.execution === execution ||
        throw(ArgumentError("source failure scratch belongs to another execution"))
    _source_failure_rows!(failures.device, array, execution)
    return _source_check_rows(readback!(failures, execution), size(array, 2), stage)
end

"""
    LinearFlow(execution, operator, state; work=nothing, tabulate=false)

Prepare a callable linear flow with its own exponent and finiteness scratch.

`operator` is a prepared diagonal generator or cumulative representation; `state` sets
the binary32 or binary64 `(n_omega,n_columns)` shape. Exponents are assembled and
evaluated in binary64, then each propagator rounds once to the state precision before
multiplication. Expansion operators need their caller-owned
weight work. Calling `flow(out,input,z0,z1)` applies the represented flow and returns
`nothing`, checking numerical failures outside leaf kernels. Input and output must be
disjoint or the exact same array; partial overlap refuses. The CPU implementation
allocates nothing after warmup. Both caller arrays must retain the preparation's
state element type, including when the table is disabled.
Device binary64 work owns scratch in its selected context and reads back only
per-column failure indices. Device binary32 flow remains unqualified.

`tabulate=true` retains one factor matrix for a constant generator, or two for a
represented varying generator. Keys are exact interval lengths for constant data
and exact endpoint pairs otherwise. Tables are session-owned and regenerated after
eviction; neither interpolation nor a phase/step tolerance changes their values.
"""
struct LinearFlow{E, O, A, W, F, C, T}
    execution::E
    operator::O
    exponent::A
    work::W
    failures::F
    table::C
    state_type::Type{T}
end

function LinearFlow(
        execution::KernelExecution, operator, state::AbstractMatrix{Complex{T}};
        work = nothing, tabulate::Bool = false,
    ) where {T <: Union{Float32, Float64}}
    if !oncpu(execution)
        T === Float64 ||
            throw(ArgumentError("device linear flow currently requires binary64 state"))
        # Validate live storage and the selected context before allocating scratch.
        copy_device!(state, state, execution)
    end
    exponent = oncpu(execution) ? similar(state, ComplexF64) :
        dzeros(execution, ComplexF64, size(state)...)
    failures = oncpu(execution) ? zeros(Int, size(state, 2)) :
        DeviceReadback(execution, Int, size(state, 2))
    return LinearFlow(
        execution, operator, exponent, work, failures,
        tabulate ? _linear_flow_table(operator, state, execution) : nothing, eltype(state),
    )
end

_flow_shape(flow::LinearFlow) = size(flow.exponent)
_inplace_flow(flow) = false
_inplace_flow(::LinearFlow) = true
_flow_start(flow, scheme, position, schedule) = nothing
_flow_arithmetic(flow, state, arithmetic) = nothing
function _flow_arithmetic(flow::LinearFlow, state, arithmetic)
    flow.state_type === eltype(state) ||
        throw(ArgumentError("march and prepared linear flow precision differ"))
    return nothing
end

# Ordinary wide-binary64 and generic callbacks retain their existing trial graph.
# A certified wide flow specializes these hooks without changing the leaf signature.
_prepare_trial_flow(flow, scheme, position, step, destination, replay_index) = flow
_check_trial_flow(flow, scheme, position, step, state, work) = nothing

function (flow::LinearFlow)(output, input, origin::Float64, destination::Float64)
    eltype(output) === eltype(input) === flow.state_type ||
        throw(ArgumentError("linear flow buffers must use their prepared state precision"))
    size(output) == size(input) == size(flow.exponent) ||
        throw(DimensionMismatch("linear flow buffers do not match their preparation"))
    _transfer_writable_destination(output)
    output !== input && _transfer_arrays_alias(output, input) &&
        throw(ArgumentError("the linear flow refuses partial input/output overlap"))
    (_retained_alias(output, flow) || _retained_alias(input, flow)) &&
        throw(ArgumentError("caller arrays must be disjoint from linear flow storage"))
    if flow.table === nothing
        operator_exponent!(
            flow.exponent, flow.operator, origin, destination, flow.execution;
            work = flow.work,
        )
        _source_finite!(flow.failures, flow.exponent, flow.execution, :exponent)
        propagate!(output, input, flow.exponent, flow.execution)
    else
        _tabulated_flow!(flow, output, input, origin, destination)
    end
    _source_finite!(flow.failures, output, flow.execution, :flow)
    return nothing
end
