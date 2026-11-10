# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public RK4IPTableau, LowStorageWork, LowStorageFacts, low_storage_trial!

"""
    RK4IPTableau()

Select classical RK4 in the midpoint interaction picture, the fixed SEP 24 scheme.

The four source positions are `(0,1/2,1/2,1)` within a step, with final weights
`(1/6,1/3,1/3,1/6)`. This marker admits no arbitrary embedded tableau. Diagonal
linear flows commute, so the fourth stage's inverse flow cancels the final forward
flow: only four forward applications remain. The register derivation is SEP 24
sections 4.2–4.3, from Lawson (1967, DOI 10.1137/0704033) and Hult (2007,
DOI 10.1109/JLT.2007.909373). There is no embedded error estimate or recycled slope.
"""
struct RK4IPTableau end

"""
    LowStorageWork(execution, state::AbstractMatrix{ComplexF64}, tableau::RK4IPTableau;
                   norm_mask=fill(true,size(state,1)))

Prepare three owned binary64 work matrices for one in-place replay step.

The caller's state is register A. `accumulator` B and `stage` C complete the three
register scheme; `source` S is one reusable output of the complete source pipeline.
No previous state, four-row slope table or error-estimate field is retained. All
matrices have shape `(n_omega,n_columns)`. The owned unpacked `norm_mask` selects
spectral rows, and per-column protected reduction scratch is only O(n_columns).
Flow/source callables own and account for their separate prepared scratch.
"""
struct LowStorageWork{A, M, R, F}
    accumulator::A
    stage::A
    source::A
    norm_mask::M
    norm_parts::R
    finite_columns::F
end

function LowStorageWork(
        execution::KernelExecution, state::AbstractMatrix{ComplexF64}, ::RK4IPTableau;
        norm_mask::AbstractVector{Bool} = fill(true, size(state, 1)),
    )
    oncpu(execution) ||
        throw(ArgumentError("low-storage preparation currently requires CPU"))
    isempty(state) && throw(ArgumentError("low-storage state must be nonempty"))
    length(norm_mask) == size(state, 1) ||
        throw(DimensionMismatch("low-storage norm mask must match the spectral rows"))
    rows, columns = size(state)
    return LowStorageWork(
        dzeros(execution, ComplexF64, rows, columns),
        dzeros(execution, ComplexF64, rows, columns),
        dzeros(execution, ComplexF64, rows, columns), upload(execution, norm_mask),
        dzeros(execution, Float64, 2, columns),
        upload(execution, fill(false, columns)),
    )
end

"""
    LowStorageFacts

Hold the numerical facts returned by a successfully completed RK4IP leaf.

`state_norm` is the protected magnitude over the norm mask, in state units, and
`finite` reports every stored state entry, including outside that mask. `source_calls`
is four and `norm_calls` is one; failures return no success facts. `error_norm=0.0`
means "not estimated", not a measured zero error. `stiffness=NaN` records the missing
optional diagnostic, as permitted for the existing resident native engine. No source
history volume or reconstructed small slope differences are used to manufacture it.
"""
struct LowStorageFacts
    state_norm::Float64
    finite::Bool
    source_calls::Int
    norm_calls::Int
    error_norm::Float64
    stiffness::Float64
end

@inline function _low_storage_callback(result)
    result === nothing ||
        throw(ArgumentError("low-storage numerical callbacks return nothing"))
    return nothing
end

function _low_storage_first!(work, state, step, execution)
    accumulator, stage, source = work.accumulator, work.stage, work.source
    sixth, half = step / 6, 0.5step
    each!(execution, size(state)...) do row, column
        anchor, slope = state[row, column], source[row, column]
        accumulator[row, column] = anchor + sixth * slope
        stage[row, column] = anchor + half * slope
    end
    return nothing
end

function _low_storage_middle!(work, state, step, stage_weight, execution)
    accumulator, stage, source = work.accumulator, work.stage, work.source
    third = step / 3
    each!(execution, size(state)...) do row, column
        slope = source[row, column]
        accumulator[row, column] += third * slope
        stage[row, column] = state[row, column] + stage_weight * slope
    end
    return nothing
end

function _low_storage_final!(work, state, step, execution)
    source = work.source
    sixth = step / 6
    each!(execution, size(state)...) do row, column
        state[row, column] += sixth * source[row, column]
    end
    return nothing
end

function _low_storage_norm!(work, state, execution)
    parts, mask, finite_columns = work.norm_parts, work.norm_mask, work.finite_columns
    rows, columns = size(state)
    each_column!(execution, columns) do column
        column_scale, column_sum = 0.0, 0.0
        column_finite = true
        for row in 1:rows
            value = state[row, column]
            column_finite &= isfinite(value)
            if mask[row]
                column_scale, column_sum = _rk_sumsq(column_scale, column_sum, real(value))
                column_scale, column_sum = _rk_sumsq(column_scale, column_sum, imag(value))
            end
        end
        parts[1, column], parts[2, column] = column_scale, column_sum
        finite_columns[column] = column_finite
    end
    scale, total = 0.0, 0.0
    finite = true
    for column in 1:columns
        scale, total = _rk_merge_sumsq(scale, total, parts[1, column], parts[2, column])
        finite &= finite_columns[column]
    end
    return Float64(scale * sqrt(total)), finite
end

"""
    low_storage_trial!(work, state, position, step, tableau, source!, flow!, execution)

Overwrite the caller's state with one fixed-schedule RK4IP step in three registers.

Positions and the finite positive step are metres. `source!(out,input,z)` is the
complete Julia source and `flow!(out,input,z0,z1)` applies the forward linear flow.
Both return `nothing` or throw; flow must support identical input/output buffers.
There are four source evaluations, at start/midpoint/midpoint/end, and four forward
flow applications. No inverse flow, embedded estimate or source reuse is introduced.

Return `LowStorageFacts` after one protected CPU norm reduction. Source callbacks
record actual attempted invocations before work begins. A callback failure returns
no facts and may leave the state overwritten by an intermediate interaction-picture
value: the session must mark the step in flight before entry and poison on failure.
The master owns schedules, source-absence shortcuts, flow-table caching, acceptance,
mask/collar application, statistics and position commits. This leaf applies none of
those policies. Preparation currently admits CPU binary64 only.
"""
function low_storage_trial!(
        work::LowStorageWork, state::AbstractMatrix{ComplexF64}, position::Float64,
        step::Float64, ::RK4IPTableau, source!::S, flow!::F, execution::KernelExecution,
    ) where {S, F}
    oncpu(execution) ||
        throw(ArgumentError("low-storage device reduction is not admitted"))
    size(state) == size(work.accumulator) ||
        throw(DimensionMismatch("low-storage state does not match its workspace"))
    if Base.mightalias(state, work.accumulator) || Base.mightalias(state, work.stage) ||
            Base.mightalias(state, work.source)
        throw(ArgumentError("low-storage state must be disjoint from its work registers"))
    end
    isfinite(position) && isfinite(step) && step > 0 && isfinite(position + step) &&
        position + step > position ||
        throw(ArgumentError("invalid low-storage trial interval"))
    midpoint, endpoint = position + 0.5step, position + step
    source, stage, accumulator = work.source, work.stage, work.accumulator

    _low_storage_callback(source!(source, state, position))
    _low_storage_callback(flow!(state, state, position, midpoint))
    _low_storage_callback(flow!(source, source, position, midpoint))
    _low_storage_first!(work, state, step, execution)

    _low_storage_callback(source!(source, stage, midpoint))
    _low_storage_middle!(work, state, step, 0.5step, execution)

    _low_storage_callback(source!(source, stage, midpoint))
    _low_storage_middle!(work, state, step, step, execution)
    _low_storage_callback(flow!(stage, stage, midpoint, endpoint))

    _low_storage_callback(source!(source, stage, endpoint))
    _low_storage_callback(flow!(state, accumulator, midpoint, endpoint))
    _low_storage_final!(work, state, step, execution)
    magnitude, finite = _low_storage_norm!(work, state, execution)
    return LowStorageFacts(magnitude, finite, 4, 1, 0.0, NaN)
end
