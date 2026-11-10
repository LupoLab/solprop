# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public EmbeddedTableau, RungeKuttaWork, rk_trial!, rk_error_norm!, rk_commit!

"""
    EmbeddedTableau(nodes, matrix, solution_weights, error_weights)

Own the coefficients of an explicit embedded Runge–Kutta method.

All coefficients are dimensionless translated binary64 values. `nodes` and both weight
vectors have `n_stages` entries; `matrix` is a strictly lower triangular
`(n_stages, n_stages)` array. The first node is zero. The engine receives the selected
tableau from solprop and does not choose coefficients or infer a seven-stage method.

# Returns
A concrete, independently owned tableau. Every trial evaluates all stages, including
the first stage of a method that could reuse its previous last source.

# Throws
- `ArgumentError`: If coefficients are nonfinite or their extents/explicitness disagree.
"""
struct EmbeddedTableau{N}
    nodes::NTuple{N, Float64}
    matrix::NTuple{N, NTuple{N, Float64}}
    solution_weights::NTuple{N, Float64}
    error_weights::NTuple{N, Float64}

    function EmbeddedTableau(nodes, matrix, solution_weights, error_weights)
        count = length(nodes)
        if count < 1 || size(matrix) != (count, count) ||
                length(solution_weights) != count || length(error_weights) != count
            throw(ArgumentError("an embedded tableau needs one row and weight per stage"))
        end
        if !all(isfinite, nodes) || !all(isfinite, matrix) ||
                !all(isfinite, solution_weights) || !all(isfinite, error_weights)
            throw(ArgumentError("embedded tableau coefficients must be finite"))
        end
        if !iszero(first(nodes)) || any(
                !iszero(matrix[row, column]) for row in 1:count for column in row:count
            )
            throw(ArgumentError("an embedded tableau must be explicit and start at zero"))
        end
        owned = new{count}(
            ntuple(index -> Float64(nodes[index]), count),
            ntuple(row -> ntuple(column -> Float64(matrix[row, column]), count), count),
            ntuple(index -> Float64(solution_weights[index]), count),
            ntuple(index -> Float64(error_weights[index]), count),
        )
        finite_coefficients = all(isfinite, owned.nodes) &&
            all(row -> all(isfinite, row), owned.matrix) &&
            all(isfinite, owned.solution_weights) && all(isfinite, owned.error_weights)
        if !finite_coefficients
            throw(ArgumentError("embedded tableau coefficients must fit in binary64"))
        end
        return owned
    end
end

"""
    RungeKuttaWork(state, tableau)

Allocate the trial buffers of an interaction-picture embedded Runge–Kutta method.

`state` is the owned `(n_omega, n_columns)` complex spectral state in the internal
Fourier convention. The work retains no reference to it. Buffers have its element type
and array backend; `tableau` determines the stage extent. The caller owns the accepted
state and explicitly commits a successful trial. All buffers are reused by later trials.

# Returns
A workspace containing stage derivatives, lab/source/increment buffers, candidate,
defect and per-column norm scratch. Every entry read by a trial or norm is overwritten.
"""
struct RungeKuttaWork{A, S, R}
    stages::S
    combination::A
    lab::A
    source::A
    transported::A
    candidate::A
    defect::A
    norm_parts::R
end

function RungeKuttaWork(
        state::AbstractMatrix{Complex{T}}, tableau::EmbeddedTableau,
    ) where {T}
    stages = similar(state, size(state)..., length(tableau.nodes))
    return RungeKuttaWork(
        stages, similar(state), similar(state), similar(state), similar(state),
        similar(state), similar(state), similar(state, T, 4, size(state, 2)),
    )
end

"""
    rk_trial!(work, state, z, dz, tableau, source!, flow!, execution)

Form an interaction-picture trial and its embedded defect without changing `state`.

`z` and positive `dz` are in metres. `source!(out, lab, position)` writes the nonlinear
derivative at the physical stage position. `flow!(out, input, origin, destination)`
applies the represented linear flow between those positions, including its inverse.
Both are concrete Julia callables, use preallocated storage, and finish before return.
They return `nothing`; numerical failures must throw after their leaf kernels return.
The source wrapper records each invocation before starting it, so failures retain exact
partial work accounting even though an incomplete trial has no return count.
`execution` is the numerical kernel execution policy. All arrays have the workspace's
`(n_omega, n_columns)` shape. The method reanchors its interaction picture at `z`.

# Returns
The number of nonlinear source evaluations, exactly the tableau's stage count.
`work.candidate` and `work.defect` are in the lab frame at `z + dz`.

# Throws
- `ArgumentError`: If the step is not finite and positive or the workspace has another
  state shape or stage count. These checks occur outside the numerical kernels.
"""
function rk_trial!(
        work::RungeKuttaWork, state, z::Float64, dz::Float64,
        tableau::EmbeddedTableau, source!::F, flow!::L, execution,
    ) where {F, L}
    count = length(tableau.nodes)
    if !isfinite(z) || !isfinite(dz) || dz <= 0.0 || !isfinite(z + dz) || z + dz <= z
        throw(ArgumentError("a Runge–Kutta trial needs a finite, advancing positive step"))
    end
    if size(work.candidate) != size(state) || size(work.stages, 3) != count
        throw(ArgumentError("Runge–Kutta workspace and state/tableau extents disagree"))
    end
    for node in tableau.nodes
        if !isfinite(z + node * dz)
            throw(ArgumentError("a Runge–Kutta stage position must remain finite"))
        end
    end
    for stage in 1:count
        _rk_stage_input!(
            work.combination, state, work.stages, tableau, dz, stage, execution
        )
        position = z + tableau.nodes[stage] * dz
        _rk_flow!(flow!, work.lab, work.combination, z, position)
        if source!(work.source, work.lab, position) !== nothing
            throw(ArgumentError("a Runge–Kutta source returns nothing or throws a failure"))
        end
        _rk_flow!(flow!, work.transported, work.source, position, z)
        store, transported = work.stages, work.transported
        each!(execution, size(state)...) do row, column
            store[row, column, stage] = transported[row, column]
        end
    end
    _rk_weighted_sum!(
        work.combination, state, work.stages, tableau.solution_weights, dz, true, execution
    )
    _rk_flow!(flow!, work.candidate, work.combination, z, z + dz)
    _rk_weighted_sum!(
        work.combination, state, work.stages, tableau.error_weights, dz, false, execution
    )
    _rk_flow!(flow!, work.defect, work.combination, z, z + dz)
    return count
end

@inline function _rk_flow!(flow!::F, output, input, origin, destination) where {F}
    if flow!(output, input, origin, destination) !== nothing
        throw(ArgumentError("a Runge–Kutta flow must throw failures and return nothing"))
    end
    return nothing
end

function _rk_stage_input!(destination, state, stages, tableau, dz, stage, execution)
    coefficients = tableau.matrix
    each!(execution, size(state)...) do row, column
        increment = zero(eltype(state))
        for previous in 1:(stage - 1)
            increment += coefficients[stage][previous] * stages[row, column, previous]
        end
        destination[row, column] = state[row, column] + dz * increment
    end
    return nothing
end

function _rk_weighted_sum!(
        destination, state, stages, weights, dz, include_state, execution,
    )
    count = length(weights)
    each!(execution, size(state)...) do row, column
        # The defect is a cancellation-sensitive sum. Keep ordinary Julia arithmetic;
        # no contraction/reassociation macro may be added to this protected helper.
        increment = zero(eltype(state))
        for stage in 1:count
            increment += weights[stage] * stages[row, column, stage]
        end
        value = dz * increment
        destination[row, column] = include_state ? state[row, column] + value : value
    end
    return nothing
end

"""
    rk_error_norm!(work, state, mask, weights, rtol, atol, execution)

Score the trial defect with the translated global weak or weighted error norm.

`mask` and positive `weights` have one entry per spectral row. Both state magnitudes
use every masked column. The scale is `rtol * weights[row] * max(norm(state),
norm(candidate)) + atol`, following solprop's controller contract; it is never a
per-bin state magnitude. Tolerances are respectively dimensionless and in state units.
This initial reduction implementation admits CPU execution; device reductions are
qualified separately. Preparation validates mask, weight and tolerance extents/ranges.

# Returns
The nonnegative scalar norm, or a nonfinite value for a trial the controller must reject.
An exactly zero numerator with exactly zero scale contributes exactly zero.
Any nonfinite entry in the old or candidate state refuses the trial, including entries
outside the norm mask. Scaled sums of squares avoid squaring physical amplitudes.

# Throws
- `ArgumentError`: If the execution policy is not a CPU policy.
"""
function rk_error_norm!(work::RungeKuttaWork, state, mask, weights, rtol, atol, execution)
    if !oncpu(execution)
        throw(ArgumentError("Runge–Kutta norm device reduction is not admitted"))
    end
    parts, candidate, defect = work.norm_parts, work.candidate, work.defect
    rows, columns = size(state)
    each_column!(execution, columns) do column
        column_old_scale, column_old_sum = zero(eltype(parts)), zero(eltype(parts))
        column_new_scale, column_new_sum = column_old_scale, column_old_sum
        for row in 1:rows
            if !isfinite(state[row, column]) || !isfinite(candidate[row, column])
                column_old_scale, column_old_sum = (
                    oftype(column_old_scale, Inf), one(column_old_sum),
                )
                column_new_scale, column_new_sum = (
                    oftype(column_new_scale, Inf), one(column_new_sum),
                )
            elseif mask[row]
                column_old_scale, column_old_sum = _rk_sumsq(
                    column_old_scale, column_old_sum, real(state[row, column]),
                )
                column_old_scale, column_old_sum = _rk_sumsq(
                    column_old_scale, column_old_sum, imag(state[row, column]),
                )
                column_new_scale, column_new_sum = _rk_sumsq(
                    column_new_scale, column_new_sum, real(candidate[row, column]),
                )
                column_new_scale, column_new_sum = _rk_sumsq(
                    column_new_scale, column_new_sum, imag(candidate[row, column]),
                )
            end
        end
        parts[1, column], parts[2, column] = column_old_scale, column_old_sum
        parts[3, column], parts[4, column] = column_new_scale, column_new_sum
    end
    old_scale, old_sum = zero(eltype(parts)), zero(eltype(parts))
    new_scale, new_sum = old_scale, old_sum
    for column in 1:columns
        old_scale, old_sum = _rk_merge_sumsq(
            old_scale, old_sum, parts[1, column], parts[2, column]
        )
        new_scale, new_sum = _rk_merge_sumsq(
            new_scale, new_sum, parts[3, column], parts[4, column]
        )
    end
    magnitude = max(old_scale * sqrt(old_sum), new_scale * sqrt(new_sum))
    isfinite(magnitude) || return Inf
    relative = rtol * magnitude
    each_column!(execution, columns) do column
        column_error_scale, column_error_sum = zero(eltype(parts)), zero(eltype(parts))
        for row in 1:rows
            if mask[row]
                scale = relative * weights[row] + atol
                value = defect[row, column]
                if !isfinite(scale)
                    column_error_scale, column_error_sum = (
                        oftype(column_error_scale, Inf), one(column_error_sum),
                    )
                elseif !(iszero(scale) && iszero(value))
                    column_error_scale, column_error_sum = _rk_sumsq(
                        column_error_scale, column_error_sum, real(value) / scale,
                    )
                    column_error_scale, column_error_sum = _rk_sumsq(
                        column_error_scale, column_error_sum, imag(value) / scale,
                    )
                end
            end
        end
        parts[1, column], parts[2, column] = column_error_scale, column_error_sum
    end
    error_scale, error_sum = zero(eltype(parts)), zero(eltype(parts))
    for column in 1:columns
        error_scale, error_sum = _rk_merge_sumsq(
            error_scale, error_sum, parts[1, column], parts[2, column]
        )
    end
    return Float64(error_scale * sqrt(error_sum))
end

# Accumulate sum(abs2(x)) as scale^2 * total, with normalized terms at most one.
# Factoring out the larger scale at each merge avoids squaring physical amplitudes.
@inline function _rk_sumsq(scale, total, value)
    magnitude = abs(value)
    isfinite(magnitude) || return (oftype(scale, Inf), one(total))
    iszero(magnitude) && return (scale, total)
    if magnitude > scale
        ratio = scale / magnitude
        return (magnitude, one(total) + total * ratio * ratio)
    end
    ratio = magnitude / scale
    return (scale, total + ratio * ratio)
end

@inline function _rk_merge_sumsq(scale, total, other_scale, other_total)
    if !isfinite(scale) || !isfinite(other_scale)
        return (oftype(scale, Inf), one(total))
    end
    iszero(other_scale) && return (scale, total)
    if other_scale > scale
        ratio = scale / other_scale
        return (other_scale, other_total + total * ratio * ratio)
    end
    ratio = other_scale / scale
    return (scale, total + other_total * ratio * ratio)
end

"""
    rk_commit!(state, work, active_mask, execution)

Commit the candidate while restoring exact zeros outside the propagated band.

`active_mask` has one Boolean per spectral row. The caller first applies any accepted
split collar to `work.candidate`; no acceptance policy or extra transform is hidden here.
The existing `state` storage is reused. The call returns `nothing`.
"""
function rk_commit!(state, work::RungeKuttaWork, active_mask, execution)
    candidate = work.candidate
    each!(execution, size(state)...) do row, column
        state[row, column] = active_mask[row] ? candidate[row, column] : zero(eltype(state))
    end
    return nothing
end
