# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public AdamsHistoryMetadata, AdamsHistoryBuffers
public adams_history_info, export_adams_history!, restore_adams_history!

"""
    AdamsHistoryMetadata(z, proposal, order, rejections, last_rejected, collar_was_quiet)

Carry the scalar numerical part of the existing portable Adams snapshot.

`z` and the positive next `proposal` are binary64 metres. `order` is the chosen next
order, which can exceed the retained count during startup; `rejections` counts
consecutive rejected trials. The Boolean flags preserve the last rejection and the
previous in-stage collar quiet state. Compatibility is checked by restore before
mutation, not by this transport record. Method/history identity, scientific precision,
noise and cumulative execution ledgers remain at the shared preparation boundary.
"""
struct AdamsHistoryMetadata
    z::Float64
    proposal::Float64
    order::Int
    rejections::Int
    last_rejected::Bool
    collar_was_quiet::Bool

    function AdamsHistoryMetadata(
            z::Float64, proposal::Float64, order::Int, rejections::Int,
            last_rejected::Bool, collar_was_quiet::Bool,
        )
        return new(z, proposal, order, rejections, last_rejected, collar_was_quiet)
    end
end

"""
    AdamsHistoryBuffers(state, positions, identities, slots, store;
                        states=nothing, rates=nothing)

Borrow detached CPU arrays for one portable Adams import or export call.

The raw internal `state` has shape `(n_omega,n_columns)` and `ComplexF64` storage,
widened losslessly for a binary32 session without restoring physical scale. Each
full numerical store has shape `(n_omega,n_columns,max_order+2)`. Source and sampled
state stores use session precision; sampled rates always use `ComplexF64`. Column
`polarisation+n_pol*(mode-1)` matches the canonical Python C-order snapshot rows
`(n_slots,1,n_modes,n_pol,n_omega)` without numerical conversion. Inactive store rows
are copied too. Compensated history requires both `states` and `rates`; plain history
requires both to be `nothing`.

The metadata vectors have exactly the retained entry count, oldest first:
`positions::Float64` contains actual sample coordinates in metres, `identities::UInt8`
uses accepted `0` or predicted `1`, and `slots::Int64` uses portable zero-based slots.
All arrays must have one-based axes and writable strided CPU storage; views are
admitted. The record retains its inputs only as a call argument. Import never retains
them in a session. The caller owns any foreign allocation for the complete call.
"""
struct AdamsHistoryBuffers{A, P, I, S, H, T, R}
    state::A
    positions::P
    identities::I
    slots::S
    store::H
    states::T
    rates::R
end

function AdamsHistoryBuffers(
        state, positions, identities, slots, store; states = nothing, rates = nothing,
    )
    return AdamsHistoryBuffers(state, positions, identities, slots, store, states, rates)
end

_adams_transfer_arrays(buffers::AdamsHistoryBuffers) = (
    buffers.state, buffers.positions, buffers.identities, buffers.slots,
    buffers.store, buffers.states, buffers.rates,
)

function _adams_transfer_shape(value, ::Type{T}, shape::NTuple{N, Int}) where {T, N}
    value isa StridedArray{T, N} || return false
    return size(value) == shape &&
        all(axis -> axes(value, axis) == Base.OneTo(shape[axis]), 1:N)
end

function _adams_transfer_layout(buffers, session, count)
    shape = size(session.field)
    stored = size(session.history.source)
    _adams_transfer_shape(buffers.state, ComplexF64, shape) ||
        return "the snapshot state needs matching strided binary64 complex storage"
    _adams_transfer_shape(buffers.positions, Float64, (count,)) ||
        return "the snapshot positions need one binary64 value per retained entry"
    _adams_transfer_shape(buffers.identities, UInt8, (count,)) ||
        return "the snapshot identities need one UInt8 sample kind per retained entry"
    _adams_transfer_shape(buffers.slots, Int64, (count,)) ||
        return "the snapshot slots need one Int64 row index per retained entry"
    storage = eltype(session.history.source)
    _adams_transfer_shape(buffers.store, storage, stored) ||
        return "the snapshot source store differs from the session precision or shape"
    if session.history.compensated
        _adams_transfer_shape(buffers.states, storage, stored) ||
            return "the snapshot sampled states differ from the session precision or shape"
        _adams_transfer_shape(buffers.rates, ComplexF64, stored) ||
            return "the snapshot sampled rates do not fit compensated history"
    elseif buffers.states !== nothing || buffers.rates !== nothing
        return "plain Adams history does not keep sampled state or rate stores"
    end
    return nothing
end

function _adams_transfer_retained_alias(buffers, session)
    for array in _adams_transfer_arrays(buffers)
        array === nothing && continue
        _retained_alias(array, session) && return true
    end
    return false
end

function _adams_transfer_destination_alias(buffers)
    arrays = _adams_transfer_arrays(buffers)
    for first in eachindex(arrays)
        arrays[first] === nothing && continue
        _transfer_writable_destination(arrays[first])
        for second in (first + 1):length(arrays)
            arrays[second] === nothing && continue
            _transfer_arrays_alias(arrays[first], arrays[second]) && return true
        end
    end
    return false
end

function _adams_transfer_finite(buffers)
    return all(isfinite, buffers.state) && all(isfinite, buffers.store) &&
        (buffers.states === nothing || all(isfinite, buffers.states)) &&
        (buffers.rates === nothing || all(isfinite, buffers.rates))
end

function _adams_transfer_metadata(session)
    control, state = session.control, session.state
    return AdamsHistoryMetadata(
        state.z, state.proposal, control.order, control.rejections,
        control.last_rejected, control.collar_was_quiet,
    )
end

"""
    adams_history_info(session, generation, observation_z)

Describe a pending committed Adams history without copying its numerical arrays.

Return `(metadata,n_entries,n_slots,compensated,nbytes)`, or `nothing` when no samples
are retained or the observed coordinate differs from the committed endpoint. Dense
observations must not publish a history from ahead of their field. `nbytes` matches
the existing snapshot retention measure: raw state plus full numerical stores, without
the small metadata vectors. A stale, closed, nonpending or poisoned generation throws
`ArgumentError`, as does a nonfinite observation coordinate. Positions are metres.
"""
function adams_history_info(
        session::AdamsSession, generation::Int, observation_z::Float64,
    )
    _session_exportable(session, generation)
    isfinite(observation_z) || throw(ArgumentError("the observation position is nonfinite"))
    count = session.control.held
    (
        count == 0 || observation_z != session.state.z ||
            _march_observed_position(session) != session.state.z
    ) && return nothing
    1 <= count <= session.policy.order + 1 ||
        throw(ArgumentError("the retained Adams history count is invalid"))
    history = session.history
    # The portable accepted state always widens to binary64. Source/sample-state
    # rows retain their numerical precision, exactly as the shared NumPy schema.
    bytes = sizeof(ComplexF64) * length(session.field) + sizeof(history.source)
    if history.compensated
        bytes += sizeof(history.states) + sizeof(history.rates)
    end
    return (
        _adams_transfer_metadata(session), count, size(history.source, 3),
        history.compensated, bytes,
    )
end

function _adams_transfer_selected(session)
    history = session.history
    for index in 1:session.control.held
        slot = session.slots[index]
        1 <= slot <= length(history.positions) && history.occupied[slot] ||
            return "the retained Adams selection names an unoccupied slot"
        position = session.positions[index]
        isfinite(position) && position == history.positions[slot] ||
            return "the retained Adams sample coordinate is inconsistent"
        history.identities[slot] in (0, 1) ||
            return "the retained Adams identity is not an accepted or predicted sample"
        if index > 1 && !(session.positions[index - 1] < position)
            return "the retained Adams sample coordinates are not strictly oldest first"
        end
        for earlier in 1:(index - 1)
            session.slots[earlier] != slot ||
                return "the retained Adams selection repeats a slot"
        end
    end
    return nothing
end

"""
    export_adams_history!(buffers, session, generation, observation_z)

Copy one eligible history losslessly into detached portable destination arrays.

Validate the pending generation, exact destination layout and all pairwise/retained
overlap before any write. Copy the raw accepted state, every full numerical store row,
and the selected oldest-first metadata. No field conversion, mask, source evaluation,
ledger update or noise operation occurs. Return `nothing`; invalid destinations,
ineligible history or inconsistent/nonfinite retained data throw `ArgumentError`.
"""
function export_adams_history!(
        buffers::AdamsHistoryBuffers, session::AdamsSession,
        generation::Int, observation_z::Float64,
    )
    info = adams_history_info(session, generation, observation_z)
    info === nothing && throw(ArgumentError("no committed Adams history is exportable"))
    reason = _adams_transfer_layout(buffers, session, info[2])
    reason === nothing || throw(ArgumentError(reason))
    (
        _adams_transfer_retained_alias(buffers, session) ||
            _adams_transfer_destination_alias(buffers)
    ) &&
        throw(ArgumentError("Adams history destinations must be detached and disjoint"))
    reason = _adams_transfer_selected(session)
    reason === nothing || throw(ArgumentError(reason))
    history = session.history
    all(isfinite, session.field) && all(isfinite, history.source) &&
        all(isfinite, history.states) && all(isfinite, history.rates) ||
        throw(ArgumentError("the retained Adams state or full store is nonfinite"))
    copyto!(buffers.state, session.field)
    copyto!(buffers.store, history.source)
    if history.compensated
        copyto!(buffers.states, history.states)
        copyto!(buffers.rates, history.rates)
    end
    for index in 1:session.control.held
        slot = session.slots[index]
        buffers.positions[index] = session.positions[index]
        buffers.identities[index] = UInt8(history.identities[slot])
        buffers.slots[index] = slot - 1
    end
    return nothing
end

function _adams_transfer_fresh(session)
    state = session.state
    return !state.closed && !state.pending && !state.finished && !state.poisoned &&
        !state.in_flight && !state.services_started && !state.endpoint_pending &&
        !state.finishing && !state.snapshot && state.generation == 0 &&
        state.observation_kind === :none && state.attempts == 0 && state.accepted == 0 &&
        state.rejected == 0 && session.attempts.count == 0 && session.control.held == 0
end

function _adams_transfer_refusal(session, metadata, buffers)
    _adams_transfer_fresh(session) ||
        return "Adams history can only be restored into a fresh unadvanced session"
    isfinite(metadata.z) && metadata.z == session.state.z ||
        return "the snapshot position differs from the resumed session position"
    isfinite(metadata.proposal) && metadata.proposal > 0.0 ||
        return "the snapshot proposal must be finite and positive"
    1 <= metadata.order <= session.policy.order ||
        return "the snapshot next order is outside the prepared Adams capacity"
    metadata.rejections >= 0 || return "the snapshot rejection count is negative"
    buffers.positions isa AbstractVector ||
        return "the snapshot positions must be a vector"
    count = length(buffers.positions)
    1 <= count <= session.policy.order + 1 ||
        return "the snapshot retained count is outside the prepared Adams capacity"
    reason = _adams_transfer_layout(buffers, session, count)
    reason === nothing || return reason
    _adams_transfer_retained_alias(buffers, session) &&
        return "the snapshot inputs must not alias retained session storage"
    for index in 1:count
        position, slot = buffers.positions[index], buffers.slots[index]
        isfinite(position) || return "the snapshot sample coordinates are nonfinite"
        if index > 1 && !(buffers.positions[index - 1] < position)
            return "the snapshot sample coordinates must be strictly oldest first"
        end
        0 <= slot < size(session.history.source, 3) ||
            return "the snapshot sample names no row of this engine"
        buffers.identities[index] in (0x00, 0x01) ||
            return "the snapshot sample kind is neither accepted nor predicted"
        for earlier in 1:(index - 1)
            buffers.slots[earlier] != slot || return "the snapshot repeats a history slot"
        end
    end
    # Match math.isclose(..., rel_tol=1e-12, abs_tol=0) in the shared host restore.
    # The actual sample is kept unchanged: landing may snap only its accepted anchor.
    newest = buffers.positions[count]
    abs(newest - metadata.z) <= 1.0e-12 * max(abs(newest), abs(metadata.z)) ||
        return "the snapshot newest sample is not at its accepted position"
    _adams_transfer_finite(buffers) ||
        return "the snapshot raw state or full numerical store is nonfinite"
    if eltype(session.field) === ComplexF32
        all(value -> isequal(ComplexF64(ComplexF32(value)), value), buffers.state) ||
            return "the snapshot raw state does not widen losslessly from binary32"
    end
    return nothing
end

"""
    restore_adams_history!(session, metadata, buffers)

Install a portable snapshot into a fresh unadvanced Adams session without evaluation.

Return `nothing` on success, or a compatibility reason string before any write. A
fresh session has no numerical work, service start, published event, outstanding
snapshot request or retained history; `plane_pending=false` at construction is valid.
Check matching shapes/types, finite raw state and every full store row, sample
kind/slot metadata, oldest-first coordinates, capacity and shared newest-position
proximity. Retained input aliases refuse. Independent input arrays may overlap each
other because they are read-only during import.

Binary32 requires an exactly widened raw state and matching binary32 source/sample-state
rows, with binary64 rates. Copy the raw field exactly, even if an independently converted
launch differs
in its last bits. Restore the chosen next order, rejection/collar flags and proposal;
keep all retained entries independently of that order. Full inactive store rows remain
unchanged. Rebuild slot occupancy and invalidate numerical trial/stencil scratch.
Do not reapply masks or input noise, evaluate a source, change delivery/replay cursors,
or restore cumulative ledgers. The shared boundary owns history keys, scientific
precision, foreign row names and retained-age phase admission before calling this leaf.
Unexpected programming or copy failures propagate rather than claiming a fresh restart.
"""
function restore_adams_history!(
        session::AdamsSession, metadata::AdamsHistoryMetadata, buffers::AdamsHistoryBuffers,
    )
    reason = _adams_transfer_refusal(session, metadata, buffers)
    reason === nothing || return reason
    history, control, state = session.history, session.control, session.state
    copyto!(session.field, buffers.state)
    copyto!(history.source, buffers.store)
    if history.compensated
        copyto!(history.states, buffers.states)
        copyto!(history.rates, buffers.rates)
    end
    fill!(history.occupied, false)
    fill!(history.positions, 0.0)
    fill!(history.identities, 0)
    fill!(session.positions, 0.0)
    fill!(session.slots, 0)
    count = length(buffers.positions)
    for index in 1:count
        slot = Int(buffers.slots[index]) + 1
        position = buffers.positions[index]
        session.positions[index], session.slots[index] = position, slot
        history.positions[slot] = position
        history.identities[slot] = buffers.identities[index]
        history.occupied[slot] = true
    end
    control.held, control.order = count, metadata.order
    control.rejections, control.last_rejected = metadata.rejections, metadata.last_rejected
    control.collar_was_quiet = metadata.collar_was_quiet
    state.z, state.proposal = metadata.z, metadata.proposal
    state.flight_z, state.flight_dz = metadata.z, 0.0
    session.work.ready = false
    session.work.position, session.work.step = 0.0, 0.0
    session.work.order, session.work.n_orders = 0, 0
    stencil = session.stencil
    fill!(stencil.slots, 0)
    fill!(stencil.orders, 0)
    fill!(stencil.nodes, 0.0)
    stencil.order, stencil.count, stencil.n_orders = 0, 0, 0
    fill!(session.norms, 0.0)
    return nothing
end
