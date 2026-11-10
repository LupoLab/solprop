# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public ModalPreparation, prepare_modal, start_runge_kutta, start_adams
public start_low_storage
public advance_session, acknowledge_session, session_status, export_session!
public session_attempts!, session_failure, session_memory, preparation_memory
public fail_session
public session_execution, session_orders!, session_budget
public restore_session_counts!
public SessionPreparation, prepare_observations, session_observation_status
public session_observations!
public copy_session_observations!

"""
    ModalPreparation(execution, data, operator, power_weight, node_chunk)

Retain immutable owned numerical preparation shared by scalar modal sessions.

`data` is `ModalSourceData`, `operator` a diagonal generator or represented cumulative
operator, and `power_weight` an `ObservationWeight`. Each started session constructs
its own source, FFT plans, flow, exports and march storage. The positive `node_chunk`
is the maximum physical nodes held at once; it does not change their quadrature.
"""
struct ModalPreparation{E, D, O, W}
    execution::E
    data::D
    operator::O
    power_weight::W
    node_chunk::Int
end

"""
    SessionPreparation(modal, observations)

Pair immutable modal preparation with one request's immutable observation policy.

`modal` is a `ModalPreparation`, and `observations` a `SessionObservationPolicy`.
Every start creates independent mutable observation work and bounded delivery storage.
The additional preparation ledger excludes arrays already owned by `modal`.
"""
struct SessionPreparation{M <: ModalPreparation, O <: SessionObservationPolicy}
    modal::M
    observations::O
end

"""
    prepare_observations(modal, transfer)

Copy one request's observation transfer through the optional PythonCall extension.

The resulting `SessionPreparation` retains no Python objects and shares only immutable
numerical inputs with `modal`. No session or delivery cursor is constructed here.
"""
function prepare_observations end

_modal_preparation(preparation::ModalPreparation) = preparation
_modal_preparation(preparation::SessionPreparation) = preparation.modal
_observation_services(preparation::ModalPreparation) = _NoSessionServices()
function _observation_services(preparation::SessionPreparation)
    modal = preparation.modal
    return ObservationServices(
        modal.execution, preparation.observations; node_chunk = modal.node_chunk,
    )
end

struct _FieldExport{W, A, C}
    weight::W
    values::A
    coefficients::C
end

function _FieldExport(execution, weight)
    count = weight.weights === nothing ? 0 : size(weight.weights.coefficients, 1)
    return _FieldExport(
        weight, dzeros(execution, Float64, weight.shape...),
        dzeros(execution, Float64, count),
    )
end

"""
    prepare_modal(transfer, threads, node_chunk)

Convert a Python modal transfer once through the PythonCall extension.

Every numerical array is copied into Julia-owned storage; no Python object is
retained. `threads` and `node_chunk` are positive execution budgets. Pure Python
admission precedes this entry; typed numerical constructors recheck the data.
"""
function prepare_modal end

"""
    start_runge_kutta(preparation, transfer, field, z, plane, plane_pending, stop)

Copy a starting Python field and resolved run transfer into an independently owned session.

The field uses `(n_modes,n_pol,n_omega)` Python storage; `z` is metres and `plane`
the authoritative zero-based next-plane cursor. `plane_pending` says whether launch
must be delivered. `stop` is the lifetime-owned atomic flag. The extension performs
all conversion before returning the session; mutable state is never shared between starts.
"""
function start_runge_kutta end

"""
    start_adams(preparation, transfer, field, z, plane, plane_pending, stop)

Copy a starting field and resolved exponential Adams policy into an owned session.

Arguments have the same ownership, shape and units as `start_runge_kutta`. This
entry currently starts fresh history; portable continuation is separately admitted.
The extension copies every policy value from the shared resolved Python record.
"""
function start_adams end

"""
    start_low_storage(preparation, transfer, field, z, plane, plane_pending, stop)

Copy a starting field and fixed itinerary into an owned low-storage RK4IP session.

Arguments have the same ownership, shape and units as `start_runge_kutta`. This
scheme retains no continuation history or embedded tableau. The source invocation
and committed-step counters remain separate when an in-place step fails.
"""
function start_low_storage end

"""
    advance_session(session, work_limit, deadline)

Advance the complete engine through a positional bridge call that may release Python's GIL.

Return a plain tuple of numerical event/report scalars. This performs no Python
entry, callback, field copy or periodic-statistics handoff while advancing.
"""
function advance_session(session, work_limit::Int, deadline::Float64)
    report = advance!(session; work_limit, deadline)
    return _bridge_report(report)
end

function _bridge_report(report)
    return (
        Int(report.kind), report.z, report.plane, report.generation, report.attempts,
        report.rejected, report.plane_attempts, report.plane_rejected, report.max_norm,
        report.mean_norm, report.wall_time, report.met_tolerance, report.poisoned,
    )
end

"""
    fail_session(session)

Issue a failed barrier at the committed state after an external service failure.

The session must be open and have no pending or terminal event. No numerical work or
delivery cursor changes; the resulting chunk remains available for a failure checkpoint.
"""
function fail_session(session)
    state = session.state
    !state.closed && !state.pending && !state.finished ||
        throw(ArgumentError("the session cannot issue another failure barrier"))
    state.failure_code = :ExternalFailure
    state.failure_message = "the Python service failed between numerical events"
    return _bridge_report(_march_issue!(session, :failed))
end

"""Acknowledge a pending numerical generation through the positional bridge."""
acknowledge_session(session, generation::Int) = acknowledge!(session, generation)

"""
    session_status(session)

Return committed position, next plane, attempt/rejection counts and export generation.

The tuple contains only host scalars; the position is metres and the plane zero-based.
The final Boolean states whether the numerical field is poisoned by an in-flight failure.
"""
function session_status(session)
    state = session.state
    return (
        state.z, state.plane - 1, state.attempts, state.rejected,
        state.generation, state.poisoned,
    )
end

function _session_chunk_pending(session, generation)
    state = session.state
    !state.closed && state.pending && state.generation == generation ||
        throw(ArgumentError("the session chunk is stale or not pending"))
    return nothing
end

"""
    session_observation_status(session, generation)

Read pending periodic row count, packed periodic width and valid plane width.

Validate the open pending generation before reading. Return zero widths/counts when
no observation services are attached. This read never drains numerical buffers.
"""
function session_observation_status(session, generation::Int)
    _session_chunk_pending(session, generation)
    services = session.services
    services isa ObservationServices || return (0, 0, 0)
    return (
        services.rows.count, size(services.rows.values, 1),
        services.plane_valid ? length(services.plane_values) : 0,
    )
end

"""
    session_observations!(positions, steps, values, plane, session, generation)

Copy a pending observation payload into detached, writable destinations.

The Float64 vectors `positions` and Int64 `steps` have the pending row count. The
Float64 matrix `values` is `(periodic_width,n_rows)`; `plane` has the valid packed
plane width or zero entries. Validate dimensions and all aliases before any writes.
The call preserves the generation and its numerical buffers for a failed-copy retry.
"""
function session_observations!(
        positions::AbstractVector{Float64}, steps::AbstractVector{Int64},
        values::AbstractMatrix{Float64}, plane::AbstractVector{Float64},
        session, generation::Int,
    )
    count, width, plane_width = session_observation_status(session, generation)
    length(positions) == length(steps) == count && size(values) == (width, count) &&
        length(plane) == plane_width ||
        throw(DimensionMismatch("observation destinations disagree with the pending chunk"))
    destinations = (positions, steps, values, plane)
    Base.require_one_based_indexing(destinations...)
    for (index, destination) in enumerate(destinations)
        (
            _retained_alias(destination, session) ||
                _retained_alias(destination, session.services)
        ) && throw(ArgumentError("observation destinations must not alias retained storage"))
        for other in (index + 1):length(destinations)
            Base.mightalias(destination, destinations[other]) &&
                throw(ArgumentError("observation destinations must be disjoint"))
        end
    end
    services = session.services
    services isa ObservationServices || return nothing
    for row in 1:count
        positions[row], steps[row] = services.rows.z[row], services.rows.step[row]
        for column in 1:width
            values[column, row] = services.rows.values[column, row]
        end
    end
    for index in 1:plane_width
        plane[index] = services.plane_values[index]
    end
    return nothing
end

"""
    copy_session_observations!(positions, steps, values, plane, session, generation)

Copy a pending payload through the PythonCall destination adapter.

Python vectors are writable C-contiguous Float64/Int64 buffers. The C-order Float64
matrix `values` is `(n_rows,periodic_width)`; Julia's numerical copy uses its transposed
storage layout. Validate every destination before writing and retain foreign owners
for the synchronous copy. No Python reference remains in the session.
"""
function copy_session_observations! end

"""
    session_execution(session, generation)

Read cumulative actual source calls and optional method counters without draining.

The logical tuple is startup, predictor, corrected, rejected, accepted steps, rejected
steps, restarts, failure restarts and dense observations, in shared `CallLedger` order.
Other methods return `nothing` for that tuple. The third result is the low-storage
tuple of committed source calls and norm reductions, or `nothing` for other methods.
A pending poisoned generation still
permits these diagnostic reads; closed, stale and acknowledged generations refuse.
"""
function session_execution(session::RungeKuttaSession, generation::Int)
    _session_chunk_pending(session, generation)
    return (session.source.calls, nothing, nothing)
end

function session_execution(session::AdamsSession, generation::Int)
    _session_chunk_pending(session, generation)
    ledger = session.ledger
    return (
        ledger.source_evaluations,
        (
            ledger.startup, ledger.predictor, ledger.corrected, ledger.rejected,
            ledger.accepted_steps, ledger.rejected_steps, ledger.restarts,
            ledger.failure_restarts, ledger.dense_observations,
        ),
        nothing,
    )
end

function session_execution(session::LowStorageSession, generation::Int)
    _session_chunk_pending(session, generation)
    counts = session.counts
    return (
        counts.attempted_source_evaluations, nothing,
        (counts.committed_source_evaluations, counts.norm_evaluations),
    )
end

"""
    restore_session_counts!(session, counts)

Install nine cumulative logical Adams counters before history or numerical work.

`counts` is an integer tuple in shared `CallLedger` order. Validate every count and
the charged total before mutation. Actual source invocations belong to this execution
and remain zero; accepted order runs are owned by the Python delivery record. A field
restart retains these charges even when its history is incompatible. Reject a session
that has already started, published an event or restored numerical history.
"""
function restore_session_counts!(session::AdamsSession, counts::NTuple{9, Int})
    _adams_transfer_fresh(session) ||
        throw(ArgumentError("call counts require a fresh unadvanced Adams session"))
    all(count -> count >= 0, counts) ||
        throw(ArgumentError("logical call counters must be nonnegative"))
    counts[1] <= typemax(Int) - counts[2] &&
        counts[1] + counts[2] <= typemax(Int) - counts[3] ||
        throw(ArgumentError("logical call total exceeds integer capacity"))
    ledger = session.ledger
    ledger.startup, ledger.predictor, ledger.corrected = counts[1], counts[2], counts[3]
    ledger.rejected, ledger.accepted_steps, ledger.rejected_steps =
        counts[4], counts[5], counts[6]
    ledger.restarts, ledger.failure_restarts, ledger.dense_observations =
        counts[7], counts[8], counts[9]
    return nothing
end

"""
    session_orders!(output, session, generation)

Copy accepted Adams orders, or zero for rejected attempts, into detached integer storage.

The destination has exactly the pending attempt count. Validate generation, length and
nonaliasing before writing; this read neither drains the chunk nor changes counters.
"""
function session_orders!(
        output::AbstractVector{Int64}, session::AdamsSession, generation::Int,
    )
    _session_chunk_pending(session, generation)
    count = session.attempts.count
    length(output) == count ||
        throw(DimensionMismatch("order destination must match the pending chunk"))
    _retained_alias(output, session) &&
        throw(ArgumentError("order destination must not alias retained storage"))
    for row in 1:count
        output[row] = session.attempts.accepted[row] ? session.orders[row] : 0
    end
    return nothing
end

"""
    session_budget(session, generation)

Read maximum, spent and required logical calls from a pending Adams budget failure.

The refused trial has not changed its history, field or ledger. Its required charge
uses the same numerical helper as admission; Python only formats the shared exception.
Throw `ArgumentError` unless the requested generation is a pending budget failure.
"""
function session_budget(session::AdamsSession, generation::Int)
    _session_chunk_pending(session, generation)
    session.state.failure_code === :AdamsBudgetError ||
        throw(ArgumentError("the pending failure is not an Adams call budget"))
    return (
        session.policy.max_calls, adams_charged_calls(session.ledger),
        _adams_session_cost(session),
    )
end

function _session_exportable(session, generation)
    state = session.state
    !state.closed && state.pending && !state.poisoned && state.generation == generation ||
        throw(ArgumentError("the session export is poisoned, stale or not pending"))
    return nothing
end

# Destinations cross an ownership boundary. Reject overlap with every retained
# numerical array, including immutable preparation and scratch, before any write.
# Vendor plans are opaque, but their input/output arrays are explicit owned fields.
function _retained_alias(destination, value)
    if value isa AbstractArray
        return Base.mightalias(destination, value)
    elseif value === nothing || isbitstype(typeof(value)) || value isa String
        return false
    elseif value isa Union{Tuple, NamedTuple}
        return any(item -> _retained_alias(destination, item), value)
    elseif parentmodule(typeof(value)) === @__MODULE__
        return any(
            name -> _retained_alias(destination, getfield(value, name)),
            fieldnames(typeof(value)),
        )
    end
    return false
end

"""
    export_session!(output, session, generation, power_weight, power, optics)

Write a pending field into detached `(n_omega,n_columns)` binary64 destination storage.

For `power=true`, first multiply by the square root of the positive dimensionless
weight. For `optics=true`, then conjugate and multiply by `(-1)^(omega_index-1)`.
This is the shared destination-export recipe in its specified operation order.
The complex products retain their explicit zero terms to preserve signed zeros;
globally unit power weights skip their multiplication as in the shared recipe.
Generation, shape and alias validation precede writes. The PythonCall extension maps
the validated C-contiguous Python destination onto these frequency-first columns.
"""
function export_session!(
        output::AbstractMatrix{ComplexF64}, session, generation::Int, power_weight,
        power::Bool, optics::Bool,
    )
    _session_exportable(session, generation)
    field = session.field
    size(output) == size(field) == size(power_weight) ||
        throw(DimensionMismatch("session export dimensions disagree"))
    (_retained_alias(output, session) || Base.mightalias(output, power_weight)) &&
        throw(ArgumentError("an export must not alias session storage"))
    scale = power && !all(isone, power_weight)
    each!(session.execution, size(field)...) do row, column
        value = field[row, column]
        re, im = real(value), imag(value)
        if scale
            root = sqrt(power_weight[row, column])
            # The shared NumPy recipe promotes the real root to a complex factor.
            # Keep its zero terms and operation order, including signed-zero results.
            re, im = re * root - im * 0.0, re * 0.0 + im * root
        end
        if optics
            conjugate = -im
            sign = isodd(row) ? 1.0 : -1.0
            re, im = sign * re - 0.0 * conjugate, sign * conjugate + 0.0 * re
        end
        output[row, column] = ComplexF64(re, im)
    end
    return nothing
end

"""
    session_attempts!(positions, lengths, norms, accepted, elapsed, session, generation)

Copy the pending attempt chunk once into detached host destinations of exact length.

The destinations have length `session.attempts.count`; positions/lengths are metres,
norms dimensionless and elapsed times seconds. Reading does not drain or change any
counter; acknowledgment releases the chunk. Python tracks whether an event delivered it.
"""
function session_attempts!(
        positions, lengths, norms, accepted, elapsed, session, generation::Int,
    )
    buffer = session.attempts
    _session_chunk_pending(session, generation)
    length(positions) == length(lengths) == length(norms) == length(accepted) ==
        length(elapsed) == buffer.count ||
        throw(DimensionMismatch("attempt destinations must match the pending chunk"))
    destinations = (positions, lengths, norms, accepted, elapsed)
    for first in eachindex(destinations)
        _retained_alias(destinations[first], session) &&
            throw(ArgumentError("attempt destinations must not alias retained storage"))
        for second in (first + 1):length(destinations)
            Base.mightalias(destinations[first], destinations[second]) &&
                throw(ArgumentError("attempt destinations must be pairwise disjoint"))
        end
    end
    for row in 1:buffer.count
        positions[row] = buffer.z[row]
        lengths[row] = buffer.dz[row]
        norms[row] = buffer.norm[row]
        accepted[row], elapsed[row] = buffer.accepted[row], buffer.elapsed[row]
    end
    return nothing
end

"""Return the terminal failure's code, message and in-flight position/length in metres."""
function session_failure(session)
    state = session.state
    return (
        String(state.failure_code), state.failure_message, state.in_flight,
        state.flight_z, state.flight_dz,
    )
end

# Ledger traversal is an observation, never a numerical hot path. Arrays shared
# with immutable preparation or through views are counted once by owner identity.
function _memory_arrays!(ledger, seen, path, value)
    if value === nothing || isbitstype(typeof(value)) || value isa String
        return nothing
    elseif value isa AbstractArray
        owner = parent(value)
        if owner !== value
            return _memory_arrays!(ledger, seen, path, owner)
        end
        if !haskey(seen, owner)
            seen[owner] = true
            ledger[Symbol(path)] = sizeof(owner)
        end
    elseif value isa Tuple
        for (index, item) in enumerate(value)
            _memory_arrays!(ledger, seen, string(path, ".", index), item)
        end
    elseif parentmodule(typeof(value)) === @__MODULE__
        for name in fieldnames(typeof(value))
            _memory_arrays!(ledger, seen, string(path, ".", name), getfield(value, name))
        end
    elseif parentmodule(typeof(value)) === FFTW
        if !haskey(seen, value)
            seen[value] = true
            ledger[Symbol(string(path, ".estimated_plan"))] = Base.summarysize(value)
        end
    end
    return nothing
end

function _memory_record(ledger)
    ordered = sort!(collect(keys(ledger)); by = string)
    return NamedTuple{Tuple(ordered)}(Tuple(ledger[key] for key in ordered))
end

"""
    session_memory(session)

Return named retained array sizes in bytes for one independent numerical session.

Shared source/operator/export preparation is excluded and reported by its owner.
Views and repeated references count their owning array once. FFT plan sizes are
explicitly marked estimates because their vendor allocations are opaque.
"""
function session_memory(session)
    ledger = Dict{Symbol, Int}()
    seen = IdDict{Any, Bool}()
    _memory_arrays!(ledger, seen, "shared.source", session.source.data)
    _memory_arrays!(ledger, seen, "shared.operator", session.flow.operator)
    if session.exports !== nothing
        _memory_arrays!(ledger, seen, "shared.weight", session.exports.weight)
    end
    if session.services isa ObservationServices
        _memory_arrays!(ledger, seen, "shared.observations", session.services.work.data)
    end
    empty!(ledger)
    _memory_arrays!(ledger, seen, "session", session)
    # Julia-owned bytes were visited above. Only foreign storage needs this entry.
    session.stop.storage === nothing && (ledger[:stop] = 1)
    return _memory_record(ledger)
end

"""
    preparation_memory(preparation)

Return named array byte counts for immutable shared modal preparation.

The source records, represented operator and export weights are owned copies.
Sessions refer to these arrays; their ledgers exclude them to avoid double counting.
"""
function preparation_memory(preparation::ModalPreparation)
    ledger = Dict{Symbol, Int}()
    _memory_arrays!(ledger, IdDict{Any, Bool}(), "preparation", preparation)
    return _memory_record(ledger)
end

function preparation_memory(preparation::SessionPreparation)
    ledger = Dict{Symbol, Int}()
    seen = IdDict{Any, Bool}()
    _memory_arrays!(ledger, seen, "modal", preparation.modal)
    empty!(ledger)
    _memory_arrays!(ledger, seen, "observations", preparation.observations)
    return _memory_record(ledger)
end

function export_session!(
        output::AbstractMatrix{ComplexF64}, session, generation::Int,
        preparation::SessionPreparation, power::Bool, optics::Bool,
    )
    _retained_alias(output, preparation) &&
        throw(ArgumentError("an export must not alias retained numerical storage"))
    return export_session!(output, session, generation, preparation.modal, power, optics)
end

function export_session!(
        output::AbstractMatrix{ComplexF64}, session, generation::Int,
        preparation::ModalPreparation, power::Bool, optics::Bool,
    )
    _session_exportable(session, generation)
    storage = session.exports
    storage !== nothing && storage.weight === preparation.power_weight ||
        throw(ArgumentError("session exports belong to another numerical preparation"))
    size(output) == size(session.field) ||
        throw(DimensionMismatch("session export dimensions disagree"))
    (_retained_alias(output, session) || _retained_alias(output, preparation)) &&
        throw(ArgumentError("an export must not alias retained numerical storage"))
    if power
        evaluate_weight!(
            storage.values, storage.weight, session.state.z, session.execution;
            work = storage.coefficients,
        )
    end
    return export_session!(output, session, generation, storage.values, power, optics)
end
