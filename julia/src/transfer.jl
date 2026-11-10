# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public EnvelopePreparation, prepare_envelope, start_runge_kutta
public advance_session, acknowledge_session, session_status, export_session!
public session_attempts!, session_failure, session_memory, preparation_memory
public fail_session

"""
    EnvelopePreparation(execution, data, operator, crop_index, power_weight)

Retain immutable owned numerical preparation shared by constant-envelope sessions.

`data` is `EnvelopeKerrData`, `operator` a diagonal generator, `crop_index` the
zero-based frequency map and `power_weight` dimensionless positive modal weights.
Each started session constructs its own source, FFT plans, flow and march storage.
The PythonCall extension lowers the typed Python transfer record into this preparation.
"""
struct EnvelopePreparation{E, D, O, I, W}
    execution::E
    data::D
    operator::O
    crop_index::I
    power_weight::W
end

"""
    prepare_envelope(transfer, threads)

Convert a Python envelope transfer once through the PythonCall extension.

Every numerical array is copied into Julia-owned storage; no Python object is
retained. `threads` is a positive execution budget. Admission precedes this bridge
entry in Python and is rechecked by the typed numerical constructors.
"""
function prepare_envelope end

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

function _session_exportable(session, generation)
    state = session.state
    !state.closed && state.pending && !state.poisoned && state.generation == generation ||
        throw(ArgumentError("the session export is poisoned, stale or not pending"))
    return nothing
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
    (Base.mightalias(output, field) || Base.mightalias(output, power_weight)) &&
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
    state, buffer = session.state, session.attempts
    !state.closed && state.pending && state.generation == generation ||
        throw(ArgumentError("the attempt chunk is stale or not pending"))
    length(positions) == length(lengths) == length(norms) == length(accepted) ==
        length(elapsed) == buffer.count ||
        throw(DimensionMismatch("attempt destinations must match the pending chunk"))
    destinations = (positions, lengths, norms, accepted, elapsed)
    retained = (buffer.z, buffer.dz, buffer.norm, buffer.accepted, buffer.elapsed)
    for first in eachindex(destinations)
        any(source -> Base.mightalias(destinations[first], source), retained) &&
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

"""
    session_memory(session)

Return named retained array sizes in bytes for the first envelope session composition.

The immutable preparation is accounted separately by its owner. FFT plans retain
vendor storage in addition to these explicit arrays; their size is reported separately
using Julia's recursive size estimate rather than silently omitted.
"""
function session_memory(session)
    source, work = session.source, session.work
    flow, attempts = session.flow, session.attempts
    return (
        field = sizeof(session.field),
        stages = sizeof(work.stages),
        combination = sizeof(work.combination),
        lab = sizeof(work.lab),
        source = sizeof(work.source),
        transported = sizeof(work.transported),
        candidate = sizeof(work.candidate),
        defect = sizeof(work.defect),
        norm_parts = sizeof(work.norm_parts),
        fine_modal = sizeof(source.fine_modal),
        fine_nodal = sizeof(source.fine_nodal),
        samples = sizeof(source.samples),
        polarisation = sizeof(source.polarisation),
        source_failures = sizeof(source.failures),
        flow_exponent = sizeof(flow.exponent),
        flow_failures = sizeof(flow.failures),
        history = sizeof(session.history.readings),
        active = sizeof(session.norm.active),
        observed = sizeof(session.norm.observed),
        weights = sizeof(session.norm.weights),
        planes = sizeof(session.schedule.planes),
        breakpoints = sizeof(session.schedule.breakpoints),
        hard_kinks = sizeof(session.schedule.hard_kinks),
        replay_steps = sizeof(session.schedule.replay_steps),
        replay_edges = sizeof(session.schedule.replay_edges),
        crop = sizeof(source.padding.crop),
        uncrop = sizeof(source.padding.uncrop),
        stop = 1,
        attempt_positions = sizeof(attempts.z),
        attempt_lengths = sizeof(attempts.dz),
        attempt_norms = sizeof(attempts.norm),
        attempt_accepted = sizeof(attempts.accepted),
        attempt_elapsed = sizeof(attempts.elapsed),
        fft_plans = Base.summarysize(source.transform),
    )
end

"""
    preparation_memory(preparation)

Return named array byte counts for immutable shared envelope preparation.

All source matrices, windows, optional carrier, operator, crop map and export weights
are owned copies. Sessions refer to these arrays; their ledgers must not count them twice.
"""
function preparation_memory(preparation::EnvelopePreparation)
    data = preparation.data
    return (
        synthesis = sizeof(data.synthesis), projection = sizeof(data.projection),
        prefactor = sizeof(data.prefactor), omega_window = sizeof(data.omega_window),
        source_window = sizeof(data.source_window),
        carrier = data.carrier === nothing ? 0 : sizeof(data.carrier),
        operator = sizeof(preparation.operator.rate),
        crop_index = sizeof(preparation.crop_index),
        power_weight = sizeof(preparation.power_weight),
    )
end
