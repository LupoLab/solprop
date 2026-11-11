# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public SessionObservationPolicy, ObservationServices, ObservationStatus
public ObservationBarrier, ObservationContinuation, TemporalCrossingRefusal
public TemporalSplitWorkspace, finish_temporal_split!
public observation_status, observation_rows, observation_planes, observation_notices

"""
    TemporalSplitWorkspace(execution, kind, rate, n_modes;
                           arithmetic=SourceArithmetic(Float64))

Prepare independent coarse-time buffers for one accepted scalar temporal split.

`kind` is `:field` or `:envelope`; the nonnegative binary64 `rate` has `n_t` entries
in m⁻¹ and is a power-loss coefficient. Copy it into owned storage. `n_modes` is
positive. Real field samples have spectrum `(n_t÷2+1,n_modes)`; complex envelope
samples have spectrum `(n_t,n_modes)`. Plans use the shared normalized inverse and
raw forward transforms in `source_type(arithmetic)`. Binary32 retains one real
damping-correction vector: evaluate `expm1` in binary64 and narrow each correction
once, without narrowing the rate or step first. Binary64 keeps its original expression and
allocates no damping vector. Samples remain in the state's stored units; this linear
operator does not change either physical scale. Refuse non-CPU execution, invalid
layouts or nonfinite rates.
"""
struct TemporalSplitWorkspace{T, S, F, D, A}
    rate::Vector{Float64}
    samples::S
    spectrum::Matrix{Complex{T}}
    transform::F
    damping::D
    kind::Symbol
    inactive::Bool
    arithmetic::A
end

function TemporalSplitWorkspace(
        execution::KernelExecution, kind::Symbol, rate::AbstractVector{Float64},
        n_modes::Int; arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    oncpu(execution) && kind in (:field, :envelope) && !isempty(rate) && n_modes > 0 ||
        throw(ArgumentError("temporal splitting requires a scalar CPU temporal layout"))
    all(value -> isfinite(value) && value >= 0, rate) ||
        throw(ArgumentError("temporal split power-loss rates must be finite nonnegative"))
    Base.require_one_based_indexing(rate)
    T = source_type(arithmetic)::Union{Type{Float32}, Type{Float64}}
    n_time = length(rate)
    samples = zeros(kind == :field ? T : Complex{T}, n_time, n_modes)
    spectrum = zeros(Complex{T}, kind == :field ? n_time ÷ 2 + 1 : n_time, n_modes)
    return TemporalSplitWorkspace(
        upload(execution, rate), samples, spectrum,
        TemporalFFT(execution, samples, spectrum),
        T === Float32 ? zeros(Float32, n_time) : nothing,
        kind, all(iszero, rate), arithmetic,
    )
end

function _temporal_damping!(samples, rate, ::Nothing, dz, execution)
    each!(execution, size(samples)...) do row, column
        samples[row, column] *= exp((-0.5 * rate[row]) * dz)
    end
    return nothing
end

function _temporal_damping!(samples, rate, damping::Vector{Float32}, dz, execution)
    each!(execution, length(damping), 1) do row, column
        damping[row] = Float32(expm1((-0.5 * rate[row]) * dz))
    end
    each!(execution, size(samples)...) do row, column
        samples[row, column] *= damping[row]
    end
    return nothing
end

_temporal_split_result!(spectrum, candidate, ::Nothing, execution) = nothing
function _temporal_split_result!(spectrum, candidate, ::Vector{Float32}, execution)
    # SEP33 puts transform rounding on the collar correction alone. Form the
    # final candidate in owned scratch so an overflowing addition cannot publish.
    each!(execution, size(spectrum)...) do row, column
        spectrum[row, column] = candidate[row, column] + spectrum[row, column]
    end
    return nothing
end

"""
    finish_temporal_split!(candidate, work, dz, execution)

Apply a temporal split absorber to an accepted spectral candidate.

The complex matrix matches `work.spectrum` in shape and precision; `dz` is finite
positive binary64 metres.
The binary64 graph is `F(F⁻¹(candidate) exp(-rate*dz/2))`. Binary32 uses the protected
correction `candidate + F(F⁻¹(candidate) Float32(expm1(-rate*dz/2)))`, with the
exponential correction evaluated wide before rounding. The rate is a power coefficient,
so its field exponent contains one half. This is the
same split operator as the NumPy reference's `apply_temporal_rate`: the initial and
final active-band masks belong to the caller. Field DC and, for even `n_t`, Nyquist
entries must be real before this call. No source evaluation or monitor decision occurs.

An all-zero rate preserves the candidate exactly. Other transform arithmetic uses
the reference's normalized inverse and unnormalized forward. The warmed CPU operation
allocates nothing. Return `nothing`. Refuse aliased buffers, nonfinite values, invalid
shape/step or nonreal field endpoints before writing. A nonfinite transformed result
raises `DomainError` before the candidate is replaced.
"""
function finish_temporal_split!(candidate, work::TemporalSplitWorkspace, dz, execution)
    candidate isa AbstractMatrix{eltype(work.spectrum)} &&
        size(candidate) == size(work.spectrum) ||
        throw(DimensionMismatch("temporal split candidate does not match its layout"))
    dz = Float64(dz)::Float64
    isfinite(dz) && dz > 0 && all(isfinite, candidate) ||
        throw(ArgumentError("temporal split needs a finite field and positive step"))
    Base.require_one_based_indexing(candidate)
    _transfer_writable_destination(candidate)
    _retained_alias(candidate, work) &&
        throw(ArgumentError("temporal split candidate aliases its retained storage"))
    if work.kind == :field
        for column in axes(candidate, 2)
            isreal(candidate[1, column]) &&
                (isodd(length(work.rate)) || isreal(candidate[end, column])) ||
                throw(ArgumentError("temporal split field endpoints must be real"))
        end
    end
    work.inactive && return nothing
    copyto!(work.spectrum, candidate)
    temporal_inverse!(work.samples, work.transform, work.spectrum, execution)
    _temporal_damping!(work.samples, work.rate, work.damping, dz, execution)
    temporal_forward!(work.spectrum, work.transform, work.samples)
    _temporal_split_result!(work.spectrum, candidate, work.damping, execution)
    all(isfinite, work.spectrum) ||
        throw(DomainError(dz, "temporal split produced a nonfinite spectrum"))
    copyto!(candidate, work.spectrum)
    return nothing
end

"""
    SessionObservationPolicy(prepared; period=1, can_rest=false, split_rate=nothing,
                             transverse=nothing, trace_chunk=1)

Retain immutable scalar observation policy for independently constructed sessions.

`prepared` is a modal or spatial observation preparation. `period` is positive accepted
steps from this session's load, independent of event yields. `can_rest` is the lowered
route/method eligibility for omitting an in-stage temporal absorber; it requires a
nonsplit monitor. Optional copied binary64 `split_rate` contains one nonnegative
power-loss rate per coarse sample, m⁻¹. It may be present without a monitor, but a
monitored split must use exactly its fourth region column. Optional `transverse` is
a `TransversePreparation` for the same scalar spatial map, frequency count and columns;
positive `trace_chunk` bounds its strip FFT batch. Refuse inconsistent policy or
nonfinite/negative rates. No physics object, callback or mutable source is retained.
"""
struct SessionObservationPolicy{P, R, T}
    observation::P
    period::Int
    can_rest::Bool
    split_rate::R
    transverse::T
    trace_chunk::Int
end

_observation_same_map(left::RadialObservationMap, right::RadialObservationMap) =
    left.synthesis == right.synthesis
_observation_same_map(left::CartesianObservationMap, right::CartesianObservationMap) =
    left.shape == right.shape
_observation_same_map(left, right) = false

function SessionObservationPolicy(
        prepared::_ObservationPreparation;
        period::Int = 1, can_rest::Bool = false, split_rate = nothing,
        transverse = nothing, trace_chunk::Int = 1,
    )
    period > 0 || throw(ArgumentError("observation period must be positive"))
    trace_chunk > 0 || throw(ArgumentError("transverse trace chunk must be positive"))
    if transverse !== nothing
        prepared isa SpatialObservationPreparation &&
            transverse isa TransversePreparation ||
            throw(ArgumentError("transverse monitoring requires scalar spatial inputs"))
        transverse.arithmetic == prepared.arithmetic ||
            throw(ArgumentError("transverse and observation arithmetic must agree"))
        transverse.n_frequency == length(prepared.grid.omega) &&
            transverse.n_columns == prepared.n_modes &&
            _observation_same_map(prepared.map, transverse.map) ||
            throw(DimensionMismatch("transverse and observation maps must agree"))
    end
    temporal = prepared.temporal
    can_rest && (temporal === nothing || temporal.policy.split || split_rate !== nothing) &&
        throw(ArgumentError("resting a source collar requires a nonsplit monitor"))
    if split_rate !== nothing
        split_rate isa AbstractVector{Float64} &&
            length(split_rate) == length(prepared.grid.time) ||
            throw(DimensionMismatch("split rate must match the coarse binary64 time grid"))
        all(value -> isfinite(value) && value >= 0, split_rate) ||
            throw(ArgumentError("split rates must be finite nonnegative"))
    end
    if temporal !== nothing
        temporal.policy.split == (split_rate !== nothing) ||
            throw(ArgumentError("temporal monitoring and split finishing disagree"))
        if split_rate !== nothing
            for row in eachindex(split_rate)
                split_rate[row] == temporal.weights.regions[row, 4] ||
                    throw(ArgumentError("split damping and monitor rates must match"))
            end
        end
    end
    owned = if split_rate === nothing
        nothing
    else
        result = Vector{Float64}(undef, length(split_rate))
        copyto!(result, split_rate)
    end
    return SessionObservationPolicy(
        prepared, period, can_rest, owned, transverse, trace_chunk,
    )
end

"""Identify no barrier, a warning awaiting service, or a decided boundary stop."""
@enum ObservationBarrier::UInt8 begin
    ObservationNoBarrier
    ObservationWarning
    ObservationBoundaryStop
end

"""Identify the unperformed numerical remainder after a monitor warning."""
@enum ObservationContinuation::UInt8 begin
    ObservationNoContinuation
    ObservationLaunchRest
    ObservationAcceptedRest
    ObservationPlaneRest
    ObservationFinishRest
    ObservationTransverseLaunch
    ObservationTransverseAccepted
    ObservationTransversePlane
end

"""
    TemporalCrossingRefusal(z, cap, step)

Carry a decided fixed-itinerary crossing failure without formatting a message.

All fields are metres: current committed `z`, newly read positive `cap`, and the
largest prescribed `step` before the next cadence check. A refusal preserves the
accepted field and does not shorten the itinerary. The bridge constructs BoundaryError.
"""
struct TemporalCrossingRefusal
    z::Float64
    cap::Float64
    step::Float64
end

"""
    ObservationStatus

Describe the service's closed numerical payload without consuming it.

Fields are the `barrier` and `continuation` enums, `row_count`, `monitor_count`,
`plane_valid`, `plane_z` in m, `notice_count`, actual source omission `quiet`, and
crossing `step_cap` in m (Inf when inactive). Counts delimit owned fixed buffers.
Session generation validation and detached transport belong to the caller.
"""
struct ObservationStatus
    barrier::ObservationBarrier
    continuation::ObservationContinuation
    row_count::Int
    monitor_count::Int
    plane_valid::Bool
    plane_z::Float64
    notice_count::Int
    quiet::Bool
    step_cap::Float64
end

mutable struct _ObservationRows
    z::Vector{Float64}
    step::Vector{Int}
    values::Matrix{Float64}
    count::Int
end

"""
    ObservationServices(execution, policy; attempt_capacity=256, node_chunk,
        noise_spectrum=nothing, temporal_continuation=nothing, continuation_plane=false,
        transverse_clean=nothing, transverse_continuation=nothing)

Own bounded scalar statistics, monitor payloads and their warning continuations.

`policy` is a `SessionObservationPolicy`; the other preparation keywords are those of
`ObservationWorkspace`. `attempt_capacity` matches the containing march's positive
AttemptBuffer capacity. Periodic storage has that capacity plus one for a row resumed
after its warning delivered the accepted attempt. Compact periodic slices follow the
selection's periodic order. One packed plane vector and at most two monitor plane rows
are retained until valid acknowledgment; one monitor notice causes an immediate barrier.
The optional transverse workspace uses `policy.trace_chunk`, the original input-noise
spectrum, and explicit clean columns only when its fresh ramp judgment needs them.
Transverse continuation is the already parsed numerical record; no noise is reapplied.

No field, session, source, Python callback or delivery receipt is retained. Numerical
acceptance hooks use the committed field; plane hooks use the selected observation,
which may lie inside the last accepted Adams window. `quadrature_error` is an explicit
master-owned alarm scalar, initially NaN when unavailable; it is never inferred from
stale RHS scratch. Copy-out accessors do not drain. The containing session validates
generations and owns event/plane cursors. Refuse invalid capacity or preparation.
"""
mutable struct ObservationServices{W, S, R, Q, T}
    work::W
    transverse::T
    split::S
    row_slices::R
    period::Int
    attempt_capacity::Int
    can_rest::Bool
    rows::_ObservationRows
    plane_values::Vector{Float64}
    plane_valid::Bool
    plane_z::Float64
    monitor_rows::Vector{TemporalPlaneRow}
    monitor_count::Int
    notices::MonitorNotices{Vector{MonitorNotice}}
    crossing_refusal::Union{Nothing, TemporalCrossingRefusal}
    barrier::ObservationBarrier
    continuation::ObservationContinuation
    continuation_z::Float64
    decision::@NamedTuple{quiet::Bool, step_cap::Float64}
    started::Bool
    finished::Bool
    left_out::Bool
    accepted_seen::Int
    committed_z::Float64
    quadrature_error::Float64
    quadrature_work::Q
    periodic_pending::Bool
    resume_pending::Bool
    transverse_restored::Bool

    function ObservationServices(
            work::W, transverse::T, split::S, row_slices::R,
            period::Int, attempt_capacity::Int,
            can_rest::Bool, rows::_ObservationRows, plane_values::Vector{Float64},
            quadrature_work::Q,
        ) where {W, S, R, Q, T}
        blank = TemporalPlaneRow(false, NaN, NaN, NaN, NaN, 0.0)
        return new{W, S, R, Q, T}(
            work, transverse, split, row_slices, period, attempt_capacity, can_rest, rows,
            plane_values, false, NaN, fill(blank, 2), 0,
            MonitorNotices(1), nothing, ObservationNoBarrier, ObservationNoContinuation,
            NaN, (quiet = false, step_cap = Inf), false, false, false, 0, NaN, NaN,
            quadrature_work, false, false, false,
        )
    end
end

_temporal_split_workspace(execution, data::ObservationPreparation, rate) =
    TemporalSplitWorkspace(
    execution, data.kind, rate, data.n_modes; arithmetic = data.arithmetic,
)

function ObservationServices(
        execution::KernelExecution, policy::SessionObservationPolicy;
        attempt_capacity::Int = 256, node_chunk::Int,
        noise_spectrum = nothing, temporal_continuation = nothing,
        continuation_plane::Bool = false,
        transverse_clean = nothing, transverse_continuation = nothing,
    )
    0 < attempt_capacity < typemax(Int) ||
        throw(ArgumentError("observation attempt capacity must be positive and bounded"))
    data = policy.observation
    work = ObservationWorkspace(
        execution, data; node_chunk, noise_spectrum, temporal_continuation,
        continuation_plane,
    )
    split = policy.split_rate === nothing ? nothing :
        _temporal_split_workspace(execution, data, policy.split_rate)
    transverse = policy.transverse === nothing ? nothing : TransverseWorkspace(
            execution, policy.transverse; trace_chunk = policy.trace_chunk,
            noise_spectrum, clean_columns = transverse_clean,
            continuation = transverse_continuation,
        )
    slots, selected = data.selection.slots, data.selection.periodic
    slices = map(eachindex(selected) |> Tuple) do ordinal
        slot = slots[selected[ordinal]]
        destination = 1
        for previous in 1:(ordinal - 1)
            destination += slots[selected[previous]].count
        end
        (slot.offset, destination, slot.count)
    end
    width = isempty(slices) ? 0 : last(slices)[2] + last(slices)[3] - 1
    capacity = isempty(slices) ? 0 : attempt_capacity + 1
    rows = _ObservationRows(
        zeros(capacity), zeros(Int, capacity), zeros(width, capacity), 0,
    )
    return ObservationServices(
        work, transverse, split, slices, policy.period, attempt_capacity,
        policy.can_rest, rows,
        zeros(data.selection.width),
        any(slot -> slot.code == 20, slots) ?
            QuadratureWorkspace(execution, data.n_modes; arithmetic = data.arithmetic) :
            nothing,
    )
end

"""Return an isbits `ObservationStatus` without consuming any payload."""
function observation_status(services::ObservationServices)
    return ObservationStatus(
        services.barrier, services.continuation, services.rows.count,
        services.monitor_count, services.plane_valid, services.plane_z,
        services.notices.count, services.left_out, services.decision.step_cap,
    )
end

"""
    observation_rows(services)

Return `(z,step,values,count)` for the owned periodic buffers without copying/draining.

Only the first `count` vector entries/value columns are valid. `step` is the accepted
attempt's index relative to this event chunk, including -1 for a continuation after a
warning delivered that attempt. Shared chunk assembly rebases this existing convention.
The caller validates the pending generation before copying to detached destinations.
"""
function observation_rows(services::ObservationServices)
    rows = services.rows
    return (z = rows.z, step = rows.step, values = rows.values, count = rows.count)
end

"""
    observation_planes(services)

Return `(values,valid,z,monitor_rows,monitor_count)` without copying or draining.

Plane values use the full original statistics slices. Only `monitor_count` fixed
rows are valid, and only rows whose numerical `emit` flag was true were appended.
The caller validates generation; retained external results require detached copies.
"""
function observation_planes(services::ObservationServices)
    return (
        values = services.plane_values, valid = services.plane_valid,
        z = services.plane_z, monitor_rows = services.monitor_rows,
        monitor_count = services.monitor_count,
    )
end

"""
    observation_notices(services)

Return `(records,count,crossing_refusal)` for decided boundary notices, without drain.

The monitor prefix contains at most one warning/stop. A fixed schedule crossing
refusal is a separate optional `TemporalCrossingRefusal`. The caller validates the
pending generation and formats these records; it makes no numerical decision.
"""
function observation_notices(services::ObservationServices)
    return (
        records = services.notices.records, count = services.notices.count,
        crossing_refusal = services.crossing_refusal,
    )
end

function _observation_service_owner(services, session)
    session.services === services &&
        length(session.attempts.z) == services.attempt_capacity ||
        throw(ArgumentError("observation services do not belong to this session capacity"))
    return nothing
end

function _observation_service_entry(services, session)
    _observation_service_owner(services, session)
    !session.state.pending && services.barrier == ObservationNoBarrier &&
        services.continuation == ObservationNoContinuation ||
        throw(ArgumentError("complete the pending observation service continuation first"))
    return nothing
end

function _observation_monitor_row!(services, row)
    (row === nothing || !row.emit) && return nothing
    services.monitor_count < length(services.monitor_rows) ||
        throw(DimensionMismatch("observation monitor row buffer is full"))
    services.monitor_count += 1
    services.monitor_rows[services.monitor_count] = row
    return nothing
end

function _observation_pause!(services, continuation, z)
    services.notices.count == 0 && return false
    services.barrier = monitor_stopped(services.notices) ?
        ObservationBoundaryStop : ObservationWarning
    services.continuation, services.continuation_z = continuation, z
    return true
end

function _observation_sync_collar!(services, session)
    monitor = services.work.monitor
    monitor === nothing && return nothing
    left_out = services.can_rest && services.decision.quiet
    if left_out != services.left_out
        set_temporal_source!(session.source, !left_out)
        services.left_out = left_out
    end
    temporal_note_collar!(monitor, left_out)
    return nothing
end

function _observation_crossing!(services, session, continuation)
    work, schedule = services.work, session.schedule
    monitor = work.monitor
    monitor === nothing && return false
    cursor = session.state.replay
    cap = services.decision.step_cap
    (
        isempty(schedule.replay_steps) || cursor > length(schedule.replay_steps) ||
            !isfinite(cap)
    ) && return false
    count = monitor.busy || !monitor.has_last ? 1 :
        max(work.data.temporal.policy.cadence - monitor.since_check, 1)
    stop = cursor + min(count, length(schedule.replay_steps) - cursor + 1) - 1
    longest = 0.0
    for index in cursor:stop
        longest = max(longest, schedule.replay_steps[index])
    end
    if longest > cap
        services.crossing_refusal = TemporalCrossingRefusal(session.state.z, cap, longest)
        services.barrier = ObservationBoundaryStop
        services.continuation, services.continuation_z = continuation, session.state.z
        return true
    end
    return false
end

function _observation_launch_rest!(services, session)
    _observation_sync_collar!(services, session)
    _observation_crossing!(services, session, ObservationLaunchRest) && return nothing
    work = services.transverse
    if work !== nothing
        steps = session.schedule.replay_steps
        maximum_step = isempty(steps) ? nothing : maximum(steps)
        transverse_monitor_launch!(
            work, session.field, session.state.z, services.notices;
            fixed_step_max = maximum_step,
        )
        _observation_pause!(services, ObservationTransverseLaunch, session.state.z)
    end
    return nothing
end

_source_has_temporal(source) = false
_source_has_temporal(source::ModalSource) = source.data.grid.temporal_rate !== nothing
_source_has_temporal(source::RadialSource) = _source_has_temporal(source.material)
_source_has_temporal(source::CartesianSource) = source.data.grid.temporal_rate !== nothing

function _session_launch!(services::ObservationServices, session)
    _observation_service_entry(services, session)
    !services.started || throw(ArgumentError("observation services already launched"))
    if services.can_rest
        _source_has_temporal(session.source) ||
            throw(ArgumentError("quiet observation policy requires a temporal source"))
    end
    row = observation_launch!(
        services.work, session.field, session.state.z, services.notices,
    )
    _observation_monitor_row!(services, row)
    services.started, services.committed_z = true, session.state.z
    services.decision = _observation_decision(services.work)
    _observation_pause!(services, ObservationLaunchRest, session.state.z) && return nothing
    _observation_launch_rest!(services, session)
    return nothing
end

function _session_finish_candidate!(
        services::ObservationServices, session, candidate, dz, destination,
    )
    _observation_service_entry(services, session)
    services.started && !services.finished ||
        throw(ArgumentError("candidate finishing requires active observation services"))
    services.split === nothing && return nothing
    capture_split!(services.work, candidate, destination, dz)
    finish_temporal_split!(candidate, services.split, dz, session.execution)
    return nothing
end

@inline _observation_copy_slices!(destination, source, ::Tuple{}, column) = nothing
@inline function _observation_copy_slices!(
        destination, source, slices::Tuple{NTuple{3, Int}, Vararg{NTuple{3, Int}}},
        column::Int,
    )
    first, output, count = Base.first(slices)
    for index in 0:(count - 1)
        destination[output + index, column] = source[first + index]
    end
    _observation_copy_slices!(destination, source, Base.tail(slices), column)
    return nothing
end

function _observation_periodic!(services, session)
    isempty(services.row_slices) && return nothing
    services.accepted_seen % services.period == 0 || return nothing
    rows = services.rows
    rows.count < length(rows.z) ||
        throw(DimensionMismatch("periodic observation buffer is full"))
    observe!(
        services.work, session.field, session.state.z;
        periodic = true, quadrature_error = services.quadrature_error,
    )
    index = rows.count + 1
    _observation_copy_slices!(rows.values, services.work.values, services.row_slices, index)
    rows.z[index], rows.step[index] = session.state.z, session.attempts.count - 1
    rows.count = index
    return nothing
end

function _observation_accepted_rest!(services, session)
    _observation_sync_collar!(services, session)
    _observation_crossing!(services, session, ObservationAcceptedRest) && return nothing
    work = services.transverse
    if work !== nothing
        steps, cursor = session.schedule.replay_steps, session.state.replay
        next_step = isempty(steps) ? session.state.proposal :
            cursor > length(steps) ? 0.0 : steps[cursor]
        transverse_monitor_endpoint!(
            work, session.field, session.state.z, next_step, services.notices,
        )
        _observation_pause!(services, ObservationTransverseAccepted, session.state.z) &&
            return nothing
    end
    # The first dense plane can deliver and drain this accepted attempt. Its
    # periodic row must already belong to that same chunk (SEP 47 section 5.3).
    # A warning continuation remains the shared explicit previous-chunk case.
    _session_endpoint!(services, session)
    return nothing
end

function _session_endpoint!(services::ObservationServices, session)
    _observation_service_entry(services, session)
    services.periodic_pending || return nothing
    _observation_periodic!(services, session)
    services.periodic_pending = false
    return nothing
end

function _march_dense_services(services::ObservationServices)
    services.split === nothing ||
        throw(ArgumentError("interpolated observations refuse the per-step split collar"))
    return nothing
end

function _session_accepted!(services::ObservationServices, session, dz)
    _observation_service_entry(services, session)
    state, attempts = session.state, session.attempts
    services.started && !services.finished && isfinite(dz) && dz > 0 &&
        state.z > services.committed_z && state.accepted == services.accepted_seen + 1 &&
        attempts.count > 0 && attempts.accepted[attempts.count] &&
        attempts.z[attempts.count] == state.z ||
        throw(ArgumentError("observation acceptance requires a fresh recorded endpoint"))
    _observation_quadrature!(services, session.source, session.execution)
    services.decision = observation_accepted!(
        services.work, session.field, state.z, dz, services.notices,
    )
    services.accepted_seen, services.committed_z = state.accepted, state.z
    services.periodic_pending = true
    _observation_pause!(services, ObservationAcceptedRest, state.z) && return nothing
    _observation_accepted_rest!(services, session)
    return nothing
end

_observation_quadrature!(services, source, execution) = nothing
function _observation_quadrature!(services, source::ModalSource, execution)
    work = services.quadrature_work
    work === nothing && return nothing
    services.quadrature_error = source.ready ? quadrature_error!(
            work, source.quadrature_fine, source.quadrature_coarse, execution,
        ) : NaN
    return nothing
end

function _observation_quadrature!(services, source::RadialSource, execution)
    return _observation_quadrature!(services, source.material, execution)
end

function _observation_plane_rest!(services, session)
    _observation_sync_collar!(services, session)
    _observation_crossing!(services, session, ObservationPlaneRest) && return nothing
    work = services.transverse
    if work !== nothing
        transverse_monitor_plane!(
            work, _march_observed_field(session), _march_observed_position(session),
            services.notices,
        )
        _observation_pause!(services, ObservationTransversePlane, session.state.z) &&
            return nothing
    end
    return _observation_plane_complete!(services, session)
end

function _observation_plane_complete!(services, session)
    services.monitor_count < length(services.monitor_rows) ||
        throw(DimensionMismatch("observation monitor row buffer is full"))
    position = _march_observed_position(session)
    _observation_monitor_row!(services, observation_plane!(services.work, position))
    _observation_plane_values!(services, _march_observed_field(session), position)
    return nothing
end

function _observation_plane_values!(services, field, z)
    services.plane_valid = false
    observe!(services.work, field, z; quadrature_error = services.quadrature_error)
    copyto!(services.plane_values, services.work.values)
    services.plane_z, services.plane_valid = z, true
    return nothing
end

function _session_observe!(services::ObservationServices, session, kind, plane)
    _observation_service_entry(services, session)
    if kind == :launch
        !services.started && plane == -1 ||
            throw(ArgumentError("launch statistics precede monitor attachment"))
        _observation_plane_values!(services, session.field, session.state.z)
        return nothing
    end
    kind in (:plane, :step) && plane >= 0 && services.started &&
        session.state.z == services.committed_z ||
        throw(ArgumentError("landed observations require the committed endpoint"))
    monitor = services.work.monitor
    if monitor !== nothing && monitor.stale
        services.decision = _observation_check_plane!(services, session)
        _observation_pause!(services, ObservationPlaneRest, session.state.z) &&
            return nothing
    end
    _observation_plane_rest!(services, session)
    return nothing
end

function _observation_check_plane!(services, session)
    work = services.work
    field, position = _march_observed_field(session), _march_observed_position(session)
    if position == session.state.z
        return observation_check!(work, field, position, services.notices)
    end
    # The shared/native dense-plane rule checks the interpolated field only if
    # the committed step was skipped. temporal_step!'s plane path leaves its
    # committed cursor ahead and counts the skipped step only once.
    _observation_field(work, field, position)
    monitor = work.monitor
    monitor !== nothing && monitor.stale && position < monitor.committed &&
        !work.monitor_work.captured ||
        throw(ArgumentError("a dense monitor check needs a stale committed step ahead"))
    _observation_read!(work, field)
    temporal_step!(
        monitor, work.data.temporal.policy, work.monitor_work.reading,
        position, services.notices; plane = true,
    )
    _observation_drain!(work.monitor_work.reading)
    return _observation_decision(work)
end

function _observation_finish_rest!(services, session)
    _observation_sync_collar!(services, session)
    services.finished && return nothing
    observation_finish!(services.work, services.notices)
    services.finished = true
    _observation_pause!(services, ObservationFinishRest, session.state.z)
    return nothing
end

function _session_finish!(services::ObservationServices, session)
    _observation_service_entry(services, session)
    services.started && !services.finished && session.state.z == services.committed_z ||
        throw(ArgumentError("observation finishing requires its active committed endpoint"))
    monitor = services.work.monitor
    if monitor !== nothing && monitor.stale
        services.decision = observation_check!(
            services.work, session.field, session.state.z, services.notices,
        )
        _observation_pause!(services, ObservationFinishRest, session.state.z) &&
            return nothing
    end
    _observation_finish_rest!(services, session)
    return nothing
end

_session_quiet(services::ObservationServices, session) = services.left_out
function _session_quiet_interval(services::ObservationServices, session)
    return services.left_out ? services.work.data.temporal.policy.interval : nothing
end
function _session_cap(services::ObservationServices, session, proposal::Float64)
    cap = services.transverse === nothing ? Inf :
        transverse_monitor_ceiling(services.transverse)
    return min(proposal, services.decision.step_cap, cap)
end

function _session_barrier(services::ObservationServices, session)
    return services.barrier == ObservationWarning ? :warning :
        services.barrier == ObservationBoundaryStop ? :boundary_stop : :none
end

function _observation_continuation_tag(continuation)
    continuation in (ObservationLaunchRest, ObservationTransverseLaunch) &&
        return :launch_rest
    continuation in (ObservationAcceptedRest, ObservationTransverseAccepted) &&
        return :accepted_rest
    continuation in (ObservationPlaneRest, ObservationTransversePlane) && return :plane_rest
    continuation == ObservationFinishRest && return :finish_rest
    return :none
end

function _session_continue!(services::ObservationServices, session)
    _observation_service_owner(services, session)
    !session.state.pending && services.barrier == ObservationNoBarrier ||
        throw(ArgumentError("acknowledge the observation warning before continuing"))
    continuation = services.continuation
    continuation == ObservationNoContinuation && return :none
    session.state.z == services.continuation_z ||
        throw(ArgumentError("observation continuation moved from its committed endpoint"))
    services.continuation = ObservationNoContinuation
    if continuation == ObservationLaunchRest
        _observation_launch_rest!(services, session)
    elseif continuation == ObservationAcceptedRest
        _observation_accepted_rest!(services, session)
    elseif continuation == ObservationPlaneRest
        _observation_plane_rest!(services, session)
    elseif continuation in (
            ObservationTransverseLaunch, ObservationTransverseAccepted,
            ObservationTransversePlane,
        )
        work = services.transverse
        work !== nothing || throw(ArgumentError("transverse continuation has no owner"))
        transverse_monitor_proceed!(work, services.notices)
        _observation_pause!(services, continuation, session.state.z) && return :none
        if continuation == ObservationTransverseAccepted
            _session_endpoint!(services, session)
        elseif continuation == ObservationTransversePlane
            _observation_plane_complete!(services, session)
        end
    else
        _observation_finish_rest!(services, session)
    end
    return services.barrier == ObservationNoBarrier ?
        _observation_continuation_tag(continuation) : :none
end

function _session_acknowledge!(services::ObservationServices, session, generation)
    _observation_service_owner(services, session)
    state = session.state
    !state.closed && state.pending && generation == state.generation ||
        throw(ArgumentError("the observation acknowledgment generation is not pending"))
    services.rows.count = services.monitor_count = 0
    services.plane_valid && !services.periodic_pending && (services.quadrature_error = NaN)
    services.plane_valid = false
    reset_notices!(services.notices)
    services.crossing_refusal = nothing
    services.barrier = ObservationNoBarrier
    return nothing
end
