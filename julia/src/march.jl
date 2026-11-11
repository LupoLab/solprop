# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public MarchSchedule, MarchNorm, AttemptBuffer, RungeKuttaSession
public SessionReport, SessionEvent
public advance!, acknowledge!, request_snapshot!, close_session!
public terminate_session!

"""
    MarchSchedule(; planes, breakpoints, hard_kinks, stretch, step_planes,
                    replay_steps=Float64[], replay_edges=Float64[])

Own resolved observation, landing and optional fixed-replay positions in metres.

All position arrays are strictly increasing and finite; `breakpoints` has at least
two entries. `planes` contains every remaining requested plane, including the end.
`stretch >= 1` is the translated dimensionless landing allowance; `step_planes >= 0`
is the accepted-step observation cadence. Empty replay arrays select adaptive control.
Replay lengths are used as supplied, even when longer than the adaptive maximum.
"""
struct MarchSchedule
    planes::Vector{Float64}
    breakpoints::Vector{Float64}
    hard_kinks::Vector{Float64}
    stretch::Float64
    step_planes::Int
    replay_steps::Vector{Float64}
    replay_edges::Vector{Float64}
end

function _march_positions(values, name; minimum = 0)
    owned = Vector{Float64}(values)
    length(owned) >= minimum && all(isfinite, owned) && all(>(0), diff(owned)) ||
        throw(ArgumentError("$name must contain increasing finite positions"))
    return owned
end

function MarchSchedule(;
        planes, breakpoints, hard_kinks, stretch::Float64, step_planes::Int,
        replay_steps = Float64[], replay_edges = Float64[],
    )
    saved = _march_positions(planes, "planes"; minimum = 1)
    edges = _march_positions(breakpoints, "operator breakpoints"; minimum = 2)
    kinks = _march_positions(hard_kinks, "hard kinks")
    isfinite(stretch) && stretch >= 1.0 && step_planes >= 0 ||
        throw(ArgumentError("invalid landing stretch or step-plane cadence"))
    all(z -> first(edges) <= z <= last(edges), saved) &&
        all(z -> first(edges) <= z <= last(edges), kinks) ||
        throw(ArgumentError("observation and kink positions must lie inside the operator"))
    steps = Vector{Float64}(replay_steps)
    itinerary = _march_positions(replay_edges, "replay edges")
    all(h -> isfinite(h) && h > 0.0, steps) ||
        throw(ArgumentError("replay lengths must be finite and positive"))
    (isempty(steps) ? isempty(itinerary) : length(itinerary) == length(steps) + 1) ||
        throw(DimensionMismatch("replay edges and lengths disagree"))
    if !isempty(steps)
        length_sum = 0.0
        for index in eachindex(steps)
            length_sum += steps[index]
            expected = first(itinerary) + length_sum
            tolerance = max(8, length(steps)) * eps(max(length_sum, abs(expected)))
            abs(itinerary[index + 1] - expected) <= tolerance ||
                throw(ArgumentError("replay edges disagree with accumulated step lengths"))
        end
    end
    return MarchSchedule(saved, edges, kinks, stretch, step_planes, steps, itinerary)
end

"""
    MarchNorm(execution, active, observed, weights, rtol, atol)

Own the active mask, error-norm mask and positive per-frequency tolerance weights.

The masks and weights have the same nonempty frequency extent. Relative tolerance is
positive and dimensionless; absolute tolerance is nonnegative in state units. Owned
Boolean arrays use ordinary byte storage, never packed bits. This binary64 preparation
refuses invalid values before allocating numerical trial buffers.
"""
struct MarchNorm{M, W}
    active::M
    observed::M
    weights::W
    rtol::Float64
    atol::Float64
end

function MarchNorm(execution, active, observed, weights, rtol::Float64, atol::Float64)
    !isempty(active) && length(active) == length(observed) == length(weights) ||
        throw(DimensionMismatch("march masks and weights must have the same extent"))
    eltype(active) === Bool && eltype(observed) === Bool ||
        throw(ArgumentError("march masks must contain Boolean values"))
    isfinite(rtol) && rtol > 0.0 && isfinite(atol) && atol >= 0.0 &&
        all(weight -> isfinite(weight) && weight > 0.0, weights) ||
        throw(ArgumentError("invalid march tolerance or spectral weights"))
    return MarchNorm(
        upload(execution, active), upload(execution, observed),
        upload(execution, Float64.(weights)), rtol, atol,
    )
end

"""
    AttemptBuffer(capacity)

Allocate bounded storage for attempts since the previous acknowledged event.

Positions and lengths are metres; norms are dimensionless executed error readings;
elapsed times are seconds of numerical work since the previous event. Rejected
nonfinite norms are represented by positive infinity. Only the first `count` entries
are valid; exporting them copies into caller-owned storage without draining early.
"""
mutable struct AttemptBuffer
    z::Vector{Float64}
    dz::Vector{Float64}
    norm::Vector{Float64}
    accepted::Vector{Bool}
    elapsed::Vector{Float64}
    count::Int
end

function AttemptBuffer(capacity::Int)
    capacity > 0 || throw(ArgumentError("attempt capacity must be positive"))
    return AttemptBuffer(
        zeros(capacity), zeros(capacity), zeros(capacity), fill(false, capacity),
        zeros(capacity), 0,
    )
end

"""Identify a fixed session observation, yield, warning or terminal barrier."""
@enum SessionEvent::UInt8 begin
    SESSION_LAUNCH
    SESSION_PLANE
    SESSION_STEP
    SESSION_PROGRESS
    SESSION_SNAPSHOT
    SESSION_COMPLETED
    SESSION_STOPPED
    SESSION_FAILED
    SESSION_WARNING
    SESSION_BOUNDARY_STOP
end

function _session_event(kind)
    kind === :launch && return SESSION_LAUNCH
    kind === :plane && return SESSION_PLANE
    kind === :step && return SESSION_STEP
    kind === :progress && return SESSION_PROGRESS
    kind === :snapshot && return SESSION_SNAPSHOT
    kind === :completed && return SESSION_COMPLETED
    kind === :stopped && return SESSION_STOPPED
    kind === :failed && return SESSION_FAILED
    kind === :warning && return SESSION_WARNING
    kind === :boundary_stop && return SESSION_BOUNDARY_STOP
    throw(ArgumentError("unknown numerical session event"))
end

"""
    SessionReport

Describe one closed numerical event without exporting the field.

`kind` is a `SessionEvent`. `z` is metres at the selected observation for a plane,
or at the committed endpoint for every other event. Dense planes may lie behind the
committed endpoint without replacing it. `plane` is the zero-based requested plane or
-1 for launch. `generation` binds exports to this pending event. Counts are cumulative;
the plane summary spans yields since the previous observation. Norms are dimensionless
and `wall_time` is numerical work in seconds. A poisoned failure cannot export a field.
"""
struct SessionReport
    kind::SessionEvent
    z::Float64
    plane::Int
    generation::Int
    attempts::Int
    rejected::Int
    plane_attempts::Int
    plane_rejected::Int
    max_norm::Float64
    mean_norm::Float64
    wall_time::Float64
    met_tolerance::Bool
    poisoned::Bool
end

mutable struct _MarchState
    z::Float64
    proposal::Float64
    plane::Int
    replay::Int
    generation::Int
    attempts::Int
    rejected::Int
    accepted::Int
    plane_attempts::Int
    plane_rejected::Int
    plane_max::Float64
    plane_sum::Float64
    plane_work::Float64
    launched::Bool
    services_started::Bool
    pending::Bool
    finished::Bool
    closed::Bool
    snapshot::Bool
    snapshot_sequence::Int
    pending_kind::Symbol
    poisoned::Bool
    in_flight::Bool
    flight_z::Float64
    flight_dz::Float64
    failure_code::Symbol
    failure_message::String
    observation_kind::Symbol
    observation_plane::Int
    endpoint_pending::Bool
    finishing::Bool
end

function _MarchState(z, proposal, plane, replay)
    return _MarchState(
        z, proposal, plane, replay, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0,
        false, false, false, false, false, false, 0, :none, false, false, z, 0.0, :none, "",
        :none, -1, false, false,
    )
end

struct _NoSessionServices end
_session_launch!(::_NoSessionServices, session) = nothing
_session_accepted!(::_NoSessionServices, session, dz) = nothing
_session_observe!(::_NoSessionServices, session, kind, plane) = nothing
_session_finish!(::_NoSessionServices, session) = nothing
_session_quiet(::_NoSessionServices, session) = false
_session_cap(::_NoSessionServices, session, proposal::Float64) = proposal
_session_finish_candidate!(
    ::_NoSessionServices, session, candidate, dz, destination,
) = nothing
_session_quiet_interval(::_NoSessionServices, session) = nothing
_session_barrier(::_NoSessionServices, session) = :none
_session_continue!(::_NoSessionServices, session) = :none
_session_acknowledge!(::_NoSessionServices, session, generation) = nothing
_session_endpoint!(services, session) = nothing

# Concrete methods own numerical trials; this internal common surface owns event
# barriers, bounded delivery, cancellation, acknowledgment and failure persistence.
abstract type _MarchSession end

# Dense methods select an observation without replacing their committed field.
# While an event is pending, its kind takes precedence over an interrupted plane:
# warning/failure exports must still describe the eligible committed endpoint.
_march_select_observation!(session::_MarchSession, kind, plane) = nothing
_march_observed_field(session) = session.field
_march_observed_position(session) = session.state.z
_march_pending_dense(session::_MarchSession) = false
_march_defer_periodic(session) = false
_march_dense_services(services) = nothing
function _march_plane_selected(session)
    state = session.state
    return state.pending ? state.pending_kind === :plane : state.observation_kind === :plane
end

"""
    RungeKuttaSession(execution, state, z, proposal, tableau, control, norm, schedule,
                      source, flow; stop=StopFlag(), capacity=256)

Own a complete binary64 Runge–Kutta march and its bounded delivery state.

`state` has shape `(n_omega,n_columns)` in internal units and is copied. `source` and
`flow` are independently owned callable numerical compositions; they return `nothing`
or throw outside leaf kernels. The session owns its trial arrays, controller history,
accepted field, counters, cursors and attempt buffers. No Python callback runs inside
`advance!`. Stop requests are read through the lifetime-owned atomic `stop` flag.

The first event observes launch before any step. Every event must be acknowledged
before advancing again; export generations expire on acknowledgment. This first march
composition supports landed planes and field restarts; other methods/services attach
through their own typed compositions without changing its transport contract.
"""
struct RungeKuttaSession{E, A, T, W, N, S, F, B, O, X} <: _MarchSession
    execution::E
    field::A
    tableau::T
    work::W
    control::PIControl
    history::NormHistory
    norm::N
    schedule::MarchSchedule
    source::S
    flow::F
    stop::B
    attempts::AttemptBuffer
    state::_MarchState
    services::O
    exports::X
end

function RungeKuttaSession(
        execution, state::AbstractMatrix{ComplexF64}, z::Float64, proposal::Float64,
        tableau::EmbeddedTableau, control::PIControl, norm::MarchNorm,
        schedule::MarchSchedule, source, flow; stop = StopFlag(), capacity::Int = 256,
        services = _NoSessionServices(), exports = nothing,
        plane_index::Union{Nothing, Int} = nothing,
        plane_pending::Bool = true,
    )
    owned, progress = _prepare_march(
        execution, state, z, proposal, norm, schedule; plane_index, plane_pending,
    )
    return RungeKuttaSession(
        execution, owned, tableau, RungeKuttaWork(owned, tableau), control,
        NormHistory(control), norm, schedule, source, flow, stop, AttemptBuffer(capacity),
        progress, services, exports,
    )
end

function _prepare_march(
        execution, state::AbstractMatrix{ComplexF64}, z::Float64, proposal::Float64,
        norm::MarchNorm, schedule::MarchSchedule;
        plane_index::Union{Nothing, Int}, plane_pending::Bool,
    )
    all(isfinite, state) && size(state, 1) == length(norm.active) &&
        size(state, 2) > 0 || throw(ArgumentError("invalid initial march field"))
    first(schedule.breakpoints) <= z <= last(schedule.planes) &&
        isfinite(proposal) && proposal > 0.0 ||
        throw(ArgumentError("invalid starting position or proposal"))
    plane = plane_index === nothing ? searchsortedlast(schedule.planes, z) + 1 :
        plane_index + 1
    1 <= plane <= length(schedule.planes) + 1 ||
        throw(ArgumentError("the starting plane cursor is outside the schedule"))
    tolerance = isempty(schedule.replay_steps) ?
        _march_slack(z, last(schedule.planes)) : _replay_slack(schedule, z)
    (plane == 1 || schedule.planes[plane - 1] <= z + tolerance) &&
        (plane > length(schedule.planes) || schedule.planes[plane] > z + tolerance) ||
        throw(ArgumentError("the starting plane cursor disagrees with its position"))
    replay = 1
    if !isempty(schedule.replay_steps)
        replay = argmin(abs.(schedule.replay_edges .- z))
        abs(schedule.replay_edges[replay] - z) <= _replay_slack(schedule, z) ||
            throw(ArgumentError("a replay must start at one of its recorded edges"))
        last(schedule.replay_edges) >=
            last(schedule.planes) - _replay_slack(schedule, last(schedule.planes)) ||
            throw(ArgumentError("the replay does not cover the requested end"))
    end
    owned = upload(execution, state)
    active = norm.active
    each!(execution, size(owned, 1), size(owned, 2)) do row, column
        if !active[row]
            owned[row, column] = zero(eltype(owned))
        end
    end
    progress = _MarchState(z, proposal, plane, replay)
    progress.launched = !plane_pending
    return owned, progress
end

@inline _march_slack(z, target) = 8eps(max(abs(z), abs(target)))
@inline _march_reached(z, target) = z >= target - _march_slack(z, target)

function _replay_slack(schedule::MarchSchedule, position)
    total = last(schedule.replay_edges) - first(schedule.replay_edges)
    return max(8, length(schedule.replay_steps)) * eps(max(total, abs(position)))
end

function _session_reached(session, target)
    isempty(session.schedule.replay_steps) && return _march_reached(session.state.z, target)
    return session.state.z >= target - _replay_slack(session.schedule, target)
end

function _march_landing(session)
    state, schedule = session.state, session.schedule
    z = state.z
    target = schedule.planes[state.plane]
    if !isempty(schedule.replay_steps)
        state.replay <= length(schedule.replay_steps) ||
            throw(ArgumentError("the fixed itinerary ended before the requested plane"))
        destination = schedule.replay_edges[state.replay + 1]
        destination <= target + _replay_slack(schedule, target) ||
            throw(ArgumentError("the fixed itinerary crosses a requested landed plane"))
        return schedule.replay_steps[state.replay], destination
    end
    index = clamp(
        searchsortedlast(schedule.breakpoints, z), 1, length(schedule.breakpoints) - 1,
    )
    width = schedule.breakpoints[index + 1] - schedule.breakpoints[index]
    proposal = state.proposal
    interval = _session_quiet_interval(session.services, session)
    if _session_quiet(session.services, session) && interval !== nothing
        proposal = min(proposal, interval)
    end
    proposal = _session_cap(session.services, session, proposal)
    capped = min(proposal, session.control.dz_max, width)
    kink = searchsortedfirst(schedule.hard_kinks, z + _march_slack(z, target))
    ahead = kink <= length(schedule.hard_kinks) ?
        min(target, schedule.hard_kinks[kink]) : target
    gap = ahead - z
    if gap <= capped || gap <= min(schedule.stretch * capped, session.control.dz_max)
        return gap, ahead
    end
    return capped, z + capped
end

function _march_record!(session, destination, dz, norm, accepted, elapsed)
    state, buffer = session.state, session.attempts
    index = buffer.count + 1
    buffer.z[index] = destination
    buffer.dz[index] = dz
    buffer.norm[index] = isfinite(norm) ? norm : Inf
    buffer.accepted[index] = accepted
    buffer.elapsed[index] = elapsed
    buffer.count = index
    state.attempts += 1
    state.plane_attempts += 1
    if accepted
        state.accepted += 1
        state.plane_max = max(state.plane_max, norm)
        state.plane_sum += norm
    else
        state.rejected += 1
        state.plane_rejected += 1
    end
    return nothing
end

function _march_issue!(session, kind; plane = session.state.plane - 1)
    state = session.state
    state.pending = true
    state.pending_kind = kind
    state.finished = kind in (:completed, :stopped, :failed, :boundary_stop)
    state.generation += 1
    accepted = state.plane_attempts - state.plane_rejected
    return SessionReport(
        _session_event(kind), _march_observed_position(session), plane, state.generation,
        state.attempts, state.rejected,
        state.plane_attempts, state.plane_rejected, state.plane_max,
        accepted == 0 ? 0.0 : state.plane_sum / accepted, state.plane_work,
        state.plane_max <= 1.0, state.poisoned,
    )
end

"""
    terminate_session!(session, kind; failure_code=:none, failure_message="")

Replace an open member's current barrier with a group-imposed terminal event.

`kind` is `:stopped`, `:failed` or `:boundary_stop`. Retain its actual committed
position, attempt/observation buffers, history, controller, counters and in-flight
failure facts. Do not acknowledge the old event or execute a deferred numerical
continuation. The returned `SessionReport` has a new pending generation; the group
separately tracks whether the preserved chunk has already been delivered and whether
a launch established export eligibility. Reject a closed or already terminal member,
an invalid kind or exhausted generation capacity before mutation.
"""
function terminate_session!(
        session::_MarchSession, kind::Symbol;
        failure_code::Symbol = :none, failure_message::String = "",
    )
    state = session.state
    kind in (:stopped, :failed, :boundary_stop) ||
        throw(ArgumentError("group termination requires a terminal event kind"))
    !state.closed && !state.finished ||
        throw(ArgumentError("group termination requires an open nonterminal member"))
    state.generation < typemax(Int) ||
        throw(ArgumentError("session export generation capacity is exhausted"))
    if kind === :failed
        state.failure_code, state.failure_message = failure_code, failure_message
    end
    return _march_issue!(session, kind)
end

function _march_timed_issue!(
        session, kind, began, previous_work; plane = session.state.plane - 1,
    )
    session.state.plane_work = previous_work + (time_ns() - began) * 1.0e-9
    return _march_issue!(session, kind; plane)
end

function _march_service_barrier!(session, began, previous_work)
    barrier = _session_barrier(session.services, session)
    barrier === :none && return nothing
    barrier in (:warning, :boundary_stop) ||
        throw(ArgumentError("unknown numerical observation barrier"))
    return _march_timed_issue!(session, barrier, began, previous_work)
end

function _march_observation_done!(session, began, previous_work)
    state = session.state
    kind, plane = state.observation_kind, state.observation_plane
    state.observation_kind = :none
    kind === :plane && (state.plane += 1)
    return _march_timed_issue!(session, kind, began, previous_work; plane)
end

function _march_observation!(session, kind, plane, began, previous_work)
    state = session.state
    state.observation_kind, state.observation_plane = kind, plane
    _march_select_observation!(session, kind, plane)
    _session_observe!(session.services, session, kind, plane)
    barrier = _march_service_barrier!(session, began, previous_work)
    barrier === nothing || return barrier
    return _march_observation_done!(session, began, previous_work)
end

function _march_trial!(session::RungeKuttaSession, began)
    state, norm = session.state, session.norm
    dz, destination = _march_landing(session)
    state.flight_z, state.flight_dz = state.z, dz
    rk_trial!(
        session.work, session.field, state.z, dz, session.tableau,
        session.source, session.flow, session.execution,
    )
    reading = rk_error_norm!(
        session.work, session.field, norm.observed, norm.weights,
        norm.rtol, norm.atol, session.execution,
    )
    replay = !isempty(session.schedule.replay_steps)
    equivalent, accepted = replay ? (reading, isfinite(reading)) :
        control_decide(session.control, reading, dz, state.proposal)
    if replay && !accepted
        throw(DomainError(reading, "the replayed field has a nonfinite error norm"))
    end
    if !replay
        _march_record!(
            session, destination, dz, reading, accepted,
            (time_ns() - began) * 1.0e-9,
        )
    end
    if accepted
        candidate, active = session.work.candidate, norm.active
        each!(session.execution, size(candidate)...) do row, column
            active[row] || (candidate[row, column] = zero(eltype(candidate)))
        end
        _session_finish_candidate!(session.services, session, candidate, dz, destination)
        all(isfinite, candidate) ||
            throw(DomainError(NaN, "the finished Runge-Kutta candidate is nonfinite"))
        state.in_flight = true
        rk_commit!(session.field, session.work, norm.active, session.execution)
        state.z = destination
        state.in_flight = false
        record_norm!(session.history, equivalent)
        if replay
            state.replay += 1
            _march_record!(
                session, destination, dz, reading, true,
                (time_ns() - began) * 1.0e-9,
            )
        else
            if session.control.auto_init && session.history.count == 1
                proposal, replacement = control_start(
                    session.control, state.proposal, equivalent,
                )
                state.proposal = proposal
                replace_last_norm!(session.history, replacement)
            end
            state.proposal = control_propose(
                session.control, state.proposal, session.history,
            )
        end
        _session_accepted!(session.services, session, dz)
    else
        state.proposal = control_reject(session.control, state.proposal, equivalent)
    end
    return accepted
end

"""
    advance!(session; work_limit=256, deadline=0.1, replay_limit=nothing)

Advance entirely in Julia until an event, an attempt bound or a wall-clock deadline.

The attempt bound is positive; the deadline is finite nonnegative seconds. A capacity
bound may return earlier, including after rejection, without changing any numerical
decision. One `SessionReport` closes the pending event; its attempt buffer is immutable
until `acknowledge!`. Failures become terminal records with their exact in-flight facts.
Neither yields nor acknowledgments change proposals, histories or accepted fields.
An optional absolute one-based `replay_limit` stops before that next-step cursor.
It requires a fixed itinerary and a limit from the current cursor through the final
edge. Complete owed continuations, planes and snapshots before yielding at the limit;
the limit changes no trial or step and may return a progress event without new work.
"""
function advance!(
        session::_MarchSession; work_limit::Int = 256, deadline::Float64 = 0.1,
        replay_limit::Union{Nothing, Int} = nothing,
    )
    state = session.state
    state.closed && throw(ArgumentError("the session is closed"))
    state.pending && throw(ArgumentError("acknowledge the pending session event first"))
    state.finished && throw(ArgumentError("the session has ended"))
    work_limit > 0 && isfinite(deadline) && deadline >= 0.0 ||
        throw(ArgumentError("invalid session work limit or deadline"))
    if replay_limit !== nothing
        !isempty(session.schedule.replay_steps) &&
            state.replay <= replay_limit <= length(session.schedule.replay_steps) + 1 ||
            throw(ArgumentError("a replay limit must name a current or future fixed edge"))
    end
    began = time_ns()
    previous_work = state.plane_work
    try
        # A warning can interrupt an endpoint between the monitor verdict and its
        # periodic row, or a plane before its statistics. Complete that numerical
        # remainder once before any new trial, without repeating its accepted work.
        _session_continue!(session.services, session)
        barrier = _march_service_barrier!(session, began, previous_work)
        barrier === nothing || return barrier
        if state.observation_kind !== :none
            return _march_observation_done!(session, began, previous_work)
        end
        if !state.launched
            state.launched = true
            return _march_observation!(session, :launch, -1, began, previous_work)
        end
        if !state.services_started
            state.services_started = true
            _session_launch!(session.services, session)
            barrier = _march_service_barrier!(session, began, previous_work)
            barrier === nothing || return barrier
        end
        limit = min(work_limit, length(session.attempts.z))
        worked = false
        while true
            if state.endpoint_pending
                # One accepted dense window may cross many requested planes.
                # Keep its endpoint owed across every acknowledged interior plane,
                # before periodic endpoint rows, snapshots or a further trial.
                if _march_pending_dense(session)
                    return _march_observation!(
                        session, :plane, state.plane - 1, began, previous_work,
                    )
                end
                _session_endpoint!(session.services, session)
                barrier = _march_service_barrier!(session, began, previous_work)
                barrier === nothing || return barrier
                state.endpoint_pending = false
                if _session_reached(session, session.schedule.planes[state.plane])
                    plane = state.plane - 1
                    return _march_observation!(session, :plane, plane, began, previous_work)
                end
                if session.schedule.step_planes > 0 &&
                        state.accepted % session.schedule.step_planes == 0
                    return _march_observation!(
                        session, :step, state.plane - 1, began, previous_work,
                    )
                end
            end
            if state.snapshot && state.accepted > 0
                state.snapshot = false
                return _march_timed_issue!(session, :snapshot, began, previous_work)
            end
            if state.plane > length(session.schedule.planes)
                if !state.finishing
                    state.finishing = true
                    _session_finish!(session.services, session)
                    barrier = _march_service_barrier!(session, began, previous_work)
                    barrier === nothing || return barrier
                end
                return _march_timed_issue!(session, :completed, began, previous_work)
            end
            spent = (time_ns() - began) * 1.0e-9 >= deadline
            if replay_limit !== nothing && state.replay >= replay_limit
                return _march_timed_issue!(session, :progress, began, previous_work)
            end
            if worked && (session.attempts.count >= limit || spent)
                return _march_timed_issue!(session, :progress, began, previous_work)
            end
            stop_requested(session.stop) &&
                return _march_timed_issue!(session, :stopped, began, previous_work)
            state.endpoint_pending = _march_trial!(session, began)
            worked = true
            barrier = _march_service_barrier!(session, began, previous_work)
            barrier === nothing || return barrier
        end
    catch error
        # An observation hook can fail after a successful branch accounted its work.
        # Recompute from the entry baseline so that failure never counts it twice.
        state.plane_work = previous_work + (time_ns() - began) * 1.0e-9
        state.poisoned = state.in_flight
        state.failure_code = nameof(typeof(error))
        state.failure_message = sprint(showerror, error)
        return _march_issue!(session, :failed)
    end
end

"""
    acknowledge!(session, generation)

Release exactly the pending event and expire its exports without numerical work.

Stale, repeated or foreign generations refuse before buffers change. The next advance
may overwrite the bounded attempt buffer. Observation acknowledgment resets only the
plane summary; controller history, total counters and the accepted field are unchanged.
"""
function acknowledge!(session::_MarchSession, generation::Int)
    state = session.state
    !state.closed && state.pending && generation == state.generation ||
        throw(ArgumentError("the session event generation is stale or not pending"))
    _session_acknowledge!(session.services, session, generation)
    if state.pending_kind in (:launch, :plane, :step)
        state.plane_attempts = state.plane_rejected = 0
        state.plane_max = state.plane_sum = state.plane_work = 0.0
    end
    session.attempts.count = 0
    state.pending = false
    return nothing
end

"""
    request_snapshot!(session)

Latch a snapshot at the next eligible accepted endpoint and return `(sequence,coalesced)`.

Repeated outstanding requests share a sequence. A launch request waits for the first
accepted endpoint. A coincident requested or step plane precedes its snapshot barrier.
Closed or terminal sessions refuse requests. The caller owns persistence and acknowledgment.
"""
function request_snapshot!(session::_MarchSession)
    state = session.state
    !state.closed && !state.finished || throw(ArgumentError("the session cannot snapshot"))
    state.snapshot && return (state.snapshot_sequence, true)
    state.snapshot = true
    state.snapshot_sequence += 1
    return (state.snapshot_sequence, false)
end

"""
    close_session!(session)

Close a numerical session idempotently and invalidate every pending export.

The bridge detaches stop listeners before releasing this owner. Close performs no
propagation, observation or output delivery; Julia releases buffers with their owner.
"""
function close_session!(session::_MarchSession)
    session.state.closed = true
    session.state.pending = false
    return nothing
end
