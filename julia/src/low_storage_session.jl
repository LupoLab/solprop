# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public LowStorageSession, LowStorageCallCounts

"""
    LowStorageCallCounts()

Own separate attempted and committed source counts for the fixed RK4IP march.

`attempted_source_evaluations` increases immediately before each physical source
invocation, including one that fails. `committed_source_evaluations` increases by
four only after a step commits; this is the reference low-storage method record's
`source_evaluations` meaning, not an actual-work counter. `norm_evaluations` counts
completed protected norm reductions, including a reduction that detects divergence.
There are no logical Adams charges or source-absence shortcuts in these counts.
"""
Base.@kwdef mutable struct LowStorageCallCounts
    attempted_source_evaluations::Int = 0
    committed_source_evaluations::Int = 0
    norm_evaluations::Int = 0
end

struct _LowStorageSessionSource{S}
    source::S
    counts::LowStorageCallCounts
end

@inline function (sample::_LowStorageSessionSource)(output, field, position)
    sample.counts.attempted_source_evaluations += 1
    return sample.source(output, field, position)
end

"""
    LowStorageSession(execution, state, z, tableau::RK4IPTableau, norm, schedule,
                      source, flow; stop=StopFlag(), capacity=256,
                      services=_NoSessionServices(), exports=nothing,
                      plane_index=nothing, plane_pending=true,
                      arithmetic=SourceArithmetic(Float64))

Own a landed binary32 or binary64 low-storage RK4IP replay and bounded delivery state.

`state` is a complex matrix `(n_omega,n_columns)` in stored internal units, copied
and hard-masked at preparation. `z` is in metres.
`arithmetic` matches its precision and the known prepared source and flow. Binary32
state and absolute tolerance have already been scaled by the caller. Generic source
callables are responsible for that same scale contract. Schedule positions and step
lengths are metres. The schedule must contain a fixed replay; its supplied step
lengths and edges are authoritative, and no adaptive proposal, cap or rejection is
introduced. `norm::MarchNorm` supplies the norm and hard-band masks. The recorded
error norm is `0.0`, meaning "not estimated". This is the SEP 24 fixed RK4IP scheme,
not an embedded tableau or a dense-output integrator.

`source` and the prepared in-place `flow` are independently owned compositions using
the prepared execution policy. They return `nothing` or throw; the linear flow
supports identical input and output arrays. Each step invokes the source four times,
even when its physical response is structurally absent. `counts` distinguishes these
attempted calls from the four calls attributed to each committed step.

The session owns its state, three `LowStorageWork` matrices and bounded attempt
storage. It keeps no method history or previous-state copy. The common scheduler
owns launch, observation, warning, snapshot, stop and acknowledgment events. A
failure before numerical entry leaves the accepted field eligible. Once a step
begins, the state is in flight through source/flow work, masks and candidate finishing;
any failure there poisons it, including a failure in the first source invocation.
Accepted-step services run after the field, position and attempt record commit.

Device binary64 sessions keep all registers and pointwise work on the device;
only compact per-column norm/finiteness summaries cross to persistent host mirrors.
Device binary32 sessions refuse before allocation.

Throw `ArgumentError` before numerical advancement for unsupported execution, an adaptive
schedule, unsupported flow, invalid capacity or initial data. Throw `DimensionMismatch`
for an incompatible linear-flow shape. Advancement failures become the common
terminal failure event with their in-flight coordinate and recorded step length.
"""
struct LowStorageSession{E, A, W, N, S, F, B, O, X, P} <: _MarchSession
    execution::E
    field::A
    tableau::RK4IPTableau
    work::W
    norm::N
    schedule::MarchSchedule
    source::S
    flow::F
    stop::B
    attempts::AttemptBuffer
    state::_MarchState
    services::O
    exports::X
    counts::LowStorageCallCounts
    sampler::_LowStorageSessionSource{S}
    arithmetic::P
end

function LowStorageSession(
        execution, state::AbstractMatrix{Complex{T}}, z::Float64, tableau::RK4IPTableau,
        norm::MarchNorm, schedule::MarchSchedule, source, flow;
        stop = StopFlag(), capacity::Int = 256, services = _NoSessionServices(),
        exports = nothing, plane_index::Union{Nothing, Int} = nothing,
        plane_pending::Bool = true,
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    ) where {T <: Union{Float32, Float64}}
    oncpu(execution) || T === Float64 ||
        throw(ArgumentError("device low-storage sessions currently require binary64"))
    _march_arithmetic(state, arithmetic, source, flow)
    _flow_start(flow, tableau, z, schedule)
    isempty(schedule.replay_steps) &&
        throw(ArgumentError("low-storage RK4IP requires a fixed replay schedule"))
    _inplace_flow(flow) ||
        throw(ArgumentError("low-storage sessions require a prepared in-place linear flow"))
    capacity > 0 || throw(ArgumentError("attempt capacity must be positive"))
    _flow_shape(flow) == size(state) ||
        throw(DimensionMismatch("low-storage linear flow and state shapes disagree"))
    owned, progress = _prepare_march(
        execution, state, z, first(schedule.replay_steps), norm, schedule;
        plane_index, plane_pending,
    )
    # The common transport carries a proposal field. In a replay it is metadata for
    # the next recorded length, never an adaptive input or a replacement itinerary.
    progress.proposal = schedule.replay_steps[min(progress.replay, end)]
    counts = LowStorageCallCounts()
    return LowStorageSession(
        execution, owned, tableau,
        LowStorageWork(execution, owned, tableau; norm_mask = norm.observed),
        norm, schedule, source, flow, stop, AttemptBuffer(capacity), progress, services,
        exports, counts, _LowStorageSessionSource(source, counts), arithmetic,
    )
end

function _low_storage_session_landing(session)
    state, schedule = session.state, session.schedule
    state.replay <= length(schedule.replay_steps) ||
        throw(ArgumentError("the fixed itinerary ended before the requested plane"))
    destination = schedule.replay_edges[state.replay + 1]
    target = schedule.planes[state.plane]
    destination <= target + _replay_slack(schedule, target) ||
        throw(ArgumentError("the fixed itinerary crosses a requested landed plane"))
    return schedule.replay_steps[state.replay], destination
end

function _low_storage_session_mask!(session)
    field, active = session.field, session.norm.active
    each!(session.execution, size(field)...) do row, column
        if !active[row]
            field[row, column] = zero(eltype(field))
        end
    end
    return nothing
end

function _march_trial!(session::LowStorageSession, began)
    state = session.state
    state.flight_z, state.flight_dz = state.z, 0.0
    dz, destination = _low_storage_session_landing(session)
    state.flight_dz = dz
    # Range admission is pretrial: no source or in-place register has been entered.
    _march_scalar(real(eltype(session.field)), dz)
    _march_scalar(real(eltype(session.field)), dz / 6)
    flow = _prepare_trial_flow(
        session.flow, session.tableau, state.z, dz, destination, state.replay,
    )
    # There is no separate trial register or rollback. Even a first-stage source
    # failure has entered the reference method's in-flight contract.
    state.in_flight = true
    facts = low_storage_trial!(
        session.work, session.field, state.z, dz, session.tableau,
        session.sampler, flow, session.execution,
    )
    session.counts.norm_evaluations += facts.norm_calls
    facts.finite && isfinite(facts.state_norm) ||
        throw(DomainError(facts.state_norm, "the replayed low-storage field is nonfinite"))
    _low_storage_session_mask!(session)
    _session_finish_candidate!(
        session.services, session, session.field, dz, destination,
    )
    _source_finite!(
        session.flow.failures, session.field, session.execution, :finished_candidate,
    )
    _low_storage_session_mask!(session)
    session.counts.committed_source_evaluations += facts.source_calls
    state.z = destination
    state.replay += 1
    state.proposal = session.schedule.replay_steps[min(state.replay, end)]
    state.in_flight = false
    # An in-flight divergence has no accepted row or invented error estimate. Its
    # attempted calls and terminal flight metadata remain available independently.
    _march_record!(
        session, destination, dz, facts.error_norm, true, (time_ns() - began) * 1.0e-9,
    )
    _session_accepted!(session.services, session, dz)
    return true
end
