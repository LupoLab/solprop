# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public AdamsSession, AdamsCallLedger, AdamsBudgetError, adams_charged_calls

"""
    AdamsCallLedger()

Own the cumulative logical Adams call counts and actual source invocations.

`startup`, `predictor` and `corrected` are charged calls, including structurally
absent sources and callbacks that fail before completion. Their sum controls the
budget, following `solprop.backends.multistep.CallLedger`. `rejected` is the subset
of predictor calls spent on rejected trials, not an additional charge.
`source_evaluations` counts actual source invocations immediately before entry.
Step and restart counts have their reference meanings; landed sessions keep
`corrected` and `dense_observations` zero. Accepted order rows live in the bounded
session attempt buffer, never an unbounded ledger list.
"""
Base.@kwdef mutable struct AdamsCallLedger
    startup::Int = 0
    predictor::Int = 0
    corrected::Int = 0
    rejected::Int = 0
    accepted_steps::Int = 0
    rejected_steps::Int = 0
    restarts::Int = 0
    failure_restarts::Int = 0
    dense_observations::Int = 0
    source_evaluations::Int = 0
end

"""
    adams_charged_calls(ledger::AdamsCallLedger)

Return startup, predictor and corrected logical charges, including absent sources.
"""
adams_charged_calls(ledger::AdamsCallLedger) =
    Base.Checked.checked_add(
    Base.Checked.checked_add(ledger.startup, ledger.predictor), ledger.corrected,
)

"""
    AdamsBudgetError(maximum, charged, required, position)

Report an unaffordable complete Adams attempt before starting any new source call.

Counts are nonnegative integers and `position` is the last accepted coordinate in
metres. The session retains the accepted field; the error retains no array or owner.
"""
struct AdamsBudgetError <: Exception
    maximum::Int
    charged::Int
    required::Int
    position::Float64
end

function Base.showerror(io::IO, error::AdamsBudgetError)
    return print(
        io, "Adams source-call budget ", error.maximum, " has ", error.charged,
        " charged at z = ", error.position, " m; the next trial requires ", error.required,
    )
end

mutable struct _AdamsSessionControl
    held::Int
    order::Int
    rejections::Int
    last_rejected::Bool
    collar_was_quiet::Bool
end

struct _AdamsSessionSource{S}
    source::S
    ledger::AdamsCallLedger
end

@inline function (sample::_AdamsSessionSource)(output, field, position)
    sample.ledger.source_evaluations =
        Base.Checked.checked_add(sample.ledger.source_evaluations, 1)
    return sample.source(output, field, position)
end

struct _AdamsSessionExponent{F}
    flow::F
end

function (sample::_AdamsSessionExponent)(output, origin, destination)
    flow = sample.flow
    operator_exponent!(
        output, flow.operator, origin, destination, flow.execution; work = flow.work,
    )
    _source_finite!(flow.failures, output, flow.execution, :exponent)
    return nothing
end

struct _AdamsSessionRate{F}
    flow::F
end

mutable struct _AdamsSessionObservation{W}
    work::W
    z::Float64
    selected::Bool
end

function (sample::_AdamsSessionRate)(output, position, before)
    flow = sample.flow
    operator_rate!(
        output, flow.operator, position, flow.execution; before, work = flow.work,
    )
    _source_finite!(flow.failures, output, flow.execution, :operator_rate)
    return nothing
end

_adams_varying_operator(::DiagonalOperator) = false
_adams_varying_operator(::Union{RepresentedOperator, ExpansionOperator}) = true
_adams_varying_operator(operator) =
    throw(ArgumentError("Adams requires a qualified diagonal cumulative operator"))

"""
    AdamsSession(execution, state, z, proposal, policy, norm, schedule, source, flow;
                 stop=StopFlag(), capacity=256, services=_NoSessionServices(),
                 exports=nothing, plane_index=nothing, plane_pending=true,
                 source_absent=false)

Own a binary64 exponential Adams session and bounded delivery state.

`state` is a complex binary64 matrix `(n_omega,n_columns)` in internal units; `z`
and `proposal` are metres. The launch is copied and hard-masked. `policy` is the
translated `AdamsControlPolicy`; `norm` and `schedule` have the shared march meanings.
The policy and schedule retain identical hard-kink/operator breakpoint arrays and
landing stretch. Landed policy save planes equal the schedule; interpolating policy
save planes are empty, while the schedule retains every requested observation.
Restart import changes the cursor, not those authoritative request arrays.
`source` and `flow::LinearFlow` are independently owned numerical compositions.
Constant, represented-table and expansion diagonal operators are admitted on CPU.
Varying operators use transported compensation unless `source_absent=true`, which
declares an exactly homogeneous source and skips all source invocations.

The session owns `history`, `work`, `stencil`, three `norms`, and oldest-first
`positions`/`slots` of capacity `policy.order+1`; `control.held` is their valid extent.
History keeps each actual predictor sample and coordinate, including a coordinate
that differs by rounding from its snapped accepted endpoint. `orders` aligns with
the valid prefix of `attempts`, whose accepted flags identify accepted order rows.
Acknowledgment permits reusing that fixed storage without changing decisions.

`ledger` distinguishes logical budget charges from actual source evaluations.
Every completed rejected trial is recorded before a floor failure, matching the
shared Julia actual-work convention. Candidate-only failures retain an eligible
accepted field; a failure during accepted history/field mutation poisons the session.
Services supply scalar quiet/cap decisions and candidate finishing through the
shared march hooks. Interpolating sessions own one retained accepted dense window,
observed separately from the committed state. Each crossed interior plane is delivered
before endpoint periodic rows or snapshots, without another source evaluation.
`ledger.dense_observations` counts each successful strict interior evaluation once;
canonical endpoint observations copy the committed field and do not increment it.

Throw `ArgumentError` for interpolation with replay or a per-step split collar,
unsupported execution/flow, inconsistent run topology or incompatible initial arrays.
Advance errors become the shared terminal failure event; no Python callback runs
inside a trial.
"""
struct AdamsSession{E, A, P, H, W, N, S, F, B, O, X, D} <: _MarchSession
    execution::E
    field::A
    policy::P
    history::H
    work::W
    stencil::AdamsStencil
    norm::N
    schedule::MarchSchedule
    source::S
    flow::F
    stop::B
    attempts::AttemptBuffer
    state::_MarchState
    services::O
    exports::X
    control::_AdamsSessionControl
    ledger::AdamsCallLedger
    positions::Vector{Float64}
    slots::Vector{Int}
    norms::Vector{Float64}
    orders::Vector{Int}
    sampler::_AdamsSessionSource{S}
    exponent::_AdamsSessionExponent{F}
    rate::_AdamsSessionRate{F}
    source_absent::Bool
    observation::D
end

function AdamsSession(
        execution, state::AbstractMatrix{ComplexF64}, z::Float64, proposal::Float64,
        policy::AdamsControlPolicy, norm::MarchNorm, schedule::MarchSchedule, source, flow;
        stop = StopFlag(), capacity::Int = 256, services = _NoSessionServices(),
        exports = nothing, plane_index::Union{Nothing, Int} = nothing,
        plane_pending::Bool = true, source_absent::Bool = false,
    )
    oncpu(execution) || throw(ArgumentError("Adams sessions currently require CPU"))
    policy.interpolate && !isempty(schedule.replay_steps) &&
        throw(ArgumentError("interpolated planes require adaptive Adams; replay lands"))
    flow isa LinearFlow || throw(ArgumentError("Adams sessions require LinearFlow"))
    capacity > 0 || throw(ArgumentError("attempt capacity must be positive"))
    policy.z_end == last(schedule.planes) ||
        throw(ArgumentError("Adams policy and schedule ends disagree"))
    matching_planes = policy.interpolate ?
        isempty(policy.save_planes) : policy.save_planes == schedule.planes
    matching_planes && policy.hard_kinks == schedule.hard_kinks &&
        policy.breakpoints == schedule.breakpoints && policy.stretch == schedule.stretch ||
        throw(ArgumentError("Adams policy and schedule topology disagree"))
    size(flow.exponent) == size(state) ||
        throw(DimensionMismatch("Adams linear flow and state shapes disagree"))
    compensated = _adams_varying_operator(flow.operator) && !source_absent
    policy.interpolate && _march_dense_services(services)
    owned, progress = _prepare_march(
        execution, state, z, proposal, norm, schedule; plane_index, plane_pending,
    )
    ledger = AdamsCallLedger()
    order = policy.variable_order ? 1 : policy.order
    # AdamsObservationWork is included after this method definition. Preparation
    # runs after module loading, so no include cycle or mutable global is needed.
    observation = policy.interpolate ? _AdamsSessionObservation(
            AdamsObservationWork(execution, owned, policy.order; compensated), NaN, false,
        ) : nothing
    return AdamsSession(
        execution, owned, policy,
        AdamsHistory(execution, owned, policy.order; compensated),
        AdamsWork(execution, owned, policy.order; compensated), AdamsStencil(policy.order),
        norm, schedule, source, flow, stop, AttemptBuffer(capacity), progress, services,
        exports, _AdamsSessionControl(0, order, 0, false, false), ledger,
        zeros(policy.order + 1), zeros(Int, policy.order + 1),
        zeros(3), zeros(Int, capacity),
        _AdamsSessionSource(source, ledger), _AdamsSessionExponent(flow),
        _AdamsSessionRate(flow), source_absent, observation,
    )
end

_march_defer_periodic(session::AdamsSession) = session.observation !== nothing

function _march_pending_dense(session::AdamsSession)
    return session.observation !== nothing &&
        session.state.plane <= length(session.schedule.planes) &&
        session.schedule.planes[session.state.plane] < session.state.z
end

function _march_select_observation!(session::AdamsSession, kind, plane)
    observation = session.observation
    observation === nothing && return nothing
    observation.selected = false
    kind === :plane || return nothing
    position = session.schedule.planes[plane + 1]
    # Canonical endpoints use the exact committed field. The shared reached rule
    # may also classify a nearby plane ahead as landed; preserve that existing
    # committed-coordinate behavior rather than extrapolating a dense polynomial.
    position < session.state.z || return nothing
    counted = Base.Checked.checked_add(session.ledger.dense_observations, 1)
    observe_adams!(observation.work, position, session.norm.active, session.execution)
    observation.z, observation.selected = position, true
    session.ledger.dense_observations = counted
    return nothing
end

function _march_observed_field(session::AdamsSession)
    observation = session.observation
    if observation !== nothing && observation.selected && _march_plane_selected(session)
        return observation.work.field
    end
    return session.field
end

function _march_observed_position(session::AdamsSession)
    observation = session.observation
    if observation !== nothing && observation.selected && _march_plane_selected(session)
        return observation.z
    end
    return session.state.z
end

function _adams_session_reset!(session)
    session.control.held = 0
    fill!(session.positions, 0)
    fill!(session.slots, 0)
    fill!(session.history.occupied, false)
    return nothing
end

_adams_session_cost(session) = session.control.held == 0 ? 2 : 1

function _adams_session_afford(session)
    cost = _adams_session_cost(session)
    maximum = session.policy.max_calls
    charged = adams_charged_calls(session.ledger)
    if maximum !== nothing && charged > maximum - cost
        throw(AdamsBudgetError(maximum, charged, cost, session.state.z))
    end
    # A portable ledger may be arbitrarily close to Int's limit. Refuse before
    # source work if either trial outcome could no longer be recorded (#658).
    Base.Checked.checked_add(charged, cost)
    ledger = session.ledger
    for counter in (
            ledger.rejected, ledger.accepted_steps, ledger.rejected_steps,
            ledger.restarts, ledger.failure_restarts,
        )
        Base.Checked.checked_add(counter, 1)
    end
    session.source_absent || Base.Checked.checked_add(ledger.source_evaluations, cost)
    return nothing
end

function _adams_session_start!(session)
    session.ledger.startup = Base.Checked.checked_add(session.ledger.startup, 1)
    adams_seed!(
        session.history, 1, session.field, session.state.z,
        session.sampler, session.rate, session.execution;
        identity = 0, # Portable history identifies the accepted or predicted sample kind.
        source_absent = session.source_absent,
    )
    session.positions[1] = session.state.z
    session.slots[1] = 1
    session.control.held = 1
    session.control.order = session.policy.variable_order ? 1 : session.policy.order
    return nothing
end

function _adams_session_landing(session)
    state, policy, schedule = session.state, session.policy, session.schedule
    if !isempty(schedule.replay_steps)
        state.replay <= length(schedule.replay_steps) ||
            throw(ArgumentError("the fixed itinerary ended before the requested plane"))
        dz = schedule.replay_steps[state.replay]
        destination = schedule.replay_edges[state.replay + 1]
        target = schedule.planes[state.plane]
        destination <= target + _replay_slack(schedule, target) ||
            throw(ArgumentError("the fixed itinerary crosses a requested landed plane"))
        kink = adams_replay_kink(policy, destination, _replay_slack(schedule, destination))
        return AdamsApproach(dz, destination, kink)
    end
    quiet = _session_quiet(session.services, session)
    proposal = adams_quiet_cap(policy, state.proposal; quiet)
    proposal = _session_cap(session.services, session, proposal)
    proposal = adams_collar_reentry(
        policy, proposal, session.control.order;
        was_quiet = session.control.collar_was_quiet, quiet,
    )
    session.control.collar_was_quiet = quiet
    target = schedule.planes[state.plane]
    stop = adams_landing(policy, state.z, proposal, target)
    return adams_approach(policy, state.z, stop, target)
end

function _adams_session_record!(session, landing, reading, accepted, began)
    index = session.attempts.count + 1
    session.orders[index] = session.stencil.order
    return _march_record!(
        session, landing.destination, landing.dz, reading, accepted,
        (time_ns() - began) * 1.0e-9,
    )
end

function _adams_session_limit(policy, step, factor)
    proposed = _control_min(step * factor, policy.dz_max)
    proposed < policy.dz_min && throw(StepFloorError(proposed, policy.dz_min))
    return proposed
end

function _adams_session_reject!(session, landing, reading, began)
    ledger, control, policy = session.ledger, session.control, session.policy
    rejected = Base.Checked.checked_add(ledger.rejected, 1)
    rejected_steps = Base.Checked.checked_add(ledger.rejected_steps, 1)
    outcome = adams_rejection(
        policy, reading, session.stencil.order;
        rejections = control.rejections, held = control.held,
    )
    failure_restarts = Base.Checked.checked_add(
        ledger.failure_restarts, Int(outcome.restart),
    )
    ledger.rejected, ledger.rejected_steps = rejected, rejected_steps
    control.last_rejected = true
    control.rejections = outcome.rejections
    if outcome.restart
        _adams_session_reset!(session)
        ledger.failure_restarts = failure_restarts
    end
    # Host/native raise a rejection floor before their outer attempt record (#651).
    # Keep its ledger counts, but retain this completed trial in the common event
    # record before reporting the floor, as the Julia RK session already does.
    _adams_session_record!(session, landing, reading, false, began)
    session.state.proposal = _adams_session_limit(policy, landing.dz, outcome.factor)
    return nothing
end

function _adams_session_mask!(field, active, execution)
    each!(execution, size(field)...) do row, column
        if !active[row]
            field[row, column] = zero(eltype(field))
        end
    end
    return nothing
end

function _adams_session_free(session, keep, offset)
    for candidate in eachindex(session.history.positions)
        used = false
        for index in 1:keep
            used |= session.slots[offset + index] == candidate
        end
        !used && return candidate
    end
    error("Adams history has no free prepared slot")
end

function _adams_session_commit!(session, landing, choice)
    control, history, policy = session.control, session.history, session.policy
    accepted_steps = Base.Checked.checked_add(session.ledger.accepted_steps, 1)
    restarts = Base.Checked.checked_add(session.ledger.restarts, Int(landing.hard_kink))
    keep = min(control.held, policy.order)
    offset = control.held - keep
    free = _adams_session_free(session, keep, offset)
    session.state.in_flight = true
    adams_accept!(
        history, free, session.work, session.execution;
        identity = 1,
    )
    fill!(history.occupied, false)
    for index in 1:keep
        session.positions[index] = session.positions[offset + index]
        session.slots[index] = session.slots[offset + index]
        history.occupied[session.slots[index]] = true
    end
    session.positions[keep + 1] = session.work.position + session.work.step
    session.slots[keep + 1] = free
    history.occupied[free] = true
    control.held = keep + 1
    field, candidate, active = session.field, session.work.candidate, session.norm.active
    each!(session.execution, size(field)...) do row, column
        field[row, column] = active[row] ? candidate[row, column] : zero(eltype(field))
    end
    session.ledger.accepted_steps = accepted_steps
    control.rejections = 0
    control.order = choice.stored_order
    control.last_rejected = false
    session.state.z = landing.destination
    if landing.hard_kink
        _adams_session_reset!(session)
        session.ledger.restarts = restarts
    end
    session.state.in_flight = false
    return nothing
end

function _march_trial!(session::AdamsSession, began)
    # Budget and startup failures precede a planned trial, even after a restart.
    # Do not describe them using an earlier attempt's origin and step length.
    session.state.flight_z, session.state.flight_dz = session.state.z, 0.0
    _adams_session_afford(session)
    session.control.held == 0 && _adams_session_start!(session)
    landing = _adams_session_landing(session)
    state, norm, stencil = session.state, session.norm, session.stencil
    state.flight_z, state.flight_dz = state.z, landing.dz
    adams_select_stencil!(
        stencil, session.policy, session.positions, session.slots,
        session.control.held, session.control.order;
        z = state.z, dz = landing.dz, last_rejected = session.control.last_rejected,
    )
    session.ledger.predictor = Base.Checked.checked_add(session.ledger.predictor, 1)
    adams_trial!(
        session.work, session.field, state.z, landing.dz, session.history, stencil,
        session.sampler, session.exponent, session.rate, session.execution;
        source_absent = session.source_absent,
    )
    adams_error_norms!(
        session.norms, session.work, session.field, norm.observed, norm.weights,
        norm.rtol, norm.atol, session.execution,
    )
    reading = session.norms[1]
    replay = !isempty(session.schedule.replay_steps)
    accepted = reading <= 1.0 || (replay && isfinite(reading))
    if !accepted
        _adams_session_reject!(session, landing, reading, began)
        replay && throw(DomainError(reading, "the replayed Adams norm is nonfinite"))
        return false
    end
    candidate = session.work.candidate
    _adams_session_mask!(candidate, norm.active, session.execution)
    _session_finish_candidate!(
        session.services, session, candidate, landing.dz, landing.destination,
    )
    _source_finite!(
        session.flow.failures, candidate, session.execution, :finished_candidate,
    )
    if session.observation !== nothing
        session.observation.selected = false
        retain_adams_observation!(session.observation.work, session, landing.destination)
    end
    choice = adams_choose_order(
        session.policy, stencil.order, stencil.orders, session.norms,
        stencil.n_orders, landing.dz; quiet = _session_quiet(session.services, session),
    )
    _adams_session_commit!(session, landing, choice)
    replay && (state.replay += 1)
    _adams_session_record!(session, landing, reading, true, began)
    factor = adams_factor(
        session.policy, choice.norm, choice.control_order;
        growth = session.policy.max_growth,
    )
    # A floor discovered here leaves this accepted state, history and attempt row
    # committed, exactly as the reference's deferred accepted-floor exception.
    state.proposal = _adams_session_limit(session.policy, landing.dz, factor)
    _session_accepted!(session.services, session, landing.dz)
    return true
end
