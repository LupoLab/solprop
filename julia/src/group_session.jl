# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public FixedScheduleGroup, GroupReport, GroupMemberReport, GroupEvent, GroupOutcome
public advance_group!, acknowledge_group!, group_report, group_member_report
public group_member_session, group_failure
public request_group_snapshot!, fail_group_member!, fail_group!, close_group!, group_memory

"""Identify a bounded progress, service, complete window or terminal group barrier."""
@enum GroupEvent::UInt8 GROUP_PROGRESS GROUP_SERVICE GROUP_WINDOW GROUP_TERMINAL

"""Identify a running, completed, stopped or failed group or member."""
@enum GroupOutcome::UInt8 GROUP_RUNNING GROUP_COMPLETED GROUP_STOPPED GROUP_FAILED

"""
    GroupReport

Describe a closed, bounded transport from a Julia-owned fixed-schedule group.

`generation` binds every member read to this pending transport. `window` is -1 for
launch, the zero-based remaining save-plane ordinal thereafter, or the plane count
for completion. `kind` distinguishes progress, an immediate service barrier, a complete
common window and group termination. `outcome` is running until group termination;
`failure_member` is the one-based primary member, or zero for an external cause.
`attempts` counts actual member attempts in this call, and `wall_time` is seconds.
A transport can end in the middle of a round; it does not assert a common position.
"""
struct GroupReport
    generation::Int
    kind::GroupEvent
    window::Int
    outcome::GroupOutcome
    failure_member::Int
    attempts::Int
    wall_time::Float64
end

"""
    GroupMemberReport

Describe one member's held event, delivery identity and export eligibility.

`available` is true only for a member participating in the pending common window.
`new_event` marks a newly closed or replaced event; `new_chunk` marks undelivered
attempt/row buffers. `chunk` is a positive per-member identity, zero before any chunk.
Replacing a held event with a terminal preserves its chunk identity, even though the
nested `report.generation` changes. Read a new chunk exactly once; a forced terminal
must not deliver an already drained chunk again. `eligible` additionally requires an
established, nonpoisoned member event. Member outcomes remain independent.
"""
struct GroupMemberReport
    available::Bool
    new_event::Bool
    new_chunk::Bool
    chunk::Int
    eligible::Bool
    outcome::GroupOutcome
    report::SessionReport
end

struct _GroupMemberSlot
    report::SessionReport
    chunk::Int
    window::Int
    new_event::Bool
    new_chunk::Bool
    established::Bool
    eligible::Bool
    outcome::GroupOutcome
end

mutable struct _GroupState
    generation::Int
    pending::Bool
    finished::Bool
    closed::Bool
    window::Int
    next_member::Int
    round_target::Int
    common_edge::Int
    round_started::Bool
    stopping::Bool
    next_window::Bool
    outcome::GroupOutcome
    failure_member::Int
    last_attempts::Int
    last_wall::Float64
    last_kind::GroupEvent
    failure_code::Symbol
    failure_message::String
end

"""
    FixedScheduleGroup(members::Tuple; stop=StopFlag(), round_steps=256)

Adopt independent, fresh entrance sessions on one fixed itinerary.

Each member is an admitted `_MarchSession` at its entrance, with its launch owed, no
pending event, method history, monitor continuation or previously attempted work. This
is the neutral SEP 47 group contract; resumed groups are not introduced here. Members
share supplied step lengths/edges and landed planes. No equality of their physical
parameter values or initial fields is imposed. Public Python preparation separately
checks the established method, shape, norm-mask/statistics and setup/device/precision
compatibility before consuming Starts; this core adds no fused-layout constraint. The
current public member construction is qualified on CPU binary64.
The group chooses every member turn in Julia. It owns no fused field or controller.

`round_steps` is a positive common-step bound. The independent monotone `stop` flag is
polled at common round edges, after launch. Member flags must be distinct, initially
clear and unlinked to external listeners; only the group flag receives external stop
requests. Stop listeners and original Start consumption remain caller responsibilities.
A caller validates neutral request compatibility before constructing any member, and
consumes original Starts only after successful group construction. Failure adopts and
closes nothing. Successful `close_group!` closes every member.

Storage is bounded by the members' existing buffers and O(member/plane-count) metadata.
Shared immutable preparation may be shared; mutable session, field and attempt owners
must be independent. Adaptive, step-plane and incompatible remaining schedules refuse
before any source call. Distinct requested planes must map to strictly increasing
positive replay edges; multiple planes inside one edge's rounding slack are refused
by name rather than producing unbounded zero-work transports. This composition uses
existing landed member sessions; dense observation admission belongs to those constructors.
"""
struct FixedScheduleGroup{M <: Tuple, S}
    members::M
    stop::S
    round_steps::Int
    replay_origins::Vector{Int}
    plane_edges::Vector{Int}
    slots::Vector{_GroupMemberSlot}
    state::_GroupState
    function FixedScheduleGroup(
            members::M, stop::S, round_steps::Int, replay_origins::Vector{Int},
            plane_edges::Vector{Int}, slots::Vector{_GroupMemberSlot}, state::_GroupState,
        ) where {M <: Tuple, S}
        return new{M, S}(
            members, stop, round_steps, replay_origins, plane_edges, slots, state,
        )
    end
end

function _group_empty_report()
    return SessionReport(
        SESSION_PROGRESS, 0.0, -1, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, true, false,
    )
end

_group_empty_history(::_MarchSession) = false
_group_empty_history(member::RungeKuttaSession) = member.history.count == 0
_group_empty_history(member::AdamsSession) =
    member.control.held == 0 && !any(member.history.occupied) &&
    all(name -> getfield(member.ledger, name) == 0, fieldnames(AdamsCallLedger))
_group_empty_history(member::LowStorageSession) =
    all(name -> getfield(member.counts, name) == 0, fieldnames(LowStorageCallCounts))
_group_fresh_services(services) = true
_group_fresh_services(services::ObservationServices) =
    services.work.monitor === nothing || !services.work.monitor.resuming

function _group_fresh_member(member)
    member isa Union{RungeKuttaSession, AdamsSession, LowStorageSession} ||
        throw(ArgumentError("the group member session type is not admitted"))
    state = member.state
    !state.closed && !state.pending && !state.finished && !state.in_flight &&
        state.generation == 0 && !state.services_started &&
        state.observation_kind === :none && !state.endpoint_pending &&
        !state.finishing && member.attempts.count == 0 && !state.launched &&
        state.attempts == state.rejected == state.accepted == 0 &&
        state.replay == state.plane == 1 && _group_empty_history(member) &&
        _group_fresh_services(member.services) ||
        throw(ArgumentError("group members require a fresh entrance without continuation"))
    schedule = member.schedule
    !isempty(schedule.replay_steps) ||
        throw(ArgumentError("a group requires a fixed replay schedule"))
    schedule.step_planes == 0 ||
        throw(ArgumentError("fixed-schedule groups do not admit step planes"))
    stop_requested(member.stop) &&
        throw(ArgumentError("a group member's private stop flag is already set"))
    1 <= state.replay <= length(schedule.replay_steps) + 1 ||
        throw(ArgumentError("the member's replay cursor is outside its itinerary"))
    state.z == first(schedule.replay_edges) ||
        throw(ArgumentError("the member is not at an accepted replay edge"))
    return nothing
end

function _group_equal_tail(left, start_left, right, start_right)
    length(left) - start_left == length(right) - start_right || return false
    for offset in 0:(length(left) - start_left)
        left[start_left + offset] == right[start_right + offset] || return false
    end
    return true
end

function _group_plane_edges(member)
    state, schedule = member.state, member.schedule
    result = Int[]
    for plane in state.plane:length(schedule.planes)
        position = schedule.planes[plane]
        edge = searchsortedfirst(schedule.replay_edges, position)
        if edge > length(schedule.replay_edges) ||
                abs(schedule.replay_edges[edge] - position) >
                _replay_slack(schedule, position)
            edge -= 1
        end
        state.replay <= edge <= length(schedule.replay_edges) &&
            abs(schedule.replay_edges[edge] - position) <=
            _replay_slack(schedule, position) ||
            throw(ArgumentError("a group save plane is not a landed replay edge"))
        relative = edge - state.replay
        relative > (isempty(result) ? 0 : last(result)) ||
            throw(
            ArgumentError(
                "group save planes must map to distinct increasing replay edges; " *
                    "separate the requested planes beyond the replay rounding slack",
            )
        )
        push!(result, relative)
    end
    return result
end

function FixedScheduleGroup(members::Tuple; stop = StopFlag(), round_steps::Int = 256)
    !isempty(members) && round_steps > 0 ||
        throw(ArgumentError("a group needs members and a positive round step bound"))
    foreach(_group_fresh_member, members)
    reference = first(members)
    before, schedule = reference.state, reference.schedule
    for (index, member) in enumerate(members)
        state, other = member.state, member.schedule
        state.z == before.z && state.launched == before.launched ||
            throw(ArgumentError("group members need matching accepted start cursors"))
        _group_equal_tail(
            schedule.replay_steps, before.replay, other.replay_steps, state.replay,
        ) && _group_equal_tail(
            schedule.replay_edges, before.replay, other.replay_edges, state.replay,
        ) && _group_equal_tail(schedule.planes, before.plane, other.planes, state.plane) ||
            throw(ArgumentError("group members need the same remaining fixed itinerary"))
        stop_address(stop) != stop_address(member.stop) ||
            throw(ArgumentError("the group stop must be independent of member stops"))
        for earlier in 1:(index - 1)
            previous = members[earlier]
            state !== previous.state && !Base.mightalias(member.field, previous.field) &&
                member.attempts !== previous.attempts &&
                stop_address(member.stop) != stop_address(previous.stop) ||
                throw(ArgumentError("group members must own independent mutable state"))
        end
    end
    origins = Int[member.state.replay for member in members]
    edges = _group_plane_edges(reference)
    window = -1
    empty_slot = _GroupMemberSlot(
        _group_empty_report(), 0, typemin(Int), false, false, false, false, GROUP_RUNNING,
    )
    state = _GroupState(
        0, false, false, false, window, 1, 0, 0, false, false, false,
        GROUP_RUNNING, 0, 0, 0.0,
        GROUP_PROGRESS, :none, "",
    )
    return FixedScheduleGroup(
        members, stop, round_steps, origins, edges,
        fill(empty_slot, length(members)), state
    )
end

# Recursion keeps heterogeneous tuples concrete at every numerical call. No member
# callback or Python object is retained, and every operation here is a singleton.
@inline function _group_apply(operation, members::Tuple, index::Int, arguments...)
    isempty(members) && throw(ArgumentError("the group member index is outside its owners"))
    if index == 1
        return operation(first(members), arguments...)
    elseif length(members) > 1
        return _group_apply(operation, Base.tail(members), index - 1, arguments...)
    end
    throw(ArgumentError("the group member index is outside its owners"))
end

function _group_check(group, generation)
    state = group.state
    !state.closed && state.pending && state.generation == generation ||
        throw(ArgumentError("the group transport is stale or not pending"))
    return nothing
end

function _group_index(group, member)
    1 <= member <= length(group.members) ||
        throw(ArgumentError("the group member index is outside its owners"))
    return member
end

function _group_outcome(kind)
    kind === SESSION_COMPLETED && return GROUP_COMPLETED
    kind === SESSION_STOPPED && return GROUP_STOPPED
    kind in (SESSION_FAILED, SESSION_BOUNDARY_STOP) && return GROUP_FAILED
    return GROUP_RUNNING
end

function _group_member_done(group, index)
    slot = group.slots[index]
    slot.outcome !== GROUP_RUNNING && return true
    slot.window == group.state.window || return false
    expected = group.state.window < 0 ? SESSION_LAUNCH :
        group.state.window == length(group.plane_edges) ? SESSION_COMPLETED : SESSION_PLANE
    return slot.report.kind === expected
end

function _group_store!(group, index, report; new_chunk = true, established = true)
    old = group.slots[index]
    chunk = new_chunk ? Base.Checked.checked_add(old.chunk, 1) : old.chunk
    group.slots[index] = _GroupMemberSlot(
        report, chunk, group.state.window, true, new_chunk || old.new_chunk,
        established || old.established,
        (established || old.established) && !report.poisoned,
        _group_outcome(report.kind),
    )
    return nothing
end

function _group_issue!(group, kind, began, attempts)
    state = group.state
    state.generation = Base.Checked.checked_add(state.generation, 1)
    state.pending = true
    state.last_attempts, state.last_wall = attempts, (time_ns() - began) * 1.0e-9
    state.finished = kind === GROUP_TERMINAL
    state.last_kind = kind
    state.next_window = kind === GROUP_WINDOW
    return GroupReport(
        state.generation, kind, state.window, state.outcome,
        state.failure_member, attempts, state.last_wall,
    )
end

function _group_advance_one(session, group, index, limit, deadline, replay_limit)
    if session.state.pending
        acknowledge!(session, session.state.generation)
    end
    return advance!(session; work_limit = limit, deadline, replay_limit)
end

_group_replay(session) = session.state.replay
_group_attempts(session) = session.state.attempts
_group_pending(session) = session.state.pending
_group_termination_ready(session) =
    !session.state.closed && !session.state.finished &&
    session.state.generation < typemax(Int)
_group_generation_ready(session) = session.state.generation <= typemax(Int) - 2

function _group_terminate_one(session, kind, code, message)
    return terminate_session!(session, kind; failure_code = code, failure_message = message)
end

function _group_end!(group, outcome, primary, code, message)
    for index in eachindex(group.slots)
        group.slots[index].outcome === GROUP_RUNNING || continue
        _group_apply(_group_termination_ready, group.members, index) ||
            throw(ArgumentError("a group member cannot issue a terminal generation"))
    end
    group.state.failure_code, group.state.failure_message = code, message
    group.state.outcome, group.state.failure_member = outcome, primary
    kind = outcome === GROUP_STOPPED ? :stopped : :failed
    for index in eachindex(group.slots)
        slot = group.slots[index]
        slot.outcome === GROUP_RUNNING || continue
        held = _group_apply(_group_pending, group.members, index)
        report = _group_apply(
            _group_terminate_one, group.members, index, kind, code, message,
        )
        _group_store!(
            group, index, report;
            new_chunk = !held && slot.established, established = false
        )
    end
    return nothing
end

function _group_failure(session)
    return session.state.failure_code, session.state.failure_message
end

function _group_reset_new!(group)
    for index in eachindex(group.slots)
        old = group.slots[index]
        group.slots[index] = _GroupMemberSlot(
            old.report, old.chunk, old.window, false, false, old.established,
            old.eligible, old.outcome,
        )
    end
    return nothing
end

"""
    advance_group!(group; work_limit=256, deadline=0.1)

Advance member turns entirely in Julia until a bounded group event closes.

`work_limit` bounds the sum of actual member attempts in this call; `deadline` is
finite nonnegative seconds across all its member turns. At least one nonpreemptible
member turn runs, even if it only observes or continues an event; within a numerical
turn the existing one-attempt minimum applies. Yields do not change fields, histories or the
supplied schedule. Warnings and snapshots return immediately for external service.
A common round can span transports; a group stop waits for its remaining members.

No pending event may be advanced or overwritten. A member's last pending barrier
remains held after group acknowledgment until its next turn. Complete windows park
all remaining members at their own save-plane barriers. Non-boundary failures end the
remaining group at actual committed positions, without rollback; boundary stops affect
one member. Return a `GroupReport`; read each member through `group_member_report`.
"""
function advance_group!(
        group::FixedScheduleGroup; work_limit::Int = 256, deadline::Float64 = 0.1,
    )
    state = group.state
    !state.closed && !state.pending && !state.finished ||
        throw(ArgumentError("the group is closed, ended or has a pending transport"))
    work_limit > 0 && isfinite(deadline) && deadline >= 0.0 ||
        throw(ArgumentError("invalid group work limit or deadline"))
    state.generation < typemax(Int) || throw(OverflowError("group generation exhausted"))
    for index in eachindex(group.slots)
        slot = group.slots[index]
        slot.outcome === GROUP_RUNNING || continue
        slot.chunk < typemax(Int) &&
            _group_apply(_group_generation_ready, group.members, index) ||
            throw(ArgumentError("group member generation or chunk capacity is exhausted"))
    end
    began = time_ns()
    attempts = 0
    completed_turn = false
    _group_reset_new!(group)
    if state.next_window
        state.window += 1
        state.next_window = false
        state.next_member = 1
    end
    while true
        if all(slot -> slot.outcome !== GROUP_RUNNING, group.slots)
            state.outcome = GROUP_COMPLETED
            return _group_issue!(group, GROUP_TERMINAL, began, attempts)
        end
        if all(index -> _group_member_done(group, index), eachindex(group.slots))
            if 0 <= state.window < length(group.plane_edges)
                state.common_edge = group.plane_edges[state.window + 1]
                state.round_target = state.common_edge
                state.round_started = false
            end
            return _group_issue!(group, GROUP_WINDOW, began, attempts)
        end
        stepping = 0 <= state.window < length(group.plane_edges)
        if stepping && !state.round_started && !state.stopping
            if stop_requested(group.stop)
                # Drain owed launch/accepted services and eligible snapshots at the
                # common edge before stopping. The replay limit admits no new trial.
                state.stopping = true
                state.round_target = state.common_edge
                state.next_member = 1
            else
                remaining = group.plane_edges[state.window + 1] - state.common_edge
                state.round_target = state.common_edge + min(group.round_steps, remaining)
            end
        end
        if state.next_member > length(group.members)
            if state.stopping
                _group_end!(group, GROUP_STOPPED, 0, :GroupStop, "the group was stopped")
                return _group_issue!(group, GROUP_TERMINAL, began, attempts)
            end
            state.common_edge = state.round_target
            state.round_started = false
            state.next_member = 1
            continue
        end
        index = state.next_member
        if _group_member_done(group, index)
            state.next_member += 1
            continue
        end
        # At most one new event per member fits a transport. Preserve its retained
        # arrays until delivery instead of copying or growing an internal history.
        if group.slots[index].new_event
            return _group_issue!(group, GROUP_PROGRESS, began, attempts)
        end
        if attempts >= work_limit ||
                (completed_turn && (time_ns() - began) * 1.0e-9 >= deadline)
            return _group_issue!(group, GROUP_PROGRESS, began, attempts)
        end
        replay_cursor = group.replay_origins[index] + state.round_target
        limit = work_limit - attempts
        if stepping
            replay = _group_apply(_group_replay, group.members, index)
            limit = min(limit, max(1, replay_cursor - replay))
        end
        previous = _group_apply(_group_attempts, group.members, index)
        report = try
            _group_apply(
                _group_advance_one, group.members, index, group, index, limit,
                max(0.0, deadline - (time_ns() - began) * 1.0e-9),
                stepping ? replay_cursor : nothing,
            )
        catch error
            made = _group_apply(_group_attempts, group.members, index) - previous
            attempts += made
            _group_end!(
                group, GROUP_FAILED, index, nameof(typeof(error)),
                sprint(showerror, error)
            )
            # An escaped call established no valid member barrier. Preserve the
            # actual poison/in-flight facts, but never claim an export for it.
            slot = group.slots[index]
            group.slots[index] = _GroupMemberSlot(
                slot.report, slot.chunk, slot.window, slot.new_event, slot.new_chunk,
                slot.established, false, slot.outcome,
            )
            return _group_issue!(group, GROUP_TERMINAL, began, attempts)
        end
        made = report.attempts - previous
        attempts += made
        completed_turn = true
        state.round_started |= made > 0
        _group_store!(group, index, report)
        if report.kind in (SESSION_WARNING, SESSION_SNAPSHOT)
            return _group_issue!(group, GROUP_SERVICE, began, attempts)
        elseif report.kind === SESSION_FAILED
            code, message = _group_apply(_group_failure, group.members, index)
            _group_end!(group, GROUP_FAILED, index, code, message)
            return _group_issue!(group, GROUP_TERMINAL, began, attempts)
        elseif report.kind === SESSION_STOPPED
            _group_end!(
                group, GROUP_FAILED, index, :MemberStop,
                "a private member stop interrupted the common group round"
            )
            return _group_issue!(group, GROUP_TERMINAL, began, attempts)
        end
        reached = stepping &&
            _group_apply(_group_replay, group.members, index) >= replay_cursor
        if _group_member_done(group, index) || reached
            state.next_member += 1
        end
    end
    return
end

"""
    acknowledge_group!(group, generation)

Release exactly one pending transport without acknowledging member barriers.

Stale, repeated or foreign generations refuse before mutation. No numerical work,
chunk reset or controller change occurs; the next Julia-owned member turn releases
that member's own held event. Group tickets expire immediately.
"""
function acknowledge_group!(group::FixedScheduleGroup, generation::Int)
    _group_check(group, generation)
    group.state.pending = false
    return nothing
end

"""
    group_report(group, generation)

Read the pending group report, including an externally imposed terminal replacement.

Invalid, closed or nonpending generations raise `ArgumentError`. The report contains
only detached scalars and changes no delivery or numerical state.
"""
function group_report(group::FixedScheduleGroup, generation::Int)
    _group_check(group, generation)
    state = group.state
    return GroupReport(
        state.generation, state.last_kind, state.window, state.outcome,
        state.failure_member, state.last_attempts, state.last_wall
    )
end

"""
    group_member_report(group, generation, member)

Read a one-based member's immutable held report under the pending group generation.

Return a `GroupMemberReport`. Members with no event or a terminal in an earlier window
have `available=false`. A forced terminal before launch is available but cannot export.
This read changes no delivery cursor.
Invalid indices or stale/closed/nonpending generations raise `ArgumentError`.
"""
function group_member_report(group::FixedScheduleGroup, generation::Int, member::Int)
    _group_check(group, generation)
    slot = group.slots[_group_index(group, member)]
    available = slot.window == group.state.window
    return GroupMemberReport(
        available, available && slot.new_event, available && slot.new_chunk,
        slot.chunk, available && slot.eligible, slot.outcome, slot.report,
    )
end

"""
    group_failure(group, generation)

Read the pending group's detached failure code, message and primary one-based member.

A primary member of zero names an external failure or common stop. Return a tuple;
no exception object or Python reference is retained. Stale transports refuse.
"""
function group_failure(group::FixedScheduleGroup, generation::Int)
    _group_check(group, generation)
    state = group.state
    return state.failure_code, state.failure_message, state.failure_member
end

_group_owner(session) = session

"""
    group_member_session(group, generation, member; exportable=false)

Borrow a held member owner for existing generation-bound chunk or export calls.

Validate the group generation and one-based identity first. The caller also passes
`group_member_report(...).report.generation` to the existing member API. The owner
remains group-owned and must never be advanced, acknowledged or closed independently.
With `exportable=true`, refuse unestablished or poisoned fields before returning it.
"""
function group_member_session(
        group::FixedScheduleGroup, generation::Int, member::Int; exportable::Bool = false,
    )
    report = group_member_report(group, generation, member)
    report.available && (!exportable || report.eligible) ||
        throw(ArgumentError("the group member has no eligible held event"))
    return _group_apply(_group_owner, group.members, member)
end

"""
    request_group_snapshot!(group, member)

Latch a one-based member's next eligible snapshot and return its sequence/coalescing pair.

An outstanding service barrier does not acknowledge or run its numerical continuation.
The member's existing plane-before-snapshot order is preserved. Closed/ended groups,
terminal members and invalid indices refuse. No arbitrary per-member stop is introduced.
"""
function request_group_snapshot!(group::FixedScheduleGroup, member::Int)
    state = group.state
    !state.closed && !state.finished || throw(ArgumentError("the group cannot snapshot"))
    index = _group_index(group, member)
    group.slots[index].outcome === GROUP_RUNNING ||
        throw(ArgumentError("the group member has ended"))
    return _group_apply(request_snapshot!, group.members, index)
end

"""
    fail_group_member!(group, generation, member)

End only a member whose held boundary warning was raised by the external warning filter.

Preserve the warning's chunk identity, committed field and pending numerical buffers;
never run its deferred statistics or acknowledge it. Other members continue. This is
not an arbitrary per-member stop. Invalid/stale/nonwarning requests refuse unchanged.
"""
function fail_group_member!(group::FixedScheduleGroup, generation::Int, member::Int)
    report = group_member_report(group, generation, member)
    report.available && report.new_event && report.report.kind === SESSION_WARNING ||
        throw(ArgumentError("only a newly held member warning can end that member"))
    ended = _group_apply(
        _group_terminate_one, group.members, member, :boundary_stop,
        :BoundaryWarning, "the boundary warning was raised by its external filter"
    )
    _group_store!(group, member, ended; new_chunk = false, established = false)
    return nothing
end

"""
    fail_group!(group, generation, code, message)

End running members after an external non-boundary service failure.

`code` and `message` are detached diagnostic scalars; no Python exception enters Julia.
The pending transport is replaced in place, preserving its generation and each existing
chunk identity. Its member reports now carry terminal events. No numerical work runs.
Read the replacement through `group_report`; group acknowledgment still releases it once.
"""
function fail_group!(
        group::FixedScheduleGroup, generation::Int, code::Symbol, message::String,
    )
    _group_check(group, generation)
    !group.state.finished || throw(ArgumentError("the group has already ended"))
    _group_end!(group, GROUP_FAILED, 0, code, message)
    group.state.finished = true
    group.state.last_kind = GROUP_TERMINAL
    group.state.next_window = false
    return nothing
end

"""
    close_group!(group)

Close every adopted member and invalidate pending group tickets, idempotently.

No propagation, service continuation or chunk delivery occurs. The caller detaches the
external group stop listener before releasing this owner.
"""
function close_group!(group::FixedScheduleGroup)
    group.state.closed && return nothing
    group.state.closed, group.state.pending = true, false
    for index in eachindex(group.slots)
        _group_apply(close_session!, group.members, index)
    end
    return nothing
end

"""
    group_memory(group)

Return named group-only retained array sizes in bytes.

Member and shared-preparation buffers remain accounted by their existing owners;
this record contains only the per-member replay origins/report slots and per-plane
round boundaries. A borrowed external stop byte is included separately. Opaque library
plans belong to the individual member ledgers, never this metadata record.
"""
function group_memory(group::FixedScheduleGroup)
    stop_bytes = group.stop.storage === nothing ? 1 : sizeof(group.stop.storage)
    return (
        replay_origins = sizeof(group.replay_origins),
        plane_edges = sizeof(group.plane_edges), slots = sizeof(group.slots),
        stop = stop_bytes,
    )
end
