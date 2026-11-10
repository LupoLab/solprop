# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public AdamsControlPolicy, AdamsOrderChoice, AdamsRejection
public AdamsLandingKind, AdamsNoLanding, AdamsMaxStep, AdamsIntervalWidth
public AdamsHardKink, AdamsSavePlane, AdamsEnd, AdamsLanding, AdamsApproach
public adams_select_stencil!, adams_factor, adams_stable_factor, adams_order_weight
public adams_choose_order, adams_rejection, adams_quiet_cap, adams_collar_reentry
public adams_landing, adams_next_landing, adams_approach, adams_replay_kink

"""
    AdamsControlPolicy(; order, variable_order, safety, max_growth, min_factor,
                         rejection_growth, hysteresis, restart_after, collar_stability,
                         real_intervals, collar_rate, carrier, carrier_cap, max_calls,
                         equal_approach, interval_cap, dz_min, dz_max, stretch, z_end,
                         save_planes, hard_kinks, breakpoints, landing_ulps,
                         interpolate, quiet_interval)

Own every translated scalar policy value of the exponential Adams march (SEP 39).

The fields match `solprop.stepper.multistep.MarchPolicy`; this record supplies no
method constant or default. `order` is in `1:6`. Safety/growth/hysteresis factors,
stability intervals and stretch are dimensionless. Positions and step bounds are in
metres; `collar_rate` is the in-stage power-loss rate divided by two, in m⁻¹.
`quiet_interval` is a positive distance or `nothing`; `max_calls` is a positive
integer or `nothing`. These two limits are retained for the session's policy, not
enforced by numerical trials. Position arrays are copied and strictly increasing;
`breakpoints` already embodies the translated interval-cap choice.

`carrier=true` refuses the carrier frame by name. No interaction formulation is
implemented. Session state, numerical fields, callbacks and history are absent.
Throw `ArgumentError` for an invalid policy field before any numerical workspace exists.
"""
struct AdamsControlPolicy{R}
    order::Int
    variable_order::Bool
    safety::Float64
    max_growth::Float64
    min_factor::Float64
    rejection_growth::Float64
    hysteresis::Float64
    restart_after::Int
    collar_stability::Float64
    real_intervals::R
    collar_rate::Float64
    carrier::Bool
    carrier_cap::Float64
    max_calls::Union{Nothing, Int}
    equal_approach::Bool
    interval_cap::Bool
    dz_min::Float64
    dz_max::Float64
    stretch::Float64
    z_end::Float64
    save_planes::Vector{Float64}
    hard_kinks::Vector{Float64}
    breakpoints::Vector{Float64}
    landing_ulps::Int
    interpolate::Bool
    quiet_interval::Union{Nothing, Float64}
end

function _adams_control_positions(points, name, minimum)
    result = Vector{Float64}(points)
    length(result) >= minimum && all(isfinite, result) && all(>(0), diff(result)) ||
        throw(ArgumentError("Adams $name must contain increasing finite positions"))
    return result
end

function AdamsControlPolicy(;
        order::Int, variable_order::Bool, safety::Float64, max_growth::Float64,
        min_factor::Float64, rejection_growth::Float64, hysteresis::Float64,
        restart_after::Int, collar_stability::Float64, real_intervals,
        collar_rate::Float64, carrier::Bool, carrier_cap::Float64,
        max_calls::Union{Nothing, Int}, equal_approach::Bool, interval_cap::Bool,
        dz_min::Float64, dz_max::Float64, stretch::Float64, z_end::Float64,
        save_planes, hard_kinks, breakpoints, landing_ulps::Int,
        interpolate::Bool, quiet_interval::Union{Nothing, Float64},
    )
    carrier && throw(ArgumentError("Julia exponential Adams refuses the carrier frame"))
    scalar_checks = (
        (:order, 1 <= order <= 6),
        (:safety, isfinite(safety) && 0 < safety <= 1),
        (:max_growth, isfinite(max_growth) && max_growth > 1),
        (:min_factor, isfinite(min_factor) && 0 < min_factor < 1),
        (:rejection_growth, isfinite(rejection_growth) && 0 < rejection_growth <= 1),
        (:hysteresis, isfinite(hysteresis) && hysteresis >= 1),
        (:restart_after, restart_after >= 1),
        (:collar_stability, isfinite(collar_stability) && collar_stability > 0),
        (:collar_rate, isfinite(collar_rate) && collar_rate >= 0),
        (:carrier_cap, isfinite(carrier_cap) && carrier_cap > 0),
        (:max_calls, max_calls === nothing || max_calls >= 1),
        (:dz_min, isfinite(dz_min) && 0 <= dz_min <= dz_max),
        (:dz_max, isfinite(dz_max) && dz_max > 0),
        (:stretch, isfinite(stretch) && stretch >= 1),
        (:z_end, isfinite(z_end)),
        (:landing_ulps, landing_ulps >= 1),
        (
            :quiet_interval, quiet_interval === nothing ||
                (isfinite(quiet_interval) && quiet_interval > 0),
        ),
    )
    for (name, valid) in scalar_checks
        valid || throw(ArgumentError("invalid Adams control policy field $name"))
    end
    intervals = Tuple(Float64(value) for value in real_intervals)
    length(intervals) >= order && all(value -> isfinite(value) && value > 0, intervals) ||
        throw(ArgumentError("Adams real stability intervals must cover every order"))
    planes = _adams_control_positions(save_planes, :save_planes, 0)
    kinks = _adams_control_positions(hard_kinks, :hard_kinks, 0)
    edges = _adams_control_positions(breakpoints, :breakpoints, 2)
    first(edges) <= z_end <= last(edges) &&
        all(z -> first(edges) <= z <= z_end, planes) &&
        all(z -> first(edges) <= z <= z_end, kinks) ||
        throw(ArgumentError("Adams landing positions must lie inside the run"))
    interpolate && !isempty(planes) &&
        throw(ArgumentError("interpolated Adams policy has no forced save planes"))
    return AdamsControlPolicy(
        order, variable_order, safety, max_growth, min_factor, rejection_growth,
        hysteresis, restart_after, collar_stability, intervals, collar_rate, carrier,
        carrier_cap, max_calls, equal_approach, interval_cap, dz_min, dz_max,
        stretch, z_end, planes, kinks, edges, landing_ulps, interpolate, quiet_interval,
    )
end

"""
    AdamsOrderChoice(stored_order, control_order, norm)

Return the order retained for the next attempt and the defect controlling its step.

The orders are positive integers and `norm` is dimensionless. Fixed-order startup
retains the requested maximum while controlling by the smaller executed order.
This is an immutable isbits decision, not mutable controller state.
"""
struct AdamsOrderChoice
    stored_order::Int
    control_order::Int
    norm::Float64
end

"""
    AdamsRejection(factor, rejections, restart)

Return a rejected attempt's step factor, consecutive count and history-restart flag.

The dimensionless `factor` multiplies the rejected step; `rejections` is reset to
zero when `restart` is true. The caller owns the actual history reset and startup call.
"""
struct AdamsRejection
    factor::Float64
    rejections::Int
    restart::Bool
end

"""Identify a cap or landing in the exact priority order of the shared landing rule."""
@enum AdamsLandingKind::UInt8 begin
    AdamsNoLanding = 0
    AdamsMaxStep = 1
    AdamsIntervalWidth = 2
    AdamsHardKink = 3
    AdamsSavePlane = 4
    AdamsEnd = 5
end

"""Keep the controller's uncapped proposal."""
AdamsNoLanding
"""Cap a proposal by the supplied maximum step."""
AdamsMaxStep
"""Cap a proposal by its containing represented interval's width."""
AdamsIntervalWidth
"""Land exactly on an operator hard kink."""
AdamsHardKink
"""Land exactly on a requested saved plane."""
AdamsSavePlane
"""Land exactly on the end of the current march."""
AdamsEnd

"""
    AdamsLanding(dz, kind, z_target, landed, stretched)

Record the shared cap/landing rule's result without mutating a march.

`dz` and a landed `z_target` are in metres. For a cap, `landed=false` and the inactive
`z_target` is exactly zero; no coordinate is inferred from that inactive field.
`stretched` reports a step lengthened to its target. The record is immutable and isbits.
"""
struct AdamsLanding
    dz::Float64
    kind::AdamsLandingKind
    z_target::Float64
    landed::Bool
    stretched::Bool
end

"""
    AdamsApproach(dz, destination, hard_kink)

Record the final Adams step length, endpoint and required hard-kink restart.

Distances are in metres. Equal approach can replace an otherwise unlanded proposal
by half the distance to the next target. The caller performs any restart after acceptance.
"""
struct AdamsApproach
    dz::Float64
    destination::Float64
    hard_kink::Bool
end

@inline function _adams_control_order(policy, order)
    1 <= order <= policy.order || throw(ArgumentError("invalid Adams control order"))
    return nothing
end

@inline function _adams_control_step(step)
    isfinite(step) && step > 0 ||
        throw(ArgumentError("Adams control requires a finite positive step"))
    return nothing
end

@inline function _adams_selection_alias(source::AbstractArray, output)
    return Base.mightalias(source, output.slots) ||
        Base.mightalias(source, output.orders) || Base.mightalias(source, output.nodes)
end

@inline _adams_selection_alias(source, output) = false

"""
    adams_select_stencil!(output, policy, positions, slots, count, stored_order;
                          z, dz, last_rejected)

Select newest-first numerical history and scored orders from explicit accepted metadata.

Binary64 `positions` and positive one-based machine-integer `slots` contain `count`
oldest-first entries in metres. `count` may reach `policy.order+1`; the caller owns
this full retained history.
`output::AdamsStencil` holds only the newest `min(count,policy.order)` entries because
no admitted predictor/defect uses the extra oldest sample. The original `count` still
controls rising-order eligibility, exactly as `adams_core.stencil`. All inactive
output cells are zero. Normalized nodes retain actual sample coordinates, including
small nonzero newest nodes after a snapped landing.

Return `nothing`. Invalid or overlapping input storage throws before changing output.
This function chooses no history identity, replacement slot or accepted-state mutation.
"""
function adams_select_stencil!(
        output::AdamsStencil, policy::AdamsControlPolicy, positions, slots,
        count::Int, stored_order::Int; z::Float64, dz::Float64, last_rejected::Bool,
    )
    _adams_control_order(policy, stored_order)
    _adams_control_step(dz)
    eltype(positions) === Float64 && eltype(slots) === Int ||
        throw(ArgumentError("Adams history positions/slots require binary64/Int storage"))
    isfinite(z) && 1 <= count <= min(length(positions), length(slots), policy.order + 1) ||
        throw(ArgumentError("invalid Adams control history extent or anchor"))
    output.max_order == policy.order ||
        throw(DimensionMismatch("Adams control and numerical stencil capacities disagree"))
    (_adams_selection_alias(positions, output) || _adams_selection_alias(slots, output)) &&
        throw(ArgumentError("Adams selection inputs must not overlap output"))
    for index in 1:count
        isfinite(positions[index]) &&
            (index == 1 || positions[index] > positions[index - 1]) &&
            isfinite((positions[index] - z) / dz) ||
            throw(ArgumentError("Adams history positions must be finite and oldest first"))
        slots[index] isa Integer && slots[index] > 0 ||
            throw(ArgumentError("Adams history slots must be positive integers"))
        for earlier in 1:(index - 1)
            slots[index] != slots[earlier] ||
                throw(ArgumentError("Adams history slots must be distinct"))
        end
    end
    order = min(stored_order, count)
    rising = !last_rejected && count > order
    selected = min(count, policy.order)
    fill!(output.slots, 0)
    fill!(output.nodes, 0)
    fill!(output.orders, 0)
    for index in 1:selected
        original = count - index + 1
        output.slots[index] = slots[original]
        output.nodes[index] = (positions[original] - z) / dz
    end
    output.order, output.count = order, selected
    output.orders[1] = order
    output.n_orders = 1
    if policy.variable_order
        for other in (order - 1, order + 1)
            if 1 <= other <= policy.order && other <= count && (other < order || rising)
                output.n_orders += 1
                output.orders[output.n_orders] = other
            end
        end
    end
    return nothing
end

"""
    adams_factor(policy, norm, order; growth)

Return the elementary defect step factor, preserving the reference operation order.

`norm` and positive `growth` are dimensionless. Nonpositive norms return `growth`.
Otherwise evaluate `safety*norm^(-1/(order+1))` with CPython's platform C power and
its ordered min/max semantics, including NaN. Nonfinite rejection handling belongs
to `adams_rejection`, which uses the supplied minimum factor before calling here.
"""
function adams_factor(
        policy::AdamsControlPolicy, norm::Float64, order::Int; growth::Float64,
    )
    _adams_control_order(policy, order)
    isfinite(growth) && growth > 0 || throw(ArgumentError("invalid Adams growth cap"))
    norm <= 0 && return growth
    raw = policy.safety * _control_pow(norm, -1.0 / (order + 1))
    return _control_min(growth, _control_max(policy.min_factor, raw))
end

"""
    adams_stable_factor(policy, order, dz; quiet)

Return the in-stage collar's stability limit as a dimensionless factor on `dz`.

For an absent or quiet collar return infinity. Otherwise use the supplied real-axis
stability interval for `order`, multiplied by the supplied stability margin and
divided by `collar_rate*dz` in that order. `dz` is finite, positive and in metres.
"""
function adams_stable_factor(
        policy::AdamsControlPolicy, order::Int, dz::Float64; quiet::Bool,
    )
    _adams_control_order(policy, order)
    _adams_control_step(dz)
    (policy.collar_rate <= 0 || quiet) && return Inf
    interval = policy.collar_stability * policy.real_intervals[order]
    return interval / (policy.collar_rate * dz)
end

"""
    adams_order_weight(policy, norm, order, dz; quiet)

Weigh an accepted trial's candidate order by defect and active-collar stability.

The dimensionless defect factor has no ordinary growth cap here. Nonpositive norms
give an infinite defect weight; the active collar may still cap it. Ordered minima
preserve the first element when a later comparison fails, including NaN.
"""
function adams_order_weight(
        policy::AdamsControlPolicy, norm::Float64, order::Int, dz::Float64; quiet::Bool,
    )
    _adams_control_order(policy, order)
    rate = norm <= 0 ? Inf : _control_pow(norm, -1.0 / (order + 1))
    allowed = policy.safety * rate
    return _control_min(allowed, adams_stable_factor(policy, order, dz; quiet))
end

"""
    adams_choose_order(policy, current, orders, norms, count, dz; quiet)

Choose a stored and control order from identical recorded error norms.

The first `count` entries of `orders` and `norms` are the one through three scored
orders in stencil order; `orders[1]==current`. Ties retain the first candidate. A
different best order is adopted only when its weight strictly exceeds the current
weight times the translated hysteresis. Fixed order stores the requested maximum
while the executed order's defect controls startup. Return `AdamsOrderChoice`.
Throw before a decision for malformed order/score storage; NaN norms remain readings.
"""
function adams_choose_order(
        policy::AdamsControlPolicy, current::Int, orders, norms, count::Int, dz::Float64;
        quiet::Bool,
    )
    _adams_control_order(policy, current)
    _adams_control_step(dz)
    1 <= count <= min(3, length(orders), length(norms)) && orders[1] == current ||
        throw(ArgumentError("Adams scored orders must start with the executed order"))
    for index in 1:count
        _adams_control_order(policy, orders[index])
        for earlier in 1:(index - 1)
            orders[index] != orders[earlier] ||
                throw(ArgumentError("Adams scored orders must be distinct"))
        end
    end
    !policy.variable_order && return AdamsOrderChoice(policy.order, current, norms[1])
    first_weight = adams_order_weight(policy, norms[1], current, dz; quiet)
    best, best_weight = 1, first_weight
    for index in 2:count
        weight = adams_order_weight(policy, norms[index], orders[index], dz; quiet)
        if weight > best_weight
            best, best_weight = index, weight
        end
    end
    if best != 1 && best_weight > policy.hysteresis * first_weight
        return AdamsOrderChoice(orders[best], orders[best], norms[best])
    end
    return AdamsOrderChoice(current, current, norms[1])
end

"""
    adams_rejection(policy, norm, order; rejections, held)

Return the factor and restart facts after a rejected trial without changing its history.

`rejections` is the previous consecutive count and `held` the full retained sample
count, including a numerical stencil's unused oldest entry. Nonfinite norms use the
supplied minimum factor. Restart occurs at the supplied rejection threshold only
when more than one sample is held; the returned consecutive count then resets to zero.
"""
function adams_rejection(
        policy::AdamsControlPolicy, norm::Float64, order::Int;
        rejections::Int, held::Int,
    )
    _adams_control_order(policy, order)
    0 <= rejections < typemax(Int) && held >= 1 ||
        throw(ArgumentError("invalid Adams rejection count or held history"))
    count = rejections + 1
    restart = count >= policy.restart_after && held > 1
    scale = isfinite(norm) ?
        adams_factor(policy, norm, order; growth = policy.rejection_growth) :
        policy.min_factor
    return AdamsRejection(scale, restart ? 0 : count, restart)
end

"""
    adams_quiet_cap(policy, proposal; quiet)

Cap a quiet collar's proposal by its supplied monitor interval, in metres.

Return the proposal unchanged without a quiet collar or without an interval. This
does not apply an ordinary maximum/floor, perform a landing or change monitor state.
"""
function adams_quiet_cap(policy::AdamsControlPolicy, proposal::Float64; quiet::Bool)
    return quiet && policy.quiet_interval !== nothing ?
        _control_min(proposal, policy.quiet_interval) : proposal
end

"""
    adams_collar_reentry(policy, proposal, order; was_quiet, quiet)

Cap the first proposal after a quiet in-stage collar returns by its stable step.

Distances are in metres. Only a true-to-false quiet transition with positive collar
rate changes the proposal. The session records the new quiet state even if unchanged.
"""
function adams_collar_reentry(
        policy::AdamsControlPolicy, proposal::Float64, order::Int;
        was_quiet::Bool, quiet::Bool,
    )
    _adams_control_order(policy, order)
    (!(was_quiet && !quiet) || policy.collar_rate <= 0) && return proposal
    interval = policy.collar_stability * policy.real_intervals[order]
    return _control_min(proposal, interval / policy.collar_rate)
end

@inline function _adams_better_bound(step, kind, previous_step, previous_kind)
    return step < previous_step || (step == previous_step && kind > previous_kind)
end

"""
    adams_landing(policy, z, proposal, target)

Apply the shared containing-interval cap, landing priorities and permitted stretch.

All coordinates/steps are in metres. The effective end is `target` for landed planes
and `policy.z_end` for dense observation. Equal distances prefer, in ascending order,
proposal, maximum step, interval width, hard kink, saved plane and end. A stretch
never exceeds the maximum step. The adaptive floor is deliberately not consulted:
a forced landing can be shorter than it. Return `AdamsLanding`.
Throw for an invalid proposal, anchor outside the operator or already reached end.
"""
function adams_landing(
        policy::AdamsControlPolicy, z::Float64, proposal::Float64, target::Float64,
    )
    _adams_control_step(proposal)
    isfinite(z) && isfinite(target) &&
        first(policy.breakpoints) <= z <= last(policy.breakpoints) ||
        throw(ArgumentError("Adams landing anchor lies outside the represented operator"))
    endpoint = policy.interpolate ? policy.z_end : target
    first(policy.breakpoints) <= endpoint <= policy.z_end ||
        throw(ArgumentError("Adams landing target lies outside the run"))
    tolerance = policy.landing_ulps * eps(_control_max(abs(z), abs(endpoint)))
    z < endpoint - tolerance ||
        throw(ArgumentError("Adams landing is already at or beyond its end"))
    index = clamp(
        searchsortedlast(policy.breakpoints, z), 1, length(policy.breakpoints) - 1
    )
    width = policy.breakpoints[index + 1] - policy.breakpoints[index]
    capped, cap_kind = proposal, AdamsNoLanding
    for (step, kind) in ((policy.dz_max, AdamsMaxStep), (width, AdamsIntervalWidth))
        if _adams_better_bound(step, kind, capped, cap_kind)
            capped, cap_kind = step, kind
        end
    end
    gap, target_kind, ahead = endpoint - z, AdamsEnd, endpoint
    for (kind, positions) in (
            (AdamsHardKink, policy.hard_kinks), (AdamsSavePlane, policy.save_planes),
        )
        found = searchsortedfirst(positions, z + tolerance)
        if found <= length(positions)
            distance = positions[found] - z
            if _adams_better_bound(distance, kind, gap, target_kind)
                gap, target_kind, ahead = distance, kind, positions[found]
            end
        end
    end
    gap <= capped && return AdamsLanding(gap, target_kind, ahead, true, false)
    if gap <= _control_min(policy.stretch * capped, policy.dz_max)
        return AdamsLanding(gap, target_kind, ahead, true, true)
    end
    return AdamsLanding(capped, cap_kind, 0.0, false, false)
end

"""
    adams_next_landing(policy, z, target)

Return the nearest forced plane/kink/end strictly beyond the landing tolerance.

Coordinates are in metres. The strict search differs deliberately from the shared
landing rule's at-or-beyond search. Return `nothing` if no candidate remains ahead.
"""
function adams_next_landing(policy::AdamsControlPolicy, z::Float64, target::Float64)
    isfinite(z) && isfinite(target) ||
        throw(ArgumentError("invalid Adams landing coordinate"))
    tolerance = policy.landing_ulps * eps(_control_max(abs(z), abs(target)))
    nearest = target
    for positions in (policy.save_planes, policy.hard_kinks)
        index = searchsortedlast(positions, z + tolerance) + 1
        index <= length(positions) && (nearest = _control_min(nearest, positions[index]))
    end
    return nearest > z + tolerance ? nearest : nothing
end

"""
    adams_approach(policy, z, stop::AdamsLanding, target)

Finish the Adams approach rule and identify a hard-kink restart after acceptance.

A landing between one and two unlanded proposals away is approached by half the
remaining distance when `equal_approach` is enabled. A saved plane coincident with a
kink still requests a restart; the run's own endpoint does not. Return `AdamsApproach`.
The session alone snaps its accepted anchor or empties its history.
"""
function adams_approach(
        policy::AdamsControlPolicy, z::Float64, stop::AdamsLanding, target::Float64,
    )
    if !stop.landed && policy.equal_approach
        ahead = adams_next_landing(policy, z, target)
        if ahead !== nothing && stop.dz < ahead - z < 2.0 * stop.dz
            half = 0.5 * (ahead - z)
            return AdamsApproach(half, z + half, false)
        end
    end
    destination = stop.landed ? stop.z_target : z + stop.dz
    tolerance = policy.landing_ulps * eps(_control_max(abs(destination), 1.0))
    interior = destination < policy.z_end - tolerance
    on_kink = interior &&
        any(kink -> abs(kink - destination) <= tolerance, policy.hard_kinks)
    return AdamsApproach(
        stop.dz, destination, stop.kind == AdamsHardKink || on_kink
    )
end

"""
    adams_replay_kink(policy, destination, slack)

Identify a hard kink within the supplied fixed-replay tolerance, in metres.

This does not adapt or cap a replay length. The caller supplies the schedule's
already computed tolerance, and owns the accepted-history restart.
"""
function adams_replay_kink(policy::AdamsControlPolicy, destination::Float64, slack::Float64)
    isfinite(destination) && isfinite(slack) && slack >= 0 ||
        throw(ArgumentError("invalid Adams replay endpoint or tolerance"))
    return any(kink -> abs(kink - destination) <= slack, policy.hard_kinks)
end
