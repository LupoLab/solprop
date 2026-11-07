# The exponential Adams march, executed by the engine (native's SEP 39 phase 2 march).
#
# solprop's `AdamsMarch` and its controller core (`solprop.backends.adams_core`) are
# the reference, and native's `adams_march.cpp` the transcription this follows: the
# march runs a run of attempts on the resident engine and returns to Python only at a
# host event (the target reached, an accepted step with the driver's work, a monitor
# check, the call's bound, the budget, the floor or an error). Each decision performs
# its Python function's operations in order, with Python's comparisons: `min(a, b)` is
# `b` only if `b < a`, `max` keeps the first unless a later is strictly greater, a
# comparison with NaN is false. Julia contracts nothing unasked, and `pow` is the C
# library's, as Python's float power is.

const LIBM = Sys.isapple() ? "libSystem.B.dylib" : "libm.so.6"
"""Python's `x ** y` for floats: the C library's `pow`."""
cpow(x::Float64, y::Float64) = ccall((:pow, LIBM), Float64, (Float64, Float64), x, y)
py_min(a, b) = b < a ? b : a
py_max(a, b) = b > a ? b : a
"""`math.ulp` at a magnitude: the spacing above `|x|`; NaN for a non-finite one."""
function ulp(x::Float64)
    m = abs(x)
    isfinite(m) ? nextfloat(m) - m : NaN
end
"""`bisect.bisect_left` and `bisect_right`: how many entries lie below (at or below)."""
bisect_left(values, x) = searchsortedfirst(values, x) - 1
bisect_right(values, x) = searchsortedlast(values, x)

"""Every number the algorithm reads: solprop's `MarchPolicy`, field for field."""
struct Policy
    order::Int
    variable_order::Bool
    safety::Float64
    max_growth::Float64
    min_factor::Float64
    rejection_growth::Float64
    hysteresis::Float64
    restart_after::Int
    collar_stability::Float64
    real_intervals::Vector{Float64}
    collar_rate::Float64
    has_budget::Bool
    max_calls::Int
    equal_approach::Bool
    dz_min::Float64
    dz_max::Float64
    stretch::Float64
    z_end::Float64
    save_planes::Vector{Float64}
    hard_kinks::Vector{Float64}
    breakpoints::Vector{Float64}
    landing_ulps::Int
    has_quiet_interval::Bool
    quiet_interval::Float64
end

"""A march's configuration: its policy, the stage count the stage scalars fill, and
the operator's rate table for a compensated trial (`translate.linop.rate_weights`)."""
struct MarchConfig{R}
    policy::Policy
    stages::Int
    varying::Bool
    compensated::Bool
    split::Bool
    rate::R
end

const NONE, MAX_STEP, INTERVAL_WIDTH, HARD_KINK, SAVE_PLANE, END = 0, 1, 2, 3, 4, 5
const REACHED, HOST_STEP, MONITOR, BOUND, BUDGET, FLOOR, ERROR, DIVERGED = 0:7
const STAGE_NONE, STARTUP, TRIAL, ACCEPT = 0:3
const CONVERGENCE, BACKEND, VALUE = 1:3

# --- the decisions (adams_core) --------------------------------------------------------

function factor(p::Policy, norm, order, growth)
    norm <= 0.0 && return growth
    raw = p.safety * cpow(norm, -1.0 / Float64(order + 1))
    py_min(growth, py_max(p.min_factor, raw))
end

"""A trial's history newest first: its nodes and rows, its order and scored orders."""
function stencil(p::Policy, z, dz, history_z, history_row, stored_order, last_rejected)
    held = length(history_z)
    nodes = [(history_z[held-i] - z) / dz for i in 0:held-1]
    rows = [history_row[held-i] for i in 0:held-1]
    order = min(stored_order, held)
    rising = !last_rejected && held > order
    orders = [order]
    if p.variable_order
        order >= 2 && order - 1 <= p.order && order - 1 <= held && push!(orders, order - 1)
        order + 1 <= p.order && order + 1 <= held && rising && push!(orders, order + 1)
    end
    (nodes=nodes, rows=rows, held=held, order=order, orders=orders)
end

function stable_factor(p::Policy, order, dz, quiet)
    (p.collar_rate <= 0.0 || quiet) && return Inf
    p.collar_stability * p.real_intervals[order] / (p.collar_rate * dz)
end

function order_weight(p::Policy, norm, order, dz, quiet)
    rate = norm <= 0.0 ? Inf : cpow(norm, -1.0 / Float64(order + 1))
    py_min(p.safety * rate, stable_factor(p, order, dz, quiet))
end

"""The order to store, and the order and norm that set the next step."""
function choose_order(p::Policy, s, norms, dz, quiet)
    current = s.order
    p.variable_order || return (p.order, current, norms[1])
    weights = [order_weight(p, norms[i], s.orders[i], dz, quiet) for i in eachindex(s.orders)]
    best = 1
    for i in 2:length(weights)
        weights[i] > weights[best] && (best = i)
    end
    if s.orders[best] != current && weights[best] > p.hysteresis * weights[1]
        return (s.orders[best], s.orders[best], norms[best])
    end
    (current, current, norms[1])
end

function rejection(p::Policy, norm, order, rejections, held)
    count = rejections + 1
    restart = count >= p.restart_after && held > 1
    scale = !isfinite(norm) ? p.min_factor : factor(p, norm, order, p.rejection_growth)
    (scale, restart ? 0 : count, restart)
end

quiet_capped(p::Policy, proposal, quiet) =
    (!quiet || !p.has_quiet_interval) ? proposal : py_min(proposal, p.quiet_interval)

function collar_reentry(p::Policy, proposal, order, was_quiet, quiet)
    (!(was_quiet && !quiet) || p.collar_rate <= 0.0) && return proposal
    py_min(proposal, p.collar_stability * p.real_intervals[order] / p.collar_rate)
end

"""`stepper.schedule.landing`: the step a landing allows, its target and end."""
function landing(p::Policy, z, dz, z_end)
    (dz > 0.0 && isfinite(dz)) || throw(ArgumentError("dz must be finite and positive, got $dz"))
    edges = p.breakpoints
    edges[1] <= z <= edges[end] || throw(ArgumentError("z = $z m lies outside the operator table"))
    tolerance = p.landing_ulps * ulp(py_max(abs(z), abs(z_end)))
    z >= z_end - tolerance && throw(ArgumentError("z = $z m is already at or past z_end"))
    found = bisect_right(edges, z)
    index = min(found == 0 ? 0 : found - 1, length(edges) - 2)
    width = edges[index+2] - edges[index+1]
    # The caps, then the landings: the shortest wins, among equals the higher target.
    better(c, h) = c[1] < h[1] ? true : h[1] < c[1] ? false : c[2] > h[2]
    cap = (dz, NONE, 0.0)
    for other in ((p.dz_max, MAX_STEP, 0.0), (width, INTERVAL_WIDTH, 0.0))
        better(other, cap) && (cap = other)
    end
    land, any = (0.0, NONE, 0.0), false
    consider(target, position) = (candidate = (position - z, target, position);
                                  if !any || better(candidate, land); land = candidate; any = true; end)
    from = z + tolerance
    k = bisect_left(p.hard_kinks, from)
    k < length(p.hard_kinks) && consider(HARD_KINK, p.hard_kinks[k+1])
    q = bisect_left(p.save_planes, from)
    q < length(p.save_planes) && consider(SAVE_PLANE, p.save_planes[q+1])
    !(z_end < from) && consider(END, z_end)
    land[1] <= cap[1] && return (dz=land[1], target=land[2], landed=true, at=land[3])
    land[1] <= py_min(p.stretch * cap[1], p.dz_max) &&
        return (dz=land[1], target=land[2], landed=true, at=land[3])
    (dz=cap[1], target=cap[2], landed=false, at=0.0)
end

"""The next forced landing strictly beyond `z` (`adams_core.next_landing`), or nothing."""
function next_landing(p::Policy, z, target)
    tolerance = p.landing_ulps * ulp(py_max(abs(z), abs(target)))
    nearest = target
    for positions in (p.save_planes, p.hard_kinks)
        index = bisect_right(positions, z + tolerance)
        index < length(positions) && (nearest = py_min(nearest, positions[index+1]))
    end
    nearest > z + tolerance ? nearest : nothing
end

"""The step to take, where it ends and whether that is a hard kink."""
function approach(p::Policy, z, stop, target)
    if !stop.landed && p.equal_approach
        ahead = next_landing(p, z, target)
        if ahead !== nothing && stop.dz < ahead - z && ahead - z < 2.0 * stop.dz
            half = 0.5 * (ahead - z)
            return (dz=half, finish=z + half, kink=false)
        end
    end
    finish = stop.landed ? stop.at : z + stop.dz
    tolerance = p.landing_ulps * ulp(py_max(abs(finish), 1.0))
    interior = finish < p.z_end - tolerance
    on_kink = interior && any(kink -> abs(kink - finish) <= tolerance, p.hard_kinks)
    (dz=stop.dz, finish=finish, kink=stop.target == HARD_KINK || on_kink)
end

proposal(p::Policy, dz, f) = py_min(dz * f, p.dz_max)
"""`TransverseEdge.due`: read here when the next proposal would carry past the read."""
read_due(has_read_at, read_at, z, ahead) =
    has_read_at && z + ahead >= read_at - 1e-12 * py_max(abs(z), 1.0)
reached(position, target, ulps) =
    position >= target - ulps * ulp(py_max(abs(position), abs(target)))

# --- the engine's part -----------------------------------------------------------------

"""`translate.linop.rate_weights`: the operator's rate terms at `z`, read on the side
of a breakpoint `before` asks, by the transcribed Clenshaw pass on that interval."""
function rate_at(table, z, before)
    edges = table.edges
    low, high = edges[1], edges[end]
    slack = 1e-10 * (high - low)
    (z < low - slack || z > high + slack) &&
        throw(ArgumentError("z = $z is outside the represented span [$low, $high]"))
    index = min(max(searchsortedlast(edges, z) - 1, 0), length(edges) - 2)
    before && index > 0 && z <= edges[index+1] && (index -= 1)
    out = zeros(1, size(table.c, 3))
    clenshaw_points!([Float64(z)], [edges[index+1], edges[index+2]], -Inf, Inf,
                     view(table.c, index+1:index+1, :, :), out)
    vec(out)
end

"""The stage scalars of a trial from `z` over `dz` (the adapter's `_prepare_stages`)."""
stage_positions(m::MarchConfig, z, dz) = [fill(Float64(z), m.stages - 1); z + dz]

function march_start!(engine, m::MarchConfig, z)
    m.varying && configure_stages_at!(engine, z, stage_positions(m, z, 0.0))
    m.compensated && configure_adams_rate!(engine, rate_at(m.rate, z, false))
    adams_start!(engine)
end

function march_trial!(engine, m::MarchConfig, z, dz, s)
    m.varying && configure_stages_at!(engine, z, stage_positions(m, z, dz))
    rate = m.compensated ? rate_at(m.rate, z + dz, true) : Float64[]
    configure_adams_nodes!(engine, s.nodes, s.order, s.orders, s.rows, rate)
    adams_trial!(engine, dz)
end

"""Adopt the held trial into row `free`, commit it; return its quadrature alarm. With
a split collar the commit also forms the power the split finds (`split_power`)."""
function march_accept!(engine, dz, free, split)
    adams_accept!(engine, free)
    commit!(engine, Float64(dz), split)
end

# --- the run ---------------------------------------------------------------------------

"""Hold a run's march configuration: the policy's numbers as `MarchPolicy` gives them."""
function configure_march!(engine, numbers, real_intervals, save_planes, hard_kinks,
                          breakpoints, stages, varying, compensated, split, rate_edges, rate_c)
    f = Vector{Float64}(numbers)
    policy = Policy(Int(f[1]), f[2] != 0, f[3], f[4], f[5], f[6], f[7], Int(f[8]), f[9],
                    Vector{Float64}(real_intervals), f[10], f[11] != 0, Int(f[12]),
                    f[13] != 0, f[14], f[15], f[16], f[17], Vector{Float64}(save_planes),
                    Vector{Float64}(hard_kinks), Vector{Float64}(breakpoints), Int(f[18]),
                    f[19] != 0, f[20])
    rate = compensated ? (edges=Vector{Float64}(rate_edges), c=Array{Float64,3}(rate_c)) :
           nothing
    engine.work.march[] = MarchConfig(policy, Int(stages), Bool(varying), Bool(compensated),
                                      Bool(split), rate)
    nothing
end

"""One call of the march: from the state Python holds to a host event (SEP 39 4.8).

`zs` = (z, proposal); `counts` = (order, rejections, last_rejected, collar_was_quiet);
the history oldest first; `ledger` (startup, predictor, rejected, accepted_steps,
rejected_steps, restarts, failure_restarts), `CallLedger`'s counters; `limits` = (target, check_interval,
last_check, read_at, read_spacing, step_cap, max_seconds) and `flags` = (quiet,
host_step, monitor, first_check, has_read_at, max_attempts). Returns the state, the
attempts' rows (end, dz, norm, accepted, order, wall), the accepted orders, the reason
and its details, the last accepted step's quadrature alarm, and the first above 1e-6
with where it was.
"""
march_run!(engine, zs, counts, history_z, history_predicted, history_row, ledger, limits,
           flags) =
    run_march(engine, engine.work.march[], Vector{Float64}(zs), Vector{Int}(counts),
              Vector{Float64}(history_z), Vector{Bool}(history_predicted),
              Vector{Int}(history_row), Vector{Int}(ledger), Vector{Float64}(limits),
              Vector{Int}(flags))

function run_march(engine, m::MarchConfig, zs, counts, hz, hp, hr, ledger, limits, flags)
    p = m.policy
    started = time()
    z, proposed = zs
    order, rejections = counts[1], counts[2]
    last_rejected, was_quiet = counts[3] != 0, counts[4] != 0
    target, check_interval, last_check, read_at, read_spacing, step_cap, max_seconds = limits
    quiet, monitor, first_check, has_read_at = flags[1] != 0, flags[3] != 0, flags[4] != 0,
                                              flags[5] != 0
    host_step, max_attempts = flags[2], flags[6]
    rows, orders = Float64[], Int[]
    reason, cost, floor_proposal, stage, category, message = REACHED, 0, 0.0, STAGE_NONE, 0, ""
    alarm, alarm_z, first_alarm = NaN, NaN, NaN
    # Each accepted step's split power and length, for the adapter's split sums.
    split_powers, split_steps = Vector{Float64}[], Float64[]
    accepted = 0
    elapsed() = time() - started
    record(finish, dz, norm, kept, o) =
        append!(rows, (finish, dz, isfinite(norm) ? norm : Inf, kept ? 1.0 : 0.0, Float64(o),
                       elapsed()))
    total() = ledger[1] + ledger[2]
    function fail(at, error)
        reason, stage = ERROR, at
        message = sprint(showerror, error)
        category = error isa ArgumentError ? VALUE : BACKEND
    end
    while !reached(z, target, p.landing_ulps)
        # The budget, before any call is made (`_afford`).
        need = (isempty(hz) ? 1 : 0) + 1
        if p.has_budget && total() + need > p.max_calls
            reason, cost = BUDGET, need
            @goto done
        end
        if isempty(hz)
            ledger[1] += 1
            failed = false
            try
                march_start!(engine, m, z)
            catch error
                fail(STARTUP, error)
                failed = true
            end
            failed && @goto done
            hz, hp, hr = [z], [false], [0]
            order = p.variable_order ? 1 : p.order
        end
        # The caps: the quiet collar's, the transverse edge's, the split's, re-entry's.
        capped = quiet_capped(p, proposed, quiet)
        capped = has_read_at ? py_min(capped, read_spacing) : capped
        capped = py_min(capped, step_cap)
        capped = collar_reentry(p, capped, order, was_quiet, quiet)
        was_quiet = quiet
        step = nothing
        try
            step = approach(p, z, landing(p, z, capped, target), target)
        catch error
            fail(TRIAL, error)
        end
        step === nothing && @goto done
        s = stencil(p, z, step.dz, hz, hr, order, last_rejected)
        ledger[2] += 1
        norms = nothing
        try
            norms = march_trial!(engine, m, z, step.dz, s)
        catch error
            fail(TRIAL, error)
        end
        norms === nothing && @goto done
        norm = norms[1]
        if !(norm <= 1.0)
            # A rejection keeps the accepted history; `restart_after` in a row restart it.
            ledger[3] += 1
            ledger[5] += 1
            last_rejected = true
            scale, rejections, restart = rejection(p, norm, s.order, rejections, length(hz))
            if restart
                hz, hp, hr = Float64[], Bool[], Int[]
                ledger[7] += 1
            end
            next = proposal(p, step.dz, scale)
            if next < p.dz_min
                reason, floor_proposal = FLOOR, next
                @goto done
            end
            proposed = next
            record(step.finish, step.dz, norm, false, s.order)
        else
            # The new sample takes the lowest row the kept entries do not use.
            keep = min(p.order, length(hz))
            kz, kp, kr = hz[end-keep+1:end], hp[end-keep+1:end], hr[end-keep+1:end]
            free = 0
            while free in kr
                free += 1
            end
            failed = false
            try
                alarm = march_accept!(engine, step.dz, free, m.split)
                if m.split
                    push!(split_powers, split_power(engine))
                    push!(split_steps, step.dz)
                end
            catch error
                fail(ACCEPT, error)
                failed = true
            end
            failed && @goto done
            # The first step past the alarm's level, for the adapter's one warning.
            if alarm > 1e-6 && isnan(alarm_z)
                alarm_z = z + step.dz
                first_alarm = alarm
            end
            hz, hp, hr = [kz; z + step.dz], [kp; true], [kr; free]
            ledger[4] += 1
            rejections = 0
            push!(orders, s.order)
            stored, control, control_norm = choose_order(p, s, norms, step.dz, quiet)
            order = stored
            last_rejected = false
            z = step.finish
            if step.kink
                hz, hp, hr = Float64[], Bool[], Int[]
                ledger[6] += 1
            end
            next = proposal(p, step.dz, factor(p, control_norm, control, p.max_growth))
            record(step.finish, step.dz, norm, true, s.order)
            if next < p.dz_min
                reason, floor_proposal = FLOOR, next
                @goto done
            end
            proposed = next
            accepted += 1
            if reached(z, target, p.landing_ulps)
                reason = REACHED
                @goto done
            end
            if (host_step > 0 && accepted == host_step) ||
               read_due(has_read_at, read_at, z, proposed)
                reason = HOST_STEP
                @goto done
            end
            if monitor && (first_check || z - last_check >= check_interval)
                reason = MONITOR
                @goto done
            end
        end
        if length(rows) ÷ 6 >= max_attempts || elapsed() >= max_seconds
            reason = BOUND
            @goto done
        end
    end
    reason = REACHED
    @label done
    (zs=[z, proposed], counts=[order, rejections, Int(last_rejected), Int(was_quiet)],
     history_z=hz, history_predicted=hp, history_row=hr, ledger=ledger, rows=rows,
     orders=orders, reason=reason, cost=cost, floor_proposal=floor_proposal, stage=stage,
     category=category, message=message, alarm=alarm, alarm_z=alarm_z,
     first_alarm=first_alarm,
     # Flat (a step's power after another): a vector's buffer path is compiled already.
     split_powers=isempty(split_powers) ? Float64[] : reduce(vcat, split_powers),
     split_steps=split_steps)
end

# The decisions a warm-up's one accepted attempt does not reach compile into the
# package image, not inside a run's first window (a rejection there took 9 ms).
for T in (Int,)
    precompile(rejection, (Policy, Float64, T, T, T))
    precompile(read_due, (Bool, Float64, Float64, Float64))
    precompile(choose_order, (Policy, NamedTuple{(:nodes, :rows, :held, :order, :orders),
                                                  Tuple{Vector{Float64}, Vector{T}, T, T, Vector{T}}},
                              Vector{Float64}, Float64, Bool))
    precompile(next_landing, (Policy, Float64, Float64))
    precompile(collar_reentry, (Policy, Float64, T, Bool, Bool))
end

const configure_march_b = configure_march!
march_run_b(args...) = flushed(march_run!, args...)
