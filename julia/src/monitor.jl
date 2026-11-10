# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public MonitorNoticeCode, MonitorNotice, MonitorNotices, reset_notices!, monitor_stopped
public TemporalMonitorPolicy, TemporalMonitorWeights, TemporalReading, TemporalFloor
public TemporalMonitorState, TemporalContinuation, TemporalPlaneRow
public temporal_reading!, split_reading!, temporal_floor, temporal_floor_taken
public temporal_launch!, temporal_step!, temporal_due, temporal_skip!, temporal_note_collar!
public temporal_step_cap, temporal_removed, temporal_plane!, temporal_finish!
public temporal_continuation, restore_temporal!

"""
    MonitorNoticeCode

Identify an already decided monitor event without formatting or Python execution.

Temporal guard warning/stop codes distinguish launch through a NaN position, matching
its existing message helper. Removal warnings carry the removed energy share. Later
transverse codes carry the declared face index and their scalar diagnostic.
"""
@enum MonitorNoticeCode::UInt8 begin
    MonitorNoNotice = 0
    TemporalGuardWarning = 1
    TemporalGuardStop = 2
    TemporalRemovalWarning = 3
    TransverseFaceWarning = 4
    TransverseFaceStop = 5
    TransverseLaunchWarning = 6
    TransverseClipWarning = 7
    TransverseCoarseWarning = 8
    TransverseReturnWarning = 9
    TransverseReturnStop = 10
    TransverseRampWarning = 11
    TransverseRampStop = 12
end

"""
    MonitorNotice(code, z, value, face, auxiliary)

Carry one fixed numerical service event: code, position in m, value and face index.

`face=0` denotes no face. `auxiliary` supplies an additional scalar to the existing
message formatter; it is zero when unused. The session owns ordering and delivery.
This is an isbits record; no text or exception is created by a monitor decision.
"""
struct MonitorNotice
    code::MonitorNoticeCode
    z::Float64
    value::Float64
    face::Int
    auxiliary::Float64
end

"""
    MonitorNotices(capacity)

Allocate a fixed caller-owned notice buffer with an explicit current count.

`records` is the complete retained array; `count` starts at zero. Clear it before each
monitor hook with `reset_notices!`. Capacity must be positive. An undersized or
unconsumed buffer raises `DimensionMismatch`, never silently drops a warning/stop.
"""
mutable struct MonitorNotices{V}
    records::V
    count::Int
end
function MonitorNotices(capacity::Int)
    capacity > 0 || throw(ArgumentError("monitor notice capacity must be positive"))
    blank = MonitorNotice(MonitorNoNotice, NaN, 0.0, 0, 0.0)
    return MonitorNotices(fill(blank, capacity), 0)
end

"""
    reset_notices!(notices)

Clear the published count in a fixed notice buffer without resizing its storage.
"""
reset_notices!(notices::MonitorNotices) = (notices.count = 0; nothing)

function _notice!(notices, code, z, value; face = 0, auxiliary = 0.0)
    notices.count < length(notices.records) ||
        throw(DimensionMismatch("monitor notice buffer is full"))
    notices.count += 1
    notices.records[notices.count] = MonitorNotice(code, z, value, face, auxiliary)
    return nothing
end

"""
    monitor_stopped(notices)

Return whether the current notice buffer contains a temporal or transverse stop.
"""
function monitor_stopped(notices::MonitorNotices)
    for index in 1:notices.count
        notices.records[index].code in (
            TemporalGuardStop, TransverseFaceStop,
            TransverseReturnStop, TransverseRampStop,
        ) && return true
    end
    return false
end

"""
    TemporalMonitorPolicy(; warn, stop, activity, quiet_share, quiet_checks,
        quiet_collar, cadence, interval, launch_floor_ratio, floor_margin, split,
        crossing_share=0.0, bucket_limits=Float64[])

Retain explicitly lowered temporal monitor thresholds and cadence policy.

`warn` and `stop` are amplitude-share thresholds; `activity` and `quiet_share` are
energy shares. `interval` and positive bucket limits are in m, with Inf permitted.
The positive integer cadence/count and nonnegative floor ratio/margin come from the
neutral monitor definition, not Julia defaults. `crossing_share` lies in `[0,1)`;
empty bucket limits disable crossing. The policy borrows immutable limits.
Invalid thresholds raise `ArgumentError`. Scalar decisions use binary64 in the same
expression order as the existing monitor, with no contraction or reassociation.
"""
struct TemporalMonitorPolicy{V}
    warn::Float64
    stop::Float64
    activity::Float64
    quiet_share::Float64
    quiet_checks::Int
    quiet_collar::Bool
    cadence::Int
    interval::Float64
    launch_floor_ratio::Float64
    floor_margin::Float64
    split::Bool
    crossing_share::Float64
    bucket_limits::V
end
function TemporalMonitorPolicy(;
        warn, stop, activity, quiet_share, quiet_checks::Int, quiet_collar::Bool,
        cadence::Int, interval, launch_floor_ratio, floor_margin, split::Bool,
        crossing_share = 0.0, bucket_limits = Float64[],
    )
    all(
        isfinite, (
            warn, stop, activity, quiet_share, launch_floor_ratio,
            floor_margin, crossing_share,
        )
    ) && 0 < warn <= stop && activity >= 0 &&
        quiet_share >= 0 && launch_floor_ratio >= 0 && floor_margin >= 0 &&
        0 <= crossing_share < 1 && quiet_checks > 0 && cadence > 0 && interval > 0 &&
        all(value -> value > 0, bucket_limits) ||
        throw(ArgumentError("invalid temporal monitor thresholds"))
    return TemporalMonitorPolicy(
        warn, stop, activity, quiet_share, quiet_checks,
        quiet_collar, cadence, interval, launch_floor_ratio, floor_margin,
        split, crossing_share, bucket_limits
    )
end

"""
    TemporalMonitorWeights(regions, columns; buckets=nothing)

Retain coarse-time region weights, column quadrature and optional spectral buckets.

`regions` is `(n_t,4)`: low guard, high guard, collar occupancy, then removal rate in
m⁻¹. Its entries and the `n_columns` column weights must be finite nonnegative.
`buckets`, when present, is an integer `(n_omega,n_columns)` table of one-based bucket
indices. Crossing occupancies use raw squared spectra and column quadrature, without
analytic endpoint or Parseval factors, matching `SplitCrossing.occupancy`.
All tables are caller-owned immutable preparation inputs; invalid shapes/weights raise.
"""
struct TemporalMonitorWeights{R, C, B}
    regions::R
    columns::C
    buckets::B
end
function TemporalMonitorWeights(regions, columns; buckets = nothing)
    ndims(regions) == 2 && size(regions, 1) > 0 && size(regions, 2) == 4 &&
        !isempty(columns) || throw(DimensionMismatch("invalid temporal monitor tables"))
    all(value -> isfinite(value) && value >= 0, regions) &&
        all(value -> isfinite(value) && value >= 0, columns) ||
        throw(ArgumentError("temporal monitor weights must be finite nonnegative"))
    if buckets !== nothing
        ndims(buckets) == 2 && size(buckets, 2) == length(columns) &&
            eltype(buckets) <: Integer && all(value -> value > 0, buckets) ||
            throw(ArgumentError("temporal crossing buckets must be positive indices"))
    end
    return TemporalMonitorWeights(regions, columns, buckets)
end

"""
    TemporalReading(n_buckets=0)

Allocate a fixed numerical reading destination with zero scalar readings.

`total` and four `regions` contain the proxy energy and weighted energies.
`removed` is the split removal since the previous check; `found` is its latest
half-factor `(total,low,high,collar)` reading when `has_found` is true. `occupancy` is
the only retained array, with one raw spectral energy per crossing bucket. The caller
owns accumulation/reset of split removals between observations.
"""
mutable struct TemporalReading{V}
    total::Float64
    regions::NTuple{4, Float64}
    removed::Float64
    found::NTuple{4, Float64}
    has_found::Bool
    occupancy::V
end
function TemporalReading(n_buckets::Int = 0)
    n_buckets >= 0 || throw(ArgumentError("bucket count must be nonnegative"))
    return TemporalReading(
        0.0, (0.0, 0.0, 0.0, 0.0), 0.0,
        (0.0, 0.0, 0.0, 0.0), false, zeros(n_buckets)
    )
end

"""
    TemporalFloor(total, regions)

Retain the input noise's total proxy energy and stationary expected region energies.

The four values match `TemporalReading.regions`; the last is an expected removal rate.
`temporal_floor` forms these from the actual noise reading and uniform time density.
The all-zero record denotes no input-noise floor.
"""
struct TemporalFloor
    total::Float64
    regions::NTuple{4, Float64}

    function TemporalFloor(total::Real, regions::NTuple{4, <:Real})
        converted_total = Float64(total)
        converted = ntuple(index -> Float64(regions[index]), 4)
        isfinite(converted_total) && converted_total >= 0 &&
            all(value -> isfinite(value) && value >= 0, converted) ||
            throw(ArgumentError("temporal noise floor must be finite nonnegative"))
        return new(converted_total, converted)
    end
end

function _monitor_pair(value)
    real_value, imaginary_value = Float64(real(value)), Float64(imag(value))
    return real_value * real_value + imaginary_value * imaginary_value
end

function _temporal_power!(power, samples, weights, execution)
    size(samples) == (length(power), length(weights.columns)) &&
        length(power) == size(weights.regions, 1) ||
        throw(DimensionMismatch("temporal monitor samples and scratch disagree"))
    all(isfinite, samples) || throw(ArgumentError("temporal monitor samples are nonfinite"))
    (
        Base.mightalias(power, samples) || Base.mightalias(power, weights.regions) ||
            Base.mightalias(power, weights.columns) ||
            (weights.buckets !== nothing && Base.mightalias(power, weights.buckets))
    ) &&
        throw(ArgumentError("temporal monitor scratch aliases its input"))
    columns = weights.columns
    each!(execution, length(power), 1) do row, _
        value = 0.0
        for column in axes(samples, 2)
            value += columns[column] * _monitor_pair(samples[row, column])
        end
        power[row] = value
    end
    all(isfinite, power) || throw(ArgumentError("temporal monitor power overflowed"))
    return nothing
end

"""
    temporal_reading!(reading, samples, weights, power, execution; spectrum=nothing)

Reduce actual coarse inverse samples into temporal proxy energy and four regions.

`samples` is real for a field or complex for an envelope, `(n_t,n_columns)`; it is
not an analytic field. `power` is caller-owned binary64 length-`n_t` scratch. Optional
raw `spectrum` is required when crossing buckets are prepared. Split `removed/found`
fields are preserved, so the last accepted split can be combined with this reading.
All writable storage must be disjoint from inputs. Invalid/nonfinite data raise.
"""
function temporal_reading!(
        reading, samples, weights, power, execution;
        spectrum = nothing
    )
    buckets = weights.buckets
    if buckets !== nothing
        spectrum !== nothing && size(spectrum) == size(buckets) &&
            !isempty(reading.occupancy) && maximum(buckets) <= length(reading.occupancy) ||
            throw(DimensionMismatch("temporal crossing requires matching raw spectra"))
        all(isfinite, spectrum) || throw(ArgumentError("crossing spectrum is nonfinite"))
        (
            Base.mightalias(power, spectrum) || Base.mightalias(power, buckets) ||
                Base.mightalias(power, reading.occupancy) ||
                Base.mightalias(reading.occupancy, spectrum) ||
                Base.mightalias(reading.occupancy, buckets) ||
                Base.mightalias(reading.occupancy, weights.columns) ||
                Base.mightalias(reading.occupancy, weights.regions) ||
                Base.mightalias(reading.occupancy, samples)
        ) &&
            throw(ArgumentError("temporal crossing buffers alias their inputs"))
    elseif !isempty(reading.occupancy)
        throw(DimensionMismatch("crossing occupancy needs prepared bucket indices"))
    end
    _temporal_power!(power, samples, weights, execution)
    totals = (0.0, 0.0, 0.0, 0.0, 0.0)
    for row in eachindex(power)
        value = power[row]
        totals = (
            totals[1] + value,
            totals[2] + value * weights.regions[row, 1],
            totals[3] + value * weights.regions[row, 2],
            totals[4] + value * weights.regions[row, 3],
            totals[5] + value * weights.regions[row, 4],
        )
    end
    all(isfinite, totals) || throw(ArgumentError("temporal monitor reduction overflowed"))
    reading.total = totals[1]
    reading.regions = (totals[2], totals[3], totals[4], totals[5])
    if buckets !== nothing
        occupancy, columns = reading.occupancy, weights.columns
        each!(execution, length(occupancy), 1) do bucket, _
            value = 0.0
            for column in axes(spectrum, 2), row in axes(spectrum, 1)
                buckets[row, column] == bucket || continue
                value += columns[column] * _monitor_pair(spectrum[row, column])
            end
            occupancy[bucket] = value
        end
        all(isfinite, occupancy) || throw(ArgumentError("crossing occupancy overflowed"))
    end
    return nothing
end

"""
    split_reading!(reading, samples, step, weights, power, execution; add=false)

Reduce one accepted split's removed energy and its halfway occupancies.

The input is the coarse inverse field immediately before its split, not the masked
committed field. `step` is positive in m. Removal uses `-expm1(-rate*step)` and found
power uses `exp(-rate*step/2)`. `add=true` accumulates removal since an earlier check;
the found tuple always replaces the previous accepted step's tuple. Other reading
fields are preserved. Buffers/validation match `temporal_reading!`.
"""
function split_reading!(
        reading, samples, step, weights, power, execution;
        add::Bool = false
    )
    isfinite(step) && step > 0 ||
        throw(ArgumentError("split monitor step must be positive"))
    Base.mightalias(power, reading.occupancy) &&
        throw(ArgumentError("split scratch aliases the retained crossing occupancy"))
    _temporal_power!(power, samples, weights, execution)
    taken = total = low = high = collar = 0.0
    for row in eachindex(power)
        rate = weights.regions[row, 4]
        taken += power[row] * -expm1(-rate * step)
        half = power[row] * exp(-0.5 * rate * step)
        total += half
        low += weights.regions[row, 1] * half
        high += weights.regions[row, 2] * half
        collar += weights.regions[row, 3] * half
    end
    removed = add ? reading.removed + taken : taken
    all(isfinite, (removed, total, low, high, collar)) ||
        throw(ArgumentError("split monitor reduction overflowed"))
    reading.removed, reading.found = removed, (total, low, high, collar)
    reading.has_found = true
    return nothing
end

"""
    temporal_floor(reading, weights)

Return the stationary expected noise regions from its actual total proxy energy.

Each expectation is total energy times the time mean of that region's weights.
No shot-noise model or default physical constant is chosen inside this reduction.
"""
function temporal_floor(reading, weights)
    total = reading.total
    isfinite(total) && total >= 0 ||
        throw(ArgumentError("noise floor must be finite nonnegative"))
    regions = ntuple(4) do region
        value = 0.0
        for row in axes(weights.regions, 1)
            value += weights.regions[row, region]
        end
        total * (value / size(weights.regions, 1))
    end
    all(isfinite, regions) || throw(ArgumentError("noise floor regions overflowed"))
    return TemporalFloor(total, regions)
end

"""
    temporal_floor_taken(floor, weights, step)

Return expected noise removal for one committed split step in the proxy's units.

This is `floor.total/n_t * sum(-expm1(-rate*step))`, retaining the existing operation
order. A nonpositive step takes zero, as a dense observation behind the committed
position does. The session adds this value only once for each actual positive step.
"""
function temporal_floor_taken(floor::TemporalFloor, weights, step)
    step <= 0 && return 0.0
    isfinite(step) || throw(ArgumentError("noise-removal step must be finite"))
    taken = 0.0
    for row in axes(weights.regions, 1)
        taken += -expm1(-weights.regions[row, 4] * step)
    end
    return floor.total / size(weights.regions, 1) * taken
end

"""
    TemporalMonitorState()

Create empty concrete scalar state for one temporal monitor.

Positions are in m, removal rates are per m, and all occupancies/removals are energy
shares. Fixed tuples retain the four noise regions, two guard maxima/positions and
plane-interval maxima. The state owns no arrays, history, callback or Python object.
The session owns saved rows, notices and resume/unwatched interval metadata.
"""
Base.@kwdef mutable struct TemporalMonitorState
    cap::Float64 = Inf
    launch::Float64 = NaN
    floor::Float64 = 0.0
    floor_regions::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0)
    raw_guard_max::Float64 = 0.0
    raw_removed::Float64 = 0.0
    floor_removed::Float64 = 0.0
    has_last::Bool = false
    last_z::Float64 = NaN
    last_rate::Float64 = 0.0
    raw_last_z::Float64 = NaN
    raw_last_rate::Float64 = 0.0
    guard_max::NTuple{2, Float64} = (0.0, 0.0)
    guard_at::NTuple{2, Float64} = (NaN, NaN)
    collar_max::Float64 = 0.0
    launch_occupancy::NTuple{3, Float64} = (NaN, NaN, NaN)
    warned::Bool = false
    collar_now::Float64 = 0.0
    since_plane::NTuple{2, Float64} = (0.0, 0.0)
    read_since_plane::Bool = false
    last_plane_z::Float64 = NaN
    steps::Int = 0
    skipped::Int = 0
    since_check::Int = 0
    busy::Bool = true
    quiet::Bool = false
    quiet_run::Int = 0
    left_out::Bool = false
    left_out_checks::Int = 0
    stale::Bool = false
    committed::Float64 = NaN
    floor_taken::Float64 = 0.0
    resuming::Bool = false
    resume_plane::Bool = false
    restore_quiet::Bool = false
    restored_quiet_run::Int = 0
    restored_quiet::Bool = false
end

"""
    TemporalPlaneRow(emit, z, collar, guard_low, guard_high, removed)

Return a fixed saved-plane row without appending to session storage.

`emit=false` means a duplicate resumed plane or a hook with no row. Unmeasured dense
planes have NaN occupancies; `removed` is the nonnegative removed energy share.
"""
struct TemporalPlaneRow
    emit::Bool
    z::Float64
    collar::Float64
    guard_low::Float64
    guard_high::Float64
    removed::Float64
end

"""
    TemporalContinuation

Retain immutable numerical continuation data from the monitor's public record.

Fields preserve original launch/guard/removal totals, quiet count, checked/skipped
counts and the last saved-plane coordinate. Row histories and textual metadata are
owned by the session. A restore rejudges neither the launch nor its quiet count.
"""
struct TemporalContinuation
    launch::Float64
    launch_occupancy::NTuple{3, Float64}
    guard_max::NTuple{2, Float64}
    guard_at::NTuple{2, Float64}
    collar_max::Float64
    raw_removed::Float64
    floor_removed::Float64
    quiet_run::Int
    quiet::Bool
    raw_guard_max::Float64
    steps::Int
    skipped::Int
    left_out_checks::Int
    last_plane_z::Float64
end

function _temporal_cap!(state, policy, reading)
    limits = policy.bucket_limits
    isempty(limits) && return nothing
    length(reading.occupancy) == length(limits) &&
        all(value -> isfinite(value) && value >= 0, reading.occupancy) ||
        throw(ArgumentError("crossing reading is missing or invalid"))
    total = 0.0
    for value in reading.occupancy
        total += value
    end
    isfinite(total) || throw(ArgumentError("crossing occupancy total overflowed"))
    state.cap = Inf
    total > 0 || return nothing
    cutoff, cumulative, bucket = policy.crossing_share * total, 0.0, length(limits)
    for index in eachindex(limits)
        cumulative += reading.occupancy[index]
        if cumulative > cutoff
            bucket = index
            break
        end
    end
    state.cap = limits[bucket]
    return nothing
end

function _temporal_judged(reading, policy, step)
    if step && policy.split
        reading.has_found && (reading.found[1] > 0 || !(reading.total > 0)) ||
            throw(ArgumentError("split reading must retain what the accepted split found"))
        total = reading.found[1]
        regions = (reading.found[2], reading.found[3], reading.found[4], reading.regions[4])
    else
        total, regions = reading.total, reading.regions
    end
    if !(total > 0) || !isfinite(total)
        return 0.0, (0.0, 0.0, 0.0, 0.0)
    end
    all(isfinite, regions) || throw(ArgumentError("temporal monitor regions are nonfinite"))
    return total, regions
end

function _temporal_above(state, policy, regions)
    above = (
        max(regions[1] - state.floor_regions[1] * policy.floor_margin, 0.0),
        max(regions[2] - state.floor_regions[2] * policy.floor_margin, 0.0),
        max(regions[3] - state.floor_regions[3] * policy.floor_margin, 0.0),
        regions[4] - state.floor_regions[4],
    )
    all(isfinite, above) || throw(ArgumentError("temporal floor subtraction overflowed"))
    return above
end

function _temporal_settle!(state, policy, shares, raw_collar = 0.0)
    if shares === nothing
        state.busy, calm = false, true
    else
        state.busy = max(shares[1], shares[2], shares[3]) > policy.activity
        calm = raw_collar < policy.quiet_share
    end
    state.quiet_run = calm ? state.quiet_run + 1 : 0
    state.quiet = policy.quiet_collar && state.quiet_run >= policy.quiet_checks
    return nothing
end

function _temporal_restore_quiet!(state, policy)
    state.restore_quiet || return nothing
    state.quiet_run = state.restored_quiet_run
    state.quiet = policy.quiet_collar && state.restored_quiet
    state.restore_quiet = false
    return nothing
end

function _temporal_judge!(state, policy, shares, z, notices)
    worst = sqrt(max(shares[1], shares[2], 0.0))
    if worst > policy.stop
        _notice!(notices, TemporalGuardStop, z, worst)
    elseif worst > policy.warn && !state.warned
        state.warned = true
        _notice!(notices, TemporalGuardWarning, z, worst)
    end
    return nothing
end

"""
    temporal_launch!(state, policy, reading, z, floor, notices)

Record a fresh or resumed launch, returning its optional fixed plane row.

`floor` is a `TemporalFloor` from actual input noise, or its zero record. Fresh guard
warnings/stops have a NaN notice position, identifying launch to the existing helper.
A resumed launch retains its original positive launch energy, does not judge again,
and restores its quiet count after settling the current reading. Invalid inputs raise.
"""
function temporal_launch!(state, policy, reading, z, floor::TemporalFloor, notices)
    isfinite(z) && isfinite(floor.total) && floor.total >= 0 &&
        all(value -> isfinite(value) && value >= 0, floor.regions) ||
        throw(ArgumentError("invalid temporal launch position or floor"))
    state.floor, state.floor_regions = floor.total, floor.regions
    state.committed, state.read_since_plane = z, true
    _temporal_cap!(state, policy, reading)
    total, regions = _temporal_judged(reading, policy, false)
    resuming, state.resuming = state.resuming, false
    !(resuming && state.launch > 0) && (state.launch = total)
    if total <= 0
        state.has_last, state.last_z, state.last_rate = true, z, 0.0
        state.raw_last_z, state.raw_last_rate = z, 0.0
        _temporal_settle!(state, policy, nothing)
        _temporal_restore_quiet!(state, policy)
        return TemporalPlaneRow(false, z, NaN, NaN, NaN, max(temporal_removed(state), 0.0))
    end
    above = _temporal_above(state, policy, regions)
    shares = ntuple(index -> above[index] / total, 4)
    all(isfinite, shares) || throw(ArgumentError("temporal occupancy shares overflowed"))
    _temporal_settle!(state, policy, shares, regions[3] / total)
    _temporal_restore_quiet!(state, policy)
    state.has_last, state.last_z, state.last_rate = true, z, above[4] / state.launch
    state.raw_last_z, state.raw_last_rate = z, regions[4] / state.launch
    state.collar_now, state.since_plane = shares[3], (shares[1], shares[2])
    if resuming
        return state.resume_plane ? temporal_plane!(state, z) :
            TemporalPlaneRow(false, z, NaN, NaN, NaN, max(temporal_removed(state), 0.0))
    end
    state.launch_occupancy = (shares[1], shares[2], shares[3])
    row = temporal_plane!(state, z)
    _temporal_judge!(state, policy, shares, NaN, notices)
    return row
end

function _temporal_commit!(state, policy, z, floor_taken)
    isfinite(z) && isfinite(floor_taken) && floor_taken >= 0 ||
        throw(ArgumentError("invalid committed monitor position or floor removal"))
    z - state.committed > 0 || return nothing
    state.committed = z
    policy.split && state.floor > 0 && (state.floor_taken += floor_taken)
    return nothing
end

"""
    temporal_step!(state, policy, reading, z, notices; plane=false, floor_taken=0.0)

Account a committed reading and emit its already decided guard notices.

A stale plane recheck stays counted among skipped steps. Split removal is accumulated
from every accepted step since the previous check; in-stage removal uses the raw-rate
trapezoid unless the collar was left out. `floor_taken` is the actual positive-step
expected removal from `temporal_floor_taken`. Zero/no-light readings settle quietly.
The routine does not clear the caller's reading or append a plane row.
"""
function temporal_step!(
        state, policy, reading, z, notices;
        plane::Bool = false, floor_taken = 0.0
    )
    rechecked = plane && state.stale
    _temporal_commit!(state, policy, z, floor_taken)
    state.read_since_plane = true
    _temporal_cap!(state, policy, reading)
    total, regions = _temporal_judged(reading, policy, true)
    !rechecked && (state.steps += 1)
    state.left_out_checks += Int(state.left_out)
    state.since_check, state.stale = 0, false
    if policy.split && state.launch > 0
        isfinite(reading.removed) && reading.removed >= 0 ||
            throw(ArgumentError("split removal must be finite nonnegative"))
        state.raw_removed += reading.removed / state.launch
        state.floor_removed += state.floor_taken / state.launch
        all(isfinite, (state.raw_removed, state.floor_removed)) ||
            throw(ArgumentError("split monitor removal totals overflowed"))
    end
    state.floor_taken = 0.0
    if !(total > policy.launch_floor_ratio * state.launch)
        state.has_last, state.last_z, state.last_rate = true, z, 0.0
        state.raw_last_z, state.raw_last_rate = z, 0.0
        state.collar_now = 0.0
        _temporal_settle!(state, policy, nothing)
        return nothing
    end
    state.launch > 0 ||
        throw(ArgumentError("positive monitor light needs a positive launch"))
    raw_rate = regions[4] / state.launch
    isfinite(raw_rate) || throw(ArgumentError("temporal removal rate overflowed"))
    previous_z = state.has_last ? state.raw_last_z : z
    previous_rate = state.has_last ? state.raw_last_rate : raw_rate
    if !(policy.split || state.left_out)
        state.raw_removed += 0.5 * (z - previous_z) * (previous_rate + raw_rate)
        state.floor_removed += (z - previous_z) * state.floor_regions[4] / state.launch
        all(isfinite, (state.raw_removed, state.floor_removed)) ||
            throw(ArgumentError("in-stage monitor removal totals overflowed"))
    end
    state.raw_last_z, state.raw_last_rate = z, raw_rate
    state.raw_guard_max = max(state.raw_guard_max, max(regions[1], regions[2]) / total)
    above = _temporal_above(state, policy, regions)
    shares = ntuple(index -> above[index] / total, 4)
    all(isfinite, shares) || throw(ArgumentError("temporal occupancy shares overflowed"))
    _temporal_settle!(state, policy, shares, regions[3] / total)
    state.has_last, state.last_z, state.last_rate = true, z, above[4] / state.launch
    state.collar_max, state.collar_now = max(state.collar_max, shares[3]), shares[3]
    state.since_plane = (
        max(state.since_plane[1], shares[1]),
        max(state.since_plane[2], shares[2]),
    )
    state.guard_at = (
        shares[1] > state.guard_max[1] ? z : state.guard_at[1],
        shares[2] > state.guard_max[2] ? z : state.guard_at[2],
    )
    state.guard_max = (
        max(state.guard_max[1], shares[1]),
        max(state.guard_max[2], shares[2]),
    )
    _temporal_judge!(state, policy, shares, z, notices)
    return nothing
end

"""
    temporal_due(state, policy, z)

Return whether activity, committed-step cadence or distance makes a reading due.
"""
function temporal_due(state, policy, z)
    (state.busy || state.since_check + 1 >= policy.cadence) && return true
    return !state.has_last || z - state.last_z >= policy.interval
end

"""
    temporal_skip!(state, policy, z; floor_taken=0.0)

Count one unchecked committed step and retain its expected split-noise removal.
"""
function temporal_skip!(state, policy, z; floor_taken = 0.0)
    _temporal_commit!(state, policy, z, floor_taken)
    state.skipped += 1
    state.since_check += 1
    state.stale = true
    return nothing
end

"""
    temporal_note_collar!(state, left_out)

Record whether the numerical stages actually omit the quiet in-stage collar.
"""
function temporal_note_collar!(state, left_out::Bool)
    state.left_out = left_out
    return nothing
end

"""
    temporal_step_cap(state)

Return the last crossing limit in m, or Inf when no crossing cap is active.
"""
temporal_step_cap(state) = state.cap

"""
    temporal_removed(state)

Return signed signal removal above at most one launched input-noise floor.
"""
function temporal_removed(state)
    floor = state.launch > 0 ? state.floor / state.launch : 0.0
    return state.raw_removed - min(state.floor_removed, floor)
end

"""
    temporal_plane!(state, z)

Return a fixed row and reset the next plane interval's guard maxima.

A repeated saved coordinate returns `emit=false`. A dense plane without an intervening
check carries NaN occupancies, retaining the previous removal total. This does not
count or judge a step; the session owns row allocation, indices and delivery.
"""
function temporal_plane!(state, z)
    isfinite(z) || throw(ArgumentError("monitor plane coordinate must be finite"))
    read, state.read_since_plane = state.read_since_plane, false
    duplicate = state.last_plane_z == z
    row = TemporalPlaneRow(
        !duplicate, z, read ? state.collar_now : NaN,
        read ? state.since_plane[1] : NaN, read ? state.since_plane[2] : NaN,
        max(temporal_removed(state), 0.0)
    )
    state.since_plane = (0.0, 0.0)
    !duplicate && (state.last_plane_z = z)
    return row
end

"""
    temporal_finish!(state, policy, notices)

Emit the closing removal warning when its amplitude exceeds the lowered budget.

The session calls this once and formats the fixed numerical notice; no logging or
Python execution occurs in this hook.
"""
function temporal_finish!(state, policy, notices)
    removed = max(temporal_removed(state), 0.0)
    removed > 0 && sqrt(removed) > policy.warn &&
        _notice!(notices, TemporalRemovalWarning, NaN, removed)
    return nothing
end

"""
    temporal_continuation(state)

Return immutable numerical restart data, without copying row or notice histories.
"""
function temporal_continuation(state)
    return TemporalContinuation(
        state.launch, state.launch_occupancy, state.guard_max,
        state.guard_at, state.collar_max, state.raw_removed, state.floor_removed,
        state.quiet_run, state.quiet, state.raw_guard_max, state.steps, state.skipped,
        state.left_out_checks, state.last_plane_z
    )
end

"""
    restore_temporal!(state, continuation; plane)

Restore a numerical public record into a fresh temporal monitor before launch.

Use `nothing` for a resumed run with no previous monitor record. The next launch
records but does not judge a resumed field; original positive launch/totals survive,
and its previously counted quiet check is not counted twice. The caller separately
restores plane histories and resume/unwatched metadata.
"""
function restore_temporal!(state, continuation; plane::Bool)
    !state.has_last || throw(ArgumentError("restore requires fresh temporal monitor state"))
    state.resuming, state.resume_plane = true, plane
    continuation === nothing && return nothing
    continuation.launch > 0 || return nothing
    state.launch, state.launch_occupancy = continuation.launch,
        continuation.launch_occupancy
    state.guard_max, state.guard_at = continuation.guard_max, continuation.guard_at
    state.collar_max = continuation.collar_max
    state.raw_removed, state.floor_removed = continuation.raw_removed,
        continuation.floor_removed
    state.restore_quiet = true
    state.restored_quiet_run, state.restored_quiet = continuation.quiet_run,
        continuation.quiet
    state.raw_guard_max = continuation.raw_guard_max
    state.steps, state.skipped = continuation.steps, continuation.skipped
    state.left_out_checks, state.last_plane_z = continuation.left_out_checks,
        continuation.last_plane_z
    return nothing
end
