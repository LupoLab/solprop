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
    TransverseLaunchStop = 13
    TransverseInsensitiveWarning = 14
    TransverseUnwatchedWarning = 15
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
            TransverseReturnStop, TransverseRampStop, TransverseLaunchStop,
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

function _monitor_temporal_power!(power, samples, weights, execution)
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
    _monitor_temporal_power!(power, samples, weights, execution)
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
    _monitor_temporal_power!(power, samples, weights, execution)
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

public TransverseStatus, TransversePolicy, TransverseTables, TransverseReading
public TransverseState, TransverseContinuation, TransverseStripLine, TransverseCurrentLine
public transverse_reading!, transverse_strip!, transverse_current!
public transverse_launch!, transverse_read!, transverse_step!, transverse_due
public transverse_wants_read, transverse_ceiling, transverse_ramp_removed
public transverse_continuation, restore_transverse!, transverse_finish!

"""
    TransverseStatus

Identify the numerical edge status, with names formatted by the session boundary.

The values are clear, warned, stopped, insensitive, coarse, unwatched and off.
"""
@enum TransverseStatus::UInt8 begin
    TransverseClear = 0
    TransverseWarned = 1
    TransverseStopped = 2
    TransverseInsensitive = 3
    TransverseCoarse = 4
    TransverseUnwatched = 5
    TransverseOff = 6
end

"""
    TransversePolicy(; warn, stop, backstop_start, backstop_max, reach_growth,
        guard_numerator, k_min, k_floor_fraction, due_tolerance, noise_factor,
        strip_floor_margin, strip_leakage, has_design, has_collar)

Retain explicitly lowered transverse monitoring policy in binary64.

`warn/stop` are energy shares. Backstops and `guard_numerator=4*guard_width` are in m;
`k_min` is in rad/m. Remaining margins are dimensionless and come from the neutral
monitor. There are no new physical defaults. A collar requires a design; positive
scales and finite nonnegative margins are validated at preparation.
"""
struct TransversePolicy
    warn::Float64
    stop::Float64
    backstop_start::Float64
    backstop_max::Float64
    reach_growth::Float64
    guard_numerator::Float64
    k_min::Float64
    k_floor_fraction::Float64
    due_tolerance::Float64
    noise_factor::Float64
    strip_floor_margin::Float64
    strip_leakage::Float64
    has_design::Bool
    has_collar::Bool
end
function TransversePolicy(;
        warn, stop, backstop_start, backstop_max, reach_growth, guard_numerator,
        k_min, k_floor_fraction, due_tolerance, noise_factor, strip_floor_margin,
        strip_leakage, has_design::Bool, has_collar::Bool,
    )
    all(
        isfinite, (
            warn, stop, backstop_start, backstop_max, reach_growth,
            guard_numerator, k_min, k_floor_fraction, due_tolerance, noise_factor,
            strip_floor_margin, strip_leakage,
        )
    ) && 0 < warn <= stop &&
        0 < backstop_start <= backstop_max && reach_growth >= 0 &&
        guard_numerator > 0 && k_min > 0 && k_floor_fraction > 0 &&
        due_tolerance >= 0 && noise_factor >= 0 && strip_floor_margin >= 0 &&
        strip_leakage >= 0 && (!has_collar || has_design) ||
        throw(ArgumentError("invalid transverse monitor policy"))
    return TransversePolicy(
        warn, stop, backstop_start, backstop_max, reach_growth,
        guard_numerator, k_min, k_floor_fraction, due_tolerance, noise_factor,
        strip_floor_margin, strip_leakage, has_design, has_collar
    )
end

"""
    TransverseTables(faces, clip, totals, column_weights, kappa, order, ramp_rates;
                     face_ranks, noise_face_share, noise_modes)

Retain prepared spatial reduction weights and exact scalar-decision ordering.

The three real matrices are `(n_nodes,4)`, with unused face columns zero; totals are
whole-grid energy, collar removal rate, launch energy outside the requested size and
an unused zero column. Vectors have `n_columns` entries. `kappa` is in rad/m, ramp
rates in m⁻¹ and `order` is the one-based ascending-kappa permutation from preparation.
`face_ranks` gives the lexical tie rank of each of one to four actual faces in declared
order. Noise geometry is the first face's measure share and its node×band sample count.
Tables are immutable caller-owned inputs; invalid dimensions/domains raise.
"""
struct TransverseTables{F, C, T, V, K, I, R, N}
    faces::F
    clip::C
    totals::T
    column_weights::V
    kappa::K
    order::I
    ramp_rates::R
    face_ranks::NTuple{N, Int}
    noise_face_share::Float64
    noise_modes::Int
end
function TransverseTables(
        faces, clip, totals, column_weights, kappa, order, ramp_rates;
        face_ranks::NTuple{N, Int}, noise_face_share, noise_modes::Int
    ) where {N}
    all(vector -> ndims(vector) == 1, (column_weights, kappa, order, ramp_rates)) &&
        eltype(order) <: Integer &&
        size(faces) == size(clip) == size(totals) && ndims(faces) == 2 &&
        size(faces, 1) > 0 && size(faces, 2) == 4 &&
        length(column_weights) == length(kappa) ==
        length(order) == length(ramp_rates) > 0 ||
        throw(DimensionMismatch("transverse monitor table dimensions disagree"))
    1 <= N <= 4 && sort(collect(Int, face_ranks)) == collect(1:N) &&
        sort(collect(order)) == collect(1:length(order)) &&
        all(index -> kappa[order[index]] >= kappa[order[index - 1]], 2:length(order)) ||
        throw(ArgumentError("transverse face ranks or kappa order are invalid"))
    all(
        table -> all(value -> isfinite(value) && value >= 0, table),
        (faces, clip, totals, column_weights, kappa, ramp_rates)
    ) &&
        isfinite(noise_face_share) && 0 <= noise_face_share <= 1 && noise_modes > 0 ||
        throw(ArgumentError("transverse monitor weights must be finite nonnegative"))
    return TransverseTables(
        faces, clip, totals, column_weights, kappa, order, ramp_rates,
        face_ranks, noise_face_share, noise_modes
    )
end

"""
    TransverseReading(n_columns, n_lines=0)

Allocate fixed transverse reading storage with no valid observation initially.

`faces/clip/totals` are four-entry binary64 tuples. Owned `columns` holds unweighted
spectral column energies; `strip` is `(n_lines,2)` inward/windowed energy. `current`
is the rectified onset current in proxy energy per m. Validity flags require callers
to form strips/current after each `transverse_reading!`, preventing stale reuse.
"""
mutable struct TransverseReading{V, A}
    faces::NTuple{4, Float64}
    clip::NTuple{4, Float64}
    totals::NTuple{4, Float64}
    columns::V
    strip::A
    current::Float64
    has_faces::Bool
    full::Bool
    has_strip::Bool
    has_current::Bool
end
function TransverseReading(n_columns::Int, n_lines::Int = 0)
    n_columns > 0 && n_lines >= 0 || throw(ArgumentError("invalid transverse reading size"))
    blank = (0.0, 0.0, 0.0, 0.0)
    return TransverseReading(
        blank, blank, blank, zeros(n_columns), zeros(n_lines, 2),
        0.0, false, false, false, false
    )
end

function _transverse_alias_tables(buffer, tables)
    return Base.mightalias(buffer, tables.faces) || Base.mightalias(buffer, tables.clip) ||
        Base.mightalias(buffer, tables.totals) ||
        Base.mightalias(buffer, tables.column_weights) ||
        Base.mightalias(buffer, tables.kappa) || Base.mightalias(buffer, tables.order) ||
        Base.mightalias(buffer, tables.ramp_rates)
end

function _transverse_numerical_check(reading, node_spectrum, column_spectrum, tables, nodes)
    ndims(node_spectrum) == ndims(column_spectrum) == 2 &&
        size(node_spectrum, 1) == size(column_spectrum, 1) > 0 &&
        size(node_spectrum, 2) == size(tables.faces, 1) == length(nodes) &&
        size(column_spectrum, 2) == length(reading.columns) ==
        length(tables.column_weights) ||
        throw(DimensionMismatch("transverse spectra and reduction storage disagree"))
    all(isfinite, node_spectrum) && all(isfinite, column_spectrum) ||
        throw(ArgumentError("transverse monitor spectra are nonfinite"))
    # Explicit calls specialize on each buffer type; no heterogeneous dynamic scan.
    (
        Base.mightalias(nodes, node_spectrum) || Base.mightalias(nodes, column_spectrum) ||
            _transverse_alias_tables(nodes, tables) ||
            Base.mightalias(nodes, reading.columns) ||
            Base.mightalias(nodes, reading.strip) ||
            Base.mightalias(reading.columns, node_spectrum) ||
            Base.mightalias(reading.columns, column_spectrum) ||
            _transverse_alias_tables(reading.columns, tables) ||
            Base.mightalias(reading.columns, reading.strip)
    ) &&
        throw(ArgumentError("transverse reduction storage aliases its inputs"))
    return nothing
end

"""
    transverse_reading!(reading, node_spectrum, column_spectrum, tables, nodes,
                        execution; full=false)

Reduce actual physical-node and spectral-column matrices into an edge reading.

Matrices are `(n_omega,n_nodes)` and `(n_omega,n_columns)` for admitted scalar routes.
The caller supplies shared spatial transforms, including zeroed inactive nodes.
`nodes` is caller-owned binary64 scratch. `full=true` certifies all launch nodes were
prepared. Ordinary reads may omit unused nodes but must include all rate/face weights.
Only raw squared norms are reduced here. Strip/current flags are cleared; their
fresh leaves run afterwards. Invalid inputs raise before any scratch write.
"""
function transverse_reading!(
        reading, node_spectrum, column_spectrum, tables,
        nodes, execution; full::Bool = false
    )
    _transverse_numerical_check(reading, node_spectrum, column_spectrum, tables, nodes)
    each_column!(execution, length(nodes)) do node
        value = 0.0
        for row in axes(node_spectrum, 1)
            value += _monitor_pair(node_spectrum[row, node])
        end
        nodes[node] = value
    end
    columns = reading.columns
    each_column!(execution, length(columns)) do column
        value = 0.0
        for row in axes(column_spectrum, 1)
            value += _monitor_pair(column_spectrum[row, column])
        end
        columns[column] = value
    end
    faces = clip = totals = (0.0, 0.0, 0.0, 0.0)
    for node in eachindex(nodes)
        value = nodes[node]
        faces = (
            faces[1] + tables.faces[node, 1] * value,
            faces[2] + tables.faces[node, 2] * value,
            faces[3] + tables.faces[node, 3] * value,
            faces[4] + tables.faces[node, 4] * value,
        )
        clip = (
            clip[1] + tables.clip[node, 1] * value,
            clip[2] + tables.clip[node, 2] * value,
            clip[3] + tables.clip[node, 3] * value,
            clip[4] + tables.clip[node, 4] * value,
        )
        totals = (
            totals[1] + tables.totals[node, 1] * value,
            totals[2] + tables.totals[node, 2] * value,
            totals[3] + tables.totals[node, 3] * value,
            totals[4] + tables.totals[node, 4] * value,
        )
    end
    all(isfinite, columns) && all(isfinite, faces) && all(isfinite, clip) &&
        all(isfinite, totals) || throw(ArgumentError("transverse monitor sums overflowed"))
    reading.faces, reading.clip, reading.totals = faces, clip, totals
    reading.has_faces, reading.full = true, full
    reading.has_strip, reading.has_current = false, false
    return nothing
end

"""
    TransverseStripLine(spectrum, spacing, measure, first, last)

Retain one line's prepared windowed, twice-padded, unnormalised forward spectrum.

`spectrum` is `(2*n_strip,n_traces)` complex storage. The inward bin interval is
one-based inclusive `first:last`; an empty interval selects no inward energy. `spacing`
is in m; `measure` is the transverse integration factor. A zero-row spectrum represents
an unresolved empty strip and yields exact zeros. The shared transform owner prepares
its Kaiser window and every padding cell, with no FFT or window policy chosen here.
"""
struct TransverseStripLine{A}
    spectrum::A
    spacing::Float64
    measure::Float64
    first::Int
    last::Int

    function TransverseStripLine(
            spectrum, spacing::Real, measure::Real,
            first::Int, last::Int
        )
        ndims(spectrum) == 2 && isfinite(spacing) && spacing > 0 &&
            isfinite(measure) && measure > 0 && 1 <= first <= size(spectrum, 1) + 1 &&
            0 <= last <= size(spectrum, 1) ||
            throw(ArgumentError("invalid transverse strip"))
        return new{typeof(spectrum)}(
            spectrum, Float64(spacing), Float64(measure), first, last
        )
    end
end

"""
    transverse_strip!(reading, lines, execution)

Fill every inward/windowed strip pair from a concrete tuple of prepared spectra.

Each energy is `sum(abs2(FFT))*spacing/n_fft*measure`; only the supplied inward bins
enter its first member. This matches the neutral strip definition. All spectra must
be finite and disjoint from the writable strip. The session computes fresh strip
spectra for the observed field before invoking this reduction.
"""
function transverse_strip!(reading, lines::Tuple, execution)
    size(reading.strip) == (length(lines), 2) ||
        throw(DimensionMismatch("transverse strip destination does not match its lines"))
    Base.mightalias(reading.strip, reading.columns) &&
        throw(ArgumentError("strip output aliases retained column energies"))
    for (index, line) in pairs(lines)
        spectrum = line.spectrum
        all(isfinite, spectrum) || throw(ArgumentError("strip spectrum is nonfinite"))
        Base.mightalias(reading.strip, spectrum) &&
            throw(ArgumentError("strip output aliases its spectrum"))
    end
    for (index, line) in pairs(lines)
        spectrum, spacing, measure = line.spectrum, line.spacing, line.measure
        inward = total = 0.0
        count = size(spectrum, 1)
        if count > 0
            for column in axes(spectrum, 2), row in axes(spectrum, 1)
                energy = _monitor_pair(spectrum[row, column]) * spacing / count
                total += energy
                line.first <= row <= line.last && (inward += energy)
            end
        end
        reading.strip[index, 1], reading.strip[index, 2] = inward * measure, total * measure
    end
    all(isfinite, reading.strip) || throw(ArgumentError("transverse strip sums overflowed"))
    reading.has_strip = true
    return nothing
end

"""
    TransverseCurrentLine(samples, node, before, after, onset_step, measure)

Retain samples and a centred derivative stencil across one collar onset.

`samples` is `(n_axis_nodes,n_traces)` complex data from shared spatial transforms.
Radial samples already contain their translated square-root-radius scaling. Indices
are one-based, `onset_step` is the positive physical denominator in m, and `measure`
is the positive line quadrature factor. `node=0` declares no resolved onset and yields
zero current. No geometry or physical wavenumber is inferred from these arrays.
"""
struct TransverseCurrentLine{A}
    samples::A
    node::Int
    before::Int
    after::Int
    onset_step::Float64
    measure::Float64

    function TransverseCurrentLine(
            samples, node::Int, before::Int, after::Int,
            onset_step::Real, measure::Real
        )
        ndims(samples) == 2 && isfinite(onset_step) && onset_step > 0 &&
            isfinite(measure) && measure > 0 && (
            node == 0 ||
                all(index -> 1 <= index <= size(samples, 1), (node, before, after))
        ) ||
            throw(ArgumentError("invalid transverse current stencil"))
        return new{typeof(samples)}(
            samples, node, before, after,
            Float64(onset_step), Float64(measure)
        )
    end
end

"""
    transverse_current!(reading, lines, k_min)

Reduce rectified inward onset current from explicit line stencils.

The neutral convention gives `Im(conj(E)*(E_after-E_before)/onset_step)/k_min`;
rectify each trace at zero before summing and applying its measure. `k_min` is positive
in rad/m. The result is proxy energy per propagation metre, independent of launch
normalisation. Finite inputs and a representable result are required.
"""
function transverse_current!(reading, lines::Tuple, k_min)
    isfinite(k_min) && k_min > 0 || throw(ArgumentError("current needs positive k_min"))
    total = 0.0
    for line in lines
        line.node == 0 && continue
        samples = line.samples
        all(isfinite, samples) || throw(ArgumentError("current samples are nonfinite"))
        value = 0.0
        for column in axes(samples, 2)
            slope = (samples[line.after, column] - samples[line.before, column]) /
                line.onset_step
            field = samples[line.node, column]
            flux = (
                Float64(real(field)) * Float64(imag(slope)) -
                    Float64(imag(field)) * Float64(real(slope))
            ) / k_min
            value += max(flux, 0.0)
        end
        total += value * line.measure
    end
    isfinite(total) || throw(ArgumentError("transverse current overflowed"))
    reading.current, reading.has_current = total, true
    return nothing
end

"""
    TransverseState(tables)

Allocate concrete transverse control state and owned previous/noise column vectors.

Face baselines and maxima are fixed tuples in declared order. All coordinates are in
m and totals are launch-normalised energy shares. `last_columns` and `noise_columns`
are the only retained arrays; no history or notices are stored inside the state.
"""
mutable struct TransverseState{N, V}
    launch::Float64
    baseline::NTuple{N, Float64}
    peak::NTuple{N, Float64}
    peak_raw::NTuple{N, Float64}
    allowance::Float64
    noise_columns::V
    has_noise::Bool
    noise_share_in_ramp::Float64
    ramp_raw::Float64
    ramp_noise::Float64
    spatial_removed::Float64
    last_rate_z::Float64
    last_rate::Float64
    last_columns::V
    last_z::Float64
    next::Float64
    spacing_now::Float64
    backstop::Float64
    reach_reference::Float64
    reads::Int
    largest_gap::Float64
    warned_at::Float64
    status::TransverseStatus
    reason::UInt8
    warned::Bool
    ramp_warned::Bool
    resumed::Bool
    declared_share::Float64
    strip_baseline::Float64
    strip_floor::Float64
    strip_peak::Float64
    current::Float64
    has_current_last::Bool
    current_last_z::Float64
    current_last::Float64
    current_baseline::Float64
    returned_peak::Float64

    # Binding the face count through tables avoids @kwdef's unparameterized
    # constructor evaluating defaults with an unbound N (issue #645).
    function TransverseState(tables::TransverseTables{F, C, T, V, K, I, R, N}) where
        {F, C, T, V, K, I, R, N}
        state = new{N, Vector{Float64}}()
        state.launch = NaN
        state.baseline = ntuple(_ -> 0.0, Val(N))
        state.peak = ntuple(_ -> 0.0, Val(N))
        state.peak_raw = ntuple(_ -> 0.0, Val(N))
        state.allowance = 0.0
        state.noise_columns = zeros(length(tables.column_weights))
        state.has_noise = false
        state.noise_share_in_ramp = 0.0
        state.ramp_raw = 0.0
        state.ramp_noise = 0.0
        state.spatial_removed = 0.0
        state.last_rate_z = NaN
        state.last_rate = 0.0
        state.last_columns = zeros(length(tables.column_weights))
        state.last_z = NaN
        state.next = Inf
        state.spacing_now = Inf
        state.backstop = Inf
        state.reach_reference = 0.0
        state.reads = 0
        state.largest_gap = 0.0
        state.warned_at = NaN
        state.status = TransverseClear
        state.reason = 0
        state.warned = false
        state.ramp_warned = false
        state.resumed = false
        state.declared_share = 0.0
        state.strip_baseline = 0.0
        state.strip_floor = 0.0
        state.strip_peak = 0.0
        state.current = 0.0
        state.has_current_last = false
        state.current_last_z = NaN
        state.current_last = 0.0
        state.current_baseline = 0.0
        state.returned_peak = 0.0
        return state
    end
end

"""
    TransverseContinuation

Retain an immutable numerical public-record continuation, with fixed face tuples.

This preserves original launch/baselines, removal/noise/return totals, warning/read
accounting and a status. No spectral columns are copied: a resumed launch prepares
its fresh last columns and optional noise columns, then restarts the spacing rule.
"""
struct TransverseContinuation{N}
    launch::Float64
    baseline::NTuple{N, Float64}
    peak::NTuple{N, Float64}
    peak_raw::NTuple{N, Float64}
    allowance::Float64
    ramp_raw::Float64
    ramp_noise::Float64
    noise_share_in_ramp::Float64
    declared_share::Float64
    spatial_removed::Float64
    current::Float64
    current_baseline::Float64
    strip_baseline::Float64
    strip_floor::Float64
    strip_peak::Float64
    returned_peak::Float64
    reads::Int
    largest_gap::Float64
    warned_at::Float64
    status::TransverseStatus
end

function _transverse_check(state, tables, reading; full = false)
    length(state.baseline) == length(tables.face_ranks) ||
        throw(DimensionMismatch("transverse state and table face counts disagree"))
    reading.has_faces && (!full || reading.full) ||
        throw(ArgumentError("transverse observation is missing fresh node reductions"))
    length(state.last_columns) == length(state.noise_columns) ==
        length(reading.columns) == length(tables.column_weights) ||
        throw(DimensionMismatch("transverse control column dimensions disagree"))
    all(isfinite, reading.faces) && all(isfinite, reading.totals) &&
        (!full || all(isfinite, reading.clip)) &&
        all(value -> isfinite(value) && value >= 0, reading.columns) ||
        throw(ArgumentError("transverse observation is nonfinite or negative"))
    (
        Base.mightalias(state.last_columns, reading.columns) ||
            Base.mightalias(state.noise_columns, reading.columns) ||
            Base.mightalias(state.last_columns, state.noise_columns) ||
            Base.mightalias(state.last_columns, reading.strip) ||
            Base.mightalias(state.noise_columns, reading.strip) ||
            _transverse_alias_tables(state.last_columns, tables) ||
            _transverse_alias_tables(state.noise_columns, tables)
    ) &&
        throw(ArgumentError("transverse control storage aliases prepared columns"))
    return nothing
end

function _weighted_columns!(destination, columns, tables)
    for index in eachindex(destination, columns, tables.column_weights)
        destination[index] = columns[index] * tables.column_weights[index]
    end
    all(isfinite, destination) || throw(ArgumentError("weighted edge columns overflowed"))
    return nothing
end

function _transverse_reach(state, policy, tables)
    cumulative, selected = 0.0, 0.0
    for position in length(tables.order):-1:1
        column = tables.order[position]
        value = state.last_columns[column]
        state.has_noise && (value = max(value - state.noise_columns[column], 0.0))
        cumulative += value
        if cumulative / state.launch > policy.warn
            selected = tables.kappa[column]
            break
        end
    end
    isfinite(cumulative) || throw(ArgumentError("transverse spectral tail overflowed"))
    longitudinal = sqrt(max(policy.k_min * policy.k_min - selected * selected, 0.0))
    reach = selected / max(longitudinal, policy.k_floor_fraction * policy.k_min)
    isfinite(reach) || throw(ArgumentError("transverse reach is unrepresentable"))
    return reach
end

function _transverse_respace!(state, policy, tables; launch)
    reach = _transverse_reach(state, policy, tables)
    grown = reach > state.reach_reference * (1.0 + policy.reach_growth)
    if launch || grown || reach == 0.0
        state.backstop, state.reach_reference = policy.backstop_start, reach
    else
        state.backstop = min(2.0 * state.backstop, policy.backstop_max)
    end
    spacing = state.backstop
    reach > 0 && (spacing = min(spacing, policy.guard_numerator / reach))
    state.spacing_now = spacing
    return nothing
end

function _transverse_rate(policy, reading, launch)
    rate = policy.has_collar ? reading.totals[2] / launch : 0.0
    isfinite(rate) || throw(ArgumentError("normalised collar rate overflowed"))
    return rate
end

function _strip_shares(reading, launch)
    reading.has_strip || throw(ArgumentError("requested collar needs a fresh return strip"))
    all(value -> isfinite(value) && value >= 0, reading.strip) ||
        throw(ArgumentError("transverse return strip is invalid"))
    best = windowed = 0.0
    for line in axes(reading.strip, 1)
        share = reading.strip[line, 1] / launch
        if share > best
            best, windowed = share, reading.strip[line, 2] / launch
        end
    end
    all(isfinite, (best, windowed)) ||
        throw(ArgumentError("normalised return strip overflowed"))
    return best, windowed
end

function _transverse_ramp_band!(state, policy, tables, columns, declared, notices)
    policy.has_design || return nothing
    total = ramp = 0.0
    for index in eachindex(columns, tables.column_weights)
        value = columns[index] * tables.column_weights[index]
        total += value
        tables.ramp_rates[index] > 0 && (ramp += value)
    end
    all(isfinite, (total, ramp)) || throw(ArgumentError("launch ramp-band sum overflowed"))
    share = total > 0 ? ramp / total : 0.0
    if declared
        state.declared_share = share
    elseif share > policy.stop
        _notice!(notices, TransverseLaunchStop, NaN, share)
    elseif share > policy.warn
        _notice!(notices, TransverseLaunchWarning, NaN, share)
    end
    return nothing
end

"""
    transverse_launch!(state, policy, tables, reading, z, notices;
        noise=nothing, clean_columns=nothing, fixed_step_max=nothing,
        reads=true, declared=false)

Record or resume the transverse launch and establish its next read spacing.

`reading` has complete launch nodes, except on resume. Optional `noise` is its own
fresh full `TransverseReading`; `clean_columns` is the noise-free unweighted spectral
column energy used for the launch ramp-band judgement. Fixed steps are represented
only by their actual maximum. A resumed launch preserves baselines/totals, refreshes
column arrays and spacing, and does not judge again. All warnings/stops are fixed
notices; stop notices return before the later launch operations, matching the oracle.
"""
function transverse_launch!(
        state::TransverseState{N}, policy, tables, reading, z, notices;
        noise = nothing, clean_columns = nothing, fixed_step_max = nothing,
        reads::Bool = true, declared::Bool = false
    ) where {N}
    isfinite(z) || throw(ArgumentError("transverse launch coordinate must be finite"))
    _transverse_check(state, tables, reading; full = !state.resumed)
    if noise !== nothing
        _transverse_check(state, tables, noise; full = true)
    end
    if clean_columns !== nothing
        length(clean_columns) == length(tables.column_weights) &&
            all(value -> isfinite(value) && value >= 0, clean_columns) ||
            throw(ArgumentError("clean launch columns must be finite nonnegative"))
        (
            Base.mightalias(clean_columns, state.last_columns) ||
                Base.mightalias(clean_columns, state.noise_columns)
        ) &&
            throw(ArgumentError("control storage aliases clean launch columns"))
    end
    if reading.has_current
        isfinite(reading.current) && reading.current >= 0 ||
            throw(ArgumentError("transverse current must be finite nonnegative"))
    end
    if state.resumed
        state.last_z = z
        if noise !== nothing
            _weighted_columns!(state.noise_columns, noise.columns, tables)
            state.has_noise = true
        end
        _weighted_columns!(state.last_columns, reading.columns, tables)
        _transverse_respace!(state, policy, tables; launch = true)
        state.next = reads ? z + state.spacing_now : Inf
        state.last_rate_z = z
        state.last_rate = _transverse_rate(policy, reading, state.launch)
        if reading.has_current && state.launch > 0
            state.has_current_last = true
            state.current_last_z, state.current_last = z, reading.current / state.launch
            isfinite(state.current_last) ||
                throw(ArgumentError("normalised transverse current overflowed"))
        end
        !reads && (state.status = TransverseUnwatched)
        return nothing
    end
    state.launch, state.last_z = reading.totals[1], z
    if !(state.launch > 0)
        state.next = Inf
        return nothing
    end
    columns = clean_columns === nothing ? reading.columns : clean_columns
    length(columns) == length(tables.column_weights) &&
        all(value -> isfinite(value) && value >= 0, columns) ||
        throw(ArgumentError("clean launch columns must be finite nonnegative"))
    _transverse_ramp_band!(state, policy, tables, columns, declared, notices)
    monitor_stopped(notices) && return nothing
    if !reads
        state.status, state.reason = TransverseUnwatched, 1
        _notice!(notices, TransverseUnwatchedWarning, z, 0.0)
        return nothing
    end
    state.baseline = ntuple(index -> reading.faces[index] / state.launch, Val(N))
    all(isfinite, state.baseline) ||
        throw(ArgumentError("normalised launch faces overflowed"))
    for index in 1:N
        clipped = reading.clip[index] / state.launch
        isfinite(clipped) || throw(ArgumentError("normalised launch clip overflowed"))
        clipped > policy.warn &&
            _notice!(notices, TransverseClipWarning, z, clipped; face = index)
    end
    if noise !== nothing
        eta = noise.totals[1] / state.launch
        state.allowance = policy.noise_factor * eta * tables.noise_face_share /
            sqrt(tables.noise_modes)
        isfinite(state.allowance) || throw(ArgumentError("edge noise allowance overflowed"))
        _weighted_columns!(state.noise_columns, noise.columns, tables)
        state.has_noise = true
        if policy.has_design
            total = 0.0
            for index in eachindex(state.noise_columns)
                tables.ramp_rates[index] > 0 && (total += state.noise_columns[index])
            end
            state.noise_share_in_ramp = total / state.launch
            isfinite(state.noise_share_in_ramp) ||
                throw(ArgumentError("normalised ramp noise overflowed"))
        end
        if state.allowance > policy.warn
            state.status, state.reason = TransverseInsensitive, 2
            _notice!(notices, TransverseInsensitiveWarning, z, state.allowance)
        end
    end
    _weighted_columns!(state.last_columns, reading.columns, tables)
    _transverse_respace!(state, policy, tables; launch = true)
    state.next = z + state.spacing_now
    if fixed_step_max !== nothing
        isfinite(fixed_step_max) && fixed_step_max > 0 ||
            throw(ArgumentError("fixed maximum step must be finite positive"))
        if fixed_step_max > state.spacing_now
            state.status == TransverseClear && (state.status = TransverseCoarse)
            state.largest_gap = fixed_step_max
            _notice!(
                notices, TransverseCoarseWarning, z, fixed_step_max;
                auxiliary = state.spacing_now
            )
        end
    end
    state.last_rate_z, state.last_rate = z, _transverse_rate(policy, reading, state.launch)
    if policy.has_collar
        state.strip_baseline = first(_strip_shares(reading, state.launch))
        noise !== nothing && (
            state.strip_floor = policy.strip_floor_margin *
                first(_strip_shares(noise, state.launch))
        )
        state.current_baseline = reading.totals[3] / state.launch
        isfinite(state.current_baseline) ||
            throw(ArgumentError("normalised outside-launch energy overflowed"))
        state.has_current_last, state.current = false, 0.0
    end
    return nothing
end

"""
    transverse_due(state, policy, z, proposal=0.0)

Return whether the next proposed step crosses the retained transverse read position.

The lowered coordinate tolerance preserves the reference's exact comparison order.
"""
transverse_due(state, policy, z, proposal = 0.0) =
    z + proposal >= state.next - policy.due_tolerance * max(abs(z), 1.0)

"""
    transverse_wants_read(state, z)

Return whether a positive-launch, watched edge has a new position to record.
"""
transverse_wants_read(state, z) =
    state.launch > 0 && state.status != TransverseUnwatched && z > state.last_z

"""
    transverse_ceiling(state)

Return the read spacing in m, or Inf when no transverse read is scheduled.
"""
transverse_ceiling(state) = isfinite(state.next) ? state.spacing_now : Inf

"""
    transverse_ramp_removed(state)

Return signed ramp removal after subtracting at most its initial noise and declaration.
"""
transverse_ramp_removed(state) = state.ramp_raw -
    min(state.ramp_noise, state.noise_share_in_ramp) - state.declared_share

function _transverse_judge_ramp!(state, policy, notices)
    removed = transverse_ramp_removed(state)
    if removed > policy.stop
        state.status = TransverseStopped
        _notice!(notices, TransverseRampStop, NaN, removed; auxiliary = policy.stop)
    elseif removed > policy.warn && !state.ramp_warned
        state.ramp_warned = true
        state.status in (TransverseClear, TransverseCoarse) &&
            (state.status = TransverseWarned)
        _notice!(notices, TransverseRampWarning, NaN, removed)
    end
    return nothing
end

function _transverse_judge_return!(state, policy, reading, z, notices)
    strip, windowed = _strip_shares(reading, state.launch)
    state.strip_peak = max(state.strip_peak, strip)
    judged_strip = max(
        strip - state.strip_baseline - state.strip_floor -
            policy.strip_leakage * windowed, 0.0
    )
    judged_current = max(state.current - state.current_baseline, 0.0)
    returned = max(judged_strip, judged_current)
    state.returned_peak = max(state.returned_peak, returned)
    if returned > policy.stop
        state.status = TransverseStopped
        _notice!(notices, TransverseReturnStop, z, returned; auxiliary = policy.stop)
    elseif returned > policy.warn && !state.warned
        state.warned, state.warned_at, state.status = true, z, TransverseWarned
        _notice!(notices, TransverseReturnWarning, z, returned)
    end
    return nothing
end

"""
    transverse_read!(state, policy, tables, reading, z, notices)

Credit ramp/collar removal, refresh spacing, and judge a new transverse observation.

Column norms and node sums are ordinary bounded reductions. The retained descending
kappa prefix uses sequential accumulation, matching the exact bucket-selection rule.
Scalar thresholds use strict comparisons. Stop ties select the lexically last face;
warning order follows declared faces and emits once. With a collar, outward edge
shares are recorded while return-strip/current and safety-ramp removal are judged.
"""
function transverse_read!(
        state::TransverseState{N}, policy, tables, reading, z,
        notices
    ) where {N}
    transverse_wants_read(state, z) || return nothing
    isfinite(z) || throw(ArgumentError("transverse read coordinate must be finite"))
    _transverse_check(state, tables, reading)
    gap = z - state.last_z
    state.reads += 1
    state.largest_gap = max(state.largest_gap, gap)
    if policy.has_design
        raw = noise = 0.0
        for index in eachindex(state.last_columns)
            taken = -expm1(-tables.ramp_rates[index] * gap)
            raw += state.last_columns[index] * taken
            state.has_noise && (noise += state.noise_columns[index] * taken)
        end
        state.ramp_raw += raw / state.launch
        state.has_noise && (state.ramp_noise += noise / state.launch)
    end
    rate = _transverse_rate(policy, reading, state.launch)
    state.spatial_removed += 0.5 * (z - state.last_rate_z) * (state.last_rate + rate)
    all(isfinite, (state.ramp_raw, state.ramp_noise, state.spatial_removed)) ||
        throw(ArgumentError("transverse removal totals overflowed"))
    state.last_rate_z, state.last_rate, state.last_z = z, rate, z
    _weighted_columns!(state.last_columns, reading.columns, tables)
    _transverse_respace!(state, policy, tables; launch = false)
    state.next = z + state.spacing_now
    state.peak_raw = ntuple(
        index -> max(
            state.peak_raw[index],
            reading.faces[index] / state.launch
        ), Val(N)
    )
    all(isfinite, state.peak_raw) ||
        throw(ArgumentError("normalised transverse face overflowed"))
    if policy.has_collar
        _transverse_judge_return!(state, policy, reading, z, notices)
        monitor_stopped(notices) && return nothing
        _transverse_judge_ramp!(state, policy, notices)
        return nothing
    end
    judged = ntuple(
        index -> reading.faces[index] / state.launch -
            state.baseline[index] - state.allowance, Val(N)
    )
    all(isfinite, judged) || throw(ArgumentError("judged transverse face overflowed"))
    state.peak = ntuple(index -> max(state.peak[index], judged[index]), Val(N))
    worst = 1
    for index in 2:N
        if judged[index] > judged[worst] ||
                (
                judged[index] == judged[worst] &&
                    tables.face_ranks[index] > tables.face_ranks[worst]
            )
            worst = index
        end
    end
    if judged[worst] > policy.stop
        state.status = TransverseStopped
        _notice!(notices, TransverseFaceStop, z, judged[worst]; face = worst)
        return nothing
    end
    for index in 1:N
        if judged[index] > policy.warn && !state.warned
            state.warned, state.warned_at = true, z
            state.status in (TransverseClear, TransverseCoarse) &&
                (state.status = TransverseWarned)
            _notice!(notices, TransverseFaceWarning, z, judged[index]; face = index)
        end
    end
    _transverse_judge_ramp!(state, policy, notices)
    return nothing
end

"""
    transverse_step!(state, policy, current, z)

Accumulate a collar's rectified inward current at every committed step.

`current` is the fresh unnormalised reduction in proxy energy per m. Normalise by the
original launch and integrate with the preceding committed reading's trapezoid.
Without a collar, a positive launch or a watched edge, no total changes.
"""
function transverse_step!(state, policy, current, z)
    (
        !policy.has_collar || !(state.launch > 0) ||
            state.status == TransverseUnwatched
    ) && return nothing
    isfinite(current) && current >= 0 && isfinite(z) ||
        throw(ArgumentError("transverse current or coordinate is invalid"))
    share = current / state.launch
    if state.has_current_last
        state.current += 0.5 * (z - state.current_last_z) * (state.current_last + share)
    end
    isfinite(state.current) && isfinite(share) ||
        throw(ArgumentError("integrated transverse current overflowed"))
    state.has_current_last = true
    state.current_last_z, state.current_last = z, share
    return nothing
end

"""
    transverse_continuation(state)

Return immutable numerical restart data with the original face and launch baselines.
"""
function transverse_continuation(state)
    return TransverseContinuation(
        state.launch, state.baseline, state.peak, state.peak_raw,
        state.allowance, state.ramp_raw, state.ramp_noise, state.noise_share_in_ramp,
        state.declared_share, state.spatial_removed, state.current, state.current_baseline,
        state.strip_baseline, state.strip_floor, state.strip_peak, state.returned_peak,
        state.reads, state.largest_gap, state.warned_at, state.status
    )
end

"""
    restore_transverse!(state, continuation)

Restore a numerical public record into a fresh edge before its resumed launch.

A nonpositive/missing launch leaves fresh monitoring in place. Positive launches keep
all recorded baselines/totals. Only clear/warned/insensitive/coarse statuses survive,
as in the neutral resume contract; a new launch refreshes previous field samples.
"""
function restore_transverse!(state, continuation)
    isnan(state.last_z) || throw(ArgumentError("transverse restore needs fresh state"))
    (continuation === nothing || !(continuation.launch > 0)) && return nothing
    length(continuation.baseline) == length(state.baseline) ||
        throw(DimensionMismatch("resumed transverse faces do not match"))
    state.resumed, state.launch = true, continuation.launch
    state.baseline, state.peak, state.peak_raw = continuation.baseline,
        continuation.peak, continuation.peak_raw
    state.allowance = continuation.allowance
    state.ramp_raw, state.ramp_noise = continuation.ramp_raw, continuation.ramp_noise
    state.noise_share_in_ramp = continuation.noise_share_in_ramp
    state.declared_share, state.spatial_removed = continuation.declared_share,
        continuation.spatial_removed
    state.current = continuation.current
    state.current_baseline = continuation.current_baseline
    state.strip_baseline = continuation.strip_baseline
    state.strip_floor = continuation.strip_floor
    state.strip_peak = continuation.strip_peak
    state.returned_peak = continuation.returned_peak
    state.reads, state.largest_gap = continuation.reads, continuation.largest_gap
    state.warned_at, state.warned = continuation.warned_at, isfinite(continuation.warned_at)
    continuation.status in (
        TransverseClear, TransverseWarned,
        TransverseInsensitive, TransverseCoarse,
    ) && (state.status = continuation.status)
    return nothing
end

"""
    transverse_finish!(state)

Return the fixed closing verdict for session-side logging and record assembly.

No numerical state changes: status, largest judged face share, signed ramp removal
and read count are an isbits named tuple. The session owns text and delivery.
"""
function transverse_finish!(state)
    return (
        status = state.status, largest = maximum(state.peak),
        ramp_removed = transverse_ramp_removed(state), reads = state.reads,
    )
end
