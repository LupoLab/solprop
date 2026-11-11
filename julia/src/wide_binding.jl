# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public WideStartSpec, WideStartHistory, WideBoundRate, WideStartBinding
public prepare_wide_start, reconstruct_wide_rate

_wide_start_nodes(::Nothing, method) = method === :low_storage ? (0.0, 0.5, 1.0) :
    (0.0, 1.0)
_wide_start_nodes(tableau::EmbeddedTableau, method) = (tableau.nodes..., 1.0)

function _wide_start_replay(start, tableau, schedule, geometry, method)
    isempty(schedule.replay_steps) && return nothing
    edges, steps = schedule.replay_edges, schedule.replay_steps
    first_index = argmin(abs.(edges .- start))
    slack = _replay_slack(schedule, start)
    abs(edges[first_index] - start) <= slack ||
        throw(ArgumentError("wide replay must start at an actual recorded edge"))
    finish = last(schedule.planes)
    last(edges) >= finish - _replay_slack(schedule, finish) ||
        throw(ArgumentError("wide replay does not cover the requested end"))
    for index in first_index:length(steps)
        edges[index] >= finish - _replay_slack(schedule, finish) && break
        edges[index + 1] <= finish + _replay_slack(schedule, finish) ||
            throw(ArgumentError("wide replay crosses the requested end"))
        geometry === nothing &&
            throw(ArgumentError("a pending wide replay needs positive-step geometry"))
        step = steps[index]
        geometry.step_min <= step <= geometry.step_max ||
            throw(ArgumentError("wide geometry omits an actual replay length"))
        origin = index == first_index ? start : edges[index]
        geometry.lower <= edges[index + 1] <= geometry.upper ||
            throw(ArgumentError("wide geometry omits an actual replay edge"))
        for node in _wide_start_nodes(tableau, method)
            # Match the noncontracted scalar-coordinate graph used by the march.
            position = origin + node * step
            isfinite(position) && geometry.lower <= position <= geometry.upper ||
                throw(ArgumentError("wide geometry omits an actual replay stage"))
        end
    end
    return nothing
end

"""
    WideStartSpec(start, tableau, schedule; geometry, arithmetic, method,
                  order_max, source_absent)

Bind actual immutable method inputs to a CPU DF32 start's coordinate domain.

`start` is the supplied binary64 accepted coordinate in metres. `schedule` is the
actual `MarchSchedule`; its arrays are immutable numerical inputs after construction.
`method` is `:rk`, `:low_storage` or `:adams`. Only `:rk` takes its actual
`EmbeddedTableau`; the other methods require `nothing`. `order_max` is the configured
Adams maximum in `1:6`, or zero for the other methods. `arithmetic` must specify
binary32 state and its original-launch scales. `source_absent` is a structural fact.

`geometry` is the conservative `PhaseSpans` lowering, not an assembly certificate.
It must cover actual replay lengths, raw stage coordinates, recorded edges and
retained samples. Replay can exceed the adaptive ceiling. A start exactly at the
final plane requires `geometry=nothing`, without inventing a positive step. Adaptive
geometry must not declare a positive minimum step. Throw `ArgumentError` for a
method/precision/geometry mismatch before preparing any numerical state.
"""
struct WideStartSpec{T, G}
    start::Float64
    tableau::T
    schedule::MarchSchedule
    geometry::G
    arithmetic::SourceArithmetic{Float32}
    method::Symbol
    order_max::Int
    source_absent::Bool

    function WideStartSpec(
            start::Float64, tableau::T,
            schedule::MarchSchedule;
            geometry::G, arithmetic::SourceArithmetic,
            method::Symbol, order_max::Int, source_absent::Bool,
        ) where {T <: Union{Nothing, EmbeddedTableau}, G <: Union{Nothing, WideGeometry}}
        arithmetic isa SourceArithmetic{Float32} ||
            throw(ArgumentError("DF32 wide binding requires binary32 state arithmetic"))
        method in (:rk, :low_storage, :adams) ||
            throw(ArgumentError("wide binding requires a supported march family"))
        (method === :rk) == (tableau isa EmbeddedTableau) ||
            throw(ArgumentError("only wide RK binding takes its actual tableau"))
        (method === :adams ? 1 <= order_max <= 6 : iszero(order_max)) ||
            throw(ArgumentError("wide binding needs the configured maximum Adams order"))
        method === :low_storage && isempty(schedule.replay_steps) &&
            throw(ArgumentError("wide low-storage binding requires fixed replay"))
        finish = last(schedule.planes)
        isfinite(start) && first(schedule.breakpoints) <= start <= finish ||
            throw(ArgumentError("wide start lies outside the actual requested interval"))
        (start == finish) == (geometry === nothing) ||
            throw(ArgumentError("only a completed wide start has no step geometry"))
        if geometry !== nothing
            geometry.lower <= start <= finish <= geometry.upper &&
                geometry.history_lower <= start <= geometry.history_upper ||
                throw(ArgumentError("wide geometry omits the actual start or finish"))
            isempty(schedule.replay_steps) && !iszero(geometry.step_min) &&
                throw(ArgumentError("adaptive wide binding has no positive step floor"))
        end
        _wide_start_replay(start, tableau, schedule, geometry, method)
        return new{T, G}(
            start, tableau, schedule, geometry, arithmetic, method, order_max,
            source_absent,
        )
    end
end

"""
    WideStartHistory(positions, identities, slots; rates=nothing, origin_wide=:unknown)

Borrow actual oldest-first restored history metadata for one preparation call.

Positions are a binary64 vector in metres; identities are UInt8 accepted `0` or
predicted `1`; slots are zero-based Int64 indices. A compensated history supplies
its full ComplexF64 rate store `(n_omega,n_columns,order_max+2)`. Plain history keeps
its coordinates and age with `rates=nothing`. `origin_wide` is `:binary64`, `:df32`
or `:unknown`; it never supplies a numerical error bound. All arrays need strided
CPU storage and remain borrowed only until `prepare_wide_start` returns. Their
compatibility, shape, finiteness and actual selection are checked there before writes.
Throw `ArgumentError` for unsupported storage or provenance.
"""
struct WideStartHistory{P, I, S, R}
    positions::P
    identities::I
    slots::S
    rates::R
    origin_wide::Symbol

    function WideStartHistory(
            positions::P, identities::I, slots::S;
            rates::R = nothing, origin_wide::Symbol = :unknown,
        ) where {P, I, S, R}
        positions isa StridedVector{Float64} && identities isa StridedVector{UInt8} &&
            slots isa StridedVector{Int64} ||
            throw(ArgumentError("wide history requires binary64/UInt8/Int64 vectors"))
        rates === nothing || rates isa StridedArray{ComplexF64, 3} ||
            throw(ArgumentError("wide history rates require strided ComplexF64 storage"))
        _wide_origin(origin_wide)
        return new{P, I, S, R}(positions, identities, slots, rates, origin_wide)
    end
end

function _wide_origin(origin::Symbol)
    origin in (:unknown, :binary64, :df32) ||
        throw(ArgumentError("wide history origin requires binary64, df32 or unknown"))
    return nothing
end

"""
    WideBoundRate

Retain a saved binary64 rate and its conservatively reconstructed DF32 provenance.

Construct with `reconstruct_wide_rate`. `saved` preserves every raw ComplexF64 value;
`sample::WideRate` contains its DF32 conversion and componentwise absolute radii
relative to the bound represented derivative. `origin_wide` retains known previous
arithmetic, or `:unknown`, without implying exact prior evaluation. These arrays are
immutable numerical inputs except for the documented `wide_rate!` refresh mechanism;
refreshing the sample invalidates operations holding its previous generation.
"""
struct WideBoundRate
    saved::Matrix{ComplexF64}
    sample::WideRate
    origin_wide::Symbol

    function WideBoundRate(
            saved::Matrix{ComplexF64}, sample::WideRate, origin_wide::Symbol,
        )
        return new(saved, sample, origin_wide)
    end
end

function _wide_saved_radius(saved::Float64, value::DF32, radius::Float64)
    # Bound the discrepancy from both exact input words, including each binary64
    # subtraction rounding. This must not treat a rounded equality as exact proof.
    difference = (_wide_host(saved) - _wide_host(Float64(value.hi))) -
        _wide_host(Float64(value.lo))
    return _wide_sum(radius, _wide_sum(abs(difference.value), difference.error))
end

"""
    reconstruct_wide_rate(execution, prepared, position, values;
                          before=false, origin_wide=:unknown)

Derive a conservative prior-error radius while retaining the actual saved rate.

`values` is a finite strided ComplexF64 matrix matching the bound operator, in m⁻¹;
`position` is its unchanged binary64 coordinate in metres. Evaluate the exact same
representation's derivative with the supplied knot side (`before=true` selects
the left side). If the certified derivative is `d` with radius `e`, then
`e+abs(values-d)` bounds the saved rate's discrepancy from that derivative, with
outward binary64 arithmetic including both DF32 words. Restoring these actual values
adds their new DF32 conversion error. No saved rate is replaced by a new derivative.

Return an owned `WideBoundRate`. Preparation may allocate, never mutates an input,
and retains no borrowed storage. Throw `ArgumentError` for nonfinite data, storage,
provenance or certificate range failures, and `DimensionMismatch` for another shape.
This rate radius still needs propagation through a complete Adams factor certificate.
"""
function reconstruct_wide_rate(
        execution::KernelExecution, prepared::WideOperator, position::Float64,
        values::AbstractMatrix{ComplexF64}; before::Bool = false,
        origin_wide::Symbol = :unknown,
    )
    oncpu(execution) || throw(ArgumentError("wide saved-rate certification requires CPU"))
    _wide_origin(origin_wide)
    values isa StridedMatrix{ComplexF64} ||
        throw(ArgumentError("wide saved rates require strided binary64 storage"))
    size(values) == prepared.shape ||
        throw(DimensionMismatch("wide saved rates must match the bound operator"))
    all(isfinite, values) || throw(ArgumentError("wide saved rates must be finite"))
    derivative = prepare_wide_rate(execution, prepared, position; before)
    saved = _wide_host_copy(values)
    prior = similar(saved)
    for index in eachindex(saved)
        value, radius = derivative.values[index], derivative.errors[index]
        prior[index] = ComplexF64(
            _wide_saved_radius(real(saved[index]), real(value), real(radius)),
            _wide_saved_radius(imag(saved[index]), imag(value), imag(radius)),
        )
    end
    sample = restore_wide_rate(prepared, position, saved; prior_errors = prior, before)
    return WideBoundRate(saved, sample, origin_wide)
end

"""
    WideStartBinding

Retain an actual CPU DF32 start's inputs and restored rate provenance.

Construct with `prepare_wide_start`. The record holds its spec, an owned
`WideOperator` (or `nothing` for a completed start), owned oldest-first positions,
UInt8 identities, zero-based slots, selected `WideBoundRate` records, and origin-wide
metadata. Plain history retains positions/kinds/slots even without rates. Completed
starts retain metadata but prepare no DF32 rate or operator buffers. No field claims
run admission: every actual numerical operation and complete factor still needs its
certificate before writes. There is no global cache and no reference to an owner,
session, source or Python object.
"""
struct WideStartBinding{S, O}
    spec::S
    operator::O
    positions::Vector{Float64}
    identities::Vector{UInt8}
    slots::Vector{Int64}
    rates::Vector{WideBoundRate}
    origin_wide::Symbol

    function WideStartBinding(
            spec::S, operator::O, positions::Vector{Float64}, identities::Vector{UInt8},
            slots::Vector{Int64}, rates::Vector{WideBoundRate}, origin_wide::Symbol,
        ) where {S <: WideStartSpec, O <: Union{Nothing, WideOperator}}
        return new{S, O}(spec, operator, positions, identities, slots, rates, origin_wide)
    end
end

function _wide_start_shape(operator::DiagonalOperator)
    operator.rate isa StridedMatrix{ComplexF64} ||
        throw(ArgumentError("wide binding requires strided CPU binary64 inputs"))
    return size(operator.rate)
end

function _wide_start_table(table)
    table.breakpoints isa StridedVector{Float64} && table.coefficients isa Union{
        StridedArray{Float64, 3}, StridedArray{ComplexF64, 3},
    } || throw(ArgumentError("wide binding tables require strided CPU binary64 inputs"))
    return nothing
end

function _wide_start_shape(operator::RepresentedOperator)
    _wide_start_table(operator.cumulative)
    return (operator.n_frequency, operator.n_columns)
end

function _wide_start_shape(operator::ExpansionOperator)
    table = operator.cumulative_weights
    _wide_start_table(table)
    operator.terms isa StridedArray{ComplexF64, 3} &&
        eltype(table.coefficients) === Float64 ||
        throw(ArgumentError("wide binding needs binary64 complex terms and real weights"))
    return size(operator.terms)[1:2]
end

function _wide_history_check(history::WideStartHistory, spec, shape, compensated)
    spec.method === :adams ||
        throw(ArgumentError("only an Adams wide start may restore history"))
    positions, slots, kinds = history.positions, history.slots, history.identities
    count = length(positions)
    1 <= count <= spec.order_max + 1 && length(slots) == count && length(kinds) == count &&
        all(isfinite, positions) && all(>(0), diff(positions)) &&
        all(kind -> kind in (0, 1), kinds) && length(unique(slots)) == count &&
        all(slot -> 0 <= slot < spec.order_max + 2, slots) ||
        throw(ArgumentError("wide restored history has invalid coordinates or selection"))
    newest = last(positions)
    abs(newest - spec.start) <= 1.0e-12 * max(abs(newest), abs(spec.start)) ||
        throw(ArgumentError("wide restored history is not at its actual accepted start"))
    geometry = spec.geometry
    if geometry !== nothing
        all(z -> geometry.history_lower <= z <= geometry.history_upper, positions) &&
            all(z -> geometry.lower <= z <= geometry.upper, positions) ||
            throw(ArgumentError("wide geometry omits actual retained history ages"))
    end
    (history.rates !== nothing) == compensated ||
        throw(ArgumentError("wide restored rates disagree with compensation policy"))
    if history.rates !== nothing
        size(history.rates) == (shape..., spec.order_max + 2) ||
            throw(DimensionMismatch("wide restored rates require the full store extent"))
        all(isfinite, history.rates) ||
            throw(ArgumentError("wide restored rate stores must be finite"))
    end
    return nothing
end

"""
    prepare_wide_start(execution, operator, spec; history=nothing)

Prepare owned CPU DF32 operator and actual restored-history bindings.

`operator` is a binary64 `DiagonalOperator`, `RepresentedOperator` or
`ExpansionOperator`. `spec::WideStartSpec` binds the exact start and method geometry.
Optional `history::WideStartHistory` supplies actual retained coordinates and full
saved rate storage. Accepted samples use the right derivative side; predicted
samples use the left. Plain history ages remain present without sampled rates.

Return a `WideStartBinding` with no borrowed arrays and no run-certificate flag.
Only the immutable method/schedule inputs in the spec are shared. A completed start
retains history metadata and validates binary64 storage but performs no numerical
assembly or DF32 range check. Throw `ArgumentError`
for unsupported CPU/operator/method/range/history and `DimensionMismatch` for an
incompatible rate store. A failure changes no caller field, history or generation.
"""
function prepare_wide_start(
        execution::KernelExecution,
        operator::Union{DiagonalOperator, RepresentedOperator, ExpansionOperator},
        spec::WideStartSpec; history::Union{Nothing, WideStartHistory} = nothing,
    )
    oncpu(execution) || throw(ArgumentError("wide start binding requires CPU execution"))
    shape = _wide_start_shape(operator)
    compensated = spec.method === :adams && !(operator isa DiagonalOperator) &&
        !spec.source_absent
    history === nothing || _wide_history_check(history, spec, shape, compensated)
    prepared = spec.geometry === nothing ? nothing :
        prepare_wide_operator(execution, operator; geometry = spec.geometry)
    positions = history === nothing ? Float64[] : _wide_host_copy(history.positions)
    identities = history === nothing ? UInt8[] : _wide_host_copy(history.identities)
    slots = history === nothing ? Int64[] : _wide_host_copy(history.slots)
    rates = WideBoundRate[]
    if history !== nothing && history.rates !== nothing && prepared !== nothing
        for index in eachindex(positions)
            values = view(history.rates, :, :, slots[index] + 1)
            push!(
                rates, reconstruct_wide_rate(
                    execution, prepared, positions[index], values;
                    before = identities[index] == 1, origin_wide = history.origin_wide,
                )
            )
        end
    end
    return WideStartBinding(
        spec, prepared, positions, identities, slots, rates,
        history === nothing ? :unknown : history.origin_wide,
    )
end

function prepare_wide_start(execution, operator, spec::WideStartSpec; history = nothing)
    throw(ArgumentError("DF32 start binding has no qualified operator representation"))
end
