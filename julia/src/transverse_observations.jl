# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public TransverseLineData, TransversePreparation, TransverseWorkspace
public TransverseMonitorRecord, transverse_clean_columns, transverse_fallback_clean_columns
public transverse_monitor_launch!, transverse_monitor_endpoint!, transverse_monitor_plane!
public transverse_monitor_proceed!, transverse_monitor_ceiling
public transverse_monitor_record, transverse_monitor_finish

"""
    TransverseLineData(axis, sample_scale; strip, window, spacing, measure,
                       inward, onset, onset_step)

Copy one translated scalar spatial collar's outward-normal sampling description.

`axis` is `:r`, `:x` or `:y`. The finite nonnegative binary64 `sample_scale` contains
sqrt(r) for radial samples and ones for Cartesian/line samples. `strip` contains
distinct one-based axis nodes in outward order; `window` is the corresponding
nonnegative Kaiser window. `spacing` is their positive mean spacing in m, and
`measure` the positive transverse integration factor. `inward=(first,last)` selects
one-based inclusive bins in the twice-padded raw forward DFT. An empty strip requires
an empty window, zero spacing and `(1,0)` bins; it has no transform and returns zero.

`onset` is `(node,before,after)` in outward order, with positive physical derivative
denominator `onset_step` in m. `nothing` and zero step specify an unresolved onset
whose current is exactly zero. All geometry, scales, windows and bins are supplied by
the existing SEP 40 translation. Invalid data raise before allocating FFT storage.
"""
struct TransverseLineData
    axis::Symbol
    sample_scale::Vector{Float64}
    strip::Vector{Int}
    window::Vector{Float64}
    spacing::Float64
    measure::Float64
    inward::NTuple{2, Int}
    onset::Union{Nothing, NTuple{3, Int}}
    onset_step::Float64

    function TransverseLineData(
            axis::Symbol, sample_scale::AbstractVector{Float64};
            strip::AbstractVector{Int}, window::AbstractVector{Float64},
            spacing::Float64, measure::Float64, inward::NTuple{2, Int},
            onset::Union{Nothing, NTuple{3, Int}}, onset_step::Float64,
        )
        axis in (:r, :x, :y) && !isempty(sample_scale) &&
            all(value -> isfinite(value) && value >= 0, sample_scale) ||
            throw(ArgumentError("transverse line needs a declared axis and finite scales"))
        length(strip) == length(window) && length(unique(strip)) == length(strip) &&
            all(index -> 1 <= index <= length(sample_scale), strip) &&
            all(value -> isfinite(value) && value >= 0, window) ||
            throw(ArgumentError("transverse strip nodes or window are invalid"))
        isfinite(measure) && measure > 0 && isfinite(spacing) &&
            isfinite(onset_step) ||
            throw(ArgumentError("transverse line measures must be finite"))
        if isempty(strip)
            spacing == 0 && inward == (1, 0) ||
                throw(ArgumentError("an unresolved strip needs zero spacing and no bins"))
        else
            count = Base.checked_mul(2, length(strip))
            length(strip) >= 4 && spacing > 0 && 1 <= inward[1] <= count + 1 &&
                0 <= inward[2] <= count ||
                throw(ArgumentError("a resolved strip needs its positive spacing and bins"))
        end
        if onset === nothing
            onset_step == 0 ||
                throw(ArgumentError("an unresolved current onset needs zero step"))
        else
            onset_step > 0 && length(unique(onset)) == 3 &&
                all(index -> 1 <= index <= length(sample_scale), onset) ||
                throw(ArgumentError("transverse onset or derivative step is invalid"))
        end
        return new(
            axis, Vector{Float64}(sample_scale), Vector{Int}(strip),
            Vector{Float64}(window), spacing, measure, inward, onset, onset_step,
        )
    end
end

function _transverse_copy_tables(tables::TransverseTables)
    arrays = (
        tables.faces, tables.clip, tables.totals, tables.column_weights,
        tables.kappa, tables.ramp_rates,
    )
    all(array -> eltype(array) === Float64, arrays) ||
        throw(ArgumentError("transverse monitoring currently requires binary64 tables"))
    return TransverseTables(
        Matrix{Float64}(tables.faces), Matrix{Float64}(tables.clip),
        Matrix{Float64}(tables.totals), Vector{Float64}(tables.column_weights),
        Vector{Float64}(tables.kappa), Vector{Int}(tables.order),
        Vector{Float64}(tables.ramp_rates); face_ranks = tables.face_ranks,
        noise_face_share = tables.noise_face_share, noise_modes = tables.noise_modes,
    )
end

function _transverse_line_shape(map::RadialObservationMap, line)
    line.axis == :r || throw(ArgumentError("radial transverse lines require the r axis"))
    return size(map.synthesis, 1), 1
end

function _transverse_line_shape(map::CartesianObservationMap, line)
    nx, ny = map.shape
    line.axis == :x && return nx, ny
    line.axis == :y && ny > 1 && return ny, nx
    throw(ArgumentError("Cartesian/line transverse sampling has an incompatible axis"))
end

function _transverse_line_node(map::RadialObservationMap, line, node, cross)
    return node
end

function _transverse_line_node(map::CartesianObservationMap, line, node, cross)
    return line.axis == :x ? node + map.shape[1] * (cross - 1) :
        cross + map.shape[1] * (node - 1)
end

"""
    TransversePreparation(execution, map, policy, tables;
                          n_frequency, rows, lines, declared, arithmetic=map.arithmetic)

Copy the translated inputs for scalar radial, Cartesian or line proxy monitoring.

`map` is an already owned read-only `RadialObservationMap` or
`CartesianObservationMap`; its coarse transform has no physical flux or temporal
weights. `policy` and `tables` are the existing transverse records. `n_frequency`
is positive, `rows` is a Boolean node mask containing every face/rate row an ordinary
read uses, and `lines` is a concrete tuple of `TransverseLineData`. A collar has one
line per face; without a collar the tuple is empty. `declared` preserves the existing
above-discard declaration. Tables, masks and line arrays are copied; the map is shared
read-only. `arithmetic` must match the map. Node reconstruction uses its field scale,
while canonical-column energies use its state scale. Binary32 strip/current work
normalizes actual weighted sample components before products or FFTs; its derived
integer exponent is restored only in physical binary64 readings. Reject unsupported
execution, inconsistent geometry or invalid inputs.
"""
struct TransversePreparation{M, T, L, A}
    map::M
    policy::TransversePolicy
    tables::T
    n_frequency::Int
    n_columns::Int
    rows::Vector{Bool}
    read_rows::Vector{Bool}
    current_rows::Vector{Bool}
    lines::L
    declared::Bool
    arithmetic::A
end

function TransversePreparation(
        execution::KernelExecution,
        map::Union{RadialObservationMap, CartesianObservationMap},
        policy::TransversePolicy, tables::TransverseTables;
        n_frequency::Int, rows::AbstractVector{Bool},
        lines::Tuple{Vararg{TransverseLineData}}, declared::Bool,
        arithmetic::SourceArithmetic = map.arithmetic,
    )
    oncpu(execution) && n_frequency > 0 ||
        throw(ArgumentError("transverse observations require scalar CPU data"))
    arithmetic == map.arithmetic ||
        throw(ArgumentError("transverse observations and map need one arithmetic policy"))
    n_nodes, n_columns = _spatial_map_shape(map)
    length(rows) == n_nodes && size(tables.faces, 1) == n_nodes &&
        length(tables.column_weights) == n_columns ||
        throw(DimensionMismatch("transverse tables and map dimensions disagree"))
    all(line -> line isa TransverseLineData, lines) &&
        (policy.has_collar ? length(lines) == length(tables.face_ranks) : isempty(lines)) ||
        throw(ArgumentError("transverse collar lines must match the declared faces"))
    for node in 1:n_nodes
        (any(>(0), view(tables.faces, node, :)) || tables.totals[node, 2] > 0) &&
            !rows[node] &&
            throw(ArgumentError("ordinary transverse rows omit a face or collar rate"))
    end
    owned_lines = Base.map(lines) do line
        n_axis, _ = _transverse_line_shape(map, line)
        length(line.sample_scale) == n_axis ||
            throw(DimensionMismatch("transverse line scales do not fit their axis"))
        TransverseLineData(
            line.axis, line.sample_scale; strip = line.strip, window = line.window,
            spacing = line.spacing, measure = line.measure, inward = line.inward,
            onset = line.onset, onset_step = line.onset_step,
        )
    end
    owned_rows = Vector{Bool}(rows)
    read_rows, current_rows = copy(owned_rows), fill(false, n_nodes)
    for line in owned_lines
        _, n_cross = _transverse_line_shape(map, line)
        for cross in 1:n_cross, node in line.strip
            read_rows[_transverse_line_node(map, line, node, cross)] = true
        end
        if line.onset !== nothing
            for cross in 1:n_cross, node in line.onset
                current_rows[_transverse_line_node(map, line, node, cross)] = true
            end
        end
    end
    return TransversePreparation(
        map, policy, _transverse_copy_tables(tables), n_frequency, n_columns,
        owned_rows, read_rows, current_rows, owned_lines, declared, arithmetic,
    )
end

"""
    transverse_clean_columns(execution, field, noise, active)

Return owned unweighted column energies for the shared noise-free ramp judgement.

Finite disjoint-or-shared read-only `field` and `noise` are ComplexF64 matrices with
shape `(n_frequency,n_columns)`. `field` is the original pre-noise launch and `active`
its Boolean frequency mask. Each result is the sum of `abs2(field)` on active bins
plus `abs2(noise)` on inactive bins, exactly the shared `_clean_launch_columns` rule
for a launch masked after adding an arbitrary input realization. This intentionally
retains inactive noise energy; it is not the norm of an unmasked pre-noise field.
Inputs are preserved, no noise is applied, and finite sums are required. Preparation
may allocate the returned vector. Reject unsupported layouts before writing it.
"""
function transverse_clean_columns(execution::KernelExecution, field, noise, active)
    oncpu(execution) && field isa StridedMatrix{ComplexF64} &&
        noise isa StridedMatrix{ComplexF64} && size(field) == size(noise) &&
        !isempty(field) && active isa AbstractVector{Bool} &&
        length(active) == size(field, 1) ||
        throw(DimensionMismatch("clean transverse columns need matching CPU launch arrays"))
    all(isfinite, field) && all(isfinite, noise) ||
        throw(ArgumentError("clean transverse launch inputs must be finite"))
    mask = Vector{Bool}(active)
    result = zeros(Float64, size(field, 2))
    each_column!(execution, length(result)) do column
        inside = outside = 0.0
        for row in axes(field, 1)
            if mask[row]
                inside += _monitor_pair(field[row, column])
            else
                outside += _monitor_pair(noise[row, column])
            end
        end
        result[column] = inside + outside
    end
    all(isfinite, result) || throw(ArgumentError("clean transverse column sums overflowed"))
    return result
end

"""
    transverse_fallback_clean_columns(execution, masked_field, original_noise)

Prepare the shared later-start ramp judgement when no fresh clean launch was retained.

The finite ComplexF64 matrices have equal `(n_frequency,n_columns)` shape. Return
owned unweighted column sums of `abs2(masked_field-original_noise)`, preserving both
inputs. This is the explicit shared driver fallback for a start which did not add a
fresh draw and has no usable restored edge totals. It is not an estimate of an
unknown pre-noise field: the original supplied realization is subtracted exactly as
the shared monitor requires, including its inactive bins. Never use this helper for
a fresh noisy launch; `transverse_clean_columns` preserves that launch's separate
active/inactive rule. No noise is applied or phased here. Preparation may allocate;
invalid shapes, nonfinite inputs or overflowing sums raise before returning.
"""
function transverse_fallback_clean_columns(execution::KernelExecution, field, noise)
    oncpu(execution) && field isa StridedMatrix{ComplexF64} &&
        noise isa StridedMatrix{ComplexF64} && size(field) == size(noise) &&
        !isempty(field) ||
        throw(DimensionMismatch("fallback transverse columns need matching CPU arrays"))
    Base.require_one_based_indexing(field, noise)
    all(isfinite, field) && all(isfinite, noise) ||
        throw(ArgumentError("fallback transverse inputs must be finite"))
    result = zeros(Float64, size(field, 2))
    each_column!(execution, length(result)) do column
        total = 0.0
        for row in axes(field, 1)
            total += _monitor_pair(field[row, column] - noise[row, column])
        end
        result[column] = total
    end
    all(isfinite, result) ||
        throw(ArgumentError("fallback transverse column sums overflowed"))
    return result
end

mutable struct _TransverseLineWork{A, P, S, C, R}
    data::TransverseLineData
    spectrum::A
    current::A
    plan::P
    strip::S
    onset::C
    reading::R
    n_traces::Int
    chunk::Int
    strip_exponent::Int
    current_exponent::Int
end

function _TransverseLineWork(execution, prepared, line, trace_chunk)
    _, n_cross = _transverse_line_shape(prepared.map, line)
    n_traces = Base.checked_mul(prepared.n_frequency, n_cross)
    chunk = min(trace_chunk, n_traces)
    T = source_type(prepared.arithmetic)
    spectrum = dzeros(execution, Complex{T}, 2length(line.strip), chunk)
    current = dzeros(execution, Complex{T}, line.onset === nothing ? 0 : 3, chunk)
    plan = isempty(line.strip) ? nothing : SpatialFFT(execution, spectrum, (1,))
    strip = isempty(line.strip) ? nothing : TransverseStripLine(
            spectrum, line.spacing, line.measure, line.inward...,
        )
    onset = line.onset === nothing ? nothing : TransverseCurrentLine(
            current, 1, 2, 3, line.onset_step, line.measure,
        )
    return _TransverseLineWork(
        line, spectrum, current, plan, strip, onset, TransverseReading(1, 1),
        n_traces, chunk, 0, 0,
    )
end

"""
    TransverseWorkspace(execution, prepared; trace_chunk, noise_spectrum=nothing,
                        clean_columns=nothing, continuation=nothing)

Own scalar spatial monitor reductions, bounded strip transforms and warning stages.

`prepared` is a `TransversePreparation`; positive `trace_chunk` bounds each line's
twice-padded FFT batch. All unused traces and padding cells are overwritten on reuse.
Optional binary64 `noise_spectrum` is the independently supplied original realization,
reduced once at preparation. A fresh ramp judgement with noise requires explicit
finite nonnegative `clean_columns` from the appropriate preparation helper; this
workspace never infers them. No clean vector is required without a ramp or with
usable positive restored totals, because neither path judges it. Only the reduced
noise data are retained. Binary32 arithmetic refuses noise. Optional
`TransverseContinuation` preserves the
original launch and totals while a later resumed launch refreshes samples unjudged.
This admitted scalar spatial attachment always reads the edge, as native sessions
do; an off request creates no workspace. The generic eager control API separately
retains its unwatched-backend capability path.

State matrices use `(n_frequency,n_columns)`. Read-only maps/tables and every owned
numeric buffer are reachable through `data`/`buffers`; no borrowed field is retained.
Band, strip/current and node scratch follow `source_type(prepared.arithmetic)`;
control readings/continuation remain binary64. Each line retains its most recently
used strip/current exponent as small integer metadata.
Fresh invalid inputs raise before attachment. Warmed lifecycle calls allocate nothing
and produce at most one existing `MonitorNotice` before their explicit continuation.
"""
mutable struct TransverseWorkspace{E, D, B, F, L, S, R, N, C, V, U}
    execution::E
    data::D
    band::B
    spatial::F
    lines::L
    state::S
    reading::R
    noise::N
    clean_columns::C
    nodes::V
    buffers::U
    started::Bool
    committed_z::Float64
    fixed_step_max::Union{Nothing, Float64}
end

function _transverse_spatial_work(
        execution, prepared::TransversePreparation{<:RadialObservationMap},
    )
    return nothing
end
function _transverse_spatial_work(
        execution, prepared::TransversePreparation{<:CartesianObservationMap},
    )
    return CartesianBandWork(
        execution, prepared.map.shape, prepared.n_frequency;
        arithmetic = prepared.arithmetic,
    )
end

function TransverseWorkspace(
        execution::KernelExecution, prepared::TransversePreparation;
        trace_chunk::Int, noise_spectrum = nothing, clean_columns = nothing,
        continuation = nothing,
    )
    oncpu(execution) && trace_chunk > 0 ||
        throw(ArgumentError("transverse workspace requires CPU and a positive trace chunk"))
    T = source_type(prepared.arithmetic)
    T === Float32 && noise_spectrum !== nothing &&
        throw(ArgumentError("binary32 transverse observations do not support input noise"))
    state = TransverseState(prepared.tables)
    restore_transverse!(state, continuation)
    shape = (prepared.n_frequency, prepared.n_columns)
    if noise_spectrum !== nothing
        noise_spectrum isa StridedMatrix{ComplexF64} && size(noise_spectrum) == shape &&
            all(isfinite, noise_spectrum) ||
            throw(ArgumentError("transverse noise requires a finite matching spectrum"))
        (!prepared.policy.has_design || state.resumed || clean_columns !== nothing) ||
            throw(ArgumentError("transverse input noise requires explicit clean columns"))
    end
    if clean_columns !== nothing
        clean_columns isa AbstractVector{Float64} &&
            length(clean_columns) == prepared.n_columns &&
            all(value -> isfinite(value) && value >= 0, clean_columns) ||
            throw(ArgumentError("transverse clean columns must be finite nonnegative"))
    end
    spatial = _transverse_spatial_work(execution, prepared)
    n_nodes = size(prepared.tables.faces, 1)
    band = spatial === nothing ?
        dzeros(execution, Complex{T}, prepared.n_frequency, n_nodes) : spatial.band
    lines = map(
        line -> _TransverseLineWork(execution, prepared, line, trace_chunk), prepared.lines,
    )
    reading = TransverseReading(prepared.n_columns, length(lines))
    noise = noise_spectrum === nothing ? nothing :
        TransverseReading(prepared.n_columns, length(lines))
    clean = clean_columns === nothing ? nothing : Vector{Float64}(clean_columns)
    nodes = dzeros(execution, T, n_nodes)
    buffers = (
        band, nodes, state.last_columns, state.noise_columns,
        reading.columns, reading.strip, clean, noise, lines,
    )
    work = TransverseWorkspace(
        execution, prepared, band, spatial, lines, state, reading, noise, clean,
        nodes, buffers, false, NaN, nothing,
    )
    if noise !== nothing && noise_spectrum !== nothing
        _transverse_form_reading!(work, noise_spectrum; faces = true, full = true)
        _transverse_copy_reading!(noise, reading)
        reading.has_faces = reading.full = reading.has_strip = reading.has_current = false
    end
    return work
end

function _transverse_copy_reading!(destination, source)
    destination.faces, destination.clip, destination.totals =
        source.faces, source.clip, source.totals
    copyto!(destination.columns, source.columns)
    copyto!(destination.strip, source.strip)
    destination.current = source.current
    destination.has_faces, destination.full = source.has_faces, source.full
    destination.has_strip, destination.has_current = source.has_strip, source.has_current
    return nothing
end

function _transverse_field(work, field)
    T = source_type(work.data.arithmetic)
    field isa StridedMatrix{Complex{T}} &&
        size(field) == (work.data.n_frequency, work.data.n_columns) ||
        throw(DimensionMismatch("transverse field requires its prepared spectral layout"))
    all(isfinite, field) || throw(ArgumentError("transverse field must be finite"))
    _retained_alias(field, work) &&
        throw(ArgumentError("transverse field aliases retained preparation or work"))
    return nothing
end

function _transverse_band!(
        work::TransverseWorkspace{E, <:TransversePreparation{<:RadialObservationMap}},
        field, faces, full, current,
    ) where {E}
    band, synthesis = work.band, work.data.map.synthesis
    read_rows, current_rows = work.data.read_rows, work.data.current_rows
    each!(work.execution, size(band)...) do row, node
        value = zero(eltype(band))
        if full || (faces && read_rows[node]) || (current && current_rows[node])
            for column in axes(field, 2)
                value += synthesis[node, column] * field[row, column]
            end
        end
        band[row, node] = value
    end
    return nothing
end

function _transverse_band!(
        work::TransverseWorkspace{E, <:TransversePreparation{<:CartesianObservationMap}},
        field, faces, full, current,
    ) where {E}
    cartesian_inverse!(work.spatial, field)
    if source_type(work.data.arithmetic) === Float32
        band, scale = work.band, work.data.map.scale
        each!(work.execution, size(band)...) do row, column
            band[row, column] *= scale
        end
    end
    return nothing
end

function _transverse_strip_chunk!(work, line, first_trace, count)
    if source_type(work.data.arithmetic) === Float32
        _transverse_strip_single!(work, line, first_trace, count)
        return nothing
    end
    spectrum, data, band = line.spectrum, line.data, work.band
    strip, window, scale = data.strip, data.window, data.sample_scale
    n_frequency = work.data.n_frequency
    radial = work.data.map isa RadialObservationMap
    nx = radial ? 0 : work.data.map.shape[1]
    along_x = data.axis == :x
    each!(work.execution, size(spectrum)...) do row, column
        value = zero(ComplexF64)
        if row <= length(strip) && column <= count
            trace = first_trace + column - 1
            frequency = mod(trace - 1, n_frequency) + 1
            cross = div(trace - 1, n_frequency) + 1
            axis_node = strip[row]
            node = radial ? axis_node :
                along_x ? axis_node + nx * (cross - 1) : cross + nx * (axis_node - 1)
            value = window[row] * (scale[axis_node] * band[frequency, node])
        end
        spectrum[row, column] = value
    end
    spatial_forward!(spectrum, line.plan)
    transverse_strip!(line.reading, (line.strip,), work.execution)
    return nothing
end

function _transverse_current_chunk!(work, line, first_trace, count)
    if source_type(work.data.arithmetic) === Float32
        _transverse_current_single!(work, line, first_trace, count)
        return nothing
    end
    samples, data, band = line.current, line.data, work.band
    onset, scale = something(data.onset, (0, 0, 0)), data.sample_scale
    n_frequency = work.data.n_frequency
    radial = work.data.map isa RadialObservationMap
    nx = radial ? 0 : work.data.map.shape[1]
    along_x = data.axis == :x
    each!(work.execution, size(samples)...) do row, column
        value = zero(ComplexF64)
        if column <= count
            trace = first_trace + column - 1
            frequency = mod(trace - 1, n_frequency) + 1
            cross = div(trace - 1, n_frequency) + 1
            axis_node = onset[row]
            node = radial ? axis_node :
                along_x ? axis_node + nx * (cross - 1) : cross + nx * (axis_node - 1)
            value = scale[axis_node] * band[frequency, node]
        end
        samples[row, column] = value
    end
    transverse_current!(line.reading, (line.onset,), work.data.policy.k_min)
    return nothing
end

function _transverse_trace_parameters(work, line)
    radial = work.data.map isa RadialObservationMap
    return (
        band = work.band, frequencies = work.data.n_frequency, radial = radial,
        nx = radial ? 0 : work.data.map.shape[1], along_x = line.axis == :x,
    )
end

@inline function _transverse_trace_sample(inputs, axis_node, trace)
    frequency = mod(trace - 1, inputs.frequencies) + 1
    cross = div(trace - 1, inputs.frequencies) + 1
    node = inputs.radial ? axis_node : inputs.along_x ?
        axis_node + inputs.nx * (cross - 1) : cross + inputs.nx * (axis_node - 1)
    return inputs.band[frequency, node]
end

# Derive each chunk exponent from the actual weighted components, then form every
# product in binary32. The common FFT input scale is metadata, never a physical power.
# A large scale paired only with zero samples cannot erase another trace's signal.
function _transverse_strip_single!(work, line, first_trace, count)
    spectrum, data = line.spectrum, line.data
    inputs = _transverse_trace_parameters(work, data)
    strip, window, sample_scale = data.strip, data.window, data.sample_scale
    scale = typemin(Int)
    for column in 1:count, row in eachindex(strip)
        node = strip[row]
        value = _transverse_trace_sample(inputs, node, first_trace + column - 1)
        factors = (window[row], sample_scale[node])
        scale = max(
            scale, _observation_product_exponent((real(value), factors...)),
            _observation_product_exponent((imag(value), factors...)),
        )
    end
    product_scale = _observation_product_scale(scale)
    amplitude = work.data.arithmetic.field_exponent + product_scale
    _observation_derived_exponents((amplitude,))
    each!(work.execution, size(spectrum)...) do row, column
        value = zero(eltype(spectrum))
        if column <= count && row <= length(strip)
            node = strip[row]
            sample = _transverse_trace_sample(inputs, node, first_trace + column - 1)
            factors = (window[row], sample_scale[node])
            value = complex(
                _observation_product((real(sample), factors...), product_scale),
                _observation_product((imag(sample), factors...), product_scale),
            )
        end
        spectrum[row, column] = value
    end
    spatial_forward!(spectrum, line.plan)
    transverse_strip!(
        line.reading, (line.strip,), work.execution;
        amplitude_exponent = amplitude
    )
    line.strip_exponent = amplitude
    return nothing
end

function _transverse_current_single!(work, line, first_trace, count)
    samples, data = line.current, line.data
    inputs = _transverse_trace_parameters(work, data)
    sample_scale = data.sample_scale
    onset = something(data.onset, (0, 0, 0))
    scale = typemin(Int)
    for column in 1:count, node in onset
        value = _transverse_trace_sample(inputs, node, first_trace + column - 1)
        scale = max(
            scale, _observation_product_exponent((real(value), sample_scale[node])),
            _observation_product_exponent((imag(value), sample_scale[node])),
        )
    end
    product_scale = _observation_product_scale(scale)
    amplitude = work.data.arithmetic.field_exponent + product_scale
    _observation_derived_exponents((amplitude,))
    each!(work.execution, size(samples)...) do row, column
        value = zero(eltype(samples))
        if column <= count
            node = onset[row]
            sample = _transverse_trace_sample(inputs, node, first_trace + column - 1)
            value = complex(
                _observation_product((real(sample), sample_scale[node]), product_scale),
                _observation_product((imag(sample), sample_scale[node]), product_scale),
            )
        end
        samples[row, column] = value
    end
    transverse_current!(
        line.reading, (line.onset,), work.data.policy.k_min;
        amplitude_exponent = amplitude
    )
    line.current_exponent = amplitude
    return nothing
end

_transverse_line_readings!(work, ::Tuple{}, index, faces, current) = nothing
function _transverse_line_readings!(work, lines::Tuple, index, faces, current)
    line = first(lines)
    inward = windowed = flux = 0.0
    for first_trace in 1:line.chunk:line.n_traces
        count = min(line.chunk, line.n_traces - first_trace + 1)
        if faces && line.plan !== nothing
            _transverse_strip_chunk!(work, line, first_trace, count)
            inward += line.reading.strip[1, 1]
            windowed += line.reading.strip[1, 2]
        end
        if current && line.onset !== nothing
            _transverse_current_chunk!(work, line, first_trace, count)
            flux += line.reading.current
        end
    end
    if faces
        work.reading.strip[index, 1], work.reading.strip[index, 2] = inward, windowed
    end
    current && (work.reading.current += flux)
    _transverse_line_readings!(work, Base.tail(lines), index + 1, faces, current)
    return nothing
end

function _transverse_form_reading!(
        work, field; faces::Bool, full::Bool = false,
        current::Bool = false
    )
    _transverse_field(work, field)
    _transverse_band!(work, field, faces, full, current)
    reading = work.reading
    reading.has_faces = reading.full = reading.has_strip = reading.has_current = false
    reading.current = 0.0
    if faces
        transverse_reading!(
            reading, work.band, field, work.data.tables, work.nodes, work.execution; full,
            node_exponent = work.data.arithmetic.field_exponent,
            column_exponent = work.data.arithmetic.state_exponent,
        )
    end
    _transverse_line_readings!(work, work.lines, 1, faces, current)
    all(isfinite, reading.strip) && isfinite(reading.current) ||
        throw(ArgumentError("transverse strip or current accumulation overflowed"))
    reading.has_strip, reading.has_current = faces && !isempty(work.lines), current
    return nothing
end

function _transverse_monitor_entry(work, notices)
    notices.count == 0 && length(notices.records) >= 1 ||
        throw(ArgumentError("acknowledge transverse notices before further monitoring"))
    work.state.stage == 0 ||
        throw(ArgumentError("finish the transverse warning continuation first"))
    return nothing
end

"""
    transverse_monitor_launch!(work, field, z, notices; fixed_step_max=nothing)

Read one launch and pause after its first warning or stop.

`field` has the workspace's scalar spectral layout and is preserved; finite `z` is in
m. `notices` is an empty fixed buffer. `fixed_step_max`, when supplied, is the positive
actual longest itinerary step in m; exceeding read spacing warns but never shortens
the itinerary. A resumed launch refreshes columns/current without judging again.
Call `transverse_monitor_proceed!` after each warning acknowledgement until no notice
is emitted. New launch attempts refuse; no field is retained across the barrier.
"""
function transverse_monitor_launch!(
        work::TransverseWorkspace, field, z::Float64, notices;
        fixed_step_max::Union{Nothing, Float64} = nothing,
    )
    _transverse_monitor_entry(work, notices)
    !work.started && isfinite(z) ||
        throw(ArgumentError("transverse monitoring requires one finite launch"))
    fixed_step_max === nothing || (isfinite(fixed_step_max) && fixed_step_max > 0) ||
        throw(ArgumentError("fixed transverse maximum step must be finite positive"))
    _transverse_form_reading!(
        work, field; faces = true, full = !work.state.resumed,
        current = work.data.policy.has_collar && work.state.resumed,
    )
    work.started, work.committed_z, work.fixed_step_max = true, z, fixed_step_max
    _transverse_begin_launch!(
        work.state, work.data.policy, work.data.tables, work.reading, z, notices;
        noise = work.noise, clean_columns = work.clean_columns,
        fixed_step_max, declared = work.data.declared,
    )
    return nothing
end

"""
    transverse_monitor_endpoint!(work, field, z, next_step, notices)

Accumulate one accepted endpoint's current and read before a due next step.

Finite `z` must strictly follow the last committed endpoint; `next_step` is the actual
nonnegative replay step or adaptive proposal in m, zero after an itinerary ends.
Current is accumulated exactly once even if a warning pauses the subsequent read.
Read when `transverse_due && transverse_wants_read`. Unread endpoints still advance
the committed cursor. A promoted warning leaves later judgement stages unperformed.
"""
function transverse_monitor_endpoint!(
        work::TransverseWorkspace, field, z::Float64, next_step::Float64, notices,
    )
    _transverse_monitor_entry(work, notices)
    work.started && isfinite(z) && z > work.committed_z &&
        isfinite(next_step) && next_step >= 0 ||
        throw(ArgumentError("transverse endpoint must advance with a finite next step"))
    read = transverse_due(work.state, work.data.policy, z, next_step) &&
        transverse_wants_read(work.state, z)
    current = work.data.policy.has_collar
    if read || current
        _transverse_form_reading!(work, field; faces = read, current)
    else
        _transverse_field(work, field)
    end
    work.committed_z = z
    current && transverse_step!(work.state, work.data.policy, work.reading.current, z)
    read && _transverse_begin_read!(
        work.state, work.data.policy, work.data.tables, work.reading, z, notices,
    )
    return nothing
end

"""
    transverse_monitor_plane!(work, field, z, notices)

Read an unmeasured observed plane without moving the committed or current cursor.

The finite coordinate is at or before the last committed endpoint. Older dense planes
already passed by a transverse read add no observation, matching the shared/native
rule. A new plane read may pause at a warning; no periodic-row policy is chosen here.
"""
function transverse_monitor_plane!(work::TransverseWorkspace, field, z::Float64, notices)
    _transverse_monitor_entry(work, notices)
    work.started && isfinite(z) && z <= work.committed_z ||
        throw(ArgumentError("transverse plane must belong to the committed interval"))
    if transverse_wants_read(work.state, z)
        _transverse_form_reading!(work, field; faces = true)
        _transverse_begin_read!(
            work.state, work.data.policy, work.data.tables, work.reading, z, notices,
        )
    else
        _transverse_field(work, field)
    end
    return nothing
end

"""
    transverse_monitor_proceed!(work, notices)

Perform retained warning remainder up to its next notice without rereading a field.

The caller has acknowledged the preceding warning and emptied `notices`. No pending
stage is an exact no-op; a stopped observation has no remainder. The held numerical
reading remains owned until this cursor drains. Return `nothing`.
"""
function transverse_monitor_proceed!(work::TransverseWorkspace, notices)
    work.started && notices.count == 0 && !isempty(notices.records) ||
        throw(ArgumentError("transverse continuation requires an acknowledged launch"))
    _transverse_proceed!(
        work.state, work.data.policy, work.data.tables, work.reading, notices;
        noise = work.noise, fixed_step_max = work.fixed_step_max,
    )
    return nothing
end

"""Return the current transverse adaptive-step ceiling in m, Inf when inactive."""
transverse_monitor_ceiling(work::TransverseWorkspace) = transverse_ceiling(work.state)

"""
    TransverseMonitorRecord

Retain detached numerical transverse summary and continuation metadata.

`continuation` preserves the existing public-record values. `reason` is zero, unread
edge or insensitive noise (0/1/2), `removed_signed` is the already computed ramp share,
and `resumed` identifies a continued launch. `finish` is the fixed status/largest/share/
read-count closing verdict. This record contains no pointers, arrays or callbacks;
face names and text remain boundary-owned. Inspection never changes the monitor.
"""
struct TransverseMonitorRecord{C, F}
    continuation::C
    reason::UInt8
    removed_signed::Float64
    resumed::Bool
    finish::F
end

"""Return the fixed transverse closing verdict without changing numerical state."""
transverse_monitor_finish(work::TransverseWorkspace) = transverse_finish!(work.state)

"""Return detached transverse continuation/summary facts, including partial launches."""
function transverse_monitor_record(work::TransverseWorkspace)
    state = work.state
    return TransverseMonitorRecord(
        transverse_continuation(state), state.reason, transverse_ramp_removed(state),
        state.resumed, transverse_monitor_finish(work),
    )
end
