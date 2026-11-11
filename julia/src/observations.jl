# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public ObservationProfiles, ObservationPopulations
public TemporalObservationData, ObservationPreparation, ObservationWorkspace
public observe!, observation_launch!, capture_split!, observation_accepted!
public observation_check!, observation_plane!, observation_finish!

abstract type _ObservationPreparation end
abstract type _ObservationWorkspace end

"""
    ObservationProfiles(; density=nothing, pressure=nothing, core_radius=nothing)

Retain independent represented scalar profiles for the existing statistics catalogue.

Each present value is an immutable `SourceProfile`. Their units are m⁻³ for density,
bar for the catalogue's pressure and m for core radius; pressure conversion belongs
to translation. A missing profile is allowed only when its statistic is not selected.
This record evaluates no material model and never substitutes one profile for another.
"""
struct ObservationProfiles{D, P, R}
    density::D
    pressure::P
    core_radius::R
end

function ObservationProfiles(; density = nothing, pressure = nothing, core_radius = nothing)
    all(
        value -> value === nothing || value isa SourceProfile,
        (density, pressure, core_radius)
    ) ||
        throw(ArgumentError("observation profiles require represented scalar data"))
    return ObservationProfiles(density, pressure, core_radius)
end

"""
    ObservationPopulations(plasma, grid, map; species_densities, channel_species,
                           peak_species)

Retain immutable scalar modal plasma data for fresh population observations.

`plasma`, `grid` and `map` are `ModalPlasma`, `ModalSourceGrid` and `ModalSourceMap`.
`plasma=nothing` with no channel indices represents a wholly non-ionising mixture.
They carry the qualified channel/rate data, fine temporal grid and physical synthesis,
including its existing padding scale. `species_densities` is a complete mixture-order
tuple of independent number-density `SourceProfile` values in m⁻³. `channel_species`
maps each plasma channel to a unique one-based species index. Non-ionising species
retain zero populations. `peak_species` is zero for the mixture peak, or a one-based
species index for the translated legacy single-population node-selection convention.
Scans use only each channel's rate and preionisation with its integration/cycle rule;
current-only density, charge and ionisation potential never enter population algebra.
No mutable source instance or stage scratch is retained. Invalid mappings or field/
envelope channel combinations raise `ArgumentError` or `DimensionMismatch`.
"""
struct ObservationPopulations{P, G, M, D, I}
    plasma::P
    grid::G
    map::M
    species_densities::D
    channel_species::I
    peak_species::Int
end

function ObservationPopulations(
        plasma::Union{Nothing, ModalPlasma}, grid::ModalSourceGrid, map::ModalSourceMap;
        species_densities::Tuple, channel_species::Tuple, peak_species::Int,
    )
    !isempty(species_densities) &&
        all(profile -> profile isa SourceProfile, species_densities) ||
        throw(ArgumentError("population observations need complete species profiles"))
    channels = plasma === nothing ? () : plasma.channels
    length(channel_species) == length(channels) ||
        throw(DimensionMismatch("population channel and species mappings disagree"))
    all(
        index -> index isa Int && 1 <= index <= length(species_densities),
        channel_species
    ) && length(unique(channel_species)) == length(channel_species) &&
        0 <= peak_species <= length(species_densities) ||
        throw(ArgumentError("population species mapping or peak selection is invalid"))
    field = grid.kind == :field
    (
        plasma === nothing ||
            (field ? plasma.rule isa RunningIntegralRule : plasma.rule === nothing)
    ) &&
        all(channel -> (channel.quadrature === nothing) == field, channels) ||
        throw(ArgumentError("population channels disagree with the temporal grid kind"))
    if field && plasma !== nothing
        plasma.rule.dt == grid.dt && length(plasma.rule.taps) <= grid.n_time ||
            throw(DimensionMismatch("population integral disagrees with its fine grid"))
    end
    return ObservationPopulations(
        plasma, grid, map, species_densities, channel_species, peak_species
    )
end

"""
    TemporalObservationData(policy, weights)

Copy a coarse-grid temporal monitor's explicit policy and numerical weights.

The arguments are `TemporalMonitorPolicy` and `TemporalMonitorWeights`. Time-region
weights, modal column quadrature, bucket indices and crossing limits retain their
existing units and conventions. No default boundary policy or geometry is inferred.
"""
struct TemporalObservationData{P, W}
    policy::P
    weights::W
end

function TemporalObservationData(
        policy::TemporalMonitorPolicy, weights::TemporalMonitorWeights,
    )
    eltype(weights.regions) === Float64 && eltype(weights.columns) === Float64 ||
        throw(ArgumentError("modal temporal observations require binary64 weights"))
    owned_policy = TemporalMonitorPolicy(;
        warn = policy.warn, stop = policy.stop, activity = policy.activity,
        quiet_share = policy.quiet_share, quiet_checks = policy.quiet_checks,
        quiet_collar = policy.quiet_collar, cadence = policy.cadence,
        interval = policy.interval, launch_floor_ratio = policy.launch_floor_ratio,
        floor_margin = policy.floor_margin, split = policy.split,
        crossing_share = policy.crossing_share, bucket_limits = copy(policy.bucket_limits),
    )
    owned_weights = TemporalMonitorWeights(
        copy(weights.regions), copy(weights.columns);
        buckets = weights.buckets === nothing ? nothing : copy(weights.buckets),
    )
    return TemporalObservationData{typeof(owned_policy), typeof(owned_weights)}(
        owned_policy, owned_weights
    )
end

"""
    ObservationPreparation(execution, grid, selection; kind, n_modes, power_weight,
        profiles=ObservationProfiles(), axis_synthesis=nothing,
        taper=SourceProfile(1.0), intensity_factor=nothing,
        populations=nothing, temporal=nothing)

Own numerical preparation for scalar modal statistics and temporal monitoring.

`grid` is a coarse `StatisticsGrid`, and `selection` fixes packed plane/periodic slots.
`kind` is `:field` or `:envelope`; `n_modes` is positive. `power_weight` is a prepared
`ObservationWeight` for the `(n_omega,n_modes)` state. Optional physical on-axis
`axis_synthesis` has one or two Cartesian rows and `n_modes` columns, with no padding
factor. It is rescaled by its own positive `taper` profile at each observation. Axis
intensity requires the explicitly translated positive `intensity_factor` in SI units.
Optional population/monitor records are `ObservationPopulations` and
`TemporalObservationData`. No Python object, session cursor or mutable source is owned.

This initial composition requires CPU binary64 and refuses spatial statistics or
vector propagation. The lowerer retains the catalogue's route eligibility rules.
Missing required families and invalid numerical layouts raise before workspace creation.
"""
struct ObservationPreparation{G, S, W, P, A, T, L, M} <: _ObservationPreparation
    kind::Symbol
    n_modes::Int
    grid::G
    selection::S
    power_weight::W
    profiles::P
    axis_synthesis::A
    taper::T
    intensity_factor::Float64
    populations::L
    temporal::M
    needs::StatisticsNeeds
    periodic_needs::StatisticsNeeds
end

function ObservationPreparation(
        execution, grid::StatisticsGrid, selection::StatisticsSelection;
        kind::Symbol, n_modes::Int, power_weight::ObservationWeight,
        profiles::ObservationProfiles = ObservationProfiles(), axis_synthesis = nothing,
        taper::SourceProfile = SourceProfile(1.0), intensity_factor = nothing,
        populations = nothing, temporal = nothing,
    )
    oncpu(execution) || throw(ArgumentError("modal observations currently require CPU"))
    n_time, n_frequency = length(grid.time), length(grid.omega)
    all(
        array -> eltype(array) === Float64,
        (grid.time, grid.omega, grid.parseval, grid.spectral_density)
    ) ||
        throw(ArgumentError("modal observation grids require binary64 arrays"))
    kind in (:field, :envelope) && n_time >= 4 && iseven(n_time) && n_modes > 0 ||
        throw(ArgumentError("invalid scalar modal observation dimensions or grid kind"))
    n_frequency == (kind == :field ? n_time ÷ 2 + 1 : n_time) &&
        power_weight.shape == (n_frequency, n_modes) ||
        throw(DimensionMismatch("coarse observation spectrum and time grid disagree"))
    needs = statistics_needs(selection)
    spatial = needs.widths || needs.axis_spectrum ||
        any(slot -> slot.code == 19, selection.slots)
    spatial &&
        throw(ArgumentError("modal observations refuse spatial-only statistics"))
    needs.populations && !(populations isa ObservationPopulations) &&
        throw(ArgumentError("selected population statistics need prepared populations"))
    if populations !== nothing
        populations isa ObservationPopulations && populations.grid.kind == kind &&
            length(populations.grid.omega) == n_frequency &&
            size(populations.map.synthesis, 2) == n_modes ||
            throw(DimensionMismatch("population preparation disagrees with modal state"))
    end
    for slot in selection.slots
        expected = if slot.code in (1, 2, 4)
            n_modes
        elseif slot.code in (3, 5)
            n_modes * size(grid.windows, 2)
        elseif slot.code in (14, 15)
            if populations === nothing
                message = "selected population statistics need prepared populations"
                throw(ArgumentError(message))
            end
            species = length(populations.species_densities)
            slot.code == 14 ? size(populations.map.synthesis, 1) * species : species
        else
            1
        end
        slot.count == expected ||
            throw(DimensionMismatch("observation slot disagrees with its numerical shape"))
    end
    for (code, profile) in (
            (11, profiles.density), (12, profiles.pressure),
            (13, profiles.core_radius),
        )
        any(slot -> slot.code == code, selection.slots) && profile === nothing &&
            throw(ArgumentError("a requested scalar profile is missing"))
    end
    if axis_synthesis !== nothing
        axis_synthesis isa AbstractMatrix{Float64} &&
            1 <= size(axis_synthesis, 1) <= 2 && size(axis_synthesis, 2) == n_modes &&
            all(isfinite, axis_synthesis) ||
            throw(ArgumentError("coarse on-axis synthesis requires finite modal rows"))
    end
    needs.intensity && axis_synthesis === nothing &&
        throw(ArgumentError("peak_intensity needs a separate coarse on-axis map"))
    if intensity_factor !== nothing
        intensity_factor isa Float64 &&
            isfinite(intensity_factor) && intensity_factor > 0 ||
            throw(ArgumentError("intensity conversion must be positive binary64"))
    elseif needs.intensity
        throw(ArgumentError("peak_intensity requires its translated SI conversion"))
    end
    if temporal !== nothing
        temporal isa TemporalObservationData &&
            size(temporal.weights.regions, 1) == n_time &&
            length(temporal.weights.columns) == n_modes &&
            (
            temporal.weights.buckets === nothing ||
                size(temporal.weights.buckets) == (n_frequency, n_modes)
        ) ||
            throw(DimensionMismatch("temporal monitor and modal grid disagree"))
        buckets, limits = temporal.weights.buckets, temporal.policy.bucket_limits
        (
            (buckets === nothing && isempty(limits)) ||
                (
                buckets !== nothing && !isempty(limits) &&
                    maximum(buckets) <= length(limits)
            )
        ) ||
            throw(DimensionMismatch("temporal crossing buckets and limits disagree"))
    end
    owned_grid = StatisticsGrid(
        copy(grid.omega), copy(grid.time), copy(grid.parseval);
        windows = copy(grid.windows), spectral_density = copy(grid.spectral_density),
        d_omega = grid.d_omega,
    )
    return ObservationPreparation(
        kind, n_modes, owned_grid, selection, power_weight, profiles,
        axis_synthesis === nothing ? nothing : upload(execution, axis_synthesis), taper,
        intensity_factor === nothing ? 1.0 : intensity_factor, populations, temporal,
        needs, statistics_needs(selection; periodic = true),
    )
end

struct _ObservationPair{A, F}
    spectrum::A
    weighted_spectrum::A
    samples::A
    weighted::A
    transform::F
    weighted_transform::F
end

function _observation_pair(execution, n_time, n_columns)
    spectrum, weighted_spectrum = zeros(ComplexF64, n_time, n_columns),
        zeros(ComplexF64, n_time, n_columns)
    samples, weighted = similar(spectrum), similar(spectrum)
    return _ObservationPair(
        spectrum, weighted_spectrum, samples, weighted,
        TemporalFFT(execution, samples, spectrum),
        TemporalFFT(execution, weighted, weighted_spectrum)
    )
end

struct _ObservationWindows{P, A}
    pair::P
    samples::A
    weighted::A
end

struct _ObservationPopulationWork{A, T, F, P, W, C}
    spectrum::A
    modal::T
    transform::F
    padding::P
    nodes::T
    workspace::W
    chunks::C
    densities::Vector{Float64}
    fractions::Matrix{Float64}
    peaks::Matrix{Float64}
end

function _observation_population_work(execution, data, n_modes, node_chunk)
    data === nothing && return nothing
    grid, maps = data.grid, data.map
    n_nodes, n_species = size(maps.synthesis, 1), length(data.species_densities)
    capacity = min(node_chunk, n_nodes)
    field = grid.kind == :field
    scalar = field ? Float64 : ComplexF64
    frequencies = field ? grid.n_time ÷ 2 + 1 : grid.n_time
    spectrum = zeros(ComplexF64, frequencies, n_modes)
    modal = zeros(scalar, grid.n_time, n_modes)
    nodes = zeros(scalar, grid.n_time, capacity)
    layout = ResponseLayout(grid.n_time, 1, capacity)
    workspace = data.plasma === nothing ? nothing :
        PopulationWorkspace(execution, Float64, layout)
    chunks = map(1:capacity:n_nodes) do first
        last = min(first + capacity - 1, n_nodes)
        (
            first = first, count = last - first + 1,
            synthesis = view(maps.synthesis, first:last, :),
        )
    end
    return _ObservationPopulationWork(
        spectrum, modal,
        TemporalFFT(execution, modal, spectrum),
        SpectralPadding(execution, grid.crop_index, frequencies, 1.0, 1.0),
        nodes, workspace, chunks,
        zeros(n_species), zeros(n_nodes, n_species), zeros(n_nodes, n_species)
    )
end

mutable struct _ObservationMonitorWork{A, T, F, R}
    spectrum::A
    samples::T
    transform::F
    power::Vector{Float64}
    reading::R
    pending::R
    floor::TemporalFloor
    captured::Bool
    capture_z::Float64
    capture_dz::Float64
end

function _observation_monitor_work(execution, data, noise)
    data.temporal === nothing && return nothing
    temporal = data.temporal
    n_time, n_modes = length(data.grid.time), data.n_modes
    spectrum = zeros(ComplexF64, length(data.grid.omega), n_modes)
    samples = zeros(data.kind == :field ? Float64 : ComplexF64, n_time, n_modes)
    buckets = temporal.weights.buckets
    n_buckets = length(temporal.policy.bucket_limits)
    reading, pending = TemporalReading(n_buckets), TemporalReading(n_buckets)
    work = _ObservationMonitorWork(
        spectrum, samples,
        TemporalFFT(execution, samples, spectrum), zeros(n_time), reading, pending,
        TemporalFloor(0.0, (0.0, 0.0, 0.0, 0.0)), false, NaN, 0.0
    )
    if noise !== nothing
        size(noise) == size(spectrum) && eltype(noise) === ComplexF64 &&
            all(isfinite, noise) ||
            throw(ArgumentError("invalid modal input-noise spectrum"))
        copyto!(spectrum, noise)
        temporal_inverse!(samples, work.transform, spectrum, execution)
        temporal_reading!(
            reading, samples, temporal.weights, work.power, execution;
            spectrum = buckets === nothing ? nothing : spectrum
        )
        work.floor = temporal_floor(reading, temporal.weights)
    end
    return work
end

_observation_arrays(::Nothing) = ()
_observation_arrays(pair::_ObservationPair) =
    (pair.spectrum, pair.weighted_spectrum, pair.samples, pair.weighted)
_observation_arrays(windows::_ObservationWindows) = _observation_arrays(windows.pair)
_observation_arrays(work::_ObservationMonitorWork) =
    (
    work.spectrum, work.samples, work.power,
    work.reading.occupancy, work.pending.occupancy,
)
_observation_arrays(work::PopulationWorkspace) = _population_arrays(work)
_observation_arrays(work::_ObservationPopulationWork) =
    (
    work.spectrum, work.modal, work.nodes, work.densities,
    work.fractions, work.peaks, _observation_arrays(work.workspace),
)

function _observation_no_alias(input, ::Tuple{})
    return nothing
end
function _observation_no_alias(input, arrays::Tuple)
    _observation_no_alias(input, first(arrays))
    _observation_no_alias(input, Base.tail(arrays))
    return nothing
end
function _observation_no_alias(input, array::AbstractArray)
    Base.mightalias(input, array) &&
        throw(ArgumentError("observation input aliases its mutable workspace"))
    return nothing
end

"""
    ObservationWorkspace(execution, prepared; node_chunk, noise_spectrum=nothing,
        temporal_continuation=nothing, continuation_plane=false)

Allocate owned numerical observation scratch without retaining a session or source.

`prepared` is an `ObservationPreparation`, and the positive `node_chunk` bounds fresh
population reconstruction. Optional binary64 complex `noise_spectrum` has the coarse
state shape and is reduced once to a numerical floor; its array is not retained.
An optional `TemporalContinuation` is restored before numerical launch initialization.

`values` is the packed statistics output. Every temporal transform, window pair,
population matrix and monitor reading is independently owned, with all arrays listed
by `buffers`. `monitor` is its optional `TemporalMonitorState`; the caller owns notices,
row capacity, export and acknowledgment. `valid` is set only after successful observe!.
Warm numerical evaluation does not allocate or increment propagation source calls.
"""
mutable struct ObservationWorkspace{E, D, P, W, A, T, L, M, C, I, B} <:
    _ObservationWorkspace
    execution::E
    data::D
    spectrum::Matrix{ComplexF64}
    weight::Matrix{Float64}
    weight_work::Vector{Float64}
    pair::P
    windows::W
    axis::A
    statistics::T
    populations::L
    monitor_work::M
    monitor::C
    inputs::I
    values::Vector{Float64}
    buffers::B
    valid::Bool
    z::Float64
end

function ObservationWorkspace(
        execution, prepared::ObservationPreparation;
        node_chunk::Int, noise_spectrum = nothing, temporal_continuation = nothing,
        continuation_plane::Bool = false
    )
    oncpu(execution) && node_chunk > 0 ||
        throw(ArgumentError("modal observations need CPU execution and a positive chunk"))
    n_time, n_modes = length(prepared.grid.time), prepared.n_modes
    needs = prepared.needs
    spectrum = zeros(ComplexF64, prepared.power_weight.shape)
    weight = ones(prepared.power_weight.shape)
    n_terms = prepared.power_weight.weights === nothing ? 0 :
        size(prepared.power_weight.weights.coefficients, 1)
    weight_work = zeros(n_terms)
    pair = needs.temporal || needs.intensity ?
        _observation_pair(execution, n_time, n_modes) : nothing
    n_windows = size(prepared.grid.windows, 2)
    windows = if needs.windows
        n_windows > 0 || throw(ArgumentError("windowed power needs spectral gates"))
        window_pair = _observation_pair(execution, n_time, n_modes * n_windows)
        _ObservationWindows(
            window_pair,
            reshape(view(window_pair.samples, :, :), n_time, n_modes, n_windows),
            reshape(view(window_pair.weighted, :, :), n_time, n_modes, n_windows)
        )
    else
        nothing
    end
    axis = if needs.intensity
        prepared.axis_synthesis === nothing &&
            throw(ArgumentError("peak_intensity needs a separate coarse on-axis map"))
        (
            zeros(ComplexF64, n_time, size(prepared.axis_synthesis, 1)),
            zeros(ComplexF64, n_time, size(prepared.axis_synthesis, 1)),
        )
    else
        nothing
    end
    populations = needs.populations ?
        _observation_population_work(execution, prepared.populations, n_modes, node_chunk) :
        nothing
    n_nodes = populations === nothing ? 0 : size(populations.fractions, 1)
    statistics = StatisticsWorkspace(
        execution, Float64;
        n_time, n_columns = n_modes, n_nodes
    )
    monitor_work = _observation_monitor_work(execution, prepared, noise_spectrum)
    monitor = monitor_work === nothing ? nothing : TemporalMonitorState()
    if temporal_continuation !== nothing
        monitor === nothing && throw(ArgumentError("monitor continuation needs a monitor"))
        restore_temporal!(monitor, temporal_continuation; plane = continuation_plane)
    end
    spectral_input = needs.spectral ? SpectralStatisticsInput(spectrum, weight) : nothing
    temporal_input = if needs.temporal || needs.windows || needs.intensity
        TemporalStatisticsInput(
            pair === nothing ? nothing : pair.samples,
            pair === nothing ? nothing : pair.weighted;
            window_analytic = windows === nothing ? nothing : windows.samples,
            window_weighted = windows === nothing ? nothing : windows.weighted,
            axis_analytic = axis === nothing ? nothing : axis[1],
            axis_weighted = axis === nothing ? nothing : axis[2],
            intensity_factor = prepared.intensity_factor,
        )
    else
        nothing
    end
    population_input = populations === nothing ? nothing : PopulationStatisticsInput(
            populations.densities, populations.fractions, populations.peaks;
            peak_species = prepared.populations.peak_species
        )
    inputs = StatisticsInputs(;
        spectral = spectral_input, temporal = temporal_input,
        populations = population_input
    )
    values = zeros(prepared.selection.width)
    buffers = (
        spectrum, weight, weight_work, values,
        _observation_arrays(pair), _observation_arrays(windows),
        axis === nothing ? () : axis, _observation_arrays(populations),
        _observation_arrays(monitor_work), statistics.power, statistics.total_power,
        statistics.column_energy, statistics.column_moment, statistics.node_total,
        statistics.fluence, statistics.magnitude, statistics.radial_average,
    )
    return ObservationWorkspace(
        execution, prepared, spectrum, weight, weight_work,
        pair, windows, axis, statistics, populations, monitor_work, monitor, inputs, values,
        buffers, false, NaN
    )
end


function _observation_field(work, field, z)
    field isa AbstractMatrix{ComplexF64} && size(field) == size(work.spectrum) ||
        throw(DimensionMismatch("observation field must match its binary64 modal state"))
    isfinite(z) && all(isfinite, field) ||
        throw(ArgumentError("observation position and field must be finite"))
    _observation_no_alias(field, work.buffers)
    return nothing
end

function _observation_analytic!(pair, field, weight, kind, windows, gated, execution)
    pair === nothing && throw(ArgumentError("analytic observation buffers are missing"))
    spectrum, weighted = pair.spectrum, pair.weighted_spectrum
    n_frequency, n_modes = size(field)
    real_field = kind == :field
    each!(execution, size(spectrum)...) do row, column
        mode = mod(column - 1, n_modes) + 1
        window = div(column - 1, n_modes) + 1
        value = if row > n_frequency || (gated && !windows[row, window])
            0.0im
        elseif real_field && (row == 1 || row == n_frequency)
            complex(real(field[row, mode]), 0.0)
        else
            (real_field ? 2.0 : 1.0) * field[row, mode]
        end
        spectrum[row, column] = value
        weighted[row, column] = row <= n_frequency ? value * weight[row, mode] : 0.0im
    end
    temporal_inverse!(pair.samples, pair.transform, spectrum, execution)
    temporal_inverse!(pair.weighted, pair.weighted_transform, weighted, execution)
    return nothing
end

function _observation_axis!(work, position)
    work.axis === nothing && throw(ArgumentError("axis observation buffers are missing"))
    data, execution = work.data, work.execution
    scale = _source_profile(data.taper, position)
    scale > 0 || throw(ArgumentError("observation taper must be positive"))
    for (destination, source) in (
            (work.axis[1], work.pair.samples),
            (work.axis[2], work.pair.weighted),
        )
        synthesize!(destination, source, data.axis_synthesis, execution; n_components = 1)
        _source_scale!(destination, scale, execution)
    end
    return nothing
end

function _observation_densities!(output, ::Tuple{}, position, offset)
    return nothing
end
function _observation_densities!(output, profiles::Tuple, position, offset)
    value = _source_profile(first(profiles), position)
    value >= 0 || throw(ArgumentError("population species density must be nonnegative"))
    output[offset] = value
    _observation_densities!(output, Base.tail(profiles), position, offset + 1)
    return nothing
end

function _observation_channel!(work, data, channel, index, chunk, position, execution)
    initial = _source_profile(channel.preionisation, position)
    0 <= initial <= 1 ||
        throw(ArgumentError("invalid observation plasma preionisation"))
    if data.plasma.rule !== nothing
        field_populations!(
            work.nodes, channel.channel.rate,
            data.plasma.rule, work.workspace, execution;
            preionisation = initial, count = chunk.count
        )
    else
        envelope_populations!(
            work.nodes, channel.channel.rate,
            channel.quadrature, work.workspace, execution;
            preionisation = initial, dt = data.grid.dt, count = chunk.count
        )
    end
    failure = plasma_failure(work.workspace.status)
    failure.code == PlasmaSuccess || throw(
        SourcePlasmaError(
            failure.code, index, chunk.first + failure.node - 1, failure.sample
        )
    )
    species = data.channel_species[index]
    fraction, rate = work.workspace.fraction, work.workspace.rate
    fractions, peaks, first = work.fractions, work.peaks, chunk.first
    each_column!(execution, chunk.count) do node
        peak = 0.0
        for row in axes(rate, 1)
            peak = max(peak, rate[row, node])
        end
        fractions[first + node - 1, species] = fraction[end, node]
        peaks[first + node - 1, species] = peak
    end
    return nothing
end

function _observation_populations!(work, data, field, position, execution)
    work === nothing && throw(ArgumentError("population observation buffers are missing"))
    _observation_densities!(work.densities, data.species_densities, position, 1)
    fill!(work.fractions, 0.0)
    fill!(work.peaks, 0.0)
    data.plasma === nothing && return nothing
    scale = _source_profile(data.map.taper, position)
    scale > 0 || throw(ArgumentError("population taper must be positive"))
    pad_spectrum!(work.spectrum, field, work.padding, execution)
    temporal_inverse!(work.modal, work.transform, work.spectrum, execution)
    for chunk in work.chunks
        synthesize!(
            work.nodes, work.modal, chunk.synthesis, execution;
            n_components = 1, active_nodes = chunk.count
        )
        _source_scale!(work.nodes, scale, execution)
        all(isfinite, work.nodes) ||
            throw(ArgumentError("population field reconstruction overflowed"))
        _source_each_indexed(data.plasma.channels) do channel, index
            _observation_channel!(work, data, channel, index, chunk, position, execution)
        end
    end
    return nothing
end

function _observation_selected(selection, code, periodic)
    for (index, slot) in pairs(selection.slots)
        slot.code == code && (!periodic || index in selection.periodic) && return true
    end
    return false
end

function _observation_profile(profile::SourceProfile, selected, position)
    return selected ? _source_profile(profile, position) : NaN
end
_observation_profile(::Nothing, selected, position) = selected ?
    throw(ArgumentError("selected observation profile is missing")) : NaN

_observation_identity(::Nothing, identity) = nothing
function _observation_identity(input::TemporalStatisticsInput, identity)
    return TemporalStatisticsInput(
        input.analytic, input.weighted,
        input.window_analytic, input.window_weighted, input.axis_analytic,
        input.axis_weighted, input.intensity_factor, identity
    )
end

"""
    observe!(work, field, z; periodic=false, quadrature_error=NaN)

Reconstruct fresh modal statistics at the actual supplied field and position.

`work` is an `ObservationWorkspace`. Binary64 complex `field` has coarse state shape
`(n_omega,n_modes)` and is preserved; `z` is finite in m. `periodic` selects exactly
the prepared periodic slots, leaving other packed slices zero. The optional alarm is
the session's accepted quadrature diagnostic, never inferred from stage scratch.
Write `work.values`, then mark `valid=true` and `z` only after successful evaluation;
return `nothing`. No source, monitor or delivery counter is advanced.

Throw for nonfinite/aliased fields, invalid represented profiles, missing selected
inputs or failed plasma quadrature. Population failure retains its typed channel/node
status. All requested transforms and populations are rebuilt from this field.
"""
function observe!(
        work::ObservationWorkspace, field, z;
        periodic::Bool = false, quadrature_error = NaN
    )
    _observation_field(work, field, z)
    work.valid = false
    data, execution = work.data, work.execution
    needs = periodic ? data.periodic_needs : data.needs
    weighted = needs.spectral || needs.temporal || needs.windows || needs.intensity
    if weighted
        copyto!(work.spectrum, field)
        evaluate_weight!(
            work.weight, data.power_weight, z, execution; work = work.weight_work,
        )
    end
    if needs.temporal || needs.intensity
        _observation_analytic!(
            work.pair, work.spectrum, work.weight, data.kind,
            data.grid.windows, false, execution
        )
    end
    if needs.windows
        windows = work.windows
        windows === nothing &&
            throw(ArgumentError("window observation buffers are missing"))
        _observation_analytic!(
            windows.pair, work.spectrum, work.weight, data.kind,
            data.grid.windows, true, execution
        )
    end
    needs.intensity && _observation_axis!(work, z)
    needs.populations &&
        _observation_populations!(work.populations, data.populations, field, z, execution)
    temporal = _observation_identity(work.inputs.temporal, all(isone, work.weight))
    profiles = (
        _observation_profile(
            data.profiles.density,
            _observation_selected(data.selection, 11, periodic), z
        ),
        _observation_profile(
            data.profiles.pressure,
            _observation_selected(data.selection, 12, periodic), z
        ),
        _observation_profile(
            data.profiles.core_radius,
            _observation_selected(data.selection, 13, periodic), z
        ),
    )
    inputs = StatisticsInputs(;
        spectral = work.inputs.spectral, temporal, populations = work.inputs.populations,
        profiles, quadrature_error
    )
    statistics!(
        work.values, data.selection, data.grid, inputs,
        work.statistics, execution; periodic
    )
    work.z, work.valid = z, true
    return nothing
end

function _observation_monitor_inverse!(work, field)
    monitor = work.monitor_work
    copyto!(monitor.spectrum, field)
    temporal_inverse!(monitor.samples, monitor.transform, monitor.spectrum, work.execution)
    return nothing
end

function _observation_read!(work, field)
    monitor, weights = work.monitor_work, work.data.temporal.weights
    _observation_monitor_inverse!(work, field)
    temporal_reading!(
        monitor.reading, monitor.samples, weights, monitor.power,
        work.execution; spectrum = weights.buckets === nothing ? nothing : monitor.spectrum
    )
    return nothing
end

function _observation_drain!(reading)
    reading.removed = 0.0
    return nothing
end

function _observation_decision(work)
    return (
        quiet = work.monitor === nothing ? false : work.monitor.quiet,
        step_cap = work.monitor === nothing ? Inf : temporal_step_cap(work.monitor),
    )
end

"""
    observation_launch!(work, field, z, notices)

Initialize the optional numerical temporal monitor from the actual launch field.

Inputs follow `observe!`; `notices` is caller-owned `MonitorNotices` with available
capacity. Return the optional `TemporalPlaneRow`, or `nothing` without a monitor.
Use the prepared noise floor and any restored continuation. This hook imposes no
event-delivery order; the session owns launch events and saved rows. Reject repeated
launch, an outstanding split capture or invalid input before numerical initialization.
"""
function observation_launch!(work::_ObservationWorkspace, field, z, notices)
    _observation_field(work, field, z)
    work.monitor === nothing && return nothing
    isnan(work.monitor.committed) && !work.monitor_work.captured ||
        throw(ArgumentError("observation monitor launch was already initialized"))
    _observation_read!(work, field)
    return temporal_launch!(
        work.monitor, work.data.temporal.policy,
        work.monitor_work.reading, z, work.monitor_work.floor, notices
    )
end

"""
    capture_split!(work, masked_pre_split_field, z, dz)

Capture one accepted temporal split without changing its masked input field.

The session first applies its active-band mask, then calls this hook, applies split
damping and the final mask, and commits the field. `z` is that accepted endpoint in m
and `dz` is its positive step in m. Save removed energy and half-split occupancies;
the committed field supplies crossing occupancy later. Return `nothing` when split
monitoring is absent. A second unconsumed capture or invalid step raises before reuse.
"""
function capture_split!(work::_ObservationWorkspace, field, z, dz)
    _observation_field(work, field, z)
    work.monitor === nothing && return nothing
    work.data.temporal.policy.split || return nothing
    monitor = work.monitor_work
    !monitor.captured && isfinite(dz) && dz > 0 && z > work.monitor.committed ||
        throw(ArgumentError("split observation needs one fresh positive accepted step"))
    _observation_monitor_inverse!(work, field)
    split_reading!(
        monitor.pending, monitor.samples, dz, work.data.temporal.weights,
        monitor.power, work.execution
    )
    monitor.captured, monitor.capture_z, monitor.capture_dz = true, z, dz
    return nothing
end

"""
    observation_accepted!(work, committed_field, z, dz, notices)

Consume one accepted split capture and update the optional temporal monitor.

Inputs follow `capture_split!`, but the field is the fully committed, finally masked
state. Accumulate each split removal and expected noise removal exactly once; read a
fresh coarse field only when the existing cadence/activity policy is due. The latest
half-split occupancy is judged while crossing uses the committed spectrum. Return
`(quiet, step_cap)` as a fixed named tuple for the caller's collar/controller policy.
No periodic statistic or output row is appended. A missing/mismatched capture, repeat
commit or invalid input raises; the session owns failure poisoning and notice delivery.
"""
function observation_accepted!(work::_ObservationWorkspace, field, z, dz, notices)
    _observation_field(work, field, z)
    work.monitor === nothing && return _observation_decision(work)
    isfinite(dz) && dz > 0 && z > work.monitor.committed ||
        throw(ArgumentError("monitor acceptance needs a new positive committed step"))
    monitor, policy = work.monitor_work, work.data.temporal.policy
    if policy.split
        monitor.captured && monitor.capture_z == z && monitor.capture_dz == dz ||
            throw(ArgumentError("committed split does not match its observation capture"))
        removed = monitor.reading.removed + monitor.pending.removed
        isfinite(removed) || throw(ArgumentError("accumulated split removal overflowed"))
        monitor.reading.removed = removed
        monitor.reading.found, monitor.reading.has_found = monitor.pending.found, true
        monitor.captured = false
    end
    floor_taken = policy.split ?
        temporal_floor_taken(monitor.floor, work.data.temporal.weights, dz) : 0.0
    if temporal_due(work.monitor, policy, z)
        _observation_read!(work, field)
        temporal_step!(work.monitor, policy, monitor.reading, z, notices; floor_taken)
        _observation_drain!(monitor.reading)
    else
        temporal_skip!(work.monitor, policy, z; floor_taken)
    end
    return _observation_decision(work)
end

"""
    observation_check!(work, field, z, notices; plane=true)

Take a requested fresh reading at the already committed monitor position.

This is a stale check at a saved plane. It consumes accumulated removals
without crediting another accepted step or noise removal. Return the same fixed
policy tuple as `observation_accepted!`. Require exact committed `z` and no pending
split capture. Without a monitor, return its inactive tuple. Require `plane=true`
and a stale reading before any mutation; already-current readings use
`observation_plane!` directly instead of counting another step.
"""
function observation_check!(
        work::_ObservationWorkspace, field, z, notices;
        plane::Bool = true
    )
    _observation_field(work, field, z)
    work.monitor === nothing && return _observation_decision(work)
    z == work.monitor.committed && !work.monitor_work.captured &&
        plane && work.monitor.stale ||
        throw(ArgumentError("monitor recheck requires a stale committed plane"))
    _observation_read!(work, field)
    temporal_step!(
        work.monitor, work.data.temporal.policy, work.monitor_work.reading,
        z, notices; plane
    )
    _observation_drain!(work.monitor_work.reading)
    return _observation_decision(work)
end

"""
    observation_plane!(work, z)

Return the optional numerical temporal plane row without reconstructing a field.

`z` is finite in m. Unmeasured dense planes retain the leaf's NaN occupancies and
resumed duplicate planes retain `emit=false`. The session owns row storage/cursors.
Return `nothing` without a monitor; reject a nonfinite position.
"""
function observation_plane!(work::_ObservationWorkspace, z)
    isfinite(z) || throw(ArgumentError("observation plane position must be finite"))
    return work.monitor === nothing ? nothing : temporal_plane!(work.monitor, z)
end

"""
    observation_finish!(work, notices)

Apply the optional final temporal removal notice to caller-owned notice storage.

No rows, counters or fields are reconstructed. Return `nothing`. An outstanding split
capture raises rather than silently losing an accepted step's removal; `notices` must
have capacity for the existing leaf's final warning. The caller checks a stale final
committed field before this hook, so all accumulated removals have been credited.
"""
function observation_finish!(work::_ObservationWorkspace, notices)
    work.monitor === nothing && return nothing
    !work.monitor_work.captured ||
        throw(ArgumentError("unconsumed split at monitor finish"))
    temporal_finish!(work.monitor, work.data.temporal.policy, notices)
    return nothing
end
