# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public ObservationProfiles, ObservationPopulations, PopulationMeasure
public ObservationAxisPopulations
public TemporalObservationData, ObservationPreparation, ObservationWorkspace
public observe!, observation_launch!, capture_split!, observation_accepted!
public observation_check!, observation_plane!, observation_finish!

abstract type _ObservationPreparation end
abstract type _ObservationWorkspace end

_observation_arithmetic(::_ObservationPreparation) = SourceArithmetic(Float64)

function _observation_axis_map(execution, matrix, arithmetic, name)
    matrix === nothing && return (nothing, nothing, nothing)
    if source_type(arithmetic) === Float64
        owned = upload(execution, matrix)
        return (owned, owned, nothing)
    end
    host = Matrix{Float64}(matrix)
    owned, range = source_narrow(
        execution, host, arithmetic, name;
        exponent = arithmetic.state_exponent - arithmetic.field_exponent,
    )
    return host, owned, range
end

function _observation_taper(arithmetic, scale)
    return source_type(arithmetic) === Float64 ? scale :
        source_coefficient(arithmetic, :observation_taper, scale)
end

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
    PopulationMeasure(weights; area=SourceProfile(1.0), taper=SourceProfile(1.0))

Copy explicit reference transverse weights for the final species ionisation inventory.

`weights` is a nonempty nonnegative finite binary64 vector. Its physical value at z
is `weights * area(z) / taper(z)^2` in m². Resolved modal data use their reference
node weights and own taper; geometry uses physical weights with unit profiles; a
mode-averaged route uses `[1]` with its represented effective area in `area`.
The caller chooses this existing route convention. Both profiles are independent
`SourceProfile` records and must evaluate positive finite where used. No map or
material response is inspected to infer the measure. Invalid inputs raise before copy.
"""
struct PopulationMeasure{W, A, T}
    weights::W
    area::A
    taper::T
end

function PopulationMeasure(
        weights::AbstractVector{Float64};
        area::SourceProfile = SourceProfile(1.0),
        taper::SourceProfile = SourceProfile(1.0)
    )
    !isempty(weights) && all(value -> isfinite(value) && value >= 0, weights) ||
        throw(ArgumentError("population measure requires finite nonnegative weights"))
    return PopulationMeasure(Vector{Float64}(weights), area, taper)
end

function _population_measure_factor(measure::PopulationMeasure, position)
    area, taper = _source_profile(measure.area, position),
        _source_profile(measure.taper, position)
    isfinite(area) && area > 0 && isfinite(taper) && taper > 0 ||
        throw(ArgumentError("population area and taper must be positive finite"))
    value = area / taper^2
    isfinite(value) && value > 0 ||
        throw(ArgumentError("population physical measure scale is out of range"))
    return value
end

function _population_inventory_data(potentials, measure, species, nodes)
    if potentials !== nothing
        potentials isa AbstractVector{Float64} && length(potentials) == species &&
            all(value -> isfinite(value) && value >= 0, potentials) ||
            throw(ArgumentError("species potentials need finite nonnegative binary64 data"))
    end
    measure === nothing || (
        measure isa PopulationMeasure &&
            length(measure.weights) == nodes
    ) ||
        throw(DimensionMismatch("population physical measure does not match its nodes"))
    return potentials === nothing ? nothing : copy(potentials)
end

"""
    ObservationPopulations(plasma, grid, map; species_densities, channel_species,
                           peak_species, potentials=nothing, measure=nothing)

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
Optional complete-species `potentials` in J and `PopulationMeasure` are required only
for the species ionisation inventory; they never enter the population recurrence.
No mutable source instance or stage scratch is retained. Invalid mappings or field/
envelope channel combinations raise `ArgumentError` or `DimensionMismatch`.
"""
struct ObservationPopulations{P, G, M, D, I, U, W}
    plasma::P
    grid::G
    map::M
    species_densities::D
    channel_species::I
    peak_species::Int
    potentials::U
    measure::W
end

function ObservationPopulations(
        plasma::Union{Nothing, ModalPlasma}, grid::ModalSourceGrid, map::ModalSourceMap;
        species_densities::Tuple, channel_species::Tuple, peak_species::Int,
        potentials = nothing, measure = nothing,
    )
    !isempty(species_densities) &&
        all(profile -> profile isa SourceProfile, species_densities) ||
        throw(ArgumentError("population observations need complete species profiles"))
    channels = plasma === nothing ? () : plasma.channels
    map.arithmetic == grid.arithmetic &&
        (plasma === nothing || plasma.arithmetic == grid.arithmetic) ||
        throw(ArgumentError("population grid, map and channels need matching arithmetic"))
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
        plasma.rule.dt == source_type(grid.arithmetic)(grid.dt) &&
            length(plasma.rule.taps) <= grid.n_time ||
            throw(DimensionMismatch("population integral disagrees with its fine grid"))
    end
    owned_potentials = _population_inventory_data(
        potentials, measure, length(species_densities), size(map.synthesis, 1),
    )
    return ObservationPopulations(
        plasma, grid, map, species_densities, channel_species, peak_species,
        owned_potentials, measure,
    )
end

# Exact products of represented binary64 data are compared at preparation only.
# A rounded cross-product equality can admit a genuinely elliptical envelope.
function _observation_scalar_axis(matrix::Matrix{Float64})
    size(matrix, 1) == 1 && return matrix
    first_maximum = maximum(abs, view(matrix, 1, :))
    second_maximum = maximum(abs, view(matrix, 2, :))
    base = first_maximum >= second_maximum ? 1 : 2
    other = 3 - base
    pivot = 1
    for column in axes(matrix, 2)
        abs(matrix[base, column]) > abs(matrix[base, pivot]) && (pivot = column)
    end
    iszero(matrix[base, pivot]) && return zeros(1, size(matrix, 2))
    first = Rational{BigInt}(matrix[base, pivot])
    second = Rational{BigInt}(matrix[other, pivot])
    for column in axes(matrix, 2)
        Rational{BigInt}(matrix[base, column]) * second ==
            Rational{BigInt}(matrix[other, column]) * first ||
            throw(ArgumentError("envelope on-axis plasma requires proportional real rows"))
    end
    ratio = matrix[other, pivot] / matrix[base, pivot]
    factor = hypot(1.0, ratio)
    owned = Matrix{Float64}(undef, 1, size(matrix, 2))
    for column in axes(matrix, 2)
        owned[1, column] = factor * matrix[base, column]
    end
    all(isfinite, owned) || throw(ArgumentError("scalar on-axis map is out of range"))
    return owned
end

"""
    ObservationAxisPopulations(plasma, grid, synthesis; species_densities,
                               channel_species, taper=SourceProfile(1.0))

Own a fresh resolved-modal axis-population reconstruction in the scalar admitted scope.

`plasma` and `grid` are matching `ModalPlasma` and fine `ModalSourceGrid` records.
`synthesis` is a finite real binary64 `(n_components,n_modes)` map with one or two
Cartesian rows and the existing fine-grid padding scale already folded in. Its own
positive `SourceProfile` taper is applied at the observed z. Species profiles and
unique one-based channel mapping retain the complete mixture order, including passive
species. A nonempty plasma channel tuple is required. No mutable current or material
source workspace is retained. Current-only coefficients in the immutable channel
record do not enter admission or evaluation of populations.

For a field grid, both real rows are reconstructed and their Euclidean magnitude
feeds the scalar rate. For an envelope grid, two rows must be exactly proportional
as represented dyadic numbers; zero rows are allowed. Compare cross-products exactly
at preparation, then collapse the larger row by `hypot(1,other[pivot]/base[pivot])`.
This preserves a complex scalar times a fixed real polarization direction, with the
ordinary binary64 preparation rounding. Nonproportional rows can represent elliptical
envelopes and raise by name before workspace allocation; no envelope magnitude
shortcut is substituted for the vector cycle model. Malformed data also raise.
"""
struct ObservationAxisPopulations{P, G, A, D, I, T, H, R}
    plasma::P
    grid::G
    synthesis::A
    species_densities::D
    channel_species::I
    taper::T
    host_synthesis::H
    range::R
end

function ObservationAxisPopulations(
        plasma::ModalPlasma, grid::ModalSourceGrid,
        synthesis; species_densities::Tuple, channel_species::Tuple,
        taper::SourceProfile = SourceProfile(1.0)
    )
    synthesis isa AbstractMatrix{Float64} && 1 <= size(synthesis, 1) <= 2 &&
        size(synthesis, 2) > 0 && all(isfinite, synthesis) ||
        throw(ArgumentError("fine on-axis synthesis needs one or two finite binary64 rows"))
    plasma.arithmetic == grid.arithmetic ||
        throw(ArgumentError("on-axis grid and channels need matching source arithmetic"))
    !isempty(species_densities) &&
        all(profile -> profile isa SourceProfile, species_densities) ||
        throw(ArgumentError("on-axis populations need complete species profiles"))
    !isempty(plasma.channels) && length(channel_species) == length(plasma.channels) &&
        all(
        index -> index isa Int && 1 <= index <= length(species_densities),
        channel_species
    ) && length(unique(channel_species)) == length(channel_species) ||
        throw(ArgumentError("on-axis population species mapping is invalid"))
    field = grid.kind == :field
    (field ? plasma.rule isa RunningIntegralRule : plasma.rule === nothing) &&
        all(channel -> (channel.quadrature === nothing) == field, plasma.channels) ||
        throw(ArgumentError("on-axis population channels disagree with their grid kind"))
    if field
        plasma.rule.dt == source_type(grid.arithmetic)(grid.dt) &&
            length(plasma.rule.taps) <= grid.n_time ||
            throw(DimensionMismatch("on-axis population integral disagrees with its grid"))
    end
    owned = Matrix{Float64}(synthesis)
    grid.kind == :envelope && (owned = _observation_scalar_axis(owned))
    arithmetic = grid.arithmetic
    narrow, range = if source_type(arithmetic) === Float64
        (owned, nothing)
    else
        _source_host_narrow(
            owned, arithmetic, :observation_axis_population;
            exponent = arithmetic.state_exponent - arithmetic.field_exponent,
        )
    end
    return ObservationAxisPopulations(
        plasma, grid, narrow, species_densities,
        channel_species, taper, owned, range
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
        populations=nothing, temporal=nothing, axis_populations=nothing,
        arithmetic=SourceArithmetic(Float64))

Own numerical preparation for scalar modal statistics and temporal monitoring.

`grid` is a coarse `StatisticsGrid`, and `selection` fixes packed plane/periodic slots.
`kind` is `:field` or `:envelope`; `n_modes` is positive. `power_weight` is a prepared
`ObservationWeight` for the `(n_omega,n_modes)` state. Optional physical on-axis
`axis_synthesis` has one or two Cartesian rows and `n_modes` columns, with no padding
factor. It is rescaled by its own positive `taper` profile at each observation. Axis
intensity requires the explicitly translated positive `intensity_factor` in SI units.
Optional population/monitor records are `ObservationPopulations` and
`TemporalObservationData`. `ObservationAxisPopulations` supplies the separate fine
axis map and scan for the four plane-only on-axis plasma statistics. No Python object,
session cursor or mutable source is owned.

`arithmetic` selects binary32 or binary64 stored work and explicit state/field powers.
Population/grid/map arithmetic must match it exactly. The coarse axis matrix folds
the state-to-field power once; its original binary64 matrix remains immutable in
`host_axis_synthesis`. Power weights evaluate wide and normalize before binary32
products. Packed values, coordinates and monitor continuations remain binary64.
This composition requires CPU and refuses spatial statistics or vector propagation.
The lowerer retains the catalogue's route eligibility rules.
Missing required families and invalid numerical layouts raise before workspace creation.
"""
struct ObservationPreparation{G, S, W, P, A, T, L, M, X, R, H, N} <:
    _ObservationPreparation
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
    axis_populations::X
    needs::StatisticsNeeds
    periodic_needs::StatisticsNeeds
    arithmetic::R
    host_axis_synthesis::H
    ranges::N
end

_observation_arithmetic(data::ObservationPreparation) = data.arithmetic

function _observation_population_requirements(selection, populations, kind)
    diagnostics = statistics_needs(selection).plasma_diagnostics
    if diagnostics
        populations !== nothing && kind == :field && populations.plasma !== nothing &&
            !isempty(populations.plasma.channels) ||
            throw(ArgumentError("plasma diagnostics require field plasma channels"))
    end
    if any(slot -> slot.code == 28, selection.slots)
        populations !== nothing && populations.potentials !== nothing &&
            populations.measure !== nothing ||
            throw(ArgumentError("species inventory needs potentials and physical measure"))
    end
    return nothing
end

function ObservationPreparation(
        execution, grid::StatisticsGrid, selection::StatisticsSelection;
        kind::Symbol, n_modes::Int,
        power_weight::Union{ObservationWeight, NormalizedObservationWeight},
        profiles::ObservationProfiles = ObservationProfiles(), axis_synthesis = nothing,
        taper::SourceProfile = SourceProfile(1.0), intensity_factor = nothing,
        populations = nothing, temporal = nothing, axis_populations = nothing,
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    oncpu(execution) || throw(ArgumentError("modal observations currently require CPU"))
    n_time, n_frequency = length(grid.time), length(grid.omega)
    T = source_type(arithmetic)
    T === Float64 && power_weight isa NormalizedObservationWeight &&
        throw(ArgumentError("binary64 observations require their original wide weights"))
    if T === Float32
        _float32_sum_count(n_time)
        _float32_sum_count(n_frequency)
        _float32_sum_count(n_modes)
    end
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
        populations.grid.arithmetic == arithmetic ||
            throw(ArgumentError("population and observation source arithmetic differ"))
        if T === Float32
            _float32_sum_count(size(populations.map.synthesis, 1))
            _float32_sum_count(length(populations.species_densities))
        end
    end
    _observation_population_requirements(selection, populations, kind)
    needs.axis_populations && !(axis_populations isa ObservationAxisPopulations) &&
        throw(ArgumentError("on-axis statistics need independent fine population data"))
    if axis_populations !== nothing
        axis_populations isa ObservationAxisPopulations &&
            axis_populations.grid.kind == kind &&
            length(axis_populations.grid.omega) == n_frequency &&
            size(axis_populations.synthesis, 2) == n_modes ||
            throw(DimensionMismatch("on-axis population preparation disagrees with state"))
        axis_populations.grid.arithmetic == arithmetic ||
            throw(ArgumentError("on-axis and observation source arithmetic differ"))
        T === Float32 &&
            _float32_sum_count(length(axis_populations.species_densities))
    end
    any(slot -> slot.code == 34, selection.slots) &&
        length(grid.spectral_density) != n_frequency &&
        throw(ArgumentError("spectral_flux needs the coarse spectral-density weights"))
    for slot in selection.slots
        expected = if slot.code in (1, 2, 4)
            n_modes
        elseif slot.code in (3, 5)
            n_modes * size(grid.windows, 2)
        elseif slot.code == 34
            n_frequency
        elseif slot.code in (32, 33)
            axis_populations === nothing &&
                throw(ArgumentError("on-axis population preparation is missing"))
            length(axis_populations.species_densities)
        elseif slot.code in (14, 15, 27, 28, 29)
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
        T === Float32 && buckets !== nothing &&
            _float32_sum_count(Base.checked_mul(n_frequency, n_modes))
    end
    owned_grid = StatisticsGrid(
        copy(grid.omega), copy(grid.time), copy(grid.parseval);
        windows = copy(grid.windows), spectral_density = copy(grid.spectral_density),
        d_omega = grid.d_omega,
    )
    host_axis, axis, axis_range = _observation_axis_map(
        execution, axis_synthesis, arithmetic, :observation_axis_synthesis
    )
    metric = T === Float32 && power_weight isa ObservationWeight ?
        NormalizedObservationWeight(execution, power_weight, arithmetic) : power_weight
    return ObservationPreparation(
        kind, n_modes, owned_grid, selection, metric, profiles, axis, taper,
        intensity_factor === nothing ? 1.0 : intensity_factor, populations, temporal,
        axis_populations, needs, statistics_needs(selection; periodic = true),
        arithmetic, host_axis, (axis = axis_range,),
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

function _observation_pair(
        execution::KernelExecution, n_time::Int, n_columns::Int,
        T::Union{Type{Float32}, Type{Float64}} = Float64,
    )
    spectrum, weighted_spectrum = dzeros(execution, Complex{T}, n_time, n_columns),
        dzeros(execution, Complex{T}, n_time, n_columns)
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

struct _ObservationPopulationWork{A, T, F, P, W, C, D, V}
    spectrum::A
    modal::T
    transform::F
    padding::P
    nodes::T
    workspace::W
    chunks::C
    densities::Vector{Float64}
    fractions::V
    peaks::V
    diagnostics::D
end

function _observation_population_work(
        execution, data, n_modes, node_chunk;
        diagnostics::Bool = false
    )
    data === nothing && return nothing
    grid, maps = data.grid, data.map
    T = source_type(grid.arithmetic)
    n_nodes, n_species = size(maps.synthesis, 1), length(data.species_densities)
    capacity = min(node_chunk, n_nodes)
    field = grid.kind == :field
    scalar = field ? T : Complex{T}
    frequencies = field ? grid.n_time ÷ 2 + 1 : grid.n_time
    spectrum = dzeros(execution, Complex{T}, frequencies, n_modes)
    modal = dzeros(execution, scalar, grid.n_time, n_modes)
    nodes = dzeros(execution, scalar, grid.n_time, capacity)
    layout = ResponseLayout(grid.n_time, 1, capacity)
    workspace = data.plasma === nothing ? nothing :
        PopulationWorkspace(execution, T, layout)
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
        SpectralPadding(execution, grid.crop_index, frequencies, one(T), one(T)),
        nodes, workspace, chunks,
        zeros(n_species), dzeros(execution, T, n_nodes, n_species),
        dzeros(execution, T, n_nodes, n_species),
        diagnostics ? PlasmaDiagnosticWorkspace(execution, T, capacity) : nothing,
    )
end

struct _ObservationAxisPopulationWork{A, T, F, P, W, V}
    spectrum::A
    modal::T
    transform::F
    padding::P
    components::T
    nodes::T
    workspace::W
    densities::Vector{Float64}
    fractions::V
    peaks::V
    diagnostics::Nothing
end

function _observation_axis_population_work(execution, data::ObservationAxisPopulations)
    grid, modes = data.grid, size(data.synthesis, 2)
    T = source_type(grid.arithmetic)
    field = grid.kind == :field
    scalar = field ? T : Complex{T}
    frequencies = field ? grid.n_time ÷ 2 + 1 : grid.n_time
    spectrum = dzeros(execution, Complex{T}, frequencies, modes)
    modal = dzeros(execution, scalar, grid.n_time, modes)
    species = length(data.species_densities)
    return _ObservationAxisPopulationWork(
        spectrum, modal,
        TemporalFFT(execution, modal, spectrum),
        SpectralPadding(execution, grid.crop_index, frequencies, one(T), one(T)),
        dzeros(execution, scalar, grid.n_time, size(data.synthesis, 1)),
        dzeros(execution, scalar, grid.n_time, 1),
        PopulationWorkspace(execution, T, ResponseLayout(grid.n_time, 1, 1)),
        zeros(species), dzeros(execution, T, 1, species),
        dzeros(execution, T, 1, species), nothing
    )
end

mutable struct _ObservationMonitorWork{A, T, F, R, V}
    spectrum::A
    samples::T
    transform::F
    power::V
    reading::R
    pending::R
    floor::TemporalFloor
    captured::Bool
    capture_z::Float64
    capture_dz::Float64
end

function _observation_monitor_work(
        execution::KernelExecution, data::_ObservationPreparation, noise,
    )
    return _observation_monitor_work(execution, data, noise, _observation_arithmetic(data))
end

function _observation_monitor_work(
        execution::KernelExecution, data::_ObservationPreparation, noise,
        arithmetic::Union{SourceArithmetic{Float32}, SourceArithmetic{Float64}},
    )
    T = source_type(arithmetic)
    data.temporal === nothing && return nothing
    temporal = data.temporal
    T === Float32 && noise !== nothing &&
        throw(ArgumentError("binary32 observations do not support input noise"))
    n_time, n_modes = length(data.grid.time), data.n_modes
    spectrum = dzeros(execution, Complex{T}, length(data.grid.omega), n_modes)
    samples = dzeros(execution, data.kind == :field ? T : Complex{T}, n_time, n_modes)
    buckets = temporal.weights.buckets
    n_buckets = length(temporal.policy.bucket_limits)
    reading, pending = TemporalReading(n_buckets), TemporalReading(n_buckets)
    work = _ObservationMonitorWork(
        spectrum, samples,
        TemporalFFT(execution, samples, spectrum), dzeros(execution, T, n_time),
        reading, pending,
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

_observation_scale_arrays(::Nothing) = ()
_observation_scale_arrays(scales::_ObservationScales) = (scales.columns, scales.nodes)

_observation_arrays(::Nothing) = ()
_observation_arrays(pair::_ObservationPair) =
    (pair.spectrum, pair.weighted_spectrum, pair.samples, pair.weighted)
_observation_arrays(windows::_ObservationWindows) = _observation_arrays(windows.pair)
_observation_arrays(work::_ObservationMonitorWork) =
    (
    work.spectrum, work.samples, work.power,
    work.reading.occupancy, work.pending.occupancy,
)
_observation_arrays(work::_ObservationAxisPopulationWork) = (
    work.spectrum, work.modal, work.components, work.nodes, work.densities,
    work.fractions, work.peaks, _population_arrays(work.workspace),
    work.padding.crop, work.padding.uncrop,
)
_observation_arrays(work::PopulationWorkspace) = _population_arrays(work)
_observation_arrays(work::PlasmaDiagnosticWorkspace) =
    (work.parts, work.maxima, work.scales === nothing ? () : work.scales)
_observation_arrays(work::ObservationWeightWorkspace) = (work.host,)
_observation_arrays(work::AbstractVector) = (work,)
_observation_arrays(work::_ObservationPopulationWork) =
    (
    work.spectrum, work.modal, work.nodes, work.densities,
    work.fractions, work.peaks, _observation_arrays(work.workspace),
    _observation_arrays(work.diagnostics),
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
    _transfer_arrays_alias(input, array) &&
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
Binary32 arithmetic refuses input noise before allocating monitor work. An optional
`TemporalContinuation` is restored before numerical launch initialization.

`values` is the packed statistics output. Every temporal transform, window pair,
population matrix and monitor reading is independently owned, with all arrays listed
by `buffers`. `monitor` is its optional `TemporalMonitorState`; the caller owns notices,
row capacity, export and acknowledgment. `valid` is set only after successful observe!.
Warm numerical evaluation does not allocate or increment propagation source calls.
"""
mutable struct ObservationWorkspace{E, D, P, W, A, T, L, M, C, I, B, X, S, V, H} <:
    _ObservationWorkspace
    execution::E
    data::D
    spectrum::S
    weight::V
    weight_work::H
    pair::P
    windows::W
    axis::A
    statistics::T
    populations::L
    axis_populations::X
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
    T = source_type(prepared.arithmetic)
    T === Float32 && noise_spectrum !== nothing &&
        throw(ArgumentError("binary32 observations do not support input noise"))
    n_time, n_modes = length(prepared.grid.time), prepared.n_modes
    needs = prepared.needs
    spectrum = dzeros(execution, Complex{T}, prepared.power_weight.shape...)
    weight = dzeros(execution, T, prepared.power_weight.shape...)
    fill!(weight, one(T))
    weight_work = _observation_weight_work(prepared.power_weight)
    pair = needs.temporal || needs.intensity ?
        _observation_pair(execution, n_time, n_modes, T) : nothing
    n_windows = size(prepared.grid.windows, 2)
    windows = if needs.windows
        n_windows > 0 || throw(ArgumentError("windowed power needs spectral gates"))
        window_pair = _observation_pair(execution, n_time, n_modes * n_windows, T)
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
            dzeros(execution, Complex{T}, n_time, size(prepared.axis_synthesis, 1)),
            dzeros(execution, Complex{T}, n_time, size(prepared.axis_synthesis, 1)),
        )
    else
        nothing
    end
    populations = needs.populations ?
        _observation_population_work(
            execution, prepared.populations, n_modes, node_chunk;
            diagnostics = needs.plasma_diagnostics
        ) :
        nothing
    axis_populations = if needs.axis_populations
        axis_data = prepared.axis_populations
        axis_data === nothing &&
            throw(ArgumentError("on-axis population preparation is missing"))
        _observation_axis_population_work(execution, axis_data)
    else
        nothing
    end
    n_nodes = populations === nothing ? 0 : size(populations.fractions, 1)
    statistics = StatisticsWorkspace(
        execution, T;
        n_time, n_columns = n_modes, n_nodes
    )
    monitor_work = _observation_monitor_work(execution, prepared, noise_spectrum)
    monitor = monitor_work === nothing ? nothing : TemporalMonitorState()
    if temporal_continuation !== nothing
        monitor === nothing && throw(ArgumentError("monitor continuation needs a monitor"))
        restore_temporal!(monitor, temporal_continuation; plane = continuation_plane)
    end
    spectral_input = needs.spectral ? SpectralStatisticsInput(
            spectrum, weight; amplitude_exponent = prepared.arithmetic.state_exponent
        ) : nothing
    temporal_input = if needs.temporal || needs.windows || needs.intensity
        TemporalStatisticsInput(
            pair === nothing ? nothing : pair.samples,
            pair === nothing ? nothing : pair.weighted;
            window_analytic = windows === nothing ? nothing : windows.samples,
            window_weighted = windows === nothing ? nothing : windows.weighted,
            axis_analytic = axis === nothing ? nothing : axis[1],
            axis_weighted = axis === nothing ? nothing : axis[2],
            intensity_factor = prepared.intensity_factor,
            amplitude_exponent = prepared.arithmetic.state_exponent,
            axis_exponent = prepared.arithmetic.field_exponent,
        )
    else
        nothing
    end
    population_input = populations === nothing ? nothing : PopulationStatisticsInput(
            populations.densities, populations.fractions, populations.peaks;
            peak_species = prepared.populations.peak_species,
            potentials = prepared.populations.potentials,
            node_weights = prepared.populations.measure === nothing ? nothing :
            prepared.populations.measure.weights,
        )
    axis_input = axis_populations === nothing ? nothing : PopulationStatisticsInput(
            axis_populations.densities, axis_populations.fractions, axis_populations.peaks
        )
    inputs = StatisticsInputs(;
        spectral = spectral_input, temporal = temporal_input,
        populations = population_input, axis_populations = axis_input,
    )
    values = zeros(prepared.selection.width)
    buffers = (
        spectrum, weight, _observation_arrays(weight_work), values,
        _observation_arrays(pair), _observation_arrays(windows),
        axis === nothing ? () : axis, _observation_arrays(populations),
        _observation_arrays(axis_populations),
        _observation_arrays(monitor_work), statistics.power, statistics.total_power,
        statistics.column_energy, statistics.column_moment, statistics.node_total,
        statistics.fluence, statistics.magnitude, statistics.radial_average,
        _observation_scale_arrays(statistics.scales),
    )
    return ObservationWorkspace(
        execution, prepared, spectrum, weight, weight_work,
        pair, windows, axis, statistics, populations, axis_populations,
        monitor_work, monitor, inputs, values,
        buffers, false, NaN
    )
end


function _observation_field(work::ObservationWorkspace, field, z)
    field isa AbstractMatrix && eltype(field) === eltype(work.spectrum) &&
        size(field) == size(work.spectrum) ||
        throw(DimensionMismatch("observation field must match its stored modal arithmetic"))
    isfinite(z) && all(isfinite, field) ||
        throw(ArgumentError("observation position and field must be finite"))
    _observation_no_alias(field, work.buffers)
    return nothing
end

function _observation_analytic!(pair, field, weight, kind, windows, gated, execution)
    pair === nothing && throw(ArgumentError("analytic observation buffers are missing"))
    spectrum, weighted = pair.spectrum, pair.weighted_spectrum
    T = real(eltype(spectrum))
    nil, real_zero, single, doubled = zero(eltype(spectrum)), zero(T), one(T), T(2)
    n_frequency, n_modes = size(field)
    real_field = kind == :field
    each!(execution, size(spectrum)...) do row, column
        mode = mod(column - 1, n_modes) + 1
        window = div(column - 1, n_modes) + 1
        value = if row > n_frequency || (gated && !windows[row, window])
            nil
        elseif real_field && (row == 1 || row == n_frequency)
            complex(real(field[row, mode]), real_zero)
        else
            (real_field ? doubled : single) * field[row, mode]
        end
        spectrum[row, column] = value
        weighted[row, column] = row <= n_frequency ? value * weight[row, mode] : nil
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
    coefficient = _observation_taper(data.arithmetic, scale)
    for (destination, source) in (
            (work.axis[1], work.pair.samples),
            (work.axis[2], work.pair.weighted),
        )
        synthesize!(destination, source, data.axis_synthesis, execution; n_components = 1)
        _source_scale!(destination, coefficient, execution)
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

function _observation_channel!(
        work, data, channel, index, chunk, position, execution;
        diagnostics::Bool = false
    )
    initial = _source_profile(channel.preionisation, position)
    0 <= initial <= 1 ||
        throw(ArgumentError("invalid observation plasma preionisation"))
    if data.plasma.rule !== nothing
        field_populations!(
            work.nodes, channel.channel.rate,
            data.plasma.rule, work.workspace, execution;
            preionisation = initial, count = chunk.count,
            field_exponent = data.grid.arithmetic.field_exponent,
        )
    else
        envelope_populations!(
            work.nodes, channel.channel.rate,
            channel.quadrature, work.workspace, execution;
            preionisation = initial, dt = data.grid.dt, count = chunk.count,
            field_exponent = data.grid.arithmetic.field_exponent,
        )
    end
    failure = plasma_failure(work.workspace.status)
    failure.code == PlasmaSuccess || throw(
        SourcePlasmaError(
            failure.code, index, chunk.first + failure.node - 1, failure.sample
        )
    )
    if diagnostics
        work.diagnostics === nothing &&
            throw(ArgumentError("plasma diagnostic buffers are missing"))
        plasma_diagnostics!(
            work.diagnostics, work.nodes, work.workspace,
            data.plasma.rule, execution; count = chunk.count
        )
    end
    species = data.channel_species[index]
    fraction, rate = work.workspace.fraction, work.workspace.rate
    fractions, peaks, first = work.fractions, work.peaks, chunk.first
    each_column!(execution, chunk.count) do node
        peak = zero(eltype(rate))
        for row in axes(rate, 1)
            peak = max(peak, rate[row, node])
        end
        fractions[first + node - 1, species] = fraction[end, node]
        peaks[first + node - 1, species] = peak
    end
    return nothing
end

function _observation_populations!(
        work, data, field, position, execution;
        diagnostics::Bool = false
    )
    work === nothing && throw(ArgumentError("population observation buffers are missing"))
    if diagnostics
        work.diagnostics === nothing &&
            throw(ArgumentError("plasma diagnostic buffers are missing"))
        reset_plasma_diagnostics!(work.diagnostics)
    end
    _observation_densities!(work.densities, data.species_densities, position, 1)
    fill!(work.fractions, 0.0)
    fill!(work.peaks, 0.0)
    data.plasma === nothing && return nothing
    scale = _source_profile(data.map.taper, position)
    scale > 0 || throw(ArgumentError("population taper must be positive"))
    coefficient = _observation_taper(data.grid.arithmetic, scale)
    pad_spectrum!(work.spectrum, field, work.padding, execution)
    temporal_inverse!(work.modal, work.transform, work.spectrum, execution)
    for chunk in work.chunks
        synthesize!(
            work.nodes, work.modal, chunk.synthesis, execution;
            n_components = 1, active_nodes = chunk.count
        )
        _source_scale!(work.nodes, coefficient, execution)
        all(isfinite, work.nodes) ||
            throw(ArgumentError("population field reconstruction overflowed"))
        _source_each_indexed(data.plasma.channels) do channel, index
            _observation_channel!(
                work, data, channel, index, chunk, position, execution;
                diagnostics
            )
        end
    end
    return nothing
end

@inline function _observation_axis_sample(samples::AbstractMatrix{<:Real}, row)
    return size(samples, 2) == 2 ? hypot(samples[row, 1], samples[row, 2]) : samples[row, 1]
end
@inline _observation_axis_sample(samples::AbstractMatrix{<:Complex}, row) = samples[row, 1]

function _observation_axis_populations!(
        work, data::ObservationAxisPopulations,
        field, position, execution
    )
    _observation_densities!(work.densities, data.species_densities, position, 1)
    fill!(work.fractions, 0.0)
    fill!(work.peaks, 0.0)
    scale = _source_profile(data.taper, position)
    isfinite(scale) && scale > 0 || throw(ArgumentError("on-axis taper must be positive"))
    coefficient = _observation_taper(data.grid.arithmetic, scale)
    pad_spectrum!(work.spectrum, field, work.padding, execution)
    temporal_inverse!(work.modal, work.transform, work.spectrum, execution)
    synthesize!(work.components, work.modal, data.synthesis, execution; n_components = 1)
    _source_scale!(work.components, coefficient, execution)
    components, output = work.components, work.nodes
    data.grid.kind == :envelope && size(components, 2) != 1 &&
        throw(ArgumentError("scalar envelope axis work requires one collapsed row"))
    each!(execution, size(output, 1), 1) do row, _
        output[row, 1] = _observation_axis_sample(components, row)
    end
    all(isfinite, output) || throw(ArgumentError("on-axis population field overflowed"))
    _source_each_indexed(data.plasma.channels) do channel, index
        _observation_channel!(
            work, data, channel, index, (first = 1, count = 1),
            position, execution
        )
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

_observation_identity(::Nothing, identity, weight_exponent) = nothing

function _observation_weight_work(weight::ObservationWeight)
    count = weight.weights === nothing ? 0 : size(weight.weights.coefficients, 1)
    return zeros(count)
end
_observation_weight_work(weight::NormalizedObservationWeight) =
    ObservationWeightWorkspace(weight)

function _observation_weight!(output, weight::ObservationWeight, position, execution, work)
    evaluate_weight!(output, weight, position, execution; work)
    return 0, all(isone, output)
end
function _observation_weight!(
        output, weight::NormalizedObservationWeight, position, execution, work
    )
    exponent = evaluate_weight!(output, weight, position, execution; work)
    return exponent, work.identity
end

_observation_spectral(::Nothing, identity, weight_exponent) = nothing
function _observation_spectral(input::SpectralStatisticsInput, identity, weight_exponent)
    return SpectralStatisticsInput(
        input.spectrum, input.power_weight; total = input.total,
        amplitude_exponent = input.amplitude_exponent, weight_exponent,
        identity_weight = identity,
    )
end
function _observation_diagnostics(::Nothing, selected)::NTuple{3, Float64}
    selected && throw(ArgumentError("plasma diagnostic buffers are missing"))
    return (0.0, 0.0, 0.0)
end
function _observation_diagnostics(
        work::PlasmaDiagnosticWorkspace, selected
    )::NTuple{3, Float64}
    return selected ? plasma_diagnostic_values(work) : (0.0, 0.0, 0.0)
end
_observation_diagnostic_work(::Nothing) = nothing
_observation_diagnostic_work(work) = work.diagnostics

_observation_population_input(::Nothing, data, position, selected) = nothing
function _observation_population_input(
        input::PopulationStatisticsInput, data,
        position, selected
    )
    factor = if selected
        data.measure === nothing && throw(ArgumentError("population measure is missing"))
        _population_measure_factor(data.measure, position)
    else
        1.0
    end
    return PopulationStatisticsInput(
        input.densities, input.fractions, input.peak_rates;
        peak_species = input.peak_species, potentials = input.potentials,
        node_weights = input.node_weights, measure_factor = factor
    )
end

function _observation_identity(input::TemporalStatisticsInput, identity, weight_exponent)
    return TemporalStatisticsInput(
        input.analytic, input.weighted;
        window_analytic = input.window_analytic, window_weighted = input.window_weighted,
        axis_analytic = input.axis_analytic, axis_weighted = input.axis_weighted,
        intensity_factor = input.intensity_factor, identity_weight = identity,
        amplitude_exponent = input.amplitude_exponent, weight_exponent,
        axis_exponent = input.axis_exponent,
    )
end

"""
    observe!(work, field, z; periodic=false, quadrature_error=NaN)

Reconstruct fresh modal statistics at the actual supplied field and position.

`work` is an `ObservationWorkspace`. Complex `field` has the prepared arithmetic and
coarse state shape `(n_omega,n_modes)`, in stored units, and is preserved. `z` is finite
in m. `periodic` selects exactly
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
    weight_exponent, identity = 0, false
    if weighted
        copyto!(work.spectrum, field)
        weight_exponent, identity = _observation_weight!(
            work.weight, data.power_weight, z, execution, work.weight_work,
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
        _observation_populations!(
        work.populations, data.populations, field, z, execution;
        diagnostics = needs.plasma_diagnostics
    )
    if needs.axis_populations
        work.axis_populations === nothing &&
            throw(ArgumentError("axis population buffers are missing"))
        data.axis_populations === nothing &&
            throw(ArgumentError("axis population preparation is missing"))
        _observation_axis_populations!(
            work.axis_populations, data.axis_populations, field, z, execution
        )
    end
    temporal = _observation_identity(work.inputs.temporal, identity, weight_exponent)
    spectral = _observation_spectral(work.inputs.spectral, identity, weight_exponent)
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
    populations = _observation_population_input(
        work.inputs.populations, data.populations, z,
        _observation_selected(data.selection, 28, periodic),
    )
    diagnostics = _observation_diagnostics(
        _observation_diagnostic_work(work.populations), needs.plasma_diagnostics,
    )
    inputs = StatisticsInputs(;
        spectral, temporal, populations,
        profiles, quadrature_error, diagnostics,
        axis_populations = work.inputs.axis_populations,
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
        work.execution;
        spectrum = weights.buckets === nothing ? nothing : monitor.spectrum,
        amplitude_exponent = _observation_arithmetic(work.data).state_exponent,
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
        monitor.power, work.execution;
        amplitude_exponent = _observation_arithmetic(work.data).state_exponent,
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
