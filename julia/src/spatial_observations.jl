# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public SpatialFluxData, RadialObservationMap, CartesianObservationMap
public SpatialObservationPopulations, SpatialObservationPreparation
public SpatialObservationWorkspace

"""
    SpatialFluxData(base::SeparableSpatialOperator; material=nothing, light_speed)

Retain the independently translated plane measure for scalar spatial observations.

`base` owns the frequency/transverse squared wave numbers and their residuals in m⁻².
Optional `material` is a direct-value complex binary64 `ChebyshevTable` with one k²
value per frequency, as `translate.spatial.FluxPlan.material` specifies for a radial
gradient. It replaces only the active high material word; the low word is retained.
The table is copied. `light_speed` is the explicitly translated positive constant in
m/s. The already owned base is shared read-only, without interpreting its loss rates.

At observation, the protected root is `sqrt((k²-kperp²)+(k²_low-kperp²_low))`.
The dimensionless admittance is `light_speed*root/omega` only on the active positive
frequency band with `real(q)>0`, and exactly zero elsewhere. The internal magnetic
partner is `conj(admittance)*field`, following SEP 34 and `axial_admittance`.
Malformed coefficients, constants or table layouts raise before workspace allocation.
"""
struct SpatialFluxData{M}
    base::SeparableSpatialOperator
    material::M
    light_speed::Float64
end

function SpatialFluxData(
        base::SeparableSpatialOperator; material = nothing, light_speed::Float64,
    )
    isfinite(light_speed) && light_speed > 0 ||
        throw(ArgumentError("spatial admittance needs a positive finite light speed"))
    if material !== nothing
        material isa ChebyshevTable && eltype(material.coefficients) === ComplexF64 &&
            eltype(material.breakpoints) === Float64 &&
            size(material.coefficients, 1) == length(base.k_squared) ||
            throw(ArgumentError("material k-squared needs a complex binary64 table"))
    end
    return SpatialFluxData(
        base, material === nothing ? nothing : _copy_source_table(material), light_speed
    )
end

abstract type _SpatialObservationMap end

"""
    RadialObservationMap(synthesis; pad_scale, arithmetic=SourceArithmetic(Float64))

Copy the scalar radial coarse physical synthesis from its translated fine matrix.

`synthesis` is a finite real binary64 `(n_nodes,n_modes)` matrix with the temporal
oversampling factor already folded in. Divide once by positive finite `pad_scale`;
coarse spectra and analytic samples then reconstruct without fine padding. Fold the
state-to-field power from `arithmetic` once into `synthesis`; its original binary64
matrix remains independently owned in `host_synthesis` (aliased for binary64 work).
`ranges.synthesis` records binary32 conversion. The on-axis row remains an independent
preparation input. Invalid data or an unrepresentable direct fold raise.
"""
struct RadialObservationMap{M, A, H, R} <: _SpatialObservationMap
    synthesis::M
    arithmetic::A
    host_synthesis::H
    ranges::R

    function RadialObservationMap(
            ::Val{:owned}, synthesis::M, arithmetic::A, host_synthesis::H, ranges::R,
        ) where {M, A, H, R}
        return new{M, A, H, R}(synthesis, arithmetic, host_synthesis, ranges)
    end
end

function RadialObservationMap(
        synthesis; pad_scale::Float64,
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    synthesis isa AbstractMatrix{Float64} && !isempty(synthesis) &&
        all(isfinite, synthesis) && isfinite(pad_scale) && pad_scale > 0 ||
        throw(ArgumentError("radial observation synthesis needs finite binary64 data"))
    host = Matrix{Float64}(synthesis)
    host ./= pad_scale
    all(isfinite, host) || throw(ArgumentError("radial coarse synthesis overflowed"))
    narrow, range = source_type(arithmetic) === Float64 ? (host, nothing) :
        _source_host_narrow(
            host, arithmetic, :radial_observation_synthesis;
            exponent = arithmetic.state_exponent - arithmetic.field_exponent,
        )
    return RadialObservationMap(
        Val(:owned), narrow, arithmetic, host, (; synthesis = range),
    )
end

"""
    CartesianObservationMap(shape::NTuple{2,Int}; arithmetic=SourceArithmetic(Float64))

Retain an x-fastest scalar Cartesian grid or line without constructing a dense map.

`shape=(n_x,n_y)` requires `n_x>=2` and `n_y>=1`; a line has `n_y=1`.
Physical reconstruction uses the normalized transverse inverse raw DFT and the once
narrowed state-to-field power from `arithmetic`. Each workspace owns its plans and
band storage. Invalid or overflowing dimensions and an unrepresentable fold raise.
"""
struct CartesianObservationMap{A, T} <: _SpatialObservationMap
    shape::NTuple{2, Int}
    arithmetic::A
    scale::T

    function CartesianObservationMap(
            shape::NTuple{2, Int};
            arithmetic::SourceArithmetic = SourceArithmetic(Float64),
        )
        _cartesian_shape(shape)
        scale = source_coefficient(
            arithmetic, :cartesian_observation_synthesis, 1.0;
            exponent = arithmetic.state_exponent - arithmetic.field_exponent,
        )
        return new{typeof(arithmetic), typeof(scale)}(shape, arithmetic, scale)
    end
end

_spatial_map_shape(map::RadialObservationMap) = size(map.synthesis)
_spatial_map_shape(map::CartesianObservationMap) =
    (_cartesian_shape(map.shape), _cartesian_shape(map.shape))

"""
    SpatialObservationPopulations(plasma, grid; species_densities, channel_species,
                                  peak_species, pad_scale, potentials=nothing,
                                  measure=nothing)

Retain independent scalar Cartesian/line population inputs for fresh observations.

`plasma` is `ModalPlasma` or `nothing`, and `grid` is the existing fine temporal
`ModalSourceGrid`. Species/profile/index conventions match `ObservationPopulations`.
`pad_scale` is the positive translated temporal oversampling factor, applied once
after spatial reconstruction. Only rate, preionisation and integration/cycle data
enter the population kernels; no source or current buffers are retained. Optional
complete-species `potentials` in J and `PopulationMeasure` supply only the final
ionisation inventory. Invalid
species mappings or field/envelope channel combinations raise before allocation.
"""
struct SpatialObservationPopulations{P, G, D, I, U, W}
    plasma::P
    grid::G
    species_densities::D
    channel_species::I
    peak_species::Int
    pad_scale::Float64
    potentials::U
    measure::W
end

function SpatialObservationPopulations(
        plasma::Union{Nothing, ModalPlasma}, grid::ModalSourceGrid;
        species_densities::Tuple, channel_species::Tuple, peak_species::Int,
        pad_scale::Float64, potentials = nothing, measure = nothing,
    )
    !isempty(species_densities) &&
        all(profile -> profile isa SourceProfile, species_densities) ||
        throw(ArgumentError("spatial populations need complete species profiles"))
    channels = plasma === nothing ? () : plasma.channels
    length(channel_species) == length(channels) &&
        all(
        index -> index isa Int && 1 <= index <= length(species_densities),
        channel_species,
    ) && length(unique(channel_species)) == length(channel_species) &&
        0 <= peak_species <= length(species_densities) ||
        throw(ArgumentError("spatial population species mapping is invalid"))
    isfinite(pad_scale) && pad_scale > 0 ||
        throw(ArgumentError("spatial population padding scale must be positive finite"))
    (plasma === nothing || plasma.arithmetic == grid.arithmetic) ||
        throw(ArgumentError("spatial population inputs must use one arithmetic policy"))
    _validate_source_responses(grid, nothing, (), plasma)
    owned_potentials = _population_inventory_data(
        potentials, measure, length(species_densities),
        measure isa PopulationMeasure ? length(measure.weights) : 0,
    )
    return SpatialObservationPopulations(
        plasma, grid, species_densities, channel_species, peak_species, pad_scale,
        owned_potentials, measure,
    )
end

"""
    SpatialObservationPreparation(execution, grid, selection; kind, map, flux,
        spectral_weights, on_axis, intensity_factor, widths=nothing,
        profiles=ObservationProfiles(), populations=nothing, temporal=nothing,
        arithmetic=map.arithmetic)

Own numerical preparation for scalar radial, Cartesian or line observations.

`grid::StatisticsGrid` is coarse and `selection::StatisticsSelection` fixes packed
outputs. `kind` is `:field` or `:envelope`. `map` is a radial or Cartesian observation
map, `flux::SpatialFluxData` is the independent plane measure, and the positive
binary64 `spectral_weights` carry transverse Parseval measure per spectral column.
The complex binary64 `on_axis` vector is geometry's exact independent reconstruction
row, without temporal padding. Positive `intensity_factor` is the translated ε₀c/2.

Optional `widths::SpatialWidthGrid` supplies the same physical nodes and routes.
Radial populations use `ObservationPopulations`; Cartesian populations use
`SpatialObservationPopulations`. Existing independent `ObservationProfiles` and
`TemporalObservationData` are read-only. The monitor uses the canonical spectrum's
coarse inverse proxy, exactly as `BoundaryMonitor.power`, not physical axial flux.

Admit total energy, profiles, population diagnostics, physical peak/axis intensity,
widths, axis spectrum and the explicit quadrature alarm. Refuse modal-only, far-field,
planar and vector statistics by name before workspace allocation. `arithmetic` must
match the map and population grid exactly. Binary32 work retains its original wide
axis row in `host_on_axis`, evaluates each admittance in host binary64 at z, normalizes
its weight/closure coefficient, then rounds once before pointwise products. No mutable
binary64 state-sized scratch is retained. Packed values and monitor decisions remain
physical binary64. Unsupported execution, layouts or inconsistent inputs raise.
"""
struct SpatialObservationPreparation{G, S, M, F, V, A, W, P, L, T, R, H, N} <:
    _ObservationPreparation
    kind::Symbol
    n_modes::Int
    grid::G
    selection::S
    map::M
    flux::F
    spectral_weights::V
    on_axis::A
    intensity_factor::Float64
    widths::W
    profiles::P
    populations::L
    temporal::T
    needs::StatisticsNeeds
    periodic_needs::StatisticsNeeds
    arithmetic::R
    host_on_axis::H
    ranges::N
end

_observation_arithmetic(data::SpatialObservationPreparation) = data.arithmetic

function SpatialObservationPreparation(
        execution::KernelExecution, grid::StatisticsGrid, selection::StatisticsSelection;
        kind::Symbol, map::_SpatialObservationMap, flux::SpatialFluxData,
        spectral_weights, on_axis, intensity_factor::Float64, widths = nothing,
        profiles::ObservationProfiles = ObservationProfiles(), populations = nothing,
        temporal = nothing, arithmetic::SourceArithmetic = map.arithmetic,
    )
    oncpu(execution) || throw(ArgumentError("spatial observations currently require CPU"))
    arithmetic == map.arithmetic ||
        throw(ArgumentError("spatial observations and map must use one arithmetic policy"))
    populations === nothing || populations.grid.arithmetic == arithmetic ||
        throw(ArgumentError("spatial populations must match observation arithmetic"))
    n_time, n_frequency = length(grid.time), length(grid.omega)
    nodes, columns = _spatial_map_shape(map)
    kind in (:field, :envelope) && n_time >= 4 && iseven(n_time) ||
        throw(ArgumentError("invalid scalar spatial observation time grid"))
    n_frequency == (kind == :field ? n_time ÷ 2 + 1 : n_time) &&
        _spatial_operator_shape(flux.base) == (n_frequency, columns) ||
        throw(DimensionMismatch("spatial flux, map and time dimensions disagree"))
    all(
        array -> eltype(array) === Float64,
        (grid.time, grid.omega, grid.parseval, grid.spectral_density),
    ) || throw(ArgumentError("spatial observation grids require binary64 arrays"))
    map isa CartesianObservationMap && flux.material !== nothing &&
        throw(ArgumentError("Cartesian observations require constant linear material"))
    spectral_weights isa AbstractVector{Float64} && length(spectral_weights) == columns &&
        all(value -> isfinite(value) && value > 0, spectral_weights) ||
        throw(ArgumentError("spatial spectral measure needs positive binary64 weights"))
    on_axis isa AbstractVector{ComplexF64} && length(on_axis) == columns &&
        all(isfinite, on_axis) ||
        throw(ArgumentError("spatial on-axis weights must be complex binary64"))
    isfinite(intensity_factor) && intensity_factor > 0 ||
        throw(ArgumentError("spatial intensity conversion must be positive finite"))
    for slot in selection.slots
        slot.code in (
            2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29,
        ) ||
            throw(
            ArgumentError(
                "spatial statistic $(_STATISTIC_NAMES[slot.code]) is unsupported",
            )
        )
    end
    needs = statistics_needs(selection)
    _spatial_observation_populations(map, populations, kind, n_frequency, columns, nodes)
    _observation_population_requirements(selection, populations, kind)
    populations === nothing || populations.measure === nothing ||
        length(populations.measure.weights) == nodes ||
        throw(DimensionMismatch("spatial population measure does not match its nodes"))
    needs.populations && populations === nothing &&
        throw(ArgumentError("spatial population inputs are missing"))
    for slot in selection.slots
        expected = if slot.code in (14, 15, 27, 28, 29)
            populations === nothing &&
                throw(ArgumentError("spatial population inputs are missing"))
            length(populations.species_densities) * (slot.code == 14 ? nodes : 1)
        elseif slot.code == 21
            2
        elseif slot.code == 23
            n_frequency
        else
            1
        end
        slot.count == expected ||
            throw(DimensionMismatch("spatial statistic slice disagrees with its shape"))
    end
    for (code, profile) in (
            (11, profiles.density), (12, profiles.pressure), (13, profiles.core_radius),
        )
        _observation_selected(selection, code, false) && profile === nothing &&
            throw(ArgumentError("a requested spatial scalar profile is missing"))
    end
    if widths !== nothing
        widths isa SpatialWidthGrid ||
            throw(ArgumentError("spatial widths require prepared geometry"))
        correct = map isa RadialObservationMap ?
            widths.kind == :radial && length(widths.x) == nodes :
            widths.kind == (map.shape[2] == 1 ? :line : :cartesian) &&
            (length(widths.x), length(widths.y)) == map.shape
        correct || throw(DimensionMismatch("spatial width geometry and map disagree"))
    end
    needs.widths && widths === nothing &&
        throw(ArgumentError("selected spatial widths need their geometry"))
    if needs.axis_spectrum || (needs.widths && map isa RadialObservationMap)
        length(grid.spectral_density) == n_frequency && isfinite(grid.d_omega) ||
            throw(ArgumentError("spatial axis fluence needs spectral density and spacing"))
    end
    _spatial_observation_temporal(temporal, n_time, n_frequency, columns)
    owned_grid = StatisticsGrid(
        copy(grid.omega), copy(grid.time), copy(grid.parseval);
        windows = copy(grid.windows), spectral_density = copy(grid.spectral_density),
        d_omega = grid.d_omega,
    )
    owned_widths = widths === nothing ? nothing : SpatialWidthGrid(
            widths.kind, copy(widths.x), copy(widths.y), copy(widths.weights)
        )
    host_axis = Vector{ComplexF64}(on_axis)
    narrow_axis, axis_range = source_type(arithmetic) === Float64 ? (host_axis, nothing) :
        source_narrow(
            execution, host_axis, arithmetic, :spatial_observation_axis;
            exponent = arithmetic.state_exponent - arithmetic.field_exponent,
        )
    return SpatialObservationPreparation(
        kind, columns, owned_grid, selection, map, flux, Vector{Float64}(spectral_weights),
        narrow_axis, intensity_factor, owned_widths, profiles, populations,
        temporal, needs, statistics_needs(selection; periodic = true), arithmetic,
        host_axis, (; axis = axis_range),
    )
end

function _spatial_observation_populations(map, data, kind, frequencies, columns, nodes)
    data === nothing && return nothing
    expected = map isa RadialObservationMap ?
        data isa ObservationPopulations : data isa SpatialObservationPopulations
    expected && data.grid.kind == kind && length(data.grid.omega) == frequencies ||
        throw(DimensionMismatch("spatial population route and temporal grid disagree"))
    if data isa ObservationPopulations
        size(data.map.synthesis) == (nodes, columns) ||
            throw(DimensionMismatch("radial population synthesis has the wrong shape"))
    end
    return nothing
end

function _spatial_observation_temporal(temporal, n_time, n_frequency, columns)
    temporal === nothing && return nothing
    temporal isa TemporalObservationData &&
        size(temporal.weights.regions, 1) == n_time &&
        length(temporal.weights.columns) == columns &&
        (
        temporal.weights.buckets === nothing ||
            size(temporal.weights.buckets) == (n_frequency, columns)
    ) || throw(DimensionMismatch("spatial temporal monitor dimensions disagree"))
    buckets, limits = temporal.weights.buckets, temporal.policy.bucket_limits
    (
        (buckets === nothing && isempty(limits)) ||
            (buckets !== nothing && !isempty(limits) && maximum(buckets) <= length(limits))
    ) || throw(DimensionMismatch("spatial crossing buckets and limits disagree"))
    return nothing
end

struct _SpatialPhysicalPair{A, B, P}
    spectrum::A
    closure::A
    band::B
    pair::P
end

function _spatial_physical_pair(execution, data, analytic)
    frequencies, n_time = length(data.grid.omega), length(data.grid.time)
    nodes, _ = _spatial_map_shape(data.map)
    T = source_type(data.arithmetic)
    band = data.map isa CartesianObservationMap ?
        CartesianBandWork(
            execution, data.map.shape, frequencies; arithmetic = data.arithmetic,
        ) :
        nothing
    return _SpatialPhysicalPair(
        dzeros(execution, Complex{T}, frequencies, nodes),
        dzeros(execution, Complex{T}, frequencies, nodes),
        band, analytic ? _observation_pair(execution, n_time, nodes, T) : nothing,
    )
end

struct _SpatialAxisPair{A, P}
    spectrum::A
    closure::A
    pair::P
end

function _spatial_axis_pair(execution, data, analytic)
    frequencies, n_time = length(data.grid.omega), length(data.grid.time)
    T = source_type(data.arithmetic)
    return _SpatialAxisPair(
        dzeros(execution, Complex{T}, frequencies, 1),
        dzeros(execution, Complex{T}, frequencies, 1),
        analytic ? _observation_pair(execution, n_time, 1, T) : nothing,
    )
end

struct _SpatialPopulationWork{S, A, F, P, W, B, C, D, V}
    spectrum::S
    coarse::S
    nodes::A
    transform::F
    padding::P
    workspace::W
    band::B
    chunks::C
    densities::Vector{Float64}
    fractions::V
    peaks::V
    diagnostics::D
end

function _spatial_population_work(execution, data, node_chunk)
    prepared = data.populations
    prepared === nothing && return nothing
    if prepared isa ObservationPopulations
        return _observation_population_work(
            execution, prepared, data.n_modes, node_chunk;
            diagnostics = data.needs.plasma_diagnostics
        )
    end
    grid = prepared.grid
    T = source_type(data.arithmetic)
    capacity = min(node_chunk, data.n_modes)
    n_time, frequencies = grid.n_time, length(data.grid.omega)
    field = data.kind == :field
    fine = field ? n_time ÷ 2 + 1 : n_time
    nodes = dzeros(execution, field ? T : Complex{T}, n_time, capacity)
    spectrum = dzeros(execution, Complex{T}, fine, capacity)
    layout = ResponseLayout(n_time, 1, capacity)
    population = prepared.plasma === nothing ? nothing :
        PopulationWorkspace(execution, T, layout)
    chunks = map(1:capacity:data.n_modes) do first
        (first = first, count = min(capacity, data.n_modes - first + 1))
    end
    n_species = length(prepared.species_densities)
    return _SpatialPopulationWork(
        spectrum, dzeros(execution, Complex{T}, frequencies, capacity), nodes,
        TemporalFFT(execution, nodes, spectrum),
        SpectralPadding(
            execution, grid.crop_index, fine,
            source_coefficient(data.arithmetic, :population_padding, prepared.pad_scale),
            one(T),
        ),
        population, CartesianBandWork(
            execution, data.map.shape, frequencies; arithmetic = data.arithmetic,
        ), chunks, zeros(n_species), dzeros(execution, T, data.n_modes, n_species),
        dzeros(execution, T, data.n_modes, n_species),
        data.needs.plasma_diagnostics ? PlasmaDiagnosticWorkspace(execution, T, capacity) :
            nothing,
    )
end

_observation_arrays(work::_SpatialPhysicalPair) = (
    work.spectrum, work.closure,
    work.band === nothing ? () : (work.band.band,), _observation_arrays(work.pair),
)
_observation_arrays(work::_SpatialAxisPair) =
    (work.spectrum, work.closure, _observation_arrays(work.pair))
_observation_arrays(work::_SpatialPopulationWork) = (
    work.spectrum, work.coarse, work.nodes, work.band.band,
    work.padding.crop, work.padding.uncrop,
    work.densities, work.fractions, work.peaks, _observation_arrays(work.workspace),
    _observation_arrays(work.diagnostics),
)

_spatial_population_arrays(::Nothing) = ()
_spatial_population_arrays(work::_SpatialPopulationWork) = _observation_arrays(work)
_spatial_population_arrays(work::_ObservationPopulationWork) = (
    _observation_arrays(work), work.padding.crop, work.padding.uncrop,
)

_spatial_statistics_samples(::Nothing) = (nothing, nothing)
_spatial_statistics_samples(pair::_ObservationPair) = (pair.samples, pair.weighted)
_spatial_statistics_pair(::Nothing) = (nothing, nothing, nothing, nothing)
function _spatial_statistics_pair(work::Union{_SpatialPhysicalPair, _SpatialAxisPair})
    samples, weighted = _spatial_statistics_samples(work.pair)
    return (work.spectrum, work.closure, samples, weighted)
end

_spatial_statistics_input(::Nothing, ::Nothing, data) = nothing
function _spatial_statistics_input(physical, axis, data)
    spectrum, closure, analytic, analytic_closure = _spatial_statistics_pair(physical)
    axis_spectrum, axis_closure, axis_analytic, axis_analytic_closure =
        _spatial_statistics_pair(axis)
    return SpatialStatisticsInput(;
        spectrum, closure, analytic, analytic_closure,
        axis_spectrum, axis_closure, axis_analytic, axis_analytic_closure,
        factor = data.intensity_factor, widths = data.widths, source_columns = data.n_modes,
        field_exponent = data.arithmetic.field_exponent,
        closure_exponent = data.arithmetic.field_exponent,
    )
end

"""
    SpatialObservationWorkspace(execution, prepared; node_chunk,
        noise_spectrum=nothing, temporal_continuation=nothing, continuation_plane=false)

Allocate independent scalar spatial statistics, reconstruction and monitor storage.

`prepared` is `SpatialObservationPreparation`. The positive `node_chunk` bounds fine
population samples; optional noise and continuation match `ObservationWorkspace`.
Binary32 refuses input noise. `weight_range` and `closure_range` retain the most recent
host coefficient normalization facts, and their integer exponents restore the final
physical statistics. These are scalar facts, not extra full-volume storage.
Only requested coarse physical/axis pairs are allocated, each linear in the field
size. Cartesian reconstruction uses FFT plans, never a quadratic dense matrix. Every
retained numerical workspace array is listed in `buffers` for the caller's ledger.

The current field is supplied to each `observe!`; neither a source nor a propagation
state is retained. Warm evaluation allocates nothing and does not advance source or
monitor counters. Invalid data, unsupported execution and inconsistent continuation
raise before the workspace is returned.
"""
mutable struct SpatialObservationWorkspace{E, D, S, W, L, A, P, T, M, C, I, B} <:
    _ObservationWorkspace
    execution::E
    data::D
    spectrum::S
    closure::S
    weight::W
    material::Vector{ComplexF64}
    physical::L
    axis::A
    populations::P
    statistics::T
    monitor_work::M
    monitor::C
    inputs::I
    values::Vector{Float64}
    buffers::B
    valid::Bool
    z::Float64
    weight_exponent::Int
    closure_exponent::Int
    weight_range::SourceRange
    closure_range::SourceRange

    function SpatialObservationWorkspace(
            execution::E, data::D, spectrum::S, closure::S, weight::W,
            material::Vector{ComplexF64}, physical::L, axis::A, populations::P,
            statistics::T, monitor_work::M, monitor::C, inputs::I,
            values::Vector{Float64}, buffers::B, valid::Bool, z::Float64,
            weight_exponent::Int, closure_exponent::Int,
            weight_range::SourceRange, closure_range::SourceRange,
        ) where {E, D, S, W, L, A, P, T, M, C, I, B}
        return new{E, D, S, W, L, A, P, T, M, C, I, B}(
            execution, data, spectrum, closure, weight, material, physical, axis,
            populations, statistics, monitor_work, monitor, inputs, values, buffers,
            valid, z, weight_exponent, closure_exponent, weight_range, closure_range,
        )
    end
end

function SpatialObservationWorkspace(
        execution::KernelExecution, prepared::SpatialObservationPreparation;
        node_chunk::Int, noise_spectrum = nothing, temporal_continuation = nothing,
        continuation_plane::Bool = false,
    )
    oncpu(execution) && node_chunk > 0 ||
        throw(ArgumentError("spatial observations need CPU and a positive node chunk"))
    frequencies, n_time = length(prepared.grid.omega), length(prepared.grid.time)
    nodes, columns = _spatial_map_shape(prepared.map)
    T = source_type(prepared.arithmetic)
    T === Float32 && noise_spectrum !== nothing &&
        throw(ArgumentError("binary32 spatial observations do not support input noise"))
    selection, needs = prepared.selection, prepared.needs
    local_peak = _observation_selected(selection, 18, false)
    axis_peak = _observation_selected(selection, 19, false)
    physical = needs.widths || local_peak ?
        _spatial_physical_pair(execution, prepared, local_peak) : nothing
    axis = needs.widths || needs.axis_spectrum || axis_peak ?
        _spatial_axis_pair(execution, prepared, axis_peak) : nothing
    spectrum = dzeros(execution, Complex{T}, frequencies, columns)
    closure = physical !== nothing || axis !== nothing ? similar(spectrum) :
        dzeros(execution, Complex{T}, 0, 0)
    weight = needs.spectral ? dzeros(execution, T, frequencies, columns) :
        dzeros(execution, T, 0, 0)
    material = prepared.flux.material === nothing ?
        ComplexF64[] : zeros(ComplexF64, frequencies)
    populations = needs.populations ?
        _spatial_population_work(execution, prepared, node_chunk) : nothing
    monitor_work = _observation_monitor_work(execution, prepared, noise_spectrum)
    monitor = monitor_work === nothing ? nothing : TemporalMonitorState()
    if temporal_continuation !== nothing
        monitor === nothing && throw(ArgumentError("monitor continuation needs a monitor"))
        restore_temporal!(monitor, temporal_continuation; plane = continuation_plane)
    end
    radii = prepared.widths === nothing ? 0 : prepared.widths.radius_capacity
    statistics = StatisticsWorkspace(
        execution, T; n_time, n_columns = columns, n_nodes = nodes, n_radii = radii,
    )
    spectral_input = needs.spectral ?
        SpectralStatisticsInput(
            spectrum, weight; total = true,
            amplitude_exponent = prepared.arithmetic.state_exponent,
        ) : nothing
    spatial_input = _spatial_statistics_input(physical, axis, prepared)
    population_input = if populations === nothing
        nothing
    else
        prepared.populations === nothing &&
            throw(ArgumentError("spatial population data are missing"))
        PopulationStatisticsInput(
            populations.densities, populations.fractions, populations.peaks;
            peak_species = prepared.populations.peak_species,
            potentials = prepared.populations.potentials,
            node_weights = prepared.populations.measure === nothing ? nothing :
                prepared.populations.measure.weights,
        )
    end
    inputs = StatisticsInputs(;
        spectral = spectral_input, spatial = spatial_input, populations = population_input,
    )
    values = zeros(selection.width)
    buffers = (
        spectrum, closure, weight, material, values, _observation_arrays(physical),
        _observation_arrays(axis), _spatial_population_arrays(populations),
        _observation_arrays(monitor_work), statistics.power, statistics.total_power,
        statistics.column_energy, statistics.column_moment, statistics.node_total,
        statistics.fluence, statistics.magnitude, statistics.radial_average,
        _observation_scale_arrays(statistics.scales),
    )
    return SpatialObservationWorkspace(
        execution, prepared, spectrum, closure, weight, material, physical, axis,
        populations, statistics, monitor_work, monitor, inputs, values, buffers, false, NaN,
        0, 0, SourceRange(:spatial_power_weight, 0, 0, true, 0.0, 0.0),
        SourceRange(:spatial_admittance, 0, 0, true, 0.0, 0.0),
    )
end

function ObservationWorkspace(
        execution::KernelExecution, prepared::SpatialObservationPreparation; kwargs...,
    )
    return SpatialObservationWorkspace(execution, prepared; kwargs...)
end

function _observation_field(work::SpatialObservationWorkspace, field, z)
    T = source_type(work.data.arithmetic)
    field isa StridedMatrix{Complex{T}} && size(field) == size(work.spectrum) ||
        throw(DimensionMismatch("observation field must match its prepared spatial state"))
    isfinite(z) && all(isfinite, field) ||
        throw(ArgumentError("observation position and field must be finite"))
    _retained_alias(field, work) &&
        throw(ArgumentError("observation field aliases its retained inputs or workspace"))
    return nothing
end

function _spatial_observation_flux!(work, position, spectral, closure)
    data, execution = work.data, work.execution
    flux, base = data.flux, data.flux.base
    high = if flux.material === nothing
        base.k_squared
    else
        evaluate_table!(work.material, flux.material, position, execution)
        all(isfinite, work.material) ||
            throw(ArgumentError("spatial material square is nonfinite at observation"))
        work.material
    end
    if source_type(data.arithmetic) === Float32
        _spatial_observation_flux_single!(work, high, spectral, closure)
        return nothing
    end
    low, transverse, residual = base.k_squared_low, base.kperp_sq, base.kperp_sq_low
    active, omega, speed = base.active, data.grid.omega, flux.light_speed
    input, partner, weights = work.spectrum, work.closure, work.weight
    measure, factor = data.spectral_weights, data.intensity_factor
    varying = flux.material !== nothing
    each!(execution, size(input)...) do row, column
        material = varying && !active[row] ? 0.0im : high[row]
        square = (material - transverse[column]) + (low[row] - residual[column])
        admittance = active[row] && omega[row] > 0 && real(square) > 0 ?
            speed * sqrt(square) / omega[row] : 0.0im
        spectral && (weights[row, column] = factor * measure[column] * real(admittance))
        closure && (partner[row, column] = conj(admittance) * input[row, column])
    end
    (!spectral || all(isfinite, weights)) && (!closure || all(isfinite, partner)) ||
        throw(ArgumentError("spatial observation admittance or weighted field overflowed"))
    return nothing
end

# Evaluate immutable host coefficients in binary64 at z, then round once into the
# normalized binary32 pass. No state-sized binary64 temporary or sample product is
# formed, and the original binary64 graph above remains untouched.
@inline function _spatial_observation_admittance(inputs, row, column)::ComplexF64
    material = inputs.varying && !inputs.active[row] ? 0.0im : inputs.high[row]
    square = (material - inputs.transverse[column]) +
        (inputs.low[row] - inputs.residual[column])
    return inputs.active[row] && inputs.omega[row] > 0 && real(square) > 0 ?
        inputs.speed * sqrt(square) / inputs.omega[row] : 0.0im
end

function _spatial_observation_range(name::Symbol, largest::Float64, smallest::Float64)
    exponent = iszero(largest) ? 0 : last(frexp(largest)) - 1
    return SourceRange(
        name, exponent, 0, true, ldexp(largest, -exponent),
        isinf(smallest) ? 0.0 : ldexp(smallest, -exponent),
    )
end

function _spatial_observation_flux_single!(
        work::SpatialObservationWorkspace, high::AbstractVector{ComplexF64},
        spectral::Bool, closure::Bool,
    )
    data, input = work.data, work.spectrum
    base = data.flux.base
    coefficients = (
        high = high, low = base.k_squared_low, transverse = base.kperp_sq,
        residual = base.kperp_sq_low, active = base.active, omega = data.grid.omega,
        speed = data.flux.light_speed, varying = data.flux.material !== nothing,
    )
    factor, measure = data.intensity_factor, data.spectral_weights
    largest_weight, largest_partner = 0.0, 0.0
    smallest_weight = smallest_partner = Inf
    for column in axes(input, 2), row in axes(input, 1)
        admittance = _spatial_observation_admittance(coefficients, row, column)
        isfinite(admittance) ||
            throw(ArgumentError("spatial observation admittance is nonfinite"))
        if spectral
            value = factor * measure[column] * real(admittance)
            isfinite(value) && value >= 0 ||
                throw(ArgumentError("spatial observation power weight is invalid"))
            largest_weight = max(largest_weight, value)
            iszero(value) || (smallest_weight = min(smallest_weight, value))
        end
        if closure
            value = abs(admittance)
            isfinite(value) ||
                throw(ArgumentError("spatial observation admittance magnitude overflowed"))
            largest_partner = max(largest_partner, value)
            iszero(value) || (smallest_partner = min(smallest_partner, value))
        end
    end
    weight_range = _spatial_observation_range(
        :spatial_power_weight, largest_weight, smallest_weight,
    )
    closure_range = _spatial_observation_range(
        :spatial_admittance, largest_partner, smallest_partner,
    )
    weight_exponent, closure_exponent = weight_range.block_exponent,
        closure_range.block_exponent
    partner, weights = work.closure, work.weight
    each!(work.execution, size(input)...) do row, column
        admittance = _spatial_observation_admittance(coefficients, row, column)
        if spectral
            value = factor * measure[column] * real(admittance)
            weights[row, column] = _observation_scaled(Float64(value), weight_exponent)
        end
        if closure
            coefficient = ComplexF32(_source_scaled(conj(admittance), -closure_exponent))
            partner[row, column] = coefficient * input[row, column]
        end
    end
    (!spectral || all(isfinite, weights)) && (!closure || all(isfinite, partner)) ||
        throw(ArgumentError("spatial observation normalized partner overflowed"))
    work.weight_exponent, work.closure_exponent = weight_exponent, closure_exponent
    work.weight_range, work.closure_range = weight_range, closure_range
    return nothing
end

function _spatial_reconstruct!(work, field, closure, map::RadialObservationMap, execution)
    synthesize!(work.spectrum, field, map.synthesis, execution; n_components = 1)
    synthesize!(work.closure, closure, map.synthesis, execution; n_components = 1)
    return nothing
end

function _spatial_reconstruct!(
        work, field, closure, map::CartesianObservationMap, execution,
    )
    copyto!(work.spectrum, cartesian_inverse!(work.band, field))
    copyto!(work.closure, cartesian_inverse!(work.band, closure))
    if source_type(map.arithmetic) === Float32
        first, second, scale = work.spectrum, work.closure, map.scale
        each!(execution, size(first)...) do row, column
            first[row, column] *= scale
            second[row, column] *= scale
        end
    end
    return nothing
end

function _spatial_analytic!(pair, first, second, kind, execution)
    spectrum, closure = pair.spectrum, pair.weighted_spectrum
    frequencies = size(first, 1)
    field = kind == :field
    T = real(eltype(spectrum))
    blank, real_zero, scale = zero(eltype(spectrum)), zero(T), field ? T(2) : one(T)
    each!(execution, size(spectrum)...) do row, column
        if row > frequencies
            spectrum[row, column], closure[row, column] = blank, blank
        elseif field && (row == 1 || row == frequencies)
            spectrum[row, column] = complex(real(first[row, column]), real_zero)
            closure[row, column] = complex(real(second[row, column]), real_zero)
        else
            spectrum[row, column] = scale * first[row, column]
            closure[row, column] = scale * second[row, column]
        end
    end
    temporal_inverse!(pair.samples, pair.transform, spectrum, execution)
    temporal_inverse!(pair.weighted, pair.weighted_transform, closure, execution)
    return nothing
end

function _spatial_axis!(work, field, closure, axis, execution)
    first, second = work.spectrum, work.closure
    each!(execution, size(first, 1), 1) do row, _
        electric = magnetic = zero(eltype(first))
        for column in axes(field, 2)
            electric += axis[column] * field[row, column]
            magnetic += axis[column] * closure[row, column]
        end
        first[row, 1], second[row, 1] = electric, magnetic
    end
    return nothing
end

function _spatial_observation_populations!(
        work, data::ObservationPopulations, field, z, ex;
        diagnostics::Bool = false
    )
    _observation_populations!(work, data, field, z, ex; diagnostics)
    return nothing
end

function _spatial_observation_populations!(
        work, data::SpatialObservationPopulations, field, position, execution;
        diagnostics::Bool = false,
    )
    diagnostics && reset_plasma_diagnostics!(work.diagnostics)
    _observation_densities!(work.densities, data.species_densities, position, 1)
    fill!(work.fractions, 0.0)
    fill!(work.peaks, 0.0)
    data.plasma === nothing && return nothing
    physical = cartesian_inverse!(work.band, field)
    coarse = work.coarse
    arithmetic = data.grid.arithmetic
    scale = source_coefficient(
        arithmetic, :spatial_population_synthesis, 1.0;
        exponent = arithmetic.state_exponent - arithmetic.field_exponent,
    )
    for chunk in work.chunks
        first, count = chunk.first, chunk.count
        each!(execution, size(coarse)...) do row, column
            coarse[row, column] = column <= count ?
                physical[row, first + column - 1] : zero(eltype(coarse))
            if source_type(arithmetic) === Float32
                coarse[row, column] *= scale
            end
        end
        pad_spectrum!(work.spectrum, coarse, work.padding, execution)
        temporal_inverse!(work.nodes, work.transform, work.spectrum, execution)
        all(isfinite, work.nodes) ||
            throw(ArgumentError("spatial population reconstruction overflowed"))
        _source_each_indexed(data.plasma.channels) do channel, index
            _observation_channel!(
                work, data, channel, index, chunk, position, execution;
                diagnostics
            )
        end
    end
    return nothing
end

_spatial_observation_input(::Nothing, exponent) = nothing
function _spatial_observation_input(input::SpatialStatisticsInput, exponent)
    return SpatialStatisticsInput(;
        spectrum = input.spectrum, closure = input.closure, analytic = input.analytic,
        analytic_closure = input.analytic_closure, axis_spectrum = input.axis_spectrum,
        axis_closure = input.axis_closure, axis_analytic = input.axis_analytic,
        axis_analytic_closure = input.axis_analytic_closure, factor = input.factor,
        widths = input.widths, source_columns = input.source_columns,
        field_exponent = input.field_exponent,
        closure_exponent = input.field_exponent + exponent,
    )
end

"""
    observe!(work::SpatialObservationWorkspace, field, z;
             periodic=false, quadrature_error=NaN)

Reconstruct fresh scalar spatial statistics from this actual observed field.

The disjoint finite complex `field` has `(n_omega,n_modes)` shape and the preparation's
stored arithmetic/scale, and is preserved. `z` is metres within every selected represented
profile's span. Periodic evaluation selects exactly the prepared periodic slots and clears
other slices.
`quadrature_error` is the caller's accepted-step alarm; no RHS is reevaluated here.

Evaluate physical axial admittance at z, construct its conjugated magnetic partner,
reconstruct only requested physical/axis pairs, and scan requested populations afresh.
Reuse the existing signed-flux statistics and width refusal policy. Only successful
completion writes `valid=true` and the observed `z`; source, monitor and controller
counters are untouched. Return `nothing`; malformed, aliased or nonfinite data and
failed plasma scans raise without publishing an observation.
"""
function observe!(
        work::SpatialObservationWorkspace, field, z;
        periodic::Bool = false, quadrature_error = NaN,
    )
    _observation_field(work, field, z)
    work.valid = false
    data, execution = work.data, work.execution
    needs = periodic ? data.periodic_needs : data.needs
    local_peak = _observation_selected(data.selection, 18, periodic)
    axis_peak = _observation_selected(data.selection, 19, periodic)
    physical = needs.widths || local_peak
    axis = needs.widths || needs.axis_spectrum || axis_peak
    if needs.spectral || physical || axis
        copyto!(work.spectrum, field)
        _spatial_observation_flux!(work, z, needs.spectral, physical || axis)
    end
    if physical
        work.physical === nothing &&
            throw(ArgumentError("physical observation work missing"))
        _spatial_reconstruct!(
            work.physical, work.spectrum, work.closure, data.map, execution,
        )
        local_peak && _spatial_analytic!(
            work.physical.pair, work.physical.spectrum, work.physical.closure,
            data.kind, execution,
        )
    end
    if axis
        work.axis === nothing && throw(ArgumentError("axis observation work missing"))
        _spatial_axis!(work.axis, work.spectrum, work.closure, data.on_axis, execution)
        axis_peak && _spatial_analytic!(
            work.axis.pair, work.axis.spectrum, work.axis.closure, data.kind, execution,
        )
    end
    if needs.populations
        work.populations === nothing &&
            throw(ArgumentError("spatial population buffers are missing"))
        data.populations === nothing &&
            throw(ArgumentError("spatial population preparation is missing"))
        _spatial_observation_populations!(
            work.populations, data.populations, field, z, execution;
            diagnostics = needs.plasma_diagnostics,
        )
    end
    profiles = (
        _observation_profile(
            data.profiles.density, _observation_selected(data.selection, 11, periodic), z,
        ),
        _observation_profile(
            data.profiles.pressure, _observation_selected(data.selection, 12, periodic), z,
        ),
        _observation_profile(
            data.profiles.core_radius,
            _observation_selected(data.selection, 13, periodic), z,
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
        spectral = _observation_spectral(
            work.inputs.spectral, false, work.weight_exponent,
        ),
        spatial = _spatial_observation_input(work.inputs.spatial, work.closure_exponent),
        populations, profiles, quadrature_error, diagnostics,
    )
    statistics!(
        work.values, data.selection, data.grid, inputs, work.statistics, execution;
        periodic,
    )
    work.z, work.valid = z, true
    return nothing
end
