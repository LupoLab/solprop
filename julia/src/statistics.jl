# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public StatisticSlot, StatisticsSelection, StatisticsNeeds, statistics_needs
public StatisticsGrid, StatisticsInputs, StatisticsWorkspace, statistics!
public SpectralStatisticsInput, TemporalStatisticsInput, PopulationStatisticsInput
public SpatialStatisticsInput, SpatialWidthGrid
public spectral_statistics!, temporal_statistics!, population_statistics!
public spatial_statistics!, half_width

const _STATISTIC_NAMES = (
    :omega0, :energy, :energy_window, :peak_power, :peak_power_window,
    :peak_power_total, :physical_energy, :analytic_flux, :fwhm_t, :fwhm_t_outer,
    :density, :pressure, :core_radius, :species_ionisation,
    :species_electron_density, :electron_density, :peak_ionisation_rate,
    :peak_intensity, :on_axis_intensity, :quadrature_error,
    :beam_radius, :fwhm_r, :on_axis_spectrum,
    :plasma_undershoot, :plasma_work_defect, :plasma_boundary_level,
    :species_density, :species_ionisation_energy, :species_peak_ionisation_rate,
)

"""
    StatisticSlot(name, offset, count)

Describe one statistic's contiguous slice in a caller-owned packed output vector.

`name` is an admitted catalogue symbol, `offset` is one-based and `count` is positive.
Shapes and public names remain in the bridge's immutable output metadata. Spatial
widths/spectrum, plasma integration diagnostics and species inventory/peak rows
remain plane-only. Unknown and far-field statistics and the
plane-only mode-model root `zdw` raise `ArgumentError` by name.
"""
struct StatisticSlot
    code::UInt8
    offset::Int
    count::Int

    function StatisticSlot(name::Symbol, offset::Integer, count::Integer)
        code = findfirst(==(name), _STATISTIC_NAMES)
        code === nothing && throw(ArgumentError("Julia does not admit statistic $name"))
        0 < offset <= typemax(Int) && 0 < count <= typemax(Int) &&
            offset - 1 <= typemax(Int) - count ||
            throw(ArgumentError("statistic slices must be positive and fit Int"))
        return new(UInt8(code), Int(offset), Int(count))
    end
end

"""
    StatisticsSelection(slots, periodic=())

Retain concrete tuples of packed plane slots and periodic slot indices.

Slots must be contiguous from offset one, with no duplicate names. Periodic indices
are unique, one-based, and may select only periodic-capable names. Preparation errors
raise `ArgumentError`. `width` is the complete packed plane width; periodic evaluation
writes its selected slices at the same offsets and clears all other slices.
"""
struct StatisticsSelection{S <: Tuple, P <: Tuple}
    slots::S
    periodic::P
    width::Int
end

function StatisticsSelection(slots::Tuple, periodic::Tuple = ())
    all(slot -> slot isa StatisticSlot, slots) ||
        throw(ArgumentError("statistics slots must be StatisticSlot records"))
    offset = 1
    for (index, slot) in pairs(slots)
        slot.offset == offset || throw(ArgumentError("statistics slots must be contiguous"))
        for previous in 1:(index - 1)
            slots[previous].code == slot.code &&
                throw(ArgumentError("duplicate statistic in selection"))
        end
        slot.count <= typemax(Int) - offset ||
            throw(ArgumentError("statistics packed width is too large"))
        offset += slot.count
    end
    all(index -> index isa Int && 1 <= index <= length(slots), periodic) ||
        throw(ArgumentError("periodic statistic index is out of range"))
    length(unique(periodic)) == length(periodic) ||
        throw(ArgumentError("duplicate periodic statistic index"))
    all(index -> slots[index].code <= 20, periodic) ||
        throw(ArgumentError("selected statistics are plane-only"))
    return StatisticsSelection(slots, periodic, offset - 1)
end

"""
    StatisticsNeeds(spectral, temporal, windows, populations, intensity, widths,
                    axis_spectrum, plasma_diagnostics=false)

Describe numerical families required to prepare one observation's inputs.

Every field is a Boolean. `windows` requests windowed analytic pairs; `intensity`
requests physical or on-axis analytic pairs, and `widths` requests local and axial
spectral pairs. `plasma_diagnostics` requests the plane-only raw field-population
scan parts as well as final populations. This record carries no state or callback.
"""
struct StatisticsNeeds
    spectral::Bool
    temporal::Bool
    windows::Bool
    populations::Bool
    intensity::Bool
    widths::Bool
    axis_spectrum::Bool
    plasma_diagnostics::Bool
end

StatisticsNeeds(spectral, temporal, windows, populations, intensity, widths, axis) =
    StatisticsNeeds(
    spectral, temporal, windows, populations, intensity, widths, axis,
    false
)

"""
    statistics_needs(selection; periodic=false)

Return the numerical input families required by the selected observation slots.

Periodic evaluation uses only the indices recorded at preparation; it does not infer
cadence or advance any cursor. The result is an allocation-free `StatisticsNeeds`.
"""
function statistics_needs(selection::StatisticsSelection; periodic::Bool = false)
    spectral = temporal = windows = populations = intensity = widths = axis = false
    diagnostics = false
    for (index, slot) in pairs(selection.slots)
        periodic && !(index in selection.periodic) && continue
        code = slot.code
        spectral |= code in (1, 2, 3, 7)
        temporal |= code in (4, 6, 8, 9, 10)
        windows |= code == 5
        populations |= 14 <= code <= 17 || 24 <= code <= 29
        intensity |= code in (18, 19)
        widths |= code in (21, 22)
        axis |= code == 23
        diagnostics |= 24 <= code <= 26
    end
    return StatisticsNeeds(
        spectral, temporal, windows, populations, intensity, widths, axis,
        diagnostics
    )
end

"""
    StatisticsGrid(omega, time, parseval; windows, spectral_density, d_omega=NaN)

Retain explicit prepared axes and Fourier weights for scalar statistics.

`omega` is a finite vector in rad/s; `time` is finite and strictly increasing in
seconds. Nonnegative finite `parseval` has one weight per frequency bin, including
`dt/n_t` and the field grid's endpoint/interior factors `(1,4,1)`. The centroid uses
unmodified spectral density, not these factors. `windows` is a dense Boolean matrix
`(n_omega,n_windows)` of half-open gates. `spectral_density` contains the separate
per-angular-frequency weights used by the on-axis spectrum, in seconds squared.
`d_omega` is the positive frequency-bin spacing in rad/s when radial widths need the
independently reconstructed axial fluence.
All arrays are caller-owned immutable preparation data; malformed data raise
`ArgumentError` or `DimensionMismatch`. Missing optional arrays have zero columns/size.
"""
struct StatisticsGrid{O, X, P, W, D, T}
    omega::O
    time::X
    parseval::P
    windows::W
    spectral_density::D
    d_omega::T
end

function StatisticsGrid(
        omega::AbstractVector, time::AbstractVector, parseval::AbstractVector;
        windows = fill(false, length(omega), 0), spectral_density = Float64[],
        d_omega = NaN,
    )
    !isempty(omega) && !isempty(time) && all(isfinite, omega) && all(isfinite, time) ||
        throw(ArgumentError("statistics axes must be finite and nonempty"))
    all(index -> time[index] > time[index - 1], 2:length(time)) ||
        throw(ArgumentError("statistics time axis must increase"))
    length(parseval) == length(omega) && size(windows, 1) == length(omega) ||
        throw(DimensionMismatch("statistics spectral weights do not match the axis"))
    all(value -> isfinite(value) && value >= 0, parseval) ||
        throw(ArgumentError("Parseval weights must be finite and nonnegative"))
    windows isa AbstractMatrix && eltype(windows) === Bool && !(windows isa BitArray) ||
        throw(ArgumentError("statistics windows require unpacked Boolean storage"))
    isempty(spectral_density) || length(spectral_density) == length(omega) ||
        throw(DimensionMismatch("spectral-density weights do not match the axis"))
    all(value -> isfinite(value) && value >= 0, spectral_density) ||
        throw(ArgumentError("spectral-density weights must be finite and nonnegative"))
    isnan(d_omega) || (isfinite(d_omega) && d_omega > 0) ||
        throw(ArgumentError("frequency-bin spacing must be positive or absent"))
    return StatisticsGrid(omega, time, parseval, windows, spectral_density, d_omega)
end

"""
    SpectralStatisticsInput(spectrum, power_weight; total=false)

Retain observed `(n_omega,n_columns)` complex spectra and matching real power weights.

For modal data the weight is the plane's `Re(n_eff)` and outputs remain per column.
For spatial energy, `total=true` sums columns and the weight already contains the
physical axial admittance, transverse measure and `epsilon_0*c/2`. No Fourier weight
is folded into it. The caller prepares data from the actual observed field in Julia.
"""
struct SpectralStatisticsInput{S, W}
    spectrum::S
    power_weight::W
    total::Bool
end
SpectralStatisticsInput(spectrum, weight; total::Bool = false) =
    SpectralStatisticsInput(spectrum, weight, total)

"""
    TemporalStatisticsInput(analytic, weighted; window_analytic=nothing,
                            window_weighted=nothing, axis_analytic=nothing,
                            axis_weighted=nothing, intensity_factor=1.0,
                            identity_weight=false)

Retain prepared analytic field pairs for temporal power and modal axial intensity.

The primary matrices `(n_t,n_columns)` are `analytic(a)` and
`analytic(a*power_weight)`, so power is their real Hermitian pairing, retaining signed
broadband excursions. Optional window pairs have shape `(n_t,n_columns,n_windows)`.
Optional axis pairs have shape `(n_t,n_axis_components)`; `intensity_factor` converts
that pairing to W/m² and is the translated `epsilon_0*c/2` for physical V/m samples.
No square root of the frequency-dependent weight replaces the magnetic member.
`identity_weight` is the preparation-time identity-metric admission result; only a true
value admits the `analytic_flux` alias. The conversion factor must be positive finite.
"""
struct TemporalStatisticsInput{A, B, W, V, X, Y, T}
    analytic::A
    weighted::B
    window_analytic::W
    window_weighted::V
    axis_analytic::X
    axis_weighted::Y
    intensity_factor::T
    identity_weight::Bool
end
function TemporalStatisticsInput(
        analytic, weighted; window_analytic = nothing, window_weighted = nothing,
        axis_analytic = nothing, axis_weighted = nothing, intensity_factor = 1.0,
        identity_weight::Bool = false,
    )
    isfinite(intensity_factor) && intensity_factor > 0 ||
        throw(ArgumentError("intensity conversion must be finite and positive"))
    return TemporalStatisticsInput(
        analytic, weighted, window_analytic, window_weighted,
        axis_analytic, axis_weighted, intensity_factor, identity_weight,
    )
end

"""
    PopulationStatisticsInput(densities, fractions, peak_rates; peak_species=0,
                              potentials=nothing, node_weights=nothing,
                              measure_factor=1.0)

Retain fresh scalar population reductions in the mixture's complete species order.

`densities` is a finite nonnegative vector in m⁻³. `fractions` and `peak_rates` have
shape `(n_nodes,n_species)`, respectively dimensionless in `[0,1]` and in s⁻¹.
Non-ionising species have zero columns. With `peak_species=0`, the reported peak rate
is the largest over all nodes/species. A positive one-based species index selects
its first largest-fraction node, preserving the legacy single-population statistic,
including zero-density cases. Electron densities always use the first largest total
number-density node. Which peak rule applies is resolved from the neutral request.

Optional `potentials` has one finite nonnegative ionisation energy in J per species;
`node_weights` has finite nonnegative physical transverse weights in m². The positive
finite `measure_factor` rescales those weights at this observation. These data are
required only by `species_ionisation_energy`, the final inventory in J/m including
preionisation, rather than the newly deposited work.
"""
struct PopulationStatisticsInput{D, F, P, U, W}
    densities::D
    fractions::F
    peak_rates::P
    peak_species::Int
    potentials::U
    node_weights::W
    measure_factor::Float64
end
PopulationStatisticsInput(densities, fractions, peaks, peak_species::Int) =
    PopulationStatisticsInput(
    densities, fractions, peaks, peak_species, nothing, nothing, 1.0
)
PopulationStatisticsInput(
    densities, fractions, peaks; peak_species::Int = 0,
    potentials = nothing, node_weights = nothing, measure_factor::Float64 = 1.0
) =
    PopulationStatisticsInput(
    densities, fractions, peaks, peak_species,
    potentials, node_weights, measure_factor
)

"""
    SpatialWidthGrid(kind, x, y, weights)

Prepare radial, Cartesian or line fluence-width geometry and fixed angular samples.

`kind` is `:radial`, `:cartesian` or `:line`. A radial grid uses positive increasing
radii `x` and positive integration `weights` in m²; `y` is empty. Cartesian axes are
uniform increasing coordinates in metres, x fastest; a line has exactly one y sample.
Threshold radii use half the finest pixel spacing and a grid-fixed angular rule, as
`solprop.spatial.cartesian_fluence_widths` defines. Caller-owned axes/weights and the
owned sine/cosine arrays are explicit retained storage. Invalid geometry raises
`ArgumentError`. `radius_capacity` bounds caller-owned radial-average scratch.
"""
struct SpatialWidthGrid{X, Y, W, A, T}
    kind::Symbol
    x::X
    y::Y
    weights::W
    cosines::A
    sines::A
    spacing::T
    radius_capacity::Int
end

function SpatialWidthGrid(kind::Symbol, x, y, weights)
    kind in (:radial, :cartesian, :line) ||
        throw(ArgumentError("unsupported scalar spatial width geometry"))
    length(x) >= 2 && all(isfinite, x) && all(isfinite, y) &&
        all(index -> x[index] > x[index - 1], 2:length(x)) ||
        throw(ArgumentError("spatial width axes must be finite and increasing"))
    if kind == :radial
        isempty(y) && length(weights) == length(x) && x[1] > 0 &&
            all(value -> isfinite(value) && value > 0, weights) ||
            throw(ArgumentError("radial widths require positive radii and quadrature"))
        return SpatialWidthGrid(kind, x, y, weights, Float64[], Float64[], 0.0, length(x))
    end
    (kind == :line ? length(y) == 1 : length(y) >= 2) ||
        throw(ArgumentError("line/Cartesian width dimensions disagree"))
    for axis in (kind == :line ? (x,) : (x, y))
        step = (axis[end] - axis[1]) / (length(axis) - 1)
        tolerance = 32eps(Float64) * max(maximum(abs, axis), step)
        step > 0 && all(
            index -> abs(axis[index] - axis[index - 1] - step) <= tolerance,
            2:length(axis)
        ) || throw(ArgumentError("Cartesian width axes must be uniform"))
    end
    spacing = kind == :line ? (x[2] - x[1]) / 2 :
        min(x[2] - x[1], y[2] - y[1]) / 2
    largest = kind == :line ? (x[end] - x[1]) / 2 :
        min(x[end] - x[1], y[end] - y[1]) / 2
    angles = kind == :line ? 0 : max(8, ceil(Int, 2pi * largest / spacing))
    cosines = [cos(2pi * index / angles) for index in 0:(angles - 1)]
    sines = [sin(2pi * index / angles) for index in 0:(angles - 1)]
    return SpatialWidthGrid(
        kind, x, y, weights, cosines, sines, spacing, floor(Int, largest / spacing) + 1
    )
end

"""
    SpatialStatisticsInput(; spectrum=nothing, closure=nothing, analytic=nothing,
        analytic_closure=nothing, axis_spectrum=nothing, axis_closure=nothing,
        axis_analytic=nothing, axis_analytic_closure=nothing, factor,
        widths=nothing, source_columns)

Retain Julia-prepared electric/magnetic pairs for scalar spatial diagnostics.

Spectral matrices have `(n_omega,n_nodes)` shape and analytic pairs `(n_t,n_nodes)`;
axis pairs have one column per physical component. The spectra use V/m and the
closure's matching convention. `factor` is the translated `epsilon_0*c/2`; all physical
conversion belongs to this explicit factor. `widths` is `SpatialWidthGrid` when needed.
`source_columns` counts the columns summed by spatial synthesis and sets the existing
fluence rounding budget `8*(n_omega+source_columns)*eps(Float64)*max(magnitude)`.
Preparation may omit unrequested pairs with `nothing`.
"""
struct SpatialStatisticsInput{S, C, A, B, X, Y, U, V, T, W}
    spectrum::S
    closure::C
    analytic::A
    analytic_closure::B
    axis_spectrum::X
    axis_closure::Y
    axis_analytic::U
    axis_analytic_closure::V
    factor::T
    widths::W
    source_columns::Int
end
function SpatialStatisticsInput(;
        spectrum = nothing, closure = nothing, analytic = nothing,
        analytic_closure = nothing, axis_spectrum = nothing, axis_closure = nothing,
        axis_analytic = nothing, axis_analytic_closure = nothing, factor,
        widths = nothing, source_columns::Int,
    )
    isfinite(factor) && factor > 0 && source_columns > 0 ||
        throw(ArgumentError("spatial statistics need positive conversion and column count"))
    return SpatialStatisticsInput(
        spectrum, closure, analytic, analytic_closure, axis_spectrum, axis_closure,
        axis_analytic, axis_analytic_closure, factor, widths, source_columns,
    )
end

"""
    StatisticsInputs(; spectral=nothing, temporal=nothing, populations=nothing,
                     spatial=nothing, profiles=(NaN,NaN,NaN), quadrature_error=NaN,
                     diagnostics=nothing)

Group explicit inputs for one actual observed field, with no mutable field cache.

The optional numerical families use the corresponding `*StatisticsInput` records.
`profiles` is `(density, pressure, core_radius)` in m⁻³, catalogue bar, and metres,
evaluated from each profile's own Julia table. `quadrature_error` is the session's
accepted-step alarm; statistics do not rescore the last trial. It is NaN before an
accepted score, without an embedded rule or for a zero material source, as in the
reference backend's `_update_alarm`. Optional `diagnostics` is the finite nonnegative
Float64 tuple `(plasma_undershoot, plasma_work_defect, plasma_boundary_level)` from
this observed field's complete scan. Other requested absent or nonfinite dependencies
raise an error rather than being replaced by a default.
"""
struct StatisticsInputs{S, T, P, G, F, Q, D}
    spectral::S
    temporal::T
    populations::P
    spatial::G
    profiles::F
    quadrature_error::Q
    diagnostics::D
end

StatisticsInputs(spectral, temporal, populations, spatial, profiles, quadrature_error) =
    StatisticsInputs(
    spectral, temporal, populations, spatial, profiles, quadrature_error,
    nothing
)
StatisticsInputs(;
    spectral = nothing, temporal = nothing, populations = nothing, spatial = nothing,
    profiles = (NaN, NaN, NaN), quadrature_error = NaN,
    diagnostics = nothing,
) = StatisticsInputs(
    spectral, temporal, populations, spatial, profiles, quadrature_error,
    diagnostics
)

"""
    StatisticsWorkspace(execution, T; n_time, n_columns, n_nodes=0, n_radii=0)

Allocate independently owned buffers for scalar statistics at preparation.

`power` is `(n_time,n_columns)`, `total_power` has `n_time` entries, `column_energy`
and `column_moment` have `n_columns`, and `node_total`, `fluence`, `magnitude` have
`n_nodes`. `radial_average` has `n_radii`. These are the complete retained workspace
arrays; output and all input arrays are caller-owned and must be disjoint. Counts
must be nonnegative, with positive temporal/column dimensions.
"""
struct StatisticsWorkspace{A, V}
    power::A
    total_power::V
    column_energy::V
    column_moment::V
    node_total::V
    fluence::V
    magnitude::V
    radial_average::V
end
function StatisticsWorkspace(
        execution, ::Type{T}; n_time::Int, n_columns::Int,
        n_nodes::Int = 0, n_radii::Int = 0,
    ) where {T <: AbstractFloat}
    n_time > 0 && n_columns > 0 && n_nodes >= 0 && n_radii >= 0 ||
        throw(ArgumentError("invalid statistics workspace dimensions"))
    return StatisticsWorkspace(
        dzeros(execution, T, n_time, n_columns),
        dzeros(execution, T, n_time), dzeros(execution, T, n_columns),
        dzeros(execution, T, n_columns), dzeros(execution, T, n_nodes),
        dzeros(execution, T, n_nodes), dzeros(execution, T, n_nodes),
        dzeros(execution, T, n_radii),
    )
end

@inline _stat_pair(a, b, prototype) =
    typeof(prototype)(real(a)) * typeof(prototype)(real(b)) +
    typeof(prototype)(imag(a)) * typeof(prototype)(imag(b))
@inline _stat_abs2(a, prototype) = _stat_pair(a, a, prototype)

@inline _stat_disjoint(destination, ::Nothing) = nothing
@inline function _stat_disjoint(destination, source::AbstractArray)
    Base.mightalias(destination, source) &&
        throw(ArgumentError("statistics writable storage aliases prepared input"))
    return nothing
end
@inline _stat_disjoint_tuple(destination, ::Tuple{}) = nothing
@inline function _stat_disjoint_tuple(destination, sources::Tuple)
    _stat_disjoint(destination, first(sources))
    _stat_disjoint_tuple(destination, Base.tail(sources))
    return nothing
end
@inline function _stat_disjoint(destination, grid::StatisticsGrid)
    _stat_disjoint_tuple(
        destination, (
            grid.omega, grid.time, grid.parseval,
            grid.windows, grid.spectral_density,
        )
    )
    return nothing
end
@inline function _stat_disjoint(destination, input::SpectralStatisticsInput)
    _stat_disjoint(destination, input.spectrum)
    _stat_disjoint(destination, input.power_weight)
    return nothing
end
@inline function _stat_disjoint(destination, input::TemporalStatisticsInput)
    _stat_disjoint_tuple(
        destination, (
            input.analytic, input.weighted,
            input.window_analytic, input.window_weighted, input.axis_analytic,
            input.axis_weighted,
        )
    )
    return nothing
end
@inline function _stat_disjoint(destination, input::PopulationStatisticsInput)
    _stat_disjoint_tuple(
        destination, (
            input.densities, input.fractions, input.peak_rates,
            input.potentials, input.node_weights,
        )
    )
    return nothing
end
@inline function _stat_disjoint(destination, input::SpatialStatisticsInput)
    _stat_disjoint_tuple(
        destination, (
            input.spectrum, input.closure,
            input.analytic, input.analytic_closure, input.axis_spectrum, input.axis_closure,
            input.axis_analytic, input.axis_analytic_closure,
        )
    )
    if input.widths !== nothing
        _stat_disjoint_tuple(
            destination, (
                input.widths.x, input.widths.y,
                input.widths.weights, input.widths.cosines, input.widths.sines,
            )
        )
    end
    return nothing
end
@inline _stat_buffer_tuple(::Tuple{}, grid, input) = nothing
@inline function _stat_buffer_tuple(buffers::Tuple, grid, input)
    # The fixed concrete tuple specializes and unrolls at compilation, without
    # dynamic field inspection or temporary arrays in the observation path.
    destination, remaining = first(buffers), Base.tail(buffers)
    _stat_disjoint_tuple(destination, remaining)
    _stat_disjoint(destination, grid)
    _stat_disjoint(destination, input)
    _stat_buffer_tuple(remaining, grid, input)
    return nothing
end
function _stat_buffers(output, workspace, grid, input)
    size(workspace.power) == (
        length(workspace.total_power),
        length(workspace.column_energy),
    ) &&
        length(workspace.column_energy) == length(workspace.column_moment) &&
        length(workspace.node_total) == length(workspace.fluence) ==
        length(workspace.magnitude) ||
        throw(DimensionMismatch("statistics companion workspace dimensions disagree"))
    buffers = (
        output, workspace.power, workspace.total_power, workspace.column_energy,
        workspace.column_moment, workspace.node_total, workspace.fluence,
        workspace.magnitude, workspace.radial_average,
    )
    _stat_buffer_tuple(buffers, grid, input)
    return nothing
end

function _stat_family(output, slot, admitted)
    slot.code in admitted || throw(ArgumentError("statistic belongs to another family"))
    slot.offset + slot.count - 1 <= length(output) ||
        throw(DimensionMismatch("packed statistics destination is too short"))
    return nothing
end

function _stat_count(slot, count)
    slot.count == count ||
        throw(DimensionMismatch("packed $(_STATISTIC_NAMES[slot.code]) count is incorrect"))
    return nothing
end
function _stat_pair_shape(first, second, rows; dimensions = 2)
    (first === nothing || second === nothing) &&
        throw(ArgumentError("requested statistic is missing its field pair"))
    ndims(first) == ndims(second) == dimensions && size(first) == size(second) &&
        size(first, 1) == rows && size(first, 2) > 0 ||
        throw(DimensionMismatch("statistics field pair has incompatible dimensions"))
    all(isfinite, first) && all(isfinite, second) ||
        throw(ArgumentError("statistics field pairs must be finite"))
    return nothing
end

"""
    half_width(time, trace; outer=false)

Return the central or outermost half-maximum width, in the time axis's units.

Use the first maximum, strict below/above comparisons and the stable line crossing
`x1+(level-y1)*(x2-x1)/(y2-y1)`. Empty-power, nonpositive or unbracketed traces return
NaN, as the statistics catalogue requires. Input arrays must be finite, matching,
nonempty vectors on a strictly increasing axis; malformed input raises `ArgumentError`.
No contraction or reassociation changes the scalar crossing expression.
"""
function half_width(time, trace; outer::Bool = false)
    length(time) == length(trace) > 0 && all(isfinite, time) && all(isfinite, trace) &&
        all(index -> time[index] > time[index - 1], 2:length(time)) ||
        throw(ArgumentError("width requires matching finite samples on an increasing axis"))
    return _half_width(time, trace, outer)
end

function _half_width(time, trace, outer)
    apex = firstindex(trace)
    peak = trace[apex]
    for index in eachindex(trace)
        if trace[index] > peak
            peak, apex = trace[index], index
        end
    end
    peak <= 0 && return oftype(peak, NaN)
    level = 0.5 * peak
    low = high = 0
    if outer
        for index in eachindex(trace)
            if trace[index] > level
                low == 0 && (low = index)
                high = index
            end
        end
        low -= 1
        high += 1
    else
        for index in 1:(apex - 1)
            trace[index] < level && (low = index)
        end
        for index in apex:length(trace)
            if trace[index] < level
                high = index
                break
            end
        end
    end
    (low < 1 || high == 0 || high > length(trace)) && return oftype(peak, NaN)
    left = time[low] + (level - trace[low]) * (time[low + 1] - time[low]) /
        (trace[low + 1] - trace[low])
    right = time[high] + (level - trace[high]) * (time[high - 1] - time[high]) /
        (trace[high - 1] - trace[high])
    return right - left
end

"""
    spectral_statistics!(output, slot, grid, input, workspace, execution)

Fill one energy, spectral-window energy or centroid slice from the observed spectrum.

The input is `SpectralStatisticsInput`. Centroids are per column; empty columns yield
NaN. With exactly unit power weights, the catalogue also treats columns holding
at most `eps(Float64)` of the largest column as empty; nonidentity Poynting centroids
use only the zero denominator rule. `total=true` makes energy a scalar and window
energy one value per window. The
identity-metric alias `physical_energy` requires unit weights. Every finite reduction
is accumulated in the output's scalar type; malformed/nonfinite input raises an error.
"""
function spectral_statistics!(output, slot, grid, input, workspace, execution)
    _stat_family(output, slot, (1, 2, 3, 7))
    _stat_buffers(output, workspace, grid, input)
    input === nothing && throw(ArgumentError("spectral statistic needs observed spectra"))
    spectrum, weight = input.spectrum, input.power_weight
    size(spectrum) == size(weight) && size(spectrum, 1) == length(grid.omega) ||
        throw(DimensionMismatch("spectral statistics dimensions disagree"))
    all(isfinite, spectrum) && all(value -> isfinite(value) && value >= 0, weight) ||
        throw(ArgumentError("spectral data must be finite; power weights nonnegative"))
    n_columns = size(spectrum, 2)
    length(workspace.column_energy) == n_columns ||
        throw(DimensionMismatch("spectral statistics workspace has wrong column count"))
    code, offset = slot.code, slot.offset
    windows = code == 3 ? size(grid.windows, 2) : 1
    windows > 0 || throw(ArgumentError("windowed energy requires spectral windows"))
    total = code == 7 || (input.total && code != 1)
    centroid_floor = code == 1 && all(isone, weight)
    code == 7 && !all(isone, weight) &&
        throw(ArgumentError("physical_energy requires the identity power weight"))
    _stat_count(slot, windows * (total ? 1 : n_columns))
    energy, moment = workspace.column_energy, workspace.column_moment
    omega, parseval, masks = grid.omega, grid.parseval, grid.windows
    for window in 1:windows
        each_column!(execution, n_columns) do column
            numerator = denominator = zero(eltype(energy))
            for row in axes(spectrum, 1)
                code == 3 && !masks[row, window] && continue
                density = _stat_abs2(spectrum[row, column], numerator) * weight[row, column]
                if code == 1
                    numerator += density * omega[row]
                    denominator += density
                else
                    numerator += density * parseval[row]
                end
            end
            energy[column], moment[column] = numerator, denominator
        end
        all(isfinite, energy) && all(isfinite, moment) ||
            throw(ArgumentError("spectral statistic overflowed"))
        if total
            result = zero(eltype(output))
            for value in energy
                result += value
            end
            isfinite(result) || throw(ArgumentError("total spectral energy overflowed"))
            output[offset + window - 1] = result
        else
            floor = centroid_floor ? eps(Float64) * maximum(moment) : 0.0
            for column in 1:n_columns
                output[offset + (window - 1) * n_columns + column - 1] = code == 1 ?
                    (
                        moment[column] <= floor ? oftype(moment[column], NaN) :
                        energy[column] / moment[column]
                    ) : energy[column]
            end
        end
    end
    return nothing
end

function _temporal_power!(workspace, grid, input, execution)
    first, second = input.analytic, input.weighted
    _stat_pair_shape(first, second, length(grid.time))
    size(first) == size(workspace.power) ||
        throw(DimensionMismatch("temporal power workspace dimensions disagree"))
    power, total = workspace.power, workspace.total_power
    each!(execution, size(power)...) do row, column
        power[row, column] = _stat_pair(
            first[row, column], second[row, column],
            zero(eltype(power))
        )
    end
    each!(execution, length(total), 1) do row, _
        value = zero(eltype(total))
        for column in axes(power, 2)
            value += power[row, column]
        end
        total[row] = value
    end
    all(isfinite, power) && all(isfinite, total) ||
        throw(ArgumentError("temporal power statistic overflowed"))
    return nothing
end

function _pair_peak(first, second, factor, rows)
    _stat_pair_shape(first, second, rows)
    peak = -oftype(factor, Inf)
    for row in 1:rows
        value = zero(factor)
        for column in axes(first, 2)
            value += _stat_pair(first[row, column], second[row, column], value)
        end
        peak = max(peak, factor * value)
    end
    isfinite(peak) || throw(ArgumentError("intensity statistic overflowed"))
    return peak
end

"""
    temporal_statistics!(output, slot, grid, input, workspace, execution)

Fill one modal temporal peak, width or on-axis intensity slice from analytic pairs.

Power is the real Hermitian pairing, without clipping signed excursions. Window
pairs must already represent the matching hard gates. A requested missing pair or
incompatible slice raises an error. Undefined widths retain their documented NaN.
"""
function temporal_statistics!(output, slot, grid, input, workspace, execution)
    _stat_family(output, slot, (4, 5, 6, 8, 9, 10, 18))
    _stat_buffers(output, workspace, grid, input)
    input === nothing && throw(ArgumentError("temporal statistic needs analytic pairs"))
    code, offset = slot.code, slot.offset
    code == 8 && !input.identity_weight &&
        throw(ArgumentError("analytic_flux requires the identity power weight"))
    if code == 18
        _stat_count(slot, 1)
        output[offset] = _pair_peak(
            input.axis_analytic, input.axis_weighted,
            input.intensity_factor, length(grid.time)
        )
    elseif code == 5
        first, second = input.window_analytic, input.window_weighted
        first === nothing && throw(ArgumentError("windowed power needs windowed pairs"))
        _stat_pair_shape(first, second, length(grid.time); dimensions = 3)
        n_columns, n_windows = size(first, 2), size(first, 3)
        n_windows == size(grid.windows, 2) > 0 ||
            throw(DimensionMismatch("windowed power does not match spectral windows"))
        _stat_count(slot, n_columns * n_windows)
        each!(execution, n_columns, n_windows) do column, window
            peak = -oftype(zero(eltype(output)), Inf)
            for row in axes(first, 1)
                value = _stat_pair(
                    first[row, column, window], second[row, column, window],
                    zero(eltype(output))
                )
                peak = max(peak, value)
            end
            output[offset + (window - 1) * n_columns + column - 1] = peak
        end
        for index in offset:(offset + slot.count - 1)
            isfinite(output[index]) || throw(ArgumentError("windowed power overflowed"))
        end
    else
        _temporal_power!(workspace, grid, input, execution)
        power, total = workspace.power, workspace.total_power
        if code == 4
            _stat_count(slot, size(power, 2))
            each_column!(execution, size(power, 2)) do column
                peak = power[1, column]
                for row in 2:size(power, 1)
                    peak = max(peak, power[row, column])
                end
                output[offset + column - 1] = peak
            end
        else
            _stat_count(slot, 1)
            output[offset] = code in (9, 10) ? _half_width(grid.time, total, code == 10) :
                maximum(total)
        end
    end
    return nothing
end

"""
    population_statistics!(output, slot, input, workspace, execution)

Fill a scalar-plasma population slice using fresh final fractions and per-node peaks.

Mixture order includes non-ionising species. Densest-node ties choose the first node;
`peak_species` explicitly preserves the reference's single-population peak rule.
All populations and rates must be finite and in their documented domains.
"""
function population_statistics!(output, slot, input, workspace, execution)
    _stat_family(output, slot, (14, 15, 16, 17, 27, 28, 29))
    _stat_buffers(output, workspace, nothing, input)
    input === nothing &&
        throw(ArgumentError("population statistic needs fresh populations"))
    densities, fractions, peaks = input.densities, input.fractions, input.peak_rates
    n_nodes, n_species = size(fractions)
    size(peaks) == size(fractions) && length(densities) == n_species && n_nodes > 0 &&
        n_species > 0 && length(workspace.node_total) == n_nodes ||
        throw(DimensionMismatch("population statistic dimensions disagree"))
    all(value -> isfinite(value) && value >= 0, densities) &&
        all(value -> isfinite(value) && 0 <= value <= 1, fractions) &&
        all(value -> isfinite(value) && value >= 0, peaks) &&
        0 <= input.peak_species <= n_species ||
        throw(ArgumentError("invalid population statistic inputs"))
    total = workspace.node_total
    each!(execution, n_nodes, 1) do node, _
        value = zero(eltype(total))
        for species in 1:n_species
            value += densities[species] * fractions[node, species]
        end
        total[node] = value
    end
    all(isfinite, total) || throw(ArgumentError("electron density overflowed"))
    densest = 1
    for node in 2:n_nodes
        total[node] > total[densest] && (densest = node)
    end
    code, offset = slot.code, slot.offset
    if code == 14
        _stat_count(slot, n_nodes * n_species)
        for species in 1:n_species, node in 1:n_nodes
            output[offset + (species - 1) * n_nodes + node - 1] = fractions[node, species]
        end
    elseif code == 15
        _stat_count(slot, n_species)
        for species in 1:n_species
            output[offset + species - 1] = densities[species] * fractions[densest, species]
        end
    elseif code == 16
        _stat_count(slot, 1)
        output[offset] = total[densest]
    elseif code == 27
        _stat_count(slot, n_species)
        for species in 1:n_species
            output[offset + species - 1] = densities[species]
        end
    elseif code == 28
        _stat_count(slot, n_species)
        potentials, weights = input.potentials, input.node_weights
        if potentials === nothing || weights === nothing
            throw(ArgumentError("species inventory needs potentials and physical weights"))
        end
        length(potentials) == n_species && length(weights) == n_nodes ||
            throw(DimensionMismatch("species inventory measure dimensions disagree"))
        all(value -> isfinite(value) && value >= 0, potentials) &&
            all(value -> isfinite(value) && value >= 0, weights) &&
            isfinite(input.measure_factor) && input.measure_factor > 0 ||
            throw(ArgumentError("species inventory measure must be finite and nonnegative"))
        for species in 1:n_species
            value = 0.0
            for node in 1:n_nodes
                electrons = densities[species] * fractions[node, species]
                weight = weights[node] * input.measure_factor
                value += weight * electrons * potentials[species]
            end
            isfinite(value) || throw(ArgumentError("species ionisation energy overflowed"))
            output[offset + species - 1] = value
        end
    elseif code == 29
        _stat_count(slot, n_species)
        for species in 1:n_species
            value = 0.0
            for node in 1:n_nodes
                value = max(value, peaks[node, species])
            end
            output[offset + species - 1] = value
        end
    else
        _stat_count(slot, 1)
        species = input.peak_species
        if species == 0
            output[offset] = maximum(peaks)
        else
            peak_node = 1
            for node in 2:n_nodes
                if fractions[node, species] > fractions[peak_node, species]
                    peak_node = node
                end
            end
            output[offset] = peaks[peak_node, species]
        end
    end
    return nothing
end

function _axis_density(input, grid, row, prototype)
    value = zero(prototype)
    for column in axes(input.axis_spectrum, 2)
        value += _stat_pair(
            input.axis_spectrum[row, column],
            input.axis_closure[row, column], prototype
        )
    end
    return input.factor * grid.spectral_density[row] * value
end

function _axis_spectral_check(input, grid)
    _stat_pair_shape(input.axis_spectrum, input.axis_closure, length(grid.omega))
    length(grid.spectral_density) == length(grid.omega) ||
        throw(ArgumentError("on-axis spectrum needs spectral-density weights"))
    return nothing
end

function _fluence!(workspace, grid, input, execution)
    _stat_pair_shape(input.spectrum, input.closure, length(grid.omega))
    n_nodes = size(input.spectrum, 2)
    length(workspace.fluence) == length(workspace.magnitude) == n_nodes ||
        throw(DimensionMismatch("fluence workspace does not match physical nodes"))
    first, second, weights = input.spectrum, input.closure, grid.parseval
    fluence, magnitude, factor = workspace.fluence, workspace.magnitude, input.factor
    each_column!(execution, n_nodes) do node
        value = bound = zero(eltype(fluence))
        for row in axes(first, 1)
            value += weights[row] * _stat_pair(first[row, node], second[row, node], value)
            bound += weights[row] * abs(first[row, node]) * abs(second[row, node])
        end
        fluence[node], magnitude[node] = factor * value, factor * bound
    end
    all(isfinite, fluence) && all(isfinite, magnitude) ||
        throw(ArgumentError("spatial fluence reduction overflowed"))
    # Match the established synthesis/pairing reduction budget, including cancellation.
    budget = 8.0 * (length(grid.omega) + Float64(input.source_columns)) * eps(Float64) *
        maximum(magnitude)
    isfinite(budget) || throw(ArgumentError("spatial fluence error bound overflowed"))
    any(value -> value < -budget, fluence) &&
        throw(ArgumentError("beam widths require nonnegative fluence beyond roundoff"))
    each!(execution, n_nodes, 1) do node, _
        fluence[node] = max(fluence[node], zero(eltype(fluence)))
    end
    return nothing
end

function _radial_crossing(radius, fluence, axis, fraction)
    peak = max(axis, maximum(fluence))
    level = fraction * peak
    axis < level && return oftype(axis, NaN)
    left, position = axis, zero(eltype(radius))
    for index in eachindex(radius)
        right = fluence[index]
        right < level && return position +
            (left - level) / (left - right) * (radius[index] - position)
        left, position = right, radius[index]
    end
    return oftype(axis, NaN)
end

function _radial_widths(widths, fluence, axis)
    radius, weights = widths.x, widths.weights
    length(fluence) == length(radius) ||
        throw(DimensionMismatch("radial fluence does not match its geometry"))
    isfinite(axis) && axis >= 0 ||
        throw(ArgumentError("radial width requires finite nonnegative axial fluence"))
    energy = moment = zero(eltype(fluence))
    for node in eachindex(fluence)
        weighted = weights[node] * fluence[node]
        energy += weighted
        moment += weighted * radius[node]^2
    end
    isfinite(energy) && isfinite(moment) ||
        throw(ArgumentError("radial fluence moments overflowed"))
    energy <= 0 && return (NaN, NaN, NaN)
    second = sqrt(2.0) * sqrt(moment / energy)
    isfinite(second) || throw(ArgumentError("radial fluence moment is not representable"))
    return (
        second, _radial_crossing(radius, fluence, axis, exp(-2.0)),
        2 * _radial_crossing(radius, fluence, axis, 0.5),
    )
end

function _ring_crossing(averaged, count, spacing, fraction)
    peak = averaged[1]
    for index in 2:count
        peak = max(peak, averaged[index])
    end
    level = fraction * peak
    averaged[1] < level && return oftype(peak, NaN)
    for index in 2:count
        if averaged[index] < level
            left, right = averaged[index - 1], averaged[index]
            return (index - 2) * spacing + (left - level) / (left - right) * spacing
        end
    end
    return oftype(peak, NaN)
end

function _linear_fluence(fluence, x, coordinate, scale)
    position = clamp((coordinate - x[1]) / (x[2] - x[1]), 0, length(x) - 1)
    left = min(floor(Int, position), length(x) - 2)
    fraction = position - left
    return (1 - fraction) * (fluence[left + 1] / scale) +
        fraction * (fluence[left + 2] / scale)
end

function _bilinear_fluence(fluence, x, y, horizontal, vertical, scale)
    horizontal = clamp((horizontal - x[1]) / (x[2] - x[1]), 0, length(x) - 1)
    vertical = clamp((vertical - y[1]) / (y[2] - y[1]), 0, length(y) - 1)
    ix, iy = min(floor(Int, horizontal), length(x) - 2),
        min(floor(Int, vertical), length(y) - 2)
    fx, fy = horizontal - ix, vertical - iy
    first = iy * length(x) + ix + 1
    second = first + length(x)
    return (1 - fy) * (
        (1 - fx) * (fluence[first] / scale) +
            fx * (fluence[first + 1] / scale)
    ) +
        fy * (
        (1 - fx) * (fluence[second] / scale) +
            fx * (fluence[second + 1] / scale)
    )
end

function _cartesian_widths(widths, fluence, averaged)
    x, y = widths.x, widths.y
    nx, ny = length(x), length(y)
    length(fluence) == nx * ny ||
        throw(DimensionMismatch("Cartesian fluence does not match its axes"))
    length(averaged) >= widths.radius_capacity ||
        throw(DimensionMismatch("radial-average scratch is too short"))
    scale = maximum(fluence)
    scale == 0 && return (NaN, NaN, NaN)
    total = horizontal = vertical = zero(eltype(fluence))
    for iy in 1:ny, ix in 1:nx
        sample = fluence[ix + nx * (iy - 1)] / scale
        total += sample
        horizontal += sample * x[ix]
        vertical += sample * y[iy]
    end
    center_x, center_y = horizontal / total, vertical / total
    extent = zero(eltype(fluence))
    for iy in 1:ny, ix in 1:nx
        distance = ny == 1 ? abs(x[ix] - center_x) :
            hypot(x[ix] - center_x, y[iy] - center_y)
        extent = max(extent, distance)
    end
    moment = zero(eltype(fluence))
    for iy in 1:ny, ix in 1:nx
        distance = ny == 1 ? abs(x[ix] - center_x) :
            hypot(x[ix] - center_x, y[iy] - center_y)
        moment += (fluence[ix + nx * (iy - 1)] / scale) * (distance / extent)^2
    end
    second = ny == 1 ? 2 * extent * sqrt(moment / total) :
        extent * sqrt(2 * moment / total)
    isfinite(second) || throw(ArgumentError("Cartesian fluence moments overflowed"))
    limit = min(center_x - x[1], x[end] - center_x)
    ny > 1 && (limit = min(limit, center_y - y[1], y[end] - center_y))
    spacing = widths.spacing
    limit < 2 * spacing && return (second, NaN, NaN)
    count = floor(Int, limit / spacing) + 1
    for index in 1:count
        radius = (index - 1) * spacing
        if ny == 1
            averaged[index] = 0.5 * (
                _linear_fluence(fluence, x, center_x - radius, scale) +
                    _linear_fluence(fluence, x, center_x + radius, scale)
            )
        else
            value = zero(eltype(averaged))
            for angle in eachindex(widths.cosines)
                value += _bilinear_fluence(
                    fluence, x, y,
                    center_x + radius * widths.cosines[angle],
                    center_y + radius * widths.sines[angle], scale
                )
            end
            averaged[index] = value / length(widths.cosines)
        end
    end
    return (
        second, _ring_crossing(averaged, count, spacing, exp(-2.0)),
        2 * _ring_crossing(averaged, count, spacing, 0.5),
    )
end

"""
    spatial_statistics!(output, slot, grid, input, workspace, execution)

Fill a scalar spatial intensity, width or on-axis spectral-density slice.

The input is `SpatialStatisticsInput`. Intensities keep signed axial-flux excursions.
Widths use the observed fluence, reject negative values beyond the established
`8*(n_omega+source_columns)*eps(Float64)` absolute-product budget, and clip only the
remaining roundoff. The two `beam_radius` entries are the second-moment and first
1/e² radii; `fwhm_r` is the first half-maximum diameter. Empty, hollow or unbracketed
widths retain catalogue NaNs. Axis spectra have `n_omega` entries in J/(rad/s)/m².

Radial widths use the independently reconstructed axial spectrum. Cartesian and line
thresholds use the existing grid-fixed angular/half-pixel radial sampling rule about
the fluence centroid. This is the same discrete observable as `solprop.spatial`, not
an extra continuum approximation. Malformed or unrepresentable inputs raise errors.
"""
function spatial_statistics!(output, slot, grid, input, workspace, execution)
    _stat_family(output, slot, (18, 19, 21, 22, 23))
    _stat_buffers(output, workspace, grid, input)
    input === nothing &&
        throw(ArgumentError("spatial statistic needs physical field pairs"))
    code, offset = slot.code, slot.offset
    if code == 18
        _stat_count(slot, 1)
        _stat_pair_shape(input.analytic, input.analytic_closure, length(grid.time))
        peak = -oftype(input.factor, Inf)
        for index in eachindex(input.analytic, input.analytic_closure)
            peak = max(
                peak, input.factor * _stat_pair(
                    input.analytic[index],
                    input.analytic_closure[index], input.factor
                )
            )
        end
        isfinite(peak) || throw(ArgumentError("spatial peak intensity overflowed"))
        output[offset] = peak
    elseif code == 19
        _stat_count(slot, 1)
        output[offset] = _pair_peak(
            input.axis_analytic, input.axis_analytic_closure,
            input.factor, length(grid.time)
        )
    elseif code == 23
        _stat_count(slot, length(grid.omega))
        _axis_spectral_check(input, grid)
        for row in eachindex(grid.omega)
            value = _axis_density(input, grid, row, zero(eltype(output)))
            isfinite(value) || throw(ArgumentError("on-axis spectral density overflowed"))
            output[offset + row - 1] = value
        end
    else
        _stat_count(slot, code == 21 ? 2 : 1)
        widths = input.widths
        widths === nothing && throw(ArgumentError("spatial widths need prepared geometry"))
        _fluence!(workspace, grid, input, execution)
        if widths.kind == :radial
            _axis_spectral_check(input, grid)
            isfinite(grid.d_omega) ||
                throw(ArgumentError("radial axial fluence needs frequency-bin spacing"))
            axis = zero(eltype(output))
            for row in eachindex(grid.omega)
                axis += _axis_density(input, grid, row, zero(eltype(output)))
            end
            result = _radial_widths(widths, workspace.fluence, axis * grid.d_omega)
        else
            result = _cartesian_widths(widths, workspace.fluence, workspace.radial_average)
        end
        output[offset] = result[code == 21 ? 1 : 3]
        code == 21 && (output[offset + 1] = result[2])
    end
    return nothing
end

"""
    statistics!(output, selection, grid, inputs, workspace, execution; periodic=false)

Evaluate the selected scalar catalogue into caller-owned packed numerical storage.

`output` is a real vector of exactly `selection.width` entries; buffers and prepared
pairs must not alias it. All inputs describe this actual observation, including any
fresh population scans. Only selected families are inspected. Periodic evaluation
fills its existing slices and clears all unselected slices to zero; the session owns
row metadata, cadence, continuation and delivery. No callback, history growth or
Python execution occurs here. All retained scratch is listed by `StatisticsWorkspace`.

Undefined widths, zero-density centroids and unavailable quadrature alarms return NaN.
Other missing, malformed, nonfinite or unrepresentable requested inputs raise
`ArgumentError` or `DimensionMismatch`.
A failed evaluation does not publish a row; the session owns that transaction.
"""
function statistics!(
        output, selection::StatisticsSelection{<:Tuple{Vararg{StatisticSlot}}},
        grid::StatisticsGrid,
        inputs::StatisticsInputs, workspace::StatisticsWorkspace, execution;
        periodic::Bool = false
    )
    length(output) == selection.width ||
        throw(DimensionMismatch("statistics output does not match its packed selection"))
    # Validate every selected family before clearing a destination that might be a
    # view of its inputs. Family entry points enforce the same boundary directly.
    _stat_buffers(output, workspace, grid, nothing)
    for (index, slot) in pairs(selection.slots)
        periodic && !(index in selection.periodic) && continue
        code = slot.code
        if code in (1, 2, 3, 7)
            _stat_buffers(output, workspace, grid, inputs.spectral)
        elseif code in (4, 5, 6, 8, 9, 10) || (code == 18 && inputs.spatial === nothing)
            _stat_buffers(output, workspace, grid, inputs.temporal)
        elseif 14 <= code <= 17 || 27 <= code <= 29
            _stat_buffers(output, workspace, grid, inputs.populations)
        elseif code in (18, 19, 21, 22, 23)
            _stat_buffers(output, workspace, grid, inputs.spatial)
        end
    end
    fill!(output, zero(eltype(output)))
    for (index, slot) in pairs(selection.slots)
        periodic && !(index in selection.periodic) && continue
        code = slot.code
        if code in (1, 2, 3, 7)
            spectral_statistics!(output, slot, grid, inputs.spectral, workspace, execution)
        elseif code in (4, 5, 6, 8, 9, 10) || (code == 18 && inputs.spatial === nothing)
            temporal_statistics!(output, slot, grid, inputs.temporal, workspace, execution)
        elseif 11 <= code <= 13
            _stat_count(slot, 1)
            value = inputs.profiles[code - 10]
            isfinite(value) || throw(ArgumentError("requested scalar profile is nonfinite"))
            output[slot.offset] = value
        elseif 14 <= code <= 17 || 27 <= code <= 29
            population_statistics!(output, slot, inputs.populations, workspace, execution)
        elseif 24 <= code <= 26
            _stat_count(slot, 1)
            inputs.diagnostics isa NTuple{3, Float64} ||
                throw(ArgumentError("plasma diagnostic statistics need a fresh scan"))
            all(value -> isfinite(value) && value >= 0, inputs.diagnostics) ||
                throw(ArgumentError("plasma diagnostics must be nonnegative finite"))
            output[slot.offset] = inputs.diagnostics[code - 23]
        elseif code == 20
            _stat_count(slot, 1)
            (
                isnan(inputs.quadrature_error) ||
                    (isfinite(inputs.quadrature_error) && inputs.quadrature_error >= 0)
            ) || throw(ArgumentError("quadrature alarm must be nonnegative finite or NaN"))
            output[slot.offset] = inputs.quadrature_error
        else
            spatial_statistics!(output, slot, grid, inputs.spatial, workspace, execution)
        end
    end
    return nothing
end
