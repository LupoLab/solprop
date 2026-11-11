# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public SourceProfile, SourcePrefactor, ModalSourceGrid, ModalSourceMap
public ModalKerr, ModalRaman, ModalPlasmaChannel, ModalPlasma, ModalSourceData
public ModalSource, SourcePlasmaError, set_temporal_source!

"""
    SourceProfile(value::Float64; name=:profile)
    SourceProfile(table::ChebyshevTable; name=:profile)

Own one independent scalar, direct-value material or taper profile.

The constant or scalar binary64 Chebyshev table keeps its translated SI units and
admitted position span. A table is copied, including its breakpoints. It is never
interpreted as a cumulative operator. `name` identifies a failed stage evaluation.
Throw `ArgumentError` for nonfinite constants or non-scalar/binary64 tables.
"""
struct SourceProfile{T}
    value::T
    name::Symbol
end

function SourceProfile(value::Float64; name::Symbol = :profile)
    isfinite(value) || throw(ArgumentError("a source profile must be finite"))
    return SourceProfile(value, name)
end

function _copy_source_table(table::ChebyshevTable)
    return ChebyshevTable(
        copy(table.breakpoints), copy(table.coefficients), table.lower, table.upper
    )
end

function SourceProfile(table::ChebyshevTable; name::Symbol = :profile)
    eltype(table.coefficients) === Float64 &&
        eltype(table.breakpoints) === Float64 && size(table.coefficients, 1) == 1 ||
        throw(ArgumentError("a source profile requires a scalar binary64 table"))
    return SourceProfile(_copy_source_table(table), name)
end

@inline _source_profile(profile::SourceProfile{Float64}, position) = profile.value

@inline function _source_profile(profile::SourceProfile{<:ChebyshevTable}, position)
    table = profile.value
    table.lower <= position <= table.upper ||
        throw(ArgumentError("source profile $(profile.name) is outside its admitted span"))
    value = _table_value(table.coefficients, table.breakpoints, 1, position)
    isfinite(value) ||
        throw(ArgumentError("source profile $(profile.name) is nonfinite at this position"))
    return value
end

"""
    SourcePrefactor(execution, values::AbstractMatrix{ComplexF64})
    SourcePrefactor(table::ChebyshevTable, shape::Tuple{Int,Int})
    SourcePrefactor(execution, terms::AbstractArray{ComplexF64,3}, weights::ChebyshevTable)

Own the translated modal source prefactor, including its existing crop scale.

Constant values have shape `(n_omega,n_modes)`. A complex binary64 table contains
`prod(shape)` values in column-major order. Expansion terms have shape
`(n_omega,n_modes,n_terms)` and their real binary64 weight table contains `n_terms`
direct values. Neither table nor weights are cumulative. All arrays/tables are copied.
Throw `ArgumentError` or `DimensionMismatch` for invalid types, sizes or values.
"""
struct SourcePrefactor{D, W}
    data::D
    weights::W
    shape::Tuple{Int, Int}
end

function SourcePrefactor(execution, values::AbstractMatrix{ComplexF64})
    !isempty(values) && all(isfinite, values) ||
        throw(ArgumentError("constant source prefactors must be finite and nonempty"))
    return SourcePrefactor(upload(execution, values), nothing, size(values))
end

function SourcePrefactor(table::ChebyshevTable, shape::Tuple{Int, Int})
    eltype(table.coefficients) === ComplexF64 &&
        eltype(table.breakpoints) === Float64 && all(>(0), shape) ||
        throw(ArgumentError("source prefactor tables require complex binary64 values"))
    prod(shape) == size(table.coefficients, 1) ||
        throw(DimensionMismatch("source prefactor table does not match its shape"))
    return SourcePrefactor(_copy_source_table(table), nothing, shape)
end

function SourcePrefactor(
        execution, terms::AbstractArray{ComplexF64, 3}, weights::ChebyshevTable,
    )
    !isempty(terms) && all(isfinite, terms) &&
        eltype(weights.coefficients) === Float64 &&
        eltype(weights.breakpoints) === Float64 ||
        throw(ArgumentError("source prefactor expansion requires finite binary64 data"))
    size(terms, 3) == size(weights.coefficients, 1) ||
        throw(DimensionMismatch("source prefactor terms and weights disagree"))
    return SourcePrefactor(
        upload(execution, terms), _copy_source_table(weights), size(terms)[1:2]
    )
end

function _source_prefactor!(output, prefactor, position, weight_work, execution)
    if prefactor.weights !== nothing
        evaluate_table!(weight_work, prefactor.weights, position, execution)
        terms = prefactor.data
        each!(execution, size(output)...) do row, column
            total = zero(eltype(output))
            for term in axes(terms, 3)
                total += weight_work[term] * terms[row, column, term]
            end
            output[row, column] = total
        end
    elseif prefactor.data isa ChebyshevTable
        evaluate_table!(output, prefactor.data, position, execution)
    else
        copyto!(output, prefactor.data)
    end
    return nothing
end

"""
    ModalSourceGrid(execution; kind, n_time, dt, crop_index, omega, active,
                    omega_window, source_window, source_window_derivative,
                    temporal_rate=nothing, carrier=nothing)

Own binary64 temporal/spectral data for a scalar modal source.

`kind` is `:field` or `:envelope`; `n_time` is the even fine temporal count and `dt`
its spacing in seconds. `crop_index` is the authoritative zero-based coarse-to-fine
map. Coarse `omega` is in rad/s; every active bin is positive. Boolean `active` and
dimensionless `omega_window` have the coarse extent. Fine `source_window` is in
`[0,1]`, its supplied time derivative is in s⁻¹, and optional `temporal_rate` is a
nonnegative power-loss rate in m⁻¹ for the in-stage absorber. Optional complex
`carrier` is the translated envelope harmonic multiplier. Padding/cropping scales
are deliberately absent: synthesis and prefactor already contain them.

Throw `ArgumentError` or `DimensionMismatch` for invalid numerical data or layout.
"""
struct ModalSourceGrid{T, C}
    kind::Symbol
    n_time::Int
    dt::Float64
    crop_index::Vector{Int}
    omega::Vector{Float64}
    active::Vector{Bool}
    omega_window::Vector{Float64}
    source_window::Vector{Float64}
    source_window_derivative::Vector{Float64}
    temporal_rate::T
    carrier::C
end

function ModalSourceGrid(
        execution; kind::Symbol, n_time::Int, dt::Float64, crop_index,
        omega::AbstractVector{Float64}, active::AbstractVector{Bool},
        omega_window::AbstractVector{Float64}, source_window::AbstractVector{Float64},
        source_window_derivative::AbstractVector{Float64}, temporal_rate = nothing,
        carrier = nothing,
    )
    kind in (:field, :envelope) && n_time >= 4 && iseven(n_time) &&
        isfinite(dt) && dt > 0 || throw(ArgumentError("invalid modal source time grid"))
    n_frequency = kind == :field ? n_time ÷ 2 + 1 : n_time
    length(crop_index) == length(omega) == length(active) == length(omega_window) ||
        throw(DimensionMismatch("modal source coarse axes disagree"))
    length(source_window) == length(source_window_derivative) == n_time ||
        throw(DimensionMismatch("modal source fine axes disagree"))
    all(isfinite, omega) && all(isfinite, source_window_derivative) &&
        all(value -> isfinite(value) && 0 <= value <= 1, omega_window) &&
        all(value -> isfinite(value) && 0 <= value <= 1, source_window) &&
        all(index -> !active[index] || omega[index] > 0, eachindex(omega)) ||
        throw(ArgumentError("invalid modal source frequencies, windows or active band"))
    padding = SpectralPadding(execution, crop_index, n_frequency, 1.0, 1.0)
    if temporal_rate !== nothing
        temporal_rate isa AbstractVector{Float64} && length(temporal_rate) == n_time &&
            all(value -> isfinite(value) && value >= 0, temporal_rate) ||
            throw(ArgumentError("the temporal source rate needs nonnegative fine values"))
    end
    if carrier !== nothing
        kind == :envelope && carrier isa AbstractVector{ComplexF64} &&
            length(carrier) == n_time && all(isfinite, carrier) ||
            throw(ArgumentError("the harmonic carrier must match an envelope time grid"))
    end
    return ModalSourceGrid(
        kind, n_time, dt, padding.crop .- 1, upload(execution, omega),
        upload(execution, active), upload(execution, omega_window),
        upload(execution, source_window), upload(execution, source_window_derivative),
        temporal_rate === nothing ? nothing : upload(execution, temporal_rate),
        carrier === nothing ? nothing : upload(execution, carrier),
    )
end

"""
    ModalSourceMap(execution; synthesis, projection, coarse_projection=nothing,
                   n_state_components=1, n_physical_components=1,
                   taper=SourceProfile(1.0; name=:taper))

Own the static scalar modal maps and their independent positive taper profile.

`synthesis[node,mode]` contains translated physical-amplitude and pad scales;
`projection[mode,node]` contains quadrature weights. Optional `coarse_projection`
has the same shape and supplies the embedded quadrature reading. At each position,
synthesis is multiplied by the dimensionless taper and projection divided by it.
The two declared component counts must both be one; vector polarisation refuses by
name. General or changing modal bases are not represented by this record.
Throw `ArgumentError` or `DimensionMismatch` before workspace allocation for bad data.
"""
struct ModalSourceMap{C, T}
    synthesis::Matrix{Float64}
    projection::Matrix{Float64}
    coarse_projection::C
    n_state_components::Int
    n_physical_components::Int
    taper::T
end

function ModalSourceMap(
        execution; synthesis::AbstractMatrix{Float64}, projection::AbstractMatrix{Float64},
        coarse_projection = nothing, n_state_components::Int = 1,
        n_physical_components::Int = 1,
        taper::SourceProfile = SourceProfile(1.0; name = :taper),
    )
    n_state_components == n_physical_components == 1 ||
        throw(ArgumentError("Julia modal source refuses vector polarisation"))
    nodes, modes = size(synthesis)
    nodes > 0 && modes > 0 && all(isfinite, synthesis) && all(isfinite, projection) ||
        throw(ArgumentError("modal maps must be nonempty and finite"))
    size(projection) == (modes, nodes) ||
        throw(DimensionMismatch("modal source synthesis and projection disagree"))
    if coarse_projection !== nothing
        coarse_projection isa AbstractMatrix{Float64} &&
            size(coarse_projection) == size(projection) &&
            all(isfinite, coarse_projection) ||
            throw(DimensionMismatch("embedded modal projection does not match the map"))
    end
    return ModalSourceMap(
        upload(execution, synthesis), upload(execution, projection),
        coarse_projection === nothing ? nothing : upload(execution, coarse_projection),
        n_state_components, n_physical_components, taper,
    )
end

"""
    ModalKerr(coefficients::KerrCoefficients{Float64}, density::SourceProfile)

Bind translated cubic coefficients to their independent stage weight.

The coefficient already contains its translated material factor; the profile is
applied once. A mixture's weighted susceptibility profile need not be a number density
or nonnegative. Grid/form compatibility is checked by `ModalSourceData`.
"""
struct ModalKerr{K, D}
    coefficients::K
    density::D
    function ModalKerr(coefficients::KerrCoefficients{Float64}, density::SourceProfile)
        return new{typeof(coefficients), typeof(density)}(coefficients, density)
    end
end

"""
    ModalRaman(kernel, decay_profiles::Tuple, density::SourceProfile;
               fraction, material_factor, thg=true, tensor=false)

Own one scalar Raman component and its independent linewidth/amplitude profiles.

`kernel` is a binary64 `RamanKernel` without epsilon₀. Each row has its own scalar
nonnegative decay-rate profile in s⁻¹; `density` is its nonnegative amplitude weight.
`fraction` is in `(0,1]`; `material_factor` is the positive translated epsilon₀ factor.
These three weights each enter once. `thg` selects the field drive; an envelope always
uses half its squared magnitude. `tensor=true` refuses tensor Raman by name.
Throw `ArgumentError` or `DimensionMismatch` for unsupported or inconsistent inputs.
"""
struct ModalRaman{K, R, D}
    kernel::K
    decay_profiles::R
    density::D
    fraction::Float64
    material_factor::Float64
    thg::Bool
end

function ModalRaman(
        kernel::RamanKernel{Float64}, decay_profiles::Tuple, density::SourceProfile;
        fraction::Float64, material_factor::Float64, thg::Bool = true, tensor::Bool = false,
    )
    tensor && throw(ArgumentError("Julia modal source refuses tensor Raman"))
    length(decay_profiles) == size(kernel.undamped, 2) &&
        all(profile -> profile isa SourceProfile, decay_profiles) ||
        throw(DimensionMismatch("each Raman row needs its own scalar decay profile"))
    isfinite(fraction) && 0 < fraction <= 1 &&
        isfinite(material_factor) && material_factor > 0 ||
        throw(ArgumentError("invalid Raman fraction or translated material factor"))
    return ModalRaman(
        RamanKernel(copy(kernel.undamped), kernel.dt), decay_profiles, density,
        fraction, material_factor, thg,
    )
end

"""
    ModalPlasmaChannel(channel, density, preionisation; quadrature=nothing)

Own one binary64 plasma channel with independent density and preionisation profiles.

`channel` is the translated `PlasmaChannel`; its rate spline is copied. Stage density
is in m⁻³ and nonnegative; preionisation is in `[0,1]`. An envelope requires its own
translated `CycleQuadrature`, whose arrays are copied. A field uses no cycle rule.
These profile ranges are checked at each source position before launching kernels.
"""
struct ModalPlasmaChannel{C, D, P, Q}
    channel::C
    density::D
    preionisation::P
    quadrature::Q
end

function ModalPlasmaChannel(
        channel::PlasmaChannel{Float64}, density::SourceProfile,
        preionisation::SourceProfile; quadrature = nothing,
    )
    rate = channel.rate
    owned = IonisationRateTable(
        copy(rate.coefficients), rate.field_min, rate.field_max, rate.rate_max,
        rate.log_field_min, rate.d_log_field,
    )
    quadrature === nothing || quadrature isa CycleQuadrature{Float64} ||
        throw(ArgumentError("envelope plasma needs a binary64 cycle rule"))
    copied_rule = quadrature === nothing ? nothing : CycleQuadrature(
            copy(quadrature.nodes), copy(quadrature.weights), quadrature.ends;
            moment_rtol = quadrature.moment_rtol,
            negligible_rate = quadrature.negligible_rate,
        )
    return ModalPlasmaChannel(
        PlasmaChannel(owned, channel.charge_mass_ratio, channel.ionisation_potential),
        density, preionisation, copied_rule,
    )
end

"""
    ModalPlasma(execution, channels::Tuple, rule::RunningIntegralRule{Float64})
    ModalPlasma(execution, channels::Tuple; drude, loss)

Own the independent plasma channels and their field or envelope assembly data.

A field copies the translated running-integral rule. An envelope copies its coarse
`drude=-1/omega²` and `loss=-im/omega` factors (including a translated frozen-carrier
policy if selected upstream), already exactly zero outside its supported band. Every
envelope channel supplies a cycle rule; field channels do not. No physical constant,
frequency approximation or population profile is inferred here.
Throw `ArgumentError` or `DimensionMismatch` for incompatible channel/assembly data.
"""
struct ModalPlasma{C, R, D, L}
    channels::C
    rule::R
    drude::D
    loss::L
end

function _source_channels(channels)
    !isempty(channels) && all(channel -> channel isa ModalPlasmaChannel, channels) ||
        throw(ArgumentError("modal plasma needs at least one declared channel"))
    return nothing
end

function ModalPlasma(execution, channels::Tuple, rule::RunningIntegralRule{Float64})
    _source_channels(channels)
    all(channel -> channel.quadrature === nothing, channels) ||
        throw(ArgumentError("field plasma does not use envelope cycle rules"))
    owned = RunningIntegralRule(
        upload(execution, rule.taps), rule.dt; floor_exposure = rule.floor_exposure
    )
    return ModalPlasma(channels, owned, nothing, nothing)
end

function ModalPlasma(
        execution, channels::Tuple; drude::AbstractVector{Float64},
        loss::AbstractVector{ComplexF64},
    )
    _source_channels(channels)
    all(channel -> channel.quadrature !== nothing, channels) ||
        throw(ArgumentError("each envelope plasma channel needs a cycle rule"))
    length(drude) == length(loss) ||
        throw(DimensionMismatch("envelope plasma spectral factors disagree"))
    all(value -> isfinite(value) && value <= 0, drude) &&
        all(value -> isfinite(value) && iszero(real(value)) && imag(value) <= 0, loss) ||
        throw(ArgumentError("invalid translated envelope plasma spectral factors"))
    return ModalPlasma(
        channels, nothing, upload(execution, drude), upload(execution, loss)
    )
end

"""
    ModalSourceData(grid, map, prefactor; kerr=nothing, raman=(), plasma=nothing)

Bind independently owned immutable records for an admitted scalar modal source.

The field/envelope kind, axes, scalar cubic form and all response grids must agree.
Raman components and plasma channels remain typed tuples in translated material order.
This record contains no workspace, session policy or Python object. Input noise belongs
to initial-state preparation; modified noise, vector/tensor, general modal and changing
bases are outside this source's admitted scope.
Throw `ArgumentError` or `DimensionMismatch` for incompatible records.
"""
struct ModalSourceData{G, M, P, K, R, L}
    grid::G
    map::M
    prefactor::P
    kerr::K
    raman::R
    plasma::L
end

function ModalSourceData(
        grid::ModalSourceGrid, map::ModalSourceMap, prefactor::SourcePrefactor;
        kerr::Union{Nothing, ModalKerr} = nothing, raman::Tuple = (),
        plasma::Union{Nothing, ModalPlasma} = nothing,
    )
    prefactor.shape == (length(grid.omega), size(map.synthesis, 2)) ||
        throw(DimensionMismatch("source prefactor does not match the modal grid"))
    _validate_source_responses(grid, kerr, raman, plasma)
    return ModalSourceData(grid, map, prefactor, kerr, raman, plasma)
end

function _validate_source_responses(
        grid::ModalSourceGrid, kerr::Union{Nothing, ModalKerr}, raman::Tuple,
        plasma::Union{Nothing, ModalPlasma},
    )
    if kerr !== nothing
        coefficients = kerr.coefficients
        if grid.kind == :envelope
            coefficients isa KerrCoefficients{Float64, :analytic} &&
                !coefficients.take_real ||
                throw(ArgumentError("an envelope source requires analytic complex Kerr"))
        elseif coefficients isa KerrCoefficients{Float64, :analytic}
            coefficients.take_real ||
                throw(ArgumentError("analytic field Kerr must take the real part"))
        end
        iszero(coefficients.harmonic_factor) || grid.carrier !== nothing ||
            throw(ArgumentError("harmonic Kerr requires the translated envelope carrier"))
    end
    for component in raman
        component isa ModalRaman || throw(ArgumentError("invalid scalar Raman component"))
        size(component.kernel.undamped, 1) == 2grid.n_time &&
            component.kernel.dt == grid.dt ||
            throw(DimensionMismatch("Raman rows must match the fine source grid"))
    end
    if plasma !== nothing
        if grid.kind == :field
            plasma.rule !== nothing && plasma.rule.dt == grid.dt ||
                throw(ArgumentError("field plasma requires the matching integral rule"))
            length(plasma.rule.taps) <= grid.n_time ||
                throw(ArgumentError("plasma FIR radius must be smaller than the time grid"))
        else
            plasma.rule === nothing ||
                throw(ArgumentError("envelope plasma requires cycle quadrature assembly"))
            length(plasma.drude) == length(grid.omega) ||
                throw(DimensionMismatch("plasma factors do not match the coarse grid"))
            all(eachindex(grid.active)) do index
                return grid.active[index] ||
                    (iszero(plasma.drude[index]) && iszero(plasma.loss[index]))
            end || throw(ArgumentError("plasma factors must vanish outside the band"))
        end
    end
    return nothing
end

struct _SourceChunk{S, P, C}
    first::Int
    count::Int
    synthesis::S
    projection::P
    coarse_projection::C
end

struct _SourceAnalytic{F}
    modal::Matrix{ComplexF64}
    spectrum::Matrix{ComplexF64}
    nodes::Matrix{ComplexF64}
    transform::F
end

struct _SourceRamanBuffers{F}
    drive::Matrix{Float64}
    frequency::Matrix{ComplexF64}
    transform::F
end

mutable struct _SourceRamanWork{D, F, C}
    data::D
    drive::RamanDrive
    time_kernel::Vector{Float64}
    spectrum::Vector{ComplexF64}
    rates::Vector{Float64}
    weights::Vector{Float64}
    forward::F
    convolution::C
    density::Float64

    function _SourceRamanWork(
            data::D, drive::RamanDrive, time_kernel::Vector{Float64},
            spectrum::Vector{ComplexF64}, rates::Vector{Float64}, weights::Vector{Float64},
            forward::F, convolution::C, density::Float64,
        ) where {D, F, C}
        return new{D, F, C}(
            data, drive, time_kernel, spectrum, rates, weights,
            forward, convolution, density,
        )
    end
end

struct _SourcePlasmaWork{W, A, C}
    workspace::W
    first::A
    second::A
    projected_first::A
    projected_second::A
    coarse_first::C
    coarse_second::C
    factors_first::Vector{ComplexF64}
    factors_second::Vector{ComplexF64}
    stage::Matrix{Float64}

    # These are already-owned prepared buffers. Keep construction from implicitly
    # converting arbitrary arrays into replacement storage with different ownership.
    function _SourcePlasmaWork(
            workspace::W, first::A, second::A, projected_first::A, projected_second::A,
            coarse_first::C, coarse_second::C, factors_first::Vector{ComplexF64},
            factors_second::Vector{ComplexF64}, stage::Matrix{Float64},
        ) where {W, A, C}
        return new{W, A, C}(
            workspace, first, second, projected_first, projected_second,
            coarse_first, coarse_second, factors_first, factors_second, stage,
        )
    end
end

"""
    SourcePlasmaError(code, channel, node, sample)

Report a failed source channel after its synchronous numerical kernel completes.

`code` is the leaf's `PlasmaCode`. Channel, global spatial node and temporal sample
indices are one-based. No later channel or projection consumes a failed channel.
"""
struct SourcePlasmaError <: Exception
    code::PlasmaCode
    channel::Int
    node::Int
    sample::Int
end

function Base.showerror(io::IO, error::SourcePlasmaError)
    return print(
        io, "plasma source ", error.code, " in channel ", error.channel,
        " at node ", error.node, ", sample ", error.sample,
    )
end

@inline _source_alias(array, ::Nothing) = false
@inline _source_alias(array, value::Float64) = false
@inline _source_alias(array, owned::AbstractArray) = _transfer_arrays_alias(array, owned)
@inline _source_alias(array, ::Tuple{}) = false
@inline function _source_alias(array, owned::Tuple{T, Vararg}) where {T}
    return _source_alias(array, first(owned)) || _source_alias(array, Base.tail(owned))
end

@inline function _source_alias(array, owned::ChebyshevTable)
    return _source_alias(array, (owned.breakpoints, owned.coefficients))
end

@inline _source_alias(array, owned::SourceProfile) = _source_alias(array, owned.value)

@inline function _source_alias(array, owned::SourcePrefactor)
    return _source_alias(array, (owned.data, owned.weights))
end

@inline function _source_alias(array, owned::ModalSourceGrid)
    return _source_alias(
        array, (
            owned.crop_index, owned.omega, owned.active, owned.omega_window,
            owned.source_window, owned.source_window_derivative,
            owned.temporal_rate, owned.carrier,
        )
    )
end

@inline function _source_alias(array, owned::ModalSourceMap)
    return _source_alias(
        array, (
            owned.synthesis, owned.projection, owned.coarse_projection, owned.taper,
        )
    )
end

@inline _source_alias(array, owned::ModalKerr) = _source_alias(array, owned.density)

@inline function _source_alias(array, owned::ModalRaman)
    return _source_alias(
        array, (
            owned.kernel.undamped, owned.decay_profiles, owned.density,
        )
    )
end

@inline function _source_alias(array, owned::ModalPlasmaChannel)
    retained = (owned.channel.rate.coefficients, owned.density, owned.preionisation)
    return _source_alias(array, retained) || (
        owned.quadrature !== nothing &&
            _source_alias(array, (owned.quadrature.nodes, owned.quadrature.weights))
    )
end

@inline function _source_alias(array, owned::ModalPlasma)
    return _source_alias(array, (owned.channels, owned.drude, owned.loss)) ||
        (owned.rule !== nothing && _source_alias(array, owned.rule.taps))
end

@inline function _source_alias(array, owned::ModalSourceData)
    return _source_alias(
        array, (
            owned.grid, owned.map, owned.prefactor, owned.kerr, owned.raman, owned.plasma,
        )
    )
end

@inline function _source_alias(array, owned::_SourceAnalytic)
    return _source_alias(array, (owned.modal, owned.spectrum, owned.nodes))
end

@inline function _source_alias(array, owned::_SourceRamanWork)
    return _source_alias(
        array, (
            owned.time_kernel, owned.spectrum, owned.rates, owned.weights,
        )
    )
end

@inline function _source_alias(array, owned::_SourceRamanBuffers)
    return _source_alias(array, (owned.drive, owned.frequency))
end

@inline function _source_alias(array, owned::FieldPlasmaWorkspace)
    return _source_alias(
        array, (
            owned.rate, owned.exposure, owned.fraction, owned.survival,
            owned.integrand, owned.current, owned.filtered, owned.status,
        )
    )
end

@inline function _source_alias(array, owned::EnvelopePlasmaWorkspace)
    return _source_alias(
        array, (
            owned.rate, owned.exposure, owned.fraction, owned.survival,
            owned.loss_moment, owned.status,
        )
    )
end

@inline function _source_alias(array, owned::_SourcePlasmaWork)
    return _source_alias(
        array, (
            owned.workspace, owned.first, owned.second, owned.projected_first,
            owned.projected_second, owned.coarse_first, owned.coarse_second,
            owned.factors_first, owned.factors_second, owned.stage,
        )
    )
end

"""
    ModalSource(execution, data::ModalSourceData; node_chunk)

Prepare owned binary64 CPU work for a scalar modal field or envelope source.

`node_chunk` is a positive maximum node count per block. Modal time/spectral arrays
use `(n_time,n_modes)` and `(n_omega,n_modes)`; nodal arrays use `(n_time,node_chunk)`.
The callable `source(out,state,z)` returns `nothing`, writes the complete derivative
in state units per metre, and charges `calls` before any validation or physical work.
Both caller arrays must be disjoint from this source's retained numerical inputs and
work, as well as from each other; direct arrays, views, foreign offset buffers and
reinterpreted aliases refuse before any numerical write. Mutable gather indices are
part of their array's retained storage. It never changes `state`. All FFTs use one
thread and numerical launches finish before return. Warmed calls allocate nothing.
Every inactive chunk cell is overwritten.

Data records may be shared; all mutable stage values, plans and work belong to this
instance. `ready` is true only after a successful call. When an embedded projection
is supplied, `quadrature_fine` and `quadrature_coarse` hold the successful call's
material source without the common temporal absorber, for the session's alarm reading.
They may be partially overwritten after a failed call and are invalid unless `ready`.
No controller, monitor decision, step finishing or Python callback runs here.

Throw `ArgumentError` for unsupported execution or chunk sizes. Calls throw
`NonfiniteSourceError`, `SourcePlasmaError`, `ArgumentError` or `DimensionMismatch`
outside scalar kernels for invalid values, profiles or layouts.
"""
mutable struct ModalSource{E, D, P, F, A, T, C, N, R, B, L, Q, H}
    execution::E
    data::D
    padding::P
    transform::F
    modal_spectrum::A
    modal_time::T
    nodes::T
    polarisation::T
    projected::T
    projection_work::T
    coarse_projected::C
    chunks::N
    analytic::R
    raman::B
    raman_buffers::L
    plasma::Q
    prefactor::A
    prefactor_weights::Vector{Float64}
    quadrature_fine::H
    quadrature_coarse::H
    failures::Vector{Int}
    calls::Int
    temporal_active::Bool
    ready::Bool
end

function _source_caller_alias(array, source::ModalSource)
    # The chunk views refer to map arrays already visited here. FFT plans retain
    # executable metadata, not additional caller-addressable numerical storage.
    return _source_alias(
        array, (
            source.data, source.padding.crop, source.padding.uncrop,
            source.modal_spectrum, source.modal_time, source.nodes, source.polarisation,
            source.projected, source.projection_work, source.coarse_projected,
            source.analytic, source.raman, source.raman_buffers, source.plasma,
            source.prefactor, source.prefactor_weights, source.quadrature_fine,
            source.quadrature_coarse, source.failures,
        )
    )
end

function _source_analytic(execution, data, modes, capacity)
    grid = data.grid
    needed = grid.kind == :field && (
        (
            data.kerr !== nothing &&
                data.kerr.coefficients isa KerrCoefficients{Float64, :analytic}
        ) ||
            any(component -> !component.thg, data.raman)
    )
    needed || return nothing
    modal = zeros(ComplexF64, grid.n_time, modes)
    spectrum = similar(modal)
    nodes = zeros(ComplexF64, grid.n_time, capacity)
    return _SourceAnalytic(modal, spectrum, nodes, TemporalFFT(execution, modal, spectrum))
end

function _source_raman(execution, data, layout)
    isempty(data.raman) && return (), nothing
    drive = zeros(Float64, 2layout.n_time, layout.n_nodes)
    frequency = zeros(ComplexF64, layout.n_time + 1, layout.n_nodes)
    transform = TemporalFFT(execution, drive, frequency)
    buffers = _SourceRamanBuffers(drive, frequency, transform)
    components = map(data.raman) do component
        time_kernel = zeros(Float64, 2layout.n_time)
        spectrum = zeros(ComplexF64, layout.n_time + 1)
        forward = FFTW.plan_rfft(
            time_kernel; flags = FFTW.ESTIMATE | FFTW.PRESERVE_INPUT, num_threads = 1
        )
        convolution = RamanConvolution(
            layout.n_time, spectrum, transform.forward, transform.inverse;
            inverse_scale = transform.inverse_scale,
        )
        return _SourceRamanWork(
            component, RamanDrive(layout; kind = data.grid.kind, thg = component.thg),
            time_kernel, spectrum, zeros(Float64, length(component.decay_profiles)),
            fill(component.fraction, length(component.decay_profiles)),
            forward, convolution, 0.0,
        )
    end
    return components, buffers
end

function _source_plasma(
        execution, data, layout, modal;
        coarse::Bool = data.map.coarse_projection !== nothing,
    )
    data.plasma === nothing && return nothing
    grid, prepared = data.grid, data.plasma
    field = grid.kind == :field
    workspace = field ? FieldPlasmaWorkspace(execution, Float64, layout) :
        EnvelopePlasmaWorkspace(execution, Float64, layout)
    first = zeros(eltype(modal), layout.n_time, layout.n_nodes)
    second = similar(first)
    projected_first, projected_second = similar(modal), similar(modal)
    coarse_first = coarse ? similar(modal) : nothing
    coarse_second = coarse ? similar(modal) : nothing
    factors_first = zeros(ComplexF64, length(grid.omega))
    factors_second = similar(factors_first)
    for index in eachindex(grid.omega)
        if field
            # SEP 27's held-current identity is G/(i*omega)-G'/(i*omega)^2.
            # Division is evaluated only on the explicitly active positive band.
            factors_first[index] = grid.active[index] ? inv(im * grid.omega[index]) : 0
            factors_second[index] = grid.active[index] ?
                -inv((im * grid.omega[index])^2) : 0
        else
            factors_first[index] = prepared.drude[index]
            factors_second[index] = prepared.loss[index]
        end
    end
    all(isfinite, factors_first) && all(isfinite, factors_second) ||
        throw(ArgumentError("plasma spectral assembly factors overflow at an active bin"))
    return _SourcePlasmaWork(
        workspace, first, second, projected_first, projected_second,
        coarse_first, coarse_second, factors_first, factors_second,
        zeros(Float64, 2, length(prepared.channels)),
    )
end

function ModalSource(execution::KernelExecution, data::ModalSourceData; node_chunk::Int)
    oncpu(execution) || throw(ArgumentError("this modal source currently requires CPU"))
    node_chunk > 0 || throw(ArgumentError("the modal source node chunk must be positive"))
    grid, maps = data.grid, data.map
    nodes, modes = size(maps.synthesis)
    capacity = min(node_chunk, nodes)
    layout = ResponseLayout(grid.n_time, 1, capacity)
    scalar = grid.kind == :field ? Float64 : ComplexF64
    frequency_count = grid.kind == :field ? grid.n_time ÷ 2 + 1 : grid.n_time
    modal = zeros(scalar, grid.n_time, modes)
    spectrum = zeros(ComplexF64, frequency_count, modes)
    nodal = zeros(scalar, grid.n_time, capacity)
    chunks = map(1:capacity:nodes) do first
        last = min(first + capacity - 1, nodes)
        rows = first:last
        return _SourceChunk(
            first, length(rows), view(maps.synthesis, rows, :),
            view(maps.projection, :, rows), maps.coarse_projection === nothing ? nothing :
                view(maps.coarse_projection, :, rows),
        )
    end
    padding = SpectralPadding(execution, grid.crop_index, frequency_count, 1.0, 1.0)
    raman, buffers = _source_raman(execution, data, layout)
    coarse = maps.coarse_projection !== nothing
    coarse_projected = coarse ? similar(modal) : nothing
    coarse_source = coarse ? zeros(ComplexF64, data.prefactor.shape) : nothing
    fine_source = coarse ? similar(coarse_source) : nothing
    n_weights = data.prefactor.weights === nothing ? 0 :
        size(data.prefactor.weights.coefficients, 1)
    return ModalSource(
        execution, data, padding, TemporalFFT(execution, modal, spectrum), spectrum,
        modal, nodal, similar(nodal), similar(modal), similar(modal), coarse_projected,
        chunks, _source_analytic(execution, data, modes, capacity), raman, buffers,
        _source_plasma(execution, data, layout, modal),
        zeros(ComplexF64, data.prefactor.shape), zeros(Float64, n_weights),
        fine_source, coarse_source, zeros(Int, max(capacity, modes)), 0,
        grid.temporal_rate !== nothing, false,
    )
end

"""
    set_temporal_source!(source::ModalSource, active::Bool)

Enable or quiet an already prepared in-stage temporal absorber without rebuilding it.

The session's monitor owns this decision. A rate cannot be enabled if none was
prepared; that raises `ArgumentError`. Split-step absorption is a separate session
finish operation. Return `nothing`; this toggle performs no source evaluation.
"""
function set_temporal_source!(source::ModalSource, active::Bool)
    active && source.data.grid.temporal_rate === nothing &&
        throw(ArgumentError("no in-stage temporal source rate was prepared"))
    source.temporal_active = active
    source.ready = false
    return nothing
end

# Independent profiles may have different concrete constant/table types. Tuple
# traversal keeps those types visible instead of boxing a dynamically indexed entry.
@inline _source_each_indexed(f, ::Tuple{}, index::Int = 1) = nothing

@inline function _source_each_indexed(
        f::F, items::Tuple{T, Vararg}, index::Int = 1
    ) where {F, T}
    f(first(items), index)
    _source_each_indexed(f, Base.tail(items), index + 1)
    return nothing
end

function _prepare_source_raman!(work, position, execution)
    component = work.data
    density = _source_profile(component.density, position)
    density >= 0 || throw(ArgumentError("Raman amplitude density must be nonnegative"))
    work.density = density
    _source_each_indexed(component.decay_profiles) do profile, index
        rate = _source_profile(profile, position)
        rate >= 0 || throw(ArgumentError("Raman decay rates must be nonnegative"))
        work.rates[index] = rate
    end
    rebuild_raman!(
        work.time_kernel, work.spectrum, component.kernel, work.rates, work.weights,
        work.forward, execution,
    )
    return nothing
end

function _prepare_source_profiles!(source, position)
    data, execution = source.data, source.execution
    taper = _source_profile(data.map.taper, position)
    taper > 0 || throw(ArgumentError("modal taper scale must be positive"))
    kerr_density = data.kerr === nothing ? 0.0 :
        _source_profile(data.kerr.density, position)
    _source_prefactor!(
        source.prefactor, data.prefactor, position, source.prefactor_weights, execution
    )
    _source_finite!(source.failures, source.prefactor, execution, :prefactor)
    foreach(work -> _prepare_source_raman!(work, position, execution), source.raman)
    if data.plasma !== nothing
        _source_each_indexed(data.plasma.channels) do channel, index
            density = _source_profile(channel.density, position)
            initial = _source_profile(channel.preionisation, position)
            density >= 0 && 0 <= initial <= 1 ||
                throw(ArgumentError("invalid plasma density or preionisation profile"))
            source.plasma.stage[1, index] = density
            source.plasma.stage[2, index] = initial
        end
    end
    return taper, kerr_density
end

function _source_analytic!(analytic, spectrum, execution)
    full = analytic.spectrum
    half = size(spectrum, 1)
    each!(execution, size(full)...) do row, column
        full[row, column] = if row > half
            zero(eltype(full))
        elseif row == 1 || row == half
            complex(real(spectrum[row, column]), 0.0)
        else
            2spectrum[row, column]
        end
    end
    temporal_inverse!(analytic.modal, analytic.transform, full, execution)
    return nothing
end

function _source_scale!(array, scale, execution)
    each!(execution, size(array)...) do row, column
        array[row, column] *= scale
    end
    return nothing
end

function _source_project_add!(output, scratch, nodal, projection, taper, count, execution)
    project!(scratch, nodal, projection, execution; n_components = 1, active_nodes = count)
    each!(execution, size(output)...) do row, column
        output[row, column] += scratch[row, column] / taper
    end
    return nothing
end

function _source_raman_response!(source, work, count)
    buffers, execution = source.raman_buffers, source.execution
    analytic = source.analytic === nothing ? nothing : source.analytic.nodes
    raman_drive!(
        buffers.drive, source.nodes, work.drive, execution;
        analytic_samples = analytic, count,
    )
    convolve_raman!(buffers.drive, buffers.frequency, work.convolution, execution)
    raman_polarisation!(
        source.polarisation, source.nodes, buffers.drive, work.drive.layout, execution;
        factor = work.data.material_factor * work.density, count, add = true,
    )
    return nothing
end

function _source_material!(source, kerr_density, count)
    data, execution = source.data, source.execution
    layout = ResponseLayout(size(source.nodes, 1), 1, size(source.nodes, 2))
    fill!(source.polarisation, 0)
    if data.kerr !== nothing
        coefficients = data.kerr.coefficients
        samples = if data.grid.kind == :field &&
                coefficients isa KerrCoefficients{Float64, :analytic}
            if source.analytic === nothing
                message = "analytic field Kerr requires prepared analytic storage"
                throw(ArgumentError(message))
            end
            source.analytic.nodes
        else
            source.nodes
        end
        kerr_polarisation!(
            source.polarisation, samples, layout, coefficients, execution;
            density = kerr_density, carrier = data.grid.carrier, count,
        )
    end
    foreach(work -> _source_raman_response!(source, work, count), source.raman)
    _source_finite!(source.failures, source.polarisation, execution, :polarisation)
    apply_weights!(
        source.polarisation, source.polarisation, data.grid.source_window, execution
    )
    return nothing
end

function _source_plasma_channel!(source, channel, index, count, first)
    data, work, execution = source.data, source.plasma, source.execution
    density, initial = work.stage[1, index], work.stage[2, index]
    if work.workspace isa FieldPlasmaWorkspace
        field_plasma!(
            work.first, work.second, source.nodes, channel.channel, data.plasma.rule,
            work.workspace, execution; density, preionisation = initial,
            window = data.grid.source_window,
            derivative = data.grid.source_window_derivative,
            count, add = index != 1,
        )
    else
        envelope_plasma!(
            work.first, work.second, source.nodes, channel.channel, channel.quadrature,
            work.workspace, execution; density, preionisation = initial,
            dt = data.grid.dt, window = data.grid.source_window, count, add = index != 1,
        )
    end
    failure = plasma_failure(work.workspace.status)
    failure.code == PlasmaSuccess || throw(
        SourcePlasmaError(
            failure.code, index, first + failure.node - 1, failure.sample
        )
    )
    _source_finite!(source.failures, work.first, execution, :plasma_first)
    _source_finite!(source.failures, work.second, execution, :plasma_second)
    return nothing
end

function _source_plasma_chunk!(source, chunk, taper)
    work, execution = source.plasma, source.execution
    work === nothing && return nothing
    _source_each_indexed(source.data.plasma.channels) do channel, index
        _source_plasma_channel!(source, channel, index, chunk.count, chunk.first)
    end
    for (nodal, projected, coarse) in (
            (work.first, work.projected_first, work.coarse_first),
            (work.second, work.projected_second, work.coarse_second),
        )
        _source_project_add!(
            projected, source.projection_work, nodal, chunk.projection,
            taper, chunk.count, execution,
        )
        if coarse !== nothing
            _source_project_add!(
                coarse, source.projection_work, nodal, chunk.coarse_projection,
                taper, chunk.count, execution,
            )
        end
    end
    return nothing
end

function _clear_source_projection!(source)
    fill!(source.projected, 0)
    source.coarse_projected === nothing || fill!(source.coarse_projected, 0)
    if source.plasma !== nothing
        work = source.plasma
        fill!(work.projected_first, 0)
        fill!(work.projected_second, 0)
        work.coarse_first === nothing || fill!(work.coarse_first, 0)
        work.coarse_second === nothing || fill!(work.coarse_second, 0)
    end
    return nothing
end

function _source_over_nodes!(source, taper, kerr_density)
    execution = source.execution
    _clear_source_projection!(source)
    for chunk in source.chunks
        synthesize!(
            source.nodes, source.modal_time, chunk.synthesis, execution;
            n_components = 1, active_nodes = chunk.count,
        )
        _source_scale!(source.nodes, taper, execution)
        _source_finite!(source.failures, source.nodes, execution, :nodal_field)
        if source.analytic !== nothing
            synthesize!(
                source.analytic.nodes, source.analytic.modal, chunk.synthesis, execution;
                n_components = 1, active_nodes = chunk.count,
            )
            _source_scale!(source.analytic.nodes, taper, execution)
            _source_finite!(
                source.failures, source.analytic.nodes, execution, :analytic_field
            )
        end
        _source_material!(source, kerr_density, chunk.count)
        _source_project_add!(
            source.projected, source.projection_work, source.polarisation,
            chunk.projection, taper, chunk.count, execution,
        )
        if source.coarse_projected !== nothing
            _source_project_add!(
                source.coarse_projected, source.projection_work, source.polarisation,
                chunk.coarse_projection, taper, chunk.count, execution,
            )
        end
        _source_plasma_chunk!(source, chunk, taper)
    end
    return nothing
end

function _source_add_stream!(output, source, projected, factor)
    execution = source.execution
    _source_finite!(source.failures, projected, execution, :projected_plasma)
    temporal_forward!(source.modal_spectrum, source.transform, projected)
    spectrum, crop = source.modal_spectrum, source.padding.crop
    each!(execution, size(output)...) do row, column
        output[row, column] += factor[row] * spectrum[crop[row], column]
    end
    return nothing
end

function _source_assemble!(output, source, projected, first, second)
    execution = source.execution
    _source_finite!(source.failures, projected, execution, :projected_polarisation)
    temporal_forward!(source.modal_spectrum, source.transform, projected)
    crop_spectrum!(output, source.modal_spectrum, source.padding, execution)
    if source.plasma !== nothing
        _source_add_stream!(output, source, first, source.plasma.factors_first)
        _source_add_stream!(output, source, second, source.plasma.factors_second)
    end
    _source_finite!(source.failures, output, execution, :combined_polarisation)
    prefactor, window, active = source.prefactor, source.data.grid.omega_window,
        source.data.grid.active
    each!(execution, size(output)...) do row, column
        output[row, column] = active[row] ?
            prefactor[row, column] * window[row] * output[row, column] : 0
    end
    _source_finite!(source.failures, output, execution, :material_source)
    return nothing
end

function _source_temporal!(output, source)
    execution = source.execution
    rate, modal, scratch = source.data.grid.temporal_rate, source.modal_time,
        source.projection_work
    rate === nothing &&
        throw(ArgumentError("an active temporal source requires a prepared rate"))
    each!(execution, size(scratch)...) do row, column
        scratch[row, column] = -0.5 * rate[row] * modal[row, column]
    end
    _source_finite!(source.failures, scratch, execution, :temporal_absorber)
    temporal_forward!(source.modal_spectrum, source.transform, scratch)
    active, crop, spectrum = source.data.grid.active, source.padding.crop,
        source.modal_spectrum
    each!(execution, size(output)...) do row, column
        output[row, column] = active[row] ?
            output[row, column] + spectrum[crop[row], column] : 0
    end
    return nothing
end

function (source::ModalSource)(output, state, position::Float64)
    source.calls += 1
    source.ready = false
    size(state) == size(output) == source.data.prefactor.shape ||
        throw(DimensionMismatch("modal source input/output does not match preparation"))
    eltype(state) === eltype(output) === ComplexF64 ||
        throw(ArgumentError("modal source input/output requires complex binary64 storage"))
    isfinite(position) || throw(ArgumentError("a source position must be finite"))
    _transfer_arrays_alias(output, state) &&
        throw(ArgumentError("a modal source must not overwrite its input state"))
    (_source_caller_alias(output, source) || _source_caller_alias(state, source)) &&
        throw(ArgumentError("caller arrays must be disjoint from source inputs and work"))
    execution, data = source.execution, source.data
    _source_finite!(source.failures, state, execution, :input)
    material = data.kerr !== nothing || !isempty(data.raman) || data.plasma !== nothing
    if material || source.temporal_active
        pad_spectrum!(source.modal_spectrum, state, source.padding, execution)
        temporal_inverse!(
            source.modal_time, source.transform, source.modal_spectrum, execution
        )
        _source_finite!(source.failures, source.modal_time, execution, :modal_field)
    end
    if material
        taper, kerr_density = _prepare_source_profiles!(source, position)
        source.analytic === nothing ||
            _source_analytic!(source.analytic, source.modal_spectrum, execution)
        _source_over_nodes!(source, taper, kerr_density)
        plasma = source.plasma
        _source_assemble!(
            output, source, source.projected,
            plasma === nothing ? nothing : plasma.projected_first,
            plasma === nothing ? nothing : plasma.projected_second,
        )
        if source.quadrature_coarse !== nothing
            copyto!(source.quadrature_fine, output)
            _source_assemble!(
                source.quadrature_coarse, source, source.coarse_projected,
                plasma === nothing ? nothing : plasma.coarse_first,
                plasma === nothing ? nothing : plasma.coarse_second,
            )
        end
    else
        fill!(output, 0)
        source.quadrature_fine === nothing || fill!(source.quadrature_fine, 0)
        source.quadrature_coarse === nothing || fill!(source.quadrature_coarse, 0)
    end
    source.temporal_active && _source_temporal!(output, source)
    _source_finite!(source.failures, output, execution, :source)
    source.ready = true
    return nothing
end
