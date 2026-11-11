# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public SourceProfile, SourcePrefactor, ModalSourceGrid, ModalSourceMap
public ModalKerr, ModalRaman, ModalPlasmaChannel, ModalPlasma, ModalSourceData
public ModalSource, SourcePlasmaError, set_temporal_source!

# Device admission precedes uploads and workspace allocation. Scalar material
# profiles remain host data; only numerical arrays enter device captures.
function _source_execution_arithmetic(execution, arithmetic)
    oncpu(execution) || source_type(arithmetic) === Float64 ||
        throw(ArgumentError("device modal sources currently require binary64"))
    return nothing
end

function _source_host_table(table::ChebyshevTable)
    KernelAbstractions.get_backend(table.breakpoints) isa KernelAbstractions.CPU &&
        KernelAbstractions.get_backend(table.coefficients) isa KernelAbstractions.CPU ||
        throw(ArgumentError("scalar source profiles require host table storage"))
    return nothing
end

function _source_require_host(array::AbstractArray, name::Symbol)
    KernelAbstractions.get_backend(array) isa KernelAbstractions.CPU ||
        throw(ArgumentError("$name requires host preparation storage"))
    return nothing
end

function _source_preparation_host(
        array::AbstractArray{T, N}, execution::KernelExecution,
    ) where {T, N}
    oncpu(execution) && return array
    owned = Array{T, N}(undef, size(array))
    copy_host!(owned, array, execution)
    return owned
end

function _copy_source_table(execution, table::ChebyshevTable)
    return ChebyshevTable(
        upload(execution, table.breakpoints), upload(execution, table.coefficients),
        table.lower, table.upper,
    )
end

@inline function _source_copy!(destination, source, execution)
    oncpu(execution) ? copyto!(destination, source) :
        copy_device!(destination, source, execution)
    return nothing
end

function _source_zero!(array, execution)
    if oncpu(execution)
        fill!(array, 0)
    else
        each!(execution, length(array), 1) do index, unused
            array[index] = zero(eltype(array))
        end
    end
    return nothing
end

"""
    SourceProfile(value::Float64; name=:profile)
    SourceProfile(table::ChebyshevTable; name=:profile)

Own one independent scalar, direct-value material or taper profile.

The constant or scalar binary64 Chebyshev table keeps its translated SI units and
admitted position span. A host table is copied, including its breakpoints. It is never
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
    _source_host_table(table)
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
    SourcePrefactor(execution, values::AbstractMatrix{ComplexF64}; arithmetic)
    SourcePrefactor(table::ChebyshevTable, shape::Tuple{Int,Int}; arithmetic)
    SourcePrefactor(execution, table::ChebyshevTable, shape::Tuple{Int,Int}; arithmetic)
    SourcePrefactor(execution, terms::AbstractArray{ComplexF64,3}, weights; arithmetic)

Own the translated modal source prefactor, including its existing crop scale.

Constant values have shape `(n_omega,n_modes)`. A complex binary64 table contains
`prod(shape)` values in column-major order. Expansion terms have shape
`(n_omega,n_modes,n_terms)` and their real binary64 weight table contains `n_terms`
direct values. Neither table nor weights are cumulative. All arrays/tables are copied.
The execution-aware table constructor copies numerical storage into its owned context;
the two-argument table constructor retains its existing CPU copy behavior. Scalar
material profiles use the separate host-only `SourceProfile` record.
`arithmetic` defaults to `SourceArithmetic(Float64)`. Constants narrow once at
preparation, with a named range fact. Varying tables/terms retain their original
binary64 coefficients: each stage evaluates in binary64, admits the complete value
range, then narrows once into response-precision work. No binary32 Clenshaw or
state-sized mutable binary64 scratch replaces that represented evaluation.
Throw `ArgumentError` or `DimensionMismatch` for invalid types, sizes or values.
"""
struct SourcePrefactor{D, W, A, R}
    data::D
    weights::W
    shape::Tuple{Int, Int}
    arithmetic::A
    ranges::R
end

function SourcePrefactor(
        execution, values::AbstractMatrix{ComplexF64};
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    _source_execution_arithmetic(execution, arithmetic)
    !isempty(values) && all(isfinite, values) ||
        throw(ArgumentError("constant source prefactors must be finite and nonempty"))
    owned, range = _source_prepare(execution, values, arithmetic, :source_prefactor)
    return SourcePrefactor(owned, nothing, size(values), arithmetic, (values = range,))
end

function SourcePrefactor(
        execution::KernelExecution, table::ChebyshevTable, shape::Tuple{Int, Int};
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    _source_execution_arithmetic(execution, arithmetic)
    eltype(table.coefficients) === ComplexF64 &&
        eltype(table.breakpoints) === Float64 && all(>(0), shape) ||
        throw(ArgumentError("source prefactor tables require complex binary64 values"))
    prod(shape) == size(table.coefficients, 1) ||
        throw(DimensionMismatch("source prefactor table does not match its shape"))
    return SourcePrefactor(
        _copy_source_table(execution, table), nothing, shape, arithmetic, ()
    )
end

function SourcePrefactor(
        table::ChebyshevTable, shape::Tuple{Int, Int};
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    KernelAbstractions.get_backend(table.coefficients) isa KernelAbstractions.CPU &&
        KernelAbstractions.get_backend(table.breakpoints) isa KernelAbstractions.CPU ||
        throw(ArgumentError("device prefactor tables require their prepared execution"))
    eltype(table.coefficients) === ComplexF64 &&
        eltype(table.breakpoints) === Float64 && all(>(0), shape) ||
        throw(ArgumentError("source prefactor tables require complex binary64 values"))
    prod(shape) == size(table.coefficients, 1) ||
        throw(DimensionMismatch("source prefactor table does not match its shape"))
    return SourcePrefactor(_copy_source_table(table), nothing, shape, arithmetic, ())
end

function SourcePrefactor(
        execution, terms::AbstractArray{ComplexF64, 3}, weights::ChebyshevTable,
        ; arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    _source_execution_arithmetic(execution, arithmetic)
    !isempty(terms) && all(isfinite, terms) &&
        eltype(weights.coefficients) === Float64 &&
        eltype(weights.breakpoints) === Float64 ||
        throw(ArgumentError("source prefactor expansion requires finite binary64 data"))
    size(terms, 3) == size(weights.coefficients, 1) ||
        throw(DimensionMismatch("source prefactor terms and weights disagree"))
    return SourcePrefactor(
        upload(execution, terms), _copy_source_table(execution, weights),
        size(terms)[1:2], arithmetic, (),
    )
end

@inline function _source_prefactor_value(prefactor, weights, index, position)
    if prefactor.weights !== nothing
        total = zero(ComplexF64)
        terms = prefactor.data
        count = prod(prefactor.shape)
        for term in axes(terms, 3)
            total += weights[term] * terms[index + (term - 1) * count]
        end
        return total
    end
    table = prefactor.data
    return _table_value(table.coefficients, table.breakpoints, index, position)
end

function _source_prefactor!(output, prefactor, position, weight_work, execution)
    if source_type(prefactor.arithmetic) === Float32
        if prefactor.weights !== nothing
            evaluate_table!(weight_work, prefactor.weights, position, execution)
        elseif prefactor.data isa ChebyshevTable
            _check_table_position(prefactor.data, position)
        else
            _source_copy!(output, prefactor.data, execution)
            return prefactor.ranges.values
        end
        # Wide scalar evaluation is host preparation, not Float32 Clenshaw with
        # rounded coefficients. Re-evaluate after range admission rather than own
        # a state-sized mutable binary64 staging array.
        values = (
            _source_prefactor_value(prefactor, weight_work, index, position)
                for index in eachindex(output)
        )
        fact, _ = _source_range(values, Float32, :source_prefactor, 0, false)
        for index in eachindex(output)
            output[index] = _source_prefactor_value(prefactor, weight_work, index, position)
        end
        return fact
    end
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
        _source_copy!(output, prefactor.data, execution)
    end
    return nothing
end

"""
    ModalSourceGrid(execution; kind, n_time, dt, crop_index, omega, active,
                    omega_window, source_window, source_window_derivative,
                    temporal_rate=nothing, carrier=nothing,
                    arithmetic=SourceArithmetic(Float64))

Own temporal/spectral data at the declared scalar source precision.

`kind` is `:field` or `:envelope`; `n_time` is the even fine temporal count and `dt`
its spacing in seconds. `crop_index` is the owned host zero-based coarse-to-fine
map. Coarse `omega` is in rad/s; every active bin is positive. Boolean `active` and
dimensionless `omega_window` have the coarse extent. Fine `source_window` is in
`[0,1]`, its supplied time derivative is in s⁻¹, and optional `temporal_rate` is a
nonnegative power-loss rate in m⁻¹ for the in-stage absorber. Optional complex
`carrier` is the translated envelope harmonic multiplier. Padding/cropping scales
are deliberately absent: synthesis and prefactor already contain them.
Translated inputs are binary64. The arithmetic record is retained; windows, rates
and carrier are narrowed once with named `ranges`, while `dt` and physical `omega`
remain host binary64 preparation metadata. Boolean masks retain exact storage.

Throw `ArgumentError` or `DimensionMismatch` for invalid numerical data or layout.
"""
struct ModalSourceGrid{I, B, W, T, C, A, R}
    kind::Symbol
    n_time::Int
    dt::Float64
    crop_index::I
    omega::Vector{Float64}
    active::B
    omega_window::W
    source_window::W
    source_window_derivative::W
    temporal_rate::T
    carrier::C
    arithmetic::A
    ranges::R
end

function ModalSourceGrid(
        execution; kind::Symbol, n_time::Int, dt::Float64, crop_index,
        omega::AbstractVector{Float64}, active::AbstractVector{Bool},
        omega_window::AbstractVector{Float64}, source_window::AbstractVector{Float64},
        source_window_derivative::AbstractVector{Float64}, temporal_rate = nothing,
        carrier = nothing, arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    _source_execution_arithmetic(execution, arithmetic)
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
    host_execution = KernelExecution(KernelAbstractions.CPU(), 1)
    padding = SpectralPadding(host_execution, crop_index, n_frequency, 1.0, 1.0)
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
    spectral, spectral_range = _source_prepare(
        execution, omega_window, arithmetic, :omega_window
    )
    window, window_range = _source_prepare(
        execution, source_window, arithmetic, :source_window
    )
    derivative, derivative_range = _source_prepare(
        execution, source_window_derivative, arithmetic, :source_window_derivative
    )
    rate, rate_range = temporal_rate === nothing ? (nothing, nothing) :
        _source_prepare(execution, temporal_rate, arithmetic, :temporal_rate)
    harmonic, harmonic_range = carrier === nothing ? (nothing, nothing) :
        _source_prepare(execution, carrier, arithmetic, :harmonic_carrier)
    ranges = (
        omega_window = spectral_range, source_window = window_range,
        source_window_derivative = derivative_range, temporal_rate = rate_range,
        carrier = harmonic_range,
    )
    return ModalSourceGrid(
        kind, n_time, dt, padding.crop .- 1, copy(omega), upload(execution, active),
        spectral, window, derivative, rate, harmonic, arithmetic, ranges,
    )
end

"""
    ModalSourceMap(execution; synthesis, projection, coarse_projection=nothing,
                   n_state_components=1, n_physical_components=1,
                   taper=SourceProfile(1.0; name=:taper),
                   arithmetic=SourceArithmetic(Float64))

Own the static scalar modal maps and their independent positive taper profile.

`synthesis[node,mode]` contains translated physical-amplitude and pad scales;
`projection[mode,node]` contains quadrature weights. Optional `coarse_projection`
has the same shape and supplies the embedded quadrature reading. At each position,
synthesis is multiplied by the dimensionless taper and projection divided by it.
The two declared component counts must both be one; vector polarisation refuses by
name. General or changing modal bases are not represented by this record.
Synthesis folds `2^(state_exponent-field_exponent)` and projection its reciprocal
before the one narrowing. `arithmetic` and named `ranges` remain explicit. Original
binary64 `host_synthesis`, `host_projection` and `host_coarse_projection` are retained
for wide preparation of composed operators; they alias their prepared maps on binary64
CPU, and are separate host copies when the numerical maps belong to a device.
Throw `ArgumentError` or `DimensionMismatch` before workspace allocation for bad data.
"""
struct ModalSourceMap{S, P, C, T, HS, HP, HC, A, R}
    synthesis::S
    projection::P
    coarse_projection::C
    n_state_components::Int
    n_physical_components::Int
    taper::T
    host_synthesis::HS
    host_projection::HP
    host_coarse_projection::HC
    arithmetic::A
    ranges::R
end

function ModalSourceMap(
        execution; synthesis::AbstractMatrix{Float64}, projection::AbstractMatrix{Float64},
        coarse_projection = nothing, n_state_components::Int = 1,
        n_physical_components::Int = 1,
        taper::SourceProfile = SourceProfile(1.0; name = :taper),
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    _source_execution_arithmetic(execution, arithmetic)
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
    shift = arithmetic.state_exponent - arithmetic.field_exponent
    forward, forward_range = _source_prepare(
        execution, synthesis, arithmetic, :source_synthesis; exponent = shift
    )
    backward, backward_range = _source_prepare(
        execution, projection, arithmetic, :source_projection; exponent = -shift
    )
    embedded, embedded_range = coarse_projection === nothing ? (nothing, nothing) :
        _source_prepare(
            execution, coarse_projection, arithmetic, :source_embedded;
            exponent = -shift
        )
    host_alias = source_type(arithmetic) === Float64 && oncpu(execution)
    return ModalSourceMap(
        forward, backward, embedded, n_state_components, n_physical_components, taper,
        host_alias ? forward : copy(synthesis), host_alias ? backward : copy(projection),
        host_alias || coarse_projection === nothing ? embedded : copy(coarse_projection),
        arithmetic, (
            synthesis = forward_range, projection = backward_range,
            coarse_projection = embedded_range,
        ),
    )
end

"""
    ModalKerr(coefficients::KerrCoefficients{Float64}, density::SourceProfile;
              arithmetic=SourceArithmetic(Float64))

Bind translated cubic coefficients to their independent stage weight.

The coefficient already contains its translated material factor; the profile is
applied once. A mixture's weighted susceptibility profile need not be a number density
or nonnegative. Grid/form/arithmetic compatibility is checked by `ModalSourceData`.
Retain binary64 coefficients and profiles; each stage combines density and the
physical-field square exponent before one response-precision rounding.
"""
struct ModalKerr{K, D, A}
    coefficients::K
    density::D
    arithmetic::A
    function ModalKerr(
            coefficients::KerrCoefficients{Float64}, density::SourceProfile;
            arithmetic::SourceArithmetic = SourceArithmetic(Float64),
        )
        return new{typeof(coefficients), typeof(density), typeof(arithmetic)}(
            coefficients, density, arithmetic
        )
    end
end

"""
    ModalRaman(kernel, decay_profiles::Tuple, density::SourceProfile;
               fraction, material_factor, thg=true, tensor=false,
               arithmetic=SourceArithmetic(Float64))

Own one scalar Raman component and its independent linewidth/amplitude profiles.

`kernel` is a binary64 `RamanKernel` without epsilon₀. Each row has its own scalar
nonnegative decay-rate profile in s⁻¹; `density` is its nonnegative amplitude weight.
`fraction` is in `(0,1]`; `material_factor` is the positive translated epsilon₀ factor.
These three weights each enter once. `thg` selects the field drive; an envelope always
uses half its squared magnitude. `tensor=true` refuses tensor Raman by name.
Binary32 work reconstructs the kernel in host binary64, normalizes its spectrum to
response precision, and retains the block exponent in the material coefficient.
No tiny kernel is silently rounded away before that exponent is applied.
Throw `ArgumentError` or `DimensionMismatch` for unsupported or inconsistent inputs.
"""
struct ModalRaman{K, R, D, A}
    kernel::K
    decay_profiles::R
    density::D
    fraction::Float64
    material_factor::Float64
    thg::Bool
    arithmetic::A
end

function ModalRaman(
        kernel::RamanKernel{Float64}, decay_profiles::Tuple, density::SourceProfile;
        fraction::Float64, material_factor::Float64, thg::Bool = true, tensor::Bool = false,
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    tensor && throw(ArgumentError("Julia modal source refuses tensor Raman"))
    _source_require_host(kernel.undamped, :raman_coefficients)
    length(decay_profiles) == size(kernel.undamped, 2) &&
        all(profile -> profile isa SourceProfile, decay_profiles) ||
        throw(DimensionMismatch("each Raman row needs its own scalar decay profile"))
    isfinite(fraction) && 0 < fraction <= 1 &&
        isfinite(material_factor) && material_factor > 0 ||
        throw(ArgumentError("invalid Raman fraction or translated material factor"))
    return ModalRaman(
        RamanKernel(copy(kernel.undamped), kernel.dt), decay_profiles, density,
        fraction, material_factor, thg, arithmetic,
    )
end

"""
    ModalPlasmaChannel(channel, density, preionisation; quadrature=nothing,
                       arithmetic=SourceArithmetic(Float64))
    ModalPlasmaChannel(execution::KernelExecution, channel, density, preionisation;
                       quadrature=nothing, arithmetic=SourceArithmetic(Float64))

Own one plasma channel with host profiles and declared response precision.

`channel` is the translated `PlasmaChannel`; its rate spline is copied. Stage density
is in m⁻³ and nonnegative; preionisation is in `[0,1]`. An envelope requires its own
translated `CycleQuadrature`, whose arrays are copied. A field uses no cycle rule.
These profile ranges are checked at each source position before launching kernels.
The input channel/rule are binary64; coefficient, node and weight arrays narrow once
with named `ranges`. Physical table bounds and material coefficients stay binary64
until combined with the field exponent. Refinement tolerances round down if necessary
so conversion never enlarges the supplied physical tolerance.
The execution overload uploads freshly validated host arrays into the selected owned
context. Device execution currently requires binary64. Scalar profiles remain host data.
"""
struct ModalPlasmaChannel{C, D, P, Q, A, R}
    channel::C
    density::D
    preionisation::P
    quadrature::Q
    arithmetic::A
    ranges::R
end

function ModalPlasmaChannel(
        channel::PlasmaChannel{Float64}, density::SourceProfile,
        preionisation::SourceProfile; quadrature = nothing,
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    rate = channel.rate
    _source_require_host(rate.coefficients, :ionisation_rate_coefficients)
    coefficients, rate_range = _source_prepare_host(
        rate.coefficients, arithmetic, :ionisation_rate_coefficients
    )
    owned = IonisationRateTable(
        coefficients, rate.field_min, rate.field_max, rate.rate_max,
        rate.log_field_min, rate.d_log_field,
    )
    quadrature === nothing || quadrature isa CycleQuadrature{Float64} ||
        throw(ArgumentError("envelope plasma needs a binary64 cycle rule"))
    copied_rule, node_range, weight_range = if quadrature === nothing
        (nothing, nothing, nothing)
    else
        _source_require_host(quadrature.nodes, :plasma_cycle_nodes)
        _source_require_host(quadrature.weights, :plasma_cycle_weights)
        nodes, node_fact = _source_prepare_host(
            quadrature.nodes, arithmetic, :plasma_cycle_nodes
        )
        weights, weight_fact = _source_prepare_host(
            quadrature.weights, arithmetic, :plasma_cycle_weights
        )
        T = source_type(arithmetic)
        tolerance, floor = T(quadrature.moment_rtol), T(quadrature.negligible_rate)
        # A downward conversion preserves the supplied upper refinement budget;
        # the numerical pass does not relax a physical tolerance for binary32.
        tolerance = tolerance > quadrature.moment_rtol ? prevfloat(tolerance) : tolerance
        floor = floor > quadrature.negligible_rate ? prevfloat(floor) : floor
        rule = CycleQuadrature(
            nodes, weights, quadrature.ends;
            moment_rtol = tolerance, negligible_rate = floor,
        )
        (rule, node_fact, weight_fact)
    end
    _rate_parameters(owned, arithmetic.field_exponent)
    return ModalPlasmaChannel(
        PlasmaChannel(owned, channel.charge_mass_ratio, channel.ionisation_potential),
        density, preionisation, copied_rule, arithmetic,
        (rate = rate_range, cycle_nodes = node_range, cycle_weights = weight_range),
    )
end

function ModalPlasmaChannel(
        execution::KernelExecution, channel::PlasmaChannel{Float64},
        density::SourceProfile, preionisation::SourceProfile; quadrature = nothing,
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    _source_execution_arithmetic(execution, arithmetic)
    # Public records retain mutable arrays. Revalidate their current values in the
    # existing host constructor before uploading any data or using a clone hook.
    prepared = ModalPlasmaChannel(channel, density, preionisation; quadrature, arithmetic)
    oncpu(execution) && return prepared
    host_rate = prepared.channel.rate
    rate = IonisationRateTable(
        _PlasmaValidatedCopy(), host_rate, upload(execution, host_rate.coefficients)
    )
    host_cycle = prepared.quadrature
    cycle = host_cycle === nothing ? nothing : CycleQuadrature(
            _PlasmaValidatedCopy(), host_cycle,
            upload(execution, host_cycle.nodes), upload(execution, host_cycle.weights),
        )
    return ModalPlasmaChannel(
        PlasmaChannel(rate, channel.charge_mass_ratio, channel.ionisation_potential),
        density, preionisation, cycle, arithmetic, prepared.ranges,
    )
end

"""
    ModalPlasma(execution, channels::Tuple, rule::RunningIntegralRule{Float64}; arithmetic)
    ModalPlasma(execution, channels::Tuple; drude, loss, arithmetic)

Own the independent plasma channels and their field or envelope assembly data.

A field copies the translated running-integral rule. An envelope copies its coarse
`drude=-1/omega²` and `loss=-im/omega` factors (including a translated frozen-carrier
policy if selected upstream), already exactly zero outside its supported band. Every
envelope channel supplies a cycle rule; field channels do not. No physical constant,
frequency approximation or population profile is inferred here.
`arithmetic` defaults to binary64 and must match every channel. Taps and spectral
factors narrow once with named range facts; their pointwise use stays in that precision.
Throw `ArgumentError` or `DimensionMismatch` for incompatible channel/assembly data.
"""
struct ModalPlasma{C, R, D, L, A, F}
    channels::C
    rule::R
    drude::D
    loss::L
    arithmetic::A
    ranges::F
end

function _source_channels(channels)
    !isempty(channels) && all(channel -> channel isa ModalPlasmaChannel, channels) ||
        throw(ArgumentError("modal plasma needs at least one declared channel"))
    return nothing
end

function ModalPlasma(
        execution, channels::Tuple, rule::RunningIntegralRule{Float64};
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    _source_execution_arithmetic(execution, arithmetic)
    _source_channels(channels)
    all(channel -> channel.arithmetic == arithmetic, channels) ||
        throw(ArgumentError("plasma channels must use the same source arithmetic"))
    all(channel -> channel.quadrature === nothing, channels) ||
        throw(ArgumentError("field plasma does not use envelope cycle rules"))
    _source_require_host(rule.taps, :plasma_integral_taps)
    spacing = source_type(arithmetic) === Float64 ? rule.dt :
        source_coefficient(arithmetic, :plasma_time_spacing, rule.dt)
    owned, range = if oncpu(execution)
        taps, fact = _source_prepare(
            execution, rule.taps, arithmetic, :plasma_integral_taps
        )
        RunningIntegralRule(taps, spacing; floor_exposure = rule.floor_exposure), fact
    else
        taps, fact = _source_prepare_host(rule.taps, arithmetic, :plasma_integral_taps)
        validated = RunningIntegralRule(taps, spacing; floor_exposure = rule.floor_exposure)
        copied = upload(execution, validated.taps)
        RunningIntegralRule(_PlasmaValidatedCopy(), validated, copied), fact
    end
    return ModalPlasma(channels, owned, nothing, nothing, arithmetic, (taps = range,))
end

function ModalPlasma(
        execution, channels::Tuple; drude::AbstractVector{Float64},
        loss::AbstractVector{ComplexF64},
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    _source_execution_arithmetic(execution, arithmetic)
    _source_channels(channels)
    all(channel -> channel.arithmetic == arithmetic, channels) ||
        throw(ArgumentError("plasma channels must use the same source arithmetic"))
    all(channel -> channel.quadrature !== nothing, channels) ||
        throw(ArgumentError("each envelope plasma channel needs a cycle rule"))
    length(drude) == length(loss) ||
        throw(DimensionMismatch("envelope plasma spectral factors disagree"))
    _source_require_host(drude, :plasma_drude)
    _source_require_host(loss, :plasma_loss)
    all(value -> isfinite(value) && value <= 0, drude) &&
        all(value -> isfinite(value) && iszero(real(value)) && imag(value) <= 0, loss) ||
        throw(ArgumentError("invalid translated envelope plasma spectral factors"))
    first, first_range = _source_prepare(execution, drude, arithmetic, :plasma_drude)
    second, second_range = _source_prepare(execution, loss, arithmetic, :plasma_loss)
    return ModalPlasma(
        channels, nothing, first, second, arithmetic,
        (drude = first_range, loss = second_range)
    )
end

"""
    ModalSourceData(grid, map, prefactor; kerr=nothing, raman=(), plasma=nothing)
    ModalSourceData(execution::KernelExecution, grid, map, prefactor;
                    kerr=nothing, raman=(), plasma=nothing)

Bind independently owned immutable records for an admitted scalar modal source.

The field/envelope kind, axes, scalar cubic form and all response grids must agree.
Raman components and plasma channels remain typed tuples in translated material order.
This record contains no workspace, session policy or Python object. Input noise belongs
to initial-state preparation; modified noise, vector/tensor, general modal and changing
bases are outside this source's admitted scope.
Use the execution overload for device plasma. It validates immutable mask/spectral
coefficients through preparation-only host copies; no evolving field is downloaded.
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
    plasma === nothing ||
        KernelAbstractions.get_backend(grid.active) isa KernelAbstractions.CPU ||
        throw(ArgumentError("device plasma source data requires its execution policy"))
    _source_data_shape(grid, map, prefactor)
    _validate_source_responses(grid, kerr, raman, plasma)
    return ModalSourceData(grid, map, prefactor, kerr, raman, plasma)
end

function _source_data_shape(grid, map, prefactor)
    prefactor.shape == (length(grid.omega), size(map.synthesis, 2)) ||
        throw(DimensionMismatch("source prefactor does not match the modal grid"))
    grid.arithmetic == map.arithmetic == prefactor.arithmetic ||
        throw(ArgumentError("modal grid, maps and prefactor must share source arithmetic"))
    return nothing
end

function ModalSourceData(
        execution::KernelExecution, grid::ModalSourceGrid, map::ModalSourceMap,
        prefactor::SourcePrefactor; kerr::Union{Nothing, ModalKerr} = nothing,
        raman::Tuple = (), plasma::Union{Nothing, ModalPlasma} = nothing,
    )
    oncpu(execution) && return ModalSourceData(grid, map, prefactor; kerr, raman, plasma)
    _source_execution_arithmetic(execution, grid.arithmetic)
    _source_data_shape(grid, map, prefactor)
    data = ModalSourceData(grid, map, prefactor, kerr, raman, plasma)
    _source_device_data(data, execution)
    plasma_arrays = if plasma !== nothing && grid.kind == :envelope &&
            plasma.drude !== nothing && plasma.loss !== nothing
        (
            _source_preparation_host(grid.active, execution),
            _source_preparation_host(plasma.drude, execution),
            _source_preparation_host(plasma.loss, execution),
        )
    else
        nothing
    end
    _validate_source_responses(grid, kerr, raman, plasma; plasma_arrays)
    return data
end

function _validate_source_responses(
        grid::ModalSourceGrid, kerr::Union{Nothing, ModalKerr}, raman::Tuple,
        plasma::Union{Nothing, ModalPlasma}; plasma_arrays = nothing,
    )
    if kerr !== nothing
        kerr.arithmetic == grid.arithmetic ||
            throw(ArgumentError("Kerr and grid source arithmetic differ"))
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
        component.arithmetic == grid.arithmetic ||
            throw(ArgumentError("Raman and grid source arithmetic differ"))
        size(component.kernel.undamped, 1) == 2grid.n_time &&
            component.kernel.dt == grid.dt ||
            throw(DimensionMismatch("Raman rows must match the fine source grid"))
    end
    if plasma !== nothing
        plasma.arithmetic == grid.arithmetic ||
            throw(ArgumentError("plasma and grid source arithmetic differ"))
        if grid.kind == :field
            plasma.rule !== nothing &&
                plasma.rule.dt == source_type(grid.arithmetic)(grid.dt) ||
                throw(ArgumentError("field plasma requires the matching integral rule"))
            length(plasma.rule.taps) <= grid.n_time ||
                throw(ArgumentError("plasma FIR radius must be smaller than the time grid"))
        else
            plasma.rule === nothing ||
                throw(ArgumentError("envelope plasma requires cycle quadrature assembly"))
            length(plasma.drude) == length(grid.omega) ||
                throw(DimensionMismatch("plasma factors do not match the coarse grid"))
            active, drude, loss = plasma_arrays === nothing ?
                (grid.active, plasma.drude, plasma.loss) : plasma_arrays
            _source_require_host(active, :plasma_active_mask)
            _source_require_host(drude, :plasma_drude)
            _source_require_host(loss, :plasma_loss)
            all(eachindex(active)) do index
                return active[index] || (iszero(drude[index]) && iszero(loss[index]))
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

struct _SourceAnalytic{A, F}
    modal::A
    spectrum::A
    nodes::A
    transform::F
end

struct _SourceRamanBuffers{R, C, F}
    drive::R
    frequency::C
    transform::F
end

mutable struct _SourceRamanWork{D, F, C, S, T, E}
    data::D
    drive::RamanDrive
    time_kernel::Vector{Float64}
    spectrum::S
    rates::Vector{Float64}
    weights::Vector{Float64}
    forward::F
    convolution::C
    density::Float64
    host_spectrum::Vector{ComplexF64}
    range::SourceRange
    factor::T
    coefficient_execution::E

    function _SourceRamanWork(
            data::D, drive::RamanDrive, time_kernel::Vector{Float64},
            spectrum::S, rates::Vector{Float64}, weights::Vector{Float64},
            forward::F, convolution::C, density::Float64,
            host_spectrum::Vector{ComplexF64}, range::SourceRange, factor::T;
            coefficient_execution::KernelExecution = KernelExecution(
                KernelAbstractions.CPU(), 1
            ),
        ) where {D, F, C, T, S <: AbstractVector{Complex{T}}}
        source_type(data.arithmetic) === T ||
            throw(ArgumentError("Raman kernel work differs from source arithmetic"))
        oncpu(coefficient_execution) ||
            throw(ArgumentError("Raman coefficient reconstruction requires CPU execution"))
        return new{D, F, C, S, T, typeof(coefficient_execution)}(
            data, drive, time_kernel, spectrum, rates, weights,
            forward, convolution, density, host_spectrum, range, factor,
            coefficient_execution,
        )
    end
end

struct _SourcePlasmaWork{W, A, C, S, R, F}
    workspace::W
    first::A
    second::A
    projected_first::A
    projected_second::A
    coarse_first::C
    coarse_second::C
    factors_first::S
    factors_second::S
    stage::Matrix{Float64}
    ranges::R
    status_readback::F

    # These are already-owned prepared buffers. Keep construction from implicitly
    # converting arbitrary arrays into replacement storage with different ownership.
    function _SourcePlasmaWork(
            workspace::W, first::A, second::A, projected_first::A, projected_second::A,
            coarse_first::C, coarse_second::C, factors_first::S,
            factors_second::S, stage::Matrix{Float64}; ranges = nothing,
            status_readback = nothing,
        ) where {W, A, C, S <: AbstractVector{<:Complex}}
        return new{W, A, C, S, typeof(ranges), typeof(status_readback)}(
            workspace, first, second, projected_first, projected_second,
            coarse_first, coarse_second, factors_first, factors_second, stage, ranges,
            status_readback,
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
@inline _source_alias(array, owned::DeviceReadback) =
    _source_alias(array, (owned.device, owned.host))
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
            owned.host_synthesis, owned.host_projection, owned.host_coarse_projection,
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
    return _source_alias(array, (owned.modal, owned.spectrum, owned.nodes, owned.transform))
end

@inline function _source_alias(array, owned::TemporalFFT)
    return _source_alias(
        array, (_retained_storage(owned.forward), _retained_storage(owned.inverse))
    )
end

@inline function _source_alias(array, owned::_SourceRamanWork)
    return _source_alias(
        array, (
            owned.time_kernel, owned.spectrum, owned.rates, owned.weights,
            owned.host_spectrum, _retained_storage(owned.forward),
        )
    )
end

@inline function _source_alias(array, owned::_SourceRamanBuffers)
    return _source_alias(array, (owned.drive, owned.frequency, owned.transform))
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
            owned.factors_first, owned.factors_second, owned.stage, owned.status_readback,
        )
    )
end

"""
    ModalSource(execution, data::ModalSourceData; node_chunk)

Prepare owned scalar field or envelope source work in the selected execution context.

`node_chunk` is a positive maximum node count per block. Modal time/spectral arrays
use `(n_time,n_modes)` and `(n_omega,n_modes)`; nodal arrays use `(n_time,node_chunk)`.
The callable `source(out,state,z)` returns `nothing`, writes the complete derivative
in state units per metre, and charges `calls` before any validation or physical work.
Both caller arrays must be disjoint from this source's retained numerical inputs and
work, as well as from each other; direct arrays, views, foreign offset buffers and
reinterpreted aliases refuse before any numerical write. Mutable gather indices are
part of their array's retained storage. It never changes `state`. CPU FFTs use one
thread; all numerical launches finish before return. Warmed CPU calls allocate nothing.
CPU outputs require a writable affine layout; read-only gathered CPU inputs remain
valid. Device inputs and outputs require dense owned arrays in the selected context.
Every inactive chunk cell is overwritten.
The caller state/output are `Complex{source_type(data.grid.arithmetic)}` in the
declared scaled state units. Pointwise/FFT buffers use that precision; represented
profiles and Raman kernel reconstruction retain their explicit host-wide roles.
Device execution admits binary64 scalar Kerr, Raman, plasma and in-stage temporal
absorption. Raman reconstructs only coefficient data on the host and uploads its
persistent spectrum. Device finite/plasma checks read back bounded per-column status,
never evolving field data.

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
mutable struct ModalSource{E, D, P, F, A, T, C, N, R, B, L, Q, H, V, W, J}
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
    prefactor_weights::W
    quadrature_fine::H
    quadrature_coarse::H
    failures::J
    calls::Int
    temporal_active::Bool
    ready::Bool
    prefactor_range::V
end

function _source_caller_alias(array, source::ModalSource)
    # Chunk records refer to complete maps already visited here. Vendor plans may
    # retain extra numerical scratch; inspect only their explicit storage hook.
    return _source_alias(
        array, (
            source.data, source.padding.crop, source.padding.uncrop,
            source.modal_spectrum, source.modal_time, source.nodes, source.polarisation,
            source.projected, source.projection_work, source.coarse_projected,
            source.analytic, source.raman, source.raman_buffers, source.plasma,
            source.prefactor, source.prefactor_weights, source.quadrature_fine,
            source.quadrature_coarse, source.failures,
            source.transform,
        )
    )
end

function _source_analytic(execution::KernelExecution, data, modes::Int, capacity::Int)
    grid = data.grid
    needed = grid.kind == :field && (
        (
            data.kerr !== nothing &&
                data.kerr.coefficients isa KerrCoefficients{Float64, :analytic}
        ) ||
            any(component -> !component.thg, data.raman)
    )
    needed || return nothing
    T = source_type(grid.arithmetic)::Union{Type{Float32}, Type{Float64}}
    modal = dzeros(execution, Complex{T}, grid.n_time, modes)
    spectrum = dzeros(execution, Complex{T}, size(modal)...)
    nodes = dzeros(execution, Complex{T}, grid.n_time, capacity)
    return _SourceAnalytic(modal, spectrum, nodes, TemporalFFT(execution, modal, spectrum))
end

function _source_raman(execution, data, layout)
    isempty(data.raman) && return (), nothing
    T = source_type(data.grid.arithmetic)
    drive = dzeros(execution, T, 2layout.n_time, layout.n_nodes)
    frequency = dzeros(execution, Complex{T}, layout.n_time + 1, layout.n_nodes)
    transform = TemporalFFT(execution, drive, frequency)
    buffers = _SourceRamanBuffers(drive, frequency, transform)
    coefficient_execution = oncpu(execution) ? execution :
        KernelExecution(KernelAbstractions.CPU(), 1)
    components = map(data.raman) do component
        time_kernel = zeros(Float64, 2layout.n_time)
        host_spectrum = zeros(ComplexF64, layout.n_time + 1)
        spectrum = oncpu(execution) && T === Float64 ? host_spectrum :
            dzeros(execution, Complex{T}, layout.n_time + 1)
        planning = coefficient_execution.planning
        forward = if planning === nothing
            FFTW.plan_rfft(
                time_kernel; flags = FFTW.ESTIMATE | FFTW.PRESERVE_INPUT, num_threads = 1,
            )
        else
            plan_owned_fft!(planning, time_kernel, host_spectrum, (1,), :r2c; threads = 1)
        end
        convolution = RamanConvolution(
            layout.n_time, spectrum, transform.forward, transform.inverse;
            inverse_scale = transform.inverse_scale,
        )
        return _SourceRamanWork(
            component, RamanDrive(layout; kind = data.grid.kind, thg = component.thg),
            time_kernel, spectrum, zeros(Float64, length(component.decay_profiles)),
            fill(component.fraction, length(component.decay_profiles)),
            forward, convolution, 0.0, host_spectrum,
            SourceRange(:raman_kernel, 0, 0, T === Float32, 0.0, 0.0), zero(T),
            ; coefficient_execution,
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
    T = source_type(grid.arithmetic)
    field = grid.kind == :field
    field || (prepared.drude !== nothing && prepared.loss !== nothing) ||
        throw(ArgumentError("envelope plasma requires both spectral factors"))
    workspace = field ? FieldPlasmaWorkspace(execution, T, layout) :
        EnvelopePlasmaWorkspace(execution, T, layout)
    first = dzeros(execution, eltype(modal), layout.n_time, layout.n_nodes)
    second = dzeros(execution, eltype(modal), size(first)...)
    projected_first = dzeros(execution, eltype(modal), size(modal)...)
    projected_second = dzeros(execution, eltype(modal), size(modal)...)
    coarse_first = coarse ? dzeros(execution, eltype(modal), size(modal)...) : nothing
    coarse_second = coarse ? dzeros(execution, eltype(modal), size(modal)...) : nothing
    factors_first = zeros(ComplexF64, length(grid.omega))
    factors_second = similar(factors_first)
    active = _source_preparation_host(grid.active, execution)
    drude = field ? nothing : _source_preparation_host(prepared.drude, execution)
    loss = field ? nothing : _source_preparation_host(prepared.loss, execution)
    for index in eachindex(grid.omega)
        if field
            # SEP 27's held-current identity is G/(i*omega)-G'/(i*omega)^2.
            # Division is evaluated only on the explicitly active positive band.
            factors_first[index] = active[index] ? inv(im * grid.omega[index]) : 0
            factors_second[index] = active[index] ?
                -inv((im * grid.omega[index])^2) : 0
        else
            (drude === nothing || loss === nothing) &&
                throw(ArgumentError("envelope plasma requires both spectral factors"))
            factors_first[index] = drude[index]
            factors_second[index] = loss[index]
        end
    end
    all(isfinite, factors_first) && all(isfinite, factors_second) ||
        throw(ArgumentError("plasma spectral assembly factors overflow at an active bin"))
    first_factor, first_range = _source_prepare(
        execution, factors_first, grid.arithmetic, :plasma_first_factor
    )
    second_factor, second_range = _source_prepare(
        execution, factors_second, grid.arithmetic, :plasma_second_factor
    )
    return _SourcePlasmaWork(
        workspace, first, second, projected_first, projected_second,
        coarse_first, coarse_second, first_factor, second_factor,
        zeros(Float64, 2, length(prepared.channels));
        ranges = (first = first_range, second = second_range),
        status_readback = oncpu(execution) ? nothing :
            DeviceReadback(execution, Int, 2, layout.n_nodes),
    )
end

@inline _source_device_array(::Nothing, execution) = nothing
@inline function _source_device_array(array::AbstractArray, execution)
    copy_device!(array, array, execution)
    return nothing
end
@inline function _source_device_array(table::ChebyshevTable, execution)
    _source_device_array(table.breakpoints, execution)
    _source_device_array(table.coefficients, execution)
    return nothing
end

function _source_device_data(data, execution)
    oncpu(execution) && return nothing
    grid, maps, prefactor = data.grid, data.map, data.prefactor
    for array in (
            grid.active, grid.omega_window, grid.source_window,
            grid.source_window_derivative, grid.temporal_rate, grid.carrier,
            maps.synthesis, maps.projection, maps.coarse_projection,
            prefactor.data, prefactor.weights,
        )
        _source_device_array(array, execution)
    end
    if data.plasma !== nothing
        plasma = data.plasma
        _source_device_array(plasma.drude, execution)
        _source_device_array(plasma.loss, execution)
        plasma.rule === nothing || _source_device_array(plasma.rule.taps, execution)
        _source_each_indexed(plasma.channels) do channel, unused
            _source_device_array(channel.channel.rate.coefficients, execution)
            if channel.quadrature !== nothing
                _source_device_array(channel.quadrature.nodes, execution)
                _source_device_array(channel.quadrature.weights, execution)
            end
        end
    end
    return nothing
end

function ModalSource(execution::KernelExecution, data::ModalSourceData; node_chunk::Int)
    _source_execution_arithmetic(execution, data.grid.arithmetic)
    node_chunk > 0 || throw(ArgumentError("the modal source node chunk must be positive"))
    _source_device_data(data, execution)
    grid, maps = data.grid, data.map
    T = source_type(grid.arithmetic)
    nodes, modes = size(maps.synthesis)
    capacity = min(node_chunk, nodes)
    layout = ResponseLayout(grid.n_time, 1, capacity)
    scalar = grid.kind == :field ? T : Complex{T}
    frequency_count = grid.kind == :field ? grid.n_time ÷ 2 + 1 : grid.n_time
    modal = dzeros(execution, scalar, grid.n_time, modes)
    spectrum = dzeros(execution, Complex{T}, frequency_count, modes)
    nodal = dzeros(execution, scalar, grid.n_time, capacity)
    chunks = map(1:capacity:nodes) do first
        last = min(first + capacity - 1, nodes)
        return _SourceChunk(
            first, last - first + 1, maps.synthesis, maps.projection,
            maps.coarse_projection,
        )
    end
    padding = SpectralPadding(execution, grid.crop_index, frequency_count, one(T), one(T))
    raman, buffers = _source_raman(execution, data, layout)
    coarse = maps.coarse_projection !== nothing
    coarse_projected = coarse ? dzeros(execution, scalar, size(modal)...) : nothing
    coarse_source = coarse ?
        dzeros(execution, Complex{T}, data.prefactor.shape...) : nothing
    fine_source = coarse ? dzeros(execution, Complex{T}, data.prefactor.shape...) : nothing
    n_weights = data.prefactor.weights === nothing ? 0 :
        size(data.prefactor.weights.coefficients, 1)
    return ModalSource(
        execution, data, padding, TemporalFFT(execution, modal, spectrum), spectrum,
        modal, nodal, dzeros(execution, scalar, size(nodal)...),
        dzeros(execution, scalar, size(modal)...),
        dzeros(execution, scalar, size(modal)...), coarse_projected,
        chunks, _source_analytic(execution, data, modes, capacity), raman, buffers,
        _source_plasma(execution, data, layout, modal),
        dzeros(execution, Complex{T}, data.prefactor.shape...),
        dzeros(execution, Float64, n_weights), fine_source, coarse_source,
        oncpu(execution) ? zeros(Int, max(capacity, modes)) :
            DeviceReadback(execution, Int, max(capacity, modes)), 0,
        grid.temporal_rate !== nothing, false,
        T === Float32 ? SourceRange(:source_prefactor, 0, 0, false, 0.0, 0.0) : nothing,
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
        work.time_kernel, work.host_spectrum, component.kernel, work.rates, work.weights,
        work.forward, work.coefficient_execution,
    )
    if source_type(component.arithmetic) === Float32
        work.range = source_narrow!(
            work.spectrum, work.host_spectrum, :raman_kernel; normalize = true
        )
        work.factor = source_coefficient(
            component.arithmetic, :raman_material, component.material_factor, density;
            exponent = 2component.arithmetic.field_exponent + work.range.block_exponent,
        )
    else
        oncpu(execution) || copy_upload!(work.spectrum, work.host_spectrum, execution)
        work.factor = component.material_factor * density
    end
    return nothing
end

function _prepare_source_profiles!(source, position)
    data, execution = source.data, source.execution
    taper = _source_profile(data.map.taper, position)
    taper > 0 || throw(ArgumentError("modal taper scale must be positive"))
    kerr_density = data.kerr === nothing ? 0.0 :
        _source_profile(data.kerr.density, position)
    source.prefactor_range = _source_prefactor!(
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
    scale = source_coefficient(data.grid.arithmetic, :source_taper, taper)
    return scale, kerr_density
end

function _source_analytic!(analytic, spectrum, execution)
    full = analytic.spectrum
    half = size(spectrum, 1)
    each!(execution, size(full)...) do row, column
        full[row, column] = if row > half
            zero(eltype(full))
        elseif row == 1 || row == half
            complex(real(spectrum[row, column]), zero(real(eltype(full))))
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

function _source_project_add!(
        output, scratch, nodal, projection, taper, count, first_node, execution,
    )
    project!(
        scratch, nodal, projection, execution;
        n_components = 1, active_nodes = count, first_node,
    )
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
        factor = work.factor, count, add = true,
    )
    return nothing
end

function _source_kerr_parameters(response::ModalKerr, density)
    arithmetic, coefficients = response.arithmetic, response.coefficients
    source_type(arithmetic) === Float64 && return coefficients, density
    factor = source_coefficient(
        arithmetic, :kerr_material, coefficients.coefficient, density;
        exponent = 2arithmetic.field_exponent,
    )
    self = source_coefficient(arithmetic, :kerr_self_factor, coefficients.self_factor)
    conjugate = source_coefficient(
        arithmetic, :kerr_conjugate_factor, coefficients.conjugate_factor
    )
    harmonic = source_coefficient(
        arithmetic, :kerr_harmonic_factor, coefficients.harmonic_factor
    )
    return _source_kerr_coefficients(coefficients, factor, self, conjugate, harmonic), 1.0f0
end

@inline function _source_kerr_coefficients(
        coefficients::KerrCoefficients{Float64, F}, factor::T, self::T, conjugate::T,
        harmonic::T,
    ) where {F, T}
    return KerrCoefficients{T, F}(factor, self, conjugate, harmonic, coefficients.take_real)
end

@inline _source_analytic_cubic(::KerrCoefficients{T, F}) where {T, F} = F === :analytic

function _source_material!(source, kerr_density, count)
    data, execution = source.data, source.execution
    layout = ResponseLayout(size(source.nodes, 1), 1, size(source.nodes, 2))
    _source_zero!(source.polarisation, execution)
    if data.kerr !== nothing
        coefficients, density = _source_kerr_parameters(data.kerr, kerr_density)
        samples = if data.grid.kind == :field && _source_analytic_cubic(coefficients)
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
            density, carrier = data.grid.carrier, count,
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
            field_exponent = data.grid.arithmetic.field_exponent, count, add = index != 1,
        )
    else
        envelope_plasma!(
            work.first, work.second, source.nodes, channel.channel, channel.quadrature,
            work.workspace, execution; density, preionisation = initial,
            dt = data.grid.dt, window = data.grid.source_window, count, add = index != 1,
            field_exponent = data.grid.arithmetic.field_exponent,
        )
    end
    failure = _source_plasma_failure!(
        work.status_readback, work.workspace.status, execution
    )
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
            taper, chunk.count, chunk.first, execution,
        )
        if coarse !== nothing
            _source_project_add!(
                coarse, source.projection_work, nodal, chunk.coarse_projection,
                taper, chunk.count, chunk.first, execution,
            )
        end
    end
    return nothing
end

function _clear_source_projection!(source)
    execution = source.execution
    _source_zero!(source.projected, execution)
    source.coarse_projected === nothing || _source_zero!(source.coarse_projected, execution)
    if source.plasma !== nothing
        work = source.plasma
        _source_zero!(work.projected_first, execution)
        _source_zero!(work.projected_second, execution)
        work.coarse_first === nothing || _source_zero!(work.coarse_first, execution)
        work.coarse_second === nothing || _source_zero!(work.coarse_second, execution)
    end
    return nothing
end

function _source_over_nodes!(source, taper, kerr_density)
    execution = source.execution
    _clear_source_projection!(source)
    for chunk in source.chunks
        synthesize!(
            source.nodes, source.modal_time, chunk.synthesis, execution;
            n_components = 1, active_nodes = chunk.count, first_node = chunk.first,
        )
        _source_scale!(source.nodes, taper, execution)
        _source_finite!(source.failures, source.nodes, execution, :nodal_field)
        if source.analytic !== nothing
            synthesize!(
                source.analytic.nodes, source.analytic.modal, chunk.synthesis, execution;
                n_components = 1, active_nodes = chunk.count, first_node = chunk.first,
            )
            _source_scale!(source.analytic.nodes, taper, execution)
            _source_finite!(
                source.failures, source.analytic.nodes, execution, :analytic_field
            )
        end
        _source_material!(source, kerr_density, chunk.count)
        _source_project_add!(
            source.projected, source.projection_work, source.polarisation,
            chunk.projection, taper, chunk.count, chunk.first, execution,
        )
        if source.coarse_projected !== nothing
            _source_project_add!(
                source.coarse_projected, source.projection_work, source.polarisation,
                chunk.coarse_projection, taper, chunk.count, chunk.first, execution,
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
    half = source_type(source.data.grid.arithmetic)(-0.5)
    each!(execution, size(scratch)...) do row, column
        scratch[row, column] = half * rate[row] * modal[row, column]
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
    storage = Complex{source_type(source.data.grid.arithmetic)}
    eltype(state) === eltype(output) === storage ||
        throw(ArgumentError("modal source storage differs from its source arithmetic"))
    isfinite(position) || throw(ArgumentError("a source position must be finite"))
    _transfer_writable_destination(output)
    _transfer_arrays_alias(output, state) &&
        throw(ArgumentError("a modal source must not overwrite its input state"))
    (_source_caller_alias(output, source) || _source_caller_alias(state, source)) &&
        throw(ArgumentError("caller arrays must be disjoint from source inputs and work"))
    execution, data = source.execution, source.data
    if !oncpu(execution)
        copy_device!(output, output, execution)
        copy_device!(state, state, execution)
    end
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
            _source_copy!(source.quadrature_fine, output, execution)
            _source_assemble!(
                source.quadrature_coarse, source, source.coarse_projected,
                plasma === nothing ? nothing : plasma.coarse_first,
                plasma === nothing ? nothing : plasma.coarse_second,
            )
        end
    else
        _source_zero!(output, execution)
        source.quadrature_fine === nothing ||
            _source_zero!(source.quadrature_fine, execution)
        source.quadrature_coarse === nothing ||
            _source_zero!(source.quadrature_coarse, execution)
    end
    source.temporal_active && _source_temporal!(output, source)
    _source_finite!(source.failures, output, execution, :source)
    source.ready = true
    return nothing
end
