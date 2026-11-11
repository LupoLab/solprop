# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public CartesianBandWork, cartesian_inverse!, cartesian_forward!
public CartesianSourceData, CartesianSource, RadialSourceData, RadialSource

function _cartesian_shape(shape::NTuple{2, Int})
    shape[1] >= 2 && shape[2] >= 1 ||
        throw(ArgumentError("Cartesian shape requires n_x>=2 and n_y>=1 (a line)"))
    return Base.checked_mul(shape...)
end

"""
    CartesianBandWork(execution, shape::NTuple{2,Int}, n_frequency;
                      arithmetic=SourceArithmetic(Float64))

Own a scalar Cartesian/line band and its bounded spatial FFT plans.

`shape=(n_x,n_y)` has x varying fastest, with `n_x>=2` and `n_y>=1`; `n_y=1`
is the line route. The owned `Complex{source_type(arithmetic)}` band has shape
`(n_frequency,n_x*n_y)` and retains the arithmetic record.
The inverse is normalized by `1/(n_x*n_y)`, and the forward is the raw negative-sign
DFT. Temporal normalization, material coefficients and physical measures are absent.
Each independent source or observation owns its own work. CPU plans use one FFTW
thread. Unsupported execution or invalid dimensions raise `ArgumentError`.
"""
struct CartesianBandWork{E, B, A, F, P}
    execution::E
    shape::NTuple{2, Int}
    band::B
    planes::A
    plan::F
    arithmetic::P
end

function CartesianBandWork(
        execution::KernelExecution, shape::NTuple{2, Int}, n_frequency::Int;
        arithmetic::SourceArithmetic = SourceArithmetic(Float64),
    )
    oncpu(execution) || throw(ArgumentError("Cartesian transforms currently require CPU"))
    n_frequency > 0 ||
        throw(ArgumentError("the Cartesian frequency count must be positive"))
    columns = _cartesian_shape(shape)
    band = dzeros(execution, Complex{source_type(arithmetic)}, n_frequency, columns)
    # A reshaped view preserves the matrix's explicit owner chain for the memory
    # ledger. Retaining a second Array header would count the same memory twice.
    planes = reshape(view(band, :, :), n_frequency, shape...)
    return CartesianBandWork(
        execution, shape, band, planes, SpatialFFT(execution, planes, (2, 3)), arithmetic,
    )
end

"""
    cartesian_inverse!(work::CartesianBandWork, input)

Copy a disjoint frequency-first spectrum and return the owned physical-node band.

`input` is strided complex storage with the work's precision and shape. Preserve it
exactly, then perform the normalized transverse inverse before any real temporal
inverse. This ordering matters: a transverse spectral column alone need not be the
spectrum of a real field. Throw on invalid storage, shape or overlap before mutation.
"""
function cartesian_inverse!(work::CartesianBandWork, input)
    input isa StridedMatrix{eltype(work.band)} && size(input) == size(work.band) ||
        throw(DimensionMismatch("Cartesian input differs from its prepared band"))
    _transfer_arrays_alias(input, work.band) &&
        throw(ArgumentError("Cartesian inverse input must be disjoint from owned work"))
    copyto!(work.band, input)
    spatial_inverse!(work.planes, work.plan, work.execution)
    return work.band
end

"""
    cartesian_forward!(output, work::CartesianBandWork)

Transform the entire owned physical-node band and copy its raw spatial spectrum out.

`output` is strided complex with the work's precision and shape. It may be exactly
`work.band`; any other overlap refuses before mutation. Return `output`. Transform
completion is synchronous and warmed calls allocate nothing.
"""
function cartesian_forward!(output, work::CartesianBandWork)
    output isa StridedMatrix{eltype(work.band)} && size(output) == size(work.band) ||
        throw(DimensionMismatch("Cartesian output differs from its prepared band"))
    output !== work.band && _transfer_arrays_alias(output, work.band) &&
        throw(ArgumentError("Cartesian forward output refuses partial work overlap"))
    spatial_forward!(work.planes, work.plan)
    output === work.band || copyto!(output, work.band)
    return output
end

"""
    CartesianSourceData(grid, shape, operator, spatial_rate, pad_scale;
                        kerr=nothing, raman=(), plasma=nothing,
                        arithmetic=grid.arithmetic)

Bind owned scalar Cartesian/line temporal, material and additive-loss records.

`grid::ModalSourceGrid` supplies the same temporal/spectral contract as modal sources.
`shape=(n_x,n_y)` has x fastest, and `operator::SeparableSpatialOperator` has the same
frequency and column extents and active mask. Its linear material vectors are constant;
varying Cartesian linear dispersion is not represented by this interface.
`spatial_rate` is a copied nonnegative node vector in m⁻¹. Positive `pad_scale` is
applied once to material fields; the translated operator prefactor already contains
crop scaling. Additive loss uses neither that prefactor nor the source/spectral windows.

Kerr, scalar Raman and scalar plasma are the existing independent owned response
records and profile tuples, including independent varying nonlinear profiles. No physics
default, density or analytic factor is added.
The arithmetic must match the grid and every response. Binary32 preparation evaluates
the separated prefactor in binary64 and folds `2^(field_exponent-state_exponent)`
before one narrowing. Material reconstruction folds the reciprocal exponent into
`pad_scale`. Additive loss stays in stored state units. Named ranges describe the
prepared spatial rate and prefactor; the original operator remains immutable wide data.
Throw `ArgumentError`/`DimensionMismatch` before workspace allocation for bad data.
"""
struct CartesianSourceData{G, K, R, P, S, F, T, A, B}
    grid::G
    shape::NTuple{2, Int}
    operator::SeparableSpatialOperator
    spatial_rate::S
    pad_scale::Float64
    kerr::K
    raman::R
    plasma::P
    spatial_active::Bool
    prefactor::F
    field_scale::T
    arithmetic::A
    ranges::B
end

function CartesianSourceData(
        grid::ModalSourceGrid, shape::NTuple{2, Int},
        operator::SeparableSpatialOperator, spatial_rate, pad_scale::Float64;
        kerr::Union{Nothing, ModalKerr} = nothing, raman::Tuple = (),
        plasma::Union{Nothing, ModalPlasma} = nothing,
        arithmetic::SourceArithmetic = grid.arithmetic,
    )
    columns = _cartesian_shape(shape)
    _spatial_operator_shape(operator) == (length(grid.omega), columns) &&
        grid.active == operator.active ||
        throw(DimensionMismatch("Cartesian geometry, operator and temporal grid disagree"))
    spatial_rate isa StridedVector{Float64} && length(spatial_rate) == columns &&
        axes(spatial_rate, 1) == Base.OneTo(columns) &&
        all(value -> isfinite(value) && value >= 0, spatial_rate) ||
        throw(ArgumentError("Cartesian spatial loss needs nonnegative binary64 node rates"))
    isfinite(pad_scale) && pad_scale > 0 ||
        throw(ArgumentError("Cartesian temporal pad scale must be finite and positive"))
    _validate_source_responses(grid, kerr, raman, plasma)
    arithmetic == grid.arithmetic ||
        throw(ArgumentError("Cartesian source and grid arithmetic differ"))
    rate, rate_range = _source_prepare_host(spatial_rate, arithmetic, :spatial_rate)
    field_scale = source_coefficient(
        arithmetic, :cartesian_field_scale, pad_scale;
        exponent = arithmetic.state_exponent - arithmetic.field_exponent,
    )
    prefactor, prefactor_range = if source_type(arithmetic) === Float32
        tables = _spatial_operator_tables(operator)
        wide = Matrix{ComplexF64}(undef, length(grid.omega), columns)
        for column in axes(wide, 2), row in axes(wide, 1)
            wide[row, column] = tables.active[row] ?
                last(_spatial_element(tables, row, column)) : 0
        end
        _source_prepare_host(
            wide, arithmetic, :cartesian_prefactor;
            exponent = arithmetic.field_exponent - arithmetic.state_exponent,
        )
    else
        nothing, nothing
    end
    return CartesianSourceData(
        grid, shape, operator, rate, pad_scale, kerr, raman, plasma,
        any(!iszero, spatial_rate), prefactor, field_scale, arithmetic,
        (spatial_rate = rate_range, prefactor = prefactor_range),
    )
end

"""
    CartesianSource(execution, data::CartesianSourceData; node_chunk)

Prepare a bounded scalar Cartesian/line source, including additive absorption.

Call `source(out,state,z)` with disjoint `(n_omega,n_x*n_y)` strided complex matrices
at the grid's response precision, in scaled state units and metres. The callable
returns `nothing`, preserves the state, and charges
`calls` before validation. Inputs/outputs must also be disjoint from retained numerical
inputs and scratch. `ready` is set only after all finite checks succeed.

The material path takes a normalized spatial inverse, bounded padded temporal inverses,
the existing response leaves, then raw temporal/spatial forwards and the translated
prefactor. One full band scratch is reused for a separate additive loss pass; a purely
spatial rate needs no temporal transform. Every inactive chunk column is overwritten.
All plans use one FFTW thread and all launches finish synchronously. Warmed calls
allocate nothing. Monitor decisions and accepted-step split operations remain separate.

Throw `ArgumentError`/`DimensionMismatch` for unsupported preparation or input data;
`NonfiniteSourceError`/`SourcePlasmaError` report numerical failures outside kernels.
"""
mutable struct CartesianSource{E, D, B, P, F, T, S, A, R, RB, L}
    execution::E
    data::D
    band::B
    padding::P
    transform::F
    spectrum::S
    nodes::T
    polarisation::T
    analytic::A
    raman::R
    raman_buffers::RB
    plasma::L
    chunk_spectrum::S
    failures::Vector{Int}
    calls::Int
    temporal_active::Bool
    ready::Bool
end

function CartesianSource(
        execution::KernelExecution, data::CartesianSourceData; node_chunk::Int,
    )
    oncpu(execution) || throw(ArgumentError("Cartesian sources currently require CPU"))
    node_chunk > 0 || throw(ArgumentError("the Cartesian node chunk must be positive"))
    grid = data.grid
    T = source_type(grid.arithmetic)
    columns, frequencies = _cartesian_shape(data.shape), length(grid.omega)
    capacity = min(node_chunk, columns)
    scalar = grid.kind == :field ? T : Complex{T}
    fine = grid.kind == :field ? grid.n_time ÷ 2 + 1 : grid.n_time
    nodes = dzeros(execution, scalar, grid.n_time, capacity)
    spectrum = dzeros(execution, Complex{T}, fine, capacity)
    layout = ResponseLayout(grid.n_time, 1, capacity)
    raman, buffers = _source_raman(execution, data, layout)
    return CartesianSource(
        execution, data, CartesianBandWork(
            execution, data.shape, frequencies; arithmetic = grid.arithmetic
        ),
        SpectralPadding(execution, grid.crop_index, fine, one(T), one(T)),
        TemporalFFT(execution, nodes, spectrum), spectrum, nodes, similar(nodes),
        _source_analytic(execution, data, capacity, capacity), raman, buffers,
        _source_plasma(execution, data, layout, nodes; coarse = false),
        dzeros(execution, Complex{T}, frequencies, capacity),
        zeros(Int, max(columns, capacity)),
        0, grid.temporal_rate !== nothing, false,
    )
end

@inline _source_alias(array, owned::SeparableSpatialOperator) =
    _spatial_operator_alias(array, owned)
@inline _source_alias(array, owned::CartesianBandWork) =
    _transfer_arrays_alias(array, owned.band)
@inline function _source_alias(array, owned::CartesianSourceData)
    return _source_alias(
        array, (
            owned.grid, owned.operator, owned.spatial_rate, owned.prefactor,
            owned.kerr, owned.raman, owned.plasma,
        )
    )
end

function _source_caller_alias(array, source::CartesianSource)
    # Independent unsafe_wrap headers need the existing address-span guard, including
    # all nested response/profile arrays, not only Julia's owner identity (#685).
    return _retained_alias(array, source)
end

function set_temporal_source!(source::CartesianSource, active::Bool)
    active && source.data.grid.temporal_rate === nothing &&
        throw(ArgumentError("no in-stage Cartesian temporal rate was prepared"))
    source.temporal_active, source.ready = active, false
    return nothing
end

function _cartesian_profiles!(source, position)
    data, execution = source.data, source.execution
    kerr_density = data.kerr === nothing ? 0.0 :
        _source_profile(data.kerr.density, position)
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
    return kerr_density
end

function _cartesian_to_time!(source, first, count, scale)
    spectrum, physical, uncrop = source.spectrum, source.band.band, source.padding.uncrop
    each!(source.execution, size(spectrum)...) do row, column
        coarse = uncrop[row]
        spectrum[row, column] = column <= count && coarse != 0 ?
            scale * physical[coarse, first + column - 1] : zero(eltype(spectrum))
    end
    temporal_inverse!(source.nodes, source.transform, spectrum, source.execution)
    _source_finite!(source.failures, source.nodes, source.execution, :spatial_node_field)
    return nothing
end

function _cartesian_add_stream!(source, stream, factors)
    temporal_forward!(source.spectrum, source.transform, stream)
    output, spectrum, crop = source.chunk_spectrum, source.spectrum, source.padding.crop
    each!(source.execution, size(output)...) do row, column
        output[row, column] += factors[row] * spectrum[crop[row], column]
    end
    return nothing
end

function _cartesian_store_chunk!(source, first, count)
    physical, chunk = source.band.band, source.chunk_spectrum
    each!(source.execution, size(chunk, 1), count) do row, column
        physical[row, first + column - 1] = chunk[row, column]
    end
    return nothing
end

function _cartesian_material!(output, source, state, position)
    execution, data = source.execution, source.data
    kerr_density = _cartesian_profiles!(source, position)
    cartesian_inverse!(source.band, state)
    capacity, columns = size(source.nodes, 2), size(state, 2)
    for first in 1:capacity:columns
        count = min(capacity, columns - first + 1)
        _cartesian_to_time!(source, first, count, data.field_scale)
        if source.analytic !== nothing
            _source_analytic!(source.analytic, source.spectrum, execution)
            copyto!(source.analytic.nodes, source.analytic.modal)
        end
        _source_material!(source, kerr_density, count)
        temporal_forward!(source.spectrum, source.transform, source.polarisation)
        crop_spectrum!(source.chunk_spectrum, source.spectrum, source.padding, execution)
        if source.plasma !== nothing
            _source_each_indexed(data.plasma.channels) do channel, index
                _source_plasma_channel!(source, channel, index, count, first)
            end
            _cartesian_add_stream!(source, source.plasma.first, source.plasma.factors_first)
            _cartesian_add_stream!(
                source, source.plasma.second, source.plasma.factors_second,
            )
        end
        _source_finite!(
            source.failures, source.chunk_spectrum, execution, :spatial_material,
        )
        _cartesian_store_chunk!(source, first, count)
    end
    cartesian_forward!(source.band.band, source.band)
    _source_finite!(
        source.failures, source.band.band, execution, :spatial_material_spectrum,
    )
    spectrum, window, active = source.band.band, data.grid.omega_window, data.grid.active
    if data.prefactor === nothing
        tables = _spatial_operator_tables(data.operator)
        each!(execution, size(output)...) do row, column
            output[row, column] = active[row] ?
                last(_spatial_element(tables, row, column)) * window[row] *
                spectrum[row, column] : zero(eltype(output))
        end
    else
        prefactor = data.prefactor
        each!(execution, size(output)...) do row, column
            output[row, column] = active[row] ?
                prefactor[row, column] * window[row] * spectrum[row, column] :
                zero(eltype(output))
        end
    end
    return nothing
end

function _cartesian_absorber!(output, source, state)
    execution, data = source.execution, source.data
    cartesian_inverse!(source.band, state)
    spatial, physical = data.spatial_rate, source.band.band
    T = source_type(data.grid.arithmetic)
    half = T(-0.5)
    if source.temporal_active
        capacity, columns = size(source.nodes, 2), size(state, 2)
        temporal, nodes, loss = data.grid.temporal_rate, source.nodes, source.polarisation
        for first in 1:capacity:columns
            count = min(capacity, columns - first + 1)
            # No material pad scale: this is an additive linear rate, transformed
            # back without the material crop-scaled prefactor (SEP 40 section4.7).
            _cartesian_to_time!(source, first, count, one(T))
            each!(execution, size(loss)...) do row, column
                loss[row, column] = column <= count ?
                    half * (spatial[first + column - 1] + temporal[row]) *
                    nodes[row, column] : zero(eltype(loss))
            end
            temporal_forward!(source.spectrum, source.transform, loss)
            crop_spectrum!(
                source.chunk_spectrum, source.spectrum, source.padding, execution,
            )
            _cartesian_store_chunk!(source, first, count)
        end
    else
        # Node-only rates commute with the temporal transform: avoid that pair.
        each!(execution, size(physical)...) do row, column
            physical[row, column] *= half * spatial[column]
        end
    end
    cartesian_forward!(physical, source.band)
    _source_finite!(source.failures, physical, execution, :spatial_absorber)
    active = data.grid.active
    each!(execution, size(output)...) do row, column
        output[row, column] = active[row] ?
            output[row, column] + physical[row, column] : zero(eltype(output))
    end
    return nothing
end

function (source::CartesianSource)(output, state, position::Float64)
    source.calls += 1
    source.ready = false
    T = Complex{source_type(source.data.grid.arithmetic)}
    output isa StridedMatrix{T} && state isa StridedMatrix{T} ||
        throw(ArgumentError("Cartesian input/output differs from its source arithmetic"))
    size(output) == size(state) == size(source.band.band) ||
        throw(DimensionMismatch("Cartesian source input/output does not match preparation"))
    _spatial_operator_position(source.data.operator, position)
    _transfer_writable_destination(output)
    (
        _transfer_arrays_alias(output, state) || _source_caller_alias(output, source) ||
            _source_caller_alias(state, source)
    ) &&
        throw(ArgumentError("Cartesian caller arrays must be disjoint from owned storage"))
    _source_finite!(source.failures, state, source.execution, :input)
    data = source.data
    material = data.kerr !== nothing || !isempty(data.raman) || data.plasma !== nothing
    if material
        _cartesian_material!(output, source, state, position)
    else
        fill!(output, 0)
    end
    if data.spatial_active || source.temporal_active
        _cartesian_absorber!(output, source, state)
    end
    _source_finite!(source.failures, output, source.execution, :source)
    source.ready = true
    return nothing
end

"""
    RadialSourceData(modal::ModalSourceData, spatial_rate, pad_scale;
                     arithmetic=modal.grid.arithmetic)

Bind scalar radial QDHT material records and a separate fixed spatial loss operator.

The square real maps are already translated: synthesis includes `pad_scale`, projection
contains the forward QDHT scale, and the radial taper is exactly one. Preserve every
independent material, prefactor and varying radial profile in `modal`. The copied
nonnegative node rate (m⁻¹) forms `B=projection*diag(-rate/2)*(synthesis/pad_scale)`.
This fixed operator commutes with temporal transforms and is additive outside material
prefactors and windows. A zero node rate retains an empty B and is an exact no-op.
Preparation is serial; no separate BLAS pool changes the execution thread budget.
Form `B` from the retained original binary64 maps, then narrow once to the source
precision; folded/narrowed material maps are never substituted into its assembly.
The arithmetic must match the modal records. Named range facts cover rate and B.
Invalid maps/rates/scales or nonfinite composed coefficients raise `ArgumentError`.
"""
struct RadialSourceData{M, S, B, A, R}
    modal::M
    spatial_rate::S
    pad_scale::Float64
    absorber::B
    arithmetic::A
    ranges::R
end

function RadialSourceData(
        modal::ModalSourceData, spatial_rate, pad_scale::Float64;
        arithmetic::SourceArithmetic = modal.grid.arithmetic,
    )
    nodes, modes = size(modal.map.synthesis)
    nodes == modes || throw(ArgumentError("a radial QDHT requires square scalar maps"))
    modal.map.taper.value isa Float64 && isone(modal.map.taper.value) ||
        throw(ArgumentError("radial spatial maps require a fixed unit taper"))
    spatial_rate isa StridedVector{Float64} && length(spatial_rate) == nodes &&
        axes(spatial_rate, 1) == Base.OneTo(nodes) &&
        all(value -> isfinite(value) && value >= 0, spatial_rate) ||
        throw(ArgumentError("radial spatial loss needs nonnegative binary64 node rates"))
    isfinite(pad_scale) && pad_scale > 0 ||
        throw(ArgumentError("radial temporal pad scale must be finite and positive"))
    arithmetic == modal.grid.arithmetic ||
        throw(ArgumentError("radial source and grid arithmetic differ"))
    absorber = if any(!iszero, spatial_rate)
        prepared = zeros(Float64, modes, modes)
        synthesis, projection = modal.map.host_synthesis, modal.map.host_projection
        for column in 1:modes, row in 1:modes
            total = 0.0
            for node in 1:nodes
                total += projection[row, node] *
                    ((-0.5 * spatial_rate[node]) * (synthesis[node, column] / pad_scale))
            end
            prepared[row, column] = total
        end
        all(isfinite, prepared) ||
            throw(ArgumentError("the composed radial spatial loss is nonfinite"))
        prepared
    else
        zeros(Float64, 0, 0)
    end
    rate, rate_range = _source_prepare_host(spatial_rate, arithmetic, :spatial_rate)
    combined, combined_range = _source_prepare_host(absorber, arithmetic, :radial_absorber)
    return RadialSourceData(
        modal, rate, pad_scale, combined, arithmetic,
        (spatial_rate = rate_range, absorber = combined_range),
    )
end

"""
    RadialSource(execution, data::RadialSourceData; node_chunk)

Own a scalar radial source with reused modal material kernels and additive spatial loss.

The callable `source(out,state,z)` uses the modal source's disjoint precision/scale
contract and returns `nothing`. Varying radial material/operator profiles are preserved;
this source does not replace their represented tables. Its fixed spatial loss is
`B*state`, added only on active bins after material and temporal-source composition.
`calls` counts attempted invocations, `ready` marks only complete successful calls, and
`set_temporal_source!` delegates the monitor's decision to the owned modal source.
Every mutable buffer is session-owned. Warmed CPU calls allocate nothing. Invalid
preparation or aliases refuse outside kernels; numerical failures retain their codes.
"""
mutable struct RadialSource{E, D, M, A}
    execution::E
    data::D
    material::M
    scratch::A
    failures::Vector{Int}
    calls::Int
    ready::Bool
end

function RadialSource(execution::KernelExecution, data::RadialSourceData; node_chunk::Int)
    material = ModalSource(execution, data.modal; node_chunk)
    scalar = Complex{source_type(data.arithmetic)}
    return RadialSource(
        execution, data, material,
        dzeros(execution, scalar, data.modal.prefactor.shape...),
        zeros(Int, data.modal.prefactor.shape[2]), 0, false,
    )
end

@inline function _source_alias(array, data::RadialSourceData)
    return _source_alias(array, (data.modal, data.spatial_rate, data.absorber))
end

function _source_caller_alias(array, source::RadialSource)
    return _retained_alias(array, source)
end

function set_temporal_source!(source::RadialSource, active::Bool)
    set_temporal_source!(source.material, active)
    source.ready = false
    return nothing
end

function (source::RadialSource)(output, state, position::Float64)
    source.calls += 1
    source.ready = false
    _transfer_writable_destination(output)
    (
        _transfer_arrays_alias(output, state) || _source_caller_alias(output, source) ||
            _source_caller_alias(state, source)
    ) &&
        throw(ArgumentError("radial caller arrays must be disjoint from inputs and work"))
    source.material(output, state, position)
    if !isempty(source.data.absorber)
        project!(
            source.scratch, state, source.data.absorber, source.execution; n_components = 1,
        )
        scratch, active = source.scratch, source.data.modal.grid.active
        _source_finite!(source.failures, scratch, source.execution, :radial_absorber)
        each!(source.execution, size(output)...) do row, column
            output[row, column] = active[row] ?
                output[row, column] + scratch[row, column] : zero(eltype(output))
        end
    end
    _source_finite!(source.failures, output, source.execution, :source)
    source.ready = true
    return nothing
end
