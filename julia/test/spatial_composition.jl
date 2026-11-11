# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import LinearAlgebra
import SolpropJulia as SP

# The independently defined temporal/response reference in composition.jl is used
# pointwise below. Spatial transforms here are explicit multidimensional DFT sums.
function spatial_dft(input, shape; inverse = false)
    nx, ny = shape
    sign, normalization = inverse ? (1, inv(nx * ny)) : (-1, 1.0)
    return ComplexF64[
        normalization * sum(
            input[row, x + nx * y + 1] *
                cis(sign * 2pi * (ix * x / nx + iy * y / ny))
                for x in 0:(nx - 1), y in 0:(ny - 1)
        ) for row in axes(input, 1), ix in 0:(nx - 1), iy in 0:(ny - 1)
    ] |> array -> reshape(array, size(input))
end

function spatial_case(execution, kind; shape = (2, 3), options...)
    base, unused = composition_case(execution, kind; coarse = false, options...)
    frequencies, columns = length(base.grid.omega), prod(shape)
    operator = SP.SeparableSpatialOperator(
        execution; k_squared = ComplexF64[8 + row + 0.02im for row in 1:frequencies],
        k_squared_low = fill(ComplexF64(1.0e-16), frequencies),
        kperp_sq = [0.3(column - 1)^2 for column in 1:columns],
        kperp_sq_low = fill(-1.0e-17, columns), frame = zeros(frequencies),
        spectral_rate = zeros(frequencies),
        prefactor_numerator = ComplexF64[-0.1im * row for row in 1:frequencies],
        prefactor_clamped = fill(-0.3im, frequencies), active = base.grid.active,
        evanescent_rate = 0.5, z_span = (0.0, 1.0),
    )
    spatial_rate = collect(range(0.0, 0.12; length = columns))
    data = SP.CartesianSourceData(
        base.grid, shape, operator, spatial_rate, 2.0;
        kerr = base.kerr, raman = base.raman, plasma = base.plasma,
    )
    state = ComplexF64[
        base.grid.active[row] ?
            (0.5 + 0.1column) * cis(0.7row + 0.2column + 0.1row * column) : 0
            for row in 1:frequencies, column in 1:columns
    ]
    return data, state
end

function spatial_reference(data, state, position; temporal_active = true)
    grid, shape = data.grid, data.shape
    columns, frequencies = prod(shape), size(state, 1)
    physical = spatial_dft(state, shape; inverse = true)
    # A diagonal material reference means each physical node is independent; it is
    # only reference bookkeeping, not the source's production map or workspace.
    identity = Matrix{Float64}(LinearAlgebra.I, columns, columns)
    ex = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    maps = SP.ModalSourceMap(
        ex; synthesis = data.pad_scale * identity, projection = identity,
    )
    reference = SP.ModalSourceData(
        grid, maps, SP.SourcePrefactor(ex, ones(ComplexF64, size(state)));
        kerr = data.kerr, raman = data.raman, plasma = data.plasma,
    )
    nodal, unused = composition_reference(
        reference, physical, position; temporal_active = false
    )
    # Bound before the spatial inverse's cancellations. The temporal/material
    # reference propagates this absolute band envelope through maps and responses.
    envelope = repeat(sum(abs.(state); dims = 2) ./ columns, 1, columns)
    unused, material_bound = composition_reference(
        reference, ComplexF64.(envelope), position; temporal_active = false
    )
    material = spatial_dft(nodal, shape)
    prefactor = zeros(ComplexF64, size(state))
    for column in 1:columns, row in 1:frequencies
        q = (data.operator.k_squared[row] - data.operator.kperp_sq[column]) +
            (data.operator.k_squared_low[row] - data.operator.kperp_sq_low[column])
        prefactor[row, column] = data.operator.active[row] ? (
                real(q) > 0 ?
                data.operator.prefactor_numerator[row] / real(sqrt(q)) :
                data.operator.prefactor_clamped[row]
            ) : 0
    end
    result = material .* prefactor
    bound = columns * material_bound * maximum(abs, prefactor)
    if data.spatial_active || (temporal_active && grid.temporal_rate !== nothing)
        fine = grid.kind == :field ? grid.n_time ÷ 2 + 1 : grid.n_time
        padded = zeros(ComplexF64, fine, columns)
        padded[grid.crop_index .+ 1, :] .= physical
        time = composition_inverse(padded, grid.n_time; real_signal = grid.kind == :field)
        rates = repeat(reshape(data.spatial_rate, 1, :), grid.n_time, 1)
        temporal_active && grid.temporal_rate !== nothing && (rates .+= grid.temporal_rate)
        loss = composition_dft(-0.5 .* rates .* time)[grid.crop_index .+ 1, :]
        result .+= spatial_dft(loss, shape) .* grid.active
        padded .= 0
        padded[grid.crop_index .+ 1, :] .= envelope
        weights = grid.kind == :field ? [1.0; fill(2.0, fine - 2); 1.0] : ones(fine)
        time_bound = sum(abs.(padded) .* weights; dims = 1) ./ grid.n_time
        bound += sum(0.5 .* rates .* time_bound)
    end
    result[.!grid.active, :] .= 0
    return result, bound
end

function spatial_inverse_allocations(work, input)
    SP.cartesian_inverse!(work, input)
    SP.cartesian_inverse!(work, input)
    return @allocated SP.cartesian_inverse!(work, input)
end

function spatial_forward_allocations(output, work)
    SP.cartesian_forward!(output, work)
    SP.cartesian_forward!(output, work)
    return @allocated SP.cartesian_forward!(output, work)
end

@testset "Cartesian bands use x-fast multidimensional transforms" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default)))), shape in ((2, 3), (4, 1))
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        input = ComplexF64[
            cis(0.31row + 0.17column) for row in 1:5, column in 1:prod(shape)
        ]
        saved = copy(input)
        work = SP.CartesianBandWork(execution, shape, 5)
        expected = spatial_dft(input, shape; inverse = true)
        SP.cartesian_inverse!(work, input)
        # Two short spatial FFTs vs explicit DFTs; use the pre-cancellation L1 scale.
        bound = 128eps(Float64) * maximum(sum(abs.(input); dims = 2))
        @test bound > 0
        @test maximum(abs, work.band - expected) <= bound
        @test input == saved
        @test parent(parent(work.planes)) === work.band
        output = similar(input)
        SP.cartesian_forward!(output, work)
        @test maximum(abs, output - input) <= bound
        @test spatial_inverse_allocations(work, input) == 0
        @test spatial_forward_allocations(output, work) == 0
        @test_throws ArgumentError SP.cartesian_inverse!(work, work.band)
        before = copy(work.band)
        @test_throws ArgumentError SP.cartesian_forward!(view(work.band, :, :), work)
        @test work.band == before
        @test_throws DimensionMismatch SP.cartesian_inverse!(work, zeros(5, prod(shape)))
    end
end

@testset "Cartesian scalar sources preserve translated composition" begin
    # Fewer than 32768 operations occur on any dependency path of these small DFT,
    # cubic, convolution, population and channel graphs. Absolute envelopes are
    # propagated before spatial/temporal/map/stream cancellation, never from |result|.
    gamma = 32768eps(Float64) / (1 - 32768eps(Float64))
    budgets = unique((1, min(2, Threads.nthreads(:default))))
    for tasks in budgets, kind in (:field, :envelope)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        for (kerr, raman, plasma, thg, temporal) in (
                (true, 0, false, true, false),
                (true, 0, false, false, false),
                (false, 2, false, false, false),
                (false, 0, true, false, false),
                (true, 2, true, false, true),
            )
            data, state = spatial_case(execution, kind; kerr, raman, plasma, thg, temporal)
            saved = copy(state)
            expected, scale = spatial_reference(data, state, 0.53)
            @test scale > 0
            for capacity in (1, 4, 6)
                source = SP.CartesianSource(execution, data; node_chunk = capacity)
                output = fill(ComplexF64(NaN), size(state))
                @test source(output, state, 0.53) === nothing
                @test all(isfinite, output)
                @test maximum(abs, output - expected) <= gamma * scale
                @test state == saved
                @test source.ready && source.calls == 1
                @test all(iszero, output[.!data.grid.active, :])
                @test composition_allocations(source, output, state, 0.53) == 0
                remaining = mod(size(state, 2), capacity)
                if remaining != 0
                    @test all(iszero, source.nodes[:, (remaining + 1):end])
                    @test all(iszero, source.polarisation[:, (remaining + 1):end])
                end
                if temporal
                    SP.set_temporal_source!(source, false)
                    @test !source.ready
                    quiet, quiet_scale = spatial_reference(
                        data, state, 0.53; temporal_active = false
                    )
                    source(output, state, 0.53)
                    @test maximum(abs, output - quiet) <= gamma * quiet_scale
                    @test composition_allocations(source, output, state, 0.53) == 0
                end
            end
        end
    end
end
function radial_case(execution, kind; mixed = false)
    base, unused = mixed ? composition_mixed_case(execution, kind) :
        composition_case(execution, kind; raman = 2, plasma = true, temporal = true)
    pad_scale = 2.0
    synthesis = pad_scale .* [1.2 0.2 -0.1; 0.3 0.8 0.1; -0.1 0.2 1.1]
    projection = [0.7 -0.1 0.2; 0.1 0.8 -0.2; -0.2 0.1 0.6]
    maps = SP.ModalSourceMap(execution; synthesis, projection)
    frequencies = length(base.grid.omega)
    prefactor = ComplexF64[
        base.grid.active[row] ? -im * (0.1 + 0.03row + 0.02column) / pad_scale : 0
            for row in 1:frequencies, column in 1:3
    ]
    terms = cat(prefactor, 0.2prefactor; dims = 3)
    weights = SP.ChebyshevTable(
        execution, [0.0, 1.0], reshape([1.0, 0.5, 0.1, -0.2], 2, 2, 1)
    )
    modal = SP.ModalSourceData(
        base.grid, maps, SP.SourcePrefactor(execution, terms, weights);
        kerr = base.kerr, raman = base.raman, plasma = base.plasma,
    )
    data = SP.RadialSourceData(modal, [0.0, 0.1, 0.3], pad_scale)
    state = ComplexF64[
        base.grid.active[row] ? (0.4 + 0.1column) * cis(0.7row + 0.2column) : 0
            for row in 1:frequencies, column in 1:3
    ]
    return data, state
end

function radial_reference(data, state, position; temporal_active = true)
    result, bound = composition_reference(data.modal, state, position; temporal_active)
    synthesis = data.modal.map.synthesis ./ data.pad_scale
    projection = data.modal.map.projection
    physical = state * transpose(synthesis)
    absorbed = physical .* reshape(-0.5 .* data.spatial_rate, 1, :)
    loss = absorbed * transpose(projection)
    result .+= loss .* data.modal.grid.active
    bound += maximum(
        (
            (abs.(state) * transpose(abs.(synthesis))) .*
                reshape(0.5 .* data.spatial_rate, 1, :)
        ) * transpose(abs.(projection))
    )
    return result, bound
end

@testset "Spatial source boundaries refuse independent overlapping array headers" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        cartesian, cartesian_state = spatial_case(execution, :envelope; temporal = true)
        radial, radial_state = radial_case(execution, :envelope)
        for (source, state) in (
                (SP.CartesianSource(execution, cartesian; node_chunk = 4), cartesian_state),
                (SP.RadialSource(execution, radial; node_chunk = 3), radial_state),
            )
            retained = source isa SP.CartesianSource ? source.nodes : source.material.nodes
            fill!(retained, 2 + 0.3im)
            saved = copy(retained)
            output = fill(3 + 0.1im, size(state))
            output_saved = copy(output)
            state_saved = copy(state)
            @test length(retained) > length(state)
            GC.@preserve source begin
                wrapped = unsafe_wrap(Array, pointer(retained, 2), size(state); own = false)
                @test !Base.mightalias(wrapped, retained)
                @test SP._source_caller_alias(wrapped, source)
                for (destination, input) in ((wrapped, state), (output, wrapped))
                    calls = source.calls
                    @test_throws ArgumentError source(destination, input, 0.53)
                    @test source.calls == calls + 1 && !source.ready
                    @test retained == saved && output == output_saved &&
                        state == state_saved
                end
            end
            # Both overlapping headers are valid inside an independently retained
            # backing array; neither shares a Julia owner identity with the other.
            backing = fill(2 + 0.2im, length(state) + 1)
            GC.@preserve backing begin
                first = unsafe_wrap(Array, pointer(backing), size(state); own = false)
                second = unsafe_wrap(Array, pointer(backing, 2), size(state); own = false)
                @test !Base.mightalias(first, second)
                @test_throws ArgumentError source(first, second, 0.53)
                @test all(==(2 + 0.2im), backing)
            end
            source(output, state, 0.53)
            @test source.ready && state == state_saved
            @test composition_allocations(source, output, state, 0.53) == 0
        end
        backing = zeros(ComplexF64, 13)
        GC.@preserve backing begin
            band = unsafe_wrap(Array, pointer(backing), (3, 4); own = false)
            planes = reshape(view(band, :, :), 3, 2, 2)
            work = SP.CartesianBandWork(
                execution, (2, 2), band, planes, SP.SpatialFFT(execution, planes, (2, 3)),
            )
            overlapping = unsafe_wrap(Array, pointer(backing, 2), (3, 4); own = false)
            @test !Base.mightalias(overlapping, band)
            @test_throws ArgumentError SP.cartesian_inverse!(work, overlapping)
            @test_throws ArgumentError SP.cartesian_forward!(overlapping, work)
            @test iszero(backing)
        end
    end
end

@testset "Radial sources preserve independent profiles and additive loss" begin
    gamma = 32768eps(Float64) / (1 - 32768eps(Float64))
    budgets = unique((1, min(2, Threads.nthreads(:default))))
    for tasks in budgets, kind in (:field, :envelope)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        for mixed in (false, true)
            data, state = radial_case(execution, kind; mixed)
            saved = copy(state)
            setprecision(BigFloat, 256) do
                synthesis = BigFloat.(data.modal.map.synthesis) ./ data.pad_scale
                projection = BigFloat.(data.modal.map.projection)
                expected = projection * LinearAlgebra.Diagonal(
                    -BigFloat.(data.spatial_rate) ./ 2
                ) * synthesis
                scale = abs.(projection) * LinearAlgebra.Diagonal(
                    BigFloat.(data.spatial_rate) ./ 2
                ) * abs.(synthesis)
                @test maximum(abs.(BigFloat.(data.absorber) - expected)) <=
                    16eps(Float64) * maximum(scale)
            end
            for capacity in (1, 2, 3), temporal_active in (true, false)
                source = SP.RadialSource(execution, data; node_chunk = capacity)
                SP.set_temporal_source!(source, temporal_active)
                output = similar(state)
                expected, scale = radial_reference(data, state, 0.53; temporal_active)
                @test scale > 0
                @test source(output, state, 0.53) === nothing
                @test maximum(abs, output - expected) <= gamma * scale
                @test state == saved
                @test source.calls == 1 && source.ready
                @test all(iszero, output[.!data.modal.grid.active, :])
                @test composition_allocations(source, output, state, 0.53) == 0
            end
        end
    end
end

@testset "Spatial source preparation and calls preserve owned storage" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    data, state = spatial_case(
        execution, :envelope; raman = 2, plasma = true, temporal = true,
    )
    source = SP.CartesianSource(execution, data; node_chunk = 4)
    output = fill(1.0 + 0.3im, size(state))
    saved = copy(output)
    @test_throws ArgumentError SP.CartesianBandWork(execution, (1, 3), 4)
    @test_throws ArgumentError SP.CartesianBandWork(execution, (3, 0), 4)
    @test_throws ArgumentError SP.CartesianBandWork(execution, (3, 2), 0)
    @test_throws ArgumentError SP.CartesianSource(execution, data; node_chunk = 0)
    @test_throws DimensionMismatch SP.CartesianSourceData(
        data.grid, (2, 2), data.operator, zeros(4), 2.0
    )
    for bad in (fill(-1.0, 6), fill(Inf, 6), zeros(5), zeros(Float32, 6))
        @test_throws ArgumentError SP.CartesianSourceData(
            data.grid, data.shape, data.operator, bad, 2.0
        )
    end
    @test_throws ArgumentError SP.CartesianSourceData(
        data.grid, data.shape, data.operator, zeros(6), 0.0
    )
    @test_throws ArgumentError source(output, state, -1.0)
    @test output == saved
    @test !source.ready && source.calls == 1
    @test_throws ArgumentError source(output, view(output, :, :), 0.5)
    @test output == saved
    @test_throws ArgumentError source(source.band.band, state, 0.5)
    @test_throws ArgumentError source(output, source.band.band, 0.5)
    @test_throws ArgumentError source(output, real.(state), 0.5)
    @test_throws DimensionMismatch source(zeros(ComplexF64, 1, 1), state, 0.5)
    damaged = copy(state)
    damaged[2, 2] = NaN
    @test_throws SP.NonfiniteSourceError source(output, damaged, 0.5)
    @test output == saved
    source(output, state, 0.5)
    @test source.ready
    ledger, seen = Dict{Symbol, Int}(), IdDict{Any, Bool}()
    SP._memory_arrays!(ledger, seen, "source", source)
    @test haskey(seen, source.band.band)
    @test !haskey(seen, source.band.planes)
    @test all(
        haskey(seen, array) for array in (
                source.spectrum, source.nodes, source.polarisation, source.chunk_spectrum,
                source.failures, source.plasma.first, source.plasma.second,
            )
    )
    @test count(key -> occursin("band.band", String(key)), keys(ledger)) == 1
    blank = SP.CartesianSourceData(data.grid, data.shape, data.operator, zeros(6), 2.0)
    blank_source = SP.CartesianSource(execution, blank; node_chunk = 4)
    SP.set_temporal_source!(blank_source, false)
    blank_source(output, state, 0.5)
    @test all(iszero, output)
    @test composition_allocations(blank_source, output, state, 0.5) == 0
    line_data, line_state = spatial_case(execution, :envelope; shape = (4, 1))
    line = SP.CartesianSource(execution, line_data; node_chunk = 3)
    line_output = similar(line_state)
    line(line_output, line_state, 0.5)
    expected, scale = spatial_reference(line_data, line_state, 0.5)
    @test maximum(abs, line_output - expected) <= 32768eps(Float64) * scale
    @test_throws ArgumentError SP.set_temporal_source!(line, true)
    radial_data, radial_state = radial_case(execution, :field)
    radial = SP.RadialSource(execution, radial_data; node_chunk = 2)
    radial_output = similar(radial_state)
    @test_throws ArgumentError radial(radial.scratch, radial_state, 0.5)
    @test_throws ArgumentError radial(radial_output, radial.material.prefactor, 0.5)
    @test !radial.ready && radial.calls == 2 && radial.material.calls == 0
    @test_throws ArgumentError SP.RadialSourceData(radial_data.modal, fill(-1.0, 3), 2.0)
    @test_throws ArgumentError SP.RadialSourceData(radial_data.modal, zeros(2), 2.0)
    @test_throws ArgumentError SP.RadialSourceData(radial_data.modal, zeros(3), NaN)
    zero_data = SP.RadialSourceData(radial_data.modal, zeros(3), 2.0)
    @test isempty(zero_data.absorber)
    no_loss = SP.RadialSource(execution, zero_data; node_chunk = 2)
    no_loss(radial_output, radial_state, 0.5)
    modal_output = similar(radial_output)
    no_loss.material(modal_output, radial_state, 0.5)
    @test radial_output == modal_output
end

@testset "Cartesian independent heterogeneous profiles remain allocation-free" begin
    gamma = 32768eps(Float64) / (1 - 32768eps(Float64))
    budgets = unique((1, min(2, Threads.nthreads(:default))))
    for tasks in budgets, kind in (:field, :envelope)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        geometry, state = spatial_case(execution, kind)
        profiles, unused = composition_mixed_case(execution, kind)
        data = SP.CartesianSourceData(
            profiles.grid, geometry.shape, geometry.operator, geometry.spatial_rate,
            geometry.pad_scale; kerr = profiles.kerr, raman = profiles.raman,
            plasma = profiles.plasma,
        )
        source = SP.CartesianSource(execution, data; node_chunk = 4)
        output = similar(state)
        for position in (0.23, 0.53, 0.83)
            expected, scale = spatial_reference(data, state, position)
            source(output, state, position)
            @test scale > 0
            @test maximum(abs, output - expected) <= gamma * scale
            @test composition_allocations(source, output, state, position) == 0
        end
    end
end

@testset "Spatial outputs refuse mutable gather layouts before writes (#689)" begin
    budgets = unique((1, min(2, Threads.nthreads(:default))))
    for tasks in budgets, kind in (:field, :envelope)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        cartesian, cartesian_state = spatial_case(execution, kind)
        radial, radial_state = radial_case(execution, kind)
        for (source, state, expected, scale) in (
                (
                    SP.CartesianSource(execution, cartesian; node_chunk = 4),
                    cartesian_state,
                    spatial_reference(cartesian, cartesian_state, 0.53)...,
                ),
                (
                    SP.RadialSource(execution, radial; node_chunk = 2), radial_state,
                    radial_reference(radial, radial_state, 0.53)...,
                ),
            )
            arrays = source isa SP.RadialSource ?
                (source.scratch, source.material.nodes) : (source.band.band, source.nodes)
            composition_output_layouts(source, state, arrays, expected, scale, 32768)
            if source isa SP.RadialSource
                # The new outer boundary refuses before its material source is called.
                calls = source.material.calls
                output = view(similar(state), collect(axes(state, 1)), :)
                @test_throws "immutable affine indices" source(output, state, 0.53)
                @test source.material.calls == calls
                # Read-only gathered radial inputs keep the existing modal contract.
                input = view(state, collect(axes(state, 1)), :)
                output = similar(state)
                source(output, input, 0.53)
                bound = 32768eps(Float64) / (1 - 32768eps(Float64)) * scale
                @test maximum(abs.(output - expected)) <= bound
                @test composition_allocations(source, output, input, 0.53) == 0
            end
        end
    end
end
