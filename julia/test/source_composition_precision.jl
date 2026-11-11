# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

# composition.jl and spatial_composition.jl provide independent DFT, convolution,
# quadratic-rate and absolute-contribution references, not a second source call.
function precision_source_grid(execution, grid, arithmetic)
    return SP.ModalSourceGrid(
        execution; grid.kind, grid.n_time, grid.dt, grid.crop_index, grid.omega,
        grid.active, grid.omega_window, grid.source_window, grid.source_window_derivative,
        grid.temporal_rate, grid.carrier, arithmetic,
    )
end

function precision_source_material(execution, data, arithmetic)
    kerr = data.kerr === nothing ? nothing :
        SP.ModalKerr(data.kerr.coefficients, data.kerr.density; arithmetic)
    raman = map(data.raman) do component
        return SP.ModalRaman(
            component.kernel, component.decay_profiles, component.density;
            component.fraction, component.material_factor, component.thg, arithmetic,
        )
    end
    plasma = if data.plasma === nothing
        nothing
    else
        channels = map(data.plasma.channels) do component
            return SP.ModalPlasmaChannel(
                component.channel, component.density, component.preionisation;
                component.quadrature, arithmetic,
            )
        end
        if data.plasma.rule === nothing
            SP.ModalPlasma(
                execution, channels; data.plasma.drude, data.plasma.loss, arithmetic
            )
        else
            SP.ModalPlasma(execution, channels, data.plasma.rule; arithmetic)
        end
    end
    return (; kerr, raman, plasma)
end

function precision_source_data(execution, data::SP.ModalSourceData, arithmetic)
    grid = precision_source_grid(execution, data.grid, arithmetic)
    maps = SP.ModalSourceMap(
        execution; data.map.synthesis, data.map.projection, data.map.coarse_projection,
        data.map.taper, arithmetic,
    )
    old = data.prefactor
    prefactor = if old.weights !== nothing
        SP.SourcePrefactor(execution, old.data, old.weights; arithmetic)
    elseif old.data isa SP.ChebyshevTable
        SP.SourcePrefactor(old.data, old.shape; arithmetic)
    else
        SP.SourcePrefactor(execution, old.data; arithmetic)
    end
    material = precision_source_material(execution, data, arithmetic)
    return SP.ModalSourceData(grid, maps, prefactor; material...)
end

function precision_source_data(execution, data::SP.CartesianSourceData, arithmetic)
    grid = precision_source_grid(execution, data.grid, arithmetic)
    material = precision_source_material(execution, data, arithmetic)
    return SP.CartesianSourceData(
        grid, data.shape, data.operator, data.spatial_rate, data.pad_scale;
        material..., arithmetic,
    )
end

function precision_source_data(execution, data::SP.RadialSourceData, arithmetic)
    modal = precision_source_data(execution, data.modal, arithmetic)
    return SP.RadialSourceData(modal, data.spatial_rate, data.pad_scale; arithmetic)
end

precision_complex_scale(value, exponent) =
    complex(ldexp(real(value), exponent), ldexp(imag(value), exponent))

function precision_source_pair(state, arithmetic)
    stored = ComplexF32.(precision_complex_scale.(state, -arithmetic.state_exponent))
    represented = precision_complex_scale.(ComplexF64.(stored), arithmetic.state_exponent)
    return stored, represented
end

@testset "binary32 modal sources preserve field/state scaling" begin
    # These 16-time, 3-node, 2-mode fixtures have <8192 finite operations along any
    # source path, including spline evaluation, protected scans and <=512-node cycle
    # reductions. Absolute envelopes precede every DFT/map/channel cancellation.
    # As in the binary64 source witness, the FFT error model includes finite twiddle
    # errors; operation count alone is not a proof of a vendor FFT implementation.
    # Four cycle tolerances cover the independently bounded moment contributions.
    unit = Float64(eps(Float32)) / 2
    gamma = 8192unit / (1 - 8192unit)
    # Supply the translated production defaults explicitly. The older binary64
    # fixture intentionally requests a tighter 2e-8 rule with only 16/32 nodes;
    # its binary32 exhaustion is tested separately, not hidden by changing a rule.
    cycle = composition_cycle(; orders = (64, 128, 256, 512), moment_rtol = 2.0e-6)
    budgets = unique((1, min(2, Threads.nthreads(:default))))
    for tasks in budgets, kind in (:field, :envelope)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        for (kerr, raman, plasma, thg, prefactor_kind) in (
                (true, 0, false, true, :constant),
                (true, 0, false, false, :table),
                (false, 2, false, false, :expansion),
                (false, 0, true, false, :constant),
                (true, 2, true, false, :expansion),
            )
            original, physical = composition_case(
                execution, kind; kerr, raman, plasma, thg, prefactor_kind,
                temporal = true, cycle,
            )
            for (state_exponent, field_exponent) in ((0, 0), (3, 1), (-12, -8))
                arithmetic = SP.SourceArithmetic(Float32; state_exponent, field_exponent)
                data = precision_source_data(execution, original, arithmetic)
                state, represented = precision_source_pair(physical, arithmetic)
                expected, scale = composition_reference(original, represented, 0.53)
                expected = precision_complex_scale.(expected, -state_exponent)
                bound = (gamma + (plasma ? 8.0e-6 : 0.0)) * ldexp(scale, -state_exponent)
                @test isfinite(bound) && bound > 0
                source = SP.ModalSource(execution, data; node_chunk = 2)
                output = similar(state)
                saved = copy(state)
                source(output, state, 0.53)
                @test source.ready && source.calls == 1
                @test maximum(abs, ComplexF64.(output) - expected) <= bound
                @test isequal(state, saved)
                @test all(iszero, output[.!data.grid.active, :])
                @test eltype(source.modal_time) === (kind === :field ? Float32 : ComplexF32)
                @test eltype(source.prefactor) === ComplexF32
                @test data.map.arithmetic === arithmetic
                @test data.map.host_synthesis == original.map.synthesis
                @test data.map.host_synthesis !== original.map.synthesis
                @test data.map.ranges.synthesis.folded_exponent ==
                    state_exponent - field_exponent
                @test composition_allocations(source, output, state, 0.53) == 0
                for response in source.raman
                    @test eltype(response.spectrum) === ComplexF32
                    @test eltype(response.host_spectrum) === ComplexF64
                    @test response.range.normalized
                    @test 1 <= response.range.largest < 2
                    @test eltype(source.raman_buffers.drive) === Float32
                end
                if source.plasma !== nothing
                    @test eltype(source.plasma.first) === eltype(source.nodes)
                    @test eltype(source.plasma.workspace.exposure) === Float32
                end
                if source.prefactor_range !== nothing
                    @test source.prefactor_range.name === :source_prefactor
                end
                # The last node chunk is incomplete. Poison its spare column and
                # require the next pass to overwrite it before any transform reads it.
                fill!(source.nodes, NaN)
                source(output, state, 0.53)
                @test all(iszero, source.nodes[:, end])
                @test maximum(abs, ComplexF64.(output) - expected) <= bound
            end
        end
    end
end

@testset "variable source prefactors evaluate wide and narrow once" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    arithmetic = SP.SourceArithmetic(Float32)
    # Rounded Float32 coefficients cancel to zero at x=1, while the represented
    # binary64 polynomial has the exactly representable value one.
    coefficients = reshape(ComplexF64[2.0^25 + 1, -2.0^25], 1, 2, 1)
    table = SP.ChebyshevTable(execution, [0.0, 1.0], coefficients)
    prepared = SP.SourcePrefactor(table, (1, 1); arithmetic)
    output = zeros(ComplexF32, 1, 1)
    work = Float64[]
    fact = SP._source_prefactor!(output, prepared, 1.0, work, execution)
    @test output[1] === ComplexF32(1)
    @test fact.largest == 1.0 && fact.name === :source_prefactor
    @test eltype(prepared.data.coefficients) === ComplexF64
    terms = reshape(ComplexF64[2.0^25 + 1, -2.0^25], 1, 1, 2)
    weights = SP.ChebyshevTable(execution, [0.0, 1.0], ones(2, 1, 1))
    expansion = SP.SourcePrefactor(execution, terms, weights; arithmetic)
    SP._source_prefactor!(output, expansion, 0.5, zeros(2), execution)
    @test output[1] === ComplexF32(1)
    @test expansion.data == terms && expansion.data !== terms
    tiny = SP.SourcePrefactor(
        SP.ChebyshevTable(execution, [0.0, 1.0], fill(1.0e-50im, 1, 1, 1)), (1, 1);
        arithmetic,
    )
    before = copy(output)
    @test_throws ArgumentError SP._source_prefactor!(output, tiny, 0.5, work, execution)
    @test output == before
end

@testset "binary32 preparation owns ranges and rejects mismatched scales" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    original, state = composition_case(execution, :envelope)
    arithmetic = SP.SourceArithmetic(Float32; state_exponent = 4, field_exponent = 2)
    data = precision_source_data(execution, original, arithmetic)
    wrong = SP.SourceArithmetic(Float32; state_exponent = 3, field_exponent = 2)
    mismatched = precision_source_data(execution, original, wrong)
    @test_throws ArgumentError SP.ModalSourceData(
        data.grid, mismatched.map, data.prefactor; kerr = data.kerr
    )
    @test_throws ArgumentError SP.ModalSourceData(
        data.grid, data.map, mismatched.prefactor; kerr = data.kerr
    )
    @test_throws ArgumentError SP.ModalSourceData(
        data.grid, data.map, data.prefactor; kerr = mismatched.kerr
    )
    @test original.map.host_synthesis === original.map.synthesis
    @test original.map.host_projection === original.map.projection
    # Binary64 still admits finite complex values whose magnitude would overflow;
    # no conversion is needed and a later scaled multiplication can remain finite.
    extreme = fill(complex(0.9floatmax(Float64), 0.9floatmax(Float64)), 1, 1)
    retained = SP.SourcePrefactor(execution, extreme)
    @test retained.data == extreme && retained.data !== extreme
    @test retained.ranges.values === nothing
    @test all(fact -> fact === nothing || fact isa SP.SourceRange, data.grid.ranges)
    @test data.grid.omega !== original.grid.omega
    @test eltype(data.grid.omega) === Float64
    source = SP.ModalSource(execution, data; node_chunk = 2)
    stored, unused = precision_source_pair(state, arithmetic)
    output = fill(ComplexF32(7), size(stored))
    @test_throws ArgumentError source(output, state, 0.3)
    @test all(==(ComplexF32(7)), output)
    @test_throws ArgumentError source(source.prefactor, stored, 0.3)
    @test !source.ready
end

@testset "tiny material factors and Raman kernels retain explicit powers" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 2, true)
    arithmetic = SP.SourceArithmetic(Float32; state_exponent = 3, field_exponent = 1)
    base, physical = composition_case(execution, :envelope; kerr = false, raman = 1)
    old = only(base.raman)
    small = SP.RamanKernel(old.kernel.undamped .* 1.0e-100, old.kernel.dt)
    component = SP.ModalRaman(
        small, old.decay_profiles, SP.SourceProfile(1.0e100);
        old.fraction, old.material_factor, old.thg,
    )
    kerr = SP.ModalKerr(
        SP.KerrCoefficients(1.0e-100, :analytic), SP.SourceProfile(-1.0e100)
    )
    original = SP.ModalSourceData(
        base.grid, base.map, base.prefactor; kerr, raman = (component,)
    )
    data = precision_source_data(execution, original, arithmetic)
    state, represented = precision_source_pair(physical, arithmetic)
    source = SP.ModalSource(execution, data; node_chunk = 2)
    output = similar(state)
    source(output, state, 0.53)
    expected, scale = composition_reference(original, represented, 0.53)
    expected = precision_complex_scale.(expected, -arithmetic.state_exponent)
    unit = Float64(eps(Float32)) / 2
    bound = 8192unit / (1 - 8192unit) * ldexp(scale, -arithmetic.state_exponent)
    @test bound > 0
    @test maximum(abs, ComplexF64.(output) - expected) <= bound
    @test source.ready
    @test only(source.raman).range.block_exponent < -300
    @test any(!iszero, only(source.raman).spectrum)
    @test composition_allocations(source, output, state, 0.53) == 0
end

@testset "binary32 cycle exhaustion preserves the requested tolerance" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    original, physical = composition_case(execution, :envelope; kerr = false, plasma = true)
    data = precision_source_data(execution, original, SP.SourceArithmetic(Float32))
    tolerances = unique(channel.quadrature.moment_rtol for channel in data.plasma.channels)
    @test only(tolerances) <=
        original.plasma.channels[1].quadrature.moment_rtol
    source = SP.ModalSource(execution, data; node_chunk = 2)
    state = ComplexF32.(physical)
    output = fill(ComplexF32(7), size(state))
    failure = try
        source(output, state, 0.53)
        nothing
    catch error
        error
    end
    @test failure isa SP.SourcePlasmaError
    @test failure.code === SP.PlasmaQuadratureFailure
    @test source.calls == 1 && !source.ready
    @test all(==(ComplexF32(7)), output)
end

@testset "binary32 spatial sources keep independent state and field units" begin
    unit = Float64(eps(Float32)) / 2
    # The spatial witness adds at most six-term DFT/contraction paths to the
    # independently conditioned temporal/material reference. Its pre-cancellation
    # envelope uses the same finite FFT/twiddle hypothesis as the modal witness.
    gamma = 32768unit / (1 - 32768unit)
    cycle = composition_cycle(; orders = (64, 128, 256, 512), moment_rtol = 2.0e-6)
    budgets = unique((1, min(2, Threads.nthreads(:default))))
    for tasks in budgets, kind in (:field, :envelope)
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        arithmetic = SP.SourceArithmetic(Float32; state_exponent = 3, field_exponent = 1)
        for shape in ((2, 3), (4, 1))
            original, physical = spatial_case(
                execution, kind; shape, raman = 2, plasma = true, temporal = true, cycle,
            )
            data = precision_source_data(execution, original, arithmetic)
            state, represented = precision_source_pair(physical, arithmetic)
            expected, scale = spatial_reference(original, represented, 0.53)
            expected = precision_complex_scale.(expected, -arithmetic.state_exponent)
            bound = (gamma + 8.0e-6) * ldexp(scale, -arithmetic.state_exponent)
            source = SP.CartesianSource(execution, data; node_chunk = 3)
            output = similar(state)
            source(output, state, 0.53)
            @test bound > 0 && all(isfinite, output)
            @test maximum(abs, ComplexF64.(output) - expected) <= bound
            @test source.ready && source.calls == 1
            @test eltype(source.band.band) === ComplexF32
            @test source.band.arithmetic === arithmetic
            @test eltype(data.prefactor) === ComplexF32
            @test eltype(data.spatial_rate) === Float32
            @test data.ranges.prefactor.folded_exponent == -2
            @test composition_allocations(source, output, state, 0.53) == 0
            SP.set_temporal_source!(source, false)
            source(output, state, 0.53)
            quiet, quiet_scale = spatial_reference(
                original, represented, 0.53; temporal_active = false
            )
            quiet = precision_complex_scale.(quiet, -arithmetic.state_exponent)
            @test maximum(abs, ComplexF64.(output) - quiet) <=
                (gamma + 8.0e-6) * ldexp(quiet_scale, -arithmetic.state_exponent)
        end
        original, physical = radial_case(execution, kind; mixed = true, cycle)
        data = precision_source_data(execution, original, arithmetic)
        state, represented = precision_source_pair(physical, arithmetic)
        expected, scale = radial_reference(original, represented, 0.53)
        expected = precision_complex_scale.(expected, -arithmetic.state_exponent)
        source = SP.RadialSource(execution, data; node_chunk = 2)
        output = similar(state)
        source(output, state, 0.53)
        @test maximum(abs, ComplexF64.(output) - expected) <=
            (gamma + 8.0e-6) * ldexp(scale, -arithmetic.state_exponent)
        @test composition_allocations(source, output, state, 0.53) == 0
        @test eltype(data.absorber) === Float32
        @test eltype(source.scratch) === ComplexF32
        @test data.ranges.absorber.folded_exponent == 0
        # Preparation computes B from original host maps, independently of the
        # differently folded forward/projection buffers used by material kernels.
        @test data.absorber == Float32.(original.absorber)
    end
end
