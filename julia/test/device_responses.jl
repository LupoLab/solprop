# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

isdefined(@__MODULE__, :_modal_source_model) || include("device_composition.jl")
include("device_response_fixture.jl")

const _response_status_only = Ref(false)

function SP.copy_host!(
        output::Array{T, N}, input::Array{T, N}, execution::_ModalSourceModelExecution,
    ) where {T, N}
    !_response_status_only[] || T === Int ||
        error("a source evaluation downloaded noncompact numerical data")
    execution.backend.downloads += 1
    execution.backend.elements += length(input)
    return copyto!(output, input)
end

function SP.copy_upload!(
        output::Array, input::Array, execution::_ModalSourceModelExecution,
    )
    size(output) == size(input) || throw(DimensionMismatch("model upload shape mismatch"))
    eltype(output) === eltype(input) || throw(ArgumentError("model upload type mismatch"))
    return copyto!(output, input)
end

struct _ResponseShapeOnly{T, N} <: AbstractArray{T, N}
    shape::NTuple{N, Int}
end
Base.size(array::_ResponseShapeOnly) = array.shape
Base.getindex(::_ResponseShapeOnly, indices...) = error("unexpected scalar inspection")

@testset "Uploaded plasma records retain validated metadata without scalar reads" begin
    token = SP._PlasmaValidatedCopy()
    rate = SP.IonisationRateTable(zeros(4, 2), 1.0, 4.0, 2.0, 0.0, 1.0)
    coefficients = _ResponseShapeOnly{Float64, 2}((4, 2))
    copied = SP.IonisationRateTable(token, rate, coefficients)
    @test copied.coefficients === coefficients
    @test all(getfield(copied, i) === getfield(rate, i) for i in 2:6)
    @test_throws DimensionMismatch SP.IonisationRateTable(
        token, rate, _ResponseShapeOnly{Float64, 2}((4, 3))
    )
    @test_throws ArgumentError SP.IonisationRateTable(token, rate, zeros(Float32, 4, 2))
    rule = SP.RunningIntegralRule([0.5, 0.5], 0.125; floor_exposure = true)
    taps = _ResponseShapeOnly{Float64, 1}((2,))
    integral = SP.RunningIntegralRule(token, rule, taps)
    @test integral.taps === taps && integral.dt === rule.dt && integral.floor_exposure
    @test_throws DimensionMismatch SP.RunningIntegralRule(token, rule, zeros(3))
    @test_throws ArgumentError SP.RunningIntegralRule(token, rule, zeros(Float32, 2))
    quadrature = composition_cycle()
    shape = size(quadrature.nodes)
    nodes, weights = _ResponseShapeOnly{Float64, 1}(shape),
        _ResponseShapeOnly{Float64, 1}(shape)
    cycle = SP.CycleQuadrature(token, quadrature, nodes, weights)
    @test cycle.nodes === nodes && cycle.weights === weights
    @test cycle.ends === quadrature.ends
    @test cycle.moment_rtol === quadrature.moment_rtol
    @test cycle.negligible_rate === quadrature.negligible_rate
    @test_throws DimensionMismatch SP.CycleQuadrature(token, quadrature, zeros(2), zeros(2))
    @test_throws ArgumentError SP.CycleQuadrature(
        token, quadrature, zeros(Float32, shape), zeros(Float32, shape)
    )
end

@testset "Device plasma preparation revalidates current host data before upload" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    execution = _modal_source_model()
    for kind in (:field, :envelope)
        host, _ = composition_case(cpu, kind; plasma = true)
        original = first(host.plasma.channels)
        before = copy(original.channel.rate.coefficients)
        copied = SP.ModalPlasmaChannel(
            execution, original.channel, original.density, original.preionisation;
            quadrature = original.quadrature,
        )
        @test copied.channel.rate.coefficients == before
        @test copied.channel.rate.coefficients !== original.channel.rate.coefficients
        count = execution.backend.allocations
        original.channel.rate.coefficients[1] = NaN
        @test_throws ArgumentError SP.ModalPlasmaChannel(
            execution, original.channel, original.density, original.preionisation;
            quadrature = original.quadrature,
        )
        @test execution.backend.allocations == count
        @test copied.channel.rate.coefficients == before
        original.channel.rate.coefficients .= before
        @test_throws ArgumentError SP.ModalPlasmaChannel(
            execution, original.channel, original.density, original.preionisation;
            quadrature = original.quadrature, arithmetic = SP.SourceArithmetic(Float32),
        )
        @test execution.backend.allocations == count
        if kind == :envelope
            quadrature = original.quadrature
            @test copied.quadrature.nodes !== quadrature.nodes
            @test copied.quadrature.weights !== quadrature.weights
            for values in (quadrature.nodes, quadrature.weights)
                saved = copy(values)
                values[1] = NaN
                @test_throws ArgumentError SP.ModalPlasmaChannel(
                    execution, original.channel, original.density, original.preionisation;
                    quadrature,
                )
                @test execution.backend.allocations == count
                values .= saved
            end
        else
            rule = host.plasma.rule
            rule.taps[1] = Inf
            @test_throws ArgumentError SP.ModalPlasma(execution, (copied,), rule)
            @test execution.backend.allocations == count
        end
    end
end

@testset "Compact plasma status preserves node-major ordering and bounded transfer" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 2, true)
    for execution in (cpu, _modal_source_model()), rows in (1, 17), columns in (1, 4)
        work = SP.DeviceReadback(execution, Int, 2, columns)
        status = zeros(UInt8, rows, columns)
        for code in (
                SP.PlasmaSuccess, SP.PlasmaNonfiniteInput,
                SP.PlasmaNonfiniteResult, SP.PlasmaQuadratureFailure,
            )
            fill!(status, 0)
            status[rows, 1] = UInt8(code)
            columns > 1 && (status[1, columns] = UInt8(SP.PlasmaNonfiniteResult))
            expected = SP.plasma_failure(status)
            before = copy(status)
            downloads = SP.oncpu(execution) ? 0 : execution.backend.elements
            actual = SP._source_plasma_failure!(work, status, execution)
            @test (actual.code, actual.node, actual.sample) ==
                (expected.code, expected.node, expected.sample)
            @test status == before
            @test SP.oncpu(execution) || execution.backend.elements - downloads == 2columns
            @test (@allocated SP._source_plasma_failure!(work, status, execution)) == 0
        end
        saved = copy(work.device)
        @test_throws ArgumentError SP._source_plasma_failure!(
            work, status, SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
        )
        @test work.device == saved
        @test_throws DimensionMismatch SP._source_plasma_failure!(
            work, zeros(UInt8, rows, columns + 1), execution
        )
        @test work.device == saved
        alias = reinterpret(UInt8, work.device)
        @test_throws ArgumentError SP._source_plasma_failure!(work, alias, execution)
        @test work.device == saved
    end
    @test_throws ArgumentError SP._source_plasma_failure!(
        nothing, zeros(UInt8, 2, 1), _modal_source_model()
    )
end

@testset "Device response model matches independent Raman and plasma source reference" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 2, true)
    gamma = 8192eps(Float64) / (1 - 8192eps(Float64))
    # See CUDA_SOURCE.md: the fixed finite fixture uses a direct DFT, a causal
    # convolution and analytic W=kE² moments; its scale precedes all cancellations.
    for kind in (:field, :envelope), combination in (:raman, :plasma, :combined),
            chunk in (1, 2, 3)
        execution = _modal_source_model()
        host, state = composition_case(
            cpu, kind; kerr = combination == :combined,
            thg = kind == :field && combination == :raman,
            raman = combination == :plasma ? 0 : 2,
            plasma = combination != :raman, temporal = combination == :combined,
            prefactor_kind = combination == :plasma ? :table : :expansion,
        )
        data = device_response_data(execution, host)
        source = SP.ModalSource(execution, data; node_chunk = chunk)
        output, before = similar(state), copy(state)
        for position in (0.0, 0.53, 1.0)
            _response_status_only[] = true
            try
                @test source(output, state, position) === nothing
            finally
                _response_status_only[] = false
            end
            expected, scale = composition_reference(host, state, position)
            @test 0 < gamma * scale < 1.0e-8
            @test device_response_rate_domain(host, state, position) < gamma * scale / 16
            @test maximum(abs, output - expected) <= gamma * scale
            @test state == before && source.ready
            @test all(iszero, output[.!host.grid.active, :])
            fine, fine_scale = composition_reference(
                host, state, position; temporal_active = false
            )
            coarse, coarse_scale = composition_reference(
                host, state, position; temporal_active = false, coarse = true
            )
            @test maximum(abs, source.quadrature_fine - fine) <= gamma * fine_scale
            @test maximum(abs, source.quadrature_coarse - coarse) <= gamma * coarse_scale
            @test composition_allocations(source, output, state, position) == 0
        end
        ledger = Dict{Symbol, Int}()
        SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "source", source)
        for (index, work) in enumerate(source.raman)
            @test SP.oncpu(work.coefficient_execution)
            @test work.coefficient_execution.tasks == 1
            @test work.spectrum !== work.host_spectrum
            @test work.spectrum == work.host_spectrum
            @test ledger[Symbol("source.raman.$index.spectrum")] == sizeof(work.spectrum)
            @test ledger[Symbol("source.raman.$index.host_spectrum")] ==
                sizeof(work.host_spectrum)
            for retained in (work.spectrum, work.host_spectrum)
                GC.@preserve retained begin
                    alias = unsafe_wrap(Array, pointer(retained), size(state); own = false)
                    saved = copy(retained)
                    @test_throws ArgumentError source(alias, state, 0.53)
                    @test retained == saved && !source.ready
                end
            end
        end
        if source.plasma !== nothing
            work = source.plasma.status_readback
            @test size(work.device) == (2, chunk)
            @test work.device !== work.host
            @test ledger[Symbol("source.plasma.status_readback.device")] ==
                2chunk * sizeof(Int)
            @test ledger[Symbol("source.plasma.status_readback.host")] ==
                2chunk * sizeof(Int)
        end
        @test source(output, state, 0.53) === nothing
        if chunk == 2
            @test all(iszero, source.nodes[:, 2])
            if source.plasma !== nothing
                @test all(iszero, source.plasma.workspace.status[:, 2])
                @test all(iszero, source.plasma.first[:, 2])
                @test all(iszero, source.plasma.second[:, 2])
            end
        end
    end
end

@testset "Device Raman and plasma preserve one and eight discrete RK histories" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope), combination in (:raman, :plasma, :combined),
            steps in (1, 8)
        execution = _modal_source_model()
        host, state = composition_case(
            cpu, kind; kerr = combination == :combined,
            thg = kind == :field && combination == :raman,
            raman = combination == :plasma ? 0 : 2, plasma = combination != :raman,
            temporal = combination == :combined, prefactor_kind = :expansion,
        )
        data = device_response_data(execution, host)
        source = SP.ModalSource(execution, data; node_chunk = 2)
        tableau = device_source_tableau()
        work = SP.RungeKuttaWork(execution, state, tableau)
        operator = SP.DiagonalOperator(execution, zeros(ComplexF64, size(state)))
        flow = SP.LinearFlow(execution, operator, state)
        expected, bound = device_response_exact_steps(state, host, steps)
        device_response_steps!(state, source, flow, work, tableau, execution, steps)
        @test maximum(abs, state - expected) <= bound
        @test source.calls == 2steps && source.ready
        @test (
            @allocated device_response_steps!(
                state, source, flow, work, tableau, execution, steps
            )
        ) == 0
    end
end

@testset "Device cycle failure retains first channel and sample before publication" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    execution = _modal_source_model()
    cycle = composition_cycle(; orders = (2, 3), moment_rtol = eps(Float64))
    host, state = composition_case(cpu, :envelope; kerr = false, plasma = true, cycle)
    data = device_response_data(execution, host)
    source = SP.ModalSource(execution, data; node_chunk = 2)
    output = fill(7.0 + 2.0im, size(state))
    before, state_before = copy(output), copy(state)
    failure = try
        source(output, state, 0.53)
    catch error
        error
    end
    @test failure isa SP.SourcePlasmaError
    @test (failure.code, failure.channel, failure.node, failure.sample) ==
        (SP.PlasmaQuadratureFailure, 1, 1, 1)
    @test output == before && state == state_before
    @test !source.ready && source.calls == 1
    @test source.plasma.status_readback.host[1, 1] == 1
    @test source.plasma.status_readback.host[2, 1] == Int(SP.PlasmaQuadratureFailure)
end

@testset "Device source mixed profiles and channel failure order" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 2, true)
    gamma = 8192eps(Float64) / (1 - 8192eps(Float64))
    for kind in (:field, :envelope)
        execution = _modal_source_model()
        host, state = composition_mixed_case(cpu, kind)
        data = device_response_data(execution, host)
        source = SP.ModalSource(execution, data; node_chunk = 2)
        output = similar(state)
        source(output, state, 0.53)
        expected, scale = composition_reference(host, state, 0.53)
        @test 0 < gamma * scale < 1.0e-8
        @test maximum(abs, output - expected) <= gamma * scale
        @test composition_allocations(source, output, state, 0.53) == 0
    end
    execution = _modal_source_model()
    host, state = composition_case(cpu, :envelope; kerr = false, plasma = true)
    first, second = host.plasma.channels
    cycle = composition_cycle(; orders = (2, 3), moment_rtol = eps(Float64))
    bad_second = SP.ModalPlasmaChannel(
        second.channel, second.density, second.preionisation; quadrature = cycle
    )
    plasma = SP.ModalPlasma(
        cpu, (first, bad_second); drude = host.plasma.drude, loss = host.plasma.loss
    )
    host = SP.ModalSourceData(host.grid, host.map, host.prefactor; plasma)
    source = SP.ModalSource(
        execution, device_response_data(execution, host); node_chunk = 2
    )
    output = fill(7.0 + 2.0im, size(state))
    before = copy(output)
    failure = try
        source(output, state, 0.53)
    catch error
        error
    end
    @test failure isa SP.SourcePlasmaError
    @test (failure.code, failure.channel, failure.node, failure.sample) ==
        (SP.PlasmaQuadratureFailure, 2, 1, 1)
    @test output == before && !source.ready && source.calls == 1
end
