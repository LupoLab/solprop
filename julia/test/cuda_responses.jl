# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import CUDA
import SolpropJulia as SP

include("modal_source_fixture.jl")
include("device_response_fixture.jl")

@test CUDA.functional()
CUDA.allowscalar(false)

@testset "CUDA binary64 Raman and plasma match independent source and RK references" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    execution = SP.device_execution(Val(:cuda), 0, 2; check_body = true)
    context, stream = CUDA.context(), CUDA.stream()
    gamma = 8192eps(Float64) / (1 - 8192eps(Float64))
    for kind in (:field, :envelope), combination in (:raman, :plasma, :combined),
            chunk in (1, 2, 3)
        host, initial = composition_case(
            cpu, kind; kerr = combination == :combined,
            thg = kind == :field && combination == :raman,
            raman = combination == :plasma ? 0 : 2, plasma = combination != :raman,
            temporal = combination == :combined,
            prefactor_kind = combination == :plasma ? :table : :expansion,
        )
        data = device_response_data(execution, host)
        source = SP.ModalSource(execution, data; node_chunk = chunk)
        state = SP.upload(execution, initial)
        output = SP.dzeros(execution, ComplexF64, size(initial)...)
        for z in (0.0, 0.53, 1.0)
            @test source(output, state, z) === nothing
            expected, scale = composition_reference(host, initial, z)
            actual = device_response_host(output, execution)
            @test 0 < gamma * scale < 1.0e-8
            @test device_response_rate_domain(host, initial, z) < gamma * scale / 16
            @test maximum(abs, actual - expected) <= gamma * scale
            @test device_response_host(state, execution) == initial
            @test all(iszero, actual[.!host.grid.active, :])
            for (samples, coarse) in (
                    (source.quadrature_fine, false), (source.quadrature_coarse, true),
                )
                reference, magnitude = composition_reference(
                    host, initial, z; temporal_active = false, coarse
                )
                @test maximum(abs, device_response_host(samples, execution) - reference) <=
                    gamma * magnitude
            end
        end
        source(output, state, 0.53)
        @test (CUDA.@allocated source(output, state, 0.53)) == 0
        @test CUDA.context() == context && CUDA.stream() == stream
        for work in source.raman
            @test work.spectrum isa CUDA.CuArray{ComplexF64, 1, CUDA.DeviceMemory}
            @test work.host_spectrum isa Vector{ComplexF64}
            @test device_response_host(work.spectrum, execution) == work.host_spectrum
            @test SP.oncpu(work.coefficient_execution)
        end
        if source.plasma !== nothing
            compact = source.plasma.status_readback
            @test compact.device isa CUDA.CuArray{Int, 2, CUDA.DeviceMemory}
            @test compact.host isa Matrix{Int}
            @test size(compact.host) == (2, chunk)
            for channel in data.plasma.channels
                @test channel.channel.rate.coefficients isa CUDA.CuArray
                if channel.quadrature !== nothing
                    @test channel.quadrature.nodes isa CUDA.CuArray
                    @test channel.quadrature.weights isa CUDA.CuArray
                end
            end
        end
        if chunk == 2
            @test all(iszero, device_response_host(source.nodes, execution)[:, 2])
            if source.plasma !== nothing
                @test all(
                    iszero, device_response_host(
                        source.plasma.workspace.status, execution
                    )[:, 2]
                )
            end
        end
        if chunk == 2
            tableau = device_source_tableau()
            work = SP.RungeKuttaWork(execution, state, tableau)
            operator = SP.DiagonalOperator(execution, zeros(ComplexF64, size(initial)))
            flow = SP.LinearFlow(execution, operator, state)
            for steps in (1, 8)
                SP.copy_upload!(state, initial, execution)
                before = source.calls
                expected, bound = device_response_exact_steps(initial, host, steps)
                device_response_steps!(state, source, flow, work, tableau, execution, steps)
                actual = device_response_host(state, execution)
                @test maximum(abs, actual - expected) <= bound
                @test source.calls - before == 2steps && source.ready
                @test (
                    CUDA.@allocated device_response_steps!(
                        state, source, flow, work, tableau, execution, steps
                    )
                ) == 0
            end
        end
    end
end

@testset "CUDA plasma status and invalid preparation preserve owned storage" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    execution = SP.device_execution(Val(:cuda), 0, 2; check_body = true)
    work = SP.DeviceReadback(execution, Int, 2, 3)
    host_status = zeros(UInt8, 17, 3)
    host_status[16, 1] = UInt8(SP.PlasmaQuadratureFailure)
    host_status[1, 2] = UInt8(SP.PlasmaNonfiniteInput)
    status = SP.upload(execution, host_status)
    failure = SP._source_plasma_failure!(work, status, execution)
    @test (failure.code, failure.node, failure.sample) ==
        (SP.PlasmaQuadratureFailure, 1, 16)
    @test device_response_host(status, execution) == host_status
    @test (CUDA.@allocated SP._source_plasma_failure!(work, status, execution)) == 0
    host, initial = composition_case(cpu, :envelope; plasma = true)
    data = device_response_data(execution, host)
    @test_throws ArgumentError SP.ModalSourceData(
        data.grid, data.map, data.prefactor; plasma = data.plasma
    )
    cycle = composition_cycle(; orders = (2, 3), moment_rtol = eps(Float64))
    failed_host, _ = composition_case(cpu, :envelope; plasma = true, cycle)
    failed = SP.ModalSource(
        execution, device_response_data(execution, failed_host); node_chunk = 2
    )
    state = SP.upload(execution, initial)
    output = SP.upload(execution, fill(7.0 + 2.0im, size(initial)))
    before = device_response_host(output, execution)
    error = try
        failed(output, state, 0.53)
    catch exception
        exception
    end
    @test error isa SP.SourcePlasmaError
    @test (error.code, error.channel, error.node, error.sample) ==
        (SP.PlasmaQuadratureFailure, 1, 1, 1)
    @test device_response_host(output, execution) == before
    @test device_response_host(state, execution) == initial
    @test !failed.ready && failed.calls == 1
end
