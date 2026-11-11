# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import CUDA
import SolpropJulia
import SolpropJulia as SP

isdefined(@__MODULE__, :device_rk_dense_budget) || include("device_rk_dense_fixture.jl")
@test CUDA.functional()
CUDA.allowscalar(false)

@testset "CUDA RK dense retains and observes the same continuous extension" begin
    device = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    context, stream = CUDA.context(), CUDA.stream()
    for rate in (0.0im, -0.25 + 0.5im, -6.0 + 0.25im, 6.0 + 0.25im)
        cpu, reference, initial, endpoint, stages = rk_dense_fixture(; rate)
        state, finish, slopes = map(
            array -> SP.upload(device, array),
            (initial, endpoint, stages)
        )
        operator = SP.DiagonalOperator(device, fill(ComplexF64(rate), size(initial)))
        work = SP.RungeKuttaObservationWork(
            device, state, rk_dense_tableau(), rk_dense_weights(), operator,
        )
        active = SP.upload(device, Bool[true, false])
        SP.retain_runge_kutta_observation!(
            work, state, finish, slopes, 0.0, 0.2, 0.2, device,
        )
        host = similar(initial)
        for position in (0.0, 0.025, 0.1, prevfloat(0.2), 0.2)
            expected = SP.observe_runge_kutta!(reference, position, Bool[true, false], cpu)
            output = SP.observe_runge_kutta!(work, position, active, device)
            SP.copy_host!(host, output, device)
            budget = device_rk_dense_budget(
                initial, stages, work.solution_weights, work.dense_weights, 0.2, position,
            )
            @test all(>(0), budget)
            @test device_rk_dense_within(host, expected, budget)
            @test all(iszero, host[2, :])
            @test CUDA.context() == context && CUDA.stream() == stream
        end
        SP.retain_runge_kutta_observation!(
            work, state, finish, slopes, 0.0, 0.2, 0.2, device,
        )
        SP.observe_runge_kutta!(work, 0.1, active, device)
        @test (
            CUDA.@allocated SP.retain_runge_kutta_observation!(
                work, state, finish, slopes, 0.0, 0.2, 0.2, device,
            )
        ) == 0
        @test (CUDA.@allocated SP.observe_runge_kutta!(work, 0.1, active, device)) == 0
        other = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
        @test_throws ArgumentError SP.observe_runge_kutta!(work, 0.1, active, other)
        @test work.ready
        for (owned, expected) in ((state, initial), (finish, endpoint), (slopes, stages))
            retained = similar(expected)
            SP.copy_host!(retained, owned, device)
            @test retained == expected
        end
        owner = work.coefficients
        CUDA.context!(device.backend.context) do
            GC.@preserve owner begin
                foreign = unsafe_wrap(
                    CUDA.CuArray, pointer(owner) + sizeof(ComplexF64),
                    size(initial); own = false
                )
                @test_throws ArgumentError SP.retain_runge_kutta_observation!(
                    work, foreign, finish, slopes, 0.0, 0.2, 0.2, device,
                )
                @test work.ready
            end
        end
    end
end

@testset "CUDA adaptive dense events preserve work and source accounting" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    device = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    reference, candidate = device_rk_dense_session(cpu), device_rk_dense_session(device)
    cpu_trace, expected = device_rk_dense_drive(reference)
    device_trace, actual = device_rk_dense_drive(candidate)
    @test device_trace == cpu_trace
    @test candidate.state.attempts == reference.state.attempts == 2
    @test length(actual) == length(expected)
    u = eps(Float64) / 2
    gamma = 256u / (1 - 256u)
    @test gamma > 0
    for (first, second) in zip(actual, expected)
        # Two committed factors and one query factor have phases below1/16;
        # the fixed budget includes complex products and finite libdevice readings.
        @test all(abs.(first - second) .<= gamma .* abs.(expected[1]))
    end
end
