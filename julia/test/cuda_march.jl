# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import CUDA
import LinearAlgebra
import SolpropJulia as SP

@test CUDA.functional()
CUDA.allowscalar(false)

mutable struct _CUDAMarchSource{E}
    execution::E
    calls::Int
end

function (source::_CUDAMarchSource)(output, input, position)
    source.calls += 1
    coefficient = 0.125im
    SP.each!(source.execution, size(input)...) do row, column
        output[row, column] = coefficient * input[row, column]
    end
    return nothing
end

function cuda_march_tableau()
    return SP.EmbeddedTableau(
        [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
    )
end

function cuda_march_fixture(execution, steps; tabulate = false)
    field = reshape(ComplexF64.(1:51), 17, 3) ./ 64
    generator = fill(-0.25 + 0.5im, size(field))
    device_state = SP.upload(execution, field)
    operator = SP.DiagonalOperator(execution, generator)
    flow = SP.LinearFlow(execution, operator, device_state; tabulate)
    source = _CUDAMarchSource(execution, 0)
    norm = SP.MarchNorm(
        execution, fill(true, 17), fill(true, 17), ones(17), 1.0e-4, 1.0e-10,
    )
    finish = 1 / 32
    step = finish / steps
    schedule = SP.MarchSchedule(
        planes = [finish], breakpoints = [0.0, finish], hard_kinks = Float64[],
        stretch = 1.1, step_planes = 0, replay_steps = fill(step, steps),
        replay_edges = collect(0:step:finish),
    )
    control = SP.PIControl(
        safety = 0.8, max_factor = 2.0, min_factor = 0.2, beta1 = 0.3, beta2 = -0.1,
        lag = 0, reject_free = false, estimator_order = 1, start_growth = 100.0,
        dz_min = 1.0e-12, dz_max = step / 4, auto_init = false,
    )
    session = SP.RungeKuttaSession(
        execution, field, 0.0, step, cuda_march_tableau(), control, norm,
        schedule, source, flow; capacity = 16,
    )
    return session, field, step
end

function cuda_march_drive(session; work_limit)
    trace = Tuple{SP.SessionEvent, Float64}[]
    while !session.state.finished
        report = SP.advance!(session; work_limit, deadline = 60.0)
        push!(trace, (report.kind, report.z))
        SP.acknowledge!(session, report.generation)
    end
    return trace
end

function cuda_march_host(session)
    output = Matrix{ComplexF64}(undef, size(session.field))
    if SP.oncpu(session.execution)
        copyto!(output, session.field)
    else
        SP.copy_host!(output, session.field, session.execution)
    end
    return output
end

@testset "CUDA binary64 RK owns short trials, compact reductions and delivery" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    gpu = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    context, stream = CUDA.context(), CUDA.stream()
    for steps in (1, 8), tabulate in (false, true), work_limit in (1, 16)
        reference, initial, step = cuda_march_fixture(cpu, steps; tabulate)
        candidate, _, _ = cuda_march_fixture(gpu, steps; tabulate)
        cpu_trace = cuda_march_drive(reference; work_limit)
        gpu_trace = cuda_march_drive(candidate; work_limit)
        actual, expected = cuda_march_host(candidate), cuda_march_host(reference)
        @test cpu_trace == gpu_trace
        @test candidate.state.attempts == reference.state.attempts == steps
        @test candidate.state.rejected == reference.state.rejected == 0
        @test candidate.source.calls == reference.source.calls == 2steps
        @test all(isfinite, actual)
        @test CUDA.context() == context && CUDA.stream() == stream

        # Check every nonzero homogeneous factor used by this constant-step Heun
        # graph against MPFR before using its primitive allowance in the march bound.
        phases = ComplexF64[-step * (-0.25 + 0.5im), step * (-0.25 + 0.5im)]
        phase_device = SP.upload(gpu, reshape(phases, 2, 1))
        factor_device = SP.dzeros(gpu, ComplexF64, 2, 1)
        SP.each!(gpu, 2, 1) do row, column
            factor_device[row, column] = exp(phase_device[row, column])
        end
        factors = zeros(ComplexF64, 2, 1)
        SP.copy_host!(factors, factor_device, gpu)
        setprecision(BigFloat, 256) do
            for row in 1:2
                exact = exp(Complex{BigFloat}(phases[row]))
                for (value, target) in zip(
                        (real(factors[row]), imag(factors[row])),
                        (real(exact), imag(exact)),
                    )
                    allowance = 8eps(Float64) * abs(target)
                    @test allowance > 0
                    @test abs(BigFloat(value) - target) <= allowance
                end
            end
        end
        # Two-stage Heun uses fewer than 256 rounded scalar operations per cell
        # and step, including the independently checked exponential allowance.
        # The scalar linear witness has sensitivity <2 over this 1/32 m window.
        operations, unit = 512steps, eps(Float64) / 2
        relative = 2 * operations * unit / (1 - operations * unit)
        bound = relative * LinearAlgebra.norm(initial)
        @test bound > 0
        @test LinearAlgebra.norm(actual - expected) <= bound
        # The exact discrete map gives a second reference independent of either
        # implementation's stage loop, including its finite Heun truncation.
        exact = setprecision(BigFloat, 256) do
            h = BigFloat(step)
            factor = exp(h * (-big(0.25) + big(0.5) * im)) *
                (1 + h * big(0.125) * im - (h * big(0.125))^2 / 2)
            ComplexF64.(Complex{BigFloat}.(initial) .* factor^steps)
        end
        @test LinearAlgebra.norm(actual - exact) <= bound

        owned = candidate.work.norm_readback
        @test owned.device === candidate.work.norm_parts
        @test owned.host isa Matrix{Float64} && size(owned.host) == (4, 3)
        ledger = Dict{Symbol, Int}()
        SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "work", candidate.work)
        @test sum(values(ledger)) == 8sizeof(initial) + 2 * 4 * 3 * sizeof(Float64)
        SP.close_session!(candidate)
        SP.close_session!(reference)
    end
end

@testset "CUDA finite-status errors and wrong execution preserve accepted storage" begin
    execution = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    host = ones(ComplexF64, 9, 2)
    host[6, 1] = Inf
    host[2, 2] = NaN
    field = SP.upload(execution, host)
    failures = SP.DeviceReadback(execution, Int, 2)
    failure = try
        SP._source_finite!(failures, field, execution, :device_test)
    catch error
        error
    end
    @test failure isa SP.NonfiniteSourceError
    @test (failure.stage, failure.row, failure.column) == (:device_test, 6, 1)
    @test failures.host == [6, 2]
    before = copy(failures.host)
    other = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    @test_throws ArgumentError SP._source_finite!(failures, field, other, :device_test)
    @test failures.host == before
    returned = similar(host)
    SP.copy_host!(returned, field, execution)
    @test isequal(returned, host)
end
