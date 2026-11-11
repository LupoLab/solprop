# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import CUDA
import LinearAlgebra
import SolpropJulia as SP

# Explicit device qualification: unavailable hardware is a failure, never a skip.
@test CUDA.functional()
CUDA.allowscalar(false)
if !isdefined(@__MODULE__, :low_storage_device_fixture)
    include("low_storage_device.jl")
end

function cuda_low_storage_factor(execution, step)
    phase = reshape(ComplexF64[0.5step * (-0.25 + 0.5im)], 1, 1)
    input = SP.upload(execution, phase)
    output = SP.dzeros(execution, ComplexF64, 1, 1)
    SP.each!(execution, 1, 1) do row, column
        output[row, column] = exp(input[row, column])
    end
    actual = zeros(ComplexF64, 1, 1)
    SP.copy_host!(actual, output, execution)
    setprecision(BigFloat, 256) do
        exact = exp(Complex{BigFloat}(only(phase)))
        for (value, target) in zip(
                (real(only(actual)), imag(only(actual))), (real(exact), imag(exact)),
            )
            # A separately tested primitive hypothesis for the actual dyadic
            # operands, not an inferred guarantee for arbitrary CUDA exp inputs.
            allowance = 8eps(Float64) * abs(target)
            @test allowance > 0
            @test abs(BigFloat(value) - target) <= allowance
        end
    end
    return nothing
end

@testset "CUDA low-storage replay matches an independent discrete scalar map" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    gpu = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    context, stream = CUDA.context(), CUDA.stream()
    for steps in (1, 8)
        cuda_low_storage_factor(gpu, 1 / (32steps))
        for tabulate in (false, true), work_limit in (1, 16)
            candidate, initial, step = low_storage_device_fixture(gpu; steps, tabulate)
            reference, _, _ = low_storage_device_fixture(cpu; steps, tabulate)
            try
                @test low_storage_device_drive(candidate; work_limit) ==
                    low_storage_device_drive(reference; work_limit)
                # These are explicit readouts after the numerical session finishes.
                # Neither advance! nor norm evaluation downloads a state volume.
                actual = low_storage_device_host(candidate)
                expected = low_storage_device_reference(initial, steps, step)
                bound = low_storage_device_bound(initial, steps, step)
                @test bound > 0
                @test LinearAlgebra.norm(actual - expected) <= bound
                @test LinearAlgebra.norm(actual - reference.field) <= bound
                @test all(isfinite, actual) && all(iszero, actual[2, :])
                @test candidate.source.calls == 4steps
                @test candidate.counts.attempted_source_evaluations == 4steps
                @test candidate.counts.committed_source_evaluations == 4steps
                @test candidate.counts.norm_evaluations == steps
                @test candidate.state.accepted == steps && candidate.state.rejected == 0
                @test !candidate.state.poisoned && !candidate.state.in_flight
                @test CUDA.context() == context && CUDA.stream() == stream

                work = candidate.work
                @test work.norm_readback.device === work.norm_parts
                @test work.finite_readback.device === work.finite_columns
                @test work.norm_readback.host isa Matrix{Float64}
                @test work.finite_readback.host isa Vector{Bool}
                @test size(work.norm_readback.host) == (2, 3)
                @test size(work.finite_readback.host) == (3,)
                @test sizeof(work.norm_readback.host) + sizeof(work.finite_readback.host) ==
                    17 * 3
                ledger = Dict{Symbol, Int}()
                SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "work", work)
                @test length(ledger) == 8
                @test sum(values(ledger)) == 3sizeof(initial) + 17 + 34 * 3
            finally
                SP.close_session!(candidate)
                SP.close_session!(reference)
            end
        end
    end
end

@testset "CUDA low-storage norms transfer protected partials and all finite flags" begin
    execution = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    host = reshape(ComplexF64.(1:51), 17, 3) ./ 64
    field = SP.upload(execution, host)
    mask = fill(true, 17)
    mask[2] = false
    work = SP.LowStorageWork(execution, field, SP.RK4IPTableau(); norm_mask = mask)
    magnitude, finite = SP._low_storage_norm!(work, field, execution)
    expected = setprecision(BigFloat, 256) do
        sqrt(sum(abs2, Complex{BigFloat}.(host[mask, :])))
    end
    operations, unit = 32length(host) + 16size(host, 2), eps(Float64) / 2
    bound = operations * unit / (1 - operations * unit) * expected
    @test bound > 0
    @test abs(BigFloat(magnitude) - expected) <= bound
    @test finite && all(work.finite_readback.host)

    # A masked NaN must still poison finiteness while leaving the selected norm
    # reduction unchanged; the finite flags inspect the complete stored state.
    host[2, 2] = NaN
    SP.copy_upload!(field, host, execution)
    magnitude, finite = SP._low_storage_norm!(work, field, execution)
    @test !finite && work.finite_readback.host == [true, false, true]
    @test abs(BigFloat(magnitude) - expected) <= bound
    before = copy(work.norm_readback.host)
    other = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    @test_throws ArgumentError SP._low_storage_norm!(work, field, other)
    @test work.norm_readback.host == before
    @test isequal(Array(field), host)
end

@testset "CUDA low-storage failures retain attempted work and poison in-flight state" begin
    execution = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    context, stream = CUDA.context(), CUDA.stream()
    for failed_call in 1:4
        session, initial, _ = low_storage_device_fixture(execution; fail_on = failed_call)
        try
            trace = low_storage_device_drive(session)
            @test last(trace)[1] == SP.SESSION_FAILED
            @test session.state.poisoned && session.state.in_flight
            @test session.source.calls == session.counts.attempted_source_evaluations ==
                failed_call
            @test session.counts.committed_source_evaluations == session.attempts.count == 0
            @test session.state.z == session.state.flight_z == 0.0
            @test session.state.flight_dz == first(session.schedule.replay_steps)
            @test_throws ArgumentError SP.advance!(session)
            if failed_call == 1
                @test low_storage_device_host(session) == initial
            end
            @test CUDA.context() == context && CUDA.stream() == stream
        finally
            SP.close_session!(session)
        end
    end
    for failure in (:finish, :nonfinite, :accepted)
        services = LowStorageDeviceServices(failure, 0, 0)
        session, _, _ = low_storage_device_fixture(execution; services)
        try
            trace = low_storage_device_drive(session)
            committed = failure === :accepted
            @test last(trace)[1] == SP.SESSION_FAILED
            @test session.state.poisoned == !committed
            @test session.state.in_flight == !committed
            @test session.counts.attempted_source_evaluations == 4
            @test session.counts.committed_source_evaluations == (committed ? 4 : 0)
            @test session.state.accepted == (committed ? 1 : 0)
            @test services.finished == 1
            @test services.accepted == (committed ? 1 : 0)
            @test CUDA.context() == context && CUDA.stream() == stream
        finally
            SP.close_session!(session)
        end
    end
    session, _, _ = low_storage_device_fixture(execution; nonfinite_on = 4)
    try
        @test last(low_storage_device_drive(session))[1] == SP.SESSION_FAILED
        @test session.state.poisoned && session.state.in_flight
        @test session.counts.norm_evaluations == 1
        @test session.counts.committed_source_evaluations == 0
        @test session.work.finite_readback.host == fill(false, 3)
    finally
        SP.close_session!(session)
    end
end

@testset "CUDA low-storage admission and exact snapshot receipts precede new work" begin
    execution = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    session, initial, _ = low_storage_device_fixture(execution)
    try
        other = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
        @test_throws ArgumentError SP.low_storage_trial!(
            session.work, session.field, 0.0, first(session.schedule.replay_steps),
            session.tableau, session.source, session.flow, other,
        )
        @test low_storage_device_host(session) == initial
        @test session.source.calls == 0
        @test_throws ArgumentError SP.LowStorageWork(
            execution, ComplexF32.(initial), SP.RK4IPTableau(),
        )
        @test_throws ArgumentError SP.LowStorageSession(
            execution, ComplexF32.(initial), 0.0, session.tableau, session.norm,
            session.schedule, session.source, session.flow;
            arithmetic = SP.SourceArithmetic(Float32),
        )
        @test session.source.calls == 0

        @test SP.request_snapshot!(session) == (1, false)
        @test SP.request_snapshot!(session) == (1, true)
        report = SP.advance!(session)
        @test report.kind == SP.SESSION_LAUNCH
        @test session.source.calls == 0
        SP.acknowledge!(session, report.generation)
        report = SP.advance!(session)
        @test report.kind == SP.SESSION_SNAPSHOT && session.state.accepted == 1
        @test session.counts.committed_source_evaluations == 4
        @test_throws ArgumentError SP.acknowledge!(session, report.generation + 1)
        SP.acknowledge!(session, report.generation)
        @test_throws ArgumentError SP.acknowledge!(session, report.generation)
        flag = session.stop
        setter = SP.stop_setter_address()
        GC.@preserve flag ccall(
            Ptr{Cvoid}(setter), Cvoid, (Ptr{UInt8},), Ptr{UInt8}(SP.stop_address(flag)),
        )
        stopped = SP.advance!(session)
        @test stopped.kind == SP.SESSION_STOPPED && !stopped.poisoned
        @test session.source.calls == 4
        @test !session.state.in_flight
        SP.acknowledge!(session, stopped.generation)
    finally
        SP.close_session!(session)
    end
    @test_throws ArgumentError SP.advance!(session)
end
