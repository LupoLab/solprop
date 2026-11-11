# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import CUDA
import SolpropJulia as SP

# Explicit device qualification: no unavailable-device skip or scalar fallback.
@test CUDA.functional()
CUDA.allowscalar(false)

function cuda_storage_ledger(value)
    ledger = Dict{Symbol, Int}()
    SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "storage", value)
    return ledger
end

@testset "CUDA live spans and retained allocations describe different boundaries" begin
    execution = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    extension = Base.get_extension(SP, :SolpropJuliaCUDAExt)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        host = T.(1:64)
        owner = SP.upload(execution, host)
        first, second = view(owner, 2:9), view(owner, 33:40)
        @test first isa extension.CUDAOwnedArray
        @test parent(first) === first
        @test SP._transfer_storage_space(owner) == SP._transfer_storage_space(first)
        @test SP._transfer_arrays_alias(owner, first)
        @test !SP._transfer_arrays_alias(first, second)
        @test !SP._transfer_arrays_alias(host, owner)
        @test !SP._transfer_arrays_alias(owner, host)
        @test SP._transfer_storage_span(first)[2] - SP._transfer_storage_span(first)[1] ==
            UInt(sizeof(first))
        ledger = cuda_storage_ledger((first, second, reshape(first, 2, 4)))
        @test length(ledger) == 1
        @test only(values(ledger)) == sizeof(owner)
        @test only(values(cuda_storage_ledger((owner, first, second)))) == sizeof(owner)
        # An explicit wrapper keeps the existing conservative full-parent rule.
        wrapped = SubArray(owner, (2:2:10,))
        @test SP._transfer_storage_span(wrapped) == SP._transfer_storage_span(owner)
        @test SP._transfer_storage_space(wrapped) == SP._transfer_storage_space(owner)
        @test SP._transfer_writable_destination(wrapped) === nothing

        CUDA.context!(execution.backend.context) do
            GC.@preserve owner begin
                foreign = unsafe_wrap(CUDA.CuArray, pointer(owner, 3), (4,); own = false)
                bytes = unsafe_wrap(
                    CUDA.CuArray, reinterpret(CUDA.CuPtr{UInt8}, pointer(owner, 4)),
                    (sizeof(T),); own = false,
                )
                @test SP._transfer_arrays_alias(foreign, first)
                @test SP._transfer_arrays_alias(first, foreign)
                @test SP._transfer_arrays_alias(bytes, foreign)
                @test SP._transfer_arrays_alias(foreign, bytes)
                @test only(values(cuda_storage_ledger(foreign))) == sizeof(foreign)
                @test SP._memory_array_owner(foreign)[1] !==
                    SP._memory_array_owner(owner)[1]
                @test_throws ArgumentError SP.copy_device!(
                    foreign, view(owner, 2:5), execution,
                )
            end
        end
        @test Array(owner) == host
        released = SP.upload(execution, host)
        CUDA.unsafe_free!(released)
        @test_throws ArgumentError SP._transfer_storage_space(released)
        @test_throws ArgumentError SP._transfer_storage_span(released)
        @test_throws ArgumentError SP._memory_array_owner(released)
        @test_throws ArgumentError SP._transfer_arrays_alias(owner, released)
        @test_throws ArgumentError SP._transfer_arrays_alias(released, owner)
        @test !SP._transfer_arrays_alias(T[], released)
        @test_throws ArgumentError SP.copy_upload!(released, host, execution)
    end
end

@testset "CUDA context tokens and cross-space gather dependencies" begin
    execution = SP.device_execution(Val(:cuda), 0, 1)
    owner = SP.upload(execution, collect(1.0:16.0))
    host_indices = [1, 3, 5, 7]
    device_indices = SP.upload(execution, host_indices)
    for indices in (host_indices, device_indices)
        gathered = SubArray(owner, (indices,))
        for wrapped in (
                gathered, reshape(gathered, 2, 2), transpose(reshape(gathered, 2, 2)),
            )
            @test SP._transfer_arrays_alias(indices, wrapped)
            @test SP._transfer_arrays_alias(wrapped, indices)
            @test_throws ArgumentError SP._transfer_writable_destination(wrapped)
        end
    end
    prior_context, prior_stream = CUDA.context(), CUDA.stream()
    CUDA.CuContext(execution.backend.device) do context
        CUDA.context!(context) do
            other = CUDA.CuArray{Float64, 1, CUDA.DeviceMemory}(undef, 16)
            try
                @test SP._transfer_storage_space(owner) != SP._transfer_storage_space(other)
                @test !SP._transfer_arrays_alias(owner, other)
                @test !SP._transfer_arrays_alias(other, owner)
                @test_throws ArgumentError SP.copy_upload!(other, zeros(16), execution)
                @test CUDA.context() == context
            finally
                CUDA.unsafe_free!(other)
                CUDA.synchronize()
            end
        end
    end
    @test CUDA.context() == prior_context && CUDA.stream() == prior_stream
end

@testset "CUDA persistent readback and reusable uploads preserve numerical storage" begin
    execution = SP.device_execution(Val(:cuda), 0, 2; check_body = true)
    extension = Base.get_extension(SP, :SolpropJuliaCUDAExt)
    prior_context, prior_stream = CUDA.context(), CUDA.stream()
    for T in (UInt8, Int, Float32, Float64, ComplexF32, ComplexF64)
        work = SP.DeviceReadback(execution, T, 7, 3)
        @test work.device isa extension.CUDAOwnedArray{T, 2}
        @test work.host isa Matrix{T}
        @test work.host !== work.device
        @test all(iszero, SP.readback!(work, execution))
        owner = work.device.data[]
        input = reshape(T.(1:21), 7, 3)
        for unused in 1:3
            @test SP.copy_upload!(work.device, input, execution) === work.device
            @test SP.readback!(work, execution) === work.host
            @test work.host == input
            @test work.device.data[] === owner
            fill!(work.host, zero(T))
        end
        @test sum(values(cuda_storage_ledger(work))) == 2sizeof(input)
        @test_throws DimensionMismatch SP.copy_upload!(
            work.device, zeros(T, 3, 7), execution,
        )
        @test_throws ArgumentError SP.copy_upload!(
            work.device, zeros(Bool, 7, 3), execution,
        )
        @test_throws ArgumentError SP.copy_upload!(
            work.device, view(input, :, :), execution,
        )
        @test_throws ArgumentError SP.copy_upload!(work.device, work.device, execution)
        @test SP.readback!(work, execution) == input
        other = SP.device_execution(Val(:cuda), 0, 2)
        @test_throws ArgumentError SP.readback!(work, other)
        CUDA.unsafe_free!(work.device)
        @test_throws ArgumentError SP.readback!(work, execution)
    end
    empty_work = SP.DeviceReadback(execution, Float64, 0, 3)
    @test SP.readback!(empty_work, execution) === empty_work.host
    @test SP.copy_upload!(empty_work.device, zeros(0, 3), execution) === empty_work.device
    @test CUDA.context() == prior_context && CUDA.stream() == prior_stream

    work = SP.DeviceReadback(execution, Float64, 32)
    input = collect(1.0:32.0)
    sentinel = ErrorException("controlled error after an enqueued transfer")
    caught = try
        extension._with_cuda_storage_transfer(execution) do
            copyto!(work.device, input)
            GC.gc(false)
            throw(sentinel)
        end
    catch error
        error
    end
    @test caught === sentinel
    @test SP.readback!(work, execution) == input
    @test CUDA.context() == prior_context && CUDA.stream() == prior_stream
end

@testset "CUDA FFT retained numerical arrays participate in caller alias admission" begin
    execution = SP.device_execution(Val(:cuda), 0, 1)
    for T in (Float32, Float64)
        time = SP.dzeros(execution, T, 8, 2)
        spectrum = SP.dzeros(execution, Complex{T}, 5, 2)
        plan = SP.TemporalFFT(execution, time, spectrum)
        for raw in (plan.forward, plan.inverse)
            buffers = SP._retained_storage(raw)
            if raw.plan.buffer === nothing
                @test buffers === ()
                @test !SP._retained_alias(spectrum, raw)
            else
                @test only(buffers) === raw.plan.buffer
                scratch = raw.plan.buffer
                SP.copy_upload!(scratch, zeros(eltype(scratch), size(scratch)), execution)
                CUDA.context!(execution.backend.context) do
                    GC.@preserve scratch begin
                        output = unsafe_wrap(
                            CUDA.CuArray,
                            reinterpret(CUDA.CuPtr{ComplexF64}, pointer(scratch, 2)),
                            (1, 2);
                            own = false,
                        )
                        before = Array(scratch)
                        @test SP._retained_alias(output, (plan,))
                        # The actual export boundary sees the plan in a scripted
                        # owner's graph and refuses before any field inspection or
                        # destination write. This does not admit GPU sessions.
                        session = (
                            state = (
                                closed = false, pending = true, poisoned = false,
                                generation = 1,
                            ),
                            field = SP.dzeros(execution, ComplexF64, 1, 2),
                            plan = plan, execution = execution,
                        )
                        weights = SP.dzeros(execution, Float64, 1, 2)
                        @test_throws ArgumentError SP.export_session!(
                            output, session, 1, weights, false, false,
                        )
                        @test Array(scratch) == before
                    end
                end
            end
        end
        complex_plan = SP.TemporalFFT(
            execution, SP.dzeros(execution, Complex{T}, 8, 2),
            SP.dzeros(execution, Complex{T}, 8, 2),
        )
        @test SP._retained_storage(complex_plan.forward) === ()
        @test SP._retained_storage(complex_plan.inverse) === ()
    end
end
