# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import CUDA
import SolpropJulia as SP

# This is an explicitly selected device suite: unavailable CUDA is a failing
# qualification prerequisite, never a silent skipped implementation test.
@test CUDA.functional()
CUDA.allowscalar(false)

function cuda_execution_fill!(output, input, execution)
    SP.each!(execution, size(input)...) do row, column
        output[row, column] = 2input[row, column] - one(eltype(input))
    end
    return nothing
end

function cuda_execution_columns!(output, input, execution)
    SP.each_column!(execution, size(input, 2)) do column
        total = zero(eltype(input))
        for row in axes(input, 1)
            total += input[row, column]
            output[row, column] = total
        end
    end
    return nothing
end

function cuda_unowned_pointer_capture!(output, address, execution)
    nested = (address = address,)
    SP.each!(execution, size(output)...) do row, column
        output[row, column] = ifelse(
            nested.address == nested.address, zero(eltype(output)), one(eltype(output)),
        )
    end
    return nothing
end

@testset "CUDA owned execution, copies and synchronous scalar launches" begin
    extension = Base.get_extension(SP, :SolpropJuliaCUDAExt)
    @test isempty(Test.detect_ambiguities(SP, extension; recursive = false))
    prior_context, prior_stream = CUDA.context(), CUDA.stream()
    execution = SP.device_execution(Val(:cuda), 0, 2; check_body = true)
    @test !SP.oncpu(execution)
    @test execution.backend.kernel.always_inline && !execution.backend.kernel.fastmath
    @test execution.backend.stream !== prior_stream
    @test CUDA.context() == prior_context && CUDA.stream() == prior_stream
    @test_throws ArgumentError SP.device_execution(Val(:cuda), -1, 1)
    @test_throws ArgumentError SP.device_execution(
        Val(:cuda), Int(length(CUDA.devices())), 1,
    )
    @test_throws ArgumentError SP.device_execution(Val(:cuda), 0, 0)
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        host = reshape(T.(1:777), 259, 3)
        input = SP.upload(execution, host)
        output = SP.dzeros(execution, T, size(host)...)
        @test eltype(input) === T && CUDA.is_device(input)
        @test size(input) == size(host)
        received = similar(host)
        @test SP.copy_host!(received, input, execution) === received
        @test received == host
        host[1] = -1
        SP.copy_host!(received, input, execution)
        @test received[1] == one(T)
        cuda_execution_fill!(output, input, execution)
        SP.copy_host!(received, output, execution)
        # Small integral values and these operations are exactly representable.
        @test received == reshape(T.(2 .* (1:777) .- 1), 259, 3)
        cuda_execution_columns!(output, input, execution)
        SP.copy_host!(received, output, execution)
        @test received == cumsum(reshape(T.(1:777), 259, 3); dims = 1)
        @test (CUDA.@allocated cuda_execution_fill!(output, input, execution)) == 0
        @test (CUDA.@allocated cuda_execution_columns!(output, input, execution)) == 0
        @test SP.copy_device!(output, input, execution) === output
        @test SP.copy_device!(output, output, execution) === output
        @test (CUDA.@allocated SP.copy_device!(output, input, execution)) == 0
        SP.copy_host!(received, output, execution)
        @test received == reshape(T.(1:777), 259, 3)
        @test CUDA.context() == prior_context && CUDA.stream() == prior_stream
    end
    empty = SP.dzeros(execution, Float32, 0, 3)
    @test isempty(empty)
    @test SP.each!((row, column) -> error("empty kernel"), execution, 0, 3) === nothing
    @test SP.each_column!(column -> error("empty kernel"), execution, 0) === nothing
    @test SP.synchronize_execution(execution) === nothing
    @test_throws ArgumentError SP.dzeros(execution, Float32, -1)
    @test_throws OverflowError SP.dzeros(execution, Float64, typemax(Int), 2)
end

@testset "CUDA copy and capture admission preserves owners" begin
    execution = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    input = SP.upload(execution, reshape(Float32.(1:24), 8, 3))
    output = SP.dzeros(execution, Float32, 8, 3)
    prior_context, prior_stream = CUDA.context(), CUDA.stream()
    @test_throws DimensionMismatch SP.copy_host!(zeros(Float32, 24), input, execution)
    @test_throws ArgumentError SP.copy_host!(zeros(Float64, 8, 3), input, execution)
    @test_throws ArgumentError SP.copy_host!(
        view(zeros(Float32, 10, 3), 1:8, :), input, execution,
    )
    @test_throws ArgumentError SP.copy_device!(output, zeros(Float32, 8, 3), execution)
    @test_throws ArgumentError SP.copy_device!(view(output, 1:2:8, :), input, execution)
    host = ones(Float32, 8, 3)
    @test_throws ArgumentError SP.each!(execution, 8, 3) do row, column
        output[row, column] = host[row, column]
    end
    nested = (arrays = (input, output), factor = 2.0f0)
    SP.each!(execution, 8, 3) do row, column
        nested.arrays[2][row, column] = nested.factor * nested.arrays[1][row, column]
    end
    @test Array(output) == 2 .* Array(input)
    extension = Base.get_extension(SP, :SolpropJuliaCUDAExt)
    before = Array(output)
    for P in (
            Ptr{Float32}, Core.LLVMPtr{Float32, 1}, CUDA.CuPtr{Float32},
            CUDA.PtrOrCuPtr{Float32}, CUDA.CuArrayPtr{Float32},
        )
        # The address is never dereferenced. A raw pointer cannot establish an
        # owner/context, even when hidden inside a captured scalar record (#730).
        address = reinterpret(P, UInt(16))
        @test_throws ArgumentError extension._cuda_capture((address = address,), nothing)
        @test_throws ArgumentError cuda_unowned_pointer_capture!(output, address, execution)
        @test Array(output) == before
    end
    @test extension._cuda_capture((factor = 2.0f0, range = (1, 4)), nothing) === nothing
    # Independent headers have different data identities but overlap occupied bytes.
    owner = SP.upload(execution, Float32.(1:40))
    CUDA.context!(execution.backend.context) do
        GC.@preserve owner begin
            first = unsafe_wrap(CUDA.CuArray, pointer(owner), (24,); own = false)
            shifted = unsafe_wrap(CUDA.CuArray, pointer(owner, 2), (24,); own = false)
            before = Array(owner)
            @test_throws ArgumentError SP.copy_device!(shifted, first, execution)
            @test Array(owner) == before
            words = unsafe_wrap(
                CUDA.CuArray, reinterpret(CUDA.CuPtr{UInt16}, pointer(owner, 2)),
                (24,); own = false,
            )
            extension = Base.get_extension(SP, :SolpropJuliaCUDAExt)
            @test extension._cuda_overlap(first, words)
        end
    end
    extension = Base.get_extension(SP, :SolpropJuliaCUDAExt)
    @test_throws ErrorException extension._with_cuda_execution(execution) do
        error("controlled failure")
    end
    @test CUDA.context() == prior_context && CUDA.stream() == prior_stream
    if length(CUDA.devices()) > 1
        second = SP.device_execution(Val(:cuda), 1, 1)
        foreign = SP.dzeros(second, Float32, 8, 3)
        before = Array(output)
        @test_throws ArgumentError SP.copy_device!(output, foreign, execution)
        @test_throws ArgumentError SP.each!(execution, 8, 3) do row, column
            output[row, column] = foreign[row, column]
        end
        @test Array(output) == before
    end
end

@testset "CUDA compiler retains gradual binary32 underflow" begin
    execution = SP.device_execution(Val(:cuda), 0, 1)
    input = SP.upload(execution, fill(floatmin(Float32), 3, 1))
    output = SP.dzeros(execution, Float32, 3, 1)
    SP.each!(execution, 3, 1) do row, column
        output[row, column] = input[row, column] * 0.5f0
    end
    @test Array(output) == fill(ldexp(1.0f0, -127), 3, 1)
end
