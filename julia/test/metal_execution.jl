# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import Metal
import SolpropJulia as SP

# Select this device suite explicitly. Lack of a device is a failed prerequisite,
# not a skipped implementation check. DF32 has its separate #739 qualification.
@test Metal.functional()
# Every admitted array uses PrivateStorage; disable its host scalar fallback.
Metal.GPUArrays.allowscalar(false)

function metal_execution_fill!(output, input, execution)
    SP.each!(execution, size(input)...) do row, column
        output[row, column] = 2input[row, column] - one(eltype(input))
    end
    return nothing
end

function metal_execution_columns!(output, input, execution)
    SP.each_column!(execution, size(input, 2)) do column
        total = zero(eltype(input))
        for row in axes(input, 1)
            total += input[row, column]
            output[row, column] = total
        end
    end
    return nothing
end

function metal_capture_value!(output, value, execution)
    nested = (value = value,)
    SP.each!(execution, size(output)...) do row, column
        output[row, column] = ifelse(
            nested.value == nested.value, zero(eltype(output)), one(eltype(output)),
        )
    end
    return nothing
end

function metal_other_task(body)
    return fetch(
        @async try
            body()
        catch error
            error
        end
    )
end

@testset "Metal owned task, copies and synchronous scalar launches" begin
    extension = Base.get_extension(SP, :SolpropJuliaMetalExt)
    @test isempty(Test.detect_ambiguities(SP, extension; recursive = false))
    prior_device = Metal.device()
    prior_queue = Metal.global_queue(prior_device)
    execution = SP.device_execution(Val(:metal), 0, 2; check_body = true)
    @test !SP.oncpu(execution)
    @test execution.backend.owner === current_task()
    @test execution.backend.queue === Metal.global_queue(execution.backend.device)
    @test Metal.device() == prior_device && Metal.global_queue(prior_device) === prior_queue
    for kernel in (extension._metal_each_kernel, extension._metal_column_kernel)
        @test SP.KernelAbstractions.compiler_options(kernel(execution.backend.kernel)) ==
            (; fastmath = false, always_inline = true)
    end
    @test_throws ArgumentError SP.device_execution(Val(:metal), -1, 1)
    @test_throws ArgumentError SP.device_execution(
        Val(:metal), Int(length(Metal.devices())), 1,
    )
    @test_throws ArgumentError SP.device_execution(Val(:metal), 0, 0)
    for T in (Float32, ComplexF32, Int32)
        host = reshape(T.(1:777), 259, 3)
        input = SP.upload(execution, host)
        output = SP.dzeros(execution, T, size(host)...)
        @test input isa Metal.MtlArray{T, 2, Metal.PrivateStorage}
        @test size(input) == size(host)
        received = similar(host)
        @test SP.copy_host!(received, input, execution) === received
        @test received == host
        host[1] = -1
        SP.copy_host!(received, input, execution)
        @test received[1] == one(T)
        metal_execution_fill!(output, input, execution)
        SP.copy_host!(received, output, execution)
        # All values, products and partial sums are small exactly represented integers.
        @test received == reshape(T.(2 .* (1:777) .- 1), 259, 3)
        metal_execution_columns!(output, input, execution)
        SP.copy_host!(received, output, execution)
        @test received == cumsum(reshape(T.(1:777), 259, 3); dims = 1)
        @test (Metal.@allocated metal_execution_fill!(output, input, execution)) == 0
        @test (Metal.@allocated metal_execution_columns!(output, input, execution)) == 0
        @test SP.copy_device!(output, input, execution) === output
        @test SP.copy_device!(output, output, execution) === output
        @test (Metal.@allocated SP.copy_device!(output, input, execution)) == 0
        SP.copy_host!(received, output, execution)
        @test received == reshape(T.(1:777), 259, 3)
        copied = SP.upload(execution, input)
        @test copied !== input && Array(copied) == Array(input)
        @test Metal.device() == prior_device
        @test Metal.global_queue(prior_device) === prior_queue
    end
    host = reshape(Float32.(1:36), 12, 3)
    @test Array(SP.upload(execution, view(host, 1:2:12, :))) == host[1:2:12, :]
    @test Array(SP.upload(execution, trues(3, 2))) == fill(true, 3, 2)
    empty = SP.dzeros(execution, Float32, 0, 3)
    @test isempty(empty)
    @test SP.each!((row, column) -> error("empty kernel"), execution, 0, 3) === nothing
    @test SP.each_column!(column -> error("empty kernel"), execution, 0) === nothing
    @test SP.synchronize_execution(execution) === nothing
    @test_throws ArgumentError SP.dzeros(execution, Float32, -1)
    @test_throws OverflowError SP.dzeros(execution, Float32, typemax(Int), 2)
    for T in (Float64, ComplexF64, Int128, UInt128, Nothing, Ptr{Float32})
        @test_throws ArgumentError SP.dzeros(execution, T, 3)
    end
end

@testset "Metal ownership and capture refusals precede writes" begin
    execution = SP.device_execution(Val(:metal), 0, 1; check_body = true)
    input = SP.upload(execution, reshape(Float32.(1:24), 8, 3))
    output = SP.dzeros(execution, Float32, 8, 3)
    extension = Base.get_extension(SP, :SolpropJuliaMetalExt)
    prior_device = Metal.device()
    prior_queue = Metal.global_queue(prior_device)
    @test_throws DimensionMismatch SP.copy_host!(zeros(Float32, 24), input, execution)
    @test_throws ArgumentError SP.copy_host!(zeros(Float64, 8, 3), input, execution)
    @test_throws ArgumentError SP.copy_host!(
        view(zeros(Float32, 10, 3), 1:8, :), input, execution,
    )
    @test_throws ArgumentError SP.copy_device!(output, zeros(Float32, 8, 3), execution)
    @test_throws ArgumentError SP.copy_device!(view(output, 1:2:8, :), input, execution)
    @test_throws ArgumentError SP.upload(execution, view(input, 1:2:8, :))
    shared = Metal.MtlArray{Float32, 2, Metal.SharedStorage}(ones(Float32, 8, 3))
    @test_throws ArgumentError SP.upload(execution, shared)
    @test_throws ArgumentError SP.copy_device!(output, shared, execution)
    host = ones(Float32, 8, 3)
    @test_throws ArgumentError SP.each!(execution, 8, 3) do row, column
        output[row, column] = host[row, column]
    end
    nested = (arrays = (input, output), factor = 2.0f0)
    SP.each!(execution, 8, 3) do row, column
        nested.arrays[2][row, column] = nested.factor * nested.arrays[1][row, column]
    end
    @test Array(output) == 2 .* Array(input)
    before = Array(output)
    for value in (2.0, Int128(2), UInt128(2), Ptr{Float32}(16), pointer(input))
        @test_throws ArgumentError metal_capture_value!(output, value, execution)
        @test Array(output) == before
    end
    @test extension._metal_capture((factor = 2.0f0, range = (1, 4)), execution) === nothing
    owner = SP.upload(execution, Float32.(1:40))
    GC.@preserve owner begin
        first = unsafe_wrap(Metal.MtlArray, pointer(owner), (24,))
        shifted = unsafe_wrap(Metal.MtlArray, pointer(owner, 2), (24,))
        saved = Array(owner)
        @test_throws ArgumentError SP.copy_device!(shifted, first, execution)
        @test Array(owner) == saved
        address = pointer(owner, 2)
        words = unsafe_wrap(
            Metal.MtlArray, Metal.MtlPtr{UInt16}(
                address.buffer, address.offset, address.managed,
            ), (24,)
        )
        @test extension._metal_overlap(first, words)
        ledger, seen = Dict{Symbol, Int}(), IdDict{Any, Bool}()
        SP._memory_arrays!(ledger, seen, "owner", owner)
        SP._memory_arrays!(ledger, seen, "first", first)
        SP._memory_arrays!(ledger, seen, "shifted", shifted)
        SP._memory_arrays!(ledger, seen, "words", words)
        @test length(ledger) == 1 && sum(values(ledger)) == Int(owner.data[].buffer.length)
    end
    @test_throws ErrorException extension._with_metal_execution(execution) do
        error("controlled failure")
    end
    @test Metal.device() == prior_device && Metal.global_queue(prior_device) === prior_queue
    for body in (
            () -> SP.dzeros(execution, Float32, 8, 3),
            () -> SP.upload(execution, ones(Float32, 8, 3)),
            () -> SP.copy_device!(output, input, execution),
            () -> SP.copy_host!(host, input, execution),
            () -> metal_execution_fill!(output, input, execution),
            () -> SP.synchronize_execution(execution),
        )
        @test metal_other_task(body) isa ArgumentError
        @test Array(output) == before
        @test host == ones(Float32, 8, 3)
    end
    if length(Metal.devices()) > 1
        second = SP.device_execution(Val(:metal), 1, 1)
        foreign = SP.dzeros(second, Float32, 8, 3)
        @test_throws ArgumentError SP.copy_device!(output, foreign, execution)
        @test Array(output) == before
    end
end
