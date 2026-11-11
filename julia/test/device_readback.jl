# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct ReadbackTestBackend end

function readback_test_roundtrip(work, values, execution)
    SolpropJulia.copy_upload!(work.device, values, execution)
    return SolpropJulia.readback!(work, execution)
end

@testset "CPU readback owns one exact reusable numerical buffer" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    for T in (UInt8, Int, Float32, Float64, ComplexF32, ComplexF64), shape in (
                (7,), (3, 2), (0, 2), (),
            )
        work = SolpropJulia.DeviceReadback(execution, T, shape...)
        @test work.device === work.host
        @test size(work.host) == shape && eltype(work.host) === T
        @test all(iszero, work.host)
        input = fill(T(3), shape)
        @test readback_test_roundtrip(work, input, execution) === work.host
        @test work.host == input
        @test work.host !== input
        @test sum(values(storage_test_ledger(work))) == sizeof(input)
        @test @allocated(readback_test_roundtrip(work, input, execution)) == 0
    end
    work = SolpropJulia.DeviceReadback(execution, Float64, 3)
    other = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2)
    @test_throws ArgumentError SolpropJulia.readback!(work, other)
    for T in (Any, String, Nothing, Tuple{Int, Int})
        @test_throws ArgumentError SolpropJulia.DeviceReadback(execution, T, 3)
    end
    @test_throws ArgumentError SolpropJulia.DeviceReadback(execution, Float64, -1)
    @test_throws OverflowError SolpropJulia.DeviceReadback(execution, Float64, typemax(Int))
    @test_throws OverflowError SolpropJulia.DeviceReadback(
        execution, UInt8, typemax(Int), 2,
    )
    unavailable = SolpropJulia.KernelExecution(ReadbackTestBackend(), 1)
    @test_throws ArgumentError SolpropJulia.DeviceReadback(unavailable, Float64, 3)
    @test_throws ArgumentError SolpropJulia.copy_upload!(zeros(3), zeros(3), unavailable)
end

@testset "Host upload refuses shape, precision and foreign overlap before writes" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    owner = collect(1.0:16.0)
    original = copy(owner)
    @test SolpropJulia.copy_upload!(owner, owner, execution) === owner
    GC.@preserve owner begin
        input = unsafe_wrap(Array, pointer(owner), (8,); own = false)
        output = unsafe_wrap(Array, pointer(owner, 2), (8,); own = false)
        @test_throws ArgumentError SolpropJulia.copy_upload!(output, input, execution)
        @test owner == original
    end
    @test_throws DimensionMismatch SolpropJulia.copy_upload!(owner, zeros(8), execution)
    @test_throws ArgumentError SolpropJulia.copy_upload!(
        owner, zeros(Float32, 16), execution,
    )
    @test_throws ArgumentError SolpropJulia.copy_upload!(view(owner, :), owner, execution)
    @test_throws ArgumentError SolpropJulia.copy_upload!(owner, view(owner, :), execution)
    @test owner == original
    @test SolpropJulia.copy_upload!(owner, zeros(16), execution) === owner
    @test all(iszero, owner)
end
