# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function execution_fill!(buffer, identifiers, execution; heavy = false)
    SolpropJulia.each!(execution, size(buffer)...; heavy) do row, column
        buffer[row, column] = row + 1000column
        identifiers[row, column] = Threads.threadid()
    end
    return nothing
end

function execution_fill_heavy!(buffer, identifiers, execution)
    return execution_fill!(buffer, identifiers, execution; heavy = true)
end

function execution_columns!(buffer, identifiers, execution)
    SolpropJulia.each_column!(execution, size(buffer, 2)) do column
        total = 0
        for row in axes(buffer, 1)
            total += row
            buffer[row, column] = total
            identifiers[row, column] = Threads.threadid()
        end
    end
    return nothing
end

function execution_boxed_body()
    counter = 0
    return (row, column) -> (counter += row + column)
end

function execution_type_body(element_type)
    return (row, column) -> element_type(row + column)
end

function execution_nested!(buffer, execution)
    captured = (buffer, (offset = 7,))
    inner = (row, column) -> row + captured[2].offset * column
    SolpropJulia.each!(execution, size(buffer)...) do row, column
        captured[1][row, column] = inner(row, column)
    end
    return nothing
end

function execution_copy!(output, input, execution)
    SolpropJulia.each!(execution, size(output)...) do row, column
        output[row, column] = input[row, column]
    end
    return nothing
end

function execution_copy_columns!(output, input, execution)
    SolpropJulia.each_column!(execution, size(output, 2)) do column
        for row in axes(output, 1)
            output[row, column] = input[row, column]
        end
    end
    return nothing
end

@testset "Packed Boolean captures are refused before launch" begin
    backend = SolpropJulia.KernelAbstractions.CPU()
    available = Threads.nthreads(:default)
    for tasks in unique((1, min(2, available), available)),
            check_body in (false, true)
        execution = SolpropJulia.KernelExecution(backend, tasks, check_body)
        packed = BitArray([isodd(row + column) for row in 1:3, column in 1:3])
        pointer = SolpropJulia.StrideArraysCore.PtrArray(packed)
        variants = (
            packed, view(packed, 1:2:3, :), reshape(vec(packed), 3, 3),
            transpose(packed), PermutedDimsArray(packed, (2, 1)),
            pointer, view(pointer, 1:2:3, :), reshape(pointer, 1, 9),
            transpose(pointer), PermutedDimsArray(pointer, (2, 1)),
        )
        GC.@preserve packed begin
            for input in variants
                output = fill(false, size(input))
                @test_throws ArgumentError execution_copy!(output, input, execution)
                @test_throws ArgumentError execution_copy_columns!(output, input, execution)
                @test iszero(output)
                @test_throws ArgumentError execution_copy!(input, output, execution)
                @test_throws ArgumentError execution_copy_columns!(input, output, execution)
            end
        end
        owned = SolpropJulia.upload(execution, packed)
        @test owned isa Matrix{Bool}
        @test owned == packed
        output = fill(false, size(owned))
        execution_copy!(output, owned, execution)
        @test output == owned
        @test @allocated(execution_copy!(output, owned, execution)) == 0
        execution_copy_columns!(output, owned, execution)
        @test output == owned
        @test @allocated(execution_copy_columns!(output, owned, execution)) == 0
    end
end

@testset "Owned numerical execution" begin
    backend = SolpropJulia.KernelAbstractions.CPU()
    serial = SolpropJulia.KernelExecution(backend, 1, true)
    @test SolpropJulia.oncpu(serial)
    @test_throws ArgumentError SolpropJulia.KernelExecution(backend, 0)
    @test_throws ArgumentError SolpropJulia.KernelExecution(backend, -1)
    original = reshape(collect(1.0:24.0), 6, 4)
    owned = SolpropJulia.upload(serial, view(original, 2:5, 1:3))
    @test owned == original[2:5, 1:3]
    original[2, 1] = -1.0
    @test owned[1, 1] == 2.0
    @test !Base.mightalias(original, owned)
    @test SolpropJulia.dzeros(serial, ComplexF64, 7, 3) == zeros(ComplexF64, 7, 3)
    @test size(SolpropJulia.dzeros(serial, Float64, 0, 4)) == (0, 4)
    mask = SolpropJulia.dzeros(serial, Bool, Int32(3), Int16(4))
    @test mask isa Matrix{Bool}
    @test mask == fill(false, 3, 4)
    independent = SolpropJulia.dzeros(serial, Bool, 3, 4)
    mask[1, 1] = true
    @test !independent[1, 1] && !Base.mightalias(mask, independent)
    scalar = SolpropJulia.dzeros(serial, Float64)
    @test scalar isa Array{Float64, 0} && scalar[] == 0.0
    @test_throws ArgumentError SolpropJulia.dzeros(serial, Float64, -1, 4)
    @test_throws ArgumentError SolpropJulia.upload(serial, Any[1, 2])
    @test_throws ArgumentError SolpropJulia.dzeros(serial, Any, 2)

    available = Threads.nthreads(:default)
    for tasks in unique((1, min(2, available), available))
        execution = SolpropJulia.KernelExecution(backend, tasks, true)
        for (n_rows, n_columns) in ((0, 4), (4, 0), (1, 1), (17, 1), (19, 3), (257, 11))
            buffer = zeros(Int, n_rows, n_columns)
            identifiers = zeros(Int, n_rows, n_columns)
            execution_fill!(buffer, identifiers, execution)
            @test buffer == [row + 1000column for row in 1:n_rows, column in 1:n_columns]
            @test length(unique(identifiers)) <= tasks
            @test @allocated(execution_fill!(buffer, identifiers, execution)) == 0
            execution_fill_heavy!(buffer, identifiers, execution)
            @test buffer == [row + 1000column for row in 1:n_rows, column in 1:n_columns]
            # A compiled caller excludes the test scope's boxed keyword tuple.
            @test @allocated(execution_fill_heavy!(buffer, identifiers, execution)) == 0
            execution_columns!(buffer, identifiers, execution)
            @test buffer == [row * (row + 1) ÷ 2 for row in 1:n_rows, _ in 1:n_columns]
            @test length(unique(identifiers)) <= tasks
            @test @allocated(execution_columns!(buffer, identifiers, execution)) == 0
        end
        buffer = zeros(Int, 4096, tasks)
        identifiers = zeros(Int, size(buffer))
        execution_fill!(buffer, identifiers, execution)
        @test length(unique(identifiers)) == tasks
        execution_columns!(buffer, identifiers, execution)
        @test length(unique(identifiers)) == tasks
        execution_nested!(buffer, execution)
        @test buffer == [row + 7column for row in 1:4096, column in 1:tasks]
        @test @allocated(execution_nested!(buffer, execution)) == 0
        strided = view(buffer, 1:2:4096, :)
        execution_nested!(strided, execution)
        @test strided == [row + 7column for row in 1:2048, column in 1:tasks]
        @test @allocated(execution_nested!(strided, execution)) == 0
    end

    @test_throws ArgumentError SolpropJulia.each!((row, column) -> nothing, serial, -1, 1)
    @test_throws OverflowError SolpropJulia.each!(
        (row, column) -> nothing, serial, typemax(Int), 2
    )
    @test_throws ArgumentError SolpropJulia.each_column!(column -> nothing, serial, -1)
    @test_throws ArgumentError SolpropJulia.each!(execution_boxed_body(), serial, 2, 2)
    @test_throws ArgumentError SolpropJulia.each!(
        execution_type_body(Float64), serial, 2, 2
    )
    bad_array = Any[1]
    @test_throws ArgumentError SolpropJulia.each!(serial, 1, 1) do row, column
        bad_array[row] = column
    end
    unsupported = SolpropJulia.KernelExecution(:unqualified, 1)
    @test !SolpropJulia.oncpu(unsupported)
    @test_throws ArgumentError SolpropJulia.each!(
        (row, column) -> nothing, unsupported, 1, 1
    )
    @test_throws ArgumentError SolpropJulia.each_column!(column -> nothing, unsupported, 1)
    @test SolpropJulia.synchronize_execution(serial) === nothing
    @test_throws ArgumentError SolpropJulia.synchronize_execution(unsupported)
    @test_throws ArgumentError SolpropJulia.device_execution(Val(:unqualified), 0, 1)
    source, destination = [1.0], [-2.0]
    @test_throws ArgumentError SolpropJulia.copy_host!(destination, source, unsupported)
    @test_throws ArgumentError SolpropJulia.copy_device!(destination, source, unsupported)
    @test destination == [-2.0] && source == [1.0]
end


struct ExecutionParameterCarrier{P} end
struct ExecutionParameterRecord{P}
    value::P
end
primitive type ExecutionAddress 64 end

@testset "Host capture metadata separates pointer values from address-free types" begin
    for P in (Ptr{Float32}, Core.LLVMPtr{Float64, 1})
        address = reinterpret(P, UInt(0x1000))
        for value in (
                address, Val(address), (tag = :stage, nested = (address,)),
                ExecutionParameterRecord(address),
                Val(Tuple{Val{address}}), Val(Union{Nothing, Val{address}}),
                Val(Tuple{Val{address}, T} where {T}),
            )
            @test SolpropJulia._capture_parameter_pointer(value)
        end
        for T in (
                Val{address}, ExecutionParameterCarrier{address},
                Array{ExecutionParameterCarrier{address}, 1},
            )
            @test SolpropJulia._capture_parameter_pointer(T)
        end
    end
    for value in (
            1, :stage, (kind = :stage, index = 2), Float64, Ptr{Float32},
            Core.LLVMPtr{Float32, 1}, Val(Ptr{Float32}),
            Tuple{Ptr{Float32}, T} where {T}, Union{Nothing, Ptr{Float32}},
            ExecutionParameterCarrier{Tuple{Int, Ptr{Float32}}},
        )
        @test !SolpropJulia._capture_parameter_pointer(value)
    end
    address = reinterpret(ExecutionAddress, UInt(0x1000))
    @test !SolpropJulia._capture_parameter_pointer(Val(address))
    family = Union{Ptr, Core.LLVMPtr, ExecutionAddress}
    @test SolpropJulia._capture_parameter_pointer(Val(address), family)
    @test !SolpropJulia._capture_parameter_pointer(Val(ExecutionAddress), family)
end
