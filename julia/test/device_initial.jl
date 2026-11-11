# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

mutable struct _InitialDeviceModel
    inside_kernel::Bool
    downloads::Int
    downloaded_elements::Int
    allocations::Int
end
const _InitialExecution = SP.KernelExecution{_InitialDeviceModel}

struct _InitialDeviceField <: AbstractMatrix{ComplexF64}
    values::Matrix{ComplexF64}
    execution::_InitialExecution
end
Base.size(field::_InitialDeviceField) = size(field.values)
function Base.getindex(field::_InitialDeviceField, row::Int, column::Int)
    field.execution.backend.inside_kernel ||
        throw(ArgumentError("initial device field was indexed on the host"))
    return field.values[row, column]
end
SP._transfer_storage_space(::_InitialDeviceField) = :initial_device_model

function SP.copy_device!(
        destination::_InitialDeviceField, source::_InitialDeviceField,
        execution::_InitialExecution,
    )
    destination === source && source.execution === execution ||
        throw(ArgumentError("invalid initial-device context or copy"))
    return destination
end

function SP._readback_storage(execution::_InitialExecution, ::Type{T}, shape) where {T}
    execution.backend.allocations += 1
    return zeros(T, shape...), zeros(T, shape...)
end

function SP.copy_host!(destination::Array, source::Array, execution::_InitialExecution)
    execution.backend.downloads += 1
    execution.backend.downloaded_elements += length(source)
    return copyto!(destination, source)
end

function SP._each_column!(body, execution::_InitialExecution, columns)
    execution.backend.inside_kernel = true
    try
        for column in 1:columns
            body(column)
        end
    finally
        execution.backend.inside_kernel = false
    end
    return nothing
end

@testset "Already uploaded launches validate through compact owned status" begin
    execution = SP.KernelExecution(_InitialDeviceModel(false, 0, 0, 0), 1)
    values = fill(3.0 + 4.0im, 7, 3)
    field = _InitialDeviceField(values, execution)
    @test SP._march_initial_finite(field, execution)
    @test execution.backend.downloads == 1
    @test execution.backend.downloaded_elements == 3
    @test !execution.backend.inside_kernel
    @test_throws ArgumentError all(isfinite, field)
    for invalid in (ComplexF64(NaN), ComplexF64(Inf))
        values[4, 2] = invalid
        before = copy(values)
        @test !SP._march_initial_finite(field, execution)
        @test isequal(values, before)
        @test !execution.backend.inside_kernel
    end
    other = SP.KernelExecution(_InitialDeviceModel(false, 0, 0, 0), 1)
    @test_throws ArgumentError SP._march_initial_finite(field, other)
    @test other.backend.allocations == other.backend.downloads == 0
    @test !SP._march_initial_finite(values, execution)
    values[4, 2] = 3.0 + 4.0im
    @test SP._march_initial_finite(values, execution)
    @test execution.backend.allocations == 3
end
