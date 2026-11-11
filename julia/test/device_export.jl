# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

# Host orchestration model: actual CUDA compilation/arithmetic is tested separately.
mutable struct _ExportReadbackDevice
    allocations::Int
    downloads::Int
    elements::Int
    fail_download::Int
end
const _ExportReadbackExecution = SP.KernelExecution{_ExportReadbackDevice}

function SP.dzeros(
        execution::_ExportReadbackExecution, ::Type{T}, shape::Integer...,
    ) where {T}
    execution.backend.allocations += 1
    return zeros(T, shape...)
end

function SP._readback_storage(
        execution::_ExportReadbackExecution, ::Type{T}, shape,
    ) where {T}
    return SP.dzeros(execution, T, shape...), zeros(T, shape...)
end

function SP.copy_device!(destination::Array, source::Array, ::_ExportReadbackExecution)
    destination === source || copyto!(destination, source)
    return destination
end

function SP.copy_host!(
        destination::Array, source::Array, execution::_ExportReadbackExecution,
    )
    device = execution.backend
    device.downloads += 1
    device.elements += length(source)
    if device.downloads == device.fail_download
        destination[firstindex(destination)] = source[firstindex(source)]
        throw(ArgumentError("injected partial device download"))
    end
    return copyto!(destination, source)
end

function SP.upload(execution::_ExportReadbackExecution, source::AbstractArray)
    return copyto!(SP.dzeros(execution, eltype(source), size(source)...), source)
end

function SP._each!(body, ::_ExportReadbackExecution, rows, columns, heavy)
    for column in 1:columns, row in 1:rows
        body(row, column)
    end
    return nothing
end

function SP._each_column!(body, ::_ExportReadbackExecution, columns)
    for column in 1:columns
        body(column)
    end
    return nothing
end

_export_readback_execution() = SP.KernelExecution(_ExportReadbackDevice(0, 0, 0, 0), 1)

function _export_session_fixture(execution, field; weight = ones(size(field)))
    owned_weight = weight === nothing ? nothing : SP.ObservationWeight(execution, weight)
    return (
        execution = execution, field = SP.upload(execution, field),
        exports = SP._FieldExport(execution, owned_weight; shape = size(field)),
        state = (
            closed = false, pending = true, poisoned = false, generation = 1, z = 0.25,
        ),
    )
end

@testset "Device export retains the complete binary64 destination recipe" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    field = reshape(
        ComplexF64[
            0.0 + 0.0im, complex(-0.0, 0.0), complex(0.0, -0.0),
            complex(-0.0, -0.0), 0.5 + 0.25im, -0.25 + 0.5im,
            1.0e-200 - 2.0e-200im, -2.0e100 + 1.0e100im,
        ], 4, 2,
    )
    for unit in (true, false), power in (false, true), optics in (false, true)
        execution = _export_readback_execution()
        weights = ones(4, 2)
        unit || (weights[4, 2] = 4.0)
        session = _export_session_fixture(execution, field; weight = weights)
        reference = _export_session_fixture(cpu, field; weight = weights)
        output, expected = similar(field), similar(field)
        SP.export_session!(expected, reference, 1, weights, power, optics)
        allocated = execution.backend.allocations
        SP.export_session!(output, session, 1, session.exports.weight.data, power, optics)
        # The host model executes the same recipe on retained data, so signed zeros
        # and all exact copy/storage contracts must be preserved, not just a norm.
        @test isequal(output, expected)
        @test isequal(session.field, field)
        @test session.state.pending && session.state.generation == 1
        @test execution.backend.allocations == allocated
        @test execution.backend.downloads == (power ? 3 : 2)
        @test execution.backend.elements == length(field) + (power ? 4 : 2) * 2
        ledger = Dict{Symbol, Int}()
        SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "export", session.exports)
        @test sum(values(ledger)) == 48length(field) + 32size(field, 2)
        @test session.exports.transfer.field.host !== session.exports.transfer.field.device
    end
    execution = _export_readback_execution()
    session = _export_session_fixture(execution, field; weight = nothing)
    output = similar(field)
    @test SP.export_session!(output, session, 1, zeros(0, 0), false, false) === nothing
    @test isequal(output, field)
    @test_throws ArgumentError SP._FieldExport(execution, nothing)
    @test SP._FieldExport(cpu, nothing).transfer === nothing
end

@testset "Device export errors preserve the destination and retry generation" begin
    execution = _export_readback_execution()
    field = fill(0.5 + 0.25im, 4, 2)
    session = _export_session_fixture(execution, field)
    output = fill(7.0 + 8.0im, size(field))
    before = copy(output)
    for invalid in (0.0, -1.0, Inf, NaN)
        weights = fill(invalid, size(field))
        @test_throws ArgumentError SP.export_session!(
            output, session, 1, weights, true, true,
        )
        @test isequal(output, before)
        @test isequal(session.field, field)
        @test session.state.pending && session.state.generation == 1
    end
    for state in (
            merge(session.state, (; closed = true)),
            merge(session.state, (; pending = false)),
            merge(session.state, (; poisoned = true)),
            merge(session.state, (; generation = 2)),
        )
        previous = execution.backend.downloads
        @test_throws ArgumentError SP.export_session!(
            output, merge(session, (; state)), 1, ones(4, 2), false, false,
        )
        @test output == before
        @test execution.backend.downloads == previous
    end
    for destination in (
            session.field, session.exports.transfer.field.device,
            session.exports.transfer.field.host,
            view(session.exports.transfer.field.host, :, :),
        )
        retained = copy(destination)
        previous = execution.backend.downloads
        @test_throws ArgumentError SP.export_session!(
            destination, session, 1, ones(4, 2), false, false,
        )
        @test isequal(destination, retained)
        @test execution.backend.downloads == previous
    end
    other = merge(session, (; execution = _export_readback_execution()))
    @test_throws ArgumentError SP.export_session!(
        output, other, 1, ones(4, 2), false, false,
    )
    @test output == before
    narrow = merge(session, (; field = ComplexF32.(field)))
    @test_throws ArgumentError SP.export_session!(
        output, narrow, 1, ones(4, 2), false, false,
    )
    @test output == before

    huge = fill(complex(ldexp(1.5, 1023), ldexp(0.5, 1023)), size(field))
    overflow = _export_session_fixture(execution, huge)
    @test_throws "physical field export exceeds binary64 range" SP.export_session!(
        output, overflow, 1, fill(4.0, 4, 2), true, true,
    )
    @test output == before
    SP.export_session!(output, overflow, 1, ones(4, 2), true, false)
    @test isequal(output, huge)
    @test overflow.state.pending
end

@testset "A partially failed download changes only export-owned storage" begin
    for failure in (1, 2)
        execution = _export_readback_execution()
        field = fill(0.5 + 0.25im, 4, 2)
        session = _export_session_fixture(execution, field)
        output = fill(7.0 + 8.0im, size(field))
        before = copy(output)
        execution.backend.fail_download = failure
        @test_throws "injected partial device download" SP.export_session!(
            output, session, 1, ones(4, 2), false, false,
        )
        @test output == before
        @test session.field == field
        @test session.state.pending && session.state.generation == 1
        execution.backend.fail_download = 0
        SP.export_session!(output, session, 1, ones(4, 2), false, false)
        @test isequal(output, field)
        @test session.state.pending
    end
end

@testset "Device power-weight tables stay owned and read back compact facts" begin
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    execution = _export_readback_execution()
    coefficients = reshape([2.0, 3.0, 4.0, 5.0, 0.125, -0.25, 0.5, -0.125], 4, 2, 1)
    table = SP.ChebyshevTable(cpu, [0.0, 1.0], coefficients)
    direct = SP.ObservationWeight(execution, table, (2, 2))
    @test direct.data.breakpoints !== table.breakpoints
    @test direct.data.coefficients !== table.coefficients
    terms = reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1)
    factors = SP.ChebyshevTable(cpu, [0.0, 1.0], reshape([2.0, 0.25], 1, 2, 1))
    expansion = SP.ObservationWeight(execution, terms, factors)
    @test expansion.weights.coefficients !== factors.coefficients
    @test expansion.weights.breakpoints !== factors.breakpoints
    @test expansion.data !== terms
    for (weight, scratch, expected) in (
            (direct, Float64[], reshape([1.9375, 3.125, 3.75, 5.0625], 2, 2)),
            (expansion, zeros(1), reshape([1.875, 3.75, 5.625, 7.5], 2, 2)),
        )
        output = fill(7.0, 2, 2)
        status = SP.DeviceReadback(execution, Int, 2, 2)
        previous = execution.backend.elements
        SP.evaluate_weight!(output, weight, 0.25, execution; work = scratch, status)
        # All inputs and polynomial values in this witness are exact dyadic values.
        @test output == expected
        @test execution.backend.elements == previous + 4
        @test status.host[1, :] == [0, 0]
        @test status.host[2, :] == [1, 1]
        fill!(output, 7.0)
        @test_throws ArgumentError SP.evaluate_weight!(
            output, weight, 0.25, execution; work = scratch,
        )
        @test all(==(7.0), output)
        @test_throws ArgumentError SP.evaluate_weight!(
            output, weight, 0.25, execution; work = scratch,
            status = SP.DeviceReadback(_export_readback_execution(), Int, 2, 2),
        )
        @test all(==(7.0), output)
        @test_throws DimensionMismatch SP.evaluate_weight!(
            output, weight, 0.25, execution; work = scratch,
            status = SP.DeviceReadback(execution, Int, 1, 2),
        )
        @test all(==(7.0), output)
    end
end
