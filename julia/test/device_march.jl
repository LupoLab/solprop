# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

# This host model tests orchestration and transfer sizes, not GPU arithmetic.
mutable struct _MarchReadbackDevice
    allocations::Int
    downloads::Int
    elements::Int
end
const _MarchReadbackExecution = SP.KernelExecution{_MarchReadbackDevice}

function SP.dzeros(
        execution::_MarchReadbackExecution, ::Type{T}, shape::Integer...,
    ) where {T}
    execution.backend.allocations += 1
    return zeros(T, shape...)
end

function SP._readback_storage(
        execution::_MarchReadbackExecution, ::Type{T}, shape,
    ) where {T}
    return SP.dzeros(execution, T, shape...), zeros(T, shape...)
end

function SP.copy_device!(
        destination::Array, source::Array, execution::_MarchReadbackExecution,
    )
    destination === source || copyto!(destination, source)
    return destination
end

function SP.copy_host!(
        destination::Array, source::Array, execution::_MarchReadbackExecution,
    )
    execution.backend.downloads += 1
    execution.backend.elements += length(source)
    return copyto!(destination, source)
end

function SP.upload(execution::_MarchReadbackExecution, source::AbstractArray)
    destination = SP.dzeros(execution, eltype(source), size(source)...)
    return copyto!(destination, source)
end

function SP._each!(body, execution::_MarchReadbackExecution, rows, columns, heavy)
    for column in 1:columns, row in 1:rows
        body(row, column)
    end
    return nothing
end

function SP._each_column!(body, execution::_MarchReadbackExecution, columns)
    for column in 1:columns
        body(column)
    end
    return nothing
end

_march_readback_execution() = SP.KernelExecution(_MarchReadbackDevice(0, 0, 0), 1)
_march_readback_tableau() = SP.EmbeddedTableau(
    [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
)

@testset "Source completion reads only owned per-column failure indices" begin
    execution = _march_readback_execution()
    field = ones(ComplexF64, 17, 3)
    status = SP.DeviceReadback(execution, Int, 3)
    @test SP._source_finite!(status, field, execution, :test_source) === nothing
    @test execution.backend.downloads == 1
    @test execution.backend.elements == 3
    field[7, 1] = Inf
    field[2, 2] = NaN
    failure = try
        SP._source_finite!(status, field, execution, :test_source)
    catch error
        error
    end
    @test failure isa SP.NonfiniteSourceError
    @test (failure.stage, failure.row, failure.column) == (:test_source, 7, 1)
    @test status.host == [7, 2, 0]
    before = copy(status.device)
    @test_throws ArgumentError SP._source_finite!(
        status, field, _march_readback_execution(), :test_source,
    )
    @test status.device == before
    @test execution.backend.downloads == 2
    short = SP.DeviceReadback(execution, Int, 2)
    @test_throws DimensionMismatch SP._source_finite!(short, field, execution, :short)
    @test all(iszero, short.device)
end

@testset "Device RK retains compact norm readback and preserves the CPU graph" begin
    execution = _march_readback_execution()
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    tableau = _march_readback_tableau()
    state = reshape(ComplexF64.(1:51), 17, 3) ./ 64
    work = SP.RungeKuttaWork(execution, state, tableau)
    reference = SP.RungeKuttaWork(cpu, state, tableau)
    @test work.norm_readback.device === work.norm_parts
    @test work.norm_readback.host !== work.norm_parts
    @test size(work.norm_readback.host) == (4, 3)
    @test reference.norm_readback === nothing
    @test reference.norm_scale === nothing
    ledger = Dict{Symbol, Int}()
    SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "work", work)
    @test sum(values(ledger)) == 8sizeof(state) + 2 * 4 * 3 * sizeof(Float64)
    mask = fill(true, 17)
    mask[3] = false
    weights = collect(range(0.5, 1.5; length = 17))
    for owner in (work, reference)
        owner.candidate .= state .* (1 + 0.125im)
        owner.defect .= state .* (1.0e-5 + 2.0e-5im)
    end
    expected = SP.rk_error_norm!(reference, state, mask, weights, 1.0e-4, 1.0e-8, cpu)
    actual = SP.rk_error_norm!(work, state, mask, weights, 1.0e-4, 1.0e-8, execution)
    # Each protected positive scaled reduction uses fewer than this many rounded
    # operations. This fixture has a positive, well-conditioned denominator.
    operations, unit = 32length(state) + 16size(state, 2), eps(Float64) / 2
    budget = operations * unit / (1 - operations * unit)
    @test 0 < budget < 1.0e-10
    @test isapprox(actual, expected; rtol = budget, atol = 0)
    @test execution.backend.downloads == 2
    @test execution.backend.elements == 24
    @test SP._rk_finished_finite!(work, work.candidate, execution)
    @test execution.backend.elements == 36
    work.candidate[3, 2] = NaN
    @test !SP._rk_finished_finite!(work, work.candidate, execution)
    @test !SP._rk_finished_finite!(reference, work.candidate, cpu)
    # Nonfinite data outside the norm mask must refuse before the defect pass.
    old_downloads = execution.backend.downloads
    @test isinf(SP.rk_error_norm!(work, state, mask, weights, 1.0e-4, 1.0e-8, execution))
    @test execution.backend.downloads == old_downloads + 1
    before = copy(work.norm_parts)
    @test_throws ArgumentError SP.rk_error_norm!(
        work, state, mask, weights, 1.0e-4, 1.0e-8, _march_readback_execution(),
    )
    @test isequal(work.norm_parts, before)
    @test_throws ArgumentError SP.rk_error_norm!(
        reference, state, mask, weights, 1.0e-4, 1.0e-8, execution,
    )
    old_allocations = execution.backend.allocations
    @test_throws ArgumentError SP.RungeKuttaWork(execution, ComplexF32.(state), tableau)
    @test execution.backend.allocations == old_allocations
end

@testset "Device linear scratch and cached factors use the prepared execution" begin
    execution = _march_readback_execution()
    state = fill(0.5 + 0.25im, 7, 2)
    operator = SP.DiagonalOperator(execution, zeros(ComplexF64, size(state)))
    for tabulate in (false, true)
        before = execution.backend.allocations
        flow = SP.LinearFlow(execution, operator, state; tabulate)
        @test flow.failures isa SP.DeviceReadback
        @test execution.backend.allocations == before + (tabulate ? 3 : 2)
        output = similar(state)
        @test flow(output, state, 0.0, 0.1) === nothing
        # A structural zero generator preserves the field exactly.
        @test output == state
        @test flow(output, output, 0.0, 0.1) === nothing
        @test output == state
        fill!(output, 0)
        @test flow(output, state, 0.1, 0.1) === nothing
        @test output == state
        old_allocations = execution.backend.allocations
        @test flow(output, state, 0.0, 0.1) === nothing
        @test execution.backend.allocations == old_allocations
        @test_throws ArgumentError flow(flow.exponent, state, 0.0, 0.1)
    end
    before = execution.backend.allocations
    @test_throws ArgumentError SP.LinearFlow(execution, operator, ComplexF32.(state))
    @test execution.backend.allocations == before
end
