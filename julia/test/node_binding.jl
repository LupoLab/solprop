# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

@testset "Request node capacities share only immutable numerical inputs" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    grid = SP.ModalSourceGrid(
        execution; kind = :envelope, n_time = 8, dt = 0.25,
        crop_index = collect(0:7), omega = collect(1.0:8.0), active = fill(true, 8),
        omega_window = ones(8), source_window = ones(8),
        source_window_derivative = zeros(8),
    )
    map = SP.ModalSourceMap(
        execution; synthesis = ones(3, 1), projection = fill(1 / 3, 1, 3),
    )
    data = SP.ModalSourceData(grid, map, SP.SourcePrefactor(execution, ones(ComplexF64, 8, 1)))
    operator = SP.DiagonalOperator(execution, zeros(ComplexF64, 8, 1))
    original = SP.ModalPreparation(execution, data, operator, nothing, 3)
    smaller = SP.with_node_chunk(original, 1)
    middle = SP.with_node_chunk(original, 2)
    @test (original.node_chunk, smaller.node_chunk, middle.node_chunk) == (3, 1, 2)
    for prepared in (smaller, middle)
        @test prepared.execution === original.execution
        @test prepared.data === original.data
        @test prepared.operator === original.operator
        @test prepared.power_weight === original.power_weight
    end
    for invalid in (0, -1, 4, typemax(Int))
        @test_throws ArgumentError SP.with_node_chunk(original, invalid)
    end
    @test original.node_chunk == 3
end
