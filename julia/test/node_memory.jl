# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset
import SolpropJulia

function node_memory_window_case(kind, tasks)
    execution = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), tasks, true,
    )
    frequency = kind == :field ? 5 : 8
    windows = fill(false, frequency, 2)
    windows[1:3, 1] .= true
    windows[4:end, 2] .= true
    grid = SolpropJulia.StatisticsGrid(
        collect(0.0:(frequency - 1)), collect(-4.0:3.0), ones(frequency); windows,
    )
    selection = SolpropJulia.StatisticsSelection(
        (SolpropJulia.StatisticSlot(:peak_power_window, 1, 4),), (),
    )
    prepared = SolpropJulia.ObservationPreparation(
        execution, grid, selection; kind, n_modes = 2,
        power_weight = SolpropJulia.ObservationWeight(execution, ones(frequency, 2)),
    )
    return SolpropJulia.ObservationWorkspace(execution, prepared; node_chunk = 1)
end

function node_memory_window_allocated(work, field)
    SolpropJulia.observe!(work, field, 0.0)
    return @allocated SolpropJulia.observe!(work, field, 0.0)
end

@testset "Window reshapes retain one numerical storage owner (#698)" begin
    for kind in (:field, :envelope), tasks in (1, 2)
        work = node_memory_window_case(kind, tasks)
        windows = work.windows
        pair = windows.pair
        @test parent(windows.samples) !== windows.samples
        @test parent(windows.weighted) !== windows.weighted
        @test pointer(windows.samples) == pointer(pair.samples)
        @test pointer(windows.weighted) == pointer(pair.weighted)
        ledger = Dict{Symbol, Int}()
        SolpropJulia._memory_arrays!(ledger, IdDict{Any, Bool}(), "window", windows)
        payload = filter(entry -> !endswith(string(first(entry)), "estimated_plan"), ledger)
        @test length(payload) == 4
        @test sum(values(payload)) == 4sizeof(pair.samples)
        @test !haskey(payload, Symbol("window.samples"))
        @test !haskey(payload, Symbol("window.weighted"))
        field = zeros(ComplexF64, length(work.data.grid.omega), 2)
        field[2, :] .= [1 + 2im, 3 - 1im]
        before = copy(field)
        @test node_memory_window_allocated(work, field) == 0
        @test field == before
        @test all(isfinite, work.values)
        # One Fourier tone has constant analytic intensity. The previously
        # qualified eight-point FFT envelope plus the magnitude square is bounded
        # by 128 unit roundoffs; doubling the field's positive bin is exact.
        scale = kind == :field ? 4.0 : 1.0
        expected = scale .* [5.0, 10.0] ./ 64
        @test isapprox(work.values[1:2], expected; rtol = 64eps(Float64), atol = 0.0)
        @test iszero(work.values[3]) && iszero(work.values[4])
    end
end
