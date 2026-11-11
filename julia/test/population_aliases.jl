# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SP

function population_alias_inputs(::Type{T}, ex) where {T}
    layout = SP.ResponseLayout(4, 1, 1)
    work = SP.PopulationWorkspace(ex, T, layout)
    table = SP.IonisationRateTable(zeros(T, 4, 8), 0.25, 4.0, 1.0, log(0.25), log(16.0) / 8)
    running = SP.RunningIntegralRule(T[1], T(0.125))
    # These tests exercise admission, not quadrature accuracy. Below-cutoff
    # envelope inputs take the exact-zero branch without a panel evaluation.
    cycle = SP.CycleQuadrature(
        T[-0.5, 0.5, -0.75, 0, 0.75], T[1, 1, 2 / 3, 2 / 3, 2 / 3], (2, 5);
        moment_rtol = T(2.0e-6), negligible_rate = T(1.0e-30)
    )
    return (; table, running, cycle, work)
end

function population_alias_work(work, rate; status = work.status)
    return SP.PopulationWorkspace(
        work.layout, rate, work.exposure, work.fraction, work.survival,
        work.filtered, status
    )
end

function population_alias_call(::Val{:field}, samples, data, work, ex)
    return SP.field_populations!(
        samples, data.table, data.running, work, ex; preionisation = 0.25
    )
end

function population_alias_call(::Val{:envelope}, samples, data, work, ex)
    return SP.envelope_populations!(
        samples, data.table, data.cycle, work, ex; preionisation = 0.25, dt = 0.125
    )
end

function population_alias_allocated(kind, samples, data, work, ex)
    population_alias_call(kind, samples, data, work, ex)
    return @allocated population_alias_call(kind, samples, data, work, ex)
end

function population_alias_preserved(kind, samples, data, work, ex)
    before = map(copy, SP._population_arrays(work))
    retained = map(
        copy, (
            data.table.coefficients, data.running.taps,
            data.cycle.nodes, data.cycle.weights,
        )
    )
    @test_throws ArgumentError population_alias_call(kind, samples, data, work, ex)
    @test all(
        isequal(left, right) for (left, right) in
            zip(before, SP._population_arrays(work))
    )
    @test all(
        isequal(left, right) for (left, right) in
            zip(
                retained, (
                    data.table.coefficients, data.running.taps,
                    data.cycle.nodes, data.cycle.weights,
                )
            )
    )
    return nothing
end

@testset "Population guards include foreign coefficient and scratch spans (#749)" begin
    for T in (Float32, Float64), tasks in (1, 2), kind in (Val(:field), Val(:envelope))
        ex = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        data = population_alias_inputs(T, ex)
        samples = kind isa Val{:field} ? ones(T, 4, 1) : zeros(Complex{T}, 4, 1)
        coefficients = data.table.coefficients
        GC.@preserve coefficients begin
            foreign = unsafe_wrap(Array, pointer(coefficients, 2), (4, 1); own = false)
            work = population_alias_work(data.work, foreign)
            @test !Base.mightalias(foreign, coefficients)
            population_alias_preserved(kind, samples, data, work, ex)
        end
        # A cross-element-type scratch overlap also lies entirely within the
        # preserved allocation; status writes would otherwise corrupt rate bytes.
        rate = data.work.rate
        GC.@preserve rate begin
            status = unsafe_wrap(
                Array, Ptr{UInt8}(pointer(rate, 2)), (4, 1); own = false
            )
            work = population_alias_work(data.work, rate; status)
            @test !Base.mightalias(rate, status)
            population_alias_preserved(kind, samples, data, work, ex)
        end
        @test population_alias_allocated(kind, samples, data, data.work, ex) == 0
        @test SP.plasma_failure(data.work.status).code == SP.PlasmaSuccess
        if kind isa Val{:field}
            @test data.work.rate == ones(T, 4, 1)
        else
            @test all(iszero, data.work.rate)
            @test all(==(T(0.25)), data.work.fraction)
        end
    end
end

@testset "Population guards include rule arrays and mutable index dependencies" begin
    for T in (Float32, Float64), tasks in (1, 2)
        ex = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        data = population_alias_inputs(T, ex)
        samples = zeros(Complex{T}, 4, 1)
        for prepared in (data.cycle.nodes, data.cycle.weights)
            GC.@preserve prepared begin
                foreign = unsafe_wrap(Array, pointer(prepared, 2), (4, 1); own = false)
                work = population_alias_work(data.work, foreign)
                @test !Base.mightalias(foreign, prepared)
                population_alias_preserved(Val(:envelope), samples, data, work, ex)
            end
        end
        taps_owner = ones(T, 8)
        running = SP.RunningIntegralRule(view(taps_owner, 2:4), T(0.125))
        field_data = (; data.table, running, data.cycle, data.work)
        GC.@preserve taps_owner begin
            foreign = unsafe_wrap(Array, pointer(taps_owner, 3), (4, 1); own = false)
            work = population_alias_work(data.work, foreign)
            @test !Base.mightalias(foreign, running.taps)
            population_alias_preserved(Val(:field), ones(T, 4, 1), field_data, work, ex)
        end
        # The samples' gather indices depend on rate storage even though their
        # numerical parent is independent. Detect that dependency before launch;
        # overwriting the first rate otherwise changes a later sample index.
        rate = data.work.rate
        GC.@preserve rate begin
            indices = unsafe_wrap(
                Array, Ptr{UInt8}(pointer(rate, 2)), (4,); own = false
            )
            indices .= UInt8[1, 2, 3, 4]
            parent = ones(T, 4, 1)
            indexed = view(parent, indices, :)
            before_parent, before_indices = copy(parent), copy(indices)
            @test !Base.mightalias(rate, indexed)
            population_alias_preserved(Val(:field), indexed, data, data.work, ex)
            @test parent == before_parent && indices == before_indices
        end
    end
end

@testset "Population guards preserve independent affine views and precision" begin
    for T in (Float32, Float64), tasks in (1, 2)
        ex = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        data = population_alias_inputs(T, ex)
        parent = reshape(T.(1:8), 8, 1)
        samples = view(parent, 1:2:7, :)
        original = copy(parent)
        @test population_alias_allocated(Val(:field), samples, data, data.work, ex) == 0
        @test parent == original
        @test SP.plasma_failure(data.work.status).code == SP.PlasmaSuccess
        @test data.work.rate == ones(T, 4, 1)
        before = map(copy, SP._population_arrays(data.work))
        @test_throws DimensionMismatch SP.field_populations!(
            ones(T, 3, 1), data.table, data.running, data.work, ex; preionisation = 0.25
        )
        @test all(
            isequal(left, right) for (left, right) in
                zip(before, SP._population_arrays(data.work))
        )
        for rows in (1:2:7, 7:-2:1)
            arrays = ntuple(_ -> view(zeros(T, 8, 1), rows, :), 5)
            strided = SP.PopulationWorkspace(
                data.work.layout, arrays..., zeros(UInt8, 4, 1)
            )
            @test population_alias_allocated(
                Val(:field), samples, data, strided, ex
            ) == 0
            @test SP.plasma_failure(strided.status).code == SP.PlasmaSuccess
            @test strided.rate == ones(T, 4, 1)
        end
    end
end

@testset "Population workspace writes require immutable affine indices" begin
    for T in (Float32, Float64), tasks in (1, 2), kind in (Val(:field), Val(:envelope))
        ex = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        data = population_alias_inputs(T, ex)
        samples = kind isa Val{:field} ? reshape(T[1, 2, 3, 1], 4, 1) :
            zeros(Complex{T}, 4, 1)
        for indices in ([1, 1, 3, 4], [1, 2, 3, 4])
            rate = view(fill(T(7), 4, 1), indices, :)
            others = ntuple(_ -> view(fill(T(7), 4, 1), [1, 2, 3, 4], :), 4)
            work = SP.PopulationWorkspace(
                data.work.layout, rate, others..., zeros(UInt8, 4, 1)
            )
            original, original_indices = copy(samples), copy(indices)
            population_alias_preserved(kind, samples, data, work, ex)
            @test samples == original && indices == original_indices
        end
    end
end
