# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import LinearAlgebra
import SolpropJulia as SP

include("modal_source_fixture.jl")

# This serial host model exercises source orchestration and exact transfer sizes.
# It supplies no evidence for CUDA arithmetic, compilation or context restoration.
mutable struct _ModalSourceDeviceModel
    allocations::Int
    downloads::Int
    elements::Int
end
const _ModalSourceModelExecution = SP.KernelExecution{_ModalSourceDeviceModel}

function SP.dzeros(
        execution::_ModalSourceModelExecution, ::Type{T}, shape::Integer...,
    ) where {T}
    execution.backend.allocations += 1
    return zeros(T, shape...)
end

function SP.upload(execution::_ModalSourceModelExecution, input::AbstractArray)
    output = SP.dzeros(execution, eltype(input), size(input)...)
    return copyto!(output, input)
end

function SP.copy_device!(
        output::Array, input::Array, execution::_ModalSourceModelExecution,
    )
    size(output) == size(input) || throw(DimensionMismatch("model copy shape mismatch"))
    eltype(output) === eltype(input) || throw(ArgumentError("model copy type mismatch"))
    output === input || copyto!(output, input)
    return output
end

function SP.copy_host!(
        output::Array, input::Array, execution::_ModalSourceModelExecution,
    )
    execution.backend.downloads += 1
    execution.backend.elements += length(input)
    return copyto!(output, input)
end

function SP._readback_storage(
        execution::_ModalSourceModelExecution, ::Type{T}, shape,
    ) where {T}
    return SP.dzeros(execution, T, shape...), zeros(T, shape...)
end

_source_model_capture(array::Array) = nothing
_source_model_capture(array::AbstractArray) =
    throw(ArgumentError("model kernels require complete owned maps, not array views"))
@generated function _source_model_capture(value::T) where {T}
    checks = [
        :(_source_model_capture(getfield(value, $index))) for index in 1:fieldcount(T)
    ]
    return quote
        $(checks...)
        nothing
    end
end

function SP._each!(body, execution::_ModalSourceModelExecution, rows, columns, heavy)
    SP._check_kernel_body(typeof(body))
    _source_model_capture(body)
    for column in 1:columns, row in 1:rows
        body(row, column)
    end
    return nothing
end

function SP._each_column!(body, execution::_ModalSourceModelExecution, columns)
    SP._check_kernel_body(typeof(body))
    _source_model_capture(body)
    for column in 1:columns
        body(column)
    end
    return nothing
end

struct _ModalSourceModelPlan{P, A}
    plan::P
    scratch::A
end
SP._retained_storage(plan::_ModalSourceModelPlan) = (plan.scratch,)
function LinearAlgebra.mul!(output, plan::_ModalSourceModelPlan, input)
    return LinearAlgebra.mul!(output, plan.plan, input)
end

function SP.TemporalFFT(
        execution::_ModalSourceModelExecution, time::Matrix{T},
        spectrum::Matrix{Complex{R}},
    ) where {T, R <: Union{Float32, Float64}}
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    plan = SP.TemporalFFT(cpu, time, spectrum)
    forward = _ModalSourceModelPlan(plan.forward, zeros(Complex{R}, size(spectrum)))
    inverse = _ModalSourceModelPlan(plan.inverse, zeros(Complex{R}, size(spectrum)))
    return SP.TemporalFFT(
        forward, inverse, plan.inverse_scale, plan.n_time, plan.n_frequency, plan.n_columns,
    )
end

_modal_source_model() = SP.KernelExecution(_ModalSourceDeviceModel(0, 0, 0), 1, true)

@testset "Modal full-map chunks preserve order and clear inactive columns" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 2, true)
    map = reshape(collect(1.0:15.0), 5, 3)
    modal = reshape(ComplexF64.(1:24), 4, 6)
    nodal = fill(ComplexF64(NaN), 4, 6)
    for first in 1:4, count in 0:min(3, 6 - first)
        SP.synthesize!(
            nodal, modal, map, execution;
            n_components = 2, first_node = first, active_nodes = count,
        )
        expected = zeros(ComplexF64, size(nodal))
        for node in 1:count, component in 1:2, mode in 1:3
            expected[:, component + 2(node - 1)] .+=
                modal[:, component + 2(mode - 1)] .* map[first + node - 1, mode]
        end
        @test nodal == expected # Exact small integer products and sums.
        projected = similar(modal)
        SP.project!(
            projected, nodal, transpose(map), execution;
            n_components = 2, first_node = first, active_nodes = count,
        )
        target = zeros(ComplexF64, size(projected))
        for mode in 1:3, component in 1:2, node in 1:count
            target[:, component + 2(mode - 1)] .+=
                nodal[:, component + 2(node - 1)] .* map[first + node - 1, mode]
        end
        @test projected == target
    end
    saved = copy(nodal)
    for first in (0, 6, typemax(Int)), count in (1, 2)
        @test_throws DimensionMismatch SP.synthesize!(
            nodal, modal, map, execution;
            n_components = 2, first_node = first, active_nodes = count,
        )
        @test isequal(nodal, saved)
    end
end

@testset "Device source model uses compact status and independent host metadata" begin
    for kind in (:field, :envelope), form in (:ordinary, :analytic),
            prefactor in (:constant, :table, :expansion), chunk in (1, 2, 3)
        execution = _modal_source_model()
        data, state, fixture = device_source_fixture(execution, kind; form, prefactor)
        source = SP.ModalSource(execution, data; node_chunk = chunk)
        output = similar(state)
        @test data.grid.crop_index == fixture.crop
        @test data.map.host_synthesis !== data.map.synthesis
        @test data.map.host_projection !== data.map.projection
        @test all(block -> block.synthesis === data.map.synthesis, source.chunks)
        @test all(block -> block.projection === data.map.projection, source.chunks)
        @test source.failures.device !== source.failures.host
        before = copy(state)
        position = 0.375
        @test source(output, state, position) === nothing
        expected, bound = setprecision(BigFloat, 256) do
            device_source_exact(state, position, fixture), device_source_budget(fixture)
        end
        @test bound > 0
        @test maximum(abs, Complex{BigFloat}.(output) - expected) <= bound
        @test source.ready && source.calls == 1
        @test state == before
        @test all(iszero, output[.!fixture.active, :])
        @test execution.backend.downloads > 0
        @test execution.backend.elements ==
            execution.backend.downloads * max(2, min(chunk, 3))
        source(output, state, position)
        @test (@allocated source(output, state, position)) == 0
        @test all(iszero, source.nodes[:, (last(source.chunks).count + 1):end])
    end
end

@testset "Device modal source refuses before outputs, including hidden FFT buffers" begin
    execution = _modal_source_model()
    data, state, _ = device_source_fixture(execution, :field)
    source = SP.ModalSource(execution, data; node_chunk = 2)
    output = fill(7.0 + 3.0im, size(state))
    saved = copy(output)
    for retained in (source.transform.forward.scratch, source.transform.inverse.scratch)
        GC.@preserve retained begin
            alias = unsafe_wrap(Array, pointer(retained), size(state); own = false)
            retained_before = copy(retained)
            @test_throws ArgumentError source(alias, state, 0.25)
            @test retained == retained_before
            @test_throws ArgumentError source(output, alias, 0.25)
            @test output == saved && retained == retained_before
        end
    end
    bad = copy(state)
    bad[3, 1] = Inf
    bad[2, 2] = NaN
    failure = try
        source(output, bad, 0.25)
    catch error
        error
    end
    @test failure isa SP.NonfiniteSourceError
    @test (failure.stage, failure.row, failure.column) == (:input, 3, 1)
    @test output == saved && !source.ready
    @test source(output, state, 0.25) === nothing
    @test source.ready && all(isfinite, output)

    arithmetic = SP.SourceArithmetic(Float32)
    count = execution.backend.allocations
    @test_throws ArgumentError SP.ModalSourceGrid(
        execution; kind = :envelope, n_time = 4, dt = 1.0, crop_index = 0:3,
        omega = ones(4), active = fill(true, 4), omega_window = ones(4),
        source_window = ones(4), source_window_derivative = zeros(4), arithmetic,
    )
    @test_throws ArgumentError SP.ModalSourceMap(
        execution; synthesis = ones(1, 1), projection = ones(1, 1), arithmetic,
    )
    @test_throws ArgumentError SP.SourcePrefactor(
        execution, ones(ComplexF64, 4, 1); arithmetic
    )
    cpu = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1)
    grid32 = SP.ModalSourceGrid(
        cpu; kind = :envelope, n_time = 4, dt = 1.0, crop_index = 0:3,
        omega = ones(4), active = fill(true, 4), omega_window = ones(4),
        source_window = ones(4), source_window_derivative = zeros(4), arithmetic,
    )
    map32 = SP.ModalSourceMap(
        cpu; synthesis = ones(1, 1), projection = ones(1, 1), arithmetic,
    )
    prefactor32 = SP.SourcePrefactor(cpu, ones(ComplexF64, 4, 1); arithmetic)
    @test_throws ArgumentError SP.ModalSource(
        execution, SP.ModalSourceData(grid32, map32, prefactor32); node_chunk = 1,
    )
    @test execution.backend.allocations == count
end

@testset "Binary64 source model completes one and eight actual RK trials" begin
    for kind in (:field, :envelope), steps in (1, 8)
        execution = _modal_source_model()
        data, state, fixture = device_source_fixture(execution, kind)
        source = SP.ModalSource(execution, data; node_chunk = 2)
        tableau = device_source_tableau()
        work = SP.RungeKuttaWork(execution, state, tableau)
        operator = SP.DiagonalOperator(execution, zeros(ComplexF64, size(state)))
        flow = SP.LinearFlow(execution, operator, state)
        expected, bound = setprecision(BigFloat, 256) do
            device_source_exact_steps(state, fixture, steps)
        end
        device_source_steps!(state, source, flow, work, tableau, execution, steps)
        @test maximum(abs, Complex{BigFloat}.(state) - expected) <= bound
        @test source.calls == 2steps && source.ready
        @test bound > 0
        @test (
            @allocated device_source_steps!(
                state, source, flow, work, tableau, execution, steps,
            )
        ) == 0
    end
end

@testset "Device model temporal source and quiet exact zero use owned launches" begin
    for kind in (:field, :envelope)
        execution = _modal_source_model()
        prepared, state, _ = device_source_fixture(execution, kind; temporal = true)
        data = SP.ModalSourceData(prepared.grid, prepared.map, prepared.prefactor)
        source = SP.ModalSource(execution, data; node_chunk = 2)
        output = similar(state)
        source(output, state, 0.25)
        # Uniform -rate/2 commutes with the normalized temporal round trip. The
        # finite-fixture FFT hypothesis and two FFTs fit inside gamma_4096.
        unit = BigFloat(eps(Float64)) / 2
        gamma = 4096unit / (1 - 4096unit)
        budget = gamma * maximum(abs, state) / 8
        @test 0 < budget < big(1.0e-10)
        @test maximum(abs, Complex{BigFloat}.(output) + Complex{BigFloat}.(state) / 8) <=
            budget
        @test all(iszero, source.quadrature_fine)
        @test all(iszero, source.quadrature_coarse)
        SP.set_temporal_source!(source, false)
        source(output, state, 0.25)
        @test all(iszero, output) && source.ready
        SP.set_temporal_source!(source, true)
        source(output, state, 0.25)
        @test (@allocated source(output, state, 0.25)) == 0
    end
end
