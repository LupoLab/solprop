# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test
import CUDA
import SolpropJulia as SP

include("modal_source_fixture.jl")

@test CUDA.functional()
CUDA.allowscalar(false)

function cuda_source_host(array, execution)
    output = Array{eltype(array)}(undef, size(array))
    SP.copy_host!(output, array, execution)
    return output
end

@testset "CUDA modal Kerr matches independent represented DFT and cubic sources" begin
    execution = SP.device_execution(Val(:cuda), 0, 2; check_body = true)
    prior_context, prior_stream = CUDA.context(), CUDA.stream()
    for (kind, form) in (
                (:field, :ordinary), (:field, :analytic),
                (:envelope, :ordinary), (:envelope, :harmonic),
            ),
            prefactor in (:constant, :table, :expansion), chunk in (1, 2, 3)
        data, initial, fixture = device_source_fixture(execution, kind; form, prefactor)
        source = SP.ModalSource(execution, data; node_chunk = chunk)
        state = SP.upload(execution, initial)
        output = SP.dzeros(execution, ComplexF64, size(initial)...)
        for position in (0.0, 0.375, 1.0)
            @test source(output, state, position) === nothing
            actual = cuda_source_host(output, execution)
            exact, budget = setprecision(BigFloat, 256) do
                device_source_exact(initial, position, fixture),
                    device_source_budget(fixture)
            end
            @test 0 < budget < big(1.0e-10)
            @test maximum(abs, Complex{BigFloat}.(actual) - exact) <= budget
            @test all(iszero, actual[.!fixture.active, :])
            @test cuda_source_host(state, execution) == initial
            @test source.ready
            # Both material projections are produced by this same successful call.
            fine = cuda_source_host(source.quadrature_fine, execution)
            coarse = cuda_source_host(source.quadrature_coarse, execution)
            @test fine == actual
            @test maximum(abs, Complex{BigFloat}.(coarse) - big(0.875) * exact) <= budget
        end
        @test source.calls == 3
        source(output, state, 0.375)
        @test (CUDA.@allocated source(output, state, 0.375)) == 0
        @test CUDA.context() == prior_context && CUDA.stream() == prior_stream
        @test data.grid.crop_index isa Vector{Int}
        @test data.grid.omega isa Vector{Float64}
        @test data.map.host_synthesis isa Matrix{Float64}
        @test data.map.host_synthesis !== data.map.synthesis
        @test all(chunk -> chunk.synthesis === data.map.synthesis, source.chunks)
        @test all(chunk -> chunk.projection === data.map.projection, source.chunks)
        @test source.failures.device isa CUDA.CuArray{Int, 1, CUDA.DeviceMemory}
        @test size(source.failures.host) == (max(chunk, 2),)
        @test all(
            iszero, cuda_source_host(source.nodes, execution)[
                :, (last(source.chunks).count + 1):end,
            ]
        )
        ledger = Dict{Symbol, Int}()
        SP._memory_arrays!(ledger, IdDict{Any, Bool}(), "source", source)
        @test ledger[Symbol("source.failures.device")] == sizeof(source.failures.device)
        @test ledger[Symbol("source.failures.host")] == sizeof(source.failures.host)
        @test ledger[Symbol("source.data.map.host_synthesis")] ==
            sizeof(data.map.host_synthesis)
        @test ledger[Symbol("source.data.map.synthesis")] == sizeof(data.map.synthesis)
        @test haskey(ledger, Symbol("source.transform.forward.workspace"))
        @test haskey(ledger, Symbol("source.transform.inverse.workspace"))
    end
end

function cuda_source_drive!(session)
    while !session.state.finished
        report = SP.advance!(session; work_limit = 1, deadline = 60.0)
        SP.acknowledge!(session, report.generation)
    end
    return nothing
end

@testset "CUDA source-bearing sessions deliver one and eight fixed steps" begin
    execution = SP.device_execution(Val(:cuda), 0, 2; check_body = true)
    for kind in (:field, :envelope), steps in (1, 8)
        data, initial, fixture = device_source_fixture(execution, kind)
        source = SP.ModalSource(execution, data; node_chunk = 2)
        prototype = SP.upload(execution, initial)
        operator = SP.DiagonalOperator(execution, zeros(ComplexF64, size(initial)))
        flow = SP.LinearFlow(execution, operator, prototype)
        step = 1 / 128
        finish = step * steps
        norm = SP.MarchNorm(
            execution, fixture.active, fixture.active, ones(size(initial, 1)),
            1.0e-5, 1.0e-10,
        )
        schedule = SP.MarchSchedule(
            planes = [finish], breakpoints = [0.0, 1.0], hard_kinks = Float64[],
            stretch = 1.1, step_planes = 0, replay_steps = fill(step, steps),
            replay_edges = collect(0:step:finish),
        )
        control = SP.PIControl(
            safety = 0.8, max_factor = 2.0, min_factor = 0.2, beta1 = 0.3,
            beta2 = -0.1, lag = 0, reject_free = false, estimator_order = 1,
            start_growth = 100.0, dz_min = 1.0e-12, dz_max = step / 4, auto_init = false,
        )
        session = SP.RungeKuttaSession(
            execution, initial, 0.0, step, device_source_tableau(), control, norm,
            schedule, source, flow; capacity = 16,
        )
        exact, budget = setprecision(BigFloat, 256) do
            device_source_exact_steps(initial, fixture, steps)
        end
        @test (CUDA.@allocated cuda_source_drive!(session)) == 0
        actual = cuda_source_host(session.field, execution)
        @test maximum(abs, Complex{BigFloat}.(actual) - exact) <= budget
        @test session.state.attempts == session.state.accepted == steps
        @test session.state.rejected == 0
        @test source.calls == 2steps
        @test session.state.z == finish
        @test all(iszero, actual[.!fixture.active, :])
        SP.close_session!(session)
    end
end

@testset "CUDA modal source failures preserve caller storage and permit a clean retry" begin
    execution = SP.device_execution(Val(:cuda), 0, 1; check_body = true)
    data, initial, _ = device_source_fixture(execution, :field)
    source = SP.ModalSource(execution, data; node_chunk = 2)
    state = SP.upload(execution, initial)
    sentinel = fill(7.0 + 3.0im, size(initial))
    output = SP.upload(execution, sentinel)
    for raw in (source.transform.forward, source.transform.inverse)
        for retained in SP._retained_storage(raw)
            CUDA.context!(execution.backend.context) do
                GC.@preserve retained begin
                    alias = unsafe_wrap(
                        CUDA.CuArray, pointer(retained), size(initial); own = false,
                    )
                    before = cuda_source_host(retained, execution)
                    @test_throws ArgumentError source(alias, state, 0.25)
                    @test isequal(cuda_source_host(retained, execution), before)
                    @test_throws ArgumentError source(output, alias, 0.25)
                    @test cuda_source_host(output, execution) == sentinel
                    @test isequal(cuda_source_host(retained, execution), before)
                end
            end
        end
    end
    @test_throws ArgumentError source(data.prefactor.data, state, 0.25)
    @test_throws ArgumentError source(output, source.prefactor, 0.25)
    @test cuda_source_host(output, execution) == sentinel
    bad = copy(initial)
    bad[3, 1] = Inf
    bad[2, 2] = NaN
    SP.copy_upload!(state, bad, execution)
    failure = try
        source(output, state, 0.25)
    catch error
        error
    end
    @test failure isa SP.NonfiniteSourceError
    @test (failure.stage, failure.row, failure.column) == (:input, 3, 1)
    @test cuda_source_host(output, execution) == sentinel
    @test !source.ready
    SP.copy_upload!(state, initial, execution)
    @test source(output, state, 0.25) === nothing
    @test source.ready && all(isfinite, cuda_source_host(output, execution))

    # Construction fails before any device allocation for an unsupported precision.
    arithmetic = SP.SourceArithmetic(Float32)
    function refused()
        try
            SP.ModalSourceGrid(
                execution; kind = :envelope, n_time = 4, dt = 1.0, crop_index = 0:3,
                omega = ones(4), active = fill(true, 4), omega_window = ones(4),
                source_window = ones(4), source_window_derivative = zeros(4), arithmetic,
            )
        catch error
            @test error isa ArgumentError
            return nothing
        end
        error("device binary32 source was unexpectedly admitted")
    end
    refused()
    @test (CUDA.@allocated refused()) == 0
end

@testset "CUDA one and eight source-bearing RK steps preserve the discrete method" begin
    execution = SP.device_execution(Val(:cuda), 0, 2; check_body = true)
    for kind in (:field, :envelope), steps in (1, 8)
        data, initial, fixture = device_source_fixture(execution, kind)
        source = SP.ModalSource(execution, data; node_chunk = 2)
        state = SP.upload(execution, initial)
        tableau = device_source_tableau()
        work = SP.RungeKuttaWork(execution, state, tableau)
        operator = SP.DiagonalOperator(execution, zeros(ComplexF64, size(initial)))
        flow = SP.LinearFlow(execution, operator, state)
        exact, budget = setprecision(BigFloat, 256) do
            device_source_exact_steps(initial, fixture, steps)
        end
        device_source_steps!(state, source, flow, work, tableau, execution, steps)
        actual = cuda_source_host(state, execution)
        @test maximum(abs, Complex{BigFloat}.(actual) - exact) <= budget
        @test 0 < budget < big(1.0e-10)
        @test source.calls == 2steps && source.ready
        @test all(iszero, actual[.!fixture.active, :])
        @test (
            CUDA.@allocated device_source_steps!(
                state, source, flow, work, tableau, execution, steps,
            )
        ) == 0
    end
end

@testset "CUDA in-stage absorption and quiet zero retain the scalar source contract" begin
    execution = SP.device_execution(Val(:cuda), 0, 2; check_body = true)
    for kind in (:field, :envelope)
        prepared, initial, _ = device_source_fixture(execution, kind; temporal = true)
        data = SP.ModalSourceData(prepared.grid, prepared.map, prepared.prefactor)
        source = SP.ModalSource(execution, data; node_chunk = 2)
        state = SP.upload(execution, initial)
        output = SP.dzeros(execution, ComplexF64, size(initial)...)
        source(output, state, 0.25)
        actual = cuda_source_host(output, execution)
        # The uniform rate commutes with the temporal round trip; gamma_4096
        # bounds these two n=16 transforms under the stated twiddle hypothesis.
        unit = BigFloat(eps(Float64)) / 2
        gamma = 4096unit / (1 - 4096unit)
        budget = gamma * maximum(abs, initial) / 8
        @test 0 < budget < big(1.0e-10)
        @test maximum(abs, Complex{BigFloat}.(actual) + Complex{BigFloat}.(initial) / 8) <=
            budget
        @test all(iszero, cuda_source_host(source.quadrature_fine, execution))
        @test all(iszero, cuda_source_host(source.quadrature_coarse, execution))
        SP.set_temporal_source!(source, false)
        source(output, state, 0.25)
        @test all(iszero, cuda_source_host(output, execution)) && source.ready
        SP.set_temporal_source!(source, true)
        source(output, state, 0.25)
        @test (CUDA.@allocated source(output, state, 0.25)) == 0
    end
end
