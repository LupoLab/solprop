# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia as SJ

function service_precision_problem(execution, kind, arithmetic)
    nf = kind == :field ? 5 : 8
    grid = SJ.ModalSourceGrid(
        execution; kind, n_time = 8, dt = 0.125, crop_index = collect(0:(nf - 1)),
        omega = collect(1.0:nf), active = fill(true, nf), omega_window = ones(nf),
        source_window = ones(8), source_window_derivative = zeros(8),
        temporal_rate = zeros(8), arithmetic,
    )
    map = SJ.ModalSourceMap(
        execution; synthesis = [1.0 0.0; 0.0 1.0],
        projection = [1.0 0.0; 0.0 1.0], arithmetic,
    )
    data = SJ.ModalSourceData(
        grid, map, SJ.SourcePrefactor(execution, zeros(ComplexF64, nf, 2); arithmetic),
    )
    return SJ.ModalPreparation(
        execution, data, SJ.DiagonalOperator(execution, zeros(ComplexF64, nf, 2)),
        SJ.ObservationWeight(execution, ones(nf, 2)), 2,
    )
end

function service_precision_observation(execution, kind, arithmetic)
    nf = kind == :field ? 5 : 8
    return SJ.ObservationPreparation(
        execution, SJ.StatisticsGrid(collect(1.0:nf), collect(0:7) ./ 8, fill(1 / 8, nf)),
        SJ.StatisticsSelection(
            (SJ.StatisticSlot(:energy, 1, 2), SJ.StatisticSlot(:quadrature_error, 3, 1)),
            (1,),
        );
        kind, n_modes = 2,
        power_weight = SJ.ObservationWeight(execution, ones(nf, 2)), arithmetic,
    )
end

function service_precision_session(execution, kind, arithmetic)
    problem = service_precision_problem(execution, kind, arithmetic)
    observation = service_precision_observation(execution, kind, arithmetic)
    policy = SJ.SessionObservationPolicy(observation; split_rate = fill(0.2, 8))
    preparation = SJ.SessionPreparation(problem, policy)
    services = SJ.ObservationServices(execution, policy; node_chunk = 2)
    T = SJ.source_type(arithmetic)
    nf = kind == :field ? 5 : 8
    physical = ComplexF64[row / 8 + column / 16 for row in 1:nf, column in 1:2]
    stored = Complex{T}.(ldexp.(real.(physical), -arithmetic.state_exponent))
    source = SJ.ModalSource(execution, problem.data; node_chunk = 2)
    norm = SJ.MarchNorm(execution, fill(true, nf), fill(true, nf), ones(nf), 0.01, 0.0)
    schedule = SJ.MarchSchedule(
        planes = [1 / 8], breakpoints = [0.0, 1 / 8], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0, replay_steps = fill(1 / 64, 8),
        replay_edges = collect(0:8) ./ 64,
    )
    tableau = SJ.EmbeddedTableau(
        [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
    )
    session = SJ.RungeKuttaSession(
        execution, stored, 0.0, 1 / 64, tableau, session_precision_control(),
        norm, schedule, source, SJ.LinearFlow(execution, problem.operator, stored);
        arithmetic, services,
    )
    return preparation, session, physical
end

@testset "Observation factories compose one precision and physical scale" begin
    for T in (Float64, Float32), kind in (:field, :envelope)
        execution = SJ.KernelExecution(SJ.KernelAbstractions.CPU(), 2, true)
        arithmetic = T === Float64 ? SJ.SourceArithmetic(T) :
            SJ.SourceArithmetic(T; state_exponent = 20, field_exponent = -10)
        preparation, session, physical = service_precision_session(
            execution, kind, arithmetic,
        )
        services = session.services
        @test preparation.observations.observation.arithmetic == arithmetic
        @test eltype(services.work.spectrum) === Complex{T}
        @test eltype(services.split.spectrum) === Complex{T}
        @test services.split.arithmetic == arithmetic
        @test eltype(services.quadrature_work.maxima) === T
        @test eltype(services.quadrature_work.sums) === T
        @test eltype(services.rows.values) === Float64
        @test services.work.data.arithmetic == arithmetic
        session_precision_one!(session)
        @test session_precision_allocations(session) == 0
        while !session.state.finished
            report = SJ.advance!(session; deadline = 60.0)
            @test report.kind != SJ.SESSION_FAILED
            SJ.acknowledge!(session, report.generation)
        end
        @test session.state.accepted == 8 && session.state.rejected == 0
        restored = ComplexF64.(session.field) .* exp2(arithmetic.state_exponent)
        expected = physical .* exp(-0.1 / 8)
        # Eight split steps each contain two eight-point transforms and one
        # exponential correction. Conditional FFT error plus the final state sum
        # fits 1024 units per step against the pre-cancellation spectral 1-norm.
        unit = Float64(eps(T)) / 2
        gamma = 8192unit / (1 - 8192unit)
        bound = gamma * sum(abs, physical)
        @test bound > 0 && maximum(abs, restored - expected) <= bound
        SJ.observe!(services.work, session.field, session.state.z)
        energy = vec(sum(abs2, expected; dims = 1)) ./ 8
        energy_bound = (2maximum(abs, expected) + bound) * bound * length(physical)
        @test energy_bound > 0
        @test maximum(abs, services.work.values[1:2] - energy) <= energy_bound
        @test isnan(services.work.values[3])
        SJ.close_session!(session)

        mismatches = (
            SJ.SourceArithmetic(T === Float32 ? Float64 : Float32),
            SJ.SourceArithmetic(Float32; state_exponent = 21, field_exponent = -10),
            SJ.SourceArithmetic(Float32; state_exponent = 20, field_exponent = -9),
        )
        for other in mismatches
            other == arithmetic && continue
            policy = SJ.SessionObservationPolicy(
                service_precision_observation(execution, kind, other),
            )
            @test_throws "source and observation arithmetic must agree" (
                SJ.SessionPreparation(preparation.problem, policy)
            )
        end
    end
end

@testset "Spatial service factories retain narrow transforms and quadrature work" begin
    execution = SJ.KernelExecution(SJ.KernelAbstractions.CPU(), 2, true)
    arithmetic = SJ.SourceArithmetic(Float32; state_exponent = 16, field_exponent = -4)
    for kind in (:field, :envelope), route in (:radial, :cartesian, :line)
        prepared, physical = spatial_observation_case(
            execution, kind, route; arithmetic, periodic = (1,),
            names = (
                :energy, :peak_intensity, :on_axis_intensity, :on_axis_spectrum,
                :quadrature_error,
            ),
        )
        policy = SJ.SessionObservationPolicy(prepared; split_rate = fill(0.2, 8))
        services = SJ.ObservationServices(execution, policy; node_chunk = 2)
        @test eltype(services.work.spectrum) === ComplexF32
        @test eltype(services.quadrature_work.maxima) === Float32
        split = route == :radial ? services.split : services.split.temporal
        @test split.arithmetic == arithmetic
        @test eltype(split.spectrum) === ComplexF32
        if route != :radial
            @test eltype(services.split.band.band) === ComplexF32
        end
        field = ComplexF32.(ldexp.(real.(physical), -16), ldexp.(imag.(physical), -16))
        SJ.finish_temporal_split!(field, services.split, 1 / 64, execution)
        SJ.observe!(services.work, field, 0.25)
        @test services.work.valid
        @test all(isfinite, services.work.values[1:(end - 1)])
        @test isnan(last(services.work.values))
    end
end

@testset "Transverse service attachment requires the complete arithmetic identity" begin
    arithmetic = SJ.SourceArithmetic(Float32; state_exponent = 12, field_exponent = -6)
    for route in (:radial, :cartesian, :line)
        execution, edge, _ = transverse_observation_case(route; arithmetic)
        prepared = transverse_session_preparation(execution, edge)
        policy = SJ.SessionObservationPolicy(prepared; transverse = edge, trace_chunk = 2)
        services = SJ.ObservationServices(execution, policy; node_chunk = 2)
        @test prepared.arithmetic == arithmetic
        @test services.transverse.data.arithmetic == arithmetic
        @test eltype(services.work.spectrum) === ComplexF32
        _, other, _ = transverse_observation_case(
            route; arithmetic = SJ.SourceArithmetic(Float32; state_exponent = 13),
        )
        @test_throws "transverse and observation arithmetic must agree" (
            SJ.SessionObservationPolicy(prepared; transverse = other)
        )
    end
end
