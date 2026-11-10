# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

@testset "Observation destinations preserve generation and retained buffers" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    session = service_case(execution)
    services = session.services
    SolpropJulia._session_observe!(services, session, :launch, -1)
    session.state.pending = true
    session.state.generation = 1
    @test SolpropJulia.session_observation_status(session, 1) == (0, 3, 4)
    positions, steps, values, plane = zeros(0), zeros(Int, 0), zeros(3, 0), zeros(4)
    SolpropJulia.session_observations!(positions, steps, values, plane, session, 1)
    @test plane == services.plane_values
    @test services.plane_valid
    @test session.state.pending
    @test_throws ArgumentError SolpropJulia.session_observations!(
        positions, steps, values, services.plane_values, session, 1,
    )
    @test_throws ArgumentError SolpropJulia.session_observation_status(session, 2)
    @test_throws DimensionMismatch SolpropJulia.session_observations!(
        positions, steps, zeros(2, 0), plane, session, 1,
    )
    services.rows.count = 2
    services.rows.z[1:2] .= (0.1, 0.2)
    services.rows.step[1:2] .= (0, 2)
    services.rows.values[:, 1:2] .= reshape(collect(1.0:6.0), 3, 2)
    positions, steps, values = zeros(2), zeros(Int, 2), zeros(3, 2)
    SolpropJulia.session_observations!(positions, steps, values, plane, session, 1)
    @test positions == [0.1, 0.2]
    @test steps == [0, 2]
    @test values == reshape(collect(1.0:6.0), 3, 2)
    @test services.rows.count == 2
    backing = fill(-1.0, 8)
    @test_throws ArgumentError SolpropJulia.session_observations!(
        view(backing, 2:3), steps, reshape(view(backing, 1:6), 3, 2), plane, session, 1,
    )
    @test all(==(-1.0), backing)
    SolpropJulia._session_acknowledge!(services, session, 1)
    session.state.pending = false
    @test_throws ArgumentError SolpropJulia.session_observations!(
        positions, steps, values, plane, session, 1,
    )
    @test positions == [0.1, 0.2]
    @test steps == [0, 2]
end

@testset "Request preparation accounts additional immutable arrays once" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    session = service_case(execution)
    observations = session.services.work.data
    modal = SolpropJulia.ModalPreparation(
        execution, service_source(execution, :envelope).data,
        SolpropJulia.DiagonalOperator(execution, zeros(ComplexF64, 8, 2)),
        observations.power_weight, 2,
    )
    prepared = SolpropJulia.SessionPreparation(
        modal, SolpropJulia.SessionObservationPolicy(observations),
    )
    ledger = SolpropJulia.preparation_memory(prepared)
    @test sum(values(ledger)) > 0
    @test all(name -> startswith(string(name), "observations."), keys(ledger))
    @test !any(name -> occursin("power_weight", string(name)), keys(ledger))
    first = SolpropJulia._observation_services(prepared)
    second = SolpropJulia._observation_services(prepared)
    @test first.work.data === second.work.data === observations
    @test first.rows.values !== second.rows.values
    @test first.work.spectrum !== second.work.spectrum
end
