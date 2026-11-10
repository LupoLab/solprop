# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

@testset "Bridge generations and destination recipes preserve owned fields" begin
    session = march_fixture(; replay = true)
    report = SolpropJulia.advance_session(session, 256, 0.1)
    @test report[1] == 0
    @test report[2] == 0.0
    generation = report[4]
    raw = copy(session.field)
    weights = reshape(collect(1.0:8.0), 8, 1)
    for power in (false, true), optics in (false, true)
        destination = similar(raw)
        SolpropJulia.export_session!(
            destination, session, generation, weights, power, optics,
        )
        expected = power ? raw .* sqrt.(weights) : copy(raw)
        if optics
            signs = reshape([isodd(i) ? 1.0 : -1.0 for i in 1:8], 8, 1)
            expected = conj.(expected) .* signs
        end
        # The recipe applies these exact operations to one retained state, not an
        # independently propagated state; export conversion is a structural contract.
        @test destination == expected
        @test session.field == raw
    end
    @test_throws ArgumentError SolpropJulia.export_session!(
        session.field, session, generation, weights, false, false
    )
    untouched = fill(9.0 + 0.0im, 8, 1)
    @test_throws ArgumentError SolpropJulia.export_session!(
        untouched, session, generation + 1, weights, false, false
    )
    @test all(==(9.0 + 0.0im), untouched)
    SolpropJulia.acknowledge_session(session, generation)
    @test_throws ArgumentError SolpropJulia.export_session!(
        untouched, session, generation, weights, false, false
    )
end

@testset "Bridge attempt copies do not consume or corrupt a pending chunk" begin
    session = march_fixture(; replay = true)
    launch = SolpropJulia.advance!(session)
    SolpropJulia.acknowledge!(session, launch.generation)
    report = SolpropJulia.advance!(session; work_limit = 2, deadline = 60.0)
    generation, count = report.generation, session.attempts.count
    destinations = (
        zeros(count), zeros(count), zeros(count), fill(false, count), zeros(count),
    )
    SolpropJulia.session_attempts!(destinations..., session, generation)
    @test destinations[1] == session.attempts.z[1:count]
    @test destinations[2] == session.attempts.dz[1:count]
    @test destinations[3] == session.attempts.norm[1:count]
    @test destinations[4] == session.attempts.accepted[1:count]
    @test destinations[5] == session.attempts.elapsed[1:count]
    @test session.attempts.count == count
    retained = copy(destinations[1])
    @test_throws ArgumentError SolpropJulia.session_attempts!(
        destinations[1], destinations[1], destinations[3:5]..., session, generation
    )
    @test destinations[1] == retained
    @test_throws ArgumentError SolpropJulia.session_attempts!(
        view(session.attempts.z, 1:count), destinations[2:5]..., session, generation
    )
    SolpropJulia.acknowledge!(session, generation)
    @test_throws ArgumentError SolpropJulia.session_attempts!(
        destinations..., session, generation
    )
end

@testset "External failure creates a barrier without numerical work" begin
    session = march_fixture(; replay = true)
    launch = SolpropJulia.advance!(session)
    @test_throws ArgumentError SolpropJulia.fail_session(session)
    SolpropJulia.acknowledge!(session, launch.generation)
    field = copy(session.field)
    report = SolpropJulia.fail_session(session)
    @test report[1] == 7
    @test report[2] == 0.0
    @test session.field == field
    @test session.source.calls == 0
    @test SolpropJulia.session_failure(session) == (
        "ExternalFailure", "the Python service failed between numerical events",
        false, 0.0, 0.0,
    )
    SolpropJulia.acknowledge!(session, report[4])
    @test_throws ArgumentError SolpropJulia.fail_session(session)
end

@testset "Session storage accounts for numerical and delivery arrays" begin
    session = march_fixture()
    sizes = SolpropJulia.session_memory(session)
    @test sizes.active == length(session.norm.active)
    @test sizes.observed == length(session.norm.observed)
    @test sizes.weights == sizeof(session.norm.weights)
    @test sizes.replay_steps == sizes.replay_edges == 0
    @test sizes.crop == sizeof(session.source.padding.crop)
    @test sizes.uncrop == sizeof(session.source.padding.uncrop)
    @test sizes.field == sizeof(session.field)
    @test all(>=(0), sizes)
    preparation = SolpropJulia.EnvelopePreparation(
        session.execution, session.source.data, session.flow.operator,
        collect(0:7), ones(8, 1),
    )
    prepared = SolpropJulia.preparation_memory(preparation)
    @test prepared.operator == sizeof(session.flow.operator.rate)
    @test prepared.carrier == 0
    @test prepared.power_weight == 8sizeof(Float64)
end
