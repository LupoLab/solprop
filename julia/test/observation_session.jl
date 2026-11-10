# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

mutable struct ObservationScriptState
    z::Float64
    accepted::Int
    replay::Int
    pending::Bool
    generation::Int
    closed::Bool
end

struct ObservationScript{E, S, F}
    execution::E
    services::S
    source::F
    field::Matrix{ComplexF64}
    attempts::SolpropJulia.AttemptBuffer
    state::ObservationScriptState
    schedule::SolpropJulia.MarchSchedule
end

function service_spectrum(samples, kind)
    n_time, n_modes = size(samples)
    frequencies = kind == :field ? n_time ÷ 2 + 1 : n_time
    result = ComplexF64[
        sum(
            Complex{BigFloat}(samples[row + 1, column]) *
                cis(-2big(pi) * frequency * row / n_time) for row in 0:(n_time - 1)
        ) for frequency in 0:(frequencies - 1), column in 1:n_modes
    ]
    if kind == :field
        result[1, :] .= real.(result[1, :])
        iseven(n_time) && (result[end, :] .= real.(result[end, :]))
    end
    return result
end

function service_samples(kind, row = 4)
    result = zeros(kind == :field ? Float64 : ComplexF64, 8, 2)
    result[row, :] .= (1.0, 0.7)
    return result
end

function service_source(execution, kind)
    count = kind == :field ? 5 : 8
    grid = SolpropJulia.ModalSourceGrid(
        execution; kind, n_time = 8, dt = 0.125, crop_index = collect(0:(count - 1)),
        omega = collect(1.0:count), active = fill(true, count),
        omega_window = ones(count), source_window = ones(8),
        source_window_derivative = zeros(8), temporal_rate = fill(0.2, 8),
    )
    maps = SolpropJulia.ModalSourceMap(
        execution; synthesis = [1.0 0.0; 0.0 1.0], projection = [1.0 0.0; 0.0 1.0],
    )
    data = SolpropJulia.ModalSourceData(
        grid, maps, SolpropJulia.SourcePrefactor(execution, zeros(ComplexF64, count, 2)),
    )
    return SolpropJulia.ModalSource(execution, data; node_chunk = 2)
end

function service_case(
        execution, kind = :envelope; monitor = false, split = false, can_rest = false,
        period = 1, capacity = 4, warn = 0.2, stop = 1.1, cadence = 1,
        interval = 0.3, replay = Float64[], continuation = nothing, noise = nothing,
        crossing = false, periodic = (3, 1), crossing_limits = [0.05],
        crossing_map = nothing,
    )
    count = kind == :field ? 5 : 8
    grid = SolpropJulia.StatisticsGrid(
        collect(1.0:count), collect((0:7) .* 0.125), fill(1 / 8, count),
    )
    selection = SolpropJulia.StatisticsSelection(
        (
            SolpropJulia.StatisticSlot(:energy, 1, 2),
            SolpropJulia.StatisticSlot(:peak_power_total, 3, 1),
            SolpropJulia.StatisticSlot(:pressure, 4, 1),
        ), periodic,
    )
    rate = collect(range(0.1, 0.8; length = 8))
    temporal = if monitor
        regions = zeros(8, 4)
        regions[1, 1] = regions[8, 2] = 1.0
        regions[1, 3] = regions[8, 3] = 1.0
        regions[:, 4] .= rate
        policy = SolpropJulia.TemporalMonitorPolicy(;
            warn, stop, activity = 0.2, quiet_share = 0.1, quiet_checks = 1,
            quiet_collar = true, cadence, interval,
            launch_floor_ratio = 1.0e-6, floor_margin = 1.2, split,
            crossing_share = crossing ? 0.1 : 0.0,
            bucket_limits = crossing ? crossing_limits : Float64[],
        )
        buckets = crossing ?
            (crossing_map === nothing ? ones(Int, count, 2) : crossing_map) : nothing
        SolpropJulia.TemporalObservationData(
            policy, SolpropJulia.TemporalMonitorWeights(regions, [1.0, 0.8]; buckets),
        )
    else
        nothing
    end
    preparation = SolpropJulia.ObservationPreparation(
        execution, grid, selection; kind, n_modes = 2,
        power_weight = SolpropJulia.ObservationWeight(execution, ones(count, 2)),
        profiles = SolpropJulia.ObservationProfiles(
            pressure = SolpropJulia.SourceProfile(1.2),
        ),
        temporal,
    )
    policy = SolpropJulia.SessionObservationPolicy(
        preparation; period, can_rest, split_rate = split ? rate : nothing,
    )
    services = SolpropJulia.ObservationServices(
        execution, policy; attempt_capacity = capacity, node_chunk = 2,
        noise_spectrum = noise, temporal_continuation = continuation,
        continuation_plane = true,
    )
    schedule = SolpropJulia.MarchSchedule(
        planes = [1.0], breakpoints = [0.0, 1.0], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0, replay_steps = replay,
        replay_edges = isempty(replay) ? Float64[] : vcat(0.0, cumsum(replay)),
    )
    return ObservationScript(
        execution, services, can_rest ? service_source(execution, kind) : nothing,
        service_spectrum(service_samples(kind), kind), SolpropJulia.AttemptBuffer(capacity),
        ObservationScriptState(0.0, 0, 1, false, 0, false), schedule,
    )
end

function service_ack!(session)
    session.state.pending = true
    session.state.generation += 1
    SolpropJulia._session_acknowledge!(session.services, session, session.state.generation)
    session.attempts.count = 0
    session.state.pending = false
    return nothing
end

function service_launch!(session)
    SolpropJulia._session_observe!(session.services, session, :launch, -1)
    service_ack!(session)
    SolpropJulia._session_launch!(session.services, session)
    return nothing
end

function service_record!(session, dz; accepted = true, destination = session.state.z + dz)
    index = session.attempts.count + 1
    session.attempts.z[index], session.attempts.dz[index] = destination, dz
    session.attempts.accepted[index] = accepted
    session.attempts.count = index
    if accepted
        session.state.z = destination
        session.state.accepted += 1
        !isempty(session.schedule.replay_steps) && (session.state.replay += 1)
    end
    return nothing
end

function service_accept!(session, dz; destination = session.state.z + dz)
    SolpropJulia._session_finish_candidate!(
        session.services, session, session.field, dz, destination,
    )
    service_record!(session, dz; destination)
    SolpropJulia._session_accepted!(session.services, session, dz)
    return nothing
end

function service_allocation!(session)
    service_ack!(session)
    service_accept!(session, 0.01)
    service_ack!(session)
    return @allocated service_accept!(session, 0.01)
end

function service_plane_allocation!(session)
    service_ack!(session)
    service_accept!(session, 0.01)
    SolpropJulia._session_observe!(session.services, session, :plane, 0)
    service_ack!(session)
    service_accept!(session, 0.01)
    return @allocated SolpropJulia._session_observe!(session.services, session, :plane, 0)
end

@testset "Owned temporal split and independent Fourier reference" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope)
        samples = service_samples(kind)
        samples[2, 1] = 0.3
        samples[7, 2] = -0.2
        spectrum = service_spectrum(samples, kind)
        rate = collect(range(0.0, 1.0; length = 8))
        work = SolpropJulia.TemporalSplitWorkspace(execution, kind, rate, 2)
        rate .= 99.0
        @test work.rate[1] == 0.0 && work.rate[end] == 1.0
        expected = service_spectrum(samples .* exp.(-0.15 .* work.rate), kind)
        candidate = copy(spectrum)
        SolpropJulia.finish_temporal_split!(candidate, work, 0.3, execution)
        # Two length-eight FFTs, one exponential and product versus a BigFloat DFT.
        @test isapprox(candidate, expected; rtol = 128eps(Float64), atol = 128eps(Float64))
        fill!(work.samples, NaN)
        fill!(work.spectrum, NaN)
        copyto!(candidate, spectrum)
        SolpropJulia.finish_temporal_split!(candidate, work, 0.3, execution)
        @test isapprox(candidate, expected; rtol = 128eps(Float64), atol = 128eps(Float64))
        zero_work = SolpropJulia.TemporalSplitWorkspace(execution, kind, zeros(8), 2)
        saved = copy(candidate)
        SolpropJulia.finish_temporal_split!(candidate, zero_work, 0.2, execution)
        @test candidate == saved
        @test_throws ArgumentError SolpropJulia.finish_temporal_split!(
            work.spectrum, work, 0.2, execution,
        )
        @test_throws ArgumentError SolpropJulia.finish_temporal_split!(
            candidate, work, 0.0, execution,
        )
        @test_throws DimensionMismatch SolpropJulia.finish_temporal_split!(
            zeros(ComplexF64, 1, 2), work, 0.2, execution,
        )
        @test candidate == saved
        if kind == :field
            candidate[1, 1] += im
            bad = copy(candidate)
            @test_throws ArgumentError SolpropJulia.finish_temporal_split!(
                candidate, work, 0.2, execution,
            )
            @test candidate == bad
        end
        copyto!(candidate, spectrum)
        SolpropJulia.finish_temporal_split!(candidate, work, 0.2, execution)
        @test (
            @allocated SolpropJulia.finish_temporal_split!(
                candidate, work, 0.2, execution,
            )
        ) == 0
        candidate[2, 1] = NaN
        @test_throws ArgumentError SolpropJulia.finish_temporal_split!(
            candidate, work, 0.2, execution,
        )
    end
    @test_throws ArgumentError SolpropJulia.TemporalSplitWorkspace(
        execution, :field, [-1.0], 2,
    )
    @test_throws ArgumentError SolpropJulia.TemporalSplitWorkspace(
        execution, :vector, ones(8), 2,
    )
end

@testset "Bounded periodic rows, plane values and acknowledgment" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope)
        session = service_case(execution, kind; capacity = 3, period = 2)
        services = session.services
        @test size(services.rows.values) == (3, 4)
        SolpropJulia._session_observe!(services, session, :launch, -1)
        launch = copy(services.plane_values)
        @test SolpropJulia.observation_planes(services).valid && !services.started
        service_ack!(session)
        @test !services.plane_valid && !services.started
        SolpropJulia._session_launch!(services, session)
        service_record!(session, 0.1; accepted = false)
        service_accept!(session, 0.1)
        @test services.rows.count == 0
        service_accept!(session, 0.1)
        @test services.rows.count == 1 && services.rows.step[1] == 2
        @test services.rows.z[1] == session.state.z
        @test services.rows.values[1, 1] == 1.2
        @test services.rows.values[2:3, 1] == launch[1:2]
        SolpropJulia._session_observe!(services, session, :plane, 0)
        @test services.plane_values == launch
        rows = SolpropJulia.observation_rows(services)
        planes = SolpropJulia.observation_planes(services)
        @test rows.values === services.rows.values
        @test planes.values === services.plane_values
        @test SolpropJulia.observation_rows(services).count == 1
        session.state.pending = true
        session.state.generation = 5
        @test_throws ArgumentError SolpropJulia._session_acknowledge!(services, session, 4)
        @test services.rows.count == 1 && services.plane_valid
        SolpropJulia._session_acknowledge!(services, session, 5)
        session.state.pending = false
        session.attempts.count = 0
        @test services.rows.count == 0 && !services.plane_valid
        @test services.accepted_seen == 2 && services.committed_z == 0.2
        @test_throws ArgumentError SolpropJulia._session_acknowledge!(services, session, 5)
        service_accept!(session, 0.1)
        @test services.rows.count == 0
        service_accept!(session, 0.1)
        @test services.rows.count == 1 && services.rows.step[1] == 1
        @test service_allocation!(session) == 0
        @test service_plane_allocation!(session) == 0
        SolpropJulia._session_finish!(services, session)
        @test services.finished
        @test_throws ArgumentError SolpropJulia._session_finish!(services, session)
        @test SolpropJulia._session_continue!(services, session) == :none
    end
end

@testset "Warning continuation precedes periodic and plane statistics" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    session = service_case(execution; monitor = true, capacity = 1)
    services = session.services
    service_launch!(session)
    @test services.monitor_count == 1
    @test SolpropJulia._session_barrier(services, session) == :none
    session.field .= service_spectrum(service_samples(:envelope, 1), :envelope)
    service_accept!(session, 0.1)
    @test SolpropJulia._session_barrier(services, session) == :warning
    @test services.rows.count == 0 && services.accepted_seen == 1
    @test services.notices.records[1].code == SolpropJulia.TemporalGuardWarning
    @test_throws ArgumentError SolpropJulia._session_continue!(services, session)
    @test_throws ArgumentError SolpropJulia._session_observe!(services, session, :plane, 0)
    @test services.rows.count == 0
    service_ack!(session)
    @test SolpropJulia._session_continue!(services, session) == :accepted_rest
    @test services.rows.count == 1 && services.rows.step[1] == -1
    @test SolpropJulia._session_continue!(services, session) == :none
    service_accept!(session, 0.1)
    @test services.rows.count == 2 && services.rows.step == [-1, 0]
    @test services.work.monitor.steps == 2
    SolpropJulia._session_observe!(services, session, :plane, 0)
    @test services.plane_valid && services.monitor_count == 1
    @test services.rows.count == 2
    # Rebasing after the warning's one attempt yields the two actual accepted rows.
    @test services.rows.step .+ 1 == [0, 1]
    pending = SolpropJulia.observation_status(services)
    @test pending.row_count == 2 && pending.plane_valid
    @test SolpropJulia.observation_notices(services).count == 0
    service_ack!(session)
    @test services.monitor_count == 0 && services.work.monitor.warned

    stale = service_case(execution; monitor = true, cadence = 4, interval = Inf)
    service_launch!(stale)
    stale.field .= service_spectrum(service_samples(:envelope, 1), :envelope)
    service_accept!(stale, 0.1)
    @test stale.services.work.monitor.stale && stale.services.rows.count == 1
    SolpropJulia._session_observe!(stale.services, stale, :plane, 0)
    @test SolpropJulia._session_barrier(stale.services, stale) == :warning
    @test !stale.services.plane_valid && stale.services.work.monitor.skipped == 1
    service_ack!(stale)
    @test SolpropJulia._session_continue!(stale.services, stale) == :plane_rest
    @test stale.services.plane_valid && stale.services.rows.count == 0
    @test stale.services.work.monitor.steps == 0 && stale.services.work.monitor.skipped == 1
end

@testset "Launch and terminal barriers retain eligible committed fields" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    session = service_case(execution; monitor = true)
    session.field .= service_spectrum(service_samples(:envelope, 1), :envelope)
    service_launch!(session)
    @test session.services.started && session.services.monitor_count == 1
    @test SolpropJulia._session_barrier(session.services, session) == :warning
    @test isnan(session.services.notices.records[1].z)
    service_ack!(session)
    @test SolpropJulia._session_continue!(session.services, session) == :launch_rest
    @test session.state.accepted == 0 && session.services.rows.count == 0

    stopped = service_case(execution; monitor = true, stop = 0.5)
    service_launch!(stopped)
    stopped.field .= service_spectrum(service_samples(:envelope, 1), :envelope)
    saved = copy(stopped.field)
    service_accept!(stopped, 0.1)
    @test SolpropJulia._session_barrier(stopped.services, stopped) == :boundary_stop
    @test stopped.field == saved && stopped.state.z == 0.1
    @test stopped.services.rows.count == 0 && stopped.services.notices.count == 1

    final = service_case(execution; monitor = true, warn = 0.1, stop = 1.1)
    service_launch!(final)
    service_accept!(final, 0.1)
    SolpropJulia._session_observe!(final.services, final, :plane, 0)
    service_ack!(final)
    SolpropJulia._session_finish!(final.services, final)
    @test final.services.finished
    @test SolpropJulia._session_barrier(final.services, final) == :warning
    @test final.services.notices.records[1].code == SolpropJulia.TemporalRemovalWarning
    service_ack!(final)
    @test SolpropJulia._session_continue!(final.services, final) == :finish_rest
    @test final.services.notices.count == 0 && final.services.finished

    twice = service_case(
        execution; monitor = true, cadence = 4, interval = Inf, warn = 0.1,
    )
    service_launch!(twice)
    twice.field .= service_spectrum(service_samples(:envelope, 1), :envelope)
    service_accept!(twice, 0.1)
    SolpropJulia._session_finish!(twice.services, twice)
    @test SolpropJulia._session_barrier(twice.services, twice) == :warning
    @test !twice.services.finished
    service_ack!(twice)
    @test SolpropJulia._session_continue!(twice.services, twice) == :none
    @test twice.services.finished &&
        twice.services.notices.records[1].code == SolpropJulia.TemporalRemovalWarning
    @test twice.services.continuation == SolpropJulia.ObservationFinishRest
    service_ack!(twice)
    @test SolpropJulia._session_continue!(twice.services, twice) == :finish_rest
    @test twice.services.notices.count == 0
end

@testset "Split capture endpoint, crossing refusal and source omission" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope)
        session = service_case(execution, kind; monitor = true, split = true)
        service_launch!(session)
        endpoint = nextfloat(0.1)
        service_accept!(session, 0.1; destination = endpoint)
        @test session.services.work.monitor.committed == endpoint
        @test !session.services.work.monitor_work.captured
        @test session.services.work.monitor.raw_removed > 0
        @test service_allocation!(session) == 0
        @test service_plane_allocation!(session) == 0
    end
    refused = service_case(
        execution; monitor = true, split = true, crossing = true,
        replay = [0.01, 0.1, 0.01], cadence = 3,
    )
    service_launch!(refused)
    @test SolpropJulia._session_barrier(refused.services, refused) == :boundary_stop
    refusal = SolpropJulia.observation_notices(refused.services).crossing_refusal
    @test refusal isa SolpropJulia.TemporalCrossingRefusal
    @test refusal.z == 0 && refusal.cap == 0.05 && refusal.step == 0.1
    @test refused.state.replay == 1 && refused.state.accepted == 0

    mapping = ones(Int, 8, 2)
    mapping[1, :] .= 2
    falling = service_case(
        execution; monitor = true, split = true, crossing = true,
        replay = [0.01, 0.1, 0.01], warn = 0.8, crossing_limits = [0.05, 0.2],
        crossing_map = mapping,
    )
    fill!(falling.field, 0)
    falling.field[1, :] .= 8
    service_launch!(falling)
    @test SolpropJulia._session_barrier(falling.services, falling) == :none
    @test SolpropJulia._session_cap(falling.services, falling, 0.3) == 0.2
    fill!(falling.field, 0)
    falling.field[5, :] .= 8
    service_accept!(falling, 0.01)
    @test SolpropJulia._session_barrier(falling.services, falling) == :boundary_stop
    @test falling.services.crossing_refusal ==
        SolpropJulia.TemporalCrossingRefusal(0.01, 0.05, 0.1)
    @test falling.state.accepted == 1 && falling.state.replay == 2
    @test falling.services.rows.count == 0

    chained = service_case(
        execution; monitor = true, split = true, crossing = true, replay = [0.1],
    )
    chained.field .= service_spectrum(service_samples(:envelope, 1), :envelope)
    service_launch!(chained)
    @test SolpropJulia._session_barrier(chained.services, chained) == :warning
    service_ack!(chained)
    @test SolpropJulia._session_continue!(chained.services, chained) == :none
    @test SolpropJulia._session_barrier(chained.services, chained) == :boundary_stop
    @test chained.services.continuation == SolpropJulia.ObservationLaunchRest

    quiet = service_case(execution; monitor = true, can_rest = true)
    service_launch!(quiet)
    @test SolpropJulia._session_quiet(quiet.services, quiet)
    @test SolpropJulia._session_quiet_interval(quiet.services, quiet) == 0.3
    @test !quiet.source.temporal_active && quiet.source.calls == 0
    quiet.field .= service_spectrum(service_samples(:envelope, 1), :envelope)
    service_accept!(quiet, 0.1)
    @test SolpropJulia._session_barrier(quiet.services, quiet) == :warning
    service_ack!(quiet)
    @test SolpropJulia._session_continue!(quiet.services, quiet) == :accepted_rest
    @test !SolpropJulia._session_quiet(quiet.services, quiet)
    @test SolpropJulia._session_quiet_interval(quiet.services, quiet) === nothing
    @test quiet.source.temporal_active && quiet.source.calls == 0
    @test !quiet.services.work.monitor.left_out

    saved = SolpropJulia.temporal_continuation(quiet.services.work.monitor)
    resumed = service_case(
        execution; monitor = true, can_rest = true, continuation = saved, period = 2,
    )
    service_launch!(resumed)
    @test resumed.services.work.monitor.steps == quiet.services.work.monitor.steps
    @test resumed.services.rows.count == 0 && resumed.services.accepted_seen == 0
    @test SolpropJulia._session_barrier(resumed.services, resumed) == :none
    service_accept!(resumed, 0.1)
    @test resumed.services.rows.count == 0 && resumed.services.accepted_seen == 1
end

@testset "Service admission and repeated hook guards" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    session = service_case(execution)
    prepared = session.services.work.data
    @test_throws ArgumentError SolpropJulia.SessionObservationPolicy(prepared; period = 0)
    @test_throws ArgumentError SolpropJulia.SessionObservationPolicy(
        prepared; can_rest = true,
    )
    @test_throws DimensionMismatch SolpropJulia.SessionObservationPolicy(
        prepared; split_rate = ones(7),
    )
    @test_throws ArgumentError SolpropJulia.SessionObservationPolicy(
        prepared; split_rate = fill(NaN, 8),
    )
    @test_throws ArgumentError SolpropJulia.ObservationServices(
        execution, SolpropJulia.SessionObservationPolicy(prepared);
        attempt_capacity = 0, node_chunk = 1,
    )
    monitored = service_case(execution; monitor = true, split = true)
    @test_throws ArgumentError SolpropJulia.SessionObservationPolicy(
        monitored.services.work.data; split_rate = fill(0.7, 8),
    )
    @test_throws ArgumentError SolpropJulia.SessionObservationPolicy(
        monitored.services.work.data,
    )
    @test_throws ArgumentError SolpropJulia._session_accepted!(
        session.services, session, 0.1,
    )
    service_launch!(session)
    @test_throws ArgumentError SolpropJulia._session_launch!(session.services, session)
    service_accept!(session, 0.1)
    @test_throws ArgumentError SolpropJulia._session_accepted!(
        session.services, session, 0.1,
    )
    @test_throws ArgumentError SolpropJulia._session_observe!(
        session.services, session, :dense, 0,
    )
    @test SolpropJulia._session_cap(session.services, session, 0.3) == 0.3
    empty = service_case(execution; periodic = ())
    service_launch!(empty)
    service_accept!(empty, 0.1)
    @test isempty(empty.services.rows.z) && size(empty.services.rows.values) == (0, 0)
    other = service_case(execution)
    @test_throws ArgumentError SolpropJulia._session_launch!(other.services, session)
end

mutable struct ObservationZeroSource
    calls::Int
end

function (source::ObservationZeroSource)(output, field, z)
    source.calls += 1
    fill!(output, 0)
    return nothing
end

function service_march_case(; capacity = 2, warn = 0.2)
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    fixture = service_case(execution; monitor = true, split = true, capacity, warn)
    source = ObservationZeroSource(0)
    frequencies = [0, 1, 2, 3, -4, -3, -2, -1]
    rate = repeat(reshape(-2pi * im .* frequencies ./ (8 * 0.025), 8, 1), 1, 2)
    flow = SolpropJulia.LinearFlow(
        execution, SolpropJulia.DiagonalOperator(execution, rate), fixture.field,
    )
    tableau = SolpropJulia.EmbeddedTableau(
        [0.0, 1.0], [0.0 0.0; 1.0 0.0], [0.5, 0.5], [-0.5, 0.5],
    )
    control = SolpropJulia.PIControl(;
        safety = 0.8, max_factor = 2.0, min_factor = 0.2, beta1 = 0.3, beta2 = -0.1,
        lag = 0, reject_free = false, estimator_order = 1, start_growth = 100.0,
        dz_min = 1.0e-12, dz_max = 0.05, auto_init = false,
    )
    norm = SolpropJulia.MarchNorm(
        execution, fill(true, 8), fill(true, 8), ones(8), 1.0e-4, 0.0,
    )
    schedule = SolpropJulia.MarchSchedule(
        planes = [0.1], breakpoints = [0.0, 0.1], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0, replay_steps = fill(0.025, 4),
        replay_edges = [0.0, 0.025, 0.05, 0.075, 0.1],
    )
    return SolpropJulia.RungeKuttaSession(
        execution, fixture.field, 0.0, 0.025, tableau, control, norm, schedule,
        source, flow; services = fixture.services, capacity,
    )
end

function service_march_allocation(session)
    launch = SolpropJulia.advance!(session; work_limit = 1, deadline = 60.0)
    SolpropJulia.acknowledge!(session, launch.generation)
    first = SolpropJulia.advance!(session; work_limit = 1, deadline = 60.0)
    SolpropJulia.acknowledge!(session, first.generation)
    return @allocated SolpropJulia.advance!(session; work_limit = 1, deadline = 60.0)
end

@testset "Real Julia march with periodic, warning, plane and snapshot continuations" begin
    session = service_march_case()
    services = session.services
    initial = copy(session.field)
    launch = SolpropJulia.advance!(session; deadline = 60.0)
    @test launch.kind == SolpropJulia.SESSION_LAUNCH && services.plane_valid
    @test !services.started && session.source.calls == 0
    SolpropJulia.acknowledge!(session, launch.generation)
    progress = SolpropJulia.advance!(session; work_limit = 2, deadline = 60.0)
    @test progress.kind == SolpropJulia.SESSION_PROGRESS
    @test session.attempts.count == 2 && services.rows.count == 2
    indices = copy(services.rows.step[1:services.rows.count])
    @test indices == [0, 1]
    SolpropJulia.acknowledge!(session, progress.generation)
    warning = SolpropJulia.advance!(session; work_limit = 2, deadline = 60.0)
    @test warning.kind == SolpropJulia.SESSION_WARNING && warning.z == 0.1
    @test session.attempts.count == 2 && services.rows.count == 1
    append!(indices, services.rows.step[1:services.rows.count] .+ 2)
    @test services.notices.records[1].code == SolpropJulia.TemporalGuardWarning
    @test session.source.calls == 8 && session.state.accepted == 4
    SolpropJulia.request_snapshot!(session)
    SolpropJulia.acknowledge!(session, warning.generation)
    plane = SolpropJulia.advance!(session; deadline = 60.0)
    @test plane.kind == SolpropJulia.SESSION_PLANE && plane.z == 0.1
    @test services.rows.count == 1 && services.rows.step[1] == -1
    append!(indices, services.rows.step[1:services.rows.count] .+ 4)
    @test indices == [0, 1, 2, 3]
    @test services.plane_valid && session.source.calls == 8
    @test services.work.monitor.steps == 4
    # A four-sample shift and the four loss rates seen at rows5:8 give this solution.
    attenuation = exp(-0.5 * 0.025 * sum(services.split.rate[5:8]))
    phase = repeat(reshape([isodd(index) ? 1.0 : -1.0 for index in 1:8], 8, 1), 1, 2)
    @test isapprox(
        session.field, initial .* phase .* attenuation;
        rtol = 512eps(Float64), atol = 512eps(Float64),
    )
    SolpropJulia.acknowledge!(session, plane.generation)
    snapshot = SolpropJulia.advance!(session; deadline = 60.0)
    @test snapshot.kind == SolpropJulia.SESSION_SNAPSHOT
    @test session.attempts.count == 0 && services.rows.count == 0
    SolpropJulia.acknowledge!(session, snapshot.generation)
    final = SolpropJulia.advance!(session; deadline = 60.0)
    @test final.kind == SolpropJulia.SESSION_WARNING && services.finished
    @test services.notices.records[1].code == SolpropJulia.TemporalRemovalWarning
    SolpropJulia.acknowledge!(session, final.generation)
    completed = SolpropJulia.advance!(session; deadline = 60.0)
    @test completed.kind == SolpropJulia.SESSION_COMPLETED
    @test session.source.calls == 8 && session.state.attempts == 4
    @test session.state.rejected == 0 && !completed.poisoned
    SolpropJulia.acknowledge!(session, completed.generation)
    @test service_march_allocation(service_march_case(; warn = 1.1)) == 0
end
