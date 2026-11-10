# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function _monitor_restore_policy(; quiet_collar = true, quiet_checks = 2)
    return SolpropJulia.TemporalMonitorPolicy(;
        warn = 0.1, stop = 0.3, activity = 1.0e-4, quiet_share = 1.0e-12,
        quiet_checks, quiet_collar, cadence = 8, interval = 0.25,
        launch_floor_ratio = 1.0e-12, floor_margin = 3.0, split = false,
    )
end

function _monitor_restore_record(;
        launch = 100.0, has_quiet_run = true, quiet_run = 0, quiet = false,
    )
    return SolpropJulia.TemporalContinuation(
        launch, (0.0, 0.0, 0.0), (0.0, 0.0), (NaN, NaN), 0.0,
        0.0, 0.0, quiet_run, quiet, 0.0, 5, 3, 2, NaN; has_quiet_run,
    )
end

function _monitor_restore_reading(; total = 100.0, guard = 0.0, collar = 0.0)
    reading = SolpropJulia.TemporalReading()
    reading.total, reading.regions = total, (guard, 0.0, collar, 0.0)
    return reading
end

function _monitor_restore_launch!(state, record, policy, reading, notices)
    SolpropJulia.restore_temporal!(state, record; plane = false)
    floor = SolpropJulia.TemporalFloor(0.0, (0.0, 0.0, 0.0, 0.0))
    SolpropJulia.temporal_launch!(state, policy, reading, 0.5, floor, notices)
    return nothing
end

function _monitor_restore_allocations(record)
    policy = _monitor_restore_policy()
    reading = _monitor_restore_reading()
    notices = SolpropJulia.MonitorNotices(1)
    warm = SolpropJulia.TemporalMonitorState()
    _monitor_restore_launch!(warm, record, policy, reading, notices)
    measured = SolpropJulia.TemporalMonitorState()
    return @allocated _monitor_restore_launch!(measured, record, policy, reading, notices)
end

@testset "Temporal continuation count presence and compatible construction" begin
    arguments = (
        100.0, (0.0, 0.0, 0.0), (0.0, 0.0), (NaN, NaN), 0.0,
        0.0, 0.0, 3, true, 0.0, 5, 3, 2, NaN,
    )
    previous = SolpropJulia.TemporalContinuation(arguments...)
    absent = SolpropJulia.TemporalContinuation(arguments..., false)
    explicit = SolpropJulia.TemporalContinuation(arguments...; has_quiet_run = false)
    @test previous.has_quiet_run && previous.quiet_run == 3 && previous.quiet
    @test !absent.has_quiet_run && !explicit.has_quiet_run
    @test isbitstype(SolpropJulia.TemporalContinuation)
    for name in fieldnames(SolpropJulia.TemporalContinuation)
        @test isequal(getfield(absent, name), getfield(explicit, name))
    end
    state = SolpropJulia.TemporalMonitorState(; quiet_run = 4, quiet = true)
    published = SolpropJulia.temporal_continuation(state)
    @test published.has_quiet_run && published.quiet_run == 4 && published.quiet
end

@testset "An absent old count preserves the newly settled launch" begin
    # These are exact integer/Boolean state transitions on identical readings,
    # including the zero-light branch; no rounding budget applies.
    for quiet_collar in (false, true), quiet_checks in (1, 2),
            total in (0.0, 100.0), calm in (false, true)
        policy = _monitor_restore_policy(; quiet_collar, quiet_checks)
        collar = calm || total == 0.0 ? 0.0 : 1.0
        reading = _monitor_restore_reading(; total, collar)
        expected_count = total == 0.0 || calm ? 1 : 0
        for (present, old_count, old_quiet) in (
                (false, 0, false), (false, 7, true),
                (true, 0, false), (true, 3, true),
            )
            state = SolpropJulia.TemporalMonitorState()
            notices = SolpropJulia.MonitorNotices(1)
            record = _monitor_restore_record(;
                has_quiet_run = present, quiet_run = old_count, quiet = old_quiet,
            )
            _monitor_restore_launch!(state, record, policy, reading, notices)
            @test state.quiet_run == (present ? old_count : expected_count)
            expected_quiet = present ? old_quiet : expected_count >= quiet_checks
            @test state.quiet == (quiet_collar && expected_quiet)
            @test !state.restore_quiet && !state.resuming
            @test notices.count == 0
            @test state.launch == 100.0
            @test state.steps == 5 && state.skipped == 3 && state.left_out_checks == 2
            published = SolpropJulia.temporal_continuation(state)
            @test published.has_quiet_run && published.quiet_run == state.quiet_run
        end
    end
end

@testset "A present count is restored only once before later readings" begin
    policy = _monitor_restore_policy()
    reading = _monitor_restore_reading()
    for present in (false, true)
        state = SolpropJulia.TemporalMonitorState()
        notices = SolpropJulia.MonitorNotices(1)
        record = _monitor_restore_record(; has_quiet_run = present)
        _monitor_restore_launch!(state, record, policy, reading, notices)
        @test state.quiet_run == (present ? 0 : 1)
        SolpropJulia.temporal_step!(state, policy, reading, 0.6, notices)
        @test state.quiet_run == (present ? 1 : 2)
        @test state.quiet == !present
        @test !state.restore_quiet
        old = SolpropJulia.temporal_continuation(state)
        @test_throws ArgumentError SolpropJulia.restore_temporal!(
            state, record; plane = false,
        )
        after = SolpropJulia.temporal_continuation(state)
        for name in fieldnames(SolpropJulia.TemporalContinuation)
            @test isequal(getfield(old, name), getfield(after, name))
        end
    end
end

@testset "Resume without usable old totals never rejudges the launch" begin
    policy = _monitor_restore_policy()
    reading = _monitor_restore_reading(; guard = 16.0)
    floor = SolpropJulia.TemporalFloor(0.0, (0.0, 0.0, 0.0, 0.0))
    for record in (
                nothing, _monitor_restore_record(; launch = 0.0),
                _monitor_restore_record(; launch = -1.0),
                _monitor_restore_record(; launch = NaN),
            ), plane in (false, true)
        state = SolpropJulia.TemporalMonitorState()
        notices = SolpropJulia.MonitorNotices(1)
        SolpropJulia.restore_temporal!(state, record; plane)
        @test state.resuming && state.resume_plane == plane && !state.restore_quiet
        row = SolpropJulia.temporal_launch!(state, policy, reading, 0.5, floor, notices)
        @test notices.count == 0 && !state.resuming
        @test row.emit == plane
        @test state.launch == 100.0 && state.quiet_run == 1
        @test state.steps == 0 && state.skipped == 0
    end
    fresh = SolpropJulia.TemporalMonitorState()
    notices = SolpropJulia.MonitorNotices(1)
    SolpropJulia.temporal_launch!(fresh, policy, reading, 0.5, floor, notices)
    @test notices.count == 1 && notices.records[1].code == SolpropJulia.TemporalGuardStop
end

@testset "Temporal count restoration allocates no warmed workspace" begin
    for record in (
            nothing, _monitor_restore_record(),
            _monitor_restore_record(; has_quiet_run = false),
        )
        @test _monitor_restore_allocations(record) == 0
    end
end
