# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function _temporal_policy(;
        split = false, quiet_collar = true,
        crossing_share = 0.0, bucket_limits = Float64[]
    )
    return SolpropJulia.TemporalMonitorPolicy(;
        warn = 0.1, stop = 0.3, activity = (0.1 * 0.1)^2,
        quiet_share = 1.0e-12, quiet_checks = 2, quiet_collar, cadence = 8,
        interval = 0.25, launch_floor_ratio = 1.0e-12, floor_margin = 3.0,
        split, crossing_share, bucket_limits
    )
end

function _temporal_reading(
        total, regions; removed = 0.0, found = nothing,
        occupancy = Float64[]
    )
    reading = SolpropJulia.TemporalReading(length(occupancy))
    reading.total, reading.regions = Float64(total), Tuple(Float64.(regions))
    reading.removed = removed
    if found !== nothing
        reading.found, reading.has_found = Tuple(Float64.(found)), true
    end
    copyto!(reading.occupancy, occupancy)
    return reading
end

function _temporal_reduce_allocations(reading, samples, tables, power, execution, spectrum)
    SolpropJulia.temporal_reading!(reading, samples, tables, power, execution; spectrum)
    return @allocated SolpropJulia.temporal_reading!(
        reading, samples, tables,
        power, execution; spectrum
    )
end

function _temporal_step_allocations(state, policy, reading, notices)
    SolpropJulia.temporal_step!(state, policy, reading, 0.5, notices)
    return @allocated SolpropJulia.temporal_step!(state, policy, reading, 0.5, notices)
end

@testset "Temporal monitor coarse proxy and split reductions" begin
    for tasks in (1, 2), kind in (:field, :envelope)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        samples = kind == :field ? reshape(sin.(1:12), 4, 3) :
            reshape(complex.(sin.(1:12), cos.(2:13)), 4, 3)
        regions = [
            1.0 0.0 0.25 0.5; 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0; 0.0 1.0 0.5 2.0
        ]
        columns = [0.25, 1.0, 2.0]
        buckets = [1 2 3; 2 3 1; 3 1 2]
        tables = SolpropJulia.TemporalMonitorWeights(regions, columns; buckets)
        reading = SolpropJulia.TemporalReading(3)
        power = fill(NaN, 4)
        spectrum = reshape(complex.(sin.(1:9), cos.(1:9)), 3, 3)
        SolpropJulia.temporal_reading!(reading, samples, tables, power, execution; spectrum)
        reference_power = [
            sum(
                BigFloat(columns[column]) *
                    abs2(Complex{BigFloat}(samples[row, column])) for column in 1:3
            ) for row in 1:4
        ]
        expected = [
            sum(reference_power); [
                sum(reference_power .* regions[:, region])
                    for region in 1:4
            ]
        ]
        actual = [reading.total; reading.regions...]
        # Three columns with two squares, a sum/product, then four time samples.
        # gamma_64 bounds each positive reduction's relative error.
        bound = 64eps(Float64) / (1 - 64eps(Float64)) * maximum(Float64.(expected))
        @test maximum(abs, actual - Float64.(expected)) <= bound
        @test _temporal_reduce_allocations(
            reading, samples, tables, power,
            execution, spectrum
        ) == 0
        occupancy = [
            sum(
                BigFloat(columns[column]) *
                    abs2(Complex{BigFloat}(spectrum[row, column]))
                    for row in 1:3, column in 1:3
                    if buckets[row, column] == bucket
            ) for bucket in 1:3
        ]
        @test maximum(abs, reading.occupancy - Float64.(occupancy)) <= bound
        step = 0.125
        SolpropJulia.split_reading!(reading, samples, step, tables, power, execution)
        removed = sum(reference_power .* -expm1.(-BigFloat.(regions[:, 4]) .* step))
        halfway = reference_power .* exp.(-BigFloat.(regions[:, 4]) .* step ./ 2)
        found = [sum(halfway); [sum(halfway .* regions[:, region]) for region in 1:3]]
        # Correctly rounded exp/expm1 plus 64 positive scalar operations, widened
        # to gamma_96 for the independent BigFloat transcendental witness.
        split_bound = 96eps(Float64) * maximum(Float64.(found))
        @test abs(reading.removed - Float64(removed)) <= split_bound
        @test maximum(abs, collect(reading.found) - Float64.(found)) <= split_bound
        old = reading.removed
        SolpropJulia.split_reading!(
            reading, samples, step, tables, power, execution;
            add = true
        )
        @test reading.removed == 2old
        floor = SolpropJulia.temporal_floor(reading, tables)
        @test floor.total == reading.total
        @test floor.regions == Tuple(reading.total .* vec(sum(regions; dims = 1)) ./ 4)
        expected_taken = floor.total / 4 * sum(-expm1.(-regions[:, 4] .* step))
        @test SolpropJulia.temporal_floor_taken(floor, tables, step) == expected_taken
        @test iszero(SolpropJulia.temporal_floor_taken(floor, tables, 0.0))
    end
end

@testset "Temporal scalar decisions, quiet cadence and dense plane rows" begin
    policy = _temporal_policy()
    state = SolpropJulia.TemporalMonitorState()
    notices = SolpropJulia.MonitorNotices(4)
    floor = SolpropJulia.TemporalFloor(0.0, (0.0, 0.0, 0.0, 0.0))
    reading = _temporal_reading(100.0, (0.0, 0.0, 0.0, 0.0))
    row = SolpropJulia.temporal_launch!(state, policy, reading, 0.0, floor, notices)
    @test row.emit && row.z == 0.0 && row.removed == 0.0
    @test state.quiet_run == 1 && !state.quiet && !state.busy
    @test !SolpropJulia.temporal_due(state, policy, 0.1)
    @test SolpropJulia.temporal_due(state, policy, 0.25)
    SolpropJulia.temporal_step!(state, policy, reading, 0.1, notices)
    @test state.quiet && state.quiet_run == 2 && state.steps == 1
    SolpropJulia.temporal_skip!(state, policy, 0.2)
    @test state.stale && state.skipped == 1 && state.since_check == 1
    SolpropJulia.temporal_step!(state, policy, reading, 0.2, notices; plane = true)
    @test !state.stale && state.steps == 1 && state.skipped == 1
    @test SolpropJulia.temporal_plane!(state, 0.2).emit
    dense = SolpropJulia.temporal_plane!(state, 0.21)
    @test dense.emit && isnan(dense.collar) && isnan(dense.guard_low)
    @test !SolpropJulia.temporal_plane!(state, 0.21).emit
    @test _temporal_step_allocations(state, policy, reading, notices) == 0
    @test notices.count == 0
    # Identical scalar readings make exact threshold decisions: equality does not warn.
    reading.regions = (1.0, 0.0, 0.0, 0.0)
    SolpropJulia.temporal_step!(state, policy, reading, 0.6, notices)
    @test notices.count == 0
    reading.regions = (4.0, 0.0, 0.0, 0.0)
    SolpropJulia.temporal_step!(state, policy, reading, 0.7, notices)
    @test notices.count == 1 && notices.records[1].code == SolpropJulia.TemporalGuardWarning
    SolpropJulia.temporal_step!(state, policy, reading, 0.8, notices)
    @test notices.count == 1
    reading.regions = (9.0, 0.0, 0.0, 0.0)
    SolpropJulia.temporal_step!(state, policy, reading, 0.9, notices)
    @test !SolpropJulia.monitor_stopped(notices)
    reading.regions = (16.0, 0.0, 0.0, 0.0)
    SolpropJulia.temporal_step!(state, policy, reading, 1.0, notices)
    @test SolpropJulia.monitor_stopped(notices)
    @test state.guard_max == (0.16, 0.0) && state.guard_at[1] == 1.0
    SolpropJulia.reset_notices!(notices)
    @test notices.count == 0
end

@testset "Temporal removal floors, left-out collar and split crossing" begin
    floor = SolpropJulia.TemporalFloor(10.0, (1.0, 1.0, 2.0, 3.0))
    policy = _temporal_policy()
    state = SolpropJulia.TemporalMonitorState()
    notices = SolpropJulia.MonitorNotices(4)
    reading = _temporal_reading(100.0, (1.0, 1.0, 2.0, 10.0))
    SolpropJulia.temporal_launch!(state, policy, reading, 0.0, floor, notices)
    SolpropJulia.temporal_step!(state, policy, reading, 1.0, notices)
    @test state.raw_removed == 0.1 && state.floor_removed == 0.03
    @test SolpropJulia.temporal_removed(state) == 0.1 - 0.03
    SolpropJulia.temporal_note_collar!(state, true)
    SolpropJulia.temporal_step!(state, policy, reading, 2.0, notices)
    @test state.raw_removed == 0.1 && state.left_out_checks == 1
    SolpropJulia.temporal_finish!(state, policy, notices)
    @test notices.records[notices.count].code == SolpropJulia.TemporalRemovalWarning
    state.floor_removed = 0.2
    @test SolpropJulia.temporal_removed(state) == 0.0

    split = _temporal_policy(;
        split = true, crossing_share = 0.25,
        bucket_limits = [0.1, 0.2, 0.3]
    )
    state = SolpropJulia.TemporalMonitorState()
    reading = _temporal_reading(
        100.0, (0.0, 0.0, 0.0, 0.0);
        found = (100.0, 0.0, 0.0, 0.0), occupancy = [1.0, 1.0, 2.0]
    )
    SolpropJulia.reset_notices!(notices)
    SolpropJulia.temporal_launch!(state, split, reading, 0.0, floor, notices)
    # searchsorted(side=right): equality at the first cumulative share chooses next.
    @test SolpropJulia.temporal_step_cap(state) == 0.2
    SolpropJulia.temporal_skip!(state, split, 0.1; floor_taken = 0.25)
    reading.removed = 2.0
    SolpropJulia.temporal_step!(state, split, reading, 0.2, notices; floor_taken = 0.75)
    @test state.raw_removed == 0.02 && state.floor_removed == 0.01
    @test state.floor_taken == 0.0
    reading.has_found = false
    @test_throws ArgumentError SolpropJulia.temporal_step!(
        state, split, reading, 0.3, notices
    )
end

@testset "Temporal continuation and no-light launch" begin
    policy = _temporal_policy()
    notices = SolpropJulia.MonitorNotices(2)
    zero_floor = SolpropJulia.TemporalFloor(0.0, (0.0, 0.0, 0.0, 0.0))
    state = SolpropJulia.TemporalMonitorState()
    reading = _temporal_reading(0.0, (0.0, 0.0, 0.0, 0.0))
    row = SolpropJulia.temporal_launch!(state, policy, reading, 0.0, zero_floor, notices)
    @test !row.emit && state.launch == 0.0
    reading.total = NaN
    SolpropJulia.temporal_step!(state, policy, reading, 0.1, notices)
    @test state.quiet && notices.count == 0
    state = SolpropJulia.TemporalMonitorState()
    reading.total = 100.0
    SolpropJulia.temporal_launch!(state, policy, reading, 0.0, zero_floor, notices)
    SolpropJulia.temporal_step!(state, policy, reading, 0.1, notices)
    SolpropJulia.temporal_plane!(state, 0.1)
    continuation = SolpropJulia.temporal_continuation(state)
    restored = SolpropJulia.TemporalMonitorState()
    SolpropJulia.restore_temporal!(restored, continuation; plane = true)
    # A resumed field already beyond the stop threshold is recorded, not judged again.
    reading.regions = (16.0, 0.0, 0.0, 0.0)
    row = SolpropJulia.temporal_launch!(restored, policy, reading, 0.1, zero_floor, notices)
    @test !row.emit && notices.count == 0
    @test restored.quiet_run == state.quiet_run && restored.quiet == state.quiet
    @test restored.launch == 100.0 && restored.steps == 1
    @test_throws ArgumentError SolpropJulia.restore_temporal!(
        restored, continuation;
        plane = true
    )
    @test_throws ArgumentError SolpropJulia.MonitorNotices(0)
    @test_throws ArgumentError SolpropJulia.TemporalReading(-1)
    @test_throws ArgumentError SolpropJulia.TemporalMonitorWeights(ones(4, 4), [-1.0])
end

@testset "Temporal monitor invalid and aliased numerical inputs" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    samples = ones(4, 2)
    regions = ones(4, 4)
    tables = SolpropJulia.TemporalMonitorWeights(regions, ones(2))
    reading = SolpropJulia.TemporalReading()
    for power in (@view(samples[:, 1]), @view(regions[:, 1]))
        retained = copy(power)
        @test_throws ArgumentError SolpropJulia.temporal_reading!(
            reading, samples,
            tables, power, execution
        )
        @test power == retained
    end
    @test_throws DimensionMismatch SolpropJulia.temporal_reading!(
        reading, samples,
        tables, zeros(3), execution
    )
    samples[1] = NaN
    @test_throws ArgumentError SolpropJulia.temporal_reading!(
        reading, samples,
        tables, zeros(4), execution
    )
    samples[1] = floatmax(Float64)
    @test_throws ArgumentError SolpropJulia.temporal_reading!(
        reading, samples,
        tables, zeros(4), execution
    )
    samples[1] = 1.0
    @test_throws ArgumentError SolpropJulia.split_reading!(
        reading, samples, 0.0,
        tables, zeros(4), execution
    )
    @test_throws ArgumentError SolpropJulia.TemporalFloor(-1.0, (0.0, 0.0, 0.0, 0.0))
    crossing = SolpropJulia.TemporalMonitorWeights(
        regions, ones(2);
        buckets = ones(Int, 4, 2)
    )
    bucket_reading = SolpropJulia.TemporalReading(1)
    scratch = fill(7.0, 4)
    @test_throws DimensionMismatch SolpropJulia.temporal_reading!(
        bucket_reading,
        samples, crossing, scratch, execution
    )
    @test scratch == fill(7.0, 4)
    @test_throws DimensionMismatch SolpropJulia.temporal_reading!(
        bucket_reading,
        samples, tables, scratch, execution
    )
    @test scratch == fill(7.0, 4)
    # A raw spectral view must be checked before the coarse-time scratch is written.
    spectrum = reshape(scratch, 4, 1)
    one_column = SolpropJulia.TemporalMonitorWeights(
        regions, ones(1);
        buckets = ones(Int, 4, 1)
    )
    @test_throws ArgumentError SolpropJulia.temporal_reading!(
        bucket_reading,
        ones(4, 1), one_column, scratch, execution; spectrum
    )
    @test scratch == fill(7.0, 4)
    # Splits preserve the prepared crossing map even though they do not read it (#634).
    retained_buckets = copy(one_column.buckets)
    aliased_power = reinterpret(Float64, vec(one_column.buckets))
    @test_throws ArgumentError SolpropJulia.split_reading!(
        bucket_reading, ones(4, 1), 0.125, one_column, aliased_power, execution
    )
    @test one_column.buckets == retained_buckets
end
