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

function _transverse_policy(; design = false, collar = false)
    return SolpropJulia.TransversePolicy(;
        warn = 0.01, stop = 0.09,
        backstop_start = 0.0625, backstop_max = 0.25, reach_growth = 0.1,
        guard_numerator = 0.2, k_min = 10.0, k_floor_fraction = 0.05,
        due_tolerance = 1.0e-12, noise_factor = 8.0, strip_floor_margin = 3.0,
        strip_leakage = 1.0e-6, has_design = design, has_collar = collar
    )
end

function _transverse_tables(; ramp = false)
    faces = zeros(4, 4)
    faces[1, 1], faces[4, 2] = 0.25, 0.25
    clip = copy(faces)
    totals = zeros(4, 4)
    totals[:, 1] .= 0.25
    totals[:, 2] .= [0.125, 0.0, 0.0, 0.25]
    totals[:, 3] .= [0.25, 0.0, 0.0, 0.25]
    return SolpropJulia.TransverseTables(
        faces, clip, totals, ones(4),
        [0.0, 1.0, 3.0, 6.0], [1, 2, 3, 4], ramp ? [0.0, 0.0, 0.0, 1.0] : zeros(4);
        face_ranks = (2, 1), noise_face_share = 0.25, noise_modes = 16
    )
end

function _transverse_reading(;
        total = 100.0, faces = (0.0, 0.0, 0.0, 0.0),
        columns = [100.0, 0.0, 0.0, 0.0], rate = 0.0, outside = 0.0,
        strip = zeros(0, 2), current = nothing
    )
    reading = SolpropJulia.TransverseReading(4, size(strip, 1))
    reading.faces, reading.totals = faces, (total, rate, outside, 0.0)
    reading.has_faces, reading.full = true, true
    copyto!(reading.columns, columns)
    copyto!(reading.strip, strip)
    reading.has_strip = true
    if current !== nothing
        reading.current, reading.has_current = current, true
    end
    return reading
end

function _transverse_reading_allocations(
        reading, nodes, columns, tables, scratch, execution
    )
    SolpropJulia.transverse_reading!(
        reading, nodes, columns, tables, scratch, execution;
        full = true
    )
    return @allocated SolpropJulia.transverse_reading!(
        reading, nodes, columns, tables,
        scratch, execution; full = true
    )
end

function _transverse_control_allocations(state, policy, tables, reading, notices)
    first_z = state.last_z + 0.125
    second_z = first_z + 0.125
    # Julia 1.13 wraps a qualified @allocated call in a closure; a reassigned z
    # would box the test's arguments. Measure the concrete kernel call directly.
    read! = SolpropJulia.transverse_read!
    read!(state, policy, tables, reading, first_z, notices)
    return @allocated read!(state, policy, tables, reading, second_z, notices)
end

function _transverse_strip_allocations(reading, lines, execution)
    SolpropJulia.transverse_strip!(reading, lines, execution)
    return @allocated SolpropJulia.transverse_strip!(reading, lines, execution)
end

@testset "Transverse warning cursors preserve unperformed stages" begin
    tables, policy = _transverse_tables(; ramp = true), _transverse_policy(; design = true)
    reading = _transverse_reading(;
        faces = (2.0, 3.0, 0.0, 0.0), columns = [98.0, 0.0, 0.0, 2.0], rate = 4.0,
    )
    reading.clip = (2.0, 3.0, 0.0, 0.0)
    noise = _transverse_reading(; total = 4.0, columns = [0.0, 0.0, 0.0, 4.0])
    state = SolpropJulia.TransverseState(tables)
    eager = SolpropJulia.TransverseState(tables)
    notices = SolpropJulia.MonitorNotices(1)
    accumulated = SolpropJulia.MonitorNotices(8)
    SolpropJulia.transverse_launch!(
        eager, policy, tables, reading, 0.0, accumulated; noise, fixed_step_max = 0.5,
    )
    SolpropJulia._transverse_begin_launch!(
        state, policy, tables, reading, 0.0, notices; noise, fixed_step_max = 0.5,
    )
    @test notices.count == 1
    @test notices.records[1].code == SolpropJulia.TransverseLaunchWarning
    @test state.baseline == (0.0, 0.0) && iszero(state.allowance)
    @test isnan(state.last_rate_z) && all(iszero, state.last_columns)
    @test_throws ArgumentError SolpropJulia._transverse_begin_read!(
        state, policy, tables, reading, 0.2, notices,
    )
    @test_throws ArgumentError SolpropJulia._transverse_begin_launch!(
        state, policy, tables, reading, 0.0, notices,
    )
    @test state.baseline == (0.0, 0.0) && state.reads == 0
    for index in 2:accumulated.count
        SolpropJulia.reset_notices!(notices)
        SolpropJulia._transverse_proceed!(
            state, policy, tables, reading, notices; noise, fixed_step_max = 0.5,
        )
        @test notices.count == 1
        @test isequal(notices.records[1], accumulated.records[index])
        if index == 2
            @test state.baseline == (0.02, 0.0)
            @test !state.has_noise && all(iszero, state.last_columns)
        elseif index == 3
            @test state.baseline == (0.02, 0.03)
            @test !state.has_noise && all(iszero, state.last_columns)
        elseif index == 4
            @test state.has_noise && state.allowance == 0.02
            @test all(iszero, state.last_columns) && isinf(state.next)
        else
            @test isfinite(state.next) && isnan(state.last_rate_z)
        end
    end
    SolpropJulia.reset_notices!(notices)
    SolpropJulia._transverse_proceed!(
        state, policy, tables, reading, notices; noise, fixed_step_max = 0.5,
    )
    @test notices.count == 0 && state.stage == 0
    for name in fieldnames(typeof(state))
        @test isequal(getfield(state, name), getfield(eager, name))
    end

    # Face and removal warnings share an observation but have separate barriers:
    # counters/removal are credited once, and ramp_warned changes only after resume.
    plain = SolpropJulia.TransverseState(tables)
    base = _transverse_reading(; columns = [98.0, 0.0, 0.0, 2.0])
    SolpropJulia.transverse_launch!(plain, policy, tables, base, 0.0, accumulated)
    base.faces = (3.0, 0.0, 0.0, 0.0)
    SolpropJulia._transverse_begin_read!(plain, policy, tables, base, 10.0, notices)
    @test notices.count == 1
    @test notices.records[1].code == SolpropJulia.TransverseFaceWarning
    @test plain.reads == 1 && !plain.ramp_warned
    credited = plain.ramp_raw
    SolpropJulia.reset_notices!(notices)
    SolpropJulia._transverse_proceed!(plain, policy, tables, base, notices)
    @test notices.count == 1
    @test notices.records[1].code == SolpropJulia.TransverseRampWarning
    @test plain.reads == 1 && plain.ramp_raw == credited && plain.ramp_warned
    @test plain.stage == 0
end

@testset "Transverse numerical norms, regions and stale flags" begin
    for tasks in (1, 2)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        tables = _transverse_tables()
        node_spectrum = reshape(complex.(sin.(1:20), cos.(2:21)), 5, 4)
        column_spectrum = 2 .* node_spectrum
        scratch = fill(NaN, 4)
        reading = SolpropJulia.TransverseReading(4, 2)
        reading.has_strip = reading.has_current = true
        SolpropJulia.transverse_reading!(
            reading, node_spectrum, column_spectrum,
            tables, scratch, execution; full = true
        )
        norm = vec(sum(abs2, Complex{BigFloat}.(node_spectrum); dims = 1))
        expected = [
            sum(norm .* table[:, region]) for region in 1:4,
                table in (tables.faces, tables.clip, tables.totals)
        ]
        actual = hcat(
            collect(reading.faces), collect(reading.clip), collect(reading.totals)
        )
        # Five complex squared norms followed by four-node positive reductions:
        # gamma_64 bounds each output using the largest absolute weighted norm.
        bound = 64eps(Float64) / (1 - 64eps(Float64)) * maximum(Float64.(expected))
        @test maximum(abs, actual - Float64.(expected)) <= bound
        @test maximum(abs, reading.columns - 4Float64.(norm)) <= 4bound
        @test !reading.has_strip && !reading.has_current && reading.full
        @test _transverse_reading_allocations(
            reading, node_spectrum, column_spectrum,
            tables, scratch, execution
        ) == 0
        @test_throws ArgumentError SolpropJulia.transverse_reading!(
            reading,
            node_spectrum, column_spectrum, tables, reading.columns, execution
        )
        @test_throws DimensionMismatch SolpropJulia.transverse_reading!(
            reading,
            node_spectrum, column_spectrum, tables, zeros(3), execution
        )
    end
end

@testset "Transverse strip Parseval factors and inward current" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    # A pure discrete Fourier mode has one nonzero raw FFT bin. Its energy is
    # n_fft*spacing times the squared amplitude, then the line measure.
    spectrum = zeros(ComplexF64, 8, 2)
    spectrum[3, :] .= [8.0, 4.0im]
    inward = SolpropJulia.TransverseStripLine(spectrum, 0.25, 2.0, 2, 4)
    outward = SolpropJulia.TransverseStripLine(spectrum, 0.25, 2.0, 5, 8)
    empty_line = SolpropJulia.TransverseStripLine(zeros(ComplexF64, 0, 0), 0.25, 2.0, 1, 0)
    reading = SolpropJulia.TransverseReading(4, 3)
    lines = (inward, outward, empty_line)
    SolpropJulia.transverse_strip!(reading, lines, execution)
    @test reading.strip == [5.0 5.0; 0.0 5.0; 0.0 0.0]
    @test reading.has_strip
    @test _transverse_strip_allocations(reading, lines, execution) == 0
    # E=1+i*q*x gives Im(conj(E)*dE/dx)=q exactly at x=0; the opposite
    # q channel is outgoing and is rectified separately before the sum.
    samples = ComplexF64[1 - 1im 1 + 2im; 1 1; 1 + 1im 1 - 2im]
    current_line = SolpropJulia.TransverseCurrentLine(samples, 2, 1, 3, 2.0, 3.0)
    SolpropJulia.transverse_current!(reading, (current_line,), 2.0)
    @test reading.current == 1.5 && reading.has_current
    @test_throws ArgumentError SolpropJulia.TransverseStripLine(spectrum, -1.0, 2.0, 1, 8)
    @test_throws ArgumentError SolpropJulia.TransverseCurrentLine(
        samples, 9, 1, 3, 2.0, 3.0
    )
end

@testset "Transverse cadence, warning order and lexical stop ties" begin
    tables, policy = _transverse_tables(), _transverse_policy()
    # The face count must come from prepared tables, not an unbound type parameter
    # in the generated unparameterized keyword constructor (issue #645).
    @test_throws MethodError SolpropJulia.TransverseState()
    @test !hasmethod(SolpropJulia.TransverseState, Tuple{})
    state = SolpropJulia.TransverseState(tables)
    @test state.baseline == state.peak == state.peak_raw == (0.0, 0.0)
    @test state.last_columns !== state.noise_columns
    reading = _transverse_reading()
    notices = SolpropJulia.MonitorNotices(8)
    SolpropJulia.transverse_launch!(state, policy, tables, reading, 0.0, notices)
    @test state.next == 0.0625 && state.backstop == 0.0625
    @test SolpropJulia.transverse_ceiling(state) == 0.0625
    @test !SolpropJulia.transverse_due(state, policy, 0.0, 0.01)
    @test SolpropJulia.transverse_due(state, policy, 0.0, 0.0625)
    @test !SolpropJulia.transverse_wants_read(state, 0.0)
    reading.columns .= [0.0, 100.0, 0.0, 0.0]
    SolpropJulia.transverse_read!(state, policy, tables, reading, 0.05, notices)
    reference = state.reach_reference
    @test reference > 0.0 && state.backstop == 0.0625
    SolpropJulia.transverse_read!(state, policy, tables, reading, 0.1, notices)
    @test state.backstop == 0.125 && state.reach_reference == reference
    SolpropJulia.transverse_read!(state, policy, tables, reading, 0.15, notices)
    @test state.backstop == 0.25
    @test _transverse_control_allocations(state, policy, tables, reading, notices) == 0
    reading.columns .= [0.0, 0.0, 100.0, 0.0]
    SolpropJulia.transverse_read!(state, policy, tables, reading, 0.5, notices)
    @test state.backstop == 0.0625 && state.reach_reference > reference * 1.1
    reading.faces = (4.0, 5.0, 0.0, 0.0)
    SolpropJulia.transverse_read!(state, policy, tables, reading, 0.6, notices)
    @test notices.count == 1 && notices.records[1].face == 1
    @test state.status == SolpropJulia.TransverseWarned
    SolpropJulia.reset_notices!(notices)
    reading.faces = (16.0, 16.0, 0.0, 0.0)
    SolpropJulia.transverse_read!(state, policy, tables, reading, 0.7, notices)
    @test SolpropJulia.monitor_stopped(notices)
    @test notices.records[1].face == 1 # x- is lexically greater than x+.
    @test state.status == SolpropJulia.TransverseStopped
end

@testset "Transverse ramp declaration, noise allowance and unwatched state" begin
    tables, policy = _transverse_tables(; ramp = true), _transverse_policy(; design = true)
    notices = SolpropJulia.MonitorNotices(8)
    reading = _transverse_reading(; columns = [98.0, 0.0, 0.0, 2.0])
    state = SolpropJulia.TransverseState(tables)
    SolpropJulia.transverse_launch!(state, policy, tables, reading, 0.0, notices)
    @test notices.records[1].code == SolpropJulia.TransverseLaunchWarning
    SolpropJulia.reset_notices!(notices)
    SolpropJulia.transverse_read!(state, policy, tables, reading, 10.0, notices)
    @test state.ramp_warned
    @test notices.records[1].code == SolpropJulia.TransverseRampWarning
    declared = SolpropJulia.TransverseState(tables)
    SolpropJulia.reset_notices!(notices)
    SolpropJulia.transverse_launch!(
        declared, policy, tables, reading, 0.0, notices;
        declared = true
    )
    SolpropJulia.transverse_read!(declared, policy, tables, reading, 10.0, notices)
    @test notices.count == 0 && declared.declared_share == 0.02
    @test SolpropJulia.transverse_ramp_removed(declared) < 0.0
    noise = _transverse_reading(; total = 4.0, columns = [0.0, 0.0, 0.0, 4.0])
    noisy = SolpropJulia.TransverseState(tables)
    SolpropJulia.transverse_launch!(
        noisy, policy, tables, reading, 0.0, notices;
        noise, clean_columns = [98.0, 0.0, 0.0, 0.0]
    )
    @test noisy.allowance == 0.02 && noisy.status == SolpropJulia.TransverseInsensitive
    @test noisy.noise_share_in_ramp == 0.04
    unwatched = SolpropJulia.TransverseState(tables)
    SolpropJulia.reset_notices!(notices)
    SolpropJulia.transverse_launch!(
        unwatched, policy, tables, reading, 0.0, notices;
        reads = false, declared = true
    )
    @test unwatched.status == SolpropJulia.TransverseUnwatched
    @test !SolpropJulia.transverse_wants_read(unwatched, 1.0)
    @test isinf(SolpropJulia.transverse_ceiling(unwatched))
end

@testset "Transverse collar returns and immutable continuation" begin
    tables = _transverse_tables()
    policy = _transverse_policy(; design = true, collar = true)
    state = SolpropJulia.TransverseState(tables)
    notices = SolpropJulia.MonitorNotices(8)
    reading = _transverse_reading(; rate = 2.0, outside = 5.0, strip = [1.0 2.0; 0.0 1.0])
    SolpropJulia.transverse_launch!(state, policy, tables, reading, 0.0, notices)
    @test state.strip_baseline == 0.01 && state.current_baseline == 0.05
    SolpropJulia.transverse_step!(state, policy, 100.0, 0.1)
    SolpropJulia.transverse_step!(state, policy, 100.0, 0.2)
    @test state.current == 0.1
    reading.faces = (50.0, 40.0, 0.0, 0.0)
    reading.strip .= [4.0 8.0; 0.0 1.0]
    SolpropJulia.transverse_read!(state, policy, tables, reading, 0.2, notices)
    @test state.peak == (0.0, 0.0) && state.peak_raw == (0.5, 0.4)
    @test state.returned_peak == 0.05
    @test notices.records[1].code == SolpropJulia.TransverseReturnWarning
    continuation = SolpropJulia.transverse_continuation(state)
    resumed = SolpropJulia.TransverseState(tables)
    SolpropJulia.restore_transverse!(resumed, continuation)
    SolpropJulia.reset_notices!(notices)
    reading.strip .= 1000.0
    reading.current, reading.has_current = 100.0, true
    SolpropJulia.transverse_launch!(resumed, policy, tables, reading, 0.2, notices)
    @test notices.count == 0 && resumed.launch == state.launch
    @test resumed.current == state.current && resumed.strip_baseline == state.strip_baseline
    @test resumed.has_current_last && resumed.current_last == 1.0
    @test SolpropJulia.transverse_finish!(resumed).reads == state.reads
    @test_throws ArgumentError SolpropJulia.restore_transverse!(resumed, continuation)
end

@testset "Transverse early refusal, launch stops and zero launch" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    tables, policy = _transverse_tables(), _transverse_policy()
    spectrum = ones(ComplexF64, 3, 4)
    reading = _transverse_reading()
    for scratch in (
            tables.column_weights, tables.kappa, tables.ramp_rates,
            reinterpret(Float64, tables.order),
        )
        retained = copy(scratch)
        @test_throws ArgumentError SolpropJulia.transverse_reading!(
            reading,
            spectrum, spectrum, tables, scratch, execution
        )
        @test isequal(scratch, retained)
    end
    reading.columns = tables.column_weights
    retained = copy(tables.column_weights)
    @test_throws ArgumentError SolpropJulia.transverse_reading!(
        reading,
        spectrum, spectrum, tables, zeros(4), execution
    )
    @test tables.column_weights == retained
    reading = _transverse_reading()
    state = SolpropJulia.TransverseState(tables)
    state.last_columns = tables.ramp_rates
    @test_throws ArgumentError SolpropJulia.transverse_launch!(
        state,
        policy, tables, reading, 0.0, SolpropJulia.MonitorNotices(8)
    )
    @test all(iszero, tables.ramp_rates)
    state = SolpropJulia.TransverseState(tables)
    @test_throws ArgumentError SolpropJulia.transverse_launch!(
        state,
        policy, tables, reading, 0.0, SolpropJulia.MonitorNotices(8);
        clean_columns = state.last_columns
    )
    @test isnan(state.launch)
    reading = SolpropJulia.TransverseReading(4, 2)
    reading.columns = vec(reading.strip)
    line = SolpropJulia.TransverseStripLine(ones(ComplexF64, 2, 2), 1.0, 1.0, 1, 1)
    @test_throws ArgumentError SolpropJulia.transverse_strip!(
        reading, (line, line), execution
    )
    @test all(iszero, reading.strip)
    reading = SolpropJulia.TransverseReading(4, 2)
    line = SolpropJulia.TransverseStripLine(reading.strip, 1.0, 1.0, 1, 1)
    @test_throws ArgumentError SolpropJulia.transverse_strip!(
        reading, (line, line), execution
    )
    @test all(iszero, reading.strip)
    reading = _transverse_reading(; total = 0.0)
    state = SolpropJulia.TransverseState(tables)
    notices = SolpropJulia.MonitorNotices(8)
    SolpropJulia.transverse_launch!(state, policy, tables, reading, 0.0, notices)
    @test state.launch == 0 && !isfinite(state.next) && notices.count == 0
    @test !SolpropJulia.transverse_wants_read(state, 1.0)
    @test !SolpropJulia.transverse_due(state, policy, 1.0)
    tables, policy = _transverse_tables(; ramp = true), _transverse_policy(; design = true)
    reading = _transverse_reading(; columns = [80.0, 0.0, 0.0, 20.0])
    state = SolpropJulia.TransverseState(tables)
    SolpropJulia.transverse_launch!(state, policy, tables, reading, 0.0, notices)
    @test SolpropJulia.monitor_stopped(notices)
    @test notices.records[1].code == SolpropJulia.TransverseLaunchStop
    @test !isfinite(state.next) && all(iszero, state.last_columns)
    @test all(iszero, state.baseline)
    SolpropJulia.reset_notices!(notices)
    @test_throws ArgumentError SolpropJulia.TransverseReading(0)
    @test_throws DimensionMismatch SolpropJulia.TransverseTables(
        zeros(4, 3),
        tables.clip, tables.totals, tables.column_weights, tables.kappa, tables.order,
        tables.ramp_rates; face_ranks = (2, 1), noise_face_share = 0.25, noise_modes = 16
    )
    @test_throws ArgumentError SolpropJulia.TransverseTables(
        tables.faces,
        tables.clip, tables.totals, tables.column_weights, tables.kappa, [1, 1, 3, 4],
        tables.ramp_rates; face_ranks = (2, 1), noise_face_share = 0.25, noise_modes = 16
    )
    @test_throws ArgumentError SolpropJulia.TransverseTables(
        tables.faces,
        tables.clip, tables.totals, tables.column_weights, tables.kappa, tables.order,
        tables.ramp_rates; face_ranks = (1, 1), noise_face_share = 0.25, noise_modes = 16
    )
    reading = _transverse_reading()
    reading.full = false
    @test_throws ArgumentError SolpropJulia.transverse_launch!(
        SolpropJulia.TransverseState(tables), policy, tables, reading, 0.0, notices
    )
    reading.full = true
    reading.columns[1] = NaN
    @test_throws ArgumentError SolpropJulia.transverse_launch!(
        SolpropJulia.TransverseState(tables), policy, tables, reading, 0.0, notices
    )
    reading = _transverse_reading(; current = Inf)
    @test_throws ArgumentError SolpropJulia.transverse_launch!(
        SolpropJulia.TransverseState(tables), policy, tables, reading, 0.0, notices
    )
    reading = _transverse_reading(; total = nextfloat(0.0), faces = (1.0, 0.0, 0.0, 0.0))
    @test_throws ArgumentError SolpropJulia.transverse_launch!(
        SolpropJulia.TransverseState(tables), policy, tables, reading, 0.0, notices
    )
    reading = _transverse_reading(; total = nextfloat(0.0), rate = 1.0)
    collar_policy = _transverse_policy(; design = true, collar = true)
    @test_throws ArgumentError SolpropJulia.transverse_launch!(
        SolpropJulia.TransverseState(tables), collar_policy, tables, reading, 0.0, notices
    )
end

@testset "Transverse face metadata must match the prepared state" begin
    two = _transverse_tables()
    @test_throws ArgumentError SolpropJulia.TransverseTables(
        two.faces, two.clip, two.totals, two.column_weights, two.kappa, two.order,
        two.ramp_rates; face_ranks = (), noise_face_share = 0.25, noise_modes = 16,
    )
    faces = copy(two.faces)
    faces[:, 2] .= 0.0
    one = SolpropJulia.TransverseTables(
        faces, two.clip, two.totals, two.column_weights, two.kappa, two.order,
        two.ramp_rates; face_ranks = (1,), noise_face_share = 0.25, noise_modes = 16
    )
    reading, policy = _transverse_reading(), _transverse_policy()
    notices = SolpropJulia.MonitorNotices(8)
    for (prepared, mismatched) in ((one, two), (two, one))
        state = SolpropJulia.TransverseState(prepared)
        @test_throws DimensionMismatch SolpropJulia.transverse_launch!(
            state, policy, mismatched, reading, 0.0, notices
        )
        @test isnan(state.launch) && all(iszero, state.last_columns)
        SolpropJulia.transverse_launch!(state, policy, prepared, reading, 0.0, notices)
        retained = SolpropJulia.transverse_continuation(state)
        previous = copy(state.last_columns)
        @test_throws DimensionMismatch SolpropJulia.transverse_read!(
            state, policy, mismatched, reading, 0.25, notices
        )
        @test isequal(SolpropJulia.transverse_continuation(state), retained)
        @test state.last_z == 0.0 && state.last_columns == previous
    end
end
