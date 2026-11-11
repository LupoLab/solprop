# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function transverse_observation_case(
        route; collar = true, tasks = 1,
        arithmetic = SolpropJulia.SourceArithmetic(Float64),
    )
    execution = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), tasks, true,
    )
    nx, ny = route == :radial ? (8, 1) : route == :line ? (8, 1) : (8, 6)
    nc, nf = nx * ny, 5
    map = if route == :radial
        synthesis = [
            cospi((node - 0.5) * (column - 1) / nx) / sqrt(nx)
                for node in 1:nx, column in 1:nx
        ]
        SolpropJulia.RadialObservationMap(synthesis; pad_scale = 1.0, arithmetic)
    else
        SolpropJulia.CartesianObservationMap((nx, ny); arithmetic)
    end
    faces = zeros(nc, 4)
    for node in 1:nc
        ix, iy = mod(node - 1, nx) + 1, div(node - 1, nx) + 1
        if route == :radial
            faces[node, 1] = ix >= nx - 1 ? 1 / nc : 0.0
        else
            faces[node, 1] = ix <= 2 ? 1 / nc : 0.0
            faces[node, 2] = ix >= nx - 1 ? 1 / nc : 0.0
            if ny > 1
                faces[node, 3] = iy <= 2 ? 1 / nc : 0.0
                faces[node, 4] = iy >= ny - 1 ? 1 / nc : 0.0
            end
        end
    end
    totals = zeros(nc, 4)
    totals[:, 1] .= 1 / nc
    totals[:, 2] .= collar ? 0.01 / nc : 0.0
    totals[:, 3] .= 0.1 / nc
    n_faces = route == :radial ? 1 : ny == 1 ? 2 : 4
    tables = SolpropJulia.TransverseTables(
        faces, zeros(nc, 4), totals, fill(1 / nc, nc),
        collect(0.0:(nc - 1)) ./ 100, collect(1:nc), zeros(nc);
        face_ranks = ntuple(identity, n_faces), noise_face_share = 0.2, noise_modes = 20,
    )
    policy = SolpropJulia.TransversePolicy(;
        warn = 0.4, stop = 0.9, backstop_start = 0.05, backstop_max = 0.2,
        reach_growth = 0.1, guard_numerator = 0.2, k_min = 100.0,
        k_floor_fraction = 0.05, due_tolerance = 1.0e-12, noise_factor = 8.0,
        strip_floor_margin = 3.0, strip_leakage = 1.0e-6,
        has_design = true, has_collar = collar,
    )
    lines = if !collar
        ()
    else
        ntuple(n_faces) do index
            axis = route == :radial ? :r : index <= 2 ? :x : :y
            count = axis == :y ? ny : nx
            positive = route == :radial || isodd(index)
            strip = positive ? collect(1:count) : collect(count:-1:1)
            node = count ÷ 2
            onset = positive ? (node, node - 1, node + 1) : (node, node + 1, node - 1)
            SolpropJulia.TransverseLineData(
                axis, route == :radial ? sqrt.(collect(1:count) ./ count) : ones(count);
                strip, window = [0.2 + 0.8sinpi((i - 1) / (count - 1))^2 for i in 1:count],
                spacing = 0.1, measure = axis == :r ? 2π : ny == 1 ? 1.0 : 0.1,
                inward = (2, count), onset, onset_step = 0.2,
            )
        end
    end
    prepared = SolpropJulia.TransversePreparation(
        execution, map, policy, tables; n_frequency = nf, rows = fill(true, nc),
        lines, declared = false,
    )
    field = [
        complex(sin(0.7row + column / 3), cos(row / 2 - 0.2column))
            for row in 1:nf, column in 1:nc
    ]
    return execution, prepared, field
end

function transverse_observation_reference(prepared, field)
    nf, nc = size(field)
    input = Complex{BigFloat}.(field)
    physical, envelope = if prepared.map isa SolpropJulia.RadialObservationMap
        synthesis = BigFloat.(prepared.map.synthesis)
        input * transpose(synthesis), abs.(input) * transpose(abs.(synthesis))
    else
        nx, ny = prepared.map.shape
        inverse = [
            cis(
                2big(π) * (
                    BigFloat(mod(a - 1, nx) * mod(b - 1, nx)) / nx +
                        BigFloat(div(a - 1, nx) * div(b - 1, nx)) / ny
                )
            ) / nc for a in 1:nc, b in 1:nc
        ]
        input * transpose(inverse), repeat(sum(abs.(input); dims = 2) ./ nc, 1, nc)
    end
    nodes = vec(sum(abs2, physical; dims = 1))
    columns = vec(sum(abs2, input; dims = 1))
    node_scale = vec(sum(abs2, envelope; dims = 1))
    faces = transpose(BigFloat.(prepared.tables.faces)) * nodes
    clip = transpose(BigFloat.(prepared.tables.clip)) * nodes
    totals = transpose(BigFloat.(prepared.tables.totals)) * nodes
    strip = zeros(BigFloat, length(prepared.lines), 2)
    current = BigFloat(0)
    strip_scale, current_scale = BigFloat(0), BigFloat(0)
    for (index, line) in pairs(prepared.lines)
        n_axis, n_cross = SolpropJulia._transverse_line_shape(prepared.map, line)
        samples = [
            BigFloat(line.sample_scale[node]) * physical[
                freq,
                SolpropJulia._transverse_line_node(prepared.map, line, node, cross),
            ]
                for node in 1:n_axis, freq in 1:nf, cross in 1:n_cross
        ]
        bound_samples = [
            BigFloat(line.sample_scale[node]) * envelope[
                freq,
                SolpropJulia._transverse_line_node(prepared.map, line, node, cross),
            ]
                for node in 1:n_axis, freq in 1:nf, cross in 1:n_cross
        ]
        samples = reshape(samples, n_axis, :)
        bound_samples = reshape(bound_samples, n_axis, :)
        count = 2length(line.strip)
        if count > 0
            for trace in axes(samples, 2), bin in 1:count
                value = sum(
                    BigFloat(line.window[i]) * samples[line.strip[i], trace] *
                        cis(-2big(π) * (bin - 1) * (i - 1) / count)
                        for i in eachindex(line.strip)
                )
                scale = sum(
                    BigFloat(line.window[i]) * bound_samples[line.strip[i], trace]
                        for i in eachindex(line.strip)
                )
                energy = abs2(value) * BigFloat(line.spacing) / count * line.measure
                strip[index, 2] += energy
                line.inward[1] <= bin <= line.inward[2] && (strip[index, 1] += energy)
                strip_scale += scale^2 * BigFloat(line.spacing) / count * line.measure
            end
        end
        if line.onset !== nothing
            node, before, after = line.onset
            for trace in axes(samples, 2)
                slope = (samples[after, trace] - samples[before, trace]) / line.onset_step
                flux = imag(conj(samples[node, trace]) * slope) / prepared.policy.k_min
                current += max(flux, 0) * line.measure
                current_scale += bound_samples[node, trace] *
                    (bound_samples[after, trace] + bound_samples[before, trace]) /
                    line.onset_step / prepared.policy.k_min * line.measure
            end
        end
    end
    # Assume the finite binary64 FFT/twiddle error model independently exercised
    # in transforms.jl, with no overflow or underflow. Butterfly counts alone do
    # not prove FFTW's twiddle-generation accuracy. Under that hypothesis these
    # fixed fixtures have <=48 spatial columns, 16 strip bins, 40 traces and four
    # lines; gamma_8192 covers FFT, norm, dot-product and chunk-accumulation depths.
    # Absolute envelopes precede both cancellations; rectification is 1-Lipschitz.
    gamma = BigFloat(8192) * eps(Float64) / (1 - BigFloat(8192) * eps(Float64))
    positive_scale = maximum(transpose(BigFloat.(prepared.tables.totals)) * node_scale)
    scales = (
        physical = maximum(envelope), columns = maximum(columns),
        nodes = max(positive_scale, maximum(node_scale)),
        strip = strip_scale, current = current_scale,
    )
    return (; physical, faces, clip, totals, columns, strip, current, gamma, scales)
end

function transverse_reading_allocations(work, field)
    SolpropJulia._transverse_form_reading!(
        work, field; faces = true, full = true, current = true,
    )
    return @allocated SolpropJulia._transverse_form_reading!(
        work, field; faces = true, full = true, current = true,
    )
end

function transverse_endpoint_allocations(work, field, notices)
    first_z, second_z = work.committed_z + 0.1, work.committed_z + 0.2
    SolpropJulia.transverse_monitor_endpoint!(work, field, first_z, 0.1, notices)
    return @allocated SolpropJulia.transverse_monitor_endpoint!(
        work, field, second_z, 0.1, notices,
    )
end

@testset "Scalar spatial transverse readings and bounded trace batches" begin
    for tasks in (1, 2), route in (:radial, :cartesian, :line), collar in (false, true)
        execution, prepared, field = transverse_observation_case(route; tasks, collar)
        original = copy(field)
        reference = transverse_observation_reference(prepared, field)
        for chunk in (1, 3, 100)
            work = SolpropJulia.TransverseWorkspace(
                execution, prepared; trace_chunk = chunk,
            )
            fill!(work.band, complex(NaN, NaN))
            for line in work.lines
                fill!(line.spectrum, complex(NaN, NaN))
                fill!(line.current, complex(NaN, NaN))
            end
            SolpropJulia._transverse_form_reading!(
                work, field; faces = true, full = true, current = collar,
            )
            reading = work.reading
            @test field == original
            @test maximum(abs, Complex{BigFloat}.(work.band) - reference.physical) <=
                reference.gamma * reference.scales.physical
            for name in (:faces, :clip, :totals)
                @test maximum(
                    abs, BigFloat.(collect(getproperty(reading, name))) -
                        getproperty(reference, name)
                ) <= reference.gamma * reference.scales.nodes
            end
            @test maximum(abs, BigFloat.(reading.columns) - reference.columns) <=
                reference.gamma * reference.scales.columns
            if collar
                @test maximum(abs, BigFloat.(reading.strip) - reference.strip) <=
                    reference.gamma * reference.scales.strip
                @test abs(BigFloat(reading.current) - reference.current) <=
                    reference.gamma * reference.scales.current
            else
                @test iszero(reading.current) && isempty(reading.strip)
            end
            @test reading.has_faces && reading.full && reading.has_current == collar
            @test reading.has_strip == collar
            @test transverse_reading_allocations(work, field) == 0
            if collar
                SolpropJulia._transverse_form_reading!(
                    work, field; faces = false, current = true,
                )
                @test !reading.has_faces && !reading.has_strip && reading.has_current
                @test abs(BigFloat(reading.current) - reference.current) <=
                    reference.gamma * reference.scales.current
                @test field == original
            end
            for line in work.lines
                @test all(isfinite, line.spectrum) && all(isfinite, line.current)
            end
        end
    end
end

@testset "Transverse launch, committed current, plane reads and continuation" begin
    for tasks in (1, 2), route in (:radial, :cartesian, :line)
        execution, prepared, field = transverse_observation_case(route; tasks)
        work = SolpropJulia.TransverseWorkspace(execution, prepared; trace_chunk = 3)
        notices = SolpropJulia.MonitorNotices(1)
        initial_record = SolpropJulia.transverse_monitor_record(work)
        @test isbitstype(typeof(initial_record))
        @test isnan(initial_record.continuation.launch)
        SolpropJulia.transverse_monitor_launch!(work, field, 0.0, notices)
        @test notices.count == 0 && work.started
        @test SolpropJulia.transverse_monitor_ceiling(work) == 0.05
        @test work.state.reads == 0 && iszero(work.state.current)
        SolpropJulia.transverse_monitor_endpoint!(work, field, 0.01, 0.01, notices)
        @test work.state.reads == 0 && work.state.has_current_last
        @test iszero(work.state.current)
        SolpropJulia.transverse_monitor_endpoint!(work, field, 0.02, 0.01, notices)
        current = work.state.current
        @test current >= 0 && work.state.reads == 0
        SolpropJulia.transverse_monitor_plane!(work, field, 0.015, notices)
        @test work.state.reads == 1 && work.state.last_z == 0.015
        @test work.committed_z == 0.02 && work.state.current == current
        SolpropJulia.transverse_monitor_plane!(work, field, 0.01, notices)
        @test work.state.reads == 1 && work.state.current == current
        SolpropJulia.transverse_monitor_endpoint!(work, field, 0.03, 0.2, notices)
        @test work.state.reads == 2 && work.state.last_z == 0.03
        @test transverse_endpoint_allocations(work, field, notices) == 0
        record = SolpropJulia.transverse_monitor_record(work)
        resumed = SolpropJulia.TransverseWorkspace(
            execution, prepared; trace_chunk = 2, continuation = record.continuation,
        )
        SolpropJulia.transverse_monitor_launch!(
            resumed, 0.2field, work.committed_z, notices,
        )
        continued = SolpropJulia.transverse_monitor_record(resumed)
        @test continued.resumed
        @test continued.continuation.launch == record.continuation.launch
        @test continued.continuation.baseline == record.continuation.baseline
        @test continued.continuation.reads == record.continuation.reads
        @test continued.continuation.current == record.continuation.current
        @test resumed.state.has_current_last
        @test resumed.state.last_columns != work.state.last_columns
        @test SolpropJulia.transverse_monitor_finish(work) == record.finish
        @test_throws ArgumentError SolpropJulia.transverse_monitor_launch!(
            work, field, 0.0, notices,
        )
        @test_throws ArgumentError SolpropJulia.transverse_monitor_endpoint!(
            work, field, work.committed_z, 0.1, notices,
        )
        @test_throws ArgumentError SolpropJulia.transverse_monitor_plane!(
            work, field, work.committed_z + 0.1, notices,
        )
    end
end

@testset "Prepared clean columns retain inactive original noise" begin
    for tasks in (1, 2)
        execution, prepared, field = transverse_observation_case(
            :line; tasks, collar = false,
        )
        noise = 0.03 .* reverse(field; dims = 1)
        active = Bool[false, true, true, false, true]
        old_field, old_noise, old_active = copy(field), copy(noise), copy(active)
        clean = SolpropJulia.transverse_clean_columns(execution, field, noise, active)
        reference = [
            sum(
                abs2(Complex{BigFloat}(field[row, column]))
                    for row in axes(field, 1) if active[row]
            ) +
                sum(
                abs2(Complex{BigFloat}(noise[row, column]))
                    for row in axes(field, 1) if !active[row]
            ) for column in axes(field, 2)
        ]
        # At most five complex squares and five positive additions per column.
        @test maximum(abs, BigFloat.(clean) - reference) <=
            32eps(Float64) * maximum(reference)
        @test field == old_field && noise == old_noise && active == old_active
        @test clean != vec(sum(abs2, field; dims = 1))
        @test_throws ArgumentError SolpropJulia.TransverseWorkspace(
            execution, prepared; trace_chunk = 3, noise_spectrum = noise,
        )
        work = SolpropJulia.TransverseWorkspace(
            execution, prepared; trace_chunk = 3, noise_spectrum = noise,
            clean_columns = clean,
        )
        noisy = field + noise
        noisy[.!active, :] .= 0
        notices = SolpropJulia.MonitorNotices(1)
        SolpropJulia.transverse_monitor_launch!(work, noisy, 0.0, notices)
        @test notices.count == 0 && work.state.has_noise
        @test work.clean_columns == clean && work.clean_columns !== clean
        @test noise == old_noise && field == old_field
        fill!(clean, NaN)
        @test all(isfinite, work.clean_columns)
        @test_throws ArgumentError SolpropJulia.transverse_clean_columns(
            execution, fill(complex(Inf, 0), size(field)), noise, active,
        )
        @test_throws DimensionMismatch SolpropJulia.transverse_clean_columns(
            execution, field, noise, active[1:2],
        )
    end
end

@testset "Later-start clean columns and unused resumed ramp inputs" begin
    for tasks in (1, 2)
        execution, prepared, field = transverse_observation_case(
            :line; tasks, collar = false,
        )
        noise = 0.03 .* reverse(field; dims = 1)
        active = Bool[false, true, true, false, true]
        masked = copy(field)
        masked[.!active, :] .= 0
        for original_noise in (noise, masked .* (1 + 8eps(Float64)))
            old_field, old_noise = copy(masked), copy(original_noise)
            result = SolpropJulia.transverse_fallback_clean_columns(
                execution, masked, original_noise,
            )
            reference = vec(
                sum(
                    abs2, Complex{BigFloat}.(masked) - Complex{BigFloat}.(original_noise);
                    dims = 1,
                )
            )
            envelope = vec(
                sum(
                    (
                        abs.(Complex{BigFloat}.(masked)) +
                            abs.(Complex{BigFloat}.(original_noise))
                    ) .^ 2; dims = 1,
                )
            )
            # Five complex differences, squared norms and positive additions.
            # The envelope precedes subtraction, so cancellation cannot erase its
            # allowance. These finite normal fixtures need no underflow term.
            budget = 32eps(Float64) .* envelope
            @test all(>(0), budget)
            @test all(abs.(BigFloat.(result) - reference) .<= budget)
            @test masked == old_field && original_noise == old_noise
        end
        fresh = SolpropJulia.transverse_clean_columns(execution, field, noise, active)
        fallback = SolpropJulia.transverse_fallback_clean_columns(execution, masked, noise)
        @test fresh != fallback
        @test_throws DimensionMismatch SolpropJulia.transverse_fallback_clean_columns(
            execution, masked, noise[:, 1:2],
        )
        @test_throws ArgumentError SolpropJulia.transverse_fallback_clean_columns(
            execution, fill(ComplexF64(Inf), size(masked)), noise,
        )
        @test_throws ArgumentError SolpropJulia.transverse_fallback_clean_columns(
            execution, fill(ComplexF64(floatmax(Float64)), size(masked)), -masked,
        )
        launch = SolpropJulia.TransverseWorkspace(
            execution, prepared; trace_chunk = 2, noise_spectrum = noise,
            clean_columns = fresh,
        )
        notices = SolpropJulia.MonitorNotices(1)
        SolpropJulia.transverse_monitor_launch!(launch, masked, 0.0, notices)
        @test notices.count == 0
        continuation = SolpropJulia.transverse_continuation(launch.state)
        resumed = SolpropJulia.TransverseWorkspace(
            execution, prepared; trace_chunk = 2, noise_spectrum = noise, continuation,
        )
        @test resumed.clean_columns === nothing && resumed.state.resumed
        SolpropJulia.transverse_monitor_launch!(resumed, masked, 0.5, notices)
        @test notices.count == 0 && resumed.state.launch == continuation.launch
        @test resumed.state.ramp_raw == continuation.ramp_raw
        @test resumed.state.last_z == 0.5 && resumed.state.has_noise
        names = fieldnames(SolpropJulia.TransversePolicy)
        values = NamedTuple{names}(Tuple(getfield(prepared.policy, name) for name in names))
        policy = SolpropJulia.TransversePolicy(; values..., has_design = false)
        no_ramp = SolpropJulia.TransversePreparation(
            execution, prepared.map, policy, prepared.tables;
            n_frequency = prepared.n_frequency, rows = prepared.rows,
            lines = (), declared = false,
        )
        work = SolpropJulia.TransverseWorkspace(
            execution, no_ramp; trace_chunk = 2, noise_spectrum = noise,
        )
        @test work.clean_columns === nothing && work.noise !== nothing
        SolpropJulia.transverse_monitor_launch!(work, masked, 0.0, notices)
        @test notices.count == 0 && work.state.has_noise
    end
end

@testset "Transverse unresolved strips, owner boundaries and early refusals" begin
    execution, prepared, field = transverse_observation_case(:line)
    empty_line = SolpropJulia.TransverseLineData(
        :x, ones(8); strip = Int[], window = Float64[], spacing = 0.0,
        measure = 1.0, inward = (1, 0), onset = nothing, onset_step = 0.0,
    )
    empty_prepared = SolpropJulia.TransversePreparation(
        execution, prepared.map, prepared.policy, prepared.tables;
        n_frequency = 5, rows = fill(true, 8), lines = (empty_line, empty_line),
        declared = false,
    )
    work = SolpropJulia.TransverseWorkspace(execution, empty_prepared; trace_chunk = 2)
    SolpropJulia._transverse_form_reading!(work, field; faces = true, current = true)
    @test all(iszero, work.reading.strip) && iszero(work.reading.current)
    @test all(line -> line.plan === nothing && line.onset === nothing, work.lines)
    @test transverse_reading_allocations(work, field) == 0
    notices = SolpropJulia.MonitorNotices(1)
    before = copy(work.band)
    @test_throws ArgumentError SolpropJulia.transverse_monitor_launch!(
        work, work.band, 0.0, notices,
    )
    @test work.band == before && !work.started
    GC.@preserve work begin
        foreign = unsafe_wrap(Array, pointer(work.band), size(work.band); own = false)
        @test_throws ArgumentError SolpropJulia.transverse_monitor_launch!(
            work, foreign, 0.0, notices,
        )
    end
    @test work.band == before && !work.started
    @test_throws ArgumentError SolpropJulia.transverse_monitor_launch!(
        work, fill(complex(NaN, 0), size(field)), 0.0, notices,
    )
    @test_throws ArgumentError SolpropJulia.TransverseWorkspace(
        execution, prepared; trace_chunk = 0,
    )
    @test_throws ArgumentError SolpropJulia.TransversePreparation(
        execution, prepared.map, prepared.policy, prepared.tables;
        n_frequency = 5, rows = fill(false, 8), lines = prepared.lines, declared = false,
    )
    @test_throws ArgumentError SolpropJulia.TransverseLineData(
        :x, ones(8); strip = Int[], window = Float64[], spacing = 0.1,
        measure = 1.0, inward = (1, 0), onset = nothing, onset_step = 0.0,
    )
    @test_throws ArgumentError SolpropJulia.TransverseLineData(
        :x, ones(8); strip = [1, 1, 2, 3], window = ones(4), spacing = 0.1,
        measure = 1.0, inward = (1, 0), onset = nothing, onset_step = 0.0,
    )

    complete = SolpropJulia.TransverseWorkspace(execution, prepared; trace_chunk = 100)
    retained = complete.lines[1].spectrum
    @test length(retained) > length(field)
    GC.@preserve retained begin
        foreign = unsafe_wrap(Array, pointer(retained, 2), size(field); own = false)
        @test_throws ArgumentError SolpropJulia.transverse_monitor_launch!(
            complete, foreign, 0.0, notices,
        )
    end
    @test !complete.started && all(iszero, complete.band)
    @test empty_prepared.lines[1].sample_scale !== empty_line.sample_scale
    fill!(empty_line.sample_scale, 17.0)
    @test all(isone, empty_prepared.lines[1].sample_scale)
end

@testset "Transverse workspace warnings retain readings rather than fields" begin
    execution, prepared, field = transverse_observation_case(:line)
    work = SolpropJulia.TransverseWorkspace(execution, prepared; trace_chunk = 3)
    notices = SolpropJulia.MonitorNotices(1)
    original = copy(field)
    SolpropJulia.transverse_monitor_launch!(
        work, field, 0.0, notices; fixed_step_max = 0.5,
    )
    @test notices.count == 1
    @test notices.records[1].code == SolpropJulia.TransverseCoarseWarning
    @test work.state.stage != 0 && isnan(work.state.last_rate_z)
    @test_throws ArgumentError SolpropJulia.transverse_monitor_proceed!(work, notices)
    @test_throws ArgumentError SolpropJulia.transverse_monitor_endpoint!(
        work, field, 0.01, 0.1, notices,
    )
    @test work.committed_z == 0.0 && work.state.reads == 0
    fill!(field, complex(NaN, NaN))
    SolpropJulia.reset_notices!(notices)
    SolpropJulia.transverse_monitor_proceed!(work, notices)
    @test notices.count == 0 && work.state.stage == 0
    @test work.state.last_rate_z == 0.0
    copyto!(field, original)
    SolpropJulia.transverse_monitor_endpoint!(work, field, 0.01, 0.1, notices)
    @test work.committed_z == 0.01 && work.state.reads == 1

    dark = SolpropJulia.TransverseWorkspace(execution, prepared; trace_chunk = 3)
    fill!(field, 0)
    SolpropJulia.transverse_monitor_launch!(dark, field, 0.0, notices)
    SolpropJulia.transverse_monitor_endpoint!(dark, field, 0.1, 0.1, notices)
    SolpropJulia.transverse_monitor_plane!(dark, field, 0.05, notices)
    @test notices.count == 0 && iszero(dark.state.launch) && dark.state.reads == 0
    @test iszero(dark.state.current) && isinf(SolpropJulia.transverse_monitor_ceiling(dark))
end
