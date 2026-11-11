# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

function reconstruction_grid(kind)
    time = collect((-4:3) ./ 8)
    frequency = kind == :field ? 5 : 8
    parseval = kind == :field ? [1.0, 4.0, 4.0, 4.0, 1.0] ./ 64 :
        fill(1 / 64, frequency)
    windows = fill(false, frequency, 2)
    windows[1:3, 1] .= true
    windows[3:end, 2] .= true
    return SolpropJulia.StatisticsGrid(
        collect(0.0:(frequency - 1)), time, parseval; windows,
        spectral_density = ones(frequency),
    )
end

function reconstruction_weight(execution, kind, varying)
    frequency = kind == :field ? 5 : 8
    entries = [1 + row / 16 + column / 8 for row in 1:frequency, column in 1:2]
    varying || return SolpropJulia.ObservationWeight(execution, entries)
    coefficients = reshape(vcat(vec(entries), vec(entries ./ 4)), 2frequency, 2, 1)
    return SolpropJulia.ObservationWeight(
        SolpropJulia.ChebyshevTable(execution, [0.0, 1.0], coefficients),
        (frequency, 2),
    )
end

function reconstruction_selection()
    names = (
        :energy, :energy_window, :omega0, :peak_power, :peak_power_total,
        :peak_power_window, :peak_intensity, :density,
    )
    counts = (2, 4, 2, 2, 1, 4, 1, 1)
    offset = 1
    slots = map(names, counts) do name, count
        slot = SolpropJulia.StatisticSlot(name, offset, count)
        offset += count
        slot
    end
    return SolpropJulia.StatisticsSelection(slots, (1, 4, 8))
end

function reconstruction_modal(
        execution, kind, arithmetic; varying = true, temporal = nothing,
    )
    grid = reconstruction_grid(kind)
    taper = SolpropJulia.SourceProfile(
        SolpropJulia.ChebyshevTable(
            execution, [0.0, 1.0], reshape([1.125, 0.125], 1, 2, 1)
        )
    )
    prepared = SolpropJulia.ObservationPreparation(
        execution, grid, reconstruction_selection();
        kind, n_modes = 2, power_weight = reconstruction_weight(execution, kind, varying),
        axis_synthesis = [0.5 -0.25; 0.25 0.5], taper, intensity_factor = 0.5,
        profiles = SolpropJulia.ObservationProfiles(
            density = SolpropJulia.SourceProfile(3.0),
        ),
        arithmetic, temporal,
    )
    return SolpropJulia.ObservationWorkspace(execution, prepared; node_chunk = 2)
end

function reconstruction_field(kind)
    frequency = kind == :field ? 5 : 8
    field = ComplexF64[
        (row + 2column) / 8 + im * (row - column) / 16
            for row in 1:frequency, column in 1:2
    ]
    if kind == :field
        field[1, :] .= real.(field[1, :])
        field[end, :] .= real.(field[end, :])
    end
    return field
end

function reconstruction_dft(spectrum, kind)
    return setprecision(BigFloat, 256) do
        complete = zeros(Complex{BigFloat}, 8, 2)
        complete[1:size(spectrum, 1), :] .= spectrum
        kind == :field && (complete[2:4, :] .*= 2)
        return [
            sum(complete[k + 1, column] * cis(2big(pi) * k * row / 8) for k in 0:7) / 8
                for row in 0:7, column in 1:2
        ]
    end
end

function reconstruction_allocated(work, field, position)
    SolpropJulia.observe!(work, field, position)
    return @allocated SolpropJulia.observe!(work, field, position)
end

@testset "Modal reconstruction preserves physical state and field powers" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope), varying in (false, true)
        physical = reconstruction_field(kind)
        reference = reconstruction_modal(
            execution, kind, SolpropJulia.SourceArithmetic(Float64);
            varying
        )
        for (state_scale, field_scale) in ((0, 0), (40, -20), (-50, 30))
            arithmetic = SolpropJulia.SourceArithmetic(
                Float32; state_exponent = state_scale, field_exponent = field_scale
            )
            work = reconstruction_modal(execution, kind, arithmetic; varying)
            stored = ComplexF32.(ldexp.(real.(physical), -state_scale)) .+
                im .* ComplexF32.(ldexp.(imag.(physical), -state_scale))
            before = copy(stored)
            @test work.data.arithmetic == arithmetic
            @test eltype(work.spectrum) === ComplexF32 && eltype(work.weight) === Float32
            @test eltype(work.pair.samples) === ComplexF32
            @test eltype(work.statistics.power) === Float32
            @test eltype(work.values) === Float64
            @test work.data.host_axis_synthesis == reference.data.axis_synthesis
            @test work.statistics.scales.nodes in work.buffers
            for position in (0.25, 0.75, 0.25)
                SolpropJulia.observe!(reference, physical, position)
                SolpropJulia.observe!(work, stored, position)
                @test work.valid && work.z == position && stored == before
                # Conditional FFT model: each eight-point transform has component
                # error <=gamma32*sum(abs(input))/8. The two paired transforms,
                # two-term axis maps, coefficient rounding, products and reductions
                # fit gamma512 times this positive fixture's absolute-expression
                # scale. Its centroid denominator is >=1, so no small divisor is
                # hidden in that count. The independent 256-bit DFT below checks
                # the actual transform instance; operation counting alone does not
                # prove a vendor's twiddle generation.
                u = Float64(eps(Float32)) / 2
                gamma = 512u / (1 - 512u)
                magnitude = max(1.0, maximum(sum(abs.(physical); dims = 1)))^2 *
                    max(1.0, maximum(reference.weight))
                bound = gamma * magnitude
                @test bound > 0
                @test all(abs.(work.values .- reference.values) .<= bound)
                analytic = reconstruction_dft(physical, kind)
                restored = ComplexF64.(work.pair.samples) .* exp2(state_scale)
                @test all(abs.(restored .- analytic) .<= bound)
                plane = copy(work.values)
                SolpropJulia.observe!(work, stored, position; periodic = true)
                for (index, slot) in pairs(work.data.selection.slots)
                    part = slot.offset:(slot.offset + slot.count - 1)
                    if index in work.data.selection.periodic
                        @test all(abs.(work.values[part] .- plane[part]) .<= bound)
                    else
                        @test all(iszero, work.values[part])
                    end
                end
            end
            @test reconstruction_allocated(work, stored, 0.25) == 0
            fill!(work.pair.samples, ComplexF32(NaN, NaN))
            fill!(work.windows.pair.spectrum, ComplexF32(NaN, NaN))
            fill!(work.axis[1], ComplexF32(NaN, NaN))
            SolpropJulia.observe!(work, zero(stored), 0.25)
            @test all(iszero, work.pair.samples) && all(iszero, work.windows.samples)
            @test all(iszero, work.axis[1])
            @test_throws ArgumentError SolpropJulia.ObservationWorkspace(
                execution, work.data; node_chunk = 1, noise_spectrum = physical
            )
        end
    end
end

# The aggregate includes observations.jl's independent BigFloat DFT and prepared
# population fixtures before this file. All new numerical work still belongs to
# the actual observation, not to a preceding source evaluation.
function reconstruction_population_reference(work, field, position)
    data = work.data.populations
    fine = data.grid.kind == :field ? 9 : 16
    padded = zeros(ComplexF64, fine, 2)
    padded[data.grid.crop_index .+ 1, :] .= field
    modal = observation_inverse(padded, 16; field = data.grid.kind == :field)
    nodes = modal * transpose(data.map.host_synthesis) .* (0.7 + 0.2position)
    fractions, peaks = zeros(3, 3), zeros(3, 3)
    for (channel, species) in enumerate((3, 1)), node in 1:3
        rate = (data.grid.kind == :field ? 0.4channel : 0.2channel) .*
            abs2.(nodes[:, node])
        exposure = 0.0625 * (sum(rate[2:15]) + (rate[1] + rate[16]) / 2)
        initial = 0.1channel + 0.02position
        fractions[node, species] = initial + (1 - initial) * -expm1(-exposure)
        peaks[node, species] = maximum(rate)
    end
    density = [3.0, 4.0 + position, 2.0]
    electrons = fractions .* transpose(density)
    totals = vec(sum(electrons; dims = 2))
    return vcat(
        vec(fractions), electrons[argmax(totals), :], maximum(totals), maximum(peaks),
    )
end

function reconstruction_population_bound(work, field, position)
    data = work.data.populations
    u = Float64(eps(Float32)) / 2
    gamma = count -> count * u / (1 - count * u)
    # Qualified transform/twiddle model gamma_(64n), followed by a two-mode
    # contraction and represented taper. The reference starts from exactly the
    # dyadic physical field represented by the supplied stored field.
    modal = 2 .* vec(sum(abs, field; dims = 1)) ./ 16
    amplitude = maximum(abs.(data.map.host_synthesis) * modal) * (0.7 + 0.2position)
    field_error = 2gamma(64 * 16 + 32) * amplitude
    # The quadratic log spline has |log(E)|<=28 on the declared [1e-12,100]
    # domain. Its translated log origin also carries ef*log(2). Under the
    # separately qualified four-ulp elementary-function model, gamma below
    # covers those operations, coefficient conversion and the cycle moments.
    # The omitted rate below the lower cutoff is bounded absolutely by 0.8e-24.
    rate_error = 0.8 * (2amplitude * field_error + field_error^2) +
        gamma(512 + 4abs(data.grid.arithmetic.field_exponent)) *
        0.8 * (amplitude + field_error)^2 + 0.8e-24
    exposure = 0.8amplitude^2
    exposure_error = rate_error + gamma(64) * (exposure + rate_error)
    # Nonnegative exposure gives a population derivative <=1. The two active
    # species densities sum to5; the passive column remains exactly zero.
    fraction_error = exposure_error + gamma(32) * (1 + exposure)
    return max(rate_error, 5fraction_error + 5gamma(16))
end

@testset "Fresh scaled populations use the physical nodal exponent" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope), (state_scale, field_scale) in ((30, -10), (-30, 20))
        arithmetic = SolpropJulia.SourceArithmetic(
            Float32; state_exponent = state_scale, field_exponent = field_scale
        )
        work = observation_population_case(
            execution, kind; arithmetic, moment_rtol = 2.0e-6
        )
        physical = reconstruction_field(kind)
        stored = ComplexF32.(physical .* exp2(-state_scale))
        @test eltype(work.populations.nodes) === (kind == :field ? Float32 : ComplexF32)
        @test eltype(work.populations.fractions) === Float32
        @test eltype(work.populations.densities) === Float64
        @test work.statistics.scales.nodes in work.buffers
        for position in (0.0, 0.25, 0.75)
            SolpropJulia.observe!(work, stored, position)
            reference = reconstruction_population_reference(work, physical, position)
            bound = reconstruction_population_bound(work, physical, position)
            @test bound > 0
            @test all(abs.(work.values .- reference) .<= bound)
            @test all(iszero, work.populations.fractions[:, 2])
            @test all(iszero, work.populations.peaks[:, 2])
        end
        @test reconstruction_allocated(work, stored, 0.25) == 0
        before = copy(stored)
        invalid = copy(stored)
        invalid[1] = ComplexF32(NaN, NaN)
        @test_throws ArgumentError SolpropJulia.observe!(work, invalid, 0.25)
        @test stored == before
    end
end

function reconstruction_temporal(execution, kind, arithmetic)
    grid = reconstruction_grid(kind)
    regions = zeros(8, 4)
    regions[1:2, 1] .= 0.25
    regions[7:8, 2] .= 0.5
    regions[:, 3] .= 0.125
    regions[:, 4] .= collect(1:8) ./ 16
    policy = SolpropJulia.TemporalMonitorPolicy(;
        warn = 0.8, stop = 0.9, activity = 1.0, quiet_share = 0.1,
        quiet_checks = 2, quiet_collar = true, cadence = 3, interval = Inf,
        launch_floor_ratio = 1.0e-6, floor_margin = 1.2, split = true,
        crossing_share = 0.1, bucket_limits = [0.1, 0.4, Inf],
    )
    buckets = [mod(row + column, 2) + 1 for row in eachindex(grid.omega), column in 1:2]
    temporal = SolpropJulia.TemporalObservationData(
        policy, SolpropJulia.TemporalMonitorWeights(regions, [0.5, 1.25]; buckets)
    )
    return reconstruction_modal(execution, kind, arithmetic; temporal)
end

function reconstruction_monitor_allocated(work, field, notices)
    SolpropJulia.reset_notices!(notices)
    SolpropJulia._observation_read!(work, field)
    return @allocated SolpropJulia._observation_read!(work, field)
end

@testset "Scaled temporal monitor readings remain physical binary64 records" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope)
        physical = reconstruction_field(kind)
        for state_scale in (-60, 60)
            reference = reconstruction_temporal(
                execution, kind, SolpropJulia.SourceArithmetic(Float64)
            )
            arithmetic = SolpropJulia.SourceArithmetic(
                Float32; state_exponent = state_scale,
            )
            work = reconstruction_temporal(execution, kind, arithmetic)
            stored = ComplexF32.(physical .* exp2(-state_scale))
            notices = SolpropJulia.MonitorNotices(2)
            SolpropJulia._observation_read!(reference, physical)
            SolpropJulia._observation_read!(work, stored)
            first, second = work.monitor_work.reading, reference.monitor_work.reading
            # The eight-point transform model above and positive weighted
            # products/reductions give gamma512 times the absolute energy scale.
            u = Float64(eps(Float32)) / 2
            bound = (512u / (1 - 512u)) * max(1.0, second.total)
            @test bound > 0
            @test abs(first.total - second.total) <= bound
            @test all(abs.(collect(first.regions) .- collect(second.regions)) .<= bound)
            @test all(abs.(first.occupancy .- second.occupancy) .<= bound)
            @test eltype(work.monitor_work.power) === Float32
            @test first.total isa Float64 && eltype(first.occupancy) === Float64
            SolpropJulia.observation_launch!(reference, physical, 0.0, notices)
            SolpropJulia.reset_notices!(notices)
            SolpropJulia.observation_launch!(work, stored, 0.0, notices)
            SolpropJulia.capture_split!(reference, physical, 0.25, 0.0625)
            SolpropJulia.capture_split!(work, stored, 0.25, 0.0625)
            @test abs(
                work.monitor_work.pending.removed - reference.monitor_work.pending.removed
            ) <= bound
            @test reconstruction_monitor_allocated(work, stored, notices) == 0
        end
    end
end

function reconstruction_axis_bound(work, field, matrix, position)
    data = work.data.axis_populations
    u = Float64(eps(Float32)) / 2
    gamma = count -> count * u / (1 - count * u)
    modal = 2 .* vec(sum(abs, field; dims = 1)) ./ 16
    amplitude = sum(abs.(matrix) * modal) * (0.7 + 0.2position)
    # Same independently qualified length16 FFT and elementary-function model as
    # the scalar population witness, with the vector norm's1-Lipschitz error and
    # two component contractions included before the rate/exposure map.
    field_error = 2gamma(64 * 16 + 64) * amplitude
    rate_error = 0.8 * (2amplitude * field_error + field_error^2) +
        gamma(512 + 4abs(data.grid.arithmetic.field_exponent)) *
        0.8 * (amplitude + field_error)^2 + 0.8e-24
    exposure = 0.8amplitude^2
    fraction_error = rate_error + gamma(64) * (exposure + rate_error) +
        gamma(32) * (1 + exposure)
    return max(rate_error, 5fraction_error + 5gamma(16))
end

@testset "Fine on-axis populations use physical amplitudes and independent maps" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    for kind in (:field, :envelope), (es, ef) in ((30, -10), (-30, 20))
        arithmetic = SolpropJulia.SourceArithmetic(
            Float32; state_exponent = es, field_exponent = ef,
        )
        matrix = kind == :field ? [1.0 -0.5; 0.25 0.75] : [1.0 -0.5; 0.5 -0.25]
        work = axis_statistics_case(
            execution, kind, matrix; arithmetic, moment_rtol = 2.0e-6,
        )
        physical = reconstruction_field(kind)
        field = ComplexF32.(physical .* exp2(-es))
        @test eltype(work.axis_populations.nodes) ===
            (kind == :field ? Float32 : ComplexF32)
        @test eltype(work.axis_populations.fractions) === Float32
        @test eltype(work.data.axis_populations.host_synthesis) === Float64
        for position in (0.0, 0.25, 0.7)
            SolpropJulia.observe!(work, field, position)
            reference = axis_statistics_reference(work, physical, matrix, position)
            bound = reconstruction_axis_bound(work, physical, matrix, position)
            @test bound > 0
            @test all(abs.(work.values[3:10] .- reference) .<= bound)
            @test work.values[6] == 0 && work.values[9] == 0
        end
        @test axis_statistics_allocated(work, field) == 0
    end
end
