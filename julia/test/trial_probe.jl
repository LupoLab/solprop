# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

using Test: @test, @testset, @test_throws
import SolpropJulia

mutable struct TrialProbeTestSource{E}
    execution::E
    rate::ComplexF64
    forcing::ComplexF64
    calls::Int
    fail_at::Int
end

function (source::TrialProbeTestSource)(output, field, position)
    source.calls += 1
    source.calls == source.fail_at && error("deliberate timing-probe source failure")
    rate, forcing = source.rate, source.forcing
    SolpropJulia.each!(source.execution, size(field)...) do row, column
        output[row, column] = rate * field[row, column] + forcing
    end
    return nothing
end

trial_probe_execution(tasks = Threads.nthreads(:default)) =
    SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), tasks, true)

function trial_probe_midpoint()
    return SolpropJulia.EmbeddedTableau(
        [0.0, 0.5], [0.0 0.0; 0.5 0.0], [0.0, 1.0], [-1.0, 1.0],
    )
end

function trial_probe_fixture(;
        low_storage = false, z = 0.25, tasks = Threads.nthreads(:default),
        initial = ComplexF64[row / 8 + im * column / 16 for row in 1:5, column in 1:2],
        active = fill(true, size(initial, 1)), noise = false, resume = false,
        forcing = 0.0 + 0.0im, fail_at = 0, split = nothing,
    )
    execution = trial_probe_execution(tasks)
    operator = SolpropJulia.DiagonalOperator(
        execution, fill(-0.125 + 0.25im, size(initial)),
    )
    flow = SolpropJulia.LinearFlow(execution, operator, initial)
    norm = SolpropJulia.MarchNorm(execution, active, active, ones(length(active)), 0.1, 0.0)
    schedule = SolpropJulia.MarchSchedule(;
        planes = [0.5, 1.0], breakpoints = [0.0, 1.0], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0,
        replay_steps = [0.25, 0.125, 0.125, 0.5],
        replay_edges = [0.0, 0.25, 0.375, 0.5, 1.0],
    )
    source = TrialProbeTestSource(execution, -0.25 + 0.125im, forcing, 0, fail_at)
    tableau = low_storage ? SolpropJulia.RK4IPTableau() : trial_probe_midpoint()
    draw = noise ? SolpropJulia.InputNoise(
            execution, fill(0.125 - 0.0625im, size(initial)), 0.25, operator,
        ) : nothing
    probe = SolpropJulia.TrialProbe(
        execution, initial, z, 1 / 64, tableau, norm, schedule, source, flow;
        split, noise = draw, resume,
    )
    return probe, initial, norm, schedule
end

function trial_probe_reference(initial, z, h; low_storage = false, repeats = 1)
    return setprecision(256) do
        rate, linear = Complex{BigFloat}(-0.25 + 0.125im),
            Complex{BigFloat}(-0.125 + 0.25im)
        step = BigFloat(h)
        factor = low_storage ? sum((rate * step)^k / factorial(k) for k in 0:4) :
            1 + rate * step + (rate * step)^2 / 2
        Complex{BigFloat}.(initial) .* (exp(linear * step) * factor)^repeats
    end
end

function trial_probe_bound(initial, h; repeats = 1)
    # For these commuting diagonal fixtures, every coordinate/product of a binary
    # power-of-two step and coefficient is exact. One midpoint or four-stage RK4
    # picture uses fewer than 256 real rounded arithmetic operations along a path,
    # including complex products, sums, reciprocal-flow factors and final scaling.
    # We assume the qualified Float64 complex exponential has <=8u relative error
    # on these small arguments; 512 units per trial includes those factors. The
    # absolute polynomial majorant also bounds cancellation and repeated-trial
    # sensitivity. This is not a bound for arbitrary nonlinear probes or libm.
    u = eps(Float64) / 2
    gamma = (512repeats * u) / (1 - 512repeats * u)
    majorant = maximum(abs, initial) *
        exp(repeats * h * (abs(-0.125 + 0.25im) + abs(-0.25 + 0.125im)))
    return gamma * majorant
end

function trial_probe_cycle!(probe, output)
    SolpropJulia.trial_probe!(probe)
    SolpropJulia.copy_trial_probe_field!(output, probe)
    return nothing
end

function trial_probe_allocations(probe, output)
    trial_probe_cycle!(probe, output)
    return @allocated trial_probe_cycle!(probe, output)
end

function trial_probe_low_allocations(probe)
    SolpropJulia.trial_probe!(probe)
    return @allocated SolpropJulia.trial_probe!(probe)
end

@testset "Owned embedded timing trials and single field consumption" begin
    for tasks in (1, 2)
        probe, initial, norm, schedule = trial_probe_fixture(; tasks)
        original = copy(initial)
        loaded = copy(probe.field)
        output = similar(initial)
        @test probe.field == original
        @test !Base.mightalias(probe.field, initial)
        @test !Base.mightalias(probe.norm.active, norm.active)
        @test !Base.mightalias(probe.norm.weights, norm.weights)
        @test_throws ArgumentError SolpropJulia.copy_trial_probe_field!(output, probe)
        for (z, h) in ((0.25, 1 / 64), (0.5, 1 / 32), (0.25, 1 / 128))
            @test SolpropJulia.trial_probe!(probe, z, h) === nothing
            @test probe.state.ready
            @test isfinite(probe.state.error_norm) && probe.state.error_norm >= 0
            @test SolpropJulia.copy_trial_probe_field!(output, probe) === nothing
            reference = trial_probe_reference(original, z, h)
            bound = trial_probe_bound(original, h)
            @test bound > 0 && maximum(abs, output - reference) <= bound
            @test !probe.state.ready
            @test_throws ArgumentError SolpropJulia.copy_trial_probe_field!(output, probe)
            @test probe.field == loaded && initial == original
            @test probe.state.loaded_z == 0.25 && probe.state.default_dz == 1 / 64
        end
        saved = copy(output)
        SolpropJulia.trial_probe!(probe, 0.5, 1 / 32)
        SolpropJulia.trial_probe!(probe)
        @test output == saved
        SolpropJulia.copy_trial_probe_field!(output, probe)
        reference = trial_probe_reference(original, 0.25, 1 / 64)
        @test maximum(abs, output - reference) <= trial_probe_bound(original, 1 / 64)
        status = SolpropJulia.trial_probe_status(probe)
        @test status.source_evaluations == probe.source.calls == 10
        @test status.norm_evaluations == status.trials == 5
        @test !status.low_storage && !status.in_flight && !status.failed
        @test !hasproperty(probe, :schedule) && !hasproperty(probe, :services)
        @test !hasproperty(probe, :history) && !hasproperty(probe, :attempts)
        @test schedule.replay_steps == [0.25, 0.125, 0.125, 0.5]
        trial_probe_allocations(probe, output)
        @test trial_probe_allocations(probe, output) == 0
        @test SolpropJulia.close_trial_probe!(probe) === nothing
        @test SolpropJulia.close_trial_probe!(probe) === nothing
        @test SolpropJulia.trial_probe_status(probe).closed
        @test_throws ArgumentError SolpropJulia.trial_probe!(probe)
        @test_throws ArgumentError SolpropJulia.copy_trial_probe_field!(output, probe)
    end
end

@testset "Probe input draw is applied once under the shared launch predicate" begin
    active = Bool[true, false, true, true, false]
    for resume in (false, true), z in (0.25, 0.5)
        probe, initial, _, _ = trial_probe_fixture(; noise = true, resume, z, active)
        original = copy(initial)
        expected = copy(initial)
        !resume && z == 0.25 && (expected .+= 0.125 - 0.0625im)
        expected[.!active, :] .= 0
        # All operands in this fixture are dyadic and exactly representable.
        @test probe.field == expected
        @test probe.noise.spectrum == fill(0.125 - 0.0625im, size(initial))
        output = similar(initial)
        for _ in 1:2
            trial_probe_cycle!(probe, output)
            @test probe.field == expected && initial == original
            @test all(iszero, output[.!active, :])
            bound = trial_probe_bound(expected, 1 / 64)
            @test maximum(abs, output - trial_probe_reference(expected, z, 1 / 64)) <= bound
        end
    end
end

@testset "Low-storage probes retain discarded registers and the loaded default" begin
    for tasks in (1, 2), (z, expected_step) in ((0.0, 0.25), (0.25, 0.125), (1.0, 0.5))
        probe, initial, _, _ = trial_probe_fixture(; tasks, low_storage = true, z)
        original = copy(initial)
        @test probe.state.default_dz == expected_step
        @test_throws ArgumentError SolpropJulia.copy_trial_probe_field!(
            similar(initial), probe,
        )
        for repeat in 1:2
            SolpropJulia.trial_probe!(probe, 0.5, nothing)
            reference = trial_probe_reference(
                original, 0.5, expected_step; low_storage = true, repeats = repeat,
            )
            bound = trial_probe_bound(original, expected_step; repeats = repeat)
            @test bound > 0 && maximum(abs, probe.field - reference) <= bound
            @test initial == original && probe.state.loaded_z == z
            @test probe.state.source_evaluations == 4repeat
            @test probe.state.norm_evaluations == probe.state.trials == repeat
            @test !probe.state.ready && !probe.state.in_flight
        end
        trial_probe_low_allocations(probe)
        @test trial_probe_low_allocations(probe) == 0
    end
    active = Bool[true, false, true, true, false]
    probe, _, _, _ = trial_probe_fixture(; low_storage = true, active, forcing = 0.25 + 0im)
    @test all(iszero, probe.field[.!active, :])
    SolpropJulia.trial_probe!(probe)
    @test all(!iszero, probe.field[.!active, :])
    @test !probe.state.ready
end

function trial_probe_snapshot(probe)
    return (
        SolpropJulia.trial_probe_status(probe), copy(probe.field),
        copy(probe.work.candidate), copy(probe.work.defect), probe.source.calls,
    )
end

@testset "Invalid trial or output preserves a valid unread trial" begin
    probe, initial, _, _ = trial_probe_fixture()
    SolpropJulia.trial_probe!(probe)
    before = trial_probe_snapshot(probe)
    for (z, h) in (
            (NaN, 0.1), (Inf, 0.1), (0.25, 0.0), (0.25, -1.0),
            (0.25, NaN), (0.25, Inf), (floatmax(Float64), 1.0),
        )
        @test_throws ArgumentError SolpropJulia.trial_probe!(probe, z, h)
        @test isequal(trial_probe_snapshot(probe), before)
    end
    for name in (:source_evaluations, :norm_evaluations, :trials)
        value = getproperty(probe.state, name)
        setproperty!(probe.state, name, typemax(Int))
        saturated = trial_probe_snapshot(probe)
        @test_throws ArgumentError SolpropJulia.trial_probe!(probe)
        @test isequal(trial_probe_snapshot(probe), saturated)
        setproperty!(probe.state, name, value)
    end
    for output in (
            zeros(ComplexF32, size(initial)), zeros(ComplexF64, 1, 1),
            probe.field, probe.work.candidate, probe.flow.exponent,
            view(initial, [1, 1, 3, 4, 5], :),
        )
        saved = copy(output)
        @test_throws Union{ArgumentError, DimensionMismatch} (
            SolpropJulia.copy_trial_probe_field!(output, probe)
        )
        @test isequal(output, saved)
        @test isequal(trial_probe_snapshot(probe), before)
    end
    # The wrapper has a shifted foreign identity but is wholly within retained
    # storage: the shared byte-span boundary, not Base.mightalias, must reject it.
    GC.@preserve probe begin
        owner = probe.work.stages
        foreign = unsafe_wrap(Array, pointer(owner, 2), size(initial); own = false)
        saved = copy(owner)
        @test_throws ArgumentError SolpropJulia.copy_trial_probe_field!(foreign, probe)
        @test owner == saved
        @test isequal(trial_probe_snapshot(probe), before)
    end
    backing = zeros(ComplexF64, 2size(initial, 1), size(initial, 2))
    destination = view(backing, (2size(initial, 1)):-2:2, :)
    SolpropJulia.copy_trial_probe_field!(destination, probe)
    @test all(iszero, backing[1:2:end, :])
    @test maximum(abs, destination - trial_probe_reference(initial, 0.25, 1 / 64)) <=
        trial_probe_bound(initial, 1 / 64)
end

@testset "Numerical probe failures preserve actual attempted work facts" begin
    for low_storage in (false, true), failure in (1, 2)
        probe, initial, _, _ = trial_probe_fixture(; low_storage, fail_at = failure)
        saved = copy(initial)
        @test_throws ErrorException SolpropJulia.trial_probe!(probe)
        status = SolpropJulia.trial_probe_status(probe)
        @test status.source_evaluations == probe.source.calls == failure
        @test status.norm_evaluations == status.trials == 0
        @test status.failed && !status.ready
        @test status.in_flight === low_storage
        @test initial == saved
        @test_throws ArgumentError SolpropJulia.trial_probe!(probe)
        @test_throws ArgumentError SolpropJulia.copy_trial_probe_field!(
            similar(initial), probe,
        )
        @test SolpropJulia.close_trial_probe!(probe) === nothing
    end
end

@testset "Timing-probe preparation refuses before changing caller storage" begin
    probe, initial, norm, schedule = trial_probe_fixture()
    original = copy(initial)
    for z in (-0.1, 0.1, 1.1, NaN, Inf), tableau in
            (trial_probe_midpoint(), SolpropJulia.RK4IPTableau())
        @test_throws ArgumentError SolpropJulia.TrialProbe(
            probe.execution, initial, z, 0.01, tableau, norm, schedule,
            probe.source, probe.flow,
        )
        @test initial == original
    end
    for step in (0.0, -1.0, NaN, Inf)
        @test_throws ArgumentError SolpropJulia.TrialProbe(
            probe.execution, initial, 0.25, step, probe.tableau, norm, schedule,
            probe.source, probe.flow,
        )
    end
    for input in (
            ComplexF32.(initial), fill(complex(NaN), size(initial)),
            probe.flow.exponent,
        )
        saved = copy(input)
        @test_throws ArgumentError SolpropJulia.TrialProbe(
            probe.execution, input, 0.25, 0.01, probe.tableau, norm, schedule,
            probe.source, probe.flow,
        )
        @test isequal(input, saved)
    end
    adaptive = SolpropJulia.MarchSchedule(;
        planes = [1.0], breakpoints = [0.0, 1.0], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0,
    )
    @test_throws ArgumentError SolpropJulia.TrialProbe(
        probe.execution, initial, 0.25, 0.01, SolpropJulia.RK4IPTableau(), norm, adaptive,
        probe.source, probe.flow,
    )
    @test_throws DimensionMismatch SolpropJulia.TrialProbe(
        probe.execution, zeros(ComplexF64, 1, 1), 0.25, 0.01, probe.tableau, norm,
        schedule, probe.source, probe.flow,
    )
end

function trial_probe_material(kind, tasks)
    execution = trial_probe_execution(tasks)
    data, initial = SolpropJulia._precompile_material_case(execution, Val(kind))
    source = SolpropJulia.ModalSource(execution, data; node_chunk = 2)
    operator = SolpropJulia.DiagonalOperator(
        execution, fill(-0.01 + 0.02im, size(initial)),
    )
    flow = SolpropJulia.LinearFlow(execution, operator, initial)
    norm = SolpropJulia.MarchNorm(
        execution, data.grid.active, data.grid.active, ones(size(initial, 1)), 1.0e-3, 0.0,
    )
    schedule = SolpropJulia.MarchSchedule(;
        planes = [0.004], breakpoints = [0.0, 0.01], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0,
    )
    return SolpropJulia.TrialProbe(
        execution, initial, 0.0, 0.001, SolpropJulia._precompile_rk_tableau(),
        norm, schedule, source, flow,
    )
end

@testset "Complete scalar material sources and owned probe memory" begin
    for kind in (:field, :envelope), tasks in (1, 2)
        probe = trial_probe_material(kind, tasks)
        output = similar(probe.field)
        trial_probe_cycle!(probe, output)
        @test all(isfinite, output)
        @test probe.state.source_evaluations == probe.source.calls == 7
        @test probe.state.norm_evaluations == probe.state.trials == 1
        @test all(iszero, output[.!probe.norm.active, :])
        trial_probe_allocations(probe, output)
        @test trial_probe_allocations(probe, output) == 0
        ledger = SolpropJulia.trial_probe_memory(probe)
        @test all(>=(0), values(ledger))
        @test any(name -> occursin("field", String(name)), keys(ledger))
        @test any(name -> occursin("work", String(name)), keys(ledger))
        @test !any(name -> occursin("data", String(name)), keys(ledger))
        @test !any(name -> occursin("operator", String(name)), keys(ledger))
        # Inspect the exact owned array multiset through the shared walker: the
        # probe's sampler repeats references but cannot double-count storage.
        manual, seen = Dict{Symbol, Int}(), IdDict{Any, Bool}()
        SolpropJulia._memory_arrays!(manual, seen, "shared", probe.source.data)
        SolpropJulia._memory_arrays!(manual, seen, "operator", probe.flow.operator)
        empty!(manual)
        for name in (:field, :work, :norm, :source, :flow, :split, :noise)
            SolpropJulia._memory_arrays!(
                manual, seen, String(name), getproperty(probe, name),
            )
        end
        @test sum(values(ledger)) == sum(values(manual))
    end
end

function trial_probe_split_reference(spectrum, rate, active, h, kind)
    return setprecision(256) do
        n = length(rate)
        masked = Complex{BigFloat}.(spectrum)
        masked[.!active, :] .= 0
        full = kind == :field ? vcat(masked, conj.(masked[(end - 1):-1:2, :])) : masked
        samples = [
            sum(full[k + 1, column] * cis(2big(pi) * k * t / n) for k in 0:(n - 1)) / n
                for t in 0:(n - 1), column in axes(spectrum, 2)
        ]
        kind == :field && (samples .= real.(samples))
        samples .*= exp.(-BigFloat(h) .* BigFloat.(rate) ./ 2)
        output = [
            sum(samples[t + 1, column] * cis(-2big(pi) * k * t / n) for t in 0:(n - 1))
                for k in 0:(size(spectrum, 1) - 1), column in axes(spectrum, 2)
        ]
        output[.!active, :] .= 0
        output
    end
end

@testset "Readback applies mask then split then mask exactly once" begin
    for kind in (:field, :envelope)
        execution = trial_probe_execution(2)
        count = kind == :field ? 5 : 8
        rate = collect(1.0:8.0) ./ 8
        split = SolpropJulia.TemporalSplitWorkspace(execution, kind, rate, 2)
        active = fill(true, count)
        active[1] = active[end] = false
        probe, _, _, _ = trial_probe_fixture(;
            initial = ones(ComplexF64, count, 2), active, split,
        )
        SolpropJulia.trial_probe!(probe)
        # Deliberately poison bins which finishing must mask before the real-field
        # inverse. Their complex DC/Nyquist would otherwise be invalid input.
        probe.work.candidate[.!active, :] .= 7 + 3im
        reference = trial_probe_split_reference(
            probe.work.candidate, rate, active, probe.state.trial_dz, kind,
        )
        scale = sum(abs, probe.work.candidate)
        output = similar(probe.field)
        SolpropJulia.copy_trial_probe_field!(output, probe)
        # This n=8 qualification assumes each FFT twiddle/exponential has <=8u
        # absolute/relative error. Fewer than 1024 real operations along a path
        # cover both transforms and damping, with the input one-norm majorant.
        u = eps(Float64) / 2
        bound = 1024u / (1 - 1024u) * scale
        @test bound > 0 && maximum(abs, output - reference) <= bound
        @test all(iszero, output[.!active, :])
        saved = copy(output)
        @test_throws ArgumentError SolpropJulia.copy_trial_probe_field!(output, probe)
        @test output == saved
        trial_probe_allocations(probe, output)
        @test trial_probe_allocations(probe, output) == 0
    end
end

@testset "A probe starts an in-stage collar active without monitor continuation" begin
    execution = trial_probe_execution(2)
    grid = SolpropJulia.ModalSourceGrid(
        execution; kind = :envelope, n_time = 8, dt = 0.125,
        crop_index = collect(0:7), omega = collect(1.0:8.0), active = fill(true, 8),
        omega_window = ones(8), source_window = ones(8),
        source_window_derivative = zeros(8), temporal_rate = fill(0.25, 8),
    )
    data = SolpropJulia.ModalSourceData(
        grid, SolpropJulia.ModalSourceMap(
            execution; synthesis = ones(1, 1), projection = ones(1, 1),
        ), SolpropJulia.SourcePrefactor(execution, zeros(ComplexF64, 8, 1)),
    )
    source = SolpropJulia.ModalSource(execution, data; node_chunk = 1)
    SolpropJulia.set_temporal_source!(source, false)
    @test !source.temporal_active
    initial = ones(ComplexF64, 8, 1)
    operator = SolpropJulia.DiagonalOperator(execution, zeros(ComplexF64, 8, 1))
    flow = SolpropJulia.LinearFlow(
        execution, operator, initial,
    )
    norm = SolpropJulia.MarchNorm(execution, grid.active, grid.active, ones(8), 0.1, 0.0)
    schedule = SolpropJulia.MarchSchedule(;
        planes = [1.0], breakpoints = [0.0, 1.0], hard_kinks = Float64[],
        stretch = 1.0, step_planes = 0,
    )
    probe = SolpropJulia.TrialProbe(
        execution, initial, 0.0, 1 / 64, trial_probe_midpoint(), norm, schedule,
        source, flow;
        resume = true,
    )
    @test source.temporal_active
    SolpropJulia.trial_probe!(probe)
    @test probe.state.source_evaluations == 2
    output = similar(initial)
    SolpropJulia.copy_trial_probe_field!(output, probe)
    @test all(abs.(output) .< abs.(initial))
end

@testset "A nonfinite trial cannot escape through field readback" begin
    probe, initial, _, _ = trial_probe_fixture()
    SolpropJulia.trial_probe!(probe)
    probe.work.candidate[3, 1] = Inf
    output = fill(-9.0 + 0im, size(initial))
    @test_throws SolpropJulia.NonfiniteSourceError (
        SolpropJulia.copy_trial_probe_field!(output, probe)
    )
    @test all(==(-9.0 + 0im), output)
    @test probe.state.failed && !probe.state.ready
    @test_throws ArgumentError SolpropJulia.trial_probe!(probe)
end
