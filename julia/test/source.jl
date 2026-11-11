# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function linear_flow_allocations(flow, output, input)
    flow(output, input, 0.0, 1.0)
    return @allocated flow(output, input, 0.0, 1.0)
end

@testset "Linear flow refuses caller and retained storage overlap before writes" begin
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        rate = zeros(ComplexF64, 4, 1)
        flow = SolpropJulia.LinearFlow(
            execution, SolpropJulia.DiagonalOperator(execution, rate), rate,
        )
        owner = reshape(ComplexF64[1, 2, 3, 4, 5], 5, 1)
        saved = copy(owner)
        GC.@preserve owner begin
            input = unsafe_wrap(Array, pointer(owner), (4, 1); own = false)
            output = unsafe_wrap(Array, pointer(owner, 2), (4, 1); own = false)
            @test !Base.mightalias(input, output)
            @test_throws ArgumentError flow(output, input, 0.0, 1.0)
            @test owner == saved
        end
        output = fill(7.0 + 2.0im, 4, 1)
        for retained in (flow.exponent, flow.operator.rate)
            fill!(retained, 2.0 + 3.0im)
            before = copy(retained)
            @test_throws ArgumentError flow(output, retained, 0.0, 1.0)
            @test retained == before
            @test all(==(7.0 + 2.0im), output)
            @test_throws ArgumentError flow(retained, output, 0.0, 1.0)
            @test retained == before
        end
        fill!(flow.operator.rate, 0)
        input = reshape(ComplexF64[1, 2, 3, 4], 4, 1)
        repeated = view(output, [1, 1, 3, 4], :)
        @test_throws ArgumentError flow(repeated, input, 0.0, 1.0)
        @test all(==(7.0 + 2.0im), output)
        @test linear_flow_allocations(flow, output, input) == 0
        # Zero-generator copies and exact same-array updates are structural contracts.
        @test output == input
        @test linear_flow_allocations(flow, input, input) == 0
        @test input == output
        strided = view(zeros(ComplexF64, 8, 1), 1:2:7, :)
        flow(strided, input, 0.0, 1.0)
        @test strided == input
    end
end

function source_data_fixture(
        execution; synthesis, projection, prefactor, omega_window, source_window,
        kerr, density, carrier = nothing,
        crop_index = collect(0:(size(prefactor, 1) - 1)),
    )
    grid = SolpropJulia.ModalSourceGrid(
        execution; kind = :envelope, n_time = length(source_window), dt = 1.0,
        crop_index, omega = ones(length(crop_index)),
        active = fill(true, length(crop_index)),
        omega_window, source_window,
        source_window_derivative = zeros(length(source_window)),
        carrier,
    )
    map = SolpropJulia.ModalSourceMap(execution; synthesis, projection)
    material = kerr === nothing ? nothing : SolpropJulia.ModalKerr(
            kerr, SolpropJulia.SourceProfile(density),
        )
    return SolpropJulia.ModalSourceData(
        grid, map, SolpropJulia.SourcePrefactor(execution, prefactor); kerr = material,
    )
end

function source_fixture_values(data)
    return (
        synthesis = data.map.synthesis, projection = data.map.projection,
        prefactor = data.prefactor.data, source_window = data.grid.source_window,
        omega_window = data.grid.omega_window, carrier = data.grid.carrier,
        kerr = data.kerr.coefficients, density = data.kerr.density.value,
    )
end

function source_fixture(execution; harmonic = 0.0, kerr = true)
    synthesis = [2.0 -0.5; 0.75 1.5; -1.0 0.25]
    projection = [0.5 0.25 -0.125; -0.25 0.5 0.75]
    prefactor = ComplexF64[-0.3im + 0.01row for row in 1:4, _ in 1:2]
    omega_window = [1.0, 0.8, 0.0, 0.6]
    source_window = [0.2, 0.4, 0.8, 1.0, 1.0, 0.8, 0.4, 0.2]
    carrier = [cis(2pi * row / 8) for row in 0:7]
    coefficients = kerr ? SolpropJulia.KerrCoefficients(
            0.7, :analytic; harmonic_factor = harmonic
        ) : nothing
    data = source_data_fixture(
        execution; synthesis, projection, prefactor, omega_window, source_window,
        kerr = coefficients, density = 1.2, carrier, crop_index = [0, 1, 2, 7],
    )
    return data, (synthesis, projection, prefactor, omega_window, source_window, carrier)
end

function direct_source(data, state, crop_index)
    data = source_fixture_values(data)
    n_time, nodes = length(data.source_window), size(data.synthesis, 1)
    modes = size(state, 2)
    # Independent defining sums, with no FFT, pad/crop or contraction helper.
    samples = zeros(ComplexF64, n_time, nodes)
    for node in 1:nodes, time in 0:(n_time - 1), mode in 1:modes, row in axes(state, 1)
        samples[time + 1, node] += data.synthesis[node, mode] * state[row, mode] *
            cis(2pi * time * crop_index[row] / n_time) / n_time
    end
    polarisation = similar(samples)
    for node in 1:nodes, time in 1:n_time
        sample = samples[time, node]
        polarisation[time, node] = data.kerr.coefficient * data.density *
            data.source_window[time] * (
            (data.kerr.self_factor + data.kerr.conjugate_factor) * abs2(sample) * sample +
                data.kerr.harmonic_factor * data.carrier[time] * sample^3
        )
    end
    result = zeros(ComplexF64, size(state))
    for mode in 1:modes, row in axes(state, 1), node in 1:nodes, time in 0:(n_time - 1)
        result[row, mode] += data.projection[mode, node] * polarisation[time + 1, node] *
            cis(-2pi * time * crop_index[row] / n_time)
    end
    return result .* data.prefactor .* data.omega_window
end

function source_allocations(source, output, state)
    source(output, state, 0.5)
    return @allocated source(output, state, 0.5)
end

function source_rounding_scale(data, state)
    data = source_fixture_values(data)
    n_time = length(data.source_window)
    # Absolute contributions before each cancellation: inverse DFT, modal map,
    # cubic, projection, forward DFT and final translated spectral weights.
    modal_bound = vec(sum(abs, state; dims = 1)) / n_time
    nodal_bound = abs.(data.synthesis) * modal_bound
    factors = abs(data.kerr.self_factor) + abs(data.kerr.conjugate_factor) +
        abs(data.kerr.harmonic_factor) * maximum(abs, data.carrier)
    cubic_bound = abs(data.kerr.coefficient * data.density) * factors * nodal_bound .^ 3
    spectral_bound = abs.(data.projection) * cubic_bound * sum(abs, data.source_window)
    return maximum(
        abs.(data.prefactor) .* abs.(data.omega_window) .* transpose(spectral_bound)
    )
end

@testset "Assembled envelope source retains translated normalization" begin
    state = ComplexF64[
        complex(0.2row - column, 0.3column + row) for row in 1:4, column in 1:2
    ]
    crop_index = [0, 1, 2, 7]
    for tasks in unique((1, Threads.nthreads(:default))), harmonic in (0.0, 0.25)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        data, inputs = source_fixture(execution; harmonic)
        source = SolpropJulia.ModalSource(execution, data; node_chunk = 2)
        expected = direct_source(data, state, crop_index)
        output = fill(ComplexF64(NaN, NaN), size(state))
        original = copy(state)
        @test source(output, state, 0.5) === nothing
        @test source.calls == 1
        # 256 rounding units cover two eight-term DFTs (including phase formation),
        # two short maps, and the cubic. The scale bounds pre-cancellation sums.
        budget = 256eps() * source_rounding_scale(data, state)
        @test budget > 0.0
        @test maximum(abs, output - expected) <= budget
        @test all(iszero, output[3, :])
        @test state == original
        for input in inputs
            fill!(input, 0)
        end
        source(output, state, 0.5)
        @test maximum(abs, output - expected) <= budget
        source_allocations(source, output, state)
        @test source_allocations(source, output, state) == 0
        other = SolpropJulia.ModalSource(execution, data; node_chunk = 2)
        @test source.nodes !== other.nodes
        @test source.polarisation !== other.polarisation
        @test source.modal_spectrum !== other.modal_spectrum
    end
end

@testset "Source failure preserves accepted fields and complete scratch writes" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    data, _ = source_fixture(execution; kerr = false)
    source = SolpropJulia.ModalSource(execution, data; node_chunk = 2)
    state = fill(1.0 + 0.2im, 4, 2)
    output = similar(state)
    fill!(source.polarisation, NaN)
    source(output, state, 0.0)
    @test all(iszero, output)
    @test_throws ArgumentError source(state, state, 0.0)
    @test source.calls == 2
    original = copy(state)
    @test_throws ArgumentError source(output, state, NaN)
    @test state == original
    state[2, 2] = Inf
    @test_throws SolpropJulia.NonfiniteSourceError source(output, state, 0.0)
    @test source.calls == 4
    state[2, 2] = original[2, 2]
    source(output, state, 0.0)
    @test all(iszero, output)
    @test_throws ArgumentError SolpropJulia.ModalSource(execution, data; node_chunk = 0)
end

function source_rk4_tableau()
    nodes = [0.0, 0.5, 0.5, 1.0]
    stages = [0.0 0.0 0.0 0.0; 0.5 0.0 0.0 0.0; 0.0 0.5 0.0 0.0; 0.0 0.0 1.0 0.0]
    return SolpropJulia.EmbeddedTableau(
        nodes, stages, [1 / 6, 1 / 3, 1 / 3, 1 / 6], zeros(4)
    )
end

@testset "Short assembled Kerr propagation converges to the nonlinear phase" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    n_time = 8
    data = source_data_fixture(
        execution; synthesis = ones(1, 1), projection = ones(1, 1),
        prefactor = fill(-0.5im, n_time, 1), omega_window = ones(n_time),
        source_window = ones(n_time), density = 1.0,
        kerr = SolpropJulia.KerrCoefficients(4 / 3, :analytic),
    )
    source = SolpropJulia.ModalSource(execution, data; node_chunk = 1)
    initial = zeros(ComplexF64, n_time, 1)
    amplitude = 1.0 + 0.3im
    initial[1] = n_time * amplitude
    rate = fill(-0.2im, size(initial))
    flow = SolpropJulia.LinearFlow(
        execution, SolpropJulia.DiagonalOperator(execution, rate), initial
    )
    tableau = source_rk4_tableau()
    mask = fill(true, n_time)
    errors = Float64[]
    length = 0.25
    exact = initial * exp((-0.2im - 0.5im * abs2(amplitude)) * length)
    for steps in (2, 4, 8)
        state = copy(initial)
        work = SolpropJulia.RungeKuttaWork(state, tableau)
        dz = length / steps
        for step in 0:(steps - 1)
            SolpropJulia.rk_trial!(
                work, state, step * dz, dz, tableau, source, flow, execution
            )
            SolpropJulia.rk_commit!(state, work, mask, execution)
        end
        push!(errors, maximum(abs, state - exact))
        @test all(iszero, state[2:end, :])
    end
    # Classical RK4's leading O(h^4) error dominates at these three short windows.
    @test 12 < errors[1] / errors[2] < 20
    @test 12 < errors[2] / errors[3] < 20
    @test errors[end] < 1.0e-8 * maximum(abs, exact)
end
