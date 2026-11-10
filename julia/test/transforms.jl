# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function transform_cycle!(time, spectrum, plan, execution)
    SolpropJulia.temporal_forward!(spectrum, plan, time)
    SolpropJulia.temporal_inverse!(time, plan, spectrum, execution)
    return nothing
end

function spatial_cycle!(buffer, plan, execution)
    SolpropJulia.spatial_forward!(buffer, plan)
    SolpropJulia.spatial_inverse!(buffer, plan, execution)
    return nothing
end

@testset "Explicit spatial axes and strides" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    flattened = ComplexF64[row / 7 + im * column / 11 for row in 1:5, column in 1:24]
    buffer = reshape(flattened, 5, 2, 3, 4)
    original = copy(buffer)
    old_threads = SolpropJulia.FFTW.get_num_threads()
    plan = SolpropJulia.SpatialFFT(ex, buffer, (3, 4))
    @test buffer == original
    @test SolpropJulia.FFTW.get_num_threads() == old_threads
    SolpropJulia.spatial_forward!(buffer, plan)
    reference = ComplexF64[
        sum(
            original[frequency, component, x, y] *
                cis(-2pi * ((x - 1) * (kx - 1) / 3 + (y - 1) * (ky - 1) / 4))
                for x in 1:3, y in 1:4
        ) for frequency in 1:5, component in 1:2, kx in 1:3, ky in 1:4
    ]
    # Twelve-term direct DFT versus the factored FFT, including explicit phase
    # evaluation. This also witnesses the translated component-first column order.
    budget = 64 * 12 * eps(Float64) * maximum(abs, original)
    @test maximum(abs, buffer - reference) <= budget
    SolpropJulia.spatial_inverse!(buffer, plan, ex)
    @test maximum(abs, buffer - original) <= budget
    spatial_cycle!(buffer, plan, ex)
    @test @allocated(spatial_cycle!(buffer, plan, ex)) == 0

    parent = fill(ComplexF64(-99), 10, 2, 3, 4)
    strided = view(parent, 1:2:10, :, :, :)
    copyto!(strided, original)
    strided_plan = SolpropJulia.SpatialFFT(ex, strided, (3, 4))
    spatial_cycle!(strided, strided_plan, ex)
    @test maximum(abs, strided - original) <= budget
    @test all(==(-99), view(parent, 2:2:10, :, :, :))
    @test @allocated(spatial_cycle!(strided, strided_plan, ex)) == 0
    @test_throws ArgumentError SolpropJulia.SpatialFFT(ex, buffer, ())
    @test_throws ArgumentError SolpropJulia.SpatialFFT(ex, buffer, (3, 3))
    @test_throws ArgumentError SolpropJulia.SpatialFFT(ex, buffer, (0,))
end

function padding_cycle!(coarse, fine, padding, execution, active)
    SolpropJulia.pad_spectrum!(fine, coarse, padding, execution; active_columns = active)
    SolpropJulia.crop_spectrum!(coarse, fine, padding, execution; active_columns = active)
    return nothing
end

function transform_map!(output, input, weights, execution, active_inputs, active_columns)
    SolpropJulia.linear_map!(
        output, input, weights, execution; active_inputs, active_columns,
    )
    return nothing
end

function transform_components!(nodal, modal, synthesis, projection, execution, active_nodes)
    SolpropJulia.synthesize!(
        nodal, modal, synthesis, execution; n_components = 2, active_nodes,
    )
    SolpropJulia.project!(
        modal, nodal, projection, execution; n_components = 2, active_nodes,
    )
    return nothing
end

@testset "Raw temporal transforms" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    for T in (Float64, Float32), real_signal in (false, true), n_time in (8, 9)
        samples = T[
            sin(2pi * (row - 1) / n_time) + column / 3
                for row in 1:n_time, column in 1:3
        ]
        time = real_signal ? samples : complex.(samples, reverse(samples; dims = 1) ./ 7)
        original = copy(time)
        n_frequency = real_signal ? div(n_time, 2) + 1 : n_time
        spectrum = zeros(Complex{T}, n_frequency, 3)
        original_threads = SolpropJulia.FFTW.get_num_threads()
        plan = SolpropJulia.TemporalFFT(ex, time, spectrum)
        @test SolpropJulia.FFTW.get_num_threads() == original_threads
        @test time == original
        @test spectrum == zeros(Complex{T}, n_frequency, 3)
        SolpropJulia.temporal_forward!(spectrum, plan, time)
        direct = ComplexF64[
            sum(
                original[row, column] * cis(-2pi * (row - 1) * (frequency - 1) / n_time)
                    for row in 1:n_time
            ) for frequency in 1:n_frequency, column in 1:3
        ]
        # A direct N-term DFT and an FFT each accumulate O(N*eps) rounding; the
        # factor 32 includes trigonometric argument reduction in the independent sum.
        budget = 32n_time * eps(T) * maximum(abs, original)
        @test maximum(abs, spectrum - direct) <= budget
        @test time == original
        saved_spectrum = copy(spectrum)
        SolpropJulia.temporal_inverse!(time, plan, spectrum, ex)
        @test spectrum == saved_spectrum
        @test maximum(abs, time - original) <= budget
        transform_cycle!(time, spectrum, plan, ex)
        @test @allocated(transform_cycle!(time, spectrum, plan, ex)) == 0
    end
    @test_throws DimensionMismatch SolpropJulia.TemporalFFT(
        ex, zeros(8, 2), zeros(ComplexF64, 8, 2),
    )
    @test_throws ArgumentError SolpropJulia.TemporalFFT(
        ex, zeros(Int, 8, 2), zeros(ComplexF64, 5, 2),
    )
    @test_throws ArgumentError SolpropJulia.TemporalFFT(
        ex, zeros(0, 2), zeros(ComplexF64, 1, 2),
    )
    shared = zeros(ComplexF64, 8, 2)
    @test_throws ArgumentError SolpropJulia.TemporalFFT(ex, shared, shared)
    for T in (Float64, ComplexF64)
        n_frequency = T === Float64 ? 5 : 8
        time_owner = ones(T, 16, 2)
        spectral_owner = zeros(ComplexF64, 2n_frequency, 2)
        @test_throws ArgumentError SolpropJulia.TemporalFFT(
            ex, view(time_owner, 1:2:16, :), zeros(ComplexF64, n_frequency, 2),
        )
        @test_throws ArgumentError SolpropJulia.TemporalFFT(
            ex, ones(T, 8, 2), view(spectral_owner, 1:2:(2n_frequency), :),
        )
        # A contiguous view can still have a different FFTW alignment at its offset.
        @test_throws ArgumentError SolpropJulia.TemporalFFT(
            ex, reshape(view(vec(time_owner), 2:17), 8, 2),
            zeros(ComplexF64, n_frequency, 2),
        )
        @test time_owner == ones(T, 16, 2)
        @test iszero(spectral_owner)
    end
end

@testset "Authoritative padding and inactive columns" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    for real_signal in (false, true)
        padding = SolpropJulia.SpectralPadding(ex, 8, 16; real_signal)
        crop = real_signal ? collect(1:5) : [1, 2, 3, 4, 13, 14, 15, 16]
        @test padding.crop == crop
        n_coarse, n_fine = length(crop), length(padding.uncrop)
        coarse = ComplexF64[row + im * column for row in 1:n_coarse, column in 1:3]
        original = copy(coarse)
        fine = fill(ComplexF64(NaN), n_fine, 4)
        SolpropJulia.pad_spectrum!(fine, coarse, padding, ex; active_columns = 2)
        expected = zeros(ComplexF64, n_fine, 4)
        expected[crop, 1:2] = 2coarse[:, 1:2]
        @test fine == expected
        @test coarse == original
        SolpropJulia.crop_spectrum!(coarse, fine, padding, ex; active_columns = 2)
        @test coarse[:, 1:2] == original[:, 1:2]
        @test iszero(coarse[:, 3])
        padding_cycle!(coarse, fine, padding, ex, 2)
        @test @allocated(padding_cycle!(coarse, fine, padding, ex, 2)) == 0
        explicit = SolpropJulia.SpectralPadding(ex, crop .- 1, n_fine, 2.0, 0.5)
        @test explicit.crop == padding.crop
        @test explicit.uncrop == padding.uncrop

        n_time = 8
        time = reshape([cos(2pi * k / n_time) for k in 0:(n_time - 1)], n_time, 1)
        real_signal || (time = complex.(time, time ./ 3))
        fine_time = zeros(eltype(time), 16, 1)
        coarse_spectrum = zeros(ComplexF64, n_coarse, 1)
        fine_spectrum = zeros(ComplexF64, n_fine, 1)
        coarse_plan = SolpropJulia.TemporalFFT(ex, time, coarse_spectrum)
        fine_plan = SolpropJulia.TemporalFFT(ex, fine_time, fine_spectrum)
        SolpropJulia.temporal_forward!(coarse_spectrum, coarse_plan, time)
        real_signal && (coarse_spectrum[end] = 0)
        SolpropJulia.pad_spectrum!(fine_spectrum, coarse_spectrum, padding, ex)
        SolpropJulia.temporal_inverse!(fine_time, fine_plan, fine_spectrum, ex)
        # Two length-16 transforms plus the independently evaluated sampled sinusoid.
        @test maximum(abs, fine_time[1:2:end, :] - time) <= 128eps(Float64)
        standalone = copy(fine_time)
        unit_padding = SolpropJulia.SpectralPadding(ex, crop .- 1, n_fine, 1.0, 1.0)
        SolpropJulia.pad_spectrum!(fine_spectrum, coarse_spectrum, unit_padding, ex)
        SolpropJulia.temporal_inverse!(fine_time, fine_plan, fine_spectrum, ex)
        # Translation may already fold pad_scale into synthesis; apply it only once.
        SolpropJulia.apply_weights!(fine_time, fine_time, fill(2.0, 16), ex)
        @test maximum(abs, fine_time - standalone) <= 128eps(Float64)
    end
    @test_throws ArgumentError SolpropJulia.SpectralPadding(ex, [0, 0], 4, 1.0, 1.0)
    @test_throws ArgumentError SolpropJulia.SpectralPadding(ex, [4], 4, 1.0, 1.0)
    @test_throws ArgumentError SolpropJulia.SpectralPadding(ex, [0], 4, NaN, 1.0)
    @test_throws ArgumentError SolpropJulia.SpectralPadding(ex, 8, 12; real_signal = true)
end

@testset "Bounded maps and component layout" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    input = ComplexF64[row + im * column for row in 1:7, column in 1:4]
    weights = [row / 7 - column / 11 for row in 1:4, column in 1:6]
    output = fill(ComplexF64(NaN), 7, 6)
    transform_map!(output, input, weights, ex, 3, 5)
    reference = input[:, 1:3] * weights[1:3, 1:5]
    # Three complex-real products and their sequential sum, with a conservative
    # 16eps multiple of the absolute product sum for the BLAS comparison.
    bound = 16eps(Float64) * maximum(abs.(input[:, 1:3]) * abs.(weights[1:3, 1:5]))
    @test maximum(abs, output[:, 1:5] - reference) <= bound
    @test iszero(output[:, 6])
    @test @allocated(transform_map!(output, input, weights, ex, 3, 5)) == 0

    synthesis = [1.0 0.5; -0.25 1.0; 2.0 -1.0]
    projection = [0.25 0.5 0.0; 0.5 -0.25 0.0]
    modal = copy(input)
    nodal = fill(ComplexF64(NaN), 7, 8)
    SolpropJulia.synthesize!(
        nodal, modal, synthesis, ex; n_components = 2, active_nodes = 2,
    )
    for node in 1:2, component in 1:2
        expected = modal[:, component] * synthesis[node, 1] +
            modal[:, component + 2] * synthesis[node, 2]
        # Two complex-real products and one addition, against the same explicit map.
        @test maximum(abs, nodal[:, component + 2(node - 1)] - expected) <=
            16eps(Float64) * max(1.0, maximum(abs, expected))
    end
    @test iszero(nodal[:, 5:8])
    original_nodal = copy(nodal)
    SolpropJulia.project!(modal, nodal, projection, ex; n_components = 2, active_nodes = 2)
    for mode in 1:2, component in 1:2
        expected = nodal[:, component] * projection[mode, 1] +
            nodal[:, component + 2] * projection[mode, 2]
        @test maximum(abs, modal[:, component + 2(mode - 1)] - expected) <=
            16eps(Float64) * max(1.0, maximum(abs, expected))
    end
    @test nodal == original_nodal
    transform_components!(nodal, modal, synthesis, projection, ex, 2)
    @test @allocated(transform_components!(nodal, modal, synthesis, projection, ex, 2)) == 0
    input[2, 1] = NaN + NaN * im
    mask = ones(7)
    mask[2] = 0
    SolpropJulia.apply_weights!(output, input, mask, ex; active_columns = 2)
    @test iszero(output[2, 1])
    @test iszero(output[:, 3:6])
    @test all(isfinite, output)
    @test_throws ArgumentError SolpropJulia.linear_map!(input, input, weights, ex)
    @test_throws DimensionMismatch SolpropJulia.linear_map!(
        output, input, weights, ex; active_inputs = 5,
    )
    @test_throws DimensionMismatch SolpropJulia.synthesize!(
        nodal, input, synthesis, ex; n_components = 3,
    )
end
