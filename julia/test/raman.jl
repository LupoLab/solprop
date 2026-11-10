# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function _raman_convolution_allocations!(drive, frequency, prepared, execution)
    SolpropJulia.convolve_raman!(drive, frequency, prepared, execution)
    return @allocated SolpropJulia.convolve_raman!(drive, frequency, prepared, execution)
end

function _raman_rebuild_allocations!(
        time, spectrum, kernel, rates, weights, forward, execution,
    )
    SolpropJulia.rebuild_raman!(time, spectrum, kernel, rates, weights, forward, execution)
    return @allocated SolpropJulia.rebuild_raman!(
        time, spectrum, kernel, rates, weights, forward, execution
    )
end

@testset "Raman doubled-grid drive and smallest non-THG products" begin
    for tasks in (1, 2), n_time in (1, 2, 7)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        layout = SolpropJulia.ResponseLayout(n_time, 1, 2)
        samples = reshape(Float64.(1:(2n_time)), n_time, 2)
        analytic = complex.(samples, reverse(samples; dims = 1))
        drive = fill(17.0, 2n_time, 2)
        prepared = SolpropJulia.RamanDrive(layout; thg = false)
        SolpropJulia.raman_drive!(
            drive, samples, prepared, execution; analytic_samples = analytic
        )
        @test drive[1:n_time, :] == abs2.(analytic) ./ 2
        @test all(iszero, drive[(n_time + 1):end, :])
        samples[:, 2] .= NaN
        analytic[:, 2] .= ComplexF64(NaN, NaN)
        SolpropJulia.raman_drive!(
            drive, samples, prepared, execution; analytic_samples = analytic, count = 1
        )
        @test all(iszero, drive[:, 2])
        @test all(isfinite, drive)
        SolpropJulia.raman_drive!(
            drive, samples, prepared, execution; analytic_samples = analytic, count = 0
        )
        @test all(iszero, drive)

        # Two physical samples reproduce issue #613: the padding loop must branch
        # before reading a third analytic sample, even when several pairs are used.
        signals = ComplexF64[1 + 2im 3 - 1im; -2 + 1im 1 + 0.5im]
        first, second = [1, 1, 2], [1, 2, 2]
        products = fill(23.0, 4, 3)
        SolpropJulia.raman_products!(
            products, signals, first, second, execution; thg = false
        )
        for pair in eachindex(first), row in 1:2
            @test products[row, pair] ==
                real(signals[row, first[pair]] * conj(signals[row, second[pair]])) / 2
        end
        @test all(iszero, products[3:4, :])
        original = copy(products)
        time_kernel = [1.0, 0.25, 0.0, 0.0]
        dt = 0.125
        kernel = SolpropJulia.FFTW.rfft(time_kernel) .* dt
        frequency = zeros(ComplexF64, 3, 3)
        forward = SolpropJulia.FFTW.plan_rfft(
            products, 1; flags = SolpropJulia.FFTW.ESTIMATE
        )
        inverse = SolpropJulia.FFTW.plan_brfft(
            frequency, 4, 1; flags = SolpropJulia.FFTW.ESTIMATE
        )
        convolution = SolpropJulia.RamanConvolution(2, kernel, forward, inverse)
        SolpropJulia.convolve_raman!(products, frequency, convolution, execution)
        reference = zeros(4, 3)
        for column in 1:3, row in 1:4, input in 1:2
            lag = row - input + 1
            if 1 <= lag <= 2
                reference[row, column] += dt * time_kernel[lag] * original[input, column]
            end
        end
        # Two length-4 FFTs and a spectral product, bounded by a conservative
        # 64-operation rounding budget times the absolute convolution scale.
        scale = dt * sum(abs, time_kernel) * maximum(abs, original)
        bound = 64eps(Float64) / (1 - 64eps(Float64)) * scale
        @test maximum(abs, products - reference) <= bound
        @test _raman_convolution_allocations!(
            products, frequency, convolution, execution
        ) == 0
    end
end

@testset "Raman independent stage weights, damping and zero kernels" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    rows = zeros(8, 2)
    rows[1:4, 1] .= [0.0, 1.0, 0.5, 0.25]
    rows[1:4, 2] .= [0.0, 0.25, -0.5, 0.125]
    prepared = SolpropJulia.RamanKernel(rows, 0.1)
    time = fill(19.0, 8)
    spectrum = fill(19.0 + 2im, 5)
    forward = SolpropJulia.FFTW.plan_rfft(time; flags = SolpropJulia.FFTW.ESTIMATE)
    weights, rates = [2.0, 5.0], [0.3, 0.7]
    SolpropJulia.rebuild_raman!(
        time, spectrum, prepared, rates, weights, forward, execution
    )
    reference = [
        Float64(
            sum(
                BigFloat(weights[k]) * rows[row, k] *
                    exp(-BigFloat(rates[k]) * (row - 1) * BigFloat(0.1)) for k in 1:2
            )
        ) for row in 1:8
    ]
    # Two terms, each with three products and one exponential. The fixed absolute
    # bound uses sum of absolute undamped contributions, not the cancelled result.
    bound = 32eps(Float64) * maximum(abs.(rows) * abs.(weights))
    @test maximum(abs, time - reference) <= bound
    @test all(iszero, time[5:8])
    saved = copy(time)
    rates[1] = 1.3
    weights[2] = 0.0
    SolpropJulia.rebuild_raman!(
        time, spectrum, prepared, rates, weights, forward, execution
    )
    @test time != saved
    @test maximum(abs, time - 2 .* rows[:, 1] .* exp.(-1.3 .* (0:7) .* 0.1)) <= bound
    @test _raman_rebuild_allocations!(
        time, spectrum, prepared, rates, weights, forward, execution
    ) == 0
    fill!(weights, 0)
    SolpropJulia.rebuild_raman!(
        time, spectrum, prepared, rates, weights, forward, execution
    )
    @test all(iszero, time)
    @test all(iszero, spectrum)
end

@testset "Raman polarisation and invalid preparation" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    layout = SolpropJulia.ResponseLayout(2, 1, 2)
    samples = ComplexF64[1 + 2im 3im; 2 - 1im -2]
    drive = fill(8.0, 4, 2)
    envelope = SolpropJulia.RamanDrive(layout; kind = :envelope, thg = true)
    SolpropJulia.raman_drive!(drive, samples, envelope, execution)
    @test drive[1:2, :] == abs2.(samples) ./ 2
    @test all(iszero, drive[3:4, :])
    output = similar(samples)
    window = [1.0, 0.5]
    SolpropJulia.raman_polarisation!(
        output, samples, drive, layout, execution; factor = 2.0, window
    )
    @test output == 2 .* samples .* drive[1:2, :] .* window
    inplace = copy(samples)
    SolpropJulia.raman_polarisation!(
        inplace, inplace, drive, layout, execution; factor = 2.0, window
    )
    @test inplace == output
    SolpropJulia.raman_polarisation!(
        output, samples, drive, layout, execution;
        factor = 2.0, window, count = 1, add = true,
    )
    @test all(iszero, output[:, 2])
    @test output[:, 1] == 2 .* inplace[:, 1]

    @test_throws ArgumentError SolpropJulia.RamanDrive(layout; kind = :unknown)
    @test_throws ArgumentError SolpropJulia.RamanDrive(SolpropJulia.ResponseLayout(2, 2, 1))
    @test_throws ArgumentError SolpropJulia.RamanKernel(ones(4, 1), 0.1)
    @test_throws ArgumentError SolpropJulia.RamanKernel(zeros(3, 1), 0.1)
    @test_throws ArgumentError SolpropJulia.RamanKernel(zeros(4, 1), 0.0)
    @test_throws ArgumentError SolpropJulia.raman_drive!(
        drive, real.(samples), SolpropJulia.RamanDrive(layout; thg = false), execution
    )
    @test_throws DimensionMismatch SolpropJulia.raman_drive!(
        zeros(3, 2), samples, envelope, execution
    )
    @test_throws ArgumentError SolpropJulia.raman_products!(
        zeros(4, 1), samples, [0], [1], execution; thg = false
    )
    @test_throws ArgumentError SolpropJulia.raman_products!(
        zeros(4, 1), samples, [1], [1], execution; thg = true
    )
end
