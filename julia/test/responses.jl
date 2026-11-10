# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function _kerr_allocation_check(destination, samples, layout, coefficients, execution)
    SolpropJulia.kerr_polarisation!(destination, samples, layout, coefficients, execution)
    return @allocated SolpropJulia.kerr_polarisation!(
        destination, samples, layout, coefficients, execution
    )
end

@testset "Kerr layout, scalar limit and independent vector cubic" begin
    for tasks in (1, 2), T in (Float32, Float64)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        layout = SolpropJulia.ResponseLayout(7, 2, 3)
        samples = reshape(T.(sin.(1:42)), 7, 6)
        destination = similar(samples)
        coefficients = SolpropJulia.KerrCoefficients(T(0.125), :real)
        density = T(2.5)
        window = T[1, 0.75, 0.5, 0.25, 0.5, 0.75, 1]
        SolpropJulia.kerr_polarisation!(
            destination, samples, layout, coefficients, execution; density, window
        )
        # At most two squares, their sum, and four products per output. Compare
        # against independent 256-bit arithmetic with a conservative gamma_16 budget.
        bound = 16eps(T) / (1 - 16eps(T)) * 2 * T(0.125) * density
        reference = map(eachindex(samples)) do index
            row, column = Tuple(CartesianIndices(samples)[index])
            first = 2 * ((column - 1) ÷ 2) + 1
            intensity = sum(BigFloat(samples[row, c])^2 for c in first:(first + 1))
            T(BigFloat(0.125) * density * intensity * samples[row, column] * window[row])
        end
        @test maximum(abs, destination - reshape(reference, size(samples))) <= bound
        # Explicit node columns must not mix another node into the intensity.
        changed = copy(samples)
        changed[:, 3:6] .*= T(4)
        result = similar(changed)
        SolpropJulia.kerr_polarisation!(
            result, changed, layout, coefficients, execution; density, window
        )
        @test result[:, 1:2] == destination[:, 1:2]
        inplace = copy(samples)
        SolpropJulia.kerr_polarisation!(
            inplace, inplace, layout, coefficients, execution; density, window
        )
        @test inplace == destination
        @test _kerr_allocation_check(
            destination, samples, layout, coefficients, execution
        ) == 0

        scalar_layout = SolpropJulia.ResponseLayout(3, 1, 1)
        scalar = reshape(Complex{T}[1 + 2im, -2 + 0.5im, 0], 3, 1)
        scalar_result = similar(scalar)
        analytic = SolpropJulia.KerrCoefficients(one(T), :analytic)
        SolpropJulia.kerr_polarisation!(
            scalar_result, scalar, scalar_layout, analytic, execution
        )
        scalar_bound = 16eps(T) * maximum(abs, scalar)^3
        @test maximum(abs, scalar_result - T(0.75) .* abs2.(scalar) .* scalar) <=
            scalar_bound
        @test iszero(scalar_result[3])
        @test _kerr_allocation_check(
            scalar_result, scalar, scalar_layout, analytic, execution
        ) == 0
    end
end

@testset "Analytic Kerr coefficients from the physical real cubic" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    layout = SolpropJulia.ResponseLayout(1, 2, 1)
    envelope = reshape(ComplexF64[0.7 + 1.1im, -1.2 + 0.3im], 1, 2)
    fundamental = similar(envelope)
    harmonic = similar(envelope)
    SolpropJulia.kerr_polarisation!(
        fundamental, envelope, layout,
        SolpropJulia.KerrCoefficients(1.0, :analytic), execution,
    )
    SolpropJulia.kerr_polarisation!(
        harmonic, envelope, layout,
        SolpropJulia.KerrCoefficients(
            1.0, :analytic; self_factor = 0.0, conjugate_factor = 0.0,
            harmonic_factor = 0.25,
        ), execution; carrier = ComplexF64[1],
    )
    # A real cubic has only harmonics ±1 and ±3. Sixteen equally spaced angles
    # integrate their Fourier coefficients exactly in exact arithmetic. Evaluate
    # the physical cubic at 256 bits, independently of the analytic coefficient
    # formula, then convert its real-field Fourier coefficients to envelopes.
    a = Complex{BigFloat}.(vec(envelope))
    first, third = zeros(Complex{BigFloat}, 2), zeros(Complex{BigFloat}, 2)
    for sample in 0:15
        theta = 2big(pi) * sample / 16
        physical = real.(a .* cis(theta))
        response = sum(abs2, physical) .* physical
        first .+= (2 // 16) .* response .* cis(-theta)
        third .+= (2 // 16) .* response .* cis(-3theta)
    end
    # The scalar kernel's complex products/sums fit within 64 rounded operations;
    # sum(abs2(a))*max(abs(a)) bounds either harmonic's absolute contribution.
    scale = sum(abs2, envelope) * maximum(abs, envelope)
    bound = 64eps(Float64) / (1 - 64eps(Float64)) * scale
    @test maximum(abs, vec(fundamental) - ComplexF64.(first)) <= bound
    @test maximum(abs, vec(harmonic) - ComplexF64.(third)) <= bound
end

@testset "Analytic Kerr harmonic, real projection and buffer reuse" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    layout = SolpropJulia.ResponseLayout(5, 2, 2)
    samples = reshape(complex.(sin.(1:20), cos.(1:20)), 5, 4)
    destination = similar(samples)
    carrier = cis.(range(0.0, 2.0; length = 5))
    coefficients = SolpropJulia.KerrCoefficients(0.25, :analytic; harmonic_factor = 0.25)
    SolpropJulia.kerr_polarisation!(
        destination, samples, layout, coefficients, execution; density = 3.0, carrier
    )
    reference = similar(destination)
    for node in 1:2, row in 1:5
        first = 2node - 1
        a, b = Complex{BigFloat}.(samples[row, first:(first + 1)])
        modulus, square = abs2(a) + abs2(b), a * a + b * b
        for (column, value) in ((first, a), (first + 1, b))
            reference[row, column] = 0.75 * (
                0.5 * modulus * value + 0.25 * square * conj(value) +
                    0.25 * Complex{BigFloat}(carrier[row]) * square * value
            )
        end
    end
    # Each input has modulus one; two components and coefficient*density=3/4
    # bound the sum of absolute cubic contributions by 3/2. The 128-operation
    # budget covers complex products, component sums and final reference rounding.
    bound = 128eps(Float64) / (1 - 128eps(Float64)) * 1.5
    @test maximum(abs, destination - reference) <= bound

    real_coefficients = SolpropJulia.KerrCoefficients(
        0.25, :analytic; harmonic_factor = 0.25, take_real = true
    )
    projected = zeros(5, 4)
    SolpropJulia.kerr_polarisation!(
        projected, samples, layout, real_coefficients, execution; density = 3.0, carrier
    )
    @test projected == real.(destination)
    retained = copy(destination)
    SolpropJulia.kerr_polarisation!(
        destination, samples, layout, coefficients, execution;
        density = 3.0, carrier, add = true,
    )
    @test destination == 2 .* retained

    # A full-capacity transform may read inactive columns; never read their NaN
    # input values while clearing the corresponding reused destination storage.
    samples[:, 3:4] .= ComplexF64(NaN, NaN)
    SolpropJulia.kerr_polarisation!(
        destination, samples, layout, coefficients, execution; carrier, count = 1
    )
    @test all(isfinite, destination)
    @test all(iszero, destination[:, 3:4])
    SolpropJulia.kerr_polarisation!(
        destination, samples, layout, coefficients, execution;
        carrier, count = 0, add = true,
    )
    @test all(iszero, destination)
end

@testset "Kerr rejects incompatible prepared inputs" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    layout = SolpropJulia.ResponseLayout(2, 1, 1)
    samples = ones(2, 1)
    output = similar(samples)
    coefficients = SolpropJulia.KerrCoefficients(1.0, :real)
    @test_throws ArgumentError SolpropJulia.ResponseLayout(0, 1, 1)
    @test_throws ArgumentError SolpropJulia.ResponseLayout(big(typemax(Int)) + 1, 1, 1)
    @test_throws ArgumentError SolpropJulia.ResponseLayout(1, typemax(Int), 2)
    @test_throws ArgumentError SolpropJulia.KerrCoefficients(Inf, :real)
    @test_throws ArgumentError SolpropJulia.KerrCoefficients(1.0, :unknown)
    @test_throws ArgumentError SolpropJulia.KerrCoefficients(
        1.0, :real; harmonic_factor = 1
    )
    @test_throws DimensionMismatch SolpropJulia.kerr_polarisation!(
        zeros(3, 1), samples, layout, coefficients, execution
    )
    @test_throws DimensionMismatch SolpropJulia.kerr_polarisation!(
        output, samples, layout, coefficients, execution; window = [1.0]
    )
    @test_throws ArgumentError SolpropJulia.kerr_polarisation!(
        output, samples, layout, coefficients, execution; count = 2
    )
    @test_throws ArgumentError SolpropJulia.kerr_polarisation!(
        output, samples, layout, coefficients, execution; density = -1
    )
    @test_throws ArgumentError SolpropJulia.kerr_polarisation!(
        output, complex.(samples), layout, coefficients, execution
    )
    harmonic = SolpropJulia.KerrCoefficients(1.0, :analytic; harmonic_factor = 0.25)
    @test_throws ArgumentError SolpropJulia.kerr_polarisation!(
        complex.(output), complex.(samples), layout, harmonic, execution
    )
    @test_throws DimensionMismatch SolpropJulia.kerr_polarisation!(
        complex.(output), complex.(samples), layout, harmonic, execution; carrier = [1im]
    )
end
