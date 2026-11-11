# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import SolpropJulia as SP

function spatial_operator_inputs(; changes...)
    coefficients = (
        k_squared = ComplexF64[4 + 0.2im, 9, 2.0^54, 2.0^54, 5 - 0.1im],
        k_squared_low = ComplexF64[1.0e-16, 0, 1, -1, -1.0e-16im],
        kperp_sq = [0.0, 4.0, 9.0, 2.0^54],
        kperp_sq_low = [0.0, 1.0e-16, -1.0e-16, 0.0],
        frame = [0.1, 1.0, 0.2, 0.3, -0.4],
        spectral_rate = [0.01, 0.02, 0.03, 0.04, 0.05],
        prefactor_numerator = ComplexF64[-0.1im, -0.2im, -0.3im, -0.4im, -0.5im],
        prefactor_clamped = fill(-0.7im, 5), active = [true, true, true, true, false],
        evanescent_rate = 0.25, z_span = (-0.1, 0.5),
        kperp_rate = [0.0, 0.1, 0.2, 0.3],
    )
    return merge(coefficients, (; changes...))
end

function spatial_operator_reference(inputs, row, column)
    # Test the actual represented q graph separately from its exact coefficient sum.
    # MPFR then solves the principal-root formula on that rounded represented input.
    square = (inputs.k_squared[row] - inputs.kperp_sq[column]) +
        (inputs.k_squared_low[row] - inputs.kperp_sq_low[column])
    exact_square = (
        Complex{BigFloat}(inputs.k_squared[row]) -
            BigFloat(inputs.kperp_sq[column])
    ) +
        (
        Complex{BigFloat}(inputs.k_squared_low[row]) -
            BigFloat(inputs.kperp_sq_low[column])
    )
    magnitudes = abs(Complex{BigFloat}(inputs.k_squared[row])) +
        abs(BigFloat(inputs.kperp_sq[column])) +
        abs(Complex{BigFloat}(inputs.k_squared_low[row])) +
        abs(BigFloat(inputs.kperp_sq_low[column]))
    @test abs(Complex{BigFloat}(square) - exact_square) <=
        4eps(Float64) * magnitudes
    root = sqrt(Complex{BigFloat}(square))
    propagating = real(square) > 0
    attenuation = propagating ? imag(root) : min(abs(imag(root)), inputs.evanescent_rate)
    rate = complex(
        -attenuation - BigFloat(inputs.kperp_rate[column]) / 2,
        -(real(root) - BigFloat(inputs.frame[row])),
    )
    prefactor = propagating ? Complex{BigFloat}(inputs.prefactor_numerator[row]) /
        real(root) : Complex{BigFloat}(inputs.prefactor_clamped[row])
    scale = abs(attenuation) + abs(real(root)) + abs(inputs.frame[row]) +
        inputs.kperp_rate[column] / 2 + inputs.spectral_rate[row] / 2
    if !inputs.active[row]
        rate, prefactor = zero(rate), zero(prefactor)
    end
    rate -= BigFloat(inputs.spectral_rate[row]) / 2
    # Complex sqrt, subtraction and reciprocal are conditioned by their absolute
    # terms. The positive-q division never takes a near-zero difference of roots.
    return rate, prefactor, 64eps(Float64) * scale, 64eps(Float64) * abs(prefactor)
end

function spatial_operator_allocations(output, operator, execution)
    SP.operator_exponent!(output, operator, 0.1, 0.3, execution)
    SP.operator_rate!(output, operator, 0.2, execution)
    SP.spatial_prefactor!(output, operator, execution)
    return @allocated begin
        SP.operator_exponent!(output, operator, 0.1, 0.3, execution)
        SP.operator_rate!(output, operator, 0.2, execution)
        SP.spatial_prefactor!(output, operator, execution)
    end
end

@testset "Separable spatial rates preserve corrected grazing branches and scales" begin
    inputs = spatial_operator_inputs()
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        operator = SP.SeparableSpatialOperator(execution; inputs...)
        rate = zeros(ComplexF64, 5, 4)
        prefactor = similar(rate)
        SP.operator_rate!(rate, operator, 0.2, execution)
        SP.spatial_prefactor!(prefactor, operator, execution)
        setprecision(BigFloat, 256) do
            for column in 1:4, row in 1:5
                expected, expected_prefactor, bound, prefactor_bound =
                    spatial_operator_reference(inputs, row, column)
                @test bound > 0
                @test abs(rate[row, column] - expected) <= bound
                if inputs.active[row]
                    @test prefactor_bound > 0
                    @test abs(prefactor[row, column] - expected_prefactor) <=
                        prefactor_bound
                else
                    @test iszero(prefactor[row, column])
                    @test rate[row, column] == -inputs.spectral_rate[row] / 2
                end
            end
        end
        @test SP._spatial_square(SP._spatial_operator_tables(operator), 3, 4) == 1
        @test SP._spatial_square(SP._spatial_operator_tables(operator), 4, 4) == -1
        @test prefactor[3, 4] == inputs.prefactor_numerator[3]
        @test prefactor[4, 4] == inputs.prefactor_clamped[4]
        exponent = similar(rate)
        SP.operator_exponent!(exponent, operator, 0.1, 0.3, execution)
        @test maximum(abs, exponent - (0.3 - 0.1) .* rate) <=
            4eps(Float64) * maximum(abs, rate)
        SP.operator_exponent!(exponent, operator, 0.2, 0.2, execution)
        @test all(iszero, exponent)
        SP.operator_rate!(exponent, operator, 0.2, execution; before = true)
        @test maximum(abs, exponent - rate) <= 4eps(Float64) * maximum(abs, rate)
        @test spatial_operator_allocations(exponent, operator, execution) == 0
        for name in (
                :k_squared, :k_squared_low, :kperp_sq, :kperp_sq_low, :frame,
                :spectral_rate, :prefactor_numerator, :prefactor_clamped, :active,
                :kperp_rate,
            )
            @test !Base.mightalias(getproperty(operator, name), getproperty(inputs, name))
        end
    end
end

@testset "Spatial preparation rejects unsupported inputs and overflow" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    inputs = spatial_operator_inputs()
    for changes in (
            (n_pol = 2,), (kappa_sq = zeros(ComplexF64, 1, 5),),
            (k_squared = ones(ComplexF32, 5),), (frame = zeros(5, 1),),
            (evanescent_rate = 0.0,), (z_span = (0.0, 0.0),),
            (kperp_sq = [-1.0, 0, 0, 0],), (spectral_rate = fill(-1.0, 5),),
            (prefactor_numerator = ones(ComplexF64, 5),),
            (k_squared_low = fill(ComplexF64(NaN), 5),),
            (kperp_rate = fill(-0.1, 4),),
            (
                k_squared = fill(complex(-floatmax(Float64)), 5),
                kperp_sq = fill(floatmax(Float64), 4),
            ),
            (
                k_squared = fill(complex(floatmin(Float64)), 5),
                k_squared_low = zeros(ComplexF64, 5), kperp_sq = zeros(4),
                kperp_sq_low = zeros(4),
                prefactor_numerator = fill(-floatmax(Float64)im, 5),
            ),
        )
        @test_throws ArgumentError SP.SeparableSpatialOperator(
            execution; merge(inputs, changes)...,
        )
    end
    for changes in ((frame = ones(4),), (kperp_rate = ones(3),))
        @test_throws DimensionMismatch SP.SeparableSpatialOperator(
            execution; merge(inputs, changes)...,
        )
    end
    without_rate = SP.SeparableSpatialOperator(
        execution; merge(inputs, (kperp_rate = nothing,))...,
    )
    @test all(iszero, without_rate.kperp_rate)
    output = fill(1.0 + 2im, 5, 4)
    before = copy(output)
    @test_throws ArgumentError SP.operator_exponent!(
        output, without_rate, -0.2, 0.2, execution,
    )
    @test_throws ArgumentError SP.operator_rate!(output, without_rate, NaN, execution)
    @test output == before
    @test_throws DimensionMismatch SP.spatial_prefactor!(
        zeros(ComplexF64, 5, 3),
        without_rate, execution
    )
    @test_throws ArgumentError SP.spatial_prefactor!(zeros(5, 4), without_rate, execution)
    single = SP.SeparableSpatialOperator(
        execution; merge(
            inputs, (
                kperp_sq = [0.0], kperp_sq_low = [0.0],
                kperp_rate = [0.0],
            )
        )...,
    )
    saved = copy(single.k_squared)
    for operation in (
            output -> SP.spatial_prefactor!(output, single, execution),
            output -> SP.operator_rate!(output, single, 0.2, execution),
            output -> SP.operator_exponent!(output, single, 0.1, 0.2, execution),
        )
        @test_throws ArgumentError operation(reshape(single.k_squared, 5, 1))
        @test single.k_squared == saved
    end
end
function spatial_flow_allocations(flow, output, input)
    flow(output, input, 0.1, 0.100000001)
    flow(output, input, 0.1, 0.100000001)
    return @allocated flow(output, input, 0.1, 0.100000001)
end

@testset "Spatial exponents convert endpoint types before subtraction" begin
    execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), 1, true)
    inputs = spatial_operator_inputs()
    operator = SP.SeparableSpatialOperator(
        execution; inputs..., z_span = (-1.0e19, 1.0e19),
    )
    output = zeros(ComplexF64, 5, 4)
    for (left, right) in (
            (typemin(Int), typemax(Int)), (typemax(Int), typemin(Int)),
        )
        SP.operator_exponent!(output, operator, left, right, execution)
        distance = BigFloat(BigInt(right) - BigInt(left))
        for column in 1:4, row in 1:5
            rate, unused, rate_bound, unused_bound =
                spatial_operator_reference(inputs, row, column)
            expected = rate * distance
            # Endpoint conversion/subtraction error plus the independently bounded
            # represented-root/rate arithmetic; an integer wrap is not in this bound.
            bound = 8eps(Float64) *
                (abs(BigFloat(left)) + abs(BigFloat(right))) * abs(rate) +
                abs(distance) * rate_bound
            @test abs(Complex{BigFloat}(output[row, column]) - expected) <= bound
        end
    end
    SP.operator_exponent!(output, operator, typemax(Int), typemax(Int), execution)
    @test iszero(output)
end

@testset "Separable coefficients drive the shared in-place linear flow" begin
    inputs = spatial_operator_inputs()
    for tasks in unique((1, min(2, Threads.nthreads(:default))))
        execution = SP.KernelExecution(SP.KernelAbstractions.CPU(), tasks, true)
        operator = SP.SeparableSpatialOperator(execution; inputs...)
        input = ComplexF64[cis(0.2row + 0.3column) for row in 1:5, column in 1:4]
        output = copy(input)
        flow = SP.LinearFlow(execution, operator, input)
        first_position = 0.1
        last_position = first_position
        for index in 1:8
            destination = first_position + index * 1.0e-9
            flow(output, output, last_position, destination)
            last_position = destination
        end
        setprecision(BigFloat, 256) do
            distance = BigFloat(last_position) - BigFloat(first_position)
            for column in 1:4, row in 1:5
                rate, unused, rate_bound, unused_bound =
                    spatial_operator_reference(inputs, row, column)
                expected = Complex{BigFloat}(input[row, column]) * exp(distance * rate)
                # Eight complex products/exp evaluations plus represented-root error.
                # The step endpoints are exact binary64 inputs to the BigFloat flow.
                relative = 512eps(Float64) * (1 + abs(distance * rate)) +
                    abs(distance) * rate_bound
                @test abs(Complex{BigFloat}(output[row, column]) - expected) <=
                    relative * abs(expected)
            end
        end
        @test spatial_flow_allocations(flow, output, input) == 0
        flow(output, input, 0.1, 0.1)
        @test output == input
    end
end
