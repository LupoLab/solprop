# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

@testset "Propagator outputs reject hidden overlap and repeated writable indices" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1)
    owner = reshape(ComplexF64[1, 2, 3, 4, 5], 5, 1)
    saved = copy(owner)
    exponent = zeros(ComplexF64, 4, 1)
    other = fill(7.0 + 2.0im, 4, 1)
    GC.@preserve owner begin
        input = unsafe_wrap(Array, pointer(owner), (4, 1); own = false)
        output = unsafe_wrap(Array, pointer(owner, 2), (4, 1); own = false)
        for operation in (SolpropJulia.propagate!, SolpropJulia.linear_increment!)
            @test_throws ArgumentError operation(output, input, exponent, execution)
            @test_throws ArgumentError operation(output, other, input, execution)
            @test owner == saved
        end
        @test_throws ArgumentError SolpropJulia.propagator_pair!(
            output, input, exponent, execution,
        )
        @test_throws ArgumentError SolpropJulia.propagator_pair!(
            output, other, input, execution,
        )
        @test owner == saved && all(==(7.0 + 2.0im), other)
    end
    repeated = view(other, [1, 1, 3, 4], :)
    for operation in (SolpropJulia.propagate!, SolpropJulia.linear_increment!)
        @test_throws ArgumentError operation(repeated, exponent, exponent, execution)
        @test all(==(7.0 + 2.0im), other)
    end
    @test_throws ArgumentError SolpropJulia.propagator_pair!(
        repeated, copy(other), exponent, execution,
    )
    @test all(==(7.0 + 2.0im), other)
end

function operator_cycle!(output, inverse, exponent, state, operator, execution)
    SolpropJulia.operator_exponent!(exponent, operator, 0.25, 0.75, execution)
    SolpropJulia.propagate!(output, state, exponent, execution)
    SolpropJulia.propagator_pair!(output, inverse, exponent, execution)
    SolpropJulia.linear_increment!(output, state, exponent, execution)
    return nothing
end

function rate_cycle!(output, operator, work, execution)
    SolpropJulia.operator_rate!(output, operator, 1.0, execution; before = true, work)
    return nothing
end

@testset "Represented operator rates and knot sides" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    rate = reshape(ComplexF64[-2im, -1 + im], 2, 1)
    constant = SolpropJulia.DiagonalOperator(ex, rate)
    table = SolpropJulia.ChebyshevTable(
        ex, [0.0, 1.0, 2.0], reshape([0.5, 0.5, 2.0, 1.0], 1, 2, 2),
    )
    expansion = SolpropJulia.ExpansionOperator(ex, reshape(rate, 2, 1, 1), table)
    cumulative = zeros(ComplexF64, 2, 2, 2)
    for interval in 1:2, degree in 1:2, row in 1:2
        cumulative[row, degree, interval] =
            rate[row] * table.coefficients[1, degree, interval]
    end
    represented = SolpropJulia.RepresentedOperator(
        ex, SolpropJulia.ChebyshevTable(ex, table.breakpoints, cumulative), 2, 1,
    )
    output = similar(rate)
    for operator in (constant, expansion, represented)
        work = operator isa SolpropJulia.ExpansionOperator ? zeros(1) : nothing
        rate_cycle!(output, operator, work, ex)
        @test output == rate
        @test @allocated(rate_cycle!(output, operator, work, ex)) == 0
        SolpropJulia.operator_rate!(output, operator, 1.0, ex; work)
        @test output == (operator === constant ? rate : 2rate)
        @test_throws DimensionMismatch SolpropJulia.operator_rate!(
            zeros(ComplexF64, 3, 1), operator, 1.0, ex; work,
        )
        @test_throws ArgumentError SolpropJulia.operator_rate!(
            output, operator, NaN, ex; work,
        )
    end
    @test_throws ArgumentError SolpropJulia.operator_rate!(
        constant.rate, constant, 0.0, ex,
    )
    @test_throws DimensionMismatch SolpropJulia.operator_rate!(
        output, expansion, 0.0, ex; work = zeros(2),
    )
end

function represented_cycle!(output, operator, work, execution)
    SolpropJulia.operator_exponent!(output, operator, 0.25, 1.75, execution; work)
    return nothing
end

@testset "Constant operator endpoint conversion precedes subtraction" begin
    ex = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 1, true)
    operator = SolpropJulia.DiagonalOperator(ex, fill(-2.0im, 2, 1))
    output = zeros(ComplexF64, 2, 1)
    for (left, right) in (
            (typemin(Int), typemax(Int)), (typemax(Int), typemin(Int)),
            (UInt(0), typemax(UInt)), (typemax(UInt), UInt(0)),
        )
        SolpropJulia.operator_exponent!(output, operator, left, right, ex)
        expected = -2im * (BigInt(right) - BigInt(left))
        # Two rounded endpoint conversions and one subtraction; scale by both
        # absolute endpoints, not the cancellation-reduced interval.
        bound = 8eps(Float64) * (abs(BigFloat(left)) + abs(BigFloat(right)))
        @test bound > 0
        @test all(abs.(Complex{BigFloat}.(output) .- expected) .<= bound)
    end
    SolpropJulia.operator_exponent!(output, operator, typemax(Int), typemax(Int), ex)
    @test iszero(output)
    fill!(output, 3 + 2im)
    saved = copy(output)
    @test_throws ArgumentError SolpropJulia.operator_exponent!(
        output, operator, -(big(2)^2048), big(2)^2048, ex,
    )
    @test output == saved
end

@testset "Represented and expanded diagonal flow" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    coefficients = zeros(2, 3, 2)
    coefficients[1, :, 1] = [3.125, 2.5, 0.375]
    coefficients[1, :, 2] = [11.125, 5.5, 0.375]
    coefficients[2, :, 1] = [0.5, 0.5, 0.0]
    coefficients[2, :, 2] = [1.5, 0.5, 0.0]
    weights = SolpropJulia.ChebyshevTable(ex, [0.0, 1.0, 2.0], coefficients)
    terms = ComplexF64[
        -row / 7 + im * column / 11 + term / 17
            for row in 1:3, column in 1:2, term in 1:2
    ]
    expansion = SolpropJulia.ExpansionOperator(ex, terms, weights)
    @test !Base.mightalias(terms, expansion.terms)
    work = zeros(2)
    output = zeros(ComplexF64, 3, 2)
    represented_cycle!(output, expansion, work, ex)
    expected = 12terms[:, :, 1] + 1.5terms[:, :, 2]
    # Two weighted complex terms, with a 32eps absolute product-sum bound.
    scale = 12abs.(terms[:, :, 1]) + 1.5abs.(terms[:, :, 2])
    @test all(abs.(output - expected) .<= 32eps(Float64) .* scale)
    @test @allocated(represented_cycle!(output, expansion, work, ex)) == 0
    represented_coefficients = zeros(ComplexF64, 6, 3, 2)
    for interval in 1:2, degree in 1:3, column in 1:2, row in 1:3
        represented_coefficients[row + 3(column - 1), degree, interval] =
            sum(
            terms[row, column, term] * coefficients[term, degree, interval]
                for term in 1:2
        )
    end
    cumulative = SolpropJulia.ChebyshevTable(ex, [0.0, 1.0, 2.0], represented_coefficients)
    represented = SolpropJulia.RepresentedOperator(ex, cumulative, 3, 2)
    represented_cycle!(output, represented, nothing, ex)
    @test all(abs.(output - expected) .<= 128eps(Float64) .* scale)
    @test @allocated(represented_cycle!(output, represented, nothing, ex)) == 0
    SolpropJulia.operator_exponent!(output, represented, 1.0, 1.0, ex)
    @test iszero(output)
    SolpropJulia.operator_exponent!(output, expansion, 1.0, 1.0, ex; work)
    @test iszero(output)
    @test_throws DimensionMismatch SolpropJulia.operator_exponent!(
        output, expansion, 0.0, 1.0, ex; work = zeros(3),
    )
    @test_throws ArgumentError SolpropJulia.operator_exponent!(
        output, represented, -1.0, 1.0, ex,
    )
    @test_throws DimensionMismatch SolpropJulia.RepresentedOperator(ex, cumulative, 2, 2)
end

@testset "Diagonal flow and stable increments" begin
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    for T in (Float64, Float32)
        rate = Complex{T}[-row / 16 + im * column / 7 for row in 1:8, column in 1:3]
        operator = SolpropJulia.DiagonalOperator(ex, rate)
        @test operator.rate == rate
        @test !Base.mightalias(operator.rate, rate)
        rate[1] = 0
        @test !iszero(operator.rate[1])
        exponent = similar(rate)
        state = Complex{T}[row / 7 + im * column / 9 for row in 1:8, column in 1:3]
        output = similar(state)
        inverse = similar(state)
        SolpropJulia.operator_exponent!(exponent, operator, T(0.25), T(0.75), ex)
        @test exponent == operator.rate / 2
        SolpropJulia.propagate!(output, state, exponent, ex)
        reference = setprecision(BigFloat, 256) do
            exp.(Complex{BigFloat}.(exponent)) .* Complex{BigFloat}.(state)
        end
        # Complex exponential and product: a conservative 32eps magnitude budget.
        @test all(abs.(output - reference) .<= 32eps(T) .* abs.(reference))
        SolpropJulia.propagator_pair!(output, inverse, exponent, ex)
        @test maximum(abs, output .* inverse .- 1) <= 32eps(T)
        operator_cycle!(output, inverse, exponent, state, operator, ex)
        allocations = @allocated operator_cycle!(
            output, inverse, exponent, state, operator, ex,
        )
        @test allocations == 0
        SolpropJulia.operator_exponent!(exponent, operator, T(0.5), T(0.5), ex)
        @test iszero(exponent)
        state[1] = complex(-zero(T), -zero(T))
        SolpropJulia.propagate!(output, state, exponent, ex)
        @test isequal(output, state)
        SolpropJulia.linear_increment!(output, state, exponent, ex)
        @test iszero(output)
    end
    for attenuation in (0.0, -1.0e-30, 1.0e-14, -1.0e-6, -0.3),
            angle in (0.0, 1.0e-30, -1.0e-14, 1.0e-6, -0.7)
        phase = complex(attenuation, angle)
        increment = SolpropJulia._complex_expm1(phase)
        reference = setprecision(BigFloat, 512) do
            exp(Complex{BigFloat}(phase)) - 1
        end
        # The stable decomposition evaluates two transcendental contributions and
        # their sum; bound each component by their absolute contribution magnitudes.
        real_scale = abs(expm1(BigFloat(attenuation)) * cos(BigFloat(angle))) +
            2sin(BigFloat(angle) / 2)^2
        imaginary_scale = abs(exp(BigFloat(attenuation)) * sin(BigFloat(angle)))
        @test abs(real(increment) - real(reference)) <= 32eps(Float64) * real_scale
        @test abs(imag(increment) - imag(reference)) <= 32eps(Float64) * imaginary_scale
        state = reshape([2.0 - 3.0im], 1, 1)
        exponent = reshape([phase], 1, 1)
        output = similar(state)
        SolpropJulia.linear_increment!(output, state, exponent, ex)
        @test abs(output[1] - reference * state[1]) <=
            64eps(Float64) * (real_scale + imaginary_scale) * abs(state[1])
    end
    @test_throws ArgumentError SolpropJulia.DiagonalOperator(
        ex, fill(ComplexF64(NaN), 2, 2),
    )
    @test_throws ArgumentError SolpropJulia.DiagonalOperator(ex, zeros(2, 2))
    operator = SolpropJulia.DiagonalOperator(ex, zeros(ComplexF64, 2, 2))
    @test_throws DimensionMismatch SolpropJulia.operator_exponent!(
        zeros(ComplexF64, 1, 2), operator, 0.0, 1.0, ex,
    )
    @test_throws ArgumentError SolpropJulia.operator_exponent!(
        zeros(ComplexF64, 2, 2), operator, NaN, 1.0, ex,
    )
    array = zeros(ComplexF64, 2, 2)
    @test_throws ArgumentError SolpropJulia.propagator_pair!(array, array, copy(array), ex)
end
