# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

function phi_reference(argument, order)
    return setprecision(BigFloat, 512) do
        point = Complex{BigFloat}(argument)
        term = one(point) / factorial(big(order))
        total = term
        # Direct infinite-series definition, without binary64's branch, downward
        # recurrence or doubling. 512 bits cover cancellation for this |x|<=64 set.
        for index in 1:1000
            term *= point / (index + order)
            total += term
            abs(term) < eps(BigFloat) * abs(total) && break
        end
        total
    end
end

function phi_batch!(output, arguments, execution)
    SolpropJulia.each!(execution, length(arguments), 1) do row, unused
        result = SolpropJulia._phi_values(arguments[row], Val(8))
        for order in 1:9
            output[row, order] = result[order]
        end
    end
    return nothing
end

@testset "Protected exponential phi functions" begin
    points = ComplexF64[
        0, 1.0e-30 + 1.0e-28im, -1.0e-10, 0.5im, -0.25 + 0.25im,
        prevfloat(3.0), 3, nextfloat(3.0), -3, 3im, -3im,
        -2 + 2im, 4, -4, 2pi * im, -40 + 5im, 20im, 32 + 32im,
    ]
    for order in 0:8
        for point in points
            result = SolpropJulia.phi_values(point, Val(order))
            @test length(result) == order + 1
            for degree in 0:order
                reference = phi_reference(point, degree)
                # At most eight cancellation-sensitive upward recurrences around
                # join 3, or three doublings and 24 Taylor terms. An absolute
                # 512eps budget against max(|phi|,1/k!) also covers phi near a zero.
                scale = max(abs(reference), inv(Float64(factorial(degree))))
                @test abs(result[degree + 1] - reference) <= 512eps(Float64) * scale
            end
        end
        zero_result = SolpropJulia.phi_values(0.0 + 0.0im, Val(order))
        @test zero_result == ntuple(
            index -> ComplexF64(1 / factorial(index - 1)), order + 1,
        )
    end
    for point in (ComplexF64(-1000), ComplexF64(700), -1.0e300 + 1.0e300im)
        result = SolpropJulia.phi_values(point, Val(8))
        @test all(isfinite, result)
        @test abs(result[2] * point - expm1(point)) <=
            16eps(Float64) * max(1, abs(expm1(point)))
    end
    for argument in (ComplexF64(NaN), ComplexF64(Inf), ComplexF64(701))
        @test_throws ArgumentError SolpropJulia.phi_values(argument, Val(8))
    end
    for order in (-1, 9, 1.0)
        @test_throws ArgumentError SolpropJulia.phi_values(1.0 + 0im, Val(order))
    end
    ex = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    output = zeros(ComplexF64, length(points), 9)
    phi_batch!(output, points, ex)
    @test @allocated(phi_batch!(output, points, ex)) == 0
    for row in eachindex(points)
        @test Tuple(output[row, :]) == SolpropJulia.phi_values(points[row], Val(8))
    end
end
