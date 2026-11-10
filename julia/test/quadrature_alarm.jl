# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function _quadrature_reference(fine, coarse)
    return setprecision(BigFloat, 256) do
        first = Complex{BigFloat}.(fine)
        second = Complex{BigFloat}.(coarse)
        sqrt(sum(abs2, second - first) / sum(abs2, first))
    end
end

function _quadrature_assert_rounding(actual, fine, coarse)
    reference = _quadrature_reference(fine, coarse)
    components = 2length(fine)
    unit = BigFloat(eps(Float64)) / 2
    tiny = BigFloat(nextfloat(0.0))
    # Each finite scalar difference has relative error <= u (subnormal differences
    # of binary64 operands are exact). Overflow-safe half-differences satisfy the
    # same bound. The independent norm scales cancel algebraically. Normalization,
    # component squares, row/column sums, quotient, square root and final scale
    # combination fit gamma_(8N+32), with N real components. Each normalized norm
    # sum is >= 1, so 8N*eta also bounds discarded subnormal products/divisions.
    # Final gradual underflow contributes at most eta/2; eta is an outward allowance.
    count = 8components + 32
    gamma = count * unit / (1 - count * unit)
    bound = (gamma + 8components * tiny) * abs(reference) + tiny
    @test bound > 0
    @test isfinite(actual)
    @test abs(BigFloat(actual) - reference) <= bound
    return nothing
end

function _quadrature_allocations(work, fine, coarse, execution)
    SolpropJulia.quadrature_error!(work, fine, coarse, execution)
    return @allocated SolpropJulia.quadrature_error!(work, fine, coarse, execution)
end

@testset "Embedded source norm quotient and independent analytic vectors" begin
    for tasks in (1, 2)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        for (rows, columns) in ((1, 1), (7, 3), (9, 5))
            fine = ComplexF64[
                complex(sin(0.3row + column), cos(row - 0.2column))
                    for row in 1:rows, column in 1:columns
            ]
            work = SolpropJulia.QuadratureWorkspace(execution, columns)
            for factor in (0.0, 0.5, 2.0, -1.0)
                coarse = factor * fine
                actual = SolpropJulia.quadrature_error!(work, fine, coarse, execution)
                # Dyadic scaling here is exact. Homogeneity gives |factor - 1|;
                # the independent norm witness additionally checks its rounding.
                @test abs(actual - abs(factor - 1)) <= 32eps(Float64)
                _quadrature_assert_rounding(actual, fine, coarse)
            end
            coarse = ComplexF64[
                complex(cos(0.7row - column), sin(row + 0.4column))
                    for row in 1:rows, column in 1:columns
            ]
            before_fine, before_coarse = copy(fine), copy(coarse)
            actual = SolpropJulia.quadrature_error!(work, fine, coarse, execution)
            _quadrature_assert_rounding(actual, fine, coarse)
            @test fine == before_fine && coarse == before_coarse
            @test _quadrature_allocations(work, fine, coarse, execution) == 0

            # Read-only source aliases are allowed; an exact equal pair is structural.
            @test SolpropJulia.quadrature_error!(work, fine, fine, execution) == 0.0
            @test _quadrature_allocations(work, fine, fine, execution) == 0
            coarse = complex.(nextfloat.(real.(fine)), prevfloat.(imag.(fine)))
            _quadrature_assert_rounding(
                SolpropJulia.quadrature_error!(work, fine, coarse, execution), fine, coarse
            )
        end
    end
end

@testset "Quadrature scaling avoids gratuitous underflow and overflow" begin
    for tasks in (1, 2)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true
        )
        work = SolpropJulia.QuadratureWorkspace(execution, 3)
        for exponent in (-1074, -1022, -500, 0, 500, 1022)
            scale = ldexp(1.0, exponent)
            fine = fill(complex(scale, scale), 4, 3)
            coarse = 2fine
            actual = SolpropJulia.quadrature_error!(work, fine, coarse, execution)
            _quadrature_assert_rounding(actual, fine, coarse)
            @test abs(actual - 1.0) <= 32eps(Float64)
        end
        fine = fill(complex(floatmax(Float64), -floatmax(Float64)), 4, 3)
        coarse = -fine
        @test any(value -> !isfinite(value), coarse - fine)
        actual = SolpropJulia.quadrature_error!(work, fine, coarse, execution)
        _quadrature_assert_rounding(actual, fine, coarse)
        @test abs(actual - 2.0) <= 32eps(Float64)
        @test _quadrature_allocations(work, fine, coarse, execution) == 0

        # Mixed overflowing and ordinary differences exercise both the intra-column
        # and inter-column maximum/exponent merges in either encounter order.
        fine = ComplexF64[
            1.0 floatmax(Float64) 2.0;
            floatmax(Float64) 2.0 3.0;
            4.0 3.0 -floatmax(Float64)
        ]
        coarse = -fine
        for first in (fine, reverse(fine; dims = 2))
            second = -first
            _quadrature_assert_rounding(
                SolpropJulia.quadrature_error!(work, first, second, execution),
                first, second
            )
        end

        work = SolpropJulia.QuadratureWorkspace(execution, 1)
        fine = fill(ComplexF64(0.1), 1024, 1)
        coarse = copy(fine)
        coarse[1] = 1.0e308
        # max(abs,coarse)/max(abs,fine) overflows although the norm ratio is finite.
        @test !isfinite(1.0e308 / 0.1)
        _quadrature_assert_rounding(
            SolpropJulia.quadrature_error!(work, fine, coarse, execution), fine, coarse
        )

        # Scaling both inputs first would discard every changed small component.
        # Independent difference scaling retains a representable subnormal norm ratio.
        fine = zeros(ComplexF64, 1025, 1)
        fine[1] = 1.0e300
        coarse = copy(fine)
        coarse[2:end] .= 1.0e-25
        actual = SolpropJulia.quadrature_error!(work, fine, coarse, execution)
        @test actual > 0
        _quadrature_assert_rounding(actual, fine, coarse)

        fine = fill(ComplexF64(nextfloat(0.0)), 1, 1)
        coarse = ones(ComplexF64, 1, 1)
        @test_throws ArgumentError SolpropJulia.quadrature_error!(
            work, fine, coarse, execution
        )
    end
end

@testset "Unavailable quadrature, poisoned reuse and strided inputs" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    work = SolpropJulia.QuadratureWorkspace(execution, 2)
    fine = reshape(complex.(1.0:12.0, 12.0:-1.0:1.0), 3, 4)
    coarse = -fine
    first, second = @view(fine[:, 1:2:4]), @view(coarse[:, 1:2:4])
    for _ in 1:2
        fill!(work.maxima, NaN)
        fill!(work.sums, NaN)
        @test isnan(SolpropJulia.quadrature_error!(work, nothing, nothing, execution))
        @test isnan(SolpropJulia.quadrature_error!(work, zero(first), first, execution))
        @test isnan(
            SolpropJulia.quadrature_error!(work, zero(first), zero(first), execution)
        )
        @test SolpropJulia.quadrature_error!(work, first, first, execution) == 0.0
        _quadrature_assert_rounding(
            SolpropJulia.quadrature_error!(work, first, second, execution), first, second
        )
        @test all(isfinite, work.maxima) && all(isfinite, work.sums)
    end
    @test _quadrature_allocations(work, first, second, execution) == 0
    @test _quadrature_allocations(work, nothing, nothing, execution) == 0
end

@testset "Quadrature admission refuses before writing scratch" begin
    execution = SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), 2, true)
    @test_throws ArgumentError SolpropJulia.QuadratureWorkspace(execution, 0)
    @test_throws ArgumentError SolpropJulia.QuadratureWorkspace(execution, -1)
    work = SolpropJulia.QuadratureWorkspace(execution, 4)
    fill!(work.maxima, 7.0)
    fill!(work.sums, 9.0)
    fine = ones(ComplexF64, 1, 4)
    for invalid in (NaN, Inf, -Inf)
        bad = copy(fine)
        bad[3] = complex(1.0, invalid)
        for (first, second) in ((bad, fine), (fine, bad), (zero(fine), bad))
            @test_throws ArgumentError SolpropJulia.quadrature_error!(
                work, first, second, execution
            )
            @test all(==(7.0), work.maxima) && all(==(9.0), work.sums)
        end
    end
    for (first, second) in (
            (nothing, fine), (fine, nothing), (real(fine), real(fine)),
            (ComplexF32.(fine), ComplexF32.(fine)),
        )
        @test_throws ArgumentError SolpropJulia.quadrature_error!(
            work, first, second, execution
        )
    end
    for (first, second) in (
            (fine, zeros(ComplexF64, 2, 4)),
            (zeros(ComplexF64, 0, 4), zeros(ComplexF64, 0, 4)),
            (ones(ComplexF64, 4, 1), ones(ComplexF64, 4, 1)),
        )
        @test_throws DimensionMismatch SolpropJulia.quadrature_error!(
            work, first, second, execution
        )
    end
    for aliased in (
            reshape(reinterpret(ComplexF64, vec(work.sums)), 1, 4),
            reshape(reinterpret(ComplexF64, @view(vec(work.maxima)[1:8])), 1, 4),
        )
        for (first, second) in ((aliased, fine), (fine, aliased))
            @test_throws ArgumentError SolpropJulia.quadrature_error!(
                work, first, second, execution
            )
            @test all(==(7.0), work.maxima) && all(==(9.0), work.sums)
        end
    end
    malformed = SolpropJulia.QuadratureWorkspace(zeros(2, 4), zeros(2, 4))
    @test_throws DimensionMismatch SolpropJulia.quadrature_error!(
        malformed, fine, fine, execution
    )
    shared = zeros(3, 4)
    aliased = SolpropJulia.QuadratureWorkspace(shared, @view(shared[1:2, :]))
    @test_throws ArgumentError SolpropJulia.quadrature_error!(
        aliased, fine, fine, execution
    )
end
