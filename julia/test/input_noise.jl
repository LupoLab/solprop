# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

function _noise_execution(tasks)
    return SolpropJulia.KernelExecution(SolpropJulia.KernelAbstractions.CPU(), tasks, true)
end

function _noise_fixture(execution; shape = (5, 2), launch = 0.25)
    spectrum = ComplexF64[
        (row - 3column) / 16 + im * (2row + column) / 32
            for row in 1:shape[1], column in 1:shape[2]
    ]
    rate = ComplexF64[
        -row / 8 + im * (row - 2column) / 4
            for row in 1:shape[1], column in 1:shape[2]
    ]
    operator = SolpropJulia.DiagonalOperator(execution, rate)
    return SolpropJulia.InputNoise(execution, spectrum, launch, operator), spectrum
end

function _noise_allocations(field, work, active, z, resume, execution)
    SolpropJulia.apply_input_noise!(field, work, active, z, resume, execution)
    return @allocated SolpropJulia.apply_input_noise!(
        field, work, active, z, resume, execution
    )
end

function _noise_phase_allocations(work, z, execution)
    SolpropJulia.input_noise_phase!(work, z, execution)
    return @allocated SolpropJulia.input_noise_phase!(work, z, execution)
end

function _noise_table(execution, count; complex_values = true)
    real_coefficients = zeros(Float64, count, 3, 2)
    for interval in 1:2, row in 1:count
        real_coefficients[row, :, interval] .= (2 + row / 8, 0.25, -0.125)
    end
    coefficients = complex_values ?
        complex.(-2 .* real_coefficients, real_coefficients) : real_coefficients
    return SolpropJulia.ChebyshevTable(execution, [0.0, 0.5, 1.0], coefficients)
end

function _noise_polynomial_reference(table, z)
    interval = z < 0.5 ? 1 : 2
    lo, hi = BigFloat.(table.breakpoints[interval:(interval + 1)])
    x = (2BigFloat(z) - lo - hi) / (hi - lo)
    return [
        Complex{BigFloat}(table.coefficients[row, 1, interval]) +
            Complex{BigFloat}(table.coefficients[row, 2, interval]) * x +
            Complex{BigFloat}(table.coefficients[row, 3, interval]) * (2x^2 - 1)
            for row in axes(table.coefficients, 1)
    ]
end

@testset "Input-noise launch predicate and exact masking" begin
    for tasks in (1, 2), shape in ((5, 1), (8, 3)), launch in (0.0, 0.25)
        execution = _noise_execution(tasks)
        work, original = _noise_fixture(execution; shape, launch)
        retained = copy(original)
        original .= NaN
        @test work.spectrum == retained
        @test !Base.mightalias(work.spectrum, original)
        active = [isodd(row) for row in 1:shape[1]]
        initial = ComplexF64[
            row / 7 + im * column / 11 for row in 1:shape[1], column in 1:shape[2]
        ]
        for resume in (false, true), z in (launch, launch + 0.125)
            field = copy(initial)
            added = SolpropJulia.apply_input_noise!(
                field, work, active, z, resume, execution
            )
            @test added === (!resume && z == launch)
            for column in axes(field, 2), row in axes(field, 1)
                if !active[row]
                    @test iszero(field[row, column])
                elseif added
                    reference = Complex{BigFloat}(initial[row, column]) +
                        Complex{BigFloat}(retained[row, column])
                    # One addition per real/imaginary component: the complex
                    # absolute error is bounded by 2eps*(|field|+|draw|).
                    scale = abs(initial[row, column]) + abs(retained[row, column])
                    bound = 2eps(Float64) * scale
                    @test bound > 0 && abs(field[row, column] - reference) <= bound
                else
                    @test field[row, column] == initial[row, column]
                end
            end
            @test work.spectrum == retained
            @test _noise_allocations(field, work, active, z, resume, execution) == 0
        end
        field = copy(initial)
        @test SolpropJulia.apply_input_noise!(field, work, active, launch, false, execution)
        launched = copy(field)
        @test !SolpropJulia.apply_input_noise!(field, work, active, launch, true, execution)
        @test field == launched
        ledger = SolpropJulia.input_noise_memory(work)
        @test ledger == (
            spectrum = 16prod(shape), scratch = 16prod(shape),
            phase = 8prod(shape), weights = 0,
        )
        @test sum(ledger) == 40prod(shape)
    end
end

@testset "Input-noise transactional failure and storage admission" begin
    execution = _noise_execution(2)
    work, _ = _noise_fixture(execution)
    field = fill(2.0 + 3.0im, size(work.spectrum))
    original = copy(field)
    active = fill(true, size(field, 1))
    for z in (-1.0, Inf, NaN)
        @test_throws ArgumentError SolpropJulia.apply_input_noise!(
            field, work, active, z, false, execution
        )
        @test field == original
    end
    @test_throws DimensionMismatch SolpropJulia.apply_input_noise!(
        field, work, Bool[true], 0.25, false, execution
    )
    @test_throws DimensionMismatch SolpropJulia.apply_input_noise!(
        zeros(ComplexF64, 3, 1), work, active, 0.25, false, execution
    )
    @test_throws ArgumentError SolpropJulia.apply_input_noise!(
        ComplexF32.(field), work, active, 0.25, false, execution
    )
    @test_throws ArgumentError SolpropJulia.apply_input_noise!(
        field, work, ones(5), 0.25, false, execution
    )
    @test_throws ArgumentError SolpropJulia.apply_input_noise!(
        field, work, trues(5), 0.25, false, execution
    )
    @test field == original
    for retained in (work.spectrum, work.scratch, work.operator.rate)
        saved = copy(retained)
        @test_throws ArgumentError SolpropJulia.apply_input_noise!(
            retained, work, active, 0.25, false, execution
        )
        @test retained == saved
    end
    # Independent unsafe wrappers have different data identities. The shared byte
    # boundary must still refuse before changing the spectrum or aliased mask.
    GC.@preserve work field begin
        foreign = unsafe_wrap(Array, pointer(work.spectrum), size(field); own = false)
        @test_throws ArgumentError SolpropJulia.apply_input_noise!(
            foreign, work, active, 0.25, false, execution
        )
        mask = unsafe_wrap(Array, Ptr{Bool}(pointer(field)), (5,); own = false)
        @test_throws ArgumentError SolpropJulia.apply_input_noise!(
            field, work, mask, 0.25, false, execution
        )
        @test field == original
        mask = unsafe_wrap(Array, Ptr{Bool}(pointer(work.scratch)), (5,); own = false)
        @test_throws ArgumentError SolpropJulia.apply_input_noise!(
            field, work, mask, 0.25, false, execution
        )
        @test field == original
    end
    indexed = @view field[[1, 2, 3, 4, 5], :]
    @test_throws ArgumentError SolpropJulia.apply_input_noise!(
        indexed, work, active, 0.25, false, execution
    )
    @test field == original
    reversed = @view field[end:-1:1, :]
    @test SolpropJulia.apply_input_noise!(reversed, work, active, 0.25, false, execution)
    @test all(isfinite, field)
    field .= original
    field[end] = NaN
    poisoned = copy(field)
    @test_throws ArgumentError SolpropJulia.apply_input_noise!(
        field, work, active, 0.25, false, execution
    )
    @test isequal(field, poisoned)
    field .= original
    work.spectrum[end] = Inf
    @test_throws ArgumentError SolpropJulia.apply_input_noise!(
        field, work, active, 0.25, true, execution
    )
    @test field == original
    huge = SolpropJulia.InputNoise(
        execution, fill(complex(floatmax(Float64), 0), size(field)), 0.25, work.operator,
    )
    field .= complex(floatmax(Float64), 0)
    before = copy(field)
    @test_throws ArgumentError SolpropJulia.apply_input_noise!(
        field, huge, active, 0.25, false, execution
    )
    @test field == before
    @test all(isfinite, huge.spectrum)
    # A fully masked bin is an exact structural zero: unused addition may not
    # overflow the candidate. A resume must not perform that addition either.
    @test SolpropJulia.apply_input_noise!(
        field, huge, fill(false, 5), 0.25, false, execution
    )
    @test all(iszero, field)
    field .= before
    @test !SolpropJulia.apply_input_noise!(field, huge, active, 0.25, true, execution)
    @test field == before
end

@testset "Input-noise cumulative phase uses the existing absolute anchor" begin
    for tasks in (1, 2)
        execution = _noise_execution(tasks)
        constant, _ = _noise_fixture(execution)
        table = _noise_table(execution, 10)
        represented = SolpropJulia.InputNoise(
            execution, constant.spectrum, 0.25,
            SolpropJulia.RepresentedOperator(execution, table, 5, 2),
        )
        weights = _noise_table(execution, 3; complex_values = false)
        terms = ComplexF64[
            -row / 8 + im * (row - 2column + term) / 16
                for row in 1:5, column in 1:2, term in 1:3
        ]
        expansion = SolpropJulia.InputNoise(
            execution, constant.spectrum, 0.25,
            SolpropJulia.ExpansionOperator(execution, terms, weights),
        )
        @test expansion.operator.terms !== terms
        @test SolpropJulia.input_noise_memory(expansion).weights == 24
        for z in (0.0, 0.125, 0.25, 0.5, 0.625, 1.0)
            phase = SolpropJulia.input_noise_phase!(constant, z, execution)
            reference = imag.(Complex{BigFloat}.(constant.operator.rate)) .* BigFloat(z)
            # One real product per phase; the positive absolute allowance also
            # covers exact zero coefficients without invoking a relative quotient.
            bound = 2eps(Float64) * max(1, maximum(abs, reference))
            @test phase === constant.phase
            @test bound > 0 && maximum(abs, phase - reference) <= bound
            phase = SolpropJulia.input_noise_phase!(represented, z, execution)
            reference = reshape(imag.(_noise_polynomial_reference(table, z)), 5, 2)
            # The selected positions give an exact dyadic interval map. Degree-2
            # Clenshaw has fewer than 8 rounded operations, so gamma_16 times the
            # coefficient one-norm bounds its absolute imaginary-part error.
            gamma = 16eps(Float64) / (1 - 16eps(Float64))
            scale = maximum(sum(abs.(imag.(table.coefficients)); dims = 2))
            bound = gamma * scale
            @test bound > 0 && maximum(abs, phase - reference) <= bound
            phase = SolpropJulia.input_noise_phase!(expansion, z, execution)
            factors = real.(_noise_polynomial_reference(weights, z))
            reference = [
                sum(BigFloat(imag(terms[row, column, k])) * factors[k] for k in 1:3)
                    for row in 1:5, column in 1:2
            ]
            # Each weight has the preceding gamma_16 bound. Three real products
            # and additions require gamma_32 on the term/coefficient one-norm sum;
            # this remains absolute when phases cancel or are exactly zero.
            scale = maximum(
                sum(
                    abs(imag(terms[row, column, k])) *
                        sum(abs, @view weights.coefficients[k, :, interval])
                        for k in 1:3
                ) for row in 1:5, column in 1:2, interval in 1:2
            )
            bound = 32eps(Float64) / (1 - 32eps(Float64)) * scale
            @test bound > 0 && maximum(abs, phase - reference) <= bound
        end
        # A represented offset survives at zero and at a fresh interior launch;
        # subtracting launch phase would violate the shared cumulative contract.
        @test any(!iszero, SolpropJulia.input_noise_phase!(represented, 0.0, execution))
        @test any(!iszero, SolpropJulia.input_noise_phase!(expansion, 0.25, execution))
        for work in (constant, represented, expansion)
            @test _noise_phase_allocations(work, 0.25, execution) == 0
            for z in (-1.0, Inf, NaN)
                @test_throws ArgumentError SolpropJulia.input_noise_phase!(
                    work, z, execution
                )
            end
        end
        for work in (represented, expansion)
            for z in (nextfloat(1.0), 2.0)
                @test_throws ArgumentError SolpropJulia.input_noise_phase!(
                    work, z, execution
                )
            end
        end
    end
end

@testset "Input-noise preparation and nonfinite phase refusals" begin
    execution = _noise_execution(1)
    work, _ = _noise_fixture(execution)
    for launch in (-1.0, Inf, NaN)
        @test_throws ArgumentError SolpropJulia.InputNoise(
            execution, work.spectrum, launch, work.operator
        )
    end
    for spectrum in (ComplexF32.(work.spectrum), zeros(5, 2), zeros(ComplexF64, 0, 2))
        @test_throws ArgumentError SolpropJulia.InputNoise(
            execution, spectrum, 0.0, work.operator
        )
    end
    for value in (NaN, Inf)
        @test_throws ArgumentError SolpropJulia.InputNoise(
            execution, fill(complex(value), 5, 2), 0.0, work.operator
        )
    end
    @test_throws DimensionMismatch SolpropJulia.InputNoise(
        execution, zeros(ComplexF64, 3, 2), 0.0, work.operator
    )
    @test_throws ArgumentError SolpropJulia.InputNoise(
        execution, work.spectrum, 0.0, nothing
    )
    single = SolpropJulia.DiagonalOperator(execution, zeros(ComplexF32, 5, 2))
    @test_throws ArgumentError SolpropJulia.InputNoise(
        execution, work.spectrum, 0.0, single
    )
    real_table = _noise_table(execution, 10; complex_values = false)
    real_only = SolpropJulia.InputNoise(
        execution, work.spectrum, 0.0,
        SolpropJulia.RepresentedOperator(execution, real_table, 5, 2),
    )
    @test all(iszero, SolpropJulia.input_noise_phase!(real_only, 0.25, execution))
    single_weights = SolpropJulia.ChebyshevTable(
        execution, Float32[0, 1], ones(Float32, 1, 1, 1)
    )
    @test_throws ArgumentError SolpropJulia.InputNoise(
        execution, work.spectrum, 0.0,
        SolpropJulia.ExpansionOperator(
            execution, ones(ComplexF64, 5, 2, 1), single_weights,
        ),
    )
    huge = SolpropJulia.InputNoise(
        execution, work.spectrum, 0.0,
        SolpropJulia.DiagonalOperator(
            execution, fill(complex(0, floatmax(Float64)), 5, 2)
        ),
    )
    @test_throws ArgumentError SolpropJulia.input_noise_phase!(huge, 2.0, execution)
    loss_only = SolpropJulia.InputNoise(
        execution, zeros(ComplexF64, 5, 2), 0.25,
        SolpropJulia.DiagonalOperator(
            execution, fill(complex(-floatmax(Float64), 0), 5, 2)
        ),
    )
    @test all(iszero, SolpropJulia.input_noise_phase!(loss_only, 2.0, execution))
end
