# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

struct AdamsPrecisionSource
    multiplier::ComplexF32
    offset::ComplexF32
end
function (source::AdamsPrecisionSource)(output, state, position)
    for index in eachindex(output)
        output[index] = source.multiplier * state[index] + source.offset
    end
    return nothing
end

struct AdamsPrecisionOperator
    value::ComplexF64
    slope::ComplexF64
end
function (operator::AdamsPrecisionOperator)(output, first, last)
    fill!(output, (last - first) * (operator.value + operator.slope * (first + last) / 2))
    return nothing
end
function (operator::AdamsPrecisionOperator)(output, position, before::Bool)
    fill!(output, operator.value + operator.slope * position)
    return nothing
end

function adams_precision_fixture(execution, order; compensated = false)
    state = ComplexF32[(row + im * column) / 8 for row in 1:3, column in 1:2]
    source = AdamsPrecisionSource(ComplexF32(0.125, -0.25), ComplexF32(0.25, 0.125))
    operator = AdamsPrecisionOperator(-0.25 + 0.5im, 0.125 + 0.25im)
    history = SolpropJulia.AdamsHistory(execution, state, 6; compensated)
    work = SolpropJulia.AdamsWork(execution, state, 6; compensated)
    stencil = SolpropJulia.AdamsStencil(6)
    for slot in 1:6
        sampled = state .* Float32(1 + slot / 8)
        SolpropJulia.adams_seed!(
            history, slot, sampled, -(slot - 1) * 0.125, source, operator, execution,
        )
    end
    scores = order == 1 ? (1, 2) : order == 6 ? (5, 6) : (order - 1, order, order + 1)
    SolpropJulia.adams_stencil!(stencil, order, 1:maximum(scores), scores)
    return (; state, source, operator, history, work, stencil)
end

function adams_precision_trial!(fixture, execution)
    return SolpropJulia.adams_trial!(
        fixture.work, fixture.state, 0.0, 0.125, fixture.history, fixture.stencil,
        fixture.source, fixture.operator, fixture.operator, execution,
    )
end

# Independent defining-series integral of t^(power-1) exp((upper-t)*phase).
# |phase|<1 in these fixtures; 160 terms leave <2^-900 absolute tail.
function adams_precision_moment(phase, power, upper = BigFloat(1))
    term = upper^power / power
    total = complex(term)
    for degree in 1:160
        term *= phase * upper / (power + degree)
        total += term
    end
    return total
end

function adams_precision_reference(
        fixture, row, column; predictor = false,
        fraction = BigFloat(1), dense = false
    )
    work, history, stencil = fixture.work, fixture.history, fixture.stencil
    phase, step = Complex{BigFloat}(work.exponent[row, column]), BigFloat(0.125)
    count = stencil.order + (predictor ? 0 : 1)
    coefficients = predictor ? work.coefficients.predictor : work.coefficients.corrector
    compensation = predictor ? work.coefficients.compensation_predictor :
        work.coefficients.compensation_corrector
    result = exp(fraction * phase) * Complex{BigFloat}(fixture.state[row, column])
    contributions = abs(result)
    for sample in 1:count
        newest = !predictor && sample == 1
        slot = newest ? 1 : stencil.slots[sample - (predictor ? 0 : 1)]
        value = Complex{BigFloat}(
            newest ? work.source[row, column] :
                history.source[row, column, slot]
        )
        weight = zero(Complex{BigFloat})
        weight_scale = BigFloat(0)
        for power in 1:count
            term = BigFloat(coefficients[sample, power]) *
                adams_precision_moment(phase, power, fraction)
            weight += term
            weight_scale += abs(term)
        end
        result += step * weight * value
        contributions += abs(step) * weight_scale * abs(value)
        if work.compensated
            node = newest ? BigFloat(1) : BigFloat(stencil.nodes[slot])
            rate = Complex{BigFloat}(
                newest ? work.endpoint_rate[row, column] :
                    history.rates[row, column, slot]
            )
            sampled = Complex{BigFloat}(
                newest ? work.predictor[row, column] :
                    history.states[row, column, slot]
            )
            ordinary = if dense
                sum(BigFloat(coefficients[sample, p]) * fraction^p / p for p in 1:count)
            else
                BigFloat(compensation[sample])
            end
            transport = exp((fraction - node) * phase)
            result += ordinary * transport * (step * rate - phase) * sampled
            contributions += abs(ordinary * transport * sampled) *
                (abs(step * rate) + abs(phase))
        end
    end
    # At most seven sample products, complex additions, coefficient narrowings,
    # and 49 wide moment terms. gamma_(32*(q+1)) covers the C32 graph; gamma_8192
    # covers the fixed wide graph plus the previously qualified small-argument phi.
    # The scale precedes each cancellation, including h*D-x and polynomial terms.
    unit32, unit64 = BigFloat(2)^-24, BigFloat(2)^-53
    operations = 32 * (count + 1)
    budget = (
        operations * unit32 / (1 - operations * unit32) +
            8192unit64 / (1 - 8192unit64)
    ) * contributions + BigFloat(2)^-140
    @test budget > 0
    return result, budget
end

@testset "Binary32 Adams layout, full factors, dense retention and copies" begin
    for tasks in unique((1, Threads.nthreads(:default))), compensated in (false, true)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        for order in 1:6
            fixture = adams_precision_fixture(execution, order; compensated)
            work, history, state, stencil =
                fixture.work, fixture.history, fixture.state, fixture.stencil
            saved = copy(state), copy(history.source), copy(history.rates)
            @test adams_precision_trial!(fixture, execution) == 1
            @test eltype(work.predictor) === eltype(work.defects) === ComplexF32
            @test eltype(work.exponent) === eltype(work.moments) === ComplexF64
            @test eltype(history.source) === eltype(history.states) === ComplexF32
            @test eltype(history.rates) === ComplexF64
            @test size(work.carried, 3) == 0
            @test size(work.norm_parts) == (4, 2)
            @test eltype(work.norm_scale.mantissa) === Float32
            @test (state, history.source, history.rates) == saved
            setprecision(BigFloat, 256) do
                for column in axes(state, 2), row in axes(state, 1)
                    for predictor in (false, true)
                        expected, budget = adams_precision_reference(
                            fixture, row, column; predictor,
                        )
                        actual = predictor ? work.predictor[row, column] :
                            work.candidate[row, column]
                        @test abs(Complex{BigFloat}(actual) - expected) <= budget
                    end
                end
            end
            dense = SolpropJulia.AdamsDense(execution, state, 6; compensated)
            SolpropJulia.adams_retain_dense!(
                dense, work, state, work.candidate, history, stencil, execution,
            )
            output = similar(state)
            for fraction in (0.125, 0.375, 0.875, prevfloat(1.0))
                position = fraction * 0.125
                SolpropJulia.adams_observe!(output, dense, position, execution)
                setprecision(BigFloat, 256) do
                    for column in axes(state, 2), row in axes(state, 1)
                        expected, budget = adams_precision_reference(
                            fixture, row, column;
                            fraction = BigFloat(fraction), dense = true
                        )
                        @test abs(Complex{BigFloat}(output[row, column]) - expected) <=
                            budget
                    end
                end
            end
            @test SolpropJulia.adams_observe!(output, dense, 0.0, execution) == state
            @test SolpropJulia.adams_observe!(
                output, dense, 0.125, execution,
            ) == work.candidate
            copied = SolpropJulia.AdamsHistory(execution, state, 6; compensated)
            SolpropJulia.adams_copy_history!(copied, history)
            @test copied.source == history.source && copied.source !== history.source
            @test copied.rates == history.rates && copied.rates !== history.rates
            retained = SolpropJulia.AdamsDense(execution, state, 6; compensated)
            SolpropJulia.adams_copy_dense!(retained, dense)
            @test retained.source == dense.source && retained.source !== dense.source
            @test retained.rates == dense.rates && retained.rates !== dense.rates
            for _ in 1:2
                adams_precision_trial!(fixture, execution)
                SolpropJulia.adams_retain_dense!(
                    dense, work, state, work.candidate, history, stencil, execution,
                )
                SolpropJulia.adams_observe!(output, dense, 0.0625, execution)
                SolpropJulia.adams_copy_history!(copied, history)
                SolpropJulia.adams_copy_dense!(retained, dense)
            end
            @test @allocated(adams_precision_trial!(fixture, execution)) == 0
            @test @allocated(
                SolpropJulia.adams_observe!(
                    output, dense, 0.0625, execution,
                )
            ) == 0
            @test @allocated(
                SolpropJulia.adams_retain_dense!(
                    dense, work, state, work.candidate, history, stencil, execution,
                )
            ) == 0
            @test @allocated(SolpropJulia.adams_copy_history!(copied, history)) == 0
            @test @allocated(SolpropJulia.adams_copy_dense!(retained, dense)) == 0
        end
    end
end

@testset "Exponent-separated compensation and factored coefficient range" begin
    setprecision(BigFloat, 512) do
        exact = log(BigFloat(2))
        hi, lo = SolpropJulia._ADAMS_LN2_HI, SolpropJulia._ADAMS_LN2_LO
        @test hi == Float64(exact)
        @test lo == Float64(exact - BigFloat(hi))
        @test abs(exact - BigFloat(hi) - BigFloat(lo)) <= BigFloat(2)^-109
        for argument in (-2.0^20, -1000.0, -750.0, -0.1, 0.0, 0.1, 750.0, 1000.0, 2.0^20)
            turns = round(Int, argument / hi)
            residual = fma(-Float64(turns), hi, argument) - Float64(turns) * lo
            error = abs(BigFloat(residual) + turns * exact - BigFloat(argument))
            bound = BigFloat(2)^-52 *
                (abs(BigFloat(residual)) + abs(BigFloat(turns) * lo)) +
                abs(turns) * BigFloat(2)^-109 + BigFloat(2)^-1073
            @test bound > 0 && error <= bound
        end
        for (step, rate, phase, span, coefficient) in (
                (1.0e300, 1.0e100 + 0im, -1000.0 + 0.2im, 1.0, 1.0e30),
                (1.0e-300, 1.0e-100 + 0im, 750.0 + 0.2im, 1.0, 1.0e-300),
                (1.0e-300, 1.0e-100 + 0im, 1000.0 + 0.2im, 1.0, 1.0e-300),
                (1.0e300, 1.0e100 + 0im, 0.0 + 0im, 1.0, 1.0e-300),
                (2.0, nextfloat(0.5) + 0im, 1.0 + 0im, 0.0, 1.0e30),
                (2.0, prevfloat(0.5) + 0im, 1.0 + 0im, 0.0, 1.0e30),
                (1.0, 0.0 + 0im, -2.0^21 + 0im, 1.0, 1.0e308),
            )
            actual = SolpropJulia._adams_compensation_factor(
                step, rate, phase, span, coefficient,
            )
            transport = BigFloat(coefficient) *
                exp(BigFloat(span) * Complex{BigFloat}(phase))
            target = transport *
                (BigFloat(step) * Complex{BigFloat}(rate) - Complex{BigFloat}(phase))
            scale = abs(transport) *
                (
                abs(BigFloat(step) * Complex{BigFloat}(rate)) +
                    abs(Complex{BigFloat}(phase))
            )
            # 64 binary64 rounded component operations plus host transport/reduction,
            # one complex narrowing; cancellation uses the uncancelled difference.
            budget = (BigFloat(2)^-22 + BigFloat(128) * 2^-53) * scale + BigFloat(2)^-149
            if abs(target) < BigFloat(floatmax(Float32))
                @test isfinite(actual)
                @test abs(Complex{BigFloat}(actual) - target) <= budget
            else
                @test !isfinite(actual)
            end
        end
        # h*D overflows binary64; only the final combined factor controls admission.
        actual = SolpropJulia._adams_compensation_factor(
            2.0^600, complex(2.0^600), 0.0im, 0.0, 2.0^-1000,
        )
        @test !isfinite(actual) # Complete coefficient 2^200 exceeds binary32.
        rescued = SolpropJulia._adams_compensation_factor(
            2.0^600, complex(2.0^600), 0.0im, 0.0, 2.0^-1070,
        )
        @test !isfinite(rescued) # 2^130 remains outside the admitted factor range.
        actual = SolpropJulia._adams_compensation_factor(
            2.0^550, complex(2.0^550), 0.0im, 0.0, 2.0^-1070,
        )
        @test actual == ComplexF32(2.0^30)
        h, rate = nextfloat(1.0), 1.0 - 2.0^-52
        @test h * rate == 1.0
        @test BigFloat(h) * BigFloat(rate) != 1
        @test !isfinite(
            SolpropJulia._adams_compensation_factor(
                h, complex(rate), 1.0 + 0im, 2.0^21, 1.0,
            )
        )
        @test iszero(
            SolpropJulia._adams_compensation_factor(
                h, complex(rate), 1.0 + 0im, 2.0^21, 0.0,
            )
        )
        @test SolpropJulia._adams_compensation_factor(1.0, 1.0im, 1.0im, 1.0e308, 1.0) == 0
        @test !isfinite(
            SolpropJulia._adams_compensation_factor(
                1.0, 0.0im, 2.0^21 + 0im, 1.0, 1.0e-300,
            )
        )
    end
end

function adams_precision_defect_reference(fixture, row, column, score)
    work, history, stencil = fixture.work, fixture.history, fixture.stencil
    count = stencil.orders[score] + 1
    phase = Complex{BigFloat}(work.exponent[row, column])
    step = BigFloat(0.125)
    difference, moment = zero(phase), zero(phase)
    difference_scale, moment_scale = BigFloat(0), BigFloat(0)
    for sample in 1:count
        sampled = Complex{BigFloat}(
            sample == 1 ? work.source[row, column] :
                history.source[row, column, stencil.slots[sample - 1]]
        )
        term = BigFloat(work.coefficients.defect_samples[sample, score]) * sampled
        difference += term
        difference_scale += abs(term)
    end
    for power in 1:count
        term = BigFloat(work.coefficients.defect_polynomial[power, score]) *
            adams_precision_moment(phase, power)
        moment += term
        moment_scale += abs(term)
    end
    result = step * difference * moment
    contributions = abs(step) * difference_scale * moment_scale
    if work.compensated
        for sample in 1:count
            newest = sample == 1
            slot = newest ? 1 : stencil.slots[sample - 1]
            node = newest ? BigFloat(1) : BigFloat(stencil.nodes[sample - 1])
            rate = Complex{BigFloat}(
                newest ? work.endpoint_rate[row, column] :
                    history.rates[row, column, slot]
            )
            sampled = Complex{BigFloat}(
                newest ? work.predictor[row, column] :
                    history.states[row, column, slot]
            )
            multiplier = BigFloat(work.coefficients.compensation_defects[sample, score]) *
                exp((1 - node) * phase)
            result += multiplier * (step * rate - phase) * sampled
            contributions += abs(multiplier * sampled) * (abs(step * rate) + abs(phase))
        end
    end
    # The protected difference adds normalized rounded weights and at most seven
    # C32 products. 64*(q+1) bounds that graph plus the final product/compensation.
    unit = BigFloat(2)^-24
    count_ops = 64count
    budget = (count_ops * unit / (1 - count_ops * unit) + BigFloat(2)^-39) *
        contributions + BigFloat(2)^-140
    @test budget > 0
    return result, budget
end

# Measure inside a compiled caller: the enclosing high-precision test closure
# otherwise boxes its reassigned fixture before the numerical function is entered.
function adams_precision_norm_allocation(output, fixture, mask, weights, execution)
    return @allocated SolpropJulia.adams_error_norms!(
        output, fixture.work, fixture.state, mask, weights, 0.125, 0.03125, execution,
    )
end

@testset "Binary32 Adams protected defects and old-state norms" begin
    for tasks in unique((1, Threads.nthreads(:default))), compensated in (false, true)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        for order in 1:6
            fixture = adams_precision_fixture(execution, order; compensated)
            adams_precision_trial!(fixture, execution)
            work = fixture.work
            setprecision(BigFloat, 256) do
                for score in 1:work.n_orders, column in 1:2, row in 1:3
                    expected, budget = adams_precision_defect_reference(
                        fixture, row, column, score,
                    )
                    @test abs(
                        Complex{BigFloat}(work.defects[row, column, score]) -
                            expected
                    ) <= budget
                end
            end
            mask, weights, output = [true, false, true], [0.25, 1.0e308, 4.0], zeros(4)
            # An excluded huge denominator cannot poison the selected rows; a huge
            # finite candidate cannot replace Adams' old-state normalization.
            work.candidate .= ComplexF32(1.0e30)
            SolpropJulia.adams_error_norms!(
                output, work, fixture.state, mask, weights, 0.125, 0.03125, execution,
            )
            setprecision(BigFloat, 256) do
                norm = sqrt(sum(abs2, Complex{BigFloat}.(fixture.state[mask, :])))
                divisors = BigFloat(0.125) * norm .* weights .+ BigFloat(0.03125)
                for score in 1:work.n_orders
                    reference = sqrt(
                        sum(
                            abs2(Complex{BigFloat}(work.defects[row, column, score])) /
                                divisors[row]^2
                                for row in (1, 3), column in 1:2
                        )
                    )
                    # Max-normalized components, two squares, Sum2s, denominator
                    # mantissa and final host merge: 128u bounds these 8 components.
                    budget = BigFloat(128) * 2^-24 * reference + BigFloat(2)^-140
                    @test budget > 0 && abs(BigFloat(output[score]) - reference) <= budget
                end
            end
            @test iszero(output[(work.n_orders + 1):end])
            for _ in 1:2
                SolpropJulia.adams_error_norms!(
                    output, work, fixture.state, mask, weights, 0.125, 0.03125,
                    execution,
                )
            end
            @test adams_precision_norm_allocation(
                output, fixture, mask, weights, execution,
            ) == 0
            work.candidate[2, 1] = ComplexF32(Inf)
            SolpropJulia.adams_error_norms!(
                output, work, fixture.state, mask, weights, 0.125, 0.03125, execution,
            )
            @test all(isinf, output[1:work.n_orders])
        end
        fixture = adams_precision_fixture(execution, 2; compensated)
        adams_precision_trial!(fixture, execution)
        mask, weights, output = fill(true, 3), ones(3), zeros(3)
        for magnitude in (
                nextfloat(0.0f0), floatmin(Float32), 1.0f-20,
                1.0f20, floatmax(Float32) / 8,
            )
            fixture.state .= ComplexF32(magnitude, -magnitude)
            fixture.work.candidate .= fixture.state
            fixture.work.defects .= fixture.state
            SolpropJulia.adams_error_norms!(
                output, fixture.work, fixture.state, mask, weights, 0.125, 0.0,
                execution,
            )
            # The ratio is invariant to the physical amplitude. At the smallest
            # subnormal, the scaled denominator still prevents underflow to zero.
            @test all(value -> abs(value - 8) <= 256eps(Float32), output)
        end
        fixture.state .= 0
        fixture.work.candidate .= 0
        fixture.work.defects .= 0
        SolpropJulia.adams_error_norms!(
            output, fixture.work, fixture.state, mask, weights, 0.0, 0.0, execution,
        )
        @test iszero(output)
        fixture.state[2, 2] = ComplexF32(NaN)
        mask[2] = false
        SolpropJulia.adams_error_norms!(
            output, fixture.work, fixture.state, mask, weights, 1.0, 0.0, execution,
        )
        @test all(isinf, output)
    end
end

function adams_precision_storage!(fixture, execution)
    SolpropJulia.adams_seed!(
        fixture.history, 7, fixture.state, -1.0, fixture.source, fixture.operator,
        execution,
    )
    SolpropJulia.adams_accept!(fixture.history, 7, fixture.work, execution; identity = 42)
    return nothing
end

@testset "Binary32 Adams failure, structural flow and precision admission" begin
    execution = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    fixture = adams_precision_fixture(execution, 2; compensated = true)
    adams_precision_trial!(fixture, execution)
    history, work, state = fixture.history, fixture.work, fixture.state
    for _ in 1:2
        adams_precision_storage!(fixture, execution)
    end
    @test @allocated(adams_precision_storage!(fixture, execution)) == 0
    @test history.rates[:, :, 7] == work.endpoint_rate
    @test history.states[:, :, 7] == work.predictor
    before = copy(history.source), copy(history.rates), copy(history.occupied)
    @test_throws ArgumentError SolpropJulia.adams_seed!(
        history, 7, ComplexF64.(state), 0.0, fixture.source, fixture.operator,
        execution,
    )
    @test before == (history.source, history.rates, history.occupied)
    other = SolpropJulia.AdamsHistory(execution, ComplexF64.(state), 6; compensated = true)
    @test_throws ArgumentError SolpropJulia.adams_copy_history!(other, history)
    @test_throws ArgumentError SolpropJulia.adams_trial!(
        work, ComplexF64.(state), 0.0, 0.125, history, fixture.stencil, fixture.source,
        fixture.operator, fixture.operator, execution,
    )
    @test !work.ready
    calls = Ref(0)
    badsource = (out, input, z) -> (calls[] += 1; fill!(out, ComplexF32(Inf)); nothing)
    old_state, old_store = copy(state), copy(history.source)
    @test_throws ArgumentError SolpropJulia.adams_trial!(
        work, state, 0.0, 0.125, history, fixture.stencil, badsource, fixture.operator,
        fixture.operator, execution,
    )
    @test calls[] == 1 && !work.ready
    @test state == old_state && history.source == old_store
    growth = AdamsPrecisionOperator(1000.0 + 0im, 0.0im)
    calls[] = 0
    @test_throws ArgumentError SolpropJulia.adams_trial!(
        work, state, 0.0, 0.125, history, fixture.stencil, badsource, growth, growth,
        execution,
    )
    @test calls[] == 0 && !work.ready
    @test state == old_state && history.source == old_store
    badrate = (out, z, before) -> (fill!(out, ComplexF64(Inf)); nothing)
    @test_throws ArgumentError SolpropJulia.adams_seed!(
        history, 7, state, 0.0, badsource, fixture.operator, execution,
    )
    @test !history.occupied[7]
    @test_throws ArgumentError SolpropJulia.adams_seed!(
        history, 7, state, 0.0, fixture.source, badrate, execution,
    )
    @test !history.occupied[7]
    @test_throws ArgumentError SolpropJulia.adams_seed!(
        history, 7, state, big"1e500", fixture.source, fixture.operator, execution,
    )
    @test !history.occupied[7]
    @test SolpropJulia.adams_trial!(
        work, state, 0.0f0, 0.125f0, history, fixture.stencil,
        fixture.source, fixture.operator, fixture.operator, execution,
    ) == 1
    plain = adams_precision_fixture(execution, 1)
    no_source = (out, input, z) -> error("structurally absent source was called")
    no_rate = (out, z, before) -> error("structurally absent rate was called")
    zero_operator = AdamsPrecisionOperator(0.0im, 0.0im)
    plain.state[1] = ComplexF32(-0.0f0, -0.0f0)
    @test SolpropJulia.adams_trial!(
        plain.work, plain.state, 0.0, 0.125, plain.history, plain.stencil, no_source,
        zero_operator, no_rate, execution; source_absent = true,
    ) == 0
    @test reinterpret(UInt32, vec(plain.work.candidate)) ==
        reinterpret(UInt32, vec(plain.state))
    @test all(isone, plain.work.flow)
    @test iszero(plain.work.defects)
    dense = SolpropJulia.AdamsDense(execution, state, 6; compensated = true)
    adams_precision_trial!(fixture, execution)
    SolpropJulia.adams_retain_dense!(
        dense, work, state, work.candidate, history, fixture.stencil, execution,
    )
    dense.exponent .= -2.0
    dense.source .= ComplexF32(NaN)
    dense.rates .= ComplexF64(NaN)
    output = similar(state)
    SolpropJulia.adams_observe!(output, dense, 0.0625f0, execution)
    reference = exp(-1.0) .* ComplexF64.(state)
    # One binary64 exp, one C32 narrowing and complex product.
    budget = 16eps(Float32) * maximum(abs, reference)
    @test budget > 0 && maximum(abs, output - reference) <= budget
    @test_throws ArgumentError SolpropJulia.adams_observe!(
        ComplexF64.(output), dense, 0.0625, execution,
    )
    other_dense = SolpropJulia.AdamsDense(
        execution, ComplexF64.(state), 6; compensated = true,
    )
    @test_throws ArgumentError SolpropJulia.adams_copy_dense!(other_dense, dense)
end

@testset "Eight-step binary32 PEC against independent exact-history recurrence" begin
    for tasks in unique((1, Threads.nthreads(:default)))
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        state = fill(ComplexF32(1, 0.5), 2, 2)
        history = SolpropJulia.AdamsHistory(execution, state, 1)
        work = SolpropJulia.AdamsWork(execution, state, 1)
        stencil = SolpropJulia.AdamsStencil(1)
        source = AdamsPrecisionSource(ComplexF32(0, 0.25), ComplexF32(0))
        operator = AdamsPrecisionOperator(-0.125 + 0.25im, 0.0im)
        SolpropJulia.adams_seed!(history, 1, state, 0.0, source, operator, execution)
        SolpropJulia.adams_stencil!(stencil, 1, (1,), (1,))
        setprecision(BigFloat, 256) do
            step = BigFloat(0.125)
            phase = step * Complex{BigFloat}(operator.value)
            linear = exp(phase)
            first = step * adams_precision_moment(phase, 1)
            second = step * adams_precision_moment(phase, 2)
            lambda = Complex{BigFloat}(source.multiplier)
            # Two-register PEC: U is corrected, V is the retained predictor sample.
            first_coefficient = linear + second * lambda * linear
            retained_coefficient =
                (first - second) * lambda + second * first * lambda^2
            matrix = [first_coefficient retained_coefficient; linear first * lambda]
            # Local roundoff sees the separate terms before either coefficient
            # sum cancels; propagation of an earlier error uses the exact matrix.
            contribution_matrix = [
                abs(linear) + abs(second * lambda * linear) (
                    (abs(first) + abs(second)) * abs(lambda) +
                        abs(second * first * lambda^2)
                );
                abs(linear) abs(first * lambda)
            ]
            exact = fill(Complex{BigFloat}(state[1]), 2)
            errors = zeros(BigFloat, 2)
            for index in 1:8
                z = (index - 1) * 0.125
                # 128 rounded scalar operations safely counts the complex source,
                # complete coefficients, predictor/corrector and both stored outputs.
                # The absolute matrix bounds propagation of each earlier local error.
                contributions = contribution_matrix * (abs.(exact) + errors)
                unit = BigFloat(2)^-24
                local_error = (128unit / (1 - 128unit) + BigFloat(2)^-39) .* contributions
                errors = abs.(matrix) * errors + local_error
                exact = matrix * exact
                SolpropJulia.adams_trial!(
                    work, state, z, 0.125, history, stencil, source, operator,
                    operator, execution,
                )
                @test errors[1] > 0 &&
                    abs(Complex{BigFloat}(work.candidate[1]) - exact[1]) <= errors[1]
                @test errors[2] > 0 &&
                    abs(Complex{BigFloat}(work.predictor[1]) - exact[2]) <= errors[2]
                SolpropJulia.adams_accept!(history, 1, work, execution)
                copyto!(state, work.candidate)
            end
        end
    end
end

@testset "Tiny dense fractions retain complete source and rate factors" begin
    for tasks in unique((1, Threads.nthreads(:default))), compensated in (false, true)
        execution = SolpropJulia.KernelExecution(
            SolpropJulia.KernelAbstractions.CPU(), tasks, true,
        )
        state = zeros(ComplexF32, 1, 1)
        dense = SolpropJulia.AdamsDense(execution, state, 6; compensated)
        dense.ready, dense.order, dense.step = true, 6, 1.0e300
        dense.coefficients[1, 7] = compensated ? 1.0 : 1.0e20
        if compensated
            dense.states[1, 1, 1] = 1
            dense.rates[1, 1, 1] = 1.0e50
        else
            dense.source[1, 1, 1] = 1
        end
        output = similar(state)
        position = 1.0e250
        fraction = position / dense.step
        @test iszero(fraction^7) # An intermediate F64 power really would underflow.
        SolpropJulia.adams_observe!(output, dense, position, execution)
        setprecision(BigFloat, 512) do
            target = BigFloat(dense.step) * BigFloat(fraction)^7 / 7 *
                (compensated ? BigFloat(1.0e50) : BigFloat(1.0e20))
            # Seven wide Horner powers, divisions, complete products and one C32
            # factor narrowing. The reference uses the actual represented xi.
            budget = (BigFloat(2)^-21 + BigFloat(2)^-39) * abs(target) + BigFloat(2)^-149
            @test target > 0 && budget > 0
            @test !iszero(output[1])
            @test abs(Complex{BigFloat}(output[1]) - target) <= budget
        end
        for _ in 1:2
            SolpropJulia.adams_observe!(output, dense, position, execution)
        end
        @test @allocated(
            SolpropJulia.adams_observe!(
                output, dense, position, execution,
            )
        ) == 0
    end
end

@testset "Dense wide weights need not be representable as binary64 scalars" begin
    execution = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    state = zeros(ComplexF32, 1, 1)
    for compensated in (false, true)
        dense = SolpropJulia.AdamsDense(execution, state, 6; compensated)
        dense.ready, dense.order, dense.step = true, 6, 1.0e-308
        dense.coefficients[1, :] .= floatmax(Float64)
        if compensated
            dense.states[1, 1, 1] = 1
            dense.rates[1, 1, 1] = 1
        else
            dense.source[1, 1, 1] = 1
        end
        output = similar(state)
        position = 0.9 * dense.step
        fraction = position / dense.step
        SolpropJulia.adams_observe!(output, dense, position, execution)
        @test iszero(dense.weights) # Unused scalar scratch is explicitly cleared.
        setprecision(BigFloat, 256) do
            weight = sum(
                BigFloat(floatmax(Float64)) * BigFloat(fraction)^power / power
                    for power in 1:7
            )
            @test weight > BigFloat(floatmax(Float64))
            target = BigFloat(dense.step) * weight
            # Seven integral terms, final factor rounding and state multiplication.
            budget = (BigFloat(2)^-21 + BigFloat(2)^-39) * target
            @test budget > 0 && abs(Complex{BigFloat}(output[1]) - target) <= budget
        end
    end
end

@testset "Transport domain precedes negative-tail and rounded-zero shortcuts" begin
    for imaginary in (Inf, -Inf, NaN)
        factor = SolpropJulia._adams_transport_exp(complex(-Inf, imaginary))
        @test !isfinite(factor.mantissa)
    end
    @test iszero(SolpropJulia._adams_transport_exp(complex(-Inf, 0.0)).mantissa)
    # Finite stored phase and span can overflow just the transported imaginary
    # component while the real component is well inside the negative-tail branch.
    phase, span = complex(-1.0, 1.0e308), 1.0e308
    @test isfinite(phase) && isfinite(span) && !isfinite(imag(span * phase))
    @test !isfinite(
        SolpropJulia._adams_compensation_factor(
            1.0, 0.0im, phase, span, 1.0,
        )
    )
    @test iszero(
        SolpropJulia._adams_compensation_factor(
            1.0, 0.0im, phase, span, 0.0,
        )
    )
end

@testset "Binary32 Adams retains actual endpoints separately from numerical step" begin
    execution = SolpropJulia.KernelExecution(
        SolpropJulia.KernelAbstractions.CPU(), Threads.nthreads(:default), true,
    )
    fixture = adams_precision_fixture(execution, 1; compensated = true)
    # Use one score so only the freshly sampled slot participates in this test.
    SolpropJulia.adams_stencil!(fixture.stencil, 1, (1,), (1,))
    operator = AdamsPrecisionOperator(0.0 + 0.25im, 0.0im)
    position, step = 1.2, 0.1
    SolpropJulia.adams_seed!(
        fixture.history, 1, fixture.state, position, fixture.source, operator, execution,
    )
    @test position + step - position != step
    SolpropJulia.adams_trial!(
        fixture.work, fixture.state, position, step, fixture.history, fixture.stencil,
        fixture.source, operator, operator, execution,
    )
    @test all(==((position + step - position) * operator.value), fixture.work.exponent)
    @test fixture.work.step == step
    @test all(!=(step * operator.value), fixture.work.exponent)
    @test !iszero(
        SolpropJulia._adams_compensation_factor(
            step, operator.value, fixture.work.exponent[1], 0.0, 1.0,
        )
    )
end
