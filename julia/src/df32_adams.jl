# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public DF32AdamsFlow, DF32AdamsSeed, DF32AdamsTrial, DF32AdamsCertificate
public prepare_df32_adams_trial!

# All certificate arithmetic is host-only. Numerical specializations below keep
# DF32 mantissas and integer powers separate until the complete state factor.
struct _DF32AdamsScaled{T}
    value::Complex{T}
    exponent::Int
end

@inline _da_value(x::DF32) = x
@inline _da_value(x::_WideBall) = x.value
@inline _da_value(x::Complex{T}) where {T} =
    Complex{DF32}(_da_value(real(x)), _da_value(imag(x)))
@inline _da_zero(::Type{T}) where {T} = Complex{T}(zero(T), zero(T))
@inline _da_cmul(x::Complex{T}, y::Complex{T}) where {T} = Complex{T}(
    real(x) * real(y) - imag(x) * imag(y),
    real(x) * imag(y) + imag(x) * real(y),
)

@inline function _da_ldexp(x::DF32, exponent::Int)
    high, low = _df32_fast_two_sum(ldexp(x.hi, exponent), ldexp(x.lo, exponent))
    return _df32_raw(high, low)
end

@inline _da_scale_radius(error::Float64, exponent::Int) =
    iszero(error) ? 0.0 : nextfloat(ldexp(error, exponent))

@inline function _da_ldexp(x::_WideBall, exponent::Int)
    value = _da_ldexp(x.value, exponent)
    radius = _wide_sum(_da_scale_radius(x.error, exponent), 3_WIDE_ETA)
    magnitude = _wide_magnitude(value)
    return _WideBall(
        value, radius, magnitude, x.valid && isfinite(radius) &&
            isfinite(value) && magnitude <= _WIDE_LIMIT
    )
end

@inline _da_ldexp(x::Complex{T}, exponent::Int) where {T} =
    Complex{T}(_da_ldexp(real(x), exponent), _da_ldexp(imag(x), exponent))

@inline function _da_scaled(value::Complex{T}, exponent::Int = 0) where {T}
    words = _da_value(value)
    largest = max(abs(real(words).hi), abs(imag(words).hi))
    shift = iszero(largest) || !isfinite(largest) ? 0 : last(frexp(largest))
    return _DF32AdamsScaled(_da_ldexp(value, -shift), exponent + shift)
end

@inline function _da_mul(x::_DF32AdamsScaled{T}, y::_DF32AdamsScaled{T}) where {T}
    return _da_scaled(_da_cmul(x.value, y.value), x.exponent + y.exponent)
end

@inline function _da_add(x::_DF32AdamsScaled{T}, y::_DF32AdamsScaled{T}) where {T}
    exponent = max(x.exponent, y.exponent)
    left, right = _da_ldexp(x.value, x.exponent - exponent),
        _da_ldexp(y.value, y.exponent - exponent)
    return _da_scaled(_wide_cadd(left, right), exponent)
end

@inline _da_neg(x::_DF32AdamsScaled) =
    _DF32AdamsScaled(_wide_cneg(x.value), x.exponent)

@inline function _da_host(x::_WideHost)
    shift = iszero(x.value) ? 0 : last(frexp(x.value))
    value = _wide_split(
        _WideHost(
            ldexp(x.value, -shift),
            _da_scale_radius(x.error, -shift)
        )
    )
    return _DF32AdamsScaled(Complex{_WideBall}(value, zero(_WideBall)), shift)
end

@inline _da_compute(x::_DF32AdamsScaled{_WideBall}) =
    _DF32AdamsScaled(_da_value(x.value), x.exponent)

function _da_complex_host(value::ComplexF64, radius::ComplexF64)
    largest = max(abs(real(value)), abs(imag(value)))
    shift = iszero(largest) ? 0 : last(frexp(largest))
    r = _wide_split(
        _WideHost(
            ldexp(real(value), -shift),
            _da_scale_radius(real(radius), -shift)
        )
    )
    i = _wide_split(
        _WideHost(
            ldexp(imag(value), -shift),
            _da_scale_radius(imag(radius), -shift)
        )
    )
    return _DF32AdamsScaled(Complex{_WideBall}(r, i), shift)
end
@inline _da_narrow(x::_DF32AdamsScaled{DF32}) = ComplexF32(
    ldexp(real(x.value).hi, x.exponent), ldexp(imag(x.value).hi, x.exponent),
)

function _da_narrow_error(x::_DF32AdamsScaled{_WideBall})
    real(x.value).valid && imag(x.value).valid ||
        throw(ArgumentError("DF32 Adams factor intermediate range refused"))
    result = _da_narrow(_da_compute(x))
    isfinite(result) || throw(ArgumentError("DF32 Adams complete factor cannot narrow"))
    error = 0.0
    for component in (real(x.value), imag(x.value))
        magnitude = _da_scale_radius(_wide_magnitude(component.value), x.exponent)
        local_error = _wide_sum(_wide_product(_WIDE_U, magnitude), 2.0^-148)
        error = _wide_sum(
            error, _wide_sum(
                _da_scale_radius(component.error, x.exponent), local_error
            )
        )
    end
    isfinite(error) || throw(ArgumentError("DF32 Adams factor error range refused"))
    return error
end

@inline function _da_positive_exp_bound(x::Float64)
    # log(2)>2/3 gives exp(x)<=2^ceil(3x/2) for positive x. No libm
    # accuracy hypothesis is needed in the host certificate's exponential bound.
    x <= 0.0 && return 1.0
    return ldexp(1.0, ceil(Int, nextfloat(1.5x)))
end

@inline function _da_product_error(u, b, v, c)
    mu = 15_WIDE_U^2
    inherited = _wide_sum(
        _wide_sum(_wide_product(u, c), _wide_product(v, b)), _wide_product(b, c),
    )
    return _wide_sum(
        _wide_product(_wide_sum(1.0, mu), inherited),
        _wide_sum(_wide_product(mu, _wide_product(u, v)), 15_WIDE_ETA),
    )
end

@inline function _da_sum_error(u, b, v, c)
    return _wide_sum(
        _wide_product(_wide_sum(1.0, _WIDE_ADD), _wide_sum(b, c)),
        _wide_product(_WIDE_ADD, _wide_sum(u, v)),
    )
end

@inline function _da_real_upper(x::DF32)
    value = _wide_host(Float64(x.hi)) + _wide_host(Float64(x.lo))
    return _wide_sum(value.value, value.error)
end

function _da_argument(x::Complex{_WideBall})
    real(x).valid && imag(x).valid ||
        throw(ArgumentError("DF32 Adams argument intermediate range refused"))
    error = _wide_sum(real(x).error, imag(x).error)
    check_df32_function_argument(_da_value(x); input_error = error)
    return error
end

function _da_exp(x::Complex{_WideBall})
    error = _da_argument(x)
    words = _da_value(x)
    magnitude = _da_positive_exp_bound(_da_real_upper(real(words)))
    bound = _wide_sum(
        _wide_product(_wide_sum(8192_WIDE_U^2, 2.0^40 * _WIDE_ETA), magnitude),
        _wide_product(
            error, _da_positive_exp_bound(
                _wide_sum(_da_real_upper(real(words)), error),
            )
        ),
    )
    value = _df32_exp(words)
    return _wide_load(value, ComplexF64(bound, bound), Val(:checked))
end

@inline _da_exp(x::Complex{DF32}) = _df32_exp(x)

function _da_phi(x::Complex{_WideBall})
    argument_error = _da_argument(x)
    words = _da_value(x)
    result = _df32_phi_values(words, Val(7))
    real_upper = _da_real_upper(real(words))
    magnitude = _da_positive_exp_bound(real_upper)
    errors = ntuple(unused -> 0.0, Val(8))
    bounds = ntuple(k -> _wide_quotient(magnitude, Float64(factorial(k - 1))), Val(8))
    if iszero(words)
        errors = ntuple(
            k -> _wide_quotient(
                2_WIDE_U^2,
                Float64(factorial(k - 1))
            ), Val(8)
        )
    elseif _df32_complex_abs2(words).hi > 9.0f0
        # The large-branch modulus is at least 2.99. A componentwise lower
        # bound sharpens its reciprocal without a host square root assumption.
        lower = 2.98
        for component in (real(words), imag(words))
            sum = _wide_host(Float64(component.hi)) + _wide_host(Float64(component.lo))
            lower = max(lower, prevfloat(abs(sum.value) - sum.error))
        end
        inverse = _wide_quotient(1.0, lower)
        inverse_error = _wide_sum(32_WIDE_U^2 * inverse, 32(1 + 2.0^21) * _WIDE_ETA)
        errors = Base.setindex(
            errors,
            _wide_product(_wide_sum(8192_WIDE_U^2, 2.0^40 * _WIDE_ETA), magnitude), 1
        )
        for k in 1:7
            constant = _wide_quotient(1.0, Float64(factorial(k - 1)))
            difference = _wide_sum(bounds[k], constant)
            error = _da_sum_error(bounds[k], errors[k], constant, 2_WIDE_U^2 * constant)
            errors = Base.setindex(
                errors,
                _da_product_error(difference, error, inverse, inverse_error), k + 1
            )
            bounds = Base.setindex(
                bounds,
                min(bounds[k + 1], _wide_product(difference, inverse)), k + 1
            )
        end
    else
        squared = _df32_complex_abs2(words)
        threshold, halvings = (1.0f0 - 4.0f0 * 2.0f0^-24) * 0.25f0, 0
        while squared.hi > threshold
            threshold *= 4.0f0
            halvings += 1
        end
        # Equations (14)-(16), with r<=1/2 and exp(r)<2. The series
        # tail is <2^-106 for all positive requested orders.
        bounds = ntuple(k -> _wide_quotient(2.0, Float64(factorial(k - 1))), Val(8))
        errors = Base.setindex(
            errors,
            _wide_sum(
                128_WIDE_U^2 * bounds[8],
                _wide_sum(1024_WIDE_ETA, 2.0^-106)
            ), 8
        )
        for k in 6:-1:1
            constant = _wide_quotient(1.0, Float64(factorial(k)))
            error = _da_product_error(0.5, 0.0, bounds[k + 2], errors[k + 2])
            errors = Base.setindex(
                errors,
                _da_sum_error(bounds[k + 1], error, constant, 2_WIDE_U^2 * constant), k + 1
            )
        end
        errors = Base.setindex(errors, _wide_sum(64_WIDE_U^2, 64_WIDE_ETA) * 2, 1)
        for turn in 1:halvings
            old_errors, old_bounds = errors, bounds
            errors = Base.setindex(
                errors,
                _da_product_error(
                    old_bounds[1], old_errors[1], old_bounds[1], old_errors[1]
                ), 1
            )
            for k in 1:7
                total = _wide_product(old_bounds[1], old_bounds[k + 1])
                error = _da_product_error(
                    old_bounds[1], old_errors[1], old_bounds[k + 1], old_errors[k + 1],
                )
                for j in 1:k
                    constant = _wide_quotient(1.0, Float64(factorial(k - j)))
                    term = _wide_product(old_bounds[j + 1], constant)
                    term_error = _da_product_error(
                        old_bounds[j + 1], old_errors[j + 1], constant,
                        2_WIDE_U^2 * constant,
                    )
                    error = _da_sum_error(total, error, term, term_error)
                    total = _wide_sum(total, term)
                end
                errors = Base.setindex(
                    errors,
                    _wide_sum(_da_scale_radius(error, -k), 3_WIDE_ETA), k + 1
                )
            end
            bounds = let upper = _da_positive_exp_bound(ldexp(real_upper, turn - halvings))
                ntuple(k -> _wide_quotient(upper, Float64(factorial(k - 1))), Val(8))
            end
        end
        errors = let previous = errors
            ntuple(k -> _wide_sum(previous[k], 1024_WIDE_ETA), Val(8))
        end
    end
    sensitivity = _da_positive_exp_bound(_wide_sum(real_upper, argument_error))
    final_errors = errors
    return ntuple(Val(8)) do k
        divisor = Float64(factorial(k == 1 ? 0 : k))
        bound = _wide_sum(
            final_errors[k], _wide_quotient(
                _wide_product(argument_error, sensitivity), divisor,
            )
        )
        _wide_load(result[k], ComplexF64(bound, bound), Val(:checked))
    end
end

@inline _da_phi(x::Complex{DF32}) = _df32_phi_values(x, Val(7))

struct _DF32AdamsCoefficients{T}
    predictor::Matrix{T}
    corrector::Matrix{T}
    defect_samples::Matrix{T}
    defect_polynomial::Matrix{T}
    compensation_predictor::Vector{T}
    compensation_corrector::Vector{T}
    compensation_defects::Matrix{T}
    nodes::Vector{T}
    basis::Matrix{T}
    polynomial::Vector{T}
    product::Vector{T}

    function _DF32AdamsCoefficients(
            predictor::Matrix{T}, corrector::Matrix{T}, defect_samples::Matrix{T},
            defect_polynomial::Matrix{T}, compensation_predictor::Vector{T},
            compensation_corrector::Vector{T}, compensation_defects::Matrix{T},
            nodes::Vector{T}, basis::Matrix{T}, polynomial::Vector{T}, product::Vector{T},
        ) where {T}
        return new{T}(
            predictor, corrector, defect_samples, defect_polynomial,
            compensation_predictor, compensation_corrector, compensation_defects,
            nodes, basis, polynomial, product
        )
    end
end

function _DF32AdamsCoefficients(maximum::Int, initial::T) where {T}
    width = maximum + 1
    return _DF32AdamsCoefficients(
        fill(initial, width, width), fill(initial, width, width),
        fill(initial, width, 3), fill(initial, width, 3),
        fill(initial, width), fill(initial, width), fill(initial, width, 3),
        fill(initial, width), fill(initial, width, width),
        fill(initial, width), fill(initial, width),
    )
end

@inline _da_coefficient_arrays(c) = (
    c.predictor, c.corrector, c.defect_samples, c.defect_polynomial,
    c.compensation_predictor, c.compensation_corrector, c.compensation_defects,
    c.nodes, c.basis, c.polynomial, c.product,
)

function _da_basis!(output, nodes, first, count, polynomial, product)
    zero_host, one_host = _wide_host(0.0), _wide_host(1.0)
    fill!(output, zero_host)
    for sample in 1:count
        fill!(polynomial, zero_host)
        polynomial[1] = one_host
        degree_count, divisor = 1, one_host
        point = nodes[first + sample - 1]
        for other in 1:count
            other == sample && continue
            root = nodes[first + other - 1]
            fill!(product, zero_host)
            for power in 1:degree_count
                product[power + 1] = polynomial[power]
            end
            for power in 1:degree_count
                product[power] = product[power] - root * polynomial[power]
            end
            degree_count += 1
            copyto!(polynomial, product)
            divisor = divisor * (point - root)
        end
        for power in 1:count
            output[sample, power] = polynomial[power] / divisor
        end
    end
    return nothing
end

function _da_integral(basis, sample, count)
    integral = _wide_host(0.0)
    for power in count:-1:1
        integral = integral + basis[sample, power] / _wide_host(Float64(power))
    end
    return integral
end

function _da_coefficients!(coefficients, stencil, positions, origin, step)
    zero_host, one_host = _wide_host(0.0), _wide_host(1.0)
    _da_zero_coefficients!(_da_coefficient_arrays(coefficients), zero_host)
    nodes = coefficients.nodes
    nodes[1] = one_host
    for index in 1:stencil.count
        nodes[index + 1] = (_wide_host(positions[index]) - _wide_host(origin)) /
            _wide_host(step)
    end
    polynomial, product, basis =
        coefficients.polynomial, coefficients.product, coefficients.basis
    order = stencil.order
    _da_basis!(coefficients.predictor, nodes, 2, order, polynomial, product)
    _da_basis!(coefficients.corrector, nodes, 1, order + 1, polynomial, product)
    for sample in 1:order
        coefficients.compensation_predictor[sample] =
            _da_integral(coefficients.predictor, sample, order)
    end
    for sample in 1:(order + 1)
        coefficients.compensation_corrector[sample] =
            _da_integral(coefficients.corrector, sample, order + 1)
    end
    for score in 1:stencil.n_orders
        scored = stencil.orders[score]
        for sample in 1:(scored + 1)
            divisor = one_host
            for other in 1:(scored + 1)
                sample == other && continue
                divisor = divisor * (nodes[sample] - nodes[other])
            end
            coefficients.defect_samples[sample, score] = one_host / divisor
        end
        fill!(polynomial, zero_host)
        polynomial[1] = one_host
        for index in 1:scored
            fill!(product, zero_host)
            for power in 1:index
                product[power + 1] = polynomial[power]
            end
            for power in 1:index
                product[power] = product[power] - nodes[index + 1] * polynomial[power]
            end
            copyto!(polynomial, product)
        end
        for power in 1:(scored + 1)
            coefficients.defect_polynomial[power, score] = polynomial[power]
        end
        _da_basis!(basis, nodes, 1, scored + 1, polynomial, product)
        for sample in 1:(scored + 1)
            coefficients.compensation_defects[sample, score] =
                _da_integral(basis, sample, scored + 1)
        end
        _da_basis!(basis, nodes, 2, scored, polynomial, product)
        for sample in 1:scored
            coefficients.compensation_defects[sample + 1, score] =
                coefficients.compensation_defects[sample + 1, score] -
                _da_integral(basis, sample, scored)
        end
    end
    _da_check_coefficients(_da_coefficient_arrays(coefficients))
    return nothing
end

@inline _da_zero_coefficients!(::Tuple{}, initial) = nothing
function _da_zero_coefficients!(arrays::Tuple, initial)
    fill!(first(arrays), initial)
    return _da_zero_coefficients!(Base.tail(arrays), initial)
end

@inline _da_check_coefficients(::Tuple{}) = nothing
function _da_check_coefficients(arrays::Tuple)
    for value in first(arrays)
        isfinite(value.value) && isfinite(value.error) ||
            throw(ArgumentError("DF32 Adams coefficient conditioning refused"))
    end
    return _da_check_coefficients(Base.tail(arrays))
end

@inline _da_split_coefficients!(::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing
function _da_split_coefficients!(host::Tuple, checked::Tuple, numerical::Tuple)
    for index in eachindex(first(host))
        split = _da_host(first(host)[index])
        first(checked)[index] = split
        first(numerical)[index] = _da_compute(split)
    end
    return _da_split_coefficients!(
        Base.tail(host), Base.tail(checked), Base.tail(numerical)
    )
end

"""
    DF32AdamsSeed(slot, position)

Describe one virtual accepted source sample before a certified startup trial.

`slot` is a positive one-based owned history slot and `position` is its finite actual
coordinate in metres. The sample uses the right derivative at a knot. No source,
state, history or counter is changed by constructing this metadata. A trial plan
validates it against its actual origin and fresh history before it may be consumed.
"""
struct DF32AdamsSeed
    slot::Int
    position::Float64

    function DF32AdamsSeed(slot::Int, position::Float64)
        slot > 0 && isfinite(position) || throw(ArgumentError("invalid DF32 Adams seed"))
        return new(slot, position)
    end
end

"""
    DF32AdamsCertificate

Retain absolute complete-factor errors for one fully checked Adams trial.

`source_error` bounds the maximum sum of real/imaginary errors of any complete
source weight, in metres; `compensation_error` is the dimensionless equivalent.
`weight_error` bounds a normalized state-precision divided-difference weight.
These are absolute bounds, including function, coefficient, stored-rate and final
narrowing errors; a small cancelled factor supplies no relative-error claim.
`argument_error` is the largest admitted function input error. Construction belongs
to `prepare_df32_adams_trial!`; it does not certify a whole run or device execution.
"""
struct DF32AdamsCertificate
    source_error::Float64
    compensation_error::Float64
    weight_error::Float64
    argument_error::Float64
end

mutable struct _DF32AdamsRate
    saved::Matrix{ComplexF64}
    sample::WideRate
    origin_wide::Symbol
    numerical::Matrix{_DF32AdamsScaled{DF32}}

    _DF32AdamsRate(
        saved::Matrix{ComplexF64}, sample::WideRate, origin_wide::Symbol,
        numerical::Matrix{_DF32AdamsScaled{DF32}}
    ) =
        new(saved, sample, origin_wide, numerical)
end

function _da_checked_rate(record, index)
    raw, words, error = record.saved[index], record.sample.values[index],
        record.sample.errors[index]
    radius = ComplexF64(
        _wide_saved_radius(real(raw), real(words), real(error)),
        _wide_saved_radius(imag(raw), imag(words), imag(error)),
    )
    return _da_complex_host(raw, radius)
end

function _da_rate_record(record::WideBoundRate)
    numerical = fill(_da_compute(_da_host(_wide_host(0.0))), size(record.saved))
    result = _DF32AdamsRate(record.saved, record.sample, record.origin_wide, numerical)
    for index in eachindex(numerical)
        numerical[index] = _da_compute(_da_checked_rate(result, index))
    end
    return result
end

mutable struct _DF32AdamsBank
    host::_DF32AdamsCoefficients{_WideHost}
    checked::_DF32AdamsCoefficients{_DF32AdamsScaled{_WideBall}}
    numerical::_DF32AdamsCoefficients{_DF32AdamsScaled{DF32}}
    stencil::AdamsStencil
    positions::Vector{Float64}
    kinds::Vector{Int}
    generations::Vector{UInt64}
    operations::Vector{_WideScalars{_WideBall}}
    bundles::Vector{_WideScalars{DF32}}
    weights::Matrix{Float32}
    normalization::Vector{Int}
    seed::Union{Nothing, DF32AdamsSeed}
    seed_rate::Union{Nothing, _DF32AdamsRate}
    endpoint_rate::Union{Nothing, _DF32AdamsRate}
    certificate::DF32AdamsCertificate
    step_checked::_DF32AdamsScaled{_WideBall}
    step_numerical::_DF32AdamsScaled{DF32}

    function _DF32AdamsBank(
            maximum::Int,
            seed_rate::Union{Nothing, _DF32AdamsRate},
            endpoint_rate::Union{Nothing, _DF32AdamsRate}
        )
        zero_host = _wide_host(0.0)
        zero_checked = _da_host(zero_host)
        return new(
            _DF32AdamsCoefficients(maximum, zero_host),
            _DF32AdamsCoefficients(maximum, zero_checked),
            _DF32AdamsCoefficients(maximum, _da_compute(zero_checked)),
            AdamsStencil(maximum), zeros(maximum), zeros(Int, maximum),
            zeros(UInt64, maximum), Vector{_WideScalars{_WideBall}}(undef, maximum + 1),
            Vector{_WideScalars{DF32}}(undef, maximum + 1),
            zeros(Float32, maximum + 1, 3), zeros(Int, 3), nothing, seed_rate,
            endpoint_rate, DF32AdamsCertificate(0, 0, 0, 0),
            zero_checked, _da_compute(zero_checked),
        )
    end
end

mutable struct _DF32AdamsPlans
    banks::Union{Nothing, NTuple{2, _DF32AdamsBank}}
    active::Int
    generation::UInt64
    site::Union{Nothing, DF32TrialSite}
    ready::Bool
    history::Union{Nothing, AdamsHistory{Array{ComplexF32, 3}, Array{ComplexF64, 3}}}
end

"""
    DF32AdamsFlow(execution, operator, state, spec, policy;
                  history=nothing, tabulate=false)

Prepare one CPU binary32 Adams owner with certified DF32 complete factors.

`state` is a finite strided ComplexF32 `(n_omega,n_columns)` matrix, used only for
shape/precision validation. `operator` is the exact original owned binary64 constant,
represented cumulative or real-weight expansion operator. `spec::WideStartSpec`
binds this actual start, schedule, source arithmetic, maximum order and optional
restored history. `policy::AdamsControlPolicy` is this session's immutable policy.

The owner retains one WideStartBinding, two scalar/coefficient banks, explicit
per-slot raw/DF32/error rate matrices for compensation and failure scratch. Restored
records are adopted without another copy. No source, history or field is advanced.
Dense, spatial, tabulated and device execution refuse before state allocation.
Throw `ArgumentError` for mismatched or unsupported inputs, and `DimensionMismatch`
for shapes. This internal preparation does not enable public DF32 admission.
"""
struct DF32AdamsFlow{E, O, B, R}
    execution::E
    operator::O
    binding::B
    policy::AdamsControlPolicy
    shape::Tuple{Int, Int}
    state_type::Type{ComplexF32}
    plans::_DF32AdamsPlans
    records::R
    occupied::Vector{Bool}
    selection::AdamsStencil
    positions::Vector{Float64}
    slots::Vector{Int}
    failures::Vector{Int}
    empty_rates::Matrix{_DF32AdamsScaled{DF32}}

    function DF32AdamsFlow(
            execution::E, operator::O, binding::B, policy::AdamsControlPolicy,
            shape::Tuple{Int, Int}, state_type::Type{ComplexF32}, plans::_DF32AdamsPlans,
            records::R, occupied::Vector{Bool}, selection::AdamsStencil,
            positions::Vector{Float64}, slots::Vector{Int}, failures::Vector{Int},
            empty_rates::Matrix{_DF32AdamsScaled{DF32}},
        ) where {
            E <: KernelExecution,
            O <: Union{DiagonalOperator, RepresentedOperator, ExpansionOperator},
            B <: WideStartBinding, R <: Tuple{Vararg{_DF32AdamsRate}},
        }
        return new{E, O, B, R}(
            execution, operator, binding, policy, shape, state_type,
            plans, records, occupied, selection, positions, slots, failures, empty_rates
        )
    end
end

function _da_empty_rate(prepared::WideOperator)
    shape = prepared.shape
    value = _da_zero(DF32)
    sample = WideRate(
        prepared, 0.0, false, fill(value, shape), zeros(ComplexF64, shape),
        :represented, zero(UInt64),
    )
    return _da_rate_record(WideBoundRate(zeros(ComplexF64, shape), sample, :df32))
end

function DF32AdamsFlow(
        execution::KernelExecution, operator, state::AbstractMatrix{ComplexF32},
        spec::WideStartSpec, policy::AdamsControlPolicy;
        history::Union{Nothing, WideStartHistory} = nothing, tabulate::Bool = false,
    )
    oncpu(execution) || throw(ArgumentError("DF32 Adams currently requires CPU"))
    spec.method === :adams && spec.order_max == policy.order ||
        throw(ArgumentError("DF32 Adams needs its actual method and maximum order"))
    policy.interpolate && throw(ArgumentError("DF32 Adams dense output is not qualified"))
    tabulate && throw(ArgumentError("DF32 Adams tabulation is not qualified"))
    operator isa Union{DiagonalOperator, RepresentedOperator, ExpansionOperator} ||
        throw(ArgumentError("DF32 Adams spatial operators are not qualified"))
    state isa StridedMatrix{ComplexF32} && !isempty(state) && all(isfinite, state) ||
        throw(ArgumentError("DF32 Adams needs finite strided binary32 state"))
    shape = _wide_start_shape(operator)
    shape == size(state) || throw(DimensionMismatch("DF32 Adams operator/state mismatch"))
    schedule = spec.schedule
    policy.z_end == last(schedule.planes) && policy.save_planes == schedule.planes &&
        policy.hard_kinks == schedule.hard_kinks &&
        policy.breakpoints == schedule.breakpoints && policy.stretch == schedule.stretch ||
        throw(ArgumentError("DF32 Adams policy and schedule topology disagree"))
    binding = prepare_wide_start(execution, operator, spec; history)
    compensated = _adams_varying_operator(operator) && !spec.source_absent
    # A completed start restores raw history but has no operator/factor domain.
    binding.operator !== nothing && compensated &&
        length(binding.rates) != length(binding.positions) &&
        throw(ArgumentError("DF32 compensated Adams needs restored rate provenance"))
    capacity = policy.order + 2
    records = _DF32AdamsRate[]
    occupied = fill(false, capacity)
    prepared = binding.operator
    if compensated && prepared !== nothing
        for slot in 1:capacity
            index = findfirst(==(slot - 1), binding.slots)
            record = index === nothing ? _da_empty_rate(prepared) :
                _da_rate_record(binding.rates[index])
            push!(records, record)
            occupied[slot] = index !== nothing
        end
    end
    rate() = compensated && prepared !== nothing ? _da_empty_rate(prepared) : nothing
    banks = prepared === nothing ? nothing :
        (
            _DF32AdamsBank(policy.order, rate(), rate()),
            _DF32AdamsBank(policy.order, rate(), rate()),
        )
    return DF32AdamsFlow(
        execution, operator, binding, policy, shape, ComplexF32,
        _DF32AdamsPlans(banks, 0, zero(UInt64), nothing, false, nothing),
        Tuple(records), occupied,
        AdamsStencil(policy.order), zeros(policy.order + 1), zeros(Int, policy.order + 1),
        zeros(Int, shape[2]), Matrix{_DF32AdamsScaled{DF32}}(undef, 0, 0),
    )
end

function DF32AdamsFlow(
        execution, operator, state, spec, policy; history = nothing,
        tabulate::Bool = false
    )
    throw(ArgumentError("DF32 Adams requires binary32 state and actual start/policy"))
end

_flow_shape(flow::DF32AdamsFlow) = flow.shape
function _da_banks(flow::DF32AdamsFlow)
    banks = flow.plans.banks
    banks === nothing && throw(ArgumentError("a completed DF32 Adams start has no trial"))
    return banks
end
function _flow_arithmetic(flow::DF32AdamsFlow, state, arithmetic)
    eltype(state) === ComplexF32 && arithmetic == flow.binding.spec.arithmetic ||
        throw(ArgumentError("DF32 Adams arithmetic or source scales disagree"))
    size(state) == flow.shape || throw(DimensionMismatch("DF32 Adams state shape differs"))
    return nothing
end

function _flow_start(flow::DF32AdamsFlow, policy::AdamsControlPolicy, position, schedule)
    policy === flow.policy && schedule === flow.binding.spec.schedule &&
        isequal(position, flow.binding.spec.start) ||
        throw(ArgumentError("DF32 Adams belongs to another actual start or policy"))
    return nothing
end

@inline _da_integer(::Type{DF32}, n::Int) = _df32_raw(Float32(n), 0.0f0)
@inline _da_integer(::Type{_WideBall}, n::Int) = _wide_split(_wide_host(Float64(n)))

@inline function _da_moments(phi::NTuple{8, Complex{T}}) where {T}
    return ntuple(Val(7)) do power
        _wide_cscale(phi[power + 1], _da_integer(T, factorial(power - 1)))
    end
end

@inline _da_moment(moments::Tuple, row, column, power) = moments[power]
@inline _da_moment(moments::AbstractArray, row, column, power) = moments[row, column, power]

@inline function _da_source_factor(
        coefficients, moments, row, column, sample, count, step::_DF32AdamsScaled{T},
    ) where {T}
    total = _da_scaled(_da_zero(T))
    for power in 1:count
        term = _da_mul(
            coefficients[sample, power],
            _da_scaled(_da_moment(moments, row, column, power))
        )
        total = _da_add(total, term)
    end
    return _da_mul(total, step)
end

@inline function _da_defect_factor(
        coefficients, moments, row, column, score, count,
        step::_DF32AdamsScaled{T}, normalization
    ) where {T}
    total = _da_scaled(_da_zero(T))
    for power in 1:count
        total = _da_add(
            total, _da_mul(
                coefficients[power, score],
                _da_scaled(_da_moment(moments, row, column, power))
            )
        )
    end
    factor = _da_mul(total, step)
    return _DF32AdamsScaled(factor.value, factor.exponent + normalization)
end

@inline function _da_compensation_factor(
        weight, phase::Complex{T}, divided::Complex{T},
        sampled::_DF32AdamsScaled{T}, step::_DF32AdamsScaled{T}, transport::T
    ) where {T}
    argument = _wide_cscale(divided, transport)
    exponential = _da_scaled(_da_exp(argument))
    difference = _da_mul(step, _da_add(sampled, _da_neg(_da_scaled(divided))))
    return _da_mul(weight, _da_mul(exponential, difference))
end

function _da_refresh_rate!(record::_DF32AdamsRate, position, before, execution)
    wide_rate!(record.sample, position, execution; before)
    for index in eachindex(record.saved)
        words, radius = record.sample.values[index], record.sample.errors[index]
        raw = ComplexF64(
            Float64(real(words).hi) + Float64(real(words).lo),
            Float64(imag(words).hi) + Float64(imag(words).lo)
        )
        real_radius = _wide_saved_radius(real(raw), real(words), real(radius))
        imag_radius = _wide_saved_radius(imag(raw), imag(words), imag(radius))
        r = _wide_split(_WideHost(real(raw), real_radius))
        i = _wide_split(_WideHost(imag(raw), imag_radius))
        r.valid && i.valid || throw(ArgumentError("DF32 Adams raw-rate conversion refused"))
        record.saved[index] = raw
        record.sample.values[index] = Complex{DF32}(r.value, i.value)
        record.sample.errors[index] = ComplexF64(r.error, i.error)
        record.numerical[index] = _da_compute(_da_checked_rate(record, index))
    end
    return nothing
end

function _da_copy_stencil!(destination, source)
    destination.max_order == source.max_order ||
        throw(DimensionMismatch("DF32 Adams stencil capacity differs"))
    copyto!(destination.slots, source.slots)
    copyto!(destination.orders, source.orders)
    copyto!(destination.nodes, source.nodes)
    destination.order, destination.count, destination.n_orders =
        source.order, source.count, source.n_orders
    return nothing
end

function _da_record(flow, bank, sample)
    if iszero(sample)
        return bank.endpoint_rate::_DF32AdamsRate
    end
    slot = bank.stencil.slots[sample]
    seed = bank.seed
    if seed !== nothing && slot == seed.slot
        return bank.seed_rate::_DF32AdamsRate
    end
    return flow.records[slot]
end

function _da_check_rate(history, slot, record)
    history.positions[slot] == record.sample.position &&
        (history.identities[slot] == 1) == record.sample.before ||
        throw(ArgumentError("DF32 Adams sampled rate position or knot side is stale"))
    for column in axes(record.saved, 2), row in axes(record.saved, 1)
        isequal(history.rates[row, column, slot], record.saved[row, column]) ||
            throw(ArgumentError("DF32 Adams stored rate differs from its provenance"))
    end
    return nothing
end

function _da_check_history(flow, history, stencil, seed, site)
    size(history.source)[1:2] == flow.shape && eltype(history.source) === ComplexF32 ||
        throw(DimensionMismatch("DF32 Adams history precision or shape differs"))
    history.max_order == stencil.max_order == flow.policy.order ||
        throw(DimensionMismatch("DF32 Adams history order differs"))
    history.compensated == (!isempty(flow.records)) ||
        throw(ArgumentError("DF32 Adams history compensation differs"))
    1 <= stencil.order <= stencil.count <= history.max_order &&
        1 <= stencil.n_orders <= 3 || throw(ArgumentError("invalid DF32 Adams stencil"))
    for score in 1:stencil.n_orders
        1 <= stencil.orders[score] <= stencil.count ||
            throw(ArgumentError("DF32 Adams score exceeds retained history"))
    end
    if seed !== nothing
        isequal(seed.position, site.origin) && !any(history.occupied) ||
            throw(ArgumentError("DF32 Adams virtual seed requires fresh history at origin"))
        seed.slot <= length(history.positions) && stencil.count == 1 &&
            stencil.slots[1] == seed.slot ||
            throw(ArgumentError("DF32 Adams virtual seed differs from its stencil"))
    end
    previous = Inf
    for index in 1:stencil.count
        slot = stencil.slots[index]
        1 <= slot <= length(history.positions) ||
            throw(ArgumentError("DF32 Adams selected slot is outside history"))
        virtual = seed !== nothing && slot == seed.slot
        virtual || history.occupied[slot] ||
            throw(ArgumentError("DF32 Adams selected an unoccupied sample"))
        point = virtual ? (seed::DF32AdamsSeed).position : history.positions[slot]
        isfinite(point) && point < previous ||
            throw(ArgumentError("DF32 Adams samples must be actual newest-first positions"))
        previous = point
        if !virtual && !isempty(flow.records)
            flow.occupied[slot] ||
                throw(ArgumentError("DF32 Adams sample has no rate provenance"))
            _da_check_rate(history, slot, flow.records[slot])
        end
    end
    return nothing
end

function _da_factor_certificate(flow, bank, site)
    prepared = flow.binding.operator::WideOperator
    source_error, compensation_error, argument_error = 0.0, 0.0, 0.0
    stencil, coefficients = bank.stencil, bank.checked
    compensated = !isempty(flow.records)
    for row in 1:prod(flow.shape)
        bundle = bank.operations[1]
        divided = _wide_graph(
            prepared.coefficients, prepared.terms,
            prepared.coefficient_errors, prepared.term_errors, prepared.kind,
            row, bundle, Val(:checked)
        )
        phase = _wide_result(divided, bundle, 0x00, divided)
        argument_error = max(argument_error, _da_argument(phase))
        flow.binding.spec.source_absent && continue
        moments = _da_moments(_da_phi(phase))
        for sample in 1:stencil.order
            factor = _da_source_factor(
                coefficients.predictor, moments,
                1, 1, sample, stencil.order, bank.step_checked
            )
            source_error = max(source_error, _da_narrow_error(factor))
        end
        for sample in 1:(stencil.order + 1)
            factor = _da_source_factor(
                coefficients.corrector, moments,
                1, 1, sample, stencil.order + 1, bank.step_checked
            )
            source_error = max(source_error, _da_narrow_error(factor))
        end
        for score in 1:stencil.n_orders
            factor = _da_defect_factor(
                coefficients.defect_polynomial, moments,
                1, 1, score, stencil.orders[score] + 1, bank.step_checked,
                bank.normalization[score]
            )
            source_error = max(source_error, _da_narrow_error(factor))
        end
        if compensated
            for sample in 0:stencil.count
                record = _da_record(flow, bank, sample)
                rate = _da_checked_rate(record, row)
                transport = iszero(sample) ? zero(_WideBall) :
                    bank.operations[sample + 1].transport
                argument_error = max(
                    argument_error,
                    _da_argument(_wide_cscale(divided, transport))
                )
                if 1 <= sample <= stencil.order
                    factor = _da_compensation_factor(
                        coefficients.compensation_predictor[sample], phase, divided,
                        rate, bank.step_checked, transport
                    )
                    compensation_error = max(compensation_error, _da_narrow_error(factor))
                end
                if sample <= stencil.order
                    factor = _da_compensation_factor(
                        coefficients.compensation_corrector[sample + 1], phase, divided,
                        rate, bank.step_checked, transport
                    )
                    compensation_error = max(compensation_error, _da_narrow_error(factor))
                end
                for score in 1:stencil.n_orders
                    sample <= stencil.orders[score] || continue
                    factor = _da_compensation_factor(
                        coefficients.compensation_defects[sample + 1, score],
                        phase, divided, rate, bank.step_checked, transport
                    )
                    compensation_error = max(compensation_error, _da_narrow_error(factor))
                end
            end
        end
    end
    flow.binding.spec.source_absent &&
        return DF32AdamsCertificate(0.0, 0.0, 0.0, argument_error)
    weight_error = 0.0
    for score in 1:stencil.n_orders, sample in 1:(stencil.orders[score] + 1)
        value = bank.checked.defect_samples[sample, score]
        normalized = _DF32AdamsScaled(
            value.value,
            value.exponent - bank.normalization[score]
        )
        weight_error = max(weight_error, _da_narrow_error(normalized))
    end
    return DF32AdamsCertificate(
        source_error, compensation_error, weight_error, argument_error
    )
end

"""
    DF32AdamsTrial

Bind a fully certified complete-factor plan to one history owner and generation.

Construct through `prepare_df32_adams_trial!`. The token retains its flow, exact
history owner, actual site and active generation. Only its matching numerical trial
and optional virtual seed may consume it. A successful later plan invalidates it;
a refused plan preserves it. It carries no independent device or full-run promise.
"""
struct DF32AdamsTrial{F, H}
    flow::F
    history::H
    site::DF32TrialSite
    generation::UInt64
end

"""
    prepare_df32_adams_trial!(flow, history, stencil, site; seed=nothing)

Certify all Adams complete factors before source, state, counters or history writes.

The flow owns the actual immutable operator/start/policy; the history is its owned
ComplexF32 source/state and ComplexF64 rate store. `stencil` selects actual newest-first
samples and scored orders. `site::DF32TrialSite` retains raw origin, supplied h,
canonical destination and replay cursor. An optional `DF32AdamsSeed` stages a new
accepted sample without inventing its source. The returned token binds every selected
rate discrepancy and generation. Only inactive coefficient/rate scratch changes on
failure; an earlier active token and all numerical/history arrays are preserved.

Throw named `ArgumentError` for unsupported domains, conditioning, stale provenance,
layout, ownership or method/coordinate mismatch. No fallback changes arithmetic or
history. Warmed CPU preparation allocates nothing.
"""
function prepare_df32_adams_trial!(
        flow::DF32AdamsFlow, history::AdamsHistory, stencil::AdamsStencil,
        site::DF32TrialSite; seed::Union{Nothing, DF32AdamsSeed} = nothing,
    )
    prepared = _df32_flow_site(flow, site)
    history isa AdamsHistory{Array{ComplexF32, 3}, Array{ComplexF64, 3}} ||
        throw(ArgumentError("DF32 Adams requires owned CPU history arrays"))
    flow.plans.history === nothing || flow.plans.history === history ||
        throw(ArgumentError("DF32 Adams flow belongs to another history owner"))
    _da_avoid_owner_arrays(
        (
            history.source, history.states, history.rates,
            history.positions, history.identities, history.occupied,
        ), flow
    )
    if stencil !== flow.selection
        _da_avoid_owner_arrays((stencil.slots, stencil.orders, stencil.nodes), flow)
    end
    _da_check_history(flow, history, stencil, seed, site)
    plans = flow.plans
    plans.generation < typemax(UInt64) ||
        throw(ArgumentError("DF32 Adams plan generation is exhausted"))
    generation = plans.generation + one(UInt64)
    bank_index = plans.active == 1 ? 2 : 1
    bank = _da_banks(flow)[bank_index]
    bank.step_checked = _da_host(_wide_host(site.step))
    bank.step_numerical = _da_compute(bank.step_checked)
    _da_copy_stencil!(bank.stencil, stencil)
    bank.seed = seed
    for index in 1:stencil.count
        slot = stencil.slots[index]
        virtual = seed !== nothing && slot == seed.slot
        bank.positions[index] = virtual ?
            (seed::DF32AdamsSeed).position : history.positions[slot]
        bank.kinds[index] = virtual ? 0 : history.identities[slot]
        bank.generations[index] = !isempty(flow.records) && !virtual ?
            flow.records[slot].sample.generation : zero(UInt64)
    end
    if !isempty(flow.records)
        if seed !== nothing
            _da_refresh_rate!(
                bank.seed_rate::_DF32AdamsRate, seed.position, false,
                flow.execution
            )
            _da_validate_publication(
                flow.records[seed.slot],
                bank.seed_rate::_DF32AdamsRate
            )
        end
        _da_refresh_rate!(
            bank.endpoint_rate::_DF32AdamsRate,
            site.origin + site.step, true, flow.execution
        )
    end
    if flow.binding.spec.source_absent
        _da_zero_coefficients!(_da_coefficient_arrays(bank.host), _wide_host(0.0))
    else
        _da_coefficients!(bank.host, stencil, bank.positions, site.origin, site.step)
    end
    _da_split_coefficients!(
        _da_coefficient_arrays(bank.host),
        _da_coefficient_arrays(bank.checked), _da_coefficient_arrays(bank.numerical)
    )
    fill!(bank.weights, 0)
    fill!(bank.normalization, 0)
    for score in 1:stencil.n_orders
        largest = 0.0
        for sample in 1:(stencil.orders[score] + 1)
            largest = max(largest, abs(bank.host.defect_samples[sample, score].value))
        end
        exponent = last(frexp(largest))
        bank.normalization[score] = exponent
        for sample in 1:(stencil.orders[score] + 1)
            weight = bank.numerical.defect_samples[sample, score]
            bank.weights[sample, score] = real(
                _da_narrow(
                    _DF32AdamsScaled(weight.value, weight.exponent - exponent)
                )
            )
        end
    end
    for index in 1:(stencil.count + 1)
        sample_z = index == 1 ? site.origin : bank.positions[index - 1]
        geometry = prepared.geometry
        geometry.history_lower <= sample_z <= geometry.history_upper ||
            throw(ArgumentError("DF32 Adams retained history exceeds its bound age"))
        bundle = _wide_scalars(
            prepared, site.origin, site.origin + site.step,
            site.step, sample_z, nothing, false, false
        )
        bank.operations[index] = bundle
        bank.bundles[index] = _wide_values(bundle)
    end
    bank.certificate = _da_factor_certificate(flow, bank, site)
    plans.active, plans.generation = bank_index, generation
    plans.site, plans.ready = site, false
    plans.history === nothing && (plans.history = history)
    return DF32AdamsTrial(flow, history, site, generation)
end

@generated function _da_owner_alias(array::A, flow::F) where {A, F <: DF32AdamsFlow}
    # Expand this complete retained graph while excluding the legitimate borrowed
    # history identity in plans. All restored matrices also occur in records.
    result = :(false)
    for name in (
            :operator, :binding, :records, :occupied, :selection, :positions,
            :slots, :failures, :empty_rates,
        )
        check = _retained_alias_expression(
            fieldtype(F, name),
            :(getfield(flow, $(QuoteNode(name))))
        )
        result = :($check || $result)
    end
    check = _retained_alias_expression(
        fieldtype(_DF32AdamsPlans, :banks), :(flow.plans.banks)
    )
    return quote
        isempty(array) && return false
        GC.@preserve array flow begin
            destination = array
            span = _transfer_storage_span(array)
            return $check || $result
        end
    end
end

@inline _da_avoid_owner_arrays(::Tuple{}, flow) = nothing
function _da_avoid_owner_arrays(arrays::Tuple, flow)
    _da_owner_alias(first(arrays), flow) &&
        throw(ArgumentError("DF32 Adams input overlaps retained factor storage"))
    return _da_avoid_owner_arrays(Base.tail(arrays), flow)
end

function _da_token(trial::DF32AdamsTrial; seeded::Bool = true)
    flow, plans, history = trial.flow, trial.flow.plans, trial.history
    plans.active != 0 && trial.generation == plans.generation &&
        trial.site === plans.site && history === plans.history ||
        throw(ArgumentError("DF32 Adams trial token is stale or belongs to another owner"))
    bank = _da_banks(flow)[plans.active]
    seed = bank.seed
    for index in 1:bank.stencil.count
        slot = bank.stencil.slots[index]
        virtual = seed !== nothing && slot == seed.slot
        if virtual && !seeded
            continue
        end
        history.occupied[slot] && isequal(history.positions[slot], bank.positions[index]) &&
            history.identities[slot] == bank.kinds[index] ||
            throw(ArgumentError("DF32 Adams trial sampled history changed"))
        if !isempty(flow.records)
            flow.occupied[slot] &&
                flow.records[slot].sample.generation == bank.generations[index] ||
                throw(ArgumentError("DF32 Adams trial sampled-rate generation is stale"))
            _da_check_rate(history, slot, flow.records[slot])
        end
    end
    return bank
end

function _da_work_arrays(work)
    return (
        work.exponent, work.flow, work.predictor, work.source, work.candidate,
        work.endpoint_rate, work.endpoint_compensation, work.moments, work.carried,
        work.defects, work.norm_parts, work.norm_scale.mantissa, work.norm_scale.exponent,
        _da_coefficient_arrays(work.coefficients)...,
    )
end

@inline _da_work_pair(array, ::Tuple{}) = nothing
function _da_work_pair(array, others::Tuple)
    _transfer_arrays_alias(array, first(others)) &&
        throw(ArgumentError("DF32 Adams numerical work buffers overlap"))
    return _da_work_pair(array, Base.tail(others))
end

@inline _da_check_work_arrays(::Tuple{}, flow, state, history) = nothing
function _da_check_work_arrays(arrays::Tuple, flow, state, history)
    array = first(arrays)
    _transfer_writable_destination(array)
    _da_owner_alias(array, flow) &&
        throw(ArgumentError("DF32 Adams workspace overlaps retained factor storage"))
    _transfer_arrays_alias(array, state) &&
        throw(ArgumentError("DF32 Adams accepted state overlaps its workspace"))
    _retained_alias(array, history) &&
        throw(ArgumentError("DF32 Adams workspace overlaps sampled history"))
    _da_work_pair(array, Base.tail(arrays))
    return _da_check_work_arrays(Base.tail(arrays), flow, state, history)
end

@inline _da_work_shapes(::Tuple{}, shape) = nothing
function _da_work_shapes(arrays::Tuple, shape)
    size(first(arrays)) == shape ||
        throw(DimensionMismatch("DF32 Adams work shape differs"))
    return _da_work_shapes(Base.tail(arrays), shape)
end

@inline _da_coefficient_shapes(::Tuple{}, ::Tuple{}) = nothing
function _da_coefficient_shapes(arrays::Tuple, shapes::Tuple)
    size(first(arrays)) == first(shapes) ||
        throw(DimensionMismatch("DF32 Adams coefficient workspace shape differs"))
    return _da_coefficient_shapes(Base.tail(arrays), Base.tail(shapes))
end

function _da_trial_buffers(flow, work, state, history)
    work.factor_context === flow && eltype(work.exponent) === Complex{DF32} &&
        eltype(work.candidate) === ComplexF32 && eltype(state) === ComplexF32 ||
        throw(ArgumentError("DF32 Adams requires its matching prepared workspace"))
    state isa StridedMatrix{ComplexF32} && size(state) == flow.shape ||
        throw(ArgumentError("DF32 Adams requires matching strided accepted state"))
    work.max_order == flow.policy.order ||
        throw(DimensionMismatch("DF32 Adams workspace maximum order differs"))
    width = work.max_order + 1
    _da_coefficient_shapes(
        _da_coefficient_arrays(work.coefficients),
        (
            (width, width), (width, width), (width, 3), (width, 3), (width,),
            (width,), (width, 3), (width,), (width, width), (width,), (width,),
        )
    )
    _da_work_shapes(
        (
            work.exponent, work.flow, work.predictor, work.source, work.candidate,
            work.endpoint_rate, work.endpoint_compensation,
        ), flow.shape
    )
    size(work.moments) == (flow.shape..., work.max_order + 1) &&
        size(work.defects) == (flow.shape..., 3) &&
        size(work.carried) == (flow.shape..., 0) &&
        size(work.norm_parts) == (4, flow.shape[2]) &&
        length(work.norm_scale.mantissa) == length(work.norm_scale.exponent) ==
        flow.shape[1] ||
        throw(DimensionMismatch("DF32 Adams companion workspace shape differs"))
    _da_owner_alias(state, flow) &&
        throw(ArgumentError("DF32 Adams accepted state overlaps retained factor storage"))
    _da_check_work_arrays(_da_work_arrays(work), flow, state, history)
    _retained_alias(state, history) &&
        throw(ArgumentError("DF32 Adams accepted state overlaps sampled history"))
    return nothing
end

function _adams_flow_start(flow::DF32AdamsFlow, policy, z, schedule, source_absent)
    _flow_start(flow, policy, z, schedule)
    flow.plans.history === nothing ||
        throw(ArgumentError("DF32 Adams flow already belongs to a history owner"))
    source_absent == flow.binding.spec.source_absent ||
        throw(ArgumentError("DF32 Adams structural source policy differs"))
    return nothing
end

function _adams_session_history(
        execution, state, order, flow::DF32AdamsFlow;
        compensated
    )
    history = AdamsHistory(execution, state, order; compensated)
    flow.plans.history = history
    return history
end

function _adams_session_work(execution, state, order, flow::DF32AdamsFlow; compensated)
    _flow_arithmetic(flow, state, flow.binding.spec.arithmetic)
    order == flow.policy.order && compensated ==
        (_adams_varying_operator(flow.operator) && !flow.binding.spec.source_absent) ||
        throw(ArgumentError("DF32 Adams workspace policy differs"))
    rows, columns = size(state)
    narrow() = dzeros(execution, ComplexF32, rows, columns)
    return AdamsWork(
        fill(_da_zero(DF32), rows, columns), narrow(), narrow(), narrow(), narrow(),
        dzeros(execution, ComplexF64, rows, columns), narrow(),
        fill(_da_zero(DF32), rows, columns, order + 1),
        dzeros(execution, ComplexF32, rows, columns, 0),
        dzeros(execution, ComplexF32, rows, columns, 3),
        dzeros(execution, Float32, 4, columns), _march_norm_scale(state),
        AdamsCoefficients(order), order, compensated, 0.0, 0.0, 0, 0, false, false, flow,
    )
end

@inline _da_copy_host!(::Tuple{}, ::Tuple{}) = nothing
function _da_copy_host!(destination::Tuple, source::Tuple)
    for index in eachindex(first(source))
        first(destination)[index] = first(source)[index].value
    end
    return _da_copy_host!(Base.tail(destination), Base.tail(source))
end

function _da_fill_wide!(work, trial, bank, execution)
    prepared = trial.flow.binding.operator::WideOperator
    coefficients, terms, kind = prepared.coefficients, prepared.terms, prepared.kind
    bundle = bank.bundles[1]
    exponent, moments = work.exponent, work.moments
    absent, width = trial.flow.binding.spec.source_absent, size(moments, 3)
    each!(execution, length(exponent), 1) do row, unused
        divided = _wide_graph(
            coefficients, terms, nothing, nothing, kind, row,
            bundle, Val(:compute)
        )
        phase = _wide_result(divided, bundle, 0x00, divided)
        exponent[row] = phase
        if !absent
            computed = _da_moments(_da_phi(phase))
            for power in 1:width
                moments[row + (power - 1) * length(exponent)] = computed[power]
            end
        end
    end
    return nothing
end

function _da_sample_arrays(flow, bank)
    # A fixed tuple of array references is formed outside kernels; no host error
    # record, binary64 raw rate or retained owner enters the numerical closure.
    return ntuple(Val(7)) do index
        !isempty(flow.records) && index <= bank.stencil.count + 1 &&
            bank.endpoint_rate !== nothing ?
            _da_record(flow, bank, index - 1).numerical : flow.empty_rates
    end
end

function _da_predict!(work, state, history, trial, bank, execution)
    phase, moments = work.exponent, work.moments
    linear, predictor = work.flow, work.predictor
    store, states, slots = history.source, history.states, bank.stencil.slots
    coefficients, compensation = bank.numerical.predictor,
        bank.numerical.compensation_predictor
    rate_arrays = _da_sample_arrays(trial.flow, bank)
    bundles, order, compensated = bank.bundles, bank.stencil.order, work.compensated
    step = bank.step_numerical
    prepared = trial.flow.binding.operator::WideOperator
    terms, original, kind, rows = prepared.terms, prepared.coefficients,
        prepared.kind, prepared.shape[1]
    each!(execution, size(state)...) do row, column
        argument = phase[row, column]
        factor = iszero(argument) ? one(ComplexF32) : _df32_exp_pair(argument)[1]
        linear[row, column] = factor
        increment = zero(ComplexF32)
        divided = compensated ? _da_divided(
                original, terms, kind, rows,
                bundles[1], row, column
            ) : argument
        for sample in 1:order
            weight = _da_source_factor(
                coefficients, moments, row, column, sample, order, step
            )
            increment += _da_narrow(weight) * store[row, column, slots[sample]]
        end
        if compensated
            for sample in 1:order
                rate = rate_arrays[sample + 1][row, column]
                transport = bundles[sample + 1].transport
                weight = _da_compensation_factor(
                    compensation[sample], argument,
                    divided, rate, step, transport
                )
                increment += _da_narrow(weight) * states[row, column, slots[sample]]
            end
        end
        baseline = iszero(argument) ? state[row, column] : factor * state[row, column]
        predictor[row, column] = baseline + increment
    end
    return nothing
end

@inline function _da_divided(coefficients, terms, kind, rows, bundle, row, column)
    index = row + (column - 1) * rows
    return _wide_graph(
        coefficients, terms, nothing, nothing, kind,
        index, bundle, Val(:compute)
    )
end

function _da_correct!(work, state, history, trial, bank, execution)
    phase, moments, linear = work.exponent, work.moments, work.flow
    source, predictor, candidate, defects = work.source, work.predictor, work.candidate,
        work.defects
    store, states, slots = history.source, history.states, bank.stencil.slots
    coefficients, compensation = bank.numerical.corrector,
        bank.numerical.compensation_corrector
    polynomial, compensation_defects = bank.numerical.defect_polynomial,
        bank.numerical.compensation_defects
    weights, normalization = bank.weights, bank.normalization
    rate_arrays = _da_sample_arrays(trial.flow, bank)
    bundles, order, compensated = bank.bundles, bank.stencil.order, work.compensated
    orders, n_orders, step = bank.stencil.orders, bank.stencil.n_orders, bank.step_numerical
    prepared = trial.flow.binding.operator::WideOperator
    terms, original, kind, rows = prepared.terms, prepared.coefficients,
        prepared.kind, prepared.shape[1]
    each!(execution, size(state)...) do row, column
        argument = phase[row, column]
        divided = compensated ? _da_divided(
                original, terms, kind, rows,
                bundles[1], row, column
            ) : argument
        increment = zero(ComplexF32)
        for sample in 1:(order + 1)
            weight = _da_source_factor(
                coefficients, moments,
                row, column, sample, order + 1, step
            )
            value = sample == 1 ? source[row, column] :
                store[row, column, slots[sample - 1]]
            increment += _da_narrow(weight) * value
            if compensated
                sampled = sample == 1 ? predictor[row, column] :
                    states[row, column, slots[sample - 1]]
                transport = sample == 1 ? zero(DF32) : bundles[sample].transport
                weight = _da_compensation_factor(
                    compensation[sample], argument,
                    divided, rate_arrays[sample][row, column], step, transport
                )
                increment += _da_narrow(weight) * sampled
            end
        end
        baseline = iszero(argument) ? state[row, column] :
            linear[row, column] * state[row, column]
        candidate[row, column] = baseline + increment
        for score in 1:3
            defect = zero(ComplexF32)
            if score <= n_orders
                scored = orders[score]
                difference = zero(ComplexF32)
                for sample in 1:(scored + 1)
                    value = sample == 1 ? source[row, column] :
                        store[row, column, slots[sample - 1]]
                    difference += weights[sample, score] * value
                end
                factor = _da_defect_factor(
                    polynomial, moments, row, column,
                    score, scored + 1, step, normalization[score]
                )
                defect = _da_narrow(factor) * difference
                if compensated
                    for sample in 1:(scored + 1)
                        sampled = sample == 1 ? predictor[row, column] :
                            states[row, column, slots[sample - 1]]
                        transport = sample == 1 ? zero(DF32) : bundles[sample].transport
                        factor = _da_compensation_factor(
                            compensation_defects[sample, score],
                            argument, divided, rate_arrays[sample][row, column],
                            step, transport
                        )
                        defect += _da_narrow(factor) * sampled
                    end
                end
            end
            defects[row, column, score] = defect
        end
    end
    return nothing
end

function _da_validate_publication(destination::_DF32AdamsRate, source::_DF32AdamsRate)
    destination.sample.generation < typemax(UInt64) ||
        throw(ArgumentError("DF32 Adams sampled-rate generation is exhausted"))
    size(destination.saved) == size(source.saved) == size(destination.sample.values) ==
        size(source.sample.values) == size(destination.sample.errors) ==
        size(source.sample.errors) == size(destination.numerical) ==
        size(source.numerical) ||
        throw(DimensionMismatch("DF32 Adams rate storage differs"))
    return nothing
end

function _da_publish_rate!(destination::_DF32AdamsRate, source::_DF32AdamsRate)
    # Shapes/generation capacity were checked before numerical/history writes.
    # Owned buffers cannot be replaced during this synchronous publication.
    copyto!(destination.saved, source.saved)
    copyto!(destination.sample.values, source.sample.values)
    copyto!(destination.sample.errors, source.sample.errors)
    copyto!(destination.numerical, source.numerical)
    destination.origin_wide = :df32
    destination.sample.position, destination.sample.before, destination.sample.provenance =
        source.sample.position, source.sample.before, :represented
    destination.sample.generation += one(UInt64)
    return nothing
end

struct _DF32AdamsSeedRate{T}
    trial::T
end

function (rate::_DF32AdamsSeedRate)(output, position, before)
    trial = rate.trial
    bank = _da_token(trial; seeded = false)
    seed = bank.seed
    seed !== nothing && position == seed.position && !before ||
        throw(ArgumentError("DF32 Adams seed-rate coordinates changed"))
    copyto!(output, (bank.seed_rate::_DF32AdamsRate).saved)
    return nothing
end

function _da_seed!(trial::DF32AdamsTrial, state, source!, execution)
    bank = _da_token(trial; seeded = false)
    seed = bank.seed
    seed !== nothing || throw(ArgumentError("DF32 Adams trial has no virtual seed"))
    history, flow = trial.history, trial.flow
    _adams_slot(history, seed.slot, state)
    if !isempty(flow.records)
        _da_validate_publication(flow.records[seed.slot], bank.seed_rate::_DF32AdamsRate)
    end
    calls = adams_seed!(
        history, seed.slot, state, seed.position, source!,
        _DF32AdamsSeedRate(trial), execution; identity = 0,
        source_absent = flow.binding.spec.source_absent
    )
    if !isempty(flow.records)
        _da_publish_rate!(flow.records[seed.slot], bank.seed_rate::_DF32AdamsRate)
        flow.occupied[seed.slot] = true
        bank.generations[1] = flow.records[seed.slot].sample.generation
    end
    return calls
end

function _adams_precommit!(flow::DF32AdamsFlow, history::AdamsHistory, slot, work, identity)
    plans = flow.plans
    site = plans.site
    plans.ready && work.ready && history === plans.history && identity == 1 &&
        work.factor_context === flow && site !== nothing ||
        throw(ArgumentError("DF32 Adams acceptance has no matching completed trial"))
    trial = DF32AdamsTrial(flow, history, site, plans.generation)
    bank = _da_token(trial)
    _adams_slot(history, slot, work.source)
    if !isempty(flow.records)
        _da_validate_publication(flow.records[slot], bank.endpoint_rate::_DF32AdamsRate)
        isequal(work.endpoint_rate, (bank.endpoint_rate::_DF32AdamsRate).saved) ||
            throw(ArgumentError("DF32 Adams accepted endpoint rate changed"))
    end
    return nothing
end

function _adams_committed!(flow::DF32AdamsFlow, history, slot, work)
    bank = _da_banks(flow)[flow.plans.active]
    if !isempty(flow.records)
        _da_publish_rate!(flow.records[slot], bank.endpoint_rate::_DF32AdamsRate)
        flow.occupied[slot] = true
    end
    flow.plans.ready = false
    flow.plans.active = 0
    return nothing
end

function _adams_reset_flow!(flow::DF32AdamsFlow)
    fill!(flow.occupied, false)
    flow.plans.ready = false
    flow.plans.active = 0
    return nothing
end

function _adams_plan_trial!(session, flow::DF32AdamsFlow)
    fresh = iszero(session.control.held)
    held = fresh ? 1 : session.control.held
    order = fresh ? (session.policy.variable_order ? 1 : session.policy.order) :
        session.control.order
    copyto!(flow.positions, session.positions)
    copyto!(flow.slots, session.slots)
    if fresh
        flow.positions[1], flow.slots[1] = session.state.z, 1
    end
    quiet = isempty(session.schedule.replay_steps) ?
        _session_quiet(session.services, session) : false
    landing = _adams_session_landing(
        session; commit_quiet = false,
        stored_order = order, supplied_quiet = quiet
    )
    adams_select_stencil!(
        flow.selection, session.policy, flow.positions, flow.slots,
        held, order; z = session.state.z, dz = landing.dz,
        last_rejected = session.control.last_rejected
    )
    replay = isempty(session.schedule.replay_steps) ? 0 : session.state.replay
    site = DF32TrialSite(session.state.z, landing.dz, landing.destination, replay)
    seed = fresh ? DF32AdamsSeed(1, session.state.z) : nothing
    # Even seed writes must be disjoint from the complete retained owner.
    _da_trial_buffers(flow, session.work, session.field, session.history)
    trial = prepare_df32_adams_trial!(flow, session.history, flow.selection, site; seed)
    if fresh
        session.ledger.startup = Base.Checked.checked_add(session.ledger.startup, 1)
        _da_seed!(trial, session.field, session.sampler, session.execution)
        session.positions[1], session.slots[1] = session.state.z, 1
        session.control.held, session.control.order = 1, order
    end
    isempty(session.schedule.replay_steps) && (session.control.collar_was_quiet = quiet)
    _da_copy_stencil!(session.stencil, flow.selection)
    session.state.flight_z, session.state.flight_dz = session.state.z, landing.dz
    return landing, trial
end

function _adams_trial_graph!(
        flow::DF32AdamsFlow, work::AdamsWork, state, position::Real, step::Real,
        history::AdamsHistory, stencil::AdamsStencil, source!, trial::DF32AdamsTrial,
        rate!, execution::KernelExecution; source_absent::Bool = false,
    )
    flow === trial.flow && history === trial.history &&
        isequal(position, trial.site.origin) && isequal(step, trial.site.step) ||
        throw(ArgumentError("DF32 Adams trial differs from its certified owner or site"))
    oncpu(execution) || throw(ArgumentError("DF32 Adams numerical trials require CPU"))
    rate! === nothing || (rate! isa _AdamsSessionRate && rate!.flow === flow) ||
        throw(ArgumentError("DF32 Adams rejects an uncertified rate callback"))
    source_absent == flow.binding.spec.source_absent ||
        throw(ArgumentError("DF32 Adams structural source policy changed"))
    bank = _da_token(trial)
    stencil.order == bank.stencil.order && stencil.count == bank.stencil.count &&
        stencil.n_orders == bank.stencil.n_orders && stencil.slots == bank.stencil.slots &&
        stencil.orders == bank.stencil.orders ||
        throw(ArgumentError("DF32 Adams selected stencil changed after certification"))
    _da_trial_buffers(flow, work, state, history)
    _adams_trial_shapes(work, state, history, stencil)
    all(isfinite, state) || throw(ArgumentError("DF32 Adams accepted state is nonfinite"))
    work.ready, flow.plans.ready = false, false
    _da_copy_host!(
        _da_coefficient_arrays(work.coefficients), _da_coefficient_arrays(bank.host)
    )
    _da_fill_wide!(work, trial, bank, execution)
    fill!(work.endpoint_compensation, 0)
    if source_absent
        exponent, candidate, predictor, linear =
            work.exponent, work.candidate, work.predictor, work.flow
        each!(execution, size(state)...) do row, column
            phase = exponent[row, column]
            factor = iszero(phase) ? one(ComplexF32) : _df32_exp_pair(phase)[1]
            value = iszero(phase) ? state[row, column] : factor * state[row, column]
            linear[row, column] = factor
            predictor[row, column], candidate[row, column] = value, value
        end
        fill!(work.source, 0)
        fill!(work.defects, 0)
        fill!(work.endpoint_rate, 0)
        fill!(work.moments, _da_zero(DF32))
    else
        _da_predict!(work, state, history, trial, bank, execution)
        all(isfinite, work.predictor) ||
            throw(ArgumentError("DF32 Adams predictor is nonfinite"))
        _adams_callback(
            source!(
                work.source, work.predictor, trial.site.origin + trial.site.step
            )
        )
        _da_token(trial)
        if work.compensated
            copyto!(work.endpoint_rate, (bank.endpoint_rate::_DF32AdamsRate).saved)
        else
            fill!(work.endpoint_rate, 0)
        end
        _da_correct!(work, state, history, trial, bank, execution)
    end
    all(isfinite, work.source) && all(isfinite, work.candidate) &&
        all(isfinite, work.defects) ||
        throw(ArgumentError("DF32 Adams completed trial is nonfinite"))
    work.position, work.step = trial.site.origin, trial.site.step
    work.order, work.n_orders = stencil.order, stencil.n_orders
    work.source_absent, work.ready, flow.plans.ready = source_absent, true, true
    return source_absent ? 0 : 1
end

function _adams_trial_graph!(
        flow::DF32AdamsFlow, work, state, position, step,
        history, stencil, source!, exponent!, rate!, execution; source_absent = false
    )
    throw(ArgumentError("DF32 Adams requires a certified complete-factor trial token"))
end
