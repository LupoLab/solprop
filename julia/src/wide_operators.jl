# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public WideGeometry, WideOperator, WideRate, WideOperation, WideCertificate
public prepare_wide_operator, prepare_wide_rate, restore_wide_rate
public wide_operation, wide_exponent!, wide_transport!, wide_compensation!, wide_rate!

# The CPU certificate and the Float32-only graph are distinct specializations of
# the same expression. WIDE_ASSEMBLY.md derives the outward radii, including gradual
# underflow. None of these host checks establishes a device arithmetic model.
const _WIDE_U = 2.0^-24
const _WIDE_ETA = 2.0^-150
const _WIDE_ADD = nextfloat(3 * _WIDE_U^2 / (1 - 4 * _WIDE_U))
const _WIDE_MUL = 7 * _WIDE_U^2
const _WIDE_LIMIT = Float64(floatmax(Float32)) / 32
const _WIDE_BUDGET = 2.0^-27

@inline _wide_sum(x::Float64, y::Float64) = iszero(x) && iszero(y) ? 0.0 : nextfloat(x + y)
@inline _wide_product(x::Float64, y::Float64) =
    iszero(x) || iszero(y) ? 0.0 : nextfloat(x * y)
@inline _wide_quotient(x::Float64, y::Float64) = iszero(x) ? 0.0 : nextfloat(x / y)
@inline _wide_magnitude(x::DF32) = _wide_sum(abs(Float64(x.hi)), abs(Float64(x.lo)))

struct _WideBall <: Real
    value::DF32
    error::Float64
    peak::Float64
    valid::Bool

    _WideBall(value::DF32, error::Float64, peak::Float64, valid::Bool) =
        new(value, error, peak, valid)
end

@inline _wide_invalid() = _WideBall(zero(DF32), Inf, Inf, false)
@inline Base.zero(::Type{_WideBall}) = _WideBall(zero(DF32), 0.0, 0.0, true)
@inline Base.:-(x::_WideBall) = _WideBall(-x.value, x.error, x.peak, x.valid)

@inline function Base.:+(x::_WideBall, y::_WideBall)
    x.valid && y.valid || return _wide_invalid()
    magnitude = _wide_sum(_wide_magnitude(x.value), _wide_magnitude(y.value))
    magnitude <= _WIDE_LIMIT || return _wide_invalid()
    result = x.value + y.value
    # |c-s| <= A|s| implies |c-s| <= A|c|/(1-A). This keeps exact
    # cancellation sharp without assuming a nonzero residual has a lower bound.
    local_error = _wide_quotient(
        _wide_product(_WIDE_ADD, _wide_magnitude(result)), prevfloat(1 - _WIDE_ADD),
    )
    error = _wide_sum(_wide_sum(x.error, y.error), local_error)
    return _WideBall(result, error, max(x.peak, y.peak, magnitude), isfinite(error))
end

@inline Base.:-(x::_WideBall, y::_WideBall) = x + (-y)

@inline function Base.:*(x::_WideBall, y::_WideBall)
    x.valid && y.valid || return _wide_invalid()
    mx, my = _wide_magnitude(x.value), _wide_magnitude(y.value)
    magnitude = _wide_product(mx, my)
    max(mx, my, magnitude) <= _WIDE_LIMIT || return _wide_invalid()
    result = x.value * y.value
    local_error = _wide_quotient(
        _wide_sum(_wide_product(_WIDE_MUL, _wide_magnitude(result)), 5 * _WIDE_ETA),
        prevfloat(1 - _WIDE_MUL),
    )
    inherited = _wide_sum(
        _wide_sum(_wide_product(mx, y.error), _wide_product(my, x.error)),
        _wide_product(x.error, y.error),
    )
    error = _wide_sum(inherited, local_error)
    return _WideBall(result, error, max(x.peak, y.peak, magnitude), isfinite(error))
end

struct _WideHost
    value::Float64
    error::Float64

    _WideHost(value::Float64, error::Float64) = new(value, error)
end

@inline _wide_host(x::Float64) = _WideHost(x, 0.0)
@inline function _wide_host_error(x::Float64)
    iszero(x) && return nextfloat(0.0)
    return _wide_sum(_wide_quotient(abs(x), 2.0^53 - 1), nextfloat(0.0))
end

@inline function Base.:+(x::_WideHost, y::_WideHost)
    result = x.value + y.value
    error = _wide_sum(_wide_sum(x.error, y.error), _wide_host_error(result))
    return _WideHost(result, error)
end

@inline Base.:-(x::_WideHost) = _WideHost(-x.value, x.error)
@inline Base.:-(x::_WideHost, y::_WideHost) = x + (-y)

@inline function Base.:*(x::_WideHost, y::_WideHost)
    result = x.value * y.value
    error = _wide_sum(
        _wide_sum(
            _wide_product(abs(x.value), y.error), _wide_product(abs(y.value), x.error),
        ),
        _wide_sum(_wide_product(x.error, y.error), _wide_host_error(result)),
    )
    return _WideHost(result, error)
end

@inline function Base.:/(x::_WideHost, y::_WideHost)
    denominator = prevfloat(abs(y.value) - y.error)
    denominator > 0 || return _WideHost(0.0, Inf)
    result = x.value / y.value
    rounding = _wide_host_error(result)
    inherited = _wide_quotient(
        _wide_sum(x.error, _wide_product(_wide_sum(abs(result), rounding), y.error)),
        denominator,
    )
    return _WideHost(result, _wide_sum(inherited, rounding))
end

function _wide_split(x::_WideHost)
    isfinite(x.value) && isfinite(x.error) && x.error >= 0 || return _wide_invalid()
    high = Float32(x.value)
    isfinite(high) || return _wide_invalid()
    residual = x.value - Float64(high)
    low = Float32(residual)
    _wide_sum(abs(Float64(high)), abs(Float64(low))) <= _WIDE_LIMIT ||
        return _wide_invalid()
    hi, lo = _df32_fast_two_sum(high, low)
    value = _df32_raw(hi, lo)
    # Both subtractions are exact by Sterbenz (or the rounded word is zero).
    # Fast2Sum preserves the word sum, also when either split word is subnormal.
    split_error = abs(residual - Float64(low))
    error = _wide_sum(x.error, split_error)
    magnitude = _wide_magnitude(value)
    return _WideBall(value, error, magnitude, magnitude <= _WIDE_LIMIT)
end

"""
    WideGeometry(lower, upper; history_lower=lower, history_upper=upper,
                 step_max=upper-lower, step_min=nothing)

Bind finite host-coordinate and actual-step limits for CPU wide assembly.

Positions and positive steps are in metres. Supply the actual remaining replay ceiling,
including steps exceeding a controller's `dz_max`. `step_min=nothing` permits arbitrarily
short positive forced landings. History bounds include the actual retained coordinates,
including a newest sample slightly after its accepted anchor. These scalar bounds can be
lowered from `PhaseSpans`; they are geometry, not an operator-error certificate.
"""
struct WideGeometry
    lower::Float64
    upper::Float64
    history_lower::Float64
    history_upper::Float64
    step_max::Float64
    step_min::Float64

    function WideGeometry(
            lower::Float64, upper::Float64; history_lower::Float64 = lower,
            history_upper::Float64 = upper, step_max::Float64 = upper - lower,
            step_min::Union{Nothing, Float64} = nothing,
        )
        minimum = step_min === nothing ? 0.0 : step_min
        all(isfinite, (lower, upper, history_lower, history_upper, step_max, minimum)) &&
            lower < upper && history_lower <= history_upper &&
            0 <= minimum <= step_max && step_max > 0 ||
            throw(ArgumentError("wide geometry requires finite ordered spans and steps"))
        return new(lower, upper, history_lower, history_upper, step_max, minimum)
    end
end

"""
    WideOperator

Retain owned DF32 operator data and separate CPU certificate inputs.

Construct with `prepare_wide_operator`. Arrays are immutable numerical inputs after
preparation; mutating them invalidates every bound and operation. No host certificate
array is read by a device kernel, and this initial implementation admits CPU only.
"""
struct WideOperator
    geometry::WideGeometry
    shape::Tuple{Int, Int}
    kind::UInt8
    coefficients::Array{Complex{DF32}, 3}
    terms::Matrix{Complex{DF32}}
    coefficient_errors::Array{ComplexF64, 3}
    term_errors::Matrix{ComplexF64}
    edges::Vector{Float64}

    function WideOperator(
            geometry::WideGeometry, shape::Tuple{Int, Int}, kind::UInt8,
            coefficients::Array{Complex{DF32}, 3}, terms::Matrix{Complex{DF32}},
            coefficient_errors::Array{ComplexF64, 3}, term_errors::Matrix{ComplexF64},
            edges::Vector{Float64},
        )
        return new(
            geometry, shape, kind, coefficients, terms,
            coefficient_errors, term_errors, edges,
        )
    end
end

function _wide_copy(input::StridedArray{T, N}) where {T <: Union{Float64, ComplexF64}, N}
    output = Array{Complex{DF32}, N}(undef, size(input))
    errors = Array{ComplexF64, N}(undef, size(input))
    for index in eachindex(input)
        real_part = _wide_split(_wide_host(Float64(real(input[index]))))
        imag_part = _wide_split(_wide_host(Float64(imag(input[index]))))
        real_part.valid && imag_part.valid ||
            throw(ArgumentError("wide operator inputs exceed the protected finite range"))
        output[index] = Complex{DF32}(real_part.value, imag_part.value)
        errors[index] = ComplexF64(real_part.error, imag_part.error)
    end
    return output, errors
end

function _wide_host_copy(input::StridedArray{T, N}) where {T, N}
    output = Array{T, N}(undef, size(input))
    for index in eachindex(output)
        output[index] = input[index]
    end
    return output
end

function _wide_prepare(execution, coefficients, terms, edges, shape, kind, geometry)
    oncpu(execution) || throw(ArgumentError("wide certification requires CPU storage"))
    owned, errors = _wide_copy(coefficients)
    fixed, fixed_errors = _wide_copy(terms)
    return WideOperator(
        geometry, shape, kind, owned, fixed, errors, fixed_errors, copy(edges),
    )
end

"""
    prepare_wide_operator(execution, operator; geometry::WideGeometry)

Copy a constant, cumulative Chebyshev or real-weight expansion operator for wide assembly.

The mathematical target is the supplied binary64 representation, including independent
interval constants and jumps. No physical operator is resampled. Certificate inputs are
derived from these copies; a caller's unbound envelope is never trusted. Coefficients and
terms must use strided CPU binary64 storage; zero and subnormal split words are permitted
under the explicit
gradual-underflow model in `WIDE_ASSEMBLY.md`. Unsupported storage or range refuses before
any caller buffer is written. Preparation allocates; operations and writes are warmed
allocation-free on qualified CPU execution.
"""
function prepare_wide_operator(
        execution::KernelExecution, operator::DiagonalOperator; geometry::WideGeometry,
    )
    operator.rate isa StridedMatrix{ComplexF64} ||
        throw(ArgumentError("wide preparation requires strided CPU binary64 inputs"))
    coefficients = reshape(_wide_host_copy(operator.rate), length(operator.rate), 1, 1)
    return _wide_prepare(
        execution, coefficients, zeros(ComplexF64, length(operator.rate), 0), Float64[],
        size(operator.rate), 0x00, geometry,
    )
end

function _wide_table_geometry(table, geometry)
    table.breakpoints isa StridedVector{Float64} && table.coefficients isa Union{
        StridedArray{Float64, 3}, StridedArray{ComplexF64, 3},
    } || throw(ArgumentError("wide tables require strided CPU binary64 inputs"))
    table.lower <= geometry.lower <= geometry.upper <= table.upper ||
        throw(ArgumentError("wide geometry exceeds the represented table's admitted span"))
    return nothing
end

function prepare_wide_operator(
        execution::KernelExecution, operator::RepresentedOperator; geometry::WideGeometry,
    )
    table = operator.cumulative
    _wide_table_geometry(table, geometry)
    shape = (operator.n_frequency, operator.n_columns)
    return _wide_prepare(
        execution, table.coefficients, zeros(ComplexF64, prod(shape), 0),
        table.breakpoints, shape, 0x01, geometry,
    )
end

function prepare_wide_operator(
        execution::KernelExecution, operator::ExpansionOperator; geometry::WideGeometry,
    )
    table = operator.cumulative_weights
    _wide_table_geometry(table, geometry)
    operator.terms isa StridedArray{ComplexF64, 3} &&
        eltype(table.coefficients) === Float64 ||
        throw(ArgumentError("wide expansion needs binary64 complex terms and real weights"))
    shape = (size(operator.terms, 1), size(operator.terms, 2))
    terms = reshape(_wide_host_copy(operator.terms), prod(shape), size(operator.terms, 3))
    return _wide_prepare(
        execution, table.coefficients, terms, table.breakpoints, shape, 0x02, geometry,
    )
end

struct _WideScalars{T}
    x0::T
    x1::T
    beta::T
    inverse_step::T
    step::T
    distance::T
    transport::T
    interval0::Int
    interval1::Int
    derivative::Bool
    zero_distance::Bool

    function _WideScalars(
            x0::T, x1::T, beta::T, inverse_step::T, step::T, distance::T, transport::T,
            interval0::Int, interval1::Int, derivative::Bool, zero_distance::Bool,
        ) where {T <: Union{DF32, _WideBall}}
        return new{T}(
            x0, x1, beta, inverse_step, step, distance, transport,
            interval0, interval1, derivative, zero_distance,
        )
    end
end

@inline function _wide_values(bundle::_WideScalars{_WideBall})
    return _WideScalars(
        bundle.x0.value, bundle.x1.value, bundle.beta.value, bundle.inverse_step.value,
        bundle.step.value, bundle.distance.value, bundle.transport.value,
        bundle.interval0, bundle.interval1, bundle.derivative, bundle.zero_distance,
    )
end

@inline _wide_load(value::Complex{DF32}, error, ::Val{:compute}) = value
@inline function _wide_load(value::Complex{DF32}, error::ComplexF64, ::Val{:checked})
    real_value = _WideBall(real(value), real(error), _wide_magnitude(real(value)), true)
    imag_value = _WideBall(imag(value), imag(error), _wide_magnitude(imag(value)), true)
    return Complex{_WideBall}(real_value, imag_value)
end

@inline function _wide_coefficient(coefficients, errors, i, j, k, ::Val{:compute})
    return coefficients[i, j, k]
end
@inline function _wide_coefficient(coefficients, errors, i, j, k, mode::Val{:checked})
    return _wide_load(coefficients[i, j, k], errors[i, j, k], mode)
end

@inline _wide_term(terms, errors, i, j, ::Val{:compute}) = terms[i, j]
@inline function _wide_term(terms, errors, i, j, mode::Val{:checked})
    return _wide_load(terms[i, j], errors[i, j], mode)
end

@inline _wide_cadd(x::Complex{T}, y::Complex{T}) where {T} =
    Complex{T}(real(x) + real(y), imag(x) + imag(y))
@inline _wide_csub(x::Complex{T}, y::Complex{T}) where {T} =
    Complex{T}(real(x) - real(y), imag(x) - imag(y))
@inline _wide_cneg(x::Complex{T}) where {T} = Complex{T}(-real(x), -imag(x))
@inline _wide_cscale(x::Complex{T}, y::T) where {T} =
    Complex{T}(real(x) * y, imag(x) * y)

@inline function _wide_value(coefficients, errors, row, interval, coordinate, mode)
    count = size(coefficients, 2)
    current = _wide_coefficient(coefficients, errors, row, count, interval, mode)
    count == 1 && return current
    previous = _wide_coefficient(coefficients, errors, row, count - 1, interval, mode)
    twice = coordinate + coordinate
    for index in (count - 2):-1:1
        saved = previous
        previous = _wide_csub(
            _wide_coefficient(coefficients, errors, row, index, interval, mode), current,
        )
        current = _wide_cadd(saved, _wide_cscale(current, twice))
    end
    return _wide_cadd(previous, _wide_cscale(current, coordinate))
end

@inline function _wide_divided(
        coefficients, errors, row, bundle::_WideScalars{T}, mode,
    ) where {T}
    count = size(coefficients, 2)
    neutral = Complex{T}(zero(T), zero(T))
    count == 1 && return neutral
    interval = bundle.interval0
    current = _wide_coefficient(coefficients, errors, row, count, interval, mode)
    count == 2 && return _wide_cscale(current, bundle.beta)
    previous = _wide_coefficient(coefficients, errors, row, count - 1, interval, mode)
    dprevious, dcurrent = neutral, neutral
    twice0, twice1, twice_beta =
        bundle.x0 + bundle.x0, bundle.x1 + bundle.x1, bundle.beta + bundle.beta
    for index in (count - 2):-1:1
        dsaved = dprevious
        dprevious = _wide_cneg(dcurrent)
        dcurrent = _wide_cadd(
            _wide_cadd(dsaved, _wide_cscale(dcurrent, twice1)),
            _wide_cscale(current, twice_beta),
        )
        # The last base update is unused. In particular c0 cancels exactly and
        # never introduces its unrelated cumulative offset into the error radius.
        if index > 1
            saved = previous
            previous = _wide_csub(
                _wide_coefficient(coefficients, errors, row, index, interval, mode),
                current,
            )
            current = _wide_cadd(saved, _wide_cscale(current, twice0))
        end
    end
    # dcurrent used the old current in its last update; the base current must
    # advance once more for the final x*C term, without forming the unused c0-P.
    current = _wide_cadd(previous, _wide_cscale(current, twice0))
    return _wide_cadd(
        _wide_cadd(dprevious, _wide_cscale(dcurrent, bundle.x1)),
        _wide_cscale(current, bundle.beta),
    )
end

@inline function _wide_table_rate(coefficients, errors, row, bundle, mode)
    if bundle.interval0 == bundle.interval1
        return _wide_divided(coefficients, errors, row, bundle, mode)
    end
    first = _wide_value(coefficients, errors, row, bundle.interval0, bundle.x0, mode)
    last = _wide_value(coefficients, errors, row, bundle.interval1, bundle.x1, mode)
    return _wide_cscale(_wide_csub(last, first), bundle.inverse_step)
end

@inline function _wide_graph(
        coefficients, terms, errors, term_errors, kind, row, bundle, mode,
    )
    if bundle.zero_distance
        neutral = zero(typeof(bundle.step))
        return Complex{typeof(neutral)}(neutral, neutral)
    end
    if kind == 0x00
        rate = _wide_coefficient(coefficients, errors, row, 1, 1, mode)
        return _wide_cscale(rate, bundle.beta)
    elseif kind == 0x01
        return _wide_table_rate(coefficients, errors, row, bundle, mode)
    end
    first = _wide_term(terms, term_errors, row, 1, mode)
    weight = _wide_table_rate(coefficients, errors, 1, bundle, mode)
    total = _wide_cscale(first, real(weight))
    for term in 2:size(terms, 2)
        fixed = _wide_term(terms, term_errors, row, term, mode)
        weight = _wide_table_rate(coefficients, errors, term, bundle, mode)
        total = _wide_cadd(total, _wide_cscale(fixed, real(weight)))
    end
    return total
end

@inline function _wide_coordinate(position, lower, upper)
    p, a, b = _wide_host(position), _wide_host(lower), _wide_host(upper)
    # Stable affine map, exactly the same mathematical coordinate as (2z-a-b)/(b-a).
    return ((p - a) + (p - b)) / (b - a)
end

function _wide_scalars(
        prepared::WideOperator, z0, z1, step, sample_z, destination, derivative, before,
    )
    zero_host, one_host = _wide_host(0.0), _wide_host(1.0)
    distance = z0 == z1 ? zero_host : _wide_host(z1) - _wide_host(z0)
    h = _wide_host(step)
    ratio = derivative ? one_host : distance / h
    transport = destination === nothing ?
        h - (_wide_host(sample_z) - _wide_host(z0)) :
        _wide_host(destination) - _wide_host(sample_z)
    i0 = isempty(prepared.edges) ? 1 : _table_interval(prepared.edges, z0, before)
    i1 = isempty(prepared.edges) ? 1 : _table_interval(prepared.edges, z1, before)
    x0, x1, beta, inverse = zero_host, zero_host, ratio, zero_host
    if !isempty(prepared.edges)
        a0, b0 = prepared.edges[i0], prepared.edges[i0 + 1]
        a1, b1 = prepared.edges[i1], prepared.edges[i1 + 1]
        x0 = _wide_coordinate(z0, a0, b0)
        x1 = _wide_coordinate(z1, a1, b1)
        beta = (_wide_host(2.0) * ratio) / (_wide_host(b0) - _wide_host(a0))
        i0 != i1 && (inverse = one_host / h)
    end
    return _WideScalars(
        _wide_split(x0), _wide_split(x1), _wide_split(beta), _wide_split(inverse),
        _wide_split(h), _wide_split(distance), _wide_split(transport), i0, i1, derivative,
        !derivative && z0 == z1,
    )
end

"""
    WideCertificate

Describe a checked whole-expression CPU error and finite-range result.

`real_error` and `imag_error` are outward maximum absolute component errors, in the
output's units. `intermediate` bounds every checked scalar operand/sum/product in the
selected graph. `status` is `:valid`, `:intermediate_range`, `:assembly_error` or
`:function_domain`. Function arguments additionally require the sum of maximum errors
at most `2^-27` and exact stored-word bounds 16 and 2^20. A rate certificate has units
of inverse metres and does not borrow the dimensionless function-input tolerance.
"""
struct WideCertificate
    real_error::Float64
    imag_error::Float64
    intermediate::Float64
    status::Symbol
end

"""
    WideRate

Own an actual sampled DF32 rate together with its prior-error provenance.

`position` is the supplied sample coordinate in metres, never snapped to a later anchor.
`errors` stores nonnegative real/imaginary absolute radii in inverse metres. The rate is
either generated by `prepare_wide_rate` from the bound operator, or restored with an
explicit prior-error array by `restore_wide_rate`. No bare array implicitly claims exact
evaluation. Only `wide_rate!` may refresh the owned arrays; direct mutation is unsupported.
"""
mutable struct WideRate
    operator::WideOperator
    position::Float64
    before::Bool
    values::Matrix{Complex{DF32}}
    errors::Matrix{ComplexF64}
    provenance::Symbol
    generation::UInt64

    function WideRate(
            operator::WideOperator, position::Float64, before::Bool,
            values::Matrix{Complex{DF32}}, errors::Matrix{ComplexF64}, provenance::Symbol,
            generation::UInt64,
        )
        return new(operator, position, before, values, errors, provenance, generation)
    end
end

"""
    WideOperation

Bind one checked operation to owned coefficients, host scalar inputs and a certificate.

Construct with `wide_operation`; do not manufacture records or mutate their retained
inputs. This record has no device-admission meaning. Its certificate pass performs the
entire scalar graph on every CPU cell, then a writer repeats the graph using DF32 only.
"""
struct WideOperation{R}
    operator::WideOperator
    sample::R
    scalars::_WideScalars{DF32}
    scalar_certificate::_WideScalars{_WideBall}
    z0::Float64
    z1::Float64
    step::Float64
    sample_z::Float64
    destination::Union{Nothing, Float64}
    kind::Symbol
    certificate::WideCertificate
    sample_generation::UInt64

    function WideOperation(
            operator::WideOperator, sample::R, scalars::_WideScalars{DF32},
            scalar_certificate::_WideScalars{_WideBall},
            z0::Float64, z1::Float64, step::Float64, sample_z::Float64,
            destination::Union{Nothing, Float64}, kind::Symbol,
            certificate::WideCertificate, sample_generation::UInt64,
        ) where {R <: Union{Nothing, WideRate}}
        return new{R}(
            operator, sample, scalars, scalar_certificate,
            z0, z1, step, sample_z, destination, kind, certificate, sample_generation,
        )
    end
end

@inline _wide_kind(kind::Symbol) = kind === :exponent ? 0x00 :
    kind === :transport ? 0x01 : kind === :compensation ? 0x02 : 0x03

@inline function _wide_result(rate, bundle, kind::UInt8, sample)
    bundle.zero_distance && return kind == 0x02 ? sample : rate
    kind == 0x00 && return _wide_cscale(rate, bundle.step)
    kind == 0x01 && return _wide_cscale(rate, bundle.transport)
    kind == 0x02 && return _wide_csub(sample, rate)
    return rate
end

@inline function _wide_sample(::Nothing, row, mode, ::Type{T}) where {T}
    return Complex{T}(zero(T), zero(T))
end
@inline function _wide_sample(sample::WideRate, row, mode, ::Type{T}) where {T}
    return _wide_load(sample.values[row], sample.errors[row], mode)
end

function _wide_certificate(prepared, bundle, kind, sample)
    real_error, imag_error, peak = 0.0, 0.0, 0.0
    domain = true
    for row in 1:prod(prepared.shape)
        rate = _wide_graph(
            prepared.coefficients, prepared.terms, prepared.coefficient_errors,
            prepared.term_errors, prepared.kind, row, bundle, Val(:checked),
        )
        sampled = kind === :compensation ?
            _wide_sample(sample, row, Val(:checked), _WideBall) :
            Complex{_WideBall}(zero(_WideBall), zero(_WideBall))
        result = _wide_result(rate, bundle, _wide_kind(kind), sampled)
        r, i = real(result), imag(result)
        r.valid && i.valid || return WideCertificate(Inf, Inf, Inf, :intermediate_range)
        real_error, imag_error = max(real_error, r.error), max(imag_error, i.error)
        peak = max(peak, r.peak, i.peak)
        domain &= _df32_abs_leq(r.value, 16.0f0) && _df32_abs_leq(i.value, 1048576.0f0)
    end
    argument = kind === :exponent || kind === :transport
    excessive = argument && _wide_sum(real_error, imag_error) > _WIDE_BUDGET
    status = argument && !domain ? :function_domain : excessive ? :assembly_error : :valid
    return WideCertificate(real_error, imag_error, peak, status)
end

function _wide_positions(prepared, z0, z1, step)
    geometry = prepared.geometry
    isfinite(z0) && isfinite(z1) &&
        geometry.lower <= z0 <= geometry.upper && geometry.lower <= z1 <= geometry.upper ||
        throw(ArgumentError("wide positions exceed the bound geometry"))
    isfinite(step) && 0 < step <= geometry.step_max && geometry.step_min <= step ||
        throw(ArgumentError("wide assembly requires an admitted actual positive step"))
    return nothing
end

"""
    wide_operation(prepared, z0, z1; step, kind=:exponent, sample=nothing,
                   sample_z=z0, destination=nothing)

Certify a dimensionless exponent/transport or an inverse-metre compensated rate.

Use `kind=:exponent` for Phi(z1)-Phi(z0), `:transport` for
`(step-(sample_z-z0))*(Phi(z1)-Phi(z0))/step`, and `:compensation` for the actual sampled
rate minus that divided difference. `step` and the actual endpoint distance remain
distinct. A dense `destination` replaces the transport numerator by
`destination-sample_z`; its exact represented-coordinate target is documented in
`WIDE_ASSEMBLY.md`. Compensation requires a `WideRate` bound to this preparation.

All positions/steps are finite binary64 metres. Host scalar formation and splitting
errors are retained. A same-interval paired Clenshaw recurrence avoids dividing two
independent endpoint errors by a tiny step. Cross-knot cumulative jumps retain that
conditioning explicitly and can return an invalid certificate. No output is written;
the writer refuses invalid certificates before mutation. No global `dz_min` is assumed.
"""
function wide_operation(
        prepared::WideOperator, z0::Float64, z1::Float64; step::Float64,
        kind::Symbol = :exponent, sample::Union{Nothing, WideRate} = nothing,
        sample_z::Float64 = sample === nothing ? z0 : sample.position,
        destination::Union{Nothing, Float64} = nothing,
    )
    kind in (:exponent, :transport, :compensation) ||
        throw(ArgumentError("wide kind requires exponent, transport or compensation"))
    _wide_positions(prepared, z0, z1, step)
    geometry = prepared.geometry
    isfinite(sample_z) && geometry.history_lower <= sample_z <= geometry.history_upper ||
        throw(ArgumentError("wide sample coordinate exceeds its retained-history geometry"))
    if destination !== nothing
        isfinite(destination) && geometry.lower <= destination <= geometry.upper ||
            throw(ArgumentError("wide dense destination exceeds the bound geometry"))
        kind === :transport ||
            throw(ArgumentError("only wide transport takes a destination"))
    end
    if sample !== nothing
        sample.operator === prepared && sample.position == sample_z ||
            throw(ArgumentError("wide operation needs its bound sampled-rate record"))
    elseif kind === :compensation
        throw(ArgumentError("wide compensation requires sampled-rate provenance"))
    end
    bundle = _wide_scalars(prepared, z0, z1, step, sample_z, destination, false, false)
    certificate = _wide_certificate(prepared, bundle, kind, sample)
    generation = sample === nothing ? zero(UInt64) : sample.generation
    return WideOperation(
        prepared, sample, _wide_values(bundle), bundle, z0, z1, step, sample_z,
        destination, kind, certificate, generation,
    )
end

function _wide_check_operation(operation::WideOperation, kind)
    operation.kind === kind || throw(ArgumentError("wide writer/operation kind disagree"))
    operation.certificate.status === :valid ||
        throw(ArgumentError("wide certificate refused: $(operation.certificate.status)"))
    sample = operation.sample
    if sample !== nothing
        sample.generation == operation.sample_generation ||
            throw(ArgumentError("the wide sampled-rate certificate is stale"))
    end
    return nothing
end

function _wide_check_output(output, operation, execution)
    oncpu(execution) || throw(ArgumentError("wide certification requires CPU execution"))
    output isa StridedMatrix{Complex{DF32}} && size(output) == operation.operator.shape ||
        throw(ArgumentError("wide output requires a matching strided Complex{DF32} matrix"))
    _transfer_writable_destination(output)
    _retained_alias(output, operation) &&
        throw(ArgumentError("wide output overlaps retained certificate or numerical data"))
    return nothing
end

function _wide_write!(output, operation, execution, kind)
    _wide_check_operation(operation, kind)
    _wide_check_output(output, operation, execution)
    prepared, bundle = operation.operator, operation.scalars
    coefficients, terms = prepared.coefficients, prepared.terms
    operator_kind = prepared.kind
    kind_code = _wide_kind(kind)
    # Only DF32 buffers and isbits scalars enter the numerical closure. The host
    # certificate arrays and binary64 coordinate records stay outside it.
    if kind !== :compensation
        each!(execution, length(output), 1) do row, unused
            rate = _wide_graph(
                coefficients, terms, nothing, nothing, operator_kind, row, bundle,
                Val(:compute),
            )
            output[row] = _wide_result(rate, bundle, kind_code, rate)
        end
    else
        sample = operation.sample
        sample === nothing &&
            throw(ArgumentError("wide compensation requires sampled-rate provenance"))
        samples = sample.values
        each!(execution, length(output), 1) do row, unused
            rate = _wide_graph(
                coefficients, terms, nothing, nothing, operator_kind, row, bundle,
                Val(:compute),
            )
            output[row] = _wide_result(rate, bundle, kind_code, samples[row])
        end
    end
    return output
end

"""
    wide_exponent!(output, operation, execution)

Write the certified dimensionless exponent into a separate Complex{DF32} matrix.

The operation must have kind `:exponent`, a valid certificate and unchanged retained
inputs. All admission, stale-provenance and overlap checks precede writes. The pure
DF32 graph has no allocation or host-coordinate arithmetic. Return `output`.
"""
wide_exponent!(output, operation::WideOperation, execution::KernelExecution) =
    _wide_write!(output, operation, execution, :exponent)

"""
    wide_transport!(output, operation, execution)

Write a certified history or dense transport exponent in dimensionless DF32 storage.

Require a valid `:transport` operation. Use the actual stored sample coordinate and
the operation's supplied step or dense destination, without snapping either coordinate.
Admission and alias rules are those of `wide_exponent!`. Return `output`.
"""
wide_transport!(output, operation::WideOperation, execution::KernelExecution) =
    _wide_write!(output, operation, execution, :transport)

"""
    wide_compensation!(output, operation, execution)

Write the actual sampled rate minus the certified divided difference, in inverse metres.

Require a valid `:compensation` operation and unchanged `WideRate` generation. Its
certificate propagates the sampled rate's explicit prior error and the divided
difference error; it does not apply the dimensionless function-argument tolerance.
Return `output`, under the ownership rules of `wide_exponent!`.
"""
wide_compensation!(output, operation::WideOperation, execution::KernelExecution) =
    _wide_write!(output, operation, execution, :compensation)

function _wide_rate_bundle(prepared, position, before)
    geometry = prepared.geometry
    geometry.history_lower <= position <= geometry.history_upper && isfinite(position) ||
        throw(ArgumentError("wide sampled rate lies outside retained-history geometry"))
    if !isempty(prepared.edges)
        geometry.lower <= position <= geometry.upper ||
            throw(ArgumentError("wide rate evaluation lies outside represented geometry"))
    end
    return _wide_scalars(prepared, position, position, 1.0, position, nothing, true, before)
end

"""
    prepare_wide_rate(execution, prepared, position; before=false)

Allocate an owned sampled-rate workspace and evaluate the bound representation there.

`position` is in metres. The rate and componentwise certificate radii are inverse metres;
`before` preserves the shared exact-knot side convention. Subsequent `wide_rate!` calls
reuse this workspace without allocation. Each successful refresh invalidates operations
bound to its previous generation; no bare value array implicitly supplies provenance.
"""
function prepare_wide_rate(
        execution::KernelExecution, prepared::WideOperator, position::Float64;
        before::Bool = false,
    )
    rates = fill(Complex{DF32}(zero(DF32), zero(DF32)), prepared.shape)
    errors = zeros(ComplexF64, prepared.shape)
    sample = WideRate(prepared, position, before, rates, errors, :represented, zero(UInt64))
    return wide_rate!(sample, position, execution; before)
end

"""
    wide_rate!(sample::WideRate, position, execution; before=false)

Refresh owned sampled-rate values and their bound error before advancing its generation.

Certify every cell before writing either owned buffer. The sampled derivative uses the
same cumulative representation, exact knot convention and host-coordinate error model
as the exponent graph. A refused refresh preserves all values, radii and metadata.
The CPU certification pass is allocation-free after compilation. Return `sample`.
"""
function wide_rate!(
        sample::WideRate, position::Float64, execution::KernelExecution;
        before::Bool = false,
    )
    oncpu(execution) || throw(ArgumentError("wide rate certification requires CPU"))
    sample.generation < typemax(UInt64) ||
        throw(ArgumentError("wide sampled-rate generation is exhausted"))
    prepared = sample.operator
    bundle = _wide_rate_bundle(prepared, position, before)
    certificate = _wide_certificate(prepared, bundle, :rate, nothing)
    certificate.status === :valid ||
        throw(ArgumentError("wide rate exceeds the protected intermediate range"))
    # This certificate pass is host-only and includes the actual rounded values.
    # Repeating it fills cellwise provenance without allocating an error workspace.
    for row in eachindex(sample.values)
        result = _wide_graph(
            prepared.coefficients, prepared.terms, prepared.coefficient_errors,
            prepared.term_errors, prepared.kind, row, bundle, Val(:checked),
        )
        sample.values[row] = Complex{DF32}(real(result).value, imag(result).value)
        sample.errors[row] = ComplexF64(real(result).error, imag(result).error)
    end
    sample.position, sample.before = position, before
    sample.provenance = :represented
    sample.generation += one(UInt64)
    return sample
end

"""
    restore_wide_rate(prepared, position, values; prior_errors, before=false)

Copy actual binary64 stored rates and their explicit prior absolute-error provenance.

Both matrices match the operator shape. `prior_errors` is complex binary64 storage whose
real and imaginary parts are nonnegative componentwise radii in inverse metres, bounding
the actual stored binary64 values relative to their intended rates. Include all earlier
evaluation, export and precision-conversion errors; an unadjusted pre-export certificate
is insufficient. The array is mandatory, copied with the actual values and augmented by
their DF32 conversion errors. This boundary binds
but cannot independently verify external provenance; compatibility and history identity
remain the caller's responsibility. A bare array never means a zero prior error.
"""
function restore_wide_rate(
        prepared::WideOperator, position::Float64, values::AbstractMatrix{ComplexF64};
        prior_errors::AbstractMatrix{ComplexF64}, before::Bool = false,
    )
    size(values) == size(prior_errors) == prepared.shape ||
        throw(DimensionMismatch("restored wide rate/errors must match the operator"))
    values isa StridedMatrix{ComplexF64} && prior_errors isa StridedMatrix{ComplexF64} ||
        throw(ArgumentError("restored wide rates need strided CPU binary64 inputs"))
    geometry = prepared.geometry
    isfinite(position) && geometry.history_lower <= position <= geometry.history_upper ||
        throw(ArgumentError("restored wide rate lies outside retained-history geometry"))
    all(x -> isfinite(x) && real(x) >= 0 && imag(x) >= 0, prior_errors) ||
        throw(ArgumentError("restored wide rates require finite nonnegative prior errors"))
    owned, errors = _wide_copy(values)
    for index in eachindex(errors)
        errors[index] = ComplexF64(
            _wide_sum(real(errors[index]), real(prior_errors[index])),
            _wide_sum(imag(errors[index]), imag(prior_errors[index])),
        )
    end
    all(isfinite, errors) || throw(ArgumentError("restored wide rate error overflowed"))
    return WideRate(prepared, position, before, owned, errors, :restored, one(UInt64))
end
