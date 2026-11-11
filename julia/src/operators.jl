# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public DiagonalOperator, operator_exponent!, propagate!, propagator_pair!, linear_increment!
public RepresentedOperator, ExpansionOperator, operator_rate!

"""
    DiagonalOperator(execution, rate)

Copy a constant diagonal linear generator in inverse metres into owned storage.

`rate` is a finite complex matrix with shape `(n_omega, n_components*n_modes)` in the
internal Fourier convention. It already contains the translated dispersion, frame and
attenuation; this record neither adds physical terms nor changes their signs.
The exponent over `[z0,z1]` is `(z1-z0)*rate`.

# Throws
- `ArgumentError`: For empty, nonfinite or unsupported numerical data.
"""
struct DiagonalOperator{A}
    rate::A
end

function DiagonalOperator(
        execution::KernelExecution, rate::AbstractMatrix{Complex{T}},
    ) where {T <: Union{Float32, Float64}}
    isempty(rate) && throw(ArgumentError("a diagonal generator must not be empty"))
    all(isfinite, rate) || throw(ArgumentError("a diagonal generator must be finite"))
    return DiagonalOperator(upload(execution, rate))
end

function DiagonalOperator(execution::KernelExecution, rate)
    throw(ArgumentError("a diagonal generator requires complex binary32 or binary64 data"))
end

"""
    operator_exponent!(output, operator::DiagonalOperator, z0, z1, execution;
                       work = nothing)

Write the dimensionless integrated diagonal generator between two positions in metres.

`output` matches the operator's `(n_omega,n_columns)` shape. A zero interval writes exact
zeros. A constant or represented table needs no `work`. An `ExpansionOperator` requires
`work` to be a caller-owned real vector with one entry per fixed term; it stores the
scalar weight differences. Positions must be finite and lie in a represented operator's
admitted span; reversed intervals are allowed. Constant operators convert both positions
to binary64 before subtracting, avoiding bounded-integer arithmetic. Return `output`.
"""
function operator_exponent!(
        output, operator::DiagonalOperator, z0::Real, z1::Real,
        execution::KernelExecution; work = nothing,
    )
    size(output) == size(operator.rate) ||
        throw(DimensionMismatch("the exponent buffer must match the diagonal generator"))
    Base.mightalias(output, operator.rate) &&
        throw(ArgumentError("the exponent output must not overwrite its generator"))
    isfinite(z0) && isfinite(z1) ||
        throw(ArgumentError("operator positions must be finite"))
    # Positions are host coordinates: convert before subtracting so finite Integer
    # inputs cannot wrap in their source type (issue #684).
    distance = Float64(z1) - Float64(z0)
    isfinite(distance) || throw(ArgumentError("the operator interval must be finite"))
    rate = operator.rate
    neutral = zero(eltype(output))
    each!(execution, size(output)...) do row, column
        output[row, column] = iszero(distance) ? neutral : rate[row, column] * distance
    end
    return output
end

"""
    RepresentedOperator(execution, cumulative, n_frequency, n_columns)

Bind one owned cumulative diagonal-operator table to its spectral matrix shape.

`cumulative` is a `ChebyshevTable` of dimensionless Phi, with values ordered as the
linear storage of `(n_frequency,n_columns)`. It is already owned; no table is shared
implicitly with a physical response profile. Exponents use represented endpoint
subtraction, with the cancellation limitation documented by `difference_table!`.
"""
struct RepresentedOperator{T}
    cumulative::T
    n_frequency::Int
    n_columns::Int
end

function RepresentedOperator(
        execution::KernelExecution, cumulative::ChebyshevTable,
        n_frequency::Integer, n_columns::Integer,
    )
    n_frequency > 0 && n_columns > 0 ||
        throw(ArgumentError("represented operator dimensions must be positive"))
    Base.checked_mul(n_frequency, n_columns) == size(cumulative.coefficients, 1) ||
        throw(DimensionMismatch("represented Phi does not match the spectral shape"))
    return RepresentedOperator(cumulative, Int(n_frequency), Int(n_columns))
end

"""
    ExpansionOperator(execution, terms, cumulative_weights)

Copy a diagonal expansion's fixed terms and bind its owned cumulative weight table.

Complex `terms` have shape `(n_frequency,n_columns,n_terms)`. The real
`cumulative_weights::ChebyshevTable` has `n_terms` values; their weighted sum is the
dimensionless Phi. Each exponent subtracts scalar endpoint weights before combining
the fixed terms, matching the translated expansion contract.
"""
struct ExpansionOperator{A, T}
    terms::A
    cumulative_weights::T
end

function ExpansionOperator(
        execution::KernelExecution, terms::AbstractArray{Complex{T}, 3},
        cumulative_weights::ChebyshevTable,
    ) where {T <: Union{Float32, Float64}}
    all(>(0), size(terms)) && all(isfinite, terms) ||
        throw(ArgumentError("operator expansion terms must be nonempty and finite"))
    eltype(cumulative_weights.coefficients) <: Real ||
        throw(ArgumentError("operator expansion weights must be real"))
    size(terms, 3) == size(cumulative_weights.coefficients, 1) ||
        throw(DimensionMismatch("operator terms and cumulative weights disagree"))
    return ExpansionOperator(upload(execution, terms), cumulative_weights)
end

function operator_exponent!(
        output, operator::RepresentedOperator, z0::Real, z1::Real,
        execution::KernelExecution; work = nothing,
    )
    size(output) == (operator.n_frequency, operator.n_columns) ||
        throw(DimensionMismatch("the exponent buffer does not match represented Phi"))
    return difference_table!(output, operator.cumulative, z0, z1, execution)
end

function operator_exponent!(
        output, operator::ExpansionOperator, z0::Real, z1::Real,
        execution::KernelExecution; work,
    )
    terms = operator.terms
    size(output) == (size(terms, 1), size(terms, 2)) ||
        throw(DimensionMismatch("the exponent buffer does not match expansion terms"))
    work isa AbstractVector && length(work) == size(terms, 3) ||
        throw(DimensionMismatch("an expansion needs one weight difference per fixed term"))
    if Base.mightalias(output, terms) || Base.mightalias(output, work) ||
            Base.mightalias(work, terms)
        throw(ArgumentError("expansion terms, exponent and weight work must be disjoint"))
    end
    difference_table!(work, operator.cumulative_weights, z0, z1, execution)
    neutral = zero(eltype(output))
    n_terms = size(terms, 3)
    each!(execution, size(output)...) do row, column
        total = neutral
        for term in 1:n_terms
            total += terms[row, column, term] * work[term]
        end
        output[row, column] = total
    end
    return output
end

"""
    operator_rate!(output, operator, position, execution; before=false, work=nothing)

Write the derivative of the represented cumulative operator in inverse metres.

`output` has the operator's spectral matrix shape. At interior knots, `before=true`
uses the interval ending at `position`; the default uses the interval starting there.
Constant generators simply copy their rates. Expansion operators require an explicit
real `work` vector with one entry per fixed term. No independently sampled physical
generator substitutes for the derivative of cumulative Phi. Return `output`.
"""
function operator_rate!(
        output, operator::DiagonalOperator, position::Real, execution::KernelExecution;
        before::Bool = false, work = nothing,
    )
    size(output) == size(operator.rate) ||
        throw(DimensionMismatch("the rate buffer must match the diagonal generator"))
    Base.mightalias(output, operator.rate) &&
        throw(ArgumentError("the rate output must not overwrite its generator"))
    isfinite(position) || throw(ArgumentError("operator positions must be finite"))
    rate = operator.rate
    each!(execution, size(output)...) do row, column
        output[row, column] = rate[row, column]
    end
    return output
end

function operator_rate!(
        output, operator::RepresentedOperator, position::Real, execution::KernelExecution;
        before::Bool = false, work = nothing,
    )
    size(output) == (operator.n_frequency, operator.n_columns) ||
        throw(DimensionMismatch("the rate buffer does not match represented Phi"))
    return derivative_table!(output, operator.cumulative, position, execution; before)
end

function operator_rate!(
        output, operator::ExpansionOperator, position::Real, execution::KernelExecution;
        before::Bool = false, work,
    )
    terms = operator.terms
    size(output) == (size(terms, 1), size(terms, 2)) ||
        throw(DimensionMismatch("the rate buffer does not match expansion terms"))
    work isa AbstractVector && length(work) == size(terms, 3) ||
        throw(DimensionMismatch("an expansion needs one weight derivative per fixed term"))
    if Base.mightalias(output, terms) || Base.mightalias(output, work) ||
            Base.mightalias(work, terms)
        throw(ArgumentError("expansion terms, rate and weight work must be disjoint"))
    end
    derivative_table!(work, operator.cumulative_weights, position, execution; before)
    neutral = zero(eltype(output))
    n_terms = size(terms, 3)
    each!(execution, size(output)...) do row, column
        total = neutral
        for term in 1:n_terms
            total += terms[row, column, term] * work[term]
        end
        output[row, column] = total
    end
    return output
end

@inline function _complex_expm1(exponent::Complex)
    real_part, angle = reim(exponent)
    iszero(exponent) && return zero(exponent)
    sine, cosine = sincos(angle)
    half_sine = sin(angle / 2)
    # exp(x+iy)-1 = expm1(x)cos(y) - 2sin(y/2)^2 + i exp(x)sin(y).
    # Keeping both small differences explicit avoids cancellation against one.
    real_increment = expm1(real_part) * cosine - 2 * half_sine * half_sine
    imaginary_increment = exp(real_part) * sine
    return complex(real_increment, imaginary_increment)
end

@inline function _operator_shapes(output, state, exponent)
    size(output) == size(state) == size(exponent) ||
        throw(DimensionMismatch("state, exponent and output shapes must agree"))
    return nothing
end

@inline function _operator_aliases(output, state, exponent)
    _transfer_writable_destination(output)
    _transfer_arrays_alias(output, exponent) &&
        throw(ArgumentError("the output must not overwrite its exponent"))
    _transfer_arrays_alias(output, state) && output !== state &&
        throw(ArgumentError("partially overlapping state buffers are not supported"))
    return nothing
end

"""
    propagate!(output, state, exponent, execution)

Apply a prepared dimensionless diagonal exponential to a spectral state.

All matrices have the same shape. In-place state use is supported, and a zero exponent
copies that state element exactly. No Fourier or physical normalization is introduced.
Return `output`; the exponent is preserved.
"""
function propagate!(output, state, exponent, execution::KernelExecution)
    _operator_shapes(output, state, exponent)
    _operator_aliases(output, state, exponent)
    each!(execution, size(output)...) do row, column
        phase = exponent[row, column]
        output[row, column] = iszero(phase) ? state[row, column] :
            exp(phase) * state[row, column]
    end
    return output
end

"""
    propagator_pair!(forward, inverse, exponent, execution)

Write `exp(exponent)` and `exp(-exponent)` into equally shaped caller-owned matrices.

The output arrays are disjoint from each other and the input. Zero exponents give exact
ones in both arrays. Return `(forward, inverse)` after synchronous completion.
"""
function propagator_pair!(forward, inverse, exponent, execution::KernelExecution)
    _operator_shapes(forward, inverse, exponent)
    _transfer_admit_destinations((forward, inverse))
    if _transfer_arrays_alias(forward, inverse) ||
            _transfer_arrays_alias(forward, exponent) ||
            _transfer_arrays_alias(inverse, exponent)
        throw(ArgumentError("propagator-pair buffers must be disjoint"))
    end
    unit = one(eltype(forward))
    each!(execution, size(exponent)...) do row, column
        phase = exponent[row, column]
        forward[row, column] = iszero(phase) ? unit : exp(phase)
        inverse[row, column] = iszero(phase) ? unit : exp(-phase)
    end
    return forward, inverse
end

"""
    linear_increment!(output, state, exponent, execution)

Write the stable diagonal increment `(exp(exponent)-1)*state` without subtracting states.

All matrices have equal shape and state/output may alias. Real attenuation uses `expm1`;
the real phase increment uses `-2sin(angle/2)^2`, retaining small-angle information lost
by `cos(angle)-1`. A zero exponent writes an exact zero. This protected helper must not
be reassociated or placed under blanket fast-math. Return `output`.
"""
function linear_increment!(output, state, exponent, execution::KernelExecution)
    _operator_shapes(output, state, exponent)
    _operator_aliases(output, state, exponent)
    neutral = zero(eltype(output))
    each!(execution, size(output)...) do row, column
        phase = exponent[row, column]
        output[row, column] = iszero(phase) ? neutral :
            _complex_expm1(phase) * state[row, column]
    end
    return output
end
