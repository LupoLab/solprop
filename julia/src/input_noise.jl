# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

public InputNoise, apply_input_noise!, input_noise_phase!, input_noise_memory

"""
    InputNoise(execution, spectrum, launch, operator)

Copy one supplied binary64 input-noise draw and allocate its numerical work.

`spectrum` is a finite complex binary64 matrix `(n_omega,n_columns)` in the existing
internal spectral amplitude units and FFT convention. `launch` is the problem's
finite nonnegative launch position in m. `operator` is the already prepared binary64
`DiagonalOperator`, `RepresentedOperator` or `ExpansionOperator` of the same shape;
its immutable data are borrowed. No random draw, normalization or model is introduced.

The record owns the copied `spectrum`, complex `scratch`, real `phase` and real
expansion `weights`. The latter is empty for a constant or direct table. Numerical
calls reuse these arrays synchronously; callers must not mutate the retained draw or
operator. The spectrum remains available for the temporal monitor floor on resume.
`input_noise_memory` accounts owned arrays separately from the shared operator.

Invalid precision, values or operator type raise `ArgumentError`; incompatible shapes
raise `DimensionMismatch` before allocation. The current boundary admits CPU storage;
kernel bodies retain the shared execution API without implying vendor qualification.
"""
struct InputNoise{A, O, P, W}
    spectrum::A
    launch::Float64
    operator::O
    scratch::A
    phase::P
    weights::W

    function InputNoise(
            spectrum::A, launch::Float64, operator::O, scratch::A, phase::P, weights::W,
        ) where {A, O, P, W}
        return new{A, O, P, W}(spectrum, launch, operator, scratch, phase, weights)
    end
end

function _input_noise_shape(operator::DiagonalOperator)
    eltype(operator.rate) === ComplexF64 ||
        throw(ArgumentError("input noise requires a binary64 linear operator"))
    return size(operator.rate), 0
end

function _input_noise_shape(operator::RepresentedOperator)
    table = operator.cumulative
    eltype(table.coefficients) in (Float64, ComplexF64) &&
        eltype(table.breakpoints) === Float64 ||
        throw(ArgumentError("input noise requires a binary64 cumulative operator table"))
    return (operator.n_frequency, operator.n_columns), 0
end

function _input_noise_shape(operator::ExpansionOperator)
    table = operator.cumulative_weights
    eltype(operator.terms) === ComplexF64 && eltype(table.coefficients) === Float64 &&
        eltype(table.breakpoints) === Float64 ||
        throw(ArgumentError("input noise requires a binary64 operator expansion"))
    return (size(operator.terms, 1), size(operator.terms, 2)), size(operator.terms, 3)
end

function _input_noise_shape(operator)
    throw(ArgumentError("input noise requires a supported prepared linear operator"))
end

function InputNoise(execution::KernelExecution, spectrum, launch::Real, operator)
    oncpu(execution) || throw(ArgumentError("binary64 input noise currently requires CPU"))
    spectrum isa AbstractMatrix{ComplexF64} && !isempty(spectrum) ||
        throw(ArgumentError("input noise requires a nonempty complex binary64 matrix"))
    Base.require_one_based_indexing(spectrum)
    all(isfinite, spectrum) || throw(ArgumentError("input noise spectrum must be finite"))
    position = Float64(launch)
    isfinite(position) && position >= 0 ||
        throw(ArgumentError("input noise launch must be finite and nonnegative"))
    shape, count = _input_noise_shape(operator)
    shape == size(spectrum) ||
        throw(DimensionMismatch("input noise and linear operator shapes disagree"))
    return InputNoise(
        upload(execution, spectrum), position, operator,
        dzeros(execution, ComplexF64, shape...), dzeros(execution, Float64, shape...),
        dzeros(execution, Float64, count),
    )
end

"""
    apply_input_noise!(field, work::InputNoise, active, z, resume, execution) -> Bool

Add the supplied draw at a fresh launch and apply the exact active-band mask.

`field` is a writable complex binary64 matrix matching `work.spectrum`; `active` is
a Boolean vector with one entry per frequency. The absolute position `z` is finite
and nonnegative, in m; `resume` is Boolean. Add the draw exactly when `!resume` and
`z == work.launch`. Fresh interior launches therefore add once, while resumes at the
original launch or an interior point do not. Return whether the draw was added.

All input values must be finite. Supported affine writable layouts and separation
from retained noise/operator data and mask storage are checked before mutation. Build
the fully masked candidate in owned scratch, check it for overflow, then commit to the
field. An invalid argument or nonfinite candidate raises `ArgumentError` without
changing the field; incompatible shapes raise `DimensionMismatch`. Inactive entries
are exact zeros. A failed candidate may change scratch but not the retained draw.
Warmed CPU calls allocate nothing and charge no propagation source evaluations.
"""
function apply_input_noise!(
        field, work::InputNoise, active, z::Real, resume::Bool,
        execution::KernelExecution,
    )
    oncpu(execution) || throw(ArgumentError("binary64 input noise currently requires CPU"))
    field isa AbstractMatrix{ComplexF64} && active isa AbstractVector{Bool} ||
        throw(ArgumentError("input noise needs a binary64 field and Boolean active mask"))
    size(field) == size(work.spectrum) && length(active) == size(field, 1) ||
        throw(DimensionMismatch("input noise field, spectrum and active mask disagree"))
    Base.require_one_based_indexing(field, active)
    isfinite(z) && z >= 0 ||
        throw(ArgumentError("input noise position must be finite and nonnegative"))
    _transfer_writable_destination(field)
    (
        _retained_alias(field, work) || _retained_alias(active, work) ||
            _transfer_arrays_alias(field, active) ||
            _transfer_arrays_alias(work.scratch, active)
    ) && throw(ArgumentError("input noise field and mask must not alias retained storage"))
    all(isfinite, field) && all(isfinite, work.spectrum) ||
        throw(ArgumentError("input noise field and spectrum must be finite"))
    added = !resume && z == work.launch
    spectrum, scratch = work.spectrum, work.scratch
    each!(execution, size(field)...) do row, column
        scratch[row, column] = !active[row] ? 0.0 + 0.0im :
            added ? field[row, column] + spectrum[row, column] : field[row, column]
    end
    all(isfinite, scratch) ||
        throw(ArgumentError("input noise produced a nonfinite launch candidate"))
    copyto!(field, scratch)
    return added
end

function _input_noise_phase!(phase, scratch, weights, operator::DiagonalOperator, z, ex)
    rate = operator.rate
    each!(ex, size(phase)...) do row, column
        phase[row, column] = imag(rate[row, column]) * z
    end
    return nothing
end

function _input_noise_phase!(phase, scratch, weights, operator::RepresentedOperator, z, ex)
    evaluate_table!(scratch, operator.cumulative, z, ex)
    each!(ex, size(phase)...) do row, column
        phase[row, column] = imag(scratch[row, column])
    end
    return nothing
end

function _input_noise_phase!(phase, scratch, weights, operator::ExpansionOperator, z, ex)
    evaluate_table!(weights, operator.cumulative_weights, z, ex)
    terms = operator.terms
    each!(ex, size(phase)...) do row, column
        total = 0.0 + 0.0im
        for term in axes(terms, 3)
            total += terms[row, column, term] * weights[term]
        end
        phase[row, column] = imag(total)
    end
    return nothing
end

"""
    input_noise_phase!(work::InputNoise, z, execution)

Evaluate the represented unitary phase at the absolute position in metres.

Return the owned real binary64 `work.phase` matrix `(n_omega,n_columns)`, in radians,
for immediate copying by the session boundary. It is overwritten by the next call.
The constant representation gives `imag(D)*z`; direct and expanded representations
give `imag(Phi(z))` from their cumulative tables, including represented offsets.
There is no subtraction at the launch or resume position. Attenuation is not a phase.
This is the shared `translate.linop.unitary_phase` convention, without a new model.

The finite nonnegative position must lie inside each supplied table's admitted span.
Invalid positions or nonfinite evaluated phases raise `ArgumentError`. The shared
operator is immutable borrowed input; no returned array owns or exposes its storage.
Warmed CPU calls allocate nothing. Constant operator span validation remains with the
session's translated position policy because a `DiagonalOperator` stores no range.
"""
function input_noise_phase!(work::InputNoise, z::Real, execution::KernelExecution)
    oncpu(execution) || throw(ArgumentError("binary64 input noise currently requires CPU"))
    isfinite(z) && z >= 0 ||
        throw(ArgumentError("input noise phase position must be finite and nonnegative"))
    _input_noise_phase!(
        work.phase, work.scratch, work.weights, work.operator, z, execution,
    )
    all(isfinite, work.phase) ||
        throw(ArgumentError("input noise unitary phase must be finite"))
    return work.phase
end

"""
    input_noise_memory(work::InputNoise)

Return a named tuple of owned noise-array storage sizes in bytes.

Entries `spectrum`, `scratch`, `phase` and `weights` count each distinct owned array
once, including zero bytes for unused expansion work. The borrowed immutable linear
operator is excluded; its preparation owner already accounts that storage. No session
state, random generator or hidden retained buffer belongs to this leaf.
"""
function input_noise_memory(work::InputNoise)
    return (
        spectrum = sizeof(work.spectrum), scratch = sizeof(work.scratch),
        phase = sizeof(work.phase), weights = sizeof(work.weights),
    )
end
