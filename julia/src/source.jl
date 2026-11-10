# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

public NonfiniteSourceError, EnvelopeKerrData, EnvelopeKerrSource, LinearFlow

"""
    NonfiniteSourceError(stage, row, column)

Report a nonfinite numerical source value at one owned matrix location.

`stage` identifies the operation; `row` and `column` are one-based indices. Source
failures occur outside leaf kernels and leave the integrator's accepted state intact.
"""
struct NonfiniteSourceError <: Exception
    stage::Symbol
    row::Int
    column::Int
end

function Base.showerror(io::IO, error::NonfiniteSourceError)
    return print(
        io, "nonfinite ", error.stage, " at row ", error.row, ", column ", error.column
    )
end

function _source_finite!(failures, array, execution, stage)
    rows, columns = size(array)
    length(failures) >= columns ||
        throw(DimensionMismatch("source check buffer is too small"))
    each_column!(execution, columns) do column
        failed_row = 0
        for row in 1:rows
            if !isfinite(array[row, column])
                failed_row = row
                break
            end
        end
        failures[column] = failed_row
    end
    for column in 1:columns
        if failures[column] != 0
            throw(NonfiniteSourceError(stage, failures[column], column))
        end
    end
    return nothing
end

"""
    EnvelopeKerrData(execution; synthesis, projection, prefactor, omega_window,
                     source_window, kerr, density, carrier=nothing)

Own the translated constant modal data for a scalar binary64 Kerr envelope source.

`synthesis[node,mode]` includes physical amplitude and spectral padding scales.
`projection[mode,node]` includes quadrature; `prefactor[omega,mode]` includes the crop
scale. `omega_window` and `source_window` are dimensionless coarse spectral and fine
temporal windows. No extra pad/crop factor is applied in the source pipeline.
`kerr` is `KerrCoefficients` or `nothing`, `density` its nonnegative constant stage
weight. An optional complex `carrier` has one entry per fine time sample. No physics
object or Python owner is retained. This first composition admits only CPU execution;
varying profiles and other responses use their own prepared source compositions.

# Throws
- `ArgumentError`: For a non-CPU policy, nonfinite data or incompatible cubic form.
- `DimensionMismatch`: For mismatched modal, node or time extents.
"""
struct EnvelopeKerrData{S, P, F, W, T, K, C}
    synthesis::S
    projection::P
    prefactor::F
    omega_window::W
    source_window::T
    kerr::K
    density::Float64
    carrier::C
end

function EnvelopeKerrData(
        execution::KernelExecution;
        synthesis::AbstractMatrix{Float64}, projection::AbstractMatrix{Float64},
        prefactor::AbstractMatrix{ComplexF64}, omega_window::AbstractVector{Float64},
        source_window::AbstractVector{Float64}, kerr, density::Float64,
        carrier = nothing,
    )
    oncpu(execution) || throw(ArgumentError("this modal source currently requires CPU"))
    nodes, modes = size(synthesis)
    nodes > 0 && modes > 0 && !isempty(source_window) ||
        throw(ArgumentError("a modal source needs nonempty mode, node and time axes"))
    size(projection) == (modes, nodes) && size(prefactor, 2) == modes &&
        size(prefactor, 1) == length(omega_window) ||
        throw(DimensionMismatch("constant modal source arrays disagree"))
    all(isfinite, synthesis) && all(isfinite, projection) && all(isfinite, prefactor) &&
        all(isfinite, omega_window) && all(isfinite, source_window) &&
        isfinite(density) && density >= 0.0 ||
        throw(ArgumentError("constant modal source data must be finite"))
    if kerr !== nothing &&
            (!(kerr isa KerrCoefficients{Float64, :analytic}) || kerr.take_real)
        throw(ArgumentError("a scalar envelope requires binary64 analytic Kerr data"))
    end
    if carrier !== nothing
        carrier isa AbstractVector{ComplexF64} &&
            length(carrier) == length(source_window) && all(isfinite, carrier) ||
            throw(DimensionMismatch("Kerr carrier must match the fine time axis"))
    elseif kerr !== nothing && !iszero(kerr.harmonic_factor)
        throw(ArgumentError("envelope third-harmonic Kerr requires a translated carrier"))
    end
    return EnvelopeKerrData(
        upload(execution, synthesis), upload(execution, projection),
        upload(execution, prefactor), upload(execution, omega_window),
        upload(execution, source_window), kerr, density,
        carrier === nothing ? nothing : upload(execution, carrier),
    )
end

"""
    EnvelopeKerrSource(execution, data, crop_index)

Prepare owned FFT, nodal and polarisation work for a constant scalar envelope source.

`data` is independently owned `EnvelopeKerrData`; the zero-based `crop_index` is the
authoritative translated coarse-to-fine frequency map. State/source arrays use
`(n_omega,n_modes)` storage. The callable `source(out,state,z)` returns `nothing`,
writes the complete derivative in state units per metre, and allocates nothing after
warmup. Each invocation increments `calls` before work starts, including failed calls.
Every scratch buffer belongs to this source instance; immutable data can be reused.
"""
mutable struct EnvelopeKerrSource{E, D, P, F, A}
    execution::E
    data::D
    padding::P
    transform::F
    fine_modal::A
    fine_nodal::A
    samples::A
    polarisation::A
    failures::Vector{Int}
    calls::Int
end

function EnvelopeKerrSource(execution::KernelExecution, data::EnvelopeKerrData, crop_index)
    oncpu(execution) || throw(ArgumentError("this modal source currently requires CPU"))
    length(crop_index) == size(data.prefactor, 1) ||
        throw(DimensionMismatch("the crop map does not match the coarse spectrum"))
    n_time = length(data.source_window)
    nodes, modes = size(data.synthesis)
    padding = SpectralPadding(execution, crop_index, n_time, 1.0, 1.0; index_base = 0)
    fine_modal = dzeros(execution, ComplexF64, n_time, modes)
    fine_nodal = dzeros(execution, ComplexF64, n_time, nodes)
    samples = similar(fine_nodal)
    polarisation = similar(fine_nodal)
    transform = TemporalFFT(execution, samples, fine_nodal)
    return EnvelopeKerrSource(
        execution, data, padding, transform, fine_modal, fine_nodal, samples,
        polarisation, zeros(Int, max(nodes, modes)), 0,
    )
end

function (source::EnvelopeKerrSource)(output, state, z::Float64)
    source.calls += 1
    data, execution = source.data, source.execution
    size(state) == size(data.prefactor) && size(output) == size(state) ||
        throw(DimensionMismatch("source input/output does not match its preparation"))
    isfinite(z) || throw(ArgumentError("a source position must be finite"))
    Base.mightalias(output, state) &&
        throw(ArgumentError("the nonlinear source must not overwrite its input state"))
    _source_finite!(source.failures, state, execution, :input)
    pad_spectrum!(source.fine_modal, state, source.padding, execution)
    synthesize!(
        source.fine_nodal, source.fine_modal, data.synthesis, execution; n_components = 1
    )
    temporal_inverse!(source.samples, source.transform, source.fine_nodal, execution)
    if data.kerr === nothing
        fill!(source.polarisation, 0)
    else
        layout = ResponseLayout(size(source.samples, 1), 1, size(source.samples, 2))
        kerr_polarisation!(
            source.polarisation, source.samples, layout, data.kerr, execution;
            density = data.density, carrier = data.carrier, window = data.source_window,
        )
    end
    _source_finite!(source.failures, source.polarisation, execution, :polarisation)
    temporal_forward!(source.fine_nodal, source.transform, source.polarisation)
    project!(
        source.fine_modal, source.fine_nodal, data.projection, execution; n_components = 1
    )
    crop_spectrum!(output, source.fine_modal, source.padding, execution)
    apply_weights!(output, output, data.prefactor, execution)
    apply_weights!(output, output, data.omega_window, execution)
    _source_finite!(source.failures, output, execution, :source)
    return nothing
end

"""
    LinearFlow(execution, operator, state; work=nothing)

Prepare a callable linear flow with its own exponent and finiteness scratch.

`operator` is a prepared diagonal generator or cumulative representation; `state` sets
the binary64 `(n_omega,n_columns)` shape. Expansion operators need their caller-owned
weight work. Calling `flow(out,input,z0,z1)` applies the represented flow and returns
`nothing`, checking numerical failures outside leaf kernels. Input and output must be
disjoint. The CPU implementation allocates nothing after warmup.
"""
struct LinearFlow{E, O, A, W}
    execution::E
    operator::O
    exponent::A
    work::W
    failures::Vector{Int}
end

function LinearFlow(
        execution::KernelExecution, operator, state::AbstractMatrix{ComplexF64};
        work = nothing,
    )
    oncpu(execution) || throw(ArgumentError("this linear flow currently requires CPU"))
    return LinearFlow(
        execution, operator, similar(state), work, zeros(Int, size(state, 2))
    )
end

function (flow::LinearFlow)(output, input, origin::Float64, destination::Float64)
    Base.mightalias(output, input) &&
        throw(ArgumentError("the linear flow requires disjoint input and output"))
    size(output) == size(input) == size(flow.exponent) ||
        throw(DimensionMismatch("linear flow buffers do not match their preparation"))
    operator_exponent!(
        flow.exponent, flow.operator, origin, destination, flow.execution; work = flow.work
    )
    _source_finite!(flow.failures, flow.exponent, flow.execution, :exponent)
    propagate!(output, input, flow.exponent, flow.execution)
    _source_finite!(flow.failures, output, flow.execution, :flow)
    return nothing
end
